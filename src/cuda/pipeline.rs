//! The CUDA backend's model: every stage's layers on one GPU, and the [`Backend`] steps the
//! public API runs them through

use eyre::{Result, ensure, eyre};

use crate::backend::{Backend, DetectionScores, ObjectPass};
use crate::cuda::detector::{Detector, DetectorImage, DetectorOutput};
use crate::cuda::gpu::{Arg, Gpu, Tensor, launch};
use crate::cuda::image::preprocess;
use crate::cuda::ops::{B256, ew, upload_i32};
use crate::cuda::point::{Decoded, MaskResize, PointDecoder, PointImage, clamped_masks, fill_holes, pick_masks, row};
use crate::cuda::text::{TextEncoder, TextFeatures};
use crate::cuda::video::{FrameAttention, FrameFeatures, FrameMemory, Tracker, mask_for_mem, upsample};
use crate::cuda::vision::{Backbone, Neck, Trunk};
use crate::host::argmax;
use crate::image::RgbImage;
use crate::masks::MaskSink;
use crate::model::point::{LOW_RES, MASK_TOKEN_ROW};
use crate::model::tracker::NO_OBJ_SCORE;
use crate::model::vision::IMAGE_SIZE;
use crate::nvtx::range;
use crate::output::HOLE_AREA;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::tokenizer::CONTEXT;
use crate::tracker::Selection;
use crate::weights::Weights;

/// The loaded model on one CUDA device
pub(crate) struct Cuda {
    pub(crate) backbone: Backbone,
    pub(crate) neck: Neck,
    pub(crate) tracker_neck: Neck,
    text: TextEncoder,
    pub(crate) detector: Detector,
    pub(crate) point: PointDecoder,
    /// Memory attention, memory encoder and pointer heads of the video tracker
    pub(crate) tracker: Tracker,
    // last, so the layers free their weights before the GPU handles go
    pub(crate) gpu: Gpu,
}

/// Features of one encoded image
pub(crate) struct Image {
    detector: DetectorImage,
    point: PointImage,
}

/// Both point decoder passes over a batch: the best second-pass mask of each prompt
/// `[K, 288^2]`, before hole filling, and its predicted IoU
pub(crate) struct PointPass {
    best: Tensor,
    scores: Vec<f32>,
}

impl Cuda {
    /// Both decoder passes over prompts with equal click counts
    pub(crate) fn decode_point_batch(
        &self,
        img: &PointImage,
        prompts: &[&[PointPrompt]],
        size: (usize, usize),
    ) -> Result<PointPass> {
        let gpu = &self.gpu;
        // mask tokens 1 to 3, the multimask candidates
        let best = |iou: &[f32]| -> Vec<usize> { iou.chunks(4).map(|c| argmax(&c[1..])).collect() };
        let first = self.point.decode_batch(gpu, &img.embed, img, prompts, size, None, 1)?;
        let mask_input = clamped_masks(gpu, &first.low_res, &best(&gpu.download(&first.iou)?))?;
        let second = self.point.decode_batch(gpu, &img.embed, img, prompts, size, Some(&mask_input), 1)?;
        let iou = gpu.download(&second.iou)?;
        let picks = best(&iou);
        let scores = picks.iter().enumerate().map(|(b, &i)| iou[b * 4 + 1 + i]).collect();
        Ok(PointPass { best: pick_masks(gpu, &second.low_res, &picks)?, scores })
    }

    /// The hole-filled best masks of a pass at full size into `sink`, group by group, and
    /// their predicted IoUs
    pub(crate) fn point_batch_masks(
        &self,
        pass: PointPass,
        (w, h): (usize, usize),
        sink: &mut MaskSink,
    ) -> Result<Vec<f32>> {
        let gpu = &self.gpu;
        let PointPass { best, scores } = pass;
        let plane = w * h;
        fill_holes(gpu, &best, HOLE_AREA)?;
        let resize = MaskResize::new(gpu, scores.len(), w, h)?;
        let group = gpu.alloc_bytes(resize.group() * plane)?;
        for part in resize.groups() {
            resize.run(gpu, &best, part.clone(), Arg::slice(group.ptr(), part.len() * plane))?;
            // the buffer comes out of the pool after the first launches, so one still
            // faulting in overlaps the GPU work; consecutive prompts download together
            sink.write(part, |offset, dst| gpu.download_range_into(&group, offset, dst))?;
        }
        Ok(scores)
    }
}

impl Backend for Cuda {
    type Device = Gpu;
    type Image = Image;
    type Text = TextFeatures;
    type Detected = DetectorOutput;
    type PointPass = PointPass;
    type Pixels = Tensor;
    type Frame = FrameFeatures;
    type Logits = Tensor;
    type Decoded = Decoded;
    type Embed = Tensor;
    type Pointer = Tensor;
    type Memory = Tensor;
    type FrameMemory = FrameMemory;
    type Attention = FrameAttention;

    fn open(device: usize, precision: Precision) -> Result<Gpu> {
        Gpu::new(device, precision)
    }

    fn load(gpu: Gpu, wts: &Weights) -> Result<Self> {
        let model = Self {
            backbone: Backbone::load(&gpu, wts)?,
            neck: Neck::load(&gpu, wts, "detector_model.vision_encoder.neck")?,
            tracker_neck: Neck::load(&gpu, wts, "tracker_neck")?,
            text: TextEncoder::load(&gpu, wts)?,
            detector: Detector::load(&gpu, wts)?,
            point: PointDecoder::load(&gpu, wts)?,
            tracker: Tracker::load(&gpu, wts)?,
            gpu,
        };
        model.gpu.synchronize()?;
        Ok(model)
    }

    fn precision(&self) -> Precision {
        self.gpu.precision()
    }

    fn bind(&self) -> Result<()> {
        self.gpu.bind()
    }

    fn encode(&self, img: &RgbImage) -> Result<Image> {
        let gpu = &self.gpu;
        let _r = range(c"encode");
        let pixels = {
            let _r = range(c"preprocess");
            preprocess(gpu, img)?
        };
        let trunk = {
            let _r = range(c"backbone");
            self.backbone.forward(gpu, &pixels)?
        };
        drop(pixels);
        let _r = range(c"necks");
        let trunk = Trunk::new(gpu, trunk)?;
        let point = self.point.prepare_image(gpu, &self.tracker_neck.forward(gpu, &trunk)?)?;
        let detector = self.detector.prepare_image(gpu, self.neck.forward(gpu, &trunk)?)?;
        gpu.synchronize()?;
        Ok(Image { detector, point })
    }

    fn encode_text(&self, tokens: &[u32; CONTEXT]) -> Result<TextFeatures> {
        self.text.forward(&self.gpu, tokens)
    }

    fn detect(&self, img: &Image, text: &TextFeatures, boxes: &[BoxPrompt]) -> Result<DetectorOutput> {
        self.detector.forward(&self.gpu, &img.detector, text, boxes)
    }

    fn detection_scores(&self, out: &DetectorOutput) -> Result<DetectionScores> {
        let gpu = &self.gpu;
        let presence = gpu.download(&out.presence)?[0];
        Ok(DetectionScores { presence, logits: gpu.download(&out.logits)?, boxes: gpu.download(&out.boxes)? })
    }

    fn detection_masks(
        &self,
        out: &DetectorOutput,
        kept: &[usize],
        (w, h): (usize, usize),
        sink: &mut MaskSink,
    ) -> Result<()> {
        let gpu = &self.gpu;
        let side = LOW_RES * LOW_RES;
        let idx = upload_i32(gpu, kept.iter().map(|&i| i as i32).collect())?;
        let picked = gpu.empty(&[kept.len(), side])?;
        launch!(gpu, "gather_rows", ew(picked.len()), B256, 0; &picked, &out.masks, idx.slice::<i32>(), side)?;
        let mask = MaskResize::new(gpu, kept.len(), w, h)?.all(gpu, &picked)?;
        // the buffer comes out of the pool after the launches, so one still faulting in
        // overlaps the GPU work
        sink.write(0..kept.len(), |offset, dst| gpu.download_range_into(&mask, offset, dst))
    }

    fn decode_points(&self, img: &Image, prompts: &[&[PointPrompt]], size: (usize, usize)) -> Result<PointPass> {
        self.decode_point_batch(&img.point, prompts, size)
    }

    fn point_masks(&self, pass: PointPass, size: (usize, usize), sink: &mut MaskSink) -> Result<Vec<f32>> {
        self.point_batch_masks(pass, size, sink)
    }

    fn preprocess(&self, frame: &RgbImage) -> Result<Tensor> {
        preprocess(&self.gpu, frame)
    }

    fn video_frame(&self, pixels: &Tensor) -> Result<FrameFeatures> {
        self.video_features(pixels)
    }

    fn frame_logits(&self, objects: usize) -> Result<Tensor> {
        self.gpu.empty(&[objects, LOW_RES * LOW_RES])
    }

    fn decode_object(
        &self,
        frame: &FrameFeatures,
        embed: Option<&Tensor>,
        points: &[PointPrompt],
        size: (usize, usize),
        first: usize,
    ) -> Result<ObjectPass<Decoded>> {
        let gpu = &self.gpu;
        let embed = embed.unwrap_or(&frame.point.embed);
        let decoded = self.point.decode(gpu, embed, &frame.point, points, size, None, first)?;
        let score = gpu.download(&self.point.obj_score(gpu, &decoded)?)?[0];
        let ious = gpu.download(&decoded.iou)?;
        Ok(ObjectPass { decoded, score, ious })
    }

    fn first_mask(&self, dec: &Decoded) -> Result<Vec<f32>> {
        self.gpu.download(&row(&self.gpu, &dec.low_res, 0)?)
    }

    fn keep_mask(&self, dec: &Decoded, index: usize, present: bool, low_res: &Tensor, b: usize) -> Result<()> {
        let gpu = &self.gpu;
        let side = LOW_RES * LOW_RES;
        let first = 4 - dec.low_res.rows();
        gpu.copy_dtod(dec.low_res.at((index - first) * side), low_res.at(b * side), side * 4)?;
        if !present {
            launch!(gpu, "clamp", ew(side), B256, 0; Arg::slice(low_res.at(b * side), side), NO_OBJ_SCORE, NO_OBJ_SCORE)?;
        }
        Ok(())
    }

    fn pointer(&self, dec: &Decoded, token: usize, present: bool) -> Result<Tensor> {
        let gpu = &self.gpu;
        self.tracker.pointer(gpu, &row(gpu, &dec.tokens, MASK_TOKEN_ROW + token)?, present)
    }

    fn prompt_memories(&self, frame: &FrameFeatures, low_res: &Tensor, present: &[bool]) -> Result<Vec<Tensor>> {
        let gpu = &self.gpu;
        let objects = present.len();
        let high = upsample(gpu, low_res.into(), objects, IMAGE_SIZE, IMAGE_SIZE)?;
        let high = if objects > 1 {
            let out = gpu.empty(high.shape())?;
            launch!(gpu, "nonoverlap", ew(out.len()), B256, 0; &out, &high, objects)?;
            out
        } else {
            high
        };
        let input = mask_for_mem(gpu, &high, true)?;
        let n = IMAGE_SIZE * IMAGE_SIZE;
        let mut mem = Vec::with_capacity(objects);
        for (b, &present) in present.iter().enumerate() {
            let mask = gpu.empty(&[n])?;
            gpu.copy_dtod(input.at(b * n), mask.ptr(), n * 4)?;
            mem.push(self.tracker.memory(gpu, &frame.coarse, &mask, present)?);
        }
        Ok(mem)
    }

    fn object_memory(&self, frame: &FrameFeatures, low_res: &Tensor, b: usize, present: bool) -> Result<Tensor> {
        let gpu = &self.gpu;
        let side = LOW_RES * LOW_RES;
        let high = upsample(gpu, Arg::slice(low_res.at(b * side), side), 1, IMAGE_SIZE, IMAGE_SIZE)?;
        let input = mask_for_mem(gpu, &high, false)?;
        self.tracker.memory(gpu, &frame.coarse, &input, present)
    }

    /// The bank frame, with each object's cached keys in BF16 mode
    fn frame_memory(&self, mem: Vec<Tensor>, ptr: Vec<Tensor>) -> Result<FrameMemory> {
        let gpu = &self.gpu;
        let keys = if gpu.precision() == Precision::Bf16 {
            mem.iter().map(|m| self.tracker.mem_attn.frame_keys(gpu, m)).collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        Ok(FrameMemory { mem, ptr, keys })
    }

    /// Drops the frame's cached keys: older frames keep their pointers but nothing the
    /// memory attention alone reads
    fn retire(frame: &mut FrameMemory) {
        frame.keys.clear();
    }

    /// BF16 reads each bank frame's cached keys from layer 0's shared start; the other
    /// modes assemble every object's memory
    fn attention(&self, frame: &FrameFeatures, sel: &Selection<'_, FrameMemory>) -> Result<FrameAttention> {
        let ptr_pos = self.pointer_pos(sel)?;
        if self.gpu.precision() != Precision::Bf16 {
            return Ok(FrameAttention::Assembled(self.memory_inputs(sel, &ptr_pos)?.into_iter()));
        }
        // layer 0 up to its cross-attention is the same for every object
        let _r = range(c"memory_attention");
        let start = self.tracker.mem_attn.frame_start(&self.gpu, &frame.coarse)?;
        Ok(FrameAttention::Cached { start, ptr_pos })
    }

    fn attend(
        &self,
        frame: &FrameFeatures,
        attention: &mut FrameAttention,
        sel: &Selection<'_, FrameMemory>,
        b: usize,
    ) -> Result<Tensor> {
        match attention {
            FrameAttention::Assembled(inputs) => {
                let (memory, memory_pos) = inputs.next().ok_or_else(|| eyre!("no memory for object {b}"))?;
                self.tracker.mem_attn.forward(&self.gpu, &frame.coarse, &memory, &memory_pos, sel.ptr_tokens())
            }
            FrameAttention::Cached { start, ptr_pos } => self.memory_attention_cached(sel, ptr_pos, b, start),
        }
    }

    fn frame_masks(&self, low_res: &Tensor, (w, h): (usize, usize), out: &mut [Vec<u8>]) -> Result<()> {
        let gpu = &self.gpu;
        // each group is downloaded before the next, so the GPU holds one group's bytes and
        // the host only the objects' own masks
        let resize = MaskResize::new(gpu, out.len(), w, h)?;
        let plane = w * h;
        ensure!(out.iter().all(|m| m.len() == plane), "mask buffers for a {w}x{h} frame");
        let mask = gpu.alloc_bytes(resize.group() * plane)?;
        for range in resize.groups() {
            resize.run(gpu, low_res, range.clone(), Arg::slice(mask.ptr(), range.len() * plane))?;
            for (i, dst) in out[range].iter_mut().enumerate() {
                gpu.download_range_into(&mask, i * plane, dst)?;
            }
        }
        Ok(())
    }

    fn download_logits(&self, low_res: &Tensor) -> Result<Vec<f32>> {
        ensure!(low_res.cols() == LOW_RES * LOW_RES, "low-res logits {:?}", low_res.shape());
        self.gpu.download(low_res)
    }
}
