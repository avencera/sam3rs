//! The Metal backend's model: every stage's layers on the Mac's GPU, and the [`Backend`]
//! steps the public API runs them through

use eyre::{Result, ensure, eyre};

use crate::backend::{Backend, DetectionScores, ObjectPass};
use crate::host::argmax;
use crate::image::RgbImage;
use crate::masks::MaskSink;
use crate::metal::detector::{Detector, DetectorImage, DetectorOutput};
use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::image::preprocess;
use crate::metal::masks::upsample_masks;
use crate::metal::ops::gather_rows;
use crate::metal::point::{Decoded, PointDecoder, PointImage, best_masks, fill_holes, row};
use crate::metal::text::{TextEncoder, TextFeatures};
use crate::metal::video::{FrameAttention, FrameFeatures, FrameMemory, Tracker, copy, mask_for_mem, upsample};
use crate::metal::vision::{Backbone, Neck, Trunk};
use crate::model::point::{LOW_RES, MASK_TOKEN_ROW};
use crate::model::tracker::NO_OBJ_SCORE;
use crate::model::vision::IMAGE_SIZE;
use crate::output::HOLE_AREA;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::tokenizer::CONTEXT;
use crate::tracker::Selection;
use crate::weights::Weights;

/// The loaded model on the Mac's GPU
pub(crate) struct Metal {
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
    /// The detector neck's levels and what the detector derives from them
    detector: DetectorImage,
    /// The point decoder's features, from the tracker neck's levels
    pub(crate) point: PointImage,
}

/// Both point decoder passes over a batch, launched without waiting: the best
/// second-pass mask of each prompt `[K, 288^2]`, before hole filling, and the second
/// pass's IoUs `[K, 4]`, whose candidate argmax picked it
pub(crate) struct PointPass {
    best: Tensor,
    iou: Tensor,
}

impl Metal {
    /// Both decoder passes over prompts with equal click counts
    pub(crate) fn decode_point_batch(
        &self,
        img: &PointImage,
        prompts: &[&[PointPrompt]],
        size: (usize, usize),
    ) -> Result<PointPass> {
        let gpu = &self.gpu;
        let first = self.point.decode_batch(gpu, &img.embed, img, prompts, size, None, 1)?;
        let mask_input = best_masks(gpu, &first.low_res, &first.iou, true)?;
        let second = self.point.decode_batch(gpu, &img.embed, img, prompts, size, Some(&mask_input), 1)?;
        Ok(PointPass { best: best_masks(gpu, &second.low_res, &second.iou, false)?, iou: second.iou })
    }

    /// The hole-filled best masks of a pass at full size into `sink`, and their
    /// predicted IoUs
    pub(crate) fn point_batch_masks(
        &self,
        pass: PointPass,
        (w, h): (usize, usize),
        sink: &mut MaskSink,
    ) -> Result<Vec<f32>> {
        let gpu = &self.gpu;
        let PointPass { best, iou } = pass;
        let n = iou.rows();
        fill_holes(gpu, &best, HOLE_AREA)?;
        let full = upsample_masks(gpu, &best, 0, n, w, h)?;
        // taken after the launches, so a buffer still faulting in overlaps the GPU work
        sink.ready()?;
        // the click's one wait: the scores, after every launch
        let iou = gpu.download(&iou)?;
        ensure!(iou.len() == 4 * n, "{} IoUs for {n} prompts", iou.len());
        let scores = iou.chunks(4).map(|c| c[1 + argmax(&c[1..])]).collect();
        // consecutive prompts download together
        sink.write(0..n, |offset, dst| gpu.read(&full.buf, offset, dst))?;
        Ok(scores)
    }
}

impl Backend for Metal {
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
        // a load that dispatched nothing still surfaces a kernel compile error here
        model.gpu.ready()?;
        model.gpu.synchronize()?;
        Ok(model)
    }

    fn precision(&self) -> Precision {
        self.gpu.precision()
    }

    /// Nothing to do: a Metal queue isn't bound to a thread
    fn bind(&self) -> Result<()> {
        Ok(())
    }

    fn encode(&self, img: &RgbImage) -> Result<Image> {
        let gpu = &self.gpu;
        let pixels = preprocess(gpu, img)?;
        let trunk = self.backbone.forward(gpu, &pixels)?;
        drop(pixels);
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
        let idx: Vec<i32> = kept.iter().map(|&i| i as i32).collect();
        let picked = gather_rows(gpu, &out.masks, &idx)?;
        let mask = upsample_masks(gpu, &picked, 0, kept.len(), w, h)?;
        // the buffer comes out of the pool after the launches are committed, so one still
        // faulting in overlaps the GPU work
        gpu.flush()?;
        sink.write(0..kept.len(), |offset, dst| gpu.read(&mask.buf, offset, dst))
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
        self.gpu.empty(&[objects, LOW_RES * LOW_RES], DType::F32)
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
        ensure!(index >= first, "mask {index} of a pass that kept masks {first}..4");
        copy(gpu, low_res, b * side, &dec.low_res, (index - first) * side, side)?;
        if !present {
            let args = [low_res.at(b * side), NO_OBJ_SCORE.into(), NO_OBJ_SCORE.into()];
            gpu.dispatch_1d("clamp_f32", side, &args)?;
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
        let high = upsample(gpu, low_res, 0, objects, IMAGE_SIZE, IMAGE_SIZE)?;
        let n = IMAGE_SIZE * IMAGE_SIZE;
        let high = if objects > 1 {
            let out = gpu.empty(high.shape(), DType::F32)?;
            gpu.dispatch_1d("nonoverlap", out.len(), &[out.arg(), high.arg(), objects.into(), n.into()])?;
            out
        } else {
            high
        };
        let input = mask_for_mem(gpu, &high, true)?;
        let mut mem = Vec::with_capacity(objects);
        for (b, &present) in present.iter().enumerate() {
            let mask = gpu.empty(&[n], DType::F32)?;
            copy(gpu, &mask, 0, &input, b * n, n)?;
            mem.push(self.tracker.memory(gpu, &frame.coarse, &mask, present)?);
        }
        Ok(mem)
    }

    fn object_memory(&self, frame: &FrameFeatures, low_res: &Tensor, b: usize, present: bool) -> Result<Tensor> {
        let gpu = &self.gpu;
        let high = upsample(gpu, low_res, b, 1, IMAGE_SIZE, IMAGE_SIZE)?;
        let input = mask_for_mem(gpu, &high, false)?;
        self.tracker.memory(gpu, &frame.coarse, &input, present)
    }

    fn frame_memory(&self, mem: Vec<Tensor>, ptr: Vec<Tensor>) -> Result<FrameMemory> {
        Ok(FrameMemory { mem, ptr })
    }

    /// Every object's assembled memory, and the frame's features through the first
    /// layer's self-attention, which sees no memory
    fn attention(&self, frame: &FrameFeatures, sel: &Selection<'_, FrameMemory>) -> Result<FrameAttention> {
        let inputs = self.memory_inputs(sel)?;
        let prepared = self.tracker.mem_attn.prepare(&self.gpu, &frame.coarse)?;
        Ok(FrameAttention { inputs: inputs.into_iter(), prepared })
    }

    fn attend(
        &self,
        _frame: &FrameFeatures,
        attention: &mut FrameAttention,
        sel: &Selection<'_, FrameMemory>,
        b: usize,
    ) -> Result<Tensor> {
        let (memory, memory_pos) = attention.inputs.next().ok_or_else(|| eyre!("no memory for object {b}"))?;
        self.tracker.mem_attn.forward(&self.gpu, &attention.prepared, &memory, &memory_pos, sel.ptr_tokens())
    }

    fn frame_masks(&self, low_res: &Tensor, (w, h): (usize, usize), out: &mut [Vec<u8>]) -> Result<()> {
        let gpu = &self.gpu;
        let plane = w * h;
        ensure!(out.iter().all(|m| m.len() == plane), "mask buffers for a {w}x{h} frame");
        let full = upsample_masks(gpu, low_res, 0, out.len(), w, h)?;
        for (b, dst) in out.iter_mut().enumerate() {
            gpu.read(&full.buf, b * plane, dst)?;
        }
        Ok(())
    }

    fn download_logits(&self, low_res: &Tensor) -> Result<Vec<f32>> {
        self.gpu.download(low_res)
    }
}
