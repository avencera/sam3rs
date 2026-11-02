//! End-to-end SAM 3 image inference: encode an image once, then answer text, box and
//! point prompts against the cached features

use std::cell::{Cell, OnceCell};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use eyre::{Result, ensure};

use crate::detector::{BoxPrompt, Detector, DetectorImage, QUERIES};
use crate::gpu::{Arg, Gpu, Precision, Tensor, launch};
use crate::image::{RgbImage, preprocess};
use crate::masks::{MaskBuffer, MaskPool};
use crate::nvtx::range;
use crate::ops::{B256, ew, upload_i32};
use crate::point::{
    LOW_RES, MaskResize, PointDecoder, PointImage, PointPrompt, argmax, clamped_masks, fill_holes, pick_masks,
};
use crate::text::{TextEncoder, TextFeatures};
use crate::tokenizer::Tokenizer;
use crate::video::Tracker;
use crate::vision::{Backbone, Neck, Trunk};
use crate::weights::Weights;

/// `Sam3Processor.confidence_threshold`
pub const SCORE_THRESHOLD: f32 = 0.5;

/// Text used for box-only prompts, as in upstream and osam
pub const VISUAL_PROMPT: &str = "visual";

/// Tells models apart, so an embedding is only used with the model that made it
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(0);

/// Features of one encoded image, kept on the GPU of the model that encoded it
pub struct Embedding {
    model: u64,
    width: usize,
    height: usize,
    detector: DetectorImage,
    point: PointImage,
}

impl Embedding {
    /// Width of the encoded image in pixels
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the encoded image in pixels
    pub fn height(&self) -> usize {
        self.height
    }
}

/// One detected instance; its mask is [`Detections::mask`]
#[derive(Clone, Debug)]
pub struct Detection {
    /// sigmoid(logit) * sigmoid(presence)
    pub score: f32,
    /// `[x0, y0, x1, y1]` in pixels, not clamped to the image
    pub bbox: [f32; 4],
    /// Index of the DETR query (0 to 199) that produced it
    pub query: usize,
}

/// The instances [`Sam3::detect`] found, in DETR query order, with all their masks in
/// one buffer
///
/// One buffer lets callers take every mask without copying it again: at 2560 px a mask
/// is 5 MB, and copying a stack of them into fresh memory cost more than the detection
#[derive(Clone, Debug, Default)]
pub struct Detections {
    instances: Vec<Detection>,
    masks: MaskBuffer,
}

impl Detections {
    /// Number of instances
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether nothing scored above [`SCORE_THRESHOLD`]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// The instances, without their masks
    pub fn iter(&self) -> std::slice::Iter<'_, Detection> {
        self.instances.iter()
    }

    /// Mask of instance `i`: `height * width` bytes, 1 = foreground
    ///
    /// Panics if `i` is out of range
    pub fn mask(&self, i: usize) -> &[u8] {
        assert!(i < self.len(), "mask {i} of {} detections", self.len());
        let pixels = self.masks.len() / self.len();
        &self.masks[i * pixels..][..pixels]
    }

    /// Every mask in instance order: `len * height * width` bytes, 1 = foreground
    pub fn masks(&self) -> &[u8] {
        &self.masks
    }

    /// Splits into the instances and the mask buffer of [`Detections::masks`]
    pub fn into_parts(self) -> (Vec<Detection>, MaskBuffer) {
        (self.instances, self.masks)
    }
}

impl IntoIterator for Detections {
    type Item = Detection;
    type IntoIter = std::vec::IntoIter<Detection>;

    fn into_iter(self) -> Self::IntoIter {
        self.instances.into_iter()
    }
}

impl<'a> IntoIterator for &'a Detections {
    type Item = &'a Detection;
    type IntoIter = std::slice::Iter<'a, Detection>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The best of the three candidate masks for a point prompt
#[derive(Clone, Debug)]
pub struct PointMask {
    /// Predicted IoU of the chosen mask
    pub score: f32,
    /// `height * width` bytes, 1 = foreground
    pub mask: MaskBuffer,
}

/// Largest background region of a point mask that hole filling turns into foreground,
/// in low-res pixels, as upstream's `postprocess_masks`
const HOLE_AREA: usize = 256;

/// Most prompts of equal click counts [`Sam3::segment_points_batch`] decodes together.
/// Batches of 32 measured no faster per prompt than 16 on an RTX 5090, and every
/// decoder activation grows with the batch
pub const POINT_BATCH: usize = 16;

/// One mask per prompt of [`Sam3::segment_points_batch`], in prompt order, with all the
/// masks in one buffer as in [`Detections`]
#[derive(Clone, Debug, Default)]
pub struct PointMasks {
    scores: Vec<f32>,
    masks: MaskBuffer,
}

impl PointMasks {
    /// Number of prompts
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether there were no prompts
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Predicted IoU of each prompt's chosen mask
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Mask of prompt `i`: `height * width` bytes, 1 = foreground
    ///
    /// Panics if `i` is out of range
    pub fn mask(&self, i: usize) -> &[u8] {
        assert!(i < self.len(), "mask {i} of {} prompts", self.len());
        let pixels = self.masks.len() / self.len();
        &self.masks[i * pixels..][..pixels]
    }

    /// Every mask in prompt order: `len * height * width` bytes, 1 = foreground
    pub fn masks(&self) -> &[u8] {
        &self.masks
    }

    /// Splits into the scores and the mask buffer of [`PointMasks::masks`]
    pub fn into_parts(self) -> (Vec<f32>, MaskBuffer) {
        (self.scores, self.masks)
    }
}

/// The loaded model on one GPU
///
/// Calls run one at a time on the GPU's single stream. The model is `Send`, so it can
/// move to another thread, but not `Sync`
pub struct Sam3 {
    id: u64,
    tokenizer: Tokenizer,
    pub(crate) backbone: Backbone,
    neck: Neck,
    pub(crate) tracker_neck: Neck,
    text: TextEncoder,
    detector: Detector,
    pub(crate) point: PointDecoder,
    /// Memory attention, memory encoder and pointer heads of the video tracker
    pub(crate) tracker: Tracker,
    /// Text features of [`VISUAL_PROMPT`], encoded on first use: every box-only prompt
    /// would otherwise run the 24-layer text encoder on the same string
    visual: OnceCell<TextFeatures>,
    /// Detections the last `detect` call kept, the guess for how many masks of host
    /// memory to set aside ahead of the next
    last_kept: Cell<usize>,
    /// Host memory of dropped results, reused for later masks
    masks: MaskPool,
    // last, so the layers free their weights before the GPU handles go
    pub(crate) gpu: Gpu,
}

impl Sam3 {
    /// Loads `model.safetensors`, `vocab.json` and `merges.txt` from `dir` onto CUDA
    /// device `device`
    ///
    /// Fails without a usable NVIDIA driver, GPU or cuBLAS 13, and when the checkpoint
    /// doesn't match the SAM 3 weight schema
    pub fn load(dir: &Path, device: usize, precision: Precision) -> Result<Self> {
        Self::load_onto(Gpu::new(device, precision)?, dir)
    }

    /// Downloads facebook/sam3 from Hugging Face (see [`crate::download_model`]), then
    /// loads it like [`Sam3::load`]
    ///
    /// Opens the GPU first, so a machine that can't run the model fails before the
    /// 3.4 GB download
    #[cfg(feature = "online")]
    pub fn from_pretrained(device: usize, precision: Precision) -> Result<Self> {
        let gpu = Gpu::new(device, precision)?;
        Self::load_onto(gpu, &crate::download_model()?)
    }

    fn load_onto(gpu: Gpu, dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::open(dir)?;
        let wts = Weights::open(&dir.join("model.safetensors"))?;
        let model = Self {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            tokenizer,
            backbone: Backbone::load(&gpu, &wts)?,
            neck: Neck::load(&gpu, &wts, "detector_model.vision_encoder.neck")?,
            tracker_neck: Neck::load(&gpu, &wts, "tracker_neck")?,
            text: TextEncoder::load(&gpu, &wts)?,
            detector: Detector::load(&gpu, &wts)?,
            point: PointDecoder::load(&gpu, &wts)?,
            tracker: Tracker::load(&gpu, &wts)?,
            visual: OnceCell::new(),
            last_kept: Cell::new(0),
            masks: MaskPool::default(),
            gpu,
        };
        model.gpu.synchronize()?;
        Ok(model)
    }

    /// Numeric mode the model was loaded with
    pub fn precision(&self) -> Precision {
        self.gpu.precision()
    }

    fn check(&self, emb: &Embedding) -> Result<()> {
        ensure!(emb.model == self.id, "the embedding was encoded by a different Sam3 model");
        self.gpu.bind()
    }

    /// Runs the vision backbone and both necks; the result serves any number of prompts
    pub fn encode(&self, img: &RgbImage) -> Result<Embedding> {
        let gpu = &self.gpu;
        gpu.bind()?;
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
        Ok(Embedding { model: self.id, width: img.width(), height: img.height(), detector, point })
    }

    /// Detects every instance matching `text` and the box prompts (pass [`VISUAL_PROMPT`]
    /// for boxes alone), keeping those scoring above [`SCORE_THRESHOLD`] in query order.
    /// Each call runs the text encoder on `text`, except for [`VISUAL_PROMPT`], whose
    /// features are encoded on the first call and reused
    pub fn detect(&self, emb: &Embedding, text: &str, boxes: &[BoxPrompt]) -> Result<Detections> {
        self.check(emb)?;
        let gpu = &self.gpu;
        let _r = range(c"detect");
        let (w, h) = (emb.width, emb.height);
        // one mask more than last time, since a box being dragged out tends to gain them
        let reserved = self.masks.reserve((self.last_kept.get() + 1) * w * h);
        let encode = || {
            let _r = range(c"text");
            self.text.forward(gpu, &self.tokenizer.tokenize(text)?)
        };
        let encoded;
        let feats = if text == VISUAL_PROMPT {
            match self.visual.get() {
                Some(feats) => feats,
                None => {
                    let feats = encode()?;
                    self.visual.get_or_init(|| feats)
                }
            }
        } else {
            encoded = encode()?;
            &encoded
        };
        let out = self.detector.forward(gpu, &emb.detector, feats, boxes)?;
        let _r = range(c"postprocess");
        let presence = gpu.download(&out.presence)?[0];
        let logits = gpu.download(&out.logits)?;
        let boxes = gpu.download(&out.boxes)?;

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let presence = sigmoid(presence);
        let kept: Vec<(usize, f32)> =
            (0..QUERIES).map(|i| (i, sigmoid(logits[i]) * presence)).filter(|(_, p)| *p > SCORE_THRESHOLD).collect();
        self.last_kept.set(kept.len());
        if kept.is_empty() {
            self.masks.release(reserved);
            return Ok(Detections::default());
        }

        let side = LOW_RES * LOW_RES;
        let idx: Vec<i32> = kept.iter().map(|(i, _)| *i as i32).collect();
        let idx = upload_i32(gpu, idx)?;
        let picked = gpu.empty(&[kept.len(), side])?;
        launch!(gpu, "gather_rows", ew(picked.len()), B256, 0; &picked, &out.masks, idx.slice::<i32>(), side)?;
        let mask = MaskResize::new(gpu, kept.len(), w, h)?.all(gpu, &picked)?;
        // taken after the launches, so a buffer still faulting in overlaps the GPU work
        let mut masks = self.masks.take(reserved, kept.len() * h * w)?;
        gpu.download_into(&mask, &mut masks)?;

        let instances = kept
            .iter()
            .map(|&(i, score)| {
                let [cx, cy, bw, bh] = [boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
                let (wf, hf) = (w as f32, h as f32);
                Detection {
                    score,
                    bbox: [(cx - bw / 2.0) * wf, (cy - bh / 2.0) * hf, (cx + bw / 2.0) * wf, (cy + bh / 2.0) * hf],
                    query: i,
                }
            })
            .collect();
        Ok(Detections { instances, masks })
    }

    /// Segments the object under the clicks: a first pass without a mask, then a second
    /// with the first pass's best mask as the mask prompt (as upstream `predict_inst` users
    /// and osam do), returning the second pass's best mask after hole filling
    pub fn segment_points(&self, emb: &Embedding, points: &[PointPrompt]) -> Result<PointMask> {
        ensure!(!points.is_empty(), "segment_points needs at least one point");
        let (mut scores, mask) = self.segment_points_batch(emb, &[points])?.into_parts();
        Ok(PointMask { score: scores.remove(0), mask })
    }

    /// [`Self::segment_points`] for several independent objects at once: prompt `i`'s
    /// clicks give object `i`, with the score and mask `segment_points` would give it
    /// alone, as upstream `predict_inst` does with a batch of prompts
    ///
    /// Prompts with the same number of clicks run through each decoder pass together, up
    /// to [`POINT_BATCH`] at a time, so the decoder's per-call costs are paid once per
    /// batch rather than once per object. Each prompt needs at least one click; no prompts
    /// give an empty result. Every mask is `height * width` bytes of host memory, so a
    /// large image times many prompts takes a lot of it: 50 MB a mask at 8160x6144
    pub fn segment_points_batch(&self, emb: &Embedding, prompts: &[&[PointPrompt]]) -> Result<PointMasks> {
        ensure!(prompts.iter().all(|p| !p.is_empty()), "every point prompt needs at least one point");
        self.check(emb)?;
        if prompts.is_empty() {
            return Ok(PointMasks::default());
        }
        let gpu = &self.gpu;
        let _r = range(c"segment_points");
        let (w, h) = (emb.width, emb.height);
        let plane = w * h;
        let reserved = self.masks.reserve(prompts.len() * plane);

        // runs of equal click counts, in input order within each run, so the common case
        // of one count is one run of consecutive masks
        let mut order: Vec<usize> = (0..prompts.len()).collect();
        order.sort_by_key(|&i| prompts[i].len());
        let mut scores = vec![f32::NAN; prompts.len()];
        let mut reserved = Some(reserved);
        let mut masks = MaskBuffer::default();
        for batch in
            order.chunk_by(|&a, &b| prompts[a].len() == prompts[b].len()).flat_map(|run| run.chunks(POINT_BATCH))
        {
            let points: Vec<&[PointPrompt]> = batch.iter().map(|&i| prompts[i]).collect();
            let (best, ious) = self.decode_points(emb, &points)?;
            for (&i, iou) in batch.iter().zip(ious) {
                scores[i] = iou;
            }

            let _r = range(c"postprocess");
            fill_holes(gpu, &best, HOLE_AREA)?;
            let resize = MaskResize::new(gpu, batch.len(), w, h)?;
            let group = gpu.alloc_bytes(resize.group() * plane)?;
            for part in resize.groups() {
                resize.run(gpu, &best, part.clone(), Arg::slice(group.ptr(), part.len() * plane))?;
                if let Some(reserved) = reserved.take() {
                    // taken after the first launches, so a buffer still faulting in
                    // overlaps the GPU work
                    masks = self.masks.take(reserved, prompts.len() * plane)?;
                }
                // consecutive prompts download together
                let ids = &batch[part];
                let mut j = 0;
                for run in ids.chunk_by(|&a, &b| b == a + 1) {
                    let dst = &mut masks[run[0] * plane..][..run.len() * plane];
                    gpu.download_range_into(&group, j * plane, dst)?;
                    j += run.len();
                }
            }
        }
        Ok(PointMasks { scores, masks })
    }

    /// Both decoder passes over prompts with equal click counts: the best second-pass
    /// mask of each `[K, 288^2]`, before hole filling, and its predicted IoU
    fn decode_points(&self, emb: &Embedding, prompts: &[&[PointPrompt]]) -> Result<(Tensor, Vec<f32>)> {
        let gpu = &self.gpu;
        let size = (emb.width, emb.height);
        let img = &emb.point;
        // mask tokens 1 to 3, the multimask candidates
        let best = |iou: &[f32]| -> Vec<usize> { iou.chunks(4).map(|c| argmax(&c[1..])).collect() };
        let first = self.point.decode_batch(gpu, &img.embed, img, prompts, size, None, 1)?;
        let mask_input = clamped_masks(gpu, &first.low_res, &best(&gpu.download(&first.iou)?))?;
        let second = self.point.decode_batch(gpu, &img.embed, img, prompts, size, Some(&mask_input), 1)?;
        let iou = gpu.download(&second.iou)?;
        let picks = best(&iou);
        let scores = picks.iter().enumerate().map(|(b, &i)| iou[b * 4 + 1 + i]).collect();
        Ok((pick_masks(gpu, &second.low_res, &picks)?, scores))
    }
}
