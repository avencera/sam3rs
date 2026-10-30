//! End-to-end SAM 3 image inference: encode an image once, then answer text, box and
//! point prompts against the cached features

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use eyre::{Result, ensure};

use crate::detector::{BoxPrompt, Detector, DetectorImage, QUERIES};
use crate::gpu::{Gpu, Precision, launch};
use crate::image::{RgbImage, preprocess};
use crate::nvtx::range;
use crate::ops::{B256, ew, upload_i32};
use crate::point::{LOW_RES, PointDecoder, PointImage, PointPrompt, clamped_mask, full_res_masks};
use crate::text::TextEncoder;
use crate::tokenizer::Tokenizer;
use crate::vision::{Backbone, Neck};
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

/// One detected instance
#[derive(Clone, Debug)]
pub struct Detection {
    /// sigmoid(logit) * sigmoid(presence)
    pub score: f32,
    /// `[x0, y0, x1, y1]` in pixels, not clamped to the image
    pub bbox: [f32; 4],
    /// `height * width` bytes, 1 = foreground
    pub mask: Vec<u8>,
    /// Index of the DETR query (0 to 199) that produced it
    pub query: usize,
}

/// The best of the three candidate masks for a point prompt
#[derive(Clone, Debug)]
pub struct PointMask {
    /// predicted IoU of the chosen mask
    pub score: f32,
    /// `height * width` bytes, 1 = foreground
    pub mask: Vec<u8>,
}

/// The loaded model on one GPU
///
/// Calls run one at a time on the GPU's single stream. The model is `Send`, so it can
/// move to another thread, but not `Sync`
pub struct Sam3 {
    id: u64,
    tokenizer: Tokenizer,
    backbone: Backbone,
    neck: Neck,
    tracker_neck: Neck,
    text: TextEncoder,
    detector: Detector,
    point: PointDecoder,
    // last, so the layers free their weights before the GPU handles go
    gpu: Gpu,
}

impl Sam3 {
    /// Loads `model.safetensors`, `vocab.json` and `merges.txt` from `dir` onto CUDA
    /// device `device`
    ///
    /// Fails without a usable NVIDIA driver, GPU or cuBLAS 13, and when the checkpoint
    /// doesn't match the SAM 3 weight schema
    pub fn load(dir: &Path, device: usize, precision: Precision) -> Result<Self> {
        let tokenizer = Tokenizer::open(dir)?;
        let wts = Weights::open(&dir.join("model.safetensors"))?;
        let gpu = Gpu::new(device, precision)?;
        let model = Self {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            tokenizer,
            backbone: Backbone::load(&gpu, &wts)?,
            neck: Neck::load(&gpu, &wts, "detector_model.vision_encoder.neck")?,
            tracker_neck: Neck::load(&gpu, &wts, "tracker_neck")?,
            text: TextEncoder::load(&gpu, &wts)?,
            detector: Detector::load(&gpu, &wts)?,
            point: PointDecoder::load(&gpu, &wts)?,
            gpu,
        };
        model.gpu.synchronize()?;
        Ok(model)
    }

    /// Downloads facebook/sam3 from Hugging Face (see [`crate::download_model`]), then
    /// loads it like [`Sam3::load`]
    #[cfg(feature = "online")]
    pub fn from_pretrained(device: usize, precision: Precision) -> Result<Self> {
        Self::load(&crate::download_model()?, device, precision)
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
        let point = self.point.prepare_image(gpu, &self.tracker_neck.forward(gpu, &trunk)?)?;
        let detector = self.detector.prepare_image(gpu, self.neck.forward(gpu, &trunk)?)?;
        gpu.synchronize()?;
        Ok(Embedding { model: self.id, width: img.width(), height: img.height(), detector, point })
    }

    /// Detects every instance matching `text` and the box prompts (pass [`VISUAL_PROMPT`]
    /// for boxes alone), keeping those scoring above [`SCORE_THRESHOLD`] in query order
    pub fn detect(&self, emb: &Embedding, text: &str, boxes: &[BoxPrompt]) -> Result<Vec<Detection>> {
        self.check(emb)?;
        let gpu = &self.gpu;
        let _r = range(c"detect");
        let feats = {
            let _r = range(c"text");
            self.text.forward(gpu, &self.tokenizer.tokenize(text)?)?
        };
        let out = self.detector.forward(gpu, &emb.detector, &feats, boxes)?;
        let _r = range(c"postprocess");
        let presence = gpu.download(&out.presence)?[0];
        let logits = gpu.download(&out.logits)?;
        let boxes = gpu.download(&out.boxes)?;

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let presence = sigmoid(presence);
        let kept: Vec<(usize, f32)> =
            (0..QUERIES).map(|i| (i, sigmoid(logits[i]) * presence)).filter(|(_, p)| *p > SCORE_THRESHOLD).collect();
        if kept.is_empty() {
            return Ok(vec![]);
        }

        let (w, h) = (emb.width, emb.height);
        let side = LOW_RES * LOW_RES;
        let idx: Vec<i32> = kept.iter().map(|(i, _)| *i as i32).collect();
        let idx = upload_i32(gpu, idx)?;
        let picked = gpu.empty(&[kept.len(), side])?;
        launch!(gpu, "gather_rows", ew(picked.len()), B256, 0; &picked, &out.masks, idx.ptr(), kept.len(), side)?;
        let up = gpu.empty(&[kept.len(), h * w])?;
        launch!(gpu, "resize_bilinear", ew(up.len()), B256, 0; &up, &picked, kept.len(), LOW_RES, LOW_RES, h, w)?;
        // sigmoid(x) > 0.5 is x > 0
        let mask = gpu.alloc_bytes(up.len())?;
        launch!(gpu, "threshold_u8", ew(up.len()), B256, 0; mask.ptr(), &up, up.len() as u64, 0.0f32)?;
        let mut masks = vec![0u8; up.len()];
        gpu.download_into(&mask, &mut masks)?;

        Ok(kept
            .iter()
            .zip(masks.chunks_exact(h * w))
            .map(|(&(i, score), mask)| {
                let [cx, cy, bw, bh] = [boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
                let (wf, hf) = (w as f32, h as f32);
                Detection {
                    score,
                    bbox: [(cx - bw / 2.0) * wf, (cy - bh / 2.0) * hf, (cx + bw / 2.0) * wf, (cy + bh / 2.0) * hf],
                    mask: mask.to_vec(),
                    query: i,
                }
            })
            .collect())
    }

    /// Segments the object under the clicks: a first pass without a mask, then a second
    /// with the first pass's best mask as the mask prompt (as upstream `predict_inst` users
    /// and osam do), returning the second pass's best mask after hole filling
    pub fn segment_points(&self, emb: &Embedding, points: &[PointPrompt]) -> Result<PointMask> {
        ensure!(!points.is_empty(), "segment_points needs at least one point");
        self.check(emb)?;
        let gpu = &self.gpu;
        let _r = range(c"segment_points");
        let (w, h) = (emb.width, emb.height);
        let first = self.point.forward(gpu, &emb.point, points, w, h, None)?;
        let mask_input = clamped_mask(gpu, &first.low_res, argmax(&first.iou))?;
        let second = self.point.forward(gpu, &emb.point, points, w, h, Some(&mask_input))?;
        let best = argmax(&second.iou);

        let _r = range(c"postprocess");
        let side = LOW_RES * LOW_RES;
        let low_res = gpu.empty(&[1, side])?;
        gpu.copy_dtod(second.low_res.at(best * side), low_res.ptr(), side * 4)?;
        let mask = full_res_masks(gpu, &low_res, 1, w, h)?;
        Ok(PointMask { score: second.iou[best], mask })
    }
}

/// First index of the maximum, like `torch.argmax`
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
