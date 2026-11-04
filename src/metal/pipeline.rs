//! End-to-end SAM 3 image inference on Metal: encode an image once, then answer text,
//! box and point prompts against the cached features

use std::cell::{Cell, OnceCell};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use eyre::{Result, ensure};

use crate::image::RgbImage;
use crate::masks::{MaskBuffer, MaskPool};
use crate::metal::detector::{Detector, DetectorImage, QUERIES};
use crate::metal::gpu::{Gpu, Tensor};
use crate::metal::image::preprocess;
use crate::metal::masks::upsample_masks;
use crate::metal::ops::gather_rows;
use crate::metal::point::{PointDecoder, PointImage, argmax, best_masks, fill_holes};
use crate::metal::text::{TextEncoder, TextFeatures};
use crate::metal::video::Tracker;
use crate::metal::vision::{Backbone, Neck, Trunk};
use crate::output::{
    Detection, Detections, HOLE_AREA, POINT_BATCH, PointMask, PointMasks, SCORE_THRESHOLD, VISUAL_PROMPT,
};
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::tokenizer::Tokenizer;
use crate::weights::Weights;

/// Tells models apart, so an embedding is only used with the model that made it
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(0);

/// Features of one encoded image, kept on the GPU of the model that encoded it
pub struct Embedding {
    model: u64,
    width: usize,
    height: usize,
    /// The detector neck's levels and what the detector derives from them
    detector: DetectorImage,
    /// The point decoder's features, from the tracker neck's levels
    pub(crate) point: PointImage,
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

/// The loaded model on the Mac's GPU
///
/// Calls run one at a time on the model's command queue. The model is `Send`, so it can
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
    /// Loads `model.safetensors`, `vocab.json` and `merges.txt` from `dir` onto the Metal
    /// GPU; `device` must be 0, the Mac's one GPU
    ///
    /// Fails without a Metal 4 GPU (Apple silicon on macOS 26 or later), with
    /// [`Precision::Tf32`], which has no Metal equivalent, and when the checkpoint
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
        // a load that dispatched nothing still surfaces a kernel compile error here
        model.gpu.ready()?;
        model.gpu.synchronize()?;
        Ok(model)
    }

    /// Numeric mode the model was loaded with
    pub fn precision(&self) -> Precision {
        self.gpu.precision()
    }

    fn check(&self, emb: &Embedding) -> Result<()> {
        ensure!(emb.model == self.id, "the embedding was encoded by a different Sam3 model");
        Ok(())
    }

    /// Runs the vision backbone and both necks; the result serves any number of prompts
    pub fn encode(&self, img: &RgbImage) -> Result<Embedding> {
        let gpu = &self.gpu;
        let pixels = preprocess(gpu, img)?;
        let trunk = self.backbone.forward(gpu, &pixels)?;
        drop(pixels);
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
        let (w, h) = (emb.width, emb.height);
        // one mask more than last time, since a box being dragged out tends to gain them
        let reserved = self.masks.reserve((self.last_kept.get() + 1) * w * h);
        let encode = || self.text.forward(gpu, &self.tokenizer.tokenize(text)?);
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
        let presence = gpu.download(&out.presence)?[0];
        let logits = gpu.download(&out.logits)?;
        let boxes = gpu.download(&out.boxes)?;
        ensure!(
            logits.len() == QUERIES && boxes.len() == QUERIES * 4,
            "{} logits and {} box values",
            logits.len(),
            boxes.len()
        );

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let presence = sigmoid(presence);
        let kept: Vec<(usize, f32)> =
            (0..QUERIES).map(|i| (i, sigmoid(logits[i]) * presence)).filter(|(_, p)| *p > SCORE_THRESHOLD).collect();
        self.last_kept.set(kept.len());
        if kept.is_empty() {
            self.masks.release(reserved);
            return Ok(Detections::default());
        }

        let idx: Vec<i32> = kept.iter().map(|(i, _)| *i as i32).collect();
        let picked = gather_rows(gpu, &out.masks, &idx)?;
        let mask = upsample_masks(gpu, &picked, 0, kept.len(), w, h)?;
        gpu.flush()?;
        // taken after the launches are committed, so a buffer still faulting in overlaps
        // the GPU work
        let mut masks = self.masks.take(reserved, kept.len() * h * w)?;
        gpu.read(&mask.buf, 0, &mut masks)?;

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
    /// with the first pass's best mask as the mask prompt, returning the second pass's
    /// best mask after hole filling
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
            let (best, iou) = self.decode_points(emb, &points)?;
            fill_holes(gpu, &best, HOLE_AREA)?;
            let full = upsample_masks(gpu, &best, 0, batch.len(), w, h)?;
            if let Some(reserved) = reserved.take() {
                // taken after the launches, so a buffer still faulting in overlaps the
                // GPU work
                masks = self.masks.take(reserved, prompts.len() * plane)?;
            }
            // the click's one wait: the scores, after every launch
            let iou = gpu.download(&iou)?;
            ensure!(iou.len() == 4 * batch.len(), "{} IoUs for {} prompts", iou.len(), batch.len());
            for (&i, c) in batch.iter().zip(iou.chunks(4)) {
                scores[i] = c[1 + argmax(&c[1..])];
            }
            // consecutive prompts download together
            let mut j = 0;
            for run in batch.chunk_by(|&a, &b| b == a + 1) {
                gpu.read(&full.buf, j * plane, &mut masks[run[0] * plane..][..run.len() * plane])?;
                j += run.len();
            }
        }
        Ok(PointMasks { scores, masks })
    }

    /// Both decoder passes over prompts with equal click counts, launched without waiting:
    /// the best second-pass mask of each `[K, 288^2]`, before hole filling, and the
    /// second pass's IoUs `[K, 4]`, whose candidate argmax picked it
    fn decode_points(&self, emb: &Embedding, prompts: &[&[PointPrompt]]) -> Result<(Tensor, Tensor)> {
        let gpu = &self.gpu;
        let size = (emb.width, emb.height);
        let img = &emb.point;
        let first = self.point.decode_batch(gpu, &img.embed, img, prompts, size, None, 1)?;
        let mask_input = best_masks(gpu, &first.low_res, &first.iou, true)?;
        let second = self.point.decode_batch(gpu, &img.embed, img, prompts, size, Some(&mask_input), 1)?;
        Ok((best_masks(gpu, &second.low_res, &second.iou, false)?, second.iou))
    }
}

/// Timing probes, ignored by default: where a click's wall time goes and per-kernel GPU
/// time of tracked video frames
#[cfg(test)]
mod probe {
    use std::time::Instant;

    use eyre::Result;

    use super::Sam3;
    use crate::image::RgbImage;
    use crate::precision::Precision;
    use crate::prompt::PointPrompt;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2] * 1e3
    }

    /// Where a click's wall time goes: the whole click, one decoder pass with its wait,
    /// passes queued back to back, and the host encoding of a pass alone
    #[test]
    #[ignore = "needs a Metal 4 GPU and the weights; timing only"]
    fn click_breakdown() -> Result<()> {
        let m = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
        let img = RgbImage::open("models/images/dogs.jpg".as_ref())?;
        let emb = m.encode(&img)?;
        let pts = [PointPrompt::new(1439.0, 504.0, true)?];
        let gpu = &m.gpu;
        let size = (emb.width, emb.height);
        let pass = || m.point.decode_batch(gpu, &emb.point.embed, &emb.point, &[&pts[..]], size, None, 1);
        let (mut full, mut one, mut encode, mut queued) = (vec![], vec![], vec![], vec![]);
        for i in 0..23 {
            let t0 = Instant::now();
            m.segment_points(&emb, &pts)?;
            let t_full = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let d = pass()?;
            let t_enc = t0.elapsed().as_secs_f64();
            gpu.download(&d.iou)?;
            let t_one = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            for _ in 0..10 {
                pass()?;
            }
            gpu.synchronize()?;
            let t_q = t0.elapsed().as_secs_f64() / 10.0;
            if i >= 3 {
                full.push(t_full);
                one.push(t_one);
                encode.push(t_enc);
                queued.push(t_q);
            }
        }
        println!(
            "click {:.2} ms; one pass {:.2} ms (host encode {:.2} ms); queued passes {:.2} ms each",
            median(full),
            median(one),
            median(encode),
            median(queued)
        );
        Ok(())
    }

    /// Per-kernel GPU time of tracked horses frames (3 objects), after 8 warm frames;
    /// set `SAM3RS_METAL_PROFILE`
    #[test]
    #[ignore = "needs a Metal 4 GPU, the weights and the horses frames; timing only"]
    fn video_profile() -> Result<()> {
        let m = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
        let mut paths: Vec<_> = std::fs::read_dir("models/video-frames/horses-720p-64")?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        paths.sort();
        let frames: Vec<RgbImage> = paths.iter().take(14).map(|p| RgbImage::open(p)).collect::<Result<_>>()?;
        let (w, h) = (frames[0].width() as f32, frames[0].height() as f32);
        let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/video_prompts.json")?)?;
        let clip = prompts.as_array().unwrap().iter().find(|c| c["clip"] == "horses").unwrap();
        let clicks = clip["objects"]
            .as_array()
            .unwrap()
            .iter()
            .take(3)
            .map(|o| {
                let (x, y) = (o["point"][0].as_f64().unwrap() as f32, o["point"][1].as_f64().unwrap() as f32);
                Ok(vec![PointPrompt::new(x * w, y * h, true)?])
            })
            .collect::<Result<Vec<_>>>()?;

        let (mut tracker, _) = m.track(&frames[0], &clicks, frames.len())?;
        for f in &frames[1..9] {
            tracker.next(f)?;
        }
        m.gpu.take_profile();
        let measured = &frames[9..];
        let t0 = Instant::now();
        for f in measured {
            tracker.next(f)?;
        }
        let wall = t0.elapsed().as_secs_f64() / measured.len() as f64;
        let mut rows = m.gpu.take_profile();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        let total: f64 = rows.iter().map(|r| r.2).sum();
        let per = 1e3 / measured.len() as f64;
        println!("frame wall {:.1} ms (profiled), GPU {:.1} ms", wall * 1e3, total * per);
        for (name, count, t) in rows {
            println!(
                "  {name:>28} {:>7.1} {:>9.2} ms {:>5.1}%",
                count as f64 / measured.len() as f64,
                t * per,
                100.0 * t / total
            );
        }
        Ok(())
    }
}
