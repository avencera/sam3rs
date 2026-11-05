//! End-to-end SAM 3 image inference: encode an image once, then answer text, box and
//! point prompts against the cached features, on whichever backend the build runs

use std::cell::{Cell, OnceCell};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use eyre::{Result, ensure};

use crate::backend::{Active, Backend, DetectionScores};
use crate::image::RgbImage;
use crate::masks::{MaskBuffer, MaskPool, MaskSink};
use crate::model::detector::QUERIES;
use crate::nvtx::range;
use crate::output::{Detection, Detections, POINT_BATCH, PointMask, PointMasks, SCORE_THRESHOLD, VISUAL_PROMPT};
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
    pub(crate) image: <Active as Backend>::Image,
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

    pub(crate) fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// The loaded model on one GPU: an NVIDIA GPU through CUDA, or the Mac's GPU through
/// Metal on macOS
///
/// Calls run one at a time on the model's single CUDA stream or Metal command queue. The
/// model is `Send`, so it can move to another thread, but not `Sync`
pub struct Sam3 {
    id: u64,
    tokenizer: Tokenizer,
    /// Text features of [`VISUAL_PROMPT`], encoded on first use: every box-only prompt
    /// would otherwise run the 24-layer text encoder on the same string
    visual: OnceCell<<Active as Backend>::Text>,
    /// Detections the last `detect` call kept, the guess for how many masks of host
    /// memory to set aside ahead of the next
    last_kept: Cell<usize>,
    /// Host memory of dropped results, reused for later masks
    masks: MaskPool,
    // last, so the layers free their weights before the GPU handles go
    pub(crate) backend: Active,
}

impl Sam3 {
    /// Loads `model.safetensors`, `vocab.json` and `merges.txt` from `dir` onto GPU
    /// `device`: a CUDA device ordinal, or 0 on macOS, where Metal has the Mac's one GPU
    ///
    /// Fails when the checkpoint doesn't match the SAM 3 weight schema, and when the
    /// machine can't run the model: on CUDA, without a usable NVIDIA driver, GPU or
    /// cuBLAS 13; on Metal, without a Metal 4 GPU (Apple silicon on macOS 26.1 or later)
    /// or with [`Precision::Tf32`], which has no Metal equivalent
    pub fn load(dir: &Path, device: usize, precision: Precision) -> Result<Self> {
        Self::load_onto(Active::open(device, precision)?, dir)
    }

    /// Downloads the weights from Hugging Face (see [`crate::download_model`]), then
    /// loads it like [`Sam3::load`]
    ///
    /// Opens the GPU first, so a machine that can't run the model fails before the
    /// 3.4 GB download
    #[cfg(feature = "online")]
    pub fn from_pretrained(device: usize, precision: Precision) -> Result<Self> {
        let gpu = Active::open(device, precision)?;
        Self::load_onto(gpu, &crate::download_model()?)
    }

    fn load_onto(device: <Active as Backend>::Device, dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::open(dir)?;
        let wts = Weights::open(&dir.join("model.safetensors"))?;
        Ok(Self {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            tokenizer,
            visual: OnceCell::new(),
            last_kept: Cell::new(0),
            masks: MaskPool::default(),
            backend: Active::load(device, &wts)?,
        })
    }

    /// Numeric mode the model was loaded with
    pub fn precision(&self) -> Precision {
        self.backend.precision()
    }

    fn check(&self, emb: &Embedding) -> Result<()> {
        ensure!(emb.model == self.id, "the embedding was encoded by a different Sam3 model");
        self.backend.bind()
    }

    /// Runs the vision backbone and both necks; the result serves any number of prompts
    pub fn encode(&self, img: &RgbImage) -> Result<Embedding> {
        self.backend.bind()?;
        let image = self.backend.encode(img)?;
        Ok(Embedding { model: self.id, width: img.width(), height: img.height(), image })
    }

    /// Detects every instance matching `text` and the box prompts (pass [`VISUAL_PROMPT`]
    /// for boxes alone), keeping those scoring above [`SCORE_THRESHOLD`] in query order.
    /// Each call runs the text encoder on `text`, except for [`VISUAL_PROMPT`], whose
    /// features are encoded on the first call and reused
    pub fn detect(&self, emb: &Embedding, text: &str, boxes: &[BoxPrompt]) -> Result<Detections> {
        self.check(emb)?;
        let b = &self.backend;
        let _r = range(c"detect");
        let (w, h) = emb.size();
        // one mask more than last time, since a box being dragged out tends to gain them
        let reserved = self.masks.reserve((self.last_kept.get() + 1) * w * h);
        let encode = || {
            let _r = range(c"text");
            b.encode_text(&self.tokenizer.tokenize(text)?)
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
        let out = b.detect(&emb.image, feats, boxes)?;
        let _r = range(c"postprocess");
        let DetectionScores { presence, logits, boxes } = b.detection_scores(&out)?;
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

        let queries: Vec<usize> = kept.iter().map(|(i, _)| *i).collect();
        let slots: Vec<usize> = (0..kept.len()).collect();
        let (mut reserved, mut masks) = (Some(reserved), MaskBuffer::default());
        let mut sink = MaskSink::new(&self.masks, &mut reserved, &mut masks, kept.len() * h * w, w * h, &slots);
        b.detection_masks(&out, &queries, (w, h), &mut sink)?;

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
        let _r = range(c"segment_points");
        let (w, h) = emb.size();
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
            let pass = self.backend.decode_points(&emb.image, &points, (w, h))?;
            let _r = range(c"postprocess");
            let mut sink = MaskSink::new(&self.masks, &mut reserved, &mut masks, prompts.len() * plane, plane, batch);
            let ious = self.backend.point_masks(pass, (w, h), &mut sink)?;
            ensure!(ious.len() == batch.len(), "{} IoUs for {} prompts", ious.len(), batch.len());
            for (&i, iou) in batch.iter().zip(ious) {
                scores[i] = iou;
            }
        }
        Ok(PointMasks { scores, masks })
    }
}
