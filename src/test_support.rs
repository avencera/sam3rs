//! What the shared GPU suites in `stage_tests` and `video_tests` run on: the PyTorch
//! fixtures, error measures and tolerances, and [`TestHooks`], the stage-by-stage access
//! each backend gives them beyond [`Backend`]
//!
//! A backend joins both suites by implementing [`TestHooks`] in its own `test_hooks`
//! module. Its tolerances there are explicit per-backend values, built from the
//! constants here

use std::path::{Path, PathBuf};

use eyre::{Result, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::backend::Backend;
use crate::model::point::LOW_RES;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::sam3::Sam3;
use crate::tracker::{Selection, VideoTracker};

/// Stage tolerance in FP32 (rel L2 on the reference's own inputs): every stage is a
/// handful of FP32 GEMMs and reductions, which reorder sums but don't lose precision
pub(crate) const STAGE_REL_L2: f32 = 1e-4;
/// The memory attention over ~36k keys and the low-res mask logits sum over more terms
pub(crate) const LONG_REL_L2: f32 = 5e-4;
/// Backbone and neck stages in the reduced-precision modes: they are the image path's BF16
/// encoder, gated end to end by `tests/parity.rs`, and drift 1.4e-2 to 2.6e-2 from FP32 on
/// the video fixtures on CUDA
pub(crate) const REDUCED_BACKBONE_REL_L2: f32 = 4e-2;
/// Outputs whose every error is below this pass whatever their rel L2, which is noise on
/// values near zero (an absent object's IoU predictions are about 1e-15)
pub(crate) const NEGLIGIBLE_ABS: f32 = 1e-6;

/// A tracked frame whose top two candidate masks have predicted IoUs closer than this is
/// an argmax near-tie: reduced precision can pick the other candidate (FP32 picks 0.768
/// over 0.767 on horses frame 35, BF16 the reverse), so such a frame may fall below the
/// worst-frame limit, though not below NEAR_TIE_MIN_IOU, on at most NEAR_TIE_FRAMES
/// frames an object
pub(crate) const NEAR_TIE: f32 = 0.005;
pub(crate) const NEAR_TIE_MIN_IOU: f32 = 0.8;
pub(crate) const NEAR_TIE_FRAMES: usize = 2;

/// rel L2 limits of the video stage checks in one precision
pub(crate) struct Tolerances {
    /// Each stage after the necks
    pub(crate) stage: f32,
    /// The memory attention and the low-res mask logits
    pub(crate) long: f32,
    /// The backbone and the tracker neck
    pub(crate) backbone: f32,
}

/// The decoder outputs the video stage checks compare, downloaded
pub(crate) struct DecodedOutputs {
    /// Predicted IoU of all four mask tokens
    pub(crate) iou: Vec<f32>,
    /// The object score logit
    pub(crate) score: f32,
    /// Low-res logits of the decoded mask tokens
    pub(crate) low_res: Vec<f32>,
    /// Every output token `[tokens, 256]`
    pub(crate) tokens: Vec<f32>,
}

/// One backend's stages one at a time, for the shared suites: each takes the reference's
/// inputs where the suites feed them, and every result can be downloaded
///
/// Channels-last device tensors are `[hw, c]`; host CHW vectors are `[c, hw]`, the
/// fixtures' layout
pub(crate) trait TestHooks: Backend {
    /// A device tensor
    type Tensor;
    /// The backbone's output, prepared for the necks
    type Trunk;
    /// A neck's fine, mid and coarse levels
    type Fpn;
    /// The point decoder's image features
    type PointImage;
    /// The detector's image features
    type DetectorImage;
    /// What the backend's own end-to-end video checks keep across frames
    type E2eChecks: Default;

    /// Hole fills a stack of `[n, 288^2]` low-res masks on a GPU of its own
    fn fill_holes_on_gpu(logits: Vec<f32>) -> Result<Vec<f32>>;

    /// Waits for every queued GPU command
    fn sync(&self) -> Result<()>;

    /// Per kernel: its name, dispatches and GPU seconds since the last call, where the
    /// backend profiles kernels (Metal with `SAM3RS_METAL_PROFILE` set); empty otherwise
    fn take_profile(&self) -> Vec<(&'static str, usize, f64)>;

    /// The video stage limits in the model's precision
    fn tolerances(&self) -> Tolerances;

    fn upload(&self, v: Vec<f32>, shape: &[usize]) -> Result<Self::Tensor>;

    fn download(&self, t: &Self::Tensor) -> Result<Vec<f32>>;

    /// A channels-last tensor as host CHW
    fn chw(&self, t: &Self::Tensor) -> Result<Vec<f32>>;

    /// Host CHW `[c, hw]` as a channels-last tensor
    fn upload_chw(&self, v: Vec<f32>, c: usize) -> Result<Self::Tensor>;

    /// Preprocessed pixels `[3, 1008, 1008]` from the host
    fn upload_pixels(&self, v: Vec<f32>) -> Result<Self::Pixels>;

    fn download_pixels(&self, px: &Self::Pixels) -> Result<Vec<f32>>;

    /// The backbone alone, `[72^2, 1024]`
    fn trunk(&self, px: &Self::Pixels) -> Result<Self::Tensor>;

    fn neck_input(&self, trunk: Self::Tensor) -> Result<Self::Trunk>;

    /// The detector's neck, or the tracker's when `tracker` is set
    fn neck(&self, trunk: &Self::Trunk, tracker: bool) -> Result<Self::Fpn>;

    /// The fine, mid and coarse levels
    fn fpn_levels<'a>(&self, fpn: &'a Self::Fpn) -> [&'a Self::Tensor; 3];

    /// The point decoder's features from the tracker neck's levels
    fn point_image(&self, fpn: &Self::Fpn) -> Result<Self::PointImage>;

    /// The point decoder's features from the reference's
    fn point_image_from(&self, embed: Self::Tensor, s0: Self::Tensor, s1: Self::Tensor) -> Self::PointImage;

    /// The point image's conv_s0 and conv_s1 levels
    fn point_levels<'a>(&self, img: &'a Self::PointImage) -> [&'a Self::Tensor; 2];

    /// One decoder pass with masks for mask tokens 1 to 3, from `mask_input [288^2]` as
    /// the mask prompt when given
    fn point_decode(
        &self,
        img: &Self::PointImage,
        points: &[PointPrompt],
        size: (usize, usize),
        mask_input: Option<&[f32]>,
    ) -> Result<Self::Decoded>;

    fn decoded_outputs(&self, dec: &Self::Decoded) -> Result<DecodedOutputs>;

    /// A pass's masks after hole filling, thresholded at full size: `n * width * height`
    /// bytes
    fn full_masks(&self, dec: &Self::Decoded, n: usize, size: (usize, usize)) -> Result<Vec<u8>>;

    /// The detector's features from the detector neck's fine, mid and coarse levels
    fn detector_image(&self, fpn: [Self::Tensor; 3]) -> Result<Self::DetectorImage>;

    fn text_features(&self, text: &Self::Text) -> Result<Vec<f32>>;

    fn detect_from(&self, img: &Self::DetectorImage, text: &Self::Text, boxes: &[BoxPrompt]) -> Result<Self::Detected>;

    /// Mask logits of every query, `[200, 288^2]`
    fn mask_logits(&self, det: &Self::Detected) -> Result<Vec<f32>>;

    /// A video frame's tracker neck coarse level, conv_s0 and conv_s1 levels
    fn frame_levels<'a>(&self, frame: &'a Self::Frame) -> [&'a Self::Tensor; 3];

    /// The memory attention's position encoding of the frame's tokens, `[72^2, 256]`
    fn memory_src_pos(&self) -> Result<Vec<f32>>;

    /// The memory attention of a frame's `src [72^2, 256]` over one object's memory
    fn memory_attention(
        &self,
        src: &Self::Tensor,
        memory: &Self::Tensor,
        memory_pos: &Self::Tensor,
        ptr_tokens: usize,
    ) -> Result<Self::Tensor>;

    /// The object pointer projection
    fn object_pointer(&self, input: &Self::Tensor) -> Result<Self::Tensor>;

    /// The memory encoder over `pix_feat [72^2, 256]` and `mask [1008^2]`, channels-last
    fn memory_encoder(&self, pix_feat: &Self::Tensor, mask: &Self::Tensor) -> Result<Self::Tensor>;

    /// The no-object embedding upstream adds to an absent object's memory, `[64]`
    fn no_obj_spatial(&self) -> Result<Vec<f32>>;

    /// Each object's memory-attention inputs over the selection: the memory and its
    /// position encoding, assembled
    fn assembled_memory(&self, sel: &Selection<'_, Self::FrameMemory>) -> Result<Vec<(Self::Tensor, Self::Tensor)>>;

    /// The backend's own checks on tracked frame `t` of `frames`, before `tracker` tracks
    /// into it from `pixels`
    fn e2e_check(
        &self,
        checks: &mut Self::E2eChecks,
        tracker: &VideoTracker<'_>,
        pixels: &Self::Pixels,
        t: usize,
        frames: usize,
        failures: &mut Vec<String>,
    ) -> Result<()>;

    /// Fails when the backend's own end-to-end checks never ran where they should have
    fn e2e_finish(&self, checks: Self::E2eChecks) -> Result<()>;
}

/// A safetensors fixture and its JSON metadata
pub(crate) struct Fixture {
    mmap: Mmap,
    pub(crate) meta: Value,
}

impl Fixture {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (_, header) = SafeTensors::read_metadata(&mmap)?;
        let meta = serde_json::from_str(&header.metadata().as_ref().ok_or_else(|| eyre!("no metadata"))?["meta"])?;
        Ok(Self { mmap, meta })
    }

    fn st(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.st().is_ok_and(|st| st.tensor(name).is_ok())
    }

    /// The tensor as f32 values and its shape
    pub(crate) fn get(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let st = self.st()?;
        let t = st.tensor(name).map_err(|e| eyre!("fixture {name}: {e}"))?;
        let v = match t.dtype() {
            Dtype::F32 => bytemuck::pod_collect_to_vec(t.data()),
            Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
            d => return Err(eyre!("{name}: unsupported {d:?}")),
        };
        Ok((v, t.shape().to_vec()))
    }

    /// The tensor as f32 values
    pub(crate) fn values(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.get(name)?.0)
    }

    pub(crate) fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let st = self.st()?;
        Ok(st.tensor(name).map_err(|e| eyre!("fixture {name}: {e}"))?.data().to_vec())
    }

    pub(crate) fn frame(&self, t: usize) -> &Value {
        &self.meta["per_frame"][t]
    }

    /// Key prefix `f{t}.{name}.{k}` of the `nth` call to `name` on frame `t`
    pub(crate) fn call(&self, t: usize, name: &str, nth: usize) -> Option<String> {
        let calls = self.frame(t)["calls"].as_array()?;
        let k = calls.iter().enumerate().filter(|(_, c)| c.as_str() == Some(name)).nth(nth)?.0;
        Some(format!("f{t}.{name}.{k}"))
    }

    pub(crate) fn frames(&self) -> usize {
        self.meta["frames"].as_u64().unwrap() as usize
    }

    /// Frames in the whole clip, which sets the tracker's temporal scale even when the
    /// fixture dumped fewer
    pub(crate) fn video_frames(&self) -> usize {
        self.meta["video"]["frames"].as_u64().unwrap() as usize
    }

    pub(crate) fn size(&self) -> (usize, usize) {
        let v = &self.meta["video"];
        (v["width"].as_u64().unwrap() as usize, v["height"].as_u64().unwrap() as usize)
    }

    /// One positive click per object, in the video's pixels
    pub(crate) fn clicks(&self) -> Result<Vec<Vec<PointPrompt>>> {
        let (w, h) = self.size();
        self.meta["objects"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| {
                let p = &o["point"];
                let (x, y) = (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32);
                Ok(vec![PointPrompt::new(x * w as f32, y * h as f32, true)?])
            })
            .collect()
    }
}

/// rel L2 of `got` against `want`, printed with the max abs error; fails on a length
/// mismatch or a non-finite value
pub(crate) fn rel_l2(name: &str, got: &[f32], want: &[f32]) -> f32 {
    rel_l2_max(name, got, want).0
}

/// [`rel_l2`] and the max abs error
pub(crate) fn rel_l2_max(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    assert!(!got.is_empty(), "{name}: empty");
    assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite output");
    let (mut max_abs, mut err, mut norm) = (0f32, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        max_abs = max_abs.max((g - w).abs());
        err += ((g - w) as f64).powi(2);
        norm += (*w as f64).powi(2);
    }
    let rel = (err / norm.max(1e-30)).sqrt() as f32;
    println!("{name:>28}: max_abs {max_abs:.3e}  rel_l2 {rel:.3e}");
    (rel, max_abs)
}

/// Records a failure when `got` is further than `tol` from `want` in rel L2 and not
/// negligibly close
pub(crate) fn check(failures: &mut Vec<String>, name: &str, got: &[f32], want: &[f32], tol: f32) {
    let (rel, max_abs) = rel_l2_max(name, got, want);
    // a NaN reference fails too
    if rel.is_nan() || (rel > tol && max_abs >= NEGLIGIBLE_ABS) {
        failures.push(format!("{name}: rel_l2 {rel:.3e} > {tol:.0e}"));
    }
}

/// Prints the max and relative L2 error, for the diagnostic stage reports; NaNs count as
/// an infinite error
pub(crate) fn compare(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut ref_sq = 0f64;
    for (g, w) in got.iter().zip(want) {
        let d = if g.is_finite() { (g - w).abs() } else { f32::INFINITY };
        max_abs = max_abs.max(d);
        sum_sq += (d as f64).powi(2);
        ref_sq += (*w as f64).powi(2);
    }
    let rel = (sum_sq / ref_sq.max(1e-30)).sqrt();
    println!("{name:>12}: max_abs {max_abs:.3e}  rel_l2 {rel:.3e}");
}

/// Object `b` of a sequence-first `[n, objects, c]` tensor, as `[n, c]`
pub(crate) fn seq_object(v: &[f32], shape: &[usize], b: usize) -> Vec<f32> {
    let (n, objects, c) = (shape[0], shape[1], shape[2]);
    (0..n).flat_map(|i| v[(i * objects + b) * c..][..c].iter().copied()).collect()
}

/// Object `b` of a batch-first tensor `[objects, ...]`
pub(crate) fn batch_object(v: &[f32], shape: &[usize], b: usize) -> Vec<f32> {
    let n = v.len() / shape[0];
    v[b * n..][..n].to_vec()
}

/// Area of `a & b` over `a | b`, 1 when both are empty
pub(crate) fn iou(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "mask sizes differ");
    assert!(!a.is_empty(), "empty mask");
    let (mut inter, mut union) = (0usize, 0usize);
    for (&x, &y) in a.iter().zip(b) {
        inter += (x != 0 && y != 0) as usize;
        union += (x != 0 || y != 0) as usize;
    }
    if union == 0 { 1.0 } else { inter as f32 / union as f32 }
}

/// Mask `b` of `np.packbits(masks [objects, h, w] > 0, axis=-1)`
pub(crate) fn unpack(packed: &[u8], b: usize, (w, h): (usize, usize)) -> Vec<u8> {
    let row = w.div_ceil(8);
    let base = b * h * row;
    (0..h * w).map(|i| (packed[base + (i / w) * row + (i % w) / 8] >> (7 - (i % w) % 8)) & 1).collect()
}

/// The video fixture, `SAM3RS_VIDEO` or the horses clip's
pub(crate) fn video_fixture_path() -> PathBuf {
    std::env::var_os("SAM3RS_VIDEO")
        .map_or_else(|| "models/video-parity/horses-720p-64-o3-fp32.safetensors".into(), PathBuf::from)
}

/// The image stage fixture, `SAM3RS_STAGES` or the dogs image's
pub(crate) fn stage_fixture_path() -> PathBuf {
    std::env::var_os("SAM3RS_STAGES").map_or_else(|| "models/parity/dogs.safetensors".into(), PathBuf::from)
}

/// `SAM3RS_PRECISION`, FP32 by default
pub(crate) fn precision() -> Result<Precision> {
    std::env::var("SAM3RS_PRECISION").as_deref().unwrap_or("fp32").parse()
}

/// The model from `models/sam3` in [`precision`]
pub(crate) fn load_model() -> Result<Sam3> {
    Sam3::load("models/sam3".as_ref(), 0, precision()?)
}

/// The host flood fill the GPU fill replaced: background regions (logit <= 0),
/// 8-connected, of at most `max_area` pixels become foreground (+10). NaN is
/// foreground, as upstream's `mask <= mask_threshold`; an older fill let a NaN pixel
/// start a region, so its regions depended on scan order
pub(crate) fn fill_holes_host(logits: &mut [f32], side: usize, max_area: usize) {
    let mut seen = vec![false; logits.len()];
    let mut stack = Vec::new();
    let mut component = Vec::new();
    for start in 0..logits.len() {
        if seen[start] || logits[start] > 0.0 || logits[start].is_nan() {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        component.clear();
        while let Some(p) = stack.pop() {
            component.push(p);
            let (y, x) = ((p / side) as isize, (p % side) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= side as isize || nx >= side as isize {
                        continue;
                    }
                    let q = ny as usize * side + nx as usize;
                    if !seen[q] && logits[q] <= 0.0 {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        if component.len() <= max_area {
            for &p in &component {
                logits[p] = 10.0;
            }
        }
    }
}

/// Low-res logits with regions of many sizes and shapes: `masks` box-blurred noise masks
/// of growing radius with NaN pixels, then an all-background and an all-foreground mask
pub(crate) fn hole_fill_logits(masks: usize) -> Vec<f32> {
    let side = LOW_RES;
    let mut state = 0x9e37_79b9_u32;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32 - 0.5
    };
    let mut out = Vec::new();
    for m in 0..masks {
        let raw: Vec<f32> = (0..side * side).map(|_| noise()).collect();
        let r = 1 + m as isize;
        let last = side as isize - 1;
        let mut mask = vec![0f32; side * side];
        for y in 0..side as isize {
            for x in 0..side as isize {
                let mut sum = 0.0;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (yy, xx) = ((y + dy).clamp(0, last), (x + dx).clamp(0, last));
                        sum += raw[yy as usize * side + xx as usize];
                    }
                }
                mask[y as usize * side + x as usize] = sum + 0.05 * m as f32;
            }
        }
        for p in (m * 7919..side * side).step_by(9973) {
            mask[p] = f32::NAN;
        }
        out.extend(mask);
    }
    out.extend(vec![-1.0; side * side]);
    out.extend(vec![1.0; side * side]);
    out
}
