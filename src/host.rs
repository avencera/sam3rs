//! Host-side computations every backend shares: the point prompt encoding, weight folds
//! made once at load, box prompt inputs and small decisions over downloaded scores. None
//! of it touches a GPU; each backend uploads the results in its own types

use std::f64::consts::PI;

use eyre::{Result, ensure};

use crate::model::point::D;
use crate::model::tracker::{self, MEM_D, NUM_MASKMEM};
use crate::model::vision::{GRID, IMAGE_SIZE};
use crate::posenc::{box_encoding, sine_2d};
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::weights::Weights;

/// First index of the maximum, like `torch.argmax`
pub(crate) fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Random Fourier features of normalized (x, y) coords in [0, 1]
fn fourier(gaussian: &[f32], x: f32, y: f32, out: &mut [f32]) {
    let (cx, cy) = (2.0 * x - 1.0, 2.0 * y - 1.0);
    let n = gaussian.len() / 2;
    for i in 0..n {
        let v = (cx * gaussian[i] + cy * gaussian[n + i]) * (2.0 * PI) as f32;
        out[i] = (v as f64).sin() as f32;
        out[n + i] = (v as f64).cos() as f32;
    }
}

/// The point decoder's prompt encoder weights that run on the host: the random Fourier
/// features of point coordinates and the point label embeddings
pub(crate) struct PointEncoding {
    gaussian: Vec<f32>,
    point_embed: Vec<f32>,
    not_a_point: Vec<f32>,
}

impl PointEncoding {
    pub(crate) fn load(wts: &Weights) -> Result<Self> {
        let pe = "tracker_model.prompt_encoder";
        Ok(Self {
            gaussian: wts.host(&format!("{pe}.shared_embedding.positional_embedding"))?.0,
            point_embed: wts.host(&format!("{pe}.point_embed.weight"))?.0,
            not_a_point: wts.host(&format!("{pe}.not_a_point_embed.weight"))?.0,
        })
    }

    /// The dense positional encoding `[72^2, 256]` of the image embedding's pixel centers
    pub(crate) fn dense_pe(&self) -> Vec<f32> {
        let mut dense = vec![0f32; GRID * GRID * D];
        for y in 0..GRID {
            for x in 0..GRID {
                let (fx, fy) = ((x as f32 + 0.5) / GRID as f32, (y as f32 + 0.5) / GRID as f32);
                fourier(&self.gaussian, fx, fy, &mut dense[(y * GRID + x) * D..][..D]);
            }
        }
        dense
    }

    /// Sparse prompt tokens: one per point plus the padding point. Without points (a
    /// tracked frame) upstream still passes one point labelled -1 before the padding, so
    /// there are two padding tokens
    pub(crate) fn sparse_tokens(&self, points: &[PointPrompt], width: usize, height: usize) -> Vec<f32> {
        if points.is_empty() {
            return [self.not_a_point.as_slice(), self.not_a_point.as_slice()].concat();
        }
        let n = points.len() + 1;
        let mut out = vec![0f32; n * D];
        let size = IMAGE_SIZE as f32;
        for (i, p) in points.iter().enumerate() {
            // pixels -> 1008 input frame, shifted to the pixel center, normalized back
            let x = (p.x() / width as f32 * size + 0.5) / size;
            let y = (p.y() / height as f32 * size + 0.5) / size;
            let row = &mut out[i * D..][..D];
            fourier(&self.gaussian, x, y, row);
            let label = p.positive() as usize;
            for (v, e) in row.iter_mut().zip(&self.point_embed[label * D..][..D]) {
                *v += e;
            }
        }
        out[(n - 1) * D..].copy_from_slice(&self.not_a_point);
        out
    }
}

/// The host inputs of the detector's geometry encoder for `boxes`, each row per box
pub(crate) struct BoxInputs {
    /// Normalized cxcywh `[n, 4]`
    pub(crate) cxcywh: Vec<f32>,
    /// Corners on the 72x72 feature grid `[n, 4]`, for ROI pooling
    pub(crate) grid_xyxy: Vec<f32>,
    /// Sine encodings `[n, 258]`
    pub(crate) encoding: Vec<f32>,
    /// 1 for a positive box, 0 for a negative one
    pub(crate) labels: Vec<i32>,
}

impl BoxInputs {
    pub(crate) fn new(boxes: &[BoxPrompt]) -> Self {
        let side = GRID as f32;
        let grid_xyxy = boxes
            .iter()
            .flat_map(|b| {
                let [cx, cy, w, h] = b.cxcywh();
                [(cx - 0.5 * w) * side, (cy - 0.5 * h) * side, (cx + 0.5 * w) * side, (cy + 0.5 * h) * side]
            })
            .collect();
        Self {
            cxcywh: boxes.iter().flat_map(BoxPrompt::cxcywh).collect(),
            grid_xyxy,
            encoding: boxes.iter().flat_map(|b| box_encoding(b.cxcywh())).collect(),
            labels: boxes.iter().map(|b| i32::from(b.positive())).collect(),
        }
    }
}

/// `o_proj(v_proj(.))` of a memory cross-attention as one 64 -> 256 linear, `Wo Wv`
/// `[256, 64]` and `Wo bv + bo` `[256]`, in f64 sums rounded once
pub(crate) fn fold_value_out(wts: &Weights, prefix: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    let d = tracker::D;
    let (wv, sv) = wts.host(&format!("{prefix}.v_proj.weight"))?;
    let (bv, _) = wts.host(&format!("{prefix}.v_proj.bias"))?;
    let (wo, so) = wts.host(&format!("{prefix}.o_proj.weight"))?;
    let (bo, _) = wts.host(&format!("{prefix}.o_proj.bias"))?;
    ensure!(sv == [d, MEM_D] && so == [d, d], "{prefix}: v_proj {sv:?}, o_proj {so:?}");
    let mut w = vec![0f32; d * MEM_D];
    let mut b = vec![0f32; d];
    for i in 0..d {
        let row = &wo[i * d..(i + 1) * d];
        for j in 0..MEM_D {
            w[i * MEM_D + j] =
                row.iter().enumerate().map(|(k, &x)| x as f64 * wv[k * MEM_D + j] as f64).sum::<f64>() as f32;
        }
        b[i] = (row.iter().zip(&bv).map(|(&x, &y)| x as f64 * y as f64).sum::<f64>() + bo[i] as f64) as f32;
    }
    Ok((w, b))
}

/// A memory fuser block's `pointwise_conv2` with its layer scale folded in: `gamma *
/// (x W2^T + b2)` as `(gamma W2)` and `gamma b2`, one rounding apart. Returns the weight,
/// its shape and the bias
pub(crate) fn fold_layer_scale(wts: &Weights, block: &str) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>)> {
    let (gamma, _) = wts.host(&format!("{block}.scale"))?;
    let (mut w2, shape) = wts.host(&format!("{block}.pointwise_conv2.weight"))?;
    let (mut b2, _) = wts.host(&format!("{block}.pointwise_conv2.bias"))?;
    ensure!(gamma.len() == shape[0], "{block}: {} scales for {shape:?}", gamma.len());
    for (o, g) in gamma.iter().enumerate() {
        w2[o * shape[1]..][..shape[1]].iter_mut().for_each(|v| *v *= g);
        b2[o] *= g;
    }
    Ok((w2, shape, b2))
}

/// `maskmem_pos_enc + maskmem_tpos_enc[i]` per temporal index i, `[5184, 64]` each
pub(crate) fn memory_positions(wts: &Weights) -> Result<Vec<Vec<f32>>> {
    let (tpos, _) = wts.host("tracker_model.memory_temporal_positional_encoding")?;
    let sine = sine_2d(GRID, MEM_D);
    let positions: Vec<Vec<f32>> =
        tpos.chunks(MEM_D).map(|t| sine.iter().enumerate().map(|(i, v)| v + t[i % MEM_D]).collect()).collect();
    ensure!(positions.len() == NUM_MASKMEM, "{} temporal encodings, expected {NUM_MASKMEM}", positions.len());
    Ok(positions)
}
