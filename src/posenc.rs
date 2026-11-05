//! Input-independent position encodings, sine and RoPE, computed on the host in f32 the
//! way PyTorch does (f32 products and divisions, then sin/cos) for every backend to upload

use std::f64::consts::PI;

use crate::model::tracker::{self, ROPE_THETA};
use crate::model::vision::{GRID, HEAD_DIM, window_to_raster};

/// `10000 ** (2 * (i // 2) / n)` for i in 0..n
fn dim_t(n: usize) -> Vec<f32> {
    (0..n).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / n as f32)).collect()
}

/// Interleaved sin/cos features of one scaled coordinate
fn sine(v: f32, dims: &[f32], out: &mut [f32]) {
    for (i, d) in dims.iter().enumerate() {
        let a = (v / d) as f64;
        out[i] = if i % 2 == 0 { a.sin() as f32 } else { a.cos() as f32 };
    }
}

/// `PositionEmbeddingSine(feats, normalize=True)` over a `side` x `side` grid,
/// channels-last `[side * side, feats]` with the y features first
pub fn sine_2d(side: usize, feats: usize) -> Vec<f32> {
    let half = feats / 2;
    let dims = dim_t(half);
    let scale = (2.0 * PI) as f32;
    let norm = |i: usize| (i + 1) as f32 / (side as f32 + 1e-6) * scale;
    let mut out = vec![0f32; side * side * 2 * half];
    for y in 0..side {
        for x in 0..side {
            let row = &mut out[(y * side + x) * 2 * half..][..2 * half];
            let (py, px) = row.split_at_mut(half);
            sine(norm(y), &dims, py);
            sine(norm(x), &dims, px);
        }
    }
    out
}

/// `PositionEmbeddingSine.encode_boxes` for one normalized cxcywh box: 128 y features,
/// 128 x features, then h and w
pub fn box_encoding(b: [f32; 4]) -> Vec<f32> {
    let dims = dim_t(128);
    let scale = (2.0 * PI) as f32;
    let mut out = vec![0f32; 258];
    sine(b[1] * scale, &dims, &mut out[..128]);
    sine(b[0] * scale, &dims, &mut out[128..256]);
    out[256] = b[3];
    out[257] = b[2];
    out
}

/// Axial RoPE tables `(cos, sin)`, each `[side * side, HEAD_DIM / 2]`, for the vision
/// backbone's `side` x `side` token grid, rows in raster order, or in window-major order
/// with `window_major`
///
/// Matches `compute_axial_cis`: the first 16 complex pairs rotate by x, the last 16 by y,
/// and positions are scaled by `scale` (1/3 for global blocks, interpolated from the
/// 24x24 pretraining grid). Angles are formed in f32 like PyTorch before cos/sin
pub fn vision_rope(side: usize, scale: f32, window_major: bool) -> (Vec<f32>, Vec<f32>) {
    let quarter = HEAD_DIM / 4;
    let freqs: Vec<f32> = (0..quarter).map(|i| 1.0 / 10000f32.powf((4 * i) as f32 / HEAD_DIM as f32)).collect();
    let half = HEAD_DIM / 2;
    let mut cos = vec![0f32; side * side * half];
    let mut sin = vec![0f32; side * side * half];
    for row in 0..side * side {
        let tok = if window_major { window_to_raster(row) } else { row };
        let tx = (tok % side) as f32 * scale;
        let ty = (tok / side) as f32 * scale;
        for p in 0..half {
            let angle = if p < quarter { tx * freqs[p] } else { ty * freqs[p - quarter] };
            cos[row * half + p] = (angle as f64).cos() as f32;
            sin[row * half + p] = (angle as f64).sin() as f32;
        }
    }
    (cos, sin)
}

/// Axial RoPE angles `(cos, sin)`, each `[5184, 128]`, of the memory attention's 72x72
/// grid at head dim 256: pairs 0 to 63 rotate with the column, 64 to 127 with the row
/// (`compute_axial_cis`)
pub fn memory_rope() -> (Vec<f32>, Vec<f32>) {
    let quarter = tracker::D / 4;
    // f32 as upstream computes them: `1 / theta ** (4i / dim)`, then position * freq
    let freqs: Vec<f32> = (0..quarter).map(|i| 1.0 / ROPE_THETA.powf((4 * i) as f32 / tracker::D as f32)).collect();
    let n = tracker::TOKENS * tracker::D / 2;
    let (mut cos, mut sin) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for t in 0..tracker::TOKENS {
        let (x, y) = ((t % GRID) as f32, (t / GRID) as f32);
        for (pos, f) in freqs.iter().map(|f| (x, f)).chain(freqs.iter().map(|f| (y, f))) {
            let angle = (pos * f) as f64;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// `0.1 * pos` of the tracker neck's 72x72 level `[5184, 256]`, which the memory
/// attention adds to its input once
pub fn memory_src_pos() -> Vec<f32> {
    // f32 products, as `output + 0.1 * src_pos` rounds them
    sine_2d(GRID, tracker::D).into_iter().map(|v| 0.1 * v).collect()
}

/// `get_1d_sine_pe(pos / (max_abs_pos - 1), 256)` per object pointer position, `[n, 256]`,
/// before the tracker's `obj_ptr_tpos_proj`
pub fn pointer_sine(positions: &[usize], max_abs_pos: usize) -> Vec<f32> {
    let t_max = max_abs_pos.saturating_sub(1).max(1) as f32;
    let half = tracker::D / 2;
    let dims: Vec<f32> = (0..half).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / half as f32)).collect();
    let mut pe = Vec::with_capacity(positions.len() * tracker::D);
    for &p in positions {
        let v = p as f32 / t_max;
        pe.extend(dims.iter().map(|d| ((v / d) as f64).sin() as f32));
        pe.extend(dims.iter().map(|d| ((v / d) as f64).cos() as f32));
    }
    pe
}
