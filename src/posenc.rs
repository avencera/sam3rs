//! Input-independent sine position encodings, computed on the host in f32 the way
//! PyTorch does (f32 products and divisions, then sin/cos)

use std::f64::consts::PI;

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
