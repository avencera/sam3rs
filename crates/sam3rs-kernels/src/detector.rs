//! Text embedding, detector geometry and mask post-processing
//!
//! One thread per output element throughout. Index math is 32-bit: the host launches one
//! thread per output element and keeps every buffer under 2^32 elements (the largest, the
//! upsampled masks, are `n * 2560 * 1600`). Each kernel's sizes come from its slice
//! lengths where a dimension can be derived from them

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::math;

/// `out[t] = tok_table[ids[t]] + pos_table[t]` over `[ids.len(), dim]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn text_embed(mut out: DisjointSlice<f32>, tok_table: &[f32], pos_table: &[f32], ids: &[i32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let dim = out.len() as u32 / ids.len() as u32;
    let Some(mut out) = out.element_thread32(t) else { return };
    let (row, d) = (i / dim, i % dim);
    // a token id outside the table is a bounds-check trap instead of a stray read
    out.write(tok_table[(ids[row as usize] as u32 * dim + d) as usize] + pos_table[i as usize]);
}

/// torchvision's bilinear sample of channel `ch` at `(y, x)` of a channels-last map
#[inline(always)]
fn roi_bilinear(f: &[f32], h: i32, w: i32, c: u32, ch: u32, mut y: f32, mut x: f32) -> f32 {
    if y < -1.0 || y > h as f32 || x < -1.0 || x > w as f32 {
        return 0.0;
    }
    if y <= 0.0 {
        y = 0.0;
    }
    if x <= 0.0 {
        x = 0.0;
    }

    let (mut y0, mut x0) = (y as i32, x as i32);
    let (y1, x1);
    if y0 >= h - 1 {
        y0 = h - 1;
        y1 = y0;
        y = y0 as f32;
    } else {
        y1 = y0 + 1;
    }
    if x0 >= w - 1 {
        x0 = w - 1;
        x1 = x0;
        x = x0 as f32;
    } else {
        x1 = x0 + 1;
    }

    let (ly, lx) = (y - y0 as f32, x - x0 as f32);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    // both coordinates are clamped to [0, h - 1] x [0, w - 1] above
    let at = |yy: i32, xx: i32| f[((yy as u32 * w as u32 + xx as u32) * c + ch) as usize];
    let (v1, v2, v3, v4) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
    hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4
}

/// torchvision `roi_align` (`aligned=False`, `spatial_scale=1`, adaptive sampling ratio)
/// over a channels-last feature map
///
/// `out [n_boxes, c * pool * pool]` in `(c, ph, pw)` order, the flattening the 7x7
/// pooling conv weight expects. `boxes` are xyxy in feature pixels, `[n_boxes, 4]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn roi_align(mut out: DisjointSlice<f32>, feat: &[f32], boxes: &[f32], h: u32, w: u32, c: u32, pool: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let pw = i % pool;
    let ph = (i / pool) % pool;
    let ch = (i / (pool * pool)) % c;
    let b = (i / (pool * pool * c)) as usize;
    let (sx, sy) = (boxes[b * 4], boxes[b * 4 + 1]);
    let rw = math::fmaxf(boxes[b * 4 + 2] - sx, 1.0);
    let rh = math::fmaxf(boxes[b * 4 + 3] - sy, 1.0);
    let (bh, bw) = (rh / pool as f32, rw / pool as f32);
    let gh = math::ceilf(rh / pool as f32) as i32;
    let gw = math::ceilf(rw / pool as f32) as i32;
    let count = math::fmaxf((gh * gw) as f32, 1.0);
    let mut acc = 0.0;
    for iy in 0..gh {
        let y = sy + ph as f32 * bh + (iy as f32 + 0.5) * bh / gh as f32;
        for ix in 0..gw {
            let x = sx + pw as f32 * bw + (ix as f32 + 0.5) * bw / gw as f32;
            acc += roi_bilinear(feat, h as i32, w as i32, c, ch, y, x);
        }
    }
    out.write(acc / count);
}

/// Log-scaled distances from each grid coordinate to a box's two edges along one axis
///
/// `out[(q * side + i) * 2 + e]` for reference boxes cxcywh `[nq, 4]`; axis 0 is x, 1 is y
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn rpb_deltas(mut out: DisjointSlice<f32>, boxes: &[f32], side: u32, axis: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let e = i % 2;
    let pos = (i / 2) % side;
    let q = (i / (2 * side)) as usize;
    let (center, size) = (boxes[q * 4 + axis as usize], boxes[q * 4 + axis as usize + 2]);
    let edge = if e == 0 { center - 0.5 * size } else { center + 0.5 * size };
    let d = (pos as f32 / side as f32 - edge) * 8.0;
    let s = (d > 0.0) as i32 as f32 - (d < 0.0) as i32 as f32;
    out.write(s * math::log2f(d.abs() + 1.0) / 3.0);
}

/// Attention bias `[heads, 1 + nq, side * side]`: row 0 (the presence token) is zero and
/// row `1 + q` is `dy[h, q, y] + dx[h, q, x]`, with `dx` and `dy` `[heads, nq * side]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn rpb_combine(mut bias: DisjointSlice<f32>, dx: &[f32], dy: &[f32], nq: u32, side: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut bias) = bias.element_thread32(t) else { return };
    let hw = side * side;
    let p = i % hw;
    let row = (i / hw) % (nq + 1);
    let h = i / (hw * (nq + 1));
    let v = if row == 0 {
        0.0
    } else {
        let base = (h * nq + row - 1) * side;
        dy[(base + p / side) as usize] + dx[(base + p % side) as usize]
    };
    bias.write(v);
}

/// `gen_sineembed_for_position` for cxcywh boxes `[n, 4] -> [n, 512]`, in y, x, w, h
/// blocks of 128
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn box_sine_embed(mut out: DisjointSlice<f32>, boxes: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let (f, block) = (i % 128, (i / 128) % 4);
    let q = i / 512;
    // block order y, x, w, h maps to box components 1, 0, 2, 3
    let comp = match block {
        0 => 1,
        1 => 0,
        b => b,
    };
    let v = boxes[(q * 4 + comp) as usize] * core::f32::consts::TAU;
    let dim_t = math::powf(10000.0, (2 * (f / 2)) as f32 / 128.0);
    let a = v / dim_t;
    out.write(if f % 2 == 0 { math::sinf(a) } else { math::cosf(a) });
}

#[inline(always)]
fn inverse_sigmoid(x: f32) -> f32 {
    let x = math::fminf(math::fmaxf(x, 0.0), 1.0);
    let (x1, x2) = (math::fmaxf(x, 1e-3), math::fmaxf(1.0 - x, 1e-3));
    math::logf(x1 / x2)
}

/// `out = sigmoid(delta + inverse_sigmoid(reference))`, elementwise over `[n, 4]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn box_refine(mut out: DisjointSlice<f32>, reference: &[f32], delta: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    if let Some(mut out) = out.element_thread32(t) {
        let v = delta[i] + inverse_sigmoid(reference[i]);
        out.write(1.0 / (1.0 + math::expf(-v)));
    }
}

/// Bilinear resize (`align_corners=False`, no antialias) of planes
/// `[n, ih, iw] -> [n, oh, ow]`, with `n` implied by the output length
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn resize_bilinear(mut out: DisjointSlice<f32>, input: &[f32], ih: u32, iw: u32, oh: u32, ow: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let (sh, sw) = (ih as f32 / oh as f32, iw as f32 / ow as f32);
    let (x, y, p) = (i % ow, (i / ow) % oh, i / (ow * oh));
    let fy = math::fmaxf((y as f32 + 0.5) * sh - 0.5, 0.0);
    let fx = math::fmaxf((x as f32 + 0.5) * sw - 0.5, 0.0);
    let (y0, x0) = (fy as u32, fx as u32);
    let (y1, x1) = (y0 + (y0 < ih - 1) as u32, x0 + (x0 < iw - 1) as u32);
    let (ly, lx) = (fy - y0 as f32, fx - x0 as f32);
    let plane = (p * ih * iw) as usize;
    let at = |yy: u32, xx: u32| input[plane + (yy * iw + xx) as usize];
    let v = (1.0 - ly) * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1)) + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
    out.write(v);
}
