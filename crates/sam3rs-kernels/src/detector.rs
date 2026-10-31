//! Text embedding, detector geometry and mask post-processing

use cuda_device::kernel;
use cuda_device::thread::{blockDim_x, blockIdx_x, gridDim_x, threadIdx_x};

use crate::common::{grid_stride, math};

/// `out[t] = tok_table[ids[t]] + pos_table[t]`
#[kernel]
pub unsafe fn text_embed(
    out: *mut f32,
    tok_table: *const f32,
    pos_table: *const f32,
    ids: *const i32,
    len: i32,
    dim: i32,
) {
    let n = len as i64 * dim as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let (d, t) = (i % dim as i64, i / dim as i64);
        unsafe {
            let tok = *ids.add(t as usize) as i64;
            *out.add(i as usize) =
                *tok_table.add((tok * dim as i64 + d) as usize) + *pos_table.add((t * dim as i64 + d) as usize);
        }
        i += stride;
    }
}

/// torchvision's bilinear sample of channel `ch` at `(y, x)` of a channels-last map
#[inline(always)]
unsafe fn roi_bilinear(f: *const f32, h: i32, w: i32, c: i32, ch: i32, mut y: f32, mut x: f32) -> f32 {
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
    let at = |yy: i32, xx: i32| unsafe { *f.add(((yy as i64 * w as i64 + xx as i64) * c as i64 + ch as i64) as usize) };
    let (v1, v2, v3, v4) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
    hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4
}

/// torchvision `roi_align` (`aligned=False`, `spatial_scale=1`, adaptive sampling ratio)
/// over a channels-last feature map
///
/// `out [n_boxes, c * pool * pool]` in `(c, ph, pw)` order, the flattening the 7x7
/// pooling conv weight expects. `boxes` are xyxy in feature pixels
#[kernel]
pub unsafe fn roi_align(
    out: *mut f32,
    feat: *const f32,
    boxes: *const f32,
    n_boxes: i32,
    h: i32,
    w: i32,
    c: i32,
    pool: i32,
) {
    let n = n_boxes as i64 * c as i64 * pool as i64 * pool as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let pw = (i % pool as i64) as i32;
        let ph = ((i / pool as i64) % pool as i64) as i32;
        let ch = ((i / (pool * pool) as i64) % c as i64) as i32;
        let b = (i / (pool as i64 * pool as i64 * c as i64)) as i32;
        let bx = unsafe { boxes.add(b as usize * 4) };
        let (sx, sy) = unsafe { (*bx, *bx.add(1)) };
        let rw = math::fmaxf(unsafe { *bx.add(2) } - sx, 1.0);
        let rh = math::fmaxf(unsafe { *bx.add(3) } - sy, 1.0);
        let (bh, bw) = (rh / pool as f32, rw / pool as f32);
        let gh = math::ceilf(rh / pool as f32) as i32;
        let gw = math::ceilf(rw / pool as f32) as i32;
        let count = math::fmaxf((gh * gw) as f32, 1.0);
        let mut acc = 0.0;
        for iy in 0..gh {
            let y = sy + ph as f32 * bh + (iy as f32 + 0.5) * bh / gh as f32;
            for ix in 0..gw {
                let x = sx + pw as f32 * bw + (ix as f32 + 0.5) * bw / gw as f32;
                acc += unsafe { roi_bilinear(feat, h, w, c, ch, y, x) };
            }
        }
        unsafe { *out.add(i as usize) = acc / count };
        i += stride;
    }
}

/// Log-scaled distances from each grid coordinate to a box's two edges along one axis
///
/// `out[(q * side + i) * 2 + e]` for reference boxes cxcywh `[nq, 4]`; axis 0 is x, 1 is y
#[kernel]
pub unsafe fn rpb_deltas(out: *mut f32, boxes: *const f32, nq: i32, side: i32, axis: i32) {
    let n = nq as i64 * side as i64 * 2;
    let (mut i, stride) = grid_stride();
    while i < n {
        let e = (i % 2) as i32;
        let pos = ((i / 2) % side as i64) as i32;
        let q = (i / (2 * side) as i64) as i32;
        let b = unsafe { boxes.add(q as usize * 4) };
        let (center, size) = unsafe { (*b.add(axis as usize), *b.add(axis as usize + 2)) };
        let edge = if e == 0 { center - 0.5 * size } else { center + 0.5 * size };
        let d = (pos as f32 / side as f32 - edge) * 8.0;
        let s = (d > 0.0) as i32 as f32 - (d < 0.0) as i32 as f32;
        unsafe { *out.add(i as usize) = s * math::log2f(d.abs() + 1.0) / 3.0 };
        i += stride;
    }
}

/// Attention bias `[heads, 1 + nq, side * side]`: row 0 (the presence token) is zero and
/// row `1 + q` is `dy[h, q, y] + dx[h, q, x]`
#[kernel]
pub unsafe fn rpb_combine(bias: *mut f32, dx: *const f32, dy: *const f32, nq: i32, side: i32, heads: i32) {
    let hw = side as i64 * side as i64;
    let n = heads as i64 * (nq + 1) as i64 * hw;
    let (mut i, stride) = grid_stride();
    while i < n {
        let p = (i % hw) as i32;
        let row = ((i / hw) % (nq + 1) as i64) as i32;
        let h = (i / (hw * (nq + 1) as i64)) as i32;
        let v = if row == 0 {
            0.0
        } else {
            let (q, y, x) = (row - 1, p / side, p % side);
            let base = (h as i64 * nq as i64 + q as i64) * side as i64;
            unsafe { *dy.add((base + y as i64) as usize) + *dx.add((base + x as i64) as usize) }
        };
        unsafe { *bias.add(i as usize) = v };
        i += stride;
    }
}

/// `gen_sineembed_for_position` for cxcywh boxes `[n, 4] -> [n, 512]`, in y, x, w, h
/// blocks of 128
#[kernel]
pub unsafe fn box_sine_embed(out: *mut f32, boxes: *const f32, n: i32) {
    let total = n as i64 * 512;
    let (mut i, stride) = grid_stride();
    while i < total {
        let (f, block) = ((i % 128) as i32, ((i / 128) % 4) as i32);
        let q = (i / 512) as i32;
        // block order y, x, w, h maps to box components 1, 0, 2, 3
        let comp = match block {
            0 => 1,
            1 => 0,
            b => b,
        };
        let v = unsafe { *boxes.add((q * 4 + comp) as usize) } * core::f32::consts::TAU;
        let dim_t = math::powf(10000.0, (2 * (f / 2)) as f32 / 128.0);
        let a = v / dim_t;
        unsafe { *out.add(i as usize) = if f % 2 == 0 { math::sinf(a) } else { math::cosf(a) } };
        i += stride;
    }
}

#[inline(always)]
fn inverse_sigmoid(x: f32) -> f32 {
    let x = math::fminf(math::fmaxf(x, 0.0), 1.0);
    let (x1, x2) = (math::fmaxf(x, 1e-3), math::fmaxf(1.0 - x, 1e-3));
    math::logf(x1 / x2)
}

/// `out = sigmoid(delta + inverse_sigmoid(ref))`, elementwise over `[n, 4]`
#[kernel]
pub unsafe fn box_refine(out: *mut f32, reference: *const f32, delta: *const f32, n: i32) {
    let stride = (gridDim_x() * blockDim_x()) as i32;
    let mut i = (blockIdx_x() * blockDim_x() + threadIdx_x()) as i32;
    while i < n * 4 {
        let v = unsafe { *delta.add(i as usize) + inverse_sigmoid(*reference.add(i as usize)) };
        unsafe { *out.add(i as usize) = 1.0 / (1.0 + math::expf(-v)) };
        i += stride;
    }
}

/// Bilinear resize (`align_corners=False`, no antialias) of `n` planes
/// `[n, ih, iw] -> [n, oh, ow]`
#[kernel]
pub unsafe fn resize_bilinear(out: *mut f32, input: *const f32, n: i32, ih: i32, iw: i32, oh: i32, ow: i32) {
    let total = n as i64 * oh as i64 * ow as i64;
    let (sh, sw) = (ih as f32 / oh as f32, iw as f32 / ow as f32);
    let (mut i, stride) = grid_stride();
    while i < total {
        let (x, y) = ((i % ow as i64) as i32, ((i / ow as i64) % oh as i64) as i32);
        let p = i / (ow as i64 * oh as i64);
        let fy = math::fmaxf((y as f32 + 0.5) * sh - 0.5, 0.0);
        let fx = math::fmaxf((x as f32 + 0.5) * sw - 0.5, 0.0);
        let (y0, x0) = (fy as i32, fx as i32);
        let (y1, x1) = (y0 + (y0 < ih - 1) as i32, x0 + (x0 < iw - 1) as i32);
        let (ly, lx) = (fy - y0 as f32, fx - x0 as f32);
        let src = unsafe { input.add((p * ih as i64 * iw as i64) as usize) };
        let at = |yy: i32, xx: i32| unsafe { *src.add((yy * iw + xx) as usize) };
        unsafe {
            *out.add(i as usize) = (1.0 - ly) * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
        }
        i += stride;
    }
}
