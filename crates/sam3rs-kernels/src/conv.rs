//! Convolutions as GEMMs: the layout kernels around cuBLAS

use cuda_device::kernel;

use crate::common::{Bf16, activate, grid_stride, ldg4, math, pack_bf16, stg_u4};
use crate::norm::group_moments;

/// `ConvTranspose2d(k=2, s=2)`: `g [h * w, o * 4]` holds `x @ W` for `W [C, o, 2, 2]`;
/// scatters to `y [2h * 2w, o]` and adds `bias` (may be null)
#[kernel]
pub unsafe fn convt2x2_scatter(y: *mut f32, g: *const f32, bias: *const f32, h: i32, w: i32, o: i32) {
    let n = 4 * h as i64 * w as i64 * o as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let oc = (i % o as i64) as i32;
        let p = i / o as i64;
        let (yy, xx) = ((p / (2 * w) as i64) as i32, (p % (2 * w) as i64) as i32);
        let (sy, sx, ky, kx) = (yy >> 1, xx >> 1, yy & 1, xx & 1);
        let src = (sy as i64 * w as i64 + sx as i64) * o as i64 * 4 + (oc * 4 + ky * 2 + kx) as i64;
        let b = if bias.is_null() { 0.0 } else { unsafe { *bias.add(oc as usize) } };
        unsafe { *y.add(i as usize) = *g.add(src as usize) + b };
        i += stride;
    }
}

/// `x [h * w, c] -> xp [(h + 2) * (w + 2) + 2, c]` zero-padded; `xp` must be zeroed
/// first. With the padded row width, the input of every 3x3 tap is one contiguous
/// matrix at a fixed offset
#[kernel]
pub unsafe fn pad1(xp: *mut f32, x: *const f32, h: i32, w: i32, c: i32) {
    let n = h as i64 * w as i64 * c as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let ch = i % c as i64;
        let p = i / c as i64;
        let (yy, xx) = (p / w as i64, p % w as i64);
        let d = ((yy + 1) * (w + 2) as i64 + xx + 1) * c as i64 + ch;
        unsafe { *xp.add(d as usize) = *x.add(i as usize) };
        i += stride;
    }
}

/// Pixel `p` shifted by 3x3 tap `t` (offsets `t / 3 - 1`, `t % 3 - 1`), or `None` outside
/// the image
#[inline(always)]
fn tap_pixel(p: i64, t: i32, h: i32, w: i32) -> Option<(i32, i32)> {
    let yy = (p / w as i64) as i32 + t / 3 - 1;
    let xx = (p % w as i64) as i32 + t % 3 - 1;
    (yy >= 0 && yy < h && xx >= 0 && xx < w).then_some((yy, xx))
}

/// 3x3, padding 1 im2col to BF16 for output pixels `[p0, p0 + n)`
///
/// `out[p - p0, t * c + ic]` is `x` at pixel `p` shifted by tap `t = dy * 3 + dx`, zero
/// outside the image, so the conv is one GEMM with `K = 9c`. `c` must be a multiple of 8
#[kernel]
pub unsafe fn im2col3x3_bf16(out: *mut Bf16, x: *const f32, h: i32, w: i32, c: i32, p0: i64, n: i64) {
    let c8 = (c / 8) as i64;
    let total = n * 9 * c8;
    let (mut i, stride) = grid_stride();
    while i < total {
        let ch = (i % c8) as i32 * 8;
        let r = i / c8;
        let t = (r % 9) as i32;
        let p = p0 + r / 9;
        let mut v = [0u32; 4];
        if let Some((yy, xx)) = tap_pixel(p, t, h, w) {
            let src = unsafe { x.add(((yy as i64 * w as i64 + xx as i64) * c as i64 + ch as i64) as usize) };
            let (a, b) = unsafe { (ldg4(src), ldg4(src.add(4))) };
            v = [pack_bf16(a[0], a[1]), pack_bf16(a[2], a[3]), pack_bf16(b[0], b[1]), pack_bf16(b[2], b[3])];
        }
        unsafe { stg_u4(out.add((r * c as i64 + ch as i64) as usize), v) };
        i += stride;
    }
}

/// [`im2col3x3_bf16`] of `fine + up2(coarse)` without materializing that sum
///
/// The input at fine pixel `(y, x)` is `fine[y, x] + coarse[y / 2, x / 2]`, as
/// `upsample2_add` computes it. With `stats`, coarse first gets the group norm and
/// activation `group_norm_apply` would apply, so neither the coarse map's normalized copy
/// nor the upsampled sum is ever written. `h` and `w` are the fine map's (even) size;
/// `groups` must divide `c` into multiples of 8
#[kernel]
pub unsafe fn im2col3x3_up_bf16(
    out: *mut Bf16,
    fine: *const f32,
    coarse: *const f32,
    stats: *const f32,
    gw: *const f32,
    gb: *const f32,
    groups: i32,
    eps: f32,
    act: i32,
    h: i32,
    w: i32,
    c: i32,
    p0: i64,
    n: i64,
) {
    let (c8, cg) = ((c / 8) as i64, c / groups);
    let count = ((h / 2) as i64 * (w / 2) as i64 * cg as i64) as f32;
    let total = n * 9 * c8;
    let (mut i, stride) = grid_stride();
    while i < total {
        let ch = (i % c8) as i32 * 8;
        let r = i / c8;
        let t = (r % 9) as i32;
        let p = p0 + r / 9;
        let mut v = [0u32; 4];
        if let Some((yy, xx)) = tap_pixel(p, t, h, w) {
            let f = unsafe { fine.add(((yy as i64 * w as i64 + xx as i64) * c as i64 + ch as i64) as usize) };
            let q = ((yy >> 1) as i64 * (w >> 1) as i64 + (xx >> 1) as i64) * c as i64 + ch as i64;
            let q = unsafe { coarse.add(q as usize) };
            let (f0, f1, q0, q1) = unsafe { (ldg4(f), ldg4(f.add(4)), ldg4(q), ldg4(q.add(4))) };
            let a = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]];
            let mut u = [q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]];
            if !stats.is_null() {
                let (mean, var) = unsafe { group_moments(stats, ch / cg, count) };
                let mut k = 0;
                #[unroll]
                while k < 8 {
                    let (gwk, gbk) = unsafe { (*gw.add(ch as usize + k), *gb.add(ch as usize + k)) };
                    u[k] = activate((u[k] - mean) * math::rsqrtf(var + eps) * gwk + gbk, act);
                    k += 1;
                }
            }
            let s = |k: usize| a[k] + u[k];
            v = [pack_bf16(s(0), s(1)), pack_bf16(s(2), s(3)), pack_bf16(s(4), s(5)), pack_bf16(s(6), s(7))];
        }
        unsafe { stg_u4(out.add((r * c as i64 + ch as i64) as usize), v) };
        i += stride;
    }
}

/// `yp [h * (w + 2), o]` (a 3x3 conv over padded rows) `-> y [h * w, o]` plus `bias` (may
/// be null)
#[kernel]
pub unsafe fn crop_bias(y: *mut f32, yp: *const f32, bias: *const f32, h: i32, w: i32, o: i32) {
    let n = h as i64 * w as i64 * o as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let oc = i % o as i64;
        let p = i / o as i64;
        let (yy, xx) = (p / w as i64, p % w as i64);
        let b = if bias.is_null() { 0.0 } else { unsafe { *bias.add(oc as usize) } };
        unsafe { *y.add(i as usize) = *yp.add(((yy * (w + 2) as i64 + xx) * o as i64 + oc) as usize) + b };
        i += stride;
    }
}

/// `out [2h * 2w, c] = fine + nearest-upsampled coarse [h * w, c]`
#[kernel]
pub unsafe fn upsample2_add(out: *mut f32, fine: *const f32, coarse: *const f32, h: i32, w: i32, c: i32) {
    let n = 4 * h as i64 * w as i64 * c as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let ch = i % c as i64;
        let p = i / c as i64;
        let (y, x) = ((p / (2 * w) as i64) as i32, (p % (2 * w) as i64) as i32);
        let q = ((y >> 1) as i64 * w as i64 + (x >> 1) as i64) * c as i64 + ch;
        unsafe { *out.add(i as usize) = *fine.add(i as usize) + *coarse.add(q as usize) };
        i += stride;
    }
}

/// Non-overlapping 2x2 patches of channels-last `x [h * w, c] -> [(h / 2) * (w / 2), c * 4]`
/// in `(c, ky, kx)` order, so a stride-2 2x2 conv is one GEMM with the `[O, C, 2, 2]`
/// weight
#[kernel]
pub unsafe fn patchify2x2(out: *mut f32, x: *const f32, h: i32, w: i32, c: i32) {
    let (oh, ow) = (h / 2, w / 2);
    let n = oh as i64 * ow as i64 * c as i64 * 4;
    let (mut i, stride) = grid_stride();
    while i < n {
        let col = (i % (c * 4) as i64) as i32;
        let p = i / (c * 4) as i64;
        let (ch, ky, kx) = (col / 4, (col / 2) % 2, col % 2);
        let (y, xx) = ((p / ow as i64) as i32, (p % ow as i64) as i32);
        let s = ((2 * y + ky) as i64 * w as i64 + (2 * xx + kx) as i64) * c as i64 + ch as i64;
        unsafe { *out.add(i as usize) = *x.add(s as usize) };
        i += stride;
    }
}
