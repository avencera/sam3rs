//! Elementwise kernels

use cuda_device::kernel;

use crate::common::{Bf16, Elem, activate, grid_stride, math};

/// `x[r, c] = act(x[r, c] + bias[c])`; `bias` may be null
#[kernel]
pub unsafe fn bias_act(x: *mut f32, bias: *const f32, rows: i64, cols: i32, act: i32) {
    let n = rows * cols as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let mut v = unsafe { *x.add(i as usize) };
        if !bias.is_null() {
            v += unsafe { *bias.add((i % cols as i64) as usize) };
        }
        unsafe { *x.add(i as usize) = activate(v, act) };
        i += stride;
    }
}

/// `out = a + b`
#[kernel]
pub unsafe fn add(out: *mut f32, a: *const f32, b: *const f32, n: i64) {
    let (mut i, stride) = grid_stride();
    while i < n {
        unsafe { *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize) };
        i += stride;
    }
}

/// `out[r, c] = a[r, c] + b[c]`, `b` broadcast over rows
#[kernel]
pub unsafe fn add_row(out: *mut f32, a: *const f32, b: *const f32, rows: i64, cols: i32) {
    let n = rows * cols as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        unsafe { *out.add(i as usize) = *a.add(i as usize) + *b.add((i % cols as i64) as usize) };
        i += stride;
    }
}

/// `x = clamp(x, lo, hi)` in place
#[kernel]
pub unsafe fn clamp(x: *mut f32, n: i64, lo: f32, hi: f32) {
    let (mut i, stride) = grid_stride();
    while i < n {
        unsafe { *x.add(i as usize) = math::fminf(math::fmaxf(*x.add(i as usize), lo), hi) };
        i += stride;
    }
}

/// `mask[i] = x[i] > threshold`
#[kernel]
pub unsafe fn threshold_u8(mask: *mut u8, x: *const f32, n: i64, threshold: f32) {
    let (mut i, stride) = grid_stride();
    while i < n {
        unsafe { *mask.add(i as usize) = (*x.add(i as usize) > threshold) as u8 };
        i += stride;
    }
}

/// `out[i] = src[idx[i]]` for row-major `[*, dim]` rows
#[kernel]
pub unsafe fn gather_rows(out: *mut f32, src: *const f32, idx: *const i32, n: i32, dim: i32) {
    let total = n as i64 * dim as i64;
    let (mut i, stride) = grid_stride();
    while i < total {
        let row = unsafe { *idx.add((i / dim as i64) as usize) } as i64;
        unsafe { *out.add(i as usize) = *src.add((row * dim as i64 + i % dim as i64) as usize) };
        i += stride;
    }
}

/// `x [rows, cols] -> [cols, rows]`
#[kernel]
pub unsafe fn transpose(out: *mut f32, x: *const f32, rows: i64, cols: i32) {
    let n = rows * cols as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let (c, r) = (i % cols as i64, i / cols as i64);
        unsafe { *out.add((c * rows + r) as usize) = *x.add(i as usize) };
        i += stride;
    }
}

/// Rows of `cols` floats with row stride `ld` into contiguous BF16 `[rows, cols]`
#[kernel]
pub unsafe fn f32_to_bf16(dst: *mut Bf16, src: *const f32, rows: i64, cols: i32, ld: i64) {
    let n = rows * cols as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let v = unsafe { *src.add(((i / cols as i64) * ld + i % cols as i64) as usize) };
        unsafe { Bf16::store(dst.add(i as usize), v) };
        i += stride;
    }
}
