//! Normalizations, softmax and row reductions

use core::ptr::addr_of_mut;

use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, gridDim_x, threadIdx_x};
use cuda_device::{SharedArray, kernel};

use crate::common::{Bf16, Elem, activate, block_max, block_sum, grid_stride, math, warp_sum};

/// LayerNorm of row `blockIdx.x`: one block per row
///
/// # Safety
/// Every thread of the block calls it; `scratch` holds 32 shared floats
#[inline(always)]
unsafe fn layernorm_row<T: Elem>(
    out: *mut T,
    x: *const f32,
    w: *const f32,
    b: *const f32,
    cols: i32,
    eps: f32,
    scratch: *mut f32,
) {
    let row = blockIdx_x() as usize;
    let xr = unsafe { x.add(row * cols as usize) };
    let yr = unsafe { out.add(row * cols as usize) };
    let (tid, block) = (threadIdx_x() as i32, blockDim_x() as i32);

    let mut s = 0.0;
    let mut c = tid;
    while c < cols {
        s += unsafe { *xr.add(c as usize) };
        c += block;
    }
    let mean = unsafe { block_sum(s, scratch) } / cols as f32;

    let mut v = 0.0;
    c = tid;
    while c < cols {
        let d = unsafe { *xr.add(c as usize) } - mean;
        v += d * d;
        c += block;
    }
    let rstd = math::rsqrtf(unsafe { block_sum(v, scratch) } / cols as f32 + eps);

    c = tid;
    while c < cols {
        let y = (unsafe { *xr.add(c as usize) } - mean) * rstd;
        let y = if w.is_null() {
            y
        } else {
            y * unsafe { *w.add(c as usize) } + if b.is_null() { 0.0 } else { unsafe { *b.add(c as usize) } }
        };
        unsafe { T::store(yr.add(c as usize), y) };
        c += block;
    }
}

/// LayerNorm over the last dim, one block per row; `out` may alias `x`, and `w`/`b` may
/// be null
#[kernel]
pub unsafe fn layernorm(out: *mut f32, x: *const f32, w: *const f32, b: *const f32, cols: i32, eps: f32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    unsafe { layernorm_row(out, x, w, b, cols, eps, addr_of_mut!(SCRATCH) as *mut f32) };
}

/// [`layernorm`] into BF16, to feed a BF16 GEMM directly
#[kernel]
pub unsafe fn layernorm_bf16(out: *mut Bf16, x: *const f32, w: *const f32, b: *const f32, cols: i32, eps: f32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    unsafe { layernorm_row(out, x, w, b, cols, eps, addr_of_mut!(SCRATCH) as *mut f32) };
}

/// Row softmax over `x [rows, n]` in place, one block per row
///
/// Row `r` is query `q = r % lq` of batch `g = r / lq`. Optional additive
/// `bias[g / bias_div][q][j]`, a key mask (nonzero = masked) shared by all rows, and
/// causal masking (`j > q`). Fully masked rows become zeros
#[kernel]
pub unsafe fn softmax(x: *mut f32, n: i32, lq: i32, bias: *const f32, bias_div: i32, key_mask: *const u8, causal: i32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    let scratch = addr_of_mut!(SCRATCH) as *mut f32;
    let r = blockIdx_x() as i64;
    let q = (r % lq as i64) as i32;
    let g = r / lq as i64;
    let xr = unsafe { x.add((r * n as i64) as usize) };
    let br = if bias.is_null() {
        bias
    } else {
        unsafe { bias.add((((g / bias_div as i64) * lq as i64 + q as i64) * n as i64) as usize) }
    };
    let (tid, block) = (threadIdx_x() as i32, blockDim_x() as i32);

    let mut m = f32::NEG_INFINITY;
    let mut j = tid;
    while j < n {
        let mut v = unsafe { *xr.add(j as usize) };
        if !br.is_null() {
            v += unsafe { *br.add(j as usize) };
        }
        if (!key_mask.is_null() && unsafe { *key_mask.add(j as usize) } != 0) || (causal != 0 && j > q) {
            v = f32::NEG_INFINITY;
        }
        unsafe { *xr.add(j as usize) = v };
        m = math::fmaxf(m, v);
        j += block;
    }
    m = unsafe { block_max(m, scratch) };

    let mut s = 0.0;
    j = tid;
    while j < n {
        let e = if m == f32::NEG_INFINITY { 0.0 } else { math::fast_expf(unsafe { *xr.add(j as usize) } - m) };
        unsafe { *xr.add(j as usize) = e };
        s += e;
        j += block;
    }
    s = unsafe { block_sum(s, scratch) };

    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
    j = tid;
    while j < n {
        unsafe { *xr.add(j as usize) *= inv };
        j += block;
    }
}

/// GroupNorm statistics for channels-last `x [hw, c]`
///
/// One block per (group `blockIdx.y`, chunk `blockIdx.x`) writes its partial sums to
/// `part[(group * gridDim.x + chunk) * 2 + {0, 1}]`. Warp `i` takes pixels `i, i + warps,
/// ...` and lanes the group's channels, so every load is one coalesced row segment and
/// each block keeps several in flight
#[kernel]
pub unsafe fn group_stats(part: *mut f32, x: *const f32, hw: i64, c: i32, groups: i32, chunk: i32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    let scratch = addr_of_mut!(SCRATCH) as *mut f32;
    let g = blockIdx_y() as i32;
    let cg = c / groups;
    let (warp, lane, warps) = (threadIdx_x() / 32, threadIdx_x() % 32, blockDim_x() / 32);
    let start = blockIdx_x() as i64 * chunk as i64;
    let end = (start + chunk as i64).min(hw);

    let (mut s, mut s2) = (0.0f32, 0.0f32);
    let step = warps as i64;
    let mut p = start + warp as i64;
    // four pixels per pass keep four loads in flight; #[unroll(4)] can't take this loop,
    // whose step is not a constant
    while p + 3 * step < end {
        let row = unsafe { x.add((p * c as i64 + (g * cg) as i64) as usize) };
        let mut k = lane as i32;
        while k < cg {
            let (v0, v1, v2, v3) = unsafe {
                let r = row.add(k as usize);
                let pitch = step as usize * c as usize;
                (*r, *r.add(pitch), *r.add(2 * pitch), *r.add(3 * pitch))
            };
            s += v0;
            s2 += v0 * v0;
            s += v1;
            s2 += v1 * v1;
            s += v2;
            s2 += v2 * v2;
            s += v3;
            s2 += v3 * v3;
            k += 32;
        }
        p += 4 * step;
    }
    while p < end {
        let row = unsafe { x.add((p * c as i64 + (g * cg) as i64) as usize) };
        let mut k = lane as i32;
        while k < cg {
            let v = unsafe { *row.add(k as usize) };
            s += v;
            s2 += v * v;
            k += 32;
        }
        p += step;
    }
    s = unsafe { block_sum(s, scratch) };
    s2 = unsafe { block_sum(s2, scratch) };

    if threadIdx_x() == 0 {
        let i = ((g as i64 * gridDim_x() as i64 + blockIdx_x() as i64) * 2) as usize;
        unsafe {
            *part.add(i) = s;
            *part.add(i + 1) = s2;
        }
    }
}

/// Sums [`group_stats`]' partials into `stats[group * 2 + {0, 1}]`, one block per group
///
/// A fixed order instead of atomics, so the statistics don't change with block
/// scheduling: atomic adds flipped mask pixels between runs whose only difference was
/// timing
#[kernel]
pub unsafe fn group_stats_reduce(stats: *mut f32, part: *const f32, blocks: i32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    let scratch = addr_of_mut!(SCRATCH) as *mut f32;
    let p = unsafe { part.add(blockIdx_x() as usize * blocks as usize * 2) };

    let (mut s, mut s2) = (0.0f32, 0.0f32);
    let mut i = threadIdx_x() as i32;
    while i < blocks {
        unsafe {
            s += *p.add(i as usize * 2);
            s2 += *p.add(i as usize * 2 + 1);
        }
        i += blockDim_x() as i32;
    }
    s = unsafe { block_sum(s, scratch) };
    s2 = unsafe { block_sum(s2, scratch) };

    if threadIdx_x() == 0 {
        let i = blockIdx_x() as usize * 2;
        unsafe {
            *stats.add(i) = s;
            *stats.add(i + 1) = s2;
        }
    }
}

/// Mean and variance of group `g` from its summed statistics over `count` values
#[inline(always)]
pub unsafe fn group_moments(stats: *const f32, g: i32, count: f32) -> (f32, f32) {
    let mean = unsafe { *stats.add(g as usize * 2) } / count;
    let var = math::fmaxf(unsafe { *stats.add(g as usize * 2 + 1) } / count - mean * mean, 0.0);
    (mean, var)
}

/// GroupNorm with affine weights and an activation, in place, from [`group_stats_reduce`]
#[kernel]
pub unsafe fn group_norm_apply(
    x: *mut f32,
    stats: *const f32,
    w: *const f32,
    b: *const f32,
    hw: i64,
    c: i32,
    groups: i32,
    eps: f32,
    act: i32,
) {
    let n = hw * c as i64;
    let cg = c / groups;
    let count = (hw * cg as i64) as f32;
    let (mut i, stride) = grid_stride();
    while i < n {
        let ch = (i % c as i64) as i32;
        let (mean, var) = unsafe { group_moments(stats, ch / cg, count) };
        let v = unsafe {
            (*x.add(i as usize) - mean) * math::rsqrtf(var + eps) * *w.add(ch as usize) + *b.add(ch as usize)
        };
        unsafe { *x.add(i as usize) = activate(v, act) };
        i += stride;
    }
}

/// Masked mean over the rows of `x [len, dim]` (mask nonzero = padding) into `out [dim]`
#[kernel]
pub unsafe fn masked_mean(out: *mut f32, x: *const f32, mask: *const u8, len: i32, dim: i32) {
    let d = (blockIdx_x() * blockDim_x() + threadIdx_x()) as i32;
    if d >= dim {
        return;
    }

    let (mut s, mut cnt) = (0.0f32, 0.0f32);
    for t in 0..len {
        if unsafe { *mask.add(t as usize) } == 0 {
            s += unsafe { *x.add((t as i64 * dim as i64 + d as i64) as usize) };
            cnt += 1.0;
        }
    }
    unsafe { *out.add(d as usize) = s / math::fmaxf(cnt, 1.0) };
}

/// `scores[q] = clamp(dot(hs[q], p) * scale, -clampv, clampv)`, one warp per query
#[kernel]
pub unsafe fn dot_scores(scores: *mut f32, hs: *const f32, p: *const f32, nq: i32, dim: i32, scale: f32, clampv: f32) {
    let q = (blockIdx_x() * (blockDim_x() / 32) + threadIdx_x() / 32) as i32;
    let lane = (threadIdx_x() & 31) as i32;
    if q >= nq {
        return;
    }

    let mut s = 0.0;
    let mut d = lane;
    while d < dim {
        s += unsafe { *hs.add((q as i64 * dim as i64 + d as i64) as usize) * *p.add(d as usize) };
        d += 32;
    }
    s = warp_sum(s);
    if lane == 0 {
        unsafe { *scores.add(q as usize) = math::fminf(math::fmaxf(s * scale, -clampv), clampv) };
    }
}
