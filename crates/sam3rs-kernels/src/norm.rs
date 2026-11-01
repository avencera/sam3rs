//! Normalizations, softmax and row reductions
//!
//! Kernels where a block owns a row (`layernorm`, `softmax`, the group statistics) keep
//! the port's block geometry and its reduction order, so numerics don't move: inputs are
//! `&[T]` and outputs `DisjointSlice`s, with writes through `get_unchecked_mut` where the
//! block's row, not a thread index, is what makes them disjoint. In-place kernels take the
//! buffer once, as the output; none takes the same buffer as both input and output. Row
//! lengths come from slice lengths and the grid, except LayerNorm's, which is a param
//! (see [`layernorm`]): the host launches one block per row, and a launch whose buffers
//! don't fit its grid does nothing instead of reading out of bounds

use core::ptr::addr_of_mut;

use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, gridDim_x, threadIdx_x};
use cuda_device::view::MatrixView32;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

use crate::common::{Bf16, Elem, activate, block_max, block_sum, math};

/// LayerNorm of one row: the block's threads take columns `tid, tid + blockDim, ...`
///
/// `load` and `store` read and write column `c` of the row. An empty `w` skips the affine
/// step and an empty `b` skips the shift
///
/// # Safety
/// Every thread of the block calls it; `scratch` holds 32 shared floats
#[inline(always)]
unsafe fn layernorm_row(
    cols: u32,
    eps: f32,
    w: &[f32],
    b: &[f32],
    scratch: *mut f32,
    load: impl Fn(u32) -> f32,
    mut store: impl FnMut(u32, f32),
) {
    // block-uniform, so a block that fails it never reaches a barrier
    if (!w.is_empty() && w.len() < cols as usize) || (!b.is_empty() && b.len() < cols as usize) {
        return;
    }
    let (tid, block) = (threadIdx_x(), blockDim_x());

    let mut s = 0.0;
    let mut c = tid;
    while c < cols {
        s += load(c);
        c += block;
    }
    // SAFETY: the caller's contract, which every block-wide call below shares
    let mean = unsafe { block_sum(s, scratch) } / cols as f32;

    let mut v = 0.0;
    c = tid;
    while c < cols {
        let d = load(c) - mean;
        v += d * d;
        c += block;
    }
    let rstd = math::rsqrtf(unsafe { block_sum(v, scratch) } / cols as f32 + eps);

    c = tid;
    while c < cols {
        let y = (load(c) - mean) * rstd;
        // SAFETY, for both reads: `c < cols`, and a non-empty `w` or `b` holds at least
        // `cols` floats; each read is inside the test that its slice is non-empty
        let y = if w.is_empty() {
            y
        } else {
            let shift = if b.is_empty() { 0.0 } else { unsafe { *b.get_unchecked(c as usize) } };
            y * unsafe { *w.get_unchecked(c as usize) } + shift
        };
        store(c, y);
        c += block;
    }
}

/// LayerNorm of row `blockIdx.x` of `x` into the same row of `out`
///
/// # Safety
/// As [`layernorm_row`]
#[inline(always)]
unsafe fn layernorm_into<T: Elem>(
    out: &mut DisjointSlice<T>,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    cols: u32,
    eps: f32,
    scratch: *mut f32,
) {
    // the checks are block-uniform, so a block that fails them never reaches a barrier
    let Some(xr) = MatrixView32::new(x, cols).row(blockIdx_x(), cols) else { return };
    if out.len() < x.len() {
        return;
    }
    let base = blockIdx_x() * cols;

    let store = |c: u32, y: f32| {
        // SAFETY: `row` checked that `base + c` is inside `x`, which `out` covers; column
        // `c` is only visited by thread `c % blockDim` of this block
        *unsafe { out.get_unchecked_mut((base + c) as usize) } = T::narrow(y);
    };
    // SAFETY: the caller's contract
    unsafe { layernorm_row(cols, eps, w, b, scratch, |c| xr.get(c).unwrap_or(0.0), store) }
}

/// LayerNorm over the last dim, one block per row; an empty `w` or `b` is skipped
///
/// `x` is `[rows, cols]` for a grid of `rows` blocks. `cols` is a param rather than
/// `x.len() / rows` because that division sits before the first load of every short
/// block: deriving it cost 2 registers and made `layernorm_bf16` 3% slower. Use
/// [`layernorm_inplace`] to normalize a buffer over itself
#[kernel]
pub fn layernorm(mut out: DisjointSlice<f32>, x: &[f32], w: &[f32], b: &[f32], cols: u32, eps: f32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    // SAFETY: every thread of the block runs this, and SCRATCH is used by nothing else
    unsafe { layernorm_into(&mut out, x, w, b, cols, eps, addr_of_mut!(SCRATCH) as *mut f32) };
}

/// [`layernorm`] into BF16, to feed a BF16 GEMM directly
#[kernel]
pub fn layernorm_bf16(mut out: DisjointSlice<Bf16>, x: &[f32], w: &[f32], b: &[f32], cols: u32, eps: f32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    // SAFETY: as in `layernorm`
    unsafe { layernorm_into(&mut out, x, w, b, cols, eps, addr_of_mut!(SCRATCH) as *mut f32) };
}

/// `x += add`, then [`layernorm_bf16`] of the sum, for a residual branch whose GEMM
/// wrote BF16: one pass instead of cuBLASLt's separate FP32 epilogue kernel and a
/// LayerNorm that re-reads the sum
///
/// Each thread adds the columns it later normalizes (`tid, tid + blockDim, ...`), so the
/// sum needs no barrier before the row statistics read it
#[kernel]
pub fn add_layernorm_bf16(
    mut out: DisjointSlice<Bf16>,
    mut x: DisjointSlice<f32>,
    add: &[Bf16],
    w: &[f32],
    b: &[f32],
    cols: u32,
    eps: f32,
) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    // block-uniform, so a block that fails it never reaches a barrier
    let end = (blockIdx_x() as usize + 1) * cols as usize;
    if cols == 0 || end > x.len() || end > add.len() || end > out.len() {
        return;
    }
    let base = blockIdx_x() * cols;
    let row = x.as_mut_ptr().wrapping_add(base as usize);

    let mut c = threadIdx_x();
    while c < cols {
        // SAFETY: the check above and `c < cols` keep `base + c` inside `x` and `add`,
        // and column `c` is only visited by thread `c % blockDim` of this block
        unsafe { *row.add(c as usize) += add.get_unchecked((base + c) as usize).widen() };
        c += blockDim_x();
    }

    // SAFETY, for both closures: as above, and `out` covers the same row
    let load = |c: u32| unsafe { *row.add(c as usize) };
    let store = |c: u32, y: f32| *unsafe { out.get_unchecked_mut((base + c) as usize) } = Bf16::narrow(y);
    // SAFETY: SCRATCH is used by nothing else, and every thread of the block runs this
    unsafe { layernorm_row(cols, eps, w, b, addr_of_mut!(SCRATCH) as *mut f32, load, store) };
}

/// [`layernorm`] over `x` itself
#[kernel]
pub fn layernorm_inplace(mut x: DisjointSlice<f32>, w: &[f32], b: &[f32], cols: u32, eps: f32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    // block-uniform, so a block that fails it never reaches a barrier
    if cols == 0 || (blockIdx_x() as usize + 1) * cols as usize > x.len() {
        return;
    }
    let row = x.as_mut_ptr().wrapping_add((blockIdx_x() * cols) as usize);

    // SAFETY, for both closures: the check above and `c < cols` keep `row + c` inside
    // `x`, and column `c` is only visited by thread `c % blockDim` of this block,
    // so each access is to a location no other thread touches
    let load = |c: u32| unsafe { *row.add(c as usize) };
    let store = |c: u32, y: f32| unsafe { *row.add(c as usize) = y };
    // SAFETY: SCRATCH is used by nothing else, and every thread of the block runs this
    unsafe { layernorm_row(cols, eps, w, b, addr_of_mut!(SCRATCH) as *mut f32, load, store) };
}

/// Row softmax over `x [rows, n]` in place, one block per row
///
/// Row `r` is query `q = r % lq` of batch `g = r / lq`. Optional additive
/// `bias[g / bias_div][q][j]`, a key mask (nonzero = masked) shared by all rows, and
/// causal masking (`j > q`); an empty `bias` or `key_mask` is skipped. Fully masked rows
/// become zeros
#[kernel]
pub fn softmax(mut x: DisjointSlice<f32>, lq: u32, bias: &[f32], bias_div: u32, key_mask: &[u8], causal: i32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    let scratch = addr_of_mut!(SCRATCH) as *mut f32;
    let r = blockIdx_x();
    let n = x.len() as u32 / gridDim_x();
    let (q, g) = (r % lq, r / lq);
    // an empty `bias` or `key_mask` is skipped
    let br = if bias.is_empty() { bias } else { &bias[(((g / bias_div) * lq + q) * n) as usize..][..n as usize] };
    let km = if key_mask.is_empty() { key_mask } else { &key_mask[..n as usize] };
    let (tid, block) = (threadIdx_x(), blockDim_x());
    let base = r * n;

    // SAFETY, for every `get_unchecked_mut` below: `r < gridDim.x` and `j < n` keep
    // `base + j` inside `x`, and column `j` of this row is only visited by thread
    // `j % blockDim` of this block
    let mut m = f32::NEG_INFINITY;
    let mut j = tid;
    while j < n {
        let xj = unsafe { x.get_unchecked_mut((base + j) as usize) };
        let mut v = *xj;
        if !br.is_empty() {
            v += br[j as usize];
        }
        if (!km.is_empty() && km[j as usize] != 0) || (causal != 0 && j > q) {
            v = f32::NEG_INFINITY;
        }
        *xj = v;
        m = math::fmaxf(m, v);
        j += block;
    }
    // SAFETY: every thread of the block calls it, and SCRATCH is used by nothing else
    m = unsafe { block_max(m, scratch) };

    let mut s = 0.0;
    j = tid;
    while j < n {
        let xj = unsafe { x.get_unchecked_mut((base + j) as usize) };
        let e = if m == f32::NEG_INFINITY { 0.0 } else { math::fast_expf(*xj - m) };
        *xj = e;
        s += e;
        j += block;
    }
    s = unsafe { block_sum(s, scratch) };

    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
    j = tid;
    while j < n {
        *unsafe { x.get_unchecked_mut((base + j) as usize) } *= inv;
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
pub fn group_stats(mut part: DisjointSlice<f32>, x: &[f32], c: u32, groups: u32, chunk: u32) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    let scratch = addr_of_mut!(SCRATCH) as *mut f32;
    let g = blockIdx_y();
    let cg = c / groups;
    let hw = x.len() as u32 / c;
    let (warp, lane, warps) = (threadIdx_x() / 32, threadIdx_x() % 32, blockDim_x() / 32);
    let start = blockIdx_x() * chunk;
    let end = (start + chunk).min(hw);
    // the group's channels start at column `g * cg` of every pixel row
    let x = &x[(g * cg) as usize..];

    let (mut s, mut s2) = (0.0f32, 0.0f32);
    let step = warps;
    let mut p = start + warp;
    // four pixels per pass keep four loads in flight; #[unroll(4)] can't take this loop,
    // whose step is not a constant
    while p + 3 * step < end {
        let pitch = step * c;
        let mut k = lane;
        while k < cg {
            let at = p * c + k;
            let (v0, v1, v2, v3) =
                (x[at as usize], x[(at + pitch) as usize], x[(at + 2 * pitch) as usize], x[(at + 3 * pitch) as usize]);
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
        let mut k = lane;
        while k < cg {
            let v = x[(p * c + k) as usize];
            s += v;
            s2 += v * v;
            k += 32;
        }
        p += step;
    }
    // SAFETY: every thread of the block calls it, and SCRATCH is used by nothing else
    s = unsafe { block_sum(s, scratch) };
    s2 = unsafe { block_sum(s2, scratch) };

    let i = ((g * gridDim_x() + blockIdx_x()) * 2) as usize;
    if threadIdx_x() == 0 && i + 1 < part.len() {
        // SAFETY: `i + 1` is inside `part`, and only thread 0 of the block for this
        // (group, chunk) pair writes the two floats at `i`
        unsafe {
            *part.get_unchecked_mut(i) = s;
            *part.get_unchecked_mut(i + 1) = s2;
        }
    }
}

/// Sums [`group_stats`]' partials into `stats[group * 2 + {0, 1}]`, one block per group
///
/// A fixed order instead of atomics, so the statistics don't change with block
/// scheduling: atomic adds flipped mask pixels between runs whose only difference was
/// timing. `part` holds `blocks` partial pairs per group, `blocks` being its length over
/// twice the grid
#[kernel]
pub fn group_stats_reduce(mut stats: DisjointSlice<f32>, part: &[f32]) {
    static mut SCRATCH: SharedArray<f32, 32> = SharedArray::UNINIT;
    let scratch = addr_of_mut!(SCRATCH) as *mut f32;
    let blocks = part.len() as u32 / (2 * gridDim_x());
    let g = blockIdx_x();
    // the checks are block-uniform, so a block that fails them never reaches a barrier
    let Some(pairs) = MatrixView32::new(part, 2 * blocks).row(g, 2 * blocks) else { return };

    let (mut s, mut s2) = (0.0f32, 0.0f32);
    let mut i = threadIdx_x();
    while i < blocks {
        s += pairs.get(i * 2).unwrap_or(0.0);
        s2 += pairs.get(i * 2 + 1).unwrap_or(0.0);
        i += blockDim_x();
    }
    // SAFETY: every thread of the block calls it, and SCRATCH is used by nothing else
    s = unsafe { block_sum(s, scratch) };
    s2 = unsafe { block_sum(s2, scratch) };

    let i = (g * 2) as usize;
    if threadIdx_x() == 0 && i + 1 < stats.len() {
        // SAFETY: `i + 1` is inside `stats`, and only thread 0 of group `g`'s block writes
        // the two floats at `i`
        unsafe {
            *stats.get_unchecked_mut(i) = s;
            *stats.get_unchecked_mut(i + 1) = s2;
        }
    }
}

/// Mean and variance from a group's summed statistics over `count` values
#[inline(always)]
fn moments(sum: f32, sum_sq: f32, count: f32) -> (f32, f32) {
    let mean = sum / count;
    (mean, math::fmaxf(sum_sq / count - mean * mean, 0.0))
}

/// GroupNorm with affine weights and an activation, in place, from [`group_stats_reduce`]
///
/// `x` is `[hw, c]`; `stats` is `[groups, 2]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn group_norm_apply(
    mut x: DisjointSlice<f32>,
    stats: &[f32],
    w: &[f32],
    b: &[f32],
    c: u32,
    groups: u32,
    eps: f32,
    act: i32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let hw = x.len() as u32 / c;
    let Some(mut x) = x.element_thread32(t) else { return };
    let ch = i % c;
    let cg = c / groups;
    let count = (hw * cg) as f32;

    let g = (ch / cg) as usize;
    let (mean, var) = moments(stats[g * 2], stats[g * 2 + 1], count);
    let v = (x.read() - mean) * math::rsqrtf(var + eps) * w[ch as usize] + b[ch as usize];
    x.write(activate(v, act));
}

/// Masked mean over the rows of `x [len, dim]` (mask nonzero = padding) into `out [dim]`
///
/// `len` is `x.len() / out.len()`; `mask` may be longer than `len`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn masked_mean(mut out: DisjointSlice<f32>, x: &[f32], mask: &[u8]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let dim = out.len() as u32;
    let Some(mut out) = out.element_thread32(t) else { return };
    let len = x.len() as u32 / dim;

    let (mut s, mut cnt) = (0.0f32, 0.0f32);
    // `col` is `None` only for an empty `x`, whose mean is zero
    if let Some(col) = MatrixView32::new(x, dim).col(i, len) {
        let mask = &mask[..len as usize];
        let mut t = 0;
        while t < len {
            if mask[t as usize] == 0 {
                s += col.get(t).unwrap_or(0.0);
                cnt += 1.0;
            }
            t += 1;
        }
    }
    out.write(s / math::fmaxf(cnt, 1.0));
}

/// `scores[q] = clamp(dot(hs[q], p) * scale, -clampv, clampv)`, one warp per query
///
/// `hs` is `[scores.len(), p.len()]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn dot_scores(mut scores: DisjointSlice<f32, thread::WarpIndex>, hs: &[f32], p: &[f32], scale: f32, clampv: f32) {
    let q = blockIdx_x() * (blockDim_x() / 32) + threadIdx_x() / 32;
    let lane = threadIdx_x() & 31;
    let dim = p.len() as u32;
    // the whole warp leaves together, so the shuffles below never see a partial warp
    let (Some(hs_row), Some(p_row)) = (MatrixView32::new(hs, dim).row(q, dim), MatrixView32::new(p, dim).row(0, dim))
    else {
        return;
    };

    let mut s = 0.0;
    let mut d = lane;
    while d < dim {
        s += hs_row.get(d).unwrap_or(0.0) * p_row.get(d).unwrap_or(0.0);
        d += 32;
    }
    s = warp::reduce_sum_f32(s);

    // `warp_index` hands lane 0 alone the slot for this warp
    if let Some(slot) = thread::warp_index()
        && let Some(score) = scores.get_mut(slot)
    {
        *score = math::fminf(math::fmaxf(s * scale, -clampv), clampv);
    }
}
