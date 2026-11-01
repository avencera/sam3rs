//! Fused attention (FlashAttention-2 style) for the BF16 precision mode
//!
//! FP32 q/k/v are rounded to FP16 as they are staged in shared memory (see
//! [`AttnIn::Qk`]), the softmax runs in FP32 with an online max, and P @ V runs on FP16
//! tensor cores, accumulating each 64-key tile in FP16 before adding it to FP32 totals.
//! Each warp owns 16 query rows; a block walks all keys in tiles of 64, so the `[lq, lk]`
//! score matrix never reaches global memory
//!
//! q, k, v and the output are batched strided views of slices: batch entry `b`, row `i`
//! starts at `b * stride + i * ld`. Each block checks the rows it will touch against the
//! slice lengths once, up front, and traps like a failed slice index if they don't fit;
//! the staging loads and output stores inside then index without checks

use core::f32::consts::LOG2_E;
use core::ptr::addr_of_mut;

use cuda_device::convert::bf16_to_f32;
use cuda_device::float::{ex2_approx_ftz_f32, fma_rn_f32};
// `#[unroll]` is only accepted in `#[kernel]` and `#[device]` bodies, and a generic
// `#[device]` fn is never inlined: it stays a `.func` whose shared and global pointers
// reach the body as generic ones, about 25% slower here. So the generic bodies below are
// plain `#[inline(always)]` fns, and their loops call the marker `#[unroll]` expands to,
// `__unroll_config::<0>()` (unroll fully), themselves
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_y, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, ptx_asm, thread, warp};

use crate::common::{Bf16, F16, Packed8, ldg_u4, ldg4, ldg8, math, pack_f16, stg2};
use crate::mma::{FragA, Mma16, SharedTile, sts2, sts4};

pub(crate) const FA_WARPS: usize = 4;
/// Keys per tile
pub(crate) const FA_BN: usize = 64;
/// Upper bound on each thread's chunks when staging a tile, 8 for a D = 64 FP32 tile. The
/// staging loops run to it, fully unrolled, and skip chunks past the tile, a test that folds
/// away once inlining makes the tile size constant; a bound that is only constant after
/// inlining leaves the loop rolled (the unroll marker is checked before inlining, and LLVM
/// ignored a partial-unroll count). They load every chunk into registers before storing any,
/// because the shared stores are asm with a memory clobber that no load moves above: one
/// load round trip per chunk left BF16 encode 2.4% slower than nvcc's build, which unrolled
/// 4x and issued the loads together
pub(crate) const STAGE_MAX_ITERS: usize = 8;
/// Score fragments per tile: one per 8 keys
pub(crate) const SJ: usize = FA_BN / 8;

#[inline(always)]
pub(crate) fn ex2(x: f32) -> f32 {
    ex2_approx_ftz_f32(x)
}

/// Traps when the host's geometry doesn't fit the buffers it passed, as a failed slice
/// index would
#[inline(always)]
pub(crate) fn require(ok: bool) {
    if !ok {
        panic!();
    }
}

/// Whether rows `[row0, row0 + rows)` of `width` elements, row `i` at `base + i * ld`, all
/// lie in a buffer of `len`. No rows always fit
///
/// Plain 64-bit math: with 32-bit strides, row indices below 2^31 and a batch offset
/// below 2^48 nothing wraps, while overflow-checked products cost about 20 instructions
/// a check. The last row index goes through u32 so it widens by zero extension, one
/// 32x32 multiply instead of a sign-extended 64x64 one
#[inline(always)]
pub(crate) fn rows_fit(len: usize, base: usize, row0: i32, rows: i32, ld: u32, width: usize) -> bool {
    rows <= 0 || base + (row0 + rows - 1) as u32 as usize * ld as usize + width <= len
}

/// One float through the read-only path
#[inline(always)]
pub(crate) unsafe fn ldg_nc(p: *const f32) -> f32 {
    let v: f32;
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; ld.global.nc.f32 %0, [g]; }",
            out("=f") v,
            in("l") p as u64,
        );
    }
    v
}

/// Two floats from an 8-byte aligned address through the read-only path
#[inline(always)]
pub(crate) unsafe fn ldg2_nc(p: *const f32) -> (f32, f32) {
    let (a, b): (f32, f32);
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.nc.v2.f32 {%0, %1}, [g]; }",
            out("=f") a,
            out("=f") b,
            in("l") p as u64,
        );
    }
    (a, b)
}

/// A batched strided matrix in a slice: batch entry `b`, row `i` starts at `b * stride + i
/// * ld`
#[derive(Clone, Copy)]
pub(crate) struct Batched<'a, T> {
    pub(crate) data: &'a [T],
    pub(crate) ld: u32,
    pub(crate) stride: u32,
}

impl<'a, T> Batched<'a, T> {
    /// Rows `[row0, row0 + rows)` of batch entry `b`, `width` elements each, or `None` if
    /// they don't all lie in the slice
    #[inline(always)]
    pub(crate) fn rows(self, b: usize, row0: i32, rows: i32, width: usize) -> Option<Rows<T>> {
        let base = b * self.stride as usize;
        let ptr = self.data.as_ptr().wrapping_add(base + row0 as u32 as usize * self.ld as usize);
        rows_fit(self.data.len(), base, row0, rows, self.ld, width).then_some(Rows { ptr, ld: self.ld as usize, rows })
    }
}

/// Rows `[0, rows)` of a strided matrix, every element of the declared width proven in
/// bounds when it was made by [`Batched::rows`]
pub(crate) struct Rows<T> {
    ptr: *const T,
    ld: usize,
    pub(crate) rows: i32,
}

impl<T> Clone for Rows<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Rows<T> {}

impl<T> Rows<T> {
    /// The rows from `r` on; `0 <= r <= rows`
    #[inline(always)]
    pub(crate) fn skip(self, r: i32) -> Self {
        Self { ptr: self.ptr.wrapping_add(r as u32 as usize * self.ld), ld: self.ld, rows: self.rows - r }
    }

    /// Element `c` of row `r`
    ///
    /// # Safety
    /// `r < rows` and `c` is within the width the rows were checked for
    #[inline(always)]
    pub(crate) unsafe fn at(self, r: usize, c: usize) -> *const T {
        unsafe { self.ptr.add(r * self.ld + c) }
    }
}

/// A q, k or v element type and how it is staged into shared memory
trait AttnIn: Copy {
    /// The operand format of QK^T for q and k of this type. FP32 sources round to FP16,
    /// whose 3 extra mantissa bits over BF16 also cover the error of summing each score's
    /// D products in FP16, at twice the FP32-accumulate rate on GeForce parts. BF16
    /// sources stay BF16
    type Qk: Mma16;

    /// `rows` rows of `[_, D]` into `dst` as QK^T operands; rows past `src.rows` are zero
    ///
    /// `threads` is the block size, a compile-time constant at every call site rather than
    /// `blockDim_x()`, so the loops unroll fully (see [`STAGE_MAX_ITERS`])
    ///
    /// # Safety
    /// Every thread of a block of exactly `threads` threads calls it between barriers;
    /// `dst` has `rows` rows, and `rows * D / 4 <= STAGE_MAX_ITERS * threads`
    unsafe fn stage_qk<const D: usize, const LD: usize>(
        dst: SharedTile<Self::Qk, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    );

    /// The same into FP16, for V
    ///
    /// # Safety
    /// As [`AttnIn::stage_qk`]
    unsafe fn stage_v<const D: usize, const LD: usize>(
        dst: SharedTile<F16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    );
}

/// FP32 sources, rounded to FP16 four at a time
///
/// # Safety
/// As [`AttnIn::stage_qk`]
#[inline(always)]
unsafe fn stage_f32_to_f16<const D: usize, const LD: usize>(
    dst: SharedTile<F16, LD>,
    src: Rows<f32>,
    rows: usize,
    threads: usize,
) {
    let n = rows * D / 4;
    let mut x = [[0.0; 4]; STAGE_MAX_ITERS];
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 4), (i % (D / 4)) * 4);
        if i < n && (r as i32) < src.rows {
            // SAFETY: r < src.rows and c + 4 <= D, inside the checked rows; D and the row
            // starts are multiples of 4 floats, and buffers 16-byte aligned
            x[it] = unsafe { ldg4(src.at(r, c)) };
        }
    }
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 4), (i % (D / 4)) * 4);
        if i < n {
            let x = x[it];
            // SAFETY: (r, c) is this thread's alone and r < rows, 8-byte aligned as c is a
            // multiple of 4
            unsafe { sts2(dst.at(r, c), pack_f16(x[0], x[1]), pack_f16(x[2], x[3])) };
        }
    }
}

/// 16-bit sources copied 16 bytes at a time
///
/// # Safety
/// As [`AttnIn::stage_qk`]
#[inline(always)]
unsafe fn stage_copy16<const D: usize, const LD: usize, T, Q>(
    dst: SharedTile<Q, LD>,
    src: Rows<T>,
    rows: usize,
    threads: usize,
) {
    let n = rows * D / 8;
    let mut x = [[0; 4]; STAGE_MAX_ITERS];
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n && (r as i32) < src.rows {
            // SAFETY: as in stage_f32_to_f16, 8 values at a time
            x[it] = unsafe { ldg_u4(src.at(r, c)) };
        }
    }
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n {
            // SAFETY: (r, c) is this thread's alone and r < rows; LD and c are multiples of 8
            unsafe { sts4(dst.at(r, c), x[it]) };
        }
    }
}

/// BF16 sources converted to FP16 through FP32, eight values at a time
///
/// # Safety
/// As [`AttnIn::stage_qk`]
#[inline(always)]
unsafe fn stage_bf16_to_f16<const D: usize, const LD: usize>(
    dst: SharedTile<F16, LD>,
    src: Rows<Bf16>,
    rows: usize,
    threads: usize,
) {
    let n = rows * D / 8;
    let mut x = [[0; 4]; STAGE_MAX_ITERS];
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n && (r as i32) < src.rows {
            // SAFETY: as in stage_copy16
            x[it] = unsafe { ldg_u4(src.at(r, c)) };
        }
    }
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n {
            let x = x[it];
            let f16 = |w: u32| pack_f16(bf16_to_f32(w as u16), bf16_to_f32((w >> 16) as u16));
            // SAFETY: as in stage_copy16
            unsafe { sts4(dst.at(r, c), [f16(x[0]), f16(x[1]), f16(x[2]), f16(x[3])]) };
        }
    }
}

impl AttnIn for f32 {
    type Qk = F16;

    #[inline(always)]
    unsafe fn stage_qk<const D: usize, const LD: usize>(
        dst: SharedTile<F16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_f32_to_f16::<D, LD>(dst, src, rows, threads) };
    }

    #[inline(always)]
    unsafe fn stage_v<const D: usize, const LD: usize>(
        dst: SharedTile<F16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_f32_to_f16::<D, LD>(dst, src, rows, threads) };
    }
}

impl AttnIn for Bf16 {
    type Qk = Bf16;

    #[inline(always)]
    unsafe fn stage_qk<const D: usize, const LD: usize>(
        dst: SharedTile<Bf16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_copy16::<D, LD, Self, Bf16>(dst, src, rows, threads) };
    }

    #[inline(always)]
    unsafe fn stage_v<const D: usize, const LD: usize>(
        dst: SharedTile<F16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_bf16_to_f16::<D, LD>(dst, src, rows, threads) };
    }
}

impl AttnIn for F16 {
    type Qk = F16;

    #[inline(always)]
    unsafe fn stage_qk<const D: usize, const LD: usize>(
        dst: SharedTile<F16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_copy16::<D, LD, Self, F16>(dst, src, rows, threads) };
    }

    #[inline(always)]
    unsafe fn stage_v<const D: usize, const LD: usize>(
        dst: SharedTile<F16, LD>,
        src: Rows<Self>,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_copy16::<D, LD, Self, F16>(dst, src, rows, threads) };
    }
}

/// Shared tiles of one block: q `[16 * WARPS, D]`, k and v `[FA_BN, D]`, rows padded to
/// `LD = D + 8`, which keeps fragment loads bank-conflict free. q and k share the QK^T
/// format `Q`
struct Tiles<Q, const LD: usize> {
    q: SharedTile<Q, LD>,
    k: SharedTile<Q, LD>,
    v: SharedTile<F16, LD>,
}

/// The box relative position bias of the DETR decoder's image cross-attention
///
/// For head `h = blockIdx.y`, query row `1 + q` and key `y * side + x` it is
/// `dy[h, q, y] + dx[h, q, x]`, both `[heads, lq - 1, side]`; row 0 (the presence token)
/// gets none. `side` must be even, so each thread's key pair shares a grid row
#[derive(Clone, Copy)]
pub(crate) struct Rpb<'a> {
    pub(crate) dx: &'a [f32],
    pub(crate) dy: &'a [f32],
    pub(crate) side: u32,
}

/// The attention of one block of `16 * WARPS` query rows
///
/// Batch entry `blockIdx.y`; `scale_log2 = log2(e) / sqrt(D)`. Keys with `key_pad[j] !=
/// 0` are skipped (an empty `key_pad` skips none), and rows with no valid key produce
/// zeros, like the softmax kernel. With a non-empty `part`, block z covers keys `[z *
/// kv_chunk, (z + 1) * kv_chunk)` only and writes its unnormalized accumulators, row max
/// and row sum to `part[z][batch][row][D + 2]` for [`flash_combine_d32`] instead of `o`,
/// so few query rows over many keys still fill the GPU. WARPS sets the query rows per
/// block: every block re-reads all of K and V, so taller blocks cut that traffic. `KK =
/// D / 16`, `DJ = D / 8`, `LD = D + 8`
///
/// # Safety
/// The block has exactly `32 * WARPS` threads (the kernels' `.reqntid`), and distinct
/// (batch, row, column) of `o` are distinct elements under its `ld` and `stride`
#[inline(always)]
unsafe fn flash_attn<
    const D: usize,
    const KK: usize,
    const DJ: usize,
    const LD: usize,
    const WARPS: usize,
    const RPB: bool,
    T: AttnIn,
    TK: AttnIn<Qk = T::Qk>,
    TV: AttnIn,
>(
    tiles: Tiles<T::Qk, LD>,
    mut o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    q: Batched<T>,
    k: Batched<TK>,
    v: Batched<TV>,
    lq: u32,
    lk: u32,
    scale_log2: f32,
    key_pad: &[u8],
    rpb: Rpb,
    mut part: DisjointSlice<f32>,
    kv_chunk: u32,
) {
    let (lq, lk) = (lq as i32, lk as i32);
    let bm = 16 * WARPS;
    // FP32 sources are staged 4 values per chunk, so a tile is rows * D / 4 chunks at most
    const { assert!(16 * WARPS * D / 4 <= STAGE_MAX_ITERS * 32 * WARPS && FA_BN * D / 4 <= STAGE_MAX_ITERS * 32 * WARPS) };
    let Tiles { q: qs, k: ks, v: vs } = tiles;

    let b = blockIdx_y() as usize;
    let q0 = (blockIdx_x() as usize * bm) as i32;
    let (warp, lane) = (threadIdx_x() as usize / 32, threadIdx_x() as usize % 32);
    let (g, t, wr) = (lane / 4, lane % 4, warp * 16);
    let has_pad = !key_pad.is_empty();
    let split = !part.is_empty();
    let (mut k_begin, mut k_end) = (0, lk);
    if split {
        k_begin = blockIdx_z() as i32 * kv_chunk as i32;
        k_end = lk.min(k_begin + kv_chunk as i32);
    }

    // everything this block reads or writes, checked at once up front: one uniform
    // branch to one trap. The RPB rows are checked for the whole block, as a check per
    // thread row would make ptxas treat the warp as divergent from there on
    let nq = (lq - q0).min(bm as i32);
    let part_base = (blockIdx_z() as usize * gridDim_y() as usize + b) * lq as usize * (D + 2);
    let out_ok = if split {
        rows_fit(part.len(), part_base, q0, nq, (D + 2) as u32, D + 2)
    } else {
        rows_fit(o.len(), b * o_stride as usize, q0, nq, o_ld, D)
    };
    let (rpb_side, rpb_r0) = (rpb.side as usize, q0.max(1));
    let rpb_base = b * (lq - 1) as usize * rpb_side;
    let rpb_len = rpb.dx.len().min(rpb.dy.len());
    let rpb_ok = !RPB
        || (lk as u64 <= rpb.side as u64 * rpb.side as u64
            && rows_fit(rpb_len, rpb_base, rpb_r0 - 1, (q0 + bm as i32).min(lq) - rpb_r0, rpb.side, rpb_side));
    let (qr, kr, vr) =
        (q.rows(b, q0, nq, D), k.rows(b, k_begin, k_end - k_begin, D), v.rows(b, k_begin, k_end - k_begin, D));
    require(qr.is_some() & kr.is_some() & vr.is_some() & out_ok & rpb_ok & (!has_pad | (key_pad.len() >= lk as usize)));
    let (Some(qr), Some(kr), Some(vr)) = (qr, kr, vr) else { return };

    // SAFETY: barriers order the staging stores before the fragment loads, and the
    // block's 32 * WARPS threads cover the tile's 16 * WARPS rows
    unsafe { T::stage_qk::<D, LD>(qs, qr, bm, 32 * WARPS) };
    thread::sync_threads();
    let mut qa = [FragA::<T::Qk>::ZERO; KK];
    let mut kk = 0;
    while kk < KK {
        __unroll_config::<0>();
        qa[kk] = unsafe { qs.frag_a(wr, kk * 16) };
        kk += 1;
    }

    let mut acc = [[0.0f32; 4]; DJ];
    let (mut m, mut l) = ([f32::NEG_INFINITY; 2], [0.0f32; 2]);
    // per fragment row: this head's dx and dy rows, or null for the presence token and
    // rows past lq
    let mut rdx = [core::ptr::null::<f32>(); 2];
    let mut rdy = [core::ptr::null::<f32>(); 2];
    if RPB {
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let row = q0 + (wr + g + r * 8) as i32;
            if row >= 1 && row < lq {
                let base = rpb_base + (row - 1) as usize * rpb_side;
                rdx[r] = rpb.dx.as_ptr().wrapping_add(base);
                rdy[r] = rpb.dy.as_ptr().wrapping_add(base);
            }
            r += 1;
        }
    }

    let mut k0 = k_begin;
    while k0 < k_end {
        thread::sync_threads();
        // SAFETY: as for q; the barriers on either side keep the previous tile's readers
        // and this tile's writers apart
        unsafe {
            TK::stage_qk::<D, LD>(ks, kr.skip(k0 - k_begin), FA_BN, 32 * WARPS);
            TV::stage_v::<D, LD>(vs, vr.skip(k0 - k_begin), FA_BN, 32 * WARPS);
        }
        thread::sync_threads();

        let mut sacc = [<T::Qk as Mma16>::ZERO; SJ];
        let mut j = 0;
        while j < SJ {
            __unroll_config::<0>();
            let mut kk = 0;
            while kk < KK {
                __unroll_config::<0>();
                // SAFETY: the whole warp runs this with warp-uniform offsets inside the tile
                unsafe {
                    let kb = ks.frag_b_pair_nk(j * 8, kk * 16);
                    sacc[j] = T::Qk::mma(sacc[j], qa[kk], kb[0]);
                    sacc[j + 1] = T::Qk::mma(sacc[j + 1], qa[kk], kb[1]);
                }
                kk += 1;
            }
            j += 2;
        }
        let mut s = [[0.0f32; 4]; SJ];
        let mut j = 0;
        while j < SJ {
            __unroll_config::<0>();
            s[j] = T::Qk::acc_f32(sacc[j]);
            j += 1;
        }

        let mut corr = [0.0f32; 2];
        // a whole tile of real keys without padding or bias: the softmax needs no masking
        // or infinity guards, and the scale folds into the exponent's FMA. A row max of
        // -inf only comes from the first tile, where corr = ex2(-inf) = 0 is right
        if !RPB && !has_pad && k0 + FA_BN as i32 <= k_end {
            let mut mx = [f32::NEG_INFINITY; 2];
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                mx[0] = math::fmaxf(mx[0], math::fmaxf(s[j][0], s[j][1]));
                mx[1] = math::fmaxf(mx[1], math::fmaxf(s[j][2], s[j][3]));
                j += 1;
            }
            let mut r = 0;
            while r < 2 {
                __unroll_config::<0>();
                mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 1));
                mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 2));
                mx[r] = math::fmaxf(m[r], mx[r] * scale_log2);
                corr[r] = ex2(m[r] - mx[r]);
                m[r] = mx[r];
                l[r] *= corr[r];
                r += 1;
            }
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    let p = ex2(fma_rn_f32(s[j][e], scale_log2, -m[e / 2]));
                    s[j][e] = p;
                    l[e / 2] += p;
                    e += 1;
                }
                j += 1;
            }
        } else {
            let mut mx = m;
            // the grid position (gy, gx) of this thread's first key pair, k0 + j * 8 + t * 2,
            // tracked as j advances instead of dividing per score
            let side = rpb.side as i32;
            let (mut gy, mut gx) = (0i32, 0i32);
            if RPB {
                gy = k0 / side;
                gx = k0 - gy * side + t as i32 * 2;
                while gx >= side {
                    gx -= side;
                    gy += 1;
                }
            }
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let mut bias = [(0.0f32, 0.0f32); 2];
                if RPB {
                    let mut r = 0;
                    while r < 2 {
                        __unroll_config::<0>();
                        if !rdx[r].is_null() && k0 + ((j * 8 + t * 2) as i32) < lk {
                            // SAFETY: the key is below lk <= side^2, so gy < side, and gx + 1
                            // < side as side is even; the row's side values were checked
                            let dy = unsafe { ldg_nc(rdy[r].add(gy as usize)) };
                            let dx = unsafe { ldg2_nc(rdx[r].add(gx as usize)) };
                            bias[r] = (dy + dx.0, dy + dx.1);
                        }
                        r += 1;
                    }
                    gx += 8;
                    while gx >= side {
                        gx -= side;
                        gy += 1;
                    }
                }
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    let col = k0 + (j * 8 + t * 2 + (e & 1)) as i32;
                    let mut x = s[j][e] * scale_log2;
                    // scaled here rather than with the loads, so it contracts into the same FMA
                    if RPB {
                        x += (if e & 1 != 0 { bias[e / 2].1 } else { bias[e / 2].0 }) * LOG2_E;
                    }
                    // SAFETY: col < k_end <= lk <= key_pad.len(), checked above
                    if col >= k_end || (has_pad && unsafe { *key_pad.get_unchecked(col as usize) } != 0) {
                        x = f32::NEG_INFINITY;
                    }
                    s[j][e] = x;
                    mx[e / 2] = math::fmaxf(mx[e / 2], x);
                    e += 1;
                }
                j += 1;
            }
            let mut r = 0;
            while r < 2 {
                __unroll_config::<0>();
                mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 1));
                mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 2));
                corr[r] = if mx[r] == f32::NEG_INFINITY { 1.0 } else { ex2(m[r] - mx[r]) };
                m[r] = mx[r];
                l[r] *= corr[r];
                r += 1;
            }
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    let p = if m[e / 2] == f32::NEG_INFINITY { 0.0 } else { ex2(s[j][e] - m[e / 2]) };
                    s[j][e] = p;
                    l[e / 2] += p;
                    e += 1;
                }
                j += 1;
            }
        }
        let mut dj = 0;
        while dj < DJ {
            __unroll_config::<0>();
            acc[dj][0] *= corr[0];
            acc[dj][1] *= corr[0];
            acc[dj][2] *= corr[1];
            acc[dj][3] *= corr[1];
            dj += 1;
        }

        // P @ V runs in FP16 with FP16 accumulation over this tile's 64 keys (p <= 1, so
        // the partial sums stay within 64 |v|), then joins the FP32 accumulator
        let mut pacc = [<F16 as Mma16>::ZERO; DJ];
        let mut kk = 0;
        while kk < FA_BN / 16 {
            __unroll_config::<0>();
            let pa = FragA::<F16>::from_acc_pair(s[2 * kk], s[2 * kk + 1]);
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                // SAFETY: as for QK^T
                unsafe {
                    let vb = vs.frag_b_pair_kn(kk * 16, dj * 8);
                    pacc[dj] = F16::mma(pacc[dj], pa, vb[0]);
                    pacc[dj + 1] = F16::mma(pacc[dj + 1], pa, vb[1]);
                }
                dj += 2;
            }
            kk += 1;
        }
        let mut dj = 0;
        while dj < DJ {
            __unroll_config::<0>();
            let p = F16::acc_f32(pacc[dj]);
            acc[dj][0] += p[0];
            acc[dj][1] += p[1];
            acc[dj][2] += p[2];
            acc[dj][3] += p[3];
            dj += 1;
        }
        k0 += FA_BN as i32;
    }

    let mut r = 0;
    while r < 2 {
        __unroll_config::<0>();
        l[r] += warp::shuffle_xor_f32(l[r], 1);
        l[r] += warp::shuffle_xor_f32(l[r], 2);
        r += 1;
    }
    // each thread stores its own two rows at columns t * 2 + 8 * dj, which no other
    // thread of any block writes, inside the rows checked up front
    if split {
        let pp = part.as_mut_ptr().wrapping_add(part_base + q0 as u32 as usize * (D + 2));
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let row = wr + g + r * 8;
            if (row as i32) < nq {
                // SAFETY: row < nq, inside the checked rows; D + 2 is even, so each pair is
                // 8-byte aligned
                unsafe {
                    let out = pp.add(row * (D + 2));
                    let mut dj = 0;
                    while dj < DJ {
                        __unroll_config::<0>();
                        stg2(out.add(dj * 8 + t * 2), acc[dj][2 * r], acc[dj][2 * r + 1]);
                        dj += 1;
                    }
                    if t == 0 {
                        *out.add(D) = m[r];
                        *out.add(D + 1) = l[r];
                    }
                }
            }
            r += 1;
        }
        return;
    }

    let mut r = 0;
    while r < 2 {
        __unroll_config::<0>();
        l[r] = if l[r] > 0.0 { 1.0 / l[r] } else { 0.0 };
        r += 1;
    }
    let op = o.as_mut_ptr().wrapping_add(b * o_stride as usize + q0 as u32 as usize * o_ld as usize);
    let mut r = 0;
    while r < 2 {
        __unroll_config::<0>();
        let row = wr + g + r * 8;
        if (row as i32) < nq {
            // SAFETY: as for part; the host keeps o_ld and o_stride even
            let out = unsafe { op.add(row * o_ld as usize) };
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                unsafe { stg2(out.add(dj * 8 + t * 2), acc[dj][2 * r] * l[r], acc[dj][2 * r + 1] * l[r]) };
                dj += 1;
            }
        }
        r += 1;
    }
}

/// Declares one kernel's shared tiles for `$rows` query rows and head dim `$d`
macro_rules! tiles {
    ($rows:expr, $d:expr) => {{
        static mut QS: SharedArray<u16, { $rows * ($d + 8) }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<u16, { FA_BN * ($d + 8) }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<u16, { FA_BN * ($d + 8) }, 16> = SharedArray::UNINIT;
        // SAFETY: each array is this kernel's own, 16-byte aligned and sized for the rows
        // flash_attn addresses
        unsafe {
            Tiles {
                q: SharedTile::new(addr_of_mut!(QS) as *mut u16),
                k: SharedTile::new(addr_of_mut!(KS) as *mut u16),
                v: SharedTile::new(addr_of_mut!(VS) as *mut u16),
            }
        }
    }};
}

/// Expands to a plain flash attention kernel for head dim `$d` and input type `$t`
macro_rules! flash_attn_kernel {
    ($name:ident, $d:literal, $t:ty) => {
        #[doc = concat!("Flash attention for head dim ", $d, " on `", stringify!($t), "` q/k/v; see [`flash_attn`]")]
        #[kernel]
        #[launch_contract(domain = 3, block = (128, 1, 1))]
        pub fn $name(
            o: DisjointSlice<f32>,
            o_ld: u32,
            o_stride: u32,
            q: &[$t],
            q_ld: u32,
            q_stride: u32,
            k: &[$t],
            k_ld: u32,
            k_stride: u32,
            v: &[$t],
            v_ld: u32,
            v_stride: u32,
            lq: u32,
            lk: u32,
            scale_log2: f32,
            key_pad: &[u8],
            part: DisjointSlice<f32>,
            kv_chunk: u32,
        ) {
            let tiles = tiles!(16 * FA_WARPS, $d);
            let rpb = Rpb { dx: &[], dy: &[], side: 0 };
            // SAFETY: `.reqntid` fixes the block at 128 threads = FA_WARPS warps
            unsafe {
                flash_attn::<$d, { $d / 16 }, { $d / 8 }, { $d + 8 }, FA_WARPS, false, $t, $t, $t>(
                    tiles,
                    o,
                    o_ld,
                    o_stride,
                    Batched { data: q, ld: q_ld, stride: q_stride },
                    Batched { data: k, ld: k_ld, stride: k_stride },
                    Batched { data: v, ld: v_ld, stride: v_stride },
                    lq,
                    lk,
                    scale_log2,
                    key_pad,
                    rpb,
                    part,
                    kv_chunk,
                )
            };
        }
    };
}

flash_attn_kernel!(flash_attn_d32, 32, f32);
flash_attn_kernel!(flash_attn_d64, 64, f32);
flash_attn_kernel!(flash_attn_bf16_d64, 64, Bf16);

/// Flash attention with the box relative position bias of [`Rpb`] for head dim 32
#[kernel]
#[launch_contract(domain = 3, block = (128, 1, 1))]
pub fn flash_attn_rpb_d32(
    o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    q: &[f32],
    q_ld: u32,
    q_stride: u32,
    k: &[f32],
    k_ld: u32,
    k_stride: u32,
    v: &[f32],
    v_ld: u32,
    v_stride: u32,
    lq: u32,
    lk: u32,
    scale_log2: f32,
    rpb_dx: &[f32],
    rpb_dy: &[f32],
    side: u32,
    part: DisjointSlice<f32>,
    kv_chunk: u32,
) {
    let tiles = tiles!(16 * FA_WARPS, 32);
    // SAFETY: `.reqntid` fixes the block at 128 threads = FA_WARPS warps
    unsafe {
        flash_attn::<32, 2, 4, 40, FA_WARPS, true, f32, f32, f32>(
            tiles,
            o,
            o_ld,
            o_stride,
            Batched { data: q, ld: q_ld, stride: q_stride },
            Batched { data: k, ld: k_ld, stride: k_stride },
            Batched { data: v, ld: v_ld, stride: v_stride },
            lq,
            lk,
            scale_log2,
            &[],
            Rpb { dx: rpb_dx, dy: rpb_dy, side },
            part,
            kv_chunk,
        )
    };
}

/// FP32 q with K and V converted once by [`kv_to_16`], in 128-row blocks
#[kernel]
#[launch_contract(domain = 3, block = (256, 1, 1))]
pub fn flash_attn_kv16_d32(
    o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    q: &[f32],
    q_ld: u32,
    q_stride: u32,
    k: &[F16],
    k_ld: u32,
    k_stride: u32,
    v: &[F16],
    v_ld: u32,
    v_stride: u32,
    lq: u32,
    lk: u32,
    scale_log2: f32,
    key_pad: &[u8],
) {
    let tiles = tiles!(16 * 8, 32);
    // SAFETY: an empty slice covers no memory
    let no_part = unsafe { DisjointSlice::from_mut_slice(&mut []) };
    // SAFETY: `.reqntid` fixes the block at 256 threads = 8 warps
    unsafe {
        flash_attn::<32, 2, 4, 40, 8, false, f32, F16, F16>(
            tiles,
            o,
            o_ld,
            o_stride,
            Batched { data: q, ld: q_ld, stride: q_stride },
            Batched { data: k, ld: k_ld, stride: k_stride },
            Batched { data: v, ld: v_ld, stride: v_stride },
            lq,
            lk,
            scale_log2,
            key_pad,
            Rpb { dx: &[], dy: &[], side: 0 },
            no_part,
            0,
        )
    };
}

/// Batch-strided FP32 views `k` and `v [batch][lk, hd]` into contiguous FP16 `kvd = [K;
/// V]`, each half `[lk, batch * hd]`, for [`flash_attn_kv16_d32`]: rounding them once
/// instead of in every query block. One thread per 8 output values; `hd` is a multiple
/// of 8
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn kv_to_16(
    mut kvd: DisjointSlice<Packed8<F16>>,
    k: &[f32],
    k_ld: u32,
    k_stride: u32,
    v: &[f32],
    v_ld: u32,
    v_stride: u32,
    batch: u32,
    hd: u32,
) {
    let t = thread::index_1d_u32(lc);
    let half = kvd.len() as u32 / 2;
    let (i, src, ld, stride) =
        if t.get() < half { (t.get(), k, k_ld, k_stride) } else { (t.get() - half, v, v_ld, v_stride) };
    let e = i * 8;
    let (d, b, row) = (e % hd, (e / hd) % batch, e / (hd * batch));
    let Some(mut out) = kvd.element_thread32(t) else { return };
    let s = b as usize * stride as usize + row as usize * ld as usize + d as usize;
    let Some(x) = ldg8(src, s) else { panic!() };
    out.write(Packed8::new([pack_f16(x[0], x[1]), pack_f16(x[2], x[3]), pack_f16(x[4], x[5]), pack_f16(x[6], x[7])]));
}

/// Merges the key splits of a split flash attention: one warp per (batch, row), lane `d
/// < D` owns output column `d`. Rows with no valid key in any split produce zeros. `N =
/// D / 32`; the split count is `part.len() / (batch * lq * (D + 2))`
#[inline(always)]
fn flash_combine<const D: usize, const N: usize>(
    mut o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    part: &[f32],
    batch: u32,
    lq: u32,
    t: u32,
) {
    let (w, lane) = (t / 32, (t % 32) as usize);
    if w >= batch * lq {
        return;
    }

    let (b, row) = (w / lq, w % lq);
    // part is `[splits, batch * lq, D + 2]`. Deriving the split count from its length
    // puts every element the loops read in bounds, which a per-split check can't do
    // without an early exit that stops the loops unrolling. A length truncated to 32 bits
    // only lowers the count, so it stays in bounds
    let step = batch * lq * (D + 2) as u32;
    let splits = part.len() as u32 / step;
    let at = |z: u32, c: usize| {
        // SAFETY: z < splits, w < batch * lq and c < D + 2, so the index is below splits *
        // step <= part.len()
        unsafe { *part.get_unchecked(z as usize * step as usize + w as usize * (D + 2) + c) }
    };

    let mut mx = f32::NEG_INFINITY;
    // four splits per pass keep four loads in flight, as nvcc chose for the C++ kernel
    for z in 0..splits {
        __unroll_config::<4>();
        mx = math::fmaxf(mx, at(z, D));
    }

    let (mut l, mut acc) = (0.0f32, [0.0f32; N]);
    if mx != f32::NEG_INFINITY {
        for z in 0..splits {
            __unroll_config::<4>();
            let mz = at(z, D);
            if mz == f32::NEG_INFINITY {
                continue;
            }
            let c = math::exp2f(mz - mx);
            l += c * at(z, D + 1);
            let mut i = 0;
            while i < N {
                __unroll_config::<0>();
                acc[i] += c * at(z, lane + i * 32);
                i += 1;
            }
        }
    }

    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
    let start = b as usize * o_stride as usize + row as usize * o_ld as usize;
    require(start + D <= o.len());
    let mut i = 0;
    while i < N {
        __unroll_config::<0>();
        // SAFETY: inside the checked row; this lane's column of this warp's row is written
        // by no other thread
        unsafe { *o.get_unchecked_mut(start + lane + i * 32) = acc[i] * inv };
        i += 1;
    }
}

/// [`flash_combine`] for head dim 32, one warp per output row
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn flash_combine_d32(o: DisjointSlice<f32>, o_ld: u32, o_stride: u32, part: &[f32], batch: u32, lq: u32) {
    let t = thread::index_1d_u32(lc);
    flash_combine::<32, 1>(o, o_ld, o_stride, part, batch, lq, t.get());
}
