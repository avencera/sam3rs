//! Fused attention (FlashAttention-2 style) for the BF16 precision mode
//!
//! FP32 q/k/v are rounded to FP16 as they are staged in shared memory (see
//! [`AttnIn::QK_F16`]), the softmax runs in FP32 with an online max, and P @ V runs on
//! FP16 tensor cores, accumulating each 64-key tile in FP16 before adding it to FP32
//! totals. Each warp owns 16 query rows; a block walks all keys in tiles of 64, so the
//! `[lq, lk]` score matrix never reaches global memory

use core::f32::consts::LOG2_E;
use core::ptr::addr_of_mut;

use cuda_device::convert::{bf16_to_f32, cvt_f32x2_f16x2};
use cuda_device::float::{ex2_approx_ftz_f32, fma_rn_f32};
// `#[unroll]` is only accepted in `#[kernel]` and `#[device]` bodies, and a generic
// `#[device]` fn is never inlined: it stays a `.func` whose shared and global pointers
// reach the body as generic ones, about 25% slower here. So the generic bodies below are
// plain `#[inline(always)]` fns, and their loops call the marker `#[unroll]` expands to,
// `__unroll_config::<0>()` (unroll fully), themselves
use cuda_device::thread::{
    __unroll_config, blockDim_x, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_y, sync_threads, threadIdx_x,
};
use cuda_device::wmma::{ldmatrix_x4_shared_u32, ldmatrix_x4_trans_shared_u32};
use cuda_device::{SharedArray, kernel, launch_bounds, ptx_asm, warp};

use crate::common::{Bf16, F16, grid_stride, ldg_u4, ldg4, math, pack_f16, stg_u4, stg2};
use crate::mma::{mma_bf16, mma_f16, smem, sts2, sts4};

const FA_WARPS: usize = 4;
/// Keys per tile
const FA_BN: usize = 64;
/// Upper bound on each thread's chunks when staging a tile, 8 for a D = 64 FP32 tile. The
/// staging loops run to it, fully unrolled, and skip chunks past the tile, a test that folds
/// away once inlining makes the tile size constant; a bound that is only constant after
/// inlining leaves the loop rolled (the unroll marker is checked before inlining, and LLVM
/// ignored a partial-unroll count). They load every chunk into registers before storing any,
/// because the shared stores are asm with a memory clobber that no load moves above: with
/// one load round trip per chunk, as the port first had it, BF16 encode was 2.4% slower
/// than nvcc's build, which unrolled 4x and issued the loads together
const STAGE_MAX_ITERS: usize = 8;
/// Score fragments per tile: one per 8 keys
const SJ: usize = FA_BN / 8;

#[inline(always)]
fn ex2(x: f32) -> f32 {
    ex2_approx_ftz_f32(x)
}

/// One float through the read-only path
#[inline(always)]
unsafe fn ldg_nc(p: *const f32) -> f32 {
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
unsafe fn ldg2_nc(p: *const f32) -> (f32, f32) {
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

/// A q, k or v element type and how it is staged into shared memory
pub trait AttnIn: Copy {
    /// Whether QK^T of this q type runs on FP16 operands with FP16 sums, else BF16 with
    /// FP32 sums. FP32 sources round to FP16, whose 3 extra mantissa bits over BF16
    /// also cover the error of summing each score's D products in FP16, at twice the
    /// FP32-accumulate rate on GeForce parts. BF16 sources stay BF16
    const QK_F16: bool;

    /// Rows `[0, rows)` of a `[rows, D]` tile into 16-bit shared memory with row stride
    /// `lds`, in the QK^T operand type; rows past `valid` are zero
    ///
    /// `threads` is the block size, a compile-time constant at every call site, not
    /// `blockDim_x()`, so the loop unrolls fully (see [`STAGE_MAX_ITERS`])
    ///
    /// # Safety
    /// Every thread of a block of exactly `threads` threads calls it; the source rows below
    /// `valid` are readable
    unsafe fn stage_qk<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    );

    /// The same into FP16, for V
    ///
    /// # Safety
    /// As [`AttnIn::stage_qk`]
    unsafe fn stage_f16<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    );
}

/// FP32 sources, rounded to FP16 four at a time
#[inline(always)]
unsafe fn stage_f32_to_f16<const D: usize>(
    dst: *mut u16,
    lds: usize,
    src: *const f32,
    ld: i64,
    valid: i32,
    rows: usize,
    threads: usize,
) {
    let n = rows * D / 4;
    let mut x = [[0.0; 4]; STAGE_MAX_ITERS];
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 4), (i % (D / 4)) * 4);
        if i < n && (r as i32) < valid {
            x[it] = unsafe { ldg4(src.offset(r as isize * ld as isize + c as isize)) };
        }
    }
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 4), (i % (D / 4)) * 4);
        if i < n {
            let x = x[it];
            unsafe { sts2(smem(dst.add(r * lds + c)), pack_f16(x[0], x[1]), pack_f16(x[2], x[3])) };
        }
    }
}

/// 16-bit sources copied 16 bytes at a time
#[inline(always)]
unsafe fn stage_copy16<const D: usize, T>(
    dst: *mut u16,
    lds: usize,
    src: *const T,
    ld: i64,
    valid: i32,
    rows: usize,
    threads: usize,
) {
    let n = rows * D / 8;
    let mut x = [[0; 4]; STAGE_MAX_ITERS];
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n && (r as i32) < valid {
            x[it] = unsafe { ldg_u4(src.offset(r as isize * ld as isize + c as isize)) };
        }
    }
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n {
            let x = x[it];
            unsafe { sts4(smem(dst.add(r * lds + c)), x) };
        }
    }
}

impl AttnIn for f32 {
    const QK_F16: bool = true;

    #[inline(always)]
    unsafe fn stage_qk<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_f32_to_f16::<D>(dst, lds, src, ld, valid, rows, threads) };
    }

    #[inline(always)]
    unsafe fn stage_f16<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_f32_to_f16::<D>(dst, lds, src, ld, valid, rows, threads) };
    }
}

impl AttnIn for Bf16 {
    const QK_F16: bool = false;

    #[inline(always)]
    unsafe fn stage_qk<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_copy16::<D, Self>(dst, lds, src, ld, valid, rows, threads) };
    }

    /// BF16 to FP16 through FP32, eight values at a time
    #[inline(always)]
    unsafe fn stage_f16<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_bf16_to_f16::<D>(dst, lds, src, ld, valid, rows, threads) };
    }
}

/// BF16 sources converted to FP16 through FP32, eight values at a time
#[inline(always)]
unsafe fn stage_bf16_to_f16<const D: usize>(
    dst: *mut u16,
    lds: usize,
    src: *const Bf16,
    ld: i64,
    valid: i32,
    rows: usize,
    threads: usize,
) {
    let n = rows * D / 8;
    let mut x = [[0; 4]; STAGE_MAX_ITERS];
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n && (r as i32) < valid {
            x[it] = unsafe { ldg_u4(src.offset(r as isize * ld as isize + c as isize)) };
        }
    }
    for it in 0..STAGE_MAX_ITERS {
        __unroll_config::<0>();
        let i = it * threads + threadIdx_x() as usize;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        if i < n {
            let x = x[it];
            let f16 = |w: u32| pack_f16(bf16_to_f32(w as u16), bf16_to_f32((w >> 16) as u16));
            unsafe { sts4(smem(dst.add(r * lds + c)), [f16(x[0]), f16(x[1]), f16(x[2]), f16(x[3])]) };
        }
    }
}

impl AttnIn for F16 {
    const QK_F16: bool = true;

    #[inline(always)]
    unsafe fn stage_qk<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_copy16::<D, Self>(dst, lds, src, ld, valid, rows, threads) };
    }

    #[inline(always)]
    unsafe fn stage_f16<const D: usize>(
        dst: *mut u16,
        lds: usize,
        src: *const Self,
        ld: i64,
        valid: i32,
        rows: usize,
        threads: usize,
    ) {
        unsafe { stage_copy16::<D, Self>(dst, lds, src, ld, valid, rows, threads) };
    }
}

/// Shared tiles of one block: q `[16 * WARPS, D + 8]`, k and v `[FA_BN, D + 8]`, all 16
/// bits. The 8-element padding keeps fragment loads bank-conflict free
struct Tiles {
    qs: *mut u16,
    ks: *mut u16,
    vs: *mut u16,
}

/// A view of a batched matrix: batch entry `b` starts at `ptr + b * stride` and row `i`
/// of it at `i * ld`
#[derive(Clone, Copy)]
struct View<T> {
    ptr: T,
    ld: i64,
    stride: i64,
}

/// The box relative position bias of the DETR decoder's image cross-attention
///
/// For head `h = blockIdx.y`, query row `1 + q` and key `y * side + x` it is
/// `dy[h, q, y] + dx[h, q, x]`; row 0 (the presence token) gets none. `side` must be
/// even, so each thread's key pair shares a grid row
#[derive(Clone, Copy)]
struct Rpb {
    dx: *const f32,
    dy: *const f32,
    side: i32,
}

/// The attention of one block of `16 * WARPS` query rows
///
/// Batch entry `blockIdx.y`; `scale_log2 = log2(e) / sqrt(D)`. Keys with
/// `key_pad[j] != 0` are skipped, and rows with no valid key produce zeros, like the
/// softmax kernel. With `part`, block z covers keys `[z * kv_chunk, (z + 1) * kv_chunk)`
/// only and writes its unnormalized accumulators, row max and row sum to
/// `part[z][batch][row][D + 2]` for [`flash_combine_d32`], so few query rows over many
/// keys still fill the GPU. WARPS sets the query rows per block: every block re-reads
/// all of K and V, so taller blocks cut that traffic. `KK = D / 16`, `DJ = D / 8`
#[inline(always)]
unsafe fn flash_attn<
    const D: usize,
    const KK: usize,
    const DJ: usize,
    const WARPS: usize,
    const RPB: bool,
    T: AttnIn,
    TK: AttnIn,
    TV: AttnIn,
>(
    tiles: Tiles,
    o: View<*mut f32>,
    q: View<*const T>,
    k: View<*const TK>,
    v: View<*const TV>,
    lq: i32,
    lk: i32,
    scale_log2: f32,
    key_pad: *const u8,
    rpb: Rpb,
    part: *mut f32,
    kv_chunk: i32,
) {
    let bm = 16 * WARPS;
    let lds = D + 8;
    // FP32 sources are staged 4 values per chunk, so a tile is rows * D / 4 chunks at most
    const { assert!(16 * WARPS * D / 4 <= STAGE_MAX_ITERS * 32 * WARPS && FA_BN * D / 4 <= STAGE_MAX_ITERS * 32 * WARPS) };
    let Tiles { qs, ks, vs } = tiles;

    let b = blockIdx_y() as i64;
    let q0 = (blockIdx_x() as usize * bm) as i32;
    let qp = unsafe { q.ptr.offset((b * q.stride + q0 as i64 * q.ld) as isize) };
    let kp = unsafe { k.ptr.offset((b * k.stride) as isize) };
    let vp = unsafe { v.ptr.offset((b * v.stride) as isize) };
    let op = unsafe { o.ptr.offset((b * o.stride) as isize) };
    let (warp, lane) = (threadIdx_x() as usize / 32, threadIdx_x() as usize % 32);
    let (g, t, wr) = (lane / 4, lane % 4, warp * 16);

    unsafe { T::stage_qk::<D>(qs, lds, qp, q.ld, lq - q0, bm, 32 * WARPS) };
    sync_threads();
    let mut qa = [[0u32; 4]; KK];
    let mut kk = 0;
    while kk < KK {
        __unroll_config::<0>();
        let p = unsafe { qs.add((wr + g) * lds + kk * 16 + t * 2) };
        qa[kk] = unsafe {
            [
                *(p as *const u32),
                *(p.add(8 * lds) as *const u32),
                *(p.add(8) as *const u32),
                *(p.add(8 * lds + 8) as *const u32),
            ]
        };
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
                let base = (b * (lq - 1) as i64 + (row - 1) as i64) * rpb.side as i64;
                rdx[r] = unsafe { rpb.dx.offset(base as isize) };
                rdy[r] = unsafe { rpb.dy.offset(base as isize) };
            }
            r += 1;
        }
    }

    let (mut k_begin, mut k_end) = (0, lk);
    if !part.is_null() {
        k_begin = blockIdx_z() as i32 * kv_chunk;
        k_end = lk.min(k_begin + kv_chunk);
    }
    let mut k0 = k_begin;
    while k0 < k_end {
        sync_threads();
        unsafe {
            TK::stage_qk::<D>(ks, lds, kp.offset((k0 as i64 * k.ld) as isize), k.ld, k_end - k0, FA_BN, 32 * WARPS);
            TV::stage_f16::<D>(vs, lds, vp.offset((k0 as i64 * v.ld) as isize), v.ld, k_end - k0, FA_BN, 32 * WARPS);
        }
        sync_threads();

        let mut s = [[0.0f32; 4]; SJ];
        let mut sh = [[0u32; 2]; SJ];
        let mut j = 0;
        while j < SJ {
            __unroll_config::<0>();
            let mut kk = 0;
            while kk < KK {
                __unroll_config::<0>();
                // matrices: keys j*8.. at k and k+8, then keys (j+1)*8.. at k and k+8
                let (mi, ri) = (lane / 8, lane % 8);
                let p = unsafe { ks.add(((j + mi / 2) * 8 + ri) * lds + kk * 16 + (mi & 1) * 8) };
                let bf = unsafe { ldmatrix_x4_shared_u32(smem(p)) };
                if T::QK_F16 {
                    sh[j] = unsafe { mma_f16(sh[j], qa[kk], bf[0], bf[1]) };
                    sh[j + 1] = unsafe { mma_f16(sh[j + 1], qa[kk], bf[2], bf[3]) };
                } else {
                    s[j] = unsafe { mma_bf16(s[j], qa[kk], bf[0], bf[1]) };
                    s[j + 1] = unsafe { mma_bf16(s[j + 1], qa[kk], bf[2], bf[3]) };
                }
                kk += 1;
            }
            j += 2;
        }
        if T::QK_F16 {
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let (lo, hi) = (cvt_f32x2_f16x2(sh[j][0]), cvt_f32x2_f16x2(sh[j][1]));
                s[j] = [lo.0, lo.1, hi.0, hi.1];
                j += 1;
            }
        }

        let mut corr = [0.0f32; 2];
        // a whole tile of real keys without padding or bias: the softmax needs no masking
        // or infinity guards, and the scale folds into the exponent's FMA. A row max of
        // -inf only comes from the first tile, where corr = ex2(-inf) = 0 is right
        if !RPB && key_pad.is_null() && k0 + FA_BN as i32 <= k_end {
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
            let (mut gy, mut gx) = (0i32, 0i32);
            if RPB {
                gy = k0 / rpb.side;
                gx = k0 - gy * rpb.side + t as i32 * 2;
                while gx >= rpb.side {
                    gx -= rpb.side;
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
                            let dy = unsafe { ldg_nc(rdy[r].add(gy as usize)) };
                            let dx = unsafe { ldg2_nc(rdx[r].add(gx as usize)) };
                            bias[r] = (dy + dx.0, dy + dx.1);
                        }
                        r += 1;
                    }
                    gx += 8;
                    while gx >= rpb.side {
                        gx -= rpb.side;
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
                    if col >= k_end || (!key_pad.is_null() && unsafe { *key_pad.add(col as usize) } != 0) {
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

        // the score fragments of two adjacent 8-key tiles form the A fragment of P. P @ V
        // runs in FP16 with FP16 accumulation over this tile's 64 keys (p <= 1, so the
        // partial sums stay within 64 |v|), then joins the FP32 accumulator
        let mut pacc = [[0u32; 2]; DJ];
        let mut kk = 0;
        while kk < FA_BN / 16 {
            __unroll_config::<0>();
            let (s0, s1) = (s[2 * kk], s[2 * kk + 1]);
            let pa = [pack_f16(s0[0], s0[1]), pack_f16(s0[2], s0[3]), pack_f16(s1[0], s1[1]), pack_f16(s1[2], s1[3])];
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                let (mi, ri) = (lane / 8, lane % 8);
                let p = unsafe { vs.add((kk * 16 + (mi & 1) * 8 + ri) * lds + (dj + mi / 2) * 8) };
                let bf = unsafe { ldmatrix_x4_trans_shared_u32(smem(p)) };
                pacc[dj] = unsafe { mma_f16(pacc[dj], pa, bf[0], bf[1]) };
                pacc[dj + 1] = unsafe { mma_f16(pacc[dj + 1], pa, bf[2], bf[3]) };
                dj += 2;
            }
            kk += 1;
        }
        let mut dj = 0;
        while dj < DJ {
            __unroll_config::<0>();
            let (lo, hi) = (cvt_f32x2_f16x2(pacc[dj][0]), cvt_f32x2_f16x2(pacc[dj][1]));
            acc[dj][0] += lo.0;
            acc[dj][1] += lo.1;
            acc[dj][2] += hi.0;
            acc[dj][3] += hi.1;
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
    if !part.is_null() {
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let row = q0 + (wr + g + r * 8) as i32;
            if row < lq {
                let i = ((blockIdx_z() as i64 * gridDim_y() as i64 + b) * lq as i64 + row as i64) * (D + 2) as i64;
                let out = unsafe { part.add(i as usize) };
                let mut dj = 0;
                while dj < DJ {
                    __unroll_config::<0>();
                    unsafe { stg2(out.add(dj * 8 + t * 2), acc[dj][2 * r], acc[dj][2 * r + 1]) };
                    dj += 1;
                }
                if t == 0 {
                    unsafe {
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
    let mut r = 0;
    while r < 2 {
        __unroll_config::<0>();
        let row = q0 + (wr + g + r * 8) as i32;
        if row < lq {
            let out = unsafe { op.offset((row as i64 * o.ld) as isize) };
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
        Tiles { qs: addr_of_mut!(QS) as *mut u16, ks: addr_of_mut!(KS) as *mut u16, vs: addr_of_mut!(VS) as *mut u16 }
    }};
}

/// Expands to a plain flash attention kernel for head dim `$d` and input type `$t`
macro_rules! flash_attn_kernel {
    ($name:ident, $d:literal, $t:ty) => {
        #[kernel]
        #[launch_bounds(128)]
        pub unsafe fn $name(
            o: *mut f32,
            o_ld: i64,
            o_stride: i64,
            q: *const $t,
            q_ld: i64,
            q_stride: i64,
            k: *const $t,
            k_ld: i64,
            k_stride: i64,
            v: *const $t,
            v_ld: i64,
            v_stride: i64,
            lq: i32,
            lk: i32,
            scale_log2: f32,
            key_pad: *const u8,
            part: *mut f32,
            kv_chunk: i32,
        ) {
            let rpb = Rpb { dx: core::ptr::null(), dy: core::ptr::null(), side: 0 };
            unsafe {
                flash_attn::<$d, { $d / 16 }, { $d / 8 }, FA_WARPS, false, $t, $t, $t>(
                    tiles!(16 * FA_WARPS, $d),
                    View { ptr: o, ld: o_ld, stride: o_stride },
                    View { ptr: q, ld: q_ld, stride: q_stride },
                    View { ptr: k, ld: k_ld, stride: k_stride },
                    View { ptr: v, ld: v_ld, stride: v_stride },
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
#[launch_bounds(128)]
pub unsafe fn flash_attn_rpb_d32(
    o: *mut f32,
    o_ld: i64,
    o_stride: i64,
    q: *const f32,
    q_ld: i64,
    q_stride: i64,
    k: *const f32,
    k_ld: i64,
    k_stride: i64,
    v: *const f32,
    v_ld: i64,
    v_stride: i64,
    lq: i32,
    lk: i32,
    scale_log2: f32,
    rpb_dx: *const f32,
    rpb_dy: *const f32,
    side: i32,
    part: *mut f32,
    kv_chunk: i32,
) {
    unsafe {
        flash_attn::<32, 2, 4, FA_WARPS, true, f32, f32, f32>(
            tiles!(16 * FA_WARPS, 32),
            View { ptr: o, ld: o_ld, stride: o_stride },
            View { ptr: q, ld: q_ld, stride: q_stride },
            View { ptr: k, ld: k_ld, stride: k_stride },
            View { ptr: v, ld: v_ld, stride: v_stride },
            lq,
            lk,
            scale_log2,
            core::ptr::null(),
            Rpb { dx: rpb_dx, dy: rpb_dy, side },
            part,
            kv_chunk,
        )
    };
}

/// FP32 q with K and V converted once by [`kv_to_16`], in 128-row blocks
#[kernel]
#[launch_bounds(256)]
pub unsafe fn flash_attn_kv16_d32(
    o: *mut f32,
    o_ld: i64,
    o_stride: i64,
    q: *const f32,
    q_ld: i64,
    q_stride: i64,
    k: *const F16,
    k_ld: i64,
    k_stride: i64,
    v: *const F16,
    v_ld: i64,
    v_stride: i64,
    lq: i32,
    lk: i32,
    scale_log2: f32,
    key_pad: *const u8,
) {
    unsafe {
        flash_attn::<32, 2, 4, 8, false, f32, F16, F16>(
            tiles!(16 * 8, 32),
            View { ptr: o, ld: o_ld, stride: o_stride },
            View { ptr: q, ld: q_ld, stride: q_stride },
            View { ptr: k, ld: k_ld, stride: k_stride },
            View { ptr: v, ld: v_ld, stride: v_stride },
            lq,
            lk,
            scale_log2,
            key_pad,
            Rpb { dx: core::ptr::null(), dy: core::ptr::null(), side: 0 },
            core::ptr::null_mut(),
            0,
        )
    };
}

/// Batch-strided FP32 views `k` and `v [batch][lk, hd]` into contiguous FP16 `kd` and
/// `vd [lk, batch * hd]`, 8 values per thread, for [`flash_attn_kv16_d32`]: rounding them
/// once instead of in every query block
#[kernel]
pub unsafe fn kv_to_16(
    kd: *mut F16,
    vd: *mut F16,
    k: *const f32,
    k_ld: i64,
    k_stride: i64,
    v: *const f32,
    v_ld: i64,
    v_stride: i64,
    lk: i32,
    batch: i32,
    hd: i32,
) {
    let n = lk as i64 * batch as i64 * hd as i64 / 8;
    let (mut i, stride) = grid_stride();
    while i < n {
        let e = i * 8;
        let (d, b) = (e % hd as i64, (e / hd as i64) % batch as i64);
        let row = e / (hd as i64 * batch as i64);
        unsafe {
            let ks = k.offset((b * k_stride + row * k_ld + d) as isize);
            let vs = v.offset((b * v_stride + row * v_ld + d) as isize);
            let (k0, k1, v0, v1) = (ldg4(ks), ldg4(ks.add(4)), ldg4(vs), ldg4(vs.add(4)));
            let pk = |a: [f32; 4], b: [f32; 4]| {
                [pack_f16(a[0], a[1]), pack_f16(a[2], a[3]), pack_f16(b[0], b[1]), pack_f16(b[2], b[3])]
            };
            stg_u4(kd.add(e as usize), pk(k0, k1));
            stg_u4(vd.add(e as usize), pk(v0, v1));
        }
        i += stride;
    }
}

/// Merges the key splits of a split flash attention: one warp per (batch, row), lane
/// `d < D` owns output column `d`. Rows with no valid key in any split produce zeros.
/// `N = D / 32`
#[inline(always)]
unsafe fn flash_combine<const D: usize, const N: usize>(
    o: *mut f32,
    o_ld: i64,
    o_stride: i64,
    part: *const f32,
    splits: i32,
    batch: i32,
    lq: i32,
) {
    let w = ((blockIdx_x() * blockDim_x() + threadIdx_x()) / 32) as i32;
    let lane = (threadIdx_x() % 32) as usize;
    if w >= batch * lq {
        return;
    }

    let (b, row) = (w / lq, w % lq);
    let step = batch as i64 * lq as i64 * (D + 2) as i64;
    let p = unsafe { part.add(w as usize * (D + 2)) };
    let mut mx = f32::NEG_INFINITY;
    // four splits per pass keep four loads in flight, as nvcc chose for the C++ kernel
    for z in 0..splits {
        __unroll_config::<4>();
        mx = math::fmaxf(mx, unsafe { *p.offset((z as i64 * step + D as i64) as isize) });
    }

    let (mut l, mut acc) = (0.0f32, [0.0f32; N]);
    if mx != f32::NEG_INFINITY {
        for z in 0..splits {
            __unroll_config::<4>();
            let ps = unsafe { p.offset((z as i64 * step) as isize) };
            let mz = unsafe { *ps.add(D) };
            if mz == f32::NEG_INFINITY {
                continue;
            }
            let c = math::exp2f(mz - mx);
            l += c * unsafe { *ps.add(D + 1) };
            let mut i = 0;
            while i < N {
                __unroll_config::<0>();
                acc[i] += c * unsafe { *ps.add(lane + i * 32) };
                i += 1;
            }
        }
    }

    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
    let out = unsafe { o.offset((b as i64 * o_stride + row as i64 * o_ld) as isize) };
    let mut i = 0;
    while i < N {
        __unroll_config::<0>();
        unsafe { *out.add(lane + i * 32) = acc[i] * inv };
        i += 1;
    }
}

#[kernel]
pub unsafe fn flash_combine_d32(
    o: *mut f32,
    o_ld: i64,
    o_stride: i64,
    part: *const f32,
    splits: i32,
    batch: i32,
    lq: i32,
) {
    unsafe { flash_combine::<32, 1>(o, o_ld, o_stride, part, splits, batch, lq) };
}
