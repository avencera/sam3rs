//! Fused attention for head dim 256, the video tracker's single-head memory attention
//!
//! FlashAttention-2 in FA2's sm86/sm89 shape for this head dim: 64 query rows per block of
//! 4 warps, 64-key tiles, and Q, K and V tiles in 99 KB of dynamic shared memory. Q stays
//! in shared memory and its fragments are reloaded for every key tile, since holding them
//! in registers next to the 16x256 FP32 accumulator would spill. Tiles arrive by
//! `cp.async`, so each V tile loads under QK^T and the next K tile under P @ V
//!
//! q, k and v are FP16, written by the RoPE pass. QK^T accumulates in FP32: a score sums
//! 256 products, too many to round in FP16. P @ V accumulates each tile's 64 keys in FP16
//! (p <= 1, so partial sums stay within 64 |v|), 64 output columns at a time, then joins
//! the FP32 totals, as the d64 kernel does. Layout and bounds checks follow
//! [`crate::attention`]

use cuda_device::float::fma_rn_f32;
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_y, threadIdx_x};
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_contract, thread, warp};

use crate::attention::{Batched, FA_BN, Rows, SJ, ex2, require, rows_fit};
use crate::common::{F16, math, stg2};
use crate::mma::{FragA, Mma16, SharedTile, cp_async_commit, cp_async_wait_all, cp_async16, mma_f16_acc_f32};

const D: usize = 256;
/// Padded rows: 528 bytes, so the 8 rows an `ldmatrix` reads fall in distinct banks
const LD: usize = D + 8;
const WARPS: usize = 4;
const THREADS: usize = 32 * WARPS;
/// Query rows per block
const BM: usize = 16 * WARPS;
const KK: usize = D / 16;
const DJ: usize = D / 8;
/// Output fragments per FP16 P @ V pass: 64 columns, 16 accumulator registers
const DG: usize = 8;
/// 16-byte chunks per thread to stage one 64-row tile
const CHUNKS: usize = FA_BN * D / 8 / THREADS;
/// Q, K and V tiles of 64 padded rows: 101376 bytes, the sm_86 to sm_120 opt-in limit
pub const FLASH_D256_SMEM: usize = 3 * 64 * LD * 2;

/// Starts copying the first 64 rows of `src` into `dst`, zeros past `src.rows`
///
/// # Safety
/// Every thread of the block calls it; `src.rows >= 1`, checked for width D. `dst` is not
/// read until the copies are waited on and a barrier has followed
#[inline(always)]
unsafe fn load_tile(dst: SharedTile<F16, LD>, src: Rows<F16>) {
    let tid = threadIdx_x() as usize;
    for it in 0..CHUNKS {
        __unroll_config::<0>();
        // one warp per 512-byte row, so each copy instruction covers one row
        let i = it * THREADS + tid;
        let (r, c) = (i / (D / 8), (i % (D / 8)) * 8);
        let full = (r as i32) < src.rows;
        // SAFETY: row r when it exists, else row 0 (rows >= 1), which zero fill doesn't
        // read; D and the row starts are multiples of 8 values in 16-byte aligned buffers
        let p = unsafe { src.at(if full { r } else { 0 }, c) };
        unsafe { cp_async16(dst.at(r, c), p as *const u8, full) };
    }
}

/// Flash attention for head dim 256 on FP16 q/k/v; see the module docs
///
/// Batch entry `blockIdx.y`; `scale_log2 = log2(e) / 16`. With a non-empty `part`, block
/// z covers keys `[z * kv_chunk, (z + 1) * kv_chunk)` and writes its unnormalized
/// accumulators, row max and row sum to `part[z][batch][row][258]` for
/// [`crate::attention::flash_combine_d256`] instead of `o`. Distinct (batch, row, column)
/// of `o` must be distinct elements under `o_ld` and `o_stride`
#[kernel]
#[launch_contract(domain = 3, block = (128, 1, 1), dynamic_shared = 101376)]
pub fn flash_attn_f16_d256(
    o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    q: &[F16],
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
    part: DisjointSlice<f32>,
    kv_chunk: u32,
) {
    let (mut o, mut part) = (o, part);
    let (q, k, v) = (
        Batched { data: q, ld: q_ld, stride: q_stride },
        Batched { data: k, ld: k_ld, stride: k_stride },
        Batched { data: v, ld: v_ld, stride: v_stride },
    );
    let (lq, lk) = (lq as i32, lk as i32);
    let smem = DynamicSharedArray::<u16, 16>::get();
    // SAFETY: the launch gives FLASH_D256_SMEM bytes, three tiles of 64 rows of LD
    let (qs, ks, vs) = unsafe {
        (
            SharedTile::<F16, LD>::new(smem),
            SharedTile::<F16, LD>::new(smem.add(64 * LD)),
            SharedTile::<F16, LD>::new(smem.add(2 * 64 * LD)),
        )
    };

    let b = blockIdx_y() as usize;
    let q0 = (blockIdx_x() as usize * BM) as i32;
    let (warp, lane) = (threadIdx_x() as usize / 32, threadIdx_x() as usize % 32);
    let (g, t, wr) = (lane / 4, lane % 4, warp * 16);
    let split = !part.is_empty();
    let (mut k_begin, mut k_end) = (0, lk);
    if split {
        k_begin = blockIdx_z() as i32 * kv_chunk as i32;
        k_end = lk.min(k_begin + kv_chunk as i32);
    }

    // everything the block reads or writes, checked at once up front as in the d64 kernel;
    // an empty key range has nothing to stage
    let nq = (lq - q0).min(BM as i32);
    let part_base = (blockIdx_z() as usize * gridDim_y() as usize + b) * lq as usize * (D + 2);
    let out_ok = if split {
        rows_fit(part.len(), part_base, q0, nq, (D + 2) as u32, D + 2)
    } else {
        rows_fit(o.len(), b * o_stride as usize, q0, nq, o_ld, D)
    };
    let (qr, kr, vr) =
        (q.rows(b, q0, nq, D), k.rows(b, k_begin, k_end - k_begin, D), v.rows(b, k_begin, k_end - k_begin, D));
    require(qr.is_some() & kr.is_some() & vr.is_some() & out_ok & (nq >= 1));
    let (Some(qr), Some(kr), Some(vr)) = (qr, kr, vr) else { return };

    let mut acc = [[0.0f32; 4]; DJ];
    let (mut m, mut l) = ([f32::NEG_INFINITY; 2], [0.0f32; 2]);
    if k_begin < k_end {
        // SAFETY: qr and kr hold at least one row; the first wait and barrier below order
        // these copies before any fragment load
        unsafe {
            load_tile(qs, qr);
            load_tile(ks, kr);
            cp_async_commit();
        }
    }

    let mut k0 = k_begin;
    while k0 < k_end {
        // K is in, and every warp is done with the last tile's V
        unsafe { cp_async_wait_all() };
        thread::sync_threads();
        // SAFETY: k0 < k_end, so vr has a row from k0 - k_begin on
        unsafe {
            load_tile(vs, vr.skip(k0 - k_begin));
            cp_async_commit();
        }

        let mut s = [[0.0f32; 4]; SJ];
        let mut kk = 0;
        while kk < KK {
            __unroll_config::<0>();
            // SAFETY: the whole warp runs this with warp-uniform offsets inside the tiles
            unsafe {
                let qa = qs.frag_a(wr, kk * 16);
                let mut j = 0;
                while j < SJ {
                    __unroll_config::<0>();
                    let kb = ks.frag_b_pair_nk(j * 8, kk * 16);
                    s[j] = mma_f16_acc_f32(s[j], qa, kb[0]);
                    s[j + 1] = mma_f16_acc_f32(s[j + 1], qa, kb[1]);
                    j += 2;
                }
            }
            kk += 1;
        }

        let mut corr = [0.0f32; 2];
        // a whole tile of real keys: no masking or infinity guards, and the scale folds
        // into the exponent's FMA. A row max of -inf only comes from the first tile, where
        // corr = ex2(-inf) = 0 is right
        if k0 + FA_BN as i32 <= k_end {
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
            // the last, partial tile: keys past k_end were staged as zeros and drop out here
            let mut mx = m;
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    let col = k0 + (j * 8 + t * 2 + (e & 1)) as i32;
                    let x = if col < k_end { s[j][e] * scale_log2 } else { f32::NEG_INFINITY };
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
        let mut pa = [FragA::<F16>::ZERO; FA_BN / 16];
        let mut kk = 0;
        while kk < FA_BN / 16 {
            __unroll_config::<0>();
            pa[kk] = FragA::<F16>::from_acc_pair(s[2 * kk], s[2 * kk + 1]);
            kk += 1;
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

        // V is in, and every warp is done with this tile's K
        unsafe { cp_async_wait_all() };
        thread::sync_threads();
        if k0 + (FA_BN as i32) < k_end {
            // SAFETY: rows from k0 + 64 - k_begin exist, as k0 + 64 < k_end
            unsafe {
                load_tile(ks, kr.skip(k0 + FA_BN as i32 - k_begin));
                cp_async_commit();
            }
        }

        let mut d0 = 0;
        while d0 < DJ {
            __unroll_config::<0>();
            let mut pacc = [<F16 as Mma16>::ZERO; DG];
            let mut kk = 0;
            while kk < FA_BN / 16 {
                __unroll_config::<0>();
                let mut dj = 0;
                while dj < DG {
                    __unroll_config::<0>();
                    // SAFETY: as for QK^T
                    unsafe {
                        let vb = vs.frag_b_pair_kn(kk * 16, (d0 + dj) * 8);
                        pacc[dj] = F16::mma(pacc[dj], pa[kk], vb[0]);
                        pacc[dj + 1] = F16::mma(pacc[dj + 1], pa[kk], vb[1]);
                    }
                    dj += 2;
                }
                kk += 1;
            }
            let mut dj = 0;
            while dj < DG {
                __unroll_config::<0>();
                let p = F16::acc_f32(pacc[dj]);
                acc[d0 + dj][0] += p[0];
                acc[d0 + dj][1] += p[1];
                acc[d0 + dj][2] += p[2];
                acc[d0 + dj][3] += p[3];
                dj += 1;
            }
            d0 += DG;
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
