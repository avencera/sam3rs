//! Memory cross-attention of the video tracker: queries and keys of 256, values of 64
//!
//! Upstream projects the 64-channel memory to 256 with `v_proj`, attends with one head of
//! 256 and applies `out_proj`. Softmax rows sum to 1, so `out_proj(P (M Wv^T + bv)) =
//! (P M) (Wo Wv)^T + Wo bv + bo`: this kernel computes `P M` on the memory itself and the
//! host folds the two projections into one 64 -> 256 linear. P @ V is a quarter of the
//! math and V a quarter of the bytes, and the 16x64 FP32 accumulator a warp holds,
//! against 16x256 in [`crate::attention_d256`], leaves room for Q's fragments in
//! registers and 8 warps a block, where the d256 kernel ran one warp per scheduler
//!
//! 128 query rows per block of 8 warps, 16 rows each, and 64-key tiles. Q is staged once
//! through the two K buffers, then held as A fragments. K and V tiles are double-buffered
//! by `cp.async`, so tile i + 1 loads during tile i's math. The kernel is bound by the
//! tensor pipe, and q.k^T is four fifths of its math, so scores sum in FP16 a 16-column
//! step at a time (twice the FP32-accumulate rate on GeForce) and the steps add up in
//! FP32 ([`QK_CHUNK`]). P @ V does the same over each 16 keys: with q.k^T at the FP16
//! rate, P @ V's FP32-accumulate HMMA was a third of the tensor time. q, k and v are
//! FP16; layout, bounds checks and the split-key partials follow
//! [`crate::attention_d256`], with rows of 64 + 2 for
//! [`crate::attention::flash_combine_d64`]

use cuda_device::float::fma_rn_f32;
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_y, threadIdx_x};
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_contract, thread, warp};

use crate::attention::{Batched, FA_BN, Rows, SJ, ex2, require, rows_fit};
use crate::common::{F16, math, stg2};
use crate::mma::{
    FragA, Mma16, SharedTile, cp_async_commit, cp_async_wait_all, cp_async_wait_one, cp_async16,
};

/// Query and key width
const DK: usize = 256;
/// Value and output width
const DV: usize = 64;
/// Padded rows (528 and 144 bytes), so the 8 rows an `ldmatrix` reads fall in distinct
/// banks
const LDK: usize = DK + 8;
const LDV: usize = DV + 8;
const WARPS: usize = 8;
const THREADS: usize = 32 * WARPS;
/// Query rows per block
const BM: usize = 16 * WARPS;
const KK: usize = DK / 16;
const DJ: usize = DV / 8;
/// 16-column steps of q.k each FP16 partial sums before it joins the FP32 score. One step
/// held the f64 kernel test to 7.6e-4 rel L2 (6.9e-5 with FP32 sums) under its 1e-3 bound;
/// two steps measured 1.08e-3
const QK_CHUNK: usize = 1;
// the step loop indexes qa[kc + i], so whole steps must tile the depth
const _: () = assert!(KK % QK_CHUNK == 0);
/// Two K tiles (which hold Q's 128 rows while it is staged) and two V tiles: 86016 bytes
pub const FLASH_MEM_SMEM: usize = 2 * FA_BN * LDK * 2 + 2 * FA_BN * LDV * 2;

/// Starts copying the first `ROWS` rows of `src`, `W` wide, into `dst`, zeros past
/// `src.rows`
///
/// # Safety
/// Every thread of the block calls it; `src.rows >= 1`, checked for width `W`. `dst` holds
/// `ROWS` rows and is not read until the copies are waited on and a barrier has followed
#[inline(always)]
unsafe fn stage<const W: usize, const LD: usize, const ROWS: usize>(dst: SharedTile<F16, LD>, src: Rows<F16>) {
    let tid = threadIdx_x() as usize;
    for it in 0..ROWS * W / 8 / THREADS {
        __unroll_config::<0>();
        let i = it * THREADS + tid;
        let (r, c) = (i / (W / 8), (i % (W / 8)) * 8);
        let full = (r as i32) < src.rows;
        // SAFETY: row r when it exists, else row 0 (rows >= 1), which zero fill doesn't
        // read; W and the row starts are multiples of 8 values in 16-byte aligned buffers
        let p = unsafe { src.at(if full { r } else { 0 }, c) };
        unsafe { cp_async16(dst.at(r, c), p as *const u8, full) };
    }
}

/// Flash attention of FP16 `q [lq, 256]` over `k [lk, 256]` and `v [lk, 64]`; see the
/// module docs
///
/// Batch entry `blockIdx.y`; `scale_log2 = log2(e) / 16`. With a non-empty `part`, block
/// z covers keys `[z * kv_chunk, (z + 1) * kv_chunk)` and writes its unnormalized
/// accumulators, row max and row sum to `part[z][batch][row][66]` instead of `o`.
/// Distinct (batch, row, column) of `o` must be distinct elements under `o_ld` and
/// `o_stride`
#[kernel]
#[launch_contract(domain = 3, block = (256, 1, 1), dynamic_shared = 86016)]
pub fn flash_attn_f16_qk256_v64(
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
    // SAFETY: the launch gives FLASH_MEM_SMEM bytes: two K tiles of 64 rows of LDK, then
    // two V tiles of 64 rows of LDV. Q's 128 rows of LDK are the two K tiles end to end
    let (k0s, k1s, v0s, v1s) = unsafe {
        (
            SharedTile::<F16, LDK>::new(smem),
            SharedTile::<F16, LDK>::new(smem.add(FA_BN * LDK)),
            SharedTile::<F16, LDV>::new(smem.add(2 * FA_BN * LDK)),
            SharedTile::<F16, LDV>::new(smem.add(2 * FA_BN * LDK + FA_BN * LDV)),
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

    // everything the block reads or writes, checked at once up front as in the d256 kernel
    let nq = (lq - q0).min(BM as i32);
    let part_base = (blockIdx_z() as usize * gridDim_y() as usize + b) * lq as usize * (DV + 2);
    let out_ok = if split {
        rows_fit(part.len(), part_base, q0, nq, (DV + 2) as u32, DV + 2)
    } else {
        rows_fit(o.len(), b * o_stride as usize, q0, nq, o_ld, DV)
    };
    let (qr, kr, vr) =
        (q.rows(b, q0, nq, DK), k.rows(b, k_begin, k_end - k_begin, DK), v.rows(b, k_begin, k_end - k_begin, DV));
    require(qr.is_some() & kr.is_some() & vr.is_some() & out_ok & (nq >= 1));
    let (Some(qr), Some(kr), Some(vr)) = (qr, kr, vr) else { return };

    let mut qa = [FragA::<F16>::ZERO; KK];
    if k_begin < k_end {
        // SAFETY: qr, kr and vr hold at least one row. Q's copies are waited on and
        // fenced before its fragments load, and a second barrier keeps K's first copies
        // off the Q rows until every warp has its fragments
        unsafe {
            stage::<DK, LDK, BM>(k0s, qr);
            cp_async_commit();
            cp_async_wait_all();
        }
        thread::sync_threads();
        let mut kk = 0;
        while kk < KK {
            __unroll_config::<0>();
            // SAFETY: the whole warp, warp-uniform offsets inside Q's 128 rows
            qa[kk] = unsafe { k0s.frag_a(wr, kk * 16) };
            kk += 1;
        }
        thread::sync_threads();
        unsafe {
            stage::<DK, LDK, FA_BN>(k0s, kr);
            stage::<DV, LDV, FA_BN>(v0s, vr);
            cp_async_commit();
        }
    }

    let mut acc = [[0.0f32; 4]; DJ];
    let (mut m, mut l) = ([f32::NEG_INFINITY; 2], [0.0f32; 2]);
    let mut odd = false;
    let mut k0 = k_begin;
    while k0 < k_end {
        let (ks, vs) = if odd { (k1s, v1s) } else { (k0s, v0s) };
        let next = k0 + FA_BN as i32;
        if next < k_end {
            // the other buffers were freed by the barrier that ended the last tile
            // SAFETY: rows from next - k_begin exist, as next < k_end
            unsafe {
                let (kn, vn) = if odd { (k0s, v0s) } else { (k1s, v1s) };
                stage::<DK, LDK, FA_BN>(kn, kr.skip(next - k_begin));
                stage::<DV, LDV, FA_BN>(vn, vr.skip(next - k_begin));
                cp_async_commit();
                cp_async_wait_one();
            }
        } else {
            unsafe { cp_async_wait_all() };
        }
        // this tile's K and V are in for every thread
        thread::sync_threads();

        // each QK_CHUNK * 16 columns of q.k sum in FP16 at twice the FP32-accumulate
        // rate, and the partials add up in FP32
        let mut s = [[0.0f32; 4]; SJ];
        let mut kc = 0;
        while kc < KK {
            __unroll_config::<0>();
            let mut part = [<F16 as Mma16>::ZERO; SJ];
            let mut i = 0;
            while i < QK_CHUNK {
                __unroll_config::<0>();
                let kk = kc + i;
                let mut j = 0;
                while j < SJ {
                    __unroll_config::<0>();
                    // SAFETY: the whole warp runs this with warp-uniform offsets inside
                    // the tile
                    unsafe {
                        let kb = ks.frag_b_pair_nk(j * 8, kk * 16);
                        part[j] = F16::mma(part[j], qa[kk], kb[0]);
                        part[j + 1] = F16::mma(part[j + 1], qa[kk], kb[1]);
                    }
                    j += 2;
                }
                i += 1;
            }
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let c = F16::acc_f32(part[j]);
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    s[j][e] = if kc == 0 { c[e] } else { s[j][e] + c[e] };
                    e += 1;
                }
                j += 1;
            }
            kc += QK_CHUNK;
        }

        let mut corr = [0.0f32; 2];
        // a whole tile of real keys: no masking or infinity guards, and the scale folds
        // into the exponent's FMA. A row max of -inf only comes from the first tile, where
        // corr = ex2(-inf) = 0 is right
        if next <= k_end {
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

        let mut dj = 0;
        while dj < DJ {
            __unroll_config::<0>();
            acc[dj][0] *= corr[0];
            acc[dj][1] *= corr[0];
            acc[dj][2] *= corr[1];
            acc[dj][3] *= corr[1];
            dj += 1;
        }
        let mut kk = 0;
        while kk < FA_BN / 16 {
            __unroll_config::<0>();
            let pa = FragA::<F16>::from_acc_pair(s[2 * kk], s[2 * kk + 1]);
            // each 16 keys' share of P @ V sums in FP16, then joins the FP32 accumulator
            let mut part = [<F16 as Mma16>::ZERO; DJ];
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                // SAFETY: as for QK^T
                unsafe {
                    let vb = vs.frag_b_pair_kn(kk * 16, dj * 8);
                    part[dj] = F16::mma(part[dj], pa, vb[0]);
                    part[dj + 1] = F16::mma(part[dj + 1], pa, vb[1]);
                }
                dj += 2;
            }
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                let c = F16::acc_f32(part[dj]);
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    acc[dj][e] += c[e];
                    e += 1;
                }
                dj += 1;
            }
            kk += 1;
        }
        // every warp is done with this tile's buffers before the next tile refills them
        thread::sync_threads();
        odd = !odd;
        k0 = next;
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
        let pp = part.as_mut_ptr().wrapping_add(part_base + q0 as u32 as usize * (DV + 2));
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let row = wr + g + r * 8;
            if (row as i32) < nq {
                // SAFETY: row < nq, inside the checked rows; DV + 2 is even, so each pair
                // is 8-byte aligned
                unsafe {
                    let out = pp.add(row * (DV + 2));
                    let mut dj = 0;
                    while dj < DJ {
                        __unroll_config::<0>();
                        stg2(out.add(dj * 8 + t * 2), acc[dj][2 * r], acc[dj][2 * r + 1]);
                        dj += 1;
                    }
                    if t == 0 {
                        *out.add(DV) = m[r];
                        *out.add(DV + 1) = l[r];
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
