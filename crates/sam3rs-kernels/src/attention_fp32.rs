//! Fused attention for the FP32 precision mode
//!
//! The online-softmax walk of [`attention`](crate::attention) on the FP32 pipes: every
//! product is an FP32 FMA, with no tensor-core rounding, so the mode keeps its "no TF32,
//! no BF16" contract while never writing the `[lq, lk]` score matrix. On GeForce parts
//! TF32 tensor cores run at the FP32 FMA rate, so split-TF32 emulation would cost three
//! times this kernel's math for the same accuracy
//!
//! A block of 128 threads owns 64 query rows and walks the keys 64 at a time. Thread (ty,
//! tx), ty in `0..8` and tx in `0..16`, holds an 8x4 piece of each tile's scores (rows
//! `ty + 8i`, keys `tx + 16j`) and of the output (rows `ty + 8i`, columns `4tx..4tx +
//! 4`), so a row's 16 threads are one half-warp and its max is four shuffles. Both
//! products read 16-byte vectors from shared memory, 12 loads for 128 FMAs
//!
//! The piece size is what bounds the kernel: a 16-byte warp load costs at least four
//! shared wavefronts, even when lanes broadcast, so an R x C piece reads 4(R + C) floats
//! for 4RC FMAs at 32 floats a clock against 128 FMAs, and runs at most RC / (4(R + C))
//! of the FMA rate: 50% for 4x4, 67% for 8x4. RTX 50 and 40 series SMs have 100 KB of
//! shared memory, 99 KB a block, so the tiles stay under half of it for two blocks an SM:
//! P takes K's place once the scores are done. 8x8 pieces in 128-row blocks needed 105 KB
//! and 255 registers

use cuda_device::float::fma_rn_f32;
use cuda_device::ptx_asm;
use cuda_device::shared::{DynamicSharedArray, cvta_generic_to_shared_u32};
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, threadIdx_x};
use cuda_device::{DisjointSlice, kernel, launch_contract, thread, warp};

use crate::attention::{Batched, Rows, ex2, require, rows_fit};
use crate::common::{ldg4, math};

/// Query rows a block
const BM: usize = 64;
/// Keys a tile
const BN: usize = 64;
const THREADS: usize = 128;
/// Rows and keys (or output columns) each thread holds, threads along a row, and the step
/// between a thread's rows
const TR: usize = 8;
const TC: usize = 4;
const TX: usize = 16;
const RS: usize = BM / TR;
/// Row strides in floats. K's 68 puts consecutive rows 4 banks apart, so a quarter-warp's
/// 16-byte loads of 8 consecutive rows cover all 32 banks. Each quarter-warp reads one Q,
/// V or P address, a broadcast, so those need no padding; P shares K's tile and stride,
/// which leaves its scalar stores 2-way conflicted
const LD_Q: usize = 64;
const LD_K: usize = 68;
const LD_V: usize = 64;
/// Dynamic shared memory [`flash_attn_f32_d64`] needs: its Q, K (then P) and V tiles,
/// 50,176 bytes, which with the 1 KB each block reserves fills exactly 100 KB with two
pub const F32_D64_SMEM_BYTES: u32 = ((BM * LD_Q + BN * LD_K + BN * LD_V) * 4) as u32;

/// The dynamic shared memory the launch gave this block, in bytes
#[inline(always)]
fn dynamic_smem_bytes() -> u32 {
    let n: u32;
    unsafe { ptx_asm!("mov.u32 %0, %%dynamic_smem_size;", out("=r") n) };
    n
}

/// A row-major FP32 tile in shared memory with rows of `LD` floats, addressed through the
/// 32-bit shared window
#[derive(Clone, Copy)]
struct F32Tile<const LD: usize>(u32);

impl<const LD: usize> F32Tile<LD> {
    /// # Safety
    /// `base` points to a 16-byte aligned shared array of as many rows of `LD` floats as
    /// callers address
    #[inline(always)]
    unsafe fn new(base: *mut f32) -> Self {
        Self(unsafe { cvta_generic_to_shared_u32(base as *const u8) })
    }

    #[inline(always)]
    fn addr(self, row: usize, col: usize) -> u32 {
        self.0 + ((row * LD + col) * 4) as u32
    }

    /// Floats `[col, col + 4)` of `row`
    ///
    /// # Safety
    /// Inside the tile, `col` a multiple of 4, and ordered after their stores by a barrier
    #[inline(always)]
    unsafe fn load4(self, row: usize, col: usize) -> [f32; 4] {
        let (a, b, c, d): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!(
                "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];",
                out("=f") a,
                out("=f") b,
                out("=f") c,
                out("=f") d,
                in("r") self.addr(row, col),
            );
        }
        [a, b, c, d]
    }

    /// # Safety
    /// As [`F32Tile::load4`], and no other thread touches these floats before the next
    /// barrier
    #[inline(always)]
    unsafe fn store4(self, row: usize, col: usize, v: [f32; 4]) {
        unsafe {
            ptx_asm!(
                "st.shared.v4.f32 [%0], {%1, %2, %3, %4};",
                in("r") self.addr(row, col),
                in("f") v[0],
                in("f") v[1],
                in("f") v[2],
                in("f") v[3],
                clobber("memory"),
            );
        }
    }

    /// # Safety
    /// As [`F32Tile::store4`], any column
    #[inline(always)]
    unsafe fn store(self, row: usize, col: usize, v: f32) {
        unsafe { ptx_asm!("st.shared.f32 [%0], %1;", in("r") self.addr(row, col), in("f") v, clobber("memory")) };
    }
}

/// Sixteen bytes to a 16-byte aligned global address
///
/// # Safety
/// `p` is 16-byte aligned and valid for four writes
#[inline(always)]
unsafe fn stg4(p: *mut f32, v: [f32; 4]) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.v4.f32 [g], {%1, %2, %3, %4}; }",
            in("l") p as u64,
            in("f") v[0],
            in("f") v[1],
            in("f") v[2],
            in("f") v[3],
            clobber("memory"),
        );
    }
}

/// `BN` rows of `[_, 64]` FP32 into `dst`; rows past `src.rows` are zero. Each of the
/// 128 threads moves 8 chunks of 4 floats, all loaded before any is stored
///
/// # Safety
/// Every thread of the block calls it between barriers, and `dst` has `BN` rows
#[inline(always)]
unsafe fn stage<const LD: usize>(dst: F32Tile<LD>, src: Rows<f32>) {
    const ITERS: usize = BN * 64 / 4 / THREADS;
    let mut x = [[0.0; 4]; ITERS];
    for it in 0..ITERS {
        __unroll_config::<0>();
        let i = it * THREADS + threadIdx_x() as usize;
        let (r, c) = (i / 16, (i % 16) * 4);
        if (r as i32) < src.rows {
            // SAFETY: r < src.rows and c + 4 <= 64, inside the checked rows; row starts are
            // multiples of 4 floats, and buffers 16-byte aligned
            x[it] = unsafe { ldg4(src.at(r, c)) };
        }
    }
    for it in 0..ITERS {
        __unroll_config::<0>();
        let i = it * THREADS + threadIdx_x() as usize;
        // SAFETY: (i / 16, i % 16 * 4) is this thread's alone, below BN rows
        unsafe { dst.store4(i / 16, (i % 16) * 4, x[it]) };
    }
}

/// FP32 flash attention for head dim 64 without masks, 128 threads a block, one block per
/// 64 query rows (x) and batch entry (y). q, k, v and o are batched strided `[_, 64]`
/// matrices as in [`attention`](crate::attention): row `i` of entry `b` at `b * stride + i
/// * ld`, with `ld` and `stride` multiples of 4 and 16-byte aligned buffers
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
pub fn flash_attn_f32_d64(
    mut o: DisjointSlice<f32>,
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
) {
    const D: usize = 64;
    // the tiles pass the 48 KB static limit, so they live in dynamic shared memory, whose
    // size the check below compares with what the launch passed
    let sm = DynamicSharedArray::<f32, 16>::get();
    // SAFETY: the four tiles are disjoint ranges of the block's dynamic shared memory,
    // which holds at least F32_D64_SMEM_BYTES once the check below passes; P reuses K's
    // tile, BM = BN rows of the same stride
    let (qs, ks, vs, ps) = unsafe {
        let ks = sm.add(BM * LD_Q);
        let vs = ks.add(BN * LD_K);
        let k = F32Tile::<LD_K>::new(ks);
        (F32Tile::<LD_Q>::new(sm), k, F32Tile::<LD_V>::new(vs), k)
    };

    let (lq, lk) = (lq as i32, lk as i32);
    let b = blockIdx_y() as usize;
    let q0 = (blockIdx_x() as usize * BM) as i32;
    let nq = (lq - q0).min(BM as i32);
    let tid = threadIdx_x() as usize;
    let (ty, tx) = (tid / TX, tid % TX);

    // one block-uniform check of everything the block reads or writes
    let (q, k, v) = (
        Batched { data: q, ld: q_ld, stride: q_stride },
        Batched { data: k, ld: k_ld, stride: k_stride },
        Batched { data: v, ld: v_ld, stride: v_stride },
    );
    let (qr, kr, vr) = (q.rows(b, q0, nq, D), k.rows(b, 0, lk, D), v.rows(b, 0, lk, D));
    let out_ok = rows_fit(o.len(), b * o_stride as usize, q0, nq, o_ld, D);
    require(qr.is_some() & kr.is_some() & vr.is_some() & out_ok & (dynamic_smem_bytes() >= F32_D64_SMEM_BYTES));
    let (Some(qr), Some(kr), Some(vr)) = (qr, kr, vr) else {
        // SAFETY: require trapped unless all three are Some
        unsafe { core::hint::unreachable_unchecked() }
    };

    // SAFETY: every thread stages; the barrier at the top of the key loop orders it
    // before the first reads
    unsafe { stage::<LD_Q>(qs, qr) };

    let mut acc = [[0.0f32; TC]; TR];
    let (mut m, mut l) = ([f32::NEG_INFINITY; TR], [0.0f32; TR]);
    let mut k0 = 0;
    while k0 < lk {
        // also keeps the previous tile's readers of K, V and P before this tile's writers
        thread::sync_threads();
        // SAFETY: every thread stages, between barriers
        unsafe {
            stage::<LD_K>(ks, kr.skip(k0));
            stage::<LD_V>(vs, vr.skip(k0));
        }
        thread::sync_threads();

        // S = Q K^T for rows ty + 8i and keys tx + 16j
        let mut s = [[0.0f32; TC]; TR];
        let mut d = 0;
        while d < D {
            __unroll_config::<4>();
            let mut qv = [[0.0f32; 4]; TR];
            let mut kv = [[0.0f32; 4]; TC];
            let mut i = 0;
            while i < TR {
                __unroll_config::<0>();
                // SAFETY: rows below 64 and columns below D, after the staging barrier
                qv[i] = unsafe { qs.load4(ty + RS * i, d) };
                i += 1;
            }
            let mut j = 0;
            while j < TC {
                __unroll_config::<0>();
                // SAFETY: as for q
                kv[j] = unsafe { ks.load4(tx + TX * j, d) };
                j += 1;
            }
            let mut i = 0;
            while i < TR {
                __unroll_config::<0>();
                let mut j = 0;
                while j < TC {
                    __unroll_config::<0>();
                    let mut c = 0;
                    while c < 4 {
                        __unroll_config::<0>();
                        s[i][j] = fma_rn_f32(qv[i][c], kv[j][c], s[i][j]);
                        c += 1;
                    }
                    j += 1;
                }
                i += 1;
            }
            d += 4;
        }

        // every thread's last K read before P overwrites the tile
        thread::sync_threads();

        // online softmax in the log2 domain; keys past lk in the last tile drop out
        let full = k0 + BN as i32 <= lk;
        let mut i = 0;
        while i < TR {
            __unroll_config::<0>();
            let mut mx = m[i];
            let mut j = 0;
            while j < TC {
                __unroll_config::<0>();
                let mut x = s[i][j] * scale_log2;
                if !full && k0 + (tx + TX * j) as i32 >= lk {
                    x = f32::NEG_INFINITY;
                }
                s[i][j] = x;
                mx = math::fmaxf(mx, x);
                j += 1;
            }
            mx = math::fmaxf(mx, warp::shuffle_xor_f32(mx, 1));
            mx = math::fmaxf(mx, warp::shuffle_xor_f32(mx, 2));
            mx = math::fmaxf(mx, warp::shuffle_xor_f32(mx, 4));
            mx = math::fmaxf(mx, warp::shuffle_xor_f32(mx, 8));
            // every tile has at least one real key, so mx is finite from the first tile on,
            // where corr = ex2(-inf) = 0
            let corr = ex2(m[i] - mx);
            m[i] = mx;
            l[i] *= corr;
            let mut c = 0;
            while c < TC {
                __unroll_config::<0>();
                acc[i][c] *= corr;
                c += 1;
            }
            let mut j = 0;
            while j < TC {
                __unroll_config::<0>();
                let p = ex2(s[i][j] - mx);
                l[i] += p;
                // SAFETY: (ty + 8i, tx + 16j) is this thread's alone; P's readers wait at
                // the barrier below
                unsafe { ps.store(ty + RS * i, tx + TX * j, p) };
                j += 1;
            }
            i += 1;
        }
        thread::sync_threads();

        // O += P V for rows ty + 8i and columns 4tx..4tx + 4
        let mut kk = 0;
        while kk < BN {
            __unroll_config::<4>();
            let mut pv = [[0.0f32; 4]; TR];
            let mut vv = [[0.0f32; TC]; 4];
            let mut i = 0;
            while i < TR {
                __unroll_config::<0>();
                // SAFETY: rows below 64 and columns below BN, after the P barrier
                pv[i] = unsafe { ps.load4(ty + RS * i, kk) };
                i += 1;
            }
            let mut c = 0;
            while c < 4 {
                __unroll_config::<0>();
                // SAFETY: rows below BN and columns below D, after the P barrier
                vv[c] = unsafe { vs.load4(kk + c, tx * 4) };
                c += 1;
            }
            let mut i = 0;
            while i < TR {
                __unroll_config::<0>();
                let mut c = 0;
                while c < 4 {
                    __unroll_config::<0>();
                    let mut e = 0;
                    while e < TC {
                        __unroll_config::<0>();
                        acc[i][e] = fma_rn_f32(pv[i][c], vv[c][e], acc[i][e]);
                        e += 1;
                    }
                    c += 1;
                }
                i += 1;
            }
            kk += 4;
        }
        k0 += BN as i32;
    }

    // each row's sum over its 16 threads, then normalized stores of this thread's columns
    let op = o.as_mut_ptr().wrapping_add(b * o_stride as usize + q0 as u32 as usize * o_ld as usize);
    let mut i = 0;
    while i < TR {
        __unroll_config::<0>();
        let mut sum = l[i];
        sum += warp::shuffle_xor_f32(sum, 1);
        sum += warp::shuffle_xor_f32(sum, 2);
        sum += warp::shuffle_xor_f32(sum, 4);
        sum += warp::shuffle_xor_f32(sum, 8);
        let inv = 1.0 / sum;
        let row = ty + RS * i;
        if (row as i32) < nq {
            let a = acc[i];
            // SAFETY: row < nq, inside the rows checked up front; no other thread of any
            // block writes these 4 columns of this row, and o_ld, o_stride and 4tx are
            // multiples of 4, so the store is 16-byte aligned
            unsafe { stg4(op.add(row * o_ld as usize + tx * 4), [a[0] * inv, a[1] * inv, a[2] * inv, a[3] * inv]) };
        }
        i += 1;
    }
}
