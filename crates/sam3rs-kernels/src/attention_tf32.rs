//! Fused attention for the TF32 precision mode
//!
//! The [`attention`](crate::attention) walk on TF32 tensor cores: q, k, v and the softmax
//! probabilities round to TF32 as they are staged or packed, and every m16n8k8 product
//! sums in FP32, the contract of the TF32 GEMMs that the materialized path runs. The
//! softmax is the same FP32 online softmax. Shared tiles hold TF32 bits in rows of `D +
//! 4` words, so every fragment load is bank-conflict free
//!
//! The block setup, softmax and output stores mirror `flash_attn`'s line for line. They
//! are helpers here only: moving them out of `flash_attn` changed the BF16 kernels' block
//! layout (8 more SASS instructions each), and BF16 mode's kernels stay as they are

use core::f32::consts::LOG2_E;
use core::ptr::addr_of_mut;

use cuda_device::float::fma_rn_f32;
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_y, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

use crate::attention::{
    Batched, FA_BN, FA_WARPS, Rows, Rpb, SJ, STAGE_MAX_ITERS, ex2, ldg_nc, ldg2_nc, require, rows_fit,
};
use crate::common::{ldg4, math, stg2};
use crate::mma::{FragA, Tf32, Tf32Tile};

/// Where one block's `16 * WARPS` query rows sit and which keys they cover, once
/// [`Block::check`] has proven every row the block touches in bounds. `g` and `t` are the
/// lane's group and thread in a fragment, `wr` the warp's first row
#[derive(Clone, Copy)]
struct Block {
    b: usize,
    q0: i32,
    nq: i32,
    lq: i32,
    lk: i32,
    k_begin: i32,
    k_end: i32,
    g: usize,
    t: usize,
    wr: usize,
    has_pad: bool,
    split: bool,
    part_base: usize,
    rpb_base: usize,
}

impl Block {
    /// This block's geometry and its q, k and v rows (`D` wide, `bm` query rows), after
    /// checking everything it reads or writes at once, up front: one uniform branch to
    /// one trap. The RPB rows are checked for the whole block, as a check per thread row
    /// would make ptxas treat the warp as divergent from there on. With a non-empty
    /// `part`, block z covers keys `[z * kv_chunk, (z + 1) * kv_chunk)` only
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn check<const D: usize, const RPB: bool, T, TK, TV>(
        bm: usize,
        o: &DisjointSlice<f32>,
        o_ld: u32,
        o_stride: u32,
        q: Batched<T>,
        k: Batched<TK>,
        v: Batched<TV>,
        lq: u32,
        lk: u32,
        key_pad: &[u8],
        rpb: Rpb,
        part: &DisjointSlice<f32>,
        kv_chunk: u32,
    ) -> (Self, Rows<T>, Rows<TK>, Rows<TV>) {
        let (lq, lk) = (lq as i32, lk as i32);
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
        require(
            qr.is_some() & kr.is_some() & vr.is_some() & out_ok & rpb_ok & (!has_pad | (key_pad.len() >= lk as usize)),
        );
        let (Some(qr), Some(kr), Some(vr)) = (qr, kr, vr) else {
            // SAFETY: require trapped unless all three are Some
            unsafe { core::hint::unreachable_unchecked() }
        };
        (Self { b, q0, nq, lq, lk, k_begin, k_end, g, t, wr, has_pad, split, part_base, rpb_base }, qr, kr, vr)
    }

    /// Per fragment row: this head's dx and dy rows of the box RPB, or null for the
    /// presence token and rows past lq
    #[inline(always)]
    fn rpb_rows<const RPB: bool>(&self, rpb: Rpb) -> ([*const f32; 2], [*const f32; 2]) {
        let (q0, lq, wr, g, rpb_base, rpb_side) = (self.q0, self.lq, self.wr, self.g, self.rpb_base, rpb.side as usize);
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
        (rdx, rdy)
    }
}

/// The online softmax of one tile of keys from `k0`: `s` holds this thread's raw QK^T
/// fragments (the m16n8 accumulator layout, keys `k0 + j * 8 + t * 2 + (e & 1)`) and
/// becomes their probabilities, and `m` and `l` carry its two rows' running max (scaled,
/// log2 domain) and sum. Returns them with each row's factor for rescaling earlier
/// accumulators. Arrays go in and out by value: behind `&mut` they stayed on the stack
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn softmax_tile<const RPB: bool>(
    mut s: [[f32; 4]; SJ],
    mut m: [f32; 2],
    mut l: [f32; 2],
    blk: &Block,
    k0: i32,
    scale_log2: f32,
    key_pad: &[u8],
    rpb: Rpb,
    rdx: [*const f32; 2],
    rdy: [*const f32; 2],
) -> ([[f32; 4]; SJ], [f32; 2], [f32; 2], [f32; 2]) {
    let Block { k_end, lk, t, has_pad, .. } = *blk;
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
    (s, m, l, corr)
}

/// Normalizes and stores this thread's two rows of the `[16 * WARPS, D]` output, columns
/// `t * 2 + 8 * dj`, which no other thread of any block writes; a split block stores its
/// unnormalized accumulators, row max and row sum to `part` instead
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn store_rows<const D: usize, const DJ: usize>(
    blk: &Block,
    acc: [[f32; 4]; DJ],
    m: [f32; 2],
    mut l: [f32; 2],
    mut o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    mut part: DisjointSlice<f32>,
) {
    let Block { b, q0, nq, g, t, wr, split, part_base, .. } = *blk;
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

/// `rows` rows of `[_, D]` FP32 into `dst` as TF32; rows past `src.rows` are zero
///
/// Loads every chunk before storing any, and runs to [`STAGE_MAX_ITERS`] fully unrolled,
/// as `attention`'s staging does and for the same reasons
///
/// # Safety
/// Every thread of a block of exactly `threads` threads calls it between barriers; `dst`
/// has `rows` rows, and `rows * D / 4 <= STAGE_MAX_ITERS * threads`
#[inline(always)]
unsafe fn stage_tf32<const D: usize, const LD: usize>(dst: Tf32Tile<LD>, src: Rows<f32>, rows: usize, threads: usize) {
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
            let v = [Tf32::round(x[0]), Tf32::round(x[1]), Tf32::round(x[2]), Tf32::round(x[3])];
            // SAFETY: (r, c) is this thread's alone and r < rows, 16-byte aligned as LD and
            // c are multiples of 4
            unsafe { dst.store4(r, c, v) };
        }
    }
}

/// The attention of one block of `16 * WARPS` query rows on TF32 tensor cores
///
/// Arguments, masking, the box RPB and key splits as in `attention::flash_attn`. `sm` is
/// the block's shared tile of `max(16 * WARPS, 2 * FA_BN)` rows: q is staged there once
/// and moves to registers, then it holds the K tile (rows `[0, FA_BN)`) and the V tile
/// (rows `[FA_BN, 2 * FA_BN)`). `DK = DJ = D / 8`, `LD = D + 4`
///
/// # Safety
/// The block has exactly `32 * WARPS` threads (the kernels' `.reqntid`), and distinct
/// (batch, row, column) of `o` are distinct elements under its `ld` and `stride`
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn flash_attn_tf32<
    const D: usize,
    const DK: usize,
    const DJ: usize,
    const LD: usize,
    const WARPS: usize,
    const RPB: bool,
>(
    sm: Tf32Tile<LD>,
    o: DisjointSlice<f32>,
    o_ld: u32,
    o_stride: u32,
    q: Batched<f32>,
    k: Batched<f32>,
    v: Batched<f32>,
    lq: u32,
    lk: u32,
    scale_log2: f32,
    key_pad: &[u8],
    rpb: Rpb,
    part: DisjointSlice<f32>,
    kv_chunk: u32,
) {
    let bm = 16 * WARPS;
    const { assert!(16 * WARPS * D / 4 <= STAGE_MAX_ITERS * 32 * WARPS && FA_BN * D / 4 <= STAGE_MAX_ITERS * 32 * WARPS) };
    let (blk, qr, kr, vr) =
        Block::check::<D, RPB, f32, f32, f32>(bm, &o, o_ld, o_stride, q, k, v, lq, lk, key_pad, rpb, &part, kv_chunk);
    let Block { k_begin, k_end, wr, .. } = blk;
    let (ks, vs) = (sm, sm.skip(FA_BN));

    // SAFETY: barriers order the staging stores before the fragment loads, and the
    // block's 32 * WARPS threads cover the tile's 16 * WARPS rows
    unsafe { stage_tf32::<D, LD>(sm, qr, bm, 32 * WARPS) };
    thread::sync_threads();
    let mut qa = [FragA::<Tf32>::ZERO; DK];
    let mut kk = 0;
    while kk < DK {
        __unroll_config::<0>();
        qa[kk] = unsafe { sm.frag_a(wr, kk * 8) };
        kk += 1;
    }

    let mut acc = [[0.0f32; 4]; DJ];
    let (mut m, mut l) = ([f32::NEG_INFINITY; 2], [0.0f32; 2]);
    let (rdx, rdy) = blk.rpb_rows::<RPB>(rpb);

    let mut k0 = k_begin;
    while k0 < k_end {
        // also orders the q fragment loads before the first K and V tiles overwrite q
        thread::sync_threads();
        // SAFETY: as for q; the barriers on either side keep the previous tile's readers
        // and this tile's writers apart
        unsafe {
            stage_tf32::<D, LD>(ks, kr.skip(k0 - k_begin), FA_BN, 32 * WARPS);
            stage_tf32::<D, LD>(vs, vr.skip(k0 - k_begin), FA_BN, 32 * WARPS);
        }
        thread::sync_threads();

        let mut s = [[0.0f32; 4]; SJ];
        let mut j = 0;
        while j < SJ {
            __unroll_config::<0>();
            let mut kk = 0;
            while kk < DK {
                __unroll_config::<0>();
                // SAFETY: the whole warp runs this with warp-uniform offsets inside the tile
                unsafe {
                    let kb = ks.frag_b_pair_nk(j * 8, kk * 8);
                    s[j] = Tf32::mma(s[j], qa[kk], kb[0]);
                    s[j + 1] = Tf32::mma(s[j + 1], qa[kk], kb[1]);
                }
                kk += 1;
            }
            j += 2;
        }

        let corr;
        (s, m, l, corr) = softmax_tile::<RPB>(s, m, l, &blk, k0, scale_log2, key_pad, rpb, rdx, rdy);
        let mut dj = 0;
        while dj < DJ {
            __unroll_config::<0>();
            acc[dj][0] *= corr[0];
            acc[dj][1] *= corr[0];
            acc[dj][2] *= corr[1];
            acc[dj][3] *= corr[1];
            dj += 1;
        }

        // P @ V one 8-key tile at a time, with P's fragment reused as the A operand
        let mut j = 0;
        while j < SJ {
            __unroll_config::<0>();
            let pa = FragA::<Tf32>::from_probs(s[j]);
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                // SAFETY: as for QK^T
                unsafe { acc[dj] = Tf32::mma(acc[dj], pa, vs.frag_b_kn_paired(j * 8, dj * 8)) };
                dj += 1;
            }
            j += 1;
        }
        k0 += FA_BN as i32;
    }

    store_rows::<D, DJ>(&blk, acc, m, l, o, o_ld, o_stride, part);
}

/// Declares one kernel's shared tile for `$rows` query rows and head dim `$d`: q, then K
/// and V, share it (see [`flash_attn_tf32`])
macro_rules! tf32_tile {
    ($rows:expr, $d:expr) => {{
        const ROWS: usize = if $rows > 2 * FA_BN { $rows } else { 2 * FA_BN };
        static mut SM: SharedArray<u32, { ROWS * ($d + 4) }, 16> = SharedArray::UNINIT;
        // SAFETY: the array is this kernel's own, 16-byte aligned and sized for the rows
        // flash_attn_tf32 addresses
        unsafe { Tf32Tile::<{ $d + 4 }>::new(addr_of_mut!(SM) as *mut u32) }
    }};
}

/// Expands to a TF32 flash attention kernel for head dim `$d` with `$warps` warps
macro_rules! flash_attn_tf32_kernel {
    ($name:ident, $d:literal, $warps:expr, $threads:literal) => {
        #[doc = concat!("TF32 flash attention for head dim ", $d, ", ", $threads, " threads a block; see [`flash_attn_tf32`]")]
        #[kernel]
        #[launch_contract(domain = 3, block = ($threads, 1, 1))]
        pub fn $name(
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
            key_pad: &[u8],
            part: DisjointSlice<f32>,
            kv_chunk: u32,
        ) {
            const { assert!($threads == 32 * $warps) };
            let sm = tf32_tile!(16 * $warps, $d);
            let rpb = Rpb { dx: &[], dy: &[], side: 0 };
            // SAFETY: `.reqntid` fixes the block at 32 * $warps threads
            unsafe {
                flash_attn_tf32::<$d, { $d / 8 }, { $d / 8 }, { $d + 4 }, $warps, false>(
                    sm,
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

flash_attn_tf32_kernel!(flash_attn_tf32_d32, 32, FA_WARPS, 128);
flash_attn_tf32_kernel!(flash_attn_tf32_d64, 64, FA_WARPS, 128);
// 128 query rows a block for long self-attention: every block re-reads all of K and V
flash_attn_tf32_kernel!(flash_attn_tf32_tall_d32, 32, 8, 256);

/// TF32 flash attention with the box relative position bias of [`Rpb`] for head dim 32
#[kernel]
#[launch_contract(domain = 3, block = (128, 1, 1))]
pub fn flash_attn_tf32_rpb_d32(
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
    let sm = tf32_tile!(16 * FA_WARPS, 32);
    // SAFETY: `.reqntid` fixes the block at 128 threads = FA_WARPS warps
    unsafe {
        flash_attn_tf32::<32, 4, 4, 36, FA_WARPS, true>(
            sm,
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
