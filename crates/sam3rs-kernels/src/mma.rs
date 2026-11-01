//! Tensor-core operands: shared-memory tiles, MMA fragments and the MMA itself, for the
//! 16-bit formats (m16n8k16) and TF32 (m16n8k8)
//!
//! The types turn three mix-ups into compile errors: a generic or global pointer where a
//! 32-bit shared-window address belongs ([`SharedAddr`]), a fragment of one format fed to
//! another's MMA (tiles and fragments carry their format), and an A fragment in a B
//! slot. Fragments only come from a tile load or from the accumulator-to-A identity, so
//! no code builds one from loose registers. Every type is a transparent wrapper over the
//! registers or the address, so none costs an instruction

use core::marker::PhantomData;

use cuda_device::convert::{cvt_f32x2_f16x2, cvt_rna_tf32_f32};
use cuda_device::ptx_asm;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::threadIdx_x;
use cuda_device::wmma::{
    ldmatrix_x4_shared_u32, ldmatrix_x4_trans_shared_u32, mma_m16n8k8_f32_tf32, mma_m16n8k16_f16_f16,
    mma_m16n8k16_f32_bf16, mma_m16n8k16_f32_f16,
};

use crate::common::{Bf16, F16, pack_f16};

/// A byte address in the CTA's shared-memory window, the operand `ldmatrix` and
/// `st.shared` take; only a [`SharedTile`] makes one
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct SharedAddr(u32);

/// Eight bytes to an 8-byte aligned shared address
///
/// # Safety
/// `addr` is 8-byte aligned and inside its tile, and no other thread accesses those bytes
/// before the next barrier
#[inline(always)]
pub unsafe fn sts2(addr: SharedAddr, a: u32, b: u32) {
    unsafe {
        ptx_asm!("st.shared.v2.u32 [%0], {%1, %2};", in("r") addr.0, in("r") a, in("r") b, clobber("memory"));
    }
}

/// Sixteen bytes to a 16-byte aligned shared address
///
/// # Safety
/// As [`sts2`], 16-byte aligned
#[inline(always)]
pub unsafe fn sts4(addr: SharedAddr, v: [u32; 4]) {
    unsafe {
        ptx_asm!(
            "st.shared.v4.u32 [%0], {%1, %2, %3, %4};",
            in("r") addr.0,
            in("r") v[0],
            in("r") v[1],
            in("r") v[2],
            in("r") v[3],
            clobber("memory"),
        );
    }
}

/// Starts a 16-byte copy from global `src` to shared `dst` that bypasses L1 (`cp.async.cg`);
/// with `full` false it reads nothing and writes 16 zero bytes
///
/// # Safety
/// `dst` is 16-byte aligned inside its tile, and `src` is 16-byte aligned with 16 readable
/// bytes when `full`. No thread touches `dst` until this thread has waited on the copy
/// ([`cp_async_wait_all`]) and a barrier has followed
#[inline(always)]
pub unsafe fn cp_async16(dst: SharedAddr, src: *const u8, full: bool) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.cg.shared.global [%0], [g], 16, %2; }",
            in("r") dst.0,
            in("l") src as u64,
            in("r") if full { 16u32 } else { 0u32 },
            clobber("memory"),
        );
    }
}

/// Closes the group of this thread's copies started since the last commit
///
/// # Safety
/// None beyond [`cp_async16`]'s, but kept unsafe to sit with the copies it orders
#[inline(always)]
pub unsafe fn cp_async_commit() {
    unsafe { ptx_asm!("cp.async.commit_group;", clobber("memory")) };
}

/// Waits until every copy this thread started has landed; other threads' copies need a
/// barrier after it
///
/// # Safety
/// As [`cp_async_commit`]
#[inline(always)]
pub unsafe fn cp_async_wait_all() {
    unsafe { ptx_asm!("cp.async.wait_group 0;", clobber("memory")) };
}

/// `c += a @ b` on FP16 operands with FP32 accumulation, half the FP16-accumulate rate on
/// GeForce parts, for sums too long to round in FP16
///
/// # Safety
/// All 32 lanes of the warp call it together
#[inline(always)]
pub unsafe fn mma_f16_acc_f32(c: [f32; 4], a: FragA<F16>, b: FragB<F16>) -> [f32; 4] {
    unsafe { mma_m16n8k16_f32_f16(c, a.0, b.0) }
}

/// A 16-bit tensor-core operand format and the m16n8k16 MMA on it
pub trait Mma16: Copy {
    /// One lane's share of a 16x8 accumulator
    type Acc: Copy;

    const ZERO: Self::Acc;

    /// `c += a @ b`
    ///
    /// # Safety
    /// All 32 lanes of the warp call it together
    unsafe fn mma(c: Self::Acc, a: FragA<Self>, b: FragB<Self>) -> Self::Acc;

    /// The accumulator as FP32, in the register order of an FP32 accumulator
    fn acc_f32(c: Self::Acc) -> [f32; 4];
}

/// BF16 operands with FP32 accumulation
impl Mma16 for Bf16 {
    type Acc = [f32; 4];

    const ZERO: Self::Acc = [0.0; 4];

    #[inline(always)]
    unsafe fn mma(c: Self::Acc, a: FragA<Self>, b: FragB<Self>) -> Self::Acc {
        unsafe { mma_m16n8k16_f32_bf16(c, a.0, b.0) }
    }

    #[inline(always)]
    fn acc_f32(c: Self::Acc) -> [f32; 4] {
        c
    }
}

/// FP16 operands with FP16 accumulation: twice the FP32-accumulate rate on GeForce parts
impl Mma16 for F16 {
    type Acc = [u32; 2];

    const ZERO: Self::Acc = [0; 2];

    #[inline(always)]
    unsafe fn mma(c: Self::Acc, a: FragA<Self>, b: FragB<Self>) -> Self::Acc {
        unsafe { mma_m16n8k16_f16_f16(c, a.0, b.0) }
    }

    #[inline(always)]
    fn acc_f32(c: Self::Acc) -> [f32; 4] {
        let (lo, hi) = (cvt_f32x2_f16x2(c[0]), cvt_f32x2_f16x2(c[1]));
        [lo.0, lo.1, hi.0, hi.1]
    }
}

/// One lane's registers of a row-major 16x16 A operand
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FragA<T>([u32; 4], PhantomData<T>);

impl<T> FragA<T> {
    pub const ZERO: Self = Self([0; 4], PhantomData);
}

impl FragA<F16> {
    /// The FP16 A operand whose left and right 16x8 halves are the FP32 accumulators
    /// `lo` and `hi`: an m16n8 accumulator has the register layout of half an A fragment,
    /// which is how P = softmax(S) feeds P @ V without a trip through shared memory
    #[inline(always)]
    pub fn from_acc_pair(lo: [f32; 4], hi: [f32; 4]) -> Self {
        Self(
            [pack_f16(lo[0], lo[1]), pack_f16(lo[2], lo[3]), pack_f16(hi[0], hi[1]), pack_f16(hi[2], hi[3])],
            PhantomData,
        )
    }
}

/// One lane's registers of a column-major 16x8 B operand
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FragB<T>([u32; 2], PhantomData<T>);

/// A row-major tile of 16-bit `T` in shared memory with rows of `LD` elements
pub struct SharedTile<T, const LD: usize> {
    base: SharedAddr,
    _format: PhantomData<T>,
}

impl<T, const LD: usize> Clone for SharedTile<T, LD> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const LD: usize> Copy for SharedTile<T, LD> {}

impl<T, const LD: usize> SharedTile<T, LD> {
    /// # Safety
    /// `base` points to a 16-byte aligned shared-memory array of 16-bit values, as many
    /// rows of `LD` as callers address
    #[inline(always)]
    pub unsafe fn new(base: *mut u16) -> Self {
        Self { base: SharedAddr(unsafe { cvta_generic_to_shared_u32(base as *const u8) }), _format: PhantomData }
    }

    /// The address of element `(row, col)`
    #[inline(always)]
    pub fn at(self, row: usize, col: usize) -> SharedAddr {
        // 32-bit math: an offset computed in 64 bits and truncated hides its constant part
        // from ptxas, which then can't fold it into the access's immediate
        SharedAddr(self.base.0 + (row as u32 * LD as u32 + col as u32) * 2)
    }

    /// The address of `(row, col)` plus this lane's `ldmatrix.x4` row: matrix `mi`'s row
    /// `ri`, at `(mi % 2 * 8 + ri, mi / 2 * 8)` when `ROW_HALF_FIRST`, else at `(mi / 2 * 8
    /// + ri, mi % 2 * 8)`
    #[inline(always)]
    fn ldsm_addr<const ROW_HALF_FIRST: bool>(self, row: usize, col: usize) -> SharedAddr {
        let lane = threadIdx_x() % 32;
        let (mi, ri) = (lane / 8, lane % 8);
        let (lr, lc) =
            if ROW_HALF_FIRST { ((mi & 1) * 8 + ri, (mi / 2) * 8) } else { ((mi / 2) * 8 + ri, (mi & 1) * 8) };
        // the lane's part first and the fragment's constant offset last: summed the other
        // way, LLVM merges the two into an `or` per load, which ptxas can't fold into the
        // load's immediate
        let lane_addr = self.base.0 + (lr * LD as u32 + lc) * 2;
        SharedAddr(lane_addr + (row as u32 * LD as u32 + col as u32) * 2)
    }
}

impl<T: Mma16, const LD: usize> SharedTile<T, LD> {
    /// The A fragment of rows `[m0, m0 + 16)` and columns `[k0, k0 + 16)`
    ///
    /// # Safety
    /// All 32 lanes call it together with the same `m0` and `k0`, inside the tile, after
    /// a barrier that orders the tile's stores
    #[inline(always)]
    pub unsafe fn frag_a(self, m0: usize, k0: usize) -> FragA<T> {
        // matrices: rows m0.. at k0, m0 + 8.. at k0, m0.. at k0 + 8, m0 + 8.. at k0 + 8,
        // the A register order
        let p = self.ldsm_addr::<true>(m0, k0);
        FragA(unsafe { ldmatrix_x4_shared_u32(p.0) }, PhantomData)
    }

    /// The B fragments of columns `[n0, n0 + 8)` and `[n0 + 8, n0 + 16)`, depth `[k0, k0 +
    /// 16)`, from a tile that holds B transposed: rows are B's columns, as K holds keys
    ///
    /// # Safety
    /// As [`SharedTile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_b_pair_nk(self, n0: usize, k0: usize) -> [FragB<T>; 2] {
        // matrices: n0.. at k0 and k0 + 8, then n0 + 8.. at k0 and k0 + 8
        let p = self.ldsm_addr::<false>(n0, k0);
        let r = unsafe { ldmatrix_x4_shared_u32(p.0) };
        [FragB([r[0], r[1]], PhantomData), FragB([r[2], r[3]], PhantomData)]
    }

    /// The same from a tile that holds B as is, rows along the depth, as V does
    ///
    /// # Safety
    /// As [`SharedTile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_b_pair_kn(self, k0: usize, n0: usize) -> [FragB<T>; 2] {
        // transposed matrices: depth k0.. and k0 + 8.. at n0, then the same at n0 + 8
        let p = self.ldsm_addr::<true>(k0, n0);
        let r = unsafe { ldmatrix_x4_trans_shared_u32(p.0) };
        [FragB([r[0], r[1]], PhantomData), FragB([r[2], r[3]], PhantomData)]
    }
}

/// The TF32 operand format: FP32's range with a 10-bit mantissa, one value per 32-bit
/// register, multiplied with FP32 accumulation. Values reach it only through
/// [`Tf32::round`]
#[derive(Clone, Copy)]
pub struct Tf32;

impl Tf32 {
    /// `x` rounded to TF32, to nearest with ties away from zero (`cvt.rna`)
    #[inline(always)]
    pub fn round(x: f32) -> u32 {
        cvt_rna_tf32_f32(x)
    }

    /// `c += a @ b` for a 16x8 A and an 8x8 B
    ///
    /// # Safety
    /// All 32 lanes of the warp call it together
    #[inline(always)]
    pub unsafe fn mma(c: [f32; 4], a: FragA<Tf32>, b: FragB<Tf32>) -> [f32; 4] {
        unsafe { mma_m16n8k8_f32_tf32(c, a.0, b.0) }
    }
}

impl FragA<Tf32> {
    /// The A operand of one 8-key tile of probabilities held in the m16n8 accumulator
    /// layout, where lane (g, t) has keys 2t and 2t + 1 of rows g and g + 8. A's column t
    /// stands for key 2t and column t + 4 for key 2t + 1, so the registers only reorder,
    /// with no shuffle; the B operand must read V's rows in the same order, as
    /// [`Tf32Tile::frag_b_kn_paired`] does
    #[inline(always)]
    pub fn from_probs(p: [f32; 4]) -> Self {
        Self([Tf32::round(p[0]), Tf32::round(p[2]), Tf32::round(p[1]), Tf32::round(p[3])], PhantomData)
    }
}

/// A row-major tile of TF32 values in shared memory with rows of `LD` 32-bit elements. `LD`
/// is a multiple of 4 and `LD % 32 == 4`, which keeps the fragment loads bank-conflict free
pub struct Tf32Tile<const LD: usize> {
    base: SharedAddr,
}

impl<const LD: usize> Clone for Tf32Tile<LD> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const LD: usize> Copy for Tf32Tile<LD> {}

impl<const LD: usize> Tf32Tile<LD> {
    /// # Safety
    /// `base` points to a 16-byte aligned shared-memory array of 32-bit values, as many
    /// rows of `LD` as callers address
    #[inline(always)]
    pub unsafe fn new(base: *mut u32) -> Self {
        Self { base: SharedAddr(unsafe { cvta_generic_to_shared_u32(base as *const u8) }) }
    }

    /// The tile from row `row` on
    #[inline(always)]
    pub fn skip(self, row: usize) -> Self {
        Self { base: self.at(row, 0) }
    }

    /// The address of element `(row, col)`
    #[inline(always)]
    pub fn at(self, row: usize, col: usize) -> SharedAddr {
        // 32-bit math, as in SharedTile::at
        SharedAddr(self.base.0 + (row as u32 * LD as u32 + col as u32) * 4)
    }

    /// Four TF32 values to `(row, col)`
    ///
    /// # Safety
    /// `col` is a multiple of 4, inside the tile, and no other thread accesses those
    /// elements before the next barrier
    #[inline(always)]
    pub unsafe fn store4(self, row: usize, col: usize, v: [u32; 4]) {
        unsafe { sts4(self.at(row, col), v) };
    }

    /// The address of `(row, col)` plus this lane's `ldmatrix.x4` row, where each 8x8
    /// 16-bit matrix is 8 rows of 4 TF32 values: matrix `mi`'s row `ri`, at `(mi % 2 * 8 +
    /// ri, mi / 2 * 4)` when `ROW_HALF_FIRST`, else at `(mi / 2 * 8 + ri, mi % 2 * 4)`
    #[inline(always)]
    fn ldsm_addr<const ROW_HALF_FIRST: bool>(self, row: usize, col: usize) -> SharedAddr {
        let lane = threadIdx_x() % 32;
        let (mi, ri) = (lane / 8, lane % 8);
        let (lr, lc) =
            if ROW_HALF_FIRST { ((mi & 1) * 8 + ri, (mi / 2) * 4) } else { ((mi / 2) * 8 + ri, (mi & 1) * 4) };
        // the lane's part first and the constant last, as in SharedTile::ldsm_addr
        let lane_addr = self.base.0 + (lr * LD as u32 + lc) * 4;
        SharedAddr(lane_addr + (row as u32 * LD as u32 + col as u32) * 4)
    }

    /// The A fragment of rows `[m0, m0 + 16)` and columns `[k0, k0 + 8)`: `ldmatrix`
    /// hands lane (g, t) word t of row g of each matrix, which for 32-bit values is the
    /// TF32 A layout
    ///
    /// # Safety
    /// All 32 lanes call it together with the same `m0` and `k0`, inside the tile, after
    /// a barrier that orders the tile's stores
    #[inline(always)]
    pub unsafe fn frag_a(self, m0: usize, k0: usize) -> FragA<Tf32> {
        // matrices: rows m0.. at k0, m0 + 8.. at k0, m0.. at k0 + 4, m0 + 8.. at k0 + 4
        let p = self.ldsm_addr::<true>(m0, k0);
        FragA(unsafe { ldmatrix_x4_shared_u32(p.0) }, PhantomData)
    }

    /// The B fragments of columns `[n0, n0 + 8)` and `[n0 + 8, n0 + 16)`, depth `[k0, k0 +
    /// 8)`, from a tile that holds B transposed, as K holds keys
    ///
    /// # Safety
    /// As [`Tf32Tile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_b_pair_nk(self, n0: usize, k0: usize) -> [FragB<Tf32>; 2] {
        // matrices: n0.. at k0 and k0 + 4, then n0 + 8.. at k0 and k0 + 4
        let p = self.ldsm_addr::<false>(n0, k0);
        let r = unsafe { ldmatrix_x4_shared_u32(p.0) };
        [FragB([r[0], r[1]], PhantomData), FragB([r[2], r[3]], PhantomData)]
    }

    /// The B fragment of columns `[n0, n0 + 8)` at depth `[k0, k0 + 8)` from a tile that
    /// holds B as is (rows along the depth, as V does), in the key order of
    /// [`FragA::from_probs`]: lane (g, t) reads rows k0 + 2t and k0 + 2t + 1 of column
    /// n0 + g, which `ldmatrix` can't gather for 32-bit values
    ///
    /// # Safety
    /// As [`Tf32Tile::frag_a`]
    #[inline(always)]
    pub unsafe fn frag_b_kn_paired(self, k0: usize, n0: usize) -> FragB<Tf32> {
        let lane = threadIdx_x() % 32;
        let (g, t) = (lane / 4, lane % 4);
        let lane_addr = self.base.0 + (2 * t * LD as u32 + g) * 4;
        let p = lane_addr + (k0 as u32 * LD as u32 + n0 as u32) * 4;
        let (b0, b1): (u32, u32);
        unsafe {
            ptx_asm!("ld.shared.u32 %0, [%1];", out("=r") b0, in("r") p);
            ptx_asm!("ld.shared.u32 %0, [%1];", out("=r") b1, in("r") p + LD as u32 * 4);
        }
        FragB([b0, b1], PhantomData)
    }
}
