//! 16-bit tensor-core operands: shared-memory tiles, MMA fragments and the MMA itself
//!
//! The types turn three mix-ups into compile errors: a generic or global pointer where a
//! 32-bit shared-window address belongs ([`SharedAddr`]), a BF16 fragment fed to an FP16
//! MMA or the reverse (tiles and fragments carry their format), and an A fragment in a B
//! slot. Fragments only come from a tile load or from the accumulator-to-A identity, so
//! no code builds one from loose registers. Every type is a transparent wrapper over the
//! registers or the address, so none costs an instruction

use core::marker::PhantomData;

use cuda_device::convert::cvt_f32x2_f16x2;
use cuda_device::ptx_asm;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::threadIdx_x;
use cuda_device::wmma::{
    ldmatrix_x4_shared_u32, ldmatrix_x4_trans_shared_u32, mma_m16n8k16_f16_f16, mma_m16n8k16_f32_bf16,
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
