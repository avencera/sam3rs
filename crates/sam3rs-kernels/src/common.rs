//! Helpers shared by every kernel: math through libdevice, grid-stride loops, warp and
//! block reductions, activations, 16-bit packing and wide global accesses

use cuda_device::convert::{bf16_to_f32, cvt_bf16x2_f32, cvt_f16x2_f32};
use cuda_device::thread::{blockDim_x, blockIdx_x, gridDim_x, sync_threads, threadIdx_x};
use cuda_device::{ptx_asm, warp};

// the CUDA C++ math library calls these same libdevice functions, so results match what
// nvcc produced for the original kernels
unsafe extern "C" {
    fn __nv_erff(x: f32) -> f32;
    fn __nv_tanhf(x: f32) -> f32;
    fn __nv_expf(x: f32) -> f32;
    fn __nv_fast_expf(x: f32) -> f32;
    fn __nv_exp2f(x: f32) -> f32;
    fn __nv_logf(x: f32) -> f32;
    fn __nv_log2f(x: f32) -> f32;
    fn __nv_powf(x: f32, y: f32) -> f32;
    fn __nv_sinf(x: f32) -> f32;
    fn __nv_cosf(x: f32) -> f32;
    fn __nv_rintf(x: f32) -> f32;
    fn __nv_ceilf(x: f32) -> f32;
    fn __nv_rsqrtf(x: f32) -> f32;
    fn __nv_fmaxf(x: f32, y: f32) -> f32;
    fn __nv_fminf(x: f32, y: f32) -> f32;
}

/// Math functions under their CUDA C++ names
pub mod math {
    #[inline(always)]
    pub fn erff(x: f32) -> f32 {
        unsafe { super::__nv_erff(x) }
    }

    #[inline(always)]
    pub fn tanhf(x: f32) -> f32 {
        unsafe { super::__nv_tanhf(x) }
    }

    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        unsafe { super::__nv_expf(x) }
    }

    /// `__expf`: the fast approximate exponential
    #[inline(always)]
    pub fn fast_expf(x: f32) -> f32 {
        unsafe { super::__nv_fast_expf(x) }
    }

    #[inline(always)]
    pub fn exp2f(x: f32) -> f32 {
        unsafe { super::__nv_exp2f(x) }
    }

    #[inline(always)]
    pub fn logf(x: f32) -> f32 {
        unsafe { super::__nv_logf(x) }
    }

    #[inline(always)]
    pub fn log2f(x: f32) -> f32 {
        unsafe { super::__nv_log2f(x) }
    }

    #[inline(always)]
    pub fn powf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_powf(x, y) }
    }

    #[inline(always)]
    pub fn sinf(x: f32) -> f32 {
        unsafe { super::__nv_sinf(x) }
    }

    #[inline(always)]
    pub fn cosf(x: f32) -> f32 {
        unsafe { super::__nv_cosf(x) }
    }

    #[inline(always)]
    pub fn rintf(x: f32) -> f32 {
        unsafe { super::__nv_rintf(x) }
    }

    #[inline(always)]
    pub fn ceilf(x: f32) -> f32 {
        unsafe { super::__nv_ceilf(x) }
    }

    #[inline(always)]
    pub fn rsqrtf(x: f32) -> f32 {
        unsafe { super::__nv_rsqrtf(x) }
    }

    /// One `max.f32`; `f32::max` lowers to a compare and a select
    #[inline(always)]
    pub fn fmaxf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fmaxf(x, y) }
    }

    /// One `min.f32`, as [`fmaxf`]
    #[inline(always)]
    pub fn fminf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fminf(x, y) }
    }
}

/// First index and stride of a 1D grid-stride loop
#[inline(always)]
pub fn grid_stride() -> (i64, i64) {
    let block = blockDim_x() as i64;
    (blockIdx_x() as i64 * block + threadIdx_x() as i64, gridDim_x() as i64 * block)
}

#[inline(always)]
pub fn warp_sum(mut v: f32) -> f32 {
    let mut o = 16;
    while o > 0 {
        v += warp::shuffle_xor_f32(v, o);
        o >>= 1;
    }
    v
}

#[inline(always)]
pub fn warp_max(mut v: f32) -> f32 {
    let mut o = 16;
    while o > 0 {
        v = math::fmaxf(v, warp::shuffle_xor_f32(v, o));
        o >>= 1;
    }
    v
}

/// Block-wide sum; `scratch` points to 32 floats of shared memory
///
/// # Safety
/// Every thread of the block calls it with the same `scratch`
#[inline(always)]
pub unsafe fn block_sum(v: f32, scratch: *mut f32) -> f32 {
    let (lane, warp) = (threadIdx_x() & 31, threadIdx_x() >> 5);
    let v = warp_sum(v);
    sync_threads();
    if lane == 0 {
        unsafe { *scratch.add(warp as usize) = v };
    }
    sync_threads();
    let warps = (blockDim_x() + 31) >> 5;
    warp_sum(if lane < warps { unsafe { *scratch.add(lane as usize) } } else { 0.0 })
}

/// Block-wide max, as [`block_sum`]
///
/// # Safety
/// Every thread of the block calls it with the same `scratch`
#[inline(always)]
pub unsafe fn block_max(v: f32, scratch: *mut f32) -> f32 {
    let (lane, warp) = (threadIdx_x() & 31, threadIdx_x() >> 5);
    let v = warp_max(v);
    sync_threads();
    if lane == 0 {
        unsafe { *scratch.add(warp as usize) = v };
    }
    sync_threads();
    let warps = (blockDim_x() + 31) >> 5;
    warp_max(if lane < warps { unsafe { *scratch.add(lane as usize) } } else { f32::NEG_INFINITY })
}

pub const ACT_RELU: i32 = 1;
pub const ACT_GELU: i32 = 2;
pub const ACT_GELU_TANH: i32 = 3;
pub const ACT_SIGMOID: i32 = 4;

/// The activation the host selects by number; anything else is the identity
#[inline(always)]
pub fn activate(x: f32, act: i32) -> f32 {
    match act {
        ACT_RELU => math::fmaxf(x, 0.0),
        ACT_GELU => 0.5 * x * (1.0 + math::erff(x * core::f32::consts::FRAC_1_SQRT_2)),
        ACT_GELU_TANH => {
            let u = 0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + math::tanhf(u))
        }
        ACT_SIGMOID => 1.0 / (1.0 + math::expf(-x)),
        _ => x,
    }
}

/// Two floats rounded to nearest-even BF16, `lo` in the low half
#[inline(always)]
pub fn pack_bf16(lo: f32, hi: f32) -> u32 {
    cvt_bf16x2_f32(lo, hi)
}

/// Two floats rounded to nearest-even FP16, `lo` in the low half
#[inline(always)]
pub fn pack_f16(lo: f32, hi: f32) -> u32 {
    cvt_f16x2_f32(lo, hi)
}

/// The raw bits of a BF16 value; a distinct type so kernels can be generic over their
/// output precision
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Bf16(pub u16);

/// The raw bits of an FP16 value
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F16(pub u16);

/// An element type kernels load as and store from FP32
pub trait Elem: Copy {
    /// # Safety
    /// `p` is valid for a read
    unsafe fn load(p: *const Self) -> f32;

    /// # Safety
    /// `p` is valid for a write
    unsafe fn store(p: *mut Self, v: f32);
}

impl Elem for f32 {
    #[inline(always)]
    unsafe fn load(p: *const Self) -> f32 {
        unsafe { *p }
    }

    #[inline(always)]
    unsafe fn store(p: *mut Self, v: f32) {
        unsafe { *p = v };
    }
}

impl Elem for Bf16 {
    #[inline(always)]
    unsafe fn load(p: *const Self) -> f32 {
        bf16_to_f32(unsafe { (*p).0 })
    }

    #[inline(always)]
    unsafe fn store(p: *mut Self, v: f32) {
        unsafe { *p = Bf16(pack_bf16(v, 0.0) as u16) };
    }
}

/// Four floats from a 16-byte aligned global address
///
/// # Safety
/// `p` is 16-byte aligned and valid for four reads
#[inline(always)]
pub unsafe fn ldg4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.v4.f32 {%0, %1, %2, %3}, [g]; }",
            out("=f") a,
            out("=f") b,
            out("=f") c,
            out("=f") d,
            in("l") p as u64,
        );
    }
    [a, b, c, d]
}

/// Sixteen bytes from a 16-byte aligned global address
///
/// # Safety
/// `p` is 16-byte aligned and valid for a 16-byte read
#[inline(always)]
pub unsafe fn ldg_u4<T>(p: *const T) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.v4.u32 {%0, %1, %2, %3}, [g]; }",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") p as u64,
        );
    }
    [a, b, c, d]
}

/// Sixteen bytes to a 16-byte aligned global address
///
/// # Safety
/// `p` is 16-byte aligned and valid for a 16-byte write
#[inline(always)]
pub unsafe fn stg_u4<T>(p: *mut T, v: [u32; 4]) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.v4.u32 [g], {%1, %2, %3, %4}; }",
            in("l") p as u64,
            in("r") v[0],
            in("r") v[1],
            in("r") v[2],
            in("r") v[3],
            clobber("memory"),
        );
    }
}

/// Two floats to an 8-byte aligned global address
///
/// # Safety
/// `p` is 8-byte aligned and valid for two writes
#[inline(always)]
pub unsafe fn stg2(p: *mut f32, a: f32, b: f32) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.v2.f32 [g], {%1, %2}; }",
            in("l") p as u64,
            in("f") a,
            in("f") b,
            clobber("memory"),
        );
    }
}
