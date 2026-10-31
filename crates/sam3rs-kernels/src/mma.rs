//! Tensor-core mma and shared-memory helpers shared by the attention and conv kernels

use cuda_device::ptx_asm;
use cuda_device::shared::cvta_generic_to_shared_u32;

/// `c += a @ b` on BF16 tensor cores with FP32 accumulation
#[inline(always)]
pub unsafe fn mma_bf16(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = c;
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};",
            inout("+f") c0,
            inout("+f") c1,
            inout("+f") c2,
            inout("+f") c3,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b0,
            in("r") b1,
            options(register_only),
        );
    }
    [c0, c1, c2, c3]
}

/// `c += a @ b` with FP16 inputs and FP16 accumulation: twice the FP32-accumulate rate
/// on GeForce parts. cuda-device only wraps the FP32-accumulate form
#[inline(always)]
pub unsafe fn mma_f16(c: [u32; 2], a: [u32; 4], b0: u32, b1: u32) -> [u32; 2] {
    let [mut c0, mut c1] = c;
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};",
            inout("+r") c0,
            inout("+r") c1,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b0,
            in("r") b1,
            options(register_only),
        );
    }
    [c0, c1]
}

/// The shared-window address of a shared-memory pointer
#[inline(always)]
pub fn smem(p: *const u16) -> u32 {
    unsafe { cvta_generic_to_shared_u32(p as *const u8) }
}

/// Eight bytes to an 8-byte aligned shared address
#[inline(always)]
pub unsafe fn sts2(addr: u32, a: u32, b: u32) {
    unsafe {
        ptx_asm!("st.shared.v2.u32 [%0], {%1, %2};", in("r") addr, in("r") a, in("r") b, clobber("memory"));
    }
}

/// Sixteen bytes to a 16-byte aligned shared address
#[inline(always)]
pub unsafe fn sts4(addr: u32, v: [u32; 4]) {
    unsafe {
        ptx_asm!(
            "st.shared.v4.u32 [%0], {%1, %2, %3, %4};",
            in("r") addr,
            in("r") v[0],
            in("r") v[1],
            in("r") v[2],
            in("r") v[3],
            clobber("memory"),
        );
    }
}
