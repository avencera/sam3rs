//! Thin CUDA layer: one context and stream, raw device buffers, the nvcc-built kernel
//! module and row-major cuBLAS GEMMs
//!
//! Everything runs on a single stream, so buffers are plain device pointers with no
//! event tracking. That keeps the prompt path capturable as a CUDA graph

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Arc;

use cudarc::cublas::sys as blas;
use cudarc::driver::sys::{self as cu, CUdeviceptr, CUfunction, CUmodule, CUstream};
use cudarc::driver::{CudaContext, CudaStream, result as cur};
use eyre::{Result, WrapErr, bail, eyre};

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sam3rs.fatbin"));

/// A device allocation, freed in stream order on drop
pub struct Buf {
    ptr: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
}

impl Buf {
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.bytes > 0 {
            // the stream outlives every buffer because Gpu owns both
            unsafe {
                let _ = cur::free_async(self.ptr, self.stream);
            }
        }
    }
}

/// A dense row-major f32 tensor on the device
pub struct Tensor {
    buf: Buf,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn ptr(&self) -> u64 {
        self.buf.ptr
    }

    /// Device address of the element at `offset`
    pub fn at(&self, offset: usize) -> u64 {
        debug_assert!(offset <= self.len());
        self.buf.ptr + (offset * 4) as u64
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of the last dimension
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// Product of all but the last dimension
    pub fn rows(&self) -> usize {
        self.len() / self.cols().max(1)
    }

    pub fn reshape(mut self, shape: &[usize]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), self.len(), "reshape {:?} -> {shape:?}", self.shape);
        self.shape = shape.to_vec();
        self
    }
}

/// A kernel argument, kept alive until the launch call returns
pub enum Arg {
    U64(u64),
    I32(i32),
    F32(f32),
}

impl From<u64> for Arg {
    fn from(v: u64) -> Self {
        Arg::U64(v)
    }
}

impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}

impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::I32(i32::try_from(v).expect("kernel int argument overflows i32"))
    }
}

impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}

impl From<&Tensor> for Arg {
    fn from(t: &Tensor) -> Self {
        Arg::U64(t.ptr())
    }
}

impl From<&mut Tensor> for Arg {
    fn from(t: &mut Tensor) -> Self {
        Arg::U64(t.ptr())
    }
}

/// Launches a kernel from the sam3rs module: `launch!(gpu, "name", grid, block, smem; args...)`
#[macro_export]
macro_rules! launch {
    ($gpu:expr, $name:expr, $grid:expr, $block:expr, $smem:expr; $($arg:expr),* $(,)?) => {
        $gpu.launch($name, $grid, $block, $smem, vec![$($crate::gpu::Arg::from($arg)),*])
    };
}

/// One operand of a row-major GEMM: base address, leading dimension, batch stride and
/// whether it is used transposed
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub ptr: u64,
    pub ld: usize,
    pub stride: usize,
    pub trans: bool,
}

impl Mat {
    pub fn new(ptr: u64, ld: usize) -> Self {
        Self { ptr, ld, stride: 0, trans: false }
    }

    pub fn t(mut self) -> Self {
        self.trans = true;
        self
    }

    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
}

pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: CUmodule,
    functions: RefCell<HashMap<&'static str, CUfunction>>,
    blas: blas::cublasHandle_t,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).wrap_err("open CUDA device")?;
        // a single stream, so cudarc's cross-stream event tracking is pure overhead
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;

        // keep freed blocks in the async pool instead of returning them to the driver at
        // every synchronize, so per-op allocations stay cheap
        unsafe {
            let mut pool = std::mem::MaybeUninit::uninit();
            cu::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), ctx.cu_device()).result()?;
            let mut threshold = u64::MAX;
            cu::cuMemPoolSetAttribute(
                pool.assume_init(),
                cu::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
            .result()?;
        }

        let module = unsafe { cur::module::load_data(FATBIN.as_ptr().cast()) }
            .wrap_err("load sam3rs kernels (built for SAM3RS_CUDA_ARCH, default sm_120)")?;

        let mut handle = std::ptr::null_mut();
        unsafe {
            check_blas(blas::cublasCreate_v2(&mut handle))?;
            check_blas(blas::cublasSetStream_v2(handle, stream.cu_stream() as _))?;
            // FP32 means FP32: the parity reference runs with TF32 off
            check_blas(blas::cublasSetMathMode(handle, blas::cublasMath_t::CUBLAS_PEDANTIC_MATH))?;
        }

        Ok(Self { ctx, stream, module, functions: RefCell::new(HashMap::new()), blas: handle })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn cu_stream(&self) -> CUstream {
        self.stream.cu_stream()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn alloc_bytes(&self, bytes: usize) -> Result<Buf> {
        let ptr = if bytes == 0 { 0 } else { unsafe { cur::malloc_async(self.cu_stream(), bytes)? } };
        Ok(Buf { ptr, bytes, stream: self.cu_stream() })
    }

    /// Uninitialized tensor
    pub fn empty(&self, shape: &[usize]) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        Ok(Tensor { buf: self.alloc_bytes(n * 4)?, shape: shape.to_vec() })
    }

    pub fn zeros(&self, shape: &[usize]) -> Result<Tensor> {
        let t = self.empty(shape)?;
        self.memset_zero(t.ptr(), t.len() * 4)?;
        Ok(t)
    }

    pub fn memset_zero(&self, ptr: u64, bytes: usize) -> Result<()> {
        if bytes > 0 {
            unsafe { cur::memset_d8_async(ptr, 0, bytes, self.cu_stream())? };
        }
        Ok(())
    }

    pub fn upload(&self, host: &[f32], shape: &[usize]) -> Result<Tensor> {
        assert_eq!(host.len(), shape.iter().product::<usize>());
        let t = self.empty(shape)?;
        self.copy_htod(host, t.ptr())?;
        Ok(t)
    }

    pub fn upload_bytes(&self, host: &[u8]) -> Result<Buf> {
        let buf = self.alloc_bytes(host.len())?;
        self.copy_htod(host, buf.ptr())?;
        Ok(buf)
    }

    pub fn copy_htod<T>(&self, host: &[T], dst: u64) -> Result<()> {
        if !host.is_empty() {
            unsafe { cur::memcpy_htod_async(dst, host, self.cu_stream())? };
        }
        Ok(())
    }

    pub fn copy_dtod(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
        if bytes > 0 {
            unsafe { cur::memcpy_dtod_async(dst, src, bytes, self.cu_stream())? };
        }
        Ok(())
    }

    /// Copies into host memory and waits for the stream
    pub fn download_into<T>(&self, src: u64, host: &mut [T]) -> Result<()> {
        if !host.is_empty() {
            unsafe { cur::memcpy_dtoh_async(host, src, self.cu_stream())? };
        }
        self.synchronize()
    }

    pub fn download(&self, t: &Tensor) -> Result<Vec<f32>> {
        let mut host = vec![0f32; t.len()];
        self.download_into(t.ptr(), &mut host)?;
        Ok(host)
    }

    pub fn clone_tensor(&self, t: &Tensor) -> Result<Tensor> {
        let out = self.empty(t.shape())?;
        self.copy_dtod(t.ptr(), out.ptr(), t.len() * 4)?;
        Ok(out)
    }

    fn function(&self, name: &'static str) -> Result<CUfunction> {
        if let Some(f) = self.functions.borrow().get(name) {
            return Ok(*f);
        }
        let f = unsafe { cur::module::get_function(self.module, CString::new(name)?) }
            .wrap_err_with(|| format!("kernel {name} missing from the fatbin"))?;
        self.functions.borrow_mut().insert(name, f);
        Ok(f)
    }

    pub fn launch(
        &self,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        args: Vec<Arg>,
    ) -> Result<()> {
        let f = self.function(name)?;
        let mut params: Vec<*mut c_void> = args
            .iter()
            .map(|a| match a {
                Arg::U64(v) => v as *const u64 as *mut c_void,
                Arg::I32(v) => v as *const i32 as *mut c_void,
                Arg::F32(v) => v as *const f32 as *mut c_void,
            })
            .collect();
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        if smem > 48 * 1024 {
            unsafe {
                cu::cuFuncSetAttribute(
                    f,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem as i32,
                )
                .result()?;
            }
        }
        unsafe { cur::launch_kernel(f, grid, block, smem, self.cu_stream(), &mut params) }
            .wrap_err_with(|| format!("launch {name}"))?;
        Ok(())
    }

    /// Row-major `C[m,n] = alpha * op(A)[m,k] @ op(B)[k,n] + beta * C`, batched by stride
    ///
    /// Row-major C is column-major C^T = op(B)^T op(A)^T, so the operands swap
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: Mat,
        b: Mat,
        c: Mat,
        batch: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let op = |t: bool| if t { blas::cublasOperation_t::CUBLAS_OP_T } else { blas::cublasOperation_t::CUBLAS_OP_N };
        let status = unsafe {
            blas::cublasGemmStridedBatchedEx(
                self.blas,
                op(b.trans),
                op(a.trans),
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                b.ptr as *const c_void,
                blas::cudaDataType::CUDA_R_32F,
                b.ld as i32,
                b.stride as i64,
                a.ptr as *const c_void,
                blas::cudaDataType::CUDA_R_32F,
                a.ld as i32,
                a.stride as i64,
                (&beta as *const f32).cast(),
                c.ptr as *mut c_void,
                blas::cudaDataType::CUDA_R_32F,
                c.ld as i32,
                c.stride as i64,
                batch as i32,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        check_blas(status).wrap_err_with(|| format!("gemm m={m} n={n} k={k} batch={batch}"))
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stream.synchronize();
            blas::cublasDestroy_v2(self.blas);
            let _ = cur::module::unload(self.module);
        }
    }
}

fn check_blas(status: blas::cublasStatus_t) -> Result<()> {
    if status == blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        bail!("cuBLAS error {status:?}")
    }
}

/// Grid size for `n` items with `block` threads each
pub fn blocks(n: usize, block: u32) -> u32 {
    u32::try_from(n.div_ceil(block as usize)).map_err(|_| eyre!("grid too large")).unwrap()
}
