//! Thin CUDA layer: one context and stream, device buffers from a private memory pool,
//! the prebuilt PTX kernel module and row-major cuBLAS GEMMs
//!
//! Everything runs on a single stream, so buffers carry no event tracking. That keeps the
//! prompt path capturable as a CUDA graph. Every buffer holds an `Arc` of the shared
//! [`Device`], so the stream, pool and context outlive each allocation whatever order
//! their owners drop in

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::cublas::sys as blas;
use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys::{self as cu, CUdeviceptr, CUfunction, CUmemoryPool, CUmodule, CUstream};
use cudarc::driver::{CudaContext, CudaStream, result as cur};
use eyre::{Result, WrapErr, bail, ensure, eyre};

use crate::blaslt::{BlasLt, Epilogue, Operands};

/// Kernels from the cuda-oxide crate `crates/sam3rs-kernels` as PTX for compute
/// capability 8.0, the first with the BF16 `mma.sync` the attention kernel uses. The
/// driver JIT-compiles it for the GPU it runs on; `scripts/build-ptx.sh` regenerates both
/// tiers
static PTX_SM80: &str = concat!(include_str!("../kernels/ptx/sam3.sm80.ptx"), "\0");

/// The same kernels as compute_120 PTX for RTX 50 series and newer. On an RTX 5090 it
/// encodes about 3% faster than JIT-compiling the compute_80 tier
static PTX_SM120: &str = concat!(include_str!("../kernels/ptx/sam3.sm120.ptx"), "\0");

/// Oldest driver API cuBLAS 13 runs on, as `cuDriverGetVersion` reports it. The PTX (ISA 8.7
/// at most) needs no newer one
const MIN_DRIVER: i32 = 13000;
const CUBLAS_MAJOR: i32 = 13;

/// Host bytes of in-flight uploads kept alive before the stream is drained, so loading
/// 3.4 GB of weights doesn't hold a second host copy of all of them
const PENDING_LIMIT: usize = 256 << 20;

/// The stream and memory pool every buffer frees into
struct Device {
    stream: Arc<CudaStream>,
    pool: CUmemoryPool,
}

// the pool is a driver handle, usable from any thread once the context is current there
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    fn bind(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // the last buffer is gone, so destroying the pool hands its memory back to the driver
        if self.bind().is_ok() {
            let _ = self.stream.synchronize();
            unsafe {
                let _ = cur::mem_pool::destroy(self.pool);
            }
        }
    }
}

/// A device allocation, freed in stream order on drop
pub struct Buf {
    ptr: CUdeviceptr,
    bytes: usize,
    device: Arc<Device>,
}

impl Buf {
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// The whole buffer as a kernel slice of `T`, the element type the kernel declares
    /// for it (`u16` for BF16 or FP16, `[u16; 8]` for packed groups of eight)
    pub fn slice<T>(&self) -> Arg {
        debug_assert!(self.bytes.is_multiple_of(size_of::<T>()));
        Arg::slice(self.ptr, self.bytes / size_of::<T>())
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        // a failed free leaves the block in the pool, which Device::drop destroys
        if self.bytes > 0 && self.device.bind().is_ok() {
            unsafe {
                let _ = cur::free_async(self.ptr, self.device.stream.cu_stream());
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
///
/// The kernels are cuda-oxide functions: each `&[T]` or `DisjointSlice<T>` parameter
/// lowers to two PTX params, the address then the length in elements of `T`, which is
/// [`Arg::Slice`]. Sizes are `u32` and the rest are 32-bit scalars. There is no bare
/// pointer kind, since no kernel takes one
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// Address and length in elements of the kernel's slice type
    Slice(u64, u64),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl Arg {
    /// An empty slice, for an optional input the kernel skips when it is empty. The
    /// address is never read; it is non-null and aligned because a Rust slice's must be,
    /// even when empty
    pub const EMPTY: Arg = Arg::Slice(256, 0);

    /// `len` elements of the kernel's slice type at `ptr`, for sub-views, offsets and
    /// buffers that aren't a [`Tensor`]
    pub fn slice(ptr: u64, len: usize) -> Self {
        Arg::Slice(ptr, len as u64)
    }

    /// The PTX params this argument fills, in order
    fn params(&self) -> &'static [Param] {
        match self {
            Arg::Slice(..) => &[Param::Ptr, Param::U64],
            // the PTX declares i32 params as .u32 too
            Arg::U32(_) | Arg::I32(_) => &[Param::U32],
            Arg::F32(_) => &[Param::F32],
        }
    }
}

impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}

impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::U32(u32::try_from(v).expect("kernel size argument overflows u32"))
    }
}

impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}

/// The whole tensor as an f32 slice
impl From<&Tensor> for Arg {
    fn from(t: &Tensor) -> Self {
        Arg::slice(t.ptr(), t.len())
    }
}

impl From<&mut Tensor> for Arg {
    fn from(t: &mut Tensor) -> Self {
        Arg::slice(t.ptr(), t.len())
    }
}

/// Launches a kernel from the sam3rs module: `launch!(gpu, "name", grid, block, smem;
/// args...)`, each argument converted with `Arg::from`: a [`Tensor`] is its f32 slice, a
/// `usize` a `u32` size. Use [`Arg::slice`] or [`Buf::slice`] for other slices
macro_rules! launch {
    ($gpu:expr, $name:expr, $grid:expr, $block:expr, $smem:expr; $($arg:expr),* $(,)?) => {
        $gpu.launch($name, $grid, $block, $smem, vec![$($crate::gpu::Arg::from($arg)),*])
    };
}
pub(crate) use launch;

/// One operand of a row-major GEMM: base address, leading dimension, batch stride and
/// whether it is used transposed
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub ptr: u64,
    pub ld: usize,
    pub stride: usize,
    pub trans: bool,
    /// BF16 elements instead of FP32 (both GEMM inputs must agree; C stays FP32)
    pub bf16: bool,
}

impl Mat {
    pub fn new(ptr: u64, ld: usize) -> Self {
        Self { ptr, ld, stride: 0, trans: false, bf16: false }
    }

    pub fn bf16(mut self) -> Self {
        self.bf16 = true;
        self
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

/// Numeric mode for the whole pipeline
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Strict FP32 (no TF32 or BF16 anywhere): the parity reference path. The vision
    /// backbone's attention runs a fused flash-attention kernel on FP32 FMAs; the
    /// detector's attention stays materialized
    Fp32,
    /// The FP32 path with TF32 tensor-core GEMMs (FP32 accumulation), the way ONNX
    /// Runtime's CUDA provider runs FP32 models by default
    Tf32,
    /// Tensor-core GEMMs with FP32 accumulation everywhere. The vision backbone, necks
    /// and text encoder MLP keep BF16 weights and activations; the rest of the text
    /// encoder, the fusion encoder, detector and point decoder keep FP32 ones, and cuBLAS
    /// may round their GEMM inputs to BF16 (`CUBLAS_COMPUTE_32F_FAST_16BF`), though it
    /// picks TF32 kernels for most of them. Unmasked attention with head dim 32 or 64,
    /// and the detector's box-biased cross-attention, run the fused kernel, which rounds
    /// q, k and v to BF16
    Bf16,
}

impl std::str::FromStr for Precision {
    type Err = eyre::Report;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "fp32" => Ok(Self::Fp32),
            "tf32" => Ok(Self::Tf32),
            "bf16" => Ok(Self::Bf16),
            _ => bail!("unknown precision {s:?}, expected fp32, tf32 or bf16"),
        }
    }
}

pub struct Gpu {
    // first, so it is dropped while the context is still alive
    lt: BlasLt,
    device: Arc<Device>,
    module: CUmodule,
    /// the PTX `module` was loaded from, which the launch guard reads params from
    ptx: &'static str,
    functions: RefCell<HashMap<&'static str, Kernel>>,
    blas: blas::cublasHandle_t,
    /// compute type of GEMMs with FP32 inputs
    fp32_compute: blas::cublasComputeType_t,
    /// the same for cuBLASLt linears with FP32 inputs
    fp32_operands: Operands,
    precision: Precision,
    /// host sources of asynchronous uploads, kept until the stream next synchronizes
    pending: RefCell<Vec<Box<dyn Any + Send>>>,
    pending_bytes: Cell<usize>,
}

// CUDA handles work from any thread once the context is current there. `bind` makes it
// current at every public entry point, and Gpu isn't Sync, so one thread uses it at a time
unsafe impl Send for Gpu {}

impl Gpu {
    pub fn new(ordinal: usize, precision: Precision) -> Result<Self> {
        check_libraries()?;
        let ctx = CudaContext::new(ordinal).wrap_err_with(|| format!("open CUDA device {ordinal}"))?;
        let (major, minor) = ctx.compute_capability()?;
        if major < 8 {
            bail!("GPU {ordinal} has compute capability {major}.{minor}; sam3rs needs 8.0 (Ampere) or newer");
        }
        // a single stream, so cudarc's cross-stream event tracking is pure overhead
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let device = Arc::new(Device { pool: create_pool(&ctx)?, stream });

        // PTX JIT-compiles forward to any newer GPU, never back to an older one
        let ptx = if major >= 12 { PTX_SM120 } else { PTX_SM80 };
        let module =
            unsafe { cur::module::load_data(ptx.as_ptr().cast()) }.wrap_err("JIT-compile the sam3rs kernels")?;

        let mut handle = std::ptr::null_mut();
        unsafe {
            check_blas(blas::cublasCreate_v2(&mut handle))?;
            check_blas(blas::cublasSetStream_v2(handle, device.stream.cu_stream() as _))?;
        }
        // pedantic math for the parity reference (default math measured the same on an RTX
        // 5090), TF32 tensor cores for Tf32, and for the FP32-input GEMMs of BF16 mode,
        // tensor cores that may round inputs to BF16
        let (math, fp32_compute) = match precision {
            Precision::Fp32 => {
                (blas::cublasMath_t::CUBLAS_PEDANTIC_MATH, blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC)
            }
            Precision::Tf32 => {
                (blas::cublasMath_t::CUBLAS_DEFAULT_MATH, blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32)
            }
            // strict FP32 compute here cost a third of a detect call with no parity gain,
            // and FAST_16BF kept more score headroom than FAST_TF32 on the parity set. cuBLAS
            // still picks TF32 kernels for most of these shapes on an RTX 5090
            Precision::Bf16 => {
                (blas::cublasMath_t::CUBLAS_DEFAULT_MATH, blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF)
            }
        };
        unsafe { check_blas(blas::cublasSetMathMode(handle, math))? };

        // every mode fuses bias and activation into cuBLASLt epilogues; FP32 mode's run with
        // plain FP32 compute, so the products are as strict as its cuBLAS GEMMs
        let lt = BlasLt::new()?;
        let fp32_operands = match fp32_compute {
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 => Operands::F32FastTf32,
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF => Operands::F32Fast16Bf,
            _ => Operands::F32,
        };

        Ok(Self {
            lt,
            device,
            module,
            ptx,
            functions: RefCell::new(HashMap::new()),
            blas: handle,
            fp32_compute,
            fp32_operands,
            precision,
            pending: RefCell::new(Vec::new()),
            pending_bytes: Cell::new(0),
        })
    }

    /// Makes this GPU's context current on the calling thread
    pub fn bind(&self) -> Result<()> {
        self.device.bind()
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    fn cu_stream(&self) -> CUstream {
        self.device.stream.cu_stream()
    }

    /// Waits for the stream, then releases the host sources of finished uploads
    pub fn synchronize(&self) -> Result<()> {
        self.device.stream.synchronize()?;
        self.pending.borrow_mut().clear();
        self.pending_bytes.set(0);
        Ok(())
    }

    pub fn alloc_bytes(&self, bytes: usize) -> Result<Buf> {
        let mut ptr = 0;
        if bytes > 0 {
            unsafe { cu::cuMemAllocFromPoolAsync(&mut ptr, bytes, self.device.pool, self.cu_stream()).result() }
                .wrap_err_with(|| format!("allocate {bytes} bytes on the GPU"))?;
        }
        Ok(Buf { ptr, bytes, device: self.device.clone() })
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

    pub fn upload(&self, host: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        ensure!(host.len() == shape.iter().product::<usize>(), "upload of {} floats as {shape:?}", host.len());
        Ok(Tensor { buf: self.upload_buf(host)?, shape: shape.to_vec() })
    }

    /// Copies `host` into a new buffer. The copy is asynchronous, so `host` is kept alive
    /// until the stream next synchronizes
    pub fn upload_buf<T: Pod + Send>(&self, host: Vec<T>) -> Result<Buf> {
        let bytes: &[u8] = bytemuck::cast_slice(&host);
        let buf = self.alloc_bytes(bytes.len())?;
        if bytes.is_empty() {
            return Ok(buf);
        }
        unsafe { cur::memcpy_htod_async(buf.ptr, bytes, self.cu_stream())? };
        self.pending_bytes.set(self.pending_bytes.get() + bytes.len());
        self.pending.borrow_mut().push(Box::new(host));
        if self.pending_bytes.get() > PENDING_LIMIT {
            self.synchronize()?;
        }
        Ok(buf)
    }

    /// Copies borrowed host bytes into a new buffer and waits for the copy, since the
    /// caller may reuse them as soon as this returns
    pub fn upload_slice(&self, host: &[u8]) -> Result<Buf> {
        let buf = self.alloc_bytes(host.len())?;
        if !host.is_empty() {
            unsafe { cur::memcpy_htod_async(buf.ptr, host, self.cu_stream())? };
            self.synchronize()?;
        }
        Ok(buf)
    }

    pub fn copy_dtod(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
        if bytes > 0 {
            unsafe { cur::memcpy_dtod_async(dst, src, bytes, self.cu_stream())? };
        }
        Ok(())
    }

    /// Copies the start of `src` into `host` and waits for the stream
    pub fn download_into<T: Pod>(&self, src: &Buf, host: &mut [T]) -> Result<()> {
        let host: &mut [u8] = bytemuck::cast_slice_mut(host);
        ensure!(host.len() <= src.bytes, "download of {} bytes from a {} byte buffer", host.len(), src.bytes);
        if !host.is_empty() {
            unsafe { cur::memcpy_dtoh_async(host, src.ptr, self.cu_stream())? };
        }
        self.synchronize()
    }

    pub fn download(&self, t: &Tensor) -> Result<Vec<f32>> {
        let mut host = vec![0f32; t.len()];
        self.download_into(&t.buf, &mut host)?;
        Ok(host)
    }

    pub fn clone_tensor(&self, t: &Tensor) -> Result<Tensor> {
        let out = self.empty(t.shape())?;
        self.copy_dtod(t.ptr(), out.ptr(), t.len() * 4)?;
        Ok(out)
    }

    /// The kernel `name`, once its PTX params match `args`. The params are read from
    /// the loaded PTX on first use and cached with the function, so later launches only
    /// compare them
    fn function(&self, name: &'static str, args: &[Arg]) -> Result<CUfunction> {
        if let Some(k) = self.functions.borrow().get(name) {
            k.check(name, args)?;
            return Ok(k.f);
        }
        let f = unsafe { cur::module::get_function(self.module, CString::new(name)?) }
            .wrap_err_with(|| format!("kernel {name} missing from the PTX"))?;
        let k = Kernel { f, params: entry_params(self.ptx, name)? };
        k.check(name, args)?;
        self.functions.borrow_mut().insert(name, k);
        Ok(f)
    }

    /// Launches `name` with one PTX param per scalar and two per slice. Fails without
    /// launching if `args` don't match the kernel's PTX params in count and type, which
    /// the driver would otherwise accept and misread
    pub fn launch(
        &self,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        args: Vec<Arg>,
    ) -> Result<()> {
        let f = self.function(name, &args)?;
        // the 32-bit thread index of the elementwise kernels is exact only up to here
        ensure!(grid.0 as u64 * block.0 as u64 <= 1 << 32, "launch {name}: more than 2^32 threads along x");
        let mut params: Vec<*mut c_void> = Vec::with_capacity(2 * args.len());
        for a in &args {
            match a {
                Arg::Slice(ptr, len) => {
                    params.extend([ptr as *const u64 as *mut c_void, len as *const u64 as *mut c_void]);
                }
                Arg::U32(v) => params.push(v as *const u32 as *mut c_void),
                Arg::I32(v) => params.push(v as *const i32 as *mut c_void),
                Arg::F32(v) => params.push(v as *const f32 as *mut c_void),
            }
        }
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

    /// See [`BlasLt::linear`]; only available in BF16 mode
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        &self,
        mnk: (usize, usize, usize),
        x: u64,
        x_ld: usize,
        w: u64,
        out: (u64, bool),
        beta: f32,
        bias: u64,
        epilogue: Epilogue,
    ) -> Result<()> {
        self.lt.linear(self.cu_stream(), Operands::Bf16, mnk, x, x_ld, w, out, beta, bias, epilogue)
    }

    /// [`Self::linear_bf16`] for FP32 `x`, `w` and `out`, with this precision's FP32
    /// compute type
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(
        &self,
        mnk: (usize, usize, usize),
        x: u64,
        x_ld: usize,
        w: u64,
        out: u64,
        beta: f32,
        bias: u64,
        epilogue: Epilogue,
    ) -> Result<()> {
        self.lt.linear(self.cu_stream(), self.fp32_operands, mnk, x, x_ld, w, (out, false), beta, bias, epilogue)
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
        assert_eq!(a.bf16, b.bf16, "gemm inputs must share a type");
        assert!(!c.bf16, "gemm output is FP32");
        let (ty, compute) = if a.bf16 {
            (blas::cudaDataType::CUDA_R_16BF, blas::cublasComputeType_t::CUBLAS_COMPUTE_32F)
        } else {
            (blas::cudaDataType::CUDA_R_32F, self.fp32_compute)
        };
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
                ty,
                b.ld as i32,
                b.stride as i64,
                a.ptr as *const c_void,
                ty,
                a.ld as i32,
                a.stride as i64,
                (&beta as *const f32).cast(),
                c.ptr as *mut c_void,
                blas::cudaDataType::CUDA_R_32F,
                c.ld as i32,
                c.stride as i64,
                batch as i32,
                compute,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        check_blas(status).wrap_err_with(|| format!("gemm m={m} n={n} k={k} batch={batch}"))
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if self.bind().is_err() {
            return;
        }
        unsafe {
            let _ = self.device.stream.synchronize();
            blas::cublasDestroy_v2(self.blas);
            let _ = cur::module::unload(self.module);
        }
    }
}

/// A PTX kernel param as the host fills it: slices are a `.u64 .ptr` address then a
/// `.u64` length
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    Ptr,
    U64,
    U32,
    F32,
}

/// A loaded kernel and its PTX params
struct Kernel {
    f: CUfunction,
    params: Vec<Param>,
}

impl Kernel {
    /// Checks that `args` fill this kernel's params exactly: `cuLaunchKernel` reads each
    /// param at the size the PTX declares, so a missing slice length or a dropped
    /// scalar would shift every later argument silently, and a float passed for a size
    /// would be read as its bits
    fn check(&self, name: &str, args: &[Arg]) -> Result<()> {
        let mut host = args.iter().flat_map(Arg::params);
        for (i, &ptx) in self.params.iter().enumerate() {
            match host.next() {
                Some(&param) if param == ptx => {}
                Some(&param) => bail!("kernel {name} param {i}: the PTX declares {ptx:?}, the host passes {param:?}"),
                None => bail!("kernel {name} takes {} params, the host passes {i}", self.params.len()),
            }
        }
        let extra = host.count();
        ensure!(
            extra == 0,
            "kernel {name} takes {} params, the host passes {}",
            self.params.len(),
            self.params.len() + extra
        );
        Ok(())
    }
}

/// The params of kernel `name` in `ptx`, in order. Read from the PTX text rather than the
/// driver: `cuFuncGetParamCount` needs a newer driver than sam3rs supports, and probing
/// `cuFuncGetParamInfo` past the last param is an API error that compute-sanitizer
/// reports on every kernel
fn entry_params(ptx: &str, name: &str) -> Result<Vec<Param>> {
    let entry = format!(".entry {name}(");
    let start = ptx.find(&entry).ok_or_else(|| eyre!("kernel {name} missing from the PTX"))? + entry.len();
    let end = ptx[start..].find(')').ok_or_else(|| eyre!("kernel {name}: unterminated PTX param list"))?;
    ptx[start..start + end]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let ty = line.strip_prefix(".param ").and_then(|rest| rest.split_whitespace().next());
            match (ty, line.contains(" .ptr ")) {
                (Some(".u64"), true) => Ok(Param::Ptr),
                (Some(".u64"), false) => Ok(Param::U64),
                (Some(".u32"), false) => Ok(Param::U32),
                (Some(".f32"), false) => Ok(Param::F32),
                _ => Err(eyre!("kernel {name}: unexpected PTX param `{line}`")),
            }
        })
        .collect()
}

/// A pool of our own instead of the device default, so keeping freed blocks cached
/// doesn't change the default pool's behavior for other CUDA users in the process
fn create_pool(ctx: &CudaContext) -> Result<CUmemoryPool> {
    let pool = unsafe {
        let mut props: cu::CUmemPoolProps = std::mem::zeroed();
        props.allocType = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = ctx.cu_device();
        cur::mem_pool::create(&props).wrap_err("create the CUDA memory pool")?
    };
    // keep freed blocks instead of returning them to the driver at every synchronize, so
    // per-op allocations stay cheap
    let mut threshold = u64::MAX;
    unsafe {
        cur::mem_pool::set_attribute(
            pool,
            cu::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut threshold as *mut u64).cast(),
        )?;
    }
    Ok(pool)
}

/// Checks the run-time libraries before cudarc touches them: its loader panics on a
/// missing library or symbol, and it also accepts older cuBLAS sonames whose ABI differs
fn check_libraries() -> Result<()> {
    unsafe {
        if !cu::is_culib_present() {
            bail!("no NVIDIA driver found (libcuda.so.1); sam3rs needs an NVIDIA GPU with driver r580 or newer");
        }
        let mut version = 0;
        cu::cuDriverGetVersion(&mut version).result()?;
        if version < MIN_DRIVER {
            bail!(
                "the NVIDIA driver supports CUDA {}.{}; sam3rs needs CUDA 13.0 or newer (driver r580+)",
                version / 1000,
                version % 1000 / 10
            );
        }

        let hint =
            "install the CUDA 13 toolkit or `pip install nvidia-cublas`, and put its lib directory on LD_LIBRARY_PATH";
        if !blas::is_culib_present() || !lt::is_culib_present() {
            bail!("cuBLAS 13 not found (libcublas.so.13, libcublasLt.so.13): {hint}");
        }
        let mut major = 0;
        check_blas(blas::cublasGetProperty(blas::libraryPropertyType::MAJOR_VERSION, &mut major))?;
        let mut lt_major = 0;
        if lt::cublasLtGetProperty(lt::libraryPropertyType::MAJOR_VERSION, &mut lt_major)
            != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
        {
            bail!("cuBLASLt version query failed");
        }
        if major != CUBLAS_MAJOR || lt_major != CUBLAS_MAJOR {
            bail!("found cuBLAS {major} and cuBLASLt {lt_major}, sam3rs needs version {CUBLAS_MAJOR}: {hint}");
        }
    }
    Ok(())
}

fn check_blas(status: blas::cublasStatus_t) -> Result<()> {
    if status == blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { bail!("cuBLAS error {status:?}") }
}

/// Grid size for `n` items with `block` threads each
pub fn blocks(n: usize, block: u32) -> u32 {
    u32::try_from(n.div_ceil(block as usize)).expect("grid too large")
}

#[cfg(test)]
mod tests {
    use super::{PTX_SM80, PTX_SM120, entry_params};

    /// The launch guard reads every kernel's params from the PTX text, so a cuda-oxide or
    /// LLVM upgrade that changes the param syntax must fail here, not at a launch
    #[test]
    fn ptx_params_parse() {
        let names: Vec<&str> = PTX_SM120
            .match_indices(".entry ")
            .map(|(i, m)| {
                let rest = &PTX_SM120[i + m.len()..];
                &rest[..rest.find('(').unwrap()]
            })
            .collect();
        assert!(names.len() > 40, "found {} entries", names.len());
        for name in names {
            let params = entry_params(PTX_SM120, name).unwrap();
            assert!(!params.is_empty(), "{name} has no params");
            assert_eq!(params, entry_params(PTX_SM80, name).unwrap(), "{name} differs between the tiers");
        }
    }
}
