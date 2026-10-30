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

/// Kernels from `kernels/sam3.cu` as PTX for compute capability 8.0, the first with the
/// BF16 `mma.sync` the attention kernel uses. The driver JIT-compiles it for the GPU it
/// runs on; `scripts/build-ptx.sh` regenerates both tiers
static PTX_SM80: &str = concat!(include_str!("../kernels/ptx/sam3.sm80.ptx"), "\0");

/// The same kernels as compute_120 PTX for RTX 50 series and newer. On an RTX 5090 it
/// encodes about 3% faster than JIT-compiling the compute_80 tier
static PTX_SM120: &str = concat!(include_str!("../kernels/ptx/sam3.sm120.ptx"), "\0");

/// Oldest driver API the PTX (ISA 9.0) and cuBLAS 13 run on, as `cuDriverGetVersion` reports it
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

    pub fn bytes(&self) -> usize {
        self.bytes
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
    /// Strict FP32 GEMMs (no TF32) and materialized attention: the parity reference path
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
    lt: Option<BlasLt>,
    device: Arc<Device>,
    module: CUmodule,
    functions: RefCell<HashMap<&'static str, CUfunction>>,
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
        let module = unsafe { cur::module::load_data(ptx.as_ptr().cast()) }
            .wrap_err("JIT-compile the sam3rs kernels (PTX ISA 9.0 needs NVIDIA driver r580 or newer)")?;

        let mut handle = std::ptr::null_mut();
        unsafe {
            check_blas(blas::cublasCreate_v2(&mut handle))?;
            check_blas(blas::cublasSetStream_v2(handle, device.stream.cu_stream() as _))?;
        }
        // pedantic math for the parity reference, TF32 tensor cores for Tf32, and for the
        // FP32-input GEMMs of BF16 mode, tensor cores that may round inputs to BF16
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

        let lt = match precision {
            Precision::Fp32 | Precision::Tf32 => None,
            Precision::Bf16 => Some(BlasLt::new()?),
        };
        let fp32_operands = match fp32_compute {
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 => Operands::F32FastTf32,
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF => Operands::F32Fast16Bf,
            _ => Operands::F32,
        };

        Ok(Self {
            lt,
            device,
            module,
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

    /// Copies consecutive runs of `src` into each of `hosts` in turn and waits once, so
    /// results land in their final buffers without a staging copy
    pub fn download_split(&self, src: &Buf, hosts: &mut [Vec<u8>]) -> Result<()> {
        let total: usize = hosts.iter().map(Vec::len).sum();
        ensure!(total <= src.bytes, "download of {total} bytes from a {} byte buffer", src.bytes);
        let mut offset = 0;
        for host in hosts.iter_mut().filter(|h| !h.is_empty()) {
            unsafe { cur::memcpy_dtoh_async(host.as_mut_slice(), src.ptr + offset as u64, self.cu_stream())? };
            offset += host.len();
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

    fn function(&self, name: &'static str) -> Result<CUfunction> {
        if let Some(f) = self.functions.borrow().get(name) {
            return Ok(*f);
        }
        let f = unsafe { cur::module::get_function(self.module, CString::new(name)?) }
            .wrap_err_with(|| format!("kernel {name} missing from the PTX"))?;
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
        let lt = self.lt.as_ref().ok_or_else(|| eyre!("cuBLASLt linears need Precision::Bf16"))?;
        lt.linear(self.cu_stream(), Operands::Bf16, mnk, x, x_ld, w, out, beta, bias, epilogue)
    }

    /// [`Self::linear_bf16`] for FP32 `x`, `w` and `out`, with this precision's FP32
    /// compute type. Returns false without launching anything when there is no cuBLASLt
    /// handle (FP32 and TF32 modes), so the caller falls back to [`Self::gemm`]
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
    ) -> Result<bool> {
        let Some(lt) = &self.lt else { return Ok(false) };
        lt.linear(self.cu_stream(), self.fp32_operands, mnk, x, x_ld, w, (out, false), beta, bias, epilogue)?;
        Ok(true)
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
