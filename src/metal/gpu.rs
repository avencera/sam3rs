//! Thin Metal layer: one device and command queue, pooled shared-memory buffers, the MSL
//! kernel library compiled at load, and command batching
//!
//! Every dispatch goes into one serial compute encoder, so kernels run in the order they
//! were encoded and each sees the writes of the ones before it, like a CUDA stream. The
//! encoder's command buffer is committed every [`FLUSH_EVERY`] dispatches so the GPU
//! starts while the host is still encoding, and [`Gpu::synchronize`] commits and waits.
//!
//! Apple silicon memory is unified: buffers use shared storage, so uploads and downloads
//! are host copies into and out of the buffer's contents, done only when no queued
//! kernel can still touch those bytes

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

use bytemuck::Pod;
use eyre::{Result, bail, ensure, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo, NSRange, NSString};
use objc2_metal::{
    MTLBinding, MTLBindingType, MTLBuffer, MTLBufferBinding, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCommonCounterSetTimestamp, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLGPUFamily,
    MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLPipelineOption, MTLResourceOptions, MTLSize, MTLStorageMode,
};

use crate::precision::Precision;

/// The kernel library's sources, compiled together in this order: the shared header
/// first, then one file per area
const SOURCES: [&str; 10] = [
    include_str!("kernels/common.metal"),
    include_str!("kernels/elementwise.metal"),
    include_str!("kernels/norm.metal"),
    include_str!("kernels/gemm.metal"),
    include_str!("kernels/attention.metal"),
    include_str!("kernels/vision.metal"),
    include_str!("kernels/masks.metal"),
    include_str!("kernels/detector.metal"),
    include_str!("kernels/point.metal"),
    include_str!("kernels/tracker.metal"),
];

/// Dispatches encoded before the open command buffer is committed, so the GPU works
/// through the first part of a long pass while the host encodes the rest
const FLUSH_EVERY: usize = 64;

/// Freed buffer bytes the pool keeps for reuse before it hands the oldest back to Metal
const POOL_LIMIT: usize = 4 << 30;

/// Element types of tensors, matching the kernels' MSL types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    U8,
    I32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::Bf16 => 2,
            DType::U8 => 1,
        }
    }

    /// The Metal data type a kernel declares for a pointer to this element type
    fn metal(self) -> MTLDataType {
        match self {
            DType::F32 => MTLDataType::Float,
            DType::Bf16 => MTLDataType::BFloat,
            DType::U8 => MTLDataType::UChar,
            DType::I32 => MTLDataType::Int,
        }
    }
}

/// An `MTLBuffer` that may move between threads
///
/// Metal buffers are thread-safe objects; what isn't is touching their contents while a
/// kernel uses them, which [`Gpu`] orders itself
struct Raw(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: see `Raw`
unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

/// A freed buffer, the batch that last used it, and its stamp in [`PoolState::order`]
struct Free {
    raw: Raw,
    batch: u64,
    stamp: u64,
}

/// Freed buffers by size class
#[derive(Default)]
struct PoolState {
    free: HashMap<usize, Vec<Free>>,
    /// The size class of every cached buffer by stamp, oldest first, so eviction drops the
    /// oldest; a reused buffer leaves it, which keeps it as long as the cache
    order: BTreeMap<u64, usize>,
    next_stamp: u64,
    bytes: usize,
}

/// Buffer pool shared by the model's buffers, which return to it on drop
struct Pool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    state: Mutex<PoolState>,
    /// The batch being encoded: a buffer freed now may still be read by it
    open: std::sync::atomic::AtomicU64,
}

// SAFETY: `MTLDevice` is thread-safe and the rest is behind the mutex or atomic
unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

/// Rounds an allocation up to its size class: powers of two up to 64 KiB, then 64 KiB
/// steps up to 4 MiB, then 1 MiB steps, so reuse wastes at most a few percent
fn size_class(bytes: usize) -> usize {
    let bytes = bytes.max(256);
    if bytes <= 64 << 10 {
        bytes.next_power_of_two()
    } else if bytes <= 4 << 20 {
        bytes.next_multiple_of(64 << 10)
    } else {
        bytes.next_multiple_of(1 << 20)
    }
}

impl Pool {
    /// A buffer of at least `bytes`. With `completed`, only a buffer whose last batch has
    /// finished, so the host can write it now
    fn take(self: &Arc<Self>, bytes: usize, completed: Option<u64>) -> Result<Buf> {
        let class = size_class(bytes);
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let found = st.free.get_mut(&class).and_then(|list| {
            let i = match completed {
                // the oldest is the likeliest to be done
                Some(done) => list.iter().position(|f| f.batch <= done)?,
                None => list.len().checked_sub(1)?,
            };
            Some(list.swap_remove(i))
        });
        let raw = match found {
            Some(free) => {
                st.order.remove(&free.stamp);
                st.bytes -= class;
                free.raw
            }
            None => {
                drop(st);
                let raw = self
                    .device
                    .newBufferWithLength_options(class, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| eyre!("Metal could not allocate a {class}-byte buffer"))?;
                Raw(raw)
            }
        };
        Ok(Buf { raw: Some(raw), bytes, pool: self.clone() })
    }

    fn give(&self, raw: Raw) {
        let class = raw.0.length();
        let batch = self.open.load(std::sync::atomic::Ordering::Relaxed);
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let stamp = st.next_stamp;
        st.next_stamp += 1;
        st.free.entry(class).or_default().push(Free { raw, batch, stamp });
        st.order.insert(stamp, class);
        st.bytes += class;
        while st.bytes > POOL_LIMIT {
            let Some((stamp, oldest)) = st.order.pop_first() else { break };
            if let Some(list) = st.free.get_mut(&oldest)
                && let Some(i) = list.iter().position(|f| f.stamp == stamp)
            {
                // the GPU retains a buffer its queued commands use, so releasing ours is safe
                list.swap_remove(i);
                st.bytes -= oldest;
            }
        }
    }
}

/// A shared-storage device allocation, returned to the pool on drop
pub struct Buf {
    raw: Option<Raw>,
    bytes: usize,
    pool: Arc<Pool>,
}

impl Buf {
    fn mtl(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw.as_ref().expect("live buffer").0
    }

    /// The whole buffer as a kernel argument
    pub fn arg(&self) -> Arg<'_> {
        Arg::Buf(self, 0, None)
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            self.pool.give(raw);
        }
    }
}

/// A dense row-major tensor on the GPU
pub struct Tensor {
    pub buf: Buf,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
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

    /// The tensor as a kernel argument, which the dispatch checks against the element
    /// type the kernel declares
    pub fn arg(&self) -> Arg<'_> {
        Arg::Buf(&self.buf, 0, Some(self.dtype))
    }

    /// The tensor from element `offset` on as a kernel argument
    pub fn at(&self, offset: usize) -> Arg<'_> {
        debug_assert!(offset <= self.len());
        Arg::Buf(&self.buf, offset * self.dtype.size(), Some(self.dtype))
    }
}

/// A kernel argument, bound to `[[buffer(i)]]` in order
///
/// Buffers carry their element type when they are a [`Tensor`], which the dispatch checks
/// against the kernel's pointer type; scalars bind as `constant` values
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    /// A buffer from a byte offset, with the tensor element type if known
    Buf(&'a Buf, usize, Option<DType>),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl From<u32> for Arg<'_> {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

impl From<i32> for Arg<'_> {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}

impl From<usize> for Arg<'_> {
    fn from(v: usize) -> Self {
        Arg::U32(u32::try_from(v).expect("kernel size argument overflows u32"))
    }
}

impl From<f32> for Arg<'_> {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}

impl<'a> From<&'a Tensor> for Arg<'a> {
    fn from(t: &'a Tensor) -> Self {
        t.arg()
    }
}

impl<'a> From<&'a Buf> for Arg<'a> {
    fn from(b: &'a Buf) -> Self {
        b.arg()
    }
}

/// What a kernel parameter binds, read from the pipeline's reflection
#[derive(Clone, Copy, Debug, PartialEq)]
enum Param {
    /// A device pointer and its element type
    Pointer(MTLDataType),
    /// A `constant` 32-bit scalar
    Scalar(MTLDataType),
}

/// A compiled kernel and its parameter list
struct Kernel {
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    params: Vec<Param>,
}

/// Every kernel of the library by function name
type Kernels = HashMap<String, Kernel>;

/// Compiles the kernel library and builds a pipeline for each of its kernels, spread over
/// a few threads: building them all up front keeps pipeline compiles out of the first
/// encode, and [`Gpu::new`] runs this on a background thread behind the weight load
fn build_kernels(device: &ProtocolObject<dyn MTLDevice>) -> Result<Kernels> {
    let options = MTLCompileOptions::new();
    // Metal 4 for the tensor ops the GEMMs use; safe math so exp, tanh and NaN
    // handling follow IEEE rather than fast-math approximations
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    options.setMathMode(MTLMathMode::Safe);
    let source = NSString::from_str(&SOURCES.concat());
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|e| eyre!("compile the Metal kernels (needs macOS 26.1 or newer): {}", e.localizedDescription()))?;
    let names: Vec<String> = library.functionNames().iter().map(|n| n.to_string()).collect();
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).min(8);
    let library = &library;
    std::thread::scope(|s| {
        let workers: Vec<_> = names
            .chunks(names.len().div_ceil(threads).max(1))
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|name| Ok((name.clone(), build_kernel(device, library, name)?)))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect();
        let mut kernels = Kernels::new();
        for w in workers {
            kernels.extend(w.join().map_err(|_| eyre!("a Metal pipeline build panicked"))??);
        }

        Ok(kernels)
    })
}

/// The pipeline of kernel `name` and its parameter list, read from reflection
fn build_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Kernel> {
    let func =
        library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| eyre!("no Metal kernel named {name}"))?;
    let mut reflection = None;
    // SAFETY: the function comes from our library and is only dispatched through
    // `Gpu::dispatch`, which checks its arguments against this reflection
    let pso = unsafe {
        device.newComputePipelineStateWithFunction_options_reflection_error(
            &func,
            MTLPipelineOption::BindingInfo | MTLPipelineOption::BufferTypeInfo,
            Some(&mut reflection),
        )
    }
    .map_err(|e| eyre!("build Metal pipeline {name}: {}", e.localizedDescription()))?;
    let reflection = reflection.ok_or_else(|| eyre!("no reflection for Metal kernel {name}"))?;
    let mut bindings: Vec<(usize, Param)> = Vec::new();
    for b in reflection.bindings().iter() {
        if b.r#type() != MTLBindingType::Buffer {
            continue;
        }
        // SAFETY: a binding of type Buffer is an MTLBufferBinding
        let buffer: &ProtocolObject<dyn MTLBufferBinding> =
            unsafe { &*(&*b as *const ProtocolObject<dyn MTLBinding>).cast() };
        let ty = buffer.bufferDataType();
        // a pointer's binding carries a pointer type; a `constant T&` scalar doesn't
        let param = if buffer.bufferPointerType().is_some_and(|p| p.elementIsArgumentBuffer())
            || buffer.bufferDataSize() != 4
            || !matches!(ty, MTLDataType::UInt | MTLDataType::Int | MTLDataType::Float)
            || is_pointer_name(&b.name().to_string())
        {
            Param::Pointer(ty)
        } else {
            Param::Scalar(ty)
        };
        bindings.push((b.index(), param));
    }
    bindings.sort_by_key(|(i, _)| *i);
    for (pos, (i, _)) in bindings.iter().enumerate() {
        ensure!(*i == pos, "Metal kernel {name}: buffer indices must run 0..n without gaps, found {i} at {pos}");
    }

    Ok(Kernel { pso, params: bindings.into_iter().map(|(_, p)| p).collect() })
}

/// The open command buffer and its encoder
struct Batch {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    dispatches: usize,
}

/// Timestamp samples a profiling window holds: 32 KB of them, the most one counter
/// sample buffer may take, for 2048 dispatches
const PROFILE_SAMPLES: usize = 4096;

/// GPU timestamps around every dispatch, for `SAM3RS_METAL_PROFILE`
///
/// Each dispatch gets its own encoder, sampled at its start and end, since Apple GPUs
/// sample only at encoder boundaries; the encoders share command buffers as usual, so
/// the GPU stays busy between kernels. Waiting for a command buffer per dispatch instead
/// let it idle and clock down, which tripled small kernels' times. A dispatch's time is
/// what it adds to the GPU timeline, launch gaps included, so the times sum to the busy
/// time; encoder switches still cost a little over unprofiled runs
struct Profiler {
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    /// The kernel of each dispatch sampled since the last resolve: dispatch i's start
    /// and end land in samples 2i and 2i + 1
    pending: Vec<&'static str>,
    /// GPU seconds of every resolved dispatch
    done: Vec<(&'static str, f64)>,
    /// CPU nanoseconds and GPU ticks when profiling began, for the tick rate
    origin: (u64, u64),
    /// The latest end sample resolved, in GPU ticks
    last_end: u64,
}

impl Profiler {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        ensure!(
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            "SAM3RS_METAL_PROFILE: this GPU can't sample timestamps at encoder boundaries"
        );
        // SAFETY: a static NSString constant of the Metal framework
        let want = unsafe { MTLCommonCounterSetTimestamp };
        let set = device
            .counterSets()
            .and_then(|sets| sets.iter().find(|s| MTLCounterSet::name(&**s).isEqualToString(want)))
            .ok_or_else(|| eyre!("SAM3RS_METAL_PROFILE: no timestamp counter set"))?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: within the 32 KB a sample buffer may hold
        unsafe { desc.setSampleCount(PROFILE_SAMPLES) };
        let samples = device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| eyre!("SAM3RS_METAL_PROFILE: {}", e.localizedDescription()))?;
        Ok(Self { samples, pending: Vec::new(), done: Vec::new(), origin: timestamps(device), last_end: 0 })
    }

    /// Whether the sample buffer has room for no more dispatches
    fn full(&self) -> bool {
        2 * self.pending.len() >= PROFILE_SAMPLES
    }

    /// A pass descriptor sampling the next dispatch, of kernel `name`
    fn sample(&mut self, name: &'static str) -> Retained<MTLComputePassDescriptor> {
        let i = 2 * self.pending.len();
        self.pending.push(name);
        let desc = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: index 0 exists on every pass descriptor, and both sample indices are
        // inside the buffer, since `full` flushes before it runs out
        unsafe {
            let att = desc.sampleBufferAttachments().objectAtIndexedSubscript(0);
            att.setSampleBuffer(Some(&self.samples));
            att.setStartOfEncoderSampleIndex(i);
            att.setEndOfEncoderSampleIndex(i + 1);
        }
        desc
    }

    /// Reads the samples of the pending dispatches, whose command buffers have completed
    fn resolve(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let n = 2 * self.pending.len();
        // SAFETY: the range lies inside the buffer and every command buffer that wrote it
        // has completed
        let data = unsafe { self.samples.resolveCounterRange(NSRange::new(0, n)) }
            .ok_or_else(|| eyre!("SAM3RS_METAL_PROFILE: could not resolve the timestamps"))?;
        let bytes = data.to_vec();
        ensure!(bytes.len() >= 8 * n, "SAM3RS_METAL_PROFILE: {} bytes of timestamps for {n}", bytes.len());
        let ticks: Vec<u64> = bytes.as_chunks::<8>().0.iter().map(|c| u64::from_ne_bytes(*c)).collect();
        let now = timestamps(device);
        let ns_per_tick = (now.0 - self.origin.0) as f64 / (now.1 - self.origin.1).max(1) as f64;
        for (i, name) in self.pending.drain(..).enumerate() {
            let (a, b) = (ticks[2 * i], ticks[2 * i + 1]);
            // a sample the GPU couldn't take reads as all ones
            if a == u64::MAX || b == u64::MAX || b < a {
                continue;
            }
            // an encoder's start sample comes before it waits for the encoders it depends
            // on, so intervals overlap; each dispatch is charged the time it adds to the
            // timeline, which sums to the GPU's busy time
            let from = a.max(self.last_end);
            self.done.push((name, b.saturating_sub(from) as f64 * ns_per_tick * 1e-9));
            self.last_end = self.last_end.max(b);
        }
        Ok(())
    }
}

/// The CPU (nanoseconds) and GPU (ticks) clocks read together
fn timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0u64, 0u64);
    // SAFETY: both pointers are to live u64s
    unsafe { device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu)) };
    (cpu, gpu)
}

/// Committed command buffers not yet seen complete, with their batch numbers
type Inflight = VecDeque<(u64, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>;

/// A failed command buffer's error description
fn failure_reason(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> String {
    cb.error().map_or_else(|| "unknown error".into(), |e| e.localizedDescription().to_string())
}

pub struct Gpu {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// The kernel build [`Gpu::new`] started, until the first use joins it
    build: RefCell<Option<JoinHandle<Result<Kernels>>>>,
    kernels: OnceCell<Kernels>,
    batch: RefCell<Option<Batch>>,
    inflight: RefCell<Inflight>,
    /// The highest batch known complete
    completed: Cell<u64>,
    /// The first failed batch's error. Every later `synchronize` returns it, like a CUDA
    /// context fault: outputs that batch left unwritten may already be cached (the visual
    /// prompt's text features, video memory), so nothing after it can be trusted
    failure: RefCell<Option<String>>,
    pool: Arc<Pool>,
    precision: Precision,
    /// GPU time of each dispatch, when `SAM3RS_METAL_PROFILE` is set
    profile: Option<RefCell<Profiler>>,
}

// SAFETY: `Gpu` is not `Sync` (its `RefCell`s), so one thread at a time uses it, and
// Metal's command buffers and encoders may move between threads between uses
unsafe impl Send for Gpu {}

impl Gpu {
    /// Opens the system's default Metal device and starts compiling the kernel library on
    /// a background thread; the first dispatch or [`Gpu::ready`] waits for it
    ///
    /// Only device 0 exists: Apple silicon Macs have one GPU
    pub fn new(ordinal: usize, precision: Precision) -> Result<Self> {
        ensure!(ordinal == 0, "Metal device {ordinal} requested, but Apple silicon has only device 0");
        ensure!(
            precision != Precision::Tf32,
            "tf32 is a CUDA mode (TF32 tensor cores); on Metal use fp32, whose GEMMs run near the BF16 rate"
        );
        // older systems lack Metal 4 and some selectors the kernel build calls, which would
        // raise an Objective-C exception instead of returning an error. 26.1 rather than
        // 26.0 because the MSL spec (table 7.3) first allows bfloat matmul2d operands and
        // get_left_input_cooperative_tensor there, and the one kernel library holds both
        let info = NSProcessInfo::processInfo();
        let v = info.operatingSystemVersion();
        ensure!(
            info.isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
                majorVersion: 26,
                minorVersion: 1,
                patchVersion: 0
            }),
            "the Metal backend needs macOS 26.1 or newer (Metal 4), and this is macOS {}.{}.{}",
            v.majorVersion,
            v.minorVersion,
            v.patchVersion
        );
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| eyre!("no Metal device"))?;
        // an Intel Mac can run macOS 26 but its GPU lacks the Apple-family features the
        // kernels use, so fail here rather than at kernel compile, after the weight load
        ensure!(
            device.supportsFamily(MTLGPUFamily::Apple7),
            "the Metal backend needs Apple silicon (M1 or newer), and this GPU is {}",
            device.name()
        );
        let queue = device.newCommandQueue().ok_or_else(|| eyre!("Metal could not create a command queue"))?;
        let build = {
            let device = device.clone();
            std::thread::Builder::new().name("sam3rs-metal-kernels".into()).spawn(move || build_kernels(&device))?
        };
        let profile = std::env::var_os("SAM3RS_METAL_PROFILE").map(|_| Profiler::new(&device)).transpose()?;
        let pool = Arc::new(Pool { device, state: Mutex::default(), open: std::sync::atomic::AtomicU64::new(1) });
        Ok(Self {
            queue,
            build: RefCell::new(Some(build)),
            kernels: OnceCell::new(),
            batch: RefCell::default(),
            inflight: RefCell::default(),
            completed: Cell::new(0),
            failure: RefCell::default(),
            pool,
            precision,
            profile: profile.map(RefCell::new),
        })
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Waits for the kernel build started by [`Gpu::new`], surfacing a compile error
    pub fn ready(&self) -> Result<()> {
        self.kernels().map(|_| ())
    }

    fn kernels(&self) -> Result<&Kernels> {
        if let Some(k) = self.kernels.get() {
            return Ok(k);
        }
        let build = self.build.borrow_mut().take().ok_or_else(|| eyre!("the Metal kernel build failed earlier"))?;
        let kernels = build.join().map_err(|_| eyre!("the Metal kernel build panicked"))??;
        Ok(self.kernels.get_or_init(|| kernels))
    }

    fn kernel(&self, name: &str) -> Result<&Kernel> {
        self.kernels()?.get(name).ok_or_else(|| eyre!("no Metal kernel named {name}"))
    }

    /// Encodes kernel `name` over `groups` threadgroups of `threads` threads
    pub fn dispatch(&self, name: &'static str, groups: [usize; 3], threads: [usize; 3], args: &[Arg]) -> Result<()> {
        self.encode(name, args, |enc, k| {
            ensure!(
                threads.iter().product::<usize>() <= k.pso.maxTotalThreadsPerThreadgroup(),
                "Metal kernel {name}: {threads:?} threads exceed its limit of {}",
                k.pso.maxTotalThreadsPerThreadgroup()
            );
            enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
            Ok(())
        })
    }

    /// Encodes kernel `name` over exactly `n` threads in threadgroups of `per_group`, the
    /// last one partial, for one-thread-per-element kernels
    pub fn dispatch_1d(&self, name: &'static str, n: usize, args: &[Arg]) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.encode(name, args, |enc, k| {
            let per_group = k.pso.maxTotalThreadsPerThreadgroup().min(256);
            enc.dispatchThreads_threadsPerThreadgroup(size([n, 1, 1]), size([per_group, 1, 1]));
            Ok(())
        })
    }

    fn encode(
        &self,
        name: &'static str,
        args: &[Arg],
        run: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>, &Kernel) -> Result<()>,
    ) -> Result<()> {
        let k = self.kernel(name)?;
        ensure!(
            args.len() == k.params.len(),
            "Metal kernel {name} takes {} buffer arguments, got {}",
            k.params.len(),
            args.len()
        );
        for (i, (arg, param)) in args.iter().zip(&k.params).enumerate() {
            let ok = match (arg, param) {
                (Arg::Buf(_, _, None), Param::Pointer(_)) => true,
                (Arg::Buf(_, _, Some(dt)), Param::Pointer(ty)) => dt.metal() == *ty,
                (Arg::U32(_), Param::Scalar(MTLDataType::UInt))
                | (Arg::I32(_), Param::Scalar(MTLDataType::Int))
                | (Arg::F32(_), Param::Scalar(MTLDataType::Float)) => true,
                _ => false,
            };
            if !ok {
                bail!("Metal kernel {name}: argument {i} doesn't match parameter {param:?}");
            }
        }

        let mut batch = self.batch.borrow_mut();
        if batch.is_none() {
            let cb = self.queue.commandBuffer().ok_or_else(|| eyre!("Metal could not create a command buffer"))?;
            let enc = self.encoder(&cb, name)?;
            *batch = Some(Batch { cb, enc, dispatches: 0 });
        } else if self.profile.is_some() {
            // one encoder per dispatch, so its boundary timestamps are the kernel's
            let b = batch.as_mut().expect("checked above");
            b.enc.endEncoding();
            b.enc = self.encoder(&b.cb, name)?;
        }
        let b = batch.as_mut().expect("opened above");
        b.enc.setComputePipelineState(&k.pso);
        for (i, arg) in args.iter().enumerate() {
            // SAFETY: the buffer outlives the dispatch (Metal retains it until the command
            // buffer completes) and the offset is inside it; scalars are 4 bytes copied now
            unsafe {
                match *arg {
                    Arg::Buf(buf, offset, _) => b.enc.setBuffer_offset_atIndex(Some(buf.mtl()), offset, i),
                    Arg::U32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                    Arg::I32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                    Arg::F32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                }
            }
        }
        run(&b.enc, k)?;
        b.dispatches += 1;
        if self.profile.as_ref().is_some_and(|p| p.borrow().full()) {
            drop(batch);
            return self.synchronize();
        }
        if b.dispatches >= FLUSH_EVERY {
            drop(batch);
            self.flush()?;
        }
        Ok(())
    }

    /// A compute encoder on `cb` for a dispatch of `name`, sampling its start and end
    /// timestamps when profiling
    fn encoder(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        name: &'static str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let enc = match &self.profile {
            None => cb.computeCommandEncoder(),
            Some(p) => {
                let desc = p.borrow_mut().sample(name);
                cb.computeCommandEncoderWithDescriptor(&desc)
            }
        };
        enc.ok_or_else(|| eyre!("Metal could not create an encoder"))
    }

    /// Per-kernel GPU seconds recorded since the last call, summed by kernel name and
    /// sorted by total, with dispatch counts; empty unless `SAM3RS_METAL_PROFILE` is set.
    /// `SAM3RS_METAL_PROFILE=trace` also prints every dispatch's time in order
    pub fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        let Some(profile) = &self.profile else { return Vec::new() };
        // the open dispatches' timestamps resolve once their command buffers complete
        if let Err(e) = self.synchronize() {
            eprintln!("SAM3RS_METAL_PROFILE: {e}");
        }
        let trace = std::env::var_os("SAM3RS_METAL_PROFILE").is_some_and(|v| v == "trace");
        let mut by_name: HashMap<&'static str, (usize, f64)> = HashMap::new();
        for (name, t) in profile.borrow_mut().done.drain(..) {
            if trace {
                eprintln!("  dispatch {name} {:.1} us", t * 1e6);
            }
            let e = by_name.entry(name).or_default();
            e.0 += 1;
            e.1 += t;
        }
        let mut rows: Vec<_> = by_name.into_iter().map(|(n, (c, t))| (n, c, t)).collect();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        rows
    }

    /// Commits the open command buffer without waiting for it
    pub fn flush(&self) -> Result<()> {
        let Some(b) = self.batch.borrow_mut().take() else { return Ok(()) };
        b.enc.endEncoding();
        b.cb.commit();
        let batch = self.pool.open.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inflight.borrow_mut().push_back((batch, b.cb));
        Ok(())
    }

    /// Commits pending work and waits for all of it
    pub fn synchronize(&self) -> Result<()> {
        self.flush()?;
        let mut inflight = self.inflight.borrow_mut();
        while let Some((batch, cb)) = inflight.pop_front() {
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error {
                self.failure.borrow_mut().get_or_insert_with(|| failure_reason(&cb));
            }
            self.completed.set(batch);
        }
        drop(inflight);
        if let Some(p) = &self.profile {
            p.borrow_mut().resolve(&self.pool.device)?;
        }
        match &*self.failure.borrow() {
            Some(why) => Err(eyre!("a Metal command buffer failed ({why}); load the model again")),
            None => Ok(()),
        }
    }

    /// Marks finished command buffers complete without waiting, so their buffers become
    /// writable by the host
    fn poll(&self) {
        let mut inflight = self.inflight.borrow_mut();
        while let Some((batch, cb)) = inflight.front() {
            match cb.status() {
                MTLCommandBufferStatus::Completed => {}
                // kept so `synchronize` fails instead of returning stale results
                MTLCommandBufferStatus::Error => {
                    self.failure.borrow_mut().get_or_insert_with(|| failure_reason(cb));
                }
                _ => break,
            }
            self.completed.set(*batch);
            inflight.pop_front();
        }
    }

    /// An uninitialized buffer for kernels to write
    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        self.pool.take(bytes, None)
    }

    /// An uninitialized tensor for kernels to write
    pub fn empty(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let buf = self.alloc(shape.iter().product::<usize>() * dtype.size())?;
        Ok(Tensor { buf, shape: shape.to_vec(), dtype })
    }

    /// A buffer holding `data`, written now by the host: it comes from buffers no queued
    /// kernel can still read
    pub fn upload_bytes<T: Pod>(&self, data: &[T]) -> Result<Buf> {
        self.poll();
        let bytes = std::mem::size_of_val(data);
        let buf = self.pool.take(bytes, Some(self.completed.get()))?;
        // SAFETY: the buffer is at least `bytes` long, shared storage, and no queued
        // command uses it (`take` with the completed batch)
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), buf.mtl().contents().as_ptr().cast(), bytes);
        }
        Ok(buf)
    }

    /// A tensor of `shape` holding `data` as `dtype`'s bit pattern
    pub fn upload<T: Pod>(&self, data: &[T], shape: &[usize], dtype: DType) -> Result<Tensor> {
        ensure!(
            std::mem::size_of_val(data) == shape.iter().product::<usize>() * dtype.size(),
            "upload of {} bytes into a {shape:?} {dtype:?} tensor",
            std::mem::size_of_val(data)
        );
        Ok(Tensor { buf: self.upload_bytes(data)?, shape: shape.to_vec(), dtype })
    }

    /// An FP32 tensor of `shape` holding `data`
    pub fn upload_f32(&self, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        self.upload(data, shape, DType::F32)
    }

    /// `data` rounded to BF16 on the host
    pub fn upload_bf16(&self, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let v: Vec<u16> = data.iter().map(|&x| half::bf16::from_f32(x).to_bits()).collect();
        self.upload(&v, shape, DType::Bf16)
    }

    /// FP32 `data` uploaded as `dtype`: FP32 as is, or BF16 rounded on the host
    pub fn upload_as(&self, data: &[f32], shape: &[usize], dtype: DType) -> Result<Tensor> {
        match dtype {
            DType::F32 => self.upload_f32(data, shape),
            DType::Bf16 => self.upload_bf16(data, shape),
            dt => bail!("upload of f32 data as {dt:?}"),
        }
    }

    /// Waits for the GPU, then copies `len` elements of `T` from byte `offset` of `buf`
    pub fn read<T: Pod>(&self, buf: &Buf, offset: usize, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        ensure!(offset + bytes <= buf.bytes, "read of {bytes} bytes at {offset} from a {}-byte buffer", buf.bytes);
        self.synchronize()?;
        // SAFETY: in bounds (checked above) and no kernel is running after the sync
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.mtl().contents().as_ptr().cast::<u8>().add(offset),
                out.as_mut_ptr().cast::<u8>(),
                bytes,
            );
        }
        Ok(())
    }

    /// Waits for the GPU, then copies an FP32 tensor to the host
    pub fn download(&self, t: &Tensor) -> Result<Vec<f32>> {
        let mut out = vec![0f32; t.len()];
        match t.dtype {
            DType::F32 => self.read(&t.buf, 0, &mut out)?,
            DType::Bf16 => {
                let mut raw = vec![0u16; t.len()];
                self.read(&t.buf, 0, &mut raw)?;
                for (o, r) in out.iter_mut().zip(raw) {
                    *o = if t.dtype == DType::Bf16 {
                        half::bf16::from_bits(r).to_f32()
                    } else {
                        half::f16::from_bits(r).to_f32()
                    };
                }
            }
            dt => bail!("download of a {dt:?} tensor as f32"),
        }
        Ok(out)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // queued kernels may still use buffers the pool is about to release
        let _ = self.synchronize();
        // what SAM3RS_METAL_PROFILE collected and no test took, for any program
        let rows = self.take_profile();
        if !rows.is_empty() {
            let total: f64 = rows.iter().map(|r| r.2).sum();
            eprintln!("SAM3RS_METAL_PROFILE: {:.2} ms of GPU time by kernel", total * 1e3);
            for (name, count, t) in rows {
                eprintln!("  {name:>28} {count:>7} {:>10.2} ms {:>5.1}%", t * 1e3, 100.0 * t / total);
            }
        }
    }
}

fn size(v: [usize; 3]) -> MTLSize {
    MTLSize { width: v[0], height: v[1], depth: v[2] }
}

/// Kernel parameters are named `p_*` when they are pointers, so a 4-byte pointer to a
/// single float or int isn't mistaken for a scalar by the argument check
fn is_pointer_name(name: &str) -> bool {
    name.starts_with("p_")
}

/// Threadgroups needed to cover `n` items at `per` a group
pub fn groups(n: usize, per: usize) -> usize {
    n.div_ceil(per)
}
