//! cuBLASLt matmuls for the linears of the BF16 precision mode: BF16 operands with FP32
//! or BF16 output, or FP32 operands and output, with the bias and activation fused into
//! the GEMM epilogue instead of a separate pass over the output

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use cudarc::cublaslt::sys as lt;
use cudarc::driver::result as cur;
use cudarc::driver::sys::CUstream;
use eyre::{Result, WrapErr, bail};

/// NVIDIA's recommended cuBLASLt workspace for Hopper and newer
const WORKSPACE: usize = 32 << 20;

/// TF32 mode times up to this many of the heuristic's candidates for each new shape
const TUNE_CANDIDATES: usize = 8;

/// A tuned candidate replaces the heuristic's first choice only when it is this much
/// faster, so most close calls keep the first choice. A candidate near the margin can still
/// win in one process and lose in the next, so TF32 outputs can differ slightly between runs
/// (on an RTX 5090, one binary gave two outputs 0.00001 apart in score)
const TUNE_MARGIN: f32 = 1.05;

/// What the epilogue applies after `x @ w^T + beta * out`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epilogue {
    None,
    Bias,
    Relu,
    ReluBias,
    /// cuBLASLt's GELU is the tanh approximation
    Gelu,
    GeluBias,
}

impl Epilogue {
    fn raw(self) -> lt::cublasLtEpilogue_t {
        use lt::cublasLtEpilogue_t::{
            CUBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_DEFAULT, CUBLASLT_EPILOGUE_GELU, CUBLASLT_EPILOGUE_GELU_BIAS,
            CUBLASLT_EPILOGUE_RELU, CUBLASLT_EPILOGUE_RELU_BIAS,
        };
        match self {
            Self::None => CUBLASLT_EPILOGUE_DEFAULT,
            Self::Bias => CUBLASLT_EPILOGUE_BIAS,
            Self::Relu => CUBLASLT_EPILOGUE_RELU,
            Self::ReluBias => CUBLASLT_EPILOGUE_RELU_BIAS,
            Self::Gelu => CUBLASLT_EPILOGUE_GELU,
            Self::GeluBias => CUBLASLT_EPILOGUE_GELU_BIAS,
        }
    }

    fn has_bias(self) -> bool {
        matches!(self, Self::Bias | Self::ReluBias | Self::GeluBias)
    }
}

/// Operand types and the compute type of a matmul
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operands {
    /// BF16 x and w, FP32 accumulation
    Bf16,
    /// FP32 x, w and output with plain FP32 compute
    F32,
    /// FP32 x, w and output on TF32 tensor cores
    F32FastTf32,
    /// FP32 x, w and output on tensor cores that may round the inputs to BF16
    F32Fast16Bf,
}

impl Operands {
    fn compute(self) -> lt::cublasComputeType_t {
        use lt::cublasComputeType_t::{CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_COMPUTE_32F_FAST_TF32};
        match self {
            Self::Bf16 | Self::F32 => CUBLAS_COMPUTE_32F,
            Self::F32FastTf32 => CUBLAS_COMPUTE_32F_FAST_TF32,
            Self::F32Fast16Bf => CUBLAS_COMPUTE_32F_FAST_16BF,
        }
    }

    fn input(self) -> lt::cudaDataType_t {
        match self {
            Self::Bf16 => lt::cudaDataType_t::CUDA_R_16BF,
            Self::F32 | Self::F32FastTf32 | Self::F32Fast16Bf => lt::cudaDataType_t::CUDA_R_32F,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    m: usize,
    n: usize,
    k: usize,
    x_ld: usize,
    epilogue: Epilogue,
    operands: Operands,
    out_bf16: bool,
}

/// Descriptors and the heuristic's algorithm for one shape
struct Plan {
    desc: lt::cublasLtMatmulDesc_t,
    w: lt::cublasLtMatrixLayout_t,
    x: lt::cublasLtMatrixLayout_t,
    out: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
}

pub struct BlasLt {
    handle: lt::cublasLtHandle_t,
    workspace: u64,
    plans: RefCell<HashMap<Key, Plan>>,
}

fn check(status: lt::cublasStatus_t) -> Result<()> {
    if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { bail!("cuBLASLt error {status:?}") }
}

impl BlasLt {
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe { check(lt::cublasLtCreate(&mut handle))? };
        let workspace = unsafe { cur::malloc_sync(WORKSPACE) }.wrap_err("cuBLASLt workspace")?;
        Ok(Self { handle, workspace, plans: RefCell::new(HashMap::new()) })
    }

    /// Row-major `out[m, n] = epilogue(x[m, k] @ w[n, k]^T + beta * out)` with `x` (row
    /// stride `x_ld`) and `w` of the operand type, and `out` and `bias[n]` FP32, or BF16
    /// for BF16 operands
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        stream: CUstream,
        operands: Operands,
        (m, n, k): (usize, usize, usize),
        x: u64,
        x_ld: usize,
        w: u64,
        (out, out_bf16): (u64, bool),
        beta: f32,
        bias: u64,
        epilogue: Epilogue,
    ) -> Result<()> {
        if out_bf16 && operands != Operands::Bf16 {
            bail!("BF16 output needs BF16 operands");
        }
        let key = Key { m, n, k, x_ld, epilogue, operands, out_bf16 };
        let mut plans = self.plans.borrow_mut();
        let plan = match plans.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let (mut plan, candidates) = self.plan(key)?;
                if candidates.len() > 1 {
                    plan.algo = self.tune(&plan, &candidates, key, stream, x, w, beta, bias)?;
                }
                e.insert(plan)
            }
        };
        let set = |attr, value: *const c_void, size| unsafe {
            check(lt::cublasLtMatmulDescSetAttribute(plan.desc, attr, value, size))
        };
        if epilogue.has_bias() {
            set(
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                (&bias as *const u64).cast(),
                size_of::<u64>(),
            )?;
        }
        let alpha = 1f32;
        unsafe {
            check(lt::cublasLtMatmul(
                self.handle,
                plan.desc,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                plan.w,
                x as *const c_void,
                plan.x,
                (&beta as *const f32).cast(),
                out as *const c_void,
                plan.out,
                out as *mut c_void,
                plan.out,
                &plan.algo,
                self.workspace as *mut c_void,
                WORKSPACE,
                stream as _,
            ))
        }
        .wrap_err_with(|| format!("cuBLASLt linear m={m} n={n} k={k}"))
    }

    /// Column-major view of the row-major problem: out^T[n, m] = w[n, k] (w^T stored) @ x^T.
    /// Returns the plan with the heuristic's first choice, and for TF32 operands its other
    /// candidates too
    fn plan(&self, key: Key) -> Result<(Plan, Vec<lt::cublasLtMatmulAlgo_t>)> {
        let Key { m, n, k, x_ld, epilogue, operands, out_bf16 } = key;
        unsafe {
            let mut desc = std::ptr::null_mut();
            check(lt::cublasLtMatmulDescCreate(&mut desc, operands.compute(), lt::cudaDataType_t::CUDA_R_32F))?;
            let set =
                |attr, value: *const c_void, size| check(lt::cublasLtMatmulDescSetAttribute(desc, attr, value, size));
            use lt::cublasLtMatmulDescAttributes_t::{
                CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_TRANSA,
                CUBLASLT_MATMUL_DESC_TRANSB,
            };
            let (trans, no_trans) = (1i32, 0i32); // CUBLAS_OP_T, CUBLAS_OP_N
            set(CUBLASLT_MATMUL_DESC_TRANSA, (&trans as *const i32).cast(), 4)?;
            set(CUBLASLT_MATMUL_DESC_TRANSB, (&no_trans as *const i32).cast(), 4)?;
            let epi = epilogue.raw();
            set(CUBLASLT_MATMUL_DESC_EPILOGUE, (&epi as *const lt::cublasLtEpilogue_t).cast(), size_of_val(&epi))?;
            // the bias has the output's type
            let out_ty = if out_bf16 { lt::cudaDataType_t::CUDA_R_16BF } else { lt::cudaDataType_t::CUDA_R_32F };
            if epilogue.has_bias() {
                set(
                    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    (&out_ty as *const lt::cudaDataType_t).cast(),
                    size_of_val(&out_ty),
                )?;
            }

            let layout = |ty, rows: usize, cols: usize, ld: usize| -> Result<lt::cublasLtMatrixLayout_t> {
                let mut l = std::ptr::null_mut();
                check(lt::cublasLtMatrixLayoutCreate(&mut l, ty, rows as u64, cols as u64, ld as i64))?;
                Ok(l)
            };
            let w = layout(operands.input(), k, n, k)?;
            let x = layout(operands.input(), k, m, x_ld)?;
            let out = layout(out_ty, n, m, n)?;

            let mut pref = std::ptr::null_mut();
            check(lt::cublasLtMatmulPreferenceCreate(&mut pref))?;
            let ws = WORKSPACE as u64;
            check(lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws as *const u64).cast(),
                8,
            ))?;
            let count = if operands == Operands::F32FastTf32 { TUNE_CANDIDATES } else { 1 };
            let mut results = vec![std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>(); count];
            let mut found = 0;
            let status = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                w,
                x,
                out,
                out,
                pref,
                count as i32,
                results.as_mut_ptr(),
                &mut found,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);
            check(status)?;
            if found == 0 {
                bail!("no cuBLASLt algorithm for m={m} n={n} k={k} {epilogue:?}");
            }
            let candidates: Vec<_> = results[..found as usize].iter().map(|r| r.algo).collect();
            Ok((Plan { desc, w, x, out, algo: candidates[0] }, candidates))
        }
    }

    /// Times each candidate on this call's operands, writing to a scratch output so a
    /// residual output stays untouched, and returns the one to keep. On an RTX 5090 the
    /// heuristic's first choice ran the backbone's GELU linear 10% and the 3x3 conv taps
    /// 16% slower than another candidate, and small-row GEMMs up to 1.5x slower than a
    /// split-K one
    #[allow(clippy::too_many_arguments)]
    fn tune(
        &self,
        plan: &Plan,
        candidates: &[lt::cublasLtMatmulAlgo_t],
        key: Key,
        stream: CUstream,
        x: u64,
        w: u64,
        beta: f32,
        bias: u64,
    ) -> Result<lt::cublasLtMatmulAlgo_t> {
        if key.epilogue.has_bias() {
            unsafe {
                check(lt::cublasLtMatmulDescSetAttribute(
                    plan.desc,
                    lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&bias as *const u64).cast(),
                    size_of::<u64>(),
                ))?;
            }
        }
        let bytes = key.m * key.n * if key.out_bf16 { 2 } else { 4 };
        let out = unsafe { cur::malloc_sync(bytes) }.wrap_err("cuBLASLt tuning output")?;
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let (start, end) = (cur::event::create(flags)?, cur::event::create(flags)?);
        let alpha = 1f32;
        let run = |algo: &lt::cublasLtMatmulAlgo_t| unsafe {
            check(lt::cublasLtMatmul(
                self.handle,
                plan.desc,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                plan.w,
                x as *const c_void,
                plan.x,
                (&beta as *const f32).cast(),
                out as *const c_void,
                plan.out,
                out as *mut c_void,
                plan.out,
                algo,
                self.workspace as *mut c_void,
                WORKSPACE,
                stream as _,
            ))
        };
        // milliseconds per call over `reps` calls
        let time = |algo: &lt::cublasLtMatmulAlgo_t, reps: usize| -> Result<f32> {
            unsafe {
                cur::event::record(start, stream)?;
                for _ in 0..reps {
                    run(algo)?;
                }
                cur::event::record(end, stream)?;
                cur::event::synchronize(end)?;
                Ok(cur::event::elapsed(start, end)? / reps as f32)
            }
        };
        let mut times = Vec::with_capacity(candidates.len());
        for algo in candidates {
            // a candidate that fails to launch never wins
            let Ok(first) = time(algo, 1) else {
                times.push(f32::INFINITY);
                continue;
            };
            // ~0.25 ms windows, best of 2: every new shape pays this on its first call
            let reps = ((0.25 / first.max(1e-3)) as usize).clamp(2, 50);
            let mut best = f32::INFINITY;
            for _ in 0..2 {
                best = best.min(time(algo, reps)?);
            }
            times.push(best);
        }
        unsafe {
            let _ = cur::event::destroy(start);
            let _ = cur::event::destroy(end);
            cur::free_sync(out)?;
        }
        let (fastest, t) =
            times.iter().enumerate().fold((0, f32::INFINITY), |a, (i, &t)| if t < a.1 { (i, t) } else { a });
        Ok(if times[0] > t * TUNE_MARGIN { candidates[fastest] } else { candidates[0] })
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        unsafe {
            for plan in self.plans.get_mut().values() {
                lt::cublasLtMatrixLayoutDestroy(plan.w);
                lt::cublasLtMatrixLayoutDestroy(plan.x);
                lt::cublasLtMatrixLayoutDestroy(plan.out);
                lt::cublasLtMatmulDescDestroy(plan.desc);
            }
            let _ = cur::free_sync(self.workspace);
            lt::cublasLtDestroy(self.handle);
        }
    }
}
