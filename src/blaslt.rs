//! cuBLASLt matmuls for the BF16 linears: BF16 operands, FP32 output, with the bias and
//! activation fused into the GEMM epilogue instead of a separate pass over the output

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use cudarc::cublaslt::sys as lt;
use cudarc::driver::result as cur;
use cudarc::driver::sys::CUstream;
use eyre::{Result, WrapErr, bail};

/// NVIDIA's recommended cuBLASLt workspace for Hopper and newer
const WORKSPACE: usize = 32 << 20;

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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    m: usize,
    n: usize,
    k: usize,
    x_ld: usize,
    epilogue: Epilogue,
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

    /// Row-major `out[m, n] = epilogue(x[m, k] @ w[n, k]^T + beta * out)` with BF16 `x`
    /// (row stride `x_ld`) and `w`, and FP32 or BF16 `out` and `bias[n]`
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        stream: CUstream,
        (m, n, k): (usize, usize, usize),
        x: u64,
        x_ld: usize,
        w: u64,
        (out, out_bf16): (u64, bool),
        beta: f32,
        bias: u64,
        epilogue: Epilogue,
    ) -> Result<()> {
        let key = Key { m, n, k, x_ld, epilogue, out_bf16 };
        let mut plans = self.plans.borrow_mut();
        let plan = match plans.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(self.plan(key)?),
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

    /// Column-major view of the row-major problem: out^T[n, m] = w[n, k] (w^T stored) @ x^T
    fn plan(&self, key: Key) -> Result<Plan> {
        let Key { m, n, k, x_ld, epilogue, out_bf16 } = key;
        unsafe {
            let mut desc = std::ptr::null_mut();
            check(lt::cublasLtMatmulDescCreate(
                &mut desc,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt::cudaDataType_t::CUDA_R_32F,
            ))?;
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
            let w = layout(lt::cudaDataType_t::CUDA_R_16BF, k, n, k)?;
            let x = layout(lt::cudaDataType_t::CUDA_R_16BF, k, m, x_ld)?;
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
            let mut result = std::mem::MaybeUninit::<lt::cublasLtMatmulHeuristicResult_t>::zeroed();
            let mut found = 0;
            let status = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                w,
                x,
                out,
                out,
                pref,
                1,
                result.as_mut_ptr(),
                &mut found,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);
            check(status)?;
            if found == 0 {
                bail!("no cuBLASLt algorithm for m={m} n={n} k={k} {epilogue:?}");
            }
            Ok(Plan { desc, w, x, out, algo: result.assume_init().algo })
        }
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
