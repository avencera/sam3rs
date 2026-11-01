//! cuBLAS and cuBLASLt entry points, opened by their CUDA 13 sonames
//!
//! cudarc's loader tries the unversioned `libcublas.so` before `libcublas.so.13`. That
//! name is a toolkit's dev symlink, so on a machine with a CUDA toolkit it opens the
//! toolkit's build even when another cuBLAS 13 (the pip wheel the Python package
//! preloads, or one on `LD_LIBRARY_PATH`) is already loaded. The process then holds two
//! cuBLAS builds, and one rejects the other's handles with errors like
//! `CUBLAS_STATUS_INVALID_VALUE`. Opening the soname reuses a copy already loaded and
//! otherwise follows the dynamic loader's search order. The types still come from
//! cudarc's bindings, which match the 13.0 headers

use std::ffi::c_int;
use std::sync::OnceLock;

use cudarc::cublas::sys as blas;
use cudarc::cublaslt::sys as lt;
use eyre::{Result, WrapErr};
use libloading::Library;

const BLAS_SONAME: &str = "libcublas.so.13";
// libcublas links against libcublasLt by soname, so this one opens first and libcublas
// then binds to the same copy
const LT_SONAME: &str = "libcublasLt.so.13";

/// Declares a table of C entry points and the code that resolves them from a library
macro_rules! entry_points {
    ($table:ident -> $status:ty { $($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)* }) => {
        pub(crate) struct $table {
            $(pub(crate) $field: unsafe extern "C" fn($($arg),*) -> $status,)*
        }

        impl $table {
            /// Resolves every entry point, naming the first one the library lacks
            ///
            /// # Safety
            /// `lib` must be the library the table is declared for, so each symbol has the
            /// signature given here
            unsafe fn resolve(lib: &Library) -> Result<Self> {
                Ok(Self {
                    $($field: *unsafe { lib.get(concat!($symbol, "\0").as_bytes()) }
                        .wrap_err(concat!("missing ", $symbol))?,)*
                })
            }
        }
    };
}

entry_points!(Blas -> blas::cublasStatus_t {
    create = "cublasCreate_v2": fn(*mut blas::cublasHandle_t);
    destroy = "cublasDestroy_v2": fn(blas::cublasHandle_t);
    set_stream = "cublasSetStream_v2": fn(blas::cublasHandle_t, blas::cudaStream_t);
    set_math_mode = "cublasSetMathMode": fn(blas::cublasHandle_t, blas::cublasMath_t);
    gemm_strided_batched_ex = "cublasGemmStridedBatchedEx": fn(
        blas::cublasHandle_t,
        blas::cublasOperation_t,
        blas::cublasOperation_t,
        c_int,
        c_int,
        c_int,
        *const std::ffi::c_void,
        *const std::ffi::c_void,
        blas::cudaDataType,
        c_int,
        i64,
        *const std::ffi::c_void,
        blas::cudaDataType,
        c_int,
        i64,
        *const std::ffi::c_void,
        *mut std::ffi::c_void,
        blas::cudaDataType,
        c_int,
        i64,
        c_int,
        blas::cublasComputeType_t,
        blas::cublasGemmAlgo_t,
    );
});

entry_points!(Lt -> lt::cublasStatus_t {
    create = "cublasLtCreate": fn(*mut lt::cublasLtHandle_t);
    destroy = "cublasLtDestroy": fn(lt::cublasLtHandle_t);
    matmul_desc_create = "cublasLtMatmulDescCreate":
        fn(*mut lt::cublasLtMatmulDesc_t, lt::cublasComputeType_t, lt::cudaDataType_t);
    matmul_desc_destroy = "cublasLtMatmulDescDestroy": fn(lt::cublasLtMatmulDesc_t);
    matmul_desc_set_attribute = "cublasLtMatmulDescSetAttribute": fn(
        lt::cublasLtMatmulDesc_t,
        lt::cublasLtMatmulDescAttributes_t,
        *const std::ffi::c_void,
        usize,
    );
    matrix_layout_create = "cublasLtMatrixLayoutCreate":
        fn(*mut lt::cublasLtMatrixLayout_t, lt::cudaDataType, u64, u64, i64);
    matrix_layout_destroy = "cublasLtMatrixLayoutDestroy": fn(lt::cublasLtMatrixLayout_t);
    preference_create = "cublasLtMatmulPreferenceCreate": fn(*mut lt::cublasLtMatmulPreference_t);
    preference_destroy = "cublasLtMatmulPreferenceDestroy": fn(lt::cublasLtMatmulPreference_t);
    preference_set_attribute = "cublasLtMatmulPreferenceSetAttribute": fn(
        lt::cublasLtMatmulPreference_t,
        lt::cublasLtMatmulPreferenceAttributes_t,
        *const std::ffi::c_void,
        usize,
    );
    algo_get_heuristic = "cublasLtMatmulAlgoGetHeuristic": fn(
        lt::cublasLtHandle_t,
        lt::cublasLtMatmulDesc_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatrixLayout_t,
        lt::cublasLtMatmulPreference_t,
        c_int,
        *mut lt::cublasLtMatmulHeuristicResult_t,
        *mut c_int,
    );
    matmul = "cublasLtMatmul": fn(
        lt::cublasLtHandle_t,
        lt::cublasLtMatmulDesc_t,
        *const std::ffi::c_void,
        *const std::ffi::c_void,
        lt::cublasLtMatrixLayout_t,
        *const std::ffi::c_void,
        lt::cublasLtMatrixLayout_t,
        *const std::ffi::c_void,
        *const std::ffi::c_void,
        lt::cublasLtMatrixLayout_t,
        *mut std::ffi::c_void,
        lt::cublasLtMatrixLayout_t,
        *const lt::cublasLtMatmulAlgo_t,
        *mut std::ffi::c_void,
        usize,
        lt::cudaStream_t,
    );
});

/// cuBLAS and cuBLASLt from one install, loaded for the life of the process
pub(crate) struct Cublas {
    pub(crate) blas: Blas,
    pub(crate) lt: Lt,
    // the entry points point into these, so they are never closed
    _libs: [Library; 2],
}

/// Only a successful load is kept, so a process that installs cuBLAS after a failed
/// load can try again
static CUBLAS: OnceLock<Cublas> = OnceLock::new();

impl Cublas {
    /// The process-wide cuBLAS, opened on first use
    pub(crate) fn get() -> Result<&'static Self> {
        if let Some(cublas) = CUBLAS.get() {
            return Ok(cublas);
        }

        // two threads racing here both open the same copies; the loser's handles only
        // drop a reference count
        let cublas = Self::open()?;
        Ok(CUBLAS.get_or_init(|| cublas))
    }

    fn open() -> Result<Self> {
        let open = |soname: &str| unsafe { Library::new(soname) }.wrap_err_with(|| format!("open {soname}"));
        let lt_lib = open(LT_SONAME)?;
        let blas_lib = open(BLAS_SONAME)?;
        // each library is the one its table is declared for
        let lt = unsafe { Lt::resolve(&lt_lib) }?;
        let blas = unsafe { Blas::resolve(&blas_lib) }?;
        Ok(Self { blas, lt, _libs: [blas_lib, lt_lib] })
    }
}
