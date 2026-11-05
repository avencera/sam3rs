//! cuBLAS and cuBLASLt entry points, opened by their CUDA 13 sonames (DLL names on
//! Windows)
//!
//! cudarc's loader tries the unversioned `libcublas.so` before `libcublas.so.13`. That
//! name is a toolkit's dev symlink, so on a machine with a CUDA toolkit it opens the
//! toolkit's build even when another cuBLAS 13 (the pip wheel the Python package
//! preloads, or one on `LD_LIBRARY_PATH`) is already loaded. The process then holds two
//! cuBLAS builds, and one rejects the other's handles with errors like
//! `CUBLAS_STATUS_INVALID_VALUE`. Opening the soname reuses a copy already loaded and
//! otherwise follows the dynamic loader's search order. The types still come from
//! cudarc's bindings, which match the 13.0 headers
//!
//! The toolkit's libcublas also binds its libcublasLt by soname, so it ran on the
//! wheel's copy, and a libcublas newer than its libcublasLt (the 13.4.1 toolkit's on the
//! 13.1.1 wheel's) fails its first GEMM with `CUBLAS_STATUS_NOT_INITIALIZED`. A version
//! check can't catch that pairing: libcublas answers `cublasGetProperty` and
//! `cublasGetVersion_v2` with its libcublasLt's release
//!
//! Windows behaves the same way: loading a DLL by name reuses a loaded module with that
//! name, wherever it came from, and `cublas64_13.dll` imports `cublasLt64_13.dll` by
//! name, so opening Lt first pins both to one install there too. Windows' default search
//! also looks in the current directory before `PATH`, where a DLL planted in a folder
//! sam3rs runs from would load and run its code, so [`open_library`] searches itself
//! and never looks in the current directory unless it is also the executable's directory
//! or an absolute `PATH` entry

use std::ffi::c_int;
use std::sync::OnceLock;

use cudarc::cublas::sys as blas;
use cudarc::cublaslt::sys as lt;
use eyre::{Result, WrapErr};
use libloading::Library;

#[cfg(not(windows))]
pub(crate) const BLAS_SONAME: &str = "libcublas.so.13";
// libcublas links against libcublasLt by soname, so this one opens first and libcublas
// then binds to the same copy
#[cfg(not(windows))]
pub(crate) const LT_SONAME: &str = "libcublasLt.so.13";
#[cfg(windows)]
pub(crate) const BLAS_SONAME: &str = "cublas64_13.dll";
#[cfg(windows)]
pub(crate) const LT_SONAME: &str = "cublasLt64_13.dll";

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
        let lt_lib = open_library(LT_SONAME)?;
        let blas_lib = open_library(BLAS_SONAME)?;
        // each library is the one its table is declared for
        let lt = unsafe { Lt::resolve(&lt_lib) }?;
        let blas = unsafe { Blas::resolve(&blas_lib) }?;
        Ok(Self { blas, lt, _libs: [blas_lib, lt_lib] })
    }
}

/// Opens a library by soname, through the dynamic loader's own search
#[cfg(not(windows))]
fn open_library(soname: &str) -> Result<Library> {
    // SAFETY: cuBLAS's initialisers have no preconditions on the caller
    unsafe { Library::new(soname) }.wrap_err_with(|| format!("open {soname}"))
}

/// Opens a DLL from wherever Windows already has it or from `PATH`, and never from the
/// current directory unless that is also the executable's directory or an absolute `PATH`
/// entry
///
/// The bare name with `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` returns a module already loaded
/// under that name, such as the Python wheel's copy, and otherwise searches only the
/// executable's directory, System32 and directories the host added with
/// `AddDllDirectory`. Failing that, each absolute `PATH` entry is tried in order, which
/// is how a CUDA toolkit is found. Relative and empty entries are skipped, since they
/// resolve against the current directory
#[cfg(windows)]
fn open_library(name: &str) -> Result<Library> {
    use std::env;

    use eyre::eyre;
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library as WinLibrary,
    };

    // SAFETY (both loads): cuBLAS's initialisers have no preconditions on the caller
    let first = unsafe { WinLibrary::load_with_flags(name, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) };
    let mut last_error = match first {
        Ok(lib) => return Ok(lib.into()),
        Err(e) => eyre!(e),
    };

    let path = env::var_os("PATH").unwrap_or_default();
    for dir in env::split_paths(&path).filter(|dir| dir.is_absolute()) {
        let candidate = dir.join(name);
        if !candidate.is_file() {
            continue;
        }

        // the DLL's own directory resolves any import it has beside it, again without
        // the current directory
        let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        match unsafe { WinLibrary::load_with_flags(&candidate, flags) } {
            Ok(lib) => return Ok(lib.into()),
            // a broken copy, such as one for another architecture, leaves later entries to try
            Err(e) => last_error = eyre!(e).wrap_err(format!("open {}", candidate.display())),
        }
    }

    Err(last_error.wrap_err(format!("open {name} from the executable's directory, System32 or an absolute PATH entry")))
}

#[cfg(all(test, windows))]
mod tests {
    use std::env;
    use std::ffi::OsString;
    use std::path::PathBuf;

    use super::open_library;

    /// Puts the working directory and PATH back even when an assertion fails
    struct Restore(PathBuf, OsString);

    impl Drop for Restore {
        fn drop(&mut self) {
            env::set_current_dir(&self.0).unwrap();
            // SAFETY: this is the only test that touches PATH, and nothing else in the
            // test process reads it
            unsafe { env::set_var("PATH", &self.1) };
        }
    }

    /// A DLL in the current directory or behind a relative PATH entry never loads, and one
    /// in an absolute PATH entry does
    #[test]
    fn dll_search_skips_the_current_directory() {
        // a copy of a system DLL under a name no Windows directory has
        let name = format!("sam3rs_search_probe_{}.dll", std::process::id());
        let dir = tempfile::tempdir().unwrap();
        let system = PathBuf::from(env::var_os("SystemRoot").unwrap()).join("System32");
        std::fs::copy(system.join("version.dll"), dir.path().join(&name)).unwrap();
        let restore = Restore(env::current_dir().unwrap(), env::var_os("PATH").unwrap_or_default());

        env::set_current_dir(dir.path()).unwrap();
        assert!(open_library(&name).is_err(), "loaded from the current directory");

        // the relative entry names the probe's directory from its parent
        env::set_current_dir(dir.path().parent().unwrap()).unwrap();
        // SAFETY (both set_var calls): as in Restore::drop
        unsafe { env::set_var("PATH", dir.path().file_name().unwrap()) };
        assert!(open_library(&name).is_err(), "loaded through a relative PATH entry");
        drop(restore);

        let restore = Restore(env::current_dir().unwrap(), env::var_os("PATH").unwrap_or_default());
        unsafe { env::set_var("PATH", dir.path()) };
        open_library(&name).expect("an absolute PATH entry");
        drop(restore);
    }
}
