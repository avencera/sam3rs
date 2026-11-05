//! Builds csrc/nvtx.c, a two-function shim over the header-only NVTX3 API, when the
//! `nvtx` feature is on. The kernels need no build step: they ship as prebuilt PTX in
//! kernels/ptx (see scripts/build-ptx.sh), so default builds need no CUDA toolkit

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "nvtx")]
    nvtx();
}

#[cfg(feature = "nvtx")]
fn nvtx() {
    use std::env::var;

    // the target's OS, not the host's, since build scripts run on the host
    let windows = var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows");
    // the Windows toolkit installer sets CUDA_PATH rather than CUDA_HOME
    let cuda = var("CUDA_HOME").or_else(|_| var("CUDA_PATH")).unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rerun-if-changed=csrc/nvtx.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    cc::Build::new().file("csrc/nvtx.c").include(std::path::Path::new(&cuda).join("include")).compile("sam3rs_nvtx");
    // NVTX3 opens the profiler's injection library with dlopen, and with LoadLibrary
    // from kernel32 on Windows, which every Windows program links already
    if !windows {
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}
