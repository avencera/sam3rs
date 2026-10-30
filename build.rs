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
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rerun-if-changed=csrc/nvtx.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    cc::Build::new().file("csrc/nvtx.c").include(std::path::Path::new(&cuda).join("include")).compile("sam3rs_nvtx");
    println!("cargo:rustc-link-lib=dylib=dl");
}
