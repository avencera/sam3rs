//! Compiles kernels/*.cu into one fatbin with nvcc
//!
//! speakrs writes its kernels with cuda-oxide, which needs nightly Rust. sam3rs stays on
//! stable, so its kernels are CUDA C++ built here. The fatbin carries an sm_120 cubin for
//! the RTX 50 series plus compute_120 PTX that newer drivers can JIT
//!
//! It also builds csrc/nvtx.c, a two-function shim over the header-only NVTX3 API, into
//! a static library so profiles can be split by pipeline stage

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = PathBuf::from(&cuda).join("bin/nvcc");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("sam3rs.fatbin");
    let arch = env::var("SAM3RS_CUDA_ARCH").unwrap_or_else(|_| "120".into());

    let sources: Vec<PathBuf> = std::fs::read_dir("kernels")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "cu" || e == "cuh"))
        .collect();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    println!("cargo:rerun-if-env-changed=SAM3RS_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let status = Command::new(&nvcc)
        .args(["-fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr"])
        .arg(format!("-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]"))
        .arg("-o")
        .arg(&out)
        .arg("kernels/sam3.cu")
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let object = out_dir.join("nvtx.o");
    println!("cargo:rerun-if-changed=csrc/nvtx.c");
    let status = Command::new("cc")
        .args(["-c", "-O2", "-fPIC", "-I"])
        .arg(PathBuf::from(&cuda).join("include"))
        .arg("-o")
        .arg(&object)
        .arg("csrc/nvtx.c")
        .status()
        .expect("failed to run cc");
    assert!(status.success(), "cc failed on csrc/nvtx.c");
    let status = Command::new("ar")
        .arg("rcs")
        .arg(out_dir.join("libsam3rs_nvtx.a"))
        .arg(&object)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sam3rs_nvtx");
    println!("cargo:rustc-link-lib=dylib=dl");
}
