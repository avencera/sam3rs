//! cuda-oxide device kernels for sam3rs
//!
//! `scripts/build-ptx.sh` compiles this crate once per PTX tier (compute_80 and
//! compute_120) and writes `kernels/ptx/sam3.sm{80,120}.ptx`, which the sam3rs crate
//! embeds and loads by kernel name. Kernels take raw device pointers and `i32`, `i64` or
//! `f32` scalars, the argument layout the host launcher passes. Feature maps are
//! channels-last (`[H*W, C]` row-major), so 1x1 convs and linears are plain cuBLAS GEMMs;
//! these kernels are the glue around them

#![no_std]

pub mod attention;
pub mod common;
pub mod conv;
pub mod detector;
pub mod elementwise;
pub mod mma;
pub mod norm;
pub mod preprocess;
pub mod vision;
