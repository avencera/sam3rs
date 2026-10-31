//! cuda-oxide device kernels for sam3rs
//!
//! `scripts/build-ptx.sh` compiles this crate once per PTX tier (compute_80 and
//! compute_120) and writes `kernels/ptx/sam3.sm{80,120}.ptx`, which the sam3rs crate
//! embeds and loads by kernel name. Buffers are `&[T]` inputs and `DisjointSlice<T>`
//! outputs, each two PTX params (address, then length in elements), and sizes the slices
//! can't give are `u32` or `i32`, with `f32` for other scalars; the host checks every
//! launch against the PTX param list. Feature maps are
//! channels-last (`[H*W, C]` row-major), so 1x1 convs and linears are plain cuBLAS GEMMs;
//! these kernels are the glue around them

#![no_std]

pub mod attention;
pub mod attention_tf32;
pub mod common;
pub mod conv;
pub mod detector;
pub mod elementwise;
pub mod mma;
pub mod norm;
pub mod preprocess;
pub mod vision;
