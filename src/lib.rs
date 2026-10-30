//! sam3rs: a native Rust/CUDA runtime for Meta SAM 3 image inference
//!
//! Loads the Hugging Face `model.safetensors` checkpoint and runs the image encoder,
//! text encoder, detector and SAM 2 style point decoder on one CUDA stream

pub mod blaslt;
pub mod detector;
pub mod gpu;
pub mod image;
pub mod nvtx;
pub mod ops;
pub mod pipeline;
pub mod point;
pub mod posenc;
pub mod text;
pub mod tokenizer;
pub mod vision;
pub mod weights;
