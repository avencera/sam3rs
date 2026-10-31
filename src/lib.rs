//! sam3rs: a native Rust/CUDA runtime for Meta SAM 3 image inference
//!
//! Loads the Hugging Face `model.safetensors` checkpoint and runs the image encoder,
//! text encoder, detector and SAM 2 style point decoder on one CUDA stream. Encode an
//! image once, then answer any number of text, box and point prompts against it:
//!
//! ```no_run
//! use sam3rs::{BoxPrompt, Precision, RgbImage, Sam3, VISUAL_PROMPT};
//!
//! # fn main() -> eyre::Result<()> {
//! let model = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
//! let image = RgbImage::open("dogs.jpg".as_ref())?;
//! let embedding = model.encode(&image)?;
//! for det in model.detect(&embedding, "dog", &[])? {
//!     println!("{:.3} {:?}", det.score, det.bbox);
//! }
//! let exemplar = BoxPrompt::from_xyxy([1233.0, 376.0, 1649.0, 691.0], image.width(), image.height(), true)?;
//! let dogs = model.detect(&embedding, VISUAL_PROMPT, &[exemplar])?;
//! # Ok(())
//! # }
//! ```
//!
//! Runs on NVIDIA GPUs with compute capability 8.0 or newer and driver r580 or newer.
//! libcuda and cuBLAS 13 are loaded at run time and the kernels ship as PTX, so building
//! needs no CUDA toolkit.

#![warn(missing_docs)]

mod blaslt;
mod detector;
mod gpu;
#[cfg(feature = "online")]
mod hub;
mod image;
mod nvtx;
mod ops;
mod pipeline;
mod point;
mod posenc;
mod text;
mod tokenizer;
mod vision;
mod weights;

#[cfg(test)]
mod stage_tests;

pub use detector::BoxPrompt;
pub use gpu::Precision;
#[cfg(feature = "online")]
pub use hub::{MODEL_FILES, MODEL_REPO, download_model};
pub use image::{MAX_IMAGE_SIDE, RgbImage};
pub use pipeline::{Detection, Embedding, PointMask, SCORE_THRESHOLD, Sam3, VISUAL_PROMPT};
pub use point::PointPrompt;
