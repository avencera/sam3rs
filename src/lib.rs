//! SAM 3 on NVIDIA GPUs, fast enough to run live. sam3rs is a native Rust/CUDA runtime for
//! [Meta SAM 3](https://github.com/facebookresearch/sam3): it loads the official Hugging
//! Face checkpoint and runs image segmentation and click-to-track video on one CUDA
//! stream, with no PyTorch or ONNX Runtime.
//!
//! On an RTX 5090 in BF16, against the official PyTorch implementation:
//!
//! | Task | sam3rs | PyTorch | Speedup |
//! |---|---|---|---|
//! | Click to mask, 2560x1600 image | 1.5 ms | 36.2 ms | **24x** |
//! | Click to mask, 50 MP photo | 5.4 ms | 461 ms | **85x** |
//! | Segment everything (16x16 clicks), 50 MP photo | 1.46 s | 105 s | **72x** |
//! | Box prompt, 2560x1600 | 5.4 ms | 21.8 ms | **4.0x** |
//! | Text prompt, 2560x1600 | 6.9 ms | 20.7 ms | **3.0x** |
//! | Encode a new image | 30.9 ms | 39.4 ms | **1.28x** |
//!
//! Fast enough that a mask can follow the cursor. In frames a second on the same card:
//!
//! | Live workload | sam3rs | PyTorch |
//! |---|---|---|
//! | Mask under a moving cursor, 50 MP photo | **120** | 2 |
//! | Mask under a moving cursor, 12 MP photo | **120** | 10 |
//! | 8 masks around the cursor on every move, 36 MP photo | **28** | 0.4 |
//! | Text prompt on every new 2560x1600 frame | **26** | 17 |
//! | Track one object, 720p video | **29** | 22 |
//! | Track three objects, 720p video | **23** | 17 |
//!
//! The cursor moves at 120 Hz, so 120 means sam3rs keeps up with every move. At 720p the
//! tracker also beats `torch.compile`, CUDA graphs and ComfyUI, in less GPU memory than
//! any of them.
//!
//! The [Python package](https://github.com/avencera/sam3rs/tree/master/crates/sam3rs-python)
//! gives the same speed with numpy in and out.
//!
//! # Install
//!
//! The `sam3rs` command for Linux x86_64 (glibc 2.28 or newer), from the latest GitHub
//! release into `~/.local/bin`:
//!
//! ```sh
//! curl -LSfs https://github.com/avencera/sam3rs/releases/latest/download/install.sh | sh
//! ```
//!
//! `--to DIR` installs somewhere else and `--tag vX.Y.Z` picks a release. The binary
//! needs cuBLAS 13 on the library path, which usually means a CUDA 13 toolkit. Without
//! one, the Python package's `sam3rs` command takes the same arguments and brings its
//! own cuBLAS (about 470 MB, downloaded on the first run):
//!
//! ```sh
//! uvx --from "sam3rs[cuda]" sam3rs dogs.jpg --text dog --out dogs.png
//! ```
//!
//! `pip install "sam3rs[cuda]"` installs the same command and the Python API. Building
//! from source needs no CUDA toolkit, though running the result still needs cuBLAS:
//!
//! ```sh
//! cargo install sam3rs --features cli
//! ```
//!
//! As a library, `cargo add sam3rs`.
//!
//! # What it runs
//!
//! - Perception Encoder ViT backbone (32 blocks, windowed and global attention with RoPE)
//!   and both SimpleFPN necks
//! - CLIP text encoder with the BPE tokenizer
//! - geometry (box) encoder, fusion encoder, DETR decoder with the presence token,
//!   dot-product scoring and the segmentation head
//! - SAM 2 style point decoder with the two-pass mask refinement and hole filling
//! - the SAM 3 video tracker: memory attention over a bank of past frames, the memory
//!   encoder, object pointers and upstream's memory selection
//!
//! Image features stay on the GPU between prompts, which is where much of the speed comes
//! from. Kernels that cuBLAS doesn't cover are Rust, written with
//! [cuda-oxide](https://github.com/NVIDIA/cuda-rust) in
//! [crates/sam3rs-kernels](https://github.com/avencera/sam3rs/tree/master/crates/sam3rs-kernels).
//! They ship as prebuilt PTX in `kernels/ptx`, so building sam3rs needs neither a CUDA
//! toolkit nor cuda-oxide. `scripts/build-ptx.sh` regenerates the PTX after a kernel
//! change.
//!
//! # Precision modes
//!
//! - `bf16` (default): BF16 tensor-core GEMMs with FP32 accumulation and a fused
//!   flash-attention kernel.
//! - `tf32`: FP32 weights and activations with TF32 tensor-core GEMMs, the way ONNX
//!   Runtime's CUDA provider runs FP32 models by default. GEMM autotuning can make
//!   scores differ by about 0.00001 between runs.
//! - `fp32`: strict FP32, with no TF32 or BF16 anywhere. The reference path for parity
//!   checks.
//!
//! [`Precision`](https://docs.rs/sam3rs/latest/sam3rs/enum.Precision.html) lists which
//! stages keep which types in each mode.
//!
//! # Requirements
//!
//! - An NVIDIA GPU with compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell)
//!   and driver r580 or newer. RTX 50 series GPUs load compute_120 PTX, others the
//!   compute_80 floor. Benchmarks and parity runs use an RTX 5090.
//! - Any cuBLAS 13 release (`libcublas.so.13`, `libcublasLt.so.13`), from a CUDA 13
//!   toolkit or the `nvidia-cublas` pip wheel. All three precision modes are tested with
//!   wheels from 13.0.0 to 13.8.2.
//! - Stable Rust (edition 2024), to build from source
//!
//! libcuda and cuBLAS load at run time, so a missing driver, GPU or cuBLAS makes
//! `Sam3::load` return an error instead of panicking. A cuBLAS already loaded in the
//! process, such as the pip wheel the Python package loads, is reused; otherwise the
//! dynamic loader finds one (`LD_LIBRARY_PATH`, then the ldconfig cache).
//!
//! Cargo features:
//!
//! - `online` (default): `Sam3::from_pretrained` and `download_model`, which fetch the
//!   weights from Hugging Face through [`hf-fetch`](https://github.com/avencera/hf-fetch)
//!   into the shared Python Hugging Face cache
//! - `cli`: the `sam3rs` command
//! - `nvtx`: NVTX ranges per pipeline stage for Nsight Systems. The only feature that
//!   needs a CUDA toolkit to build, for its headers (`CUDA_HOME`, default
//!   `/usr/local/cuda`).
//!
//! # Weights
//!
//! SAM 3 weights are under Meta's SAM License and aren't part of this repository or its
//! packages. [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face is
//! gated: accept Meta's terms on the model page while logged in, then run
//! `hf auth login` or set `HF_TOKEN` to a token of that account.
//! `Sam3::from_pretrained` and the CLI without `--model` download the checkpoint into
//! the Hugging Face cache on first use.
//!
//! For the tests and benchmarks, `scripts/fetch-models.sh` downloads the checkpoint into
//! `models/sam3` with the same token, fetches the test images into `models/images`, and
//! checks the weights against the official sha256 hashes:
//!
//! ```sh
//! scripts/fetch-models.sh
//! ```
//!
//! # Usage
//!
//! ```sh
//! # text prompt: every dog, with an overlay
//! sam3rs dogs.jpg --text dog --out dogs.png
//!
//! # box exemplars (pixels x0,y0,x1,y1), optionally negative ones
//! sam3rs dogs.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600
//!
//! # clicks for a single object
//! sam3rs dogs.jpg --point 1439,504 --out dog.png
//! ```
//!
//! Each detection prints as a JSON line with its score and box. Without `--model DIR`,
//! the command downloads the weights from Hugging Face. As a library:
//!
//! ```no_run
//! use sam3rs::{BoxPrompt, Precision, RgbImage, Sam3, VISUAL_PROMPT};
//!
//! # fn main() -> eyre::Result<()> {
//! // or Sam3::load(dir, 0, Precision::Bf16) for a local copy
//! # #[cfg(feature = "online")]
//! let model = Sam3::from_pretrained(0, Precision::Bf16)?;
//! # #[cfg(not(feature = "online"))]
//! # let model = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
//! let image = RgbImage::open("dogs.jpg".as_ref())?;
//! let embedding = model.encode(&image)?;
//! let dogs = model.detect(&embedding, "dog", &[])?;
//! for (i, det) in dogs.iter().enumerate() {
//!     // each mask is height * width bytes, 1 = foreground
//!     let pixels = dogs.mask(i).iter().filter(|&&m| m != 0).count();
//!     println!("{:.3} {:?} {pixels} px", det.score, det.bbox);
//! }
//!
//! // box exemplars: find more objects like the positive box
//! let exemplar = BoxPrompt::from_xyxy([1233.0, 376.0, 1649.0, 691.0], image.width(), image.height(), true)?;
//! let similar = model.detect(&embedding, VISUAL_PROMPT, &[exemplar])?;
//! # Ok(())
//! # }
//! ```
//!
//! All of a call's masks share one host buffer. Drop the result before the next call and
//! the model reuses that memory instead of faulting in fresh pages.
//!
//! ## Many objects from clicks
//!
//! `segment_points` segments one object from its clicks. For several objects in the same
//! image, pass one prompt per object to `segment_points_batch`, which decodes them
//! together and returns the scores and masks the single calls would:
//!
//! ```no_run
//! use sam3rs::{PointPrompt, Precision, RgbImage, Sam3};
//!
//! # fn main() -> eyre::Result<()> {
//! # let model = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
//! # let embedding = model.encode(&RgbImage::open("dogs.jpg".as_ref())?)?;
//! // one prompt per object; each may hold several clicks, background ones included
//! let dog = [PointPrompt::new(1439.0, 504.0, true)?];
//! let other = [PointPrompt::new(700.0, 900.0, true)?, PointPrompt::new(760.0, 980.0, false)?];
//! let objects = model.segment_points_batch(&embedding, &[&dog, &other])?;
//! for (i, score) in objects.scores().iter().enumerate() {
//!     // objects.mask(i) is height * width bytes, 1 = foreground
//!     println!("object {i}: {score:.3}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Prompts with the same click count share every decoder launch, up to `POINT_BATCH`
//! (16) at a time. Scores match the single calls exactly in `fp32`. In `tf32` and `bf16`,
//! cuBLAS picks other GEMM kernels for the taller batch, so scores can move by about as
//! much as those modes already differ from `fp32`.
//!
//! ## Video tracking
//!
//! Click objects on the first frame, then feed the frames one at a time:
//!
//! ```no_run
//! use sam3rs::{PointPrompt, Precision, RgbImage, Sam3};
//!
//! # fn main() -> eyre::Result<()> {
//! # #[cfg(feature = "online")]
//! let model = Sam3::from_pretrained(0, Precision::Bf16)?;
//! # #[cfg(not(feature = "online"))]
//! # let model = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
//! let frames: Vec<RgbImage> = vec![/* decoded frames, all one size */];
//! // one Vec of clicks per object, in the first frame's pixels
//! let clicks = vec![vec![PointPrompt::new(640.0, 360.0, true)?]];
//! let (mut tracker, first) = model.track(&frames[0], &clicks, frames.len())?;
//! for frame in &frames[1..] {
//!     let out = tracker.next(frame)?;
//!     for obj in &out.objects {
//!         // obj.score > 0 means present; obj.mask is height * width bytes
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The tracker keeps a fixed-size memory bank, so memory use doesn't grow with the video's
//! length. Clicks go on frame 0 only; new clicks on later frames, adding objects mid-video
//! and SAM 3.1's multiplex tracker aren't implemented.
//!
//! # Results
//!
//! Milliseconds for `dogs.jpg` (2560x1600) on an RTX 5090, following the osam benchmark
//! protocol, with full-resolution masks returned on the host:
//!
//! | Runtime | Encode | Point | Box | Neg box | Text |
//! |---|---|---|---|---|---|
//! | osam 0.9.2, ONNX Runtime 1.31.1 CUDA (FP32, TF32 matmuls) | 134.4 | 19.6 | 34.0 | 34.1 | 35.7 |
//! | **sam3rs `tf32`** | **63.7** (2.1x) | **1.4** (14x) | **7.8** (4.4x) | **7.8** (4.4x) | **9.2** (3.9x) |
//! | PyTorch 2.14.1 BF16 (official `Sam3Processor`) | 39.4 | 36.2 | 21.8 | 22.1 | 20.7 |
//! | **sam3rs `bf16`** | **30.9** (1.28x) | **1.5** (24x) | **5.4** (4.0x) | **5.4** (4.1x) | **6.9** (3.0x) |
//! | PyTorch 2.14.1 FP32 (TF32 off) | 129.5 | 25.4 | 33.3 | 33.6 | 31.7 |
//! | **sam3rs `fp32`** | **117.6** (1.10x) | **1.7** (15x) | **26.6** (1.25x) | **26.9** (1.25x) | **28.8** (1.10x) |
//!
//! Each sam3rs row sits under the runtime with the same numerics, and the speedup in
//! parentheses is over that row.
//!
//! Outside the encoder, most of the gap is overhead that sam3rs avoids. osam downloads
//! 133 MB of image features after encoding and uploads 117 MB of them for every detector
//! call. PyTorch's click spends 28 of its 36 ms copying six float32 candidate masks to the
//! host; sam3rs keeps the first pass on the GPU and downloads one mask at a byte per
//! pixel. PyTorch's box and text prompts launch about 2.8k kernels and keep the GPU 46%
//! busy, where sam3rs keeps it about 91% busy. Encode is GEMM-bound, so that's where the
//! margin is thinnest.
//!
//! [Benchmarks](https://github.com/avencera/sam3rs/blob/master/docs/benchmarks.md) has
//! the full protocol, where each millisecond goes, the large-photo and video tables, and
//! the commands that reproduce every row.
//!
//! # Parity
//!
//! `tests/parity.rs` runs four images and 13 prompts through all three precision modes
//! and compares them with PyTorch FP32 outputs of the official implementation. Each check
//! must keep the same DETR queries, with scores within 0.01, box corners within 0.5% of
//! the image's longer side and mask IoU of at least 0.99. Worst case over 43 checks:
//!
//! | Mode | Max score delta | Max box delta | Min mask IoU |
//! |---|---|---|---|
//! | `fp32` | 0.0008 | 0.02 px | 0.9987 |
//! | `tf32` | 0.0052 | 0.19 px | 0.9987 |
//! | `bf16` | 0.0070 | 1.26 px | 0.9964 |
//!
//! The video tracker is checked stage by stage and end to end against upstream's tracker
//! in FP32. On both test clips every mode tracks closer to FP32 than PyTorch's own BF16
//! run:
//!
//! | Mode | Worst stage rel L2 | Mean IoU per object (horses, 3 objects) | Grazing (1 object) |
//! |---|---|---|---|
//! | `fp32` | 2.3e-6 | 0.9999 / 1.0000 / 1.0000 | 1.0000 |
//! | `tf32` | 8.0e-4 | 0.9996 / 0.9987 / 1.0000 | 0.9998 |
//! | `bf16` | 4.4e-3 | 0.9949 / 0.9977 / 0.9999 | 0.9997 |
//! | PyTorch BF16 | | 0.9760 / 0.9918 / 0.9963 | 0.9966 |
//!
//! [Parity](https://github.com/avencera/sam3rs/blob/master/docs/parity.md) has each
//! gate, the stage tests and the commands to run them.
//!
//! # Known differences
//!
//! - JPEG decoding uses the `image` crate rather than PIL's libjpeg-turbo, which moves
//!   scores by up to a few thousandths.
//! - Text cleaning lowercases and collapses whitespace but doesn't port `ftfy` or HTML
//!   unescaping.
//! - Video tracking is SAM 3's tracker-only path from first-frame clicks. The per-frame
//!   detector, text prompts on video and SAM 3.1 aren't implemented.
//! - In `bf16`, upstream rounds the memory's sine position encoding to BF16, and sam3rs
//!   keeps it in FP32.
//! - In `bf16`, hole filling can turn a one-pixel rounding difference into a 64-pixel mask
//!   difference on a point prompt. The score is unaffected.

#![warn(missing_docs)]

mod cuda;
#[cfg(feature = "online")]
mod hub;
mod image;
mod masks;
mod output;
mod posenc;
mod precision;
mod prompt;
mod tokenizer;
mod weights;

pub use cuda::{Embedding, Sam3, VideoTracker};
#[cfg(feature = "online")]
pub use hub::{MODEL_FILES, MODEL_REPO, download_model};
pub use image::{MAX_IMAGE_SIDE, RgbImage};
pub use masks::MaskBuffer;
pub use output::{
    Detection, Detections, POINT_BATCH, PointMask, PointMasks, SCORE_THRESHOLD, TrackedFrame, TrackedObject,
    VISUAL_PROMPT,
};
pub use precision::Precision;
pub use prompt::{BoxPrompt, PointPrompt};
