//! A native Rust/CUDA runtime for [Meta SAM 3](https://github.com/facebookresearch/sam3)
//! image inference and click-to-track video. It loads the official Hugging Face
//! `model.safetensors` checkpoint and runs the whole image path and the video tracker on
//! one CUDA stream without ONNX Runtime or PyTorch:
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
//! Image features stay on the GPU between prompts. Kernels that cuBLAS doesn't cover are
//! Rust, written with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) in
//! [crates/sam3rs-kernels](https://github.com/avencera/sam3rs/tree/master/crates/sam3rs-kernels). They ship as prebuilt PTX in
//! `kernels/ptx` that the driver compiles when the model loads, so building sam3rs needs no
//! CUDA toolkit or cuda-oxide. `scripts/build-ptx.sh` regenerates the PTX after a kernel
//! change.
//!
//! There is also a Python package with numpy in and out; see
//! [crates/sam3rs-python](https://github.com/avencera/sam3rs/tree/master/crates/sam3rs-python).
//!
//! Model downloads use [`hf-fetch`](https://github.com/avencera/hf-fetch), a small
//! blocking client with rustls and the shared Python Hugging Face cache. The `online`
//! feature enables downloads; local model loading works without it.
//!
//! # Precision modes
//!
//! - `bf16` (default): tensor-core GEMMs with FP32 accumulation, bias and activations
//!   fused into cuBLASLt epilogues, and a fused flash-attention kernel. The backbone,
//!   necks, text encoder MLP and fusion encoder FFN keep BF16 weights and activations.
//!   The rest of the text encoder and fusion encoder, the detector and the point decoder
//!   keep FP32 ones, and cuBLAS may round their GEMM inputs to BF16, though on an RTX 5090
//!   it picks TF32 kernels for most of them. Their unmasked attention, and the detector's
//!   box-biased cross-attention, also run the fused kernel, which rounds q, k and v to
//!   FP16 and sums scores in FP16. In every fused attention, v is FP16 and each 64-key
//!   P @ V tile is summed in FP16 before it is added to FP32 totals.
//! - `tf32`: the `fp32` path with TF32 tensor-core GEMMs (FP32 accumulation), the way
//!   ONNX Runtime's CUDA provider runs FP32 models by default, with bias and activations
//!   fused into cuBLASLt epilogues. The first call of each GEMM shape times cuBLASLt's top
//!   candidate algorithms and switches from its first choice only to one more than 5%
//!   faster. A candidate near that margin can win in one run and lose in the next, so
//!   outputs can differ slightly between runs (about 0.00001 in score). FP32 weights and
//!   activations. Attention without a causal mask or additive bias, the detector's
//!   box-biased cross-attention included, runs a fused flash-attention kernel on TF32
//!   tensor cores with an FP32 softmax; the rest stays materialized.
//! - `fp32`: strict FP32, with no TF32 or BF16 anywhere: cuBLAS GEMMs in pedantic math and
//!   cuBLASLt linears with FP32 compute, with bias and activations fused into their
//!   epilogues. The vision backbone's attention runs a fused flash-attention kernel on FP32
//!   FMAs with an online softmax, so it never writes the score matrix; the detector's
//!   attention stays materialized. This is the reference path for parity checks.
//!
//! # Requirements
//!
//! - An NVIDIA GPU with compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell)
//!   and driver r580 or newer. RTX 50 series GPUs load compute_120 PTX, others the
//!   compute_80 floor.
//! - cuBLAS 13 (`libcublas.so.13`, `libcublasLt.so.13`) on the library path, from a CUDA 13
//!   toolkit or the `nvidia-cublas` pip wheel
//! - Stable Rust (edition 2024)
//!
//! libcuda and cuBLAS load at run time. A missing driver, GPU or cuBLAS makes
//! `Sam3::load` return an error rather than panic. cuBLAS opens by those two sonames, so
//! a copy already in the process, such as the pip wheel the Python package loads, is
//! reused; otherwise the loader's search order picks one (`LD_LIBRARY_PATH`, then the
//! ldconfig cache).
//!
//! Cargo features:
//!
//! - `online` (default): `Sam3::from_pretrained` and `download_model`, which fetch the
//!   weights from Hugging Face
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
//! cargo build --release --features cli
//!
//! # text prompt: every dog, with an overlay
//! ./target/release/sam3rs models/images/dogs.jpg --text dog --out dogs.png
//!
//! # box exemplars (pixels x0,y0,x1,y1), optionally negative ones
//! ./target/release/sam3rs models/images/dogs.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600
//!
//! # clicks for a single object
//! ./target/release/sam3rs models/images/dogs.jpg --point 1439,504 --out dog.png
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
//! All of a call's masks share one host buffer. Dropping the result hands that memory
//! back to the model, so the next call writes its masks into pages that are already
//! mapped. Faulting in fresh memory for 27 masks at 2560 px took about 27 ms.
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
//! Milliseconds on an RTX 5090 (driver 595.99, CUDA 13.2), following the osam benchmark
//! protocol: `dogs.jpg` (2560x1600), one warm-up, then the median of 5. Model load and JPEG
//! decoding are excluded. Prompts are one point (1439, 504), one box
//! (1233, 376, 1649, 691), that box plus a negative box (1400, 500, 1500, 600), and the text
//! "dog". Each prompt returns full-resolution masks on the host. The text prompt runs the
//! text encoder on every call. Box prompts use the fixed text "visual", whose features
//! sam3rs encodes once and reuses, while osam and PyTorch encode it again on every call.
//!
//! | Runtime | Encode | Point | Box | Neg box | Text |
//! |---|---|---|---|---|---|
//! | osam 0.9.2, ONNX Runtime 1.31.1 CUDA (FP32, TF32 matmuls) | 134.4 | 19.6 | 34.0 | 34.1 | 35.7 |
//! | **sam3rs `tf32`** | **64.7** | **1.9** | **7.9** | **7.8** | **9.2** |
//! | PyTorch 2.14.1 BF16 (official `Sam3Processor`) | 39.4 | 36.2 | 21.8 | 22.1 | 20.7 |
//! | **sam3rs `bf16`** | **31.1** | **2.0** | **5.4** | **5.4** | **6.9** |
//! | PyTorch 2.14.1 FP32 (TF32 off) | 129.5 | 25.4 | 33.3 | 33.6 | 31.7 |
//! | **sam3rs `fp32`** | **120.6** | **2.2** | **26.8** | **27.2** | **29.1** |
//!
//! Each sam3rs row sits under the runtime that uses the same numerics:
//!
//! - `tf32` against osam, both FP32 with TF32 matmuls: 2.1x on encode, 10x on points,
//!   4.3x on box prompts and 3.9x on text prompts.
//! - `bf16` against PyTorch BF16: 1.27x on encode, 18x on points, 4.0x on box prompts and
//!   3.0x on text prompts.
//! - `fp32` against PyTorch FP32 with TF32 off: 1.07x on encode, 12x on points, 1.24x on
//!   box prompts and 1.09x on text prompts.
//!
//! The sam3rs rows come from three runs of one queued session, reporting the median of
//! each column. The PyTorch rows come from an earlier queued session on the same machine,
//! three runs each, and the osam row from another.
//!
//! Where the time goes:
//!
//! - Image features stay on the GPU. osam downloads 133 MB of features at encode and
//!   re-uploads 117 MB on every detector call, and part of its point decoder runs on the
//!   CPU.
//! - The point path is small kernels plus hole filling on the host. PyTorch spends about
//!   17 ms of each point prompt filling holes with GPU connected components.
//! - PyTorch's box and text prompts launch about 2.8k kernels each and keep the GPU 46%
//!   busy. sam3rs issues about 0.6k GPU operations for a box prompt and 0.95k for a text
//!   prompt, and keeps the GPU about 91% busy. A BF16 box prompt spends about 1.8 ms in the
//!   fusion encoder, 1.6 ms in the mask head and 1.2 ms in the DETR decoder, then about
//!   0.3 ms per kept mask copying it to the host, into memory reused from earlier calls.
//!   The Python package views those bytes as numpy bool arrays without copying them.
//! - Encode is GEMM-bound. The backbone MLP and qkv GEMMs run at 195-210 TFLOPS, close to
//!   the card's dense BF16 rate with FP32 accumulation, so the margin over PyTorch BF16 is
//!   small. The rest is fused bias and GELU epilogues, BF16 activations fed straight into
//!   the next GEMM, and the fused attention kernel.
//! - Strict FP32 has no tensor cores to lean on, so its encode is bound by the FP32 pipes.
//!   The backbone's attention runs fused on them and takes about 16 ms of the encode,
//!   where materializing the scores took 42 ms. Its linears run the same CUTLASS SIMT
//!   GEMMs as PyTorch's, with bias and GELU fused into their epilogues.
//!
//! The bench example reproduces the sam3rs rows:
//!
//! ```sh
//! cargo run --release --example bench -- --precision bf16
//! cargo run --release --example bench -- --precision fp32
//! ```
//!
//! ## Video
//!
//! Steady-state milliseconds a frame on the same RTX 5090, BF16, one click per object on
//! frame 0, masks at the video's size on the host: the median over frames 8 onward of a
//! 64-frame 720p clip (`horses`), after one warm-up run. PyTorch runs upstream's
//! tracker-only path (`build_sam3_video_model().tracker`) under BF16 autocast; ComfyUI is
//! the core `SAM3_VideoTrack` node with SAM 3.1 multiplex FP16 weights, its node time over
//! the frame count.
//!
//! | Runtime | 1 object | 3 objects | Peak VRAM |
//! |---|---|---|---|
//! | PyTorch 2.14.1 eager | 45.5 | 60.1 | 5.6-6.0 GB |
//! | PyTorch 2.14.1 `torch.compile` | 42.3 | 55.9 | 5.8-6.5 GB |
//! | PyTorch 2.14.1 CUDA graphs | 45.1 | 59.7 | 10-20 GB |
//! | ComfyUI core (SAM 3.1 multiplex) | 44.7 | 46.3 | 3.8-4.0 GB |
//! | **sam3rs `bf16`** | **34.8** | **44.2** | **3.4-3.7 GB** |
//! | sam3rs `fp32` | 148.5 | 214.0 | 5.3-5.4 GB |
//!
//! At 1080p and 256 frames sam3rs takes 35.7 and 45.6 ms. The image encoder and tracker
//! neck, shared by all objects, are about 30 ms of each frame. Each object adds 4.7 ms,
//! about 1.8 of them in the memory cross-attention kernel and 0.44 in the memory
//! self-attention. The GPU is about 99% busy, so CUDA graphs would
//! save little. ComfyUI's multiplex tracker costs the same for 1 to 16 objects.
//!
//! ```sh
//! cargo run --release --example bench_video -- --frames DIR --clip horses --objects 3
//! ```
//!
//! `--frames` is a directory of decoded frames, read in name order. Clicks come from
//! `tests/video_prompts.json` (`--prompts`). `--outputs DIR` writes the last run's
//! scores and masks for an A/B gate.
//!
//! # Parity
//!
//! `tests/parity.rs` runs four images and 13 prompts (text, box, negative box, points)
//! through all three precision modes and compares against PyTorch FP32 outputs of the official
//! implementation. The fixtures come from `scripts/reference.py dump`. The gate: the same
//! kept DETR queries, scores within 0.01, box corners within 0.5% of the image's longer
//! side, and mask IoU of at least 0.99.
//!
//! Worst case over 43 checks:
//!
//! | Mode | Max score delta | Max box delta | Min mask IoU |
//! |---|---|---|---|
//! | `fp32` | 0.0008 | 0.02 px | 0.9987 |
//! | `tf32` | 0.0052 | 0.19 px | 0.9987 |
//! | `bf16` | 0.0070 | 1.26 px | 0.9964 |
//!
//! ```sh
//! cargo test --release --test parity -- --nocapture
//! ```
//!
//! The ignored `stage_tests::stages` unit test (`cargo test --release --lib -- --ignored`)
//! compares every intermediate stage (pixels, trunk, FPN levels, text features, scores,
//! boxes, mask logits) for one fixture.
//!
//! The video tracker's gates are ignored tests in `src/video_tests.rs`, against fixtures
//! from `scripts/video_reference.py dump`: upstream's tracker in true FP32 on a clip, with
//! every stage's inputs and outputs on a few frames.
//!
//! - `video_stages` runs each stage (memory attention, memory encoder, decoder, pointer
//!   MLP) on the reference's exact inputs: rel L2 within 1e-4 in `fp32` and 5e-3 in
//!   `tf32` and `bf16` (4e-2 for the backbone and neck, the image path's encoder).
//! - `video_e2e` tracks the whole clip and needs the reference's memory length, pointer
//!   count and object presence on every frame, plus a mask IoU per object against FP32:
//!   at least 0.99 in `fp32`, and in the reduced modes no worse than PyTorch's own BF16
//!   run (mean minus 0.01, and its worst frame). Up to two frames an object are excused
//!   from the worst-frame limit where the decoder's top two candidates were within 0.005
//!   predicted IoU.
//! - `video_e2e` in `bf16` also runs each checked frame's memory attention over the cached
//!   per-frame keys against the FP32 model's on the same inputs (rel L2 within 5e-3).
//! - `attention_d256` checks the two fused attention kernels (values of 256 for
//!   self-attention, of 64 for the memory cross-attention) against an f64 softmax: rel L2
//!   within 1e-3.
//!
//! | Mode | Worst stage rel L2 | Mean IoU per object (horses, 3 objects) | Grazing (1 object) |
//! |---|---|---|---|
//! | `fp32` | 2.3e-6 | 0.9999 / 1.0000 / 1.0000 | 1.0000 |
//! | `tf32` | 8.0e-4 | 0.9996 / 0.9987 / 1.0000 | 0.9998 |
//! | `bf16` | 4.4e-3 | 0.9949 / 0.9977 / 0.9999 | 0.9997 |
//! | PyTorch BF16 | | 0.9760 / 0.9918 / 0.9963 | 0.9966 |
//!
//! ```sh
//! SAM3RS_VIDEO=models/video-parity/horses-720p-64-o3-fp32.safetensors SAM3RS_PRECISION=bf16 \
//!     cargo test --release --lib video_tests -- --ignored --nocapture --test-threads 1
//! ```
//!
//! # Known differences
//!
//! - JPEG decoding uses the `image` crate, which differs from PIL's libjpeg-turbo on a few
//!   percent of bytes. Scores move by up to a few thousandths. The parity test feeds the
//!   reference's decoded RGB to isolate the model.
//! - Text cleaning lowercases and collapses whitespace but doesn't port `ftfy` or HTML
//!   unescaping.
//! - Video tracking follows SAM 3's tracker-only path ("SAM 3 for SAM 2 video task")
//!   from clicks on the first frame. The session predictors' per-frame detector, text
//!   prompts on video, and SAM 3.1 aren't implemented.
//! - In `bf16`, upstream rounds the memory's sine position encoding to BF16 under
//!   autocast, and sam3rs keeps it in FP32.
//! - The lowest `bf16` mask IoU (0.9914, a point prompt) comes from hole filling. A
//!   rounding difference flips one low-resolution pixel, which cuts a 64-pixel background
//!   region off from the rest. Hole filling turns background regions of up to 256 pixels
//!   into foreground, so that region becomes part of the mask. The score is unaffected.
//!
//! # Not yet verified
//!
//! - Only measured on one RTX 5090 (sm_120) with CUDA 13.2. The compute_80 PTX that other
//!   GPUs load passes parity on the 5090 but hasn't run on another architecture.
//! - Video parity covers two clips (720p, 64 frames, 1 and 3 objects).
//! - Timings use one image and one prompt of each kind. Scaling with many box exemplars
//!   or long text prompts hasn't been measured.
//! - Parity covers four images and 13 prompts, and the BF16 worst score delta is within
//!   about 0.005 of the 0.01 gate.

#![warn(missing_docs)]

mod blaslt;
mod cublas;
mod detector;
mod gpu;
#[cfg(feature = "online")]
mod hub;
mod image;
mod masks;
mod nvtx;
mod ops;
mod pipeline;
mod point;
mod posenc;
mod text;
mod tokenizer;
mod video;
mod vision;
mod weights;

#[cfg(test)]
mod stage_tests;
#[cfg(test)]
mod video_tests;

pub use detector::BoxPrompt;
pub use gpu::Precision;
#[cfg(feature = "online")]
pub use hub::{MODEL_FILES, MODEL_REPO, download_model};
pub use image::{MAX_IMAGE_SIDE, RgbImage};
pub use masks::MaskBuffer;
pub use pipeline::{Detection, Detections, Embedding, PointMask, SCORE_THRESHOLD, Sam3, VISUAL_PROMPT};
pub use point::PointPrompt;
pub use video::{TrackedFrame, TrackedObject, VideoTracker};
