# sam3rs

A native Rust/CUDA runtime for [Meta SAM 3](https://github.com/facebookresearch/sam3)
image inference. It loads the official Hugging Face `model.safetensors` checkpoint and
runs the whole image path on one CUDA stream without ONNX Runtime or PyTorch:

- Perception Encoder ViT backbone (32 blocks, windowed and global attention with RoPE)
  and both SimpleFPN necks
- CLIP text encoder with the BPE tokenizer
- geometry (box) encoder, fusion encoder, DETR decoder with the presence token,
  dot-product scoring and the segmentation head
- SAM 2 style point decoder with the two-pass mask refinement and hole filling

Image features stay on the GPU between prompts. Kernels that cuBLAS doesn't cover are
Rust, written with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) in
[crates/sam3rs-kernels](crates/sam3rs-kernels). They ship as prebuilt PTX in
`kernels/ptx` that the driver compiles when the model loads, so building sam3rs needs no
CUDA toolkit or cuda-oxide. `scripts/build-ptx.sh` regenerates the PTX after a kernel
change.

There is also a Python package with numpy in and out; see
[crates/sam3rs-python](crates/sam3rs-python/README.md).

## Precision modes

- `bf16` (default): tensor-core GEMMs with FP32 accumulation everywhere, bias and
  activations fused into cuBLASLt epilogues, and a fused flash-attention kernel. The
  backbone, necks, text encoder MLP and fusion encoder FFN keep BF16 weights and
  activations. The rest of the text encoder and fusion encoder, the detector and the
  point decoder keep FP32 ones, and cuBLAS may round their GEMM inputs to BF16, though
  on an RTX 5090 it picks TF32 kernels for most of them. Their unmasked attention, and
  the detector's box-biased cross-attention, also run the fused kernel, which rounds q,
  k and v to FP16 and sums scores and P @ V tiles in FP16.
- `tf32`: the `fp32` path with TF32 tensor-core GEMMs (FP32 accumulation), the way
  ONNX Runtime's CUDA provider runs FP32 models by default, with bias and activations
  fused into cuBLASLt epilogues. The first call of each GEMM shape times cuBLASLt's top
  candidate algorithms and switches from its first choice only to one more than 5%
  faster, so close calls keep the same algorithm from run to run. FP32 weights and
  activations. Attention without a causal mask or additive bias, the detector's
  box-biased cross-attention included, runs a fused flash-attention kernel on TF32
  tensor cores with an FP32 softmax; the rest stays materialized.
- `fp32`: strict FP32 (cuBLAS pedantic math, no TF32) with materialized attention. This
  is the reference path for parity checks.

## Requirements

- An NVIDIA GPU with compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell)
  and driver r580 or newer. RTX 50 series GPUs load compute_120 PTX, others the
  compute_80 floor.
- cuBLAS 13 (`libcublas.so.13`, `libcublasLt.so.13`) on the library path, from a CUDA 13
  toolkit or the `nvidia-cublas` pip wheel
- Stable Rust (edition 2024)

libcuda and cuBLAS load at run time. A missing driver, GPU or cuBLAS makes
`Sam3::load` return an error rather than panic.

Cargo features:

- `online` (default): `Sam3::from_pretrained` and `download_model`, which fetch the
  weights from Hugging Face
- `cli`: the `sam3rs` command
- `nvtx`: NVTX ranges per pipeline stage for Nsight Systems. The only feature that
  needs a CUDA toolkit to build, for its headers (`CUDA_HOME`, default
  `/usr/local/cuda`).

## Weights

SAM 3 weights are under Meta's SAM License and aren't part of this repository or its
packages. [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face is
gated: accept Meta's terms on the model page while logged in, then run
`hf auth login` or set `HF_TOKEN` to a token of that account.
`Sam3::from_pretrained` and the CLI without `--model` download the checkpoint into
the Hugging Face cache on first use.

For the tests and benchmarks, `scripts/fetch-models.sh` downloads the checkpoint into
`models/sam3` with the same token, fetches the test images into `models/images`, and
checks the weights against the official sha256 hashes:

```sh
scripts/fetch-models.sh
```

## Usage

```sh
cargo build --release --features cli

# text prompt: every dog, with an overlay
./target/release/sam3rs models/images/dogs.jpg --text dog --out dogs.png

# box exemplars (pixels x0,y0,x1,y1), optionally negative ones
./target/release/sam3rs models/images/dogs.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600

# clicks for a single object
./target/release/sam3rs models/images/dogs.jpg --point 1439,504 --out dog.png
```

Each detection prints as a JSON line with its score and box. Without `--model DIR`,
the command downloads the weights from Hugging Face. As a library:

```rust
use sam3rs::{BoxPrompt, Precision, RgbImage, Sam3, VISUAL_PROMPT};

// or Sam3::load(dir, 0, Precision::Bf16) for a local copy
let model = Sam3::from_pretrained(0, Precision::Bf16)?;
let image = RgbImage::open("dogs.jpg".as_ref())?;
let embedding = model.encode(&image)?;
for det in model.detect(&embedding, "dog", &[])? {
    println!("{:.3} {:?}", det.score, det.bbox);
}

// box exemplars: find more objects like the positive box
let exemplar = BoxPrompt::from_xyxy([1233.0, 376.0, 1649.0, 691.0], image.width(), image.height(), true)?;
let dogs = model.detect(&embedding, VISUAL_PROMPT, &[exemplar])?;
```

## Results

Milliseconds on an RTX 5090 (driver 595.99, CUDA 13.2), following the osam benchmark
protocol: `dogs.jpg` (2560x1600), one warm-up, then the median of 5. Model load and JPEG
decoding are excluded. Prompts are one point (1439, 504), one box
(1233, 376, 1649, 691), that box plus a negative box (1400, 500, 1500, 600), and the text
"dog". Each prompt returns full-resolution masks on the host. The text prompt runs the
text encoder on every call. Box prompts use the fixed text "visual", whose features
sam3rs encodes once and reuses, while osam and PyTorch encode it again on every call.

| Runtime | Encode | Point | Box | Neg box | Text |
|---|---|---|---|---|---|
| osam 0.9.2, ONNX Runtime 1.31.1 CUDA (FP32, TF32 matmuls) | 134.4 | 19.6 | 34.0 | 34.1 | 35.7 |
| **sam3rs `tf32`** | **63.9** | **1.9** | **8.0** | **8.0** | **9.3** |
| PyTorch 2.14.1 BF16 (official `Sam3Processor`) | 39.6 | 35.8 | 21.1 | 21.4 | 20.5 |
| **sam3rs `bf16`** | **33.3** | **2.0** | **5.5** | **5.5** | **7.0** |
| PyTorch 2.14.1 FP32 (TF32 off) | 129.2 | 23.8 | 33.4 | 33.7 | 31.8 |
| sam3rs `fp32` | 143.1 | 2.3 | 27.2 | 27.5 | 29.5 |

Each sam3rs row sits under the runtime that uses the same numerics:

- `tf32` against osam, both FP32 with TF32 matmuls: 2.1x on encode, 10x on points,
  4.3x on box prompts and 3.8x on text prompts.
- `bf16` against PyTorch BF16: 1.19x on encode, 18x on points, 3.8x on box prompts and
  2.9x on text prompts.
- `fp32` against PyTorch FP32 with TF32 off: PyTorch encodes 11% faster. sam3rs is 10x
  faster on points, 1.2x on box prompts and 1.08x on text prompts.

The sam3rs rows come from one queued session, three runs each, reporting the median of
each column. The osam and PyTorch rows come from earlier sessions on the same machine.

Where the time goes:

- Image features stay on the GPU. osam downloads 133 MB of features at encode and
  re-uploads 117 MB on every detector call, and part of its point decoder runs on the
  CPU.
- The point path is small kernels plus hole filling on the host. PyTorch spends about
  17 ms of each point prompt filling holes with GPU connected components.
- PyTorch's box and text prompts launch about 2.8k kernels each and keep the GPU 46%
  busy. sam3rs issues about 0.6k GPU operations for a box prompt and 0.95k for a text
  prompt, and keeps the GPU about 91% busy. A BF16 box prompt spends about 1.8 ms in the
  fusion encoder, 1.6 ms in the mask head and 1.2 ms in the DETR decoder, then about
  0.3 ms per kept mask copying it to the host.
- Encode is GEMM-bound. The backbone MLP and qkv GEMMs run at 195-210 TFLOPS, close to
  the card's dense BF16 rate with FP32 accumulation, so the margin over PyTorch BF16 is
  small. The rest is fused bias and GELU epilogues, BF16 activations fed straight into
  the next GEMM, and the fused attention kernel.

The bench example reproduces the sam3rs rows:

```sh
cargo run --release --example bench -- --precision bf16
cargo run --release --example bench -- --precision fp32
```

## Parity

`tests/parity.rs` runs four images and 13 prompts (text, box, negative box, points)
through all three precision modes and compares against PyTorch FP32 outputs of the official
implementation. The fixtures come from `scripts/reference.py dump`. The gate: the same
kept DETR queries, scores within 0.01, box corners within 0.5% of the image's longer
side, and mask IoU of at least 0.99.

Worst case over 43 checks:

| Mode | Max score delta | Max box delta | Min mask IoU |
|---|---|---|---|
| `fp32` | 0.0008 | 0.02 px | 0.9987 |
| `tf32` | 0.0052 | 0.19 px | 0.9987 |
| `bf16` | 0.0046 | 0.71 px | 0.9914 |

```sh
cargo test --release --test parity -- --nocapture
```

The ignored `stage_tests::stages` unit test (`cargo test --release --lib -- --ignored`)
compares every intermediate stage (pixels, trunk, FPN levels, text features, scores,
boxes, mask logits) for one fixture.

## Known differences

- JPEG decoding uses the `image` crate, which differs from PIL's libjpeg-turbo on a few
  percent of bytes. Scores move by up to a few thousandths. The parity test feeds the
  reference's decoded RGB to isolate the model.
- Text cleaning lowercases and collapses whitespace but doesn't port `ftfy` or HTML
  unescaping.
- Only the image path is implemented: no video tracking or memory bank.
- The lowest `bf16` mask IoU (0.9914, a point prompt) comes from hole filling. A
  rounding difference flips one low-resolution pixel, which cuts a 64-pixel background
  region off from the rest. Hole filling turns background regions of up to 256 pixels
  into foreground, so that region becomes part of the mask. The score is unaffected.

## Not yet verified

- Only measured on one RTX 5090 (sm_120) with CUDA 13.2. The compute_80 PTX that other
  GPUs load passes parity on the 5090 but hasn't run on another architecture.
- Timings use one image and one prompt of each kind. Scaling with many box exemplars
  or long text prompts hasn't been measured.
- Parity covers four images and 13 prompts, and the BF16 worst score delta is within
  about 0.005 of the 0.01 gate.

## License

TODO(license): pick the license before release.
