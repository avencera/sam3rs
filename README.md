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
CUDA C++ in `kernels/sam3.cu`, built by `build.rs` with nvcc.

## Precision modes

- `bf16` (default): BF16 tensor-core GEMMs with FP32 accumulation for the backbone,
  necks and convs, bias and GELU fused into cuBLASLt epilogues, and a fused
  flash-attention kernel. Activations between layers stay FP32 outside the backbone.
- `fp32`: strict FP32 (cuBLAS pedantic math, no TF32) with materialized attention. This
  is the reference path for parity checks.

## Requirements

- An NVIDIA GPU. Kernels are built for `sm_120` (RTX 50 series) by default; set
  `SAM3RS_CUDA_ARCH` (for example `90` or `89`) to target another architecture.
- CUDA toolkit 13.x with `nvcc` (`CUDA_HOME`, default `/usr/local/cuda`)
- Stable Rust (edition 2024)

## Weights

SAM 3 weights are under Meta's SAM License and aren't part of this repository.
`facebook/sam3` on Hugging Face is gated, so `scripts/fetch-models.sh` downloads from a
mirror and checks every file against the official sha256 hashes:

```sh
scripts/fetch-models.sh
```

## Usage

```sh
cargo build --release

# text prompt: every dog, with an overlay
./target/release/sam3rs models/images/dogs.jpg --text dog --out dogs.png

# box exemplars (pixels x0,y0,x1,y1), optionally negative ones
./target/release/sam3rs models/images/dogs.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600

# clicks for a single object
./target/release/sam3rs models/images/dogs.jpg --point 1439,504 --out dog.png
```

Each detection prints as a JSON line with its score and box. As a library:

```rust
use sam3rs::gpu::Precision;
use sam3rs::image::RgbImage;
use sam3rs::pipeline::Sam3;

let model = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
let image = RgbImage::open("dogs.jpg".as_ref())?;
let embedding = model.encode(&image)?;
for det in model.detect(&embedding, "dog", &[])? {
    println!("{:.3} {:?}", det.score, det.bbox);
}
```

## Results

Milliseconds on an RTX 5090 (driver 595.99, CUDA 13.2), following the osam benchmark
protocol: `dogs.jpg` (2560x1600), one warm-up, then the median of 5. Model load and JPEG
decoding are excluded. Prompts are one point (1439, 504), one box
(1233, 376, 1649, 691), that box plus a negative box (1400, 500, 1500, 600), and the text
"dog". Each prompt returns full-resolution masks on the host.

| Runtime | Encode | Point | Box | Neg box | Text |
|---|---|---|---|---|---|
| osam 0.9.2, ONNX Runtime 1.31.1 CUDA (FP32, TF32 matmuls) | 134.1 | 19.4 | 33.8 | 33.8 | 35.5 |
| PyTorch 2.14.1 BF16 (official `Sam3Processor`) | 39.6 | 35.8 | 21.1 | 21.4 | 20.5 |
| PyTorch 2.14.1 FP32 (TF32 off) | 129.2 | 23.8 | 33.4 | 33.7 | 31.8 |
| **sam3rs `bf16`** | **36.7** | **2.3** | **12.0** | **12.0** | **11.8** |
| sam3rs `fp32` | 147.2 | 2.5 | 31.0 | 31.1 | 31.3 |

sam3rs `bf16` against osam: 3.7x on encode, 8.4x on points and 2.8-3.0x on box and text
prompts. Against PyTorch BF16: 1.08x on encode, 15x on points and 1.7-1.8x on box and
text prompts.

Where the time goes:

- Image features stay on the GPU. osam downloads 133 MB of features at encode and
  re-uploads 117 MB on every detector call, and part of its point decoder runs on the
  CPU.
- The point path is small kernels plus hole filling on the host. PyTorch spends about
  17 ms of each point prompt filling holes with GPU connected components.
- PyTorch's box and text prompts launch about 2.8k kernels each and keep the GPU 46%
  busy. sam3rs launches about 1.1k and keeps it 88% busy.
- Encode is GEMM-bound. The backbone MLP and qkv GEMMs run at 195-210 TFLOPS, close to
  the card's dense BF16 rate with FP32 accumulation, so the margin over PyTorch BF16 is
  small. The rest is fused bias and GELU epilogues, BF16 activations fed straight into
  the next GEMM, and the fused attention kernel.

`target/release/bench` reproduces the sam3rs rows:

```sh
./target/release/bench --precision bf16
./target/release/bench --precision fp32
```

## Parity

`tests/parity.rs` runs four images and 13 prompts (text, box, negative box, points)
through both precision modes and compares against PyTorch FP32 outputs of the official
implementation. The fixtures come from `scripts/reference.py dump`. The gate: the same
kept detections, scores within 0.01, and mask IoU of at least 0.99.

Worst case over 34 checks:

| Mode | Max score delta | Min mask IoU |
|---|---|---|
| `fp32` | 0.0008 | 0.9987 |
| `bf16` | 0.0063 | 0.9964 |

```sh
cargo test --release --test parity -- --nocapture
```

`examples/stages.rs` compares every intermediate stage (pixels, trunk, FPN levels, text
features, scores, boxes, mask logits) for one fixture.

## Known differences

- JPEG decoding uses the `image` crate, which differs from PIL's libjpeg-turbo on a few
  percent of bytes. Scores move by up to a few thousandths. The parity test feeds the
  reference's decoded RGB to isolate the model.
- Text cleaning lowercases and collapses whitespace but doesn't port `ftfy` or HTML
  unescaping.
- Only the image path is implemented: no video tracking or memory bank.

## Not yet verified

- Only measured on one RTX 5090 (sm_120) with CUDA 13.2. Other architectures build via
  `SAM3RS_CUDA_ARCH` but haven't been run.
- Timings use one image and one prompt of each kind. Scaling with many box exemplars
  or long text prompts hasn't been measured.
- Parity covers four images and 13 prompts. BF16 mode keeps the detector and fusion
  layers on FP32 because moving them to BF16 pushed score deltas to 0.009, near the gate.
