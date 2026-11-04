# Benchmarks

Every number here is from one RTX 5090 (driver 595.99, CUDA 13.2), except under
[Apple silicon](#apple-silicon). The headline tables
and the image results are in the [README](../README.md#results).

## Image prompts

The image table follows the osam benchmark protocol: `dogs.jpg` (2560x1600), one
warm-up, then the median of 5, and each row is the median of three such runs. Model load
and JPEG decoding are excluded. Prompts are one point (1439, 504), one box
(1233, 376, 1649, 691), that box plus a negative box (1400, 500, 1500, 600), and the text
"dog". Each prompt returns full-resolution masks on the host. The text prompt runs the
text encoder on every call. Box prompts use the fixed text "visual", whose features
sam3rs encodes once and reuses, while osam and PyTorch encode it again on every call.

Each sam3rs row is compared with the runtime that uses the same numerics: `tf32`
against osam, both FP32 with TF32 matmuls; `bf16` against PyTorch BF16; `fp32` against
PyTorch FP32 with TF32 off.

The bench example reproduces the sam3rs rows:

```sh
cargo run --release --example bench -- --precision bf16
cargo run --release --example bench -- --precision fp32
```

### Where the time goes

- Image features stay on the GPU. osam downloads 133 MB of features at encode and
  re-uploads 117 MB on every detector call, and part of its point decoder runs on the
  CPU.
- The point path is small kernels, hole filling included: a union-find over the
  low-res masks in four launches. PyTorch's hole filling costs about 0.2 ms; most of
  its point prompt is copying masks to the host. It upsamples all three candidates of
  both passes to full size as float32 and copies them into freshly allocated host
  memory: at 2560x1600, two 49 MB copies, 28 ms of a 36-38 ms prompt. sam3rs keeps
  the first pass on the GPU and downloads one mask at one byte per pixel.
- PyTorch's box and text prompts launch about 2.8k kernels each and keep the GPU 46%
  busy. sam3rs issues about 0.6k GPU operations for a box prompt and 0.95k for a text
  prompt, and keeps the GPU about 91% busy. A BF16 box prompt spends about 1.8 ms in the
  fusion encoder, 1.6 ms in the mask head and 1.2 ms in the DETR decoder, then about
  0.3 ms per kept mask copying it to the host, into memory reused from earlier calls.
  The Python package views those bytes as numpy bool arrays without copying them.
- All of a call's masks share one host buffer, and dropping the result hands it back to
  the model, so the next call writes into pages that are already mapped. Faulting in
  fresh memory for 27 masks at 2560 px took about 27 ms.
- Encode is GEMM-bound. The backbone MLP and qkv GEMMs run at 195-210 TFLOPS, close to
  the card's dense BF16 rate with FP32 accumulation, so the margin over PyTorch BF16 is
  small. The rest is fused bias and GELU epilogues, BF16 activations fed straight into
  the next GEMM, and the fused attention kernel.
- Strict FP32 has no tensor cores to lean on, so its encode is bound by the FP32 pipes.
  The backbone's attention runs fused on them and takes about 16 ms of the encode,
  where materializing the scores took 42 ms. Its linears run the same CUTLASS SIMT
  GEMMs as PyTorch's, with bias and GELU fused into their epilogues.

## Live rates

The live table in the README counts answers a second while the input keeps changing.
The cursor moves at 120 Hz, and each side answers the newest position whenever it
finishes the last, so 120 means sam3rs answers every position. The 8-mask row is one
batched call per move. A new frame with a text prompt is an encode plus the prompt,
31 + 7 ms against 39 + 21 ms.

## Large photos

Measured through the Python package with both runtimes on cuBLAS 13.8.2, each photo at
its own size and the mask returned at that size. A click runs both decoder passes and
hole filling; the cursor columns count masks a second while a cursor moves at 120 Hz:

| Photo | sam3rs click | PyTorch click | Speedup | sam3rs cursor | PyTorch cursor |
|---|---|---|---|---|---|
| 1600x1200 (2 MP) | 1.6 ms | 7.8 ms | 4.9x | 120/s | 120/s |
| 4288x2848 (12 MP) | 2.4 ms | 103.2 ms | 43x | 120/s | 10/s |
| 7360x4912 (36 MP) | 4.3 ms | 316.1 ms | 74x | 120/s | 3/s |
| 8160x6144 (50 MP) | 5.4 ms | 460.9 ms | 85x | 120/s | 2/s |
| 12000x9000 (108 MP) | 9.9 ms | 924.7 ms | 93x | 90/s | 1/s |

The gap grows with the photo because PyTorch's `predict_inst` returns all three
candidates of both passes at full size as float32, 24 bytes per pixel, in freshly
allocated host memory. sam3rs returns one mask at one byte per pixel.

Eight clicks per cursor move in one batched call, on the 36 MP photo, take 35 ms
against 2.6 s: 28 updates a second against 0.4.

Segmenting everything, as a 16x16 grid of clicks run in batches through
`segment_points_batch` and PyTorch's batched `predict_inst`, takes 1.46 s against
105.3 s on the 50 MP photo (72x) and 2.77 s against 221.7 s at 108 MP (80x).

## Video

Steady-state milliseconds a frame in BF16, one click per object on frame 0, masks at the
video's size on the host: the median over frames 8 onward of a 64-frame 720p clip
(`horses`), after one warm-up run. PyTorch runs upstream's tracker-only path
(`build_sam3_video_model().tracker`) under BF16 autocast; ComfyUI is the core
`SAM3_VideoTrack` node with SAM 3.1 multiplex FP16 weights, its node time over the frame
count.

| Runtime | 1 object | 3 objects | Peak VRAM |
|---|---|---|---|
| PyTorch 2.14.1 eager | 45.5 ms, 22 fps | 60.1 ms, 17 fps | 5.6-6.0 GB |
| PyTorch 2.14.1 `torch.compile` | 42.3 ms, 24 fps | 55.9 ms, 18 fps | 5.8-6.5 GB |
| PyTorch 2.14.1 CUDA graphs | 45.1 ms, 22 fps | 59.7 ms, 17 fps | 10-20 GB |
| ComfyUI core (SAM 3.1 multiplex) | 44.7 ms, 22 fps | 46.3 ms, 22 fps | 3.8-4.0 GB |
| **sam3rs `bf16`** | **34.8 ms, 29 fps** (1.31x) | **44.2 ms, 23 fps** (1.36x) | **3.4-3.7 GB** |
| sam3rs `fp32` | 148.5 ms, 7 fps | 214.0 ms, 5 fps | 5.3-5.4 GB |

The speedups in parentheses are over PyTorch eager. Over the fastest other runtime in
each column, `torch.compile` for 1 object and ComfyUI for 3, they are 1.22x and 1.05x.

At 1080p and 256 frames sam3rs takes 35.7 and 45.6 ms, 28 and 22 fps. The image encoder
and tracker neck, shared by all objects, are about 30 ms of each frame. Each object adds
4.7 ms, about 1.8 of them in the memory cross-attention kernel and 0.44 in the memory
self-attention. The GPU is about 99% busy, so CUDA graphs would save little. ComfyUI's
multiplex tracker costs the same for 1 to 16 objects.

```sh
cargo run --release --example bench_video -- --frames DIR --clip horses --objects 3
```

`--frames` is a directory of decoded frames, read in name order. Clicks come from
`tests/video_prompts.json` (`--prompts`). `--outputs DIR` writes the last run's
scores and masks for an A/B gate.

`scripts/video_reference.py bench` times the PyTorch rows. `scripts/bench_comfy.py
video-click horses-720p-64:1 horses-720p-64:3` times the ComfyUI row through a stock
ComfyUI server, with per-node times from `scripts/comfy_launch.py`.
`scripts/nsys_frames.py` turns an nsys capture of either into per-frame GPU busy
time, launches and stage times.

## Apple silicon

On an M4 Pro (20-core GPU, macOS 27), with the image protocol above: `dogs.jpg`, the
same prompts, full-resolution masks on the host. The official SAM 3 package runs on
PyTorch's MPS backend, patched to run there, in BF16 autocast (FP32 with the FP32 MLP
for its FP32 row). A click is the two-pass prediction the parity reference makes, and
box prompts go through the detector in every runtime. Milliseconds, best of two rounds
for the other runtimes:

| Runtime | Encode | Click | Box | Neg box | Text | Peak memory |
|---|---|---|---|---|---|---|
| sam3rs `bf16` | 824 | 6.2 | 93 | 92 | 103 | 4.7 GB |
| sam3rs `fp32` | 1071 | 6.2 | 95 | 95 | 105 | 5.7 GB |
| Official SAM 3, MPS BF16 | 982 | 18.1 | 127 | 128 | 125 | 7.3 GB |
| Official SAM 3, MPS FP32 | 1109 | 18.8 | 136 | 138 | 135 | 7.3 GB |
| osam, ONNX Runtime on Core ML | 1213 | 34.8 | 281 | 236 | 240 | 11.1 GB |

osam misses the parity gate on 32 of its 43 checks. The fastest other ports encode in
1.02-1.11 s (MLX) and answer text prompts in 111 ms (Hugging Face transformers on MPS,
with masks left at 288x288). Video tracking at 720p takes 931 ms a frame with one object
and 1194 ms with three in `bf16` (1161 and 1421 in `fp32`), most of it the image encoder.

### Where the time goes

- The image encoder is matrix math at the GPU's limit: in `bf16` its linears run at
  about 7.4 TFLOPS, close to the 7.55 that Metal 4's `matmul2d` reaches on this GPU,
  and the whole encode is within about 1.05x of the summed floors of its parts.
- Unmasked attention (the backbone, the fusion encoder's self-attention and the
  tracker's memory attention) runs a flash-attention loop on tensor ops with an FP32
  softmax. The tracker's 256-wide self-attention instead runs as a score GEMM, a row
  softmax and a second GEMM, about twice as fast here as the fused loop. Its memory
  cross-attention folds the value projection into the output projection, so it reads
  the 64-channel memory rather than 256-channel values, computes each score tile once
  for four SIMD groups and splits the keys across the GPU.
- The Neural Engine runs the backbone's matrix math at about the GPU's rate but its
  global attention five times slower, so the whole backbone takes over 1.3 s there.
  sam3rs doesn't use it.

```sh
cargo run --release --example bench -- --precision bf16
cargo run --release --example bench -- --precision fp32
```
