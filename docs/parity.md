# Parity

sam3rs is checked against PyTorch FP32 outputs of the official implementation, in all
three precision modes. The summary tables are in the [README](../README.md#parity).

## Images

`tests/parity.rs` runs four images and 13 prompts (text, box, negative box, points)
through each mode. The fixtures come from `scripts/reference.py dump`. The gate: the
same kept DETR queries, scores within 0.01, box corners within 0.5% of the image's
longer side, and mask IoU of at least 0.99.

```sh
cargo test --release --test parity -- --nocapture
```

The ignored `stage_tests::stages` unit test (`cargo test --release --lib -- --ignored`)
compares every intermediate stage (pixels, trunk, FPN levels, text features, scores,
boxes, mask logits) for one fixture.

## Video

The video tracker's gates are ignored tests in `src/cuda/video_tests.rs`, against fixtures
from `scripts/video_reference.py dump`: upstream's tracker in true FP32 on a clip, with
every stage's inputs and outputs on a few frames.

- `video_stages` runs each stage (memory attention, memory encoder, decoder, pointer
  MLP) on the reference's exact inputs: rel L2 within 1e-4 in `fp32` and 5e-3 in
  `tf32` and `bf16` (4e-2 for the backbone and neck, the image path's encoder).
- `video_e2e` tracks the whole clip and needs the reference's memory length, pointer
  count and object presence on every frame, plus a mask IoU per object against FP32:
  at least 0.99 in `fp32`, and in the reduced modes no worse than PyTorch's own BF16
  run (mean minus 0.01, and its worst frame). Up to two frames an object are excused
  from the worst-frame limit where the decoder's top two candidates were within 0.005
  predicted IoU.
- `video_e2e` in `bf16` also runs each checked frame's memory attention over the cached
  per-frame keys against the FP32 model's on the same inputs (rel L2 within 5e-3).
- `attention_d256` checks the two fused attention kernels (values of 256 for
  self-attention, of 64 for the memory cross-attention) against an f64 softmax: rel L2
  within 1e-3.

```sh
SAM3RS_VIDEO=models/video-parity/horses-720p-64-o3-fp32.safetensors SAM3RS_PRECISION=bf16 \
    cargo test --release --lib video_tests -- --ignored --nocapture --test-threads 1
```

## cuBLAS versions

`scripts/check-cublas-versions.sh` runs the Python package's tests on a GPU against the
oldest cuBLAS 13 wheel, the one PyTorch's cu130 wheels pin and the newest.

## Known differences in detail

- JPEG decoding uses the `image` crate, which differs from PIL's libjpeg-turbo on a few
  percent of bytes. Scores move by up to a few thousandths. The parity test feeds the
  reference's decoded RGB to isolate the model.
- Text cleaning lowercases and collapses whitespace but doesn't port `ftfy` or HTML
  unescaping.
- Video tracking follows SAM 3's tracker-only path ("SAM 3 for SAM 2 video task")
  from clicks on the first frame. The session predictors' per-frame detector, text
  prompts on video, and SAM 3.1 aren't implemented.
- In `bf16`, upstream rounds the memory's sine position encoding to BF16 under
  autocast, and sam3rs keeps it in FP32.
- The lowest `bf16` mask IoU (0.9914, a point prompt) comes from hole filling. A
  rounding difference flips one low-resolution pixel, which cuts a 64-pixel background
  region off from the rest. Hole filling turns background regions of up to 256 pixels
  into foreground, so that region becomes part of the mask. The score is unaffected.
