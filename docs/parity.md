# Parity

sam3rs is checked against PyTorch FP32 outputs of the official implementation, in every
precision mode of each backend (three on CUDA, two on Metal). The summary tables are in the [README](../README.md#parity).

## Images

`tests/parity.rs` runs four images and 13 prompts (text, box, negative box, points)
through each mode of the backend. The fixtures come from `scripts/reference.py dump`. The gate: the
same kept DETR queries, scores within 0.01, box corners within 0.5% of the image's
longer side, and mask IoU of at least 0.99.

```sh
cargo test --release --test parity -- --nocapture
```

The ignored tests in `src/stage_tests.rs` compare every intermediate stage for one
fixture, each from the reference's output of the stage before: `stages` the pixels,
trunk and FPN levels, `point_stages` the point decoder's two passes and masks, and
`detector_stages` the text features, scores, boxes and mask logits. They print a report
rather than gate:

```sh
SAM3RS_STAGES=models/parity/dogs.safetensors \
    cargo test --release --lib stage_tests:: -- --ignored --nocapture --test-threads 1
```

## Video

The video tracker's gates are ignored tests in `src/video_tests.rs`, against fixtures
from `scripts/video_reference.py dump`:
upstream's tracker in true FP32 on a clip, with every stage's inputs and outputs on a
few frames.

- `video_stages` runs each stage (memory attention, memory encoder, decoder, pointer
  MLP) on the reference's exact inputs: rel L2 within 1e-4 in `fp32`, and 4e-2 for the
  backbone and neck (the image path's encoder) in the reduced modes. On CUDA every later
  stage is held to 5e-3 in `tf32` and `bf16`.
- `video_e2e` tracks the whole clip and needs the reference's memory length, pointer
  count and object presence on every frame, plus a mask IoU per object against FP32:
  at least 0.99 in `fp32`, and in the reduced modes no worse than PyTorch's own BF16
  run (mean minus 0.01, and its worst frame). Up to two frames an object are excused
  from the worst-frame limit where the decoder's top two candidates were within 0.005
  predicted IoU.
- `video_e2e` in `bf16` on CUDA also runs each checked frame's memory attention over
  the cached per-frame keys against the FP32 model's on the same inputs (rel L2 within
  5e-3).
- `cuda::kernel_tests::attention_d256` checks the two fused attention kernels (values of 256 for
  self-attention, of 64 for the memory cross-attention) against an f64 softmax: rel L2
  within 1e-3.

Every backend runs the same stage and video suites. Each gives them its stages through
its `test_hooks` module (the `TestHooks` trait in `src/test_support.rs`), with its own
limits per precision and any checks of its own. On Metal, every stage after the necks
runs in FP32, so `video_stages` holds those stages to the `fp32` limits in both modes.
The cached-key check and `attention_d256` are CUDA only; on Macs the ignored `metal::kernel_tests` check every attention kernel (all masks,
ragged lengths, split key ranges) and the linears against f64. Metal's `bf16` misses the
one-object clip's gate: two candidate masks on the prompted frame score within 0.001 of
each other and BF16 rounding picks the other one, so the object is then tracked at mean
IoU 0.987 against FP32.

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
