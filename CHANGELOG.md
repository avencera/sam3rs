# Changelog

All notable changes to sam3rs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-04

### Added

- Apple silicon support: on macOS, sam3rs builds a Metal backend with the same API, every
  prompt type and the video tracker, in `bf16` and `fp32` (`tf32` returns an error). On
  an M4 Pro it encodes an image in 824 ms, answers a click in 6.2 ms, a box in 93 ms and
  a text prompt in 103 ms, 1.19-2.9x faster than the official SAM 3 package on PyTorch's
  MPS backend, and tracks three objects through 720p video at 1.2 s a frame. It needs
  Apple silicon and macOS 26.1 or newer.
- macOS release artifacts for Apple silicon: a Python wheel, so `pip install sam3rs`
  and `uvx sam3rs` work there, and a prebuilt `sam3rs` command that the install script
  now picks on an Apple silicon Mac.

### Changed

- The README is shorter. Benchmark protocol, per-stage timings, the large-photo and
  video tables and the reproduction commands moved to `docs/benchmarks.md`, the parity
  gates and test commands to `docs/parity.md`, and the per-mode precision details to
  the `Precision` docs.
- The crate and PyPI descriptions name Apple silicon next to NVIDIA GPUs.

## [0.1.1] - 2026-11-03

### Changed

- The README and the Python package README open with the speedups over PyTorch and the
  workloads that run live, in frames a second: a mask under a 120 Hz cursor at 120 masks
  a second on a 50 MP photo, where PyTorch draws 2; a text prompt on every new 2560x1600
  frame at 26 frames a second; and video tracking at 29 frames a second at 720p.
- New large-photo results: a click is 4.9x faster than PyTorch at 2 MP and 93x faster at
  108 MP, and segmenting everything in a 50 MP photo takes 1.46 s against 105 s.
- The video results show frames a second next to each time.
- The crate and PyPI descriptions say what sam3rs is for: real-time SAM 3 on NVIDIA GPUs.

## [0.1.0] - 2026-11-03

First release.

### Added

- Native Rust/CUDA SAM 3 runtime that loads the official Hugging Face checkpoint and runs
  without PyTorch or ONNX Runtime, in `bf16`, `tf32` and strict `fp32`.
- Image prompts: text, box exemplars with negative boxes, and clicks, with image features
  kept on the GPU between prompts.
- `segment_points_batch`, which segments many objects from their clicks in one call.
- Click-to-track video with the SAM 3 tracker and a fixed-size memory bank.
- The `sam3rs` command, as a prebuilt Linux x86_64 binary with an install script, and
  through `cargo install sam3rs --features cli`.
- The `sam3rs` Python package on PyPI, with numpy in and out, the same command, and
  cuBLAS from the `cuda` extra.

[Unreleased]: https://github.com/avencera/sam3rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/avencera/sam3rs/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/avencera/sam3rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/avencera/sam3rs/releases/tag/v0.1.0
