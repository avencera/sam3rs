# Changelog

All notable changes to sam3rs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The README is shorter. Benchmark protocol, per-stage timings, the large-photo and
  video tables and the reproduction commands moved to `docs/benchmarks.md`, the parity
  gates and test commands to `docs/parity.md`, and the per-mode precision details to
  the `Precision` docs.

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

[Unreleased]: https://github.com/avencera/sam3rs/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/avencera/sam3rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/avencera/sam3rs/releases/tag/v0.1.0
