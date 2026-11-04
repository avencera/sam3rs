# sam3rs

SAM 3 image segmentation on NVIDIA GPUs and Apple silicon from Python, fast enough to
run live. Encode an image once, then find objects by noun phrase, box exemplars or
clicks. The model runs in a native Rust runtime ([sam3rs](https://github.com/avencera/sam3rs)),
on CUDA or Metal, with no PyTorch dependency.

On an RTX 5090 in BF16, against the official PyTorch implementation:

| Task | sam3rs | PyTorch | Speedup |
|---|---|---|---|
| Click to mask, 2560x1600 image | 1.5 ms | 36.2 ms | **24x** |
| Click to mask, 50 MP photo | 5.4 ms | 461 ms | **85x** |
| Segment everything (16x16 clicks), 50 MP photo | 1.46 s | 105 s | **72x** |
| Box prompt, 2560x1600 | 5.4 ms | 21.8 ms | **4.0x** |
| Text prompt, 2560x1600 | 6.9 ms | 20.7 ms | **3.0x** |
| Encode a new image | 30.9 ms | 39.4 ms | **1.28x** |

That is fast enough to run live. In frames a second:

| Live workload | sam3rs | PyTorch |
|---|---|---|
| Mask under a moving cursor, 50 MP photo | **120** | 2 |
| Mask under a moving cursor, 12 MP photo | **120** | 10 |
| 8 masks around the cursor on every move, 36 MP photo | **28** | 0.4 |
| Text prompt on every new 2560x1600 frame | **26** | 17 |

The cursor moves at 120 Hz, so 120 means sam3rs answers every position. A new frame
with a text prompt is an encode plus the prompt, 31 + 7 ms against 39 + 21 ms. Masks
arrive as numpy bool arrays without a copy. Methods and the full tables are in the
[sam3rs README](https://github.com/avencera/sam3rs#results).

## Install

```sh
pip install "sam3rs[cuda]"
```

On Linux x86_64 you need:

- an NVIDIA GPU with compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell)
- NVIDIA driver r580 or newer
- cuBLAS 13, which the `cuda` extra installs from the `nvidia-cublas` wheel. Any 13.x
  release works; it is tested with releases from 13.0.0 to 13.8.2. Leave the extra out
  if a CUDA 13 toolkit is already on the library path.

On macOS you need Apple silicon (M1 or newer) with macOS 26 or newer. The same command
works there: the `cuda` extra installs nothing on a Mac, so `pip install sam3rs` is
enough. `precision` takes `"bf16"` or `"fp32"`; `"tf32"` is a CUDA mode and raises
`sam3rs.Sam3rsError`. On an M4 Pro in BF16, encoding an image takes 824 ms, a click
6.2 ms, a box prompt 93 ms and a text prompt 103 ms, 1.19x to 2.9x faster than the
official SAM 3 package on PyTorch's MPS backend
([Apple silicon results](https://github.com/avencera/sam3rs#apple-silicon)).

The package imports on any machine. Loading a model without a usable GPU, driver or
cuBLAS, or on a macOS older than 26, raises `sam3rs.Sam3rsError` that says what is
missing.

## Weights

The weights come from the gated [facebook/sam3](https://huggingface.co/facebook/sam3)
repo. Accept Meta's terms on the model page while logged in, then give sam3rs a token
from that account, with `hf auth login` or the `HF_TOKEN` environment variable.
`Sam3.from_pretrained()` downloads the 3.4 GB checkpoint into the Hugging Face cache on
first use. `Sam3.load(path)` reads a directory that already holds `model.safetensors`,
`vocab.json` and `merges.txt`.

## Usage

```python
import sam3rs

model = sam3rs.Sam3.from_pretrained(precision="bf16")
embedding = model.encode("dogs.jpg")  # or an (height, width, 3) uint8 RGB array

dogs = model.detect(embedding, text="dog")
print(dogs.scores)  # (n,) float32
print(dogs.boxes)  # (n, 4) float32, x0 y0 x1 y1 in pixels
print(dogs.masks.shape)  # (n, height, width) bool

# box exemplars: find more things like the positive box, but not like the negative one
similar = model.detect(
    embedding,
    boxes=[[1233, 376, 1649, 691]],
    negative_boxes=[[1400, 500, 1500, 600]],
)

# clicks: label 1 is foreground, 0 background
click = model.segment_points(embedding, [(1439, 504), (1500, 600)], labels=[1, 0])
print(click.score, click.mask.sum())

# many objects in one call, one prompt per object: here three single clicks
objects = model.segment_points_batch(embedding, [(1439, 504), (700, 900), (2000, 1000)])
print(objects.scores)  # (3,) float32
print(objects.masks.shape)  # (3, height, width) bool

tinted = sam3rs.overlay(sam3rs.load_image("dogs.jpg"), dogs.masks)
```

An embedding works only with the model that made it, and costs nothing to reuse, so
encode once and run as many prompts as you need against it.

`segment_points_batch` takes `(k, 2)` single clicks, `(k, n, 2)` prompts of `n` clicks,
or a list of prompts whose click counts differ, with `labels` shaped to match. Each
object gets the score and mask `segment_points` would give its clicks alone (exactly
in FP32, and within the usual BF16 and TF32 rounding otherwise), with the decoder
batched on the GPU.

`precision="bf16"` (the default) runs the vision backbone, the text encoder's MLP and
the fusion encoder's feed-forward layers in BF16, every other matrix multiply on tensor
cores with BF16 or TF32 inputs, and most attention fused on FP16 tensor cores. `precision="tf32"` keeps FP32
weights and activations but runs the matrix multiplies on TF32 tensor cores, as ONNX
Runtime does for FP32 models, with most attention fused on TF32 tensor cores.
`precision="fp32"` is the strict FP32 reference path. On Metal, `bf16` runs the vision
backbone and both necks in BF16 and every later stage in FP32. On an RTX 5090, encoding an image
takes about 31 ms in BF16, 64 ms in TF32 and 118 ms in FP32; a point prompt 1.4 to
1.7 ms; a box prompt 5.4, 7.8 and 27 ms; and a text prompt 6.9, 9.2 and 29 ms.

Calls release the GIL while the GPU works. Calls on one model from several threads
take turns. Masks are writable numpy views of memory the runtime filled, not copies;
once they are garbage collected, the model reuses that memory for later masks.

## Command line

```sh
sam3rs dogs.jpg --text dog --out dogs.png
sam3rs dogs.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600
sam3rs dogs.jpg --point 1439,504 --neg-point 1500,600 --model ./sam3
```

Prints one JSON object per result. `--out` writes the image with the masks drawn on it.

To run the command without installing anything, `uvx` fetches the package and cuBLAS
(about 470 MB) into a cached environment on the first run:

```sh
uvx --from "sam3rs[cuda]" sam3rs dogs.jpg --text dog --out dogs.png
```

## License

sam3rs is licensed under the [Apache License
2.0](https://github.com/avencera/sam3rs/blob/master/LICENSE). It is an independent
implementation and contains no SAM 3 weights. Meta distributes the weights under the
[SAM License](https://huggingface.co/facebook/sam3), and each user accepts it on Hugging
Face before downloading them.
