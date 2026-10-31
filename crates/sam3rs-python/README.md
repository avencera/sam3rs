# sam3rs

SAM 3 image segmentation on NVIDIA GPUs, from Python. Encode an image once, then find
objects by noun phrase, box exemplars or clicks. The model runs in a native Rust/CUDA
runtime ([sam3rs](https://github.com/avencera/sam3rs)) with no PyTorch dependency.

## Install

```sh
pip install "sam3rs[cuda]"
```

Linux x86_64 only. You need:

- an NVIDIA GPU with compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell)
- NVIDIA driver r580 or newer
- cuBLAS 13, which the `cuda` extra installs from the `nvidia-cublas` wheel. Leave the
  extra out if a CUDA 13 toolkit is already on the library path.

The package imports on any machine. Loading a model without a usable GPU, driver or
cuBLAS raises `sam3rs.Sam3rsError` that says what is missing.

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

tinted = sam3rs.overlay(sam3rs.load_image("dogs.jpg"), dogs.masks)
```

An embedding works only with the model that made it, and costs nothing to reuse, so
encode once and run as many prompts as you need against it.

`precision="bf16"` (the default) runs the vision backbone, the text encoder's MLP and
the fusion encoder's feed-forward layers in BF16 with fused attention, and every other
matrix multiply on tensor cores with BF16 or TF32 inputs. `precision="tf32"` keeps FP32
weights and activations but runs the matrix multiplies on TF32 tensor cores, as ONNX
Runtime does for FP32 models, with most attention fused on TF32 tensor cores.
`precision="fp32"` is the strict FP32 reference path. On an RTX 5090, encoding an image
takes about 33 ms in BF16, 64 ms in TF32 and 143 ms in FP32; a point prompt 1.9 to
2.3 ms; a box prompt 5.5, 8.0 and 27 ms; and a text prompt 7.0, 9.3 and 30 ms.

Calls release the GIL while the GPU works. Calls on one model from several threads
take turns.

## Command line

```sh
sam3rs dogs.jpg --text dog --out dogs.png
sam3rs dogs.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600
sam3rs dogs.jpg --point 1439,504 --neg-point 1500,600 --model ./sam3
```

Prints one JSON object per result. `--out` writes the image with the masks drawn on it.

## License

TODO(license): pick the license before release.
