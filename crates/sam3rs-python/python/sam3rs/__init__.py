"""SAM 3 image segmentation on NVIDIA GPUs, backed by the sam3rs Rust/CUDA runtime.

Encode an image once, then answer any number of text, box and point prompts::

    import sam3rs

    model = sam3rs.Sam3.from_pretrained()
    embedding = model.encode("dogs.jpg")
    dogs = model.detect(embedding, text="dog")
    print(dogs.scores, dogs.boxes, dogs.masks.shape)

    click = model.segment_points(embedding, [(1439, 504)])
    print(click.score, click.mask.sum())
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import numpy.typing as npt

from . import _native
from ._cuda import prepare_libraries
from ._errors import Sam3rsError

__all__ = [
    "MODEL_REPO",
    "SCORE_THRESHOLD",
    "Detections",
    "Embedding",
    "PointMask",
    "Precision",
    "Sam3",
    "Sam3rsError",
    "__version__",
    "load_image",
    "overlay",
]

__version__: str = _native.__version__

#: Hugging Face repo :meth:`Sam3.from_pretrained` downloads; it is gated, so accept
#: Meta's terms on its model page first
MODEL_REPO: str = _native.MODEL_REPO
#: Detections scoring at or below this are dropped, as in upstream ``Sam3Processor``
SCORE_THRESHOLD: float = _native.SCORE_THRESHOLD

Precision = Literal["bf16", "tf32", "fp32"]
#: An ``(height, width, 3)`` uint8 RGB array, or a path to a PNG or JPEG
ImageInput = Union[npt.NDArray[np.uint8], str, "os.PathLike[str]"]
Embedding = _native.Embedding


@dataclass(frozen=True)
class Detections:
    """Instances found by :meth:`Sam3.detect`, in DETR query order."""

    #: ``(n,)`` float32, sigmoid(logit) * sigmoid(presence)
    scores: npt.NDArray[np.float32]
    #: ``(n, 4)`` float32 ``[x0, y0, x1, y1]`` in pixels, not clamped to the image
    boxes: npt.NDArray[np.float32]
    #: ``(n, height, width)`` bool
    masks: npt.NDArray[np.bool_]
    #: ``(n,)`` int64 index of the DETR query (0 to 199) behind each instance
    queries: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.scores)


@dataclass(frozen=True)
class PointMask:
    """The best of the three candidate masks for a point prompt."""

    #: Predicted IoU of the chosen mask
    score: float
    #: ``(height, width)`` bool
    mask: npt.NDArray[np.bool_]


class Sam3:
    """The SAM 3 model loaded on one CUDA GPU.

    Needs an NVIDIA GPU with compute capability 8.0 or newer, driver r580 or newer,
    and cuBLAS 13 (``pip install "sam3rs[cuda]"`` or a CUDA 13 toolkit). Calls from
    several threads take turns; each releases the GIL while the GPU works.
    """

    def __init__(self, native: _native.Model) -> None:
        self._native = native

    @classmethod
    def from_pretrained(cls, device: int = 0, precision: Precision = "bf16") -> Sam3:
        """Download the gated ``facebook/sam3`` checkpoint and load it.

        Accept Meta's SAM License at https://huggingface.co/facebook/sam3, then log
        in with ``hf auth login`` or set ``HF_TOKEN``. Files land in the standard
        Hugging Face cache, so later calls skip the 3.4 GB download.
        """
        prepare_libraries()
        with _cuda_hint():
            return cls(_native.Model.from_pretrained(device, precision))

    @classmethod
    def load(cls, path: str | os.PathLike[str], device: int = 0, precision: Precision = "bf16") -> Sam3:
        """Load ``model.safetensors``, ``vocab.json`` and ``merges.txt`` from a directory.

        ``bf16`` runs the vision backbone on BF16 tensor cores with fused attention;
        ``fp32`` is the strict FP32 reference path, about 3.5x slower to encode; ``tf32``
        is the FP32 path with TF32 tensor-core GEMMs, as ONNX Runtime runs FP32 models.
        """
        prepare_libraries()
        with _cuda_hint():
            return cls(_native.Model.load(os.fspath(path), device, precision))

    @property
    def precision(self) -> Precision:
        """Numeric mode the model was loaded with."""
        return self._native.precision  # type: ignore[return-value]

    def encode(self, image: ImageInput) -> Embedding:
        """Run the image encoder; the embedding serves any number of prompts."""
        if isinstance(image, (str, os.PathLike)):
            return self._native.encode_path(os.fspath(image))
        rgb = _rgb(image)
        return self._native.encode(rgb.shape[1], rgb.shape[0], rgb.tobytes())

    def detect(
        self,
        embedding: Embedding,
        text: str | None = None,
        boxes: npt.ArrayLike | None = None,
        negative_boxes: npt.ArrayLike | None = None,
    ) -> Detections:
        """Find every instance matching a noun phrase, box exemplars, or both.

        Boxes are ``[x0, y0, x1, y1]`` in pixels of the encoded image. Positive boxes
        show what to find, negative boxes what to leave out.
        """
        exemplars = [(x0, y0, x1, y1, True) for x0, y0, x1, y1 in _rows(boxes, 4, "boxes")]
        exemplars += [(x0, y0, x1, y1, False) for x0, y0, x1, y1 in _rows(negative_boxes, 4, "negative_boxes")]
        if text is None and not exemplars:
            raise ValueError("give text, boxes or negative_boxes")
        prompt = _native.VISUAL_PROMPT if text is None else text
        scores, bboxes, masks, queries = self._native.detect(embedding, prompt, exemplars)
        n, h, w = len(scores), embedding.height, embedding.width
        return Detections(
            scores=np.asarray(scores, dtype=np.float32),
            boxes=np.asarray(bboxes, dtype=np.float32).reshape(n, 4),
            masks=_bool_view(masks, n, h, w),
            queries=np.asarray(queries, dtype=np.int64),
        )

    def segment_points(
        self,
        embedding: Embedding,
        points: npt.ArrayLike,
        labels: npt.ArrayLike | None = None,
    ) -> PointMask:
        """Segment the object under one or more clicks.

        ``points`` are ``(x, y)`` pixels; ``labels`` marks each as foreground (1) or
        background (0), and defaults to all foreground.
        """
        xy = _rows(points, 2, "points")
        if not xy:
            raise ValueError("give at least one point")
        if labels is None:
            flags = [True] * len(xy)
        else:
            values = np.asarray(labels).reshape(-1)
            # SAM's -1 padding label would otherwise read as foreground
            if not np.isin(values, (0, 1)).all():
                raise ValueError(f"labels must be 0 (background) or 1 (foreground), got {values.tolist()}")
            flags = [bool(v) for v in values]
        if len(flags) != len(xy):
            raise ValueError(f"{len(xy)} points but {len(flags)} labels")
        score, mask = self._native.segment_points(embedding, [(x, y, f) for (x, y), f in zip(xy, flags)])
        h, w = embedding.height, embedding.width
        return PointMask(score=score, mask=_bool_view(mask, 1, h, w)[0])


def load_image(path: str | os.PathLike[str]) -> npt.NDArray[np.uint8]:
    """Decode a PNG or JPEG into an ``(height, width, 3)`` uint8 RGB array."""
    width, height, rgb = _native.open_image(os.fspath(path))
    return np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3).copy()


def overlay(image: npt.ArrayLike, masks: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Tint each ``(height, width)`` mask over an RGB image in its own color."""
    rgb = _rgb(image)
    stack = np.asarray(masks, dtype=bool)
    if stack.ndim == 2:
        stack = stack[None]
    if stack.ndim != 3 or stack.shape[1:] != rgb.shape[:2]:
        raise ValueError(f"masks of shape {stack.shape} don't match an image of shape {rgb.shape}")
    h, w = rgb.shape[:2]
    out = _native.overlay(w, h, rgb.tobytes(), stack.astype(np.uint8).tobytes())
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3).copy()


@contextmanager
def _cuda_hint() -> Iterator[None]:
    try:
        yield
    except Sam3rsError as e:
        if "cuBLAS" in str(e):
            e.add_note('with pip, install the cuda extra: pip install "sam3rs[cuda]"')
        raise


def _bool_view(buffer: _native.HostBuffer, n: int, h: int, w: int) -> npt.NDArray[np.bool_]:
    # the runtime writes only 0 and 1, which are valid numpy bools, so the masks are
    # viewed in place: a converted copy is another fresh array of n full-size masks
    return np.frombuffer(buffer, dtype=np.bool_).reshape(n, h, w)


def _rgb(image: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (height, width, 3) RGB array, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected a uint8 RGB array, got {rgb.dtype}")
    return np.ascontiguousarray(rgb)


def _rows(values: npt.ArrayLike | None, width: int, name: str) -> list[tuple[float, ...]]:
    if values is None:
        return []
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be rows of {width} numbers, got shape {array.shape}")
    return [tuple(float(v) for v in row) for row in array]
