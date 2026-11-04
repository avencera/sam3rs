"""Shared fixtures: the local model directory and the PyTorch parity fixtures."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import sam3rs

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("SAM3RS_MODEL_DIR", REPO_ROOT / "models" / "sam3"))
PARITY_DIR = REPO_ROOT / "models" / "parity"

_DTYPES = {"F32": np.float32, "U8": np.uint8, "I64": np.int64}


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Tensors and string metadata of a safetensors file, without the safetensors package."""
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    meta = header.pop("__metadata__", {})
    body = memoryview(data)[8 + size :]
    tensors = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        tensors[name] = np.frombuffer(body[start:end], dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])
    return tensors, meta


def _gpu_ready() -> str | None:
    if not (MODEL_DIR / "model.safetensors").is_file():
        return f"no weights in {MODEL_DIR}"
    return None


# tf32 names a CUDA tensor-core mode, which the Metal backend rejects
PRECISIONS = ["fp32", "bf16"] if sys.platform == "darwin" else ["fp32", "tf32", "bf16"]


@pytest.fixture(scope="session", params=PRECISIONS)
def model(request: pytest.FixtureRequest) -> sam3rs.Sam3:
    """The local checkpoint in each precision the backend runs; skips without weights."""
    reason = _gpu_ready()
    if reason:
        pytest.skip(reason)
    return sam3rs.Sam3.load(MODEL_DIR, precision=request.param)
