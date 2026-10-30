"""API tests: argument checks and failures run anywhere, parity needs a GPU and weights.

The parity subset repeats the Rust gate in tests/parity.rs on the dogs fixture
from ``scripts/reference.py dump``: the same kept queries, scores within 0.01,
boxes within 0.5% of the longer side and mask IoU of at least 0.99.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sam3rs
from conftest import PARITY_DIR, read_safetensors

SCORE_TOL = 0.01
MIN_IOU = 0.99
BOX_TOL = 0.005


def test_imports_without_gpu() -> None:
    assert sam3rs.__version__
    assert sam3rs.MODEL_REPO == "facebook/sam3"
    assert 0 < sam3rs.SCORE_THRESHOLD < 1


def test_load_fails_cleanly_without_gpu(tmp_path: Path) -> None:
    # hiding every device stands in for a machine without one; on a CI runner without
    # a driver the same call fails on the missing libcuda instead
    code = (
        "import sam3rs, sys\n"
        "try:\n"
        f"    sam3rs.Sam3.load({str(tmp_path)!r})\n"
        "except sam3rs.Sam3rsError as e:\n"
        "    print(e)\n"
        "    sys.exit(0)\n"
        "sys.exit('loaded without a GPU')\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", code], check=False, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    message = out.stdout.lower()
    assert "driver" in message or "device" in message, message
    # the GPU is checked before the model files, so a GPU-less machine never reads them
    assert "vocab" not in message, message


def test_cli_rejects_mixed_prompts(capsys: pytest.CaptureFixture[str]) -> None:
    from sam3rs.cli import main

    with pytest.raises(SystemExit):
        main(["image.jpg", "--text", "dog", "--point", "1,2"])
    assert "can't be combined" in capsys.readouterr().err


def test_overlay_checks_mask_size() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 2] = True
    out = sam3rs.overlay(image, mask)
    assert out.shape == image.shape
    assert out[1, 2].any() and not out[0, 0].any()
    with pytest.raises(ValueError):
        sam3rs.overlay(image, np.zeros((6, 4), dtype=bool))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


@pytest.fixture(scope="module")
def dogs() -> tuple[dict[str, np.ndarray], list[dict]]:
    path = PARITY_DIR / "dogs.safetensors"
    if not path.is_file():
        pytest.skip(f"no parity fixture at {path}")
    tensors, meta = read_safetensors(path)
    return tensors, json.loads(meta["meta"])["prompts"]


@pytest.mark.gpu
def test_matches_pytorch(model: sam3rs.Sam3, dogs: tuple[dict[str, np.ndarray], list[dict]]) -> None:
    tensors, prompts = dogs
    image = tensors["image_rgb"]
    height, width = image.shape[:2]
    emb = model.encode(image)
    assert (emb.width, emb.height) == (width, height)

    checked = 0
    for prompt in prompts:
        key = prompt["name"].replace(" ", "_")
        masks = tensors[f"{key}.masks"]
        if "points" in prompt:
            points = [p[:2] for p in prompt["points"]]
            labels = [p[2] for p in prompt["points"]]
            got = model.segment_points(emb, points, labels)
            want_iou = tensors[f"{key}.iou"]
            best = int(np.argmax(want_iou))
            assert abs(got.score - want_iou[best]) <= SCORE_TOL, key
            assert _iou(got.mask, masks[best] != 0) >= MIN_IOU, key
            checked += 1
            continue

        boxes = prompt.get("boxes", [])
        got = model.detect(
            emb,
            text=prompt.get("text"),
            boxes=[b[:4] for b in boxes if b[4]] or None,
            negative_boxes=[b[:4] for b in boxes if not b[4]] or None,
        )
        want = np.flatnonzero(tensors[f"{key}.scores_all"] > sam3rs.SCORE_THRESHOLD)
        np.testing.assert_array_equal(got.queries, want, err_msg=key)
        np.testing.assert_allclose(got.scores, tensors[f"{key}.scores"], atol=SCORE_TOL, err_msg=key)
        box_tol = BOX_TOL * max(width, height)
        np.testing.assert_allclose(got.boxes, tensors[f"{key}.boxes_xyxy"], atol=box_tol, err_msg=key)
        for i, mask in enumerate(got.masks):
            assert _iou(mask, masks[i] != 0) >= MIN_IOU, f"{key} det {i}"
        checked += 1
    assert checked == len(prompts) > 0


@pytest.mark.gpu
def test_encode_path_matches_array(model: sam3rs.Sam3, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    path = tmp_path / "noise.png"
    sam3rs._native.save_image(64, 48, image.tobytes(), str(path))
    np.testing.assert_array_equal(sam3rs.load_image(path), image)
    a = model.detect(model.encode(image), text="cat")
    b = model.detect(model.encode(path), text="cat")
    np.testing.assert_array_equal(a.queries, b.queries)


@pytest.mark.gpu
def test_rejects_bad_prompts(model: sam3rs.Sam3) -> None:
    emb = model.encode(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        model.detect(emb)
    with pytest.raises(sam3rs.Sam3rsError):
        model.detect(emb, boxes=[[0, 0, float("nan"), 10]])
    with pytest.raises(ValueError):
        model.segment_points(emb, [])
    with pytest.raises(ValueError):
        model.segment_points(emb, [(1, 2)], labels=[1, 0])
