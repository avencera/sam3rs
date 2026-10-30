"""``sam3rs`` command: segment an image with a text, box or point prompt.

    sam3rs image.jpg --text dog --out dogs.png
    sam3rs image.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600
    sam3rs image.jpg --point 1439,504 --neg-point 1500,600

Prints one JSON object per result. Without ``--model``, the weights come from the
gated facebook/sam3 Hugging Face repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import Sam3, Sam3rsError, _native, load_image, overlay


def _coords(n: int):
    def parse(text: str) -> list[float]:
        try:
            values = [float(v) for v in text.split(",")]
        except ValueError:
            values = []
        if len(values) != n:
            raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got {text!r}")
        return values

    return parse


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sam3rs", description="SAM 3 image segmentation on CUDA")
    p.add_argument("image", type=Path)
    p.add_argument("--text", help="noun phrase to detect, alone or with boxes")
    p.add_argument(
        "--box",
        dest="boxes",
        type=_coords(4),
        action="append",
        default=[],
        help="positive box exemplar x0,y0,x1,y1 in pixels",
    )
    p.add_argument(
        "--neg-box",
        dest="neg_boxes",
        type=_coords(4),
        action="append",
        default=[],
        help="negative box exemplar x0,y0,x1,y1",
    )
    p.add_argument(
        "--point",
        dest="points",
        type=_coords(2),
        action="append",
        default=[],
        help="foreground click x,y; points can't be combined with text or boxes",
    )
    p.add_argument(
        "--neg-point", dest="neg_points", type=_coords(2), action="append", default=[], help="background click x,y"
    )
    p.add_argument("--out", type=Path, help="PNG with the masks drawn over the image")
    p.add_argument(
        "--model",
        type=Path,
        help="directory with model.safetensors, vocab.json and merges.txt "
        "[default: download facebook/sam3 from Hugging Face]",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--precision",
        choices=["bf16", "tf32", "fp32"],
        default="bf16",
        help="fp32 (strict reference path), tf32 (FP32 with TF32 tensor-core GEMMs) or bf16 (BF16 backbone, fused attention)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    has_points = bool(args.points or args.neg_points)
    has_boxes = bool(args.boxes or args.neg_boxes)
    if has_points and (has_boxes or args.text is not None):
        parser.error("point prompts can't be combined with text or boxes")
    if not (has_points or has_boxes or args.text is not None):
        parser.error("give --text, --box or --point")

    try:
        image = load_image(args.image)
        if args.model is None:
            model = Sam3.from_pretrained(args.device, args.precision)
        else:
            model = Sam3.load(args.model, args.device, args.precision)
        emb = model.encode(image)
        if has_points:
            points = args.points + args.neg_points
            labels = [1] * len(args.points) + [0] * len(args.neg_points)
            m = model.segment_points(emb, points, labels)
            results = [{"score": m.score}]
            masks = m.mask[None]
        else:
            dets = model.detect(emb, args.text, args.boxes or None, args.neg_boxes or None)
            results = [{"score": float(s), "bbox_xyxy": b.tolist()} for s, b in zip(dets.scores, dets.boxes)]
            masks = dets.masks
        for r in results:
            print(json.dumps(r))
        if args.out is not None:
            out = overlay(image, masks) if len(masks) else image
            h, w = out.shape[:2]
            _native.save_image(w, h, np.ascontiguousarray(out).tobytes(), str(args.out))
            print(f"wrote {args.out}", file=sys.stderr)
    except Sam3rsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
