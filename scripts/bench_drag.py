"""Time box prompts through the Python package as a live box drag sends them.

    python -I scripts/bench_drag.py IMAGE --box X0,Y0,X1,Y1 [--side 2560] [--limit-ms 15]
    python -I scripts/bench_drag.py IMAGE --box X0,Y0,X1,Y1 --repeat 21
    python -I scripts/bench_drag.py IMAGE --text dog --repeat 21

The box is in fractions of the image size. By default the box grows from its top-left
corner in --steps calls (from 30% to 100% of its size), and the drag runs --passes times
on one embedding. --repeat N instead times the whole box N times and reports the median,
and --text times that text prompt N times.
Each time is one `Sam3.detect` call, until the masks are bool arrays on the host.

The image is resized with PIL's LANCZOS filter so its long side is --side pixels, as
the side-by-side demo does. Its photos make the cases: breakfast-rouen.jpg with box
0.615,0.17,0.755,0.44 returns 5 or 6 masks late in the drag, and market-africa.jpg with
box 0.365,0.56,0.54,0.81 returns 17. At --side 3840 (their full size), --text dog on
dogs-marymoor.jpg returns several dogs

Each call also reports the kernel's direct-compaction stalls (compact_stall in
/proc/vmstat) taken while it ran, which separate memory-management stalls from GPU time.
With --limit-ms, exits 1 if any timed call takes longer
"""

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import PIL.Image

import sam3rs

ROOT = pathlib.Path(__file__).resolve().parent.parent


def compaction_stalls() -> int:
    try:
        with open("/proc/vmstat") as f:
            return next(int(line.split()[1]) for line in f if line.startswith("compact_stall "))
    except (OSError, StopIteration):
        return 0


def load(path: pathlib.Path, side: int) -> np.ndarray:
    image = PIL.Image.open(path).convert("RGB")
    scale = side / max(image.size)
    if scale < 1:
        size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
        image = image.resize(size, PIL.Image.Resampling.LANCZOS)
    return np.asarray(image)


def drag(box: list[float], steps: int) -> list[list[float]]:
    x0, y0, x1, y1 = box
    fractions = [0.3 + 0.7 * i / steps for i in range(1, steps + 1)]
    return [[x0, y0, round(x0 + (x1 - x0) * f, 5), round(y0 + (y1 - y0) * f, 5)] for f in fractions]


def timed(model: sam3rs.Sam3, emb: sam3rs.Embedding, box: list[float] | str) -> tuple[float, int, int]:
    w, h = emb.width, emb.height
    stalls = compaction_stalls()
    start = time.perf_counter()
    if isinstance(box, str):
        found = model.detect(emb, text=box)
    else:
        found = model.detect(emb, boxes=[[box[0] * w, box[1] * h, box[2] * w, box[3] * h]])
    ms = (time.perf_counter() - start) * 1000
    return ms, len(found), compaction_stalls() - stalls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=pathlib.Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--box", help="x0,y0,x1,y1 as fractions of the image size")
    prompt.add_argument("--text", help="time this text prompt; needs --repeat")
    parser.add_argument("--side", type=int, default=2560, help="long side to resize to")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=0, help="time the whole box this many times instead")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--model", type=pathlib.Path, default=ROOT / "models" / "sam3")
    parser.add_argument("--limit-ms", type=float, help="fail if any call takes longer")
    args = parser.parse_args()

    if args.text is not None and not args.repeat:
        parser.error("--text needs --repeat")
    box = args.text if args.text is not None else [float(v) for v in args.box.split(",")]
    rgb = load(args.image, args.side)
    model = sam3rs.Sam3.load(args.model, precision=args.precision)
    emb = model.encode(rgb)
    print(f"sam3rs {sam3rs.__version__} {model.precision} {args.image.name} {emb.width}x{emb.height} prompt {box}")
    # first calls pay one-time costs (the "visual" text features, kernel loads)
    for _ in range(3):
        model.detect(emb, boxes=[[emb.width * 0.4, emb.height * 0.4, emb.width * 0.6, emb.height * 0.6]])

    calls = []
    if args.repeat:
        calls = [timed(model, emb, box) for _ in range(args.repeat)]
        ms = [c[0] for c in calls]
        print(
            f"whole prompt x{args.repeat}: median {statistics.median(ms):.2f} ms, max {max(ms):.2f} ms, masks {calls[0][1]}"
        )
    else:
        boxes = drag(box, args.steps)
        for p in range(args.passes):
            rows = [timed(model, emb, b) for b in boxes]
            calls += rows
            worst = max(rows)
            print(f"pass {p}: max {worst[0]:.1f} ms ({worst[1]} masks), compaction stalls {sum(r[2] for r in rows)}")
            print("  " + " ".join(f"{ms:.1f}/{n}" for ms, n, _ in rows))

    by_count: dict[int, list[float]] = {}
    for ms, n, _ in calls:
        by_count.setdefault(n, []).append(ms)
    for n in sorted(by_count):
        ms = by_count[n]
        print(f"{n:3d} masks: {len(ms):3d} calls, median {statistics.median(ms):6.2f} ms, max {max(ms):6.2f} ms")
    if args.limit_ms is not None:
        over = [c for c in calls if c[0] > args.limit_ms]
        if over:
            print(f"FAIL: {len(over)} of {len(calls)} calls over {args.limit_ms} ms")
            sys.exit(1)
        print(f"OK: every call within {args.limit_ms} ms")


if __name__ == "__main__":
    main()
