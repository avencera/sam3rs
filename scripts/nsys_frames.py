"""Per-frame breakdown of an nsys capture taken with `video_reference.py bench --nvtx`.

    python -I scripts/nsys_frames.py profile.sqlite [--skip 8] [--json out.json]

Reads the sqlite export (`nsys export --type sqlite`, or the file `nsys stats` leaves
next to the report). For every `frame N` NVTX range it reports wall time, GPU-busy share
(union of kernel, memcpy and memset intervals over the range), kernels run, launch API
calls (runtime and driver, so Triton's cuLaunchKernel counts), CUDA graph launches, time
the host spent blocked in synchronizing calls (stream, event and device syncs, and
synchronous or device-to-host copies), and memcpy traffic by direction. Kernels are
assigned to the innermost stage range (`backbone`, `mem_attn`, ...) open when their launch
call ran, so stage GPU time follows the host code that issued it. The first `--skip`
frames are left out of the summary; they hold the prompt and the memory-bank warm-up.
`--range "node SAM3_VideoTrack" --frames 256` instead takes each range with that label as
one window and divides counts by the frame count, for runtimes without per-frame ranges
"""

import argparse
import bisect
import json
import re
import sqlite3
import statistics
from collections import defaultdict

COPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP", 9: "HtoH"}
LAUNCH = re.compile(r"(LaunchKernel|LaunchCooperativeKernel|cuLaunch)")
GRAPH_LAUNCH = re.compile(r"GraphLaunch")
# calls that block the host until the device catches up
BLOCKING = re.compile(r"(Synchronize|^cudaMemcpy_|^cudaMemcpy$|^cuMemcpyDtoH|^cudaMemcpy2D_v)")


def table_exists(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def load(path):
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))

    nvtx = []
    for start, end, text, text_id, tid in db.execute(
        "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND eventType IN (59, 60)"
    ):
        label = text if text is not None else strings.get(text_id, "")
        nvtx.append((start, end, label, tid))

    kernels = list(
        db.execute(
            "SELECT k.start, k.end, k.correlationId, COALESCE(s.value, '') FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "LEFT JOIN StringIds s ON s.id = k.shortName"
        )
    )
    copies = []
    if table_exists(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        copies = list(db.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    sets = []
    if table_exists(db, "CUPTI_ACTIVITY_KIND_MEMSET"):
        sets = list(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"))

    api = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table_exists(db, table):
            for start, end, corr, name_id, tid in db.execute(
                f"SELECT start, end, correlationId, nameId, globalTid FROM {table}"
            ):
                api.append((start, end, corr, strings.get(name_id, ""), tid))
    return nvtx, kernels, copies, sets, api


def union_within(intervals, lo, hi):
    """Total length of the union of sorted intervals clipped to [lo, hi]"""
    total, cur_s, cur_e = 0, None, None
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def in_window(sorted_items, starts, lo, hi):
    i = bisect.bisect_left(starts, lo)
    j = bisect.bisect_left(starts, hi)
    return sorted_items[i:j]


def analyze(path, skip, window=None):
    nvtx, kernels, copies, sets, api = load(path)
    if window:
        # one window per range with this label (a ComfyUI node), numbered from `skip`
        picked = sorted((s, e, tid) for s, e, label, tid in nvtx if label == window)
        frames = [(s, e, skip + i, tid) for i, (s, e, tid) in enumerate(picked)]
    else:
        frames = sorted((s, e, int(label.split()[1]), tid) for s, e, label, tid in nvtx if re.fullmatch(r"frame \d+", label))
    if not frames:
        raise SystemExit("no matching NVTX ranges; was the capture taken with --nvtx?")
    main_tid = frames[0][3]
    stages = sorted((s, e, label) for s, e, label, tid in nvtx if tid == main_tid and not label.startswith("frame "))
    stage_starts = [s for s, _, _ in stages]

    gpu_ops = sorted([(s, e) for s, e, *_ in kernels] + [(s, e) for s, e, *_ in copies] + list(sets))
    kernels.sort()
    kstarts = [k[0] for k in kernels]
    copies.sort()
    cstarts = [c[0] for c in copies]
    main_api = sorted(a for a in api if a[4] == main_tid)
    astarts = [a[0] for a in main_api]
    launch_time = {a[2]: a[0] for a in api if LAUNCH.search(a[3]) or GRAPH_LAUNCH.search(a[3])}

    def stage_of(t):
        i = bisect.bisect_right(stage_starts, t) - 1
        while i >= 0:
            s, e, label = stages[i]
            if s <= t < e:
                return label
            i -= 1
        return "other"

    per_frame = []
    stage_gpu = defaultdict(float)
    stage_kernels = defaultdict(int)
    stage_wall = defaultdict(float)
    kernel_time = defaultdict(float)
    kernel_count = defaultdict(int)
    steady = [f for f in frames if f[2] >= skip]
    for s, e, idx, _ in frames:
        wall = e - s
        busy = union_within(gpu_ops, s, e)
        ks = in_window(kernels, kstarts, s, e)
        cs = in_window(copies, cstarts, s, e)
        calls = in_window(main_api, astarts, s, e)
        blocked = sum(min(ae, e) - max(as_, s) for as_, ae, _, name, _ in calls if BLOCKING.search(name))
        launches = sum(1 for c in calls if LAUNCH.search(c[3]))
        graph_launches = sum(1 for c in calls if GRAPH_LAUNCH.search(c[3]))
        launch_api = sum(c[1] - c[0] for c in calls if LAUNCH.search(c[3]) or GRAPH_LAUNCH.search(c[3]))
        copy_stats = defaultdict(lambda: [0, 0, 0.0])
        for cs_, ce, nbytes, kind in cs:
            st = copy_stats[COPY_KINDS.get(kind, str(kind))]
            st[0] += 1
            st[1] += nbytes
            st[2] += ce - cs_
        per_frame.append(
            {
                "frame": idx,
                "wall_ms": wall / 1e6,
                "busy": busy / wall if wall else 0.0,
                "gpu_idle_ms": (wall - busy) / 1e6,
                "kernels": len(ks),
                "launch_calls": launches,
                "graph_launches": graph_launches,
                "launch_api_ms": launch_api / 1e6,
                "blocked_ms": blocked / 1e6,
                "host_ms": (wall - blocked) / 1e6,
                "api_calls": len(calls),
                "copies": {k: {"count": v[0], "bytes": v[1], "ms": v[2] / 1e6} for k, v in copy_stats.items()},
            }
        )
        if idx >= skip:
            for ks_, ke, corr, name in ks:
                label = stage_of(launch_time.get(corr, ks_))
                stage_gpu[label] += (ke - ks_) / 1e6
                stage_kernels[label] += 1
                kernel_time[name] += (ke - ks_) / 1e6
                kernel_count[name] += 1
    for s, e, label in stages:
        frame = next((f for f in steady if f[0] <= s < f[1]), None)
        if frame is not None:
            stage_wall[label] += (e - s) / 1e6
    n = max(len(steady), 1)
    return per_frame, {
        "frames": n,
        "stages": {
            k: {"wall_ms": stage_wall.get(k, 0.0) / n, "gpu_ms": stage_gpu[k] / n, "kernels": stage_kernels[k] / n}
            for k in sorted(set(stage_gpu) | set(stage_wall))
        },
        "top_kernels": sorted(
            ({"name": k, "ms_per_frame": v / n, "calls_per_frame": kernel_count[k] / n} for k, v in kernel_time.items()),
            key=lambda d: -d["ms_per_frame"],
        )[:20],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--range", help="analyze whole ranges with this label instead of 'frame N' ranges")
    ap.add_argument("--frames", type=int, default=1, help="with --range: frames per range, to report per-frame figures")
    args = ap.parse_args()
    per_frame, agg = analyze(args.sqlite, args.skip, args.range)
    if args.range and args.frames > 1:
        # report per-frame averages; GPU-busy share is a ratio and stays as is
        for f in per_frame:
            for key in ("wall_ms", "gpu_idle_ms", "kernels", "launch_calls", "graph_launches", "launch_api_ms", "blocked_ms", "host_ms", "api_calls"):
                f[key] /= args.frames
            for c in f["copies"].values():
                c["count"] /= args.frames
                c["bytes"] /= args.frames
                c["ms"] /= args.frames
        for v in agg["stages"].values():
            for key in v:
                v[key] /= args.frames
        for k in agg["top_kernels"]:
            k["ms_per_frame"] /= args.frames
            k["calls_per_frame"] /= args.frames
    steady = [f for f in per_frame if f["frame"] >= args.skip] or per_frame

    def med(key):
        return statistics.median(f[key] for f in steady)

    copy_kinds = sorted({k for f in steady for k in f["copies"]})
    print(f"frames {len(per_frame)} (steady {len(steady)}, skip {args.skip})")
    print("| wall ms | GPU busy | GPU idle ms | kernels | launch calls | graph launches | launch API ms | host blocked ms | host other ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {med('wall_ms'):.2f} | {100 * med('busy'):.1f}% | {med('gpu_idle_ms'):.2f} | {med('kernels'):.0f} | "
        f"{med('launch_calls'):.0f} | {med('graph_launches'):.0f} | {med('launch_api_ms'):.2f} | {med('blocked_ms'):.2f} | {med('host_ms'):.2f} |"
    )
    for kind in copy_kinds:
        cnt = statistics.median(f["copies"].get(kind, {}).get("count", 0) for f in steady)
        mb = statistics.median(f["copies"].get(kind, {}).get("bytes", 0) for f in steady) / 2**20
        ms = statistics.median(f["copies"].get(kind, {}).get("ms", 0) for f in steady)
        print(f"memcpy {kind}: {cnt:.0f} per frame, {mb:.2f} MiB, {ms:.3f} ms GPU")
    print("\n| stage | wall ms/frame | GPU ms/frame | kernels/frame |")
    print("|---|---:|---:|---:|")
    for k, v in sorted(agg["stages"].items(), key=lambda kv: -kv[1]["gpu_ms"]):
        print(f"| {k} | {v['wall_ms']:.2f} | {v['gpu_ms']:.2f} | {v['kernels']:.0f} |")
    print("\n| kernel | ms/frame | calls/frame |")
    print("|---|---:|---:|")
    for k in agg["top_kernels"]:
        print(f"| {k['name'][:80]} | {k['ms_per_frame']:.3f} | {k['calls_per_frame']:.1f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"per_frame": per_frame, **agg}, f, indent=1)


if __name__ == "__main__":
    main()
