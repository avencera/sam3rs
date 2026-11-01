"""Time stock ComfyUI's SAM 3 nodes end to end through the server API, as a user runs them.

    python -I scripts/bench_comfy.py --comfy-dir _scratch/comfyui --out _scratch/comfy-bench \\
        video-click grazing-720p-64:1 horses-1080p-256:3
    python -I scripts/bench_comfy.py ... video-text horses-1080p-256
    python -I scripts/bench_comfy.py ... image-click grazing-1080p-64 --clicks 10

Starts ComfyUI through scripts/comfy_launch.py (per-node timing), links the clips into
its input directory (hard links), and posts API-format workflows built from core nodes only:

- `video-click`: LoadVideo, frame 0 to one SAM3_Detect per object with that object's click
  from tests/video_prompts.json, Batch Masks, SAM3_VideoTrack from those masks (the
  detector doesn't run per frame), SAM3_TrackPreview (the h264 preview a user watches)
- `video-text`: SAM3_VideoTrack with CLIPTextEncode of the clip's query, detecting on
  every frame (detect_interval 1), capped at the clip's object count
- `image-click`: LoadImage of frame 0 and SAM3_Detect with one click, re-queued with a
  new click each time the way an interactive user clicks; `--refine` sets the decoder
  passes (the node default is 2, and each pass re-runs the image encoder)

Each video config runs `--runs` times with a `bench_run` key added to the tracker's inputs:
the node ignores it, but it changes ComfyUI's cache key, so the tracker re-executes with the
same settings and the decoded video stays cached; the first run is
the warm-up and includes video decoding. One JSON line per config goes to
`OUT/comfy-<scenario>.jsonl`. `--nsys PREFIX` runs the server under nsys and profiles the
`--profile-prompt`-th prompt
"""

import argparse
import json
import os
import pathlib
import random
import shlex
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = "sam3.1_multiplex_fp16.safetensors"
QUERIES = {"horses": ("horse", 3), "cyclists": ("person", 3), "grazing": ("horse", 1)}


class GpuMemory:
    """Peak device memory across all compute processes, sampled every 100 ms (the GPU is
    exclusive to this job, so the sum is the server)"""

    def __init__(self):
        self.peak = 0
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()
        self.base = None

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line.isdigit():
                v = int(line)
                if self.base is None:
                    self.base = v
                self.peak = max(self.peak, v)

    def reset(self):
        self.peak = 0

    def stop(self):
        self.proc.terminate()


class Server:
    def __init__(self, args, out: pathlib.Path):
        self.url = f"http://127.0.0.1:{args.port}"
        self.times = out / "node-times.jsonl"
        env = dict(os.environ, COMFY_NODE_TIMES=str(self.times), COMFY_PROFILE_PROMPT=str(args.profile_prompt))
        cmd = [
            args.python,
            "-I",
            str(ROOT / "scripts/comfy_launch.py"),
            args.comfy_dir,
            "--",
            "--listen",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--disable-auto-launch",
            "--output-directory",
            str(out / "output"),
        ]
        self.nsys = bool(args.nsys)
        if self.nsys:
            cmd = shlex.split(args.nsys) + cmd
        self.log = open(out / "server.log", "a")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with {self.proc.returncode}; see {out / 'server.log'}")
            try:
                self.get("/system_stats")
                return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.5)
        raise RuntimeError("ComfyUI did not start")

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=10) as r:
            return json.loads(r.read())

    def run(self, workflow):
        body = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
        req = urllib.request.Request(self.url + "/prompt", data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                prompt_id = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"prompt rejected: {e.read().decode()[:2000]}") from e
        while True:
            hist = self.get(f"/history/{prompt_id}")
            if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed") is not None:
                status = hist[prompt_id]["status"]
                if status.get("status_str") != "success":
                    raise RuntimeError(f"prompt failed: {json.dumps(status)[:3000]}")
                break
            time.sleep(0.005)
        total = time.perf_counter() - t0
        nodes = []
        if self.times.exists():
            for line in self.times.read_text().splitlines():
                rec = json.loads(line)
                if rec["prompt_id"] == prompt_id:
                    nodes.append(rec)
        return total, nodes

    def stop(self):
        self.proc.terminate()
        try:
            # nsys writes and exports its report after the server exits
            self.proc.wait(timeout=600 if self.nsys else 60)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def prompts():
    return {s["clip"]: s for s in json.loads((ROOT / "tests/video_prompts.json").read_text())}


def video_size(path):
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return s.codec_context.width, s.codec_context.height, s.frames


def frame0_png(clip_file: pathlib.Path, dst: pathlib.Path):
    import av

    if dst.exists():
        return
    with av.open(str(clip_file)) as c:
        frame = next(c.decode(video=0))
        frame.to_image().save(dst)


def base_nodes(clip_name):
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "video": {"class_type": "LoadVideo", "inputs": {"file": f"{clip_name}.mp4"}},
        "frames": {"class_type": "GetVideoComponents", "inputs": {"video": ["video", 0]}},
    }


def click_masks(wf, image_ref, objects, w, h, refine):
    """One SAM3_Detect per click, batched into one mask per object"""
    refs = []
    for i, obj in enumerate(objects):
        x, y = obj["point"]
        wf[f"pt{i}"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps([{"x": round(x * w), "y": round(y * h)}])}}
        wf[f"det{i}"] = {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["ckpt", 0],
                "image": image_ref,
                "positive_coords": [f"pt{i}", 0],
                "threshold": 0.5,
                "refine_iterations": refine,
                "individual_masks": True,
            },
        }
        refs.append([f"det{i}", 0])
    if len(refs) == 1:
        return refs[0]
    wf["batch"] = {"class_type": "BatchMasksNode", "inputs": {f"masks.mask{i}": r for i, r in enumerate(refs)}}
    return ["batch", 0]


def video_click(server, args, clip, n_obj, spec, out):
    w, h, frames = video_size(ROOT / "models/video" / f"{clip}.mp4")
    rows = []
    for r in range(args.runs):
        wf = base_nodes(clip)
        wf["first"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["frames", 0], "batch_index": 0, "length": 1}}
        mask = click_masks(wf, ["first", 0], spec["objects"][:n_obj], w, h, args.refine)
        wf["track"] = {
            "class_type": "SAM3_VideoTrack",
            "inputs": {
                "images": ["frames", 0],
                "model": ["ckpt", 0],
                "initial_mask": mask,
                # varied per run only so the cache re-runs the tracker; no text, so it is unused
                "detection_threshold": 0.5,
                "max_objects": n_obj,
                # not a node input, so the node never sees it, but it is part of ComfyUI's
                # cache key: each run re-executes the tracker and keeps the decoded video
                "bench_run": r,
                "detect_interval": 1,
            },
        }
        wf["preview"] = {"class_type": "SAM3_TrackPreview", "inputs": {"track_data": ["track", 0], "images": ["frames", 0], "opacity": 0.5, "fps": 24.0}}
        rows.append(timed(server, wf, r))
    return summarize("video-click", clip, n_obj, frames, rows)


def video_text(server, args, clip, spec, out):
    _, _, frames = video_size(ROOT / "models/video" / f"{clip}.mp4")
    query, cap = QUERIES[clip.split("-")[0]]
    rows = []
    for r in range(args.runs):
        wf = base_nodes(clip)
        wf["text"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": query}}
        wf["track"] = {
            "class_type": "SAM3_VideoTrack",
            "inputs": {
                "images": ["frames", 0],
                "model": ["ckpt", 0],
                "conditioning": ["text", 0],
                # fixed, since text detections near it would change the tracked objects
                "detection_threshold": 0.5,
                "max_objects": cap,
                "bench_run": r,
                "detect_interval": 1,
            },
        }
        wf["preview"] = {"class_type": "SAM3_TrackPreview", "inputs": {"track_data": ["track", 0], "images": ["frames", 0], "opacity": 0.5, "fps": 24.0}}
        rows.append(timed(server, wf, r))
    return summarize("video-text", clip, cap, frames, rows)


def image_click(server, args, clip, spec, out):
    w, h, _ = video_size(ROOT / "models/video" / f"{clip}.mp4")
    png = f"{clip}-frame0.png"
    frame0_png(ROOT / "models/video" / f"{clip}.mp4", pathlib.Path(args.comfy_dir) / "input" / png)
    rng = random.Random(0)
    x0, y0 = spec["objects"][0]["point"]
    rows = []
    for r in range(args.clicks + 1):
        # a new click near the object each time, as a user refining the selection would
        x = min(max(x0 + rng.uniform(-0.01, 0.01), 0.0), 1.0)
        y = min(max(y0 + rng.uniform(-0.01, 0.01), 0.0), 1.0)
        wf = {
            "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
            "img": {"class_type": "LoadImage", "inputs": {"image": png}},
        }
        click_masks(wf, ["img", 0], [{"point": [x, y]}], w, h, args.refine)
        wf["toimg"] = {"class_type": "MaskToImage", "inputs": {"mask": ["det0", 0]}}
        wf["preview"] = {"class_type": "PreviewImage", "inputs": {"images": ["toimg", 0]}}
        rows.append(timed(server, wf, r))
    return summarize("image-click", clip, 1, 1, rows, refine=args.refine)


def timed(server, wf, r):
    server.mem.reset()
    total, nodes = server.run(wf)
    by_class = {}
    for n in nodes:
        by_class[n["class"]] = by_class.get(n["class"], 0.0) + n["s"]
    row = {
        "run": r,
        "total_s": round(total, 4),
        "nodes_s": {k: round(v, 4) for k, v in by_class.items()},
        "peak_alloc_mib": round(max((n["peak_alloc_mib"] for n in nodes), default=0)),
        "peak_smi_mib": server.mem.peak,
    }
    print(json.dumps(row), flush=True)
    return row


def summarize(scenario, clip, n_obj, frames, rows, refine=None):
    timed_rows = rows[1:] if len(rows) > 1 else rows
    track = [x["nodes_s"].get("SAM3_VideoTrack") for x in timed_rows if "SAM3_VideoTrack" in x["nodes_s"]]
    detect = [x["nodes_s"].get("SAM3_Detect") for x in timed_rows if "SAM3_Detect" in x["nodes_s"]]
    out = {
        "scenario": scenario,
        "clip": clip,
        "objects": n_obj,
        "frames": frames,
        "refine": refine,
        "median_total_s": round(statistics.median(x["total_s"] for x in timed_rows), 4),
        "runs": rows,
    }
    if track:
        med = statistics.median(track)
        out["track_node_s"] = round(med, 4)
        out["track_ms_per_frame"] = round(1000 * med / frames, 2)
        out["track_fps"] = round(frames / med, 2)
    if detect:
        out["detect_node_ms"] = round(1000 * statistics.median(detect), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=["video-click", "video-text", "image-click"])
    ap.add_argument("clips", nargs="+", help="clip names; video-click takes clip:objects")
    ap.add_argument("--comfy-dir", default=str(ROOT / "_scratch/comfyui"))
    ap.add_argument("--python", default=str(ROOT / "_scratch/venv-comfy/bin/python"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--clicks", type=int, default=10)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for output names, joined with a dash")
    ap.add_argument("--nsys", default="")
    ap.add_argument("--profile-prompt", type=int, default=0)
    args = ap.parse_args()

    # the launcher changes into the ComfyUI directory, so relative paths would split
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = pathlib.Path(args.comfy_dir) / "input"
    inputs.mkdir(exist_ok=True)
    for f in (ROOT / "models/video").glob("*.mp4"):
        dst = inputs / f.name
        # LoadVideo resolves symlinks and rejects targets outside the input directory
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            try:
                os.link(f.resolve(), dst)
            except OSError:
                shutil.copy2(f.resolve(), dst)
    specs = prompts()
    mem = GpuMemory()
    server = Server(args, out)
    server.mem = mem
    results = out / f"comfy-{args.scenario}{"-" + args.tag if args.tag else ""}.jsonl"
    try:
        for item in args.clips:
            clip, _, n = item.partition(":")
            spec = specs[clip.split("-")[0]]
            if args.scenario == "video-click":
                n_obj = int(n or 1)
                if not 1 <= n_obj <= len(spec["objects"]):
                    raise SystemExit(f"{clip}: {n_obj} objects asked, the prompt file has {len(spec['objects'])}")
                row = video_click(server, args, clip, n_obj, spec, out)
            elif args.scenario == "video-text":
                row = video_text(server, args, clip, spec, out)
            else:
                row = image_click(server, args, clip, spec, out)
            row["idle_smi_mib"] = mem.base
            with results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print({k: v for k, v in row.items() if k != "runs"}, flush=True)
    finally:
        server.stop()
        mem.stop()


if __name__ == "__main__":
    main()
