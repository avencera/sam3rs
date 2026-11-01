"""Baseline and parity reference for the official SAM 3 / SAM 3.1 PyTorch video trackers.

    python -I scripts/video_reference.py prompts --ckpt models/sam3/sam3.pt
    python -I scripts/video_reference.py bench --path sam3-tracker --variant eager \\
        --clips grazing-1080p-256 horses-1080p-256 --out _scratch/video-bench
    python -I scripts/video_reference.py dump --clip horses-720p-64 --out models/video-parity
    python -I scripts/video_reference.py compare _scratch/video-bench

`prompts` runs the SAM 3 image model on frame 0 of each clip with a text query and writes
one positive click per object (the deepest point of each detected mask) to
`tests/video_prompts.json`, so every runtime tracks the same objects from the same clicks.

`bench` times one tracker path in one variant over a list of clips, one JSON line per clip
and object count. Paths:

- `sam3-tracker`: the tracker-only "SAM 3 for SAM 2 video task" path (ViT, tracker neck,
  memory attention, mask decoder, memory encoder), all objects batched in one state
- `sam3-session`, `sam3.1-session`: the official session predictors. They run the full
  detector on every frame, even for click prompts, and `--mode text` tracks every
  detection of the clip's text query
- `sam3.1-tracker`: the multiplex tracker without the detector (best effort: upstream
  only drives it from the session predictor)

Variants: `eager`; `compile`, upstream's own torch.compile settings (plus the ViT for the
tracker-only path, which upstream leaves eager there); `graphs`, per-shape CUDA graphs
captured around the ViT, memory attention, prompt encoder, mask decoder and memory
encoder of the tracker-only path, leaving the Python glue between them eager.

Each clip runs `--runs` times in one process; the first run warms up (and compiles or
captures), the rest are timed. A frame's latency is the wall time between two frames'
masks reaching the host, with a device sync after each. `--nvtx` adds a range per frame
and per stage, and `--profile` brackets the last run with cudaProfilerStart/Stop for
`nsys --capture-range=cudaProfilerApi`.

`bench` loads clips with upstream's cv2 video loader on every path, which feeds the model
unscaled 0-255 pixels (see `use_processor_loader`); timings don't depend on pixel values.

`dump` writes parity fixtures for the tracker-only path: per-frame intermediates in true
FP32 (TF32 off, BF16 autocast off; the memory bank keeps upstream's BF16 storage), plus
the BF16 run's masks. Its frames go through the image processor's transform by default
(`--loader processor`), the preprocessing sam3rs implements. `compare` reports mask IoU of
every variant against eager.
"""

import argparse
import gc
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from torch.utils._pytree import tree_flatten, tree_map_only, tree_unflatten

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PROMPTS = ROOT / "tests/video_prompts.json"
CKPT = {"sam3": ROOT / "models/sam3/sam3.pt", "sam3.1": ROOT / "models/sam3.1/sam3.1_multiplex.pt"}
# text query and object cap per clip, for both the click derivation and `--mode text`
CLIP_QUERIES = {"horses": ("horse", 3), "cyclists": ("person", 3), "grazing": ("horse", 1)}
# steady-state statistics skip the prompted frame and the frames that fill the memory bank
WARM_FRAMES = 8
# NVTX mark inside the `frame N` range whose next() found the video exhausted
END_MARK = "frames exhausted"


def clip_path(name: str) -> pathlib.Path:
    return ROOT / "models/video" / f"{name}.mp4"


def clip_info(path: pathlib.Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info


def first_frame(path: pathlib.Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sam3_commit() -> str:
    import sam3

    src = pathlib.Path(sam3.__file__).resolve().parent.parent
    out = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------- prompts


@torch.inference_mode()
def prompts(args) -> None:
    from PIL import Image
    from reference import build, run_prompt

    processor = build(args.ckpt, "fp32")
    specs = []
    vis_dir = pathlib.Path(args.vis) if args.vis else None
    for clip, (query, cap) in CLIP_QUERIES.items():
        rgb = first_frame(clip_path(f"{clip}-1080p-64"))
        state = processor.set_image(Image.fromarray(rgb))
        result = run_prompt(processor, state, {"text": query})
        scores = result["scores"].float().cpu().numpy()
        masks = result["masks"].cpu().numpy()
        boxes = result["boxes_xyxy"].float().cpu().numpy()
        order = np.argsort(-scores)[:cap]
        h, w = rgb.shape[:2]
        objects = []
        for i in order:
            # the pixel farthest from the mask edge is a click that stays inside the object
            dist = cv2.distanceTransform(masks[i].astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            x0, y0, x1, y1 = boxes[i].tolist()
            objects.append(
                {
                    "point": [round(x / w, 5), round(y / h, 5)],
                    "box": [round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5), round(y1 / h, 5)],
                    "score": round(float(scores[i]), 4),
                }
            )
        specs.append({"clip": clip, "text": query, "objects": objects})
        print(clip, query, [(o["point"], o["score"]) for o in objects])
        if vis_dir:
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis = rgb.copy()
            for k, i in enumerate(order):
                color = np.array([(255, 60, 60), (60, 255, 60), (60, 60, 255)][k % 3], dtype=np.uint8)
                vis[masks[i]] = (0.5 * vis[masks[i]] + 0.5 * color).astype(np.uint8)
                px, py = objects[k]["point"]
                cv2.circle(vis, (int(px * w), int(py * h)), 12, (255, 255, 0), -1)
            cv2.imwrite(str(vis_dir / f"{clip}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    PROMPTS.write_text(json.dumps(specs, indent=2) + "\n")
    print(f"wrote {PROMPTS}")


def load_prompts() -> dict:
    return {spec["clip"]: spec for spec in json.loads(PROMPTS.read_text())}


# ---------------------------------------------------------------------------- graphs


class ShapeGraphs:
    """Replays a callable as one CUDA graph per distinct input signature.

    The signature is the pytree structure plus every tensor's shape, dtype and stride and
    every non-tensor leaf, so memory attention gets one graph per memory-bank length.
    Outputs are cloned out of the graph's static buffers because the tracker keeps them in
    its memory bank. A callable that can't be captured (a host sync inside) falls back to
    eager for good, and the reason is kept in `failed`
    """

    def __init__(self, name: str, fn, max_graphs: int = 96):
        self.name = name
        self.fn = fn
        self.max_graphs = max_graphs
        self.graphs = {}
        self.failed = None
        self.captures = 0
        self.replays = 0

    def __call__(self, *args, **kwargs):
        if self.failed is not None:
            return self.fn(*args, **kwargs)
        leaves, spec = tree_flatten((args, kwargs))
        key = (str(spec),) + tuple(
            ("T", tuple(x.shape), x.dtype, x.stride(), x.device) if isinstance(x, torch.Tensor) else ("V", repr(x))
            for x in leaves
        )
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= self.max_graphs:
                return self.fn(*args, **kwargs)
            try:
                entry = self._capture(leaves, spec)
            except Exception as e:  # noqa: BLE001 - any capture failure means run eager
                self.failed = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                torch.cuda.synchronize()
                return self.fn(*args, **kwargs)
            self.graphs[key] = entry
        static_in, graph, static_out = entry
        for dst, src in zip(static_in, (x for x in leaves if isinstance(x, torch.Tensor))):
            dst.copy_(src)
        graph.replay()
        self.replays += 1
        return tree_map_only(torch.Tensor, torch.clone, static_out)

    def _capture(self, leaves, spec):
        static_leaves = [x.clone() if isinstance(x, torch.Tensor) else x for x in leaves]
        static_in = [x for x in static_leaves if isinstance(x, torch.Tensor)]
        args, kwargs = tree_unflatten(static_leaves, spec)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # autocast's weight cast cache would leak capture-time casts across graphs
        with torch.cuda.stream(side), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for _ in range(2):
                self.fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), torch.cuda.graph(graph):
            static_out = self.fn(*args, **kwargs)
        self.captures += 1
        return static_in, graph, static_out

    def stats(self) -> dict:
        return {"graphs": len(self.graphs), "captures": self.captures, "replays": self.replays, "failed": self.failed}


def use_cv2_loader() -> None:
    """Load mp4 clips with upstream's cv2 loader on every path.

    The tracker-only predictor imports the SAM 2 loader, which needs decord for mp4, and
    the SAM 3.1 demo passes `use_torchcodec`/`use_cv2` to a loader that takes
    `video_loader_type` (both at f1647e8). The session predictors already use the cv2
    loader, so every path now decodes and resizes frames the same way
    """
    import sam3.model.io_utils as io
    import sam3.model.sam3_tracking_predictor as tp
    import sam3.model.video_tracking_multiplex_demo as demo

    def cv2_frames(video_path, image_size, offload_video_to_cpu, async_loading_frames=False, **_ignored):
        return io.load_video_frames(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            video_loader_type="cv2",
        )

    tp.load_video_frames = cv2_frames
    demo.load_video_frames = cv2_frames


def use_processor_loader() -> None:
    """Load mp4 clips through the SAM 3 image processor's transform on the tracker-only path.

    Upstream's cv2 video loader normalizes 0-255 pixels without dividing by 255 (inputs
    land in [-1, 509] at f1647e8), while its image-folder and SAM 2 loaders scale to [0, 1]
    first. The fixtures use the image processor's transform instead (antialiased resize of
    the uint8 frame, then [-1, 1]), which sam3rs's preprocessing already matches
    """
    from torchvision.transforms import v2

    import sam3.model.sam3_tracking_predictor as tp

    def processor_frames(video_path, image_size, offload_video_to_cpu, async_loading_frames=False, **_ignored):
        transform = v2.Compose(
            [
                v2.Resize(size=(image_size, image_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        cap = cv2.VideoCapture(video_path)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            # on the GPU, like the image processor's own transform
            frames.append(transform(rgb.cuda()))
        cap.release()
        video = torch.stack(frames)
        return (video.cpu() if offload_video_to_cpu else video), height, width

    tp.load_video_frames = processor_frames


def drop_unknown_kwargs(fn):
    """Calls `fn` with only the keyword arguments its signature accepts"""
    import inspect

    accepted = set(inspect.signature(fn).parameters)

    def wrapped(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in accepted})

    return wrapped


def nvtx_wrap(label: str, fn):
    def wrapped(*args, **kwargs):
        nvtx.range_push(label)
        try:
            return fn(*args, **kwargs)
        finally:
            nvtx.range_pop()

    return wrapped


def patch_method(obj, attr: str, wrapper):
    setattr(obj, attr, wrapper(getattr(obj, attr)))


# ---------------------------------------------------------------------------- paths


class Sam3Tracker:
    """Tracker-only SAM 3: `build_sam3_video_model().tracker` with the detector's backbone"""

    def __init__(self, variant: str, nvtx_on: bool):
        from sam3.model_builder import build_sam3_video_model

        use_cv2_loader()
        model = build_sam3_video_model(checkpoint_path=str(CKPT["sam3"]), load_from_HF=False)
        self.p = model.tracker
        self.p.backbone = model.detector.backbone
        del model
        gc.collect()
        torch.cuda.empty_cache()
        self.graphs = []
        p = self.p
        if variant == "compile":
            p.compile_all_components = True
            p._compile_all_components()
            # upstream compiles the ViT only in the session predictors; give it here too
            p.backbone.forward_image = torch.compile(p.backbone.forward_image, mode="max-autotune", fullgraph=False)
        elif variant == "graphs":
            targets = [
                ("backbone", p, "forward_image"),
                ("mem_attn", p.transformer.encoder, "forward"),
                ("prompt_enc", p.sam_prompt_encoder, "forward"),
                ("mask_dec", p.sam_mask_decoder, "forward"),
                ("mem_enc", p.maskmem_backbone, "forward"),
            ]
            for name, obj, attr in targets:
                g = ShapeGraphs(name, getattr(obj, attr))
                setattr(obj, attr, g)
                self.graphs.append(g)
        if nvtx_on:
            patch_method(p, "forward_image", lambda f: nvtx_wrap("backbone", f))
            patch_method(p.transformer.encoder, "forward", lambda f: nvtx_wrap("mem_attn", f))
            patch_method(p.sam_prompt_encoder, "forward", lambda f: nvtx_wrap("prompt_enc", f))
            patch_method(p.sam_mask_decoder, "forward", lambda f: nvtx_wrap("mask_dec", f))
            patch_method(p.maskmem_backbone, "forward", lambda f: nvtx_wrap("mem_enc", f))
            patch_method(p, "_get_orig_video_res_output", lambda f: nvtx_wrap("upsample", f))
            import sam3.model.sam3_tracker_base as tb
            import sam3.model.sam3_tracking_predictor as tp

            for mod in (tb, tp):
                if hasattr(mod, "fill_holes_in_mask_scores"):
                    mod.fill_holes_in_mask_scores = nvtx_wrap("fill_holes", mod.fill_holes_in_mask_scores)

    def start(self, clip):
        return self.p.init_state(video_path=str(clip), offload_video_to_cpu=False, async_loading_frames=False)

    def add(self, state, spec, mode, n_objects):
        if mode != "points":
            raise ValueError("the tracker-only path takes clicks only")
        for i, obj in enumerate(spec["objects"][:n_objects]):
            self.p.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=i + 1,
                points=torch.tensor([obj["point"]], dtype=torch.float32),
                labels=torch.tensor([1], dtype=torch.int32),
                clear_old_points=True,
            )

    def frames(self, state, n):
        for frame_idx, obj_ids, low_res, video_res, scores in self.p.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False, tqdm_disable=True, propagate_preflight=True
        ):
            yield frame_idx, list(obj_ids), _binary_masks(frame_idx, video_res, low_res, scores)

    def close(self, state):
        state.clear()

    def stats(self) -> dict:
        return {g.name: g.stats() for g in self.graphs}


class Session:
    """The official session predictors (`handle_request` API, called directly)"""

    def __init__(self, version: str, variant: str, nvtx_on: bool):
        if variant == "graphs":
            raise ValueError("graphs are only wired for the tracker-only path")
        compile_ = variant == "compile"
        if version == "sam3":
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor

            self.pred = Sam3VideoPredictor(checkpoint_path=str(CKPT["sam3"]), compile=compile_)
            tracker = self.pred.model.tracker
        else:
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            # FA3 is Hopper-only; upstream falls back to SDPA without flash_attn_interface
            self.pred = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(CKPT["sam3.1"]), compile=compile_, use_fa3=False, async_loading_frames=False
            )
            tracker = self.pred.model.tracker.model
            # start_session passes offload_state_to_cpu, which this init_state lacks (f1647e8)
            self.pred.model.init_state = drop_unknown_kwargs(self.pred.model.init_state)
        if nvtx_on:
            m = self.pred.model
            for name in ("run_backbone_and_detection", "_prepare_backbone_feats"):
                if hasattr(m, name):
                    patch_method(m, name, lambda f: nvtx_wrap("backbone_detector", f))
            for label, attr in (("mem_attn", "transformer"), ("mask_dec", "sam_mask_decoder"), ("mem_enc", "maskmem_backbone")):
                mod = getattr(tracker, attr, None)
                if mod is None:
                    continue
                target = mod.encoder if label == "mem_attn" and hasattr(mod, "encoder") else mod
                patch_method(target, "forward", lambda f, label=label: nvtx_wrap(label, f))

    def start(self, clip):
        return self.pred.start_session(resource_path=str(clip))["session_id"]

    def add(self, sid, spec, mode, n_objects):
        if mode == "text":
            self.pred.add_prompt(sid, frame_idx=0, text=spec["text"])
            return
        for i, obj in enumerate(spec["objects"][:n_objects]):
            self.pred.add_prompt(sid, frame_idx=0, points=[obj["point"]], point_labels=[1], obj_id=i + 1)

    def frames(self, sid, n):
        for r in self.pred.propagate_in_video(
            sid, propagation_direction="forward", start_frame_idx=0, max_frame_num_to_track=n
        ):
            out = r["outputs"]
            yield r["frame_index"], list(np.asarray(out["out_obj_ids"]).tolist()), np.asarray(out["out_binary_masks"])

    def close(self, sid):
        self.pred.close_session(sid)

    def stats(self) -> dict:
        return {}


class Sam31Tracker:
    """SAM 3.1 multiplex tracker without the detector, from the merged checkpoint.

    Upstream only drives this model from the session predictor, which feeds it detector
    features; here the base class's video-loading `init_state` and its own backbone
    (vision trunk plus the tri-head neck) stand in
    """

    def __init__(self, variant: str, nvtx_on: bool):
        from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo
        from sam3.model_builder import build_sam3_multiplex_video_model

        if variant != "eager":
            raise ValueError("only eager is wired for the SAM 3.1 tracker-only path")
        use_cv2_loader()
        self.base = VideoTrackingMultiplexDemo
        m = build_sam3_multiplex_video_model(
            checkpoint_path=None, load_from_HF=False, use_fa3=False, use_rope_real=True, strict_state_dict_loading=False
        )
        ckpt = torch.load(CKPT["sam3.1"], map_location="cpu", weights_only=True, mmap=True)
        ckpt = ckpt.get("model", ckpt)
        remapped = {}
        for k, v in ckpt.items():
            if k.startswith("tracker.model."):
                remapped[k[len("tracker.model.") :]] = v
            elif k.startswith("detector.backbone.vision_backbone."):
                remapped[k[len("detector.") :]] = v
        missing, unexpected = m.load_state_dict(remapped, strict=False)
        # rope tables are buffers the model rebuilds, so only other keys mean a bad remap
        missing = [k for k in missing if "freqs_cis" not in k]
        print(f"sam3.1 tracker-only: {len(missing)} missing, {len(unexpected)} unexpected keys", missing[:6], unexpected[:6])
        if missing or unexpected:
            raise RuntimeError("SAM 3.1 tracker-only weights did not map onto the model")
        self.m = m.cuda().eval()
        self.bf16 = torch.autocast("cuda", dtype=torch.bfloat16)
        self.bf16.__enter__()
        if nvtx_on:
            patch_method(self.m, "forward_image", lambda f: nvtx_wrap("backbone", f))

    def start(self, clip):
        return self.base.init_state(
            self.m, video_path=str(clip), offload_video_to_cpu=False, offload_state_to_cpu=False, use_cv2=True
        )

    def add(self, state, spec, mode, n_objects):
        for i, obj in enumerate(spec["objects"][:n_objects]):
            self.m.add_new_points(
                state,
                frame_idx=0,
                obj_id=i + 1,
                points=torch.tensor([obj["point"]], dtype=torch.float32),
                labels=torch.tensor([1], dtype=torch.int32),
                clear_old_points=True,
            )

    def frames(self, state, n):
        self.m.propagate_in_video_preflight(state)
        for out in self.base.propagate_in_video(
            self.m, state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False, tqdm_disable=True
        ):
            # the multiplex demo yields frame_idx, obj_ids, low_res_masks, video_res_masks
            frame_idx, obj_ids, low_res, video_res = out
            yield frame_idx, list(obj_ids), _binary_masks(frame_idx, video_res, low_res)

    def close(self, state):
        state.clear()

    def stats(self) -> dict:
        return {}


def make_path(name: str, variant: str, nvtx_on: bool):
    if name == "sam3-tracker":
        return Sam3Tracker(variant, nvtx_on)
    if name == "sam3.1-tracker":
        return Sam31Tracker(variant, nvtx_on)
    if name == "sam3-session":
        return Session("sam3", variant, nvtx_on)
    if name == "sam3.1-session":
        return Session("sam3.1", variant, nvtx_on)
    raise ValueError(name)


# ---------------------------------------------------------------------------- bench


class VramSampler:
    """Peak device memory of this process from nvidia-smi, sampled every 100 ms.

    torch's allocator peaks miss the CUDA context, cuBLAS workspaces and graph pools
    """

    def __init__(self):
        self.pid = os.getpid()
        self.peak = 0
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid) and parts[1].isdigit():
                self.peak = max(self.peak, int(parts[1]))

    def reset(self):
        self.peak = 0

    def stop(self):
        self.proc.terminate()


def summarize(frame_s: list) -> dict:
    steady = frame_s[WARM_FRAMES:] if len(frame_s) > WARM_FRAMES * 2 else frame_s[1:]
    ms = sorted(x * 1000 for x in steady)
    return {
        "frames": len(frame_s),
        "total_s": round(sum(frame_s), 4),
        "fps_total": round(len(frame_s) / sum(frame_s), 2),
        "steady_median_ms": round(statistics.median(ms), 2),
        "steady_mean_ms": round(statistics.fmean(ms), 2),
        "steady_p90_ms": round(ms[int(0.9 * (len(ms) - 1))], 2),
        "fps_steady": round(1000 / statistics.fmean(ms), 2),
    }


@torch.inference_mode()
def bench(args) -> None:
    specs = load_prompts()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = VramSampler()
    t_load = time.perf_counter()
    path = make_path(args.path, args.variant, args.nvtx)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load
    gpu = torch.cuda.get_device_name()
    results = out_dir / f"{args.path}-{args.variant}{'-' + args.mode if args.mode != 'points' else ''}{"-" + args.tag if args.tag else ""}.jsonl"
    for clip in args.clips:
        spec = specs[clip.split("-")[0]]
        info = clip_info(clip_path(clip))
        counts = [len(spec["objects"])] if args.mode == "text" else sorted({min(n, len(spec["objects"])) for n in args.objects})
        for n_obj in counts:
            runs = []
            masks_keep = None
            for r in range(args.runs):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                sampler.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                state = path.start(clip_path(clip))
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                path.add(state, spec, args.mode, n_obj)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                profile = args.profile and r == args.runs - 1
                if profile:
                    torch.cuda.profiler.start()
                frame_s, areas, ids_seen, masks_run = [], [], set(), []
                it = path.frames(state, info["frames"])
                last = time.perf_counter()
                i = 0
                while True:
                    if args.nvtx:
                        nvtx.range_push(f"frame {i}")
                    try:
                        _fi, obj_ids, masks = next(it)
                    except StopIteration:
                        if args.nvtx:
                            # this range held no frame; nsys_frames.py drops ranges with this mark
                            nvtx.mark(END_MARK)
                            nvtx.range_pop()
                        break
                    torch.cuda.synchronize()
                    if args.nvtx:
                        nvtx.range_pop()
                    now = time.perf_counter()
                    frame_s.append(now - last)
                    last = now
                    i += 1
                    ids_seen.update(obj_ids)
                    areas.append([int(m.sum()) for m in masks])
                    if args.keep_masks:
                        masks_run.append((obj_ids, np.packbits(masks.astype(bool), axis=-1)))
                if profile:
                    torch.cuda.profiler.stop()
                torch.cuda.synchronize()
                peak_alloc = torch.cuda.max_memory_allocated() / 2**20
                peak_reserved = torch.cuda.max_memory_reserved() / 2**20
                time.sleep(0.3)
                path.close(state)
                run = {
                    "init_s": round(t1 - t0, 3),
                    "prompt_ms": round((t2 - t1) * 1000, 1),
                    **summarize(frame_s),
                    "peak_alloc_mib": round(peak_alloc),
                    "peak_reserved_mib": round(peak_reserved),
                    "peak_smi_mib": sampler.peak,
                    "objects_tracked": len(ids_seen),
                    "mean_area_px": round(float(np.mean([sum(a) for a in areas])), 1) if areas else 0,
                    "frame_ms": [round(x * 1000, 2) for x in frame_s],
                }
                runs.append(run)
                masks_keep = masks_run
                print(
                    f"{args.path} {args.variant} {clip} obj={n_obj} run={r}: init {run['init_s']} s, "
                    f"prompt {run['prompt_ms']} ms, steady {run['steady_median_ms']} ms "
                    f"({run['fps_steady']} fps), total {run['fps_total']} fps, peak {run['peak_smi_mib']} MiB, "
                    f"tracked {run['objects_tracked']}",
                    flush=True,
                )
            row = {
                "path": args.path,
                "variant": args.variant,
                "mode": args.mode,
                "clip": clip,
                "video": info,
                "objects": n_obj,
                "runs": runs,
                "model_load_s": round(load_s, 1),
                "graph_stats": path.stats(),
                "gpu": gpu,
                "torch": torch.__version__,
                "sam3_commit": sam3_commit(),
                "nvtx": args.nvtx,
                "profiled": args.profile,
            }
            with results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if args.keep_masks and masks_keep:
                mask_dir = out_dir / "masks"
                mask_dir.mkdir(exist_ok=True)
                name = f"{args.path}-{args.mode}-{clip}-o{n_obj}-{args.variant}{"-" + args.tag if args.tag else ""}.npz"
                np.savez_compressed(
                    mask_dir / name,
                    ids=np.array([json.dumps(ids) for ids, _ in masks_keep]),
                    **{f"f{k}": m for k, (_ids, m) in enumerate(masks_keep)},
                )
    print(json.dumps(path.stats()))
    sampler.stop()


# ---------------------------------------------------------------------------- compare


def compare(args) -> None:
    """Mask IoU per frame of each variant's kept masks against the eager run of the same config"""
    mask_dir = pathlib.Path(args.dir) / "masks"
    files = sorted(mask_dir.glob("*.npz"))
    groups = {}
    failed = compared = 0
    for f in files:
        stem = f.stem
        for variant in ("eager", "compile", "graphs"):
            if f"-{variant}" in stem:
                groups.setdefault(stem.split(f"-{variant}")[0], {})[variant + stem.split(f"-{variant}")[1]] = f
    for config, variants in sorted(groups.items()):
        ref_key = next((k for k in variants if k.startswith("eager")), None)
        if ref_key is None:
            raise SystemExit(f"{config} has no eager run to compare against")
        ref = np.load(variants[ref_key])
        for key, f in sorted(variants.items()):
            if key == ref_key:
                continue
            other = np.load(f)
            ious = []
            # a frame or object only one run produced counts as IoU 0, so a dropped
            # object can't hide behind the ones both runs kept
            n_ref, n_other = len(ref["ids"]), len(other["ids"])
            missing = abs(n_ref - n_other)
            ious.extend([0.0] * missing)
            for k in range(min(n_ref, n_other)):
                a = _masks_by_id(ref, k)
                b = _masks_by_id(other, k)
                for o in sorted(a.keys() | b.keys()):
                    if o not in a or o not in b:
                        missing += 1
                        ious.append(0.0)
                        continue
                    union = (a[o] | b[o]).sum()
                    ious.append(1.0 if union == 0 else (a[o] & b[o]).sum() / union)
            ious = np.array(ious)
            print(
                f"{config} {key} vs {ref_key}: frames {n_ref} vs {n_other}, mean IoU {ious.mean():.5f}, "
                f"min {ious.min():.5f}, <0.99: {(ious < 0.99).sum()} of {len(ious)}, missing {missing}"
            )
            compared += 1
            # graph capture and compile change rounding only, so a mean below the limit
            # means a variant tracked something else
            failed += missing > 0 or ious.mean() < args.min_mean_iou
    if not compared:
        raise SystemExit(f"no variant to compare against an eager run in {mask_dir}")
    if failed:
        raise SystemExit(f"{failed} comparisons are missing frames or objects or below mean IoU {args.min_mean_iou}")


def _binary_masks(frame_idx: int, video_res, *raw) -> np.ndarray:
    """Masks from `video_res` logits `[objects, 1, h, w]`, after checking they and the other
    raw outputs (low-res logits, scores) are finite: thresholding would turn NaN into
    background, which a comparison against an empty mask then counts as a match"""
    for t in (video_res, *raw):
        if t is not None and not bool(torch.isfinite(t).all()):
            raise RuntimeError(f"frame {frame_idx}: non-finite tracker output")
    return (video_res[:, 0] > 0).cpu().numpy()


def _masks_by_id(npz, k: int) -> dict:
    """Frame k's masks keyed by object ID, failing when IDs and mask planes disagree"""
    ids = json.loads(npz["ids"][k])
    planes = np.unpackbits(npz[f"f{k}"], axis=-1).astype(bool)
    if len(ids) != len(planes):
        raise RuntimeError(f"frame {k}: {len(ids)} object IDs but {len(planes)} masks")
    return dict(zip(ids, planes))


# ---------------------------------------------------------------------------- dump


def _cpu(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu().contiguous()
    return v


@torch.inference_mode()
def dump(args) -> None:
    """Per-frame intermediates of the tracker-only path, FP32 reference plus BF16 masks"""
    from safetensors.torch import save_file

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = args.clip
    spec = load_prompts()[clip.split("-")[0]]
    n_obj = min(args.objects, len(spec["objects"]))
    info = clip_info(clip_path(clip))
    n_frames = min(args.frames, info["frames"])

    from sam3.model_builder import build_sam3_video_model

    if args.loader == "processor":
        use_processor_loader()
    else:
        use_cv2_loader()
    model = build_sam3_video_model(checkpoint_path=str(CKPT["sam3"]), load_from_HF=False)
    p = model.tracker
    p.backbone = model.detector.backbone
    del model
    gc.collect()
    if args.precision == "fp32":
        import sam3.model.vitdet as vitdet
        from reference import _fp32_addmm_act

        # the tracker enters BF16 autocast for the whole process when it is built
        p.bf16_context.__exit__(None, None, None)
        vitdet.addmm_act = _fp32_addmm_act
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # every module call lands in `pending` and is assigned to the frame yielded next
    pending = []
    full_frames = set(range(args.full_frames)) | {7, 8, 15, 16, n_frames - 1}

    def hook(name):
        def fn(_m, a, kw, out):
            pending.append((name, a, kw, out))

        return fn

    handles = [
        p.transformer.encoder.register_forward_hook(hook("mem_attn"), with_kwargs=True),
        p.sam_mask_decoder.register_forward_hook(hook("mask_dec"), with_kwargs=True),
        p.maskmem_backbone.register_forward_hook(hook("mem_enc"), with_kwargs=True),
        p.obj_ptr_proj.register_forward_hook(hook("obj_ptr"), with_kwargs=True),
    ]
    orig_forward_image = p.forward_image

    def forward_image(img):
        out = orig_forward_image(img)
        pending.append(("backbone", (img,), {}, out))
        return out

    p.forward_image = forward_image

    state = p.init_state(video_path=str(clip_path(clip)), offload_video_to_cpu=False, async_loading_frames=False)
    tensors = {}
    meta = {
        "clip": clip,
        "video": info,
        "precision": args.precision,
        "loader": args.loader,
        "objects": spec["objects"][:n_obj],
        "frames": n_frames,
        "sam3_commit": sam3_commit(),
        "torch": torch.__version__,
        "per_frame": [],
    }
    for i, obj in enumerate(spec["objects"][:n_obj]):
        p.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=i + 1,
            points=torch.tensor([obj["point"]], dtype=torch.float32),
            labels=torch.tensor([1], dtype=torch.int32),
            clear_old_points=True,
        )
    for frame_idx, obj_ids, low_res, video_res, scores in p.propagate_in_video(
        # upstream's end frame is inclusive: start + max_frame_num_to_track
        state, start_frame_idx=0, max_frame_num_to_track=n_frames - 1, reverse=False, tqdm_disable=True, propagate_preflight=True
    ):
        t = frame_idx
        full = args.precision == "fp32" and t in full_frames
        frame_meta = {"frame": t, "obj_ids": list(obj_ids), "calls": [n for n, *_ in pending]}
        if t < n_frames and args.precision == "fp32":
            tensors[f"f{t}.input"] = (state["images"][t].float().cpu() if t < len(state["images"]) else torch.empty(0))
        for k, (name, a, kw, out) in enumerate(pending):
            key = f"f{t}.{name}.{k}"
            if name == "backbone" and full:
                fpn = out["backbone_fpn"]
                for lvl in range(len(fpn)):
                    tensors[f"{key}.fpn{lvl}"] = _cpu(fpn[lvl])
            elif name == "mem_attn":
                prompt = kw["prompt"]
                frame_meta["mem_tokens"] = int(prompt.shape[0])
                frame_meta["num_obj_ptr_tokens"] = int(kw.get("num_obj_ptr_tokens", 0))
                if full:
                    tensors[f"{key}.src"] = _cpu(kw["src"][0])
                    tensors[f"{key}.src_pos"] = _cpu(kw["src_pos"][0])
                    tensors[f"{key}.prompt"] = _cpu(prompt)
                    tensors[f"{key}.prompt_pos"] = _cpu(kw["prompt_pos"])
                    if kw.get("prompt_key_padding_mask") is not None:
                        tensors[f"{key}.prompt_key_padding_mask"] = _cpu(kw["prompt_key_padding_mask"])
                    tensors[f"{key}.out"] = _cpu(out["memory"])
            elif name == "mask_dec":
                low, ious, tokens, obj_scores = out
                tensors[f"{key}.ious"] = _cpu(ious)
                tensors[f"{key}.obj_score"] = _cpu(obj_scores)
                frame_meta.setdefault("mask_dec_multimask", []).append(bool(kw["multimask_output"]))
                if full:
                    tensors[f"{key}.low_res_multimasks"] = _cpu(low)
                    tensors[f"{key}.tokens"] = _cpu(tokens)
                    tensors[f"{key}.image_embeddings"] = _cpu(kw["image_embeddings"])
                    tensors[f"{key}.image_pe"] = _cpu(kw["image_pe"])
                    tensors[f"{key}.sparse"] = _cpu(kw["sparse_prompt_embeddings"])
                    tensors[f"{key}.dense"] = _cpu(kw["dense_prompt_embeddings"])
                    for lvl, feat in enumerate(kw.get("high_res_features") or []):
                        tensors[f"{key}.high_res{lvl}"] = _cpu(feat)
            elif name == "mem_enc":
                # `features` is the tensor upstream later adds `no_obj_embed_spatial` to in
                # place (sam3_tracker_base.py:845), so absent objects' rows include it
                if full:
                    tensors[f"{key}.pix_feat"] = _cpu(a[0])
                    tensors[f"{key}.mask"] = _cpu(a[1])
                    tensors[f"{key}.features"] = _cpu(out["vision_features"])
            elif name == "obj_ptr":
                if full:
                    tensors[f"{key}.in"] = _cpu(a[0])
                tensors[f"{key}.out"] = _cpu(out)
        pending.clear()
        tensors[f"f{t}.low_res"] = _cpu(low_res)
        tensors[f"f{t}.scores"] = _cpu(scores)
        tensors[f"f{t}.masks"] = torch.from_numpy(
            np.packbits(_binary_masks(t, video_res, low_res, scores), axis=-1)
        )
        meta["per_frame"].append(frame_meta)
    if len(meta["per_frame"]) != n_frames:
        raise RuntimeError(f"tracked {len(meta['per_frame'])} frames, the fixture declares {n_frames}")
    for h in handles:
        h.remove()
    name = f"{clip}-o{n_obj}-{args.precision}"
    tensors = {k: (v if v.dtype == torch.uint8 else v.float()).contiguous() for k, v in tensors.items() if v.numel()}
    save_file(tensors, out_dir / f"{name}.safetensors", metadata={"meta": json.dumps(meta)})
    size = sum(v.numel() * v.element_size() for v in tensors.values()) / 2**20
    print(f"wrote {name}: {len(tensors)} tensors, {size:.0f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prompts")
    pr.add_argument("--ckpt", default=str(CKPT["sam3"]))
    pr.add_argument("--vis", default=None, help="directory for overlay images of the derived clicks")
    b = sub.add_parser("bench")
    b.add_argument("--path", required=True, choices=["sam3-tracker", "sam3.1-tracker", "sam3-session", "sam3.1-session"])
    b.add_argument("--variant", default="eager", choices=["eager", "compile", "graphs"])
    b.add_argument("--mode", default="points", choices=["points", "text"])
    b.add_argument("--clips", nargs="+", required=True)
    b.add_argument("--objects", nargs="+", type=int, default=[1, 3])
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--out", required=True)
    b.add_argument("--tag", default="", help="suffix for output names, joined with a dash")
    b.add_argument("--nvtx", action="store_true")
    b.add_argument("--profile", action="store_true")
    b.add_argument("--keep-masks", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("dir")
    c.add_argument("--min-mean-iou", type=float, default=0.95, help="fail a variant whose mean IoU is lower")
    d = sub.add_parser("dump")
    d.add_argument("--clip", default="horses-720p-64")
    d.add_argument("--objects", type=int, default=3)
    d.add_argument("--frames", type=int, default=24)
    d.add_argument("--full-frames", type=int, default=3, help="frames with every intermediate kept")
    d.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    d.add_argument(
        "--loader",
        choices=["processor", "cv2"],
        default="processor",
        help="frame transform: the image processor's (sam3rs's), or upstream's cv2 video loader",
    )
    d.add_argument("--out", default=str(ROOT / "models/video-parity"))
    args = parser.parse_args()
    {"prompts": prompts, "bench": bench, "compare": compare, "dump": dump}[args.cmd](args)


if __name__ == "__main__":
    main()
