"""Time osam's SAM 3 ONNX Runtime pipeline with the osam benchmark protocol.

    HOME=<dir with osam blobs> python -I scripts/bench_osam.py [--ort-profile DIR]

Mirrors osam's examples/benchmark.py (dogs.jpg, median of 5 after one warm-up, model
load excluded) and wraps every session.run in an NVTX range named after the session, so
nsys can split GPU time by stage. --ort-profile turns on ONNX Runtime's per-node profiler
and summarizes time by execution provider, which shows any CPU-placed ops
"""

import argparse
import collections
import json
import pathlib
import statistics
import time

import numpy as np
import nvtx
import onnxruntime
import PIL.Image

import osam
import osam.types._model as osam_model

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROMPTS = {
    "point": osam.types.Prompt(points=np.array([[1439, 504]]), point_labels=np.array([1])),
    "box": osam.types.Prompt(
        points=np.array([[1233, 376], [1649, 691]]), point_labels=np.array([2, 3])
    ),
    "negative box": osam.types.Prompt(
        points=np.array([[1233, 376], [1649, 691], [1400, 500], [1500, 600]]),
        point_labels=np.array([2, 3, 4, 5]),
    ),
    "text": osam.types.Prompt(texts=["dog"]),
}


class _NvtxSession:
    """Forwards to an InferenceSession, wrapping run() in an NVTX range"""

    def __init__(self, name, session):
        self._name = name
        self._session = session

    def run(self, *args, **kwargs):
        with nvtx.annotate(f"ort:{self._name}", color="green"):
            return self._session.run(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._session, attr)


def _median_ms(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def _profile_session_factory(profile_dir: pathlib.Path):
    original = osam_model._create_session

    def create(blob, providers):
        sess_options = onnxruntime.SessionOptions()
        if "CUDAExecutionProvider" in providers and osam_model._setup_cuda():
            sess_options.add_session_config_entry("session.use_env_allocators", "1")
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = str(profile_dir / pathlib.Path(blob.path).name[:20])
        return onnxruntime.InferenceSession(blob.path, sess_options, providers=providers)

    return create if profile_dir else original


def summarize_profile(path: pathlib.Path) -> dict:
    """Total node time per (provider, op type) from an ORT profile json"""
    events = json.loads(path.read_text())
    by_provider = collections.Counter()
    by_op = collections.Counter()
    for e in events:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
            continue
        args = e.get("args", {})
        provider = args.get("provider", "?")
        by_provider[provider] += e["dur"]
        by_op[(provider, args.get("op_name", "?"))] += e["dur"]
    return {
        "us_by_provider": dict(by_provider),
        "top_ops": [(p, op, us) for (p, op), us in by_op.most_common(25)],
        "cpu_ops": sorted({op for (p, op) in by_op if p == "CPUExecutionProvider"}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=str(ROOT / "models/images/dogs.jpg"))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--ort-profile", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.ort_profile:
        args.ort_profile.mkdir(parents=True, exist_ok=True)
        osam_model._create_session = _profile_session_factory(args.ort_profile)

    model_type = osam.apis.get_model_type_by_name("sam3")
    model = model_type(device=args.device)
    model._inference_sessions = {
        k: _NvtxSession(k, s) for k, s in model._inference_sessions.items()
    }
    print("providers:", {k: s.get_providers()[0] for k, s in model._inference_sessions.items()})
    image = np.asarray(PIL.Image.open(args.image).convert("RGB"))

    with nvtx.annotate("encode warmup"):
        embedding = model.encode_image(image)

    def encode():
        with nvtx.annotate("encode", color="red"):
            model.encode_image(image)

    row = [f"{_median_ms(encode, args.repeat):.1f}"]
    for name, prompt in PROMPTS.items():
        request = osam.types.GenerateRequest(model=model.name, image_embedding=embedding, prompt=prompt)
        with nvtx.annotate(f"{name} warmup"):
            model.generate(request)

        def generate(name=name, request=request):
            with nvtx.annotate(name, color="blue"):
                model.generate(request)

        row.append(f"{_median_ms(generate, args.repeat):.1f}")

    print("| Runtime | Encode | " + " | ".join(PROMPTS) + " |")
    print(f"| osam ORT {args.device} | " + " | ".join(row) + " |")

    if args.ort_profile:
        for name, session in model._inference_sessions.items():
            path = pathlib.Path(session.end_profiling())
            summary = summarize_profile(path)
            (args.ort_profile / f"{name}.summary.json").write_text(json.dumps(summary, indent=1))
            print(name, "us by provider:", summary["us_by_provider"], "cpu ops:", summary["cpu_ops"])


if __name__ == "__main__":
    main()
