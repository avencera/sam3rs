"""Start a ComfyUI server with per-node timing, for scripts/bench_comfy.py.

    python -I scripts/comfy_launch.py COMFY_DIR -- --listen 127.0.0.1 --port 8189

ComfyUI itself is unchanged. When its `execution` module loads, this wraps
`get_output_data` (one call per executed node) with a device sync on both sides, an NVTX
range named after the node class, and a JSON line in `$COMFY_NODE_TIMES` with the node's
wall time and the torch allocator peak. `$COMFY_PROFILE_PROMPT=k` brackets the k-th prompt
(1-based) with cudaProfilerStart/Stop for `nsys --capture-range=cudaProfilerApi`
"""

import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import time


def patch_execution(module):
    import torch
    from torch.cuda import nvtx

    times_path = os.environ.get("COMFY_NODE_TIMES")
    profile_prompt = int(os.environ.get("COMFY_PROFILE_PROMPT", "0"))
    state = {"prompt": 0}
    original = module.get_output_data

    async def get_output_data(prompt_id, unique_id, obj, *args, **kwargs):
        name = obj.__name__ if isinstance(obj, type) else type(obj).__name__
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        nvtx.range_push(f"node {name}")
        t0 = time.perf_counter()
        try:
            return await original(prompt_id, unique_id, obj, *args, **kwargs)
        finally:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            nvtx.range_pop()
            if times_path:
                with open(times_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "prompt_id": prompt_id,
                                "node": unique_id,
                                "class": name,
                                "s": dt,
                                "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
                            }
                        )
                        + "\n"
                    )

    module.get_output_data = get_output_data

    executor = module.PromptExecutor
    original_execute = executor.execute_async

    async def execute_async(self, *args, **kwargs):
        state["prompt"] += 1
        profile = state["prompt"] == profile_prompt
        if profile:
            torch.cuda.profiler.start()
        nvtx.range_push(f"prompt {state['prompt']}")
        try:
            return await original_execute(self, *args, **kwargs)
        finally:
            torch.cuda.synchronize()
            nvtx.range_pop()
            if profile:
                torch.cuda.profiler.stop()

    executor.execute_async = execute_async


class PatchOnLoad(importlib.abc.MetaPathFinder):
    """Patches `execution` right after ComfyUI imports it, so main.py's own setup runs first"""

    def find_spec(self, fullname, path, target=None):
        if fullname != "execution":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch_execution(module)

        loader.exec_module = exec_and_patch
        return spec


def main():
    comfy_dir = os.path.abspath(sys.argv[1])
    rest = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    os.chdir(comfy_dir)
    sys.path.insert(0, comfy_dir)
    sys.meta_path.insert(0, PatchOnLoad())
    sys.argv = [os.path.join(comfy_dir, "main.py"), *rest]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
