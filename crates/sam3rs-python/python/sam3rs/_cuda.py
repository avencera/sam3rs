"""Make pip-installed cuBLAS visible to the native runtime.

sam3rs opens the NVIDIA driver and cuBLAS 13 at run time by name:
``libcuda.so.1``, ``libcublas.so.13`` and ``libcublasLt.so.13`` on Linux, and
``nvcuda.dll``, ``cublas64_13.dll`` and ``cublasLt64_13.dll`` on Windows. The
``nvidia-cublas`` wheel installed by the ``cuda`` extra puts cuBLAS inside
site-packages, where neither loader looks, so it is loaded here first. On Linux
``RTLD_GLOBAL`` makes glibc resolve the runtime's soname lookups to the
already-loaded copies; Windows reuses a loaded DLL with the requested name, from
whatever directory it came from. Either way the wheel wins even when a CUDA
toolkit on the loader path has its own cuBLAS. A system CUDA 13 toolkit on the
loader path works without the extra.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

_WINDOWS = sys.platform == "win32"
# libcublas links against libcublasLt by name, so Lt loads first. The path is the
# library's directory inside the wheel's ``nvidia`` namespace package
_CUBLAS: tuple[tuple[str, ...], tuple[str, ...]] = (
    (("cu13", "bin", "x86_64"), ("cublasLt64_13.dll", "cublas64_13.dll"))
    if _WINDOWS
    else (("cu13", "lib"), ("libcublasLt.so.13", "libcublas.so.13"))
)
_prepared = False
# a second thread must wait for the first to finish loading, not skip ahead
_lock = threading.Lock()


def prepare_libraries() -> None:
    """Load the pip cuBLAS libraries once, before the first model loads."""
    global _prepared
    # macOS runs on Metal and needs no cuBLAS
    if not (_WINDOWS or sys.platform.startswith("linux")):
        return
    with _lock:
        if not _prepared:
            _load()
            _prepared = True


def _load() -> None:
    try:
        import nvidia  # type: ignore[import-not-found]  # namespace package of the nvidia-* wheels
    except ImportError:
        return

    subdir, names = _CUBLAS
    for root in getattr(nvidia, "__path__", []):
        lib_dir = Path(root).joinpath(*subdir)
        for name in names:
            path = lib_dir / name
            if path.is_file():
                try:
                    # Windows ignores the mode; an absolute path there also searches the
                    # DLL's own directory for its imports
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # a broken install falls back to the system search path
                    pass
