"""Make pip-installed cuBLAS visible to the native runtime.

sam3rs opens ``libcuda.so.1`` and cuBLAS 13 (``libcublas.so.13``,
``libcublasLt.so.13``) at run time by soname. The ``nvidia-cublas`` wheel
installed by the ``cuda`` extra puts cuBLAS inside site-packages, where the
dynamic loader does not look, so it is loaded here first with ``RTLD_GLOBAL``:
glibc then resolves the runtime's soname lookups to the already-loaded copies.
A system CUDA 13 toolkit on the loader path works without the extra.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

# libcublas links against libcublasLt, so Lt loads first
_CUBLAS = ("libcublasLt.so.13", "libcublas.so.13")
_prepared = False
# a second thread must wait for the first to finish loading, not skip ahead
_lock = threading.Lock()


def prepare_libraries() -> None:
    """Load the pip cuBLAS libraries once, before the first model loads."""
    global _prepared
    if not sys.platform.startswith("linux"):
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

    for root in getattr(nvidia, "__path__", []):
        lib_dir = Path(root) / "cu13" / "lib"
        for name in _CUBLAS:
            path = lib_dir / name
            if path.is_file():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # a broken install falls back to the system search path
                    pass
