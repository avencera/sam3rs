"""The exception type raised for every sam3rs failure."""

from __future__ import annotations


class Sam3rsError(Exception):
    """A sam3rs operation failed: no usable GPU, driver or cuBLAS, a bad checkpoint,
    a failed download, an unreadable image or an invalid prompt."""
