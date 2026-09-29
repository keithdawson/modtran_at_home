"""NumPy or CuPy: the array module of the arrays a function is given.

Resolved runs on the CUDA backend keep each chunk's fine-grid arrays on the
GPU (see ``mwir_atmos.api._resolved_chunk``), so the physics that follows the
line-by-line sums -- continuum, aerosol, Rayleigh, path transmittance,
single-scatter radiance and the instrument reduction -- is written once for
both: those functions take their array module from their spectral inputs
(normally ``SpectralGrid.wavenumber_cm1``). CuPy is imported only when a CuPy
array is actually passed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["array_namespace", "is_device", "to_host"]


def is_device(array) -> bool:
    """True for a CuPy array."""
    return type(array).__module__.partition(".")[0] == "cupy"


def array_namespace(*arrays):
    """``cupy`` if any argument is a CuPy array, else ``numpy``."""
    for array in arrays:
        if is_device(array):
            import cupy
            return cupy
    return np


def to_host(array):
    """A NumPy copy of a CuPy array; anything else unchanged."""
    return array.get() if is_device(array) else array
