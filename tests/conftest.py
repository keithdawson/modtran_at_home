"""Shared test set-up.

The default backend is ``auto`` (the GPU when CuPy sees one). Physics
thresholds were set on the CPU engine, so the suite runs on the CPU unless a
test names a backend; ``MWIR_BACKEND`` set before pytest starts still wins.
"""
import os

os.environ.setdefault("MWIR_BACKEND", "cpu")
