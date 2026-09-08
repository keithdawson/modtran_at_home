"""Optical turbulence (Cn^2, Fried parameter, isoplanatic angle, scintillation).

DEV-ALPHA owned (wave 3 / stretch goal). See :mod:`mwir_atmos.turbulence.cn2`
for the Cn^2(h) vertical profile models (HV5/7, SLC Day/Night, constant) and
:mod:`mwir_atmos.turbulence.metrics` for :func:`compute_turbulence`, the
entry point ``mwir_atmos.api._attach_turbulence`` calls.
"""

from __future__ import annotations

from .metrics import compute_turbulence

__all__ = ["compute_turbulence"]
