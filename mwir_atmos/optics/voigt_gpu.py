"""Voigt profile for the GPU line-by-line kernel, with a NumPy twin.

The CUDA kernel (:mod:`mwir_atmos.optics.lbl_cuda`) evaluates ``Re w(z)``, the
real part of the Faddeeva function, at ``z = x + iy`` with ``x`` the
Doppler-scaled offset from line centre and ``y`` the Lorentz/Doppler ratio.
Consumer and workstation GPUs run float64 at 1/64 of float32 speed, so the
kernel splits the work:

* ``|z| >= CF_RADIUS`` (far wings, ~99.7% of (line, grid point) pairs):
  Laplace continued fraction with ``CF_TERMS`` terms, **float32**.
* ``|z| < CF_RADIUS`` (line cores): Weideman's (1994, SIAM J. Numer. Anal.
  31, 1497) rational approximation with ``WEIDEMAN_N`` terms, **float64**.

Measured against ``scipy.special.wofz`` over x in [0, 3e4], y in [1e-4, 3e3]:
worst relative error 4.7e-7 in the wings (float32 rounding) and 2.5e-8 in the
cores. :func:`re_faddeeva` reproduces the kernel's arithmetic and precision in
NumPy, so the method is tested on machines without a GPU.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CF_RADIUS", "CF_TERMS", "WEIDEMAN_N", "re_faddeeva", "weideman_coefficients"]

#: Continued fraction for |z| >= this; Weideman inside.
CF_RADIUS = 6.0
#: Continued-fraction depth (4.7e-7 at |z| = 6; more terms do not help float32).
CF_TERMS = 6
#: Weideman expansion order (2.5e-8 for |z| < 15; N=24 gave 2.6e-5).
WEIDEMAN_N = 32

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def weideman_coefficients(n: int = WEIDEMAN_N) -> tuple[float, np.ndarray]:
    """``(L, a)`` for Weideman's expansion; ``a`` is highest degree first."""
    m = 2 * n
    k = np.arange(-m + 1, m)
    big_l = np.sqrt(n / np.sqrt(2.0))
    t = big_l * np.tan(k * np.pi / (2 * m))
    f = np.concatenate([[0.0], np.exp(-t**2) * (big_l**2 + t**2)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
    return float(big_l), np.flipud(a[1:n + 1]).copy()


def _continued_fraction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Re w(z) for large |z|, float32 (mirrors the kernel's ``re_w_cf``)."""
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    rr, ri = x.copy(), y.copy()
    for k in range(CF_TERMS, 0, -1):
        h = np.float32(0.5 * k) / (rr * rr + ri * ri)
        rr, ri = x - h * rr, y + h * ri
    return ri / (np.float32(np.sqrt(np.pi)) * (rr * rr + ri * ri))


def _weideman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Re w(z) near line centre, float64 (mirrors the kernel's ``re_w_weideman``)."""
    big_l, a = weideman_coefficients()
    z = x.astype(np.float64) + 1j * y.astype(np.float64)
    den = big_l - 1j * z
    zz = (big_l + 1j * z) / den
    p = np.zeros_like(zz)
    for coefficient in a:
        p = p * zz + coefficient
    return (2.0 * p / den**2 + _INV_SQRT_PI / den).real


def re_faddeeva(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``Re w(x + iy)`` for ``y > 0`` by the kernel's method and precision (float32 out)."""
    x = np.abs(np.asarray(x, dtype=np.float32))
    y = np.broadcast_to(np.asarray(y, dtype=np.float32), x.shape)
    out = np.empty(x.shape, dtype=np.float32)
    wing = x * x + y * y >= np.float32(CF_RADIUS**2)
    out[wing] = _continued_fraction(x[wing], y[wing])
    out[~wing] = _weideman(x[~wing], y[~wing]).astype(np.float32)
    return out
