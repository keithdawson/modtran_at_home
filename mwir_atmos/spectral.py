"""Spectral grid construction, Planck functions and band integration.

OVERSEER-OWNED. Worker agents must not edit this file -- both workers depend on
these helpers and a concurrent edit would break the other one. Missing helper?
Say so in your report.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .constants import (
    BANDS_CM1,
    C1_RAD,
    C2_RAD,
    DEFAULT_RESOLUTION_CM1,
    MWIR_GRID_MAX_CM1,
    MWIR_GRID_MIN_CM1,
)
from .types import SpectralGrid

__all__ = [
    "make_grid",
    "grid_from_wavelength_um",
    "resolution_um_to_cm1",
    "resolution_cm1_to_um",
    "resolution_cm1_for_wavelength_step",
    "planck",
    "planck_per_um",
    "dplanck_dt",
    "band_integrate",
    "band_average",
    "band_slice",
    "convolve_gaussian",
    "radiance_to_per_um",
]


# =========================================================================
# Grid construction
# =========================================================================


def make_grid(
    wavenumber_min_cm1: float = MWIR_GRID_MIN_CM1,
    wavenumber_max_cm1: float = MWIR_GRID_MAX_CM1,
    resolution_cm1: float = DEFAULT_RESOLUTION_CM1,
) -> SpectralGrid:
    """Build a uniform wavenumber grid, inclusive of both endpoints."""
    if wavenumber_max_cm1 <= wavenumber_min_cm1:
        raise ValueError("wavenumber_max_cm1 must exceed wavenumber_min_cm1")
    if resolution_cm1 <= 0:
        raise ValueError("resolution_cm1 must be positive")
    n = int(np.floor((wavenumber_max_cm1 - wavenumber_min_cm1) / resolution_cm1)) + 1
    nu = wavenumber_min_cm1 + resolution_cm1 * np.arange(n, dtype=np.float64)
    return SpectralGrid(wavenumber_cm1=nu, resolution_cm1=float(resolution_cm1))


def resolution_um_to_cm1(delta_um: float, wavelength_um: float) -> float:
    """Wavenumber step equivalent to a wavelength step at a given wavelength.

    ``dnu = 1e4 * dlambda / lambda^2``. Note the two are *not* interchangeable
    across a band: a fixed wavenumber step is a widening wavelength step as
    wavelength grows. Over 3-5 um a 1 cm^-1 grid spans 0.0009 um at the short
    end and 0.0025 um at the long end.
    """
    if wavelength_um <= 0 or delta_um <= 0:
        raise ValueError("delta_um and wavelength_um must be positive")
    return 1.0e4 * delta_um / (wavelength_um * wavelength_um)


def resolution_cm1_to_um(delta_cm1: float, wavelength_um: float) -> float:
    """Wavelength step equivalent to a wavenumber step at a given wavelength."""
    if wavelength_um <= 0 or delta_cm1 <= 0:
        raise ValueError("delta_cm1 and wavelength_um must be positive")
    return delta_cm1 * wavelength_um * wavelength_um / 1.0e4


def resolution_cm1_for_wavelength_step(
    delta_um: float, wavenumber_min_cm1: float, wavenumber_max_cm1: float
) -> float:
    """Wavenumber step guaranteeing a wavelength step no coarser than ``delta_um``.

    The wavelength step grows as lambda^2, so the binding constraint is the
    LONGEST wavelength in the band -- i.e. the lowest wavenumber. Sizing at band
    centre would leave the red end under-sampled.
    """
    if wavenumber_min_cm1 <= 0:
        raise ValueError("wavenumber_min_cm1 must be positive")
    longest_um = 1.0e4 / wavenumber_min_cm1
    return resolution_um_to_cm1(delta_um, longest_um)


def grid_from_wavelength_um(
    wavelength_min_um: float,
    wavelength_max_um: float,
    resolution_cm1: float = DEFAULT_RESOLUTION_CM1,
) -> SpectralGrid:
    """Build a wavenumber grid spanning a wavelength interval given in microns."""
    if wavelength_min_um <= 0 or wavelength_max_um <= wavelength_min_um:
        raise ValueError("require 0 < wavelength_min_um < wavelength_max_um")
    return make_grid(1.0e4 / wavelength_max_um, 1.0e4 / wavelength_min_um, resolution_cm1)


# =========================================================================
# Planck
# =========================================================================


def planck(wavenumber_cm1: np.ndarray, temperature_k: float | np.ndarray) -> np.ndarray:
    """Planck spectral radiance in W cm^-2 sr^-1 / cm^-1.

    ``wavenumber_cm1`` has shape ``(n,)``. ``temperature_k`` may be a scalar
    (result ``(n,)``) or shape ``(m,)`` (result ``(m, n)``, one row per
    temperature -- this is the form the layer thermal integration wants).

    Temperatures <= 0 yield zeros rather than raising, so that "no surface" and
    "no source" cases can be handled without branching.
    """
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    scalar_t = t.ndim == 0
    t2 = np.atleast_1d(t)[:, None]

    out = np.zeros((t2.shape[0], nu.size), dtype=np.float64)
    valid = (t2 > 0.0).ravel()
    if np.any(valid):
        x = C2_RAD * nu[None, :] / t2[valid]
        # expm1 keeps precision for the small-x (long wave) limit.
        out[valid] = C1_RAD * nu[None, :] ** 3 / np.expm1(x)
    return out[0] if scalar_t else out


def planck_per_um(wavelength_um: np.ndarray, temperature_k: float | np.ndarray) -> np.ndarray:
    """Planck spectral radiance in W m^-2 sr^-1 um^-1, for plotting/reporting."""
    lam = np.asarray(wavelength_um, dtype=np.float64)
    nu = 1.0e4 / lam
    b_nu = planck(nu, temperature_k)          # W cm^-2 sr^-1 / cm^-1
    return radiance_to_per_um(b_nu, nu)


def dplanck_dt(wavenumber_cm1: np.ndarray, temperature_k: float) -> np.ndarray:
    """dB/dT in W cm^-2 sr^-1 / cm^-1 / K -- useful for NEdT style figures."""
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    if temperature_k <= 0:
        return np.zeros_like(nu)
    x = C2_RAD * nu / temperature_k
    ex = np.exp(x)
    return C1_RAD * nu**3 * ex * x / (temperature_k * (ex - 1.0) ** 2)


def radiance_to_per_um(radiance_per_cm1: np.ndarray, wavenumber_cm1: np.ndarray) -> np.ndarray:
    """Convert W cm^-2 sr^-1 /cm^-1  ->  W m^-2 sr^-1 um^-1.

    The Jacobian is |dnu/dlambda| = nu^2 / 1e4 (per um), and 1 W cm^-2 = 1e4 W m^-2.
    """
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    return np.asarray(radiance_per_cm1, dtype=np.float64) * nu**2


# =========================================================================
# Band integration
# =========================================================================


def band_slice(grid: SpectralGrid, band: str | tuple[float, float]) -> slice:
    """Index slice selecting a named band (see ``constants.BANDS_CM1``) or range."""
    lo, hi = BANDS_CM1[band] if isinstance(band, str) else band
    idx = np.nonzero((grid.wavenumber_cm1 >= lo) & (grid.wavenumber_cm1 <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"band {band!r} does not overlap the grid")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def band_integrate(
    grid: SpectralGrid,
    spectrum: np.ndarray,
    band: str | tuple[float, float] = "MWIR_3_5um",
) -> float | np.ndarray:
    """Integrate a spectral quantity over a band (trapezoid, in wavenumber).

    W cm^-2 sr^-1 /cm^-1 integrates to W cm^-2 sr^-1. Accepts ``(n_wn,)`` or
    ``(..., n_wn)``; integration is always over the last axis.
    """
    sl = band_slice(grid, band)
    y = np.asarray(spectrum, dtype=np.float64)[..., sl]
    return np.trapezoid(y, x=grid.wavenumber_cm1[sl], axis=-1)


def band_average(
    grid: SpectralGrid,
    spectrum: np.ndarray,
    band: str | tuple[float, float] = "MWIR_3_5um",
) -> float | np.ndarray:
    """Band-mean value (e.g. mean in-band transmittance), unweighted."""
    sl = band_slice(grid, band)
    width = grid.wavenumber_cm1[sl][-1] - grid.wavenumber_cm1[sl][0]
    if width <= 0:
        return np.asarray(spectrum, dtype=np.float64)[..., sl].mean(axis=-1)
    return band_integrate(grid, spectrum, band) / width


# =========================================================================
# Instrument line shape
# =========================================================================


def convolve_gaussian(grid: SpectralGrid, spectrum: np.ndarray, fwhm_cm1: float) -> np.ndarray:
    """Degrade a spectrum to an instrument resolution with a Gaussian ILS.

    Operates on the last axis. ``fwhm_cm1`` <= the grid spacing is a no-op.
    Edges use reflective padding so band ends are not artificially darkened.
    """
    y = np.asarray(spectrum, dtype=np.float64)
    if fwhm_cm1 <= grid.resolution_cm1:
        return y.copy()

    sigma_pts = (fwhm_cm1 / grid.resolution_cm1) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = int(np.ceil(4.0 * sigma_pts))
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_pts) ** 2)
    kernel /= kernel.sum()

    flat = np.atleast_2d(y)
    padded = np.pad(flat, ((0, 0), (half, half)), mode="reflect")
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = np.convolve(padded[i], kernel, mode="valid")
    return out.reshape(y.shape)
