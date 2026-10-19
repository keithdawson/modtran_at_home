"""Rayleigh (molecular) scattering optical depth.

Physically all but irrelevant in the MWIR: Rayleigh cross-section scales as
lambda^-4, so going from the visible (~0.5 um) to 3-5 um cuts it by a factor
of ~(0.5/4)^4 ~ 2e-4. It is nonetheless computed with the standard molecular-
scattering formula (not a fudge factor) because ``LayerOptics.tau_rayleigh``
is a required field and the radiance model sums it into total scattering
optical depth regardless of magnitude.

Formula (King-factor-corrected Rayleigh cross-section of air):

    sigma(lambda) = 24*pi^3*(n(lambda)^2-1)^2 / (lambda^4 * N_s^2 * (n(lambda)^2+2)^2) * F_king

References
----------
Bodhaine, B.A., Wood, N.B., Dutton, E.G., Slusser, J.R. (1999), "On Rayleigh
Optical Depth Calculations," J. Atmos. Oceanic Technol., 16, 1854-1861.
  -- cross-section and wavelength-dependent N2/O2 King factors.
Peck, E.R. & Reeder, K. (1972), "Dispersion of air," J. Opt. Soc. Am. 62,
  958-962 -- refractive index of standard air, n(lambda), used here.

The refractive index is referenced to 288.15 K, so its matching molecular
density is 2.54743e19 cm^-3 (Bodhaine et al.), not the 273.15 K Loschmidt
number. Published cross-section comparisons: docs/physics_validation.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .._array import array_namespace
from ..types import PathLayer, SpectralGrid

__all__ = ["rayleigh_optical_depth"]

# Standard dry air at 288.15 K, matching the refractive-index reference.
_REFERENCE_DENSITY_CM3 = 2.54743e19


def _king_factor(wavelength_um: np.ndarray) -> np.ndarray:
    """Bodhaine et al. (1999), dry air with 300 ppm CO2."""
    inv2 = 1.0 / wavelength_um**2
    nitrogen = 1.034 + 3.17e-4*inv2
    oxygen = 1.096 + 1.385e-3*inv2 + 1.448e-4*inv2**2
    return (78.084*nitrogen + 20.946*oxygen + 0.934 + 0.03*1.15) / 99.994


# Peck & Reeder (1972) dispersion formula coefficients for standard air.
_PR_A = 8060.51
_PR_B = 2480990.0
_PR_C = 132.274
_PR_D = 17455.7
_PR_E = 39.32957


def _refractive_index_minus_1(wavelength_um: np.ndarray) -> np.ndarray:
    """(n - 1) for standard air, Peck & Reeder (1972)."""
    sigma2 = (1.0 / wavelength_um) ** 2  # (um^-1)^2, "sigma" is their wavenumber-in-um^-1
    return 1.0e-8 * (_PR_A + _PR_B / (_PR_C - sigma2) + _PR_D / (_PR_E - sigma2))


def _rayleigh_cross_section_cm2(wavelength_um: np.ndarray) -> np.ndarray:
    """Rayleigh scattering cross-section per air molecule, cm^2."""
    n = 1.0 + _refractive_index_minus_1(wavelength_um)
    n2m1 = n * n - 1.0
    lam_cm = wavelength_um * 1.0e-4
    return (
        (24.0 * np.pi**3 * n2m1**2)
        / (lam_cm**4 * _REFERENCE_DENSITY_CM3**2 * (n * n + 2.0) ** 2)
        * _king_factor(wavelength_um)
    )


def rayleigh_optical_depth(layers: Sequence[PathLayer], grid: SpectralGrid) -> np.ndarray:
    """Per-layer, per-wavenumber Rayleigh scattering optical depth.

    ``tau = sigma_Rayleigh(lambda) * air_column_cm2``, shape
    ``(len(layers), grid.n)``. Pure scattering (no absorption component) --
    contributes only to ``LayerOptics.tau_rayleigh``.
    """
    sigma = _rayleigh_cross_section_cm2(grid.wavelength_um)  # (n_wn,)
    xp = array_namespace(sigma)
    air_col = xp.asarray(np.array([layer.air_column_cm2 for layer in layers],
                                  dtype=np.float64))[:, None]
    return air_col * sigma[None, :]
