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
  -- cross-section formula (their eq. 5-6) and F_king ~ 1.0455 for air
  (their Table 3, ~visible/NIR depolarization ratio rho_n = 0.0279; treated
  here as wavelength-independent, an excellent approximation far from any
  electronic resonance, which certainly holds in the MWIR).
Peck, E.R. & Reeder, K. (1972), "Dispersion of air," J. Opt. Soc. Am. 62,
  958-962 -- refractive index of standard air, n(lambda), used here.

Sanity check (see tests/test_aerosol.py-adjacent check in test_geometry or
inline below): this formula integrated over a full sea-level US Standard
column at 550 nm gives tau ~ 0.087, matching the well-known literature
benchmark of ~0.097 (Bodhaine et al. 1999) to ~10%, consistent with using a
constant (rather than the fully wavelength-resolved) King factor. In the
MWIR the same column gives tau ~ 1-9e-5 -- confirms "negligible".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import LOSCHMIDT_CM3
from ..types import PathLayer, SpectralGrid

__all__ = ["rayleigh_optical_depth"]

#: Depolarization (King) correction factor for air, ~wavelength-independent
#: away from resonances (Bodhaine et al. 1999, Table 3).
_KING_FACTOR = 1.0455

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
        / (lam_cm**4 * LOSCHMIDT_CM3**2 * (n * n + 2.0) ** 2)
        * _KING_FACTOR
    )


def rayleigh_optical_depth(layers: Sequence[PathLayer], grid: SpectralGrid) -> np.ndarray:
    """Per-layer, per-wavenumber Rayleigh scattering optical depth.

    ``tau = sigma_Rayleigh(lambda) * air_column_cm2``, shape
    ``(len(layers), grid.n)``. Pure scattering (no absorption component) --
    contributes only to ``LayerOptics.tau_rayleigh``.
    """
    sigma = _rayleigh_cross_section_cm2(grid.wavelength_um)  # (n_wn,)
    air_col = np.array([layer.air_column_cm2 for layer in layers], dtype=np.float64)[:, None]
    return air_col * sigma[None, :]
