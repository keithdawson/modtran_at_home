"""H2O self + foreign continuum absorption.

Away from individual line centres, water vapour absorbs far more than the
sum of Voigt line wings predicts (collision-induced absorption and the
far wings of very strong lines beyond any reasonable per-line cutoff). This
"continuum" is essential in the MWIR atmospheric windows (3.7-4.0 um,
4.5-5.0 um): without it, long horizontal paths come out far too transparent
there even though the line-by-line engine (:mod:`mwir_atmos.optics.lbl`) is
doing everything right.

Data source: MT_CKD (Mlawer-Tobin-Clough-Kneizys-Davies) water vapor
continuum model, version 4.3, (c) Atmospheric and Environmental Research,
Inc. (AER) -- distributed for scientific/research use, copyright notice and
AER acknowledgment reproduced here per that license (see
https://github.com/AER-RC/MT_CKD_H2O/blob/master/LICENSE.md). General
reference: Mlawer, E.J. et al. (2012), "Development and recent evaluation
of the MT_CKD model of continuum absorption", Phil. Trans. R. Soc. A 370,
doi:10.1098/rsta.2011.0295. The MT_CKD_4.3 release notes (Jul 2023) describe
revisions to the H2O foreign continuum above 4000 cm^-1 informed by the
Campargue group's laboratory measurements.

The reference coefficients (self- and foreign-broadened absorption at
P=1013 mb, T=296 K, plus a per-wavenumber self-continuum temperature
exponent) were downloaded once during development (2026-08-02) from
https://github.com/AER-RC/MT_CKD_H2O/blob/master/data/absco-ref_wv-mt-ckd.nc
(a small, ~2000-point, 10 cm^-1-resolution NetCDF-classic table -- not the
full per-molecule cross-section database) and re-saved as plain arrays in
``data/hitran/mt_ckd_h2o_v4.3.npz`` (keys ``wavenumber_cm1``,
``self_absco_ref``, ``for_absco_ref``, ``self_texp``, ``ref_press_mbar``,
``ref_temp_k``). This module never touches the network; it only reads that
cache and linearly interpolates onto the working grid (a simplification of
MT_CKD's own cubic-Hermite interpolation -- reasonable given the continuum
is smooth by construction and differences between continuum model versions
are themselves 10s of percent).

Physics (transcribed from AER's reference implementation,
``src/mt_ckd_h2o_module.f90`` in the repository above, and reproduced here
in vectorised numpy rather than a per-line Fortran loop):

    rho_rat(p, T)   = (p / p_ref) * (T_ref / T)                     # density ratio to reference
    self_coeff(nu)  = self_absco_ref(nu) * (T_ref/T)**self_texp(nu) * h2o_vmr * rho_rat
    for_coeff(nu)   = for_absco_ref(nu) * (1 - h2o_vmr) * rho_rat
    radiation(nu,T) = nu * tanh(c2*nu / (2T))                        # "radiation term"
    tau(nu)         = (self_coeff(nu) + for_coeff(nu)) * radiation(nu,T) * N_H2O

``radiation(nu, T)`` is algebraically identical to AER's piecewise
``myradfn`` (their branches at small/large ``x = c2*nu/T`` are just a
numerically-safe implementation of ``x -> tanh(x/2)``'s two limits;
``numpy.tanh`` is already numerically stable across that range, so no
piecewise branching is needed here). ``N_H2O`` is the water vapour column
(molecules/cm^2) -- note both the self *and* foreign terms are scaled by
the H2O column, not the air column: the continuum is H2O's own absorption,
merely broadened by either H2O-H2O or H2O-air collisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..constants import C2_RAD
from ..types import PathLayer, Species, SpectralGrid

__all__ = ["h2o_continuum_optical_depth", "co2_continuum_optical_depth"]

_COEFF_PATH = Path(__file__).resolve().parents[2] / "data" / "hitran" / "mt_ckd_h2o_v4.3.npz"

_coeff_cache: dict[str, object] | None = None


def _load_coefficients() -> dict[str, object]:
    global _coeff_cache
    if _coeff_cache is None:
        if not _COEFF_PATH.exists():
            raise FileNotFoundError(
                f"MT_CKD H2O continuum cache not found at {_COEFF_PATH}. This "
                "ships with the repo; see the mwir_atmos.optics.continuum module "
                "docstring for how it was built."
            )
        with np.load(_COEFF_PATH) as data:
            _coeff_cache = {
                "wavenumber_cm1": np.array(data["wavenumber_cm1"]),
                "self_absco_ref": np.array(data["self_absco_ref"]),
                "for_absco_ref": np.array(data["for_absco_ref"]),
                "self_texp": np.array(data["self_texp"]),
                "ref_press_mbar": float(data["ref_press_mbar"]),
                "ref_temp_k": float(data["ref_temp_k"]),
            }
    return _coeff_cache


def _interpolated_reference(grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """MT_CKD reference arrays interpolated onto ``grid`` (cached per-grid-size
    isn't done here; callers loop over layers, not grids, so this runs once
    per :func:`h2o_continuum_optical_depth` call)."""
    c = _load_coefficients()
    nu = grid.wavenumber_cm1
    self_ref = np.interp(nu, c["wavenumber_cm1"], c["self_absco_ref"])
    for_ref = np.interp(nu, c["wavenumber_cm1"], c["for_absco_ref"])
    texp = np.interp(nu, c["wavenumber_cm1"], c["self_texp"])
    return self_ref, for_ref, texp, c["ref_press_mbar"], c["ref_temp_k"]


def h2o_continuum_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
) -> np.ndarray:
    """MT_CKD-style H2O self + foreign continuum optical depth.

    Returns shape ``(n_layers, grid.n)``. Layers with zero H2O column
    contribute zero. Uses ``PathLayer.p_eff_mbar[H2O]`` /
    ``PathLayer.t_eff_k[H2O]`` (Curtis-Godson effective state), consistent
    with :mod:`mwir_atmos.optics.lbl`.
    """
    n_layers = len(layers)
    tau = np.zeros((n_layers, grid.n), dtype=np.float64)

    self_ref, for_ref, texp, p_ref, t_ref = _interpolated_reference(grid)
    nu = grid.wavenumber_cm1

    for row, layer in enumerate(layers):
        h2o_col = layer.column_cm2.get(Species.H2O, 0.0)
        if h2o_col <= 0.0:
            continue
        p_eff = layer.p_eff_mbar.get(Species.H2O, layer.p_mbar)
        t_eff = layer.t_eff_k.get(Species.H2O, layer.t_k)
        if t_eff <= 0.0:
            continue
        h2o_vmr = h2o_col / layer.air_column_cm2 if layer.air_column_cm2 > 0 else 0.0
        h2o_vmr = min(max(h2o_vmr, 0.0), 1.0)

        rho_rat = (p_eff / p_ref) * (t_ref / t_eff)
        self_coeff = self_ref * (t_ref / t_eff) ** texp * h2o_vmr * rho_rat
        for_coeff = for_ref * (1.0 - h2o_vmr) * rho_rat
        radiation = nu * np.tanh(C2_RAD * nu / (2.0 * t_eff))

        tau[row] = (self_coeff + for_coeff) * radiation * h2o_col

    return tau


def co2_continuum_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
) -> np.ndarray:
    """CO2 collision-induced/continuum absorption.

    Returns zeros. CO2's MWIR continuum (collision-induced absorption near
    the 4.3 um band, and the far line wings beyond the +/-25 cm^-1 cutoff
    used by :mod:`mwir_atmos.optics.lbl`) is a small correction relative to
    the CO2 line database already fetched for this package -- unlike the
    H2O continuum, it is not essential for the required test bands (the
    4.2-4.4 um opacity test is dominated by CO2 line absorption, which is
    already essentially opaque from lines alone). CONTRACT.md explicitly
    allows this function to return zeros; left as a documented stub rather
    than a fabricated coefficient table. A real implementation would use
    MT_CKD's CO2 continuum (also published by AER) the same way
    :func:`h2o_continuum_optical_depth` uses the H2O tables.
    """
    n_layers = len(layers)
    return np.zeros((n_layers, grid.n), dtype=np.float64)
