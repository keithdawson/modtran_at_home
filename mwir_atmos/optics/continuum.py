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
cache. Temperature/density scaling and the radiation term are applied on
AER's native grid before its four-point cubic-Hermite interpolation. The
radiation constant and small/large argument branches follow MT_CKD 4.3
exactly for reproducibility against the original Fortran implementation.
Both continuum coefficients multiply the H2O column, in molecules/cm^2.

The local H2O line calculation must subtract the Lorentz pedestal at
+/-25 cm^-1, which is already included in these continuum coefficients.
See https://hitran.org/mtckd/ and PHYSICS_VALIDATION.md.

Copyright (c) Atmospheric and Environmental Research, Inc., 2022.
The translated interpolation and radiation routines are subject to the
AER notice reproduced in AER_LICENSE.txt alongside this module.

"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import PathLayer, Species, SpectralGrid

__all__ = ["h2o_continuum_optical_depth", "co2_continuum_optical_depth"]

from ..data_paths import hitran_data_dir

_COEFF_PATH = hitran_data_dir() / "mt_ckd_h2o_v4.3.npz"

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


def _mt_ckd_coefficients(grid: SpectralGrid, pressure_mbar: float,
                         temperature_k: float, h2o_vmr: float) -> tuple[np.ndarray, np.ndarray]:
    """Self/foreign absorption cross-sections (cm^2/molecule), AER MT_CKD 4.3.

    Matches MYRADFN/MYXINT, including AER's historical radiation constant.
    No extrapolation beyond the native table's complete four-point stencil.
    """
    c = _load_coefficients()
    native = c["wavenumber_cm1"]
    step = native[1] - native[0]
    position = (grid.wavenumber_cm1 - native[0]) / step
    j = np.floor(position + 0.001).astype(int)  # AER's ONEPL=1.001
    if np.any((j < 1) | (j + 2 >= len(native))):
        raise ValueError("Spectral grid exceeds the MT_CKD four-point interpolation range")
    p = position - j
    cubic = (3.0 - 2.0*p)*p*p
    b = 0.5*p*(1.0-p)
    b1, b2 = b*(1.0-p), b*p

    x = native * 1.4387752 / temperature_k
    radiation = native.copy()
    small = x <= 0.01
    medium = (x > 0.01) & (x <= 10.0)
    radiation[small] = 0.5*x[small]*native[small]
    ex = np.exp(-x[medium])
    radiation[medium] = native[medium]*(1.0-ex)/(1.0+ex)
    ratio = c["ref_temp_k"] / temperature_k
    rho = pressure_mbar / c["ref_press_mbar"] * ratio
    self_native = c["self_absco_ref"] * ratio**c["self_texp"] * h2o_vmr * rho * radiation
    foreign_native = c["for_absco_ref"] * (1.0-h2o_vmr) * rho * radiation

    def interpolate(a):
        return (-a[j-1]*b1 + a[j]*(1.0-cubic+b2)
                + a[j+1]*(cubic+b1) - a[j+2]*b2)

    return interpolate(self_native), interpolate(foreign_native)


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

        self_coeff, for_coeff = _mt_ckd_coefficients(grid, p_eff, t_eff, h2o_vmr)
        tau[row] = (self_coeff + for_coeff) * h2o_col

    return tau


def co2_continuum_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
) -> np.ndarray:
    """CO2 collision-induced/continuum absorption.

    Returns zeros: CO2 continuum, collision-induced absorption and line
    mixing are not implemented. Opacity in the band core does not bound
    errors in band wings; the size of the omission is not validated here.
    A future implementation requires a compatible published coefficient
    set and line-wing convention, with independent spectral benchmarks.
    """
    n_layers = len(layers)
    return np.zeros((n_layers, grid.n), dtype=np.float64)
