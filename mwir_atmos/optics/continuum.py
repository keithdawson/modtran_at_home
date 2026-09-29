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

from .._array import array_namespace, to_host
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
    j, weights = _stencil(grid.wavenumber_cm1)
    self_native, foreign_native = _native_coefficients(pressure_mbar, temperature_k, h2o_vmr)
    return _interpolate(self_native, j, weights), _interpolate(foreign_native, j, weights)


def _native_coefficients(pressure_mbar: float, temperature_k: float,
                         h2o_vmr: float) -> tuple[np.ndarray, np.ndarray]:
    """Self/foreign cross-sections on AER's native grid, radiation term included (MYRADFN)."""
    c = _load_coefficients()
    native = c["wavenumber_cm1"]
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
    return self_native, foreign_native


def _stencil(wavenumber_cm1):
    """``(j, weights)``: AER's four-point cubic-Hermite stencil (MYXINT) at ``wavenumber_cm1``.

    Native points ``j-1 .. j+2`` carry ``weights[0..3]``. Depends only on the
    grid, so it is computed once for all layers. NumPy or CuPy, following
    ``wavenumber_cm1``.
    """
    xp = array_namespace(wavenumber_cm1)
    native = _load_coefficients()["wavenumber_cm1"]
    step = native[1] - native[0]
    position = (wavenumber_cm1 - native[0]) / step
    j = xp.floor(position + 0.001).astype(xp.int64)  # AER's ONEPL=1.001
    if bool(xp.any((j < 1) | (j + 2 >= len(native)))):
        raise ValueError("Spectral grid exceeds the MT_CKD four-point interpolation range")
    p = position - j
    cubic = (3.0 - 2.0*p)*p*p
    b = 0.5*p*(1.0-p)
    b1, b2 = b*(1.0-p), b*p
    return j, (-b1, 1.0-cubic+b2, cubic+b1, -b2)


def _interpolate(values, j, weights):
    """Interpolate native-grid ``values`` (along the last axis) with a :func:`_stencil`.

    AER's ``-a[j-1]*b1 + a[j]*(1-cubic+b2) + a[j+1]*(cubic+b1) - a[j+2]*b2``.
    Every grid point between two native points shares the same four values,
    so each run of equal ``j`` is one small matrix product
    ``values[..., j-1:j+3] @ weights[:, run]`` (12x faster than gathering the
    four terms for a fine grid; the sum is in BLAS order).
    """
    xp = array_namespace(j)
    values = xp.asarray(values)
    rows = values.reshape(-1, values.shape[-1])
    out = xp.empty((rows.shape[0], j.size), dtype=np.float64)
    if j.size:
        stacked = xp.stack(weights)
        j_host = to_host(j)
        breaks = np.flatnonzero(np.diff(j_host)) + 1
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, j_host.size]):
            k = int(j_host[a])
            xp.matmul(rows[:, k - 1:k + 3], stacked[:, a:b], out=out[:, a:b])
    return out.reshape(values.shape[:-1] + (j.size,))


def h2o_continuum_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
) -> np.ndarray:
    """MT_CKD-style H2O self + foreign continuum optical depth.

    Returns shape ``(n_layers, grid.n)``. Layers with zero H2O column
    contribute zero. Uses ``PathLayer.p_eff_mbar[H2O]`` /
    ``PathLayer.t_eff_k[H2O]`` (Curtis-Godson effective state), consistent
    with :mod:`mwir_atmos.optics.lbl`.

    Self and foreign coefficients are summed on the native grid and
    interpolated once, for all layers together. A CuPy grid gives a CuPy result.
    """
    xp = array_namespace(grid.wavenumber_cm1)
    n_layers = len(layers)
    rows, columns, natives = [], [], []
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
        self_native, foreign_native = _native_coefficients(p_eff, t_eff, h2o_vmr)
        rows.append(row)
        columns.append(h2o_col)
        natives.append(self_native + foreign_native)

    if not rows:
        return xp.zeros((n_layers, grid.n), dtype=np.float64)
    j, weights = _stencil(grid.wavenumber_cm1)
    values = _interpolate(np.array(natives), j, weights)
    values *= xp.asarray(np.array(columns))[:, None]
    if len(rows) == n_layers:
        return values
    tau = xp.zeros((n_layers, grid.n), dtype=np.float64)
    tau[xp.asarray(rows)] = values
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
    return array_namespace(grid.wavenumber_cm1).zeros((n_layers, grid.n), dtype=np.float64)
