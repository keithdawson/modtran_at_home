"""Tests for mwir_atmos.optics.continuum: the MT_CKD-style H2O self+foreign
continuum, and the (documented no-op) CO2 continuum stub.

Offline: reads data/hitran/mt_ckd_h2o_v4.3.npz, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.constants import LOSCHMIDT_CM3, STD_PRESSURE_MBAR, STD_TEMP_K
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics.continuum import co2_continuum_optical_depth, h2o_continuum_optical_depth
from mwir_atmos.optics.lbl import molecular_optical_depth
from mwir_atmos.spectral import band_average, make_grid
from mwir_atmos.testing import make_synthetic_profile
from mwir_atmos.types import PathLayer, Species

LINELISTS = ensure_linelists()


def _horizontal_layer(ds_km: float) -> PathLayer:
    prof = make_synthetic_profile()
    p0 = float(prof.pressure_mbar[0])
    t0 = float(prof.temperature_k[0])
    n_air0 = LOSCHMIDT_CM3 * (p0 / STD_PRESSURE_MBAR) * (STD_TEMP_K / t0)
    air_col = n_air0 * ds_km * 1.0e5

    column, p_eff, t_eff = {}, {}, {}
    for species, vmr_profile in prof.vmr_ppmv.items():
        vmr = float(vmr_profile[0]) * 1.0e-6
        column[species] = air_col * vmr
        p_eff[species] = p0
        t_eff[species] = t0

    return PathLayer(
        index=0, z_bottom_km=0.0, z_top_km=0.0, ds_km=ds_km,
        p_mbar=p0, t_k=t0, rh_pct=50.0,
        air_column_cm2=air_col, column_cm2=column, p_eff_mbar=p_eff, t_eff_k=t_eff,
    )


# =========================================================================
# Shape / basic contract
# =========================================================================


def test_h2o_continuum_shape():
    grid = make_grid()
    layers = [_horizontal_layer(1.0), _horizontal_layer(5.0)]
    tau = h2o_continuum_optical_depth(layers, grid)
    assert tau.shape == (2, grid.n)
    assert np.all(np.isfinite(tau))
    assert np.all(tau >= 0.0)


def test_co2_continuum_is_documented_zero_stub():
    grid = make_grid()
    layers = [_horizontal_layer(1.0), _horizontal_layer(5.0)]
    tau = co2_continuum_optical_depth(layers, grid)
    assert tau.shape == (2, grid.n)
    assert np.all(tau == 0.0)


def test_h2o_continuum_zero_when_no_h2o():
    grid = make_grid()
    layer = _horizontal_layer(1.0)
    layer.column_cm2[Species.H2O] = 0.0
    tau = h2o_continuum_optical_depth([layer], grid)
    assert np.all(tau == 0.0)


# =========================================================================
# Physics: monotonicity and relative magnitude
# =========================================================================


def test_h2o_continuum_increases_with_path_length():
    grid = make_grid()
    lengths = [1.0, 5.0, 25.0]
    layers = [_horizontal_layer(ds) for ds in lengths]
    tau = h2o_continuum_optical_depth(layers, grid)
    window = (2000.0, 2222.22)  # 4.5-5.0 um, where the continuum matters most
    band_tau = np.array([band_average(grid, tau[i], window) for i in range(len(lengths))])
    assert np.all(np.diff(band_tau) > 0), "continuum optical depth must grow with path length"


def test_h2o_continuum_stronger_toward_longer_wavelength():
    """MT_CKD's H2O self continuum in this range is markedly stronger
    toward 5 um than toward 4 um (see the reference coefficients cited in
    continuum.py: self_absco_ref at 2000 cm^-1 is ~20x that at 2500 cm^-1)."""
    grid = make_grid()
    layer = _horizontal_layer(10.0)
    tau = h2o_continuum_optical_depth([layer], grid)
    tau_5um = band_average(grid, tau[0], (2000.0, 2100.0))
    tau_4um = band_average(grid, tau[0], (2450.0, 2550.0))
    assert tau_5um > tau_4um


# =========================================================================
# Physics: the continuum must matter (the job's core concern)
# =========================================================================


def test_continuum_measurably_reduces_window_transmittance_on_long_path():
    """"H2O continuum is essential -- without it long horizontal paths in
    the 4-5 um window come out far too transparent." Verify directly: a 25
    km sea-level horizontal path's window transmittance must be
    meaningfully lower with the continuum included than without."""
    grid = make_grid()
    layer = _horizontal_layer(25.0)

    tau_with, _ = molecular_optical_depth([layer], grid, linelists=LINELISTS, include_continuum=True)
    tau_without, _ = molecular_optical_depth([layer], grid, linelists=LINELISTS, include_continuum=False)

    window = (2000.0, 2222.22)  # 4.5-5.0 um
    t_with = band_average(grid, np.exp(-tau_with[0]), window)
    t_without = band_average(grid, np.exp(-tau_without[0]), window)

    assert t_with < t_without
    # Not just a rounding-level difference -- the job explicitly calls this
    # "far too transparent" without the continuum, so demand a clear effect.
    relative_drop = (t_without - t_with) / t_without
    assert relative_drop > 0.10, f"continuum only changed window transmittance by {relative_drop:.1%}"


def test_h2o_continuum_uses_effective_state_not_bulk_layer():
    """Continuum must key off PathLayer.p_eff_mbar/t_eff_k[H2O], not just
    p_mbar/t_k -- construct a layer where they differ and confirm the
    continuum responds to the effective values."""
    grid = make_grid()
    layer = _horizontal_layer(10.0)
    # Perturb only the H2O effective temperature away from the bulk t_k.
    layer.t_eff_k[Species.H2O] = layer.t_k + 30.0

    tau_perturbed = h2o_continuum_optical_depth([layer], grid)

    layer.t_eff_k[Species.H2O] = layer.t_k
    tau_baseline = h2o_continuum_optical_depth([layer], grid)

    assert not np.allclose(tau_perturbed, tau_baseline)


def test_h2o_continuum_matches_manual_formula_at_native_knots():
    """Check high-frequency scaling at native knots without interpolation.

    Independent off-knot Fortran reference tests live in test_physics_references.
    """
    from mwir_atmos.optics.continuum import _load_coefficients

    grid = make_grid(2500.0, 4300.0, 10.0)
    layer = _horizontal_layer(10.0)
    tau = h2o_continuum_optical_depth([layer], grid)

    coeffs = _load_coefficients()
    nu = grid.wavenumber_cm1
    self_ref = np.interp(nu, coeffs["wavenumber_cm1"], coeffs["self_absco_ref"])
    for_ref = np.interp(nu, coeffs["wavenumber_cm1"], coeffs["for_absco_ref"])
    texp = np.interp(nu, coeffs["wavenumber_cm1"], coeffs["self_texp"])
    p_ref, t_ref = coeffs["ref_press_mbar"], coeffs["ref_temp_k"]

    h2o_col = layer.column_cm2[Species.H2O]
    p_eff = layer.p_eff_mbar[Species.H2O]
    t_eff = layer.t_eff_k[Species.H2O]
    h2o_vmr = h2o_col / layer.air_column_cm2

    rho_rat = (p_eff / p_ref) * (t_ref / t_eff)
    self_coeff = self_ref * (t_ref / t_eff) ** texp * h2o_vmr * rho_rat
    for_coeff = for_ref * (1.0 - h2o_vmr) * rho_rat
    radiation = nu  # AER MYRADFN high-frequency branch, c2*nu/T > 10
    expected = (self_coeff + for_coeff) * radiation * h2o_col

    np.testing.assert_allclose(tau[0], expected, rtol=1e-10)
