"""Tests for mwir_atmos.optics.aerosol and .rayleigh.

Physically meaningful checks per CONTRACT.md: visibility-driven optical
depth via the Koschmieder relation, wavelength dependence across 3-5 um,
RH-driven hygroscopic growth, the navy-maritime wind-speed dependence,
absorption/scattering correctly separated with absorption dominant in the
MWIR, and a sanity check that Rayleigh scattering is indeed negligible there.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mwir_atmos.atmosphere.profiles import get_profile
from mwir_atmos.geometry.path import build_path
from mwir_atmos.optics.aerosol import aerosol_optical_depth
from mwir_atmos.optics.rayleigh import rayleigh_optical_depth
from mwir_atmos.spectral import band_integrate, make_grid
from mwir_atmos.types import AerosolModel, ModelAtmosphere, SpectralGrid


@pytest.fixture(scope="module")
def grid() -> SpectralGrid:
    return make_grid()


@pytest.fixture(scope="module")
def layers(grid):
    profile = get_profile(ModelAtmosphere.MIDLAT_SUMMER)
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=20)
    return path.layers


def _total_tau(layers, grid, model, **kwargs):
    ta, ts, g = aerosol_optical_depth(layers, grid, model, **kwargs)
    return ta, ts, g


# =========================================================================
# Visibility (Koschmieder relation)
# =========================================================================


@pytest.mark.parametrize("model", [AerosolModel.RURAL, AerosolModel.URBAN, AerosolModel.DESERT])
def test_optical_depth_increases_as_visibility_decreases(layers, grid, model):
    vis_values = [50.0, 23.0, 10.0, 5.0, 1.0]
    totals = []
    for vis in vis_values:
        ta, ts, _ = _total_tau(copy.deepcopy(layers), grid, model, visibility_km=vis)
        totals.append(float((ta + ts).sum()))
    assert all(a < b for a, b in zip(totals, totals[1:])), (vis_values, totals)


def test_visibility_must_be_positive(layers, grid):
    with pytest.raises(ValueError):
        aerosol_optical_depth(layers, grid, AerosolModel.RURAL, visibility_km=0.0)


# =========================================================================
# Wavelength dependence (Angstrom law across 3-5 um)
# =========================================================================


def test_extinction_decreases_with_wavelength_for_fine_aerosol(layers, grid):
    """Rural/urban/tropospheric aerosol is fine-mode dominated -> strong
    Angstrom falloff -> more extinction at 3um than 5um."""
    ta, ts, _ = _total_tau(copy.deepcopy(layers), grid, AerosolModel.RURAL, visibility_km=10.0)
    tau_total = (ta + ts).sum(axis=0)
    idx_3um = int(np.argmin(np.abs(grid.wavelength_um - 3.0)))
    idx_5um = int(np.argmin(np.abs(grid.wavelength_um - 5.0)))
    assert tau_total[idx_3um] > tau_total[idx_5um]


def test_maritime_and_desert_are_spectrally_greyer_than_rural(layers, grid):
    """Large-particle maritime/desert aerosol should have a weaker 3-5um
    contrast (smaller Angstrom exponent) than fine-mode rural aerosol."""

    def contrast(model):
        ta, ts, _ = _total_tau(copy.deepcopy(layers), grid, model, visibility_km=10.0)
        tau_total = (ta + ts).sum(axis=0)
        idx_3um = int(np.argmin(np.abs(grid.wavelength_um - 3.0)))
        idx_5um = int(np.argmin(np.abs(grid.wavelength_um - 5.0)))
        return tau_total[idx_3um] / tau_total[idx_5um]

    rural_contrast = contrast(AerosolModel.RURAL)
    assert contrast(AerosolModel.NAVY_MARITIME) < rural_contrast
    assert contrast(AerosolModel.DESERT) < rural_contrast


# =========================================================================
# RH-driven hygroscopic growth
# =========================================================================


def test_hygroscopic_growth_increases_extinction_with_rh(layers, grid):
    totals = []
    for rh in (10.0, 50.0, 80.0, 95.0):
        modified = copy.deepcopy(layers)
        for layer in modified:
            layer.rh_pct = rh
        ta, ts, _ = _total_tau(modified, grid, AerosolModel.NAVY_MARITIME, visibility_km=23.0)
        totals.append(float((ta + ts).sum()))
    assert all(a < b for a, b in zip(totals, totals[1:])), totals


def test_desert_dust_is_weakly_hygroscopic(layers, grid):
    """Desert mineral dust should grow much less with RH than sea-salt-
    dominated maritime aerosol (Hanel gamma near zero vs ~0.6)."""

    def growth_ratio(model):
        dry = copy.deepcopy(layers)
        for layer in dry:
            layer.rh_pct = 10.0
        wet = copy.deepcopy(layers)
        for layer in wet:
            layer.rh_pct = 90.0
        ta_dry, ts_dry, _ = _total_tau(dry, grid, model, visibility_km=23.0)
        ta_wet, ts_wet, _ = _total_tau(wet, grid, model, visibility_km=23.0)
        return float((ta_wet + ts_wet).sum() / (ta_dry + ts_dry).sum())

    assert growth_ratio(AerosolModel.DESERT) < growth_ratio(AerosolModel.NAVY_MARITIME)


# =========================================================================
# Navy-maritime wind speed
# =========================================================================


def test_navy_wind_speed_increases_extinction(layers, grid):
    totals = []
    for wind in (1.0, 5.0, 10.0, 20.0):
        ta, ts, _ = _total_tau(
            copy.deepcopy(layers), grid, AerosolModel.NAVY_MARITIME, visibility_km=23.0, wind_speed_m_s=wind
        )
        totals.append(float((ta + ts).sum()))
    assert all(a < b for a, b in zip(totals, totals[1:])), totals


def test_wind_speed_has_no_effect_on_non_maritime_models(layers, grid):
    ta_lo, ts_lo, _ = _total_tau(
        copy.deepcopy(layers), grid, AerosolModel.RURAL, visibility_km=23.0, wind_speed_m_s=1.0
    )
    ta_hi, ts_hi, _ = _total_tau(
        copy.deepcopy(layers), grid, AerosolModel.RURAL, visibility_km=23.0, wind_speed_m_s=20.0
    )
    assert np.array_equal(ta_lo, ta_hi)
    assert np.array_equal(ts_lo, ts_hi)


# =========================================================================
# Absorption vs scattering split, asymmetry parameter
# =========================================================================


@pytest.mark.parametrize(
    "model",
    [AerosolModel.RURAL, AerosolModel.URBAN, AerosolModel.NAVY_MARITIME, AerosolModel.DESERT,
     AerosolModel.TROPOSPHERIC],
)
def test_absorption_dominates_scattering_in_mwir(layers, grid, model):
    ta, ts, _ = _total_tau(copy.deepcopy(layers), grid, model, visibility_km=15.0)
    assert ta.sum() > ts.sum()
    single_scattering_albedo = ts.sum() / (ta.sum() + ts.sum())
    assert single_scattering_albedo < 0.5


def test_asymmetry_parameter_in_physical_range(layers, grid):
    _, _, g = _total_tau(copy.deepcopy(layers), grid, AerosolModel.RURAL, visibility_km=15.0)
    assert np.all(g >= 0.0) and np.all(g <= 1.0)


def test_aerosol_none_model_is_all_zero(layers, grid):
    working = copy.deepcopy(layers)
    ta, ts, g = aerosol_optical_depth(working, grid, AerosolModel.NONE, visibility_km=1.0)
    assert np.all(ta == 0.0)
    assert np.all(ts == 0.0)
    assert np.all(g == 0.0)
    assert all(l.aerosol_od_550nm == 0.0 for l in working)


def test_aerosol_od_550nm_written_back_onto_layers(layers, grid):
    working = copy.deepcopy(layers)
    aerosol_optical_depth(working, grid, AerosolModel.URBAN, visibility_km=5.0)
    assert any(l.aerosol_od_550nm > 0.0 for l in working)
    assert all(l.aerosol_od_550nm >= 0.0 for l in working)


def test_output_shapes(layers, grid):
    ta, ts, g = aerosol_optical_depth(copy.deepcopy(layers), grid, AerosolModel.RURAL)
    assert ta.shape == (len(layers), grid.n)
    assert ts.shape == (len(layers), grid.n)
    assert g.shape == (len(layers), grid.n)


# =========================================================================
# Rayleigh scattering
# =========================================================================


def test_rayleigh_shape_and_positivity(layers, grid):
    tau = rayleigh_optical_depth(layers, grid)
    assert tau.shape == (len(layers), grid.n)
    assert np.all(tau > 0.0)


def test_rayleigh_negligible_compared_to_typical_aerosol(layers, grid):
    """At the contract's default 23km visibility, MWIR Rayleigh optical
    depth should be a small fraction (order 1%, not comparable) of the
    aerosol optical depth -- confirmed via a direct standalone check against
    the literature 550nm benchmark in rayleigh.py's module docstring; here
    we just check the two are not remotely the same order of magnitude."""
    tau_ray = rayleigh_optical_depth(layers, grid).sum(axis=0)
    ta, ts, _ = _total_tau(copy.deepcopy(layers), grid, AerosolModel.RURAL, visibility_km=23.0)
    tau_aer = (ta + ts).sum(axis=0)
    assert np.all(tau_ray < 0.05 * np.maximum(tau_aer, 1.0e-6))


def test_rayleigh_decreases_with_wavelength():
    """lambda^-4 scaling: tau at the short-wavelength (high wavenumber) end
    of the grid should exceed tau at the long-wavelength end."""
    grid_local = make_grid()
    profile = get_profile(ModelAtmosphere.US_STANDARD)
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=10)
    tau = rayleigh_optical_depth(path.layers, grid_local).sum(axis=0)
    assert tau[-1] > tau[0]  # grid is increasing wavenumber == decreasing wavelength
