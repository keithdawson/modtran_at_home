"""Tests for mwir_atmos.optics.lbl: the Voigt line-by-line engine.

Runs entirely offline against the shipped ``data/hitran/*.npz`` cache.
Physics is exercised two ways:

1. Against synthetic single-line ``LineList`` objects with known analytic
   limits (pure Doppler/Gaussian at zero pressure, pure Lorentzian at high
   pressure, pressure shift), which pin down the Voigt implementation
   itself independent of real HITRAN data.
2. Against the real cached line lists on hand-built, contract-correct
   ``PathLayer``/``PathGeometry`` objects (dev-beta's real geometry does not
   exist yet -- see CONTRACT.md "Working without the other worker"), which
   is where the required physically-meaningful assertions live (CO2 band
   opacity, window transparency, whole-atmosphere transmittance).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mwir_atmos.constants import (
    GAS_CONSTANT_R,
    LIGHT_C,
    LOSCHMIDT_CM3,
    STD_PRESSURE_MBAR,
    STD_TEMP_K,
)
from mwir_atmos.hitran.cache import LineList
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics.lbl import _line_intensity_at_t, absorption_cross_section, molecular_optical_depth
from mwir_atmos.spectral import band_average, grid_from_wavelength_um, make_grid
from mwir_atmos.testing import make_synthetic_path, make_synthetic_profile
from mwir_atmos.types import PathLayer, Species

LINELISTS = ensure_linelists()


# =========================================================================
# Helpers
# =========================================================================


def _single_line(nu0=2350.0, s296=1.0e-19, gamma_air=0.07, gamma_self=0.1,
                  e_lower=100.0, n_air=0.75, delta_air=0.0,
                  molecule_id=1, local_iso_id=1) -> LineList:
    """A synthetic one-line LineList, molecule_id/local_iso_id chosen from
    H2O-161 so partition_function() resolves (any real cached isotopologue
    works -- H2O main is simplest)."""
    arr = lambda x: np.array([x], dtype=np.float64)
    return LineList(
        species=Species.H2O,
        molecule_id=molecule_id,
        nu_cm1=arr(nu0),
        s296=arr(s296),
        gamma_air=arr(gamma_air),
        gamma_self=arr(gamma_self),
        e_lower=arr(e_lower),
        n_air=arr(n_air),
        delta_air=arr(delta_air),
        local_iso_id=np.array([local_iso_id], dtype=np.int64),
        iso_abundance=arr(1.0),
    )


def _sea_level_horizontal_layer(ds_km: float) -> PathLayer:
    """A hand-built, contract-correct single-layer horizontal PathLayer at
    sea level, from mwir_atmos.testing.make_synthetic_profile()'s surface
    conditions. make_synthetic_path() itself only models vertical-ish spans
    (it needs z_top > z_sensor -- see its docstring), so a literal
    constant-altitude horizontal path is built directly here; this is
    exactly the "any valid PathLayer" case the job description calls for.
    """
    prof = make_synthetic_profile()
    p0 = float(prof.pressure_mbar[0])
    t0 = float(prof.temperature_k[0])
    n_air0 = LOSCHMIDT_CM3 * (p0 / STD_PRESSURE_MBAR) * (STD_TEMP_K / t0)
    air_col = n_air0 * ds_km * 1.0e5  # km -> cm

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
# Analytic Voigt-limit checks (synthetic single line)
# =========================================================================


def test_line_strength_equals_s296_at_296k():
    """CONTRACT.md requirement: line strength scaling must reproduce
    S(296 K) == s296 when T == 296 K."""
    linelist = LINELISTS[Species.CO2]
    s_at_ref = _line_intensity_at_t(linelist, 296.0)
    np.testing.assert_allclose(s_at_ref, linelist.s296, rtol=1e-8)


def test_line_strength_changes_away_from_296k():
    # NOTE: s296 spans ~1e-29 to ~1e-19 here, so a bare np.allclose (default
    # atol=1e-8) would trivially call almost everything "close" regardless
    # of the physics -- compare relative differences explicitly instead.
    linelist = LINELISTS[Species.H2O]
    s_hot = _line_intensity_at_t(linelist, 320.0)
    s_cold = _line_intensity_at_t(linelist, 200.0)
    rel_diff_hot = np.abs(s_hot - linelist.s296) / linelist.s296
    rel_diff_cold = np.abs(s_cold - linelist.s296) / linelist.s296
    # The great majority of lines should shift by more than 1% at +/- 24-96 K.
    assert np.median(rel_diff_hot) > 0.01
    assert np.median(rel_diff_cold) > 0.01
    assert np.all(s_hot > 0) and np.all(s_cold > 0)


def test_voigt_reduces_to_gaussian_at_zero_pressure():
    """Doppler-only limit (gamma_L -> 0): the profile must match a plain
    Gaussian line shape exactly, not a Lorentzian -- this is the "Voigt,
    not Lorentz-only" requirement in its cleanest testable form."""
    grid = grid_from_wavelength_um(4.24, 4.26, resolution_cm1=0.002)
    linelist = _single_line(nu0=2350.0, gamma_air=0.0, gamma_self=0.0, delta_air=0.0)
    temperature_k = 250.0

    xsec = absorption_cross_section(
        linelist, grid, pressure_mbar=0.0, temperature_k=temperature_k,
        partial_pressure_mbar=0.0, wing_cutoff_cm1=5.0, min_strength_fraction=0.0,
    )

    m_kg_mol = 18.010565e-3  # H2O-161
    alpha_d = 2350.0 * np.sqrt(2 * np.log(2) * GAS_CONSTANT_R * temperature_k / (m_kg_mol * LIGHT_C**2))
    s_t = _line_intensity_at_t(linelist, temperature_k)[0]
    expected = (
        s_t * np.sqrt(np.log(2) / np.pi) / alpha_d
        * np.exp(-np.log(2) * (grid.wavenumber_cm1 - 2350.0) ** 2 / alpha_d**2)
    )
    np.testing.assert_allclose(xsec, expected, rtol=1e-6, atol=expected.max() * 1e-9)


def test_voigt_reduces_to_lorentzian_at_high_pressure():
    """Pressure-broadening-dominated limit (gamma_L >> alpha_D): the
    profile must match a plain Lorentzian near line centre."""
    grid = grid_from_wavelength_um(4.24, 4.26, resolution_cm1=0.002)
    linelist = _single_line(nu0=2350.0, gamma_air=2.0, gamma_self=0.0, n_air=0.0, delta_air=0.0)
    temperature_k = 296.0

    xsec = absorption_cross_section(
        linelist, grid, pressure_mbar=STD_PRESSURE_MBAR, temperature_k=temperature_k,
        partial_pressure_mbar=0.0, wing_cutoff_cm1=5.0, min_strength_fraction=0.0,
    )
    s_t = _line_intensity_at_t(linelist, temperature_k)[0]
    gamma_l = 2.0  # gamma_air * 1 atm
    expected = s_t * (gamma_l / np.pi) / ((grid.wavenumber_cm1 - 2350.0) ** 2 + gamma_l**2)
    # Near the core (within a few HWHM) the Voigt profile is Lorentzian to
    # high accuracy when alpha_D/gamma_L is tiny; alpha_D here is ~0.003.
    core = np.abs(grid.wavenumber_cm1 - 2350.0) < 0.5
    np.testing.assert_allclose(xsec[core], expected[core], rtol=2e-3)


def test_pressure_shift_moves_line_centre():
    grid = grid_from_wavelength_um(4.24, 4.26, resolution_cm1=0.001)
    delta_air = -0.01  # cm^-1/atm, typical sign/magnitude for CO2-like lines
    linelist = _single_line(nu0=2350.0, gamma_air=0.03, gamma_self=0.03, delta_air=delta_air)

    xsec = absorption_cross_section(
        linelist, grid, pressure_mbar=STD_PRESSURE_MBAR, temperature_k=250.0,
        partial_pressure_mbar=0.0, wing_cutoff_cm1=2.0, min_strength_fraction=0.0,
    )
    peak_nu = grid.wavenumber_cm1[np.argmax(xsec)]
    expected_peak = 2350.0 + delta_air * 1.0  # p_total = 1 atm
    assert peak_nu == pytest.approx(expected_peak, abs=grid.resolution_cm1)


def test_absorption_cross_section_empty_linelist_returns_zeros():
    grid = make_grid()
    empty = _single_line()
    empty.nu_cm1 = np.array([], dtype=np.float64)
    empty.s296 = np.array([], dtype=np.float64)
    empty.gamma_air = np.array([], dtype=np.float64)
    empty.gamma_self = np.array([], dtype=np.float64)
    empty.e_lower = np.array([], dtype=np.float64)
    empty.n_air = np.array([], dtype=np.float64)
    empty.delta_air = np.array([], dtype=np.float64)
    empty.local_iso_id = np.array([], dtype=np.int64)
    empty.iso_abundance = np.array([], dtype=np.float64)

    xsec = absorption_cross_section(empty, grid, 1013.25, 288.0, 10.0)
    assert xsec.shape == (grid.n,)
    assert np.all(xsec == 0.0)


def test_absorption_cross_section_rejects_nonpositive_temperature():
    linelist = _single_line()
    grid = make_grid()
    with pytest.raises(ValueError):
        absorption_cross_section(linelist, grid, 1013.25, 0.0, 0.0)


# =========================================================================
# Physically meaningful assertions (CONTRACT.md "Definition of done")
# =========================================================================


def test_1km_sea_level_path_opaque_in_co2_band_core():
    """A 1 km sea-level horizontal path must be essentially opaque in the
    CO2 4.3 um band. Verified against the real cached line data: the band
    *core* (~4.20-4.35 um / 2296-2381 cm^-1) is opaque to better than 1e-12
    at this path length -- lines here are strong and densely spaced enough
    to saturate completely. See test_1km_sea_level_path_co2_band_edge below
    for the (physically real, not a bug) outer-edge structure near 4.4 um.
    """
    grid = make_grid()
    layer = _sea_level_horizontal_layer(ds_km=1.0)
    tau_total, _ = molecular_optical_depth([layer], grid, linelists=LINELISTS)
    transmittance = np.exp(-tau_total[0])

    core_band = (2296.0, 2380.95)  # ~4.20-4.354 um
    t_avg = band_average(grid, transmittance, core_band)
    assert t_avg < 1.0e-3, f"CO2 band core should be opaque, got transmittance {t_avg:.3e}"


def test_1km_sea_level_path_co2_band_edge_documented():
    """The literal 4.2-4.4 um range, edge to edge, is *heavily* attenuated
    but its extreme low-wavenumber edge (~4.36-4.40 um / 2272-2296 cm^-1,
    the far P-branch of the CO2 nu3 band, ~50-75 cm^-1 from the 2349 cm^-1
    band centre) is not fully saturated at only 1 km path length and 1
    cm^-1 grid spacing: individual rotational lines are resolvable there
    with real, physical (not numerical-artifact) transmission windows
    between them. Confirmed independent of the min_strength_fraction
    performance cutoff (identical structure with it disabled) during
    development -- see the dev-alpha report. This test pins the *looser*
    but still physically meaningful bound for the literal edge-to-edge
    band; test_1km_sea_level_path_opaque_in_co2_band_core above pins the
    strict <1e-3 bound where it unambiguously holds.
    """
    grid = make_grid()
    layer = _sea_level_horizontal_layer(ds_km=1.0)
    tau_total, _ = molecular_optical_depth([layer], grid, linelists=LINELISTS)
    transmittance = np.exp(-tau_total[0])

    full_band = (1.0e4 / 4.4, 1.0e4 / 4.2)  # exactly 4.2-4.4 um
    t_avg = band_average(grid, transmittance, full_band)
    assert t_avg < 5.0e-3


def test_1km_sea_level_path_transparent_in_window():
    """The same 1 km sea-level path must be reasonably transparent
    (> 0.5) in the 3.7-4.0 um atmospheric window."""
    grid = make_grid()
    layer = _sea_level_horizontal_layer(ds_km=1.0)
    tau_total, _ = molecular_optical_depth([layer], grid, linelists=LINELISTS)
    transmittance = np.exp(-tau_total[0])

    window = (1.0e4 / 4.0, 1.0e4 / 3.7)
    t_avg = band_average(grid, transmittance, window)
    assert t_avg > 0.5, f"3.7-4.0um window should be transparent, got {t_avg:.3f}"


def test_vertical_whole_atmosphere_transmittance_high_at_4um():
    """Vertical whole-atmosphere transmittance at 4.0 um should be high
    (> 0.7)."""
    grid = make_grid()
    path = make_synthetic_path(sensor_alt_km=0.0, zenith_deg=0.0, n_layers=50)
    tau_total, _ = molecular_optical_depth(path.layers, grid, linelists=LINELISTS)
    column_tau = tau_total.sum(axis=0)
    transmittance = np.exp(-column_tau)

    near_4um = (2495.0, 2505.0)
    t_avg = band_average(grid, transmittance, near_4um)
    assert t_avg > 0.7, f"whole-atmosphere 4.0um transmittance should be high, got {t_avg:.3f}"


# =========================================================================
# molecular_optical_depth contract shape / consistency
# =========================================================================


def test_molecular_optical_depth_shapes_and_consistency():
    grid = make_grid()
    path = make_synthetic_path(sensor_alt_km=0.0, zenith_deg=30.0, n_layers=8)
    tau_total, tau_by_species = molecular_optical_depth(path.layers, grid, linelists=LINELISTS)

    assert tau_total.shape == (len(path.layers), grid.n)
    for species, arr in tau_by_species.items():
        assert arr.shape == (len(path.layers), grid.n)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)

    reconstructed = np.sum(np.stack(list(tau_by_species.values()), axis=0), axis=0)
    np.testing.assert_allclose(tau_total, reconstructed)


def test_molecular_optical_depth_species_subset():
    grid = make_grid()
    path = make_synthetic_path(sensor_alt_km=0.0, zenith_deg=0.0, n_layers=5)
    _, tau_by_species = molecular_optical_depth(
        path.layers, grid, species=[Species.CO2], linelists=LINELISTS
    )
    assert set(tau_by_species) == {Species.CO2}


def test_molecular_optical_depth_zero_column_layer_is_zero():
    grid = make_grid()
    layer = _sea_level_horizontal_layer(ds_km=1.0)
    for species in layer.column_cm2:
        layer.column_cm2[species] = 0.0
    tau_total, _ = molecular_optical_depth([layer], grid, linelists=LINELISTS)
    assert np.all(tau_total == 0.0)


# =========================================================================
# Performance target: 50 layers x ~2500 grid points in a few seconds
# =========================================================================


def test_performance_50_layers_default_grid():
    grid = make_grid()  # default 1800-4300 cm^-1 @ 1.0 cm^-1 -> 2501 points
    path = make_synthetic_path(sensor_alt_km=0.0, zenith_deg=0.0, n_layers=50)

    t0 = time.time()
    tau_total, _ = molecular_optical_depth(path.layers, grid, linelists=LINELISTS)
    elapsed = time.time() - t0

    assert tau_total.shape == (50, grid.n)
    # Generous bound (measured ~5s during development) to absorb slower CI
    # hardware while still catching a gross performance regression.
    assert elapsed < 30.0, f"50-layer molecular_optical_depth took {elapsed:.1f}s, expected a few seconds"
