"""End-to-end invariants of ``api.run_scenario``.

This is the contract the CLI and the web GUI both depend on, so the assertions
here are deliberately about properties that must hold for *every* scenario
rather than about specific numbers: transmittance bounded and ordered, radiance
non-negative and additive, columns conserved, geometry self-consistent.

All scenarios use a narrow band at coarse resolution. The line-by-line engine
dominates runtime and scales with grid points x layers, so a wide grid here
would make the suite take minutes instead of seconds.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from mwir_atmos.api import run_scenario
from mwir_atmos.spectral import band_average
from mwir_atmos.types import (
    AerosolModel,
    ModelAtmosphere,
    Scenario,
    Species,
    SurfaceType,
)

NARROW = dict(wavenumber_min_cm1=2350.0, wavenumber_max_cm1=2450.0, resolution_cm1=20.0)


def _sc(**kw) -> Scenario:
    base = dict(**NARROW, local_datetime=None)
    base.update(kw)
    return Scenario(**base)


def _assert_physical(result) -> None:
    """Invariants every scenario must satisfy, whatever its configuration."""
    t = result.transmittance
    n = result.grid.n

    assert t.total.shape == (n,)
    assert np.all(np.isfinite(t.total))
    assert np.all(t.total >= 0.0) and np.all(t.total <= 1.0 + 1e-12)

    # Total extinction can never be weaker than any single mechanism.
    for component in (t.molecular, t.aerosol, t.rayleigh):
        assert np.all(t.total <= component + 1e-12)
    for species_trans in t.by_species.values():
        assert np.all(t.molecular <= species_trans + 1e-12)

    # Cumulative transmittance: starts at one, ends at the total, never rises.
    assert t.cumulative is not None
    assert t.cumulative.shape == (result.path.n_layers + 1, n)
    np.testing.assert_allclose(t.cumulative[0], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(t.cumulative[-1], t.total, rtol=1e-10, atol=0)
    assert np.all(np.diff(t.cumulative, axis=0) <= 1e-12)

    # Geometry self-consistency.
    assert result.path.n_layers >= 1
    assert result.path.total_length_km > 0.0
    for layer in result.path.layers:
        assert layer.ds_km > 0.0
        assert layer.t_k > 0.0
        assert layer.p_mbar > 0.0
        assert layer.air_column_cm2 > 0.0
        for species in (Species.H2O, Species.CO2):
            assert layer.column_cm2[species] >= 0.0
    if result.path.hits_ground:
        assert result.path.ground_temp_k is not None

    # Radiance.
    if result.radiance is not None:
        r = result.radiance
        parts = (
            r.path_thermal, r.surface_emitted, r.surface_reflected_solar,
            r.surface_reflected_lunar, r.solar_single_scatter, r.lunar_single_scatter,
            r.direct_solar, r.direct_lunar, r.source_emitted,
        )
        for part in parts:
            assert part.shape == (n,)
            assert np.all(np.isfinite(part))
            assert np.all(part >= 0.0)
        np.testing.assert_allclose(r.total, sum(parts), rtol=1e-10, atol=0)


# --------------------------------------------------------------------------
# Coverage across the option space
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", list(ModelAtmosphere))
def test_every_model_atmosphere_runs(model):
    _assert_physical(run_scenario(_sc(atmosphere=model)))


@pytest.mark.parametrize("aerosol", list(AerosolModel))
def test_every_aerosol_model_runs(aerosol):
    _assert_physical(run_scenario(_sc(aerosol=aerosol, visibility_km=10.0)))


@pytest.mark.parametrize("surface", list(SurfaceType))
def test_every_surface_type_runs_looking_down(surface):
    result = run_scenario(
        _sc(surface_type=surface, sensor_alt_km=5.0, sensor_zenith_deg=180.0)
    )
    assert result.path.hits_ground
    _assert_physical(result)


@pytest.mark.parametrize(
    "zenith",
    [0.0, 30.0, 60.0, 89.0, 90.0, 91.0, 120.0, 180.0],
)
def test_geometry_sweep_from_the_ground(zenith):
    _assert_physical(run_scenario(_sc(sensor_alt_km=0.5, sensor_zenith_deg=zenith)))


@pytest.mark.parametrize("altitude", [0.5, 1.0, 10.0, 20.0, 50.0])
def test_altitude_sweep_looking_down(altitude):
    _assert_physical(run_scenario(_sc(sensor_alt_km=altitude, sensor_zenith_deg=170.0)))


def test_ground_sensor_looking_down_is_rejected():
    """A sensor on the ground looking below the horizontal has nowhere to go."""
    with pytest.raises(ValueError, match="degenerate"):
        run_scenario(_sc(sensor_alt_km=0.0, sensor_zenith_deg=170.0))


def test_limb_path_from_altitude():
    result = run_scenario(_sc(sensor_alt_km=20.0, sensor_zenith_deg=92.0))
    assert not result.path.hits_ground
    assert result.path.tangent_alt_km is not None
    _assert_physical(result)


def test_finite_range_horizontal_path():
    result = run_scenario(_sc(sensor_alt_km=2.0, sensor_zenith_deg=90.0, source_range_km=5.0))
    assert result.path.total_length_km == pytest.approx(5.0, rel=1e-6)
    assert not result.path.hits_ground
    _assert_physical(result)


# --------------------------------------------------------------------------
# Monotonic responses -- these catch sign errors that shape checks miss
# --------------------------------------------------------------------------


def test_transmittance_falls_as_visibility_drops():
    values = [
        band_average(
            (r := run_scenario(_sc(aerosol=AerosolModel.DESERT, visibility_km=v))).grid,
            r.transmittance.total,
            (2350.0, 2450.0),
        )
        for v in (50.0, 23.0, 10.0, 5.0, 2.0)
    ]
    assert values == sorted(values, reverse=True)


def test_transmittance_falls_as_path_lengthens():
    values = [
        band_average(
            (r := run_scenario(_sc(sensor_alt_km=0.0, sensor_zenith_deg=90.0, source_range_km=d))).grid,
            r.transmittance.total,
            (2350.0, 2450.0),
        )
        for d in (0.5, 1.0, 2.0, 5.0)
    ]
    assert values == sorted(values, reverse=True)


def test_transmittance_falls_as_zenith_angle_grows():
    values = [
        band_average(
            (r := run_scenario(_sc(sensor_alt_km=0.0, sensor_zenith_deg=z))).grid,
            r.transmittance.total,
            (2350.0, 2450.0),
        )
        for z in (0.0, 30.0, 60.0, 80.0)
    ]
    assert values == sorted(values, reverse=True)


def test_humid_atmosphere_is_less_transmissive_than_dry():
    tropical = run_scenario(_sc(atmosphere=ModelAtmosphere.TROPICAL,
                                sensor_alt_km=0.0, sensor_zenith_deg=90.0, source_range_km=5.0))
    arctic = run_scenario(_sc(atmosphere=ModelAtmosphere.SUBARCTIC_WINTER,
                              sensor_alt_km=0.0, sensor_zenith_deg=90.0, source_range_km=5.0))
    band = (2350.0, 2450.0)
    assert band_average(tropical.grid, tropical.transmittance.total, band) < band_average(
        arctic.grid, arctic.transmittance.total, band
    )


# --------------------------------------------------------------------------
# Surface overrides propagate
# --------------------------------------------------------------------------


def test_ground_temperature_override_reaches_the_profile():
    result = run_scenario(_sc(ground_temp_k=315.0))
    assert result.profile.temperature_k[0] == pytest.approx(315.0, abs=0.01)
    _assert_physical(result)


def test_ground_humidity_override_changes_the_water_column():
    dry = run_scenario(_sc(ground_rh_pct=5.0, sensor_zenith_deg=90.0, source_range_km=5.0))
    wet = run_scenario(_sc(ground_rh_pct=95.0, sensor_zenith_deg=90.0, source_range_km=5.0))
    dry_col = sum(l.column_cm2[Species.H2O] for l in dry.path.layers)
    wet_col = sum(l.column_cm2[Species.H2O] for l in wet.path.layers)
    assert wet_col > 3.0 * dry_col


def test_co2_override_changes_the_carbon_column():
    low = run_scenario(_sc(co2_ppmv=280.0))
    high = run_scenario(_sc(co2_ppmv=800.0))
    low_col = sum(l.column_cm2[Species.CO2] for l in low.path.layers)
    high_col = sum(l.column_cm2[Species.CO2] for l in high.path.layers)
    assert high_col > 2.0 * low_col


# --------------------------------------------------------------------------
# Illumination plumbing
# --------------------------------------------------------------------------


def test_longitude_shifts_solar_time_at_fixed_clock_and_offset():
    """Holding clock time and UTC offset, moving east advances true solar time.

    Sites east of their timezone meridian reach solar noon earlier by the clock,
    so at a fixed 09:00 the sun is further past sunrise the further east you go.
    """
    zeniths = []
    for lon in (-30.0, -15.0, 0.0, 15.0):
        r = run_scenario(
            _sc(latitude_deg=40.0, longitude_deg=lon, utc_offset_hours=0.0,
                local_datetime=datetime(2026, 6, 21, 9, 0))
        )
        zeniths.append(r.sun.zenith_deg)
    # Morning: later solar time means a higher sun, i.e. smaller zenith.
    assert zeniths == sorted(zeniths, reverse=True)


def test_sunrise_sunset_calibration_reaches_run_scenario():
    common = dict(latitude_deg=40.42, longitude_deg=-3.70,
                  local_datetime=datetime(2026, 6, 21, 9, 0))
    truth = run_scenario(_sc(**common, utc_offset_hours=2.0))
    guessed = run_scenario(_sc(**common))
    calibrated = run_scenario(
        _sc(**common, sunrise_local_hours=6.733, sunset_local_hours=21.800)
    )
    assert abs(guessed.sun.zenith_deg - truth.sun.zenith_deg) > 10.0
    assert abs(calibrated.sun.zenith_deg - truth.sun.zenith_deg) < 0.5


def test_daytime_and_nighttime_both_produce_valid_results():
    for hour in (1, 6, 12, 18, 23):
        result = run_scenario(
            _sc(latitude_deg=35.0, longitude_deg=-100.0, utc_offset_hours=-6.0,
                local_datetime=datetime(2026, 6, 21, hour, 0),
                sensor_alt_km=0.0, sensor_zenith_deg=40.0)
        )
        _assert_physical(result)


def test_turbulence_is_none_when_not_requested():
    assert run_scenario(_sc()).turbulence is None


# --------------------------------------------------------------------------
# Determinism and grid handling
# --------------------------------------------------------------------------


def test_repeated_runs_are_bit_identical():
    a = run_scenario(_sc())
    b = run_scenario(_sc())
    np.testing.assert_array_equal(a.transmittance.total, b.transmittance.total)
    np.testing.assert_array_equal(a.radiance.total, b.radiance.total)


def test_band_average_converges_once_lines_are_resolved():
    """Band averages agree between fine grids, but NOT with a coarse one.

    A band average integrates exp(-tau), which is violently non-linear across a
    line. Sampling a line-dense band too sparsely aliases: measured in the CO2
    nu3 R-branch against a converged 0.25 cm^-1 reference, a 20 cm^-1 grid is
    36% wrong, 5 cm^-1 is 2.9% wrong, and 1 cm^-1 is 0.3% wrong. Smooth window
    regions are far more forgiving. This is a sampling property, not a defect --
    but it means coarse-resolution band averages in structured bands must not be
    trusted quantitatively.
    """
    band = (2350.0, 2450.0)
    fine = run_scenario(_sc(resolution_cm1=1.0))
    finer = run_scenario(_sc(resolution_cm1=0.5))
    assert band_average(fine.grid, fine.transmittance.total, band) == pytest.approx(
        band_average(finer.grid, finer.transmittance.total, band), rel=0.02
    )


def test_coarse_sampling_aliases_a_line_dense_band():
    """Guard the caveat above: it must stay visible if someone 'fixes' it."""
    band = (2350.0, 2450.0)
    coarse = run_scenario(_sc(resolution_cm1=20.0))
    fine = run_scenario(_sc(resolution_cm1=0.5))
    coarse_avg = band_average(coarse.grid, coarse.transmittance.total, band)
    fine_avg = band_average(fine.grid, fine.transmittance.total, band)
    assert abs(coarse_avg - fine_avg) / fine_avg > 0.15
    assert coarse_avg < fine_avg      # sparse sampling over-weights the line cores here


def test_line_strength_cutoff_reaches_the_engine():
    """A looser cutoff drops weak lines, so it cannot absorb more than a tight one."""
    loose = run_scenario(_sc(line_strength_cutoff=1e-2))
    tight = run_scenario(_sc(line_strength_cutoff=1e-6))
    assert np.all(loose.transmittance.molecular >= tight.transmittance.molecular - 1e-12)
    assert not np.array_equal(loose.transmittance.total, tight.transmittance.total)
