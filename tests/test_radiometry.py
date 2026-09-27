"""Sensor-aperture radiance.

Tests use a narrow band and coarse resolution so the suite stays fast; the
line-by-line engine dominates runtime and cost scales with grid points x layers.

Note on the opaque-band test: an optically thick atmosphere must radiate as a
blackbody, but at the temperature it actually emits from -- which is the
*emission-weighted* layer temperature, not the surface air temperature. With 1 km
layers at the surface and a 6.5 K/km lapse rate, layer 0's mean sits ~2.3 K below
the surface value, so comparing against the surface temperature under-reads by
~8%. That is discretisation, not a transfer error.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from mwir_atmos.api import run_scenario
from mwir_atmos.optics.transmittance import compute_layer_optics
from mwir_atmos.radiometry.radiance import airmass, lunar_irradiance, solar_irradiance
from mwir_atmos.radiometry.surface import surface_emissivity
from mwir_atmos.spectral import band_average, band_integrate, make_grid, planck
from mwir_atmos.types import SpectralSampling
from mwir_atmos.types import AerosolModel, ModelAtmosphere, Scenario, SurfaceType

CO2_BAND = (2273.0, 2381.0)      # 4.2-4.4 um, opaque at any useful path length
WINDOW = (2500.0, 2857.0)        # 3.5-4.0 um, the clean MWIR window


def _scenario(**kw) -> Scenario:
    base = dict(
        atmosphere=ModelAtmosphere.MIDLAT_SUMMER,
        aerosol=AerosolModel.RURAL,
        wavenumber_min_cm1=2200.0,
        wavenumber_max_cm1=2900.0,
        resolution_cm1=8.0,
        local_datetime=None,
    )
    base.update(kw)
    return Scenario(**{"spectral_sampling": SpectralSampling.POINT, **base})


@pytest.fixture(scope="module")
def uplooking_night():
    return run_scenario(_scenario(sensor_alt_km=0.0, sensor_zenith_deg=0.0))


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_run_scenario_populates_radiance(uplooking_night):
    assert uplooking_night.radiance is not None
    assert uplooking_night.sun is not None
    assert uplooking_night.moon is not None


def test_every_component_is_an_array_never_none(uplooking_night):
    r = uplooking_night.radiance
    for name in (
        "path_thermal", "surface_emitted", "surface_reflected_solar",
        "surface_reflected_lunar", "solar_single_scatter", "lunar_single_scatter",
        "direct_solar", "direct_lunar", "source_emitted", "total",
    ):
        arr = getattr(r, name)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (uplooking_night.grid.n,)
        assert np.all(np.isfinite(arr))


def test_total_equals_the_sum_of_components(uplooking_night):
    r = uplooking_night.radiance
    summed = (
        r.path_thermal + r.surface_emitted + r.surface_reflected_solar
        + r.surface_reflected_lunar + r.solar_single_scatter + r.lunar_single_scatter
        + r.direct_solar + r.direct_lunar + r.source_emitted
    )
    np.testing.assert_allclose(r.total, summed, rtol=1e-12, atol=0.0)


def test_radiance_is_non_negative(uplooking_night):
    assert np.all(uplooking_night.radiance.total >= 0.0)


# --------------------------------------------------------------------------
# The headline physics test
# --------------------------------------------------------------------------


def test_opaque_band_radiates_as_a_blackbody_at_its_own_temperature(uplooking_night):
    """An optically thick atmosphere is a blackbody at the temperature it emits from."""
    r = uplooking_night
    assert band_average(r.grid, r.transmittance.total, CO2_BAND) < 1e-6, "band not opaque"

    optics = compute_layer_optics(
        r.path, r.grid, aerosol=r.scenario.aerosol, visibility_km=r.scenario.visibility_km
    )
    sel = (r.grid.wavenumber_cm1 >= CO2_BAND[0]) & (r.grid.wavenumber_cm1 <= CO2_BAND[1])
    emission = (
        (1.0 - np.exp(-optics.tau_absorption[:, sel])) * r.transmittance.cumulative[:-1][:, sel]
    ).mean(axis=1)
    weights = emission / emission.sum()
    t_effective = float(sum(w * l.t_k for w, l in zip(weights, r.path.layers)))

    observed = band_average(r.grid, r.radiance.total, CO2_BAND)
    expected = band_average(r.grid, planck(r.grid.wavenumber_cm1, t_effective), CO2_BAND)
    assert observed / expected == pytest.approx(1.0, abs=0.02)


def test_opaque_band_emission_comes_from_the_lowest_few_hundred_metres(uplooking_night):
    """A ground sensor looking up in an opaque band sees only the air next to it.

    With the refined boundary layer the first slab is 10 m thick, so no single
    layer dominates any more -- the physical statement is about depth, not layer
    index: essentially all the signal originates below ~300 m.
    """
    r = uplooking_night
    optics = compute_layer_optics(
        r.path, r.grid, aerosol=r.scenario.aerosol, visibility_km=r.scenario.visibility_km
    )
    sel = (r.grid.wavenumber_cm1 >= CO2_BAND[0]) & (r.grid.wavenumber_cm1 <= CO2_BAND[1])
    emission = (
        (1.0 - np.exp(-optics.tau_absorption[:, sel])) * r.transmittance.cumulative[:-1][:, sel]
    ).mean(axis=1)

    below_300m = sum(
        e for e, l in zip(emission, r.path.layers) if l.z_top_km <= 0.3 + 1e-9
    )
    assert below_300m / emission.sum() > 0.95
    # And the very first slab still carries the single largest share.
    assert emission[0] == emission.max()


def test_uplooking_night_radiance_is_all_path_thermal(uplooking_night):
    r = uplooking_night.radiance
    assert band_average(uplooking_night.grid, r.path_thermal, CO2_BAND) == pytest.approx(
        band_average(uplooking_night.grid, r.total, CO2_BAND), rel=1e-9
    )


def test_window_is_far_dimmer_than_the_opaque_band_looking_up(uplooking_night):
    """A transparent window looking at cold space must be much darker."""
    r = uplooking_night
    assert band_average(r.grid, r.radiance.total, WINDOW) < 0.5 * band_average(
        r.grid, r.radiance.total, CO2_BAND
    )


# --------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------


def test_lookdown_window_approaches_surface_emission():
    r = run_scenario(
        _scenario(
            sensor_alt_km=8.0,
            sensor_zenith_deg=180.0,
            surface_type=SurfaceType.SEA_WATER,
            surface_skin_temp_k=300.0,
        )
    )
    observed = band_average(r.grid, r.radiance.total, WINDOW)
    expected = 0.98 * band_average(r.grid, planck(r.grid.wavenumber_cm1, 300.0), WINDOW)
    # Slightly below eps*B because of path attenuation, plus a little path emission.
    assert 0.93 < observed / expected < 1.02
    assert band_average(r.grid, r.radiance.surface_emitted, WINDOW) / observed > 0.9


def test_no_surface_terms_when_the_path_misses_the_ground(uplooking_night):
    assert not uplooking_night.path.hits_ground
    assert np.all(uplooking_night.radiance.surface_emitted == 0.0)


def test_hotter_skin_gives_more_radiance():
    common = dict(sensor_alt_km=8.0, sensor_zenith_deg=180.0, surface_type=SurfaceType.LAND_VEGETATED)
    cold = run_scenario(_scenario(**common, surface_skin_temp_k=280.0))
    hot = run_scenario(_scenario(**common, surface_skin_temp_k=320.0))
    assert band_average(hot.grid, hot.radiance.total, WINDOW) > band_average(
        cold.grid, cold.radiance.total, WINDOW
    )


def test_surface_emissivity_override_wins():
    assert surface_emissivity(SurfaceType.DESERT_SAND, override=0.5) == 0.5


def test_sea_water_emissivity_falls_at_grazing_incidence():
    nadir = surface_emissivity(SurfaceType.SEA_WATER, incidence_deg=0.0)
    mid = surface_emissivity(SurfaceType.SEA_WATER, incidence_deg=60.0)
    grazing = surface_emissivity(SurfaceType.SEA_WATER, incidence_deg=85.0)
    assert nadir == pytest.approx(0.98, abs=1e-9)
    assert mid < nadir
    assert grazing < mid < nadir
    assert grazing < 0.8


def test_land_emissivity_is_angle_independent():
    a = surface_emissivity(SurfaceType.LAND_VEGETATED, incidence_deg=0.0)
    b = surface_emissivity(SurfaceType.LAND_VEGETATED, incidence_deg=70.0)
    assert a == b == pytest.approx(0.96)


def test_sand_is_less_emissive_than_vegetation():
    assert surface_emissivity(SurfaceType.DESERT_SAND) < surface_emissivity(
        SurfaceType.LAND_VEGETATED
    )


# --------------------------------------------------------------------------
# Illumination
# --------------------------------------------------------------------------


def test_sun_below_horizon_zeroes_every_solar_term():
    r = run_scenario(
        _scenario(
            sensor_alt_km=0.0, sensor_zenith_deg=0.0, latitude_deg=40.0, longitude_deg=-90.0,
            utc_offset_hours=-6.0, local_datetime=datetime(2026, 1, 15, 2, 0),
        )
    )
    assert not r.sun.is_above_horizon
    assert np.all(r.radiance.direct_solar == 0.0)
    assert np.all(r.radiance.solar_single_scatter == 0.0)
    assert np.all(r.radiance.surface_reflected_solar == 0.0)


def test_daytime_adds_solar_scatter():
    common = dict(sensor_alt_km=0.0, sensor_zenith_deg=30.0, sensor_azimuth_deg=180.0,
                  latitude_deg=35.0, longitude_deg=-100.0, utc_offset_hours=-6.0)
    day = run_scenario(_scenario(**common, local_datetime=datetime(2026, 6, 21, 12, 0)))
    assert day.sun.is_above_horizon
    assert day.radiance.solar_single_scatter.max() > 0.0
    assert day.radiance.total.max() >= day.radiance.path_thermal.max()


def test_include_solar_false_suppresses_solar():
    common = dict(sensor_alt_km=0.0, sensor_zenith_deg=30.0, latitude_deg=35.0,
                  longitude_deg=-100.0, utc_offset_hours=-6.0,
                  local_datetime=datetime(2026, 6, 21, 12, 0))
    off = run_scenario(_scenario(**common, include_solar=False))
    assert np.all(off.radiance.solar_single_scatter == 0.0)


def test_solar_irradiance_integrates_to_the_solar_constant():
    """Over a very wide grid the normalisation must recover 1361 W/m^2."""
    wide = make_grid(1.0, 60000.0, 5.0)
    total = band_integrate(wide, solar_irradiance(wide), (1.0, 60000.0))
    assert total == pytest.approx(1361.0 * 1e-4, rel=0.02)


def test_solar_distance_factor_scales_irradiance():
    g = make_grid(2200.0, 2900.0, 20.0)
    assert solar_irradiance(g, 1.03).max() > solar_irradiance(g, 0.97).max()


# --------------------------------------------------------------------------
# Lunar
# --------------------------------------------------------------------------


def test_lunar_thermal_dominates_reflected_sunlight_in_the_mwir():
    """The usual mistake is modelling the moon as albedo-scaled sunlight."""
    g = make_grid(2200.0, 2900.0, 20.0)
    sc = _scenario(moon_phase=0.5, latitude_deg=35.0, longitude_deg=-100.0,
                   utc_offset_hours=-6.0, local_datetime=datetime(2026, 6, 21, 1, 0))
    from mwir_atmos.ephemeris.moon import lunar_position
    from mwir_atmos.ephemeris.sun import solar_position

    moon = lunar_position(sc, solar_position(sc))
    thermal, reflected = lunar_irradiance(g, moon)
    assert thermal.mean() > 10.0 * reflected.mean()


def test_new_moon_emits_nothing():
    g = make_grid(2200.0, 2900.0, 20.0)
    sc = _scenario(moon_phase=0.0, latitude_deg=35.0, longitude_deg=-100.0,
                   utc_offset_hours=-6.0, local_datetime=datetime(2026, 6, 21, 1, 0))
    from mwir_atmos.ephemeris.moon import lunar_position
    from mwir_atmos.ephemeris.sun import solar_position

    thermal, reflected = lunar_irradiance(g, lunar_position(sc, solar_position(sc)))
    assert thermal.max() == pytest.approx(0.0, abs=1e-30)
    assert reflected.max() == pytest.approx(0.0, abs=1e-30)


def test_full_moon_brighter_than_quarter():
    g = make_grid(2200.0, 2900.0, 20.0)
    from mwir_atmos.ephemeris.moon import lunar_position
    from mwir_atmos.ephemeris.sun import solar_position

    common = dict(latitude_deg=35.0, longitude_deg=-100.0, utc_offset_hours=-6.0,
                  local_datetime=datetime(2026, 6, 21, 1, 0))
    full = _scenario(**common, moon_phase=0.5)
    quarter = _scenario(**common, moon_phase=0.25)
    t_full, _ = lunar_irradiance(g, lunar_position(full, solar_position(full)))
    t_qtr, _ = lunar_irradiance(g, lunar_position(quarter, solar_position(quarter)))
    assert t_full.max() > t_qtr.max()


# --------------------------------------------------------------------------
# Explicit target
# --------------------------------------------------------------------------


def test_hot_source_dominates_a_short_transparent_path():
    r = run_scenario(
        _scenario(
            sensor_alt_km=1.0, sensor_zenith_deg=90.0, source_range_km=1.0,
            source_temp_k=800.0, source_emissivity=1.0,
        )
    )
    assert band_average(r.grid, r.radiance.source_emitted, WINDOW) > 0.9 * band_average(
        r.grid, r.radiance.total, WINDOW
    )


def test_no_source_term_without_a_source_temperature(uplooking_night):
    assert np.all(uplooking_night.radiance.source_emitted == 0.0)


# --------------------------------------------------------------------------
# Airmass helper
# --------------------------------------------------------------------------


def test_airmass_is_unity_at_zenith_and_finite_at_the_horizon():
    assert airmass(0.0) == pytest.approx(1.0, abs=1e-3)
    assert airmass(60.0) == pytest.approx(2.0, abs=0.02)
    assert 30.0 < airmass(90.0) < 45.0
