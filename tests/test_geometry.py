"""Tests for mwir_atmos.geometry.path.build_path.

Physically meaningful checks per docs/build_contract.md: upward/downward/horizontal/
limb/finite-range paths, the airmass limits at 60 and 90 degrees (the
"catches a plane-parallel bug" test), ground-intersection flags, tangent-path
flags, and column-density conservation under re-layering.
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.atmosphere.profiles import get_profile
from mwir_atmos.constants import EARTH_RADIUS_KM, TOA_ALTITUDE_KM
from mwir_atmos.geometry.path import build_path
from mwir_atmos.types import ModelAtmosphere, Species


@pytest.fixture(scope="module")
def profile():
    return get_profile(ModelAtmosphere.US_STANDARD)


def _total_air_column(path) -> float:
    return float(sum(layer.air_column_cm2 for layer in path.layers))


def _total_species_column(path, species: Species) -> float:
    return float(sum(layer.column_cm2[species] for layer in path.layers))


# =========================================================================
# Airmass limits -- the plane-parallel-bug catcher
# =========================================================================


def test_airmass_zenith_60_is_about_2(profile):
    vertical = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=60)
    slant = build_path(profile, sensor_alt_km=0.0, zenith_deg=60.0, max_layers=60)
    airmass = _total_air_column(slant) / _total_air_column(vertical)
    assert 1.9 < airmass < 2.1


def test_airmass_zenith_90_is_finite_and_large(profile):
    """The plane-parallel secant blows up (1/cos(90)=inf) at exactly the
    horizon; the spherical treatment must not."""
    vertical = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=60)
    horizontal = build_path(profile, sensor_alt_km=0.0, zenith_deg=90.0, max_layers=60)
    airmass = _total_air_column(horizontal) / _total_air_column(vertical)
    assert np.isfinite(airmass)
    assert 30.0 < airmass < 45.0  # docs/build_contract.md: "~35-40"


def test_airmass_increases_monotonically_with_zenith(profile):
    vertical_col = _total_air_column(build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=60))
    prev = 0.0
    for zen in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 89.0, 90.0):
        col = _total_air_column(build_path(profile, sensor_alt_km=0.0, zenith_deg=zen, max_layers=60))
        airmass = col / vertical_col
        assert airmass >= prev - 1.0e-9
        prev = airmass


# =========================================================================
# Upward / downward / horizontal / limb classification
# =========================================================================


def test_upward_looking_reaches_space_by_default(profile):
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=20.0, max_layers=30)
    assert path.reaches_space is True
    assert path.hits_ground is False
    assert path.tangent_alt_km is None
    assert path.layers[-1].z_top_km == pytest.approx(TOA_ALTITUDE_KM, abs=1.0e-6)


def test_downward_looking_hits_ground(profile):
    path = build_path(profile, sensor_alt_km=5.0, zenith_deg=150.0, max_layers=30)
    assert path.hits_ground is True
    assert path.reaches_space is False
    assert path.tangent_alt_km is None
    assert path.ground_temp_k is not None
    assert path.ground_temp_k == pytest.approx(profile.temperature_k[0])
    assert path.layers[-1].z_bottom_km == pytest.approx(0.0, abs=1.0e-6)


def test_downward_from_ground_sensor_is_degenerate(profile):
    """A ground-level sensor looking below the local horizontal has nothing
    to integrate -- this must raise rather than silently return an empty or
    nonsensical path."""
    with pytest.raises(ValueError):
        build_path(profile, sensor_alt_km=0.0, zenith_deg=91.0, max_layers=30)


def test_horizontal_path_is_spherical_not_plane_parallel(profile):
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=90.0, max_layers=60)
    assert path.hits_ground is False
    assert path.tangent_alt_km is None  # sensor is its own tangent point, not a limb "turnover"
    assert np.isfinite(path.total_length_km)
    assert path.total_length_km > 500.0  # much longer than any local/plane-parallel estimate


def test_limb_path_turns_over_without_hitting_ground(profile):
    # From 50 km, a few degrees past 90 clears the ~25.5 km tangent (see
    # module smoke-test derivation); well clear of ground.
    path = build_path(profile, sensor_alt_km=50.0, zenith_deg=95.0, max_layers=60)
    assert path.hits_ground is False
    assert path.tangent_alt_km is not None
    assert 0.0 < path.tangent_alt_km < 50.0
    # The minimum altitude actually visited by any layer should be close to
    # the reported tangent altitude.
    min_alt = min(0.5 * (l.z_bottom_km + l.z_top_km) for l in path.layers)
    assert min_alt == pytest.approx(path.tangent_alt_km, abs=1.0)


def test_limb_path_too_steep_hits_ground_instead(profile):
    # A much steeper look angle from a low platform cannot clear the Earth.
    path = build_path(profile, sensor_alt_km=10.0, zenith_deg=150.0, max_layers=40)
    assert path.hits_ground is True
    assert path.tangent_alt_km is None


# =========================================================================
# Sensor above/below source, and finite source_range_km
# =========================================================================


def test_sensor_above_source_below(profile):
    path = build_path(profile, sensor_alt_km=10.0, zenith_deg=170.0, source_alt_km=0.0, max_layers=30)
    assert path.hits_ground is True  # source_alt_km == the surface
    assert path.total_length_km < build_path(
        profile, sensor_alt_km=10.0, zenith_deg=170.0, max_layers=30
    ).total_length_km + 1.0e-6


def test_sensor_below_source_above(profile):
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=10.0, source_alt_km=20.0, max_layers=30)
    assert path.hits_ground is False
    assert path.reaches_space is False
    assert path.layers[-1].z_top_km == pytest.approx(20.0, abs=0.5)


def test_finite_source_range_terminates_mid_atmosphere(profile):
    path = build_path(profile, sensor_alt_km=2.0, zenith_deg=80.0, source_range_km=5.0, max_layers=20)
    assert path.total_length_km == pytest.approx(5.0, abs=1.0e-6)
    assert path.hits_ground is False
    assert path.reaches_space is False


def test_source_range_shorter_than_full_path_reduces_column(profile):
    full = build_path(profile, sensor_alt_km=0.0, zenith_deg=45.0, max_layers=40)
    short = build_path(
        profile, sensor_alt_km=0.0, zenith_deg=45.0, source_range_km=10.0, max_layers=40
    )
    assert _total_air_column(short) < _total_air_column(full)


def test_unreachable_source_raises(profile):
    # From 50 km at zenith=95 the tangent altitude is ~25.5 km (see limb
    # test); a source below that is geometrically unreachable.
    with pytest.raises(ValueError):
        build_path(profile, sensor_alt_km=50.0, zenith_deg=95.0, source_alt_km=10.0)


def test_sensor_below_profile_base_raises(profile):
    with pytest.raises(ValueError):
        build_path(profile, sensor_alt_km=-1.0, zenith_deg=0.0)


# =========================================================================
# Layer bookkeeping
# =========================================================================


def test_layer_zero_adjacent_to_sensor_upward(profile):
    path = build_path(profile, sensor_alt_km=2.0, zenith_deg=10.0, max_layers=20)
    assert path.layers[0].index == 0
    assert path.layers[0].z_bottom_km == pytest.approx(2.0, abs=1.0e-6)


def test_layer_zero_adjacent_to_sensor_downward(profile):
    path = build_path(profile, sensor_alt_km=8.0, zenith_deg=170.0, max_layers=20)
    assert path.layers[0].index == 0
    assert path.layers[0].z_top_km == pytest.approx(8.0, abs=1.0e-6)


def test_max_layers_respected(profile):
    path = build_path(profile, sensor_alt_km=50.0, zenith_deg=95.0, max_layers=15)
    assert path.n_layers <= 15


@pytest.mark.parametrize("max_layers", [8, 20, 40, 80, 150])
def test_column_conserved_across_layer_counts(profile, max_layers):
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=60.0, max_layers=max_layers)
    reference = _total_air_column(
        build_path(profile, sensor_alt_km=0.0, zenith_deg=60.0, max_layers=200)
    )
    col = _total_air_column(path)
    assert col == pytest.approx(reference, rel=0.01)


def test_curtis_godson_h2o_pressure_biased_toward_surface(profile):
    """H2O is concentrated near the ground, so its column-weighted (Curtis-
    Godson) effective pressure in a thick low-altitude layer should exceed
    the layer's simple midpoint pressure -- the whole point of populating
    p_eff_mbar/t_eff_k separately per docs/build_contract.md."""
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=8)
    layer0 = path.layers[0]
    assert layer0.p_eff_mbar[Species.H2O] > layer0.p_mbar
    assert layer0.p_eff_mbar[Species.H2O] >= layer0.p_eff_mbar[Species.CO2]


def test_ds_km_sums_to_total_length(profile):
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=45.0, max_layers=25)
    assert sum(l.ds_km for l in path.layers) == pytest.approx(path.total_length_km, rel=1.0e-9)
