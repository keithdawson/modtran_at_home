"""Solar and lunar placement.

The model is deliberately low-precision, so tolerances here are degrees rather
than arcseconds. They are still tight enough to catch sign errors, hemisphere
flips and azimuth-convention mistakes, which are the failure modes that actually
matter for scattering geometry.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from mwir_atmos.ephemeris.moon import illuminated_fraction_from_phase, lunar_position
from mwir_atmos.ephemeris.sun import (
    declination_from_day_length,
    equation_of_time_min,
    line_of_sight_angle_deg,
    solar_declination_deg,
    solar_position,
)
from mwir_atmos.types import Scenario


def _sc(**kw) -> Scenario:
    return Scenario(**kw)


# --------------------------------------------------------------------------
# Solar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date, expected_decl, tol",
    [
        # Solstices are declination turning points, so the fit is tight there.
        (datetime(2026, 6, 21, 12), 23.44, 0.6),
        (datetime(2026, 12, 21, 12), -23.44, 0.6),
        # Equinox crossings move fastest (~0.4 deg/day) and the exact instant
        # drifts through the leap cycle -- 2026's autumnal equinox is the 23rd,
        # so a noon-on-the-22nd sample is legitimately off zero. A degree of
        # slack here reflects the model's stated precision, not a defect.
        (datetime(2026, 3, 20, 12), 0.0, 1.0),
        (datetime(2026, 9, 22, 12), 0.0, 1.0),
    ],
)
def test_declination_matches_the_seasons(date, expected_decl, tol):
    assert solar_declination_deg(date) == pytest.approx(expected_decl, abs=tol)


def test_declination_never_exceeds_the_obliquity():
    for day in range(1, 366):
        d = solar_declination_deg(datetime(2026, 1, 1) + __import__("datetime").timedelta(days=day - 1))
        assert -23.6 <= d <= 23.6


def test_equation_of_time_stays_within_its_known_envelope():
    """EoT ranges roughly -14 to +16 minutes over the year."""
    values = [
        equation_of_time_min(datetime(2026, 1, 1) + __import__("datetime").timedelta(days=d))
        for d in range(365)
    ]
    assert -17.0 < min(values) < -10.0
    assert 12.0 < max(values) < 18.0


def test_sun_is_near_zenith_at_equatorial_equinox_noon():
    sc = _sc(latitude_deg=0.0, longitude_deg=0.0, utc_offset_hours=0.0,
             local_datetime=datetime(2026, 3, 20, 12, 0))
    sun = solar_position(sc)
    assert sun.zenith_deg < 3.0
    assert sun.is_above_horizon


def test_sun_is_below_the_horizon_at_local_midnight():
    sc = _sc(latitude_deg=40.0, longitude_deg=-90.0, utc_offset_hours=-6.0,
             local_datetime=datetime(2026, 1, 15, 0, 30))
    sun = solar_position(sc)
    assert not sun.is_above_horizon
    assert sun.zenith_deg > 90.0


def test_sun_is_south_at_noon_in_the_northern_hemisphere():
    """Azimuth is clockwise from north, so local solar noon is near 180 deg."""
    sc = _sc(latitude_deg=45.0, longitude_deg=0.0, utc_offset_hours=0.0,
             local_datetime=datetime(2026, 6, 21, 12, 0))
    sun = solar_position(sc)
    assert abs(sun.azimuth_deg - 180.0) < 15.0


def test_morning_and_afternoon_azimuths_straddle_the_meridian():
    common = dict(latitude_deg=40.0, longitude_deg=0.0, utc_offset_hours=0.0)
    morning = solar_position(_sc(**common, local_datetime=datetime(2026, 6, 21, 9, 0)))
    evening = solar_position(_sc(**common, local_datetime=datetime(2026, 6, 21, 15, 0)))
    assert morning.azimuth_deg < 180.0 < evening.azimuth_deg


def test_polar_night_keeps_the_sun_down_all_day():
    for hour in range(0, 24, 3):
        sc = _sc(latitude_deg=80.0, longitude_deg=0.0, utc_offset_hours=0.0,
                 local_datetime=datetime(2026, 12, 21, hour, 0))
        assert not solar_position(sc).is_above_horizon


def test_missing_datetime_yields_a_well_defined_night():
    sun = solar_position(_sc(local_datetime=None))
    assert not sun.is_above_horizon


def test_timezone_is_approximated_from_longitude_when_absent():
    """Without an explicit offset the sun should still be up at local noon."""
    for lon in (-120.0, -60.0, 0.0, 75.0, 140.0):
        sc = _sc(latitude_deg=20.0, longitude_deg=lon,
                 local_datetime=datetime(2026, 3, 20, 12, 0))
        assert solar_position(sc).is_above_horizon, lon


def test_distance_factor_tracks_the_eccentric_orbit():
    jan = solar_position(_sc(local_datetime=datetime(2026, 1, 3, 12), utc_offset_hours=0.0))
    jul = solar_position(_sc(local_datetime=datetime(2026, 7, 4, 12), utc_offset_hours=0.0))
    assert jan.distance_factor > jul.distance_factor          # perihelion in January
    assert 1.02 < jan.distance_factor < 1.04
    assert 0.96 < jul.distance_factor < 0.98


# --------------------------------------------------------------------------
# Scattering geometry
# --------------------------------------------------------------------------


def test_line_of_sight_angle_is_zero_when_pointing_at_the_body():
    assert line_of_sight_angle_deg(30.0, 120.0, 30.0, 120.0) == pytest.approx(0.0, abs=1e-6)


def test_line_of_sight_angle_is_180_when_pointing_away():
    assert line_of_sight_angle_deg(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0, abs=1e-6)


def test_line_of_sight_angle_is_90_for_perpendicular_directions():
    assert line_of_sight_angle_deg(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0, abs=1e-6)


def test_azimuth_separation_is_symmetric():
    a = line_of_sight_angle_deg(45.0, 0.0, 45.0, 90.0)
    b = line_of_sight_angle_deg(45.0, 90.0, 45.0, 0.0)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------
# Lunar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase, expected",
    [(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, 0.5), (1.0, 0.0)],
)
def test_illuminated_fraction_follows_the_phase(phase, expected):
    assert illuminated_fraction_from_phase(phase) == pytest.approx(expected, abs=1e-9)


def test_new_moon_sits_close_to_the_sun():
    sc = _sc(latitude_deg=30.0, longitude_deg=0.0, utc_offset_hours=0.0,
             local_datetime=datetime(2026, 6, 21, 12, 0), moon_phase=0.0)
    sun = solar_position(sc)
    moon = lunar_position(sc, sun)
    assert abs(moon.zenith_deg - sun.zenith_deg) < 5.0
    assert moon.illuminated_fraction == pytest.approx(0.0, abs=1e-9)


def test_full_moon_is_up_when_the_sun_is_down():
    sc = _sc(latitude_deg=30.0, longitude_deg=0.0, utc_offset_hours=0.0,
             local_datetime=datetime(2026, 6, 21, 0, 30), moon_phase=0.5)
    sun = solar_position(sc)
    moon = lunar_position(sc, sun)
    assert not sun.is_above_horizon
    assert moon.is_above_horizon
    assert moon.illuminated_fraction == pytest.approx(1.0, abs=1e-9)


def test_full_moon_is_down_at_local_noon():
    sc = _sc(latitude_deg=30.0, longitude_deg=0.0, utc_offset_hours=0.0,
             local_datetime=datetime(2026, 6, 21, 12, 0), moon_phase=0.5)
    moon = lunar_position(sc, solar_position(sc))
    assert not moon.is_above_horizon


# --------------------------------------------------------------------------
# Sunrise/sunset calibration
# --------------------------------------------------------------------------


def test_day_length_recovers_the_solstice_declination():
    """Published 2026-06-21 day lengths should all imply declination ~+23.44."""
    for latitude, day_length in (
        (39.74, 20.523 - 5.532),      # Denver
        (40.42, 21.800 - 6.733),      # Madrid
        (51.48, 20.353 - 3.717),      # Greenwich
    ):
        decl = declination_from_day_length(latitude, day_length)
        assert decl == pytest.approx(23.44, abs=0.4)


def test_equinox_day_is_longer_than_twelve_hours():
    """Refraction and the solar semi-diameter lengthen the apparent day.

    At declination zero the sun's *centre* is at 90.833 deg when its upper limb
    touches the horizon, so an equinox day runs ~12 h 09 m at mid-latitudes, not
    12 h 00 m. Consequently a day of exactly 12.00 h implies a slightly negative
    declination -- the equal-day-and-night date falls a few days before the
    vernal equinox, which is the real and often-surprising 'equilux' effect.
    """
    assert declination_from_day_length(45.0, 12.0) == pytest.approx(-1.18, abs=0.15)
    assert declination_from_day_length(45.0, 12.155) == pytest.approx(0.0, abs=0.1)


def test_impossible_day_length_returns_none():
    assert declination_from_day_length(0.0, 23.9) is None
    assert declination_from_day_length(45.0, 0.0) is None
    assert declination_from_day_length(45.0, 24.0) is None


def test_sunrise_sunset_beats_the_timezone_guess():
    """Madrid keeps CET despite sitting near the Greenwich meridian."""
    lat, lon, true_tz = 40.42, -3.70, 2.0
    when = datetime(2026, 6, 21, 9, 0)
    truth = solar_position(_sc(latitude_deg=lat, longitude_deg=lon,
                               local_datetime=when, utc_offset_hours=true_tz))
    guessed = solar_position(_sc(latitude_deg=lat, longitude_deg=lon, local_datetime=when))
    calibrated = solar_position(_sc(latitude_deg=lat, longitude_deg=lon, local_datetime=when,
                                    sunrise_local_hours=6.733, sunset_local_hours=21.800))

    err_guess = abs(guessed.zenith_deg - truth.zenith_deg)
    err_cal = abs(calibrated.zenith_deg - truth.zenith_deg)
    assert err_guess > 10.0        # the timezone guess is badly wrong here
    assert err_cal < 0.5           # calibration essentially removes it
    assert err_cal < err_guess / 10.0


def test_calibration_puts_the_sun_on_the_horizon_at_the_stated_sunrise():
    sc = _sc(latitude_deg=39.74, longitude_deg=-104.99,
             local_datetime=datetime(2026, 6, 21, 5, 32),
             sunrise_local_hours=5.532, sunset_local_hours=20.523)
    assert solar_position(sc).zenith_deg == pytest.approx(90.833, abs=0.5)


def test_calibration_puts_the_sun_highest_at_the_midpoint():
    common = dict(latitude_deg=39.74, longitude_deg=-104.99,
                  sunrise_local_hours=5.532, sunset_local_hours=20.523)
    noon = 0.5 * (5.532 + 20.523)
    at_noon = solar_position(_sc(**common, local_datetime=datetime(2026, 6, 21, 13, 1)))
    before = solar_position(_sc(**common, local_datetime=datetime(2026, 6, 21, 10, 0)))
    after = solar_position(_sc(**common, local_datetime=datetime(2026, 6, 21, 16, 0)))
    assert at_noon.zenith_deg < before.zenith_deg
    assert at_noon.zenith_deg < after.zenith_deg
    assert abs(noon - 13.0275) < 0.01


def test_calibration_ignores_utc_offset_entirely():
    """With sunrise/sunset supplied, the declared timezone must not matter."""
    common = dict(latitude_deg=40.42, longitude_deg=-3.70,
                  local_datetime=datetime(2026, 6, 21, 9, 0),
                  sunrise_local_hours=6.733, sunset_local_hours=21.800)
    a = solar_position(_sc(**common, utc_offset_hours=0.0))
    b = solar_position(_sc(**common, utc_offset_hours=9.0))
    assert a.zenith_deg == pytest.approx(b.zenith_deg)


@pytest.mark.parametrize("rise, set_", [(20.0, 6.0), (5.0, 5.0), (-1.0, 12.0), (5.0, 25.0)])
def test_nonsensical_sunrise_sunset_falls_back_without_crashing(rise, set_):
    sc = _sc(latitude_deg=40.0, longitude_deg=0.0, utc_offset_hours=0.0,
             local_datetime=datetime(2026, 6, 21, 12, 0),
             sunrise_local_hours=rise, sunset_local_hours=set_)
    fallback = solar_position(_sc(latitude_deg=40.0, longitude_deg=0.0, utc_offset_hours=0.0,
                                  local_datetime=datetime(2026, 6, 21, 12, 0)))
    assert solar_position(sc).zenith_deg == pytest.approx(fallback.zenith_deg)


def test_only_one_of_sunrise_sunset_is_ignored():
    common = dict(latitude_deg=40.0, longitude_deg=0.0, utc_offset_hours=0.0,
                  local_datetime=datetime(2026, 6, 21, 12, 0))
    fallback = solar_position(_sc(**common))
    assert solar_position(_sc(**common, sunrise_local_hours=6.0)).zenith_deg == pytest.approx(
        fallback.zenith_deg
    )


def test_phase_wraps_around_unity():
    common = dict(latitude_deg=30.0, longitude_deg=0.0, utc_offset_hours=0.0,
                  local_datetime=datetime(2026, 6, 21, 3, 0))
    a = lunar_position(_sc(**common, moon_phase=0.25), solar_position(_sc(**common)))
    b = lunar_position(_sc(**common, moon_phase=1.25), solar_position(_sc(**common)))
    assert a.zenith_deg == pytest.approx(b.zenith_deg)
