"""Low-precision solar position.

Uses the Spencer (1971) Fourier expansions for solar declination and the
equation of time, as adopted by NOAA's solar calculator. Declination is good to
roughly 0.01 deg and the equation of time to a few tenths of a minute, giving a
solar position accurate to about 0.1-0.3 deg once the timezone approximation
below is accounted for.

Timezone handling is intentionally loose, per the design requirement that the
sun need not be perfectly pinned down by the time zone: if
``Scenario.utc_offset_hours`` is None the offset is approximated as
``round(longitude / 15)``. That can differ from the political timezone by an
hour or more, which maps to up to ~15 deg of hour angle. Supply
``utc_offset_hours`` explicitly when that matters.

Refraction and parallax are not applied. No attempt is made to model the
equation of centre beyond the Fourier fit, so this is not suitable for eclipse
or transit work.

Reference: Spencer, J.W. (1971), "Fourier series representation of the position
of the sun", Search 2(5), 172.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from ..types import Scenario, SkyPosition

__all__ = ["solar_position", "solar_declination_deg", "equation_of_time_min", "line_of_sight_angle_deg"]


def _fractional_year_rad(dt: datetime) -> float:
    """Spencer's fractional year angle, radians."""
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    days_in_year = 366.0 if _is_leap(dt.year) else 365.0
    return 2.0 * math.pi * (day_of_year - 1 + (hour - 12.0) / 24.0) / days_in_year


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def solar_declination_deg(dt: datetime) -> float:
    """Solar declination in degrees (Spencer 1971)."""
    g = _fractional_year_rad(dt)
    decl_rad = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.001480 * math.sin(3 * g)
    )
    return math.degrees(decl_rad)


def equation_of_time_min(dt: datetime) -> float:
    """Equation of time in minutes (Spencer 1971)."""
    g = _fractional_year_rad(dt)
    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )


def sun_earth_distance_factor(dt: datetime) -> float:
    """Irradiance multiplier from the Earth-Sun distance, ~0.967 to ~1.035.

    The orbit is eccentric (e ~ 0.0167), so top-of-atmosphere irradiance varies
    by about +/-3.4% over the year. Perihelion is near 3 January.
    """
    day_of_year = dt.timetuple().tm_yday
    # (a/r)^2 to first order in eccentricity.
    return 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.25)


def _utc_offset_hours(scenario: Scenario) -> float:
    if scenario.utc_offset_hours is not None:
        return float(scenario.utc_offset_hours)
    # Loose default: nearest whole hour of longitude.
    return float(round(scenario.longitude_deg / 15.0))


def line_of_sight_angle_deg(
    los_zenith_deg: float, los_azimuth_deg: float, body_zenith_deg: float, body_azimuth_deg: float
) -> float:
    """Angle between the sensor line of sight and a body, degrees.

    Both directions are unit vectors in the local horizon frame with x toward
    true north, y toward east and z toward zenith; the angle is the arccosine of
    their dot product. This is the scattering angle that the phase function
    needs, so getting the azimuth convention wrong here swaps forward and
    backward scattering.
    """
    lz, la = math.radians(los_zenith_deg), math.radians(los_azimuth_deg)
    bz, ba = math.radians(body_zenith_deg), math.radians(body_azimuth_deg)
    cos_theta = (
        math.sin(lz) * math.cos(la) * math.sin(bz) * math.cos(ba)
        + math.sin(lz) * math.sin(la) * math.sin(bz) * math.sin(ba)
        + math.cos(lz) * math.cos(bz)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def horizon_coordinates(
    latitude_deg: float, declination_deg: float, hour_angle_deg: float
) -> tuple[float, float]:
    """Convert equatorial (declination, hour angle) to (zenith, azimuth) degrees.

    Azimuth is measured clockwise from true north.
    """
    lat = math.radians(latitude_deg)
    decl = math.radians(declination_deg)
    ha = math.radians(hour_angle_deg)

    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zenith = math.degrees(math.acos(cos_zen))

    sin_zen = math.sqrt(max(0.0, 1.0 - cos_zen * cos_zen))
    if sin_zen < 1.0e-9:
        return zenith, 0.0

    cos_az = (math.sin(decl) - math.sin(lat) * cos_zen) / (math.cos(lat) * sin_zen)
    azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if math.sin(ha) > 0.0:          # afternoon: sun has crossed to the west
        azimuth = 360.0 - azimuth
    return zenith, azimuth


#: Geometric zenith angle of the sun's centre at published sunrise/sunset.
#: The disc's upper limb touches the horizon while the centre is still 50'
#: below it: 34' of atmospheric refraction plus the 16' solar semi-diameter.
SUNRISE_ZENITH_DEG = 90.833


def declination_from_day_length(latitude_deg: float, day_length_hours: float) -> Optional[float]:
    """Solar declination implied by an observed day length, degrees.

    Inverts ``cos(z0) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(H0)`` for the
    declination, with ``H0`` the sunrise hour angle (half the day length) and
    ``z0`` = :data:`SUNRISE_ZENITH_DEG`. Returns None when no declination in the
    physical range reproduces that day length -- which happens near the poles,
    and for day lengths that cannot occur at the given latitude.
    """
    if not (0.0 < day_length_hours < 24.0):
        return None
    lat = math.radians(latitude_deg)
    h0 = math.radians(15.0 * day_length_hours / 2.0)
    target = math.cos(math.radians(SUNRISE_ZENITH_DEG))

    def residual(dec_deg: float) -> float:
        d = math.radians(dec_deg)
        return math.sin(lat) * math.sin(d) + math.cos(lat) * math.cos(d) * math.cos(h0) - target

    lo, hi = -23.5, 23.5
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        return None
    for _ in range(80):                      # bisection; ~1e-22 deg after 80 halvings
        mid = 0.5 * (lo + hi)
        if residual(lo) * residual(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _calibrated_from_sunrise_sunset(scenario: Scenario) -> Optional[SkyPosition]:
    """Solar position from observed sunrise/sunset, bypassing the timezone.

    Returns None when the inputs are absent or cannot be inverted, so the caller
    falls back to the date-and-timezone path.
    """
    rise, set_ = scenario.sunrise_local_hours, scenario.sunset_local_hours
    if rise is None or set_ is None or scenario.local_datetime is None:
        return None
    if not (0.0 <= rise < 24.0 and 0.0 < set_ <= 24.0) or set_ <= rise:
        return None

    day_length = set_ - rise
    decl = declination_from_day_length(scenario.latitude_deg, day_length)
    if decl is None:
        return None

    solar_noon_clock = 0.5 * (rise + set_)
    dt = scenario.local_datetime
    clock = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle = 15.0 * (clock - solar_noon_clock)
    hour_angle = (hour_angle + 180.0) % 360.0 - 180.0

    zenith, azimuth = horizon_coordinates(scenario.latitude_deg, decl, hour_angle)
    return SkyPosition(
        zenith_deg=zenith,
        azimuth_deg=azimuth,
        scattering_angle_deg=line_of_sight_angle_deg(
            scenario.sensor_zenith_deg, scenario.sensor_azimuth_deg, zenith, azimuth
        ),
        is_above_horizon=zenith < 90.0,
        distance_factor=sun_earth_distance_factor(dt),
        illuminated_fraction=1.0,
    )


def solar_position(scenario: Scenario) -> SkyPosition:
    """Apparent solar position for the scenario's site and local time.

    When ``Scenario.local_datetime`` is None the sun is reported below the
    horizon with zero illuminated fraction, so every solar term downstream
    evaluates to zero. That makes a bare ``Scenario()`` a well-defined
    night-time / thermal-only case rather than an error.
    """
    dt: Optional[datetime] = scenario.local_datetime
    if dt is None:
        return SkyPosition(
            zenith_deg=180.0,
            azimuth_deg=0.0,
            scattering_angle_deg=180.0,
            is_above_horizon=False,
            distance_factor=1.0,
            illuminated_fraction=1.0,
        )

    # Observed sunrise/sunset, when supplied, pins the sun far better than the
    # longitude-derived timezone can.
    calibrated = _calibrated_from_sunrise_sunset(scenario)
    if calibrated is not None:
        return calibrated

    eqtime = equation_of_time_min(dt)
    decl = solar_declination_deg(dt)
    offset_h = _utc_offset_hours(scenario)

    # True solar time in minutes from local midnight.
    local_min = dt.hour * 60.0 + dt.minute + dt.second / 60.0
    time_offset = eqtime + 4.0 * scenario.longitude_deg - 60.0 * offset_h
    true_solar_time = (local_min + time_offset) % 1440.0

    hour_angle = true_solar_time / 4.0 - 180.0      # degrees, negative before noon
    zenith, azimuth = horizon_coordinates(scenario.latitude_deg, decl, hour_angle)

    return SkyPosition(
        zenith_deg=zenith,
        azimuth_deg=azimuth,
        scattering_angle_deg=line_of_sight_angle_deg(
            scenario.sensor_zenith_deg, scenario.sensor_azimuth_deg, zenith, azimuth
        ),
        is_above_horizon=zenith < 90.0,
        distance_factor=sun_earth_distance_factor(dt),
        illuminated_fraction=1.0,
    )
