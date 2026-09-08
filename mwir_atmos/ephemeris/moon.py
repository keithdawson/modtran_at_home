"""Kinematic lunar placement derived from the sun and a phase input.

This is NOT an ephemeris. The moon is placed by the defining relationship
between phase and elongation: the illuminated fraction of the disc is set by the
sun-earth-moon angle, so a phase input alone fixes how far the moon sits from
the sun along its path. Phase 0.0 puts the moon with the sun (new), 0.5 puts it
opposite (full).

Two real effects are deliberately ignored:
  * The 5.14 deg inclination of the lunar orbit to the ecliptic, so the modelled
    declination can be off by several degrees.
  * The ~5.5% variation in lunar distance over an anomalistic month, which
    changes the solid angle by ~11%.

The result is good enough to decide whether the moon is up, roughly where, and
how much of it is lit -- which is all the MWIR radiance model needs, given that
lunar terms are small compared with atmospheric thermal emission.
"""

from __future__ import annotations

import math

from ..types import Scenario, SkyPosition
from .sun import horizon_coordinates, line_of_sight_angle_deg

__all__ = ["lunar_position", "illuminated_fraction_from_phase"]


def illuminated_fraction_from_phase(moon_phase: float) -> float:
    """Fraction of the lunar disc that is lit, 0 at new and 1 at full.

    The terminator is a half-ellipse, so the lit fraction follows
    ``(1 - cos(elongation)) / 2`` with elongation ``2*pi*phase``.
    """
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * float(moon_phase)))


def lunar_position(scenario: Scenario, sun: SkyPosition) -> SkyPosition:
    """Apparent lunar position implied by the solar position and the phase.

    Returns the moon below the horizon when the sun position is undefined
    (no ``local_datetime``), so lunar terms vanish alongside solar ones.
    """
    phase = float(scenario.moon_phase) % 1.0
    lit = illuminated_fraction_from_phase(phase)

    if not (-90.0 <= sun.zenith_deg <= 180.0) or scenario.local_datetime is None:
        return SkyPosition(
            zenith_deg=180.0,
            azimuth_deg=0.0,
            scattering_angle_deg=180.0,
            is_above_horizon=False,
            distance_factor=1.0,
            illuminated_fraction=lit,
        )

    # Recover the sun's hour angle and declination from its horizon coordinates
    # rather than recomputing the date, then offset the moon along the ecliptic.
    lat = math.radians(scenario.latitude_deg)
    sun_zen = math.radians(sun.zenith_deg)
    sun_az = math.radians(sun.azimuth_deg)

    sun_decl = math.asin(
        math.sin(lat) * math.cos(sun_zen)
        + math.cos(lat) * math.sin(sun_zen) * math.cos(sun_az)
    )
    cos_ha_num = math.cos(sun_zen) - math.sin(lat) * math.sin(sun_decl)
    cos_ha_den = math.cos(lat) * math.cos(sun_decl)
    cos_ha = 1.0 if abs(cos_ha_den) < 1.0e-12 else max(-1.0, min(1.0, cos_ha_num / cos_ha_den))
    sun_ha = math.degrees(math.acos(cos_ha))
    if math.sin(sun_az) > 0.0:      # west of the meridian
        sun_ha = -sun_ha

    # The moon trails the sun by 360*phase degrees of elongation; in hour angle
    # that is a lag of the same size (the moon rises later each night).
    moon_ha = (sun_ha - 360.0 * phase + 180.0) % 360.0 - 180.0
    # Declination swings with the sun at new moon and against it at full.
    moon_decl = math.degrees(sun_decl) * math.cos(2.0 * math.pi * phase)

    zenith, azimuth = horizon_coordinates(scenario.latitude_deg, moon_decl, moon_ha)

    return SkyPosition(
        zenith_deg=zenith,
        azimuth_deg=azimuth,
        scattering_angle_deg=line_of_sight_angle_deg(
            scenario.sensor_zenith_deg, scenario.sensor_azimuth_deg, zenith, azimuth
        ),
        is_above_horizon=zenith < 90.0,
        distance_factor=1.0,
        illuminated_fraction=lit,
    )
