"""Shared validation for Python, YAML, CLI and GUI scenario inputs."""
from numbers import Real
from typing import get_args, get_type_hints
import math

from .types import Scenario


def validate_scenario(scenario: Scenario) -> None:
    """Reject invalid inputs before allocating grids or computing optics."""
    for name, annotation in get_type_hints(Scenario).items():
        value = getattr(scenario, name)
        choices = get_args(annotation) or (annotation,)
        if value is None:
            if type(None) in choices:
                continue
            raise ValueError(f"{name} cannot be null")
        expected = next(t for t in choices if t is not type(None))
        if expected is float:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        elif expected is bool:
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false (without quotes)")
        elif not isinstance(value, expected):
            raise ValueError(f"{name} must be a valid {expected.__name__}")

    positive = (
        "visibility_km", "ground_temp_k", "surface_skin_temp_k", "source_temp_k",
        "resolution_cm1", "wavenumber_min_cm1", "line_wing_cutoff_cm1",
        "sensor_aperture_diameter_m",
    )
    nonnegative = (
        "wind_speed_m_s", "co2_ppmv", "surface_altitude_km", "sensor_alt_km",
        "source_alt_km", "source_range_km", "source_area_m2", "turbulence_cn2_ground",
    )
    for name in positive + nonnegative:
        value = getattr(scenario, name)
        if value is not None and (value <= 0 if name in positive else value < 0):
            raise ValueError(f"{name} must be {'positive' if name in positive else 'non-negative'}")
    bounds = {
        "ground_rh_pct": (0, 100), "surface_emissivity": (0, 1),
        "source_emissivity": (0, 1), "moon_phase": (0, 1),
        "line_strength_cutoff": (0, 1), "sensor_zenith_deg": (0, 180),
        "latitude_deg": (-90, 90), "longitude_deg": (-180, 180),
        "utc_offset_hours": (-24, 24), "sunrise_local_hours": (0, 24),
        "sunset_local_hours": (0, 24),
    }
    for name, (lo, hi) in bounds.items():
        value = getattr(scenario, name)
        if value is not None and not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
    if scenario.wavenumber_max_cm1 <= scenario.wavenumber_min_cm1:
        raise ValueError("wavenumber_max_cm1 must exceed wavenumber_min_cm1")
    if scenario.sensor_alt_km < scenario.surface_altitude_km:
        raise ValueError("sensor_alt_km must be at or above surface_altitude_km")
