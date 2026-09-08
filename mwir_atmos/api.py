"""Top-level orchestration: a :class:`Scenario` in, a :class:`ScenarioResult` out.

OVERSEER-OWNED. This is the single entry point the CLI and the web GUI both call,
so that they cannot drift apart in behaviour.

Pipeline:
    model atmosphere -> surface overrides -> slant path -> layer optics
    -> transmittance -> (radiance, once the radiometry layer exists)
"""

from __future__ import annotations

from typing import Optional

from .atmosphere.humidity import apply_surface_conditions
from .atmosphere.profiles import get_profile
from .geometry.path import build_path
from .optics.transmittance import compute_layer_optics, path_transmittance
from .spectral import make_grid
from .validation import validate_scenario
from .types import (
    AtmosphereProfile,
    LayerOptics,
    PathGeometry,
    Scenario,
    ScenarioResult,
    SpectralGrid,
)

__all__ = ["build_profile", "build_scenario_path", "run_scenario"]


def build_profile(scenario: Scenario) -> AtmosphereProfile:
    """Model atmosphere with the scenario's surface overrides applied."""
    profile = get_profile(scenario.atmosphere)
    needs_override = (
        scenario.ground_temp_k is not None
        or scenario.ground_rh_pct is not None
        or scenario.co2_ppmv is not None
        or scenario.surface_altitude_km != 0.0
    )
    if not needs_override:
        return profile
    return apply_surface_conditions(
        profile,
        ground_temp_k=scenario.ground_temp_k,
        ground_rh_pct=scenario.ground_rh_pct,
        co2_ppmv=scenario.co2_ppmv,
        surface_altitude_km=scenario.surface_altitude_km,
    )


def build_scenario_path(scenario: Scenario, profile: AtmosphereProfile) -> PathGeometry:
    """Slant path implied by the scenario's sensor and source placement."""
    return build_path(
        profile,
        sensor_alt_km=scenario.sensor_alt_km,
        zenith_deg=scenario.sensor_zenith_deg,
        azimuth_deg=scenario.sensor_azimuth_deg,
        source_alt_km=scenario.source_alt_km,
        source_range_km=scenario.source_range_km,
    )


def run_scenario(
    scenario: Scenario,
    *,
    include_radiance: bool = True,
    include_turbulence: bool = True,
    grid: Optional[SpectralGrid] = None,
) -> ScenarioResult:
    """Run a full scenario.

    ``include_radiance`` and ``include_turbulence`` degrade gracefully: if the
    corresponding module has not been built yet, the matching field of the
    result is left as ``None`` rather than raising. Transmittance is always
    computed.
    """
    validate_scenario(scenario)
    if grid is None:
        grid = make_grid(
            scenario.wavenumber_min_cm1,
            scenario.wavenumber_max_cm1,
            scenario.resolution_cm1,
        )

    profile = build_profile(scenario)
    path = build_scenario_path(scenario, profile)

    optics: LayerOptics = compute_layer_optics(
        path,
        grid,
        aerosol=scenario.aerosol,
        visibility_km=scenario.visibility_km,
        wind_speed_m_s=scenario.wind_speed_m_s,
        wing_cutoff_cm1=scenario.line_wing_cutoff_cm1,
        min_strength_fraction=scenario.line_strength_cutoff,
    )
    transmittance = path_transmittance(optics)

    result = ScenarioResult(
        scenario=scenario,
        grid=grid,
        profile=profile,
        path=path,
        transmittance=transmittance,
    )

    if include_radiance:
        _attach_radiance(result, optics)
    if include_turbulence and scenario.turbulence.value != "none":
        _attach_turbulence(result)

    return result


def _attach_radiance(result: ScenarioResult, optics: LayerOptics) -> None:
    """Populate radiance and sky positions if the radiometry layer is present."""
    try:
        from .ephemeris.moon import lunar_position
        from .ephemeris.sun import solar_position
        from .radiometry.radiance import compute_radiance
    except ImportError:
        return  # wave 2 not built yet; transmittance-only result is still valid

    scenario = result.scenario
    sun = solar_position(scenario)
    moon = lunar_position(scenario, sun)
    result.sun = sun
    result.moon = moon
    result.radiance = compute_radiance(
        scenario=scenario,
        grid=result.grid,
        profile=result.profile,
        path=result.path,
        optics=optics,
        transmittance=result.transmittance,
        sun=sun,
        moon=moon,
    )


def _attach_turbulence(result: ScenarioResult) -> None:
    """Populate turbulence metrics if the turbulence layer is present."""
    try:
        from .turbulence.metrics import compute_turbulence
    except ImportError:
        return  # wave 3 stretch goal not built yet

    result.turbulence = compute_turbulence(
        scenario=result.scenario, path=result.path, profile=result.profile
    )
