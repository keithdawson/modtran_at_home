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
        "sensor_aperture_diameter_m", "fine_step_cm1", "instrument_fwhm_cm1",
    )
    nonnegative = (
        "wind_speed_m_s", "co2_ppmv", "surface_altitude_km", "sensor_alt_km",
        "source_alt_km", "source_range_km", "source_area_m2", "turbulence_cn2_ground",
        "line_optical_depth_cutoff",
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
    if scenario.spectral_sampling.value == "resolved" and scenario.sensor_response_file:
        raise ValueError("sensor_response_file is not supported with resolved spectral sampling")
    if type(scenario.disort_streams) is not int or scenario.disort_streams not in (8, 16, 32, 64):
        raise ValueError("disort_streams must be one of 8, 16, 32, 64")
    for name in ('refraction','polarization'):
        if type(getattr(scenario,name)) is not bool: raise ValueError(f'{name} must be boolean')
    if (scenario.refraction or scenario.polarization) and scenario.radiance_solver.value!='spherical_mc':
        raise ValueError('Refraction and polarization require the spherical_mc solver')
    if scenario.refractive_index_file and not scenario.refraction: raise ValueError('Enable refraction to use refractive_index_file')
    if scenario.sensor_response_file and not scenario.polarization: raise ValueError('Enable polarization to apply a sensor analyzer')
    if scenario.aerosol_polarization not in ('hg_depolarizing','mie_sphere'): raise ValueError('Unknown aerosol_polarization model')
    if scenario.surface_polarization not in ('lambertian','fresnel'): raise ValueError('Unknown surface_polarization model')
    if (scenario.aerosol_polarization=='mie_sphere' or scenario.surface_polarization=='fresnel') and not scenario.polarization:
        raise ValueError('Mie and Fresnel polarization options require polarization enabled')
    if scenario.surface_polarization=='fresnel' and scenario.surface_emissivity is not None:
        raise ValueError('Fresnel derives emissivity from the material index; remove surface_emissivity override')
    for name in ('mie_radius_um','mie_index_real','surface_index_real','refraction_reference_um'):
        value=getattr(scenario,name)
        if not math.isfinite(value) or value<=0: raise ValueError(f'{name} must be finite and positive')
    for name in ('mie_index_imag','surface_index_imag'):
        value=getattr(scenario,name)
        if not math.isfinite(value) or value<0: raise ValueError(f'{name} must be finite and nonnegative')
    for name, lo, hi in (("spherical_photons", 128, 1000000), ("spherical_seed", 0, 2**32-1), ("spherical_max_events", 16, 100000)):
        value = getattr(scenario, name)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    if scenario.aerosol.value.startswith("lowtran_"):
        if not 2. <= scenario.visibility_km <= 50.:
            raise ValueError("LOWTRAN table aerosols require visibility_km between 2 and 50")
        if scenario.wavenumber_min_cm1 < 250. or scenario.wavenumber_max_cm1 > 50000.:
            raise ValueError("LOWTRAN aerosol tables support wavelengths from 0.2 to 40 micrometers")
    if scenario.sensor_alt_km < scenario.surface_altitude_km:
        raise ValueError("sensor_alt_km must be at or above surface_altitude_km")
