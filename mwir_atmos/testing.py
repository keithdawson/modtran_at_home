"""Synthetic fixtures so each worker can test without the other's code.

OVERSEER-OWNED. Worker agents must not edit this file.

The path built here uses a deliberately crude exponential atmosphere. It exists
only so that the line-by-line / optics code can be exercised before the real
model atmospheres and spherical path geometry land. It is NOT a physics
reference and must never be imported by production code paths -- only by tests.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    BACKGROUND_VMR_PPMV,
    EARTH_RADIUS_KM,
    LOSCHMIDT_CM3,
    STD_PRESSURE_MBAR,
    STD_TEMP_K,
)
from .types import (
    AtmosphereProfile,
    ModelAtmosphere,
    PathGeometry,
    PathLayer,
    Species,
)

__all__ = ["make_synthetic_profile", "make_synthetic_path"]

_SCALE_HEIGHT_KM = 8.0
_H2O_SCALE_HEIGHT_KM = 2.0


def make_synthetic_profile(n_levels: int = 51, top_km: float = 100.0) -> AtmosphereProfile:
    """Crude isothermal-ish exponential atmosphere for smoke tests."""
    z = np.linspace(0.0, top_km, n_levels)
    p = STD_PRESSURE_MBAR * np.exp(-z / _SCALE_HEIGHT_KM)
    # Simple troposphere lapse then isothermal stratosphere.
    t = np.where(z < 11.0, 288.15 - 6.5 * z, 216.65)

    h2o_surface_ppmv = 7000.0
    vmr = {
        Species.H2O: h2o_surface_ppmv * np.exp(-z / _H2O_SCALE_HEIGHT_KM),
        Species.CO2: np.full_like(z, BACKGROUND_VMR_PPMV["CO2"]),
        Species.O3: 0.05 * np.exp(-((z - 22.0) ** 2) / (2 * 5.0**2)),
        Species.N2O: np.full_like(z, BACKGROUND_VMR_PPMV["N2O"]),
        Species.CO: np.full_like(z, BACKGROUND_VMR_PPMV["CO"]),
        Species.CH4: np.full_like(z, BACKGROUND_VMR_PPMV["CH4"]),
        Species.O2: np.full_like(z, BACKGROUND_VMR_PPMV["O2"]),
    }
    return AtmosphereProfile(
        altitude_km=z,
        pressure_mbar=p,
        temperature_k=t,
        vmr_ppmv=vmr,
        model=ModelAtmosphere.US_STANDARD,
        surface_altitude_km=0.0,
    )


def _number_density_cm3(p_mbar: float, t_k: float) -> float:
    """Ideal-gas total number density, molecules cm^-3."""
    return LOSCHMIDT_CM3 * (p_mbar / STD_PRESSURE_MBAR) * (STD_TEMP_K / t_k)


def make_synthetic_path(
    sensor_alt_km: float = 0.0,
    zenith_deg: float = 0.0,
    source_alt_km: float | None = None,
    n_layers: int = 20,
    profile: AtmosphereProfile | None = None,
) -> PathGeometry:
    """Build a plane-parallel-with-curvature-correction test path.

    Uses a secant airmass with a simple spherical correction; adequate for
    unit-testing optical depth code, wrong for real limb geometry.
    """
    prof = profile if profile is not None else make_synthetic_profile()
    z_top = float(prof.altitude_km[-1]) if source_alt_km is None else float(source_alt_km)
    z0 = float(sensor_alt_km)
    looking_down = zenith_deg > 90.0

    lo, hi = (0.0, z0) if looking_down else (z0, z_top)
    if hi <= lo:
        raise ValueError("degenerate synthetic path")
    edges = np.linspace(lo, hi, n_layers + 1)

    theta = np.radians(min(abs(180.0 - zenith_deg) if looking_down else zenith_deg, 89.9))
    # Chapman-ish spherical secant, bounded so horizontal paths stay finite.
    airmass = 1.0 / np.sqrt(
        1.0 - (EARTH_RADIUS_KM / (EARTH_RADIUS_KM + 0.5 * (lo + hi)) * np.sin(theta)) ** 2
    )

    layers: list[PathLayer] = []
    total_km = 0.0
    order = range(n_layers - 1, -1, -1) if looking_down else range(n_layers)
    for out_idx, i in enumerate(order):
        zb, zt = float(edges[i]), float(edges[i + 1])
        zm = 0.5 * (zb + zt)
        p = float(np.interp(zm, prof.altitude_km, prof.pressure_mbar))
        t = float(np.interp(zm, prof.altitude_km, prof.temperature_k))
        ds_km = (zt - zb) * airmass
        total_km += ds_km

        n_air = _number_density_cm3(p, t)
        air_col = n_air * ds_km * 1.0e5          # km -> cm
        column = {}
        p_eff = {}
        t_eff = {}
        for sp, prof_vmr in prof.vmr_ppmv.items():
            vmr = float(np.interp(zm, prof.altitude_km, prof_vmr)) * 1.0e-6
            column[sp] = air_col * vmr
            p_eff[sp] = p
            t_eff[sp] = t

        layers.append(
            PathLayer(
                index=out_idx,
                z_bottom_km=zb,
                z_top_km=zt,
                ds_km=ds_km,
                p_mbar=p,
                t_k=t,
                rh_pct=50.0,
                air_column_cm2=air_col,
                column_cm2=column,
                p_eff_mbar=p_eff,
                t_eff_k=t_eff,
                aerosol_od_550nm=0.0,
            )
        )

    return PathGeometry(
        layers=layers,
        sensor_alt_km=z0,
        sensor_zenith_deg=zenith_deg,
        sensor_azimuth_deg=0.0,
        source_alt_km=source_alt_km,
        total_length_km=total_km,
        hits_ground=looking_down,
        reaches_space=(not looking_down) and source_alt_km is None,
        ground_temp_k=float(prof.temperature_k[0]) if looking_down else None,
    )
