"""Consistent plane-parallel geometry for DISORT, including finite target paths.

The complete column and the line of sight share identical homogeneous slabs.
Columns are integrated vertically by the existing profile integrator. Slant
columns are obtained by exact secant scaling; horizontal paths occupy one slab.
"""
from dataclasses import replace
import numpy as np

from .path import build_path, DEFAULT_SURFACE_REFINEMENT_KM
from ..types import Scenario, AtmosphereProfile, PathGeometry, LayerOptics


def geometry(scenario: Scenario, profile: AtmosphereProfile):
    """Return full column (bottom-up), LOS, column indices and path multipliers."""
    ground, top = float(profile.altitude_km[0]), float(profile.altitude_km[-1])
    sensor = scenario.sensor_alt_km
    if not ground <= sensor <= top:
        raise ValueError("DISORT requires the sensor within the atmospheric column")
    mu = float(np.cos(np.deg2rad(scenario.sensor_zenith_deg)))
    horizontal = abs(mu) < 1e-10
    if horizontal:
        if scenario.source_range_km is None or scenario.source_range_km <= 0:
            raise ValueError("A horizontal plane-parallel view requires a positive source_range_km")
        if scenario.source_alt_km is not None and abs(scenario.source_alt_km - sensor) > 1e-8:
            raise ValueError("A horizontal plane-parallel target must have the sensor altitude")
        distance, end = scenario.source_range_km, sensor
    else:
        boundary = top if mu > 0 else ground
        candidates = [(boundary - sensor) / mu]
        if scenario.source_range_km is not None:
            candidates.append(scenario.source_range_km)
        if scenario.source_alt_km is not None:
            target_range = (scenario.source_alt_km - sensor) / mu
            if target_range < 0:
                raise ValueError("Target altitude is behind the plane-parallel viewing direction")
            candidates.append(target_range)
        distance = min(candidates)
        end = sensor + mu * distance
    if distance <= 0:
        raise ValueError("DISORT line of sight must have positive length")
    extra = [z - ground for z in (sensor, end) if ground < z < top]
    column = build_path(profile, sensor_alt_km=ground, zenith_deg=0.,
                        surface_refinement_km=(*DEFAULT_SURFACE_REFINEMENT_KM, *extra),
                        max_layers=300, uniform_fill=40)
    if horizontal:
        edges = [l.z_top_km for l in column.layers]
        indices = np.array([min(int(np.searchsorted(edges, sensor, side='right')), len(edges)-1)])
        factors = np.array([distance / column.layers[indices[0]].ds_km])
    else:
        lo, hi = sorted((sensor, end))
        indices = np.array([i for i, l in enumerate(column.layers)
                            if l.z_bottom_km >= lo-1e-8 and l.z_top_km <= hi+1e-8])
        if mu < 0:
            indices = indices[::-1]
        factors = np.full(len(indices), 1. / abs(mu))
    layers = []
    for j, (i, factor) in enumerate(zip(indices, factors)):
        original = column.layers[i]
        layers.append(replace(original, index=j, ds_km=original.ds_km*factor,
                              z_bottom_km=sensor if horizontal else original.z_bottom_km,
                              z_top_km=sensor if horizontal else original.z_top_km,
                              air_column_cm2=original.air_column_cm2*factor,
                              column_cm2={sp: n*factor for sp,n in original.column_cm2.items()}))
    hits = not horizontal and mu < 0 and abs(end-ground) < 1e-8
    space = not horizontal and mu > 0 and abs(end-top) < 1e-8
    path = PathGeometry(layers, sensor, scenario.sensor_zenith_deg, scenario.sensor_azimuth_deg,
                        scenario.source_alt_km, float(sum(l.ds_km for l in layers)), hits, space,
                        ground_temp_k=float(profile.temperature_k[0]) if hits else None)
    return column, path, indices, factors


def sightline_optics(column: LayerOptics, path: PathGeometry, indices, factors) -> LayerOptics:
    def scaled(a):
        return a[indices] * factors[:, None]
    return LayerOptics(tau_molecular=scaled(column.tau_molecular),
                       tau_aerosol_abs=scaled(column.tau_aerosol_abs),
                       tau_aerosol_sca=scaled(column.tau_aerosol_sca),
                       tau_rayleigh=scaled(column.tau_rayleigh), grid=column.grid,
                       tau_by_species={sp: scaled(a) for sp,a in column.tau_by_species.items()},
                       aerosol_asymmetry=column.aerosol_asymmetry[indices])
