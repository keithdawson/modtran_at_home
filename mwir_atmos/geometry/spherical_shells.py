"""Straight characteristics through concentric, homogeneous spherical shells."""
from dataclasses import replace
import numpy as np
from ..constants import EARTH_RADIUS_KM
from .path import build_path


def direction(zenith, azimuth):
    """Local north/east/up coordinates, with the observer on the +z axis."""
    z, a = np.deg2rad([zenith, azimuth])
    return np.array([np.sin(z)*np.cos(a), np.sin(z)*np.sin(a), np.cos(z)])


def segments(position, travel, radii, limit=np.inf):
    """Batch ray/shell intersections; return distances, shell IDs, end, ground.

    Directions must be unit length. Rays start within the atmosphere (including
    either boundary). Zero-length intervals at shell crossings are retained.
    """
    p, d = np.atleast_2d(position), np.atleast_2d(travel)
    dot = np.sum(p*d, axis=1)
    r2 = np.sum(p*p, axis=1)
    disc = dot[:, None]**2 + radii[None, :]**2-r2[:, None]
    root = np.sqrt(np.maximum(disc, 0.))
    near, far = -dot[:, None]-root, -dot[:, None]+root
    ground = (disc[:, 0] >= 0) & (dot < 0) & (near[:, 0] >= -1e-7)
    end = np.where(ground, np.maximum(near[:, 0], 0.), far[:, -1])
    ground &= end <= np.asarray(limit)+1e-8
    end = np.maximum(np.minimum(end, limit), 0.)
    roots = np.concatenate([near, far], axis=1)
    roots = np.where(np.concatenate([disc, disc], axis=1) >= 0, roots, 0.)
    roots = np.clip(roots, 0., end[:, None])
    edges = np.sort(np.column_stack([np.zeros(len(p)), roots, end]), axis=1)
    mid = (edges[:, 1:]+edges[:, :-1])/2
    radius = np.sqrt(np.maximum(r2[:, None]+2*dot[:, None]*mid+mid**2, 0.))
    ids = np.clip(np.searchsorted(radii, radius, side='right')-1, 0, len(radii)-2)
    return edges, ids, end, ground


def geometry(scenario, profile):
    """Use one shell discretization for the LOS and all scattered photon paths."""
    if not profile.altitude_km[0] <= scenario.sensor_alt_km <= profile.altitude_km[-1]:
        raise ValueError('Spherical MC requires the sensor within the atmospheric column')
    original = build_path(profile, sensor_alt_km=scenario.sensor_alt_km,
                          zenith_deg=scenario.sensor_zenith_deg, azimuth_deg=scenario.sensor_azimuth_deg,
                          source_alt_km=scenario.source_alt_km, source_range_km=scenario.source_range_km)
    column = build_path(profile, sensor_alt_km=float(profile.altitude_km[0]), zenith_deg=0.,
                        max_layers=300, uniform_fill=40)
    radii = EARTH_RADIUS_KM + np.r_[column.layers[0].z_bottom_km, [l.z_top_km for l in column.layers]]
    p = np.array([0., 0., EARTH_RADIUS_KM+scenario.sensor_alt_km])
    d = direction(scenario.sensor_zenith_deg, scenario.sensor_azimuth_deg)
    edges, ids, _, _ = segments(p, d, radii, original.total_length_km)
    ds = np.diff(edges[0]); keep = ds > 1e-10
    indices = ids[0, keep]; lengths = ds[keep]
    factors = lengths / np.array([column.layers[i].ds_km for i in indices])
    layers = []
    for j, (i, length, factor, start, stop) in enumerate(zip(indices, lengths, factors, edges[0,:-1][keep], edges[0,1:][keep])):
        l = column.layers[i]
        z = np.linalg.norm(p+np.array([start,stop])[:,None]*d, axis=1)-EARTH_RADIUS_KM
        layers.append(replace(l, index=j, ds_km=float(length), z_bottom_km=float(min(z)), z_top_km=float(max(z)),
                              air_column_cm2=l.air_column_cm2*factor,
                              column_cm2={sp:n*factor for sp,n in l.column_cm2.items()}))
    return column, replace(original, layers=layers), indices, factors
