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
    if scenario.refraction:
        return refracted_geometry(scenario,profile)
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


def refracted_geometry(scenario,profile):
    """One reference wavelength for geometry; range means curved arc length."""
    from .refraction import trace,profile_index,advance
    from .path import DEFAULT_SURFACE_REFINEMENT_KM
    from ..types import PathGeometry
    ground=float(profile.altitude_km[0])
    extra=[scenario.sensor_alt_km-ground]
    if scenario.source_alt_km is not None: extra.append(scenario.source_alt_km-ground)
    column=build_path(profile,sensor_alt_km=ground,zenith_deg=0.,max_layers=300,uniform_fill=40,
        surface_refinement_km=(*DEFAULT_SURFACE_REFINEMENT_KM,*extra))
    heights=np.r_[column.layers[0].z_bottom_km,[l.z_top_km for l in column.layers]]
    radii=EARTH_RADIUS_KM+heights; index=profile_index(scenario,profile,heights)
    p=np.array([0.,0.,EARTH_RADIUS_KM+scenario.sensor_alt_km]); d=direction(scenario.sensor_zenith_deg,scenario.sensor_azimuth_deg)
    flight=trace(p,d,radii,index,scenario.source_range_km if scenario.source_range_km is not None else np.inf)
    ends,_=advance(flight.starts[0],flight.directions[0],flight.lengths[0],flight.layer[0],radii,index)
    end_z=np.linalg.norm(ends,axis=1)-EARTH_RADIUS_KM
    lengths=flight.lengths[0].copy(); hit=bool(flight.hits_ground[0]); space=bool(flight.reaches_space[0])
    if scenario.source_alt_km is not None:
        candidates=np.flatnonzero((abs(end_z-scenario.source_alt_km)<1e-6)&(lengths>1e-9))
        if not len(candidates):
            if scenario.source_range_km is None: raise ValueError('Refracted ray does not reach the target altitude')
        else:
            lengths[candidates[0]+1:]=0; hit &= abs(scenario.source_alt_km-ground)<1e-6
            space &= abs(scenario.source_alt_km-heights[-1])<1e-6
    keep=lengths>1e-9; ids=flight.layer[0,keep]; lengths=lengths[keep]
    if not len(lengths): raise ValueError('Refracted line of sight must have positive length')
    factors=lengths/np.array([column.layers[i].ds_km for i in ids])
    start_z=np.linalg.norm(flight.starts[0,keep],axis=1)-EARTH_RADIUS_KM
    end_z=end_z[keep]; layers=[]
    for j,(i,length,factor,z0,z1) in enumerate(zip(ids,lengths,factors,start_z,end_z)):
        l=column.layers[i]
        layers.append(replace(l,index=j,ds_km=float(length),z_bottom_km=min(z0,z1),z_top_km=max(z0,z1),
            air_column_cm2=l.air_column_cm2*factor,column_cm2={sp:n*factor for sp,n in l.column_cm2.items()}))
    path=PathGeometry(layers,scenario.sensor_alt_km,scenario.sensor_zenith_deg,scenario.sensor_azimuth_deg,
        scenario.source_alt_km,float(sum(lengths)),hit,space,ground_temp_k=float(profile.temperature_k[0]) if hit else None)
    return column,path,ids,factors
