"""Spherical-earth slant-path layering (impact-parameter ray tracing).

No refraction: rays are geometric straight lines through the spherical
atmosphere (``constants.EARTH_RADIUS_KM``). This is the LOWTRAN/MODTRAN
"no refractive ray bending" simplification -- real rays curve slightly toward
the surface (refraction adds on the order of a percent to airmass near the
horizon), which this module does not model. Flagged here explicitly per
CONTRACT.md's requirement to document the omission.

Geometry
--------
A straight ray leaving the sensor (geocentric radius
``r_sensor = EARTH_RADIUS_KM + sensor_alt_km``) at zenith angle
``zenith_deg`` from local vertical has, at arc length ``s`` along the ray,
geocentric radius

    r(s)^2 = r_sensor^2 + 2*r_sensor*s*cos(zenith) + s^2

(law of cosines in the plane containing the ray and the Earth's center). This
single relation -- not a plane-parallel secant -- covers every case
CONTRACT.md requires, because it is exact for a straight line through
concentric spherical shells:

* ``zenith < 90``: cos(zenith) > 0, so r(s) increases monotonically with s
  (upward-looking).
* ``zenith == 90`` exactly: cos(zenith) == 0, r(s) still increases
  monotonically for s > 0 -- the sensor *is* its own tangent point. This is
  why a horizontal path gives a large but finite airmass (~35-40 at sea level
  for an ~8 km scale height, matching the classical Chapman-function result
  sqrt(pi*Re / 2H)), never the plane-parallel secant's infinity.
* ``zenith > 90``: cos(zenith) < 0, so r(s) *decreases* to a minimum (the
  tangent / impact-parameter radius ``b = r_sensor*sin(zenith)``) at
  ``s_min = -r_sensor*cos(zenith)``, then increases again. Whether the ray
  hits the ground before reaching ``s_min`` (ordinary downward-looking path)
  or ``b`` clears the ground and the ray "turns over" (limb/tangent path) is
  decided purely by comparing ``b`` to the ground radius -- one formula, no
  separate branch for "did it hit the ground".

Every stopping condition (ground, top-of-atmosphere, an explicit
``source_alt_km``, or an explicit ``source_range_km``) reduces to "solve
r(s) = r_target for the smallest s >= 0" via the quadratic above; the path
terminates at whichever candidate is reached first (smallest arc length).

Each layer is sub-integrated (``_N_SUB`` points) rather than evaluated only
at its midpoint, because limb-geometry layers can be geometrically thick in
altitude while representing a long, slowly-varying path length near a
tangent point -- a bare midpoint rule would misrepresent the column there.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..constants import EARTH_RADIUS_KM, LOSCHMIDT_CM3, STD_PRESSURE_MBAR, STD_TEMP_K, TOA_ALTITUDE_KM
from ..types import AtmosphereProfile, PathGeometry, PathLayer, Species

__all__ = ["build_path"]

#: Sub-integration points per layer for column density / Curtis-Godson averages.
_N_SUB = 9


def _number_density_cm3(p_mbar: np.ndarray, t_k: np.ndarray) -> np.ndarray:
    """Ideal-gas total number density, molecules cm^-3.

    Same formula as ``testing._number_density_cm3``, reimplemented here
    (rather than imported) because ``testing.py`` is overseer-owned and
    explicitly "must never be imported by production code paths".
    """
    return LOSCHMIDT_CM3 * (p_mbar / STD_PRESSURE_MBAR) * (STD_TEMP_K / t_k)


def _roots(r_target: float, b: float, s_min: float) -> list[float]:
    """All s >= 0 solving r(s) == r_target, given impact parameter b and the
    arc length s_min of the ray's closest approach. 0, 1, or 2 values."""
    disc = r_target * r_target - b * b
    if disc < 0.0:
        return []
    root = float(np.sqrt(disc))
    out = []
    for s in (s_min - root, s_min + root):
        if s >= -1.0e-9:
            out.append(max(s, 0.0))
    return out


def _first_root(r_target: float, b: float, s_min: float) -> Optional[float]:
    """Smallest s >= 0 solving r(s) == r_target, or None if never reached."""
    roots = _roots(r_target, b, s_min)
    return min(roots) if roots else None


#: Extra layer boundaries near the surface, as altitudes (km) above the
#: profile's base level. The native AFGL grid is 1 km at the bottom, which is
#: too coarse for the boundary layer: a surface sensor viewing an opaque band
#: sees only the first layer, so a 1 km slab under-reads the surface brightness
#: temperature by ~2-3 K. Spacing starts at 10 m, doubles outward, and merges
#: into the native 1 km grid by 6 km. Override via ``build_path`` when a
#: scenario needs something different; pass an empty sequence to disable.
DEFAULT_SURFACE_REFINEMENT_KM: tuple[float, ...] = (
    0.010, 0.025, 0.050, 0.075,
    0.100, 0.150, 0.200, 0.300, 0.400, 0.500,
    0.650, 0.800, 1.000, 1.250, 1.500, 1.750,
    2.000, 2.500, 3.000, 3.500, 4.000, 5.000, 6.000,
)


def build_path(
    profile: AtmosphereProfile,
    *,
    sensor_alt_km: float,
    zenith_deg: float,
    azimuth_deg: float = 0.0,
    source_alt_km: Optional[float] = None,
    source_range_km: Optional[float] = None,
    max_layers: int = 140,
    surface_refinement_km: Optional[Sequence[float]] = None,
    uniform_fill: int = 40,
) -> PathGeometry:
    """Resolve a sensor-to-source slant path through ``profile`` into layers.

    ``azimuth_deg`` is recorded on the result but does not affect the
    geometry -- the atmosphere is treated as horizontally stratified (the
    standard simplification for a 1-D vertical profile), so only the zenith
    angle determines the slant path.

    The path always stops at whichever of {ground, ``source_alt_km``,
    ``source_range_km``, top of atmosphere} is reached *first* (smallest arc
    length from the sensor). Raises ``ValueError`` if ``source_alt_km`` is
    geometrically unreachable (below the ray's tangent altitude), or if
    ``sensor_alt_km`` is below the profile's base level.

    Altitudes above ``constants.TOA_ALTITUDE_KM`` are not tabulated; profile
    values there are edge-clamped to the 100 km values by interpolation
    (that layer is already ~3e-4 mbar, so the approximation is immaterial).
    """
    if sensor_alt_km < profile.altitude_km[0] - 1.0e-6:
        raise ValueError(
            f"sensor_alt_km={sensor_alt_km} is below the profile's base level "
            f"({profile.altitude_km[0]} km)"
        )
    if not (0.0 <= zenith_deg <= 180.0):
        raise ValueError(f"zenith_deg must be in [0, 180], got {zenith_deg}")
    if max_layers < 1:
        raise ValueError("max_layers must be >= 1")
    if surface_refinement_km is None:
        surface_refinement_km = DEFAULT_SURFACE_REFINEMENT_KM

    r_sensor = EARTH_RADIUS_KM + sensor_alt_km
    r_ground = EARTH_RADIUS_KM + float(profile.altitude_km[0])
    r_top = EARTH_RADIUS_KM + TOA_ALTITUDE_KM

    zen_rad = np.radians(zenith_deg)
    mu0 = float(np.cos(zen_rad))
    b = float(r_sensor * np.sin(zen_rad))  # impact parameter (>= 0)
    s_min = -r_sensor * mu0  # arc length at closest approach to Earth's center
    looking_up = mu0 >= 0.0  # zenith <= 90: r(s) increases monotonically for s >= 0

    def _radius(s):
        return np.sqrt(r_sensor * r_sensor + 2.0 * r_sensor * mu0 * s + s * s)

    # --- candidate stopping distances, first-reached wins ------------------
    candidates: dict[str, float] = {}

    if not looking_up:
        s_ground = _first_root(r_ground, b, s_min)
        if s_ground is not None:
            candidates["ground"] = s_ground

    s_top = _first_root(r_top, b, s_min)
    if s_top is not None:
        candidates["top"] = s_top

    if source_alt_km is not None:
        r_src = EARTH_RADIUS_KM + source_alt_km
        if r_src < b - 1.0e-9:
            raise ValueError(
                f"source_alt_km={source_alt_km} is geometrically unreachable: this "
                f"line of sight's closest approach to the surface is "
                f"{b - EARTH_RADIUS_KM:.3f} km, above the requested source altitude"
            )
        s_src = _first_root(r_src, b, s_min)
        if s_src is not None:
            candidates["source_alt"] = s_src

    if source_range_km is not None:
        if source_range_km <= 0.0:
            raise ValueError("source_range_km must be positive")
        candidates["source_range"] = float(source_range_km)

    if not candidates:
        raise ValueError("no valid path: unable to resolve a stopping condition")

    stop_kind = min(candidates, key=candidates.get)
    s_end = candidates[stop_kind]
    if s_end <= 1.0e-6:
        raise ValueError(
            "degenerate path: zero length (sensor is already at the "
            "stopping surface, e.g. a ground-level sensor looking below "
            "the local horizontal)"
        )

    hits_ground = stop_kind == "ground"
    reaches_space = stop_kind == "top"
    tangent_alt_km: Optional[float] = None
    if (not looking_up) and (not hits_ground) and 0.0 < s_min < s_end:
        tangent_alt_km = b - EARTH_RADIUS_KM

    # --- breakpoints: native profile-level crossings + a uniform backup grid,
    # subsampled to at most max_layers segments -----------------------------
    breakpoints = [0.0, s_end]
    if tangent_alt_km is not None:
        breakpoints.append(s_min)
    for z_lvl in profile.altitude_km:
        r_lvl = EARTH_RADIUS_KM + float(z_lvl)
        for s_cross in _roots(r_lvl, b, s_min):
            if 0.0 < s_cross < s_end:
                breakpoints.append(s_cross)
    # Boundary-layer refinement: the AFGL levels are 1 km apart at the bottom,
    # which is far too coarse where the lapse rate and humidity gradient are
    # steepest. In an opaque band a surface sensor sees essentially only the
    # first layer, so a 1 km slab reports its own mid-layer temperature -- about
    # 3 K below the surface value at a standard lapse rate. These extra levels
    # are altitudes above the surface, refined hardest near the ground and
    # relaxing back to the native profile spacing by ~6 km.
    z_surface = float(profile.altitude_km[0])
    for dz in surface_refinement_km:
        r_lvl = EARTH_RADIUS_KM + z_surface + float(dz)
        for s_cross in _roots(r_lvl, b, s_min):
            if 0.0 < s_cross < s_end:
                breakpoints.append(s_cross)

    breakpoints.extend(np.linspace(0.0, s_end, max(2, uniform_fill + 1)).tolist())

    bp = np.unique(np.round(np.array(breakpoints, dtype=np.float64), 9))
    bp = bp[(bp >= 0.0) & (bp <= s_end)]
    if bp[0] > 1.0e-9:
        bp = np.concatenate(([0.0], bp))
    if bp[-1] < s_end - 1.0e-9:
        bp = np.concatenate((bp, [s_end]))

    if bp.size - 1 > max_layers:
        keep = set(np.round(np.linspace(0, bp.size - 1, max_layers + 1)).astype(int).tolist())
        if tangent_alt_km is not None:
            tangent_idx = int(np.argmin(np.abs(bp - s_min)))
            if tangent_idx not in keep:
                keep.add(tangent_idx)
                droppable = sorted(k for k in keep if k not in (0, bp.size - 1, tangent_idx))
                if len(keep) > max_layers + 1 and droppable:
                    keep.discard(droppable[len(droppable) // 2])
        bp = bp[sorted(keep)]

    n_layers = bp.size - 1
    if n_layers < 1:
        raise ValueError("degenerate path: fewer than one layer resolved")

    # --- build layers, sensor-to-source order (layer 0 adjacent to sensor) -
    species_list = list(profile.vmr_ppmv.keys())
    have_rh = profile.relative_humidity_pct is not None
    layers: list[PathLayer] = []
    total_km = 0.0
    for i in range(n_layers):
        s0, s1 = float(bp[i]), float(bp[i + 1])
        ds_km = s1 - s0
        total_km += ds_km

        s_sub = np.linspace(s0, s1, _N_SUB)
        z_sub = _radius(s_sub) - EARTH_RADIUS_KM

        p_sub = np.interp(z_sub, profile.altitude_km, profile.pressure_mbar)
        t_sub = np.interp(z_sub, profile.altitude_km, profile.temperature_k)
        n_sub = _number_density_cm3(p_sub, t_sub)

        air_column_cm2 = float(np.trapezoid(n_sub, s_sub) * 1.0e5)  # km -> cm

        column_cm2: dict[Species, float] = {}
        p_eff_mbar: dict[Species, float] = {}
        t_eff_k: dict[Species, float] = {}
        for sp in species_list:
            vmr_sub = np.interp(z_sub, profile.altitude_km, profile.vmr_ppmv[sp]) * 1.0e-6
            nsp_sub = n_sub * vmr_sub
            column_cm2[sp] = float(np.trapezoid(nsp_sub, s_sub) * 1.0e5)
            weight_sum = float(np.trapezoid(nsp_sub, s_sub))
            if weight_sum > 0.0:
                p_eff_mbar[sp] = float(np.trapezoid(nsp_sub * p_sub, s_sub) / weight_sum)
                t_eff_k[sp] = float(np.trapezoid(nsp_sub * t_sub, s_sub) / weight_sum)
            else:
                # No absorber in this layer at all: fall back to the
                # unweighted layer-mean state so nothing downstream divides
                # by a zero-amount column.
                p_eff_mbar[sp] = float(np.mean(p_sub))
                t_eff_k[sp] = float(np.mean(t_sub))

        z_mid = float(_radius(np.array([0.5 * (s0 + s1)]))[0] - EARTH_RADIUS_KM)
        p_mid = float(np.interp(z_mid, profile.altitude_km, profile.pressure_mbar))
        t_mid = float(np.interp(z_mid, profile.altitude_km, profile.temperature_k))
        rh_mid = (
            float(np.interp(z_mid, profile.altitude_km, profile.relative_humidity_pct))
            if have_rh
            else 0.0
        )

        layers.append(
            PathLayer(
                index=i,
                z_bottom_km=float(min(z_sub[0], z_sub[-1])),
                z_top_km=float(max(z_sub[0], z_sub[-1])),
                ds_km=ds_km,
                p_mbar=p_mid,
                t_k=t_mid,
                rh_pct=rh_mid,
                air_column_cm2=air_column_cm2,
                column_cm2=column_cm2,
                p_eff_mbar=p_eff_mbar,
                t_eff_k=t_eff_k,
                aerosol_od_550nm=0.0,  # populated by optics.aerosol.aerosol_optical_depth
            )
        )

    ground_temp_k = float(profile.temperature_k[0]) if hits_ground else None

    return PathGeometry(
        layers=layers,
        sensor_alt_km=float(sensor_alt_km),
        sensor_zenith_deg=float(zenith_deg),
        sensor_azimuth_deg=float(azimuth_deg),
        source_alt_km=source_alt_km,
        total_length_km=total_km,
        hits_ground=hits_ground,
        reaches_space=reaches_space,
        tangent_alt_km=tangent_alt_km,
        ground_temp_k=ground_temp_k,
    )
