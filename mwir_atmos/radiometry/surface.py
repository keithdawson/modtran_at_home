"""Surface emissivity in the mid-wave infrared.

Values are broadband MWIR (3-5 um) hemispherical emissivities at nadir, taken as
representative of the ECOSTRESS/ASTER spectral library classes:

  * sea water          ~0.98 at nadir, falling steeply past ~60 deg incidence
  * vegetated land     ~0.96, nearly grey across the MWIR
  * desert sand        ~0.90, notably lower than vegetation; quartz has strong
                       reststrahlen structure in the LWIR but is comparatively
                       featureless (and less emissive) through the MWIR
  * snow               ~0.98

Sea water is the only surface given an angular dependence, because a maritime
look-down at grazing incidence is a real use case and the Fresnel drop is large:
emissivity falls from 0.98 at nadir to below 0.7 past 80 deg. Land surfaces are
treated as Lambertian and angle-independent, which is a simplification.

Reflectance is taken as ``1 - emissivity`` (opaque surface, Kirchhoff), and the
reflected component is treated as Lambertian.

Reference: Baldridge et al. (2009), "The ASTER spectral library version 2.0",
Remote Sensing of Environment 113(4), 711-715.
"""

from __future__ import annotations

import math
from typing import Optional

from ..types import SurfaceType

__all__ = ["surface_emissivity", "NADIR_EMISSIVITY"]

#: Representative nadir MWIR emissivity by surface class.
NADIR_EMISSIVITY: dict[SurfaceType, float] = {
    SurfaceType.SEA_WATER: 0.98,
    SurfaceType.LAND_VEGETATED: 0.96,
    SurfaceType.DESERT_SAND: 0.90,
    SurfaceType.SNOW: 0.98,
    SurfaceType.BLACKBODY: 1.0,
}

#: Real part of the refractive index of liquid water averaged over 3-5 um.
_WATER_N_MWIR = 1.35


def _fresnel_reflectance(incidence_deg: float, n: float = _WATER_N_MWIR) -> float:
    """Unpolarised Fresnel reflectance for a dielectric, incidence in degrees."""
    theta_i = math.radians(min(abs(incidence_deg), 89.999))
    sin_t = math.sin(theta_i) / n
    if sin_t >= 1.0:
        return 1.0
    theta_t = math.asin(sin_t)

    cos_i, cos_t = math.cos(theta_i), math.cos(theta_t)
    r_s = ((cos_i - n * cos_t) / (cos_i + n * cos_t)) ** 2
    r_p = ((n * cos_i - cos_t) / (n * cos_i + cos_t)) ** 2
    return 0.5 * (r_s + r_p)


def surface_emissivity(
    surface_type: SurfaceType,
    *,
    incidence_deg: float = 0.0,
    override: Optional[float] = None,
) -> float:
    """MWIR emissivity for a surface, optionally angle-dependent.

    ``incidence_deg`` is measured from the surface normal (0 = nadir view).
    ``override`` short-circuits everything and is returned as-is, which is how
    ``Scenario.surface_emissivity`` takes precedence.
    """
    if override is not None:
        return float(override)

    base = NADIR_EMISSIVITY.get(surface_type, 0.95)
    if surface_type is not SurfaceType.SEA_WATER:
        return base

    # Scale the Fresnel angular shape so that nadir reproduces the tabulated value.
    r_nadir = _fresnel_reflectance(0.0)
    r_theta = _fresnel_reflectance(incidence_deg)
    emissivity = base * (1.0 - r_theta) / (1.0 - r_nadir)
    return max(0.0, min(1.0, emissivity))
