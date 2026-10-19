"""Saturation vapor pressure, relative humidity, and surface-condition overrides.

Saturation vapor pressure uses the Buck (1996) enhanced formulation (a
refinement of Buck 1981), a widely used, accurate (<0.2% error over its
range) closed-form fit to the Wexler/Goff-Gratch reference data:

    Buck, A. L. (1996), "Buck Research CR-1A User's Manual", Buck Research
    Instruments L.L.C., Appendix 1.

Relative humidity follows the WMO/standard meteorological convention: the
saturation pressure used is with respect to liquid water for T >= 0 C and
with respect to ice for T < 0 C.

``apply_surface_conditions`` rescales the boundary layer to user-specified
surface temperature/RH and relaxes back to the unmodified model atmosphere
over ``_BLEND_HEIGHT_KM`` (2.5 km, inside docs/build_contract.md's suggested 2-3 km).
Pressure is re-derived by hydrostatic integration of the (possibly modified)
temperature profile rather than simply blended, so the result stays
hydrostatically consistent by construction rather than by luck. The
integration uses a constant g0 (not gravity varying with geometric altitude)
because that is the convention the AFGL tables themselves are evidently built
on -- see the hydrostatic-consistency check in tests/test_profiles.py, which
found a constant-g0 hypsometric check to fit the published tables better
than one with g(z) = g0*(Re/(Re+z))^2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..constants import GAS_CONSTANT_R, G0_M_S2, M_AIR_G_MOL
from ..types import AtmosphereProfile, Species

__all__ = [
    "saturation_vapor_pressure_mbar",
    "relative_humidity_pct",
    "apply_surface_conditions",
    "hydrostatic_pressure_mbar",
]

#: Boundary-layer relaxation depth for apply_surface_conditions (docs/build_contract.md: ~2-3 km).
_BLEND_HEIGHT_KM = 2.5


def saturation_vapor_pressure_mbar(temperature_k):
    """Saturation vapor pressure of water, mbar (== hPa).

    Buck (1996) formulation: liquid-water branch for T >= 0 C, ice branch for
    T < 0 C (the WMO convention for "relative humidity" aloft). Valid over
    roughly -80 C to +50 C, comfortably spanning the model atmospheres here
    (subarctic winter surface ~-16 C to tropical surface ~27 C, plus the
    stratosphere/mesosphere where H2O VMR is a fixed trace value rather than
    saturation-limited).
    """
    t_k = np.asarray(temperature_k, dtype=np.float64)
    t_c = t_k - 273.15
    es_water = 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
    es_ice = 6.1115 * np.exp((23.036 - t_c / 333.7) * (t_c / (279.82 + t_c)))
    return np.where(t_c >= 0.0, es_water, es_ice)


def relative_humidity_pct(temperature_k, pressure_mbar, h2o_vmr_ppmv):
    """Relative humidity (%), from water vapor partial pressure / saturation.

    ``e = vmr * pressure`` (ppmv treated as a mole fraction of the moist-air
    mixture, which is accurate to better than 0.1% at the VMRs involved here).
    Clipped to [0, 100]: supersaturation is not represented and would break
    the aerosol hygroscopic-growth formula in ``optics.aerosol``, which
    diverges as RH -> 100%.
    """
    t_k = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(pressure_mbar, dtype=np.float64)
    vmr = np.asarray(h2o_vmr_ppmv, dtype=np.float64)
    e_mbar = vmr * 1.0e-6 * p
    es_mbar = saturation_vapor_pressure_mbar(t_k)
    es_safe = np.where(es_mbar > 0.0, es_mbar, np.inf)
    rh = 100.0 * e_mbar / es_safe
    return np.clip(rh, 0.0, 100.0)


def hydrostatic_pressure_mbar(
    altitude_km: np.ndarray, temperature_k: np.ndarray, surface_pressure_mbar: float
) -> np.ndarray:
    """Integrate dP/dz = -P g0 M_air / (R T) upward from ``altitude_km[0]``.

    ``altitude_km`` must be strictly increasing and ``temperature_k`` aligned
    with it. Returns pressure (mbar) with ``pressure[0] == surface_pressure_mbar``
    exactly. Used by :func:`apply_surface_conditions` to keep pressure
    consistent with a modified temperature profile.
    """
    z_m = np.asarray(altitude_km, dtype=np.float64) * 1000.0
    t_k = np.asarray(temperature_k, dtype=np.float64)
    integrand = (G0_M_S2 * (M_AIR_G_MOL * 1.0e-3)) / (GAS_CONSTANT_R * t_k)  # 1/m
    d = np.diff(z_m)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    cum_dlnp = np.concatenate(([0.0], np.cumsum(avg * d)))
    return float(surface_pressure_mbar) * np.exp(-cum_dlnp)


def _log_interp1(x_new: float, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.exp(np.interp(x_new, x, np.log(y))))


def apply_surface_conditions(
    profile: AtmosphereProfile,
    *,
    ground_temp_k: Optional[float] = None,
    ground_rh_pct: Optional[float] = None,
    co2_ppmv: Optional[float] = None,
    surface_altitude_km: float = 0.0,
) -> AtmosphereProfile:
    """Override surface temperature/RH/CO2 and/or re-anchor the surface altitude.

    Independent knobs, applied in this order:

    1. **surface_altitude_km** (default 0.0, meaning "leave as-is" when it
       already matches ``profile.altitude_km[0]``): re-anchors the profile's
       base level to a new altitude by log-pressure/log-VMR/linear-T
       interpolation at that altitude, then truncating everything below it.
       Raises if it's below the existing base (would require extrapolating
       the model atmosphere downward, which is not supported).
    2. **ground_temp_k**: shifts temperature by a constant offset at the
       surface, tapering to zero (i.e. the unmodified model atmosphere) over
       ``_BLEND_HEIGHT_KM`` via a raised-cosine (smoothstep-like, zero slope
       at both ends) weight. Pressure is then rebuilt by hydrostatic
       integration of the new temperature profile from the (unchanged)
       surface pressure -- this is the only case, along with #1, that touches
       pressure, so a call with neither leaves the authentic tabulated
       pressure untouched rather than accumulating interpolation error.
    3. **ground_rh_pct**: rescales the H2O VMR profile by a multiplicative
       factor (ratio of target to current surface VMR, both evaluated at
       whatever the surface temperature is *after* step 2) raised to the same
       blend weight as step 2 -- so it decays back to the unmodified profile
       over the same boundary layer, and uses the "new" surface temperature
       if both overrides are given together.
    4. **co2_ppmv**: flat override, no blending (CO2 is well-mixed).

    ``profile.relative_humidity_pct`` is always recomputed at the end so it
    reflects the final T/P/H2O.
    """
    orig_alt = np.asarray(profile.altitude_km, dtype=np.float64)
    p_mbar = np.array(profile.pressure_mbar, dtype=np.float64, copy=True)
    t_k = np.array(profile.temperature_k, dtype=np.float64, copy=True)
    vmr: dict[Species, np.ndarray] = {
        sp: np.array(arr, dtype=np.float64, copy=True) for sp, arr in profile.vmr_ppmv.items()
    }

    if surface_altitude_km < orig_alt[0] - 1.0e-6:
        raise ValueError(
            f"surface_altitude_km={surface_altitude_km!r} km is below this "
            f"profile's base level ({orig_alt[0]:.3f} km); cannot extrapolate "
            "the model atmosphere downward"
        )

    alt = orig_alt
    altitude_changed = not np.isclose(surface_altitude_km, orig_alt[0], atol=1.0e-6)
    if altitude_changed:
        idx = int(np.searchsorted(orig_alt, surface_altitude_km, side="right"))
        if idx > 0 and np.isclose(orig_alt[idx - 1], surface_altitude_km, atol=1.0e-6):
            start = idx - 1
            alt = orig_alt[start:]
            p_mbar = p_mbar[start:]
            t_k = t_k[start:]
            vmr = {sp: v[start:] for sp, v in vmr.items()}
        else:
            p0 = _log_interp1(surface_altitude_km, orig_alt, p_mbar)
            t0 = float(np.interp(surface_altitude_km, orig_alt, t_k))
            v0 = {sp: _log_interp1(surface_altitude_km, orig_alt, v) for sp, v in vmr.items()}
            alt = np.concatenate(([surface_altitude_km], orig_alt[idx:]))
            p_mbar = np.concatenate(([p0], p_mbar[idx:]))
            t_k = np.concatenate(([t0], t_k[idx:]))
            vmr = {sp: np.concatenate(([v0[sp]], v[idx:])) for sp, v in vmr.items()}

    surface_km = float(alt[0])
    dz = np.clip(alt - surface_km, 0.0, _BLEND_HEIGHT_KM)
    weight = 0.5 * (1.0 + np.cos(np.pi * dz / _BLEND_HEIGHT_KM))  # 1 at surface, 0 by +blend km

    temp_changed = ground_temp_k is not None
    if temp_changed:
        delta_t0 = float(ground_temp_k) - t_k[0]
        t_k = t_k + delta_t0 * weight

    if altitude_changed or temp_changed:
        p_mbar = hydrostatic_pressure_mbar(alt, t_k, p_mbar[0])

    if ground_rh_pct is not None:
        rh_target = float(np.clip(ground_rh_pct, 0.0, 100.0))
        es0 = float(saturation_vapor_pressure_mbar(t_k[0]))
        target_e0 = rh_target / 100.0 * es0
        target_h2o0 = target_e0 / p_mbar[0] * 1.0e6  # ppmv
        h2o0 = max(float(vmr[Species.H2O][0]), 1.0e-6)
        ratio = max(target_h2o0, 0.0) / h2o0
        vmr[Species.H2O] = vmr[Species.H2O] * ratio**weight

    if co2_ppmv is not None:
        vmr[Species.CO2] = np.full_like(alt, float(co2_ppmv))

    rh = relative_humidity_pct(t_k, p_mbar, vmr[Species.H2O])

    return AtmosphereProfile(
        altitude_km=alt,
        pressure_mbar=p_mbar,
        temperature_k=t_k,
        vmr_ppmv=vmr,
        model=profile.model,
        surface_altitude_km=float(alt[0]),
        relative_humidity_pct=rh,
    )
