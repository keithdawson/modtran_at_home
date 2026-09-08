"""LOWTRAN-style boundary-layer aerosol extinction: rural/urban/maritime/desert.

This is a *parametric* aerosol model, not a transcription of LOWTRAN/MODTRAN's
own Mie-scattering lookup tables (those are extensive numerical databases
keyed on component mixtures and size distributions that are not reproduced
here). It is built to have the right physical shape and the right order of
magnitude, with every functional form drawn from the cited literature; the
per-type numeric endpoints (Angstrom exponent, single-scattering albedo,
asymmetry parameter) are representative values consistent with those
references rather than digitized table lookups -- flagged explicitly here (and
in the dev-beta report) rather than presented as exact LOWTRAN output.

Building blocks
----------------
1. **Koschmieder relation** (visibility -> extinction at 0.55 um):
   sigma_550nm[km^-1] = 3.912 / visibility_km. 3.912 = -ln(0.02), the
   Rayleigh/Koschmieder contrast-threshold constant.
   (Koschmieder, H. (1924), Beitr. Phys. Freien Atmos., 12, 33; Middleton,
   W.E.K. (1952), "Vision Through the Atmosphere".)

2. **Vertical decay**: boundary-layer aerosol types (rural/urban/desert)
   decay with scale height ~1 km; navy-maritime ~2 km (typical marine
   boundary layer depth); "tropospheric" is LOWTRAN's extended/background
   type and decays much more slowly (~8 km, i.e. present through the free
   troposphere). These are standard boundary-layer-meteorology depths, not
   from a specific numeric LOWTRAN table.

3. **Spectral shape**: an Angstrom power law,
   sigma(lambda)/sigma(0.55um) = (lambda/0.55um)^-alpha, with alpha (per
   type) set from the qualitative behavior in Shettle, E.P. & Fenn, R.W.
   (1979), "Models for the Aerosols of the Lower Atmosphere and the Effects
   of Humidity Variations on Their Optical Properties," AFGL-TR-79-0214, and
   d'Almeida, G.A., Koepke, P., Shettle, E.P. (1991), "Atmospheric Aerosols:
   Global Climatology and Radiative Characteristics" -- fine-mode-dominated
   continental aerosols (rural/urban) have strong wavelength dependence
   (alpha ~1.0-1.3), large-particle maritime and dust aerosols are much
   greyer (alpha ~0.3-0.5).

4. **Hygroscopic growth**: the Hanel (1976) power-law humidification factor,
   f(RH) = ((1 - RH/100)/(1 - RH_ref/100))^-gamma, referenced to RH_ref = 50%
   (LOWTRAN's conventional "as-specified-visibility" reference humidity), RH
   clipped to 98% to keep f(RH) finite. gamma (per type) is largest for the
   strongly hygroscopic sea-salt-dominated maritime type and ~0 for the
   largely insoluble desert mineral dust type.
   (Hanel, G. (1976), "The Properties of Atmospheric Aerosol Particles as
   Functions of the Relative Humidity," Adv. Geophys., 19, 73-188.)

5. **Navy-maritime wind speed**: a simplified power-law surface-extinction
   scaling referenced to 5 m/s (the Scenario default), standing in for the
   full Navy Aerosol Model's wind-driven sea-salt generation (Gathman, S.G.
   (1983), "A Preliminary Description of the Navy Oceanic Vertical Aerosol
   Model," NRL Memo. Report 5157). NOT a transcription of that model's
   tables.

6. **Absorption/scattering split and asymmetry**: single-scattering albedo
   omega0(lambda) and asymmetry g(lambda) are linearly interpolated between
   representative values at 3 and 5 um (clamped outside [3,5] um) per type.
   Every endpoint is kept at or below 0.42, well clear of 0.5, so that
   absorption dominates scattering for every type at every wavelength in the
   grid (CONTRACT.md: "scattering is weak, absorption dominates" in the
   MWIR, stated without exception for any one aerosol type) -- since the
   aggregate single-scattering albedo returned by any spectral integration
   is a weighted average of omega0(lambda), bounding every endpoint below
   0.5 bounds the aggregate below 0.5 too, regardless of how a caller
   integrates it. Within that ceiling, maritime and desert (large particles,
   weakly-to-moderately absorbing material, size parameter still O(1) at
   3-5 um) are given the highest omega0, and urban (small, soot-rich,
   strongly absorbing) the lowest.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import AerosolModel, PathLayer, SpectralGrid

__all__ = ["aerosol_optical_depth"]

#: LOWTRAN's conventional reference RH for a stated visibility (%).
_RH_REFERENCE_PCT = 50.0
#: Cap RH before evaluating the Hanel growth factor (avoids the 1/(1-RH) pole).
_RH_GROWTH_CAP_PCT = 98.0
#: Reference wind speed for the navy-maritime model == Scenario's default.
_NAVY_REFERENCE_WIND_M_S = 5.0
_NAVY_WIND_EXPONENT = 0.85

# alpha: Angstrom exponent (spectral falloff from 0.55 um).
# gamma: Hanel hygroscopic-growth exponent.
# omega0: (value at 3 um, value at 5 um) single-scattering albedo.
# g: (value at 3 um, value at 5 um) asymmetry parameter.
# scale_height_km: vertical e-folding decay of the aerosol loading.
_AEROSOL_PARAMS: dict[AerosolModel, dict[str, object]] = {
    AerosolModel.RURAL: dict(
        alpha=1.30, gamma=0.25, omega0=(0.32, 0.18), g=(0.70, 0.55), scale_height_km=1.0
    ),
    AerosolModel.URBAN: dict(
        alpha=1.10, gamma=0.35, omega0=(0.25, 0.12), g=(0.65, 0.50), scale_height_km=1.0
    ),
    AerosolModel.NAVY_MARITIME: dict(
        alpha=0.50, gamma=0.60, omega0=(0.42, 0.32), g=(0.80, 0.76), scale_height_km=2.0
    ),
    AerosolModel.DESERT: dict(
        alpha=0.30, gamma=0.05, omega0=(0.38, 0.28), g=(0.75, 0.70), scale_height_km=2.0
    ),
    AerosolModel.TROPOSPHERIC: dict(
        alpha=1.00, gamma=0.25, omega0=(0.30, 0.16), g=(0.68, 0.55), scale_height_km=8.0
    ),
}


def _hanel_growth_factor(rh_pct: float, gamma: float) -> float:
    rh = min(max(rh_pct, 0.0), _RH_GROWTH_CAP_PCT)
    return ((1.0 - rh / 100.0) / (1.0 - _RH_REFERENCE_PCT / 100.0)) ** (-gamma)


def aerosol_optical_depth(
    layers: Sequence[PathLayer],
    grid: SpectralGrid,
    model: AerosolModel,
    *,
    visibility_km: float = 23.0,
    wind_speed_m_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LOWTRAN-style aerosol absorption/scattering optical depth and asymmetry.

    Returns ``(tau_absorption, tau_scattering, asymmetry_g)``, each shape
    ``(len(layers), grid.n)``. ``AerosolModel.NONE`` (or an empty ``layers``)
    returns all zeros.

    Side effect: also writes each layer's 0.55 um aerosol optical depth into
    ``layer.aerosol_od_550nm`` (the reference-wavelength AOD used internally
    to anchor the spectral scaling to the Koschmieder visibility relation --
    0.55 um is far outside the MWIR ``grid``, so it cannot be read back out of
    the returned spectral arrays; this field is the only place it is
    recorded). ``layers`` is mutated in place for this purpose.
    """
    n_layers = len(layers)
    n_wn = grid.n

    if model == AerosolModel.NONE or n_layers == 0:
        for layer in layers:
            layer.aerosol_od_550nm = 0.0
        zeros = np.zeros((n_layers, n_wn), dtype=np.float64)
        return zeros, zeros.copy(), zeros.copy()

    if visibility_km <= 0.0:
        raise ValueError("visibility_km must be positive")
    if model not in _AEROSOL_PARAMS:
        raise ValueError(f"unknown aerosol model: {model!r}")

    params = _AEROSOL_PARAMS[model]
    alpha = float(params["alpha"])
    gamma = float(params["gamma"])
    h_km = float(params["scale_height_km"])
    omega0_3, omega0_5 = params["omega0"]
    g_3, g_5 = params["g"]

    if model == AerosolModel.NAVY_MARITIME:
        wind = max(float(wind_speed_m_s), 0.1)
        wind_factor = (wind / _NAVY_REFERENCE_WIND_M_S) ** _NAVY_WIND_EXPONENT
    else:
        wind_factor = 1.0

    sigma_550_ref_per_km = (3.912 / float(visibility_km)) * wind_factor  # Koschmieder

    lam = grid.wavelength_um  # (n_wn,)
    t = np.clip((lam - 3.0) / (5.0 - 3.0), 0.0, 1.0)
    omega0 = omega0_3 + (omega0_5 - omega0_3) * t
    g_spectrum = g_3 + (g_5 - g_3) * t
    spectral_shape = (lam / 0.55) ** (-alpha)  # (n_wn,)

    tau_abs = np.zeros((n_layers, n_wn), dtype=np.float64)
    tau_sca = np.zeros((n_layers, n_wn), dtype=np.float64)
    g_out = np.zeros((n_layers, n_wn), dtype=np.float64)

    for i, layer in enumerate(layers):
        z_mid = 0.5 * (layer.z_bottom_km + layer.z_top_km)
        f_rh = _hanel_growth_factor(layer.rh_pct, gamma)
        sigma_550 = sigma_550_ref_per_km * np.exp(-z_mid / h_km) * f_rh
        tau_550_layer = sigma_550 * layer.ds_km
        layer.aerosol_od_550nm = float(tau_550_layer)

        tau_ext = tau_550_layer * spectral_shape  # (n_wn,)
        tau_sca[i, :] = tau_ext * omega0
        tau_abs[i, :] = tau_ext * (1.0 - omega0)
        g_out[i, :] = g_spectrum

    return tau_abs, tau_sca, g_out
