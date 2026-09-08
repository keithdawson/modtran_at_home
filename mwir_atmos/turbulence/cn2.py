"""Vertical profiles of the refractive-index structure parameter Cn^2(h).

``h`` is always **height above local ground level** (metres), i.e. the same
convention :mod:`mwir_atmos.turbulence.metrics` feeds in after subtracting
``AtmosphereProfile.surface_altitude_km`` from a layer's altitude -- these
are empirical boundary-layer/free-troposphere models, not derived from the
thermodynamic ``AtmosphereProfile`` (temperature/pressure/humidity), so they
take a bare height array, not a profile.

Models
------
**Hufnagel-Valley (HV5/7)** -- the standard three-term empirical fit (a
high-altitude wind-driven term that produces the well-known tropopause
turbulence "bump" around 10 km + a broad, slowly-decaying background term +
a near-surface boundary-layer term added by Valley (1980)):

    Cn2(h) = 5.94e-53 * (w/27)^2 * h^10 * exp(-h/1000)
           + 2.7e-16 * exp(-h/1500)
           + A * exp(-h/100)

with ``h`` in metres, ``w`` the "pseudowind" (rms high-altitude wind speed,
m/s) and ``A`` the ground-level Cn2 (m^-2/3). Source: Hufnagel, R.E. (1974),
"Propagation through atmospheric turbulence," in *The Infrared Handbook*,
Ch. 6, USGPO; Valley, G.C. (1980), "Isoplanatic degradation of tilt
correction and short-term imaging systems," Appl. Opt. 19, 574-577. The
"5/7" designation is w=21 m/s, A=1.7e-14 m^-2/3, which is defined to give
r0 ~ 5 cm and theta0 ~ 7 urad at 0.5 um for a vertical path to space (see
the anchor test in tests/test_turbulence.py, which reproduces exactly this).
Constants cross-checked against Lawson, J.K. & Carrano, C.J. (2006), "Using
Historic Models of Cn2 to predict r0...," SPIE 6304 / LLNL UCRL-CONF-222540,
Eq. 4, and the AGI STK Components ``HufnagelValleyRefractiveIndexStructure
ParameterModel`` reference implementation (both quote the identical
constants: 5.94e-53 == 0.00594*(1e-5)^10, 2.7e-16, A=1.7e-14, w=21).

**SLC Day / SLC Night** ("Submarine Laser Communication" day/night models,
per the task spec; also seen as "Satellite Laser Communications" in some
secondary literature) -- parameter-free piecewise power laws fitted to data
from the AMOS site atop Mt. Haleakala, Maui. No adjustable constant: unlike
HV5/7 these are not rescaled by ``Scenario.turbulence_cn2_ground`` (there is
no "ground anchor" free parameter in the published fit to rescale).
Source: Good, R.E., Beland, R.R., Murphy, E.A., Brown, J.H., Dewan, E.M.
(1988), "Atmospheric Models of Optical Turbulence," SPIE 928, 165-186
(original fit); reproduced in Beland, R.R. (1993), "Propagation through
Atmospheric Optical Turbulence," in *The Infrared and Electro-Optical
Systems Handbook*, Vol. 2, SPIE Optical Engineering Press, Sec 2.3; Parenti,
R.R. & Sasiela, R.J. (1994), J. Opt. Soc. Am. A 11, 288-309; Roggemann,
M.C. & Welsh, B.M. (1996), *Imaging Through Turbulence*, CRC Press, p.62.
The SLC-Day table (every breakpoint and exponent) was verified directly
against Lawson & Carrano (2006), op. cit., Eq. 6 (their "version 1", the one
that retains the ground-layer term -- their Eq. 7 "version 2" is an
alternate fit some other secondary sources reproduce that goes to zero at
h=0, "unsuitable for ground-based observations" in their words, and is not
used here). SLC-Night is the standard paired nighttime fit from the same
Good et al. (1988) source, cross-checked against multiple independent
secondary reproductions.

**CONSTANT** -- uniform Cn2 = ``Scenario.turbulence_cn2_ground`` along the
whole path, for idealised/homogeneous-path sanity checks (e.g. reproducing
the textbook homogeneous-path Rytov variance analytically).
"""

from __future__ import annotations

import numpy as np

from ..types import TurbulenceModel

__all__ = [
    "HV57_PSEUDOWIND_M_S",
    "HV57_GROUND_CN2_M23",
    "BUFTON_GROUND_WIND_M_S",
    "BUFTON_PEAK_WIND_M_S",
    "BUFTON_PEAK_ALTITUDE_M",
    "BUFTON_PEAK_WIDTH_M",
    "hufnagel_valley",
    "slc_day",
    "slc_night",
    "constant_cn2",
    "bufton_wind_speed",
    "cn2_profile",
]

#: HV5/7 defaults -- see module docstring for the "why 5/7" derivation.
HV57_PSEUDOWIND_M_S = 21.0
HV57_GROUND_CN2_M23 = 1.7e-14

#: Bufton (1973) wind model, tied to HV5/7's pseudowind (see
#: :func:`bufton_wind_speed`): ground wind 5 m/s plus a 30 m/s jet-stream-like
#: bump centred at 9.4 km with a 4.8 km width. Source: Bufton, J.L. (1973),
#: "Comparison of vertical profile turbulence structure with stellar
#: observations," Appl. Opt. 12, 1785-1793; the specific numbers here (5, 30,
#: 9400, 4800) are the standard parameterisation reproduced e.g. in Hardy,
#: J.W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford
#: University Press, Sec. 3.3.2 -- chosen because they are the profile whose
#: rms over 5-20 km altitude (Greenwood & Fried's "pseudowind" definition)
#: works out to ~21 m/s, i.e. this is *the* wind profile HV5/7's "w=21" is
#: implicitly built on top of.
BUFTON_GROUND_WIND_M_S = 5.0
BUFTON_PEAK_WIND_M_S = 30.0
BUFTON_PEAK_ALTITUDE_M = 9400.0
BUFTON_PEAK_WIDTH_M = 4800.0


def hufnagel_valley(
    height_agl_m,
    *,
    pseudowind_m_s: float = HV57_PSEUDOWIND_M_S,
    ground_cn2_m23: float = HV57_GROUND_CN2_M23,
) -> np.ndarray:
    """Hufnagel-Valley Cn2(h), m^-2/3. ``height_agl_m`` may be scalar or array.

    Negative heights (which should not occur but could arise from floating
    point noise at the surface) are clamped to 0.
    """
    h = np.maximum(np.asarray(height_agl_m, dtype=np.float64), 0.0)
    term_wind = 5.94e-53 * (pseudowind_m_s / 27.0) ** 2 * h**10 * np.exp(-h / 1000.0)
    term_tropopause = 2.7e-16 * np.exp(-h / 1500.0)
    term_ground = float(ground_cn2_m23) * np.exp(-h / 100.0)
    return term_wind + term_tropopause + term_ground


def _piecewise_power_law(
    h: np.ndarray, breakpoints_m: tuple[float, ...], coeffs, exponents
) -> np.ndarray:
    """Evaluate a Cn2(h) = coeff * h^exponent piecewise model.

    ``breakpoints_m`` has one more entry than ``coeffs``/``exponents``: the
    i-th segment covers ``[breakpoints_m[i], breakpoints_m[i+1])``. Above the
    last breakpoint Cn2 is treated as negligible (0) -- both SLC models are
    only published out to 20 km, well above where either model or HV5/7
    contributes meaningfully to any of these path integrals.
    """
    h_safe = np.maximum(h, 1.0e-6)  # avoids 0**negative warnings; never selected at h~0
    conditions = [
        (h >= breakpoints_m[i]) & (h < breakpoints_m[i + 1]) for i in range(len(coeffs))
    ]
    choices = [c * h_safe**e for c, e in zip(coeffs, exponents)]
    return np.select(conditions, choices, default=0.0)


#: SLC-Day, Good/Beland/Murphy/Brown/Dewan (1988); see module docstring.
_SLC_DAY_BREAKPOINTS_M = (0.0, 18.5, 240.0, 880.0, 7200.0, 20000.0)
_SLC_DAY_COEFFS = (1.70e-14, 3.13e-13, 1.30e-15, 8.87e-7, 2.00e-16)
_SLC_DAY_EXPONENTS = (0.0, -1.05, 0.0, -3.0, -0.5)

#: SLC-Night, same source, paired nighttime fit.
_SLC_NIGHT_BREAKPOINTS_M = (0.0, 18.5, 110.0, 1500.0, 7200.0, 20000.0)
_SLC_NIGHT_COEFFS = (8.4e-15, 2.87e-12, 2.5e-16, 8.87e-7, 2.00e-16)
_SLC_NIGHT_EXPONENTS = (0.0, -2.0, 0.0, -3.0, -0.5)


def slc_day(height_agl_m) -> np.ndarray:
    """SLC-Day Cn2(h), m^-2/3 (piecewise power law; see module docstring)."""
    h = np.maximum(np.asarray(height_agl_m, dtype=np.float64), 0.0)
    return _piecewise_power_law(h, _SLC_DAY_BREAKPOINTS_M, _SLC_DAY_COEFFS, _SLC_DAY_EXPONENTS)


def slc_night(height_agl_m) -> np.ndarray:
    """SLC-Night Cn2(h), m^-2/3 (piecewise power law; see module docstring)."""
    h = np.maximum(np.asarray(height_agl_m, dtype=np.float64), 0.0)
    return _piecewise_power_law(
        h, _SLC_NIGHT_BREAKPOINTS_M, _SLC_NIGHT_COEFFS, _SLC_NIGHT_EXPONENTS
    )


def constant_cn2(height_agl_m, ground_cn2_m23: float) -> np.ndarray:
    """Uniform Cn2 = ``ground_cn2_m23`` at every height, m^-2/3."""
    h = np.asarray(height_agl_m, dtype=np.float64)
    return np.full_like(h, float(ground_cn2_m23))


def bufton_wind_speed(height_agl_m, *, ground_wind_m_s: float = BUFTON_GROUND_WIND_M_S) -> np.ndarray:
    """Bufton (1973) wind speed profile, m/s. See module docstring."""
    h = np.asarray(height_agl_m, dtype=np.float64)
    return ground_wind_m_s + BUFTON_PEAK_WIND_M_S * np.exp(
        -(((h - BUFTON_PEAK_ALTITUDE_M) / BUFTON_PEAK_WIDTH_M) ** 2)
    )


def cn2_profile(
    model: TurbulenceModel, height_agl_m, *, ground_cn2_m23: float | None = None
) -> np.ndarray:
    """Dispatch to the Cn2(h) model named by ``model``, m^-2/3.

    ``ground_cn2_m23`` is ``Scenario.turbulence_cn2_ground``: required for
    CONSTANT, and -- "where it makes sense" per CONTRACT.md's spec for this
    module -- used as the HV5/7 ground-anchor term ``A`` (replacing the
    default 1.7e-14) when supplied, since HV5/7 is explicitly parameterised
    by a free ground constant. It does *not* apply to SLC Day/Night, which
    are parameter-free fits with no ground-constant knob to rescale (see
    module docstring); a caller supplying it with an SLC model has no effect,
    by design.
    """
    h = np.asarray(height_agl_m, dtype=np.float64)
    if model == TurbulenceModel.HV57:
        kwargs = {} if ground_cn2_m23 is None else {"ground_cn2_m23": float(ground_cn2_m23)}
        return hufnagel_valley(h, **kwargs)
    if model == TurbulenceModel.SLC_DAY:
        return slc_day(h)
    if model == TurbulenceModel.SLC_NIGHT:
        return slc_night(h)
    if model == TurbulenceModel.CONSTANT:
        if ground_cn2_m23 is None:
            raise ValueError("TurbulenceModel.CONSTANT requires Scenario.turbulence_cn2_ground")
        return constant_cn2(h, ground_cn2_m23)
    if model == TurbulenceModel.NONE:
        return np.zeros_like(h)
    raise ValueError(f"unknown turbulence model: {model!r}")
