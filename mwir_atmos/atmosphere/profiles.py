"""The six standard LOWTRAN/MODTRAN model atmospheres.

Data source
-----------
AFGL-TR-86-0110, "AFGL Atmospheric Constituent Profiles (0-120km)"
(Anderson, G.P., Clough, S.A., Kneizys, F.X., Chetwynd, J.H., Shettle, E.P.,
Air Force Geophysics Laboratory, 1986). This is the report that defines the
six reference atmospheres used throughout LOWTRAN/MODTRAN (its Tables 1a-1f).

Numeric values below were taken from the machine-readable digitisation of
that report published at https://github.com/rayference/afgl1986
(``tables/1a.csv`` .. ``tables/1f.csv``; MIT licensed transcription of the
AFGL tables, columns z/p/t/n/H2O/O3/N2O/CO/CH4, values quoted to the same
significant figures as the original report). Every level tabulated here
(0-25 km at 1 km spacing, then 30/35/40/45/50/70/100 km) lands exactly on a
row of the source table, so no interpolation was needed to build the
canonical arrays below -- they are direct transcriptions.

Per CONTRACT.md, CO2/N2O/CO/CH4 are NOT taken from the AFGL tables (which
reflect ~1986 background levels); they are filled from
``constants.BACKGROUND_VMR_PPMV`` (current-era well-mixed values), broadcast
uniformly with altitude. O2 is likewise filled from the same table for
completeness (not in ``REQUIRED_SPECIES`` but present in ``Species`` and in
``testing.make_synthetic_profile`` for parity).

Sanity checks against the report / well-known values (see tests/test_profiles.py):
  * Tropical surface: T=299.7 K, H2O=25900 ppmv (~300 K, very moist) -- matches
    CONTRACT.md's stated expectation.
  * Subarctic winter surface: T=257.2 K, H2O=1410 ppmv (~257 K, very dry) --
    matches CONTRACT.md's stated expectation.
  * US Standard surface: T=288.2 K, P=1013 mbar -- matches US Standard
    Atmosphere 1976 sea-level values (288.15 K, 1013.25 mbar) to the report's
    rounding.
  * Precipitable water computed from the tabulated P/H2O columns: midlatitude
    summer ~2.95 cm, subarctic winter ~0.42 cm -- both inside the ranges
    CONTRACT.md calls out (2-3 cm and ~0.4 cm respectively).

Desert and sea (maritime) conditions are deliberately NOT separate vertical
profiles here -- per CONTRACT.md they are handled by ``SurfaceType`` and
``AerosolModel`` instead, layered on top of one of these six atmospheres.
"""

from __future__ import annotations

import numpy as np

from ..constants import BACKGROUND_VMR_PPMV, TOA_ALTITUDE_KM
from ..types import AtmosphereProfile, ModelAtmosphere, Species
from .humidity import relative_humidity_pct

__all__ = ["get_profile", "available_models"]

# =========================================================================
# Canonical levels: 0-25 km @ 1 km, then 30/35/40/45/50/70/100 km (33 levels).
# =========================================================================

_ALT_KM: tuple[float, ...] = tuple(float(z) for z in range(0, 26)) + (
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    70.0,
    100.0,
)

# =========================================================================
# AFGL-TR-86-0110 Tables 1a-1f, subsampled to the levels above.
# Columns: pressure (mbar), temperature (K), H2O (ppmv), O3 (ppmv).
# =========================================================================

_TROPICAL_P_MBAR = (
    1013, 904, 805, 715, 633, 559,
    492, 432, 378, 329, 286, 247,
    213, 182, 156, 132, 111, 93.7,
    78.9, 66.6, 56.5, 48, 40.9, 35,
    30, 25.7, 12.2, 6, 3.05, 1.59,
    0.854, 0.058, 0.000289,
)
_TROPICAL_T_K = (
    299.7, 293.7, 287.7, 283.7, 277, 270.3,
    263.6, 257, 250.3, 243.6, 237, 230.1,
    223.6, 217, 210.3, 203.7, 197, 194.8,
    198.8, 202.7, 206.7, 210.7, 214.6, 217,
    219.2, 221.4, 232.3, 243.1, 254, 264.8,
    270.2, 218.9, 190.7,
)
_TROPICAL_H2O_PPMV = (
    2.59e4, 1.95e4, 1.53e4, 8600, 4440, 3350,
    2100, 1290, 764, 410, 191, 73.1,
    29.1, 9.9, 6.22, 4, 3, 2.9,
    2.75, 2.6, 2.6, 2.65, 2.8, 2.9,
    3.2, 3.25, 4, 4.6, 5.2, 5.7,
    6, 4.5, 0.4,
)
_TROPICAL_O3_PPMV = (
    0.0287, 0.0315, 0.0334, 0.035, 0.0356, 0.0377,
    0.0399, 0.0422, 0.0447, 0.05, 0.0559, 0.0661,
    0.0781, 0.0929, 0.105, 0.126, 0.144, 0.25,
    0.5, 0.95, 1.4, 1.8, 2.4, 3.4,
    4.3, 5.4, 9.3, 9.7, 7.5, 4.5,
    2.8, 0.3, 0.4,
)

_MIDLAT_SUMMER_P_MBAR = (
    1013, 902, 802, 710, 628, 554,
    487, 426, 372, 324, 281, 243,
    209, 179, 153, 130, 111, 95,
    81.2, 69.5, 59.5, 51, 43.7, 37.6,
    32.2, 27.7, 13.2, 6.52, 3.33, 1.76,
    0.951, 0.067, 0.000258,
)
_MIDLAT_SUMMER_T_K = (
    294.2, 289.7, 285.2, 279.2, 273.2, 267.2,
    261.2, 254.7, 248.2, 241.7, 235.3, 228.8,
    222.3, 215.8, 215.7, 215.7, 215.7, 215.7,
    216.8, 217.9, 219.2, 220.4, 221.6, 222.8,
    223.9, 225.1, 233.7, 245.2, 257.5, 269.9,
    275.7, 218.1, 190.5,
)
_MIDLAT_SUMMER_H2O_PPMV = (
    1.88e4, 1.38e4, 9680, 5980, 3810, 2230,
    1510, 1020, 646, 413, 247, 95.6,
    29.4, 8, 5, 3.4, 3.3, 3.2,
    3.15, 3.2, 3.3, 3.45, 3.6, 3.85,
    4, 4.2, 4.7, 4.95, 5.1, 5.45,
    5.5, 3.7, 0.4,
)
_MIDLAT_SUMMER_O3_PPMV = (
    0.0302, 0.0334, 0.0369, 0.0422, 0.0482, 0.0551,
    0.0641, 0.0776, 0.0913, 0.111, 0.13, 0.179,
    0.223, 0.3, 0.44, 0.5, 0.6, 0.7,
    1, 1.5, 2, 2.4, 2.9, 3.4,
    4, 4.8, 7, 8.9, 7.55, 4.5,
    2.8, 0.4, 0.4,
)

_MIDLAT_WINTER_P_MBAR = (
    1018, 897.3, 789.7, 693.8, 608.1, 531.3,
    462.7, 401.6, 347.3, 299.3, 256.8, 219.9,
    188.2, 161.1, 137.8, 117.8, 100.7, 86.1,
    73.6, 62.8, 53.7, 45.8, 39.1, 33.4,
    28.6, 24.4, 11.1, 5.18, 2.53, 1.29,
    0.683, 0.047, 0.0004074,
)
_MIDLAT_WINTER_T_K = (
    272.2, 268.7, 265.2, 261.7, 255.7, 249.7,
    243.7, 237.7, 231.7, 225.7, 219.7, 219.2,
    218.7, 218.2, 217.7, 217.2, 216.7, 216.2,
    215.7, 215.2, 215.2, 215.2, 215.2, 215.2,
    215.2, 215.2, 217.4, 227.9, 243.2, 258.5,
    265.7, 230.7, 218.6,
)
_MIDLAT_WINTER_H2O_PPMV = (
    4320, 3450, 2790, 2090, 1280, 824,
    510, 232, 108, 55.7, 29.6, 10,
    6, 5, 4.8, 4.7, 4.6, 4.5,
    4.5, 4.5, 4.5, 4.5, 4.53, 4.55,
    4.6, 4.65, 4.75, 4.85, 4.95, 5,
    4.95, 3.3, 0.4,
)
_MIDLAT_WINTER_O3_PPMV = (
    0.0278, 0.028, 0.0285, 0.032, 0.0357, 0.0472,
    0.0584, 0.0789, 0.104, 0.157, 0.237, 0.362,
    0.523, 0.704, 0.8, 0.9, 1.1, 1.4,
    1.8, 2.3, 2.9, 3.5, 3.9, 4.3,
    4.7, 5.1, 6.1, 7.1, 6.9, 4.6,
    2.75, 0.32, 0.4,
)

_SUBARCTIC_SUMMER_P_MBAR = (
    1010, 896, 792.9, 700, 616, 541,
    474, 413, 359, 310.8, 267.7, 230,
    197.7, 170, 146, 126, 108, 92.8,
    79.8, 68.6, 59, 50.7, 43.6, 37.5,
    32.28, 27.8, 13.4, 6.61, 3.4, 1.82,
    0.987, 0.071, 0.000248,
)
_SUBARCTIC_SUMMER_T_K = (
    287.2, 281.7, 276.3, 270.9, 265.5, 260.1,
    253.1, 246.1, 239.2, 232.2, 225.2, 225.2,
    225.2, 225.2, 225.2, 225.2, 225.2, 225.2,
    225.2, 225.2, 225.2, 225.2, 225.2, 225.2,
    226.6, 228.1, 235.1, 247.2, 262.1, 273.6,
    277.2, 216.6, 190.4,
)
_SUBARCTIC_SUMMER_H2O_PPMV = (
    1.19e4, 8700, 6750, 4820, 3380, 2220,
    1330, 797, 400, 130, 42.4, 13.3,
    6, 4.45, 4, 4, 4, 4.05,
    4.3, 4.5, 4.6, 4.7, 4.8, 4.83,
    4.85, 4.9, 5, 5, 5, 5,
    4.95, 3.3, 0.4,
)
_SUBARCTIC_SUMMER_O3_PPMV = (
    0.0241, 0.0294, 0.0338, 0.0389, 0.0448, 0.0533,
    0.0656, 0.0774, 0.0911, 0.142, 0.189, 0.305,
    0.41, 0.5, 0.6, 0.7, 0.85, 1,
    1.3, 1.7, 2.1, 2.7, 3.3, 3.7,
    4.2, 4.5, 5.7, 7.7, 7, 4.2,
    2.5, 0.4, 0.4,
)

_SUBARCTIC_WINTER_P_MBAR = (
    1013, 887.8, 777.5, 679.8, 593.2, 515.8,
    446.7, 385.3, 330.8, 282.9, 241.8, 206.7,
    176.6, 151, 129.1, 110.3, 94.31, 80.58,
    68.82, 58.75, 50.14, 42.77, 36.47, 31.09,
    26.49, 22.56, 10.2, 4.701, 2.243, 1.113,
    0.5719, 0.04, 0.000423,
)
_SUBARCTIC_WINTER_T_K = (
    257.2, 259.1, 255.9, 252.7, 247.7, 240.9,
    234.1, 227.3, 220.9, 217.2, 217.2, 217.2,
    217.2, 217.2, 217.2, 217.2, 216.6, 216,
    215.4, 214.8, 214.2, 213.6, 213, 212.4,
    211.8, 211.2, 216, 222.3, 234.7, 247,
    259.3, 245.4, 218.5,
)
_SUBARCTIC_WINTER_H2O_PPMV = (
    1410, 1620, 1430, 1170, 790, 431,
    237, 147, 33.8, 29.8, 20, 10,
    6, 4.45, 4.5, 4.55, 4.6, 4.65,
    4.7, 4.75, 4.8, 4.85, 4.9, 4.95,
    5, 5, 5, 5, 5, 5,
    4.95, 3.3, 0.4,
)
_SUBARCTIC_WINTER_O3_PPMV = (
    0.018, 0.0207, 0.0234, 0.0277, 0.0325, 0.038,
    0.0445, 0.0725, 0.104, 0.21, 0.3, 0.35,
    0.4, 0.65, 0.9, 1.2, 1.5, 1.9,
    2.45, 3.1, 3.7, 4, 4.2, 4.5,
    4.6, 4.7, 5.4, 6.2, 5.9, 4.1,
    2.6, 0.5, 0.4,
)

_US_STANDARD_P_MBAR = (
    1013, 898.8, 795, 701.2, 616.6, 540.5,
    472.2, 411.1, 356.5, 308, 265, 227,
    194, 165.8, 141.7, 121.1, 103.5, 88.5,
    75.65, 64.67, 55.29, 47.29, 40.47, 34.67,
    29.72, 25.49, 11.97, 5.746, 2.871, 1.491,
    0.7978, 0.0522, 0.00032,
)
_US_STANDARD_T_K = (
    288.2, 281.7, 275.2, 268.7, 262.2, 255.7,
    249.2, 242.7, 236.2, 229.7, 223.3, 216.8,
    216.7, 216.7, 216.7, 216.7, 216.7, 216.7,
    216.7, 216.7, 216.7, 217.6, 218.6, 219.6,
    220.6, 221.6, 226.5, 236.5, 250.4, 264.2,
    270.7, 219.6, 195.1,
)
_US_STANDARD_H2O_PPMV = (
    7750, 6070, 4630, 3180, 2160, 1400,
    925, 572, 367, 158, 70, 36.1,
    19.1, 10.9, 5.93, 5, 3.95, 3.85,
    3.83, 3.85, 3.9, 3.98, 4.07, 4.2,
    4.3, 4.43, 4.73, 4.9, 5.03, 5.23,
    5.23, 3.5, 0.4,
)
_US_STANDARD_O3_PPMV = (
    0.0266, 0.0293, 0.0324, 0.0332, 0.0339, 0.0377,
    0.0411, 0.0501, 0.0597, 0.0917, 0.131, 0.215,
    0.31, 0.385, 0.503, 0.651, 0.87, 1.19,
    1.59, 2.03, 2.58, 3.03, 3.65, 4.17,
    4.63, 5.12, 6.55, 7.84, 7.3, 5.25,
    3.1, 0.3, 0.4,
)

_TABLES: dict[ModelAtmosphere, dict[str, tuple[float, ...]]] = {
    ModelAtmosphere.TROPICAL: dict(
        p_mbar=_TROPICAL_P_MBAR, t_k=_TROPICAL_T_K,
        h2o_ppmv=_TROPICAL_H2O_PPMV, o3_ppmv=_TROPICAL_O3_PPMV,
    ),
    ModelAtmosphere.MIDLAT_SUMMER: dict(
        p_mbar=_MIDLAT_SUMMER_P_MBAR, t_k=_MIDLAT_SUMMER_T_K,
        h2o_ppmv=_MIDLAT_SUMMER_H2O_PPMV, o3_ppmv=_MIDLAT_SUMMER_O3_PPMV,
    ),
    ModelAtmosphere.MIDLAT_WINTER: dict(
        p_mbar=_MIDLAT_WINTER_P_MBAR, t_k=_MIDLAT_WINTER_T_K,
        h2o_ppmv=_MIDLAT_WINTER_H2O_PPMV, o3_ppmv=_MIDLAT_WINTER_O3_PPMV,
    ),
    ModelAtmosphere.SUBARCTIC_SUMMER: dict(
        p_mbar=_SUBARCTIC_SUMMER_P_MBAR, t_k=_SUBARCTIC_SUMMER_T_K,
        h2o_ppmv=_SUBARCTIC_SUMMER_H2O_PPMV, o3_ppmv=_SUBARCTIC_SUMMER_O3_PPMV,
    ),
    ModelAtmosphere.SUBARCTIC_WINTER: dict(
        p_mbar=_SUBARCTIC_WINTER_P_MBAR, t_k=_SUBARCTIC_WINTER_T_K,
        h2o_ppmv=_SUBARCTIC_WINTER_H2O_PPMV, o3_ppmv=_SUBARCTIC_WINTER_O3_PPMV,
    ),
    ModelAtmosphere.US_STANDARD: dict(
        p_mbar=_US_STANDARD_P_MBAR, t_k=_US_STANDARD_T_K,
        h2o_ppmv=_US_STANDARD_H2O_PPMV, o3_ppmv=_US_STANDARD_O3_PPMV,
    ),
}

for _name, _tab in _TABLES.items():
    for _key, _vals in _tab.items():
        if len(_vals) != len(_ALT_KM):
            raise AssertionError(
                f"internal error: {_name} {_key} has {len(_vals)} values, "
                f"expected {len(_ALT_KM)}"
            )


def available_models() -> list[ModelAtmosphere]:
    """The six standard model atmospheres, in AFGL report table order."""
    return [
        ModelAtmosphere.TROPICAL,
        ModelAtmosphere.MIDLAT_SUMMER,
        ModelAtmosphere.MIDLAT_WINTER,
        ModelAtmosphere.SUBARCTIC_SUMMER,
        ModelAtmosphere.SUBARCTIC_WINTER,
        ModelAtmosphere.US_STANDARD,
    ]


def _log_interp(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate y > 0 log-linearly in x (used for pressure and VMR)."""
    return np.exp(np.interp(x_new, x, np.log(y)))


def get_profile(model: ModelAtmosphere, *, n_levels: int | None = None) -> AtmosphereProfile:
    """Build one of the six standard AFGL/LOWTRAN/MODTRAN model atmospheres.

    With ``n_levels=None`` (the default) this returns the canonical AFGL
    tabulation verbatim: 0-25 km at 1 km spacing then 30/35/40/45/50/70/100 km
    (33 levels), pressure/temperature/H2O/O3 transcribed directly from the
    source table (see module docstring for citation) -- no interpolation.

    With ``n_levels`` given, the profile is resampled onto ``n_levels`` levels
    evenly spaced from the surface (0 km) to :data:`constants.TOA_ALTITUDE_KM`.
    Temperature and volume mixing ratios are interpolated (temperature
    linearly, VMR log-linearly, since H2O/O3 vary over decades); pressure is
    then *log-linearly* interpolated from the same canonical curve rather than
    re-derived, because the canonical curve is itself already close to
    hydrostatic (see tests/test_profiles.py) and log-linear interpolation of a
    smooth, already-consistent P(z) preserves that to within interpolation
    error. (Contrast with :func:`mwir_atmos.atmosphere.humidity.
    apply_surface_conditions`, which *modifies* T and therefore must
    re-derive P by hydrostatic integration.)

    CO2/N2O/CO/CH4 are filled from ``constants.BACKGROUND_VMR_PPMV``, constant
    with altitude, per CONTRACT.md (the AFGL 1986 report's own values for
    these are ~40 years stale). O2 is filled the same way for completeness.

    ``profile.relative_humidity_pct`` is populated from the tabulated T/P/H2O.
    """
    if model not in _TABLES:
        raise ValueError(f"unknown model atmosphere: {model!r}")
    tab = _TABLES[model]
    alt = np.asarray(_ALT_KM, dtype=np.float64)
    p_mbar = np.asarray(tab["p_mbar"], dtype=np.float64)
    t_k = np.asarray(tab["t_k"], dtype=np.float64)
    h2o = np.asarray(tab["h2o_ppmv"], dtype=np.float64)
    o3 = np.asarray(tab["o3_ppmv"], dtype=np.float64)

    if n_levels is not None:
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")
        new_alt = np.linspace(alt[0], TOA_ALTITUDE_KM, int(n_levels))
        p_mbar = _log_interp(new_alt, alt, p_mbar)
        t_k = np.interp(new_alt, alt, t_k)
        h2o = _log_interp(new_alt, alt, h2o)
        o3 = _log_interp(new_alt, alt, o3)
        alt = new_alt

    vmr_ppmv: dict[Species, np.ndarray] = {
        Species.H2O: h2o,
        Species.CO2: np.full_like(alt, BACKGROUND_VMR_PPMV["CO2"]),
        Species.O3: o3,
        Species.N2O: np.full_like(alt, BACKGROUND_VMR_PPMV["N2O"]),
        Species.CO: np.full_like(alt, BACKGROUND_VMR_PPMV["CO"]),
        Species.CH4: np.full_like(alt, BACKGROUND_VMR_PPMV["CH4"]),
        Species.O2: np.full_like(alt, BACKGROUND_VMR_PPMV["O2"]),
    }

    rh = relative_humidity_pct(t_k, p_mbar, vmr_ppmv[Species.H2O])

    return AtmosphereProfile(
        altitude_km=alt,
        pressure_mbar=p_mbar,
        temperature_k=t_k,
        vmr_ppmv=vmr_ppmv,
        model=model,
        surface_altitude_km=float(alt[0]),
        relative_humidity_pct=rh,
    )
