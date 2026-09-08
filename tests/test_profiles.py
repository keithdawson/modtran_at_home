"""Tests for mwir_atmos.atmosphere.profiles and .humidity.

Physically meaningful checks, not just shape checks:

* Hydrostatic consistency of every one of the six standard profiles, via the
  hypsometric relation (a per-layer check, tolerant enough for the AFGL
  table's own rounding -- see the module docstring in profiles.py for the
  constant-g0-vs-variable-g(z) investigation that set these tolerances).
* Precipitable water columns against the numeric ranges CONTRACT.md calls
  out (midlatitude summer ~2-3 cm, subarctic winter ~0.4 cm).
* Known surface checkpoints (tropical ~300 K & very moist, subarctic winter
  ~257 K & very dry, US Standard ~288 K/1013 mbar).
* apply_surface_conditions: overrides land exactly at the surface, taper to
  the unmodified profile aloft, keep pressure hydrostatically consistent,
  and leave pressure bit-for-bit alone when nothing is actually overridden.
* saturation_vapor_pressure_mbar against known steam-table-ish reference
  points.
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.atmosphere.humidity import (
    apply_surface_conditions,
    hydrostatic_pressure_mbar,
    relative_humidity_pct,
    saturation_vapor_pressure_mbar,
)
from mwir_atmos.atmosphere.profiles import available_models, get_profile
from mwir_atmos.constants import GAS_CONSTANT_R, G0_M_S2, M_AIR_G_MOL, M_H2O_G_MOL, TOA_ALTITUDE_KM
from mwir_atmos.types import ModelAtmosphere, Species

ALL_MODELS = [
    ModelAtmosphere.TROPICAL,
    ModelAtmosphere.MIDLAT_SUMMER,
    ModelAtmosphere.MIDLAT_WINTER,
    ModelAtmosphere.SUBARCTIC_SUMMER,
    ModelAtmosphere.SUBARCTIC_WINTER,
    ModelAtmosphere.US_STANDARD,
]


def _precipitable_water_cm(pressure_mbar: np.ndarray, h2o_vmr_ppmv: np.ndarray) -> float:
    """PW (cm) = (1/g0) * integral(q dP), q = specific humidity (kg/kg)."""
    p_pa = np.asarray(pressure_mbar, dtype=np.float64) * 100.0
    vmr = np.asarray(h2o_vmr_ppmv, dtype=np.float64) * 1.0e-6
    mass_ratio = M_H2O_G_MOL / M_AIR_G_MOL
    q = vmr * mass_ratio / (1.0 + vmr * (mass_ratio - 1.0))
    order = np.argsort(p_pa)
    pw_mm = np.trapezoid(q[order], p_pa[order]) / G0_M_S2
    return pw_mm / 10.0


def _max_hydrostatic_error(altitude_km, pressure_mbar, temperature_k, *, z_max=None) -> float:
    """Max relative error between the tabulated dlnP per layer and the
    hypsometric prediction from mean layer temperature (constant g0)."""
    z = np.asarray(altitude_km, dtype=np.float64)
    p = np.asarray(pressure_mbar, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    max_err = 0.0
    for i in range(len(z) - 1):
        if z_max is not None and z[i + 1] > z_max:
            continue
        dz_m = (z[i + 1] - z[i]) * 1000.0
        t_mean = 0.5 * (t[i] + t[i + 1])
        dlnp_pred = -G0_M_S2 * (M_AIR_G_MOL * 1.0e-3) * dz_m / (GAS_CONSTANT_R * t_mean)
        dlnp_actual = np.log(p[i + 1] / p[i])
        err = abs((dlnp_pred - dlnp_actual) / dlnp_actual)
        max_err = max(max_err, err)
    return max_err


# =========================================================================
# available_models / get_profile basics
# =========================================================================


def test_available_models_has_all_six():
    models = available_models()
    assert set(models) == set(ALL_MODELS)
    assert len(models) == 6


@pytest.mark.parametrize("model", ALL_MODELS)
def test_profile_shape_and_monotonicity(model):
    prof = get_profile(model)
    assert prof.n_levels == 33  # 0-25 @ 1km (26) + 30/35/40/45/50/70/100 (7)
    assert prof.altitude_km[0] == 0.0
    assert prof.altitude_km[-1] == TOA_ALTITUDE_KM
    assert np.all(np.diff(prof.altitude_km) > 0)
    assert np.all(np.diff(prof.pressure_mbar) < 0)  # pressure strictly decreasing
    assert np.all(prof.temperature_k > 150.0) and np.all(prof.temperature_k < 320.0)
    assert np.all(prof.vmr_ppmv[Species.H2O] > 0.0)
    assert np.all(prof.vmr_ppmv[Species.O3] > 0.0)
    # CO2/N2O/CO/CH4 come from constants.BACKGROUND_VMR_PPMV, constant w/ altitude.
    assert np.all(prof.vmr_ppmv[Species.CO2] == prof.vmr_ppmv[Species.CO2][0])
    assert prof.relative_humidity_pct is not None
    assert np.all(prof.relative_humidity_pct >= 0.0) and np.all(prof.relative_humidity_pct <= 100.0)


def test_tropical_surface_hot_and_moist():
    prof = get_profile(ModelAtmosphere.TROPICAL)
    assert 295.0 < prof.temperature_k[0] < 305.0  # CONTRACT.md: "~300 K"
    assert prof.vmr_ppmv[Species.H2O][0] > 20000.0  # "very moist"


def test_subarctic_winter_surface_cold_and_dry():
    prof = get_profile(ModelAtmosphere.SUBARCTIC_WINTER)
    assert 252.0 < prof.temperature_k[0] < 262.0  # CONTRACT.md: "~257 K"
    assert prof.vmr_ppmv[Species.H2O][0] < 2000.0  # "very dry"


def test_us_standard_matches_1976_sea_level():
    prof = get_profile(ModelAtmosphere.US_STANDARD)
    assert prof.temperature_k[0] == pytest.approx(288.2, abs=0.1)
    assert prof.pressure_mbar[0] == pytest.approx(1013.0, abs=1.0)


# =========================================================================
# Hydrostatic consistency
# =========================================================================


@pytest.mark.parametrize("model", ALL_MODELS)
def test_hydrostatic_consistency_0_25km(model):
    """Tight check over the 1 km-spaced troposphere/lower stratosphere."""
    prof = get_profile(model)
    err = _max_hydrostatic_error(prof.altitude_km, prof.pressure_mbar, prof.temperature_k, z_max=25.0)
    assert err < 0.05, f"{model}: hydrostatic error {err:.3%} over 0-25km exceeds 5%"


@pytest.mark.parametrize("model", ALL_MODELS)
def test_hydrostatic_consistency_full_column(model):
    """Looser check including the coarse 70/100 km layers (big single-segment
    hypsometric steps are inherently less accurate against a 2-point mean-T
    approximation), but still tight enough to catch a real unit/constant bug
    (a factor-of-N error would blow this out to >>15%)."""
    prof = get_profile(model)
    err = _max_hydrostatic_error(prof.altitude_km, prof.pressure_mbar, prof.temperature_k)
    assert err < 0.15, f"{model}: hydrostatic error {err:.3%} over full column exceeds 15%"


# =========================================================================
# Precipitable water
# =========================================================================


def test_precipitable_water_midlat_summer():
    prof = get_profile(ModelAtmosphere.MIDLAT_SUMMER)
    pw = _precipitable_water_cm(prof.pressure_mbar, prof.vmr_ppmv[Species.H2O])
    assert 2.0 < pw < 3.0, f"midlat summer PW={pw:.2f}cm, expected 2-3cm"


def test_precipitable_water_subarctic_winter():
    prof = get_profile(ModelAtmosphere.SUBARCTIC_WINTER)
    pw = _precipitable_water_cm(prof.pressure_mbar, prof.vmr_ppmv[Species.H2O])
    assert 0.2 < pw < 0.6, f"subarctic winter PW={pw:.2f}cm, expected ~0.4cm"


def test_precipitable_water_ordering_across_models():
    """Wetter climates should have more column water than drier ones."""
    pw = {}
    for model in ALL_MODELS:
        prof = get_profile(model)
        pw[model] = _precipitable_water_cm(prof.pressure_mbar, prof.vmr_ppmv[Species.H2O])
    assert pw[ModelAtmosphere.TROPICAL] > pw[ModelAtmosphere.MIDLAT_SUMMER]
    assert pw[ModelAtmosphere.MIDLAT_SUMMER] > pw[ModelAtmosphere.MIDLAT_WINTER]
    assert pw[ModelAtmosphere.SUBARCTIC_SUMMER] > pw[ModelAtmosphere.SUBARCTIC_WINTER]
    assert pw[ModelAtmosphere.SUBARCTIC_WINTER] == min(pw.values())


# =========================================================================
# n_levels resampling
# =========================================================================


def test_get_profile_resampling():
    base = get_profile(ModelAtmosphere.US_STANDARD)
    fine = get_profile(ModelAtmosphere.US_STANDARD, n_levels=200)
    assert fine.n_levels == 200
    assert fine.altitude_km[0] == 0.0
    assert fine.altitude_km[-1] == TOA_ALTITUDE_KM
    # Resampled onto a much finer grid, T/P should closely track the original
    # at shared altitudes (interpolation, not a different model).
    t_check = np.interp(base.altitude_km, fine.altitude_km, fine.temperature_k)
    assert np.max(np.abs(t_check - base.temperature_k)) < 1.0
    with pytest.raises(ValueError):
        get_profile(ModelAtmosphere.US_STANDARD, n_levels=1)


# =========================================================================
# saturation vapor pressure / relative humidity
# =========================================================================


def test_saturation_vapor_pressure_known_points():
    # 0 C liquid-water triple/reference point, Buck (1996): 6.1121 mbar exactly.
    assert saturation_vapor_pressure_mbar(273.15) == pytest.approx(6.1121, abs=0.001)
    # -40 C over ice, well-known steam-table value ~0.1283 mbar.
    assert saturation_vapor_pressure_mbar(233.15) == pytest.approx(0.1283, rel=0.02)
    # Monotonically increasing with temperature.
    t = np.array([220.0, 250.0, 273.15, 290.0, 310.0])
    es = saturation_vapor_pressure_mbar(t)
    assert np.all(np.diff(es) > 0)


def test_relative_humidity_round_trip():
    # By construction e = vmr*p; RH should recover the same ratio vs es.
    t_k, p_mbar = 288.0, 1013.0
    es = saturation_vapor_pressure_mbar(t_k)
    for target_rh in (10.0, 50.0, 90.0):
        vmr_ppmv = target_rh / 100.0 * es / p_mbar * 1.0e6
        rh = relative_humidity_pct(t_k, p_mbar, vmr_ppmv)
        assert rh == pytest.approx(target_rh, abs=0.05)


def test_relative_humidity_clipped_to_100():
    rh = relative_humidity_pct(273.15, 1013.0, 1.0e7)  # absurdly high VMR
    assert rh == 100.0


# =========================================================================
# hydrostatic_pressure_mbar helper
# =========================================================================


def test_hydrostatic_pressure_mbar_isothermal_scale_height():
    """For isothermal T, dP/dz = -P/H with H = R T / (g0 M) -- check against
    the closed-form exponential."""
    t_iso = 250.0
    z = np.linspace(0.0, 20.0, 401)
    p = hydrostatic_pressure_mbar(z, np.full_like(z, t_iso), 1000.0)
    h_km = (GAS_CONSTANT_R * t_iso) / (G0_M_S2 * (M_AIR_G_MOL * 1.0e-3)) / 1000.0
    expected = 1000.0 * np.exp(-z / h_km)
    assert np.allclose(p, expected, rtol=1.0e-6)


# =========================================================================
# apply_surface_conditions
# =========================================================================


def test_apply_surface_conditions_hits_target_exactly():
    base = get_profile(ModelAtmosphere.MIDLAT_SUMMER)
    mod = apply_surface_conditions(base, ground_temp_k=305.0, ground_rh_pct=15.0)
    assert mod.temperature_k[0] == pytest.approx(305.0, abs=1.0e-6)
    assert mod.relative_humidity_pct[0] == pytest.approx(15.0, abs=0.1)


def test_apply_surface_conditions_relaxes_aloft():
    base = get_profile(ModelAtmosphere.MIDLAT_WINTER)
    mod = apply_surface_conditions(base, ground_temp_k=base.temperature_k[0] + 20.0)
    # Above the blend height (2.5 km, per humidity.py), T must be unchanged.
    idx_aloft = np.searchsorted(base.altitude_km, 4.0)
    assert np.allclose(mod.temperature_k[idx_aloft:], base.temperature_k[idx_aloft:])
    # At the surface, the full 20K offset must apply.
    assert mod.temperature_k[0] - base.temperature_k[0] == pytest.approx(20.0, abs=1.0e-6)
    # Partway through the blend, the offset must be strictly between 0 and 20.
    idx_mid = np.searchsorted(base.altitude_km, 1.0)
    delta_mid = mod.temperature_k[idx_mid] - base.temperature_k[idx_mid]
    assert 0.0 < delta_mid < 20.0


def test_apply_surface_conditions_stays_hydrostatic():
    base = get_profile(ModelAtmosphere.SUBARCTIC_SUMMER)
    mod = apply_surface_conditions(base, ground_temp_k=base.temperature_k[0] - 15.0, ground_rh_pct=60.0)
    err = _max_hydrostatic_error(mod.altitude_km, mod.pressure_mbar, mod.temperature_k, z_max=25.0)
    assert err < 0.05


def test_apply_surface_conditions_noop_preserves_pressure_exactly():
    base = get_profile(ModelAtmosphere.US_STANDARD)
    mod = apply_surface_conditions(base)  # no overrides at all
    assert np.array_equal(mod.pressure_mbar, base.pressure_mbar)
    assert np.array_equal(mod.temperature_k, base.temperature_k)


def test_apply_surface_conditions_co2_override_is_flat():
    base = get_profile(ModelAtmosphere.US_STANDARD)
    mod = apply_surface_conditions(base, co2_ppmv=500.0)
    assert np.all(mod.vmr_ppmv[Species.CO2] == 500.0)
    # Nothing else about the profile should have moved.
    assert np.array_equal(mod.pressure_mbar, base.pressure_mbar)
    assert np.array_equal(mod.temperature_k, base.temperature_k)


def test_apply_surface_conditions_reanchor_altitude():
    base = get_profile(ModelAtmosphere.US_STANDARD)
    mod = apply_surface_conditions(base, surface_altitude_km=2.0)
    assert mod.altitude_km[0] == pytest.approx(2.0)
    assert mod.surface_altitude_km == pytest.approx(2.0)
    # Surface pressure should match a log-linear interpolate of the original
    # table between the 2km bracketing levels (which land exactly on 2.0 km
    # in the canonical table, so this should be an exact hit).
    assert mod.pressure_mbar[0] == pytest.approx(base.pressure_mbar[2], rel=1.0e-6)
    assert mod.n_levels == base.n_levels - 2  # levels 0,1 dropped, level 2 becomes new base


def test_apply_surface_conditions_rejects_downward_extrapolation():
    base = get_profile(ModelAtmosphere.US_STANDARD)
    with pytest.raises(ValueError):
        apply_surface_conditions(base, surface_altitude_km=-1.0)
