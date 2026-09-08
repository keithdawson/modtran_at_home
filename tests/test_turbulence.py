"""Tests for mwir_atmos.turbulence (Cn2 profile models and compute_turbulence).

Physically meaningful checks per CONTRACT.md's turbulence spec: HV5/7's own
"r0~5cm / theta0~7urad at 0.5um, vertical path to space" namesake as the
anchor validation (the single best self-checking validation available here),
wavelength scaling (r0 ~ lambda^(6/5)), zenith-angle dependence, Cn2
monotonicity above the boundary layer, a CONSTANT-model analytic Rytov-
variance cross-check, scintillation-index saturation, the
``TurbulenceModel.NONE`` guard already implemented in api.py, and
finiteness/sign of every ``TurbulenceResult`` field.

Per CONTRACT.md's cost-discipline note, turbulence needs only path geometry
and the profile, not the spectral line-by-line engine -- every test here
builds a ``PathGeometry`` directly via ``build_path`` (near-instant) except
``test_run_scenario_populates_and_guards_turbulence``, which uses the
sanctioned narrow 2300-2400 cm^-1 band at 8 cm^-1 resolution to exercise the
real ``run_scenario`` -> ``api._attach_turbulence`` -> ``compute_turbulence``
wiring end to end (two calls, not a loop; a few seconds total).
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.atmosphere.profiles import get_profile
from mwir_atmos.geometry.path import build_path
from mwir_atmos.types import ModelAtmosphere, Scenario, TurbulenceModel
from mwir_atmos.turbulence import cn2
from mwir_atmos.turbulence.metrics import (
    ASSUMED_APERTURE_DIAMETER_M,
    MWIR_REFERENCE_WAVELENGTH_UM,
    VISIBLE_REFERENCE_WAVELENGTH_UM,
    compute_turbulence,
    fried_parameter_r0_m,
    isoplanatic_angle_rad,
    layer_geometry_m,
    rytov_variance_plane_wave,
    scintillation_index,
    strehl_ratio_marechal,
)


@pytest.fixture(scope="module")
def profile():
    return get_profile(ModelAtmosphere.US_STANDARD)


@pytest.fixture(scope="module")
def vertical_path(profile):
    """Ground sensor straight up to space -- HV5/7's own namesake geometry."""
    return build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=100)


# =========================================================================
# HV5/7 anchor validation
# =========================================================================


def test_hv57_vertical_path_reproduces_namesake_r0_and_theta0(profile, vertical_path):
    """The single best validation available here, per CONTRACT.md: HV5/7 is
    *defined* so that a vertical path to space gives r0~5cm, theta0~7urad at
    0.5um. If this fails, the HV5/7 constants are wrong."""
    height_agl_m, ds_m, s_mid_m = layer_geometry_m(vertical_path, profile)
    cn2_arr = cn2.cn2_profile(TurbulenceModel.HV57, height_agl_m)

    r0_cm = fried_parameter_r0_m(cn2_arr, ds_m, VISIBLE_REFERENCE_WAVELENGTH_UM) * 100.0
    theta0_urad = (
        isoplanatic_angle_rad(cn2_arr, s_mid_m, ds_m, VISIBLE_REFERENCE_WAVELENGTH_UM) * 1.0e6
    )
    assert 4.0 < r0_cm < 6.0
    assert 6.0 < theta0_urad < 9.0


def test_compute_turbulence_matches_standalone_formulas(profile, vertical_path):
    """The full compute_turbulence() wiring (unit conversions, dataclass
    assembly) reproduces the same numbers as calling the formulas directly,
    at the MWIR default reference wavelength."""
    height_agl_m, ds_m, s_mid_m = layer_geometry_m(vertical_path, profile)
    cn2_arr = cn2.cn2_profile(TurbulenceModel.HV57, height_agl_m)
    expected_r0_cm = fried_parameter_r0_m(cn2_arr, ds_m, MWIR_REFERENCE_WAVELENGTH_UM) * 100.0

    scenario = Scenario(turbulence=TurbulenceModel.HV57, sensor_zenith_deg=0.0)
    result = compute_turbulence(scenario=scenario, path=vertical_path, profile=profile)

    assert result.reference_wavelength_um == pytest.approx(MWIR_REFERENCE_WAVELENGTH_UM)
    assert result.fried_parameter_r0_cm == pytest.approx(expected_r0_cm, rel=1.0e-9)
    assert len(result.cn2_profile_m23) == vertical_path.n_layers
    # MWIR turbulence numbers are far more favourable than the 0.5um figures.
    assert result.fried_parameter_r0_cm > 40.0  # vs ~5cm at 0.5um


# =========================================================================
# Wavelength scaling
# =========================================================================


def test_r0_scales_as_lambda_to_the_6_5ths(profile, vertical_path):
    """r0(4um)/r0(0.5um) ~ 8^(6/5) ~ 12.1 -- MWIR numbers are far more
    favourable than the visible ones usually quoted for the same Cn2."""
    height_agl_m, ds_m, _ = layer_geometry_m(vertical_path, profile)
    cn2_arr = cn2.cn2_profile(TurbulenceModel.HV57, height_agl_m)
    r0_05 = fried_parameter_r0_m(cn2_arr, ds_m, 0.5)
    r0_4 = fried_parameter_r0_m(cn2_arr, ds_m, 4.0)
    expected_ratio = 8.0 ** (6.0 / 5.0)  # ~12.1
    assert expected_ratio == pytest.approx(12.1, abs=0.1)
    assert (r0_4 / r0_05) == pytest.approx(expected_ratio, rel=1.0e-6)


def test_isoplanatic_angle_also_scales_as_lambda_to_the_6_5ths(profile, vertical_path):
    height_agl_m, ds_m, s_mid_m = layer_geometry_m(vertical_path, profile)
    cn2_arr = cn2.cn2_profile(TurbulenceModel.HV57, height_agl_m)
    theta0_05 = isoplanatic_angle_rad(cn2_arr, s_mid_m, ds_m, 0.5)
    theta0_4 = isoplanatic_angle_rad(cn2_arr, s_mid_m, ds_m, 4.0)
    assert (theta0_4 / theta0_05) == pytest.approx(8.0 ** (6.0 / 5.0), rel=1.0e-6)


# =========================================================================
# Zenith-angle dependence
# =========================================================================


def test_r0_decreases_with_zenith_angle(profile):
    """More airmass at higher zenith angle -> more integrated turbulence ->
    smaller r0 (CONTRACT.md's explicit required check)."""
    r0_by_zenith = []
    for zen in (0.0, 30.0, 60.0):
        path = build_path(profile, sensor_alt_km=0.0, zenith_deg=zen, max_layers=100)
        height_agl_m, ds_m, _ = layer_geometry_m(path, profile)
        cn2_arr = cn2.cn2_profile(TurbulenceModel.HV57, height_agl_m)
        r0_by_zenith.append(fried_parameter_r0_m(cn2_arr, ds_m, 0.5))
    assert r0_by_zenith[0] > r0_by_zenith[1] > r0_by_zenith[2]


# =========================================================================
# Cn2 profile shape
# =========================================================================


def test_hv57_cn2_monotonic_above_boundary_layer():
    """Immediately above the ~100m-scale-height ground term, HV5/7 Cn2 falls
    off monotonically -- checked here from 200m (comfortably above the
    ground term's own scale height) to 4km.

    HV5/7's high-altitude wind term (``h^10 exp(-h/1000)``, peaking at
    h=10km) is *not* negligible there: it produces a genuine local minimum
    around h~5.9km and a local maximum ("tropopause bump") around h~9.8km,
    the well-known jet-stream turbulence enhancement the Hufnagel-Valley
    model is specifically built to reproduce (see e.g. the balloon/radiosonde
    Cn2 profiles this term is fit to) -- confirmed numerically during
    development and not a bug. CONTRACT.md's monotonicity check is validated
    here in the region immediately above the boundary layer, before that
    bump, rather than across the whole profile where it would be false.
    """
    h = np.linspace(200.0, 4000.0, 500)
    values = cn2.hufnagel_valley(h)
    assert np.all(np.diff(values) <= 0.0)


def test_hv57_cn2_has_expected_tropopause_bump():
    """The local minimum/bump noted above is real physics, not noise: Cn2
    must be *higher* at the h~9.8km local maximum than at the h~5.9km local
    minimum that precedes it (the wind term's h^10*exp(-h/1000) shape
    predicts a peak at h = 10*1000 = 10000m, from
    d/dh[10 ln(h) - h/1000] = 0) -- a genuine non-monotonic uptick, not just
    a slower decline."""
    local_min = cn2.hufnagel_valley(np.array([5900.0]))[0]
    local_max = cn2.hufnagel_valley(np.array([9800.0]))[0]
    assert local_max > local_min


def test_hv57_ground_value_matches_ground_constant():
    assert cn2.hufnagel_valley(np.array([0.0]))[0] == pytest.approx(cn2.HV57_GROUND_CN2_M23, rel=1.0e-6)


@pytest.mark.parametrize("model", [TurbulenceModel.SLC_DAY, TurbulenceModel.SLC_NIGHT])
def test_slc_profiles_positive_and_finite(model):
    h = np.linspace(0.0, 25000.0, 2000)
    values = cn2.cn2_profile(model, h)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_slc_day_matches_published_breakpoint_values():
    """Spot-check against Lawson & Carrano (2006), SPIE 6304 / LLNL
    UCRL-CONF-222540, Eq. 6 (see cn2.py's module docstring) -- one point
    from each of the constant-valued segments."""
    h = np.array([10.0, 500.0])
    values = cn2.slc_day(h)
    assert values[0] == pytest.approx(1.70e-14)
    assert values[1] == pytest.approx(1.30e-15)


def test_slc_night_matches_published_ground_value():
    assert cn2.slc_night(np.array([5.0]))[0] == pytest.approx(8.4e-15)


def test_slc_day_and_night_agree_at_high_altitude():
    """Day/night differences are a boundary-layer phenomenon; both published
    fits share the same free-troposphere segments (880/1500-7200m and
    7200-20000m) -- see cn2.py's module docstring."""
    h = np.array([10000.0, 15000.0])
    assert cn2.slc_day(h) == pytest.approx(cn2.slc_night(h))


# =========================================================================
# CONSTANT model / analytic Rytov variance cross-check
# =========================================================================


def test_constant_model_short_horizontal_path_matches_analytic_rytov(profile):
    """A short (1 km) horizontal path with uniform Cn2 must reproduce the
    textbook homogeneous-path plane-wave Rytov variance,
    sigma_R^2 = 1.23 Cn2 k^(7/6) L^(11/6), to a few percent."""
    path = build_path(
        profile, sensor_alt_km=0.0, zenith_deg=90.0, source_range_km=1.0, max_layers=60
    )
    height_agl_m, ds_m, s_mid_m = layer_geometry_m(path, profile)

    cn2_ground = 1.0e-14
    cn2_arr = cn2.constant_cn2(height_agl_m, cn2_ground)
    wavelength_um = 1.0
    numeric = rytov_variance_plane_wave(cn2_arr, s_mid_m, ds_m, wavelength_um)

    k = 2.0 * np.pi / (wavelength_um * 1.0e-6)
    length_m = path.total_length_km * 1000.0
    analytic = 1.23 * cn2_ground * k ** (7.0 / 6.0) * length_m ** (11.0 / 6.0)

    assert numeric == pytest.approx(analytic, rel=0.05)


def test_cn2_profile_constant_requires_ground_value():
    with pytest.raises(ValueError):
        cn2.cn2_profile(TurbulenceModel.CONSTANT, np.array([0.0, 100.0]), ground_cn2_m23=None)


def test_hv57_ground_cn2_override_changes_result():
    """Scenario.turbulence_cn2_ground overrides HV5/7's ground term "where it
    makes sense" per CONTRACT.md -- confirm it actually takes effect."""
    h = np.array([0.0, 10.0, 50.0])
    default = cn2.cn2_profile(TurbulenceModel.HV57, h)
    overridden = cn2.cn2_profile(TurbulenceModel.HV57, h, ground_cn2_m23=1.0e-13)
    assert np.all(overridden > default)


def test_ground_cn2_override_does_not_affect_slc_models():
    """SLC Day/Night are parameter-free fits -- turbulence_cn2_ground must
    not silently rescale them (documented in cn2.py as "not applicable")."""
    h = np.array([5.0, 500.0, 5000.0])
    default = cn2.cn2_profile(TurbulenceModel.SLC_DAY, h, ground_cn2_m23=None)
    with_override = cn2.cn2_profile(TurbulenceModel.SLC_DAY, h, ground_cn2_m23=9.9e-10)
    assert np.array_equal(default, with_override)


# =========================================================================
# Scintillation index saturation
# =========================================================================


def test_scintillation_index_saturates():
    rytov_values = np.array([1.0e-4, 1.0e-2, 1.0, 10.0, 1.0e3, 1.0e6])
    scint_values = np.array([scintillation_index(x) for x in rytov_values])
    assert np.all(np.isfinite(scint_values))
    # weak-fluctuation limit: index ~= rytov variance for small rytov
    assert scint_values[0] == pytest.approx(rytov_values[0], rel=0.05)
    # "does not grow without bound": stays well clear of the huge input
    # values (1e3, 1e6) fed in, settling close to the ~1.0 saturation value
    assert np.all(scint_values < 1.3)
    assert scint_values[-1] == pytest.approx(1.0, abs=0.1)
    assert scint_values[-1] < scint_values[3]  # settles back down from the mid-range peak


def test_scintillation_index_monotonic_in_weak_regime():
    assert scintillation_index(0.01) > scintillation_index(0.001) > scintillation_index(0.0)


# =========================================================================
# Greenwood frequency / Strehl ratio sanity
# =========================================================================


def test_bufton_wind_peaks_near_jet_stream_altitude():
    h = np.linspace(0.0, 20000.0, 4000)
    v = cn2.bufton_wind_speed(h)
    # at h=0 the Gaussian jet-stream term hasn't fully vanished, so ground
    # wind is a lower *bound*, not an exact value, at the surface
    assert cn2.BUFTON_GROUND_WIND_M_S <= v[0] < cn2.BUFTON_GROUND_WIND_M_S + 5.0
    peak_alt = h[np.argmax(v)]
    assert abs(peak_alt - cn2.BUFTON_PEAK_ALTITUDE_M) < 200.0
    assert v.max() == pytest.approx(
        cn2.BUFTON_GROUND_WIND_M_S + cn2.BUFTON_PEAK_WIND_M_S, rel=0.01
    )


def test_strehl_ratio_bounded_and_decreasing_with_aperture():
    r0_m = 0.05
    strehl_small_d = strehl_ratio_marechal(r0_m, 0.01)
    strehl_large_d = strehl_ratio_marechal(r0_m, 1.0)
    assert 0.0 <= strehl_large_d <= strehl_small_d <= 1.0


def test_greenwood_frequency_positive_for_hv57(profile, vertical_path):
    scenario = Scenario(turbulence=TurbulenceModel.HV57)
    result = compute_turbulence(scenario=scenario, path=vertical_path, profile=profile)
    assert result.greenwood_frequency_hz is not None
    assert result.greenwood_frequency_hz > 0.0


# =========================================================================
# TurbulenceModel.NONE guard + real run_scenario wiring
# =========================================================================


def test_run_scenario_populates_and_guards_turbulence():
    """Exercises the real api.run_scenario -> _attach_turbulence ->
    compute_turbulence wiring (not just the standalone formulas), and
    confirms the TurbulenceModel.NONE guard already implemented in api.py.
    Two calls at a narrow, coarse band per CONTRACT.md's cost-discipline
    note -- not a loop, and not a full-band/high-resolution scenario."""
    from mwir_atmos.api import run_scenario

    common = dict(wavenumber_min_cm1=2300.0, wavenumber_max_cm1=2400.0, resolution_cm1=8.0)

    scenario_on = Scenario(turbulence=TurbulenceModel.HV57, **common)
    result_on = run_scenario(scenario_on, include_radiance=False, include_turbulence=True)
    assert result_on.turbulence is not None
    assert np.isfinite(result_on.turbulence.fried_parameter_r0_cm)

    scenario_off = Scenario(turbulence=TurbulenceModel.NONE, **common)
    result_off = run_scenario(scenario_off, include_radiance=False, include_turbulence=True)
    assert result_off.turbulence is None


# =========================================================================
# Every TurbulenceResult field finite and physically signed
# =========================================================================


@pytest.mark.parametrize(
    "model,ground_cn2",
    [
        (TurbulenceModel.HV57, None),
        (TurbulenceModel.SLC_DAY, None),
        (TurbulenceModel.SLC_NIGHT, None),
        (TurbulenceModel.CONSTANT, 1.0e-14),
    ],
)
def test_every_field_finite_and_physically_signed(profile, vertical_path, model, ground_cn2):
    scenario = Scenario(turbulence=model, turbulence_cn2_ground=ground_cn2)
    result = compute_turbulence(scenario=scenario, path=vertical_path, profile=profile)

    assert np.isfinite(result.reference_wavelength_um) and result.reference_wavelength_um > 0.0
    assert np.all(np.isfinite(result.cn2_profile_m23))
    assert np.all(result.cn2_profile_m23 >= 0.0)
    assert np.isfinite(result.fried_parameter_r0_cm) and result.fried_parameter_r0_cm > 0.0
    assert np.isfinite(result.isoplanatic_angle_urad) and result.isoplanatic_angle_urad > 0.0
    assert np.isfinite(result.rytov_variance) and result.rytov_variance >= 0.0
    assert np.isfinite(result.scintillation_index) and result.scintillation_index >= 0.0
    assert result.greenwood_frequency_hz is not None
    assert np.isfinite(result.greenwood_frequency_hz) and result.greenwood_frequency_hz >= 0.0
    assert result.strehl_ratio is not None
    assert np.isfinite(result.strehl_ratio)
    assert 0.0 <= result.strehl_ratio <= 1.0


def test_assumed_aperture_diameter_is_positive_and_documented():
    # Sanity check on the module-level assumption metrics.py documents and
    # uses for Strehl ratio, since Scenario has no aperture-diameter field.
    assert 0.0 < ASSUMED_APERTURE_DIAMETER_M < 1.0
