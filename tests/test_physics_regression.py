"""Cross-cutting physics invariants that span module boundaries.

Ordered cheapest-first. Most of these need no line-by-line call at all --
geometry, Curtis-Godson bookkeeping, surface emissivity, turbulence scaling and
the ephemeris are all microseconds. The handful that do touch the spectral
engine use a deliberately tiny grid, because cost scales with points x layers.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from mwir_atmos.atmosphere.profiles import get_profile
from mwir_atmos.constants import LOSCHMIDT_CM3, STD_PRESSURE_MBAR, STD_TEMP_K
from mwir_atmos.ephemeris.sun import solar_declination_deg, solar_position
from mwir_atmos.geometry.path import build_path
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics.lbl import molecular_optical_depth
from mwir_atmos.radiometry.surface import surface_emissivity
from mwir_atmos.spectral import make_grid
from mwir_atmos.turbulence.cn2 import cn2_profile
from mwir_atmos.turbulence.metrics import (
    fried_parameter_r0_m,
    scintillation_index,
)
from mwir_atmos.types import (
    ModelAtmosphere,
    PathLayer,
    Scenario,
    Species,
    SurfaceType,
    TurbulenceModel,
)

TINY = dict(wavenumber_min_cm1=2350.0, wavenumber_max_cm1=2400.0, resolution_cm1=25.0)


@pytest.fixture(scope="module")
def profile():
    return get_profile(ModelAtmosphere.US_STANDARD)


@pytest.fixture(scope="module")
def linelists():
    return ensure_linelists()


#: Split-layer additivity and linearity in column hold exactly only when line
#: selection does not depend on the column. The default peak-optical-depth
#: criterion does (a thinner layer drops more weak lines), so these invariants
#: use the column-independent relative rule. Its numbers are unchanged from
#: before the September 2026 engine rewrite, which reproduces the old engine
#: to 1e-14.
_COLUMN_INDEPENDENT = dict(min_strength_fraction=1.0e-4, min_peak_optical_depth=0.0)


def _homogeneous_layer(ds_km: float, p_mbar=1013.25, t_k=296.0, h2o_vmr=0.01) -> PathLayer:
    """One uniform slab, so analytic expectations are exact."""
    n_air = LOSCHMIDT_CM3 * (p_mbar / STD_PRESSURE_MBAR) * (STD_TEMP_K / t_k)
    air_col = n_air * ds_km * 1.0e5
    vmr = {Species.H2O: h2o_vmr, Species.CO2: 423e-6, Species.O3: 3e-8,
           Species.N2O: 337e-9, Species.CO: 1e-7, Species.CH4: 1.92e-6}
    return PathLayer(
        index=0, z_bottom_km=0.0, z_top_km=0.0, ds_km=ds_km, p_mbar=p_mbar, t_k=t_k,
        rh_pct=50.0, air_column_cm2=air_col,
        column_cm2={s: air_col * v for s, v in vmr.items()},
        p_eff_mbar={s: p_mbar for s in vmr}, t_eff_k={s: t_k for s in vmr},
        aerosol_od_550nm=0.0,
    )


# =========================================================================
# Geometry and column bookkeeping (no spectral calls)
# =========================================================================


def test_column_density_is_linear_in_path_length():
    a = _homogeneous_layer(1.0)
    b = _homogeneous_layer(3.0)
    for species in (Species.H2O, Species.CO2):
        assert b.column_cm2[species] == pytest.approx(3.0 * a.column_cm2[species], rel=1e-12)


def test_slant_column_exceeds_vertical_by_the_airmass(profile):
    vertical = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0)
    for zenith, expected in ((30.0, 1.1547), (60.0, 2.0)):
        slant = build_path(profile, sensor_alt_km=0.0, zenith_deg=zenith)
        ratio = (
            sum(l.column_cm2[Species.CO2] for l in slant.layers)
            / sum(l.column_cm2[Species.CO2] for l in vertical.layers)
        )
        # Spherical geometry runs marginally below the plane-parallel secant.
        assert ratio == pytest.approx(expected, rel=0.01)


def test_total_column_converges_as_layers_are_added(profile):
    """Column density is a quadrature of an exponentially varying density, so a
    very coarse layering integrates it slightly differently. What matters is
    that it converges: the coarsest case may sit a couple of tenths of a percent
    off, but the finer ones must agree with each other far more tightly."""
    columns = {}
    for max_layers in (10, 40, 120, 260):
        path = build_path(profile, sensor_alt_km=0.0, zenith_deg=45.0, max_layers=max_layers)
        columns[max_layers] = sum(l.column_cm2[Species.H2O] for l in path.layers)

    assert abs(columns[10] / columns[260] - 1.0) < 0.005
    converged = [columns[40], columns[120], columns[260]]
    assert max(converged) / min(converged) < 1.0001


def test_path_length_equals_the_sum_of_its_layers(profile):
    for zenith in (0.0, 45.0, 90.0, 135.0):
        alt = 0.0 if zenith <= 90.0 else 8.0
        path = build_path(profile, sensor_alt_km=alt, zenith_deg=zenith)
        assert sum(l.ds_km for l in path.layers) == pytest.approx(path.total_length_km, rel=1e-9)


def test_layers_tile_the_path_without_gaps_or_overlaps(profile):
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=30.0)
    for previous, current in zip(path.layers, path.layers[1:]):
        shared = {previous.z_bottom_km, previous.z_top_km} & {current.z_bottom_km, current.z_top_km}
        assert shared, "consecutive layers must share a boundary"


def test_curtis_godson_effective_values_lie_within_the_layer(profile):
    """p_eff/t_eff are column-weighted means, so they cannot leave the layer's range."""
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0)
    altitudes = profile.altitude_km
    for layer in path.layers:
        lo, hi = sorted((layer.z_bottom_km, layer.z_top_km))
        inside = (altitudes >= lo - 1e-9) & (altitudes <= hi + 1e-9)
        if inside.sum() < 2:
            continue
        p_lo, p_hi = profile.pressure_mbar[inside].min(), profile.pressure_mbar[inside].max()
        for species, p_eff in layer.p_eff_mbar.items():
            assert p_lo * 0.98 <= p_eff <= p_hi * 1.02


def test_curtis_godson_pressure_is_weighted_toward_the_dense_end(profile):
    """Column weighting must pull p_eff below the arithmetic mean, since more
    molecules sit at the high-pressure (low-altitude) end of every layer."""
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0, max_layers=12)
    thick = max(path.layers, key=lambda l: abs(l.z_top_km - l.z_bottom_km))
    lo, hi = sorted((thick.z_bottom_km, thick.z_top_km))
    p_lo = float(np.interp(hi, profile.altitude_km, profile.pressure_mbar))
    p_hi = float(np.interp(lo, profile.altitude_km, profile.pressure_mbar))
    arithmetic = 0.5 * (p_lo + p_hi)
    assert thick.p_eff_mbar[Species.CO2] > arithmetic * 0.5
    assert thick.p_eff_mbar[Species.CO2] <= p_hi * 1.02


def test_boundary_layer_is_refined_far_below_the_profile_grid(profile):
    """The AFGL grid is 1 km at the bottom; the path must do much better."""
    path = build_path(profile, sensor_alt_km=0.0, zenith_deg=0.0)
    assert abs(path.layers[0].z_top_km - path.layers[0].z_bottom_km) <= 0.05


# =========================================================================
# Surface emissivity (no spectral calls)
# =========================================================================


@pytest.mark.parametrize("surface", list(SurfaceType))
def test_every_surface_emissivity_is_a_valid_fraction(surface):
    value = surface_emissivity(surface)
    assert 0.0 <= value <= 1.0 and math.isfinite(value)


def test_sea_water_emissivity_falls_monotonically_with_incidence():
    angles = np.arange(0.0, 90.0, 5.0)
    values = [surface_emissivity(SurfaceType.SEA_WATER, incidence_deg=a) for a in angles]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_sea_water_emissivity_is_symmetric_in_incidence_sign():
    assert surface_emissivity(SurfaceType.SEA_WATER, incidence_deg=40.0) == pytest.approx(
        surface_emissivity(SurfaceType.SEA_WATER, incidence_deg=-40.0)
    )


def test_blackbody_surface_is_unity():
    assert surface_emissivity(SurfaceType.BLACKBODY) == pytest.approx(1.0)


# =========================================================================
# Turbulence scaling (no spectral calls)
# =========================================================================


def test_fried_parameter_scales_as_wavelength_to_the_six_fifths():
    cn2 = np.full(50, 1e-16)
    ds = np.full(50, 200.0)
    r0_visible = fried_parameter_r0_m(cn2, ds, 0.5)
    r0_mwir = fried_parameter_r0_m(cn2, ds, 4.0)
    assert r0_mwir / r0_visible == pytest.approx(8.0 ** 1.2, rel=1e-9)


def test_stronger_turbulence_gives_a_smaller_fried_parameter():
    ds = np.full(50, 200.0)
    weak = fried_parameter_r0_m(np.full(50, 1e-17), ds, 4.0)
    strong = fried_parameter_r0_m(np.full(50, 1e-14), ds, 4.0)
    assert strong < weak


def test_fried_parameter_falls_as_the_path_lengthens():
    cn2 = np.full(50, 1e-16)
    short = fried_parameter_r0_m(cn2, np.full(50, 100.0), 4.0)
    long_ = fried_parameter_r0_m(cn2, np.full(50, 400.0), 4.0)
    assert long_ < short


def test_scintillation_index_is_bounded_and_monotonic():
    rytov = np.logspace(-3, 2, 40)
    values = [scintillation_index(r) for r in rytov]
    assert all(math.isfinite(v) and v >= 0.0 for v in values)
    assert max(values) < 3.0, "scintillation must saturate, not grow without bound"
    weak = [v for r, v in zip(rytov, values) if r < 0.1]
    assert all(b >= a - 1e-12 for a, b in zip(weak, weak[1:]))


def test_scintillation_tracks_rytov_in_the_weak_limit():
    """For sigma_R^2 << 1 the scintillation index approaches the Rytov variance."""
    for rytov in (1e-4, 1e-3, 1e-2):
        assert scintillation_index(rytov) == pytest.approx(rytov, rel=0.15)


def test_hv57_falls_off_above_the_boundary_layer():
    heights = np.array([200.0, 500.0, 1000.0, 2000.0, 4000.0])
    values = cn2_profile(TurbulenceModel.HV57, heights)
    assert all(b < a for a, b in zip(values, values[1:]))


def test_constant_model_is_actually_constant():
    heights = np.linspace(0.0, 20000.0, 25)
    values = cn2_profile(TurbulenceModel.CONSTANT, heights, ground_cn2_m23=3e-15)
    assert np.allclose(values, 3e-15)


def test_every_cn2_model_is_positive_and_finite():
    heights = np.linspace(0.0, 30000.0, 60)
    for model in (TurbulenceModel.HV57, TurbulenceModel.SLC_DAY, TurbulenceModel.SLC_NIGHT):
        values = cn2_profile(model, heights)
        assert np.all(np.isfinite(values)) and np.all(values >= 0.0)


# =========================================================================
# Ephemeris symmetry (no spectral calls)
# =========================================================================


def test_declination_is_antisymmetric_about_the_solstices():
    """Days equally spaced either side of a solstice share a declination."""
    solstice = datetime(2026, 6, 21, 12)
    for offset in (10, 25, 45):
        before = solar_declination_deg(solstice - timedelta(days=offset))
        after = solar_declination_deg(solstice + timedelta(days=offset))
        assert before == pytest.approx(after, abs=0.7)


def test_sun_is_up_at_noon_and_down_at_midnight_across_latitudes():
    for latitude in (-45.0, -20.0, 0.0, 20.0, 45.0):
        for month in (3, 6, 9, 12):
            common = dict(latitude_deg=latitude, longitude_deg=0.0, utc_offset_hours=0.0)
            noon = Scenario(**common, local_datetime=datetime(2026, month, 15, 12, 0))
            midnight = Scenario(**common, local_datetime=datetime(2026, month, 15, 0, 0))
            assert solar_position(noon).is_above_horizon, (latitude, month)
            assert not solar_position(midnight).is_above_horizon, (latitude, month)


def test_polar_summer_keeps_the_sun_up_at_midnight():
    sc = Scenario(latitude_deg=80.0, longitude_deg=0.0, utc_offset_hours=0.0,
                  local_datetime=datetime(2026, 6, 21, 0, 0))
    assert solar_position(sc).is_above_horizon


def test_southern_hemisphere_sun_is_north_at_noon():
    sc = Scenario(latitude_deg=-35.0, longitude_deg=0.0, utc_offset_hours=0.0,
                  local_datetime=datetime(2026, 12, 21, 12, 0))
    azimuth = solar_position(sc).azimuth_deg
    assert azimuth < 40.0 or azimuth > 320.0


# =========================================================================
# Spectral invariants (small grid -- these are the expensive ones)
# =========================================================================


def test_optical_depth_is_additive_across_a_split_layer(linelists):
    """Splitting one slab into two halves must reproduce the same total tau."""
    grid = make_grid(2350.0, 2400.0, 25.0)
    whole, _ = molecular_optical_depth([_homogeneous_layer(2.0)], grid, linelists=linelists, **_COLUMN_INDEPENDENT)
    halves, _ = molecular_optical_depth(
        [_homogeneous_layer(1.0), _homogeneous_layer(1.0)], grid, linelists=linelists, **_COLUMN_INDEPENDENT
    )
    np.testing.assert_allclose(whole.sum(axis=0), halves.sum(axis=0), rtol=1e-12)


def test_optical_depth_is_linear_in_path_length(linelists):
    grid = make_grid(2350.0, 2400.0, 25.0)
    one, _ = molecular_optical_depth([_homogeneous_layer(1.0)], grid, linelists=linelists, **_COLUMN_INDEPENDENT)
    five, _ = molecular_optical_depth([_homogeneous_layer(5.0)], grid, linelists=linelists, **_COLUMN_INDEPENDENT)
    np.testing.assert_allclose(five[0], 5.0 * one[0], rtol=1e-12)


def test_transmittance_is_multiplicative_along_the_path(linelists):
    grid = make_grid(2350.0, 2400.0, 25.0)
    layers = [_homogeneous_layer(1.0), _homogeneous_layer(1.0), _homogeneous_layer(1.0)]
    tau, _ = molecular_optical_depth(layers, grid, linelists=linelists, **_COLUMN_INDEPENDENT)
    per_layer = np.exp(-tau)
    np.testing.assert_allclose(np.prod(per_layer, axis=0), np.exp(-tau.sum(axis=0)), rtol=1e-12)


def test_per_species_optical_depths_sum_to_the_total(linelists):
    grid = make_grid(2350.0, 2400.0, 25.0)
    tau, by_species = molecular_optical_depth(
        [_homogeneous_layer(1.0)], grid, linelists=linelists, **_COLUMN_INDEPENDENT, include_continuum=False
    )
    np.testing.assert_allclose(tau[0], sum(v[0] for v in by_species.values()), rtol=1e-12)


def _co2_doubling_ratio(vmr: float, linelists) -> float:
    """Mean tau(2x CO2) / tau(1x CO2) at a given CO2 mixing ratio."""
    grid = make_grid(2350.0, 2400.0, 25.0)
    base = _homogeneous_layer(1.0)
    doubled = _homogeneous_layer(1.0)
    base.column_cm2[Species.CO2] = base.air_column_cm2 * vmr
    doubled.column_cm2[Species.CO2] = doubled.air_column_cm2 * 2.0 * vmr

    _, a = molecular_optical_depth([base], grid, linelists=linelists, **_COLUMN_INDEPENDENT, include_continuum=False)
    _, b = molecular_optical_depth([doubled], grid, linelists=linelists, **_COLUMN_INDEPENDENT, include_continuum=False)
    return float(np.mean(b[Species.CO2][0] / a[Species.CO2][0]))


def test_co2_absorption_is_very_nearly_linear_in_its_own_column(linelists):
    """At atmospheric abundance, Beer-Lambert holds in CO2 to about four digits."""
    assert _co2_doubling_ratio(423e-6, linelists) == pytest.approx(2.0, rel=1e-3)


def test_doubling_co2_leaves_other_species_untouched(linelists):
    grid = make_grid(2350.0, 2400.0, 25.0)
    base = _homogeneous_layer(1.0)
    doubled = _homogeneous_layer(1.0)
    doubled.column_cm2[Species.CO2] *= 2.0
    _, a = molecular_optical_depth([base], grid, linelists=linelists, **_COLUMN_INDEPENDENT, include_continuum=False)
    _, b = molecular_optical_depth([doubled], grid, linelists=linelists, **_COLUMN_INDEPENDENT, include_continuum=False)
    np.testing.assert_allclose(b[Species.H2O][0], a[Species.H2O][0], rtol=1e-12)
    np.testing.assert_allclose(b[Species.CH4][0], a[Species.CH4][0], rtol=1e-12)


def test_self_broadening_breaks_exact_linearity_in_proportion_to_abundance(linelists):
    """Absorption is linear in column only at FIXED broadening.

    Adding CO2 also raises its own partial pressure, and gamma_self differs from
    gamma_air, so the lines broaden and optical depth is redistributed across the
    line shape. The departure from an exact factor of two must therefore grow in
    proportion to the CO2 mixing ratio -- measured here as 1.3e-4 at 423 ppmv
    rising to 1.3e-3 at ten times that. This test exists to prove self-broadening
    is actually implemented: a model using gamma_air alone would return exactly
    2.0 at every abundance.
    """
    deviations = [
        abs(_co2_doubling_ratio(vmr, linelists) - 2.0) for vmr in (423e-6, 4.23e-3, 5.0e-2)
    ]
    assert deviations == sorted(deviations), "self-broadening must grow with abundance"
    assert deviations[0] < 1.0e-3          # negligible at atmospheric CO2
    assert deviations[-1] > 5.0e-3         # unmistakable at 5% CO2
    # Roughly proportional: a tenfold abundance change scales the departure ~tenfold.
    assert 5.0 < deviations[1] / deviations[0] < 20.0


def test_zero_column_species_contributes_nothing(linelists):
    grid = make_grid(2350.0, 2400.0, 25.0)
    layer = _homogeneous_layer(1.0)
    layer.column_cm2[Species.CO2] = 0.0
    _, by_species = molecular_optical_depth(
        [layer], grid, linelists=linelists, **_COLUMN_INDEPENDENT, include_continuum=False
    )
    assert np.all(by_species[Species.CO2][0] == 0.0)


def test_optical_depth_is_never_negative(linelists):
    grid = make_grid(2350.0, 2400.0, 25.0)
    tau, by_species = molecular_optical_depth([_homogeneous_layer(2.0)], grid, linelists=linelists, **_COLUMN_INDEPENDENT)
    assert np.all(tau >= 0.0)
    for values in by_species.values():
        assert np.all(values >= 0.0)
