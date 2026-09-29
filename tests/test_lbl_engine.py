"""Line-by-line engine internals: the fast paths are exact, and line selection is physical.

Pins the September 2026 engine changes (see PHYSICS_VALIDATION.md, "Line
selection and spectral sampling"):

* the ragged direct evaluation equals a naive per-line Voigt sum;
* multigrid wing interpolation stays within its documented error bound;
* spectral chunks sharing a lattice reproduce an undivided evaluation;
* the thread count never changes results;
* the peak-optical-depth selection keeps exactly the lines above threshold,
  including the weak window water lines the former relative cutoff discarded.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import wofz

from mwir_atmos.constants import C2_RAD, GAS_CONSTANT_R, LIGHT_C, LOSCHMIDT_CM3, STD_PRESSURE_MBAR, STD_TEMP_K
from mwir_atmos.hitran.cache import LineList
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics import lbl
from mwir_atmos.optics.lbl import absorption_cross_section, molecular_optical_depth
from mwir_atmos.spectral import make_grid
from mwir_atmos.testing import make_synthetic_path
from mwir_atmos.types import Species, SpectralGrid

LINELISTS = ensure_linelists()


def _water_lines(nu, s296=1.0e-20, e_lower=(200.0,)) -> LineList:
    """Synthetic H2O-161 lines (molecule 1, isotopologue 1)."""
    nu = np.asarray(nu, dtype=np.float64)
    n = nu.size
    return LineList(
        species=Species.H2O, molecule_id=1, nu_cm1=nu,
        s296=s296 * np.linspace(1.0, 3.0, n),
        gamma_air=np.linspace(0.06, 0.10, n), gamma_self=np.full(n, 0.4),
        e_lower=np.resize(np.asarray(e_lower, dtype=np.float64), n),
        n_air=np.full(n, 0.7), delta_air=np.full(n, -0.005),
        local_iso_id=np.ones(n, dtype=np.int64), iso_abundance=np.ones(n),
    )


def _lattice_grid(origin: float, step: float, first: int, stop: int) -> SpectralGrid:
    index = np.arange(first, stop, dtype=np.float64)
    return SpectralGrid(wavenumber_cm1=origin + step * index, resolution_cm1=step,
                        lattice_origin_cm1=origin, lattice_index0=first)


def _uses_multigrid(linelist, grid, p, t, p_self, cutoff, pedestal) -> bool:
    lines = lbl._select_lines(lbl._prepare(linelist), grid, p, t, p_self, cutoff, pedestal,
                              0.0, None, None, 0.0)
    return lbl._multigrid_plan(grid, lines, cutoff) is not None


# =========================================================================
# Exactness of the fast paths
# =========================================================================


def test_direct_evaluation_equals_a_naive_voigt_sum():
    """Ragged batches, sorting and window selection change nothing physical."""
    # Lines inside, straddling and outside the grid, and one beyond the cutoff.
    linelist = _water_lines([1998.5, 2009.999, 2002.345, 2005.0, 2014.5], e_lower=(50., 900., 300.))
    grid = make_grid(2000.0, 2010.0, 0.01)
    p, t, p_self, cutoff = 800.0, 260.0, 8.0, 3.0

    actual = absorption_cross_section(linelist, grid, p, t, p_self, wing_cutoff_cm1=cutoff,
                                      min_strength_fraction=0.0, method="direct")

    s_t = lbl._line_intensity_at_t(linelist, t)
    mass = lbl._isotopologue_mass_g_mol(linelist) * 1.0e-3
    alpha = linelist.nu_cm1 * np.sqrt(2.0 * np.log(2.0) * GAS_CONSTANT_R * t / (mass * LIGHT_C**2))
    p_atm, self_atm = p / STD_PRESSURE_MBAR, p_self / STD_PRESSURE_MBAR
    gamma = ((linelist.gamma_air * (p_atm - self_atm) + linelist.gamma_self * self_atm)
             * (296.0 / t) ** linelist.n_air)
    centre = linelist.nu_cm1 + linelist.delta_air * p_atm
    expected = np.zeros(grid.n)
    for s, a, g, c in zip(s_t, alpha, gamma, centre):
        dnu = grid.wavenumber_cm1 - c
        z = np.sqrt(np.log(2.0)) * (dnu + 1j * g) / a
        profile = np.sqrt(np.log(2.0) / np.pi) / a * wofz(z).real
        expected += np.where(np.abs(dnu) <= cutoff, s * profile, 0.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=expected.max() * 1e-15)
    assert actual[-1] > 0.0                       # the line at 2009.999 reaches the edge


def test_line_intensity_keeps_the_linelists_own_order():
    """The engine sorts lines internally; the public helper must not."""
    linelist = _water_lines([2300.0, 2100.0, 2200.0], e_lower=(10.0, 1500.0, 700.0))
    t = 240.0
    q_ratio = (lbl.partition_function(1, 1, 296.0) / lbl.partition_function(1, 1, t))
    c2 = C2_RAD
    expected = (linelist.s296 * q_ratio * np.exp(-c2 * linelist.e_lower * (1 / t - 1 / 296.0))
                * np.expm1(-c2 * linelist.nu_cm1 / t) / np.expm1(-c2 * linelist.nu_cm1 / 296.0))
    np.testing.assert_allclose(lbl._line_intensity_at_t(linelist, t), expected, rtol=1e-6)


def test_reassigning_a_linelist_field_invalidates_the_cache():
    grid = make_grid(2000.0, 2010.0, 0.01)
    line = _water_lines([2004.0])
    first = absorption_cross_section(line, grid, 1013.25, 296.0, 0.0, wing_cutoff_cm1=3.0)
    line.nu_cm1 = np.array([2006.0])
    second = absorption_cross_section(line, grid, 1013.25, 296.0, 0.0, wing_cutoff_cm1=3.0)
    shift = grid.wavenumber_cm1[np.argmax(second)] - grid.wavenumber_cm1[np.argmax(first)]
    assert shift == pytest.approx(2.0, abs=0.011)


@pytest.mark.parametrize("species,lo,hi,step,p,t,vmr", [
    (Species.CO2, 2380.0, 2400.0, 0.005, 1013.25, 295.0, 420e-6),
    (Species.O3, 2140.0, 2160.0, 0.01, 1013.25, 295.0, 5e-8),
    (Species.H2O, 3700.0, 3710.0, 0.002, 1013.25, 295.0, 0.015),
])
def test_multigrid_matches_direct_evaluation(species, lo, hi, step, p, t, vmr):
    """Coarse-mesh wings: within ~1e-4 of exact where the cross-section matters.

    Cubic interpolation error is ~2.8*(H/d)**4 of the local wing value, and
    the mesh spacing H is a twelfth of the exactly evaluated core. Points
    below 1e-6 of the maximum are excluded from the relative comparison.
    The relative line cutoff (1e-4) only keeps the test fast.
    """
    linelist = LINELISTS[species]
    grid = make_grid(lo, hi, step)
    pedestal = species == Species.H2O
    args = (linelist, grid, p, t, p * vmr)
    kwargs = dict(wing_cutoff_cm1=25.0, subtract_pedestal=pedestal, min_strength_fraction=1e-4)
    assert _uses_multigrid(linelist, grid, p, t, p * vmr, 25.0, pedestal)

    direct = absorption_cross_section(*args, method="direct", **kwargs)
    multigrid = absorption_cross_section(*args, method="multigrid", **kwargs)
    floor = 1.0e-6 * direct.max()
    relative = np.abs(multigrid - direct) / np.maximum(np.abs(direct), floor)
    assert relative.max() < 5.0e-4
    assert np.median(relative) < 2.0e-5


def test_multigrid_chunks_reproduce_the_undivided_grid():
    """A shared lattice makes spectral chunking invisible (to rounding)."""
    linelist = LINELISTS[Species.CO2]
    origin, step, n = 2380.0, 0.005, 2001
    full_grid = _lattice_grid(origin, step, 0, n)
    args = (1013.25, 295.0, 1013.25 * 420e-6)
    kwargs = dict(wing_cutoff_cm1=25.0, min_strength_fraction=1e-4, method="multigrid")
    assert _uses_multigrid(linelist, full_grid, *args, 25.0, False)

    full = absorption_cross_section(linelist, full_grid, *args, **kwargs)
    parts = [absorption_cross_section(linelist, _lattice_grid(origin, step, a, b), *args, **kwargs)
             for a, b in ((0, 700), (700, 1401), (1401, n))]
    np.testing.assert_allclose(np.concatenate(parts), full, rtol=1e-11, atol=full.max() * 1e-14)


def test_thread_count_does_not_change_results():
    grid = make_grid(2600.0, 2620.0, 0.5)
    layers = make_synthetic_path(sensor_alt_km=0.0, zenith_deg=40.0, n_layers=4).layers
    one, one_species = molecular_optical_depth(layers, grid, linelists=LINELISTS, n_threads=1)
    many, many_species = molecular_optical_depth(layers, grid, linelists=LINELISTS, n_threads=4)
    np.testing.assert_array_equal(one, many)
    for species in one_species:
        np.testing.assert_array_equal(one_species[species], many_species[species])


def test_thread_count_resolution(monkeypatch):
    monkeypatch.delenv("MWIR_THREADS", raising=False)
    assert lbl.resolve_thread_count(3) == 3
    assert lbl.resolve_thread_count() >= 1
    monkeypatch.setenv("MWIR_THREADS", "2")
    assert lbl.resolve_thread_count() == 2
    monkeypatch.setenv("MWIR_THREADS", "many")
    with pytest.raises(ValueError, match="MWIR_THREADS"):
        lbl.resolve_thread_count()
    with pytest.raises(ValueError):
        lbl.resolve_thread_count(0)


# =========================================================================
# Line selection
# =========================================================================


def test_peak_optical_depth_cutoff_is_an_exact_threshold():
    """A line is evaluated iff S(T) * column * V(0) reaches the threshold."""
    grid = make_grid(2000.0, 2010.0, 0.005)
    line = _water_lines([2005.0], s296=3.0e-22)
    column = 2.0e21
    # At 296 K and zero self pressure: S = s296, gamma_L = gamma_air.
    alpha = 2005.0 * np.sqrt(2 * np.log(2) * GAS_CONSTANT_R * 296.0
                             / (lbl._isotopologue_mass_g_mol(line)[0] * 1.0e-3 * LIGHT_C**2))
    y = np.sqrt(np.log(2.0)) * line.gamma_air[0] / alpha
    peak = np.sqrt(np.log(2.0) / np.pi) / alpha * wofz(1j * y).real
    tau_peak = line.s296[0] * column * peak

    def evaluate(threshold, column_cm2=column):
        return absorption_cross_section(line, grid, STD_PRESSURE_MBAR, 296.0, 0.0,
                                        wing_cutoff_cm1=3.0, column_cm2=column_cm2,
                                        min_peak_optical_depth=threshold)

    assert evaluate(tau_peak * (1 - 1e-9)).max() > 0.0
    assert np.all(evaluate(tau_peak * (1 + 1e-9)) == 0.0)
    assert evaluate(tau_peak * 2.0, column_cm2=None).max() > 0.0   # no column: no criterion


@pytest.mark.parametrize("atmosphere, zenith, strength_fraction", [
    ("midlat_summer", 0.0, 0.0),       # surface to space, all below 296 K
    ("tropical", 90.0, 0.0),           # hot (> 296 K) humid surface layer
    ("subarctic_winter", 0.0, 1e-9),   # cold, plus the strength-fraction test
])
def test_peak_prefilter_does_not_change_line_selection(monkeypatch, atmosphere, zenith,
                                                       strength_fraction):
    """The pre-filter bound only skips lines the exact peak test would drop."""
    from mwir_atmos.api import build_profile, build_scenario_path
    from mwir_atmos.types import ModelAtmosphere, Scenario

    scenario = Scenario(atmosphere=ModelAtmosphere(atmosphere), sensor_zenith_deg=zenith,
                        source_range_km=5.0 if zenith == 90.0 else None)
    path = build_scenario_path(scenario, build_profile(scenario))
    grid = make_grid(2300.0, 2450.0, 0.01)
    rows = sorted({0, len(path.layers) // 3, len(path.layers) - 1})
    compared = 0
    for row in rows:
        layer = path.layers[row]
        for sp, linelist in LINELISTS.items():
            column = layer.column_cm2.get(sp, 0.0)
            if column <= 0.0:
                continue
            p = layer.p_eff_mbar.get(sp, layer.p_mbar)
            args = (lbl._prepare(linelist), grid, p, layer.t_eff_k.get(sp, layer.t_k),
                    p * column / layer.air_column_cm2, 25.0, sp == Species.H2O,
                    strength_fraction, None, column, lbl.DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF)
            monkeypatch.setattr(lbl, "_PEAK_PREFILTER", True)
            fast = lbl._select_lines(*args)
            monkeypatch.setattr(lbl, "_PEAK_PREFILTER", False)
            exact = lbl._select_lines(*args)
            for name in ("center", "strength", "alpha_d", "gamma_l"):
                np.testing.assert_array_equal(getattr(fast, name), getattr(exact, name))
            assert (fast.pedestal is None) == (exact.pedestal is None)
            if fast.pedestal is not None:
                np.testing.assert_array_equal(fast.pedestal, exact.pedestal)
            assert fast.plan_hwhm == exact.plan_hwhm
            compared += exact.center.size
    assert compared > 1000


def test_weak_window_water_lines_survive_default_line_selection():
    """Regression: the former default discarded every 3.7 um window water line.

    The old rule kept lines stronger than 1e-4 of the list's strongest line
    (2.48e-19 at 296 K). Every main-isotopologue and HDO line between 2440 and
    2860 cm^-1 is weaker than 1e-23, so all were dropped -- 24 percentage
    points of transmittance on a 10 km tropical path. The peak-optical-depth
    rule keeps the lines that are optically significant for the actual column.
    """
    linelist = LINELISTS[Species.H2O]
    grid = make_grid(2600.0, 2700.0, 0.5)
    vmr, t = 0.015, 296.0
    column = LOSCHMIDT_CM3 * (STD_TEMP_K / t) * vmr * 10.0e5          # 10 km at 1 atm
    args = (linelist, grid, STD_PRESSURE_MBAR, t, STD_PRESSURE_MBAR * vmr)

    legacy = absorption_cross_section(*args, subtract_pedestal=True, min_strength_fraction=1e-4)
    assert np.all(legacy == 0.0)

    kept = lbl._select_lines(lbl._prepare(linelist), grid, STD_PRESSURE_MBAR, t,
                             STD_PRESSURE_MBAR * vmr, 25.0, True, 0.0, None, column,
                             lbl.DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF)
    assert kept.center.size > 200
    tau = column * absorption_cross_section(
        *args, subtract_pedestal=True, column_cm2=column,
        min_peak_optical_depth=lbl.DEFAULT_LINE_OPTICAL_DEPTH_CUTOFF)
    assert tau.mean() > 0.02


def test_library_and_scenario_line_selection_defaults_agree():
    import inspect
    from mwir_atmos.optics.transmittance import compute_layer_optics
    from mwir_atmos.types import Scenario

    scenario = Scenario()
    for function in (molecular_optical_depth, compute_layer_optics):
        parameters = inspect.signature(function).parameters
        assert parameters["min_strength_fraction"].default == scenario.line_strength_cutoff
        assert parameters["min_peak_optical_depth"].default == scenario.line_optical_depth_cutoff
