"""Line-resolved spectral sampling (mwir_atmos.resolved) and its api wiring.

Unit tests pin the instrument functions, the fine-grid plan and the
correlated-k combination identities. The end-to-end tests check resolved
output against an independent calculation: a point-sampled run on the same
fine grid, averaged by hand through the same instrument function.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mwir_atmos import resolved
from mwir_atmos.api import run_scenario
from mwir_atmos.types import RADIANCE_COMPONENTS
from mwir_atmos.constants import GAS_CONSTANT_R, LIGHT_C, STD_PRESSURE_MBAR
from mwir_atmos.types import (
    InstrumentFunction,
    LayerOptics,
    PathLayer,
    RadianceResult,
    RadianceSolver,
    Scenario,
    SpectralGrid,
    SpectralSampling,
)

BOX, TRIANGLE, GAUSSIAN = InstrumentFunction.BOX, InstrumentFunction.TRIANGLE, InstrumentFunction.GAUSSIAN


def _output_grid(first=2000.0, n=3, step=1.0) -> SpectralGrid:
    return SpectralGrid(wavenumber_cm1=first + step * np.arange(n, dtype=np.float64),
                        resolution_cm1=step)


def _layer(p_mbar, t_k, air_column_cm2) -> PathLayer:
    return PathLayer(index=0, z_bottom_km=0.0, z_top_km=1.0, ds_km=1.0, p_mbar=p_mbar, t_k=t_k,
                     rh_pct=0.0, air_column_cm2=air_column_cm2, column_cm2={}, p_eff_mbar={},
                     t_eff_k={})


# =========================================================================
# Instrument functions and degradation
# =========================================================================


def test_box_weights_are_the_trapezoid_rule_or_a_plain_mean():
    np.testing.assert_allclose(resolved.instrument_weights(BOX, 1.0, 0.25),
                               [0.125, 0.25, 0.25, 0.25, 0.125], rtol=1e-14)
    np.testing.assert_allclose(resolved.instrument_weights(BOX, 0.75, 0.25),
                               [1 / 3, 1 / 3, 1 / 3], rtol=1e-14)


def test_triangle_and_gaussian_weights():
    np.testing.assert_allclose(resolved.instrument_weights(TRIANGLE, 0.5, 0.25),
                               [0.25, 0.5, 0.25], rtol=1e-14)
    w = resolved.instrument_weights(GAUSSIAN, 1.0, 0.1)
    centre = w.size // 2
    assert w.size == 41                                        # truncated at +/-2 FWHM
    assert w.sum() == pytest.approx(1.0, rel=1e-14)
    np.testing.assert_allclose(w, w[::-1], rtol=1e-14)
    assert w[centre + 5] / w[centre] == pytest.approx(0.5, rel=1e-12)   # half maximum at FWHM/2


@pytest.mark.parametrize("fwhm,step", [(0.0, 0.1), (-1.0, 0.1), (1.0, 0.0), (1.0, float("nan"))])
def test_instrument_weights_reject_nonphysical_input(fwhm, step):
    with pytest.raises(ValueError):
        resolved.instrument_weights(BOX, fwhm, step)


@pytest.mark.parametrize("function", list(InstrumentFunction))
def test_degrade_preserves_constants_and_straight_lines(function):
    plan = resolved.plan_fine_grid(_output_grid(), fine_step_cm1=0.1, function=function,
                                   fwhm_cm1=None)
    n_out = 3
    n_fine = (n_out - 1) * plan.ratio + 2 * plan.support + 1
    nu = plan.origin_cm1 + plan.step_cm1 * (np.arange(n_fine) - plan.support)
    np.testing.assert_allclose(resolved.degrade(np.full(n_fine, 3.7), plan, n_out), 3.7, rtol=1e-14)
    line = 0.25 + 1.0e-3 * nu
    expected = 0.25 + 1.0e-3 * _output_grid().wavenumber_cm1
    np.testing.assert_allclose(resolved.degrade(line, plan, n_out), expected, rtol=1e-12)
    stacked = resolved.degrade(np.stack([line, 2 * line]), plan, n_out)
    np.testing.assert_allclose(stacked, np.stack([expected, 2 * expected]), rtol=1e-12)


def test_fine_grid_plan_subdivides_the_output_step():
    plan = resolved.plan_fine_grid(_output_grid(), fine_step_cm1=0.03, function=BOX, fwhm_cm1=None)
    assert plan.ratio == 34
    assert plan.step_cm1 * plan.ratio == pytest.approx(1.0, rel=1e-14)
    assert plan.step_cm1 <= 0.03
    assert plan.origin_cm1 == 2000.0
    # A fine step coarser than a quarter of the FWHM is refined to resolve the instrument.
    narrow = resolved.plan_fine_grid(_output_grid(), fine_step_cm1=0.1, function=TRIANGLE,
                                     fwhm_cm1=0.2)
    assert narrow.step_cm1 <= 0.05


def test_chunks_tile_the_output_grid_on_one_lattice(monkeypatch):
    monkeypatch.setattr(resolved, "_MIN_CHUNK_FINE_POINTS", 1)
    monkeypatch.setattr(resolved, "_CHUNK_ELEMENTS", 1000)
    output = _output_grid(n=37)
    plan = resolved.plan_fine_grid(output, fine_step_cm1=0.1, function=TRIANGLE, fwhm_cm1=None)
    covered = []
    for first, stop, fine in resolved.chunks(plan, output.n, n_layers=5):
        covered.extend(range(first, stop))
        assert fine.lattice_origin_cm1 == plan.origin_cm1
        assert fine.lattice_index0 == first * plan.ratio - plan.support
        assert fine.n == (stop - 1 - first) * plan.ratio + 2 * plan.support + 1
        centres = fine.wavenumber_cm1[plan.support::plan.ratio][: stop - first]
        np.testing.assert_allclose(centres, output.wavenumber_cm1[first:stop], rtol=1e-12)
    assert covered == list(range(output.n))


# =========================================================================
# Automatic fine step
# =========================================================================


def _reference_step(p_mbar, t_k, nu):
    gamma = 0.07 * p_mbar / STD_PRESSURE_MBAR * (296.0 / t_k) ** 0.75
    alpha = nu * np.sqrt(2 * np.log(2) * GAS_CONSTANT_R * t_k / (44.0e-3 * LIGHT_C**2))
    return (0.5346 * gamma + np.sqrt(0.2166 * gamma**2 + alpha**2)) / 2.0


def test_automatic_fine_step_resolves_the_relevant_line_widths():
    surface = _layer(1013.25, 288.0, 1.0e25)
    step = resolved.automatic_fine_step([surface], 2000.0)
    assert step == pytest.approx(_reference_step(1013.25, 288.0, 2000.0), rel=1e-12)

    # A mesospheric layer holding a negligible share of the column is ignored...
    thin_top = _layer(0.01, 220.0, 1.0e20)
    assert resolved.automatic_fine_step([surface, thin_top], 2000.0) == pytest.approx(step, rel=1e-12)
    # ...but a narrow-lined layer holding much of the column sets the step.
    thick_top = _layer(1.0, 220.0, 1.0e25)
    assert resolved.automatic_fine_step([surface, thick_top], 2000.0) == pytest.approx(
        _reference_step(1.0, 220.0, 2000.0), rel=1e-12)
    assert resolved.automatic_fine_step([], 2000.0) == resolved._FINE_STEP_BOUNDS[1]


# =========================================================================
# Correlated-k binning for per-spectral-point solvers
# =========================================================================


def _random_column_optics(n_layers, n_fine, seed=3):
    rng = np.random.default_rng(seed)
    tau = lambda scale: scale * rng.lognormal(0.0, 2.0, (n_layers, n_fine))
    grid = SpectralGrid(wavenumber_cm1=np.linspace(1999.0, 2003.0, n_fine), resolution_cm1=0.1)
    return LayerOptics(tau_molecular=tau(0.3), tau_aerosol_abs=tau(0.01), tau_aerosol_sca=tau(0.02),
                       tau_rayleigh=tau(1e-4), grid=grid, tau_by_species={},
                       aerosol_asymmetry=rng.uniform(0.2, 0.8, (n_layers, n_fine)))


def _stub_result(grid, total, path_thermal, errors):
    zeros = np.zeros(grid.n)
    fields = {f.name: zeros for f in dataclasses.fields(RadianceResult)
              if f.name in RADIANCE_COMPONENTS}
    fields.update(total=total, path_thermal=path_thermal)
    result = RadianceResult(grid=grid, **fields)
    result.sampling_standard_error = {"total": errors}
    return result


def test_binned_radiance_reproduces_linear_functionals_exactly():
    """Interval weights sum to the instrument weights, and optics are averaged
    with the same weights, so any solver *linear* in the layer optical depths
    reproduces the instrument-degraded fine spectrum exactly."""
    output = _output_grid()
    plan = resolved.plan_fine_grid(output, fine_step_cm1=0.1, function=TRIANGLE, fwhm_cm1=None)
    n_fine = (output.n - 1) * plan.ratio + 2 * plan.support + 1
    column = _random_column_optics(3, n_fine)
    seen = {}

    def solve(grid, optics):
        seen["grid"], seen["optics"] = grid, optics
        return _stub_result(grid, total=optics.tau_total.sum(axis=0), path_thermal=np.ones(grid.n),
                            errors=np.ones(grid.n))

    out = resolved.binned_radiance(column, plan, output.n, output.wavenumber_cm1, solve, output)
    np.testing.assert_allclose(out.total, resolved.degrade(column.tau_total.sum(axis=0), plan, output.n),
                               rtol=1e-12)
    np.testing.assert_allclose(out.path_thermal, 1.0, rtol=1e-12)
    assert np.all((out.sampling_standard_error["total"] > 0) & (out.sampling_standard_error["total"] <= 1))

    pseudo = seen["grid"]
    assert pseudo.n <= output.n * (resolved._G_EDGES.size - 1)
    assert set(np.unique(pseudo.wavenumber_cm1)) <= set(output.wavenumber_cm1)
    g = seen["optics"].aerosol_asymmetry
    assert np.all((g >= 0.2 - 1e-12) & (g <= 0.8 + 1e-12))



def test_reference_correction_replaces_the_binning_error_of_the_reference_solver():
    """binned(solve) + exact(ref) - binned(ref): with ref = solve the result is
    exact, and a solver differing from ref by a linear term keeps only that
    term's (exact) binning -- a nonlinear shared term cancels."""
    output = _output_grid()
    plan = resolved.plan_fine_grid(output, fine_step_cm1=0.1, function=TRIANGLE, fwhm_cm1=None)
    n_fine = (output.n - 1) * plan.ratio + 2 * plan.support + 1
    column = _random_column_optics(3, n_fine)

    def reference(grid, optics):                     # nonlinear in optical depth
        t = np.exp(-optics.tau_total.sum(axis=0))
        return _stub_result(grid, total=t, path_thermal=1 - t, errors=np.zeros(grid.n))

    def solver(grid, optics):                        # reference + a linear term
        r = reference(grid, optics)
        r.total = r.total + optics.tau_total.sum(axis=0)
        return r

    fine = SpectralGrid(wavenumber_cm1=np.arange(n_fine, dtype=float), resolution_cm1=plan.step_cm1)
    exact = resolved.degrade_radiance(reference(fine, column), plan, output.n, output)
    out = resolved.binned_radiance(column, plan, output.n, output.wavenumber_cm1, solver, output,
                                   reference_solve=reference, reference_exact=exact)
    linear = resolved.degrade(column.tau_total.sum(axis=0), plan, output.n)
    np.testing.assert_allclose(out.total, exact.total + linear, rtol=1e-12)
    np.testing.assert_allclose(out.path_thermal, exact.path_thermal, rtol=1e-12)
    uncorrected = resolved.binned_radiance(column, plan, output.n, output.wavenumber_cm1, solver, output)
    assert np.abs(uncorrected.total - exact.total - linear).max() > 1e-6

# =========================================================================
# run_scenario with resolved sampling
# =========================================================================

#: 1 km horizontal surface path over 2140-2143 cm^-1 (CO and H2O lines).
_RESOLVED = Scenario(sensor_alt_km=0.01, sensor_zenith_deg=90.0, source_range_km=1.0,
                     wavenumber_min_cm1=2140.0, wavenumber_max_cm1=2143.0, resolution_cm1=1.0,
                     spectral_sampling=SpectralSampling.RESOLVED, fine_step_cm1=0.05)


def _box_average(values, n_out, ratio=20):
    weights = np.r_[0.5, np.ones(ratio - 1), 0.5] / ratio
    return np.stack([values[..., j * ratio: j * ratio + ratio + 1] @ weights for j in range(n_out)],
                    axis=-1)


def test_resolved_sampling_equals_a_hand_averaged_fine_point_run():
    result = run_scenario(_RESOLVED)
    assert result.grid.n == 4 and result.fine_step_cm1 == pytest.approx(0.05)
    np.testing.assert_array_equal(result.grid.wavenumber_cm1,
                                  run_scenario(dataclasses.replace(
                                      _RESOLVED, spectral_sampling=SpectralSampling.POINT),
                                      include_radiance=False).grid.wavenumber_cm1)

    # The same physics on the fine grid (support of 10 fine points each side).
    fine = SpectralGrid(wavenumber_cm1=2140.0 + 0.05 * np.arange(-10, 3 * 20 + 11),
                        resolution_cm1=0.05)
    point = run_scenario(dataclasses.replace(_RESOLVED, spectral_sampling=SpectralSampling.POINT),
                         grid=fine)
    np.testing.assert_allclose(result.transmittance.total, _box_average(point.transmittance.total, 4),
                               rtol=1e-9)
    np.testing.assert_allclose(result.transmittance.cumulative,
                               _box_average(point.transmittance.cumulative, 4), rtol=1e-9)
    np.testing.assert_allclose(result.radiance.total, _box_average(point.radiance.total, 4), rtol=1e-9)


def test_resolved_components_still_sum_and_bound():
    r = run_scenario(_RESOLVED)
    parts = [getattr(r.radiance, k) for k in RADIANCE_COMPONENTS if k != "total"]
    np.testing.assert_allclose(r.radiance.total, sum(parts), rtol=1e-12)
    assert np.all((r.transmittance.total >= 0.0) & (r.transmittance.total <= 1.0))
    for values in r.transmittance.by_species.values():
        assert values.shape == (r.grid.n,)


def test_resolved_results_do_not_depend_on_chunking(monkeypatch):
    whole = run_scenario(_RESOLVED)
    monkeypatch.setattr(resolved, "_MIN_CHUNK_FINE_POINTS", 1)
    monkeypatch.setattr(resolved, "_CHUNK_ELEMENTS", 1)          # one output point per chunk
    split = run_scenario(_RESOLVED)
    np.testing.assert_allclose(split.transmittance.total, whole.transmittance.total, rtol=1e-12)
    np.testing.assert_allclose(split.radiance.total, whole.radiance.total, rtol=1e-12)


def test_resolved_instrument_width_is_independent_of_grid_spacing():
    wide = run_scenario(dataclasses.replace(_RESOLVED, instrument_function=TRIANGLE,
                                            instrument_fwhm_cm1=2.0), include_radiance=False)
    narrow = run_scenario(_RESOLVED, include_radiance=False)
    assert wide.grid.n == narrow.grid.n
    np.testing.assert_array_equal(wide.grid.wavenumber_cm1, narrow.grid.wavenumber_cm1)
    assert np.all((wide.transmittance.total >= 0.0) & (wide.transmittance.total <= 1.0))
    assert not np.allclose(wide.transmittance.total, narrow.transmittance.total, rtol=1e-6)


def test_resolved_disort_uses_binned_columns():
    s = dataclasses.replace(
        _RESOLVED, radiance_solver=RadianceSolver.DISORT, disort_streams=8, sensor_alt_km=0.1,
        sensor_zenith_deg=60.0, source_range_km=None, wavenumber_min_cm1=2500.0,
        wavenumber_max_cm1=2501.0, include_solar=False, include_lunar=False)
    r = run_scenario(s)
    assert r.fine_step_cm1 == pytest.approx(0.05)
    assert np.all(np.isfinite(r.radiance.total)) and np.all(r.radiance.total > 0)
    parts = [getattr(r.radiance, k) for k in RADIANCE_COMPONENTS if k != "total"]
    np.testing.assert_allclose(r.radiance.total, sum(parts), rtol=1e-10)


def test_worker_processes_reproduce_the_serial_result(monkeypatch):
    """Chunks evaluated in worker processes match the serial run (to rounding:
    parallel runs use smaller chunks, which changes summation order)."""
    from mwir_atmos import api
    scenario = dataclasses.replace(_RESOLVED, wavenumber_max_cm1=2150.0)
    monkeypatch.setenv("MWIR_PARALLEL", "threads")
    serial = run_scenario(scenario, n_threads=2)
    monkeypatch.setenv("MWIR_PARALLEL", "processes")
    monkeypatch.setattr(api, "_PROCESS_MIN_ELEMENTS", 0)
    monkeypatch.setattr(resolved, "_MIN_CHUNK_FINE_POINTS", 1)
    calls = []
    original = api._run_chunks_in_processes
    monkeypatch.setattr(api, "_run_chunks_in_processes",
                        lambda *a: calls.append(len(a[1])) or original(*a))
    parallel = run_scenario(scenario, n_threads=2)
    assert calls and calls[0] > 1
    np.testing.assert_allclose(parallel.transmittance.total, serial.transmittance.total, rtol=1e-13)
    np.testing.assert_allclose(parallel.radiance.total, serial.radiance.total, rtol=1e-13)
