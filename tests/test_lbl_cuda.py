"""CUDA line-by-line backend.

The kernel's algorithm runs here in NumPy (``lbl_cuda.evaluate_emulated``), so
the Voigt approximation, precision choices, packing and MT_CKD pedestal handling
are checked on every machine. Tests marked ``gpu`` compare the real kernel with
the CPU engine and skip without CuPy and a CUDA device.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pytest
from scipy.special import wofz

from mwir_atmos.api import build_profile, build_scenario_path, run_scenario
from mwir_atmos.hitran.fetch import ensure_linelists
from mwir_atmos.optics import lbl, lbl_cuda, lbl_cuda_dispatch
from mwir_atmos.optics.voigt_gpu import re_faddeeva
from mwir_atmos.spectral import make_grid
from mwir_atmos.types import AerosolModel, ModelAtmosphere, Scenario, Species, SpectralSampling

gpu = pytest.mark.skipif(not lbl_cuda.available(), reason="needs CuPy and a CUDA device")


def test_voigt_approximation_matches_wofz_everywhere():
    x = np.concatenate([np.linspace(0, 20, 1001), np.geomspace(20, 3e4, 300)])
    y = np.geomspace(1e-4, 3e3, 200)
    xx, yy = np.meshgrid(x, y)
    reference = wofz(xx + 1j * yy).real
    error = np.abs(re_faddeeva(xx, yy) / reference - 1.0)
    assert error.max() < 1e-6
    np.testing.assert_array_equal(re_faddeeva(-xx, yy), re_faddeeva(xx, yy))


def _tasks(grid, scenario: Scenario, rows=(0, 5, -1), species=(Species.H2O, Species.CO2, Species.CO)):
    """(lines, cutoff, column) tasks for a few real layers, and CPU direct optical depths."""
    path = build_scenario_path(scenario, build_profile(scenario))
    linelists = ensure_linelists()
    tasks, expected = [], []
    for row in rows:
        layer = path.layers[row]
        for sp in species:
            column = layer.column_cm2[sp]
            p = layer.p_eff_mbar.get(sp, layer.p_mbar)
            t = layer.t_eff_k.get(sp, layer.t_k)
            water = sp == Species.H2O
            args = (p, t, p * column / layer.air_column_cm2, 25.0, water, 0.0, None, column, 3e-6)
            lines = lbl._select_lines(lbl._prepare(linelists[sp]), grid, *args)
            tasks.append((lines, 25.0, column))
            expected.append(column * lbl.absorption_cross_section(
                linelists[sp], grid, p, t, p * column / layer.air_column_cm2,
                subtract_pedestal=water, method="direct", column_cm2=column,
                min_peak_optical_depth=3e-6))
    return tasks, np.array(expected)


@pytest.mark.parametrize("step", [0.002, 0.05])
def test_emulated_kernel_matches_cpu_direct_sum(step):
    """Ground-to-space layers (surface to stratosphere), H2O with pedestal, CO2, CO."""
    grid = make_grid(2140.0, 2145.0, step)
    tasks, expected = _tasks(grid, Scenario(atmosphere=ModelAtmosphere.MIDLAT_SUMMER))
    got = lbl_cuda.evaluate_emulated(grid.wavenumber_cm1, lbl_cuda.pack(tasks))
    scale = np.abs(expected).max(axis=1, keepdims=True)
    assert np.abs(got - expected).max() <= 2e-6 * scale.max()
    assert np.all(np.abs(got - expected) <= 2e-6 * scale + 1e-30)


def test_pack_sorts_lines_and_keeps_task_boundaries():
    grid = make_grid(2140.0, 2145.0, 0.05)
    tasks, _ = _tasks(grid, Scenario())
    packed = lbl_cuda.pack(tasks)
    assert packed.start[0] == 0 and packed.stop[-1] == packed.center.size
    np.testing.assert_array_equal(packed.start[1:], packed.stop[:-1])
    for a, b in zip(packed.start, packed.stop):
        assert np.all(np.diff(packed.center[a:b]) >= 0)
    assert list(packed.has_pedestal) == [1, 0, 0] * 3


def test_cuda_path_with_the_numpy_twin_matches_cpu(monkeypatch):
    """Whole ``backend="cuda"`` path (selection threads, packing, row mapping)
    with the kernel replaced by its NumPy twin: every layer and species."""
    monkeypatch.setenv("MWIR_CUDA_SELECTION", "cpu")
    monkeypatch.setattr(lbl_cuda, "available", lambda: True)
    monkeypatch.setattr(lbl_cuda, "evaluate_groups", lbl_cuda.evaluate_groups_emulated)
    scenario = Scenario()
    path = build_scenario_path(scenario, build_profile(scenario))
    grid = make_grid(2140.0, 2142.0, 0.01)
    cpu_total, cpu_species = lbl.molecular_optical_depth(path.layers, grid, backend="cpu",
                                                         method="direct")
    gpu_total, gpu_species = lbl.molecular_optical_depth(path.layers, grid, backend="cuda",
                                                         n_threads=3)
    np.testing.assert_allclose(gpu_total, cpu_total, rtol=2e-6, atol=1e-12)
    for sp in cpu_species:
        np.testing.assert_allclose(gpu_species[sp], cpu_species[sp], rtol=2e-6, atol=1e-12)


def test_selection_switch_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("MWIR_CUDA_SELECTION", "fpga")
    with pytest.raises(ValueError, match="MWIR_CUDA_SELECTION"):
        lbl_cuda_dispatch._gpu_selection_enabled()
    monkeypatch.delenv("MWIR_CUDA_SELECTION")
    assert lbl_cuda_dispatch._gpu_selection_enabled()


def test_task_groups_fill_their_rows_and_leave_others_zero(monkeypatch):
    """Grouped evaluation (CPU selection of group k+1 while the GPU runs group k)
    places every task in its (species, layer) row, however the tasks are grouped."""
    grid = make_grid(2140.0, 2145.0, 0.05)
    tasks, expected = _tasks(grid, Scenario())
    single = lbl_cuda.evaluate_emulated(grid.wavenumber_cm1, lbl_cuda.pack(tasks))
    rows = [7, 0, 3, 8, 1, 5, 2, 6, 4]
    groups = [(lbl_cuda.pack([tasks[k] for k in (0, 4)]), [rows[0], rows[4]]),
              (lbl_cuda.concatenate([lbl_cuda.pack_task(*tasks[k]) for k in (1, 2, 3)]),
               rows[1:4]),
              (lbl_cuda.pack(tasks[5:]), rows[5:])]
    got = lbl_cuda.evaluate_groups_emulated(grid.wavenumber_cm1, groups, 11)
    np.testing.assert_array_equal(got[rows], single)
    assert np.all(got[9:] == 0.0)


def _selection_pairs(grid, scenario: Scenario, min_strength_fraction=0.0):
    """Every (layer, species) task of ``scenario`` as ``molecular_optical_depth`` builds it,
    with the GPU selection's groups (run in NumPy) and the CPU-selected packing of each."""
    path = build_scenario_path(scenario, build_profile(scenario))
    linelists = ensure_linelists()
    species = list(lbl.REQUIRED_SPECIES)
    row0 = {sp: k * len(path.layers) for k, sp in enumerate(species)}
    tasks, arguments = [], {}
    for row, layer in enumerate(path.layers):
        for sp in species:
            column = layer.column_cm2.get(sp, 0.0)
            if column > 0.0 and sp in linelists:
                p = layer.p_eff_mbar.get(sp, layer.p_mbar)
                tasks.append((row, sp, column, linelists[sp]))
                arguments[row, sp] = dict(
                    pressure_mbar=p, temperature_k=layer.t_eff_k.get(sp, layer.t_k),
                    partial_pressure_mbar=p * column / layer.air_column_cm2,
                    wing_cutoff_cm1=25.0, subtract_pedestal=sp == Species.H2O,
                    min_strength_fraction=min_strength_fraction, strength_reference=None,
                    column_cm2=column, min_peak_optical_depth=3e-6)
    by_row = {row0[sp] + row: (row, sp, column, ll) for row, sp, column, ll in tasks}
    for packed, rows in lbl_cuda_dispatch._gpu_selected_groups(np, tasks, lambda t: arguments[t[0], t[1]],
                                                 grid, row0):
        expected = []
        for r in rows:
            row, sp, column, linelist = by_row[r]
            a = arguments[row, sp]
            lines = lbl._select_lines(
                lbl._prepare(linelist), grid, a["pressure_mbar"], a["temperature_k"],
                a["partial_pressure_mbar"], 25.0, a["subtract_pedestal"], min_strength_fraction,
                None, column, 3e-6)
            expected.append(lbl_cuda.pack_task(lines, 25.0, column))
        yield packed, lbl_cuda.concatenate(expected)


@pytest.mark.parametrize("min_strength_fraction", [0.0, 1e-4])
def test_selection_in_numpy_equals_the_cpu_selection(monkeypatch, min_strength_fraction):
    """The GPU selection's arithmetic, run in NumPy, keeps the same lines with
    bit-identical packed parameters as ``_select_lines`` + ``pack_task``, also
    when a line list is split over several passes."""
    monkeypatch.setattr(lbl_cuda_dispatch, "_SELECT_PAIRS", 20_000)
    grid = make_grid(2140.0, 2145.0, 0.05)
    groups = 0
    for packed, expected in _selection_pairs(grid, Scenario(), min_strength_fraction):
        groups += 1
        for field in dataclasses.fields(lbl_cuda.Packed):
            np.testing.assert_array_equal(getattr(packed, field.name),
                                          getattr(expected, field.name), err_msg=field.name)
    assert groups > len(lbl.REQUIRED_SPECIES)


def test_cuda_path_with_numpy_selection_matches_cpu_selection(monkeypatch):
    """GPU selection (the default) through ``molecular_optical_depth`` (row mapping,
    grouping by line list), with NumPy for CuPy and the kernel's NumPy twin."""
    monkeypatch.setattr(lbl_cuda, "available", lambda: True)
    monkeypatch.setattr(lbl_cuda, "evaluate_groups", lbl_cuda.evaluate_groups_emulated)
    monkeypatch.setattr(lbl_cuda, "selection_module", lambda: np)
    scenario = Scenario()
    path = build_scenario_path(scenario, build_profile(scenario))
    grid = make_grid(2140.0, 2142.0, 0.01)
    monkeypatch.setenv("MWIR_CUDA_SELECTION", "cpu")
    cpu_total, cpu_species = lbl.molecular_optical_depth(path.layers, grid, backend="cuda")
    monkeypatch.delenv("MWIR_CUDA_SELECTION")
    gpu_total, gpu_species = lbl.molecular_optical_depth(path.layers, grid, backend="cuda")
    np.testing.assert_array_equal(gpu_total, cpu_total)
    for sp in cpu_species:
        np.testing.assert_array_equal(gpu_species[sp], cpu_species[sp])


def test_cuda_backend_is_refused_without_a_device(monkeypatch):
    monkeypatch.setattr(lbl_cuda, "available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        lbl.resolve_backend("cuda")
    monkeypatch.setenv("MWIR_BACKEND", "cpu")
    assert lbl.resolve_backend() == "cpu"
    with pytest.raises(ValueError):
        lbl.resolve_backend("opencl")


# =========================================================================
# On a GPU
# =========================================================================


@gpu
def test_kernel_matches_its_numpy_twin():
    grid = make_grid(2140.0, 2145.0, 0.002)
    tasks, _ = _tasks(grid, Scenario())
    packed = lbl_cuda.pack(tasks)
    gpu_tau = lbl_cuda.evaluate(grid.wavenumber_cm1, packed)
    emulated = lbl_cuda.evaluate_emulated(grid.wavenumber_cm1, packed)
    np.testing.assert_allclose(gpu_tau, emulated, rtol=2e-6, atol=1e-12 * np.abs(emulated).max())


@gpu
def test_molecular_optical_depth_cuda_matches_cpu():
    scenario = Scenario()
    grid = make_grid(2000.0, 2100.0, 0.01)
    path = build_scenario_path(scenario, build_profile(scenario))
    cpu_total, _ = lbl.molecular_optical_depth(path.layers, grid, backend="cpu", method="direct")
    gpu_total, _ = lbl.molecular_optical_depth(path.layers, grid, backend="cuda")
    np.testing.assert_allclose(gpu_total, cpu_total, rtol=2e-6, atol=1e-9)


@gpu
@pytest.mark.parametrize("aerosol", [AerosolModel.RURAL, AerosolModel.LOWTRAN_RURAL])
def test_device_resident_chunks_match_the_host_path(monkeypatch, aerosol):
    """Daytime look-down path (surface, solar and scattered terms): keeping the
    fine-grid arrays on the GPU changes nothing but rounding, and frees them."""
    import cupy
    from mwir_atmos import api

    scenario = Scenario(wavenumber_min_cm1=2500.0, wavenumber_max_cm1=2530.0,
                        spectral_sampling=SpectralSampling.RESOLVED, aerosol=aerosol,
                        sensor_alt_km=5.0, sensor_zenith_deg=150.0,
                        local_datetime=datetime(2026, 6, 21, 12, 0))
    device = run_scenario(scenario, backend="cuda")
    assert cupy.get_default_memory_pool().total_bytes() == 0
    monkeypatch.setattr(api, "_CUDA_DEVICE_RESIDENT", False)
    host = run_scenario(scenario, backend="cuda")
    assert device.radiance.solar_single_scatter.max() > 0.0
    assert device.radiance.surface_reflected_solar.max() > 0.0
    for name in ("total", "molecular", "aerosol", "rayleigh"):
        np.testing.assert_allclose(getattr(device.transmittance, name),
                                   getattr(host.transmittance, name), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(device.transmittance.cumulative, host.transmittance.cumulative,
                               rtol=1e-12, atol=1e-15)
    for sp, values in host.transmittance.by_species.items():
        np.testing.assert_allclose(device.transmittance.by_species[sp], values, rtol=1e-12, atol=1e-15)
    for name in ("total", "path_thermal", "surface_emitted", "surface_reflected_solar",
                 "solar_single_scatter", "direct_solar"):
        np.testing.assert_allclose(getattr(device.radiance, name), getattr(host.radiance, name),
                                   rtol=1e-10, atol=1e-20)
    assert all(isinstance(v, np.ndarray) for v in (device.transmittance.total,
                                                  device.radiance.total))


@gpu
def test_resolved_run_on_cuda_matches_cpu():
    scenario = Scenario(wavenumber_min_cm1=2500.0, wavenumber_max_cm1=2530.0,
                        spectral_sampling=SpectralSampling.RESOLVED)
    cpu = run_scenario(scenario, backend="cpu")
    gpu_result = run_scenario(scenario, backend="cuda")
    # The CPU uses multigrid far wings (<= 1e-4 of each wing); the GPU sums directly.
    np.testing.assert_allclose(gpu_result.transmittance.total, cpu.transmittance.total, atol=1e-4)
    np.testing.assert_allclose(gpu_result.radiance.total, cpu.radiance.total, rtol=5e-4)


@gpu
def test_gpu_selection_matches_the_numpy_selection():
    """CuPy selection against the same arithmetic in NumPy: only exp, expm1,
    pow and erfcx may round differently, so a line right at a threshold may
    flip; everything kept by both has the same parameters to rounding."""
    import cupy

    grid = make_grid(2000.0, 2100.0, 0.01)
    kept = flipped = 0
    scenario = Scenario()
    path = build_scenario_path(scenario, build_profile(scenario))
    linelists = ensure_linelists()
    for sp in (sp for sp in lbl.REQUIRED_SPECIES if sp in linelists):
        prepared = lbl._prepare(linelists[sp])
        specs = []
        for layer in path.layers:
            column = layer.column_cm2.get(sp, 0.0)
            if column <= 0.0:
                continue
            p = layer.p_eff_mbar.get(sp, layer.p_mbar)
            specs.append(lbl_cuda_dispatch._selection_task(
                prepared, grid.wavenumber_cm1[0], grid.wavenumber_cm1[-1], p,
                layer.t_eff_k.get(sp, layer.t_k), p * column / layer.air_column_cm2, 25.0,
                sp == Species.H2O, 0.0, None, column, 3e-6))
        lo, hi = grid.wavenumber_cm1[0], grid.wavenumber_cm1[-1]
        host = lbl_cuda_dispatch._select_packed(np, prepared, specs, lo, hi)
        device = lbl_cuda_dispatch._select_packed(cupy, prepared, specs, lo, hi)
        d = {f.name: cupy.asnumpy(cupy.asarray(getattr(device, f.name)))
             for f in dataclasses.fields(lbl_cuda.Packed)}
        kept += host.center.size
        if np.array_equal(d["stop"], host.stop):
            np.testing.assert_array_equal(d["center"], host.center)
            for name in ("inv_alpha", "y", "amp", "pedestal"):
                np.testing.assert_allclose(d[name], getattr(host, name), rtol=3e-7, atol=0.0,
                                           err_msg=f"{sp}: {name}")
        else:
            flipped += int(np.abs(d["stop"] - d["start"] - (host.stop - host.start)).sum())
    assert flipped <= 1e-5 * kept


@gpu
def test_gpu_selection_path_matches_cpu_selection_path(monkeypatch):
    scenario = Scenario()
    grid = make_grid(2000.0, 2100.0, 0.01)
    path = build_scenario_path(scenario, build_profile(scenario))
    monkeypatch.setenv("MWIR_CUDA_SELECTION", "cpu")
    cpu_selected, _ = lbl.molecular_optical_depth(path.layers, grid, backend="cuda")
    monkeypatch.setenv("MWIR_CUDA_SELECTION", "gpu")
    gpu_selected, _ = lbl.molecular_optical_depth(path.layers, grid, backend="cuda")
    # A line flipped at the 3e-6 peak-optical-depth threshold moves tau by < 3e-6.
    np.testing.assert_allclose(gpu_selected, cpu_selected, rtol=1e-6, atol=3e-6)
