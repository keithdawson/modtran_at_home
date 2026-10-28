"""Multi-GPU resolved runs (``MWIR_CUDA_DEVICES``), checked without a GPU.

The pool's workers only pin ``MWIR_CUDA_DEVICE`` and run the ordinary chunk
function, so real spawned workers are exercised here on the CPU backend; the
routing of a CUDA run is checked with the pool replaced by a CPU stand-in.
GPU runs are covered by ``scripts/validation/multi_gpu_check.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos import api, multi_gpu, resolved
from mwir_atmos.optics import lbl_cuda
from mwir_atmos.types import Scenario, SpectralSampling


@pytest.mark.parametrize("value, expected", [
    ("", None), ("all", (0, 1, 2, 3, 4)), ("2", (2,)), ("0, 3,1", (0, 3, 1))])
def test_device_list(monkeypatch, value, expected):
    monkeypatch.setenv("MWIR_CUDA_DEVICES", value)
    assert multi_gpu.cuda_devices(device_count=5) == expected


@pytest.mark.parametrize("value", ["0,0", "5", "-1", "gpu0", ","])
def test_device_list_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv("MWIR_CUDA_DEVICES", value)
    with pytest.raises(ValueError, match="MWIR_CUDA_DEVICES"):
        multi_gpu.cuda_devices(device_count=5)


def _pass_and_jobs(scenario, min_chunks):
    profile = api.build_profile(scenario)
    geometry = api.build_solver_geometry(scenario, profile)
    spectral_pass = api._SpectralPass(scenario, profile, geometry, None, None, True, 1,
                                      backend="cpu")
    grid = api.make_grid(scenario.wavenumber_min_cm1, scenario.wavenumber_max_cm1,
                         scenario.resolution_cm1)
    step = resolved.automatic_fine_step(spectral_pass.path.layers, grid.wavenumber_cm1[0])
    plan = resolved.plan_fine_grid(grid, fine_step_cm1=step,
                                   function=scenario.instrument_function,
                                   fwhm_cm1=scenario.instrument_fwhm_cm1)
    jobs = [(plan, first, stop, fine, grid.wavenumber_cm1[first:stop], grid.resolution_cm1, False)
            for first, stop, fine in resolved.chunks(plan, grid.n, len(spectral_pass.path.layers),
                                                     min_chunks=min_chunks)]
    return spectral_pass, jobs


def test_worker_pool_pins_one_device_per_worker_and_keeps_order():
    scenario = Scenario(wavenumber_min_cm1=2500.0, wavenumber_max_cm1=2540.0,
                        include_solar=False, include_lunar=False)
    spectral_pass, jobs = _pass_and_jobs(scenario, min_chunks=6)
    assert len(jobs) >= 3
    try:
        parts, devices = multi_gpu.run_chunks(spectral_pass, jobs, [3, 1], with_devices=True)
        again = multi_gpu.run_chunks(spectral_pass, jobs[:2], [3, 1])   # the same pool
        assert multi_gpu._POOL_DEVICES == (3, 1)
    finally:
        multi_gpu.shutdown()
    assert set(devices) <= {1, 3}
    for (t, r), job, repeat in zip(parts, jobs, again + [None] * len(jobs)):
        t_serial, r_serial = api._resolved_chunk(spectral_pass, *job)
        np.testing.assert_array_equal(t.total, t_serial.total)
        np.testing.assert_array_equal(r.total, r_serial.total)
        if repeat is not None:
            np.testing.assert_array_equal(repeat[0].total, t_serial.total)


def test_cuda_run_fans_out_over_the_listed_devices(monkeypatch):
    """A resolved CUDA run with two listed GPUs splits into several chunks (three
    per GPU where the band allows; a 40 cm^-1 band allows three) and hands them
    to the pool; here the pool is a CPU stand-in."""
    import copy
    monkeypatch.setattr(lbl_cuda, "available", lambda: True)
    monkeypatch.setattr(lbl_cuda, "release_memory", lambda: None)
    monkeypatch.setattr(api, "_CUDA_DEVICE_RESIDENT", False)
    monkeypatch.setattr(api, "_MULTI_GPU_MIN_ELEMENTS", 1)
    monkeypatch.setenv("MWIR_CUDA_DEVICES", "0,1")
    monkeypatch.setattr(multi_gpu, "cuda_devices", lambda: (0, 1))
    calls = []

    def stand_in(spectral_pass, jobs, devices):
        calls.append((len(jobs), tuple(devices)))
        cpu_pass = copy.copy(spectral_pass)
        cpu_pass.backend = "cpu"
        return [api._resolved_chunk(cpu_pass, *job) for job in jobs]

    monkeypatch.setattr(multi_gpu, "run_chunks", stand_in)
    scenario = Scenario(wavenumber_min_cm1=2500.0, wavenumber_max_cm1=2540.0,
                        spectral_sampling=SpectralSampling.RESOLVED,
                        include_solar=False, include_lunar=False)
    fanned = api.run_scenario(scenario, backend="cuda", include_turbulence=False)
    assert calls and calls[0][0] >= 3 and calls[0][1] == (0, 1)
    reference = api.run_scenario(scenario, backend="cpu", include_turbulence=False, n_threads=1)
    np.testing.assert_allclose(fanned.transmittance.total, reference.transmittance.total,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(fanned.radiance.total, reference.radiance.total, rtol=1e-10)


def test_cost_estimate_divides_large_runs_over_the_gpus(monkeypatch):
    from mwir_atmos import run_cost
    monkeypatch.setattr(lbl_cuda, "available", lambda: True)
    monkeypatch.setattr(lbl_cuda, "device_count", lambda: 4)
    scenario = Scenario(include_solar=False, include_lunar=False)
    one = run_cost.estimate_run_cost(scenario, backend="cuda").est_seconds
    monkeypatch.setenv("MWIR_CUDA_DEVICES", "all")
    four = run_cost.estimate_run_cost(scenario, backend="cuda").est_seconds
    assert four < one / 2
