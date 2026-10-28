"""DISORT and Monte Carlo spectral points split over worker processes.

Each piece is solved by the ordinary solver, Monte Carlo seeds offset to the
piece's first point, so the joined result equals one serial call exactly.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from mwir_atmos import api, solver_pool
from mwir_atmos.types import RadianceSolver, Scenario, SpectralSampling


@pytest.fixture(autouse=True)
def _small_pool(monkeypatch):
    monkeypatch.setattr(solver_pool, "MIN_POINTS", 4)
    yield
    solver_pool.shutdown()


@pytest.mark.parametrize("scenario", [
    Scenario(wavenumber_min_cm1=2500.0, wavenumber_max_cm1=2520.0, resolution_cm1=1.0,
             radiance_solver=RadianceSolver.DISORT, disort_streams=8,
             spectral_sampling=SpectralSampling.POINT, sensor_alt_km=5.0,
             sensor_zenith_deg=150.0, local_datetime=datetime(2026, 6, 21, 12, 0)),
    Scenario(wavenumber_min_cm1=2500.0, wavenumber_max_cm1=2510.0, resolution_cm1=1.0,
             radiance_solver=RadianceSolver.SPHERICAL_MC, spherical_photons=128,
             spectral_sampling=SpectralSampling.POINT, spherical_seed=7),
], ids=["disort-daytime", "monte-carlo"])
def test_split_solver_equals_one_serial_call(monkeypatch, scenario):
    calls = []
    original = solver_pool.solve

    def counted(*args, **kwargs):
        calls.append(kwargs["workers"])
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_pool, "solve", counted)
    split = api.run_scenario(scenario, backend="cpu", n_threads=3, include_turbulence=False)
    serial = api.run_scenario(scenario, backend="cpu", n_threads=1, include_turbulence=False)
    assert calls == [3]
    for name in ("total", "path_thermal", "surface_emitted", "solar_multiple_scatter"):
        a, b = getattr(split.radiance, name), getattr(serial.radiance, name)
        if b is not None:
            np.testing.assert_array_equal(a, b, err_msg=name)
    for k, v in serial.radiance.sampling_standard_error.items():
        np.testing.assert_array_equal(split.radiance.sampling_standard_error[k], v)


def test_threads_mode_and_one_thread_do_not_split(monkeypatch):
    scenario = Scenario(radiance_solver=RadianceSolver.DISORT)
    assert solver_pool.use_pool(scenario, 1000, 8)
    assert not solver_pool.use_pool(scenario, 1000, 1)
    assert not solver_pool.use_pool(scenario, 2, 8)
    assert not solver_pool.use_pool(Scenario(), 1000, 8)          # single scatter
    monkeypatch.setenv("MWIR_PARALLEL", "threads")
    assert not solver_pool.use_pool(scenario, 1000, 8)
