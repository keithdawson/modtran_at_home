"""DISORT and spherical Monte Carlo radiance over CPU worker processes.

Both solvers handle one spectral point at a time in Python (DISORT: ~12 ms
per point and source on one core; the full default band binned is ~15 min
on one core). On the CPU backend, resolved runs already spread whole chunks
over worker processes. On the CUDA backend the optics come from the GPU in
this process, and point sampling never used processes, so the solver ran on
one core. :func:`solve` splits the spectral points of one solver call over a
persistent pool of spawned worker processes (spawned: CUDA may already be
initialised here), each solving its contiguous piece with the ordinary
solver. Points are independent and Monte Carlo seeds each point
``spherical_seed + index``, so a piece starting at ``first`` gets
``spherical_seed + first``: results equal one serial call exactly. The
sensor response (an integral over the spectrum) is applied once, to the
joined result.

Used when the caller may use more than one CPU (``n_threads``/
``MWIR_THREADS``; resolved-chunk workers run with one, so pools never nest),
``MWIR_PARALLEL`` is not ``threads``, and the call has at least
:data:`MIN_POINTS` points.
"""
from __future__ import annotations

import atexit
import dataclasses
import threading
import warnings
from typing import Optional

import numpy as np

from .types import RadianceSolver, SpectralGrid

__all__ = ["MIN_POINTS", "shutdown", "solve", "use_pool"]

#: Smallest solver call split over processes: ~64 points x 2+ sources of
#: DISORT is ~1.5 s on one core; the pool starts once per process (~2-3 s).
MIN_POINTS = 64
#: Pieces per worker, so uneven point costs (Monte Carlo event counts) balance.
_PIECES_PER_WORKER = 4

_POOL = None
_POOL_SIZE = 0
_POOL_LOCK = threading.Lock()


def use_pool(scenario, n_points: int, workers: int) -> bool:
    from .api import _parallel_mode
    return (scenario.radiance_solver in (RadianceSolver.DISORT, RadianceSolver.SPHERICAL_MC)
            and workers > 1 and n_points >= MIN_POINTS and _parallel_mode() == "processes")


def _slice(value, a: int, b: int, n: int, grid: SpectralGrid):
    """``value`` restricted to spectral points ``a:b`` (arrays whose last axis is ``n``)."""
    if isinstance(value, SpectralGrid):
        return grid
    if isinstance(value, np.ndarray) and value.ndim and value.shape[-1] == n:
        return value[..., a:b]
    if isinstance(value, dict):
        return {k: _slice(v, a, b, n, grid) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _slice(getattr(value, f.name), a, b, n, grid)
                                             for f in dataclasses.fields(value) if f.init})
    return value


def _piece(scenario, grid, profile, path, optics, column, column_optics, transmittance,
           sun, moon, a: int, b: int):
    n = grid.n
    piece_grid = SpectralGrid(wavenumber_cm1=grid.wavenumber_cm1[a:b],
                              resolution_cm1=grid.resolution_cm1)
    piece_scenario = dataclasses.replace(scenario, spherical_seed=scenario.spherical_seed + a,
                                         sensor_response_file=None)
    return (piece_scenario, piece_grid, profile, path,
            _slice(optics, a, b, n, piece_grid), column,
            None if column_optics is None else _slice(column_optics, a, b, n, piece_grid),
            _slice(transmittance, a, b, n, piece_grid), sun, moon)


def _solve_piece(args):
    from .api import _solve_radiance
    return _solve_radiance(*args)


def _pool(workers: int):
    global _POOL, _POOL_SIZE
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from .api import _SINGLE_THREAD_ENV
    import os

    with _POOL_LOCK:
        if _POOL is not None and _POOL_SIZE == workers:
            return _POOL
        shutdown()
        # Workers must not each start a BLAS/OpenMP pool the size of the machine.
        saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
        os.environ.update({k: "1" for k in _SINGLE_THREAD_ENV})
        try:
            _POOL = ProcessPoolExecutor(max_workers=workers,
                                        mp_context=multiprocessing.get_context("spawn"))
            # Start every worker now, while the environment says one thread.
            list(_POOL.map(int, range(workers)))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _POOL_SIZE = workers
        return _POOL


def shutdown() -> None:
    """Stop the solver worker processes (they also stop when this process exits)."""
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL, _POOL_SIZE = None, 0


atexit.register(shutdown)


def solve(scenario, grid, profile, path, optics, column, column_optics, transmittance,
          sun, moon, workers: int):
    """``api._solve_radiance`` with its spectral points split over ``workers`` processes.

    Returns None (after a warning) when the processes cannot run; the caller
    then solves in this process.
    """
    from concurrent.futures.process import BrokenProcessPool
    from . import resolved

    n = grid.n
    count = max(1, min(n, workers * _PIECES_PER_WORKER))
    edges = np.linspace(0, n, count + 1).round().astype(int)
    pieces = [_piece(scenario, grid, profile, path, optics, column, column_optics,
                     transmittance, sun, moon, int(a), int(b))
              for a, b in zip(edges[:-1], edges[1:]) if b > a]
    try:
        parts = list(_pool(workers).map(_solve_piece, pieces))
    except (BrokenProcessPool, OSError) as exc:
        shutdown()
        warnings.warn(f"radiance solver: worker processes unavailable ({exc!r}); "
                      "solving in this process", RuntimeWarning, stacklevel=3)
        return None
    result = resolved.concatenate_radiance(parts, grid)
    if scenario.sensor_response_file and result.stokes:
        from .radiometry.sensor import apply_sensor_response, load_sensor_response
        result.sensor_signal = apply_sensor_response(
            grid, result.stokes["total"], load_sensor_response(scenario.sensor_response_file))
    return result
