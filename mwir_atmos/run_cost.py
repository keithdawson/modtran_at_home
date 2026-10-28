"""Up-front estimate of a scenario's size, memory and runtime.

Resolution is never capped or silently clamped: narrow-window, fine-resolution
work (e.g. 0.001 cm^-1 over a few cm^-1 to resolve an emission feature) is a
primary use case and stays fast. The trade-off is shown to the user instead.
Before a run, the CLI and the GUI both call :func:`estimate_run_cost` and ask
for confirmation when :func:`needs_confirmation` says the run is large
(``mwir --yes`` skips the question).

The estimate builds the real path geometry (so the layer count is right) but
never allocates a spectral array or loads line data.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from . import api, resolved
from .optics.lbl import evaluations_per_line, resolve_backend, resolve_thread_count
from .types import (
    REQUIRED_SPECIES,
    PathGeometry,
    RadianceSolver,
    Scenario,
    SpectralGrid,
    SpectralSampling,
)
from .validation import validate_scenario

__all__ = [
    "CONFIRM_BYTES_THRESHOLD",
    "CONFIRM_SECONDS_THRESHOLD",
    "CostEstimate",
    "estimate_run_cost",
    "format_bytes",
    "format_duration",
    "needs_confirmation",
]

#: Above roughly 1 GB of layer arrays or a few minutes, a run asks for
#: confirmation. Guidelines, not limits.
CONFIRM_BYTES_THRESHOLD = 1 * 1024**3
CONFIRM_SECONDS_THRESHOLD = 180.0

# -------------------------------------------------------------------------
# Calibration
#
# Two structural drivers dominate mwir_atmos.optics.lbl's cost:
#   - grid size: the per-layer optical-depth arrays are (n_layers, grid.n);
#   - the per-line Voigt window: every surviving line is evaluated over
#     2*line_wing_cutoff_cm1/step + 1 grid points, independent of how narrow
#     the requested window is. At fine resolution with the default 25 cm^-1
#     cutoff this window alone can be tens of thousands of points even for a
#     2 cm^-1 request.
# Whichever is larger sets the estimate. This is an ORDER-OF-MAGNITUDE
# guide: congested regions (a CO2 Q-branch) and sparse ones at identical
# settings differ by up to ~3x, because the number of lines in the window is
# not knowable without loading the line data.
# -------------------------------------------------------------------------

#: Point sampling, calibrated 2026-08-02: a 2501-point / 60-layer
#: 1800-4300 cm^-1 @ 1 cm^-1 run measured ~8 s, and 1800-4300 cm^-1 @
#: 0.001 cm^-1 ~2.2 h. These predate the September 2026 engine rewrite and
#: err slow (the default point run now takes ~10 s).
_COST_FIXED_OVERHEAD_S = 4.0
_COST_RATE_ELEMENTS_PER_S = 19_000.0
#: Resolved sampling, per worker process: fine points x layers per second for
#: the whole pipeline (line-by-line, continuum, transmittance, single-scatter
#: radiance). Measured at 0.23e6 on an 8-core desktop
#: (docs/validation/scaling_benchmark.json); rounded down for slower cores.
_COST_RESOLVED_RATE_PER_WORKER = 150_000.0
#: Parallel speedup ~ workers ** this (measured 5.4x at 8 workers, 6.7x at 16
#: SMT threads on 8 cores); errs slow on machines that scale better.
_COST_SPEEDUP_EXPONENT = 0.7
#: CUDA backend, resolved sampling: fixed seconds + fine points x layers per
#: second, fitted to the RTX 4090 with GPU line selection (2026-10, commit
#: 5f7b199): full default band, 41.2 M elements, 0.548 s; 2000-2200 cm^-1,
#: 3.28 M elements, 0.18 s. The first run in a process adds CUDA context
#: set-up and kernel compilation (~1 s), not counted.
_COST_CUDA_FIXED_S = 0.15
_COST_CUDA_RATE_ELEMENTS_PER_S = 1.0e8
#: Point sampling on the CUDA backend (default point run: ~0.3 s).
_COST_CUDA_POINT_FIXED_S = 0.3
#: float64 (n_layers[+1], grid.n) arrays materialised per run: tau_molecular,
#: one tau_by_species per REQUIRED_SPECIES, tau_aerosol_abs, tau_aerosol_sca,
#: tau_rayleigh, aerosol_asymmetry, and TransmittanceResult.cumulative.
_COST_N_LAYER_ARRAYS = 1 + len(REQUIRED_SPECIES) + 5
_COST_BYTES_PER_ELEMENT = 8


@dataclasses.dataclass
class CostEstimate:
    n_points: int
    n_layers: int
    wing_window_points: float
    est_bytes: int
    est_seconds: float


@dataclasses.dataclass
class _SpectralSize:
    """Spectral point counts that drive cost."""

    step_cm1: float          # step the line-by-line engine evaluates on
    evaluated_points: int    # points the line-by-line engine fills
    resident_points: int     # points held in layer arrays at once
    solver_points: int       # points DISORT / Monte Carlo solve
    plan: Optional[resolved.FineGridPlan] = None   # resolved sampling only


def estimate_run_cost(scenario: Scenario, n_threads: Optional[int] = None,
                      backend: Optional[str] = None) -> CostEstimate:
    """Points x layers, estimated memory, and estimated runtime for ``scenario``.

    ``backend`` as for :func:`~mwir_atmos.api.run_scenario` (default: ``auto``).

    Raises whatever :func:`~mwir_atmos.api.run_scenario` would raise for an
    invalid or geometrically impossible scenario (e.g. an unreachable
    source), before any spectral work happens.
    """
    validate_scenario(scenario)
    n_points = int(np.floor((scenario.wavenumber_max_cm1 - scenario.wavenumber_min_cm1)
                            / scenario.resolution_cm1)) + 1
    geometry = api.build_solver_geometry(scenario, api.build_profile(scenario))
    n_layers = geometry.optics_path.n_layers

    size = _spectral_size(scenario, n_points, n_layers, geometry.path)
    wing_pts = evaluations_per_line(size.step_cm1, scenario.line_wing_cutoff_cm1)
    elements = max(float(size.evaluated_points), wing_pts) * n_layers

    resident_points = size.resident_points
    speedup = 1.0
    if resolve_backend(backend) == "cuda":
        # One process; DISORT and Monte Carlo below run on its CPU, unparallelised.
        fixed = (_COST_CUDA_FIXED_S if scenario.spectral_sampling == SpectralSampling.RESOLVED
                 else _COST_CUDA_POINT_FIXED_S)
        gpus = 1
        if (scenario.spectral_sampling == SpectralSampling.RESOLVED
                and elements >= api._MULTI_GPU_MIN_ELEMENTS):
            from . import multi_gpu
            gpus = len(multi_gpu.cuda_devices() or (0,))   # MWIR_CUDA_DEVICES; even split
        est_seconds = fixed + elements / (_COST_CUDA_RATE_ELEMENTS_PER_S * gpus)
        if scenario.spectral_sampling == SpectralSampling.RESOLVED:
            resident_points = min(size.resident_points,
                                  max(1, api._CUDA_CHUNK_ELEMENTS // max(n_layers, 1)))
    elif scenario.spectral_sampling == SpectralSampling.RESOLVED:
        speedup, resident_points = _resolved_parallelism(size, elements, n_threads)
        est_seconds = _COST_FIXED_OVERHEAD_S + elements / (_COST_RESOLVED_RATE_PER_WORKER * speedup)
    else:
        est_seconds = _COST_FIXED_OVERHEAD_S + elements / _COST_RATE_ELEMENTS_PER_S
    est_bytes = int(resident_points * n_layers * _COST_BYTES_PER_ELEMENT * _COST_N_LAYER_ARRAYS)

    if scenario.radiance_solver == RadianceSolver.DISORT:
        # Additional full-column arrays and sequential angular solves. Rough:
        # the actual Fourier work depends on illumination.
        est_bytes *= 2
        sources = 2 + int(scenario.include_solar) + int(scenario.include_lunar)
        est_seconds += (size.solver_points * .005 * sources
                        * (scenario.disort_streams / 16.)**4 / speedup)
    if scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        est_bytes += min(scenario.spherical_photons, 4096) * n_layers * 200
        est_seconds += size.solver_points * (scenario.spherical_photons / 4096.) * 2. / speedup

    return CostEstimate(
        n_points=n_points,
        n_layers=n_layers,
        wing_window_points=wing_pts,
        est_bytes=est_bytes,
        est_seconds=est_seconds,
    )


def _spectral_size(scenario: Scenario, n_points: int, n_layers: int,
                   path: PathGeometry) -> _SpectralSize:
    if scenario.spectral_sampling != SpectralSampling.RESOLVED:
        return _SpectralSize(scenario.resolution_cm1, n_points, n_points, n_points)

    fine = scenario.fine_step_cm1
    if fine is None:
        fine = resolved.automatic_fine_step(path.layers, scenario.wavenumber_min_cm1)
    # A one-point grid is enough to size the plan; it never allocates the band.
    plan = resolved.plan_fine_grid(
        SpectralGrid(wavenumber_cm1=np.array([scenario.wavenumber_min_cm1]),
                     resolution_cm1=scenario.resolution_cm1),
        fine_step_cm1=fine, function=scenario.instrument_function,
        fwhm_cm1=scenario.instrument_fwhm_cm1)
    evaluated = (n_points - 1) * plan.ratio + 2 * plan.support + 1
    resident = min(evaluated, max(resolved._MIN_CHUNK_FINE_POINTS,
                                  resolved._CHUNK_ELEMENTS // max(n_layers, 1)))
    solver = n_points * (len(resolved._G_EDGES) - 1)
    return _SpectralSize(plan.step_cm1, evaluated, resident, solver, plan)


def _resolved_parallelism(size: _SpectralSize, elements: float,
                          n_threads: Optional[int]) -> tuple[float, int]:
    """``(speedup, resident points)`` for resolved chunks, mirroring
    :func:`mwir_atmos.api._run_resolved`'s choice of worker processes."""
    workers = resolve_thread_count(n_threads)
    if not (workers > 1 and api._parallel_mode() == "processes"
            and elements >= api._PROCESS_MIN_ELEMENTS):
        return 1.0, size.resident_points
    chunk = max(resolved._MIN_CHUNK_FINE_POINTS,
                resolved._MIN_CHUNK_OVER_SUPPORT * size.plan.support)
    workers = max(1, min(workers, int(size.evaluated_points // chunk)))
    resident = min(size.resident_points,
                   max(chunk, size.evaluated_points // (api._CHUNKS_PER_WORKER * workers)))
    # One chunk per worker is in memory at once.
    return workers ** _COST_SPEEDUP_EXPONENT, resident * workers


def needs_confirmation(cost: CostEstimate) -> bool:
    """Whether ``cost`` crosses the confirmation thresholds."""
    return (cost.est_bytes >= CONFIRM_BYTES_THRESHOLD
            or cost.est_seconds >= CONFIRM_SECONDS_THRESHOLD)


def format_bytes(n_bytes: float) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.2f} h"
