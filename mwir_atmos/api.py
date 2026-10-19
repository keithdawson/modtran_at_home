"""Top-level orchestration: a :class:`Scenario` in, a :class:`ScenarioResult` out.

The single entry point the CLI, the web GUI and :mod:`mwir_atmos.setups` all
call, so they cannot drift apart in behaviour.

Pipeline (:func:`run_scenario`):
    model atmosphere + surface overrides  (:func:`build_profile`)
    -> path geometry for the radiance solver  (:func:`build_solver_geometry`)
    -> layer optics and transmittance on a spectral grid  (:class:`_SpectralPass`)
    -> radiance  (:func:`_solve_radiance`)
    -> turbulence diagnostics

Point sampling runs one spectral pass on the output grid. Resolved sampling
(:func:`_run_resolved`) runs it on fine-grid chunks, in worker processes or
on the GPU, and reduces each chunk through the instrument function.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .atmosphere.humidity import apply_surface_conditions
from .atmosphere.profiles import get_profile
from .ephemeris.moon import lunar_position
from .ephemeris.sun import solar_position
from .geometry.path import build_path
from .optics.lbl import resolve_backend, resolve_thread_count
from .optics.transmittance import compute_layer_optics, path_transmittance
from .radiometry.radiance import compute_radiance
from .spectral import make_grid
from .turbulence.metrics import compute_turbulence
from .validation import validate_scenario
from .types import (
    AtmosphereProfile,
    LayerOptics,
    PathGeometry,
    RadianceResult,
    Scenario,
    ScenarioResult,
    RadianceSolver,
    SpectralGrid,
    SpectralSampling,
    TurbulenceModel,
)

__all__ = ["SolverGeometry", "build_profile", "build_scenario_path", "build_solver_geometry",
           "run_scenario"]


def build_profile(scenario: Scenario) -> AtmosphereProfile:
    """Model atmosphere with the scenario's surface overrides applied."""
    profile = get_profile(scenario.atmosphere)
    needs_override = (
        scenario.ground_temp_k is not None
        or scenario.ground_rh_pct is not None
        or scenario.co2_ppmv is not None
        or scenario.surface_altitude_km != 0.0
    )
    if not needs_override:
        return profile
    return apply_surface_conditions(
        profile,
        ground_temp_k=scenario.ground_temp_k,
        ground_rh_pct=scenario.ground_rh_pct,
        co2_ppmv=scenario.co2_ppmv,
        surface_altitude_km=scenario.surface_altitude_km,
    )


@dataclass
class SolverGeometry:
    """Where a scenario's physics is evaluated.

    ``path`` is always the sensor's sightline. The single-scatter solver
    evaluates optics directly on it. DISORT and the spherical Monte Carlo
    solver evaluate optics on a separate vertical ``column`` and map them
    onto the sightline: sightline layer ``j`` is ``factors[j]`` times column
    layer ``indices[j]``.
    """

    path: PathGeometry
    column: Optional[PathGeometry] = None
    indices: Optional[Sequence[int]] = None
    factors: Optional[Sequence[float]] = None

    @property
    def optics_path(self) -> PathGeometry:
        """The geometry layer optics are computed on (column, else sightline)."""
        return self.column if self.column is not None else self.path


def build_solver_geometry(scenario: Scenario, profile: AtmosphereProfile) -> SolverGeometry:
    """Sightline (and, for multiple scattering, vertical column) for ``scenario``."""
    if scenario.radiance_solver == RadianceSolver.DISORT:
        from .geometry.plane_parallel import geometry
    elif scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        from .geometry.spherical_shells import geometry
    else:
        return SolverGeometry(path=build_path(
            profile,
            sensor_alt_km=scenario.sensor_alt_km,
            zenith_deg=scenario.sensor_zenith_deg,
            azimuth_deg=scenario.sensor_azimuth_deg,
            source_alt_km=scenario.source_alt_km,
            source_range_km=scenario.source_range_km,
        ))
    column, path, indices, factors = geometry(scenario, profile)
    return SolverGeometry(path=path, column=column, indices=indices, factors=factors)


def build_scenario_path(scenario: Scenario, profile: AtmosphereProfile) -> PathGeometry:
    """Slant path implied by the scenario's sensor and source placement."""
    return build_solver_geometry(scenario, profile).path


def run_scenario(
    scenario: Scenario,
    *,
    include_radiance: bool = True,
    include_turbulence: bool = True,
    grid: Optional[SpectralGrid] = None,
    n_threads: Optional[int] = None,
    backend: Optional[str] = None,
) -> ScenarioResult:
    """Run a full scenario.

    Transmittance is always computed. ``include_radiance=False`` leaves
    ``result.radiance`` (and the sun/moon positions) as ``None``;
    ``include_turbulence`` adds turbulence diagnostics when the scenario
    selects a Cn^2 model.

    ``grid`` overrides the output grid (uniform spacing). ``n_threads`` sets
    the line-by-line worker threads (default: ``MWIR_THREADS`` or every CPU);
    results do not depend on it. ``backend`` selects the line-by-line engine,
    ``"cpu"`` or ``"cuda"`` (default: ``MWIR_BACKEND``, else ``"cpu"``; see
    :func:`mwir_atmos.optics.lbl.resolve_backend`). With
    ``spectral_sampling="resolved"`` the physics runs on a fine internal grid
    and is convolved with the instrument function onto the output grid -- see
    :mod:`mwir_atmos.resolved`.
    """
    validate_scenario(scenario)
    backend = resolve_backend(backend)
    if grid is None:
        grid = make_grid(
            scenario.wavenumber_min_cm1,
            scenario.wavenumber_max_cm1,
            scenario.resolution_cm1,
        )

    profile = build_profile(scenario)
    geometry = build_solver_geometry(scenario, profile)
    sun = moon = None
    if include_radiance:
        sun = solar_position(scenario)
        moon = lunar_position(scenario, sun)

    spectral_pass = _SpectralPass(scenario, profile, geometry, sun, moon,
                                  include_radiance, n_threads, backend=backend)
    fine_step = None
    try:
        if scenario.spectral_sampling == SpectralSampling.RESOLVED:
            transmittance, radiance, fine_step = _run_resolved(spectral_pass, grid)
        else:
            _, _, transmittance, radiance = spectral_pass.run(grid)
    finally:
        if backend == "cuda":
            from .optics import lbl_cuda
            lbl_cuda.release_memory()   # the GPU may be shared: keep nothing cached

    result = ScenarioResult(
        scenario=scenario,
        grid=grid,
        profile=profile,
        path=geometry.path,
        transmittance=transmittance,
        radiance=radiance,
        sun=sun,
        moon=moon,
        fine_step_cm1=fine_step,
    )
    if include_turbulence and scenario.turbulence != TurbulenceModel.NONE:
        result.turbulence = compute_turbulence(
            scenario=scenario, path=result.path, profile=profile)
    return result


# =========================================================================
# One spectral pass: layer optics -> transmittance -> radiance on one grid
# =========================================================================


class _SpectralPass:
    """Layer optics, transmittance and radiance for one spectral grid.

    Point sampling runs this once on the output grid; resolved sampling runs
    it on each fine-grid chunk (possibly in worker processes, so it must
    stay picklable).
    """

    def __init__(self, scenario: Scenario, profile: AtmosphereProfile, geometry: SolverGeometry,
                 sun, moon, include_radiance: bool, n_threads: Optional[int],
                 strength_references=None, backend: str = "cpu"):
        self.scenario = scenario
        self.profile = profile
        self.path = geometry.path
        self.column = geometry.column
        self.indices = geometry.indices
        self.factors = geometry.factors
        self.sun = sun
        self.moon = moon
        self.include_radiance = include_radiance
        self.n_threads = n_threads
        self.strength_references = strength_references
        self.backend = backend
        #: Resolved chunks run on the GPU with device arrays (see _device_resident).
        self.device = False

    @property
    def optics_layers(self):
        return (self.column if self.column is not None else self.path).layers

    def optics(self, grid: SpectralGrid):
        """``(sightline optics, column optics or None, transmittance)`` on ``grid``."""
        scenario, column = self.scenario, self.column
        optics: LayerOptics = compute_layer_optics(
            column if column is not None else self.path,
            grid,
            aerosol=scenario.aerosol,
            visibility_km=scenario.visibility_km,
            wind_speed_m_s=scenario.wind_speed_m_s,
            aerosol_winter=scenario.atmosphere.value.endswith("winter"),
            wing_cutoff_cm1=scenario.line_wing_cutoff_cm1,
            min_strength_fraction=scenario.line_strength_cutoff,
            min_peak_optical_depth=scenario.line_optical_depth_cutoff,
            n_threads=self.n_threads,
            strength_references=self.strength_references,
            backend=self.backend,
        )
        if scenario.aerosol_polarization == "mie_sphere":
            from .radiometry.spherical_scattering import apply_mie_optics
            optics = apply_mie_optics(scenario, column, optics, grid)
        if column is None:
            return optics, None, path_transmittance(optics)

        # Multiple scattering: optics were computed on the column; map them
        # (and the aerosol loading the radiometry reports) onto the sightline.
        from .geometry.plane_parallel import sightline_optics
        column_optics = optics
        for j, i in enumerate(self.indices):
            self.path.layers[j].aerosol_od_550nm = column.layers[i].aerosol_od_550nm * self.factors[j]
        optics = sightline_optics(column_optics, self.path, self.indices, self.factors)
        return optics, column_optics, path_transmittance(optics)

    def radiance(self, grid, optics, column_optics, transmittance) -> RadianceResult:
        return _solve_radiance(self.scenario, grid, self.profile, self.path, optics,
                               self.column, column_optics, transmittance, self.sun, self.moon)

    def run(self, grid: SpectralGrid):
        optics, column_optics, transmittance = self.optics(grid)
        radiance = (self.radiance(grid, optics, column_optics, transmittance)
                    if self.include_radiance else None)
        return optics, column_optics, transmittance, radiance


# =========================================================================
# Resolved sampling: fine-grid chunks, worker processes, GPU residency
# =========================================================================


#: Fine point x layer elements per resolved chunk on the CUDA backend. Chunks
#: are large, because each one repeats line selection over its 25 cm^-1
#: cutoff margins. Full default band on an RTX 4090 (device-resident):
#: 2**22 -> 1.30 s, 0.5 GiB of GPU memory; 2**23 -> 0.95 s, 1.0 GiB;
#: 2**24 -> 0.92 s, 2.0 GiB. The GPU may be shared, so the middle one.
_CUDA_CHUNK_ELEMENTS = 1 << 23
#: Resolved chunks on the CUDA backend keep every fine-grid array on the GPU
#: and copy back only the instrument-reduced output (see _device_resident).
#: Off only for comparisons with the host path.
_CUDA_DEVICE_RESIDENT = True
#: Resolved runs smaller than this many (fine point x layer) elements stay in
#: one process: worker start-up (~1 s) would cost more than it saves.
_PROCESS_MIN_ELEMENTS = 1 << 21
#: Parallel resolved runs are split into at least this many chunks per
#: worker, so uneven chunk costs (line density varies across the band) balance.
_CHUNKS_PER_WORKER = 3
#: Correct correlated-k binned DISORT / Monte Carlo radiance with the exact
#: fine-grid single-scatter result (resolved.binned_radiance). Off only for
#: validation of the uncorrected binning.
_CK_SINGLE_SCATTER_CORRECTION = True
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run_resolved(spectral_pass: _SpectralPass, grid: SpectralGrid):
    """Resolved sampling: fine-grid chunks reduced by the instrument function."""
    from . import resolved
    from .hitran.fetch import ensure_linelists
    from .optics.lbl import line_strength_reference
    from .types import REQUIRED_SPECIES

    scenario = spectral_pass.scenario
    if grid.n > 1 and not np.allclose(np.diff(grid.wavenumber_cm1), grid.resolution_cm1,
                                      rtol=1e-9, atol=0.0):
        raise ValueError("Resolved sampling requires a uniform output grid")
    layers = spectral_pass.optics_layers
    step = scenario.fine_step_cm1
    if step is None:
        step = resolved.automatic_fine_step(spectral_pass.path.layers,
                                            float(grid.wavenumber_cm1[0]))
    plan = resolved.plan_fine_grid(grid, fine_step_cm1=step,
                                   function=scenario.instrument_function,
                                   fwhm_cm1=scenario.instrument_fwhm_cm1)

    if scenario.line_strength_cutoff > 0.0:
        # Every chunk must prune the same lines as an undivided evaluation.
        linelists = ensure_linelists()
        spectral_pass.strength_references = {
            (row, sp): line_strength_reference(linelists[sp], layer.t_eff_k.get(sp, layer.t_k))
            for row, layer in enumerate(layers) for sp in REQUIRED_SPECIES if sp in linelists}

    binned = (spectral_pass.include_radiance
              and scenario.radiance_solver != RadianceSolver.SINGLE_SCATTER)
    workers = resolve_thread_count(spectral_pass.n_threads)
    n_fine = (grid.n - 1) * plan.ratio + 2 * plan.support + 1
    gpu = spectral_pass.backend == "cuda"
    spectral_pass.device = gpu and _device_resident(spectral_pass)
    parallel = (not gpu and workers > 1 and _parallel_mode() == "processes"
                and n_fine * len(layers) >= _PROCESS_MIN_ELEMENTS)
    pieces = list(resolved.chunks(plan, grid.n, len(layers),
                                  min_chunks=_CHUNKS_PER_WORKER * workers if parallel else 1,
                                  max_elements=_CUDA_CHUNK_ELEMENTS if gpu else None))
    jobs = [(plan, first, stop, fine, grid.wavenumber_cm1[first:stop], grid.resolution_cm1, binned)
            for first, stop, fine in pieces]
    parts = _run_chunks_in_processes(spectral_pass, jobs, workers) if parallel and len(jobs) > 1 else None
    if parts is None:
        parts = [_resolved_chunk(spectral_pass, *job) for job in jobs]

    transmittance = resolved.concatenate_transmittance([t for t, _ in parts], grid)
    radiances = [r for _, r in parts if r is not None]
    radiance = resolved.concatenate_radiance(radiances, grid) if radiances else None
    return transmittance, radiance, plan.step_cm1


def _parallel_mode() -> str:
    """``MWIR_PARALLEL``: ``processes`` (default) or ``threads``."""
    mode = os.environ.get("MWIR_PARALLEL", "processes").strip().lower() or "processes"
    if mode not in ("processes", "threads"):
        raise ValueError(f"MWIR_PARALLEL must be 'processes' or 'threads', got {mode!r}")
    return mode


def _device_resident(spectral_pass: "_SpectralPass") -> bool:
    """Whether resolved chunks can run entirely on the GPU.

    True for the single-scatter solver: continuum, aerosol, Rayleigh, path
    transmittance, radiance and the instrument reduction all accept CuPy
    arrays (their array module follows the grid), so a chunk's fine-grid
    arrays never leave the device. DISORT and Monte Carlo (binned with
    ``scipy.sparse`` on the CPU) and Mie aerosol optics use the host path.
    """
    scenario = spectral_pass.scenario
    return (_CUDA_DEVICE_RESIDENT and spectral_pass.column is None
            and scenario.radiance_solver == RadianceSolver.SINGLE_SCATTER
            and scenario.aerosol_polarization != "mie_sphere")


def _resolved_chunk(spectral_pass: "_SpectralPass", plan, first: int, stop: int,
                    fine: SpectralGrid, output_wavenumbers: np.ndarray, resolution: float,
                    binned: bool):
    """Physics on one fine chunk, reduced to its output points."""
    from . import resolved

    n_out = stop - first
    out_grid = SpectralGrid(wavenumber_cm1=output_wavenumbers, resolution_cm1=resolution)
    if spectral_pass.device:
        from .optics import lbl_cuda
        fine = SpectralGrid(wavenumber_cm1=lbl_cuda.to_device(fine.wavenumber_cm1),
                            resolution_cm1=fine.resolution_cm1,
                            lattice_origin_cm1=fine.lattice_origin_cm1,
                            lattice_index0=fine.lattice_index0)
        optics, column_optics, transmittance = spectral_pass.optics(fine)
        t_out = resolved.degrade_transmittance(transmittance, plan, n_out, out_grid)
        r_out = None
        if spectral_pass.include_radiance:
            r_out = _host_arrays(resolved.degrade_radiance(
                spectral_pass.radiance(fine, optics, column_optics, transmittance),
                plan, n_out, out_grid))
        return _host_arrays(t_out), r_out

    optics, column_optics, transmittance = spectral_pass.optics(fine)
    t_out = resolved.degrade_transmittance(transmittance, plan, n_out, out_grid)
    if not spectral_pass.include_radiance:
        return t_out, None
    if binned:
        reference = {}
        if _CK_SINGLE_SCATTER_CORRECTION:
            # Exact single-scatter radiance on the fine grid corrects the binning
            # error shared by both solvers (see resolved.binned_radiance).
            reference = dict(
                reference_solve=lambda g, col: _binned_solve(spectral_pass, g, col,
                                                             single_scatter=True),
                reference_exact=resolved.degrade_radiance(
                    _single_scatter(spectral_pass, fine, optics, transmittance),
                    plan, n_out, out_grid))
        return t_out, resolved.binned_radiance(
            column_optics, plan, n_out, out_grid.wavenumber_cm1,
            lambda g, col: _binned_solve(spectral_pass, g, col), out_grid, **reference)
    return t_out, resolved.degrade_radiance(
        spectral_pass.radiance(fine, optics, column_optics, transmittance), plan, n_out, out_grid)


def _host_arrays(result):
    """``result`` (a result dataclass) with its CuPy arrays, and dicts of them, copied to NumPy."""
    from dataclasses import fields
    from ._array import to_host
    for f in fields(result):
        value = getattr(result, f.name)
        if isinstance(value, dict):
            setattr(result, f.name, {k: to_host(v) for k, v in value.items()})
        elif f.name != "grid":
            setattr(result, f.name, to_host(value))
    return result


_WORKER_PASS: Optional["_SpectralPass"] = None


def _init_chunk_worker(spectral_pass: "_SpectralPass") -> None:
    global _WORKER_PASS
    _WORKER_PASS = spectral_pass


def _worker_chunk(*job):
    return _resolved_chunk(_WORKER_PASS, *job)


def _run_chunks_in_processes(spectral_pass: "_SpectralPass", jobs: list, workers: int):
    """Evaluate chunks in worker processes; None if processes are unavailable.

    Every worker evaluates whole chunks single-threaded, so the whole
    pipeline (not only the line-by-line step) runs in parallel; results match
    a serial run to rounding (chunk sizes change summation order). Linux uses
    ``fork`` when this process has no other threads (workers inherit the
    loaded line lists); otherwise ``spawn``, which re-imports the calling
    script, so scripts need the usual
    ``if __name__ == "__main__":`` guard (set ``MWIR_PARALLEL=threads`` to
    avoid processes entirely).
    """
    import multiprocessing
    import threading
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    from .hitran.fetch import ensure_linelists
    from .optics.lbl import prepare_linelists

    n_proc = min(workers, len(jobs))
    method = ("fork" if "fork" in multiprocessing.get_all_start_methods()
              and threading.active_count() == 1 else "spawn")
    if method == "fork":
        prepare_linelists(ensure_linelists())      # inherited by every worker
    worker_pass = copy.copy(spectral_pass)
    worker_pass.n_threads = 1
    # Spawned workers must not each start a BLAS/OpenMP pool the size of the machine.
    saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
    os.environ.update({k: "1" for k in _SINGLE_THREAD_ENV})
    try:
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=multiprocessing.get_context(method),
                                 initializer=_init_chunk_worker, initargs=(worker_pass,)) as pool:
            return list(pool.map(_worker_chunk, *zip(*jobs)))
    except (BrokenProcessPool, OSError, RuntimeError) as exc:
        warnings.warn(f"resolved sampling: worker processes unavailable ({exc!r}); "
                      "continuing in this process with threads", RuntimeWarning, stacklevel=3)
        return None
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# =========================================================================
# Radiance solvers
# =========================================================================


def _binned_solve(spectral_pass: _SpectralPass, grid: SpectralGrid, column_optics: LayerOptics,
                  single_scatter: bool = False):
    from .geometry.plane_parallel import sightline_optics
    optics = sightline_optics(column_optics, spectral_pass.path, spectral_pass.indices,
                              spectral_pass.factors)
    transmittance = path_transmittance(optics)
    if single_scatter:
        return _single_scatter(spectral_pass, grid, optics, transmittance)
    return spectral_pass.radiance(grid, optics, column_optics, transmittance)


def _single_scatter(spectral_pass: _SpectralPass, grid: SpectralGrid, optics: LayerOptics,
                    transmittance) -> RadianceResult:
    """Single-scattering radiance along the scenario's sightline."""
    return compute_radiance(scenario=spectral_pass.scenario, grid=grid,
                            profile=spectral_pass.profile, path=spectral_pass.path,
                            optics=optics, transmittance=transmittance,
                            sun=spectral_pass.sun, moon=spectral_pass.moon)


def _solve_radiance(scenario, grid, profile, path, optics, column, column_optics,
                    transmittance, sun, moon) -> RadianceResult:
    """Dispatch to the configured radiance solver."""
    if scenario.radiance_solver == RadianceSolver.DISORT:
        from .radiometry.multiple_scattering import compute_multiple_scattering
        return compute_multiple_scattering(
            scenario=scenario, grid=grid, profile=profile, path=path,
            optics=optics, column=column, column_optics=column_optics,
            transmittance=transmittance, sun=sun, moon=moon)
    if scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        from .radiometry.spherical_scattering import compute_spherical_scattering
        return compute_spherical_scattering(
            scenario=scenario, grid=grid, profile=profile, path=path,
            optics=optics, column=column, column_optics=column_optics,
            transmittance=transmittance, sun=sun, moon=moon)
    return compute_radiance(
        scenario=scenario,
        grid=grid,
        profile=profile,
        path=path,
        optics=optics,
        transmittance=transmittance,
        sun=sun,
        moon=moon,
    )
