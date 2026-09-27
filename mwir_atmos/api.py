"""Top-level orchestration: a :class:`Scenario` in, a :class:`ScenarioResult` out.

OVERSEER-OWNED. This is the single entry point the CLI and the web GUI both call,
so that they cannot drift apart in behaviour.

Pipeline:
    model atmosphere -> surface overrides -> slant path -> layer optics
    -> transmittance -> (radiance, once the radiometry layer exists)
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np

from .atmosphere.humidity import apply_surface_conditions
from .atmosphere.profiles import get_profile
from .geometry.path import build_path
from .optics.lbl import resolve_thread_count
from .optics.transmittance import compute_layer_optics, path_transmittance
from .spectral import make_grid
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
)

__all__ = ["build_profile", "build_scenario_path", "run_scenario"]


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


def build_scenario_path(scenario: Scenario, profile: AtmosphereProfile) -> PathGeometry:
    """Slant path implied by the scenario's sensor and source placement."""
    if scenario.radiance_solver == RadianceSolver.DISORT:
        from .geometry.plane_parallel import geometry
        return geometry(scenario, profile)[1]
    if scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        from .geometry.spherical_shells import geometry
        return geometry(scenario, profile)[1]
    return build_path(
        profile,
        sensor_alt_km=scenario.sensor_alt_km,
        zenith_deg=scenario.sensor_zenith_deg,
        azimuth_deg=scenario.sensor_azimuth_deg,
        source_alt_km=scenario.source_alt_km,
        source_range_km=scenario.source_range_km,
    )


def run_scenario(
    scenario: Scenario,
    *,
    include_radiance: bool = True,
    include_turbulence: bool = True,
    grid: Optional[SpectralGrid] = None,
    n_threads: Optional[int] = None,
) -> ScenarioResult:
    """Run a full scenario.

    ``include_radiance`` and ``include_turbulence`` degrade gracefully: if the
    corresponding module has not been built yet, the matching field of the
    result is left as ``None`` rather than raising. Transmittance is always
    computed.

    ``grid`` overrides the output grid (uniform spacing). ``n_threads`` sets
    the line-by-line worker threads (default: ``MWIR_THREADS`` or every CPU);
    results do not depend on it. With ``spectral_sampling="resolved"`` the
    physics runs on a fine internal grid and is convolved with the instrument
    function onto the output grid -- see :mod:`mwir_atmos.resolved`.
    """
    validate_scenario(scenario)
    if grid is None:
        grid = make_grid(
            scenario.wavenumber_min_cm1,
            scenario.wavenumber_max_cm1,
            scenario.resolution_cm1,
        )

    profile = build_profile(scenario)
    column = indices = factors = None
    if scenario.radiance_solver == RadianceSolver.DISORT:
        from .geometry.plane_parallel import geometry
        column, path, indices, factors = geometry(scenario, profile)
    elif scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        from .geometry.spherical_shells import geometry
        column, path, indices, factors = geometry(scenario, profile)
    else:
        path = build_scenario_path(scenario, profile)

    sun = moon = None
    if include_radiance:
        sky = _sky_positions(scenario)
        if sky is None:
            include_radiance = False  # radiometry layer absent: transmittance only
        else:
            sun, moon = sky

    spectral_pass = _SpectralPass(scenario, profile, path, column, indices, factors, sun, moon,
                                  include_radiance, n_threads)
    fine_step = None
    if scenario.spectral_sampling == SpectralSampling.RESOLVED:
        transmittance, radiance, fine_step = _run_resolved(spectral_pass, grid)
    else:
        _, _, transmittance, radiance = spectral_pass.run(grid)

    result = ScenarioResult(
        scenario=scenario,
        grid=grid,
        profile=profile,
        path=path,
        transmittance=transmittance,
        radiance=radiance,
        sun=sun,
        moon=moon,
        fine_step_cm1=fine_step,
    )
    if include_turbulence and scenario.turbulence.value != "none":
        _attach_turbulence(result)

    return result


def _sky_positions(scenario: Scenario):
    """``(sun, moon)``, or None if the radiometry/ephemeris layer is absent."""
    try:
        from .ephemeris.moon import lunar_position
        from .ephemeris.sun import solar_position
        from .radiometry.radiance import compute_radiance  # noqa: F401 (availability)
    except ImportError:
        return None  # wave 2 not built yet; transmittance-only result is still valid
    sun = solar_position(scenario)
    return sun, lunar_position(scenario, sun)


class _SpectralPass:
    """Layer optics, transmittance and radiance for one spectral grid.

    Point sampling runs this once on the output grid; resolved sampling runs
    it on each fine-grid chunk.
    """

    def __init__(self, scenario, profile, path, column, indices, factors, sun, moon,
                 include_radiance, n_threads, strength_references=None):
        self.scenario = scenario
        self.profile = profile
        self.path = path
        self.column = column
        self.indices = indices
        self.factors = factors
        self.sun = sun
        self.moon = moon
        self.include_radiance = include_radiance
        self.n_threads = n_threads
        self.strength_references = strength_references

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
        )
        if scenario.aerosol_polarization == "mie_sphere":
            from .radiometry.spherical_scattering import apply_mie_optics
            optics=apply_mie_optics(scenario,column,optics,grid)
        column_optics = optics if column is not None else None
        if column is not None:
            from .geometry.plane_parallel import sightline_optics
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
    parallel = (workers > 1 and _parallel_mode() == "processes"
                and n_fine * len(layers) >= _PROCESS_MIN_ELEMENTS)
    pieces = list(resolved.chunks(plan, grid.n, len(layers),
                                  min_chunks=_CHUNKS_PER_WORKER * workers if parallel else 1))
    jobs = [(plan, first, stop, fine, grid.wavenumber_cm1[first:stop], grid.resolution_cm1, binned)
            for first, stop, fine in pieces]
    parts = _run_chunks_in_processes(spectral_pass, jobs, workers) if parallel and len(jobs) > 1 else None
    if parts is None:
        parts = [_resolved_chunk(spectral_pass, *job) for job in jobs]

    transmittance = resolved.concatenate_transmittance([t for t, _ in parts], grid)
    radiances = [r for _, r in parts if r is not None]
    radiance = resolved.concatenate_radiance(radiances, grid) if radiances else None
    return transmittance, radiance, plan.step_cm1


#: Resolved runs smaller than this many (fine point x layer) elements stay in
#: one process: worker start-up (~1 s) would cost more than it saves.
_PROCESS_MIN_ELEMENTS = 1 << 24
#: Parallel resolved runs are split into at least this many chunks per
#: worker, so uneven chunk costs (line density varies across the band) balance.
_CHUNKS_PER_WORKER = 3
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _parallel_mode() -> str:
    """``MWIR_PARALLEL``: ``processes`` (default) or ``threads``."""
    mode = os.environ.get("MWIR_PARALLEL", "processes").strip().lower() or "processes"
    if mode not in ("processes", "threads"):
        raise ValueError(f"MWIR_PARALLEL must be 'processes' or 'threads', got {mode!r}")
    return mode


def _resolved_chunk(spectral_pass: "_SpectralPass", plan, first: int, stop: int,
                    fine: SpectralGrid, output_wavenumbers: np.ndarray, resolution: float,
                    binned: bool):
    """Physics on one fine chunk, reduced to its output points."""
    from . import resolved

    n_out = stop - first
    out_grid = SpectralGrid(wavenumber_cm1=output_wavenumbers, resolution_cm1=resolution)
    optics, column_optics, transmittance = spectral_pass.optics(fine)
    t_out = resolved.degrade_transmittance(transmittance, plan, n_out, out_grid)
    if not spectral_pass.include_radiance:
        return t_out, None
    if binned:
        return t_out, resolved.binned_radiance(
            column_optics, plan, n_out, out_grid.wavenumber_cm1,
            lambda g, col: _binned_solve(spectral_pass, g, col), out_grid)
    return t_out, resolved.degrade_radiance(
        spectral_pass.radiance(fine, optics, column_optics, transmittance), plan, n_out, out_grid)


_WORKER_PASS: Optional["_SpectralPass"] = None


def _init_chunk_worker(spectral_pass: "_SpectralPass") -> None:
    global _WORKER_PASS
    _WORKER_PASS = spectral_pass


def _worker_chunk(*job):
    return _resolved_chunk(_WORKER_PASS, *job)


def _run_chunks_in_processes(spectral_pass: "_SpectralPass", jobs: list, workers: int):
    """Evaluate chunks in worker processes; None if processes are unavailable.

    Every worker evaluates whole chunks single-threaded, so the whole
    pipeline (not only the line-by-line step) runs in parallel and results
    are identical to a serial run. Linux uses ``fork`` when this process has
    no other threads (workers inherit the loaded line lists); otherwise
    ``spawn``, which re-imports the calling script, so scripts need the usual
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


def _binned_solve(spectral_pass: _SpectralPass, grid: SpectralGrid, column_optics: LayerOptics):
    from .geometry.plane_parallel import sightline_optics
    optics = sightline_optics(column_optics, spectral_pass.path, spectral_pass.indices,
                              spectral_pass.factors)
    return spectral_pass.radiance(grid, optics, column_optics, path_transmittance(optics))


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
    from .radiometry.radiance import compute_radiance
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


def _attach_turbulence(result: ScenarioResult) -> None:
    """Populate turbulence metrics if the turbulence layer is present."""
    try:
        from .turbulence.metrics import compute_turbulence
    except ImportError:
        return  # wave 3 stretch goal not built yet

    result.turbulence = compute_turbulence(
        scenario=result.scenario, path=result.path, profile=result.profile
    )
