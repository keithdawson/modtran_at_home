"""Pending item 3: error of the correlated-k binning used by DISORT / Monte Carlo.

Resolved sampling runs DISORT and spherical Monte Carlo once per correlated-k
interval (at most 8 per output point) instead of at every fine point. This
script compares that binned radiance with the exact fine-grid radiance (the
solver run at every fine point, then instrument-averaged) for:

* ``single_scatter`` -- binning applied to the single-scattering solver, and
* ``disort`` -- the multiple-scattering solver the binning exists for,

with and without the sun, on every geometry and 20 cm^-1 window, at a fixed
fine step (the binning error, not the step, is under test). Reports the worst
1 cm^-1 bin relative error and the band-integrated error.

    python scripts/validation/correlated_k_accuracy.py --jobs 190
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np

import _common as c
from fine_step_convergence import WINDOWS

SOLVERS = ["single_scatter", "disort"]
#: Noon on the June solstice at 35 N: the sun 11.6 deg from zenith.
DAY = dict(include_solar=True, local_datetime=datetime(2026, 6, 21, 12, 0), utc_offset_hours=0.0)


def _chunk_function(solver: str, binned: bool):
    """Replacement for ``api._resolved_chunk`` forcing binned or exact radiance."""
    from mwir_atmos import api, resolved
    from mwir_atmos.optics.transmittance import path_transmittance
    from mwir_atmos.types import SpectralGrid

    # Workers run several jobs: always wrap the unpatched function.
    original = api.__dict__.setdefault("_original_resolved_chunk", api._resolved_chunk)
    if solver != "single_scatter":
        return lambda sp, *job: original(sp, *job[:-1], binned)
    if not binned:
        return original

    def single_scatter_binned(sp, plan, first, stop, fine, out_wn, resolution, _binned):
        n_out = stop - first
        out_grid = SpectralGrid(wavenumber_cm1=out_wn, resolution_cm1=resolution)
        optics, _, transmittance = sp.optics(fine)
        return (resolved.degrade_transmittance(transmittance, plan, n_out, out_grid),
                resolved.binned_radiance(
                    optics, plan, n_out, out_wn,
                    lambda g, o: sp.radiance(g, o, None, path_transmittance(o)), out_grid))
    return single_scatter_binned


def job(geometry: str, window: str, solver: str, day: bool, binned: bool, step: float):
    from mwir_atmos import api
    from mwir_atmos.types import RadianceSolver
    api._resolved_chunk = _chunk_function(solver, binned)
    lo, hi = WINDOWS[window]
    sc = c.scenario(geometry, "4.5-5.0um", wavenumber_min_cm1=lo, wavenumber_max_cm1=hi,
                    fine_step_cm1=step, radiance_solver=RadianceSolver(solver), **(DAY if day else {}))
    t0 = time.perf_counter()
    r = api.run_scenario(sc, n_threads=1)
    return r.radiance.total, time.perf_counter() - t0


def main() -> None:
    p = c.parser(__doc__)
    p.add_argument("--fine-step", type=float, default=0.005)
    p.add_argument("--windows", nargs="*", default=list(WINDOWS))
    p.add_argument("--solvers", nargs="*", default=SOLVERS)
    a = p.parse_args()
    jobs = [(g, w, s, d, b, a.fine_step) for g in a.geometries for w in a.windows
            for s in a.solvers for d in (False, True) for b in (False, True)]
    jobs.sort(key=lambda j: (j[2] == "single_scatter", j[4]))       # exact DISORT first
    out = dict(zip(jobs, c.run_jobs(job, jobs, a.jobs)))

    rows = []
    for g in a.geometries:
        for w in a.windows:
            for s in a.solvers:
                for d in (False, True):
                    exact, t_exact = out[(g, w, s, d, False, a.fine_step)]
                    binned, t_binned = out[(g, w, s, d, True, a.fine_step)]
                    rel = np.abs(binned - exact) / np.maximum(np.abs(exact), 1e-30)
                    rows.append(dict(geometry=g, window=w, solver=s, sun=d,
                                     max_pct=float(100 * rel.max()),
                                     band_pct=float(100 * (binned.sum() - exact.sum()) / exact.sum()),
                                     exact_seconds=t_exact, binned_seconds=t_binned))
    print(f"\n{'geometry':26s} {'win':5s} {'solver':15s} {'sun':4s} {'max %':>7s} {'band %':>7s} {'speedup':>8s}")
    for r in rows:
        print(f"{r['geometry']:26s} {r['window']:5s} {r['solver']:15s} {'yes' if r['sun'] else 'no':4s} "
              f"{r['max_pct']:7.3f} {r['band_pct']:7.3f} {r['exact_seconds'] / r['binned_seconds']:7.1f}x")
    c.write(a.out or c.OUT_DIR / "correlated_k_accuracy.json", dict(fine_step_cm1=a.fine_step, rows=rows))


if __name__ == "__main__":
    main()
