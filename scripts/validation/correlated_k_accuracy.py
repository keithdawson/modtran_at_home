"""Pending item 3: error of the correlated-k binning used by DISORT / Monte Carlo.

Resolved sampling runs DISORT and spherical Monte Carlo once per correlated-k
interval (at most 8 per output point) instead of at every fine point, and
corrects the result with the exact fine-grid single-scatter radiance
(``resolved.binned_radiance``). This script compares DISORT radiance from

* ``corrected`` -- the production path (binned + single-scatter correction),
* ``binned``    -- the binning alone (``api._CK_SINGLE_SCATTER_CORRECTION = False``),

with ``exact`` DISORT run at every fine point and instrument-averaged, with and
without the sun, on every geometry and 20 cm^-1 window, at a fixed fine step.
Reports the worst 1 cm^-1 bin relative error and the window-integrated error.

    python scripts/validation/correlated_k_accuracy.py --jobs 190
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np

import _common as c
from fine_step_convergence import WINDOWS

MODES = ["exact", "binned", "corrected"]
#: Noon on the June solstice at 35 N: the sun 11.6 deg from zenith.
DAY = dict(include_solar=True, local_datetime=datetime(2026, 6, 21, 12, 0), utc_offset_hours=0.0)


def job(geometry: str, window: str, day: bool, mode: str, step: float):
    from mwir_atmos import api
    from mwir_atmos.types import RadianceSolver
    # Workers run several jobs: always wrap the unpatched function.
    original = api.__dict__.setdefault("_original_resolved_chunk", api._resolved_chunk)
    api._resolved_chunk = lambda sp, *job: original(sp, *job[:-1], mode != "exact")
    api._CK_SINGLE_SCATTER_CORRECTION = mode == "corrected"
    lo, hi = WINDOWS[window]
    sc = c.scenario(geometry, "4.5-5.0um", wavenumber_min_cm1=lo, wavenumber_max_cm1=hi,
                    fine_step_cm1=step, radiance_solver=RadianceSolver.DISORT,
                    **(DAY if day else {}))
    t0 = time.perf_counter()
    r = api.run_scenario(sc, n_threads=1)
    return r.radiance.total, time.perf_counter() - t0


def main() -> None:
    p = c.parser(__doc__)
    p.add_argument("--fine-step", type=float, default=0.005)
    p.add_argument("--windows", nargs="*", default=list(WINDOWS))
    p.add_argument("--modes", nargs="*", default=MODES[1:], choices=MODES[1:])
    a = p.parse_args()
    modes = ["exact"] + a.modes
    jobs = [(g, w, d, m, a.fine_step) for g in a.geometries for w in a.windows
            for d in (False, True) for m in modes]
    jobs.sort(key=lambda j: j[3] != "exact")                 # slow exact runs first
    out = dict(zip(jobs, c.run_jobs(job, jobs, a.jobs)))

    rows = []
    for g in a.geometries:
        for w in a.windows:
            for d in (False, True):
                exact, t_exact = out[(g, w, d, "exact", a.fine_step)]
                for m in a.modes:
                    approx, t = out[(g, w, d, m, a.fine_step)]
                    rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-30)
                    rows.append(dict(geometry=g, window=w, sun=d, mode=m,
                                     max_pct=float(100 * rel.max()),
                                     band_pct=float(100 * (approx.sum() - exact.sum()) / exact.sum()),
                                     exact_seconds=t_exact, seconds=t))
    print(f"\n{'geometry':26s} {'win':5s} {'sun':4s} {'mode':10s} {'max %':>7s} {'band %':>7s} {'speedup':>8s}")
    for r in rows:
        print(f"{r['geometry']:26s} {r['window']:5s} {'yes' if r['sun'] else 'no':4s} {r['mode']:10s} "
              f"{r['max_pct']:7.3f} {r['band_pct']:7.3f} {r['exact_seconds'] / r['seconds']:7.1f}x")
    c.write(a.out or c.OUT_DIR / "correlated_k_accuracy.json",
            dict(fine_step_cm1=a.fine_step, solver="disort", rows=rows))


if __name__ == "__main__":
    main()
