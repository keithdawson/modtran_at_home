"""Pending item 1: choose the default ``line_optical_depth_cutoff``.

For every geometry and window band, runs resolved sampling (1 cm^-1 box, fixed
fine step) at each cutoff and compares band transmittance with the reference
cutoff (default 1e-8; pass ``--reference 0`` to keep every line, ~10-20x
slower). Reports mean and worst-bin differences in percentage points, and the
run time of each setting.

    python scripts/validation/line_cutoff_sweep.py --jobs 190
"""
from __future__ import annotations

import time

import numpy as np

import _common as c

CUTOFFS = [1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7]


def job(geometry: str, band: str, cutoff: float, fine_step: float):
    from mwir_atmos.api import run_scenario
    sc = c.scenario(geometry, band, line_optical_depth_cutoff=cutoff, fine_step_cm1=fine_step)
    t0 = time.perf_counter()
    r = run_scenario(sc, include_radiance=False, n_threads=1)
    return r.transmittance.total, time.perf_counter() - t0


def main() -> None:
    p = c.parser(__doc__)
    p.add_argument("--reference", type=float, default=1e-8)
    p.add_argument("--cutoffs", type=float, nargs="*", default=CUTOFFS)
    p.add_argument("--fine-step", type=float, default=0.01)
    a = p.parse_args()
    settings = [a.reference] + [x for x in a.cutoffs if x != a.reference]
    jobs = [(g, b, x, a.fine_step) for g in a.geometries for b in a.bands for x in settings]
    # Most expensive (smallest cutoff) first, so no long job starts last.
    jobs.sort(key=lambda j: j[2])
    out = dict(zip(jobs, c.run_jobs(job, jobs, a.jobs)))

    rows = []
    for g in a.geometries:
        for b in a.bands:
            ref, ref_s = out[(g, b, a.reference, a.fine_step)]
            for x in settings[1:]:
                t, s = out[(g, b, x, a.fine_step)]
                d = 100.0 * (t - ref)
                rows.append(dict(geometry=g, band=b, cutoff=x, mean_pp=float(d.mean()),
                                 max_abs_pp=float(np.abs(d).max()), seconds=s,
                                 reference_seconds=ref_s, band_transmittance=float(ref.mean())))
    print(f"\n{'geometry':26s} {'band':10s} {'cutoff':>7s} {'mean pp':>9s} {'max pp':>8s} {'time':>7s}")
    for r in rows:
        print(f"{r['geometry']:26s} {r['band']:10s} {r['cutoff']:7.0e} {r['mean_pp']:9.4f} "
              f"{r['max_abs_pp']:8.4f} {r['seconds'] / r['reference_seconds']:6.2f}x")
    c.write(a.out or c.OUT_DIR / "line_cutoff_sweep.json",
            dict(reference=a.reference, fine_step_cm1=a.fine_step, rows=rows))


if __name__ == "__main__":
    main()
