"""Pending item 2: is the automatic fine step converged?

For every geometry and 20 cm^-1 window, runs resolved sampling (1 cm^-1 box,
single-scatter radiance, no sun/moon) at the automatic step and at fixed steps,
and compares with a reference step (default 0.0005 cm^-1). Reports the worst
1 cm^-1 bin error in transmittance (percentage points) and radiance (percent).

    python scripts/validation/fine_step_convergence.py --jobs 190
"""
from __future__ import annotations

import time

import numpy as np

import _common as c

#: 20 cm^-1 windows: 4.8 um (CO/H2O/O3), 4.0 um (N2O/H2O continuum window),
#: 3.7 um (weak H2O lines) and 3.3 um (CH4/H2O).
WINDOWS = {"2050": (2050.0, 2070.0), "2500": (2500.0, 2520.0),
           "2700": (2700.0, 2720.0), "3000": (3000.0, 3020.0)}
STEPS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


def job(geometry: str, window: str, step):
    from mwir_atmos.api import run_scenario
    lo, hi = WINDOWS[window]
    sc = c.scenario(geometry, "4.5-5.0um", wavenumber_min_cm1=lo, wavenumber_max_cm1=hi,
                    fine_step_cm1=step)
    t0 = time.perf_counter()
    r = run_scenario(sc, n_threads=1)
    return r.transmittance.total, r.radiance.total, r.fine_step_cm1, time.perf_counter() - t0


def main() -> None:
    p = c.parser(__doc__)
    p.add_argument("--reference", type=float, default=0.0005)
    p.add_argument("--steps", type=float, nargs="*", default=STEPS)
    p.add_argument("--windows", nargs="*", default=list(WINDOWS))
    a = p.parse_args()
    settings = [a.reference, None] + [s for s in a.steps if s != a.reference]
    jobs = [(g, w, s) for g in a.geometries for w in a.windows for s in settings]
    jobs.sort(key=lambda j: j[2] if j[2] is not None else 0.001)
    out = dict(zip(jobs, c.run_jobs(job, jobs, a.jobs)))

    rows = []
    for g in a.geometries:
        for w in a.windows:
            t_ref, l_ref, _, s_ref = out[(g, w, a.reference)]
            for s in settings[1:]:
                t, l, used, sec = out[(g, w, s)]
                rel = np.abs(l - l_ref) / np.maximum(np.abs(l_ref), 1e-30)
                rows.append(dict(geometry=g, window=w, requested=s, step=used,
                                 t_max_pp=float(100 * np.abs(t - t_ref).max()),
                                 t_mean_pp=float(100 * (t - t_ref).mean()),
                                 l_max_pct=float(100 * rel.max()),
                                 l_mean_pct=float(100 * ((l - l_ref).sum() / l_ref.sum())),
                                 seconds=sec, reference_seconds=s_ref))
    print(f"\n{'geometry':26s} {'win':5s} {'step':>8s} {'T max pp':>9s} {'L max %':>8s} {'L mean %':>9s} {'time':>7s}")
    for r in rows:
        tag = f"{r['step']:.4f}" + ("*" if r["requested"] is None else " ")
        print(f"{r['geometry']:26s} {r['window']:5s} {tag:>8s} {r['t_max_pp']:9.4f} "
              f"{r['l_max_pct']:8.3f} {r['l_mean_pct']:9.3f} {r['seconds']:6.1f}s")
    print("* = automatic step")
    c.write(a.out or c.OUT_DIR / "fine_step_convergence.json", dict(reference=a.reference, rows=rows))


if __name__ == "__main__":
    main()
