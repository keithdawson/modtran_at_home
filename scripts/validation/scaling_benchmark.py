"""Pending item 4: parallel scaling and cost-estimate calibration.

Times resolved runs of the default ground-to-space scenario (thermal radiance)
at increasing worker counts, in process mode (the default) and, for
comparison, thread mode. Two problem sizes:

* ``small``: 2000-2200 cm^-1 -- per-core throughput at 1..16 workers;
* ``full``:  the default 1800-4300 cm^-1 band -- strong scaling at >= 16 workers.

Writes elapsed time and throughput (fine points x layers per second) per run,
plus machine details, to docs/validation/scaling_benchmark.json.

    python scripts/validation/scaling_benchmark.py                 # everything
    python scripts/validation/scaling_benchmark.py --sizes small   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import _common as c


def _workers(max_workers: int, lo: int, hi: int) -> list[int]:
    out, w = [], 1
    while w <= min(hi, max_workers):
        if w >= lo:
            out.append(w)
        w *= 2
    if lo <= max_workers <= hi and max_workers not in out:
        out.append(max_workers)
    return out


def one_run(size: str, workers: int, mode: str) -> dict:
    """Run in a fresh interpreter so every run starts cold and env is clean."""
    code = f"""
import json, os, time
os.environ["MWIR_PARALLEL"] = {mode!r}
from mwir_atmos.api import run_scenario
from mwir_atmos.types import Scenario
kw = dict(include_solar=False, include_lunar=False)
if {size!r} == "small":
    kw.update(wavenumber_min_cm1=2000.0, wavenumber_max_cm1=2200.0)
if __name__ == "__main__":
    sc = Scenario(**kw)
    t0 = time.perf_counter()
    r = run_scenario(sc, n_threads={workers})
    dt = time.perf_counter() - t0
    fine = (r.grid.n - 1) * round(sc.resolution_cm1 / r.fine_step_cm1) + 1
    print(json.dumps(dict(seconds=dt, fine_points=fine, layers=r.path.n_layers,
                          fine_step_cm1=r.fine_step_cm1)))
"""
    env = dict(os.environ)
    env.pop("MWIR_THREADS", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return dict(size=size, workers=workers, mode=mode, error=proc.stderr[-2000:])
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row.update(size=size, workers=workers, mode=mode,
               elements_per_s=row["fine_points"] * row["layers"] / row["seconds"])
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--sizes", nargs="*", default=["small", "full"])
    p.add_argument("--no-threads", action="store_true", help="Skip thread-mode comparisons.")
    p.add_argument("--out", default=str(c.OUT_DIR / "scaling_benchmark.json"))
    a = p.parse_args()

    plan = []
    if "small" in a.sizes:
        plan += [("small", w, "processes") for w in _workers(a.max_workers, 1, 16)]
        if not a.no_threads:
            plan += [("small", w, "threads") for w in _workers(a.max_workers, 4, 16)]
    if "full" in a.sizes:
        plan += [("full", w, "processes") for w in _workers(a.max_workers, 16, 1 << 20)]
        if not a.no_threads:
            plan += [("full", a.max_workers, "threads")]

    rows = []
    for size, w, mode in plan:
        row = one_run(size, w, mode)
        rows.append(row)
        if "error" in row:
            print(f"{size:5s} {mode:9s} {w:4d} workers  ERROR\n{row['error']}", flush=True)
        else:
            print(f"{size:5s} {mode:9s} {w:4d} workers  {row['seconds']:8.1f} s  "
                  f"{row['elements_per_s'] / 1e6:7.2f} M elem/s", flush=True)
        c.write(__import__("pathlib").Path(a.out), dict(
            machine=dict(cpu_count=os.cpu_count(), platform=platform.platform(),
                         python=sys.version, processor=platform.processor()),
            rows=rows))


if __name__ == "__main__":
    main()
