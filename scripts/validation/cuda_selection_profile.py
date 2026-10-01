"""Where the time of a full-band CUDA run goes, with CPU or GPU line selection.

    python scripts/validation/cuda_selection_profile.py

1. Uninstrumented full default band (resolved, ground-to-space): CPU and GPU
   line selection (``MWIR_CUDA_SELECTION``) alternated, five runs each.
2. One instrumented run with GPU selection, the GPU synchronised around each
   stage (this removes overlap, so the stages add up to more than step 1):
   host-side task setup, each selection pass (pairs, kept lines, seconds),
   and the Voigt kernels.
3. The same for CPU selection: selection threads' time and the kernels.

Writes docs/validation/cuda_selection_profile_<hostname>.json.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import _common as c

#: Exit status when there is no usable GPU (run_all.py reports it as skipped).
NO_DEVICE = 3


def _run(scenario, selection: str) -> float:
    from mwir_atmos.api import run_scenario
    os.environ["MWIR_CUDA_SELECTION"] = selection
    t0 = time.perf_counter()
    run_scenario(scenario, backend="cuda")
    return time.perf_counter() - t0


def _instrumented(scenario, selection: str) -> dict:
    """One run with the GPU synchronised around selection and kernel launches."""
    import cupy
    from mwir_atmos.optics import lbl, lbl_cuda

    sync = cupy.cuda.Device().synchronize
    stats = dict(task_setup_s=0.0, select_s=0.0, kernel_s=0.0, passes=[], cpu_select_calls=0)
    originals = (lbl._selection_task, lbl._select_packed, lbl_cuda.launch, lbl._select_lines)

    def selection_task(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return originals[0](*args, **kwargs)
        finally:
            stats["task_setup_s"] += time.perf_counter() - t0

    def select_packed(xp, prepared, tasks, lo, hi):
        sync()
        t0 = time.perf_counter()
        packed = originals[1](xp, prepared, tasks, lo, hi)
        sync()
        seconds = time.perf_counter() - t0
        stats["select_s"] += seconds
        stats["passes"].append(dict(molecule=prepared.molecule_id, tasks=len(tasks),
                                    pairs=int(sum(t.last - t.first for t in tasks)),
                                    kept=int(packed.center.size), seconds=seconds))
        return packed

    def launch(out, nu, packed, rows):
        sync()
        t0 = time.perf_counter()
        originals[2](out, nu, packed, rows)
        sync()
        stats["kernel_s"] += time.perf_counter() - t0

    def select_lines(*args, **kwargs):
        stats["cpu_select_calls"] += 1
        return originals[3](*args, **kwargs)

    lbl._selection_task, lbl._select_packed, lbl_cuda.launch, lbl._select_lines = (
        selection_task, select_packed, launch, select_lines)
    try:
        stats["total_s"] = _run(scenario, selection)
    finally:
        lbl._selection_task, lbl._select_packed, lbl_cuda.launch, lbl._select_lines = originals
    passes = stats["passes"]
    stats["pairs"] = sum(p["pairs"] for p in passes)
    stats["kept"] = sum(p["kept"] for p in passes)
    stats["slowest_passes"] = sorted(passes, key=lambda p: -p["seconds"])[:8]
    del stats["passes"]
    stats["n_passes"] = len(passes)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from mwir_atmos.optics import lbl_cuda
    from mwir_atmos.types import Scenario
    if not lbl_cuda.available():
        print("CuPy or a CUDA device is not available.")
        return NO_DEVICE
    report = dict(host=socket.gethostname(), device=lbl_cuda.device_summary())
    print("GPU:", report["device"], flush=True)

    base = dict(include_solar=False, include_lunar=False)
    full = Scenario(**base)
    warm = Scenario(wavenumber_min_cm1=2000.0, wavenumber_max_cm1=2010.0, **base)
    for selection in ("cpu", "gpu"):     # CUDA context, NVRTC and CuPy kernel compilation
        _run(warm, selection)

    times = {"cpu": [], "gpu": []}
    for _ in range(a.repeat):
        for selection in ("cpu", "gpu"):
            times[selection].append(_run(full, selection))
    report["full_band_seconds"] = times
    for selection, values in times.items():
        print(f"full band, {selection} selection: " + " ".join(f"{t:.3f}" for t in values), flush=True)

    for selection in ("gpu", "cpu"):
        stats = _instrumented(full, selection)
        report[f"instrumented_{selection}_selection"] = stats
        print(f"instrumented, {selection} selection: total {stats['total_s']:.3f} s, "
              f"task setup {stats['task_setup_s']:.3f} s, GPU selection {stats['select_s']:.3f} s "
              f"({stats['n_passes']} passes, {stats['pairs'] / 1e6:.1f} M pairs, "
              f"{stats['kept'] / 1e6:.2f} M kept), kernels {stats['kernel_s']:.3f} s, "
              f"CPU _select_lines calls {stats['cpu_select_calls']}", flush=True)
    os.environ.pop("MWIR_CUDA_SELECTION", None)

    out = Path(a.out) if a.out else c.OUT_DIR / f"cuda_selection_profile_{socket.gethostname()}.json"
    c.write(out, report)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
