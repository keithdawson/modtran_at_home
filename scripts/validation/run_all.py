"""Run every pending validation study, one after another, and log each.

    python scripts/validation/run_all.py                 # all steps, all CPUs
    python scripts/validation/run_all.py --only cutoff hapi
    python scripts/validation/run_all.py --jobs 6        # leave cores free

Results land in docs/validation/*.json (HAPI: tests/reference_data/), logs in
docs/validation/logs/. A step whose result file already exists is skipped
unless --force. A failing step is logged and the rest still run. Commit and
push docs/validation/ and tests/reference_data/hapi_window_reference.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "docs" / "validation"
LOGS = RESULTS / "logs"

#: name -> (command, result file, description)
STEPS = {
    "cutoff": ([HERE / "line_cutoff_sweep.py"], RESULTS / "line_cutoff_sweep.json",
               "1. line_optical_depth_cutoff sweep (5 paths x 3 windows x 8 cutoffs)"),
    "finestep": ([HERE / "fine_step_convergence.py"], RESULTS / "fine_step_convergence.json",
                 "2. fine-step convergence (5 paths x 4 windows x 8 steps)"),
    "corrk": ([HERE / "correlated_k_accuracy.py"], RESULTS / "correlated_k_accuracy.json",
              "3. correlated-k binning vs exact (single-scatter + DISORT, day/night)"),
    "hapi": ([ROOT / "scripts" / "physics_reference" / "generate_hapi_reference.py"],
             ROOT / "tests" / "reference_data" / "hapi_window_reference.json",
             "4. independent HAPI line-by-line reference for window-band tests"),
    "scaling": ([HERE / "scaling_benchmark.py", "--sizes", "small"], RESULTS / "scaling_benchmark.json",
                "5. parallel scaling / cost calibration (small problem, 1..N workers)"),
    # Round 2: after the defaults changed (cutoff 3e-6, 10% fine-step rule,
    # single-scatter correction of binned radiance).
    "corrk2": ([HERE / "correlated_k_accuracy.py", "--out", RESULTS / "correlated_k_accuracy_round2.json"],
               RESULTS / "correlated_k_accuracy_round2.json",
               "R2-1. correlated-k: binned and corrected DISORT vs exact"),
    "finestep2": ([HERE / "fine_step_convergence.py", "--out", RESULTS / "fine_step_convergence_round2.json"],
                  RESULTS / "fine_step_convergence_round2.json",
                  "R2-2. fine-step convergence under the 10% rule"),
    "default": ([HERE / "scaling_benchmark.py", "--sizes", "small", "full", "--no-threads",
                 "--out", RESULTS / "scaling_round2.json"], RESULTS / "scaling_round2.json",
                "R2-3. scaling at the new step + full default-band run time"),
    # GPU backend (needs CuPy and an NVIDIA GPU; skipped otherwise).
    "cuda": ([HERE / "cuda_check.py"], RESULTS / f"cuda_check_{socket.gethostname()}.json",
             "GPU. CUDA backend: tests, then CPU vs GPU timing and agreement"),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=list(STEPS), default=list(STEPS))
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--force", action="store_true", help="Re-run steps whose result exists.")
    a = p.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    status = {}
    for name in a.only:
        command, result, text = STEPS[name]
        if result.exists() and not a.force:
            print(f"== {text}: {result.relative_to(ROOT)} exists, skipped (--force to redo)")
            status[name] = "skipped"
            continue
        if name == "cuda" and importlib.util.find_spec("cupy") is None:
            print(f"== {text}: needs CuPy (pip install \"cupy-cuda12x[ctk]\"); skipped")
            status[name] = "missing cupy"
            continue
        if name == "hapi" and importlib.util.find_spec("hapi") is None:
            print(f"== {text}: needs `pip install hitran-api==1.3.0.0`; skipped")
            status[name] = "missing hitran-api"
            continue
        args = [sys.executable, str(command[0]), *map(str, command[1:])]
        script = Path(command[0]).name
        if script == "scaling_benchmark.py":
            args += ["--max-workers", str(a.jobs)]
        elif script != "cuda_check.py":
            args += ["--jobs", str(a.jobs)]
        log = LOGS / f"{name}.log"
        print(f"== {text}\n   log: {log.relative_to(ROOT)}", flush=True)
        t0 = time.perf_counter()
        with log.open("w") as fh:
            code = subprocess.call(args, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
        minutes = (time.perf_counter() - t0) / 60
        if name == "cuda" and code == 3:                # cuda_check.NO_DEVICE
            status[name] = "skipped (no CUDA device)"
            log.unlink()
        else:
            status[name] = f"{'ok' if code == 0 else f'FAILED (exit {code})'} in {minutes:.1f} min"
        print(f"   {status[name]}", flush=True)

    print("\nSummary:")
    for name, s in status.items():
        print(f"  {name:9s} {s}")
    return 0 if all(not s.startswith("FAILED") for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
