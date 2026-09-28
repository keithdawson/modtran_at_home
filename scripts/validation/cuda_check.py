"""CUDA backend: correctness and speed on this machine's GPU.

    pip install "cupy-cuda12x[ctk]"            # or cupy-cuda13x[ctk] for a CUDA 13 driver
    python scripts/validation/cuda_check.py

1. Prints the device and runs tests/test_lbl_cuda.py (kernel vs its NumPy
   twin, and vs the CPU engine).
2. Times the default resolved run (full 1800-4300 cm^-1 band, ground-to-space)
   and a 2000-2200 cm^-1 slice on the CPU (all cores) and on the GPU, and
   records the largest transmittance and radiance differences.

Writes docs/validation/cuda_check_<hostname>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _common as c

#: Exit status when there is no usable GPU (run_all.py reports it as skipped).
NO_DEVICE = 3


def _timed(scenario, backend: str, repeat: int = 1):
    from mwir_atmos.api import run_scenario
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = run_scenario(scenario, backend=backend)
        best = min(best, time.perf_counter() - t0)
    return best, result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-cpu-full", action="store_true",
                   help="Skip the CPU full-band run (the slowest part).")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from mwir_atmos.optics import lbl_cuda
    from mwir_atmos.types import Scenario
    report = dict(machine=dict(host=socket.gethostname(), platform=platform.platform(),
                               cpu_count=os.cpu_count(), python=sys.version))
    if not lbl_cuda.available():
        print("CuPy or a CUDA device is not available -- see the docstring for installation.")
        return NO_DEVICE
    report["device"] = lbl_cuda.device_summary()
    print("GPU:", report["device"], flush=True)

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            str(c.ROOT / "tests" / "test_lbl_cuda.py")],
                           capture_output=True, text=True, cwd=str(c.ROOT))
    report["tests"] = dict(returncode=tests.returncode, tail=tests.stdout[-3000:] + tests.stderr[-2000:])
    print(tests.stdout[-1500:], flush=True)

    base = dict(include_solar=False, include_lunar=False)
    cases = {"slice_2000_2200": Scenario(wavenumber_min_cm1=2000.0, wavenumber_max_cm1=2200.0, **base),
             "default_full_band": Scenario(**base)}
    # Warm-up: CUDA context and kernel compilation (NVRTC) happen once.
    _timed(Scenario(wavenumber_min_cm1=2000.0, wavenumber_max_cm1=2010.0, **base), "cuda")
    report["runs"] = {}
    for name, scenario in cases.items():
        row = {}
        g_seconds, g = _timed(scenario, "cuda", repeat=2)
        row["cuda_seconds"] = g_seconds
        print(f"{name:18s} cuda {g_seconds:8.2f} s", flush=True)
        if not (a.skip_cpu_full and name == "default_full_band"):
            c_seconds, cpu = _timed(scenario, "cpu")
            row.update(cpu_seconds=c_seconds, speedup=c_seconds / g_seconds,
                       max_abs_transmittance_diff=float(np.abs(g.transmittance.total
                                                               - cpu.transmittance.total).max()),
                       max_rel_radiance_diff=float(np.abs(g.radiance.total / cpu.radiance.total - 1).max()))
            print(f"{name:18s} cpu  {c_seconds:8.2f} s  speedup {row['speedup']:.1f}x  "
                  f"dT {row['max_abs_transmittance_diff']:.1e}  dL {row['max_rel_radiance_diff']:.1e}",
                  flush=True)
        report["runs"][name] = row

    out = Path(a.out) if a.out else c.OUT_DIR / f"cuda_check_{socket.gethostname()}.json"
    c.write(out, report)
    return 0 if tests.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
