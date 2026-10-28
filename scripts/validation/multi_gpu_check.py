"""Multi-GPU resolved runs: speed and agreement with one GPU.

    python scripts/validation/multi_gpu_check.py                 # every GPU
    python scripts/validation/multi_gpu_check.py --devices 0,1,2,3

For each case (the full default band, and the band at 0.1 cm^-1 output
resolution, ~10x the fine points) times one GPU in this process, then the
listed GPUs (``MWIR_CUDA_DEVICES``): the first multi-GPU run includes worker
start-up, later runs reuse the pool. Records the largest transmittance and
radiance differences from the one-GPU result.

Writes docs/validation/multi_gpu_check_<hostname>.json (``--out``).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

import _common as c

NO_DEVICE = 3


def _timed(scenario, repeat: int):
    from mwir_atmos.api import run_scenario
    times, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = run_scenario(scenario, backend="cuda", include_turbulence=False)
        times.append(time.perf_counter() - t0)
    return times, result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", default="all", help="MWIR_CUDA_DEVICES value (default: all)")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from mwir_atmos import multi_gpu
    from mwir_atmos.optics import lbl_cuda
    from mwir_atmos.types import Scenario
    if lbl_cuda.device_count() < 1:
        print("CuPy or a CUDA device is not available.")
        return NO_DEVICE
    os.environ["MWIR_CUDA_DEVICES"] = a.devices
    devices = multi_gpu.cuda_devices()
    os.environ.pop("MWIR_CUDA_DEVICES")
    report = dict(host=socket.gethostname(), devices=list(devices), names={})
    import cupy
    for d in devices:
        with cupy.cuda.Device(d):
            report["names"][d] = lbl_cuda.device_summary()
    print("GPUs:", *report["names"].values(), sep="\n  ")

    base = dict(include_solar=False, include_lunar=False)
    cases = {"default_full_band": Scenario(**base),
             "full_band_0.1cm-1": Scenario(resolution_cm1=0.1, **base)}
    _timed(Scenario(wavenumber_min_cm1=2000.0, wavenumber_max_cm1=2010.0, **base), 1)  # warm-up
    report["runs"] = {}
    for name, scenario in cases.items():
        one_times, one = _timed(scenario, a.repeat)
        os.environ["MWIR_CUDA_DEVICES"] = ",".join(map(str, devices))
        try:
            many_times, many = _timed(scenario, a.repeat + 1)
        finally:
            os.environ.pop("MWIR_CUDA_DEVICES")
        row = dict(one_gpu_seconds=one_times, multi_gpu_seconds=many_times,
                   speedup_warm=min(one_times) / min(many_times[1:]),
                   max_abs_transmittance_diff=float(np.abs(many.transmittance.total
                                                           - one.transmittance.total).max()),
                   max_rel_radiance_diff=float(np.abs(many.radiance.total
                                                      / one.radiance.total - 1).max()))
        report["runs"][name] = row
        print(f"{name}: 1 GPU " + " ".join(f"{t:.2f}" for t in one_times)
              + f" s; {len(devices)} GPUs " + " ".join(f"{t:.2f}" for t in many_times)
              + f" s (first includes start-up); warm speedup {row['speedup_warm']:.2f}x; "
              f"dT {row['max_abs_transmittance_diff']:.1e} dL {row['max_rel_radiance_diff']:.1e}",
              flush=True)
    multi_gpu.shutdown()

    out = Path(a.out) if a.out else c.OUT_DIR / f"multi_gpu_check_{socket.gethostname()}.json"
    c.write(out, report)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
