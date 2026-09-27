"""Shared geometries and a process-parallel job runner for validation studies.

Every study runs many independent (geometry, band, setting) jobs. Each job
runs single-threaded in its own process, so a study fills all cores on any
machine (``--jobs``, default every CPU) and results never depend on the
worker count.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mwir_atmos.types import ModelAtmosphere, Scenario

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "validation"

#: Five paths spanning humid/dry, horizontal/vertical, low/high pressure.
GEOMETRIES: dict[str, dict] = {
    "tropical_10km_horizontal": dict(atmosphere=ModelAtmosphere.TROPICAL, sensor_alt_km=0.01,
                                     sensor_zenith_deg=90.0, source_range_km=10.0),
    "mls_ground_to_space": dict(atmosphere=ModelAtmosphere.MIDLAT_SUMMER, sensor_alt_km=0.0,
                                sensor_zenith_deg=0.0),
    "mls_20km_nadir": dict(atmosphere=ModelAtmosphere.MIDLAT_SUMMER, sensor_alt_km=20.0,
                           sensor_zenith_deg=180.0),
    "us_std_limb_15km": dict(atmosphere=ModelAtmosphere.US_STANDARD, sensor_alt_km=15.0,
                             sensor_zenith_deg=93.0),          # tangent height ~6 km
    "saw_10km_horizontal": dict(atmosphere=ModelAtmosphere.SUBARCTIC_WINTER, sensor_alt_km=0.01,
                                sensor_zenith_deg=90.0, source_range_km=10.0),
}

#: Window bands (cm^-1) where weak lines and line shape matter most, plus
#: the 4.3 um CO2 band edge.
BANDS: dict[str, tuple[float, float]] = {
    "4.5-5.0um": (2000.0, 2222.0),
    "3.5-4.1um": (2439.0, 2857.0),
    "2.9-3.5um": (2857.0, 3448.0),
}


def scenario(geometry: str, band: str, **overrides) -> Scenario:
    lo, hi = BANDS[band]
    base = dict(GEOMETRIES[geometry], wavenumber_min_cm1=lo, wavenumber_max_cm1=hi,
                resolution_cm1=1.0, include_solar=False, include_lunar=False)
    base.update(overrides)
    return Scenario(**base)


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                   help="Worker processes (default: every CPU).")
    p.add_argument("--geometries", nargs="*", default=list(GEOMETRIES))
    p.add_argument("--bands", nargs="*", default=list(BANDS))
    p.add_argument("--out", type=Path, default=None, help="JSON output path.")
    return p


def _init_worker() -> None:
    os.environ["MWIR_THREADS"] = "1"


def run_jobs(func, jobs: list[tuple], n_workers: int) -> list:
    """Run ``func(*job)`` for every job in worker processes; results in job order."""
    results = [None] * len(jobs)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, min(n_workers, len(jobs))),
                             initializer=_init_worker) as pool:
        futures = {pool.submit(func, *job): i for i, job in enumerate(jobs)}
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            print(f"[{done}/{len(jobs)}] {time.perf_counter() - start:7.0f} s  {jobs[futures[fut]]}",
                  flush=True)
    return results


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=float) + "\n")
    print(f"wrote {path}")


def replace(sc: Scenario, **kw) -> Scenario:
    return dataclasses.replace(sc, **kw)
