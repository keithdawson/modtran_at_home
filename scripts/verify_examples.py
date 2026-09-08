"""Run all shipped examples offline and save machine-readable and visual output.

Run from the checkout: python scripts/verify_examples.py
Requires the gui extra for PNG rendering. Outputs go to artifacts/examples/.
"""
from pathlib import Path
import sys
import json
import time
import dataclasses

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from mwir_atmos.api import run_scenario
from mwir_atmos.cli import load_scenario_yaml, _band_rows, _Logger, _write_json, _write_csv
from mwir_atmos.spectral import radiance_to_per_um
from mwir_atmos.webgui.plots import line_chart_png


def main():
    destination = ROOT / "artifacts" / "examples"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in sorted((ROOT / "scenarios").glob("*.yaml")):
        scenario = load_scenario_yaml(config)
        start = time.perf_counter()
        result = run_scenario(scenario)
        elapsed = time.perf_counter() - start
        t = result.transmittance
        assert np.all(np.isfinite(t.total)) and np.all((t.total >= 0) & (t.total <= 1))
        np.testing.assert_allclose(t.total, t.molecular * t.aerosol * t.rayleigh, atol=1e-15)
        radiance = result.radiance
        assert radiance is not None
        components = [getattr(radiance, f.name) for f in dataclasses.fields(radiance)
                      if f.name not in ("total", "grid")]
        assert np.all(np.isfinite(radiance.total)) and np.all(radiance.total >= 0)
        np.testing.assert_allclose(radiance.total, np.sum(components, axis=0), atol=1e-18)
        rows = _band_rows(result, [("computed_window", (result.grid.wavenumber_cm1[0],
                          result.grid.wavenumber_cm1[-1]))], _Logger(quiet=True, verbose=False))
        with (destination / f"{config.stem}.json").open("w", encoding="utf-8") as fh:
            _write_json(result, rows, fh)
        with (destination / f"{config.stem}.csv").open("w", encoding="utf-8", newline="") as fh:
            _write_csv(result, fh)
        for label, series, unit in [
            ("transmittance", {"total": t.total, "molecular": t.molecular, "aerosol": t.aerosol}, "Transmittance"),
            ("radiance", {"total": radiance_to_per_um(radiance.total, result.grid.wavenumber_cm1)},
             "Radiance (W m^-2 sr^-1 / um)"),
        ]:
            png = line_chart_png(result.grid.wavelength_um, series, x_label="Wavelength (um)",
                                 y_label=unit, title=config.stem.replace("_", " "))
            if png is None:
                raise RuntimeError("PNG verification requires matplotlib; install the gui extra")
            (destination / f"{config.stem}_{label}.png").write_bytes(png)
        summary = dict(scenario=config.stem, seconds=round(elapsed, 3), points=result.grid.n,
                       layers=result.path.n_layers, path_km=result.path.total_length_km,
                       transmittance_min=float(t.total.min()), transmittance_max=float(t.total.max()))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
