"""Window transmittance and radiance against an independent line-by-line code.

``tests/reference_data/hapi_window_reference.json`` was produced by
``scripts/physics_reference/generate_hapi_reference.py``: HITRAN's own HAPI
evaluates every line (S >= 1e-27) of the same HITRAN line list with its own
Voigt profile and partition sums, on a 0.005 cm^-1 grid over 10 cm^-1 windows.
Here the full model (default path layering, default line selection) runs on
the same windows at the same step, and window means are compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mwir_atmos.api import run_scenario
from mwir_atmos.types import (AerosolModel, ModelAtmosphere, Scenario, SpectralSampling,
                              SurfaceType)

_PATH = Path(__file__).parent / "reference_data" / "hapi_window_reference.json"
if not _PATH.exists():
    pytest.skip("HAPI reference not generated yet (scripts/validation/README.md, step 4)",
                allow_module_level=True)
REFERENCE = json.loads(_PATH.read_text())
STEP = REFERENCE["fine_step_cm1"]

#: Absolute window-mean transmittance tolerance; measured worst 0.0010.
#: Engine-vs-HAPI line sums agree to ~1e-3 relative; the rest is the
#: reference's coarser layering (< 1e-3 of the column) and the default line
#: selection (peak optical depth >= 3e-6).
T_TOL = 0.002
#: Relative tolerance on look-down window radiance; measured worst 0.38% (the
#: reference uses isothermal layers and approximates the reflected sky).
L_TOL = 0.006


def _run(spec: dict, lo: float, hi: float):
    kw = dict(spec)
    kw["atmosphere"] = ModelAtmosphere(kw["atmosphere"])
    kw["aerosol"] = AerosolModel(kw["aerosol"])
    if "surface_type" in kw:
        kw["surface_type"] = SurfaceType(kw["surface_type"])
    radiance = "surface_type" in kw
    sc = Scenario(wavenumber_min_cm1=lo, wavenumber_max_cm1=hi, resolution_cm1=STEP,
                  spectral_sampling=SpectralSampling.POINT, **kw)
    return run_scenario(sc, include_radiance=radiance, include_turbulence=False)


_WINDOWS = [(case, band, i)
            for case, entry in REFERENCE["cases"].items()
            for band, result in entry["results"].items()
            for i in range(len(result["windows"]))]


@pytest.mark.parametrize("case,band,index", _WINDOWS)
def test_window_matches_hapi(case, band, index):
    entry = REFERENCE["cases"][case]
    ref = entry["results"][band]["windows"][index]
    lo, hi = ref["window_cm1"]
    result = _run(entry["scenario"], lo, hi)
    t = float(result.transmittance.total.mean())
    assert abs(t - ref["transmittance"]) < T_TOL, (case, band, lo, t, ref["transmittance"])
    if "radiance" in ref:
        radiance = float(result.radiance.total.mean())
        assert radiance == pytest.approx(ref["radiance"], rel=L_TOL), (case, lo)


def test_reference_documents_the_window_expectations():
    """Values the window-band tests elsewhere are anchored to (see their comments)."""
    cases = REFERENCE["cases"]
    saw = cases["saw_15km_horizontal"]["results"]["2500-2750"]["mean_transmittance"]
    trop = cases["tropical_15km_horizontal"]["results"]["2500-2750"]["mean_transmittance"]
    assert trop < saw - 0.15
    assert np.isfinite(saw) and np.isfinite(trop)
