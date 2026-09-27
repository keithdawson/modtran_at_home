"""Dev-only reference: independent line-by-line window transmittance with HAPI.

    pip install hitran-api==1.3.0.0
    python scripts/physics_reference/generate_hapi_reference.py --jobs 8

HITRAN's own Python API (Kochanov et al. 2016, JQSRT 177, 15-30) evaluates the
line absorption: its own Voigt (HTP) profile, TIPS-2021 partition sums,
temperature scaling, pressure broadening and shift. It is fed this package's
cached HITRAN line list, so the comparison isolates the line-by-line
*implementation*, not the database. Everything else is shared on purpose --
atmospheric profile, path layers and their Curtis-Godson p/T/columns, MT_CKD
continuum (validated separately against AER's code), aerosol and Rayleigh --
because the question is whether the engine sums the lines correctly now that
the weak window lines are kept.

Conventions matched: 25 cm^-1 line cutoff for every species; H2O lines have
their Lorentz value at 25 cm^-1 subtracted (the MT_CKD pedestal). Every line
with S(T) >= 1e-27 cm/molecule is evaluated (peak optical depths below ~1e-7
on these paths), on a 0.005 cm^-1 grid. Each band is sampled by 10 cm^-1 windows every
50 cm^-1; a window's transmittance is its mean over the fine grid (the exact
window average), and ``mean_transmittance`` averages the windows.

Paths use a coarse Curtis-Godson layering (one layer for the 10 m horizontal
paths, whose altitude changes by < 20 m; six for the 8 km look-down), which
changes air columns by < 1e-3 relative to the package's default layering.

Radiance for the look-down case is an independent Schwarzschild integral on the
fine grid: isothermal layers at their layer temperature, surface emission
eps*B(Ts) plus reflected downwelling from the same layers (diffusivity 1.66).

Writes tests/reference_data/hapi_window_reference.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "reference_data" / "hapi_window_reference.json"
STEP = 0.005
CUTOFF = 25.0
#: Each band is sampled by 10 cm^-1 windows starting every WINDOW_SPACING cm^-1
#: (HAPI evaluates lines in a Python loop; whole bands would take days).
WINDOW_WIDTH = 10.0
WINDOW_SPACING = 50.0
S_MIN = 1e-27

CASES = {
    # tests/test_physical_benchmarks.py::test_climatological_soundings_polar_vs_tropical
    "saw_15km_horizontal": dict(scenario=dict(atmosphere="subarctic_winter", sensor_alt_km=0.01,
                                              sensor_zenith_deg=90.0, source_range_km=15.0,
                                              aerosol="none"),
                                bands={"2500-2750": (2500.0, 2750.0)}, layers=1),
    "tropical_15km_horizontal": dict(scenario=dict(atmosphere="tropical", sensor_alt_km=0.01,
                                                   sensor_zenith_deg=90.0, source_range_km=15.0,
                                                   aerosol="none"),
                                     bands={"2500-2750": (2500.0, 2750.0)}, layers=1),
    # tests/test_physical_benchmarks.py::test_us_standard_10km_horizontal_transmission_benchmarks
    "us_std_10km_horizontal": dict(scenario=dict(atmosphere="us_standard", sensor_alt_km=0.01,
                                                 sensor_zenith_deg=90.0, source_range_km=10.0,
                                                 aerosol="rural", visibility_km=23.0),
                                   bands={"2550-2750": (2550.0, 2750.0),
                                          "2100-2170": (2100.0, 2170.0)}, layers=1),
    # tests/test_radiometry.py::test_lookdown_window_approaches_surface_emission
    "mls_8km_lookdown_sea": dict(scenario=dict(atmosphere="midlat_summer", sensor_alt_km=8.0,
                                               sensor_zenith_deg=180.0, aerosol="rural",
                                               surface_type="sea_water", surface_skin_temp_k=300.0,
                                               local_datetime=None, include_solar=False,
                                               include_lunar=False),
                                 bands={"2500-2857": (2500.0, 2857.0)}, layers=6, radiance=True),
}


def windows(lo: float, hi: float) -> list[tuple[float, float]]:
    starts = np.arange(lo, hi - WINDOW_WIDTH + 1e-9, WINDOW_SPACING)
    return [(float(a), float(a + WINDOW_WIDTH)) for a in starts]


def reference_path(sc, profile, n_layers: int):
    """Coarse Curtis-Godson layering of the scenario path (see module docstring)."""
    from mwir_atmos.geometry.path import build_path
    return build_path(profile, sensor_alt_km=sc.sensor_alt_km, zenith_deg=sc.sensor_zenith_deg,
                      source_range_km=sc.source_range_km, max_layers=n_layers,
                      uniform_fill=n_layers, surface_refinement_km=())


def _scenario(spec: dict, lo: float, hi: float):
    from mwir_atmos.types import (AerosolModel, ModelAtmosphere, Scenario, SpectralSampling,
                                  SurfaceType)
    kw = dict(spec)
    kw["atmosphere"] = ModelAtmosphere(kw["atmosphere"])
    kw["aerosol"] = AerosolModel(kw["aerosol"])
    if "surface_type" in kw:
        kw["surface_type"] = SurfaceType(kw["surface_type"])
    return Scenario(wavenumber_min_cm1=lo, wavenumber_max_cm1=hi, resolution_cm1=STEP,
                    spectral_sampling=SpectralSampling.POINT, **kw)


def _hapi():
    with contextlib.redirect_stdout(io.StringIO()):
        import hapi
    return hapi


def _load_table(hapi, species):
    """Put this package's line list for ``species`` into HAPI's table cache."""
    from mwir_atmos.hitran.fetch import ensure_linelists
    ll = ensure_linelists()[species]
    name = species.value
    if name in hapi.LOCAL_TABLE_CACHE:
        return name, ll
    n = len(ll)
    data = {
        "molec_id": np.full(n, ll.molecule_id, dtype=int), "local_iso_id": ll.local_iso_id.astype(int),
        "nu": ll.nu_cm1.astype(float), "sw": ll.s296.astype(float), "a": np.zeros(n),
        "gamma_air": ll.gamma_air.astype(float), "gamma_self": ll.gamma_self.astype(float),
        "elower": ll.e_lower.astype(float), "n_air": ll.n_air.astype(float),
        "delta_air": ll.delta_air.astype(float),
    }
    header = {"number_of_rows": n, "order": list(data), "format": {}, "default": {},
              "table_name": name, "size_in_bytes": -1, "table_type": "column-fixed"}
    hapi.LOCAL_TABLE_CACHE[name] = {"header": header, "data": {k: list(v) for k, v in data.items()}}
    return name, ll


def _pedestal(hapi, ll, nu_grid, p_atm, t_k, vmr):
    """MT_CKD pedestal: each H2O line's Lorentz value at 25 cm^-1, within +/-25 cm^-1."""
    t_ref = 296.0
    c2 = 1.4387769
    iso = ll.local_iso_id.astype(int)
    q = {i: hapi.partitionSum(ll.molecule_id, i, [t_k])[0] / hapi.partitionSum(ll.molecule_id, i, [t_ref])[0]
         for i in np.unique(iso)}
    ratio = np.array([q[i] for i in iso])
    s = (ll.s296 / ratio * np.exp(-c2 * ll.e_lower * (1 / t_k - 1 / t_ref))
         * -np.expm1(-c2 * ll.nu_cm1 / t_k) / -np.expm1(-c2 * ll.nu_cm1 / t_ref))
    gamma = (ll.gamma_air * p_atm * (1 - vmr) + ll.gamma_self * p_atm * vmr) * (t_ref / t_k) ** ll.n_air
    ped = s * gamma / (math.pi * (CUTOFF**2 + gamma**2))
    centre = ll.nu_cm1 + ll.delta_air * p_atm
    order = np.argsort(centre)
    cum = np.concatenate([[0.0], np.cumsum(ped[order])])
    c_sorted = centre[order]
    lo = np.searchsorted(c_sorted, nu_grid - CUTOFF, side="left")
    hi = np.searchsorted(c_sorted, nu_grid + CUTOFF, side="right")
    return cum[hi] - cum[lo]


def layer_tau(args):
    """HAPI line optical depth of one (layer, species), shape (n_grid,)."""
    species, p_mbar, t_k, vmr, column, lo, hi = args
    hapi = _hapi()
    name, ll = _load_table(hapi, species)
    nu = np.arange(lo, hi + 0.5 * STEP, STEP)
    isos = sorted(set(int(i) for i in ll.local_iso_id))
    comps = [(ll.molecule_id, i, hapi.ISO[(ll.molecule_id, i)][hapi.ISO_INDEX["abundance"]])
             for i in isos if (ll.molecule_id, i) in hapi.ISO]
    p_atm = p_mbar / 1013.25
    with contextlib.redirect_stdout(io.StringIO()):
        _, xsec = hapi.absorptionCoefficient_Voigt(
            Components=comps, SourceTables=name, WavenumberGrid=list(nu),
            Environment={"p": p_atm, "T": t_k}, Diluent={"air": 1.0 - vmr, "self": vmr},
            WavenumberWing=CUTOFF, WavenumberWingHW=0.0, IntensityThreshold=S_MIN,
            HITRAN_units=True)
    xsec = np.asarray(xsec)
    if species.value == "H2O":
        xsec = xsec - _pedestal(hapi, ll, nu, p_atm, t_k, vmr)
    return xsec * column


def run_case(name: str, case: dict, pool) -> dict:
    from mwir_atmos.api import build_profile
    from mwir_atmos.optics.continuum import co2_continuum_optical_depth, h2o_continuum_optical_depth
    from mwir_atmos.optics.transmittance import compute_layer_optics
    from mwir_atmos.radiometry.surface import surface_emissivity
    from mwir_atmos.spectral import planck
    from mwir_atmos.types import REQUIRED_SPECIES, SpectralGrid

    out = {}
    for band, (b_lo, b_hi) in case["bands"].items():
        sc = _scenario(case["scenario"], b_lo, b_hi)
        path = reference_path(sc, build_profile(sc), case["layers"])
        # One HAPI task per (window, layer, species).
        tasks, keys = [], []
        for w, (lo, hi) in enumerate(windows(b_lo, b_hi)):
            for row, layer in enumerate(path.layers):
                for sp in REQUIRED_SPECIES:
                    column = layer.column_cm2.get(sp, 0.0)
                    if column > 0:
                        tasks.append((sp, layer.p_eff_mbar.get(sp, layer.p_mbar),
                                      layer.t_eff_k.get(sp, layer.t_k),
                                      column / layer.air_column_cm2, column, lo, hi))
                        keys.append((w, row))
        taus = list(pool.map(layer_tau, tasks))
        per_window = []
        for w, (lo, hi) in enumerate(windows(b_lo, b_hi)):
            nu = np.arange(lo, hi + 0.5 * STEP, STEP)
            grid = SpectralGrid(wavenumber_cm1=nu, resolution_cm1=STEP)
            tau_lines = np.zeros((path.n_layers, nu.size))
            for (kw, row), tau in zip(keys, taus):
                if kw == w:
                    tau_lines[row] += tau
            # Shared, separately validated terms: continuum, aerosol, Rayleigh.
            tau_cont = (h2o_continuum_optical_depth(path.layers, grid)
                        + co2_continuum_optical_depth(path.layers, grid))
            optics = compute_layer_optics(path, grid, aerosol=sc.aerosol,
                                          visibility_km=sc.visibility_km,
                                          aerosol_winter=sc.atmosphere.value.endswith("winter"))
            tau = tau_lines + tau_cont + optics.tau_aerosol_abs + optics.tau_aerosol_sca + optics.tau_rayleigh
            t_layer = np.exp(-tau)
            t_total = np.prod(t_layer, axis=0)
            row = {"window_cm1": [lo, hi], "transmittance": float(t_total.mean()),
                   "molecular_transmittance": float(np.exp(-(tau_lines + tau_cont).sum(axis=0)).mean())}
            if case.get("radiance"):
                # Layers run sensor -> ground; t_near = transmittance sensor -> layer near edge.
                t_near = np.vstack([np.ones(nu.size), np.cumprod(t_layer, axis=0)[:-1]])
                b_layer = np.array([planck(nu, l.t_k) for l in path.layers])
                path_emission = (b_layer * (1 - t_layer) * t_near).sum(axis=0)
                eps = surface_emissivity(sc.surface_type)
                b_surf = planck(nu, sc.surface_skin_temp_k)
                # Downwelling at the surface along the diffusivity angle, reflected
                # (layers below the sensor only; the term is (1-eps) ~ 2% of a small one).
                t_diff = np.exp(-1.66 * tau[::-1])
                t_near_d = np.vstack([np.ones(nu.size), np.cumprod(t_diff, axis=0)[:-1]])
                down = (b_layer[::-1] * (1 - t_diff) * t_near_d).sum(axis=0)
                total = path_emission + t_total * (eps * b_surf + (1 - eps) * down)
                row.update(radiance=float(total.mean()), blackbody_300k=float(b_surf.mean()),
                           surface_emitted=float((t_total * eps * b_surf).mean()))
            per_window.append(row)
            print(name, band, row, flush=True)
        out[band] = {"n_layers": path.n_layers, "windows": per_window,
                     "mean_transmittance": float(np.mean([r["transmittance"] for r in per_window]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    hapi = _hapi()
    results = json.loads(OUT.read_text())["cases"] if OUT.exists() else {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for name in a.cases:
            results[name] = dict(CASES[name], results=run_case(name, CASES[name], pool))
    OUT.write_text(json.dumps({
        "generator": "scripts/physics_reference/generate_hapi_reference.py",
        "hapi_version": hapi.HAPI_VERSION, "fine_step_cm1": STEP, "line_cutoff_cm1": CUTOFF,
        "min_line_intensity": S_MIN, "cases": results}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
