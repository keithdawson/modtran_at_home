# Architecture

How `mwir_atmos` is put together, for someone about to change it. For the
current state of work and known traps read
[DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md); for the physics of spectral
sampling and the GPU backend, [spectral_sampling.md](spectral_sampling.md).

## Data flow

Every front end builds a `Scenario` (`types.py`) and calls one function,
`api.run_scenario`, which returns a `ScenarioResult`:

```
Scenario
  -> api.build_profile          model atmosphere + surface overrides   (atmosphere/)
  -> api.build_solver_geometry  sightline, plus a vertical column for
                                DISORT / spherical Monte Carlo        (geometry/)
  -> api._SpectralPass          on one spectral grid:
       optics.transmittance.compute_layer_optics
         lbl (HITRAN line-by-line) + continuum + aerosol + rayleigh   (optics/, hitran/)
       optics.transmittance.path_transmittance
       api._solve_radiance      single scatter | DISORT | Monte Carlo (radiometry/, ephemeris/)
  -> turbulence.compute_turbulence (diagnostics only)                (turbulence/)
ScenarioResult
```

Point sampling runs the spectral pass once on the output grid. Resolved
sampling (the default) runs it on fine-grid chunks — in worker processes on
the CPU backend, device-resident on the CUDA backend — and reduces each chunk
through the instrument function (`resolved.py`, `api._run_resolved`).

## Package map

| Module | Role |
|---|---|
| `types.py` | Every cross-module dataclass and enum (`Scenario`, `PathGeometry`, `LayerOptics`, `RadianceResult`, ...) |
| `constants.py`, `spectral.py` | Units, physical constants, named bands; grids, Planck, band averaging |
| `validation.py` | `validate_scenario`: types, nullability and ranges, shared by every entry point |
| `api.py` | `run_scenario` orchestration, resolved-chunk execution (processes / GPU) |
| `resolved.py` | Fine-grid plan, instrument functions, chunking, correlated-k binning |
| `hitran/` | Line lists (`cache.py`), the only network code (`fetch.py`), partition sums (`tips.py`) |
| `optics/lbl.py` | Voigt line-by-line engine: line preparation, selection, direct and multigrid sums |
| `optics/lbl_cuda_dispatch.py` | CUDA backend orchestration: GPU line selection (default) or grouped CPU selection overlapping the kernels |
| `optics/lbl_cuda.py`, `optics/voigt_gpu.py` | Optional CUDA kernel (CuPy) and its NumPy emulation; the kernel's Voigt profile |
| `optics/lbl_cuda_select.py` | Fused CUDA kernels for GPU line selection (also compiled as host C++ in the tests) |
| `solver_pool.py` | DISORT / Monte Carlo spectral points over persistent CPU worker processes |
| `multi_gpu.py` | Resolved CUDA chunks over several GPUs (`MWIR_CUDA_DEVICES`): persistent spawned worker pool |
| `optics/continuum.py`, `aerosol.py`, `lowtran_aerosol.py`, `rayleigh.py` | Continuum, aerosol and Rayleigh optics |
| `optics/transmittance.py` | Assembles layer optics; path transmittance |
| `atmosphere/` | Six AFGL model atmospheres; surface temperature/humidity overrides |
| `geometry/` | Spherical slant paths, plane-parallel and spherical-shell geometries, refraction |
| `radiometry/` | Single-scatter radiance, DISORT (vendored `_vendor/pythonic_disort`), spherical Monte Carlo, polarization, surface emissivity |
| `ephemeris/` | Low-precision sun and moon positions |
| `turbulence/` | Cn² profiles and Fried / isoplanatic / scintillation metrics |
| `scenario_io.py` | `Scenario` <-> dict / YAML (CLI `--config`, GUI setups, `setups.py`) |
| `run_cost.py` | Size / memory / runtime estimate and the confirmation threshold (CLI and GUI) |
| `report.py` | Band summaries; table, CSV, JSON and NPZ writers |
| `setups.py` | Script-facing helpers: load a setup with overrides, distance sweeps |
| `cli.py` | `mwir` command: argparse, flag resolution, confirmation prompt |
| `webgui/` | Streamlit app: `app.py` (entry, cache, cost gate), `sidebar.py`, `results.py`, `sweep.py`, `plots.py` |
| `testing.py` | Synthetic paths for tests only; never imported by production code |
| `_array.py`, `data_paths.py` | NumPy/CuPy dispatch; locating shipped data tables |

Other top-level directories: `data/` (shipped HITRAN, MT_CKD and aerosol
tables — the runtime never downloads), `scenarios/` (example YAML),
`scripts/validation/` (validation studies writing into `docs/validation/`),
`scripts/physics_reference/` (independent reference generators), and the
deployment files at the repository root (`Dockerfile.*`, `install_offline.*`,
`OFFLINE_DEPLOYMENT.md`).

## Where to make common changes

- **A new `Scenario` field**: add it to `types.py`; add a flag in
  `cli.build_parser` (`tests/test_cli.py` fails until every field has one); add a
  GUI control in `webgui/sidebar.py` if users should set it; add a range check
  to `validation.py` if it has one. YAML support is automatic.
- **A new output column or format**: `report.py`.
- **A new radiance solver**: an enum member in `RadianceSolver`, a geometry
  branch in `api.build_solver_geometry` if it needs a column, a branch in
  `api._solve_radiance`, and its cost in `run_cost.estimate_run_cost`.
- **Performance constants** sit next to the code they tune, each with the
  measurement that set it (`lbl._BATCH_ELEMENTS`, `api._CUDA_CHUNK_ELEMENTS`,
  `run_cost._COST_*`, ...). Change them with a new measurement, recorded in
  `docs/spectral_sampling.md` or `docs/validation/`.

## Invariants

- The default grid is 1800-4300 cm⁻¹ and must keep covering 3-5 µm with margin.
- Spectral radiance is W cm⁻² sr⁻¹ / cm⁻¹ internally; convert only for display.
- Runtime code never requires the network: line data ships in `data/`.
- Physics thresholds in tests come from independent references
  (`scripts/physics_reference/`), never from the model's own new output.

The original specification the package was built against, including the
physics requirements many docstrings cite, is
[build_contract.md](build_contract.md).
