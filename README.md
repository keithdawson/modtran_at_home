# mwir_atmos — MODTRAN-at-home

A mid-wave infrared (MWIR, ~2.3–5.6 µm) atmospheric transmittance and
radiance calculator: a line-by-line physics core, a scriptable command-line
interface, and a Streamlit web GUI, built to run entirely offline once the
HITRAN line data is cached.

Give it a model atmosphere, a sensor/source geometry, and a spectral window,
and it computes sensor-to-source transmittance (total, molecular vs. aerosol
vs. Rayleigh, and per-species) and, once the radiometry layer is present,
full spectral radiance decomposed by physical origin (path thermal emission,
surface emission and reflection, solar/lunar scattering, a discrete source's
own emission).

## Physics basis

- **Molecular absorption**: HITRAN line-by-line data (HITRAN2024 line
  lists, fetched once from hitran.org and cached under `data/hitran/` as
  `.npz` files that ship with the repo — no network access at runtime),
  evaluated as Voigt profiles (`scipy.special.wofz`) with the full HITRAN
  temperature-scaling relation, pressure broadening/shift, and per-line
  isotopologue mass for the Doppler width. See
  `mwir_atmos/optics/lbl.py`.
- **Water vapor continuum**: MT_CKD 4.3 (Mlawer–Tobin–Clough–Kneizys–Davies),
  self + foreign broadened, from AER's reference coefficient table (cached
  locally, see `mwir_atmos/optics/continuum.py`). This matters a lot in the
  3.7–4.0 µm and 4.5–5.0 µm atmospheric windows — the line-by-line engine
  alone is not enough there. Local water lines use the matching +/-25 cm^-1
  pedestal subtraction. See [reference checks](PHYSICS_VALIDATION.md).
- **Model atmospheres**: the six standard AFGL/LOWTRAN/MODTRAN profiles
  (tropical, mid-latitude summer/winter, subarctic summer/winter, US
  Standard), transcribed from AFGL-TR-86-0110. See
  `mwir_atmos/atmosphere/profiles.py`.
- **Geometry**: spherical-Earth slant-path ray tracing (upward, downward,
  horizontal, and limb/tangent paths, ground intersection, and a finite
  source range or altitude), with Curtis–Godson effective pressure/temperature
  per layer per species. See `mwir_atmos/geometry/path.py`.
- **Aerosols and Rayleigh scattering**: original LOWTRAN 7 table options for
  rural/urban/maritime/tropospheric aerosols, with humidity-dependent optical
  properties and background altitude profiles. Legacy parametric models remain
  available. Molecular Rayleigh scattering is calculated separately. See
  [aerosol data and validation](docs/lowtran_aerosols.md).
- **Radiance and ephemeris**: the legacy single-scattering solver and an optional
  **DISORT multiple-scattering** solver. DISORT includes full-column thermal,
  solar and lunar transfer with repeated Lambertian ground reflection, using
  consistent plane-parallel paths. Supports saved setups and finite targets.
  A separate experimental **spherical Monte Carlo** solver traces all photon
  paths through curved shells and reports sampling uncertainty.
  See [DISORT](docs/multiple_scattering.md) and [spherical transfer](docs/spherical_scattering.md).

See **Limitations** below before trusting any specific number this tool
produces — it is a from-scratch reconstruction of MODTRAN/LOWTRAN-style
physics, not a validated replacement for either.

## Install

Requires Python ≥3.10 (developed and tested on 3.11.6, numpy 2.3.3, scipy
1.16.2). Core dependencies (`numpy`, `scipy`, `pyyaml`) are enough for the
physics core and the CLI; the GUI needs `streamlit` and, ideally, `plotly`.

```bash
# Core + CLI
pip install -e .

# + the web GUI
pip install -e ".[gui]"

# + the test suite
pip install -e ".[dev]"
```

This registers the `mwir` console script (`pyproject.toml`
`[project.scripts]`). If you'd rather not install the package, everything
below also works by running from the repository root with the project on
`PYTHONPATH`:

```bash
# Windows PowerShell
$env:PYTHONPATH = "path\to\modtran_at_home"
python -m mwir_atmos.cli --help

# POSIX
PYTHONPATH=path/to/modtran_at_home python -m mwir_atmos.cli --help
```

**Offline data.** HITRAN line lists, the TIPS partition-function table, and
the MT_CKD continuum coefficients are pre-fetched into `data/hitran/*.npz`
and ship with the repository and built wheels — normal runs never touch the network.
Missing line lists raise an actionable error; downloading requires an explicit
`ensure_linelists(allow_download=True)` call on a connected machine (`force=True`
explicitly refreshes the cache). Nothing extra to download for
an air-gapped install.

### Reproducible verification

```bash
python -m pytest -q
python scripts/verify_examples.py
```

The second command runs all five shipped scenarios at their declared resolution,
checks finite/bounded transmittance and radiance component sums, and writes JSON,
CSV, and PNG output to `artifacts/examples/`. The test suite includes actual
Streamlit reruns, display-unit switching, input validation, and offline-cache
failure checks. GitHub Actions runs these on Windows and Linux and retains the
outputs as downloadable artifacts.

Scenario inputs reject nonfinite numbers, invalid physical ranges and quoted
booleans (use YAML `false`, not `"false"`). Wavelength radiance plots use
W m^-2 sr^-1 / um; wavenumber plots and spectral CSV exports retain the internal
W cm^-2 sr^-1 / cm^-1 convention. Changing display units never changes the physics.

Wheel installs place the eight scientific tables under
`<environment>/share/mwir-atmos/hitran`; source/editable installs use `data/hitran`.
The offline `wheels/` dependency cache and generated `dist/` bundles are excluded
from Git. Prepare dependencies with `scripts/download_wheels.py` on a connected
machine before building an air-gap bundle. A failed download now exits with an error.

## Command line

```bash
mwir --atmosphere tropical --aerosol navy_maritime --visibility-km 10 \
     --sensor-zenith-deg 100 --format table
```

Every field of `mwir_atmos.types.Scenario` is settable as a flag, grouped in
`mwir --help` as: **configuration** (`--config`/`--save-config`),
**atmosphere**, **sensor**, **source**, **illumination**, **spectral**,
**accuracy**, **turbulence** (stretch goal, currently a no-op — see
Limitations), and **output**.

- **`--config scenario.yaml`** loads a `Scenario`; any other flag on the
  command line overrides that field. **`--save-config out.yaml`** writes the
  fully-resolved scenario (config file + overrides) back out — useful for
  capturing exactly what a run used. See `scenarios/*.yaml` for worked
  examples, and `mwir_atmos.cli.scenario_to_dict`/`scenario_from_dict` if
  you want to build scenario files programmatically.
- **`--format {table,csv,json,npz}`**. `table` is a human-readable summary.
  `csv`/`json`/`npz` all carry the **full spectral arrays** — wavenumber,
  wavelength, transmittance (total/molecular/aerosol/Rayleigh and every
  species), and every radiance component once available — not just band
  summaries, so you can pipe a run straight into your own analysis.
- **`--bands NAME_OR_MIN:MAX ...`** reports band-averaged transmittance and
  band-integrated radiance over named bands from `constants.BANDS_CM1`
  (`--list-bands` to see them) and/or custom `'2600:2650'` cm⁻¹ ranges, e.g.
  `--bands CO2_4_3um 2600:2650`. Only affects the `table`/`json` summary
  tables — csv/npz's per-wavenumber arrays are always the full grid.
- **`-q`/`--quiet`** and **`-v`/`--verbose`** control stderr diagnostics
  (stdout stays clean for piping table/csv/json output). Exit codes: `0`
  success, `1` a runtime error (bad config, a geometrically invalid
  scenario, a write failure), `2` a usage error (bad flag/value). `mwir
  --version` and `python -m mwir_atmos.cli` both work; `main(argv=None)` is
  importable and returns the exit code, so it's directly testable (see
  `tests/test_cli.py`).

```bash
# Load a worked example, override resolution, get the full spectrum as CSV.
mwir --config scenarios/horizontal_haze.yaml --resolution-cm1 2.0 \
     --format csv --output run1.csv

# Resolve a scenario and capture exactly what was run, without redoing it.
mwir --config scenarios/limb_view.yaml --sensor-alt-km 22 \
     --save-config resolved.yaml --quiet --format npz -o run.npz
```

**Radiance.** `mwir_atmos.radiometry` and `mwir_atmos.ephemeris` are present,
so `ScenarioResult.radiance`/`.sun`/`.moon` are populated and every CLI writer
emits the full component breakdown. The degradation path is still live: if
those modules are ever absent, `mwir_atmos.api.run_scenario` returns a
transmittance-only result rather than raising, and the writers report
"radiance unavailable" instead of fabricating a number.

For an accurate sun, pass observed sunrise and sunset — see **Limitations**;
without them the timezone is guessed from longitude and can be tens of degrees
wrong:

```bash
mwir --atmosphere midlat_summer --latitude-deg 40.42 --longitude-deg -3.70 --local-datetime 2026-06-21T09:00 --sunrise-local-hours 6.733 --sunset-local-hours 21.800
```

## Web GUI

```bash
streamlit run mwir_atmos/webgui/app.py
```

The sidebar builds a `Scenario` (same grouping as the CLI); the main panel
shows a scenario summary, transmittance vs. wavelength with a toggleable
total/molecular/aerosol/Rayleigh breakdown, a per-species breakdown,
radiance vs. wavelength (stacked by component, once available), the
atmospheric temperature and water-vapor profile, and a band summary table —
each chart has a CSV and/or PNG download button. The x-axis defaults to
wavelength in µm (a wavenumber toggle is next to it) over the package's
default 2.33–5.56 µm grid, which keeps the 3–5 µm MWIR core comfortably in
view along with the CO₂ 4.3 µm and H₂O 2.7 µm band wings.

The GUI defaults to a coarser resolution (4 cm⁻¹) than the CLI/`Scenario`
default (1 cm⁻¹) so that changing a widget stays interactive; a row of
spectral-range presets (**Full band**, **MWIR core 3–5 µm**, **Narrow + fine
CO line demo**) makes picking a fast narrow window one click, and a live
cost estimate plus a confirmation checkbox (mirroring the CLI's `--yes` gate
exactly — same thresholds, same estimator) sits in front of any run that
would be expensive. Recomputation is cached on the resolved scenario (see
"Choosing spectral resolution" below and the module docstring of
`mwir_atmos/webgui/app.py` for exactly how — `Scenario`'s enums/datetime
need a bit of care to make hashable for `st.cache_data`); toggling a
display-only control (units, which components to plot, which bands to
tabulate) never re-runs the physics.

## Worked example scenarios

`scenarios/*.yaml` — load any of these with `--config` (CLI) or as a
starting point for hand-written configs; each file's header comment
explains what it demonstrates and why its geometry was chosen:

| File | Demonstrates |
|---|---|
| `horizontal_haze.yaml` | Low-altitude 5 km horizontal path through urban haze — aerosol/visibility dominating the atmospheric windows. |
| `limb_view.yaml` | A 50 km-altitude limb-sounding path grazing a ~20 km tangent altitude without hitting the ground. |
| `hot_target_radiance.yaml` | A discrete hot ground target (`source_*` fields) viewed obliquely at mid-day — every radiance component applies here once that layer lands. |
| `fine_resolution_co_line.yaml` | The narrow-window/fine-resolution use case: a single resolved CO line, with the accuracy controls tuned for it. |
| `tropical_maritime_quicklook.yaml` | A shipboard sensor in humid tropical air with a wind-driven navy-maritime aerosol loading. |

## Choosing spectral resolution

`resolution_cm1` is **never capped or silently clamped** — set it as fine as
your data needs. The tradeoff is cost, and cost is pushed onto you as
*information* rather than a hidden limit: both the CLI and the GUI print an
estimated point count, estimated layer-array memory, and an estimated
runtime **before running**, and refuse to proceed past roughly **1 GB or 3
minutes** without explicit confirmation (`--yes` on the CLI; a checkbox in
the GUI).

**Cost is not simply "resolution".** It is driven by two independent
things:

1. **Grid size** — points × layers. Points scale as bandwidth ÷ resolution,
   so a *narrow* window is cheap even at very fine resolution: a 2.5 cm⁻¹
   window around a single CO line (`scenarios/fine_resolution_co_line.yaml`)
   at 0.001 cm⁻¹ resolution (2501 points) measured **~20–30 s and ~14 MB**
   on the development machine. The full default 1800–4300 cm⁻¹ band at the
   same 0.001 cm⁻¹ resolution is the expensive case this package protects
   against: **~2 hours and ~13 GB** of layer arrays (60 layers × 2.5M
   points × 12 float64 arrays).
2. **The line-wing cutoff** (`line_wing_cutoff_cm1`, default 25 cm^-1) sets
   the local evaluation width for gases other than water. H2O with MT_CKD
   always uses +/-25 cm^-1 and subtracts the Lorentz pedestal already included
   in the continuum. Evaluation windows are clipped to the requested grid
   before allocation, so a narrow grid now limits the per-line working array.
   The runtime estimator retains its older, conservative wing-width calibration.

`line_strength_cutoff` (default `1e-4`) is the other accuracy control: it
drops lines weaker than that fraction of the strongest line in each
(layer, species) evaluation, which is a good tradeoff for full-band runs
near 1 cm⁻¹ but **can visibly distort an individual line core** at fine
resolution even though it barely moves a band average. For narrow,
fine-resolution work, lower it to `1e-6` or `1e-7`; `0.0` keeps every line
(exact, but slow across a full-width path).

Practical guidance: if you need fine resolution, narrow the *wavenumber
window* to what you actually need rather than reducing resolution over the
full band. Reducing the wing cutoff changes the absorption model for other
gases; use convergence checks before relying on a smaller cutoff.

Both estimates are calibrated directly against measurements on this
codebase (see `mwir_atmos/cli.py`'s `_estimate_run_cost` docstring for the
exact calibration and its data points) and are explicitly an
order-of-magnitude guide, not a scheduler — real cost also depends on how
many HITRAN lines happen to populate the requested window, which is not
knowable without loading the line data (the expensive step the estimate is
trying to help you avoid paying for blindly).

## Limitations

This is a from-scratch reconstruction of MODTRAN/LOWTRAN-style physics for
offline use, not a drop-in replacement for either. Specifically, per the
physics modules' own docstrings:

- **Aerosols offer two backends.** Existing names preserve the legacy parametric
  approximation. `lowtran_rural`, `lowtran_urban`, `lowtran_maritime`, and
  `lowtran_tropospheric` use original LOWTRAN 7 optical tables and background
  altitude profiles, checked against its Fortran routines. Select **LOWTRAN
  table:** in the GUI or use the same value in YAML/CLI. These options support
  visibility 2–50 km and aerosol wavelengths 0.2–40 µm; this does not extend the
  validated spectral range of the gas/radiance models. See
  [data sources, validation and limitations](docs/lowtran_aerosols.md).
- **Refraction is omitted.** Rays are geometric straight lines through the
  spherical atmosphere; real rays curve slightly toward the surface
  (refraction adds on the order of a percent to airmass near the horizon).
  See `mwir_atmos/geometry/path.py`.
- **CO2 continuum, collision-induced absorption and line mixing are missing.**
  `co2_continuum_optical_depth` returns zero. An opaque band core does not
  establish accuracy in the wings: the 4.17 micrometer edge is sensitive
  to the line cutoff, and its error is not bounded by the present tests.
- **The ephemeris is loose by design, and the timezone is the weak link.**
  The underlying algorithm (Spencer 1971) is good to ~0.4° in declination and
  ~0.03° in hour angle. What dominates the error is the *timezone*: with no
  `utc_offset_hours`, it is guessed as `round(longitude / 15)`, and real
  timezones are political. Measured against the true offset on 2026-06-21:

  | Site | tz guess error | with sunrise/sunset |
  |---|---|---|
  | Perth | 0.00° | — |
  | Denver | 13.75° | 0.09° |
  | Madrid | 27.47° | 0.04° |
  | Kashgar | 41.11° | 3.64° |

  Madrid keeps Central European Time despite sitting on the Greenwich
  meridian; Kashgar runs on Beijing time three hours east of its solar
  longitude. **Supply `sunrise_local_hours` and `sunset_local_hours`** (or
  `--sunrise-local-hours` / `--sunset-local-hours`) to remove this entirely:
  their midpoint is local solar noon on the observer's own clock, which
  fixes the offset exactly, and their separation gives the day length, which
  pins the declination. With both supplied the declared timezone is ignored
  and position is good to ~0.1°. Lunar position is derived kinematically from
  the sun plus the phase and ignores the 5.14° lunar orbital inclination, so
  it stays a few degrees loose regardless.
- **Boundary-layer resolution is finite.** The path is refined near the
  surface (10 m at the ground, relaxing to the native 1 km profile grid by
  6 km — see `geometry.path.DEFAULT_SURFACE_REFINEMENT_KM`). This matters
  because a surface sensor viewing an opaque band sees essentially only the
  air immediately in front of it: with the AFGL grid's 1 km bottom slab, the
  retrieved brightness temperature under-read the true surface air
  temperature by 2.3 K (8.4% in radiance). With refinement that falls to
  0.36%. Going finer still (2 m) buys almost nothing, so the default sits at
  the point of diminishing returns; it costs ~25% more runtime than the
  unrefined grid.
- **Rayleigh scattering** uses a dry-air approximation with the standard-air
  reference density and wavelength-dependent King factors. It agrees within
  0.22% with Tomasi et al.'s moist-air cross-sections over 0.5-4 micrometers;
  humidity and composition do not update refractivity or the King factor.
- **The MT_CKD continuum** follows AER 4.3 native-grid scaling, radiation
  factors and cubic-Hermite interpolation. Numerical agreement with AER's
  Fortran kernel does not validate the complete radiative-transfer model.
  See [Physics validation](PHYSICS_VALIDATION.md) for external reference
  data, tolerances, reproducible calculations and remaining priorities.
- **Turbulence** (`mwir_atmos/turbulence/`) is an explicit wave-3 stretch
  goal and is not built — `TurbulenceModel` exists in `types.py` and
  `Scenario.turbulence` can be set, but `ScenarioResult.turbulence` will
  always be `None` until that module exists; neither the CLI nor the GUI
  expose turbulence controls for this reason.
- **CO₂/N₂O/CO/CH₄ are well-mixed background values**
  (`constants.BACKGROUND_VMR_PPMV`, a 2025-era estimate for CO₂), applied
  uniformly with altitude and independent of the scenario's date — there is
  no historical/future CO₂ trend model.
- **Scattering has three solver options.** The default preserves the legacy
  single-scattering calculation and spherical paths. Optional `disort` solves
  all scattering orders in a plane-parallel column with repeated Lambertian
  surface reflection. It uses scalar HG/Rayleigh phase moments, not polarization
  or a full Mie phase function. Experimental `spherical_mc` traces all scattering
  orders in concentric shells, with reported Monte Carlo uncertainty; it still
  omits polarization and refraction. See [DISORT limits](docs/multiple_scattering.md)
  and [spherical solver validation](docs/spherical_scattering.md).

Do not treat any single number from this tool as validated against a real
MODTRAN/LOWTRAN run; treat it as a physically-grounded estimate with the
above caveats.

## Testing

```bash
pytest tests/ -q
```

`tests/test_cli.py` covers `main()` across a range of argv (every output
format, `--config`/`--save-config` round-tripping, the confirmation gate's
both outcomes, every documented error path returning the right non-zero
exit code) and unit-tests the scenario (de)serialisation and band-token
parsing helpers directly. It shares one narrow (10 cm⁻¹), coarse (2 cm⁻¹)
scenario across the handful of tests that need to run real physics, to keep
the suite fast.

Other test files exercise the physics core directly: `test_hitran.py`,
`test_lbl.py`, `test_continuum.py` (line-by-line engine and HITRAN data),
`test_profiles.py`, `test_geometry.py`, `test_aerosol.py` (model
atmospheres, slant-path geometry, aerosol optics).

## Project layout

See `CONTRACT.md` for the full package layout, file ownership, and the
binding interfaces between the physics core, the CLI, and the GUI. In
short: `mwir_atmos/{hitran,optics,atmosphere,geometry}` are the physics
core; `mwir_atmos/{radiometry,ephemeris}` are the radiance layer;
`mwir_atmos/{types,spectral,api}.py` are the shared contracts every other
module is built against; `mwir_atmos/cli.py` and `mwir_atmos/webgui/` are
the two front ends, both calling nothing but `mwir_atmos.api.run_scenario`.

## Saved setups and distance sweeps

In the app, expand **Save setup / use from a script** and choose **Download
setup (YAML)**. The file contains all active scenario inputs and uses the same
format as CLI `--config` / `--save-config`. To reopen it, choose **Saved YAML**
in the sidebar. The optional **Overrides (YAML)** mapping changes only named
fields; `null` clears an optional field. Current controls are hidden while a
saved file is active, so they do not silently replace its settings.

Python scripts can load a setup once and vary just a few inputs:

```python
from mwir_atmos.setups import load_setup, with_overrides
from mwir_atmos.api import run_scenario

base = load_setup("mwir_setup.yaml")
trial = with_overrides(base, source_range_km=5.0, source_alt_km=None)
result = run_scenario(trial)
```

`load_setup(path, visibility_km=10.0)` also accepts overrides directly.
`save_setup(scenario, path)` writes all resolved inputs. Overrides are validated,
unknown fields fail clearly, and the original object/file remains unchanged.
`run_setup(path, overrides={...})` combines loading and running.

Choose **Calculation -> Distance sweep** to increase slant range until the
mean transmittance in a chosen band drops below a threshold. Set the start,
maximum, step and band, then click **Run distance sweep**. The chart and CSV
report completed samples; normal app reruns reuse the result. The maximum
endpoint is included even when it is not aligned with the step. No crossing
means only that the threshold was not crossed within the tested distances.

For command-line use:

```sh
python scripts/distance_sweep.py mwir_setup.yaml --start 0.1 --stop 10 --step 0.5 --threshold 0.1 --output sweep.csv
```

For your own detectability/SNR calculation, supply a Python callback:

```python
from mwir_atmos.setups import load_setup, sweep_distance
from my_detector import calculate_snr  # Your target/background/sensor model.

base = load_setup("mwir_setup.yaml")
sweep = sweep_distance(
    base,
    [0.5 * i for i in range(1, 101)],
    metric=calculate_snr,  # Receives the full ScenarioResult at each distance.
    threshold=5.0,
    include_radiance=True,
)
print(sweep.last_passing_distance_km, sweep.first_failing_distance_km)
```

Equality passes; the first strictly lower score stops the loop. A custom
`on_sample` callback can save intermediate results. The sweep stores scalar
samples rather than every spectrum. It does not assume that arbitrary custom
metrics are monotonic or claim a global maximum range.

Distance controls source placement along the saved sensor direction, so sweeps
clear `source_alt_km` and replace `source_range_km`. All other scenario inputs
are retained unless overridden. If the ray hits ground or leaves the modeled
atmosphere before the requested distance, the sweep raises instead of silently
labeling a clipped path with the requested distance. Band transmittance is
atmospheric throughput, **not a detector detectability model**; use your own
metric for target contrast, sensor noise and detection thresholds. Saved YAML
contains simulation inputs; GUI sweep limits and metric settings are supplied
separately when running the sweep.
