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
  alone is not enough there.
- **Model atmospheres**: the six standard AFGL/LOWTRAN/MODTRAN profiles
  (tropical, mid-latitude summer/winter, subarctic summer/winter, US
  Standard), transcribed from AFGL-TR-86-0110. See
  `mwir_atmos/atmosphere/profiles.py`.
- **Geometry**: spherical-Earth slant-path ray tracing (upward, downward,
  horizontal, and limb/tangent paths, ground intersection, and a finite
  source range or altitude), with Curtis–Godson effective pressure/temperature
  per layer per species. See `mwir_atmos/geometry/path.py`.
- **Aerosols and Rayleigh scattering**: LOWTRAN-style rural/urban/navy-maritime/
  desert/tropospheric aerosol models (Koschmieder visibility relation,
  Ångström spectral falloff, Hänel hygroscopic growth) and a standard
  molecular-scattering Rayleigh formula. See `mwir_atmos/optics/aerosol.py`
  and `mwir_atmos/optics/rayleigh.py`.
- **Radiance and ephemeris** (`mwir_atmos/radiometry/`, `mwir_atmos/ephemeris/`):
  path thermal emission, surface emission/reflection, solar/lunar
  scattering and direct terms, and a discrete source's own emission, using a
  deliberately "loose" (not ephemeris-grade) sun/moon position. See
  **Limitations** below for exactly what "loose" means and whether this
  layer is present in your build.

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
and ship with the repository — normal runs never touch the network
(`mwir_atmos.hitran.fetch.ensure_linelists` is the only function permitted
to fetch, and only if the cache is missing). Nothing extra to download for
an air-gapped install.

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
2. **The line-wing cutoff** (`line_wing_cutoff_cm1`, default 25 cm⁻¹) —
   every surviving spectral line is evaluated over a window of
   `2*line_wing_cutoff_cm1/resolution_cm1 + 1` grid points, **independent of
   how narrow the requested band is**. At the default 25 cm⁻¹ cutoff and
   0.001 cm⁻¹ resolution, that window alone is ~50,000 points *per line* —
   measured directly, a 2 cm⁻¹-wide window at that resolution took anywhere
   from ~50 s to ~160 s depending on how many real HITRAN lines happened to
   fall nearby, *before* narrowing the cutoff. Shrinking
   `line_wing_cutoff_cm1` to match the window (a few cm⁻¹, as
   `fine_resolution_co_line.yaml` does) is what actually makes a narrow,
   fine-resolution run fast — narrowing the wavenumber window alone is not
   sufficient on its own if the wing cutoff is left at its full-band
   default. Both CLI and GUI cost estimates account for this.

`line_strength_cutoff` (default `1e-4`) is the other accuracy control: it
drops lines weaker than that fraction of the strongest line in each
(layer, species) evaluation, which is a good tradeoff for full-band runs
near 1 cm⁻¹ but **can visibly distort an individual line core** at fine
resolution even though it barely moves a band average. For narrow,
fine-resolution work, lower it to `1e-6` or `1e-7`; `0.0` keeps every line
(exact, but slow across a full-width path).

Practical guidance: if you need fine resolution, narrow the *wavenumber
window* to what you actually need rather than reducing resolution over the
full band, and lower `line_wing_cutoff_cm1` to roughly match that window's
width.

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

- **Aerosols are a parametric model**, not a transcription of
  LOWTRAN/MODTRAN's digitized Mie-scattering lookup tables. Extinction
  shape (Koschmieder + Ångström spectral falloff + Hänel hygroscopic
  growth) is physically grounded and cited to the literature, but the
  per-type numeric endpoints (Ångström exponent, single-scattering albedo,
  asymmetry parameter) are representative values, not digitized table
  lookups. See `mwir_atmos/optics/aerosol.py`.
- **Refraction is omitted.** Rays are geometric straight lines through the
  spherical atmosphere; real rays curve slightly toward the surface
  (refraction adds on the order of a percent to airmass near the horizon).
  See `mwir_atmos/geometry/path.py`.
- **The CO₂ continuum is a documented zero stub** —
  `mwir_atmos.optics.continuum.co2_continuum_optical_depth` always returns
  zero. This is a small correction relative to the CO₂ line database
  already used (the 4.2–4.4 µm band is already essentially opaque from
  lines alone), not a missing headline effect, but it means CO₂-band
  numbers slightly underestimate absorption relative to a full continuum
  treatment.
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
- **Rayleigh scattering** uses a constant (not spectrally-resolved)
  King depolarization factor — an excellent approximation this far from
  any electronic resonance, and negligible in the MWIR either way
  (τ ≈ 1–9×10⁻⁵ for a full sea-level column, vs. ≈0.09 at 550 nm).
- **The MT_CKD continuum** is linearly interpolated onto the working grid
  from AER's reference table, a simplification of MT_CKD's own
  cubic-Hermite interpolation (the continuum is smooth by construction, so
  this is a minor difference).
- **Turbulence** (`mwir_atmos/turbulence/`) is an explicit wave-3 stretch
  goal and is not built — `TurbulenceModel` exists in `types.py` and
  `Scenario.turbulence` can be set, but `ScenarioResult.turbulence` will
  always be `None` until that module exists; neither the CLI nor the GUI
  expose turbulence controls for this reason.
- **CO₂/N₂O/CO/CH₄ are well-mixed background values**
  (`constants.BACKGROUND_VMR_PPMV`, a 2025-era estimate for CO₂), applied
  uniformly with altitude and independent of the scenario's date — there is
  no historical/future CO₂ trend model.
- **Scattering is single-scatter only.** Solar and lunar photons are allowed
  exactly one scattering event. This is a good approximation in the MWIR,
  where scattering optical depths are small next to molecular absorption, but
  it under-predicts diffuse illumination in heavy aerosol at short MWIR
  wavelengths. Illumination-beam transmittance is also reconstructed by
  de-slanting the line-of-sight layers to vertical and re-slanting along the
  sun/moon direction with a Kasten-Young airmass — exact for a
  plane-parallel atmosphere, approximate otherwise, and blind to horizontal
  inhomogeneity.

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
