# mwir_atmos — build contract

MODTRAN-at-home: a mid-wave infrared atmospheric transmittance and radiance
calculator with a scriptable CLI and a Streamlit web GUI.

This file is the coordination point for parallel work. **Read it fully before
editing anything.** Two coding workers run concurrently against it.

---

## Hard invariants (do not negotiate these)

1. **Spectral coverage.** The working grid MUST cover at least
   **2000–3333.33 cm⁻¹ (3–5 µm)**. The package default is **1800–4300 cm⁻¹
   (2.33–5.56 µm)**, deliberately wider so the CO₂ 4.3 µm and H₂O 2.7 µm band
   wings are resolved rather than clipped at a grid edge. **Never narrow the
   default.** HITRAN fetches must span the grid plus the line-wing cutoff
   margin — fetch **1775–4325 cm⁻¹**.
2. **Units.** As declared in `constants.py`. Spectral radiance is
   `W cm⁻² sr⁻¹ / cm⁻¹` (MODTRAN convention) everywhere internally; convert
   only at the display layer.
3. **Shared types.** Every cross-module value is a dataclass from `types.py`.
   Do not invent a parallel representation.
4. **Overseer-owned files.** `constants.py`, `types.py`, `spectral.py`,
   `testing.py`, `api.py`, `CONTRACT.md`, `pyproject.toml`. **Workers must not
   edit these.** Need a change? Implement around it and say so in your report.
5. **Offline-capable end state.** The final package must run with no network.
   HITRAN data is downloaded *now*, during development, and cached into
   `data/hitran/` as versioned `.npz` files that ship with the repo. Runtime
   code must never require a live fetch — it loads the cache, and only falls
   back to fetching if explicitly asked.
6. **No git operations.** The overseer owns version control.

---

## Package layout and file ownership

```
modtran_at_home/
  pyproject.toml                    OVERSEER
  CONTRACT.md                       OVERSEER
  data/hitran/                      dev-alpha  (generated cache, committed)
  mwir_atmos/
    __init__.py                     OVERSEER
    constants.py                    OVERSEER
    types.py                        OVERSEER
    spectral.py                     OVERSEER   grid, Planck, band integration
    testing.py                      OVERSEER   synthetic fixtures
    api.py                          OVERSEER   run_scenario() orchestration
    hitran/
      __init__.py                   dev-alpha
      fetch.py                      dev-alpha  hitran.org download
      cache.py                      dev-alpha  LineList + npz persistence
      tips.py                       dev-alpha  partition sums Q(T)
    optics/
      __init__.py                   dev-alpha
      lbl.py                        dev-alpha  Voigt line-by-line engine
      continuum.py                  dev-alpha  H2O self+foreign continuum
      aerosol.py                    dev-beta   rural/urban/navy/desert
      rayleigh.py                   dev-beta
      transmittance.py              OVERSEER   assembles LayerOptics (wave 2)
    atmosphere/
      __init__.py                   dev-beta
      profiles.py                   dev-beta   6 standard model atmospheres
      humidity.py                   dev-beta   ground T/RH override
    geometry/
      __init__.py                   dev-beta
      path.py                       dev-beta   spherical slant path
    radiometry/                     wave 2
    ephemeris/                      wave 2
    cli.py                          wave 2
    webgui/                         wave 2
    turbulence/                     wave 3 (stretch)
  tests/
    test_hitran.py, test_lbl.py, test_continuum.py        dev-alpha
    test_profiles.py, test_geometry.py, test_aerosol.py   dev-beta
```

Each worker owns its subtree exclusively. **Never edit a file owned by the
other worker**, including its `__init__.py`.

---

## Working without the other worker

The two jobs are decoupled by `mwir_atmos/testing.py`:

- **dev-alpha** consumes `PathLayer` objects. Real ones come from dev-beta's
  geometry, which does not exist yet — so test against
  `testing.make_synthetic_path()`, which produces contract-correct `PathLayer`s
  with crude physics. Your code must work on *any* valid `PathLayer`.
- **dev-beta** produces `PathLayer` objects and never needs the line data.
  Validate column densities and geometry on their own terms (hydrostatic
  consistency, known airmass limits, published profile values).

---

## Interfaces each worker must expose

Signatures are binding. Keyword-only args after `*` may be extended with
additional optional keywords, but the positional contract is fixed.

### dev-alpha

```python
# mwir_atmos/hitran/cache.py
@dataclass
class LineList:
    species: Species
    molecule_id: int                 # HITRAN molecule number
    nu_cm1: np.ndarray               # line centre, cm^-1 (vacuum)
    s296: np.ndarray                 # intensity @296 K, cm^-1/(molecule cm^-2)
    gamma_air: np.ndarray            # air-broadened HWHM @296 K, 1 atm, cm^-1/atm
    gamma_self: np.ndarray           # self-broadened HWHM, cm^-1/atm
    e_lower: np.ndarray              # lower-state energy, cm^-1
    n_air: np.ndarray                # T exponent of gamma_air
    delta_air: np.ndarray            # air pressure shift, cm^-1/atm
    local_iso_id: np.ndarray         # HITRAN local isotopologue number
    iso_abundance: np.ndarray        # terrestrial abundance of each line's isotopologue

def save_linelist(linelist: LineList, path) -> None
def load_linelist(path) -> LineList

# mwir_atmos/hitran/fetch.py
def fetch_linelist(species: Species, numin: float, numax: float, *,
                   isotopologues: Sequence[int] | None = None) -> LineList
def ensure_linelists(*, cache_dir=None, force: bool = False,
                     numin: float = 1775.0, numax: float = 4325.0
                     ) -> dict[Species, LineList]
    """Load from data/hitran/ if present, else fetch and cache. The ONLY
    function permitted to touch the network."""

# mwir_atmos/hitran/tips.py
def partition_function(molecule_id: int, local_iso_id: int,
                       temperature_k: float | np.ndarray) -> float | np.ndarray
    """Total internal partition sum Q(T), 70-3000 K."""

# mwir_atmos/optics/lbl.py
def absorption_cross_section(linelist: LineList, grid: SpectralGrid,
                             pressure_mbar: float, temperature_k: float,
                             partial_pressure_mbar: float, *,
                             wing_cutoff_cm1: float = 25.0,
                             min_strength_fraction: float = 0.0,
                             subtract_pedestal: bool = False,
                             method: str = "direct",          # | "multigrid" | "auto"
                             strength_reference: float | None = None,
                             column_cm2: float | None = None,
                             min_peak_optical_depth: float = 0.0) -> np.ndarray
    """Voigt cross-section, cm^2/molecule, shape (grid.n,)."""

def molecular_optical_depth(layers: Sequence[PathLayer], grid: SpectralGrid, *,
                            species: Sequence[Species] | None = None,
                            linelists: dict[Species, LineList] | None = None,
                            include_continuum: bool = True,
                            wing_cutoff_cm1: float = 25.0,
                            min_strength_fraction: float = 0.0,
                            method: str = "auto",
                            n_threads: int | None = None,
                            strength_references: Mapping | None = None,
                            min_peak_optical_depth: float = 3e-6,
                            backend: str | None = None,       # "cpu" | "cuda"
                            ) -> tuple[np.ndarray, dict[Species, np.ndarray]]
    """THE headline function. Returns (tau_total, tau_by_species), each
    shape (len(layers), grid.n). Results never depend on n_threads."""

# mwir_atmos/optics/continuum.py
def h2o_continuum_optical_depth(layers, grid) -> np.ndarray   # (n_layers, grid.n)
def co2_continuum_optical_depth(layers, grid) -> np.ndarray   # may return zeros
```

Physics requirements for dev-alpha:
- Voigt profile (Humlicek or `scipy.special.wofz`) — **not** Lorentz-only; at
  MWIR altitudes above ~20 km Doppler width dominates and a Lorentz-only model
  is visibly wrong.
- Line intensity temperature scaling with the full HITRAN relation, including
  the `Q(296)/Q(T)`, Boltzmann lower-state, and stimulated-emission factors.
- Pressure shift `delta_air`, and `gamma = (gamma_air*(p_air) + gamma_self*p_self)*(296/T)^n_air`
  with pressures in atm.
- Weight each line by its isotopologue's terrestrial abundance (HITRAN `sw`
  is already abundance-scaled — **verify this and document which convention
  you used**, it is the single most common factor-of-N bug here).
- H₂O continuum matters enormously in the 4–5 µm window; a missing continuum
  makes long horizontal paths far too transparent. MT_CKD-style self + foreign
  parameterisation is acceptable; document the source of your coefficients.
- Performance: a 50-layer, 2500-point grid must complete in a few seconds.
  Cache cross-sections; vectorise over lines, not Python loops per line.

### dev-beta

```python
# mwir_atmos/atmosphere/profiles.py
def get_profile(model: ModelAtmosphere, *, n_levels: int | None = None) -> AtmosphereProfile
def available_models() -> list[ModelAtmosphere]

# mwir_atmos/atmosphere/humidity.py
def saturation_vapor_pressure_mbar(temperature_k) -> np.ndarray
def relative_humidity_pct(temperature_k, pressure_mbar, h2o_vmr_ppmv) -> np.ndarray
def apply_surface_conditions(profile: AtmosphereProfile, *,
                             ground_temp_k: float | None = None,
                             ground_rh_pct: float | None = None,
                             co2_ppmv: float | None = None,
                             surface_altitude_km: float = 0.0) -> AtmosphereProfile
    """Rescale the boundary layer to match user-specified surface T and RH,
    relaxing back to the model atmosphere aloft (blend over ~2-3 km). Must
    stay hydrostatically consistent and must populate
    profile.relative_humidity_pct."""

# mwir_atmos/geometry/path.py
def build_path(profile: AtmosphereProfile, *, sensor_alt_km: float,
               zenith_deg: float, azimuth_deg: float = 0.0,
               source_alt_km: float | None = None,
               source_range_km: float | None = None,
               max_layers: int = 60) -> PathGeometry

# mwir_atmos/optics/aerosol.py
def aerosol_optical_depth(layers: Sequence[PathLayer], grid: SpectralGrid,
                          model: AerosolModel, *, visibility_km: float = 23.0,
                          wind_speed_m_s: float = 5.0,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    """Returns (tau_absorption, tau_scattering, asymmetry_g), each
    shape (len(layers), grid.n)."""

# mwir_atmos/optics/rayleigh.py
def rayleigh_optical_depth(layers, grid) -> np.ndarray
```

Physics requirements for dev-beta:
- The six standard LOWTRAN/MODTRAN profiles, tabulated at the canonical
  levels (0–25 km at 1 km, then 30/35/40/45/50/70/100 km). Include P, T, and
  H₂O/O₃ VMR per model; CO₂/N₂O/CO/CH₄ from `BACKGROUND_VMR_PPMV`. Cite your
  source in a module docstring.
- `build_path` must handle: upward-looking (zenith < 90°), downward-looking
  (> 90°) including ground intersection, horizontal (≈90°), limb/tangent paths
  that turn over without hitting ground, sensor above source and below, and a
  finite `source_range_km` that terminates mid-atmosphere. Spherical earth,
  `EARTH_RADIUS_KM`. Refraction may be omitted — say so in a docstring.
- Curtis–Godson: populate `p_eff_mbar` / `t_eff_k` per species as
  column-weighted means. dev-alpha's band physics depends on these.
- Aerosols: rural / urban / navy-maritime / desert with LOWTRAN-style
  wavelength-dependent extinction, RH-driven hygroscopic growth (use
  `PathLayer.rh_pct`), and a visibility-driven surface extinction via the
  Koschmieder relation. Navy model uses `wind_speed_m_s`. In the MWIR,
  scattering is weak and absorption dominates — but return both, correctly
  separated, because the radiance layer needs them apart.
- Cite the source of every coefficient table in a docstring.

---

### September 2026 additions (backward compatible)

- `SpectralGrid` gained `lattice_origin_cm1: float | None = None` and
  `lattice_index0: int = 0`: a grid may declare its position on a larger
  uniform lattice so spectral chunks of one calculation evaluate lines
  identically. Grids from `make_grid` are their own lattice.
- `Scenario` gained `spectral_sampling` (`SpectralSampling.POINT` |
  `RESOLVED`, the default), `fine_step_cm1`, `instrument_function`
  (`InstrumentFunction`), `instrument_fwhm_cm1` and
  `line_optical_depth_cutoff`. `line_strength_cutoff` now defaults to 0.0.
- `ScenarioResult.fine_step_cm1` is set for resolved sampling (else None).
- `run_scenario(..., n_threads=None, backend=None)` (`backend`: `"cpu"` or
  `"cuda"`, default `MWIR_BACKEND` or `"cpu"`); `mwir_atmos.resolved` implements the
  instrument-function sampling. Radiance solvers receive whatever grid they
  are given; with resolved sampling that may be a fine chunk grid or, for
  DISORT/Monte Carlo, a pseudo grid whose wavenumbers repeat (one entry per
  correlated-k interval). Solvers must not assume unique wavenumbers.

---

## Wave 2 interfaces (radiometry, ephemeris, interfaces)

`api.py` already calls these exact signatures. They are binding — `run_scenario`
imports them lazily and silently degrades to a transmittance-only result if they
are missing, so a signature mismatch will fail *silently*. Match them exactly.

```python
# mwir_atmos/ephemeris/sun.py
def solar_position(scenario: Scenario) -> SkyPosition

# mwir_atmos/ephemeris/moon.py
def lunar_position(scenario: Scenario, sun: SkyPosition) -> SkyPosition

# mwir_atmos/radiometry/radiance.py
def compute_radiance(*, scenario: Scenario, grid: SpectralGrid,
                     profile: AtmosphereProfile, path: PathGeometry,
                     optics: LayerOptics, transmittance: TransmittanceResult,
                     sun: SkyPosition, moon: SkyPosition) -> RadianceResult
```

`RadianceResult` fields are already fixed in `types.py`; every component must be
zero-filled rather than `None` when it does not apply, and `total` must equal the
sum of the applicable components.

Wave 2 ownership:
```
    mwir_atmos/radiometry/**          dev-alpha
    mwir_atmos/ephemeris/**           dev-alpha
    tests/test_radiometry.py          dev-alpha
    tests/test_ephemeris.py           dev-alpha
    mwir_atmos/cli.py                 dev-beta
    mwir_atmos/webgui/**              dev-beta
    scenarios/*.yaml                  dev-beta
    tests/test_cli.py                 dev-beta
    README.md                         dev-beta
```

Known performance constraint: a full 3-5 um scenario at 1 cm^-1 over ~60 layers
currently takes 7-10 s, dominated by the line-by-line engine. The GUI must not
recompute on every widget interaction — cache aggressively and let the user pick
a coarser resolution for interactive work.

## Definition of done (both workers)

- Code runs on Python 3.11, numpy 2.3, scipy 1.16 (already installed).
- `pytest` tests for your subtree pass, including at least one physically
  meaningful check (e.g. sea-level 1 km horizontal path in the 4.2–4.4 µm CO₂
  band must be essentially opaque; a 3.7 µm window path must not be).
- Docstrings cite data sources. No silent magic numbers.
- Report back in the required STATUS / CHANGES / VERIFICATION / NOTES format.
