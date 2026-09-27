# Line selection, spectral sampling and the fast line-by-line engine

Status: **implemented, validation in progress** (September 2026). The
measurements quoted here were made during development; the items under
[Pending validation](#pending-validation) must be completed before the new
defaults are treated as settled. See also [PHYSICS_VALIDATION.md](../PHYSICS_VALIDATION.md).

## Two accuracy problems

### 1. The old line cutoff removed the window water lines

The former default, `line_strength_cutoff = 1e-4`, skipped every line weaker
than 1e-4 of its species' strongest line. Measured by *summed* line strength
this looks harmless (every species keeps >99% of it), but window absorption
comes from the weak lines. The strongest H2O line in the cached list (the
2.7 um band, 2.48e-19 cm/molecule at 296 K) put the threshold at 2.5e-23.
Every H2O line between 2440 and 2860 cm^-1 is weaker than that -- main
isotopologue at most 6.9e-24, HDO at most 9.7e-24 -- so the rule dropped all
6,162 of them, leaving only the MT_CKD continuum.

Transmittance error of the old default against keeping every line, 1 cm^-1
point sampling, band means in percentage points:

| Path | 4.5-5.0 um | 3.5-4.1 um | 2.9-3.5 um | Largest single point |
|---|---:|---:|---:|---:|
| Tropical, 10 km horizontal at 10 m | +5.00 | **+23.65** | +3.47 | 74.1 pp at 2624 cm^-1 |
| Mid-latitude summer, ground to space | +9.62 | +9.79 | +5.04 | 91.7 pp at 2780 cm^-1 |
| 20 km nadir to ground | +9.87 | +9.81 | +5.04 | 92.1 pp at 2780 cm^-1 |

On the tropical path the error is almost entirely water: in the 3.5-4.1 um
band the H2O partial transmittance is 0.578 with all lines and 0.869 with the
old rule, while every other species differs by at most 0.22 pp.

**Now:** a line is evaluated in a layer when its *peak optical depth there*,
`S(T) * N_layer * V(0)`, reaches `line_optical_depth_cutoff` (default 1e-5).
`V(0)` is the exact Voigt peak, `sqrt(ln2/pi)/alpha_D * erfcx(y)`. The rule
follows LBLRTM's absolute line-rejection threshold. It depends on the actual
amount of gas: a thick humid layer keeps the weak water lines, a thin
stratospheric layer drops O3 lines that cannot matter there. The relative
rule remains available and is off by default (`line_strength_cutoff = 0`);
both can be combined.

The criterion is applied per layer, so optical depth is additive over split
layers only to within the peak optical depths of dropped lines. Tests of
exact additivity use a column-independent rule
(`tests/test_physics_regression.py`).

### 2. Point sampling misses line cores

With `spectral_sampling = "point"` (the default until September 2026) the physics runs only at the
output grid points. Pressure-broadened lines are ~0.07 cm^-1 wide at the
surface and a few thousandths of a cm^-1 in the stratosphere, so on a 1 cm^-1
grid each sample falls at a random place relative to the nearby lines. Every
sample is an exact *monochromatic* value, but not what a 1 cm^-1 instrument
measures, and since transmittance is nonlinear in optical depth the errors do
not average out.

On a 2 km horizontal mid-latitude-summer path, 2040-2060 cm^-1, 1 cm^-1
point samples differed from line-resolved 1 cm^-1 bin averages (0.005 cm^-1
reference) by up to **3.76 pp**, 0.67 pp on average, and the band mean was
0.603 pp high (0.4468 vs 0.4408). That comparison used the old line cutoff
for both runs; the sampling effect is separate from it.

## Line-resolved sampling

```yaml
spectral_sampling: resolved      # default (alternative: point)
resolution_cm1: 1.0              # output grid spacing, as before
instrument_function: box         # box | triangle | gaussian
instrument_fwhm_cm1: null        # null = resolution_cm1
fine_step_cm1: null              # null = automatic
```

CLI: `--spectral-sampling resolved` (default) `--instrument-function triangle
--instrument-fwhm-cm1 2 --fine-step-cm1 0.005`. GUI: *Spectral sampling* in
the spectral sidebar section.

Method (`mwir_atmos/resolved.py`):

1. The output grid is exactly the point-sampling grid.
2. A fine lattice with step `h = resolution / r` (integer `r`) contains every
   output point. It extends past both band edges by the instrument
   function's support, because an instrument centred near the edge sees
   radiation beyond it. The step never exceeds a quarter of the FWHM.
3. Consecutive output points are processed in chunks of bounded memory
   (fine points x layers <= 2^21). For each chunk the full physics -- layer
   optics, transmittance, radiance -- runs on the fine grid, and every
   spectral array is reduced with normalised weights:
   `out_j = sum_k w_k f(nu_j + k h)`.
4. Chunks share one lattice (`SpectralGrid.lattice_origin_cm1`), so line
   evaluation does not depend on where the spectrum is split
   (`test_resolved_results_do_not_depend_on_chunking`).

Instrument functions (`resolved.instrument_weights`):

| Function | Shape | Discrete weights |
|---|---|---|
| `box` | uniform over +/-FWHM/2 | overlap of each fine cell with the box: trapezoid rule when FWHM spans an even number of steps, a plain mean when odd |
| `triangle` | half-base = FWHM | sampled at fine points |
| `gaussian` | truncated at +/-2 FWHM (tails hold 3e-6 of the area) | sampled at fine points |

What changes in the output:

- Every transmittance and radiance array is instrument-averaged, including
  `cumulative` and the per-species and per-component breakdowns.
- **Band transmittance does not factorise.** The average of a product is not
  the product of averages, so `total` equals `molecular * aerosol * rayleigh`
  only approximately (aerosol and Rayleigh are smooth, so nearly), and
  per-species transmittances do not multiply to `molecular`. This is physical
  (MODTRAN band transmittances behave the same way), not an error.
  Radiance components still sum exactly to `total`.
- `ScenarioResult.fine_step_cm1` records the fine step used.

### Automatic fine step

`resolved.automatic_fine_step` gives each path layer the Voigt half-width of
a reference line (CO2-like: 0.07 cm^-1/atm air broadening, n = 0.75, mass
44) at the band's lowest wavenumber, where Doppler widths are narrowest.
Layers are ranked from narrowest to broadest lines, and the narrowest are
skipped while their cumulative share of the path's air column stays below 1%
(the mesosphere on a ground-to-space path). The step is the remaining
minimum half-width divided by 2, clamped to 0.0005-0.05 cm^-1. A 2 km
surface path gets 0.034 cm^-1; paths through the stratosphere get roughly
0.001-0.002 cm^-1.

This is a sizing rule, not an error bound. Check convergence by setting
`fine_step_cm1` explicitly (see Pending validation).

### DISORT and spherical Monte Carlo

Both solvers handle one spectral point at a time; running them on the fine
grid would multiply their cost by `r` (tens to thousands). For them,
`resolved.binned_radiance` applies the correlated-k (spectral mapping)
approximation within each instrument function:

1. The fine points under output point `j` are ranked by column extinction
   optical depth.
2. Their cumulative instrument weight `g` is split at
   `0, 0.30, 0.55, 0.75, 0.875, 0.94, 0.975, 0.99, 1` (finer where optical
   depth changes fastest).
3. Within each interval, every layer's optical depths are weight-averaged;
   the aerosol asymmetry parameter is averaged with scattering weights.
4. The solver runs once per non-empty interval (at most 8 per output point),
   with Planck, solar and lunar sources evaluated at the output wavenumber.
5. Output radiance is the interval-weighted sum. Monte Carlo standard errors
   combine in quadrature with the same weights.

Assumptions: the spectral ordering by column optical depth holds in every
layer, and sources and surface properties are constant across one instrument
function. For any quantity linear in the layer optical depths the
combination is exact (`test_binned_radiance_reproduces_linear_functionals_exactly`).
Transmittance is always the exact fine-grid result, never binned.

Not supported with resolved sampling: `sensor_response_file` (rejected by
validation). The experimental Mie aerosol option recomputes Mie coefficients
at every fine point, which is correct but slow.

## Fast line-by-line engine

`mwir_atmos/optics/lbl.py`:

- **Per-list constants** (wavenumber sort, isotopologue masses and partition
  indices, stimulated-emission reference, width statistics) are computed once
  per `LineList` and cached. Reassigning a list's arrays invalidates the cache.
- **Window selection** by binary search: each evaluation touches only lines
  whose wings can reach the grid.
- **Ragged evaluation**: `(line, grid point)` pairs are flattened, so windows
  clipped at the grid edge cost nothing. Batches of 2^14 pairs keep
  temporaries in L2 cache. On Windows, 2^17 was 4x slower single-threaded,
  because of allocator and page-fault cost.
- **Multigrid wings** for fine grids. Each line is sampled on a coarse mesh,
  every `m`-th lattice point with `H = m*h` <= core/12. The summed mesh is
  interpolated with four-point cubic Lagrange weights. Each line is then
  corrected exactly (profile minus its own interpolant) over its core, which
  is at least 12 Voigt half-widths, and over the stretches where the stencil
  straddles its cutoff edges. The remaining approximation is cubic
  interpolation of smooth far wings, measured at <=1.0e-4 relative (points
  above 1e-6 of the maximum). The mesh plan uses list-wide width statistics,
  so it is the same in every chunk. It is chosen automatically only when
  expected to be at least 2x cheaper than direct evaluation; coarse grids use
  direct evaluation.
- **Threads** over `(layer, species)` pairs. NumPy and `wofz` release the
  GIL; each pair writes only its own output row, so results are independent
  of the thread count. Control: `n_threads=` (API), `--threads` (CLI), or the
  `MWIR_THREADS` environment variable; default is every CPU. On the target
  server (2 x 96 cores), set `MWIR_THREADS` explicitly when sharing the
  machine.

Measured during development (a shared 16-thread workstation; timings are
indicative only):

| Check | Result |
|---|---|
| Direct evaluation vs pre-rewrite engine (all species, 3 grids, 3 atmospheric states) | max relative difference 7.9e-15 |
| Full default scenario LBL (76 layers, 2501 points), single thread, former line selection for a like-for-like comparison | 12.7 s -> ~5.4 s |
| Multigrid vs direct (4 windows x 6 species x 4 states, h = 0.001-0.01) | <=1.0e-4 relative; 2-10x faster per call |

## Pending validation

Scripts and a runbook for every item: [validation/README.md](validation/README.md).

1. **Line-selection threshold.** Sweep `line_optical_depth_cutoff` (1e-3 to
   1e-6) against keeping every line, over five geometries (humid horizontal,
   ground-to-space, nadir, limb, dry arctic). Choose the default that keeps
   window errors below ~0.1 pp. The current default of 1e-5 is provisional.
2. **Fine-step convergence.** Compare resolved sampling at automatic and
   fixed steps (0.05-0.002) against a 0.001 cm^-1 reference over four windows
   and five geometries; confirm or adjust the automatic rule.
3. **Correlated-k accuracy.** Compare binned radiance with exact fine-grid
   radiance using the single-scattering solver (which supports both), to
   bound the binning error before relying on it for DISORT and Monte Carlo.
4. **Performance** on an idle machine: thread scaling, full-band resolved
   cost (now the default; a ground-to-space 2000-2200 cm^-1 run took 115 s on
   4 cloud cores at the automatic 0.001 cm^-1 step, so ~25 min for the full
   default band), and recalibration of the CLI/GUI cost estimate
   (`cli._COST_RATE_ELEMENTS_PER_S`).
5. **Re-baselining** of tests whose thresholds were set while the window
   water lines were missing: `test_climatological_soundings_polar_vs_tropical`
   (subarctic 15 km > 0.90, now 0.84),
   `test_lookdown_window_approaches_surface_emission` (window radiance ratio
   > 0.93, now 0.92) and `test_us_standard_10km_horizontal_transmission_benchmarks`
   (window bands). These tests pin `spectral_sampling: point`, like the other
   physics tests written before resolved sampling became the default. Change a threshold only with an independent reference, not
   to match the new output.
