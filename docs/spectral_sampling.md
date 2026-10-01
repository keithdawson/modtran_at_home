# Line selection, spectral sampling and the fast line-by-line engine

Status: **implemented and validated** (September 2026). The defaults below
were chosen from the studies in [Validation results](#validation-results);
their scripts, raw results and runbook are in [validation/](validation/README.md).
See also [PHYSICS_VALIDATION.md](../PHYSICS_VALIDATION.md).

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
`S(T) * N_layer * V(0)`, reaches `line_optical_depth_cutoff` (default 3e-6,
chosen in [Validation results](#1-line-selection-threshold)).
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
   `out_j = sum_k w_k f(nu_j + k h)`. Large runs split into at least three
   chunks per worker, evaluated in parallel worker processes (see
   [Parallelism](#parallelism)).
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
skipped while their cumulative share of the path's air column stays below 10%
(everything above ~16 km on a ground-to-space path). The step is the
remaining minimum half-width divided by 2, clamped to 0.0005-0.05 cm^-1.
Surface horizontal paths get 0.035-0.039 cm^-1; ground-to-space, nadir and
limb paths get 0.005-0.007 cm^-1.

This is a sizing rule, not an error bound, but it was checked against a
0.0005 cm^-1 reference on five paths: every 1 cm^-1 bin is within 0.045 pp
of transmittance and 0.11% of radiance ([results](#2-fine-step)). Check
other cases by setting `fine_step_cm1` explicitly.

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
6. **Single-scatter correction.** The single-scattering solver also runs on
   the same bins, and exactly on the fine grid (it is cheap there). Every
   radiance array becomes `binned(solver) + exact(single) - binned(single)`,
   so only the difference between the solvers -- multiple scattering -- is
   binned; thermal emission and direct transmission are exact.

Assumptions: the spectral ordering by column optical depth holds in every
layer, and sources and surface properties are constant across one instrument
function. The first fails on non-uniform paths: a nadir or limb view sees
stratospheric line cores and tropospheric gaps at different fine points.
Without step 6 the binning erred by up to 47% in a 1 cm^-1 bin and 4.5% over a
20 cm^-1 window there ([results](#3-correlated-k-binning)). For any quantity
linear in the layer optical depths the combination is exact
(`test_binned_radiance_reproduces_linear_functionals_exactly`), and the
correction is exact for the reference solver itself
(`test_reference_correction_replaces_the_binning_error_of_the_reference_solver`).
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
- **Peak pre-filter** (`_peak_candidates`). Before the exact
  peak-optical-depth test, every line in the window is compared with an
  upper bound on its peak optical depth built from precomputed per-line
  numbers: the Voigt peak is at most the smaller of the Doppler and Lorentz
  peaks, S(T) is bounded by its largest partition, Boltzmann and
  stimulated-emission factors, and gamma_L by its air-broadened part. Only
  the lines that pass get S(T), their widths and `erfcx`. On the full default
  band (76 layers x 6 species, 65.8 M (line, layer) pairs scanned) 16% pass
  and 10.3% are kept; single-threaded selection went from 11.1 s to 2.0 s,
  with bit-identical selections
  (`test_peak_prefilter_does_not_change_line_selection`, including a layer
  above 296 K).
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
- **Threads** over `(layer, species)` pairs for point sampling. Each pair
  writes only its own output row, so results are independent of the thread
  count. Threads scale poorly (Python's global interpreter lock): see
  [Parallelism](#parallelism) for what resolved runs do instead.

### GPU backend (experimental)

`backend="cuda"` (`--backend cuda`, `MWIR_BACKEND=cuda`) evaluates the line
sums on an NVIDIA GPU through CuPy (`pip install .[cuda]`). Line-by-line work
is ~98% of a resolved run (a 100 cm^-1 ground-to-space run spent 77 of 78 s
in it), and every (line, grid point) pair is independent. Consumer and
workstation GPUs run float64 at 1/64 speed, so the far wings (~99.7% of
pairs) use a float32 continued fraction and the line cores a float64
Weideman expansion (`optics/voigt_gpu.py`); offsets and sums are float64.
The profile is within 4.7e-7 of `scipy.special.wofz`, and a NumPy twin of
the kernel matches the CPU engine's exact sums to 2e-6 on
surface-to-stratosphere layers (`tests/test_lbl_cuda.py`).

How a resolved run uses the GPU (`optics/lbl_cuda.py`, `api._resolved_chunk`):

1. The band is split into chunks of 2^23 fine point x layer elements (five
   for the default band; ~1.0 GiB of GPU memory at the peak, released after
   the run, since the GPU may be shared).
2. The CPU selects lines (with the peak pre-filter above) and the GPU sums
   them by direct evaluation. A chunk's (layer, species) tasks are dealt into
   eight groups; the CPU selects group k+1 while the GPU evaluates group k,
   and each task writes its own row of one (species x layer, point) array.
   Selection then hides behind the kernels even on one thread, so the CUDA
   backend uses 4 selection threads unless `n_threads`/`MWIR_THREADS` says
   otherwise (point sampling: 0.34 s on 4 threads, 0.32 s on 16).
3. For the single-scatter solver the chunk stays on the GPU: continuum,
   aerosol, Rayleigh, path transmittance, radiance and the instrument
   reduction run on CuPy arrays through the same code as on the CPU (each
   function takes its array module from the grid, `mwir_atmos/_array.py`),
   and only the output-grid arrays are copied back. DISORT, Monte Carlo and
   Mie aerosol optics copy the optical depths back once and continue on the
   CPU (the "host path").

On an RTX 4090 (`validation/cuda_check_DESKTOP-RLDQTSU.json`) the on-device
tests pass, and against all 16 CPU threads of the same desktop:

| Run | CPU | GPU | Speedup | Max difference |
|---|---:|---:|---:|---|
| 2000-2200 cm^-1, ground-to-space | 9.4 s | 0.18 s | 51.7x | 3.4e-6 T, 7.6e-6 L |
| Full default band 1800-4300 cm^-1 | 35.7 s | 0.93 s | 38.2x | 4.5e-6 T, 1.0e-5 L |

(With the float32 kernel described below, `e8b70e4`. Before it, at
`6432598`, the same check measured 8.3 s / 0.20 s and 33.4 s / 0.95 s; the
CPU code is the same, so the CPU times differ by run-to-run variation only.)

Before the changes above the GPU took 0.62 s and 7.6 s (14.3x and 4.8x); on
the day of the new measurement the old code took 0.56 s and 6.58 s. Its
full-band time, profiled (`run_scenario(Scenario(), backend="cuda")`, three
chunks of 2^24, stage timers synchronised with the GPU):

| Stage | Old | New |
|---|---:|---:|
| Kernels (CUDA events) | 0.75 s | 0.77 s |
| Line selection (16 threads) + packing | ~1.1 s | hidden behind the kernels |
| Copying (layer, species) rows back and into per-species arrays | 0.64 s | 0 (stays on the GPU) |
| MT_CKD continuum (recomputed its stencil per layer, two interpolations) | 2.01 s | 0.035 s |
| Per-species copies, continuum additions, species sum (`np.stack`) | ~0.6 s | on the GPU, in place |
| Aerosol + Rayleigh | 0.32 s | 0.010 s |
| Path transmittance | 0.61 s | 0.021 s |
| Radiance | 0.61 s | 0.010 s |
| Instrument reduction (+ copy to host) | 0.013 s | 0.014 s |
| **Wall time, uninstrumented** | **6.58 s** | **0.95 s** |

The new GPU results equal the old ones to 3e-15 relative (transmittance,
cumulative, per-species) and 4e-16 (radiance); the device-resident chunks
equal the host path to 3e-16 T and 1.4e-15 L. The host path takes 2.8 s for
the same band. The CPU backend is essentially unchanged (33.3 s vs 33.5-33.9 s
for the old code, same session): its time is Voigt evaluation.

Since then the kernel does no float64 work per (line, point) pair (offsets
as float32 hi/lo pairs from a per-block reference, per-line cutoff ranges
found once per tile with the float64 test, compensated float32 tile sums):
kernels 0.85 s -> 0.33 s for the full band, accuracy against the CPU float64
sums unchanged (worst 1.3e-6 relative in upper layers, 1.2e-6 before). The
full band gains much less: CPU line selection (GIL-bound, ~0.8 s) is now the
limit, and it slows after the first two full-band runs in a process because
Windows lets the selection threads share physical cores. Ten uninstrumented
runs per process, idle desktop (Ryzen 7 9800X3D, High performance plan):

| Selection threads | Runs 1-2 | Runs 3-10 (median) |
|---|---:|---:|
| 4 (default) | 0.79-0.81 s | 0.95-0.96 s |
| 4, process on one logical CPU per core (mask 0x5555) | 0.76-0.78 s | 0.81 s |
| 4, selection threads alone on one logical CPU per core | 0.79-0.80 s | 0.84-0.85 s |
| 4, ideal-processor hints 0, 2, 4, 6 | 0.79-0.80 s | 0.95 s |
| 4, one thread pool reused across chunks and runs | 0.81-0.83 s | 0.97 s |
| 2 | 0.77-0.78 s | 0.98 s |
| 1 | 0.85-0.89 s | 0.89 s |

Instrumented probes (fresh process each; thread cycle counters around
`_select_lines`): the same 2280 selections cost 1.04 thread-CPU-s in runs 1-2
and 1.33 after on 4 threads, 0.62 -> 0.67 on one thread, and 0.61-0.63 s per
pass when replayed single-threaded without the GPU. The old kernel and runs
with the kernel launches skipped drift the same way; private memory (2.10
GiB), page faults (~80k per run) and GC (0.2 ms) stay flat; not freeing the
CuPy pools changes nothing. With 30 s pauses between runs every run after the
first is slow (1.18-1.23 s: clocks also drop while idle). The library keeps
4 threads and no affinity (Windows-only, and a hard limit on a shared
machine); moving selection to the GPU removes this CPU work. See
[DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md).

#### GPU line selection (opt-in, not yet measured on a GPU)

`MWIR_CUDA_SELECTION=gpu` selects lines on the GPU
(`lbl._gpu_selected_groups` / `lbl._select_packed`). Per line list, every
(task, window line) pair of up to 2^23 pairs is one element of a flat array
(the default full band: 76 layers x 6 line lists, ~66 M pairs, ~10 passes);
the per-element operations are `_select_lines`' and `pack_task`'s, in the
same order, written once for NumPy and CuPy. Lines are kept, sorted by
(task, shifted centre) with two stable argsorts, and the packed arrays go to
the kernel without leaving the GPU. The CPU only finds each task's window
(`searchsorted`) and partition-function ratios. The peak pre-filter is
skipped: it removes only lines the exact test drops.

Run in NumPy, the code gives bit-identical packed arrays to the CPU path on
every task of the default scenario: a 2140-2145 cm^-1 slice (456 tasks, 241k
lines) and the full 1800-4300 cm^-1 band at 0.5 cm^-1 (6.79 M lines), and
also with `min_strength_fraction` and a list split over several passes
(`test_selection_in_numpy_equals_the_cpu_selection`). On the GPU every basic
operation is one correctly rounded CuPy kernel, so only `exp`, `expm1`, `pow`
and `erfcx` may differ by an ulp; a line within that of the 3e-6 threshold
can flip, changing tau by < 3e-6. The `gpu` tests check that at most 1e-5 of
the kept lines flip, and `cuda_check.py` times five runs per case with
both selections. To do: those numbers.

### Parallelism

Resolved runs of at least 2^21 fine-point x layer elements are split into at
least three chunks per worker, and whole chunks -- line-by-line, continuum,
transmittance, radiance, instrument reduction -- run in worker processes,
each single-threaded. Linux forks when no other threads are alive (workers
inherit the loaded line lists); otherwise workers are spawned, with BLAS and
OpenMP limited to one thread each, and scripts calling `run_scenario` need an
`if __name__ == "__main__":` guard. Results match a serial run to rounding.

Control: `n_threads=` (API), `--threads` (CLI) or `MWIR_THREADS` sets the
worker count (default: every logical CPU); `MWIR_PARALLEL=threads` keeps
everything in one process. On the work server (2 x 96 cores), set
`MWIR_THREADS` when sharing the machine.

Measured during development (a shared 16-thread workstation; timings are
indicative only):

| Check | Result |
|---|---|
| Direct evaluation vs pre-rewrite engine (all species, 3 grids, 3 atmospheric states) | max relative difference 7.9e-15 |
| Full default scenario LBL (76 layers, 2501 points), single thread, former line selection for a like-for-like comparison | 12.7 s -> ~5.4 s |
| Multigrid vs direct (4 windows x 6 species x 4 states, h = 0.001-0.01) | <=1.0e-4 relative; 2-10x faster per call |

## Validation results

Raw results: `docs/validation/*.json`, logs in `docs/validation/logs/`; scripts
and runbook in [validation/README.md](validation/README.md). Studies ran on an
8-core desktop (September 2026) unless stated.

### 1. Line-selection threshold

Resolved window transmittance (1 cm^-1 box, 0.01 cm^-1 fine step) at each
`line_optical_depth_cutoff`, against 1e-8 (which agrees with keeping every
line to 0.0005 pp). Five paths: tropical and subarctic-winter 10 km
horizontal, mid-latitude summer ground-to-space, 20 km nadir, US-standard
15 km limb; three windows (4.5-5.0, 3.5-4.1, 2.9-3.5 um). Worst case over all
paths and windows:

| Cutoff | Band-mean error | Worst 1 cm^-1 bin | Line cost vs 1e-8 |
|---:|---:|---:|---:|
| 1e-4 | 0.24 pp | 1.10 pp | 0.09x |
| 3e-5 | 0.11 pp | 0.42 pp | 0.14x |
| 1e-5 | 0.048 pp | 0.19 pp | 0.20x |
| **3e-6** | **0.021 pp** | **0.078 pp** | **0.30x** |
| 1e-6 | 0.009 pp | 0.030 pp | 0.40x |
| 1e-7 | 0.0015 pp | 0.0045 pp | 0.67x |

Dropped lines always make the window too transparent. **Default: 3e-6**, the
largest cutoff keeping every bin under 0.1 pp (was 1e-5, provisional).

### 2. Fine step

Resolved transmittance and thermal radiance on four 20 cm^-1 windows (2050,
2500, 2700, 3000 cm^-1) against a 0.0005 cm^-1 reference. Worst 1 cm^-1 bin
per path:

| Path | Old rule (1%): step, T, L | New rule (10%): step, T, L |
|---|---|---|
| Tropical 10 km horizontal | 0.035; 0.026 pp; 0.11% | unchanged |
| Subarctic winter 10 km horizontal | 0.039; 0.045 pp; 0.11% | unchanged |
| Ground-to-space | 0.0011-0.0015; 0.0002 pp; <0.001% | 0.0047; 0.022 pp; 0.011% |
| 20 km nadir | 0.0031-0.0034; 0.0002 pp; 0.002% | 0.0071; 0.0018 pp; 0.005% |
| 15 km limb | 0.0016-0.0020; 0.0003 pp; 0.002% | 0.0070; 0.012 pp; 0.093% |

The 1% rule resolved the stratosphere about 100x more accurately than the
horizontal paths, at 4-6x their cost. The 10% rule (checked in the cloud
session with the new 3e-6 cutoff) gives every path the same worst-bin
accuracy. Fixed steps for comparison, worst over all paths: 0.002 -> 0.0009
pp; 0.005 -> 0.027 pp; 0.01 -> 0.12 pp; 0.02 -> 0.64 pp; 0.05 -> 0.88 pp.

### 3. Correlated-k binning

DISORT radiance from the binned path against DISORT run at every fine point
(0.005 cm^-1) and instrument-averaged, same windows and paths, with and
without the sun. Binning alone:

| Paths | Worst 1 cm^-1 bin | Worst 20 cm^-1 window | Speedup |
|---|---:|---:|---:|
| Horizontal, ground-to-space | 1.3-2.9% | 0.43-0.58% | 16-24x |
| 20 km nadir, 15 km limb | 40-47% | 2.1-4.5% | 16-24x |

The worst bins sit in the opaque CH4 Q-branch near 3015 cm^-1 on the nadir
path, where emission comes from different layers at different fine points.
The binning error of the single-scattering solver on the same path was the
same (47.4% vs 47.1%), which motivated the single-scatter correction (step 6
above). Round 2 (`validation/correlated_k_accuracy_round2.json`, all paths,
windows, day and night) with the correction:

| Path | Binned: worst bin / window | Corrected: worst bin / window |
|---|---:|---:|
| Tropical 10 km horizontal | 2.84% / 0.43% | 0.091% / 0.014% |
| Subarctic winter 10 km horizontal | 1.30% / 0.58% | 0.49% / 0.092% |
| Ground-to-space | 1.82% / 0.49% | 0.012% / 0.002% |
| 20 km nadir | 47.1% / 2.06% | 0.082% / 0.006% |
| 15 km limb | 40.4% / 4.49% | 0.23% / 0.038% |

Both run 21-27x faster than DISORT at every fine point; the correction costs
nothing measurable.

### 4. Performance

Default ground-to-space scenario, 2000-2200 cm^-1, resolved (old 1% step
rule), 8-core desktop:

| Workers | Processes | Threads |
|---:|---:|---:|
| 1 | 64.3 s | |
| 2 | 34.4 s | |
| 4 | 18.2 s | 41.6 s |
| 8 | 11.9 s | 45.7 s |
| 16 (SMT) | 9.6 s | 49.0 s |

Processes scale 5.4x on 8 cores; threads get slower as they are added. At
the new default step (round 2, `validation/scaling_round2.json`) the same
slice takes 42.0 / 16.0 / 9.1 / 8.8 s on 1 / 4 / 8 / 16 workers, and the full
default band 36.3 s on 16.
With the new defaults (10% step rule: 0.0046 cm^-1, ~0.53M fine points x 76
layers), the full default 1800-4300 cm^-1 run took 247 s on 4 cloud cores,
against ~25 min at the old 0.001 cm^-1 step.
The CLI/GUI runtime estimate for resolved runs now uses 0.15 M fine-point x
layer elements per second per worker (0.23 M measured, rounded down for slower
cores) and a speedup of `workers ** 0.7`.

### 5. Window-band tests against an independent reference

HITRAN's HAPI (Kochanov et al. 2016) evaluated every line of the same line
list with its own Voigt profile and partition sums on 10 cm^-1 windows of the
paths behind the window-band tests
(`scripts/physics_reference/generate_hapi_reference.py`). The full model
agrees in every window to 0.0010 in transmittance and 0.38% in look-down
radiance (`tests/test_hapi_reference.py`). The reference also sets the
re-baselined thresholds: subarctic-winter 15 km window transmittance > 0.80
(HAPI 0.84-0.92, was > 0.90) and look-down window radiance > 0.88 of the
surface's (HAPI 0.88-0.99, was > 0.93). Both earlier bounds dated from when
the window water lines were being dropped.
