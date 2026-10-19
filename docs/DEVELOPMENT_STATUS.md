# Development status and handoff

Written at the end of a cloud session (29 September 2026) so a local session
can pick up the work, and kept up to date since. Physics and results live in
[spectral_sampling.md](spectral_sampling.md); the validation runbook in
[validation/README.md](validation/README.md). This file records *state*,
*why*, *what's next*, and the traps.

## Where things stand

Commits of this line of work, oldest first:

| Commit | What |
|---|---|
| `f807792` | Line-resolved sampling, peak-optical-depth line selection, fast LBL engine (multigrid wings, threads), refraction/polarization helpers |
| `292e89e` | CLI flags for every new Scenario field |
| `ca60613` | Line-resolved sampling made the default; physics tests pinned to point sampling |
| `9c85c71` | Resolved chunks evaluated in worker processes; validation scripts |
| `64be940` | Runbook, `run_all.py`, scaling benchmark, HAPI test (skipped until data exists) |
| `25e4c17` | Round-1 validation results from the 8-core desktop (the user's commit) |
| `307ac81` | Defaults from round 1: cutoff 3e-6, 10% fine-step rule, single-scatter correction of binned radiance, recalibrated cost estimate, HAPI-based test thresholds |
| `2262168` | Experimental CUDA backend (`backend="cuda"`) |
| `026f4ad` | Round-2 validation and RTX 4090 CUDA check (the user's commit) |
| `510c051`, `15764f4` | Handoff docs (this file, `CLAUDE.md`); round-2 results in the docs |
| `6432598` | GPU path's CPU side: peak pre-filter for line selection, vectorised MT_CKD continuum, selection overlapped with grouped kernels, device-resident resolved chunks (full band 6.6 s -> 0.95 s on the RTX 4090) |
| `e8b70e4` | CUDA kernel without float64 work per (line, point) pair (kernels 0.85 s -> 0.33 s) |

Test suite: 619 passed, 0 skipped on the RTX 4090 desktop (`pytest -n auto`,
67-78 s; last run on `e8b70e4`); without a GPU the 5 GPU-only tests skip.

## Decisions and their evidence

| Decision | Evidence | Where |
|---|---|---|
| `line_optical_depth_cutoff` 3e-6 | every 1 cm^-1 window bin within 0.078 pp of all lines (1e-5: 0.19 pp) on 5 paths x 3 windows | `validation/line_cutoff_sweep.json` |
| 1e-8 is a valid "all lines" reference | within 0.0005 pp of cutoff 0, half the cost | cloud check, same sweep script |
| Auto fine step sized at 10% of the air column | all paths <= 0.045 pp / 0.11% worst bin; vertical paths 4-6x coarser than the old 1% rule, full default run ~25 min -> ~4 min on 4 cores | `validation/fine_step_convergence.json`, cloud re-check |
| Binned DISORT/MC radiance + exact single-scatter correction | uncorrected binning: 40-47% worst bin, 2-4.5% window on nadir/limb; corrected worst nadir case 0.055% / -0.012% | `validation/correlated_k_accuracy.json`; full re-check is round 2 |
| Processes, not threads, for resolved runs | 5.4x at 8 workers vs threads getting *slower* (GIL) | `validation/scaling_benchmark.json` |
| Window-test thresholds (subarctic > 0.80, look-down > 0.88) | independent HAPI line-by-line reference; model agrees to 0.001 T, 0.38% L | `tests/reference_data/hapi_window_reference.json` |
| Cost estimate: 0.15 M elements/s/worker, speedup workers**0.7 | desktop measured 0.23 M at the *old* step | `mwir_atmos/run_cost.py` `_COST_*` |
| GPU: float32 wings, float64 cores/offsets/sums | GPUs run fp64 at 1/64 rate; profile within 4.7e-7 of wofz | `optics/voigt_gpu.py` |
| Peak pre-filter before exact line selection | 16% of (line, layer) pairs pass the bound, 10.3% are kept; selection 11.1 s -> 2.0 s single-threaded, bit-identical | `lbl._peak_candidates`, `test_peak_prefilter_does_not_change_line_selection` |
| Resolved CUDA chunks stay on the GPU (single scatter) | full band 2.8 s on the host path vs 0.95 s device-resident, equal to 1.4e-15 | `api._device_resident`, [spectral_sampling.md](spectral_sampling.md#gpu-backend-experimental) |
| CUDA chunks of 2^23 elements | 0.95 s / 1.0 GiB, vs 0.92 s / 2.0 GiB at 2^24 and 1.30 s at 2^22 | `api._CUDA_CHUNK_ELEMENTS` |
| 4 selection threads on the CUDA backend | resolved full band 0.96 s on 1 thread, 0.97 s on 16; point sampling 0.34 s on 4, 0.32 s on 16 | `lbl_cuda_dispatch._CUDA_SELECTION_THREADS` |

## Open work, in order

Round 2 and the RTX 4090 check are done (commit `026f4ad`; results in
[spectral_sampling.md](spectral_sampling.md#validation-results)): corrected
binning <= 0.49% worst bin on every path, fine step confirmed.

The GPU path's CPU side is done, and lines are selected on the GPU: the full
default band takes 0.55 s on the RTX 4090 (was 6.6-7.6 s originally, 0.89-0.95
s with CPU selection), ~70x the 16-thread CPU backend (38.3 s); a 200 cm^-1 slice
0.18 s. Profiles, before and after, in
[spectral_sampling.md](spectral_sampling.md#gpu-backend-experimental) and
[GPU line selection](spectral_sampling.md#gpu-line-selection-default).
CPU selection (`MWIR_CUDA_SELECTION=cpu`) was GIL-bound (~0.8 s) and slowed
after two runs on Windows (selection threads sharing physical cores); GPU
selection has no drift.

1. **Faster GPU selection and the rest of a CUDA run** (full band, GPU
   synchronised per stage: kernels 0.30 s, selection 0.15 s, task setup
   0.01 s, the rest ~0.13 s).
   - Selection: ~40 CuPy kernels per pass, float64 temporaries, ~2 ns per
     (task, line) pair. One `ElementwiseKernel` for the keep test and one
     for the packed values would cut the memory traffic several times
     (expected ~0.03 s). Compile with `--fmad=false` to keep the CPU
     rounding, and check against `_select_packed` in NumPy.
   - Selection waits for the previous group's kernels at each pass's
     `flatnonzero` (one synchronisation per pass): a second stream for
     selection would overlap the two.
   - Kernels (0.30 s) are now the largest part.
2. **Other CUDA follow-ups**:
   - Multi-GPU (work server: 4 x Blackwell 6000 on CPU 0, 1 x Ada 6000 on
     CPU 1): one worker process per GPU, **spawned** (never fork after CUDA
     is initialised), each with `MWIR_CUDA_DEVICE`.
   - GPU-aware runtime estimate; GUI control for the backend.
   - CuPy warns "CUDA path could not be detected" on the desktop; harmless
     with `cupy-cuda12x[ctk]`, silence by setting `CUDA_PATH` if wanted.
   - Consider making `cuda` the default when a GPU is present: the full
     band is now 35x faster than the CPU backend on the desktop.
   - DISORT, Monte Carlo and Mie aerosol optics on the CUDA backend use the
     host path: optical depths are copied back once and the rest runs on the
     CPU (2.8 s for the full band with the single-scatter solver, against
     0.95 s device-resident). Not profiled with those solvers.
   - Keep `lbl_cuda.evaluate_emulated` in step with the kernel source;
     `tests/test_lbl_cuda.py` relies on it.
3. **Work server scaling**: `python scripts/validation/scaling_benchmark.py
   --out docs/validation/scaling_benchmark_server.json`. With 384 logical CPUs
   the default band only splits into ~130 chunks (`_MIN_CHUNK_FINE_POINTS`,
   `_MIN_CHUNK_OVER_SUPPORT` in `resolved.py`), so parallelism caps there.
4. **Cost estimate**: round 2 measured 0.08 M elements/s per worker at the
   new step (1 worker) and 36 s for the full band on 16 workers; the
   estimate (`_COST_RESOLVED_RATE_PER_WORKER = 150_000`) says ~68 s on 8, so it
   errs slow by ~2x. Recalibrate from `validation/scaling_round2.json`.
5. **Smaller items**
   - The 1 GB memory gate (`run_cost.CONFIRM_BYTES_THRESHOLD`) trips for the
     default resolved run at >= 8 workers (one chunk per worker is resident),
     so the GUI asks for confirmation on the desktop. Raise the gate or cap
     per-worker chunk memory.
   - Point sampling still uses threads (GIL-bound) and the pre-rewrite cost
     constants; it is fast anyway (~10 s default).

## Traps

- **Windows spawns worker processes**: any script calling `run_scenario` on a
  large resolved run needs `if __name__ == "__main__":`. `MWIR_PARALLEL=threads`
  avoids processes.
- **Tests**: physics tests pin `spectral_sampling=POINT` (their thresholds
  predate resolved sampling); GUI tests start via `_point_app()` because a
  resolved full-band first render takes minutes. New tests should follow suit
  unless they are about sampling.
- **NumPy or CuPy after line-by-line**: continuum, aerosol, Rayleigh,
  `path_transmittance`, `compute_radiance`, `planck` and `resolved.degrade`
  take their array module from their spectral input
  (`mwir_atmos._array.array_namespace`), so resolved CUDA chunks run them on
  the GPU. New code there must not build spectral arrays with `np.zeros`,
  `np.array` or `np.asarray`; use the array module (`xp`). CuPy raises a
  `TypeError` on mixing, and `test_device_resident_chunks_match_the_host_path`
  (GPU only) exercises the whole single-scatter path.
- **Never loosen a physics threshold to match new output.** Use an independent
  reference (`scripts/physics_reference/generate_hapi_reference.py`,
  dev-only `hitran-api==1.3.0.0`).
- **Validation scripts** write into `docs/validation/`; `run_all.py` skips a
  step whose output exists (`--force` to redo). Push logs even on failure.
- **Kernel edits without a GPU**: `pip install nvidia-cuda-nvrtc-cu12` then
  `python scripts/validation/nvrtc_compile_check.py` (sm_89, sm_120).
- **Environment variables**: `MWIR_THREADS` (workers), `MWIR_PARALLEL`
  (`processes`|`threads`), `MWIR_BACKEND` (`cpu`|`cuda`), `MWIR_CUDA_DEVICE`.
- **Fast tests**: `pip install pytest-xdist`, then `python -m pytest -n auto`
  (~1.5 min on 4 cores).
- **Old desktop stash**: the desktop may still hold `stash@{0}` "desktop WIP
  (moved to cloud as f807792)"; it duplicates `f807792` and can be dropped
  once `git diff --stat f807792 "stash@{0}"` is empty.
