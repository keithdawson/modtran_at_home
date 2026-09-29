# Development status and handoff

Written at the end of a cloud session (29 September 2026) so a local session
can pick up the work. Physics and results live in
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
| next | Handoff docs (this file, `CLAUDE.md`) |

Test suite: 610 passed, 3 skipped (GPU-only) on Linux; the GPU tests pass on
the RTX 4090.

## Decisions and their evidence

| Decision | Evidence | Where |
|---|---|---|
| `line_optical_depth_cutoff` 3e-6 | every 1 cm^-1 window bin within 0.078 pp of all lines (1e-5: 0.19 pp) on 5 paths x 3 windows | `validation/line_cutoff_sweep.json` |
| 1e-8 is a valid "all lines" reference | within 0.0005 pp of cutoff 0, half the cost | cloud check, same sweep script |
| Auto fine step sized at 10% of the air column | all paths <= 0.045 pp / 0.11% worst bin; vertical paths 4-6x coarser than the old 1% rule, full default run ~25 min -> ~4 min on 4 cores | `validation/fine_step_convergence.json`, cloud re-check |
| Binned DISORT/MC radiance + exact single-scatter correction | uncorrected binning: 40-47% worst bin, 2-4.5% window on nadir/limb; corrected worst nadir case 0.055% / -0.012% | `validation/correlated_k_accuracy.json`; full re-check is round 2 |
| Processes, not threads, for resolved runs | 5.4x at 8 workers vs threads getting *slower* (GIL) | `validation/scaling_benchmark.json` |
| Window-test thresholds (subarctic > 0.80, look-down > 0.88) | independent HAPI line-by-line reference; model agrees to 0.001 T, 0.38% L | `tests/reference_data/hapi_window_reference.json` |
| Cost estimate: 0.15 M elements/s/worker, speedup workers**0.7 | desktop measured 0.23 M at the *old* step | `mwir_atmos/cli.py` `_COST_*` |
| GPU: float32 wings, float64 cores/offsets/sums | GPUs run fp64 at 1/64 rate; profile within 4.7e-7 of wofz | `optics/voigt_gpu.py` |

## Open work, in order

Round 2 and the RTX 4090 check are done (commit `026f4ad`; results in
[spectral_sampling.md](spectral_sampling.md#validation-results)): corrected
binning <= 0.49% worst bin on every path, fine step confirmed, GPU tests pass
with 14.3x on a 200 cm^-1 slice but only 4.8x on the full band (7.6 s vs
36.3 s), because the CPU side dominates there.

1. **Make the GPU path's CPU side fast** (biggest remaining win for the
   full band). Profile `run_scenario(Scenario(), backend="cuda")` first.
   Likely: overlap chunk k+1's line selection with chunk k's kernel, run the
   non-LBL per-chunk work (continuum, transmittance, radiance, instrument
   reduction) in worker processes or on the GPU, and consider line selection
   on the GPU.
2. **Other CUDA follow-ups**:
   - Multi-GPU (work server: 4 x Blackwell 6000 on CPU 0, 1 x Ada 6000 on
     CPU 1): one worker process per GPU, **spawned** (never fork after CUDA
     is initialised), each with `MWIR_CUDA_DEVICE`.
   - GPU-aware runtime estimate; GUI control for the backend.
   - CuPy warns "CUDA path could not be detected" on the desktop; harmless
     with `cupy-cuda12x[ctk]`, silence by setting `CUDA_PATH` if wanted.
   - Consider making `cuda` the default when a GPU is present, once the
     full-band speedup is worth it.
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
   - The 1 GB memory gate (`cli._CONFIRM_BYTES_THRESHOLD`) trips for the
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
