# Validation runbook

Studies behind the [pending validation items](../spectral_sampling.md#pending-validation).
Each one runs many independent jobs, one per worker process, and writes its
results as JSON into this folder. Run them anywhere; results do not depend on
the machine or the worker count, only the timings do.

## Run everything (home desktop, 8 cores: roughly 1.5-2 h)

```powershell
git pull
pip install -e ".[dev]"
pip install hitran-api==1.3.0.0          # step 4 only; dev tool, not a runtime dependency
python scripts/validation/run_all.py     # add --jobs 6 to keep two cores free
git add docs/validation tests/reference_data/hapi_window_reference.json
git commit -m "Validation results"
git push
```

`run_all.py` skips steps whose result file already exists (`--force` to
redo), keeps going if one fails, and logs each step to `logs/<step>.log`.
Run a subset with `--only cutoff hapi`. Push the logs too, including after a
failure.

| Step | Script | Output | Rough time, 8 cores |
|---|---|---|---|
| `cutoff` | `line_cutoff_sweep.py` | `line_cutoff_sweep.json` | 30-45 min |
| `finestep` | `fine_step_convergence.py` | `fine_step_convergence.json` | 15 min |
| `corrk` | `correlated_k_accuracy.py` | `correlated_k_accuracy.json` | 20-30 min |
| `hapi` | `../../scripts/physics_reference/generate_hapi_reference.py` | `tests/reference_data/hapi_window_reference.json` | 15-30 min |
| `scaling` | `scaling_benchmark.py --sizes small` | `scaling_benchmark.json` | 10 min |

What each answers:

1. **cutoff**: resolved window transmittance at `line_optical_depth_cutoff`
   1e-3 to 1e-7, against 1e-8. 1e-8 agrees with keeping every line to
   0.0005 percentage points and is twice as fast. Five paths: tropical 10 km
   horizontal, mid-latitude-summer ground-to-space, 20 km nadir, 15 km limb,
   subarctic-winter 10 km horizontal. Picks the default that keeps window
   error under ~0.1 pp.
2. **finestep**: the automatic fine step and fixed steps 0.05-0.001 against
   0.0005 cm^-1, transmittance and thermal radiance, on four 20 cm^-1
   windows.
3. **corrk**: correlated-k binned radiance against the solver run at every
   fine point (single-scatter and DISORT, with and without the sun).
4. **hapi**: HITRAN's own HAPI evaluates the same lines independently, for
   the paths behind the window-band tests awaiting new thresholds.
   `tests/test_hapi_reference.py` compares the model with it (skipped until
   the file exists).
5. **scaling**: per-core throughput at 1, 2, 4 ... N workers, process vs
   thread mode, used to recalibrate the CLI/GUI runtime estimate.

## Round 2 (after the new defaults) -- done, commit `026f4ad`

Round 1 set the defaults: cutoff 3e-6, 10% fine-step rule, and the
single-scatter correction of binned DISORT/Monte Carlo radiance (see
[results](../spectral_sampling.md#validation-results)). Round 2 checks them.
Round-1 files already exist, so the same command runs only round 2:

```powershell
git pull
python scripts/validation/run_all.py
git add docs/validation
git commit -m "Validation results, round 2"
git push
```

| Step | Output | What it checks | Rough time, 8 cores |
|---|---|---|---|
| `corrk2` | `correlated_k_accuracy_round2.json` | binned vs corrected vs exact DISORT, 5 paths x 4 windows, day/night | 30-40 min |
| `finestep2` | `fine_step_convergence_round2.json` | automatic step under the 10% rule, plus fixed steps | 10 min |
| `default` | `scaling_round2.json` | per-core rate at the new step (1-8 workers), and the full default band at 8 workers | 10 min |

## GPU backend (home 4090, then the work server)

The CUDA line-by-line backend is new and off by default. Install CuPy with
its CUDA libraries, then run the check (it is also the `cuda` step of
`run_all.py`, skipped where there is no GPU):

```powershell
pip install "cupy-cuda12x[ctk]"      # CUDA 12 driver; use cupy-cuda13x[ctk] for CUDA 13
python scripts/validation/cuda_check.py
git add docs/validation
git commit -m "CUDA check"
git push
```

It runs `tests/test_lbl_cuda.py` on the GPU (kernel vs its NumPy twin and vs
the CPU engine), then times a 200 cm^-1 slice and the full default band on
CPU and GPU and records how far they differ. Output:
`cuda_check_<hostname>.json` (a few minutes; `--skip-cpu-full` skips the
slowest CPU run). If CuPy cannot compile the kernel, push the traceback --
it is usually a driver/toolkit mismatch (Blackwell needs CUDA 12.8+).

Use it anywhere with `--backend cuda` (CLI), `backend="cuda"`
(`run_scenario`) or `MWIR_BACKEND=cuda` (API, GUI, scripts).
`MWIR_CUDA_DEVICE=N` picks the GPU. After editing the kernel on a machine
without a GPU, `python scripts/validation/nvrtc_compile_check.py` compiles it
for sm_89 and sm_120 (needs `pip install nvidia-cuda-nvrtc-cu12`).

## Work server (2 x 96 cores)

Only the scaling benchmark needs the big machine: the full-band strong-scaling
run at 16 ... 384 workers (about 15 minutes).

```bash
python scripts/validation/scaling_benchmark.py --out docs/validation/scaling_benchmark_server.json
```

On the airgapped server, bring the repository over with its data as usual
(OFFLINE_DEPLOYMENT.md); the benchmark needs nothing beyond the runtime
dependencies.

The other steps finish in minutes there: `run_all.py --force --jobs 190`.

## Parallelism knobs

- `MWIR_THREADS` / `--threads` / `n_threads=`: worker count (default: every
  CPU, including SMT threads). Set it when sharing the machine.
- `MWIR_PARALLEL=threads`: never start worker processes. The default,
  `processes`, splits large resolved runs (>= 2^21 fine-point x layer
  elements) into chunks evaluated by worker processes. Results agree
  to rounding either way. On Windows, and wherever `fork` is unavailable,
  workers are spawned, and a script calling `run_scenario` needs the usual
  `if __name__ == "__main__":` guard.
