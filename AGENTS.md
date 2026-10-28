# mwir_atmos (MODTRAN-at-home)

Mid-wave infrared atmospheric transmittance and radiance: HITRAN line-by-line
Voigt engine, MT_CKD continuum, aerosols, single-scatter / DISORT / spherical
Monte Carlo radiance; CLI (`mwir`), Streamlit GUI, Python API
(`mwir_atmos.api.run_scenario`). Owner-only project: pushing to `main` is fine
when a change improves the model and tests pass.

**Start here:** [docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md)
(current state, open work, traps), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(data flow, module map, where to make common changes), then
[docs/spectral_sampling.md](docs/spectral_sampling.md) (line selection,
resolved sampling, validation results, GPU backend) and
[docs/validation/README.md](docs/validation/README.md) (runbook).

## Machines

- Home desktop: Windows, 8 cores, RTX 4090. Repo at `D:\AI-projects\modtran_at_home`.
- Work server: airgapped Rocky Linux 9, 2 x EPYC 9665 (384 threads), 2.3 TB RAM,
  4 x Blackwell 6000 + 1 x Ada 6000 (`SERVER CONFIGURATION.txt`,
  `OFFLINE_DEPLOYMENT.md`).
- The user may be training on the desktop; ask before long CPU/GPU runs.

## Commands

```bash
pip install -e ".[dev]"                     # add [gui] for the GUI, [cuda] for the GPU backend
python -m pytest -n auto                    # needs pytest-xdist; ~1.5 min
python scripts/validation/run_all.py        # validation studies (skip-if-done)
python scripts/validation/cuda_check.py     # GPU tests + CPU vs GPU timing
mwir --help                                 # CLI; --threads N, --backend auto|cpu|cuda
```

## Conventions

- Physics tests pin `spectral_sampling=POINT`; GUI tests use `_point_app()`.
- Never change a physics threshold to match new output; use an independent
  reference (HAPI script in `scripts/physics_reference/`).
- Scripts that run large resolved scenarios need `if __name__ == "__main__":`
  (worker processes are spawned on Windows).
- Record measurements and decisions in `docs/spectral_sampling.md` /
  `docs/validation/`, with the numbers.
