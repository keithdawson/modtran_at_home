# Offline & Air-Gapped Deployment Guide

This guide details how to deploy, run, and optimize `mwir_atmos` (MODTRAN-at-home) in strictly air-gapped (isolated, zero internet connectivity) environments across **Rocky Linux 9.6** (Pandora Server), **Ubuntu 20.04/22.04/24.04 LTS**, and **Windows 10/11 / Server 2022**.

---

## 1. Package Architecture & Air-Gap Compliance

`mwir_atmos` is engineered for 100% offline self-containment:
- **Spectroscopic Databases**: Pre-computed and cached line-by-line parameters for all target molecules ($\text{H}_2\text{O}, \text{CO}_2, \text{O}_3, \text{N}_2\text{O}, \text{CO}, \text{CH}_4$), MT_CKD v4.3 water vapor continuum tables, and TIPS-2021 total internal partition sums are bundled in `data/hitran/`.
- **Python Binary Wheels**: 88+ platform-matched binary wheels for Linux (`manylinux_x86_64`) and Windows (`win_amd64`) for Python 3.10, 3.11, and 3.12 are bundled in `wheels/`.
- **Zero Network Invariants**: All core solvers, geometry tracers, aerosol models, radiance integrators, and Web GUIs operate strictly without internet connectivity.

---

## 2. Quickstart Installation (Direct Bare-Metal)

### A. Rocky Linux 9.6 / RHEL 9 (Pandora Server)

1. Extract the air-gap bundle on the target server:
   ```bash
   tar -xvf modtran_at_home_airgap_v0.1.0.tar.gz
   cd modtran_at_home_airgap_v0.1.0
   ```
2. Run the automated offline installer:
   ```bash
   bash install_offline.sh
   ```
3. Activate the environment and use:
   ```bash
   source venv/bin/activate
   mwir --help
   ```

### B. Ubuntu 22.04 / 24.04 LTS

1. Extract and install:
   ```bash
   tar -xvf modtran_at_home_airgap_v0.1.0.tar.gz
   cd modtran_at_home_airgap_v0.1.0
   bash install_offline.sh
   ```

### C. Windows (10 / 11 / Server 2022)

1. Extract `modtran_at_home_airgap_v0.1.0.zip`.
2. Run `install_offline.bat` (Command Prompt) or `install_offline.ps1` (PowerShell).
3. Activate and use:
   ```cmd
   call venv\Scripts\activate.bat
   mwir --help
   ```

---

## 3. Docker Container Deployment & Offline Validation

The package includes pre-configured Docker definitions for Rocky Linux 9 and Ubuntu with hardware-level network isolation.

### Build and Run with Docker Compose

1. **Verify Offline Execution with Strict Zero-Network Mode (`--network none`)**:
   ```bash
   docker compose run --rm modtran-offline-verify
   ```

2. **Run Streamlit Web GUI (Port 8501)**:
   ```bash
   docker compose up -d modtran-gui
   # Access GUI at: http://localhost:8501
   ```

3. **Run Interactive Rocky Linux 9 Container**:
   ```bash
   docker compose run --rm modtran-rocky bash
   ```

4. **Run Ubuntu Container**:
   ```bash
   docker compose run --rm modtran-ubuntu bash
   ```

---

## 4. Hardware Optimization for Pandora Server

### Hardware Specification (`SERVER CONFIGURATION.txt`)
- **Motherboard**: Gigabyte G493-ZB3
- **OS**: Rocky Linux 9.6 (Air-Gapped)
- **CPUs**: 2x AMD EPYC 9665 (96 cores / 192 threads each $\to$ **384 total threads**)
- **Memory**: 24x 96 GB DDR5 5200MHz DIMMs (**2.3 TB RAM across 24 channels / 2 NUMA nodes**)
- **GPUs**: 4x NVIDIA Blackwell 6000 Server GPUs (96 GB GDDR7 VRAM each on CPU 0 via Broadcom PCIe 5.0 switch) + 1x Ada 6000 (48 GB on CPU 1)

### Tuning & Performance Guidelines

1. **NUMA Node Thread Binding**:
   To maximize DDR5 memory bandwidth and prevent inter-socket UPI cross-talk during heavy line-by-line calculations, bind execution to specific NUMA domains:
   ```bash
   # Run scenario on NUMA node 0 (CPU 0 cores + local memory)
   numactl --cpunodebind=0 --membind=0 mwir --config scenarios/tropical_maritime_quicklook.yaml --output run_cpu0.csv

   # Run scenario on NUMA node 1 (CPU 1 cores + local memory)
   numactl --cpunodebind=1 --membind=1 mwir --config scenarios/subarctic_winter_laser_path.yaml --output run_cpu1.csv
   ```

2. **OpenMP / BLAS Thread Management**:
   Set environment variables to prevent thread oversubscription when running batch simulations:
   ```bash
   export OMP_NUM_THREADS=96
   export OPENBLAS_NUM_THREADS=96
   export MKL_NUM_THREADS=96
   export NUMEXPR_NUM_THREADS=96
   ```

3. **Batch Scenario Processing**:
   With 384 CPU threads and 2.3 TB of memory, thousands of atmospheric paths and Monte Carlo turbulence realization sweeps can be executed concurrently using Python's `multiprocessing` or `concurrent.futures` across the entire core topology without memory constraints.

---

## 5. Verification & Health Check Commands

To run full validation inside any environment:

```bash
# 1. Run all 429 unit, regression, and physical benchmark tests
pytest tests/ -v

# 2. Run physical benchmark tests specifically
pytest tests/test_physical_benchmarks.py -v

# 3. Test CLI scenario execution
mwir --config scenarios/us_standard_downlook_clearsky.yaml --output test_run.csv
```
