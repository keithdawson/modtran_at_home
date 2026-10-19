# Verification — 2026-09-08

The original source and scientific data were pushed to the private GitHub
repository before application changes, at commit `8f94877`.

## Changes

- Validate scenario types, finite numbers, physical bounds and spectral ranges
  consistently across Python, YAML, CLI and GUI inputs. Quoted booleans no longer
  silently enable illumination.
- Estimate grid cost without allocating the requested grid first; support
  Gaussian smoothing of multidimensional spectral arrays.
- Require an explicit request to download missing HITRAN line lists. Package
  all eight scientific tables in wheels and resolve their installed locations.
- Keep optional GUI controls inside their expanders, shorten the title, preserve
  component colors when selections change, and bound the scenario cache to eight
  entries. Display wavelength radiance per micrometre with the correct Jacobian;
  retain internal radiance units in CSV exports and wavenumber plots.
- Correct deployment GUI paths and CSV output, propagate Windows installer and
  wheel-download failures, and exclude generated artifacts from Docker contexts.
- Add 23 tests, a reproducible example-output runner, and Windows/Linux GitHub CI.

## Executed checks

| Check | Result |
| --- | --- |
| Original Windows suite | 429 passed, 409.55 s |
| Final Windows suite, Python 3.11 | 452 passed, 453.99 s |
| Rocky Linux 9 container, Python 3.11.13, network disabled | 452 passed, 170.57 s |
| Container CLI maritime example, network disabled | Passed; CSV generated |
| Standard wheel installation outside checkout | Passed; installed scientific tables loaded and radiance computed |
| All five YAML examples at their configured resolutions | Passed; finite/bounded transmittance and component-sum checks |
| Streamlit reruns | Passed; axes, numerical radiance conversion, cache reuse, expanders and invalid-input feedback |
| Python compilation, PowerShell parsing, Compose configuration, Git whitespace checks | Passed |
| Dated ZIP and tar.gz bundles | Built; selected source/data hashes match checkout; ZIP integrity passed |

The full example runs produced 2,501 spectral samples each. Recorded Windows
runtimes were 42.77 s for the fine CO line, 8.36 s for horizontal haze, 11.16 s
for the hot target, 9.54 s for the limb view, and 8.28 s for the maritime case.
Other verification processes were running concurrently; these are observations,
not controlled performance benchmarks.

## Visual review and artifacts

Inspected the live GUI before and after changes, including the sidebar layout,
spectral charts, radiance units and atmospheric profiles. Reviewed PNG plots of
transmittance and radiance for all five scenarios, including the resolved CO
absorption feature and the hot-target spectrum.

Local outputs (excluded from Git):

- `artifacts/examples/`: scenario JSON, CSV, ten individual PNG charts and `overview.png`.
- `artifacts/pytest-results.xml`: final Windows test results.
- `artifacts/docker-verification.log`: full offline Linux test and CLI log.
- `artifacts/wheel/mwir_atmos-0.1.0-py3-none-any.whl`: improved installable wheel.
- `dist/modtran_at_home_airgap_verified_20260908.zip` and `.tar.gz`: refreshed
  bundles, approximately 506 MB each. Original distribution archives remain.
- `artifacts/bundle-checksums.json`: SHA-256 checksums for refreshed bundles.

These checks establish software behavior and the existing physical invariants;
they do not establish equivalence to a licensed MODTRAN reference calculation.
The existing model approximations and limitations in README.md still apply.
Rocky Linux was built and executed locally; the Ubuntu Docker image was not
separately built. Initial Docker builds need base-image/OS-repository access;
the completed verification container ran without networking.
