"""Invalid inputs must stop before any expensive physics or network work."""
import dataclasses
import numpy as np
import pytest

from mwir_atmos.api import run_scenario
from mwir_atmos.cli import scenario_from_dict
from mwir_atmos.types import Scenario, SpectralSampling
from mwir_atmos.spectral import make_grid, convolve_gaussian


@pytest.mark.parametrize("field,value", [
    ("include_solar", "false"), ("include_lunar", 1),
    ("visibility_km", None), ("visibility_km", True),
    ("resolution_cm1", float("nan")), ("sensor_alt_km", float("inf")),
    ("surface_emissivity", 1.1), ("ground_rh_pct", -1),
    ("source_temp_k", -100), ("latitude_deg", 91),
    ("wavenumber_min_cm1", -1), ("line_strength_cutoff", 2),
    ("fine_step_cm1", 0.0), ("instrument_fwhm_cm1", -1.0),
    ("line_optical_depth_cutoff", -1.0e-6), ("fine_step_cm1", float("nan")),
])
def test_invalid_config_and_python_inputs_fail_before_optics(field, value, monkeypatch):
    monkeypatch.setattr("mwir_atmos.api.compute_layer_optics",
                        lambda *a, **k: pytest.fail("invalid input reached optics"))
    with pytest.raises(ValueError, match=field):
        scenario_from_dict({field: value})
    with pytest.raises(ValueError, match=field):
        run_scenario(dataclasses.replace(Scenario(), **{field: value}))


@pytest.mark.parametrize("lo,hi,step", [(-1, 10, 1), (0, 10, 1),
    (2000, float("inf"), 1), (float("nan"), 3000, 1), (2000, 3000, float("nan"))])
def test_grid_rejects_nonphysical_or_nonfinite_parameters(lo, hi, step):
    with pytest.raises(ValueError):
        make_grid(lo, hi, step)


def test_convolution_preserves_independent_cube_spectra():
    grid = make_grid(2000, 2100, 1)
    cube = np.zeros((2, 3, grid.n))
    cube[0, 1, 50] = 1
    cube[1, 2, :] = 4
    result = convolve_gaussian(grid, cube, 5)
    assert result.shape == cube.shape
    np.testing.assert_allclose(result[1, 2], 4)
    assert result[0, 1].sum() == pytest.approx(1)
    assert result[0, 0].sum() == 0
    assert 0 < result[0, 1, 50] < 1


def test_missing_cache_never_fetches_by_default(tmp_path, monkeypatch):
    from mwir_atmos.hitran import fetch
    monkeypatch.setattr(fetch, "fetch_linelist",
                        lambda *a, **k: pytest.fail("unexpected network access"))
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        fetch.ensure_linelists(cache_dir=tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_explicit_download_populates_cache(tmp_path, monkeypatch):
    from mwir_atmos.hitran import fetch
    original = fetch.ensure_linelists()
    calls = []
    def download(species, *args):
        calls.append(species)
        return original[species]
    monkeypatch.setattr(fetch, "fetch_linelist", download)
    monkeypatch.setattr(fetch.time, "sleep", lambda *a: None)
    fetch.ensure_linelists(cache_dir=tmp_path, allow_download=True)
    assert set(calls) == set(fetch.MOLECULE_IDS)
    calls.clear()
    fetch.ensure_linelists(cache_dir=tmp_path)
    assert not calls


def test_cost_estimate_does_not_allocate_an_enormous_grid(monkeypatch):
    from mwir_atmos import cli
    monkeypatch.setattr(cli, "make_grid", lambda *a, **k: pytest.fail("grid allocation in estimate"))
    cost = cli._estimate_run_cost(Scenario(resolution_cm1=1e-8,
                                             spectral_sampling=SpectralSampling.POINT))
    assert cost.n_points == 250_000_000_001
    assert cost.est_bytes > 1e12
