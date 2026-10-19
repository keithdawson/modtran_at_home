"""Tests for mwir_atmos.cli.

Covers: main() with various argv, --config/--save-config round-tripping, every
output format (table/csv/json/npz) actually parsing back into structured
data, and every error path returning a non-zero exit code with a message on
stderr.

Speed: mwir_atmos.optics.lbl reloads the HITRAN line-list cache from disk on
every call (no in-process memoisation -- see NOTES in the job report), which
is a real, unavoidable ~4-8 s fixed cost per full run_scenario() regardless
of how small the grid is. This file keeps the number of tests that actually
run the physics (i.e. call main() without hitting an error/refusal before
run_scenario()) to a handful, all sharing one narrow (10 cm^-1), coarse
(2 cm^-1) window; everything else -- argument parsing, band-token parsing,
scenario (de)serialisation, and the confirmation-gate REFUSAL path -- is
tested without ever reaching the line-by-line engine.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from datetime import datetime

import numpy as np
import pytest
import yaml

from mwir_atmos import cli, report, run_cost
from mwir_atmos.constants import BANDS_CM1
from mwir_atmos.types import (
    AerosolModel,
    InstrumentFunction,
    ModelAtmosphere,
    Scenario,
    SpectralSampling,
    SurfaceType,
)

#: Shared fast scenario: 10 cm^-1 window (6 points at 2 cm^-1 resolution),
#: sea-level nadir path -- real physics, real HITRAN lines, but nowhere near
#: the "narrow window at fine resolution" cost regime this package protects
#: against (see README.md "Choosing spectral resolution").
_FAST_ARGV = [
    "--wavenumber-min-cm1", "2300", "--wavenumber-max-cm1", "2310",
    "--resolution-cm1", "2.0",
]


def _run(argv, **kwargs):
    return cli.main(argv, **kwargs) if kwargs else cli.main(argv)


# =========================================================================
# main() contract
# =========================================================================


def test_main_returns_int_and_is_importable():
    assert callable(cli.main)
    rc = cli.main(["--list-bands"])
    assert isinstance(rc, int)
    assert rc == cli.EXIT_OK


def test_version_flag_exits_zero(capsys):
    rc = cli.main(["--version"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "mwir_atmos" in out


def test_help_flag_exits_zero(capsys):
    rc = cli.main(["--help"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "usage:" in out.lower()


def test_list_bands_prints_catalog_and_exits_zero(capsys):
    rc = cli.main(["--list-bands"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    for name in BANDS_CM1:
        assert name in out


# =========================================================================
# Error paths -- every one below returns non-zero WITHOUT reaching the
# line-by-line engine (cheap: no HITRAN I/O), and prints a message on stderr.
# =========================================================================


def test_missing_config_file_returns_runtime_error(capsys):
    rc = cli.main(["--config", "this/path/does/not/exist.yaml"])
    assert rc == cli.EXIT_RUNTIME_ERROR
    err = capsys.readouterr().err
    assert "not found" in err


def test_config_with_unknown_field_returns_runtime_error(tmp_path, capsys):
    bad = tmp_path / "typo.yaml"
    bad.write_text("visiblity_km: 5.0\n", encoding="utf-8")  # typo'd field name
    rc = cli.main(["--config", str(bad)])
    assert rc == cli.EXIT_RUNTIME_ERROR
    err = capsys.readouterr().err
    assert "unknown scenario field" in err


def test_config_that_is_not_a_yaml_mapping_returns_runtime_error(tmp_path, capsys):
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    rc = cli.main(["--config", str(bad)])
    assert rc == cli.EXIT_RUNTIME_ERROR
    assert "did not parse to a YAML mapping" in capsys.readouterr().err


def test_bad_local_datetime_returns_runtime_error(capsys):
    rc = cli.main(["--local-datetime", "not-a-real-date"])
    assert rc == cli.EXIT_RUNTIME_ERROR
    assert "invalid --local-datetime" in capsys.readouterr().err


def test_unknown_band_token_returns_usage_error(capsys):
    rc = cli.main([*_FAST_ARGV, "--bands", "not_a_real_band_name"])
    assert rc == cli.EXIT_USAGE_ERROR
    assert "unknown band" in capsys.readouterr().err


def test_bad_band_range_returns_usage_error(capsys):
    rc = cli.main([*_FAST_ARGV, "--bands", "2650:2600"])  # max <= min
    assert rc == cli.EXIT_USAGE_ERROR
    assert "invalid band range" in capsys.readouterr().err


def test_format_npz_without_output_returns_usage_error(capsys):
    rc = cli.main([*_FAST_ARGV, "--format", "npz"])
    assert rc == cli.EXIT_USAGE_ERROR
    assert "--format npz requires --output" in capsys.readouterr().err


def test_unreachable_sensor_altitude_returns_runtime_error(capsys):
    """sensor_alt_km below the model atmosphere's base level -- caught by
    _estimate_run_cost (via build_scenario_path) before any HITRAN I/O."""
    rc = cli.main([*_FAST_ARGV, "--sensor-alt-km", "-5"])
    assert rc == cli.EXIT_RUNTIME_ERROR
    err = capsys.readouterr().err
    assert "sensor_alt_km" in err


def test_unreachable_source_altitude_returns_runtime_error(capsys):
    rc = cli.main([
        *_FAST_ARGV, "--sensor-zenith-deg", "150", "--sensor-alt-km", "0",
        "--source-alt-km", "50",  # above this line of sight's tangent altitude
    ])
    assert rc == cli.EXIT_RUNTIME_ERROR
    assert "error resolving scenario geometry" in capsys.readouterr().err


# =========================================================================
# Confirmation gate -- REFUSAL path never reaches run_scenario(), so this is
# cheap even though it exercises real cost-estimate + --save-config logic.
# =========================================================================


def test_confirm_gate_blocks_without_yes_and_still_saves_config(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(run_cost, "CONFIRM_BYTES_THRESHOLD", 1)
    monkeypatch.setattr(run_cost, "CONFIRM_SECONDS_THRESHOLD", 0.0)
    saved = tmp_path / "resolved.yaml"

    rc = cli.main([
        *_FAST_ARGV, "--quiet",  # -q: guarantees the interactive-prompt branch is skipped, never calls input()
        "--line-strength-cutoff", "1e-6", "--line-wing-cutoff-cm1", "3.0",
        "--save-config", str(saved),
    ])

    assert rc == cli.EXIT_USAGE_ERROR
    err = capsys.readouterr().err
    assert "refusing to run without confirmation" in err
    # --save-config runs before the gate, so it must still have written the
    # fully-resolved scenario, including the accuracy overrides.
    assert saved.exists()
    data = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert data["line_strength_cutoff"] == pytest.approx(1.0e-6)
    assert data["line_wing_cutoff_cm1"] == pytest.approx(3.0)
    assert data["wavenumber_min_cm1"] == pytest.approx(2300.0)


def test_confirm_gate_bypassed_with_yes_runs_successfully(capsys, monkeypatch):
    monkeypatch.setattr(run_cost, "CONFIRM_BYTES_THRESHOLD", 1)
    monkeypatch.setattr(run_cost, "CONFIRM_SECONDS_THRESHOLD", 0.0)

    rc = cli.main([*_FAST_ARGV, "--yes", "--format", "table"])

    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "mwir_atmos scenario summary" in out


# =========================================================================
# Output formats -- each of these is a real (small) run_scenario() call.
# =========================================================================


def test_format_table_end_to_end(capsys):
    rc = cli.main([
        *_FAST_ARGV, "--atmosphere", "midlat_summer", "--aerosol", "rural",
        "--bands", "CO2_4_3um", "2302:2308",
    ])
    assert rc == cli.EXIT_OK
    captured = capsys.readouterr()

    out = captured.out
    assert "mwir_atmos scenario summary" in out
    assert "atmosphere        : midlat_summer" in out
    assert "aerosol           : rural" in out
    assert "CO2_4_3um" in out
    assert "2302-2308cm-1" in out
    assert "band-averaged transmittance by species" in out
    # The radiometry layer has since landed, so the table must carry a real
    # band-integrated radiance section rather than the transmittance-only
    # fallback this suite originally asserted.
    assert "radiance          : unavailable" not in out
    assert "radiance unavailable -- transmittance-only summary" not in out
    assert "band-integrated radiance" in out

    err = captured.err
    assert "estimated cost:" in err


def test_format_table_is_quiet_with_q_flag(capsys):
    rc = cli.main([*_FAST_ARGV, "--quiet"])
    assert rc == cli.EXIT_OK
    err = capsys.readouterr().err
    assert err == ""


def test_format_csv_has_full_spectral_arrays(tmp_path):
    out_path = tmp_path / "spectrum.csv"
    rc = cli.main([*_FAST_ARGV, "--quiet", "--format", "csv", "--output", str(out_path)])
    assert rc == cli.EXIT_OK

    with out_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # (4300 replaced by our narrow window) 10 cm^-1 / 2 cm^-1 + 1 = 6 points
    assert len(rows) == 6
    fieldnames = set(rows[0].keys())
    for col in ("wavenumber_cm1", "wavelength_um", "trans_total", "trans_molecular", "trans_aerosol", "trans_rayleigh"):
        assert col in fieldnames
    for species in ("H2O", "CO2", "O3", "N2O", "CO", "CH4"):
        assert f"trans_{species}" in fieldnames
    # Radiance has landed, so its columns must now be present and populated.
    # (When ScenarioResult.radiance is None they are omitted entirely rather
    # than written as empty or zero -- never fake radiance numbers.)
    assert "radiance_total" in fieldnames
    assert "radiance_path_thermal" in fieldnames
    assert all(float(row["radiance_total"]) >= 0.0 for row in rows)

    assert float(rows[0]["wavenumber_cm1"]) == pytest.approx(2300.0)
    assert float(rows[-1]["wavenumber_cm1"]) == pytest.approx(2310.0)
    for row in rows:
        t = float(row["trans_total"])
        assert 0.0 <= t <= 1.0 + 1.0e-9


def test_format_json_has_full_spectral_arrays_and_radiance(capsys):
    rc = cli.main([*_FAST_ARGV, "--quiet", "--format", "json"])
    assert rc == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["radiance"] is not None
    assert len(payload["radiance"]["total"]) == 6
    assert set(payload["radiance"]).issuperset({"total", "path_thermal", "surface_emitted"})
    assert payload["scenario"]["wavenumber_min_cm1"] == pytest.approx(2300.0)
    assert len(payload["transmittance"]["total"]) == 6
    assert set(payload["transmittance"]["by_species"]) == {"H2O", "CO2", "O3", "N2O", "CO", "CH4"}
    # default --bands is "every named band" -- all of BANDS_CM1 should appear
    # as keys, even ones that don't overlap this narrow window (those are
    # just skipped with a warning by report.band_rows, not an error).
    assert set(payload["bands"]).issubset(set(BANDS_CM1))
    assert "CO2_4_3um" in payload["bands"]
    assert payload["bands"]["CO2_4_3um"]["radiance"] is not None


def test_format_npz_has_full_spectral_and_band_arrays(tmp_path):
    out_path = tmp_path / "spectrum.npz"
    rc = cli.main([*_FAST_ARGV, "--quiet", "--format", "npz", "--output", str(out_path), "--bands", "CO2_4_3um"])
    assert rc == cli.EXIT_OK

    with np.load(out_path) as data:
        assert data["wavenumber_cm1"].shape == (6,)
        assert data["trans_total"].shape == (6,)
        assert data["trans_H2O"].shape == (6,)
        assert data["radiance_total"].shape == (6,)   # radiance has landed
        assert list(data["band_label"]) == ["CO2_4_3um"]
        assert data["band_trans_total"].shape == (1,)


# =========================================================================
# --config / --save-config round trip and flag-override composition
# =========================================================================


def test_config_round_trip_and_flag_override(tmp_path):
    base = Scenario(
        atmosphere=ModelAtmosphere.TROPICAL,
        aerosol=AerosolModel.NAVY_MARITIME,
        surface_type=SurfaceType.SEA_WATER,
        visibility_km=12.0,
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2310.0,
        resolution_cm1=2.0,
        local_datetime=datetime(2026, 7, 15, 13, 30, 0),
    )
    cfg_path = tmp_path / "in.yaml"
    cli.save_scenario_yaml(base, cfg_path)

    resolved_path = tmp_path / "resolved.yaml"
    rc = cli.main([
        "--config", str(cfg_path), "--quiet",
        "--visibility-km", "20",  # flag override on top of the config file
        "--save-config", str(resolved_path), "--format", "npz", "-o", str(tmp_path / "x.npz"),
    ])
    assert rc == cli.EXIT_OK

    resolved = cli.load_scenario_yaml(resolved_path)
    assert resolved.atmosphere is ModelAtmosphere.TROPICAL           # from the config file
    assert resolved.aerosol is AerosolModel.NAVY_MARITIME            # from the config file
    assert resolved.visibility_km == pytest.approx(20.0)             # flag override wins
    assert resolved.local_datetime == datetime(2026, 7, 15, 13, 30, 0)


# =========================================================================
# Unit-level tests for the (de)serialisation and parsing helpers -- no
# main() call, no physics, effectively instantaneous.
# =========================================================================


def test_scenario_to_dict_from_dict_round_trip():
    sc = Scenario(
        atmosphere=ModelAtmosphere.SUBARCTIC_WINTER,
        surface_type=SurfaceType.SNOW,
        aerosol=AerosolModel.DESERT,
        ground_temp_k=250.0,
        local_datetime=datetime(2026, 1, 2, 3, 4, 5),
        line_strength_cutoff=1.0e-7,
        line_wing_cutoff_cm1=5.0,
    )
    data = cli.scenario_to_dict(sc)
    assert data["atmosphere"] == "subarctic_winter"
    assert data["local_datetime"] == "2026-01-02T03:04:05"

    back = cli.scenario_from_dict(data)
    assert back == sc


def test_scenario_from_dict_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown scenario field"):
        cli.scenario_from_dict({"not_a_real_field": 1.0})


def test_scenario_from_dict_fills_defaults_for_omitted_fields():
    sc = cli.scenario_from_dict({"aerosol": "urban"})
    assert sc.aerosol is AerosolModel.URBAN
    assert sc.atmosphere is Scenario().atmosphere  # untouched fields keep their dataclass default


def test_save_and_load_scenario_yaml_round_trip(tmp_path):
    sc = Scenario(source_temp_k=650.0, source_range_km=5.0, moon_phase=0.25)
    path = tmp_path / "scenario.yaml"
    cli.save_scenario_yaml(sc, path)
    loaded = cli.load_scenario_yaml(path)
    assert loaded == sc


def test_load_scenario_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        cli.load_scenario_yaml("definitely/does/not/exist.yaml")


@pytest.mark.parametrize(
    "token,expected_label,expected_spec",
    [
        ("CO2_4_3um", "CO2_4_3um", "CO2_4_3um"),
        ("2600:2650", "2600-2650cm-1", (2600.0, 2650.0)),
    ],
)
def test_parse_band_token(token, expected_label, expected_spec):
    label, spec = report.parse_band_token(token)
    assert label == expected_label
    assert spec == expected_spec


def test_parse_band_token_invalid_range_raises():
    with pytest.raises(ValueError, match="MAX must exceed MIN"):
        report.parse_band_token("2650:2600")


def test_parse_band_token_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown band"):
        report.parse_band_token("not_a_real_band_name")


def test_resolve_band_specs_default_is_every_named_band():
    specs = report.resolve_band_specs(None)
    assert {name for name, _ in specs} == set(BANDS_CM1)


def test_resolve_band_specs_empty_list_is_no_bands():
    assert report.resolve_band_specs([]) == []


def test_build_parser_exposes_every_scenario_field():
    """Regression guard: every Scenario dataclass field must have a CLI
    flag. (This caught line_strength_cutoff/line_wing_cutoff_cm1 missing
    entirely from an earlier draft of this module.)"""
    field_names = {f.name for f in dataclasses.fields(Scenario)}
    parser = cli.build_parser()
    dests = {a.dest for a in parser._actions if a.dest and a.dest != "help"}
    missing = field_names - dests
    assert not missing, f"Scenario fields with no CLI flag: {sorted(missing)}"


def test_accuracy_flags_are_settable_and_documented_in_help(capsys):
    rc = cli.main(["--help"])
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "--line-strength-cutoff" in out
    assert "--line-wing-cutoff-cm1" in out


# =========================================================================
# Cost estimate helpers -- pure/cheap (geometry only, no HITRAN I/O)
# =========================================================================


def test_estimate_run_cost_scales_with_resolution():
    coarse = Scenario(spectral_sampling=SpectralSampling.POINT, wavenumber_min_cm1=2300.0, wavenumber_max_cm1=2310.0, resolution_cm1=2.0)
    fine = dataclasses.replace(coarse, resolution_cm1=0.5)

    coarse_cost = run_cost.estimate_run_cost(coarse)
    fine_cost = run_cost.estimate_run_cost(fine)

    assert fine_cost.n_points > coarse_cost.n_points
    assert fine_cost.est_bytes > coarse_cost.est_bytes
    assert fine_cost.est_seconds > coarse_cost.est_seconds


def test_estimate_run_cost_reflects_wing_cutoff_at_fine_resolution():
    """The per-line evaluation window (2*wing/resolution + 1) can dominate
    over grid size at fine resolution -- shrinking line_wing_cutoff_cm1 to
    match a narrow window should visibly lower the estimate (this is the
    mechanism behind the "narrow + fine stays fast" epilog example)."""
    base = Scenario(
        spectral_sampling=SpectralSampling.POINT, wavenumber_min_cm1=2171.5, wavenumber_max_cm1=2174.0, resolution_cm1=0.01,
        line_wing_cutoff_cm1=25.0,
    )
    narrow_wing = dataclasses.replace(base, line_wing_cutoff_cm1=3.0)

    assert run_cost.estimate_run_cost(narrow_wing).est_seconds < run_cost.estimate_run_cost(base).est_seconds


def test_format_bytes_and_duration():
    assert run_cost.format_bytes(500) == "500 B"
    assert "KB" in run_cost.format_bytes(2048)
    assert "MB" in run_cost.format_bytes(5 * 1024**2)
    assert "GB" in run_cost.format_bytes(2 * 1024**3)
    assert run_cost.format_duration(30) == "30.0 s"
    assert "min" in run_cost.format_duration(150)
    assert "h" in run_cost.format_duration(4000)


def test_spectral_sampling_and_line_selection_flags_reach_the_scenario():
    args = cli.build_parser().parse_args([
        "--spectral-sampling", "resolved", "--fine-step-cm1", "0.01",
        "--instrument-function", "triangle", "--instrument-fwhm-cm1", "2.0",
        "--line-optical-depth-cutoff", "1e-6",
    ])
    scenario = cli._resolve_scenario(args)
    assert scenario.spectral_sampling is SpectralSampling.RESOLVED
    assert scenario.fine_step_cm1 == pytest.approx(0.01)
    assert scenario.instrument_function is InstrumentFunction.TRIANGLE
    assert scenario.instrument_fwhm_cm1 == pytest.approx(2.0)
    assert scenario.line_optical_depth_cutoff == pytest.approx(1.0e-6)
    # Defaults: resolved sampling, peak-optical-depth selection, legacy rule off.
    default = cli._resolve_scenario(cli.build_parser().parse_args([]))
    assert default.spectral_sampling is SpectralSampling.RESOLVED
    assert default.line_strength_cutoff == 0.0


@pytest.mark.parametrize("value", ["0", "-2", "two"])
def test_threads_flag_rejects_non_positive_integers(value, capsys):
    assert _run(["--threads", value, *_FAST_ARGV]) == cli.EXIT_USAGE_ERROR
    assert "positive integer" in capsys.readouterr().err


def test_cost_estimate_counts_the_fine_grid_for_resolved_sampling():
    point = Scenario(spectral_sampling=SpectralSampling.POINT, wavenumber_min_cm1=2300.0, wavenumber_max_cm1=2310.0, resolution_cm1=1.0)
    fine = dataclasses.replace(point, spectral_sampling=SpectralSampling.RESOLVED, fine_step_cm1=0.01)
    a, b = run_cost.estimate_run_cost(point), run_cost.estimate_run_cost(fine)
    assert a.n_points == b.n_points == 11
    assert b.est_seconds > a.est_seconds
