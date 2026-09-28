"""Scriptable command-line interface for mwir_atmos.

DEV-BETA owned (CONTRACT.md "Wave 2 interfaces"). Thin argparse wrapper around
:func:`mwir_atmos.api.run_scenario` -- every :class:`~mwir_atmos.types.Scenario`
field is settable by flag, a scenario can be loaded from / saved to YAML, and
results can be emitted as a human-readable table or as CSV/JSON/NPZ containing
the full spectral arrays for downstream analysis.

Installed as the ``mwir`` console script (see ``pyproject.toml``); also
runnable as ``python -m mwir_atmos.cli`` or importable for tests via
:func:`main`.

Exit codes
----------
``0``
    Success.
``1``
    Runtime error: a bad ``--config`` file, a scenario the physics layer
    rejected (e.g. an unreachable geometry), or a failure writing output.
``2``
    Usage error: a bad flag/value (argparse itself, or e.g. an unparsable
    ``--bands`` token, or ``--format npz`` without ``--output``).

Radiance handling
------------------
``ScenarioResult.radiance`` is ``None`` until the radiometry layer
(:mod:`mwir_atmos.radiometry.radiance`) exists -- :func:`run_scenario`
degrades silently rather than raising. Every writer below treats that as a
first-class, expected state: it is reported plainly ("radiance unavailable"),
never faked.

Cost estimation and the confirmation gate
------------------------------------------
Resolution is never capped or silently clamped -- a user may legitimately
need e.g. 0.001 cm^-1 over a narrow window to resolve a fine emission
feature. Instead, before every run this module estimates the spectral-point
x layer count, the memory the layer arrays will occupy, and a rough runtime,
and prints that estimate (see :func:`_estimate_run_cost`). If the estimate
crosses ``_CONFIRM_BYTES_THRESHOLD`` / ``_CONFIRM_SECONDS_THRESHOLD`` the run
refuses to proceed without ``--yes`` (or an interactive "proceed anyway?"
answer on a real terminal) -- see :func:`main`. A narrow window at fine
resolution stays fast and is never gated by grid size alone; see the module
docstring of :mod:`mwir_atmos.optics.lbl` for why the *wing* window
(``line_wing_cutoff_cm1`` / ``resolution_cm1``) can dominate cost instead,
which is why the estimate accounts for it too.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import yaml

from . import __version__
from .api import build_profile, build_scenario_path, run_scenario
from .constants import BANDS_CM1
from .spectral import (
    band_average,
    band_integrate,
    make_grid,
    resolution_cm1_for_wavelength_step,
    resolution_cm1_to_um,
)
from .types import (
    AerosolModel,
    RadianceSolver,
    InstrumentFunction,
    ModelAtmosphere,
    REQUIRED_SPECIES,
    Scenario,
    ScenarioResult,
    SkyPosition,
    SpectralSampling,
    SurfaceType,
    TurbulenceModel,
)

__all__ = [
    "main",
    "build_parser",
    "scenario_to_dict",
    "scenario_from_dict",
    "load_scenario_yaml",
    "save_scenario_yaml",
]

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2

#: RadianceResult's fixed field order (types.py), used everywhere a writer
#: needs to walk "every radiance component" without hardcoding the list twice.
RADIANCE_COMPONENTS: tuple[str, ...] = (
    "path_thermal",
    "surface_emitted",
    "surface_reflected_solar",
    "surface_reflected_lunar",
    "solar_single_scatter",
    "lunar_single_scatter",
    "direct_solar",
    "direct_lunar",
    "source_emitted",
    "solar_multiple_scatter",
    "lunar_multiple_scatter",
    "total",
)

#: Short header labels for the radiance table. The full names are up to 23
#: characters and run together at any readable column width.
_RADIANCE_ABBREV: dict[str, str] = {
    "path_thermal": "path_therm",
    "surface_emitted": "surf_emit",
    "surface_reflected_solar": "surf_r_sol",
    "surface_reflected_lunar": "surf_r_lun",
    "solar_single_scatter": "sol_scat",
    "lunar_single_scatter": "lun_scat",
    "direct_solar": "dir_sol",
    "direct_lunar": "dir_lun",
    "source_emitted": "src_emit",
    "solar_multiple_scatter": "sol_multi",
    "lunar_multiple_scatter": "lun_multi",
}

#: Scenario fields that are enums, mapped to their enum class -- drives both
#: argparse ``choices=`` and YAML/CLI (de)serialisation.
_ENUM_FIELD_TYPES: dict[str, type[Enum]] = {
    "atmosphere": ModelAtmosphere,
    "surface_type": SurfaceType,
    "aerosol": AerosolModel,
    "turbulence": TurbulenceModel,
    "radiance_solver": RadianceSolver,
    "spectral_sampling": SpectralSampling,
    "instrument_function": InstrumentFunction,
}

_FORMATS = ("table", "csv", "json", "npz")


# =========================================================================
# Up-front cost estimate + confirmation gate
#
# Resolution is a free dial (CONTRACT.md / job brief: never cap or clamp it
# -- narrow-window fine-resolution work is the primary use case and must
# stay fast). The trade-off is pushed onto the user as information instead:
# estimate points x layers, memory, and runtime before running, and refuse
# to proceed past a couple of rough thresholds without --yes.
# =========================================================================


@dataclasses.dataclass
class _CostEstimate:
    n_points: int
    n_layers: int
    wing_window_points: float
    est_bytes: int
    est_seconds: float


#: "Roughly 1 GB or a few minutes" (job brief) -- thresholds for the
#: confirmation gate, not hard limits; --yes always bypasses them.
_CONFIRM_BYTES_THRESHOLD = 1 * 1024**3
_CONFIRM_SECONDS_THRESHOLD = 180.0

#: Empirical calibration for the runtime estimate. Two structural drivers
#: dominate mwir_atmos.optics.lbl's cost, and this module deliberately does
#: not own or reimplement that engine -- it only estimates against it:
#:   - grid size: the per-layer optical-depth arrays are (n_layers, grid.n);
#:   - the per-line Voigt window: every surviving line is evaluated over
#:     2*line_wing_cutoff_cm1/resolution_cm1 + 1 grid points, a cost that is
#:     independent of how narrow the *requested* window is -- at fine
#:     resolution with the default 25 cm^-1 wing cutoff this window alone
#:     can be tens of thousands of points even for a 2 cm^-1 request, and it
#:     dominates runtime instead of the grid size. Whichever driver is
#:     larger sets the estimate.
#: Calibrated 2026-08-02 directly against this repository: an unmodified
#: 2501-point/60-layer 1800-4300 cm^-1 @ 1 cm^-1 run measured ~8 s here
#: (matches CONTRACT.md's documented 7-10 s), and this rate/overhead
#: reproduce the job brief's other reference point -- 1800-4300 cm^-1 @
#: 0.001 cm^-1 -- to ~2.2 h, matching what was measured for that briefing
#: almost exactly. This is an ORDER-OF-MAGNITUDE guide, not a scheduler:
#: spot checks in congested regions (e.g. a CO2 Q-branch) versus sparse ones
#: at identical grid/layer/wing settings varied the *measured* runtime by
#: up to ~3x here, because real cost also depends on how many lines
#: populate the requested window, which is not knowable without loading the
#: line data -- exactly the expensive step this estimate is trying to avoid
#: paying for up front.
#: These point-sampling constants predate the September 2026 engine rewrite
#: and err on the slow side (the default point run takes ~10 s).
_COST_FIXED_OVERHEAD_S = 4.0
_COST_RATE_ELEMENTS_PER_S = 19_000.0
#: Resolved sampling, per worker process: fine points x layers per second for
#: the whole pipeline (line-by-line, continuum, transmittance, single-scatter
#: radiance). Measured at 0.23e6 on an 8-core desktop
#: (docs/validation/scaling_benchmark.json); rounded down for slower cores.
_COST_RESOLVED_RATE_PER_WORKER = 150_000.0
#: Parallel speedup ~ workers ** this (measured 5.4x at 8 workers, 6.7x at 16
#: SMT threads on 8 cores); errs slow on machines that scale better.
_COST_SPEEDUP_EXPONENT = 0.7
#: float64 (n_layers[+1], grid.n) arrays materialised per run: tau_molecular,
#: one tau_by_species per REQUIRED_SPECIES, tau_aerosol_abs, tau_aerosol_sca,
#: tau_rayleigh, aerosol_asymmetry, and TransmittanceResult.cumulative.
_COST_N_LAYER_ARRAYS = 1 + len(REQUIRED_SPECIES) + 5
_COST_BYTES_PER_ELEMENT = 8


def _estimate_run_cost(scenario: Scenario, n_threads: Optional[int] = None) -> _CostEstimate:
    """Points x layers, estimated memory, and estimated runtime for ``scenario``.

    Counts grid points without allocating a spectral array, and builds the
    real path geometry so the estimate reflects the actual layer count, then applies
    the calibration above. Raises whatever :func:`~mwir_atmos.api.build_profile`
    / :func:`~mwir_atmos.api.build_scenario_path` raise for a geometrically
    invalid scenario (e.g. an unreachable source) -- the same error
    :func:`~mwir_atmos.api.run_scenario` would raise later, just surfaced
    before any spectral work happens.
    """
    from .validation import validate_scenario
    validate_scenario(scenario)
    n_points = int(np.floor((scenario.wavenumber_max_cm1 - scenario.wavenumber_min_cm1)
                           / scenario.resolution_cm1)) + 1
    profile = build_profile(scenario)
    path = build_scenario_path(scenario, profile)
    n_layers = path.n_layers
    if scenario.radiance_solver == RadianceSolver.DISORT:
        from .geometry.plane_parallel import geometry
        n_layers = geometry(scenario, profile)[0].n_layers
    if scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        from .geometry.spherical_shells import geometry
        n_layers = geometry(scenario, profile)[0].n_layers
    from .optics.lbl import evaluations_per_line
    step = scenario.resolution_cm1
    evaluated_points = n_points      # spectral points the line-by-line engine fills
    resident_points = n_points       # spectral points held in layer arrays at once
    solver_points = n_points         # spectral points DISORT / Monte Carlo solve
    if scenario.spectral_sampling == SpectralSampling.RESOLVED:
        from . import resolved
        from .types import SpectralGrid
        fine = scenario.fine_step_cm1
        if fine is None:
            fine = resolved.automatic_fine_step(path.layers, scenario.wavenumber_min_cm1)
        plan = resolved.plan_fine_grid(
            SpectralGrid(wavenumber_cm1=np.array([scenario.wavenumber_min_cm1]),
                         resolution_cm1=scenario.resolution_cm1),
            fine_step_cm1=fine, function=scenario.instrument_function,
            fwhm_cm1=scenario.instrument_fwhm_cm1)
        step = plan.step_cm1
        evaluated_points = (n_points - 1) * plan.ratio + 2 * plan.support + 1
        resident_points = min(evaluated_points, max(resolved._MIN_CHUNK_FINE_POINTS,
                                                    resolved._CHUNK_ELEMENTS // max(n_layers, 1)))
        solver_points = n_points * (len(resolved._G_EDGES) - 1)
    wing_pts = evaluations_per_line(step, scenario.line_wing_cutoff_cm1)

    elements = max(float(evaluated_points), wing_pts) * n_layers
    est_seconds = _COST_FIXED_OVERHEAD_S + elements / _COST_RATE_ELEMENTS_PER_S
    speedup = 1.0
    if scenario.spectral_sampling == SpectralSampling.RESOLVED:
        from .api import _PROCESS_MIN_ELEMENTS, _parallel_mode
        from .optics.lbl import resolve_thread_count
        workers = resolve_thread_count(n_threads)
        if workers > 1 and _parallel_mode() == "processes" and elements >= _PROCESS_MIN_ELEMENTS:
            chunk = max(resolved._MIN_CHUNK_FINE_POINTS,
                        resolved._MIN_CHUNK_OVER_SUPPORT * plan.support)
            workers = max(1, min(workers, int(evaluated_points // chunk)))
            speedup = workers ** _COST_SPEEDUP_EXPONENT
            resident_points = min(resident_points, max(chunk, evaluated_points // (3 * workers)))
            resident_points *= workers       # one chunk per worker in memory at once
        est_seconds = _COST_FIXED_OVERHEAD_S + elements / (_COST_RESOLVED_RATE_PER_WORKER * speedup)
    est_bytes = int(resident_points * n_layers * _COST_BYTES_PER_ELEMENT * _COST_N_LAYER_ARRAYS)
    if scenario.radiance_solver == RadianceSolver.DISORT:
        # Additional full-column arrays and sequential angular solves. This is
        # a rough estimate; actual Fourier work depends on illumination.
        est_bytes *= 2
        sources = 2 + int(scenario.include_solar) + int(scenario.include_lunar)
        est_seconds += solver_points * .005 * sources * (scenario.disort_streams / 16.)**4 / speedup
    if scenario.radiance_solver == RadianceSolver.SPHERICAL_MC:
        est_bytes += min(scenario.spherical_photons,4096) * n_layers * 200
        est_seconds += solver_points * (scenario.spherical_photons / 4096.) * 2. / speedup
    return _CostEstimate(
        n_points=n_points,
        n_layers=n_layers,
        wing_window_points=wing_pts,
        est_bytes=est_bytes,
        est_seconds=est_seconds,
    )


def _format_bytes(n_bytes: float) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.2f} h"


_EPILOG = """\
examples:
  # Quick look, human-readable summary table over the default bands.
  mwir --atmosphere tropical --aerosol navy_maritime --visibility-km 10 \\
       --sensor-zenith-deg 100 --format table

  # Load a worked example, override resolution, get the full spectrum as CSV.
  mwir --config scenarios/horizontal_haze.yaml --resolution-cm1 2.0 \\
       --format csv --output run1.csv

  # Just the CO2 and a custom 2600-2650 cm^-1 window, as JSON on stdout.
  mwir --bands CO2_4_3um 2600:2650 --format json

  # Resolve a scenario and capture exactly what was run, without redoing it.
  mwir --config scenarios/limb_view.yaml --sensor-alt-km 22 \\
       --save-config resolved.yaml --quiet --format npz -o run.npz

  # Narrow window at fine resolution: cheap, and unobstructed by the
  # confirmation gate -- stays under it because the window (not just the
  # grid) is narrow and the wing cutoff is trimmed to match. Lower
  # --line-strength-cutoff too, since the 1e-4 default can distort line
  # cores at this resolution (see --help under 'accuracy').
  mwir --wavenumber-min-cm1 2349 --wavenumber-max-cm1 2351 --resolution-cm1 0.001 \\
       --line-wing-cutoff-cm1 3 --line-strength-cutoff 1e-6 --format csv

  # A run large enough to trip the confirmation gate needs --yes to proceed
  # non-interactively (e.g. from a script/CI):
  mwir --resolution-cm1 0.01 --format npz -o full_band_fine.npz --yes

See --list-bands for the named bands available to --bands, and
scenarios/*.yaml for further worked examples.
"""


# =========================================================================
# Scenario <-> plain-dict / YAML (de)serialisation
#
# Shared by the CLI's --config/--save-config and reusable as-is by the
# Streamlit GUI (mwir_atmos.webgui.app imports these directly) so the two
# front ends never grow two different ideas of what a scenario file looks
# like.
# =========================================================================


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Plain-Python (YAML/JSON-safe) representation of a :class:`Scenario`.

    Enums become their ``.value`` string, ``datetime`` becomes an ISO 8601
    string; everything else (float, bool, None) passes through unchanged.
    Field order follows :class:`Scenario`'s declaration order.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(Scenario):
        val = getattr(scenario, f.name)
        if isinstance(val, Enum):
            val = val.value
        elif isinstance(val, datetime):
            val = val.isoformat()
        out[f.name] = val
    return out


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    """Inverse of :func:`scenario_to_dict`.

    Unset fields fall back to :class:`Scenario`'s own defaults (a config file
    only needs to mention the fields it wants to override). Raises
    ``ValueError`` on an unknown field name -- deliberately strict, since a
    silently-ignored typo in a scenario file (e.g. ``visiblity_km``) would
    otherwise run the *wrong* scenario without any indication.
    """
    field_names = {f.name for f in dataclasses.fields(Scenario)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ValueError(f"unknown scenario field(s) in config: {unknown}")

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(Scenario):
        if f.name not in data:
            continue
        val = data[f.name]
        if val is None:
            kwargs[f.name] = None
        elif f.name in _ENUM_FIELD_TYPES:
            kwargs[f.name] = _ENUM_FIELD_TYPES[f.name](val)
        elif f.name == "local_datetime":
            # PyYAML auto-parses ISO-8601-looking scalars into datetime
            # already; accept either that or a plain string (e.g. from JSON).
            kwargs[f.name] = val if isinstance(val, datetime) else _parse_datetime(str(val))
        else:
            kwargs[f.name] = val
    from .validation import validate_scenario
    scenario = dataclasses.replace(Scenario(), **kwargs)
    validate_scenario(scenario)
    return scenario


def load_scenario_yaml(path: Union[str, Path]) -> Scenario:
    """Load a :class:`Scenario` from a YAML file written by
    :func:`save_scenario_yaml` (or hand-written, see ``scenarios/*.yaml``)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"config file {p} did not parse to a YAML mapping (got {type(data).__name__})"
        )
    return scenario_from_dict(data)


def save_scenario_yaml(scenario: Scenario, path: Union[str, Path]) -> None:
    """Write a :class:`Scenario` to YAML, in field-declaration order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = scenario_to_dict(scenario)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid --local-datetime {value!r}: expected ISO 8601, "
            "e.g. 2026-08-02T14:30:00"
        ) from exc


# =========================================================================
# argparse
# =========================================================================


def _flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def _add_float(group, field: str, help_text: str, *, metavar: Optional[str] = None) -> None:
    group.add_argument(_flag(field), dest=field, type=float, default=None, metavar=metavar, help=help_text)


def _add_enum(group, field: str, enum_cls: type[Enum], help_text: str) -> None:
    group.add_argument(
        _flag(field),
        dest=field,
        type=str,
        default=None,
        choices=[e.value for e in enum_cls],
        help=help_text,
    )


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mwir`` argument parser.

    Every :class:`~mwir_atmos.types.Scenario` field gets a flag, grouped as
    atmosphere / sensor / source / illumination / spectral / accuracy /
    turbulence / output, all defaulting to ``None`` regardless of the
    field's real default -- ``None`` means "not given on the command line",
    which is what lets ``--config`` values and CLI overrides compose
    correctly (see :func:`_resolve_scenario`).
    """
    parser = argparse.ArgumentParser(
        prog="mwir",
        description=(
            "MODTRAN-at-home: mid-wave infrared atmospheric transmittance and "
            "radiance calculator. HITRAN line-by-line molecular absorption, "
            "AFGL model atmospheres, MT_CKD H2O continuum, LOWTRAN-style "
            "aerosols and Rayleigh scattering."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mwir_atmos {__version__}")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress/diagnostic messages on stderr."
    )
    verbosity.add_argument(
        "-v", "--verbose", action="store_true",
        help="Extra diagnostic messages on stderr (timing, layer counts, tracebacks on error).",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation gate for runs estimated above ~1 GB of layer arrays or "
             "~3 minutes (see the 'estimated cost' message printed before every run). Required "
             "in non-interactive contexts (scripts/CI) once a run crosses that threshold -- "
             "without a terminal attached, mwir refuses rather than hanging on a prompt.",
    )

    cfg = parser.add_argument_group("configuration")
    cfg.add_argument(
        "--config", metavar="PATH", default=None,
        help="Load a Scenario from a YAML file (see scenarios/*.yaml). Flags below override its values.",
    )
    cfg.add_argument(
        "--save-config", metavar="PATH", default=None,
        help="Write the fully-resolved scenario (config file + flag overrides) back out as YAML.",
    )

    atmo = parser.add_argument_group("atmosphere")
    _add_enum(atmo, "atmosphere", ModelAtmosphere, "Model atmosphere (default: midlat_summer).")
    _add_enum(atmo, "surface_type", SurfaceType, "Lower boundary surface type (default: land_vegetated).")
    _add_enum(atmo, "aerosol", AerosolModel, "Boundary-layer aerosol model (default: rural).")
    _add_float(atmo, "visibility_km", "Surface visibility, km; drives the Koschmieder aerosol extinction (default: 23.0).")
    _add_float(atmo, "ground_temp_k", "Override surface air temperature, K (default: model-atmosphere value).")
    _add_float(atmo, "ground_rh_pct", "Override surface relative humidity, %% (default: model-atmosphere value).")
    _add_float(atmo, "surface_skin_temp_k", "Ground/sea skin temperature, K (default: ground_temp_k).")
    _add_float(atmo, "surface_emissivity", "Surface emissivity override, 0-1 (default: none -- radiometry layer decides from surface_type).")
    _add_float(atmo, "co2_ppmv", "Override CO2 mixing ratio, ppmv (default: constants.BACKGROUND_VMR_PPMV, currently 423.0).")
    _add_float(atmo, "wind_speed_m_s", "Surface wind speed, m/s -- navy_maritime aerosol only (default: 5.0).")
    _add_float(atmo, "surface_altitude_km", "Re-anchor the model atmosphere's surface to this altitude, km (default: 0.0).")

    sensor = parser.add_argument_group("sensor")
    _add_float(sensor, "sensor_alt_km", "Sensor altitude, km (default: 0.0).")
    _add_float(sensor, "sensor_zenith_deg", "Sensor look angle from local vertical, deg: 0=up, 90=horizontal, 180=down (default: 0.0).")
    _add_float(sensor, "sensor_azimuth_deg", "Sensor look azimuth, deg clockwise from true north (default: 0.0).")
    _add_float(sensor, "latitude_deg", "Sensor latitude, deg (default: 35.0).")
    _add_float(sensor, "longitude_deg", "Sensor longitude, deg (default: 0.0).")
    _add_float(
        sensor, "sensor_aperture_diameter_m",
        "Clear aperture diameter of the receiving optic, m (default: 0.10). "
        "Sets the diffraction limit and, with the turbulence models, the Strehl "
        "ratio and point-source blur. Turbulence only degrades a point source "
        "once this exceeds the Fried parameter r0.",
    )

    source = parser.add_argument_group("source")
    _add_float(source, "source_alt_km", "Source (target) altitude, km (default: none -- path runs to the ground or space).")
    _add_float(source, "source_range_km", "Sensor-to-source slant range, km (default: none -- path runs to the ground or space).")
    _add_float(source, "source_temp_k", "Discrete source/target temperature, K (default: none).")
    _add_float(source, "source_emissivity", "Discrete source/target emissivity, 0-1 (default: 1.0).")
    _add_float(source, "source_area_m2", "Discrete source/target area, m^2 (default: none).")

    illum = parser.add_argument_group("illumination")
    illum.add_argument(
        "--local-datetime", dest="local_datetime", type=str, default=None, metavar="ISO8601",
        help="Local clock time, e.g. 2026-08-02T14:30:00 (default: none). "
             "Timezone is approximated from longitude unless --utc-offset-hours is given.",
    )
    _add_float(illum, "utc_offset_hours", "UTC offset, hours (default: approximated from longitude).")
    _add_float(
        illum, "sunrise_local_hours",
        "Observed local sunrise at the sensor, in clock hours (6.75 = 06:45). "
        "Supply WITH --sunset-local-hours to calibrate the sun exactly: their "
        "midpoint is local solar noon and their separation gives the day length, "
        "which together override the timezone. Without them the timezone is "
        "guessed from longitude and can be tens of degrees wrong.",
    )
    _add_float(
        illum, "sunset_local_hours",
        "Observed local sunset at the sensor, in clock hours (21.8 = 21:48). "
        "See --sunrise-local-hours; both are required for calibration.",
    )
    _add_float(illum, "moon_phase", "Moon phase, 0=new, 0.5=full, 1=new again (default: 0.5).")
    illum.add_argument(
        "--solar", dest="include_solar", action=argparse.BooleanOptionalAction, default=None,
        help="Include/exclude solar illumination (default: include).",
    )
    illum.add_argument(
        "--lunar", dest="include_lunar", action=argparse.BooleanOptionalAction, default=None,
        help="Include/exclude lunar illumination (default: include).",
    )

    spectral = parser.add_argument_group("spectral")
    _add_float(spectral, "wavenumber_min_cm1", "Grid lower bound, cm^-1 (default: 1800.0 = 5.56 um, covering the full 3-5 um MWIR band with margin). Narrower is fine and is the fast path for fine-resolution work -- see 'accuracy' below.")
    _add_float(spectral, "wavenumber_max_cm1", "Grid upper bound, cm^-1 (default: 4300.0 = 2.33 um).")
    _add_float(
        spectral, "resolution_cm1",
        "Grid spacing, cm^-1 (default: 1.0). Never capped -- set it as fine as your data needs. "
        "Cost scales with points x layers (points = bandwidth/resolution): a narrow window at fine "
        "resolution is cheap (a 2 cm^-1 window at 0.001 cm^-1 is a handful of seconds), the full "
        "1800-4300 cm^-1 default band at 0.001 cm^-1 is not (~2 h, ~13 GB of layer arrays). mwir "
        "prints a cost estimate before every run and gates large ones behind --yes; narrow the "
        "window rather than reducing resolution if you need both speed and fine detail.",
    )
    spectral.add_argument(
        "--resolution-um", dest="resolution_um", type=float, default=None, metavar="UM",
        help="Set the grid spacing by WAVELENGTH step instead of wavenumber, in um. "
             "Converted to the wavenumber step that guarantees at least this "
             "resolution across the whole band (sized at the longest wavelength, "
             "since a fixed cm^-1 step widens as lambda^2). Overrides "
             "--resolution-cm1. For scale on 3-5 um: 0.15 um = 60 cm^-1 (only 23 "
             "points across the band -- very coarse), 0.01 um = 4 cm^-1, "
             "0.001 um = 0.4 cm^-1.",
    )

    _add_enum(
        spectral, "spectral_sampling", SpectralSampling,
        "point: physics evaluated only at the output grid points (fast; a grid coarser than "
        "the ~0.07 cm^-1 absorption lines samples line cores at random). resolved: evaluated "
        "on a line-resolving internal grid and convolved with --instrument-function onto the "
        "output grid, i.e. what an instrument of that resolution measures (default: resolved).",
    )
    _add_float(
        spectral, "fine_step_cm1",
        "Internal monochromatic step for --spectral-sampling resolved, cm^-1 (default: chosen "
        "from the line widths along the path). Set explicitly to check convergence.",
    )
    _add_enum(spectral, "instrument_function", InstrumentFunction,
              "Instrument line shape for resolved sampling: box (bin average), triangle, "
              "or gaussian (default: box).")
    _add_float(
        spectral, "instrument_fwhm_cm1",
        "Instrument FWHM for resolved sampling, cm^-1 (default: --resolution-cm1).",
    )

    accuracy = parser.add_argument_group("accuracy")
    _add_enum(accuracy, "radiance_solver", RadianceSolver,
              "single_scatter, disort (plane-parallel), or spherical_mc (spherical multiple scattering).")
    accuracy.add_argument("--spherical-photons", type=int, default=None, help="Independent photon histories per spectral point (default 4096).")
    accuracy.add_argument("--spherical-seed", type=int, default=None, help="Reproducible Monte Carlo seed (default 0).")
    accuracy.add_argument("--spherical-max-events", type=int, default=None, help="Safety limit: reaching this raises an error, never silently truncates.")
    accuracy.add_argument("--disort-streams", type=int, choices=[8,16,32,64], default=None,
                         help="Discrete-ordinate streams; increase and compare for angular convergence (default: 32).")
    scat = parser.add_argument_group("refraction and polarization (spherical_mc only)")
    scat.add_argument(
        "--refraction", dest="refraction", action=argparse.BooleanOptionalAction, default=None,
        help="Trace refracted (curved) rays through the spherical shells (default: off).",
    )
    _add_float(scat, "refraction_reference_um",
               "Wavelength at which air refractivity is evaluated for ray bending, um (default: 4.0).")
    scat.add_argument("--refractive-index-file", dest="refractive_index_file", default=None, metavar="PATH",
                      help="Refractivity table overriding the built-in Mathar data (requires --refraction).")
    scat.add_argument(
        "--polarization", dest="polarization", action=argparse.BooleanOptionalAction, default=None,
        help="Vector (Stokes) Monte Carlo radiance (default: off).",
    )
    scat.add_argument("--aerosol-polarization", dest="aerosol_polarization", default=None,
                      choices=["hg_depolarizing", "mie_sphere"],
                      help="Aerosol scattering matrix model (default: hg_depolarizing).")
    _add_float(scat, "mie_radius_um", "Aerosol sphere radius for mie_sphere, um (default: 0.1).")
    _add_float(scat, "mie_index_real", "Aerosol refractive index, real part (default: 1.5).")
    _add_float(scat, "mie_index_imag", "Aerosol refractive index, imaginary part (default: 0.0).")
    scat.add_argument("--surface-polarization", dest="surface_polarization", default=None,
                      choices=["lambertian", "fresnel"],
                      help="Surface reflection model (default: lambertian).")
    _add_float(scat, "surface_index_real", "Fresnel surface refractive index, real part (default: 1.5).")
    _add_float(scat, "surface_index_imag", "Fresnel surface refractive index, imaginary part (default: 0.0).")
    scat.add_argument("--sensor-response-file", dest="sensor_response_file", default=None, metavar="PATH",
                      help="Sensor polarization analyzer response (requires --polarization; "
                           "not supported with resolved sampling).")

    _add_float(
        accuracy, "line_strength_cutoff",
        "Drop lines weaker than this fraction of the strongest line in each per-layer, "
        "per-species evaluation (default: 0.0, off). Legacy rule: at its former default "
        "(1e-4) it discarded every water line in the 3.5-4.1 um window, overstating "
        "transmittance there by up to ~24 percentage points. Prefer --line-optical-depth-cutoff.",
    )
    _add_float(
        accuracy, "line_optical_depth_cutoff",
        "Drop a line from a layer when its peak optical depth in that layer is below this "
        "(default: 3e-6). Tied to the actual gas amount, so thick humid layers keep the weak "
        "water lines that matter in the windows. 0.0 keeps every line.",
    )
    _add_float(
        accuracy, "line_wing_cutoff_cm1",
        "Local-line half-width for gases other than water, cm^-1 (default: 25.0). "
        "H2O with MT_CKD always uses 25 cm^-1 with pedestal subtraction. "
        "Evaluation windows are clipped to the requested grid. Smaller cutoffs can omit "
        "absorption; check convergence before relying on them.",
    )

    turb = parser.add_argument_group("turbulence (stretch goal)")
    _add_enum(turb, "turbulence", TurbulenceModel, "Cn^2 vertical profile model (default: none).")
    _add_float(turb, "turbulence_cn2_ground", "Ground-level Cn^2, m^-2/3; model dependent (default: none).")

    out = parser.add_argument_group("output")
    out.add_argument("--format", choices=_FORMATS, default="table", help="Output format (default: table).")
    out.add_argument(
        "-o", "--output", metavar="PATH", default=None,
        help="Write to this file instead of stdout. Required for --format npz "
             "(numpy appends '.npz' if the name doesn't already end with it).",
    )
    out.add_argument(
        "--bands", nargs="*", default=None, metavar="BAND",
        help="Bands to summarize in --format table/json: names from constants.BANDS_CM1 "
             "(see --list-bands) and/or custom 'MIN:MAX' cm^-1 ranges (e.g. 2600:2650). "
             "Default: every named band. Pass --bands with no values for none. "
             "Ignored by --format csv/npz's per-wavenumber arrays, which are always the full grid.",
    )
    out.add_argument("--list-bands", action="store_true", help="Print the named bands available to --bands and exit.")
    out.add_argument(
        "--no-radiance", action="store_true",
        help="Skip the radiometry layer even if available (transmittance only; faster).",
    )
    out.add_argument(
        "--no-turbulence", action="store_true", help="Skip the turbulence layer even if available."
    )

    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--threads", type=_positive_int, default=None, metavar="N",
        help="Line-by-line worker threads (default: MWIR_THREADS or every CPU). "
             "Results do not depend on this.",
    )
    perf.add_argument(
        "--backend", choices=["cpu", "cuda"], default=None,
        help="Line-by-line engine: cpu, or cuda (NVIDIA GPU via CuPy: pip install "
             "cupy-cuda12x). Default: MWIR_BACKEND or cpu. Results agree to ~1e-6.",
    )

    return parser


# =========================================================================
# Scenario resolution (config file + flag overrides)
# =========================================================================


def _resolve_scenario(args: argparse.Namespace) -> Scenario:
    """``--config`` (or fresh defaults) with CLI flag overrides applied on top."""
    base = load_scenario_yaml(args.config) if args.config else Scenario()

    overrides: dict[str, Any] = {}
    for f in dataclasses.fields(Scenario):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if val is None:
            continue
        if f.name in _ENUM_FIELD_TYPES:
            val = _ENUM_FIELD_TYPES[f.name](val)
        elif f.name == "local_datetime":
            val = _parse_datetime(val)
        overrides[f.name] = val

    resolved = dataclasses.replace(base, **overrides) if overrides else base

    # --resolution-um is a convenience over resolution_cm1 rather than a
    # Scenario field of its own, so it is applied last and wins. It is sized at
    # the band's longest wavelength so the stated resolution holds everywhere.
    resolution_um = getattr(args, "resolution_um", None)
    if resolution_um is not None:
        if resolution_um <= 0.0:
            raise ValueError("--resolution-um must be positive")
        resolved = dataclasses.replace(
            resolved,
            resolution_cm1=resolution_cm1_for_wavelength_step(
                resolution_um, resolved.wavenumber_min_cm1, resolved.wavenumber_max_cm1
            ),
        )
    from .validation import validate_scenario
    validate_scenario(resolved)
    return resolved


# =========================================================================
# Band tokens ("--bands MWIR_window_1 2600:2650 ...")
# =========================================================================

BandSpec = Union[str, tuple[float, float]]


def _parse_band_token(token: str) -> tuple[str, BandSpec]:
    """One ``--bands`` token -> ``(display_label, band_spec)``.

    ``band_spec`` is whatever :func:`mwir_atmos.spectral.band_average` /
    :func:`~mwir_atmos.spectral.band_integrate` accept directly: a
    ``constants.BANDS_CM1`` key, or a ``(lo, hi)`` cm^-1 tuple parsed from a
    ``'MIN:MAX'`` token.
    """
    if token in BANDS_CM1:
        return token, token
    if ":" in token:
        lo_s, hi_s = token.split(":", 1)
        try:
            lo, hi = float(lo_s), float(hi_s)
        except ValueError as exc:
            raise ValueError(f"invalid band range {token!r}: expected 'MIN:MAX' in cm^-1") from exc
        if hi <= lo:
            raise ValueError(f"invalid band range {token!r}: MAX must exceed MIN")
        return f"{lo:g}-{hi:g}cm-1", (lo, hi)
    raise ValueError(
        f"unknown band {token!r}: not a name in constants.BANDS_CM1 "
        f"({', '.join(sorted(BANDS_CM1))}) and not a 'MIN:MAX' cm^-1 range"
    )


def _resolve_band_specs(tokens: Optional[Sequence[str]]) -> list[tuple[str, BandSpec]]:
    """``None`` (flag omitted) -> every named band. ``[]`` (flag given empty) -> none."""
    if tokens is None:
        return [(name, name) for name in BANDS_CM1]
    return [_parse_band_token(t) for t in tokens]


def _print_band_catalog(fh=None) -> None:
    # fh's default is resolved here, not at def-time -- `fh=sys.stdout` as a
    # parameter default binds whatever sys.stdout *was* when this module was
    # first imported, which silently stops being the live stdout under
    # anything that reassigns sys.stdout per-call (e.g. pytest's capsys).
    if fh is None:
        fh = sys.stdout
    print(f"{'name':<20}{'cm^-1 range':<20}{'um range':<16}", file=fh)
    for name, (lo, hi) in BANDS_CM1.items():
        um_lo, um_hi = 1.0e4 / hi, 1.0e4 / lo
        print(f"{name:<20}{f'{lo:.2f}-{hi:.2f}':<20}{f'{um_lo:.3f}-{um_hi:.3f}':<16}", file=fh)


# =========================================================================
# Band summary rows (shared by table/json/npz writers)
# =========================================================================


@dataclasses.dataclass
class _BandRow:
    label: str
    lo_cm1: float
    hi_cm1: float
    lo_um: float
    hi_um: float
    transmittance: dict[str, float]
    species_transmittance: dict[str, float]
    radiance: Optional[dict[str, float]]


def _band_rows(
    result: ScenarioResult, band_specs: Sequence[tuple[str, BandSpec]], log: "_Logger"
) -> list[_BandRow]:
    grid = result.grid
    t = result.transmittance
    rows: list[_BandRow] = []
    for label, spec in band_specs:
        try:
            lo, hi = BANDS_CM1[spec] if isinstance(spec, str) else spec
            trans = {
                "total": float(band_average(grid, t.total, spec)),
                "molecular": float(band_average(grid, t.molecular, spec)),
                "aerosol": float(band_average(grid, t.aerosol, spec)),
                "rayleigh": float(band_average(grid, t.rayleigh, spec)),
            }
            species_trans = {
                sp.value: float(band_average(grid, arr, spec)) for sp, arr in t.by_species.items()
            }
            radiance: Optional[dict[str, float]] = None
            if result.radiance is not None:
                radiance = {
                    name: float(band_integrate(grid, getattr(result.radiance, name), spec))
                    for name in RADIANCE_COMPONENTS
                }
        except ValueError as exc:
            log.warn(f"skipping band {label!r}: {exc}")
            continue
        rows.append(_BandRow(label, lo, hi, 1.0e4 / hi, 1.0e4 / lo, trans, species_trans, radiance))
    return rows


# =========================================================================
# Output writers
# =========================================================================


def _sky_position_to_dict(pos: Optional[SkyPosition]) -> Optional[dict[str, Any]]:
    if pos is None:
        return None
    return {
        "zenith_deg": pos.zenith_deg,
        "azimuth_deg": pos.azimuth_deg,
        "scattering_angle_deg": pos.scattering_angle_deg,
        "is_above_horizon": pos.is_above_horizon,
        "distance_factor": pos.distance_factor,
        "illuminated_fraction": pos.illuminated_fraction,
    }


def _write_table(result: ScenarioResult, rows: list[_BandRow], fh) -> None:
    sc = result.scenario
    grid = result.grid
    path = result.path

    print("mwir_atmos scenario summary", file=fh)
    print(f"  atmosphere        : {sc.atmosphere.value}", file=fh)
    print(f"  surface type      : {sc.surface_type.value}", file=fh)
    print(f"  aerosol           : {sc.aerosol.value} (visibility {sc.visibility_km:g} km)", file=fh)
    print(
        f"  sensor            : alt {sc.sensor_alt_km:g} km, "
        f"zenith {sc.sensor_zenith_deg:g} deg, azimuth {sc.sensor_azimuth_deg:g} deg",
        file=fh,
    )
    src_bits = []
    if sc.source_alt_km is not None:
        src_bits.append(f"alt {sc.source_alt_km:g} km")
    if sc.source_range_km is not None:
        src_bits.append(f"range {sc.source_range_km:g} km")
    if sc.source_temp_k is not None:
        src_bits.append(f"T {sc.source_temp_k:g} K")
    print(
        f"  source            : {', '.join(src_bits) if src_bits else '(none -- path runs to ground/space)'}",
        file=fh,
    )
    print(
        f"  spectral grid     : {grid.wavenumber_cm1[0]:.2f}-{grid.wavenumber_cm1[-1]:.2f} cm^-1 "
        f"({grid.wavelength_um[-1]:.3f}-{grid.wavelength_um[0]:.3f} um), "
        f"resolution {grid.resolution_cm1:g} cm^-1, {grid.n} points",
        file=fh,
    )
    tangent_bit = f", tangent_alt={path.tangent_alt_km:.3f} km" if path.tangent_alt_km is not None else ""
    print(
        f"  path              : {path.n_layers} layers, {path.total_length_km:.3f} km, "
        f"hits_ground={path.hits_ground}, reaches_space={path.reaches_space}{tangent_bit}",
        file=fh,
    )
    if result.sun is not None:
        print(
            f"  sun               : zenith {result.sun.zenith_deg:.2f} deg, "
            f"azimuth {result.sun.azimuth_deg:.2f} deg, above_horizon={result.sun.is_above_horizon}",
            file=fh,
        )
    if result.moon is not None:
        print(
            f"  moon              : zenith {result.moon.zenith_deg:.2f} deg, "
            f"azimuth {result.moon.azimuth_deg:.2f} deg, above_horizon={result.moon.is_above_horizon}, "
            f"illuminated={result.moon.illuminated_fraction:.2f}",
            file=fh,
        )
    if result.radiance is None:
        print(
            "  radiance          : unavailable (radiometry module not present in this build -- "
            "transmittance only)",
            file=fh,
        )
    print(file=fh)

    if not rows:
        print("(no bands selected -- pass --bands to see a band summary)", file=fh)
        return

    header = f"{'band':<20}{'range (um)':<16}{'trans_total':>12}{'trans_mol':>12}{'trans_aer':>12}{'trans_ray':>12}"
    print(header, file=fh)
    print("-" * len(header), file=fh)
    for r in rows:
        rng = f"{r.lo_um:.3f}-{r.hi_um:.3f}"
        print(
            f"{r.label:<20}{rng:<16}"
            f"{r.transmittance['total']:>12.4e}{r.transmittance['molecular']:>12.4e}"
            f"{r.transmittance['aerosol']:>12.4e}{r.transmittance['rayleigh']:>12.4e}",
            file=fh,
        )
    print(file=fh)

    species_names = sorted({sp for r in rows for sp in r.species_transmittance})
    if species_names:
        print("band-averaged transmittance by species", file=fh)
        header2 = f"{'band':<20}" + "".join(f"{sp:>12}" for sp in species_names)
        print(header2, file=fh)
        print("-" * len(header2), file=fh)
        for r in rows:
            line = f"{r.label:<20}"
            for sp in species_names:
                v = r.species_transmittance.get(sp)
                line += f"{v:>12.4e}" if v is not None else f"{'--':>12}"
            print(line, file=fh)
        print(file=fh)

    if any(r.radiance is not None for r in rows):
        print("band-integrated radiance (W cm^-2 sr^-1)", file=fh)
        # Full component names run to 23 characters and collide at any sane
        # column width, so the header is abbreviated and expanded in a legend.
        abbrev = {c: _RADIANCE_ABBREV.get(c, c)[:11] for c in RADIANCE_COMPONENTS}
        header3 = f"{'band':<20}" + "".join(f"{abbrev[c]:>12}" for c in RADIANCE_COMPONENTS)
        print(header3, file=fh)
        print("-" * len(header3), file=fh)
        for r in rows:
            if r.radiance is None:
                continue
            line = f"{r.label:<20}" + "".join(f"{r.radiance[c]:>12.4e}" for c in RADIANCE_COMPONENTS)
            print(line, file=fh)
        legend = ", ".join(f"{abbrev[c]}={c}" for c in RADIANCE_COMPONENTS if abbrev[c] != c)
        if legend:
            print(f"  ({legend})", file=fh)
        print(file=fh)
    elif result.radiance is None:
        print("(radiance unavailable -- transmittance-only summary above)", file=fh)

    _write_turbulence_block(result, fh)


def _write_turbulence_block(result, fh) -> None:
    """Optical-turbulence summary, when a Cn^2 model was requested.

    These are diagnostics, not corrections: turbulence redistributes a point
    source in angle and time, it does not absorb, so no radiance value above is
    modified by them.
    """
    t = getattr(result, "turbulence", None)
    if t is None:
        return

    print("optical turbulence (diagnostic -- does not modify the radiance above)", file=fh)
    print(f"  Cn^2 model            : {result.scenario.turbulence.value}", file=fh)
    print(f"  reference wavelength  : {t.reference_wavelength_um:.3f} um", file=fh)
    print(f"  Fried parameter r0    : {t.fried_parameter_r0_cm:.2f} cm", file=fh)
    print(f"  isoplanatic angle     : {t.isoplanatic_angle_urad:.2f} urad", file=fh)
    print(f"  Rytov variance        : {t.rytov_variance:.4f}", file=fh)
    print(f"  scintillation index   : {t.scintillation_index:.4f}", file=fh)
    if t.greenwood_frequency_hz is not None:
        print(f"  Greenwood frequency   : {t.greenwood_frequency_hz:.2f} Hz", file=fh)
    if t.aperture_diameter_m is not None:
        regime = "SEEING-limited" if t.is_seeing_limited else "diffraction-limited"
        print(f"  aperture              : {t.aperture_diameter_m * 100:.1f} cm  ({regime})", file=fh)
        print(f"  diffraction limit     : {t.diffraction_limit_urad:.2f} urad (lambda/D)", file=fh)
        print(f"  seeing blur           : {t.seeing_limit_urad:.2f} urad (lambda/r0)", file=fh)
        print(f"  effective spot size   : {t.effective_blur_urad:.2f} urad", file=fh)
        print(f"  Strehl / peak ratio   : {t.strehl_ratio:.4f}", file=fh)
    print(file=fh)


def _write_csv(result: ScenarioResult, fh) -> None:
    """Full per-wavenumber spectral arrays, one row per grid point.

    Always the complete grid regardless of --bands: CSV is the scriptable
    path and must carry the real arrays, not a summary.
    """
    grid = result.grid
    t = result.transmittance
    species = sorted(t.by_species, key=lambda s: s.value)
    has_radiance = result.radiance is not None

    fieldnames = ["wavenumber_cm1", "wavelength_um", "trans_total", "trans_molecular", "trans_aerosol", "trans_rayleigh"]
    fieldnames += [f"trans_{sp.value}" for sp in species]
    if has_radiance:
        fieldnames += [f"radiance_{c}" for c in RADIANCE_COMPONENTS]
        fieldnames += [f"radiance_stderr_{c}" for c in result.radiance.sampling_standard_error]

    columns: list[np.ndarray] = [grid.wavenumber_cm1, grid.wavelength_um, t.total, t.molecular, t.aerosol, t.rayleigh]
    columns += [t.by_species[sp] for sp in species]
    if has_radiance:
        columns += [getattr(result.radiance, c) for c in RADIANCE_COMPONENTS]
        columns += list(result.radiance.sampling_standard_error.values())

    writer = csv.writer(fh)
    writer.writerow(fieldnames)
    for i in range(grid.n):
        writer.writerow([col[i] for col in columns])


def _write_json(result: ScenarioResult, rows: list[_BandRow], fh) -> None:
    grid = result.grid
    t = result.transmittance
    payload: dict[str, Any] = {
        "scenario": scenario_to_dict(result.scenario),
        "grid": {
            "wavenumber_cm1": grid.wavenumber_cm1.tolist(),
            "wavelength_um": grid.wavelength_um.tolist(),
            "resolution_cm1": grid.resolution_cm1,
        },
        "path": {
            "n_layers": result.path.n_layers,
            "total_length_km": result.path.total_length_km,
            "hits_ground": result.path.hits_ground,
            "reaches_space": result.path.reaches_space,
            "tangent_alt_km": result.path.tangent_alt_km,
            "sensor_alt_km": result.path.sensor_alt_km,
            "source_alt_km": result.path.source_alt_km,
        },
        "transmittance": {
            "total": t.total.tolist(),
            "molecular": t.molecular.tolist(),
            "aerosol": t.aerosol.tolist(),
            "rayleigh": t.rayleigh.tolist(),
            "by_species": {sp.value: arr.tolist() for sp, arr in t.by_species.items()},
        },
        "radiance": None,
        "sun": _sky_position_to_dict(result.sun),
        "moon": _sky_position_to_dict(result.moon),
        "bands": {
            r.label: {
                "range_cm1": [r.lo_cm1, r.hi_cm1],
                "range_um": [r.lo_um, r.hi_um],
                "transmittance": r.transmittance,
                "transmittance_by_species": r.species_transmittance,
                "radiance": r.radiance,
            }
            for r in rows
        },
    }
    if result.radiance is not None:
        payload["radiance"] = {c: getattr(result.radiance, c).tolist() for c in RADIANCE_COMPONENTS}
        if result.radiance.sampling_standard_error:
            payload["radiance_sampling_standard_error"] = {k:v.tolist() for k,v in result.radiance.sampling_standard_error.items()}
    json.dump(payload, fh, indent=2)
    fh.write("\n")


def _write_npz(result: ScenarioResult, rows: list[_BandRow], path: str) -> None:
    grid = result.grid
    t = result.transmittance
    arrays: dict[str, np.ndarray] = {
        "wavenumber_cm1": grid.wavenumber_cm1,
        "wavelength_um": grid.wavelength_um,
        "trans_total": t.total,
        "trans_molecular": t.molecular,
        "trans_aerosol": t.aerosol,
        "trans_rayleigh": t.rayleigh,
    }
    for sp, arr in t.by_species.items():
        arrays[f"trans_{sp.value}"] = arr
    if result.radiance is not None:
        for c in RADIANCE_COMPONENTS:
            arrays[f"radiance_{c}"] = getattr(result.radiance, c)

        for name, error in result.radiance.sampling_standard_error.items():
            arrays[f"radiance_stderr_{name}"] = error

    if rows:
        arrays["band_label"] = np.array([r.label for r in rows])
        arrays["band_lo_cm1"] = np.array([r.lo_cm1 for r in rows], dtype=np.float64)
        arrays["band_hi_cm1"] = np.array([r.hi_cm1 for r in rows], dtype=np.float64)
        for key in ("total", "molecular", "aerosol", "rayleigh"):
            arrays[f"band_trans_{key}"] = np.array([r.transmittance[key] for r in rows], dtype=np.float64)
        if any(r.radiance is not None for r in rows):
            for c in RADIANCE_COMPONENTS:
                arrays[f"band_radiance_{c}"] = np.array(
                    [(r.radiance[c] if r.radiance is not None else np.nan) for r in rows], dtype=np.float64
                )

    np.savez_compressed(path, **arrays)


def _open_text_output(path: Optional[str]):
    """Context manager yielding ``sys.stdout`` or an open file handle."""
    if path is None:
        return _NullContext(sys.stdout)
    return open(path, "w", encoding="utf-8", newline="")


class _NullContext:
    """Wrap an already-open stream so ``with`` doesn't close stdout."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc_info) -> None:
        return None


def _write_output(
    result: ScenarioResult, args: argparse.Namespace, band_specs: Sequence[tuple[str, BandSpec]], log: "_Logger"
) -> None:
    if args.format == "csv":
        with _open_text_output(args.output) as fh:
            _write_csv(result, fh)
        return

    rows = _band_rows(result, band_specs, log)

    if args.format == "npz":
        _write_npz(result, rows, args.output)
        return

    with _open_text_output(args.output) as fh:
        if args.format == "table":
            _write_table(result, rows, fh)
        elif args.format == "json":
            _write_json(result, rows, fh)
        else:  # pragma: no cover - argparse `choices=` already prevents this
            raise AssertionError(f"unhandled --format {args.format!r}")


# =========================================================================
# Logging (stderr only -- keeps stdout clean for csv/json/table piping)
# =========================================================================


class _Logger:
    def __init__(self, *, quiet: bool, verbose: bool) -> None:
        self.quiet = quiet
        self.verbose = verbose

    def info(self, msg: str) -> None:
        if not self.quiet:
            print(f"mwir: {msg}", file=sys.stderr)

    def warn(self, msg: str) -> None:
        if not self.quiet:
            print(f"mwir: warning: {msg}", file=sys.stderr)

    def debug(self, msg: str) -> None:
        if self.verbose and not self.quiet:
            print(f"mwir: debug: {msg}", file=sys.stderr)


# =========================================================================
# main
# =========================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``mwir`` console script. Never raises SystemExit."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code is None else int(exc.code)

    if args.list_bands:
        _print_band_catalog()
        return EXIT_OK

    log = _Logger(quiet=args.quiet, verbose=args.verbose)

    try:
        scenario = _resolve_scenario(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"mwir: error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    if args.save_config:
        try:
            save_scenario_yaml(scenario, args.save_config)
            log.info(f"wrote resolved scenario to {args.save_config}")
        except OSError as exc:
            print(f"mwir: error writing --save-config: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR

    if args.format == "npz" and not args.output:
        print("mwir: error: --format npz requires --output PATH", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        band_specs = _resolve_band_specs(args.bands)
    except ValueError as exc:
        print(f"mwir: error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    log.info(
        f"running scenario: {scenario.atmosphere.value}/{scenario.aerosol.value}, "
        f"{scenario.wavenumber_min_cm1:g}-{scenario.wavenumber_max_cm1:g} cm^-1 "
        f"@ {scenario.resolution_cm1:g} cm^-1 resolution"
    )

    try:
        cost = _estimate_run_cost(scenario, args.threads)
    except Exception as exc:  # noqa: BLE001 - same boundary run_scenario() uses below
        print(f"mwir: error resolving scenario geometry: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return EXIT_RUNTIME_ERROR

    log.info(
        f"estimated cost: {cost.n_points:,} spectral points x {cost.n_layers} layers "
        f"({cost.n_points * cost.n_layers:,} point-layers), ~{_format_bytes(cost.est_bytes)} "
        f"of layer arrays, ~{_format_duration(cost.est_seconds)} estimated runtime "
        "(order-of-magnitude -- see --help under 'accuracy')"
    )
    over_threshold = (
        cost.est_bytes >= _CONFIRM_BYTES_THRESHOLD or cost.est_seconds >= _CONFIRM_SECONDS_THRESHOLD
    )
    if over_threshold and not args.yes:
        proceed = False
        if sys.stdin.isatty() and not args.quiet:
            print(
                f"mwir: this run is estimated at ~{_format_bytes(cost.est_bytes)} and "
                f"~{_format_duration(cost.est_seconds)}, above the confirmation threshold "
                f"(~{_format_bytes(_CONFIRM_BYTES_THRESHOLD)} or "
                f"~{_format_duration(_CONFIRM_SECONDS_THRESHOLD)}). Consider narrowing "
                "--wavenumber-min-cm1/--wavenumber-max-cm1, raising --resolution-cm1, or "
                "shrinking --line-wing-cutoff-cm1 to match a narrow window.",
                file=sys.stderr,
            )
            try:
                answer = input("Proceed anyway? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            proceed = answer.strip().lower() in ("y", "yes")
        if not proceed:
            print(
                f"mwir: error: refusing to run without confirmation -- estimated "
                f"~{_format_bytes(cost.est_bytes)} / ~{_format_duration(cost.est_seconds)}, above the "
                f"~{_format_bytes(_CONFIRM_BYTES_THRESHOLD)} / ~{_format_duration(_CONFIRM_SECONDS_THRESHOLD)} "
                "threshold. Pass --yes to proceed, or shrink the run (see the suggestions above).",
                file=sys.stderr,
            )
            return EXIT_USAGE_ERROR

    t0 = time.time()
    try:
        result = run_scenario(
            scenario,
            include_radiance=not args.no_radiance,
            include_turbulence=not args.no_turbulence,
            n_threads=args.threads,
            backend=args.backend,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't crash
        print(f"mwir: error running scenario: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return EXIT_RUNTIME_ERROR
    log.debug(
        f"scenario complete in {time.time() - t0:.2f} s "
        f"({result.grid.n} spectral points, {result.path.n_layers} layers)"
    )
    if result.radiance is None:
        log.info("radiance unavailable (radiometry module not present yet) -- reporting transmittance only")

    try:
        _write_output(result, args, band_specs, log)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't crash
        print(f"mwir: error writing output: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return EXIT_RUNTIME_ERROR

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
