"""The ``mwir`` command line: flags -> :class:`Scenario` -> run -> write output.

A thin argparse wrapper around :func:`mwir_atmos.api.run_scenario`. Every
:class:`~mwir_atmos.types.Scenario` field is settable by flag, a scenario can
be loaded from / saved to YAML (:mod:`mwir_atmos.scenario_io`), and results
are written as a human-readable table or as CSV/JSON/NPZ with the full
spectral arrays (:mod:`mwir_atmos.report`). Before every run the estimated
cost is printed, and large runs need confirmation (:mod:`mwir_atmos.run_cost`).

Installed as the ``mwir`` console script (see ``pyproject.toml``); also
runnable as ``python -m mwir_atmos.cli`` or callable from tests via
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
    ``--bands`` token, ``--format npz`` without ``--output``), or a large run
    refused without confirmation.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import traceback
from enum import Enum
from typing import Optional, Sequence

import yaml

from . import __version__, report, run_cost
from .api import run_scenario
from .scenario_io import (
    coerce_field,
    load_scenario_yaml,
    save_scenario_yaml,
    scenario_from_dict,
    scenario_to_dict,
)
from .spectral import resolution_cm1_for_wavelength_step
from .types import (
    AerosolModel,
    InstrumentFunction,
    ModelAtmosphere,
    RadianceSolver,
    Scenario,
    ScenarioResult,
    SpectralSampling,
    SurfaceType,
    TurbulenceModel,
)
from .validation import validate_scenario

__all__ = [
    "main",
    "build_parser",
    # Re-exported from mwir_atmos.scenario_io for existing scripts.
    "scenario_to_dict",
    "scenario_from_dict",
    "load_scenario_yaml",
    "save_scenario_yaml",
]

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2

_FORMATS = ("table", "csv", "json", "npz")


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

    overrides = {
        f.name: coerce_field(f.name, getattr(args, f.name))
        for f in dataclasses.fields(Scenario)
        if getattr(args, f.name, None) is not None
    }
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
    validate_scenario(resolved)
    return resolved


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


def _error(msg: str, *, verbose: bool = False) -> None:
    """Print ``mwir: <msg>`` on stderr; with ``--verbose``, the active traceback too."""
    print(f"mwir: {msg}", file=sys.stderr)
    if verbose:
        traceback.print_exc()


# =========================================================================
# Confirmation gate and output
# =========================================================================


def _confirm_large_run(cost: run_cost.CostEstimate, args: argparse.Namespace) -> bool:
    """Whether a run may proceed: small runs always; large ones with ``--yes``
    or an interactive "y". Without a terminal, a large run is refused rather
    than hanging on a prompt."""
    if args.yes or not run_cost.needs_confirmation(cost):
        return True
    size = (f"~{run_cost.format_bytes(cost.est_bytes)} / "
            f"~{run_cost.format_duration(cost.est_seconds)}")
    threshold = (f"~{run_cost.format_bytes(run_cost.CONFIRM_BYTES_THRESHOLD)} / "
                 f"~{run_cost.format_duration(run_cost.CONFIRM_SECONDS_THRESHOLD)}")
    if sys.stdin.isatty() and not args.quiet:
        print(
            f"mwir: this run is estimated at {size}, above the confirmation threshold "
            f"({threshold}). Consider narrowing --wavenumber-min-cm1/--wavenumber-max-cm1, "
            "raising --resolution-cm1, or shrinking --line-wing-cutoff-cm1 to match a "
            "narrow window.",
            file=sys.stderr,
        )
        try:
            answer = input("Proceed anyway? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer.strip().lower() in ("y", "yes"):
            return True
    _error(f"error: refusing to run without confirmation -- estimated {size}, above the "
           f"{threshold} threshold. Pass --yes to proceed, or shrink the run "
           "(see the suggestions above).")
    return False


def _write_output(result: ScenarioResult, args: argparse.Namespace,
                  band_specs: Sequence[tuple[str, report.BandSpec]], log: _Logger) -> None:
    if args.format == "csv":
        with _open_text_output(args.output) as fh:
            report.write_csv(result, fh)
        return

    rows = report.band_rows(result, band_specs, warn=log.warn)
    if args.format == "npz":
        report.write_npz(result, rows, args.output)
        return

    with _open_text_output(args.output) as fh:
        if args.format == "table":
            report.write_table(result, rows, fh)
        elif args.format == "json":
            report.write_json(result, rows, fh)
        else:  # pragma: no cover - argparse `choices=` already prevents this
            raise AssertionError(f"unhandled --format {args.format!r}")


def _open_text_output(path: Optional[str]):
    """Context manager yielding ``sys.stdout`` (left open) or a new file."""
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
        report.print_band_catalog()
        return EXIT_OK

    log = _Logger(quiet=args.quiet, verbose=args.verbose)

    try:
        scenario = _resolve_scenario(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        _error(f"error: {exc}")
        return EXIT_RUNTIME_ERROR

    if args.save_config:
        try:
            save_scenario_yaml(scenario, args.save_config)
            log.info(f"wrote resolved scenario to {args.save_config}")
        except OSError as exc:
            _error(f"error writing --save-config: {exc}")
            return EXIT_RUNTIME_ERROR

    if args.format == "npz" and not args.output:
        _error("error: --format npz requires --output PATH")
        return EXIT_USAGE_ERROR

    try:
        band_specs = report.resolve_band_specs(args.bands)
    except ValueError as exc:
        _error(f"error: {exc}")
        return EXIT_USAGE_ERROR

    log.info(
        f"running scenario: {scenario.atmosphere.value}/{scenario.aerosol.value}, "
        f"{scenario.wavenumber_min_cm1:g}-{scenario.wavenumber_max_cm1:g} cm^-1 "
        f"@ {scenario.resolution_cm1:g} cm^-1 resolution"
    )

    try:
        cost = run_cost.estimate_run_cost(scenario, args.threads)
    except Exception as exc:  # noqa: BLE001 - same boundary run_scenario() uses below
        _error(f"error resolving scenario geometry: {exc}", verbose=args.verbose)
        return EXIT_RUNTIME_ERROR
    log.info(
        f"estimated cost: {cost.n_points:,} spectral points x {cost.n_layers} layers "
        f"({cost.n_points * cost.n_layers:,} point-layers), "
        f"~{run_cost.format_bytes(cost.est_bytes)} of layer arrays, "
        f"~{run_cost.format_duration(cost.est_seconds)} estimated runtime "
        "(order-of-magnitude -- see --help under 'accuracy')"
    )
    if not _confirm_large_run(cost, args):
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
        _error(f"error running scenario: {exc}", verbose=args.verbose)
        return EXIT_RUNTIME_ERROR
    log.debug(
        f"scenario complete in {time.time() - t0:.2f} s "
        f"({result.grid.n} spectral points, {result.path.n_layers} layers)"
    )

    try:
        _write_output(result, args, band_specs, log)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't crash
        _error(f"error writing output: {exc}", verbose=args.verbose)
        return EXIT_RUNTIME_ERROR

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
