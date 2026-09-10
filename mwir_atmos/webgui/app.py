"""Streamlit GUI: mid-wave infrared atmospheric transmittance and radiance.

DEV-BETA owned (CONTRACT.md "Wave 2 interfaces"). The sidebar builds a
:class:`~mwir_atmos.types.Scenario` field-by-field (the same fields as
:mod:`mwir_atmos.cli`'s flags, grouped the same way: atmosphere / sensor /
source / illumination / spectral / accuracy); the main panel calls
:func:`mwir_atmos.api.run_scenario` exactly once per distinct scenario (see
:func:`_run_scenario_cached`) and renders transmittance, a per-species
breakdown, radiance (when the radiometry layer is present), the atmospheric
profile, and a band summary table.

Run with::

    streamlit run mwir_atmos/webgui/app.py

Caching
-------
``st.cache_data`` needs a hashable key. ``Scenario`` is a dataclass carrying
enums and an ``Optional[datetime]``, which Streamlit's default hasher does
not treat predictably across versions (job brief: "handle that properly
rather than disabling the cache"). Rather than fight the hasher with
``hash_funcs``, this module reduces a ``Scenario`` to the same plain-dict
form the CLI already uses for ``--config``/``--save-config``
(:func:`mwir_atmos.cli.scenario_to_dict` / :func:`~mwir_atmos.cli.scenario_from_dict`)
and caches on its JSON string -- a ``str`` is trivially and unambiguously
hashable, and reusing the CLI's (de)serialisation means the two front ends
can never grow two different ideas of what a scenario looks like (see
``cli.py``'s own module docstring, which anticipates exactly this reuse).

Widgets that are purely about *display* (axis units, which components to
plot, which bands to tabulate) are kept outside the Scenario entirely, so
toggling them reruns the script (Streamlit's normal execution model) but
hits the cache immediately instead of recomputing. Scenario-affecting
controls use ``number_input``/``selectbox``/``checkbox`` rather than
continuous-drag ``slider`` widgets, so an in-progress drag does not fire a
cascade of intermediate (uncached) physics runs -- only a committed value
does.

Import style
------------
Streamlit execs an app file directly (as `streamlit run path/to/app.py`
would), not as a package submodule -- there is no parent package context at
run time, so package-relative imports (``from .. import x``) raise
``ImportError: attempted relative import with no known parent package``
(confirmed here with ``streamlit.testing.v1.AppTest``, which runs a script
exactly the way ``streamlit run`` does). Every ``mwir_atmos`` import below
is therefore absolute (``from mwir_atmos... import ...``), with a small
``sys.path`` bootstrap so ``mwir_atmos`` is importable even when this file
is launched with the project root *not* already on ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import io
import json
import yaml
from datetime import date, datetime, time as dtime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from mwir_atmos import constants
from mwir_atmos.api import run_scenario
from mwir_atmos.setups import with_overrides, sweep_distance
from mwir_atmos.spectral import band_average
from mwir_atmos.cli import (
    RADIANCE_COMPONENTS,
    _band_rows,
    _CONFIRM_BYTES_THRESHOLD,
    _CONFIRM_SECONDS_THRESHOLD,
    _estimate_run_cost,
    _format_bytes,
    _format_duration,
    _Logger,
    _write_csv,
    scenario_from_dict,
    scenario_to_dict,
)
from mwir_atmos.types import (
    AerosolModel,
    InstrumentFunction,
    RadianceSolver,
    SpectralSampling,
    ModelAtmosphere,
    REQUIRED_SPECIES,
    Scenario,
    ScenarioResult,
    Species,
    SurfaceType,
    TurbulenceModel,
)
from mwir_atmos.webgui import plots

st.set_page_config(page_title="mwir_atmos", page_icon=":thermometer:", layout="wide")

_MODEL_LABELS = {m: m.value.replace("_", " ").title() for m in ModelAtmosphere}
_SURFACE_LABELS = {s: s.value.replace("_", " ").title() for s in SurfaceType}
_AEROSOL_LABELS = {a: a.value.replace("_", " ").title() for a in AerosolModel}
_AEROSOL_LABELS.update({a: "LOWTRAN table: " + a.value.removeprefix("lowtran_").title()
                        for a in AerosolModel if a.value.startswith("lowtran_")})

#: Quick spectral-range presets (job brief: "make selecting a narrow window
#: quick and obvious"). Selecting one pins wavenumber-min/max/resolution and
#: the accuracy controls; switching to "Custom" (or just editing again) frees
#: the fields for hand-tuning -- see _spectral_and_accuracy_controls.
_SPECTRAL_PRESETS: dict[str, dict[str, float]] = {
    "Full band (default)": dict(
        wn_min=constants.MWIR_GRID_MIN_CM1, wn_max=constants.MWIR_GRID_MAX_CM1,
        resolution=4.0, wing=25.0, strength=0.0, tau=1.0e-5,
    ),
    "MWIR core, 3-5 um": dict(wn_min=2000.0, wn_max=3333.33, resolution=2.0, wing=25.0,
                              strength=0.0, tau=1.0e-5),
    "Narrow + fine (CO line demo)": dict(wn_min=2171.5, wn_max=2174.0, resolution=0.01, wing=3.0,
                                         strength=0.0, tau=1.0e-7),
}

_SAMPLING_LABELS = {
    SpectralSampling.POINT: "Point (fast: physics at output points only)",
    SpectralSampling.RESOLVED: "Line-resolved + instrument function (accurate)",
}


# =========================================================================
# Scenario <-> Streamlit cache key
# =========================================================================


def _scenario_cache_key(scenario: Scenario) -> str:
    """A hashable, order-stable stand-in for ``scenario`` (see module docstring)."""
    return json.dumps(scenario_to_dict(scenario), sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=8)
def _run_scenario_cached(cache_key: str) -> ScenarioResult:
    scenario = scenario_from_dict(json.loads(cache_key))
    # Turbulence is computed whenever the sidebar selects a Cn^2 model; with
    # TurbulenceModel.NONE the api layer skips it, so this stays cheap.
    return run_scenario(scenario, include_radiance=True, include_turbulence=True)


# =========================================================================
# Small sidebar helpers
# =========================================================================


def _optional_number(label: str, *, key: str, default: float, help: Optional[str] = None, **kwargs) -> Optional[float]:
    """A checkbox-gated number_input: unchecked -> None (keep the model/scenario
    default), matching the CLI's "flag omitted -> None" semantics exactly."""
    enabled = st.checkbox(f"Override {label}", key=f"{key}__on")
    if not enabled:
        return None
    return float(st.number_input(label, value=default, key=key, help=help, **kwargs))


def _enum_select(label: str, options: dict, current, *, key: str):
    keys = list(options.keys())
    idx = keys.index(current) if current in keys else 0
    choice = st.sidebar.selectbox(label, keys, index=idx, format_func=lambda k: options[k], key=key)
    return choice


# =========================================================================
# Sidebar: build a Scenario
# =========================================================================


def _atmosphere_controls() -> dict:
    st.sidebar.header("Atmosphere")
    atmosphere = _enum_select("Model atmosphere", _MODEL_LABELS, ModelAtmosphere.MIDLAT_SUMMER, key="atmosphere")
    surface_type = _enum_select("Surface type", _SURFACE_LABELS, SurfaceType.LAND_VEGETATED, key="surface_type")
    aerosol = _enum_select("Aerosol model", _AEROSOL_LABELS, AerosolModel.RURAL, key="aerosol")
    if aerosol.value.startswith("lowtran_"):
        st.sidebar.caption(
            "LOWTRAN 7 optical tables, RH 0–99%, background altitude profiles. "
            "Visibility: 2–50 km. Winter profiles follow winter atmospheres; "
            "otherwise spring/summer. Rayleigh is included separately."
        )
    visibility_km = st.sidebar.number_input(
        "Visibility (km)", min_value=0.1, value=23.0, step=1.0, key="visibility_km",
        help="Legacy models use Koschmieder extinction. LOWTRAN tables use their original 550 nm profiles (2–50 km visibility).",
    )
    wind_speed_m_s = st.sidebar.number_input(
        "Wind speed (m/s)", min_value=0.0, value=5.0, step=0.5, key="wind_speed_m_s",
        help="navy_maritime aerosol only.",
    )
    with st.sidebar.expander("Advanced atmosphere overrides", expanded=False):
        ground_temp_k = _optional_number("ground temperature (K)", key="ground_temp_k", default=288.2, min_value=150.0, max_value=340.0)
        ground_rh_pct = _optional_number("ground RH (%)", key="ground_rh_pct", default=50.0, min_value=0.0, max_value=100.0)
        surface_skin_temp_k = _optional_number("surface skin temperature (K)", key="surface_skin_temp_k", default=288.2, min_value=150.0, max_value=340.0)
        surface_emissivity = _optional_number("surface emissivity", key="surface_emissivity", default=0.95, min_value=0.0, max_value=1.0, step=0.01)
        co2_ppmv = _optional_number("CO2 (ppmv)", key="co2_ppmv", default=constants.BACKGROUND_VMR_PPMV["CO2"], min_value=0.0)
        surface_altitude_km = st.number_input(
            "Re-anchor surface altitude (km)", value=0.0, min_value=0.0, step=0.1, key="surface_altitude_km",
            help="Raises the model atmosphere's base level, e.g. for a plateau site.",
        )
    return dict(
        atmosphere=atmosphere, surface_type=surface_type, aerosol=aerosol,
        visibility_km=visibility_km, wind_speed_m_s=wind_speed_m_s,
        ground_temp_k=ground_temp_k, ground_rh_pct=ground_rh_pct,
        surface_skin_temp_k=surface_skin_temp_k, surface_emissivity=surface_emissivity,
        co2_ppmv=co2_ppmv, surface_altitude_km=surface_altitude_km,
    )


def _sensor_controls() -> dict:
    st.sidebar.header("Sensor")
    sensor_alt_km = st.sidebar.number_input("Sensor altitude (km)", value=0.0, min_value=0.0, step=0.1, key="sensor_alt_km")
    sensor_zenith_deg = st.sidebar.number_input(
        "Sensor zenith (deg)", value=0.0, min_value=0.0, max_value=180.0, step=1.0, key="sensor_zenith_deg",
        help="0 = straight up, 90 = horizontal, 180 = straight down.",
    )
    sensor_azimuth_deg = st.sidebar.number_input(
        "Sensor azimuth (deg)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="sensor_azimuth_deg",
        help="Clockwise from true north. Does not affect the (horizontally stratified) geometry, only sun/moon relative angles.",
    )
    latitude_deg = st.sidebar.number_input("Latitude (deg)", value=35.0, min_value=-90.0, max_value=90.0, step=1.0, key="latitude_deg")
    longitude_deg = st.sidebar.number_input("Longitude (deg)", value=0.0, min_value=-180.0, max_value=180.0, step=1.0, key="longitude_deg")
    sensor_aperture_diameter_m = st.sidebar.number_input(
        "Aperture diameter (m)", value=0.10, min_value=0.001, max_value=10.0, step=0.05,
        format="%.3f", key="sensor_aperture_diameter_m",
        help="Clear aperture of the receiving optic. Sets the diffraction limit "
             "and, with a turbulence model selected, the Strehl ratio and "
             "point-source blur. Turbulence only degrades a point source once "
             "this exceeds the Fried parameter r0 (~60 cm at 4 um, zenith).",
    )
    return dict(
        sensor_alt_km=sensor_alt_km, sensor_zenith_deg=sensor_zenith_deg,
        sensor_azimuth_deg=sensor_azimuth_deg, latitude_deg=latitude_deg, longitude_deg=longitude_deg,
        sensor_aperture_diameter_m=sensor_aperture_diameter_m,
    )


def _source_controls() -> dict:
    with st.sidebar.expander("Source / target", expanded=False):
        st.caption("A discrete target along the path (e.g. a hot spot). Leave everything off to integrate to the end of the path (ground or space).")
        source_alt_km = _optional_number("source altitude (km)", key="source_alt_km", default=0.0, min_value=0.0)
        source_range_km = _optional_number("source range (km)", key="source_range_km", default=5.0, min_value=0.01)
        source_temp_k = _optional_number("source temperature (K)", key="source_temp_k", default=650.0, min_value=0.0)
        source_area_m2 = _optional_number("source area (m^2)", key="source_area_m2", default=1.0, min_value=0.0)
        source_emissivity = st.number_input("Source emissivity", value=1.0, min_value=0.0, max_value=1.0, step=0.01, key="source_emissivity")
    return dict(
        source_alt_km=source_alt_km, source_range_km=source_range_km, source_temp_k=source_temp_k,
        source_area_m2=source_area_m2, source_emissivity=source_emissivity,
    )


def _illumination_controls() -> dict:
    with st.sidebar.expander("Illumination", expanded=False):
        use_dt = st.checkbox("Set local date/time", key="use_local_datetime")
        local_datetime = None
        if use_dt:
            d = st.date_input("Local date", value=date(2026, 7, 15), key="local_date")
            t = st.time_input("Local time", value=dtime(13, 30), key="local_time")
            local_datetime = datetime.combine(d, t)
        utc_offset_hours = _optional_number("UTC offset (hours)", key="utc_offset_hours", default=0.0, min_value=-14.0, max_value=14.0)
        st.caption(
            "Without an explicit UTC offset the timezone is guessed from longitude, "
            "which can be an hour or more out (Madrid keeps CET on the Greenwich "
            "meridian) — tens of degrees of solar position error. Supplying observed "
            "sunrise **and** sunset below overrides the timezone entirely and pins "
            "the sun to ~0.1°."
        )
        sunrise_local_hours = _optional_number(
            "sunrise (local clock hours)", key="sunrise_local_hours", default=6.75,
            min_value=0.0, max_value=24.0,
        )
        sunset_local_hours = _optional_number(
            "sunset (local clock hours)", key="sunset_local_hours", default=21.80,
            min_value=0.0, max_value=24.0,
        )
        moon_phase = st.number_input(
            "Moon phase", value=0.5, min_value=0.0, max_value=1.0, step=0.05, key="moon_phase",
            help="0 = new, 0.5 = full, 1 = new again.",
        )
        include_solar = st.checkbox("Include solar illumination", value=True, key="include_solar")
        include_lunar = st.checkbox("Include lunar illumination", value=True, key="include_lunar")

    with st.sidebar.expander("Optical turbulence", expanded=False):
        st.caption(
            "Diagnostic only — turbulence redistributes a point source in angle "
            "and time, it does not absorb, so no radiance value changes."
        )
        turbulence = st.selectbox(
            "Cn² profile model", options=list(TurbulenceModel),
            format_func=lambda m: m.value, index=0, key="turbulence",
        )
        turbulence_cn2_ground = _optional_number(
            "ground Cn² (m^-2/3)", key="turbulence_cn2_ground", default=1.7e-14,
            min_value=0.0, format="%.3e",
        )

    return dict(
        local_datetime=local_datetime, utc_offset_hours=utc_offset_hours, moon_phase=moon_phase,
        include_solar=include_solar, include_lunar=include_lunar,
        sunrise_local_hours=sunrise_local_hours, sunset_local_hours=sunset_local_hours,
        turbulence=turbulence, turbulence_cn2_ground=turbulence_cn2_ground,
    )


def _spectral_and_accuracy_controls() -> dict:
    st.sidebar.header("Spectral range & resolution")
    for k, v in dict(
        wn_min=constants.MWIR_GRID_MIN_CM1, wn_max=constants.MWIR_GRID_MAX_CM1,
        resolution_cm1=4.0,  # coarser than the CLI/Scenario default (1.0) -- keeps the GUI interactive
        line_wing_cutoff_cm1=25.0, line_strength_cutoff=0.0, line_optical_depth_cutoff=1.0e-5,
    ).items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("_last_spectral_preset", None)

    preset = st.sidebar.selectbox("Quick preset", list(_SPECTRAL_PRESETS) + ["Custom"], index=0, key="spectral_preset")
    if preset != "Custom" and st.session_state["_last_spectral_preset"] != preset:
        p = _SPECTRAL_PRESETS[preset]
        st.session_state["wn_min"] = p["wn_min"]
        st.session_state["wn_max"] = p["wn_max"]
        st.session_state["resolution_cm1"] = p["resolution"]
        st.session_state["line_wing_cutoff_cm1"] = p["wing"]
        st.session_state["line_strength_cutoff"] = p["strength"]
        st.session_state["line_optical_depth_cutoff"] = p["tau"]
    st.session_state["_last_spectral_preset"] = None if preset == "Custom" else preset

    wn_min = st.sidebar.number_input("Wavenumber min (cm^-1)", min_value=400.0, max_value=8000.0, step=1.0, key="wn_min")
    wn_max = st.sidebar.number_input("Wavenumber max (cm^-1)", min_value=400.0, max_value=8000.0, step=1.0, key="wn_max")
    if wn_max > wn_min:
        st.sidebar.caption(f"-> {1.0e4 / wn_max:.4f} - {1.0e4 / wn_min:.4f} um")
    resolution_cm1 = st.sidebar.number_input(
        "Resolution (cm^-1)", min_value=1.0e-4, max_value=50.0, step=0.1, format="%.4f", key="resolution_cm1",
        help="Never capped -- set it as fine as your data needs. Cost scales with points x layers; "
             "see the cost estimate below before running a wide window at fine resolution.",
    )

    sampling = _enum_select("Spectral sampling", _SAMPLING_LABELS, SpectralSampling.POINT,
                               key="spectral_sampling")
    resolved_fields: dict = {}
    if sampling == SpectralSampling.RESOLVED:
        resolved_fields["instrument_function"] = _enum_select(
            "Instrument function", {f: f.value.title() for f in InstrumentFunction},
            InstrumentFunction.BOX, key="instrument_function")
        fwhm = st.sidebar.number_input(
            "Instrument FWHM (cm^-1, 0 = resolution)", min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.4f", key="instrument_fwhm_cm1")
        fine = st.sidebar.number_input(
            "Fine step (cm^-1, 0 = automatic)", min_value=0.0, max_value=1.0, value=0.0,
            step=0.001, format="%.4f", key="fine_step_cm1")
        resolved_fields["instrument_fwhm_cm1"] = fwhm if fwhm > 0 else None
        resolved_fields["fine_step_cm1"] = fine if fine > 0 else None
        st.sidebar.caption(
            "The physics runs on a grid fine enough to resolve absorption lines, then is "
            "averaged through the instrument function onto the output grid. Cost follows the "
            "bandwidth and the fine step, not the output resolution.")

    st.sidebar.subheader("Accuracy")
    # NOTE: value= is deliberately omitted on both widgets below -- the key
    # is pre-seeded into session_state above (and by the preset block), and
    # passing both value= and a session_state-backed key for the same widget
    # is what Streamlit's "handle caching/state properly" warning is about.
    line_optical_depth_cutoff = st.sidebar.number_input(
        "Line optical-depth cutoff", min_value=0.0, max_value=1.0, step=1.0e-6, format="%.2e",
        key="line_optical_depth_cutoff",
        help="Drop a line from a layer when its peak optical depth there is below this. "
             "Tied to the actual gas amount, so humid layers keep the weak water lines that "
             "matter in the windows. 0.0 keeps every line.",
    )
    line_strength_cutoff = st.sidebar.number_input(
        "Legacy line strength cutoff", min_value=0.0, max_value=1.0, step=1.0e-5, format="%.2e",
        key="line_strength_cutoff",
        help="Drop lines weaker than this fraction of each species' strongest line. Off (0.0) "
             "by default: the former default of 1e-4 discarded every water line in the "
             "3.5-4.1 um window.",
    )
    line_wing_cutoff_cm1 = st.sidebar.number_input(
        "Line wing cutoff (cm^-1)", min_value=0.1, max_value=200.0, step=0.5, key="line_wing_cutoff_cm1",
        help="Local-line half-width for gases other than water. H2O with MT_CKD always uses "
             "25 cm^-1 with pedestal subtraction. Smaller cutoffs can omit absorption; "
             "check convergence. Evaluation windows are clipped to the requested grid.",
    )
    return dict(
        wavenumber_min_cm1=wn_min, wavenumber_max_cm1=wn_max, resolution_cm1=resolution_cm1,
        line_strength_cutoff=line_strength_cutoff, line_wing_cutoff_cm1=line_wing_cutoff_cm1,
        line_optical_depth_cutoff=line_optical_depth_cutoff, spectral_sampling=sampling,
        **resolved_fields,
    )


def build_scenario_from_sidebar() -> Scenario:
    st.sidebar.title("mwir_atmos")
    source = st.sidebar.radio("Setup source", ["Current controls", "Saved YAML"], key="setup_source")
    if source == "Saved YAML":
        uploaded = st.sidebar.file_uploader("Open saved setup", type=["yaml", "yml"], key="setup_upload")
        st.sidebar.caption("Use a downloaded setup or an existing CLI configuration. Overrides replace only the named values.")
        overrides = st.sidebar.text_area("Overrides (YAML)", value="{}", key="setup_overrides",
                                         help="Example: source_range_km: 5.0. Use null to clear an optional value.")
        if uploaded is None:
            st.info("Choose a YAML setup in the sidebar to continue.")
            st.stop()
        try:
            data = yaml.safe_load(uploaded.getvalue().decode("utf-8-sig"))
            changes = yaml.safe_load(overrides)
            if not isinstance(data, dict) or not isinstance(changes, dict):
                raise ValueError("Setup and overrides must be YAML mappings")
            return with_overrides(scenario_from_dict(data), **changes)
        except (ValueError, TypeError, UnicodeError, yaml.YAMLError) as exc:
            st.error(f"Could not load saved setup: {exc}")
            st.stop()
    fields: dict = {}
    fields.update(_atmosphere_controls())
    fields.update(_sensor_controls())
    fields.update(_source_controls())
    fields.update(_illumination_controls())
    fields.update(_spectral_and_accuracy_controls())
    solver = _enum_select("Radiance solver", {
        RadianceSolver.SINGLE_SCATTER: "Single scattering (spherical paths)",
        RadianceSolver.DISORT: "DISORT multiple scattering (plane-parallel)",
        RadianceSolver.SPHERICAL_MC: "Spherical multiple scattering (Monte Carlo)",
    }, RadianceSolver.SINGLE_SCATTER, key="radiance_solver")
    fields['radiance_solver'] = solver
    if solver == RadianceSolver.DISORT:
        fields['disort_streams'] = st.sidebar.selectbox("DISORT streams", [8,16,32,64], index=2, key="disort_streams")
        st.sidebar.caption("Solves the full vertical column and repeated surface reflection. "
                           "Plane-parallel geometry replaces spherical paths. Horizontal views need a finite target range. "
                           "Increase streams to check angular convergence; near-horizon curvature is not modeled.")
    if solver == RadianceSolver.SPHERICAL_MC:
        fields["spherical_photons"] = st.sidebar.number_input("Photon histories per spectral point", min_value=128, max_value=1000000, value=4096, step=1024)
        fields["spherical_seed"] = st.sidebar.number_input("Monte Carlo seed", min_value=0, max_value=2**32-1, value=0)
        st.sidebar.caption("Experimental solver: curved atmospheric shells for every photon path; scalar, no refraction. Sampling noise decreases approximately as 1/sqrt(histories). Start with a narrow spectrum.")
    return Scenario(**fields)


# =========================================================================
# Cost estimate + confirmation gate (mirrors mwir_atmos.cli.main exactly --
# same thresholds, same estimator -- see cli.py for the calibration notes)
# =========================================================================


def _cost_and_gate(scenario: Scenario, cache_key: str) -> bool:
    try:
        cost = _estimate_run_cost(scenario)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
        st.error(f"Could not resolve this scenario's geometry: {exc}")
        return False

    c1, c2, c3 = st.columns(3)
    c1.metric("Spectral points x layers", f"{cost.n_points:,} x {cost.n_layers}")
    c2.metric("Estimated layer-array memory", _format_bytes(cost.est_bytes))
    c3.metric("Estimated runtime", _format_duration(cost.est_seconds))
    st.caption(
        "Order-of-magnitude estimate, not a scheduler -- actual cost also depends on how many "
        "spectral lines populate the requested window. See README.md 'Choosing spectral resolution'."
    )

    over_threshold = cost.est_bytes >= _CONFIRM_BYTES_THRESHOLD or cost.est_seconds >= _CONFIRM_SECONDS_THRESHOLD
    if not over_threshold:
        return True
    return st.checkbox(
        f"This run is estimated at ~{_format_bytes(cost.est_bytes)} / ~{_format_duration(cost.est_seconds)}, "
        f"above the usual ~{_format_bytes(_CONFIRM_BYTES_THRESHOLD)} / ~{_format_duration(_CONFIRM_SECONDS_THRESHOLD)} "
        "guideline. Run it anyway?",
        key=f"confirm_large_run::{cache_key}",
    )


# =========================================================================
# Main panel sections
# =========================================================================


def _x_axis(grid, use_wavelength: bool) -> tuple[np.ndarray, str]:
    if use_wavelength:
        return grid.wavelength_um, "Wavelength (um)"
    return grid.wavenumber_cm1, "Wavenumber (cm^-1)"


def _scenario_summary(result: ScenarioResult) -> None:
    sc = result.scenario
    path = result.path
    bits = [
        f"**{_MODEL_LABELS[sc.atmosphere]}** atmosphere, **{_AEROSOL_LABELS[sc.aerosol]}** aerosol "
        f"(visibility {sc.visibility_km:g} km)",
        f"sensor alt {sc.sensor_alt_km:g} km, zenith {sc.sensor_zenith_deg:g} deg, azimuth {sc.sensor_azimuth_deg:g} deg",
        f"path: {path.n_layers} layers, {path.total_length_km:.3f} km, "
        f"hits_ground={path.hits_ground}, reaches_space={path.reaches_space}"
        + (f", tangent_alt={path.tangent_alt_km:.2f} km" if path.tangent_alt_km is not None else ""),
        f"grid: {result.grid.wavenumber_cm1[0]:.2f}-{result.grid.wavenumber_cm1[-1]:.2f} cm^-1 "
        f"({result.grid.wavelength_um[-1]:.3f}-{result.grid.wavelength_um[0]:.3f} um), "
        f"{result.grid.resolution_cm1:g} cm^-1 resolution, {result.grid.n} points"
        + (f", line-resolved at {result.fine_step_cm1:.4g} cm^-1 through a "
           f"{sc.instrument_function.value} instrument function"
           if result.fine_step_cm1 is not None else ", point-sampled"),
    ]
    if sc.radiance_solver == RadianceSolver.DISORT:
        bits.insert(0, f"**DISORT multiple scattering**, {sc.disort_streams} streams, **plane-parallel geometry**")
    if sc.radiance_solver == RadianceSolver.SPHERICAL_MC:
        bits.insert(0, f"**Spherical multiple scattering**, {sc.spherical_photons} photon histories per spectral point, seed {sc.spherical_seed}")
    if result.sun is not None:
        bits.append(
            f"sun: zenith {result.sun.zenith_deg:.1f} deg, above_horizon={result.sun.is_above_horizon}"
        )
    if result.moon is not None:
        bits.append(
            f"moon: zenith {result.moon.zenith_deg:.1f} deg, above_horizon={result.moon.is_above_horizon}, "
            f"illuminated={result.moon.illuminated_fraction:.2f}"
        )
    st.markdown("  \n".join(f"- {b}" for b in bits))
    if result.radiance is None:
        st.info(
            "Radiance unavailable: the radiometry module (mwir_atmos.radiometry.radiance) is not "
            "present in this build yet. Showing transmittance only -- rerun once it lands."
        )


def _transmittance_section(result: ScenarioResult, x: np.ndarray, x_label: str) -> None:
    st.subheader(f"Transmittance vs {'wavelength' if x_label.startswith('Wavelength') else 'wavenumber'}")
    t = result.transmittance
    available = {"total": t.total, "molecular": t.molecular, "aerosol": t.aerosol, "rayleigh": t.rayleigh}
    chosen = st.multiselect(
        "Components", list(available), default=["total", "molecular", "aerosol"], key="trans_components"
    )
    series = {name: available[name] for name in chosen}
    plots.line_chart(st, x, series, x_label=x_label, y_label="Transmittance", title="")
    png = plots.line_chart_png(x, series, x_label=x_label, y_label="Transmittance")
    _download_row(f"transmittance_{'_'.join(chosen) or 'none'}", png)


def _species_section(result: ScenarioResult, x: np.ndarray, x_label: str) -> None:
    st.subheader("Per-species transmittance")
    by_species = result.transmittance.by_species
    ordered_names = [sp.value for sp in REQUIRED_SPECIES if sp in by_species]
    chosen = st.multiselect("Species", ordered_names, default=ordered_names, key="species_components")
    name_to_species = {sp.value: sp for sp in REQUIRED_SPECIES}
    series = {name: by_species[name_to_species[name]] for name in chosen if name_to_species[name] in by_species}
    plots.line_chart(st, x, series, x_label=x_label, y_label="Transmittance", title="")
    png = plots.line_chart_png(x, series, x_label=x_label, y_label="Transmittance")
    _download_row(f"species_{'_'.join(chosen) or 'none'}", png)


def _radiance_section(result: ScenarioResult, x: np.ndarray, x_label: str) -> None:
    st.subheader(f"Radiance vs {'wavelength' if x_label.startswith('Wavelength') else 'wavenumber'}")
    if result.radiance is None:
        st.info(
            "Radiance unavailable (radiometry module not present in this build). Every "
            "RadianceResult component will render here, stacked, once it lands."
        )
        return
    if result.radiance.sampling_standard_error:
        error = result.radiance.sampling_standard_error["total"]
        relative = np.divide(error, result.radiance.total, out=np.zeros_like(error), where=result.radiance.total>0)
        st.caption(f"Monte Carlo sampling uncertainty: maximum relative standard error {100*float(np.max(relative)):.2g}% over this spectrum. One standard error, not a bound on physical-model error; rare unsampled contributions can be missed. Increase histories and change the seed to check convergence.")
    components = {name: getattr(result.radiance, name) for name in RADIANCE_COMPONENTS if name != "total"}
    if result.scenario.radiance_solver in (RadianceSolver.DISORT, RadianceSolver.SPHERICAL_MC):
        st.caption("Thermal components include subsequent scattering and surface reflection. "
                   "Solar/lunar multiple-scattering components include all diffuse atmospheric and surface contributions. "
                   "Direct discs and target emission are separate.")
    from mwir_atmos.spectral import radiance_to_per_um
    total = result.radiance.total
    y_label = "Radiance (W cm^-2 sr^-1 / cm^-1)"
    if x_label.startswith("Wavelength"):
        components = {name: radiance_to_per_um(y, result.grid.wavenumber_cm1)
                      for name, y in components.items()}
        total = radiance_to_per_um(total, result.grid.wavenumber_cm1)
        y_label = "Radiance (W m^-2 sr^-1 / um)"
    plots.stacked_area_chart(
        st, x, components, total=total,
        x_label=x_label, y_label=y_label, title="",
    )
    png = plots.stacked_area_png(
        x, components, total=total, x_label=x_label, y_label=y_label
    )
    _download_row("radiance", png)


def _turbulence_section(result: ScenarioResult) -> None:
    """Optical turbulence diagnostics, shown only when a Cn^2 model is selected."""
    t = result.turbulence
    if t is None:
        return

    st.subheader("Optical turbulence")
    st.caption(
        f"Cn² model **{result.scenario.turbulence.value}**, evaluated at "
        f"{t.reference_wavelength_um:.2f} µm. These are diagnostics: turbulence "
        "redistributes a point source in angle and time but absorbs nothing, so "
        "no radiance above is modified."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fried parameter r₀", f"{t.fried_parameter_r0_cm:.1f} cm",
                help="Aperture size over which the wavefront stays roughly flat. "
                     "Scales as λ^(6/5), so MWIR values are ~12x the visible ones.")
    col2.metric("Isoplanatic angle θ₀", f"{t.isoplanatic_angle_urad:.1f} µrad",
                help="Angular field over which the turbulence-induced wavefront error is common.")
    col3.metric("Scintillation index", f"{t.scintillation_index:.4f}",
                help="Normalised intensity variance. Saturates near 1 in strong turbulence.")
    col4.metric("Greenwood freq.", "n/a" if t.greenwood_frequency_hz is None
                else f"{t.greenwood_frequency_hz:.1f} Hz",
                help="Bandwidth an adaptive-optics loop would need to track the turbulence.")

    if t.aperture_diameter_m is None:
        return

    st.markdown("**Point-source response** for a %.0f cm aperture" % (t.aperture_diameter_m * 100))
    regime = "seeing-limited" if t.is_seeing_limited else "diffraction-limited"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diffraction limit λ/D", f"{t.diffraction_limit_urad:.2f} µrad")
    c2.metric("Seeing blur λ/r₀", f"{t.seeing_limit_urad:.2f} µrad")
    c3.metric("Effective spot", f"{t.effective_blur_urad:.2f} µrad")
    c4.metric("Strehl / peak ratio", f"{t.strehl_ratio:.3f}")

    if t.is_seeing_limited:
        st.warning(
            f"Aperture ({t.aperture_diameter_m * 100:.0f} cm) exceeds r₀ "
            f"({t.fried_parameter_r0_cm:.0f} cm) — **{regime}**. A point source is "
            f"blurred to ~{t.effective_blur_urad:.1f} µrad rather than the "
            f"{t.diffraction_limit_urad:.1f} µrad the optics could deliver, and "
            f"on-axis peak irradiance falls to {t.strehl_ratio:.0%} of the "
            "turbulence-free case. Total collected power is unchanged."
        )
    else:
        st.success(
            f"Aperture ({t.aperture_diameter_m * 100:.0f} cm) is smaller than r₀ "
            f"({t.fried_parameter_r0_cm:.0f} cm) — still **{regime}**. Turbulence "
            f"costs only {1 - t.strehl_ratio:.1%} of peak irradiance here."
        )

    with st.expander("Cn² vs altitude", expanded=False):
        st.caption(
            "Log axis: Cn² spans ~30 orders of magnitude between the surface and "
            "the top of the atmosphere, so a linear axis shows nothing."
        )
        altitudes = [0.5 * (l.z_bottom_km + l.z_top_km) for l in result.path.layers]
        plots.profile_chart(
            st, t.cn2_profile_m23, altitudes,
            x_label="Cn² (m^-2/3)", title="Refractive-index structure constant", log_x=True,
        )
        png = plots.profile_png(
            t.cn2_profile_m23, altitudes,
            x_label="Cn² (m^-2/3)", title="Refractive-index structure constant", log_x=True,
        )
        _download_row("turbulence_cn2", png)


def _profile_section(result: ScenarioResult) -> None:
    st.subheader("Atmospheric profile")
    profile = result.profile
    col1, col2 = st.columns(2)
    with col1:
        plots.profile_chart(st, profile.temperature_k, profile.altitude_km, x_label="Temperature (K)", title="Temperature")
        png = plots.profile_png(profile.temperature_k, profile.altitude_km, x_label="Temperature (K)", title="Temperature")
        _download_row("profile_temperature", png)
    with col2:
        h2o = profile.vmr_ppmv.get(Species.H2O)
        if h2o is not None:
            plots.profile_chart(st, h2o, profile.altitude_km, x_label="H2O (ppmv)", title="Water vapor")
            png = plots.profile_png(h2o, profile.altitude_km, x_label="H2O (ppmv)", title="Water vapor")
            _download_row("profile_h2o", png)


def _band_table_section(result: ScenarioResult) -> None:
    st.subheader("Band summary")
    band_names = list(constants.BANDS_CM1)
    chosen = st.multiselect("Bands", band_names, default=band_names, key="band_selection")
    if not chosen:
        st.info("No bands selected.")
        return
    band_specs = [(name, name) for name in chosen]
    rows = _band_rows(result, band_specs, _Logger(quiet=False, verbose=False))
    if not rows:
        st.info("None of the selected bands overlap the computed grid.")
        return

    trans_df = pd.DataFrame(
        [
            {
                "band": r.label,
                "range (um)": f"{r.lo_um:.3f}-{r.hi_um:.3f}",
                "total": r.transmittance["total"],
                "molecular": r.transmittance["molecular"],
                "aerosol": r.transmittance["aerosol"],
                "rayleigh": r.transmittance["rayleigh"],
                **{f"species: {sp}": v for sp, v in r.species_transmittance.items()},
            }
            for r in rows
        ]
    ).set_index("band")
    st.caption("Band-averaged transmittance")
    st.dataframe(trans_df, width="stretch")

    if any(r.radiance is not None for r in rows):
        radiance_df = pd.DataFrame(
            [{"band": r.label, **(r.radiance or {})} for r in rows if r.radiance is not None]
        ).set_index("band")
        st.caption("Band-integrated radiance (W cm^-2 sr^-1)")
        st.dataframe(radiance_df, width="stretch")

    csv_buf = io.StringIO()
    trans_df.to_csv(csv_buf)
    st.download_button(
        "Download band table (CSV)", data=csv_buf.getvalue(),
        file_name="mwir_band_summary.csv", mime="text/csv", key="download_band_csv",
    )


def _download_row(name: str, png_bytes: Optional[bytes]) -> None:
    if png_bytes is not None:
        st.download_button(
            "Download chart as PNG", data=png_bytes, file_name=f"mwir_{name}.png", mime="image/png",
            key=f"download_png_{name}",
        )
    else:  # pragma: no cover - only reached if matplotlib is unavailable
        st.caption(
            "PNG export needs matplotlib (not available in this environment); use the CSV download "
            "above, or a plotly chart's own camera-icon toolbar button, instead."
        )


def _setup_download(scenario: Scenario) -> None:
    with st.expander("Save setup / use from a script"):
        st.caption("Saves every active simulation input. Load this YAML in the sidebar, CLI or Python; override just the fields you want to vary.")
        st.download_button("Download setup (YAML)",
                           data=yaml.safe_dump(scenario_to_dict(scenario), sort_keys=False),
                           file_name="mwir_setup.yaml", mime="application/yaml", key="download_setup")
        st.code('from mwir_atmos.setups import load_setup, with_overrides\n'
                'from mwir_atmos.api import run_scenario\n\n'
                'base = load_setup("mwir_setup.yaml")\n'
                'trial = with_overrides(base, source_range_km=5.0, source_alt_km=None)\n'
                'result = run_scenario(trial)', language="python")
        st.caption("For repeated runs with a custom detectability score, see the distance-sweep example in README.md.")


def _distance_sweep_section(scenario: Scenario) -> None:
    st.subheader("Distance sweep")
    st.caption("Move the source outward along the saved sensor direction, holding other inputs fixed. Source-altitude limits are cleared. Stop at the first sampled band transmittance below the threshold.")
    st.info("Transmittance is atmospheric throughput, not detector detectability. A Python script can supply its own detectability or SNR metric.")
    with st.form("distance_sweep_controls"):
        c1, c2, c3 = st.columns(3)
        start = c1.number_input("Start distance (km)", min_value=0.001, value=0.1, key="sweep_start")
        stop = c2.number_input("Maximum distance (km)", min_value=0.001, value=10.0, key="sweep_stop")
        step = c3.number_input("Distance step (km)", min_value=0.001, value=1.0, key="sweep_step")
        c1, c2, c3 = st.columns(3)
        lo = c1.number_input("Band minimum (cm^-1)", min_value=0.001,
                             value=float(scenario.wavenumber_min_cm1), key="sweep_band_lo")
        hi = c2.number_input("Band maximum (cm^-1)", min_value=0.001,
                             value=float(scenario.wavenumber_max_cm1), key="sweep_band_hi")
        threshold = c3.number_input("Stop below transmittance", min_value=0.0, max_value=1.0,
                                    value=0.1, step=0.01, format="%.4f", key="sweep_threshold")
        submitted = st.form_submit_button("Run distance sweep")
    signature = (_scenario_cache_key(scenario), start, stop, step, lo, hi, threshold)
    if submitted:
        try:
            if stop < start:
                raise ValueError("Maximum distance must be at least the start distance")
            count = int(np.floor((stop-start)/step))+1
            if count > 200:
                raise ValueError("Use at most 200 distances per sweep; increase the distance step")
            distances = start + np.arange(count)*step
            if not np.isclose(distances[-1], stop):
                distances = np.append(distances, stop)
            if len(distances) > 200:
                raise ValueError("Use at most 200 distances per sweep")
            with_overrides(scenario)  # Validate inputs before sizing a grid.
            if lo >= hi or lo < scenario.wavenumber_min_cm1 or hi > scenario.wavenumber_max_cm1:
                raise ValueError("Choose a band inside the saved spectral window, with minimum below maximum")
            first_index = np.ceil((lo-scenario.wavenumber_min_cm1)/scenario.resolution_cm1)
            last_index = np.floor((hi-scenario.wavenumber_min_cm1)/scenario.resolution_cm1)
            if last_index-first_index < 1:
                raise ValueError("The band must contain at least two computed spectral samples")
            progress = st.progress(0.0, text="Starting distance sweep")
            completed = []
            def update(sample):
                completed.append(sample)
                progress.progress(len(completed)/len(distances), text=f"Computed {sample.distance_km:g} km: transmittance {sample.score:.4f}")
            sweep = sweep_distance(scenario, distances, threshold=threshold,
                                   metric=lambda r: band_average(r.grid, r.transmittance.total, (lo, hi)),
                                   include_radiance=False, on_sample=update)
            progress.empty()
            st.session_state["distance_sweep_result"] = (signature, sweep)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Distance sweep stopped: {exc}")
            return
    stored = st.session_state.get("distance_sweep_result")
    if stored is None or stored[0] != signature:
        return
    sweep = stored[1]
    if sweep.crossed_threshold:
        if sweep.last_passing_distance_km is None:
            st.warning(f"Already below threshold at the first sample ({sweep.first_failing_distance_km:g} km). Try a shorter start distance.")
        else:
            st.success(f"First sampled crossing: {sweep.last_passing_distance_km:g} to {sweep.first_failing_distance_km:g} km. Reduce the step to refine this interval.")
    else:
        st.info("Threshold not crossed within the sampled distances. This does not establish a maximum range.")
    frame = pd.DataFrame({"distance_km": [v.distance_km for v in sweep.samples],
                          "band_transmittance": [v.score for v in sweep.samples]})
    st.line_chart(frame.set_index("distance_km").assign(threshold=threshold))
    st.dataframe(frame, hide_index=True)
    st.download_button("Download distance sweep (CSV)", frame.to_csv(index=False),
                       file_name="mwir_distance_sweep.csv", mime="text/csv", key="download_sweep")


# =========================================================================
# App
# =========================================================================


def main() -> None:
    st.title("MODTRAN-at-home")
    st.caption("MWIR atmospheric transmittance & radiance")
    st.caption(
        "HITRAN line-by-line molecular absorption, AFGL model atmospheres, MT_CKD H2O continuum, "
        "LOWTRAN-style aerosols and Rayleigh scattering. See README.md for the physics basis and limitations."
    )

    scenario = build_scenario_from_sidebar()
    cache_key = _scenario_cache_key(scenario)
    _setup_download(scenario)
    mode = st.radio("Calculation", ["Single calculation", "Distance sweep"], horizontal=True, key="calculation_mode")
    if mode == "Distance sweep":
        _distance_sweep_section(scenario)
        return

    proceed = _cost_and_gate(scenario, cache_key)
    if not proceed:
        st.warning("Waiting for confirmation above before running this scenario.")
        st.stop()

    with st.spinner("Running scenario..."):
        try:
            result = _run_scenario_cached(cache_key)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
            st.error(f"Error running scenario: {exc}")
            st.stop()
            return

    _scenario_summary(result)

    use_wavelength = st.radio(
        "X-axis units", ["Wavelength (um)", "Wavenumber (cm^-1)"], index=0, horizontal=True, key="x_axis_units"
    ) == "Wavelength (um)"
    x, x_label = _x_axis(result.grid, use_wavelength)

    full_csv = io.StringIO()
    _write_csv(result, full_csv)
    st.download_button(
        "Download full spectral data (CSV)", data=full_csv.getvalue(),
        file_name="mwir_spectrum.csv", mime="text/csv", key="download_full_csv",
    )

    _transmittance_section(result, x, x_label)
    _species_section(result, x, x_label)
    _radiance_section(result, x, x_label)
    _turbulence_section(result)
    _profile_section(result)
    _band_table_section(result)


main()
