"""Streamlit GUI: mid-wave infrared atmospheric transmittance and radiance.

Run with::

    streamlit run mwir_atmos/webgui/app.py

Layout of this package:

* :mod:`~mwir_atmos.webgui.sidebar` -- widgets -> :class:`~mwir_atmos.types.Scenario`
* :mod:`~mwir_atmos.webgui.results` -- the main panel for one computed result
* :mod:`~mwir_atmos.webgui.sweep` -- the "Distance sweep" mode
* :mod:`~mwir_atmos.webgui.plots` -- plotly charts and matplotlib PNG export
* this file -- page setup, the result cache, the cost gate, and the order of it all

Caching
-------
``st.cache_data`` needs a hashable key, and Streamlit's default hasher does not
treat a dataclass of enums and an ``Optional[datetime]`` predictably across
versions. So the cache is keyed on the scenario's JSON string, built with the
same serialisation the CLI uses for ``--config``
(:func:`mwir_atmos.scenario_io.scenario_to_dict`). Each distinct scenario runs
:func:`mwir_atmos.api.run_scenario` once.

Import style
------------
Streamlit execs this file directly, not as a package submodule, so relative
imports (``from .. import x``) fail here. Every import is absolute, after a
small ``sys.path`` bootstrap that makes ``mwir_atmos`` importable even when
the project root is not on ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import os
import time
from datetime import datetime

# Set parallel execution to threads in Streamlit to avoid multi-process spawn bombs
os.environ.setdefault("MWIR_PARALLEL", "threads")

import streamlit as st

from mwir_atmos import api, constants, report, run_cost
from mwir_atmos.optics import lbl_cuda
from mwir_atmos.scenario_io import scenario_from_dict, scenario_to_dict
from mwir_atmos.setups import with_overrides
from mwir_atmos.source_attenuation import choose_reference_range_km
from mwir_atmos.types import Scenario, ScenarioResult
from mwir_atmos.webgui.results import render_result, setup_download
from mwir_atmos.webgui.sidebar import backend_from_sidebar, build_scenario_from_sidebar
from mwir_atmos.webgui.sweep import distance_sweep_section

try:
    CUDA_AVAILABLE = lbl_cuda.available()
    CUDA_DEVICE_INFO = lbl_cuda.device_summary() if CUDA_AVAILABLE else None
except Exception:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_INFO = None

# If CUDA is available, default to GPU backend unless explicitly overridden in env
if CUDA_AVAILABLE and not os.environ.get("MWIR_BACKEND"):
    os.environ["MWIR_BACKEND"] = "cuda"

st.set_page_config(page_title="mwir_atmos", page_icon=":thermometer:", layout="wide")

# Expand the sidebar / input column to take up more of the screen
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 540px !important;
            max-width: 720px !important;
            width: 600px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _scenario_cache_key(scenario: Scenario, backend: str = "cuda") -> str:
    """A hashable, order-stable stand-in for ``scenario`` and compute backend."""
    data = scenario_to_dict(scenario)
    data["_backend"] = backend
    return json.dumps(data, sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=8)
def _run_scenario_cached(cache_key: str) -> ScenarioResult:
    data = json.loads(cache_key)
    backend = data.pop("_backend", "cuda" if CUDA_AVAILABLE else "cpu")
    scenario = scenario_from_dict(data)
    # Turbulence is computed whenever the sidebar selects a Cn^2 model; with
    # TurbulenceModel.NONE the api layer skips it, so this stays cheap.
    return api.run_scenario(scenario, include_radiance=True, include_turbulence=True, backend=backend)


def _cost_and_gate(scenario: Scenario, cache_key: str, backend: str = "cpu") -> bool:
    """Show the cost estimate; large runs need a confirmation tick (as ``mwir --yes``)."""
    try:
        cost = run_cost.estimate_run_cost(scenario, backend=backend)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
        st.error(f"Could not resolve this scenario's geometry: {exc}")
        return False

    c1, c2, c3 = st.columns(3)
    c1.metric("Spectral points x layers", f"{cost.n_points:,} x {cost.n_layers}")
    c2.metric("Estimated layer-array memory", run_cost.format_bytes(cost.est_bytes))
    if backend == "cuda":
        est_gpu_s = max(0.2, cost.est_seconds / 40.0)
        c3.metric("Estimated runtime", f"~{est_gpu_s:.1f}s (⚡ GPU)", help="RTX 4090 executes ~40x faster than CPU")
    else:
        c3.metric("Estimated runtime", run_cost.format_duration(cost.est_seconds))
    st.caption(
        "Order-of-magnitude estimate, not a scheduler -- actual cost also depends on how many "
        "spectral lines populate the requested window. See README.md 'Choosing spectral resolution'."
    )

    if backend == "cuda" or not run_cost.needs_confirmation(cost):
        return True
    return st.checkbox(
        f"This run is estimated at ~{run_cost.format_bytes(cost.est_bytes)} / "
        f"~{run_cost.format_duration(cost.est_seconds)}, above the usual "
        f"~{run_cost.format_bytes(run_cost.CONFIRM_BYTES_THRESHOLD)} / "
        f"~{run_cost.format_duration(run_cost.CONFIRM_SECONDS_THRESHOLD)} guideline. Run it anyway?",
        key=f"confirm_large_run::{cache_key}",
    )


def main() -> None:
    st.title("MODTRAN-at-home")
    st.caption("MWIR atmospheric transmittance & radiance")
    st.caption(
        "HITRAN line-by-line molecular absorption, AFGL model atmospheres, MT_CKD H2O continuum, "
        "LOWTRAN-style aerosols and Rayleigh scattering. See README.md for the physics basis and limitations."
    )

    # Hardware compute backend
    backend_choice = backend_from_sidebar()
    os.environ["MWIR_BACKEND"] = backend_choice

    scenario = build_scenario_from_sidebar()
    cache_key = _scenario_cache_key(scenario, backend=backend_choice)
    setup_download(scenario)
    mode = st.radio("Calculation", ["Single calculation", "Distance sweep"], horizontal=True, key="calculation_mode")
    if mode == "Distance sweep":
        distance_sweep_section(scenario, cache_key, backend=backend_choice)
        return

    gate_ok = _cost_and_gate(scenario, cache_key, backend=backend_choice)

    # Primary run buttons: both at the top of the sidebar and in the main area
    run_sidebar = st.sidebar.button(
        "Run calculation", type="primary", key="run_calc_sidebar",
        disabled=not gate_ok, use_container_width=True,
    )
    if st.sidebar.button("Clear cache & re-run", key="btn_clear_cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state["current_result"] = None
        st.session_state["current_result_key"] = None
        st.rerun()

    sidebar_indicator = st.sidebar.empty()

    run_main = st.button(
        "Run calculation", type="primary", key="run_calc_main",
        disabled=not gate_ok,
    )
    run_clicked = run_sidebar or run_main

    # Track computed calculation state in session state
    st.session_state.setdefault("current_result", None)
    st.session_state.setdefault("current_result_key", None)

    if run_clicked:
        if not gate_ok:
            st.warning("Waiting for confirmation above before running this scenario.")
            st.stop()
        t0 = time.time()
        dev_label = "NVIDIA GeForce RTX 4090" if backend_choice == "cuda" else "CPU multi-threaded"

        spinner_html = f"""
        <div style="display: flex; align-items: center; gap: 14px; padding: 14px 18px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.3); border-left: 5px solid #2563eb; border-radius: 8px; margin: 16px 0;">
            <div style="border: 4px solid rgba(37, 99, 235, 0.2); border-top: 4px solid #2563eb; border-radius: 50%; width: 28px; height: 28px; animation: mwir-spin 0.75s linear infinite; flex-shrink: 0;"></div>
            <div>
                <div style="font-size: 15px; font-weight: 600; color: #1e3a8a;">Running scenario calculation...</div>
                <div style="font-size: 13px; color: #3b82f6;">Line-by-line Voigt engine active on {dev_label}</div>
            </div>
        </div>
        <style>
        @keyframes mwir-spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        sidebar_spinner_html = f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: rgba(59, 130, 246, 0.08); border-radius: 6px; margin: 6px 0;">
            <div style="border: 2px solid rgba(37, 99, 235, 0.2); border-top: 2px solid #2563eb; border-radius: 50%; width: 16px; height: 16px; animation: mwir-spin 0.75s linear infinite; flex-shrink: 0;"></div>
            <span style="font-size: 13px; font-weight: 600; color: #1e3a8a;">Calculating on {backend_choice.upper()}...</span>
        </div>
        """
        main_status = st.empty()
        main_status.markdown(spinner_html, unsafe_allow_html=True)
        sidebar_indicator.markdown(sidebar_spinner_html, unsafe_allow_html=True)
        time.sleep(0.05)  # flush websocket frame to browser so animated circle appears immediately
        try:
            result = _run_scenario_cached(cache_key)
            if result.scenario.source_range_km is not None and result.scenario.source_range_km > 0.0:
                ref_km = choose_reference_range_km(result.scenario.source_range_km)
                sc_ref = with_overrides(result.scenario, source_range_km=ref_km, source_alt_km=None)
                _ = _run_scenario_cached(_scenario_cache_key(sc_ref, backend=backend_choice))
            st.session_state["current_result"] = result
            st.session_state["current_result_key"] = cache_key
            st.session_state["last_run_time"] = datetime.now().strftime("%I:%M:%S %p")
            dur = time.time() - t0
            st.session_state["last_run_duration"] = dur
            st.session_state["last_run_backend"] = backend_choice
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
            sidebar_indicator.error("❌ Calculation failed")
            main_status.error(f"Error running scenario: {exc}")
            st.stop()
            return
        finally:
            main_status.empty()

    last_time = st.session_state.get("last_run_time")
    last_dur = st.session_state.get("last_run_duration")
    last_backend = st.session_state.get("last_run_backend", "cpu")
    if last_time is not None:
        tag = "⚡ GPU (RTX 4090)" if last_backend == "cuda" else "💻 CPU"
        sidebar_indicator.success(f"✔ Done at {last_time} ({last_dur:.2f}s | {tag})")
        st.success(f"✔ Calculation completed at {last_time} (elapsed: {last_dur:.2f}s using {tag})")

    current_result = st.session_state.get("current_result")
    current_key = st.session_state.get("current_result_key")

    if current_result is None:
        st.info("👈 Configure scenario parameters in the sidebar and click **Run calculation** to begin.")
        st.stop()

    if current_key != cache_key:
        st.warning("⚠️ Sidebar settings have changed since the last calculation. Click **Run calculation** to update.")

    def _run_scenario_for_scenario(sc: Scenario) -> ScenarioResult:
        return _run_scenario_cached(_scenario_cache_key(sc, backend=backend_choice))

    render_result(current_result, run_fn=_run_scenario_for_scenario)


if __name__ == "__main__":
    main()
