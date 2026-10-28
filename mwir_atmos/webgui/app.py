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

import streamlit as st

from mwir_atmos import api, run_cost
from mwir_atmos.scenario_io import scenario_from_dict, scenario_to_dict
from mwir_atmos.types import Scenario, ScenarioResult
from mwir_atmos.webgui.results import render_result, setup_download
from mwir_atmos.webgui.sidebar import backend_from_sidebar, build_scenario_from_sidebar
from mwir_atmos.webgui.sweep import distance_sweep_section

st.set_page_config(page_title="mwir_atmos", page_icon=":thermometer:", layout="wide")


def _scenario_cache_key(scenario: Scenario) -> str:
    """A hashable, order-stable stand-in for ``scenario`` (see module docstring)."""
    return json.dumps(scenario_to_dict(scenario), sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=8)
def _run_scenario_cached(cache_key: str, backend: str = "auto") -> ScenarioResult:
    scenario = scenario_from_dict(json.loads(cache_key))
    # Turbulence is computed whenever the sidebar selects a Cn^2 model; with
    # TurbulenceModel.NONE the api layer skips it, so this stays cheap.
    return api.run_scenario(scenario, include_radiance=True, include_turbulence=True,
                            backend=backend)


def _cost_and_gate(scenario: Scenario, cache_key: str, backend: str = "auto") -> bool:
    """Show the cost estimate; large runs need a confirmation tick (as ``mwir --yes``)."""
    try:
        cost = run_cost.estimate_run_cost(scenario, backend=backend)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
        st.error(f"Could not resolve this scenario's geometry: {exc}")
        return False

    c1, c2, c3 = st.columns(3)
    c1.metric("Spectral points x layers", f"{cost.n_points:,} x {cost.n_layers}")
    c2.metric("Estimated layer-array memory", run_cost.format_bytes(cost.est_bytes))
    c3.metric("Estimated runtime", run_cost.format_duration(cost.est_seconds))
    st.caption(
        "Order-of-magnitude estimate, not a scheduler -- actual cost also depends on how many "
        "spectral lines populate the requested window. See README.md 'Choosing spectral resolution'."
    )

    if not run_cost.needs_confirmation(cost):
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

    scenario = build_scenario_from_sidebar()
    backend = backend_from_sidebar()
    cache_key = _scenario_cache_key(scenario)
    setup_download(scenario)
    mode = st.radio("Calculation", ["Single calculation", "Distance sweep"], horizontal=True, key="calculation_mode")
    if mode == "Distance sweep":
        distance_sweep_section(scenario, cache_key, backend)
        return

    if not _cost_and_gate(scenario, cache_key, backend):
        st.warning("Waiting for confirmation above before running this scenario.")
        st.stop()

    with st.spinner("Running scenario..."):
        try:
            result = _run_scenario_cached(cache_key, backend)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
            st.error(f"Error running scenario: {exc}")
            st.stop()
            return

    render_result(result)


main()
