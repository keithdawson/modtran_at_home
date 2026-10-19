"""GUI "Distance sweep" mode: move the source outward until band transmittance
drops below a threshold (:func:`mwir_atmos.setups.sweep_distance`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from mwir_atmos.setups import sweep_distance, with_overrides
from mwir_atmos.spectral import band_average
from mwir_atmos.types import Scenario

__all__ = ["distance_sweep_section", "sweep_distances"]

#: Largest number of distances one GUI sweep may run.
_MAX_DISTANCES = 200


def sweep_distances(start: float, stop: float, step: float) -> np.ndarray:
    """``start, start + step, ...`` up to and including ``stop``."""
    if stop < start:
        raise ValueError("Maximum distance must be at least the start distance")
    count = int(np.floor((stop - start) / step)) + 1
    if count > _MAX_DISTANCES:
        raise ValueError(f"Use at most {_MAX_DISTANCES} distances per sweep; increase the distance step")
    distances = start + np.arange(count) * step
    if not np.isclose(distances[-1], stop):
        distances = np.append(distances, stop)
    if len(distances) > _MAX_DISTANCES:
        raise ValueError(f"Use at most {_MAX_DISTANCES} distances per sweep")
    return distances


def _check_band(scenario: Scenario, lo: float, hi: float) -> None:
    """The band must lie inside the scenario's grid and hold two samples."""
    if lo >= hi or lo < scenario.wavenumber_min_cm1 or hi > scenario.wavenumber_max_cm1:
        raise ValueError("Choose a band inside the saved spectral window, with minimum below maximum")
    first_index = np.ceil((lo - scenario.wavenumber_min_cm1) / scenario.resolution_cm1)
    last_index = np.floor((hi - scenario.wavenumber_min_cm1) / scenario.resolution_cm1)
    if last_index - first_index < 1:
        raise ValueError("The band must contain at least two computed spectral samples")


def distance_sweep_section(scenario: Scenario, cache_key: str) -> None:
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

    # Results persist across reruns until any input changes.
    signature = (cache_key, start, stop, step, lo, hi, threshold)
    if submitted:
        try:
            distances = sweep_distances(start, stop, step)
            with_overrides(scenario)  # Validate inputs before sizing a grid.
            _check_band(scenario, lo, hi)
            sweep = _run_with_progress(scenario, distances, lo, hi, threshold)
            st.session_state["distance_sweep_result"] = (signature, sweep)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Distance sweep stopped: {exc}")
            return
    stored = st.session_state.get("distance_sweep_result")
    if stored is None or stored[0] != signature:
        return
    _show_sweep(stored[1], threshold)


def _run_with_progress(scenario: Scenario, distances: np.ndarray, lo: float, hi: float,
                       threshold: float):
    progress = st.progress(0.0, text="Starting distance sweep")
    completed = []

    def update(sample):
        completed.append(sample)
        progress.progress(len(completed) / len(distances),
                          text=f"Computed {sample.distance_km:g} km: transmittance {sample.score:.4f}")

    sweep = sweep_distance(scenario, distances, threshold=threshold,
                           metric=lambda r: band_average(r.grid, r.transmittance.total, (lo, hi)),
                           include_radiance=False, on_sample=update)
    progress.empty()
    return sweep


def _show_sweep(sweep, threshold: float) -> None:
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
