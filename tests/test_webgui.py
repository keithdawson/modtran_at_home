"""Exercise real Streamlit reruns, chart units and invalid-input feedback."""
from pathlib import Path
import json
import base64
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "mwir_atmos/webgui/app.py"


def test_gui_axis_units_controls_and_validation(monkeypatch):
    import streamlit as st
    from mwir_atmos import api
    original = api.run_scenario
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "run_scenario", counted)
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert not app.error
    advanced = next(e for e in app.sidebar.expander if e.label == "Advanced atmosphere overrides")
    assert any(c.key == "ground_temp_k__on" for c in advanced.checkbox)
    charts = [json.loads(c.proto.spec) for c in app.get("plotly_chart")]
    radiance = charts[2]
    def array(data):
        if isinstance(data, dict):
            return np.frombuffer(base64.b64decode(data["bdata"]), dtype=data["dtype"])
        return np.asarray(data)
    wavelength_y = array(radiance["data"][-1]["y"])
    assert radiance["layout"]["yaxis"]["title"]["text"] == "Radiance (W m^-2 sr^-1 / um)"
    app.radio(key="x_axis_units").set_value("Wavenumber (cm^-1)").run()
    assert not app.exception
    assert any(h.value == "Radiance vs wavenumber" for h in app.subheader)
    charts = [json.loads(c.proto.spec) for c in app.get("plotly_chart")]
    assert charts[2]["layout"]["yaxis"]["title"]["text"] == "Radiance (W cm^-2 sr^-1 / cm^-1)"
    trace = charts[2]["data"][-1]
    np.testing.assert_allclose(wavelength_y, array(trace["y"]) * array(trace["x"])**2)
    assert len(calls) == 1, "display changes must reuse the physics result"
    app.number_input(key="wn_min").set_value(4400.0).run()
    assert not app.exception
    assert any("wavenumber_max_cm1" in e.value for e in app.error)


def test_gui_sweep_runs_only_on_submit_and_keeps_results(monkeypatch):
    import streamlit as st
    from mwir_atmos import api
    original = api.run_scenario
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0].source_range_km)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, 'run_scenario', counted)
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert any(e.label == 'Download setup (YAML)' for e in app.get('download_button'))
    previous = len(calls)
    app.radio(key='calculation_mode').set_value('Distance sweep').run()
    assert not app.exception and len(calls) == previous
    app.number_input(key='sweep_stop').set_value(.2)
    app.number_input(key='sweep_step').set_value(.1)
    app.number_input(key='sweep_threshold').set_value(0.)
    next(b for b in app.button if b.label == 'Run distance sweep').click().run()
    assert not app.exception and not app.error
    assert len(calls) == previous+2
    assert len(app.dataframe[0].value) == 2
    assert any('Threshold not crossed' in e.value for e in app.info)
    app.run()
    assert len(calls) == previous+2, 'reruns/downloads must not repeat a sweep'
    app.number_input(key='sweep_stop').set_value(.01)
    next(b for b in app.button if b.label == 'Run distance sweep').click().run()
    assert any('Maximum distance' in e.value for e in app.error)
    assert len(calls) == previous+2


def test_gui_lowtran_selection_and_visibility_bounds():
    from mwir_atmos.types import AerosolModel
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key='aerosol').set_value(AerosolModel.LOWTRAN_MARITIME).run()
    assert not app.exception and not app.error
    assert any('LOWTRAN 7 optical tables' in c.value for c in app.sidebar.caption)
    assert any('LOWTRAN table: Maritime' in m.value for m in app.markdown)
    app.number_input(key='visibility_km').set_value(1.).run()
    assert not app.exception
    assert any('between 2 and 50' in e.value for e in app.error)


def test_gui_disort_solver_controls_and_result_label():
    from mwir_atmos.types import RadianceSolver
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.number_input(key='wn_min').set_value(2500.)
    app.number_input(key='wn_max').set_value(2504.)
    app.selectbox(key='radiance_solver').set_value(RadianceSolver.DISORT).run()
    assert not app.exception and not app.error
    assert app.selectbox(key='disort_streams').value == 32
    assert any('plane-parallel geometry' in m.value for m in app.markdown)
    assert any('Direct discs and target emission are separate' in c.value for c in app.caption)
