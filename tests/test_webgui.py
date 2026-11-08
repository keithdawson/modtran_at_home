"""Exercise real Streamlit reruns, chart units and invalid-input feedback."""
from pathlib import Path
import json
import base64
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "mwir_atmos/webgui/app.py"


def _point_app():
    """Load the app in point-sampling mode: the calculation starts when Run calculation is clicked."""
    from mwir_atmos.types import SpectralSampling
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state["spectral_sampling"] = SpectralSampling.POINT
    app.run()
    run_btn = [b for b in app.button if b.label == "Run calculation"]
    if run_btn:
        run_btn[0].click().run()
    return app


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
    app = _point_app()
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
    app.sidebar.radio(key="spectral_unit").set_value("Wavenumber (cm^-1)").run()
    app.number_input(key="wn_min").set_value(4400.0).run()
    assert not app.exception
    assert any("wavenumber_max_cm1" in e.value for e in app.error)


def test_gui_changing_settings_does_not_run_until_button_clicked(monkeypatch):
    import streamlit as st
    from mwir_atmos import api
    from mwir_atmos.types import AerosolModel
    calls = []
    original = api.run_scenario
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "run_scenario", counted)
    st.cache_data.clear()
    app = _point_app()
    assert len(calls) == 1  # initial run triggered by clicking Run in _point_app()

    # Changing sidebar options must NOT trigger physics runs
    app.selectbox(key='aerosol').set_value(AerosolModel.URBAN).run()
    app.number_input(key='visibility_km').set_value(15.0).run()
    assert len(calls) == 1, "Modifying settings without clicking Run calculation must not run physics"

    # Clicking Run calculation must trigger the new physics run
    next(b for b in app.button if b.label == "Run calculation").click().run()
    assert len(calls) == 2, "Clicking Run calculation must execute the run"


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
    app = _point_app()
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
    app = _point_app()
    app.selectbox(key='aerosol').set_value(AerosolModel.LOWTRAN_MARITIME).run()
    assert not app.exception and not app.error
    assert any('LOWTRAN 7 optical tables' in c.value for c in app.sidebar.caption)
    next(b for b in app.button if b.label == 'Run calculation').click().run()
    assert any('LOWTRAN table: Maritime' in m.value for m in app.markdown)
    app.number_input(key='visibility_km').set_value(1.).run()
    assert not app.exception
    assert any('between 2 and 50' in e.value for e in app.error)


def test_gui_disort_solver_controls_and_result_label():
    from mwir_atmos.types import RadianceSolver
    app = _point_app()
    app.sidebar.radio(key="spectral_unit").set_value("Wavenumber (cm^-1)").run()
    app.number_input(key='wn_min').set_value(2500.)
    app.number_input(key='wn_max').set_value(2504.)
    app.selectbox(key='radiance_solver').set_value(RadianceSolver.DISORT).run()
    assert not app.exception and not app.error
    assert app.selectbox(key='disort_streams').value == 32
    next(b for b in app.button if b.label == 'Run calculation').click().run()
    assert any('plane-parallel geometry' in m.value for m in app.markdown)
    assert any('Direct discs and target emission are separate' in c.value for c in app.caption)


def test_gui_spherical_solver_and_sampling_uncertainty():
    from mwir_atmos.types import RadianceSolver
    app = _point_app()
    app.sidebar.radio(key="spectral_unit").set_value("Wavenumber (cm^-1)").run()
    app.number_input(key='wn_min').set_value(2500.)
    app.number_input(key='wn_max').set_value(2504.)
    app.selectbox(key='radiance_solver').set_value(RadianceSolver.SPHERICAL_MC).run()
    assert not app.exception and not app.error
    next(b for b in app.button if b.label == 'Run calculation').click().run()
    assert any('Spherical multiple scattering' in m.value for m in app.markdown)
    assert any('sampling uncertainty' in c.value for c in app.caption)


def test_gui_line_resolved_sampling_controls_and_summary():
    from mwir_atmos.types import InstrumentFunction, SpectralSampling
    app = _point_app()
    app.sidebar.radio(key="spectral_unit").set_value("Wavenumber (cm^-1)").run()
    app.number_input(key='wn_min').set_value(2500.)
    app.number_input(key='wn_max').set_value(2504.)
    app.selectbox(key='spectral_sampling').set_value(SpectralSampling.RESOLVED).run()
    assert not app.exception and not app.error
    assert app.selectbox(key='instrument_function').value == InstrumentFunction.BOX
    next(b for b in app.button if b.label == 'Run calculation').click().run()
    assert any('line-resolved at' in m.value for m in app.markdown)


def test_gui_saved_yaml_toggle_preserves_dropdown_values():
    from mwir_atmos.types import AerosolModel, ModelAtmosphere, SurfaceType
    app = _point_app()
    app.selectbox(key="atmosphere").set_value(ModelAtmosphere.SUBARCTIC_WINTER).run()
    app.selectbox(key="aerosol").set_value(AerosolModel.URBAN).run()
    app.selectbox(key="surface_type").set_value(SurfaceType.SNOW).run()

    # Switch to Saved YAML
    app.radio(key="setup_source").set_value("Saved YAML").run()
    assert not app.exception and not app.error
    assert app.selectbox(key="atmosphere").value == ModelAtmosphere.SUBARCTIC_WINTER
    assert app.selectbox(key="aerosol").value == AerosolModel.URBAN
    assert app.selectbox(key="surface_type").value == SurfaceType.SNOW

    # Switch back to Current controls
    app.radio(key="setup_source").set_value("Current controls").run()
    assert not app.exception and not app.error
    assert app.selectbox(key="atmosphere").value == ModelAtmosphere.SUBARCTIC_WINTER
    assert app.selectbox(key="aerosol").value == AerosolModel.URBAN
    assert app.selectbox(key="surface_type").value == SurfaceType.SNOW


def test_gui_upload_yaml_populates_dropdowns():
    from mwir_atmos.types import AerosolModel, ModelAtmosphere, SurfaceType
    app = _point_app()
    app.radio(key="setup_source").set_value("Saved YAML").run()
    yaml_path = Path(__file__).resolve().parents[1] / "scenarios/tropical_maritime_quicklook.yaml"
    app.file_uploader(key="setup_upload").upload(yaml_path.name, yaml_path.read_bytes()).run()
    assert app.selectbox(key="atmosphere").value == ModelAtmosphere.TROPICAL
    assert app.selectbox(key="surface_type").value == SurfaceType.SEA_WATER
    assert app.selectbox(key="aerosol").value == AerosolModel.NAVY_MARITIME
    assert app.number_input(key="visibility_km").value == 10.0
    assert app.number_input(key="wind_speed_m_s").value == 8.0
    assert app.number_input(key="sensor_alt_km").value == 0.02
    assert app.number_input(key="sensor_zenith_deg").value == 100.0


def test_gui_source_range_override_displays_intensity_remaining():
    app = _point_app()
    app.checkbox(key="source_range_km__on").check().run()
    app.number_input(key="source_range_km").set_value(5.0).run()
    next(b for b in app.button if b.label == "Run calculation").click().run()
    assert not app.exception and not app.error
    assert any(h.value == "Source Target Attenuation" for h in app.subheader)
    assert any("source intensity remains" in m.value for m in app.markdown)


def test_gui_compute_backend_selection():
    app = _point_app()
    radio_keys = [r.key for r in app.sidebar.radio]
    if "compute_backend" in radio_keys:
        assert app.sidebar.radio(key="compute_backend").value in ("cuda", "cpu")
        app.sidebar.radio(key="compute_backend").set_value("cpu").run()
        assert not app.exception and not app.error
        assert app.sidebar.radio(key="compute_backend").value == "cpu"


def test_sweep_distances_validation():

    from mwir_atmos.webgui.sweep import sweep_distances
    with pytest.raises(ValueError, match="Start distance must be greater than zero"):
        sweep_distances(0.0, 10.0, 1.0)
    with pytest.raises(ValueError, match="Distance step must be greater than zero"):
        sweep_distances(1.0, 10.0, 0.0)
    dists = sweep_distances(0.1, 1.0, 0.5)
    assert dists[0] == pytest.approx(0.1)
    assert dists[-1] == pytest.approx(1.0)


def test_gui_number_inputs_step_and_min_alignment():
    app = _point_app()
    assert not app.exception and not app.error

    # Verify that all sidebar number inputs have min_value aligned with step
    for ni in app.number_input:
        if ni.min is not None and ni.step is not None:
            val = ni.value
            offset = val - ni.min
            rem = offset % ni.step
            # Check alignment within floating point roundoff
            aligned = np.isclose(rem, 0.0, atol=1e-5) or np.isclose(rem, ni.step, atol=1e-5)
            assert aligned, f"Widget key '{ni.key}' value {val} (min={ni.min}, step={ni.step}) is not aligned: rem={rem}"


def test_gui_sweep_wavelength_inputs():
    app = _point_app()
    app.radio(key="calculation_mode").set_value("Distance sweep").run()
    assert not app.exception and not app.error

    lo_um = app.number_input(key="sweep_band_lo_um")
    hi_um = app.number_input(key="sweep_band_hi_um")
    assert lo_um is not None and hi_um is not None
    assert pytest.approx(lo_um.value, abs=1e-3) == 2.326
    assert pytest.approx(hi_um.value, abs=1e-3) == 5.556

    lo_um.set_value(3.0)
    hi_um.set_value(4.0)
    app.number_input(key="sweep_stop").set_value(0.2)
    app.number_input(key="sweep_step").set_value(0.1)
    next(b for b in app.button if b.label == "Run distance sweep").click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 2


def test_gui_sidebar_spectral_units_toggle():
    app = _point_app()
    assert not app.exception and not app.error

    # Verify defaults to Wavelength (um)
    assert app.sidebar.radio(key="spectral_unit").value == "Wavelength (um)"
    wl_min = app.sidebar.number_input(key="wl_min")
    wl_max = app.sidebar.number_input(key="wl_max")
    assert pytest.approx(wl_min.value, abs=1e-3) == 2.326
    assert pytest.approx(wl_max.value, abs=1e-3) == 5.556

    # Edit wavelength values and run
    wl_min.set_value(3.0)
    wl_max.set_value(5.0)
    next(b for b in app.button if b.label == "Run calculation").click().run()
    assert not app.exception and not app.error

    # Switch view to Wavenumber (cm^-1)
    app.sidebar.radio(key="spectral_unit").set_value("Wavenumber (cm^-1)").run()
    assert not app.exception and not app.error
    wn_min = app.sidebar.number_input(key="wn_min")
    wn_max = app.sidebar.number_input(key="wn_max")
    assert pytest.approx(wn_min.value, abs=0.1) == 2000.0
    assert pytest.approx(wn_max.value, abs=0.1) == 3333.33


def test_gui_backend_control(monkeypatch):
    """The Performance expander selects the compute backend; asking for the
    GPU without one is reported, not run on the CPU behind the user's back."""
    import streamlit as st
    from mwir_atmos import api
    from mwir_atmos.optics import lbl_cuda
    monkeypatch.setattr(lbl_cuda, "available", lambda: False)
    seen = []
    original = api.run_scenario
    def recorded(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "run_scenario", recorded)
    st.cache_data.clear()
    app = _point_app()
    assert not app.exception
    performance = next(e for e in app.sidebar.expander if e.label == "Performance")
    assert any("Auto runs on the CPU" in c.value for c in performance.caption)
    app.selectbox(key="compute_backend").set_value("cpu").run()
    assert not app.exception
    assert seen == ["auto", "cpu"]
    app.selectbox(key="compute_backend").set_value("cuda").run()
    assert any("not available" in e.value for e in app.error)



