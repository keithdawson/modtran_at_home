"""The web GUI's plotting layer.

These exercise the pure rendering functions without a Streamlit runtime: the
PNG exporters are ordinary matplotlib, and the on-screen chart functions take a
container, so a recording stub is enough to check what they emit.

Nothing here runs the physics, so the file is fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.webgui import plots

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def xy():
    x = np.linspace(3.0, 5.0, 64)
    return x, {"total": np.linspace(0.9, 0.4, 64), "molecular": np.linspace(0.95, 0.5, 64)}


class RecordingContainer:
    """Stands in for the Streamlit container the chart helpers draw into."""

    def __init__(self):
        self.figures = []
        self.frames = []
        self.messages = []

    def plotly_chart(self, fig, **kwargs):
        self.figures.append(fig)

    def pyplot(self, fig, **kwargs):
        self.figures.append(fig)

    def line_chart(self, df, **kwargs):
        self.frames.append(df)

    def area_chart(self, df, **kwargs):
        self.frames.append(df)

    def info(self, msg):
        self.messages.append(msg)

    @property
    def drew_something(self) -> bool:
        return bool(self.figures or self.frames)


# --------------------------------------------------------------------------
# Backend detection
# --------------------------------------------------------------------------


def test_backend_flags_match_what_is_actually_importable():
    try:
        import plotly.graph_objects  # noqa: F401
        plotly_available = True
    except ImportError:
        plotly_available = False
    try:
        import matplotlib  # noqa: F401
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False

    assert plots.HAS_PLOTLY is plotly_available
    assert plots.HAS_MATPLOTLIB is matplotlib_available


def test_at_least_one_rendering_backend_is_present():
    """With neither, every chart silently degrades to a bare dataframe."""
    assert plots.HAS_PLOTLY or plots.HAS_MATPLOTLIB


# --------------------------------------------------------------------------
# Colour assignment
# --------------------------------------------------------------------------


def test_colour_assignment_is_stable_across_calls():
    """A chart must not change colours between runs of the same app."""
    first = [plots._color_for(i) for i in range(12)]
    second = [plots._color_for(i) for i in range(12)]
    assert first == second


def test_palette_overflows_to_a_neutral_rather_than_recycling_a_hue():
    """Past the 8 categorical slots the palette folds to one neutral.

    Deliberate: recycling hue 0 would make two different series identical, and
    generating a 9th hue produces something indistinguishable from an existing
    one under colour-vision deficiency. A single obviously-neutral grey is the
    honest signal that the chart has more series than the palette can name.
    """
    n = len(plots.CATEGORICAL_COLORS)
    assert plots._color_for(n) == plots.OVERFLOW_COLOR
    assert plots._color_for(n + 3) == plots.OVERFLOW_COLOR
    assert plots.OVERFLOW_COLOR not in plots.CATEGORICAL_COLORS


def test_negative_index_also_folds_to_the_overflow_colour():
    """Guards against a negative index silently wrapping to the last hue."""
    assert plots._color_for(-1) == plots.OVERFLOW_COLOR


def test_early_colours_are_distinct():
    n = min(6, len(plots.CATEGORICAL_COLORS))
    assert len({plots._color_for(i) for i in range(n)}) == n


def test_total_line_colour_is_not_a_series_colour():
    """The dotted 'total' overlay must not be confusable with a component."""
    assert plots.TOTAL_LINE_COLOR not in plots.CATEGORICAL_COLORS


# --------------------------------------------------------------------------
# PNG exporters
# --------------------------------------------------------------------------


def test_line_chart_png_is_a_real_png(xy):
    x, series = xy
    data = plots.line_chart_png(x, series, x_label="Wavelength (um)", y_label="Transmittance")
    assert data is not None and data.startswith(PNG_MAGIC) and len(data) > 1000


def test_line_chart_png_with_no_series_returns_none(xy):
    x, _ = xy
    assert plots.line_chart_png(x, {}, x_label="x", y_label="y") is None


def test_stacked_area_png_is_a_real_png(xy):
    x, series = xy
    data = plots.stacked_area_png(
        x, series, total=series["total"], x_label="Wavelength (um)", y_label="Radiance"
    )
    assert data is not None and data.startswith(PNG_MAGIC)


def test_stacked_area_png_with_no_components_returns_none(xy):
    x, _ = xy
    assert plots.stacked_area_png(x, {}, x_label="x", y_label="y") is None


def test_stacked_area_png_drops_all_zero_components(xy):
    """An all-zero component would render as an invisible legend entry."""
    x, _ = xy
    only_zeros = {"a": np.zeros_like(x), "b": np.zeros_like(x)}
    assert plots.stacked_area_png(x, only_zeros, x_label="x", y_label="y") is None


def test_profile_png_is_a_real_png():
    value = np.linspace(300.0, 220.0, 40)
    altitude = np.linspace(0.0, 30.0, 40)
    data = plots.profile_png(value, altitude, x_label="Temperature (K)")
    assert data is not None and data.startswith(PNG_MAGIC)


# --------------------------------------------------------------------------
# Log axis -- Cn^2 spans ~30 decades, so this is not cosmetic
# --------------------------------------------------------------------------


def test_profile_png_log_axis_renders():
    value = np.logspace(-14, -20, 40)
    altitude = np.linspace(0.0, 30.0, 40)
    data = plots.profile_png(value, altitude, x_label="Cn2", log_x=True)
    assert data is not None and data.startswith(PNG_MAGIC)


def test_log_axis_drops_non_positive_values_without_crashing():
    value = np.array([1e-14, 0.0, 1e-16, -1e-18, 1e-20])
    altitude = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert plots.profile_png(value, altitude, x_label="Cn2", log_x=True) is not None


def test_log_axis_with_nothing_positive_returns_none_rather_than_raising():
    value = np.array([0.0, -1.0, -2.0])
    altitude = np.array([0.0, 1.0, 2.0])
    assert plots.profile_png(value, altitude, x_label="Cn2", log_x=True) is None


def test_screen_and_download_agree_on_log_filtering():
    """A downloaded chart that differs from the on-screen one is a bug."""
    value = np.array([1e-14, 0.0, 1e-16, -5.0, 1e-20])
    altitude = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    container = RecordingContainer()
    plots.profile_chart(container, value, altitude, x_label="Cn2", log_x=True)
    assert container.drew_something

    assert plots.profile_png(value, altitude, x_label="Cn2", log_x=True) is not None

    empty = np.array([0.0, -1.0])
    alt2 = np.array([0.0, 1.0])
    container2 = RecordingContainer()
    plots.profile_chart(container2, empty, alt2, x_label="Cn2", log_x=True)
    assert not container2.drew_something and container2.messages
    assert plots.profile_png(empty, alt2, x_label="Cn2", log_x=True) is None


def test_linear_axis_keeps_zeros_that_the_log_axis_drops():
    value = np.array([1.0, 0.0, 2.0])
    altitude = np.array([0.0, 1.0, 2.0])
    linear = RecordingContainer()
    plots.profile_chart(linear, value, altitude, x_label="v", log_x=False)
    assert linear.drew_something


# --------------------------------------------------------------------------
# On-screen chart helpers
# --------------------------------------------------------------------------


def test_profile_chart_draws_into_the_container():
    container = RecordingContainer()
    plots.profile_chart(
        container, np.linspace(300.0, 220.0, 20), np.linspace(0.0, 30.0, 20), x_label="T (K)"
    )
    assert container.drew_something


def test_chart_helpers_tolerate_a_single_point():
    """Degenerate inputs must not raise -- a one-layer path is legal."""
    container = RecordingContainer()
    plots.profile_chart(container, np.array([1.0]), np.array([0.0]), x_label="v")
    assert container.drew_something


def test_png_exporters_are_deterministic():
    """Same input, same bytes -- so a download is reproducible."""
    value = np.linspace(300.0, 220.0, 30)
    altitude = np.linspace(0.0, 30.0, 30)
    a = plots.profile_png(value, altitude, x_label="T (K)", title="Temperature")
    b = plots.profile_png(value, altitude, x_label="T (K)", title="Temperature")
    assert a == b


def test_exporters_do_not_mutate_their_inputs():
    value = np.logspace(-14, -20, 20)
    altitude = np.linspace(0.0, 30.0, 20)
    value_before, altitude_before = value.copy(), altitude.copy()
    plots.profile_png(value, altitude, x_label="Cn2", log_x=True)
    np.testing.assert_array_equal(value, value_before)
    np.testing.assert_array_equal(altitude, altitude_before)
