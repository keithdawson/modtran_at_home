"""Chart rendering for the Streamlit GUI.

Every function here takes already-computed numpy arrays and a Streamlit
container to draw into -- none of them import :mod:`mwir_atmos.api` or touch
a physics module.

Backend selection (checked at import time, never assumed): plotly if
installed (richer interactivity -- zoom, per-trace legend toggling, unified
hover), else matplotlib, else :func:`streamlit.line_chart`/``area_chart``.
Both optional libraries are already declared in ``pyproject.toml``'s ``gui``
extra (plotly) or happened to be present in this environment (matplotlib);
neither is a new dependency this module introduces.

Color: fixed-order categorical palette from the ``dataviz`` skill's
reference instance (``references/palette.md``) -- eight hues validated for
adjacent-pair colorblind separation in both light and dark modes, assigned
to series in a stable order (never cycled/reassigned per scenario) so a
given transmittance component, species, or radiance term keeps the same
color everywhere it appears. A stacked chart is an "adjacent pairs" context
per that skill (only neighbouring layers touch), so the full eight slots are
usable here, not the stricter three-slot "all-pairs" cap that applies to
scatter/small-multiples forms.
"""

from __future__ import annotations

import io
from typing import Mapping, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - exercised only if plotly is absent
    HAS_PLOTLY = False

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs inside Streamlit's server process
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - exercised only if matplotlib is absent
    HAS_MATPLOTLIB = False

__all__ = [
    "HAS_PLOTLY",
    "HAS_MATPLOTLIB",
    "CATEGORICAL_COLORS",
    "TOTAL_LINE_COLOR",
    "line_chart",
    "stacked_area_chart",
    "profile_chart",
    "line_chart_png",
    "stacked_area_png",
    "profile_png",
]

#: Fixed-order categorical palette: blue, orange, aqua, yellow, magenta,
#: green, violet, red (dataviz skill `references/palette.md`). Assigned to
#: series by position -- see module docstring.
CATEGORICAL_COLORS: tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
#: Overlay color for a "total"/sum line drawn over a stack or multi-line
#: chart -- not a categorical slot, since a sum is not a series identity.
TOTAL_LINE_COLOR = "#52514e"
#: A series beyond the 8-color ceiling folds to this neutral rather than a
#: generated 9th hue (dataviz skill: a generated hue is indistinguishable
#: from an existing one under CVD). In practice this should not trigger for
#: the charts in this module -- transmittance has 4 components, per-species
#: breakdown has 6, and the radiance stack filters to non-zero components
#: first, which keeps real scenarios well under 8 -- but it is here as an
#: honest fallback rather than a silent index error if it ever does.
OVERFLOW_COLOR = "#898781"

_LEGEND_HORIZONTAL = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)


def _color_for(index: int) -> str:
    if 0 <= index < len(CATEGORICAL_COLORS):
        return CATEGORICAL_COLORS[index]
    return OVERFLOW_COLOR


_SERIES_COLORS = {
    name: CATEGORICAL_COLORS[i]
    for names in (
        ("total", "molecular", "aerosol", "rayleigh"),
        ("H2O", "CO2", "O3", "N2O", "CO", "CH4"),
        ("path_thermal", "surface_emitted", "surface_reflected_solar", "surface_reflected_lunar",
         "solar_single_scatter", "lunar_single_scatter", "direct_solar", "direct_lunar"),
    )
    for i, name in enumerate(names)
}
_SERIES_COLORS["source_emitted"] = OVERFLOW_COLOR


def _series_color(name: str, fallback_index: int) -> str:
    """Filtering or reordering a known physical component preserves its color."""
    return _SERIES_COLORS.get(name, _color_for(fallback_index))


def _mpl_legend(ax, n_entries: int) -> None:
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(max(n_entries, 1), 4), frameon=False
    )


# =========================================================================
# On-screen charts
# =========================================================================


def line_chart(
    container,
    x: np.ndarray,
    series: Mapping[str, np.ndarray],
    *,
    x_label: str,
    y_label: str,
    title: str = "",
    log_y: bool = False,
    height: int = 420,
) -> None:
    """Multi-line chart -- one line per ``series`` entry, in insertion order.

    Used for transmittance-vs-wavelength (total/molecular/aerosol/rayleigh)
    and the per-species breakdown. "Tell distinct series apart" is a
    categorical job (dataviz skill `references/choosing-a-form.md`), hence
    fixed-order color, a horizontal legend, and 2px lines throughout.
    """
    if not series:
        container.info("Nothing selected to plot.")
        return

    if HAS_PLOTLY:
        fig = go.Figure()
        for i, (name, y) in enumerate(series.items()):
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=_series_color(name, i), width=2))
            )
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=height,
            margin=dict(l=10, r=10, t=40 if title else 10, b=10),
            legend=_LEGEND_HORIZONTAL,
            hovermode="x unified",
        )
        if log_y:
            fig.update_yaxes(type="log")
        container.plotly_chart(fig, width="stretch", theme="streamlit")
    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for i, (name, y) in enumerate(series.items()):
            ax.plot(x, y, label=name, color=_series_color(name, i), linewidth=1.6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, linewidth=0.5, alpha=0.3)
        _mpl_legend(ax, len(series))
        fig.tight_layout()
        container.pyplot(fig, clear_figure=True)
    else:
        df = pd.DataFrame({name: np.asarray(y) for name, y in series.items()}, index=x)
        df.index.name = x_label
        container.line_chart(df, height=height)


def stacked_area_chart(
    container,
    x: np.ndarray,
    components: Mapping[str, np.ndarray],
    *,
    total: Optional[np.ndarray] = None,
    x_label: str,
    y_label: str,
    title: str = "",
    height: int = 420,
) -> None:
    """Stacked-area part-to-whole chart, with an optional neutral "total" line.

    Used for radiance-vs-wavelength. Only components with at least one
    non-zero sample are drawn -- :class:`~mwir_atmos.types.RadianceResult`
    zero-fills every component that does not apply to a scenario, and a
    stack padded out with several always-flat-zero layers is just clutter
    (and, per the dataviz skill, still spends categorical color slots on
    nothing).
    """
    nonzero = {name: y for name, y in components.items() if np.any(np.abs(np.asarray(y)) > 0.0)}
    if not nonzero:
        container.info("Every radiance component is zero for this scenario -- nothing to stack.")
        return

    if HAS_PLOTLY:
        fig = go.Figure()
        for i, (name, y) in enumerate(nonzero.items()):
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="lines", name=name, stackgroup="components",
                    line=dict(width=0.5, color=_series_color(name, i)), fillcolor=_series_color(name, i),
                )
            )
        if total is not None:
            fig.add_trace(
                go.Scatter(
                    x=x, y=total, mode="lines", name="total",
                    line=dict(color=TOTAL_LINE_COLOR, width=2, dash="dot"),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=height,
            margin=dict(l=10, r=10, t=40 if title else 10, b=10),
            legend=_LEGEND_HORIZONTAL,
            hovermode="x unified",
        )
        container.plotly_chart(fig, width="stretch", theme="streamlit")
    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        names = list(nonzero.keys())
        ax.stackplot(x, list(nonzero.values()), labels=names, colors=[_series_color(name, i) for i, name in enumerate(names)], alpha=0.9)
        if total is not None:
            ax.plot(x, total, color=TOTAL_LINE_COLOR, linewidth=2, linestyle=":", label="total")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        ax.grid(True, linewidth=0.5, alpha=0.3)
        _mpl_legend(ax, len(names) + (1 if total is not None else 0))
        fig.tight_layout()
        container.pyplot(fig, clear_figure=True)
    else:
        df = pd.DataFrame({name: np.asarray(y) for name, y in nonzero.items()}, index=x)
        df.index.name = x_label
        container.area_chart(df, height=height)


def profile_chart(
    container,
    value: np.ndarray,
    altitude_km: np.ndarray,
    *,
    x_label: str,
    title: str = "",
    height: int = 420,
    log_x: bool = False,
) -> None:
    """Single-series vertical profile: altitude on the shared y-axis (the
    atmospheric-science convention -- see README "atmospheric profile"),
    value on x. One series takes the first categorical slot with no legend
    box (dataviz skill: a single series needs no legend -- the title names
    it).

    ``log_x`` puts the value axis on a log scale, which quantities spanning
    many decades need to be readable at all -- Cn^2 covers roughly thirty
    orders of magnitude between the surface and the top of the atmosphere.
    Non-positive values are dropped rather than clipped, since a log axis has
    no meaningful place to put them.
    """
    value = np.asarray(value, dtype=np.float64)
    altitude_km = np.asarray(altitude_km, dtype=np.float64)
    if log_x:
        keep = value > 0.0
        value, altitude_km = value[keep], altitude_km[keep]
        if value.size == 0:
            container.info(f"No positive {x_label} values to plot on a log axis.")
            return

    if HAS_PLOTLY:
        fig = go.Figure(
            go.Scatter(x=value, y=altitude_km, mode="lines", line=dict(color=CATEGORICAL_COLORS[0], width=2))
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title="Altitude (km)",
            height=height,
            margin=dict(l=10, r=10, t=40 if title else 10, b=10),
            showlegend=False,
        )
        if log_x:
            fig.update_xaxes(type="log")
        container.plotly_chart(fig, width="stretch", theme="streamlit")
    elif HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(4.4, 5.2))
        ax.plot(value, altitude_km, color=CATEGORICAL_COLORS[0], linewidth=1.8)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Altitude (km)")
        if title:
            ax.set_title(title)
        ax.grid(True, linewidth=0.5, alpha=0.3)
        fig.tight_layout()
        container.pyplot(fig, clear_figure=True)
    else:
        df = pd.DataFrame({x_label: value}, index=altitude_km)
        df.index.name = "Altitude (km)"
        container.line_chart(df, height=height)


# =========================================================================
# Static PNG export (download buttons)
#
# Server-side PNG export from a plotly figure needs the optional `kaleido`
# package, which is not installed in this environment and is deliberately
# not being added (the airgapped install keeps dependencies light; see
# README.md "limitations"). matplotlib renders PNG natively with no
# extra package, so these three helpers reuse it purely as a headless PNG
# backend for the download buttons regardless of which library is drawing
# the on-screen chart above them. Each returns None (the caller shows a
# caption instead of a broken button) if matplotlib is not available either.
# =========================================================================


def _finish_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def line_chart_png(
    x: np.ndarray, series: Mapping[str, np.ndarray], *, x_label: str, y_label: str, title: str = ""
) -> Optional[bytes]:
    if not HAS_MATPLOTLIB or not series:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, y, label=name, color=_series_color(name, i), linewidth=1.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    _mpl_legend(ax, len(series))
    fig.tight_layout()
    return _finish_png(fig)


def stacked_area_png(
    x: np.ndarray,
    components: Mapping[str, np.ndarray],
    *,
    total: Optional[np.ndarray] = None,
    x_label: str,
    y_label: str,
    title: str = "",
) -> Optional[bytes]:
    if not HAS_MATPLOTLIB:
        return None
    nonzero = {name: y for name, y in components.items() if np.any(np.abs(np.asarray(y)) > 0.0)}
    if not nonzero:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    names = list(nonzero.keys())
    ax.stackplot(x, list(nonzero.values()), labels=names, colors=[_series_color(name, i) for i, name in enumerate(names)], alpha=0.9)
    if total is not None:
        ax.plot(x, total, color=TOTAL_LINE_COLOR, linewidth=2, linestyle=":", label="total")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    _mpl_legend(ax, len(names) + (1 if total is not None else 0))
    fig.tight_layout()
    return _finish_png(fig)


def profile_png(
    value: np.ndarray, altitude_km: np.ndarray, *, x_label: str, title: str = "",
    log_x: bool = False,
) -> Optional[bytes]:
    """PNG twin of :func:`profile_chart`. ``log_x`` must be passed to match it --
    a downloaded chart that differs from the one on screen is a bug."""
    if not HAS_MATPLOTLIB:
        return None
    value = np.asarray(value, dtype=np.float64)
    altitude_km = np.asarray(altitude_km, dtype=np.float64)
    if log_x:
        keep = value > 0.0
        value, altitude_km = value[keep], altitude_km[keep]
        if value.size == 0:
            return None
    fig, ax = plt.subplots(figsize=(4.4, 5.2), dpi=150)
    ax.plot(value, altitude_km, color=CATEGORICAL_COLORS[0], linewidth=1.8)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Altitude (km)")
    if title:
        ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    return _finish_png(fig)
