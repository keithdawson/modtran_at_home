"""mwir_atmos -- mid-wave infrared atmospheric transmittance and radiance.

A MODTRAN-flavoured line-by-line MWIR calculator. The default spectral grid
spans 1800-4300 cm^-1 (2.33-5.56 um), which fully contains the 3-5 um MWIR
band with margin for the CO2 4.3 um and H2O 2.7 um band wings.

Only the stable, overseer-owned surface is re-exported here. Physics modules
are imported directly (``from mwir_atmos.optics.lbl import ...``) so that a
module still under construction cannot break ``import mwir_atmos``.
"""

from __future__ import annotations

from . import constants, spectral
from .spectral import band_average, band_integrate, make_grid, planck
from .types import (
    AerosolModel,
    AtmosphereProfile,
    LayerOptics,
    ModelAtmosphere,
    PathGeometry,
    PathLayer,
    RadianceResult,
    Scenario,
    ScenarioResult,
    SkyPosition,
    SpectralGrid,
    Species,
    SurfaceType,
    TransmittanceResult,
    TurbulenceModel,
    TurbulenceResult,
)

__version__ = "0.1.0"

__all__ = [
    "constants",
    "spectral",
    "make_grid",
    "planck",
    "band_integrate",
    "band_average",
    "AerosolModel",
    "AtmosphereProfile",
    "LayerOptics",
    "ModelAtmosphere",
    "PathGeometry",
    "PathLayer",
    "RadianceResult",
    "Scenario",
    "ScenarioResult",
    "SkyPosition",
    "SpectralGrid",
    "Species",
    "SurfaceType",
    "TransmittanceResult",
    "TurbulenceModel",
    "TurbulenceResult",
]
