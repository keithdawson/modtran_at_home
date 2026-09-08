"""Loose solar and lunar positioning for illumination geometry.

Deliberately low-precision. The sun is placed from latitude, longitude and local
clock time with the timezone approximated from longitude unless given, which is
accurate to about a degree -- ample for placing a source in the sky for
single-scatter radiometry, and nowhere near ephemeris grade. The moon is placed
kinematically from the sun plus a phase input rather than from an orbit model.
"""

from __future__ import annotations

from .moon import lunar_position
from .sun import solar_position

__all__ = ["solar_position", "lunar_position"]
