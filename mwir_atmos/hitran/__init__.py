"""HITRAN line data acquisition, caching, and partition sums.

Fetches (:mod:`mwir_atmos.hitran.fetch`) and caches (:mod:`mwir_atmos.hitran.cache`)
line-by-line spectroscopic parameters from hitran.org, plus total internal
partition sums (:mod:`mwir_atmos.hitran.tips`). See CONTRACT.md invariant 5:
:func:`ensure_linelists` is the only function anywhere in ``mwir_atmos`` that
is permitted to touch the network; everything downstream loads the
``data/hitran/*.npz`` cache this package ships with.
"""

from __future__ import annotations

from .cache import LineList, load_linelist, save_linelist
from .fetch import ensure_linelists, fetch_linelist
from .tips import partition_function

__all__ = [
    "LineList",
    "load_linelist",
    "save_linelist",
    "fetch_linelist",
    "ensure_linelists",
    "partition_function",
]
