"""Scenario <-> plain dict / YAML.

One serialisation shared by the CLI (``--config`` / ``--save-config``), the
GUI (setup download/upload and its cache key) and :mod:`mwir_atmos.setups`,
so the front ends cannot disagree about what a scenario file contains.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Union

import yaml

from .types import (
    AerosolModel,
    InstrumentFunction,
    ModelAtmosphere,
    RadianceSolver,
    Scenario,
    SpectralSampling,
    SurfaceType,
    TurbulenceModel,
)
from .validation import validate_scenario

__all__ = [
    "ENUM_FIELD_TYPES",
    "coerce_field",
    "load_scenario_yaml",
    "parse_datetime",
    "save_scenario_yaml",
    "scenario_from_dict",
    "scenario_to_dict",
]

#: Scenario fields that are enums, mapped to their enum class. Drives the
#: CLI's ``choices=`` and the string <-> enum conversion below.
ENUM_FIELD_TYPES: dict[str, type[Enum]] = {
    "atmosphere": ModelAtmosphere,
    "surface_type": SurfaceType,
    "aerosol": AerosolModel,
    "turbulence": TurbulenceModel,
    "radiance_solver": RadianceSolver,
    "spectral_sampling": SpectralSampling,
    "instrument_function": InstrumentFunction,
}


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Plain-Python (YAML/JSON-safe) representation of a :class:`Scenario`.

    Enums become their ``.value`` string, ``datetime`` becomes an ISO 8601
    string; everything else (float, bool, None) passes through unchanged.
    Field order follows :class:`Scenario`'s declaration order.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(Scenario):
        val = getattr(scenario, f.name)
        if isinstance(val, Enum):
            val = val.value
        elif isinstance(val, datetime):
            val = val.isoformat()
        out[f.name] = val
    return out


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    """Inverse of :func:`scenario_to_dict`, validated.

    Unset fields fall back to :class:`Scenario`'s own defaults (a config file
    only needs to mention the fields it wants to override). Raises
    ``ValueError`` on an unknown field name -- deliberately strict, since a
    silently-ignored typo in a scenario file (e.g. ``visiblity_km``) would
    otherwise run the *wrong* scenario without any indication.
    """
    field_names = {f.name for f in dataclasses.fields(Scenario)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ValueError(f"unknown scenario field(s) in config: {unknown}")

    kwargs = {name: coerce_field(name, value) for name, value in data.items()}
    scenario = dataclasses.replace(Scenario(), **kwargs)
    validate_scenario(scenario)
    return scenario


def coerce_field(name: str, value: Any) -> Any:
    """``value`` for Scenario field ``name``: enum strings to enums, ISO strings to datetime."""
    if value is None:
        return None
    if name in ENUM_FIELD_TYPES:
        return ENUM_FIELD_TYPES[name](value)
    if name == "local_datetime" and not isinstance(value, datetime):
        # PyYAML already parses ISO-8601-looking scalars into datetime; JSON
        # and the CLI hand over plain strings.
        return parse_datetime(str(value))
    return value


def load_scenario_yaml(path: Union[str, Path]) -> Scenario:
    """Load a :class:`Scenario` from a YAML file written by
    :func:`save_scenario_yaml` (or hand-written, see ``scenarios/*.yaml``)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"config file {p} did not parse to a YAML mapping (got {type(data).__name__})"
        )
    return scenario_from_dict(data)


def save_scenario_yaml(scenario: Scenario, path: Union[str, Path]) -> None:
    """Write a :class:`Scenario` to YAML, in field-declaration order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = scenario_to_dict(scenario)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid --local-datetime {value!r}: expected ISO 8601, "
            "e.g. 2026-08-02T14:30:00"
        ) from exc
