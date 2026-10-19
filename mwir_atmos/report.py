"""Band summaries and result writers (table, CSV, JSON, NPZ).

Shared by the CLI (``mwir --format ...``), the GUI's download buttons and
``scripts/verify_examples.py``. Band summaries use
:func:`~mwir_atmos.spectral.band_average` for transmittance and
:func:`~mwir_atmos.spectral.band_integrate` for radiance.

``ScenarioResult.radiance`` is ``None`` when radiance was not requested
(``mwir --no-radiance``); every writer reports that plainly and writes
transmittance only.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from typing import Any, Callable, Optional, Sequence, TextIO, Union

import numpy as np

from .constants import BANDS_CM1
from .scenario_io import scenario_to_dict
from .spectral import band_average, band_integrate
from .types import RADIANCE_COMPONENTS, ScenarioResult, SkyPosition

__all__ = [
    "BandRow",
    "BandSpec",
    "band_rows",
    "parse_band_token",
    "print_band_catalog",
    "resolve_band_specs",
    "write_csv",
    "write_json",
    "write_npz",
    "write_table",
]

#: A ``constants.BANDS_CM1`` name, or a ``(lo, hi)`` cm^-1 range.
BandSpec = Union[str, tuple[float, float]]

#: Short header labels for the radiance table. The full names are up to 23
#: characters and run together at any readable column width.
_RADIANCE_ABBREV: dict[str, str] = {
    "path_thermal": "path_therm",
    "surface_emitted": "surf_emit",
    "surface_reflected_solar": "surf_r_sol",
    "surface_reflected_lunar": "surf_r_lun",
    "solar_single_scatter": "sol_scat",
    "lunar_single_scatter": "lun_scat",
    "direct_solar": "dir_sol",
    "direct_lunar": "dir_lun",
    "source_emitted": "src_emit",
    "solar_multiple_scatter": "sol_multi",
    "lunar_multiple_scatter": "lun_multi",
}

_TRANSMITTANCE_PARTS = ("total", "molecular", "aerosol", "rayleigh")


# =========================================================================
# Band selection ("--bands MWIR_window_1 2600:2650 ...")
# =========================================================================


def parse_band_token(token: str) -> tuple[str, BandSpec]:
    """One ``--bands`` token -> ``(display_label, band_spec)``.

    ``band_spec`` is a ``constants.BANDS_CM1`` key, or a ``(lo, hi)`` cm^-1
    tuple parsed from a ``'MIN:MAX'`` token.
    """
    if token in BANDS_CM1:
        return token, token
    if ":" in token:
        lo_s, hi_s = token.split(":", 1)
        try:
            lo, hi = float(lo_s), float(hi_s)
        except ValueError as exc:
            raise ValueError(f"invalid band range {token!r}: expected 'MIN:MAX' in cm^-1") from exc
        if hi <= lo:
            raise ValueError(f"invalid band range {token!r}: MAX must exceed MIN")
        return f"{lo:g}-{hi:g}cm-1", (lo, hi)
    raise ValueError(
        f"unknown band {token!r}: not a name in constants.BANDS_CM1 "
        f"({', '.join(sorted(BANDS_CM1))}) and not a 'MIN:MAX' cm^-1 range"
    )


def resolve_band_specs(tokens: Optional[Sequence[str]]) -> list[tuple[str, BandSpec]]:
    """``None`` (flag omitted) -> every named band. ``[]`` (flag given empty) -> none."""
    if tokens is None:
        return [(name, name) for name in BANDS_CM1]
    return [parse_band_token(t) for t in tokens]


def print_band_catalog(fh: Optional[TextIO] = None) -> None:
    # Resolve stdout at call time: a ``fh=sys.stdout`` default would bind the
    # stream at import and miss later reassignment (e.g. pytest's capsys).
    if fh is None:
        fh = sys.stdout
    print(f"{'name':<20}{'cm^-1 range':<20}{'um range':<16}", file=fh)
    for name, (lo, hi) in BANDS_CM1.items():
        um_lo, um_hi = 1.0e4 / hi, 1.0e4 / lo
        print(f"{name:<20}{f'{lo:.2f}-{hi:.2f}':<20}{f'{um_lo:.3f}-{um_hi:.3f}':<16}", file=fh)


# =========================================================================
# Band summary rows (shared by the table/json/npz writers and the GUI)
# =========================================================================


@dataclasses.dataclass
class BandRow:
    label: str
    lo_cm1: float
    hi_cm1: float
    lo_um: float
    hi_um: float
    transmittance: dict[str, float]
    species_transmittance: dict[str, float]
    radiance: Optional[dict[str, float]]


def band_rows(result: ScenarioResult, band_specs: Sequence[tuple[str, BandSpec]],
              warn: Optional[Callable[[str], None]] = None) -> list[BandRow]:
    """Band-averaged transmittance and band-integrated radiance per band.

    Bands that do not overlap the computed grid are skipped, with a message
    to ``warn`` if given.
    """
    grid = result.grid
    t = result.transmittance
    rows: list[BandRow] = []
    for label, spec in band_specs:
        try:
            lo, hi = BANDS_CM1[spec] if isinstance(spec, str) else spec
            trans = {part: float(band_average(grid, getattr(t, part), spec))
                     for part in _TRANSMITTANCE_PARTS}
            species_trans = {
                sp.value: float(band_average(grid, arr, spec)) for sp, arr in t.by_species.items()
            }
            radiance: Optional[dict[str, float]] = None
            if result.radiance is not None:
                radiance = {
                    name: float(band_integrate(grid, getattr(result.radiance, name), spec))
                    for name in RADIANCE_COMPONENTS
                }
        except ValueError as exc:
            if warn is not None:
                warn(f"skipping band {label!r}: {exc}")
            continue
        rows.append(BandRow(label, lo, hi, 1.0e4 / hi, 1.0e4 / lo, trans, species_trans, radiance))
    return rows


# =========================================================================
# Human-readable table
# =========================================================================


def write_table(result: ScenarioResult, rows: list[BandRow], fh: TextIO) -> None:
    _write_scenario_header(result, fh)
    if not rows:
        print("(no bands selected -- pass --bands to see a band summary)", file=fh)
        return
    _write_transmittance_table(rows, fh)
    _write_species_table(rows, fh)
    if any(r.radiance is not None for r in rows):
        _write_radiance_table(rows, fh)
    elif result.radiance is None:
        print("(radiance not computed -- transmittance-only summary above)", file=fh)
    _write_turbulence_block(result, fh)


def _write_scenario_header(result: ScenarioResult, fh: TextIO) -> None:
    sc = result.scenario
    grid = result.grid
    path = result.path

    def line(label: str, text: str) -> None:
        print(f"  {label:<18}: {text}", file=fh)

    print("mwir_atmos scenario summary", file=fh)
    line("atmosphere", sc.atmosphere.value)
    line("surface type", sc.surface_type.value)
    line("aerosol", f"{sc.aerosol.value} (visibility {sc.visibility_km:g} km)")
    line("sensor", f"alt {sc.sensor_alt_km:g} km, zenith {sc.sensor_zenith_deg:g} deg, "
                   f"azimuth {sc.sensor_azimuth_deg:g} deg")
    source = []
    if sc.source_alt_km is not None:
        source.append(f"alt {sc.source_alt_km:g} km")
    if sc.source_range_km is not None:
        source.append(f"range {sc.source_range_km:g} km")
    if sc.source_temp_k is not None:
        source.append(f"T {sc.source_temp_k:g} K")
    line("source", ", ".join(source) if source else "(none -- path runs to ground/space)")
    line("spectral grid",
         f"{grid.wavenumber_cm1[0]:.2f}-{grid.wavenumber_cm1[-1]:.2f} cm^-1 "
         f"({grid.wavelength_um[-1]:.3f}-{grid.wavelength_um[0]:.3f} um), "
         f"resolution {grid.resolution_cm1:g} cm^-1, {grid.n} points")
    tangent = f", tangent_alt={path.tangent_alt_km:.3f} km" if path.tangent_alt_km is not None else ""
    line("path", f"{path.n_layers} layers, {path.total_length_km:.3f} km, "
                 f"hits_ground={path.hits_ground}, reaches_space={path.reaches_space}{tangent}")
    if result.sun is not None:
        line("sun", f"zenith {result.sun.zenith_deg:.2f} deg, "
                    f"azimuth {result.sun.azimuth_deg:.2f} deg, "
                    f"above_horizon={result.sun.is_above_horizon}")
    if result.moon is not None:
        line("moon", f"zenith {result.moon.zenith_deg:.2f} deg, "
                     f"azimuth {result.moon.azimuth_deg:.2f} deg, "
                     f"above_horizon={result.moon.is_above_horizon}, "
                     f"illuminated={result.moon.illuminated_fraction:.2f}")
    if result.radiance is None:
        line("radiance", "not computed (transmittance only)")
    print(file=fh)


def _write_transmittance_table(rows: list[BandRow], fh: TextIO) -> None:
    header = (f"{'band':<20}{'range (um)':<16}"
              f"{'trans_total':>12}{'trans_mol':>12}{'trans_aer':>12}{'trans_ray':>12}")
    print(header, file=fh)
    print("-" * len(header), file=fh)
    for r in rows:
        rng = f"{r.lo_um:.3f}-{r.hi_um:.3f}"
        values = "".join(f"{r.transmittance[part]:>12.4e}" for part in _TRANSMITTANCE_PARTS)
        print(f"{r.label:<20}{rng:<16}{values}", file=fh)
    print(file=fh)


def _write_species_table(rows: list[BandRow], fh: TextIO) -> None:
    species_names = sorted({sp for r in rows for sp in r.species_transmittance})
    if not species_names:
        return
    print("band-averaged transmittance by species", file=fh)
    header = f"{'band':<20}" + "".join(f"{sp:>12}" for sp in species_names)
    print(header, file=fh)
    print("-" * len(header), file=fh)
    for r in rows:
        line = f"{r.label:<20}"
        for sp in species_names:
            v = r.species_transmittance.get(sp)
            line += f"{v:>12.4e}" if v is not None else f"{'--':>12}"
        print(line, file=fh)
    print(file=fh)


def _write_radiance_table(rows: list[BandRow], fh: TextIO) -> None:
    print("band-integrated radiance (W cm^-2 sr^-1)", file=fh)
    # Abbreviated headers, expanded in a legend below the table.
    abbrev = {c: _RADIANCE_ABBREV.get(c, c)[:11] for c in RADIANCE_COMPONENTS}
    header = f"{'band':<20}" + "".join(f"{abbrev[c]:>12}" for c in RADIANCE_COMPONENTS)
    print(header, file=fh)
    print("-" * len(header), file=fh)
    for r in rows:
        if r.radiance is None:
            continue
        print(f"{r.label:<20}" + "".join(f"{r.radiance[c]:>12.4e}" for c in RADIANCE_COMPONENTS),
              file=fh)
    legend = ", ".join(f"{abbrev[c]}={c}" for c in RADIANCE_COMPONENTS if abbrev[c] != c)
    if legend:
        print(f"  ({legend})", file=fh)
    print(file=fh)


def _write_turbulence_block(result: ScenarioResult, fh: TextIO) -> None:
    """Optical-turbulence summary, when a Cn^2 model was requested.

    These are diagnostics, not corrections: turbulence redistributes a point
    source in angle and time, it does not absorb, so no radiance value above is
    modified by them.
    """
    t = result.turbulence
    if t is None:
        return

    def line(label: str, text: str) -> None:
        print(f"  {label:<22}: {text}", file=fh)

    print("optical turbulence (diagnostic -- does not modify the radiance above)", file=fh)
    line("Cn^2 model", result.scenario.turbulence.value)
    line("reference wavelength", f"{t.reference_wavelength_um:.3f} um")
    line("Fried parameter r0", f"{t.fried_parameter_r0_cm:.2f} cm")
    line("isoplanatic angle", f"{t.isoplanatic_angle_urad:.2f} urad")
    line("Rytov variance", f"{t.rytov_variance:.4f}")
    line("scintillation index", f"{t.scintillation_index:.4f}")
    if t.greenwood_frequency_hz is not None:
        line("Greenwood frequency", f"{t.greenwood_frequency_hz:.2f} Hz")
    if t.aperture_diameter_m is not None:
        regime = "SEEING-limited" if t.is_seeing_limited else "diffraction-limited"
        line("aperture", f"{t.aperture_diameter_m * 100:.1f} cm  ({regime})")
        line("diffraction limit", f"{t.diffraction_limit_urad:.2f} urad (lambda/D)")
        line("seeing blur", f"{t.seeing_limit_urad:.2f} urad (lambda/r0)")
        line("effective spot size", f"{t.effective_blur_urad:.2f} urad")
        line("Strehl / peak ratio", f"{t.strehl_ratio:.4f}")
    print(file=fh)


# =========================================================================
# Full spectral arrays: CSV, JSON, NPZ
# =========================================================================


def _spectral_columns(result: ScenarioResult) -> dict[str, np.ndarray]:
    """Every per-wavenumber array of ``result``, keyed by its CSV/NPZ column name."""
    grid = result.grid
    t = result.transmittance
    columns: dict[str, np.ndarray] = {
        "wavenumber_cm1": grid.wavenumber_cm1,
        "wavelength_um": grid.wavelength_um,
    }
    for part in _TRANSMITTANCE_PARTS:
        columns[f"trans_{part}"] = getattr(t, part)
    for sp in sorted(t.by_species, key=lambda s: s.value):
        columns[f"trans_{sp.value}"] = t.by_species[sp]
    if result.radiance is not None:
        for c in RADIANCE_COMPONENTS:
            columns[f"radiance_{c}"] = getattr(result.radiance, c)
        for name, error in result.radiance.sampling_standard_error.items():
            columns[f"radiance_stderr_{name}"] = error
    return columns


def write_csv(result: ScenarioResult, fh: TextIO) -> None:
    """Full per-wavenumber spectral arrays, one row per grid point.

    Always the complete grid regardless of band selection: CSV is the
    scriptable path and carries the real arrays, not a summary.
    """
    columns = _spectral_columns(result)
    writer = csv.writer(fh)
    writer.writerow(list(columns))
    writer.writerows(zip(*columns.values()))


def write_json(result: ScenarioResult, rows: list[BandRow], fh: TextIO) -> None:
    grid = result.grid
    t = result.transmittance
    payload: dict[str, Any] = {
        "scenario": scenario_to_dict(result.scenario),
        "grid": {
            "wavenumber_cm1": grid.wavenumber_cm1.tolist(),
            "wavelength_um": grid.wavelength_um.tolist(),
            "resolution_cm1": grid.resolution_cm1,
        },
        "path": {
            "n_layers": result.path.n_layers,
            "total_length_km": result.path.total_length_km,
            "hits_ground": result.path.hits_ground,
            "reaches_space": result.path.reaches_space,
            "tangent_alt_km": result.path.tangent_alt_km,
            "sensor_alt_km": result.path.sensor_alt_km,
            "source_alt_km": result.path.source_alt_km,
        },
        "transmittance": {
            **{part: getattr(t, part).tolist() for part in _TRANSMITTANCE_PARTS},
            "by_species": {sp.value: arr.tolist() for sp, arr in t.by_species.items()},
        },
        "radiance": None,
        "sun": _sky_position_to_dict(result.sun),
        "moon": _sky_position_to_dict(result.moon),
        "bands": {
            r.label: {
                "range_cm1": [r.lo_cm1, r.hi_cm1],
                "range_um": [r.lo_um, r.hi_um],
                "transmittance": r.transmittance,
                "transmittance_by_species": r.species_transmittance,
                "radiance": r.radiance,
            }
            for r in rows
        },
    }
    if result.radiance is not None:
        payload["radiance"] = {c: getattr(result.radiance, c).tolist() for c in RADIANCE_COMPONENTS}
        if result.radiance.sampling_standard_error:
            payload["radiance_sampling_standard_error"] = {
                k: v.tolist() for k, v in result.radiance.sampling_standard_error.items()}
    json.dump(payload, fh, indent=2)
    fh.write("\n")


def write_npz(result: ScenarioResult, rows: list[BandRow], path: str) -> None:
    arrays = _spectral_columns(result)
    if rows:
        arrays["band_label"] = np.array([r.label for r in rows])
        arrays["band_lo_cm1"] = np.array([r.lo_cm1 for r in rows], dtype=np.float64)
        arrays["band_hi_cm1"] = np.array([r.hi_cm1 for r in rows], dtype=np.float64)
        for part in _TRANSMITTANCE_PARTS:
            arrays[f"band_trans_{part}"] = np.array([r.transmittance[part] for r in rows],
                                                    dtype=np.float64)
        if any(r.radiance is not None for r in rows):
            for c in RADIANCE_COMPONENTS:
                arrays[f"band_radiance_{c}"] = np.array(
                    [(r.radiance[c] if r.radiance is not None else np.nan) for r in rows],
                    dtype=np.float64)
    np.savez_compressed(path, **arrays)


def _sky_position_to_dict(pos: Optional[SkyPosition]) -> Optional[dict[str, Any]]:
    if pos is None:
        return None
    return {
        "zenith_deg": pos.zenith_deg,
        "azimuth_deg": pos.azimuth_deg,
        "scattering_angle_deg": pos.scattering_angle_deg,
        "is_above_horizon": pos.is_above_horizon,
        "distance_factor": pos.distance_factor,
        "illuminated_fraction": pos.illuminated_fraction,
    }
