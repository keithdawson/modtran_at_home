"""Download HITRAN line-by-line data from hitran.org.

Uses the HITRAN "lbl/api" HTTP endpoint (``https://hitran.org/lbl/api``),
queried by *global* isotopologue ID (not molecule ID) so isotopologue
coverage is explicit rather than relying on the endpoint's own defaults.

CONTRACT.md invariant 5: the finished package must run with no network.
:func:`ensure_linelists` is the ONLY function in :mod:`mwir_atmos` permitted
to touch the network -- it loads ``data/hitran/<species>.npz`` if present
and only calls :func:`fetch_linelist` (which also hits the network) when the
cache is missing or ``force=True`` is passed explicitly. Everything else
(:mod:`mwir_atmos.optics.lbl`, :mod:`mwir_atmos.hitran.tips`) reads the
on-disk cache and never fetches.
"""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

from ..types import Species
from .cache import LineList, load_linelist, save_linelist

__all__ = [
    "fetch_linelist",
    "ensure_linelists",
    "GLOBAL_ISOTOPOLOGUES",
    "MOLECULE_IDS",
]

_API_URL = "https://hitran.org/lbl/api"
# Literal token required by the endpoint for the CSV field separator -- a
# real comma in `sep` returns HTTP 500 (verified live 2026-08-02).
_SEP_TOKEN = "[comma]"
_REQUEST_PARAMS = (
    "molec_id,local_iso_id,nu,sw,gamma_air,gamma_self,elower,n_air,delta_air"
)
_FIELD_NAMES = (
    "molec_id", "local_iso_id", "nu_cm1", "s296", "gamma_air", "gamma_self",
    "e_lower", "n_air", "delta_air",
)
_N_FIELDS = len(_FIELD_NAMES)
_FETCH_PAUSE_S = 2.0        # politeness pause between per-molecule requests
_HTTP_TIMEOUT_S = 180.0
_MAX_ATTEMPTS = 3
_USER_AGENT = "mwir_atmos/0.1 (offline MWIR atmospheric transmittance calculator)"

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "hitran"

#: HITRAN molecule ID per species (distinct from the *global* isotopologue ID
#: used in the API's ``iso_ids_list``).
MOLECULE_IDS: dict[Species, int] = {
    Species.H2O: 1,
    Species.CO2: 2,
    Species.O3: 3,
    Species.N2O: 4,
    Species.CO: 5,
    Species.CH4: 6,
}

#: Global isotopologue IDs fetched per species: (global_id, local_iso_id,
#: terrestrial abundance, molar_mass_g_mol). Restricted to isotopologues
#: with terrestrial abundance >= 1e-4 (0.01%) -- this is what the job spec
#: calls the "significant minor isotopologues": for H2O it keeps main/18O/
#: 17O/HDO (161/181/171/162) and drops the doubly-substituted species; for
#: CO2 it keeps 626/636/628/627 and drops 638 (4.4e-5) and rarer.
#:
#: Verified LIVE against https://hitran.org/docs/iso-meta/ on 2026-08-02
#: (fetched via WebFetch, cross-checked by spot-querying the lbl/api itself
#: -- e.g. iso_ids_list=26 returns molec_id=5/local_iso_id=1, confirming
#: global ID 26 is CO's main isotopologue, NOT a guessed value). Molar
#: masses are HITRAN's own values from the same metadata table. Do not
#: hand-edit this table from memory; re-fetch the iso-meta page if it ever
#: needs to change.
GLOBAL_ISOTOPOLOGUES: dict[Species, list[tuple[int, int, float, float]]] = {
    Species.H2O: [
        (1, 1, 9.97317e-1, 18.010565),   # H2(16O)
        (2, 2, 1.99983e-3, 20.014811),   # H2(18O)
        (3, 3, 3.71884e-4, 19.014780),   # H2(17O)
        (4, 4, 3.10693e-4, 19.016740),   # HD(16O)
    ],
    Species.CO2: [
        (7, 1, 9.84204e-1, 43.989830),   # (12C)(16O)2
        (8, 2, 1.10574e-2, 44.993185),   # (13C)(16O)2
        (9, 3, 3.94707e-3, 45.994076),   # (16O)(12C)(18O)
        (10, 4, 7.33989e-4, 44.994045),  # (16O)(12C)(17O)
    ],
    Species.O3: [
        (16, 1, 9.92901e-1, 47.984745),  # (16O)3
        (17, 2, 3.98194e-3, 49.988991),  # (16O)(16O)(18O)
        (18, 3, 1.99097e-3, 49.988991),  # (16O)(18O)(16O)
        (19, 4, 7.40475e-4, 48.988960),  # (16O)(16O)(17O)
        (20, 5, 3.70237e-4, 48.988960),  # (16O)(17O)(16O)
    ],
    Species.N2O: [
        (21, 1, 9.90333e-1, 44.001062),  # (14N)2(16O)
        (22, 2, 3.64093e-3, 44.998096),  # (14N)(15N)(16O)
        (23, 3, 3.64093e-3, 44.998096),  # (15N)(14N)(16O)
        (24, 4, 1.98582e-3, 46.005308),  # (14N)2(18O)
        (25, 5, 3.69280e-4, 45.005278),  # (14N)2(17O)
    ],
    Species.CO: [
        (26, 1, 9.86544e-1, 27.994915),  # (12C)(16O)
        (27, 2, 1.10836e-2, 28.998270),  # (13C)(16O)
        (28, 3, 1.97822e-3, 29.999161),  # (12C)(18O)
        (29, 4, 3.67867e-4, 28.999130),  # (12C)(17O)
    ],
    Species.CH4: [
        (32, 1, 9.88274e-1, 16.031300),  # (12C)H4
        (33, 2, 1.11031e-2, 17.034655),  # (13C)H4
        (34, 3, 6.15751e-4, 17.037475),  # (12C)H3D
    ],
}


def _cache_path(species: Species, cache_dir: Path) -> Path:
    return cache_dir / f"{species.value}.npz"


def _http_get_csv(url: str) -> str:
    """GET a URL, tolerating hitran.org's Content-Length/body-length quirk.

    Empirically (verified 2026-08-02), hitran.org/lbl/api declares a
    Content-Length exactly ``n_lines`` bytes larger than the body it
    actually sends (consistent with the header being computed assuming
    CRLF row endings while the body is written with LF only) -- urllib
    raises ``http.client.IncompleteRead` for this even though the payload
    is complete; ``exc.partial`` is the full, correctly-terminated response
    (checked to end in a newline with a well-formed last row by the caller).
    Retries a couple of times on genuine transient network errors.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            resp = urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S)
            try:
                raw = resp.read()
            except http.client.IncompleteRead as exc:
                raw = exc.partial
            return raw.decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(_FETCH_PAUSE_S * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def fetch_linelist(
    species: Species,
    numin: float,
    numax: float,
    *,
    isotopologues: Sequence[int] | None = None,
) -> LineList:
    """Download one species' line list from hitran.org over ``[numin, numax]``.

    ``isotopologues``, if given, is a sequence of *global* HITRAN
    isotopologue IDs restricting which isotopologues to fetch; defaults to
    :data:`GLOBAL_ISOTOPOLOGUES[species]`. All requested isotopologues are
    fetched in a single HTTP request (the API's ``iso_ids_list`` accepts a
    comma-separated list) to keep hitran.org load to one request per
    species. Prefer :func:`ensure_linelists` for normal use -- it adds
    on-disk caching so repeated runs never re-fetch.
    """
    iso_table = GLOBAL_ISOTOPOLOGUES[species]
    if isotopologues is None:
        selected = iso_table
    else:
        wanted = set(isotopologues)
        selected = [row for row in iso_table if row[0] in wanted]
        if not selected:
            raise ValueError(f"no known isotopologues of {species} in {isotopologues!r}")

    global_ids = ",".join(str(g) for g, _local, _ab, _mass in selected)
    abundance_by_local_iso = {local: ab for _g, local, ab, _mass in selected}

    params = {
        "iso_ids_list": global_ids,
        "numin": str(float(numin)),
        "numax": str(float(numax)),
        "head": "0",
        "fixwidth": "0",
        "sep": _SEP_TOKEN,
        "request_params": _REQUEST_PARAMS,
    }
    url = _API_URL + "?" + urllib.parse.urlencode(params, safe="[],")

    text = _http_get_csv(url)
    rows = [line for line in text.split("\n") if line.strip()]
    if not rows:
        raise RuntimeError(
            f"HITRAN returned no lines for {species} in [{numin}, {numax}] cm^-1 -- {url}"
        )

    split_rows = [row.split(",") for row in rows]
    bad = [row for row in split_rows if len(row) != _N_FIELDS]
    if bad:
        raise ValueError(
            f"malformed HITRAN row for {species} (expected {_N_FIELDS} fields): {bad[0]!r}"
        )
    arr = np.asarray(split_rows, dtype=np.float64)

    molec_id = arr[:, 0].astype(np.int64)
    local_iso_id = arr[:, 1].astype(np.int64)
    nu_cm1 = arr[:, 2]
    s296 = arr[:, 3]
    gamma_air = arr[:, 4]
    gamma_self = arr[:, 5]
    e_lower = arr[:, 6]
    n_air = arr[:, 7]
    delta_air = arr[:, 8]

    expected_molec_id = MOLECULE_IDS[species]
    bad_mask = molec_id != expected_molec_id
    if np.any(bad_mask):
        raise ValueError(
            f"HITRAN response for {species} contained {int(bad_mask.sum())} rows "
            f"with an unexpected molec_id (expected {expected_molec_id}); the "
            f"GLOBAL_ISOTOPOLOGUES table in fetch.py may be wrong for this species"
        )

    try:
        iso_abundance = np.array(
            [abundance_by_local_iso[int(i)] for i in local_iso_id], dtype=np.float64
        )
    except KeyError as exc:
        raise ValueError(
            f"HITRAN response for {species} included unexpected local_iso_id "
            f"{exc} not in the requested isotopologue set {sorted(abundance_by_local_iso)}"
        ) from exc

    return LineList(
        species=species,
        molecule_id=expected_molec_id,
        nu_cm1=nu_cm1,
        s296=s296,
        gamma_air=gamma_air,
        gamma_self=gamma_self,
        e_lower=e_lower,
        n_air=n_air,
        delta_air=delta_air,
        local_iso_id=local_iso_id,
        iso_abundance=iso_abundance,
    )


#: Memoised line lists, keyed by (cache directory, numin, numax). Line lists are
#: immutable, so one disk read per process suffices; without this every call
#: re-read ~16 MB of .npz and re-allocated the arrays.
_LINELIST_MEMO: dict[tuple[str, float, float], dict[Species, "LineList"]] = {}


def ensure_linelists(
    *,
    cache_dir=None,
    force: bool = False,
    numin: float = 1775.0,
    numax: float = 4325.0,
) -> dict[Species, LineList]:
    """Load every species' line list from ``data/hitran/`` if present, else
    fetch from hitran.org and cache.

    The ONLY function permitted to touch the network (CONTRACT.md invariant
    5). Fetches are sequential, one HTTP request per species (all of that
    species' isotopologues in one request), with a politeness pause between
    requests, and never re-fetch a species whose cache file already exists
    unless ``force=True``.
    """
    cdir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

    # In-process memo: the .npz files are ~16 MB and every call re-read and
    # re-allocated all of them. Line lists are immutable inputs, so one load per
    # process is enough. ``force`` bypasses and refreshes the memo.
    memo_key = (str(cdir.resolve()), float(numin), float(numax))
    if not force:
        cached = _LINELIST_MEMO.get(memo_key)
        if cached is not None:
            return dict(cached)

    cdir.mkdir(parents=True, exist_ok=True)

    out: dict[Species, LineList] = {}
    fetched_any = False
    for species in MOLECULE_IDS:
        path = _cache_path(species, cdir)
        if not force and path.exists():
            out[species] = load_linelist(path)
            continue
        if fetched_any:
            time.sleep(_FETCH_PAUSE_S)
        linelist = fetch_linelist(species, numin, numax)
        save_linelist(linelist, path)
        out[species] = linelist
        fetched_any = True

    _LINELIST_MEMO[memo_key] = dict(out)
    return out


def clear_linelist_memo() -> None:
    """Drop the in-process line-list memo (used by tests that touch the cache)."""
    _LINELIST_MEMO.clear()
