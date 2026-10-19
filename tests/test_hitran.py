"""Tests for mwir_atmos.hitran: cache persistence, the shipped line-data
cache, and TIPS partition sums.

These tests run entirely offline against the ``data/hitran/*.npz`` cache
committed to the repo -- they never touch the network (``ensure_linelists``
only fetches when its cache file is missing, which should never be true for
a checked-out repo).
"""

from __future__ import annotations

import numpy as np
import pytest

from mwir_atmos.hitran.cache import LineList, load_linelist, save_linelist
from mwir_atmos.hitran.fetch import GLOBAL_ISOTOPOLOGUES, MOLECULE_IDS, ensure_linelists
from mwir_atmos.hitran.tips import partition_function
from mwir_atmos.types import Species

ALL_SPECIES = (Species.H2O, Species.CO2, Species.O3, Species.N2O, Species.CO, Species.CH4)


# =========================================================================
# cache.py: LineList round-trip
# =========================================================================


def _make_dummy_linelist(n: int = 5) -> LineList:
    rng = np.random.default_rng(0)
    return LineList(
        species=Species.CO2,
        molecule_id=2,
        nu_cm1=np.sort(rng.uniform(2000.0, 2400.0, n)),
        s296=rng.uniform(1e-30, 1e-20, n),
        gamma_air=rng.uniform(0.03, 0.1, n),
        gamma_self=rng.uniform(0.05, 0.15, n),
        e_lower=rng.uniform(0.0, 3000.0, n),
        n_air=rng.uniform(0.5, 0.9, n),
        delta_air=rng.uniform(-0.02, 0.0, n),
        local_iso_id=rng.integers(1, 4, n).astype(np.int64),
        iso_abundance=rng.uniform(1e-3, 1.0, n),
    )


def test_linelist_roundtrip(tmp_path):
    original = _make_dummy_linelist()
    path = tmp_path / "dummy.npz"
    save_linelist(original, path)
    loaded = load_linelist(path)

    assert loaded.species == original.species
    assert loaded.molecule_id == original.molecule_id
    for field in (
        "nu_cm1", "s296", "gamma_air", "gamma_self", "e_lower",
        "n_air", "delta_air", "local_iso_id", "iso_abundance",
    ):
        np.testing.assert_allclose(getattr(loaded, field), getattr(original, field))
    assert len(loaded) == len(original) == 5


def test_linelist_roundtrip_creates_parent_dirs(tmp_path):
    original = _make_dummy_linelist(n=2)
    nested = tmp_path / "a" / "b" / "c.npz"
    save_linelist(original, nested)
    assert nested.exists()
    loaded = load_linelist(nested)
    np.testing.assert_allclose(loaded.nu_cm1, original.nu_cm1)


# =========================================================================
# fetch.py: isotopologue table sanity (no network -- static data check)
# =========================================================================


def test_global_isotopologue_table_covers_all_species():
    assert set(GLOBAL_ISOTOPOLOGUES) == set(ALL_SPECIES)
    assert set(MOLECULE_IDS) == set(ALL_SPECIES)


def test_main_isotopologue_dominates_abundance():
    """local_iso_id 1 is HITRAN's convention for the most abundant
    (terrestrial) isotopologue of every molecule -- sanity check the table
    reflects that (all our species have a main isotopologue > 90%)."""
    for species in ALL_SPECIES:
        rows = {local: ab for _gid, local, ab, _mass in GLOBAL_ISOTOPOLOGUES[species]}
        assert rows[1] > 0.9, f"{species} local_iso_id=1 abundance {rows[1]} looks wrong"


def test_isotopologue_abundances_are_physical():
    for species in ALL_SPECIES:
        for global_id, local_iso_id, abundance, molar_mass in GLOBAL_ISOTOPOLOGUES[species]:
            assert 0.0 < abundance <= 1.0
            assert 1.0 < molar_mass < 60.0  # every isotopologue of our 6 species fits this


# =========================================================================
# The shipped cache (data/hitran/*.npz)
# =========================================================================


def test_ensure_linelists_loads_all_species_offline():
    linelists = ensure_linelists()
    assert set(linelists) == set(ALL_SPECIES)
    for species, linelist in linelists.items():
        assert linelist.species == species
        assert linelist.molecule_id == MOLECULE_IDS[species]
        assert len(linelist) > 0


@pytest.mark.parametrize("species", ALL_SPECIES)
def test_cached_linelist_within_fetch_margin(species):
    """docs/build_contract.md invariant 1: HITRAN fetches must span 1775-4325 cm^-1 --
    i.e. no line was fetched from outside the requested query range. This
    does NOT require every species to have lines populating the entire
    range: CO, for instance, genuinely has no HITRAN lines between its
    fundamental band's P-branch high-J cutoff (~1854 cm^-1) and 1775 cm^-1
    -- a real gap in the spectroscopy, not a fetch bug (verified against
    the live API during development)."""
    linelist = ensure_linelists()[species]
    assert linelist.nu_cm1.min() >= 1775.0 - 1e-6
    assert linelist.nu_cm1.max() <= 4325.0 + 1e-6


@pytest.mark.parametrize("species", (Species.H2O, Species.CO2, Species.O3, Species.N2O, Species.CH4))
def test_densely_lined_species_cover_working_grid_edges(species):
    """Unlike CO (see above), these five species have dense enough line
    forests to populate close to both edges of the fetch margin."""
    linelist = ensure_linelists()[species]
    assert linelist.nu_cm1.min() <= 1800.0
    assert linelist.nu_cm1.max() >= 4300.0


@pytest.mark.parametrize("species", ALL_SPECIES)
def test_cached_linelist_fields_are_physical(species):
    linelist = ensure_linelists()[species]
    assert np.all(linelist.s296 > 0.0)
    # A small minority of weak/obscure O3 lines (rarer isotopologues, high
    # E") ship from HITRAN with gamma_air exactly 0.0 -- no lab-measured
    # air-broadening coefficient exists for them (confirmed: ~1.3% of O3's
    # 206k lines, all local_iso_id != 1). Real HITRAN data, not a parsing
    # bug -- absorption_cross_section() handles gamma_L == 0 correctly
    # (pure-Doppler limit), so >= 0 is the right physical bound here.
    assert np.all(linelist.gamma_air >= 0.0)
    assert np.all(linelist.gamma_self > 0.0)
    assert np.all(linelist.e_lower >= 0.0)
    assert np.all(linelist.iso_abundance > 0.0) and np.all(linelist.iso_abundance <= 1.0)
    # every local_iso_id present must be one we actually requested
    requested = {local for _g, local, _a, _m in GLOBAL_ISOTOPOLOGUES[species]}
    assert set(np.unique(linelist.local_iso_id).tolist()) <= requested


@pytest.mark.parametrize("species", ALL_SPECIES)
def test_cached_iso_abundance_matches_table(species):
    """iso_abundance stored per line must match GLOBAL_ISOTOPOLOGUES, not be
    left over/mismatched from a stale fetch."""
    linelist = ensure_linelists()[species]
    expected = {local: ab for _g, local, ab, _m in GLOBAL_ISOTOPOLOGUES[species]}
    for local_iso in np.unique(linelist.local_iso_id):
        mask = linelist.local_iso_id == local_iso
        line_abundances = np.unique(linelist.iso_abundance[mask])
        assert line_abundances.size == 1
        assert line_abundances[0] == pytest.approx(expected[int(local_iso)])


# =========================================================================
# tips.py: partition functions
# =========================================================================


def test_partition_function_h2o_296k_matches_reference():
    # H2O-161 (main), commonly-cited reference Q(296 K) ~ 174.58 (TIPS).
    q = partition_function(1, 1, 296.0)
    assert q == pytest.approx(174.58, rel=1e-3)


def test_partition_function_co2_296k_matches_reference():
    # CO2-626 (main), commonly-cited reference Q(296 K) ~ 286.09 (TIPS).
    q = partition_function(2, 1, 296.0)
    assert q == pytest.approx(286.09, rel=1e-3)


def test_partition_function_monotonic_increasing():
    for species in ALL_SPECIES:
        molecule_id = MOLECULE_IDS[species]
        local_iso_id = GLOBAL_ISOTOPOLOGUES[species][0][1]
        t = np.linspace(150.0, 320.0, 20)  # atmospheric range
        q = partition_function(molecule_id, local_iso_id, t)
        assert np.all(np.diff(q) > 0), f"Q(T) not monotonic for {species}"


def test_partition_function_accepts_array_and_scalar():
    q_scalar = partition_function(1, 1, 250.0)
    q_array = partition_function(1, 1, np.array([250.0]))
    assert isinstance(q_scalar, float)
    assert q_array.shape == (1,)
    assert q_scalar == pytest.approx(float(q_array[0]))


def test_partition_function_clamps_outside_native_range_without_raising():
    # O3's tables only reach 1000 K natively; must clamp, not raise.
    q_1000 = partition_function(3, 1, 1000.0)
    q_2000 = partition_function(3, 1, 2000.0)
    assert q_1000 == pytest.approx(q_2000)  # flat extrapolation
    q_below = partition_function(1, 1, 10.0)  # below 70 K floor
    assert np.isfinite(q_below) and q_below > 0


def test_partition_function_unknown_isotopologue_raises():
    with pytest.raises(KeyError):
        partition_function(999, 1, 296.0)
