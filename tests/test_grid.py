"""Grid representation, bitmask encoding and dihedral symmetry tests."""

from __future__ import annotations

import pytest

from fixtures import ALL_FIXTURES, REAL_TEST_0, REAL_TEST_1, REAL_TEST_2
from sokogen.data.boxoban import GRID_STR_LEN, is_structurally_valid
from sokogen.solver.grid import (DIHEDRAL_N, Grid, bit, cells, idx,
                                 player_reachable, rc)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def test_index_roundtrip():
    for i in range(100):
        r, c = rc(i)
        assert idx(r, c) == i
        assert 0 <= r < 10 and 0 <= c < 10


def test_cells_and_bit_are_inverse():
    mask = bit(0) | bit(45) | bit(99)
    assert cells(mask) == [0, 45, 99]


# ---------------------------------------------------------------------------
# Parsing / round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_string_roundtrip(name, grid):
    assert Grid.from_string(grid).to_string() == grid


@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_masks_are_disjoint(name, grid):
    gr = Grid.from_string(grid)
    assert gr.walls & gr.boxes == 0, "a box sits inside a wall"
    assert gr.walls & bit(gr.player) == 0, "the player sits inside a wall"
    assert gr.boxes & bit(gr.player) == 0, "the player sits inside a box"
    # Goals may coincide with nothing else in the corpus (no '*' or '+'),
    # verified in Phase 1.
    assert gr.goals & gr.boxes == 0
    assert gr.goals & bit(gr.player) == 0


def test_parse_rejects_malformed_input():
    with pytest.raises(ValueError):
        Grid.from_string("too short")


# ---------------------------------------------------------------------------
# Dihedral group: 4 rotations x optional flip
# ---------------------------------------------------------------------------
def test_there_are_eight_transforms():
    assert DIHEDRAL_N == 8


@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_transform_zero_is_identity(name, grid):
    gr = Grid.from_string(grid)
    assert gr.transform(0) == gr


@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_transforms_preserve_tile_counts(name, grid):
    gr = Grid.from_string(grid)
    for k in range(DIHEDRAL_N):
        t = gr.transform(k)
        assert bin(t.walls).count("1") == bin(gr.walls).count("1")
        assert bin(t.boxes).count("1") == bin(gr.boxes).count("1")
        assert bin(t.goals).count("1") == bin(gr.goals).count("1")
        assert len(t.to_string()) == GRID_STR_LEN


@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_transforms_are_distinct_or_symmetric(name, grid):
    """The 8 transforms form a group: applying all 8 to any grid yields a set
    whose size divides 8."""
    gr = Grid.from_string(grid)
    variants = {gr.transform(k).to_string() for k in range(DIHEDRAL_N)}
    assert DIHEDRAL_N % len(variants) == 0


@pytest.mark.parametrize("grid", [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2])
def test_transforms_preserve_structural_validity(grid):
    gr = Grid.from_string(grid)
    for k in range(DIHEDRAL_N):
        assert is_structurally_valid(gr.transform(k).to_string())


def test_canonical_form_is_transform_invariant():
    """Two grids related by a symmetry share a canonical form.

    This is what makes the novelty metric symmetry-aware: a 90-degree rotation
    of a training level must not count as novel.
    """
    gr = Grid.from_string(REAL_TEST_0)
    canon = gr.canonical_string()
    for k in range(DIHEDRAL_N):
        assert gr.transform(k).canonical_string() == canon


def test_canonical_form_differs_for_unrelated_levels():
    a = Grid.from_string(REAL_TEST_0).canonical_string()
    b = Grid.from_string(REAL_TEST_1).canonical_string()
    assert a != b


# ---------------------------------------------------------------------------
# Player reachability and canonicalisation (spec 4.2)
# ---------------------------------------------------------------------------
def test_player_reachable_excludes_walls_and_boxes():
    gr = Grid.from_string(REAL_TEST_0)
    region = player_reachable(gr.walls, gr.boxes, gr.player)
    assert region & gr.walls == 0
    assert region & gr.boxes == 0
    assert region & bit(gr.player) != 0


def test_canonical_player_is_min_of_region():
    gr = Grid.from_string(REAL_TEST_0)
    region = player_reachable(gr.walls, gr.boxes, gr.player)
    assert gr.canonical_player() == min(cells(region))


@pytest.mark.parametrize("grid", [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2])
def test_canonical_player_identical_across_the_region(grid):
    """Every player start in one region must canonicalise to the same value --
    this is the whole point of the state key (spec 4.2)."""
    gr = Grid.from_string(grid)
    region = cells(player_reachable(gr.walls, gr.boxes, gr.player))
    canon = {Grid(gr.walls, gr.goals, gr.boxes, p).canonical_player() for p in region}
    assert len(canon) == 1
    assert canon.pop() == min(region)


def test_canonicalisation_collapses_a_large_region_to_one_key():
    """Many distinct player positions must collapse to a single state key.

    Uses an open room so the region is large by construction.  The *magnitude*
    of the saving on real levels is measured empirically by
    ``scripts/02_validate_solver.py`` rather than asserted here: on the 1,000
    real test levels the start-state region averages ~19 cells (median 22.5,
    max 46), and 150 of those levels start with the player walled into a single
    cell -- so the spec's assumed 30-60x is optimistic for Boxoban.
    """
    from fixtures import g
    open_room = g(
        "##########",
        "#        #",
        "#        #",
        "#  $     #",
        "#   @    #",
        "#     .  #",
        "#        #",
        "#        #",
        "#        #",
        "##########",
    )
    gr = Grid.from_string(open_room)
    region = cells(player_reachable(gr.walls, gr.boxes, gr.player))
    assert len(region) > 30, f"expected a large open region, got {len(region)}"
    keys = {Grid(gr.walls, gr.goals, gr.boxes, p).canonical_player() for p in region}
    assert keys == {min(region)}, "region members did not collapse to one key"
