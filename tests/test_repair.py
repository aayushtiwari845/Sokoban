"""Repair tests (spec 9.2): idempotence and identity on already-valid grids."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import REAL_TEST_0, REAL_TEST_1, REAL_TEST_2
from sokogen.data.boxoban import (N_BOX, N_GOAL, N_PLAYER, is_structurally_valid,
                                  tile_counts)
from sokogen.decoding.repair import (BORDER, INTERIOR, grid_to_tiles,
                                     repair_grid, tiles_to_grid)

REAL = [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2]


@pytest.mark.parametrize("grid", REAL)
def test_repair_of_a_valid_grid_is_the_identity(grid):
    assert repair_grid(grid) == grid


@pytest.mark.parametrize("grid", REAL)
def test_repair_is_idempotent(grid):
    once = repair_grid(grid)
    assert repair_grid(once) == once


def test_grid_tiles_roundtrip():
    for grid in REAL:
        assert tiles_to_grid(grid_to_tiles(grid)) == grid


# ---------------------------------------------------------------------------
# Repairing broken inputs
# ---------------------------------------------------------------------------
def _broken(counts):
    """Build a grid string with the given (players, boxes, goals) counts."""
    body = [" "] * 100
    for i in range(100):
        r, c = divmod(i, 10)
        if r in (0, 9) or c in (0, 9):
            body[i] = "#"
    slots = [i for i in INTERIOR]
    k = 0
    for ch, n in zip("@$.", counts):
        for _ in range(n):
            body[slots[k]] = ch
            k += 1
    return "".join("".join(body[r * 10:(r + 1) * 10]) + "\n" for r in range(10))


@pytest.mark.parametrize("counts", [(0, 0, 0), (3, 9, 1), (1, 4, 0),
                                    (0, 4, 4), (5, 0, 7), (1, 12, 4)])
def test_repair_yields_a_structurally_valid_grid(counts):
    broken = _broken(counts)
    fixed = repair_grid(broken)
    assert is_structurally_valid(fixed), f"{counts} -> {fixed}"
    c = tile_counts(fixed)
    assert (c["@"], c["$"], c["."]) == (N_PLAYER, N_BOX, N_GOAL)


@pytest.mark.parametrize("counts", [(0, 0, 0), (3, 9, 1), (5, 0, 7)])
def test_repair_of_broken_grid_is_idempotent(counts):
    once = repair_grid(_broken(counts))
    assert repair_grid(once) == once


def test_repair_forces_the_wall_border():
    holed = _broken((1, 4, 4)).replace("\n", "")
    holed = list(holed)
    holed[5] = " "        # a hole in the top wall
    holed[90] = "$"       # a box on the bottom border
    grid = "".join("".join(holed[r * 10:(r + 1) * 10]) + "\n" for r in range(10))
    fixed = repair_grid(grid)
    assert is_structurally_valid(fixed)
    body = fixed.replace("\n", "")
    for i in BORDER:
        assert body[i] == "#"


def test_repair_never_places_specials_on_the_border():
    for counts in [(0, 0, 0), (1, 12, 4), (5, 0, 7)]:
        body = repair_grid(_broken(counts)).replace("\n", "")
        assert all(body[i] == "#" for i in BORDER)


def test_repair_handles_out_of_alphabet_characters():
    """Unconstrained decoding can emit '?' for special tokens."""
    junk = ("?" * 100)
    grid = "".join(junk[r * 10:(r + 1) * 10] + "\n" for r in range(10))
    fixed = repair_grid(grid)
    assert is_structurally_valid(fixed)


def test_repair_handles_wrong_length_input():
    assert is_structurally_valid(repair_grid("##########\n" * 3))
    assert is_structurally_valid(repair_grid("#" * 400))


# ---------------------------------------------------------------------------
# Probability-guided choices
# ---------------------------------------------------------------------------
def test_repair_keeps_the_highest_probability_boxes():
    """With 6 boxes and a quota of 4, the two least-confident ones must go."""
    body = [" "] * 100
    for i in range(100):
        r, c = divmod(i, 10)
        if r in (0, 9) or c in (0, 9):
            body[i] = "#"
    box_cells = INTERIOR[:6]
    for i in box_cells:
        body[i] = "$"
    body[INTERIOR[10]] = "@"
    for i in INTERIOR[20:24]:
        body[i] = "."
    grid = "".join("".join(body[r * 10:(r + 1) * 10]) + "\n" for r in range(10))

    probs = np.zeros((5, 100))
    # Make the last two box cells the least confident.
    for rank, i in enumerate(box_cells):
        probs[2, i] = 1.0 - 0.1 * rank
    fixed = repair_grid(grid, probs).replace("\n", "")
    kept = [i for i in box_cells if fixed[i] == "$"]
    assert kept == box_cells[:4]


def test_repair_promotes_the_highest_probability_cells_when_short():
    grid = _broken((1, 2, 4))
    probs = np.zeros((5, 100))
    favoured = [INTERIOR[40], INTERIOR[41]]
    for i in favoured:
        probs[2, i] = 1.0
    fixed = repair_grid(grid, probs).replace("\n", "")
    assert all(fixed[i] == "$" for i in favoured)
    assert is_structurally_valid(repair_grid(grid, probs))
