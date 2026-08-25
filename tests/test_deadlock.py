"""Deadlock-detection tests.

Both prunes must be **sound**: they may only fire on states from which no
solution exists.  An unsound prune produces false "proven unsolvable" verdicts,
which is the worst failure mode in this project -- it corrupts the headline
number invisibly and in a plausible direction.  Every test here is therefore
one-sided: we check that the detector does not fire on a live state, and we
verify firing states against the independent reference solver.
"""

from __future__ import annotations

import pytest

from fixtures import (CORNER_DEADLOCK, REAL_TEST_0, REAL_TEST_1, REAL_TEST_2,
                      TWO_BOXES_FROZEN, WALL_SLIDE_OK, g, reference_solve)
from sokogen.solver.deadlock import (dead_squares, is_freeze_deadlock,
                                     push_distance_maps)
from sokogen.solver.grid import Grid, bit, cells, idx


# ---------------------------------------------------------------------------
# Dead squares (spec 4.5.1)
# ---------------------------------------------------------------------------
def test_dead_squares_include_non_goal_corners():
    gr = Grid.from_string(CORNER_DEADLOCK)
    dead = dead_squares(gr.walls, gr.goals)
    assert dead & bit(idx(1, 1)), "the (1,1) corner should be a dead square"


def test_dead_squares_never_include_goals():
    """A goal is reachable from itself in zero pushes, so it is always live."""
    for grid in (REAL_TEST_0, REAL_TEST_1, REAL_TEST_2, CORNER_DEADLOCK):
        gr = Grid.from_string(grid)
        assert dead_squares(gr.walls, gr.goals) & gr.goals == 0


def test_dead_squares_never_include_walls():
    for grid in (REAL_TEST_0, REAL_TEST_1, REAL_TEST_2):
        gr = Grid.from_string(grid)
        assert dead_squares(gr.walls, gr.goals) & gr.walls == 0


def test_real_level_boxes_start_on_live_squares():
    """Real Boxoban levels are solvable, so no starting box may be on a dead
    square.  A failure here is an unsound dead-square computation."""
    for grid in (REAL_TEST_0, REAL_TEST_1, REAL_TEST_2):
        gr = Grid.from_string(grid)
        dead = dead_squares(gr.walls, gr.goals)
        assert dead & gr.boxes == 0, "a real level's box sits on a 'dead' square"


def test_wall_adjacent_square_is_live_when_it_can_slide_to_a_goal():
    """A box against a wall is NOT dead if it can slide along the wall to a
    goal.  Guards against conflating "against a wall" with "dead"."""
    gr = Grid.from_string(WALL_SLIDE_OK)
    dead = dead_squares(gr.walls, gr.goals)
    assert dead & gr.boxes == 0


# ---------------------------------------------------------------------------
# Push-distance maps (spec 4.4)
# ---------------------------------------------------------------------------
def test_push_distance_to_own_goal_is_zero():
    gr = Grid.from_string(REAL_TEST_0)
    maps = push_distance_maps(gr.walls, gr.goals)
    for gi, dmap in zip(cells(gr.goals), maps):
        assert dmap[gi] == 0


def test_push_distance_respects_walls():
    gr = Grid.from_string(REAL_TEST_0)
    maps = push_distance_maps(gr.walls, gr.goals)
    for dmap in maps:
        for w in cells(gr.walls):
            assert dmap[w] == float("inf")


def test_push_distance_is_at_least_manhattan():
    """Walls can only lengthen a path, never shorten it below Manhattan."""
    gr = Grid.from_string(REAL_TEST_0)
    maps = push_distance_maps(gr.walls, gr.goals)
    for gi, dmap in zip(cells(gr.goals), maps):
        gr_, gc_ = divmod(gi, 10)
        for c in range(100):
            if dmap[c] == float("inf"):
                continue
            r_, c_ = divmod(c, 10)
            assert dmap[c] >= abs(r_ - gr_) + abs(c_ - gc_)


# ---------------------------------------------------------------------------
# Freeze deadlock (spec 4.5.2)
# ---------------------------------------------------------------------------
def test_two_boxes_against_a_wall_are_frozen():
    gr = Grid.from_string(TWO_BOXES_FROZEN)
    dead = dead_squares(gr.walls, gr.goals)
    assert is_freeze_deadlock(gr.walls, gr.goals, gr.boxes, dead)


def test_freeze_does_not_fire_on_real_starting_positions():
    """Real levels are solvable, so their start state must not be a deadlock.

    This is the single most important soundness test in the file.
    """
    for grid in (REAL_TEST_0, REAL_TEST_1, REAL_TEST_2, WALL_SLIDE_OK):
        gr = Grid.from_string(grid)
        dead = dead_squares(gr.walls, gr.goals)
        assert not is_freeze_deadlock(gr.walls, gr.goals, gr.boxes, dead), \
            "freeze detector fired on a solvable real level"


def test_frozen_boxes_all_on_goals_is_not_a_deadlock():
    """If every immobile box already sits on a goal, the level is solved, not
    deadlocked (spec 4.5.2: "if any frozen box is *not on a goal*")."""
    gr = Grid.from_string(REAL_TEST_0)
    dead = dead_squares(gr.walls, gr.goals)
    assert not is_freeze_deadlock(gr.walls, gr.goals, gr.goals, dead)


def test_single_box_in_corner_on_a_goal_is_not_a_deadlock():
    grid = g(
        "##########",
        "#.       #",
        "#        #",
        "#   @    #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "##########",
    )
    gr = Grid.from_string(grid)
    boxes = bit(idx(1, 1))  # the box sits exactly on the corner goal
    dead = dead_squares(gr.walls, gr.goals)
    assert not is_freeze_deadlock(gr.walls, gr.goals, boxes, dead)


def test_freeze_terminates_on_cyclic_box_clusters():
    """A 2x2 block of boxes is mutually blocking; the recursion must terminate
    via the ``visiting`` set rather than recursing forever."""
    grid = g(
        "##########",
        "#        #",
        "#  $$    #",
        "#  $$    #",
        "#   @    #",
        "#       .#",
        "#      . #",
        "#     .  #",
        "#    .   #",
        "##########",
    )
    gr = Grid.from_string(grid)
    dead = dead_squares(gr.walls, gr.goals)
    assert is_freeze_deadlock(gr.walls, gr.goals, gr.boxes, dead)


@pytest.mark.parametrize("grid", [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2,
                                 WALL_SLIDE_OK, CORNER_DEADLOCK,
                                 TWO_BOXES_FROZEN])
def test_deadlock_verdicts_agree_with_reference_solver(grid):
    """If the detector calls the start state a deadlock, the independent
    reference solver must agree the level is unsolvable."""
    gr = Grid.from_string(grid)
    dead = dead_squares(gr.walls, gr.goals)
    fires = (dead & gr.boxes) != 0 or is_freeze_deadlock(
        gr.walls, gr.goals, gr.boxes, dead)
    if fires:
        assert reference_solve(grid)["status"] == "unsolvable", \
            "deadlock detector fired on a SOLVABLE level -- unsound prune"
