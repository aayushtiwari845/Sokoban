"""Solver correctness tests.  WRITTEN BEFORE THE SOLVER (spec operating rule 2).

The solver is the ground truth for every number in the paper, so these tests
are the project's load-bearing safety net.  The most important property is
**soundness of the "unsolvable" verdict**: a false "proven unsolvable" corrupts
the headline number invisibly and in a plausible direction.
"""

from __future__ import annotations

import pytest

from fixtures import (ALL_FIXTURES, SOLVABLE_FIXTURES, UNSOLVABLE_FIXTURES,
                      REAL_TEST_0, REAL_TEST_1, REAL_TEST_2, reference_solve)
from sokogen.data.solutions import replay_actions
from sokogen.solver.astar import solve
from sokogen.solver.grid import Grid

CAP = 400_000
TIME_CAP = 60.0


def _solve(grid, **kw):
    kw.setdefault("node_cap", CAP)
    kw.setdefault("time_cap_s", TIME_CAP)
    return solve(grid, **kw)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_solvable_fixtures_are_solved(name, grid, expected):
    res = _solve(grid)
    assert res.status == "solved", f"{name}: expected solved, got {res.status}"


@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_push_optimal_length_is_exact(name, grid, expected):
    """Push-optimal length must match exactly.

    The first four fixtures are hand-computed; the three real levels come from
    the independent reference solver in fixtures.py.
    """
    res = _solve(grid)
    assert res.push_length == expected, (
        f"{name}: push_length {res.push_length} != expected {expected}")


@pytest.mark.parametrize("name,grid", UNSOLVABLE_FIXTURES)
def test_unsolvable_fixtures_are_proven_unsolvable(name, grid):
    res = _solve(grid)
    assert res.status == "unsolvable", f"{name}: expected unsolvable, got {res.status}"


# ---------------------------------------------------------------------------
# Soundness: never claim "unsolvable" for something that is solvable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_deadlocks", [True, False])
@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_never_false_unsolvable(name, grid, expected, use_deadlocks):
    """With pruning on *and* off, a solvable level is never called unsolvable."""
    res = _solve(grid, use_deadlocks=use_deadlocks)
    assert res.status != "unsolvable", (
        f"{name}: FALSE UNSOLVABLE with use_deadlocks={use_deadlocks}")
    assert res.push_length == expected


@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_pruning_does_not_change_the_verdict(name, grid):
    """Deadlock pruning is an optimisation, never a semantic change.

    This is the test that catches an unsound freeze detector.
    """
    on = _solve(grid, use_deadlocks=True)
    off = _solve(grid, use_deadlocks=False)
    assert on.status == off.status, (
        f"{name}: pruning changed the verdict {off.status} -> {on.status}")
    assert on.push_length == off.push_length, (
        f"{name}: pruning changed push_length {off.push_length} -> {on.push_length}")


@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_agrees_with_independent_reference_solver(name, grid):
    """Differential test against a solver that shares no code or representation."""
    ref = reference_solve(grid, node_cap=CAP)
    if ref["status"] == "timeout":
        pytest.skip("reference solver hit its cap")
    res = _solve(grid)
    assert res.status == ref["status"], (
        f"{name}: solver says {res.status}, reference says {ref['status']}")
    assert res.push_length == ref["push_length"], (
        f"{name}: push_length {res.push_length} != reference {ref['push_length']}")


# ---------------------------------------------------------------------------
# The returned solution must actually solve the level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_returned_action_string_replays_to_solved(name, grid, expected):
    """A "solved" verdict must come with a move sequence that really works.

    Replayed by ``sokogen.data.solutions.replay_actions``, which is the same
    independent simulator used to validate the published Boxoban solutions.
    """
    res = _solve(grid)
    assert res.action_string is not None
    rep = replay_actions(grid, res.action_string)
    assert rep.solved, f"{name}: returned actions do not solve the level ({rep.failure})"
    assert rep.pushes == res.push_length, (
        f"{name}: replay counted {rep.pushes} pushes, solver reported {res.push_length}")
    assert rep.moves == res.move_length, (
        f"{name}: replay counted {rep.moves} moves, solver reported {res.move_length}")


@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_move_length_is_an_upper_bound_at_least_push_length(name, grid, expected):
    res = _solve(grid)
    assert res.move_length >= res.push_length


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,grid", ALL_FIXTURES)
def test_determinism(name, grid):
    """Same input => identical verdict, length and node count."""
    a = _solve(grid)
    b = _solve(grid)
    assert (a.status, a.push_length, a.move_length, a.nodes_expanded) == \
           (b.status, b.push_length, b.move_length, b.nodes_expanded)
    assert a.action_string == b.action_string


# ---------------------------------------------------------------------------
# Node cap semantics: a cap hit is a timeout, NEVER "unsolvable"
# ---------------------------------------------------------------------------
def test_node_cap_hit_reports_timeout_not_unsolvable():
    """Exhaustion proves unsolvability; a cap hit proves nothing (spec 4.6)."""
    res = solve(REAL_TEST_0, node_cap=1, time_cap_s=TIME_CAP)
    assert res.status == "timeout", f"cap hit reported as {res.status}"
    assert res.push_length is None


def test_time_cap_hit_reports_timeout_not_unsolvable():
    res = solve(REAL_TEST_0, node_cap=CAP, time_cap_s=0.0)
    assert res.status == "timeout"


@pytest.mark.parametrize("name,grid", UNSOLVABLE_FIXTURES)
def test_unsolvable_implies_the_open_set_actually_emptied(name, grid):
    """"unsolvable" is a proof by exhaustion, so it may only be returned when
    the search finished strictly inside its budget (spec 4.6)."""
    res = _solve(grid)
    assert res.status == "unsolvable"
    assert res.nodes_expanded < CAP, (
        f"{name}: claimed unsolvable while at/over the node cap")


# ---------------------------------------------------------------------------
# Symmetry: the 8 dihedral transforms preserve the verdict and the length
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_dihedral_transforms_preserve_solution_length(name, grid, expected):
    for k in range(8):
        t = Grid.from_string(grid).transform(k)
        res = _solve(t)
        assert res.status == "solved", f"{name} transform {k}: {res.status}"
        assert res.push_length == expected, (
            f"{name} transform {k}: push_length {res.push_length} != {expected}")


@pytest.mark.parametrize("name,grid", UNSOLVABLE_FIXTURES)
def test_dihedral_transforms_preserve_unsolvability(name, grid):
    for k in range(8):
        t = Grid.from_string(grid).transform(k)
        assert _solve(t).status == "unsolvable", f"{name} transform {k}"


# ---------------------------------------------------------------------------
# cost_mode="moves" -- the exact mode used to validate the reconstructed bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,grid,expected", SOLVABLE_FIXTURES)
def test_moves_mode_is_move_optimal_and_bounded_by_reconstruction(name, grid, expected):
    """Exact move-optimal length <= reconstructed move length (spec 4.6, unknown 6)."""
    exact = _solve(grid, cost_mode="moves")
    recon = _solve(grid, cost_mode="pushes")
    assert exact.status == "solved"
    assert exact.move_length <= recon.move_length, (
        f"{name}: exact move-optimal {exact.move_length} exceeds reconstructed "
        f"{recon.move_length}; the reconstruction is supposed to be an upper bound")
    rep = replay_actions(grid, exact.action_string)
    assert rep.solved and rep.moves == exact.move_length


def test_moves_mode_matches_published_move_optimal_lengths():
    """Their A* minimised moves, so our exact move mode must match its lengths.

    Published move-optimal lengths for unfiltered/test levels 0, 1 and 2.
    """
    for grid, published in ((REAL_TEST_0, 23), (REAL_TEST_1, 44), (REAL_TEST_2, 21)):
        res = _solve(grid, cost_mode="moves")
        assert res.status == "solved"
        assert res.move_length == published, (
            f"move-optimal {res.move_length} != published {published}")


# ---------------------------------------------------------------------------
# Goal state handling
# ---------------------------------------------------------------------------
def test_already_solved_state_costs_zero():
    """A state whose boxes already sit on the goals is solved in 0 pushes.

    The corpus alphabet has no '*', so this cannot be written as a grid string;
    it is built directly from bitmasks instead.
    """
    base = Grid.from_string(REAL_TEST_0)
    solved = Grid(walls=base.walls, goals=base.goals, boxes=base.goals,
                  player=base.player)
    res = _solve(solved)
    assert res.status == "solved"
    assert res.push_length == 0
    assert res.move_length == 0
