"""Metric tests: Wilson intervals, symmetry-aware novelty, distances."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import REAL_TEST_0, REAL_TEST_1, REAL_TEST_2
from sokogen.eval.metrics import (Proportion, controllability,
                                  dihedral_variants, flag_small_denominator,
                                  grids_to_array, mean_pairwise_distance,
                                  nearest_neighbour_distances, novelty,
                                  outcome_breakdown, structural_validity,
                                  two_proportion_z_test, wilson)
from sokogen.solver.grid import Grid

REAL = [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2]


# ---------------------------------------------------------------------------
# Wilson intervals
# ---------------------------------------------------------------------------
def test_wilson_contains_the_point_estimate():
    for k, n in [(0, 10), (1, 10), (5, 10), (10, 10), (500, 500), (3, 1000)]:
        p = wilson(k, n)
        assert p.lo <= p.p <= p.hi


def test_wilson_stays_inside_zero_one_at_the_extremes():
    """The reason for Wilson over the normal approximation."""
    lo = wilson(0, 20)
    hi = wilson(20, 20)
    assert lo.lo == 0.0 and lo.hi > 0.0
    assert hi.hi == 1.0 and hi.lo < 1.0


def test_wilson_narrows_with_more_data():
    small = wilson(50, 100)
    large = wilson(5000, 10000)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_wilson_handles_zero_samples():
    p = wilson(0, 0)
    assert p.n == 0 and p.fmt() == "n/a"


def test_two_proportion_z_test_detects_a_real_difference():
    res = two_proportion_z_test(90, 100, 10, 100)
    assert res["p_value"] < 1e-6
    assert res["diff_pct"] == pytest.approx(80.0)


def test_two_proportion_z_test_on_identical_rates():
    res = two_proportion_z_test(50, 100, 50, 100)
    assert res["p_value"] == pytest.approx(1.0)


def test_small_denominator_is_flagged():
    assert flag_small_denominator(wilson(4, 10)) is not None
    assert flag_small_denominator(wilson(200, 500)) is None


# ---------------------------------------------------------------------------
# Encoding and symmetry
# ---------------------------------------------------------------------------
def test_grids_to_array_shape_and_alphabet():
    arr = grids_to_array(REAL)
    assert arr.shape == (3, 100)
    assert arr.max() <= 4


def test_grids_to_array_tolerates_junk_characters():
    arr = grids_to_array(["?" * 100])
    assert arr.shape == (1, 100)


def test_dihedral_variants_match_the_solver_transforms():
    """The metric's symmetry group must be the solver's, not a private copy."""
    arr = grids_to_array([REAL_TEST_0])
    variants = dihedral_variants(arr)
    chars = "# $.@"
    got = set()
    for v in variants:
        s = "".join(chars[int(t)] for t in v[0])
        got.add("".join(s[r * 10:(r + 1) * 10] + "\n" for r in range(10)))
    expected = {Grid.from_string(REAL_TEST_0).transform(k).to_string()
                for k in range(8)}
    assert got == expected


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------
def test_a_training_level_is_not_novel():
    out = novelty([REAL_TEST_0], REAL, compute_nn=True)
    assert out["exact_copy"]["k"] == 1
    assert out["novel_pct"] == 0.0
    assert out["nn_distance"]["min"] == 0


def test_a_rotated_training_level_is_not_novel():
    """The whole point of symmetry-aware novelty (spec 10.2)."""
    for k in range(8):
        rotated = Grid.from_string(REAL_TEST_0).transform(k).to_string()
        out = novelty([rotated], REAL, compute_nn=False)
        assert out["exact_copy"]["k"] == 1, f"transform {k} counted as novel"


def test_an_unrelated_level_is_novel():
    other = ("##########\n" + "#@$     .#\n" + "#  $   . #\n" + "#   $  . #\n"
             + "#    $ . #\n" + "#        #\n" + "#        #\n" + "#        #\n"
             + "#        #\n" + "##########\n")
    out = novelty([other], REAL, compute_nn=True)
    assert out["exact_copy"]["k"] == 0
    assert out["novel_pct"] == 100.0
    assert out["nn_distance"]["min"] > 0


def test_retrieval_style_output_is_zero_percent_novel():
    """The retrieval baseline must score 0% novel by construction."""
    out = novelty(REAL * 4, REAL, compute_nn=False)
    assert out["novel_pct"] == 0.0


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def test_distance_to_self_is_zero():
    arr = grids_to_array([REAL_TEST_0])
    assert int(nearest_neighbour_distances(arr, arr)[0]) == 0


def test_mean_pairwise_distance_of_identical_levels_is_zero():
    arr = grids_to_array([REAL_TEST_0] * 5)
    assert mean_pairwise_distance(arr)["mean"] == 0.0


def test_mean_pairwise_distance_is_positive_for_distinct_levels():
    arr = grids_to_array(REAL)
    d = mean_pairwise_distance(arr)
    assert d["mean"] > 0
    assert d["n_pairs"] == 3


def test_distance_is_bounded_by_cell_count():
    arr = grids_to_array(REAL)
    assert mean_pairwise_distance(arr)["mean"] <= 100


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def test_outcome_breakdown_is_three_way_and_sums():
    statuses = ["solved"] * 7 + ["unsolvable"] * 2 + ["timeout"]
    out = outcome_breakdown(statuses)
    assert out["counts"] == {"solved": 7, "unsolvable": 2, "timeout": 1}
    assert sum(out["counts"].values()) == out["n"]
    assert out["solved"]["pct"] == pytest.approx(70.0)


def test_structural_validity_counts_only_valid_grids():
    bad = "#" * 100
    bad = "".join(bad[r * 10:(r + 1) * 10] + "\n" for r in range(10))
    p = structural_validity(REAL + [bad])
    assert p.k == 3 and p.n == 4


# ---------------------------------------------------------------------------
# Controllability
# ---------------------------------------------------------------------------
def test_controllability_perfect_match():
    req = [{"density": 1, "clustering": 0, "solution_length": 30}] * 10
    ach = [{"density": 1, "clustering": 0, "solution_length": 30}] * 10
    out = controllability(req, ach, [True] * 10)
    assert out["density"]["bin_accuracy"]["pct"] == 100.0
    assert out["solution_length"]["censoring_rate"] == 0.0


def test_controllability_reports_censoring_from_unsolved_levels():
    req = [{"density": 1, "solution_length": 20 + i} for i in range(10)]
    ach = [{"density": 1, "solution_length": 20 + i} for i in range(10)]
    mask = [True] * 4 + [False] * 6
    out = controllability(req, ach, mask)
    assert out["solution_length"]["n_with_achieved_length"] == 4
    assert out["solution_length"]["censoring_rate"] == pytest.approx(0.6)


def test_controllability_spearman_detects_monotone_relationship():
    req = [{"solution_length": v} for v in range(10, 60, 5)]
    ach = [{"solution_length": 2 * v + 3} for v in range(10, 60, 5)]
    out = controllability(req, ach, [True] * len(req))
    assert out["solution_length"]["spearman"] == pytest.approx(1.0)


def test_controllability_marks_which_attributes_are_grid_computable():
    """Density is readable off the grid; solution length needs search."""
    req = [{"density": 1, "solution_length": 30}] * 5
    out = controllability(req, req, [True] * 5)
    assert out["density"]["directly_computable_from_grid"] is True
    assert "directly_computable_from_grid" not in out["solution_length"]
