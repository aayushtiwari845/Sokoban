"""Caption feature, vocabulary and prompt-suite tests.

These feed every conditioning number in the paper: a bug here would silently
corrupt controllability without failing anything else.
"""

from __future__ import annotations

import pytest

from fixtures import REAL_TEST_0, REAL_TEST_1, REAL_TEST_2, g
from sokogen.data import vocab
from sokogen.data.boxoban import GRID_STR_LEN
from sokogen.data.captions import (CAPTION_VEC_DIM, CLUSTERING_WORDS,
                                   CONNECTIVITY_WORDS, DENSITY_WORDS,
                                   DIFFICULTY_WORDS, N_INTERIOR, CaptionBins,
                                   bin_indices, box_clustering, caption_text,
                                   caption_vec, difficulty_bin, features,
                                   fit_bins, max_caption_length,
                                   mean_floor_degree, raw_features,
                                   round_length, wall_density)
from sokogen.eval.prompts import build_full_suite

BINS = CaptionBins(difficulty_terciles=(26.0, 35.0), density_median=0.5,
                   degree_median=2.8889, clustering_median=3.5,
                   length_min=4.0, length_max=130.0)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_interior_is_64_cells():
    assert N_INTERIOR == 64


def test_wall_density_of_an_empty_room_is_zero():
    empty = g("##########", "#        #", "#        #", "#        #",
              "#   @    #", "#  $$$$  #", "#  ....  #", "#        #",
              "#        #", "##########")
    assert wall_density(_G(empty)) == 0.0


def test_wall_density_of_a_solid_interior_is_one():
    solid = g(*(["##########"] * 10))
    from sokogen.solver.grid import Grid
    gr = Grid(walls=(1 << 100) - 1, goals=0, boxes=0, player=0)
    assert wall_density(gr) == 1.0


def _G(grid_str):
    from sokogen.solver.grid import Grid
    return Grid.from_string(grid_str)


@pytest.mark.parametrize("grid", [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2])
def test_wall_density_in_unit_range(grid):
    assert 0.0 <= wall_density(_G(grid)) <= 1.0


def test_floor_degree_of_open_room_is_near_four():
    empty = g("##########", "#        #", "#        #", "#        #",
              "#   @    #", "#  $$$$  #", "#  ....  #", "#        #",
              "#        #", "##########")
    d = mean_floor_degree(_G(empty))
    assert 3.0 < d <= 4.0, d


def test_floor_degree_of_a_corridor_is_near_two():
    corridor = g("##########", "#@$     .#", "##########", "##########",
                 "##########", "##########", "##########", "##########",
                 "##########", "##########")
    d = mean_floor_degree(_G(corridor))
    assert d < 2.1, d


def test_box_clustering_of_adjacent_boxes_is_small():
    clustered = g("##########", "#        #", "#  $$    #", "#  $$    #",
                  "#   @    #", "#  ....  #", "#        #", "#        #",
                  "#        #", "##########")
    scattered = g("##########", "#$      $#", "#        #", "#        #",
                  "#   @    #", "#  ....  #", "#        #", "#        #",
                  "#$      $#", "##########")
    assert box_clustering(_G(clustered)) < box_clustering(_G(scattered))


def test_raw_features_accepts_string_or_grid():
    a = raw_features(REAL_TEST_0)
    b = raw_features(_G(REAL_TEST_0))
    assert a == b


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def test_difficulty_bins_split_at_the_terciles():
    assert difficulty_bin(10, BINS) == 0
    assert difficulty_bin(26, BINS) == 0     # boundary is inclusive below
    assert difficulty_bin(27, BINS) == 1
    assert difficulty_bin(35, BINS) == 1
    assert difficulty_bin(36, BINS) == 2


def test_round_length_rounds_to_nearest_five():
    assert round_length(41) == 40
    assert round_length(43) == 45
    assert round_length(23) == 25
    assert round_length(1) == 5   # never below 5


def test_fit_bins_recovers_known_percentiles():
    rows = [{"solution_length": float(i), "wall_density": i / 100,
             "floor_degree": 2 + i / 100, "box_clustering": i / 10}
            for i in range(1, 101)]
    bins = fit_bins(rows)
    assert 33 <= bins.difficulty_terciles[0] <= 35
    assert 66 <= bins.difficulty_terciles[1] <= 68
    assert bins.length_min == 1.0 and bins.length_max == 100.0


# ---------------------------------------------------------------------------
# Caption text and vector agree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grid", [REAL_TEST_0, REAL_TEST_1, REAL_TEST_2])
def test_caption_text_and_vector_encode_the_same_bins(grid):
    feats = features(grid, 41)
    b = bin_indices(feats, BINS)
    text = caption_text(feats, BINS)
    vec = caption_vec(feats, BINS)

    assert DIFFICULTY_WORDS[b["difficulty"]] in text
    assert DENSITY_WORDS[b["density"]] in text
    assert CONNECTIVITY_WORDS[b["connectivity"]] in text
    assert CLUSTERING_WORDS[b["clustering"]] in text

    assert len(vec) == CAPTION_VEC_DIM
    assert vec[b["difficulty"]] == 1.0
    assert sum(vec[0:3]) == 1.0
    assert vec[4 + b["density"]] == 1.0
    assert vec[6 + b["connectivity"]] == 1.0
    assert vec[8 + b["clustering"]] == 1.0


def test_caption_vector_length_channel_is_normalised():
    feats = features(REAL_TEST_0, 4)
    assert caption_vec(feats, BINS)[3] == pytest.approx(0.0)
    feats = features(REAL_TEST_0, 130)
    assert caption_vec(feats, BINS)[3] == pytest.approx(1.0)


def test_caption_vector_length_channel_is_clipped():
    feats = features(REAL_TEST_0, 400)
    assert caption_vec(feats, BINS)[3] == 1.0


def test_caption_text_quotes_the_rounded_length():
    feats = features(REAL_TEST_0, 41)
    assert "~40 moves" in caption_text(feats, BINS)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def test_vocab_covers_every_caption_character():
    for d in range(3):
        for de in range(2):
            for co in range(2):
                for cl in range(2):
                    from sokogen.data.captions import caption_text_from_bins
                    text = caption_text_from_bins(d, 130, de, co, cl)
                    vocab.encode_text(text)  # raises if a char is missing


def test_vocab_rejects_out_of_alphabet_characters():
    with pytest.raises(ValueError):
        vocab.encode_text("UPPERCASE")


def test_every_caption_fits_max_len():
    assert max_caption_length() <= vocab.MAX_CAPTION_LEN
    assert vocab.MAX_LEN == 1 + vocab.MAX_CAPTION_LEN + 1 + GRID_STR_LEN + 1


def test_encode_example_roundtrips_the_grid():
    from sokogen.data.captions import caption_text_from_bins
    text = caption_text_from_bins(2, 40, 1, 0, 1)
    ids = vocab.encode_example(text, REAL_TEST_0)
    assert vocab.grid_from_ids(ids) == REAL_TEST_0
    assert len(ids) <= vocab.MAX_LEN


def test_encode_example_rejects_a_wrong_length_grid():
    with pytest.raises(ValueError):
        vocab.encode_example("easy, ~10 moves, open room, corridors, "
                             "boxes clustered", "##########\n")


def test_prompt_prefix_ends_with_sep():
    ids = vocab.encode_prompt("easy, ~10 moves, open room, corridors, "
                              "boxes clustered")
    assert ids[0] == vocab.BOS_ID
    assert ids[-1] == vocab.SEP_ID


def test_pad_to_pads_with_pad_id():
    ids = vocab.encode_prompt("easy, ~10 moves, open room, corridors, "
                              "boxes clustered")
    padded = vocab.pad_to(ids)
    assert len(padded) == vocab.MAX_LEN
    assert set(padded[len(ids):]) == {vocab.PAD_ID}


def test_pad_to_rejects_overlong_sequences():
    with pytest.raises(ValueError):
        vocab.pad_to([0] * (vocab.MAX_LEN + 1))


# ---------------------------------------------------------------------------
# Prompt suite
# ---------------------------------------------------------------------------
def test_suite_has_600_in_distribution_samples():
    suite = build_full_suite(BINS)
    assert sum(p.n_samples for p in suite["in_distribution"]) == 600
    assert len(suite["in_distribution"]) == 40


def test_suite_varies_every_attribute_independently():
    """A confounded suite cannot support per-attribute controllability."""
    suite = build_full_suite(BINS)
    prompts = suite["in_distribution"]
    for attr in ("density", "connectivity", "clustering"):
        for value in (0, 1):
            subset = [p for p in prompts if getattr(p, attr) == value]
            assert len(subset) == 20
            # Every length appears with both values of every other attribute.
            assert len({p.requested_length for p in subset}) == 5


def test_ood_suite_requests_lengths_beyond_training():
    suite = build_full_suite(BINS)
    ood = suite["out_of_distribution"]
    assert all(p.ood for p in ood)
    assert {p.requested_length for p in ood} == {90, 110, 150}
    assert all(p.requested_length > BINS.difficulty_terciles[1] for p in ood)


def test_suite_captions_are_encodable():
    suite = build_full_suite(BINS)
    for group in suite.values():
        for p in group:
            ids = vocab.encode_prompt(p.caption_text)
            assert len(ids) <= vocab.MAX_LEN
            assert len(p.caption_vec) == CAPTION_VEC_DIM
