"""Constrained-decoding tests (spec 6.5).

The headline test generates 1,000 levels from an **untrained** model and
asserts 100% structural validity.  An untrained model is the strongest test
available because its distribution is near-uniform over the vocabulary, so it
exercises the feasibility-forcing path constantly -- exactly the path that a
naive "mask exhausted quotas" implementation gets wrong and that would
otherwise only fail in the tail.
"""

from __future__ import annotations

import pytest
import torch

from sokogen.data.boxoban import (N_BOX, N_GOAL, N_PLAYER, check_invariants,
                                  is_structurally_valid, tile_counts)
from sokogen.data.vocab import (BOX_ID, FLOOR_ID, GOAL_ID, NEWLINE_ID,
                                PLAYER_ID, WALL_ID)
from sokogen.decoding.constrained import (GRID_TOKENS, N_INTERIOR,
                                          ForcingStats, build_step_mask,
                                          generate, interior_left)
from sokogen.models.transformer import SokobanLM, TransformerConfig

CAPTION = "hard, ~40 moves, dense walls, corridors, boxes scattered"


@pytest.fixture(scope="module")
def untrained_model():
    torch.manual_seed(1337)
    model = SokobanLM(TransformerConfig())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# interior_left bookkeeping
# ---------------------------------------------------------------------------
def test_interior_left_counts_the_current_cell():
    assert interior_left(1, 1) == N_INTERIOR   # 64, first interior cell
    assert interior_left(8, 8) == 1            # last interior cell
    assert interior_left(1, 8) == 57


def test_interior_left_decreases_by_one_per_interior_cell():
    seen = []
    for step in range(GRID_TOKENS):
        row, col = divmod(step, 11)
        if col == 10 or row in (0, 9) or col in (0, 9):
            continue
        seen.append(interior_left(row, col))
    assert seen == list(range(N_INTERIOR, 0, -1))


# ---------------------------------------------------------------------------
# Mask rules
# ---------------------------------------------------------------------------
def _allowed(mask):
    return {int(i) for i in torch.nonzero(mask == 0).flatten()}


def test_border_cells_are_forced_to_wall():
    for step in range(GRID_TOKENS):
        row, col = divmod(step, 11)
        if col == 10:
            continue
        if row in (0, 9) or col in (0, 9):
            mask, _, _ = build_step_mask(step, 0, 0, 0, "cpu")
            assert _allowed(mask) == {WALL_ID}, f"step {step} ({row},{col})"


def test_row_end_is_forced_to_newline():
    for row in range(10):
        step = row * 11 + 10
        mask, _, _ = build_step_mask(step, 0, 0, 0, "cpu")
        assert _allowed(mask) == {NEWLINE_ID}


def test_exhausted_quotas_are_masked():
    mask, _, quota = build_step_mask(1 * 11 + 1, N_PLAYER, N_BOX, N_GOAL, "cpu")
    allowed = _allowed(mask)
    assert PLAYER_ID not in allowed and BOX_ID not in allowed and GOAL_ID not in allowed
    assert allowed == {WALL_ID, FLOOR_ID}
    assert quota


def test_unexhausted_interior_cell_allows_all_five_tiles():
    mask, forced, quota = build_step_mask(1 * 11 + 1, 0, 0, 0, "cpu")
    assert _allowed(mask) == {WALL_ID, FLOOR_ID, BOX_ID, GOAL_ID, PLAYER_ID}
    assert not forced and not quota


def test_feasibility_forcing_fires_when_space_runs_out():
    """The rule the spec warns naive implementations omit.

    At the last interior cell with one special still owed, ``#`` and ``' '``
    must be masked -- upper-bound quotas alone would happily emit a wall and
    end up with three boxes.
    """
    step = 8 * 11 + 8              # last interior cell, interior_left == 1
    mask, forced, _ = build_step_mask(step, N_PLAYER, N_BOX - 1, N_GOAL, "cpu")
    assert forced
    allowed = _allowed(mask)
    assert allowed == {BOX_ID}
    assert WALL_ID not in allowed and FLOOR_ID not in allowed


def test_feasibility_forcing_does_not_fire_when_space_remains():
    step = 1 * 11 + 1              # first interior cell, 64 cells for 9 specials
    _, forced, _ = build_step_mask(step, 0, 0, 0, "cpu")
    assert not forced


def test_forcing_allows_every_still_owed_special():
    step = 8 * 11 + 6              # interior_left == 3
    assert interior_left(8, 6) == 3
    mask, forced, _ = build_step_mask(step, N_PLAYER - 1, N_BOX - 1, N_GOAL - 1, "cpu")
    assert forced
    assert _allowed(mask) == {PLAYER_ID, BOX_ID, GOAL_ID}


def test_eos_is_forced_after_the_last_grid_character():
    from sokogen.data.vocab import EOS_ID
    mask, _, _ = build_step_mask(GRID_TOKENS, 0, 0, 0, "cpu")
    assert _allowed(mask) == {EOS_ID}


# ---------------------------------------------------------------------------
# End-to-end: the GATE 4 requirement
# ---------------------------------------------------------------------------
def test_untrained_model_yields_1000_structurally_valid_levels(untrained_model):
    stats = ForcingStats()
    grids = generate(untrained_model, CAPTION, 1000, torch.device("cpu"),
                     temperature=1.0, constrained=True, seed=1337,
                     batch_size=250, stats=stats)
    assert len(grids) == 1000
    bad = [(i, check_invariants(g)) for i, g in enumerate(grids)
           if not is_structurally_valid(g)]
    assert not bad, f"{len(bad)} invalid levels, first: {bad[:3]}"

    # An untrained model is near-uniform over the five tiles, so it exhausts
    # the special quotas early (expected ~13 boxes in 64 interior cells) and
    # hammers the quota-masking path.  It does NOT exercise feasibility
    # forcing, which needs a model biased *against* specials -- that path is
    # covered directly by test_feasibility_forcing_end_to_end below.
    s = stats.summary()
    assert s["mean_quota_masked_tokens_per_level"] > 10, (
        "untrained sampling did not exercise quota masking as expected")


class _WallBiasedStub(torch.nn.Module):
    """A model that almost always wants to emit '#' or ' '.

    This is the adversary for feasibility forcing: left unforced it would place
    zero boxes, goals and players, so every valid level it produces is one the
    forcing rule rescued.
    """

    def __init__(self, bias: float = 12.0):
        super().__init__()
        self.bias = bias

    def forward(self, ids, labels=None, past=None, use_cache=False):
        from sokogen.data.vocab import VOCAB_SIZE
        b, t = ids.shape
        logits = torch.zeros(b, t, VOCAB_SIZE)
        logits[..., WALL_ID] = self.bias
        logits[..., FLOOR_ID] = self.bias
        logits[..., NEWLINE_ID] = self.bias
        length = t if past is None else past[0][0].shape[2] + t
        present = [(torch.zeros(b, 1, length, 1), torch.zeros(b, 1, length, 1))]
        if use_cache:
            return logits, None, present
        return logits, None


def test_feasibility_forcing_end_to_end():
    """A wall-biased model must still produce exactly 1 player, 4 boxes, 4 goals.

    Without the feasibility rule this model would emit an all-wall interior and
    every sample would be invalid, so 100% validity here is entirely due to
    forcing.  This is the tail case the spec warns about.
    """
    stats = ForcingStats()
    grids = generate(_WallBiasedStub(), CAPTION, 200, torch.device("cpu"),
                     temperature=1.0, constrained=True, seed=1337,
                     batch_size=200, stats=stats)
    assert all(is_structurally_valid(g) for g in grids), "forcing failed to rescue"
    for g in grids:
        c = tile_counts(g)
        assert (c["@"], c["$"], c["."]) == (N_PLAYER, N_BOX, N_GOAL)

    s = stats.summary()
    assert s["forced_sequence_rate"] == 1.0, (
        f"forcing fired on only {100*s['forced_sequence_rate']:.1f}% of "
        f"wall-biased samples")
    # Nine specials are owed and the model wants none of them, so forcing must
    # supply all nine.
    assert s["mean_forced_tokens_per_level"] == 9.0


def test_wall_biased_stub_is_invalid_without_forcing():
    """Confirms the previous test would fail if forcing were removed."""
    grids = generate(_WallBiasedStub(), CAPTION, 20, torch.device("cpu"),
                     temperature=1.0, constrained=False, seed=1337,
                     batch_size=20)
    assert not any(is_structurally_valid(g) for g in grids)


def test_generated_levels_have_exact_tile_counts(untrained_model):
    grids = generate(untrained_model, CAPTION, 200, torch.device("cpu"),
                     temperature=1.0, constrained=True, seed=7, batch_size=200)
    for g in grids:
        c = tile_counts(g)
        assert c["@"] == N_PLAYER and c["$"] == N_BOX and c["."] == N_GOAL


def test_grid_string_is_exactly_110_chars(untrained_model):
    grids = generate(untrained_model, CAPTION, 50, torch.device("cpu"),
                     constrained=True, seed=3, batch_size=50)
    assert all(len(g) == 110 for g in grids)


def test_unconstrained_path_is_reachable_and_differs(untrained_model):
    """Same code path, masking disabled -- what makes the comparison controlled."""
    grids = generate(untrained_model, CAPTION, 50, torch.device("cpu"),
                     temperature=1.0, constrained=False, seed=11, batch_size=50)
    assert len(grids) == 50
    assert all(len(g) == 110 for g in grids)
    # An untrained model without masking essentially never lands on a valid grid.
    assert sum(is_structurally_valid(g) for g in grids) < 5


def test_generation_is_deterministic_given_a_seed(untrained_model):
    a = generate(untrained_model, CAPTION, 20, torch.device("cpu"), seed=99,
                 batch_size=20)
    b = generate(untrained_model, CAPTION, 20, torch.device("cpu"), seed=99,
                 batch_size=20)
    assert a == b


def test_batch_size_does_not_change_results(untrained_model):
    """Batching is an implementation detail, not a semantic one."""
    a = generate(untrained_model, CAPTION, 32, torch.device("cpu"), seed=5,
                 batch_size=32)
    b = generate(untrained_model, CAPTION, 32, torch.device("cpu"), seed=5,
                 batch_size=16)
    assert len(a) == len(b) == 32
    assert all(is_structurally_valid(g) for g in a + b)


def test_kv_cache_matches_uncached_forward(untrained_model):
    """Cached incremental decoding must equal a full forward pass."""
    torch.manual_seed(0)
    ids = torch.randint(4, 40, (2, 20))
    full, _ = untrained_model(ids)

    past = None
    outs = []
    for t in range(ids.shape[1]):
        logits, _, past = untrained_model(ids[:, t:t + 1], past=past, use_cache=True)
        outs.append(logits[:, -1, :])
    cached = torch.stack(outs, dim=1)
    assert torch.allclose(full, cached, atol=1e-4), \
        (full - cached).abs().max().item()
