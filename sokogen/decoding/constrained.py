"""Grid-guard constrained decoding (spec 6.4).

A manual sampling loop, deliberately not ``model.generate()`` and not a
``LogitsProcessor``: per-sequence constraint state is the fiddly part and
HuggingFace's API makes it worse for no benefit over 110 tokens.

Constraint state per sequence: ``row``, ``col``, ``n_player``, ``n_box``,
``n_goal``, ``interior_left``.

Masking rules
-------------
* Border cells (row 0, row 9, col 0, col 9): force ``#``.
* After 10 tile characters in a row: force ``\\n``.
* Interior cell: mask any tile whose quota is exhausted (``@`` once one player
  is placed, ``$`` after four boxes, ``.`` after four goals).
* **Feasibility forcing.**  Let
  ``remaining_specials = (1 - n_player) + (4 - n_box) + (4 - n_goal)``.  When
  ``interior_left == remaining_specials``, mask ``#`` and ``' '`` so only
  specials can still be emitted.  Masking upper bounds alone enforces "at most
  four boxes", not "exactly four" -- with 9 specials in 64 interior cells this
  rarely binds, which is exactly the trap: it looks correct on a handful of
  samples and fails in the tail.
* After the last grid character: force ``<eos>``.

Limitation stated in the paper
------------------------------
Mask-then-renormalise at each step does **not** sample from the model's
distribution conditioned on the valid set.  It is a greedy local approximation
to it, and it shifts the sampling distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from ..data.boxoban import GRID_H, GRID_W, N_BOX, N_GOAL, N_PLAYER
from ..data.vocab import (BOX_ID, EOS_ID, FLOOR_ID, GOAL_ID, NEWLINE_ID,
                          PAD_ID, PLAYER_ID, VOCAB_SIZE, WALL_ID,
                          encode_prompt, itos)

# Total interior (non-border) cells in a bordered 10x10 grid.
N_INTERIOR = (GRID_H - 2) * (GRID_W - 2)  # 64
GRID_TOKENS = GRID_H * (GRID_W + 1)       # 110
TOTAL_SPECIALS = N_PLAYER + N_BOX + N_GOAL  # 9

NEG = float("-inf")

# Rendering table for emitted tokens.  Special tokens become a single "?" so an
# unconstrained sample is still exactly 110 characters long and fails
# ``check_invariants`` with "characters outside alphabet" -- rendering "<pad>"
# literally would change the string length and confuse the validity report.
INVALID_CHAR = "?"
RENDER: List[str] = [
    tok if len(tok) == 1 else INVALID_CHAR for tok in itos
]


@dataclass
class ForcingStats:
    """Instrumentation for how often the model needed rescuing (spec 6.4).

    A free, reportable measure of how well the model learned the counting
    constraints on its own: if forcing never fires, the model is already
    counting correctly.
    """

    n_sequences: int = 0
    n_sequences_forced: int = 0
    forced_tokens: List[int] = field(default_factory=list)
    quota_masked_tokens: List[int] = field(default_factory=list)

    def summary(self) -> dict:
        import numpy as np
        ft = np.array(self.forced_tokens, dtype=float) if self.forced_tokens else \
            np.zeros(1)
        qm = np.array(self.quota_masked_tokens, dtype=float) if \
            self.quota_masked_tokens else np.zeros(1)
        return {
            "n_sequences": self.n_sequences,
            "n_sequences_with_forcing": self.n_sequences_forced,
            "forced_sequence_rate": (self.n_sequences_forced / self.n_sequences
                                     if self.n_sequences else 0.0),
            "mean_forced_tokens_per_level": float(ft.mean()),
            "max_forced_tokens_per_level": float(ft.max()),
            "mean_quota_masked_tokens_per_level": float(qm.mean()),
        }


def interior_left(row: int, col: int) -> int:
    """Interior cells still to be emitted, counting the current one.

    Only meaningful when ``(row, col)`` is itself an interior cell.
    """
    return (GRID_W - 1 - col) + (GRID_H - 2 - row) * (GRID_W - 2)


def _is_border(row: int, col: int) -> bool:
    return row == 0 or row == GRID_H - 1 or col == 0 or col == GRID_W - 1


def build_step_mask(step: int, n_player: int, n_box: int, n_goal: int,
                    device) -> tuple:
    """Additive logit mask for one decoding step of the grid.

    ``step`` counts grid tokens emitted so far, 0..110.  Returns
    ``(mask[V], forced, quota_masked)`` where ``forced`` is True when
    feasibility forcing removed ``#``/``' '`` and ``quota_masked`` is True when
    an exhausted quota removed a tile.
    """
    mask = torch.full((VOCAB_SIZE,), NEG, device=device)

    if step >= GRID_TOKENS:
        mask[EOS_ID] = 0.0
        return mask, False, False

    row, col = divmod(step, GRID_W + 1)
    if col == GRID_W:
        mask[NEWLINE_ID] = 0.0            # end of a row
        return mask, False, False

    if _is_border(row, col):
        mask[WALL_ID] = 0.0               # the wall border is mandatory
        return mask, False, False

    allowed = [WALL_ID, FLOOR_ID]
    quota_masked = False
    if n_player < N_PLAYER:
        allowed.append(PLAYER_ID)
    else:
        quota_masked = True
    if n_box < N_BOX:
        allowed.append(BOX_ID)
    else:
        quota_masked = True
    if n_goal < N_GOAL:
        allowed.append(GOAL_ID)
    else:
        quota_masked = True

    remaining_specials = ((N_PLAYER - n_player) + (N_BOX - n_box)
                          + (N_GOAL - n_goal))
    left = interior_left(row, col)
    forced = False
    if left <= remaining_specials:
        # Every remaining interior cell must carry a special, so drop the fillers.
        allowed = [a for a in allowed
                   if a not in (WALL_ID, FLOOR_ID)] or [WALL_ID, FLOOR_ID]
        forced = True

    for a in allowed:
        mask[a] = 0.0
    return mask, forced, quota_masked


@torch.no_grad()
def generate(model, caption: str, n: int, device, temperature: float = 1.0,
             constrained: bool = True, seed: Optional[int] = None,
             batch_size: int = 256, stats: Optional[ForcingStats] = None,
             top_p: float = 1.0) -> List[str]:
    """Sample ``n`` grids for one caption.

    With ``constrained=False`` the identical code path runs without masking,
    which is what makes the constrained/unconstrained comparison a controlled
    one -- the only difference is the mask.

    Returns raw grid strings; structural validity is *not* enforced when
    ``constrained=False`` and is reported as a metric instead.
    """
    model.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    prompt = encode_prompt(caption)
    out: List[str] = []

    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        ids = torch.tensor(prompt, device=device, dtype=torch.long
                           ).unsqueeze(0).expand(b, -1).contiguous()

        n_player = torch.zeros(b, dtype=torch.long, device=device)
        n_box = torch.zeros(b, dtype=torch.long, device=device)
        n_goal = torch.zeros(b, dtype=torch.long, device=device)
        forced_counts = torch.zeros(b, dtype=torch.long, device=device)
        quota_counts = torch.zeros(b, dtype=torch.long, device=device)
        emitted = [[] for _ in range(b)]

        past = None
        cur = ids
        for step in range(GRID_TOKENS):
            logits, _, past = model(cur, past=past, use_cache=True)
            logits = logits[:, -1, :].float()

            if constrained:
                # Group sequences by their (n_player, n_box, n_goal) state so
                # each distinct mask is built once per step, not once per row.
                keys = (n_player * 100 + n_box * 10 + n_goal)
                for key in torch.unique(keys):
                    sel = keys == key
                    p = int(key // 100)
                    bx = int((key // 10) % 10)
                    gl = int(key % 10)
                    mask, forced, quota = build_step_mask(step, p, bx, gl, device)
                    logits[sel] = logits[sel] + mask
                    if forced:
                        forced_counts[sel] += 1
                    if quota:
                        quota_counts[sel] += 1

            if temperature <= 0:
                nxt = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    probs = _top_p_filter(probs, top_p)
                nxt = torch.multinomial(probs, num_samples=1, generator=gen
                                        ).squeeze(-1)

            # One device->host transfer per step, not one per sequence.
            for i, tok in enumerate(nxt.tolist()):
                emitted[i].append(RENDER[tok])
            n_player += (nxt == PLAYER_ID).long()
            n_box += (nxt == BOX_ID).long()
            n_goal += (nxt == GOAL_ID).long()

            cur = nxt.unsqueeze(1)

        for i in range(b):
            out.append("".join(emitted[i]))

        if stats is not None:
            stats.n_sequences += b
            stats.n_sequences_forced += int((forced_counts > 0).sum())
            stats.forced_tokens.extend(int(x) for x in forced_counts.tolist())
            stats.quota_masked_tokens.extend(int(x) for x in quota_counts.tolist())

    return out


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    keep = cum - sorted_probs < top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs * keep)
    total = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return filtered / total
