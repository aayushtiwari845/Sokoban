"""Shared helpers for the non-learned baselines."""

from __future__ import annotations

import random
from typing import List, Sequence

from ..data.boxoban import GRID_H, GRID_W, N_BOX, N_GOAL, N_PLAYER

INTERIOR: List[int] = [i for i in range(100)
                       if 0 < i // GRID_W < GRID_H - 1
                       and 0 < i % GRID_W < GRID_W - 1]


def render(walls: Sequence[int], player: int, boxes: Sequence[int],
           goals: Sequence[int]) -> str:
    """Build a canonical 110-char grid string from cell-index collections."""
    body = [" "] * 100
    for i in range(100):
        r, c = divmod(i, GRID_W)
        if r in (0, GRID_H - 1) or c in (0, GRID_W - 1):
            body[i] = "#"
    for i in walls:
        body[i] = "#"
    for i in goals:
        body[i] = "."
    for i in boxes:
        body[i] = "$"
    body[player] = "@"
    return "".join("".join(body[r * GRID_W:(r + 1) * GRID_W]) + "\n"
                   for r in range(GRID_H))


def place_specials(free: List[int], rng: random.Random):
    """Draw one player, four boxes and four goals from distinct free cells."""
    picks = rng.sample(free, N_PLAYER + N_BOX + N_GOAL)
    return picks[0], picks[1:1 + N_BOX], picks[1 + N_BOX:]
