"""Project an invalid grid onto the nearest valid one (spec 9.2).

Repair uses the model's own predicted probabilities so the projection is
minimally violent: it keeps the cells the model was most confident about.

Why this matters for the paper: post-repair solvability decomposes the
comparison into two separable questions --

* *Can the model count?*             -> structural validity
* *Does it understand spatial structure?* -> post-repair solvability

Without it a reader cannot tell whether the GAN is bad at Sokoban or merely bad
at arithmetic.

Rules
-----
* Border cells are forced to wall.
* Boxes: if more than four, drop the lowest-probability ones; if fewer, promote
  the highest-probability interior floor cells.  Same for goals and the player.
* A box or goal is never placed on a border cell.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..data.boxoban import GRID_H, GRID_W, N_BOX, N_GOAL, N_PLAYER

# Channel order shared with the one-shot models.
WALL, FLOOR, BOX, GOAL, PLAYER = range(5)
CHARS = ("#", " ", "$", ".", "@")

INTERIOR = [i for i in range(100) if 0 < i // GRID_W < GRID_H - 1
            and 0 < i % GRID_W < GRID_W - 1]
BORDER = [i for i in range(100) if i not in set(INTERIOR)]

QUOTAS = {PLAYER: N_PLAYER, BOX: N_BOX, GOAL: N_GOAL}


def repair_tiles(tiles: Sequence[int],
                 probs: Optional[np.ndarray] = None) -> List[int]:
    """Repair a length-100 array of channel indices.

    ``probs`` is an optional ``[5, 100]`` array of per-cell channel
    probabilities.  Without it, confidence ties are broken by cell order, which
    keeps the function deterministic and total.
    """
    out = list(int(t) for t in tiles)
    if probs is None:
        probs = np.zeros((5, 100), dtype=float)
        for i, t in enumerate(out):
            probs[t, i] = 1.0

    for i in BORDER:
        out[i] = WALL

    # Resolve the three quota'd channels in a fixed order so repair is
    # deterministic; each pass only ever touches interior cells.
    for channel in (PLAYER, BOX, GOAL):
        quota = QUOTAS[channel]
        held = [i for i in INTERIOR if out[i] == channel]

        if len(held) > quota:
            # Keep the cells the model was most confident about.
            held.sort(key=lambda i: (-probs[channel, i], i))
            for i in held[quota:]:
                out[i] = FLOOR
        elif len(held) < quota:
            need = quota - len(held)
            # Promote the most confident cells that are not already spoken for.
            free = [i for i in INTERIOR
                    if out[i] in (FLOOR, WALL) and out[i] != channel]
            free.sort(key=lambda i: (-probs[channel, i], i))
            for i in free[:need]:
                out[i] = channel
            if len(free) < need:  # pragma: no cover - 64 interior cells >> 9
                raise ValueError("not enough interior cells to satisfy quotas")

    return out


def tiles_to_grid(tiles: Sequence[int]) -> str:
    body = "".join(CHARS[int(t)] for t in tiles)
    return "".join(body[r * GRID_W:(r + 1) * GRID_W] + "\n" for r in range(GRID_H))


def grid_to_tiles(grid: str) -> List[int]:
    """Parse a grid string to channel indices.

    Any character outside the verified alphabet becomes floor, so repair is
    total over whatever an unconstrained decoder emits.
    """
    lookup = {c: i for i, c in enumerate(CHARS)}
    body = grid.replace("\n", "")
    return [lookup.get(c, FLOOR) for c in body]


def repair_grid(grid: str, probs: Optional[np.ndarray] = None) -> str:
    """Repair a grid string.  Returns a structurally valid grid string.

    If the input is not exactly 100 tiles it is padded or truncated with floor
    first, so a malformed unconstrained sample still repairs to something valid.
    """
    tiles = grid_to_tiles(grid)
    if len(tiles) < 100:
        tiles = tiles + [FLOOR] * (100 - len(tiles))
    elif len(tiles) > 100:
        tiles = tiles[:100]
    return tiles_to_grid(repair_tiles(tiles, probs))
