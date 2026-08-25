"""Random-placement baseline: the floor of the results table (spec 9.1).

Uniform random interior walls at the **training median** density, then a
player, four boxes and four goals dropped uniformly on the remaining free
cells.  No connectivity check, no solvability check.
"""

from __future__ import annotations

import random
from typing import List

from .common import INTERIOR, place_specials, render

# Training-median interior wall density, measured in Phase 3.
TRAIN_MEDIAN_DENSITY = 0.5


def generate_one(rng: random.Random, density: float = TRAIN_MEDIAN_DENSITY) -> str:
    n_walls = int(round(density * len(INTERIOR)))
    walls = rng.sample(INTERIOR, n_walls)
    free = [i for i in INTERIOR if i not in set(walls)]
    if len(free) < 9:
        # Too dense to hold nine specials; free up the cells we need.
        walls = walls[: max(0, len(INTERIOR) - 9)]
        free = [i for i in INTERIOR if i not in set(walls)]
    player, boxes, goals = place_specials(free, rng)
    return render(walls, player, boxes, goals)


def generate(n: int, seed: int = 1337,
             density: float = TRAIN_MEDIAN_DENSITY) -> List[str]:
    rng = random.Random(seed)
    return [generate_one(rng, density) for _ in range(n)]
