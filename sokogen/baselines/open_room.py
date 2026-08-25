"""Open-room baseline -- the most important baseline in the project (spec 9.1).

Border walls only, **zero interior walls**, with a player, four boxes and four
goals dropped uniformly at random on the open floor.

This is the baseline that can invalidate the headline number.  An empty room is
likely solvable a large fraction of the time -- the usual failure is a box
landing in a non-goal corner or getting stuck against a wall.  If it scores near
the transformer, then solvability alone says little, and the results section has
to be read as a joint constraint over solvability x difficulty control x
novelty.  That has to be known *before* the abstract is written.
"""

from __future__ import annotations

import random
from typing import List

from .common import INTERIOR, place_specials, render


def generate_one(rng: random.Random) -> str:
    player, boxes, goals = place_specials(list(INTERIOR), rng)
    return render([], player, boxes, goals)


def generate(n: int, seed: int = 1337) -> List[str]:
    rng = random.Random(seed)
    return [generate_one(rng) for _ in range(n)]
