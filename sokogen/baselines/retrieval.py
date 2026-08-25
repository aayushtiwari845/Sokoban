"""Retrieval baseline: sample a training level verbatim (spec 9.1).

100% solvable and 0% novel by construction.  It exists to make the point from
the other side that **solvability alone is a degenerate objective**, maximised
by copying the training data.  Reported alongside the open-room baseline, it
forces the results table to be read as a joint constraint.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence


def generate(n: int, train_rows: Sequence[Dict], seed: int = 1337) -> List[str]:
    rng = random.Random(seed)
    return [train_rows[rng.randrange(len(train_rows))]["level"] for _ in range(n)]
