"""Precomputed A* solutions for Boxoban levels.

Source: Garriga-Alonso, Taufeeque & Gleave, "Planning behavior in a recurrent
neural network that plays Sokoban", ICML 2024 Mechanistic Interpretability
Workshop.  Distributed as the HuggingFace dataset
``AlignmentResearch/boxoban-astar-solutions``.  Cite the paper, not the URL.

Format facts VERIFIED in Phase 1 (scripts/00_verify_data.py), not assumed:

* Columns are ``File, Level, Actions, Steps, SearchSteps``.
* The CSVs must be read with ``dtype=str`` and indexed by ``("File", "Level")``.
  **Both index levels are zero-padded 3-digit strings** (``("000", "023")``),
  not integers.  Indexing with ints raises ``KeyError``.
* ``Actions`` is a digit string over ``{0,1,2,3}`` = Up, Right, Down, Left
  (verified by replay: 926/1000 test and 99.5% of train levels reach the goal
  state exactly under this mapping; no other permutation of the four
  directions, with or without transposition, replays even a single one of the
  levels that fail).
* ``Steps == len(Actions)`` and therefore counts **player moves, not pushes**.
  Their A* minimised moves, so ``Steps`` is move-optimal where it is valid.
  Replaying gives the push count of that move-optimal solution, which is an
  *upper* bound on the push-optimal length, not equal to it.

Data-quality finding (Phase 1, not documented upstream)
-------------------------------------------------------
The dataset README documents rows whose ``Actions`` is ``SEARCH_STATE_FAILED``
or ``NOT_FOUND`` (11 of 1,000 in unfiltered_test, 495 of 900,000 in
unfiltered_train).  Phase 1 additionally found rows that *look* well formed --
a plain digit string -- but **do not replay to a solved state**:

* ``unfiltered_test``:  63 / 1000  (6.3%)
* ``unfiltered_train``: ~0.46% (measured on a 20,000-level sample)

These fail by walking into a wall or making an impossible push part-way
through, and no direction convention fixes them.  They are silently wrong, so
**every solution used in this project is validated by replay** and rows that do
not replay are dropped.  ``load_solutions`` does this for you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .boxoban import BOX, GOAL, GRID_W, PLAYER, WALL

# 0=Up, 1=Right, 2=Down, 3=Left in (row, col) with row increasing downwards.
ACTION_VECTORS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("Up", "Right", "Down", "Left")

# Rows whose Actions field is a real action string (the upstream check).
_ACTION_RE = re.compile(r"^ ?[0-3]+$")

SOLUTION_FILES = {
    "unfiltered-train": "unfiltered_train.csv.gz",
    "unfiltered-valid": "unfiltered_valid.csv.gz",
    "unfiltered-test": "unfiltered_test.csv.gz",
    "medium-valid": "medium_valid.csv.gz",
}


@dataclass(frozen=True)
class Replay:
    """Outcome of replaying an action string against a level."""

    solved: bool
    moves: int
    pushes: int
    failure: Optional[str] = None  # None iff the replay was legal throughout


def replay_actions(grid: str, actions: str) -> Replay:
    """Replay an action string against a grid under strict Sokoban rules.

    Strict means a move into a wall, or a push blocked by a wall or a second
    box, is an *error* rather than a no-op.  Phase 1 verified that no-op
    semantics does not rescue any of the failing rows, so strict replay is the
    correct validity test.

    Returns a ``Replay``; ``solved`` is True only if every action was legal and
    the final box set equals the goal set.
    """
    rows = grid.rstrip("\n").split("\n")
    walls = set()
    goals = set()
    boxes = set()
    player = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == WALL:
                walls.add((r, c))
            elif ch == BOX:
                boxes.add((r, c))
            elif ch == GOAL:
                goals.add((r, c))
            elif ch == PLAYER:
                player = (r, c)
    if player is None:
        return Replay(False, 0, 0, "no player")

    pushes = 0
    for i, a in enumerate(actions):
        dr, dc = ACTION_VECTORS[int(a)]
        nr, nc = player[0] + dr, player[1] + dc
        if (nr, nc) in walls:
            return Replay(False, i, pushes, f"move into wall at step {i}")
        if (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            if (br, bc) in walls or (br, bc) in boxes:
                return Replay(False, i, pushes, f"blocked push at step {i}")
            boxes.discard((nr, nc))
            boxes.add((br, bc))
            pushes += 1
        player = (nr, nc)

    solved = boxes == goals
    return Replay(solved, len(actions), pushes, None if solved else "not solved at end")


def is_action_string(value: str) -> bool:
    return isinstance(value, str) and _ACTION_RE.fullmatch(value) is not None


def load_solutions_frame(solutions_root: str, split: str) -> pd.DataFrame:
    """Load one solutions CSV with the verified dtype/index conventions."""
    if split not in SOLUTION_FILES:
        raise KeyError(f"no solution file for split {split!r}")
    path = f"{solutions_root}/{SOLUTION_FILES[split]}"
    return pd.read_csv(path, dtype=str, index_col=("File", "Level"))


def solution_key(source_file: str, source_index: int) -> Tuple[str, str]:
    """Build the zero-padded index key used by the solution CSVs."""
    return (f"{int(source_file):03d}", f"{source_index:03d}")


def lookup_actions(df: pd.DataFrame, source_file: str, source_index: int) -> Optional[str]:
    """Return the raw action string for a level, or None if absent/failed."""
    try:
        value = df.at[solution_key(source_file, source_index), "Actions"]
    except KeyError:
        return None
    if not is_action_string(value):
        return None
    return value.strip()


@dataclass(frozen=True)
class VerifiedSolution:
    """A replay-validated solution length pair."""

    moves: int   # move-optimal (their A* objective), == Steps
    pushes: int  # pushes taken by that move-optimal solution (upper bound on push-optimal)


def verified_solution(df: pd.DataFrame, grid: str, source_file: str, source_index: int
                      ) -> Optional[VerifiedSolution]:
    """Look up and replay-validate one level's solution.

    Returns ``None`` when the row is missing, marked failed upstream, or does
    not replay to a solved state.  Callers should drop such levels rather than
    trusting the stored ``Steps``.
    """
    actions = lookup_actions(df, source_file, source_index)
    if actions is None:
        return None
    rep = replay_actions(grid, actions)
    if not rep.solved:
        return None
    return VerifiedSolution(moves=rep.moves, pushes=rep.pushes)
