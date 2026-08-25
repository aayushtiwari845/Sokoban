"""Loading and parsing of Boxoban level files.

Format facts VERIFIED in Phase 1 (scripts/00_verify_data.py), not assumed:

* Files are ASCII with **CRLF** (``\\r\\n``) line endings.  The project spec
  assumed the alphabet ``{'#', '@', '$', '.', ' ', '\\n'}``; the real files also
  contain ``\\r``.  We strip it at parse time and every downstream consumer sees
  LF-only text.
* Levels are separated by a line ``"; <n>"`` (semicolon, space, puzzle number),
  followed by exactly 10 rows of 10 characters, followed by a blank line.
* The tile alphabet is exactly ``{'#', '@', '$', '.', ' '}``.  Neither ``*``
  (box on goal) nor ``+`` (player on goal) occurs anywhere in the corpus, so a
  box never *starts* on a goal.  The 5-channel one-hot tensor, the 6-symbol
  grid vocabulary and the "exactly four ``$`` and four ``.``" constraint are
  therefore all well defined.

Canonical in-memory grid representation
---------------------------------------
A level is a 110-character string: 10 rows of 10 tile characters, each row
terminated by ``\\n`` (including the last).  ``len(grid) == 110`` always.
This is the exact string the transformer is trained to emit.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator, List, Sequence

# --- format constants (verified, see module docstring) -----------------------
GRID_H = 10
GRID_W = 10
N_CELLS = GRID_H * GRID_W          # 100
GRID_STR_LEN = GRID_H * (GRID_W + 1)  # 110, rows are newline-terminated

WALL = "#"
FLOOR = " "
BOX = "$"
GOAL = "."
PLAYER = "@"

TILES = (WALL, FLOOR, BOX, GOAL, PLAYER)
TILE_SET = frozenset(TILES)

# Required counts per level (verified over the full corpus in Phase 1).
N_PLAYER = 1
N_BOX = 4
N_GOAL = 4

SPLITS = {
    "unfiltered-train": "unfiltered/train",
    "unfiltered-valid": "unfiltered/valid",
    "unfiltered-test": "unfiltered/test",
    "medium-train": "medium/train",
    "medium-valid": "medium/valid",
    "hard": "hard",
}


@dataclass(frozen=True)
class Level:
    """One Boxoban puzzle.

    Attributes
    ----------
    grid:
        Canonical 110-char newline-terminated grid string.
    source_file:
        File stem as used by the solution CSVs, e.g. ``"000"``.
    source_index:
        Puzzle number as written after the ``;`` separator in the file.
    split:
        Split key, e.g. ``"unfiltered-train"``.
    """

    grid: str
    source_file: str
    source_index: int
    split: str

    @property
    def rows(self) -> List[str]:
        return self.grid.rstrip("\n").split("\n")


def parse_level_text(text: str, source_file: str, split: str) -> List[Level]:
    """Parse the contents of one Boxoban ``.txt`` file into levels.

    Tolerates CRLF, trailing whitespace and a missing final blank line.  Raises
    ``ValueError`` on anything structurally unexpected rather than guessing --
    silent tolerance here would corrupt every downstream number.
    """
    # Normalise CRLF -> LF.  Verified necessary: the corpus is CRLF throughout.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    levels: List[Level] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if not line.startswith(";"):
            raise ValueError(
                f"{source_file}: expected a ';' separator at line {i}, got {line!r}"
            )
        try:
            index = int(line[1:].strip())
        except ValueError as exc:  # pragma: no cover - corpus is well formed
            raise ValueError(f"{source_file}: bad puzzle number in {line!r}") from exc

        rows = lines[i + 1 : i + 1 + GRID_H]
        if len(rows) != GRID_H:
            raise ValueError(
                f"{source_file}#{index}: expected {GRID_H} rows, got {len(rows)}"
            )
        for r, row in enumerate(rows):
            if len(row) != GRID_W:
                raise ValueError(
                    f"{source_file}#{index}: row {r} has width {len(row)}, "
                    f"expected {GRID_W}: {row!r}"
                )
        grid = "".join(row + "\n" for row in rows)
        levels.append(Level(grid, source_file, index, split))
        i += 1 + GRID_H

    return levels


def load_level_file(path: str, split: str) -> List[Level]:
    with open(path, "rb") as fh:
        text = fh.read().decode("ascii")
    stem = os.path.splitext(os.path.basename(path))[0]
    return parse_level_text(text, stem, split)


def split_files(root: str, split: str) -> List[str]:
    """Sorted list of level files for a split key."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; known: {sorted(SPLITS)}")
    pattern = os.path.join(root, SPLITS[split], "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no level files matched {pattern}")
    return files


def iter_levels(
    root: str, split: str, max_files: int | None = None
) -> Iterator[Level]:
    """Stream levels from a split in deterministic (file, index) order."""
    files = split_files(root, split)
    if max_files is not None:
        files = files[:max_files]
    for path in files:
        yield from load_level_file(path, split)


def load_levels(
    root: str, split: str, max_files: int | None = None
) -> List[Level]:
    return list(iter_levels(root, split, max_files))


# --- grid predicates used by verification and by the metrics ----------------

def tile_counts(grid: str) -> dict:
    body = grid.replace("\n", "")
    return {t: body.count(t) for t in TILES}


def check_invariants(grid: str) -> List[str]:
    """Return a list of invariant violations for one grid ('' list == valid).

    Checks exactly the structural-validity definition used throughout the
    paper: 10x10, wall border, exactly one player, four boxes, four goals, and
    no characters outside the verified alphabet.
    """
    problems: List[str] = []

    if len(grid) != GRID_STR_LEN:
        problems.append(f"length {len(grid)} != {GRID_STR_LEN}")
        return problems

    rows = grid.rstrip("\n").split("\n")
    if len(rows) != GRID_H:
        problems.append(f"{len(rows)} rows != {GRID_H}")
        return problems
    for r, row in enumerate(rows):
        if len(row) != GRID_W:
            problems.append(f"row {r} width {len(row)} != {GRID_W}")
    if problems:
        return problems

    bad = set("".join(rows)) - TILE_SET
    if bad:
        problems.append(f"characters outside alphabet: {sorted(bad)}")

    # Wall border: row 0, row 9, col 0, col 9.
    if any(c != WALL for c in rows[0]):
        problems.append("top row is not all wall")
    if any(c != WALL for c in rows[GRID_H - 1]):
        problems.append("bottom row is not all wall")
    if any(row[0] != WALL for row in rows):
        problems.append("left column is not all wall")
    if any(row[GRID_W - 1] != WALL for row in rows):
        problems.append("right column is not all wall")

    counts = tile_counts(grid)
    if counts[PLAYER] != N_PLAYER:
        problems.append(f"{counts[PLAYER]} players != {N_PLAYER}")
    if counts[BOX] != N_BOX:
        problems.append(f"{counts[BOX]} boxes != {N_BOX}")
    if counts[GOAL] != N_GOAL:
        problems.append(f"{counts[GOAL]} goals != {N_GOAL}")

    return problems


def is_structurally_valid(grid: str) -> bool:
    return not check_invariants(grid)


def normalize_grid(rows: Sequence[str]) -> str:
    """Build a canonical 110-char grid string from 10 row strings."""
    if len(rows) != GRID_H or any(len(r) != GRID_W for r in rows):
        raise ValueError("expected 10 rows of width 10")
    return "".join(r + "\n" for r in rows)
