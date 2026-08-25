"""Bitmask grid representation, dihedral symmetry, and player canonicalisation.

Representation (spec 4.1)
-------------------------
Cell index ``i = row * 10 + col``, ``i in [0, 100)``.  ``walls``, ``goals`` and
``boxes`` are Python ``int`` bitmasks; ``player`` is a single cell index.

Python ints are arbitrary precision, hash cheaply and support the shift/mask
flood fill below, which is roughly an order of magnitude faster in CPython than
a ``frozenset`` of positions.  On a ~5,000-search evaluation sweep that is the
difference between a 2-hour and a 20-hour run, so the representation is not
negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

H = 10
W = 10
N = H * W

WALL = "#"
FLOOR = " "
BOX = "$"
GOAL = "."
PLAYER = "@"
BOX_ON_GOAL = "*"     # never occurs in the corpus (Phase 1); mid-search only
PLAYER_ON_GOAL = "+"  # never occurs in the corpus (Phase 1); mid-search only

FULL = (1 << N) - 1

# Column guards for the shift-based flood fill: shifting by +/-1 must not wrap
# from column 9 to column 0 of the next row.
NOT_COL0 = 0
NOT_COL9 = 0
for _i in range(N):
    if _i % W != 0:
        NOT_COL0 |= 1 << _i
    if _i % W != W - 1:
        NOT_COL9 |= 1 << _i

# Direction order matches the Boxoban action encoding: 0=Up, 1=Right, 2=Down,
# 3=Left.  Keeping it identical means a reconstructed action string can be fed
# straight to the independent replay simulator in sokogen.data.solutions.
DIR_DELTAS: Tuple[int, ...] = (-W, +1, +W, -1)
DIR_NAMES: Tuple[str, ...] = ("Up", "Right", "Down", "Left")
N_DIRS = 4


def idx(r: int, c: int) -> int:
    return r * W + c


def rc(i: int) -> Tuple[int, int]:
    return divmod(i, W)


def bit(i: int) -> int:
    return 1 << i


def cells(mask: int) -> List[int]:
    """Ascending list of set-bit indices.  Deterministic iteration order."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def popcount(mask: int) -> int:
    return mask.bit_count()


# ---------------------------------------------------------------------------
# Neighbour tables with explicit bounds (never rely on the wall border)
# ---------------------------------------------------------------------------
NEIGHBOURS: List[Tuple[int, ...]] = []
STEP: List[List[int]] = []  # STEP[i][d] -> neighbour index or -1
for _i in range(N):
    _r, _c = divmod(_i, W)
    _row = []
    for _d, _delta in enumerate(DIR_DELTAS):
        if _d == 0:
            _ok = _r > 0
        elif _d == 1:
            _ok = _c < W - 1
        elif _d == 2:
            _ok = _r < H - 1
        else:
            _ok = _c > 0
        _row.append(_i + _delta if _ok else -1)
    STEP.append(_row)
    NEIGHBOURS.append(tuple(n for n in _row if n >= 0))


# ---------------------------------------------------------------------------
# Dihedral group of the square: 4 rotations x optional flip
# ---------------------------------------------------------------------------
DIHEDRAL_N = 8


def _build_perms() -> List[Tuple[int, ...]]:
    """PERMS[k][i] = index that cell i maps to under transform k."""
    perms = []
    for flip in (False, True):
        for rot in range(4):
            perm = [0] * N
            for i in range(N):
                r, c = divmod(i, W)
                if flip:
                    c = W - 1 - c
                for _ in range(rot):
                    r, c = c, H - 1 - r  # rotate 90 degrees clockwise
                perm[i] = r * W + c
            perms.append(tuple(perm))
    return perms


PERMS: List[Tuple[int, ...]] = _build_perms()


def transform_mask(mask: int, perm: Tuple[int, ...]) -> int:
    out = 0
    while mask:
        low = mask & -mask
        out |= 1 << perm[low.bit_length() - 1]
        mask ^= low
    return out


# ---------------------------------------------------------------------------
# Player reachability (spec 4.2)
# ---------------------------------------------------------------------------
def player_reachable(walls: int, boxes: int, player: int) -> int:
    """Bitmask of cells the player can reach without pushing anything.

    Shift-and-mask flood fill: each iteration expands the frontier in all four
    directions at once with a handful of big-int operations, converging in
    O(diameter) iterations rather than O(cells) Python-level BFS steps.
    """
    free = ~(walls | boxes) & FULL
    region = bit(player) & free
    if not region:
        return 0
    while True:
        grown = (region
                 | ((region << 1) & NOT_COL0)
                 | ((region >> 1) & NOT_COL9)
                 | (region << W)
                 | (region >> W)) & free
        if grown == region:
            return region
        region = grown


def canonical_player(walls: int, boxes: int, player: int) -> int:
    """Canonical player value for state hashing: ``min`` of the reachable region.

    The player's exact cell is irrelevant to what is achievable -- only which
    region it occupies.  Without this, states differing only by player position
    within one region hash differently and the search space inflates by roughly
    the region size (~30-60x on these levels).
    """
    region = player_reachable(walls, boxes, player)
    if not region:
        return player
    return (region & -region).bit_length() - 1


@dataclass(frozen=True)
class Grid:
    """A Sokoban level as four integers.

    ``walls``, ``goals``, ``boxes`` are bitmasks; ``player`` is a cell index.
    """

    walls: int
    goals: int
    boxes: int
    player: int

    # -- construction ------------------------------------------------------
    @classmethod
    def from_string(cls, grid: str) -> "Grid":
        rows = grid.rstrip("\n").split("\n")
        if len(rows) != H or any(len(r) != W for r in rows):
            raise ValueError(
                f"expected {H} rows of width {W}, got "
                f"{len(rows)} rows of widths {sorted({len(r) for r in rows})}")
        walls = goals = boxes = 0
        player = -1
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                i = r * W + c
                if ch == WALL:
                    walls |= 1 << i
                elif ch == BOX:
                    boxes |= 1 << i
                elif ch == GOAL:
                    goals |= 1 << i
                elif ch == PLAYER:
                    if player >= 0:
                        raise ValueError("more than one player")
                    player = i
                elif ch == BOX_ON_GOAL:
                    boxes |= 1 << i
                    goals |= 1 << i
                elif ch == PLAYER_ON_GOAL:
                    if player >= 0:
                        raise ValueError("more than one player")
                    player = i
                    goals |= 1 << i
                elif ch != FLOOR:
                    raise ValueError(f"unexpected character {ch!r} at ({r},{c})")
        if player < 0:
            raise ValueError("no player in grid")
        return cls(walls, goals, boxes, player)

    def to_string(self) -> str:
        out = []
        for r in range(H):
            row = []
            for c in range(W):
                i = r * W + c
                is_goal = (self.goals >> i) & 1
                if (self.walls >> i) & 1:
                    row.append(WALL)
                elif (self.boxes >> i) & 1:
                    row.append(BOX_ON_GOAL if is_goal else BOX)
                elif i == self.player:
                    row.append(PLAYER_ON_GOAL if is_goal else PLAYER)
                elif is_goal:
                    row.append(GOAL)
                else:
                    row.append(FLOOR)
            out.append("".join(row))
        return "".join(r + "\n" for r in out)

    # -- symmetry ----------------------------------------------------------
    def transform(self, k: int) -> "Grid":
        if not 0 <= k < DIHEDRAL_N:
            raise ValueError(f"transform index {k} outside [0,{DIHEDRAL_N})")
        perm = PERMS[k]
        return Grid(
            walls=transform_mask(self.walls, perm),
            goals=transform_mask(self.goals, perm),
            boxes=transform_mask(self.boxes, perm),
            player=perm[self.player],
        )

    def canonical_string(self) -> str:
        """Lexicographically smallest of the 8 symmetric renderings.

        Two levels related by a rotation or reflection share this string, which
        is what makes the novelty metric symmetry-aware.
        """
        return min(self.transform(k).to_string() for k in range(DIHEDRAL_N))

    # -- state helpers -----------------------------------------------------
    def canonical_player(self) -> int:
        return canonical_player(self.walls, self.boxes, self.player)

    @property
    def n_boxes(self) -> int:
        return self.boxes.bit_count()

    @property
    def n_goals(self) -> int:
        return self.goals.bit_count()

    @property
    def is_solved(self) -> bool:
        return self.boxes == self.goals
