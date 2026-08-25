"""Hand-built solver fixtures with known verdicts, plus an independent
reference solver used for differential testing.

The reference solver deliberately shares **no code and no representation** with
``sokogen.solver``: it works on ``(row, col)`` tuples and Python sets instead of
integer bitmasks, uses no heuristic, no player canonicalisation and no deadlock
pruning.  It is obviously correct and hopelessly slow, which is exactly what a
differential reference should be.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

Pos = Tuple[int, int]

# 0=Up, 1=Right, 2=Down, 3=Left -- same convention as the Boxoban solutions.
DIRS: Tuple[Pos, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def g(*rows: str) -> str:
    """Build a canonical 110-char grid string from 10 row literals."""
    assert len(rows) == 10, f"expected 10 rows, got {len(rows)}"
    for r in rows:
        assert len(r) == 10, f"row {r!r} has width {len(r)}, expected 10"
    return "".join(r + "\n" for r in rows)


def parse(grid: str) -> Tuple[Set[Pos], Set[Pos], Set[Pos], Pos]:
    """grid string -> (walls, goals, boxes, player) as (row, col) sets."""
    walls: Set[Pos] = set()
    goals: Set[Pos] = set()
    boxes: Set[Pos] = set()
    player: Optional[Pos] = None
    for r, row in enumerate(grid.rstrip("\n").split("\n")):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            elif ch == "$":
                boxes.add((r, c))
            elif ch == ".":
                goals.add((r, c))
            elif ch == "@":
                player = (r, c)
    assert player is not None, "fixture has no player"
    return walls, goals, boxes, player


def _reachable(player: Pos, walls: Set[Pos], boxes: FrozenSet[Pos]) -> Set[Pos]:
    seen = {player}
    dq = deque([player])
    while dq:
        r, c = dq.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if n in seen or n in walls or n in boxes:
                continue
            seen.add(n)
            dq.append(n)
    return seen


def reference_solve(grid: str, node_cap: int = 400_000) -> Dict:
    """Breadth-first search over pushes.  Push-optimal, exhaustive, unpruned.

    Returns ``{"status": "solved"|"unsolvable"|"timeout", "push_length": int|None,
    "nodes": int}``.  BFS layer order over unit-cost push edges gives the
    push-optimal length; running the frontier to exhaustion proves
    unsolvability.
    """
    walls, goals, boxes0, player0 = parse(grid)
    if len(boxes0) != len(goals):
        return {"status": "unsolvable", "push_length": None, "nodes": 0}

    start = (player0, frozenset(boxes0))
    if start[1] == frozenset(goals):
        return {"status": "solved", "push_length": 0, "nodes": 0}

    seen = {start}
    dq = deque([(start, 0)])
    nodes = 0
    while dq:
        (player, boxes), depth = dq.popleft()
        nodes += 1
        if nodes > node_cap:
            return {"status": "timeout", "push_length": None, "nodes": nodes}
        region = _reachable(player, walls, boxes)
        for box in boxes:
            for dr, dc in DIRS:
                dest = (box[0] + dr, box[1] + dc)
                stand = (box[0] - dr, box[1] - dc)
                if dest in walls or dest in boxes:
                    continue
                if stand in walls or stand in boxes or stand not in region:
                    continue
                nb = frozenset((boxes - {box}) | {dest})
                if nb == frozenset(goals):
                    return {"status": "solved", "push_length": depth + 1,
                            "nodes": nodes}
                nxt = (box, nb)
                if nxt not in seen:
                    seen.add(nxt)
                    dq.append((nxt, depth + 1))
    return {"status": "unsolvable", "push_length": None, "nodes": nodes}


# ---------------------------------------------------------------------------
# Fixtures with hand-computed verdicts
# ---------------------------------------------------------------------------

# One push right: box (4,5) -> goal (4,6).
TRIVIAL_ONE_PUSH = g(
    "##########",
    "##########",
    "##########",
    "##########",
    "####@$.###",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
)

# Two pushes right: box (4,4) -> goal (4,6).
TWO_PUSHES = g(
    "##########",
    "##########",
    "##########",
    "##########",
    "###@$ .###",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
)

# Three pushes with a turn.  The player starts in a stub that only allows a
# push to the right; it must then walk around through (3,4)-(3,5) to push the
# box down twice.  Box (4,4) -> goal (6,5) is Manhattan distance 3, and a
# 3-push solution exists, so 3 is optimal.
THREE_PUSHES_TURN = g(
    "##########",
    "##########",
    "####  ####",
    "####  ####",
    "###@$ ####",
    "####  ####",
    "#### .####",
    "##########",
    "##########",
    "##########",
)

# Box in a non-goal corner -> dead square -> unsolvable.
CORNER_DEADLOCK = g(
    "##########",
    "#$       #",
    "#        #",
    "#       .#",
    "#   @    #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
)

# Two boxes side by side against the top wall: each blocks the other's only
# push axis, so both are permanently immobile.  Neither is on a goal.
TWO_BOXES_FROZEN = g(
    "##########",
    "#  $$    #",
    "#        #",
    "#      ..#",
    "#   @    #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
)

# One-wide corridor, player at the left, two boxes to the right.  The player
# can only ever push the nearer box, which jams against the far one, so this is
# UNSOLVABLE despite looking like a simple "push both right" level.  It exists
# to catch a solver that mistakes "a push is available" for progress.
CORRIDOR_JAM = g(
    "##########",
    "##########",
    "##########",
    "#@$ $  ..#",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
)

# A box against a wall can still slide along it; this must NOT be called a
# deadlock (guards against an over-eager freeze detector).
WALL_SLIDE_OK = g(
    "##########",
    "#@$     .#",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
)

# Unequal box/goal counts -> cannot cover every goal.
UNEQUAL_COUNTS = g(
    "##########",
    "#@$    ..#",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
)

# Real Boxoban levels from unfiltered/test (file 000, levels 0-2).  These are
# the "non-obvious push order" cases: their push-optimal lengths below were
# produced by the independent reference solver above and cross-checked against
# the published A* solutions (which are move-optimal and therefore use >= as
# many pushes: 15, 16 and 11 respectively).
REAL_TEST_0 = g(
    "##########",
    "###    . #",
    "## .   $.#",
    "##    .$ #",
    "#####    #",
    "####   ###",
    "##### $###",
    "#####$ ###",
    "#####@####",
    "##########",
)
REAL_TEST_1 = g(
    "##########",
    "###.#   .#",
    "# $ .. $ #",
    "#@$    $##",
    "######  ##",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
)
REAL_TEST_2 = g(
    "##########",
    "#####. ###",
    "#####.   #",
    "#####  $ #",
    "##### $###",
    "##### .# #",
    "###   $  #",
    "###    $@#",
    "##     .##",
    "##########",
)

SOLVABLE_FIXTURES: List[Tuple[str, str, int]] = [
    # (name, grid, push-optimal length)
    # -- the first three are hand-computed, per spec 4.7 --
    ("trivial_one_push", TRIVIAL_ONE_PUSH, 1),
    ("two_pushes", TWO_PUSHES, 2),
    ("three_pushes_turn", THREE_PUSHES_TURN, 3),
    ("wall_slide_ok", WALL_SLIDE_OK, 6),
    # -- real levels, lengths from the independent reference solver --
    ("real_test_0", REAL_TEST_0, 13),
    ("real_test_1", REAL_TEST_1, 14),
    ("real_test_2", REAL_TEST_2, 11),
]

UNSOLVABLE_FIXTURES: List[Tuple[str, str]] = [
    ("corner_deadlock", CORNER_DEADLOCK),
    ("two_boxes_frozen", TWO_BOXES_FROZEN),
    ("unequal_counts", UNEQUAL_COUNTS),
    ("corridor_jam", CORRIDOR_JAM),
]

ALL_FIXTURES: List[Tuple[str, str]] = (
    [(n, gr) for n, gr, _ in SOLVABLE_FIXTURES] + UNSOLVABLE_FIXTURES
)
