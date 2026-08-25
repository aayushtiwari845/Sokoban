"""Sound deadlock detection: dead squares and freeze deadlocks (spec 4.5).

Only two prunes, both **sound**: they fire only on states from which no
solution exists.  Nothing heuristic is admitted here.  An unsound prune
produces false "proven unsolvable" verdicts, which is the worst failure mode in
this project -- it corrupts the headline number invisibly and in a plausible
direction.  ``scripts/02_validate_solver.py`` runs a pruning-disabled ablation
that must reproduce the pruned verdict set exactly.
"""

from __future__ import annotations

from typing import List, Sequence, Set

from .grid import DIR_DELTAS, FULL, N, STEP, bit, cells

INF = float("inf")


def push_distance_maps(walls: int, goals: int) -> List[List[float]]:
    """For each goal, the minimum pushes needed to bring a box to it.

    ``maps[j][c]`` = pushes to move a box from cell ``c`` to the ``j``-th goal
    (goals in ascending cell order), ignoring other boxes but respecting walls.

    Computed by BFS backwards from the goal: a box at ``p`` could have arrived
    from ``q = p - d`` if both ``q`` and ``q - d`` are non-wall -- ``q`` is
    where the box was and ``q - d`` is where the player had to stand to push
    it.  Ignoring other boxes and player reachability can only *shorten* a
    path, so these distances are a lower bound on the true push count, which is
    what makes the derived heuristic admissible.
    """
    maps: List[List[float]] = []
    for g in cells(goals):
        dist: List[float] = [INF] * N
        dist[g] = 0
        frontier = [g]
        while frontier:
            nxt = []
            for p in frontier:
                dp = dist[p] + 1
                for d in range(4):
                    q = STEP[p][d ^ 2]        # q = p - delta_d
                    if q < 0 or (walls >> q) & 1:
                        continue
                    stand = STEP[q][d ^ 2]    # q - delta_d
                    if stand < 0 or (walls >> stand) & 1:
                        continue
                    if dp < dist[q]:
                        dist[q] = dp
                        nxt.append(q)
            frontier = nxt
        maps.append(dist)
    return maps


def dead_squares(walls: int, goals: int, maps: Sequence[Sequence[float]] | None = None
                 ) -> int:
    """Bitmask of non-wall cells from which a box can never reach any goal.

    Any box on a dead square is an immediate, provable deadlock.  This subsumes
    corner deadlocks (a non-goal corner is dead for every goal), so no separate
    corner check exists.
    """
    if maps is None:
        maps = push_distance_maps(walls, goals)
    dead = 0
    for c in range(N):
        if (walls >> c) & 1:
            continue
        if all(m[c] == INF for m in maps):
            dead |= bit(c)
    return dead


def _blocked_along(cell: int, axis: int, walls: int, boxes: int, dead: int,
                   goals: int, visiting: Set[int]) -> bool:
    """Is ``cell`` blocked along ``axis`` (0 = vertical, 1 = horizontal)?

    A blocker is a wall, a box currently under evaluation (the standard sound
    convention -- see ``is_frozen``), or a neighbouring box that is itself
    frozen.  Additionally, if *both* neighbours along the axis are dead squares
    the box can only ever move onto a square from which no goal is reachable,
    which is equivalent to being blocked.
    """
    a = STEP[cell][0 if axis == 0 else 3]  # Up  / Left
    b = STEP[cell][2 if axis == 0 else 1]  # Down / Right

    a_wall = a < 0 or bool((walls >> a) & 1) or a in visiting
    b_wall = b < 0 or bool((walls >> b) & 1) or b in visiting
    if a_wall or b_wall:
        return True

    a_dead = bool((dead >> a) & 1)
    b_dead = bool((dead >> b) & 1)
    if a_dead and b_dead:
        return True

    for n in (a, b):
        if (boxes >> n) & 1:
            if is_frozen(n, walls, boxes, dead, goals, visiting):
                return True
    return False


def is_frozen(cell: int, walls: int, boxes: int, dead: int, goals: int,
              visiting: Set[int] | None = None) -> bool:
    """Is the box at ``cell`` permanently immobile?

    A box is frozen when it is blocked along both the horizontal and the
    vertical axis.  The recursion terminates via ``visiting``: a box currently
    under evaluation is treated as a wall for the boxes it is being compared
    against.  That is the standard convention and it is sound -- it can only
    make the detector *more* conservative about calling something movable in
    the recursive branch, and a mutually-blocking cluster really is immobile.
    """
    if visiting is None:
        visiting = set()
    visiting.add(cell)
    try:
        vertical = _blocked_along(cell, 0, walls, boxes, dead, goals, visiting)
        if not vertical:
            return False
        return _blocked_along(cell, 1, walls, boxes, dead, goals, visiting)
    finally:
        visiting.discard(cell)


def is_freeze_deadlock(walls: int, goals: int, boxes: int, dead: int) -> bool:
    """True if some box is frozen off a goal, which proves the state is lost.

    A frozen box can never move again.  If it is not on a goal it can never be
    on one, so the level can never be solved.  Frozen boxes that all sit on
    goals are fine -- that is a solved (or partially solved) position, not a
    deadlock.
    """
    for b in cells(boxes):
        if (goals >> b) & 1:
            continue  # already on a goal: being stuck there is harmless
        if is_frozen(b, walls, boxes, dead, goals, set()):
            return True
    return False


def is_deadlock(walls: int, goals: int, boxes: int, dead: int) -> bool:
    """Combined sound deadlock test used by the search."""
    if boxes & dead:
        return True
    return is_freeze_deadlock(walls, goals, boxes, dead)
