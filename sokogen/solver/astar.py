"""Exhaustive A* Sokoban solver (spec 4.6).

Why this matters for the paper: most generative verifiers are *sound but
incomplete* -- unit tests cannot prove a program wrong.  Exhaustive A* over a
bounded Sokoban state space is a **decision procedure**: when the open set
empties under sound pruning, unsolvability is proven, not merely unobserved.
That is why ``SolveResult.status`` has three values and never collapses
"timed out" into "unsolvable".

Cost modes
----------
``cost_mode="pushes"`` (default)
    Edge cost 1 per push, state key ``(canonical_player, boxes)``.  With player
    canonicalisation this is **exactly push-optimal**.

``cost_mode="moves"``
    State key ``(exact_player, boxes)``, edge cost 1 per single step.
    Move-optimal but far slower; used only to validate the reconstructed move
    bound on a small sample.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .deadlock import INF, dead_squares, is_deadlock, push_distance_maps
from .grid import (DIR_DELTAS, STEP, Grid, bit, canonical_player, cells,
                   player_reachable)

# Sentinel standing in for an unreachable assignment cost.  Any optimal
# assignment totalling at least this cannot place every box on a distinct goal.
BIG = 1 << 30

DEFAULT_NODE_CAP = 200_000
DEFAULT_TIME_CAP_S = 10.0

# Precomputed permutations for the small assignment problems we actually face
# (4 boxes, 4 goals).  Brute force over 24 permutations is exact and beats the
# per-call overhead of scipy at this size; scipy is used for anything larger.
_PERMS_BY_N: Dict[int, Tuple[Tuple[int, ...], ...]] = {
    n: tuple(itertools.permutations(range(n))) for n in range(1, 7)
}


@dataclass(frozen=True)
class SolveResult:
    """Outcome of one search.

    ``status``
        ``"solved"``     -- an optimal solution was found.
        ``"unsolvable"`` -- **proven**: the open set emptied inside the budget
                            under sound pruning only.
        ``"timeout"``    -- the node or wall-clock cap was hit.  Proves nothing.
    ``push_length``
        Push-optimal length when ``cost_mode="pushes"``; otherwise the number of
        pushes in the move-optimal solution.  ``None`` unless solved.
    ``move_length``
        In ``cost_mode="pushes"`` this is **reconstructed** by BFS-walking the
        player between consecutive push positions, and is an **upper bound on
        the move-optimal length, not the optimum** -- canonicalisation discards
        the player's exact position, so the push order returned is optimal in
        pushes and merely feasible in moves.  In ``cost_mode="moves"`` it is
        exactly move-optimal.
    ``action_string``
        The full move sequence as digits over 0=Up, 1=Right, 2=Down, 3=Left --
        the same encoding as the published Boxoban solutions, so it can be fed
        straight to an independent replay simulator for verification.
    """

    status: str
    push_length: Optional[int]
    move_length: Optional[int]
    nodes_expanded: int
    wall_time_s: float
    action_string: Optional[str] = None
    cost_mode: str = "pushes"
    node_cap: int = DEFAULT_NODE_CAP
    deadlocks_enabled: bool = True


def _assignment_cost(box_list: Sequence[int],
                     maps: Sequence[Sequence[float]]) -> float:
    """Minimum-cost assignment of boxes to goals under the push-distance maps.

    Admissible: every push moves exactly one box one cell, so the total number
    of pushes needed is at least the cost of the cheapest perfect matching.
    Strictly tighter than sum-of-Manhattan because it respects walls.
    """
    n = len(box_list)
    if n == 0:
        return 0.0
    rows = []
    for b in box_list:
        row = [m[b] for m in maps]
        if all(v == INF for v in row):
            return INF  # this box can reach no goal at all
        rows.append([BIG if v == INF else v for v in row])

    perms = _PERMS_BY_N.get(n)
    if perms is not None:
        best = BIG * n
        for perm in perms:
            total = 0
            for i in range(n):
                total += rows[i][perm[i]]
                if total >= best:
                    break
            else:
                best = total
        return INF if best >= BIG else float(best)

    # Larger instances: exact Hungarian algorithm.
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    cost = np.array(rows, dtype=float)
    r, c = linear_sum_assignment(cost)
    total = float(cost[r, c].sum())
    return INF if total >= BIG else total


def _shortest_walk(walls: int, boxes: int, src: int, dst: int) -> Optional[List[int]]:
    """Shortest player walk from ``src`` to ``dst`` avoiding walls and boxes.

    Returns the list of direction indices, or ``None`` if unreachable.
    """
    if src == dst:
        return []
    blocked = walls | boxes
    prev: Dict[int, Tuple[int, int]] = {}
    seen = {src}
    dq = deque([src])
    while dq:
        cur = dq.popleft()
        for d in range(4):
            nxt = STEP[cur][d]
            if nxt < 0 or nxt in seen or (blocked >> nxt) & 1:
                continue
            prev[nxt] = (cur, d)
            if nxt == dst:
                path = []
                node = dst
                while node != src:
                    node, d2 = prev[node]
                    path.append(d2)
                path.reverse()
                return path
            seen.add(nxt)
            dq.append(nxt)
    return None


def _reconstruct_moves(grid: Grid, pushes: Sequence[Tuple[int, int]]
                       ) -> Tuple[str, int]:
    """Turn a push sequence into a full move sequence.

    Walks the player to each push's standing cell with BFS, then takes the
    pushing step.  The result is feasible by construction but only an *upper
    bound* on the move-optimal solution (see ``SolveResult.move_length``).
    """
    boxes = grid.boxes
    player = grid.player
    actions: List[str] = []
    for b, d in pushes:
        stand = STEP[b][d ^ 2]
        walk = _shortest_walk(grid.walls, boxes, player, stand)
        if walk is None:  # pragma: no cover - would indicate a search bug
            raise AssertionError("push sequence is not walkable")
        actions.extend(str(x) for x in walk)
        actions.append(str(d))
        dest = STEP[b][d]
        boxes = boxes ^ bit(b) ^ bit(dest)
        player = b
    return "".join(actions), len(actions)


def _solve_pushes(grid: Grid, node_cap: int, time_cap_s: float,
                  use_deadlocks: bool) -> SolveResult:
    t0 = time.perf_counter()
    walls, goals = grid.walls, grid.goals

    maps = push_distance_maps(walls, goals)
    dead = dead_squares(walls, goals, maps) if use_deadlocks else 0

    start_boxes = grid.boxes
    start_player = canonical_player(walls, start_boxes, grid.player)

    def elapsed() -> float:
        return time.perf_counter() - t0

    if start_boxes == goals:
        return SolveResult("solved", 0, 0, 0, elapsed(), "", "pushes",
                           node_cap, use_deadlocks)

    h0 = _assignment_cost(cells(start_boxes), maps)
    if h0 == INF or (use_deadlocks and is_deadlock(walls, goals, start_boxes, dead)):
        return SolveResult("unsolvable", None, None, 0, elapsed(), None,
                           "pushes", node_cap, use_deadlocks)

    start = (start_player, start_boxes)
    best_g: Dict[Tuple[int, int], int] = {start: 0}
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], int, int]] = {}
    counter = itertools.count()
    heap = [(h0, next(counter), 0, start)]

    nodes = 0
    goal_state: Optional[Tuple[int, int]] = None

    while heap:
        f, _, g, state = heapq.heappop(heap)
        if g > best_g.get(state, INF):
            continue

        if (nodes & 255) == 0 and elapsed() > time_cap_s:
            return SolveResult("timeout", None, None, nodes, elapsed(), None,
                               "pushes", node_cap, use_deadlocks)
        nodes += 1
        if nodes > node_cap:
            return SolveResult("timeout", None, None, nodes, elapsed(), None,
                               "pushes", node_cap, use_deadlocks)

        player, boxes = state
        if boxes == goals:
            goal_state = state
            break

        region = player_reachable(walls, boxes, player)
        for b in cells(boxes):
            for d in range(4):
                dest = STEP[b][d]
                if dest < 0 or (walls >> dest) & 1 or (boxes >> dest) & 1:
                    continue
                stand = STEP[b][d ^ 2]
                if stand < 0 or (walls >> stand) & 1 or (boxes >> stand) & 1:
                    continue
                if not (region >> stand) & 1:
                    continue

                nb = boxes ^ bit(b) ^ bit(dest)
                if use_deadlocks and nb != goals:
                    if nb & dead:
                        continue
                    if is_deadlock(walls, goals, nb, dead):
                        continue

                ng = g + 1
                nstate = (canonical_player(walls, nb, b), nb)
                if ng >= best_g.get(nstate, INF):
                    continue

                hh = _assignment_cost(cells(nb), maps)
                if hh == INF:
                    continue
                best_g[nstate] = ng
                parent[nstate] = (state, b, d)
                heapq.heappush(heap, (ng + hh, next(counter), ng, nstate))

    if goal_state is None:
        return SolveResult("unsolvable", None, None, nodes, elapsed(), None,
                           "pushes", node_cap, use_deadlocks)

    pushes: List[Tuple[int, int]] = []
    node = goal_state
    while node in parent:
        prev_state, b, d = parent[node]
        pushes.append((b, d))
        node = prev_state
    pushes.reverse()

    actions, move_len = _reconstruct_moves(grid, pushes)
    return SolveResult("solved", len(pushes), move_len, nodes, elapsed(),
                       actions, "pushes", node_cap, use_deadlocks)


def _solve_moves(grid: Grid, node_cap: int, time_cap_s: float,
                 use_deadlocks: bool) -> SolveResult:
    """Exact move-optimal search over ``(exact_player, boxes)`` states."""
    t0 = time.perf_counter()
    walls, goals = grid.walls, grid.goals
    maps = push_distance_maps(walls, goals)
    dead = dead_squares(walls, goals, maps) if use_deadlocks else 0

    def elapsed() -> float:
        return time.perf_counter() - t0

    if grid.boxes == goals:
        return SolveResult("solved", 0, 0, 0, elapsed(), "", "moves",
                           node_cap, use_deadlocks)

    h0 = _assignment_cost(cells(grid.boxes), maps)
    if h0 == INF or (use_deadlocks and is_deadlock(walls, goals, grid.boxes, dead)):
        return SolveResult("unsolvable", None, None, 0, elapsed(), None,
                           "moves", node_cap, use_deadlocks)

    start = (grid.player, grid.boxes)
    best_g: Dict[Tuple[int, int], int] = {start: 0}
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
    counter = itertools.count()
    heap = [(h0, next(counter), 0, start)]
    nodes = 0
    goal_state = None

    while heap:
        f, _, g, state = heapq.heappop(heap)
        if g > best_g.get(state, INF):
            continue
        if (nodes & 255) == 0 and elapsed() > time_cap_s:
            return SolveResult("timeout", None, None, nodes, elapsed(), None,
                               "moves", node_cap, use_deadlocks)
        nodes += 1
        if nodes > node_cap:
            return SolveResult("timeout", None, None, nodes, elapsed(), None,
                               "moves", node_cap, use_deadlocks)

        player, boxes = state
        if boxes == goals:
            goal_state = state
            break

        for d in range(4):
            nxt = STEP[player][d]
            if nxt < 0 or (walls >> nxt) & 1:
                continue
            nb = boxes
            if (boxes >> nxt) & 1:
                dest = STEP[nxt][d]
                if dest < 0 or (walls >> dest) & 1 or (boxes >> dest) & 1:
                    continue
                nb = boxes ^ bit(nxt) ^ bit(dest)
                if use_deadlocks and nb != goals and is_deadlock(walls, goals, nb, dead):
                    continue
            ng = g + 1
            nstate = (nxt, nb)
            if ng >= best_g.get(nstate, INF):
                continue
            hh = _assignment_cost(cells(nb), maps) if nb != boxes else None
            if hh is None:
                hh = _assignment_cost(cells(nb), maps)
            if hh == INF:
                continue
            best_g[nstate] = ng
            parent[nstate] = (state, d)
            heapq.heappush(heap, (ng + hh, next(counter), ng, nstate))

    if goal_state is None:
        return SolveResult("unsolvable", None, None, nodes, elapsed(), None,
                           "moves", node_cap, use_deadlocks)

    dirs: List[int] = []
    node = goal_state
    while node in parent:
        prev_state, d = parent[node]
        dirs.append(d)
        node = prev_state
    dirs.reverse()

    boxes = grid.boxes
    player = grid.player
    n_pushes = 0
    for d in dirs:
        nxt = STEP[player][d]
        if (boxes >> nxt) & 1:
            dest = STEP[nxt][d]
            boxes = boxes ^ bit(nxt) ^ bit(dest)
            n_pushes += 1
        player = nxt

    return SolveResult("solved", n_pushes, len(dirs), nodes, elapsed(),
                       "".join(str(d) for d in dirs), "moves", node_cap,
                       use_deadlocks)


def solve(grid, node_cap: int = DEFAULT_NODE_CAP,
          time_cap_s: float = DEFAULT_TIME_CAP_S,
          cost_mode: str = "pushes",
          use_deadlocks: bool = True) -> SolveResult:
    """Solve a level, or prove it unsolvable, or time out.

    ``grid`` may be a ``Grid`` or a 110-character grid string.

    ``status="unsolvable"`` is returned **only** when the open set empties
    before any cap is hit.  Every cap hit is ``"timeout"``.

    ``use_deadlocks=False`` disables both deadlock prunes for the soundness
    ablation.  Pruning implied by an infinite heuristic (a box that can reach no
    goal, or no perfect box-to-goal matching) stays on in both settings: it is a
    property of the admissible heuristic rather than a deadlock heuristic, and
    A* would never expand an infinite-f node anyway.
    """
    if isinstance(grid, str):
        grid = Grid.from_string(grid)
    if cost_mode not in ("pushes", "moves"):
        raise ValueError(f"unknown cost_mode {cost_mode!r}")

    # Every goal must be coverable by exactly one box.
    if grid.boxes.bit_count() != grid.goals.bit_count():
        return SolveResult("unsolvable", None, None, 0, 0.0, None, cost_mode,
                           node_cap, use_deadlocks)

    if cost_mode == "pushes":
        return _solve_pushes(grid, node_cap, time_cap_s, use_deadlocks)
    return _solve_moves(grid, node_cap, time_cap_s, use_deadlocks)
