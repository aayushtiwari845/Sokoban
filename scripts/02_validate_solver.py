"""Phase 2 -- solver validation.  GATE 2 (hard, blocking).

Four validations (spec 4.8):

 1. **Real-level validation** on all 1,000 ``unfiltered/test`` levels.  They are
    generated solvable, so the solve rate should be ~100%.  Any ``unsolvable``
    verdict on a real level is a solver bug.  Reported lengths are cross-checked
    against the published A* solutions, accounting for the push-vs-move
    distinction established in Phase 1.
 2. **Soundness ablation**: on a 200-level sample, rerun with deadlock pruning
    disabled and a 10x node cap; the verdict set must be identical.  This is the
    check that catches an unsound freeze detector.
 3. **Medium split** (200 levels) to characterise solver strength beyond the
    easy split.
 4. **Solve rate as a function of node cap** (1k / 10k / 100k / 1M), which shows
    whether the headline number is a converged lower bound.

Plus the move-bound validation for Known Unknown 6: exact ``cost_mode="moves"``
on a <=200-level sample, compared against the reconstructed upper bound.

GATE 2 passes at >=99.5% solve rate on the real test levels, zero false
``unsolvable``, and an identical ablation verdict set.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sokogen.data.boxoban import load_level_file, split_files
from sokogen.data.solutions import (load_solutions_frame, lookup_actions,
                                    replay_actions)
from sokogen.provenance import stamp, write_artifact
from sokogen.solver.astar import solve
from sokogen.solver.grid import Grid, cells, player_reachable


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


# --------------------------------------------------------------------------
# Worker (top level so it pickles under Windows spawn)
# --------------------------------------------------------------------------
def _work(job: Tuple) -> Dict:
    grid, node_cap, time_cap, use_deadlocks, cost_mode, tag = job
    res = solve(grid, node_cap=node_cap, time_cap_s=time_cap,
                cost_mode=cost_mode, use_deadlocks=use_deadlocks)
    return {
        "tag": tag,
        "status": res.status,
        "push_length": res.push_length,
        "move_length": res.move_length,
        "nodes": res.nodes_expanded,
        "time_s": res.wall_time_s,
        "actions": res.action_string,
    }


def run_pool(jobs: List[Tuple], workers: int, desc: str) -> List[Dict]:
    t0 = time.perf_counter()
    if workers <= 1:
        out = [_work(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            out = pool.map(_work, jobs, chunksize=8)
    dt = time.perf_counter() - t0
    print(f"  {desc}: {len(jobs)} searches in {dt:.1f}s "
          f"({1000*dt/max(1,len(jobs)):.1f} ms/search, {workers} workers)")
    return out


def summarise(results: List[Dict]) -> Dict:
    counts = {"solved": 0, "unsolvable": 0, "timeout": 0}
    for r in results:
        counts[r["status"]] += 1
    nodes = sorted(r["nodes"] for r in results)
    times = sorted(r["time_s"] for r in results)
    n = len(results)

    def q(a, p):
        return float(a[min(len(a) - 1, int(p * (len(a) - 1)))]) if a else None

    return {
        "n": n,
        "counts": counts,
        "solve_rate": counts["solved"] / n if n else None,
        "nodes_median": q(nodes, 0.5),
        "nodes_p99": q(nodes, 0.99),
        "nodes_max": max(nodes) if nodes else None,
        "time_median_s": q(times, 0.5),
        "time_p99_s": q(times, 0.99),
        "time_max_s": max(times) if times else None,
        "total_time_s": float(sum(times)),
    }


def print_summary(name: str, s: Dict) -> None:
    c = s["counts"]
    print(f"  {name}: n={s['n']}")
    print(f"     solved={c['solved']} ({pct(c['solved'], s['n'])})  "
          f"unsolvable={c['unsolvable']}  timeout={c['timeout']}")
    print(f"     nodes  median={s['nodes_median']:.0f}  p99={s['nodes_p99']:.0f}  "
          f"max={s['nodes_max']}")
    print(f"     time   median={1000*s['time_median_s']:.1f}ms  "
          f"p99={1000*s['time_p99_s']:.1f}ms  max={1000*s['time_max_s']:.1f}ms")


# --------------------------------------------------------------------------
# 1. Real-level validation
# --------------------------------------------------------------------------
def validate_real(levels, node_cap, time_cap, workers, solutions_root) -> Dict:
    hr("VALIDATION 1  Real levels (unfiltered/test)")
    jobs = [(l.grid, node_cap, time_cap, True, "pushes", l.source_index)
            for l in levels]
    results = run_pool(jobs, workers, "solve")
    s = summarise(results)
    print_summary("unfiltered/test", s)

    false_unsolvable = [r["tag"] for r in results if r["status"] == "unsolvable"]
    if false_unsolvable:
        print(f"\n  *** {len(false_unsolvable)} 'unsolvable' verdicts on REAL levels ***")
        print(f"  *** These are generated solvable -- investigate before proceeding ***")
        print(f"  first 20: {false_unsolvable[:20]}")
    else:
        print("\n  => zero 'unsolvable' verdicts on real levels (as required)")

    # Every "solved" verdict must come with actions that really solve the level.
    bad_replay = []
    by_tag = {r["tag"]: r for r in results}
    for l in levels:
        r = by_tag[l.source_index]
        if r["status"] != "solved":
            continue
        rep = replay_actions(l.grid, r["actions"])
        if not rep.solved or rep.pushes != r["push_length"] or rep.moves != r["move_length"]:
            bad_replay.append(l.source_index)
    print(f"  => self-verification: {len(results) - len(bad_replay)}/"
          f"{sum(1 for r in results if r['status']=='solved')} returned solutions "
          f"replay to a solved state" + (f"  BAD: {bad_replay[:10]}" if bad_replay else ""))

    # Cross-check against the published solutions.
    df = load_solutions_frame(solutions_root, "unfiltered-test")
    ours_push, ours_move, theirs_move, theirs_push = [], [], [], []
    for l in levels:
        r = by_tag[l.source_index]
        if r["status"] != "solved":
            continue
        actions = lookup_actions(df, l.source_file, l.source_index)
        if actions is None:
            continue
        rep = replay_actions(l.grid, actions)
        if not rep.solved:
            continue
        ours_push.append(r["push_length"])
        ours_move.append(r["move_length"])
        theirs_move.append(rep.moves)
        theirs_push.append(rep.pushes)

    xcheck: Dict = {"n_compared": len(ours_push)}
    if ours_push:
        op = np.array(ours_push); om = np.array(ours_move)
        tm = np.array(theirs_move); tp = np.array(theirs_push)
        push_le = int((op <= tp).sum())
        move_ge = int((om >= tm).sum())
        from scipy.stats import spearmanr
        xcheck.update({
            "pearson_ourpush_vs_theirmoves": float(np.corrcoef(op, tm)[0, 1]),
            "spearman_ourpush_vs_theirmoves": float(spearmanr(op, tm).statistic),
            "pearson_ourmove_vs_theirmoves": float(np.corrcoef(om, tm)[0, 1]),
            "spearman_ourmove_vs_theirmoves": float(spearmanr(om, tm).statistic),
            "our_push_le_their_push_frac": push_le / len(op),
            "our_move_ge_their_move_frac": move_ge / len(om),
            "mean_our_push": float(op.mean()), "mean_their_push": float(tp.mean()),
            "mean_our_move": float(om.mean()), "mean_their_move": float(tm.mean()),
            "mean_move_excess_over_optimal": float((om - tm).mean()),
        })
        print()
        print(f"  Cross-check vs published solutions (n={len(op)} replay-validated):")
        print(f"     our push-optimal <= their pushes : {push_le}/{len(op)} "
              f"({pct(push_le, len(op))})   [must be 100%]")
        print(f"     our reconstructed moves >= their move-optimal : {move_ge}/{len(om)} "
              f"({pct(move_ge, len(om))})   [must be 100%]")
        print(f"     mean pushes  ours={op.mean():.2f}  theirs={tp.mean():.2f}")
        print(f"     mean moves   ours={om.mean():.2f}  theirs={tm.mean():.2f}  "
              f"(excess {float((om-tm).mean()):+.2f})")
        print(f"     Spearman(our push, their moves) = "
              f"{xcheck['spearman_ourpush_vs_theirmoves']:.4f}")

    # Player-region sizes: the empirical magnitude of the canonicalisation win.
    regions = []
    for l in levels:
        g = Grid.from_string(l.grid)
        regions.append(len(cells(player_reachable(g.walls, g.boxes, g.player))))
    reg = np.array(regions)
    print()
    print(f"  Player region size at start: mean={reg.mean():.2f} "
          f"median={np.median(reg):.1f} max={reg.max()} "
          f"(levels with region==1: {int((reg==1).sum())})")

    return {"summary": s, "false_unsolvable": false_unsolvable,
            "bad_replay": bad_replay, "cross_check": xcheck,
            "player_region": {"mean": float(reg.mean()),
                              "median": float(np.median(reg)),
                              "max": int(reg.max()),
                              "n_region_1": int((reg == 1).sum())},
            "per_level": [{"level": r["tag"], "status": r["status"],
                           "push_length": r["push_length"],
                           "move_length": r["move_length"],
                           "nodes": r["nodes"], "time_s": r["time_s"]}
                          for r in results]}


# --------------------------------------------------------------------------
# 2. Soundness ablation
# --------------------------------------------------------------------------
def soundness_ablation(levels, node_cap, time_cap, workers, n_sample, seed) -> Dict:
    hr(f"VALIDATION 2  Soundness ablation (pruning OFF, {10}x node cap)")
    rng = random.Random(seed)
    sample = rng.sample(levels, min(n_sample, len(levels)))

    on = run_pool([(l.grid, node_cap, time_cap, True, "pushes", l.source_index)
                   for l in sample], workers, "pruning ON ")
    off = run_pool([(l.grid, node_cap * 10, time_cap * 10, False, "pushes",
                     l.source_index) for l in sample], workers, "pruning OFF")

    on_by = {r["tag"]: r for r in on}
    off_by = {r["tag"]: r for r in off}
    disagree_status, disagree_length = [], []
    for tag in on_by:
        a, b = on_by[tag], off_by[tag]
        if a["status"] != b["status"]:
            disagree_status.append({"level": tag, "pruned": a["status"],
                                    "unpruned": b["status"]})
        elif a["push_length"] != b["push_length"]:
            disagree_length.append({"level": tag, "pruned": a["push_length"],
                                    "unpruned": b["push_length"]})

    nodes_on = float(np.median([r["nodes"] for r in on]))
    nodes_off = float(np.median([r["nodes"] for r in off]))
    print()
    print(f"  verdict disagreements : {len(disagree_status)}")
    print(f"  length disagreements  : {len(disagree_length)}")
    if disagree_status:
        print(f"  *** {disagree_status[:10]}")
    print(f"  median nodes  pruned={nodes_on:.0f}  unpruned={nodes_off:.0f}  "
          f"(speedup {nodes_off/max(1.0,nodes_on):.2f}x)")
    passed = not disagree_status and not disagree_length
    print(f"\n  => ablation {'PASSES' if passed else 'FAILS'}: pruning is "
          f"{'sound' if passed else 'UNSOUND'} on this sample")

    return {"n_sample": len(sample), "passed": passed,
            "disagree_status": disagree_status,
            "disagree_length": disagree_length,
            "median_nodes_pruned": nodes_on,
            "median_nodes_unpruned": nodes_off,
            "node_reduction_factor": nodes_off / max(1.0, nodes_on)}


# --------------------------------------------------------------------------
# 3. Medium split
# --------------------------------------------------------------------------
def validate_medium(levels_root, node_cap, time_cap, workers, n, seed) -> Dict:
    hr(f"VALIDATION 3  Medium split ({n} levels)")
    files = split_files(levels_root, "medium-valid")
    rng = random.Random(seed)
    levels = []
    for path in files[:5]:
        levels.extend(load_level_file(path, "medium-valid"))
    sample = rng.sample(levels, min(n, len(levels)))
    results = run_pool([(l.grid, node_cap, time_cap, True, "pushes",
                         f"{l.source_file}#{l.source_index}") for l in sample],
                       workers, "solve")
    s = summarise(results)
    print_summary("medium/valid", s)
    return {"summary": s}


# --------------------------------------------------------------------------
# 4. Node-cap sensitivity
# --------------------------------------------------------------------------
def node_cap_curve(levels, caps, time_cap, workers) -> Dict:
    hr("VALIDATION 4  Solve rate vs node cap")
    curve = []
    for cap in caps:
        results = run_pool([(l.grid, cap, time_cap, True, "pushes", l.source_index)
                            for l in levels], workers, f"cap={cap:>9,}")
        s = summarise(results)
        c = s["counts"]
        curve.append({"node_cap": cap, "solved": c["solved"],
                      "unsolvable": c["unsolvable"], "timeout": c["timeout"],
                      "solve_rate": s["solve_rate"],
                      "total_time_s": s["total_time_s"]})
        print(f"     cap={cap:>9,}  solved={c['solved']:>4} ({pct(c['solved'],s['n'])})  "
              f"unsolvable={c['unsolvable']:>3}  timeout={c['timeout']:>3}")
    print()
    print("  => a flat tail means the headline number is a converged lower bound")
    return {"curve": curve}


# --------------------------------------------------------------------------
# 5. Move-bound validation (Known Unknown 6)
# --------------------------------------------------------------------------
def validate_move_bound(levels, node_cap, time_cap, workers, n, seed) -> Dict:
    hr(f"VALIDATION 5  Reconstructed move length vs exact move-optimal ({n} levels)")
    rng = random.Random(seed)
    sample = rng.sample(levels, min(n, len(levels)))
    recon = run_pool([(l.grid, node_cap, time_cap, True, "pushes", l.source_index)
                      for l in sample], workers, "pushes mode")
    exact = run_pool([(l.grid, node_cap, time_cap * 6, True, "moves", l.source_index)
                      for l in sample], workers, "moves mode ")

    rb = {r["tag"]: r for r in recon}
    eb = {r["tag"]: r for r in exact}
    pairs = [(rb[t]["move_length"], eb[t]["move_length"]) for t in rb
             if rb[t]["status"] == "solved" and eb[t]["status"] == "solved"]
    violations = [(t) for t in rb if rb[t]["status"] == "solved"
                  and eb[t]["status"] == "solved"
                  and rb[t]["move_length"] < eb[t]["move_length"]]
    out: Dict = {"n_compared": len(pairs), "n_exact_timeout":
                 sum(1 for r in exact if r["status"] == "timeout"),
                 "bound_violations": violations}
    if pairs:
        r_ = np.array([p[0] for p in pairs], dtype=float)
        e_ = np.array([p[1] for p in pairs], dtype=float)
        excess = r_ - e_
        out.update({
            "mean_reconstructed": float(r_.mean()),
            "mean_move_optimal": float(e_.mean()),
            "mean_excess": float(excess.mean()),
            "median_excess": float(np.median(excess)),
            "max_excess": float(excess.max()),
            "frac_exactly_optimal": float((excess == 0).mean()),
            "mean_relative_excess": float((excess / e_).mean()),
        })
        print(f"  compared {len(pairs)} levels "
              f"({out['n_exact_timeout']} exact-mode timeouts excluded)")
        print(f"  reconstructed mean = {r_.mean():.2f}, move-optimal mean = {e_.mean():.2f}")
        print(f"  excess: mean={excess.mean():+.2f}  median={np.median(excess):+.1f}  "
              f"max={excess.max():+.0f}")
        print(f"  exactly move-optimal on {100*float((excess==0).mean()):.1f}% of levels")
        print(f"  mean relative excess = {100*out['mean_relative_excess']:.1f}%")
        print(f"  bound violations (reconstructed < optimal): {len(violations)}  "
              f"[must be 0]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-root", default="data_raw/boxoban-levels")
    ap.add_argument("--solutions-root", default="data_raw/astar-solutions")
    ap.add_argument("--out", default="results/solver_validation.json")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--node-cap", type=int, default=200_000)
    ap.add_argument("--time-cap", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--ablation-n", type=int, default=200)
    ap.add_argument("--medium-n", type=int, default=200)
    ap.add_argument("--movebound-n", type=int, default=200)
    ap.add_argument("--cap-curve", type=int, nargs="*",
                    default=[1_000, 10_000, 100_000, 1_000_000])
    args = ap.parse_args()

    print(f"seed={args.seed}  node_cap={args.node_cap:,}  time_cap={args.time_cap}s  "
          f"workers={args.workers}")

    levels = load_level_file(
        os.path.join(args.levels_root, "unfiltered", "test", "000.txt"),
        "unfiltered-test")
    print(f"loaded {len(levels)} real test levels")

    real = validate_real(levels, args.node_cap, args.time_cap, args.workers,
                         args.solutions_root)
    ablation = soundness_ablation(levels, args.node_cap, args.time_cap,
                                  args.workers, args.ablation_n, args.seed)
    medium = validate_medium(args.levels_root, args.node_cap, args.time_cap,
                             args.workers, args.medium_n, args.seed)
    curve = node_cap_curve(levels, args.cap_curve, max(args.time_cap, 60.0),
                           args.workers)
    movebound = validate_move_bound(levels, args.node_cap, args.time_cap,
                                    args.workers, args.movebound_n, args.seed)

    solve_rate = real["summary"]["solve_rate"]
    gate_solve = solve_rate >= 0.995
    gate_unsolv = len(real["false_unsolvable"]) == 0
    gate_abl = ablation["passed"]
    gate_replay = len(real["bad_replay"]) == 0
    gate = gate_solve and gate_unsolv and gate_abl and gate_replay

    payload = {
        "provenance": stamp(vars(args), args.seed),
        "gate2": {
            "passed": bool(gate),
            "solve_rate": solve_rate,
            "solve_rate_threshold": 0.995,
            "solve_rate_ok": bool(gate_solve),
            "zero_false_unsolvable": bool(gate_unsolv),
            "ablation_passed": bool(gate_abl),
            "all_solutions_replay": bool(gate_replay),
        },
        "validation1_real_levels": real,
        "validation2_soundness_ablation": ablation,
        "validation3_medium": medium,
        "validation4_node_cap_curve": curve,
        "validation5_move_bound": movebound,
    }
    write_artifact(args.out, payload)

    hr("GATE 2 SUMMARY")
    print(f"  solve rate on 1,000 real test levels : {100*solve_rate:.2f}%  "
          f"(need >= 99.50%)  {'OK' if gate_solve else 'FAIL'}")
    print(f"  false 'unsolvable' verdicts          : "
          f"{len(real['false_unsolvable'])}  {'OK' if gate_unsolv else 'FAIL'}")
    print(f"  returned solutions all replay        : "
          f"{'OK' if gate_replay else 'FAIL'}")
    print(f"  soundness ablation                   : "
          f"{'OK' if gate_abl else 'FAIL'}")
    print(f"\n  GATE 2: {'PASS' if gate else 'FAIL'}")
    print(f"  artifact -> {args.out}")
    if not gate:
        print("\n  Degraded mode (spec 4.8): disable freeze pruning, raise the node "
              "cap, reduce N per condition, and report the larger timeout bucket.")
        sys.exit(1)


if __name__ == "__main__":
    main()
