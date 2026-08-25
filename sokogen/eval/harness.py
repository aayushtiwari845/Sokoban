"""Evaluation harness: solve every condition and compute every metric.

Runs the CPU solver sweep.  Never run this while a GPU training or sampling job
is active: the GPU job heats the package, the CPU cores throttle, and solver
throughput silently halves mid-sweep, corrupting the timing measurements
(spec 0).

Two cost modes are used deliberately:

* **Solvability verdicts** use ``cost_mode="pushes"`` -- fast, exhaustive and
  exactly push-optimal.
* **Achieved solution length** for controllability uses ``cost_mode="moves"``.
  Captions are written in move units, and Phase 2 measured the reconstructed
  move length to be **+40.5% over move-optimal on average** (exactly optimal on
  only 6.6% of levels), which would badly distort the correlation.  Exact mode
  costs ~114 ms/level, which the budget absorbs comfortably.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.boxoban import is_structurally_valid, tile_counts
from ..data.captions import CaptionBins, bin_indices, raw_features
from ..solver.astar import solve
from . import metrics as M


# ---------------------------------------------------------------------------
# Solver workers (top level so they pickle under Windows spawn)
# ---------------------------------------------------------------------------
def _solve_job(job: Tuple) -> Dict:
    grid, node_cap, time_cap, cost_mode, tag = job
    r = solve(grid, node_cap=node_cap, time_cap_s=time_cap, cost_mode=cost_mode)
    return {"tag": tag, "status": r.status, "push_length": r.push_length,
            "move_length": r.move_length, "nodes": r.nodes_expanded,
            "time_s": r.wall_time_s}


def solve_many(grids: Sequence[str], node_cap: int, time_cap: float,
               cost_mode: str, workers: int, chunksize: int = 8) -> List[Dict]:
    jobs = [(g, node_cap, time_cap, cost_mode, i) for i, g in enumerate(grids)]
    if not jobs:
        return []
    if workers <= 1:
        return [_solve_job(j) for j in jobs]
    with mp.Pool(workers) as pool:
        return pool.map(_solve_job, jobs, chunksize=chunksize)


# ---------------------------------------------------------------------------
# Proportion of a product of two rates
# ---------------------------------------------------------------------------
def product_proportion(p_valid: M.Proportion, p_solv: M.Proportion) -> Dict:
    """Solvable-out-of-all-samples = P(valid) x P(solvable | valid).

    Invalid levels are not solvable -- they are not even well-formed puzzles --
    so the overall rate is the product of the two measured rates.  The interval
    uses the delta method on that product; it is reported separately from
    ``solvable | valid`` and the two are never merged, because merging them
    compares decoding schemes rather than model families.
    """
    if p_valid.n == 0 or p_solv.n == 0:
        return {"pct": None, "ci_lo_pct": None, "ci_hi_pct": None}
    p, q = p_valid.p, p_solv.p
    var_p = p * (1 - p) / p_valid.n
    var_q = q * (1 - q) / p_solv.n
    est = p * q
    se = math.sqrt(max(0.0, q * q * var_p + p * p * var_q))
    return {
        "pct": 100 * est,
        "ci_lo_pct": 100 * max(0.0, est - 1.96 * se),
        "ci_hi_pct": 100 * min(1.0, est + 1.96 * se),
        "method": "delta method on P(valid) x P(solvable|valid)",
    }


# ---------------------------------------------------------------------------
# Loading generated samples
# ---------------------------------------------------------------------------
def load_samples(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def tile_count_stats(samples: Sequence[Dict]) -> Dict:
    """Boxes / goals / player per generated level -- Figure 3.

    Computed over **raw draws**, valid or not: the histogram of how many boxes a
    family emits is the most direct evidence for the counting-constraint claim,
    and restricting it to valid levels would define the failure away.
    """
    counts = {"@": [], "$": [], ".": []}
    for s in samples:
        c = tile_counts(s["grid"])
        for k in counts:
            counts[k].append(c.get(k, 0))
    out = {}
    for k, v in counts.items():
        a = np.array(v, dtype=float)
        name = {"@": "player", "$": "box", ".": "goal"}[k]
        out[name] = {
            "mean": float(a.mean()) if len(a) else None,
            "std": float(a.std()) if len(a) else None,
            "exact_rate": float((a == (1 if k == "@" else 4)).mean()) if len(a) else None,
            "histogram": {str(int(x)): int((a == x).sum())
                          for x in sorted(set(a.tolist()))},
            "n": len(a),
        }
    return out


# ---------------------------------------------------------------------------
# Controllability
# ---------------------------------------------------------------------------
def achieved_bins(grid: str, bins: CaptionBins,
                  solution_length: Optional[float]) -> Optional[Dict]:
    """Measure what a generated level actually is, in the same bin space."""
    if not is_structurally_valid(grid):
        return None
    feats = raw_features(grid)
    feats["solution_length"] = solution_length if solution_length is not None else 0.0
    b = bin_indices(feats, bins)
    b["solution_length"] = solution_length
    return b


def evaluate_controllability(samples: Sequence[Dict], solved: Sequence[Dict],
                             bins: CaptionBins) -> Dict:
    """Per-attribute control accuracy on the fixed prompt suite."""
    requested, achieved, mask = [], [], []
    n_invalid = 0
    for s, r in zip(samples, solved):
        req = dict(s["requested"])
        ach = achieved_bins(s["grid"], bins,
                            r["move_length"] if r["status"] == "solved" else None)
        if ach is None:
            n_invalid += 1
            continue
        requested.append(req)
        achieved.append(ach)
        mask.append(r["status"] == "solved")

    out = M.controllability(requested, achieved, mask)
    out["n_samples"] = len(samples)
    out["n_invalid_excluded"] = n_invalid
    out["invalid_exclusion_rate"] = n_invalid / len(samples) if samples else None
    out["note"] = ("Bin accuracy is computed on structurally valid samples "
                   "only; invalid levels have no well-defined tile statistics. "
                   "The exclusion rate is reported so the reader can see how "
                   "much of the sample that removes.")
    return out


# ---------------------------------------------------------------------------
# One condition end to end
# ---------------------------------------------------------------------------
def evaluate_condition(name: str, gen_dir: str, manifest: Dict,
                       train_grids: Sequence[str], train_arr: np.ndarray,
                       bins: CaptionBins, node_cap: int, time_cap: float,
                       length_cost_mode: str, workers: int,
                       min_denominator: int = 30) -> Dict:
    samples = load_samples(os.path.join(gen_dir, f"{name}.jsonl"))
    raw = load_samples(os.path.join(gen_dir, f"{name}.raw.jsonl"))
    suite = load_samples(os.path.join(gen_dir, f"{name}.suite.jsonl"))
    ood = load_samples(os.path.join(gen_dir, f"{name}.ood.jsonl"))

    man = manifest.get(name, {})
    drawn = man.get("samples_drawn", len(raw))
    n_valid = man.get("n_valid", sum(1 for s in raw if s.get("valid")))

    validity = M.wilson(n_valid, drawn)
    result: Dict = {
        "name": name,
        "samples_drawn": drawn,
        "n_valid_seen": n_valid,
        "n_evaluated": len(samples),
        "reached_target": man.get("reached_target"),
        "structural_validity": validity.to_dict(),
        "structural_validity_note": (
            "Sampling stops once the valid target is reached, so this is an "
            "inverse-binomial rather than fixed-n proportion; the Wilson "
            "interval is a close approximation but not exact."),
        "generation_time_s": man.get("generation_time_s"),
        "generation_time_per_sample_ms": man.get("generation_time_per_sample_ms"),
        "model_params": man.get("model_params"),
        "forcing": man.get("forcing"),
        "tile_counts_raw_draws": tile_count_stats(raw),
    }
    if man.get("notes"):
        result["notes"] = man["notes"]

    # -- solvability -------------------------------------------------------
    grids = [s["grid"] for s in samples]
    t0 = time.perf_counter()
    solved = solve_many(grids, node_cap, time_cap, "pushes", workers)
    result["solver_time_total_s"] = time.perf_counter() - t0

    statuses = [r["status"] for r in solved]
    result["outcomes"] = M.outcome_breakdown(statuses)
    n_solved = sum(1 for s in statuses if s == "solved")
    solv_given_valid = M.wilson(n_solved, len(statuses))
    result["solvable_given_valid"] = solv_given_valid.to_dict()
    result["solvable_overall"] = product_proportion(validity, solv_given_valid)

    flag = M.flag_small_denominator(solv_given_valid, min_denominator)
    if flag:
        result["small_denominator_flag"] = flag

    if solved:
        times = np.array([r["time_s"] for r in solved])
        nodes = np.array([r["nodes"] for r in solved])
        result["solver_cost"] = {
            "time_per_level_ms_median": float(np.median(times) * 1000),
            "time_per_level_ms_mean": float(times.mean() * 1000),
            "nodes_median": float(np.median(nodes)),
            "nodes_p99": float(np.percentile(nodes, 99)),
        }
        lens = [r["push_length"] for r in solved if r["status"] == "solved"]
        if lens:
            result["push_length"] = {
                "mean": float(np.mean(lens)), "std": float(np.std(lens)),
                "median": float(np.median(lens)),
                "min": int(min(lens)), "max": int(max(lens))}

    # -- novelty and diversity --------------------------------------------
    if grids:
        result["novelty"] = M.novelty(grids, train_grids, train_arr=train_arr)
        result["diversity"] = M.mean_pairwise_distance(M.grids_to_array(grids))

    # -- controllability ---------------------------------------------------
    for tag, rows in (("controllability", suite), ("ood", ood)):
        if not rows:
            continue
        sgrids = [s["grid"] for s in rows]
        # Valid rows get an exact move-optimal length; invalid ones are skipped
        # by achieved_bins anyway, so do not spend solver time on them.
        solved_suite: List[Dict] = []
        idx_valid = [i for i, s in enumerate(rows) if is_structurally_valid(s["grid"])]
        sub = solve_many([sgrids[i] for i in idx_valid], node_cap, time_cap,
                         length_cost_mode, workers)
        by_i = {i: r for i, r in zip(idx_valid, sub)}
        for i in range(len(rows)):
            solved_suite.append(by_i.get(i, {"status": "invalid",
                                             "move_length": None,
                                             "push_length": None,
                                             "nodes": 0, "time_s": 0.0}))
        result[tag] = evaluate_controllability(rows, solved_suite, bins)
        result[tag]["outcomes"] = M.outcome_breakdown(
            [r["status"] for r in solved_suite if r["status"] != "invalid"])

    return result
