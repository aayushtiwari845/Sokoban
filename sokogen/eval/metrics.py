"""Evaluation metrics (spec 10.2).

Design decisions that the paper states explicitly:

* **Structural validity and solvability are never merged.**  Merging them
  compares decoding schemes rather than model families and invalidates the
  central claim.  ``solvable_pct`` (of all samples) and ``solvable_given_valid``
  (of valid samples) are separate columns with separate Wilson intervals.
* **Three-way outcomes.**  solved / proven unsolvable / timed out, never
  collapsed to two.  Exhaustive A* is a decision procedure, so "proven
  unsolvable" is a real result and must not be hidden inside "not solved".
* **Novelty is symmetry-aware.**  A 90-degree rotation of a training level is
  not novel, so a generated level counts as a copy if *any* of its 8 dihedral
  variants appears in the training set.
* **Distance is cell-wise Hamming over the 100 tiles**, not Levenshtein.  All
  grids are exactly 10x10, so insertions and deletions would shift cells across
  row boundaries and compare cells that are not spatially corresponding --
  semantically wrong for a grid.  On equal-length strings Levenshtein reduces to
  substitutions in almost every case anyway, and Hamming is exact and vectorised.
* **Reference distributions are mandatory.**  "Mean pairwise distance = 34.2" is
  uninterpretable alone, so every novelty and diversity number is reported
  beside the same measurement on held-out real levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..data.boxoban import is_structurally_valid
from ..solver.grid import PERMS

TILE_ORDER = {"#": 0, " ": 1, "$": 2, ".": 3, "@": 4}
Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Proportion:
    """A count with a Wilson score interval."""

    k: int
    n: int
    p: float
    lo: float
    hi: float

    def to_dict(self) -> Dict:
        return {"k": self.k, "n": self.n, "pct": 100 * self.p,
                "ci_lo_pct": 100 * self.lo, "ci_hi_pct": 100 * self.hi}

    def fmt(self) -> str:
        if self.n == 0:
            return "n/a"
        return f"{100*self.p:.1f} [{100*self.lo:.1f}, {100*self.hi:.1f}]"


def wilson(k: int, n: int, z: float = Z95) -> Proportion:
    """Wilson score interval -- correct near 0% and 100%, unlike normal-approx."""
    if n == 0:
        return Proportion(k, n, float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = centre - half, centre + half
    # The Wilson interval provably contains p; at k == 0 or k == n rounding can
    # place an endpoint a few ULPs the wrong side of it, so clamp to [0,1] and
    # to p itself rather than reporting an interval that excludes its estimate.
    lo = min(max(0.0, lo), p)
    hi = max(min(1.0, hi), p)
    return Proportion(k, n, p, lo, hi)


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> Dict:
    """Pooled two-proportion z-test for pairwise comparisons (spec 10.4)."""
    if n1 == 0 or n2 == 0:
        return {"z": None, "p_value": None}
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0, "diff_pct": 100 * (p1 - p2)}
    z = (p1 - p2) / se
    from scipy.stats import norm
    return {"z": z, "p_value": float(2 * (1 - norm.cdf(abs(z)))),
            "diff_pct": 100 * (p1 - p2)}


# ---------------------------------------------------------------------------
# Grid encoding and distances
# ---------------------------------------------------------------------------
def grids_to_array(grids: Sequence[str]) -> np.ndarray:
    """``[N, 100]`` uint8 of tile indices.  Unknown chars map to floor."""
    out = np.empty((len(grids), 100), dtype=np.uint8)
    for i, g in enumerate(grids):
        body = g.replace("\n", "")
        body = (body + " " * 100)[:100]
        out[i] = np.frombuffer(
            bytes(TILE_ORDER.get(c, 1) for c in body), dtype=np.uint8)
    return out


def dihedral_variants(arr: np.ndarray) -> List[np.ndarray]:
    """All 8 symmetric variants of an ``[N, 100]`` tile array."""
    return [arr[:, list(perm)] for perm in _INVERSE_PERMS]


# PERMS[k][i] maps source cell i -> destination index.  To build the transformed
# array by gathering we need the inverse: destination j <- source perm^-1[j].
_INVERSE_PERMS: List[Tuple[int, ...]] = []
for _perm in PERMS:
    _inv = [0] * 100
    for _src, _dst in enumerate(_perm):
        _inv[_dst] = _src
    _INVERSE_PERMS.append(tuple(_inv))


def mean_pairwise_distance(arr: np.ndarray, max_n: int = 500,
                           seed: int = 1337) -> Dict:
    """Mean pairwise cell-wise Hamming distance -- the diversity metric."""
    n = len(arr)
    if n < 2:
        return {"mean": None, "std": None, "n": n}
    if n > max_n:
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(n, max_n, replace=False)]
    dists = []
    for i in range(len(arr) - 1):
        d = (arr[i + 1:] != arr[i]).sum(axis=1)
        dists.append(d)
    d = np.concatenate(dists)
    return {"mean": float(d.mean()), "std": float(d.std()),
            "median": float(np.median(d)), "n_pairs": int(d.size),
            "n": int(len(arr))}


def nearest_neighbour_distances(gen: np.ndarray, train: np.ndarray,
                                symmetry_aware: bool = True) -> np.ndarray:
    """Distance from each generated level to its closest training level.

    With ``symmetry_aware`` the minimum is taken over all 8 dihedral variants of
    the generated level, matching the novelty definition.
    """
    variants = dihedral_variants(gen) if symmetry_aware else [gen]
    best = np.full(len(gen), 101, dtype=np.int32)
    for var in variants:
        for i in range(len(var)):
            d = (train != var[i]).sum(axis=1).min()
            if d < best[i]:
                best[i] = d
    return best


def novelty(gen_grids: Sequence[str], train_grids: Sequence[str],
            train_arr: Optional[np.ndarray] = None,
            compute_nn: bool = True, max_train: Optional[int] = None) -> Dict:
    """Symmetry-aware exact-copy rate plus the nearest-neighbour distribution.

    A generated level counts as a copy when **any** of its 8 dihedral variants
    is byte-identical to a training level.
    """
    train_hashes = set(train_grids)
    gen_arr = grids_to_array(gen_grids)

    # Compare as exact strings so "copy" means byte-identical, not merely close.
    copies = 0
    chars = "# $.@"
    for i in range(len(gen_arr)):
        for perm in _INVERSE_PERMS:
            body = gen_arr[i][list(perm)]
            s = "".join(chars[int(t)] for t in body)
            grid = "".join(s[r * 10:(r + 1) * 10] + "\n" for r in range(10))
            if grid in train_hashes:
                copies += 1
                break

    out: Dict = {
        "n": len(gen_grids),
        "exact_copy": wilson(copies, len(gen_grids)).to_dict(),
        "novel_pct": 100 * (1 - copies / len(gen_grids)) if gen_grids else None,
        "symmetry_aware": True,
    }

    if compute_nn and len(gen_grids):
        if train_arr is None:
            tg = list(train_grids)
            if max_train is not None and len(tg) > max_train:
                tg = tg[:max_train]
            train_arr = grids_to_array(tg)
        nn = nearest_neighbour_distances(gen_arr, train_arr)
        out["nn_distance"] = {
            "mean": float(nn.mean()), "std": float(nn.std()),
            "median": float(np.median(nn)), "min": int(nn.min()),
            "max": int(nn.max()),
            "deciles": [float(np.percentile(nn, q)) for q in range(0, 101, 10)],
        }
    return out


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def structural_validity(grids: Sequence[str]) -> Proportion:
    return wilson(sum(1 for g in grids if is_structurally_valid(g)), len(grids))


def outcome_breakdown(statuses: Sequence[str]) -> Dict:
    """Three-way solved / unsolvable / timeout counts and rates."""
    n = len(statuses)
    counts = {s: sum(1 for x in statuses if x == s)
              for s in ("solved", "unsolvable", "timeout")}
    return {
        "n": n,
        "counts": counts,
        "solved": wilson(counts["solved"], n).to_dict(),
        "unsolvable": wilson(counts["unsolvable"], n).to_dict(),
        "timeout": wilson(counts["timeout"], n).to_dict(),
    }


# ---------------------------------------------------------------------------
# Controllability
# ---------------------------------------------------------------------------
def controllability(requested: Sequence[Dict], achieved: Sequence[Dict],
                    solved_mask: Sequence[bool]) -> Dict:
    """Per-attribute control accuracy.  Never reported as a single number.

    Wall density and box clustering are computable directly from the grid, so a
    model that reproduces surface statistics will nail them.  Solution length is
    **not** computable without search, so it is the only attribute that tests
    real understanding -- and the gap between them is the interesting finding.

    Solution length uses Spearman rank correlation and reports the **censoring
    rate**: an achieved length exists only for *solved* levels, so the
    correlation is computed on a non-random subsample biased toward easy levels,
    which is the direction that inflates it.
    """
    out: Dict = {}

    for attr in ("density", "clustering", "connectivity", "difficulty"):
        req = [r[attr] for r in requested if attr in r]
        ach = [a[attr] for a in achieved if attr in a]
        if not req or len(req) != len(ach):
            continue
        hits = sum(1 for a, b in zip(req, ach) if a == b)
        out[attr] = {
            "bin_accuracy": wilson(hits, len(req)).to_dict(),
            "n": len(req),
            "directly_computable_from_grid": attr in ("density", "clustering",
                                                      "connectivity"),
        }

    req_len, ach_len = [], []
    for r, a, ok in zip(requested, achieved, solved_mask):
        if not ok or a.get("solution_length") is None:
            continue
        req_len.append(r["solution_length"])
        ach_len.append(a["solution_length"])

    n_total = len(requested)
    n_used = len(req_len)
    censoring = 1.0 - (n_used / n_total) if n_total else None
    length: Dict = {
        "n_requested": n_total,
        "n_with_achieved_length": n_used,
        "censoring_rate": censoring,
        "censoring_note": (
            "Achieved length exists only for solved levels; timeouts and "
            "proven-unsolvable levels have none, so this correlation is "
            "computed on a subsample biased toward easy levels."),
    }
    if n_used >= 3 and len(set(req_len)) > 1 and len(set(ach_len)) > 1:
        from scipy.stats import spearmanr
        res = spearmanr(req_len, ach_len)
        length.update({
            "spearman": float(res.statistic),
            "spearman_p": float(res.pvalue),
            "pearson": float(np.corrcoef(req_len, ach_len)[0, 1]),
            "mean_requested": float(np.mean(req_len)),
            "mean_achieved": float(np.mean(ach_len)),
            "mean_abs_error": float(np.mean(np.abs(
                np.array(req_len) - np.array(ach_len)))),
        })
    out["solution_length"] = length
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def flag_small_denominator(prop: Proportion, threshold: int = 30) -> Optional[str]:
    """Inline warning for a ratio computed on too few samples (spec 10.1).

    A GAN at 2% validity yields ~10 levels, and "40%" there is noise a reader
    will misread as signal.
    """
    if prop.n < threshold:
        return (f"DENOMINATOR {prop.n} < {threshold}: this percentage is noise, "
                f"not signal")
    return None
