"""Phase 3 -- build the captioned datasets.  GATE 3.

Writes ``data/{train,val,test}.jsonl`` plus ``data/caption_bins.json`` and the
artifact ``results/dataset_build.json``.  Prints the fitted bin boundaries and a
sample of 20 (level, caption) pairs for manual sanity checking, which is the
GATE 3 requirement.

Run:
    python scripts/01_build_dataset.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sokogen.data.build import (build_rows, collect_levels, fit_bins_for,
                                levels_with_lengths, write_jsonl)
from sokogen.data.captions import (CLUSTERING_WORDS, CONNECTIVITY_WORDS,
                                   DENSITY_WORDS, DIFFICULTY_WORDS, bin_indices,
                                   features)
from sokogen.data.vocab import MAX_LEN, VOCAB_SIZE, encode_example
from sokogen.provenance import stamp, write_artifact


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-root", default="data_raw/boxoban-levels")
    ap.add_argument("--solutions-root", default="data_raw/astar-solutions")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--results", default="results/dataset_build.json")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=5_000)
    ap.add_argument("--n-test", type=int, default=1_000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    hr("Collecting levels and replay-validated solution lengths")
    spec = [("train", "unfiltered-train", args.n_train),
            ("val", "unfiltered-valid", args.n_val),
            ("test", "unfiltered-test", args.n_test)]

    pairs = {}
    stats = {}
    for name, split, n in spec:
        levels = collect_levels(args.levels_root, split, n)
        p, st = levels_with_lengths(levels, args.solutions_root, split)
        pairs[name] = p
        stats[name] = st
        print(f"  {name:<6} {split:<18} requested={n:>7,}  "
              f"kept={st['kept']:>7,}  dropped={st['dropped_no_valid_solution']:>5,} "
              f"({100*st['dropped_no_valid_solution']/max(1,st['total']):.2f}%)")

    hr("Fitting caption bins on the TRAINING split only")
    bins = fit_bins_for(pairs["train"])
    lo, hi = bins.difficulty_terciles
    print(f"  difficulty terciles (moves) : {lo:.2f}, {hi:.2f}")
    print(f"      {DIFFICULTY_WORDS[0]:<7} moves <= {lo:.0f}")
    print(f"      {DIFFICULTY_WORDS[1]:<7} {lo:.0f} < moves <= {hi:.0f}")
    print(f"      {DIFFICULTY_WORDS[2]:<7} moves > {hi:.0f}")
    print(f"  wall-density median         : {bins.density_median:.4f}  "
          f"(<= -> '{DENSITY_WORDS[0]}', > -> '{DENSITY_WORDS[1]}')")
    print(f"  floor-degree median         : {bins.degree_median:.4f}  "
          f"(<= -> '{CONNECTIVITY_WORDS[0]}', > -> '{CONNECTIVITY_WORDS[1]}')")
    print(f"  box-clustering median       : {bins.clustering_median:.4f}  "
          f"(<= -> '{CLUSTERING_WORDS[0]}', > -> '{CLUSTERING_WORDS[1]}')")
    print(f"  length range for normalising: [{bins.length_min:.0f}, "
          f"{bins.length_max:.0f}]")

    bins_path = os.path.join(args.out_dir, "caption_bins.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(bins_path, "w", encoding="utf-8") as fh:
        json.dump(bins.to_dict(), fh, indent=2)
    print(f"\n  bins -> {bins_path}")

    hr("Writing datasets")
    balance = {}
    longest_seq = 0
    for name, _, _ in spec:
        rows = build_rows(pairs[name], bins)
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        write_jsonl(path, rows)
        for r in rows[: min(len(rows), 20000)]:
            longest_seq = max(longest_seq, len(encode_example(r["caption_text"],
                                                              r["level"])))
        counts = collections.Counter()
        for r in rows:
            b = bin_indices(features(r["level"], r["solution_length"]), bins)
            counts[(DIFFICULTY_WORDS[b["difficulty"]],
                    DENSITY_WORDS[b["density"]],
                    CONNECTIVITY_WORDS[b["connectivity"]],
                    CLUSTERING_WORDS[b["clustering"]])] += 1
        balance[name] = {
            "n_rows": len(rows),
            "difficulty": dict(collections.Counter(k[0] for k in counts.elements())),
            "density": dict(collections.Counter(k[1] for k in counts.elements())),
            "connectivity": dict(collections.Counter(k[2] for k in counts.elements())),
            "clustering": dict(collections.Counter(k[3] for k in counts.elements())),
        }
        print(f"  {name:<6} {len(rows):>7,} rows -> {path}")

    print(f"\n  longest encoded sequence seen: {longest_seq} (MAX_LEN={MAX_LEN}, "
          f"vocab={VOCAB_SIZE})")
    assert longest_seq <= MAX_LEN, "a training sequence exceeds MAX_LEN"

    hr("Caption balance (train)")
    for key in ("difficulty", "density", "connectivity", "clustering"):
        total = sum(balance["train"][key].values())
        parts = ", ".join(f"{k}={100*v/total:.1f}%"
                          for k, v in sorted(balance["train"][key].items()))
        print(f"  {key:<13}: {parts}")

    hr("GATE 3  Twenty (level, caption) pairs for manual inspection")
    sample = rng.sample(range(len(pairs["train"])), 20)
    train_rows = build_rows([pairs["train"][i] for i in sample], bins)
    for k, r in enumerate(train_rows):
        print(f"\n  [{k+1}] {r['caption_text']}")
        print(f"      solution_length={r['solution_length']} moves  "
              f"vec={[round(x,3) for x in r['caption_vec']]}")
        for line in r["level"].rstrip("\n").split("\n"):
            print(f"      {line}")

    payload = {
        "provenance": stamp(vars(args), args.seed),
        "split_stats": stats,
        "caption_bins": bins.to_dict(),
        "balance": balance,
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "longest_sequence_observed": longest_seq,
        "connectivity_feature_note": (
            "The spec offered 'count of connected floor components' or 'mean "
            "corridor width'.  Measured over 20,000 training levels every "
            "Boxoban level has exactly ONE floor component, so the component "
            "count is constant and cannot bin anything.  We use mean floor "
            "degree (non-wall neighbours per floor cell) instead."),
    }
    write_artifact(args.results, payload)

    hr("GATE 3 SUMMARY")
    for name, _, _ in spec:
        print(f"  {name:<6}: {balance[name]['n_rows']:>7,} rows")
    print(f"  bins logged      -> {bins_path}")
    print(f"  artifact         -> {args.results}")


if __name__ == "__main__":
    main()
