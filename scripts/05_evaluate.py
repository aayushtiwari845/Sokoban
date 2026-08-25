"""Phase 8b -- solve every condition, compute every metric, write every table.

This module deliberately imports **no torch**.  Windows uses spawn for
multiprocessing, so every solver worker re-imports this file as ``__mp_main__``;
importing torch here would make each of ~19 workers load the CUDA DLLs, which is
both wasteful and flaky, and the resulting contention corrupts the very
per-level solver timings this script reports.

    python scripts/05_evaluate.py --config configs/main.yaml --out results/

Produces ``results/evaluation.json`` (the single source for all tables and
figures), the supplementary per-level verdict dump, and then regenerates the
tables and figures from that JSON.  No number in the paper is ever typed by
hand.

Runs the CPU solver sweep -- do not run concurrently with a GPU job (spec 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from sokogen.data.captions import CaptionBins
from sokogen.eval import metrics as M
from sokogen.eval.harness import (evaluate_condition, load_samples, solve_many)
from sokogen.provenance import stamp, write_artifact


def load_jsonl(path):
    """Local, dependency-free reader.

    Deliberately not imported from ``sokogen.data.build``: that module pulls in
    pandas, and every spawned solver worker would pay for it on startup.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows

# Order of rows in the main results table (spec 11).
ROW_ORDER = [
    "random_placement", "open_room", "rule_based", "retrieval",
    "gan_raw", "gan_repaired",
    "vae_argmax", "vae_sample", "vae_repaired",
    "transformer_unconstrained", "transformer_constrained",
    "distilgpt2_constrained",
    "real_boxoban",
]


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def emit_tables_and_figures(args, cfg) -> None:
    """Render every table, figure and the README from ``evaluation.json``.

    Kept separate so it can be rerun with ``--tables-only`` after a formatting
    fix, without repeating the solver sweep that produced the numbers.
    """
    hr("Tables and figures")
    from sokogen.eval import figures as F
    from sokogen.eval import tables as T
    from sokogen.eval.report import write_readme

    results_dir = os.path.dirname(args.out)
    tables_dir = os.path.join(results_dir, "tables")
    for name, path in T.write_all(args.out, tables_dir, results_dir).items():
        print(f"  table  {name:<28} -> {path}")
    for name, paths in F.write_all(args.out, args.gen_dir,
                                   cfg["paths"]["figures_dir"],
                                   results_dir).items():
        print(f"  figure {name:<28} -> {paths[0]}")
    readme = write_readme(args.out, results_dir, "README.md",
                          os.path.join(results_dir, "solver_validation.json"))
    print(f"  readme {'(regenerated from artifacts)':<28} -> {readme}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--gen-dir", default="results/generated")
    ap.add_argument("--manifest", default="results/generation.json")
    ap.add_argument("--out", default="results/evaluation.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-train-nn", type=int, default=100000,
                    help="training levels used for nearest-neighbour novelty")
    ap.add_argument("--tables-only", action="store_true",
                    help="regenerate tables, figures and README from an "
                         "existing evaluation.json without re-running the "
                         "solver sweep")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed = args.seed if args.seed is not None else cfg["seed"]
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    node_cap = cfg["solver"]["node_cap"]
    time_cap = cfg["solver"]["time_cap_s"]
    length_mode = cfg["solver"].get("length_cost_mode", "moves")
    min_denom = cfg["evaluation"]["min_denominator_flag"]

    if args.tables_only:
        emit_tables_and_figures(args, cfg)
        return

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)["conditions"]

    data_dir = cfg["paths"]["data_dir"]
    bins = CaptionBins.from_dict(
        json.load(open(os.path.join(data_dir, "caption_bins.json"),
                       encoding="utf-8")))
    train_rows = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    test_rows = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    train_grids = [r["level"] for r in train_rows]
    nn_grids = train_grids[: args.max_train_nn]
    train_arr = M.grids_to_array(nn_grids)

    print(f"  workers={workers}  node_cap={node_cap:,}  time_cap={time_cap}s")
    print(f"  solvability cost_mode=pushes | achieved length cost_mode={length_mode}")
    print(f"  train levels for novelty: {len(nn_grids):,}")

    names = [n for n in ROW_ORDER if n in manifest]
    extra = [n for n in manifest if n not in ROW_ORDER]
    names += extra

    hr("Evaluating conditions")
    results = {}
    t0 = time.perf_counter()
    for name in names:
        t = time.perf_counter()
        results[name] = evaluate_condition(
            name, args.gen_dir, manifest, train_grids, train_arr, bins,
            node_cap, time_cap, length_mode, workers, min_denom)
        r = results[name]
        sv = r["structural_validity"]
        sg = r["solvable_given_valid"]
        so = r["solvable_overall"]
        print(f"  {name:<28} valid={sv['pct']:>6.2f}%  "
              f"solv|valid={sg['pct'] if sg['pct']==sg['pct'] else float('nan'):>6.2f}%  "
              f"solv_all={so['pct'] if so['pct'] is not None else float('nan'):>6.2f}%  "
              f"drawn={r['samples_drawn']:>7,}  ({time.perf_counter()-t:.1f}s)")
        if "small_denominator_flag" in r:
            print(f"      ! {r['small_denominator_flag']}")
        if "notes" in r:
            print(f"      ! {r['notes']}")

    # -- reference distributions on held-out real levels -------------------
    hr("Reference distributions (held-out real levels)")
    real_grids = [r["level"] for r in test_rows][:500]
    reference = {
        "n": len(real_grids),
        "novelty": M.novelty(real_grids, train_grids, train_arr=train_arr),
        "diversity": M.mean_pairwise_distance(M.grids_to_array(real_grids)),
    }
    print(f"  real levels: novelty NN mean = "
          f"{reference['novelty']['nn_distance']['mean']:.2f}, "
          f"diversity = {reference['diversity']['mean']:.2f}")
    print("  (every novelty/diversity number in the table is read against these)")

    # -- pairwise significance tests ---------------------------------------
    hr("Pairwise comparisons (two-proportion z-tests)")
    comparisons = {}
    pairs = [("transformer_constrained", "open_room"),
             ("transformer_constrained", "rule_based"),
             ("transformer_constrained", "transformer_unconstrained"),
             ("transformer_unconstrained", "vae_sample"),
             ("transformer_constrained", "real_boxoban"),
             ("vae_repaired", "vae_argmax")]
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        ra, rb = results[a], results[b]
        ka = ra["outcomes"]["counts"]["solved"]
        na = ra["outcomes"]["n"]
        kb = rb["outcomes"]["counts"]["solved"]
        nb = rb["outcomes"]["n"]
        t = M.two_proportion_z_test(ka, na, kb, nb)
        comparisons[f"{a}__vs__{b}"] = {
            "metric": "solvable_given_valid", "a": a, "b": b,
            "a_k": ka, "a_n": na, "b_k": kb, "b_n": nb, **t}
        if t["p_value"] is not None:
            print(f"  {a} vs {b}: diff={t['diff_pct']:+.1f}pp  "
                  f"z={t['z']:.2f}  p={t['p_value']:.2e}")

    payload = {
        "provenance": stamp(vars(args), seed),
        "config": cfg,
        "caption_bins": bins.to_dict(),
        "conditions": results,
        "reference_distributions": reference,
        "comparisons": comparisons,
        "row_order": names,
        "total_eval_time_s": time.perf_counter() - t0,
        "seed_policy": ("3 seeds on the main transformer config only; every "
                        "other row is single-seed and declared as such."),
    }
    write_artifact(args.out, payload)

    # -- supplementary: per-level verdicts ---------------------------------
    supp_dir = os.path.join(os.path.dirname(args.out), "supplementary")
    os.makedirs(supp_dir, exist_ok=True)
    for name in names:
        samples = load_samples(os.path.join(args.gen_dir, f"{name}.jsonl"))
        if not samples:
            continue
        solved = solve_many([s["grid"] for s in samples], node_cap, time_cap,
                            "pushes", workers)
        with open(os.path.join(supp_dir, f"{name}_verdicts.jsonl"), "w",
                  encoding="utf-8") as fh:
            for s, r in zip(samples, solved):
                fh.write(json.dumps({
                    "grid": s["grid"], "caption_text": s["caption_text"],
                    "requested": s["requested"], "status": r["status"],
                    "push_length": r["push_length"],
                    "move_length": r["move_length"],
                    "nodes_expanded": r["nodes"],
                    "solver_time_s": r["time_s"]}) + "\n")

    emit_tables_and_figures(args, cfg)

    hr("DONE")
    print(f"  evaluation    -> {args.out}")
    print(f"  supplementary -> {supp_dir}/")
    print(f"  tables        -> {tables_dir}/")
    print(f"  figures       -> {cfg['paths']['figures_dir']}/")
    print(f"  total time    -> {payload['total_eval_time_s']:.0f}s")


if __name__ == "__main__":
    main()
