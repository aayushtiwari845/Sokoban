"""Seed variance on the main transformer configuration (spec 10.4).

Three independent *training* runs of the primary configuration, each sampled and
solved identically, reported as mean +/- sd.  Every other row in the paper is
single-seed and its table caption says so; this script is what licenses that
statement.

    python scripts/03_train.py --model transformer --seed 1337
    python scripts/03_train.py --model transformer --seed 1338 --suffix _s1338
    python scripts/03_train.py --model transformer --seed 1339 --suffix _s1339
    python scripts/06_seed_variance.py

Sampling uses the GPU and solving uses the CPU, so the two phases are run
strictly one after the other (spec 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from sokogen.provenance import stamp, write_artifact


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def sample_one_seed(ckpt: str, prompts, target_valid: int, seed: int) -> Dict:
    """Generate constrained samples from one checkpoint (GPU phase)."""
    import torch
    from sokogen.decoding.constrained import ForcingStats
    from sokogen.eval.generate import (PromptSampler, collect,
                                       transformer_drawer)
    from sokogen.models.common import load_checkpoint, setup_device
    from sokogen.models.transformer import SokobanLM, TransformerConfig

    dev = setup_device(verbose=False)
    sd, meta = load_checkpoint(ckpt, map_location=dev.device)
    c = meta.get("config", {})
    model = SokobanLM(TransformerConfig(
        d_model=c.get("d_model", 384), n_layers=c.get("n_layers", 6),
        n_heads=c.get("n_heads", 6), d_ff=c.get("d_ff", 1536),
        dropout=c.get("dropout", 0.1)))
    model.load_state_dict(sd)
    model = model.to(dev.device).eval()

    stats = ForcingStats()
    draw = transformer_drawer(model, dev.device, constrained=True, seed=seed,
                              stats=stats)
    out = collect(f"seed{seed}", draw, PromptSampler(prompts, seed=seed),
                  target_valid=target_valid, max_drawn=10 * target_valid)
    del model
    torch.cuda.empty_cache()
    return {
        "checkpoint": ckpt,
        "val_loss": meta.get("val_loss"),
        "n_params": meta.get("n_params"),
        "grids": [s.grid for s in out.samples],
        "samples_drawn": out.samples_drawn,
        "n_valid": out.n_valid_seen,
        "generation_time_s": out.generation_time_s,
        "forcing": stats.summary(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--suite", default="configs/prompt_suite.json")
    ap.add_argument("--out", default="results/seed_variance.json")
    ap.add_argument("--target-valid", type=int, default=500)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seeds = cfg["evaluation"]["main_seeds"]
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    from sokogen.eval.prompts import load_suite
    prompts = load_suite(args.suite)["in_distribution"]

    ck = cfg["paths"]["checkpoints_dir"]
    paths = []
    for i, s in enumerate(seeds):
        p = os.path.join(ck, "transformer.pt" if i == 0
                         else f"transformer_s{s}.pt")
        if not os.path.exists(p):
            print(f"  missing checkpoint for seed {s}: {p}")
            continue
        paths.append((s, p))

    if len(paths) < 2:
        print("  need at least two trained seeds; train them first")
        sys.exit(1)

    # -- GPU phase: sample every seed before any solving starts -------------
    hr("Sampling (GPU)")
    sampled = {}
    for s, p in paths:
        t = time.perf_counter()
        sampled[s] = sample_one_seed(p, prompts, args.target_valid, s)
        print(f"  seed {s}: {len(sampled[s]['grids'])} valid from "
              f"{sampled[s]['samples_drawn']:,} draws  "
              f"val_loss={sampled[s]['val_loss']:.4f}  "
              f"({time.perf_counter()-t:.1f}s)")

    # -- CPU phase: solve, with no GPU job running -------------------------
    hr("Solving (CPU)")
    from sokogen.eval import metrics as M
    from sokogen.eval.harness import solve_many

    node_cap = cfg["solver"]["node_cap"]
    time_cap = cfg["solver"]["time_cap_s"]
    per_seed: Dict[int, Dict] = {}
    for s, _ in paths:
        grids = sampled[s]["grids"]
        solved = solve_many(grids, node_cap, time_cap, "pushes", workers)
        statuses = [r["status"] for r in solved]
        valid = M.wilson(sampled[s]["n_valid"], sampled[s]["samples_drawn"])
        sgv = M.wilson(sum(1 for x in statuses if x == "solved"), len(statuses))
        per_seed[s] = {
            "val_loss": sampled[s]["val_loss"],
            "samples_drawn": sampled[s]["samples_drawn"],
            "structural_validity": valid.to_dict(),
            "solvable_given_valid": sgv.to_dict(),
            "outcomes": M.outcome_breakdown(statuses),
            "diversity": M.mean_pairwise_distance(M.grids_to_array(grids)),
            "forcing": sampled[s]["forcing"],
        }
        print(f"  seed {s}: valid={valid.p*100:.2f}%  "
              f"solvable|valid={sgv.p*100:.2f}%  "
              f"diversity={per_seed[s]['diversity']['mean']:.2f}")

    # -- aggregate ---------------------------------------------------------
    hr("Seed variance on the main transformer configuration")

    def agg(fn):
        vals = [fn(per_seed[s]) for s, _ in paths]
        return {"values": vals, "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    summary = {
        "seeds": [s for s, _ in paths],
        "structural_validity_pct": agg(lambda d: d["structural_validity"]["pct"]),
        "solvable_given_valid_pct": agg(lambda d: d["solvable_given_valid"]["pct"]),
        "diversity": agg(lambda d: d["diversity"]["mean"]),
        "val_loss": agg(lambda d: d["val_loss"]),
        "forced_sequence_rate": agg(
            lambda d: 100 * d["forcing"]["forced_sequence_rate"]),
    }
    for key, a in summary.items():
        if key == "seeds":
            continue
        vals = ", ".join(f"{v:.3f}" for v in a["values"])
        print(f"  {key:<28} {a['mean']:.3f} +/- {a['sd']:.3f}   [{vals}]")

    write_artifact(args.out, {
        "provenance": stamp(vars(args), cfg["seed"]),
        "n_seeds": len(paths),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "summary": summary,
        "note": ("Three independent training runs of the primary transformer "
                 "configuration. Every other row in the paper is single-seed "
                 "and its caption says so."),
    })
    print(f"\n  artifact -> {args.out}")


if __name__ == "__main__":
    main()
