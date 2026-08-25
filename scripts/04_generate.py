"""Phase 8a -- sample every evaluation condition.

Writes one JSONL per condition to ``results/generated/<condition>.jsonl`` plus a
manifest at ``results/generation.json``.  Solving happens in
``scripts/05_evaluate.py`` so that the GPU sampling job and the CPU solver sweep
never run at the same time (spec 0).

    python scripts/04_generate.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from sokogen.baselines import open_room, random_gen, rule_based
from sokogen.decoding.constrained import ForcingStats
from sokogen.eval.generate import (DEFAULT_MAX_DRAWN, DEFAULT_TARGET_VALID,
                                   PromptSampler, baseline_drawer, collect,
                                   fixed_set_drawer, gan_drawer,
                                   generate_for_suite, retrieval_drawer,
                                   transformer_drawer, vae_drawer)
from sokogen.eval.prompts import load_suite
from sokogen.models.common import (load_checkpoint, load_jsonl, set_seed,
                                   setup_device)
from sokogen.provenance import stamp, write_artifact


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_transformer(path, device, kind="transformer"):
    if not os.path.exists(path):
        return None
    sd, meta = load_checkpoint(path, map_location=device)
    if kind == "transformer":
        from sokogen.models.transformer import SokobanLM, TransformerConfig
        c = meta.get("config", {})
        model = SokobanLM(TransformerConfig(
            d_model=c.get("d_model", 384), n_layers=c.get("n_layers", 6),
            n_heads=c.get("n_heads", 6), d_ff=c.get("d_ff", 1536),
            dropout=c.get("dropout", 0.1)))
    else:
        from sokogen.models.distilgpt2 import build_distilgpt2
        from sokogen.data.vocab import VOCAB_SIZE
        model = build_distilgpt2(VOCAB_SIZE)
    model.load_state_dict(sd)
    return model.to(device).eval(), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--suite", default="configs/prompt_suite.json")
    ap.add_argument("--out-dir", default="results/generated")
    ap.add_argument("--manifest", default="results/generation.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--target-valid", type=int, default=DEFAULT_TARGET_VALID)
    ap.add_argument("--max-drawn", type=int, default=DEFAULT_MAX_DRAWN)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these condition names")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    dev = setup_device()

    suite = load_suite(args.suite)
    prompts = suite["in_distribution"]
    data_dir = cfg["paths"]["data_dir"]
    train_rows = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    test_rows = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    print(f"  {len(prompts)} suite captions | train {len(train_rows):,} | "
          f"test {len(test_rows):,}")

    ck = cfg["paths"]["checkpoints_dir"]
    conditions: Dict[str, Dict] = {}

    # -- non-learned baselines ------------------------------------------
    conditions["random_placement"] = {
        "draw": baseline_drawer(random_gen.generate, seed)}
    conditions["open_room"] = {"draw": baseline_drawer(open_room.generate, seed)}
    conditions["rule_based"] = {"draw": baseline_drawer(rule_based.generate, seed)}
    conditions["retrieval"] = {"draw": retrieval_drawer(train_rows, seed)}
    conditions["real_boxoban"] = {
        "draw": fixed_set_drawer([r["level"] for r in test_rows], seed)}

    # -- transformer ------------------------------------------------------
    forcing_stats: Dict[str, ForcingStats] = {}
    tf = load_transformer(os.path.join(ck, "transformer.pt"), dev.device)
    if tf is not None:
        model, meta = tf
        print(f"  transformer loaded: {meta.get('n_params'):,} params, "
              f"val_loss={meta.get('val_loss'):.4f}")
        forcing_stats["transformer_constrained"] = ForcingStats()
        conditions["transformer_constrained"] = {
            "draw": transformer_drawer(
                model, dev.device, constrained=True, seed=seed,
                stats=forcing_stats["transformer_constrained"]),
            "meta": meta}
        conditions["transformer_unconstrained"] = {
            "draw": transformer_drawer(model, dev.device, constrained=False,
                                       seed=seed),
            "meta": meta}
        # Temperature sweep for the solvability-vs-diversity Pareto (Figure 5).
        # These rows need only solvability and diversity, so they skip the
        # controllability suites.
        for temp in cfg["evaluation"]["temperatures"]:
            conditions[f"transformer_constrained_t{temp}"] = {
                "draw": transformer_drawer(model, dev.device, constrained=True,
                                           temperature=temp, seed=seed),
                "meta": meta, "suite": False, "temperature": temp}
    else:
        print("  transformer checkpoint missing -- skipping")

    dg = load_transformer(os.path.join(ck, "distilgpt2.pt"), dev.device,
                          kind="distilgpt2")
    if dg is not None:
        model, meta = dg
        forcing_stats["distilgpt2_constrained"] = ForcingStats()
        conditions["distilgpt2_constrained"] = {
            "draw": transformer_drawer(
                model, dev.device, constrained=True, seed=seed,
                stats=forcing_stats["distilgpt2_constrained"]),
            "meta": meta}
    else:
        print("  distilgpt2 checkpoint missing -- skipping (ablation)")

    # -- VAE ---------------------------------------------------------------
    vae_path = os.path.join(ck, "vae.pt")
    if os.path.exists(vae_path):
        from sokogen.models.vae import ConditionalVAE
        sd, meta = load_checkpoint(vae_path, map_location=dev.device)
        hp = meta.get("config", {})
        vae = ConditionalVAE(latent_dim=hp.get("latent_dim", 32),
                             hidden=hp.get("hidden", 512))
        vae.load_state_dict(sd)
        vae = vae.to(dev.device).eval()
        print(f"  vae loaded: {meta.get('n_params'):,} params")
        for mode in ("argmax", "sample", "repaired"):
            conditions[f"vae_{mode}"] = {
                "draw": vae_drawer(vae, dev.device, mode, seed), "meta": meta}
    else:
        print("  vae checkpoint missing -- skipping")

    # -- GAN ---------------------------------------------------------------
    gan_path = os.path.join(ck, "gan.pt")
    if os.path.exists(gan_path):
        from sokogen.models.gan import Generator
        sd, meta = load_checkpoint(gan_path, map_location=dev.device)
        hp = meta.get("config", {})
        G = Generator(hp.get("z_dim", 64), hp.get("hidden", 512))
        G.load_state_dict(sd)
        G = G.to(dev.device).eval()
        print(f"  gan loaded: {meta.get('n_params'):,} params")
        for mode in ("argmax", "repaired"):
            name = "gan_raw" if mode == "argmax" else "gan_repaired"
            conditions[name] = {
                "draw": gan_drawer(G, dev.device, mode,
                                   z_dim=hp.get("z_dim", 64), seed=seed),
                "meta": meta}
    else:
        print("  gan checkpoint missing -- skipping")

    if args.only:
        conditions = {k: v for k, v in conditions.items() if k in args.only}

    os.makedirs(args.out_dir, exist_ok=True)
    manifest: Dict[str, Dict] = {}

    hr("Sampling")
    for name, spec in conditions.items():
        sampler = PromptSampler(prompts, seed=seed + abs(hash(name)) % 100000)
        out = collect(name, spec["draw"], sampler,
                      target_valid=args.target_valid,
                      max_drawn=args.max_drawn)
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for s in out.samples:
                fh.write(json.dumps(s.to_dict()) + "\n")
        # A bounded record of raw draws, for tile-count histograms.
        raw_path = os.path.join(args.out_dir, f"{name}.raw.jsonl")
        with open(raw_path, "w", encoding="utf-8") as fh:
            for s in out.all_drawn:
                fh.write(json.dumps(s.to_dict()) + "\n")

        # Fixed controllability suites: a set number of samples per caption,
        # keeping failures, so per-attribute control is measured on what the
        # model produced for each specific request.
        for tag, suite_prompts in ((("suite", suite["in_distribution"]),
                                    ("ood", suite["out_of_distribution"]))
                                   if spec.get("suite", True) else ()):
            suite_samples = generate_for_suite(spec["draw"], suite_prompts)
            spath = os.path.join(args.out_dir, f"{name}.{tag}.jsonl")
            with open(spath, "w", encoding="utf-8") as fh:
                for s_ in suite_samples:
                    fh.write(json.dumps(s_.to_dict()) + "\n")

        summary = out.summary()
        if "meta" in spec:
            summary["model_params"] = spec["meta"].get("n_params")
        if "temperature" in spec:
            summary["temperature"] = spec["temperature"]
        manifest[name] = summary
        flag = "" if out.reached_target else "  <-- SHORTFALL"
        print(f"  {name:<28} drawn={out.samples_drawn:>7,}  "
              f"valid={out.n_valid_seen:>6,}  kept={len(out.samples):>4}  "
              f"{out.generation_time_s:>6.1f}s{flag}")

    for key, st in forcing_stats.items():
        if key in manifest and st.n_sequences:
            manifest[key]["forcing"] = st.summary()

    write_artifact(args.manifest, {
        "provenance": stamp(vars(args), seed),
        "target_valid": args.target_valid,
        "max_drawn": args.max_drawn,
        "conditions": manifest,
    })

    hr("DONE")
    print(f"  samples   -> {args.out_dir}/")
    print(f"  manifest  -> {args.manifest}")
    for key, st in forcing_stats.items():
        if st.n_sequences:
            s = st.summary()
            print(f"  forcing [{key}]: fired on "
                  f"{100*s['forced_sequence_rate']:.2f}% of levels, "
                  f"mean {s['mean_forced_tokens_per_level']:.3f} forced "
                  f"tokens/level")


if __name__ == "__main__":
    main()
