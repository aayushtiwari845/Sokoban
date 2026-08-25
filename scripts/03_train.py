"""Phase 4-6 -- train one model family.

    python scripts/03_train.py --model transformer
    python scripts/03_train.py --model distilgpt2
    python scripts/03_train.py --model vae
    python scripts/03_train.py --model gan

Writes a checkpoint to ``checkpoints/<model>.pt`` and a loss-curve artifact to
``results/train_<model>.json``.

Scheduling constraint (spec 0): never run this concurrently with the CPU solver
sweep.  The GPU job heats the package, the CPU cores throttle, and solver
throughput silently halves mid-sweep, corrupting the timing measurements.
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
import torch
import yaml

from sokogen.models.common import (count_params, cosine_lr, encode_cond_dataset,
                                   encode_lm_dataset, load_jsonl,
                                   make_lm_labels, save_checkpoint, set_seed,
                                   setup_device)
from sokogen.provenance import stamp, write_artifact


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Language-model training (transformer and the DistilGPT-2 ablation)
# ---------------------------------------------------------------------------
def train_lm(args, cfg, dev) -> Dict:
    from sokogen.data.vocab import VOCAB_SIZE

    key = "transformer" if args.model == "transformer" else "distilgpt2"
    hp = cfg[key]

    train_rows = load_jsonl(os.path.join(cfg["paths"]["data_dir"], "train.jsonl"))
    val_rows = load_jsonl(os.path.join(cfg["paths"]["data_dir"], "val.jsonl"))
    print(f"  train {len(train_rows):,} rows | val {len(val_rows):,} rows")

    tr_ids, tr_start = encode_lm_dataset(train_rows)
    va_ids, va_start = encode_lm_dataset(val_rows)

    if args.model == "transformer":
        from sokogen.models.transformer import SokobanLM, TransformerConfig
        model = SokobanLM(TransformerConfig(
            d_model=hp["d_model"], n_layers=hp["n_layers"], n_heads=hp["n_heads"],
            d_ff=hp["d_ff"], dropout=hp["dropout"],
            tie_embeddings=hp["tie_embeddings"])).to(dev.device)
    else:
        from sokogen.models.distilgpt2 import build_distilgpt2
        model = build_distilgpt2(VOCAB_SIZE).to(dev.device)

    n_params = count_params(model)
    print(f"  parameters: {n_params:,}")

    batch = hp["batch_size"]
    epochs = hp["epochs"]
    steps_per_epoch = len(tr_ids) // batch
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=cfg["transformer"]["weight_decay"])
    print(f"  {steps_per_epoch:,} steps/epoch x {epochs} epochs = {total_steps:,} steps")

    tr_ids_t = torch.from_numpy(tr_ids)
    tr_start_t = torch.from_numpy(tr_start.astype(np.int64))
    va_ids_t = torch.from_numpy(va_ids)
    va_start_t = torch.from_numpy(va_start.astype(np.int64))

    def evaluate() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_ids_t), 256):
                ids = va_ids_t[i:i + 256].to(dev.device).long()
                st = va_start_t[i:i + 256].to(dev.device)
                labels = make_lm_labels(ids, st)
                with torch.autocast("cuda", dtype=dev.dtype, enabled=dev.bf16):
                    _, loss = model(ids, labels=labels)
                total += float(loss) * ids.shape[0]
                n += ids.shape[0]
        model.train()
        return total / max(1, n)

    history: List[Dict] = []
    best_val = float("inf")
    step = 0
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"],
                             f"{args.model}{args.suffix}.pt")

    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(len(tr_ids_t))
        running, seen = 0.0, 0
        for b in range(steps_per_epoch):
            sel = torch.from_numpy(perm[b * batch:(b + 1) * batch].copy())
            ids = tr_ids_t[sel].to(dev.device).long()
            st = tr_start_t[sel].to(dev.device)
            labels = make_lm_labels(ids, st)

            lr = cosine_lr(step, total_steps, hp["lr"], cfg["transformer"]["warmup_steps"])
            for group in opt.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=dev.dtype, enabled=dev.bf16):
                _, loss = model(ids, labels=labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["transformer"]["grad_clip"])
            opt.step()

            running += float(loss)
            seen += 1
            step += 1
            if step % 200 == 0:
                print(f"    epoch {epoch+1} step {step:>5}/{total_steps}  "
                      f"train_loss={running/seen:.4f}  lr={lr:.2e}  "
                      f"({time.perf_counter()-t0:.0f}s)")
                history.append({"step": step, "train_loss": running / seen, "lr": lr})
                running, seen = 0.0, 0

        val_loss = evaluate()
        print(f"  epoch {epoch+1}: val_loss={val_loss:.4f}")
        history.append({"step": step, "epoch": epoch + 1, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_path, model, {
                "model": args.model, "epoch": epoch + 1, "val_loss": val_loss,
                "config": hp, "n_params": n_params, "vocab_size": VOCAB_SIZE})
            print(f"    checkpoint saved (best) -> {ckpt_path}")

    return {"n_params": n_params, "best_val_loss": best_val,
            "total_steps": total_steps, "history": history,
            "train_time_s": time.perf_counter() - t0,
            "checkpoint": ckpt_path}


# ---------------------------------------------------------------------------
# One-shot families
# ---------------------------------------------------------------------------
def train_vae(args, cfg, dev) -> Dict:
    from sokogen.models.vae import ConditionalVAE, vae_loss

    hp = cfg["vae"]
    train_rows = load_jsonl(os.path.join(cfg["paths"]["data_dir"], "train.jsonl"))
    val_rows = load_jsonl(os.path.join(cfg["paths"]["data_dir"], "val.jsonl"))
    tr_tiles, tr_cond = encode_cond_dataset(train_rows)
    va_tiles, va_cond = encode_cond_dataset(val_rows)
    print(f"  train {len(tr_tiles):,} | val {len(va_tiles):,}")

    model = ConditionalVAE(latent_dim=hp["latent_dim"], hidden=hp["hidden"]).to(dev.device)
    n_params = count_params(model)
    print(f"  parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    batch = hp["batch_size"]
    steps_per_epoch = len(tr_tiles) // batch
    rng = np.random.default_rng(args.seed)

    tr_t = torch.from_numpy(tr_tiles).long()
    tr_c = torch.from_numpy(tr_cond)
    va_t = torch.from_numpy(va_tiles).long()
    va_c = torch.from_numpy(va_cond)

    history: List[Dict] = []
    best_val = float("inf")
    t0 = time.perf_counter()
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "vae.pt")

    for epoch in range(hp["epochs"]):
        model.train()
        perm = rng.permutation(len(tr_t))
        agg = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
        for b in range(steps_per_epoch):
            sel = torch.from_numpy(perm[b * batch:(b + 1) * batch].copy())
            tiles = tr_t[sel].to(dev.device)
            cond = tr_c[sel].to(dev.device)
            logits, mu, logvar = model(tiles, cond)
            loss, recon, kl = vae_loss(logits, tiles, mu, logvar,
                                       free_bits=hp["free_bits"],
                                       kl_weight=hp["kl_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg["loss"] += float(loss); agg["recon"] += float(recon)
            agg["kl"] += float(kl); agg["n"] += 1

        model.eval()
        with torch.no_grad():
            vl = vr = vk = 0.0
            nb = 0
            for i in range(0, len(va_t), 512):
                tiles = va_t[i:i + 512].to(dev.device)
                cond = va_c[i:i + 512].to(dev.device)
                logits, mu, logvar = model(tiles, cond)
                loss, recon, kl = vae_loss(logits, tiles, mu, logvar,
                                           free_bits=hp["free_bits"],
                                           kl_weight=hp["kl_weight"])
                vl += float(loss); vr += float(recon); vk += float(kl); nb += 1
            vl /= nb; vr /= nb; vk /= nb

        print(f"  epoch {epoch+1:>2}: train_loss={agg['loss']/agg['n']:.4f} "
              f"(recon={agg['recon']/agg['n']:.4f} kl={agg['kl']/agg['n']:.4f})  "
              f"val_loss={vl:.4f} (recon={vr:.4f} kl={vk:.4f})")
        history.append({"epoch": epoch + 1,
                        "train_loss": agg["loss"] / agg["n"],
                        "train_recon": agg["recon"] / agg["n"],
                        "train_kl": agg["kl"] / agg["n"],
                        "val_loss": vl, "val_recon": vr, "val_kl": vk})
        if vl < best_val:
            best_val = vl
            save_checkpoint(ckpt_path, model, {
                "model": "vae", "epoch": epoch + 1, "val_loss": vl,
                "config": hp, "n_params": n_params})

    return {"n_params": n_params, "best_val_loss": best_val, "history": history,
            "train_time_s": time.perf_counter() - t0, "checkpoint": ckpt_path}


def train_gan(args, cfg, dev) -> Dict:
    from sokogen.models.gan import train_conditional_gan
    return train_conditional_gan(args, cfg, dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["transformer", "distilgpt2", "vae", "gan"])
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--suffix", default="",
                    help="appended to the checkpoint and artifact names, so "
                         "multi-seed runs do not overwrite each other")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.seed is None:
        args.seed = cfg["seed"]

    hr(f"Training {args.model}  (seed {args.seed})")
    set_seed(args.seed)
    dev = setup_device()
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)

    if args.model in ("transformer", "distilgpt2"):
        out = train_lm(args, cfg, dev)
    elif args.model == "vae":
        out = train_vae(args, cfg, dev)
    else:
        out = train_gan(args, cfg, dev)

    payload = {"provenance": stamp({"model": args.model, "config": args.config},
                                   args.seed),
               "model": args.model, **out}
    path = os.path.join(cfg["paths"]["results_dir"],
                        f"train_{args.model}{args.suffix}.json")
    write_artifact(path, payload)

    hr("DONE")
    print(f"  params      : {out['n_params']:,}")
    print(f"  train time  : {out['train_time_s']:.0f}s")
    if "best_val_loss" in out:
        print(f"  best val    : {out['best_val_loss']:.4f}")
    print(f"  checkpoint  : {out['checkpoint']}")
    print(f"  artifact    : {path}")


if __name__ == "__main__":
    main()
