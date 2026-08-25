"""Conditional GAN over 10x10 tile grids (spec 8).

Time-boxed.  The result the paper needs from this family is a low
structural-validity number, and a poorly-trained GAN provides that just as well
as a well-trained one.  Mode collapse is a **reportable finding**, not a project
failure, so training curves and the diversity number are logged regardless of
outcome.

Discretisation: **straight-through Gumbel-softmax**, temperature annealed
1.0 -> 0.3.  The naive alternative -- train on continuous logits and argmax only
at sampling time -- is not a valid option: the discriminator can then separate
real one-hot tensors from soft generator outputs purely by per-cell entropy,
training collapses to that trivial discriminator, and the generator learns
nothing about layout.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 5
N_CELLS = 100
COND_DIM = 10


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64, hidden: int = 512, cond_dim: int = COND_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_CHANNELS * N_CELLS),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Returns logits ``[B, 5, 100]``."""
        return self.net(torch.cat([z, cond], dim=1)).reshape(-1, N_CHANNELS, N_CELLS)

    def sample_soft(self, z, cond, tau: float, hard: bool = True) -> torch.Tensor:
        """Straight-through Gumbel-softmax over the channel dimension."""
        logits = self.forward(z, cond)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)


class Discriminator(nn.Module):
    def __init__(self, hidden: int = 512, cond_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_CHANNELS * N_CELLS + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, tiles_onehot: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = tiles_onehot.reshape(tiles_onehot.shape[0], -1)
        return self.net(torch.cat([x, cond], dim=1))


def anneal_tau(step: int, total: int, start: float, end: float) -> float:
    frac = min(1.0, max(0.0, step / max(1, total)))
    return start + (end - start) * frac


def train_conditional_gan(args, cfg, dev) -> Dict:
    from .common import count_params, encode_cond_dataset, load_jsonl, save_checkpoint

    hp = cfg["gan"]
    rows = load_jsonl(os.path.join(cfg["paths"]["data_dir"], "train.jsonl"))
    tiles, cond = encode_cond_dataset(rows)
    print(f"  train {len(tiles):,} levels")

    G = Generator(hp["z_dim"], hp["hidden"]).to(dev.device)
    D = Discriminator(hp["hidden"]).to(dev.device)
    n_params = count_params(G) + count_params(D)
    print(f"  parameters: G={count_params(G):,} D={count_params(D):,} "
          f"total={n_params:,}")

    betas = tuple(hp["betas"])
    optG = torch.optim.Adam(G.parameters(), lr=hp["lr"], betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=hp["lr"], betas=betas)

    batch = hp["batch_size"]
    steps_per_epoch = len(tiles) // batch
    total_steps = steps_per_epoch * hp["epochs"]
    rng = np.random.default_rng(args.seed)

    tiles_t = torch.from_numpy(tiles).long()
    cond_t = torch.from_numpy(cond)

    history: List[Dict] = []
    t0 = time.perf_counter()
    deadline = t0 + hp["time_box_hours"] * 3600
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "gan.pt")
    real_target = hp["label_smoothing"]
    step = 0
    stopped_early = False

    for epoch in range(hp["epochs"]):
        perm = rng.permutation(len(tiles_t))
        agg = {"d": 0.0, "g": 0.0, "n": 0}
        for b in range(steps_per_epoch):
            sel = torch.from_numpy(perm[b * batch:(b + 1) * batch].copy())
            real_idx = tiles_t[sel].to(dev.device)
            c = cond_t[sel].to(dev.device)
            real = F.one_hot(real_idx, N_CHANNELS).permute(0, 2, 1).float()

            tau = anneal_tau(step, total_steps, hp["gumbel_tau_start"],
                             hp["gumbel_tau_end"])

            # -- discriminator --
            z = torch.randn(real.shape[0], hp["z_dim"], device=dev.device)
            fake = G.sample_soft(z, c, tau=tau, hard=True)
            d_real = D(real, c)
            d_fake = D(fake.detach(), c)
            loss_d = (F.binary_cross_entropy_with_logits(
                          d_real, torch.full_like(d_real, real_target))
                      + F.binary_cross_entropy_with_logits(
                          d_fake, torch.zeros_like(d_fake)))
            optD.zero_grad(set_to_none=True)
            loss_d.backward()
            optD.step()

            # -- generator (non-saturating) --
            z = torch.randn(real.shape[0], hp["z_dim"], device=dev.device)
            fake = G.sample_soft(z, c, tau=tau, hard=True)
            d_out = D(fake, c)
            loss_g = F.binary_cross_entropy_with_logits(
                d_out, torch.ones_like(d_out))
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()

            agg["d"] += float(loss_d.detach())
            agg["g"] += float(loss_g.detach())
            agg["n"] += 1
            step += 1

        print(f"  epoch {epoch+1:>2}: loss_D={agg['d']/agg['n']:.4f}  "
              f"loss_G={agg['g']/agg['n']:.4f}  tau={tau:.3f}  "
              f"({time.perf_counter()-t0:.0f}s)")
        history.append({"epoch": epoch + 1, "loss_d": agg["d"] / agg["n"],
                        "loss_g": agg["g"] / agg["n"], "tau": tau})
        save_checkpoint(ckpt_path, G, {
            "model": "gan", "epoch": epoch + 1, "config": hp,
            "n_params": n_params,
            "discriminator": D.state_dict()})

        if time.perf_counter() > deadline:
            print(f"  time box of {hp['time_box_hours']}h reached -- stopping "
                  f"and shipping what we have (spec 8)")
            stopped_early = True
            break

    return {"n_params": n_params, "history": history,
            "train_time_s": time.perf_counter() - t0,
            "checkpoint": ckpt_path, "stopped_at_time_box": stopped_early}
