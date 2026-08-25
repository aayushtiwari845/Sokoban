"""Conditional VAE over 10x10 tile grids (spec 7).

MLP, **not convolutional**.  DCGAN-style conv stacks assume power-of-2 spatial
dims with strided convolutions; 10x10 does not factor cleanly, and at 100 cells
convolutions buy nothing while costing shape bugs.

Representation: 5x10x10 one-hot over (wall, floor, box, goal, player).  Phase 1
verified that no box ever starts on a goal, so one channel per cell is faithful.

Loss: **per-cell categorical cross-entropy** over the channel dimension, plus
KL.  Not MSE -- MSE on one-hot tiles is a common and wrong default (it treats
the channel axis as a metric space when it is categorical), and the paper says
so explicitly.

Posterior collapse is prevented with **free bits**: per-dimension KL is clamped
at a 0.05-nat floor.  One line, no schedule to tune, preferred over KL
annealing.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 5   # wall, floor, box, goal, player
N_CELLS = 100
COND_DIM = 10


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 512,
                 cond_dim: int = COND_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        flat = N_CHANNELS * N_CELLS  # 500

        self.encoder = nn.Sequential(
            nn.Linear(flat + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, flat),
        )

    @staticmethod
    def one_hot(tiles: torch.Tensor) -> torch.Tensor:
        """``[B, 100]`` channel indices -> ``[B, 5, 100]`` one-hot floats."""
        return F.one_hot(tiles, N_CHANNELS).permute(0, 2, 1).float()

    def encode(self, tiles: torch.Tensor, cond: torch.Tensor):
        x = self.one_hot(tiles).reshape(tiles.shape[0], -1)
        h = self.encoder(torch.cat([x, cond], dim=1))
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Returns logits shaped ``[B, 5, 100]`` (channel dim is categorical)."""
        out = self.decoder(torch.cat([z, cond], dim=1))
        return out.reshape(-1, N_CHANNELS, N_CELLS)

    def forward(self, tiles: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(tiles, cond)
        z = self.reparameterise(mu, logvar)
        return self.decode(z, cond), mu, logvar

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, generator=None) -> torch.Tensor:
        z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device,
                        generator=generator)
        return self.decode(z, cond)


def vae_loss(logits: torch.Tensor, tiles: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, free_bits: float = 0.05,
             kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    """Per-cell categorical cross-entropy + free-bits KL.

    ``free_bits`` clamps each latent dimension's KL at a floor, so the optimiser
    gains nothing by driving a dimension to exactly match the prior.  Returns
    ``(total, recon, kl)``, all summed per sample and averaged over the batch.
    """
    recon = F.cross_entropy(logits, tiles, reduction="none").sum(dim=1).mean()

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_clamped.sum(dim=1).mean()

    return recon + kl_weight * kl, recon, kl


@torch.no_grad()
def decode_grids(logits: torch.Tensor, mode: str = "argmax",
                 generator=None) -> torch.Tensor:
    """Turn decoder logits into tile indices under one of two protocols.

    Both protocols are mandatory in the results table (spec 7.2), because the
    decoder emits an **independent categorical per cell**:

    ``argmax``
        Deterministic.  Exposes marginal collapse: boxes occupy 4/100 cells,
        goals 4/100 and the player 1/100, so the per-cell argmax picks floor or
        wall almost everywhere and the output is a near-empty room.  The argmax
        of a product of marginals is not the mode of the joint, and rare tiles
        are exactly what gets annihilated.
    ``sample``
        Samples each cell from its categorical.  Preserves marginal tile rates,
        so roughly the right *number* of boxes with the wrong *placement* -- a
        different failure signature, and a fairer protocol to report alongside.
    """
    if mode == "argmax":
        return logits.argmax(dim=1)
    if mode != "sample":
        raise ValueError(f"unknown decode mode {mode!r}")
    probs = torch.softmax(logits, dim=1)                  # [B, 5, 100]
    b, c, n = probs.shape
    flat = probs.permute(0, 2, 1).reshape(b * n, c)
    idx = torch.multinomial(flat, num_samples=1, generator=generator)
    return idx.reshape(b, n)
