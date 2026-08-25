"""Decoder-only transformer LM, from scratch (spec 6.1).

Deliberately *not* a fine-tuned DistilGPT-2: the primary model is trained from
scratch so the family comparison is roughly parameter-fair (~10.7M vs ~1.5M vs
~3M).  An 82M-parameter pretrained model against a 2M VAE would not be.

Architecture: d_model 384, 6 layers, 6 heads (head dim 64), d_ff 1536, learned
positional embeddings, **pre-norm**, GELU, dropout 0.1, tied input/output
embeddings.  ~10.7M parameters.

The autoregressive factorisation is the point of the paper: a model emitting
cell-by-cell can condition on how many boxes it has already placed, which is
what makes the hard counting constraints satisfiable.  A one-shot decoder
emitting all 100 cells from a latent vector cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.vocab import MAX_LEN, PAD_ID, VOCAB_SIZE


@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_len: int = MAX_LEN
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, past=None, use_cache: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)

        # With a cache the single new query legitimately attends to everything
        # already generated, so no causal mask applies; a fresh prefill needs one.
        causal = past is None and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(y))
        return (out, (k, v)) if use_cache else (out, None)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, past=None, use_cache: bool = False):
        attn_out, present = self.attn(self.ln1(x), past=past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class SokobanLM(nn.Module):
    def __init__(self, cfg: TransformerConfig | None = None):
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(c.max_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)
        # Scaled init on residual projections (GPT-2 convention).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None,
                past=None, use_cache: bool = False):
        """``ids`` is ``[B, T]``.  Returns ``(logits, loss)`` or, with
        ``use_cache``, ``(logits, loss, presents)``.

        ``labels`` align position-for-position with ``ids``; the standard
        one-position shift is applied here, so a label of -100 at position ``t``
        means "do not train on predicting token ``t``".
        """
        B, T = ids.shape
        offset = 0 if past is None else past[0][0].shape[2]
        pos = torch.arange(offset, offset + T, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))

        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            x, present = block(x, past=None if past is None else past[i],
                               use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if use_cache:
            return logits, loss, presents
        return logits, loss

    @torch.no_grad()
    def logits_for_next(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits for the token following the last position of ``ids``."""
        logits, _ = self.forward(ids)
        return logits[:, -1, :]
