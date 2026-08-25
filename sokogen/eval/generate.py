"""Sampling every evaluation condition under one protocol (spec 10.1).

Protocol
--------
For each condition, draw samples until **500 structurally valid levels** are
collected or a hard cap of **100,000 samples drawn** is reached.  Generation is
one forward pass and only *valid* samples are ever handed to the solver, so
drawing 50k GAN samples costs seconds and no solver time.

``samples_drawn`` is reported as its own column: it is a real efficiency metric
and it makes the validity gap vivid.  If a family cannot reach 500 valid within
the cap, the count obtained is reported with its Wilson interval and the
shortfall is stated.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.boxoban import is_structurally_valid
from ..decoding.repair import repair_grid
from ..models.common import tiles_to_grid
from .prompts import Prompt

DEFAULT_TARGET_VALID = 500
DEFAULT_MAX_DRAWN = 100_000

# A drawer turns a list of conditioning prompts into one sample each.  The same
# signature serves both the main sampling protocol and the controllability
# suite, so no family has two code paths that could drift apart.
Drawer = Callable[[Sequence["Prompt"]], List["Sample"]]


@dataclass
class Sample:
    grid: str
    caption_text: str
    caption_vec: List[float]
    requested: Dict
    valid: bool

    def to_dict(self) -> Dict:
        return {"grid": self.grid, "caption_text": self.caption_text,
                "caption_vec": self.caption_vec, "requested": self.requested,
                "valid": self.valid}


@dataclass
class ConditionOutput:
    name: str
    samples: List[Sample] = field(default_factory=list)      # valid ones kept
    all_drawn: List[Sample] = field(default_factory=list)    # a bounded record
    samples_drawn: int = 0
    n_valid_seen: int = 0
    generation_time_s: float = 0.0
    reached_target: bool = False
    notes: str = ""

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "samples_drawn": self.samples_drawn,
            "n_valid": self.n_valid_seen,
            "n_kept": len(self.samples),
            "generation_time_s": self.generation_time_s,
            "generation_time_per_sample_ms": (
                1000 * self.generation_time_s / max(1, self.samples_drawn)),
            "reached_target": self.reached_target,
            "notes": self.notes,
        }


def collect(name: str, draw: "Drawer", sampler: "PromptSampler",
            target_valid: int = DEFAULT_TARGET_VALID,
            max_drawn: int = DEFAULT_MAX_DRAWN,
            chunk: int = 512, keep_invalid: int = 2000) -> ConditionOutput:
    """Draw in chunks until the valid target or the hard cap is reached.

    ``draw(prompts)`` returns one sample per prompt with ``valid`` already set.
    A bounded sample of *invalid* draws is retained too, so tile-count
    histograms and the repair analysis can be computed over what the model
    actually emits rather than only over its successes.
    """
    out = ConditionOutput(name=name)
    t0 = time.perf_counter()
    while out.n_valid_seen < target_valid and out.samples_drawn < max_drawn:
        n = min(chunk, max_drawn - out.samples_drawn)
        batch = draw(sampler.draw(n))
        out.samples_drawn += len(batch)
        for s in batch:
            if len(out.all_drawn) < keep_invalid:
                out.all_drawn.append(s)
            if s.valid:
                out.n_valid_seen += 1
                if len(out.samples) < target_valid:
                    out.samples.append(s)
    out.generation_time_s = time.perf_counter() - t0
    out.reached_target = len(out.samples) >= target_valid
    if not out.reached_target:
        out.notes = (f"SHORTFALL: only {len(out.samples)} valid levels in "
                     f"{out.samples_drawn:,} draws (cap {max_drawn:,})")
    return out


# ---------------------------------------------------------------------------
# Prompt sampling shared by every conditional family
# ---------------------------------------------------------------------------
class PromptSampler:
    """Draws captions uniformly from the in-distribution suite.

    Every family sees the same caption distribution, which is what makes the
    cross-family comparison a controlled one.
    """

    def __init__(self, prompts: Sequence[Prompt], seed: int = 1337):
        self.prompts = list(prompts)
        self.rng = random.Random(seed)

    def draw(self, n: int) -> List[Prompt]:
        return [self.prompts[self.rng.randrange(len(self.prompts))]
                for _ in range(n)]


def _make_sample(grid: str, prompt: Prompt) -> Sample:
    return Sample(grid=grid, caption_text=prompt.caption_text,
                  caption_vec=list(prompt.caption_vec),
                  requested=prompt.requested_bins(),
                  valid=is_structurally_valid(grid))


# ---------------------------------------------------------------------------
# Non-learned baselines
# ---------------------------------------------------------------------------
def baseline_drawer(fn, seed: int = 1337) -> "Drawer":
    """Wrap a baseline ``generate(n, seed)`` into the sampling protocol.

    Baselines are unconditional, so a caption is attached only to keep the
    record shape uniform; their controllability is measured, and expected to be
    at chance, exactly like any other condition.
    """
    state = {"seed": seed}

    def draw(prompts: Sequence[Prompt]) -> List[Sample]:
        state["seed"] += 1
        grids = fn(len(prompts), seed=state["seed"])
        return [_make_sample(g, p) for g, p in zip(grids, prompts)]

    return draw


def retrieval_drawer(train_rows: Sequence[Dict], seed: int = 1337) -> "Drawer":
    rng = random.Random(seed)

    def draw(prompts: Sequence[Prompt]) -> List[Sample]:
        return [_make_sample(train_rows[rng.randrange(len(train_rows))]["level"], p)
                for p in prompts]

    return draw


def fixed_set_drawer(grids: Sequence[str], seed: int = 1337) -> "Drawer":
    """Cycles a fixed corpus (used for the real-Boxoban ceiling row)."""
    order = list(range(len(grids)))
    rng = random.Random(seed)
    rng.shuffle(order)
    state = {"i": 0}

    def draw(prompts: Sequence[Prompt]) -> List[Sample]:
        out = []
        for p in prompts:
            g = grids[order[state["i"] % len(order)]]
            state["i"] += 1
            out.append(_make_sample(g, p))
        return out

    return draw


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
def transformer_drawer(model, device, constrained: bool,
                       temperature: float = 1.0, seed: int = 1337,
                       stats=None) -> "Drawer":
    """Groups a chunk by caption so each forward batch shares one prompt."""
    from ..decoding.constrained import generate as constrained_generate
    state = {"seed": seed}

    def draw(prompts: Sequence[Prompt]) -> List[Sample]:
        by_caption: Dict[str, List[Prompt]] = {}
        for p in prompts:
            by_caption.setdefault(p.caption_text, []).append(p)
        out: List[Sample] = []
        for caption, group in by_caption.items():
            state["seed"] += 1
            grids = constrained_generate(
                model, caption, len(group), device, temperature=temperature,
                constrained=constrained, seed=state["seed"],
                batch_size=min(512, len(group)), stats=stats)
            out.extend(_make_sample(g, p) for g, p in zip(grids, group))
        return out

    return draw


# ---------------------------------------------------------------------------
# One-shot families (VAE, GAN)
# ---------------------------------------------------------------------------
def _logits_to_samples(logits: torch.Tensor, prompts: Sequence[Prompt],
                       mode: str, generator=None) -> List[Sample]:
    """Decode ``[B, 5, 100]`` logits under one of three protocols."""
    if mode == "argmax":
        tiles = logits.argmax(dim=1)
    elif mode == "sample":
        probs = torch.softmax(logits, dim=1)
        b, c, ncell = probs.shape
        flat = probs.permute(0, 2, 1).reshape(b * ncell, c)
        tiles = torch.multinomial(flat, 1, generator=generator).reshape(b, ncell)
    elif mode == "repaired":
        tiles = logits.argmax(dim=1)
    else:
        raise ValueError(f"unknown decode mode {mode!r}")

    tiles_np = tiles.detach().cpu().numpy()
    out: List[Sample] = []
    if mode == "repaired":
        probs_np = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for i, p in enumerate(prompts):
            grid = repair_grid(tiles_to_grid(tiles_np[i]), probs_np[i])
            out.append(_make_sample(grid, p))
    else:
        for i, p in enumerate(prompts):
            out.append(_make_sample(tiles_to_grid(tiles_np[i]), p))
    return out


def vae_drawer(model, device, mode: str, seed: int = 1337) -> "Drawer":
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    @torch.no_grad()
    def draw(prompts: Sequence[Prompt]) -> List[Sample]:
        cond = torch.tensor([p.caption_vec for p in prompts],
                            dtype=torch.float32, device=device)
        logits = model.sample(cond, generator=gen)
        return _logits_to_samples(logits, prompts, mode, generator=gen)

    return draw


def gan_drawer(model, device, mode: str, z_dim: int = 64,
               seed: int = 1337) -> "Drawer":
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    @torch.no_grad()
    def draw(prompts: Sequence[Prompt]) -> List[Sample]:
        cond = torch.tensor([p.caption_vec for p in prompts],
                            dtype=torch.float32, device=device)
        z = torch.randn(len(prompts), z_dim, device=device, generator=gen)
        logits = model(z, cond)
        return _logits_to_samples(logits, prompts, mode, generator=gen)

    return draw


def generate_for_suite(draw: "Drawer", prompts: Sequence[Prompt],
                       chunk: int = 512) -> List[Sample]:
    """Sample the fixed controllability suite: ``n_samples`` per prompt.

    Unlike ``collect``, this draws a *fixed* number per caption and keeps
    invalid samples too -- controllability is measured on what the model
    produced for each specific request, and silently dropping its failures
    would bias the result toward the captions it happened to handle.
    """
    expanded: List[Prompt] = []
    for p in prompts:
        expanded.extend([p] * p.n_samples)
    out: List[Sample] = []
    for i in range(0, len(expanded), chunk):
        out.extend(draw(expanded[i:i + chunk]))
    return out
