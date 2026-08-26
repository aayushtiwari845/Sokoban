"""Shared training utilities: seeding, bf16 setup, checkpointing, data loading.

Hardware target is a single RTX 3050 6GB (Ampere, sm_86), which supports bf16
natively -- no loss scaling and no ``GradScaler`` needed.  ``setup_device``
verifies that at startup rather than assuming it.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.vocab import MAX_LEN, PAD_ID, SEP_ID, encode_example, pad_to


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    dtype: torch.dtype
    bf16: bool
    name: str


def setup_device(prefer_bf16: bool = True, verbose: bool = True) -> DeviceConfig:
    """Pick the device and autocast dtype, verifying bf16 support explicitly."""
    # device_count() is checked too: with CUDA_VISIBLE_DEVICES="" the runtime
    # reports itself available while exposing no device, and querying device 0
    # then raises rather than falling back.
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        if verbose:
            print("  [device] CUDA unavailable -- running on CPU in float32")
        return DeviceConfig(torch.device("cpu"), torch.float32, False, "cpu")

    name = torch.cuda.get_device_name(0)
    bf16 = bool(torch.cuda.is_bf16_supported()) and prefer_bf16
    dtype = torch.bfloat16 if bf16 else torch.float16
    if verbose:
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [device] {name} sm_{cap[0]}{cap[1]} {total:.1f}GB  "
              f"bf16_supported={torch.cuda.is_bf16_supported()}  using {dtype}")
    return DeviceConfig(torch.device("cuda"), dtype, bf16, name)


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


def save_checkpoint(path: str, model: torch.nn.Module, meta: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, map_location="cpu") -> Tuple[Dict, Dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    return blob["state_dict"], blob.get("meta", {})


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def encode_lm_dataset(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Encode rows into padded token ids and per-row grid start offsets.

    Returns ``(ids[N, MAX_LEN] uint8, grid_start[N] int16)``.  ``grid_start`` is
    the index of the first grid character, i.e. one past ``<sep>``; the training
    loss is masked so that only grid characters and ``<eos>`` are predicted --
    the caption prefix is context, not a target (spec 6.1).

    uint8 is safe because the vocabulary has 48 symbols.
    """
    ids = np.full((len(rows), MAX_LEN), PAD_ID, dtype=np.uint8)
    grid_start = np.zeros(len(rows), dtype=np.int16)
    for i, r in enumerate(rows):
        seq = encode_example(r["caption_text"], r["level"])
        ids[i, : len(seq)] = np.array(seq, dtype=np.uint8)
        grid_start[i] = seq.index(SEP_ID) + 1
    return ids, grid_start


def make_lm_labels(ids: torch.Tensor, grid_start: torch.Tensor) -> torch.Tensor:
    """Build labels with the caption prefix and padding masked to -100.

    ``ids`` is ``[B, T]``; the returned labels align with ``ids`` position for
    position, and the model applies the usual one-position shift internally.
    """
    labels = ids.clone().long()
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    before_grid = positions < grid_start.unsqueeze(1)
    labels[before_grid] = -100
    labels[ids == PAD_ID] = -100
    return labels


def encode_cond_dataset(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Encode rows for the one-shot families.

    Returns ``(tiles[N, 100] uint8, cond[N, 10] float32)`` where ``tiles`` holds
    channel indices in the order (wall, floor, box, goal, player).  Phase 1
    verified that no box ever starts on a goal, so a single index per cell is a
    faithful representation.
    """
    from ..data.boxoban import BOX, FLOOR, GOAL, PLAYER, WALL

    order = {WALL: 0, FLOOR: 1, BOX: 2, GOAL: 3, PLAYER: 4}
    tiles = np.zeros((len(rows), 100), dtype=np.uint8)
    cond = np.zeros((len(rows), 10), dtype=np.float32)
    for i, r in enumerate(rows):
        body = r["level"].replace("\n", "")
        tiles[i] = np.frombuffer(
            bytes(order[c] for c in body), dtype=np.uint8)
        cond[i] = np.array(r["caption_vec"], dtype=np.float32)
    return tiles, cond


def tiles_to_grid(tiles) -> str:
    """Channel indices [100] -> canonical 110-char grid string."""
    chars = ("#", " ", "$", ".", "@")
    body = "".join(chars[int(t)] for t in tiles)
    return "".join(body[r * 10:(r + 1) * 10] + "\n" for r in range(10))


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay to 10% of base."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))
