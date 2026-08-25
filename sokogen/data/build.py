"""Build the train/val/test JSONL datasets (spec 5.3).

Splits
------
train : 100,000 levels from ``unfiltered/train``
val   :   5,000 levels from ``unfiltered/valid``  (the **official** split -- no
          validation set is carved out of train)
test  :   1,000 levels from ``unfiltered/test``   (frozen: solver validation and
          the real-level ceiling row only)

Each row is ``{level, caption_text, caption_vec, solution_length, source_file,
source_index}``.

Solution lengths are **replay-validated** (see ``sokogen.data.solutions``):
Phase 1 found that ~0.7% of upstream rows are well-formed digit strings that do
not actually solve their level.  Levels whose solution fails to replay are
dropped, because a caption built on a wrong length is worse than no example.

Caption bins are fitted on the training split only and reused verbatim for val,
test and every generated level.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .boxoban import Level, load_level_file, split_files
from .captions import (CaptionBins, caption_text, caption_vec, features,
                       fit_bins)
from .solutions import load_solutions_frame, verified_solution


def collect_levels(levels_root: str, split: str, n_levels: int) -> List[Level]:
    """Deterministic prefix of a split, in (file, index) order."""
    out: List[Level] = []
    for path in split_files(levels_root, split):
        out.extend(load_level_file(path, split))
        if len(out) >= n_levels:
            break
    return out[:n_levels]


def levels_with_lengths(levels: List[Level], solutions_root: str, split: str
                        ) -> Tuple[List[Tuple[Level, int]], Dict[str, int]]:
    """Attach replay-validated move-optimal lengths, dropping what fails.

    Returns the surviving ``(level, moves)`` pairs and a stats dict recording
    exactly how many were dropped and why.
    """
    df = load_solutions_frame(solutions_root, split)
    kept: List[Tuple[Level, int]] = []
    stats = {"total": len(levels), "dropped_no_valid_solution": 0}
    for lev in levels:
        sol = verified_solution(df, lev.grid, lev.source_file, lev.source_index)
        if sol is None:
            stats["dropped_no_valid_solution"] += 1
            continue
        kept.append((lev, sol.moves))
    stats["kept"] = len(kept)
    return kept, stats


def build_rows(pairs: List[Tuple[Level, int]], bins: CaptionBins) -> List[Dict]:
    rows = []
    for lev, moves in pairs:
        feats = features(lev.grid, moves)
        rows.append({
            "level": lev.grid,
            "caption_text": caption_text(feats, bins),
            "caption_vec": caption_vec(feats, bins),
            "solution_length": moves,
            "source_file": lev.source_file,
            "source_index": lev.source_index,
            "split": lev.split,
        })
    return rows


def fit_bins_for(pairs: List[Tuple[Level, int]]) -> CaptionBins:
    return fit_bins([features(lev.grid, moves) for lev, moves in pairs])


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows
