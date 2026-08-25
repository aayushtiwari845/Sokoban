"""The fixed prompt suite for controllability (spec 10.3).

Controllability is undefined without a fixed grid of prompts, so this is
generated once, seeded, and stored in ``configs/prompt_suite.json``.

Deviation from the spec's "5 difficulty bins x 3 wall-density settings x 40"
-----------------------------------------------------------------------------
Wall density is a **two-bin** feature by construction (spec 5.1 bins it at the
training median into ``open room`` / ``dense walls``), so three wall-density
settings do not exist.  We keep the specified total of **600 prompts** and
spend the freed factor on a balanced full factorial instead:

    5 lengths x 2 density x 2 connectivity x 2 clustering = 40 captions
    40 captions x 15 samples = 600 prompts

This is strictly better for the metric that matters: every conditioned
attribute varies independently of the others, so per-attribute control accuracy
is not confounded.  Reporting controllability per attribute (never as one
number) requires exactly that.

Out-of-distribution suite
-------------------------
Requesting lengths beyond what training contains is a clean negative result and
populates the Failure Analysis section.  Measured on the 99,266-level training
split: p90 = 47, p99 = 67, p99.9 = 87, max = 130.  The OOD targets are 90
(99.93rd percentile, 68 training examples), 110 (9 examples) and 150 (beyond
the training maximum entirely).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

from ..data.captions import (CaptionBins, caption_text_from_bins,
                             caption_vec_from_bins, difficulty_bin)

# Spanning roughly p2 to p96 of the training distribution.
IN_DIST_LENGTHS = (15, 25, 35, 45, 55)
# Beyond the practical training range; 150 exceeds the training maximum of 130.
OOD_LENGTHS = (90, 110, 150)

SAMPLES_PER_CAPTION = 15
OOD_SAMPLES_PER_CAPTION = 5


@dataclass(frozen=True)
class Prompt:
    """One conditioning request, with everything needed to score it."""

    caption_text: str
    caption_vec: List[float]
    requested_length: int
    difficulty: int
    density: int
    connectivity: int
    clustering: int
    n_samples: int
    ood: bool

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Prompt":
        return cls(**d)

    def requested_bins(self) -> Dict:
        return {"difficulty": self.difficulty, "density": self.density,
                "connectivity": self.connectivity, "clustering": self.clustering,
                "solution_length": self.requested_length}


def build_suite(bins: CaptionBins, lengths=IN_DIST_LENGTHS,
                samples: int = SAMPLES_PER_CAPTION, ood: bool = False
                ) -> List[Prompt]:
    out: List[Prompt] = []
    for length in lengths:
        for density in (0, 1):
            for connectivity in (0, 1):
                for clustering in (0, 1):
                    diff = difficulty_bin(length, bins)
                    out.append(Prompt(
                        caption_text=caption_text_from_bins(
                            diff, length, density, connectivity, clustering),
                        caption_vec=caption_vec_from_bins(
                            diff, length, density, connectivity, clustering, bins),
                        requested_length=length,
                        difficulty=diff,
                        density=density,
                        connectivity=connectivity,
                        clustering=clustering,
                        n_samples=samples,
                        ood=ood,
                    ))
    return out


def build_full_suite(bins: CaptionBins) -> Dict[str, List[Prompt]]:
    return {
        "in_distribution": build_suite(bins, IN_DIST_LENGTHS,
                                       SAMPLES_PER_CAPTION, ood=False),
        "out_of_distribution": build_suite(bins, OOD_LENGTHS,
                                           OOD_SAMPLES_PER_CAPTION, ood=True),
    }


def save_suite(path: str, suite: Dict[str, List[Prompt]], bins: CaptionBins,
               seed: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "seed": seed,
        "caption_bins": bins.to_dict(),
        "design": {
            "in_distribution": "5 lengths x 2 density x 2 connectivity x "
                               "2 clustering x 15 samples = 600",
            "out_of_distribution": "3 lengths x 8 settings x 5 samples = 120",
            "note": "Wall density is a 2-bin feature, so the spec's '3 "
                    "wall-density settings' does not exist; the 600-prompt "
                    "total is preserved with a balanced factorial instead.",
        },
        "in_distribution": [p.to_dict() for p in suite["in_distribution"]],
        "out_of_distribution": [p.to_dict() for p in suite["out_of_distribution"]],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_suite(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        "seed": payload["seed"],
        "caption_bins": CaptionBins.from_dict(payload["caption_bins"]),
        "in_distribution": [Prompt.from_dict(d)
                            for d in payload["in_distribution"]],
        "out_of_distribution": [Prompt.from_dict(d)
                                for d in payload["out_of_distribution"]],
    }
