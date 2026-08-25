"""Caption features, text captions and conditioning vectors (spec 5.1).

Every feature is computed from the grid; nothing is hand-labelled.  Bin
boundaries are fitted on the **training** split only and then applied unchanged
to val, test and generated levels, so the conditioning scale is identical
everywhere.

Two parallel encodings of the same information
----------------------------------------------
``caption_text``  e.g. ``"hard, ~40 moves, dense walls, corridors, boxes scattered"``
    for the transformer.
``caption_vec``   10-dim float vector
    difficulty one-hot(3) + normalised solution length(1) + density one-hot(2)
    + connectivity one-hot(2) + clustering one-hot(2), for the VAE and GAN.

**Known limitation, stated in the paper:** the text prefix and the conditioning
vector are different information channels, so the family comparison is not
perfectly controlled on conditioning.

Connectivity: an empirically forced choice
------------------------------------------
The spec offered two options, "count of connected floor components" or "mean
corridor width via flood fill".  Measured over 20,000 training levels, **every
Boxoban level has exactly one connected floor component** (min = max = 1), so
the component count carries zero information and cannot bin anything.  We
therefore use the corridor-width alternative: the mean number of non-wall
neighbours over floor cells, which ranges 2.19-3.36 with median 2.889.  Low
degree means narrow winding corridors; high degree means an open chamber.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from ..solver.grid import STEP, Grid, cells

# The interior of a bordered 10x10 grid is 8x8 = 64 cells.
INTERIOR_CELLS: Tuple[int, ...] = tuple(
    i for i in range(100) if 0 < i // 10 < 9 and 0 < i % 10 < 9)
N_INTERIOR = len(INTERIOR_CELLS)  # 64

DIFFICULTY_WORDS = ("easy", "medium", "hard")
DENSITY_WORDS = ("open room", "dense walls")
CONNECTIVITY_WORDS = ("corridors", "single chamber")
CLUSTERING_WORDS = ("boxes clustered", "boxes scattered")

CAPTION_VEC_DIM = 10

# Solution lengths are quoted to the nearest 5 moves.
LENGTH_ROUNDING = 5


@dataclass(frozen=True)
class CaptionBins:
    """Bin boundaries fitted on the training split.

    ``difficulty_terciles`` are the 33.3rd and 66.7th percentiles of the
    training solution-length distribution; the other three are training
    medians.  ``length_min``/``length_max`` normalise the scalar length channel.
    """

    difficulty_terciles: Tuple[float, float]
    density_median: float
    degree_median: float
    clustering_median: float
    length_min: float
    length_max: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["difficulty_terciles"] = list(self.difficulty_terciles)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "CaptionBins":
        return cls(
            difficulty_terciles=tuple(d["difficulty_terciles"]),
            density_median=d["density_median"],
            degree_median=d["degree_median"],
            clustering_median=d["clustering_median"],
            length_min=d["length_min"],
            length_max=d["length_max"],
        )


# ---------------------------------------------------------------------------
# Raw feature extraction (grid only -- no solver required)
# ---------------------------------------------------------------------------
def wall_density(grid: Grid) -> float:
    """Interior wall count / 64."""
    n = sum(1 for i in INTERIOR_CELLS if (grid.walls >> i) & 1)
    return n / N_INTERIOR


def mean_floor_degree(grid: Grid) -> float:
    """Mean number of non-wall neighbours over floor cells (corridor width proxy).

    2.0 means every floor cell is a straight corridor; 4.0 means a fully open
    room.  See the module docstring for why this replaces component counting.
    """
    floor = [i for i in range(100) if not (grid.walls >> i) & 1]
    if not floor:
        return 0.0
    total = 0
    for f in floor:
        for d in range(4):
            n = STEP[f][d]
            if n >= 0 and not (grid.walls >> n) & 1:
                total += 1
    return total / len(floor)


def box_clustering(grid: Grid) -> float:
    """Mean pairwise Manhattan distance between boxes.

    Small means the boxes sit in a clump; large means they are spread out.
    """
    pos = [divmod(b, 10) for b in cells(grid.boxes)]
    if len(pos) < 2:
        return 0.0
    dists = [abs(a[0] - b[0]) + abs(a[1] - b[1])
             for i, a in enumerate(pos) for b in pos[i + 1:]]
    return sum(dists) / len(dists)


def raw_features(grid) -> Dict[str, float]:
    """Grid-only features.  Accepts a ``Grid`` or a grid string."""
    if isinstance(grid, str):
        grid = Grid.from_string(grid)
    return {
        "wall_density": wall_density(grid),
        "floor_degree": mean_floor_degree(grid),
        "box_clustering": box_clustering(grid),
    }


def features(grid, solution_length: float) -> Dict[str, float]:
    """Full feature dict: grid features plus the solution length in moves."""
    f = raw_features(grid)
    f["solution_length"] = float(solution_length)
    return f


# ---------------------------------------------------------------------------
# Fitting bins on the training split
# ---------------------------------------------------------------------------
def fit_bins(feature_rows: Sequence[Dict[str, float]]) -> CaptionBins:
    import numpy as np
    lengths = np.array([r["solution_length"] for r in feature_rows], dtype=float)
    dens = np.array([r["wall_density"] for r in feature_rows], dtype=float)
    deg = np.array([r["floor_degree"] for r in feature_rows], dtype=float)
    clus = np.array([r["box_clustering"] for r in feature_rows], dtype=float)
    return CaptionBins(
        difficulty_terciles=(float(np.percentile(lengths, 100 / 3)),
                             float(np.percentile(lengths, 200 / 3))),
        density_median=float(np.median(dens)),
        degree_median=float(np.median(deg)),
        clustering_median=float(np.median(clus)),
        length_min=float(lengths.min()),
        length_max=float(lengths.max()),
    )


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def difficulty_bin(length: float, bins: CaptionBins) -> int:
    lo, hi = bins.difficulty_terciles
    if length <= lo:
        return 0
    if length <= hi:
        return 1
    return 2


def density_bin(density: float, bins: CaptionBins) -> int:
    return 1 if density > bins.density_median else 0


def connectivity_bin(degree: float, bins: CaptionBins) -> int:
    """0 = corridors (low degree), 1 = single chamber (high degree)."""
    return 1 if degree > bins.degree_median else 0


def clustering_bin(clustering: float, bins: CaptionBins) -> int:
    """0 = clustered (small mean distance), 1 = scattered."""
    return 1 if clustering > bins.clustering_median else 0


def round_length(length: float) -> int:
    """Round to the nearest 5 moves, never below 5."""
    return max(LENGTH_ROUNDING, int(round(length / LENGTH_ROUNDING) * LENGTH_ROUNDING))


def bin_indices(feats: Dict[str, float], bins: CaptionBins) -> Dict[str, int]:
    return {
        "difficulty": difficulty_bin(feats["solution_length"], bins),
        "density": density_bin(feats["wall_density"], bins),
        "connectivity": connectivity_bin(feats["floor_degree"], bins),
        "clustering": clustering_bin(feats["box_clustering"], bins),
    }


# ---------------------------------------------------------------------------
# Caption construction
# ---------------------------------------------------------------------------
def caption_text_from_bins(difficulty: int, length: float, density: int,
                           connectivity: int, clustering: int) -> str:
    return (f"{DIFFICULTY_WORDS[difficulty]}, "
            f"~{round_length(length)} moves, "
            f"{DENSITY_WORDS[density]}, "
            f"{CONNECTIVITY_WORDS[connectivity]}, "
            f"{CLUSTERING_WORDS[clustering]}")


def caption_text(feats: Dict[str, float], bins: CaptionBins) -> str:
    b = bin_indices(feats, bins)
    return caption_text_from_bins(b["difficulty"], feats["solution_length"],
                                  b["density"], b["connectivity"], b["clustering"])


def caption_vec_from_bins(difficulty: int, length: float, density: int,
                          connectivity: int, clustering: int,
                          bins: CaptionBins) -> List[float]:
    """10-dim conditioning vector; see the module docstring for the layout."""
    vec = [0.0] * CAPTION_VEC_DIM
    vec[difficulty] = 1.0                       # 0..2  difficulty one-hot
    span = max(1e-8, bins.length_max - bins.length_min)
    norm = (length - bins.length_min) / span
    vec[3] = float(min(1.0, max(0.0, norm)))    # 3     normalised length
    vec[4 + density] = 1.0                      # 4..5  density one-hot
    vec[6 + connectivity] = 1.0                 # 6..7  connectivity one-hot
    vec[8 + clustering] = 1.0                   # 8..9  clustering one-hot
    return vec


def caption_vec(feats: Dict[str, float], bins: CaptionBins) -> List[float]:
    b = bin_indices(feats, bins)
    return caption_vec_from_bins(b["difficulty"], feats["solution_length"],
                                 b["density"], b["connectivity"],
                                 b["clustering"], bins)


def max_caption_length() -> int:
    """Longest caption string producible, used to size the sequence length."""
    longest = 0
    for d in range(3):
        for de in range(2):
            for co in range(2):
                for cl in range(2):
                    # 999 is a safe upper bound on any quoted length.
                    s = caption_text_from_bins(d, 999, de, co, cl)
                    longest = max(longest, len(s))
    return longest
