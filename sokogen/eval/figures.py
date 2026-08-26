"""Figures generated from ``results/evaluation.json``.

All seven figures are written here; the paper carries the three that are
load-bearing and the rest are repository-only.

**In the paper** (spec 12 caps the paper at five figures; measured against the
6-8 page limit, five came to nine pages and three fit in eight, so the two most
redundant with prose were moved to the repository):

 1. Structural validity vs solvability -- the money figure.
 2. Tile-count histograms per family against the real distribution, which
    carries the counting-constraint claim more convincingly than any percentage.
 3. Temperature Pareto: solvability vs diversity.

**Repository only**: pipeline architecture (a schematic that the Method section
already states in full), the qualitative sample grid, node-cap sensitivity, GAN
and transformer training curves, and VAE latent interpolation.

Each figure is saved as PDF (for the paper) and PNG (for the README).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .tables import DISPLAY_NAMES

# wall, floor, box, goal, player
TILE_COLOURS = ["#3b3b4f", "#f2f2f0", "#c8783c", "#6aa84f", "#3d6fb4"]
TILE_CMAP = ListedColormap(TILE_COLOURS)
TILE_ORDER = {"#": 0, " ": 1, "$": 2, ".": 3, "@": 4}

FAMILY_COLOURS = {
    "transformer": "#3d6fb4",
    "vae": "#c8783c",
    "gan": "#a04b6b",
    "baseline": "#7a7a8c",
    "real": "#4f8f4f",
}


def _family(name: str) -> str:
    if name.startswith("transformer") or name.startswith("distilgpt2"):
        return "transformer"
    if name.startswith("vae"):
        return "vae"
    if name.startswith("gan"):
        return "gan"
    if name == "real_boxoban":
        return "real"
    return "baseline"


def _save(fig, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


def grid_to_array(grid: str) -> np.ndarray:
    body = grid.replace("\n", "")
    body = (body + " " * 100)[:100]
    return np.array([TILE_ORDER.get(c, 1) for c in body]).reshape(10, 10)


def draw_grid(ax, grid: str, title: Optional[str] = None,
              title_size: int = 7) -> None:
    ax.imshow(grid_to_array(grid), cmap=TILE_CMAP, vmin=0, vmax=4,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color("#999")
    if title:
        ax.set_title(title, fontsize=title_size, pad=2)


# ---------------------------------------------------------------------------
# Figure 1: pipeline
# ---------------------------------------------------------------------------
def figure_pipeline(out_dir: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def box(x, y, w, h, label, sub, colour):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6", linewidth=1.1,
            edgecolor=colour, facecolor=colour + "22"))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=8.5, weight="bold", color="#222")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=6.8, color="#444")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color="#666"))

    box(1, 20, 19, 11, "Boxoban corpus", "900k levels, 10x10", "#7a7a8c")
    box(1, 4, 19, 11, "A* solutions", "replay-validated", "#7a7a8c")
    box(24, 12, 18, 11, "Captions", "5 features -> text + vec", "#4f8f4f")
    box(46, 20, 20, 11, "Transformer", "10.7M, autoregressive", "#3d6fb4")
    box(46, 4, 20, 11, "VAE / GAN", "one-shot decoders", "#c8783c")
    box(70, 20, 13, 11, "Constrained\ndecoding", "grid guard", "#3d6fb4")
    box(70, 4, 13, 11, "Repair", "project to valid", "#c8783c")
    box(87, 12, 12, 11, "Exhaustive\nA* solver", "3-way verdict", "#a04b6b")

    arrow(20, 25, 24, 20)
    arrow(20, 10, 24, 15)
    arrow(42, 19, 46, 25)
    arrow(42, 16, 46, 10)
    arrow(66, 25, 70, 25)
    arrow(66, 10, 70, 10)
    arrow(83, 25, 87, 21)
    arrow(83, 10, 87, 15)

    ax.text(93, 8.5, "solved /\nproven unsolvable /\ntimeout", ha="center",
            va="center", fontsize=6.5, style="italic", color="#555")
    return _save(fig, out_dir, "fig1_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: validity vs solvability (the money figure)
# ---------------------------------------------------------------------------
def figure_validity_vs_solvability(ev: Dict, out_dir: str) -> List[str]:
    names = [n for n in ev["row_order"] if n in DISPLAY_NAMES]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    y = np.arange(len(names))
    valid = [ev["conditions"][n]["structural_validity"]["pct"] for n in names]
    sgv = [ev["conditions"][n]["solvable_given_valid"]["pct"] for n in names]
    sgv = [0.0 if (v is None or v != v) else v for v in sgv]
    colours = [FAMILY_COLOURS[_family(n)] for n in names]

    h = 0.38
    ax1.barh(y - h / 2, valid, height=h, color=colours, alpha=0.95,
             label="Structural validity")
    ax1.barh(y + h / 2, sgv, height=h, color=colours, alpha=0.45, hatch="//",
             label="Solvable | valid")
    ax1.set_yticks(y)
    ax1.set_yticklabels([DISPLAY_NAMES[n] for n in names], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("percent", fontsize=9)
    ax1.set_xlim(0, 104)
    ax1.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("Structural validity and solvability are different axes",
                  fontsize=10)
    # Every row spans the full width, so an inset legend would cover a bar.
    ax1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.10),
               ncol=2, frameon=False)

    # Most conditions sit at exactly 100% validity, so labels collide badly if
    # placed naively.  Anchor labels to the left for points on the right edge
    # and stagger vertically whenever two points are close.
    pts = []
    for n in names:
        c = ev["conditions"][n]
        v = c["structural_validity"]["pct"]
        s = c["solvable_given_valid"]["pct"]
        if s is None or s != s:
            continue
        pts.append((v, s, n))
    pts.sort(key=lambda p: (-p[1], p[0]))

    placed: List[tuple] = []
    for v, s, n in pts:
        ax2.scatter(v, s, s=70, color=FAMILY_COLOURS[_family(n)],
                    edgecolor="white", linewidth=0.8, zorder=3)
        right_edge = v > 90
        dy = 5.0
        # Nudge downward until this label clears the ones already placed.
        while any(abs(s + dy - py) < 6.5 and abs(v - px) < 12
                  for px, py in placed):
            dy -= 6.5
        ax2.annotate(DISPLAY_NAMES[n], (v, s), fontsize=6.2,
                     xytext=(-6 if right_edge else 6, dy),
                     textcoords="offset points", color="#333",
                     ha="right" if right_edge else "left", va="center")
        placed.append((v, s + dy))

    ax2.set_xlabel("structural validity %", fontsize=9)
    ax2.set_ylabel("solvable | valid %", fontsize=9)
    ax2.set_xlim(-8, 118)
    ax2.set_ylim(-12, 115)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("Counting vs spatial structure", fontsize=10)

    fig.suptitle("Can the model count, and does it understand the puzzle?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_validity_vs_solvability")


# ---------------------------------------------------------------------------
# Figure 3: tile-count histograms
# ---------------------------------------------------------------------------
def figure_tile_counts(ev: Dict, out_dir: str,
                       families: Optional[Sequence[str]] = None) -> List[str]:
    if families is None:
        # vae_argmax is included even though it yields no valid level: its
        # near-empty histogram is the clearest picture of marginal collapse.
        families = [n for n in ("transformer_unconstrained", "vae_argmax",
                                "vae_sample", "gan_raw", "real_boxoban")
                    if n in ev["conditions"]]
    tiles = [("box", 4), ("goal", 4), ("player", 1)]
    fig, axes = plt.subplots(len(tiles), len(families),
                             figsize=(2.5 * len(families), 5.4), squeeze=False)

    for r, (tile, required) in enumerate(tiles):
        for c, name in enumerate(families):
            ax = axes[r][c]
            stats = ev["conditions"][name].get("tile_counts_raw_draws", {})
            hist = stats.get(tile, {}).get("histogram", {})
            if hist:
                ks = sorted(int(k) for k in hist)
                vs = [hist[str(k)] for k in ks]
                total = sum(vs)
                cols = ["#4f8f4f" if k == required else FAMILY_COLOURS[_family(name)]
                        for k in ks]
                ax.bar(ks, [100 * v / total for v in vs], color=cols, width=0.72)
            ax.axvline(required, color="#4f8f4f", linestyle="--", linewidth=1.0)
            ax.set_xlim(-0.8, 10.8)
            ax.set_ylim(0, 105)
            ax.tick_params(labelsize=6.5)
            if r == 0:
                ax.set_title(DISPLAY_NAMES.get(name, name), fontsize=7.6)
            if c == 0:
                ax.set_ylabel(f"{tile}s\n(% of draws)", fontsize=7.6)
            if r == len(tiles) - 1:
                ax.set_xlabel("count per level", fontsize=7.4)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_axisbelow(True)

    fig.suptitle("Tile counts per generated level (dashed = required)",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_tile_counts")


# ---------------------------------------------------------------------------
# Figure 4: qualitative samples
# ---------------------------------------------------------------------------
def figure_qualitative(gen_dir: str, out_dir: str, ev: Dict,
                       n_cols: int = 5) -> List[str]:
    families = [n for n in ("transformer_constrained", "transformer_unconstrained",
                            "vae_sample", "vae_repaired", "gan_repaired",
                            "open_room", "real_boxoban")
                if n in ev["conditions"]]
    def pick(path: str) -> List[Dict]:
        """Choose samples spanning distinct captions, not just the first few."""
        if not os.path.exists(path):
            return []
        by_caption: Dict[str, Dict] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = json.loads(line)
                by_caption.setdefault(s["caption_text"], s)
        ordered = sorted(by_caption.values(),
                         key=lambda s: (s["requested"]["difficulty"],
                                        s["requested"]["density"],
                                        s["caption_text"]))
        if len(ordered) <= n_cols:
            return ordered
        # Even spread across the sorted captions.
        idx = [round(i * (len(ordered) - 1) / (n_cols - 1)) for i in range(n_cols)]
        return [ordered[i] for i in idx]

    rows = []
    for name in families:
        samples = pick(os.path.join(gen_dir, f"{name}.jsonl"))
        if samples:
            rows.append((name, samples))

    if not rows:
        return []
    cell = 1.45
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(cell * n_cols + 1.4,
                                      cell * len(rows) + 0.7),
                             squeeze=False)
    for r, (name, samples) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(samples):
                draw_grid(ax, samples[c]["grid"])
                if r == 0:
                    parts = samples[c]["caption_text"].split(", ")
                    ax.set_title(f"{parts[0]}, {parts[1]}\n{parts[2]}",
                                 fontsize=5.8, pad=3)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(DISPLAY_NAMES.get(name, name), fontsize=6.6,
                              rotation=0, ha="right", va="center", labelpad=46)
    fig.suptitle("Generated levels, one row per family "
                 "(columns share a conditioning caption)", fontsize=9.5)
    fig.subplots_adjust(hspace=0.06, wspace=0.06, top=0.93, left=0.22,
                        right=0.99, bottom=0.01)
    return _save(fig, out_dir, "fig4_qualitative")


# ---------------------------------------------------------------------------
# Figure 5: temperature Pareto
# ---------------------------------------------------------------------------
def figure_temperature_pareto(ev: Dict, out_dir: str) -> List[str]:
    rows = []
    for name, c in ev["conditions"].items():
        if not name.startswith("transformer_constrained_t"):
            continue
        temp = c.get("temperature")
        if temp is None:
            try:
                temp = float(name.rsplit("t", 1)[1])
            except (ValueError, IndexError):
                continue
        s = c["solvable_given_valid"]["pct"]
        d = c.get("diversity", {}).get("mean")
        if s is None or d is None or s != s:
            continue
        rows.append((float(temp), s, d,
                     c["solvable_given_valid"]["ci_lo_pct"],
                     c["solvable_given_valid"]["ci_hi_pct"]))
    if not rows:
        return []
    rows.sort()
    temps = [r[0] for r in rows]
    solv = [r[1] for r in rows]
    div = [r[2] for r in rows]
    lo = [r[1] - r[3] for r in rows]
    hi = [r[4] - r[1] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax1.errorbar(div, solv, yerr=[lo, hi], fmt="o-", color="#3d6fb4",
                 capsize=3, linewidth=1.4, markersize=6)
    for t, s, d in zip(temps, solv, div):
        ax1.annotate(f"T={t}", (d, s), fontsize=7, xytext=(5, 4),
                     textcoords="offset points")
    ax1.set_xlabel("diversity (mean pairwise Hamming distance)", fontsize=9)
    ax1.set_ylabel("solvable | valid %", fontsize=9)
    ax1.set_title("Solvability vs diversity trade-off", fontsize=10)
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)

    ax2.plot(temps, solv, "o-", color="#3d6fb4", label="solvable | valid %")
    ax2b = ax2.twinx()
    ax2b.plot(temps, div, "s--", color="#c8783c", label="diversity")
    ax2.set_xlabel("sampling temperature", fontsize=9)
    ax2.set_ylabel("solvable | valid %", fontsize=9, color="#3d6fb4")
    ax2b.set_ylabel("diversity", fontsize=9, color="#c8783c")
    ax2.set_title("Both metrics vs temperature", fontsize=10)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, out_dir, "fig5_temperature_pareto")


# ---------------------------------------------------------------------------
# Repo-only figures
# ---------------------------------------------------------------------------
def figure_node_cap(solver_json: str, out_dir: str) -> List[str]:
    if not os.path.exists(solver_json):
        return []
    with open(solver_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    curve = data.get("validation4_node_cap_curve", {}).get("curve", [])
    if not curve:
        return []
    caps = [c["node_cap"] for c in curve]
    rates = [100 * c["solve_rate"] for c in curve]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogx(caps, rates, "o-", color="#a04b6b", linewidth=1.6, markersize=7)
    ax.set_xlabel("node cap", fontsize=9)
    ax.set_ylabel("solve rate on real test levels (%)", fontsize=9)
    ax.set_title("Solve rate vs node cap\n(a flat tail means the headline "
                 "number has converged)", fontsize=9.5)
    ax.grid(alpha=0.3, which="both", linewidth=0.6)
    ax.set_ylim(min(rates) - 3, 101.5)
    ax.set_axisbelow(True)
    return _save(fig, out_dir, "figA_node_cap_sensitivity")


def figure_training_curves(results_dir: str, out_dir: str) -> List[str]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    made = False

    tpath = os.path.join(results_dir, "train_transformer.json")
    if os.path.exists(tpath):
        with open(tpath, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        hist = d.get("history", [])
        tr = [(h["step"], h["train_loss"]) for h in hist if "train_loss" in h]
        va = [(h["step"], h["val_loss"]) for h in hist if "val_loss" in h]
        if tr:
            axes[0].plot(*zip(*tr), color="#3d6fb4", linewidth=1.3,
                         label="train")
        if va:
            axes[0].plot(*zip(*va), "o--", color="#a04b6b", label="val")
        axes[0].set_title("Transformer", fontsize=10)
        axes[0].set_xlabel("step", fontsize=8.5)
        axes[0].set_ylabel("cross-entropy (grid tokens)", fontsize=8.5)
        axes[0].legend(fontsize=7.5)
        made = True

    vpath = os.path.join(results_dir, "train_vae.json")
    if os.path.exists(vpath):
        with open(vpath, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        hist = d.get("history", [])
        ep = [h["epoch"] for h in hist]
        axes[1].plot(ep, [h["train_recon"] for h in hist], color="#c8783c",
                     label="recon (train)")
        axes[1].plot(ep, [h["val_recon"] for h in hist], "--", color="#a04b6b",
                     label="recon (val)")
        ax1b = axes[1].twinx()
        ax1b.plot(ep, [h["train_kl"] for h in hist], ":", color="#4f8f4f",
                  label="KL")
        ax1b.set_ylabel("KL (nats)", fontsize=8.5, color="#4f8f4f")
        axes[1].set_title("VAE (free bits: no posterior collapse)", fontsize=10)
        axes[1].set_xlabel("epoch", fontsize=8.5)
        axes[1].legend(fontsize=7.5, loc="upper right")
        made = True

    gpath = os.path.join(results_dir, "train_gan.json")
    if os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        hist = d.get("history", [])
        ep = [h["epoch"] for h in hist]
        axes[2].plot(ep, [h["loss_d"] for h in hist], color="#a04b6b",
                     label="discriminator")
        axes[2].plot(ep, [h["loss_g"] for h in hist], color="#3d6fb4",
                     label="generator")
        axes[2].set_title("GAN", fontsize=10)
        axes[2].set_xlabel("epoch", fontsize=8.5)
        axes[2].legend(fontsize=7.5)
        made = True

    if not made:
        plt.close(fig)
        return []
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out_dir, "figB_training_curves")


def figure_vae_interpolation(checkpoint: str, out_dir: str, suite_path: str,
                             n_steps: int = 7, n_rows: int = 4,
                             seed: int = 1337) -> List[str]:
    """Repo-only: walk the VAE latent between two samples (spec 12).

    Each row interpolates ``z`` linearly between two draws from the prior while
    holding the caption vector fixed, decoding by argmax at every step.  It is a
    direct picture of the marginal-collapse failure: the whole path stays a
    near-empty room, because the argmax of a product of per-cell marginals
    almost never selects a rare tile anywhere along it.
    """
    if not os.path.exists(checkpoint):
        return []
    try:
        import torch

        from ..models.common import load_checkpoint, tiles_to_grid
        from ..models.vae import ConditionalVAE
    except ImportError:  # pragma: no cover - torch is optional for figures
        return []

    sd, meta = load_checkpoint(checkpoint, map_location="cpu")
    hp = meta.get("config", {})
    model = ConditionalVAE(latent_dim=hp.get("latent_dim", 32),
                           hidden=hp.get("hidden", 512))
    model.load_state_dict(sd)
    model.eval()

    with open(suite_path, "r", encoding="utf-8") as fh:
        suite = json.load(fh)["in_distribution"]

    g = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(n_rows, n_steps,
                             figsize=(1.25 * n_steps + 0.5, 1.25 * n_rows + 0.6),
                             squeeze=False)
    with torch.no_grad():
        for r in range(n_rows):
            cond = torch.tensor(
                [suite[(r * 7) % len(suite)]["caption_vec"]], dtype=torch.float32)
            z0 = torch.randn(1, model.latent_dim, generator=g)
            z1 = torch.randn(1, model.latent_dim, generator=g)
            for c in range(n_steps):
                t = c / (n_steps - 1)
                z = (1 - t) * z0 + t * z1
                tiles = model.decode(z, cond).argmax(dim=1)[0].numpy()
                draw_grid(axes[r][c], tiles_to_grid(tiles))
                if r == 0:
                    axes[r][c].set_title(f"t={t:.2f}", fontsize=6.5, pad=2)

    fig.suptitle("VAE latent interpolation (argmax decoding)\n"
                 "the whole path is a near-empty room: rare tiles never win "
                 "the per-cell argmax", fontsize=9)
    fig.subplots_adjust(hspace=0.06, wspace=0.06, top=0.84)
    return _save(fig, out_dir, "figC_vae_interpolation")


def write_all(evaluation_json: str, gen_dir: str, out_dir: str,
              results_dir: str) -> Dict[str, List[str]]:
    with open(evaluation_json, "r", encoding="utf-8") as fh:
        ev = json.load(fh)
    made: Dict[str, List[str]] = {}
    made["fig1_pipeline"] = figure_pipeline(out_dir)
    made["fig2_validity_vs_solvability"] = figure_validity_vs_solvability(ev, out_dir)
    made["fig3_tile_counts"] = figure_tile_counts(ev, out_dir)
    made["fig4_qualitative"] = figure_qualitative(gen_dir, out_dir, ev)
    made["fig5_temperature_pareto"] = figure_temperature_pareto(ev, out_dir)
    made["figA_node_cap"] = figure_node_cap(
        os.path.join(results_dir, "solver_validation.json"), out_dir)
    made["figB_training_curves"] = figure_training_curves(results_dir, out_dir)
    made["figC_vae_interpolation"] = figure_vae_interpolation(
        os.path.join("checkpoints", "vae.pt"), out_dir,
        os.path.join("configs", "prompt_suite.json"))
    return {k: v for k, v in made.items() if v}
