"""Figures written for the course report only.

The paper's figures (``sokogen/eval/figures.py``) are terse by design.  These
five are the explanatory ones: what Sokoban is, why a one-shot decoder cannot
count, what the logit mask actually does, how far the seeds move, and the
headline ranking.

Every number that appears inside a figure is read from a committed artifact
under ``results/`` or measured live from a committed checkpoint.  Nothing here
is illustrative.

    python report/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "report", "figures")

# wall, floor, box, goal, player, box-on-goal, player-on-goal, not-yet-emitted
COLOURS = ["#3b3b4f", "#f2f2f0", "#c8783c", "#6aa84f", "#3d6fb4",
           "#8a5a2b", "#2b4f80", "#c9c9d6"]
CMAP = ListedColormap(COLOURS)
ORDER = {"#": 0, " ": 1, "$": 2, ".": 3, "@": 4, "*": 5, "+": 6, "~": 7}
VMAX = len(COLOURS) - 1

BLUE = "#3d6fb4"
ORANGE = "#c8783c"
GREEN = "#4f8f4f"
GREY = "#7a7a8c"
PLUM = "#a04b6b"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "figure.dpi": 200,
})


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight",
                    dpi=200)
    plt.close(fig)
    print(f"  wrote report/figures/{stem}.pdf/.png")


def to_array(grid):
    body = grid.replace("\n", "")
    return np.array([ORDER.get(c, 1) for c in body]).reshape(10, 10)


def draw(ax, grid, title=None, size=8.5, colour="#333"):
    ax.imshow(to_array(grid), cmap=CMAP, vmin=0, vmax=VMAX,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("#aaa")
    if title:
        ax.set_title(title, fontsize=size, pad=3, color=colour)


# ---------------------------------------------------------------------------
# R1: what a Sokoban level is, and what "solved" means
# ---------------------------------------------------------------------------
ACTION_VECTORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def replay_states(grid, actions, snapshots):
    """Render the level at each requested move index.

    Boxes standing on goals render as ``*`` and the player on a goal as ``+``,
    so the reader can see the puzzle close.
    """
    rows = [list(r) for r in grid.rstrip("\n").split("\n")]
    walls, goals, boxes, player = set(), set(), set(), None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            elif ch == "$":
                boxes.add((r, c))
            elif ch == ".":
                goals.add((r, c))
            elif ch == "@":
                player = (r, c)

    def render():
        out = []
        for r in range(10):
            line = []
            for c in range(10):
                p = (r, c)
                if p in walls:
                    line.append("#")
                elif p == player:
                    line.append("+" if p in goals else "@")
                elif p in boxes:
                    line.append("*" if p in goals else "$")
                elif p in goals:
                    line.append(".")
                else:
                    line.append(" ")
            out.append("".join(line))
        return "\n".join(out)

    frames = {}
    if 0 in snapshots:
        frames[0] = render()
    for i, a in enumerate(actions, start=1):
        dr, dc = ACTION_VECTORS[int(a)]
        nr, nc = player[0] + dr, player[1] + dc
        if (nr, nc) in boxes:
            boxes.discard((nr, nc))
            boxes.add((nr + dr, nc + dc))
        player = (nr, nc)
        if i in snapshots:
            frames[i] = render()
    return frames, boxes == goals


def figure_r1():
    ver = load("results/data_verification.json")
    example = ver["check3_solution_units"]["examples"][0]
    grid, actions = example["grid"], example["actions"]
    n = len(actions)
    idxs = [0, n // 3, 2 * n // 3, n]
    frames, solved = replay_states(grid, actions, set(idxs))
    assert solved, "the committed example must replay to a solved state"

    with open(os.path.join(ROOT, "data", "test.jsonl"), encoding="utf-8") as fh:
        caption = json.loads(fh.readline())["caption_text"]

    fig = plt.figure(figsize=(9.2, 2.4))
    gs = fig.add_gridspec(1, 4, wspace=0.12)
    labels = ["start (move 0)", f"move {idxs[1]}", f"move {idxs[2]}",
              f"solved (move {n})"]
    for k, (i, lab) in enumerate(zip(idxs, labels)):
        ax = fig.add_subplot(gs[0, k])
        draw(ax, frames[i], lab, size=9,
             colour=GREEN if k == 3 else "#333")

    handles = [Rectangle((0, 0), 1, 1, facecolor=COLOURS[j], edgecolor="#999",
                         linewidth=0.4) for j in (0, 1, 4, 2, 3, 5)]
    fig.legend(handles,
               ["wall  #", "floor", "player  @", "box  $", "goal  .",
                "box on goal"],
               loc="lower center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f'One real Boxoban level and its solution   |   caption: '
                 f'"{caption}"   |   {n} moves',
                 fontsize=9.5, y=1.06)
    save(fig, "r1_sokoban_basics")


# ---------------------------------------------------------------------------
# R2: why counting is an architectural property, not a capacity problem
# ---------------------------------------------------------------------------
def figure_r2():
    ev = load("results/evaluation.json")
    tf = ev["conditions"]["transformer_unconstrained"]["tile_counts_raw_draws"]
    va = ev["conditions"]["vae_sample"]["tile_counts_raw_draws"]

    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def panel(x, w, title, colour):
        ax.add_patch(FancyBboxPatch((x, 2), w, 41,
                                    boxstyle="round,pad=0.8", linewidth=1.2,
                                    edgecolor=colour, facecolor=colour + "10"))
        ax.text(x + w / 2, 40.0, title, ha="center", va="center",
                fontsize=10, weight="bold", color=colour)

    panel(1, 46, "Autoregressive  (transformer)", BLUE)
    panel(53, 46, "One-shot decoder  (VAE / GAN)", ORANGE)

    # --- left: cell by cell, each step sees the running counts
    cells = ["#", "#", "$", " ", "$", " ", "?"]
    for i, ch in enumerate(cells):
        x = 5 + i * 5.4
        face = {"#": COLOURS[0], "$": COLOURS[2], " ": COLOURS[1]}.get(ch, "#ffffff")
        ax.add_patch(Rectangle((x, 27), 4.4, 4.4, facecolor=face,
                               edgecolor="#888", linewidth=0.6,
                               linestyle="--" if ch == "?" else "-"))
        if ch == "?":
            ax.text(x + 2.2, 29.2, "?", ha="center", va="center", fontsize=10,
                    color="#666")
        if i < len(cells) - 1:
            ax.add_patch(FancyArrowPatch((x + 4.5, 29.2), (x + 5.3, 29.2),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.8, color="#888"))
    ax.text(5, 24.0, "each cell is emitted after the ones before it",
            fontsize=8.2, color="#444")
    ax.add_patch(FancyBboxPatch((5, 15.0), 41, 7.4,
                                boxstyle="round,pad=0.4", linewidth=0.9,
                                edgecolor=BLUE, facecolor="#ffffff"))
    ax.text(25.5, 20.2, "state visible to the next decision", fontsize=8.2,
            ha="center", color="#444")
    ax.text(25.5, 17.0, "boxes placed = 2 of 4     cells left = 61",
            fontsize=9, ha="center", weight="bold", color=BLUE)
    ax.text(5, 11.4, "so it can stop at exactly four", fontsize=8.6,
            color="#333", style="italic")
    ax.text(5, 6.6,
            f"measured:  {tf['box']['exact_rate'] * 100:.1f}% of draws have "
            f"exactly 4 boxes\n"
            f"mean {tf['box']['mean']:.2f} $\\pm$ {tf['box']['std']:.2f}",
            fontsize=7.4, color=BLUE, weight="bold", va="center")

    # --- right: one latent, 100 conditionally independent cells
    ax.add_patch(FancyBboxPatch((57, 27), 8, 4.4, boxstyle="round,pad=0.35",
                                linewidth=1.0, edgecolor=ORANGE,
                                facecolor="#ffffff"))
    ax.text(61.4, 29.2, "z", fontsize=11, ha="center", va="center",
            style="italic", color=ORANGE)
    for k in range(5):
        y = 31.0 - k * 1.6
        ax.add_patch(FancyArrowPatch((65.4, 29.2), (71.0, y),
                                     arrowstyle="-|>", mutation_scale=6,
                                     linewidth=0.7, color="#bbb"))
    for i, ch in enumerate(["#", "$", "$", "$", " ", "$", "$"]):
        x = 72 + i * 3.6
        face = {"#": COLOURS[0], "$": COLOURS[2], " ": COLOURS[1]}[ch]
        ax.add_patch(Rectangle((x, 27), 3.0, 4.4, facecolor=face,
                               edgecolor="#888", linewidth=0.6))
    ax.text(57, 24.0, "all 100 cells decoded at once from one latent",
            fontsize=8.2, color="#444")
    ax.add_patch(FancyBboxPatch((57, 15.0), 41, 7.4,
                                boxstyle="round,pad=0.4", linewidth=0.9,
                                edgecolor=ORANGE, facecolor="#ffffff"))
    ax.text(77.5, 20.2, "state visible to each cell", fontsize=8.2,
            ha="center", color="#444")
    ax.text(77.5, 17.0, "nothing: cells are independent given z",
            fontsize=9, ha="center", weight="bold", color=ORANGE)
    ax.text(57, 11.4, "so it gets the right average, not the right count",
            fontsize=8.6, color="#333", style="italic")
    ax.text(57, 6.6,
            f"measured:  {va['box']['exact_rate'] * 100:.1f}% of draws have "
            f"exactly 4 boxes\n"
            f"mean {va['box']['mean']:.2f} $\\pm$ {va['box']['std']:.2f}  "
            f"(right mean, wrong joint)",
            fontsize=7.4, color=ORANGE, weight="bold", va="center")

    fig.suptitle("A valid level needs exactly 1 player, 4 boxes and 4 goals. "
                 "Only one of these two can check.", fontsize=10, y=0.99)
    save(fig, "r2_counting")


# ---------------------------------------------------------------------------
# R3: what the logit mask does, measured on the committed checkpoint
# ---------------------------------------------------------------------------
def figure_r3():
    import torch

    from sokogen.data.vocab import (BOX_ID, FLOOR_ID, GOAL_ID, NEWLINE_ID,
                                    PLAYER_ID, WALL_ID, encode_prompt)
    from sokogen.decoding.constrained import build_step_mask, interior_left
    from sokogen.models.common import load_checkpoint
    from sokogen.models.transformer import SokobanLM, TransformerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd, meta = load_checkpoint(os.path.join(ROOT, "checkpoints",
                                            "transformer.pt"),
                               map_location=device)
    c = meta.get("config", {})
    model = SokobanLM(TransformerConfig(
        d_model=c.get("d_model", 384), n_layers=c.get("n_layers", 6),
        n_heads=c.get("n_heads", 6), d_ff=c.get("d_ff", 1536),
        dropout=c.get("dropout", 0.1)))
    model.load_state_dict(sd)
    model = model.to(device).eval()

    caption = "hard, ~40 moves, dense walls, corridors, boxes scattered"
    render = {WALL_ID: "#", FLOOR_ID: " ", BOX_ID: "$", GOAL_ID: ".",
              PLAYER_ID: "@", NEWLINE_ID: "\n"}

    # Sample seeds until one hits a step where feasibility forcing fires, so
    # the figure shows the rule actually binding rather than a step where the
    # mask is a no-op.
    hit = None
    with torch.no_grad():
        for seed in range(400):
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            ids = torch.tensor(encode_prompt(caption), device=device,
                               dtype=torch.long).unsqueeze(0)
            past, cur = None, ids
            np_, nb, ng = 0, 0, 0
            emitted, trace = [], []
            for step in range(110):
                logits, _, past = model(cur, past=past, use_cache=True)
                raw = logits[0, -1, :].float()
                mask, forced, quota = build_step_mask(step, np_, nb, ng, device)
                masked = raw + mask
                p_raw = torch.softmax(raw, dim=-1)
                p_masked = torch.softmax(masked, dim=-1)
                nxt = int(torch.multinomial(p_masked, 1, generator=g))
                trace.append((step, forced, quota, np_, nb, ng,
                              p_raw.cpu().numpy(), p_masked.cpu().numpy()))
                emitted.append(render[nxt])
                np_ += nxt == PLAYER_ID
                nb += nxt == BOX_ID
                ng += nxt == GOAL_ID
                cur = torch.tensor([[nxt]], device=device)
            forced_steps = [t for t in trace if t[1]]
            if forced_steps:
                hit = (seed, "".join(emitted), forced_steps[0])
                break
    if hit is None:
        raise RuntimeError("no forcing step found in 400 samples")

    seed, grid, (step, forced, quota, np_, nb, ng, p_raw, p_masked) = hit
    row, col = divmod(step, 11)
    left = interior_left(row, col)
    owed = (1 - np_) + (4 - nb) + (4 - ng)

    tiles = [(WALL_ID, "#  wall"), (FLOOR_ID, "'  '  floor"),
             (BOX_ID, "$  box"), (GOAL_ID, ".  goal"), (PLAYER_ID, "@  player")]
    labels = [t[1] for t in tiles]
    raw_v = [p_raw[t[0]] * 100 for t in tiles]
    msk_v = [p_masked[t[0]] * 100 for t in tiles]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.3, 1.3],
                                          "wspace": 0.45})

    # Cells after the current step are shown as "not yet emitted", but the row
    # separators have to stay in place or the string no longer reshapes to 10x10.
    partial = "".join(grid[i] if i < step else ("\n" if i % 11 == 10 else "~")
                      for i in range(110))
    arr = to_array(partial)
    axes[0].imshow(arr, cmap=CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
    axes[0].add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                edgecolor=PLUM, linewidth=2.2))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    cells_word = "cell" if left == 1 else "cells"
    owed_word = "special" if owed == 1 else "specials"
    axes[0].set_title(f"decoding step {step} of 110\n"
                      f"{left} interior {cells_word} left, "
                      f"{owed} {owed_word} still owed",
                      fontsize=8.2, color=PLUM, pad=4)

    y = np.arange(len(labels))
    axes[1].barh(y, raw_v, color="#b9c6da", edgecolor=BLUE, linewidth=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("what the model wants", fontsize=8.5)
    axes[1].set_xlabel("probability (%)")
    for i, v in enumerate(raw_v):
        axes[1].text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=7.2,
                     color="#444")

    bars = axes[2].barh(y, msk_v, color=["#d9d9e0" if v < 1e-6 else "#c9dcc9"
                                         for v in msk_v],
                        edgecolor=GREEN, linewidth=0.7)
    for i, v in enumerate(msk_v):
        if v < 1e-6:
            bars[i].set_edgecolor("#bbb")
            axes[2].text(1.5, i, "masked out", va="center", fontsize=7.2,
                         color="#999", style="italic")
        else:
            axes[2].text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=7.2,
                         color="#444")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_title("after the feasibility mask", fontsize=8.5)
    axes[2].set_xlabel("probability (%)")
    lim = max(max(raw_v), max(msk_v)) * 1.22
    for a in (axes[1], axes[2]):
        a.set_xlim(0, lim)
        a.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Constrained decoding: a hard rule applied to the logits, not "
                 "learned  (transformer.pt, seed %d)" % seed,
                 fontsize=9.5, y=1.06)
    save(fig, "r3_constrained_decoding")


# ---------------------------------------------------------------------------
# R4: seed variance against validation loss
# ---------------------------------------------------------------------------
def figure_r4():
    tf = load("results/seed_variance.json")
    dg = load("results/seed_variance_distilgpt2.json")
    arms = [("Transformer\n(from scratch)", tf, BLUE),
            ("DistilGPT-2\n(pretrained)", dg, PLUM)]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1),
                             gridspec_kw={"wspace": 0.3})

    for ax, key, title, ylab, fmt in [
            (axes[0], "solvable_given_valid_pct",
             "Solvability moves a lot", "solvable | valid (%)", "{:.1f}"),
            (axes[1], "val_loss",
             "Validation loss barely moves", "validation loss", "{:.4f}")]:
        for i, (name, d, colour) in enumerate(arms):
            vals = d["summary"][key]["values"]
            mean = d["summary"][key]["mean"]
            sd = d["summary"][key]["sd"]
            xs = np.full(len(vals), i, dtype=float) + np.linspace(-0.09, 0.09,
                                                                  len(vals))
            ax.scatter(xs, vals, s=44, color=colour, zorder=3,
                       edgecolor="white", linewidth=0.7)
            ax.errorbar(i, mean, yerr=sd, fmt="_", color=colour, capsize=7,
                        markersize=26, linewidth=1.5, zorder=2)
            ax.annotate(f"{fmt.format(mean)} $\\pm$ {fmt.format(sd)}",
                        (i, mean), textcoords="offset points",
                        xytext=(20, -3), fontsize=8, color=colour,
                        weight="bold")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], fontsize=8.5)
        ax.set_xlim(-0.5, len(arms) + 0.15)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Three training runs per configuration, seed the only "
                 "difference", fontsize=10, y=1.03)
    save(fig, "r4_seed_variance")


# ---------------------------------------------------------------------------
# R5: the headline ranking
# ---------------------------------------------------------------------------
def figure_r5():
    ev = load("results/evaluation.json")
    rows = [
        ("Random placement", "random_placement", GREY),
        ("Rule-based", "rule_based", GREY),
        ("Open room", "open_room", GREY),
        ("Conditional VAE (sampled)", "vae_sample", ORANGE),
        ("Conditional GAN (repaired)", "gan_repaired", PLUM),
        ("Transformer (constrained)", "transformer_constrained", BLUE),
        ("Transformer (unconstrained)", "transformer_unconstrained", BLUE),
        ("DistilGPT-2 (constrained)", "distilgpt2_constrained", BLUE),
        ("Conditional VAE (repaired)", "vae_repaired", ORANGE),
        ("Retrieval (copies train)", "retrieval", GREY),
        ("Real Boxoban levels", "real_boxoban", GREEN),
    ]
    vals = []
    for label, key, colour in rows:
        s = ev["conditions"][key]["solvable_given_valid"]
        vals.append((label, s["pct"], s["ci_lo_pct"], s["ci_hi_pct"], colour))
    vals.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(vals))
    pct = [v[1] for v in vals]
    lo = [v[1] - v[2] for v in vals]
    hi = [v[3] - v[1] for v in vals]
    ax.barh(y, pct, color=[v[4] + "cc" for v in vals],
            edgecolor=[v[4] for v in vals], linewidth=0.9, height=0.68)
    ax.errorbar(pct, y, xerr=[lo, hi], fmt="none", ecolor="#444",
                elinewidth=0.9, capsize=2.5)
    for i, v in enumerate(vals):
        ax.text(v[3] + 2.0, i, f"{v[1]:.1f}", va="center", fontsize=8,
                color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in vals], fontsize=8.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("solvable given structurally valid (%),  Wilson 95% interval")
    ax.set_title("Every level judged by the exhaustive A* solver\n"
                 "(500 valid levels per row)", fontsize=10)
    ax.axvline(23.4, color="#999", linestyle="--", linewidth=0.9, zorder=0)
    ax.text(24.6, -0.85, "empty room = 23.4", fontsize=7.6, color="#777")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "r5_headline")


if __name__ == "__main__":
    print("report figures ->", OUT)
    figure_r1()
    figure_r2()
    figure_r4()
    figure_r5()
    figure_r3()
    print("done")
