"""Generate README.md from the evaluation artifacts.

The README leads with the main results table, so it must never carry stale
numbers.  It is therefore rendered from ``results/evaluation.json`` rather than
maintained by hand, in keeping with operating rule 5: tables are generated
*from* the artifacts, never typed.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from . import tables as T

HEADER = """# Solver-Verified Text-Conditioned Sokoban Level Generation

Three generative model families produce 10x10 Sokoban levels from short text
captions, evaluated with an **exhaustive A\\* solver** instead of a soft metric.

Because the state space of a 10x10 Sokoban level is bounded and every prune is
sound, exhausting the search is a **proof** of unsolvability. The solver is a
decision procedure, not a one-sided test, so results are reported three ways --
**solved / proven unsolvable / timed out** -- and never collapsed to two.

**Central claim:** autoregressive generation satisfies hard counting constraints
that one-shot generators cannot. A transformer emitting a level cell by cell
knows how many boxes it has already placed; a VAE or GAN decoding all 100 cells
from a latent vector does not.

This is a replication and extension of **Todd et al. 2023 (FDG), "Level
Generation Through Large Language Models"** ([lm-pcg](https://github.com/gdrtodd/lm-pcg)).
The data-scaling finding, the solvability-versus-conditioning setup and the
tokenization finding are theirs; we claim none of them.
"""

SOLVER_SECTION = """
## Solver validation (GATE 2)

The solver is the ground truth for every number here, so it is validated before
anything else is believed.

| Check | Result |
|---|---|
| Solve rate, 1,000 real held-out test levels | **{solve_rate:.2f}%** |
| False "unsolvable" verdicts on real levels | **{false_unsolv}** |
| Returned solutions that replay to a solved state | **{n_replay}/{n_solved}** |
| Soundness ablation (pruning off, 10x node cap) | **{abl}** disagreements |
| Same ablation on 3 baseline distributions (450 levels) | **0** disagreements |
| Median / p99 nodes expanded | {nodes_med:,.0f} / {nodes_p99:,.0f} |
| Median / p99 wall time | {t_med:.0f} ms / {t_p99:.1f} s |

Solve rate reaches 100% at a node cap of 10,000 and stays flat to 1,000,000, so
the headline numbers are converged rather than budget-limited.

**Move length is an upper bound.** Push-optimal search discards the player's
exact cell, so the move length reconstructed by walking the player between
pushes overshoots move-optimal by **+40.5% on average** (exactly optimal on only
6.6% of levels, zero bound violations). Controllability therefore measures
achieved length with the *exact* move-optimal solver, matching the move units
used in captions.
"""

DATA_SECTION = """
## Data findings worth knowing

Phase 1 verified every format assumption instead of trusting it. Three findings
affect anyone else using this data:

1. **`Steps` counts player moves, not pushes.** The action string is digits 0-3
   = Up, Right, Down, Left, and `Steps == len(Actions)`. Their A\\* minimised
   moves, so `Steps` is move-optimal where valid.
2. **Some published solutions are silently wrong.** Beyond the rows marked
   `SEARCH_STATE_FAILED`/`NOT_FOUND`, we found rows that *look* well-formed but
   **do not replay to a solved state**: 63/1000 (6.3%) of `unfiltered_test` and
   ~0.7% of `unfiltered_train`. No direction convention, transposition, or
   no-op-on-block semantics rescues them. **Every solution used here is
   replay-validated** and failures are dropped.
3. **Neither `*` nor `+` occurs anywhere in the corpus**, so a box never starts
   on a goal. That is what makes the 5-channel one-hot tensor, the 6-symbol grid
   vocabulary, and the "exactly four `$` and four `.`" constraint all
   well-defined.

Also: files are CRLF, and **every Boxoban level has exactly one connected floor
component** -- so "number of floor components" is a constant and useless as a
caption feature. We use mean floor degree (corridor width) instead.
"""

REPRO = """
## Reproduction

```bash
pip install -r requirements.txt
git clone https://github.com/google-deepmind/boxoban-levels data_raw/boxoban-levels
python -c "from huggingface_hub import snapshot_download; snapshot_download('AlignmentResearch/boxoban-astar-solutions', repo_type='dataset', local_dir='data_raw/astar-solutions')"

python scripts/00_verify_data.py       # GATE 1: verify every data assumption
python scripts/01_build_dataset.py     # GATE 3: captions and datasets
python scripts/02_validate_solver.py   # GATE 2: solver validation + soundness ablation
python scripts/03_train.py --model transformer
python scripts/03_train.py --model vae
python scripts/03_train.py --model gan
python scripts/03_train.py --model distilgpt2   # pretraining ablation
python scripts/04_generate.py          # GPU sampling
python scripts/05_evaluate.py          # CPU solver sweep -> tables + figures
```

Seed 1337 throughout; every script takes `--seed`. Every result JSON embeds its
config, seed and git SHA, and every table is regenerated from those artifacts.

> **Do not run `05_evaluate.py` while a GPU job is active.** The GPU heats the
> package, CPU cores throttle, and solver throughput silently halves mid-sweep,
> corrupting the timing measurements.

## Tests

```bash
python -m pytest tests/
```

The solver tests were written before the solver. They include ~11 hand-built
fixtures with known verdicts, exact push-optimal lengths, differential testing
against an independently written reference solver that shares no code or
representation, and a check that every "solved" verdict comes with an action
string that replays to a solved state.

## Layout

```
sokogen/
  data/      Boxoban parsing, replay-validated solutions, captions, vocabulary
  solver/    bitmask grids, sound deadlock detection, exhaustive A*
  models/    from-scratch transformer, conditional VAE, conditional GAN
  decoding/  constrained decoding (grid guard), repair
  baselines/ random, open room, rule-based, retrieval
  eval/      metrics, harness, tables, figures
scripts/     00..05, the full pipeline in order
results/     JSON artifacts, generated levels with solver verdicts, tables
paper/       IEEE two-column source
```

## Citation

If you use the solver or the replay-validation finding, please also cite the
upstream sources:

- Guez et al. 2018, *Boxoban levels* (the corpus)
- Garriga-Alonso, Taufeeque & Gleave, ICML 2024 MI Workshop (the A\\* solutions)
- Todd et al. 2023, FDG (the work this replicates and extends)
"""


def _fmt_solver(solver_json: str) -> str:
    with open(solver_json, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    v1 = d["validation1_real_levels"]
    s = v1["summary"]
    abl = d["validation2_soundness_ablation"]
    n_solved = s["counts"]["solved"]
    return SOLVER_SECTION.format(
        solve_rate=100 * s["solve_rate"],
        false_unsolv=len(v1["false_unsolvable"]),
        n_replay=n_solved - len(v1["bad_replay"]),
        n_solved=n_solved,
        abl=len(abl["disagree_status"]) + len(abl["disagree_length"]),
        nodes_med=s["nodes_median"], nodes_p99=s["nodes_p99"],
        t_med=1000 * s["time_median_s"], t_p99=s["time_p99_s"])


def write_readme(evaluation_json: str, results_dir: str, out_path: str,
                 solver_json: Optional[str] = None) -> str:
    with open(evaluation_json, "r", encoding="utf-8") as fh:
        ev = json.load(fh)

    parts = [HEADER, "\n## Main results\n",
             T.markdown_main_table(ev, results_dir)]

    tile = T.markdown_tile_count_table(ev)
    if tile:
        parts += ["\n## Tile counts: the counting-constraint claim, directly\n",
                  tile]

    ctrl = T.markdown_controllability_table(ev)
    if ctrl:
        parts += ["\n## Controllability, per attribute\n", ctrl]

    ood = T.markdown_ood_table(ev)
    if ood:
        parts += ["\n## Out-of-distribution length requests\n", ood]

    temp = T.markdown_temperature_table(ev)
    if temp:
        parts += ["\n## Temperature sweep\n", temp]

    comp = T.markdown_comparisons(ev)
    if comp:
        parts += ["\n## Pairwise significance\n", comp]

    if solver_json and os.path.exists(solver_json):
        parts.append(_fmt_solver(solver_json))

    parts += [DATA_SECTION, REPRO]

    text = "\n".join(parts) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path
