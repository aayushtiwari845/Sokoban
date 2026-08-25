# Model Card — Solver-Verified Text-Conditioned Sokoban Level Generation

Three generative models that produce 10×10 Sokoban levels from short text
captions, evaluated with an exhaustive A\* solver rather than a soft metric.

This work is a **replication and extension of Todd et al. 2023 (FDG), "Level
Generation Through Large Language Models"** (`github.com/gdrtodd/lm-pcg`). The
data-scaling finding, the solvability-versus-conditioning setup and the
tokenization finding are theirs, not ours.

---

## 1. Overview

| | Transformer (primary) | Conditional VAE | Conditional GAN | DistilGPT-2 (ablation) |
|---|---|---|---|---|
| Architecture | decoder-only LM, from scratch | MLP encoder/decoder | MLP generator/discriminator | pretrained body, new vocabulary |
| Parameters | 10,734,336 | 1,098,292 | 1,082,357 (G+D) | 43,352,064 |
| Conditioning | caption **text** prefix | caption **vector** (10-dim) | caption **vector** (10-dim) | caption **text** prefix |
| Output | 110 characters, autoregressive | 5×10×10 logits, one shot | 5×10×10 logits, one shot | 110 characters, autoregressive |
| Precision | bf16 (Ampere native) | bf16 | bf16 | bf16 |

All models were trained on a single RTX 3050 6GB laptop GPU.

---

## 2. Intended use

Research on procedural content generation and on **verifiable** evaluation of
generative models. The models produce puzzle levels for a single game at a
single fixed size. They are not intended for, and have not been evaluated for,
any use involving people, text about people, or decisions affecting people.

**Out of scope:** anything other than 10×10 Boxoban-style Sokoban level
generation. The vocabulary contains 48 characters and the models have never seen
natural language beyond the fixed caption grammar below.

---

## 3. Training data

- **Levels:** [Boxoban](https://github.com/google-deepmind/boxoban-levels)
  (Guez et al., 2018), `unfiltered` split. 100,000 training levels, 5,000 from
  the **official** validation split, 1,000 held-out test levels.
- **Solution lengths:** `AlignmentResearch/boxoban-astar-solutions`
  (Garriga-Alonso, Taufeeque & Gleave, ICML 2024 MI Workshop).

### Verified data properties (Phase 1)

| Property | Finding |
|---|---|
| Tile alphabet | Exactly `{'#', '@', '$', '.', ' '}`. **Neither `*` nor `+` occurs**, so a box never starts on a goal — the 5-channel one-hot representation is faithful. |
| Line endings | Files are **CRLF**; the parser normalises them. |
| Grid invariants | 10,000/10,000 levels are 10×10 with a wall border and exactly (1 player, 4 boxes, 4 goals). |
| Solution units | `Steps` counts **player moves, not pushes** (`Steps == len(Actions)`; actions are digits 0–3 = Up, Right, Down, Left). |
| Cross-split leakage | Zero collisions between train and test/validation. |

### Data-quality finding (not documented upstream)

The solutions dataset documents rows marked `SEARCH_STATE_FAILED`/`NOT_FOUND`.
We additionally found rows that **look** well-formed — a plain digit string —
but **do not replay to a solved state**: 63/1000 (6.3%) in `unfiltered_test` and
~0.7% in `unfiltered_train`. No direction convention, transposition, or
no-op-on-block semantics rescues them.

**Every solution used here is replay-validated**, and levels whose solution
fails to replay are dropped (0.73% of train, 7.4% of val/test). A caption built
on a wrong length is worse than no example.

---

## 4. Captions

Five features, all computed from the grid — nothing is hand-labelled.

| Feature | Computation | Phrase |
|---|---|---|
| Difficulty | solution length, tercile of the training distribution (≤26 / ≤35 / >35 moves) | `easy` / `medium` / `hard` |
| Solution length | rounded to nearest 5 | `~40 moves` |
| Wall density | interior `#` / 64, split at the training median (0.500) | `open room` / `dense walls` |
| Connectivity | mean non-wall neighbours per floor cell, split at the training median (2.889) | `corridors` / `single chamber` |
| Box clustering | mean pairwise Manhattan distance between boxes, split at the training median (3.500) | `boxes clustered` / `boxes scattered` |

Example: `"hard, ~40 moves, dense walls, corridors, boxes scattered"`.

**Connectivity was chosen empirically.** The specification offered "count of
connected floor components" as an alternative; measured over 20,000 training
levels, **every Boxoban level has exactly one floor component**, so that feature
is constant and cannot bin anything.

Two parallel encodings of the same information are used: `caption_text` for the
transformer and a 10-dim `caption_vec` for the VAE and GAN.

---

## 5. Training configuration

Hyperparameters marked **untuned** were chosen once and never swept. We state
this rather than inventing a justification for them.

| | Transformer | VAE | GAN | DistilGPT-2 |
|---|---|---|---|---|
| Optimiser | AdamW, wd 0.01 | Adam | Adam, β=(0.5, 0.999) | AdamW |
| Learning rate | 3e-4 **untuned** | 1e-3 **untuned** | 2e-4 **untuned** | 3e-4 **untuned** |
| Schedule | cosine, 200 warmup **untuned** | constant | constant | cosine |
| Batch size | 64 | 128 | 128 | 32 |
| Epochs | 3 **untuned** | 20 **untuned** | 20 **untuned** | 3 **untuned** |
| Loss | next-token CE, masked over the caption prefix | per-cell categorical CE + free-bits KL | non-saturating GAN, one-sided label smoothing 0.9 | as transformer |

Design choices that are **not** arbitrary, and why:

- **Per-cell categorical cross-entropy, not MSE**, for the VAE. MSE on one-hot
  tiles treats a categorical channel axis as a metric space; it is a common and
  wrong default.
- **Free bits (0.05 nats/dim)** instead of KL annealing to prevent posterior
  collapse — one line, no schedule to tune. Measured KL was 8.8 nats, far above
  the 1.6-nat floor, so the posterior did not collapse.
- **Straight-through Gumbel-softmax** for the GAN, temperature annealed 1.0 →
  0.3. Training on continuous logits and taking argmax only at sampling time is
  not a valid option: the discriminator can then separate real one-hot tensors
  from soft generator output purely by per-cell entropy.
- **MLPs, not convolutions**, for both one-shot models. 10×10 does not factor
  cleanly for strided convolutions, and at 100 cells convolutions buy nothing
  while costing shape bugs.

---

## 6. Decoding

### Constrained decoding (transformer)

A manual sampling loop applies a per-step logit mask enforcing: a wall border,
newline at end of row, per-tile quotas, and — critically — **feasibility
forcing**. When the remaining interior cells exactly equal the specials still
owed, `#` and `' '` are masked so only specials can be emitted. Masking upper
bounds alone enforces "at most four boxes", not "exactly four".

The **forced-token rate** is logged as a free measure of how well the model
learned to count unaided.

**Limitation:** mask-then-renormalise at each step does not sample from the
model's distribution conditioned on the valid set. It is a greedy local
approximation, and it shifts the sampling distribution.

### Repair (one-shot families)

Invalid grids are projected onto the nearest valid grid using the model's own
predicted probabilities: border forced to wall; excess boxes/goals/players
dropped lowest-probability first; deficits filled highest-probability first;
specials never placed on the border. Repair is idempotent and is the identity on
already-valid grids.

Post-repair solvability separates two questions: *can the model count?*
(structural validity) versus *does it understand spatial structure?* (post-repair
solvability).

---

## 7. Evaluation

Levels are verified by an **exhaustive A\* solver**, not a learned or heuristic
proxy. Because the state space is bounded and every prune is sound, exhaustion
is a **proof** of unsolvability — the solver is a decision procedure, not a
one-sided test. Results are therefore reported three ways: **solved / proven
unsolvable / timed out**.

Solver design: push-based moves, player-region canonicalisation, an
assignment-based admissible heuristic (`scipy.optimize.linear_sum_assignment`
over per-goal push-distance maps), and two sound prunes (dead squares, freeze
deadlocks).

### Solver validation (GATE 2)

| Check | Result |
|---|---|
| Solve rate, 1,000 real test levels | **100.00%** (0 unsolvable, 0 timeout) |
| False "unsolvable" on real levels | **0** |
| Returned solutions that replay to a solved state | **1000/1000** |
| Soundness ablation (pruning off, 10× node cap) | **0 verdict disagreements** |
| Same ablation on 3 baseline distributions | **0 disagreements** (450 levels) |
| Our push-optimal ≤ published solution's pushes | 926/926 |
| Our reconstructed moves ≥ published move-optimal | 926/926 |
| Median / p99 nodes | 176 / 3,458 |
| Median / p99 wall time | 75 ms / 2.0 s |

Solve rate reaches 100% at a node cap of 10,000 and stays flat to 1,000,000, so
the headline number is a converged lower bound rather than a budget artifact.

### Reconstructed move length is an upper bound

Push-optimal search discards the player's exact position, so the move length
reconstructed by walking the player between pushes is an **upper bound**, not
the optimum. Measured against exact move-optimal search on 196 levels: mean
excess **+40.5%**, exactly optimal on only 6.6% of levels, **zero** bound
violations.

Because of that gap, **achieved solution length for the controllability metric
is measured with the exact move-optimal solver**, matching the move units used
in captions, rather than with the cheap reconstruction.

---

## 8. Limitations

- Constrained decoding shifts the sampling distribution away from the model's
  own conditional distribution.
- **The conditioning channels differ across families** — text prefix for the
  transformer, a 10-dim vector for the VAE and GAN — so the family comparison is
  not perfectly controlled on conditioning.
- Reconstructed move length is an upper bound on move-optimal (quantified above).
- A single game, a single fixed 10×10 size, four boxes and four goals.
- Ablations are single-seed; only the main transformer configuration is run with
  three seeds. Table captions state this.
- Learning rate, batch size and epoch count were not tuned for any model.
- Structural-validity percentages come from an inverse-binomial sampling
  procedure (draw until 500 valid), so the Wilson interval is a close
  approximation rather than exact.
- Solution-length controllability is computed only on *solved* levels, a
  subsample biased toward easy levels. The censoring rate is reported alongside.

---

## 9. Ethics

The models generate puzzle layouts for a single game. They consume no personal
data, produce no text about people, and make no decisions about people. The
training corpus (Boxoban) is procedurally generated by DeepMind and contains no
human-authored or personal content.

The one genuine risk is misuse of the *evaluation* claim: "solver-verified" is a
strong guarantee **only** within the stated bounds — a 10×10 grid, four boxes,
a 200,000-node cap and a 10-second wall-clock cap. Outside those bounds the
timeout bucket is not a proof of anything, which is exactly why timeouts are
reported as their own column rather than folded into "unsolvable".

---

## 10. Reproduction

```bash
python scripts/00_verify_data.py       # GATE 1: verify every data assumption
python scripts/01_build_dataset.py     # GATE 3: captions and datasets
python scripts/02_validate_solver.py   # GATE 2: solver validation + ablation
python scripts/03_train.py --model transformer
python scripts/03_train.py --model vae
python scripts/03_train.py --model gan
python scripts/03_train.py --model distilgpt2
python scripts/04_generate.py          # GPU sampling
python scripts/05_evaluate.py          # CPU solver sweep, tables, figures
```

Fixed seed 1337 throughout; every script accepts `--seed`. Every result JSON
embeds its config, seed and git SHA.

**Never run `05_evaluate.py` while a GPU job is active** — the GPU heats the
package, CPU cores throttle, and solver throughput silently halves mid-sweep.
