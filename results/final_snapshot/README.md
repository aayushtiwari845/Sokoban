# Committed results snapshot

`results/*.json` is gitignored so that re-running the pipeline does not produce
noisy diffs. This directory is the committed snapshot backing every number in
the paper and the top-level README.

| File | What it backs |
|---|---|
| `data_verification.json` | GATE 1 — alphabet, invariants, solution units, coverage, duplicates |
| `solver_validation.json` | GATE 2 — solve rate, soundness ablation, node-cap curve, move-bound check |
| `dataset_build.json` | GATE 3 — caption bin boundaries and split statistics |
| `generation.json` | Samples drawn per condition, forced-token rates |
| `evaluation.json` | **The single source for every table and figure** |
| `seed_variance.json` | Three training seeds of the primary transformer |
| `train_*.json` | Loss curves, parameter counts, wall-clock training times |

Each file embeds the config, random seed, git SHA and platform it was produced
on (see the `provenance` block).

## Model checkpoints

Checkpoints total ~297 MB and are deliberately **not** committed; they are
written to `checkpoints/` by `scripts/03_train.py`:

| File | Size | Model |
|---|---|---|
| `transformer.pt` | 43 MB | primary, seed 1337 |
| `transformer_s1338.pt`, `transformer_s1339.pt` | 43 MB each | seed-variance runs |
| `vae.pt` | 4.4 MB | conditional VAE |
| `gan.pt` | 4.3 MB | conditional GAN (generator + discriminator state) |
| `distilgpt2.pt` | 173 MB | pretraining ablation |

Every one is reproducible from a single command with a fixed seed; see the
reproduction section of the top-level README.
