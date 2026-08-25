| Model | Difficulty bin | Density bin | Connectivity bin | Clustering bin | Length Spearman | Length censoring % | n |
|---|---|---|---|---|---|---|---|
| Random placement | 40.0 | 50.0 | 50.0 | 49.5 | -- | 100.0 | 0 |
| Open room | 37.5 | 50.0 | 50.0 | 49.7 | 0.057 | 83.5 | 99 |
| Rule-based | 39.8 | 50.0 | 50.0 | 49.0 | -- | 99.7 | 2 |
| Retrieval | 32.7 | 49.0 | 51.5 | 51.2 | 0.021 | 0.8 | 595 |
| Conditional GAN (raw) | 53.3 | 75.6 | 64.4 | 60.0 | -0.289 (n=22, NOISE) | 51.1 | 22 |
| Conditional GAN (repaired) | 39.0 | 85.7 | 64.0 | 53.8 | 0.074 | 59.8 | 241 |
| Conditional VAE (argmax) | -- | -- | -- | -- | -- | -- | 0 |
| Conditional VAE (sampled) | 50.0 | 100.0 | 87.5 | 87.5 | 0.866 (n=3, NOISE) | 62.5 | 3 |
| Conditional VAE (repaired) | 46.2 | 88.5 | 74.2 | 82.3 | 0.389 | 27.0 | 438 |
| Transformer (unconstrained) | 41.3 | 88.3 | 84.9 | 78.6 | 0.516 | 47.8 | 249 |
| Transformer (constrained) | 43.3 | 88.8 | 80.0 | 76.3 | 0.484 | 55.2 | 269 |
| DistilGPT-2 (constrained, ablation) | 53.5 | 91.5 | 85.5 | 86.2 | 0.673 | 37.8 | 373 |
| Real Boxoban levels | 31.0 | 48.8 | 53.2 | 52.2 | -0.048 | 1.2 | 593 |

Correlations computed on fewer than 30 levels are marked NOISE inline; they are reported for completeness, not as evidence.

Wall density, connectivity and box clustering are computable directly from the grid, so a model that reproduces surface statistics will score well on them.  **Solution length is not computable without search**, so it is the only attribute that tests real understanding.

Censoring % is the fraction of suite samples with no achieved length (invalid, proven unsolvable, or timed out).  The correlation is computed on the remaining subsample, which is biased toward easy levels -- the direction that inflates it.

Achieved solution length uses the exact move-optimal solver (`cost_mode="moves"`), matching the move units used in captions.
