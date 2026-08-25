| Model | Struct. valid % | Solvable % | Solvable \| valid % | Timeout % | Samples drawn / 500 valid | Novel % | Novelty (NN dist) | Diversity | Params | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Random placement | 100.0 [99.3, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.8] | 0.0 | 512 | 100.0 | 23.97 | 39.20 | -- | -- |
| Open room | 100.0 [99.3, 100.0] | 23.4 [19.7, 27.1] | 23.4 [19.9, 27.3] | 0.0 | 512 | 100.0 | 20.78 | 16.22 | -- | -- |
| Rule-based | 100.0 [99.3, 100.0] | 0.4 [0.0, 1.0] | 0.4 [0.1, 1.4] | 0.0 | 512 | 100.0 | 21.55 | 38.14 | -- | -- |
| Retrieval | 100.0 [99.3, 100.0] | 100.0 [100.0, 100.0] | 100.0 [99.2, 100.0] | 0.0 | 512 | 0.0 | 0.00 | 38.73 | -- | -- |
| Conditional GAN (raw) | 8.6 [8.0, 9.4] | 4.3 [3.8, 4.9] | 50.2 [45.8, 54.6] | 0.0 | 6,144 | 100.0 | 11.63 | 34.18 | 1,082,357 | 153 |
| Conditional GAN (repaired) | 100.0 [99.3, 100.0] | 40.2 [35.9, 44.5] | 40.2 [36.0, 44.6] | 0.0 | 512 | 100.0 | 12.60 | 36.79 | 1,082,357 | 153 |
| Conditional VAE (argmax) | 0.0 [0.0, 0.0] | -- | -- | -- | 100,000 | -- | -- | -- | 1,098,292 | 98 |
| Conditional VAE (sampled) | 1.7 [1.6, 1.9] | 0.5 [0.4, 0.6] | 30.0 [26.1, 34.2] | 0.0 | 29,184 | 100.0 | 12.84 | 38.54 | 1,098,292 | 98 |
| Conditional VAE (repaired) | 100.0 [99.3, 100.0] | 76.6 [72.9, 80.3] | 76.6 [72.7, 80.1] | 0.0 | 512 | 100.0 | 11.08 | 37.82 | 1,098,292 | 98 |
| Transformer (unconstrained) | 82.6 [80.2, 84.8] | 46.3 [42.4, 50.1] | 56.0 [51.6, 60.3] | 0.0 | 1,024 | 100.0 | 12.65 | 39.01 | 10,734,336 | 661 |
| Transformer (constrained) | 100.0 [99.3, 100.0] | 49.6 [45.2, 54.0] | 49.6 [45.2, 54.0] | 0.0 | 512 | 100.0 | 12.63 | 38.86 | 10,734,336 | 661 |
| DistilGPT-2 (constrained, ablation) | 100.0 [99.3, 100.0] | 63.6 [59.4, 67.8] | 63.6 [59.3, 67.7] | 0.0 | 512 | 100.0 | 12.18 | 38.69 | 43,352,064 | 4479 |
| Real Boxoban levels | 100.0 [99.3, 100.0] | 100.0 [100.0, 100.0] | 100.0 [99.2, 100.0] | 0.0 | 512 | 100.0 | 11.96 | 38.76 | -- | -- |

Percentages carry Wilson 95% intervals.  **Structural validity and solvability are reported separately and never merged**: `Solvable %` is of all samples drawn, `Solvable | valid %` is of structurally valid samples only.

Reference distributions on 500 held-out real levels: nearest-neighbour distance 11.92, diversity 38.81.  Novelty and diversity numbers are only interpretable against these.

Distance is cell-wise Hamming over the 100 tiles; novelty is symmetry-aware over all 8 dihedral transforms.

Single-seed unless stated otherwise; 3 seeds are run on the main transformer configuration only.

**Flags:**
- **Conditional VAE (argmax)**: DENOMINATOR 0 < 30: this percentage is noise, not signal
