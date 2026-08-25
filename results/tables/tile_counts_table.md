| Model | Boxes (mean +/- sd) | Exactly 4 boxes % | Goals (mean +/- sd) | Exactly 4 goals % | Players (mean) | Exactly 1 player % |
|---|---|---|---|---|---|---|
| Random placement | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Open room | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Rule-based | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Retrieval | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Conditional GAN (raw) | 3.93 +/- 0.89 | 48.0 | 3.72 +/- 0.94 | 40.5 | 0.68 | 44.2 |
| Conditional GAN (repaired) | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Conditional VAE (argmax) | 1.38 +/- 1.23 | 4.7 | 0.01 +/- 0.12 | 0.0 | 0.01 | 0.5 |
| Conditional VAE (sampled) | 3.95 +/- 1.72 | 21.8 | 4.05 +/- 1.85 | 20.1 | 1.02 | 35.4 |
| Conditional VAE (repaired) | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Transformer (unconstrained) | 3.87 +/- 0.34 | 87.1 | 3.95 +/- 0.24 | 94.1 | 1.00 | 98.6 |
| Transformer (constrained) | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| DistilGPT-2 (constrained, ablation) | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |
| Real Boxoban levels | 4.00 +/- 0.00 | 100.0 | 4.00 +/- 0.00 | 100.0 | 1.00 | 100.0 |

Computed over **raw draws**, valid or not: restricting to valid levels would define the counting failure away.  A valid level needs exactly one player, four boxes and four goals.
