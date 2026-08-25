| Model | OOD length Spearman | OOD censoring % | Mean requested | Mean achieved | n |
|---|---|---|---|---|---|
| Random placement | -- | 100.0 | -- | -- | 0 |
| Open room | -0.047 | 77.5 | 116.7 | 26.9 | 27 |
| Rule-based | -- | 100.0 | -- | -- | 0 |
| Retrieval | 0.099 | 7.5 | 116.1 | 32.5 | 111 |
| Conditional GAN (raw) | -- | 77.8 | -- | -- | 2 |
| Conditional GAN (repaired) | -0.131 | 62.5 | 117.1 | 24.9 | 45 |
| Conditional VAE (argmax) | -- | -- | -- | -- | 0 |
| Conditional VAE (sampled) | -- | 66.7 | -- | -- | 1 |
| Conditional VAE (repaired) | -0.059 | 55.0 | 117.8 | 42.3 | 54 |
| Transformer (unconstrained) | 0.394 | 63.9 | 96.2 | 37.7 | 13 |
| Transformer (constrained) | -0.394 | 60.0 | 115.8 | 29.5 | 48 |
| DistilGPT-2 (constrained, ablation) | -0.274 | 51.7 | 120.0 | 30.4 | 58 |
| Real Boxoban levels | 0.131 | 0.8 | 116.9 | 30.3 | 119 |

Out-of-distribution requests ask for solution lengths of 90, 110 and 150 moves.  Measured on the training split: p90 = 47, p99 = 67, p99.9 = 87, maximum = 130.  Failure to extrapolate is the expected result and populates the Failure Analysis section.
