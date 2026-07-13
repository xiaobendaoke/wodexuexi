# Force-Admission Sensitivity Audit

## Summary

- Audited experiments: UE 数量敏感性, UAV 算力敏感性.
- Minimum processed request ratio across all raw records: 100.00%.
- Generated/processed mismatch records: 0.
- DSR is audited as request-weighted DSR when available; it is not treated as service admission rate.

## Detailed Checks

| Experiment | Variable | Algorithm | Raw records | Summary samples | Generated | Processed | Min processed | Forced admission | Natural coverage | Mean DSR | Mismatches |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UE 数量敏感性 | 60 | heuristic | 180 | 30 | 3565713 | 3565713 | 100.00% | 54.69% | 45.31% | 0.7666 | 0 |
| UE 数量敏感性 | 60 | vanilla_mappo | 180 | 30 | 4323420 | 4323420 | 100.00% | 44.64% | 55.36% | 0.5277 | 0 |
| UE 数量敏感性 | 60 | proposed | 180 | 30 | 3180128 | 3180128 | 100.00% | 45.38% | 54.62% | 0.5928 | 0 |
| UE 数量敏感性 | 80 | heuristic | 180 | 30 | 5367949 | 5367949 | 100.00% | 54.34% | 45.66% | 0.6139 | 0 |
| UE 数量敏感性 | 80 | vanilla_mappo | 180 | 30 | 4754314 | 4754314 | 100.00% | 35.87% | 64.13% | 0.3819 | 0 |
| UE 数量敏感性 | 80 | proposed | 180 | 30 | 4045726 | 4045726 | 100.00% | 46.02% | 53.98% | 0.4996 | 0 |
| UE 数量敏感性 | 100 | heuristic | 180 | 30 | 5473128 | 5473128 | 100.00% | 57.00% | 43.00% | 0.5726 | 0 |
| UE 数量敏感性 | 100 | vanilla_mappo | 180 | 30 | 5746829 | 5746829 | 100.00% | 35.38% | 64.62% | 0.3042 | 0 |
| UE 数量敏感性 | 100 | proposed | 180 | 30 | 6397666 | 6397666 | 100.00% | 48.07% | 51.93% | 0.3684 | 0 |
| UE 数量敏感性 | 120 | heuristic | 180 | 30 | 6952771 | 6952771 | 100.00% | 57.12% | 42.88% | 0.4669 | 0 |
| UE 数量敏感性 | 120 | vanilla_mappo | 180 | 30 | 8534569 | 8534569 | 100.00% | 35.32% | 64.68% | 0.2156 | 0 |
| UE 数量敏感性 | 120 | proposed | 180 | 30 | 6770564 | 6770564 | 100.00% | 50.17% | 49.83% | 0.3432 | 0 |
| UE 数量敏感性 | 140 | heuristic | 180 | 30 | 7701856 | 7701856 | 100.00% | 54.84% | 45.16% | 0.3872 | 0 |
| UE 数量敏感性 | 140 | vanilla_mappo | 180 | 30 | 8468528 | 8468528 | 100.00% | 38.86% | 61.14% | 0.2086 | 0 |
| UE 数量敏感性 | 140 | proposed | 180 | 30 | 8138553 | 8138553 | 100.00% | 51.50% | 48.50% | 0.2862 | 0 |
| UAV 算力敏感性 | 0.6 | heuristic | 180 | 30 | 7549468 | 7549468 | 100.00% | 56.88% | 43.12% | 0.3187 | 0 |
| UAV 算力敏感性 | 0.6 | vanilla_mappo | 180 | 30 | 5431928 | 5431928 | 100.00% | 35.27% | 64.73% | 0.2296 | 0 |
| UAV 算力敏感性 | 0.6 | proposed | 180 | 30 | 6914736 | 6914736 | 100.00% | 47.23% | 52.77% | 0.2126 | 0 |
| UAV 算力敏感性 | 0.8 | heuristic | 180 | 30 | 5856318 | 5856318 | 100.00% | 57.10% | 42.90% | 0.4565 | 0 |
| UAV 算力敏感性 | 0.8 | vanilla_mappo | 180 | 30 | 6617466 | 6617466 | 100.00% | 34.69% | 65.31% | 0.2289 | 0 |
| UAV 算力敏感性 | 0.8 | proposed | 180 | 30 | 4955285 | 4955285 | 100.00% | 48.98% | 51.02% | 0.3536 | 0 |
| UAV 算力敏感性 | 1.0 | heuristic | 180 | 30 | 7154957 | 7154957 | 100.00% | 56.99% | 43.01% | 0.4808 | 0 |
| UAV 算力敏感性 | 1.0 | vanilla_mappo | 180 | 30 | 5530339 | 5530339 | 100.00% | 36.49% | 63.51% | 0.3053 | 0 |
| UAV 算力敏感性 | 1.0 | proposed | 180 | 30 | 5443109 | 5443109 | 100.00% | 50.00% | 50.00% | 0.4003 | 0 |
| UAV 算力敏感性 | 1.2 | heuristic | 180 | 30 | 5071208 | 5071208 | 100.00% | 57.09% | 42.91% | 0.6064 | 0 |
| UAV 算力敏感性 | 1.2 | vanilla_mappo | 180 | 30 | 5227758 | 5227758 | 100.00% | 36.23% | 63.77% | 0.3416 | 0 |
| UAV 算力敏感性 | 1.2 | proposed | 180 | 30 | 6075655 | 6075655 | 100.00% | 48.69% | 51.31% | 0.3860 | 0 |
| UAV 算力敏感性 | 1.4 | heuristic | 180 | 30 | 5715145 | 5715145 | 100.00% | 57.04% | 42.96% | 0.5876 | 0 |
| UAV 算力敏感性 | 1.4 | vanilla_mappo | 180 | 30 | 5642744 | 5642744 | 100.00% | 35.66% | 64.34% | 0.3341 | 0 |
| UAV 算力敏感性 | 1.4 | proposed | 180 | 30 | 6360260 | 6360260 | 100.00% | 48.54% | 51.46% | 0.4030 | 0 |

## Interpretation Guardrails

- Force-admission guarantees that service requests enter the processing chain; it does not guarantee deadline satisfaction.
- Higher forced admission ratio means more requests required nearest-UAV fallback access, not necessarily worse execution quality by itself.
- Sensitivity figures should present DSR together with EEE and fairness to avoid overstating DSR-only performance.
