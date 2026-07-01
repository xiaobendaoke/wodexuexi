# Baseline Matrix Audit Report

- Input directory: `results/final_force_admission/evaluation_20260630`
- Metrics files: 240
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_full | 30 | 180 | 5866201 | 5866201 | 0 | 2325694 | 100.00% | 39.65% | 39.65% |
| ablation_no_attention | 30 | 180 | 5867125 | 5867125 | 0 | 2356035 | 100.00% | 40.16% | 40.16% |
| ablation_no_lagrange | 30 | 180 | 5862244 | 5862244 | 0 | 1988260 | 100.00% | 33.92% | 33.92% |
| ablation_no_mask | 30 | 180 | 5860202 | 5860202 | 0 | 2269376 | 100.00% | 38.73% | 38.73% |
| dsr_priority_full | 30 | 180 | 5860493 | 5860493 | 0 | 2059435 | 100.00% | 35.14% | 35.14% |
| main_full_hierarchical | 30 | 180 | 5869677 | 5869677 | 0 | 2201799 | 100.00% | 37.51% | 37.51% |
| main_lower_only_fixed_upper | 30 | 180 | 5866201 | 5866201 | 0 | 2325694 | 100.00% | 39.65% | 39.65% |
| main_upper_only | 30 | 180 | 5864851 | 5864851 | 0 | 2900150 | 100.00% | 49.45% | 49.45% |

## Admission Audit

| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ablation_full | 1178941 | 4687260 | 4687260 | 79.90% | 20.10% |
| ablation_no_attention | 1167694 | 4699431 | 4699431 | 80.10% | 19.90% |
| ablation_no_lagrange | 1191072 | 4671172 | 4671172 | 79.68% | 20.32% |
| ablation_no_mask | 1190946 | 4669256 | 4669256 | 79.68% | 20.32% |
| dsr_priority_full | 3155160 | 2705333 | 2705333 | 46.16% | 53.84% |
| main_full_hierarchical | 3069868 | 2799809 | 2799809 | 47.70% | 52.30% |
| main_lower_only_fixed_upper | 1178941 | 4687260 | 4687260 | 79.90% | 20.10% |
| main_upper_only | 2945739 | 2919112 | 2919112 | 49.77% | 50.23% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation_full | 1.0000 +/- 0.0000 | 0.4108 +/- 0.1087 | 0.4108 +/- 0.1087 | 0.0504 +/- 0.0471 | 0.0504 +/- 0.0471 | 0 |
| ablation_no_attention | 1.0000 +/- 0.0000 | 0.4171 +/- 0.1064 | 0.4171 +/- 0.1064 | 0.0483 +/- 0.0728 | 0.0483 +/- 0.0728 | 0 |
| ablation_no_lagrange | 1.0000 +/- 0.0000 | 0.3527 +/- 0.1099 | 0.3527 +/- 0.1099 | 0.3580 +/- 0.2240 | 0.3580 +/- 0.2240 | 0 |
| ablation_no_mask | 1.0000 +/- 0.0000 | 0.4037 +/- 0.1089 | 0.4037 +/- 0.1089 | 0.0003 +/- 0.0010 | 0.0003 +/- 0.0010 | 0 |
| dsr_priority_full | 1.0000 +/- 0.0000 | 0.3612 +/- 0.0790 | 0.3612 +/- 0.0790 | 0.2292 +/- 0.2557 | 0.2292 +/- 0.2557 | 0 |
| main_full_hierarchical | 1.0000 +/- 0.0000 | 0.3855 +/- 0.0673 | 0.3855 +/- 0.0673 | 0.0385 +/- 0.0543 | 0.0385 +/- 0.0543 | 0 |
| main_lower_only_fixed_upper | 1.0000 +/- 0.0000 | 0.4108 +/- 0.1087 | 0.4108 +/- 0.1087 | 0.0504 +/- 0.0471 | 0.0504 +/- 0.0471 | 0 |
| main_upper_only | 1.0000 +/- 0.0000 | 0.5073 +/- 0.0775 | 0.5073 +/- 0.0775 | 0.0250 +/- 0.0104 | 0.0250 +/- 0.0104 | 0 |

## Red-Flag Checks

- `ablation_full`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `ablation_no_attention`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `ablation_no_lagrange`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `ablation_no_mask`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `dsr_priority_full`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `main_full_hierarchical`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `main_lower_only_fixed_upper`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `main_upper_only`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
