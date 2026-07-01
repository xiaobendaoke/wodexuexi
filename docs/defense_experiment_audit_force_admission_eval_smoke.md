# Baseline Matrix Audit Report

- Input directory: `results/final_force_admission/evaluation_20260630_smoke`
- Metrics files: 1
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main_full_hierarchical | 1 | 1 | 39759 | 39759 | 0 | 15408 | 100.00% | 38.75% | 38.75% |

## Admission Audit

| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| main_full_hierarchical | 22045 | 17714 | 17714 | 44.55% | 55.45% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main_full_hierarchical | 1.0000 | 0.3875 | 0.3875 | 0.0196 | 0.0196 | 0 |

## Red-Flag Checks

- `main_full_hierarchical`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
