# Baseline Matrix Audit Report

- Input directory: `results/baseline_matrix_force_admission_joint_smoke_20260627`
- Metrics files: 1
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_mappo | 1 | 1 | 39927 | 39927 | 0 | 15890 | 100.00% | 39.80% | 39.80% |

## Admission Audit

| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint_mappo | 10647 | 29280 | 29280 | 73.33% | 26.67% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_mappo | 1.0000 | 0.3980 | 0.3980 | 0.2345 | 0.2345 | 0 |

## Red-Flag Checks

- `joint_mappo`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
