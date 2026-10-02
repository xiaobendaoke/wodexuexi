# Baseline Matrix Audit Report

- Input directory: `results/baseline_matrix_audit_20260624_smoke`
- Metrics files: 3
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed | 1 | 2 | 70898 | 29217 | 41681 | 15173 | 41.21% | 21.40% | 51.93% |
| random | 1 | 2 | 70776 | 28885 | 41891 | 12231 | 40.81% | 17.28% | 42.34% |
| uniform | 1 | 2 | 71014 | 20199 | 50815 | 13613 | 28.44% | 19.17% | 67.39% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed | 0.4128 +/- 0.0057 | 0.2147 +/- 0.0055 | 0.5200 +/- 0.0062 | 0.2119 +/- 0.0120 | 0.0875 +/- 0.0061 | 0 |
| random | 0.4061 +/- 0.0161 | 0.1728 +/- 0.0002 | 0.4263 +/- 0.0174 | 0.3075 +/- 0.0083 | 0.1247 +/- 0.0016 | 0 |
| uniform | 0.2843 +/- 0.0013 | 0.1938 +/- 0.0165 | 0.6822 +/- 0.0612 | 0.1656 +/- 0.0080 | 0.0471 +/- 0.0025 | 0 |

## Red-Flag Checks

- `proposed`: has unprocessed service requests; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `random`: has unprocessed service requests; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `uniform`: has unprocessed service requests; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
