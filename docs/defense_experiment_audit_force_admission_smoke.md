# Baseline Matrix Audit Report

- Input directory: `results/baseline_matrix_force_admission_20260624_smoke`
- Metrics files: 3
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed | 1 | 2 | 69880 | 69880 | 0 | 15236 | 100.00% | 21.80% | 21.80% |
| random | 1 | 2 | 70893 | 70893 | 0 | 27912 | 100.00% | 39.37% | 39.37% |
| uniform | 1 | 2 | 71014 | 71014 | 0 | 36243 | 100.00% | 51.04% | 51.04% |

## Admission Audit

| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| proposed | 16531 | 53349 | 53349 | 76.34% | 23.66% |
| random | 24798 | 46095 | 46095 | 65.02% | 34.98% |
| uniform | 20199 | 50815 | 50815 | 71.56% | 28.44% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed | 1.0000 +/- 0.0000 | 0.2250 +/- 0.0603 | 0.2250 +/- 0.0603 | 0.3078 +/- 0.1149 | 0.3078 +/- 0.1149 | 0 |
| random | 1.0000 +/- 0.0000 | 0.4007 +/- 0.0547 | 0.4007 +/- 0.0547 | 0.3131 +/- 0.0208 | 0.3131 +/- 0.0208 | 0 |
| uniform | 1.0000 +/- 0.0000 | 0.5178 +/- 0.0568 | 0.5178 +/- 0.0568 | 0.1733 +/- 0.0011 | 0.1733 +/- 0.0011 | 0 |

## Red-Flag Checks

- `proposed`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `random`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `uniform`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
