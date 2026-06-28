# Baseline Matrix Audit Report

- Input directory: `results/baseline_matrix_force_admission_20260624`
- Metrics files: 60
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ippo | 30 | 180 | 5792284 | 5792284 | 0 | 1609346 | 100.00% | 27.78% | 27.78% |
| vanilla_mappo | 30 | 180 | 5813094 | 5813094 | 0 | 1670040 | 100.00% | 28.73% | 28.73% |

## Admission Audit

| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ippo | 3877953 | 1914331 | 1914331 | 33.05% | 66.95% |
| vanilla_mappo | 3725278 | 2087816 | 2087816 | 35.92% | 64.08% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ippo | 1.0000 +/- 0.0000 | 0.2818 +/- 0.0527 | 0.2818 +/- 0.0527 | 0.2940 +/- 0.1008 | 0.2940 +/- 0.1008 | 0 |
| vanilla_mappo | 1.0000 +/- 0.0000 | 0.2941 +/- 0.0591 | 0.2941 +/- 0.0591 | 0.0868 +/- 0.0422 | 0.0868 +/- 0.0422 | 0 |

## Red-Flag Checks

- `ippo`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `vanilla_mappo`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
