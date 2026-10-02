# Baseline Matrix Audit Report

- Input directory: `results/baseline_matrix_force_admission_20260626_resume`
- Metrics files: 140
- Trajectory files: 0
- This report is an audit artifact and should not be treated as a thesis result table by itself.

## Service Coverage Chain

| Method | Units | Episodes | Generated | Processed | Unprocessed | Deadline satisfied | Processed / generated | DSR request-weighted | Satisfied / processed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ippo | 30 | 180 | 5792284 | 5792284 | 0 | 1609346 | 100.00% | 27.78% | 27.78% |
| joint_mappo | 30 | 180 | 5815659 | 5815659 | 0 | 2419162 | 100.00% | 41.60% | 41.60% |
| proposed | 30 | 180 | 5869677 | 5869677 | 0 | 2201799 | 100.00% | 37.51% | 37.51% |
| random | 10 | 60 | 1951289 | 1951289 | 0 | 740903 | 100.00% | 37.97% | 37.97% |
| uniform | 10 | 60 | 1954338 | 1954338 | 0 | 884373 | 100.00% | 45.25% | 45.25% |
| vanilla_mappo | 30 | 180 | 5813094 | 5813094 | 0 | 1670040 | 100.00% | 28.73% | 28.73% |

## Admission Audit

| Method | Naturally covered service requests | Uncovered service requests | Forced service admissions | Forced admission ratio | Natural coverage ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ippo | 3877953 | 1914331 | 1914331 | 33.05% | 66.95% |
| joint_mappo | 1265772 | 4549887 | 4549887 | 78.24% | 21.76% |
| proposed | 3069868 | 2799809 | 2799809 | 47.70% | 52.30% |
| random | 667686 | 1283603 | 1283603 | 65.78% | 34.22% |
| uniform | 839746 | 1114592 | 1114592 | 57.03% | 42.97% |
| vanilla_mappo | 3725278 | 2087816 | 2087816 | 35.92% | 64.08% |

## Ratio Field Audit

| Method | processed_request_ratio | dsr_request_weighted | deadline_satisfied_per_processed | offloading_ratio_mbs_processed | mbs_load_ratio_generated | MBS denominator mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ippo | 1.0000 +/- 0.0000 | 0.2818 +/- 0.0527 | 0.2818 +/- 0.0527 | 0.2940 +/- 0.1008 | 0.2940 +/- 0.1008 | 0 |
| joint_mappo | 1.0000 +/- 0.0000 | 0.4240 +/- 0.0892 | 0.4240 +/- 0.0892 | 0.2799 +/- 0.0583 | 0.2799 +/- 0.0583 | 0 |
| proposed | 1.0000 +/- 0.0000 | 0.3855 +/- 0.0673 | 0.3855 +/- 0.0673 | 0.0385 +/- 0.0543 | 0.0385 +/- 0.0543 | 0 |
| random | 1.0000 +/- 0.0000 | 0.3888 +/- 0.0648 | 0.3888 +/- 0.0648 | 0.3226 +/- 0.0179 | 0.3226 +/- 0.0179 | 0 |
| uniform | 1.0000 +/- 0.0000 | 0.4602 +/- 0.0712 | 0.4602 +/- 0.0712 | 0.1845 +/- 0.0142 | 0.1845 +/- 0.0142 | 0 |
| vanilla_mappo | 1.0000 +/- 0.0000 | 0.2941 +/- 0.0591 | 0.2941 +/- 0.0591 | 0.0868 +/- 0.0422 | 0.0868 +/- 0.0422 | 0 |

## Red-Flag Checks

- `ippo`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `joint_mappo`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `proposed`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `random`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `uniform`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.
- `vanilla_mappo`: all generated service requests processed; DSR is lower than processed ratio; MBS ratio denominators consistent.

## Trajectory Audit

- No trajectory JSON files were found under the input directory.
- Matrix metrics files do not contain UAV positions, so trajectory distance/boundary/separation checks are unavailable for this run.
