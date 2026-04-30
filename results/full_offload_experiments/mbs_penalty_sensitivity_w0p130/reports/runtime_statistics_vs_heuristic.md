# Experiment Statistics

Reference: `heuristic_offloading`

## Metric Summary

| Policy | Metric | Mean | Std | CI Low | CI High | N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| heuristic_offloading | latency | 1477.55 | 105.9 | 1309.05 | 1646.05 | 4 |
| heuristic_offloading | energy | 107808.27 | 21021.93 | 74357.68 | 141258.85 | 4 |
| heuristic_offloading | deadline_satisfaction_rate | 0.2447 | 0.04868 | 0.1672 | 0.3221 | 4 |
| heuristic_offloading | offloading_ratio_local | 0.3963 | 0.01915 | 0.3658 | 0.4268 | 4 |
| heuristic_offloading | offloading_ratio_cooperative | 0.2637 | 0.03183 | 0.213 | 0.3144 | 4 |
| heuristic_offloading | offloading_ratio_mbs | 0.3135 | 0.02583 | 0.2724 | 0.3546 | 4 |
| heuristic_offloading | mbs_load_ratio | 0.08053 | 0.01317 | 0.05958 | 0.1015 | 4 |
| surrogate_baseline | latency | 1479.58 | 105 | 1312.48 | 1646.68 | 4 |
| surrogate_baseline | energy | 87235.42 | 9631.29 | 71909.89 | 102560.94 | 4 |
| surrogate_baseline | deadline_satisfaction_rate | 0.2226 | 0.0427 | 0.1547 | 0.2906 | 4 |
| surrogate_baseline | offloading_ratio_local | 0.7879 | 0.0647 | 0.6849 | 0.8908 | 4 |
| surrogate_baseline | offloading_ratio_cooperative | 0.08629 | 0.01754 | 0.05839 | 0.1142 | 4 |
| surrogate_baseline | offloading_ratio_mbs | 0.09932 | 0.01032 | 0.0829 | 0.1157 | 4 |
| surrogate_baseline | mbs_load_ratio | 0.02465 | 0.001501 | 0.02226 | 0.02703 | 4 |
| rich_reduced_runtime_policy | latency | 1478.82 | 105.4 | 1311.08 | 1646.56 | 4 |
| rich_reduced_runtime_policy | energy | 109801.13 | 19430.11 | 78883.48 | 140718.77 | 4 |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | 0.2305 | 0.04361 | 0.1612 | 0.2999 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.7251 | 0.02422 | 0.6866 | 0.7636 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | 0.1487 | 0.01683 | 0.1219 | 0.1755 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_mbs | 0.09969 | 0.002585 | 0.09558 | 0.1038 | 4 |
| rich_reduced_runtime_policy | mbs_load_ratio | 0.02806 | 0.0039 | 0.02185 | 0.03426 | 4 |

## Paired Comparison

| Policy | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| surrogate_baseline | latency | 2.029 | -0.1112 | 4.169 | 0.05689 | 0.125 | regresses |
| surrogate_baseline | energy | -20572.85 | -42667.46 | 1521.76 | 0.05939 | 0.125 | improves |
| surrogate_baseline | deadline_satisfaction_rate | -0.02207 | -0.04445 | 0.0003122 | 0.05174 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_local | 0.3916 | 0.2751 | 0.508 | 0.001745 | 0.125 | improves |
| surrogate_baseline | offloading_ratio_cooperative | -0.1774 | -0.2518 | -0.103 | 0.004743 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_mbs | -0.2142 | -0.2681 | -0.1602 | 0.00107 | 0.125 | improves |
| surrogate_baseline | mbs_load_ratio | -0.05589 | -0.07706 | -0.03472 | 0.003537 | 0.125 | improves |
| rich_reduced_runtime_policy | latency | 1.273 | 0.262 | 2.284 | 0.02788 | 0.125 | regresses |
| rich_reduced_runtime_policy | energy | 1992.86 | -1502.22 | 5487.94 | 0.1672 | 0.125 | regresses |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | -0.01413 | -0.02663 | -0.001633 | 0.03681 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.3288 | 0.2811 | 0.3765 | 0.0002078 | 0.125 | improves |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | -0.115 | -0.1425 | -0.08752 | 0.0009149 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_mbs | -0.2138 | -0.2542 | -0.1734 | 0.0004561 | 0.125 | improves |
| rich_reduced_runtime_policy | mbs_load_ratio | -0.05248 | -0.0696 | -0.03536 | 0.002289 | 0.125 | improves |
