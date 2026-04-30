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
| surrogate_baseline | latency | 1479.53 | 105 | 1312.43 | 1646.64 | 4 |
| surrogate_baseline | energy | 84127.90 | 10164.45 | 67953.99 | 100301.80 | 4 |
| surrogate_baseline | deadline_satisfaction_rate | 0.2227 | 0.04289 | 0.1545 | 0.2909 | 4 |
| surrogate_baseline | offloading_ratio_local | 0.7715 | 0.06607 | 0.6664 | 0.8766 | 4 |
| surrogate_baseline | offloading_ratio_cooperative | 0.07539 | 0.01617 | 0.04966 | 0.1011 | 4 |
| surrogate_baseline | offloading_ratio_mbs | 0.1266 | 0.01295 | 0.106 | 0.1472 | 4 |
| surrogate_baseline | mbs_load_ratio | 0.03097 | 0.002015 | 0.02776 | 0.03417 | 4 |
| rich_reduced_runtime_policy | latency | 1478.54 | 105.5 | 1310.66 | 1646.42 | 4 |
| rich_reduced_runtime_policy | energy | 107440.22 | 21306.29 | 73537.16 | 141343.28 | 4 |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | 0.2344 | 0.04535 | 0.1622 | 0.3066 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.6637 | 0.02785 | 0.6194 | 0.708 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | 0.1373 | 0.01428 | 0.1145 | 0.16 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_mbs | 0.1725 | 0.007268 | 0.161 | 0.1841 | 4 |
| rich_reduced_runtime_policy | mbs_load_ratio | 0.04551 | 0.003473 | 0.03999 | 0.05104 | 4 |

## Paired Comparison

| Policy | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| surrogate_baseline | latency | 1.984 | -0.1818 | 4.15 | 0.06173 | 0.125 | regresses |
| surrogate_baseline | energy | -23680.37 | -45129.20 | -2231.54 | 0.0391 | 0.125 | improves |
| surrogate_baseline | deadline_satisfaction_rate | -0.02197 | -0.04458 | 0.0006438 | 0.05364 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_local | 0.3752 | 0.2562 | 0.4942 | 0.002107 | 0.125 | improves |
| surrogate_baseline | offloading_ratio_cooperative | -0.1883 | -0.2605 | -0.1161 | 0.003665 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_mbs | -0.1869 | -0.2463 | -0.1275 | 0.002119 | 0.125 | improves |
| surrogate_baseline | mbs_load_ratio | -0.04957 | -0.07198 | -0.02716 | 0.005889 | 0.125 | improves |
| rich_reduced_runtime_policy | latency | 0.9895 | 0.07487 | 1.904 | 0.04115 | 0.125 | regresses |
| rich_reduced_runtime_policy | energy | -368 | -4928.25 | 4192.16 | 0.8139 | 0.875 | improves |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | -0.01028 | -0.02058 | 3.135e-05 | 0.05037 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.2674 | 0.2089 | 0.3259 | 0.0007038 | 0.125 | improves |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | -0.1264 | -0.1563 | -0.09657 | 0.000885 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_mbs | -0.141 | -0.1864 | -0.09553 | 0.002208 | 0.125 | improves |
| rich_reduced_runtime_policy | mbs_load_ratio | -0.03502 | -0.05314 | -0.0169 | 0.008644 | 0.125 | improves |
