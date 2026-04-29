# Experiment Statistics

Reference: `heuristic_offloading`

## Metric Summary

| Policy | Metric | Mean | Std | CI Low | CI High | N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| heuristic_offloading | latency | 1477.33 | 101.9 | 1315.17 | 1639.49 | 4 |
| heuristic_offloading | energy | 107305.72 | 16051.80 | 81763.73 | 132847.72 | 4 |
| heuristic_offloading | deadline_satisfaction_rate | 0.2464 | 0.04507 | 0.1747 | 0.3181 | 4 |
| heuristic_offloading | offloading_ratio_local | 0.4099 | 0.01019 | 0.3937 | 0.4261 | 4 |
| heuristic_offloading | offloading_ratio_cooperative | 0.2293 | 0.02346 | 0.192 | 0.2666 | 4 |
| heuristic_offloading | offloading_ratio_mbs | 0.3246 | 0.02673 | 0.282 | 0.3671 | 4 |
| heuristic_offloading | mbs_load_ratio | 0.08408 | 0.01216 | 0.06474 | 0.1034 | 4 |
| surrogate_baseline | latency | 1477.64 | 101.8 | 1315.60 | 1639.67 | 4 |
| surrogate_baseline | energy | 125190.64 | 18715.31 | 95410.40 | 154970.88 | 4 |
| surrogate_baseline | deadline_satisfaction_rate | 0.2447 | 0.0444 | 0.174 | 0.3153 | 4 |
| surrogate_baseline | offloading_ratio_local | 0.5196 | 0.008182 | 0.5065 | 0.5326 | 4 |
| surrogate_baseline | offloading_ratio_cooperative | 0.2629 | 0.02324 | 0.2259 | 0.2999 | 4 |
| surrogate_baseline | offloading_ratio_mbs | 0.1813 | 0.0177 | 0.1531 | 0.2094 | 4 |
| surrogate_baseline | mbs_load_ratio | 0.04879 | 0.008712 | 0.03493 | 0.06266 | 4 |
| rich_reduced_runtime_policy | latency | 1478.12 | 101.4 | 1316.76 | 1639.49 | 4 |
| rich_reduced_runtime_policy | energy | 126569.00 | 18020.94 | 97893.66 | 155244.34 | 4 |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | 0.2392 | 0.04122 | 0.1736 | 0.3048 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.5112 | 0.004689 | 0.5038 | 0.5187 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | 0.2979 | 0.03643 | 0.2399 | 0.3558 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_mbs | 0.1547 | 0.005958 | 0.1452 | 0.1641 | 4 |
| rich_reduced_runtime_policy | mbs_load_ratio | 0.03892 | 0.002386 | 0.03513 | 0.04272 | 4 |

## Paired Comparison

| Policy | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| surrogate_baseline | latency | 0.3022 | 0.1509 | 0.4536 | 0.007886 | 0.125 | regresses |
| surrogate_baseline | energy | 17884.92 | 13347.18 | 22422.66 | 0.001092 | 0.125 | regresses |
| surrogate_baseline | deadline_satisfaction_rate | -0.001727 | -0.003522 | 6.742e-05 | 0.05487 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_local | 0.1097 | 0.09385 | 0.1255 | 0.000204 | 0.125 | improves |
| surrogate_baseline | offloading_ratio_cooperative | 0.03364 | 0.02895 | 0.03833 | 0.0001839 | 0.125 | improves |
| surrogate_baseline | offloading_ratio_mbs | -0.1433 | -0.1586 | -0.128 | 8.336e-05 | 0.125 | improves |
| surrogate_baseline | mbs_load_ratio | -0.03529 | -0.041 | -0.02958 | 0.0002871 | 0.125 | improves |
| rich_reduced_runtime_policy | latency | 0.7876 | -0.3312 | 1.906 | 0.1109 | 0.125 | regresses |
| rich_reduced_runtime_policy | energy | 19263.28 | 15797.01 | 22729.54 | 0.0003941 | 0.125 | regresses |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | -0.007141 | -0.0185 | 0.004222 | 0.1393 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.1013 | 0.0918 | 0.1109 | 5.695e-05 | 0.125 | improves |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | 0.06858 | 0.03908 | 0.09807 | 0.005106 | 0.125 | improves |
| rich_reduced_runtime_policy | offloading_ratio_mbs | -0.1699 | -0.2053 | -0.1345 | 0.0006095 | 0.125 | improves |
| rich_reduced_runtime_policy | mbs_load_ratio | -0.04516 | -0.06109 | -0.02923 | 0.002877 | 0.125 | improves |
