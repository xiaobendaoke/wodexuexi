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
| surrogate_baseline | latency | 1479.51 | 105 | 1312.38 | 1646.63 | 4 |
| surrogate_baseline | energy | 83448.54 | 10293.04 | 67070.01 | 99827.06 | 4 |
| surrogate_baseline | deadline_satisfaction_rate | 0.2228 | 0.04287 | 0.1546 | 0.291 | 4 |
| surrogate_baseline | offloading_ratio_local | 0.7695 | 0.06673 | 0.6633 | 0.8757 | 4 |
| surrogate_baseline | offloading_ratio_cooperative | 0.06663 | 0.01322 | 0.04559 | 0.08766 | 4 |
| surrogate_baseline | offloading_ratio_mbs | 0.1373 | 0.01579 | 0.1122 | 0.1625 | 4 |
| surrogate_baseline | mbs_load_ratio | 0.03336 | 0.001797 | 0.0305 | 0.03622 | 4 |
| rich_reduced_runtime_policy | latency | 1478.97 | 105.1 | 1311.67 | 1646.27 | 4 |
| rich_reduced_runtime_policy | energy | 64417.66 | 9992.98 | 48516.60 | 80318.72 | 4 |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | 0.2287 | 0.04277 | 0.1606 | 0.2968 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.6231 | 0.05074 | 0.5424 | 0.7039 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | 0.07246 | 0.01199 | 0.05339 | 0.09154 | 4 |
| rich_reduced_runtime_policy | offloading_ratio_mbs | 0.2779 | 0.003293 | 0.2726 | 0.2831 | 4 |
| rich_reduced_runtime_policy | mbs_load_ratio | 0.06787 | 0.003747 | 0.06191 | 0.07383 | 4 |

## Paired Comparison

| Policy | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| surrogate_baseline | latency | 1.955 | -0.135 | 4.046 | 0.05874 | 0.125 | regresses |
| surrogate_baseline | energy | -24359.73 | -45741.55 | -2977.91 | 0.0361 | 0.125 | improves |
| surrogate_baseline | deadline_satisfaction_rate | -0.0219 | -0.04409 | 0.0002851 | 0.0516 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_local | 0.3732 | 0.2538 | 0.4926 | 0.002161 | 0.125 | improves |
| surrogate_baseline | offloading_ratio_cooperative | -0.1971 | -0.266 | -0.1282 | 0.002804 | 0.125 | regresses |
| surrogate_baseline | offloading_ratio_mbs | -0.1762 | -0.2407 | -0.1116 | 0.00321 | 0.125 | improves |
| surrogate_baseline | mbs_load_ratio | -0.04717 | -0.06954 | -0.0248 | 0.00675 | 0.125 | improves |
| rich_reduced_runtime_policy | latency | 1.421 | -0.3324 | 3.175 | 0.08184 | 0.125 | regresses |
| rich_reduced_runtime_policy | energy | -43390.61 | -63780.86 | -23000.36 | 0.006579 | 0.125 | improves |
| rich_reduced_runtime_policy | deadline_satisfaction_rate | -0.01597 | -0.03467 | 0.002729 | 0.07267 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_local | 0.2268 | 0.1335 | 0.3201 | 0.004488 | 0.125 | improves |
| rich_reduced_runtime_policy | offloading_ratio_cooperative | -0.1912 | -0.2587 | -0.1238 | 0.002879 | 0.125 | regresses |
| rich_reduced_runtime_policy | offloading_ratio_mbs | -0.0356 | -0.07962 | 0.008416 | 0.08221 | 0.125 | improves |
| rich_reduced_runtime_policy | mbs_load_ratio | -0.01266 | -0.02934 | 0.004012 | 0.09444 | 0.125 | improves |
