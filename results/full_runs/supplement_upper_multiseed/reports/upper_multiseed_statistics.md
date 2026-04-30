# Experiment Statistics

Reference: `uncoordinated_greedy`

## Metric Summary

| Policy | Metric | Mean | Std | CI Low | CI High | N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attention_mappo | reward | -1221.25 | 28.78 | -1267.04 | -1175.45 | 4 |
| attention_mappo | latency | 1141829.30 | 160282.03 | 886784.83 | 1396873.77 | 4 |
| attention_mappo | energy | 113841059.59 | 9592702.60 | 98576929.12 | 129105190.06 | 4 |
| attention_mappo | fairness | 0.9252 | 0.02767 | 0.8811 | 0.9692 | 4 |
| attention_mappo | offline_rate | 0.6814 | 0.08781 | 0.5417 | 0.8211 | 4 |
| attention_mappo | deadline_satisfaction_rate | 0.4302 | 0.08151 | 0.3005 | 0.5599 | 4 |
| attention_mappo | offloading_ratio_local | 0.3181 | 0.02212 | 0.2829 | 0.3533 | 4 |
| attention_mappo | offloading_ratio_cooperative | 0.468 | 0.02999 | 0.4203 | 0.5157 | 4 |
| attention_mappo | offloading_ratio_mbs | 0.2032 | 0.04805 | 0.1267 | 0.2796 | 4 |
| attention_mappo | mbs_load_ratio | 0.1089 | 0.03721 | 0.04968 | 0.1681 | 4 |
| attention_mappo | service_fallback_count | 0 | 0 | 0 | 0 | 4 |
| attention_mappo | service_predict_exception_fallback_count | 0 | 0 | 0 | 0 | 4 |
| attention_mappo | episode | 30.5 | 0 | 30.5 | 30.5 | 4 |
| attention_mappo | service_heuristic_decision_count | 12293.44 | 3957.19 | 5996.67 | 18590.21 | 4 |
| attention_mappo | service_learned_decision_count | 0 | 0 | 0 | 0 | 4 |
| attention_mappo | service_offload_policy_loaded | 0 | 0 | 0 | 0 | 4 |
| attention_mappo | time | 234.7 | 7.902 | 222.2 | 247.3 | 4 |
| uncoordinated_greedy | reward | -1445.17 | 21.03 | -1478.62 | -1411.71 | 4 |
| uncoordinated_greedy | latency | 1087948.78 | 29827.42 | 1040486.70 | 1135410.86 | 4 |
| uncoordinated_greedy | energy | 124741462.53 | 14370588.79 | 101874648.93 | 147608276.13 | 4 |
| uncoordinated_greedy | fairness | 0.7716 | 0.01339 | 0.7503 | 0.7929 | 4 |
| uncoordinated_greedy | offline_rate | 0.6637 | 0.03198 | 0.6128 | 0.7146 | 4 |
| uncoordinated_greedy | deadline_satisfaction_rate | 0.4459 | 0.01615 | 0.4202 | 0.4716 | 4 |
| uncoordinated_greedy | offloading_ratio_local | 0.3102 | 0.01516 | 0.286 | 0.3343 | 4 |
| uncoordinated_greedy | offloading_ratio_cooperative | 0.4922 | 0.0277 | 0.4481 | 0.5362 | 4 |
| uncoordinated_greedy | offloading_ratio_mbs | 0.1948 | 0.01734 | 0.1672 | 0.2224 | 4 |
| uncoordinated_greedy | mbs_load_ratio | 0.1051 | 0.008427 | 0.09173 | 0.1185 | 4 |
| uncoordinated_greedy | service_fallback_count | 0 | 0 | 0 | 0 | 4 |
| uncoordinated_greedy | service_predict_exception_fallback_count | 0 | 0 | 0 | 0 | 4 |
| uncoordinated_greedy | episode | 30.5 | 0 | 30.5 | 30.5 | 4 |
| uncoordinated_greedy | service_heuristic_decision_count | 12325.99 | 2065.25 | 9039.72 | 15612.27 | 4 |
| uncoordinated_greedy | service_learned_decision_count | 0 | 0 | 0 | 0 | 4 |
| uncoordinated_greedy | service_offload_policy_loaded | 0 | 0 | 0 | 0 | 4 |
| uncoordinated_greedy | time | 170.8 | 7.688 | 158.6 | 183 | 4 |

## Paired Comparison

| Policy | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| attention_mappo | reward | 223.9 | 198.2 | 249.7 | 0.0001036 | 0.125 | improves |
| attention_mappo | latency | 53880.52 | -191567.10 | 299328.15 | 0.5351 | 0.625 | regresses |
| attention_mappo | energy | -10900402.94 | -19988868.35 | -1811937.53 | 0.03164 | 0.125 | improves |
| attention_mappo | fairness | 0.1536 | 0.114 | 0.1931 | 0.001143 | 0.125 | improves |
| attention_mappo | offline_rate | 0.01771 | -0.07209 | 0.1075 | 0.5748 | 0.625 | regresses |
| attention_mappo | deadline_satisfaction_rate | -0.01572 | -0.147 | 0.1155 | 0.7285 | 0.625 | regresses |
| attention_mappo | offloading_ratio_local | 0.007949 | -0.03625 | 0.05215 | 0.6072 | 0.625 | improves |
| attention_mappo | offloading_ratio_cooperative | -0.02414 | -0.04209 | -0.006191 | 0.02343 | 0.125 | regresses |
| attention_mappo | offloading_ratio_mbs | 0.008332 | -0.04454 | 0.06121 | 0.6505 | 0.625 | regresses |
| attention_mappo | mbs_load_ratio | 0.003758 | -0.04681 | 0.05433 | 0.8283 | 1 | regresses |
| attention_mappo | service_fallback_count | 0 | 0 | 0 | NA | 1 | tie |
| attention_mappo | service_predict_exception_fallback_count | 0 | 0 | 0 | NA | 1 | tie |
| attention_mappo | episode | 0 | 0 | 0 | NA | 1 | tie |
| attention_mappo | service_heuristic_decision_count | -32.55 | -3203.48 | 3138.38 | 0.976 | 0.875 | regresses |
| attention_mappo | service_learned_decision_count | 0 | 0 | 0 | NA | 1 | tie |
| attention_mappo | service_offload_policy_loaded | 0 | 0 | 0 | NA | 1 | tie |
| attention_mappo | time | 63.95 | 54.12 | 73.78 | 0.0002462 | 0.125 | improves |
