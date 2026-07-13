# Baseline Matrix v2 Report

Generated at: 2026-06-28T13:52:36.262421

## Notes

- Learning: one unit=(training_seed, workload_seed). Non-learning: one unit=workload_seed.
- Global ratios are recomputed from per-episode request/offload counts within each unit.
- MBS offload ratio uses processed service requests as denominator.
- MBS load ratio uses all generated service requests as denominator.

## Method Summary

### ippo (learning, N=30)
- reward: -981.7591 +/- 235.9713 [95% CI -1069.87, -893.6460]
- latency: 600373.76 +/- 36525.26 [95% CI 586735.00, 614012.51]
- energy: 54761144.15 +/- 11692143.03 [95% CI 50395226.19, 59127062.11]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0002]
- fairness_final: 0.9027 +/- 0.0313 [95% CI 0.8911, 0.9144]
- fairness_step_mean: 0.8541 +/- 0.0361 [95% CI 0.8406, 0.8675]
- offline_rate_final: 0.0986 +/- 0.0360 [95% CI 0.0851, 0.1120]
- offline_rate_step_mean: 0.0129 +/- 0.0060 [95% CI 0.0106, 0.0151]
- dsr_request_weighted: 0.2780 +/- 0.0394 [95% CI 0.2633, 0.2928]
- offloading_ratio_local_processed: 0.3593 +/- 0.1404 [95% CI 0.3069, 0.4117]
- offloading_ratio_cooperative_processed: 0.3454 +/- 0.1519 [95% CI 0.2886, 0.4021]
- offloading_ratio_mbs_processed: 0.2954 +/- 0.0633 [95% CI 0.2717, 0.3190]
- mbs_load_ratio_generated: 0.2954 +/- 0.0633 [95% CI 0.2717, 0.3190]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.2780 +/- 0.0394 [95% CI 0.2633, 0.2928]

### vanilla_mappo (learning, N=30)
- reward: -925.3765 +/- 193.8360 [95% CI -997.7561, -852.9969]
- latency: 631211.55 +/- 42674.54 [95% CI 615276.61, 647146.49]
- energy: 55481883.77 +/- 13548872.99 [95% CI 50422651.45, 60541116.09]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0002]
- fairness_final: 0.9056 +/- 0.0273 [95% CI 0.8954, 0.9158]
- fairness_step_mean: 0.8585 +/- 0.0282 [95% CI 0.8479, 0.8690]
- offline_rate_final: 0.0772 +/- 0.0324 [95% CI 0.0651, 0.0893]
- offline_rate_step_mean: 0.0098 +/- 0.0052 [95% CI 0.0078, 0.0117]
- dsr_request_weighted: 0.2878 +/- 0.0433 [95% CI 0.2716, 0.3039]
- offloading_ratio_local_processed: 0.6547 +/- 0.1141 [95% CI 0.6121, 0.6973]
- offloading_ratio_cooperative_processed: 0.2632 +/- 0.1246 [95% CI 0.2167, 0.3097]
- offloading_ratio_mbs_processed: 0.0821 +/- 0.0294 [95% CI 0.0711, 0.0930]
- mbs_load_ratio_generated: 0.0821 +/- 0.0294 [95% CI 0.0711, 0.0930]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.2878 +/- 0.0433 [95% CI 0.2716, 0.3039]

### joint_mappo (learning, N=30)
- reward: -9081.23 +/- 1107.93 [95% CI -9494.94, -8667.53]
- latency: 1143569.90 +/- 83869.67 [95% CI 1112252.45, 1174887.35]
- energy: 96182401.65 +/- 12568922.57 [95% CI 91489088.84, 100875714.47]
- energy_efficiency_global: 0.0001 +/- 0.0000 [95% CI 0.0001, 0.0002]
- fairness_final: 0.7885 +/- 0.0547 [95% CI 0.7680, 0.8089]
- fairness_step_mean: 0.7557 +/- 0.0446 [95% CI 0.7390, 0.7724]
- offline_rate_final: 0.0379 +/- 0.0545 [95% CI 0.0176, 0.0583]
- offline_rate_step_mean: 0.0091 +/- 0.0165 [95% CI 0.0029, 0.0152]
- dsr_request_weighted: 0.4163 +/- 0.0400 [95% CI 0.4013, 0.4312]
- offloading_ratio_local_processed: 0.2023 +/- 0.0416 [95% CI 0.1868, 0.2179]
- offloading_ratio_cooperative_processed: 0.5205 +/- 0.0547 [95% CI 0.5000, 0.5409]
- offloading_ratio_mbs_processed: 0.2772 +/- 0.0317 [95% CI 0.2654, 0.2890]
- mbs_load_ratio_generated: 0.2772 +/- 0.0317 [95% CI 0.2654, 0.2890]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.4163 +/- 0.0400 [95% CI 0.4013, 0.4312]

### proposed (learning, N=30)
- reward: -1096.85 +/- 292.8011 [95% CI -1206.18, -987.5152]
- latency: 764148.39 +/- 53136.22 [95% CI 744307.00, 783989.79]
- energy: 49980270.24 +/- 4711969.16 [95% CI 48220792.04, 51739748.44]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0003]
- fairness_final: 0.9452 +/- 0.0139 [95% CI 0.9400, 0.9503]
- fairness_step_mean: 0.9009 +/- 0.0154 [95% CI 0.8952, 0.9067]
- offline_rate_final: 0.0225 +/- 0.0152 [95% CI 0.0168, 0.0282]
- offline_rate_step_mean: 0.0023 +/- 0.0019 [95% CI 0.0016, 0.0030]
- dsr_request_weighted: 0.3762 +/- 0.0283 [95% CI 0.3656, 0.3868]
- offloading_ratio_local_processed: 0.5560 +/- 0.1472 [95% CI 0.5010, 0.6110]
- offloading_ratio_cooperative_processed: 0.4081 +/- 0.1086 [95% CI 0.3676, 0.4487]
- offloading_ratio_mbs_processed: 0.0359 +/- 0.0419 [95% CI 0.0202, 0.0515]
- mbs_load_ratio_generated: 0.0359 +/- 0.0419 [95% CI 0.0202, 0.0515]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.3762 +/- 0.0283 [95% CI 0.3656, 0.3868]

### random (non_learning, N=10)
- reward: -1426.55 +/- 171.4077 [95% CI -1549.17, -1303.93]
- latency: 990382.05 +/- 44829.87 [95% CI 958312.70, 1022451.41]
- energy: 75014164.27 +/- 7535839.48 [95% CI 69623349.46, 80404979.08]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0001, 0.0002]
- fairness_final: 0.8906 +/- 0.0230 [95% CI 0.8741, 0.9070]
- fairness_step_mean: 0.8319 +/- 0.0430 [95% CI 0.8011, 0.8626]
- offline_rate_final: 0.0245 +/- 0.0377 [95% CI -0.0025, 0.0515]
- offline_rate_step_mean: 0.0038 +/- 0.0084 [95% CI -0.0022, 0.0098]
- dsr_request_weighted: 0.3802 +/- 0.0234 [95% CI 0.3635, 0.3970]
- offloading_ratio_local_processed: 0.3224 +/- 0.0068 [95% CI 0.3175, 0.3272]
- offloading_ratio_cooperative_processed: 0.3548 +/- 0.0134 [95% CI 0.3453, 0.3644]
- offloading_ratio_mbs_processed: 0.3228 +/- 0.0066 [95% CI 0.3181, 0.3275]
- mbs_load_ratio_generated: 0.3228 +/- 0.0066 [95% CI 0.3181, 0.3275]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.3802 +/- 0.0234 [95% CI 0.3635, 0.3970]

### uniform (non_learning, N=10)
- reward: -5320.14 +/- 98.2332 [95% CI -5390.41, -5249.86]
- latency: 846818.68 +/- 74552.73 [95% CI 793486.87, 900150.50]
- energy: 71357237.57 +/- 9561327.55 [95% CI 64517475.88, 78196999.26]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0002]
- fairness_final: 0.8603 +/- 0.0265 [95% CI 0.8413, 0.8793]
- fairness_step_mean: 0.8245 +/- 0.0238 [95% CI 0.8074, 0.8415]
- offline_rate_final: 0.0230 +/- 0.0176 [95% CI 0.0104, 0.0356]
- offline_rate_step_mean: 0.0028 +/- 0.0024 [95% CI 0.0011, 0.0045]
- dsr_request_weighted: 0.4529 +/- 0.0214 [95% CI 0.4376, 0.4682]
- offloading_ratio_local_processed: 0.2015 +/- 0.0057 [95% CI 0.1974, 0.2055]
- offloading_ratio_cooperative_processed: 0.6148 +/- 0.0108 [95% CI 0.6070, 0.6225]
- offloading_ratio_mbs_processed: 0.1837 +/- 0.0052 [95% CI 0.1801, 0.1874]
- mbs_load_ratio_generated: 0.1837 +/- 0.0052 [95% CI 0.1801, 0.1874]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.4529 +/- 0.0214 [95% CI 0.4376, 0.4682]

## Paired Comparisons

### proposed_vs_ippo (learning_paired)
- reward: delta=-115.0898, p_t=0.1899, p_wilcoxon=0.3599, N=30
- latency: delta=163774.64, p_t=0.0000, p_wilcoxon=0.0000, N=30
- energy: delta=-4780873.91, p_t=0.0907, p_wilcoxon=0.1772, N=30
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0000, N=30
- fairness_final: delta=0.0424, p_t=0.0000, p_wilcoxon=0.0000, N=30
- fairness_step_mean: delta=0.0468, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offline_rate_final: delta=-0.0761, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offline_rate_step_mean: delta=-0.0106, p_t=0.0000, p_wilcoxon=0.0000, N=30
- dsr_request_weighted: delta=0.0981, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offloading_ratio_local_processed: delta=0.1967, p_t=0.0002, p_wilcoxon=0.0004, N=30
- offloading_ratio_cooperative_processed: delta=0.0628, p_t=0.1219, p_wilcoxon=0.3599, N=30
- offloading_ratio_mbs_processed: delta=-0.2595, p_t=0.0000, p_wilcoxon=0.0000, N=30
- mbs_load_ratio_generated: delta=-0.2595, p_t=0.0000, p_wilcoxon=0.0000, N=30
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=NA, N=30
- deadline_satisfied_per_processed: delta=0.0981, p_t=0.0000, p_wilcoxon=0.0000, N=30

### proposed_vs_vanilla_mappo (learning_paired)
- reward: delta=-171.4724, p_t=0.0077, p_wilcoxon=0.0128, N=30
- latency: delta=132936.84, p_t=0.0000, p_wilcoxon=0.0000, N=30
- energy: delta=-5501613.54, p_t=0.0605, p_wilcoxon=0.1191, N=30
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0000, N=30
- fairness_final: delta=0.0395, p_t=0.0000, p_wilcoxon=0.0000, N=30
- fairness_step_mean: delta=0.0425, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offline_rate_final: delta=-0.0547, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offline_rate_step_mean: delta=-0.0075, p_t=0.0000, p_wilcoxon=0.0000, N=30
- dsr_request_weighted: delta=0.0884, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offloading_ratio_local_processed: delta=-0.0987, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offloading_ratio_cooperative_processed: delta=0.1449, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offloading_ratio_mbs_processed: delta=-0.0462, p_t=0.0004, p_wilcoxon=0.0001, N=30
- mbs_load_ratio_generated: delta=-0.0462, p_t=0.0004, p_wilcoxon=0.0001, N=30
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=NA, N=30
- deadline_satisfied_per_processed: delta=0.0884, p_t=0.0000, p_wilcoxon=0.0000, N=30

### proposed_vs_joint_mappo (learning_paired)
- reward: delta=7984.39, p_t=0.0000, p_wilcoxon=0.0000, N=30
- latency: delta=-379421.51, p_t=0.0000, p_wilcoxon=0.0000, N=30
- energy: delta=-46202131.42, p_t=0.0000, p_wilcoxon=0.0000, N=30
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0000, N=30
- fairness_final: delta=0.1567, p_t=0.0000, p_wilcoxon=0.0000, N=30
- fairness_step_mean: delta=0.1452, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offline_rate_final: delta=-0.0154, p_t=0.1251, p_wilcoxon=0.4490, N=30
- offline_rate_step_mean: delta=-0.0068, p_t=0.0299, p_wilcoxon=0.3469, N=30
- dsr_request_weighted: delta=-0.0401, p_t=0.0002, p_wilcoxon=0.0001, N=30
- offloading_ratio_local_processed: delta=0.3537, p_t=0.0000, p_wilcoxon=0.0000, N=30
- offloading_ratio_cooperative_processed: delta=-0.1123, p_t=0.0003, p_wilcoxon=0.0002, N=30
- offloading_ratio_mbs_processed: delta=-0.2413, p_t=0.0000, p_wilcoxon=0.0000, N=30
- mbs_load_ratio_generated: delta=-0.2413, p_t=0.0000, p_wilcoxon=0.0000, N=30
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=NA, N=30
- deadline_satisfied_per_processed: delta=-0.0401, p_t=0.0002, p_wilcoxon=0.0001, N=30

### proposed_vs_random (workload_paired)
- reward: delta=329.7036, p_t=0.0000, p_wilcoxon=0.0020, N=10
- latency: delta=-226233.66, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy: delta=-25033894.03, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_final: delta=0.0546, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_step_mean: delta=0.0691, p_t=0.0006, p_wilcoxon=0.0020, N=10
- offline_rate_final: delta=-0.0020, p_t=0.8588, p_wilcoxon=0.2324, N=10
- offline_rate_step_mean: delta=-0.0015, p_t=0.5738, p_wilcoxon=0.3223, N=10
- dsr_request_weighted: delta=-0.0040, p_t=0.6336, p_wilcoxon=0.4922, N=10
- offloading_ratio_local_processed: delta=0.2337, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_cooperative_processed: delta=0.0533, p_t=0.0008, p_wilcoxon=0.0059, N=10
- offloading_ratio_mbs_processed: delta=-0.2869, p_t=0.0000, p_wilcoxon=0.0020, N=10
- mbs_load_ratio_generated: delta=-0.2869, p_t=0.0000, p_wilcoxon=0.0020, N=10
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=1.0000, N=10
- deadline_satisfied_per_processed: delta=-0.0040, p_t=0.6336, p_wilcoxon=0.4922, N=10

### proposed_vs_uniform (workload_paired)
- reward: delta=4223.29, p_t=0.0000, p_wilcoxon=0.0020, N=10
- latency: delta=-82670.29, p_t=0.0386, p_wilcoxon=0.0371, N=10
- energy: delta=-21376967.33, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy_efficiency_global: delta=0.0000, p_t=0.0012, p_wilcoxon=0.0039, N=10
- fairness_final: delta=0.0848, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_step_mean: delta=0.0765, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offline_rate_final: delta=-0.0005, p_t=0.9239, p_wilcoxon=0.7500, N=10
- offline_rate_step_mean: delta=-0.0005, p_t=0.5041, p_wilcoxon=0.6953, N=10
- dsr_request_weighted: delta=-0.0767, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_local_processed: delta=0.3545, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_cooperative_processed: delta=-0.2067, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_mbs_processed: delta=-0.1479, p_t=0.0000, p_wilcoxon=0.0020, N=10
- mbs_load_ratio_generated: delta=-0.1479, p_t=0.0000, p_wilcoxon=0.0020, N=10
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=1.0000, N=10
- deadline_satisfied_per_processed: delta=-0.0767, p_t=0.0000, p_wilcoxon=0.0020, N=10
