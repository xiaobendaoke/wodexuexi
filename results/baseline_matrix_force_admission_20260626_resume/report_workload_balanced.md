# Baseline Matrix v2 Report

Generated at: 2026-06-28T13:52:36.281195

## Notes

- Workload-balanced: one unit=one workload seed for every method. Learning methods are averaged over training seeds within each workload before statistics.
- Global ratios are recomputed from per-episode request/offload counts within each unit.
- MBS offload ratio uses processed service requests as denominator.
- MBS load ratio uses all generated service requests as denominator.

## Method Summary

### ippo (learning, N=10)
- reward: -981.7591 +/- 104.5792 [95% CI -1056.57, -906.9477]
- latency: 600373.76 +/- 23530.33 [95% CI 583541.18, 617206.34]
- energy: 54761144.15 +/- 2340568.27 [95% CI 53086802.48, 56435485.82]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0002]
- fairness_final: 0.9027 +/- 0.0271 [95% CI 0.8834, 0.9221]
- fairness_step_mean: 0.8541 +/- 0.0310 [95% CI 0.8319, 0.8762]
- offline_rate_final: 0.0986 +/- 0.0300 [95% CI 0.0771, 0.1200]
- offline_rate_step_mean: 0.0129 +/- 0.0049 [95% CI 0.0093, 0.0164]
- dsr_request_weighted: 0.2780 +/- 0.0079 [95% CI 0.2724, 0.2837]
- offloading_ratio_local_processed: 0.3593 +/- 0.0170 [95% CI 0.3471, 0.3714]
- offloading_ratio_cooperative_processed: 0.3454 +/- 0.0163 [95% CI 0.3337, 0.3570]
- offloading_ratio_mbs_processed: 0.2954 +/- 0.0194 [95% CI 0.2815, 0.3093]
- mbs_load_ratio_generated: 0.2954 +/- 0.0194 [95% CI 0.2815, 0.3093]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.2780 +/- 0.0079 [95% CI 0.2724, 0.2837]

### vanilla_mappo (learning, N=10)
- reward: -925.3765 +/- 100.3334 [95% CI -997.1507, -853.6023]
- latency: 631211.55 +/- 27105.95 [95% CI 611821.12, 650601.98]
- energy: 55481883.77 +/- 2600858.49 [95% CI 53621341.69, 57342425.85]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0002]
- fairness_final: 0.9056 +/- 0.0231 [95% CI 0.8891, 0.9221]
- fairness_step_mean: 0.8585 +/- 0.0237 [95% CI 0.8415, 0.8754]
- offline_rate_final: 0.0772 +/- 0.0264 [95% CI 0.0583, 0.0961]
- offline_rate_step_mean: 0.0098 +/- 0.0048 [95% CI 0.0063, 0.0132]
- dsr_request_weighted: 0.2878 +/- 0.0102 [95% CI 0.2804, 0.2951]
- offloading_ratio_local_processed: 0.6547 +/- 0.0175 [95% CI 0.6422, 0.6672]
- offloading_ratio_cooperative_processed: 0.2632 +/- 0.0148 [95% CI 0.2527, 0.2738]
- offloading_ratio_mbs_processed: 0.0821 +/- 0.0111 [95% CI 0.0741, 0.0900]
- mbs_load_ratio_generated: 0.0821 +/- 0.0111 [95% CI 0.0741, 0.0900]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.2878 +/- 0.0102 [95% CI 0.2804, 0.2951]

### joint_mappo (learning, N=10)
- reward: -9081.23 +/- 190.5405 [95% CI -9217.54, -8944.93]
- latency: 1143569.90 +/- 53367.55 [95% CI 1105393.06, 1181746.75]
- energy: 96182401.65 +/- 5729342.29 [95% CI 92083877.08, 100280926.23]
- energy_efficiency_global: 0.0001 +/- 0.0000 [95% CI 0.0001, 0.0002]
- fairness_final: 0.7885 +/- 0.0167 [95% CI 0.7765, 0.8004]
- fairness_step_mean: 0.7557 +/- 0.0170 [95% CI 0.7435, 0.7679]
- offline_rate_final: 0.0379 +/- 0.0297 [95% CI 0.0167, 0.0592]
- offline_rate_step_mean: 0.0091 +/- 0.0084 [95% CI 0.0031, 0.0150]
- dsr_request_weighted: 0.4163 +/- 0.0262 [95% CI 0.3976, 0.4350]
- offloading_ratio_local_processed: 0.2023 +/- 0.0086 [95% CI 0.1962, 0.2085]
- offloading_ratio_cooperative_processed: 0.5205 +/- 0.0187 [95% CI 0.5071, 0.5338]
- offloading_ratio_mbs_processed: 0.2772 +/- 0.0119 [95% CI 0.2687, 0.2857]
- mbs_load_ratio_generated: 0.2772 +/- 0.0119 [95% CI 0.2687, 0.2857]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.4163 +/- 0.0262 [95% CI 0.3976, 0.4350]

### proposed (learning, N=10)
- reward: -1096.85 +/- 114.3307 [95% CI -1178.64, -1015.06]
- latency: 764148.39 +/- 49063.74 [95% CI 729050.31, 799246.48]
- energy: 49980270.24 +/- 3165839.14 [95% CI 47715565.35, 52244975.13]
- energy_efficiency_global: 0.0002 +/- 0.0000 [95% CI 0.0002, 0.0003]
- fairness_final: 0.9452 +/- 0.0095 [95% CI 0.9383, 0.9520]
- fairness_step_mean: 0.9009 +/- 0.0120 [95% CI 0.8924, 0.9095]
- offline_rate_final: 0.0225 +/- 0.0098 [95% CI 0.0155, 0.0295]
- offline_rate_step_mean: 0.0023 +/- 0.0012 [95% CI 0.0014, 0.0032]
- dsr_request_weighted: 0.3762 +/- 0.0193 [95% CI 0.3623, 0.3900]
- offloading_ratio_local_processed: 0.5560 +/- 0.0236 [95% CI 0.5391, 0.5729]
- offloading_ratio_cooperative_processed: 0.4081 +/- 0.0257 [95% CI 0.3898, 0.4265]
- offloading_ratio_mbs_processed: 0.0359 +/- 0.0044 [95% CI 0.0327, 0.0390]
- mbs_load_ratio_generated: 0.0359 +/- 0.0044 [95% CI 0.0327, 0.0390]
- processed_request_ratio: 1.0000 +/- 0.0000 [95% CI 1.0000, 1.0000]
- deadline_satisfied_per_processed: 0.3762 +/- 0.0193 [95% CI 0.3623, 0.3900]

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
- reward: delta=-115.0898, p_t=0.0571, p_wilcoxon=0.1055, N=10
- latency: delta=163774.64, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy: delta=-4780873.91, p_t=0.0019, p_wilcoxon=0.0098, N=10
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_final: delta=0.0424, p_t=0.0002, p_wilcoxon=0.0020, N=10
- fairness_step_mean: delta=0.0468, p_t=0.0001, p_wilcoxon=0.0020, N=10
- offline_rate_final: delta=-0.0761, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offline_rate_step_mean: delta=-0.0106, p_t=0.0000, p_wilcoxon=0.0020, N=10
- dsr_request_weighted: delta=0.0981, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_local_processed: delta=0.1967, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_cooperative_processed: delta=0.0628, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_mbs_processed: delta=-0.2595, p_t=0.0000, p_wilcoxon=0.0020, N=10
- mbs_load_ratio_generated: delta=-0.2595, p_t=0.0000, p_wilcoxon=0.0020, N=10
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=1.0000, N=10
- deadline_satisfied_per_processed: delta=0.0981, p_t=0.0000, p_wilcoxon=0.0020, N=10

### proposed_vs_vanilla_mappo (learning_paired)
- reward: delta=-171.4724, p_t=0.0013, p_wilcoxon=0.0020, N=10
- latency: delta=132936.84, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy: delta=-5501613.54, p_t=0.0001, p_wilcoxon=0.0020, N=10
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_final: delta=0.0395, p_t=0.0001, p_wilcoxon=0.0020, N=10
- fairness_step_mean: delta=0.0425, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offline_rate_final: delta=-0.0547, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offline_rate_step_mean: delta=-0.0075, p_t=0.0003, p_wilcoxon=0.0020, N=10
- dsr_request_weighted: delta=0.0884, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_local_processed: delta=-0.0987, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_cooperative_processed: delta=0.1449, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_mbs_processed: delta=-0.0462, p_t=0.0000, p_wilcoxon=0.0020, N=10
- mbs_load_ratio_generated: delta=-0.0462, p_t=0.0000, p_wilcoxon=0.0020, N=10
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=1.0000, N=10
- deadline_satisfied_per_processed: delta=0.0884, p_t=0.0000, p_wilcoxon=0.0020, N=10

### proposed_vs_joint_mappo (learning_paired)
- reward: delta=7984.39, p_t=0.0000, p_wilcoxon=0.0020, N=10
- latency: delta=-379421.51, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy: delta=-46202131.42, p_t=0.0000, p_wilcoxon=0.0020, N=10
- energy_efficiency_global: delta=0.0001, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_final: delta=0.1567, p_t=0.0000, p_wilcoxon=0.0020, N=10
- fairness_step_mean: delta=0.1452, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offline_rate_final: delta=-0.0154, p_t=0.0596, p_wilcoxon=0.1055, N=10
- offline_rate_step_mean: delta=-0.0068, p_t=0.0182, p_wilcoxon=0.0840, N=10
- dsr_request_weighted: delta=-0.0401, p_t=0.0025, p_wilcoxon=0.0059, N=10
- offloading_ratio_local_processed: delta=0.3537, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_cooperative_processed: delta=-0.1123, p_t=0.0000, p_wilcoxon=0.0020, N=10
- offloading_ratio_mbs_processed: delta=-0.2413, p_t=0.0000, p_wilcoxon=0.0020, N=10
- mbs_load_ratio_generated: delta=-0.2413, p_t=0.0000, p_wilcoxon=0.0020, N=10
- processed_request_ratio: delta=0.0000, p_t=NA, p_wilcoxon=1.0000, N=10
- deadline_satisfied_per_processed: delta=-0.0401, p_t=0.0025, p_wilcoxon=0.0059, N=10

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
