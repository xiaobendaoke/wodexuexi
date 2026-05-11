# Offloading Policy Validity Check

## Core Answers
1. Current top-line accuracy is still heavily driven by heuristic copying. The full surrogate reaches non-template macro_F1=0.241, and a latency-only diagnostic remains at macro_F1=0.285, while the current 5-field no-latency baseline drops to macro_F1=0.375. This means a large fraction of the performance comes from features that sit very close to the heuristic scoring rule itself.
2. The current classifier family does not collapse completely on programmatically generated non-template scenarios, but generalization quality depends on the feature set. The reduced-leakage rich-state candidate reaches macro_F1=0.883 on the held-out non-template split, which is meaningfully stronger than the current no-latency baseline and therefore shows some independent offline generalization value.
3. Under a fixed static trajectory policy, the surrogate classifier changes the offloading mix but hurts latency, energy, and deadline satisfaction relative to heuristic. The reduced-leakage candidate does not currently create a measurable end-to-end separation from heuristic in the reference environment, so the current evidence is still stronger for offline generalization than for system-level superiority.

## Template Held-Out Evaluation
- Train scenarios: local_cached_backhaul_bottleneck, cooperative_neighbor_cached, mbs_backhaul_advantage
- Test scenarios: local_compute_advantage, cooperative_queue_relief, mbs_large_job
- full_features: IID acc=1.0000, IID macro_F1=1.0000, held-out acc=0.8738, held-out macro_F1=0.8692
- latency_only: IID acc=0.9956, IID macro_F1=0.9956, held-out acc=0.6618, held-out macro_F1=0.5524
- no_latency_features: IID acc=0.8644, IID macro_F1=0.8643, held-out acc=0.8764, held-out macro_F1=0.8773
- rich_reduced_features: IID acc=1.0000, IID macro_F1=1.0000, held-out acc=0.9991, held-out macro_F1=0.9991

## Non-Template Evaluation
- surrogate_full: acc=0.3723, macro_F1=0.2412
- latency_only: acc=0.3563, macro_F1=0.2850
- no_latency_features: acc=0.3937, macro_F1=0.3746
- rich_reduced_template_only: acc=0.4017, macro_F1=0.3540
- rich_reduced_candidate: acc=0.8830, macro_F1=0.8830

## Fixed-Trajectory System Comparison
| Policy | Latency | Energy | Deadline Sat | Offload Local | Offload Coop | Offload MBS | MBS Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic_offloading | 1434.8233 | 72047.1708 | 0.2846 | 0.4081 | 0.2903 | 0.2483 | 0.1093 |
| surrogate_classifier | 1437.1857 | 68462.1687 | 0.2484 | 0.3359 | 0.6096 | 0.0011 | 0.0008 |
| reduced_leakage_policy | 1435.1210 | 60123.8516 | 0.2813 | 0.4340 | 0.2163 | 0.2964 | 0.1069 |

## Delta Vs Heuristic
- surrogate_classifier: latency=2.3624, energy=-3585.0021, deadline_satisfaction_rate=-0.0362, offloading_ratio_local=-0.0721, offloading_ratio_cooperative=0.3194, offloading_ratio_mbs=-0.2472, mbs_load_ratio=-0.1084
- reduced_leakage_policy: latency=0.2977, energy=-11923.3192, deadline_satisfaction_rate=-0.0033, offloading_ratio_local=0.0259, offloading_ratio_cooperative=-0.0740, offloading_ratio_mbs=0.0481, mbs_load_ratio=-0.0023

## Final Judgment
- Surrogate baseline: heuristic surrogate baseline
- Reduced-leakage candidate: partially generalizable learned policy
- Recommended paper baseline: full-feature classifier explicitly labeled as a heuristic surrogate / distillation baseline
- Recommended paper candidate: rich reduced-leakage policy trained on template plus non-template procedural raw-state data
- Remaining key issues: 1. The reduced-leakage policy still lacks a measurable end-to-end system win over heuristic under fixed trajectories, and in the current runtime evaluation it effectively collapses back to heuristic-like behavior. 2. There is still a deployment gap between the richer external validation features and the raw state officially exposed by the repository offloading module, so a paper main claim would benefit from promoting richer raw features into the official module rather than leaving them only in an external validator.