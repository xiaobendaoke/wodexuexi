# Offloading Policy Validity Check

## Core Answers
1. Current top-line accuracy is still heavily driven by heuristic copying. The full surrogate reaches non-template macro_F1=0.385, and a latency-only diagnostic remains at macro_F1=0.369, while the current 5-field no-latency baseline drops to macro_F1=0.349. This means a large fraction of the performance comes from features that sit very close to the heuristic scoring rule itself.
2. The current classifier family does not collapse completely on programmatically generated non-template scenarios, but generalization quality depends on the feature set. The reduced-leakage rich-state candidate reaches macro_F1=0.901 on the held-out non-template split, which is meaningfully stronger than the current no-latency baseline and therefore shows some independent offline generalization value.
3. Under a fixed static trajectory policy, the surrogate classifier changes the offloading mix but hurts latency, energy, and deadline satisfaction relative to heuristic. The reduced-leakage candidate does not currently create a measurable end-to-end separation from heuristic in the reference environment, so the current evidence is still stronger for offline generalization than for system-level superiority.

## Template Held-Out Evaluation
- Train scenarios: local_cached_backhaul_bottleneck, cooperative_neighbor_cached, mbs_backhaul_advantage
- Test scenarios: local_compute_advantage, cooperative_queue_relief, mbs_large_job
- full_features: IID acc=1.0000, IID macro_F1=1.0000, held-out acc=0.9969, held-out macro_F1=0.9969
- latency_only: IID acc=0.9944, IID macro_F1=0.9945, held-out acc=0.6627, held-out macro_F1=0.5527
- no_latency_features: IID acc=0.9089, IID macro_F1=0.9089, held-out acc=0.3404, held-out macro_F1=0.1816
- rich_reduced_features: IID acc=1.0000, IID macro_F1=1.0000, held-out acc=0.7187, held-out macro_F1=0.7003

## Non-Template Evaluation
- surrogate_full: acc=0.4637, macro_F1=0.3846
- latency_only: acc=0.4577, macro_F1=0.3685
- no_latency_features: acc=0.4310, macro_F1=0.3491
- rich_reduced_template_only: acc=0.4950, macro_F1=0.4407
- rich_reduced_candidate: acc=0.9010, macro_F1=0.9012

## Fixed-Trajectory System Comparison
| Policy | Latency | Energy | Deadline Sat | Offload Local | Offload Coop | Offload MBS | MBS Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic_offloading | 1533.0483 | 400.0000 | 0.2308 | 0.0000 | 0.0000 | 0.9967 | 0.2308 |
| surrogate_classifier | 1542.5317 | 3250.0927 | 0.1427 | 0.5824 | 0.1954 | 0.2189 | 0.0528 |
| reduced_leakage_policy | 1533.0483 | 400.0000 | 0.2308 | 0.0000 | 0.0000 | 0.9967 | 0.2308 |

## Delta Vs Heuristic
- surrogate_classifier: latency=9.4834, energy=2850.0927, deadline_satisfaction_rate=-0.0881, offloading_ratio_local=0.5824, offloading_ratio_cooperative=0.1954, offloading_ratio_mbs=-0.7777, mbs_load_ratio=-0.1780
- reduced_leakage_policy: latency=0.0000, energy=0.0000, deadline_satisfaction_rate=0.0000, offloading_ratio_local=0.0000, offloading_ratio_cooperative=0.0000, offloading_ratio_mbs=0.0000, mbs_load_ratio=0.0000

## Final Judgment
- Surrogate baseline: heuristic surrogate baseline
- Reduced-leakage candidate: not yet sufficient for paper main claim
- Recommended paper baseline: full-feature classifier explicitly labeled as a heuristic surrogate / distillation baseline
- Recommended paper candidate: rich reduced-leakage policy trained on template plus non-template procedural raw-state data
- Remaining key issues: 1. The reduced-leakage policy still lacks a measurable end-to-end system win over heuristic under fixed trajectories, and in the current runtime evaluation it effectively collapses back to heuristic-like behavior. 2. There is still a deployment gap between the richer external validation features and the raw state officially exposed by the repository offloading module, so a paper main claim would benefit from promoting richer raw features into the official module rather than leaving them only in an external validator.