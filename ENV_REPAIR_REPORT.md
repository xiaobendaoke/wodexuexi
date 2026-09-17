# Canonical Environment GREEN Repair Report (Phase 2B)

- **Execution Mode**: Dual-Machine Architecture (Local macOS Darwin editing + Remote Linux x86_64 verification)
- **Status**: `STATUS = ENV_GREEN_REPAIR_COMPLETE`
- **Semantics Version**: `ENV_SEMANTICS_VERSION = clean_v1`
- **Remote Host**: `100.69.44.85:22` (CentOS 7, Conda `drone` PyTorch 2.6.0+cu124)
- **Git Commit SHA**: `33796a02b2f55d0e5887f7973d782a0cd3b2f808`
- **Branch**: `双层注意力MAPPO多智能体强化学习`

---

## 1. Executive Summary

In Phase 2B, all canonical environment physical semantics specified in the frozen `CANONICAL_SEMANTICS_SPEC.md` were implemented and verified on the remote experiment server.

- **Total Test Suites**: 12
- **Total Individual Tests**: 26
- **Environment Tests**: 21 / 21 **GREEN** (100% passed)
- **Lower Policy Contract Tests (Phase 3 Reserved)**: 3 **RED** (strictly preserved, untouched)
- **Total Passed**: 23 / 26
- **Runtime Errors**: 0
- **Deterministic Smoke Test**: **PASS** (100% bitwise equivalence across 2 independent episodes)
- **Model Training**: Strictly prohibited & zero training conducted.

---

## 2. Modified Production & Test Files

| File | Change Category | Description |
| :--- | :--- | :--- |
| `config.py` | Config Deprecation | Hard-locked `FORCE_SERVICE_ADMISSION = False` (ENV-R2). Marked legacy flag deprecated. |
| `environment/uavs.py` | Core Physics | Added dual-state cache (`cache_snapshot`, `pending_cache`, `commit_pending_cache`), two-phase batch processor sharing (Phase A offload target resolution + Phase B equal processor sharing without sequential C2 decrement), component energy instrumentation, and preserved slot-t energy inspection after step. |
| `environment/env.py` | Core Physics & Lifecycle | Implemented `SlotSnapshot` immutable frozen dataclass (ENV-R1), rejection sampling initial separation $\ge 200$m (ENV-R7), natural admission only (ENV-R2), alternating projection collision & speed ball resolution (ENV-R8), canonical temporal order S0–S7 (ENV-R6), and preserved slot-t snapshot upon step return. |
| `tests/test_collision_joint_feasibility.py` | Test Fixture | Corrected fixture placement for non-colliding UAVs to ensure initial separation $\ge 200$m. |
| `tests/test_order_permutation_invariance.py` | Test Fixture | Added RNG re-seeding prior to `env2` initialization to isolate permutation effects under identical initial geometry. |
| `tests/test_env_temporal_alignment.py` | Test Assertion | Strengthened terminal-step energy assertion across fly vs. hover. |
| `tests/test_service_snapshot.py` | Test Assertion | Added functional invariance test comparing slot-start rates with post-step geometry. |
| `tests/deterministic_smoke.py` | Verification Script | Added 2-episode deterministic smoke verification script testing bitwise reproducibility. |

---

## 3. Specification Implementation & Before/After Comparison

### ENV-R1: Slot-Start Service Snapshot
- **Before**: Environment maintained mutable, in-place state. As UAVs moved or cache updated, transmission rates and cache hit decisions were evaluated against changing geometry, causing causal leaks.
- **After**: Implemented frozen immutable dataclass `SlotSnapshot` captured at S0 of each slot. All transmission channel gains, UE-UAV rates, UAV-UAV rates, UAV-MBS rates, and cache hit checks evaluate strictly against this frozen snapshot.
- **Tests**: `test_service_snapshot.py` (3 tests: `test_canonical_slot_snapshot_structure_exists`, `test_service_channel_rates_match_slot_start_geometry`, `test_service_snapshot_functional_invariance`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R2: Natural Admission Only (Forced Admission Elimination)
- **Before**: UEs outside all UAV coverage radii were forcefully reassigned to the nearest UAV, distorting coverage physics and inflating perceived UAV load.
- **After**: `_associate_ues_to_uavs()` strictly admits UEs if and only if $d(ue, uav) \le R_c$ (100m). Uncovered UEs remain unadmitted (`assigned = False`), receive non-served penalty, and are accounted for in system DSR.
- **Tests**: `test_service_admission.py` (3 tests: `test_out_of_coverage_ue_not_assigned`, `test_in_coverage_ue_naturally_admitted`, `test_forced_admission_canonical_is_false`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R3: Batch Processor Sharing
- **Before**: Local requests were processed sequentially with $C2$ decrements in an arbitrary loop order, causing later requests to receive degraded compute shares and permutation variance.
- **After**: Two-phase execution: Phase A resolves offload targets and identifies local requests ($K_{\text{local}}$); Phase B computes $f_{\text{share}} = F_{\text{UAV}} / K_{\text{local}}$ uniformly for all local requests. MBS offloaded requests are excluded from UAV local load.
- **Tests**: `test_batch_processor_sharing.py` (2 tests: `test_equal_sharing_for_all_tasks_on_same_uav`, `test_mbs_tasks_excluded_from_uav_final_load`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R4: Full Permutation Invariance
- **Before**: Output latencies and energies varied depending on the index ordering of requests in the input array.
- **After**: With Phase A/B batch processing and immutable slot-start snapshots, processing is strictly invariant to input permutation.
- **Tests**: `test_order_permutation_invariance.py` (`test_request_order_permutation_invariance`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R5: Cache Lifecycle & Dual-State Invariance
- **Before**: In-slot cache fetches were immediately written into `uav.cache`, allowing intra-slot instant hits for subsequent requests and causing cross-UAV cache wiping.
- **After**: Implemented dual-state cache architecture: reads evaluate strictly against `cache_snapshot`; writes queue into `pending_cache`; commits occur deterministically at end-of-slot S7 via `commit_pending_cache()` ordered by EMA/size descending.
- **Tests**: `test_cache_lifecycle.py` (3 tests: `test_same_slot_cache_fetch_not_instant_hit`, `test_cross_uav_cache_cooperation_not_wiped`, `test_cache_commit_permutation_invariance`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R6: Temporal Alignment
- **Before**: `step()` calculated motion energy at the beginning using `uav._dist_moved` from the previous slot (initially 0.0), causing step 0 to record zero flight energy and terminal step energy to be dropped.
- **After**: Reordered `step()` execution to S0–S7:
  - S0: Freeze slot-start snapshot & reset slot energy accumulators;
  - S1: Apply upper trajectory actions & alternating projection repair;
  - S2: Lower service offloading & execution on frozen geometry;
  - S3: WPT execution;
  - S4: Content caching execution;
  - S5: Compute slot rewards and metrics (accounting for current action's flight/hover energy);
  - S6: Record metrics and audit;
  - S7: Commit cache updates, advance UE mobility, generate next observations.
- **Tests**: `test_env_temporal_alignment.py` (3 tests: `test_first_step_flight_energy_accounted`, `test_terminal_step_energy_not_lost`, `test_zero_vs_max_movement_energy_separation_at_step_0`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R7: Pairwise Initial Separation
- **Before**: `env.reset()` placed UAVs randomly without guaranteeing minimum separation, leading to initial collisions.
- **After**: Bounded rejection sampling placed UAVs with pairwise distance $\ge d_{\min}$ (200m).
- **Tests**: `test_collision_joint_feasibility.py` (`test_reset_positions_are_pairwise_separated`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R8: Collision Resolution & Speed Ball Joint Feasibility
- **Before**: Collision repulsion pushed UAVs apart without re-projecting onto the reachability ball, causing $\text{dist\_moved} > v_{\max}\Delta t$ and negative hover times.
- **After**: Iterative alternating projection alternately resolves pairwise separation and projects back onto $B(p_t, v_{\max}\Delta t)$ and boundary constraints $[d_{\text{bound}}, W-d_{\text{bound}}]$.
- **Tests**: `test_collision_joint_feasibility.py` (`test_head_on_collision_joint_feasibility`).
- **Status**: RED $\to$ **GREEN**.

### ENV-R9: Metric Semantics
- **Before**: DSR denominator excluded unadmitted requests; critical UE ratio was conflated with dead UE ratio; UAV fleet total energy included MBS grid energy.
- **After**: DSR denominator strictly equals all service requests generated by UEs in the system; critical UE ratio equals proportion with battery $< 50$J; UAV fleet energy accounts only for UAV battery consumption.
- **Tests**: `test_metric_semantics.py` (3 tests: `test_system_dsr_denominator_includes_unadmitted_requests`, `test_critical_ue_ratio_definition`, `test_uav_fleet_total_energy_excludes_mbs_energy`).
- **Status**: RED $\to$ **GREEN**.

---

## 4. Test Suite Execution Matrix (Remote Server: 100.69.44.85)

| Test Module | Test Method | Target Component | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `test_env_temporal_alignment` | `test_first_step_flight_energy_accounted` | ENV-R6 | **GREEN** | Flight energy accounted in step 0 |
| `test_env_temporal_alignment` | `test_terminal_step_energy_not_lost` | ENV-R6 | **GREEN** | Terminal step energy captured |
| `test_env_temporal_alignment` | `test_zero_vs_max_movement_energy_separation_at_step_0` | ENV-R6 | **GREEN** | Fly vs hover energy separation |
| `test_service_snapshot` | `test_canonical_slot_snapshot_structure_exists` | ENV-R1 | **GREEN** | SlotSnapshot dataclass verified |
| `test_service_snapshot` | `test_service_channel_rates_match_slot_start_geometry` | ENV-R1 | **GREEN** | Rates match slot-start geometry |
| `test_service_snapshot` | `test_service_snapshot_functional_invariance` | ENV-R1 | **GREEN** | Rates invariant to post-step movement |
| `test_service_admission` | `test_out_of_coverage_ue_not_assigned` | ENV-R2 | **GREEN** | Out-of-coverage UEs unadmitted |
| `test_service_admission` | `test_in_coverage_ue_naturally_admitted` | ENV-R2 | **GREEN** | In-coverage UEs naturally admitted |
| `test_service_admission` | `test_forced_admission_canonical_is_false` | ENV-R2 | **GREEN** | Config flag locked False |
| `test_batch_processor_sharing` | `test_equal_sharing_for_all_tasks_on_same_uav` | ENV-R3 | **GREEN** | Equal compute share verified |
| `test_batch_processor_sharing` | `test_mbs_tasks_excluded_from_uav_final_load` | ENV-R3 | **GREEN** | MBS tasks excluded from UAV CPU load |
| `test_order_permutation_invariance` | `test_request_order_permutation_invariance` | ENV-R4 | **GREEN** | Permutation invariance verified |
| `test_cache_lifecycle` | `test_same_slot_cache_fetch_not_instant_hit` | ENV-R5 | **GREEN** | Intra-slot cache fetch not instant hit |
| `test_cache_lifecycle` | `test_cross_uav_cache_cooperation_not_wiped` | ENV-R5 | **GREEN** | Cross-UAV cooperative cache preserved |
| `test_cache_lifecycle` | `test_cache_commit_permutation_invariance` | ENV-R5 | **GREEN** | Deterministic cache commit order |
| `test_collision_joint_feasibility` | `test_reset_positions_are_pairwise_separated` | ENV-R7 | **GREEN** | Reset separation $\ge 200$m |
| `test_collision_joint_feasibility` | `test_head_on_collision_joint_feasibility` | ENV-R8 | **GREEN** | Alternating projection speed ball & separation |
| `test_bandwidth_invariant_to_offload` | `test_bandwidth_and_rate_invariant_across_actions` | ENV-R1 | **GREEN** | Bandwidth invariant to lower offloading |
| `test_metric_semantics` | `test_system_dsr_denominator_includes_unadmitted_requests` | ENV-R9 | **GREEN** | Canonical DSR denominator verified |
| `test_metric_semantics` | `test_critical_ue_ratio_definition` | ENV-R9 | **GREEN** | Critical UE ratio definition verified |
| `test_metric_semantics` | `test_uav_fleet_total_energy_excludes_mbs_energy` | ENV-R9 | **GREEN** | UAV energy excludes MBS energy |
| `test_lower_logprob_contract` | `test_single_request_matches_exact_logprob` | Lower Policy | **GREEN** | Baseline single-request contract |
| `test_lower_logprob_contract` | `test_numerical_stability_at_scale` | Lower Policy | **GREEN** | Stability at 10, 30, 50 requests |
| `test_lower_logprob_contract` | `test_joint_logprob_is_sum_not_mean` | Lower Policy | **RED (Reserved)** | Phase 3 Lower Policy Contract |
| `test_lower_reward_causality` | `test_lower_reward_invariant_to_uav_movement` | Lower Policy | **RED (Reserved)** | Phase 3 Lower Reward Causality |
| `test_no_silent_truncation` | `test_fifty_admitted_requests_all_decided_by_policy` | Lower Policy | **RED (Reserved)** | Phase 3 Capacity Truncation |

**Summary**:
- Total Tests: 26
- Passed: 23
- Failed (Phase 3 Reserved RED): 3
- Errors: 0

---

## 5. Phase 3 Boundary Compliance Declaration

We strictly certify that in Phase 2B:
1. `marl_models/offload_mappo/offload_mappo.py` was **NOT** modified.
2. `marl_models/joint_mappo/` was **NOT** modified.
3. Lower PPO training objectives, loss functions, logprob definitions, and advantage calculations were **NOT** modified.
4. Lower reward function in `run_hierarchical_mappo_experiment.py` (`lower_rewards_from_metrics`) was **NOT** modified.
5. `MAX_OFFLOAD_REQUESTS_PER_UAV = 30` was **NOT** expanded.
6. The 3 lower-policy RED tests (`test_joint_logprob_is_sum_not_mean`, `test_lower_reward_invariant_to_uav_movement`, `test_fifty_admitted_requests_all_decided_by_policy`) failed with the exact expected Phase 3 root causes and remain strictly RED.

---

## 6. Deterministic Environment Smoke Test

- **Script**: `tests/deterministic_smoke.py`
- **Episodes**: 2 independent episodes (5 steps per episode)
- **RNG Seed**: 42
- **Comparison Dimensions**:
  1. UAV 3D coordinates ($p_t$): 100% bitwise equal across all steps.
  2. UE 2D coordinates ($q_t$): 100% bitwise equal across all steps.
  3. UAV cache arrays: 100% bitwise equal across all steps.
  4. Step metrics (JFI, DSR, latency, energy, offline rate, critical ratio): 100% equal ($< 10^{-6}$).
  5. Component energy breakdown (flight, hover, compute, comm, fetch, content, WPT): 100% equal.
  6. Rewards and admission counts: 100% equal.
  7. Numerical anomalies: Zero NaN, zero Inf, zero warnings, zero exceptions.
- **Verification Result**: `ALL DETERMINISM AND INTEGRITY CHECKS PASSED: 100% BITWISE EQUIVALENCE!`

---

## 7. RNG Infrastructure Status

- `TEST_RNG_ISOLATED = YES` (All unit tests explicitly control their own RNG seeds).
- `CANONICAL_RNG_INFRASTRUCTURE_PENDING = YES` (Unified episode/eval RNG manager will be integrated during Phase 4 canonical pipeline execution).
