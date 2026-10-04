# Canonical Environment GREEN Repair Report (Phase 2B)

- **Execution Mode**: Dual-Machine Architecture (Local macOS Darwin editing + Remote Linux x86_64 verification)
- **Status**: `STATUS = ENV_GREEN_REPAIR_COMPLETE`
- **Semantics Version**: `ENV_SEMANTICS_VERSION = clean_v1`
- **Remote Host**: `100.69.44.85:22` (CentOS 7, Conda `drone`)
- **Branch**: `双层注意力MAPPO多智能体强化学习`

---

## 1. Provenance & Commit Hierarchy

| Milestone | Git Commit SHA | Description |
| :--- | :--- | :--- |
| **PHASE2A_ENTRY_SHA** | `28488d23c929c7d669bbd8b866456efe41aed921` | Baseline entry commit (Legacy RED baseline documented). Note: test-strengthening commit is `62576db30658679ee6bfaf761ab216ad3bb3d1d6`. |
| **PHASE2B_FINAL_PRODUCTION_SHA** | `ac1e10309b8c42f98e9bef574892fef45f02cb9f` | Final production code commit where all environment repairs were completed, turning all 21 environment tests GREEN. |
| **PHASE2B_FINAL_REPORT_SHA** | `cdc3ed709437d5c1f54673bf5dbc564cd5bf616f` | Initial Phase 2B delivery report commit (followed by this documentation errata). |

---

## 2. Remote Runtime Manifest (Server: 100.69.44.85)

The following environment details were directly queried on the remote host via `conda activate drone`:

```text
which python:             /home/PengYanghan/miniconda3/envs/drone/bin/python
python --version:         Python 3.12.11
torch.__version__:        2.6.0+cpu
torch.cuda.is_available:  False
torch.version.cuda:       None
numpy.__version__:        2.0.1
```

> [!NOTE]
> **Resolution of Runtime Discrepancy**: Historical notes intermittently mentioned `torch 2.6.0+cu124`. The authoritative remote environment manifest above proves that the canonical `drone` environment runs `torch 2.6.0+cpu` with CPU execution only (`torch.cuda.is_available = False`), eliminating the earlier documentation contradiction.

---

## 3. Executive Summary & Test Accounting

In Phase 2B, all canonical environment physical semantics specified in the frozen `CANONICAL_SEMANTICS_SPEC.md` were implemented and verified on the remote experiment server.

### Exact Test Suite Arithmetic

```text
TOTAL_TESTS = 26
  ├── ENVIRONMENT_TESTS_GREEN    = 21  (All 9 Environment test suites)
  ├── LOWER_CONTRACT_TESTS_GREEN =  2  (Baseline lower logprob contract tests)
  └── PHASE3_RESERVED_RED        =  3  (LOWER-01, LOWER-02, LOWER-04)

TOTAL_GREEN = ENVIRONMENT_TESTS_GREEN (21) + LOWER_CONTRACT_TESTS_GREEN (2) = 23
TOTAL_RED   = PHASE3_RESERVED_RED = 3
ERRORS      = 0
```

- **Environment Tests**: 21 / 21 **GREEN** (100% passed).
- **Lower Policy Contract Tests (Phase 3 Reserved)**: 3 **RED** (strictly preserved, untouched).
- **Deterministic Smoke Test**: **PASS** (100% bitwise equivalence across 2 independent episodes).
- **Model Training**: Strictly prohibited & zero training conducted.

---

## 4. Modified Production & Test Files

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

## 5. Specification Implementation & Before/After Comparison

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
- **After (Canonical Scientific Definition)**:
  The computing processor-sharing denominator on UAV $j$ is strictly defined as the total number of compute tasks finally assigned to execute on UAV $j$:
  $$K_j = \{m \mid \text{task } m \text{ finally executed on UAV } j\} = K_{\text{local}, j} \cup K_{\text{coop-in}, j}$$
  $$N_{\text{assigned}, j} = |K_j|$$
  $$f_{mj} = \frac{F_j}{N_{\text{assigned}, j}}$$
  MBS offloaded requests are processed by the base station grid and are strictly excluded from UAV compute load ($N_{\text{assigned}, j}$).
- **Production Implementation**:
  In `uav.process_requests()`, Phase A resolves all offloading targets, sets initial local load `self._current_service_request_count = local_service_count`, and increments `tuav._current_service_request_count += 1` for each cooperative request targeting neighbor `tuav`. In Phase B, all local tasks share computing capacity $F_j / N_{\text{assigned}, j}$ equally.
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

### ENV-R6: Temporal Alignment & Production Timeline
- **Before**: `step()` calculated motion energy at the beginning using `uav._dist_moved` from the previous slot (initially 0.0), causing step 0 to record zero flight energy and terminal step energy to be dropped.
- **After**: Production `Env.step()` follows the exact execution sequence below:
  - **S0 (Slot-Start Snapshot & Accumulators Reset)**: Freeze immutable `SlotSnapshot` at $p_t$; clear pending cache; reset slot energy accumulators (`_energy_current_slot = 0.0`, component energies = 0.0); compute initial load.
  - **S1–S3 (Service Offloading & Execution on Frozen Snapshot)**: Process requests using slot-start geometry & transmission rates (`uav.process_requests()`), evaluating compute, communication, and cache reading strictly against `cache_snapshot`. Update unassigned UE batteries & service coverage.
  - **S4 (Upper Actions & Kinematic Update)**: Apply upper trajectory actions $a_t$ via `_apply_actions_to_env(actions)`, executing alternating projection to resolve collisions and boundaries while respecting maximum speed ball ($p_t \to p_{t+1}$).
  - **S5 (Current-Action Motion Energy)**: Calculate flight and hover energy from action $a_t$'s actual displacement $\Delta p$ via `uav.update_energy_consumption()`.
  - **S6 (Rewards & Metrics Aggregation)**: Compute transition rewards $r_t$ and metrics $m_t$ via `_get_rewards_and_metrics()`, aggregating motion energy and service compute/comm energy into slot $t$. Record step audit.
  - **S7 (End-of-Slot State Transition & Next Obs)**: Commit pending caches via `uav.update_ema_and_cache()`; periodic GDSF update; advance UE positions; reset transient UAV step counters; construct `next_obs` for $t+1$. `slot_snapshot` is preserved upon return.
- **Four Core Scientific Guarantees**:
  1. Current-slot service physics evaluates strictly against the slot-start snapshot;
  2. Upper movement does not retroactively alter current-slot service channels;
  3. Action $a_t$ movement energy enters transition reward $r_t$ in the same step;
  4. Terminal action energy is accounted for without loss.
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

### ENV-R9: Regression-Preserved Metric Semantics (Regression Guard)
- **Clarification of Historical Baseline**: In Phase 2A, the metric semantics tests were already **GREEN**. Phase 2B did not perform a RED $\to$ GREEN fix here; instead, Phase 2B served as a **Regression Guard**, proving that after the full physics and lifecycle overhaul, canonical metric definitions remained intact:
  - System DSR denominator strictly equals all service requests generated by UEs in the system (including unadmitted requests);
  - Critical-UE Ratio strictly equals the proportion of UEs with battery $< 50$J (`UE_CRITICAL_THRESHOLD`);
  - UAV Fleet Total Energy strictly excludes MBS grid energy.
- **Tests**: `test_metric_semantics.py` (3 tests: `test_system_dsr_denominator_includes_unadmitted_requests`, `test_critical_ue_ratio_definition`, `test_uav_fleet_total_energy_excludes_mbs_energy`).
- **Status**: **GREEN $\to$ GREEN (Regression Guard Verified)**.

---

## 6. ENV-R10 Request Accounting

Production `Env` (in `_associate_ues_to_uavs()` and `_get_rewards_and_metrics()`) computes and exports the following metrics:

| Canonical Semantic Name | Actual Code Metric Key | Source Expression | Description |
| :--- | :--- | :--- | :--- |
| `generated_service_count` | `generated_service_count` (also `service_requests_generated`) | `sum(1 for ue in self._ues if ue.current_request.is_service)` | Total service requests generated by all UEs in system |
| `admitted_service_count` | `admitted_service_count` | `natural_service_requests` (`stats["naturally_covered_service_requests"]`) | Service requests within natural UAV coverage ($d \le R_c$) |
| `unadmitted_service_count` | `unadmitted_service_count` (also `service_requests_uncovered`) | `uncovered_service_requests` (`stats["service_requests_uncovered"]`) | Service requests outside all UAV coverage radii |
| `processed_service_count` | `processed_service_count` (also `service_requests_processed`) | `sum(uav.service_request_count for uav in self._uavs)` | Total service requests processed across all UAVs |
| `forced_service_admissions` | `forced_service_admissions` | `stats["forced_service_admissions"]` | Forced admissions (permanently 0 under canonical) |

All canonical accounting keys are natively exposed in `metrics`, providing full visibility into request generation, admission, and execution without adding redundant production fields.

---

## 7. Test Suite Execution Matrix (Remote Server: 100.69.44.85)

| Test Module | Test Method | Target Component | Status | Category |
| :--- | :--- | :--- | :---: | :--- |
| `test_env_temporal_alignment` | `test_first_step_flight_energy_accounted` | ENV-R6 | **GREEN** | Environment |
| `test_env_temporal_alignment` | `test_terminal_step_energy_not_lost` | ENV-R6 | **GREEN** | Environment |
| `test_env_temporal_alignment` | `test_zero_vs_max_movement_energy_separation_at_step_0` | ENV-R6 | **GREEN** | Environment |
| `test_service_snapshot` | `test_canonical_slot_snapshot_structure_exists` | ENV-R1 | **GREEN** | Environment |
| `test_service_snapshot` | `test_service_channel_rates_match_slot_start_geometry` | ENV-R1 | **GREEN** | Environment |
| `test_service_snapshot` | `test_service_snapshot_functional_invariance` | ENV-R1 | **GREEN** | Environment |
| `test_service_admission` | `test_out_of_coverage_ue_not_assigned` | ENV-R2 | **GREEN** | Environment |
| `test_service_admission` | `test_in_coverage_ue_naturally_admitted` | ENV-R2 | **GREEN** | Environment |
| `test_service_admission` | `test_forced_admission_canonical_is_false` | ENV-R2 | **GREEN** | Environment |
| `test_batch_processor_sharing` | `test_equal_sharing_for_all_tasks_on_same_uav` | ENV-R3 | **GREEN** | Environment |
| `test_batch_processor_sharing` | `test_mbs_tasks_excluded_from_uav_final_load` | ENV-R3 | **GREEN** | Environment |
| `test_order_permutation_invariance` | `test_request_order_permutation_invariance` | ENV-R4 | **GREEN** | Environment |
| `test_cache_lifecycle` | `test_same_slot_cache_fetch_not_instant_hit` | ENV-R5 | **GREEN** | Environment |
| `test_cache_lifecycle` | `test_cross_uav_cache_cooperation_not_wiped` | ENV-R5 | **GREEN** | Environment |
| `test_cache_lifecycle` | `test_cache_commit_permutation_invariance` | ENV-R5 | **GREEN** | Environment |
| `test_collision_joint_feasibility` | `test_reset_positions_are_pairwise_separated` | ENV-R7 | **GREEN** | Environment |
| `test_collision_joint_feasibility` | `test_head_on_collision_joint_feasibility` | ENV-R8 | **GREEN** | Environment |
| `test_bandwidth_invariant_to_offload` | `test_bandwidth_and_rate_invariant_across_actions` | ENV-R1 | **GREEN** | Environment |
| `test_metric_semantics` | `test_system_dsr_denominator_includes_unadmitted_requests` | ENV-R9 | **GREEN** | Environment (Regression Guard) |
| `test_metric_semantics` | `test_critical_ue_ratio_definition` | ENV-R9 | **GREEN** | Environment (Regression Guard) |
| `test_metric_semantics` | `test_uav_fleet_total_energy_excludes_mbs_energy` | ENV-R9 | **GREEN** | Environment (Regression Guard) |
| `test_lower_logprob_contract` | `test_single_request_matches_exact_logprob` | Lower Policy | **GREEN** | Lower Contract (Single Req) |
| `test_lower_logprob_contract` | `test_numerical_stability_at_scale` | Lower Policy | **GREEN** | Lower Contract (Scale Stability) |
| `test_lower_logprob_contract` | `test_joint_logprob_is_sum_not_mean` | Lower Policy | **RED (Reserved)** | Phase 3 Contract (LOWER-01) |
| `test_lower_reward_causality` | `test_lower_reward_invariant_to_uav_movement` | Lower Policy | **RED (Reserved)** | Phase 3 Causality (LOWER-02) |
| `test_no_silent_truncation` | `test_fifty_admitted_requests_all_decided_by_policy` | Lower Policy | **RED (Reserved)** | Phase 3 Truncation (LOWER-04) |

---

## 8. Phase 3 Boundary Compliance Declaration

We strictly certify that in Phase 2B:
1. `marl_models/offload_mappo/offload_mappo.py` was **NOT** modified.
2. `marl_models/joint_mappo/` was **NOT** modified.
3. Lower PPO training objectives, loss functions, logprob definitions, and advantage calculations were **NOT** modified.
4. Lower reward function in `run_hierarchical_mappo_experiment.py` (`lower_rewards_from_metrics`) was **NOT** modified.
5. `MAX_OFFLOAD_REQUESTS_PER_UAV = 30` was **NOT** expanded.
6. The 3 lower-policy RED tests (`test_joint_logprob_is_sum_not_mean`, `test_lower_reward_invariant_to_uav_movement`, `test_fifty_admitted_requests_all_decided_by_policy`) failed with the exact expected Phase 3 root causes and remain strictly RED.

---

## 9. Deterministic Environment Smoke Test

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

## 10. RNG Infrastructure Status

- `TEST_RNG_ISOLATED = YES` (All unit tests and deterministic smoke scripts explicitly seed and isolate their own RNG).
- `CANONICAL_RNG_INFRASTRUCTURE_PENDING = YES` (Deterministic unit-test smoke passes do not imply that the formal multi-seed experiment training/evaluation RNG infrastructure is complete; this remains scheduled for Phase 4).

---

## 11. Final Exit Gate

```text
SSOT_MODIFIED = NO
FORMAL_TRAINING_STARTED = NO
ENVIRONMENT_TESTS_GREEN = YES
ENVIRONMENT_DETERMINISM_SMOKE = PASS
PHASE3_EXPECTED_REDS_REMAIN = YES
CANONICAL_RNG_INFRASTRUCTURE_PENDING = YES
STATUS = READY_FOR_LOWER_REPAIR
ENV_SEMANTICS_VERSION = clean_v1
```
