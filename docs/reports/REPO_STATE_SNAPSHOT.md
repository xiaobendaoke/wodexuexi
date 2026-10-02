# Repository State Snapshot — Full Technical Project Map

- **Snapshot Date**: 2026-09-18
- **Current Milestone**: Pre-Phase 3 Architectural Lock
- **Document Role**: Exhaustive, read-only technical inventory and architectural reference for external technical review
- **Environment Semantics Version**: `clean_v1`
- **SSOT Status**: FROZEN (`CANONICAL_SEMANTICS_SPEC.md`)

---

## 1. Git / Provenance Snapshot

### Worktree Status

```text
CURRENT_BRANCH: 双层注意力MAPPO多智能体强化学习
CURRENT_HEAD:   4b9652a5ae1e12cc4b34902a2c7569e2fbe96335
WORKTREE_CLEAN: YES (Tracked source tree is fully committed; untracked items limited to build/transient artifacts)
```

### Key Milestone Commit Provenance

| Project Phase | Commit SHA | Commit Message / Description |
| :--- | :--- | :--- |
| **SSOT Freeze** | `2e96ae8b1d9bf5b2723cfb8b7ea7c5553047d1df` | `docs: add CLEAN_SLATE_AUDIT.md and frozen CANONICAL_SEMANTICS_SPEC.md with dual-machine rules` |
| **Phase 2A Baseline** | `28488d23c929c7d669bbd8b866456efe41aed921` | `docs: add Phase 2A Legacy RED Baseline report` |
| **Phase 2B Repair** | `ac1e10309b8c42f98e9bef574892fef45f02cb9f` | `fix(env): align slot snapshot lifecycle, energy inspection, and test fixtures` |
| **Phase 2B Delivery** | `cdc3ed709437d5c1f54673bf5dbc564cd5bf616f` | `docs: add Phase 2B Canonical Environment GREEN Repair report` |
| **Phase 2B Errata** | `4b9652a5ae1e12cc4b34902a2c7569e2fbe96335` | `docs: apply Phase 2B documentation closeout errata to ENV_REPAIR_REPORT.md` |

---

## 2. Repository Tree & Storage Breakdown

### Directory Size and File Inventory

```text
wodexuexi/ (Total: ~8.0 GB on disk)
├── .git/                      [2.9 GB, version database]
├── recent/                    [7,805 files, 3.73 GB, archived legacy experiments & runs]
├── latex/                     [258 files, 524.56 MB, thesis LaTeX source & compiled assets]
├── saved_models/              [144 files, 286.96 MB, historical model checkpoints]
├── .venv/                     [Linux x86_64 legacy virtual environment, 204 MB]
├── docs/                      [68 files, 74.69 MB, technical audit reports & figures]
├── train_logs/                [301 files, 35.98 MB, training logs & configs]
├── results/                   [522 files, 20.76 MB, historical multi-seed benchmark results]
├── output/                    [13 MB, generated plot artifacts]
├── marl_models/               [70 files, 450 KB, RL agent & policy implementations]
├── environment/               [15 files, 260 KB, simulation physics, UAV, UE models]
├── utils/                     [9 files, 140 KB, plotting, logger, metrics helpers]
├── scripts/                   [12 files, 130 KB, analysis and evaluation scripts]
├── tests/                     [15 files, 60 KB, unit & regression test suite]
└── root Python/Shell scripts  [26 top-level files]
```

### Representative Files in Heavy Directories

- `saved_models/`:
  - `ippo_baseline_baseline_matrix_v2_ippo_seed42_20260606_182647/final/ippo_baseline.pth`
  - `offload_mappo_baseline_matrix_v2_ippo_seed42_20260624_181345/final/offload_mappo.pth`
  - `vanilla_mappo_baseline_matrix_v2_vanilla_mappo_seed84_20260625_012047/final/vanilla_mappo.pth`
  - `joint_mappo_baseline_matrix_v2_joint_mappo_seed42_20260627_122212/final/joint_mappo.pth`
- `results/`:
  - `baseline_matrix_force_admission_20260624/manifest.json`
  - `baseline_matrix_force_admission_20260624/ippo/seed_42/workload_252_metrics.json`
  - `baseline_matrix_v2_smoke/manifest.json`
- `recent/`:
  - Archived multi-run scripts, nohup logs, and deprecated PID tracking from June/July 2026 exploratory sweeps (`archive_20260701_old_paper_artifacts`, `archive_20260703_old_data_cleanup`).

---

## 3. Top-Level File Inventory

| Path | Type | Current Role | Status |
| :--- | :--- | :--- | :--- |
| `AGENTS.md` | Markdown | Guidance for Codex/Agents, dual-machine execution protocol | **Canonical** |
| `CLAUDE.md` | Markdown | Guidance for Claude Code, dual-machine boundaries | **Canonical** |
| `CANONICAL_SEMANTICS_SPEC.md` | Markdown | Frozen Single Source of Truth (SSOT) for system physics | **Canonical (Frozen)** |
| `docs/provenance/CLEAN_SLATE_AUDIT.md` | Markdown | Phase 1 full codebase audit report | **Historical / Provenance** |
| `docs/provenance/TDD_RED_BASELINE.md` | Markdown | Phase 2A test suite baseline capture report | **Historical / Provenance** |
| `docs/reports/ENV_REPAIR_REPORT.md` | Markdown | Phase 2B environment GREEN repair & errata closeout | **Canonical Report** |
| `config.py` | Python | Global configuration, environment & training hyperparameters | **Canonical Production** |
| `paths.py` | Python | Repository root and results path resolver | **Canonical Production** |
| `run_hierarchical_mappo_experiment.py` | Python | Main training runner for hierarchical Attention-MAPPO | **Current Production Runner** |
| `run_joint_end_to_end_mappo_experiment.py` | Python | Training runner for joint end-to-end MAPPO baseline | **Current Baseline Runner** |
| `run_baseline_matrix_v2.py` | Python | Matrix evaluation runner across workloads and seeds | **Legacy / Historical** |
| `run_baseline_comparison_experiment.py` | Python | Baseline comparative evaluation script | **Legacy / Historical** |
| `run_sensitivity_full.py` | Python | Hyperparameter and workload sensitivity sweep runner | **Legacy / Historical** |
| `run_linear_v2_full.sh` | Shell | Shell runner for 3 seeds × 10 workloads matrix | **Legacy / Historical** |
| `run_linear_v2_dsr_strong.sh` | Shell | Shell runner for DSR-strong parameter configuration | **Legacy / Historical** |
| `analyze_experiment_statistics.py` | Python | Statistical hypothesis testing (t-test, Wilcoxon, 95% CI) | **Reusable Tool** |
| `analyze_baseline_matrix_v2.py` | Python | Aggregation and LaTeX table generator for matrix v2 | **Legacy Tool** |
| `analyze_references.py` | Python | Academic reference extraction and metadata analysis | **Auxiliary Tool** |
| `analyze_all_references.py` | Python | Reference cross-referencing against thesis text | **Auxiliary Tool** |
| `analyze_all_references_v2.py` | Python | Extended reference auditor | **Auxiliary Tool** |
| `extract_all_pdfs.py` | Python | PDF text extraction utility | **Auxiliary Tool** |
| `extract_convergence_data.py` | Python | Training convergence curve extractor from logs | **Auxiliary Tool** |
| `reference_analysis.json` | JSON | Output database of analyzed thesis references | **Auxiliary Data** |
| `wodexuexi.code-workspace` | JSON | Visual Studio Code multi-root workspace settings | **Auxiliary Config** |
| `README.md` | Markdown | High-level repository overview and documentation | **Informational** |
| `LICENSE` | Text | MIT open-source license | **Canonical** |

---

## 4. Python Module Architecture Map (`environment/`)

The environment package models the physical simulation of 5 UAVs, 100 UEs, and 1 Macro Base Station (MBS) in a 700m × 700m geographic area.

```text
environment/
├── __init__.py           # Package marker
├── comm_model.py         # Channel gain and Shannon wireless capacity formulas
├── request_types.py      # Request class and service/content/energy request definitions
├── user_equipments.py    # UE mobility, battery dynamics, and request generation
├── uavs.py               # UAV agent physics, computing, dual-state cache, batch execution
└── env.py                # Core Env class, SlotSnapshot, S0-S7 lifecycle, observation construction
```

### Detailed Component Analysis

#### `environment/comm_model.py`
- **Functions**:
  - `calculate_channel_gain(pos1, pos2)`: Free-space path loss with carrier frequency 2 GHz, product $G_0 g_0 = 2.2846 \times 1.42 \times 10^{-4}$.
  - `calculate_ue_uav_rate(channel_gain, num_covered_ues)`: Downlink/uplink Shannon rate $R = \frac{B_{\text{edge}}}{N_{\text{covered}}} \log_2(1 + \frac{P \cdot g}{\sigma^2})$.
  - `calculate_uav_uav_rate(channel_gain)`: Inter-UAV rate $R = B_{\text{inter}} \log_2(1 + \frac{P \cdot g}{\sigma^2})$.
  - `calculate_uav_mbs_rate(channel_gain)`: UAV-to-MBS backhaul rate $R = B_{\text{backhaul}} \log_2(1 + \frac{P \cdot g}{\sigma^2})$.
- **Imported by**: `env.py`, `uavs.py`, `test_service_snapshot.py`.
- **State**: Stateless pure mathematical functions.

#### `environment/request_types.py`
- **Classes**: `Request`
- **Attributes**: `req_type`, `req_id`, `req_size`, `deadline`, `priority`.
- **Properties**: `is_service`, `is_content`, `is_energy`.
- **Imported by**: `user_equipments.py`, `uavs.py`, `env.py`, test suites.

#### `environment/user_equipments.py`
- **Classes**: `UE`
- **Key Methods**:
  - `generate_hotspots()` / `reset_hotspots()`: Sets centers for 2 Gaussian user clusters.
  - `generate_request()`: Generates Zipf-distributed requests ($S=25$ services, $K=50$ contents, or energy).
  - `update_position()`: Bounded 2D movement (max 15m/slot) biased towards hotspot centers.
  - `update_battery(harvested_energy, consumed_energy)`: Battery accounting, threshold at 50J (`UE_CRITICAL_THRESHOLD`).
  - `is_service_deadline_satisfied()`: Evaluates if latency $\le$ request deadline.
- **Key Mutable State**: `pos`, `battery_level`, `current_request`, `assigned`, `latency_current_request`, `service_coverage`.

#### `environment/uavs.py`
- **Classes**: `UAV`, `ServiceOffloadContext`
- **Key Physical Methods**:
  - `reset_for_next_step()`: Clears per-step transient associations (`_current_covered_ues`, `_neighbors`, counters). Slot-t energies are retained until S0 of next step.
  - `commit_pending_cache()`: Commits pending fetched files to storage at slot boundary, ordered by $\text{score} = \text{EMA} / \text{size}$ descending.
  - `update_position(pos)`: Updates physical coordinates, sets `_dist_moved`.
  - `update_energy_consumption()`: Evaluates flight power $P_{\text{move}}$ and hover power $P_{\text{hover}}$ over slot duration $\tau=1.0$s.
  - `process_requests(sample_recorder, offload_actions)`:
    - **Phase A**: Iterates admitted service requests, resolves offload target (0=Local, 1=MBS, 2+j=Coop UAV j). Computes local load $K_{\text{local}}$ and increments coop-in targets `tuav._current_service_request_count`.
    - **Phase B**: Computes uniform processor sharing $f_{mj} = F_j / N_{\text{assigned}, j}$ for all local tasks; executes communication and cache reads against `cache_snapshot`.
  - `update_ema_and_cache()`: Updates file access frequencies via exponential moving average ($\alpha=0.75$).
  - `gdsf_cache_update()`: Periodic GreedyDual-Size-Frequency cache eviction.
- **Key Mutable State**: `pos`, `cache`, `cache_snapshot`, `pending_cache`, `_energy_current_slot`, component energies (`flight_energy`, `hover_energy`, `service_compute_energy`, `service_comm_energy`, `service_fetch_or_backhaul_energy`, `content_related_energy`, `wpt_energy`).

#### `environment/env.py`
- **Dataclass**: `SlotSnapshot` (frozen, immutable slot-start snapshot containing UAV positions, UE positions, all link rates, associations, and cache state).
- **Class**: `Env`
- **Core Lifecycle**:
  - `reset()`: Deterministic rejection sampling ensures initial pairwise UAV separation $\ge 200$m. Freezes slot 0 `SlotSnapshot`. Returns initial obs.
  - `step(actions, offloading_actions)`:
    - **S0**: Capture immutable `SlotSnapshot`; reset per-slot energy accumulators.
    - **S1–S3**: Execute requests on frozen geometry and rates; update UE battery & coverage.
    - **S4**: Apply actions $a_t$ with alternating projection ensuring speed ball $\|p_{t+1}-p_t\| \le v_{\max}\Delta t$ and separation $\ge 200$m.
    - **S5**: Calculate flight/hover energy from displacement $\Delta p$.
    - **S6**: Aggregate transition rewards $r_t$ and metrics $m_t$.
    - **S7**: Commit pending caches; advance UEs; build `next_obs`. Slot snapshot is preserved on env.
  - `_associate_ues_to_uavs()`: Natural admission only ($d \le R_c = 100$m). Uncovered UEs remain unadmitted (`assigned=False`).
  - `get_offloading_obs_and_masks()`: Extracts lower-agent observations and quality/feasibility action masks.

---

## 5. MARL Architecture Map (`marl_models/`)

The repository contains implementations of both upper trajectory controllers and lower request offloaders:

```text
marl_models/
├── base_model.py                # MARLModel base class and ExperienceBatch typing
├── buffer_and_helpers.py        # Rollout buffers (Attention, Discrete Offload, Joint)
├── attention.py                 # CrossAttentionExtractor, AttentionActorBase, AttentionCriticBase
├── utils.py                     # Factory get_model(), model save/load helpers
├── offload_policy.py            # Standalone classifier interface (heuristic/oracle fallback)
├── attention_mappo/             # Upper Trajectory MAPPO with multi-head cross attention
├── vanilla_mappo/               # Upper Trajectory MAPPO with standard MLP
├── ippo_baseline/               # Independent PPO trajectory baseline
├── joint_mappo/                 # End-to-end joint trajectory + offloading MAPPO
├── offload_mappo/               # Lower Request-Level Offloading MAPPO
├── uncoordinated_greedy_baseline/ # Heuristic non-cooperative greedy baseline
├── random_baseline/             # Random valid offloading baseline
└── uniform_baseline/            # Round-robin uniform offloading baseline
```

### Algorithm Specification Matrix

| Algorithm | Model Directory | Actor Architecture | Critic Architecture | Observation Dim | Action Dim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention-MAPPO** (Upper) | `marl_models/attention_mappo/` | Cross-Attention over Neighbors & UEs + Gaussian 2D Head | Cross-Attention Centralized Critic | 327 | 2 (Continuous: $\Delta x, \Delta y$) |
| **Vanilla-MAPPO** (Upper) | `marl_models/vanilla_mappo/` | 2-layer MLP (128 units) + Gaussian Head | Centralized MLP Critic | 327 | 2 (Continuous: $\Delta x, \Delta y$) |
| **IPPO Baseline** (Upper) | `marl_models/ippo_baseline/` | 2-layer MLP (128 units) + Gaussian Head | Decentralized Local MLP Critic | 327 | 2 (Continuous: $\Delta x, \Delta y$) |
| **Joint-MAPPO** (Unified) | `marl_models/joint_mappo/` | Joint Actor: Trajectory (2D Gaussian) + Offload ($K \times 7$ Categorical) | Centralized Value Critic | 327 | Continuous (2) + Discrete ($K \times 7$) |
| **Offload-MAPPO** (Lower) | `marl_models/offload_mappo/` | Self-Attention over Request Tokens + Head ($K \times 7$ Categorical) | Centralized Request-Attention Critic | $5 + 14 \times K$ | Discrete per request ($0 \dots 6$) |

---

## 6. Current Lower MAPPO State (Phase 3 Baseline)

### Key Parameter References

```text
MAX_OFFLOAD_REQUESTS_PER_UAV: 30  (Configured equal to MAX_ASSOCIATED_UES = 30)
MAX_ASSOCIATED_UES:           30  (Upper bound of associated UEs visible to single UAV)
OFFLOAD_NUM_ACTIONS:          7   (0: Local, 1: MBS, 2..6: Cooperative UAV 0..4)
OFFLOAD_REQUEST_FEATURE_DIM:  14  (Per-request feature vector)
OFFLOAD_OBS_DIM_SINGLE:       425 (5 own features + 30 * 14 request features)
```

### Lower Observation and Action Pipeline

```text
1. Request Generation
   └── 100 UEs generate service/content/energy requests via user_equipments.py

2. Natural Service Admission (ENV-R2)
   └── Only UEs within 100m radius of UAV j are admitted to UAV j's covered list

3. Feature Construction & Sorting
   └── env.py:get_offloading_obs_and_masks()
   └── Admitted service UEs sorted by distance: service_ues[:MAX_OFFLOAD_REQUESTS_PER_UAV]
   └── Request features: [valid, req_size, req_id, deadline, priority, cache_hit, coop_avail,
                          local_queue, local_ratio, coop_ratio, mbs_ratio, best_neighbor_share,
                          delta_x, delta_y] (14 dims)

4. Tensor Padding
   └── Zero-padded to shape (NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV, 14)
   └── Own features: [pos_x, pos_y, queue_ratio, neighbor_ratio, mbs_rate] (5 dims)
   └── Total obs shape per UAV: (425,)

5. Action Mask Construction
   └── Shape: (NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV, 7)
   └── Action 0 (Local): Always 1.0
   └── Action 1 (MBS): 1.0 (masked to 0.0 only if latency exceeds deadline ratio clip)
   └── Action 2+j (Coop UAV j): 1.0 if neighbor j within sensing range AND coop_feasible

6. Policy Sampling & Action Selection
   └── marl_models/offload_mappo/offload_mappo.py:OffloadActor
   └── Request tokens processed via MultiheadAttention(64, 8 heads)
   └── Categorical distribution over 7 actions for each of the 30 slots

7. Rollout Storage & PPO Update
   └── DiscreteOffloadRolloutBuffer stores obs, actions, masks, log_probs, rewards, values
   └── GAE advantage computation + PPO clipped surrogate objective update
```

### Action Semantics

- **`0`**: Execute locally on the covering UAV.
- **`1`**: Offload to Macro Base Station (MBS) via backhaul link.
- **`2 + j`** (where $j \in \{0, 1, 2, 3, 4\}$): Offload to cooperative UAV $j$ via inter-UAV link.

### Logprob Arithmetic & Identified Defect (LOWER-01)

In `marl_models/offload_mappo/offload_mappo.py`:
- In `get_action_and_value()` (Lines 243–246):
  ```python
  slot_valid = (masks_tensor.sum(dim=-1) > 0.0).float()
  action_log_probs = dist.log_prob(actions) * slot_valid
  denom = slot_valid.sum(dim=-1).clamp_min(1.0)
  log_probs = action_log_probs.sum(dim=-1) / denom  # <--- ARITHMETIC MEAN INSTEAD OF SUM
  ```
- In `update()` (Lines 282–285):
  ```python
  slot_log_probs = dist.log_prob(actions_batch) * valid_slots
  denom = valid_slots.sum(dim=-1).clamp_min(1.0)
  new_log_probs = slot_log_probs.sum(dim=-1) / denom  # <--- ARITHMETIC MEAN INSTEAD OF SUM
  entropy = (dist.entropy() * valid_slots).sum(dim=-1) / denom
  ```
- **Consequence**: Under the product rule for independent categorical choices across valid requests, the joint log-likelihood must be $\log P(A) = \sum_{k=1}^K \log P(a_k)$. Dividing by $K$ turns the joint probability into the geometric mean probability $P(A)^{1/K}$, distorting PPO importance sampling ratios $r(\theta) = \exp(\log P_\theta - \log P_{\text{old}})$. This causes `test_joint_logprob_is_sum_not_mean` to be **RED**.

### Capacity Truncation & Identified Defect (LOWER-04)

- Current `config.MAX_OFFLOAD_REQUESTS_PER_UAV` is set to 30.
- In `environment/env.py` line 290, `service_ues` is sliced to `[:30]`.
- In `environment/uavs.py` line 423:
  ```python
  if offload_actions is not None and service_action_cursor < int(config.MAX_OFFLOAD_REQUESTS_PER_UAV):
      # decided by learned policy
  else:
      # silently falls back to heuristic target selection!
  ```
- **Consequence**: When more than 30 UEs are admitted to a UAV (up to 100 UEs total in system), requests 31–100 are silently stripped from the policy and offloaded via fallback heuristic. This causes `test_fifty_admitted_requests_all_decided_by_policy` to be **RED**.

---

## 7. Lower Reward Pipeline & Motion Pollution (LOWER-02)

### Reward Formulation in `run_hierarchical_mappo_experiment.py`

Function `lower_rewards_from_metrics()` (Lines 70–94):

```python
deadline_penalty = 1.0 - float(metrics["deadline_satisfaction_rate"])
latency_term     = float(metrics["latency"]) / (config.NUM_UES * config.NON_SERVED_LATENCY_PENALTY + config.EPSILON)
energy_term      = float(metrics["energy"]) / (config.NUM_UAVS * config.OFFLOAD_ENERGY_NORM_REF + config.EPSILON)
mbs_term         = float(metrics["mbs_load_ratio"])
coop_term        = float(metrics["offloading_ratio_cooperative"])
success_term     = float(metrics["deadline_satisfaction_rate"])

lower_reward = (
    0.40 * success_term
    + 0.08 * coop_term
    - 3.00 * deadline_penalty
    - 4.00 * latency_term
    - 1.20 * energy_term       # <--- POLLUTED BY FLIGHT & HOVER ENERGY
    - 0.35 * mbs_term
)
lower_reward -= constraint_penalty
```

### Identified Causal Violation
- `metrics["energy"]` represents `sum(uav.energy for uav in self._uavs)`, which includes UAV mechanical propulsion energy ($P_{\text{move}} \cdot \Delta t_{\text{move}} + P_{\text{hover}} \cdot \Delta t_{\text{hover}} \approx 150 \sim 300$ W).
- In contrast, service computing energy is on the order of millijoules to joules ($10^{-27} \cdot C \cdot f^2$).
- Thus, the lower offloading agent's reward is heavily dominated by whether the upper trajectory controller decided to fly fast or hover, violating Markov decision process causal attribution for offloading decisions. This causes `test_lower_reward_invariant_to_uav_movement` to be **RED**.

---

## 8. Constraint Mechanism Map

### Adaptive Dual Multiplier Implementation

Implemented in `run_hierarchical_mappo_experiment.py`:

```python
# S5 Violation Computation
dsr_violation      = max(0.0, float(config.OFFLOAD_DSR_TARGET) - float(metrics["deadline_satisfaction_rate"]))
mbs_load_violation = max(0.0, float(metrics["mbs_load_ratio"]) - float(config.OFFLOAD_MBS_LOAD_CEILING))

# Constraint Penalty Formulation
constraint_penalty = lambda_dsr * dsr_violation + lambda_mbs * mbs_load_violation

# Multiplier Update Rule (End of Rollout)
lambda_dsr = float(np.clip(lambda_dsr + config.OFFLOAD_LAGRANGE_LR * mean_dsr_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))
lambda_mbs = float(np.clip(lambda_mbs + config.OFFLOAD_LAGRANGE_LR * mean_mbs_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))
```

### Parameters
- $\text{DSR}_{\text{target}}$: `config.OFFLOAD_DSR_TARGET = 0.18`
- $\text{MBS}_{\text{ceiling}}$: `config.OFFLOAD_MBS_LOAD_CEILING = 0.03`
- $\alpha_{\text{lag}}$: `config.OFFLOAD_LAGRANGE_LR = 0.1`
- $\lambda_{\max}$: `config.OFFLOAD_LAGRANGE_MAX = 10.0`

### Scientific Nature vs Terminology
- In code and paper, this is termed `Lagrange constraint mode` (`OFFLOAD_CONSTRAINT_MODE = "lagrange"`).
- In mathematical optimization terms, this operates as an **adaptive penalty method / dual subgradient ascent on constraint violations**, rather than an exact augmented Lagrangian method.

---

## 9. Attention & Mask Architecture

### Upper Attention (`marl_models/attention.py`)
- **Query**: Agent's own state (pos + cache vector, 77 dims $\to$ projected to 64 dims).
- **Key / Value**:
  - Neighbor tokens: up to 4 neighbors $\times$ 2 dims (relative pos).
  - UE tokens: up to 30 UEs $\times$ 8 dims (relative pos + request signature + battery).
- **Attention Type**: Scaled dot-product multi-head attention (8 heads, head dim = 8, embed dim = 64).
- **Positional Encoding**: **None**. Relative spatial coordinates $(\Delta x, \Delta y)$ are explicitly provided in token features.
- **Pooling**: Key padding mask zeroes out inactive neighbor/UE slots; output context vector is concatenated with own features and passed to MLP policy head.

### Lower Attention (`marl_models/offload_mappo/offload_mappo.py`)
- **Tokens**: Up to 30 request tokens $\times$ 14 features each.
- **Own Context**: 5 features (own pos, queue length, neighbor count, MBS link rate).
- **Architecture**:
  - Request tokens linearly encoded to 64 dims.
  - Multi-head self-attention across requests (8 heads, embed dim = 64) modeling inter-request resource competition.
  - Fused token representation concatenated with own context embedding $\to$ linear projection to 7 action logits per request.
- **Action Masking**:
  - Hard physical feasibility: Inter-UAV links beyond sensing range are masked out (0.0).
  - Quality masking: Cooperative actions exceeding deadline/latency ratio thresholds are masked out.
  - Feasibility and quality masks are **jointly fused** into `all_masks`.

---

## 10. Experiment Entrypoint Map

| Script | Purpose | Underlying Algorithm | Mode | Canonical Status |
| :--- | :--- | :--- | :--- | :--- |
| `run_hierarchical_mappo_experiment.py` | Primary training and evaluation script for Hierarchical Attention-MAPPO | Upper Attention-MAPPO + Lower Offload MAPPO | Train & Eval | **CURRENT (Target for Phase 3/4)** |
| `run_joint_end_to_end_mappo_experiment.py` | Training runner for joint single-policy baseline | Joint-MAPPO (single actor outputs continuous + discrete) | Train | **CURRENT Baseline** |
| `evaluate_joint_end_to_end_mappo.py` | Evaluation runner for trained Joint-MAPPO checkpoints | Joint-MAPPO | Eval | **CURRENT Baseline** |
| `run_baseline_matrix_v2.py` | Multi-seed (3 seeds × 10 workloads) baseline matrix execution | IPPO, Vanilla, Proposed, Joint, Random, Uniform | Train & Eval | **HISTORICAL (Runs under old semantics)** |
| `run_baseline_comparison_experiment.py` | Legacy comparative performance evaluator | All baselines | Eval | **HISTORICAL** |
| `run_sensitivity_full.py` | Parameter sensitivity sweep over capacities & thresholds | Proposed method | Eval | **HISTORICAL** |
| `run_linear_v2_full.sh` | Shell launcher for full baseline matrix | Bash script invoking `run_baseline_matrix_v2.py` | Batch Job | **HISTORICAL** |
| `run_linear_v2_dsr_strong.sh` | Shell launcher for DSR-strong experiment suite | Bash script invoking `run_baseline_matrix_v2.py` | Batch Job | **HISTORICAL** |
| `tests/run_all_tests.py` | Automated test runner for the 12 canonical TDD test suites | Python unittest runner | Test | **CURRENT Test Runner** |
| `tests/deterministic_smoke.py` | 2-episode deterministic smoke verification script | Environment simulation | Test | **CURRENT Verification** |

---

## 11. Chapter-to-Code Mapping

### Chapter 3: System Model & Problem Formulation
- **System Model**: Implemented in `environment/` (`comm_model.py`, `uavs.py`, `user_equipments.py`, `env.py`).
- **Energy Models**: UAV flight ($P_{\text{move}}$), hover ($P_{\text{hover}}$), computing ($k C f^2$), communication, WPT.
- **Problem Statement**: Minimizing composite cost (fairness, latency, energy, offline rate, DSR violation).

### Chapter 4: Task Offloading Optimization
- **Heuristic Baselines**:
  - `marl_models/uncoordinated_greedy_baseline/`: Uncoordinated greedy heuristic.
  - `marl_models/random_baseline/`: Random valid action selector.
  - `marl_models/uniform_baseline/`: Uniform round-robin selector.
- **Standalone Snapshot Evaluators**:
  - Monte Carlo Iterative Task Re-allocation (MC-ITR): **NOT IMPLEMENTED**.
  - Exact-Small Snapshot Evaluator: **NOT IMPLEMENTED**.
  *(Offloading is currently executed dynamically within the MARL environment loop).*

### Chapter 5: Dual-Layer Attention-MAPPO
- **Upper Trajectory Layer**: `marl_models/attention_mappo/` (Attention-MAPPO), `vanilla_mappo/`, `ippo_baseline/`.
- **Lower Offloading Layer**: `marl_models/offload_mappo/` (Constrained-Attention-Offload-MAPPO).
- **Joint Baseline**: `marl_models/joint_mappo/` (Joint End-to-End MAPPO).
- **Ablations**: `no_attention_offload_mappo`, `no_lagrange`, `no_mask` toggled in `run_hierarchical_mappo_experiment.py`.

---

## 12. Tests Inventory (`tests/`)

Tested on Remote Server (`100.69.44.85:22`) running PyTorch 2.6.0+cpu:

| Test File | Method Count | Target Subsystem | Current Status | Notes |
| :--- | :---: | :--- | :---: | :--- |
| `test_env_temporal_alignment.py` | 3 | ENV-R6: Temporal Alignment | **GREEN** | S0-S7 order, terminal energy |
| `test_service_snapshot.py` | 3 | ENV-R1: Slot-Start Snapshot | **GREEN** | Frozen channel rates, immutable snapshot |
| `test_service_admission.py` | 3 | ENV-R2: Service Admission | **GREEN** | Forced admission disabled, natural coverage only |
| `test_batch_processor_sharing.py` | 2 | ENV-R3: Processor Sharing | **GREEN** | Equal compute share, MBS exclusion |
| `test_order_permutation_invariance.py` | 1 | ENV-R4: Permutation Invariance | **GREEN** | Sequence order invariance |
| `test_cache_lifecycle.py` | 3 | ENV-R5: Cache Lifecycle | **GREEN** | Snapshot read, pending write, S7 commit |
| `test_collision_joint_feasibility.py` | 2 | ENV-R7 / ENV-R8: Kinematics | **GREEN** | Reset separation $\ge 200$m, alternating projection |
| `test_bandwidth_invariant_to_offload.py` | 1 | ENV-R1: Bandwidth Invariance | **GREEN** | Bandwidth invariant to lower actions |
| `test_metric_semantics.py` | 3 | ENV-R9: Metric Semantics | **GREEN** | Regression Guard: DSR, Critical UE, UAV energy |
| `test_lower_logprob_contract.py` | 3 | Lower Policy Contract | **1 RED / 2 GREEN** | `test_joint_logprob_is_sum_not_mean` is RED (Phase 3) |
| `test_lower_reward_causality.py` | 1 | Lower Reward Causality | **RED** | `test_lower_reward_invariant_to_uav_movement` is RED (Phase 3) |
| `test_no_silent_truncation.py` | 1 | Lower Capacity Truncation | **RED** | `test_fifty_admitted_requests_all_decided_by_policy` is RED (Phase 3) |

```text
TEST SUMMARY:
Total Tests Run: 26
Passed (GREEN):  23  (21 Environment + 2 Lower Contract)
Failed (RED):     3  (Phase 3 Reserved: LOWER-01, LOWER-02, LOWER-04)
Errors:           0
Duration:         1.60s
```

---

## 13. Runtime Architecture

### Local Development Host (macOS Darwin arm64)
- **OS**: macOS Darwin 24.6.0 (Apple Silicon M-series, arm64)
- **Primary Role**: Code editing, static code audit, LaTeX compilation (`make main`), Git version control.
- **Constraints**: Repo `.venv` is Linux x86_64; local PyTorch dynamic execution is disabled. No heavy MARL training permitted locally.

### Remote Experiment Server (`100.69.44.85:22`)
- **OS**: Linux localhost.localdomain 3.10.0-1160.119.1.el7.x86_64 (CentOS 7)
- **Architecture**: x86_64 (Intel Xeon / AMD)
- **Conda Environment**: `/home/PengYanghan/miniconda3/envs/drone`
- **Python Path & Version**: `/home/PengYanghan/miniconda3/envs/drone/bin/python` (Python 3.12.11)
- **PyTorch Version**: `2.6.0+cpu` (`torch.cuda.is_available() = False`, `torch.version.cuda = None`)
- **NumPy Version**: `2.0.1`
- **Primary Role**: All MARL model training, test suite execution, benchmark evaluation.

---

## 14. Scientific Parameters Summary (`config.py`)

| Parameter | Symbol | Value | Unit / Description |
| :--- | :--- | :--- | :--- |
| `NUM_UAVS` | $U$ | 5 | Number of UAV agents |
| `NUM_UES` | $M$ | 100 | Number of user equipments |
| `AREA_WIDTH` / `AREA_HEIGHT` | $X_{\max}, Y_{\max}$ | 700 × 700 | Simulation area in meters |
| `UAV_ALTITUDE` | $H$ | 100.0 | Fixed flight altitude in meters |
| `UAV_SPEED` | $v_{\max}$ | 15.0 | Maximum horizontal flight speed (m/s) |
| `TIME_SLOT_DURATION` | $\tau$ | 1.0 | Discrete time slot duration (seconds) |
| `UAV_COVERAGE_RADIUS` | $R_c$ | 100.0 | Natural communication coverage radius (m) |
| `MIN_UAV_SEPARATION` | $d_{\min}$ | 200.0 | Minimum safety separation distance between UAVs (m) |
| `UAV_SENSING_RANGE` | $R_{\text{sense}}$ | 460.0 | Sensing & inter-UAV communication range (m) |
| `MBS_POS` | $p_{\text{MBS}}$ | [350.0, 350.0, 30.0] | Coordinates of central Macro Base Station (m) |
| `UAV_STORAGE_CAPACITY` | $S_u$ | $120 \sim 180 \times 10^6$ | Storage capacity per UAV (bytes) [STATIC_RANDOM] |
| `UAV_COMPUTING_CAPACITY` | $F_u$ | $40 \sim 90 \times 10^9$ | Computing frequency per UAV (cycles/s) [STATIC_RANDOM] |
| `MBS_COMPUTING_CAPACITY` | $F_{\text{mbs}}$ | $200 \times 10^9$ | Computing frequency of MBS (cycles/s) |
| `K_CPU` | $\kappa$ | $10^{-27}$ | CPU effective capacitance coefficient ($P = \kappa f^3$) |
| `BANDWIDTH_EDGE` | $B_{\text{edge}}$ | 40 MHz | Total access bandwidth per UAV |
| `BANDWIDTH_INTER` | $B_{\text{inter}}$ | 20 MHz | Inter-UAV cooperative bandwidth |
| `BANDWIDTH_BACKHAUL` | $B_{\text{backhaul}}$ | 750 kHz | UAV-to-MBS wireless backhaul bandwidth |
| `POWER_MOVE` / `POWER_HOVER` | $P_{\text{move}}, P_{\text{hover}}$ | 300 W / 150 W | UAV kinematic propulsion power |
| `UE_BATTERY_CAPACITY` | $B_{\max}$ | 500.0 J | Maximum UE battery capacity |
| `UE_CRITICAL_THRESHOLD` | $B_{\text{low}}$ | 50.0 J (0.10 $B_{\max}$) | Critical low battery threshold |
| `WPT_TRANSMIT_POWER` | $P^{\text{WPT}}$ | 50.0 W | UAV wireless power transfer transmit power |
| `WPT_EFFICIENCY` | $\eta$ | 0.80 | Energy harvesting efficiency |
| `NUM_SERVICES` / `NUM_CONTENTS` | $S, K$ | 25 / 50 | Service types / Content files (Total files = 75) |
| `ZIPF_BETA` | $\beta_{\text{Zipf}}$ | 0.80 | Request popularity skew parameter |
| `MAX_ASSOCIATED_UES` | - | 30 | Upper association observation slot limit |
| `MAX_OFFLOAD_REQUESTS_PER_UAV` | - | 30 | Lower offload action slot limit (Phase 3 target: 100) |
| `PPO_CLIP_EPS` | $\epsilon_{\text{clip}}$ | 0.20 | PPO surrogate clipping parameter |
| `PPO_GAE_LAMBDA` | $\lambda_{\text{GAE}}$ | 0.95 | Generalized Advantage Estimation parameter |
| `DISCOUNT_FACTOR` | $\gamma$ | 0.99 | Reinforcement learning discount factor |
| `ACTOR_LR` / `CRITIC_LR` | - | $3 \times 10^{-4}$ | Adam learning rates |

> [!WARNING]
> **Static Import-Time Random Variables (`STATIC_RANDOM_AT_IMPORT = YES`)**:
> In `config.py`, the following variables are evaluated at module import time using global `np.random`:
> - `UAV_STORAGE_CAPACITY`: `np.random.choice(..., size=5)`
> - `UAV_COMPUTING_CAPACITY`: `np.random.choice(..., size=5)`
> - `CPU_CYCLES_PER_BYTE`: `np.random.randint(1200, 2800, size=25)`
> - `FILE_SIZES`: `np.random.randint(10**6, 5*10**6, size=75)`
> - `AVG_FILE_SIZE`: `np.mean(FILE_SIZES)`
> In Phase 4, these must be frozen into deterministic scenario manifests to prevent seed leakage across imports.

---

## 15. Random Number Generator (RNG) Map

Randomness in the repository originates from five distinct functional areas:

```text
1. Import-time parameter generation (config.py: lines 48, 99, 101, 140, 142)
   └── Controlled by: np.random.seed(SEED) at import time.

2. Episode reset geometry (environment/env.py: line 172, user_equipments.py: lines 40, 63)
   └── Rejection sampling for UAV starting coordinates, hotspot coordinates, UE coordinates.
   └── Controlled by: Global NumPy RNG state at reset() call.

3. Request generation & UE mobility (environment/user_equipments.py)
   └── UE random-walk displacement, request type/id Zipf draws, file size & deadline draws.
   └── Controlled by: Global NumPy RNG state during env.step().

4. Exploration action sampling
   └── Continuous upper action: Gaussian distribution sampling (marl_models/attention.py).
   └── Discrete lower action: Categorical.sample() (marl_models/offload_mappo/offload_mappo.py).
   └── Controlled by: Global PyTorch RNG (torch.manual_seed).

5. Minibatch sampling in rollout buffer (marl_models/buffer_and_helpers.py: line 75)
   └── np.random.randint(0, size, batch_size) for PPO training updates.
```

- **Current Isolation Status**: Unit tests explicitly isolate seeds (`TEST_RNG_ISOLATED = YES`).
- **Pending Phase 4 Work**: Full experiment multi-seed RNG isolation (`CANONICAL_RNG_INFRASTRUCTURE_PENDING = YES`).

---

## 16. Results & Checkpoints Inventory

- **`results/` Directory**: 522 files, 20.76 MB total.
- **`saved_models/` Directory**: 144 files, 286.96 MB total.
- **`recent/` Directory**: 7,805 files, 3.73 GB total.
- **Status of Isolation**: `CANONICAL_RESULT_ISOLATION = NOT IMPLEMENTED`. Historical runs and legacy baseline outputs currently coexist in `results/`. Phase 4 will establish a clean, dedicated `results/canonical_clean/` target.
- **Legacy Status**: All existing checkpoints and results are designated **`HISTORICAL_ONLY`** under the Clean-Slate Principle.

---

## 17. Result Provenance Audit

Representative JSON sample audit (`results/baseline_matrix_force_admission_20260624/manifest.json`):

| Provenance Field | Present in Manifest? | Present in Metrics JSON? | Target Canonical Status |
| :--- | :---: | :---: | :--- |
| `git_commit_hash` | **YES** | NO | Mandatory in both |
| `git_branch` | **YES** | NO | Mandatory |
| `created_at` (timestamp) | **YES** | NO | Mandatory |
| `training_seed` / `workload_seed` | **YES** | **YES** | Mandatory |
| `force_service_admission` | **YES** (`true`) | NO | Hard-locked `false` in canonical |
| `config_snapshot` | **YES** | NO | Mandatory |
| `environment_semantics_version` | **NO** | **NO** | Mandatory in Phase 4 (`clean_v1`) |
| `scenario_hash` | **NO** | **NO** | Mandatory in Phase 4 |
| `checkpoint_path` | **NO** | **NO** | Mandatory in Phase 4 |

---

## 18. Checkpoint Compatibility Map

### Architecture Checkpoint Structure

- Upper Trajectory Model (`attention_mappo.pth`):
  - Keys: `actor`, `critic`, `actor_optimizer`, `critic_optimizer`, `metadata`.
  - State Dict: `nn.Module` weights for `CrossAttentionExtractor` and MLP heads.
- Lower Offloading Model (`offload_mappo.pth`):
  - Keys: `actor`, `critic`, `actor_optimizer`, `critic_optimizer`, `metadata`.
  - `OffloadCritic` contains: `layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))`.

### Capacity Expansion Impact ($30 \to 100$)
- In Phase 3, expanding `MAX_OFFLOAD_REQUESTS_PER_UAV` from 30 to 100 will expand `obs_dim` from 425 to 1405.
- The first linear layer of `OffloadCritic` will change input dimension from 425 to 1405.
- **Compatibility Verdict**: **Zero backward compatibility**. Any attempt to load legacy 30-request checkpoints into a 100-request architecture will raise `RuntimeError: size mismatch`. This confirms the necessity of clean-slate retraining in Phase 4.

---

## 19. LaTeX Thesis Structure (`latex/`)

### Document Map (`latex/main.tex`)

```text
main.tex
├── frontmatter (abstract.tex, notation.tex)
├── mainmatter
│   ├── docs/chap01.tex  (第1章 绪论)
│   ├── docs/chap02.tex  (第2章 相关技术简介)
│   ├── docs/chap03.tex  (第3章 无人机辅助 MEC 系统中的轨迹与任务卸载协同优化算法)
│   │   ├── Section 3.1: 系统模型与问题建模
│   │   ├── Section 3.2: 基于 Attention-MAPPO 的双层协同优化算法
│   │   ├── Section 3.3: 仿真设置与性能评估
│   │   └── Section 3.4: 本章小结
│   └── docs/chap06.tex  (第4章/第6章 总结与展望)
└── backmatter (complementary.tex, acknowledgements.tex, achievements.tex)
```

### Legacy Terminology Audit in Thesis Source
A regex audit of `latex/docs/*.tex` reveals:
- **“拉格朗日” (Lagrangian)**: Appears in `abstract.tex` (2), `chap01.tex` (3), `chap03.tex` (7), `chap06.tex` (3). (Must be aligned with adaptive subgradient penalty terminology in final thesis polishing).
- **“离线率” (Offline Rate)**: Appears 7 times in `chap03.tex`. (Must be verified against canonical battery depletion definition).
- **“强制接纳” / `force_admission`**: **0 occurrences** in LaTeX body (clean).
- **`C2`**: **0 occurrences** in LaTeX body (clean).

---

## 20. Legacy / Dead / Ambiguous Assets

| Asset Path | Nature | Classification | Action in Clean Pipeline |
| :--- | :--- | :--- | :--- |
| `recent/` | Historical runs, nohup logs, old backup configs (3.7 GB) | **CONFIRMED_UNREFERENCED** | Retain as historical archive; exclude from training |
| `saved_models/` | Checkpoints trained under legacy semantics (287 MB) | **CONFIRMED_UNREFERENCED** | Mark `HISTORICAL_ONLY`; do not load |
| `results/baseline_matrix_*` | Previous benchmark runs using forced admission | **CONFIRMED_UNREFERENCED** | Mark `HISTORICAL_ONLY`; do not cite |
| `run_linear_v2_*.sh` | Old shell wrappers calling matrix runners | **POSSIBLY_LEGACY** | Superseded by canonical runner in Phase 4 |
| `run_baseline_matrix_v2.py` | Matrix runner for legacy baselines | **POSSIBLY_LEGACY** | Update for canonical pipeline in Phase 4 |
| `run_hierarchical_mappo_experiment.py` | Active hierarchical training runner | **CONFIRMED_REFERENCED** | Core target for Phase 3/4 integration |
| `paths.py` / `config.py` | Global paths and configuration | **CONFIRMED_REFERENCED** | Active SSOT source |

---

## 21. Current Canonical Status Matrix

| Subsystem | Canonical Status | Current Evidence | Next Milestone |
| :--- | :---: | :--- | :--- |
| **SSOT Specification** | **FROZEN** | `CANONICAL_SEMANTICS_SPEC.md` locked at commit `2e96ae8b` | Maintain strictly read-only |
| **Environment Physics** | **GREEN** | 21/21 tests GREEN, `deterministic_smoke.py` PASS | Phase 2B complete (`clean_v1`) |
| **Service Admission** | **GREEN** | Natural coverage only ($d \le 100$m), forced admission disabled | Retain |
| **Batch Processor Sharing** | **GREEN** | $F_j / N_{\text{assigned}, j}$ uniform sharing, MBS load excluded | Retain |
| **Cache Dual-State Lifecycle** | **GREEN** | Read snapshot, queue pending, S7 deterministic commit | Retain |
| **Kinematics & Collision** | **GREEN** | Alternating projection, speed ball $\le 15$m, separation $\ge 200$m | Retain |
| **Metric Semantics** | **GREEN** | Full-system DSR denominator, Critical UE 50J, UAV-only fleet energy | Retain |
| **Lower Joint Logprob** | **NEEDS_PHASE3** | Dividing by `denom` (arithmetic mean); `test_joint_logprob_is_sum_not_mean` RED | **Phase 3 (LOWER-01)** |
| **Lower Reward Causality** | **NEEDS_PHASE3** | Polluted by upper flight energy in `metrics["energy"]`; test RED | **Phase 3 (LOWER-02)** |
| **Lower Offload Capacity** | **NEEDS_PHASE3** | Truncated at 30 requests; `test_no_silent_truncation` RED | **Phase 3 (LOWER-04)** |
| **Upper Trajectory MAPPO** | **STABLE** | Continuous 2D CrossAttention MAPPO | Retain for canonical training |
| **Joint MAPPO Baseline** | **STABLE** | Single actor continuous+discrete MAPPO baseline | Retain for canonical comparison |
| **Test Infrastructure** | **ACTIVE** | 26 tests: 23 GREEN, 3 Phase 3 Reserved RED, 0 Errors | Turn 3 REDs GREEN in Phase 3 |
| **RNG Infrastructure** | **PENDING** | Unit tests isolated; global scenario seed manager not yet integrated | Phase 4 Integration |
| **Experiment Provenance** | **PENDING** | Missing scenario hash and semantics version in manifest | Phase 4 Integration |
| **Results Isolation** | **PENDING** | Canonical clean results directory not yet isolated | Phase 4 Integration |
| **Ch4 Snapshot Evaluator** | **NOT IMPLEMENTED** | Standalone MC-ITR / Exact-Small evaluator absent | Future Research / Appendix |
| **LaTeX Synchronization** | **PENDING** | Awaiting clean-slate training data to regenerate figures & tables | Phase 5 Final Polish |

---

## 22. Known Remaining Work

### P0: Phase 3 Blockers (Lower MAPPO Canonical Repair)
1. **LOWER-01**: Repair joint action logprob calculation in `marl_models/offload_mappo/offload_mappo.py` from mean to exact product sum $\sum_{k=1}^K \log P(a_k)$.
2. **LOWER-02**: Decouple lower reward in `run_hierarchical_mappo_experiment.py:lower_rewards_from_metrics()` from upper flight/hover propulsion energy, restricting energy terms to service compute and transmission.
3. **LOWER-04**: Expand `config.MAX_OFFLOAD_REQUESTS_PER_UAV = 100` (equal to `NUM_UES`), eliminating silent truncation and heuristic fallback for requests 31–100.
4. **Regression Verification**: Execute remote test suite to confirm all 26 tests turn **26 / 26 GREEN (0 RED, 0 ERRORS)**.

### P1: Phase 4 Infrastructure Blockers (Canonical Experiment Pipeline)
1. **RNG Pipeline**: Implement `RNGManager` ensuring strict independence and reproducibility across training seeds and evaluation workload seeds.
2. **Result Isolation**: Establish `results/canonical_clean/` to permanently isolate new results from legacy historical runs.
3. **Automated Provenance Manifest**: Record git commit SHA, scenario hash, semantics version (`clean_v1`), and complete parameter dictionary in every output folder.

### P2: Later Experiment Execution & Thesis Synchronization
1. Retrain canonical models (Attention-MAPPO + Constrained-Attention-Offload-MAPPO) on remote server (`100.69.44.85`).
2. Train all baselines (Vanilla MAPPO, IPPO, Joint MAPPO, Uncoordinated Greedy, Random, Uniform).
3. Run statistical significance tests (paired t-test, Wilcoxon, 95% CI).
4. Re-compile thesis figures via `docs/figures/scripts/` and compile final PDF via `cd latex && make main`.

---

## 23. File Modification Boundary Verification

```text
Pre-Snapshot Commit: 4b9652a5ae1e12cc4b34902a2c7569e2fbe96335
Files Modified:      REPO_STATE_SNAPSHOT.md (ONLY)
Production Code:     STRICTLY UNCHANGED (0 edits)
Test Files:          STRICTLY UNCHANGED (0 edits)
SSOT Specification:  STRICTLY UNCHANGED (0 edits)
```

---

## 24. Architectural State Block

```text
CURRENT_HEAD = 4b9652a5ae1e12cc4b34902a2c7569e2fbe96335
WORKTREE_CLEAN = YES

SSOT_FROZEN = YES
ENV_SEMANTICS_VERSION = clean_v1
ENVIRONMENT_STATUS = GREEN

LOWER_LOGPROB_STATUS = NEEDS_PHASE3
LOWER_REWARD_STATUS = NEEDS_PHASE3
LOWER_CAPACITY_STATUS = NEEDS_PHASE3

CANONICAL_RNG_INFRASTRUCTURE = PENDING
CANONICAL_RESULT_ISOLATION = PENDING
CANONICAL_PROVENANCE_PIPELINE = PENDING
CH4_FIXED_SNAPSHOT_EVALUATOR = NOT_IMPLEMENTED

FORMAL_CANONICAL_TRAINING_STARTED = NO

READ_ONLY_VIOLATION = NO

STATUS = REPOSITORY_STATE_CAPTURED
```
