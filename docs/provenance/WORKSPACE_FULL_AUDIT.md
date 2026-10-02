# WORKSPACE_FULL_AUDIT.md

- **Audit Date**: 2026-09-19 (refreshed full re-verification)
- **Supersedes**: 2026-09-18 version of this file (same deliverable, re-audited from evidence)
- **Auditor**: Static read-only audit (ZCode agent)
- **Audit Mode**: STRICT READ-ONLY — no production source, tests, LaTeX, results, checkpoints, or SSOT modified
- **Repository**: `/Users/wangpengfei/Documents/毕业论文/我的论文/wodexuexi`
- **Branch**: `双层注意力MAPPO多智能体强化学习`
- **HEAD**: `4b9652a5ae1e12cc4b34902a2c7569e2fbe96335`
- **SSOT Status**: FROZEN (`CANONICAL_SEMANTICS_SPEC.md`)
- **Environment Semantics**: `clean_v1` (GREEN)
- **Formal Canonical Training Started**: NO

> **Refresh note (2026-09-19)**: Every material claim of the 2026-09-18 audit was re-verified against source with exact `file:line` evidence. All previously reported findings were confirmed unchanged. Three corrections/refinements were made: (1) the constraint-penalty violation signals are also system-scope, folded into NEW-02; (2) the portability claim was corrected — hardcoded `/home/PengYanghan/` paths do exist in legacy auxiliary scripts; (3) former NEW-15 (empty `reports/` at root) no longer applies and was replaced by the corrected portability finding. See §20 and Appendix A.

---

## 1. Executive Summary

This repository implements a multi-UAV assisted Mobile Edge Computing (MEC) multi-agent reinforcement learning system for a master's thesis at Sun Yat-sen University. The project has progressed through Phase 1 (Clean-Slate Audit), Phase 1.5 (SSOT Freeze), Phase 2A (TDD RED Baseline), and Phase 2B (Environment GREEN Repair). The environment physics layer (`clean_v1`) is verified GREEN. The next milestone is Phase 3: Lower MAPPO Canonical Repair.

The audit re-confirms the three known Phase 3 defects (LOWER-01 logprob MEAN→SUM, LOWER-02 reward energy contamination, LOWER-04 capacity 30→100) with exact file:line evidence. It also confirms **16 findings** beyond the known defects (0 P0, 4 P1, 8 P2, 4 P3) — most notably a Joint-MAPPO trajectory-action clamp inconsistency that biases PPO ratios, a system-level-vs-admitted-domain reward scope mismatch that also contaminates the constraint-penalty signals, and static import-time RNG coupling that freezes UAV heterogeneity parameters across all training seeds.

No P0 new findings were detected. The workspace is **READY_FOR_PHASE3** with pre-Phase-3 hygiene recommendations.

---

## 2. Project Understanding (PROJECT_UNDERSTANDING)

### Research Goal

The system optimizes a multi-UAV assisted MEC network: 5 UAVs, 100 UEs, and 1 MBS operate in a 700×700m area. The objective is to jointly optimize UAV trajectory control (where drones fly) and request-level task offloading (where each service request is computed — local UAV, cooperative UAV, or MBS) under constraints of latency, energy, fairness, deadline satisfaction rate (DSR), and MBS load.

### Physical Environment

- **UAVs (5)**: Fixed altitude 100m, max speed 15 m/s, coverage radius 100m, sensing range 460m, min separation 200m (`config.py:95-113`). Carry computing capacity, storage cache, and perform WPT. Energy components: flight, hover, compute, communication, cache fetch, content download, WPT.
- **UEs (100)**: Ground devices with battery dynamics and critical threshold. Generate service/content/energy requests via Zipf distribution.
- **MBS (1)**: Central macro base station; grid-powered computing explicitly excluded from UAV Fleet Total Energy (SSOT §7.1.2).
- **Communication links**: UE-UAV access (slot-start static bandwidth slice `B_edge / N_assoc`), UAV-UAV cooperative, UAV-MBS backhaul; free-space path loss.
- **Cache**: Per-UAV file storage (75 files = 25 services + 50 contents), GDSF eviction, dual-state snapshot/pending → end-slot atomic commit with deterministic priority + file_id tie-break.

### MARL Architecture

| Algorithm | Directory | Role | Status |
| :--- | :--- | :--- | :--- |
| **Attention-MAPPO** (Upper) | `marl_models/attention_mappo/` | Trajectory control, multi-head cross-attention over neighbors+UEs | STABLE |
| **Offload-MAPPO** (Lower) | `marl_models/offload_mappo/` | Request-level discrete offloading | NEEDS_PHASE3 |
| **Joint-MAPPO** | `marl_models/joint_mappo/` | End-to-end single-actor continuous+discrete baseline | STABLE (with defects) |
| **Vanilla-MAPPO** | `marl_models/vanilla_mappo/` | MLP ablation (no attention) | STABLE |
| **IPPO** | `marl_models/ippo_baseline/` | Independent PPO (decentralized critic) | STABLE |
| **Random / Uniform / Greedy** | `marl_models/random_baseline/`, `uniform_baseline/`, `uncoordinated_greedy_baseline/` | Non-learning baselines | STABLE |

### Current Phase

- **Fixed (Phase 2B, GREEN)**: Environment physics — slot-start snapshot, natural admission, batch final-load equal sharing, cache dual-state lifecycle, joint-feasibility collision repair, temporal alignment, metric semantics.
- **Pending (Phase 3)**: Lower MAPPO canonical repair — logprob SUM, reward causality + admitted-domain scope, capacity 30→100. Joint-MAPPO probability-measure alignment.
- **Pending (Phase 4)**: Canonical RNG infrastructure, provenance manifests, result isolation, 5×10 statistical protocol, formal training.

### Canonical / Legacy Boundary

- **Trusted (Canonical)**: `CANONICAL_SEMANTICS_SPEC.md` (frozen SSOT), `environment/` (`clean_v1` GREEN), `config.py`, `paths.py`, `tests/` (23 GREEN + 3 expected Phase 3 RED), `marl_models/` source (algorithmically stable, pending Phase 3 repair).
- **Historical Only**: `results/` (519 tracked JSON files, generated under `force_admission`), `saved_models/` (144 checkpoints, obs_dim 425 vs canonical 1405 — dimension-incompatible), `train_logs/`, `recent/` (3.73GB archived legacy runs incl. dead algorithm implementations), `logs/`. All existing experimental data is `HISTORICAL_ONLY` and must not be cited as final results.

```
PROJECT_CONTEXT_UNDERSTOOD = YES
```

---

## 3. Git / Workspace State

```text
CURRENT_BRANCH: 双层注意力MAPPO多智能体强化学习
CURRENT_HEAD:   4b9652a5ae1e12cc4b34902a2c7569e2fbe96335
PACKED_OBJECTS: 20,053 (3 packs, 2.87 GiB); 10 loose objects (84 KiB)
TRACKED FILES:  934
```

**Worktree status**: Not clean. Staged changes are Wave-1 hygiene only: documentation reorganization (`CLEAN_SLATE_AUDIT.md`/`TDD_RED_BASELINE.md` → `docs/provenance/`, `ENV_REPAIR_REPORT.md` → `docs/reports/`), new reports (`REPO_CLEANUP_PLAN.md`, `REPO_STATE_SNAPSHOT.md`), `.gitignore` update, 11 `.DS_Store` deletions, deletion of `.codex-vscode-write-test`. **No production source file is modified.**

### Untracked Items Classification

| Item | Classification | Rationale |
| :--- | :--- | :--- |
| `WORKSPACE_FULL_AUDIT.md` | EXPECTED_UNTRACKED | This audit deliverable |
| `docs/figures/prism_source_data/` | EXPECTED_UNTRACKED | GraphPad Prism source tables (CSV/TSV/.prism) for thesis figures |
| `latex/main_compressed.pdf` | GENERATED | Compressed PDF build artifact |
| `output/` (12 files) | EXPECTED_UNTRACKED | Presentation slide assets |
| `tools/export_prism_source_tables.py` | SUSPICIOUS_UNTRACKED | Active utility script not yet tracked (also flagged in REPO_CLEANUP_PLAN) |
| `.tmp/nature-skills-inspect` | SUSPICIOUS_UNTRACKED | Tracked as **gitlink mode 160000** (`git ls-files -s .tmp`), no `.gitmodules`; nested repo content shows ` m` in status |

---

## 4. Repository Inventory

### Directory Size and File Inventory

| Path | Size | Files | Git Tracked | Purpose | Status |
| :--- | ---: | ---: | ---: | :--- | :--- |
| `.git/` | 2.9 GB | 20,053 objects | internal | Version database | Bloat from duplicated TIFF figures |
| `recent/` | 3.73 GB | 7,805 | NO (gitignored) | Archived legacy experiments, dead algorithms (`attention_maddpg`, `attention_masac`…), `README*.bak` | HISTORICAL_ARCHIVE |
| `latex/` | 525 MB | 258 | YES | Thesis LaTeX source & compiled assets | ACTIVE (large TIFF figures duplicated across `docs/figures/` and `latex/docs/figures/`) |
| `saved_models/` | 287 MB | 144 | NO (gitignored) | Historical model checkpoints | TOXIC_HISTORICAL_ONLY |
| `.venv/` | 204 MB | — | NO | Linux x86_64 virtual environment | LEGACY (server-only; never execute locally) |
| `.plot-venv/` | 78 MB | — | NO | Local plotting venv | AUXILIARY |
| `docs/` | 75 MB | 73 | YES | Reports, figures, scripts | ACTIVE |
| `.tmp/` | 41 MB | — | PARTIAL (1 gitlink) | Transient scratch | DEAD_ARTIFACT |
| `train_logs/` | 37 MB | 301 | NO | Historical training logs | HISTORICAL_ONLY |
| `results/` | 21 MB | 522 | YES (519) | Historical experiment JSON | HISTORICAL_ONLY |
| `output/` | 13 MB | 12 | NO | Presentation assets | AUXILIARY |
| `marl_models/` | 608 KB | 70 | YES | RL agents & policies | CANONICAL_PRODUCTION |
| `environment/` | 292 KB | 15 | YES | Simulation physics (clean_v1) | CANONICAL_PRODUCTION |
| `scripts/` | 164 KB | 12 | YES (7) | Analysis/plotting/legacy scripts | MIXED |
| `utils/` | 160 KB | 9 | YES | Logger, metrics helpers | CANONICAL_PRODUCTION |
| `tests/` | 92 KB | 15 | YES | TDD suite (26 methods) | CANONICAL_TEST |
| `logs/` | 92 KB | 19 | NO | Force-admission nohup logs | HISTORICAL_ONLY |
| Root scripts | ~120 KB | 16 | YES | Runners + legacy reference tools | MIXED (see §13) |

### Hidden Directory Audit

| Directory | Contents | Status |
| :--- | :--- | :--- |
| `.git/` | 2.87 GiB packs; largest blobs 51.7 / 48.6 / 34.2 / 31.8 MB (`git verify-pack -v`) | Large — TIFF figure duplicates |
| `.vscode/` | `settings.json` (watcher/search excludes) | Machine-specific IDE config, benign |
| `.claude/` | Superpowers skill files; `.DS_Store` staged for deletion | Agent instructions |
| `.spec-workflow/` | 8 template files | Scaffolding, unused |
| `.tmp/` | `nature-skills-inspect/` nested git repo | Tracked as gitlink 160000, no `.gitmodules` |
| `.venv/` | Linux x86_64 Python | LEGACY — must not execute locally |
| `.plot-venv/` | Local plotting venv | AUXILIARY |
| `.codex/`, `.github/` | Not present | — |
| `__pycache__/` | 10 `.pyc` (cpython-312 + cpython-39 mixed) | GENERATED, gitignored |

---

## 5. Canonical vs Legacy Boundary

### Trusted Assets

| Asset | Evidence |
| :--- | :--- |
| `CANONICAL_SEMANTICS_SPEC.md` | Frozen SSOT (Errata Pass closed; §0 decision locks verified against prompt §2) |
| `environment/` (5 production files) | Phase 2B GREEN; 9 SSOT contracts statically MATCH (§6) |
| `config.py`, `paths.py` | Canonical configuration; `FORCE_SERVICE_ADMISSION=False` (`config.py:82`) |
| `tests/` (15 files, 26 methods) | 23 GREEN + 3 expected Phase 3 RED |
| Active runners | `run_hierarchical_mappo_experiment.py`, `run_joint_end_to_end_mappo_experiment.py` |

### Historical-Only Assets

| Asset | Size | Why Legacy |
| :--- | ---: | :--- |
| `results/` (6 subdirs, 519 tracked JSON) | 21 MB | Generated under `force_admission`, capacity=30, pre-repair energy accounting |
| `saved_models/` (144 checkpoints) | 287 MB | Trained with obs_dim=425 (5+30×14); canonical requires 1405 (5+100×14) → `RuntimeError: size mismatch` on load |
| `train_logs/` (301) | 37 MB | Legacy training logs |
| `recent/` (7,805) | 3.73 GB | Archived pre-defense iterations, dead algorithms (MADDPG/MASAC/etc.), backup files |
| `logs/` (19) | 92 KB | Force-admission nohup stdout |

### Contamination Risk Assessment

**Current runners do NOT load legacy checkpoints by default.** Both current runners create fresh models via `get_model()` (`run_hierarchical_mappo_experiment.py:150,163`); no `torch.load`/`load_state_dict` of pre-existing checkpoints exists in any runner. `config.SERVICE_OFFLOAD_POLICY_CHECKPOINT` defaults to `None` (`config.py:180`) and is only consumed by the standalone offload-policy path (`marl_models/offload_policy.py:693-697`), which raises if unset — no silent legacy load. No "latest checkpoint" auto-selection exists. **No P0 contamination found.**

Residual risk (P2): new runs write into the same top-level directories as legacy assets — `saved_models/{model}_{timestamp}/`, `results/reports/hierarchical_mappo_summary_{timestamp}.json` (`run_hierarchical_mappo_experiment.py:385,390,418`), `train_logs/hierarchical_mappo/` (`:168`) — so new and legacy artifacts coexist without canonical/historical isolation.

---

## 6. Environment Integrity

SSOT contracts statically re-verified against production source:

| SSOT Contract | Code Location | Classification | Evidence |
| :--- | :--- | :--- | :--- |
| Slot-start snapshot | `env.py:35` `SlotSnapshot`, `:99` `_capture_slot_snapshot`, `:129` `cache_snapshots = uav.cache.copy()` | **MATCH** | Frozen immutable snapshot captured before decisions |
| Natural admission (no force) | `env.py:530-560` `_associate_ues_to_uavs()`; `config.py:82` | **MATCH** | Only `distance <= UAV_COVERAGE_RADIUS` admitted; uncovered → `ue.assigned = False`; forced admission permanently disabled |
| Batch final-load equal sharing | `uavs.py` `process_requests()` | **MATCH** | `f_mj = F_j / N_assigned,j`; per-UAV `service_compute_energy` accumulators reset at `env.py:212-214` |
| Cache snapshot→pending→commit | `env.py:129-131,205`; `uavs.py` dual-state cache | **MATCH** | Read snapshot during slot, queue pending, end-slot deterministic commit |
| Same-transition movement energy | `env.py` S4→S5→S6 ordering | **MATCH** | Verified by `test_env_temporal_alignment.py` (GREEN) |
| Collision joint feasibility | `env.py` alternating projections, `COLLISION_AVOIDANCE_ITERATIONS=20` (`config.py:117`) | **MATCH** | Speed ball + separation + boundary + non-negative hover; verified by `test_collision_joint_feasibility.py` (GREEN) |
| System DSR denominator | `env.py:578,602` `total_service_requests_generated` over ALL UEs | **MATCH** | Includes unadmitted requests (comment at `env.py:595` explicit) |
| Critical-UE Ratio | `env.py:574-576` `battery_level < UE_CRITICAL_THRESHOLD` | **MATCH** | Code key `offline_rate` retained for compat (sanctioned by SSOT §7.1.4) |
| UAV Fleet Total Energy | `env.py:653` `sum(uav.energy)` | **MATCH** | MBS grid energy excluded; breakdown keys exported (`env.py:655-663`) |

**Environment verdict**: `ENVIRONMENT_STATUS = GREEN`. All 9 frozen contracts MATCH. No contradictions found.

---

## 7. Upper MAPPO Audit

### Observation Dimensions — Correct

```
OBS_DIM_SINGLE = SELF_OBS_DIM(77) + MAX_UAV_NEIGHBORS(4) × NEIGHBOR_OBS_DIM(2) + MAX_ASSOCIATED_UES(30) × UE_OBS_DIM(8) = 325
```

`config.py:286` defines the formula; `SELF_OBS_DIM = 2 + NUM_FILES(75) = 77`; `MAX_ASSOCIATED_UES = min(30, 100//5+10) = 30` (`config.py:124`); `UE_OBS_DIM = 8`.

### MAX_ASSOCIATED_UES = 30 — Observation Budget Only

**Confirmed**: the 30-cap limits which UEs populate upper attention tokens, NOT physical admission or service processing.

- Physical admission admits ALL covered UEs (`env.py:559` `best_uav.current_covered_ues.append(ue)` — no cap).
- Upper observation truncates only the token list: `env.py:409-410` `ues = sorted(...)[: config.MAX_ASSOCIATED_UES]`.
- All admitted requests are physically served; only learned-decision visibility is capped.

### Attention & Actor/Critic Consistency — Correct

- 8 heads, hidden 64 (`config.py` ATTN_*); boolean mask semantics match PyTorch 2.6 SDPA; all-padding fallback via `nan_to_num`.
- `attention_mappo.py:70,75`: unclipped `dist.sample()` stored; `log_prob(sample).sum(dim=-1)` (SUM over 2D dims) used identically in update — **no clamp inconsistency**. Vanilla (`vanilla_mappo.py:40,42`) and IPPO (`ippo_baseline.py:54,57`) equally consistent.

---

## 8. Lower MAPPO Audit

### Pipeline Trace

| Stage | File:Function | Shape | Config Dependency |
| :--- | :--- | :--- | :--- |
| Request generation | `user_equipments.py:generate_request()` | per-UE | `NUM_UES=100`, Zipf |
| Admission | `env.py:_associate_ues_to_uavs()` | per-UAV list | `UAV_COVERAGE_RADIUS=100` |
| Obs construction | `env.py:get_offloading_obs_and_masks()` | `(5, 425)` | `MAX_OFFLOAD_REQUESTS_PER_UAV=30` |
| Truncation | `env.py:290` `service_ues[: config.MAX_OFFLOAD_REQUESTS_PER_UAV]` | ≤30 requests | **LOWER-04 site** |
| Padding | zero-init obs/mask arrays | `(5,30,14)` obs, `(5,30,7)` mask | unused slots all-zero |
| Actor forward | `offload_mappo.py:OffloadActor.forward()` | logits `(5,30,7)` | `OFFLOAD_NUM_ACTIONS=7` |
| Action sampling | `Categorical(logits).sample()` | per-slot | — |
| Logprob | `offload_mappo.py:243-246` | scalar per agent | **LOWER-01: `/ denom` MEAN** |
| Rollout storage | `buffer_and_helpers.py:DiscreteOffloadRolloutBuffer` | `(1000, 5, *)` | buffer_size |
| Target resolution | `uavs.py:process_requests()` Phase A | per-request | **LOWER-04 cursor gate `uavs.py:423`** |
| Reward | `run_hierarchical_mappo_experiment.py:70-94` | scalar per agent | **LOWER-02 + NEW-02** |
| Advantage | GAE | `(1000, 5)` | γ=0.99 (`config.py:297`), λ=0.95 (`config.py:335`) |
| PPO update | `offload_mappo.py:update()` | minibatches | clip 0.2, grad-norm 0.5, entropy 0.005 |

### Known Defects (Re-confirmed with Evidence)

**LOWER-01 — Joint logprob uses MEAN, not SUM** (KNOWN / ALREADY_PLANNED)

- `offload_mappo.py:246` (`get_action_and_value`): `log_probs = action_log_probs.sum(dim=-1) / denom`
- `offload_mappo.py:284` (`update`): `new_log_probs = slot_log_probs.sum(dim=-1) / denom`
- `offload_mappo.py:285` (entropy): same `/ denom` pattern
- SSOT §6.1 requires `Σ log π_m` with **no division by valid_count**
- Test: `tests/test_lower_logprob_contract.py` `test_joint_logprob_is_sum_not_mean` — expected RED

**LOWER-02 — Lower reward energy includes UAV propulsion** (KNOWN / ALREADY_PLANNED)

- `run_hierarchical_mappo_experiment.py:78`: `energy_term = float(metrics["energy"]) / (config.NUM_UAVS * config.OFFLOAD_ENERGY_NORM_REF + config.EPSILON)`
- `metrics["energy"] = total_energy = sum(uav.energy)` (`env.py:569,653`) = flight + hover + WPT + service compute/comm/fetch + content energy
- SSOT §7.2.1 requires `ASSIGNMENT_CAUSAL_NONFLIGHT` (compute + comm + fetch only)
- The causal breakdown already exists in metrics (`service_compute_energy`, `service_comm_energy`, `service_fetch_or_backhaul_energy` exported at `env.py:660-662`) — the fix is to sum those three instead of `metrics["energy"]`
- Test: `tests/test_lower_reward_causality.py` — expected RED

**LOWER-04 — Silent capacity truncation at 30** (KNOWN / ALREADY_PLANNED)

- `config.py:124`: `MAX_ASSOCIATED_UES = min(30, NUM_UES // NUM_UAVS + 10)`
- `config.py:186`: `MAX_OFFLOAD_REQUESTS_PER_UAV = MAX_ASSOCIATED_UES` (=30)
- `env.py:290`: obs/mask request list truncated to 30
- `uavs.py:423`: execution gate `service_action_cursor < int(config.MAX_OFFLOAD_REQUESTS_PER_UAV)`
- `uavs.py:434-442`: `else` branch silently routes requests 31+ to heuristic `_select_service_offloading_target`
- SSOT §8.1 requires `MAX_OFFLOAD_REQUESTS_PER_UAV = NUM_UES = 100`, `SILENT_TRUNCATION_ALLOWED = NO`
- Test: `tests/test_no_silent_truncation.py` — expected RED

### NEW-02 — Reward-Domain Scope Mismatch (NEW_FINDING, P1)

SSOT §7.2.2 defines lower reward terms `R_dsr`, `R_lat`, `R_mbs` over the admitted set `M`; SSOT §3.2 (last bullet) forbids unadmitted-request latency/timeout in the lower reward. The code feeds **system-level** metrics into the lower reward:

- `latency_term` (`run_hierarchical_mappo_experiment.py:77`) ← `metrics["latency"]` which includes `NON_SERVED_LATENCY_PENALTY` (20s) for every unadmitted UE (`env.py:566-568`) — **mixes unadmitted latency**
- `deadline_penalty` / `success_term` (lines 75, 81) ← system DSR (denominator = all generated requests, `env.py:602`)
- `mbs_term` (line 79) ← `metrics["mbs_load_ratio"]` = `total_mbs_offloads / total_generated` (system, `env.py:603`)
- **Refinement vs 2026-09-18 audit**: the constraint-violation signals are ALSO system-scope — `compute_constraint_diagnostics` (`run_hierarchical_mappo_experiment.py:59-60`) uses `metrics["deadline_satisfaction_rate"]` and `metrics["mbs_load_ratio"]`, while SSOT §7.3 defines the penalty over `DSR^lower`/`MBS^lower` (admitted domain). The constraint term shares NEW-02's root cause and must be repaired in the same change, not treated as separately-GREEN.

The upper reward (`env.py:617-637`) correctly uses system metrics (appropriate for trajectory control). Contamination is confined to the lower reward path.

- **Fix phase**: PHASE3 (same function `lower_rewards_from_metrics` as LOWER-02)

### Capacity Constant Classification

| Reference | File:Line | Classification |
| :--- | :--- | :--- |
| `NUM_UES = 100` | `config.py:65` | Total UE pool (source of SSOT capacity target) |
| `MAX_ASSOCIATED_UES = min(30, …)` | `config.py:124` | **UPPER_OBSERVATION_CAP** (token budget; no physical truncation) |
| `MAX_OFFLOAD_REQUESTS_PER_UAV = MAX_ASSOCIATED_UES` | `config.py:186` | **LOWER_POLICY_CAP** (aliases 30; gates obs slice + execution cursor) |
| `OFFLOAD_OBS_DIM_SINGLE = 5 + 30×14 = 425` | `config.py:193` | LOWER_POLICY_CAP (derived) |
| `OBS_DIM_SINGLE` (uses MAX_ASSOCIATED_UES) | `config.py:286` | UPPER_OBSERVATION_CAP |
| `service_ues[:MAX_OFFLOAD_REQUESTS_PER_UAV]` | `env.py:290` | LOWER_POLICY_CAP (truncation site) |
| `service_action_cursor < MAX_OFFLOAD_REQUESTS_PER_UAV` | `uavs.py:423` | LOWER_POLICY_CAP (execution gate / fallback trigger) |
| `max_requests` param | `buffer_and_helpers.py`, `offload_mappo.py:18-202` | LOWER_POLICY_CAP (storage/actor shapes) |
| `max_requests = config.MAX_OFFLOAD_REQUESTS_PER_UAV` | `joint_mappo.py:21` | LOWER_POLICY_CAP (joint baseline) |
| `ues[:MAX_ASSOCIATED_UES]` | `env.py:410` | UPPER_OBSERVATION_CAP (observation only) |

---

## 9. Joint-MAPPO Audit

### Architecture

- Single actor outputs continuous 2D trajectory (Normal, tanh-squashed mean) + discrete offload (30 independent Categoricals over 7 actions); joint obs = 325 + 425 = 750; `max_requests = 30` (`joint_mappo.py:21`).

### Offload Branch Logprob — MEAN (Shared LOWER-01)

- `joint_mappo.py:62` (`get_action_and_value`) and `:100` (`update`): `offload_log_probs = offload_slot_log_probs.sum(dim=-1) / denom`
- **Definitive answer: offload branch uses MEAN** (sum divided by valid count). Joint logprob = trajectory SUM(2D) + offload MEAN (`:63`, `:101`); entropy likewise (`:102-104`).
- Old and new offload formulas are identical (no intra-branch actor/update mismatch).

### NEW-01 — Trajectory-Action Clamp Inconsistency (NEW_FINDING, P1) — Re-confirmed

`old_log_probs` are computed on the **unclipped** sampled trajectory action, but `new_log_probs` on the **clipped** stored action:

- `joint_mappo.py:52-53`: `trajectory_actions = trajectory_dist.sample()` (unclipped); `:57` logprob computed on unclipped sample; stored as `old_log_probs`
- `joint_mappo.py:66`: returned/stored trajectory action is `torch.clamp(trajectory_actions, -1.0, 1.0)` — **clipped**
- `joint_mappo.py:75,97`: `update` computes `trajectory_dist.log_prob(trajectory_actions_batch)` on the clipped stored action
- Consequence: `ratio = exp(new − old) ≠ 1` under unchanged weights whenever a sample fell outside [−1,1] (common — Normal std starts ≈1.0). This biases the PPO ratio for the Joint baseline.
- Not present in upper/vanilla/IPPO (all store and evaluate unclipped actions consistently).
- **Fix phase**: PHASE3 (Joint baseline validity affects comparisons)

---

## 10. Rewards & Constraints

### Reward Components Map

**Upper/system reward** (`env.py:617-637`): Fairness (JFI, ×`REWARD_W_FAIR`), system latency (×`REWARD_W_LAT`), fleet energy (×`REWARD_W_ENERGY`), Critical-UE ratio (`offline_rate`, ×`REWARD_W_OFFLINE`), system DSR (×`REWARD_W_DSR`), + MBS load penalty `0.5 × mbs_load_ratio`. Scope: system-level — correct for trajectory policy.

**Lower reward** (`run_hierarchical_mappo_experiment.py:70-94`):

| Component | Metric Source | Code Scope | SSOT Required Scope |
| :--- | :--- | :--- | :--- |
| Success (DSR) | `metrics["deadline_satisfaction_rate"]` | System (all generated) | Admitted set M |
| Coop ratio | `metrics["offloading_ratio_cooperative"]` | Admitted | Admitted — OK |
| Deadline penalty | `1 − system DSR` | System | Admitted set M |
| Latency | `metrics["latency"] / (NUM_UES × T_penalty)` | System (incl. unadmitted 20s) | Admitted set M |
| **Energy** | `metrics["energy"] / (NUM_UAVS × E_ref)` | **System (incl. flight+hover+WPT)** | **Assignment-causal non-flight** |
| MBS load | `metrics["mbs_load_ratio"]` | System | Admitted set M |
| Constraint | `λ_dsr·dsr_violation + λ_mbs·mbs_violation` | System | Admitted-domain (`DSR^lower`, §7.3) |

Broadcast: `[float(lower_reward)] * len(system_rewards)` (line 94) → GLOBAL_TEAM scalar; matches SSOT `LOWER_REWARD_SCOPE = GLOBAL_TEAM` (credit dilution is a documented method limitation, SSOT §13.6).

**Joint-MAPPO reward** (`run_joint_end_to_end_mappo_experiment.py`): uses `system_rewards` from `env.step` directly (single joint policy) — appropriate.

No accidental reward mixing beyond the documented lower-path issues: flight/hover/WPT enter only via `metrics["energy"]` (LOWER-02); content energy and MBS grid energy never enter either reward.

### Constraint Mechanism Audit

**Update rule is semantically correct** (violation-driven, monotone non-decreasing, clipped):

```python
# run_hierarchical_mappo_experiment.py:362-364
lambda_dsr = float(np.clip(lambda_dsr + config.OFFLOAD_LAGRANGE_LR * mean_dsr_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))
lambda_mbs = float(np.clip(lambda_mbs + config.OFFLOAD_LAGRANGE_LR * mean_mbs_violation, 0.0, config.OFFLOAD_LAGRANGE_MAX))
```

Matches SSOT §7.3: `ρ_{k+1} = clip(ρ_k + η·v_k, 0, ρ_max)` with η=`OFFLOAD_LAGRANGE_LR=0.1`, ρ_max=`OFFLOAD_LAGRANGE_MAX=10.0` (`config.py:206-207`); targets `OFFLOAD_DSR_TARGET=0.18`, `OFFLOAD_MBS_LOAD_CEILING=0.03` (`config.py:204-205`). NOT a true Lagrangian dual ascent (no bidirectional swing, no saddle-point claim) — consistent with the frozen mechanism name.

**Terminology mismatches** (all TERMINOLOGY_ONLY; local-variable `lambda_*` names are sanctioned by SSOT §7.3.4):

| Layer | Code Name | SSOT Canonical | Classification |
| :--- | :--- | :--- | :--- |
| Config | `OFFLOAD_CONSTRAINT_MODE = "lagrange"` | Violation-driven | TERMINOLOGY_ONLY |
| Config | `OFFLOAD_LAGRANGE_LR`, `OFFLOAD_LAGRANGE_MAX` | η, ρ_max | TERMINOLOGY_ONLY |
| Code var | `lambda_dsr`, `lambda_mbs` (local) | ρ_dsr, ρ_mbs | LEGACY_ONLY (sanctioned) |
| Log/report keys | `final_lambda_dsr/mbs` | ρ_dsr, ρ_mbs | TERMINOLOGY_ONLY (risk of leaking into paper) |
| CLI | `--constraint_mode` ∈ {none, lagrange} | Violation-driven | TERMINOLOGY_ONLY |

No `rho_dsr`/`rho_mbs` identifiers exist anywhere in code. The **scope** of the violation signals is addressed under NEW-02 (not a separate mechanism defect).

---

## 11. Action Masks & Fallbacks

### Mask Architecture

Single merged 3D mask `all_masks[NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV, OFFLOAD_NUM_ACTIONS]` (`env.py:get_offloading_obs_and_masks()`). Three conceptual categories are NOT represented separately:

1. **Padding mask**: slots beyond `service_ues` remain all-zero via zero-init — implicit, not explicit.
2. **Hard feasibility**: cooperative actions set only for UAVs in `uav.neighbors` (loop at `env.py:332-349` never touches non-neighbor slots); Local always 1.0 (`:325`); MBS initially 1.0.
3. **Quality pruning**: cooperative gated by `coop_feasible` (deadline ratio, relative latency, compute-share ratio — `env.py:344-352`); MBS gated by latency-ratio clip (`:355-359`); both config-toggled via `OFFLOAD_MASK_MODE == "quality"`.

### No-Quality-Mask Ablation Safety

With `OFFLOAD_MASK_MODE != "quality"`: the `coop_feasible` branch passes all neighbors (`env.py:353`), MBS gating is skipped — but non-neighbor coop slots remain 0 and the all-masked safety net (`env.py:370-371` force LOCAL=1.0) still applies.

**Verdict: PARTIALLY_READY.** Today's No-Quality-Mask ablation safely disables quality pruning without enabling physically impossible actions. However hard feasibility and quality pruning are merged in one array with no explicit separation; a future clean ablation should compose `final = hard_mask & quality_mask & padding_mask`.

### Silent Fallback Audit

| Fallback | Location | Trigger | Logged/Counted | Classification |
| :--- | :--- | :--- | :--- | :--- |
| Heuristic offload for requests 31+ | `uavs.py:434-442` | `service_action_cursor >= 30` | Counted (`_service_request_count`) but decisions are heuristic and reward-visible | **CAPACITY_FALLBACK** (LOWER-04) |
| Requested coop UAV not a neighbor | `uavs.py:497-502` | Invalid coop target | `_service_fallback_count += 1` | SAFETY_FALLBACK (counted) |
| All actions masked on a real request | `env.py:370-371` | Full mask row | `coop_masked_count`-adjacent counters | SAFETY_FALLBACK (force LOCAL) |
| Attention all-padding NaN | `offload_mappo.py:60-63,72` | Zero valid slots | `nan_to_num` + unmask slot 0 | SAFETY_FALLBACK |

The CAPACITY_FALLBACK at `uavs.py:434` is the most significant: it silently bypasses the learned policy for requests beyond 30 while those heuristic decisions still feed the global reward (credit contamination). This is LOWER-04.

---

## 12. RNG / Reproducibility

### RNG Dependency Map

| Functional Area | RNG Source | Seeding | Isolation |
| :--- | :--- | :--- | :--- |
| Static import-time params | `np.random` in `config.py` (`np.random.seed(SEED)` at `config.py:48`) | Import-time only | **FROZEN — not re-seeded per train/workload seed** |
| Episode reset geometry / UE placement | Global `np.random` | `set_global_seed(seed)` (`run_hierarchical_mappo_experiment.py:19-25`) | Per-seed but order-coupled |
| Request generation / mobility / hotspots | Global `np.random` (same stream) | — | Order-coupled |
| Policy exploration (upper+lower) | Global `torch` RNG | `torch.manual_seed` | Order-coupled |
| Minibatch sampling | `torch.randperm` (same stream) | — | Order-coupled |
| Unit tests | Own seeded RNG | isolated | TEST_RNG_ISOLATED = YES |

### NEW-03 — Static Import-Time RNG Coupling (NEW_FINDING, P1) — Re-confirmed

Four config arrays are sampled at module import under seed 42 and are never re-evaluated when `set_global_seed(seed)` runs later:

| Variable | File:Line | Impact |
| :--- | :--- | :--- |
| `UAV_STORAGE_CAPACITY` | `config.py:99` | UAV storage heterogeneity identical across all training seeds |
| `UAV_COMPUTING_CAPACITY` | `config.py:101` | UAV compute heterogeneity identical across all seeds |
| `CPU_CYCLES_PER_BYTE` | `config.py:140` | Per-service compute density identical across all seeds |
| `FILE_SIZES` | `config.py:142` | Content file sizes identical across all seeds |

The 10 workload seeds therefore sample only environment dynamics (UE placement, requests, mobility, hotspots), NOT UAV heterogeneity or task-compute density — under-sampling parameter uncertainty in the SSOT §11.1 protocol.

- **Fix phase**: BEFORE_CANONICAL_TRAINING (Phase 4 RNG infrastructure)

### NEW-04 — Training Seed Count Mismatch (NEW_FINDING, P1) — Re-confirmed

SSOT §11.1 freezes `N_TRAIN_SEEDS = 5` (42, 84, 126, 168, 210). `run_linear_v2_full.sh:42` still passes `--training_seeds 42 84 126` (3 seeds); workload seeds (10) match (`:43`).

- **Fix phase**: PHASE4 (canonical experiment pipeline)

### Additional Coupling

`UE.global_ranks`, `UE.global_probs`, `UE.hotspot_centers` are class-level mutable state shared across `Env` instances in one process (re-initialized per `reset()`); multiple concurrent environments in a process would share this state. Canonical RNG infrastructure remains pending.

---

## 13. Experiment Entrypoints

| Script | Purpose | Semantics | Result Path | Loads Legacy? | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `run_hierarchical_mappo_experiment.py` | Primary hierarchical Attention-MAPPO training | `clean_v1`; overrides config via `apply_lower_ablation` (`:28`) | `saved_models/`, `results/reports/` (`:418`), `train_logs/` (`:168,390`) | NO (fresh `get_model()`) | **CURRENT_CANONICAL_TARGET** |
| `run_joint_end_to_end_mappo_experiment.py` | Joint-MAPPO baseline training | `clean_v1` | `saved_models/`, `results/reports/` | NO | **CURRENT_CANONICAL_TARGET** |
| `run_baseline_comparison_experiment.py` | 6-method comparison runner | Legacy (force-admission era) | `results/baseline_comparison/` | Unverified | **LEGACY_ONLY** |
| `run_baseline_matrix_v2.py` | Seed/workload matrix sweep | Legacy (capacity 30) | `results/baseline_matrix_*` | Unverified | **LEGACY_ONLY** |
| `run_sensitivity_full.py` | Sensitivity sweep | Legacy | `results/sensitivity_*` | Unverified | **LEGACY_ONLY** |
| `run_linear_v2_full.sh` | 3-seed × 10-workload launcher | Legacy (3 seeds ≠ SSOT 5) | — | — | **LEGACY_ONLY** |
| `run_linear_v2_dsr_strong.sh` | DSR-priority config launcher | Legacy | — | — | **LEGACY_ONLY** |
| `scripts/evaluate_force_admission_trained_variants.py` | Force-admission variant eval | Legacy | — | YES (explicitly legacy) | **LEGACY_ONLY** |
| `scripts/audit_*.py`, `scripts/launch_*.sh`, `scripts/run_force_admission_replacement_job.sh` | Legacy audits/launchers | Legacy | — | — | **LEGACY_ONLY** |
| `scripts/plot_baseline_matrix_v2_figures.py`, `scripts/visualize_attention_weights.py` | Figure/attention tooling | Aux | `docs/figures/` | — | **AUXILIARY** |
| `analyze_experiment_statistics.py` | Paired t-test/Wilcoxon/CI | Reusable | — | — | **AUXILIARY** |
| `analyze_references*.py`, `extract_*.py`, `extract_convergence_data.py`, `analyze_baseline_matrix_v2.py` | Thesis reference mining / legacy extraction | Legacy; **hardcoded `/home/PengYanghan/` paths** | `extracted_references/`, `analysis_results/` | — | **LEGACY_ONLY** |
| `tests/run_all_tests.py`, `tests/deterministic_smoke.py` | Test runners | Current | — | — | **CURRENT** |

---

## 14. Results / Checkpoints / Logs

- `results/`: 522 files / 20.76 MB / 519 tracked. Six subdirectories, all HISTORICAL_ONLY (`baseline_matrix_audit_20260624_smoke`, `baseline_matrix_force_admission_20260624`, `…_smoke`, `…_resume`, `final_force_admission`, `sensitivity_force_admission`). **No canonical clean results directory exists yet.**
- `saved_models/`: 144 files / 287 MB, untracked. All checkpoints obs_dim=425; canonical 1405 → load-protected by dimension mismatch, but must never be treated as canonical.
- `train_logs/`: 301 files / 37 MB; `logs/`: 19 files / 92 KB — HISTORICAL_ONLY.
- **Provenance gaps** in existing manifests: have `git_commit_hash`, `git_branch`, seeds, `config_snapshot`; **missing** `environment_semantics_version`, `scenario_hash`, `checkpoint_path` (mandatory in Phase 4).

---

## 15. Tests & Coverage Gaps

### Test Inventory (15 files, 26 test methods — counts re-verified)

| Test File | Methods | Target | Status | SSOT Contract |
| :--- | :---: | :--- | :--- | :--- |
| `test_env_temporal_alignment.py` | 3 | ENV-R6 | GREEN | §2.2-2.3 |
| `test_service_snapshot.py` | 3 | ENV-R1 | GREEN | §2.1, §5 |
| `test_service_admission.py` | 3 | ENV-R2 | GREEN | §3 |
| `test_batch_processor_sharing.py` | 2 | ENV-R3 | GREEN | §4.1-4.4 |
| `test_order_permutation_invariance.py` | 1 | ENV-R4 | GREEN | §4.4 |
| `test_cache_lifecycle.py` | 3 | ENV-R5 | GREEN | §5 |
| `test_collision_joint_feasibility.py` | 2 | ENV-R7/R8 | GREEN | §4.6 |
| `test_bandwidth_invariant_to_offload.py` | 1 | ENV-R1 | GREEN | §4.5 |
| `test_metric_semantics.py` | 3 | ENV-R9 | GREEN | §7.1 |
| `test_lower_logprob_contract.py` | 3 | Lower logprob | 2 GREEN / 1 expected RED | §6.1 (LOWER-01) |
| `test_lower_reward_causality.py` | 1 | Lower reward | expected RED | §7.2 (LOWER-02) |
| `test_no_silent_truncation.py` | 1 | Lower capacity | expected RED | §8.1 (LOWER-04) |
| `run_all_tests.py` / `deterministic_smoke.py` / `__init__.py` | — | runners | CURRENT | — |

(Statuses per Phase 2B report; local machine cannot execute torch tests — dual-machine protocol.)

### Coverage Gaps

| SSOT Contract / Area | Existing Test? | Gap |
| :--- | :--- | :--- |
| Lower reward admitted-domain scope (latency/DSR/MBS/constraint) | NO | Only flight-energy invariance is tested |
| Joint-MAPPO offload logprob contract (SUM vs MEAN) | NO | Joint baseline untested against §6.1 |
| Joint-MAPPO trajectory clamp consistency | NO | No old/new logprob same-action test |
| 100-request capacity | YES (expected RED) | `test_no_silent_truncation.py` covers 50 requests |
| Mask separation (hard vs quality) | NO | No test that disabling quality keeps hard feasibility |
| Fallback accounting | NO | No test that heuristic decisions are counted/reported |
| Constraint-mechanism scope/terminology | NO | ρ naming + admitted-domain violation signals untested |
| RNG determinism across seeds | NO | Config array re-seeding untested |
| Provenance manifest completeness | NO | `environment_semantics_version`/`scenario_hash` untested |

---

## 16. Documentation Consistency

### Stale Statements (Re-confirmed)

| Document | Location | Stale Statement | SSOT Canonical | Classification |
| :--- | :--- | :--- | :--- | :--- |
| `CLAUDE.md` | `:122` | "至多 30 个请求做离散卸载决策" | capacity = 100 | TERMINOLOGY_ONLY |
| `CLAUDE.md` | `:123` | "Lagrange 约束机制" | Violation-Driven Adaptive Constraint Penalty | TERMINOLOGY_ONLY |
| `CLAUDE.md` | `:153` | "可学习的 lambda_dsr 和 lambda_mbs 拉格朗日乘子" | ρ_dsr, ρ_mbs | TERMINOLOGY_ONLY |
| `AGENTS.md` | `:137-138, :168` | Same three statements (mirrors CLAUDE.md) | — | TERMINOLOGY_ONLY |
| `README.md` | `:58-73` | Full numerical baseline table (Energy=45.44M, EE=134.63…) | HISTORICAL_ONLY | DOCUMENTATION_ONLY |
| `latex/docs/abstract.tex` | `:14` | "拉格朗日约束机制" | Violation-Driven | LATEX |
| `latex/docs/chap03.tex` | `:348-389` region | `λ_dsr`/`λ_mbs` symbols, "Lagrange 乘子" | ρ_dsr, ρ_mbs | LATEX |
| `latex/docs/chap03.tex` | `:139,150,188,193,205,608,776` | "离线率" (offline rate) | Critical-UE Ratio | LATEX |
| `latex/docs/chap06.tex` | `:35` | "Lagrange 乘子自适应更新策略" | Violation-Driven | LATEX |

### Numerical Claims in LaTeX — All HISTORICAL_ONLY

`latex/docs/chap03.tex:776-778` (re-read and confirmed): fairness 0.9452; improvements 4.70/4.37/19.86/6.13/9.86 %; offline rate 2.25 %; DSR 0.3762 vs IPPO 0.2780 / Vanilla 0.2878 / Joint 0.4163 / Uniform 0.4529; UAV energy 49.98M with reductions 8.73/9.92/48.04/33.37/29.96 %; plus sensitivity results (`:818,832`) and DSR-priority comparison (`:864`). **Formal canonical training has not started — none of these may be cited as final thesis results.**

---

## 17. Thesis-Code Alignment

The thesis uses a **compressed chapter structure** — `latex/docs/` contains `abstract`, `chap01`, `chap02`, `chap03`, `chap06` (+ backmatter: notation, achievements, acknowledgements, complementary). **No `chap04.tex` or `chap05.tex` exists.** Intended Chapter 3 (trajectory) and Chapter 5 (hierarchical) content are merged into `chap03.tex` (~63KB).

| Intended Chapter | Implemented in Code? | Present in LaTeX? | Aligned? |
| :--- | :--- | :--- | :--- |
| Ch3: Trajectory optimization (Attention-MAPPO) | YES (`marl_models/attention_mappo/`) | YES (within chap03) | ALIGNED |
| Ch4: Fixed-snapshot offloading (MC-ITR / Exact-Small) | **NOT_IMPLEMENTED** | **NOT_PRESENT** | N/A |
| Ch5: Hierarchical dual-layer Attention-MAPPO | YES (`marl_models/offload_mappo/`, hierarchical runner) | YES (within chap03) | ALIGNED |

**MC-ITR / Exact-Small are explicitly NOT implemented** (confirmed: no such module/entrypoint exists in code; absent from LaTeX). They are referenced only as future work in `docs/reports/REPO_STATE_SNAPSHOT.md`. The thesis does not claim they exist.

---

## 18. Git / Storage / Portability

### Git History Size Diagnosis (Re-verified)

`.git/` = 2.87 GiB packed (20,053 objects, 3 packs). Largest blobs via `git verify-pack -v`: **51.77 MB, 48.58 MB, 34.16 MB, 31.77 MB** — TIFF thesis figures. TIFFs are tracked in HEAD under BOTH `docs/figures/` and `latex/docs/figures/` (tree-level duplication), and multiple historical copies exist from directory moves. **No destructive git operations were run or authorized** (`filter-repo`/BFG/aggressive gc forbidden).

### `.tmp/nature-skills-inspect` — Nested Git Repo (Re-verified)

`git ls-files -s .tmp` → `160000 c9b874a6…` — tracked as a **gitlink (submodule-style entry) without `.gitmodules`**; the nested repo's working tree shows as modified (` m` in status). Accidentally tracked transient artifact.

### Portability

| Aspect | Status | Evidence |
| :--- | :--- | :--- |
| `paths.py` | PORTABLE | `REPO_ROOT = Path(__file__).resolve().parent` |
| Current runners | PORTABLE from repo root | Relative `Path("saved_models")`, `Path("results")`, `train_logs` — CWD-fragile but functional |
| Legacy reference scripts | **MACHINE_SPECIFIC** | `analyze_all_references.py:9-10`, `analyze_all_references_v2.py:9-10`, `extract_all_pdfs.py:10-11`, `analyze_references.py:10` hardcode `/home/PengYanghan/...` |
| LaTeX build | MACHINE_SPECIFIC (intentional) | `latex/latexmk-wrapper.sh:3-5` exports remote TeX Live `/home/PengYanghan/texlive/2022/...` for server-side compilation |
| `.venv/` | MACHINE_SPECIFIC | Linux x86_64 — never execute on macOS arm64 |
| sys.path bootstrap | PORTABLE | `tests/run_all_tests.py`, `tests/deterministic_smoke.py`, `scripts/visualize_attention_weights.py`, `docs/figures/scripts/*` use `sys.path.insert(REPO_ROOT)` |

> **Correction vs 2026-09-18 audit**: that version stated "No hardcoded `/Users/` or `/home/` paths — grep found 0 occurrences in production runners". That is true **for production runners only**; repository-wide, hardcoded `/home/PengYanghan/` paths exist in 4 legacy reference scripts and the intentional remote TeX wrapper. Recorded as NEW-15R (P3).

### Security Hygiene

Pattern scan (`API_KEY|SECRET|PASSWORD|PRIVATE_KEY|Bearer|Authorization|TOKEN`) across all `.py/.sh/.json/.yaml/.yml` (excluding vendored/archive trees): **zero matches**.

```text
SECURITY_STATUS = CLEAN
POTENTIAL_SECRET_FOUND: NONE
```

---

## 19. Security Hygiene

No credentials, API keys, tokens, or secrets detected anywhere in the repository (see §18). No `.env` files, no auth tokens in configs.

```text
SECURITY_STATUS = CLEAN
```

---

## 20. P0/P1/P2/P3 Findings

### P0 Findings

```text
P0_NEW_FINDINGS = 0
```

No new P0 findings. All P0-level issues are the known Phase 3 defects (LOWER-01/02/04), already planned.

### P1 Findings (4)

| ID | Category | Finding | Confidence | Fix Phase |
| :--- | :--- | :--- | :--- | :--- |
| **NEW-01** | ALGORITHM | Joint-MAPPO trajectory clamp inconsistency: old logprob on unclipped sample vs new logprob on clipped stored action → biased PPO ratio (`joint_mappo.py:52-53,57,66,75,97`) | HIGH | PHASE3 |
| **NEW-02** | SCIENTIFIC_SEMANTICS | Lower reward uses system-level metrics (latency incl. unadmitted 20s, system DSR, system MBS ratio) instead of admitted-domain — violates SSOT §7.2.2/§3.2; **refined: constraint-violation signals (`run_hierarchical_mappo_experiment.py:59-60`) share the same system-scope defect** | HIGH | PHASE3 |
| **NEW-03** | RNG | Import-time RNG coupling freezes `UAV_STORAGE_CAPACITY`, `UAV_COMPUTING_CAPACITY`, `CPU_CYCLES_PER_BYTE`, `FILE_SIZES` at seed 42 across all training seeds (`config.py:48,99,101,140,142`) | HIGH | BEFORE_CANONICAL_TRAINING |
| **NEW-04** | EXPERIMENT_PIPELINE | `run_linear_v2_full.sh:42` uses 3 training seeds vs SSOT's frozen 5 (`run_linear_v2_full.sh:42-43`) | HIGH | PHASE4 |

### P2 Findings (8)

| ID | Category | Finding | Confidence | Fix Phase |
| :--- | :--- | :--- | :--- | :--- |
| **NEW-05** | DOCUMENTATION | LaTeX pervasive Lagrangian terminology (`λ_dsr`/`λ_mbs`, "拉格朗日约束机制/乘子") vs SSOT ρ formulation (`abstract.tex:14`, `chap03.tex:348-389`, `chap06.tex:35`) | HIGH | AFTER_CANONICAL_EXPERIMENTS |
| **NEW-06** | DOCUMENTATION | LaTeX uses "离线率" (offline rate) instead of Critical-UE Ratio / 低电量用户比例 (`chap03.tex:139,150,188,193,205,608,776`) | HIGH | AFTER_CANONICAL_EXPERIMENTS |
| **NEW-07** | PROVENANCE | README (`:58-73`) and chap03 (`:776-778,818,832,864`) carry extensive numerical results that are HISTORICAL_ONLY | HIGH | AFTER_CANONICAL_EXPERIMENTS |
| **NEW-08** | DOCUMENTATION | CLAUDE.md/AGENTS.md still state "至多 30 个请求", "Lagrange 约束", "lambda_dsr/lambda_mbs 拉格朗日乘子" (`CLAUDE.md:122-123,153`; `AGENTS.md:137-138,168`) | HIGH | BEFORE_PHASE3 |
| **NEW-09** | ORGANIZATION | Current runners write `saved_models/`, `results/reports/`, `train_logs/` shared with legacy assets — no canonical/historical isolation (`run_hierarchical_mappo_experiment.py:168,385,390,418`) | HIGH | AFTER_PHASE3_BEFORE_PHASE4 |
| **NEW-10** | STORAGE | Git history 2.87 GiB bloat from duplicated 30-52MB TIFF figures across `docs/figures/`, `latex/docs/figures/`, and history | HIGH | OPTIONAL |
| **NEW-11** | GIT | `.tmp/nature-skills-inspect` tracked as gitlink mode 160000 without `.gitmodules`; nested repo shows local modifications | HIGH | BEFORE_PHASE3 |
| **NEW-12** | TEST_GAP | No tests for Joint-MAPPO logprob/clamp contract, lower reward admitted-domain scope, mask separation, fallback accounting, RNG re-seeding, manifest completeness | MEDIUM | PHASE3 |

### P3 Findings (4)

| ID | Finding | Fix Phase |
| :--- | :--- | :--- |
| **NEW-13** | 11 `.DS_Store` files tracked (staged for deletion in current worktree) | BEFORE_PHASE3 |
| **NEW-14** | `__pycache__/` holds mixed cpython-312 and cpython-39 bytecode | OPTIONAL |
| **NEW-15R** | Hardcoded `/home/PengYanghan/` paths in 4 legacy reference scripts (`analyze_all_references.py:9-10`, `analyze_all_references_v2.py:9-10`, `extract_all_pdfs.py:10-11`, `analyze_references.py:10`); `latex/latexmk-wrapper.sh:3-5` is machine-specific **by design** (remote TeX). *(Replaces 2026-09-18 NEW-15 "empty reports/ at root" — that directory no longer exists.)* | OPTIONAL |
| **NEW-16** | Redundant `Categorical` construction in `joint_mappo.py:53,57` | OPTIONAL |

---

## 21. Known vs New Findings

### Known Phase 3 Findings (Confirmed)

| ID | Description | Status | Evidence |
| :--- | :--- | :--- | :--- |
| **LOWER-01** | Lower joint logprob MEAN instead of SUM | KNOWN / ALREADY PLANNED | `offload_mappo.py:246,284,285`; `joint_mappo.py:62,100` |
| **LOWER-02** | Lower reward energy includes flight/hover/WPT propulsion | KNOWN / ALREADY PLANNED | `run_hierarchical_mappo_experiment.py:78` ← `env.py:569,653` |
| **LOWER-04** | Capacity 30, silent heuristic fallback for 31+ | KNOWN / ALREADY PLANNED | `config.py:124,186`; `env.py:290`; `uavs.py:423,434-442` |

```text
KNOWN_PHASE3_FINDINGS_CONFIRMED = 3
```

Other prompt-listed Phase 3 verification areas were checked: reward admitted-domain scope (→ NEW-02, confirmed real), Violation-Driven terminology (→ §10, terminology-only), hard-vs-quality mask separation (→ §11, PARTIALLY_READY), Joint-MAPPO offload probability contract (→ §9, MEAN + clamp defect).

### Newly Discovered Findings

16 findings: 4 P1 (NEW-01..04), 8 P2 (NEW-05..12), 4 P3 (NEW-13..16, incl. corrected NEW-15R). Most significant: Joint clamp inconsistency; lower reward scope mismatch (extended to constraint signals); import-time RNG coupling; seed-count mismatch.

---

## 22. Recommended Fix Timing

| Finding | Fix Timing |
| :--- | :--- |
| LOWER-01, LOWER-02, LOWER-04 | PHASE3 |
| NEW-01 (Joint clamp), NEW-02 (reward scope incl. constraint signals) | PHASE3 |
| NEW-12 (accompanying regression tests) | PHASE3 |
| NEW-08 (agent-docs stale terms), NEW-11 (nested gitlink), NEW-13 (.DS_Store) | BEFORE_PHASE3 |
| NEW-03 (RNG coupling) | BEFORE_CANONICAL_TRAINING |
| NEW-09 (results isolation) | AFTER_PHASE3_BEFORE_PHASE4 |
| NEW-04 (seed count) | PHASE4 |
| NEW-05, NEW-06, NEW-07 (LaTeX/README terminology & historical numbers) | AFTER_CANONICAL_EXPERIMENTS |
| NEW-10 (git bloat), NEW-14, NEW-15R, NEW-16 | OPTIONAL |

---

## 23. Critical Path

Derived from repository evidence:

```text
Repository Hygiene (finish Wave 1 staging commit; NEW-08/11/13 doc & gitlink cleanup)
    │
    ▼
Phase 3: Lower MAPPO Canonical Repair
  ├── LOWER-01: logprob MEAN → SUM (offload_mappo.py:246,284; joint_mappo.py:62,100)
  ├── LOWER-02: reward energy → assignment-causal triple (runner:78 ← env.py:660-662)
  ├── NEW-02:  reward scope → admitted domain, incl. constraint violation signals (runner:59-60,75-81)
  ├── LOWER-04: capacity 30 → 100 (config.py:124,186,193; env.py:290; uavs.py:423)
  ├── NEW-01:  Joint trajectory clamp consistency (joint_mappo.py:52-66,75,97)
  └── NEW-12:  regression tests for all of the above
    │
    ▼
Historical Asset Isolation (results/, saved_models/, train_logs/ → canonical subtrees; NEW-09)
    │
    ▼
Phase 4: Canonical RNG & Provenance Infrastructure
  ├── NEW-03: deterministic scenario manifests for config arrays
  ├── NEW-04: 5 train seeds × 10 workload seeds pipeline
  ├── Provenance manifests: semantics version, scenario hash, checkpoint path
  └── canonical results/checkpoint isolation
    │
    ▼
Formal Canonical Training & Evaluation (remote 100.69.44.85, conda `drone`)
    │
    ▼
Thesis Synchronization
  ├── NEW-05/06: ρ_dsr/ρ_mbs terminology + Critical-UE Ratio
  ├── NEW-07: replace HISTORICAL_ONLY numbers with canonical results
  └── Chapter 4 snapshot optimizers (MC-ITR / Exact-Small) remain NOT_IMPLEMENTED —
      decide implement-vs-rescope before final thesis structure is locked
```

---

## 24. Final Readiness Matrix

| Area | Status | Evidence | Blocker? | Fix Phase |
| :--- | :--- | :--- | :--- | :--- |
| **SSOT** | GREEN | Frozen; all §0 decision locks re-verified against prompt contract | NO | — |
| **Environment** | GREEN | 9/9 SSOT contracts MATCH statically; Phase 2B 21/21 GREEN | NO | — |
| **Upper MAPPO** | GREEN | Obs dims consistent; 30-cap is observation-only; actor/critic consistent | NO | — |
| **Lower Logprob** | NEEDS_REPAIR | `offload_mappo.py:246,284` MEAN (LOWER-01) | YES (Phase 3) | PHASE3 |
| **Lower Reward** | NEEDS_REPAIR | Flight energy (LOWER-02) + system-scope terms incl. constraint signals (NEW-02) | YES (Phase 3) | PHASE3 |
| **Lower Capacity** | NEEDS_REPAIR | Truncation at 30 + silent fallback (LOWER-04) | YES (Phase 3) | PHASE3 |
| **Joint MAPPO** | NEEDS_REPAIR | Offload MEAN (LOWER-01) + trajectory clamp inconsistency (NEW-01) | YES (Phase 3) | PHASE3 |
| **Masks** | PARTIALLY_READY | Ablation-safe today; hard/quality/padding not explicitly separated | NO | OPTIONAL |
| **Constraint Mechanism** | GREEN (semantics) / scope via NEW-02 | Update rule matches §7.3; terminology legacy-sanctioned; violation scope needs Phase 3 | NO | PHASE3 |
| **RNG** | NEEDS_REPAIR | Import-time coupling (NEW-03); seed-count mismatch (NEW-04) | NO (Phase 4) | PHASE4 |
| **Provenance** | PENDING | Manifests lack semantics version / scenario hash / checkpoint path | NO (Phase 4) | PHASE4 |
| **Results Isolation** | PENDING | New runs coexist with legacy in shared top-level dirs | NO (Wave 2) | AFTER_PHASE3 |
| **Chapter 4 Optimizer** | NOT_IMPLEMENTED | MC-ITR / Exact-Small absent from code and LaTeX | NO (scope decision pending) | DECISION REQUIRED |
| **Tests** | NEEDS_REPAIR | 23 GREEN / 3 expected RED; Joint & reward-scope coverage missing | NO | PHASE3 |
| **LaTeX Alignment** | NEEDS_REPAIR | Lagrangian terminology, 离线率 naming, HISTORICAL_ONLY numbers | NO | AFTER_CANONICAL |
| **Workspace Hygiene** | NEEDS_REPAIR | Nested gitlink, tracked .DS_Store (staged), stale agent docs, untracked `tools/` script | NO | BEFORE_PHASE3 |

---

## Final Exit Block

```text
PROJECT_CONTEXT_UNDERSTOOD = YES

SSOT_MODIFIED = NO
PRODUCTION_SOURCE_MODIFIED = NO
TESTS_MODIFIED = NO
LATEX_MODIFIED = NO
RESULTS_MODIFIED = NO
CHECKPOINTS_MODIFIED = NO

FORMAL_TRAINING_STARTED = NO

P0_NEW_FINDINGS = 0
P1_NEW_FINDINGS = 4
P2_NEW_FINDINGS = 8
P3_NEW_FINDINGS = 4

KNOWN_PHASE3_FINDINGS_CONFIRMED = 3

WORKSPACE_AUDIT_COMPLETE = YES

STATUS = READY_FOR_PHASE3
```

---

## Appendix A — Changes vs 2026-09-18 Audit Version

1. **Constraint-signal scope folded into NEW-02 (refinement, not regression)**: the 2026-09-18 version marked the constraint mechanism's system-scope violation inputs as "System (OK)". Re-reading SSOT §7.3 shows the penalty is defined over `DSR^lower`/`MBS^lower` (admitted domain, per §7.2.2 conventions), while `compute_constraint_diagnostics` (`run_hierarchical_mappo_experiment.py:59-60`) consumes system-level metrics. The mechanism's *update rule* remains GREEN; its *input scope* joins NEW-02's Phase 3 repair.
2. **Portability claim corrected**: "0 hardcoded `/home/` paths" held only for production runners; 4 legacy reference scripts hardcode `/home/PengYanghan/...` and `latex/latexmk-wrapper.sh` pins remote TeX Live (intentional). Added as NEW-15R (P3).
3. **Former NEW-15 removed**: the empty root `reports/` directory cited on 2026-09-18 no longer exists; the finding slot was reused for NEW-15R to keep IDs stable.

All other findings, evidence lines, counts, and the READY_FOR_PHASE3 verdict were re-verified unchanged.

---

*Audit executed in strict read-only mode. No files were modified, moved, deleted, or renamed other than this deliverable. No training was started. No Phase 3 repairs were applied.*
