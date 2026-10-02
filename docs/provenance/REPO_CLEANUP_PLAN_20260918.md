# Repository Hygiene & Canonical Layout Plan (Phase 2C-A)

- **Document Version**: `1.0.0`
- **Date**: 2026-09-18
- **Milestone**: Phase 2C-A — Repository Hygiene & Canonical Layout Planning
- **Operational Mode**: **STRICT READ-ONLY PLANNING** (Zero file moves, zero deletions, zero modifications to existing code)
- **Environment Semantics**: `clean_v1` (Phase 2B GREEN Verified)
- **SSOT Status**: FROZEN (`CANONICAL_SEMANTICS_SPEC.md`)
- **Formal Canonical Training Started**: **NO**

---

## 1. Current Mess Diagnosis

### 1.1 Context & Evolution
The codebase has evolved through three distinct evolutionary phases:
1. **Exploratory Sweeps (May – June 2026)**: Rapid parameter searches, heuristic ablations, and ad-hoc runner scripts (e.g., `run_linear_v2_full.sh`, `run_sensitivity_full.py`, `run_baseline_matrix_v2.py`).
2. **Defense Preparation & Patching (June – July 2026)**: Pre-defense sprint introducing "force admission" patches, resulting in massive run logs, legacy evaluation scripts (`scripts/evaluate_force_admission_trained_variants.py`), and loose checkpoint folders.
3. **Clean-Slate Standardization (September 2026)**: Freezing `CANONICAL_SEMANTICS_SPEC.md`, executing Phase 2A RED baseline testing, and Phase 2B GREEN environment repair (`clean_v1`).

### 1.2 Core Organization Symptoms
Despite achieving semantic correctness in `environment/` (`clean_v1`), the physical repository structure presents severe engineering hygiene issues:
- **Root Directory Overcrowding**: The root directory contains 30 files (27 non-dot, 3 dot) and 26 subdirectories. 14 Python scripts and 2 shell scripts sit loosely in the root.
- **Entanglement of Production vs Auxiliary Scripts**: Canonical production runners (`run_hierarchical_mappo_experiment.py`) sit directly beside one-off thesis reference parsers (`analyze_references.py`, `extract_all_pdfs.py`), deprecated matrix sweepers (`run_baseline_matrix_v2.py`), and log extractors (`extract_convergence_data.py`).
- **Absence of Canonical vs Historical Data Boundaries**: `results/` holds 6 directories containing 519 tracked JSON files (20.8 MB), all generated under legacy, buggy semantics (`force_admission`, `MAX_OFFLOAD_REQUESTS_PER_UAV=30`, energy accounting flaws). Without explicit structural segregation, subsequent phases risk inadvertently comparing canonical runs against invalid historical runs.
- **Checkpoint Incompatibility & Toxic Assets**: `saved_models/` contains 143 model checkpoint directories (287 MB), all trained with input dimension 425 ($5 + 30 \times 14$). In canonical clean semantics ($M=100$, input dim 1405 = $5 + 100 \times 14$), these checkpoints are completely incompatible and must never be loaded.
- **Massive Untracked Archival Bloat**: The `recent/` directory consumes 3.73 GB across 7,805 files (untracked nohup logs, superseded algorithm implementations like `attention_maddpg`, `matd3`, and old paper artifacts).
- **Git Tracking Inconsistencies**: 11 `.DS_Store` files and large (>5 MB) raw result JSONs are tracked in Git, while active helper scripts like `tools/export_prism_source_tables.py` remain untracked.

---

## 2. Root File Audit & Classification

The root directory contains 27 non-dot files and 3 dot files. Below is the complete empirical audit verified via repository grep and AST analysis:

| File | Actual Purpose | Referenced By | Status | Move Candidate | Risk Class |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `AGENTS.md` | AI Agent instructions & dual-machine protocol | Agent system, `CLAUDE.md`, `REPO_STATE_SNAPSHOT.md` | **CANONICAL** | **KEEP_IN_ROOT** | RISK-0 |
| `CLAUDE.md` | Claude Code rules & execution protocol | Agent system, `AGENTS.md`, `REPO_STATE_SNAPSHOT.md` | **CANONICAL** | **KEEP_IN_ROOT** | RISK-0 |
| `CANONICAL_SEMANTICS_SPEC.md` | Frozen Single Source of Truth (SSOT) | All test suites, `config.py`, `CLAUDE.md`, reports | **CANONICAL (SSOT)** | **KEEP_IN_ROOT** | RISK-0 |
| `README.md` | Human-facing project entrypoint | Git platforms, humans | **CANONICAL** | **KEEP_IN_ROOT** | RISK-0 |
| `LICENSE` | MIT software license | Legal / open-source distribution | **CANONICAL** | **KEEP_IN_ROOT** | RISK-0 |
| `config.py` | Global hyperparameter configuration | 120+ internal files, all models, env, tests | **CANONICAL** | **KEEP_IN_ROOT** | RISK-3 / 4 |
| `paths.py` | Repository root & result path resolver | 58 internal files, env, models, tests | **CANONICAL** | **KEEP_IN_ROOT** | RISK-3 |
| `wodexuexi.code-workspace` | Multi-root VS Code workspace configuration | VS Code IDE | **AUXILIARY** | **KEEP_IN_ROOT** (or `.vscode/`) | RISK-0 |
| `.gitignore` | Git exclusion rules | Git VCS | **CANONICAL** | **KEEP_IN_ROOT** | RISK-0 |
| `.codex-vscode-write-test` | 0-byte transient IDE test file | None | **DEAD_ARTIFACT** | Delete in Wave 1 | RISK-0 |
| `.DS_Store` | macOS filesystem metadata | macOS Finder | **OS_ARTIFACT** | Untrack in Wave 1 | RISK-0 |
| `CLEAN_SLATE_AUDIT.md` | Phase 1 full codebase audit report | `CLAUDE.md`, `REPO_STATE_SNAPSHOT.md` | **PROVENANCE** | `docs/provenance/` | RISK-1 |
| `TDD_RED_BASELINE.md` | Phase 2A RED test suite baseline capture | `REPO_STATE_SNAPSHOT.md` | **PROVENANCE** | `docs/provenance/` | RISK-1 |
| `ENV_REPAIR_REPORT.md` | Phase 2B GREEN environment repair report | `REPO_STATE_SNAPSHOT.md`, `AGENTS.md` | **PROVENANCE** | `docs/reports/` | RISK-1 |
| `REPO_STATE_SNAPSHOT.md` | Phase 2B closeout architectural snapshot | External review | **PROVENANCE** | `docs/reports/` | RISK-1 |
| `run_hierarchical_mappo_experiment.py` | Primary training runner (Attention-MAPPO) | `AGENTS.md`, `CLAUDE.md`, shell scripts, docs | **CURRENT_CANONICAL** | **KEEP_IN_ROOT_FOR_NOW** | RISK-2 / 3 |
| `run_joint_end_to_end_mappo_experiment.py` | Baseline training runner (Joint-MAPPO) | `AGENTS.md`, `CLAUDE.md`, `REPO_STATE_SNAPSHOT.md` | **CURRENT_CANONICAL** | **KEEP_IN_ROOT_FOR_NOW** | RISK-2 / 3 |
| `run_baseline_comparison_experiment.py` | Multi-algorithm comparison runner | Historical plans, `REPO_STATE_SNAPSHOT.md` | **HISTORICAL_EXPERIMENT** | `scripts/legacy/` | RISK-2 |
| `run_baseline_matrix_v2.py` | Workload/seed matrix evaluation runner | Historical runbooks, `REPO_STATE_SNAPSHOT.md` | **HISTORICAL_EXPERIMENT** | `scripts/legacy/` | RISK-2 |
| `run_sensitivity_full.py` | Sensitivity sweep across capacities/UEs | Historical runbooks, `REPO_STATE_SNAPSHOT.md` | **HISTORICAL_EXPERIMENT** | `scripts/legacy/` | RISK-2 |
| `run_linear_v2_full.sh` | Shell launcher for 3 seeds × 10 workloads | `AGENTS.md`, `CLAUDE.md`, `REPO_STATE_SNAPSHOT.md` | **HISTORICAL_EXPERIMENT** | `scripts/legacy/` | RISK-2 |
| `run_linear_v2_dsr_strong.sh` | Shell launcher for DSR-strong run | `AGENTS.md`, `CLAUDE.md`, `REPO_STATE_SNAPSHOT.md` | **HISTORICAL_EXPERIMENT** | `scripts/legacy/` | RISK-2 |
| `analyze_experiment_statistics.py` | Hypothesis testing tool (t-test, Wilcoxon, CI) | `README.md`, `AGENTS.md`, `CLAUDE.md` | **ANALYSIS_TOOL** | `scripts/analysis/` | RISK-1 / 2 |
| `analyze_baseline_matrix_v2.py` | Table generator for legacy matrix v2 | `REPO_STATE_SNAPSHOT.md` | **HISTORICAL_TOOL** | `scripts/analysis/` | RISK-2 |
| `extract_convergence_data.py` | Training curve parser from log files | `REPO_STATE_SNAPSHOT.md` | **ANALYSIS_TOOL** | `scripts/analysis/` | RISK-2 |
| `analyze_references.py` | BibTeX / thesis reference analyzer | `REPO_STATE_SNAPSHOT.md` | **AUXILIARY_TOOL** | `scripts/utilities/` | RISK-1 |
| `analyze_all_references.py` | Reference matcher against thesis chapters | `REPO_STATE_SNAPSHOT.md` | **AUXILIARY_TOOL** | `scripts/utilities/` | RISK-1 |
| `analyze_all_references_v2.py` | Extended thesis reference checker | `REPO_STATE_SNAPSHOT.md` | **AUXILIARY_TOOL** | `scripts/utilities/` | RISK-1 |
| `extract_all_pdfs.py` | Standalone PDF text extractor utility | `REPO_STATE_SNAPSHOT.md` | **AUXILIARY_TOOL** | `scripts/utilities/` | RISK-1 |
| `reference_analysis.json` | Extracted references JSON database | `analyze_references.py`, `analyze_all_*.py` | **AUXILIARY_DATA** | `latex/data/` | RISK-1 |

---

## 3. Entrypoint Classification & Script Matrix

### 3.1 Classification Criteria
- **CURRENT_CANONICAL_TARGET**: Active entrypoint targeted for Phase 3 and Phase 4 execution. Directly implements frozen canonical semantics or active baseline comparison.
- **HISTORICAL_EXPERIMENT**: Runner or launcher tailored to pre-Phase-2B semantics (e.g. `force_admission` flag, capacity limit 30). Retained strictly for reproducibility provenance.
- **ANALYSIS_TOOL**: Algorithmic or statistical analysis tool operating on results or log files.
- **AUXILIARY_TOOL**: Non-RL helper utility (e.g. LaTeX reference inspection, PDF text extraction).
- **UNKNOWN**: Scripts whose lineage or caller cannot be verified via repository references.

### 3.2 Script Classification Matrix

| Script Name | Category | Execution Engine | Primary Callers | Relocation Plan |
| :--- | :--- | :--- | :--- | :--- |
| `run_hierarchical_mappo_experiment.py` | `CURRENT_CANONICAL_TARGET` | Python / PyTorch | User, Shell, `AGENTS.md` | **KEEP_IN_ROOT_FOR_NOW** |
| `run_joint_end_to_end_mappo_experiment.py` | `CURRENT_CANONICAL_TARGET` | Python / PyTorch | User, Shell, `AGENTS.md` | **KEEP_IN_ROOT_FOR_NOW** |
| `evaluate_joint_end_to_end_mappo.py` | `HISTORICAL_ARCHIVE` (Archived) | Python / PyTorch | `AGENTS.md`, `CLAUDE.md` | Archived in `recent/archive_20260607/`; do NOT restore legacy code. Phase 4 should reconstruct a canonical clean_v1 evaluator |
| `run_baseline_comparison_experiment.py` | `HISTORICAL_EXPERIMENT` | Python / PyTorch | Standalone CLI | Move to `scripts/legacy/` in Wave 3 |
| `run_baseline_matrix_v2.py` | `HISTORICAL_EXPERIMENT` | Python / PyTorch | Standalone CLI | Move to `scripts/legacy/` in Wave 3 |
| `run_sensitivity_full.py` | `HISTORICAL_EXPERIMENT` | Python / PyTorch | Standalone CLI | Move to `scripts/legacy/` in Wave 3 |
| `run_linear_v2_full.sh` | `HISTORICAL_EXPERIMENT` | Bash / Nohup | User shell | Move to `scripts/legacy/` in Wave 3 |
| `run_linear_v2_dsr_strong.sh` | `HISTORICAL_EXPERIMENT` | Bash / Nohup | User shell | Move to `scripts/legacy/` in Wave 3 |
| `scripts/run_force_admission_replacement_job.sh` | `HISTORICAL_EXPERIMENT` | Bash / Conda | Legacy runbook | Move to `scripts/legacy/` in Wave 3 |
| `scripts/launch_force_admission_replacement.sh` | `HISTORICAL_EXPERIMENT` | Bash / Nohup | Legacy runbook | Move to `scripts/legacy/` in Wave 3 |
| `scripts/evaluate_force_admission_trained_variants.py`| `HISTORICAL_EXPERIMENT` | Python / PyTorch | Legacy runbook | Move to `scripts/legacy/` in Wave 3 |
| `analyze_experiment_statistics.py` | `ANALYSIS_TOOL` | Python / Scipy | User, `README.md` | Move to `scripts/analysis/` in Wave 3 |
| `analyze_baseline_matrix_v2.py` | `ANALYSIS_TOOL` (Legacy) | Python / Stdlib | Legacy audit | Move to `scripts/analysis/` in Wave 3 |
| `extract_convergence_data.py` | `ANALYSIS_TOOL` | Python / Numpy | Plotting pipelines | Move to `scripts/analysis/` in Wave 3 |
| `scripts/plot_baseline_matrix_v2_figures.py` | `ANALYSIS_TOOL` (Plotting) | Python / Matplotlib | Legacy docs | Move to `scripts/plotting/` in Wave 3 |
| `scripts/visualize_attention_weights.py` | `ANALYSIS_TOOL` (Plotting) | Python / Matplotlib | Documentation | Move to `scripts/plotting/` in Wave 3 |
| `scripts/audit_baseline_matrix_results.py` | `ANALYSIS_TOOL` (Audit) | Python / Stdlib | Legacy audit | Move to `scripts/analysis/` in Wave 3 |
| `scripts/audit_force_admission_sensitivity.py` | `ANALYSIS_TOOL` (Audit) | Python / Stdlib | Legacy audit | Move to `scripts/analysis/` in Wave 3 |
| `tools/export_prism_source_tables.py` | `ANALYSIS_TOOL` (Prism Export)| Python / Stdlib | GraphPad Prism export | Move to `scripts/utilities/` in Wave 3 |
| `analyze_references.py` | `AUXILIARY_TOOL` | Python / Stdlib | Thesis writing | Move to `scripts/utilities/` in Wave 3 |
| `analyze_all_references.py` | `AUXILIARY_TOOL` | Python / Stdlib | Thesis writing | Move to `scripts/utilities/` in Wave 3 |
| `analyze_all_references_v2.py` | `AUXILIARY_TOOL` | Python / Stdlib | Thesis writing | Move to `scripts/utilities/` in Wave 3 |
| `extract_all_pdfs.py` | `AUXILIARY_TOOL` | Python / pdftotext | Thesis writing | Move to `scripts/utilities/` in Wave 3 |

---

## 4. Proposed Script & Runner Layout

### 4.1 Script Relocation Impact Analysis
Moving root scripts into `scripts/` introduces runtime dependencies:
1. **Python `sys.path` Resolution**:
   Root scripts execute `import config`, `from environment.env import Env`, and `import marl_models`. If moved into `scripts/analysis/`, Python adds `scripts/analysis` to `sys.path[0]`, causing `ModuleNotFoundError: No module named 'config'` unless guarded by:
   ```python
   import sys
   from pathlib import Path
   REPO_ROOT = Path(__file__).resolve().parents[2]
   if str(REPO_ROOT) not in sys.path:
       sys.path.insert(0, str(REPO_ROOT))
   ```
2. **Shell Invocations & Documentation**:
   `README.md`, `AGENTS.md`, and `CLAUDE.md` instruct users to run `python analyze_experiment_statistics.py`. Moving it requires updating all instruction files and providing root delegation wrappers or updated instructions.

### 4.2 Runner Placement: Root vs `experiments/`
**Recommendation: `KEEP_IN_ROOT_FOR_NOW`**
For the canonical runners (`run_hierarchical_mappo_experiment.py` and `run_joint_end_to_end_mappo_experiment.py`), maintaining them in the root directory is strongly recommended:
- **No import regressions**: Eliminates any possibility of path or module import breakage.
- **Zero workflow disruption**: User command habits on the remote server (`python run_hierarchical_mappo_experiment.py --seed 42`) remain completely intact.
- **Direct visibility**: First-time reviewers immediately identify the exact experiment entrypoint upon opening the repository.

---

## 5. Historical Data, Checkpoints, and Logs Inventory

### 5.1 Results Directory (`results/`)
- **Total Size**: 20.8 MB (522 files on disk, 519 tracked by Git).
- **Subdirectories**:
  1. `baseline_matrix_audit_20260624_smoke/` (Smoke test results)
  2. `baseline_matrix_force_admission_20260624/` (Full IPPO & Vanilla MAPPO runs)
  3. `baseline_matrix_force_admission_20260624_smoke/` (Smoke runs)
  4. `baseline_matrix_force_admission_20260626_resume/` (Resumed matrix runs across all algorithms)
  5. `final_force_admission/` (Ablation variants: no_mask, no_lagrange, no_attention, etc.)
  6. `sensitivity_force_admission/` (UE count, CPU scale, and convergence sweeps)
- **Status**: **HISTORICAL_ONLY**. All 6 subdirectories reflect pre-Phase-2B semantics (`force_admission`, request capacity 30, pre-repair energy bugs).
- **Proposed Boundary Structure**:
  ```text
  results/
  ├── historical_legacy/
  │   ├── baseline_matrix_audit_20260624_smoke/
  │   ├── baseline_matrix_force_admission_20260624/
  │   ├── baseline_matrix_force_admission_20260624_smoke/
  │   ├── baseline_matrix_force_admission_20260626_resume/
  │   ├── final_force_admission/
  │   └── sensitivity_force_admission/
  └── canonical_clean/
      ├── chapter4/
      └── chapter5/
  ```

### 5.2 Saved Models Directory (`saved_models/`)
- **Total Size**: 286.96 MB (144 files, 143 subdirectories, 0 tracked by Git).
- **Subdirectories**: Prefixes include `attention_mappo_*`, `ippo_baseline_*`, `joint_mappo_*`, `offload_mappo_*`, `vanilla_mappo_*`.
- **Checkpoint Compatibility**:
  - **Input Observation Dimension**: 425 ($5 + 30 \times 14$).
  - **Canonical Model Dimension**: 1405 ($5 + 100 \times 14$).
  - **Result**: Immediate PyTorch `RuntimeError: size mismatch` upon loading.
- **Status**: **TOXIC_HISTORICAL_ONLY**. Must never be reused or loaded.
- **Target Structure**:
  Retaining the directory name `saved_models/` (rather than renaming to `checkpoints/`) is strongly advised to prevent breaking `paths.py` and `config.py`.
  ```text
  saved_models/
  ├── historical_legacy/   # Contains all 143 legacy checkpoints
  └── canonical_clean/     # Target for Phase 3/4 checkpoints
  ```

### 5.3 Logs Directories (`train_logs/` & `logs/`)
- **`train_logs/`**: 35.98 MB, 301 files (untracked). Contains logs for `baseline_comparison/`, `hierarchical_mappo/`, `joint_end_to_end_mappo/`.
- **`logs/`**: 92 KB, 19 files (untracked). Contains `force_admission_replacement/` nohup stdout logs.
- **Target Structure**:
  Consolidate `logs/force_admission_replacement/` into `train_logs/historical_legacy/`, establishing:
  ```text
  train_logs/
  ├── historical_legacy/
  └── canonical_clean/
  ```

---

## 6. Audit of `recent/` Directory

- **Disk Space**: 3.73 GB (7,805 files).
- **Git Tracking**: **Untracked** (excluded via `.gitignore:32`).
- **Internal Breakdown**:
  - `archive_20260607/` (2.56 GB): Pre-defense experimental iterations, including obsolete checkpoints and legacy runners.
  - `archive_20260701_old_paper_artifacts/` (927.7 MB): Superseded paper figures, LaTeX drafts, and experiment outputs.
  - `archive_baseline_before_energy_efficiency_20260604/` (239.9 MB): Early baseline evaluations before energy efficiency formulation.
  - `archive_20260703_old_data_cleanup/` (5.9 MB): Cleaned-up scratch files.
  - Dead MARL algorithm implementations (total < 1 MB): `attention_maddpg/`, `attention_masac/`, `attention_matd3/`, `maddpg/`, `mappo/`, `masac/`, `matd3/`, `nearest_greedy_baseline/`, `random_baseline/`, `static_baseline/`.
  - 48 loose root files: `.bak` files, `.pid` files, `*.nohup.log` files, and outdated README variants (`README_NEW.md`, `README_OLD.md`).
- **Active Code References**: Zero. Grep confirmed no active code references `recent/` except historical notes in documentation and `.gitignore`.
- **Classification**: **HISTORICAL_ARCHIVE**.
- **Migration Plan**: In Wave 2, rename `recent/` to `archive/legacy_runs/` and update `.gitignore`.

---

## 7. Documentation Layout

### 7.1 Current Documentation State
Documentation is split across the root directory and `docs/`:
- **Root Directory**: Contains 8 markdown files, mixing active SSOT (`CANONICAL_SEMANTICS_SPEC.md`), agent guidelines (`AGENTS.md`, `CLAUDE.md`), user guide (`README.md`), and historical milestone audit reports (`CLEAN_SLATE_AUDIT.md`, `TDD_RED_BASELINE.md`, `ENV_REPAIR_REPORT.md`, `REPO_STATE_SNAPSHOT.md`).
- **`docs/` Directory**: Contains 17 markdown/tsv audit files from July 2026 defense preparation and `docs/figures/`.

### 7.2 Target Documentation Structure
```text
docs/
├── architecture/           # System design and MARL structural diagrams
├── audits/                 # External and milestone audits
│   └── defense_20260702/   # July 2026 defense preparation audits
├── provenance/             # Historical engineering provenance
│   ├── CLEAN_SLATE_AUDIT.md
│   └── TDD_RED_BASELINE.md
├── reports/                # Active milestone delivery reports
│   ├── ENV_REPAIR_REPORT.md
│   ├── REPO_STATE_SNAPSHOT.md
│   └── REPO_CLEANUP_PLAN.md
├── presentations/          # Slide decks
│   └── group_meeting_progress_cn.pptx
└── figures/                # Thesis figure assets, source data, scripts
    ├── data/
    ├── prism_source_data/
    └── scripts/
```

### 7.3 Preserved in Root
- `CANONICAL_SEMANTICS_SPEC.md` (SSOT)
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`

---

## 8. Test Suite Layout (`tests/`)

- **Current State**: 15 files in a single flat directory:
  - 1 package marker (`__init__.py`)
  - 2 test execution entrypoints (`run_all_tests.py`, `deterministic_smoke.py`)
  - 12 regression test cases (`test_*.py`)
- **Evaluation of Subdirectory Splitting**:
  - `run_all_tests.py` utilizes `unittest.TestLoader().discover(start_dir="tests", pattern="test_*.py")`.
  - Splitting into subdirectories (`tests/environment/`, `tests/lower/`, `tests/integration/`) adds `__init__.py` overhead, complicates discovery, and risks import path failures for shared fixtures.
  - 15 files is lightweight, cohesive, and easily navigated.
- **Recommendation: `KEEP_FLAT`**. Do not create subdirectories.

---

## 9. Core Python Packages Layout

### 9.1 MARL Models (`marl_models/`)
- **Current State**: 8 algorithm subdirectories (`attention_mappo/`, `vanilla_mappo/`, `joint_mappo/`, `offload_mappo/`, `ippo_baseline/`, `random_baseline/`, `uniform_baseline/`, `uncoordinated_greedy_baseline/`) and 5 core shared modules (`attention.py`, `base_model.py`, `buffer_and_helpers.py`, `offload_policy.py`, `utils.py`).
- **Evaluation**: The layout matches the thesis algorithm taxonomy exactly. No dead modules or naming conflicts exist.
- **Recommendation: `KEEP_CURRENT_STRUCTURE`**. No restructuring.

### 9.2 Simulation Environment (`environment/`)
- **Current State**: 5 modular files (`env.py`, `uavs.py`, `user_equipments.py`, `comm_model.py`, `request_types.py`).
- **Evaluation**: The physical semantics and state transitions (`clean_v1`) have been verified GREEN. Module responsibilities are cleanly decoupled.
- **Recommendation: `KEEP_CURRENT_STRUCTURE`**. Strictly forbid unnecessary fragmentation.

---

## 10. Git Tracking Hygiene & Exclusions

### 10.1 Tracked Artifact Anomalies
1. **`.DS_Store` Files**: 11 `.DS_Store` files are tracked in the Git index:
   - `.DS_Store`, `.claude/.DS_Store`, `.spec-workflow/.DS_Store`, `.tmp/.DS_Store`, `docs/.DS_Store`, `latex/.DS_Store`, `latex/docs/.DS_Store`, `latex/image/template/.DS_Store`, `marl_models/.DS_Store`, `results/.DS_Store`, `tasks/.DS_Store`.
2. **Large JSON Files in `results/`**:
   - `results/sensitivity_force_admission/ue_count/raw/ue_count_results.json` (5.37 MB)
   - `results/sensitivity_force_admission/uav_cpu_scale/raw/uav_cpu_scale_results.json` (5.36 MB)
   - 4 trajectory files (~0.67 MB each)
3. **Empty/Dead Directories**:
   - `reports/` (0 bytes, empty directory)
   - `.codex-vscode-write-test` (0 bytes)

### 10.2 Recommended `.gitignore` Updates
The current `.gitignore` must be enhanced to cover:
```gitignore
# macOS metadata
.DS_Store
._*

# Temporary files & IDE scratchpad
.codex-vscode-write-test
.tmp/
output/

# Heavy figure source tables & compressed builds
docs/figures/prism_source_data/
latex/main_compressed.pdf
*.tiff

# Archive directory
archive/
```

---

## 11. Storage Hygiene Breakdown

| Directory / Target | Disk Size | File Count | Git Tracked? | Primary Problem Nature | Recommended Action |
| :--- | ---: | ---: | :--- | :--- | :--- |
| `.git/` | 2.9 GB | 61 | Internal VCS | Git History Problem | Do NOT run destructive git-filter/repack now |
| `recent/` | 3.7 GB | 7,805 | NO | Storage Problem | Move to `archive/legacy_runs/` |
| `latex/` | 525 MB | 258 | YES (244 files) | Document Assets | Retain; clean build artifacts via `make clean` |
| `saved_models/` | 287 MB | 144 | NO | Checkpoint Hygiene | Segregate into `saved_models/historical_legacy/` |
| `.venv/` | 204 MB | 4,784 | NO | Local Environment | Retain for server/linux reference; keep ignored |
| `.plot-venv/` | 78 MB | 2,771 | NO | Local Environment | Retain for local plotting; keep ignored |
| `docs/` | 75 MB | 68 | YES (45 files) | Storage (TIFF figures) | Reorganize docs; keep figures intact |
| `.tmp/` | 41 MB | 475 | YES (2 files) | Transient Cache | Gitignore and untrack transient cache |
| `train_logs/` | 37 MB | 301 | NO | Log Hygiene | Segregate into `train_logs/historical_legacy/` |
| `results/` | 21 MB | 522 | YES (519 files) | Repository Org & Git | Segregate into `results/historical_legacy/` |
| `output/` | 13 MB | 12 | NO | Auxiliary Asset | Relocate presentation slides to `docs/` |
| `extracted_references/` | 1.4 MB | 20 | YES (20 files) | Auxiliary Text | Retain or relocate under `docs/references/` |
| `marl_models/` | 608 KB | 70 | YES (24 files) | Production Source | **KEEP_CURRENT_STRUCTURE** |
| `environment/` | 292 KB | 15 | YES (5 files) | Production Source | **KEEP_CURRENT_STRUCTURE** |
| `scripts/` | 164 KB | 12 | YES (7 files) | Auxiliary Code | Restructure into subcategories in Wave 3 |
| `utils/` | 160 KB | 9 | YES (5 files) | Production Source | **KEEP_CURRENT_STRUCTURE** |
| `tests/` | 92 KB | 15 | YES (15 files) | Production Test | **KEEP_FLAT** |
| `logs/` | 92 KB | 19 | NO | Log Hygiene | Merge into `train_logs/historical_legacy/` |
| `analysis_results/` | 80 KB | 3 | YES (3 files) | Auxiliary Data | Relocate under `docs/` or `archive/` |
| `.spec-workflow/` | 56 KB | 8 | YES (8 files) | Workflow Metadata | Retain |
| `tasks/` | 8.0 KB | 1 | YES (1 file) | Empty Workflow | Clean up empty folders |
| `tools/` | 8.0 KB | 1 | NO | Utility Code | Relocate `export_prism_source_tables.py` to `scripts/utilities/` |
| `reports/` | 0 B | 0 | NO | Empty Folder | Remove empty directory |

---

## 12. Canonical vs Historical Naming Standards

To eliminate ambiguity across code, documentation, and data files, the repository adopts standard status tags:

| Status Tag | Semantic Definition | Application Scope |
| :--- | :--- | :--- |
| `CANONICAL` | Formally frozen Single Source of Truth (`clean_v1`) | `CANONICAL_SEMANTICS_SPEC.md`, frozen contracts |
| `CURRENT` | Active production code or active test suites for Phase 3/4 | `environment/`, `marl_models/`, `tests/`, active runners |
| `HISTORICAL_ONLY`| Assets generated under pre-Phase-2B semantics; invalid for canonical benchmarks | `results/historical_legacy/`, `saved_models/historical_legacy/` |
| `LEGACY_CODE` | Code superseded by canonical architecture | `run_baseline_matrix_v2.py`, `run_sensitivity_full.py` |
| `PROVENANCE` | Audit reports recording historical milestones | `CLEAN_SLATE_AUDIT.md`, `TDD_RED_BASELINE.md` |
| `AUXILIARY` | Supporting tools (reference checkers, format converters) | `analyze_references.py`, `export_prism_source_tables.py` |
| `ARCHIVED` | Large historical run archives and dead algorithms | `archive/legacy_runs/` (formerly `recent/`) |

---

## 13. Proposed Target Repository Tree

```text
wodexuexi/
├── README.md                              # Top-level entrypoint & project navigation
├── AGENTS.md                              # AI agent rules & dual-machine protocol
├── CLAUDE.md                              # Claude Code instructions
├── CANONICAL_SEMANTICS_SPEC.md            # Frozen SSOT for physical simulation
├── config.py                              # Global configuration
├── paths.py                               # Global path resolution
├── LICENSE                                # MIT license
├── wodexuexi.code-workspace               # VS Code workspace configuration
│
├── run_hierarchical_mappo_experiment.py   # Primary canonical training runner (Attention-MAPPO)
├── run_joint_end_to_end_mappo_experiment.py# Canonical baseline training runner (Joint-MAPPO)
│
├── environment/                           # Simulation physics & environment (KEEP_CURRENT_STRUCTURE)
│   ├── comm_model.py
│   ├── env.py
│   ├── request_types.py
│   ├── uavs.py
│   └── user_equipments.py
│
├── marl_models/                           # RL algorithms & network architectures (KEEP_CURRENT_STRUCTURE)
│   ├── attention.py
│   ├── base_model.py
│   ├── buffer_and_helpers.py
│   ├── offload_policy.py
│   ├── utils.py
│   ├── attention_mappo/
│   ├── ippo_baseline/
│   ├── joint_mappo/
│   ├── offload_mappo/
│   ├── random_baseline/
│   ├── uncoordinated_greedy_baseline/
│   ├── uniform_baseline/
│   └── vanilla_mappo/
│
├── tests/                                 # Verification test suite (KEEP_FLAT)
│   ├── deterministic_smoke.py
│   ├── run_all_tests.py
│   └── test_*.py (12 test modules)
│
├── utils/                                 # Logging, metrics, and visualization utilities
│   ├── baseline_metrics.py
│   ├── comparative_plots.py
│   ├── logger.py
│   ├── plot_logs.py
│   └── plot_snapshots.py
│
├── scripts/                               # Structured auxiliary scripts
│   ├── analysis/                          # Metrics analysis & statistical testing
│   │   ├── analyze_experiment_statistics.py
│   │   ├── analyze_baseline_matrix_v2.py
│   │   ├── audit_baseline_matrix_results.py
│   │   ├── audit_force_admission_sensitivity.py
│   │   └── extract_convergence_data.py
│   ├── plotting/                          # Figure generation & trajectory plotting
│   │   ├── plot_baseline_matrix_v2_figures.py
│   │   └── visualize_attention_weights.py
│   ├── utilities/                         # Document & reference tools
│   │   ├── analyze_references.py
│   │   ├── analyze_all_references.py
│   │   ├── analyze_all_references_v2.py
│   │   ├── export_prism_source_tables.py
│   │   └── extract_all_pdfs.py
│   └── legacy/                            # Superseded launchers & sweep scripts
│       ├── evaluate_force_admission_trained_variants.py
│       ├── launch_force_admission_replacement.sh
│       ├── run_baseline_comparison_experiment.py
│       ├── run_baseline_matrix_v2.py
│       ├── run_force_admission_replacement_job.sh
│       ├── run_linear_v2_dsr_strong.sh
│       ├── run_linear_v2_full.sh
│       └── run_sensitivity_full.py
│
├── docs/                                  # Project documentation & reports
│   ├── audits/
│   │   └── defense_20260702/              # July 2026 defense preparation audits
│   ├── provenance/                        # Historical milestone reports
│   │   ├── CLEAN_SLATE_AUDIT.md
│   │   └── TDD_RED_BASELINE.md
│   ├── reports/                           # Canonical delivery reports
│   │   ├── ENV_REPAIR_REPORT.md
│   │   ├── REPO_STATE_SNAPSHOT.md
│   │   └── REPO_CLEANUP_PLAN.md
│   ├── presentations/                     # Slide decks
│   │   └── group_meeting_progress_cn.pptx
│   └── figures/                           # Thesis figures and plotting data
│       ├── data/
│       ├── prism_source_data/
│       └── scripts/
│
├── results/                               # Benchmark and training results
│   ├── historical_legacy/                 # All pre-Phase-2B experimental results
│   │   ├── baseline_matrix_audit_20260624_smoke/
│   │   ├── baseline_matrix_force_admission_20260624/
│   │   ├── baseline_matrix_force_admission_20260624_smoke/
│   │   ├── baseline_matrix_force_admission_20260626_resume/
│   │   ├── final_force_admission/
│   │   └── sensitivity_force_admission/
│   └── canonical_clean/                   # Formal canonical training results (Phase 4+)
│       ├── chapter4/
│       └── chapter5/
│
├── saved_models/                          # Checkpoints
│   ├── historical_legacy/                 # 143 incompatible legacy checkpoints
│   └── canonical_clean/                   # Phase 3 & Phase 4 canonical checkpoints
│
├── train_logs/                            # Training logs
│   ├── historical_legacy/                 # Historical training logs & nohup logs
│   └── canonical_clean/                   # Phase 3 & Phase 4 canonical logs
│
├── archive/                               # Large cold storage
│   └── legacy_runs/                       # Renamed from recent/ (3.73 GB)
│
└── latex/                                 # Thesis LaTeX project (KEEP_INTACT)
```

---

## 14. Old → Proposed Path Mapping Table

| Current Path | Proposed Path | Wave | Risk Class | Reference Impact Summary |
| :--- | :--- | :--- | :--- | :--- |
| `CLEAN_SLATE_AUDIT.md` | `docs/provenance/CLEAN_SLATE_AUDIT.md` | Wave 1 | RISK-1 | Referenced in `CLAUDE.md`, `REPO_STATE_SNAPSHOT.md` |
| `TDD_RED_BASELINE.md` | `docs/provenance/TDD_RED_BASELINE.md` | Wave 1 | RISK-1 | Referenced in `REPO_STATE_SNAPSHOT.md` |
| `ENV_REPAIR_REPORT.md` | `docs/reports/ENV_REPAIR_REPORT.md` | Wave 1 | RISK-1 | Referenced in `REPO_STATE_SNAPSHOT.md`, `AGENTS.md` |
| `REPO_STATE_SNAPSHOT.md` | `docs/reports/REPO_STATE_SNAPSHOT.md` | Wave 1 | RISK-1 | Referenced in external audit guidelines |
| `.codex-vscode-write-test` | *DELETED* | Wave 1 | RISK-0 | 0-byte scratch artifact; zero references |
| `reports/` (empty) | *DELETED* | Wave 1 | RISK-0 | Empty directory; zero references |
| `results/*` | `results/historical_legacy/*` | Wave 2 | RISK-2 | 519 tracked JSON files; references in legacy scripts/docs |
| `saved_models/*` | `saved_models/historical_legacy/*` | Wave 2 | RISK-0 | Untracked; prevents accidental checkpoint loading |
| `train_logs/*` | `train_logs/historical_legacy/*` | Wave 2 | RISK-0 | Untracked logs |
| `logs/force_admission_replacement` | `train_logs/historical_legacy/force_admission_replacement` | Wave 2 | RISK-0 | Untracked; eliminates redundant `logs/` directory |
| `recent/` | `archive/legacy_runs/` | Wave 2 | RISK-1 | Untracked (3.73 GB); referenced in `.gitignore` and `README.md` |
| `output/group_meeting_progress_cn.pptx` | `docs/presentations/group_meeting_progress_cn.pptx` | Wave 2 | RISK-0 | Untracked presentation asset |
| `output/assets/` | `docs/presentations/assets/` | Wave 2 | RISK-0 | Untracked presentation asset |
| `analyze_experiment_statistics.py` | `scripts/analysis/analyze_experiment_statistics.py` | Wave 3 | RISK-2 | Referenced in `README.md`, `AGENTS.md`, `CLAUDE.md` |
| `analyze_baseline_matrix_v2.py` | `scripts/analysis/analyze_baseline_matrix_v2.py` | Wave 3 | RISK-2 | Referenced in `REPO_STATE_SNAPSHOT.md` |
| `extract_convergence_data.py` | `scripts/analysis/extract_convergence_data.py` | Wave 3 | RISK-2 | Referenced in `REPO_STATE_SNAPSHOT.md` |
| `scripts/audit_baseline_matrix_results.py` | `scripts/analysis/audit_baseline_matrix_results.py` | Wave 3 | RISK-2 | Internal legacy script |
| `scripts/audit_force_admission_sensitivity.py` | `scripts/analysis/audit_force_admission_sensitivity.py` | Wave 3 | RISK-2 | Internal legacy script |
| `scripts/plot_baseline_matrix_v2_figures.py` | `scripts/plotting/plot_baseline_matrix_v2_figures.py` | Wave 3 | RISK-2 | Internal legacy script |
| `scripts/visualize_attention_weights.py` | `scripts/plotting/visualize_attention_weights.py` | Wave 3 | RISK-2 | Add `sys.path` guard for `environment` / `marl_models` |
| `analyze_references.py` | `scripts/utilities/analyze_references.py` | Wave 3 | RISK-1 | Relocate reference scraper |
| `analyze_all_references.py` | `scripts/utilities/analyze_all_references.py` | Wave 3 | RISK-1 | Relocate reference checker |
| `analyze_all_references_v2.py` | `scripts/utilities/analyze_all_references_v2.py` | Wave 3 | RISK-1 | Relocate reference checker |
| `extract_all_pdfs.py` | `scripts/utilities/extract_all_pdfs.py` | Wave 3 | RISK-1 | Relocate PDF extractor |
| `tools/export_prism_source_tables.py` | `scripts/utilities/export_prism_source_tables.py` | Wave 3 | RISK-2 | Update path references to results |
| `reference_analysis.json` | `latex/data/reference_analysis.json` | Wave 3 | RISK-1 | Update reference scripts to point to new path |
| `run_baseline_comparison_experiment.py` | `scripts/legacy/run_baseline_comparison_experiment.py` | Wave 3 | RISK-2 | Add `sys.path` guard; update docs |
| `run_baseline_matrix_v2.py` | `scripts/legacy/run_baseline_matrix_v2.py` | Wave 3 | RISK-2 | Add `sys.path` guard; update docs |
| `run_sensitivity_full.py` | `scripts/legacy/run_sensitivity_full.py` | Wave 3 | RISK-2 | Add `sys.path` guard; update docs |
| `run_linear_v2_full.sh` | `scripts/legacy/run_linear_v2_full.sh` | Wave 3 | RISK-2 | Update path invocations |
| `run_linear_v2_dsr_strong.sh` | `scripts/legacy/run_linear_v2_dsr_strong.sh` | Wave 3 | RISK-2 | Update path invocations |
| `scripts/run_force_admission_replacement_job.sh` | `scripts/legacy/run_force_admission_replacement_job.sh` | Wave 3 | RISK-2 | Relocate legacy job script |
| `scripts/launch_force_admission_replacement.sh` | `scripts/legacy/launch_force_admission_replacement.sh` | Wave 3 | RISK-2 | Relocate legacy launch script |
| `scripts/evaluate_force_admission_trained_variants.py` | `scripts/legacy/evaluate_force_admission_trained_variants.py` | Wave 3 | RISK-2 | Relocate legacy eval script |

---

## 15. Migration Risk Classification

Each proposed move is assigned a strict risk category:

- **RISK-0 (No code/reference impact)**:
  - Deleting 0-byte transient files (`.codex-vscode-write-test`).
  - Removing empty directories (`reports/`).
  - Segregating untracked checkpoints (`saved_models/` $\to$ `saved_models/historical_legacy/`).
  - Consolidating untracked logs (`train_logs/` $\to$ `train_logs/historical_legacy/`).
- **RISK-1 (Only documentation / Markdown references affected)**:
  - Relocating audit markdown reports (`CLEAN_SLATE_AUDIT.md`, `TDD_RED_BASELINE.md`, `ENV_REPAIR_REPORT.md`, `REPO_STATE_SNAPSHOT.md`) to `docs/provenance/` and `docs/reports/`.
  - Relocating reference parsing scripts and `reference_analysis.json`.
  - Renaming `recent/` $\to$ `archive/legacy_runs/`.
- **RISK-2 (Script invocation / relative path updates required)**:
  - Moving `results/*` $\to$ `results/historical_legacy/*` (requires updating path constants in legacy plotting scripts like `plot_baseline_matrix_v2_figures.py`).
  - Relocating analysis and legacy runners to `scripts/` (requires adding `sys.path.insert(0, str(REPO_ROOT))` in each moved Python script so root modules can still be imported).
  - Updating shell script execution paths.
- **RISK-3 (Python imports / runtime behavior affected)**:
  - Modifying `config.py` or `paths.py`.
  - Moving `environment/`, `marl_models/`, `utils/`, or `tests/`.
  - **Policy**: STRICTLY POSTPONED / FORBIDDEN.
- **RISK-4 (Scientific / experiment reproducibility risk)**:
  - Moving or renaming `run_hierarchical_mappo_experiment.py` or `run_joint_end_to_end_mappo_experiment.py`.
  - Overwriting or discarding historical results/checkpoints.
  - **Policy**: STRICTLY POSTPONED / FORBIDDEN.

---

## 16. Cleanup Waves (Phased Execution Plan)

### Wave 1 — Zero-Risk Hygiene (Recommended BEFORE Phase 3)
- **Scope**: Documentation organization, `.gitignore` updates, OS artifact exclusion, removing dead zero-byte files.
- **Risk Level**: RISK-0 and RISK-1 only.
- **Actions**:
  1. Create `docs/provenance/` and `docs/reports/`.
  2. Move `CLEAN_SLATE_AUDIT.md` and `TDD_RED_BASELINE.md` into `docs/provenance/`.
  3. Move `ENV_REPAIR_REPORT.md` and `REPO_STATE_SNAPSHOT.md` into `docs/reports/`.
  4. Update `.gitignore` to exclude `.DS_Store`, `.codex-vscode-write-test`, `.tmp/`, `output/`, `*.tiff`.
  5. Delete 0-byte file `.codex-vscode-write-test` and empty folder `reports/`.
  6. Execute `git rm --cached` on the 11 tracked `.DS_Store` files.
- **Commit Target**: Atomic commits 1 & 2.

### Wave 2 — Historical Asset Isolation (Recommended AFTER Phase 3, BEFORE Phase 4)
- **Scope**: Segregate old results, untracked checkpoints, and logs; archive `recent/`.
- **Risk Level**: RISK-0 and RISK-2.
- **Actions**:
  1. Create `results/historical_legacy/` and `results/canonical_clean/`.
  2. Move existing 6 subdirectories in `results/` into `results/historical_legacy/`.
  3. Create `saved_models/historical_legacy/` and `saved_models/canonical_clean/`.
  4. Move 143 checkpoint folders in `saved_models/` into `saved_models/historical_legacy/`.
  5. Consolidate `logs/` and `train_logs/` into `train_logs/historical_legacy/` and create `train_logs/canonical_clean/`.
  6. Rename `recent/` $\to$ `archive/legacy_runs/`.
- **Commit Target**: Atomic commits 3, 4, 5 & 6.

### Wave 3 — Script Organization (Recommended in Phase 4 Infrastructure Sprint)
- **Scope**: Relocate auxiliary analysis, plotting, and legacy sweep scripts into `scripts/`.
- **Risk Level**: RISK-2.
- **Actions**:
  1. Populate `scripts/analysis/`, `scripts/plotting/`, `scripts/utilities/`, `scripts/legacy/`.
  2. Add `sys.path.insert(0, str(REPO_ROOT))` boilerplate to all relocated scripts to guarantee imports function without `PYTHONPATH` manipulation.
  3. Relocate `tools/export_prism_source_tables.py` into `scripts/utilities/`.
  4. Relocate `reference_analysis.json` into `latex/data/`.
  5. Update execution paths in documentation (`README.md`, `AGENTS.md`, `CLAUDE.md`).
- **Commit Target**: Atomic commits 7 & 8.

### Wave 4 — Architecture Refactoring (POSTPONED INDEFINITELY)
- **Scope**: Reorganizing `marl_models/` or `environment/` into `src/wodexuexi/`.
- **Risk Level**: RISK-3 / RISK-4.
- **Recommendation**: **DO NOT EXECUTE**. Refactoring package hierarchies during active thesis development introduces severe import regression risks with zero scientific value.

---

## 17. Recommended Commit Sequence

All cleanup changes must be committed as distinct atomic units:

```bash
# Wave 1 Commits (Documentation & Git Hygiene)
git add docs/provenance/ docs/reports/ CLEAN_SLATE_AUDIT.md TDD_RED_BASELINE.md ENV_REPAIR_REPORT.md REPO_STATE_SNAPSHOT.md
git commit -m "docs: organize provenance and milestone reports under docs/"

git rm --cached .DS_Store .claude/.DS_Store .spec-workflow/.DS_Store .tmp/.DS_Store docs/.DS_Store latex/.DS_Store latex/docs/.DS_Store latex/image/template/.DS_Store marl_models/.DS_Store results/.DS_Store tasks/.DS_Store
git add .gitignore
git commit -m "chore(git): update .gitignore and untrack .DS_Store files"

# Wave 2 Commits (Historical Asset Boundary Isolation)
git mv results/* results/historical_legacy/ # (excluding canonical_clean)
git commit -m "chore(results): isolate historical experimental results into results/historical_legacy/"

# Untracked filesystem operations (saved_models, train_logs, recent)
# Note: Since saved_models, train_logs, and recent are untracked, these are filesystem moves.
git commit --allow-empty -m "chore(storage): isolate legacy checkpoints and logs into historical_legacy/ and archive recent/"

# Wave 3 Commits (Script Organization)
git add scripts/
git commit -m "chore(scripts): organize analysis, plotting, and utility scripts under scripts/"

git add scripts/legacy/ run_baseline_*.py run_sensitivity_*.py run_linear_*.sh
git commit -m "chore(scripts): relocate legacy runners and shell launchers to scripts/legacy/"

git add README.md AGENTS.md CLAUDE.md
git commit -m "docs: refresh repository navigation and entrypoint documentation"
```

---

## 18. Files Explicitly Recommended NOT to Move

The following files must **STAY IN PLACE** throughout all cleanup phases:
1. `config.py`: The single global configuration module. Moving it breaks over 120 absolute imports across production code, baseline algorithms, and unit tests.
2. `paths.py`: Resolves `REPO_ROOT` dynamically. Any path change breaks filesystem resolution across the entire stack.
3. `CANONICAL_SEMANTICS_SPEC.md`: The frozen SSOT for physical simulation semantics. Must remain prominently in the repository root.
4. `README.md`, `AGENTS.md`, `CLAUDE.md`: Universal human and AI entrypoints.
5. `run_hierarchical_mappo_experiment.py`: Primary production runner for the proposed algorithm. Moving it causes command failures for external researchers and remote server scripts.
6. `run_joint_end_to_end_mappo_experiment.py`: Primary baseline runner. Must remain alongside the hierarchical runner.
7. `environment/` (entire package): 100% GREEN verified physical simulation. Keep intact.
8. `marl_models/` (entire package): Directly corresponds to the thesis algorithmic framework. Keep intact.
9. `tests/` (entire package): 15 cohesive test files. Keep flat.
10. `latex/` (entire package): Complete thesis LaTeX source and build pipeline. Keep intact.

---

## 19. Items Requiring Human Review & Decision

Before executing Wave 1 and Wave 2, human decision is required on the following points:

1. **Tracked Historical Results in Git**:
   `results/` currently contains 519 tracked JSON files (20.8 MB). When moving these into `results/historical_legacy/`, they will remain tracked in Git history unless explicitly untracked (`git rm -r --cached results/`). Does the user prefer to keep historical results tracked in Git for provenance, or untrack them to lighten the working tree?
2. **Heavy TIFF Figure Assets in `docs/figures/`**:
   `docs/figures/` contains 3 `.tiff` files totaling 76.9 MB (`图3-3_主配置性能工作点.tiff`, `图5-16_地面节点数量敏感性三指标对比.tiff`, `图5-17_UAV算力敏感性三指标对比.tiff`). These are high-resolution publication assets. Should these be retained in the working tree, or compressed/archived?
3. **Archiving `recent/` (3.73 GB)**:
   Does the user approve renaming `recent/` to `archive/legacy_runs/` locally on disk, or should this directory be moved outside the repository root entirely?
4. **Status of `evaluate_joint_end_to_end_mappo.py`**:
   `AGENTS.md` and `CLAUDE.md` reference `python evaluate_joint_end_to_end_mappo.py`, but the script was archived to `recent/archive_20260607/evaluate_joint_end_to_end_mappo.py` in June 2026. **Decision**: `RESTORE_LEGACY_EVALUATOR = NO`. Phase 4 should create or reconstruct a canonical `clean_v1` Joint-MAPPO evaluator from current production semantics. Archived legacy evaluator may be consulted for interface reference only. It must not be restored as CURRENT code.

---

## 20. Timing & Phase Boundary Protocol

To protect the integrity of Phase 3 development, the cleanup must strictly adhere to the following timing schedule:

```text
┌─────────────────────────────────────────────────────────────┐
│ NOW: Phase 2C-A Plan Ready (Zero moves executed)            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ BEFORE Phase 3 (Wave 1 Only - Zero Risk):                   │
│ - Move provenance reports to docs/provenance/ and reports/  │
│ - Update .gitignore and untrack .DS_Store                   │
│ - Delete 0-byte scratch files                               │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3 Execution (Lower MAPPO Canonical Repair):           │
│ - Purely focused on Lower policy, logprob arithmetic,       │
│   reward causality, and Lagrange dual multipliers           │
│ - ZERO directory reorganization or file moving permitted    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ AFTER Phase 3, BEFORE Phase 4 (Wave 2):                     │
│ - Isolate results/historical_legacy/ vs canonical_clean/    │
│ - Isolate saved_models/historical_legacy/ vs canonical_clean│
│ - Isolate train_logs/historical_legacy/ vs canonical_clean  │
│ - Move recent/ to archive/legacy_runs/                      │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4 Execution (Canonical Experiments Infrastructure):   │
│ - Wave 3 script organization (scripts/analysis/, etc.)      │
│ - Multi-seed training pipeline execution                    │
└─────────────────────────────────────────────────────────────┘
```

**CRITICAL RULE**: Under no circumstances should repository cleanup operations be combined with Phase 3 algorithmic fixes in the same Git commit.

---

## 21. Draft README Navigation Section

Below is the proposed navigation block to be incorporated into `README.md` in Wave 3 (without modifying `README.md` at this stage):

```markdown
## Repository Navigation & Directory Map

| Path | Purpose | Status |
| :--- | :--- | :--- |
| [`CANONICAL_SEMANTICS_SPEC.md`](CANONICAL_SEMANTICS_SPEC.md) | System Physics & Simulation Single Source of Truth | **Frozen SSOT** |
| [`config.py`](config.py) | Global Hyperparameter & Network Configuration | **Canonical Production** |
| [`paths.py`](paths.py) | Dynamic Repository & Result Path Resolver | **Canonical Production** |
| [`run_hierarchical_mappo_experiment.py`](run_hierarchical_mappo_experiment.py) | Primary Attention-MAPPO Training Entrypoint | **Current Production** |
| [`run_joint_end_to_end_mappo_experiment.py`](run_joint_end_to_end_mappo_experiment.py) | End-to-End Joint MAPPO Baseline Entrypoint | **Current Baseline** |
| [`environment/`](environment/) | Simulation Environment & Physics Engine (`clean_v1`) | **Canonical Source** |
| [`marl_models/`](marl_models/) | MARL Algorithms, Policies, and Attention Networks | **Canonical Source** |
| [`tests/`](tests/) | Canonical TDD Unit & Regression Test Suite | **Canonical Test** |
| [`results/canonical_clean/`](results/canonical_clean/) | Formal Canonical Benchmark Experimental Results | **Canonical Data** |
| [`results/historical_legacy/`](results/historical_legacy/) | Pre-Phase-2B Experimental Results (Provenance Only) | **Historical Data** |
| [`saved_models/`](saved_models/) | Model Checkpoints (`canonical_clean/` & `historical_legacy/`) | **Checkpoints** |
| [`docs/reports/`](docs/reports/) | System Audit, Environment Repair & Architecture Reports | **Milestone Reports** |
| [`latex/`](latex/) | Master's Thesis LaTeX Source & Compilation Pipeline | **Academic Source** |
```

---

## 22. Architectural State Verification Block

```text
PRODUCTION_SOURCE_MODIFIED = NO
FILES_MOVED = 0
FILES_DELETED = 0
TESTS_MODIFIED = NO
SSOT_MODIFIED = NO
FORMAL_TRAINING_STARTED = NO

CLEANUP_PLAN_READY = YES

STATUS = READY_FOR_HUMAN_REVIEW
```
