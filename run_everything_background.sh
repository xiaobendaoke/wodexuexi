#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-paper_full_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"

HIER_EPISODES="${HIER_EPISODES:-200}"
RL_TRAIN_EPISODES="${RL_TRAIN_EPISODES:-100}"
RL_TEST_EPISODES="${RL_TEST_EPISODES:-20}"
OFFLOAD_EPOCHS="${OFFLOAD_EPOCHS:-25}"
VALIDITY_EPOCHS="${VALIDITY_EPOCHS:-20}"
PER_CLASS_TARGET="${PER_CLASS_TARGET:-2000}"
PROCEDURAL_TRAIN_PER_CLASS="${PROCEDURAL_TRAIN_PER_CLASS:-1800}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60000}"
RUNTIME_SEEDS="${RUNTIME_SEEDS:-42 84 126 168}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-4}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-100}"

SENSITIVITY_EPOCHS="${SENSITIVITY_EPOCHS:-30}"
SENSITIVITY_EPISODES_PER_SEED="${SENSITIVITY_EPISODES_PER_SEED:-4}"
SCALE_EPISODES_PER_SEED="${SCALE_EPISODES_PER_SEED:-3}"
SC_OGO_EPISODES_PER_SEED="${SC_OGO_EPISODES_PER_SEED:-4}"

RUN_ROOT="results/full_runs/${RUN_TAG}"
OFFLOAD_ROOT="results/full_offload_experiments/${RUN_TAG}_offload"
LOG_DIR="run_outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" results/reports

MASTER_LOG="${LOG_DIR}/master.log"
STATUS_FILE="${LOG_DIR}/status.tsv"
PID_FILE="${LOG_DIR}/pid"
echo "$$" > "$PID_FILE"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  local step_log="${LOG_DIR}/${name}.log"
  log "START ${name}"
  printf '%s\tSTART\t%s\n' "$(date '+%F %T')" "$name" >> "$STATUS_FILE"
  if "$@" > "$step_log" 2>&1; then
    log "DONE  ${name}"
    printf '%s\tDONE\t%s\n' "$(date '+%F %T')" "$name" >> "$STATUS_FILE"
  else
    local code=$?
    log "FAIL  ${name} exit=${code}; see ${step_log}"
    printf '%s\tFAIL\t%s\texit=%s\n' "$(date '+%F %T')" "$name" "$code" >> "$STATUS_FILE"
    exit "$code"
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "Python not found or not executable: $PYTHON_BIN"
  exit 1
fi

log "RUN_TAG=${RUN_TAG}"
log "ROOT_DIR=${ROOT_DIR}"
log "PYTHON_BIN=${PYTHON_BIN}"
log "Logs: ${LOG_DIR}"

run_step "00_python_env" "$PYTHON_BIN" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
except Exception as exc:
    print("torch_check_error", repr(exc))
PY

run_step "01_hierarchical_mappo" \
  "$PYTHON_BIN" run_hierarchical_mappo_experiment.py \
    --num_episodes "$HIER_EPISODES" \
    --timestamp "${RUN_TAG}_hierarchical_mappo_${HIER_EPISODES}ep"

HIER_SUMMARY="results/reports/hierarchical_mappo_summary_${RUN_TAG}_hierarchical_mappo_${HIER_EPISODES}ep.json"
HIER_LOG="train_logs/hierarchical_mappo/log_data_${RUN_TAG}_hierarchical_mappo_${HIER_EPISODES}ep.json"

run_step "02_main_full_suite" \
  "$PYTHON_BIN" run_all_experiments.py \
    --name "$RUN_TAG" \
    --train_episodes "$RL_TRAIN_EPISODES" \
    --test_episodes "$RL_TEST_EPISODES" \
    --per_class_target "$PER_CLASS_TARGET" \
    --procedural_train_per_class "$PROCEDURAL_TRAIN_PER_CLASS" \
    --max_attempts "$MAX_ATTEMPTS" \
    --runtime_seeds $RUNTIME_SEEDS \
    --offload_epochs "$OFFLOAD_EPOCHS" \
    --validity_epochs "$VALIDITY_EPOCHS" \
    --device "$DEVICE" \
    --episodes_per_seed "$EPISODES_PER_SEED" \
    --steps_per_episode "$STEPS_PER_EPISODE"

SURROGATE_CKPT="${OFFLOAD_ROOT}/checkpoints/offload_policy_surrogate_runtime.pt"
RICH_CKPT="${OFFLOAD_ROOT}/checkpoints/offload_policy_rich_runtime.pt"

if [[ ! -f "$SURROGATE_CKPT" || ! -f "$RICH_CKPT" ]]; then
  log "Expected offload checkpoints not found under ${OFFLOAD_ROOT}; falling back to saved_offload_policies."
  SURROGATE_CKPT="saved_offload_policies/offload_policy_surrogate_runtime.pt"
  RICH_CKPT="saved_offload_policies/offload_policy_rich_runtime.pt"
fi

run_step "03_cql_sensitivity" \
  "$PYTHON_BIN" run_cql_sensitivity.py \
    --output_root "results/full_offload_experiments/${RUN_TAG}_cql_sensitivity" \
    --surrogate_checkpoint "$SURROGATE_CKPT" \
    --rich_checkpoint "$RICH_CKPT" \
    --per_class_target "$PER_CLASS_TARGET" \
    --procedural_train_per_class "$PROCEDURAL_TRAIN_PER_CLASS" \
    --max_attempts "$MAX_ATTEMPTS" \
    --epochs "$SENSITIVITY_EPOCHS" \
    --device "$DEVICE" \
    --runtime_seeds $RUNTIME_SEEDS \
    --episodes_per_seed "$SENSITIVITY_EPISODES_PER_SEED" \
    --steps_per_episode "$STEPS_PER_EPISODE"

CQL_CKPT="results/full_offload_experiments/${RUN_TAG}_cql_sensitivity/dsr_guard_dw6_mbs008/checkpoints/offload_policy_cql.pt"
if [[ ! -f "$CQL_CKPT" ]]; then
  CQL_CKPT="saved_offload_policies/offload_policy_cql.pt"
fi

run_step "04_radcc_sensitivity" \
  "$PYTHON_BIN" run_radcc_sensitivity.py \
    --output_root "results/full_offload_experiments/${RUN_TAG}_radcc_sensitivity" \
    --surrogate_checkpoint "$SURROGATE_CKPT" \
    --rich_checkpoint "$RICH_CKPT" \
    --cql_checkpoint "$CQL_CKPT" \
    --per_class_target "$PER_CLASS_TARGET" \
    --procedural_train_per_class "$PROCEDURAL_TRAIN_PER_CLASS" \
    --max_attempts "$MAX_ATTEMPTS" \
    --epochs "$SENSITIVITY_EPOCHS" \
    --device "$DEVICE" \
    --runtime_seeds $RUNTIME_SEEDS \
    --episodes_per_seed "$SENSITIVITY_EPISODES_PER_SEED" \
    --steps_per_episode "$STEPS_PER_EPISODE"

run_step "05_sc_ogo_ablation" \
  "$PYTHON_BIN" run_sc_ogo_ablation.py \
    --surrogate_checkpoint "$SURROGATE_CKPT" \
    --seeds $RUNTIME_SEEDS \
    --episodes_per_seed "$SC_OGO_EPISODES_PER_SEED" \
    --steps_per_episode "$STEPS_PER_EPISODE" \
    --output "results/reports/${RUN_TAG}_sc_ogo_ablation.json"

run_step "06_scale_generalization" \
  "$PYTHON_BIN" run_scale_generalization.py \
    --surrogate_checkpoint "$SURROGATE_CKPT" \
    --seeds $RUNTIME_SEEDS \
    --episodes_per_seed "$SCALE_EPISODES_PER_SEED" \
    --steps_per_episode "$STEPS_PER_EPISODE" \
    --output "results/reports/${RUN_TAG}_scale_generalization.json"

if [[ -f "$HIER_LOG" ]]; then
  run_step "07_hierarchical_figures" \
    "$PYTHON_BIN" generate_hierarchical_mappo_figures.py \
      --train_log "$HIER_LOG"
else
  log "SKIP hierarchical figures; missing ${HIER_LOG}"
fi

log "ALL DONE"
log "Manifest: ${RUN_ROOT}/experiment_manifest.json"
log "Hierarchical summary: ${HIER_SUMMARY}"
log "Status: ${STATUS_FILE}"
