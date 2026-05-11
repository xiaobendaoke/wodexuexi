#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-paper_full_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
RUN_MODE="${RUN_MODE:-all}"  # all or remaining
CPU_THREADS="${CPU_THREADS:-$(nproc)}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_MAX_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export TORCH_NUM_THREADS="$CPU_THREADS"

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

RUN_ROOT="results/full_runs/${RUN_TAG}"
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
log "RUN_MODE=${RUN_MODE}"
log "ROOT_DIR=${ROOT_DIR}"
log "PYTHON_BIN=${PYTHON_BIN}"
log "CPU_THREADS=${CPU_THREADS}"
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

HIER_SUMMARY="results/reports/hierarchical_mappo_summary_${RUN_TAG}_hierarchical_mappo_${HIER_EPISODES}ep.json"
HIER_LOG="train_logs/hierarchical_mappo/log_data_${RUN_TAG}_hierarchical_mappo_${HIER_EPISODES}ep.json"

if [[ -f "$HIER_SUMMARY" && -f "$HIER_LOG" ]]; then
  log "SKIP  01_hierarchical_mappo; existing outputs found for ${RUN_TAG}"
else
  run_step "01_hierarchical_mappo" \
    "$PYTHON_BIN" run_hierarchical_mappo_experiment.py \
      --num_episodes "$HIER_EPISODES" \
      --timestamp "${RUN_TAG}_hierarchical_mappo_${HIER_EPISODES}ep"
fi

RL_DONE_MARKER="${RUN_ROOT}/comparisons/training/comparison_summary.png"

if [[ "$RUN_MODE" == "remaining" && -f "$RL_DONE_MARKER" ]]; then
  log "SKIP  02_main_full_suite RL; existing RL comparison found at ${RL_DONE_MARKER}"
  run_step "02_offload_suite_remaining" \
    "$PYTHON_BIN" -u run_all_experiments.py \
      --name "$RUN_TAG" \
      --skip_rl \
      --per_class_target "$PER_CLASS_TARGET" \
      --procedural_train_per_class "$PROCEDURAL_TRAIN_PER_CLASS" \
      --max_attempts "$MAX_ATTEMPTS" \
      --runtime_seeds $RUNTIME_SEEDS \
      --offload_epochs "$OFFLOAD_EPOCHS" \
      --validity_epochs "$VALIDITY_EPOCHS" \
      --device "$DEVICE" \
      --episodes_per_seed "$EPISODES_PER_SEED" \
      --steps_per_episode "$STEPS_PER_EPISODE"
else
  run_step "02_main_full_suite" \
    "$PYTHON_BIN" -u run_all_experiments.py \
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
fi

if [[ -f "$HIER_LOG" ]]; then
  run_step "03_hierarchical_figures" \
    "$PYTHON_BIN" -u generate_hierarchical_mappo_figures.py \
      --train_log "$HIER_LOG"
else
  log "SKIP hierarchical figures; missing ${HIER_LOG}"
fi

log "ALL DONE"
log "Manifest: ${RUN_ROOT}/experiment_manifest.json"
log "Hierarchical summary: ${HIER_SUMMARY}"
log "Status: ${STATUS_FILE}"
