#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${1:-learned_neighbor_paper_full_20260515}"
CPU_CORES="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 1)"
LOG="${RUN_NAME}.nohup.log"
PID_FILE="${RUN_NAME}.pid"

nohup ./.venv/bin/python run_learned_neighbor_tuning_experiments.py \
  --run_name "$RUN_NAME" \
  --profiles current local_mbs_rescue local_mbs_light_v3 \
  --source_run learned_neighbor_e2e_full_20260514 \
  --source_episodes 200 \
  --train_episodes 200 \
  --episodes_per_seed 6 \
  --steps_per_episode 1000 \
  --all_workloads \
  --max_workers "$CPU_CORES" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started $RUN_NAME"
echo "PID: $PID"
echo "CPU cores requested: $CPU_CORES"
echo "Log: $LOG"
echo "PID file: $PID_FILE"
echo "Watch: tail -f $LOG"
echo "Summary after finish: results/learned_neighbor_tuning/$RUN_NAME/merged/statistics.md"
