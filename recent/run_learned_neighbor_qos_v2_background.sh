#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${1:-learned_neighbor_qos_tune_v2_20260515}"
LOG="${RUN_NAME}.nohup.log"
PID_FILE="${RUN_NAME}.pid"

nohup ./.venv/bin/python run_learned_neighbor_tuning_experiments.py \
  --run_name "$RUN_NAME" \
  --profiles current local_mbs_light_v2 local_mbs_light_v3 local_mbs_rescue qos_local_balanced \
  --train_episodes 100 \
  --all_workloads \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started $RUN_NAME"
echo "PID: $PID"
echo "Log: $LOG"
echo "PID file: $PID_FILE"
echo "Watch: tail -f $LOG"
echo "Summary after finish: results/learned_neighbor_tuning/$RUN_NAME/merged/statistics.md"
