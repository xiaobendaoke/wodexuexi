#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RUN_NAME="${1:-extra_baselines_$(date +%Y%m%d)}"
LOG="${RUN_NAME}.nohup.log"
PID_FILE="${RUN_NAME}.pid"

echo "Starting extra baselines: $RUN_NAME"
echo "Log: $LOG"

nohup ./.venv/bin/python eval_extra_baselines.py \
  --run_name "$RUN_NAME" \
  --training_seeds 42 84 126 \
  --workload_seeds 42 84 126 168 210 252 294 336 378 420 \
  --episodes_per_seed 6 \
  --steps_per_episode 1000 \
  --baselines all_local all_mbs random_offload fixed_position \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "Started (PID: $PID)"
echo "Watch: tail -f $LOG"
