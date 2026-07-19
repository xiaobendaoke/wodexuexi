#!/usr/bin/env bash
set -euo pipefail
cd "."
# DSR-Aware - 强配置，目标 DSR >= 0.27

RUN_NAME="${1:-linear_v2_dsr_strong_$(date +%Y%m%d)}"
CPU_CORES="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 256)"
LOG="${RUN_NAME}.nohup.log"
PID_FILE="${RUN_NAME}.pid"

echo "===== DSR-Aware Strong ====="
echo "Run: $RUN_NAME | CPUs: $CPU_CORES"
echo ""

cp config.py config.py.dsr_backup
./.venv/bin/python _apply_dsr_config.py strong

nohup ./.venv/bin/python run_full_learned_neighbor_cpu_experiments.py \
  --run_name "$RUN_NAME" --suite all \
  --training_seeds 42 84 126 \
  --workload_seeds 42 84 126 168 210 252 294 336 378 420 \
  --train_episodes 200 --episodes_per_seed 6 --steps_per_episode 1000 \
  --lower_ablations full no_mask no_lagrange no_attention \
  --max_workers "$CPU_CORES" --retries 2 \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "Started PID=$PID. Auto-restore config.py on exit."

(
    while kill -0 $PID 2>/dev/null; do sleep 60; done
    cp config.py.dsr_backup config.py
    echo "$(date): config.py restored" >> "$LOG"
) &
