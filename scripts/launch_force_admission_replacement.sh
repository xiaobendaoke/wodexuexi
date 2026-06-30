#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/PengYanghan/Lunwen/wodexuexi"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TARGET="${1:-all}"
LOG_ROOT="$ROOT/logs/force_admission_replacement/$RUN_ID"
mkdir -p "$LOG_ROOT"

cd "$ROOT"
nohup bash "$ROOT/scripts/run_force_admission_replacement_job.sh" "$TARGET" "$RUN_ID" \
  > "$LOG_ROOT/${TARGET}.out" 2> "$LOG_ROOT/${TARGET}.err" &
PID=$!
echo "$PID" > "$LOG_ROOT/${TARGET}.pid"

cat <<EOF
Started force-admission replacement job.
  target: $TARGET
  run_id: $RUN_ID
  pid: $PID
  stdout: $LOG_ROOT/${TARGET}.out
  stderr: $LOG_ROOT/${TARGET}.err

Check status:
  tail -f $LOG_ROOT/${TARGET}.out
  ps -p $PID -o pid,etime,cmd
EOF
