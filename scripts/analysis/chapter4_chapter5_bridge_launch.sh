#!/usr/bin/env bash
# Run disjoint bridge jobs on remote Linux; retain completed serial jobs.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
cd "$PROJECT_ROOT/source"
export PYTHONPATH=.
mkdir -p "$PROJECT_ROOT/validation/bridge_20260928"
printf '%s\n' "$$" > "$PROJECT_ROOT/validation/bridge_20260928/coordinator.pid"
pids=()
for shard in {0..7}; do
  bash run_remote_python.sh "$RUN_DIR/shard.py" \
    --runner "$RUN_DIR/runner.py" \
    --suite-root "$PROJECT_ROOT/results/main" \
    --output-root "$PROJECT_ROOT/results/bridge_20260928/full" \
    --shards 8 --shard-index "$shard" \
    > "$PROJECT_ROOT/validation/bridge_20260928/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  printf 'START shard=%s pid=%s\n' "$shard" "$!"
done
failed=0
for shard in {0..7}; do
  if wait "${pids[$shard]}"; then
    printf 'PASS shard=%s\n' "$shard"
  else
    printf 'FAILED shard=%s\n' "$shard"
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi
bash run_remote_python.sh "$RUN_DIR/shard.py" \
  --runner "$RUN_DIR/runner.py" \
  --suite-root "$PROJECT_ROOT/results/main" \
  --output-root "$PROJECT_ROOT/results/bridge_20260928/full" \
  --aggregate-only
