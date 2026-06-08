#!/usr/bin/env bash
# Baseline Matrix v2 实验脚本
# 用法: bash run_baseline_matrix_v2.sh
# 后台运行: nohup bash run_baseline_matrix_v2.sh > baseline_matrix_v2.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

SMOKE_DIR="results/baseline_matrix_v2_smoke"
FULL_DIR="results/baseline_matrix_v2"

echo "=========================================="
echo "Baseline Matrix v2 Experiment"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

echo "=========================================="
echo "Phase 1: Smoke Test"
echo "=========================================="
python run_baseline_matrix_v2.py \
    --methods proposed random \
    --training_seeds 42 \
    --workload_seeds 42 \
    --train_episodes 2 \
    --eval_episodes 1 \
    --output_dir "$SMOKE_DIR" \
    --overwrite

python analyze_baseline_matrix_v2.py --input_dir "$SMOKE_DIR"

test -f "$SMOKE_DIR/statistics.json"
test -f "$SMOKE_DIR/summary_table.tsv"
test -f "$SMOKE_DIR/report.md"
echo "Smoke test passed."
echo ""

echo "=========================================="
echo "Phase 2: Full Experiment"
echo "=========================================="
python run_baseline_matrix_v2.py \
    --methods random uniform ippo vanilla_mappo joint_mappo proposed \
    --training_seeds 42 84 126 \
    --workload_seeds 42 84 126 168 210 252 294 336 378 420 \
    --train_episodes 200 \
    --eval_episodes 6 \
    --output_dir "$FULL_DIR" \
    --resume

echo "Full experiment completed."
echo ""

echo "=========================================="
echo "Phase 3: Statistical Analysis"
echo "=========================================="
python analyze_baseline_matrix_v2.py --input_dir "$FULL_DIR"

echo ""
echo "=========================================="
echo "All phases completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "Output files:"
echo "  - $FULL_DIR/manifest.json"
echo "  - $FULL_DIR/statistics.json"
echo "  - $FULL_DIR/summary_table.tsv"
echo "  - $FULL_DIR/report.md"
