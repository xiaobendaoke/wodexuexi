#!/usr/bin/env bash
# 全量 baseline 对比实验脚本
# 用法: bash recent/run_baseline_comparison_full.sh
# 后台运行: nohup bash recent/run_baseline_comparison_full.sh > baseline_full.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
source .venv/bin/activate

# 限制线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# 实验参数
TRAIN_EPISODES=200
EVAL_EPISODES=10
TRAINING_SEEDS=(42 84 126)
WORKLOAD_SEEDS=(1001 1002 1003 1004 1005 1006 1007 1008 1009 1010)

# 非学习 baseline
NON_LEARNING_BASELINES=("random" "uniform")

# 学习 baseline
LEARNING_BASELINES=("ippo" "vanilla_mappo" "joint_mappo" "proposed")

echo "=========================================="
echo "Baseline Comparison Full Experiment"
echo "Train episodes: $TRAIN_EPISODES"
echo "Eval episodes: $EVAL_EPISODES"
echo "Training seeds: ${TRAINING_SEEDS[*]}"
echo "Workload seeds: ${WORKLOAD_SEEDS[*]}"
echo "=========================================="
echo ""

# 非学习 baseline：只用 workload seeds，seed 固定 42
for baseline in "${NON_LEARNING_BASELINES[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting non-learning baseline: $baseline"
    for wseed in "${WORKLOAD_SEEDS[@]}"; do
        echo "  [$baseline] workload_seed=$wseed"
        python run_baseline_comparison_experiment.py \
            --baseline "$baseline" \
            --eval_episodes "$EVAL_EPISODES" \
            --seed 42 \
            --workload_seed "$wseed" \
            --output_dir results/baseline_comparison_full
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $baseline"
    echo ""
done

# 学习 baseline：training seeds × workload seeds
for baseline in "${LEARNING_BASELINES[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting learning baseline: $baseline"
    for tseed in "${TRAINING_SEEDS[@]}"; do
        echo "  [$baseline] training_seed=$tseed"
        python run_baseline_comparison_experiment.py \
            --baseline "$baseline" \
            --train_episodes "$TRAIN_EPISODES" \
            --eval_episodes "$EVAL_EPISODES" \
            --seed "$tseed" \
            --workload_seed 1001 \
            --output_dir results/baseline_comparison_full
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $baseline"
    echo ""
done

echo "=========================================="
echo "All experiments completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
