#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ============================================================
# 论文修正版全量实验脚本
# 使用修改后的 config.py 参数：
#   - UE电池 500J, 阈值 50J, 待机 0.01W
#   - 飞行能耗 POWER_MOVE=300W, POWER_HOVER=150W
#   - DSR目标 0.18, MBS上限 0.03
#   - 线性归一化奖励函数
# ============================================================

RUN_NAME="${1:-paper_revised_full_$(date +%Y%m%d)}"
CPU_CORES="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 256)"
LOG="${RUN_NAME}.nohup.log"
PID_FILE="${RUN_NAME}.pid"

echo "============================================"
echo "论文修正版全量实验"
echo "============================================"
echo "Run name:    $RUN_NAME"
echo "CPU cores:   $CPU_CORES"
echo "Log:         $LOG"
echo ""

nohup ./.venv/bin/python run_full_learned_neighbor_cpu_experiments.py \
  --run_name "$RUN_NAME" \
  --suite all \
  --training_seeds 42 84 126 \
  --workload_seeds 42 84 126 168 210 252 294 336 378 420 \
  --train_episodes 200 \
  --episodes_per_seed 6 \
  --steps_per_episode 1000 \
  --lower_ablations full no_mask no_lagrange no_attention \
  --max_workers "$CPU_CORES" \
  --retries 2 \
  --skip_smoke \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started (PID: $PID)"
echo ""
echo "监控命令:"
echo "  ssh 100.69.44.85 tail -f /Users/wangpengfei/Lunwen/wodexuexi/"
echo ""
echo "完成后查看结果:"
echo "  results/joint_experiments/$RUN_NAME/merged/statistics.md"
echo ""
echo "预计时间: 取决于CPU核心数，256核约需 3-6 小时"
