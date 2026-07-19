#!/usr/bin/env bash
set -euo pipefail
cd "."

# ============================================================
# 线性归一化版全量实验脚本
#
# 修改内容：
#   - run_hierarchical_mappo_experiment.py: log → linear 归一化
#   - config.py: OFFLOAD_REWARD_LATENCY_WEIGHT 0.35→4.0
#                OFFLOAD_REWARD_ENERGY_WEIGHT 0.10→1.2
#                OFFLOAD_ENERGY_NORM_REF = 25000 (新增)
#
# 实验规模：
#   - 5 组主实验 + 4 组消融 = 9 组策略
#   - 3 training seeds × 10 workload seeds × 6 eval episodes
#   - 200 train episodes per seed
# ============================================================

RUN_NAME="${1:-linear_v2_full_$(date +%Y%m%d)}"
CPU_CORES="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 256)"
LOG="${RUN_NAME}.nohup.log"
PID_FILE="${RUN_NAME}.pid"

echo "============================================"
echo "线性归一化版全量实验"
echo "============================================"
echo "Run name:      $RUN_NAME"
echo "CPU cores:     $CPU_CORES"
echo "Log:           $LOG"
echo ""
echo "修改确认:"
echo "  latency_term:  log → linear (÷ NUM_UES × T_penalty)"
echo "  energy_term:   log → linear (÷ NUM_UAVS × OFFLOAD_ENERGY_NORM_REF)"
echo "  latency weight: 0.35 → 4.0"
echo "  energy weight:  0.10 → 1.2"
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
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started (PID: $PID)"
echo ""
echo "监控命令:"
echo "  ssh 100.69.44.85 tail -f ~/Lunwen/wodexuexi/$LOG"
echo ""
echo "查看进程:"
echo "  ssh 100.69.44.85 ps aux | grep run_full_learned"
echo ""
echo "完成后结果目录:"
echo "  results/joint_experiments/$RUN_NAME/"
