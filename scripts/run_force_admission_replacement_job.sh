#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/home/PengYanghan/Lunwen/wodexuexi"
PY="/home/PengYanghan/miniconda3/envs/drone/bin/python"
SEEDS=(42 84 126)
WORKLOAD_SEEDS=(42 84 126 168 210 252 294 336 378 420)
LOG_ROOT="$ROOT/logs/force_admission_replacement/$RUN_ID"
mkdir -p "$LOG_ROOT"
cd "$ROOT"

run_cmd() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  echo "Command: $*" | tee "$LOG_ROOT/${name}.cmd"
  "$@" > "$LOG_ROOT/${name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $name"
}

run_main() {
  echo "=== Force-admission main/layer runs ==="
  for seed in "${SEEDS[@]}"; do
    run_cmd "main_upper_only_seed${seed}" \
      "$PY" run_hierarchical_mappo_experiment.py \
      --mode upper_only \
      --num_episodes 200 \
      --seed "$seed" \
      --timestamp "force_admission_main_upper_only_seed${seed}_${RUN_ID}" \
      --force_service_admission

    run_cmd "main_lower_only_fixed_upper_seed${seed}" \
      "$PY" run_hierarchical_mappo_experiment.py \
      --mode lower_only_fixed_upper \
      --num_episodes 200 \
      --seed "$seed" \
      --timestamp "force_admission_main_lower_only_seed${seed}_${RUN_ID}" \
      --force_service_admission

    run_cmd "main_full_hierarchical_seed${seed}" \
      "$PY" run_hierarchical_mappo_experiment.py \
      --mode full_hierarchical \
      --num_episodes 200 \
      --seed "$seed" \
      --timestamp "force_admission_main_full_seed${seed}_${RUN_ID}" \
      --force_service_admission
  done
}

run_ablation() {
  echo "=== Force-admission lower-layer ablations ==="
  for seed in "${SEEDS[@]}"; do
    for ablation in full no_mask no_lagrange no_attention; do
      run_cmd "ablation_${ablation}_seed${seed}" \
        "$PY" run_hierarchical_mappo_experiment.py \
        --mode lower_only_fixed_upper \
        --lower_ablation "$ablation" \
        --num_episodes 200 \
        --seed "$seed" \
        --timestamp "force_admission_ablation_${ablation}_seed${seed}_${RUN_ID}" \
        --force_service_admission
    done
  done
}

run_dsr_priority() {
  echo "=== Force-admission DSR-priority reruns ==="
  echo "Note: current runner exposes dsr_target and mbs_load_ceiling, but not all reward-weight CLI knobs from the old strong config."
  for seed in "${SEEDS[@]}"; do
    run_cmd "dsr_priority_full_seed${seed}" \
      "$PY" run_hierarchical_mappo_experiment.py \
      --mode full_hierarchical \
      --num_episodes 200 \
      --seed "$seed" \
      --timestamp "force_admission_dsr_priority_full_seed${seed}_${RUN_ID}" \
      --dsr_target 0.28 \
      --mbs_load_ceiling 0.08 \
      --force_service_admission
  done
}

run_sensitivity() {
  echo "=== Force-admission sensitivity runs ==="
  "$PY" - <<PYWRAP > "$LOG_ROOT/sensitivity_wrapper.log" 2>&1
from pathlib import Path
import config
from run_sensitivity_full import (
    run_convergence_experiment,
    run_ue_count_experiment,
    run_uav_cpu_scale_experiment,
)

config.FORCE_SERVICE_ADMISSION = True
seeds = [42, 84, 126]
workload_seeds = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]
root = Path("results/sensitivity_force_admission")
root.mkdir(parents=True, exist_ok=True)

run_convergence_experiment(
    training_seeds=seeds,
    workload_seed=workload_seeds[0],
    eval_episodes=6,
    output_dir=str(root / "effective_efficiency_convergence"),
)
run_ue_count_experiment(
    ue_counts=[60, 80, 100, 120, 140],
    training_seeds=seeds,
    workload_seeds=workload_seeds,
    eval_episodes=6,
    output_dir=str(root / "ue_count"),
)
run_uav_cpu_scale_experiment(
    scales=[0.6, 0.8, 1.0, 1.2, 1.4],
    training_seeds=seeds,
    workload_seeds=workload_seeds,
    eval_episodes=6,
    output_dir=str(root / "uav_cpu_scale"),
)
PYWRAP
}

run_figures_after_existing_results() {
  echo "=== Regenerate figures that already have force-admission data ==="
  run_cmd "fig_baseline_matrix_force_admission" \
    "$PY" scripts/plot_baseline_matrix_v2_figures.py --skip_sensitivity
  run_cmd "fig_trajectory_static_force_admission" \
    "$PY" docs/figures/scripts/generate_force_admission_trajectory_static.py
}

case "$TARGET" in
  main)
    run_main
    ;;
  ablation)
    run_ablation
    ;;
  dsr|dsr_priority)
    run_dsr_priority
    ;;
  sensitivity)
    run_sensitivity
    ;;
  figures)
    run_figures_after_existing_results
    ;;
  all)
    run_main
    run_ablation
    run_dsr_priority
    run_sensitivity
    run_figures_after_existing_results
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Use one of: main, ablation, dsr_priority, sensitivity, figures, all" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T')] ALL DONE target=$TARGET run_id=$RUN_ID"
