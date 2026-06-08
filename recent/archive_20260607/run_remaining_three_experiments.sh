#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/PengYanghan/Lunwen/wodexuexi"
PY="$ROOT/.venv/bin/python"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

cd "$ROOT"

echo "========== 0. Check environment =========="
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "PYTHONPATH=$PYTHONPATH"

"$PY" - <<'PYEOF'
import sys
import numpy
import torch
import matplotlib
import config
print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("matplotlib:", matplotlib.__version__)
print("config loaded:", config.__file__)
print("cuda available:", torch.cuda.is_available())
PYEOF

echo "========== 1. Check required trained upper models =========="
test -f "$ROOT/saved_models/vanilla_mappo_upper_vanilla_seed42_200ep/final/vanilla_mappo.pth"
test -f "$ROOT/saved_models/attention_mappo_upper_attention_seed42_200ep/final/attention_mappo.pth"

echo "Found:"
echo "  $ROOT/saved_models/vanilla_mappo_upper_vanilla_seed42_200ep/final"
echo "  $ROOT/saved_models/attention_mappo_upper_attention_seed42_200ep/final"

echo "========== 2. Evaluate vanilla vs attention upper policies with heuristic offloading =========="
"$PY" recent/run_joint_trajectory_offload_experiment.py \
  --name upper_vanilla_vs_attention_heuristic \
  --trajectory_run_root . \
  --combos vanilla_mappo__heuristic:vanilla_mappo:heuristic attention_mappo__heuristic:attention_mappo:heuristic \
  --trajectory_model_dirs vanilla_mappo=saved_models/vanilla_mappo_upper_vanilla_seed42_200ep/final attention_mappo=saved_models/attention_mappo_upper_attention_seed42_200ep/final \
  --seeds 42 84 126 168 \
  --episodes_per_seed 8

echo "========== 3. Visualize attention weights =========="
"$PY" scripts/visualize_attention_weights.py \
  --trajectory_model_dir saved_models/attention_mappo_upper_attention_seed42_200ep/final \
  --trajectory_model attention_mappo \
  --seed 42 \
  --steps 1000 \
  --sample_interval 20 \
  --output_dir results/attention_visualization_upper

echo "========== 4. Train true end-to-end joint MAPPO =========="
"$PY" run_joint_end_to_end_mappo_experiment.py \
  --num_episodes 200 \
  --seed 42 \
  --timestamp joint_e2e_seed42_200ep

echo "========== 5. Evaluate true end-to-end joint MAPPO =========="
"$PY" evaluate_joint_end_to_end_mappo.py \
  --model_dir saved_models/joint_mappo_joint_e2e_seed42_200ep/final \
  --name joint_e2e_eval_seed42 \
  --seeds 42 84 126 168 \
  --episodes_per_seed 8

echo "========== 6. Summarize key outputs =========="
"$PY" - <<'PYEOF'
import json
from pathlib import Path

paths = [
    Path("results/joint_experiments/upper_vanilla_vs_attention_heuristic/joint_experiment_summary.json"),
    Path("results/attention_visualization_upper/seed42/attention_samples.json"),
    Path("results/reports/joint_e2e_mappo_summary_joint_e2e_seed42_200ep.json"),
    Path("results/joint_e2e_evaluations/joint_e2e_eval_seed42/joint_e2e_evaluation_summary.json"),
]

for path in paths:
    print("\n==", path)
    print("exists:", path.exists())
    if not path.exists():
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("json read failed:", exc)
        continue
    if "aggregate" in data:
        print("aggregate:")
        for metric, value in data["aggregate"].items():
            if isinstance(value, dict):
                print(f"  {metric}: mean={value.get('mean')} std={value.get('std')}")
            else:
                print(f"  {metric}: {value}")
    if "policy_results" in data:
        print("policy_results:")
        for key, value in data["policy_results"].items():
            print(" ", key)
            if isinstance(value, dict):
                for m, mv in value.items():
                    if isinstance(mv, dict) and "mean" in mv:
                        print(f"    {m}: mean={mv.get('mean')} std={mv.get('std')}")
    if "model_dir" in data:
        print("model_dir:", data["model_dir"])
    if "mean_recent_reward" in data:
        print("mean_recent_reward:", data["mean_recent_reward"])
    if "samples" in data:
        print("num_attention_samples:", len(data["samples"]))
PYEOF

echo "========== ALL REMAINING EXPERIMENTS FINISHED =========="
echo "Upper comparison summary: results/joint_experiments/upper_vanilla_vs_attention_heuristic/joint_experiment_summary.json"
echo "Attention visualization: results/attention_visualization_upper/seed42"
echo "Joint E2E train summary: results/reports/joint_e2e_mappo_summary_joint_e2e_seed42_200ep.json"
echo "Joint E2E evaluation summary: results/joint_e2e_evaluations/joint_e2e_eval_seed42/joint_e2e_evaluation_summary.json"
