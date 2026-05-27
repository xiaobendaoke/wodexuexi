from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

from environment.env import Env
from marl_models.utils import get_model


def _to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _save_heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    if matrix.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention")
    plt.title(title)
    plt.xlabel("target index")
    plt.ylabel("source/head index")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize upper/lower attention weights for trained Attention-MAPPO policies.")
    parser.add_argument("--trajectory_model_dir", type=str, required=True)
    parser.add_argument("--lower_model_dir", type=str, default=None)
    parser.add_argument("--trajectory_model", type=str, default="attention_mappo")
    parser.add_argument("--lower_model", type=str, default="constrained_attention_offload_mappo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample_interval", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/attention_visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Env()
    trajectory_model = get_model(args.trajectory_model)
    trajectory_model.load(args.trajectory_model_dir)
    lower_model = None
    if args.lower_model_dir is not None:
        lower_model = get_model(args.lower_model)
        lower_model.load(args.lower_model_dir)

    obs = env.reset()
    obs_arr = np.asarray(obs, dtype=np.float32)
    samples: list[dict[str, object]] = []
    neighbor_weight_stack: list[np.ndarray] = []
    ue_weight_stack: list[np.ndarray] = []
    request_weight_stack: list[np.ndarray] = []

    for step in range(int(args.steps)):
        offload_actions = None
        offload_obs = None
        offload_masks = None
        if lower_model is not None:
            offload_obs, offload_masks = env.get_offloading_obs_and_masks()
            offload_actions, _, _ = lower_model.get_action_and_value(offload_obs, masks=offload_masks, exploration=False)

        if step % int(args.sample_interval) == 0:
            sample: dict[str, object] = {"step": int(step)}
            if hasattr(trajectory_model, "extract_attention_weights"):
                upper_weights = trajectory_model.extract_attention_weights(obs_arr)
                neighbor_weight_stack.append(upper_weights["neighbor_weights"])
                ue_weight_stack.append(upper_weights["ue_weights"])
                sample["upper"] = {key: _to_list(value) for key, value in upper_weights.items()}
            if lower_model is not None and offload_obs is not None and hasattr(lower_model, "extract_request_attention_weights"):
                lower_weights = lower_model.extract_request_attention_weights(offload_obs)
                if lower_weights:
                    request_weight_stack.append(lower_weights["request_weights"])
                    sample["lower"] = {key: _to_list(value) for key, value in lower_weights.items()}
            samples.append(sample)

        traj_actions = trajectory_model.select_actions(obs_arr, exploration=False)
        next_obs, _, _ = env.step(traj_actions, offloading_actions=offload_actions)
        obs_arr = np.asarray(next_obs, dtype=np.float32)

    payload = {
        "seed": int(args.seed),
        "steps": int(args.steps),
        "sample_interval": int(args.sample_interval),
        "trajectory_model_dir": str(Path(args.trajectory_model_dir).resolve()),
        "lower_model_dir": str(Path(args.lower_model_dir).resolve()) if args.lower_model_dir is not None else None,
        "samples": samples,
    }
    (output_dir / "attention_samples.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if neighbor_weight_stack:
        neighbor = np.concatenate(neighbor_weight_stack, axis=0)
        neighbor_mean = neighbor.mean(axis=0)
        _save_heatmap(neighbor_mean, output_dir / "neighbor_attention_heatmap.png", "Upper neighbor attention")
        np.save(output_dir / "neighbor_attention_mean.npy", neighbor_mean)
    if ue_weight_stack:
        ue = np.concatenate(ue_weight_stack, axis=0)
        ue_mean = ue.mean(axis=0)
        _save_heatmap(ue_mean, output_dir / "ue_attention_heatmap.png", "Upper UE attention")
        np.save(output_dir / "ue_attention_mean.npy", ue_mean)
    if request_weight_stack:
        request = np.concatenate(request_weight_stack, axis=0)
        request_mean = request.mean(axis=(0, 1))
        _save_heatmap(request_mean, output_dir / "request_attention_heatmap.png", "Lower request attention")
        np.save(output_dir / "request_attention_mean.npy", request_mean)

    print(json.dumps({"output_dir": str(output_dir), "num_samples": len(samples)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
