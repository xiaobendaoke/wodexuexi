"""
Train RADCC-Offload: a risk-aware distributional cost critic for request-level offloading.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from marl_models.offload_policy import (
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    OFFLOAD_NUM_CLASSES,
    OFFLOAD_TARGET_NAMES,
    DistributionalCostNetwork,
    get_offload_feature_dim,
    get_offload_feature_specs,
    save_radcc_offload_policy_checkpoint,
)
from train_offload_policy import parse_hidden_dims, select_device, standardize_feature_splits


def load_radcc_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"RADCC dataset not found: {dataset_file}")
    dataset = np.load(dataset_file, allow_pickle=False)
    required = ("states", "action_costs", "quantile_targets", "action_masks")
    missing = [field for field in required if field not in dataset]
    if missing:
        raise KeyError(f"RADCC dataset is missing required fields: {missing}")
    states = np.asarray(dataset["states"], dtype=np.float32)
    action_costs = np.asarray(dataset["action_costs"], dtype=np.float32)
    quantile_targets = np.asarray(dataset["quantile_targets"], dtype=np.float32)
    action_masks = np.asarray(dataset["action_masks"], dtype=np.float32)
    expected_dim = get_offload_feature_dim(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)
    if states.ndim != 2 or states.shape[1] != expected_dim:
        raise ValueError(f"Unexpected states shape {states.shape}; expected (*, {expected_dim}).")
    if action_costs.shape != (states.shape[0], OFFLOAD_NUM_CLASSES):
        raise ValueError(f"Unexpected action_costs shape {action_costs.shape}.")
    if quantile_targets.ndim != 3 or quantile_targets.shape[:2] != (states.shape[0], OFFLOAD_NUM_CLASSES):
        raise ValueError(f"Unexpected quantile_targets shape {quantile_targets.shape}.")
    if action_masks.shape != (states.shape[0], OFFLOAD_NUM_CLASSES):
        raise ValueError(f"Unexpected action_masks shape {action_masks.shape}.")
    return {
        "states": states,
        "action_costs": action_costs,
        "quantile_targets": quantile_targets,
        "action_masks": action_masks,
    }


def split_indices(sample_count: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    val_count = max(1, min(int(round(sample_count * val_ratio)), sample_count - 1))
    return indices[val_count:].astype(np.int64), indices[:val_count].astype(np.int64)


def quantile_huber_loss(predictions: torch.Tensor, targets: torch.Tensor, quantile_levels: torch.Tensor) -> torch.Tensor:
    errors = targets - predictions
    abs_errors = torch.abs(errors)
    huber = torch.where(abs_errors <= 1.0, 0.5 * errors.pow(2), abs_errors - 0.5)
    weights = torch.abs(quantile_levels.view(1, 1, -1) - (errors.detach() < 0.0).float())
    return torch.mean(weights * huber)


def evaluate_radcc(
    model: DistributionalCostNetwork,
    states: np.ndarray,
    action_costs: np.ndarray,
    action_masks: np.ndarray,
    device: torch.device,
    risk_beta: float,
    cvar_alpha: float,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        state_tensor = torch.from_numpy(states).to(device)
        mask_tensor = torch.from_numpy(action_masks).to(device)
        quantiles = model(state_tensor)
        mean_cost = torch.mean(quantiles, dim=-1)
        cutoff = max(1, int(np.ceil(float(cvar_alpha) * quantiles.shape[-1])))
        tail_cost = torch.mean(quantiles[:, :, cutoff - 1 :], dim=-1)
        scores = mean_cost + float(risk_beta) * tail_cost
        scores = scores.masked_fill(mask_tensor <= 0.0, 1.0e9)
        actions = torch.argmin(scores, dim=1).cpu().numpy()
    oracle_actions = np.argmin(np.where(action_masks > 0.0, action_costs, 1.0e9), axis=1)
    action_counts = np.bincount(actions, minlength=OFFLOAD_NUM_CLASSES)
    oracle_counts = np.bincount(oracle_actions, minlength=OFFLOAD_NUM_CLASSES)
    chosen_costs = action_costs[np.arange(action_costs.shape[0]), actions]
    oracle_costs = action_costs[np.arange(action_costs.shape[0]), oracle_actions]
    return {
        "oracle_match_rate": float(np.mean(actions == oracle_actions)) if actions.size else 0.0,
        "mean_chosen_cost": float(np.mean(chosen_costs)) if chosen_costs.size else 0.0,
        "mean_oracle_cost": float(np.mean(oracle_costs)) if oracle_costs.size else 0.0,
        "greedy_action_counts": {OFFLOAD_TARGET_NAMES[idx]: int(action_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "oracle_action_counts": {OFFLOAD_TARGET_NAMES[idx]: int(oracle_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
    }


def train_radcc_offload_policy(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    hidden_dims: tuple[int, ...] = (128, 128),
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    val_ratio: float = 0.2,
    seed: int = config.SEED,
    device: str = "auto",
    risk_beta: float = config.RADCC_OFFLOAD_RISK_BETA,
    cvar_alpha: float = config.RADCC_OFFLOAD_CVAR_ALPHA,
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    arrays = load_radcc_dataset(dataset_path)
    train_idx, val_idx = split_indices(arrays["states"].shape[0], val_ratio=val_ratio, seed=seed)
    x_train, x_val, scaler_mean, scaler_std = standardize_feature_splits(arrays["states"][train_idx], arrays["states"][val_idx])
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(arrays["quantile_targets"][train_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device_obj = select_device(device)
    input_dim = get_offload_feature_dim(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)
    num_quantiles = int(arrays["quantile_targets"].shape[2])
    model = DistributionalCostNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_actions=OFFLOAD_NUM_CLASSES,
        num_quantiles=num_quantiles,
    ).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    quantile_levels = torch.linspace(0.05, 0.95, num_quantiles, device=device_obj)
    best_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, object] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_states, batch_targets in train_loader:
            batch_states = batch_states.to(device_obj)
            batch_targets = batch_targets.to(device_obj)
            predictions = model(batch_states)
            loss = quantile_huber_loss(predictions, batch_targets, quantile_levels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        eval_summary = evaluate_radcc(
            model,
            x_val,
            arrays["action_costs"][val_idx],
            arrays["action_masks"][val_idx],
            device_obj,
            risk_beta=risk_beta,
            cvar_alpha=cvar_alpha,
        )
        history.append({"epoch": float(epoch), "loss": float(mean_loss)})
        if best_eval is None or mean_loss < float(best_eval["loss"]):
            best_state = copy.deepcopy(model.state_dict())
            best_eval = {**eval_summary, "loss": float(mean_loss)}
            best_epoch = epoch
        print(
            f"epoch={epoch:03d} loss={mean_loss:.6f} "
            f"val_match={float(eval_summary['oracle_match_rate']):.4f} "
            f"val_cost={float(eval_summary['mean_chosen_cost']):.4f}"
        )

    if best_state is None or best_eval is None:
        raise RuntimeError("RADCC training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    summary: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(output_path),
        "device": str(device_obj),
        "feature_family": OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
        "feature_dim": int(input_dim),
        "feature_names": [spec["name"] for spec in get_offload_feature_specs(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)],
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "hidden_dims": list(hidden_dims),
        "num_quantiles": int(num_quantiles),
        "risk_beta": float(risk_beta),
        "cvar_alpha": float(cvar_alpha),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "best_epoch": int(best_epoch),
        "best_eval": best_eval,
        "epoch_history": history,
    }
    saved_path = save_radcc_offload_policy_checkpoint(
        model,
        output_path,
        feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
        hidden_dims=hidden_dims,
        num_quantiles=num_quantiles,
        risk_beta=risk_beta,
        cvar_alpha=cvar_alpha,
        scaler_mean=scaler_mean.tolist(),
        scaler_std=scaler_std.tolist(),
        metrics=summary,
        extra_metadata={"dataset_seed": seed, "dataset_rows": int(arrays["states"].shape[0])},
    )
    summary["checkpoint_path"] = saved_path
    Path(saved_path).with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RADCC-Offload distributional cost critic.")
    parser.add_argument("--dataset", type=str, default="offload_datasets/offload_dataset_radcc_cost.npz")
    parser.add_argument("--output", type=str, default="saved_offload_policies/offload_policy_radcc.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--hidden_dims", type=str, default="128,128")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--risk_beta", type=float, default=config.RADCC_OFFLOAD_RISK_BETA)
    parser.add_argument("--cvar_alpha", type=float, default=config.RADCC_OFFLOAD_CVAR_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_radcc_offload_policy(
        dataset_path=args.dataset,
        output_path=args.output,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        risk_beta=args.risk_beta,
        cvar_alpha=args.cvar_alpha,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
