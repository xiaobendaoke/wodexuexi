"""
Train a constrained CQL-DQN request-level offloading policy from offline transitions.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from marl_models.offload_policy import (
    OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
    OFFLOAD_NUM_CLASSES,
    OFFLOAD_TARGET_NAMES,
    OffloadQNetwork,
    get_offload_feature_dim,
    get_offload_feature_specs,
    save_cql_offload_policy_checkpoint,
)
from train_offload_policy import parse_hidden_dims, select_device, standardize_feature_splits


def load_cql_transition_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"CQL transition dataset not found: {dataset_file}")
    dataset = np.load(dataset_file, allow_pickle=False)
    required_fields = ("states", "actions", "rewards", "next_states", "dones", "action_masks")
    missing_fields = [field for field in required_fields if field not in dataset]
    if missing_fields:
        raise KeyError(f"CQL dataset is missing required fields: {missing_fields}")

    arrays = {field: np.asarray(dataset[field]) for field in required_fields}
    states = arrays["states"].astype(np.float32)
    next_states = arrays["next_states"].astype(np.float32)
    actions = arrays["actions"].astype(np.int64)
    rewards = arrays["rewards"].astype(np.float32)
    dones = arrays["dones"].astype(np.float32)
    action_masks = arrays["action_masks"].astype(np.float32)

    expected_dim = get_offload_feature_dim(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)
    if states.ndim != 2 or states.shape[1] != expected_dim:
        raise ValueError(f"Unexpected states shape {states.shape}; expected (*, {expected_dim}).")
    if next_states.shape != states.shape:
        raise ValueError(f"next_states shape {next_states.shape} must match states shape {states.shape}.")
    if actions.shape != rewards.shape or actions.shape != dones.shape:
        raise ValueError("actions, rewards, and dones must be aligned 1D arrays.")
    if actions.shape[0] != states.shape[0]:
        raise ValueError("Transition arrays must have the same first dimension.")
    if action_masks.shape != (states.shape[0], OFFLOAD_NUM_CLASSES):
        raise ValueError(f"Unexpected action_masks shape {action_masks.shape}.")
    if np.any(actions < 0) or np.any(actions >= OFFLOAD_NUM_CLASSES):
        raise ValueError("CQL actions contain out-of-range values.")
    if np.any(np.take_along_axis(action_masks, actions[:, None], axis=1).squeeze(1) <= 0.0):
        raise ValueError("CQL dataset contains behavior actions masked as illegal.")

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "action_masks": action_masks,
    }


def split_transition_indices(sample_count: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 2:
        raise ValueError("CQL training requires at least two transitions.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    val_count = max(1, min(int(round(sample_count * val_ratio)), sample_count - 1))
    return indices[val_count:].astype(np.int64), indices[:val_count].astype(np.int64)


def evaluate_cql_policy(
    model: OffloadQNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    action_masks: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        state_tensor = torch.from_numpy(states).to(device)
        mask_tensor = torch.from_numpy(action_masks).to(device)
        q_values = model(state_tensor)
        masked_q_values = q_values.masked_fill(mask_tensor <= 0.0, -1.0e9)
        greedy_actions = torch.argmax(masked_q_values, dim=1).cpu().numpy()
        chosen_q = q_values.gather(1, torch.from_numpy(actions).to(device).unsqueeze(1)).squeeze(1).cpu().numpy()

    action_counts = np.bincount(greedy_actions, minlength=OFFLOAD_NUM_CLASSES)
    behavior_counts = np.bincount(actions, minlength=OFFLOAD_NUM_CLASSES)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        "mean_behavior_q": float(np.mean(chosen_q)) if chosen_q.size else 0.0,
        "greedy_action_counts": {OFFLOAD_TARGET_NAMES[idx]: int(action_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "behavior_action_counts": {OFFLOAD_TARGET_NAMES[idx]: int(behavior_counts[idx]) for idx in range(OFFLOAD_NUM_CLASSES)},
        "behavior_match_rate": float(np.mean(greedy_actions == actions)) if actions.size else 0.0,
    }


def train_cql_offload_policy(
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
    gamma: float = config.CQL_OFFLOAD_GAMMA,
    cql_alpha: float = config.CQL_OFFLOAD_ALPHA,
    target_update_freq: int = config.CQL_OFFLOAD_TARGET_UPDATE_FREQ,
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    arrays = load_cql_transition_dataset(dataset_path)
    train_idx, val_idx = split_transition_indices(arrays["states"].shape[0], val_ratio=val_ratio, seed=seed)
    x_train_raw = arrays["states"][train_idx]
    x_val_raw = arrays["states"][val_idx]
    x_train, x_val, scaler_mean, scaler_std = standardize_feature_splits(x_train_raw, x_val_raw)
    safe_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)
    next_states_std = ((arrays["next_states"] - scaler_mean.reshape(1, -1)) / safe_std.reshape(1, -1)).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(arrays["actions"][train_idx]),
        torch.from_numpy(arrays["rewards"][train_idx]),
        torch.from_numpy(next_states_std[train_idx]),
        torch.from_numpy(arrays["dones"][train_idx]),
        torch.from_numpy(arrays["action_masks"][train_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device_obj = select_device(device)
    input_dim = get_offload_feature_dim(OFFLOAD_FEATURE_FAMILY_RICH_REDUCED)
    model = OffloadQNetwork(input_dim=input_dim, hidden_dims=hidden_dims, num_actions=OFFLOAD_NUM_CLASSES).to(device_obj)
    target_model = copy.deepcopy(model).to(device_obj)
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, object] | None = None
    best_epoch = 0
    global_step = 0
    epoch_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_td_loss = 0.0
        epoch_cql_loss = 0.0
        batches = 0

        for batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_masks in train_loader:
            batch_states = batch_states.to(device_obj)
            batch_actions = batch_actions.to(device_obj)
            batch_rewards = batch_rewards.to(device_obj)
            batch_next_states = batch_next_states.to(device_obj)
            batch_dones = batch_dones.to(device_obj)
            batch_masks = batch_masks.to(device_obj)

            q_values = model(batch_states)
            chosen_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_model(batch_next_states)
                masked_next_q = next_q_values.masked_fill(batch_masks <= 0.0, -1.0e9)
                max_next_q = torch.max(masked_next_q, dim=1).values
                td_target = batch_rewards + float(gamma) * (1.0 - batch_dones) * max_next_q

            td_loss = mse_loss(chosen_q, td_target)
            cql_loss = torch.logsumexp(q_values, dim=1).mean() - chosen_q.mean()
            loss = td_loss + float(cql_alpha) * cql_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if target_update_freq > 0 and global_step % int(target_update_freq) == 0:
                target_model.load_state_dict(model.state_dict())

            epoch_loss += float(loss.item())
            epoch_td_loss += float(td_loss.item())
            epoch_cql_loss += float(cql_loss.item())
            batches += 1

        target_model.load_state_dict(model.state_dict())
        eval_summary = evaluate_cql_policy(
            model,
            x_val,
            arrays["actions"][val_idx],
            arrays["rewards"][val_idx],
            arrays["action_masks"][val_idx],
            device_obj,
        )
        mean_epoch_loss = epoch_loss / max(batches, 1)
        mean_td_loss = epoch_td_loss / max(batches, 1)
        mean_cql_loss = epoch_cql_loss / max(batches, 1)
        epoch_history.append({"epoch": float(epoch), "loss": mean_epoch_loss, "td_loss": mean_td_loss, "cql_loss": mean_cql_loss})
        if best_eval is None or mean_epoch_loss < float(best_eval["loss"]):
            best_state_dict = copy.deepcopy(model.state_dict())
            best_eval = {**eval_summary, "loss": float(mean_epoch_loss), "td_loss": float(mean_td_loss), "cql_loss": float(mean_cql_loss)}
            best_epoch = epoch

        print(
            f"epoch={epoch:03d} loss={mean_epoch_loss:.6f} td={mean_td_loss:.6f} "
            f"cql={mean_cql_loss:.6f} val_match={float(eval_summary['behavior_match_rate']):.4f}"
        )

    if best_state_dict is None or best_eval is None:
        raise RuntimeError("CQL training did not produce a valid checkpoint.")
    model.load_state_dict(best_state_dict)

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
        "gamma": float(gamma),
        "cql_alpha": float(cql_alpha),
        "target_update_freq": int(target_update_freq),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "best_epoch": int(best_epoch),
        "best_eval": best_eval,
        "epoch_history": epoch_history,
    }
    saved_path = save_cql_offload_policy_checkpoint(
        model,
        output_path,
        feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
        hidden_dims=hidden_dims,
        scaler_mean=scaler_mean.tolist(),
        scaler_std=scaler_std.tolist(),
        metrics=summary,
        extra_metadata={"dataset_seed": seed, "dataset_rows": int(arrays["states"].shape[0])},
    )
    summary["checkpoint_path"] = saved_path
    Path(saved_path).with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a constrained CQL-DQN request-level offloading policy.")
    parser.add_argument("--dataset", type=str, default="offload_datasets/offload_dataset_cql_transition.npz", help="Path to the CQL transition dataset (.npz).")
    parser.add_argument("--output", type=str, default="saved_offload_policies/offload_policy_cql.pt", help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--hidden_dims", type=str, default="128,128", help="Comma-separated hidden layer sizes.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument("--gamma", type=float, default=config.CQL_OFFLOAD_GAMMA, help="Discount factor.")
    parser.add_argument("--cql_alpha", type=float, default=config.CQL_OFFLOAD_ALPHA, help="CQL conservative regularization weight.")
    parser.add_argument("--target_update_freq", type=int, default=config.CQL_OFFLOAD_TARGET_UPDATE_FREQ, help="Target-network hard update frequency in gradient steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_cql_offload_policy(
        dataset_path=args.dataset,
        output_path=args.output,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        gamma=args.gamma,
        cql_alpha=args.cql_alpha,
        target_update_freq=args.target_update_freq,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
