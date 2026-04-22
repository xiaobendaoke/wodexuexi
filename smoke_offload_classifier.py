from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

import config
from collect_offload_dataset import collect_offload_dataset
from environment.env import Env
from marl_models.offload_policy import LearnedClassifierOffloadPolicy, ServiceOffloadContext
from train_offload_policy import train_offload_policy


def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key in dir(config):
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return snapshot


def restore_config(snapshot: dict[str, object]) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


def main() -> None:
    snapshot = snapshot_config()
    results: dict[str, object] = {}

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "offload_dataset_smoke.npz"
            checkpoint_path = tmp_path / "offload_policy_smoke.pt"

            results["dataset_collection"] = collect_offload_dataset(
                output_path=dataset_path,
                target_samples=128,
                max_steps=1500,
                seed=123,
            )

            results["training"] = train_offload_policy(
                dataset_path=dataset_path,
                output_path=checkpoint_path,
                epochs=4,
                batch_size=64,
                learning_rate=1e-3,
                val_ratio=0.2,
                seed=123,
                device="cpu",
            )

            policy = LearnedClassifierOffloadPolicy(checkpoint_path, device="cpu")
            manual_context = ServiceOffloadContext(
                local_latency=1.0,
                cooperative_latency=0.8,
                mbs_latency=0.2,
                deadline=1.0,
                priority=3,
                local_cache_hit=False,
                cooperative_available=True,
                local_queue_length=2,
            )
            results["single_request_inference"] = {
                "prediction": int(policy.predict(manual_context)),
                "checkpoint_exists": checkpoint_path.exists(),
            }

            config.SERVICE_OFFLOAD_POLICY = "learned"
            config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = str(checkpoint_path)
            np.random.seed(123)
            env = Env()
            env.reset()
            policy_loaded = all(getattr(uav, "_service_offload_policy", None) is not None for uav in env.uavs)
            _, rewards, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
            results["env_classifier_path"] = {
                "policy_loaded_for_all_uavs": bool(policy_loaded),
                "reward_agents": len(rewards),
                "processed_service_requests": float(metrics["service_requests_processed"]),
            }

            config.SERVICE_OFFLOAD_POLICY = "learned"
            config.SERVICE_OFFLOAD_POLICY_CHECKPOINT = str(tmp_path / "missing_checkpoint.pt")
            np.random.seed(456)
            env_fallback = Env()
            env_fallback.reset()
            fallback_used = all(getattr(uav, "_service_offload_policy", None) is None for uav in env_fallback.uavs)
            _, _, fallback_metrics = env_fallback.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
            results["fallback"] = {
                "fallback_used": bool(fallback_used),
                "processed_service_requests": float(fallback_metrics["service_requests_processed"]),
            }

        print(json.dumps({"status": "ok", "results": results}, indent=2, ensure_ascii=False))
    finally:
        restore_config(snapshot)


if __name__ == "__main__":
    main()
