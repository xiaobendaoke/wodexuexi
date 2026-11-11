"""Bounded remote-only environment integration validation; no formal training.

Run from the repository root: python tests/environment_validation_smoke.py
"""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() != "Linux":
    raise SystemExit("Run this integration check on the remote Linux server.")

import numpy as np
import torch
import config
from environment.env import Env
from marl_models.utils import get_model


def check_transition(env, obs, rewards, metrics, before):
    assert np.isfinite(obs).all() and np.isfinite(rewards).all()
    assert all(np.isfinite(v) for v in metrics.values())
    positions = np.array([u.pos[:2] for u in env.uavs])
    env._validate_positions(positions)
    assert np.all(np.linalg.norm(positions - before, axis=1) <= config.UAV_SPEED * config.TIME_SLOT_DURATION + 1e-3)
    assert metrics["generated_service_count"] == metrics["admitted_service_count"] + metrics["unadmitted_service_count"]
    assert metrics["processed_service_count"] == metrics["admitted_service_count"]
    for uav in env.uavs:
        assert np.sum(uav.cache * config.FILE_SIZES) <= config.UAV_STORAGE_CAPACITY[uav.id]


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
              "formal_training_started": False, "source_sha256": {}, "scenarios": [], "upper_smokes": {}}
    for name in ("config.py", "environment/env.py", "environment/uavs.py"):
        report["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for seed in (42, 2026):
        for mode in ("heuristic", "local", "mbs", "cooperative"):
            np.random.seed(seed)
            env = Env()
            env.reset()
            actions_rng = np.random.default_rng(seed + 100)
            total_admitted = 0
            for _ in range(100):
                before = np.array([u.pos[:2].copy() for u in env.uavs])
                lower = None
                if mode != "heuristic":
                    lower = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
                    if mode == "mbs":
                        lower[:] = config.OFFLOAD_ACTION_MBS
                    elif mode == "cooperative":
                        for i, uav in enumerate(env.uavs):
                            if uav.neighbors:
                                lower[i] = config.OFFLOAD_ACTION_COOP_BASE + min(n.id for n in uav.neighbors)
                obs, rewards, metrics = env.step(actions_rng.uniform(-1, 1, (config.NUM_UAVS, 2)), offloading_actions=lower)
                check_transition(env, obs, rewards, metrics, before)
                total_admitted += int(metrics["admitted_service_count"])
            report["scenarios"].append({"seed": seed, "mode": mode, "steps": 100, "admitted_requests": total_admitted})

    # One optimizer step per upper model verifies compatibility with real
    # environment observations, without checkpoints or experiment outputs.
    for name in ("attention_mappo", "vanilla_mappo"):
        np.random.seed(42)
        torch.manual_seed(42)
        env = Env()
        obs = np.asarray(env.reset(), dtype=np.float32)
        model = get_model(name)
        state = obs.flatten()
        actions, log_probs, values = model.get_action_and_value(obs, state)
        before = np.array([u.pos[:2].copy() for u in env.uavs])
        next_obs, rewards, metrics = env.step(actions)
        check_transition(env, next_obs, rewards, metrics, before)
        batch = {"obs": torch.tensor(obs[None]), "states": torch.tensor(state[None]),
                 "actions": torch.tensor(actions[None]), "old_log_probs": torch.tensor(log_probs[None]),
                 "old_values": torch.tensor(values[None]), "advantages": torch.ones((1, config.NUM_UAVS)),
                 "returns": torch.tensor(values[None]) + 0.1}
        losses = model.update(batch)
        assert all(np.isfinite(v) for v in losses.values())
        for module in (model.actor, model.critic):
            assert all(torch.isfinite(p).all() for p in module.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
        report["upper_smokes"][name] = {"optimizer_steps": 1, "losses": losses, "status": "PASS"}
    report["elapsed_seconds"] = time.monotonic() - start
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
