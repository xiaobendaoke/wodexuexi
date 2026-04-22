from __future__ import annotations

import copy
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np

import config
from environment.env import Env
from environment.request_types import Request
from utils.logger import load_configs, refresh_derived_config_fields
from utils.plot_logs import generate_plots


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


def check_request_api_and_observation_shape() -> dict[str, object]:
    np.random.seed(101)
    env = Env()
    obs = env.reset()
    assert len(obs) == config.NUM_UAVS
    assert all(ob.shape == (config.OBS_DIM_SINGLE,) for ob in obs)
    sample_request = env.ues[0].current_request
    assert hasattr(sample_request, "is_service")
    assert hasattr(sample_request, "is_content")
    assert hasattr(sample_request, "is_energy")
    return {
        "obs_agents": len(obs),
        "obs_dim": int(obs[0].shape[0]),
        "request_type": int(sample_request.req_type),
    }


def prepare_controlled_env(policy_name: str) -> tuple[Env, float, float, float]:
    np.random.seed(123)
    config.SERVICE_OFFLOAD_POLICY = policy_name
    env = Env()
    env.reset()

    for uav in env.uavs:
        uav.current_covered_ues.clear()
    for ue in env.ues:
        ue.assigned = False
        ue.battery_level = config.UE_BATTERY_CAPACITY
        ue.current_request = Request.content(req_id=config.NUM_SERVICES)

    anchor_uav = env.uavs[0]
    anchor_xy = anchor_uav.pos[:2].copy()

    service_ue = env.ues[0]
    content_ue = env.ues[1]
    energy_ue = env.ues[2]

    service_ue.pos[:2] = anchor_xy + np.array([1.0, 1.0], dtype=np.float32)
    service_ue.current_request = Request.service(
        req_size=config.MIN_INPUT_SIZE,
        req_id=0,
        deadline=1.5 * config.TIME_SLOT_DURATION,
        priority=config.SERVICE_PRIORITY_MAX,
    )

    content_ue.pos[:2] = anchor_xy + np.array([2.0, 0.0], dtype=np.float32)
    content_ue.current_request = Request.content(req_id=config.NUM_SERVICES)

    energy_ue.pos[:2] = anchor_xy + np.array([0.0, 2.0], dtype=np.float32)
    energy_ue.battery_level = 0.2 * config.UE_CRITICAL_THRESHOLD
    energy_ue.current_request = Request.energy()

    service_before = float(service_ue.battery_level)
    content_before = float(content_ue.battery_level)
    energy_before = float(energy_ue.battery_level)

    env._associate_ues_to_uavs()
    for uav in env.uavs:
        uav.set_neighbors(env.uavs)

    return env, service_before, content_before, energy_before


def check_branch_behavior(policy_name: str) -> dict[str, object]:
    env, service_before, content_before, energy_before = prepare_controlled_env(policy_name)
    _, rewards, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
    assert len(rewards) == config.NUM_UAVS
    assert metrics["service_requests_processed"] >= 1.0
    assert abs(
        metrics["service_offloads_local"] + metrics["service_offloads_cooperative"] + metrics["service_offloads_mbs"] - metrics["service_requests_processed"]
    ) < 1e-9
    assert float(env.ues[2].battery_level) > energy_before
    assert float(env.ues[1].battery_level) < content_before
    assert float(env.ues[0].battery_level) < service_before
    return {
        "policy": policy_name,
        "service_requests_processed": float(metrics["service_requests_processed"]),
        "offloading_ratio_mbs": float(metrics["offloading_ratio_mbs"]),
    }


def check_metric_conventions(policy_name: str) -> dict[str, object]:
    np.random.seed(777)
    config.SERVICE_OFFLOAD_POLICY = policy_name
    env = Env()
    env.reset()
    _, _, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))

    processed = float(metrics["service_requests_processed"])
    generated = float(metrics["service_requests_generated"])
    ratio_sum = (
        float(metrics["offloading_ratio_local"])
        + float(metrics["offloading_ratio_cooperative"])
        + float(metrics["offloading_ratio_mbs"])
    )
    if processed > 0:
        assert np.isclose(ratio_sum, 1.0)
    else:
        assert np.isclose(ratio_sum, 0.0)
    if generated > 0:
        assert np.isclose(float(metrics["mbs_load_ratio"]), float(metrics["service_offloads_mbs"]) / generated)
    else:
        assert np.isclose(float(metrics["mbs_load_ratio"]), 0.0)
        assert np.isclose(float(metrics["deadline_satisfaction_rate"]), 0.0)
    assert 0.0 <= float(metrics["deadline_satisfaction_rate"]) <= 1.0
    return {
        "policy": policy_name,
        "ratio_sum": float(ratio_sum),
        "processed_service_requests": processed,
        "generated_service_requests": generated,
    }


def check_single_uav_safety() -> dict[str, object]:
    config.NUM_UAVS = 1
    config.SERVICE_OFFLOAD_POLICY = "learned"
    refresh_derived_config_fields()

    np.random.seed(321)
    env = Env()
    obs = env.reset()
    assert len(obs) == 1
    _, _, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
    assert np.isclose(float(metrics["offloading_ratio_cooperative"]), 0.0)
    return {
        "obs_agents": len(obs),
        "max_uav_neighbors": int(config.MAX_UAV_NEIGHBORS),
        "offloading_ratio_cooperative": float(metrics["offloading_ratio_cooperative"]),
    }


def check_invalid_policy_fallback() -> dict[str, object]:
    np.random.seed(202)
    config.SERVICE_OFFLOAD_POLICY = "invalid_policy_name"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = Env()
        env.reset()
        assert all(getattr(uav, "_service_offload_policy", None) is None for uav in env.uavs)
        _, _, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
    return {
        "service_requests_processed": float(metrics["service_requests_processed"]),
        "fallback_used": True,
        "warnings_captured": len(caught),
    }


def check_old_log_compatibility() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        log_path = tmp_path / "legacy_log.json"
        output_dir = tmp_path / "plots"
        legacy_entries = [
            {
                "episode": 1,
                "reward": 1.0,
                "latency": 2.0,
                "energy": 3.0,
                "fairness": 0.5,
                "offline_rate": 0.1,
                "time": 0.2,
            }
        ]
        log_path.write_text(json.dumps(legacy_entries, indent=2), encoding="utf-8")
        generate_plots(str(log_path), str(output_dir), "legacy", "smoke", smoothing_window=1)
        assert any(output_dir.glob("*.png"))
        return {"legacy_plot_files": len(list(output_dir.glob("*.png")))}


def check_old_config_defaults() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config_path = tmp_path / "legacy_config.json"
        legacy_config = {
            "NUM_UAVS": 5,
            "NUM_UES": 100,
            "NUM_SERVICES": 25,
            "NUM_CONTENTS": 50,
            "FILE_SIZES": config.FILE_SIZES.tolist(),
            "UAV_STORAGE_CAPACITY": config.UAV_STORAGE_CAPACITY.tolist(),
            "UAV_COMPUTING_CAPACITY": config.UAV_COMPUTING_CAPACITY.tolist(),
            "SELF_OBS_DIM": 77,
            "UE_OBS_DIM": 6,
            "OBS_DIM_SINGLE": 267,
        }
        config_path.write_text(json.dumps(legacy_config, indent=2), encoding="utf-8")
        load_configs(str(config_path))
        expected_obs_dim = config.SELF_OBS_DIM + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM) + (config.MAX_ASSOCIATED_UES * config.UE_OBS_DIM)
        assert config.REQUEST_OBS_DIM == 5
        assert config.UE_OBS_DIM == 2 + config.REQUEST_OBS_DIM + 1
        assert config.OBS_DIM_SINGLE == expected_obs_dim
        return {
            "request_obs_dim": int(config.REQUEST_OBS_DIM),
            "ue_obs_dim": int(config.UE_OBS_DIM),
            "obs_dim_single": int(config.OBS_DIM_SINGLE),
            "service_offload_policy_default": str(config.SERVICE_OFFLOAD_POLICY),
        }


def main() -> None:
    snapshot = snapshot_config()
    results: dict[str, object] = {}
    try:
        results["request_api_and_observation_shape"] = check_request_api_and_observation_shape()
        restore_config(snapshot)

        results["controlled_branch_behavior_heuristic"] = check_branch_behavior("heuristic")
        restore_config(snapshot)

        results["controlled_branch_behavior_learned"] = check_branch_behavior("learned")
        restore_config(snapshot)

        results["metric_conventions_heuristic"] = check_metric_conventions("heuristic")
        restore_config(snapshot)

        results["metric_conventions_learned"] = check_metric_conventions("learned")
        restore_config(snapshot)

        results["single_uav_safety"] = check_single_uav_safety()
        restore_config(snapshot)

        results["invalid_policy_fallback"] = check_invalid_policy_fallback()
        restore_config(snapshot)

        results["old_log_compatibility"] = check_old_log_compatibility()
        restore_config(snapshot)

        results["old_config_defaults"] = check_old_config_defaults()
        restore_config(snapshot)

        print(json.dumps({"status": "ok", "results": results}, indent=2, ensure_ascii=False))
    finally:
        restore_config(snapshot)


if __name__ == "__main__":
    main()
