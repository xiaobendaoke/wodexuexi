"""
中文注释说明：smoke_offloading_consistency.py

文件作用：
    快速检查任务卸载相关接口和标签生成逻辑在不同调用路径下是否保持一致。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - snapshot_config(): 全局配置模块，保存环境参数和训练超参数。
    - restore_config(): 全局配置模块，保存环境参数和训练超参数。
    - check_request_api_and_observation_shape(): 用户设备产生的任务请求。
    - prepare_controlled_env(): 仿真环境对象，承载无人机、用户设备、任务请求和奖励计算。
    - check_branch_behavior(): 关键函数，承载本模块的一段可复用实验逻辑。
    - check_metric_conventions(): 关键函数，承载本模块的一段可复用实验逻辑。
    - check_single_uav_safety(): 无人机对象，包含位置、电量、计算资源和缓存服务。
    - check_invalid_policy_fallback(): 关键函数，承载本模块的一段可复用实验逻辑。
    - check_old_log_compatibility(): 关键函数，承载本模块的一段可复用实验逻辑。
    - check_old_config_defaults(): 全局配置模块，保存环境参数和训练超参数。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    copy, json, tempfile, warnings, pathlib, numpy, config, environment, utils

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

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
from environment.uavs import OFFLOAD_TARGET_MBS, _tx_latency_bytes
from utils.logger import load_configs, refresh_derived_config_fields
from utils.plot_logs import generate_plots


# 函数 snapshot_config：全局配置模块，保存环境参数和训练超参数。
def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    # 循环处理：遍历 key 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for key in dir(config):
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return snapshot


# 函数 restore_config：全局配置模块，保存环境参数和训练超参数，主要参数：snapshot。
def restore_config(snapshot: dict[str, object]) -> None:
    # 循环处理：遍历 (key, value) 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


# 函数 check_request_api_and_observation_shape：用户设备产生的任务请求。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "obs_agents": len(obs),
        "obs_dim": int(obs[0].shape[0]),
        "request_type": int(sample_request.req_type),
    }


def check_transmission_unit_conversion() -> dict[str, object]:
    latency = _tx_latency_bytes(1.0, 8.0)
    assert np.isclose(latency, 1.0)
    return {"one_byte_over_8bps_latency": float(latency)}


def check_mbs_service_compute_latency() -> dict[str, object]:
    np.random.seed(404)
    env = Env()
    env.reset()
    uav = env.uavs[0]
    uav._uav_mbs_rate = 16.0
    request = Request.service(req_size=4, req_id=0, deadline=10.0, priority=config.SERVICE_PRIORITY_MAX)
    ue_uav_rate = 32.0
    cpu_cycles = float(config.CPU_CYCLES_PER_BYTE[0]) * float(request.req_size)
    pure_transmission = _tx_latency_bytes(request.req_size, ue_uav_rate) + _tx_latency_bytes(request.req_size, uav._uav_mbs_rate)
    expected_compute = cpu_cycles / float(config.MBS_COMPUTING_CAPACITY)
    estimated = uav._estimate_mbs_service_latency(request, ue_uav_rate)
    assert np.isclose(estimated - pure_transmission, expected_compute)
    return {
        "pure_transmission_latency": float(pure_transmission),
        "mbs_compute_latency": float(expected_compute),
        "estimated_mbs_latency": float(estimated),
    }


def check_content_receive_energy() -> dict[str, object]:
    np.random.seed(505)
    env = Env()
    env.reset()
    uav = env.uavs[0]
    ue = env.ues[0]
    req_id = config.NUM_SERVICES
    ue.current_request = Request.content(req_id=req_id)
    ue.battery_level = config.UE_BATTERY_CAPACITY
    before = float(ue.battery_level)
    rate = float(config.FILE_SIZES[req_id]) * float(config.BITS_PER_BYTE)
    expected_receive_time = _tx_latency_bytes(config.FILE_SIZES[req_id], rate)
    uav._uav_mbs_rate = rate
    uav._process_content_request(ue, rate, OFFLOAD_TARGET_MBS, None)
    expected_drop = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION + config.UE_RECEIVE_POWER * expected_receive_time
    actual_drop = before - float(ue.battery_level)
    assert np.isclose(actual_drop, expected_drop)
    return {
        "receive_time": float(expected_receive_time),
        "battery_drop": float(actual_drop),
    }


def check_uav_communication_energy_mbs_paths() -> dict[str, object]:
    np.random.seed(606)
    env = Env()
    env.reset()
    uav = env.uavs[0]
    service_ue = env.ues[0]
    content_ue = env.ues[1]

    service_req = Request.service(req_size=4, req_id=0, deadline=10.0, priority=config.SERVICE_PRIORITY_MAX)
    service_ue.current_request = service_req
    service_ue.battery_level = config.UE_BATTERY_CAPACITY
    ue_uav_rate = 32.0
    uav._uav_mbs_rate = 16.0
    uav._energy_current_slot = 0.0
    service_upload = _tx_latency_bytes(service_req.req_size, ue_uav_rate)
    service_backhaul = _tx_latency_bytes(service_req.req_size, uav._uav_mbs_rate)
    expected_service_comm = (
        config.UAV_COMM_RX_POWER * service_upload
        + config.UAV_BACKHAUL_TX_POWER * service_backhaul
    )
    uav._process_service_request(service_ue, ue_uav_rate, OFFLOAD_TARGET_MBS, None)
    assert np.isclose(float(uav.energy), expected_service_comm)

    content_req_id = config.NUM_SERVICES
    content_ue.current_request = Request.content(req_id=content_req_id)
    content_ue.battery_level = config.UE_BATTERY_CAPACITY
    file_size = float(config.FILE_SIZES[content_req_id])
    content_rate = file_size * float(config.BITS_PER_BYTE)
    uav._uav_mbs_rate = content_rate
    uav._energy_current_slot = 0.0
    content_ue_uav_download = _tx_latency_bytes(file_size, content_rate)
    content_backhaul = _tx_latency_bytes(file_size, uav._uav_mbs_rate)
    expected_content_comm = (
        config.UAV_COMM_TX_POWER * content_ue_uav_download
        + config.UAV_BACKHAUL_RX_POWER * content_backhaul
    )
    uav._process_content_request(content_ue, content_rate, OFFLOAD_TARGET_MBS, None)
    assert np.isclose(float(uav.energy), expected_content_comm)

    return {
        "service_mbs_comm_energy": float(expected_service_comm),
        "content_mbs_comm_energy": float(expected_content_comm),
    }


# 函数 prepare_controlled_env：仿真环境对象，承载无人机、用户设备、任务请求和奖励计算，主要参数：policy_name。
def prepare_controlled_env(policy_name: str) -> tuple[Env, float, float, float]:
    np.random.seed(123)
    config.SERVICE_OFFLOAD_POLICY = policy_name
    env = Env()
    env.reset()

    # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for uav in env.uavs:
        uav.current_covered_ues.clear()
    # 循环处理：遍历 ue 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
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
    # 循环处理：遍历 uav 对应的数据集合，逐项执行环境交互、训练更新或结果统计。
    for uav in env.uavs:
        uav.set_neighbors(env.uavs)

    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return env, service_before, content_before, energy_before


# 函数 check_branch_behavior：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_name。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "policy": policy_name,
        "service_requests_processed": float(metrics["service_requests_processed"]),
        "offloading_ratio_mbs": float(metrics["offloading_ratio_mbs"]),
    }


# 函数 check_metric_conventions：关键函数，承载本模块的一段可复用实验逻辑，主要参数：policy_name。
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
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if processed > 0:
        assert np.isclose(ratio_sum, 1.0)
    else:
        assert np.isclose(ratio_sum, 0.0)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if generated > 0:
        assert np.isclose(float(metrics["mbs_load_ratio"]), float(metrics["service_offloads_mbs"]) / generated)
    else:
        assert np.isclose(float(metrics["mbs_load_ratio"]), 0.0)
        assert np.isclose(float(metrics["deadline_satisfaction_rate"]), 0.0)
    assert 0.0 <= float(metrics["deadline_satisfaction_rate"]) <= 1.0
    assert 0.0 <= float(metrics["mbs_load_ratio"]) <= 1.0
    assert 0.0 <= float(metrics["offloading_ratio_local"]) <= 1.0
    assert 0.0 <= float(metrics["offloading_ratio_cooperative"]) <= 1.0
    assert 0.0 <= float(metrics["offloading_ratio_mbs"]) <= 1.0
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "policy": policy_name,
        "ratio_sum": float(ratio_sum),
        "processed_service_requests": processed,
        "generated_service_requests": generated,
    }


# 函数 check_single_uav_safety：无人机对象，包含位置、电量、计算资源和缓存服务。
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
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "obs_agents": len(obs),
        "max_uav_neighbors": int(config.MAX_UAV_NEIGHBORS),
        "offloading_ratio_cooperative": float(metrics["offloading_ratio_cooperative"]),
    }


# 函数 check_invalid_policy_fallback：关键函数，承载本模块的一段可复用实验逻辑。
def check_invalid_policy_fallback() -> dict[str, object]:
    np.random.seed(202)
    config.SERVICE_OFFLOAD_POLICY = "invalid_policy_name"
    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = Env()
        env.reset()
        assert all(getattr(uav, "_service_offload_policy", None) is None for uav in env.uavs)
        _, _, metrics = env.step(np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32))
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return {
        "service_requests_processed": float(metrics["service_requests_processed"]),
        "fallback_used": True,
        "warnings_captured": len(caught),
    }


# 函数 check_old_log_compatibility：关键函数，承载本模块的一段可复用实验逻辑。
def check_old_log_compatibility() -> dict[str, object]:
    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
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
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {"legacy_plot_files": len(list(output_dir.glob("*.png")))}


# 函数 check_old_config_defaults：全局配置模块，保存环境参数和训练超参数。
def check_old_config_defaults() -> dict[str, object]:
    # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
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
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return {
            "request_obs_dim": int(config.REQUEST_OBS_DIM),
            "ue_obs_dim": int(config.UE_OBS_DIM),
            "obs_dim_single": int(config.OBS_DIM_SINGLE),
            "service_offload_policy_default": str(config.SERVICE_OFFLOAD_POLICY),
        }


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    snapshot = snapshot_config()
    results: dict[str, object] = {}
    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        results["request_api_and_observation_shape"] = check_request_api_and_observation_shape()
        restore_config(snapshot)

        results["transmission_unit_conversion"] = check_transmission_unit_conversion()
        restore_config(snapshot)

        results["mbs_service_compute_latency"] = check_mbs_service_compute_latency()
        restore_config(snapshot)

        results["content_receive_energy"] = check_content_receive_energy()
        restore_config(snapshot)

        results["uav_communication_energy_mbs_paths"] = check_uav_communication_energy_mbs_paths()
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


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
