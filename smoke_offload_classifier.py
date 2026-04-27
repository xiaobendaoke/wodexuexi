"""
中文注释说明：smoke_offload_classifier.py

文件作用：
    快速冒烟测试任务卸载分类器的训练、保存和推理流程是否可用。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - snapshot_config(): 全局配置模块，保存环境参数和训练超参数。
    - restore_config(): 全局配置模块，保存环境参数和训练超参数。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    copy, json, tempfile, pathlib, numpy, config, collect_offload_dataset, environment, marl_models, train_offload_policy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

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


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main() -> None:
    snapshot = snapshot_config()
    results: dict[str, object] = {}

    # 异常与收尾保护：确保关键流程出错时仍能执行清理、恢复或错误处理逻辑。
    try:
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
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


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
