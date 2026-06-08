#!/usr/bin/env python3
"""测试敏感性实验脚本是否工作。"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import config
import numpy as np
from environment.env import Env
from marl_models.utils import get_model

def test_environment():
    """测试环境是否正常工作。"""
    print("Testing environment...")

    # 测试不同UE数量
    for ue_count in [60, 80, 100, 120, 140]:
        original_num_ues = config.NUM_UES
        config.NUM_UES = ue_count

        env = Env()
        obs = env.reset()
        print(f"UE count={ue_count}: obs shape={np.array(obs).shape}")

        config.NUM_UES = original_num_ues

    print("Environment test passed!")

def test_model_loading():
    """测试模型加载。"""
    print("\nTesting model loading...")

    # 测试proposed模型
    model = get_model("attention_mappo")
    print(f"Created attention_mappo model")

    # 测试heuristic模型
    heuristic_model = get_model("uncoordinated_greedy_baseline")
    print(f"Created uncoordinated_greedy_baseline model")

    print("Model loading test passed!")

def test_episode_run():
    """测试运行单个episode。"""
    print("\nTesting episode run...")

    from utils.baseline_metrics import run_single_episode, set_global_seed

    set_global_seed(42)
    env = Env()
    model = get_model("uncoordinated_greedy_baseline")

    episode_metrics, _ = run_single_episode(
        env, model, offload_model=None,
        exploration=False, record_trajectory=False,
        policy_type="non_learning",
    )

    print(f"Episode metrics keys: {list(episode_metrics.keys())}")
    print(f"Energy efficiency: {episode_metrics.get('energy_efficiency', 'N/A')}")
    print(f"Deadline satisfaction rate: {episode_metrics.get('deadline_satisfaction_rate', 'N/A')}")

    print("Episode run test passed!")

def main():
    print("="*60)
    print("Testing sensitivity experiment components")
    print("="*60)

    test_environment()
    test_model_loading()
    test_episode_run()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()
