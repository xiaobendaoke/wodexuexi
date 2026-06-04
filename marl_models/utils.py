"""
中文注释说明：marl_models/utils.py

文件作用：
    实现多智能体强化学习相关模型组件，供训练、测试和算法对比脚本调用。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - get_device(): 模型训练和推理使用的计算设备。
    - get_model(): 当前训练或测试的多智能体模型实例。
    - save_models(): 保存模型参数或实验结果。
    - load_step_count(): 加载模型参数或实验数据。

主要依赖：
    marl_models, config, torch, os

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel
from marl_models.attention_mappo.attention_mappo import AttentionMAPPO
from marl_models.joint_mappo.joint_mappo import JointMAPPO
from marl_models.offload_mappo.offload_mappo import OffloadMAPPO
from marl_models.uncoordinated_greedy_baseline.uncoordinated_greedy_model import UncoordinatedGreedyModel
from marl_models.vanilla_mappo.vanilla_mappo import VanillaMAPPO
import config
import torch
import os


# 函数 get_device：模型训练和推理使用的计算设备。
def get_device() -> str:
    """Check if GPU is available and set device accordingly."""
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if torch.cuda.is_available():
        print("\nFound GPU, using CUDA.\n")
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return "cuda"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif torch.backends.mps.is_available():
        print("\nUsing MPS (Apple Silicon GPU).\n")
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return "mps"
    else:
        print("\nNo GPU available, using CPU.\n")
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return "cpu"


# 函数 get_model：当前训练或测试的多智能体模型实例，主要参数：model_name。
def get_model(model_name: str) -> MARLModel:
    device = get_device()
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if model_name == "attention_mappo":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return AttentionMAPPO(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    if model_name == "vanilla_mappo":
        return VanillaMAPPO(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    if model_name == "joint_mappo":
        return JointMAPPO(
            model_name=model_name,
            num_agents=config.NUM_UAVS,
            obs_dim=config.OBS_DIM_SINGLE + config.OFFLOAD_OBS_DIM_SINGLE,
            action_dim=config.ACTION_DIM,
            device=device,
        )
    elif model_name in {"offload_mappo", "constrained_attention_offload_mappo", "no_attention_offload_mappo"}:
        return OffloadMAPPO(
            model_name=model_name,
            num_agents=config.NUM_UAVS,
            obs_dim=config.OFFLOAD_OBS_DIM_SINGLE,
            action_dim=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
            device=device,
        )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif model_name == "uncoordinated_greedy":
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return UncoordinatedGreedyModel(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "random_baseline":
        from marl_models.random_baseline.random_baseline import RandomBaseline
        return RandomBaseline(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "uniform_baseline":
        from marl_models.uniform_baseline.uniform_baseline import UniformBaseline
        return UniformBaseline(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "ippo_baseline":
        from marl_models.ippo_baseline.ippo_baseline import IPPOBaseline
        return IPPOBaseline(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    else:
        # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
        raise ValueError(f"Unknown model type: {model_name}.")


# 函数 save_models：保存模型参数或实验结果，主要参数：model, progress_step, name, timestamp, final, total_steps。
def save_models(model: MARLModel, progress_step: int, name: str, timestamp: str, final: bool = False, total_steps: int = 0):
    save_dir: str = f"saved_models/{model.model_name}_{timestamp}"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if final:
        save_dir = f"{save_dir}/final"
    else:
        save_dir = f"{save_dir}/{name.lower()}_{progress_step:04d}"
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if total_steps > 0:
        step_count_path: str = os.path.join(save_dir, "total_steps.txt")
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with open(step_count_path, "w") as f:
            f.write(str(total_steps))

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if final:
        print(f"Final models saved in: {save_dir}\n")
    else:
        print(f"Models saved for {name.lower()} {progress_step} in: {save_dir}\n")


# 函数 load_step_count：加载模型参数或实验数据，主要参数：directory。
def load_step_count(directory: str) -> int:
    step_count_path: str = os.path.join(directory, "total_steps.txt")
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if os.path.exists(step_count_path):
        # 资源上下文：集中管理文件、图像或推理模式等需要成对进入和退出的资源。
        with open(step_count_path, "r") as f:
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return int(f.read().strip())
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return 0
