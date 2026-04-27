"""
中文注释说明：main.py

文件作用：
    提供项目的主入口示例，负责创建环境和模型，并按配置执行训练或测试流程。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - configure_fp32_precision(): 关键函数，承载本模块的一段可复用实验逻辑。
    - configure_compile_backend(): 关键函数，承载本模块的一段可复用实验逻辑。
    - start_training(): 关键函数，承载本模块的一段可复用实验逻辑。
    - start_testing(): 关键函数，承载本模块的一段可复用实验逻辑。

主要依赖：
    marl_models, environment, train, test, utils, paths, config, torch, numpy, argparse

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from marl_models.base_model import MARLModel
from environment.env import Env
from marl_models.utils import get_model, load_step_count
from train import train_on_policy, train_off_policy, train_baselines
from test import test_model
from utils.logger import Logger, load_configs
from utils.plot_logs import generate_plots
from paths import results_path
import config
import torch
import numpy as np
import argparse
import warnings
import os
from datetime import datetime


# 函数 configure_fp32_precision：关键函数，承载本模块的一段可复用实验逻辑。
def configure_fp32_precision() -> None:
    """
    Use one TF32 control style consistently to avoid runtime conflicts with torch.compile.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Please use the new API settings to control TF32 behavior.*",
        category=UserWarning,
    )
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# 函数 configure_compile_backend：关键函数，承载本模块的一段可复用实验逻辑。
def configure_compile_backend() -> None:
    """
    On Windows, disable torch.compile to avoid runtime dependency on MSVC cl.exe.
    """
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if os.name == "nt":
        # 函数 _no_compile：关键函数，承载本模块的一段可复用实验逻辑，主要参数：module。
        def _no_compile(module, *args, **kwargs):
            # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
            return module

        torch.compile = _no_compile  # type: ignore[attr-defined]


configure_fp32_precision()
configure_compile_backend()


# 函数 start_training：关键函数，承载本模块的一段可复用实验逻辑，主要参数：args。
def start_training(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\nTraining started at {timestamp} for {args.num_episodes} episodes")

    resume_training: bool = args.resume_path is not None
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if resume_training:
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if args.config_path is None:
            # 主动报错：当输入或状态不满足实验前提时，立即给出明确错误。
            raise ValueError("If --resume_path is provided, --config_path must also be provided.")
        load_configs(args.config_path)  # Resume training with old config
    else:  # Fresh training
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if args.config_path is not None:
            warnings.warn("--config_path is ignored during training unless --resume_path is also provided.")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)

    model_log_dir: str = str(results_path("train_logs", model_name))
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)

    logger: Logger = Logger(model_log_dir, timestamp)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not resume_training:
        logger.log_configs()  # Save config for fresh training

    total_step_count: int = 0  # for off policy models
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if resume_training:
        model.load(args.resume_path)
        total_step_count = load_step_count(args.resume_path)
        print(f"Models loaded successfully from {args.resume_path}")
        print(f"Resumed training from: {args.resume_path}\n")

    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if model_name in ["maddpg", "attention_maddpg", "matd3", "attention_matd3", "masac", "attention_masac"]:
        train_off_policy(env, model, logger, args.num_episodes, total_step_count)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif model_name in ["mappo", "attention_mappo"]:
        train_on_policy(env, model, logger, args.num_episodes)
    else:  # "random", "static", "nearest_greedy", "uncoordinated_greedy"
        train_baselines(env, model, logger, args.num_episodes)

    print("Training completed.\n")
    print("Generating plots...")

    generate_plots(
        f"{model_log_dir}/log_data_{timestamp}.json",
        str(results_path("train_plots", model_name)),
        "train",
        timestamp,
    )


# 函数 start_testing：关键函数，承载本模块的一段可复用实验逻辑，主要参数：args。
def start_testing(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\nTesting started at {timestamp} for {args.num_episodes} episodes")

    load_configs(args.config_path)

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)

    model_log_dir: str = str(results_path("test_logs", model_name))
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)

    logger: Logger = Logger(model_log_dir, timestamp)

    model.load(args.model_path)
    print(f"Models loaded successfully from {args.model_path}")

    test_model(env, model, logger, args.num_episodes)

    print("Testing completed.\n")
    print("Generating plots...")

    generate_plots(
        f"{model_log_dir}/log_data_{timestamp}.json",
        str(results_path("test_plots", model_name)),
        "test",
        timestamp,
        smoothing_window=2,
    )


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--num_episodes", type=int, required=True)
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.add_argument("--resume_path", type=str, default=None)
    train_parser.add_argument("--config_path", type=str, default=None)

    test_parser = subparsers.add_parser("test", parents=[parent_parser])
    test_parser.add_argument("--model_path", type=str, required=True)
    test_parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    if args.mode == "train":
        start_training(args)
    # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
    elif args.mode == "test":
        start_testing(args)
    print("All done!")
