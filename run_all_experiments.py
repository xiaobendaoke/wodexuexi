"""
中文注释说明：run_all_experiments.py

文件作用：
    一键运行完整论文实验套件，是本仓库最高层的实验调度入口。它把多智能体强化学习算法训练/测试、
    训练曲线对比、请求级任务卸载分类器流水线、离线有效性评估和高级论文有效性验证统一串起来，
    最终生成一个 experiment_manifest.json，记录每个子实验的路径、耗时和核心结果。

整体流程：
    1. 解析命令行参数，确定要运行的 RL 算法列表、训练/测试回合数和卸载分类器参数。
    2. 配置 PyTorch 浮点矩阵精度，并在 Windows 上关闭 torch.compile，规避本地编译后端不稳定问题。
    3. 备份 config.py 中的全局大写配置项，避免不同子实验临时改配置后相互污染。
    4. 如果未跳过 RL，则逐个训练/测试 ALL_MODELS 中的算法，复制模型权重，生成单算法曲线和算法对比图。
    5. 如果未跳过卸载实验，则调用 run_full_offload_experiment() 训练卸载分类器并运行运行时策略对比。
    6. 可选运行 evaluate_offload_validity.py 和 verify_offload_policy_validity.py，补充论文有效性报告。
    7. 无论子实验是否成功，finally 中都会恢复 config 快照，最后写出总 manifest。

关键变量与对象：
    - ALL_MODELS: 默认纳入总实验的算法列表，包含 8 个学习型算法和 4 个基线策略。
    - run_name: 本次完整实验的结果目录名，对应 results/full_runs/{run_name}。
    - base_snapshot: config.py 全局配置快照，用于在 RL 与卸载实验之间恢复初始配置。
    - run_root: 本次完整实验的根目录，存放训练日志、测试日志、对比图、报告和 manifest。
    - timestamp: 单个 RL 算法运行的时间戳，保证日志目录和保存模型目录一一对应。
    - train_logger / test_logger: 分别记录训练和测试阶段指标的 Logger 实例。
    - copied_model_dir / final_model_dir: 从 saved_models 复制到本次 run_root 下的模型目录。
    - validity_dataset: 用于离线论文有效性评估的数据集路径；空字符串会被转换为 None 以跳过该阶段。
    - summary: 最终写入 experiment_manifest.json 的总实验索引。
    - snapshot_config() / restore_config(): 防止子实验修改全局配置后影响后续实验。
    - run_single_rl_experiment(): 单个 RL 算法的训练、绘图、保存和测试闭环。
    - run_rl_suite(): 多算法批量运行与训练曲线对比。
    - run_offload_suite(): 卸载分类器流水线与有效性验证的总调度。
    - parse_hidden_dims(): 将命令行中的隐藏层字符串转换为分类器训练所需的整数元组。
    - parse_args(): 解析命令行参数，并为实验脚本提供可覆盖的默认配置。
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    argparse, copy, json, os, shutil, time, warnings, datetime, pathlib, numpy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
import verify_offload_policy_validity as advanced_offload_validity
from environment.env import Env
from evaluate_offload_validity import run_validity_evaluation
from marl_models.base_model import MARLModel
from marl_models.utils import get_model
from paths import results_path
from run_full_offload_experiment import run_full_offload_experiment
from test import test_model
from train import train_baselines, train_off_policy, train_on_policy
from utils.comparative_plots import compare_algorithms
from utils.logger import Logger
from utils.plot_logs import generate_plots


# 关键变量 ALL_MODELS：默认完整实验会覆盖的算法集合，前 8 个是学习型 MARL，后 4 个是对照基线。
ALL_MODELS: tuple[str, ...] = (
    "maddpg",
    "matd3",
    "mappo",
    "masac",
    "attention_maddpg",
    "attention_matd3",
    "attention_mappo",
    "attention_masac",
    "random",
    "static",
    "nearest_greedy",
    "uncoordinated_greedy",
)


# 函数 configure_fp32_precision：配置 PyTorch 的 FP32/TF32 行为，在支持的 GPU 上提升矩阵运算效率。
def configure_fp32_precision() -> None:
    # 过滤 PyTorch 关于旧 TF32 API 的提醒，避免长时间实验日志被重复 warning 干扰。
    warnings.filterwarnings(
        "ignore",
        message=r"Please use the new API settings to control TF32 behavior.*",
        category=UserWarning,
    )
    # 只有 CUDA 可用时才开启 TF32；CPU 或其他后端不需要设置 cuda/cudnn 开关。
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # high 精度是 PyTorch 推荐的折中项，通常比 highest 更快，也比默认设置更适合训练吞吐。
    torch.set_float32_matmul_precision("high")


# 函数 configure_compile_backend：在 Windows 环境下关闭 torch.compile，减少本地实验的编译后端兼容问题。
def configure_compile_backend() -> None:
    # Windows 上 torch.compile 可能依赖额外编译链；这里替换成直通函数，保持模型代码可以统一调用。
    if os.name == "nt":
        # 函数 _no_compile：保持和 torch.compile 相同的调用形态，但直接返回原模块。
        def _no_compile(module, *args, **kwargs):
            # 返回原始 module，表示不做图编译优化。
            return module

        torch.compile = _no_compile  # type: ignore[attr-defined]


# 函数 snapshot_config：复制 config.py 中所有大写配置项，作为完整实验开始前的干净基线。
def snapshot_config() -> dict[str, object]:
    snapshot: dict[str, object] = {}
    # 遍历 config 模块中的名字，只保存约定为全局配置的大写变量。
    for key in dir(config):
        # 跳过 __dunder__ 名称，避免把 Python 模块内部属性纳入配置快照。
        if key.isupper() and not key.startswith("__"):
            value = getattr(config, key)
            # numpy 数组和普通对象都做拷贝，避免后续原地修改污染快照。
            snapshot[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    # 返回完整配置快照，后续每个子实验开始前都可以恢复。
    return snapshot


# 函数 restore_config：把 snapshot_config() 生成的快照写回 config.py，隔离不同实验阶段。
def restore_config(snapshot: dict[str, object]) -> None:
    # 对每个配置项都重新拷贝后写回，避免恢复后再次被子实验原地修改。
    for key, value in snapshot.items():
        setattr(config, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


# 函数 is_off_policy：判断模型是否使用离策略训练接口，例如 MADDPG、MATD3 和 MASAC 系列。
def is_off_policy(model_name: str) -> bool:
    # 离策略算法依赖经验回放，训练函数需要 total_step_count 等额外状态。
    return model_name in {"maddpg", "attention_maddpg", "matd3", "attention_matd3", "masac", "attention_masac"}


# 函数 is_on_policy：判断模型是否使用在策略训练接口，目前对应 MAPPO 和 Attention-MAPPO。
def is_on_policy(model_name: str) -> bool:
    # 在策略算法按轨迹更新，不复用离策略经验池。
    return model_name in {"mappo", "attention_mappo"}


# 函数 copy_saved_model_run：把训练函数默认写入 saved_models 的权重复制到本次完整实验目录下。
def copy_saved_model_run(model_name: str, timestamp: str, run_root: Path) -> Path | None:
    source_dir = Path("saved_models") / f"{model_name}_{timestamp}"
    # 某些基线或失败运行可能不产生模型目录，此时返回 None，让测试阶段复用内存中的模型实例。
    if not source_dir.exists():
        # 返回 None 表示没有可复制的模型权重。
        return None

    destination_dir = run_root / "saved_models" / source_dir.name
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    # dirs_exist_ok=True 允许同名实验重跑时覆盖补齐目录内容，但不会删除未覆盖文件。
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    # 返回复制后的目录，后续测试阶段优先从 final 子目录加载权重。
    return destination_dir


# 函数 run_single_rl_experiment：完成单个 RL 算法的训练、训练曲线绘制、模型复制、测试和测试曲线绘制。
def run_single_rl_experiment(
    *,
    model_name: str,
    train_episodes: int,
    test_episodes: int,
    run_name: str,
) -> dict[str, object]:
    # run_root 是完整实验的根目录；timestamp 把同一算法的不同运行批次区分开。
    run_root = results_path("full_runs", run_name)
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{model_name}")

    # 训练前写入当前模型名并固定随机种子，保证同一配置下实验尽量可复现。
    config.MODEL = model_name
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # 为当前算法创建新的环境和模型，避免复用上一算法残留状态。
    env = Env()
    model: MARLModel = get_model(model_name)

    # train_logger 会保存配置和每回合指标，后续 generate_plots() 直接读取它生成曲线。
    train_log_dir = run_root / "train_logs" / model_name
    train_logger = Logger(str(train_log_dir), timestamp)
    train_logger.log_configs()

    train_start = time.time()
    # 根据算法类型选择对应训练循环：离策略、在策略和手写基线的更新方式不同。
    if is_off_policy(model_name):
        train_score = train_off_policy(env, model, train_logger, train_episodes, total_step_count=0)
    # MAPPO 系列使用在策略训练函数。
    elif is_on_policy(model_name):
        train_score = train_on_policy(env, model, train_logger, train_episodes)
    else:
        # random/static/greedy 等基线没有神经网络训练，但仍按统一接口记录表现。
        train_score = train_baselines(env, model, train_logger, train_episodes)
    train_seconds = float(time.time() - train_start)

    # 训练结束后立即生成单算法训练曲线，方便长批次运行中途检查结果质量。
    train_json_path = train_log_dir / f"log_data_{timestamp}.json"
    train_plot_dir = run_root / "train_plots" / model_name
    generate_plots(str(train_json_path), str(train_plot_dir), "train", timestamp)

    # 将训练权重复制到 full_runs 下，使总实验目录自包含，后续归档时不依赖 saved_models。
    copied_model_dir = copy_saved_model_run(model_name, timestamp, run_root)
    final_model_dir = copied_model_dir / "final" if copied_model_dir is not None else None

    test_summary: dict[str, object] | None = None
    # test_episodes 为 0 时只训练不测试；大批量调参时可以用这个开关节省时间。
    if test_episodes > 0:
        # 测试阶段重新固定种子，并重新创建环境，使测试结果不受训练环境末状态影响。
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        test_env = Env()
        test_model_instance: MARLModel = get_model(model_name)
        # 如果训练产生了 final 权重，则加载权重测试；否则使用刚训练完的内存模型或基线实例。
        if final_model_dir is not None and final_model_dir.exists():
            test_model_instance.load(str(final_model_dir))
        else:
            test_model_instance = model

        # 测试日志和训练日志分开存放，避免绘图时混用 train/test 指标。
        test_log_dir = run_root / "test_logs" / model_name
        test_logger = Logger(str(test_log_dir), timestamp)
        test_start = time.time()
        test_model(test_env, test_model_instance, test_logger, test_episodes)
        test_seconds = float(time.time() - test_start)

        test_json_path = test_log_dir / f"log_data_{timestamp}.json"
        test_plot_dir = run_root / "test_plots" / model_name
        generate_plots(str(test_json_path), str(test_plot_dir), "test", timestamp, smoothing_window=2)

        test_summary = {
            "episodes": int(test_episodes),
            "seconds": test_seconds,
            "log_dir": str(test_log_dir),
            "plot_dir": str(test_plot_dir),
            "log_json_path": str(test_json_path),
        }

    # 返回单算法摘要，run_rl_suite() 会把所有算法摘要合并并写入总 manifest。
    return {
        "model": model_name,
        "timestamp": timestamp,
        "train": {
            "episodes": int(train_episodes),
            "score": float(train_score),
            "seconds": train_seconds,
            "log_dir": str(train_log_dir),
            "plot_dir": str(train_plot_dir),
            "log_json_path": str(train_json_path),
        },
        "saved_model_dir": str(copied_model_dir) if copied_model_dir is not None else None,
        "saved_model_final_dir": str(final_model_dir) if final_model_dir is not None else None,
        "test": test_summary,
    }


# 函数 run_rl_suite：批量运行多个 RL 算法，并在所有算法完成后生成训练曲线对比图。
def run_rl_suite(
    *,
    models: list[str],
    train_episodes: int,
    test_episodes: int,
    run_name: str,
    smoothing_window: int,
) -> dict[str, object]:
    run_root = results_path("full_runs", run_name)
    rl_results: list[dict[str, object]] = []

    # 按用户传入的 models 顺序逐个运行，保证日志和最终摘要中的算法顺序一致。
    for model_name in models:
        print(f"[rl] training/testing model={model_name}")
        rl_results.append(
            run_single_rl_experiment(
                model_name=model_name,
                train_episodes=train_episodes,
                test_episodes=test_episodes,
                run_name=run_name,
            )
        )

    # 收集每个算法训练日志目录，用统一绘图函数生成跨算法对比曲线。
    train_log_dirs = [entry["train"]["log_dir"] for entry in rl_results]
    comparison_dir = run_root / "comparisons" / "training"
    compare_algorithms(train_log_dirs, models, str(comparison_dir), smoothing_window=smoothing_window)

    # 返回 RL 套件摘要，包含算法列表、训练/测试回合数、对比图路径和每个算法的详细结果。
    return {
        "models": models,
        "train_episodes": int(train_episodes),
        "test_episodes": int(test_episodes),
        "comparison_dir": str(comparison_dir),
        "runs": rl_results,
    }


# 函数 run_offload_suite：调度卸载分类器完整流水线，并按需补充离线有效性和高级论文验证。
def run_offload_suite(
    *,
    run_name: str,
    per_class_target: int,
    procedural_train_per_class: int,
    max_attempts: int,
    dataset_seed: int,
    train_seed: int,
    runtime_seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    hidden_dims: tuple[int, ...],
    device: str,
    sampler_mode: str,
    episodes_per_seed: int,
    steps_per_episode: int,
    validity_dataset: str | None,
    validity_epochs: int,
    run_advanced_validity: bool,
    label_mode: str,
) -> dict[str, object]:
    print("[offload] running full pipeline")
    # 先运行完整卸载流水线，生成数据集、两个分类器权重和运行时策略对比报告。
    full_pipeline_summary = run_full_offload_experiment(
        experiment_name=f"{run_name}_offload",
        per_class_target=per_class_target,
        procedural_train_per_class=procedural_train_per_class,
        max_attempts=max_attempts,
        dataset_seed=dataset_seed,
        train_seed=train_seed,
        runtime_seeds=runtime_seeds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        hidden_dims=hidden_dims,
        device=device,
        sampler_mode=sampler_mode,
        episodes_per_seed=episodes_per_seed,
        steps_per_episode=steps_per_episode,
        label_mode=label_mode,
    )

    validity_report_path: str | None = None
    validity_report: dict[str, object] | None = None
    # validity_dataset 不为空时，额外训练/评估一个离线分类器，用于输出论文有效性相关统计。
    if validity_dataset is not None:
        print(f"[offload] running paper-validity evaluation on {validity_dataset}")
        validity_report_path = str(results_path("full_runs", run_name, "reports", "offload_policy_validity_report.json"))
        validity_report = run_validity_evaluation(
            dataset_path=validity_dataset,
            output_path=validity_report_path,
            seed=train_seed,
            epochs=validity_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            sampler_mode=sampler_mode,
        )

    advanced_validity_outputs: dict[str, str] | None = None
    # 高级验证脚本会生成独立 JSON 与 Markdown 报告，适合直接放入论文实验材料核对。
    if run_advanced_validity:
        print("[offload] running advanced repository validator")
        advanced_offload_validity.main()
        advanced_validity_outputs = {
            "json_report": str(results_path("reports", "offload_policy_paper_validity.json")),
            "markdown_summary": str(results_path("reports", "offload_policy_paper_validity_summary.md")),
        }

    # 返回卸载套件摘要，run_all_experiments 的 main() 会把它合并到总 manifest。
    return {
        "full_pipeline": full_pipeline_summary,
        "validity_dataset": validity_dataset,
        "validity_report_path": validity_report_path,
        "validity_report": validity_report,
        "advanced_validity_outputs": advanced_validity_outputs,
    }


# 函数 parse_hidden_dims：把命令行中的 "64,64" 转成 (64, 64)，供卸载分类器 MLP 使用。
def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    # 去掉空白项，允许用户输入 "128, 64" 这类更易读的形式。
    dims = tuple(int(token.strip()) for token in hidden_dims_arg.split(",") if token.strip())
    # 空字符串或只有逗号都不能构成有效网络结构，需要尽早报错。
    if not dims:
        # 主动报错，避免后续构造 MLP 时才出现更难定位的维度错误。
        raise ValueError("At least one hidden dimension must be provided.")
    # 返回整数元组，后续会直接传入 train_offload_policy()。
    return dims


# 函数 parse_args：定义完整实验的命令行接口，覆盖 RL、卸载分类器和有效性验证三类参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repository's full RL and offloading experiment suite.")
    # 基础运行参数：决定输出目录和是否运行 RL/卸载/高级验证阶段。
    parser.add_argument("--name", type=str, default=time.strftime("all_exp_%Y%m%d_%H%M%S"), help="Experiment run name.")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), help="RL models to include in the RL suite.")
    parser.add_argument("--train_episodes", type=int, default=100, help="Training episodes per RL model.")
    parser.add_argument("--test_episodes", type=int, default=20, help="Testing episodes per RL model.")
    parser.add_argument("--comparison_smoothing", type=int, default=5, help="Smoothing window for RL comparison plots.")
    parser.add_argument("--skip_rl", action="store_true", help="Skip the RL algorithm suite.")
    parser.add_argument("--skip_offload", action="store_true", help="Skip the offloading suite.")
    parser.add_argument("--skip_advanced_validity", action="store_true", help="Skip verify_offload_policy_validity.py.")
    # 数据集采集参数：控制任务卸载监督学习数据的规模、随机性和采样上限。
    parser.add_argument("--per_class_target", type=int, default=2000, help="Template samples per class for offload dataset collection.")
    parser.add_argument("--procedural_train_per_class", type=int, default=1800, help="Procedural rich samples per class.")
    parser.add_argument("--max_attempts", type=int, default=60000, help="Maximum attempts during offload dataset collection.")
    parser.add_argument("--dataset_seed", type=int, default=42, help="Dataset collection seed.")
    parser.add_argument("--train_seed", type=int, default=42, help="Training seed for offload classifier stages.")
    parser.add_argument("--runtime_seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="Runtime comparison seeds.")
    # 分类器训练参数：控制 MLP 训练轮数、批量大小、学习率、验证集比例和采样方式。
    parser.add_argument("--offload_epochs", type=int, default=25, help="Training epochs for offload classifiers.")
    parser.add_argument("--validity_epochs", type=int, default=20, help="Epochs for evaluate_offload_validity.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for offload classifiers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for offload classifiers.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio for offload classifier training.")
    parser.add_argument("--hidden_dims", type=str, default="64,64", help="Comma-separated hidden dims for offload classifiers.")
    parser.add_argument("--device", type=str, default="auto", help="Offload classifier device: auto, cpu, cuda, or mps.")
    parser.add_argument("--sampler_mode", type=str, default="auto", choices=["auto", "weighted", "none"], help="Offload classifier sampler mode.")
    # 运行时对比参数：控制每组随机种子下仿真的回合数和每回合步数。
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="Episodes per seed for runtime offload comparison.")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Steps per episode for runtime offload comparison.")
    parser.add_argument("--offload_label_mode", type=str, default=config.OFFLOAD_LABEL_MODE, choices=["heuristic", "enhanced_oracle"], help="Label generator for supervised offload training.")
    parser.add_argument(
        "--validity_dataset",
        type=str,
        default="offload_datasets/offload_dataset_balanced.npz",
        help="Dataset for evaluate_offload_validity. Set empty string to skip.",
    )
    # 返回解析后的参数对象，main() 会继续做路径空值处理和 hidden_dims 类型转换。
    return parser.parse_args()


# 函数 main：完整实验总入口，负责按开关运行 RL 套件和卸载套件，并写出总 manifest。
def main() -> None:
    args = parse_args()
    configure_fp32_precision()
    configure_compile_backend()

    # 保存初始 config，后续每个子套件运行前恢复，避免 MODEL 等全局变量跨阶段串扰。
    base_snapshot = snapshot_config()
    run_root = results_path("full_runs", args.name)
    run_root.mkdir(parents=True, exist_ok=True)

    # summary 是总实验的索引结构；即使用户跳过某个阶段，对应字段也保留为 None。
    summary: dict[str, object] = {
        "run_name": args.name,
        "run_root": str(run_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "rl_suite": None,
        "offload_suite": None,
    }

    # 使用 try/finally 保护配置恢复：即使某个算法训练失败，也尽量把 config 恢复到进入脚本前的状态。
    try:
        # 未指定 --skip_rl 时，先运行强化学习算法套件。
        if not args.skip_rl:
            restore_config(base_snapshot)
            summary["rl_suite"] = run_rl_suite(
                models=args.models,
                train_episodes=args.train_episodes,
                test_episodes=args.test_episodes,
                run_name=args.name,
                smoothing_window=args.comparison_smoothing,
            )

        # 未指定 --skip_offload 时，再运行任务卸载分类器套件。
        if not args.skip_offload:
            restore_config(base_snapshot)
            # 命令行传入空字符串表示跳过 evaluate_offload_validity.py 的离线有效性评估。
            validity_dataset = args.validity_dataset if args.validity_dataset.strip() else None
            summary["offload_suite"] = run_offload_suite(
                run_name=args.name,
                per_class_target=args.per_class_target,
                procedural_train_per_class=args.procedural_train_per_class,
                max_attempts=args.max_attempts,
                dataset_seed=args.dataset_seed,
                train_seed=args.train_seed,
                runtime_seeds=[int(seed) for seed in args.runtime_seeds],
                epochs=args.offload_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                val_ratio=args.val_ratio,
                hidden_dims=parse_hidden_dims(args.hidden_dims),
                device=args.device,
                sampler_mode=args.sampler_mode,
                episodes_per_seed=args.episodes_per_seed,
                steps_per_episode=args.steps_per_episode,
                validity_dataset=validity_dataset,
                validity_epochs=args.validity_epochs,
                run_advanced_validity=not args.skip_advanced_validity,
                label_mode=args.offload_label_mode,
            )
    finally:
        # 恢复进入脚本时的配置，避免这个脚本作为库函数被调用时污染外部进程。
        restore_config(base_snapshot)

    # 记录完成时间并写出总 manifest；ensure_ascii=False 保留中文路径和中文注释可读性。
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path = run_root / "experiment_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
