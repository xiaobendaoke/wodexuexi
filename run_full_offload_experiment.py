"""
中文注释说明：run_full_offload_experiment.py

文件作用：
    组织完整的任务卸载策略实验流水线，专门用于验证“请求级任务卸载分类器”是否能替代或增强
    原始启发式卸载策略。它会在同一次实验目录下完成数据集采集、数据分布画像、两类分类器训练、
    运行时策略对比和最终 JSON 汇总，适合作为论文中卸载策略有效性实验的主入口。

整体流程：
    1. 创建 results/full_offload_experiments/{experiment_name} 目录，并划分 datasets、checkpoints、reports。
    2. 调用 collect_offload_dataset() 采集 rich_candidate_mixed 数据集，保证每个卸载类别有足够样本。
    3. 调用 profile_offload_dataset() 统计数据集类别分布和特征范围，写出 profile JSON。
    4. 分别训练 full_features 代理基线分类器和 rich_reduced_features 运行时分类器。
    5. 调用 compare_runtime_offload_policies() 对比启发式策略、代理基线和精简运行时策略。
    6. 将数据集信息、训练摘要和运行时对比结果合并写入 experiment_summary.json。

关键变量与对象：
    - experiment_name: 本次卸载实验的目录名，用于隔离不同批次的结果。
    - per_class_target: 模板场景中每个卸载类别希望采集到的目标样本数。
    - procedural_train_per_class: 程序化富样本采集中每个类别的训练样本目标数。
    - max_attempts: 采样过程的最大尝试次数，防止类别不平衡时无限循环。
    - dataset_seed / train_seed / runtime_seeds: 分别控制数据采集、分类器训练和运行时仿真的随机性。
    - hidden_dims: 卸载分类器 MLP 隐藏层结构，例如 (64, 64)。
    - label_mode: 监督标签生成方式，heuristic 表示启发式标签，enhanced_oracle 表示增强 oracle 标签。
    - dataset_path: 采集后保存的 .npz 数据集路径。
    - surrogate_checkpoint: full_features 代理基线模型的权重文件。
    - rich_checkpoint: rich_reduced_features 运行时模型的权重文件。
    - comparison_report_path: 不同卸载策略运行时对比报告的 JSON 输出路径。
    - run_full_offload_experiment(): 核心流水线函数，返回可继续汇总到总实验 manifest 的字典。

主要依赖：
    argparse, json, time, pathlib, collect_offload_dataset, compare_runtime_offload_policies, marl_models, paths, profile_offload_dataset, train_offload_policy

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from collect_offload_dataset import collect_offload_dataset
from compare_runtime_offload_policies import compare_runtime_offload_policies
from marl_models.offload_policy import OFFLOAD_FEATURE_FAMILY_FULL, OFFLOAD_FEATURE_FAMILY_RICH_REDUCED
from paths import results_path
from profile_offload_dataset import profile_offload_dataset
from train_offload_policy import parse_hidden_dims, train_offload_policy


# 函数 _default_experiment_name：按当前时间生成默认实验名，避免多次运行覆盖同一目录。
def _default_experiment_name() -> str:
    # 返回形如 offload_full_20260427_153000 的目录名，便于按时间追踪实验批次。
    return time.strftime("offload_full_%Y%m%d_%H%M%S")


# 函数 run_full_offload_experiment：串联“采集数据-训练分类器-运行时对比-写报告”的完整卸载实验。
def run_full_offload_experiment(
    *,
    experiment_name: str,
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
    label_mode: str,
) -> dict[str, object]:
    """运行当前仓库的端到端请求级任务卸载实验流水线。

    本函数只调用当前项目已有模块，不额外引入外部实验框架：
    1. 采集运行时候选卸载数据集；
    2. 统计数据集画像；
    3. 训练完整特征代理基线分类器；
    4. 训练精简特征运行时卸载策略；
    5. 比较启发式策略、代理基线策略和精简运行时策略。
    """

    # output_root 是本次卸载实验的根目录，所有中间文件和最终报告都放在它下面。
    output_root = results_path("full_offload_experiments", experiment_name)
    dataset_dir = output_root / "datasets"
    checkpoint_dir = output_root / "checkpoints"
    report_dir = output_root / "reports"

    # 提前创建目录，保证后续采集数据、保存模型和写报告时不会因为父目录缺失而失败。
    dataset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 这些路径构成本实验的主要产物：数据集、数据画像、两个模型权重、运行时对比报告和总摘要。
    dataset_path = dataset_dir / "offload_dataset_runtime_candidate.npz"
    dataset_profile_path = dataset_dir / "offload_dataset_runtime_candidate.profile.json"
    surrogate_checkpoint = checkpoint_dir / "offload_policy_surrogate_runtime.pt"
    rich_checkpoint = checkpoint_dir / "offload_policy_rich_runtime.pt"
    comparison_report_path = report_dir / "runtime_offload_policy_comparison.json"
    experiment_summary_path = output_root / "experiment_summary.json"

    print(f"[full-exp] collecting dataset -> {dataset_path}")
    # 采集任务卸载监督学习数据；rich_candidate_mixed 会混合模板场景与程序化场景，提高样本覆盖面。
    dataset_metadata = collect_offload_dataset(
        output_path=dataset_path,
        mode="rich_candidate_mixed",
        per_class_target=per_class_target,
        max_attempts=max_attempts,
        procedural_train_per_class=procedural_train_per_class,
        seed=dataset_seed,
        label_mode=label_mode,
    )

    print(f"[full-exp] profiling dataset -> {dataset_profile_path}")
    # 数据画像用于检查类别是否均衡、特征是否异常，后续也会写入总摘要方便论文复现实验核对。
    dataset_profile = profile_offload_dataset(dataset_path)
    dataset_profile_path.write_text(json.dumps(dataset_profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[full-exp] training surrogate baseline -> {surrogate_checkpoint}")
    # full_features 使用完整特征，作为监督学习卸载策略的上限或代理基线。
    surrogate_summary = train_offload_policy(
        dataset_path=dataset_path,
        output_path=surrogate_checkpoint,
        feature_family=OFFLOAD_FEATURE_FAMILY_FULL,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        seed=train_seed,
        device=device,
        sampler_mode=sampler_mode,
    )

    print(f"[full-exp] training rich reduced runtime policy -> {rich_checkpoint}")
    # rich_reduced_features 使用更精简的运行时特征，更贴近真实在线决策时可获得的信息。
    rich_summary = train_offload_policy(
        dataset_path=dataset_path,
        output_path=rich_checkpoint,
        feature_family=OFFLOAD_FEATURE_FAMILY_RICH_REDUCED,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        seed=train_seed,
        device=device,
        sampler_mode=sampler_mode,
    )

    print(f"[full-exp] comparing runtime policies -> {comparison_report_path}")
    # 在仿真环境中用多组随机种子对比策略，避免只看离线分类精度而忽略运行时收益。
    comparison_report = compare_runtime_offload_policies(
        surrogate_checkpoint=surrogate_checkpoint,
        rich_reduced_checkpoint=rich_checkpoint,
        seeds=runtime_seeds,
        episodes_per_seed=episodes_per_seed,
        steps_per_episode=steps_per_episode,
        output_path=comparison_report_path,
    )

    # summary 是本脚本给上层 run_all_experiments.py 返回的统一接口，也会落盘成为本实验的索引文件。
    summary: dict[str, object] = {
        "experiment_name": experiment_name,
        "label_mode": label_mode,
        "dataset": {
            "path": str(dataset_path),
            "metadata": dataset_metadata,
            "profile_path": str(dataset_profile_path),
            "profile": dataset_profile,
        },
        "surrogate_training": surrogate_summary,
        "rich_reduced_training": rich_summary,
        "runtime_comparison_path": str(comparison_report_path),
        "runtime_comparison": comparison_report,
    }
    experiment_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # 返回完整实验摘要，方便总实验入口继续合并 RL 实验、有效性验证等其他结果。
    return summary


# 函数 parse_args：解析命令行参数，让命令行运行时可以覆盖采样规模、训练轮数和运行时对比设置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行当前仓库完整的请求级任务卸载实验流水线。")
    parser.add_argument("--name", type=str, default=_default_experiment_name(), help="实验输出目录名称。")
    parser.add_argument("--per_class_target", type=int, default=2000, help="模板场景中每个类别的目标样本数。")
    parser.add_argument("--procedural_train_per_class", type=int, default=1800, help="程序化富场景中每个类别的训练样本数。")
    parser.add_argument("--max_attempts", type=int, default=60000, help="模板场景采样的最大尝试次数。")
    parser.add_argument("--dataset_seed", type=int, default=42, help="数据集采集随机种子。")
    parser.add_argument("--train_seed", type=int, default=42, help="分类器训练随机种子。")
    parser.add_argument("--runtime_seeds", type=int, nargs="+", default=[42, 84, 126, 168], help="运行时对比实验使用的随机种子列表。")
    parser.add_argument("--epochs", type=int, default=25, help="分类器训练轮数。")
    parser.add_argument("--batch_size", type=int, default=256, help="分类器训练批量大小。")
    parser.add_argument("--lr", type=float, default=1e-3, help="分类器学习率。")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集划分比例。")
    parser.add_argument("--hidden_dims", type=str, default="64,64", help="分类器隐藏层维度，多个维度用逗号分隔。")
    parser.add_argument("--device", type=str, default="auto", help="训练设备，可选 auto、cpu、cuda 或 mps。")
    parser.add_argument("--sampler_mode", type=str, default="auto", choices=["auto", "weighted", "none"], help="分类器训练采样方式。")
    parser.add_argument("--episodes_per_seed", type=int, default=4, help="运行时对比中每个随机种子的回合数。")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="运行时对比中每个回合的步数。")
    parser.add_argument("--label_mode", type=str, default="enhanced_oracle", choices=["heuristic", "enhanced_oracle"], help="监督卸载训练使用的标签生成方式。")
    # 返回 argparse.Namespace，main() 会把字符串形式的隐藏层等参数进一步转换成内部需要的类型。
    return parser.parse_args()


# 函数 main：命令行入口，负责把参数解析结果映射到 run_full_offload_experiment()。
def main() -> None:
    args = parse_args()
    # parse_hidden_dims() 将 "64,64" 这类命令行字符串转换为 (64, 64)，供 MLP 构造函数使用。
    summary = run_full_offload_experiment(
        experiment_name=args.name,
        per_class_target=args.per_class_target,
        procedural_train_per_class=args.procedural_train_per_class,
        max_attempts=args.max_attempts,
        dataset_seed=args.dataset_seed,
        train_seed=args.train_seed,
        runtime_seeds=[int(seed) for seed in args.runtime_seeds],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        device=args.device,
        sampler_mode=args.sampler_mode,
        episodes_per_seed=args.episodes_per_seed,
        steps_per_episode=args.steps_per_episode,
        label_mode=args.label_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
