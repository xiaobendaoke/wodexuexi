#!/usr/bin/env python3
"""
中文注释说明：compare_algorithms.py

文件作用：
    比较不同强化学习算法或基线方法的实验日志，并生成汇总指标和对比图。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - main(): 脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。

主要依赖：
    argparse, sys, pathlib, utils

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""


import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.comparative_plots import compare_algorithms


# 函数 main：脚本主流程入口，串联参数解析、对象创建、训练评估和结果输出。
def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple RL algorithm runs and generate comparative plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="List of log directories containing log_data_*.json files",
    )

    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="List of algorithm names (must match --logs length)",
    )

    parser.add_argument(
        "--output",
        default="comparative_plots",
        help="Output directory for plots (default: comparative_plots)",
    )

    parser.add_argument(
        "--smoothing",
        type=int,
        default=5,
        help="Smoothing window size for moving average (default: 5)",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.logs) != len(args.names):
        print("ERROR: Number of --logs must match number of --names")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Comparative Analysis Tool")
    print("=" * 60)
    print(f"Algorithms: {', '.join(args.names)}")
    print(f"Output directory: {args.output}")
    print(f"Smoothing window: {args.smoothing}")
    print("=" * 60 + "\n")

    compare_algorithms(args.logs, args.names, args.output, smoothing_window=args.smoothing)


# 脚本入口：直接运行本文件时，从 main() 开始执行完整流程。
if __name__ == "__main__":
    main()
