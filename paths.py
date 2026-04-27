"""
中文注释说明：paths.py

文件作用：
    统一管理项目根目录、结果目录和实验输出路径，避免各个脚本手写路径导致输出位置不一致。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - REPO_ROOT: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - RESULTS_ROOT: 全局常量或配置项，会影响环境规模、训练过程或实验输出。
    - results_path(): 关键函数，承载本模块的一段可复用实验逻辑。

主要依赖：
    pathlib

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

from __future__ import annotations

from pathlib import Path


# 关键变量 REPO_ROOT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
REPO_ROOT = Path(__file__).resolve().parent
# 关键变量 RESULTS_ROOT：全局常量或配置项，会影响环境规模、训练过程或实验输出。
RESULTS_ROOT = REPO_ROOT / "results"


# 函数 results_path：关键函数，承载本模块的一段可复用实验逻辑。
def results_path(*parts: str) -> Path:
    # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
    return RESULTS_ROOT.joinpath(*parts)
