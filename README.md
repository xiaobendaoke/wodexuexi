# 基于 Attention-MAPPO 的多无人机 MEC 轨迹与任务卸载协同优化方法

Attention-MAPPO Based Collaborative Trajectory and Task Offloading Optimization for Multi-UAV Mobile Edge Computing

本仓库对应一篇关于多无人机辅助移动边缘计算的论文实验工程。研究目标是在动态用户请求、无线链路变化、UAV 能耗、UE 电池状态和宏基站负载约束共同存在的场景下，联合优化 UAV 轨迹控制与请求级任务卸载策略。

本文方法采用双层多智能体强化学习框架：

- 上层使用 Attention-MAPPO 控制多架 UAV 的二维轨迹动作，建模 UAV 邻居关系和覆盖 UE 状态。
- 下层使用 constrained attention offload MAPPO 处理每个 UAV 当前覆盖的服务请求，在 local UAV、cooperative UAV 和 MBS 三类目标之间做离散卸载决策。
- 请求级动作使用 quality-aware mask 屏蔽明显不可行的协作或 MBS 动作，减少无效探索。
- 下层奖励引入 Lagrange 约束，刻画 deadline satisfaction rate 与 MBS load ratio 之间的权衡。
- 系统模型显式统计 UAV 飞行能耗、计算能耗、通信能耗、WPT 能耗以及 UE 电池动态。

完整论文正文以 `latex/docs/*.tex` 为准；本 README 作为项目主页、最新结果索引和复现实验导航。

## 最新实验结论

最新可用 baseline 对比数据保存在：

```text
results/baseline_comparison/summary_energy_efficiency.tsv
```

该轮实验使用：

```text
seed = 42
workload_seed = 42
eval_episodes = 10
train_episodes = 200  # learning baselines
```

新增有效能效指标定义为：

```text
Effective Energy Efficiency = deadline_satisfied_service_requests / total_energy
```

README 中展示时换算为 `tasks/MJ`，即每消耗 1 MJ UAV 能量，完成多少个满足 deadline 的服务请求。

| Method | Reward | Latency | Energy | EE tasks/MJ | Fairness | DSR | MBS Load | OffL | OffC | OffM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | -1445.46 | 1341.77K | 68.13M | 71.31 | 0.903 | 0.160 | 0.108 | 0.311 | 0.376 | 0.308 |
| uniform | -5731.45 | 1280.09K | 87.94M | 77.23 | 0.774 | **0.217** | 0.067 | 0.263 | 0.555 | 0.169 |
| ippo | -1437.41 | **776.50K** | 55.66M | 109.15 | **0.920** | 0.199 | 0.030 | 0.436 | 0.517 | 0.047 |
| vanilla_mappo | **-1005.43** | 935.13K | 50.10M | 119.26 | 0.918 | 0.193 | **0.009** | 0.743 | 0.238 | 0.019 |
| joint_mappo | -9419.66 | 1420.18K | 54.22M | 96.59 | 0.529 | 0.175 | 0.054 | 0.389 | 0.375 | 0.191 |
| proposed | -2832.92 | 1104.01K | **45.44M** | **134.63** | 0.870 | 0.201 | 0.094 | 0.343 | 0.464 | 0.192 |

可以直接用于论文表述的结论：

- `proposed` 在 UAV 总能耗上最低，`Energy = 45.44M`。
- `proposed` 在有效能效上最高，`EE = 134.63 tasks/MJ`。
- `ippo` 在平均时延上最低，`Latency = 776.50K`。
- `uniform` 在 DSR 上最高，`DSR = 0.217`。
- `vanilla_mappo` 的 MBS load 最低，`MBS_Load = 0.009`。

因此，本轮 baseline 结果不支持“本文方法在所有指标上全面最优”的说法。更准确的论文主线是：本文方法并非单纯追求最低时延，而是在保持可接受 DSR 的同时显著降低 UAV 侧能耗，从而取得最高有效能效。

## 方法与 Baseline

本仓库当前 baseline 对比包含 6 种方法：

- `proposed`：本文方法，upper attention-MAPPO + lower constrained attention offload MAPPO。
- `ippo`：独立 PPO/IPPO 上层轨迹策略 + 同一套学习式下层卸载策略，用于对比独立智能体轨迹学习。
- `vanilla_mappo`：无 attention 的 MAPPO 上层轨迹策略 + 同一套学习式下层卸载策略，用于对比 attention 编码作用。
- `joint_mappo`：端到端联合动作 MAPPO，将轨迹和卸载放入同一联合动作空间，用于验证直接联合学习的训练难度。
- `random`：非学习随机轨迹 + mask-aware 随机卸载。
- `uniform`：非学习中心移动/负载均衡式启发策略。

## 数据路径索引

### 最新可用 Baseline 数据

| 用途 | 路径 |
|---|---|
| 最新结果汇总表 | `results/baseline_comparison/summary_energy_efficiency.tsv` |
| 最新结果说明 | `results/baseline_comparison/README_latest_energy_efficiency.txt` |
| 单方法指标 JSON | `results/baseline_comparison/<method>/metrics_seed_42_workload_42.json` |
| 单方法轨迹 JSON | `results/baseline_comparison/<method>/trajectory_seed_42_workload_42.json` |
| 单方法配置快照 | `results/baseline_comparison/<method>/config_seed_42.json` |
| 学习方法训练曲线 | `results/baseline_comparison/<method>/training_curve_seed_42.json` |
| 学习方法训练摘要 | `results/baseline_comparison/<method>/training_summary_seed_42.json` |
| 运行日志 | `logs/baseline_<method>_energy_efficiency.log` |
| 并行运行主日志 | `logs/baseline_energy_efficiency_parallel.log` |

其中 `<method>` 可取：

```text
random
uniform
ippo
vanilla_mappo
joint_mappo
proposed
```

注意：`random`、`uniform` 是非学习 baseline，没有 `training_curve_seed_42.json` 和 `training_summary_seed_42.json`。`joint_mappo` 当前有 `training_summary_seed_42.json`，没有单独的 `training_curve_seed_42.json`。

### 旧数据归档

旧 baseline 数据已归档到：

```text
recent/archive_baseline_before_energy_efficiency_20260604/
```

该目录中的数据包括旧 `workload_1001` 结果和旧 `baseline_comparison_full` 结果。这些结果生成时还没有最新的 `energy_efficiency` 指标，不作为本 README 最新主结论依据。

归档说明文件：

```text
recent/archive_baseline_before_energy_efficiency_20260604/README.txt
```

### 论文与方案文档

| 用途 | 路径 |
|---|---|
| LaTeX 主文件 | `latex/main.tex` |
| 第 1 章 | `latex/docs/chap01.tex` |
| 第 2 章 | `latex/docs/chap02.tex` |
| 第 3 章 | `latex/docs/chap03.tex` |
| 第 4 章 | `latex/docs/chap04.tex` |
| 第 5 章 | `latex/docs/chap05.tex` |
| 第 6 章 | `latex/docs/chap06.tex` |
| 参考文献 | `latex/reference.bib` |
| baseline 对比方案 | `docs/baseline_comparison_plan.md` |
| 画图方案 | `docs/figure_generation_plan.md` |

## 复现实验命令

建议使用项目虚拟环境：

```bash
.venv/bin/python --version
```

单个 baseline 示例：

```bash
.venv/bin/python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 200 --eval_episodes 10 --seed 42 --workload_seed 42
```

完整复现最新 6 个 baseline：

```bash
.venv/bin/python run_baseline_comparison_experiment.py --baseline random --eval_episodes 10 --seed 42 --workload_seed 42
.venv/bin/python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 10 --seed 42 --workload_seed 42
.venv/bin/python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 200 --eval_episodes 10 --seed 42 --workload_seed 42
.venv/bin/python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 200 --eval_episodes 10 --seed 42 --workload_seed 42
.venv/bin/python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 200 --eval_episodes 10 --seed 42 --workload_seed 42
.venv/bin/python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 200 --eval_episodes 10 --seed 42 --workload_seed 42
```

如果希望多进程并行跑，可参考之前使用的日志路径：

```text
logs/baseline_<method>_energy_efficiency.log
```

并行运行时建议为每个进程单独写日志，避免输出混在一起。

## 代码入口

| 模块 | 路径 | 说明 |
|---|---|---|
| baseline 实验入口 | `run_baseline_comparison_experiment.py` | 统一训练/评估 random、uniform、ippo、vanilla_mappo、joint_mappo、proposed |
| 指标统计 | `utils/baseline_metrics.py` | 汇总 reward、latency、energy、energy_efficiency、DSR、offloading ratios 等 |
| 环境 step 与 metrics | `environment/env.py` | 生成环境状态、执行动作、返回系统指标 |
| UAV 请求处理 | `environment/uavs.py` | 处理 local/cooperative/MBS 卸载、统计服务请求和能耗 |
| 模型注册 | `marl_models/utils.py` | 注册各类 MARL 模型和 baseline 模型 |
| random baseline | `marl_models/random_baseline/random_baseline.py` | 非学习随机策略 |
| uniform baseline | `marl_models/uniform_baseline/uniform_baseline.py` | 非学习负载均衡启发策略 |
| IPPO baseline | `marl_models/ippo_baseline/ippo_baseline.py` | 独立 PPO 轨迹 baseline |

## 图表与论文写作建议

最新 baseline 结果适合生成以下图表：

- 有效能效柱状图：使用 `EE_tasks_per_MJ`，突出 `proposed` 最高。
- UAV 总能耗柱状图：使用 `Energy`，突出 `proposed` 最低。
- 时延对比图：说明 `ippo` 时延最低，`proposed` 不是 latency-optimal。
- DSR 对比图：说明 `uniform` DSR 最高，`proposed` 保持中上水平。
- 卸载比例堆叠图：使用 `OffL / OffC / OffM`，展示不同方法的卸载结构。

正式论文表述建议：

```text
本文方法在 UAV 侧总能耗和有效能效方面取得最优表现，说明双层协同策略能够以更低能量代价完成满足 deadline 的服务请求。与此同时，IPPO 在平均时延上更具优势，Uniform 在 DSR 上表现较高，Vanilla MAPPO 对 MBS 负载控制更强。这表明本文方法的优势主要体现在能耗-服务质量综合权衡，而非单一指标全面最优。
```

## 当前 README 使用的数据口径

本 README 只采用最新能效 baseline 数据作为首页主结论：

```text
results/baseline_comparison/summary_energy_efficiency.tsv
```

更早的 `results/final/`、`results/full_cpu_runs/`、`results/joint_experiments/` 等目录可能仍包含历史主实验、消融实验或调参实验结果，但它们不作为本 README 的最新 baseline 结论来源。需要回溯旧实验时，优先查看对应目录中的 `statistics.json`、`statistics.md`、`manifest.json` 或控制日志。
