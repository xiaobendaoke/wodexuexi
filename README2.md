# 面向多无人机移动边缘计算的双层协同优化：轨迹控制与 Oracle-Guided 任务卸载

## 当前状态

截至 2026-04-29，本仓库已经完成并重新生成了论文候选图表，图表统一位于 `docs/figures/`。下层卸载实验、统计分析和分类器质量分析已经有可复现结果；上层已有 `attention_mappo` 与 `uncoordinated_greedy` 的独立测试结果；端到端四组联合消融的调度脚本已经通过 smoke 检查，但正式 `10 seeds × 6 episodes` 的 `joint_four_way_journal` 结果尚未跑完。

因此，当前 README2 的结论口径是：

1. 可以稳妥使用 **上层独立实验** 说明 attention-MAPPO 改善轨迹控制效果。
2. 可以稳妥使用 **下层独立实验** 说明 oracle-guided 卸载显著压缩 MBS fallback。
3. 不能把 smoke 结果写成正式端到端收益；四组联合消融仍是投稿前最优先补齐的主实验。
4. 超参数敏感性图当前仍是模板；只有运行 MBS penalty 扫描后，才可作为附录实验结论。

## 摘要

多无人机辅助移动边缘计算（UAV-assisted MEC）能够在灾害救援、热点区域保障、临时通信和复杂空地网络中提供灵活的接入与计算服务。但在多 UAV 系统中，轨迹控制、覆盖关系、协作链路、任务卸载、缓存状态、计算队列和 MBS 负载强耦合。端到端联合优化容易导致状态和动作空间膨胀；只优化轨迹或只优化卸载，又难以解释二者之间的动态反馈。

本文采用双层协同优化框架。上层使用 attention-MAPPO 学习多 UAV 轨迹与协同控制，使 UAV 能够根据用户热点、邻居状态和服务压力调整运动；下层使用 oracle-guided request-level policy learning，在每个请求到达时从 local UAV、cooperative UAV 和 MBS fallback 中选择执行位置。下层标签来自增强 oracle，代价函数考虑候选时延、deadline、队列压力、MBS 负载和协作收益；在线推理只使用当前状态特征，不使用未来请求或未来轨迹信息。

本文的核心定位不是再次声称“首次联合优化轨迹、卸载、缓存和资源”，而是强调 **学习式双层分解、可解释关系建模、在线决策复杂度下降，以及 MBS fallback 显式压缩**。

关键词：多无人机；移动边缘计算；任务卸载；多智能体强化学习；attention-MAPPO；oracle-guided policy learning

## 核心贡献

1. 构建多 UAV MEC 双层协同优化框架，将上层轨迹控制与下层请求级卸载拆分建模，并通过链路、缓存、队列和 MBS 负载刻画耦合关系。
2. 设计 attention-MAPPO 轨迹控制策略，使 UAV 在多热点、多邻居和负载压力下进行协同移动，而不是只做局部贪心。
3. 设计 oracle-guided 卸载策略，用增强 oracle 生成请求级标签，并训练轻量分类器实现在线快速推理。
4. 在增强 oracle 中引入 MBS 负载惩罚、队列压力和协作收益，使策略能够在服务质量基本稳定时减少 MBS 依赖。
5. 补充图表和统计分析，包括 Pareto-style 权衡、配对 seed 改进图、请求去向结构、分类器质量、置信区间和显著性检验。

## 方法概览

### 上层：Attention-MAPPO 轨迹控制

上层把每架 UAV 视为一个智能体。观测包括自身位置、邻居状态、关联 UE 信息、缓存/服务压力、能量状态和局部负载。动作是 UAV 的移动方向和距离。奖励综合考虑服务时延、能耗、deadline satisfaction、公平性、offline rate、安全距离和边界约束。

采用集中训练、分散执行范式：训练阶段 critic 可利用更完整的联合状态，执行阶段每架 UAV 仅依赖本地可见信息决策。attention 模块用于建模 UAV 间关系和服务压力差异，避免简单拼接邻居状态造成关系表达不足。

### 下层：Oracle-Guided 请求级卸载

下层在每个请求到达时运行，候选动作包括：

1. `local`：由当前接入 UAV 执行；
2. `cooperative`：转发到可达协作 UAV 执行；
3. `mbs`：通过回传链路卸载到 MBS。

增强 oracle 对候选执行位置 `c` 定义代价：

```text
C_c =
  alpha_T * latency_c / deadline
  + alpha_D * deadline_violation_c
  + alpha_Q * queue_pressure_c
  + alpha_M * I[c = mbs] * mbs_load
  - alpha_C * I[c = coop] * cooperative_relief
```

标签为代价最小的候选执行位置。该 oracle 不等价于最小时延规则：当时延差异较小时，它倾向于减少 MBS fallback；当 deadline 压力过大或 UAV 队列过长时，它仍保留 MBS 作为必要回退路径。

## 已完成图表

所有图表已重新生成到 `docs/figures/`。当前建议主文优先使用以下图：

| 图 | 文件 | 用途 | 当前状态 |
| --- | --- | --- | --- |
| 图 1 | `fig_framework.png` | 双层协同优化框架 | 已生成 |
| 图 2 | `fig_pareto_tradeoff_scatter.png` | MBS 依赖与 DSR/性能权衡 | 已生成 |
| 图 3 | `fig_paired_seed_slope.png` | 同 seed 配对改进 | 已生成 |
| 图 4 | `fig_request_destination_stack.png` | 请求去向结构变化 | 已生成 |
| 图 5 | `fig_offload_classifier_quality.png` | 下层分类器质量与校准 | 已生成 |

附录图：

| 图 | 文件 | 说明 |
| --- | --- | --- |
| 图 A1 | `fig_training_convergence.png` | 当前为已有训练日志 rolling mean；正式投稿建议补多 seed CI |
| 图 A2 | `fig_lower_runtime.png` | 下层在线性能对比 |
| 图 A3 | `fig_mbs_dsr_confidence_tradeoff.png` | MBS 减负与 DSR 置信区间 |
| 图 A4 | `fig_seedwise_delta_ci.png` | seed-wise delta 与 CI |
| 图 A5 | `fig_scenario_sensitivity.png` | 多场景敏感性 |
| 图 A6 | `fig_trajectory_coverage.png` | 轨迹覆盖示意 |
| 图 A7 | `fig_uav_hotspot_fallback_heatmap.png` | 热点与 fallback 空间解释，需谨慎使用 |
| 图 A8 | `fig_hyperparameter_sensitivity_template.png` | 若 MBS penalty 扫描未运行，则仍是模板 |

图表目录文件：`docs/figures/FIGURE_CATALOG.md`。

## 已有实验结果

### 上层独立实验

数据来源：

- `results/full_runs/wpt_fix_thesis_run/test_logs/attention_mappo/`
- `results/full_runs/wpt_fix_thesis_run/test_logs/uncoordinated_greedy/`

测试日志均值如下：

| 策略 | Reward | Latency | Energy | DSR | Fairness | MBS Ratio | MBS Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uncoordinated_greedy` | -1466.68 | 1064784.72 | 105967844.94 | 45.27% | 0.7853 | 21.19% | 11.14% |
| `attention_mappo` | -1260.12 | 912568.18 | 101164953.60 | 54.69% | 0.9579 | 26.15% | 15.86% |

结论口径：attention-MAPPO 在 reward、latency、energy、DSR 和 fairness 上优于无协调贪心基线，说明上层关系建模有助于多 UAV 协同控制。需要注意，当前上层独立实验仍使用 heuristic offloading，因此 MBS ratio 和 MBS load 不一定下降；MBS 减负主要由下层 oracle-guided 卸载器承担。

### 下层卸载实验

数据来源：

- `results/full_offload_experiments/wpt_fix_thesis_run_offload/experiment_summary.json`
- `results/full_offload_experiments/wpt_fix_thesis_run_offload/reports/runtime_offload_policy_statistics.md`

在线运行结果：

| 策略 | Latency | Energy | DSR | Local Ratio | Coop Ratio | MBS Ratio | MBS Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `heuristic_offloading` | 1477.33 | 107305.72 | 24.64% | 40.99% | 22.93% | 32.46% | 8.41% |
| `surrogate_baseline` | 1477.64 | 125190.64 | 24.47% | 51.96% | 26.29% | 18.13% | 4.88% |
| `rich_reduced_runtime_policy` | 1478.12 | 126569.00 | 23.92% | 51.12% | 29.79% | 15.47% | 3.89% |

相对 `heuristic_offloading`，主展示策略 `surrogate_baseline` 的平均时延仅增加约 0.0205%，DSR 下降约 0.70%，但 MBS ratio 降低约 44.16%，MBS load 降低约 41.97%。这说明 oracle-guided 卸载器的主要价值是把一部分原本流向 MBS 的请求转移到本地 UAV 和协作 UAV，从而降低中心节点依赖。

`rich_reduced_runtime_policy` 更激进，进一步降低 MBS ratio 和 MBS load，但 DSR 与能耗代价更明显。因此当前主文建议以 `surrogate_baseline` 作为主策略，`rich_reduced_runtime_policy` 作为更强减负但代价更高的消融。

### 下层分类器质量

数据来源：

- `results/full_offload_experiments/wpt_fix_thesis_run_offload/reports/classifier_quality_summary.json`

验证集结果：

| 指标 | 数值 |
| --- | ---: |
| Validation samples | 3600 |
| Accuracy | 0.9447 |
| Macro-F1 | 0.9448 |
| ECE | 0.0224 |
| Brier score | 0.0845 |

结论口径：下层分类器较好拟合增强 oracle 的三分类边界，并且概率输出具有可接受的校准表现。该结果证明的是“学习器能稳定近似 oracle”，不是单独证明系统级最优。

## 端到端联合消融状态

正式四组联合消融尚未完成。当前只有 smoke 检查：

- 目录：`results/joint_experiments/joint_four_way_journal_check_smoke/`
- 规模：1 seed × 1 episode × 2 steps
- 作用：验证四组组合、模型加载、oracle-guided 卸载策略加载和统计报告生成是否正常

smoke 检查已经通过：

| 检查项 | 状态 |
| --- | --- |
| 四组 combo 是否齐全 | 通过 |
| oracle-guided 组合是否有 learned decisions | 通过 |
| `service_predict_exception_fallback_count == 0` | 通过 |
| 统计报告是否生成 | 通过 |

正式联合消融应运行：

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py --phase joint
```

正式输出目录：

```text
results/joint_experiments/joint_four_way_journal/
```

正式实验完成后，主文应报告四组：

| 组合 | 上层轨迹 | 下层卸载 | 目的 |
| --- | --- | --- | --- |
| `uncoordinated_greedy__heuristic` | uncoordinated greedy | heuristic | 强启发式基线 |
| `attention_mappo__heuristic` | attention-MAPPO | heuristic | 仅上层学习 |
| `uncoordinated_greedy__oracle_guided` | uncoordinated greedy | oracle-guided | 仅下层学习 |
| `attention_mappo__oracle_guided` | attention-MAPPO | oracle-guided | 完整双层方法 |

关键比较口径：

1. 上层收益：`attention_mappo__heuristic` vs `uncoordinated_greedy__heuristic`
2. 下层收益：`uncoordinated_greedy__oracle_guided` vs `uncoordinated_greedy__heuristic`
3. 完整双层收益：`attention_mappo__oracle_guided` vs `uncoordinated_greedy__heuristic`
4. 完整方法相对仅上层：`attention_mappo__oracle_guided` vs `attention_mappo__heuristic`

## 复现实验命令

### 重新生成统计报告和图表

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all_thesis_figures.ps1
```

该脚本会执行：

1. `py_compile` 关键脚本；
2. 重新生成下层 seed-wise 统计报告；
3. 重新生成分类器质量报告；
4. 重新生成 `docs/figures/` 下所有候选图。

### 验证四组联合消融脚本

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py `
  --phase joint `
  --smoke `
  --joint_name joint_four_way_journal_check
```

### 正式运行四组联合消融

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py --phase joint
```

默认设置：

- seeds：`42 84 126 168 210 252 294 336 378 420`
- episodes per seed：`6`
- steps per episode：使用配置默认值 `1000`

### 运行全部补充实验

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py --phase all
```

该命令会依次运行：

1. 上层 `attention_mappo` 多 seed 训练与同 seed `uncoordinated_greedy` 评估；
2. 端到端四组联合消融；
3. MBS penalty 轻量敏感性扫描；
4. 图表重新生成。

## 论文写作建议

当前最稳妥的论文叙事是：

1. 引言和相关工作中主动承认已有研究已经覆盖“轨迹 + 卸载 + 服务放置/缓存 + 资源分配”等联合优化，不把“联合优化本身”作为创新点。
2. 方法部分强调双层学习式分解：上层处理多 UAV 协同轨迹，下层处理请求级卸载。
3. 下层不要写成简单 surrogate，而应写成 oracle-guided behavior cloning / request-level policy learning。
4. 结果部分先展示上层独立有效性和下层独立有效性；正式联合消融完成前，不写“完整系统端到端收益已证实”。
5. 对 MBS 减负必须同时报告 DSR、latency 和 energy，避免被质疑只是把负载转嫁给 UAV。

## 投稿前待办

1. 跑正式四组联合消融 `run_supplement_experiments.py --phase joint`。
2. 跑上层补充训练 seeds `84 126 168`，形成多 seed training CI。
3. 跑 MBS penalty 扫描，替换当前超参数敏感性模板。
4. 将正式联合消融数值回填本文档和论文正文。
5. 补全 BibTeX，重点加入 UAV-MEC 综述、SPUN、INFOCOM 2022 双时间尺度、Collaborative Service Provisioning、JSAC 2025、MAPPO、Attention-Critic、Jain fairness 和 imitation learning 相关文献。

## 关键文件

| 文件或目录 | 作用 |
| --- | --- |
| `run_all_thesis_figures.ps1` | 一键生成统计报告、分类器质量报告和论文图表 |
| `generate_thesis_figures.py` | 生成 `docs/figures/` 下所有候选图 |
| `run_supplement_experiments.py` | 期刊补充实验调度入口 |
| `run_joint_trajectory_offload_experiment.py` | 端到端四组联合消融入口 |
| `run_full_offload_experiment.py` | 下层卸载策略完整实验入口 |
| `analyze_experiment_statistics.py` | mean/std/CI、paired delta 和显著性检验 |
| `analyze_offload_classifier_quality.py` | 下层分类器 confusion matrix、calibration、ECE、Brier、macro-F1 |
| `docs/figures/` | 当前已生成的论文候选图表 |
| `results/full_runs/wpt_fix_thesis_run/` | 当前上层独立实验结果 |
| `results/full_offload_experiments/wpt_fix_thesis_run_offload/` | 当前下层卸载实验结果 |
| `results/joint_experiments/joint_four_way_journal_check_smoke/` | 四组联合消融 smoke 检查结果 |
