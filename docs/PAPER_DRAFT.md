# 论文草图

## 题目候选

**面向多无人机移动边缘计算的双层协同优化：轨迹控制与学习式任务卸载**

可替换版本：

- 面向多无人机 MEC 的双层协同优化方法研究
- 基于轨迹控制与请求级卸载协同的多无人机移动边缘计算优化

## 摘要草稿

随着多无人机辅助移动边缘计算系统的发展，用户服务质量不仅受到无人机轨迹控制的影响，也受到请求到达后的任务卸载方式影响。已有研究已经较充分地讨论了轨迹、卸载、计算资源、服务放置和缓存的联合优化，因此本文不把“联合优化”本身作为主要创新点，而是关注如何用可部署的学习式双层分解降低端到端动作空间和在线求解复杂度。本文提出一种面向多无人机 MEC 的双层协同优化框架：上层采用 attention-MAPPO 建模 UAV 间协同关系并控制 UAV 运动，下层采用 oracle-guided request-level offloading policy，在本地 UAV、协作 UAV 与宏基站之间自适应选择执行位置。下层训练标签由增强 oracle 生成，仅使用当前请求上下文中的候选时延、deadline、队列压力、MBS 负载和协作收益，不使用未来信息。本文通过分阶段实验和四组联合消融验证上层、下层及二者组合的系统级作用，并使用同 seed 成对比较、置信区间和显著性检验增强实验可信度。

关键词：多无人机；移动边缘计算；双层协同优化；多智能体强化学习；任务卸载

## 第1章 绪论

### 1.1 研究背景

- 多无人机可通过机动部署增强边缘覆盖与计算服务能力。
- 在 MEC 场景下，服务性能不仅受 UAV 位置影响，也受请求卸载位置影响。
- 轨迹控制与任务卸载之间存在天然耦合关系。

### 1.2 研究意义

- 仅研究轨迹控制不足以完整刻画系统优化问题。
- 仅研究卸载决策也忽视了 UAV 运动对链路和协作机会的影响。
- 因此有必要构建双层协同优化框架。

### 1.3 国内外研究现状

建议按以下主线组织，而不是简单分成“传统方法 / 强化学习方法”：

1. MEC/UAV 与任务卸载基础：说明 UAV 作为空中接入节点、边缘计算节点和协作节点的系统背景。
2. 单 UAV 轨迹-卸载-资源联合优化：承认已有工作已经联合考虑轨迹、卸载和计算资源，指出其多 UAV 动态扩展成本。
3. 服务放置/缓存/双时间尺度：放入 SPUN、INFOCOM 2022、2024 Collaborative Service Provisioning、JSAC 2025 等工作，明确它们已经覆盖 service placement、service caching/content caching、task offloading、trajectory 和 resource allocation 的联合优化。
4. 多 UAV 协同与鲁棒 MEC：引出热点、链路瓶颈、MBS 拥塞和多 UAV 协同控制问题。
5. MARL 与 attention 方法：自然过渡到 attention-MAPPO 的关系建模作用。
6. 本文定位：双层学习式分解、oracle-guided request-level policy learning、在线复杂度下降和 MBS fallback 显式压缩。

### 1.4 本文主要工作

可直接写成：

1. 构建了一个面向多无人机移动边缘计算的双层协同优化框架，其中上层负责 UAV 轨迹控制，下层负责请求级服务卸载。
2. 在保持原有 UAV 运动动作空间不变的前提下，引入学习式请求级卸载机制，实现本地计算、协作计算和 MBS 卸载三类服务执行模式的统一决策。
3. 设计分阶段实验流程，分别筛选上层轨迹控制器和下层卸载策略，并通过联合实验评估双层组合的系统级性能。
4. 使用同 seed 四组联合消融、置信区间和 paired test 分析上层控制、下层 oracle-guided 卸载以及二者组合的系统级收益。

### 1.5 论文结构

- 第1章：绪论
- 第2章：系统模型与问题描述
- 第3章：双层协同优化框架
- 第4章：上层轨迹控制方法
- 第5章：下层学习式任务卸载方法
- 第6章：实验设计与结果分析
- 第7章：总结与展望

## 第2章 系统模型与问题描述

### 2.1 系统模型

建议画一个双层结构图：

- 上层：UAV trajectory control
- 下层：request-level offloading decision

系统元素：

- UAV 集群
- UE 集群
- MBS
- 无线链路：UE-UAV、UAV-UAV、UAV-MBS

### 2.2 请求模型

- 服务请求
- 内容请求
- 紧急能量请求

其中论文主线重点放在：

- 服务请求的请求级卸载

### 2.3 上层与下层耦合关系

- 轨迹影响 UE-UAV 链路速率
- 轨迹影响邻居 UAV 可达性
- 轨迹影响 UAV-MBS 回传能力
- 卸载策略影响时延、能耗、MBS 负载

### 2.4 问题定义

可写成：

目标是在满足系统运行约束的前提下，联合提升：

- 截止满足率
- 时延表现
- 能耗表现
- 负载分布合理性
- 用户公平性

## 第3章 双层协同优化框架

### 3.1 总体框架

这里明确两层：

- 上层：UAV 轨迹控制
- 下层：请求级卸载决策

### 3.2 框架设计原则

- 尽量复用原始环境中的轨迹控制接口
- 不把 UAV 动作空间扩展成“同时控制多个 UE 卸载”
- 将下层卸载设计成服务请求发生时的局部决策模块

### 3.3 框架优势

- 代码改动可控
- 系统逻辑清晰
- 适合逐层筛选方法
- 适合论文实验组织

## 第4章 上层轨迹控制方法

### 4.1 上层状态、动作与奖励

结合代码中的观测、动作和奖励设计来写。

### 4.2 候选轨迹控制算法

- `attention_mappo`
- `uncoordinated_greedy`

可补充：

- 其他方法用于早期筛选，但正文重点展示主算法与强基线

### 4.3 上层筛选实验设计

目标：

- 从候选方法中选出最终联合实验的上层控制器

指标：

- `deadline_satisfaction_rate`
- `latency`
- `energy`
- `fairness`

### 4.4 上层结果分析

上层多训练 seed 稳健性实验表明，`attention_mappo` 相比 `uncoordinated_greedy` 在 reward、energy 和 fairness 上具有更稳定优势。具体而言，attention-MAPPO 的 reward delta 为 +223.9，95% CI 为 [198.2, 249.7]，paired t-test 的 `p_t=0.000104`；energy delta 为 -10.90M，95% CI 为 [-19.99M, -1.81M]，`p_t=0.0316`；fairness delta 为 +0.1536，95% CI 为 [0.1140, 0.1931]，`p_t=0.00114`。latency、DSR、MBS ratio 和 MBS load 在独立上层多 seed 实验中不显著，因此这些服务质量收益主要以端到端四组联合消融作为主证据。

综合来看，attention-MAPPO 更适合作为双层协同框架的上层控制器：它能够提升协同覆盖公平性和整体 reward，并在完整双层实验中与下层 oracle-guided 卸载形成互补。

## 第5章 下层学习式任务卸载方法

### 5.1 问题描述

对于每个服务请求，下层策略需要在以下三种执行方式之间进行选择：

- 本地 UAV 执行
- 协作 UAV 执行
- MBS 执行

### 5.2 启发式规则与 oracle-guided 学习策略

- `heuristic_offloading`：传统规则
- `surrogate_baseline`：基于增强 oracle 标签训练的 oracle-guided offloading policy

建议写清楚：

- 该策略与 imitation / behavior cloning 接近，但标签不是简单复制启发式规则。
- 训练阶段 oracle 可以计算三类候选执行路径的代价；在线推理阶段模型只使用当前状态特征，不使用未来任务或未来轨迹信息。
- 增强 oracle 的关键作用是把 deadline 违约、MBS 负载、队列压力和协作分担收益写入标签生成过程，从而显式压缩 MBS fallback 依赖。

### 5.3 特征族与消融策略

- `rich_reduced_runtime_policy`

写法建议：

- 强调其使用更丰富的运行时状态，例如链路速率、协作邻居、计算资源和缓存置信度。
- 将其作为更激进 MBS 减负策略的消融，而不是替代 `surrogate_baseline` 的唯一主策略。

### 5.4 下层筛选实验设计

指标分两类：

1. 离线泛化能力
2. 在线系统级效果

离线指标：

- `accuracy`
- `macro_f1`
- `cross_scenario_split`

系统级指标：

- `latency`
- `deadline_satisfaction_rate`
- `mbs_load_ratio`

### 5.5 下层结果分析

实验结果表明，基于增强 oracle 的 `surrogate_baseline` 在基本保持 latency 和 DSR 的同时显著降低 MBS 依赖。相对 `heuristic_offloading`，`surrogate_baseline` 的 latency 仅增加约 0.0205%，DSR 下降约 0.70%，但 MBS ratio 降低约 44.16%，MBS load 降低约 41.97%。这说明 oracle-guided 标签能够把系统从过度依赖 MBS 的集中式处理引导到本地、协作和 MBS 混合分担。

`rich_reduced_runtime_policy` 进一步降低 MBS ratio 和 MBS load，但其 DSR 与能耗代价更明显。因此，本文将 `surrogate_baseline` 作为下层主策略，将 `rich_reduced_runtime_policy` 作为更激进负载分散的消融策略。

## 第6章 实验设计与结果分析

### 6.1 实验环境与参数设置

默认仿真区域为 `700 m x 700 m`，系统包含 5 架 UAV、100 个 UE 和 1 个 MBS。每个 episode 包含 1000 个 time slots，每个 time slot 时长为 1 s。UAV 飞行高度为 100 m，最大速度为 15 m/s，覆盖半径为 100 m，感知范围为 460 m，最小 UAV 间距为 200 m。系统包含 25 类服务和 50 类内容文件，服务 deadline 在 `[0.65, 2.10] s` 范围内生成。

主联合实验使用 10 个 workload seeds：`42, 84, 126, 168, 210, 252, 294, 336, 378, 420`。每个 seed 运行 6 个 episodes，统计单元为 seed-level episode mean，并报告 mean、std、95% CI、paired delta、paired t-test 和 Wilcoxon 检验。

### 6.2 上层轨迹控制筛选结果

数据来源：`results/full_runs/supplement_upper_multiseed/reports/upper_multiseed_statistics.md`。

| 方法 | Deadline Satisfaction | Latency | Energy | Fairness |
| --- | ---: | ---: | ---: | ---: |
| `attention_mappo` | 43.02% | 1141829.30 | 113841059.59 | 0.9252 |
| `uncoordinated_greedy` | 44.59% | 1087948.78 | 124741462.53 | 0.7716 |

多 seed paired comparison 显示，attention-MAPPO 的 reward、energy 和 fairness 改善更稳定；latency 与 DSR 在该独立上层实验中不显著。因此，上层筛选结果主要用于支撑 attention-MAPPO 的协同控制和公平性优势，正式服务质量收益以四组联合消融为主证据。

### 6.3 下层卸载策略筛选结果

下层分类器质量如下：

| 指标 | 数值 |
| --- | ---: |
| Validation samples | 3600 |
| Accuracy | 0.9447 |
| Macro-F1 | 0.9448 |
| ECE | 0.0224 |
| Brier score | 0.0845 |

下层在线系统级表现如下。数据来源：`results/full_offload_experiments/wpt_fix_thesis_run_offload/reports/runtime_offload_policy_statistics.md`。

| 策略 | Latency | Deadline Satisfaction | MBS Load Ratio |
| --- | ---: | ---: | ---: |
| `heuristic_offloading` | 1477.33 | 24.64% | 8.41% |
| `surrogate_baseline` | 1477.64 | 24.47% | 4.88% |
| `rich_reduced_runtime_policy` | 1478.12 | 23.92% | 3.89% |

`surrogate_baseline` 能够在 latency 和 DSR 变化较小的情况下显著降低 MBS load；`rich_reduced_runtime_policy` 的减负更激进，但 DSR 与能耗代价更明显。因此，后续联合实验采用 `surrogate_baseline` 对应的 oracle-guided runtime policy 作为下层主策略。

### 6.4 联合实验结果

数据来源：

- `results/joint_experiments/joint_four_way_journal/joint_experiment_summary.json`
- `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_uncoordinated_heuristic.md`
- `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_attention_heuristic.md`

| 组合方案 | Latency | Energy | Deadline Satisfaction | Fairness | MBS Load Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uncoordinated_greedy + heuristic` | 1147588.82 | 124688169.38 | 41.80% | 0.7585 | 9.43% |
| `attention_mappo + heuristic` | 949043.86 | 116501996.54 | 52.89% | 0.9434 | 15.19% |
| `uncoordinated_greedy + oracle_guided` | 1148210.60 | 108502049.08 | 41.72% | 0.7586 | 2.57% |
| `attention_mappo + oracle_guided` | 947602.60 | 101925121.62 | 52.83% | 0.9461 | 4.73% |

相对 `uncoordinated_greedy + heuristic`，完整方法 `attention_mappo + oracle_guided` 的 latency 降低 17.43%，energy 降低 18.26%，DSR 提升 11.02 个百分点，fairness 提升 0.1876，MBS load ratio 降低 49.90%。相对 `attention_mappo + heuristic`，完整方法在 reward 与 DSR 不显著下降的情况下，energy 降低 12.51%，MBS ratio 降低 69.86%，MBS load ratio 降低 68.87%。这说明上层 attention-MAPPO 主要提升协同轨迹控制和服务质量，下层 oracle-guided 卸载主要压缩 MBS fallback 依赖。

统计协议：

- 所有组合使用同一批 workload seeds、UE 地图和任务到达过程。
- 对主要指标报告 mean ± std、95% CI、相对参考组的 paired delta。
- 对 seed-wise paired comparison 报告 paired t-test；若分布不稳定，同时报告 Wilcoxon。
- 对 DSR、MBS ratio、MBS load 等比例指标，报告中保留 ratio metric note；当前统计单元为 seed-level mean。

### 6.5 结果讨论

实验结果可以从三个层次理解：

1. 上层 attention-MAPPO 主要影响 UAV 协同轨迹、覆盖公平性和服务质量。在四组联合消融中，仅替换上层控制器即可明显提升 reward、latency、DSR 和 fairness，但同时也会提高 MBS fallback 依赖。
2. 下层 oracle-guided 学习策略主要改变请求去向。相对 heuristic，它显著减少 MBS ratio 和 MBS load，并将更多请求转移到 local 与 cooperative 执行路径。
3. 完整双层方法的优势不是单纯刷新 reward，而是在保持上层服务质量收益的同时显著降低 energy 和 MBS load。这说明轨迹控制和请求级卸载在系统目标上具有互补性。

## 第7章 总结与展望

### 7.1 总结

本文围绕多无人机移动边缘计算中的轨迹控制与任务卸载问题，提出了一种双层协同优化框架。该框架在上层采用 attention-MAPPO 进行 UAV 轨迹控制，在下层采用 oracle-guided request-level policy learning 进行请求级卸载决策。通过分阶段筛选实验与最终四组联合消融，本文验证了双层建模对刻画系统耦合关系的有效性。实验结果表明，上层 attention-MAPPO 能提升协同轨迹控制和服务质量，下层 oracle-guided 卸载能显著降低 MBS fallback 依赖；完整方法在保持上层 reward 与 DSR 收益的同时进一步降低 energy、MBS ratio 和 MBS load。因此，本文工作为多无人机 MEC 场景下轨迹控制与任务卸载的协同优化提供了一种可运行、可分析的研究范式。

### 7.2 展望

- 引入更强的下层学习策略
- 实现真正端到端联合训练，但不把当前方法表述为全局最优端到端优化
- 引入更多动态场景和非模板分布测试
- 将缓存、能量和轨迹进一步统一建模

## 已同步结果清单

当前草稿已经同步以下正式结果：

- 第4章上层多训练 seed 稳健性结果
- 第5章下层在线卸载策略结果
- 第6章四组联合消融主表
- 第7章总结中的核心实验结论
