# 论文草图

## 题目候选

**Quality-aware Constrained Hierarchical MARL for Multi-UAV MEC**

可替换版本：

- 面向多无人机 MEC 的质量感知约束式双层多智能体强化学习方法
- 面向多无人机移动边缘计算的双层 MARL 轨迹控制与请求级卸载

## 摘要草稿

随着多无人机辅助移动边缘计算系统的发展，用户服务质量不仅受到无人机轨迹控制的影响，也受到请求到达后的任务卸载方式影响。已有研究已经较充分地讨论了轨迹、卸载、计算资源、服务放置和缓存的联合优化，因此本文不把“联合优化”本身作为主要创新点，而是关注如何用可部署的双层多智能体强化学习分解降低端到端动作空间和在线求解复杂度。本文提出一种 Quality-aware Constrained Hierarchical MARL 框架：上层采用 attention-MAPPO 建模 UAV 间协同关系并控制 UAV 轨迹，下层采用 constrained attention offload MAPPO 对服务请求进行 local UAV、cooperative UAV 与 MBS 三类离散卸载决策。下层策略使用质量感知动作 mask 抑制明显不可行的协作与 MBS 决策，并通过 Lagrange 约束项刻画 deadline satisfaction 与 MBS load 的权衡。oracle-guided 分类器不再作为主方法，而作为 teacher/reference baseline，用于衡量纯 RL 下层策略相对强教师策略的系统级表现。本文通过主表、下层消融和 paired statistical tests 验证上下层 MARL 的贡献。

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
6. 本文定位：双层 MARL 分解、质量感知动作 mask、约束式请求级卸载学习、在线复杂度下降和 MBS fallback 显式压缩。

### 1.4 本文主要工作

可直接写成：

1. 构建了一个面向多无人机 MEC 的双层 MARL 框架，其中上层 attention-MAPPO 负责 UAV 轨迹控制，下层 constrained attention offload MAPPO 负责请求级服务卸载。
2. 设计质量感知动作 mask，在保持空请求 slot 安全屏蔽的基础上，对明显不可行的 cooperative UAV 与 MBS 决策进行运行时约束。
3. 在下层 MAPPO reward 中引入 deadline satisfaction、latency、energy、MBS load 与 cooperative bonus，并用 Lagrange 乘子显式控制 DSR/MBS load 权衡。
4. 使用训练 seed 与 workload seed 的成对统计单元，比较 heuristic、oracle-guided teacher/reference、lower-MAPPO 与 full hierarchical MARL，并报告置信区间和 paired tests。

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

本文框架由两个纯 MARL 决策层组成：

- 上层：attention-MAPPO，根据 UAV 局部观测、邻居关系和关联 UE 请求状态输出连续轨迹动作。
- 下层：constrained attention offload MAPPO，对每架 UAV 当前服务请求 slot 输出离散卸载动作：local、cooperative UAV 或 MBS。

上下层在同一环境 step 内协同工作：下层先读取当前请求观测与动作 mask，生成请求级卸载动作；环境处理服务请求、统计 DSR/MBS load/latency/energy；上层轨迹动作随后更新 UAV 下一时刻位置，从而影响下一步链路、邻居和请求覆盖。

### 3.2 框架设计原则

- 不把上层 UAV 动作扩展成混合大动作，避免把轨迹和多个请求卸载拼接成高维联合动作。
- 下层仍然是 MARL：每架 UAV 是一个 lower-layer agent，对本 UAV 的 bounded request slots 做离散 MAPPO 决策。
- 所有学习式主方法不使用 oracle imitation warm-start；oracle-guided 仅作为 teacher/reference baseline。
- 空请求 slot 始终被 mask 掉，保证 no-mask 消融只关闭质量 mask，而不处理不存在的请求。

### 3.3 框架优势

- 双层分解降低端到端动作空间，便于分别分析轨迹控制和请求级卸载的贡献。
- 下层质量 mask 将不可行协作和过慢 MBS 路径显式排除，减少无效探索。
- Lagrange 约束使下层策略能够在 deadline satisfaction 与 MBS load 之间形成可解释权衡。
- 训练与评估都支持 upper-only、fixed-upper lower-only 和 full hierarchical 三种模式，便于主表和消融实验复现。

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

上层多训练 seed 稳健性实验表明，`attention_mappo` 相比 `uncoordinated_greedy` 在 reward、energy 和 fairness 上具有更稳定优势。具体而言，attention-MAPPO 的 reward delta 为 +223.9，95% CI 为 [198.2, 249.7]，paired t-test 的 `p_t=0.000104`；energy delta 为 -10.90M，95% CI 为 [-19.99M, -1.81M]，`p_t=0.0316`；fairness delta 为 +0.1536，95% CI 为 [0.1140, 0.1931]，`p_t=0.00114`。latency、DSR、MBS ratio 和 MBS load 在独立上层多 seed 实验中不显著，因此这些服务质量收益主要以后续六组主联合实验作为主证据。

综合来看，attention-MAPPO 更适合作为双层 MARL 框架的上层控制器：它能够提升协同覆盖公平性和整体 reward，并在完整双层实验中与下层 constrained attention offload MAPPO 形成互补。oracle-guided 策略仅作为 teacher/reference baseline，用于说明强监督式请求级策略可以达到的卸载质量。

## 第5章 下层学习式任务卸载方法

### 5.1 问题描述

对于每个服务请求，下层策略需要在以下三种执行方式之间进行选择：

- 本地 UAV 执行
- 协作 UAV 执行
- MBS 执行

### 5.2 Constrained attention offload MAPPO

下层主方法是 constrained attention offload MAPPO。每架 UAV 是一个下层 agent，观测由 UAV 自身状态和最多 `MAX_OFFLOAD_REQUESTS_PER_UAV` 个服务请求 slot 组成；每个有效 slot 的动作空间为：

- `0`：local UAV 执行
- `1`：cooperative UAV 执行
- `2`：MBS 执行

actor 使用 request attention 编码同一 UAV 内多个请求之间的相对重要性，critic 使用 UAV 间 attention 估计集中式 value。reward 同时包含 deadline satisfaction、latency、energy、MBS load 和 cooperative bonus，并在 Lagrange 模式下加入 DSR 与 MBS load 约束惩罚。训练过程完全来自环境交互，不使用 oracle imitation 或 warm-start。

### 5.3 质量感知动作 mask 与消融

质量 mask 对每个请求 slot 分别判断 cooperative UAV 和 MBS 动作是否明显不可取：

- cooperative mask：当无可用邻居、协作时延过高、相对非协作路径无优势或协作算力份额过低时关闭 cooperative 动作。
- MBS mask：当 MBS 路径时延显著超过 deadline clip 阈值时关闭 MBS 动作。
- empty slot mask：无请求 slot 永远不产生真实卸载动作；该保护在 `no_mask` 消融中也保留。

下层消融包括：

- `full`：attention actor + quality mask + Lagrange。
- `no_mask`：关闭 cooperative/MBS quality mask，仅保留 empty slot 保护。
- `no_lagrange`：保留 quality mask，关闭约束乘子。
- `no_attention`：用 MLP actor/critic 替代 request attention，其他训练口径保持一致。

### 5.4 Teacher/reference baselines

本文保留两个非主方法卸载基线：

- `heuristic_offloading`：传统启发式规则。
- `oracle_guided` / `surrogate_baseline`：由增强 oracle 标签训练的强教师参考策略。

oracle-guided 策略与 imitation / behavior cloning 接近，但本文不再把它作为下层主策略。它的作用是提供 teacher/reference：训练阶段 oracle 可以计算三类候选执行路径的代价；在线推理阶段模型只使用当前状态特征，不使用未来任务或未来轨迹信息。`rich_reduced_runtime_policy` 作为更激进的 MBS 减负参考。

### 5.5 下层实验设计

下层主实验报告：

- reward、latency、energy、deadline satisfaction、fairness、offline rate。
- local/cooperative/MBS ratio、MBS load ratio。
- lower actor loss、critic loss、entropy。
- `lambda_dsr`、`lambda_mbs`、constraint penalty、coop/MBS masked count。

一致性检查包括：lower_mappo 路径下 learned decision count 必须大于 0；heuristic fallback 只应来自非法动作或 cooperative 不可执行；empty request slot 不产生真实卸载动作；`no_mask` 消融仍然不会处理不存在的请求 slot。

### 5.6 下层结果分析

待完整重跑后，本节以 `full` lower MAPPO 作为下层主方法，比较 `no_mask`、`no_lagrange` 和 `no_attention`。分析重点不是分类准确率，而是质量 mask 是否减少无效协作/MBS 动作、Lagrange 是否稳定 DSR/MBS load 权衡，以及 request attention 是否改善多请求 slot 下的卸载分配。

已有 oracle-guided 结果可作为 teacher/reference baseline：该策略在基本保持 latency 和 DSR 的同时显著降低 MBS 依赖，说明请求级学习策略确实能够压缩 MBS fallback。但正式主线应报告纯 MAPPO 下层与 oracle-guided teacher 的差距，而不是把 oracle-guided 写成最终主方法。

## 第6章 实验设计与结果分析

### 6.1 实验环境与参数设置

默认仿真区域为 `700 m x 700 m`，系统包含 5 架 UAV、100 个 UE 和 1 个 MBS。每个 episode 包含 1000 个 time slots，每个 time slot 时长为 1 s。UAV 飞行高度为 100 m，最大速度为 15 m/s，覆盖半径为 100 m，感知范围为 460 m，最小 UAV 间距为 200 m。系统包含 25 类服务和 50 类内容文件，服务 deadline 在 `[0.65, 2.10] s` 范围内生成。

主联合实验使用 3 个训练 seeds：`42, 84, 126`，以及 10 个 workload seeds：`42, 84, 126, 168, 210, 252, 294, 336, 378, 420`。每个 workload seed 运行 6 个 episodes，每个 episode 1000 steps。统计单元为 `(training_seed, workload_seed)` 的 episode mean，并报告 mean、std、95% CI、paired delta、paired t-test 和 Wilcoxon 检验。

### 6.2 上层轨迹控制筛选结果

数据来源：`results/full_runs/` 下的主线上层多 seed 训练与测试日志。

| 方法 | Deadline Satisfaction | Latency | Energy | Fairness |
| --- | ---: | ---: | ---: | ---: |
| `attention_mappo` | 43.02% | 1141829.30 | 113841059.59 | 0.9252 |
| `uncoordinated_greedy` | 44.59% | 1087948.78 | 124741462.53 | 0.7716 |

多 seed paired comparison 显示，attention-MAPPO 的 reward、energy 和 fairness 改善更稳定；latency 与 DSR 在该独立上层实验中不显著。因此，上层筛选结果主要用于支撑 attention-MAPPO 的协同控制和公平性优势，正式服务质量收益以六组主联合实验为主证据。

### 6.3 下层卸载策略与消融

下层主方法和消融如下：

| 方法 | Attention | Quality mask | Lagrange |
| --- | --- | --- | --- |
| `full_hierarchical_marl` | yes | yes | yes |
| `no_mask` | yes | no | yes |
| `no_lagrange` | yes | yes | no |
| `no_attention_lower` | no | yes | yes |

每个消融仍然使用 MAPPO，不使用 oracle imitation warm-start。报告指标包括 reward、latency、energy、DSR、MBS load ratio、lower actor/critic loss、entropy、`lambda_dsr`、`lambda_mbs`、constraint penalty 和 mask count。

oracle-guided 分类器保留为 teacher/reference baseline。已有结果显示 teacher 能显著降低 MBS 依赖，因此它用于衡量纯 RL 下层策略的距离，而不作为本文主方法。

### 6.4 联合实验结果

数据来源：

- `results/joint_experiments/joint_four_way_journal/joint_experiment_summary.json`
- `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_uncoordinated_heuristic.md`
- `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_attention_heuristic.md`

| 组合方案 | 轨迹控制 | 卸载策略 | 角色 |
| --- | --- | --- | --- |
| `uncoordinated_greedy + heuristic` | uncoordinated greedy | heuristic | 基础参考 |
| `attention_mappo + heuristic` | attention-MAPPO | heuristic | 上层贡献 |
| `attention_mappo + oracle_guided` | attention-MAPPO | oracle-guided teacher | 强教师参考 |
| `uncoordinated_greedy + lower_mappo` | uncoordinated greedy | lower MAPPO | 下层 MARL 迁移/固定上层口径 |
| `attention_mappo + lower_mappo` | attention-MAPPO | lower MAPPO | 分别训练的上下层组合 |
| `full_hierarchical_marl` | attention-MAPPO | lower MAPPO | 主方法 |

完整重跑后，本节应以 `full_hierarchical_marl` 作为主方法，重点比较：

- 相对 `attention_mappo + heuristic`：下层 MAPPO 是否降低 MBS ratio/MBS load，并保持 DSR。
- 相对 `attention_mappo + oracle_guided`：纯 RL 下层与 teacher/reference 的差距。
- 相对 `attention_mappo + lower_mappo`：full hierarchical 同训是否优于固定上层下层训练。
- 相对 `uncoordinated_greedy + lower_mappo`：同一下层 MARL 口径下，上层 attention-MAPPO 是否提供更好的系统级环境分布。

统计协议：

- 所有组合使用同一批 training seeds、workload seeds、UE 地图和任务到达过程。
- 对主要指标报告 mean ± std、95% CI、相对参考组的 paired delta。
- 对 `(training_seed, workload_seed)` paired comparison 报告 paired t-test；若分布不稳定，同时报告 Wilcoxon。
- 对 DSR、MBS ratio、MBS load 等比例指标，报告中保留 ratio metric note；当前统计单元为 paired episode mean。

### 6.5 结果讨论

实验结果可以从三个层次理解：

1. 上层 attention-MAPPO 主要影响 UAV 协同轨迹、覆盖公平性和服务质量。在主联合实验中，仅替换上层控制器即可明显提升 reward、latency、DSR 和 fairness，但同时也可能提高 MBS fallback 依赖。
2. 下层 constrained attention offload MAPPO 主要改变请求去向。相对 heuristic，它应减少无效 MBS fallback，并将更多请求转移到 local 与 cooperative 执行路径。
3. 完整双层 MARL 的优势不是单纯刷新 reward，而是在保持上层服务质量收益的同时控制 energy、DSR 与 MBS load。这说明轨迹控制和请求级卸载在系统目标上具有互补性。

## 第7章 总结与展望

### 7.1 总结

本文围绕多无人机移动边缘计算中的轨迹控制与任务卸载问题，提出了一种质量感知约束式双层 MARL 框架。该框架在上层采用 attention-MAPPO 进行 UAV 轨迹控制，在下层采用 constrained attention offload MAPPO 进行请求级卸载决策。通过主表、下层消融和 teacher/reference baseline 对比，本文验证双层 MARL 分解、质量感知动作 mask 和 Lagrange 约束对系统级服务质量与 MBS 负载控制的作用。因此，本文工作为多无人机 MEC 场景下轨迹控制与任务卸载的协同优化提供了一种可运行、可分析的研究范式。

### 7.2 展望

- 引入更强的下层学习策略
- 实现真正端到端联合训练，但不把当前方法表述为全局最优端到端优化
- 引入更多动态场景和非模板分布测试
- 将缓存、能量和轨迹进一步统一建模

## 已同步结果清单

当前草稿已经同步以下正式结果：

- 第4章上层多训练 seed 稳健性结果
- 第5章下层 MAPPO 主方法、消融和 teacher/reference 口径
- 第6章六组主表与四组下层消融的实验设计
- 第7章总结中的纯 MARL 主线表述
