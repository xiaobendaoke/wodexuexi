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

这里先留模板：

- 若 `attention_mappo` 更优：强调其在动态环境中的学习能力
- 若 `uncoordinated_greedy` 更优：强调强启发式基线的竞争力，同时说明学习型控制器具备更好的扩展潜力

可填空模板：

“实验结果表明，在 ______ 指标上，`attention_mappo` / `uncoordinated_greedy` 表现更优；在 ______ 指标上，另一方法具有一定优势。综合考虑 ______，本文选取 `attention_mappo` 作为双层协同框架的上层控制器。”

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

模板：

“实验结果表明，基于增强 oracle 的 `surrogate_baseline` 在基本保持 ______ 的同时显著降低 ______，说明 oracle-guided 标签能够把系统从过度依赖 MBS 的集中式处理引导到本地、协作和 MBS 混合分担。`rich_reduced_runtime_policy` 进一步降低 ______，但代价是 ______，因此本文将其作为更激进负载分散的消融策略。”

## 第6章 实验设计与结果分析

### 6.1 实验环境与参数设置

这里后续补：

- UAV 数量
- UE 数量
- 区域尺寸
- 时间步长
- 训练轮数
- 测试轮数
- 硬件环境

### 6.2 上层轨迹控制筛选结果

可插入：

- 图：训练/测试曲线
- 表：`attention_mappo` vs `uncoordinated_greedy`

表格模板：

| 方法 | Deadline Satisfaction | Latency | Energy | Fairness |
| --- | ---: | ---: | ---: | ---: |
| attention_mappo | TBD | TBD | TBD | TBD |
| uncoordinated_greedy | TBD | TBD | TBD | TBD |

### 6.3 下层卸载策略筛选结果

表格模板 1：离线泛化

| 策略 | IID Accuracy | IID Macro-F1 | Cross-scenario Accuracy | Cross-scenario Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| heuristic | - | - | - | - |
| surrogate | TBD | TBD | TBD | TBD |
| rich_reduced | TBD | TBD | TBD | TBD |

表格模板 2：系统级表现

| 策略 | Latency | Deadline Satisfaction | MBS Load Ratio |
| --- | ---: | ---: | ---: |
| heuristic | TBD | TBD | TBD |
| surrogate | TBD | TBD | TBD |
| rich_reduced | TBD | TBD | TBD |

### 6.4 联合实验结果

最终主实验至少展示四组：

- `uncoordinated_greedy + heuristic`
- `attention_mappo + heuristic`
- `uncoordinated_greedy + oracle_guided`
- `attention_mappo + oracle_guided`

表格模板：

| 组合方案 | Latency | Energy | Deadline Satisfaction | Fairness | MBS Load Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| uncoordinated_greedy + heuristic | TBD | TBD | TBD | TBD | TBD |
| attention_mappo + heuristic | TBD | TBD | TBD | TBD | TBD |
| uncoordinated_greedy + oracle_guided | TBD | TBD | TBD | TBD | TBD |
| attention_mappo + oracle_guided | TBD | TBD | TBD | TBD | TBD |

统计呈现要求：

- 所有组合使用同一批 workload seeds、UE 地图和任务到达过程。
- 对主要指标报告 mean ± std、95% CI、相对参考组的 paired delta。
- 对 seed-wise paired comparison 报告 paired t-test；若分布不稳定，同时报告 Wilcoxon。
- 对 DSR、MBS ratio、MBS load 等比例指标，至少给出 bootstrap CI；若后续保存请求级成功/总数，可补充 binomial CI。

### 6.5 结果讨论

可以围绕三点展开：

1. 上层轨迹控制是否显著影响下层策略效果
2. oracle-guided 学习策略与启发式规则的差别
3. 当前双层框架的优势与局限

## 第7章 总结与展望

### 7.1 总结

模板：

本文围绕多无人机移动边缘计算中的轨迹控制与任务卸载问题，提出了一种双层协同优化框架。该框架在上层采用 ______ 进行 UAV 轨迹控制，在下层采用 ______ 进行请求级卸载决策。通过分阶段筛选实验与最终联合实验，验证了双层建模对刻画系统耦合关系的有效性。实验结果表明，______。因此，本文工作为多无人机 MEC 场景下轨迹控制与任务卸载的协同优化提供了一种可运行、可分析的研究范式。

### 7.2 展望

- 引入更强的下层学习策略
- 实现真正端到端联合训练，但不把当前方法表述为全局最优端到端优化
- 引入更多动态场景和非模板分布测试
- 将缓存、能量和轨迹进一步统一建模

## 结果填写提醒

后续你跑完实验后，优先把下面这些位置补上：

- 第4章上层筛选结果
- 第5章下层筛选结果
- 第6章联合实验表格
- 摘要中的核心实验结论
- 第7章总结中的结果句子
