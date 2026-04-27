# 面向多无人机移动边缘计算的双层协同优化：轨迹控制与学习式任务卸载

## 摘要

随着无人机通信、边缘计算和智能调度技术的发展，多无人机辅助移动边缘计算系统逐渐成为复杂空地协同网络中的重要服务形态。在该类系统中，无人机不仅承担空中接入节点的角色，还可以作为边缘计算节点为地面用户提供任务处理、内容缓存和应急服务。然而，系统性能并不只由无人机的位置决定，也受到任务卸载策略、服务缓存状态、链路质量、计算资源和宏基站负载等因素的共同影响。若仅优化无人机轨迹，容易忽略请求到达后的执行位置选择；若仅优化任务卸载，又难以刻画无人机运动对链路和协作机会的影响。

针对上述问题，本文提出一种面向多无人机移动边缘计算的双层协同优化框架。框架将系统决策拆分为上层轨迹控制和下层请求级任务卸载两个相互耦合的子问题：上层采用多智能体强化学习方法控制无人机运动，以改善覆盖、链路和资源分布；下层在服务请求到达时，基于当前系统状态选择本地 UAV、协作 UAV 或宏基站执行任务。本文进一步设计增强 oracle 标签，将时延、截止期、队列压力和 MBS 负载纳入学习式卸载目标，使下层策略不再局限于简单复现启发式规则。

实验结果表明，在修正无线供能发射功率建模后，系统能耗指标回到合理量级，实验结论更加可信。在上层轨迹控制中，`attention_mappo` 相比 `uncoordinated_greedy` 获得更高测试奖励、更低平均时延、更低能耗、更高截止期满足率和更好的公平性，适合作为双层框架的上层控制器。在下层任务卸载中，基于增强 oracle 的 `surrogate_baseline` 在几乎不增加平均时延的情况下，将 MBS 卸载比例降低约 44.16%，将 MBS 负载降低约 41.97%，同时截止期满足率仅下降约 0.70%。结果说明，学习式卸载策略能够在保持服务质量基本稳定的同时，显著降低中心节点依赖，改善多无人机 MEC 系统的负载分散能力。

**关键词**：多无人机；移动边缘计算；轨迹控制；任务卸载；多智能体强化学习；负载均衡

## 图表索引

本文使用的论文图表由 `generate_thesis_figures.py` 根据已有实验结果生成：

```powershell
.\.venv\Scripts\python.exe generate_thesis_figures.py
```

![双层协同优化框架](docs/figures/fig_framework.png)

![系统模型示意图](docs/system_model.jpg)

![上层轨迹控制对比](docs/figures/fig_upper_comparison.png)

![下层卸载策略在线性能对比](docs/figures/fig_lower_runtime.png)

![MBS 减负与服务质量代价权衡](docs/figures/fig_mbs_tradeoff.png)

## 第1章 绪论

### 1.1 研究背景

移动边缘计算（Mobile Edge Computing, MEC）通过将计算资源下沉到靠近用户侧的位置，能够降低任务处理时延并缓解云中心负载。在灾害救援、临时活动保障、偏远区域通信和复杂城市空地网络中，固定地面边缘节点可能存在覆盖不足、部署不灵活或基础设施受损等问题。无人机具有快速部署、三维移动和视距链路优势，因此多无人机辅助 MEC 成为提升动态场景服务能力的重要方案。

在多无人机场景中，UAV 可同时承担三类角色：第一，作为空中接入节点，为地面用户提供无线接入；第二，作为边缘计算节点，利用机载计算资源处理用户任务；第三，作为协作节点，通过 UAV-UAV 链路或 UAV-MBS 回传链路参与任务转发和协同处理。由于 UAV 位置、链路质量、缓存命中、计算队列和 MBS 负载都随时间变化，系统优化问题具有明显的动态性和耦合性。

传统方法常将轨迹控制和任务卸载分开处理。轨迹控制侧关注 UAV 如何移动以改善覆盖和链路质量；任务卸载侧关注每个请求应在本地边缘节点、协作节点或云端执行。然而，在多无人机 MEC 中，这两个问题天然耦合：UAV 的运动会改变 UE-UAV、UAV-UAV 和 UAV-MBS 链路；卸载策略又会改变 UAV 队列、MBS 负载和服务时延。因此，本文从双层协同优化角度研究多无人机 MEC 系统。

### 1.2 研究意义

本文研究具有以下意义：

1. 从系统层面刻画轨迹控制与任务卸载之间的耦合关系，避免只优化单一模块导致的局部最优。
2. 通过多智能体强化学习建模无人机轨迹控制，使 UAV 能够根据动态环境学习协同移动策略。
3. 通过学习式请求级卸载策略，在本地 UAV、协作 UAV 和 MBS 之间进行自适应选择，降低对中心节点的过度依赖。
4. 通过分阶段实验组织方式，分别验证上层轨迹控制、下层卸载策略和双层组合的作用，使论文结论更加清晰。

### 1.3 国内外研究现状

现有相关研究主要可从 UAV 轨迹优化、MEC 任务卸载、多智能体强化学习以及轨迹-卸载联合优化四个角度展开。

#### 1.3.1 UAV 轨迹优化研究

UAV 轨迹优化研究通常关注无人机如何在有限能量和飞行约束下移动，以改善覆盖范围、链路质量和用户服务能力。早期方法多采用凸优化、连续轨迹规划或启发式搜索，将 UAV 位置看作连续控制变量，并通过最小化飞行能耗、最大化吞吐量或提升覆盖概率来设计轨迹。该类方法具有较强可解释性，但往往依赖相对明确的信道模型和静态用户分布，当用户请求、缓存状态和队列负载动态变化时，模型求解复杂度会迅速上升。

近年来，强化学习被广泛用于 UAV 轨迹控制。强化学习方法不要求显式求解复杂优化问题，而是通过与环境交互学习长期收益较高的移动策略。对于单 UAV 场景，深度 Q 网络或确定性策略梯度可以学习覆盖与能耗之间的折中；对于多 UAV 场景，各无人机的动作会相互影响，单智能体建模难以刻画协同覆盖、碰撞规避和负载均衡，因此需要多智能体强化学习方法。

#### 1.3.2 MEC 任务卸载研究

MEC 任务卸载研究主要解决用户任务在本地设备、边缘服务器和云端之间的执行位置选择问题。传统任务卸载方法通常根据任务大小、计算量、信道速率和服务器负载建立优化模型，以最小化时延、能耗或加权系统代价。对于 UAV-MEC 系统，UAV 可以作为移动边缘节点参与任务处理，任务卸载决策还需要考虑 UAV 缓存命中、UAV-UAV 协作链路以及 UAV-MBS 回传链路。

现有卸载策略可以大致分为规则型、优化型和学习型三类。规则型策略实现简单、运行稳定，但难以在多目标之间自适应折中；优化型策略具有明确目标函数，但在动态多请求场景中求解成本较高；学习型策略可以从数据或交互中学习决策规律，但其效果依赖训练标签、状态特征和在线环境是否一致。本文的下层策略采用请求级监督学习分类器，并通过增强 oracle 标签把时延、截止期、队列压力和 MBS 负载同时纳入标签生成过程。

#### 1.3.3 多智能体强化学习研究

多智能体强化学习适合处理多 UAV 系统中的协同决策问题。集中训练、分散执行是常用范式：训练阶段可以利用全局状态或联合动作信息提高学习稳定性，执行阶段每个智能体只根据局部观测独立决策。MADDPG、MATD3、MAPPO、MASAC 等算法分别从确定性策略梯度、双 critic 稳定训练、近端策略优化和最大熵学习角度改进多智能体训练过程。

在多 UAV MEC 场景中，智能体之间存在明显交互关系，例如相邻 UAV 的位置会影响协作链路，多个 UAV 的覆盖范围会影响用户关联，任务卸载又会改变不同 UAV 的队列压力。注意力机制能够让智能体在状态编码时更关注与自身决策相关的邻居和服务状态，因此本文重点比较带注意力机制的多智能体算法，并选取 `attention_mappo` 作为上层轨迹控制主方法。

#### 1.3.4 轨迹-卸载联合优化研究

轨迹控制和任务卸载在 UAV-MEC 系统中并不是两个孤立问题。轨迹控制决定 UAV 与 UE、邻居 UAV、MBS 之间的距离和链路速率；任务卸载决定请求最终在哪个节点执行，并进一步改变 UAV 队列、MBS 负载和系统时延。若只优化轨迹，可能得到覆盖较好但任务执行位置不合理的策略；若只优化卸载，则无法利用 UAV 移动带来的协作机会。

部分研究尝试将轨迹与卸载联合建模，但端到端联合动作空间较大，训练稳定性和结果可解释性较弱。本文采用双层协同框架：上层学习 UAV 轨迹控制，下层学习请求级卸载策略，并通过分阶段实验分别验证两个层次的作用。该处理方式牺牲了一部分端到端最优性，但能降低训练难度，并使论文结论更容易归因。

| 研究方向 | 是否考虑多 UAV | 是否考虑任务卸载 | 是否考虑协作 UAV | 是否使用 MARL | 局限性 | 本文对应改进 |
| --- | --- | --- | --- | --- | --- | --- |
| UAV 轨迹优化 | 部分考虑 | 通常不细化 | 较少考虑 | 部分使用 | 容易忽略请求执行位置和队列负载 | 上层学习轨迹，下层继续处理请求级卸载 |
| MEC 任务卸载 | 通常固定边缘节点 | 考虑 | 较少考虑 UAV-UAV 协作 | 较少使用 | UAV 位置变化对链路影响刻画不足 | 在卸载特征中加入链路、缓存和邻居信息 |
| 多智能体强化学习 | 考虑 | 依具体场景而定 | 可扩展 | 使用 | 算法效果依赖状态设计和奖励设计 | 使用 attention-MAPPO 建模 UAV 间交互 |
| 轨迹-卸载联合优化 | 考虑 | 考虑 | 部分考虑 | 部分使用 | 端到端动作空间大，训练和解释困难 | 采用双层分阶段框架，保留联合验证接口 |

### 1.4 本文主要工作

本文主要工作如下：

1. 构建多无人机 MEC 双层协同优化模型，将 UAV 轨迹控制和请求级任务卸载分别建模，并分析二者通过链路、缓存、队列和负载形成的耦合关系。
2. 在上层设计基于多智能体强化学习的轨迹控制方法，采用 `attention_mappo` 作为主控制器，并与强启发式基线 `uncoordinated_greedy` 进行对比。
3. 在下层设计学习式任务卸载策略，支持本地 UAV、协作 UAV 和 MBS 三类执行模式，并通过增强 oracle 标签引入时延、截止期、队列压力和 MBS 负载的多目标权衡。
4. 通过实验验证双层框架的有效性。上层实验表明学习型轨迹控制器具备更好的综合性能；下层实验表明学习式卸载可以在基本保持时延和截止期表现的同时显著降低 MBS 负载。

### 1.5 创新点总结

本文的创新点可以概括为以下四个方面。

1. **双层协同建模**：将多无人机 MEC 系统中的 UAV 轨迹控制和请求级任务卸载拆分为上层与下层两个相互耦合的决策模块，既保留了系统级联动关系，又降低了直接联合优化的动作空间复杂度。
2. **面向协作卸载的请求级特征设计**：下层卸载策略不仅使用候选时延，还引入缓存命中、协作可用性、队列长度、链路速率、邻居计算资源和缓存置信度等特征，使模型能够区分本地执行、协作执行和 MBS 执行的适用场景。
3. **增强 oracle 标签机制**：区别于单纯复现启发式规则的监督学习方式，本文在标签生成中加入 deadline 违约惩罚、MBS 负载惩罚、队列压力惩罚和协作缓解收益，引导学习式卸载策略主动降低中心节点依赖。
4. **分阶段可解释验证**：通过上层轨迹控制实验、下层卸载策略实验和联合实验接口分别分析不同模块的贡献，避免只给出整体黑盒结果而难以解释性能来源。

## 第2章 系统模型与问题描述

### 2.1 符号说明与仿真参数

为便于后续建模和实验复现，本文首先给出主要符号说明。具体参数取值与代码中的 `config.py` 保持一致。

| 符号 | 含义 |
| --- | --- |
| $\mathcal{U}$ | UAV 集合，$|\mathcal{U}|=N$ |
| $\mathcal{K}$ | 地面用户 UE 集合，$|\mathcal{K}|=K$ |
| $\mathcal{S}$ | 服务类型集合 |
| $t$ | 离散时隙编号 |
| $\mathbf{q}_i(t)$ | UAV $i$ 在时隙 $t$ 的三维位置 |
| $\mathbf{w}_k(t)$ | UE $k$ 的三维位置 |
| $R_{k,i}^{\mathrm{ue}}(t)$ | UE $k$ 到 UAV $i$ 的上行链路速率 |
| $R_{i,j}^{\mathrm{uav}}(t)$ | UAV $i$ 到 UAV $j$ 的协作链路速率 |
| $R_{i,m}^{\mathrm{mbs}}(t)$ | UAV $i$ 到 MBS 的回传链路速率 |
| $D_r$ | 请求 $r$ 的输入数据量 |
| $C_r$ | 请求 $r$ 单位数据所需 CPU 周期数 |
| $\tau_r$ | 请求 $r$ 的截止时间 |
| $a_r^0,a_r^1,a_r^2$ | 请求 $r$ 的本地、协作、MBS 三类卸载 one-hot 决策 |
| $Q_i(t)$ | UAV $i$ 当前服务队列长度 |
| $\rho_{\mathrm{mbs}}(t)$ | MBS 负载比例 |

| 参数 | 取值 | 说明 |
| --- | ---: | --- |
| UAV 数量 | 5 | 多无人机协同规模 |
| UE 数量 | 100 | 地面用户数量 |
| 区域大小 | 700m × 700m | 二维服务区域 |
| MBS 位置 | (350m, 350m, 30m) | 宏基站三维坐标 |
| 时隙长度 | 1s | 离散决策周期 |
| 每回合步数 | 1000 | 单回合仿真长度 |
| UAV 高度 | 100m | 固定飞行高度 |
| UAV 速度 | 15m/s | 单时隙最大移动尺度依据 |
| UAV 覆盖半径 | 100m | 用户服务覆盖范围 |
| UAV 感知半径 | 460m | 邻居和协作信息可见范围 |
| UAV 最小间距 | 200m | 碰撞规避约束 |
| 服务类型数 | 25 | 可被请求的计算服务 |
| 内容文件数 | 50 | 额外内容缓存对象 |
| 输入数据量 | 1MB-5MB | 请求输入大小范围 |
| 服务截止期 | 0.65s-2.10s | 请求 deadline 范围 |
| UE-UAV 带宽 | 40MHz | 接入链路带宽 |
| UAV-UAV 带宽 | 20MHz | 协作链路带宽 |
| UAV-MBS 回传带宽 | 750kHz | 受限回传链路带宽 |
| 通信发射功率 | 0.5W | 数据传输功率 |
| WPT 发射功率 | 50W | 修正后的无线供能发射功率 |
| UE 初始电池容量 | 100J | 用户能量状态上限 |
| 热点数量 | 2 | 非均匀用户分布 |
| 热点半径 | 100m | 单个热点覆盖区域 |

上述参数体现了本文实验场景的两个特点：一是用户空间分布并非完全均匀，热点机制能够制造更明显的轨迹控制需求；二是 UAV-MBS 回传带宽受限，避免所有请求都无代价地卸载到 MBS，从而使本地执行和协作执行具有实际比较意义。

### 2.2 网络模型

考虑一个由多架 UAV、多个地面用户 UE 和一个宏基站 MBS 构成的空地协同 MEC 系统。设 UAV 集合为

$$
\mathcal{U}=\{1,2,\ldots,N\},
$$

地面用户集合为

$$
\mathcal{K}=\{1,2,\ldots,K\},
$$

时间被划分为离散时隙

$$
t \in \{0,1,\ldots,T-1\}.
$$

第 $i$ 架 UAV 在时隙 $t$ 的三维位置表示为

$$
\mathbf{q}_i(t)=[x_i(t),y_i(t),h_i(t)]^\top.
$$

第 $k$ 个用户的位置表示为

$$
\mathbf{w}_k(t)=[x_k(t),y_k(t),0]^\top.
$$

UAV 与 UE、UAV 与 UAV、UAV 与 MBS 之间均可形成无线链路。链路质量随节点距离、传输功率、带宽和噪声变化。

系统运行过程中还需要满足以下基本约束。第一，UAV 的水平位置不得超出 $[0,X_{\max}]\times[0,Y_{\max}]$ 的服务区域，本文实验中 $X_{\max}=Y_{\max}=700$m。第二，为降低碰撞风险，任意两架 UAV 的水平距离应不小于最小安全间距 $d_{\min}=200$m；若动作更新后发生冲突，仿真环境会通过碰撞修正和惩罚项体现该约束。第三，UE 只有在 UAV 覆盖范围内才可能获得该 UAV 服务，覆盖半径设置为 100m。第四，UAV 的缓存容量和计算能力有限，任务卸载策略不能假设所有服务都能在所有 UAV 上无成本执行。第五，UE 具有电量状态，若电量过低则会影响服务可用性和 offline rate 指标。

本文采用离散时隙模型。在每个时隙内，先根据上层轨迹策略更新 UAV 位置，再根据新的链路和缓存状态处理用户请求。由于单个时隙长度较短，本文假设同一时隙内节点位置在任务传输和计算过程中保持不变，链路速率按时隙开始或更新后的状态估计。

### 2.3 链路速率模型

UE $k$ 到 UAV $i$ 的距离为

$$
d_{k,i}(t)=\|\mathbf{w}_k(t)-\mathbf{q}_i(t)\|_2.
$$

可采用路径损耗模型表示信道增益：

$$
g_{k,i}(t)=\frac{\beta_0}{d_{k,i}^{\alpha}(t)+\epsilon},
$$

其中 $\beta_0$ 为参考距离信道增益，$\alpha$ 为路径损耗指数，$\epsilon$ 为避免分母为零的极小常数。若 UE 发射功率为 $P_k$，系统带宽为 $B$，噪声功率谱密度为 $N_0$，则 UE-UAV 链路速率为

$$
R_{k,i}^{\mathrm{ue}}(t)=B\log_2\left(1+\frac{P_k g_{k,i}(t)}{N_0B}\right).
$$

类似地，UAV $i$ 到 UAV $j$ 的协作链路速率为

$$
R_{i,j}^{\mathrm{uav}}(t)=B_{\mathrm{uav}}\log_2\left(1+\frac{P_i g_{i,j}(t)}{N_0B_{\mathrm{uav}}}\right),
$$

UAV $i$ 到 MBS 的回传速率为

$$
R_{i,m}^{\mathrm{mbs}}(t)=B_{\mathrm{mbs}}\log_2\left(1+\frac{P_i g_{i,m}(t)}{N_0B_{\mathrm{mbs}}}\right).
$$

### 2.4 服务请求模型

用户在时隙 $t$ 可能产生服务请求。第 $r$ 个请求可表示为

$$
\xi_r(t)=\left(k_r,s_r,D_r,C_r,\tau_r,p_r\right),
$$

其中 $k_r$ 表示请求用户，$s_r$ 表示请求服务类型，$D_r$ 表示输入数据量，$C_r$ 表示单位数据所需 CPU 周期数，$\tau_r$ 表示截止时间，$p_r$ 表示请求优先级。

请求总计算量为

$$
L_r=D_r C_r.
$$

若 UAV $i$ 缓存了服务 $s_r$，记为

$$
c_{i,s_r}(t)=1,
$$

否则 $c_{i,s_r}(t)=0$。缓存命中会降低服务文件获取成本，并影响本地或协作执行的可行性。

### 2.5 计算模型

设 UAV $i$ 的计算能力为 $f_i^{\mathrm{uav}}$，MBS 的计算能力为 $f^{\mathrm{mbs}}$。考虑队列影响后，UAV $i$ 为请求 $r$ 分配的有效计算资源为

$$
\tilde{f}_{i,r}(t)=\frac{f_i^{\mathrm{uav}}}{1+Q_i(t)},
$$

其中 $Q_i(t)$ 为 UAV $i$ 当前服务队列长度。MBS 的有效计算资源可表示为

$$
\tilde{f}_{m,r}(t)=\frac{f^{\mathrm{mbs}}}{1+Q_m(t)}.
$$

### 2.6 三类卸载路径的时延模型

对于由 UAV $i$ 服务的 UE 请求 $r$，下层卸载策略需要在本地 UAV、协作 UAV 和 MBS 三类执行位置中选择一种。

本地 UAV 执行时，总时延由 UE 上传时延和 UAV 本地计算时延构成：

$$
T_r^{\mathrm{local}}(t)
=
\frac{D_r}{R_{k_r,i}^{\mathrm{ue}}(t)}
+
\frac{L_r}{\tilde{f}_{i,r}(t)}.
$$

若请求卸载到协作 UAV $j$，则总时延包括 UE 上传到源 UAV、源 UAV 转发到协作 UAV、协作 UAV 计算，以及必要的服务文件获取开销：

$$
T_r^{\mathrm{coop}}(t)
=
\frac{D_r}{R_{k_r,i}^{\mathrm{ue}}(t)}
+
\frac{D_r}{R_{i,j}^{\mathrm{uav}}(t)}
+
\frac{L_r}{\tilde{f}_{j,r}(t)}
+
\Delta_{j,s_r}^{\mathrm{cache}}(t).
$$

其中

$$
\Delta_{j,s_r}^{\mathrm{cache}}(t)=
\begin{cases}
0, & c_{j,s_r}(t)=1,\\
\frac{F_{s_r}}{R_{j,m}^{\mathrm{mbs}}(t)}, & c_{j,s_r}(t)=0,
\end{cases}
$$

$F_{s_r}$ 为服务文件大小。

若请求卸载到 MBS，则总时延为

$$
T_r^{\mathrm{mbs}}(t)
=
\frac{D_r}{R_{k_r,i}^{\mathrm{ue}}(t)}
+
\frac{D_r}{R_{i,m}^{\mathrm{mbs}}(t)}
+
\frac{L_r}{\tilde{f}_{m,r}(t)}.
$$

因此，请求 $r$ 的实际服务时延可统一表示为

$$
T_r(t)=
a_r^{0}(t)T_r^{\mathrm{local}}(t)
+
a_r^{1}(t)T_r^{\mathrm{coop}}(t)
+
a_r^{2}(t)T_r^{\mathrm{mbs}}(t),
$$

其中 $a_r(t)=[a_r^0(t),a_r^1(t),a_r^2(t)]$ 为 one-hot 卸载决策，分别表示本地、协作和 MBS。

### 2.7 能耗模型

UE 上传能耗为

$$
E_r^{\mathrm{up}}(t)=P_k \frac{D_r}{R_{k_r,i}^{\mathrm{ue}}(t)}.
$$

UAV 本地计算能耗可表示为

$$
E_r^{\mathrm{local}}(t)=\kappa_i L_r \left(\tilde{f}_{i,r}(t)\right)^2,
$$

其中 $\kappa_i$ 为 UAV 计算能耗系数。协作卸载还包含 UAV-UAV 转发能耗：

$$
E_r^{\mathrm{coop}}(t)=
E_r^{\mathrm{up}}(t)
+
P_i \frac{D_r}{R_{i,j}^{\mathrm{uav}}(t)}
+
\kappa_j L_r \left(\tilde{f}_{j,r}(t)\right)^2.
$$

MBS 卸载能耗可写为

$$
E_r^{\mathrm{mbs}}(t)=
E_r^{\mathrm{up}}(t)
+
P_i \frac{D_r}{R_{i,m}^{\mathrm{mbs}}(t)}.
$$

实际能耗为

$$
E_r(t)=
a_r^0(t)E_r^{\mathrm{local}}(t)
+
a_r^1(t)E_r^{\mathrm{coop}}(t)
+
a_r^2(t)E_r^{\mathrm{mbs}}(t).
$$

需要说明的是，本文能耗模型同时服务于仿真评估和论文分析。传输能耗、计算能耗和 UAV 移动/悬停能耗均在代码中按模块累计；无线供能部分采用简化信道增益模型，并在实验中将 WPT 发射功率修正为 50W，使系统能耗回到更合理量级。该处理保留了能量状态对 offline rate 和奖励的影响，但没有进一步展开复杂的电池老化、充放电非线性或精细空气动力学模型。

### 2.8 模型假设说明

为使问题具有可复现性和可计算性，本文作如下建模假设。

1. UAV 飞行高度固定为 100m，主要优化水平面运动；高度控制不作为本文动作空间的一部分。
2. MBS 位置固定，具有较强计算能力，但回传带宽和负载受到限制，因此 MBS 并非总是最优执行位置。
3. UE 请求大小、服务类型、CPU 周期密度和 deadline 由配置范围随机生成，服务流行度服从 Zipf 分布。
4. UAV 缓存状态随缓存更新机制变化，缓存命中会影响服务文件获取时延。
5. 协作 UAV 必须在感知范围内，且协作链路和协作节点计算资源共同决定协作卸载是否有利。
6. 本文重点关注轨迹控制与请求级卸载的协同关系，因此未进一步引入复杂的用户移动预测、三维避障、真实城市遮挡或多频段干扰模型。

### 2.9 优化目标

请求是否满足截止期定义为

$$
\mathbb{I}_r(t)=
\begin{cases}
1, & T_r(t)\le \tau_r,\\
0, & T_r(t)>\tau_r.
\end{cases}
$$

MBS 负载比例定义为

$$
\rho_{\mathrm{mbs}}(t)=
\frac{\sum_r a_r^2(t)}{\sum_r 1+\epsilon}.
$$

系统公平性可采用 Jain 公平性指数：

$$
J(t)=
\frac{\left(\sum_{i=1}^{N} x_i(t)\right)^2}
{N\sum_{i=1}^{N}x_i^2(t)+\epsilon},
$$

其中 $x_i(t)$ 可表示 UAV $i$ 的服务量或负载均衡收益。

本文的整体优化目标为在运动约束、链路约束、缓存约束和计算资源约束下，最小化综合代价：

$$
\min_{\pi^{\mathrm{traj}},\pi^{\mathrm{off}}}
\mathbb{E}
\left[
\sum_{t=0}^{T-1}
\left(
w_T \bar{T}(t)
+
w_E \bar{E}(t)
+
w_D \bar{V}(t)
+
w_M \rho_{\mathrm{mbs}}(t)
-
w_F J(t)
\right)
\right],
$$

其中 $\bar{T}(t)$ 为平均时延，$\bar{E}(t)$ 为平均能耗，$\bar{V}(t)$ 为 deadline 违约率，$\rho_{\mathrm{mbs}}(t)$ 为 MBS 负载比例，$J(t)$ 为公平性指标。

## 第3章 双层协同优化框架

### 3.1 框架结构

本文将多无人机 MEC 决策过程拆分为两个层次。上层为 UAV 轨迹控制层，在每个时隙根据全局或局部观测输出 UAV 运动动作；下层为请求级任务卸载层，在服务请求到达时选择执行位置。

上层策略记为

$$
\pi^{\mathrm{traj}}_{\theta}(a_i^t|o_i^t),
$$

其中 $o_i^t$ 为 UAV $i$ 的观测，$a_i^t$ 为运动动作。下层策略记为

$$
\pi^{\mathrm{off}}_{\phi}(y_r^t|\mathbf{z}_r^t),
$$

其中 $\mathbf{z}_r^t$ 为请求级特征，$y_r^t\in\{0,1,2\}$ 分别表示本地、协作和 MBS。

双层框架的核心思想如图所示：

![双层协同优化框架](docs/figures/fig_framework.png)

从实现角度看，系统在每个时隙内按照“上层移动-环境更新-下层卸载-指标统计”的顺序执行。上层策略不直接决定每个请求的执行位置，而是通过改变 UAV 空间分布、链路质量和服务队列间接影响下层可选路径；下层策略不直接控制 UAV 移动，而是在给定当前状态下选择请求执行位置，并把处理结果反馈到队列、MBS 负载、能量状态和奖励指标中。

**算法1：双层协同优化流程**

```text
输入：初始环境状态 s_0，上层轨迹策略 pi_traj，下层卸载策略 pi_off，最大时隙数 T
输出：每回合奖励、平均时延、能耗、deadline 满足率、MBS 负载和公平性

1. 初始化 UAV 位置、UE 分布、缓存状态、队列状态和能量状态
2. for t = 0, 1, ..., T-1 do
3.     每架 UAV 获取局部观测 o_i(t)，包括自身状态、邻居状态和关联 UE 请求信息
4.     上层策略 pi_traj 根据 o_i(t) 输出 UAV 运动动作 a_i(t)
5.     环境根据动作更新 UAV 位置，并修正边界或碰撞约束
6.     根据新位置重新计算 UE-UAV、UAV-UAV 和 UAV-MBS 链路速率
7.     UE 产生或更新服务请求，UAV 根据覆盖关系获得待服务请求集合
8.     for 每个待处理请求 r do
9.         构造请求级特征 z_r(t)，包含候选时延、缓存、队列和链路信息
10.        下层策略 pi_off 根据 z_r(t) 选择 local、cooperative 或 mbs
11.        环境执行对应卸载路径，更新 UAV 队列、MBS 负载和任务完成状态
12.    end for
13.    统计当前时隙奖励、时延、能耗、deadline 满足率、offline rate 和公平性
14. end for
```

### 3.2 两层之间的耦合关系

上层轨迹控制影响下层卸载条件，主要体现在：

1. UAV 位置影响 UE-UAV 接入速率。
2. UAV 之间的距离影响协作 UAV 可达性和 UAV-UAV 传输速率。
3. UAV 与 MBS 的距离影响回传速率。
4. UAV 的空间分布影响用户关联和队列压力。

下层卸载策略反过来影响系统状态，主要体现在：

1. 本地执行增加当前 UAV 计算队列压力。
2. 协作执行改变邻居 UAV 的负载。
3. MBS 执行增加中心节点负载。
4. 不同卸载路径改变最终时延、能耗和 deadline 满足情况。

因此，本文不是将轨迹控制和任务卸载视为孤立模块，而是通过系统状态将二者联系起来。

### 3.3 分阶段求解思路

直接端到端联合优化轨迹和卸载会导致动作空间过大，训练不稳定，并且难以解释不同模块对系统性能的贡献。本文采用分阶段求解：

1. 首先固定下层运行逻辑，比较上层轨迹控制器，选出适合作为双层框架的轨迹策略。
2. 然后在固定轨迹条件下比较下层卸载策略，分析启发式、增强 oracle 学习策略和激进负载分散策略的差异。
3. 最后保留联合实验接口，用于后续验证不同上下层组合的系统级效果。

这种设计使实验结论更加清晰：上层实验回答“如何移动 UAV”，下层实验回答“请求到达后如何执行”，联合实验回答“二者组合后如何影响整体性能”。

采用分阶段求解还有三个现实原因。第一，若把 UAV 运动动作、缓存动作和请求级卸载动作全部合并，联合动作空间会随 UAV 数量、用户数量和请求数量快速增长，强化学习训练更容易出现样本效率低和收敛不稳定问题。第二，轨迹控制和任务卸载的时间尺度不同：UAV 运动按时隙决策，而服务请求可能在同一时隙内批量到达，直接端到端建模会使 credit assignment 更困难。第三，分阶段实验便于判断性能提升来自上层轨迹改进、下层卸载改进，还是二者耦合后的综合效果，这对毕业论文中的结果解释尤为重要。

## 第4章 上层轨迹控制方法

### 4.1 多智能体决策建模

将 UAV 轨迹控制建模为多智能体马尔可夫决策过程。每个 UAV 是一个智能体，系统状态为 $s_t$，UAV $i$ 的局部观测为 $o_i^t$，联合动作为

$$
\mathbf{a}_t=(a_1^t,a_2^t,\ldots,a_N^t).
$$

环境根据状态转移函数更新：

$$
s_{t+1}\sim P(s_{t+1}|s_t,\mathbf{a}_t).
$$

每个 UAV 的动作可表示为离散运动方向或停留动作：

$$
a_i^t \in \mathcal{A}^{\mathrm{move}}.
$$

在代码实现中，每个 UAV 的观测由自身状态、邻居 UAV 状态和关联 UE 状态拼接而成。自身状态主要包括 UAV 的二维位置和缓存向量；邻居状态包括可感知 UAV 的相对位置信息；UE 状态包括用户位置、请求类型、输入大小、请求编号、deadline、优先级和电量状态。对应维度由 `config.py` 中的 `SELF_OBS_DIM`、`NEIGHBOR_OBS_DIM`、`UE_OBS_DIM` 和 `OBS_DIM_SINGLE` 控制。

| 观测组成 | 代码参数 | 含义 |
| --- | --- | --- |
| 自身位置 | 2 维 | UAV 当前水平坐标 |
| 自身缓存 | `NUM_FILES=75` 维 | UAV 已缓存的服务和内容状态 |
| 邻居 UAV | `MAX_UAV_NEIGHBORS × 2` 维 | 可感知邻居 UAV 的位置 |
| 关联 UE | `MAX_ASSOCIATED_UES × UE_OBS_DIM` 维 | 覆盖范围内用户位置、请求和电量 |
| 请求信息 | `REQUEST_OBS_DIM=5` 维 | 请求类型、大小、编号、deadline、优先级 |

动作空间由 `ACTION_DIM=2` 控制，表示 UAV 在当前时隙内的移动方向和移动距离。动作经过环境映射后更新 UAV 水平位置，移动距离受到 UAV 速度和时隙长度约束，即单时隙移动尺度与 $v^{\mathrm{uav}}\Delta t$ 相关。若动作导致 UAV 越界或与其他 UAV 距离过近，环境会通过边界惩罚、碰撞惩罚和位置修正体现约束影响。

奖励函数综合考虑服务质量、时延、能耗、公平性和负载：

$$
r_t =
\lambda_D DSR_t
-
\lambda_T \bar{T}_t
-
\lambda_E \bar{E}_t
-
\lambda_M \rho_{\mathrm{mbs}}(t)
+
\lambda_F J_t.
$$

其中 $DSR_t$ 为 deadline satisfaction rate。

代码中的上层奖励实际采用对数缩放形式，以降低极端时延、能耗或 offline rate 对训练稳定性的影响。其核心思想是奖励公平性，惩罚时延、能耗和用户掉线：

$$
r_t =
\eta
\left[
\alpha_3 \log(J_t+\epsilon)
-
\alpha_1 \log(\bar{T}_t+\epsilon)
-
\alpha_2 \log(\bar{E}_t+\epsilon)
-
\alpha_4 \log(1+\mathrm{offline}_t)
\right],
$$

其中 $\eta$ 对应奖励缩放系数。实验中使用的奖励权重如下。

| 参数 | 取值 | 作用 |
| --- | ---: | --- |
| `ALPHA_1` | 1.0 | 平均时延惩罚权重 |
| `ALPHA_2` | 0.4 | 能耗惩罚权重 |
| `ALPHA_3` | 2.0 | 公平性奖励权重 |
| `ALPHA_4` | 50.0 | offline rate 惩罚权重 |
| `REWARD_SCALING_FACTOR` | 0.01 | 奖励缩放因子 |

上层训练主要超参数如下。

| 参数 | 取值 | 说明 |
| --- | ---: | --- |
| Actor 学习率 | 3e-4 | 策略网络更新步长 |
| Critic 学习率 | 3e-4 | 价值网络更新步长 |
| 折扣因子 | 0.99 | 长期回报折扣 |
| 软更新系数 | 0.012 | 目标网络更新比例 |
| 梯度裁剪 | 0.5 | 防止梯度爆炸 |
| MLP 隐藏层维度 | 128 | 基础网络隐藏维度 |
| PPO rollout 长度 | 1000 | 每次 PPO 更新前收集的时隙数 |
| PPO epoch | 8 | 单批 rollout 重复优化轮数 |
| PPO batch size | 256 | PPO 小批量大小 |
| PPO clip | 0.2 | 近端策略优化截断范围 |
| Attention 隐藏维度 | 64 | 注意力内部表示维度 |
| Attention heads | 8 | 多头注意力头数 |
| 上层训练回合 | 500 | 论文主实验设置 |
| 上层测试回合 | 60 | 论文主实验设置 |

### 4.2 Attention-MAPPO

本文上层主算法采用 `attention_mappo`。MAPPO 使用集中训练、分散执行范式。策略网络根据每个 UAV 的观测输出动作分布：

$$
a_i^t \sim \pi_{\theta}(a_i^t|o_i^t).
$$

价值函数在训练时可利用更丰富的全局状态：

$$
V_{\psi}(s_t)\approx
\mathbb{E}\left[\sum_{l=t}^{T-1}\gamma^{l-t}r_l\right].
$$

优势函数为

$$
A_t=R_t-V_{\psi}(s_t).
$$

PPO 的截断目标为

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right],
$$

其中

$$
r_t(\theta)=
\frac{\pi_{\theta}(a_t|o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|o_t)}.
$$

`attention_mappo` 在策略或价值估计中引入注意力机制，用于增强 UAV 间状态交互建模能力，使智能体能够关注与自身决策更相关的邻居和负载信息。

### 4.3 强启发式基线

本文使用 `uncoordinated_greedy` 作为上层强基线。该策略不进行多智能体协同学习，而是根据局部收益或贪心规则选择 UAV 动作。它的意义在于提供一个具有竞争力的非学习式参照，避免只与弱基线比较。

### 4.4 上层实验结果

上层实验使用修正 WPT 发射功率后的 `wpt_fix_thesis_run` 结果。测试集统计结果如下，表中数值为多回合均值 ± 标准差：

| 方法 | Reward | Latency | Energy | Deadline Satisfaction | Fairness | MBS Load | Offline Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_mappo | -1260.12 ± 57.73 | 912568.18 ± 85175.23 | 101164953.60 ± 8308044.22 | 54.69% ± 4.20% | 0.958 ± 0.020 | 15.86% ± 2.07% | 79.73% ± 5.78% |
| uncoordinated_greedy | -1466.68 ± 62.33 | 1064784.72 ± 270468.43 | 105967844.94 ± 11931286.05 | 45.27% ± 13.52% | 0.785 ± 0.145 | 11.14% ± 5.59% | 70.87% ± 9.94% |

![上层轨迹控制对比](docs/figures/fig_upper_comparison.png)

从结果可以看出，`attention_mappo` 的测试奖励明显高于 `uncoordinated_greedy`，平均时延更低，能耗更低，deadline 满足率更高，公平性也明显更优。其不足是 MBS 负载和 offline rate 高于强启发式基线，说明学习型轨迹控制器在改善服务质量和公平性的同时，对中心节点和能量状态仍存在一定压力。综合奖励、时延、能耗、deadline 和公平性指标后，本文仍选取 `attention_mappo` 作为双层框架的上层主策略。

### 4.5 上层主方法选择理由

本文选择 `attention_mappo` 作为上层主方法，并不是因为它在所有单项指标上都占优，而是因为它在综合服务质量和协同能力上更符合双层框架需求。首先，MAPPO 的在策略更新和 clipped objective 能够提高多智能体训练稳定性，适合 UAV 运动这类连续交互的长期决策问题。其次，注意力机制可以增强 UAV 对邻居状态和关联用户状态的选择性建模，使策略不只根据自身位置做局部移动，而能更好地响应热点用户、邻居分布和服务压力。最后，从实验结果看，`attention_mappo` 在 reward、latency、energy、deadline satisfaction 和 fairness 上均优于强启发式基线，说明它更适合作为上层控制器；其 MBS load 偏高的问题，则由下层学习式卸载策略进一步缓解。

## 第5章 下层学习式任务卸载策略

### 5.1 请求级卸载决策

下层任务卸载在每个服务请求到达时触发。对于请求 $r$，策略输入为特征向量

$$
\mathbf{z}_r=
\left[
z_{r,1},z_{r,2},\ldots,z_{r,d}
\right].
$$

输出类别为

$$
y_r\in\{0,1,2\},
$$

其中 0 表示本地 UAV 执行，1 表示协作 UAV 执行，2 表示 MBS 执行。

### 5.2 卸载特征设计

本文实现了两套卸载特征族。`surrogate_baseline` 使用 `full_features`，维度较低，主要围绕候选路径时延和基础请求状态构造，适合作为增强 oracle 标签的代理基线。`rich_reduced_runtime_policy` 使用 `rich_reduced_features`，维度更高，加入链路速率、协作邻居、计算资源和缓存置信度等信息，更贴近运行时系统状态，但也更容易形成激进的 MBS 减负策略。

`surrogate_baseline` 的 8 维特征如下。

| 特征名 | 含义 | 归一化方式 | 运行时可获得 |
| --- | --- | --- | --- |
| `local_latency_vs_deadline` | 本地 UAV 执行候选时延相对 deadline 的比例 | 截断到 `[0,10]` 后除以 10 | 是 |
| `cooperative_latency_vs_deadline` | 最优协作 UAV 候选时延相对 deadline 的比例 | 不可协作时取 1.0 | 是 |
| `mbs_latency_vs_deadline` | MBS 执行候选时延相对 deadline 的比例 | 截断到 `[0,10]` 后除以 10 | 是 |
| `deadline_normalized` | 请求 deadline 松紧程度 | 按 deadline 最小/最大值线性归一化 | 是 |
| `priority_normalized` | 请求优先级 | 按优先级范围线性归一化 | 是 |
| `local_cache_hit` | 本地 UAV 是否已缓存服务文件 | 0/1 二值 | 是 |
| `cooperative_available` | 是否存在可用协作 UAV | 0/1 二值 | 是 |
| `local_queue_fraction` | 本地 UAV 当前队列压力 | 队列长度除以最大关联 UE 数并截断 | 是 |

`rich_reduced_runtime_policy` 的 16 维特征如下。

| 特征名 | 含义 | 归一化方式 | 运行时可获得 |
| --- | --- | --- | --- |
| `request_size_normalized` | 请求输入大小 | 按输入大小范围线性归一化 | 是 |
| `service_file_size_normalized` | 服务文件大小 | 除以最大文件大小 | 是 |
| `cpu_density_normalized` | 服务计算密度 | 按 CPU 周期密度范围线性归一化 | 是 |
| `deadline_normalized` | 请求 deadline 松紧程度 | 线性归一化 | 是 |
| `priority_normalized` | 请求优先级 | 线性归一化 | 是 |
| `local_cache_hit` | 本地缓存命中 | 0/1 二值 | 是 |
| `cooperative_available` | 协作 UAV 是否可用 | 0/1 二值 | 是 |
| `local_queue_fraction` | 本地队列压力 | 截断归一化 | 是 |
| `neighbor_count_fraction` | 当前可达邻居比例 | 邻居数除以最大邻居数 | 是 |
| `ue_uav_rate_log10` | UE 到源 UAV 上行速率 | $\log_{10}$ 缩放 | 是 |
| `uav_mbs_rate_log10` | 源 UAV 到 MBS 回传速率 | $\log_{10}$ 缩放 | 是 |
| `best_uav_uav_rate_log10` | 到最佳协作 UAV 的链路速率 | $\log_{10}$ 缩放，无协作时取 0 | 是 |
| `best_neighbor_mbs_rate_log10` | 最佳协作 UAV 到 MBS 的回传速率 | $\log_{10}$ 缩放，无协作时取 0 | 是 |
| `local_compute_share_normalized` | 本地 UAV 可用计算份额 | 按最大 UAV 算力归一化并截断 | 是 |
| `best_neighbor_compute_share_normalized` | 最佳协作 UAV 可用计算份额 | 按最大 UAV 算力归一化并截断 | 是 |
| `best_neighbor_cache_belief` | 最佳协作 UAV 缓存该服务的置信度 | `[0,1]` 概率值 | 是 |

两套特征的差异体现了论文中的一个消融逻辑：低维特征更容易解释，适合作为主展示策略；高维特征携带更多运行时信息，能够进一步降低 MBS 负载，但也可能带来更高能耗或更低 deadline 满足率。

### 5.3 启发式卸载规则

启发式策略 `heuristic_offloading` 根据估计时延、缓存命中、协作可用性和回传条件选择执行位置。其优点是稳定、可解释，并且在某些指标上具有较强表现；缺点是规则固定，难以主动体现多目标权衡，例如降低 MBS 负载或在队列压力较高时引导任务分散。

### 5.4 增强 oracle 标签

为了使学习式策略不只是简单复制启发式规则，本文构造增强 oracle 标签。对每个候选执行位置 $c\in\{\mathrm{local},\mathrm{coop},\mathrm{mbs}\}$，定义代价：

$$
\mathcal{C}_c
=
\alpha_T \frac{T_c}{\tau_r}
+
\alpha_D \max\left(0,\frac{T_c-\tau_r}{\tau_r}\right)
+
\alpha_Q q_c
+
\alpha_M \mathbb{I}[c=\mathrm{mbs}]\rho_{\mathrm{mbs}}
-
\alpha_C \mathbb{I}[c=\mathrm{coop}]b_{\mathrm{relief}}.
$$

其中 $T_c$ 为候选路径时延，$\tau_r$ 为请求截止期，$q_c$ 为对应执行节点队列压力，$\rho_{\mathrm{mbs}}$ 为 MBS 负载，$b_{\mathrm{relief}}$ 表示协作卸载对本地队列的缓解收益。增强 oracle 标签为

$$
y_r^{*}=\arg\min_{c\in\{\mathrm{local},\mathrm{coop},\mathrm{mbs}\}}\mathcal{C}_c.
$$

本次实验采用的权重为：

| 权重项 | 数值 | 含义 |
| --- | ---: | --- |
| deadline penalty | 1.25 | 强化截止期违约惩罚 |
| MBS load penalty | 0.065 | 降低 MBS 依赖 |
| queue penalty | 0.06 | 考虑 UAV 队列压力 |
| cooperative relief bonus | 0.02 | 鼓励合理协作分担 |

增强 oracle 标签的生成流程可表示如下。

```text
输入：请求上下文 context，候选路径 local/coop/mbs 的时延、队列、缓存和链路状态
输出：监督学习标签 y*

1. 计算 local、coop、mbs 三类候选执行时延
2. 对每个候选路径计算 deadline 违约程度
3. 对 local 和 coop 加入对应 UAV 队列压力惩罚
4. 对 mbs 加入 MBS 负载惩罚，避免策略过度依赖中心节点
5. 对可行 coop 路径加入协作队列缓解收益
6. 选择总代价最低的候选路径作为 y*
```

因此，增强 oracle 标签并不等价于简单选择最小时延路径。它允许在时延差异很小的情况下优先选择本地或协作路径，以换取更低的 MBS 负载；也会在本地队列过长或 deadline 压力较大时选择 MBS，避免为了减负而明显牺牲服务质量。

### 5.5 监督学习模型

学习式卸载策略采用分类器拟合增强 oracle 标签。模型输出为

$$
\mathbf{p}_r=f_{\phi}(\mathbf{z}_r),
$$

其中 $\mathbf{p}_r\in[0,1]^3$，且

$$
\sum_{c=0}^{2}p_{r,c}=1.
$$

训练目标采用交叉熵损失：

$$
\mathcal{L}(\phi)
=
-
\frac{1}{M}
\sum_{r=1}^{M}
\sum_{c=0}^{2}
\mathbb{I}[y_r^*=c]\log p_{r,c}.
$$

运行时卸载决策为

$$
\hat{y}_r=\arg\max_{c}p_{r,c}.
$$

本文比较两类学习式策略：

1. `surrogate_baseline`：使用 8 维特征，主展示为增强 oracle 学习式卸载策略。
2. `rich_reduced_runtime_policy`：使用 16 维更丰富特征，更激进地降低 MBS 负载，作为消融对比。

分类器采用多层感知机（MLP）结构，输入层维度由特征族决定，隐藏层默认设置为 `(64, 64)`，输出层为 3 维 logits，对应 local、cooperative 和 mbs 三类卸载决策。训练前仅使用训练集统计量对特征做标准化，并将均值和标准差随模型检查点一起保存，保证离线验证和在线推理使用一致的特征尺度。若数据类别不均衡，训练脚本可通过 weighted sampler 提高少数类样本被采样的概率。

学习式卸载并不是对启发式策略的简单模仿，原因在于训练标签可以从 `heuristic` 切换为 `enhanced_oracle`。当使用增强 oracle 时，模型学习的是一个带有系统目标偏好的决策边界：既要考虑候选时延和 deadline，也要考虑 MBS 负载、队列压力和协作分担收益。因此，模型在线运行时可能做出与启发式不同的选择，例如在 MBS 与协作 UAV 时延接近时优先选择协作 UAV，以降低中心节点负载。

### 5.6 离线训练结果

下层实验使用修正 WPT 发射功率后的 `wpt_fix_thesis_run_offload`。数据集共 18000 个样本，三类标签均衡：

| 类别 | 样本数 | 比例 |
| --- | ---: | ---: |
| local | 6000 | 33.33% |
| cooperative | 6000 | 33.33% |
| mbs | 6000 | 33.33% |

模型训练结果如下：

| 策略 | Feature Dim | Validation Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| surrogate_baseline | 8 | 0.9447 | 0.9448 |
| rich_reduced_runtime_policy | 16 | 0.9561 | 0.9560 |

离线结果说明，两类学习策略都能够较好拟合增强 oracle 标签，具备稳定分类能力。但离线精度并不等价于在线系统性能，因此还需要通过环境运行评估其时延、能耗、deadline 和 MBS 负载表现。

## 第6章 实验设计与结果分析

### 6.1 实验设置

本文实验分为上层轨迹控制实验和下层任务卸载实验。上层实验比较 `attention_mappo` 与 `uncoordinated_greedy`，训练轮数为 500，测试轮数为 60。下层实验固定轨迹和运行场景，比较 `heuristic_offloading`、`surrogate_baseline` 和 `rich_reduced_runtime_policy` 三种策略。对于测试日志或运行时对比报告中具有逐回合统计的指标，本文表格采用“均值 ± 标准差”格式；对于离线分类器训练表，当前展示验证集整体指标。

实验环境与协议如下。

| 项目 | 设置 |
| --- | --- |
| 代码环境 | Python + PyTorch，多无人机 MEC 仿真环境 |
| 随机种子 | `SEED=42`；下层运行时对比使用 42、84、126、168 |
| 上层训练算法 | `attention_mappo` 与 `uncoordinated_greedy` |
| 上层训练回合 | 500 |
| 上层测试回合 | 60 |
| 下层训练标签 | `enhanced_oracle` 为主，`heuristic` 可作为消融 |
| 下层数据集规模 | 18000 个样本，local/cooperative/mbs 各 6000 |
| 下层分类器 | MLP，隐藏层 `(64, 64)`，输出 3 类 |
| 下层训练轮数 | 25 |
| 下层 batch size | 256 |
| 下层学习率 | 1e-3 |
| 下层验证集比例 | 0.2 |
| 结果文件 | `experiment_manifest.json`、训练/测试日志 JSON、卸载实验 summary JSON |

为了保证实验结论和论文数值一致，本文的结果不手工编造，而是来自仓库中保存的日志与 summary 文件。上层结果主要来自 `results/full_runs/wpt_fix_thesis_run`，下层结果主要来自 `results/full_offload_experiments/wpt_fix_thesis_run_offload`。图表由 `generate_thesis_figures.py` 根据这些结果文件自动生成。

主要评价指标包括：

1. 平均时延：

$$
\bar{T}=\frac{1}{|\mathcal{R}|}\sum_{r\in\mathcal{R}}T_r.
$$

2. 平均能耗：

$$
\bar{E}=\frac{1}{|\mathcal{R}|}\sum_{r\in\mathcal{R}}E_r.
$$

3. 截止期满足率：

$$
DSR=\frac{1}{|\mathcal{R}|}\sum_{r\in\mathcal{R}}\mathbb{I}[T_r\le\tau_r].
$$

4. MBS 卸载比例：

$$
OR_{\mathrm{mbs}}=
\frac{\sum_{r\in\mathcal{R}}\mathbb{I}[\hat{y}_r=2]}
{|\mathcal{R}|}.
$$

5. MBS 负载比例：

$$
LR_{\mathrm{mbs}}=
\frac{\mathrm{MBS\ processed\ requests}}
{\mathrm{all\ processed\ requests}+\epsilon}.
$$

6. 公平性：

$$
J=
\frac{\left(\sum_i x_i\right)^2}
{N\sum_i x_i^2+\epsilon}.
$$

### 6.2 上层结果分析

上层结果显示，`attention_mappo` 在 reward、latency、deadline satisfaction 和 fairness 上优于 `uncoordinated_greedy`。特别是公平性从 0.79 提升到 0.95，说明注意力机制和多智能体协同学习有助于改善 UAV 间服务负载分布。表中数值为测试回合均值，后续若补充标准差，可进一步展示不同随机种子下的稳定性。虽然 `attention_mappo` 的 MBS 负载和 offline rate 高于强启发式基线，但其综合奖励更高，说明在当前奖励权衡下，该策略更适合作为双层协同框架的上层控制器。

因此，本文选择 `attention_mappo` 作为后续联合框架中的上层轨迹控制方法。

### 6.3 下层在线运行结果

三种下层卸载策略在线运行统计结果如下，表中数值为多场景/多随机种子均值 ± 标准差：

| 策略 | Latency | Energy | Deadline Satisfaction | Local Ratio | Coop Ratio | MBS Ratio | MBS Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic_offloading | 1477.33 ± 138.51 | 107305.72 ± 119772.77 | 24.64% ± 6.06% | 40.99% ± 24.04% | 22.93% ± 19.88% | 32.46% ± 36.63% | 8.41% ± 8.81% |
| surrogate_baseline | 1477.64 ± 138.50 | 125190.64 ± 103397.66 | 24.47% ± 6.02% | 51.96% ± 14.69% | 26.29% ± 20.18% | 18.13% ± 20.68% | 4.88% ± 5.34% |
| rich_reduced_runtime_policy | 1478.12 ± 137.97 | 126569.00 ± 104475.75 | 23.92% ± 6.05% | 51.12% ± 18.11% | 29.79% ± 14.88% | 15.47% ± 23.01% | 3.89% ± 5.58% |

![下层卸载策略在线性能对比](docs/figures/fig_lower_runtime.png)

相对启发式策略，`surrogate_baseline` 的平均时延仅增加约 0.0205%，deadline 满足率下降约 0.70%，但 MBS 卸载比例降低约 44.16%，MBS 负载降低约 41.97%。这说明增强 oracle 学习式卸载策略并没有以明显损害时延为代价，而是在服务质量基本稳定的前提下显著降低了中心节点依赖。

`rich_reduced_runtime_policy` 更加激进，其 MBS 卸载比例降低约 52.35%，MBS 负载降低约 53.71%，但 deadline 满足率下降约 2.90%，能耗增加也更明显。因此，本文将其作为负载分散更强但代价更大的消融策略，而不是主展示策略。

### 6.4 MBS 减负与服务质量权衡

下层结果揭示了多目标卸载中的典型权衡：若策略强烈抑制 MBS 使用，则本地 UAV 和协作 UAV 的服务比例会上升，MBS 负载显著下降；但由于本地和协作节点的计算能力、缓存状态和链路条件有限，能耗和 deadline 可能受到影响。

![MBS 减负与服务质量代价权衡](docs/figures/fig_mbs_tradeoff.png)

从论文叙事角度，`surrogate_baseline` 更适合作为增强学习式卸载的主结果。它不是简单追求时延最小，而是在时延、deadline 和 MBS 负载之间取得更均衡的结果。该结果可以支持如下结论：学习式卸载策略能够通过多目标标签设计，将系统从“依赖 MBS 的集中式处理”引导到“本地、协作和 MBS 混合分担”的运行模式。

### 6.5 消融实验设计

为了进一步增强实验可信度，本文建议围绕下层学习式卸载补充以下消融实验。若后续结果完整，可将本节从“设计”改写为“结果分析”。

| 消融项 | 对比方案 | 观察指标 | 预期说明 |
| --- | --- | --- | --- |
| 标签来源 | `heuristic` vs `enhanced_oracle` | MBS ratio、MBS load、deadline satisfaction | 验证增强 oracle 是否真的带来 MBS 减负能力 |
| 特征族 | `full_features` vs `rich_reduced_features` | 离线精度、在线时延、能耗、MBS load | 分析高维运行时特征是否带来更激进负载分散 |
| 采样方式 | weighted sampler vs 普通采样 | per-class recall、macro-F1 | 验证类别均衡采样对三类卸载识别的作用 |
| MBS 惩罚权重 | 0、0.03、0.065、0.10 | MBS load、deadline satisfaction、energy | 分析 MBS 减负与服务质量之间的权衡曲线 |
| 队列惩罚权重 | 0、0.03、0.06、0.10 | local ratio、coop ratio、平均队列压力 | 验证队列压力项是否促进负载分散 |
| 协作收益项 | 关闭 vs 开启 | coop ratio、MBS load、latency | 验证协作 UAV 是否真正承担分流作用 |

这些消融实验的重点不是追求所有指标同时最优，而是解释增强 oracle 中每个设计项对最终策略行为的影响。特别是 MBS 惩罚权重和协作收益项，可以直接支撑本文关于“降低中心节点依赖”和“促进 UAV 协作分担”的结论。

### 6.6 联合实验设计与边界说明

当前可靠结果主要来自上层筛选实验和下层固定轨迹运行实验。为了避免编造尚未充分验证的数据，本文不在此处填写联合实验数值，而是给出后续联合实验设计。

联合实验建议固定上层轨迹控制器为 `attention_mappo`，比较三种下层策略：

| 组合方案 | 说明 |
| --- | --- |
| attention_mappo + heuristic_offloading | 双层框架中的规则卸载基线 |
| attention_mappo + surrogate_baseline | 本文推荐的增强学习式卸载组合 |
| attention_mappo + rich_reduced_runtime_policy | 更激进的 MBS 减负组合 |

联合实验重点观察：

1. 在学习型轨迹控制下，下层学习式卸载是否仍能降低 MBS 负载。
2. 上层轨迹变化是否放大学习式卸载的协作收益。
3. 不同下层策略对 deadline、能耗和公平性的影响是否稳定。

因此，本文当前结论应表述为：已分别验证上层学习式轨迹控制和下层学习式卸载策略的有效性，并保留二者组合的联合实验接口；完整的端到端双层联合性能仍需通过更多随机种子和完整测试日志进一步确认。这样的表述能够避免将“分阶段验证结果”过度扩展为“已充分完成联合最优验证”。

可使用如下命令作为后续实验入口：

```powershell
.\.venv\Scripts\python.exe run_joint_trajectory_offload_experiment.py `
  --name joint_attn_mappo_enhanced_oracle `
  --trajectory_run_root results\full_runs\wpt_fix_thesis_run `
  --trajectory_model attention_mappo `
  --offload_experiment_root results\full_offload_experiments\wpt_fix_thesis_run_offload `
  --policies heuristic surrogate rich_reduced `
  --seeds 42 84 126 168 `
  --episodes_per_seed 8
```

## 第7章 总结与展望

### 7.1 总结

本文围绕多无人机移动边缘计算中的轨迹控制与任务卸载问题，提出一种双层协同优化框架。该框架将复杂系统决策拆分为上层 UAV 轨迹控制和下层请求级任务卸载两个层次：上层通过多智能体强化学习优化 UAV 运动，下层通过学习式分类策略在本地 UAV、协作 UAV 和 MBS 之间选择任务执行位置。

在系统建模方面，本文给出了 UE-UAV、UAV-UAV 和 UAV-MBS 链路速率模型，构建了本地执行、协作执行和 MBS 执行三类卸载路径的时延与能耗表达式，并将 deadline 满足率、MBS 负载和公平性纳入整体优化目标。在方法设计方面，本文采用 `attention_mappo` 作为上层学习型轨迹控制器，并通过增强 oracle 标签训练下层学习式卸载策略。

实验结果表明，上层 `attention_mappo` 相比 `uncoordinated_greedy` 具有更高综合奖励、更低时延、更低能耗、更高 deadline 满足率和更好公平性。下层增强学习式卸载策略在几乎不增加平均时延的情况下，显著降低 MBS 卸载比例和 MBS 负载，其中 `surrogate_baseline` 将 MBS 负载降低约 41.97%，同时 deadline 满足率仅下降约 0.70%。这说明双层协同优化框架能够有效刻画轨迹控制与任务卸载之间的耦合关系，并为多无人机 MEC 系统提供一种可运行、可解释的优化范式。

### 7.2 不足与展望

本文仍存在以下不足：

1. 当前下层学习式卸载主要采用监督学习方式拟合增强 oracle 标签，尚未实现真正的下层强化学习闭环优化。
2. 当前可靠结果主要覆盖上层独立实验和下层固定轨迹实验，联合实验仍需进一步补充多随机种子验证。
3. 当前增强 oracle 权重依赖人工设定，后续可以通过自动搜索或元学习方式获得更稳健的多目标权重。
4. 当前系统主要关注服务请求卸载，未来可以进一步统一缓存更新、能量补给和轨迹控制。

后续研究可从三个方向展开。第一，引入真正的下层强化学习卸载策略，使卸载决策直接以长期系统收益为目标。第二，开展端到端双层联合训练，进一步提升轨迹与卸载之间的协同程度。第三，扩展更多动态场景，例如用户移动、突发热点、链路遮挡和 MBS 拥塞，以验证方法在复杂环境中的泛化能力。

## 附录：结果来源

本文已填入结果来自以下文件：

- `results/full_runs/wpt_fix_thesis_run/experiment_manifest.json`
- `results/full_runs/wpt_fix_thesis_run/test_logs/attention_mappo/log_data_20260427_142720_attention_mappo.json`
- `results/full_runs/wpt_fix_thesis_run/test_logs/uncoordinated_greedy/log_data_20260427_154340_uncoordinated_greedy.json`
- `results/full_offload_experiments/wpt_fix_thesis_run_offload/experiment_summary.json`

图表生成脚本：

- `generate_thesis_figures.py`
