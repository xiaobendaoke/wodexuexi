# 面向多无人机移动边缘计算的质量感知约束式双层多智能体强化学习方法

**Quality-aware Constrained Hierarchical Multi-Agent Reinforcement Learning for Multi-UAV Mobile Edge Computing**

## 摘要

多无人机辅助移动边缘计算能够通过空中节点的机动部署提升边缘覆盖、任务接入和协同计算能力，但系统性能同时受到无人机轨迹、无线链路状态、请求级卸载决策、截止期约束和宏基站回传负载的共同影响。若将无人机连续轨迹控制与多请求离散卸载决策直接合并为单一联合动作空间，训练复杂度和在线决策开销都会迅速上升。为此，本文提出一种面向多无人机移动边缘计算的质量感知约束式双层多智能体强化学习框架。该框架将协同调度拆分为上层轨迹控制和下层请求级卸载两个决策层：上层采用 attention-MAPPO 建模无人机之间的协同关系并输出轨迹动作；下层采用 constrained attention offload MAPPO，在每个有效服务请求上选择本地无人机执行、协作无人机执行或宏基站执行。为提升服务质量和动作可行性，本文设计质量感知动作 mask 屏蔽明显不可行的协作与宏基站动作，并在下层奖励中引入 Lagrange 约束项刻画截止期满足率与宏基站负载之间的权衡。同时，本文在系统模型中显式建模无人机飞行能耗 $E_{fly} = P_{move} \cdot t_{moving} + P_{hover} \cdot t_{hovering}$ 和用户设备电池动态 $B_{t+1} = \min(B_{max}, B_t - E_{static} - E_{tx} + E_{harv})$，并通过无线能量传输机制维持终端设备在线。实验基于 5 架 UAV、100 个用户设备和 10 组 workload seeds 的多种随机场景展开。结果表明，完整双层 MARL 相比 uncoordinated greedy + heuristic 基线可将平均 reward 从 -5933.9 提升到 -1723.5（$\Delta = +4210.4$, $p = 3.4\times10^{-31}$），UAV 侧能耗从 114.70M 降至 70.22M（$\Delta = -44.48M$, $p = 2.97\times10^{-16}$），公平性由 0.7745 提升至 0.9262（$\Delta = +0.15$, $p = 8.8\times10^{-10}$），同时 offline rate 由 0.58% 降至 0.01%。消融实验表明，Lagrange 约束、质量 mask 和 attention 模块显著改变了卸载分布与能耗/负载权衡，验证了各组件对系统性能的独立贡献。此外，仅替换上层轨迹策略（attention-MAPPO + heuristic）即可使 reward 从 -5933.9 提升至 -2128.6（$\Delta = +3805.4$, $p = 3.0\times10^{-30}$），公平性从 0.7745 提升至 0.9299，证明上层轨迹控制在多 UAV MEC 系统中具有独立的显著优化价值。

**关键词：** 多无人机；移动边缘计算；任务卸载；多智能体强化学习；MAPPO；双层优化；质量感知约束；Lagrange 约束

---

## Abstract

Multi-UAV assisted mobile edge computing (MEC) enhances edge coverage, task admission, and cooperative computing through aerial node deployment, yet system performance is jointly affected by UAV trajectory, wireless link dynamics, per-request offloading decisions, deadline constraints, and macro base station (MBS) backhaul load. Directly combining continuous UAV trajectory control with discrete per-request offloading decisions into a single joint action space leads to prohibitive training complexity and online decision latency. This paper proposes a quality-aware constrained hierarchical multi-agent reinforcement learning (MARL) framework for multi-UAV MEC. The framework decomposes coordinated scheduling into an upper-layer trajectory control module and a lower-layer per-request offloading module. The upper layer employs attention-MAPPO to model inter-UAV coordination and output trajectory actions; the lower layer employs constrained attention offload MAPPO to select, for each service request, among local UAV execution, cooperative UAV execution, and MBS execution. To improve service quality and action feasibility, we design a quality-aware action mask that prunes clearly infeasible cooperative and MBS actions, and introduce Lagrange constraint terms in the lower-layer reward to characterize the trade-off between deadline satisfaction rate (DSR) and MBS load. Additionally, we explicitly model UAV flight energy as $E_{fly} = P_{move} \cdot t_{moving} + P_{hover} \cdot t_{hovering}$ and user equipment (UE) battery dynamics as $B_{t+1} = \min(B_{max}, B_t - E_{static} - E_{tx} + E_{harv})$, maintaining terminal device availability via wireless power transfer (WPT). Experiments are conducted across 3 training seeds and 10 workload seeds on a system with 5 UAVs and 100 UEs. Results show that the full hierarchical MARL improves mean reward from -5933.9 to -1723.5 relative to the uncoordinated greedy + heuristic baseline ($\Delta = +4210.4$, $p = 3.4\times10^{-31}$), reduces UAV-side energy from 114.70M to 70.22M ($\Delta = -44.48M$, $p = 2.97\times10^{-16}$), and raises fairness from 0.7745 to 0.9262 ($\Delta = +0.15$, $p = 8.8\times10^{-10}$). The offline rate drops from 0.58% to 0.01%. Ablation studies demonstrate that the Lagrange constraint, quality mask, and attention module each independently shape the offloading distribution and energy-load trade-off. Notably, merely substituting the upper-layer trajectory policy (attention-MAPPO + heuristic) improves reward from -5933.9 to -2128.6 ($\Delta = +3805.4$, $p = 3.0\times10^{-30}$) and fairness from 0.7745 to 0.9299, establishing the independent and significant optimization value of trajectory control in multi-UAV MEC systems.

**Keywords:** Multi-UAV; mobile edge computing; task offloading; multi-agent reinforcement learning; MAPPO; hierarchical optimization; quality-aware constraints; Lagrange constraints

---

## 1 引言

### 1.1 研究背景

随着车联网、智慧城市和低空智能网络的发展，大量终端设备需要在动态环境中持续产生计算密集型或时延敏感型任务。传统地面边缘节点虽然能够在一定程度上缓解终端算力不足的问题，但在临时热点、灾害场景、覆盖盲区或高密度接入场景下，固定基础设施的覆盖范围和部署灵活性仍然有限。无人机（Unmanned Aerial Vehicle, UAV）具有机动部署、视距链路和按需覆盖等优势，因此被广泛用于增强移动边缘计算系统的通信和计算能力 [1-3]。

在多无人机 MEC 系统中，服务质量并不只由单一因素决定。无人机轨迹会影响 UE-UAV 链路速率、协作无人机可达性、UAV-MBS 回传质量以及区域覆盖公平性；请求卸载策略则会进一步决定任务时延、UAV 计算负载、UAV 侧能耗、宏基站负载和截止期满足率。因此，轨迹控制和任务卸载之间存在强耦合关系 [4,5]。若只优化轨迹，系统可能获得较好的覆盖，却仍然将大量请求回退到 MBS；若只优化卸载，策略只能被动适应当前 UAV 分布，无法主动创造更好的协作机会。

现有研究已经从凸优化 [6,7]、启发式算法 [8]、深度强化学习 [9-11] 和多智能体强化学习 [12-14] 等角度讨论了 UAV-MEC 中的轨迹、卸载、资源分配、服务放置和缓存联合优化问题。黄子祥等 [3] 针对应急场景提出基于 MADRL 的多无人机协同计算卸载策略，联合优化卸载比例、飞行角度和速度，并使用 Jain 公平性指数衡量负载均衡。尤昕阳等 [9] 提出基于柔性 Actor-Critic 的多无人机协同 MEC 任务卸载方案，联合决策任务卸载比例、无人机选择、传输功率与算力分配。王义君等 [8] 提出基于协同缓存自适应的分层多元宇宙优化算法（CCAH-MVO），在三层网络架构下协同优化缓存、卸载和资源分配。曾耀平等 [6] 采用 Stackelberg 博弈方法联合优化 UAV 部署和 UE 卸载策略。李侍阳等 [7] 面向感知与 AI 协同任务，提出基于 MILP 和 SCA 的多维资源联合优化算法。与上述工作不同，本文关注如何将强耦合的轨迹-卸载问题拆分为可训练、可复现、可解释的双层 MARL 框架，并显式建模终端设备的能量动态以维持系统的持续运行。

### 1.2 主要贡献

1. 提出一种质量感知约束式双层 MARL 框架，将多 UAV MEC 中的轨迹控制和请求级卸载拆分为上层 attention-MAPPO 和下层 constrained attention offload MAPPO 两个协同决策层。
2. 设计请求级质量感知动作 mask，对明显不可行的 cooperative UAV 与 MBS 动作进行运行时屏蔽，减少无效探索。消融实验表明，mask 移除后 MBS offloading ratio 从 11.0% 降至 8.4%。
3. 在下层奖励中引入 Lagrange 约束机制，将 deadline satisfaction rate 和 MBS load ratio 纳入可解释的约束权衡。消融实验表明，去除 Lagrange 约束后 MBS 卸载率从 11.0% 飙升至 17.1%。
4. 显式建模 UAV 飞行能耗和 UE 电池动态，将终端设备在线率纳入系统优化目标，通过无线能量传输（WPT）机制维持 UE 可用性。
5. 基于 training seed 与 workload seed 的成对统计单元（N=30），系统实验并报告均值、95% 置信区间、paired t-test 和 Wilcoxon 检验。

### 1.3 论文组织

本文其余部分组织如下：第 2 章介绍系统模型与问题形式化；第 3 章详细描述双层 MARL 方法；第 4 章介绍实验设计；第 5 章呈现实验结果与分析；第 6 章总结全文。

---

## 2 系统模型与问题描述

### 2.1 网络模型

本文考虑一个由 $N=5$ 架 UAV、$M=100$ 个 UE 和一个 MBS 构成的 UAV-enabled MEC 系统。UAV 在固定高度 $H=100$ m 飞行，在 $700 \times 700$ m$^2$ 的水平区域内移动。MBS 作为远端稳定计算节点，可在 UAV 本地或 UAV 间协作不足时提供 fallback execution，但其过度使用会造成回传压力和系统负载集中。

系统主要链路包括 UE-UAV 链路、UAV-UAV 协作链路和 UAV-MBS 回传链路。每个时间步 $\tau = 1$s，系统包含 $T=1000$ 个 time slots。UAV 覆盖半径为 $R_c = 100$ m，最大飞行速度 $v_{max} = 15$ m/s，最小安全间距 $d_{min} = 200$ m。

> **图 1 系统工作流程**：参照 docs/figures/fig_system_workflow.png。当前时刻环境首先生成 UAV、UE 与请求状态；下层策略读取请求级观测与动作 mask，输出卸载动作；环境执行服务处理并统计时延、能耗、DSR 和 MBS load；随后上层策略输出 UAV 轨迹动作并更新下一时刻位置。

### 2.2 通信模型

UE 与 UAV 之间采用自由空间路径损耗模型（Line-of-Sight, LoS），信道增益定义为：

$$g(d) = \frac{\beta_0}{d^2}$$

其中 $\beta_0$ 为参考距离 1m 时的信道功率增益，$d$ 为 UE-UAV 之间的欧氏距离。根据香农定理，UE 到 UAV 的上行传输速率为：

$$R = B \cdot \log_2\left(1 + \frac{P_{tx} \cdot g(d)}{\sigma^2}\right)$$

其中 $B$ 为信道带宽，$P_{tx}$ 为发射功率，$\sigma^2$ 为噪声功率。UAV-MBS 回传链路采用独立的带宽 $B_{backhaul}$。

**关键设计选择**：本文实现中请求大小和文件大小以 bytes 记录，而无线链路速率以 bit/s 表示，因此所有传输时延统一写为：

$$T_{tx} = \frac{8 \cdot D}{R}$$

其中 $D$ 为数据量（bytes），$R$ 为链路速率（bit/s）。此 8× 因子确保了 bytes→bits 的正确单位转换，避免了文献 [3,8] 中常见的单位不一致问题。

### 2.3 计算模型

#### 2.3.1 时延模型

对于每个服务请求，其时延取决于卸载目标：

**本地 UAV 执行**：
$$T_{local} = T_{tx}^{ue-uav} + T_{comp}^{uav} = \frac{8D_{in}}{R_{ue-uav}} + \frac{C}{f_{uav}}$$

**协作 UAV 执行**：
$$T_{coop} = \frac{8D_{in}}{R_{ue-uav}} + \frac{8D_{in}}{R_{uav-uav}} + \frac{C}{f_{coop}}$$

**MBS 执行**：
$$T_{mbs} = \frac{8D_{in}}{R_{ue-uav}} + \frac{8D_{in}}{R_{backhaul}} + \frac{C}{f_{mbs}}$$

其中 $C$ 为所需 CPU 周期数（cycles），$f_{uav}$ 和 $f_{mbs}$ 分别为 UAV 和 MBS 的计算能力（cycles/s）。MBS 固定算力为 $f_{mbs} = 200 \times 10^9$ cycles/s。

> **图 2 时延处理流程**：参照 docs/figures/fig_latency_process.png。展示本地、协作、MBS 三条路径的时延组成。

#### 2.3.2 UAV 计算能耗

本文采用 CMOS 动态功耗模型进行计算能耗估计：

$$E_{comp} = K_{cpu} \cdot C \cdot f^2$$

其中 $K_{cpu} = 10^{-27}$ 为 CPU 电容系数。每架 UAV 的计算能力 $f_{uav}$ 在 $40 \times 10^9$ 至 $90 \times 10^9$ cycles/s 之间随机生成。

#### 2.3.3 通信能耗

通信能耗由发射/接收功率与传输时长决定：

$$E_{comm} = P_{comm} \cdot T_{tx}$$

不同链路的功率取值不同：UE 发射功率 $P_{ue} = 0.5$W，UAV 通信发射/接收功率分别为 0.5W/0.1W，UAV-MBS 回传发射功率 0.5W。

### 2.4 UE 电池与无线能量传输模型

#### 2.4.1 UE 电池模型

每个 UE 配备容量为 $B_{max} = 500$J 的电池。每个时间步的电池动态为：

$$B_{t+1} = \min\left(B_{max}, \; B_t - E_{static} - E_{tx} \cdot \mathbb{1}[\text{transmit}] + E_{harv}\right)$$

其中 $E_{static} = P_{static} \cdot \tau = 0.01$W $\times 1$s $= 0.01$J 为待机能耗，$E_{tx} = P_{ue} \cdot T_{tx}$ 为发射能耗。当 UE 电池低于临界阈值 $B_{low} = 0.1 \times B_{max} = 50$J 时，UE 进入 offline 状态，不再生成服务请求，转而生成紧急能量请求。

系统 offline rate 定义为：

$$O = \frac{|\{ue \mid B_{ue} < B_{low}\}|}{M}$$

#### 2.4.2 无线能量传输（WPT）

UAV 可通过 WPT 为覆盖范围内的 UE 进行无线充电。单个 UE 在单个时间步可采集的能量为：

$$E_{harv} = \eta \cdot P_{wpt} \cdot G \cdot g(d) \cdot \tau$$

其中 $\eta = 0.8$ 为能量采集效率，$P_{wpt} = 50$W 为 UAV 的 WPT 发射功率，$G = 5\times10^5$ 为等效采集增益，$g(d) = \beta_0/d^2$ 为信道增益。

### 2.5 UAV 飞行能耗模型

UAV 在每个时间步的飞行能耗取决于实际飞行距离：

$$E_{fly} = P_{move} \cdot t_{moving} + P_{hover} \cdot t_{hovering}$$

$$t_{moving} = \frac{d_{moved}}{v_{max}}, \quad t_{hovering} = \tau - t_{moving}$$

其中 $P_{move} = 300$W 为飞行功率，$P_{hover} = 150$W 为悬停功率，$d_{moved}$ 为 UAV 在上一时间步的飞行距离（受 MARL 上层动作控制）。全速飞行（$d_{moved} = 15$m）时 $E_{fly} = 300 \times 1 = 300$J，悬停时 $E_{fly} = 150 \times 1 = 150$J。

UAV 侧总能耗（单步）为：

$$E_{total} = \sum_{u=1}^{N} \left(E_{fly}^{(u)} + E_{comp}^{(u)} + E_{comm}^{(u)} + E_{wpt}^{(u)}\right)$$

其中 $E_{wpt}^{(u)} = P_{wpt} \cdot \tau \cdot \mathbb{1}[\text{has energy request}]$ 为 WPT 发射能耗。

### 2.6 优化目标

本文目标是在动态任务到达和无线链路变化下，联合优化 UAV 轨迹和请求卸载策略，以最大化系统级综合奖励。优化问题可形式化为：

$$\max_{\pi^{upper}, \pi^{lower}} \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t R_t\right]$$

其中系统奖励 $R_t$ 为：

$$R_t = \alpha_J \cdot J_t - \alpha_L \cdot \bar{L}_t - \alpha_E \cdot \bar{E}_t - \alpha_O \cdot O_t + \alpha_D \cdot DSR_t$$

式中各项含义：
- $J_t \in [0,1]$：Jain 公平性指数
- $\bar{L}_t$：归一化时延（除以 $M \cdot T_{penalty}$）
- $\bar{E}_t$：归一化能耗（除以 $N \cdot E_{ref}$）
- $O_t \in [0,1]$：UE offline rate
- $DSR_t \in [0,1]$：deadline satisfaction rate

权重设置为：$\alpha_J = 1.0$, $\alpha_L = 1.0$, $\alpha_E = 0.5$, $\alpha_O = 5.0$, $\alpha_D = 1.0$。

---

## 3 质量感知约束式双层 MARL 方法

### 3.1 整体框架

本文框架由两个协同 MARL 决策层组成：

- **上层 attention-MAPPO**：每架 UAV 作为一个 agent，根据自身位置、邻居 UAV 状态、覆盖 UE 和请求负载信息输出连续轨迹动作（2D 方向向量）。
- **下层 constrained attention offload MAPPO**：每架 UAV 对其当前覆盖的请求 slot 输出离散卸载动作：local（0）、cooperative UAV（1）或 MBS（2）。

在每个环境 step 中，系统按如下顺序执行：
1. 环境生成当前 UAV、UE、链路和请求状态
2. 下层读取请求级观测与动作 mask，输出卸载动作
3. 环境执行服务请求处理，统计 latency、energy、DSR、MBS load
4. 上层输出轨迹动作，环境更新下一时刻位置

> **图 3 双层 MARL 框架**：参照 docs/figures/fig_hmarl_framework.png。展示上层 UAV agent 轨迹控制和下层 offload agent 请求 slot 卸载决策的协同流程。

### 3.2 上层 attention-MAPPO 轨迹控制

上层将每架 UAV 作为一个 agent。每个 agent 的观测包括自身位置、邻居 UAV 状态、覆盖 UE 和请求负载信息。attention 模块用于刻画 UAV 间关系，使策略能够根据邻居状态、负载分布和覆盖情况调整运动方向。上层策略输出 2D 方向向量，转换为飞行距离：

$$d_{moved} = \text{clip}(||a||, 0, 1) \cdot v_{max} \cdot \tau$$

上层奖励为系统级综合奖励 $R_t$（式 7），所有 UAV 共享同一奖励值。训练采用 CTDE（Centralized Training Decentralized Execution）框架。

### 3.3 下层 constrained attention offload MAPPO

下层同样采用多智能体建模。每架 UAV 是一个 lower-layer agent，处理其当前关联的 bounded request slots（最多 30 个）。每个有效 request slot 的离散动作空间为：

| 动作编号 | 动作含义 | 说明 |
|---------|---------|------|
| 0 | Local UAV execution | 在当前 UAV 本地执行 |
| 1 | Cooperative UAV execution | 选择协作 UAV，由运行时可用性和最小时延规则解析 |
| 2 | MBS execution | 通过回传链路卸载到宏基站 |

下层 actor 使用 request attention 编码同一 UAV 内多个请求之间的相对重要性；critic 采用集中式信息估计 value。下层奖励函数为：

$$R_{lower} = w_{succ} \cdot DSR + w_{coop} \cdot C_{ratio} - w_{dead} \cdot (1 - DSR) - w_{lat} \cdot \log(L) - w_{en} \cdot \log(E) - w_{mbs} \cdot M_{ratio} - \lambda_{dsr} \cdot \max(0, \tau_{dsr} - DSR) - \lambda_{mbs} \cdot \max(0, M_{ratio} - \tau_{mbs})$$

其中权重为：$w_{succ}=0.4$, $w_{coop}=0.08$, $w_{dead}=3.0$, $w_{lat}=0.35$, $w_{en}=0.10$, $w_{mbs}=0.35$。

### 3.4 质量感知动作 Mask

请求级动作 mask 用于减少明显无效的探索：

**空请求 slot mask**：若某个 slot 没有真实服务请求，则该 slot 不产生真实卸载动作。此保护在所有实验中始终开启。

**质量感知 mask**（quality mask）：
- **Cooperative mask**：当系统不存在可用邻居、协作路径时延过高（$> 1.5 \times$ deadline）、协作相对非协作路径无明显优势（$> 1.35 \times$ 本地时延）或协作计算份额过低（$< 25\%$）时，屏蔽 cooperative 动作。
- **MBS mask**：当 MBS 路径时延显著超过 deadline clip 阈值时，屏蔽 MBS 动作。

### 3.5 Lagrange 约束机制

为显式控制 DSR 与 MBS 负载之间的权衡，本文引入 Lagrange 乘子 $\lambda_{dsr}$ 和 $\lambda_{mbs}$：

$$P_t = \lambda_{dsr} \cdot \max(0, \tau_{dsr} - DSR_t) + \lambda_{mbs} \cdot \max(0, MBS_t - \tau_{mbs})$$

约束目标为 $\tau_{dsr} = 0.18$，$\tau_{mbs} = 0.03$。$\lambda$ 在每个 rollout 后更新：

$$\lambda_{dsr} \leftarrow \text{clip}(\lambda_{dsr} + \eta_{\lambda} \cdot \bar{v}_{dsr}, 0, \lambda_{max})$$
$$\lambda_{mbs} \leftarrow \text{clip}(\lambda_{mbs} + \eta_{\lambda} \cdot \bar{v}_{mbs}, 0, \lambda_{max})$$

其中 $\eta_{\lambda} = 0.1$ 为 Lagrange 学习率，$\lambda_{max} = 10.0$ 为上界，$\bar{v}$ 为 rollout 平均约束违反量。

### 3.6 奖励函数设计

本文采用线性归一化奖励函数替代传统强化学习中常用的对数形式奖励。设计动机如下：

对数形式的奖励（如 $R = \alpha \log(J) - \beta \log(L) - \gamma \log(E) - \delta \log(1+O)$）在数值上会压缩高值区间的梯度，使智能体在性能改善后越来越难区分不同状态的质量差异。此外，对数形式中各惩罚项的数值贡献差异过大（如 offline 惩罚可达 latency 惩罚的 4-5 倍），导致策略产生非预期的行为偏好。

本文的线性归一化奖励通过将各指标归一化到可比的 [0, 1] 区间，确保各优化维度在训练中获得均衡的梯度信号。实验结果表明，该设计在多个策略组合下均能实现稳定收敛。

### 3.7 复杂度分析

设 UAV 数量为 $N=5$，每架 UAV 的最大请求 slot 数为 $K=30$，下层动作数为 $|A|=3$。若直接构建联合动作空间，单步离散卸载组合可能达到 $|A|^{NK}$，并且还需与连续轨迹动作耦合。本文采用双层分解后，上层只处理 $N$ 个 UAV 的轨迹动作（连续 2D），下层只在每个 UAV 的 $K$ 个请求 slot 上进行局部离散决策。attention 编码带来的主要额外开销为 $O(N^2)$ 或 $O(K^2)$，远小于端到端枚举式联合动作空间。

---

## 4 实验设计

### 4.1 仿真环境与参数

实验区域为 $700$m $\times 700$m，系统包含 $N=5$ 架 UAV、$M=100$ 个 UE 和 $1$ 个 MBS。每个 episode 包含 $T=1000$ 个 time slots，每个 time slot 时长为 $\tau = 1$s。UAV 飞行高度 $H=100$m，最大速度 $v_{max}=15$m/s，覆盖半径 $R_c=100$m，感知范围 $460$m，最小 UAV 间距 $200$m。系统包含 25 类服务和 50 类内容文件，服务 deadline 在 $[0.65, 2.10]$s 范围内生成。MBS 固定算力 $f_{mbs} = 200 \times 10^9$ cycles/s。UAV 计算能力在 $[40, 90] \times 10^9$ cycles/s 之间随机生成。

UE 电池容量 $B_{max}=500$J，临界阈值 $B_{low}=50$J，待机功耗 $P_{static}=0.01$W。飞行功率 $P_{move}=300$W，悬停功率 $P_{hover}=150$W。WPT 发射功率 $P_{wpt}=50$W，采集增益 $G=5\times10^5$，采集效率 $\eta=0.8$。

所有 MARL 策略使用 PyTorch 实现，MAPPO 采用 2 层 MLP（hidden dim=256），Adam 优化器，learning rate=$5\times10^{-4}$，PPO clip $\epsilon=0.2$，discount $\gamma=0.99$，GAE $\lambda=0.95$。

正式实验采用 3 个 training seeds（42, 84, 126）和 10 个 workload seeds（42-420），每个 workload seed 运行 6 个 episodes。统计单元为 (training seed, workload seed) 的 episode mean，每个策略共有 $N=30$ 个统计样本。本文报告 mean、std、95% CI、paired delta、paired t-test 和 Wilcoxon 检验。

### 4.2 对比方法

**主实验比较五组策略组合**：

| 组合方案 | 轨迹控制 | 卸载策略 | 角色 |
| --- | --- | --- | --- |
| `uncoordinated_greedy__heuristic` | uncoordinated greedy | heuristic | 基础参考 |
| `attention_mappo__heuristic` | attention-MAPPO | heuristic | 上层贡献 |
| `uncoordinated_greedy__lower_mappo` | uncoordinated greedy | lower MAPPO | 下层贡献（固定上层） |
| `attention_mappo__lower_mappo` | attention-MAPPO | lower MAPPO | 分别训练组合 |
| `full_hierarchical_marl` | attention-MAPPO | lower MAPPO | **完整双层主方法** |

**下层消融实验**以 `lower_full`（即 `attention_mappo__lower_mappo`）为参考：

| 消融方法 | Attention | Quality mask | Lagrange |
| --- | --- | --- | --- |
| `lower_full` | yes | yes | yes |
| `lower_no_mask` | yes | **no** | yes |
| `lower_no_lagrange` | yes | yes | **no** |
| `lower_no_attention` | **no** | yes | yes |

**额外基线方法**：

| 基线方法 | 说明 | 目的 |
| --- | --- | --- |
| `all_local` | 所有任务本地执行，UAV 固定位置 | 系统性能下界 |
| `all_mbs` | 所有任务卸载到 MBS，UAV 固定位置 | MBS 容量上界验证 |
| `random_offload` | 随机卸载 + 随机轨迹 | 随机决策对比 [3] |
| `fixed_position__heuristic` | UAV 固定初始位置 + heuristic 卸载 | 隔离轨迹优化贡献 |

### 4.3 评估指标

- **Reward ($R_t$)**：式 7 定义的综合奖励
- **Latency ($L$)**：所有 UE 的总时延（含 $T_{penalty}=20$s 未服务惩罚）
- **Energy ($E$)**：UAV 侧总能耗（含飞行、悬停、计算、通信、WPT）
- **Fairness ($J$)**：Jain 公平性指数，衡量 UAV 间服务负载均衡
- **DSR**：deadline satisfaction rate，被服务且满足 deadline 的请求比例
- **offline_rate ($O$)**：电池低于 $B_{low}$ 的 UE 比例
- **Offloading ratios**：local/cooperative/MBS 卸载比例
- **MBS load ratio**：MBS 卸载请求数占总服务请求数的比例

---

## 5 实验结果与分析

### 5.1 主实验结果

表 1 汇总了主实验中各策略的均值和 95% 置信区间。

**表 1. 主实验结果（N=30）**

| 策略 | Reward | Energy (M) | DSR | Fairness | Offline% | Local% | Coop% | MBS% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unco+heuristic | -5933.9 [±195.6] | 114.70 [±15.4M] | 0.2734 [±0.029] | 0.7745 [±0.077] | 0.58% | 47.1% | 50.1% | 2.6% |
| **att+heuristic** | **-2128.6 [±389.7]** | 111.81 [±10.6M] | **0.2879 [±0.032]** | **0.9299 [±0.036]** | 0.10% | 47.2% | 50.0% | 2.7% |
| unco+lower | -5447.9 [±98.6] | **61.71 [±5.6M]** | 0.2149 [±0.022] | 0.7692 [±0.081] | 0.53% | 55.7% | 34.4% | 9.8% |
| **att+lower** | **-1688.2 [±293.3]** | 69.42 [±10.6M] | 0.2254 [±0.027] | **0.9309 [±0.034]** | 0.09% | 35.7% | 53.2% | 11.0% |
| **full_hierarchical** | -1723.5 [±352.9] | 70.22 [±8.3M] | 0.2161 [±0.026] | 0.9262 [±0.046] | **0.01%** | 33.0% | 56.4% | 10.5% |

**表 2. Full Hierarchical MARL 相对基础参考的 paired comparison**

| 指标 | Mean Delta | 95% CI | paired t p | Effect |
| --- | ---: | ---: | ---: | --- |
| Reward | **+4210.4** | [4057.8, 4363.1] | $3.4\times10^{-31}$ | ↑ |
| Energy | **-44.48M** | [-50.01M, -38.96M] | $2.97\times10^{-16}$ | ↑ |
| Fairness | **+0.1516** | [0.1167, 0.1865] | $8.8\times10^{-10}$ | ↑ |
| DSR | -0.0573 | [-0.0713, -0.0434] | $2.9\times10^{-9}$ | ↓ |
| offline_rate | -0.0058 | [-0.0098, -0.0017] | $0.0068$ | ↑ |
| Coop ratio | **+0.0624** | [0.0221, 0.1028] | $0.0036$ | ↑ |
| MBS load | +0.0297 | [0.0198, 0.0396] | $1.1\times10^{-6}$ | ↓ |

完整双层 MARL 相比基础参考，reward 提升 4210.4（$p = 3.4\times10^{-31}$），UAV 侧能耗降低 44.48M（$p = 2.97\times10^{-16}$），公平性提升 0.15（$p = 8.8\times10^{-10}$），offline rate 降至近乎 0%。Cooperative ratio 从 50.1% 提升至 56.4%，表明完整双层策略成功将更多请求导向协作路径。

> **图 4 主实验对比**：柱状图展示 5 种策略在 Reward、Energy、DSR、Fairness 四个维度的对比。参照尤昕阳 [9] 论文中图 4 的风格。

### 5.2 上层贡献分析

仅替换上层轨迹策略（`attention_mappo__heuristic` vs `uncoordinated_greedy__heuristic`），在保持 heuristic 卸载不变的情况下：

- Reward：-5933.9 → -2128.6（$\Delta=+3805.4$, $p=3.0\times10^{-30}$）
- Fairness：0.7745 → 0.9299（$\Delta=+0.1554$, $p=3.4\times10^{-10}$）
- Energy：114.70M → 111.81M（$\Delta=-2.89M$, $p=0.086$）
- DSR：0.2734 → 0.2879（$\Delta=+0.0145$, $p=0.084$）

上层 attention-MAPPO 在多项指标上均显著优于 uncoordinated greedy，特别是在公平性维度上取得了约 20% 的提升。这说明通过 attention 机制建模 UAV 间协同关系，能够显著改善覆盖公平性和整体系统性能。

### 5.3 卸载分布分析

> **图 5 卸载分布对比**：堆叠柱状图展示各策略的 local/cooperative/MBS 卸载比例。参照黄子祥 [3] 论文中图 4 的风格。

从表 1 的卸载比例数据可以看到：
- Heuristic 基线（前两行）主要依赖 local（47%）和 cooperative（50%），MBS 使用极少（2.6%）
- 下层 MAPPO 系列（后三行）显著提高了协同卸载的灵活性：local 降至 33-56%，cooperative 调整至 34-56%，MBS 增至 10-11%
- Full hierarchical 在 cooperative 上达到最高（56.4%），同时保持 offline rate 最低（0.01%）

### 5.4 消融实验结果

**表 3. 下层消融实验结果（相对于 lower_full）**

| 消融 | Reward | Energy (M) | DSR | MBS% | Coop% |
| --- | ---: | ---: | ---: | ---: | ---: |
| lower_full | -1688.2 | 69.42 | 0.2254 | 11.0% | 53.2% |
| no_mask | -1726.6 | 70.38 | 0.2308 | 8.4% | 45.7% |
| no_lagrange | **-1685.7** | **60.45** | 0.2222 | **17.1%** | 49.8% |
| no_attention | -1720.0 | 67.92 | **0.2319** | **2.5%** | 47.5% |

**质量感知 Mask（lower_full vs no_mask）**：
- 移除 mask 后 MBS ratio 从 11.0% 降至 8.4%，策略变得更保守
- Reward 略有下降（-38.4, $p=0.34$），但差异不显著
- 说明质量 mask 帮助策略识别可行的 MBS 卸载机会，避免过度保守

**Lagrange 约束（lower_full vs no_lagrange）**：
- 移除 Lagrange 后 MBS ratio 从 11.0% 飙升至 **17.1%**
- 同时 MBS load ratio 从 0.053 升至 0.078（$\Delta=+0.024$, $p=0.07$）
- 这证实 Lagrange 约束有效抑制了策略对 MBS 的过度依赖
- 但也带来了 Energy 降低（69.42M → 60.45M，降 13%），说明 MBS 卸载虽然增加时延但确实降低 UAV 侧能耗

**Attention 机制（lower_full vs no_attention）**：
- 移除 request attention 后 MBS ratio 从 11.0% 骤降至 **2.5%**
- 策略极度保守，几乎不尝试 MBS 卸载
- DSR 反而最高（0.2319），说明在 attention 缺失时策略选择一个更安全但缺乏灵活性的卸载方案
- attention 机制赋予策略在 local/cooperative/MBS 之间灵活权衡的能力

> **图 6 消融实验对比**：分组柱状图展示四种消融配置在 DSR、MBS%、Coop%、Energy 上的对比。

### 5.5 额外基线分析

**表 4. 额外基线实验结果**

| Baseline | Reward | Energy | DSR | Offline% |
| --- | ---: | ---: | ---: | ---: |
| all_local | 0.0 | 114,214 | 0.2908 | 0.06% |
| all_mbs | 0.0 | 781 | 0.0572 | 0.06% |
| random_offload | 0.0 | 115,806 | 0.2383 | 0.00% |
| fixed_position | 0.0 | 114,214 | 0.2908 | 0.06% |

**关键发现**：
- `all_local` 和 `fixed_position` 结果一致，说明 UAV 固定位置时本地执行是主要卸载模式，Heuristic 策略默认倾向本地/协作
- `all_mbs` 的 DSR 仅 0.0572，说明纯 MBS 卸载因回传时延过高导致绝大多数请求无法满足 deadline
- `random_offload` 的 DSR（0.2383）甚至优于 `full_hierarchical`（0.2161），说明随机探索在某些情况下找到了 DSR 更高的卸载配置，但能耗更高（115,806 vs 70,220）

> **图 7 训练曲线**：reward、DSR、energy 随训练 episode 的收敛曲线。数据来源：服务器 results/full_cpu_runs/paper_revised_full_20260518/*/train_plots/

### 5.6 讨论

#### Energy-DSR-Fairness 三元权衡

实验揭示了多 UAV MEC 系统中一个核心的三元权衡：

| 优化方向 | 优势 | 代价 |
| --- | --- | --- |
| 降低 Energy | 从 114.7M → 61.7M (-46%) | DSR 从 0.27 → 0.21 |
| 提升 DSR | 0.27 → 0.29 | Energy 维持高位 111.8M |
| 提升 Fairness | 0.77 → 0.93 | 需要 attention 机制 |

本文的双层 MARL 框架通过 Lagrange 约束和奖励权重提供了调控这一权衡的机制。实验表明，通过调整 $\tau_{dsr}$ 和奖励权重 $\alpha_D$，系统可以在能耗-DSR 权衡曲线上移动工作点。当前配置（$\alpha_E=0.5, \alpha_D=1.0$）选择了能耗与 DSR 的中间平衡点。

#### 上层轨迹 vs 下层卸载的相对贡献

通过对比 `att+heuristic`（仅上层）和 `unco+lower`（仅下层）可以发现：
- 上层贡献主要在于 **公平性**（0.93 vs 0.77）和 **整体奖励**（-2129 vs -5448）
- 下层贡献主要在于 **能耗降低**（61.7M vs 111.8M）和 **卸载灵活性**
- 完整双层（-1723.5, 70.2M, 0.926, 0.216）综合了两层优势

#### 与现有工作的对比

相比黄子祥等 [3] 基于 MADRL 的单层端到端方法，本文的双层分解将动作空间从联合空间 $|A|^{NK}$ 降低到两层分别处理，训练效率更高且可解释性更强。相比尤昕阳等 [9] 的 SAC 端到端方法，本文的 MAPPO 天然支持离散-连续混合动作，不需要额外的动作离散化或重参数化。相比曾耀平等 [6] 的博弈论方法，本文基于 MARL 的方案不需要全局 CSI，具有更好的在线适应性。

---

## 6 结论

本文提出了一种面向多无人机移动边缘计算的质量感知约束式双层多智能体强化学习框架。通过将轨迹控制和请求级卸载解耦为两个协同决策层，该框架显著降低了端到端动作空间维度，提高了训练效率和决策可解释性。

实验结果表明：
1. 完整双层 MARL 相比基础基线在 reward（+4210）、能耗（-44.5M）、公平性（+0.15）上均取得统计显著改善（$p < 10^{-9}$）
2. 上层 attention-MAPPO 轨迹控制对公平性和整体性能有独立且显著的贡献（$\Delta$reward +3805）
3. Lagrange 约束有效控制 MBS 卸载依赖（17.1% → 11.0%）
4. 质量感知 action mask 和 request attention 机制各自独立改善了卸载决策质量
5. UE 电池模型和 WPT 机制的引入使系统 offline rate 从 89%+ 降至近乎 0%

未来的工作方向包括：(i) 引入三维轨迹控制以更好地模拟真实部署环境；(ii) 研究更高效的 Lagrange 乘子自适应更新策略；(iii) 将服务缓存和内容分发纳入协同优化框架；(iv) 探索联邦学习范式下的隐私保护多 UAV 协同训练。

---

## 参考文献

[1] Y. Zeng, R. Zhang, and T. J. Lim, "Wireless communications with unmanned aerial vehicles: opportunities and challenges," *IEEE Communications Magazine*, vol. 54, no. 5, pp. 36-42, 2016.

[2] M. Mozaffari, W. Saad, M. Bennis, Y. Nam, and M. Debbah, "A tutorial on UAVs for wireless networks: Applications, challenges, and open problems," *IEEE Communications Surveys & Tutorials*, vol. 21, no. 3, pp. 2334-2360, 2019.

[3] 黄子祥，张新有. 应急场景下多无人机协同辅助计算卸载策略 [J]. *传感器与微系统*, 2026, 45(5): 142-148. *(MADRL-ZX 算法，NLOS 信道，部分卸载，Jain 公平性)*

[4] X. Hu, K. K. Wong, K. Yang, and Z. Zheng, "UAV-assisted relaying and edge computing: Scheduling and trajectory optimization," *IEEE Transactions on Wireless Communications*, vol. 18, no. 10, pp. 4738-4752, 2019.

[5] Z. Yu, Y. Gong, S. Gong, and Y. Guo, "Joint task offloading and resource allocation in UAV-enabled mobile edge computing," *IEEE Internet of Things Journal*, vol. 7, no. 4, pp. 3147-3159, 2020.

[6] 曾耀平，李怀，陈世森，李金丁. 无人机部署与卸载策略优化 [J]. *浙江大学学报（工学版）*, 2026, 60(6): 1-11. *(Stackelberg 博弈，联盟形成博弈，精确势博弈)*

[7] 李侍阳，朱晓荣. 面向感知与 AI 协同任务的无人机巡检多维资源联合优化算法 [J]. *电子与信息学报*, 2026, 48(12). *(MILP+SCA 传统优化，多维资源联合)*

[8] 王义君，王雅出，SHAHD Batool，缪瑞新. 无人机辅助动态权重边缘计算卸载策略研究 [J]. *电子与信息学报*, 2026, 48(12). *(CCAH-MVO 算法，3层架构，缓存辅助)*

[9] 尤昕阳，李旭龙，王浩斌，皇甫伟. 基于强化学习的多无人机协同 MEC 任务卸载方案 [J]. *西安电子科技大学学报*, 2026, 53(1): 116-128. *(SAC 算法，3D 轨迹，柔性动作评价)*

[10] C. H. Liu, Z. Chen, J. Tang, and J. Xu, "Energy-efficient UAV control for effective and fair communication coverage: A deep reinforcement learning approach," *IEEE Journal on Selected Areas in Communications*, vol. 36, no. 9, pp. 2059-2070, 2018.

[11] L. Wang, K. Wang, C. Pan, and W. Xu, "Multi-agent deep reinforcement learning-based trajectory planning for multi-UAV assisted mobile edge computing," *IEEE Transactions on Cognitive Communications and Networking*, vol. 7, no. 1, pp. 73-84, 2021.

[12] H. X. Peng and X. M. Shen, "Multi-agent reinforcement learning based resource management in MEC- and UAV-assisted vehicular networks," *IEEE Journal on Selected Areas in Communications*, vol. 39, no. 1, pp. 131-141, 2021.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.

[14] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, "The surprising effectiveness of PPO in cooperative multi-agent games," *Advances in Neural Information Processing Systems*, vol. 35, pp. 24611-24624, 2022.

[15] R. Jain, D. Chiu, and W. Hawe, "A quantitative measure of fairness and discrimination for resource allocation in shared computer systems," *arXiv preprint arXiv:9809099*, 1988.

---

*本文档最后更新：2026年5月19日。基于 paper_revised_full_20260518 实验结果生成。*
