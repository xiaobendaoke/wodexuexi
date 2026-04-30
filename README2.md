# 面向多无人机移动边缘计算的双层协同优化研究：轨迹控制与 SC-OGO 请求级任务卸载

> 当前代码已支持多种下层请求级卸载策略：`heuristic`、`learned`、`cql`、`radcc` 和 `sc_ogo`。最新实验表明，Constrained CQL-DQN 与 RADCC-Offload 能降低部分 MBS load，但会带来较明显 DSR 损失或能耗代价；更适合作为主方法的是 **SC-OGO（Safety-Constrained Oracle-Guided Offloading）**：它在 oracle-guided surrogate classifier 之后加入安全约束重排序，在基本守住 DSR 的同时降低 MBS load、MBS ratio 和能耗。最新组件消融进一步表明，MBS fallback penalty 是 MBS 依赖压缩的主要来源；尺度泛化实验显示 SC-OGO 在 12 组 UAV/UE/热点/backhaul 变化场景中均能降低 MBS ratio 和 MBS load。旧的 oracle-guided classifier 仍保留为 imitation-learning baseline，CQL-DQN 和 RADCC 可作为深度强化学习探索性对照或消融方法。

## 摘要

多无人机辅助移动边缘计算（multi-UAV assisted mobile edge computing, multi-UAV MEC）通过在空中部署具备通信、计算、缓存和移动能力的无人机节点，能够为热点区域通信、临时网络部署、灾害应急保障和边缘智能服务提供灵活支撑。与固定基站或固定边缘服务器相比，无人机节点可以根据地面用户分布和业务压力动态调整位置，从而改善覆盖质量、缩短接入距离并提升边缘服务能力。然而，多无人机 MEC 系统中的轨迹控制、用户覆盖、服务缓存、任务卸载、协作链路、计算队列和宏基站（macro base station, MBS）回退负载存在强耦合关系。若将所有决策统一建模为单层端到端联合优化问题，动作空间会随无人机数量、用户数量和请求数量快速扩大，在线决策复杂度和训练难度显著增加；若仅优化无人机轨迹或仅优化任务卸载，又难以刻画二者之间的动态反馈。

针对上述问题，本文构建一种面向多无人机 MEC 的双层协同优化框架。上层采用 attention-MAPPO 学习多无人机轨迹与协同控制策略，使每架无人机能够根据自身位置、邻居无人机、关联用户、请求状态和服务压力调整运动方向与移动距离。下层采用 SC-OGO request-level offloading policy，在每个服务请求到达或被处理时，从本地无人机执行、协作无人机执行和 MBS 回退执行三类候选路径中选择服务执行位置。SC-OGO 以 oracle-guided surrogate classifier 为基础，先由增强 oracle 监督训练轻量分类器，再在在线推理阶段根据 deadline safety margin、MBS fallback 代价、队列压力和协作可用性进行安全约束重排序。该方法仅使用当前请求上下文，不使用未来请求、未来轨迹或未来链路信息。

基于正式四组端到端联合消融实验，完整方法 `attention_mappo__oracle_guided` 相比强基线 `uncoordinated_greedy__heuristic` 的 reward 提升 212.4，latency 降低 17.43%，energy 降低 18.26%，deadline satisfaction rate（DSR）提升 11.02 个百分点，fairness 提升 0.1876，MBS ratio 降低 57.88%，MBS load ratio 降低 49.90%。在最新下层在线对比中，`sc_ogo_offloading` 相比 `heuristic_offloading` 的 DSR 基本不下降（+0.002 个百分点），同时 MBS load ratio 降低 1.275 个百分点、MBS ratio 降低 4.946 个百分点、energy 降低 2843.78。进一步的组件消融显示，移除 MBS fallback penalty 后 SC-OGO 的 MBS ratio 和 MBS load 优势基本消失；尺度泛化结果显示，SC-OGO 在 12 组规模与 backhaul 条件变化中均降低 MBS load，平均 DSR 变化仅约 -0.011 个百分点。相比 CQL-DQN 和 RADCC-Offload，SC-OGO 更符合“守住 DSR，降低 MBS load”的论文主目标。因此，本文建议将 `attention_mappo__sc_ogo` 作为后续主方法口径，将 oracle-guided surrogate classifier 作为 imitation baseline，将 CQL-DQN 和 RADCC 作为探索性深度学习对照。

**关键词**：多无人机；移动边缘计算；双层协同优化；多智能体强化学习；attention-MAPPO；任务卸载；SC-OGO；oracle-guided policy learning

## 第1章 绪论

### 1.1 研究背景

移动边缘计算（mobile edge computing, MEC）通过将计算与缓存资源下沉到网络边缘，使用户终端能够将计算密集型任务卸载到邻近边缘节点，从而降低端到端服务时延、节省终端能耗并提升服务响应能力。在传统蜂窝网络中，边缘服务器通常部署在固定基站或接入点附近，其服务范围和资源位置相对固定。当用户分布呈现时空动态性，或在应急通信、临时活动、灾害恢复等固定基础设施不足的场景中，固定边缘节点难以及时适配业务热点变化。

无人机（unmanned aerial vehicle, UAV）具备高机动性、部署灵活和空地链路视距条件较好的特点，因此逐渐成为 MEC 系统中的重要空中边缘节点。无人机既可以作为空中接入点为地面用户提供无线覆盖，也可以搭载轻量计算和缓存资源处理服务请求，还可以通过无人机间链路形成协作服务网络。当多个无人机共同服务同一片区域时，系统能够通过协同移动覆盖多个热点区域，通过协作卸载分担单个无人机的计算压力，并通过必要的 MBS fallback 保证服务可达性。

然而，多无人机 MEC 的优势也伴随更复杂的决策耦合。无人机轨迹会改变 UE-UAV 接入距离、UAV-UAV 协作距离和 UAV-MBS 回传质量；任务卸载决策会改变无人机本地队列、协作节点负载、MBS 负载和后续奖励反馈；缓存状态会影响服务是否可在某个无人机上直接执行；用户热点分布会导致某些无人机区域服务压力较高，而另一些无人机资源利用不足。因此，多无人机 MEC 中的轨迹控制与任务卸载不应被简单割裂，也不宜粗暴合并为巨大单层动作空间。

### 1.2 研究意义

已有 UAV-MEC 研究已经广泛讨论轨迹规划、任务卸载、通信资源分配、计算资源分配、服务放置和缓存优化等问题。传统优化方法通常依赖较明确的数学模型和较稳定的环境假设，能够在特定问题规模下给出结构化解法；深度强化学习和多智能体强化学习则更适用于动态环境中的序贯决策。然而，在多无人机、多用户、多请求和多服务类型同时存在的场景中，若把无人机运动、服务执行位置、协作选择和资源分配全部塞入单个策略输出，动作空间会随系统规模迅速扩大，策略学习难度、在线部署复杂度和论文解释难度都会增加。

因此，本文关注的重点不是简单宣称“联合优化轨迹与卸载”这一宽泛目标，而是研究如何在保持上层无人机运动动作空间相对稳定的前提下，将请求级任务卸载设计为一个可训练、可部署、可解释的下层策略模块。该思路具有三个方面的意义。

第一，从系统建模角度看，双层分解能够保留轨迹控制与任务卸载之间的反馈关系。上层轨迹影响下层候选执行路径和链路质量，下层卸载影响时延、能耗、队列压力和 MBS 负载，并最终反馈到上层奖励。二者不是相互独立的两个问题，而是在同一环境中通过状态和指标闭环耦合。

第二，从算法实现角度看，双层框架避免将每个服务请求的卸载动作直接并入上层多智能体连续控制动作。对于 5 架 UAV，上层每个 time slot 只需输出 `5 x 2` 维连续运动动作；而请求级卸载由下层分类器在请求发生时局部完成，从而降低上层策略训练难度。

第三，从论文实验角度看，双层框架便于分阶段分析上层和下层的作用。上层 attention-MAPPO 的贡献可以通过轨迹控制和覆盖公平性体现；下层 oracle-guided 卸载的贡献可以通过请求去向结构和 MBS load 降低体现；最终四组联合消融能够明确回答“上层是否有效、下层是否有效、二者组合是否互补”。

### 1.3 国内外研究现状与本文定位

UAV-MEC 相关研究通常围绕以下几条主线展开。

第一类研究关注 UAV 轨迹与用户覆盖优化。无人机位置直接决定空地链路距离和服务覆盖范围，因此轨迹规划常被用于提升吞吐量、降低时延或改善覆盖公平性。在单 UAV 场景中，轨迹优化可通过凸优化、动态规划或强化学习求解；在多 UAV 场景中，还需要考虑无人机间协作、碰撞避免、覆盖重叠和负载均衡。

第二类研究关注任务卸载与资源分配。MEC 系统中的用户任务可以在本地终端、边缘服务器、协作节点或云端执行。任务卸载决策通常需要综合考虑传输时延、计算时延、能耗、服务 deadline 和资源占用。对于 UAV-MEC，卸载对象不仅包括固定基站和云服务器，还包括可移动的无人机边缘节点，因此卸载性能会随 UAV 轨迹动态变化。

第三类研究关注服务放置、缓存和内容分发。服务文件和内容文件是否缓存在边缘节点上，会直接影响服务执行路径的可行性和时延开销。多无人机场景下，缓存状态还会影响协作卸载收益：若邻近 UAV 已缓存目标服务，则将请求转发至协作 UAV 可能比回退至 MBS 更合适。

第四类研究引入深度强化学习和多智能体强化学习。多无人机系统天然具备多智能体特征，每架 UAV 都需要在局部观测下选择动作，同时其行为又会影响全局服务质量。MAPPO 等集中训练、分散执行方法适合处理此类问题；注意力机制则有助于从邻居 UAV 和关联 UE 中提取更关键的信息。

与上述研究相比，本文不把“联合优化”作为唯一卖点，而是将研究定位为一种双层学习式分解框架：上层学习多无人机轨迹控制，下层学习请求级卸载策略。上层不直接控制每个请求的卸载动作，下层也不改变 UAV 运动轨迹；两层通过链路、队列、缓存、时延、能耗和奖励反馈发生耦合。该设计使论文能够更清楚地解释系统机制，并通过实验区分上层协同控制收益和下层 MBS fallback 压缩收益。

### 1.4 本文主要工作

本文主要工作包括以下几个方面。

1. 构建面向多无人机 MEC 的双层协同优化框架。该框架将上层 UAV 轨迹控制与下层请求级任务卸载解耦建模，同时保留二者在链路质量、候选执行路径、队列压力、缓存状态和系统奖励上的动态反馈关系。

2. 设计基于 attention-MAPPO 的上层多智能体轨迹控制器。每架 UAV 根据自身状态、邻居 UAV 信息、关联 UE 请求状态和缓存状态输出二维连续移动动作；注意力模块用于建模 UAV 间关系和服务压力差异。

3. 设计 SC-OGO request-level offloading policy。下层卸载器将每个服务请求建模为 local、cooperative 和 MBS 三分类问题，利用增强 oracle 生成训练标签，在线阶段先由轻量分类模型近似 oracle 决策，再通过安全约束代价对候选动作进行重排序。

4. 在增强 oracle 与 SC-OGO 重排序中引入 deadline 违约惩罚、deadline margin、本地队列压力、MBS fallback 代理惩罚和协作执行代价，使下层策略不只是复制最小时延规则，而是在守住 DSR 的前提下学习更符合系统目标的负载分散策略。

5. 通过上层多训练 seed 稳健性实验、下层在线运行实验、SC-OGO 组件消融、尺度泛化实验、复杂度部署分析和正式四组端到端联合消融验证方法有效性，并使用同 seed paired comparison、95% CI、paired t-test 与 Wilcoxon 检验增强实验可信度。

## 第2章 系统模型与问题建模

### 2.1 网络场景与集合定义

本文考虑一个由多架 UAV、多个地面用户设备（user equipment, UE）和一个宏基站 MBS 构成的 UAV-MEC 系统。设 UAV 集合为

$$
\mathcal{U}=\{1,2,\ldots,U\},
$$

UE 集合为

$$
\mathcal{M}=\{1,2,\ldots,M\},
$$

服务类型集合为

$$
\mathcal{S}=\{1,2,\ldots,S\},
$$

内容文件集合为

$$
\mathcal{K}=\{1,2,\ldots,K\}.
$$

在当前实验配置中，系统包含 `U=5` 架 UAV、`M=100` 个 UE、`S=25` 类服务和 `K=50` 类内容文件。仿真区域为 `700 m x 700 m`，MBS 位于

$$
\mathbf{p}^{\mathrm{MBS}}=(350,350,30),
$$

UAV 固定飞行高度为 `100 m`。每个 episode 包含 `1000` 个 time slots，每个 time slot 时长为

$$
\Delta t=1\ \mathrm{s}.
$$

UE 按热点分布生成，默认包含 2 个半径为 100 m 的热点区域，热点 UE 概率为 0.85。该设置用于形成非均匀业务压力，使 UAV 轨迹控制和协作卸载之间的耦合更明显。

![图 1 双层协同优化框架](docs/figures/fig_framework.png)

图 1 展示了本文双层协同优化框架。上层轨迹控制器根据 UAV、UE 和请求状态输出 UAV 移动动作；下层请求级卸载器在服务请求到达时选择 local、cooperative 或 MBS 执行路径；环境根据轨迹、链路、缓存和卸载结果更新时延、能耗、队列压力和奖励。

### 2.2 UAV 运动模型

第 `t` 个 time slot 中，第 `u` 架 UAV 的三维位置记为

$$
\mathbf{p}_u(t)=\left[x_u(t),y_u(t),H\right],
$$

其中 `H=100 m` 为 UAV 飞行高度。上层策略输出二维连续动作

$$
\mathbf{a}_u(t)=\left[a_{u,x}(t),a_{u,y}(t)\right],
$$

该动作经过裁剪和缩放后映射为当前 time slot 的水平位移。若 UAV 最大飞行速度为 `v_{\max}=15 m/s`，则单步最大移动距离为

$$
d_u^{\max}=v_{\max}\Delta t.
$$

因此 UAV 位置更新可抽象表示为

$$
\mathbf{p}_u(t+1)=\Pi_{\mathcal{A}}\left(\mathbf{p}_u(t)+\Delta \mathbf{p}_u(t)\right),
$$

其中

$$
\|\Delta \mathbf{p}_u(t)\|_2 \leq v_{\max}\Delta t,
$$

`Π_A(·)` 表示区域边界投影或边界约束处理。UAV 必须满足飞行区域约束：

$$
0\leq x_u(t)\leq X_{\max},\quad 0\leq y_u(t)\leq Y_{\max},
$$

其中 `X_max=Y_max=700 m`。为避免 UAV 之间距离过近，系统还设置最小安全间距：

$$
\|\mathbf{p}_u(t)-\mathbf{p}_v(t)\|_2 \geq d_{\min},\quad \forall u\neq v,
$$

当前配置中 `d_min=200 m`。若发生边界违反或碰撞约束违反，环境会施加额外惩罚。

### 2.3 距离与链路模型

UE `m` 在 time slot `t` 的位置记为

$$
\mathbf{q}_m(t)=\left[x_m(t),y_m(t),0\right].
$$

UE-UAV 距离为

$$
d_{m,u}^{\mathrm{edge}}(t)=\|\mathbf{q}_m(t)-\mathbf{p}_u(t)\|_2.
$$

UAV-UAV 协作链路距离为

$$
d_{u,v}^{\mathrm{inter}}(t)=\|\mathbf{p}_u(t)-\mathbf{p}_v(t)\|_2.
$$

UAV-MBS 回传链路距离为

$$
d_u^{\mathrm{backhaul}}(t)=\|\mathbf{p}_u(t)-\mathbf{p}^{\mathrm{MBS}}\|_2.
$$

当 UE 与 UAV 的距离不超过 UAV 覆盖半径 `R=100 m` 时，UE 可由该 UAV 覆盖；当 UAV-UAV 距离处于感知范围 `R_sense=460 m` 内时，二者可作为潜在协作节点。系统考虑 UE-UAV、UAV-UAV 和 UAV-MBS 三类链路，其一般速率形式为

$$
R=B\log_2\left(1+\frac{P h}{\sigma^2}\right),
$$

其中 `B` 为链路带宽，`P` 为发射功率，`h` 为信道增益，`σ^2` 为噪声功率。当前配置中，UE-UAV edge bandwidth 为 `40 MHz`，UAV-UAV inter-UAV bandwidth 为 `20 MHz`，UAV-MBS backhaul bandwidth 为 `750 kHz`。受限 backhaul 设置用于避免所有请求无差别回退到 MBS，从而凸显本地 UAV 和协作 UAV 的作用。

![图 2 UAV 轨迹与覆盖机制示意](docs/figures/fig_trajectory_coverage.png)

图 2 从空间角度说明 UAV 轨迹对覆盖和协作机会的影响。UAV 运动会改变 UE 是否处于覆盖半径内、邻居 UAV 是否可达以及 MBS 回传路径质量。因此，上层轨迹控制并不只是移动问题，它直接影响下层卸载的候选集合。

### 2.4 服务请求与计算模型

服务请求记为

$$
r_i=\left(m_i,s_i,b_i,c_i,d_i,\rho_i\right),
$$

其中 `m_i` 表示发起请求的 UE，`s_i` 表示服务类型，`b_i` 表示输入数据大小，`c_i` 表示单位字节所需 CPU cycles，`d_i` 表示服务 deadline，`ρ_i` 表示请求优先级。当前配置中，请求输入大小在 `[1,5] MB` 范围内生成，服务文件大小在 `[1,5] MB` 范围内生成，服务 deadline 在 `[0.65,2.10] s` 范围内生成，priority 在 1 到 3 之间。

对任意候选执行节点 `n`，若其计算能力为 `F_n` cycles/s，则计算时延可写为

$$
T_{i,n}^{\mathrm{comp}}=\frac{b_i c_i}{F_n}.
$$

若请求需要先通过链路传输输入数据，则传输时延为

$$
T_{i}^{\mathrm{tx}}=\frac{b_i}{R}.
$$

服务总时延由传输时延、计算时延、必要的排队时延和服务获取开销共同构成。为便于描述，本文将候选执行路径集合写为

$$
\mathcal{C}_i=\{\mathrm{local},\mathrm{coop},\mathrm{mbs}\}.
$$

本地 UAV 执行时，请求由覆盖 UE 的 UAV `u` 处理，其总时延可表示为

$$
T_i^{\mathrm{local}}=
T_{i,m_i\rightarrow u}^{\mathrm{edge}}
+T_{i,u}^{\mathrm{queue}}
+T_{i,u}^{\mathrm{comp}}
+T_{i,u}^{\mathrm{cache}},
$$

其中 `T_cache` 表示服务缓存或服务文件相关开销。若服务已在本地 UAV 缓存，则该项较小；若未缓存，则可能需要额外获取服务文件。

协作 UAV 执行时，请求先到达本地 UAV，再转发至协作 UAV `v`，其总时延为

$$
T_i^{\mathrm{coop}}=
T_{i,m_i\rightarrow u}^{\mathrm{edge}}
+T_{i,u\rightarrow v}^{\mathrm{inter}}
+T_{i,v}^{\mathrm{queue}}
+T_{i,v}^{\mathrm{comp}}
+T_{i,v}^{\mathrm{cache}}.
$$

MBS 执行时，请求通过 UAV-MBS 回传链路回退至 MBS，抽象总时延为

$$
T_i^{\mathrm{mbs}}=
T_{i,m_i\rightarrow u}^{\mathrm{edge}}
+T_{i,u\rightarrow \mathrm{MBS}}^{\mathrm{backhaul}}
+T_{i,\mathrm{MBS}}^{\mathrm{queue}}
+T_{i,\mathrm{MBS}}^{\mathrm{comp}}.
$$

请求是否满足 deadline 由下式判断：

$$
\mathbb{I}_i^{\mathrm{succ}}=
\begin{cases}
1, & T_i \leq d_i,\\
0, & T_i > d_i.
\end{cases}
$$

### 2.5 能耗模型

系统能耗包含 UAV 运动能耗、悬停能耗、通信能耗和计算能耗。UAV 在一个 time slot 内的移动/悬停能耗可写为

$$
E_u^{\mathrm{move}}(t)=P_{\mathrm{move}}\cdot \frac{\|\Delta \mathbf{p}_u(t)\|_2}{v_{\max}},
$$

$$
E_u^{\mathrm{hover}}(t)=P_{\mathrm{hover}}\cdot \Delta t.
$$

当前配置中 `P_move=100 W`，`P_hover=80 W`。通信能耗可表示为

$$
E_i^{\mathrm{comm}}=P_{\mathrm{tx}}T_i^{\mathrm{tx}},
$$

其中 `P_tx=0.5 W`。计算能耗通常与 CPU cycles 和工作频率相关，可抽象为

$$
E_{i,n}^{\mathrm{comp}}=\kappa b_i c_i F_n^2,
$$

其中 `κ=10^{-27}` 为 CPU capacitance coefficient。总能耗指标由各 UAV、链路和请求执行路径上的能耗累加得到。

### 2.6 优化目标与评价指标

本文目标不是单一最小时延，而是在服务质量、能耗、公平性和 MBS 负载之间取得平衡。上层强化学习奖励由 fairness、latency、energy 和 offline rate 组成：

$$
r=
\alpha_3\log(\mathrm{fairness})
-\alpha_1\log(\mathrm{latency})
-\alpha_2\log(\mathrm{energy})
-\alpha_4\log(1+\mathrm{offline\_rate}).
$$

当前默认权重为

$$
\alpha_1=1.0,\quad
\alpha_2=0.4,\quad
\alpha_3=2.0,\quad
\alpha_4=50.0.
$$

最终 reward 乘以缩放因子

$$
\eta_r=0.01.
$$

若 UAV 发生碰撞约束或边界约束违反，则额外施加惩罚。本文主要评价指标包括 reward、latency、energy、DSR、fairness、local ratio、cooperative ratio、MBS ratio 和 MBS load ratio。

DSR 定义为满足 deadline 的服务请求比例：

$$
\mathrm{DSR}=\frac{\sum_i \mathbb{I}_i^{\mathrm{succ}}}{N_{\mathrm{served}}}.
$$

Jain fairness index 用于衡量 UE 服务覆盖或服务成功分布的均衡性：

$$
J(\mathbf{x})=\frac{\left(\sum_{m=1}^{M}x_m\right)^2}{M\sum_{m=1}^{M}x_m^2},
$$

其中 `x_m` 表示 UE `m` 的服务量或服务成功统计。

MBS ratio 表示已处理服务请求中流向 MBS 的比例：

$$
\mathrm{MBS\ ratio}=
\frac{N_{\mathrm{mbs}}}{N_{\mathrm{served}}}.
$$

MBS load ratio 表示 MBS offload 数量相对于所有生成服务请求的比例：

$$
\mathrm{MBS\ load\ ratio}=
\frac{N_{\mathrm{mbs}}}{N_{\mathrm{generated}}}.
$$

本文特别区分 MBS ratio 和 MBS load ratio。前者反映已处理请求内部的去向结构，后者反映 MBS 对全体业务流的实际承载压力。

## 第3章 双层协同优化框架

### 3.1 框架总体思想

本文将多无人机 MEC 控制问题拆分为上层轨迹控制和下层请求级卸载两个层次。上层在每个 time slot 为所有 UAV 生成移动动作，改变覆盖关系、链路质量和协作机会；下层在服务请求处理时进行局部卸载决策，改变请求执行位置、系统时延、能耗、队列压力和 MBS 负载。

这种双层设计并不意味着两个问题相互独立。相反，上层和下层通过环境状态紧密耦合。上层 UAV 轨迹决定 UE 是否被覆盖、哪些 UAV 可作为协作节点、回传链路是否拥塞；下层卸载策略决定请求是否在本地处理、是否转移至协作 UAV、是否回退到 MBS，并通过服务质量指标反馈给上层奖励。二者形成一种“上层改变机会集合、下层改变请求流向”的协同关系。

### 3.2 双层分解的必要性

若将轨迹控制和请求卸载全部并入单层动作，上层动作可能需要同时包含每架 UAV 的移动方向、移动距离、每个 UE 或每个请求的卸载位置以及资源分配比例。假设每个 time slot 内存在多个请求，则动作空间会随请求数量动态变化；对于多智能体强化学习，这会导致策略输出维度不稳定、训练样本效率下降和探索难度增加。

本文采用的分解方式保持上层动作空间固定。对于 5 架 UAV，上层策略每步输出 `5 x 2` 个连续动作；下层卸载器则在请求粒度上输出三分类结果：

$$
y_i\in\{\mathrm{local},\mathrm{coop},\mathrm{mbs}\}.
$$

因此，上层强化学习可以专注于多 UAV 协同覆盖和移动控制，下层请求级策略可以专注于 local、cooperative 和 MBS 三类执行路径之间的细粒度权衡。本文最新实现中，下层主策略采用 SC-OGO：先使用 oracle-guided surrogate classifier 给出候选卸载动作，再通过安全约束代价函数检查该动作是否会引入明显 deadline 风险或不必要的 MBS fallback。

### 3.3 在线执行流程

在每个 time slot 中，系统按如下流程运行。

1. 环境收集当前 UAV 位置、UE 位置、缓存状态、请求状态、邻居 UAV 信息和电量状态。
2. 上层 attention-MAPPO 根据每架 UAV 的局部观测输出二维移动动作。
3. 环境更新 UAV 位置、覆盖关系、邻居关系和链路状态。
4. 当服务请求需要处理时，下层 SC-OGO offloading policy 根据当前请求上下文输出执行位置；若 surrogate 预测动作不安全或代价明显高于其他合法动作，则使用安全约束重排序后的动作。
5. 环境根据执行路径计算通信时延、计算时延、队列压力、能耗和 deadline 是否满足。
6. 系统记录 reward、latency、energy、DSR、fairness、请求去向比例和 MBS load。

在线阶段，下层策略不读取未来请求，也不读取未来 UAV 轨迹。因此，该方法可以被解释为一种当前上下文驱动的请求级实时卸载策略。

## 第4章 上层 Attention-MAPPO 轨迹控制方法

### 4.1 多智能体建模

上层将每架 UAV 视为一个智能体。第 `u` 架 UAV 在 time slot `t` 的局部观测记为

$$
o_u(t)\in\mathcal{O}_u.
$$

观测包含 UAV 自身归一化位置、缓存状态、邻居 UAV 相对位置、关联 UE 相对位置、请求类型、请求大小、请求 id、deadline、priority 和电量状态等信息。所有 UAV 的联合观测为

$$
\mathbf{o}(t)=\left[o_1(t),o_2(t),\ldots,o_U(t)\right].
$$

第 `u` 架 UAV 的动作为

$$
a_u(t)\in\mathbb{R}^{2},
$$

联合动作为

$$
\mathbf{a}(t)=\left[a_1(t),a_2(t),\ldots,a_U(t)\right].
$$

策略函数写为

$$
\pi_\theta(\mathbf{a}(t)\mid \mathbf{o}(t))
=\prod_{u=1}^{U}\pi_{\theta_u}(a_u(t)\mid o_u(t)),
$$

其中集中训练阶段可以利用全局信息训练 critic，分散执行阶段每个 UAV 只根据自身观测输出动作。

### 4.2 注意力机制

在多 UAV 场景中，不同邻居 UAV 和不同关联 UE 对当前 UAV 的决策影响并不相同。简单拼接所有邻居信息会使观测维度变大，并且难以表达“哪些对象更重要”。因此，本文在 MAPPO 上层控制器中引入注意力机制。

设某个 UAV 的查询向量、键向量和值向量分别为

$$
Q=W_Q h,\quad K_j=W_K h_j,\quad V_j=W_V h_j,
$$

其中 `h` 为当前 UAV 表示，`h_j` 为邻居对象或关联对象表示。注意力权重为

$$
\beta_j=
\frac{\exp\left(QK_j^\top/\sqrt{d_k}\right)}
{\sum_{\ell}\exp\left(QK_\ell^\top/\sqrt{d_k}\right)}.
$$

聚合后的上下文表示为

$$
z=\sum_j \beta_j V_j.
$$

该表示随后输入 actor 和 critic 网络，使策略能够关注更相关的邻居 UAV、热点 UE 或高压力请求区域。当前配置中 attention hidden dim 为 64，attention heads 为 8。

### 4.3 MAPPO 目标函数

MAPPO 属于集中训练、分散执行范式。其 actor 更新采用 PPO clipped surrogate objective。设旧策略为 `π_{θ_old}`，新策略为 `π_θ`，概率比为

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}.
$$

则 clipped objective 为

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

当前配置中 PPO clip epsilon 为 `0.2`。优势函数采用 GAE 估计：

$$
\hat{A}_t=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},
$$

其中

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

当前配置中折扣因子 `γ=0.99`，GAE 参数 `λ=0.95`。critic 的价值损失为

$$
L^V(\phi)=
\mathbb{E}_t\left[
\left(V_\phi(s_t)-\hat{R}_t\right)^2
\right].
$$

综合熵正则项后，优化目标可写为

$$
L(\theta,\phi)=
L^{\mathrm{CLIP}}(\theta)
-c_v L^V(\phi)
c_e\mathcal{H}\left(\pi_\theta\right),
$$

其中 `c_e` 为 entropy coefficient，当前配置中 PPO entropy coefficient 为 `0.005`。

### 4.4 上层方法作用

上层 attention-MAPPO 的作用不是直接决定服务请求去哪执行，而是通过轨迹改变系统结构条件。若 UAV 能够更合理地靠近热点区域，则 UE-UAV 链路质量提升，服务请求更容易在本地或协作 UAV 上完成；若 UAV 间相对位置更合理，则协作链路更稳定；若轨迹控制改善覆盖公平性，则服务成功分布更均衡。因此，上层方法主要贡献体现为 reward、latency、DSR、fairness 和 energy 等系统级指标改善。

不过，仅优化轨迹并不必然降低 MBS 依赖。正式联合实验中，`attention_mappo__heuristic` 虽然显著改善 latency、DSR 和 fairness，但 MBS ratio 与 MBS load ratio 相对无协调启发式基线反而增加。这说明上层轨迹控制和下层卸载策略必须协同设计，否则上层策略可能在提升服务质量的同时增加中心节点 fallback 负担。

## 第5章 下层 SC-OGO 请求级任务卸载方法

### 5.1 请求级卸载问题定义

对于每个服务请求 `r_i`，下层策略需要在三类执行路径中选择一种：

$$
y_i\in\{\mathrm{local},\mathrm{coop},\mathrm{mbs}\}.
$$

其中 local 表示由当前覆盖 UE 的本地 UAV 执行，coop 表示转发给可达协作 UAV 执行，mbs 表示通过回传链路卸载到 MBS 执行。传统启发式方法通常偏向选择时延较小或规则优先级较高的路径，但这种方式难以同时表达 MBS 负载控制、本地队列压力和协作分担收益。

本文将下层卸载建模为 safety-constrained oracle-guided policy learning。训练阶段，增强 oracle 根据当前请求上下文计算三类候选路径的代价，并生成分类标签；在线阶段，轻量分类器先根据当前特征预测卸载类别，随后 SC-OGO 使用安全约束代价函数对预测动作进行检查和必要重排序。该方法接近 behavior cloning，但标签并非来自人类专家，也不是简单复制原始启发式规则，而是由面向系统目标设计的增强代价函数生成。SC-OGO 的额外作用是避免 surrogate 在少数边界状态中过度追求 MBS 减负或误选高 deadline 风险动作。

### 5.2 增强 Oracle 代价函数

对每个候选执行路径 `c`，首先计算其相对 deadline 时延：

$$
\rho_c=\frac{T_c}{d_i},\quad c\in\{\mathrm{local},\mathrm{coop},\mathrm{mbs}\}.
$$

若 `ρ_c <= 1`，说明候选路径预计能够满足 deadline；若 `ρ_c > 1`，说明候选路径存在 deadline 违约风险。增强 oracle 的代价函数定义为

$$
C_c=
\rho_c
+w_D\max(\rho_c-1,0)
+\mathbb{I}[c=\mathrm{local}]w_Q q_{\mathrm{local}}
+\mathbb{I}[c=\mathrm{mbs}]w_M
-\mathbb{I}[c=\mathrm{coop}]w_C q_{\mathrm{local}}.
$$

其中 `q_local` 为归一化本地队列压力，`w_D` 为 deadline 违约惩罚权重，`w_Q` 为本地队列压力权重，`w_M` 为 MBS fallback 代理惩罚权重，`w_C` 为协作队列缓解收益权重。当前配置中：

$$
w_D=1.25,\quad
w_Q=0.06,\quad
w_M=0.065,\quad
w_C=0.02.
$$

最终 oracle 标签为代价最小的候选路径：

$$
y_i^{\star}=\arg\min_{c\in\mathcal{C}_i} C_c.
$$

若当前不存在可用协作 UAV，则 cooperative 候选被置为不可选。该 oracle 不等价于最小时延规则：当 MBS 时延略优但 fallback 惩罚较高时，oracle 可能选择 local 或 cooperative；当 deadline 压力过大或本地队列过长时，oracle 仍允许 MBS 作为必要回退路径。

### 5.3 下层分类器训练目标

设下层分类器参数为 `ψ`，输入特征为 `x_i`，输出为三类概率分布：

$$
p_\psi(y\mid x_i)=
\mathrm{softmax}(f_\psi(x_i)).
$$

训练目标采用交叉熵损失：

$$
\mathcal{L}_{\mathrm{CE}}(\psi)=
-\frac{1}{N}\sum_{i=1}^{N}
\sum_{c\in\mathcal{C}}
\mathbb{I}[y_i^{\star}=c]\log p_\psi(c\mid x_i).
$$

下层特征包括候选执行路径相对 deadline 的时延、deadline tightness、priority、本地缓存命中、协作 UAV 是否可用、局部队列占用等。更丰富的 `rich_reduced_features` 还包含请求大小、服务文件大小、计算密度、邻居数量、链路速率、计算资源占用和邻居缓存置信度等信息。

![图 3 下层分类器质量](docs/figures/fig_offload_classifier_quality.png)

图 3 展示下层分类器与增强 oracle 标签的一致性。混淆矩阵用于观察三类标签是否被正确预测，calibration curve 用于观察模型置信度是否与真实准确率匹配。验证集 accuracy 和 macro-F1 约为 0.945，说明分类器能够较好拟合增强 oracle 的决策边界。

### 5.4 SC-OGO 在线安全约束重排序

仅使用 oracle-guided classifier 时，模型会近似增强 oracle 的离线标签，但在线系统中的队列、链路和协作可用性可能出现边界状态。为提升运行稳定性，SC-OGO 在分类器输出后增加一层轻量安全约束重排序。对于三类候选动作，定义在线代价：

$$
\tilde{C}_c=
\rho_c
+\lambda_D\max(\rho_c-1,0)
+\lambda_S\max(\rho_c-\rho_{\mathrm{safe}},0)
+\mathbb{I}[c=\mathrm{mbs}]\lambda_M
+\mathbb{I}[c=\mathrm{coop}]\lambda_C
+\mathbb{I}[c=\mathrm{local}]\lambda_Q q_{\mathrm{local}}.
$$

其中 `rho_safe` 是 deadline safety margin，`lambda_D` 表示 deadline 违约惩罚，`lambda_S` 表示接近 deadline 时的安全边际惩罚，`lambda_M` 表示 MBS fallback 代价，`lambda_C` 表示协作执行代价，`lambda_Q` 表示本地执行路径上的队列压力代价。队列项只作用于 local 候选，否则会对三类动作同加常数而无法改变在线重排序。当前默认配置为：

$$
\lambda_D=3.0,\quad
\lambda_S=0.45,\quad
\rho_{\mathrm{safe}}=0.90,\quad
\lambda_M=0.08,\quad
\lambda_Q=0.04,\quad
\lambda_C=0.04.
$$

在线推理时，SC-OGO 首先保留 surrogate classifier 的预测动作；若该动作非法、deadline ratio 超过硬阈值且存在更安全动作，或其代价高于最优合法动作超过 `SC_OGO_RERANK_TOLERANCE=0.02`，则改选在线代价最小的合法动作。该设计使方法仍保持轻量推理和可解释性，同时比单纯分类器更强调 DSR 安全约束。

### 5.5 下层方法作用

下层 SC-OGO 策略主要改变请求去向结构。它并不直接控制 UAV 轨迹，而是在当前轨迹和链路条件下决定请求是否由本地 UAV、协作 UAV 或 MBS 执行。若启发式策略过度依赖 MBS，则下层策略可以通过 MBS fallback 代理惩罚将部分请求重新分配至本地或协作 UAV，从而降低中心节点负载；若 surrogate 分类器在边界状态下给出高风险动作，则 SC-OGO 的安全约束重排序会优先保护 deadline 满足率。

![图 4 请求去向比例堆叠图](docs/figures/fig_request_destination_stack.png)

图 4 展示不同下层策略下 local、cooperative 和 MBS 三类请求去向比例。该图的意义在于说明 MBS load 降低的机制来源：请求不是消失了，而是从 MBS 回退路径转移到了 UAV 本地执行或 UAV 间协作执行路径。因此，在分析下层策略时，不能只报告 DSR 或 latency，还必须同时报告请求去向结构和 MBS load。

## 第6章 实验设计与结果分析

### 6.1 实验设置

默认仿真区域为 `700 m x 700 m`，系统包含 5 架 UAV、100 个 UE 和 1 个 MBS。每个 episode 包含 1000 个 time slots，每个 time slot 时长为 1 s。UAV 飞行高度为 100 m，最大速度为 15 m/s，覆盖半径为 100 m，感知范围为 460 m，最小 UAV 间距为 200 m。系统包含 25 类服务和 50 类内容文件，服务 deadline 在 `[0.65,2.10] s` 范围内生成。

| 参数 | 取值 |
| --- | ---: |
| UAV 数量 | 5 |
| UE 数量 | 100 |
| 区域大小 | `700 m x 700 m` |
| 每个 episode 步数 | 1000 |
| time slot 时长 | 1 s |
| UAV 高度 | 100 m |
| UAV 最大速度 | 15 m/s |
| UAV 覆盖半径 | 100 m |
| UAV 感知范围 | 460 m |
| 最小 UAV 间距 | 200 m |
| 服务数量 | 25 |
| 内容数量 | 50 |
| service deadline | `0.65-2.10 s` |
| edge bandwidth | 40 MHz |
| inter-UAV bandwidth | 20 MHz |
| backhaul bandwidth | 750 kHz |

主联合实验使用 10 个 workload seeds：

$$
\{42,84,126,168,210,252,294,336,378,420\}.
$$

每个 seed 运行 6 个 episodes，每个 episode 默认 1000 steps。统计单元为 seed-level episode mean。报告包含 mean、std、95% CI、paired delta、paired t-test 和 Wilcoxon 检验。由于 DSR、MBS ratio 和 MBS load ratio 是比例指标，报告中保留 ratio metric note。

### 6.2 对比方法

正式端到端联合实验采用四组组合：

1. `uncoordinated_greedy__heuristic`：无协调贪心轨迹控制 + 启发式卸载。
2. `attention_mappo__heuristic`：attention-MAPPO 轨迹控制 + 启发式卸载。
3. `uncoordinated_greedy__oracle_guided`：无协调贪心轨迹控制 + oracle-guided 卸载。
4. `attention_mappo__oracle_guided`：attention-MAPPO 轨迹控制 + oracle-guided 卸载。

该四组设计能够分别隔离上层收益、下层收益和完整双层收益。若只比较完整方法与一个单一基线，就无法判断性能提升来自轨迹控制还是来自任务卸载；四组联合消融则可以清楚分析二者的独立作用和组合效果。

### 6.3 四组联合消融主结果

数据来源：

- `results/joint_experiments/joint_four_way_journal/joint_experiment_summary.json`
- `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_uncoordinated_heuristic.md`
- `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_attention_heuristic.md`

| 组合 | Reward | Latency | Energy | DSR | Fairness | Local | Coop | MBS | MBS Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uncoordinated_greedy__heuristic` | -1432.39 | 1147588.82 | 124688169.38 | 41.80% | 0.7585 | 31.77% | 49.90% | 18.01% | 9.43% |
| `attention_mappo__heuristic` | -1218.30 | 949043.86 | 116501996.54 | 52.89% | 0.9434 | 30.72% | 43.73% | 25.16% | 15.19% |
| `uncoordinated_greedy__oracle_guided` | -1429.12 | 1148210.60 | 108502049.08 | 41.72% | 0.7586 | 54.69% | 40.20% | 4.79% | 2.57% |
| `attention_mappo__oracle_guided` | -1219.99 | 947602.60 | 101925121.62 | 52.83% | 0.9461 | 52.64% | 39.51% | 7.58% | 4.73% |

相对 `uncoordinated_greedy__heuristic`，`attention_mappo__heuristic` 的 reward 提升 214.1，latency 降低 17.30%，energy 降低 6.57%，DSR 提升 11.09 个百分点，fairness 提升 0.1848。这说明上层 attention-MAPPO 能显著改善轨迹控制和服务质量。但该组的 MBS ratio 和 MBS load ratio 分别增加 7.15 和 5.75 个百分点，说明仅优化上层轨迹可能使更多请求进入 MBS fallback。

相对 `uncoordinated_greedy__heuristic`，`uncoordinated_greedy__oracle_guided` 的 energy 降低 12.98%，MBS ratio 降低 73.43%，MBS load ratio 降低 72.73%，而 latency 仅增加 0.054%，DSR 下降 0.080 个百分点。这说明下层 oracle-guided 卸载器能够在基本保持服务质量的同时显著降低 MBS 依赖。

完整方法 `attention_mappo__oracle_guided` 同时继承两层优势。相对 `uncoordinated_greedy__heuristic`，完整方法 reward 提升 212.4，latency 降低 17.43%，energy 降低 18.26%，DSR 提升 11.02 个百分点，fairness 提升 0.1876，MBS ratio 降低 57.88%，MBS load ratio 降低 49.90%。这表明双层框架并不是简单追求单一 reward 最大化，而是在服务质量、能耗和中心节点减负之间取得更均衡的结果。

### 6.4 完整方法与仅上层方法对比

相对 `attention_mappo__heuristic`，完整方法 `attention_mappo__oracle_guided` 的 reward delta 为 -1.70，`p_t=0.8169`；DSR delta 为 -0.069 个百分点，`p_t=0.936`。二者均不显著。与此同时，完整方法 energy 降低 12.51%，MBS ratio 降低 69.86%，MBS load ratio 降低 68.87%，且 energy 与 MBS load 的 paired t-test 均显著。

这一结果是本文最适合强调的“双层协同”证据：下层 oracle-guided 卸载并没有破坏上层 attention-MAPPO 带来的 reward 和 DSR 收益，却显著缓解了 heuristic offloading 下 MBS 依赖升高的问题。因此，完整方法的优势不是单纯刷新 reward，而是在保持上层服务质量收益的同时显著压缩 MBS fallback。

### 6.5 下层在线性能与权衡分析

下层独立实验用于说明请求级卸载策略本身的机制。旧版本 oracle-guided 下层实验数据来源为：

- `results/full_offload_experiments/wpt_fix_thesis_run_offload/reports/runtime_offload_policy_statistics.md`

最新 SC-OGO / CQL-DQN / RADCC 对比数据来源为：

- `results/reports/runtime_offload_policy_comparison_sc_ogo.json`

旧版 oracle-guided classifier 的在线表现如下：

| 策略 | Latency | Deadline Satisfaction | MBS Load Ratio |
| --- | ---: | ---: | ---: |
| `heuristic_offloading` | 1477.33 | 24.64% | 8.41% |
| `surrogate_baseline` | 1477.64 | 24.47% | 4.88% |
| `rich_reduced_runtime_policy` | 1478.12 | 23.92% | 3.89% |

最新 10 seeds、每 seed 4 episodes、每 episode 100 steps 的运行时对比结果如下：

| 策略 | DSR | MBS Load | MBS Ratio | Energy | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `heuristic_offloading` | 25.673% | 8.958% | 33.303% | 111494.23 | 1447.82 |
| `surrogate_baseline` | 25.571% | 6.465% | 23.680% | 114892.92 | 1448.14 |
| `cql_dqn_offloading` | 24.616% | 7.131% | 27.462% | 96209.88 | 1448.81 |
| `radcc_offloading` | 24.718% | 6.767% | 25.381% | 124861.02 | 1448.60 |
| `sc_ogo_offloading` | 25.674% | 7.682% | 28.357% | 108650.45 | 1447.88 |

相对 `heuristic_offloading`，各方法的关键变化为：

| 策略 | DSR Delta | MBS Load Delta | MBS Ratio Delta | Energy Delta | Latency Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `surrogate_baseline` | -0.101 pp | -2.493 pp | -9.623 pp | +3398.69 | +0.32 |
| `cql_dqn_offloading` | -1.057 pp | -1.827 pp | -5.841 pp | -15284.34 | +0.99 |
| `radcc_offloading` | -0.954 pp | -2.191 pp | -7.922 pp | +13366.80 | +0.78 |
| `sc_ogo_offloading` | +0.002 pp | -1.275 pp | -4.946 pp | -2843.78 | +0.06 |

`surrogate_baseline` 的 MBS load 降低最明显，但 DSR 略降且能耗略升；`cql_dqn_offloading` 和 `radcc_offloading` 说明更复杂的深度强化学习式下层并不必然更适合作主方法，因为二者都带来接近 1 个百分点的 DSR 损失。`sc_ogo_offloading` 的减负幅度没有 surrogate 那么激进，但它几乎不牺牲 DSR，并同时降低 MBS load、MBS ratio 和 energy，更符合本文“守住 DSR，降低 MBS load”的约束优先级。因此，后续论文主方法建议采用 `attention_mappo__sc_ogo`，并将 `surrogate_baseline`、`cql_dqn_offloading` 和 `radcc_offloading` 作为下层对照或消融。

![图 5 下层卸载策略在线性能](docs/figures/fig_lower_runtime.png)

图 5 展示下层策略在线运行时的 latency、DSR、MBS ratio 和 MBS load ratio。若后续采用 SC-OGO 作为主方法，建议重新生成该图，将 `sc_ogo_offloading` 加入对比，并在图注中强调 SC-OGO 的核心优势是 DSR 安全性更稳，而不是单纯最大化 MBS load 降幅。

![图 6 MBS load 与 DSR 的 Pareto-style 权衡](docs/figures/fig_pareto_tradeoff_scatter.png)

图 6 展示不同策略和场景在 MBS load ratio 与 DSR 之间的权衡关系。横轴 MBS load ratio 越低越好，纵轴 DSR 越高越好。SC-OGO 若相对 heuristic 更靠左且纵向基本不下降，说明该策略将系统推向更低中心节点依赖区域，同时保留服务 deadline 安全性。需要注意，该图是 Pareto-style 可视化，不是严格多目标优化求得的理论 Pareto 前沿。

### 6.6 SC-OGO 组件消融与机制验证

为进一步回答“SC-OGO 为什么有效”，本文补充运行时组件消融。该实验固定同一个 surrogate classifier checkpoint，仅改变 SC-OGO 在线重排序项，因而主要用于解释在线安全约束层的机制来源。数据来源为：

- `results/reports/sc_ogo_component_ablation.json`

实验使用 5 个 workload seeds：`42,84,126,168,210`；每个 seed 运行 4 个 episodes，每个 episode 100 steps，并在默认运行时场景集合上统计。相对 `heuristic_offloading` 的结果如下：

| 策略 | Latency Delta | Energy Delta | DSR Delta | MBS Ratio Delta | MBS Load Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `classifier_only` | +0.3462 | +6494.37 | -0.130 pp | -9.650 pp | -2.507 pp |
| `sc_ogo_full` | +0.0586 | -2031.35 | -0.008 pp | -4.827 pp | -1.221 pp |
| `sc_ogo_no_mbs_penalty` | -0.0013 | -5285.67 | +0.029 pp | +0.448 pp | +0.111 pp |
| `sc_ogo_no_deadline_margin` | +0.0598 | -2003.38 | -0.010 pp | -4.867 pp | -1.229 pp |
| `sc_ogo_no_queue_pressure` | +0.0655 | -2011.16 | -0.006 pp | -4.949 pp | -1.252 pp |
| `sc_ogo_no_coop_term` | +0.0588 | +954.45 | -0.022 pp | -5.570 pp | -1.426 pp |

这组结果说明三点。第一，MBS fallback penalty 是压缩 MBS 依赖的核心机制：一旦移除 `SC_OGO_MBS_WEIGHT`，MBS ratio 和 MBS load 不再下降，甚至相对 heuristic 略有上升。第二，`classifier_only` 能更激进地降低 MBS load，但会带来能耗增加和 DSR 轻微下降；SC-OGO 的价值不是最大化减负幅度，而是在减负、能耗和 DSR 之间做更保守的安全重排序。第三，deadline margin、queue pressure 和 coop term 在当前运行时场景中的边际影响相对较小，其中 deadline margin 未被强烈触发，queue pressure 只产生轻微请求流向调整，coop term 更像协作路径的能耗/代价调节项。

因此，论文中最稳妥的机制表述是：**SC-OGO 的主要可验证机制是显式 MBS fallback penalty 压缩中心节点依赖，安全重排序用于避免 surrogate 分类器过度激进；deadline margin 和 queue/coop 项是辅助稳定项，不应被夸大为主要收益来源。**

### 6.7 尺度泛化与部署复杂度分析

为验证下层请求级策略在不同系统规模和链路条件下是否仍保持稳定趋势，本文补充一因子尺度泛化实验。该实验使用静态 UAV 位置与零移动动作，隔离下层 offloading policy 的泛化能力；若未来要验证完整 `attention_mappo + sc_ogo` 在不同 UAV 数量下的端到端泛化，仍需要重新训练或准备兼容的上层模型。数据来源为：

- `results/reports/scale_generalization_runtime.json`

尺度扫描包括 UAV 数量 `3/5/7`、UE 数量 `50/100/150`、热点数量 `1/2/3/4` 和 backhaul bandwidth `120 kHz/350 kHz/750 kHz/1.5 MHz/5 MHz`。相对 heuristic，SC-OGO 在 12 组尺度场景中的统计结论如下：

| 指标 | 结果 |
| --- | ---: |
| MBS load 降低场景数 | 12 / 12 |
| MBS ratio 降低场景数 | 12 / 12 |
| Energy 降低场景数 | 11 / 12 |
| DSR 非下降场景数 | 6 / 12 |
| DSR delta 最小值 | -0.094 pp |
| DSR delta 平均值 | -0.011 pp |

该结果说明 SC-OGO 的 MBS 减负趋势具有较好的跨规模一致性：无论改变 UAV 数量、UE 数量、热点数量还是 backhaul 条件，MBS ratio 与 MBS load 均下降，且 DSR 平均变化很小。需要注意，latency 在所有尺度场景中均没有下降，说明 SC-OGO 的目标不是降低单请求最小时延，而是以极小 latency 代价换取更低中心节点依赖。`backhaul=5 MHz` 场景中 SC-OGO 仍显著降低 MBS load，但 energy 上升，说明当 MBS 回传足够强时，过度压缩 MBS fallback 可能不再节能；这可以作为方法适用边界，而不是失败结果。

复杂度与部署分析数据来源为：

- `results/reports/complexity_deployment_analysis.md`
- `results/reports/complexity_deployment_analysis.json`

默认 `U=5` 时，上层轨迹控制每个 time slot 只输出 `2U=10` 维连续动作；下层请求级策略输出 local、cooperative 和 MBS 三分类结果。当前 surrogate checkpoint 包含 4931 个参数，每个请求约 4800 次 linear MAC。若一个 time slot 内有 `R_t` 个服务请求，单层枚举式三分类请求去向组合为 `3^{R_t}`，显式 UAV/MBS 执行节点枚举为 `(U+1)^{R_t}`；而本文框架保持上层动作接口固定，并将请求相关离散选择拆成每请求一次轻量推理。该分析支撑本文“可部署双层分解”的定位，但不应被写成全局最优复杂度证明。

### 6.8 Paired Seed 稳健性分析

正式联合实验采用同一批 workload seeds 进行 paired comparison，因此能够观察同一随机种子下不同策略的变化方向。相比只报告均值，paired seed 分析更能说明改善是否由个别随机种子偶然造成。

![图 7 同 seed 下 heuristic 到 oracle-guided 的变化](docs/figures/fig_paired_seed_slope.png)

图 7 中，每条线连接同一个 seed 下 heuristic 与 oracle-guided 的指标值。对于 MBS load 和 MBS ratio，线从左到右下降是有利变化；对于 DSR，线从左到右上升是有利变化。如果多数 seed 的 MBS load 和 MBS ratio 均下降，而 DSR 变化较小，则说明 oracle-guided 策略的减负效果具有跨 seed 一致性。

### 6.9 空间机制解释

多无人机 MEC 的一个重要特点是空间结构会影响服务路径。热点区域、UAV 轨迹和 MBS fallback 的空间分布可以帮助理解为什么某些请求更容易回退到 MBS，为什么协作 UAV 能够缓解局部压力。

![图 8 热点、轨迹与 MBS fallback 空间解释图](docs/figures/fig_uav_hotspot_fallback_heatmap.png)

图 8 展示 UAV 轨迹、热点区域和 MBS fallback 位置的空间关系。该图应作为空间解释图使用，而不应被解释为严格统计热力分析。其作用是帮助说明：当热点区域与 UAV 覆盖、协作链路或回传能力不匹配时，某些区域更容易出现 MBS fallback，因此下层卸载策略需要显式考虑中心节点减负。

## 第7章 总结与展望

### 7.1 总结

本文围绕多无人机移动边缘计算中的轨迹控制与任务卸载问题，提出了一种双层协同优化框架。该框架在上层采用 attention-MAPPO 进行多 UAV 轨迹控制，在下层采用 SC-OGO request-level policy learning 进行请求级卸载决策。上层通过注意力机制建模 UAV 间关系和服务压力差异，下层通过增强 oracle 将 deadline 违约、队列压力、MBS fallback 惩罚和协作缓解收益纳入标签生成过程，并在在线阶段通过安全约束重排序进一步控制 deadline 风险。

实验结果表明，上层 attention-MAPPO 能够显著改善协同轨迹控制、服务质量和覆盖公平性，但仅依靠上层方法可能增加 MBS fallback 依赖；下层 oracle-guided / SC-OGO 卸载能够降低 MBS ratio 和 MBS load ratio，并将更多请求转移至 local 或 cooperative 执行路径。最新下层在线对比显示，SC-OGO 相比 heuristic 在 DSR 基本不下降的前提下降低 MBS load、MBS ratio 和能耗，比 CQL-DQN 与 RADCC-Offload 更适合作为当前主方法。组件消融进一步表明，MBS fallback penalty 是 SC-OGO 压缩中心节点依赖的主要机制，而 deadline margin、queue pressure 和 coop term 更多体现为辅助稳定和代价调节。尺度泛化实验显示，SC-OGO 在 12 组 UAV/UE/热点/backhaul 变化场景中均降低 MBS ratio 和 MBS load，说明该请求级策略具有较稳定的减负趋势。复杂度分析则表明，双层分解能够保持上层 `2U` 连续动作接口固定，并将请求相关离散选择转化为轻量三分类推理。由此可见，轨迹控制与请求级卸载在多无人机 MEC 中具有明显互补性，双层学习式分解能够在动作空间可控、在线推理可部署和结果可解释之间取得较好平衡。

### 7.2 展望

后续工作可以从以下方向继续扩展。

第一，引入更强的下层学习策略。当前代码已经实现并初步测试了 Constrained CQL-DQN 与 RADCC-Offload，但现有结果表明更复杂的深度强化学习方法并不自动优于安全约束 surrogate。未来可以继续探索 offline RL、distributional RL 或多目标学习，但应以 DSR 安全性、MBS load 和能耗的在线系统指标为准，而不是仅以训练损失或分类准确率判断方法强弱。

第二，实现更紧密的双层联合训练。当前框架中，上层和下层通过环境反馈耦合，但训练过程仍以分阶段和组合验证为主。未来可研究在不显著扩大动作空间的前提下，使上层轨迹控制器感知下层卸载策略变化，从而形成更强的协同学习。

第三，扩展动态场景和真实业务分布。本文使用热点分布和多 seed workload 验证方法有效性，后续可引入更复杂的 UE 移动模型、突发业务、服务流行度漂移和非平稳信道条件。

第四，进一步统一缓存、能量和轨迹建模。当前系统已经包含缓存和能量相关参数，但论文主线聚焦轨迹控制与请求级卸载。未来可以将服务缓存更新、无线能量传输和电池约束纳入更完整的多时间尺度优化框架。

## 附录 A 图表说明与使用建议

### 图 A.1 `fig_framework.png`

![图 A.1 双层协同优化框架](docs/figures/fig_framework.png)

该图适合放在方法总览部分，用于说明上层 UAV 轨迹控制和下层请求级卸载之间的关系。正文中应强调两层通过链路、缓存、队列、时延、能耗和奖励反馈耦合。

### 图 A.2 `fig_training_convergence.png`

![图 A.2 上层训练收敛趋势](docs/figures/fig_training_convergence.png)

该图展示上层训练过程中的 reward、latency、energy 和 DSR rolling mean。它适合用于说明训练过程未明显发散，但不应作为最终性能优劣的主证据。最终性能结论应以同 seed 评估和四组联合消融为准。

### 图 A.3 `fig_lower_runtime.png`

![图 A.3 下层卸载策略在线性能](docs/figures/fig_lower_runtime.png)

该图用于展示 heuristic、oracle-guided 和 rich-reduced 等下层策略在 latency、DSR、MBS ratio 和 MBS load 上的差异。写作时建议强调 oracle-guided 的主要贡献是降低 MBS fallback 依赖。

### 图 A.4 `fig_mbs_tradeoff.png`

![图 A.4 MBS 减负与代价权衡](docs/figures/fig_mbs_tradeoff.png)

该图展示 MBS load 降低、DSR 损失和能耗代价之间的关系。它适合用于讨论“减负不是免费的”：若只报告 MBS load 下降，而不报告 DSR 和 energy，可能掩盖负载转移带来的代价。

### 图 A.5 `fig_seedwise_delta_ci.png`

![图 A.5 Seed-wise paired delta 与置信区间](docs/figures/fig_seedwise_delta_ci.png)

该图适合放在统计稳健性部分，用于说明相对 heuristic 的指标变化是否在 paired seed 统计上稳定。阅读时需要注意指标方向：latency 和 MBS load 越低越好，DSR 越高越好。

### 图 A.6 `fig_mbs_dsr_confidence_tradeoff.png`

![图 A.6 MBS load 降低与 DSR 损失置信区间](docs/figures/fig_mbs_dsr_confidence_tradeoff.png)

该图强调 MBS load 降低幅度和 DSR 损失的置信区间。若横向误差线整体远离 0，说明减负结论较稳；若纵向误差线接近 0，说明 DSR 损失较小或统计上不强。

### 图 A.7 `fig_pareto_tradeoff_scatter.png`

![图 A.7 MBS load 与 DSR 的 Pareto-style 权衡](docs/figures/fig_pareto_tradeoff_scatter.png)

该图适合作为主文权衡图。写作时建议称为 “Pareto-style tradeoff” 或“非支配边界可视化”，避免将其表述为严格理论 Pareto 前沿。

### 图 A.8 `fig_paired_seed_slope.png`

![图 A.8 同 seed 下 heuristic 到 oracle-guided 的变化](docs/figures/fig_paired_seed_slope.png)

该图直观展示同一随机种子下策略切换后的指标变化。若多数线条在 MBS load 和 MBS ratio 上下降，说明减负效果不是由单个 seed 偶然造成。

### 图 A.9 `fig_request_destination_stack.png`

![图 A.9 请求去向比例堆叠图](docs/figures/fig_request_destination_stack.png)

该图最适合解释 oracle-guided 策略“把请求转移到哪里”。如果 MBS 部分缩短，同时 local 或 cooperative 部分增加，说明 MBS load 下降来自请求流向结构调整。

### 图 A.10 `fig_offload_classifier_quality.png`

![图 A.10 下层分类器质量](docs/figures/fig_offload_classifier_quality.png)

该图用于证明 learned offloading policy 能够较好拟合增强 oracle 标签。它支撑“模型学得像 oracle”，但系统级收益仍应由在线实验和四组联合消融支撑。

### 图 A.11 `fig_scenario_sensitivity.png`

![图 A.11 场景敏感性分析](docs/figures/fig_scenario_sensitivity.png)

该图用于说明下层策略在不同业务压力场景中的表现。不同场景的绝对 DSR 不宜直接横向比较，更合适的是在同一场景内比较不同策略。

### 图 A.12 `fig_trajectory_coverage.png`

![图 A.12 UAV 轨迹与覆盖机制示意](docs/figures/fig_trajectory_coverage.png)

该图可放在系统模型或方法机制部分，用于解释 UAV 轨迹如何改变热点覆盖、邻居可达性和 MBS fallback 可能性。

### 图 A.13 `fig_uav_hotspot_fallback_heatmap.png`

![图 A.13 热点、轨迹与 MBS fallback 空间解释图](docs/figures/fig_uav_hotspot_fallback_heatmap.png)

该图可能混合真实 spatial trace 和配置示意，因此应谨慎使用。建议表述为“空间解释图”或“trace/示意辅助图”，不要将其写成严格统计热力分析。

### 图 A.14 `fig_hyperparameter_sensitivity_template.png`

![图 A.14 MBS penalty 敏感性分析](docs/figures/fig_hyperparameter_sensitivity_template.png)

虽然文件名保留 `template` 后缀，但当前内容已经由 MBS penalty 扫描结果生成。写论文时可以将图题改为 “MBS penalty sensitivity”，避免读者误解为空模板。

## 附录 B 数据来源与结果文件

本文当前版本保留以下正式数据来源，便于写论文时核对数值。

| 文件或目录 | 作用 |
| --- | --- |
| `results/joint_experiments/joint_four_way_journal/` | 正式四组端到端联合消融结果 |
| `results/joint_experiments/joint_four_way_journal/joint_experiment_summary.json` | 联合消融汇总 JSON |
| `results/joint_experiments/joint_four_way_journal/reports/` | 联合消融统计、验证和关键比较 |
| `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_uncoordinated_heuristic.md` | 相对无协调启发式基线的统计报告 |
| `results/joint_experiments/joint_four_way_journal/reports/joint_four_way_statistics_vs_attention_heuristic.md` | 相对仅上层 attention-MAPPO 的统计报告 |
| `results/full_runs/supplement_upper_multiseed/` | 上层多训练 seed 稳健性实验 |
| `results/full_runs/supplement_upper_multiseed/reports/upper_multiseed_statistics.md` | 上层多 seed 统计报告 |
| `results/full_offload_experiments/wpt_fix_thesis_run_offload/` | 下层卸载实验与分类器质量 |
| `results/full_offload_experiments/wpt_fix_thesis_run_offload/reports/runtime_offload_policy_statistics.md` | 下层在线运行统计报告 |
| `results/reports/runtime_offload_policy_comparison_sc_ogo.json` | SC-OGO、CQL-DQN、RADCC 与 heuristic/surrogate 的最新运行时对比 |
| `results/reports/sc_ogo_component_ablation.json` | SC-OGO 在线重排序组件消融，验证 MBS penalty、deadline margin、queue pressure 和 coop term 的作用 |
| `results/reports/scale_generalization_runtime.json` | UAV/UE/热点/backhaul 一因子尺度泛化实验 |
| `results/reports/complexity_deployment_analysis.md` | 双层框架复杂度与部署分析的论文文字报告 |
| `results/reports/complexity_deployment_analysis.json` | 双层框架复杂度与部署分析的结构化结果 |
| `results/reports/runtime_offload_policy_comparison_cql.json` | Constrained CQL-DQN 初始运行时对比 |
| `results/reports/runtime_offload_policy_comparison_radcc.json` | RADCC-Offload 初始运行时对比 |
| `results/full_offload_experiments/cql_sensitivity/` | CQL-DQN 下层敏感性实验 |
| `results/full_offload_experiments/radcc_sensitivity_coop/` | RADCC-Offload 协作代价敏感性实验 |
| `saved_offload_policies/offload_policy_cql.pt` | CQL-DQN 下层 checkpoint |
| `saved_offload_policies/offload_policy_radcc.pt` | RADCC-Offload 下层 checkpoint |
| `saved_offload_policies/offload_policy_surrogate_runtime.pt` | surrogate / SC-OGO 使用的分类器 checkpoint |
| `results/full_offload_experiments/mbs_penalty_sensitivity_summary.json` | MBS penalty 敏感性扫描汇总 |
| `docs/figures/` | 当前论文候选图表 |
| `run_sc_ogo_ablation.py` | 重跑 SC-OGO 组件消融 |
| `run_scale_generalization.py` | 重跑尺度泛化实验 |
| `analyze_complexity_deployment.py` | 生成复杂度与部署分析报告 |
| `run_supplement_experiments.py` | 期刊/论文补充实验调度入口 |
| `generate_thesis_figures.py` | 论文图表生成脚本 |

## 附录 C 复现实验命令

以下命令用于复现、抽查或在改动代码后重新生成结果，不是待办清单。当前 README2 使用的核心结果已经来自正式四组联合消融、上层多训练 seed、下层分类质量分析、SC-OGO 组件消融、尺度泛化、复杂度部署分析和 MBS penalty 敏感性扫描。

重新生成统计报告和图表：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all_thesis_figures.ps1
```

快速冒烟验证四组联合消融脚本：

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py `
  --phase joint `
  --smoke `
  --joint_name joint_four_way_journal_check
```

重跑正式四组联合消融：

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py --phase joint
```

重跑上层多训练 seed：

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py --phase upper
```

重跑 MBS penalty 敏感性扫描：

```powershell
.\.venv\Scripts\python.exe run_supplement_experiments.py --phase sensitivity
```

生成 CQL-DQN transition dataset：

```powershell
.\.venv\Scripts\python.exe collect_offload_dataset.py `
  --mode cql_transition_mixed `
  --output offload_datasets/offload_dataset_cql_transition.npz `
  --label_mode enhanced_oracle
```

训练 CQL-DQN 下层策略：

```powershell
.\.venv\Scripts\python.exe train_cql_offload_policy.py `
  --dataset offload_datasets/offload_dataset_cql_transition.npz `
  --output saved_offload_policies/offload_policy_cql.pt
```

训练 RADCC-Offload 下层策略：

```powershell
.\.venv\Scripts\python.exe collect_offload_dataset.py `
  --mode radcc_cost_mixed `
  --output offload_datasets/offload_dataset_radcc_cost.npz `
  --label_mode enhanced_oracle

.\.venv\Scripts\python.exe train_radcc_offload_policy.py `
  --dataset offload_datasets/offload_dataset_radcc_cost.npz `
  --output saved_offload_policies/offload_policy_radcc.pt
```

重跑最新下层运行时对比，包括 heuristic、surrogate、CQL-DQN、RADCC 和 SC-OGO：

```powershell
.\.venv\Scripts\python.exe compare_runtime_offload_policies.py `
  --surrogate_checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --rich_checkpoint saved_offload_policies/offload_policy_rich_runtime.pt `
  --cql_checkpoint saved_offload_policies/offload_policy_cql.pt `
  --radcc_checkpoint saved_offload_policies/offload_policy_radcc.pt `
  --sc_ogo_checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/runtime_offload_policy_comparison_sc_ogo.json `
  --seeds 42 84 126 168 210 252 294 336 378 420 `
  --episodes_per_seed 4 `
  --steps_per_episode 100
```

重跑 SC-OGO 组件消融：

```powershell
.\.venv\Scripts\python.exe run_sc_ogo_ablation.py `
  --surrogate_checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/sc_ogo_component_ablation.json `
  --seeds 42 84 126 168 210 `
  --episodes_per_seed 4 `
  --steps_per_episode 100
```

重跑尺度泛化实验：

```powershell
.\.venv\Scripts\python.exe run_scale_generalization.py `
  --surrogate_checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/scale_generalization_runtime.json `
  --seeds 42 84 126 168 `
  --episodes_per_seed 3 `
  --steps_per_episode 100
```

生成复杂度与部署分析：

```powershell
.\.venv\Scripts\python.exe analyze_complexity_deployment.py `
  --checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/complexity_deployment_analysis.md `
  --json_output results/reports/complexity_deployment_analysis.json
```

## 附录 D 写作口径与注意事项

1. 不建议把本文方法写成“全局最优端到端联合优化”。更准确的表述是“双层学习式协同优化框架”，其中上层和下层分别学习轨迹控制与请求级卸载。

2. 不建议只强调 reward 提升。完整方法相对 `attention_mappo__heuristic` 的 reward 和 DSR 并未显著提升，真正重要的结论是：在保持上层服务质量收益的同时显著降低 energy、MBS ratio 和 MBS load ratio。

3. 不建议将下层 oracle-guided 策略描述为简单最小时延规则。增强 oracle 中显式包含 deadline 违约、MBS fallback 惩罚、本地队列压力和协作缓解收益。

4. 若采用最新主方法口径，建议将下层写为 SC-OGO，而不是简单写成 CQL-DQN 或 RADCC。当前实验中 CQL-DQN 和 RADCC 都能降低一部分 MBS load，但 DSR 下降接近 1 个百分点，不符合“守住 DSR，降低 MBS load”的优先级；SC-OGO 的优势是 DSR 基本不降，同时降低 MBS load、MBS ratio 和 energy。

5. 组件消融中最强的机制证据是 MBS fallback penalty。移除该项后 MBS ratio 和 MBS load 优势基本消失，因此论文中应把它写成 SC-OGO 减负的主要来源；deadline margin、queue pressure 和 coop term 可写为辅助稳定或代价调节，不宜夸大。

6. 尺度泛化实验主要隔离下层请求级策略，使用静态 UAV 位置和零移动动作。它可以证明 SC-OGO 在不同规模下有稳定减负趋势，但不能替代 `attention_mappo + sc_ogo` 在不同 UAV 数量下的端到端重新训练实验。

7. 复杂度分析应作为部署可行性和动作空间分解的论据，而不是理论最优性证明。更稳妥的表述是：本文保持上层 `2U` 连续动作接口固定，并将请求相关离散决策转化为每请求一次轻量三分类推理。

8. 使用空间解释图时要谨慎。`fig_trajectory_coverage.png` 和 `fig_uav_hotspot_fallback_heatmap.png` 适合解释机制，不适合作为严格统计性能证据。

9. 使用比例指标时要注明统计单元。当前 DSR、MBS ratio 和 MBS load ratio 的统计单元为 seed-level episode mean，报告中保留 ratio metric note。

10. 投稿或毕业论文定稿前，需要人工补全 BibTeX，尤其是 UAV-MEC 综述、轨迹-卸载联合优化、服务放置/缓存、多智能体强化学习、MAPPO、attention mechanism、Jain fairness、imitation learning / behavior cloning、offline RL、CQL 和 distributional / risk-sensitive RL 相关文献。

## 附录 E 投稿前人工检查清单

1. 核对正文所有百分比、CI、p 值与 JSON/MD 报告一致。
2. 根据学校论文模板统一章节编号、图题、表题和公式编号。
3. 将 README 中的 Markdown 图片路径转换为论文模板中的正式插图路径。
4. 补全参考文献，并在第1章和相关工作中加入真实引用。
5. 根据目标篇幅压缩附录中的工程命令，仅在论文附录或实验复现说明中保留。
6. 若后续重新运行实验，需要同步更新主结果表、paired comparison 文本和图表说明。
