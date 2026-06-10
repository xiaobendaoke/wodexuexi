doi:10.19665/j.issn1001-2400.20251111

# 基于强化学习的多无人机协同 MEC任务卸载方案

尤 昕 阳 ， 李 旭 龙 ， 王 浩 彬 ， 皇 甫 伟

（北京科技大学 计算机与通信工程学院 ，北京 ）

摘要 ： 针对现有无人机辅助移动边缘计算系统中对用户公平性与任务成功率优化研究不足的问题 ， 构建了一个多无人机协助多物联网设备的空中边缘计算系统模型 ， 考虑三维轨迹设计以贴近实际部署环境在此基础上 ，提出以任务成功率与用户公平性为核心优化目标的在线决策框架 ，并采用基于柔性动作 评价的多无人机辅助计算任务卸载算法实现端到端策略学习。 该方法能够联合决策任务卸载比例、无 人 机选择 传输功率与算力分配 ，形成连续与离散混合动作结构 ； 同时通过熵调节与经验重放机制提升训练稳定性与样本效率 ，并在状态建模中融合频谱、信道、能量和位置信息 ，实现对动态可行域的自适应搜索。 实验结果表明 ，该方法在多无人机协作场景下能够显著提升系统任务执行成功率 ，并在用户间实现更高水平的公平性 相比基线方法 ， 所提方法在任务成功率方面平均提升约 ， 在公平性方面平均提升约

关键词 ： 移动边缘计算 ；无人机 ；任务卸载 ；用户公平性 ；任务成功率 ；强化学习

中图分类号 ： 文献标识码 ： 文章编号 ： （ ）

# ReinforcementlearningbasedmultiUAVcooperative MECtaskoffloadingscheme

Xinyang ，LI Xulong ，WANG Haobin ， HUANGFU Wei

（ ， ， ， ）2026 2 Feb．2026

Abstract： To address the limitations of user fairness and task success rate in existing unmanned aerial vehicle(UAV)-asisted mobile edge computing（MEC ） systems，this paper develops an aerial edge computing model with multiple UAVs assisting multiple IoT devices，considering threedimensional trajectory design to better capture practical deployment scenarios,on the basis of which we propose an online decision-making framework that optimizes the task success rate and user fairness,and design a multi-UAV task offloading algorithm based on soft actor-critic（SAC）to achieve end-to-end policy learning.The proposed method jointly determines task offloading ratio，UAV selection，transmission power，and computational resource allocation,thus forming a hybrid action structure with both continuous and discrete variables.Meanwhile,entropy regularization and experience replay are incorporated to enhance training stability and sample eficiency,while spectrum,channel,energy,and location informationare integrated into state modeling to enable adaptive exploration of the dynamic feasible domain.Simulation results demonstrate that the proposed approach significantly improves the task execution success rate and achieves a higher level ．19665／j．issn10012400．20251111

baseline methods reveals an average improvement of approximately 3.9% in the task success rate and an average improvement of approximately 4.O% in fairness.

Key Words ： （ ） ； （ ） ； ；fairness;task success rate; reinforcement learning

## 1 引 言

近年来，物联网（ ， ）的广泛部署和迅猛发展正在深刻改变人类社会的生产与生活方式［ ］ 。 随着智慧城市、车联网、工业互联网和远程医疗等应用的兴起 ，海量终端设备被接入网络 ，并产生持续增长的数据与计算需求［ ］ 。 这些应用往往对实时性、可靠性和智能化水平具有极高要求 ，例如自动驾驶需要毫秒级的响应延迟 ，智能工厂需要对复杂任务进行快速调度与执行。 然而， 受限于 终端自身的计算能力、存储空间以及电池能量 ，单靠本地处理已难以支撑这些日益复杂和多样化的应用需求［ ］ 。

为解决终端资源受限问题 ， 移动边缘计算（ ， ）应运而生。 将计算资源下沉至靠近用户的网络边缘 ，使得计算任务无需回传至远端云端即可在近端完成 ，从而显著降低了端到端时延并提高系统的可靠性［ ］ 。 凭借这些优势， 已逐渐成为面向超低时延和高可靠通信场景的重要支撑技术，并在标准化和系统架构方面取得了快速进展 ，为任务卸载与协同计算提供了工程化基础［ ］ 。

然而，单纯依赖传统 架构仍存在一定局限性 地面基站或接入点的部署往往受到地理条件和建设成本的约束 ，其覆盖范围和服务能力在面对动态场景时表现不足［ ］ 。 例如，在大规模突发事件、自然灾害、远程山区或临时性热 点 区 域 中 ， 地面基础设施可能受损或不足 ， 无法持续保证低时延与高可靠的计算服务［ ］ ，在沙漠、荒野等环境 中， 部署大量固定服务器并不现实也不经济［ ］ 。 这些问题在一定程度上限制了在大范围、多场景中的适用性。 为解决上述困境， 无人机（ ， ）的引入成为重要的研究方向。 具备灵活的三维机动能力和可快速部署的特性 ， 能够在无需预先建设固定基础设施的情况下，短时间内提供动态的通信与计算覆盖［ ］ 。 同时， 由于 通信链路通常具有良好的视距（ ， ）特性，可以显著减弱信道衰落和干扰 ， 从而提升数据传输的可靠性和速率［ ］ 。 更为重要的是 的飞行高度和位置可根据网络需求进行动态调整 使其具备天然的自适应性 能够弥补地面覆盖的不足［ ］

凭借这些优势，基于 的 架构（ ）在多种应用场景中展现出巨大的潜力。例如，在应急救援和灾害恢复中 ， 可以迅速部署并为受灾地区的 终端提供通信与计算支持［ ］ ；在1大型活动或人群聚集的动态热点场景中 ， 能够灵活调度以缓解基站过载［ ］ ； 在智慧农业和 环 境 监 测2中， 可结合边缘计算实现数据的近端处理 ， 减少对云端的依赖并提升实时性［ ］ 。 尽管 1 MEC已成为近年来的研究热点 但其实际应用仍面临动态网络拓扑 资源分配公平性以及能效约束等诸多挑战［ ］ 。 许多学者针对这些问题展开了深入研究 ，并提出了多种优化方案。

文献［ ］研究了包含单个无人机的 ，虽然单个无人机可以辅助 任务卸载，fairnesstasksuccessratereinforcementlearning但其覆盖面积和计算能力有限 ，难以满足大规模 设备的任务需求 ， 因此需要多个无人机协同工作以提4升系统整体性能 而且上述研究中无人机的轨迹都设定成高度不变 ，平面移动的形式，这并不完全符合真实情况，因为无人机通常是可以上下移动的。

文献［ ］研究了多无人机协助的 其中文献［ ］的目标是最小化时延 文献［ ］的目标是最小6化能耗，但上述研究都没有考虑用户公平性和任务成功率 ，但在应急救援、车联网安全、智慧城市、远程医疗和工业物联网等场景中 ，用户公平性保证了所有用户／设备的服务可及性与系统整体可靠性［ ］ ，而任务成功7 8率则是衡量 是 否 能 真 正 满 足 业 务 需 求 的 核 心 指 标［ ］ 。 举 例 来 说 ， 在 工 业 物 联 网（ ， ）与智慧农业场景中 ， 如果大规模工业生产或农田环境里只有部分传感器的任务能被处理 ，可能造成“监测盲区” ，导致生产风险或农作物损失。 而且任务失败可能导致关键数据丢9失，从而使预测和控制措施失效。 因此，同时优化任务成功率与用户公平性具有显著的现实意义。

10基于上述观察，面对有 协助的智慧农业、环境监测、工业互联网等在时间尺度上产生数据较均匀的场景，文中以优化任务成功率与用户公平性为核心目标 ，面向多 设备与多 协同卸载的空中边缘场景，提出一个在线决策框架 ，并采用强化学习方法进行端到端策略学习。 主要工作与贡献包括 ： ①提出了一个联合优化任务成功率与用户公平性的方法 ， 以解决多 协助的边缘计算场景中的轨迹优化与卸载决策问题。 ②将问题建模成一个最优化问题 ，以便用强化学习方法解决。 ③提出了一种基于柔性动作 评价（ ， ）算法的多无人机辅助计算任务卸载 （ ， ） 算法，同时输出卸载比例／目标 选择／功率与算力分配的连续 混合动作结构， 并通过熵调节与经验重放提升训练稳定性与样本效率 ，在状态构造上融合了频谱 信道 能量与位置等多模态信息 ，实现对时变可行域的自适应搜索

## 2 系统建模与问题描述

## 2．1 系统模型

## 系统组成

考虑如图 所示的多无人机协助 场景，该系统由 K 个无人机，I 个 设备，以及 个 地面服务器组成，整个区域呈正方形 ，边长为 A，随机部署着 I 个 设备，场景中的多个无人机会协同做出轨迹与卸载相关的决策 设备的集合表示为 $i \in \mathbf { I } { = } \{ 1 , 2 , \cdots , I \} , \mathrm { U A V }$ 的集合表示为 $k \in { \bf K } = \{ 1 , 2 , \cdots , K \}$ ，服务器用 s 表示，此外，时隙表示为 $n { \in } \mathbf { N } { = } \left\{ 1 , 2 , \cdots , N \right\}$ ，时隙长度为 $\Delta t _ { \circ } ~ \operatorname { I o T }$ 设备的位置为 ${ \boldsymbol { \varpi } } _ { i } = ( { \boldsymbol { \mathscr { x } } } _ { i }$ ，$y _ { i } ) , \mathrm { U A V }$ 的平面位置为 $q _ { k } \left[ n \right] = ( X _ { k } \left[ n \right] , Y _ { k } \left[ n \right] )$ ，高度为 $Z _ { k }  { [ n ] }$ ，服务器 s 的位置为 $\boldsymbol { \varpi } _ { s } = ( x _ { s } , y _ { s } )$ 。 因此 ，设备 i 与无人机 k 的距离为

$$
d _ { i k } [ n ] = ( \mathrm { ~  ~ | ~ } { \pmb q } _ { k } [ { n } ] - { \pmb w } _ { i } \mathrm { ~  ~ | ~ } ^ { 2 } + Z _ { k } [ { n } ] ^ { 2 } ) ^ { 1 / 2 }\tag{}
$$

无人机 k 与 服务器距离为

$$
d _ { k s }  { [ n ] } = (  { \Vert { \textbf { q } } _ { k }  { [ n ] } } - { \pmb { w } } _ { s }  { \Vert { \mathbf { \varphi } } ^ { 2 } } + Z _ { k }  { [ n ] } ^ { 2 } ) ^ { 1 / 2 }\tag{}
$$

<!-- image-->  
图 多无人机协助 系统模型

## 空地信道速率

假设 设备与无人机、 服务器与无人机之间的信道为莱斯信道 ，则 设备 i 与无人机 k 之间在第n 个时隙的信道增益 $h _ { i k } \left[ n \right]$ 可以表示为

$$
h _ { i k } \left[ n \right] = ( \rho d _ { i k } \left[ n \right] ^ { - \varepsilon } ) ^ { 1 / 2 } \Big ( \left( \frac { \beta } { \beta { + } 1 } \right) ^ { 1 / 2 } h _ { i k } ^ { \mathrm { L o S } } \left[ n \right] { + } \left( \frac { 1 } { \beta { + } 1 } \right) ^ { 1 / 2 } h _ { i k } ^ { \mathrm { N L o S } } \left[ n \right] \Big )\tag{}
$$

无人机 k 与 服务器 s 之间在第 $n$ 个时隙的信道增益 $h _ { k s } [ n _ { - } ^ { - }$ 可以表示为

$$
h _ { k s } \big [ n \big ] = ( \rho d _ { k s } \big [ n \big ] ^ { - \xi } ) ^ { 1 / 2 } \Big ( \Big ( \frac { \beta } { \beta { + } 1 } \Big ) ^ { 1 / 2 } h _ { k s } ^ { \mathrm { L o S } } \big [ n \big ] { + } \Big ( \frac { 1 } { \beta { + } 1 } \Big ) ^ { 1 / 2 } h _ { k s } ^ { \mathrm { N L o S } } \big [ n \big ] \Big )\tag{}
$$

其中 $, \rho$ 表示参考距离为 时的功率增益 $, \beta$ 表示莱斯因子 ， $. h _ { i k } ^ { \mathrm { L o S } }$ 与h  表示信道的视距（ ， ）分量，且满足 $| h _ { i k } ^ { \mathrm { L o S } } | = | h _ { k s } ^ { \mathrm { L o S } } | = 1 , h _ { i k } ^ { \mathrm { N L o S } }$ 和 h  表示信道的非视距 （ ， ） 分量， 且满足$h _ { i k } ^ { \mathrm { N L o S } } { \sim } C N ( 0 , 1 ) , h _ { k s } ^ { \mathrm { N L o S } } { \sim } C N ( 0 , 1 ) , \xi$ 表示链路路径损耗因子 是关于 $Z _ { k }  { [ n ] }$ 的单调递减函数，表达式如下 ：

$$
\hat { \varsigma } _ { ( } Z _ { k } \big [ \boldsymbol { n } _ { - } \big ] ) = \operatorname* { m a x } _ { ( } { \boldsymbol { \phi } } _ { 1 } - { \boldsymbol { \phi } } _ { 2 } ~ \big \vert \mathbf { g } _ { ( } Z _ { k } \big [ { \boldsymbol { n } } _ { - } \big ] ) ~ )\tag{}
$$

其中， $\phi _ { 1 } { > } 0 , \phi _ { 2 } { > } 0$ ，是基于信道测量的经验拟合参数 ， $\phi _ { 1 }$ 表示路径损耗的衰减速率系数 ，反映信号强度随无人机高度变化的对数衰减斜率 ， $\phi _ { 2 }$ 为环境补偿项 ，用于描述不同传播环境（如城区 郊区或空旷区域）下的常数偏移量 它们共同决定了路径损耗随高度变化的特性 ，通常通过仿真或实测信道数据进行参数拟合获得

由此可以得出 设备与 之间的传输速率为

$$
R _ { i k } \left[ n \right] = b _ { i k } \left[ n \right] \mathrm { l b } \left( 1 + \frac { P _ { i } \left[ n \right] h _ { i k } \left[ n \right] } { b _ { i k } \left[ n \right] N _ { 0 } } \right)\tag{}
$$

与 服务器之间的传输速率为

$$
R _ { k s } \left[ n \right] = b _ { k s } \left[ n \right] 1 \mathrm { b } \left( 1 + \frac { P _ { k } \left[ n \right] h _ { k s } \left[ n \right] } { N _ { 0 } b _ { k s } \left[ n \right] } \right)\tag{}
$$

其中 $, b _ { i k }$ 表示无人机 k 给 设备 i 分配的带宽， $, b _ { k s }$ 表示 服务器 s 给无人机 k 分配的带宽， $P _ { i }$ 和 $P _ { \textit { k } }$ 分别是 设备 i 和无人机 k 的发射功率， $N _ { 0 }$ 表示噪声的功率谱密度

## 传输和计算时延

首先定义 设备与无人机之间的连接情况为

$$
x _ { i k } \left[ n \right] = \left\{ 0 , 1 \right\}\tag{}
$$

当 $x _ { i k } [ n ] = 0$ 12时，表示 设备 i 与无人机 k 之间没有通信 ，当 $x _ { i k } \left[ n \right] = 1$ 时，表示 设备 i 与无人机 k 之间正在通信。

LoS LoS每个 设备在每个时隙都会产生一个任务 ，在时隙 n 产生的任务大小为 $L _ { i } [ n ]$ ，计算密度即处理 字LoS LoS节任务所需的 转数为 $c _ { i } \left[ n \right]$ NLoS NLoS， 其 能 接 受 的 最 大 时 延 为 $T _ { i } ^ { \mathrm { m a x } } \left[ n \right]$ 。 设 备 会 将 计 算 任 务 全 部 卸 载 到NLoS NLoS， 会根据情况做出任务切分决策 ，任务切分比例由 $\alpha _ { i k } \left[ n \right] \in \left[ 0 , 1 \right]$ 表示，有 $\alpha _ { i k } \left[ n \right] L _ { i } \left[ n \right]$ 字节的任务在 计算，剩余的 $( 1 - \alpha _ { i k } \left[ n \right] ) L _ { i } { \left[ n \right] }$ 1 2传输到 服务器。 因此对 设备 i 在第n 个时隙产生的任务，1 2设备到无人机 k 的传输时延 $T _ { i } ^ { \mathrm { U L } } [ n ]$ 可表示为

$$
T _ { i } ^ { \mathrm { U L } } [ n ] { = } \frac { L _ { i } [ n ] } { R _ { i k } [ n ] } \quad ,\tag{}
$$

无人机的计算时延 $T _ { i } ^ { C , \mathrm { U A V } } [ n ]$ 为

$$
T _ { i } ^ { C , \mathrm { U A V } } [ n ] { = } \frac { \alpha _ { i k } \left[ n \right] { L _ { i } \left[ n \right] } c _ { i } \left[ \bar { n } \right] } { f _ { k i } \left[ n \right] }\tag{}
$$

其中， $f _ { k i } \left[ n \right]$ 表示无人机 k 在时隙n 中分配给 设备 i 的 频率

无人机的传输剩余任务到 服务器的时延为

$$
T _ { i } ^ { \mathrm { B H } } [ n ] = \frac { ( 1 - \alpha _ { i k } \left[ n \right] ) L _ { i } \left[ n \right] } { R _ { k s } \left[ n \right] }\tag{3}
$$

MEC服务器 s 的计算时延为

$$
T _ { i } ^ { C , \mathrm { M E C } } [ n ] = \frac { ( 1 - \alpha _ { i k } \left[ n \right] ) L _ { i } \left[ n \right] c _ { i } \left[ n \right] } { f _ { s i } \left[ n \right] }\tag{4}
$$

其中， $f _ { s i } \left[ n \right]$ 1m表示 s 在时隙n 中分配给 设备 i 的 频率。

＝ ＝1因此处理该任务的总时延可以写成

$$
T _ { i } \big [ \boldsymbol { n } _ { \mathrm { 1 } } \big ] = T _ { i } ^ { \mathrm { U L } } \big [ \boldsymbol { n } _ { \mathrm { 2 } } \big ] + \operatorname* { m a x } \{ T _ { i } ^ { \mathrm { c , U A V } } \big [ \boldsymbol { n } _ { \mathrm { 2 } } \big ] , T _ { i } ^ { \mathrm { B H } } \big [ \boldsymbol { n } _ { \mathrm { 2 } } \big ] + T _ { i } ^ { \mathrm { c , u g c } } \big [ \boldsymbol { n } _ { \mathrm { 2 } } \big ] \}\tag{}
$$

总时延的计算中的无人机部分需要取无人机计算时延与无人机传输时延和 服务器计算时延的和的最大值，是因为一个计算任务在无人机环节被成功处理完毕的前提是其本地计算阶段和在 服务器上的计算阶段都必须完成。 取两者中的最大值 ，能够确保该任务的所有必要环节均已完成 ，从而准确表征任务的最终完成时刻。

## 2.1.4能耗

根据前述的时延可以得到能耗的表达式 设备 i 在时隙n 的总能耗等于发射能耗 表示为

$$
E _ { i } \big [ { n } \big ] { = } E _ { i } ^ { \mathrm { U L } } \big [ { n } \big ] { = } P _ { i } \big [ { n } \big ] T _ { i } ^ { \mathrm { U L } } \big [ { n } \big ]\tag{}
$$

其中， $\bar { P } _ { i } \big [ n _ { - } ^ { \bar { } }$ 指 设备的发射功率

无人机在时隙 n 的总能耗包括发射能耗 飞行能耗以及计算能耗 ：

$$
E _ { k } \left[ { n } \right] = E _ { k } ^ { \mathrm { B H } } \left[ { n } \right] + E _ { k } ^ { \mathrm { f l y } } \left[ { n } \right] + E _ { k } ^ { C } \left[ { n } \right]\tag{}
$$

无人机的发射能耗为

$$
E _ { k } ^ { \mathrm { B H } } [ n ] { = } P _ { k } [ n ] T _ { i } ^ { \mathrm { B H } } [ n ]\tag{}
$$

无人机的飞行能耗为

$$
E _ { k } ^ { \mathrm { f l y } } [ n ] = P _ { k } ^ { \mathrm { f l y } } [ n ] \Delta t \qquad ,\tag{}
$$

其中， $P _ { k } ^ { \mathrm { f l y } } [ n ]$ 是无人机的航空推进功率 ，计算式如下 ：

$$
P _ { k } ^ { \mathrm { f l y } } \big [ n \big ] = P _ { \circ } \Big ( 1 + \frac { 3 v _ { k } \big [ n \big ] ^ { 2 } } { U _ { \mathrm { t i p } } ^ { 2 } } \Big ) + \frac { P _ { 1 } v _ { 0 } } { v _ { k } \big [ n \big ] } + \frac { d _ { 0 } \rho s A v _ { k } \big [ n \big ] ^ { 3 } } { 2 }\tag{}
$$

其中， ${ \bf \nabla } , P _ { 0 }$ 和 $P _ { 1 }$ 分别表示无人机在悬停状态下的剖面功率和诱导功率 ${ \bf { \sigma } } _ { \mathrm { { s } } } U _ { \mathrm { { i p } } }$ 表示桨叶尖端速度 $\smash { \{ d _ { 0 } , s _ { 0 } , \upsilon _ { 0 } \setminus \rho }$ 和分别表示机身阻力比、旋翼紧致度、悬停时的平均旋翼诱导速度、空气密度以及旋翼盘面积 ， $\mathcal { V } _ { k }$ 是无人机的飞行速度，其表达式为

$$
\ v _ { k } \left[ n \right] = \frac { \parallel \ q _ { k } \left[ n \right] - q _ { k } \left[ n - 1 \right] \parallel } { \Delta t }\tag{}
$$

无人机的计算能耗为

$$
E _ { k } ^ { C } [ n ] = \sum _ { i } ^ { I } \alpha _ { i k } \left[ n \right] \kappa _ { k } f _ { k i } \left[ n \right] 2 T _ { i } ^ { C , \mathrm { U A V } } \left[ n \right]\tag{}
$$

其中 $, \kappa _ { k }$ 表示无人机的硬件系数。

服务器通常具有供电设施 ，电量充足，所以其计算能耗不计入在系统能耗内。

## 2．2 问题描述

BH fly为了最大限度提高计算任务处理成功率和系统公平性的同时 ，也尽可能控制时延， 引入 公平性指53标［ ］来衡量系统的公平性 即 ：

$$
F \lbrack n \rbrack = { \frac { \bigl ( \sum _ { i = 1 } ^ { I } \eta _ { i } \left. n \right. \bigr ) ^ { 2 } } { I \sum _ { i = 1 } ^ { I } \eta _ { i } \left. n \right. ^ { 2 } } }\tag{}
$$

fly其中， i［n］表示物联网设备 i 从时间槽 到时间槽 n 的计算任务处理成功率 ，可表示为

$$
\eta _ { i } [ n ] = \sum _ { j = 1 } ^ { n } { \frac { 1 ( T _ { i } [ j ] ) } { n } }\tag{}
$$

其中， $T _ { i } \left[ j \right] = ( T _ { i } \left[ j \right] \leqslant T _ { i } ^ { \operatorname* { m a x } } \left[ j \right] ) \land ( E _ { i } \left[ j \right] \leqslant E _ { i } ^ { \operatorname { r e m } } \left[ j \right] ) , E _ { i } ^ { \operatorname { r e m } }$ tip 0 0指 设备 i 的剩余电量。 （X）是指示函数 ， 当为真时取值为 ，否则取值为 。 公平性指标 $F [ n ] \in \left[ { \frac { 1 } { I } } , 1 \right]$ 描述了系统中所有物联网设备从开始到时间间隔 n 期间计算任务执行成功率的公平性。 该值越大，公平性越高。 当所有物联网设备的计算任务均以 的成功率执行时，即达到最佳公平性。

该优化问题需协同优化物联网设备的计算卸载策略与无人机的轨迹规划策略 ，可以写成

$$
\begin{array} { r l }  \left. \begin{array} { l l } { 1 } & { \gamma _ { 2 } \sigma _ { 2 } ^ { \prime } } & { \sum _ { j = 1 } ^ { \infty } \gamma _ { j = 1 } ^ { \infty } \gamma _ { j = 1 } \zeta _ { j } \zeta _ { j } \zeta _ { j } \zeta _ { j } \zeta _ { j } \right. = \frac { \zeta _ { j } ^ { \prime } } { 2 } \left. \begin{array} { l l } { 1 } & { \gamma _ { 2 } \sigma _ { 2 } ^ { \prime } } & { \gamma _ { 1 } \zeta _ { j } } \\ { 1 } & { \gamma _ { 2 } \sigma _ { 2 } ^ { \prime } } & { \gamma _ { 2 } \zeta _ { j } } \end{array} \right. } \\ & { \times \epsilon _ { \mathrm { L } } \epsilon _ { \sigma _ { 2 } } \epsilon _ { 2 } \epsilon _ { 3 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { } \epsilon _ { 1 } \epsilon _ { } \epsilon _ { 1 } \epsilon _ { } \epsilon _ { 2 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { 1 } \epsilon _ { } } \\ & { \times \epsilon _ { \mathrm { L } } \epsilon _ { \sigma _ { 2 } } ^ { \prime } } & { \sum _ { j = 1 } ^ { \infty } \epsilon _ { j = 1 } ^ { \infty } \epsilon _ { j = 1 } ^ { \infty } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } } \\ &  \times \epsilon _ { \sigma _ { 2 } } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _ { j } \epsilon _  j \end{array} \end{array}\tag{}
$$

其中， $\mathbf { \nabla } . \boldsymbol { b } _ { 0 }$ 和 $b _ { 1 }$ 是两个大于零的常数 ，V  和 $U _ { k } ^ { \mathrm { m a x } }$ 分别代表 在水平方向和垂直方向上的最大移动速度 ，$X = \{ x _ { i k } \lfloor n \rfloor , P _ { i } \lfloor n \rfloor , \} , \forall i \in { \bf I } , n \in { \bf N }$ 是物联网设备的计算任务卸载决策变量 $Q = \{ X _ { k } \left[ n \right] , Y _ { k } \left[ n \right] , Z _ { k } \left[ n \right]$ ，$\alpha _ { 1 k } [ n ] , \cdots , \alpha _ { l k } [ n ] , b _ { i k } [ n ] , b _ { k s } [ n ] , f _ { k , i } [ n ] \} , \forall n \in \mathbf { N } , k \in \mathbf { K }$ 是 的轨迹规划与资源分配决策变量 ； 和是连接约束，一个 设备一个时隙内最多连接一个 ，所有 在一个时隙内最多连接 M 个 设1 1备； 和 是发射功率约束， 设备和 的发射功率不能超过其自身最大限制 ； 和 是能耗约束， 设备和 的总能耗不能超过其电量 ； 和 是频谱约束，所有 设备上传给 的带宽和不能超过总带宽限制 ，所有 与 服务器通信时的带宽和也不能超过总带宽限制 ； 是计算频率约max束， 给每个任务分配的 频率总和不能超过其自身 频率； ～ 是 的轨迹限制，其1必须在规定范围内以不超限的速度上下左右移动1

## 13 基于SAC的多 UAV辅助计算任务卸载算法

## 3．1 算法选择

在多 协同的移动边缘计算场景中 ， 设备的任务卸载决策涉及用户选择、任务切分、功率与算1力分配等多个连续与离散变量的联合优化。 该问题在状态与动作空间上维度高、动态性强，且受信道波动和任务到达随机性影响， 传统确定性优化方法难以在复杂环境下获得稳健解 因此， 采用强化学习1（ ， ）方法，通过与环境的交互学习近似最优的卸载策略。

max然而，传统 以最大化期望累计奖励为目标 ，通常存在探索不足与策略收敛不稳定的问题。 在高动态max多 协同的 场景中，这类方法容易陷入局部最优 ， 使得任务成功率与用户公平性难以同时保证。为此，引入基于最大熵的强化学习框架 ，并选用 算法作为基础设计 算法作为求解工具。

max max在该框架下，策略不仅追求最大累计奖励 ，还会追求最大策略熵 ，从而鼓励探索更大的动作空间。 因此C30该框架可以做到提升探索能力 ，避免过早陷入次优解 ；在不确定环境中增强策略鲁棒性 ；改善样本效率与收C401敛速度。 在多 协同的 场景中，系统存在高度动态性和连续动作空间 ，例如无人机的三维轨迹控制 功率分配以及计算资源分配 ，均具有显著的随机性和时变性 若仅依赖传统确定性策略 ，容易陷入局部最优，导致部分用户长期无法获得公平资源或任务成功率下降 采用基于最大熵的强化学习框架 ，可以在训练过程中保持策略的探索能力 ，避免过拟合某些特定状态 ，从而在复杂场景下更好地平衡任务成功率与用户公平性，同时提高训练的稳定性与收敛速度。

## 3．2 问题重构

首先， 上述问题需要被重构为马尔可夫决策过程（ ， ） 。 在中 无人机通过确定自身位置 用户调度 发射功率及任务切分比例来实现系统任务成功率与用户公平性的优化 此外，系统环境在先前状态与历史操作的共同作用下会进入新的随机状态 在此情境下，任务卸载优化问题可建模为 。 典型 可定义为五元组 $< S , A , P , R , \gamma >$ ，其中 S 为状态空间，A 为动作空间 ， $P _ { ( } s [ n + 1 ] \mid s [ n ] , a [ n ] ,$ 表示执行动作 $a { \left[ n _ { - } \right] }$ 后从当前状态 $s [ n ] \in S$ 过渡到下一个状态 $s \left[ n + 1 \right] \in S$ 的概率，R 为奖励函数， $\gamma \in \left[ 0 , 1 \right]$ 表示折扣因子

在该系统中，状态空间由 I 个 设备，K 个无人机，以及它们所处的环境共同决定 因此，系统在第 n个时隙的状态可被定义为

$$
S [ \boldsymbol { n } ] = \{ q _ { 1 } [ \boldsymbol { n } ]  , \cdots , q _ { K } [ \boldsymbol { n } ] \} , \boldsymbol { Z } _ { 1 } [ \boldsymbol { n } ] , \cdots , \boldsymbol { Z } _ { K } [ \boldsymbol { n } ] , \boldsymbol { w } _ { 1 } [ \boldsymbol { n } ] , \cdots , \boldsymbol { w } _ { l } [ \boldsymbol { n } ] , \boldsymbol { L } _ { 1 } , \cdots , \boldsymbol { L } _ { l } , \boldsymbol { c } _ { 1 } , \cdots , \boldsymbol { c } _ { l } , \boldsymbol { \Psi } _ { l } [ [ \boldsymbol { n } ] , \boldsymbol { w } _ { 1 } [ \boldsymbol { n } ] , \boldsymbol { w } _ { 2 } [ \boldsymbol { n } ] ] \} ,
$$

$$
E _ { 1 } ^ { \mathrm { r e m } } [ n ] , \cdots , E _ { I } ^ { \mathrm { r e m } } [ n ] , E _ { 1 } ^ { \mathrm { r e m } } [ n ] , \cdots , E _ { K } ^ { \mathrm { r e m } } [ n ] , \eta _ { 1 } [ n ] , \cdots , \eta _ { I } [ n ] \}\tag{}
$$

其中， $E _ { i } ^ { \mathrm { r e m } } [ n _ { - }$ 和 $E _ { k } ^ { \mathrm { r e m } } [ n$ ］分别表示 设备 i 和无人机 k 在时隙n 时的剩余电量。

物联网设备需就其与无人机的连接情况以及传输功率做出决策 无人机则需决定其飞行位移，计算任务卸载比率，自身 频率，自身发射功率。 此外， 服务器需要确定分配给每个任务的 频率。 因此，在时隙 n 内， 设备 i 的动作可以定义为

$$
A _ { i }  { [ n ] } = \{ \mathbf { x } _ { i k }  { [ \mathrm { n } ] } , P _ { i }  { [ n ] } \}\tag{}
$$

无人机 k 的动作可以定义为

$$
A _ { \iota } \left[ \boldsymbol { n } \right] = \langle \alpha _ { \iota \iota } \left[ \boldsymbol { n } \right] , \cdots , \alpha _ { \iota \iota } \left[ \boldsymbol { n } \right] , P _ { \iota } \left[ \boldsymbol { n } \right] , \Delta X _ { \iota } \left[ \boldsymbol { n } \right] , \Delta Y _ { \iota } \left[ \boldsymbol { n } \right] , \Delta Z _ { \iota } \left[ \boldsymbol { n } \right] , f _ { \iota \iota } , \cdots , f _ { \iota \iota } \rangle\tag{}
$$

服务器 s 的动作可以定义为

$$
A _ { s } [ n ] { = } \{ f _ { 1 s } , \cdots , f _ { I s } \}\tag{}
$$

整个系统在时隙 n 内的动作可以表示为

$$
A [ n ] { = } \{ A _ { 1 } [ n ] , \cdots , A _ { I } [ n ] , A _ { 1 } [ n ] , \cdots , A _ { K } [ n ] , A _ { s } [ n ] \}\tag{}
$$

最后，奖励函数用于评估智能体在给定观测下所采取行动的表现 根据优化问题，在时隙n 内 设备i的奖励函数可以写成

$$
r _ { i } \big [ n \big ] { = } 1 ( T _ { i } \big [ n \big ] ) { - } b _ { 0 } \ \frac { T _ { i } \big [ n \big ] } { T _ { i } ^ { \mathrm { m a x } } \big [ n \big ] } { + } \frac { b _ { 1 } } { I } F \big [ n \big ]\tag{}
$$

全部无人机的奖励函数可以定义为

$$
r _ { \mathrm { U A V } } \big [ n \big ] = \sum _ { i = 1 } ^ { I } \Big [ 1 ( T _ { i } \big [ n \big ] ) - b _ { 0 } \ \frac { T _ { i } \big [ n \big ] } { T _ { i } ^ { \mathrm { m a x } } \big [ n \big ] } \Big ] + b _ { 1 } F \big [ n \big ] - k \big [ n \big ] p\tag{}
$$

其中， $\boldsymbol { \phi }$ 1 1是无人机飞出目标区域时的惩罚值 ， $k \lbrack n$ 1 1 1］是时间槽 n 内飞出目标区域的无人机数量。

## 3．3 算法设计

rem rem文中设计的 算法可分为两个组件 ，如图 可见 ：策略网络和评论家网络 ，评价网络用于拟合动作价值函数。 在训练时，以系统状态 $s _ { t }$ 和联合动作 $\boldsymbol { a } _ { t }$ 为输入，输出为系统状态 st 时采取动作 $\boldsymbol { a } _ { t }$ 后的动作价值，即累计折扣奖励。 由于动作价值函数可能被过估计 ，且过估计值会累积 ，所以评价网络包括两个主评价网络 ${ Q _ { \varphi _ { 1 } } } \mathrm { ~ } \cdot { Q _ { \varphi _ { 2 } } }$ ，网络参数分别为 $\varphi _ { 1 } \ldots \varphi _ { 2 }$ ，其主要作用是评估当前策略的价值 ，在训练过程会中选择 Q值较小的一个，避免动作价值函数被高估的问题 ，提高算法的稳定性。 另外有两个目标评价网络 $Q _ { \varphi _ { 1 } ^ { ' } } \ldots Q _ { \varphi _ { 2 } ^ { ' } }$ ，网络参数分别为 $\varphi _ { \mathrm { ~ 1 ~ } \cdot \varphi _ { \mathrm { ~ 2 ~ } } ^ { \prime } } ^ { \prime }$ ，其作用则是提供一个稳定的价值估计 ，用于计算主评论家网络 Q的回归目标，降低训练过程MarkovDecisionProcess MDP UAVassisted1中的不稳定 性 与 方 差。 值 得 一 提 的 是， 文中采用集中式训练与分布式执行 （MEC ， ）的强化学习框架。 在该框架下，评论家网络用于学习全局状态 动作价值函数，其输入为整个系统的联合状态与联合动作 ， 而非单个无人机的局部状态。 因此， 评论家网络的数量与数量无直接对应关系 策略网络根据移动设备的本地观测信息生成策略 ，其以系统状态 S 为输入，输出动作 $a _ { t }$ 每个智能体模型有一个策略网络 $\pi _ { \theta }$ ，网络参数为 θ 其目标函数为

$$
y _ { j } = r _ { j } + \gamma [ \operatorname* { m i n } _ { k \in \{ 1 , 2 \} } Q _ { \overline { { \varphi } } _ { k } } ( s _ { \ j } ^ { \prime } , a _ { \ j } ^ { \prime } ) - \varepsilon \mathrm { l o g } \pi _ { \theta } ( a _ { \ j } ^ { \prime } | s _ { \ j } ^ { \prime } ) ]\tag{}
$$

其中， $r _ { j }$ 表示即时奖励， $Q _ { \varphi _ { b } } \ : ( s ^ { \prime } { } _ { j } \ : , a ^ { \prime } { } _ { j }$ ）表示动作价值 ， $\log _ { \varphi } ( \boldsymbol { a } _ { \textit { j } } ^ { \prime } | \boldsymbol { s } _ { \textit { j } } ^ { \prime } )$ 表示负对数策略熵 ε 为温度系数 用于平衡奖励与熵。 熵值越大，说明策略在动作选择上越分散 ， 探索性更强； 熵值越小， 则策略更倾向于确定性选择，利用性更强。 在 框架中，策略优化通过最大化包含熵正则项的期望累计奖励来实现 ，从而在“高收益”与“高探索”之间取得平衡。 具体而言，策略网络以系统状态为输入 ，通过最小化目标函数更新参数以提升动作价值并保持策略分布的多样性 与此同时 评论家网络通过最小化均方误差来逼近真实的动作价值函数 二者交替迭代更新 直至收敛 该过程确保了策略在复杂动态环境下的稳定学习 并实现了对任务成功率与用户公平性的联合最优。

<!-- image-->  
图 算法结构

文中算法可通过算法 进行描述 初始阶段 智能体从环境中收集信息以构建状态空间 在将状态输入到网络之前，需要预处理其初始状态 $s _ { t }$ ，也就是将其归一化 ，既能消除不同数据维度间的数量级差异 ，又能提升算法性能与稳定性 ，文中采用文献［ ］中的归一化方法进行归一化。 获得归一化后的初始状态 $\hat { \boldsymbol s } _ { t }$ 后，将当前状态输入策略演员网络 ，该网络利用参数化深度神经网络（ ， ）输出动作1 MEC $a _ { t }$ 。同时，当执行动作 $a _ { t }$ 时，环境状态从 st 转换为 $s _ { t + 1 }$ 。 获取下一个状态 $s _ { t + }$ 和奖励 rt 后，将元组 $\langle s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } \rangle$ 存储于回放缓冲区 B 中 通过随机抽取元组打破样本间相关性来更新网络参数 当从回放缓冲区 B 随机选取一批元组更新网络时 ，评论家（Q函数）的参数会进行梯度下降更新

算法1 算法

输入 ：状态维度 $d _ { s }$ 、动作维度 $d _ { a }$ 、折扣因子 γ、温度系数初值 ε、软更新系数 τ、批大小 N、探索步数阈值$T _ { \mathrm { e x p } l }$ 、学习率 $\eta _ { \pi } , \eta _ { Q }$ 、重放缓存容量 C

V输出 ：训练后的策略参数 θ、评论家参数 $\varphi _ { 1 }$ 和 $\varphi _ { i }$

初始化

＝①初始化可重参数化策略网络 $\pi _ { \theta } ( a \vert s )$ ；评论家 $Q _ { \varphi _ { 1 } } , Q _ { \varphi }$ l

②复制目标网络 $Q _ { \varphi _ { 1 } } {  } Q _ { \varphi _ { 1 } ^ { ' } } , Q _ { \varphi _ { \varphi } } {  } Q _ { \varphi ^ { ' } }$

③初始化缓存 B 容量C，设目标熵 $H _ { \mathrm { { t a r g e t } } }$ －，并令 ε 为可学习参数

训练循环（直至达到最大步数或收敛） ：

步骤一 ：环境交互与经验收集

④观测当前状态 st

⑤若 $t { < } T _ { \mathrm { e x p } l }$ ，以均匀随机动作 $a _ { t }$ 探索；否则从策略 $\pi _ { \theta }$ 采样at（重参数化采样，并得到 $\log \pi _ { \theta } ( a _ { t } \mid s _ { t } ) )$

⑥在环境中执行 $a _ { t }$ ，获得 $( r _ { t } , s _ { t + 1 } )$ ，并将 $( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } )$ 存入 B

步骤二 ：小批量采样

⑦若 $| B | \geqslant N$ ，从 B 均匀采样 $\{ ( s _ { j } , a _ { j } , r _ { j } , s ^ { ' } { } _ { j } ) \} _ { j = } ^ { N }$

步骤三 ：目标值计算

⑧用当前策略在下一个状态上采样 $\alpha _ { \ j } ^ { \prime } { \sim } _ { \pi _ { \theta } } ( \ \cdot \ | \ s _ { \ j } ^ { \prime } )$ ，并计算 $\log { \pi _ { \theta } ( a _ { \textit { j } } ^ { \prime } | s _ { \textit { j } } ^ { \prime } ) }$

⑨以目标网络计算目标值

步骤四 ：评论家（Q函数）更新

⑩基于均方误差最小化两路评论家损失 ：

$$
L _ { Q _ { m } } = \frac { 1 } { N } { \sum _ { j = 1 } ^ { N } } ( { Q _ { \varphi _ { m } } } ( s _ { j } , a _ { j } ) - y _ { j } ) ^ { 2 } , m = 1 , 2 \qquad \circ\tag{}
$$

 11分别对 $\varphi _ { 1 } \bullet \varphi _ { 2 }$ 进行一次梯度下降更新（学习率 $\eta _ { Q } )$

步骤五 ：策略更新

 12在当前状态批上，用策略的重参数化样本 $a _ { j } \sim _ { \pi _ { \theta } } ( \ \bullet \ | \ s _ { j } )$ 与其 $\log \pi _ { \theta } ( a _ { j } \mid s _ { j } )$ ） 计 算 ：

$$
L _ { \pi } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left( \alpha \mathrm { l o g } \pi _ { \theta } ( a _ { j } \ \mid \ s _ { j } ) - \operatorname * { m i n } _ { m \in \left\{ 1 , 2 \right\} } Q _ { \varphi _ { m } } \left( s _ { j } , a _ { j } \right) \right)\tag{}
$$

 13对 θ 做一次梯度下降更新（学习率 $\eta _ { \pi }$ ）

步骤六 ：温度系数自适应

 14通过计算

$$
\underset { \varepsilon } { \mathrm { a r g m i n } } L _ { \varepsilon } = - \frac { 1 } { N } { \sum _ { j = 1 } ^ { N } } \log \varepsilon ( \log \pi _ { \theta } ( a _ { j } \mid s _ { j } ) + H _ { \mathrm { t a r g e t } } )\tag{}
$$

更新 ε

步骤七 ：目标网络软更新

 15对 $m = 1 , 2$ 执行软更新 ：

$$
\varphi ^ { \prime } { } _ { m } { \gets } ( 1 { - } \tau ) \varphi ^ { \prime } { } _ { m } { + } \tau \varphi ^ { \prime } { } _ { m }\tag{}
$$

步骤 $\bar { n }$ ：循环与终止

 16令 $t {  } t { + } 1$ ，返回步骤一；达到终止条件后输出 $\pi _ { \theta }$ 和 ${ Q _ { \varphi _ { 1 } } } , { Q _ { \varphi _ { 2 } } }$

## 3．4 输出映射

的动作输出落数值会落在［ ， ］ ，但在实际应用中，需要把它映射到任务卸载环境里的真实物理范围（如功率分配 算力分配 卸载比例 无人机位置等） 采用了线性缩放与 函数实现映射，线性缩531放用于将连续动作值映射到实际执行范围 ，确保动作可行性； 函数则用于生成动作概率分布， 促进1 2策略探索的多样性和稳定性 ，二者结合保障了算法在连续动作空间中的有效学习和收敛。 线性缩放的表达式可以写成以下形式 ：

$$
a _ { \mathrm { r e a l } } = \frac { a _ { \mathrm { n o r m } } + 1 } { 2 } ( a _ { \mathrm { m a x } } - a _ { \mathrm { m i n } } ) + a _ { \mathrm { m i n } } \qquad ,\tag{}
$$

其中， $\smash { a _ { \mathrm { n o r m } } \in [ - 1 , 1 ] }$ 表示 网络输出的动作， $a _ { \mathrm { m i n } }$ 和 $a _ { \mathrm { m a x } }$ 表示该动作在物理场景下允许的最小值和最大值 $, a _ { \mathrm { r e a l } }$ log表示映射到实际环境中的真实动作。 线性缩放可以用于计算发射功率、算力分配、任务卸载比例等物理量值里。

关于 设备与无人机的连接情况 ， 的动作输出通过一个 函数转化为每个 被选中的概率，然后取概率最大的 作为最终连接对象 如果动作落在“未连接”类别，就表示该 设备在当前时隙不选择任何 。

lo的位置由球坐标系从 的［ ， ］输出映射到真实的三维位置。 具体来说， 给出的归一化动作向量 $\pm [ - 1 , 1 ] ^ { 3 }$ ，会被分别映射为球坐标的半径、方位角与极角／仰角， 然后再转换为笛卡尔坐标系。 半径、方位角、极角／仰角都可以通过上文介绍的线性缩放获得。

## 4 实验结果与分析

在本节中，所有实验均在一台配置为 ＠ 的计算机上运行 ，搭建仿真环境所使用的编程语言是 版本为 使用的仿真软件为 本节通过平均累积奖励 任务连接成功率与公平性指数 个指标，验证所提出的 方法在多 协同的辅助计算场景 中 的 有 效 性。 选用的对比算法包括双延迟深度确定性策略梯度算法 （， ） 、近端策略优化算法（ ， ）和随机算法，所有算法的超参数均经过调优以获得最佳性能表现 ，以确保实验结果的公平性和可靠性 具体实验关键参数如表 所示 另外， 设备的位置采用随机均匀分布 每次实验运行前 ，会在指定的正方形区域内随机生成所有 设备的坐标，以模拟设备部署的不确定性 ，并确保实验结果具有统计显著性

表1 实验关键参数
<table><tr><td>参数</td><td>数值</td></tr><tr><td>区域边长A/km</td><td>1</td></tr><tr><td>无人机接收总带宽  $B _ { \mathrm { U L } } / \mathrm { M H z }$ </td><td>6.5</td></tr><tr><td>无人机发射总带宽  $B _ { \mathrm { B H } } / \mathrm { M H z }$ </td><td>6.5</td></tr><tr><td>时隙长度  $\varDelta t / \mathrm { m s }$ </td><td>40</td></tr><tr><td>IoT发射功率P范围/W</td><td>[0.01,0.1]</td></tr><tr><td>无人机高度限制/m</td><td>[50,120]</td></tr><tr><td>无人机初始位置</td><td>(0,0,85)</td></tr><tr><td>无人机硬件系数κ</td><td>10-28</td></tr><tr><td>任务大小  $L _ { i } / \mathrm { M B }$ </td><td>[0.2,1]</td></tr><tr><td>计算密度  $C _ { i } / ( \mathrm { c y c } \cdot \mathrm { b i t } ^ { - 1 } )$ </td><td>[500,2 000]</td></tr><tr><td>任务可接受最大时延  $T _ { i } ^ { \mathrm { m a x } } / \mathrm { m s }$ </td><td>[20,40]</td></tr><tr><td>莱斯因子  $\beta / \mathrm { d B }$ </td><td>10</td></tr></table>

首先观察当无人机数量为 个， 设备数量为 个的情况 算法的总体表现如图 所示 随着训练推进， 的平均累计奖励在早期即出现陡峭增长 ，在 轮后进入平台期并在整段稳态区间内保持最高水平； 的上升速度与最终平台值次之 ； 的收敛速度与平台值均低于前两者 ，而随机策略长期徘徊在较低区间 值得注意的是， 的阴影带始终较窄 ，尤其在中后期显著小于其他方法 ，表明其跨种子的方差更小、学习过程更稳定。 该现象与算法机制一致 ： 最大熵策略在复杂的连续 混合动作空间中提升了有效探索覆盖 而双评论家与最小化目标在价值估计上抑制了过估计 从而在样本效率与稳定性上同时1 MEC受 益 。

<!-- image-->  
图 总体表现

任务连接成功率的对比结果见图 。 在约 轮后成功率即可超过 ，并在平台期稳定在 左右 最终稳定在约 ， 收敛至约 ～ 区间， 而随机策略则维持在约 ～该结果反映了在受限的“传输 计算”可行域内 策略需要在链路质量 功率与算力分配之间实现细粒度的协调。 的策略熵与连续动作解耦 ，使其在时变可行域中更易找到“传输 计算”的协调点，从而提升成功率。 实验结果表明， 能显著提升关键业务可达性 ，对时延敏感场景更具实用价值。

<!-- image-->  
图 任务成功率

公平性指数的对比如图 。 的公平性曲线最高且波动最小 ，其平台期约为 ～ ；与 分别在约 与约 附近收敛；随机策略仅维持在 左右波动。 公平性的提升来源于奖励构造中对负载均衡的显式约束与连续 混合动作的联合优化。 在相同的资源预算下 ， 倾向于抑制个别 过载并避免空载 ，从而在不显著牺牲总效用的情况下获得更均衡的吞吐配置 由于公平性直接影响系统在多用户共存与高可靠业务中的服务质量一致性 ，上述结果意味着所提方法在工程落地中能够以稳定的方式兼顾总体效用与用户级别体验 可以看出， 在保证总体效用提升的同时 ，能够维持更均衡的多 负载分配，这对于时延敏感和可靠性要求较高的连接业务尤为关键。

<!-- image-->  
图 用户公平性

4 MUSAC 1000 0．75从 项指标的一致性可见 ， 在样本效率、最终性能与训练稳定性方面形成互相印证的优势 ：一方面 其在不足一半训练轮次时已逼近最终平台值 显示出更高的样本利用效率 另一方面 收敛后的波动幅0．55度更小，跨种子表现更集中 ，说明价值网络与策略更新的耦合更为平稳 ；此外，在平均累计奖励、成功率与公平性上同时领先 表明该方法在多目标优化场景中具备更好的整体统筹能力 相较之下 受限于裁剪－近似与较高的梯度方差 ，收敛速度与平台值均受影响 ； 在最终平台值上接近 ，但在早期样本效率与跨种子稳定性上略逊一筹。

图 展示了在 在不同 数量下的规模扩展效果 可以看出，当 数量增加时，平均累计奖励指标效果也随之上升 因为随着 数量增加 系统可用于辅助 设备卸载的计算与中继资源同步增多， 节点在决策时拥有更多可选链路与算力组合 ，任务更易成功传输并完成 同时，各曲线在训练过程中均平稳趋稳 ，未因状态／动作空间扩大而出现方差放大或震荡加剧 ，体现出学习过程的鲁棒性与稳定 性 。

<!-- image-->  
图 无人机数量变化时的收益演化

综合图 ～图 ，在统一训练预算与相同环境条件下 ， 在平均累计奖励、任务连接成功率与公平性 项核心指标上均明显优于 、 与随机策略； 同时， 其训练方差更小、收敛更快、平台值更高，体现出良好的样本效率与稳定性。 随着 数量增加，系统效用持续、平稳提升，进一步验证了 在协同计算场景中的可扩展性与实际部署潜力 ， 为后续在更大规模集群与更复杂业务需求下的推广奠定了基础。

## 5 结束语

文中提出的 算法在 辅助的移动边缘计算场景中展现出显著优势 通过与及随机策略的对比实验 ，结果表明 在平均累计奖励、任务连接成功率与资源分配公平性 方面均明显优于基线方法 ，并在训练收敛速度与稳定性上表现突出。 具体而言，该算法在较少训练轮次下即可获得接近最优的性能水平 ，体现出较强的样本效率 ；在连续动作与高随机性的 连接环境中， 能够有效平衡探索与利用，抑制过估计问题；在多 场景下，其性能随系统规模平稳提升 ，进一步验证了方法的鲁棒性与可扩展性。 综合来看， 不仅提升了任务成功率与系统效用 ，还兼顾了公平性和可扩展性 ，展示了在 协同计算中的广阔应用潜力。1

## 参考文献 ：

[1 ] WANG D,BAI Y,HAUNG G,et al. Cache-Aided MEC for IoT:Resource Allcation Using Dee Graph Reinforcement［ ］ ， ， （ ） ：

[2 ]NGUYEN T V,DAO NN,NOH W,et al. User-Aware and Flexible Proactive Caching Using LSTM and Ensemble Learning in IoT-MEC Networks[J].IEEE Internet of Things Journal,2021,9(5):3251-3269.

[3 ]HU H,SONG W,WANG Q,et al.Energy Eficiency and Delay Tradeof in an MEC-Enabled Mobile IoT Network[J]. IEEE Internet of Things Journal,2022,9(17):15942-15956.

[4 ]LI X,QIN Y,HUOJ,et al.Computation Ofloading and Trajectory Planning of Multi-UAV-Enabled MEC:A Knowledge Assisted Multiagent Reinforcement Learning Approach[J].IEEE Transactions on Vehicular Technology,2023,73(5): 7077-7088.

[5]XUY,ZHANG H,JI H,et al.Transaction Throughput Optimization for Integrated Blockchain and MEC System in IT[J]. IEEE Transactions on Wireless Communications,2021,21(2):1022-1036.

[6 ]HOSSAIN M A, HOSSAIN A R,ANSARI N.Numerology-Capable UAV-MEC for Future Generation Massive IoT［ ］ ， ， （ ） ：

［ ］ ， ， ，      ［ ］Transactions on Wireless Communications,2021,20(12):7712-7727.

[8 ] LU W,DING Y,GAO Y,et al. Secure NOMA-Based UAV-MEC Network Towards a Flying Eavesdropper[J].IEEE Transactions on Communications,2022,70(5):3364-3376.

[9 ] PERVEZ F,SULTANA A,YANG C,et al. Energy and Latency Eficient Joint Communication and Computation Optimization ina Multi-UAV-Asisted MEC Network[J].IEEE Transactions on WirelessCommunications,2023,23(3): 1728-1741.

[10 ]ZHANG Y,KUANG Z,FENG Y,et al.Task Offloading and Trajectory Optimization for Secure Communications in Dynamic User Multi-UAV MEC Systems[J].IEEE Transactions on Mobile Computing,2024.

[11]ABRAR M,AJMAL U,ALMOHAIMEED Z M,et al.Energy Eficient UAV-Enabled Mobile Edge Computing for IoT： ［ ］ ， ， ：

［ ］ ， ， ，     ［ ］， ， （ ） ：

[13 ]KHURSHID T,AHMED W,REHAN M,et al.A DRL Strategy for Optimal Resource Alocation Along with 3D Trajectory Dynamics in UAV-MEC Network[J]. IEEE Access,2023,11:54664-54678.

[14]LIU K,ZHENG J.UAV Trajectory Optimization for Time-Constrained Data Collction in UAV-Enabled Environmental Monitoring Systems[J].IEEE Internet of Things Journal,2022,9(23):24300-24314.

[15 ]HE Y,XIANG K,CAO X,et al.Task Scheduling and Trajectory Optimization Based on Fairness and Communication Security for Multi-UAV-MEC System[J].IEEE Internet of Things Journal,2024,11(19):30510-30523.

[16 ]WANG D,TIAN J,ZHANG H,et al.Task Offloading and Trajectory Scheduling for UAV-Enabled MEC Networks:An Optimal Transport Theory Perspective[J].IEEE Wireless Communications Lettrs,2021,11(1):150-154.

[17 ]LIU B,WAN Y,ZHOU F,et al.Resource Allocation and Trajectory Design for MISO UAV-Asisted MEC Networks[J].－ － ， ， （ ） ：

[18] ZHENG G,XU C,WEN M,et al.Service Caching Based Aerial Cooperative Computing and Resource Allcation in Multi-UAV Enabled MEC Systems[J].IEEE Transactions on Vehicular Technology,2022,71(10):10934-10947.

[19 ]HE Y,GAN Y,CUI H,et al.Fairness-Based 3-D Multi-UAV Trajectory Optimization in Multi-UAV-Assisted MEC［ ］ ， ， （ ） ：

[20 ]WANG S,SONG X,SONG T,et al.Fairness-Aware Computation Ofloading with Trajectory Optimization and Phase－     ［ ］  ， ， （ ） ：20547-20561.

[21]WANG X,GURSOY M C,ERPEK T,et al. Learning-Based UAV Path Planning for Data Colection with Integrated－ －［ ］ ， ， （ ） ：

[22 ]JAIN R K,CHIU D M W,HAWE W R.A Quantitative Measure of Fairness and Discrimination[J]. Eastern Research－， ， ， ， ， （ ） ：

7 XU Y ZHANG T LIU Y etal．UAVAssisted MEC Networks with Aerialand GroundCooperation J ．IEEE［ ］ ， ， ，TransactionsonWirelessCommunications20212012 77127727．－ －［ ］ ， ， （ ）：

－（ 编辑 ： 牛姗姗 ）