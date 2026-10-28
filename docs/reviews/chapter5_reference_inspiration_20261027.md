# 第五章参考文献启发评议

日期：2026-10-27。

阅读对象：当前 `latex/docs/chap05.tex`、第一章相关工作、现有参考文献表，以及相邻“参考文献”目录中的44份Markdown笔记（18篇英文、26篇中文）。本次逐份筛查了笔记的方法与启发部分，对关键文献进一步回查对应PDF的摘要和相关方法页；不表示已逐页精读44篇原文或核验全部数值、证明与实验。末表的P表示回查过原文相关内容，M表示本次仅据笔记作初筛。

本次只做文献评议和轻量文本核验，未修改论文正文、算法与封存结果，未运行科研统计或新实验。以下新增公式与实验方案均为本次分析建议，不是原论文结论，也不是当前方法已经实现的功能。

**后续执行补充（2026-10-27）**：编制详细方案时进一步对照历史试验，发现缓存敏感边、冷暖混合、条件校准、同池重排、零价格及额外fetch特征已有开发诊断；旧PPO预算截断与多方法选择器也已有未过门槛的记录。下文的“建议诊断”应先复用这些历史材料，不能理解为全部尚未开展。尚需区分的是当前解析方法的长回合分布、受控组件比较与新场景/预算策略。具体去重和执行顺序以 [详细后续方案](../plans/chapter5_followup_20261027.md) 第1、3、6—9节为准；历史五块快照结果不能替代当前20块解析方法的机制证明。

## 1. 总体判断

这些文献能够启发第五章，而且有几篇与当前方法非常接近。最有价值的是：帮助界定“学习、匹配、定价各自解决什么问题”，以及找出当前证据尚未回答的机制问题。

第五章现有主线已经明确：冻结需求感知Attention-MAPPO上层，下层以缓存命中/未命中混合评分、固定配额容量槽指派、有限候选和能耗价格组织任务，并将完整软件耗时加入当前服务评价后继续状态演化。新20块确认支持相对G和二值价格策略的计费DSR收益。这条主线值得保留。

文献带来的重点不宜是继续增加算法名词，而应是以下三个问题：

1. **信息价值**：缓存信念的精确程度什么时候会改变分配并提高实际服务成功率？
2. **计算价值**：何时值得多做一次匹配，何时应直接执行已有方案？
3. **资源取舍**：服务收益中有多少依赖更多能耗、MBS资源或特定算力共享假设？

## 2. 最值得优先回看的文献

| 文献编号 | 已核对的核心思路 | 对当前第五章的具体意义 | 使用边界 |
|---|---|---|---|
| E17 TJCCT | 短尺度以价格激励分配算力、用匹配确定卸载，长尺度优化轨迹 | 是“定价＋匹配＋轨迹分层”的重要近邻工作，应补进研究定位 | 原文价格是用户与服务器交易的资源单价；本章价格是候选选择中的能耗惩罚系数 |
| E01 AAV matching-enabled DRL | SAC处理移动与功率，TMA处理用户关联 | 支持把结构明确的离散子问题从RL动作空间中分离 | 用户关联匹配与本章已接纳请求的执行目标指派不同；第一章已讨论这篇，不能当全新遗漏 |
| C13 张秋平 | 情景感知组合多臂赌博机处理服务缓存，偏好双边匹配处理卸载 | 若以后恢复下层学习，可把学习放在未知需求/缓存收益上，让确定性匹配继续承担合法分配 | 原文缓存放置动作不是本章缓存信念估计；稳定匹配不等于最大化按时完成数 |
| C12 贾子晔 | 用历史数据构造任务大小分布的不确定集，优化最差分布下时延 | 启发缓存信念误差与公开预测误差的鲁棒性诊断 | 原文不确定对象是任务大小，不能说其已解决本章缓存信念校准 |
| C25 李云服务编排 | 用LSTM捕获请求的时间依赖，在部分观测下联合编排与资源分配 | 若缓存/需求有可利用的历史规律，学习历史信息可能比再次学习配额更有针对性 | 增加记忆网络是否有效须看数据和费用；不能宣称LSTM消除了部分观测问题 |
| E11 Lyapunov MAPPO负载均衡 | 把卸载、跨机迁移、资源分配、轨迹分成不同子问题；迁移用动态优先级 | 启发当前最多八个配额迁移提议的排序与候选覆盖诊断 | 当前是无持续任务队列的批次模型，不能照搬队列积压或稳定性结论 |
| E05 Energy and Makespan | 用负载、截止期、能量等构造位置优先级，并用自适应触发设计轨迹 | “按需启动优化”的思想可启发匹配预算随任务状态变化 | 原文触发对象是轨迹；下文的匹配停止规则是迁移建议，原文没有替本章验证 |
| E15 Multi-Objective | 同时考虑时延、UAV能耗与卸载量，采用标量化和分块求解 | 启发质量、能耗、外部资源的联合呈现，以及均分CPU假设的对照 | 原文实际采用线性加权；算力子问题还使用二分算法，不能照笔记写成无迭代闭式解 |
| C26 李小平异构调度 | 分离节点排序、任务排序、卸载、调度、调优，并在开发实例上比较组件 | 对“收益究竟来自哪个组件”的实验设计很有启发 | 不应直接把既有20块确认改造成调参集，也不能机械套ANOVA忽略配对和重复观测 |
| C16 徐渊博准入控制 | 通信阈值、UAV接收阈值和生灭过程；UAV预处理后继续交给基站 | 启发检查MBS拥塞、准入与接纳后完成质量的关系 | 与本章自然覆盖接纳及完整分配不同，引入准入会改变问题及分母结构 |

## 3. 最有价值的具体启发

### 3.1 先分析“缓存信念在哪里有用”，再考虑增加学习网络

这是我认为最值得优先做的补充。第五章已有结果显示，主候选计费DSR为33.76188%，冷启动半概率为33.78529%，二者配对差区间跨零；现有证据不能确认精确缓存信念不可替代。这里只复述正文封存值，没有重新统计。

可以直接从本章评分式解释为什么这件事值得研究。给定同一个配额、名义软件费用与固定分支时延，令

\[
I_h=\mathbf1\{T^{hit}_{mj}(N_j)+\delta_0\le D_m\},\qquad
I_m=\mathbf1\{T^{miss}_{mj}(N_j)+\delta_0\le D_m\}.
\]

则

\[
s_{mj}=I_m+p_{mj}(I_h-I_m).
\]

在有效分解满足命中时延不大于未命中时延的条件下，只有三种情况：

| 分支状态 | 分数 | 精确信念在该固定边评分中的作用 |
|---|---:|---|
| 两个分支都按时 | 1 | 不影响分数 |
| 两个分支都超时 | 0 | 不影响分数 |
| 命中按时、未命中超时 | p | 直接影响分数 |

因此，一个可检验解释是：**如果真正会因命中状态而跨越截止期的候选边较少，或这些边很少改变最终匹配，精确信念就未必带来可见收益。** 这目前只是解释假设，尚无占比或归因结果。

建议后续在服务器上分层诊断：

- 检查允许协作边中“命中按时、未命中超时”的比例，以及它们进入最终分配的比例。
- 区分信念数值变化、匹配身份变化、配额变化与实际按时完成数变化。
- 对全部可观测缓存反馈评估信念校准；若只能观察被选边的命中，需承认选择偏差。
- 在独立开发/诊断数据上比较半概率、先验、历史估计、校准估计和人为误差扰动。理想真实缓存只能作明确标注的信息增强诊断，不能混入正式公开信息方法。
- 固定同一快照的分配变化只能说明局部机制；实际收益仍需计费闭环比较。

由C12的不确定集思想还可提出一个轻量变体：对固定分支的信念区间 \(p\in[\underline p,\overline p]\)，使用

\[
s^{rob}=\min_{p\in[\underline p,\overline p]}[I_m+p(I_h-I_m)].
\]

在上述单调分支下，关键边取下界概率即可。它只是固定时延、固定候选边上的保守评分，不是整个系统DRO求解；区间须有开发数据依据。C25、C13更适合作为后续学习历史需求或缓存收益的依据，而不是证明再加一个PPO必然更好。

### 3.2 把固定匹配上限扩展为“是否值得继续计算”的决策

本章已把完整调用时间计入服务结果，但当前候选池一次生成，至多10次匹配；评分使用固定10 ms名义费用。当前方法尚未主动决定下一次匹配是否值得做。

E05的按需触发和E11的轻量优先级排序提供了方向：根据截止期余量、预计可救援任务、候选质量上界与已耗软件时间决定是否继续。新增计算会让整个等待批次一起多等，不能只计算“新方案救回几个任务”，还要计入原本能按时的任务被额外等待拖过截止期的风险。

一个研究用的决策表述是

\[
\text{继续搜索，当预计净增成功数}>
\text{新增等待造成的预计成功数损失}+\text{预定风险余量}.
\]

这不是已可部署的规则：边际收益和下一次调用耗时都需要在独立开发数据上估计，且在线只能使用当前已知信息，不能偷看下一次求解结果。必须保留已有合法方案作为随时可返回的解，并将触发器自身开销计入完整时间。真实硬实时保证还需要平台与最坏执行时间依据。

如果将来重新引入学习，下层可以学习“继续/停止”“优先搜索哪个源—目标对”，让精确匹配继续负责完整合法分配。这样的学习对象直接对应本章的决策费用问题，但仍须与简单解析停止规则、固定预算和同费用随机规则比较，不能预设学习有正收益。

### 3.3 先厘清价格作用，再考虑动态能耗价格

TJCCT（E17）的价格是资源交易单价，本章的 \(\lambda\) 是候选间质量—预测分配能耗的权衡系数。二者可以作方法定位上的比较，不能直接移植交易均衡、唯一性或资源分配最优性证明。

同样，本章上下层有职责分工，并不自动等于TJCCT的双时间尺度。若要使用“双时间尺度”作为本章方法名称或性质，必须有明确的大帧/小时隙更新频率与保持规则，不能仅将“上层/下层”换一种称呼。

当前主候选相对未加价混合的DSR只高0.00243个百分点，区间跨零，能耗约低8.72%。这支持描述资源取舍，尚不能声称价格提高DSR或无损节能。二值与混合还采用各自冻结价格，因此0.26243个百分点的完整策略差异不能全部归于评分。

C26启发的优先动作是做控制变量设计：

1. **评分组件比较**：同一上层、同一候选生成规则、预算和相同价格设置，比较二值与混合评分；另用固定快照区分局部评分影响和闭环状态变化。
2. **价格组件比较**：同一评分与候选规则，比较零价格、冻结价格；价格只在开发集选择。
3. **完整策略比较**：允许各方法按预定程序在开发集独立调价，再在新确认集比较，并明确它回答的是完整策略优劣。

不能把现有20块用于反复挑价格后仍称同一次独立确认。已确认数据上的新机制分析应标为探索性；改进方法需要新的独立确认。

E06、E18等队列优化工作进一步启发“预算反馈价格”：能耗高于目标时提高价格，低于目标时降低价格。但若反馈用完整机队能耗而候选成本用预测分配能耗，两者的控制对应关系需要先验证。近似候选搜索、硬筛选和非精确单步优化也使文献的队列稳定性及性能界不能直接沿用。

### 3.4 资源模型的扩展中，优先检查MBS与CPU分配

主候选相对G的机队能耗增加约1.87%，MBS占接纳比例由4.06670%升到5.94387%；本章又规定MBS单请求计算能力不随批量共享。这使“对外部算力的依赖”成为值得检验的实际边界。

结合C16、E06、E18，可优先设计有限MBS容量、回传拥塞或跨时隙积压的独立扩展。需要重新定义执行时序、任务期限与资源占用，并让所有方法共用新环境；不能直接修改旧实验后继续沿用原结论。若引入主动拒绝，仍应报告S/G、A/G、S/A，避免通过拒绝难任务仅提高接纳后成功率。

另一个方向是E15中的资源分配：当前每个节点按请求数均分算力，而任务周期数与剩余时限不同，均分不一定适合截止期目标。在固定通信时延、确定性分支、静态连续CPU分配的简化模型下，可从

\[
f^{min}_{mj}=\frac{C_m}{D_m-b_{mj}-\delta}
\]

观察一个请求按时完成所需的最低算力，其中分母必须为正；同节点所选请求还须满足 \(\sum_m f_{mj}\le F_j\)。这仅是该简化条件下的可行性观察，缓存不确定、动态资源和能耗目标需另外处理。

这里的 \(b_{mj}\) 必须取所讨论确定性分支的非计算时延；若代入平均缓存时延，得到的只是平均预测模型下的条件，不能视为真实时限保证。

尤其需要注意：**改成异构CPU份额后，给定节点请求数未必足以确定边成本。** 当前固定配额后的容量槽指派结构与理论边界需要重推，不能把一个KKT资源模块直接插进去而宣称原条件最优性保持。因此它更适合作为后续算法扩展，而非当前第五章的轻量补写。

## 4. 本轮发现的笔记误读

以下均已回查本地原文相关内容。这里不改写原笔记，避免覆盖来源材料。

| 文献 | 笔记中的问题 | 原文相关内容与影响 |
|---|---|---|
| E17 TJCCT | 把短尺度写成服务缓存优化、BSUMM/ADMM，并扩写了缩写含义 | PDF第1、9—10页：计算资源价格激励、Rubinstein议价与卸载匹配。应按原文重新理解，它与本章“匹配＋价格”的关系更直接 |
| C12 贾子晔 | 把主要求解结构写成分支定界＋ADMM | PDF第1—2页：分支定界与二进制鲸鱼算法；鲁棒对象是任务大小分布。不能用该笔记证明ADMM解法 |
| C24 王翔分布式缓存 | 写成双超网络/QMIX价值分解 | PDF第1页摘要：联邦多智能体深度强化学习，并用注意力进行差异化参数聚合。笔记的核心算法描述需要重核 |
| E15 Multi-Objective | 宣称突破标量加权、遍历Pareto曲面；又把KKT写成无迭代闭式计算 | PDF第7页式(19)明确采用线性加权，Remark 1讨论Pareto枚举开销；第10页Algorithm 1使用二分算法。不能据此批评标量化或声称无迭代 |
| E08 Trajectory/Offloading/Caching/Migration | 将缓存启发写成直接复用计算结果，容易与服务缓存混淆 | PDF第4页：任务可缓存在UAV并于后续时隙处理，属于待处理任务暂存。本章缓存的是服务程序，二者都不能直接等同于结果复用 |
| C25 李云服务编排 | DOI与页码存在错误 | 本地PDF及期刊官网为10.11959/j.issn.1000-436x.2023125，44(7):51—63；笔记写2023126、24—36 |

此外，许多笔记使用“彻底消除”“绝对零碰撞”“全局最优”“必然有限步收敛”等强表述。本次不据此认定原论文都错；应将这些措辞、精确提升百分比和数据集名称逐项回到原文核验后再引用。“加一个模块就提高论文档次”的评价也不能替代与本章问题的实际关联。

## 5. 对当前章节的落地顺序

| 优先级 | 建议 | 对应位置 | 当前能否直接做 |
|---|---|---|---|
| 1 | 补TJCCT近邻比较，明确价格、匹配对象与时间结构的区别 | 第一章相关工作；第五章5.1、5.3—5.4 | 可以先写定性比较；BibTeX需按原文补全 |
| 1 | 用三类分支表解释缓存信念的信息价值，明确这是分析而非实测归因 | 第五章5.3与5.6组件边界 | 可写代数解释；不能填未统计的比例 |
| 1 | 把当前收益归因与资源代价说清楚 | 第五章5.6 | 现有封存结果足以支撑谨慎表述 |
| 2 | 做相同价格/相同预算的评分组件比较和信念价值诊断 | 新的开发及诊断实验 | 需要服务器执行；新比较不可冒充预定主检验 |
| 2 | 设计费用感知的自适应匹配预算，与固定预算比较 | 后续方法版本 | 需要实现、开发与新的计费闭环确认 |
| 3 | 研究动态价格、MBS容量、连续队列或非均分CPU | 第六章展望或新研究版本 | 会改变算法/模型，不宜作为本章已有功能补写 |

我最推荐的技术推进顺序是：**信念价值诊断 → 费用感知匹配预算 → 受控的价格/资源扩展**。若目标是尽快完成现有论文，优先完成前三项写作补强，保留现有方法与证据即可。

## 6. 公开原始来源索引

- [TJCCT作者预印本摘要](https://arxiv.org/abs/2403.15828)：可核对短尺度定价、匹配与长尺度轨迹的基本分工。正式期刊版方法细节以本地PDF为准。
- [贾子晔等，期刊HTML原文](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT240799?viewType=HTML)：可核对不确定对象与算法结构。
- [王翔，期刊文章页](https://cjournal.hep.com.cn/1674-8425/CN/1179775867334861267)：可核对联邦学习与注意力聚合的摘要描述。
- [李云等，通信学报文章页](https://www.joconline.com.cn/zh/article/doi/10.11959/j.issn.1000-436x.2023125/)：可核对DOI、页码与出版信息。

## 7. 全部44份笔记筛选表

以下保留各笔记的独立身份；M类条目的方法细节仍需原文核验，不能仅凭本表加入正文。P也只表示核对了本次用到的原文部分，并非整篇可信性认证。

| 编号 | 笔记与原文 | 关联度 | 核对 | 针对第五章的判断 |
|---|---|---|---|---|
| E01 | [AAV-Assisted_Joint_Mobile_Edge_Computing_and_Data_Collection_via_Matching-Enabled_Deep_Reinforcement_Learning](<../../../参考文献/英文参考文献/AAV-Assisted_Joint_Mobile_Edge_Computing_and_Data_Collection_via_Matching-Enabled_Deep_Reinforcement_Learning.md>) · [PDF](<../../../参考文献/英文参考文献/AAV-Assisted_Joint_Mobile_Edge_Computing_and_Data_Collection_via_Matching-Enabled_Deep_Reinforcement_Learning.pdf>) | 高 | P | 学习连续动作、匹配离散关联的近邻工作；第一章已讨论。 |
| E02 | [Cooperative_Multiagent_Deep_Reinforcement_Learning_Methods_for_UAV-Aided_Mobile_Edge_Computing_Networks](<../../../参考文献/英文参考文献/Cooperative_Multiagent_Deep_Reinforcement_Learning_Methods_for_UAV-Aided_Mobile_Edge_Computing_Networks.md>) · [PDF](<../../../参考文献/英文参考文献/Cooperative_Multiagent_Deep_Reinforcement_Learning_Methods_for_UAV-Aided_Mobile_Edge_Computing_Networks.pdf>) | 中 | P | 消息动作与图注意力可启发通信压缩；需计真实消息时延和代价。 |
| E03 | [DRL-Based_Trajectory_Optimization_and_Task_Offloading_in_Hierarchical_Aerial_MEC](<../../../参考文献/英文参考文献/DRL-Based_Trajectory_Optimization_and_Task_Offloading_in_Hierarchical_Aerial_MEC.md>) | 低 | M | 分层空中卸载与状态归一化；与当前下层结构的直接关联较弱。 |
| E04 | [Dynamic_Trajectory_Design_for_Multi-UAV-Assisted_Mobile_Edge_Computing](<../../../参考文献/英文参考文献/Dynamic_Trajectory_Design_for_Multi-UAV-Assisted_Mobile_Edge_Computing.md>) | 低 | M | 动力学与控制跟踪主要关联第三章或部署边界。 |
| E05 | [Energy_and_Makespan_Bi-Objective_Optimization_for_UAV-Assisted_MEC_Task_Offloading](<../../../参考文献/英文参考文献/Energy_and_Makespan_Bi-Objective_Optimization_for_UAV-Assisted_MEC_Task_Offloading.md>) · [PDF](<../../../参考文献/英文参考文献/Energy_and_Makespan_Bi-Objective_Optimization_for_UAV-Assisted_MEC_Task_Offloading.pdf>) | 高 | P | 任务紧迫度与按需触发可迁移为匹配预算研究，不能直接搬部分卸载模型。 |
| E06 | [Energy_Efficient_Task_Offloading_and_Resource_Allocation_in_Air-Ground_Integrated_MEC_Systems_A_Distributed_Online_Approach](<../../../参考文献/英文参考文献/Energy_Efficient_Task_Offloading_and_Resource_Allocation_in_Air-Ground_Integrated_MEC_Systems_A_Distributed_Online_Approach.md>) · [PDF](<../../../参考文献/英文参考文献/Energy_Efficient_Task_Offloading_and_Resource_Allocation_in_Air-Ground_Integrated_MEC_Systems_A_Distributed_Online_Approach.pdf>) | 高 | P | 随机任务、队列及分布式在线资源分配；启发预算反馈与队列扩展。 |
| E07 | [Energy-Efficient_UAV-Assisted_MEC_With_Dual-Priority_Scheduling_and_Multi-Agent_Reinforcement_Learning](<../../../参考文献/英文参考文献/Energy-Efficient_UAV-Assisted_MEC_With_Dual-Priority_Scheduling_and_Multi-Agent_Reinforcement_Learning.md>) | 中 | M | 优先级排队可启发异构截止期研究；当前不存在该持续队列。 |
| E08 | [Joint_Optimization_of_Trajectory_Offloading_Caching_and_Migration_for_UAV-Assisted_MEC](<../../../参考文献/英文参考文献/Joint_Optimization_of_Trajectory_Offloading_Caching_and_Migration_for_UAV-Assisted_MEC.md>) · [PDF](<../../../参考文献/英文参考文献/Joint_Optimization_of_Trajectory_Offloading_Caching_and_Migration_for_UAV-Assisted_MEC.pdf>) | 高 | P | 跨时隙待处理任务暂存与迁移；缓存语义须与服务程序分开。 |
| E09 | [Joint_Task_Offloading_Resource_Allocation_and_Trajectory_Design_for_Multi-UAV_Cooperative_Edge_Computing_With_Task_Priority](<../../../参考文献/英文参考文献/Joint_Task_Offloading_Resource_Allocation_and_Trajectory_Design_for_Multi-UAV_Cooperative_Edge_Computing_With_Task_Priority.md>) · [PDF](<../../../参考文献/英文参考文献/Joint_Task_Offloading_Resource_Allocation_and_Trajectory_Design_for_Multi-UAV_Cooperative_Edge_Computing_With_Task_Priority.pdf>) | 中 | P | 优先级与隐空间混合动作；可作联合动作方案对照，非直接替代解析下层。 |
| E10 | [JtextC5A_Service_Delay_Minimization_for_Aerial_MEC-Assisted_Industrial_Cyber-Physical_Systems](<../../../参考文献/英文参考文献/JtextC5A_Service_Delay_Minimization_for_Aerial_MEC-Assisted_Industrial_Cyber-Physical_Systems.md>) · [PDF](<../../../参考文献/英文参考文献/JtextC5A_Service_Delay_Minimization_for_Aerial_MEC-Assisted_Industrial_Cyber-Physical_Systems.pdf>) | 高 | P | 服务缓存、卸载与3C资源联合优化的近邻；对照主动缓存与利用缓存信念。 |
| E11 | [Load-balancing multi-UAV collaborative MEC offloading-migration and trajectory optimization based on Lyapunov MAPPO](<../../../参考文献/英文参考文献/Load-balancing multi-UAV collaborative MEC offloading-migration and trajectory optimization based on Lyapunov MAPPO.md>) · [PDF](<../../../参考文献/英文参考文献/Load-balancing multi-UAV collaborative MEC offloading-migration and trajectory optimization based on Lyapunov MAPPO.pdf>) | 高 | P | 分层混合求解及动态迁移优先级；启发候选提议排序与覆盖诊断。 |
| E12 | [Minimizing_Response_Delay_in_UAV-Assisted_Mobile_Edge_Computing_by_Joint_UAV_Deployment_and_Computation_Offloading](<../../../参考文献/英文参考文献/Minimizing_Response_Delay_in_UAV-Assisted_Mobile_Edge_Computing_by_Joint_UAV_Deployment_and_Computation_Offloading.md>) | 中 | M | 部署与卸载分块优化；其求解时间是否可接受须对照本章截止期。 |
| E13 | [Multi-agent reinforcement learning for task offloading with hybrid decision space in multi-access edge computing](<../../../参考文献/英文参考文献/Multi-agent reinforcement learning for task offloading with hybrid decision space in multi-access edge computing.md>) | 中 | M | 条件混合动作设计适合未来联合动作学习；不支持仅换网络就有净增益。 |
| E14 | [Multi-Agent_Collaborative_Optimization_of_UAV_Trajectory_and_Latency-Aware_DAG_Task_Offloading_in_UAV-Assisted_MEC](<../../../参考文献/英文参考文献/Multi-Agent_Collaborative_Optimization_of_UAV_Trajectory_and_Latency-Aware_DAG_Task_Offloading_in_UAV-Assisted_MEC.md>) | 低 | M | DAG依赖会改变任务与完成判据，适合未来工作。 |
| E15 | [Multi-Objective_Optimization_for_Multi-UAV-Assisted_Mobile_Edge_Computing](<../../../参考文献/英文参考文献/Multi-Objective_Optimization_for_Multi-UAV-Assisted_Mobile_Edge_Computing.md>) · [PDF](<../../../参考文献/英文参考文献/Multi-Objective_Optimization_for_Multi-UAV-Assisted_Mobile_Edge_Computing.pdf>) | 高 | P | 标量化、资源分配与多指标取舍；笔记关于Pareto与闭式求解有误读。 |
| E16 | [Safe_and_Energy-Efficient_Trajectory_Planning_for_Heterogeneous_Multi-UAV_Enabled_Mobile_Edge_Computing](<../../../参考文献/英文参考文献/Safe_and_Energy-Efficient_Trajectory_Planning_for_Heterogeneous_Multi-UAV_Enabled_Mobile_Edge_Computing.md>) | 低 | M | 安全干预主要关联轨迹；绝对安全措辞须逐项核验条件。 |
| E17 | [TJCCT_A_Two-Timescale_Approach_for_UAV-Assisted_Mobile_Edge_Computing](<../../../参考文献/英文参考文献/TJCCT_A_Two-Timescale_Approach_for_UAV-Assisted_Mobile_Edge_Computing.md>) · [PDF](<../../../参考文献/英文参考文献/TJCCT_A_Two-Timescale_Approach_for_UAV-Assisted_Mobile_Edge_Computing.pdf>) | 高 | P | 资源议价、卸载匹配与双时间尺度；应优先补充近邻定位。 |
| E18 | [UAV-Aided_Mobile_Edge_Computing_An_Optimization_Framework_for_Dynamic_Offloading-Trajectory_Co-design](<../../../参考文献/英文参考文献/UAV-Aided_Mobile_Edge_Computing_An_Optimization_Framework_for_Dynamic_Offloading-Trajectory_Co-design.md>) · [PDF](<../../../参考文献/英文参考文献/UAV-Aided_Mobile_Edge_Computing_An_Optimization_Framework_for_Dynamic_Offloading-Trajectory_Co-design.pdf>) | 高 | P | 任务与UAV能量队列启发预算反馈；稳定性结论不能直接移植。 |
| C01 | [安全驱动的空地协同边缘计算网络中的服务缓存与计算卸载策略_吴梦如](<../../../参考文献/中文参考文献/安全驱动的空地协同边缘计算网络中的服务缓存与计算卸载策略_吴梦如.md>) | 低 | M | 保密约束和交叉熵会扩展研究问题，当前第五章不宜为增模块而加入。 |
| C02 | [多机协同自主任务规划系统研究综述_柴蓉](<../../../参考文献/中文参考文献/多机协同自主任务规划系统研究综述_柴蓉.md>) | 低 | M | 适合总体研究背景与模块关系梳理，不能代替具体方法证据。 |
| C03 | [多无人机辅助的移动边缘计算任务卸载及路径优化方法_巨涛](<../../../参考文献/中文参考文献/多无人机辅助的移动边缘计算任务卸载及路径优化方法_巨涛.md>) | 中 | M | 协同负载和轨迹分工；需分清其部分卸载与本章不可拆分任务。 |
| C04 | [多智能体强化学习在飞行器协同控制中的研究进展_杨浅舒](<../../../参考文献/中文参考文献/多智能体强化学习在飞行器协同控制中的研究进展_杨浅舒.md>) | 低 | M | 适合第二章算法谱系；优先引用原始算法论文证明具体机制。 |
| C05 | [基于李雅普诺夫的RIS辅助无人机边缘计算网络中任务卸载与资源分配研究_邝祝芳](<../../../参考文献/中文参考文献/基于李雅普诺夫的RIS辅助无人机边缘计算网络中任务卸载与资源分配研究_邝祝芳.md>) | 低 | M | 队列优化思路可参考，RIS会引入额外物理层问题。 |
| C06 | [基于强化学习的任务缓存和计算卸载效益最大化算法研究_袁晓铭](<../../../参考文献/中文参考文献/基于强化学习的任务缓存和计算卸载效益最大化算法研究_袁晓铭.md>) | 中 | M | 任务/结果缓存与服务缓存须先核实语义；不应直接换成DQN族。 |
| C07 | [基于强化学习的无人机辅助移动边缘计算任务卸载综述_姜来为](<../../../参考文献/中文参考文献/基于强化学习的无人机辅助移动边缘计算任务卸载综述_姜来为.md>) | 中 | M | 用于研究挑战与相关工作分类，不作为某项算法必然有效的依据。 |
| C08 | [基于DQN的无人机辅助移动边缘计算卸载优化_冯毅雄](<../../../参考文献/中文参考文献/基于DQN的无人机辅助移动边缘计算卸载优化_冯毅雄.md>) | 低 | M | 动作离散化基础参照；现有协议未包含，不能新增后沿用原主检验。 |
| C09 | [基于Stackelberg的无人机移动边缘计算卸载任务方法_唐宇龙](<../../../参考文献/中文参考文献/基于Stackelberg的无人机移动边缘计算卸载任务方法_唐宇龙.md>) | 中 | M | 适合比较经济定价与能耗惩罚的含义，不必新增商业博弈场景。 |
| C10 | [考虑信道资源约束的多无人机航迹与通信策略协同规划_王辰](<../../../参考文献/中文参考文献/考虑信道资源约束的多无人机航迹与通信策略协同规划_王辰.md>) | 中 | M | 信道容量提醒真实控制通信约束；主要关联轨迹与部署扩展。 |
| C11 | [空天地场景下的移动边缘计算卸载策略算法_胡智娟](<../../../参考文献/中文参考文献/空天地场景下的移动边缘计算卸载策略算法_胡智娟.md>) | 低 | M | 卫星三级卸载改变拓扑，直接用于本章的收益有限。 |
| C12 | [面向低空智联网的分布式鲁棒任务卸载方法_贾子晔](<../../../参考文献/中文参考文献/面向低空智联网的分布式鲁棒任务卸载方法_贾子晔.md>) · [PDF](<../../../参考文献/中文参考文献/面向低空智联网的分布式鲁棒任务卸载方法_贾子晔.pdf>) | 高 | P | 从任务大小不确定集借鉴误差鲁棒诊断；笔记ADMM描述有误。 |
| C13 | [面向多边缘设备协作的任务卸载和服务缓存在线联合优化机制_张秋平](<../../../参考文献/中文参考文献/面向多边缘设备协作的任务卸载和服务缓存在线联合优化机制_张秋平.md>) · [PDF](<../../../参考文献/中文参考文献/面向多边缘设备协作的任务卸载和服务缓存在线联合优化机制_张秋平.pdf>) | 高 | P | 未知服务需求学习与匹配分工；稳定性与最优成功数须分开。 |
| C14 | [面向多用户的任务卸载和服务缓存策略研究_王翔](<../../../参考文献/中文参考文献/面向多用户的任务卸载和服务缓存策略研究_王翔.md>) | 中 | M | 缓存决策与连续优化分层；须核验其变量与本章是否相同。 |
| C15 | [面向近海通信的移动边缘网络动态服务缓存策略_庄颖瑜](<../../../参考文献/中文参考文献/面向近海通信的移动边缘网络动态服务缓存策略_庄颖瑜.md>) | 中 | M | 服务需求变化与动态缓存适合作为迁移场景，不能仅改行业背景。 |
| C16 | [面向无人机辅助边缘计算的卸载任务准入控制机制_徐渊博](<../../../参考文献/中文参考文献/面向无人机辅助边缘计算的卸载任务准入控制机制_徐渊博.md>) · [PDF](<../../../参考文献/中文参考文献/面向无人机辅助边缘计算的卸载任务准入控制机制_徐渊博.pdf>) | 高 | P | 准入、通信拥塞与基站处理启发MBS边界检验；保持S/G分母。 |
| C17 | [面向DAG任务的分布式智能计算卸载和服务缓存联合优化_李云](<../../../参考文献/中文参考文献/面向DAG任务的分布式智能计算卸载和服务缓存联合优化_李云.md>) | 低 | M | 任务依赖和服务部署是后续独立模型，不宜直接拼入当前批次。 |
| C18 | [时延约束与能量感知的车辆边缘计算协同资源分配与任务卸载_高文轩](<../../../参考文献/中文参考文献/时延约束与能量感知的车辆边缘计算协同资源分配与任务卸载_高文轩.md>) | 中 | M | 截止期、能量与邻域搜索可作设计参照；车辆链路不可直接照搬。 |
| C19 | [数字孪生云边网络下服务缓存与计算卸载优化_郑家瑜](<../../../参考文献/中文参考文献/数字孪生云边网络下服务缓存与计算卸载优化_郑家瑜.md>) · [PDF](<../../../参考文献/中文参考文献/数字孪生云边网络下服务缓存与计算卸载优化_郑家瑜.pdf>) | 中 | P | 学习与凸优化分工有参考价值；不能仅加数字孪生名称而无实体同步模型。 |
| C20 | [无人飞行器集群自主控制_基于联盟形成博弈的任务分配_姜斌](<../../../参考文献/中文参考文献/无人飞行器集群自主控制_基于联盟形成博弈的任务分配_姜斌.md>) | 低 | M | 联盟与故障重分配需要新协作问题和故障协议。 |
| C21 | [无人机辅助边缘计算网络的任务卸载与资源分配联合优化_周晓天__](<../../../参考文献/中文参考文献/无人机辅助边缘计算网络的任务卸载与资源分配联合优化_周晓天__.md>) | 中 | M | 双跳带宽与资源分工可能有用，求解性质和具体算法先回原文。 |
| C22 | [无人机辅助MEC车辆任务卸载与功率控制近端策略优化算法_谭国平](<../../../参考文献/中文参考文献/无人机辅助MEC车辆任务卸载与功率控制近端策略优化算法_谭国平.md>) | 低 | M | PPO基础及车辆功率控制；不支持强行增加NOMA来增强本章。 |
| C23 | [移动边缘计算时延最小化的协作计算卸载与服务缓存_王渝中](<../../../参考文献/中文参考文献/移动边缘计算时延最小化的协作计算卸载与服务缓存_王渝中.md>) | 中 | M | 协作服务缓存与时间尺度值得复核；本次只核见网络首发封面，未验证方法。 |
| C24 | [移动边缘计算中分布式智能服务缓存和资源分配联合优化_王翔](<../../../参考文献/中文参考文献/移动边缘计算中分布式智能服务缓存和资源分配联合优化_王翔.md>) · [PDF](<../../../参考文献/中文参考文献/移动边缘计算中分布式智能服务缓存和资源分配联合优化_王翔.pdf>) | 中 | P | 原文为联邦多智能体与注意力聚合，不能按笔记当作QMIX先例。 |
| C25 | [移动边缘计算中智能服务编排和算网资源分配联合优化方法_李云](<../../../参考文献/中文参考文献/移动边缘计算中智能服务编排和算网资源分配联合优化方法_李云.md>) · [PDF](<../../../参考文献/中文参考文献/移动边缘计算中智能服务编排和算网资源分配联合优化方法_李云.pdf>) | 高 | P | 请求时间依赖与LSTM可启发信息学习；笔记DOI与页码需纠正。 |
| C26 | [异构边缘资源的任务卸载和协同调度_李小平](<../../../参考文献/中文参考文献/异构边缘资源的任务卸载和协同调度_李小平.md>) · [PDF](<../../../参考文献/中文参考文献/异构边缘资源的任务卸载和协同调度_李小平.pdf>) | 高 | P | 组件分离、开发标定及截止期多因素设计适合归因研究。 |
