# 第五章改造：近期研究筛选与适配判断

检索日期：2026-09-29。范围：优先 2025–2026 年 UAV/MEC 卸载、共享容量协调、截止期调度；补充 2024–2025 年有方法与代码支持的神经组合优化。本文是定向研究筛选，不是系统综述、完整查新证明或已完成实验结论。借鉴建议均为本项目的待验证设计。

## 核心判断

建议优先验证“MAPPO 初始卸载方案＋截止期敏感的批量竞争修正”，暂不同时加入图网络、联邦学习、世界模型、长期队列与教师蒸馏。较低通信需求的备选方向是共享容量约束，但长期平均约束不能保证单批请求按时完成。

近期文献已经覆盖 MAPPO 与优化分解、容量协调、学习引导匹配、有限步搜索及自改进训练。单纯把其中两种技术连接起来不足以证明新颖性。可研究的具体差异是：在本项目平分算力、自然覆盖和请求截止期口径下，如何快速识别一次分配对同节点其他请求造成的成功数变化，并在给定通信与计算预算内修正。

## 检索与核验范围

- 工作流：nature-academic-search / multi-source-search。
- 当前无可调用的 academic-search 专用 MCP。附带 OpenAlex 脚本先因 Python 3.9 类型注解不兼容失败，改用本机 Python 3.11 后两项发现查询均收到 HTTP 429。
- 随后使用网页检索定位 arXiv、AAAI、出版社与作者机构原始页面；Crossref 对 MD-MAPPO 的标题、作者、DOI、2026 年 8 月出版信息核验成功，对 MAPPO-BD 查询限流。
- 主要检索词包括 `UAV MEC multi agent reinforcement learning 2025 task offloading graph`、`2025 neural combinatorial optimization learn improve local search`、`task offloading matching reinforcement 2025`、`task offloading 2026 MAPPO`，随后按准确标题、DOI 查询。
- 预印本与正式版按标题、作者、DOI/arXiv 号合并，不计为不同成果。MAPPO-BD 的预印本为 2025 年，机构最新记录的正式卷年为 2026 年；不按 DOI 中的 2025 推断卷年。
- 下列代码仓库只核验了可访问页面和论文关联，未安装、运行或复现实验。没有用其他论文的百分比预测本项目收益。

## 候选文献

### 1. MD-MAPPO：最直接的近期同领域对照

Jiaming Zhang, Yufei Niu, Ziqiong Lin, Jingmin Yang, Wenjie Zhang. **Load-balancing multi-UAV collaborative MEC offloading-migration and trajectory optimization based on Lyapunov MAPPO**. *Computer Communications*, 257, 108605, 2026.

- DOI：[10.1016/j.comcom.2026.108605](https://doi.org/10.1016/j.comcom.2026.108605)。[出版社预览](https://www.sciencedirect.com/science/article/abs/pii/S0140366426001957)。
- 原文要点：把长期队列与能耗问题分解；分别使用 MI-ADMM、动态优先级、凸优化和 MAPPO 处理卸载、迁移、资源分配与轨迹。
- 可借鉴：轨迹学习与批量卸载优化分工；合作迁移需考虑负载。
- 不能直接照搬：长期队列、可优化 CPU 分配与当前按最终任务数平分算力不是同一模型。仅加入 Lyapunov 名称或额外惩罚不能继承其稳定性性质。
- 核验深度：出版社摘要/引言预览＋Crossref 元数据；未获得方法全文，不能据此声称已复现或完成算法逐式比较。

### 2. DCC：以共享约束协调拥塞

Andrea Fox, Francesco De Pellegrini, Eitan Altman. **Multi-Agent Reinforcement Learning for Task Offloading in Wireless Edge Networks**, 2025.

- [arXiv:2509.01257v2](https://arxiv.org/abs/2509.01257v2)；页面注明 AI4NextG @ NeurIPS 2025 Workshop oral，不是 NeurIPS 主会论文。
- [全文](https://arxiv.org/html/2509.01257v2)；[作者代码](https://github.com/Andrea-Fox/multiAgentTaskOffloading)。
- 原文要点：各智能体解本地 CMDP，以低频更新的共享约束向量协调拥塞动作；策略、乘子与协调约束按不同时间尺度更新。
- 可借鉴：对具体执行目标建立容量协调信号，而不是仅累计本 UAV 已发出的任务。
- 适配限制：原文采用特定拥塞模型和期望近似；不能将其理论推广到多目标、异质请求、单时隙截止期而不重新推导。低频平均约束对突发拥塞可能反应不足。
- 核验深度：阅读全文模型和 DCC 机制，核验仓库页面；未运行。

### 3. 学习引导匹配：执行节点也参与分配

Yi Wang, Anja Klein, Lin Xiang. **Learning-Guided Matching Game for Decentralized Task Offloading to Multi-Functional UAVs**. IEEE ICC 2025, pp. 789–794.

- DOI：[10.1109/ICC52391.2025.11161164](https://doi.org/10.1109/ICC52391.2025.11161164)。[作者机构全文](https://www.kt.tu-darmstadt.de/media/kt/publikationen_1/20/25/ICC_Wang_2025.pdf)。
- 原文要点：设备提出卸载请求，UAV 选择接纳子集；用梯度式多臂老虎机学习分散提案。未被接受的请求回到设备本地计算。
- 可借鉴：目标节点看见整批提案之后再协调，能直接处理跨源请求竞争。
- 适配限制：本项目“本地”是服务 UAV，并不等价于该论文的设备本地计算。不能凭空加入 UE 本地执行，也不能删除被拒请求或改变 DSR 分母。原文忽略提案/回应时延，本项目新通信协议不能自动沿用零成本假设。
- 核验深度：作者全文中的系统模型、提案—接受机制及假设；未找到并核验可运行代码。

### 4. LRBS：限制搜索展开，而非无限求精

Federico Julian Camerota Verdù, Lorenzo Castelli, Luca Bortolussi. **Scaling Combinatorial Optimization Neural Improvement Heuristics with Online Search and Adaptation**. AAAI 2025, 39(25), 27135–27143.

- DOI：[10.1609/aaai.v39i25.34921](https://doi.org/10.1609/aaai.v39i25.34921)。[会议正文](https://ojs.aaai.org/index.php/AAAI/article/view/34921/37076)；[作者代码](https://github.com/federico-camerota/LRBS)。
- 原文要点：Limited Rollout Beam Search 为已训练的改进策略保留有限候选并短程展开，研究搜索及策略适配。
- 可借鉴：控制候选数、短程展开深度、保存最好方案，形成质量—计算预算曲线。
- 适配限制：原问题为 TSP 及变体，策略执行的是改进动作。现有卸载 actor 并非这种改进策略；不能直接套代码或声称获得 1 秒实时性。
- 核验深度：会议元数据、正文算法 1 与仓库页面；未运行。

### 5. Gumbeldore：训练阶段从更好方案中学习

Jonathan Pirnay, Dominik G. Grimm. **Self-Improvement for Neural Combinatorial Optimization: Sample Without Replacement, but Improvement**. TMLR, June 2024（arXiv 标注 Featured Certification）。

- [论文与发表记录](https://arxiv.org/abs/2403.15180)；[全文](https://arxiv.org/html/2403.15180v2)；[作者代码](https://github.com/grimmlab/gumbeldore)。
- 原文要点：对实例生成多个候选，以优质方案形成伪专家训练数据；无放回采样与搜索树更新提高候选利用效率。
- 可借鉴：若修正有效但在线太慢，可在训练阶段学习改善方案；减少对昂贵精确教师的依赖。
- 适配限制：静态组合优化与长期闭环控制不同。辅助模仿不能伪装成 PPO on-policy 样本；只能使用训练分割生成标签，单列额外搜索预算。全局教师若使用学生不可见信息，可能存在不可学习的冲突。
- 核验深度：正文训练算法、限制和仓库页面；未运行。建议第二阶段才考虑。

### 6. MH-DDQN：按服务类型区分决策

Ayeh Mahjoubi, Arunselvan Ramaswamy, Karl-Johan Grinnemo. **Deadline-aware service scheduling via multi-head MARL in device-edge-cloud environments**. *Computer Networks*, 277, 112019, 2026.

- DOI：[10.1016/j.comnet.2026.112019](https://doi.org/10.1016/j.comnet.2026.112019)。[出版社页面](https://www.sciencedirect.com/science/article/pii/S1389128626000319)；[机构全文入口](https://kau.diva-portal.org/smash/get/diva2%3A2028969/FULLTEXT02.pdf)。
- 原文要点：设备—边缘—云协作调度，每类服务设置 Q 网络输出头，比较 MH-DQN、MH-DDQN 与模拟退火。
- 可借鉴：审计不同服务类型的错误是否被总体均值掩盖；确有冲突时再研究类型条件化表示。
- 适配限制：论文处理依赖工作流；不能把其服务成功比例直接与自然覆盖下全部实际服务请求分母比较。“多头”是服务类型输出头，不等于多头注意力。
- 核验深度：出版社摘要、亮点与机构论文索引；机构 PDF 直接打开失败，未逐式审阅全文。

### 7. MAPPO-BD：队列与连续动作的参考

Muhammet Hevesli, Abegaz Mohammed Seid, Aiman Erbad, Mohamed Abdallah. **Multi-Agent DRL for Queue-Aware Task Offloading in Hierarchical MEC-Enabled Air-Ground Networks**. *IEEE Transactions on Cognitive Communications and Networking*, 12, 217–236, 2026；2025 年预印本/早期记录。

- DOI：[10.1109/TCCN.2025.3555440](https://doi.org/10.1109/TCCN.2025.3555440)。[机构正式记录](https://elmi.hbku.edu.qa/en/publications/multi-agent-drl-for-queue-aware-task-offloading-in-hierarchical-m/)；[预印本全文](https://arxiv.org/html/2503.03391v1)。
- 原文要点：UAV/HAPS 多层系统中的队列感知卸载，使用 Beta 分布处理有界连续动作。
- 可借鉴：负载信息应进入决策；上层若存在动作裁剪问题，可以单独诊断。
- 适配限制：本项目下层是离散动作，换成 Beta 分布不能解决它的批量拥塞问题；不能引入持续队列后仍与原主场景数值直接比较。
- 核验深度：预印本方法正文与机构正式元数据；未核对正式版逐页差异。

### 8. AirFed：近邻架构与创新边界

Zhiyu Wang, Suman Raj, Rajkumar Buyya. **AirFed: A Federated Graph-Enhanced Multi-Agent Reinforcement Learning Framework for Multi-UAV Cooperative Mobile Edge Computing**, 2025.

- [arXiv:2510.23053v2](https://arxiv.org/abs/2510.23053v2)；[全文](https://arxiv.org/html/2510.23053v2)。本次核验为预印本，未核验正式录用。
- 原文要点：双层动态图注意力、双 actor 与单 critic、联邦协作；考虑覆盖和截止期。
- 可借鉴：将服务关系与协作关系分别表示；可作为需要深入比较的近邻工作。
- 适配限制：当前只有 5 UAV，没有证据说明联邦学习是主要瓶颈。其自报高截止期满足率不能与本项目 DSR 排名；奖励惩罚本身不自动构成硬约束保证。
- 核验深度：全文图结构与 QoS 奖励章节；未复现。

### 9. 2026 年动态协作 MAPPO：进一步的近邻边界

Tiankui Zhang, Wenlong Xu, Tianyi Shi, Xiaoxia Xu, Arumugam Nallanathan. **Distributed Trajectory Planning and Resource Allocation for Dynamic Multi-UAV Collaborative Computing**, 2026-08-24.

- [arXiv:2608.23123](https://arxiv.org/abs/2608.23123)。本次只核验到预印本。
- 原文要点：动态 UAV 参与、Stackelberg 分层，MAPPO 控制轨迹/迁移/关联/价格，终端通过两阶段迭代选择卸载比例与资源。
- 可借鉴：学习与求解器分工已有近期先例，不应把这种组合本身声称为首次提出。
- 适配限制：收费、可分任务、参与集合变化不是本项目当前问题；不建议同时移植。
- 核验深度：arXiv 元数据与摘要，未审阅全文细节。

## 与现有代码的对应

审计对象为 `coverage-optimization/wodexuexi` 实验工作树，不将主工作区同名文件视为相同版本。具体工作树位置由 `git worktree list` 确认。

- `environment/env.py:get_offloading_obs_and_masks`：动作对应具体协作 UAV；请求特征主要提供最佳协作候选的汇总量，缺少与各具体动作一一对应的目标特征。
- `environment/uavs.py:_get_computing_latency_and_energy`：UAV 算力按最终服务任务数平分。
- `environment/uavs.py:execute_request_batch`：先冻结全批最终目标任务数，再执行服务请求。
- `marl_models/offload_mappo/offload_mappo.py:BatchOffloadActor` 的对应批次 actor 实现：本地前缀只包含本 UAV 已分配任务，不是跨 UAV 的最终目标负载。定位应以类定义和 `prefix_count` 为准，后续重命名可能改变类名。

2026-09-29 pilot 中 A/B/C/D 的平均评估单元 DSR 分别为 27.70%/26.43%/27.35%/28.05%。B/C 未晋级；D 与 A 的差值区间跨零。该试验未证明增加批次特征有效，也未支持同信息 Attention 架构的 DSR 优势。这些数值来自该工作树 `results/batch_offload_20260928/pilot/batch_offload_analysis.json`，不是本次新实验。

## 推荐改法：以截止期外部影响为核心的有限预算修正

这是一项设计推断，不是上述文献已在本环境证明的结论。

对 UAV 目标 j，令 F_j 为算力（cycles/s），c_r 为请求 r 计算量（cycles），n_j 为同批最终分配数。按当前平分机制，计算时延为 c_r n_j/F_j。在暂时固定通信与缓存取数估计 b_rj 时，总时延近似为：

`T_hat(r,j,n_j) = b_rj + c_r*n_j/F_j`

由此可计算请求的并发容忍阈值 `floor(F_j*(deadline_r-b_rj)/c_r)`；若 `deadline_r <= b_rj`，则该估计下即使有算力也不能满足截止期。阈值只适用于这个条件化预测，不是硬可行性证明；MBS 需要沿用其自己的计算模型，不能套用 UAV 平分公式。

候选机制：

1. 保留现有 MAPPO 初始分配，先不增加训练。
2. 按目标收集拟执行请求，识别追加/移出一个请求会跨越哪些截止期阈值。
3. 评价迁移对受影响请求成功数的净影响，避免只降低被迁移请求的时延。
4. 对可能改善的候选执行现行通信、缓存和能耗模型下的完整复核。缓存状态、取数共享或其他耦合变化时，扩大受影响集合，不能只更新任务数。
5. 在固定候选数预算内做单迁移，必要时短程交换；保留初始方案和当前最好方案，达到预算返回。
6. 所有已接纳请求仍必须有去向。不能通过删除困难请求、引入不存在的 UE 本地处理或改变分母制造收益。

若需要当前跨 UAV 提案，必须显式设置通信与协调流程。它改变原先的信息协议；所有匹配对照获得同等信息获取机会，记录各自实际通信/计算开销，不能直接拼接旧结果。只用本地摘要的分散方案与集中修正方案应作为不同协议。

这里的候选贡献是“针对截止期外部影响的低开销评价与搜索”，不是阈值公式本身、不是 MAPPO、不是泛化的学习加搜索。只有与已有近邻方法进一步比较，并证明同预算下质量或速度优势后，才可以形成最终创新陈述。

## 分阶段检验与停止条件

1. **开发阶段，不新增训练**：复用已有模型，比较原分配、普通限步局部搜索、截止期敏感修正。覆盖全部预定快照，按来源/负载分层报告；同时运行已验证的完整重评估作为预测误差检查，不能把执行后真值送入在线策略。
2. **闭环小评估**：网络冻结，比较 A 和同信息 Vanilla 各自有/无相同修正。修正后的动作必须真实影响后续环境；统计全部实际生成请求 DSR、接纳、接纳后质量、潜在请求固定分母、能耗、MBS 与完整 p95 时间。
3. **必要对照**：贪心初始化＋同修正；普通搜索与提出搜索的同候选评估次数比较、同墙钟预算比较。若贪心加修正同样强，收缩学习初始化的贡献主张。
4. **训练阶段仅在前两步有效后进行**：固定上层，比较原 MLP、同逐节点输入的普通 MLP、共享逐节点打分器；再决定是否做端到端联合训练。不能把多项改动收益全部归功于网络结构。
5. **自改进仅为后续选项**：若修正质量有效但在线成本过高，再考虑 Gumbeldore 启发的改善方案模仿。额外搜索数据只来自训练集；记录教师可见信息、生成耗时和训练计算预算。PPO 保存提案动作的概率，改善动作进入独立监督目标，不能混算概率比。
6. **独立确认**：开发数据不得充当最终独立确认数据，使用新的训练种子与工作负载、提前冻结目标与停止规则。DSR 没改善就不能事后以能耗改善替换原主指标宣布成功。

本次只完成文献检索、代码机制对照和研究建议；未修改模型/环境，未启动训练，未改写论文主张。
