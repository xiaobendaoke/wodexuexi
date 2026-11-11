# Reviewer 2 独立审稿报告

**审阅日期** 2026 年 11 月 9 日

**预分配重点** 系统模型、数学定义、算法与实现一致性、工程假设和复现信息，同时检查全部五项评价轴。

## Review setup

**Input scope** 仅使用 `.tmp/nature_review_20261109/packet/REVIEW_PACKET.json` 确定的共同快照证据包。已逐行完整阅读 `latex/main.tex` 第 1 至 52 行、摘要第 1 至 29 行、符号表第 1 至 46 行、第一章第 1 至 110 行、第二章第 1 至 337 行、第三章第 1 至 681 行、第四章第 1 至 475 行、第五章第 1 至 496 行、第六章第 1 至 33 行，以及第五章三张外置表和末尾成果、致谢页源码。

**Assessment boundary** 以硕士论文的技术可信度、可答辩性和可复核性为主要标准。Nature 的原创性、科学意义、跨领域读者兴趣、技术可靠性及非专业读者可读性仅作为检查视角，不作为学位论文的录用门槛。本报告未读取其他审稿报告、既有全面审计或历史归档，未搜索网络，未运行训练或独立重现实验。

**Shared manuscript claim summary** 论文研究以全部实际生成请求为分母的 UAV-MEC 时限服务。协议 I 研究需求观测扩展与固定快照批量字典序分配，协议 II 研究跨时隙队列中的规划蒸馏、反事实辅助微调与 Top-6 MPC。三个主要经验结果分别是观测扩展提高 DSR 约 2.85 个百分点，MC-ITR 降低固定快照平均超时数，联合微调与神经引导 MPC 在固定候选接口及指定排队协议中改善服务质量与平均计算成本。

**Visible evidence base** 在全文基础上，静态核对了包内 `utils/assignment_search.py`、`utils/queued_joint_rl.py`、`utils/queued_joint_training.py`、`utils/queued_arrival_adaptation.py`、`utils/queued_waiting_rescue.py`、`utils/queued_quality_amortization.py`、`environment/queued_service.py`、`marl_models/queued_joint_rl.py` 及对应实验驱动的相关实现段落，并查阅正式图源 CSV、桥接 JSON 的相关字段。直接查看了第五章配对效果图与决策耗时图的 PNG。

**Missing materials affecting confidence** 未阅读编译 PDF，故不声称完成逐页排版验收。未对外部引用原文、最新 SOTA、原始服务器运行环境和全部模型检查点作独立核验。没有把包内未提供的材料自动判断为不存在或实验错误。本文的静态结论也不等同于完整代码审计。

**指针约定** 以下文件名及行号均指共同快照中的原始相对路径和原文件行号，快照统一前缀为 `.tmp/nature_review_20261109/packet/`。例如 `latex/docs/chap05.tex L254` 对应包内第五章原文件第 254 行。所有问题在本报告冻结前独立形成。

## Overall assessment

论文已经形成适合硕士学位论文的完整研究链条。最有价值的部分是明确区分覆盖接纳与接纳后执行质量，承认增加需求可见性可能同时加剧资源竞争，并用冻结快照隔离任务分配外部性，最后在独立的持续排队协议中研究规划知识摊销。第四章的有限下降论证和单请求局部最优边界基本清晰，第五章对辅助目标偏置、固定父谱系和非硬实时结果也有较充分披露。

当前主要答辩风险集中在第五章新增的形式化方法描述。特征表和排队近似定义与所提供实现存在可以逐项定位的不一致，CRN 定理的标题和机制解释又超出了已证明的范围。另有一处把训练组合比较解释为热启动单因素贡献，与本章前文自己的限制说明冲突。这些问题不构成现有结果全部失效的证据，但会使评委难以确认论文证明和实验到底对应哪一个算法。建议在答辩定稿前完成实质修订。没有发现必须通过新增大规模训练才能解决的阻断项。

## Who would be interested in the results, and why

无人机辅助边缘计算、带截止期的服务调度、多智能体学习和规划蒸馏研究者会关注这些结果。尤其值得关注的是，在覆盖受限系统中，后端优化无法直接挽救未接纳请求，而增加接纳又会改变后端竞争。这种分母和瓶颈分解也对移动机器人提供计算服务、临时通信保障和资源受限的在线调度具有方法借鉴价值。跨领域意义主要来自这一问题组织方式，当前证据仍是指定合成协议内的方法验证，尚不足以支持广泛实际部署结论。

## Major strengths

1. 自然覆盖接纳、服务执行路径和分母口径贯穿正文。第三章还明确区分单元平均比值与池化计数比值，避免用均值乘积错误解释 DSR。见 `latex/docs/chap03.tex L233–254、L637–641`。
2. 生产执行与学习近似被多次主动区分。第三章说明缓存信念不等于真实命中、当前轨迹不追溯改变当前接纳，第四章说明私有副本评估和局部最优出口。见 `chap03.tex L84–103、L219–226`，`chap04.tex L193–206、L259–268`。
3. 第四章没有把有限下降夸大为全局最优，也没有把小规模穷举一致性外推到主配置。其构造反例和复杂度说明有助于答辩。见 `chap04.tex L245–286、L291–308、L443`。
4. 第五章保留了去耦和无期望向量比较的跨零区间，明确指出独立结构增益未达统计显著。零在线前瞻也没有被写成纯一次网络前向，尾时延和不计计算等待的限制已披露。见 `chap05.tex L271、L318、L451、L488–490`。
5. 直接查看的第五章配对效果图保留零线及不显著结果，决策耗时图同时展示均值、p95 和超过一秒的比例，与其限制讨论一致。未发现需要为这两张 PNG 另设图形问题的依据。

## Major Concerns

### R2-M1 CRN 定理没有证明实际辅助更新无偏，也没有给出一般方差下降条件

**Concern ID** R2-M1

**Severity** Major

**Blocking** No

**Axis** technical soundness，mechanism-evidence

**issue_key** crn-centering-theorem-scope

**claim_pointer** 第五章定理 5.4.1 标题声称反事实组中心化基线的无偏性与方差缩减，第六章将其概括为已证明的理论成果。第五章消融解释进一步把性能提升归因于消除了外生突发噪声。

**evidence_pointer** `latex/docs/chap05.tex L233–256、L269–271、L450`，`latex/docs/chap06.tex L22`。实际候选包含已执行动作和确定性 Top-3，组内均值及标准差在 `utils/queued_joint_rl.py L718–739` 由该动作相关集合计算。

**evidence_status** located

**Concern** 定理第 1 部分在所假设的加性分解下确实消去了公共项，但消去一个公共分量不等于总方差一定降低。正文只要求残差零均值，没有约束残差与公共项的协方差。一个满足这些条件的两动作反例是，令公共波动为零均值随机变量 X，两个残差分别为 −X 和 X，两个均值均为零。此时第一个原评分恒为零，而减去组均值后变为 −X，其方差反而由零增大到 Var(X)。因此现有假设不足以支持一般的方差缩减表述。

定理第 2 部分证明的是任意与当前动作无关的状态基线恒等式，作为这个恒等式本身成立。但是实际组均值取决于包含已执行动作的候选集合，不能直接当作满足该前提的基线；组标准化、截断和确定性候选加入又进一步区别于该恒等式。正文 L271 已正确承认总体辅助优化有偏，所以本问题不是要求作者再次承认偏置，而是要求新加入的定理、总结与已有正确限制相一致。

**Why it matters** CRN 是第五章的重要机制，理论解释会直接影响答辩中对收益来源的判断。当前消融支持启用整套反事实辅助的经验收益，没有单独识别 CRN 相对独立随机数的效果，也没有证明实际辅助梯度无偏。不能以泛化的定理标题为这种机制结论背书。

**Resolution test** 把已证明内容准确收缩为给定加性公共扰动下的代数消去性质，明确独立动作基线恒等式的适用前提，并与实际候选组的有偏更新区分。若保留一般方差下降，则必须补足协方差等充分条件及相应证明。把 L450 和第六章机制总结改为与该范围一致的经验解释即可，不要求为保留现有性能结论增加训练。

### R2-M2 特征张量表没有描述实际训练网络接收的特征

**Concern ID** R2-M2

**Severity** Major

**Blocking** No

**Axis** technical soundness，reproducibility

**issue_key** feature-contract-implementation-mismatch

**claim_pointer** 表 5 的特征张量构成及相邻正文给出了候选级策略的具体输入定义，并据此解释双分支的信息耦合。

**evidence_pointer** `latex/docs/chap05.tex L92–106`，特别是表 `tab:ch5-feature-spec`。实际公共上下文见 `utils/queued_joint_rl.py L365–390` 和 `utils/queued_arrival_adaptation.py L50–53`；实际分配特征见 `utils/queued_joint_rl.py L479–505`，运动特征见同文件 `L515–544`。网络输入的消费方式见 `marl_models/queued_joint_rl.py L101–135`。

**evidence_status** located

**Concern** 维度数字基本对得上，但分量含义有实质差异。表中公共上下文声称包含平均松弛度、近截止任务比例、UAV 平均与最大积压和 MBS 积压。实现却拼接八维历史速率特征，再加压力、三步速率、拥塞指示、已接纳数、等待数和阶段计数总和，没有表中这些松弛度和节点负载分量。表中分配特征写入平均完成松弛度、本地路由占比、新分配占比和确定性排序；实现明确包含安全方案指示、与安全方案的一致率、上游神经评分均值等另一组变量。运动特征表中的平均松弛度、链路增益、断链风险及相对悬停增量，也不对应实际 L527–543 的字段，其中包含拥塞乘救援、压力乘位移等交互项。

**Why it matters** 这不是术语同义替换。输入中是否有队列松弛度、安全启发式标签或上游评分，会改变学习器的信息能力、可复现实现以及关于耦合机制的解释。现有表无法被第三方直接重建为实验使用的网络输入。

**Resolution test** 以冻结版本的特征提取代码为准，逐索引给出真实字段、归一化方式、所属维度和相关模块来源，修正正文中的解释。核对四维反馈、32/44 维分支及 34 维 Critic 的拼接关系。若实验实际使用另一个实现，则应给出其准确版本并解释包内代码的角色。本项可以通过方法说明与可追溯版本修正关闭，不预设需重训。

### R2-M3 释放时间近似的正式定义与实现不是同一评价算子

**Concern ID** R2-M3

**Severity** Major

**Blocking** No

**Axis** technical soundness，reproducibility

**issue_key** release-time-operator-contract

**claim_pointer** 定义 5.3.1 被作为候选特征与奖励势函数共同使用的两阶段递推算子，随后用 932 个验证动作对说明该近似与 Engine 的接近性。

**evidence_pointer** `latex/docs/chap05.tex L118–135、L205–212`。通信递推实现见 `utils/queued_joint_rl.py L181–241`；就绪计算与评分实现见同文件 `L87–151`。

**evidence_status** located

**Concern** 至少三个可定位差异需要统一。第一，正文说同一有向转发/回传通道内各自串行递推，代码却对全部机间传输共享一个 `t_inter_free`，对全部回传共享一个 `t_bh_free`，并非每有向通道独立尾时刻。第二，正文把单任务剩余传输超过 0.75 秒写成普遍的 10 秒截断规则，代码只在当前可连接且运动后部分失联的上传分支检查累计 `t_up_done > 0.75`，一直可连接的上传和转发没有这个统一截断。第三，正文 L133 描述窗口内按剩余松弛度给出 0.85 加权、窗口外给 0.15 折减；代码在窗口内加 1.0，窗口外加 0.85 并把 0.15 加入溢出分，没有该松弛度加权。

**Why it matters** 正文已诚实说明这只是近似，本报告也不要求它精确求解一般队列。问题在于近似本身定义错误会使候选评分、奖励塑形及验证误差无法指向同一个可复现对象。高相关系数不能解决对象定义不一致。

**Resolution test** 逐项对齐通信尾时刻的共享范围、接触几何分支、截断条件、完成和溢出评分，并明确 932 动作对验证的是哪个冻结函数。保留已有近似和仿真区别说明。若采用包内实现，只需给出其忠实数学描述及一两个手工状态的对应结果，不要求改变算法或重做大规模实验。

### R2-M4 热启动单因素归因超出了已披露的训练组合比较

**Concern ID** R2-M4

**Severity** Major

**Blocking** No

**Axis** scientific importance，causal-vs-correlative

**issue_key** warmstart-causal-attribution

**claim_pointer** 反事实消融表脚注、递进机制分析与第六章把纯在策略暖启动相对冷启动的 3.80 个百分点全部归为规划蒸馏热启动贡献，并称反事实辅助是显著超越蒸馏学生的必要驱动力。

**evidence_pointer** `latex/docs/chap05.tex L281–294、L428–450、L496`，`latex/docs/chap06.tex L22`。冻结驱动中的冷暖优化配置差异见 `scripts/analysis/queued_joint_rl_experiment.py L462–496`。

**evidence_status** located

**Concern** 本章 L294 已正确指出冷暖比较同时改变初始化、参考锚点和优化参数，不能识别热启动单因子的纯效应。关闭暖组反事实项可以比较辅助项开关，却没有同时消除冷暖组之间的 Actor 学习率、epoch、batch、熵正则及参考锚点差异。因此 L444、L449 和小结将 3.80 个百分点精确分配给热启动，仍与前述限制冲突。纯在策略微调相对蒸馏的区间跨零，也不证明它只能维持初值或证明反事实是所有训练预算下的必要条件。

**Why it matters** 这影响第五章核心机制贡献的可信度，但不推翻完整训练方案的性能比较，也不否认同暖启动接口下启用反事实的正向经验增量。问题是把可支持的组合收益改写成已被严格分离的单因素贡献。

**Resolution test** 将 3.80 个百分点写为当前冻结暖启动训练方案相对冷启动训练方案的差异，与表中多项配置变化同时呈现；将必要性和无法改善的措辞收缩为本预算和配置下未观察到显著改善。同步修正第五章小结、第六章总结及摘要中相应机制用语。若作者一定要保留纯初始化因果归因，才需要另行匹配对照，本报告不把新增实验作为默认要求。

## Minor Comments

### R2-m1 观测可见性不保证网络非零敏感度

**Concern ID** R2-m1

**Severity** Minor

**Axis** technical soundness

**claim_pointer** 引理 3.3.1 的证明末句从终端坐标进入键值输入，推到策略输出对位置具有非零敏感度。

**evidence_pointer** `latex/docs/chap03.tex L398–400`。

**Issue** 输入包含位置信息只说明观测能够区分相关状态，不保证任意已训练策略都会使用该信息。常数策略、零权重或局部饱和网络即可具有零敏感度。引理关于维度与确定性注入的主要内容可以成立。

**Required correction** 删除非零敏感度的无条件保证，改为消除了该可见子集的输入信息截断，允许策略学习不同输出。无需增加实验。

### R2-m2 DSR 被反复称为三因子，但实际公式只有两个乘法因子

**Concern ID** R2-m2

**Severity** Minor

**Axis** writing-clarity

**claim_pointer** 摘要及第二章把 DSR 关系称为三因子精确分解。

**evidence_pointer** `latex/docs/abstract.tex L10、L22`，`latex/docs/chap02.tex L8、L206–225、L337`，对照 `latex/docs/chap03.tex L235`。

**Issue** G、A、S 是三种计数，DSR 的乘法分解是接纳率与条件完成率两个因子。第二章定理给出的配对变化也是双分量分解，尚无正文对应的第三乘法因子。

**Required correction** 统一为三计数、双因子分解，或直接使用端到端 DSR 精确分解，避免理论条目名称与公式不符。

### R2-m3 桥接实验训练种子列表写错

**Concern ID** R2-m3

**Severity** Minor

**Axis** reproducibility

**claim_pointer** 第四章桥接说明把第三章独立确认训练种子写为 7101 至 7110。

**evidence_pointer** `latex/docs/chap04.tex L458`。包内 `docs/figures/data/chapter34_20261004/bridge_analysis.json L6–23` 和 `scripts/analysis/run_chapter34_energy_free.py L52–54` 给出的训练种子为 7013、7283、7541、7823、8089、8353、8627、8887、9151、9419。

**Issue** 这是正文与正式桥接数据的直接来源标识不一致，不据此推断桥接数值错误。

**Required correction** 用实际列表或准确的确认种子集合名称替换，并保留桥接工作负载与第三章确认工作负载不同的说明。

### R2-m4 协议 II 无线传输 FCFS 与实现的紧迫任务优先不一致

**Concern ID** R2-m4

**Severity** Minor

**Axis** technical soundness

**claim_pointer** 第二章把协议 II 的无线传输阶段定义为先到先服务。

**evidence_pointer** `latex/docs/chap02.tex L158`，对照 `latex/docs/chap05.tex L82、L119` 和 `environment/queued_service.py L205–218`。

**Issue** 实现按 `(deadline, uid)` 遍历并为每 UE 或有向链路取首个任务，第五章也写为最紧迫任务，不能同时称为 FCFS。不同任务的期限预算可不同，最早截止顺序不等同于到达顺序。

**Required correction** 统一无线通道内的任务选择规则，并说明与 CPU 先占式 EDF 的区别。本项属于统一模型总述中的局部错误。

### R2-m5 二元字典序的容差定义在章节之间不一致

**Concern ID** R2-m5

**Severity** Minor

**Axis** technical soundness

**claim_pointer** 第二章二元优序要求第二目标至少改善正容差，第四章定义与实现使用精确元组比较。

**evidence_pointer** `latex/docs/chap02.tex L149–156`，`latex/docs/chap04.tex L161、L259–266`，`utils/assignment_search.py L29–35`。

**Issue** 带正容差的无改善出口最多保证容差意义的局部最优，精确字典序则是另一判据。第四章实现显式要求 tuple 搜索 `tau=0`，所以其本章证明基本对应实现，问题是统一模型给出的关系不同。

**Required correction** 对协议 I 的正式比较统一使用零容差精确字典序，或区分两种关系及保证。本报告不要求为当前精确比较代码改行为。

### R2-m6 对未实现的增量评价器给出了未经界定的毫秒结论

**Concern ID** R2-m6

**Severity** Minor

**Axis** claim-moderation

**claim_pointer** 第四章称解析增量评价器在百请求、约 150 至 200 次候选评价时，单快照纯算术耗时低于 1 毫秒。

**evidence_pointer** `latex/docs/chap04.tex L306–310`，与本章实测计时边界 `L308` 对照。

**Issue** 局部增量的运算量论证可以成立，但具体墙钟上界还依赖实现、硬件、接受轮数和数据布局。该段没有把低于 1 毫秒标为实测结果、理论假设估计或已有实现，也与前段对计时外推的谨慎措辞不一致。

**Required correction** 保留可能显著减少评价成本的复杂度分析，删除具体毫秒保证，或明确标注估算前提及测量来源。无需开发增量评价器来完成本轮修订。

## Technical failings that need to be addressed before the case is established

未识别到 Blocking Yes 问题。现有经验结论仍可在各自冻结协议内成立，不能因方法文字错误就推断所有数值无效。但是要把第五章作为可答辩、可复核的完整方法，必须解决 R2-M1 的定理适用范围、R2-M2 的真实输入契约、R2-M3 的近似算子定义以及 R2-M4 的因果归因。它们主要需要忠实的方法修订、版本对应和结论收缩，而非为形式完整新增实验。

第三章对信息扩展的实证检验与第四章对批量外部性的有限邻域搜索论证，在已披露的简化假设下具有清楚的解释路径。MBS 对每任务给完整频率、协议 I 不保留未完成工作量、理想状态上报、有限候选、二维线性运动、固定父谱系和计算等待不进入仿真，均已在正文明确说明，故未把这些已披露模型边界另列为缺失验证。

## Assessment against Nature-style criteria

| 五轴 | 本稿评价 |
|---|---|
| originality | 在自然覆盖分母、观测缺口、批量迁移外部性和规划辅助队列控制的组合组织上具有明确的硕士论文研究增量。核心算法组件多有既有基础，不应把模型恒等式或共享打分的结构性质单独等同于重大原创理论。未核对外部原文，无法给出优先权认证。 |
| scientific importance | 问题有实际工程意义，统一统计分母和分离接纳与执行质量有助于避免错误优化。当前证据的重要性主要在受控仿真与方法设计层面，不能扩大到一般低空网络运行规律。 |
| interdisciplinary readership | 边缘计算、机器人调度与强化学习读者可能受益。合成协议及模型接口使跨领域迁移具有启发性，但实际系统适用性尚不能由现有结果直接建立。 |
| technical soundness | 第三、四章主体推导和限制较稳健；第五章方法定义与实现不一致、CRN 理论边界及热启动归因需要实质修正。修正前不宜把全部理论条目当作已确立成果。 |
| readability for nonspecialists | 章节递进、三路径模型和构造例有帮助。符号较多但大多解释充分。三因子名称错误以及第五章不忠实的特征表，会使读者形成错误算法印象，应优先于一般措辞润色处理。 |

## 十二轴适用性简表

| 内部检查轴 | 适用性 | 本次判断依据 |
|---|---|---|
| novelty-significance | applicable | 第一章对最近工作和具体增量作对照；未进行外部优先权复核。 |
| mechanism-evidence | applicable | 需求可见性、迁移外部性及 CRN 机制均有主张；R2-M1 指向理论范围。 |
| experimental-design | applicable | 有独立确认、冻结轨迹与多臂比较；R2-M4 指向训练组合归因。 |
| statistical-rigor | applicable | 核对训练种子、工作负载、池化与均值区分以及跨零结果的文字解释；没有独立重算全部置信区间。 |
| reproducibility | applicable | 检查方法公式、特征、种子与包内实现对应；涉及 R2-M2、R2-M3、R2-m3。 |
| clinical-validity | not applicable | 没有临床队列或医疗有效性主张。 |
| ethical-governance | not applicable | 已见工作使用合成用户与流量，未见人体、动物或真实个人数据研究。 |
| data-resource-quality | applicable | 有正式图源数据、协议标识和种子列表，局部核对来源一致性；没有完整服务器来源鉴定。 |
| figures-and-tables | applicable | 阅读全部正文图注与外置表，查看两张 PNG；未完成全部图件视觉及 PDF 分页审核。 |
| writing-clarity | applicable | 检查术语、章节关系及公式解释，涉及三因子等局部问题。 |
| claim-moderation | applicable | 认可正文已披露的有限候选、固定谱系与非硬实时边界；仍需收缩必要性、无偏性及毫秒外推。 |
| causal-vs-correlative | applicable | 受控反事实开关可识别当前辅助组合效果，冷暖比较仍混合多个因素。 |

## Recommendation posture 与修改建议

以硕士论文质量为准，建议完成上述实质性方法修订后进入答辩定稿，不建议仅作语言润色便将本稿视为最终版。

1. 首先冻结并列明第五章实际用于实验的特征与近似函数，把表格、定义和实现逐项对齐，关闭 R2-M2、R2-M3。
2. 其次收缩 CRN 定理与机制解释，明确代数消去、一般方差结论和实际有偏辅助优化的不同层次，关闭 R2-M1。
3. 将热启动贡献和反事实必要性改为当前训练组合与预算下的经验差异，统一第五、六章，关闭 R2-M4。
4. 修正六项局部定义、来源和措辞问题，并在答辩材料中继续保留协议 I/II、固定父谱系、计算等待和有限候选的证据边界。

## Risk / unsupported claims

未对新父模型重训泛化、其他突发业务、真实无线网络或实机安全作验证。正文已经主动限制这些范围，本报告没有把它们当作本轮必须补做的实验。候选库在包内实现中由既有 bank、安全方案和可去重的救援动作动态扩展，见 `utils/queued_joint_training.py L169–238`、`utils/queued_waiting_rescue.py L44–97`。因此正文固定四乘六的槽位语义及每一方向的来源仍应在整理 R2-M2 的方法契约时核对，但本报告未重放上游候选构造，不据此断言实际确认运行超出四乘六或某个候选一定缺失。

本报告独立冻结，不包含跨审稿比较或事后综合结论。
