Reviewer 1 独立冻结报告  
冻结日期为 2026-11-08。本报告未与本轮其他审阅报告比较，未修改文件，未启动训练。

**审阅范围与边界**

已从头阅读中英文摘要、第一至六章、符号表、主入口及封面配置，检查第三四章数据包和第五章冻结协议，并抽查相关实现、实验脚本和实际图像。评判标准为硕士学位论文所需的问题完整性、方法清楚性和证据可信度，不以 Nature 的发表门槛要求本稿。历史交接材料仅用于定位来源，未把其中已有审计意见当成本报告结论。

未重新训练、重新执行全量统计或复现全部原始轨迹。因此，下列判断是对现有稿件与提供证据的独立审阅，不是完整计算复现认证。

**Overall assessment**

论文已经具有可成立的统一研究问题，即有限空间覆盖和计算资源下，如何提高动态地面请求的截止期完成比例。第三章研究可见需求与空间接纳，第四章研究给定状态的批量资源竞争，第五章研究持续队列中的规划知识摊销和交互改进。这种“问题逐步丰富”的结构可以支撑硕士论文。

但它不是“第三章上层方法，加第四章下层方法，最终在第五章集成验证”的算法递进。正文多数位置已经诚实说明这一点，这是明显优点。当前主要问题是第一二章仍残留另一套配额搜索方法的介绍，同时第五章真正依赖的候选生成、教师及上游模型说明不足。需要优先完成方法身份和贡献定位的结构修订，未发现必须先增加大型实验才能修复的主叙事问题。

当前可支持已有规定协议内的结果；本次审阅没有识别出足以否定全部主要性能结论的阻塞问题。以下 Major 均为 Blocking No，但属于定稿前应解决的实质问题。

**Who would be interested in the results, and why**

多 UAV 移动计算服务、资源分配和学习控制研究者会关心需求可见性、请求接纳与执行质量之间的关系。学习辅助优化研究者会关心昂贵规划器的候选偏好如何转移到较低测试成本的策略。航空航天方向读者可以将其理解为简化机动条件下的任务规划研究，不能将其理解为已验证的实际飞行控制系统。

**Major strengths**

- 第三章明确处理未接纳请求，计数分解及不同聚合层级的说明较完整。探索和独立确认分开，避免用筛选结果冒充确认结果。
- 第四章的完整分配重评价、严格字典序与单请求邻域性质解释清楚。对空快照、局部最优、精确小规模验证及搜索开销均保留限制。
- 第三章候选轨迹到第四章固定快照之间存在实际桥接证据。
- 第五章区分真实轨迹奖励、反事实辅助评分和 MPC 目标，也保留耦合消融跨零、固定父谱系及尾延迟限制。没有用完整方案改善偷换组件独立有效。
- 摘要和第六章的数值与边界总体相容，不再把不同执行协议的绝对 DSR 串成统一性能提升。

**Major Concerns**

**R1-M1**

- Severity 为 Major。Blocking 为 No。
- Axis 为 writing-clarity。
- Claim pointer 为第一章将第五章描述为学习配额迁移，再通过条件化匹配生成实际请求分配。
- Evidence pointer 为 [chap01.tex 第88行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:88)，第二章 [第632行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex:632)、[第649行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex:649)和[第672行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex:672)。现行第五章实际动作定义见 [chap05.tex 第96行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:96)，联合分布见 [第130行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:130)。
- Concern 为第一二章仍在为另一套配额搜索方法铺垫。现行第五章选择完整分配候选与联合运动候选，并非在正文所述意义下学习节点任务数量迁移。这不只是背景冗余，而是对本论文实际采用方法的明确误述。
- Why it matters 为读者会误以为第五章依赖固定配额后的线性指派，以及第二章基于批量均分推导的边代价。第五章实际采用 EDF 持续队列，其计算模型和动作层次均不同。
- Resolution test 为清除“第五章配额搜索”的错误指向。若保留指派和配额内容，应说明其确切实际用途与所依赖模型，不能继续作为现行第五章核心理论基础。用候选动作生成、规划蒸馏和持续队列控制的必要基础取代无直接用途的大段铺垫。本项可通过改写解决，无需新增训练。

**R1-M2**

- Severity 为 Major。Blocking 为 No。
- Axis 为 reproducibility。
- Claim pointer 为第五章以冻结上游提案器、评分器和已有三步规划器构建候选及蒸馏教师，再以候选级联合策略学习。
- Evidence pointer 为 [chap05.tex 第98行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:98)、[第100行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:100)和[第142行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:142)。实现见 [queued_joint_training.py 第140行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py:140)、[第183行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py:183)和[第685行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py:685)。
- Concern 为论文给出了选择器的详细概率公式，却把决定动作支持和监督标签的主要模块压缩成名称。正文没有充分交代上游提案器和评分器分别学什么、来自哪一阶段，分配候选和运动候选如何产生，安全分配及救援候选如何进入候选库。教师 \(Q_T\) 也只有定性描述，没有与实现中未来服务修正、过载惩罚、安全方案奖励和拥塞救援惩罚对应的完整定义。
- Why it matters 为有限候选框架中的策略只能选择被生成的动作。候选库和教师是实际性能的重要承载部分，不能作为不影响研究解释的工程细节略去。源码可核查，但目前论文本身尚不能让读者重建核心方法，也不易判断新学习模块相对继承系统究竟增加了什么。
- Resolution test 为补充一个自包含的方法说明或正式附录，覆盖上游模块来源与输入输出、候选生成与去重、安全及救援规则、候选规模、教师完整评分、特征定义与归一化。给出模块和冻结模型的映射，区分继承模块与本章新增模块。并补报教师标注量、额外反事实分支量及可取得的训练计算记录，使“规划计算摊销”的成本边界可检查。优先整理既有代码和日志，无须为补写方法说明重新训练。

**R1-M3**

- Severity 为 Major。Blocking 为 No。
- Axis 为 novelty-significance。
- Claim pointer 为三项主要贡献，尤其 MC-ITR 和第五章联合表示、训练组合与控制方式的贡献定位。
- Evidence pointer 为 [chap01.tex 第137行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:137)、[第138行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:138)、[第139行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:139)，相关工作讨论见 [第69行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:69)和[第118行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:118)。MC-ITR 实际搜索形式见 [chap04.tex 第213行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex:213)，第五章自述定位见 [chap05.tex 第18行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:18)。
- Concern 为目前较好地区分了“本方法没有证明什么”，但没有同样清楚地区分“本方法相对最接近已有方法具体增加什么”。MC-ITR 的搜索骨架是严格字典序下的最佳单迁移改善，有限终止与邻域局部最优也是该结构的直接性质。第五章引用了经典蒸馏、DAgger 和 PPO，但尚缺针对学习辅助规划、候选选择与模型辅助策略更新的集中比较。
- Why it matters 为受控比较可以支持本协议中的收益，却不能单独完成方法原创性定位。硕士论文可以以任务建模、针对性的算法适配和可靠实证构成贡献，不必把所有标准组件包装成新的通用算法。
- Resolution test 为围绕最接近方法建立简明比较，列出问题假设、动作支持、目标、规划调用位置及新增设计。明确第三章主要贡献是观测合同的修正与确认，第四章是特定批量模型下整体代价判据及求解验证，第五章是既定候选框架中的训练和控制组合。若不主张新通用优化原理，应直接按应用方法贡献写清楚。本项首先需要文献定位和叙事修订，不要求盲目增加基线或追求发表级新颖性。

**Minor Comments**

**R1-m1**

- Severity 为 Minor。Blocking 为 No。Axis 为 data-resource-quality。
- Affected element 为桥接快照来源。
- Evidence pointer 为 [chap04.tex 第505行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex:505)，[运行脚本第52行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/run_chapter34_energy_free.py:52)和[桥接数据第18行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/bridge_analysis.json:18)。
- Issue 为正文称从第三章十个训练种子和五个工作负载中选取五十任务，但确认负载为 2203、2243、2287、2333、2377，桥接负载为 3001、3037、3061、3097、3121。复用的是训练所得检查点，桥接工作负载另有一组。
- Required correction 为改成复用第三章确认得到的十个候选检查点，在另五个桥接工作负载上采样，并明确两组种子。此修正增强来源清楚性，不否定桥接结果。

**R1-m2**

- Severity 为 Minor。Blocking 为 No。Axis 为 claim-moderation。
- Affected element 为第六章对同信息替代结构实验的回顾。
- Evidence pointer 为 [chap06.tex 第15行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex:15)，相应第三章边界见 [chap03.tex 第684行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex:684)。
- Issue 为第六章具体提到“同信息替代结构的探索性比较”，但现行第三章未给出该比较结果和可识别来源。不能让读者分辨它属于当前无能量协议，还是另一历史协议。
- Required correction 为给出明确来源和协议范围，或将句子收缩为“本章同网络观测对照尚未识别注意力的独立结构优势”。无需为这一总结句补新实验。

**R1-m3**

- Severity 为 Minor。Blocking 为 No。Axis 为 reproducibility。
- Affected element 为第三章训练预算。
- Evidence pointer 为 [chap03.tex 第496行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex:496)、[第583行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex:583)，实际配置见 [运行脚本第549行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/run_chapter34_energy_free.py:549)。
- Issue 为正文详细报告 rollout 和 PPO 更新参数，却将训练回合数保留为算法输入 \(E\)，实验协议没有明确写出当前实际使用的 200 回合及末检查点规则。
- Required correction 为在协议表加入每次训练 200 回合、每回合 1000 步、评估检查点及确定性动作规则，并给出种子或来源索引。

**R1-m4**

- Severity 为 Minor。Blocking 为 No。Axis 为 writing-clarity。
- Affected element 为跨协议符号与量纲。
- Evidence pointer 为 [notation.tex 第24行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/notation.tex:24)和[chap05.tex 第57行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:57)。
- Issue 为符号表主要覆盖第三四章。其计算密度定义为 cycles/byte，第五章使用 cycles/bit，并新增候选索引、释放时间、剩余工作与三种评分而没有集中索引。正文局部单位已写对，但跨章查阅仍容易误用。
- Required correction 为在符号表按协议标明两种计算密度的单位，并补入第五章核心符号。不要把两种密度直接统一成相同数值。

**R1-m5**

- Severity 为 Minor。Blocking 为 No。Axis 为 claim-moderation。
- Affected element 为第四章对汇总结果的机制解释。
- Evidence pointer 为 [chap04.tex 第407行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex:407)和[第414行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex:414)。
- Issue 为前文说明没有逐项统计迁移机制类别，后文却直接称“两种方向相反的效应在主配置样本上相互抵消”。总均值差接近零不能唯一识别这一抵消机制。
- Required correction 为改为“该结果与两类效应可能抵消相容”，保留构造推导与实际机制识别的区别。无需新增逐迁移实验才能保留主结果。

**R1-m6**

- Severity 为 Minor。Blocking 为 No。Axis 为 writing-clarity。
- Affected element 为交付状态。
- Evidence pointer 为 [sysusetup.tex 第6行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/sysusetup.tex:6)、[第18行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/sysusetup.tex:18)，以及 [achievements.tex 第5行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/achievements.tex:5)。
- Issue 为作者、专业、导师、学号仍是模板占位，已启用的成果页仍含 TODO。
- Required correction 为正式送审前填写或按学校规范处理。这些占位不构成科研证据问题，但当前文件不是可直接提交的完整定稿。

**章节联系与应采用的叙事**

| 章节联系 | 证据判断 | 合适表述 |
|---|---|---|
| 第三章内部的信息改动与服务变化 | 有同网络、同物理半径、分别训练的独立确认 | 扩展可见 UE 信息的完整改动在指定主场景有效 |
| 第三章到第四章 | 有候选学习轨迹的冻结快照桥接 | 改善接纳后仍存在离线任务分配余量 |
| 第四章到第五章 | 有概念联系，没有 MC-ITR 模块或教师迁移 | 批量竞争诊断促使研究进一步转向持续队列和在线计算预算 |
| 第五章内部蒸馏到完整微调 | 有独立确认，完整微调相对学生初值提高 | 支持该训练组合对固定父谱系的改善 |
| 第五章耦合或反事实项单独收益 | 尚未分离确证 | 作为设计组成和待解释机制，不能作为已确认性能来源 |
| 三章合成一个最终系统 | 无对应统一协议的集成证据 | 将论文定位为共享服务目标下的三项递进研究，不作收益相加 |

无需为了让第四五章看起来“连起来”，强行把 MC-ITR 接入第五章或把它改写成蒸馏教师。更应修复 R1-M1，清楚承认从第四章到第五章是研究问题和执行模型的推进。

**五维评价**

- **Originality**  
  具有具体问题适配、实现和实证工作。第三章的贡献边界清楚。第四五章相对最接近已有方法的新增内容尚需集中说明，不能从标准搜索或训练组件名称推导新颖性。
- **Scientific importance / significance**  
  对 UAV 计算服务的建模与评价有实际研究价值。当前支持有限仿真条件下的领域内贡献，已足以形成硕士论文问题，不需要宣称普遍重大突破。
- **Interdisciplinary readership interest**  
  可连接航空任务规划、通信计算资源管理和学习辅助优化。跨领域读者首先需要理解实际动作和服务过程，候选库的黑箱描述削弱了这一点。
- **Technical soundness**  
  指标、统计单位和负结果边界总体谨慎，第三章及第四章的受控证据较有说服力。第五章主结果可按固定父谱系解释；现有材料不足以支持组件因果归因、重新训练总体优势或现实闭环有效性。
- **Readability for nonspecialists**  
  摘要、绪论中心问题和第六章已较一致。主要阅读障碍是过时配额叙述、真正核心模块说明不足，以及大量重复的边界说明超过了方法的正面解释。可集中保留一次完整边界表，再让各节更多解释具体做法。

**支持与不支持的边界**

支持第三章完整观测改动在规定主场景中的收益，支持第四章相对指定自身代价对照的固定快照改善，支持第五章完整微调相对蒸馏学生、以及学习引导 MPC 相对指定规划基线的条件性结果。

不支持同信息下注意力的独立优势、MC-ITR 闭环长期 DSR 改善、第五章耦合与反事实项各自必需、零在线前瞻策略对规划基线的优效、全部重新训练总体上的普遍优势、硬实时或现实飞行部署结论。论文多数位置已经正确限制这些结论，应保持。

**Technical failings that need to be addressed before the case is established**

优先解决 R1-M1 的方法身份混写与 R1-M2 的核心方法说明缺口，再完成 R1-M3 的贡献定位。无需把已有阴性消融强行变成阳性，也不应将新增重训视为本轮叙事修订的默认前提。

**Recommendation posture**

按硕士论文标准，研究内容和主要实证足以支撑进一步定稿。建议完成上述实质性结构修订后送审，不宜直接按现稿提交。这个判断针对论文完整性，不是 Nature 适配或编辑决定。
