Reviewer 3 独立报告，已冻结

本报告按硕士论文标准评价。审阅过程只读，未读取其他审阅报告，未修改论文或实验文件，未启动训练。审查覆盖当前六章正文、摘要、符号表、参考文献库及指定实现和结果材料。重点核对了第三、四章统计口径，以及第五章816份确认回合的计数、比较单位和统计实现。

总体判断

论文已有可成立的、范围明确的硕士研究结果。第三章扩展观测的主场景改善、第四章固定快照分配质量余量，以及第五章完整微调方案改善蒸馏学生，均具有具体实现和数值支持。全文已较好地区分批量服务与动态排队协议，保留阴性消融，并避免把离线快照收益写成闭环DSR收益。

目前宜作实质修订后定稿。主要问题是第五章冻结来源未覆盖实际依赖的全部上游模型、31项同时区间对小样本推断方法的敏感性，以及现行方法相对规划学习文献的定位不足。本次没有发现足以推翻全部中心结论的已确认错误，故下列Major均标为Blocking No。

读者与价值

面向多无人机MEC、调度与学习控制的研究者，论文的价值在于把接纳机会、批量竞争和持续排队分开验证。对其他领域读者，最容易理解且最有用的认识是，覆盖更多需求与接纳后质量可能反向变化，较低平均决策耗时也不等于满足截止期执行要求。

已独立核对的主要结果

- 第三章确认材料支持等权单元平均DSR从0.2450001提高至0.2735088，差2.8509个百分点，训练种子配对区间为[1.6452, 4.0566]个百分点。9/10训练种子为正。池化分母均为9,805,180，与摘要和正文一致。
- 第四章主配置MC-ITR与Iterative-Base每快照平均超时数分别为2.045和2.215。正文正确限定为固定快照效果。
- 第五章816份原始回合全部满足生成数等于成功数加过期数。每臂252,024个生成请求，联合蒸馏、联合微调、在线规划和微调MPC的成功数分别为121,079、122,842、122,382和124,280。摘要所列48.04%、48.74%、48.56%、49.31%及1763、1898个额外成功请求可复算。
- 第五章分析代码确实条件于三个固定父谱系，重采样工作负载及其动作流，没有将816回合或25万请求当作独立训练重复。冷暖训练组合、耦合消融、微调MPC相对蒸馏MPC的结论边界总体恰当。

Major Concerns

R3-M1　实际在线依赖的上游检查点没有全部进入冻结清单

- Severity为Major。Blocking为No。主轴为可复现性。
- Claim pointer为[chap05.tex第346行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:346)，正文报告15个科学源码、6个上游父检查点及51个本实验模型的冻结摘要。
- Evidence pointer为[utils/queued_joint_training.py第140行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py:140)。`prepare_slot_base`在第147行调用`distribution(..., 'hard_joint', parent * 2, ...)`。该调用经[scripts/analysis/queued_score_control.py第46行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_score_control.py:46)及第56至60行，加载`queued_learning_causes_20261104/models/pool_hard_joint/{0,2,4}.pkl.gz`。这三个实际参与先验分布构造的模型不在现有`parent_checkpoint_sha256`清单中。清单记录的是三个proposer和三个`reg_joint` scorer。业务生成器`utils/queued_service_business_refinement.py`及其依赖也未列入15个源码摘要。
- 问题为冻结材料能够标识部分核心文件，但不足以唯一确定实际确认控制器和外生场景生成流程。缺失的`pool_hard_joint`模型不是无关历史文件，其输出进入当前候选先验。
- 影响为在保留现有六个父文件和51个模型的情况下，改变未登记模型仍可能改变确认运行。现有计数可以成立，但冻结清单不能独自提供所暗示的完整复现实验保障。
- 解决判据为补充实际依赖图及全部科学依赖、上游检查点的来源索引，区分确认前已有摘要与现在补登记的摘要。若历史冻结副本仍可核验，应将其关联到本轮回合；若只能补录当前文件，应如实说明，不能追认其为确认前冻结。不要求因此自动重训。

R3-M2　31项同时区间需要披露八个工作负载下的方法敏感性

- Severity为Major。Blocking为No。主轴为统计严谨性。
- Claim pointer为[abstract.tex第16行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex:16)、[chap05.tex第344行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:344)和第387行，微调MPC相对在线规划的全31项校正区间为[+0.25,+1.42]个百分点。
- Evidence pointer为[scripts/analysis/review_queued_joint_rl.py第104行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/review_queued_joint_rl.py:104)。实际方法为10000次百分位分块Bootstrap，31项区间使用两侧各`0.05/62`的经验分位数，每侧仅约8个Bootstrap尾部样本。
- 问题为统计层级识别正确，但独立外生工作负载只有八个，极端百分位区间的小样本覆盖与数值稳定性没有相应检查。Bonferroni变换不能修复单项区间本身的覆盖误差。
- 独立敏感性核对为保留同一池化计数比率估计量、固定三个父谱系，并以工作负载为簇作比率线性化和自由度7的t区间。联合微调相对蒸馏仍得到约[+0.384,+1.015]个百分点的31项区间；微调MPC相对在线规划则约为[-0.164,+1.670]个百分点，双侧p约0.00467。后者仍通过原八项主比较的Bonferroni门槛，其八项区间约为[+0.042,+1.464]个百分点。
- 影响为原预定八项族下的主要结论仍有支持，但“扩大至31项仍被稳健确证”的强度具有方法依赖。上述敏感性不是对原Bootstrap结果的算术反证，也不说明t方法必然更优。
- 解决判据为补充保留配对结构的小样本敏感性及Bootstrap尾部分位数稳定性检查，并明确31项区间是具体百分位Bootstrap方法下的估计。若不同合理方法下结论不同，可将主推断回到预定八项族，把31项结果作补充披露，无需把全部主效应删除。

R3-M3　新贡献与规划学习、学习辅助MPC的关系尚未充分建立

- Severity为Major。Blocking为No。主轴为原创性与重要性。
- Claim pointer为[chap01.tex第118行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:118)、第139行，以及[chap05.tex第18行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:18)。
- Evidence pointer为第一章相关工作及第二章理论部分。现稿引用PPO、GAE、Policy Distillation和DAgger，解释了组成工具，却没有系统比较规划引导策略学习、MPC模仿与学习先验辅助规划这一直接相关的方法谱系。
- 外部原始来源表明，通过规划或轨迹优化引导策略学习已有成熟先例，例如[Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html)；MPC与在策略模仿的关系也已有专门研究，例如[Model Predictive Control via On-Policy Imitation Learning](https://proceedings.mlr.press/v211/ahn23a.html)。这些工作不等同于本文方法，不能据此否定本文原创性，但需要说明区别。
- 影响为论文目前比较充分地论证了完整方案是否有效，尚不足以让读者准确判断新增知识究竟是任务协议、表示设计、训练组合，还是一种新的通用优化原理。尤其耦合和反事实项的独立收益尚未确认时，原创性更应落实到可验证的具体设计。
- 解决判据为增加简洁的相关方法比较，说明教师形式、候选支持、学习信号、在线前瞻和理论保证的差别；把原创贡献限定在确实新增且有证据支持的部分。按硕士论文标准，不要求证明所有组成模块独立优越，也不要求为补文献重做全套实验。

Minor Comments

R3-m1　绪论仍把旧配额搜索写成当前第五章方法

- Severity为Minor。Blocking为No。主轴为可读性。
- Affected element为[chap01.tex第88行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex:88)，“本文采用后者处理第五章的配额搜索，策略选择节点任务数量的迁移方向”。
- Evidence pointer为[chap05.tex第96行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:96)及第115至138行，现行第五章选择候选分配与联合运动，采用32/44维双分支评分，不是此处描述的配额迁移策略。
- Required correction为改成当前候选级策略描述。第二章第632至682行的配额搜索可保留为背景，但需说明实际章节用途，避免读者误认为它是第五章主体算法。

R3-m2　非劣效与等效判断应明确为界值敏感性分析

- Severity为Minor。Blocking为No。主轴为统计解释。
- Affected element为[chap05.tex第391行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex:391)。
- Evidence pointer为`review_queued_joint_rl.py`第141至144行。现有90%区间确实未通过±0.50个百分点等效判据，通过±0.75个百分点判据；95%区间下界高于-0.50个百分点。数值判断本身正确。
- Issue为冻结主比较清单未提供这些容许界值的预先业务依据，也未将这组非劣效、等效判断纳入明确检验族。
- Required correction为显式标注“事后界值敏感性分析”，不将其作为预先设计的确认性替代结论；若需要工程非劣效主张，应事先确定业务界值和检验层级。

R3-m3　第四章把相容解释写成了已观察到的抵消机制

- Severity为Minor。Blocking为No。主轴为机制证据。
- Affected element为[chap04.tex第414行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex:414)，“两种方向相反的效应在主配置样本上相互抵消”。
- Evidence pointer为同章第407行，正文承认汇总数据没有逐项统计接受迁移类别；第412行只显示平均归一化时延差的区间跨零。
- Required correction为改成“可能相互抵消，与汇总结果相容”。未检出平均差异不能确认具体抵消机制。

R3-m4　Exact-Small子集中的差值分母需要修正

- Severity为Minor。Blocking为No。主轴为图表与统计口径。
- Affected element为[chap04.tex第484行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex:484)。
- Evidence pointer为第475行的600快照总体平均差`3.33×10^-5`，以及第484行“在这七个多请求快照上”之后重复使用该数。
- Issue为总体600快照平均差与七个多请求快照子集平均差不能直接互换。现稿容易把总体稀释后的差值读成子集差值。
- Required correction为注明该数仍以全部600快照为分母，或另从逐快照结果计算非空、多请求子集的均值。

R3-m5　提交元数据尚有模板占位

- Severity为Minor。Blocking为No。主轴为完整性与可读性。
- Evidence pointer为[sysusetup.tex第6行](C:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/sysusetup.tex:6)起的姓名、专业、导师及学号占位；成果部分也仍为模板。
- Required correction为正式提交前填入经作者核实的信息，并按学校要求处理无成果条目。此项不影响科学结论。

五维评价

| 维度 | 评价 |
|---|---|
| 原创性 | 有明确的任务建模和方法实现工作；完整方案与具体组件的原创性边界需要进一步落实 |
| 科学重要性 | 对自然覆盖MEC仿真和资源调度有实际研究价值，属于有意义的领域内进展 |
| 跨学科兴趣 | 接纳与条件质量的分解、规划计算摊销可被调度和学习控制读者理解；尚不支持现实应急部署结论 |
| 技术可靠性 | 主计数、重复单位和多数结论边界较扎实；冻结依赖完整性及31项区间稳健性需要修订 |
| 非专业读者可读性 | 六章递进基本清楚；旧配额方法残留和大量边界说明会干扰主线，宜集中整理 |

引用核验范围与边界

本轮外部核验使用原始论文、作者公开论文或出版机构页面。已核对Li等鲁棒多UAV-MEC研究的作者、研究对象及Beta-MAPPO描述，与现稿一致。[原始论文](https://arxiv.org/abs/2308.12756)

已核对Hu等UAV中继与边缘计算、Yu等联合卸载与资源分配研究的基本对象及优化变量，未发现当前引述中的实质错配。[Hu等作者稿](https://repository.essex.ac.uk/26888/1/Accepted%20version_UAV-Assisted%20Relaying%20and%20Edge%20Computing%20Scheduling%20and%20Trajectory%20Optimization.pdf)，[Yu等作者公开论文](https://guoyuanxiong.github.io/assets/pdf/yu2020joint_IoTJ.pdf)

已核对Skaltsis综述、曾耀平等部署与卸载、王义君等缓存卸载研究的核心描述，当前正文所概述的方法和适用问题基本对应。[Skaltsis出版页](https://link.springer.com/article/10.1007/s10846-023-02011-0)，[曾耀平等原文](https://www.zjujournals.com/eng/article/2026/1008-973X/zjdxgxb-2026-8-1686.shtml)，[王义君等出版页](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT260054)

PPO、GAE、MAPPO、DAgger和Policy Distillation的基本出处可对应原始来源，未发现将这些通用算法冒称为本文原创的表述。[PPO](https://arxiv.org/abs/1707.06347)，[GAE](https://arxiv.org/abs/1506.02438)，[MAPPO](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)，[DAgger](https://proceedings.mlr.press/v15/ross11a.html)，[Policy Distillation](https://arxiv.org/abs/1511.06295)

未完成全文全部引用的逐字段外部核验。Tao等及李侍阳等部分出版页面访问受限，不能宣称已经全文核实其算法细节。没有重新执行完整训练或逐步重放第三、四章远程原始实验，也未将本地缺少三个proposer文件判断为实验未完成。当前结果不支持完整父模型重训总体、未知突发业务、能量约束或计入真实决策等待后的部署收益，这些边界在正文中大多已有恰当披露。

推荐姿态

完成上述来源、统计稳健性和方法定位修订后，可继续按现有有界结论推进硕士论文定稿。无需为了形成“所有模块显著有效”的叙述删除阴性结果，也无需把跨协议数值强行合并。
