# 中山大学硕士学位论文模拟盲审总评与交叉综合报告

## 一、评审协议与物理隔离说明

本次评审严格按照双盲学位论文评阅规范与多审稿人独立评审流程组织实施，全过程分为两个物理隔离阶段。

1. **第一阶段（三路无源码独立盲审）**：启动三个相互隔离的只读评审子智能体，分别担任侧重数学建模与公式推导的审稿人（[reviewer_1_report.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/reviewer_1_report.md)）、侧重实验设计与逻辑闭环的审稿人（[reviewer_2_report.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/reviewer_2_report.md)）以及侧重学术表达与学位论文规范的大同行审稿人（[reviewer_3_report.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/reviewer_3_report.md)）。三位审稿人仅阅读送审论文 LaTeX 正文、表格、参考文献与正式导出图件，完全不接触仓库 Python 源码与实验 JSON/CSV 统计文件，三份报告在交叉汇总前已完整落盘冻结。
2. **第二阶段（冻结后代码与源数据取证核查）**：在三份盲审意见锁定后，独立对活跃工作区源码（[config.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/config.py)、`environment/`、`marl_models/`、`utils/`、`scripts/analysis/`）与正式数据源（`docs/figures/data/`、`results/`）逐项核验，形成取证报告 [forensic_code_data_audit.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/forensic_code_data_audit.md) 与结构化验证记录 [audit_verification_summary.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/validation/audit_verification_summary.json)。取证阶段发现的内部偏差单独归入作者核查层，不回溯篡改独立盲审共识。

---

## 二、站在硕士毕业论文盲审评委角度的总体判断

### 1. 模拟盲审综合定性与风险评级
- **总体学术基础与工作量评估**：**优良（工作量饱满、统计意识突出、实证态度诚实）**。论文围绕受限自然覆盖下的无人机集群时限计算服务，依次构建了预接纳扩展观测分层轨迹规划（第 3 章，600 回合探索与独立确认）、固定快照二元字典序多轮协同分配 MC-ITR（第 4 章，410 个标准评估单元 + 600 个小规模穷举快照 + 400 个桥接快照）以及跨时隙持续排队下的规划蒸馏、共同随机数反事实辅助微调与神经引导 Top-6 MPC（第 5 章，17 个评估方案、816 个盲测回合），整体研发体量与统计配对规范显著高于工学硕士论文平均水准。
- **当前直接送审风险评级**：**中高风险（三位独立盲审人一致给出“重大修改后重审/答辩（Major Revision）”倾向；若不修改直接送盲审，存在因封面学位类别冲突与占位符被形式审查直接退回，或因第 5 章黑盒模块与实时性矛盾被评委判定“修改后重审/暂缓答辩”的实质风险）**。好消息是，核心实验主表数据真实扎实（无主表数据造假或方向性颠倒），绝大部分问题可通过补齐数学定义、收缩过度拔高的表述、清理代码化文风以及修正常数笔误在论文修改层面彻底闭合。

### 2. 对四大核心关切与七项硕士盲审补充维度的逐项回答

| 审查维度 | 核心诊断结论 | 关键证据与对应问题编号 |
|---|---|---|
| **1. 方法合理性、公式合理性与推导正确性** | 已写出的形式化引理与定理（引理 2.2.1、定理 2.3.1、定理 3.2.1、引理 4.2.1、定理 4.3.1~4.3.3、引理 5.3.1、定理 5.4.1、引理 5.4.1）代数推导逻辑正确；但存在**关键模块公式漏写**（第 5 章无线传输队列方程、蒸馏教师 $Q_T$、上游父模型、第 3 章下层网络与惩罚时延）以及**局部机理定性失准**（定义 5.3.1 第一阶段并非工作守恒，`0.85/0.15` 分支在 $D_n\le 3\,\mathrm{s}$ 下恒为空集，$U^{(3)}$ 存在窗口末端拒接偏置，假设 2.2.1 回传仰角 $9.37^\circ < 45^\circ$ 且混淆 TDMA 与 FDMA 噪声）。 | `SYN-B1`、`SYN-B3`、`R1-M4`、`SYN-M1`、`AUD-05`~`AUD-07` |
| **2. 表述充分性、语病与方法描述清晰度** | 第 1、2、6 章行文较顺畅，但第 3~5 章存在严重的**“代码审计笔记直译”文风**（裸用类名 `Engine`、“生产环境”、“仿真合同”、“经核对两配置源码”、“父谱系”、“评估臂”、“池化”、“总人口”及密集防御性免责短句），且第 3~5 章多处核心基线与先验缺乏无源码可复现说明，另有主宾搭配不当、欧化残句与汉字/阿拉伯数字混排问题。 | `SYN-B1`、`SYN-B3`、`R3-M4`、`R3-m1`~`R3-m6` |
| **3. 论文数据与仓库源码/数据一致性** | **全部核心实验表（表 3-5、3-6、4-5、4-6、5-5、5-6、5-8）主指标与统计区间 100% 精确对上**；但取证核查捕获了 **7 项确认内部不一致**（表 5-4 微步长写 `0.05 s` 而代码为 `0.1 s`；第 5 章旧 Lindley 近似 RMSE 写 `0.6382` 而 JSON 为 `0.6180`；第 1 章称“毫秒级”而实测均值 `0.439~1.602 s`；3 处跨章/图文舍入尾数冲突；Jain 公平性代码混入内容请求并按 `1.0 s` 判定；第 2 章多出未定义参数 $\beta_m,\beta_a,\eta_{\mathrm{fb}}$；下层 Critic 实为单机局部观测 MLP）与 **1 项溯源缺口**（表 5-7 $w_{\mathrm{cf}}=0$ 缺活跃原始 CSV/JSON）。 | `AUD-01`~`AUD-11` |
| **4.1 跨章节逻辑闭环（协议 I 与协议 II 衔接）** | 协议 I（第 3、4 章）到协议 II（第 5 章）除引入跨时隙排队外，同步移除了终端移动性与 75 类服务/内容缓存、将计算密度调高约 3 倍并放弃了分散式局部观测与 MC-ITR 搜索，呈现出一定的“两套课题拼接感”，需在 2.3.2 节与 5.1 节强化科学必要性论证。 | `R2-M3`、`R3-M6` |
| **4.2 创新点含金量与消融证据匹配度** | 第 5 章真正产生统计显著增益的是“规划蒸馏（`+3.74 pp`）+ 反事实辅助微调（`+0.69 pp`）+ 先验裁剪 MPC”，而摘要与第 6 章创新点重点渲染的“上层期望耦合向量 $\mathbf{C}_{\mathrm{upper}}$”在确认集上相对去耦（`+0.33 pp`）与无期望向量（`+0.11 pp`）置信区间均跨零，“幂次温度校准”按引理 5.4.1 根本不改变 Top-6 候选集合；第 4 章桥接实验 `-2.633` 超时降幅中逾六成来自移除在线 `3%` MBS 配额限制后的单轮自利贪心（`Greedy-Single-Pass` 已达 `7.658`）。 | `R2-M1`、`R2-M2`、`R1-M7`、`AUD-09` |
| **4.3 航空航天工程物理边界与实时因果性** | 作为航空航天学院学位论文，全文省略旋翼推进功耗与加减速惯性却缺乏额定动力续航包线假设说明；第 5 章在 $1.0\,\mathrm{s}$ 时隙与 $2\sim 3\,\mathrm{s}$ 任务寿命下，微调策略与 Top-6 MPC 分别有 `16.9%` 与 `24.5%` 步耗时超 $1.0\,\mathrm{s}$ 且物理环境按零等待结算，不能直接外推为硬实时在线优势。 | `SYN-B2`、`SYN-M1` |
| **4.4 学位论文学术规范、符号系统与文献质量** | 全书存在严重同符异义（$\mathcal{U}, \mathcal{I}, \mathcal{M}, M, \tau, b$）；第 1 章（140 行）与第 6 章（50 行）偏薄；第 4 章全章零引文；57 篇参考文献虽 100% 正文实引，但近五年（2022~2026）仅占 `31.6%`（18/57），缺少字典序指派、Lyapunov 队列调度与 MPC 经典文献，且 `reference.bib` 存在大小写保护缺失、会议缺页码与键名错位。 | `SYN-M3`、`R2-M6`、`R3-M6`、`R3-m10`~`R3-m11` |
| **4.5 统计推断单元严谨性** | 第 3 章探索/确认隔离及第 5 章 Bonferroni 校正很规范；但第 5 章仅条件于 $N=3$ 个固定上游父模型（未覆盖上游模型重训方差），且需明确区分池化比率与回合算术均值、第 3 章逐单元对称分解与宏平均分解。 | `R2-M4`、`R2-m5`、`AUD-08` |
| **4.6 图表自明性与排版规范** | 15 张图底部烤入重复灰色说明文字、5 张图顶部烤入总标题；`图3-2` 缺灰色叉号实体、`图3-4` 漏画集中式 Critic 输入与 PPO 更新连线、`图3-5` 覆盖圆内仅见 5 个绿点（标注 6 个）、`图5-3` 漏释四色菱形含义、`图4-8(c)` 数值尾数与正文不一致。 | `R3-M7`、`R1-m5`、`R2-m2`~`R2-m3`、`R3-m8`~`R3-m9` |
| **4.7 送审合规与盲审一票否决项筛查** | **发现一票否决级前置格式缺陷**：[main.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/main.tex#L7) 设 `degree-type=professional`（专硕），而 [sysusetup.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/sysusetup.tex#L31) 与 [thesis-layout.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L84) 写死“工学硕士”（学硕）；封面扉页保留 `YOUR_NAME`、`YOUR_SPECIALITY`、`00000000` 英文占位符；成果页空白、致谢页保留 `% TODO`。 | `R3-M2`、`R1-m6`、`R2-m6` |

---

## 三、Cross-review synthesis (post-review; not shown to reviewers)

### 1. 多审稿人独立共识阻塞项（Consensus Blocking Concerns，至少两位盲审人独立标为 `Blocking: Yes`）

#### `SYN-B1` 第 5 章核心动力学、规划蒸馏教师与上游父谱系模型存在黑盒断层，阻碍独立复现
- **独立来源编号** `R1-M1`（Blocking: Yes）、`R2-M3`（Blocking: Yes）、`R3-M1`（Blocking: Yes）
- **问题综合** 三位独立盲审人全部将第 5 章的可复现性缺失列为首要阻塞项。具体包括四个层面。其一，5.2.2 节仅用式 (5-3) 给出目标节点 CPU 的先占式 EDF 方程，未写出第一阶段三类无线链路在活跃有向链路间的速率切分、同链路多任务调度规则及断链计数 $\Delta D_t$ 判据；其二，在零前瞻策略较初始先验取得的 `+4.44 pp` 总增益中，两轮规划蒸馏贡献了 `+3.74 pp`（占 `84.2%`），但 5.3.3 节未给出蒸馏教师评分 $Q_T(s,k,m)$ 与教师先验 $p_T^0(k,m\mid s)$ 的数学定义；其三，全部 17 个评估方案依赖“三个固定父谱系（上游基础检查点对）”生成候选集合 $\mathcal{K}(s)\times\mathcal{M}(s)$、先验 $p_a^0, p_m^0$、安全分配 $k_{\mathrm{safe}}$ 及上游神经评分 $s_{km}^{\mathrm{neu}}$，但手稿未交代上游模型的结构、输入特征与训练方式；其四，直接进入策略特征与评价目标的 $P(s)$、$R_{\mathrm{rescue}}(s,m)$、$\widehat{S}_{km}^{\mathrm{fut}}$、$\widehat{O}_{km}^{\mathrm{fut}}$ 以及核心对照“在线规划基线”的搜索规则和固定终端/时变热点的空间生成分布均无显式公式。此外，`R2-M3` 同时指出协议 I 与协议 II 在移动性、缓存、计算密度量纲上的多维断裂需正面论证。

#### `SYN-B2` 第 5 章在线决策尾部耗时越过 $1\,\mathrm{s}$ 宏时隙且环境零等待结算，与实时性主张矛盾
- **独立来源编号** `R1-M2`（Blocking: Yes）、`R2-M4`（Blocking: No，实验与实时性主问题，另见 `R2-m1`）
- **问题综合** 协议 II 设定宏决策步长为 $1.0\,\mathrm{s}$、任务总相对截止期仅为 $2.0\sim 3.0\,\mathrm{s}$，但表 5-8 显示零前瞻联合微调与微调 Top-6 MPC 单步均值分别为 `0.520 s` 与 `0.722 s`，P95 分别达 `1.297 s` 与 `2.203 s`，分别有 `16.9%` 与 `24.5%` 的决策步墙钟耗时直接超过 $1.0\,\mathrm{s}$ 时隙长度（在线规划基线 P95 更达 `5.998 s`、超限 `29.7%`），且手稿未提供测试 CPU/GPU 硬件与并发配置。尽管 5.5.3 节末段坦承物理仿真未计入控制器等待时间，但 [chap01.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex#L111) 错误宣称“毫秒至亚秒级在线协同决策”，摘要与第 6 章亦未限定“零控制延迟理想仿真”前提。

#### `SYN-B3` 第 3、4 章下层网络超参数、核心惩罚时延公式、外部基准表与初始启发式规则缺失
- **独立来源编号** `R1-M3`（Blocking: Yes）、`R2-M5`（Blocking: No）、`R3-M3`（Blocking: No）
- **问题综合** 三位盲审人一致指出第 3、4 章存在复现与基准缺口。第一，第 3 章仅给出上层 Attention-MAPPO 结构与表 3-3 超参数，完全遗漏了作为主方法组成部分的“下层无注意力请求级 MAPPO”及对照“完整双注意力下层”的网络拓扑、参数聚合方式与训练超参数；第二，上层奖励中的 $T_m^{\mathrm{metric}}$、下层奖励中的 $T^{\mathrm{pen}}$、文件大小 $B_s$ 范围以及表 3-5“每生成服务请求惩罚系统时延（`33.73 s` 降至 `28.15 s`）”均无定义式；第三，3.4.1 节标题虽为“多算法基准对比”，正文却仅有双注意力与无注意力两个内部变体的口头数字，缺少标准 MARL 基线（如 IPPO、标准 MAPPO）与非学习部署启发式对比表，且“无注意力反超双注意力”缺乏固定上层的解耦证据；第四，第 4 章未写出生成全部参考轨迹并提供搜索初值的 `Heuristic-Control` 具体节点选择公式及参考上层运动策略。

---

### 2. 多审稿人非阻塞重要共识项（Consensus Non-Blocking Major Concerns）

- **`SYN-M1` 航空航天平台去能耗/无惯性包线界定缺失及第 2 章物理信道建模瑕疵**（`R1-M5`、`R2-M6`、`R3-M6`）：三位审稿人共同指出，作为航空航天学院硕士论文，5 架旋翼无人机搭载 $50\sim 85\,\mathrm{GHz}$ 机载算力（总和超地面宏基站 $200\,\mathrm{GHz}$）连续高机动近 $10\,\mathrm{km}$ 并出现图 3-9 密集锯齿状瞬时反向折返，全文完全忽略推进功耗与动力学惯性，必须在第 2 章明确界定“额定动力续航窗口内任务层规划与底层姿态控制时间尺度分离”的工程包线。同时，`R1-M5` 精准捕获了第 2 章“同构旋翼无人机”与后续异构算力/缓存矛盾、边界处回传宏基站仰角低至 $9.37^\circ \ll 45^\circ$、以及将等时分多址（TDMA，固定全带宽噪声 $\sigma^2$）误述为正交频分（FDMA）的三处物理公式瑕疵。
- **`SYN-M2` 统计推断层级边界与跨章/图文舍入精度冲突**（`R1-m5`、`R2-M4`、`R2-m1`、`R2-m5`、`R3-m7`）：三位审稿人独立查出第 6 章 Jain 公平性增量 `+0.0545`（表 3-5 为 `+0.0544`）与 `图4-8(c)` 桥接差值 `-2.632`（正文为 `-2.633`）两处尾数冲突；`R2-M4` 与 `R2-m5` 进一步指出第 5 章统计推断条件于仅 $N=3$ 个固定父谱系，需显式写出分块 Bootstrap 估计量并统一池化比率与回合算术均值口径。
- **`SYN-M3` 跨章数学符号重载撞车与第 2、5 章公式参数不一致**（`R1-M7`、`R2-m4`、`R3-M5`）：三位审稿人一致指出全书符号系统严重割裂，包括 $\mathcal{U}_u(t)$ 与 $\mathcal{U}_m(t)$ 语义反转、$\mathcal{I}$ 在终端集与无人机集间跳变、$\mathcal{M}$ 在接纳请求集与候选运动集间跳变、字母 $M$ 五重过载（同一张表 5-4 内既指终端数 $100$ 又指运动候选数 $6$，表 5-6 又指比较数 $31$）、$\tau$ 三重过载、接纳率与条件完成率出现四套记号，以及第 2 章双分支公式比第 5 章多出未定义系数 $\beta_m,\beta_a,\eta_{\mathrm{fb}}$。
- **`SYN-M4` 首尾章节偏薄、第 4 章全章零引文与近五年权威文献占比偏低**（`R2-M6`、`R3-M6`）：绪论（140 行）与总结展望（50 行）篇幅单薄，`chap04.tex` 全章零文献引用，`reference.bib` 中 2022~2026 年文献仅占 `31.6%`（18/57），缺少多目标字典序组合分配、Lyapunov/MaxWeight 动态队列调度及学习引导 MPC 领域的经典与前沿权威文献对话。

---

### 3. 单审稿人独立识别的重要问题与权重差异（Important Single-Reviewer Concerns）

下列问题由特定专业视角的审稿人独立深入挖掘（两位审稿人将其列为各自视角下的 `Blocking: Yes`），技术含金量极高，但按多审稿人综合规则与多方共识项分开呈现。

- **`R2-M1`（Reviewer 2 标为 `Blocking: Yes`，`R1-M7` 与 `R1-M4` 亦提出对应非阻塞批评）第 5 章架构创新声称超出确认集消融证据**：在确认集受控消融中，联合微调（`48.74%`）相对严格去耦微调（`48.41%`，差值 `+0.33 pp`，单比较区间 `[-0.08, +0.84]`）与无期望向量微调（`48.63%`，差值 `+0.11 pp`，区间 `[-0.15, +0.41]`）均无统计显著差异，且引理 5.4.1 证明正幂次温度缩放完全不改变非并列 Top-6 候选集合（代数上仅等价于在策略低熵时将式 5-20 中的对数先验权重压缩至四分之一）。摘要与第 6 章将 $\mathbf{C}_{\mathrm{upper}}$ 可微协同与温度校准拔高为核心增效创新，必须向实证显著的三项机制（规划蒸馏、CRN 反事实辅助微调、先验裁剪搜索）收缩对齐。
- **`R2-M2`（Reviewer 2 标为 `Blocking: Yes`）第 4 章桥接实验存在 MBS 配额约束不对称与归因夸大**：第三章在线策略受 `3%` MBS 卸载上限惩罚约束，而第四章离线重评估移除了该惩罚并允许自由卸载至无带宽/算力分摊竞争的 $200\,\mathrm{GHz}$ MBS；结合取证核查 `AUD-09`，单轮自利基线 `Greedy-Single-Pass`（`7.658`）与 `Iterative-Base`（`7.478`）已消除 `-2.633` 总降幅中的 `60.0%` 与 `66.9%`，MC-ITR 相对 `Iterative-Base` 的真实外部性边际贡献为 `-0.873`。
- **`R3-M2`（Reviewer 3 标为 `Blocking: Yes`，`R1-m6` 与 `R2-m6` 亦指出专业占位符问题）送审与盲审体例一票否决硬伤**：`main.tex` 的 `degree-type=professional` 与 `sysusetup.tex` / `thesis-layout.tex` 固定页眉“工学硕士”直接冲突；封面保留 `YOUR_NAME`、`YOUR_SPECIALITY`、`00000000` 英文模板词；`achievements.tex` 空标题、`acknowledgements.tex` 保留 `% TODO`，送审前必须彻底整改。
- **`R1-M4`（Reviewer 1 独有深度推导审计）第 5 章队列近似与反事实目标的三处内在数学机理缺陷**：（1）定义 5.3.1 第一阶段按 `(deadline, uid)` 静态顺序直接覆写全局转发空闲时刻，实际上违反工作守恒，且在 $D_n \in [2,3]\,\mathrm{s}$ 下 `0.85/0.15` 分支恒为空集；（2）三步目标中的 $-\eta(s)U^{(3)}$ 双重惩罚已过期任务并错误惩罚窗口末段新接纳的健康在途任务，恰解释了表 5-5 中全部 Top-6 MPC 方案接纳率一致回落的现象；（3）反事实集合 $\mathcal{B}_t$ 非按 $\pi_{\mathrm{old}}$ 采样却在式 (5-17) 中直接等权代入重要性采样比 $\rho_\theta$。
- **`R1-M6`（Reviewer 1 独有复杂度与基准审计）第 4 章环境克隆软件开销与数学复杂度混淆及 Exact-Small 竞争样本过少**：单次迁移增量重算仅需 $O(N_i+N_j)$ 次算术运算，秒级耗时源于深度克隆整个 Python 环境对象而非算法数学下界；表 4-5/4-6 未计入共享 Greedy 初值构造耗时（`1.083 s`）；Exact-Small 600 个快照中仅 7 个含多请求竞争（$K\ge 2$）。
- **`R3-M4`（Reviewer 3 独有深度文本审计，`R2` 可读性评述亦呼应）第 3~5 章软件开发黑话与防御性免责文风**：系统梳理了裸用 `Engine`、“生产环境/生产评价器/仿真合同”、“经核对两配置源码”、“父谱系”、“评估臂”、“池化”、“总人口”及密集“不声称……”免责句对盲审阅读体验的伤害。
- **`R3-M7`（Reviewer 3 独有全量图件视觉审计）插图双层重复图注与绘制错误**：查出 15 张图底部重复烤入灰字图注、5 张图顶部烤入总标题、`图3-2` 空图例、`图3-4` 训练更新连线断裂及 `.2`/`.95` 非规范数字、`图3-5` 仅见 5 个绿点。

---

### 4. 冻结后代码与源数据取证核查专层汇总（Author / Editor Forensic Layer）

以下发现来自盲审冻结后的仓库源码与 JSON/CSV 核查（完整证据链见 [forensic_code_data_audit.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/forensic_code_data_audit.md)），独立盲审人无法仅凭手稿确认其中源码级事实，修改时须一并修正。

1. **`AUD-01`（参数错写）**：[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L404) 表 5-4 写微步长为 `0.05 s`，而 [queued_service.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L28) 实际运行值为 `Protocol.dt = 0.1 s`。
2. **`AUD-02`（数据抄录错误）**：[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L144) 将旧单次 Lindley 近似的验证集 RMSE 写为 `0.6382`，而 [tandem_approximation_audit.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/tandem_approximation_audit.json#L8-L12) 真实值为 **`0.6180`**（`0.618041`）。
3. **`AUD-04`（附加舍入误差）**：除盲审人发现的 `+0.0545` 与 `-2.632` 外，[chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L488) 将桥接集在线下层平均归一化时延 `1.25847` 误写为 `1.259`（应为 `1.258`）。
4. **`AUD-05`（公平性实现偏离）**：[user_equipments.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/user_equipments.py#L158-L162) 的 `is_successful_for_fairness()` 统计了包含三分之二内容请求在内的全部请求且统一按 `<= 1.0 s` 判定，与第 2 章及符号表“达标服务覆盖频率（按各自截止期 $D_m$）”不一致。
5. **`AUD-06`（下层 Critic 架构偏离）**：[offload_mappo.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/offload_mappo/offload_mappo.py#L173-L187) 的 `OffloadMLPCritic` 仅接收单机局部 `1405` 维观测而非五机联合状态 $O_t$。
6. **`AUD-08`（惩罚时延算式错位）**：[env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L656-L658) 与 [env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L706) 对全体 100 个终端累加 `20 s` 未完成惩罚却仅除以约 `33.8` 个生成服务请求，致使表 3-5 惩罚系统时延高达 `33.73 s`（超过 `20 s` 单请求惩罚上限）。
7. **`AUD-10`（溯源缺口）**：表 5-7 中 $w_{\mathrm{cf}}=0$ 的三组确认集计数（`120,759 / 120,701 / 121,100`）在当前活跃 `results/` 与 `docs/figures/data/` 下缺失原始 CSV/JSON 文件，且 3 个上游 `proposer .pt` 检查点不在本地。

---

### 5. 细微修改核对清单（Consolidated Minor Checklist）

- **摘要与双语对应**：英文摘要补齐 `1.602 s -> 0.722 s` 的对比基准 `online amortized planning baseline`，将穷尽列举处的 `including` 改为 `, namely,`（`R3-m1`）。
- **第 3 章细节闭合**：明确式 (3-24) 服务/内容请求字段编码以闭合 $>10^{-5}$ 非空掩码判据，讨论内容请求占据三分之二未覆盖槽位的影响（`R1-m1`）；区分不同物理量纲的数值保护项 $\epsilon_0$ 并说明一步滞后奖励信用分配（`R1-m2`）；修正 L103“全文负载”范围断言及 L80/L206/L226 语病（`R3-m3`）。
- **第 4 章细节闭合**：引理 4.2.1 补全被迁移请求自身增量 $\Delta T_m$，定义 4.3.1 补全 $\le_{\mathrm{lex}}$ 并区分目标空间全序与解空间完全预序，说明增量浮点比较容差（`R1-m3`）；修正 L391 对图 4-4 的误引（`R2-m2`）；修正“总人口”、“偷偷改写”及汉字数字混排（`R3-m4`）。
- **第 5 章细节闭合**：修正 5.3.2 节分支维度 `32` 与 `44` 顺序颠倒，说明表 5-2 公共上下文索引 9 重复通道与指数平滑系数，解释式 (5-14) 阻尼系数 `0.5` 并补齐优化器、正则权重与价值网络结构（`R1-m4`、`R3-m5`）；补齐高峰 `[50,100) s` 与非高峰窗口的定量 DSR 统计（`R2-m3`）。
- **缩略语、图表命名与文献卫生**：首次出现处展开 `GDSF`、`GRPO` 并统一 `P95`（`R3-m6`）；理顺第 3 章 `图3-4/3-5` 与 `图3-9/3-10` 文件名倒置（`R1-m5`、`R2-m2`、`R3-m8`）；补齐 `图5-3` 四色菱形图例、`图5-5/5-6` 子图标题、表 5-6 表头自明性及图 4-5/4-6 英文题注（`R2-m3`、`R3-m8`、`R3-m9`）；为 `reference.bib` 补加 `{QMIX}`、`{UAV}`、`{MEC}`、`{DRL}`、`{MAPPO}` 等大小写花括号保护，补齐 ICLR 会议页码/电子标识并核验 `he2019uav`、`fan2021uav`、`zhao2024robust`、`liu2025tmac` 等键名错位条目（`R2-m6`、`R3-m10`、`R3-m11`）；修正 `thesis-layout.tex` 第 6 章英文目录为 `Conclusions and Future Work` 并清理各章废弃历史 `\label`（`R3-m12`）。

---

### 6. 审稿人成对重合度描述性统计（Pairwise Reviewer Overlap Summary）

将三位审稿人的全部意见映射至归一化主题键集（`|K_1| = 13`、` |K_2| = 12`、`|K_3| = 15`，分母取两报告中较小主题数 `min(|K_i|, |K_j|)`），成对重合度统计如下。

| 审稿人配对 | 共享主题键数 | 较小报告主题数 | 重合比例 | 共享核心议题与互补分工特征 |
|---|---:|---:|---:|---|
| Reviewer 1 vs Reviewer 2 | 7 | 12 | `58.3%` | 共同锁定第 5 章黑盒模块、第 5 章时隙超限与毫秒级失实表述、第 3 章下层与惩罚时延缺失、航空包线、跨章符号及两处舍入误差；R1 独有队列算子非工作守恒推导，R2 独有桥接 MBS 配额不对称与消融跨零拔高分析 |
| Reviewer 1 vs Reviewer 3 | 7 | 13 | `53.8%` | 共同锁定第 5 章与第 3/4 章复现缺口、去能耗边界、符号撞车、第 5 章特征维度颠倒/重复通道、舍入误差与扉页专业占位符；R3 额外捕获代码化黑话、学位类别冲突与 15 张图双层图注 |
| Reviewer 2 vs Reviewer 3 | 8 | 12 | `66.7%` | 共同锁定第 5 章上游父谱系黑盒與协议断层、第 3 章模型缺失、去能耗与文献不足、符号冲突、舍入误差、第 5 章图例/子图缺陷、盲审扉页硬伤与 BibTeX 著录瑕疵 |
| **三方完全交集（R1 ∩ R2 ∩ R3）** | **6** | **12** | **`50.0%`** | 第 5 章核心模块可复现性、第 3/4 章下层与启发式规格、航空平台去能耗建模边界、跨章数学符号冲突、跨章/图文舍入尾数差异、扉页专业占位符规范 |

---

## 四、Risk / unsupported claims（超出证据边界或存在内部冲突的高风险声明清单）

| 序号 | 论文原文高风险声明（含位置） | 风险类型 | 实证与代码核查事实 | 建议修正口径 |
|---|---|---|---|---|
| 1 | “能在**毫秒至亚秒级时间**内完成在线协同决策”（[chap01.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex#L111)） | 数据直接反驳（`SYN-B2`、`AUD-03`） | 表 5-8 全部受测策略单步平均耗时为 `0.439~1.602 s`（无任何毫秒级方案），P95 为 `1.220~5.998 s`，`15.0%~29.7%` 的决策步耗时超过 `1.0 s` 宏时隙，且仿真按零控制延迟结算 | 改为“在零控制等待理想仿真原型下，将单步平均在线决策耗时由在线规划基线的 `1.602 s` 降至亚秒级（`0.520~0.722 s`）” |
| 2 | 摘要与第 6 章创新点将“通过上层期望运动耦合向量 $\mathbf{C}_{\mathrm{upper}}$ 实现双分支可微协同”与“温度校准 Top-6 MPC”并列为核心增效来源（[abstract.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex#L16)、[chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L36)） | 消融区间跨零 / 代数保序（`R2-M1`、`R1-M7`） | 确认集上联合微调较严格去耦微调仅高 `+0.33 pp`（95% CI `[-0.08, +0.84]`），较无期望向量微调仅高 `+0.11 pp`（95% CI `[-0.15, +0.41]`）；引理 5.4.1 证明正幂次校准不改变非并列 Top-6 集合且无独立开关消融 | 将核心增效机制严格锚定在规划蒸馏（`+3.74 pp`）、CRN 反事实辅助微调（`+0.69 pp`）与先验裁剪 MPC 上，将 $\mathbf{C}_{\mathrm{upper}}$ 定位为保持置换同变的统一参数化探索 |
| 3 | 第 4 章桥接快照上“MC-ITR 将平均超时请求数由在线下层的 `9.238` 大幅降至 `6.605`（降低 `28.50%`），证明显式计入批量算力外部性的巨大潜力”（[chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L488)、[chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L34)） | 约束不对称与归因混杂（`R2-M2`、`AUD-09`） | 在线下层受 `3%` MBS 卸载配额惩罚约束，离线桥接设 `mbs_penalty = 0.0`；且无外部性重算的 `Greedy-Single-Pass`（`7.658`）与 `Iterative-Base`（`7.478`）已消除总降幅的 `60.0%` 与 `66.9%`，MC-ITR 较 `Iterative-Base` 净降幅为 `-0.873` | 补齐桥接集全部 5 种方法数值表，明确区分“放宽在线 MBS 配额惩罚与贪心修正带来的 `-1.760` 降幅”与“MC-ITR 计入同批外部性较自利迭代进一步带来的 `-0.873` 净降幅” |
| 4 | 定义 5.3.1“释放时间**工作守恒**串联排队近似消除未就绪任务阻塞已就绪任务”，且窗口外任务按 `+0.85/+0.15` 连续折减（[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L116-L143)、[chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L17)） | 数学机理与源码不符（`R1-M4`、`AUD-07`） | 第一阶段按 `(deadline, uid)` 静态顺序覆写 `t_bh_free/t_inter_free`，早截止期慢上传任务仍会阻塞已上传就绪任务；且因 $D_n\in[2,3]\,\mathrm{s}$ 恒有 $s_n\le 3\,\mathrm{s}$，`0.85/0.15` 分支恒不可达 | 改为“第一阶段单通道尾时刻递推 + 第二阶段节点释放时间工作守恒 EDF 推进”，并说明在 $D_n\le 3\,\mathrm{s}$ 设定下窗口外折减分支自动退化 |
| 5 | 表 5-4“微步长 `0.05 s`”与 5.3.1 节“旧单次 Lindley 近似 RMSE `0.6382`”（[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L144)、[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L404)） | 内部数值失实（`AUD-01`、`AUD-02`） | `queued_service.py` 实际运行 `dt = 0.1 s`；`tandem_approximation_audit.json` 中旧近似真实 RMSE 为 `0.6180` | 直接更正为 `0.1 s` 与 `0.6180` |
| 6 | 第 2 章“$N$ 架同构旋翼无人机”、“正交频分信道假设与仰角 $\ge 45^\circ$”及“达标服务覆盖公平性”（[chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L20)、[chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L77)、[chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L213)） | 跨章与源码矛盾（`R1-M5`、`AUD-05`） | 第 3~5 章无人机算力与缓存容量均异构；回传宏基站最低仰角仅 $9.37^\circ$ 且固定噪声功率 $\sigma^2$ 对应 TDMA；公平性源码统计全体服务+内容请求且按 `<= 1.0 s` 判定 | 修正为“机动几何同构、机载算力与缓存异构”，修正多址噪声与回传视距假设表述，如实对齐公平性统计口径 |

---

## 五、分优先级可执行修改路线图（Prioritized Revision Roadmap）

建议按 **`P0`（送审一票否决与核心失实修复） -> `P1`（复现黑盒补齐与创新主张校准） -> `P2`（学术文风、符号图表与文献提质）** 三阶段推进修改（多数工作仅需编辑 LaTeX 文本与重导局部图件，无需在服务器重跑大规模 MARL 训练）。

### `P0` 阶段（送审前必须清零的形式硬伤、数据笔误与过界声明）
1. **修复封面扉页与盲审合规配置（`R3-M2`）**：核定本人攻读学位类别（专硕或学硕），统一 [main.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/main.tex#L7)、[sysusetup.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/sysusetup.tex#L6-L31) 与 [thesis-layout.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L84) 页眉；填写真实专业名称，按中大盲审规定规范隐去姓名、学号、导师并规范处理 [achievements.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/achievements.tex) 与 [acknowledgements.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/acknowledgements.tex)。
2. **修正全部确认的数值与参数错误（`AUD-01`~`AUD-04`）**：表 5-4 微步长 `0.05 -> 0.1`；5.3.1 节旧近似 RMSE `0.6382 -> 0.6180`；第 1 章删除“毫秒至”；第 6 章 Jain 公平性增量 `+0.0545 -> +0.0544`；第 4 章桥接在线归一化时延 `1.259 -> 1.258` 并统一 `图4-8(c)` 的 `-2.632/-2.633` 尾数；从远程服务器补存表 5-7（$w_{\mathrm{cf}}=0$）原始汇总 CSV/JSON 至活跃数据目录（`AUD-10`）。
3. **校准核心创新声明与桥接实验归因（`R2-M1`、`R2-M2`、`SYN-B2`、`AUD-09`）**：在摘要、第 1 章与第 6 章收敛对 $\mathbf{C}_{\mathrm{upper}}$ 与温度校准的拔高表述，突出蒸馏、反事实辅助微调与先验裁剪 MPC；在摘要、5.5.3 节与第 6 章明确注明在线耗时结论基于“零控制等待仿真原型”并补齐计时软硬件环境；在 4.4.3 节补出桥接集全部 5 种方法的对比表，如实说明 MBS 配额约束差异及 `Iterative-Base`（`7.478`）与 `MC-ITR`（`6.605`）的分段贡献。

### `P1` 阶段（补齐无源码独立可复现定义与物理建模严谨性）
1. **补齐第 5 章黑盒模块数学定义（`SYN-B1`）**：对照 [queued_service.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py)、[queued_planner_distillation.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_planner_distillation.py)、[queued_quality_amortization.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py) 与 [queued_waiting_rescue.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_waiting_rescue.py)，在 5.2~5.4 节（或正式附录）补齐第一阶段无线链路共享推进方程、三个上游候选预训练模型结构与先验 $p_a^0,p_m^0,p_T^0,k_{\mathrm{safe}}$、特征算子 $P(s),R_{\mathrm{rescue}},\widehat{S}^{\mathrm{fut}},\widehat{O}^{\mathrm{fut}}$、蒸馏教师评分 $Q_T(s,k,m)$ 以及“在线规划基线”算法规则；同时修正定义 5.3.1 第一阶段非工作守恒定性与 `0.85/0.15` 退化说明，补充 $U^{(3)}$ 截断偏置讨论（`R1-M4`、`AUD-07`）。
2. **补齐第 3、4 章网络结构、指标算式与基准对比表（`SYN-B3`、`AUD-05`、`AUD-06`、`AUD-08`）**：对照 [offload_mappo.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/offload_mappo/offload_mappo.py) 增补下层 Actor/局部 Critic 结构与超参数表；如实写出 `env.py` 中综合惩罚系统时延、$T_m^{\mathrm{metric}}$、$T^{\mathrm{pen}}$ 与 Jain 公平性的真实计算式；利用 `canonical_baseline_analysis.json` 中已有的多基线评测数据在 3.4.1 节补出完整的多算法基准对比表；对照 `utils/chapter4.py` 补写第 4 章 `Heuristic-Control` 公式并统一表 4-5/4-6 含初值总耗时说明（`R1-M6`）。
3. **闭合航空工程包线与跨协议演进论证（`SYN-M1`、`R2-M3`）**：在 2.2 节增设旋翼平台额定续航窗口与高低层控制时间尺度分离假设，修正同构/异构、回传视距仰角及 TDMA 噪声表述，并在 2.3.2 节系统阐述协议 I 向协议 II 简化移动性与缓存以聚焦跨时隙排队效应的实验控制逻辑。

### `P2` 阶段（全面清除代码黑话、统一符号体系、修缮插图与扩充近五年文献）
1. **学术中文重构（`R3-M4`、`R3-m1`~`R3-m6`）**：全文检索替换 `Engine`、“生产环境/生产评价器/仿真合同”、“核对两配置源码”、“父谱系”、“评估臂”、“池化”、“总人口”，精简合并琐碎的防御性免责句，统一阿拉伯数字计量规范。
2. **全书符号消歧（`SYN-M3`）**：以 [notation.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/notation.tex) 与第 2 章为唯一基准，消除 $\mathcal{U}, \mathcal{I}, \mathcal{M}, M, \tau$ 的跨章撞车，删除第 2 章未定义的 $\beta_m,\beta_a$ 并清理残留历史 `\label`（`R3-m12`）。
3. **图件与文献升级（`R3-M6`、`R3-M7`、`R3-m8`~`R3-m11`）**：运行修订后的绘图脚本去除 15 张图底部重复图注与 5 张图顶部总标题，修复 `图3-2`（补灰色叉号）、`图3-4`（补 Critic 与 PPO 更新连线）、`图3-5`（避免第 6 个绿点被遮挡）与 `图5-3`（补四色图例）；扩充第 1 章与第 6 章篇幅，为第 4 章及全文补充 10~15 篇 2022~2026 年通信/控制权威期刊及字典序优化、Lyapunov 队列调度与学习引导 MPC 文献，并修复 `reference.bib` 大小写花括号与页码缺失。
