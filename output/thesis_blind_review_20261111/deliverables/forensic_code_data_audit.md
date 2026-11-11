# 论文—代码—实验数据取证式一致性核查报告（Forensic Code & Data Audit）

> **协议隔离声明**  
> 本报告属于三份独立盲审报告（[reviewer_1_report.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/reviewer_1_report.md)、[reviewer_2_report.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/reviewer_2_report.md)、[reviewer_3_report.md](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/deliverables/reviewer_3_report.md)）落盘冻结之后执行的**作者/编辑级代码与源数据交叉审计（Post-Freeze Forensic Audit）**。三位独立盲审人完全不接触仓库 Python 源码、`results/` 与 `docs/figures/data/` 下的统计文件；本报告发现的代码与源数据偏差不反向计入“审稿人独立共识”，而是在总评综合报告中单独列出，专门回答“论文表述、数学公式与仓库代码及实验数据是否存在不一致”的核心问题。核查全过程严格限制在当前活跃工作区，未读取或引用任何历史归档目录文件。

---

## 一、核查范围与总体结论

本次取证核查逐行比对了论文全部 LaTeX 源码（[main.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/main.tex)、[sysusetup.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/sysusetup.tex)、[abstract.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex)、[chap01.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex) 至 [chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex)、[notation.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/notation.tex) 及第 5 章三张独立表格文件）、活跃仿真与学习源码（[config.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/config.py)、`environment/`、`marl_models/`、`utils/`、`scripts/analysis/`）以及正式封存数据源（`docs/figures/data/chapter34_20261004/`、`docs/figures/data/queued_joint_rl_20261106/`、`results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/`）。机器可读汇总已同步写入 [audit_verification_summary.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_blind_review_20261111/validation/audit_verification_summary.json)。

| 状态分类（Audit Status） | 数量 | 涉及编号 | 核心定性结论 |
|---|---|---|---|
| `confirmed_internal_error`（确认内部不一致） | 7 | `AUD-01` 至 `AUD-07` | 含微步长参数错写、串联近似 RMSE 抄录错误、毫秒级耗时失实声称、跨章舍入尾数冲突及 3 类公式与源码实现偏离 |
| `aggregation_ambiguity`（聚合口径或约束混杂） | 2 | `AUD-08`、`AUD-09` | 第 3 章对称分解聚合层级与 100-UE 惩罚时延口径未交代；第 4 章桥接实验移除 3% MBS 配额惩罚且正文漏报中间基线 |
| `provenance_gap`（活跃工作区溯源缺口） | 1 | `AUD-10` | 表 5-7（$w_{\mathrm{cf}}=0$ 受控消融）缺少原始 JSON/CSV 文件，且 3 个上游提议器 `.pt` 检查点与原始轨迹未存于本地 |
| `passed`（严格核验通过） | 1（综合组） | `AUD-11` | 第 3、4、5 章全部主实验表、敏感性表、置信区间、独立性与回放守恒审计、以及 57 篇文献正文引用率均 100% 精确吻合 |

---

## 二、逐项取证核查明细（`AUD-01` 至 `AUD-11`）

### `AUD-01` 协议 II 物理仿真微步长（`dt`）表格记载值与源码实际值相差一倍
- **状态分类** `confirmed_internal_error`
- **严重程度** Major
- **论文位置** [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L404)（表 5-4 `tab:ch5-sim-params` 第 1 行右侧）
- **代码与数据位置** [queued_service.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L28)（`Protocol.dt` 定义）；[queued_joint_rl_experiment.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L129)（`Engine(Protocol(), ...)` 实例化）
- **不一致事实** 表 5-4 明确列出“宏决策步长 / 微步长 = $1.0$ / $0.05\,\mathrm{s}$”，即每个 $1.0\,\mathrm{s}$ 决策时隙划分为 20 个离散推进微步。然而检查 [queued_service.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L28) 可见 `Protocol` 数据类默认设定 `dt: float = 0.1`（每时隙 10 个微步），且主实验驱动脚本 [queued_joint_rl_experiment.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L129) 直接以默认 `Protocol()` 构建环境，从未覆盖 `dt=0.05`。
- **修复验证建议** 将 [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L404) 表 5-4 中的 `1.0 / 0.05` 更正为与源码一致的 `1.0 / 0.1`。

### `AUD-02` 第 5 章旧单次 Lindley 排队近似的验证集 RMSE 数值抄录错误
- **状态分类** `confirmed_internal_error`
- **严重程度** Major
- **论文位置** [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L144)
- **代码与数据位置** [tandem_approximation_audit.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/tandem_approximation_audit.json#L1-L13)；[protocol.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/protocol.json#L13-L23)
- **不一致事实** [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L144) 正文写道，在 932 个验证动作对上，“释放时间近似与三步 Engine 参照的相关系数为 $0.9996$，平均绝对误差（MAE）为 $0.2758$，均方根误差（RMSE）为 $0.6444$；旧单次 Lindley 近似对应相关系数 $0.99963$、MAE $0.2704$、RMSE $0.6382$”。核对 [tandem_approximation_audit.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/tandem_approximation_audit.json#L8-L12) 原始记录，`static_single_pass_lindley` 的真实数值为 `correlation = 0.99962995`、`mae = 0.27038627`、**`rmse = 0.61804065`**（四位小数应为 **`0.6180`**，而非 `0.6382`）。
- **修复验证建议** 将 [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L144) 中的 `RMSE 0.6382` 修正为 `RMSE 0.6180`，并统一两套近似指标的相关系数小数保留位数（如均保留五位小数 `0.99957` 与 `0.99963`，或四位小数 `0.9996`）。

### `AUD-03` 第 1 章“毫秒至亚秒级在线决策”声明与第 5 章实测计时数据直接冲突
- **状态分类** `confirmed_internal_error`
- **严重程度** Major
- **论文位置** [chap01.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex#L111) 对照 [ch5_dynamic_cost.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/tables/ch5_dynamic_cost.tex#L9-L16) 及 [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L503-L515)
- **代码与数据位置** [confirmation_analysis.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/confirmation_analysis.json)
- **不一致事实** [chap01.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex#L111) 声称第 5 章方法“能在毫秒至亚秒级时间内完成在线协同决策”。但在 `confirmation_analysis.json` 与表 5-8 中，全部 7 种受测方案的单步平均耗时介于 `0.439 s`（联合蒸馏）至 `1.602 s`（在线规划基线）之间，没有任何方案落入毫秒级（$10^{-3}\sim 10^{-2}\,\mathrm{s}$）区间；且 P95 耗时介于 `1.220 s` 至 `5.998 s`，单步超过 $1.0\,\mathrm{s}$ 宏时隙的比例高达 `15.0%` 至 `29.7%`（其中联合微调为 `16.9%`，微调 Top-6 MPC 为 `24.5%`）。
- **修复验证建议** 删除 [chap01.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex#L111) 中的“毫秒至”，并在摘要、第 1 章与第 6 章涉及在线耗时的表述中明确限定为“零决策等待仿真原型下的亚秒级平均计算耗时”。

### `AUD-04` 跨章及图文四舍五入尾数不一致（3 处）
- **状态分类** `confirmed_internal_error`
- **严重程度** Minor
- **论文位置**
  1. [chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L32) 对照 [chap03.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex#L649)（表 3-5）与 [chap03.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex#L724)
  2. `docs/figures/图4-8_候选轨迹快照桥接对比.png` 子图 (c) 对照 [chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L488)、[abstract.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex#L14) 与 [chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L21)
  3. [chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L488) 桥接在线下层归一化时延均值
- **代码与数据位置** [confirmation_paired_comparison.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/confirmation_paired_comparison.json)；[bridge_analysis.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/bridge_analysis.json)；[generate_chapter34_figures_20261002.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/scripts/generate_chapter34_figures_20261002.py)
- **不一致事实**
  1. **第 3 章 Jain 公平性差值**：`confirmation_paired_comparison.json` 中 `baseline_mean = 0.86652410`、`candidate_mean = 0.92096738`，精确配对差值 `mean_difference = +0.05444328`，四位小数舍入为 **`+0.0544`**（与表 3-5 及第 3 章正文一致），但 [chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L32) 按舍入后均值相减（`0.9210 - 0.8665`）误写为 **`+0.0545`**。
  2. **第 4 章桥接超时请求数差值**：`bridge_analysis.json` 中在线下层均值为精确的 `9.2375`（三位小数 `9.238`），`mc_itr` 均值为精确的 `6.6050`（`6.605`），精确配对差值为 **`-2.6325`**。Python 银行家舍入（round-to-even）使 `图4-8` 子图 (c) 标题显示为 **`均值 -2.632`**，而摘要、第 4 章正文与第 6 章均写为 **`-2.633`**。
  3. **第 4 章桥接在线下层平均归一化时延**：`bridge_analysis.json` 50 个任务汇总下 `online_lower.mean_normalized_latency = 1.25847139`，三位小数应为 **`1.258`**（`mc_itr = 0.98015016` 即 `0.980`，差值 `-0.27832123` 即 `-0.278`），而 [chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L488) 误写为“由 `1.259` 降至 `0.980`（降低 `0.278`）”。
- **修复验证建议** 将 [chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L32) 的 `+0.0545` 改为 `+0.0544`；将 [chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L488) 的 `1.259` 改为 `1.258`；统一 `图4-8(c)` 与正文对 `-2.6325` 的舍入规则（或保留四位小数 `-2.6325`）。

### `AUD-05` Jain 公平性指标的论文数学定义与 `user_equipments.py` 实际统计逻辑不符
- **状态分类** `confirmed_internal_error`
- **严重程度** Major
- **论文位置** [chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L213-L218)（式 2-15）；[chap03.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex#L256-L263)（式 3-21）；[notation.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/notation.tex#L13)
- **代码与数据位置** [user_equipments.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/user_equipments.py#L158-L162)（`is_successful_for_fairness`）；[env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L552-L558)
- **不一致事实** 第 2 章式 (2-15) 与符号表将公平性输入量定义为终端的“达标服务覆盖频率”（按各自任务截止期 $D_m$ 衡量服务请求）；第 3 章式 (3-21) 则写为“被接纳且时延不超过 $\tau$ 的累计频次”。核查源码 [user_equipments.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/user_equipments.py#L158-L162) 发现，`is_successful_for_fairness()` 既未按请求类型过滤（**同时计入了约占总数三分之二的内容请求与三分之一的服务请求**），也未使用各请求的真实截止期 $D_m \in [0.65, 2.10]\,\mathrm{s}$，而是硬编码判定 `total_latency <= config.TIME_SLOT_DURATION`（$1.0\,\mathrm{s}$）。这不仅导致第 2 章与第 3 章公式互相矛盾，也使论文读者误以为表 3-5 的 Jain 公平性仅度量计算服务请求按各自截止期的完成均衡性。
- **修复验证建议** 在不重跑实验的前提下，须修正第 2 章 2.3.3 节、`notation.tex` 与第 3 章 3.2.4 节的公平性文字与公式定义，如实写明其统计口径为“全体终端各类请求（含服务与内容请求）在单时隙步长 $\tau = 1.0\,\mathrm{s}$ 内完成交互的累计频率均衡度”。

### `AUD-06` 第 2 章总纲网络/策略公式与第 3、5 章模型源码存在两处结构偏离
- **状态分类** `confirmed_internal_error`
- **严重程度** Major
- **论文位置** [chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L321)（集中式价值函数 $V_\phi(O_t)$）、[chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L325-L332) 对照 [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L150-L168)
- **代码与数据位置** [queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L216-L245)；[offload_mappo.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/offload_mappo/offload_mappo.py#L173-L187)
- **不一致事实**
  1. **未定义的幻影缩放系数**：[chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L325-L332) 在介绍第 5 章双分支策略时引入了先验权重 $\beta_m, \beta_a$ 与反馈门控系数 $\eta_{\mathrm{fb}}$，但这三个符号在全文从未定义；[queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L216-L245) 直接计算 `motion_log_prior + motion_delta` 与 `alloc_log_prior + alloc_delta`（恒有 $\beta_m=\beta_a=1$），仅通过布尔开关 `use_expectation_feedback` 控制是否将 $\mathbf C_{\mathrm{upper}}$ 置零（即 $\eta_{\mathrm{fb}}\in\{0,1\}$）。
  2. **第 3 章下层 Critic 并非全局联合状态集中式 Critic**：第 2 章式 (2-33) 将 MAPPO 统一表述为联合全局状态价值函数 $V_\phi(O_t)$，且第 3 章始终称下层为“无注意力请求级 MAPPO”；但核查 [offload_mappo.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/offload_mappo/offload_mappo.py#L173-L187) 的 `OffloadMLPCritic` 可见，其输入张量形状为 `[B, N, obs_dim]`（单机局部 1405 维观测经 `1405 -> 256 -> 256 -> 1` 独立前向传播），并未拼接或聚合其余 4 架无人机的观测，在架构上属于共享参数的局部观测 Critic（IPPO 式价值估计），与上层真正跨机聚合的 `CentralizedAttentionCritic` 不同。
- **修复验证建议** 删除 [chap02.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L325-L332) 中多余的 $\beta_m, \beta_a$ 并注明 $\eta_{\mathrm{fb}}\in\{0,1\}$ 为消融开关；在 3.3.3/3.3.4 节补齐下层 `OffloadMAPPO` 网络结构时，准确说明下层 Actor 采用共享主干加百槽逐请求评分头、下层 Critic 采用共享参数单机局部观测多层感知机。

### `AUD-07` 第 5 章定义 5.3.1 第一阶段非工作守恒覆写、`0.85/0.15` 死分支及上下文重复特征
- **状态分类** `confirmed_internal_error`
- **严重程度** Major
- **论文位置** [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L103)（表 5-2）、[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L116-L143)（定义 5.3.1 及计分规则）
- **代码与数据位置** [queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L139-L145)、[queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L224-L240)、[queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L380-L390)；[queued_service_business.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_service_business.py#L16)；[queued_service.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L223-L227)
- **不一致事实**
  1. **第一阶段通信递推违反工作守恒且与真实环境带宽共享模型不一致**：[queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L224-L240) 按静态 `(deadline, uid)` 顺序遍历任务，用 `t_bh_free = t_tr_done` 和 `t_inter_free = t_tr_done` 直接覆写全局通道空闲时刻。当截止期较早的任务尚在慢速上传（`t_up_done` 较大）而截止期稍晚的任务已完成上传（`t_up_done = 0`）时，该算子强行让转发信道空等前者，并未实现第 5 章与第 6 章所宣称的“消除未就绪任务阻塞已就绪任务的工作守恒近似”；同时，真实环境 `Engine._rates`（[queued_service.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L223-L227)）对机间链路与回传链路是按活跃有向节点对等分总速率（`rate / len(group)`），而近似算子将其视为全网单一串行队列。
  2. **`0.85 / 0.15` 窗口外折减分支在当前协议参数下恒不可达**：[queued_service_business.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_service_business.py#L16) 生成的所有任务相对截止期均满足 $D_n \in [2.0, 3.0]\,\mathrm{s}$，因此任意已到达任务的剩余期限恒有 $s_n = d_n - t \le 3.0\,\mathrm{s}$，故 $\min(3.0, s_n) \equiv s_n$。由此导致 [queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L139-L145) 与 [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L142) 描述的 $\min(3.0, s_n) < \tilde\tau_{n,j}^{\mathrm{done}} \le s_n$（`+0.85 / +0.15`）条件对所有现存任务恒为假（数学死分支），因此单任务近似完成分实际上退化为 $\{0, 1\}$ 整数计数。
  3. **公共上下文特征重复拼接**：[queued_joint_rl.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L380-L390) 中 `context` 向量的第 3 维与第 9 维完全相同（均为 `arrival_stats.win3 / 60.0`），表 5-2 如实反映了代码内容，但未说明该冗余通道系特征拼接遗留。
- **修复验证建议** 将定义 5.3.1 准确表述为“第一阶段单通道静态序尾时刻递推 + 第二阶段节点释放时间工作守恒 EDF 推进”的两阶段近似，如实说明在 $D_n \le 3.0\,\mathrm{s}$ 参数设定下窗口外折减分支自动退化为二值计分，并在表 5-2 表注中简短注明公共上下文索引 9 为历史特征对齐保留的重复通道。

### `AUD-08` 第 3 章对称双分量分解的聚合层级歧义与 100-UE 惩罚系统时延公式缺失
- **状态分类** `aggregation_ambiguity`
- **严重程度** Major
- **论文位置** [chap03.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex#L648)（表 3-5 第 5 行）、[chap03.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex#L724)；[chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L19)
- **代码与数据位置** [confirmation_units.csv](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/confirmation_units.csv)；[env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L656-L658) 与 [env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L706)；[config.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/config.py)（`NON_SERVED_LATENCY_PENALTY = 20.0`）
- **不一致事实**
  1. **对称分解聚合层级说明**：对 [confirmation_units.csv](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/confirmation_units.csv) 中 50 个配对评估单元逐单元应用对称分解再取算术平均，可得接纳项 `+4.8664 pp`（`+4.87 pp`）、条件完成率项 `-2.0155 pp`（`-2.02 pp`），之和严格等于单元平均 DSR 差值 `+2.8509 pp`（`+2.85 pp`），证明正文数值真实准确；但若读者直接将表 3-5 宏平均（`0.4769, 0.5751, 0.5170, 0.4786`）代入式 (2-16)，会算出 `+4.89 pp` 与 `-2.02 pp`（和为 `+2.86 pp`），需在正文点明“逐单元分解后取 50 单元均值”。
  2. **惩罚系统时延超过惩罚上限 `20 s` 的反常算式**：单请求失败惩罚常数为 `NON_SERVED_LATENCY_PENALTY = 20.0 s`，按常理“每生成服务请求平均惩罚时延”不应超过 `20.0 s`，但表 3-5 汇报值为 `33.73 s` 与 `28.15 s`。核查 [env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L656-L658) 与 [env.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/env.py#L706) 发现，代码对当步**全部 100 个地面终端中未按时完成请求的终端（包含约 66 个发起内容请求或未获服务计算请求的终端）**全部累加了 `20.0 s` 惩罚，随后在 `env.py:706` 却除以当步仅约 `33.8` 个“生成服务请求数”，导致分子口径（全体 100 个终端）与分母口径（约 33.8 个服务请求）错位放大了约 3 倍，且该公式在论文正文中完全未定义。
- **修复验证建议** 在 3.2.4 节或表 3-5 表注中如实写出该综合惩罚时延指标的分子（全体终端总惩罚时延和）与分母（生成服务请求总数）定义式，避免审稿人因数值突破 `20 s` 上限而质疑数据造假或统计错误；同时为 `+4.87 / -2.02 pp` 注明逐评估单元配对平均口径。

### `AUD-09` 第 4 章桥接实验移除在线策略受控的 3% MBS 配额惩罚且正文隐去自利贪心基线
- **状态分类** `aggregation_ambiguity`
- **严重程度** Major
- **论文位置** [chap04.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex#L478-L488)；[abstract.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex#L14)；[chap06.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex#L21)
- **代码与数据位置** [bridge_analysis.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/bridge_analysis.json)；[chapter4.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/chapter4.py#L21)；[config.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/config.py)（`MBS_MAX_LOAD_SHARE = 0.03`）
- **不一致事实** 在线无注意力 MAPPO 下层在训练中受 `MBS_MAX_LOAD_SHARE = 0.03` 违例惩罚乘子严格限制（确认集实际 MBS 卸载占比仅 `4.73%`）；而离线桥接重评估代码 [chapter4.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/chapter4.py#L21) 设定 `mbs_penalty = 0.0`，允许离线求解器无上限使用在协议 I 中既不均分回传带宽也不均分 $200\,\mathrm{GHz}$ 算力的理想 MBS 通道。进一步汇总 [bridge_analysis.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/chapter34_20261004/bridge_analysis.json) 全部 50 个任务（400 个快照）的均值可见，完全忽略同批算力外部性的单遍自利基线 `greedy_single_pass`（`7.6575`）与 `iterative_base`（`7.4775`）已分别消除了在线下层（`9.2375`）至 `mc_itr`（`6.6050`）总超时降幅（`-2.6325`）中的 **`60.02%`**（`-1.5800`）与 **`66.86%`**（`-1.7600`），而 `mc_itr` 相对 `iterative_base` 的真实外部性净削减量为 **`-0.8725`**（且 `mc_itr` 的平均归一化时延 `0.98015` 略高于 `iterative_base` 的 `0.97968`）。正文 4.4.3 节省略了中间基线数据，易引导读者将 `-2.633` 全部归功于批量外部性迭代。
- **修复验证建议** 在 4.4.3 节补齐桥接集全部 5 种方法的完整数值表（`9.238, 11.983, 7.658, 7.478, 6.605`），明确剥离“移除在线 MBS 配额惩罚与单遍贪心修正的收益（`-1.760`）”与“MC-ITR 相对 `Iterative-Base` 的外部性重评估净收益（`-0.873`）”。

### `AUD-10` 表 5-7（$w_{\mathrm{cf}}=0$ 消融）缺失活跃源数据文件，且上游提议器 `.pt` 检查点本地缺席
- **状态分类** `provenance_gap`
- **严重程度** Major
- **论文位置** [chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L449-L469)（表 5-7 `tab:ch5-cf-ablation`）；[chap05.tex](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L90)
- **代码与数据位置** [source_manifest.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/queued_joint_rl_20261106/source_manifest.json#L8-L11)；`results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/`
- **不一致事实**
  1. 表 5-7 汇报的关闭反事实辅助项（$w_{\mathrm{cf}}=0$）三组确认集完成请求数 `120,759`（`47.92%`）、`120,701`（`47.89%`）与 `121,100`（`48.05%`）在当前活跃目录 `docs/figures/data/queued_joint_rl_20261106/` 及 `results/` 中**不存在对应的原始回合 JSON 或汇总 CSV 文件**（仅在历史审计记录 [data_table_crosscheck.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/thesis_full_audit_20261109/validation/data_table_crosscheck.json) 中记载其来自被后续提交原地覆盖的 `commit 97b34d7`）。
  2. [source_manifest.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/docs/figures/data/queued_joint_rl_20261106/source_manifest.json#L8-L11) 明确标注 `"3 scorer files locally available; 3 proposer .pt files absent"` 以及 `"original rl_rollouts absent locally"`，意味着活跃工作区缺少上游基础提议器权重文件，无法在本地从头重跑第 5 章候选生成与闭环评估。
- **修复验证建议** 从远程实验服务器（`100.69.44.85`）找回或导出 `commit 97b34d7`（$w_{\mathrm{cf}}=0$）的 48 回合配对汇总 CSV/JSON 归入 `docs/figures/data/queued_joint_rl_20261106/`，并核实远程服务器上游 3 个父谱系 `proposer .pt` 检查点的归档路径与完整性。

### `AUD-11` 核心实验数据表、统计区间、回放守恒审计与参考文献引用的全面通过项
- **状态分类** `passed`
- **严重程度** Info
- **核验通过范围**
  1. **第 3 章（协议 I）**：主场景下层无注意力强基线 `0.2507` 对照双注意力 `0.2390`（`-1.16 pp`）、未覆盖占比 `52.13%` / `68.55%`、探索批次 `0.2396 -> 0.2738`（`+3.43 pp`）、独立确认批次 `0.2450 -> 0.2735`（`+2.85 pp`，配对 $t$ 区间 `[+1.65, +4.06] pp`，累计生成请求 `G = 9,805,180` 完全一致）、表 3-5 全部均值与置信区间、以及表 3-6 代表性评估单元（种子 `7541`、工作负载 `2203`：`46.71% -> 60.28%`、`54.86% -> 47.95%`、`25.62% -> 28.90%`）均与 `docs/figures/data/chapter34_20261004/` 精确一致。
  2. **第 4 章（协议 I）**：410 个标准快照评估单元主配置均值（`2.888, 2.227, 2.215, 2.045`，差值 `-0.170` 及区间 `[-0.222, -0.122]`）、表 4-6 全部 9 行负载与算力敏感性数值、Exact-Small 全部 600 个快照构成（`511` 空、`82` 单请求、`7` 多请求，超时均值 `0.025`）均与 `chapter4_analysis.json` 和 `chapter4_units.csv` 逐位吻合。
  3. **第 5 章（协议 II）**：816 个盲测确认回合（每臂 48 回合、`G = 252,024`）在表 5-5 中全部 17 个评估臂的池化完成数、接纳率、条件完成率与逐父谱系完成数（如联合微调 `41,050 / 40,892 / 40,900` 对照联合蒸馏 `40,468 / 40,250 / 40,361`），表 5-6 全部 10 组配对比较及 `M=8` / `M=31` Bonferroni 区间，表 5-8 全部耗时均值、P95、超 `1 s` 比例与前瞻调用计数，以及 `decomposed_independence_audit.json`（最大跨分支依赖度 `0.0`）和 `rollout_replay_audit.json`（`1,152` 回合零误差）均 100% 核对无误。
  4. **参考文献引用覆盖率与编译日志**：联合匹配 `\cite{...}` 与 `\inlinecite{...}` 后，[reference.bib](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/reference.bib) 中全部 **57/57** 条文献均在正文中被实际引用（第 1 章 55 篇、第 2 章 25 篇、第 3 章 2 篇、第 4 章 0 篇、第 5 章 9 篇、第 6 章 0 篇）；`latex/main.log` 无未定义引用（Undefined references/citations）报错。
