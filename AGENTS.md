# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) and Antigravity when working with code in this repository.

## 分支与最新实验证据（2026-11-09 全文去能耗版定稿口径）

- **核心口径声明**：当前论文全文（`latex/main.tex` 及第 1–6 章）已彻底移除能耗（无飞行推进能耗、无计算/通信能耗、无无线携能传输 WPT、无 UE 电池电量与退网机制），全系统聚焦通信、计算与截止期服务指标（DSR = 满足截止时间的服务请求数 $S$ / 全部实际生成的服务请求数 $G$）。
- **第三、四章（协议 I：单时隙批量结算与缓存耦合协议，`canonical_energy_free_v1`）**：
  - 统一驱动脚本：`scripts/analysis/run_chapter34_energy_free.py`；正式图源数据与统计 JSON：`docs/figures/data/chapter34_20261004/`。
  - **第三章（自然覆盖下预接纳需求观测）**：主场景下层无注意力强基线 DSR `0.2507` 优于原完整双注意力 `0.2390`（`-1.16 pp`）；瓶颈诊断显示未覆盖请求占生成请求的 `52.13%`、占失败请求的 `68.55%`；探索阶段 DSR `0.2396 -> 0.2738`（`+3.43 pp`）；10 新训练种子 × 5 新工作负载独立确认（`confirmation_paired_comparison.json`）DSR 从 `0.2450` 提升至 `0.2735`（`+2.85 pp`，配对 $t$ 95% CI `[+1.65, +4.06] pp`，合并计数 `G = 9,805,180` 完全一致）。
  - **第四章（固定快照二元字典序重评估 MC-ITR）**：严格二元字典序目标 $F(\mathbf{a})=(F_1(\mathbf{a}), F_2(\mathbf{a}))$（超时请求总数 $F_1$ 与平均归一化时延 $F_2$，不含任何能耗项）。410 个评估单元（`chapter4_analysis.json`）主配置每快照平均超时数：Heuristic `2.888`、Greedy `2.227`、Iterative-Base `2.215`、MC-ITR `2.045`（相对 Iterative-Base 差值 `-0.170`，95% CI `[-0.222, -0.122]`）；第三章扩展观测轨迹 400 个桥接快照上超时数由 `9.238` 降至 `6.605`（`-2.633`，95% CI `[-2.925, -2.340]`）。
- **第五章（协议 II：跨时隙持续 EDF 排队控制协议，`queued_joint_rl_v2_20261106`）**：
  - 核心实现：`environment/queued_service.py`、`marl_models/queued_joint_rl.py`、`utils/queued_joint_rl.py`、`scripts/analysis/queued_joint_rl_experiment.py`；正式图源数据：`docs/figures/data/queued_joint_rl_20261106/`。
  - 17 个评估臂在 3 个固定父谱系 × 8 个确认工作负载（`955000–955007`）× 2 个动作流 = 816 个盲测确认回合（每臂 48 回合、池化生成请求 `252,024`）上配对评估：初始先验选择器 `uncoordinated_init` `44.30%`，联合冷启动 `joint_rl` `44.25%`，两轮规划蒸馏 `joint_distilled` `48.04%`（`121,079`），联合反事实微调 `joint_rl_finetuned` `48.74%`（`122,842`，较蒸馏提升 `+0.70 pp`，31 项 Bonferroni CI `[+0.39, +0.98] pp`；纯在策略暖启动 `cf=0` 为 `48.05%`，反事实辅助净贡献 `+0.69 pp`；池化接纳率 `52.58%` 即 `132,524/252,024`，池化条件完成率 `92.69%` 即 `122,842/132,524`，48 回合均值 `92.73%`），在线规划基线 `quality_amortized` `48.56%`（`122,382`，池化接纳 `130,436`，池化条件完成率 `93.83%`），微调 Top-6 MPC `joint_rl_mpc` `49.31%`（`124,280`，较在线规划提升 `+0.75 pp`，31 项 Bonferroni CI `[+0.25, +1.42] pp`，平均耗时从 `1.602 s` 降至 `0.722 s`，降低 `54.9%`）。
- **严禁事项**：严禁在论文或分析中重新引入旧版能耗公式、旧版三元字典序能耗目标或已废弃的过渡分支缩写（如旧 `SF-HLM`）。

## 当前活跃文件架构与 `recent/` 历史归档索引（防混淆红线）

为彻底杜绝历史迭代产物对当前去能耗版论文造成上下文污染，本仓库已于 2026-11-09 执行严格瘦身：
- **主工作区仅保留当前去能耗版论文（第 1–6 章）正在使用的源码、数据、图表与测试**；
- **所有历史迭代文件均按原相对目录结构归档至 `recent/` 目录下**（完整迁移清单见 `docs/provenance/RECENT_ARCHIVE_MANIFEST_20261109.json` 与 `recent/README.md`）。

### 1. 主工作区活跃文件对照表（当前论文唯一权威来源）

| 模块 / 章节 | 活跃代码与脚本 | 正式数据源与检查点 | 论文源码与图表 |
|---|---|---|---|
| **全仓基础配置** | `config.py`、`paths.py`、`scripts/maintenance/check_layout.py`、`scripts/maintenance/root_allowlist.py` | — | `latex/main.tex`、`latex/sysusetup.tex`、`latex/reference.bib`、`latex/docs/{abstract,chap01,chap02,chap06,notation}.tex` |
| **第 2 章（基础示意）** | `docs/figures/scripts/generate_foundation_figures.py`、`docs/figures/scripts/thesis_figure_style.py` | — | `docs/figures/图2-1_MEC三层架构示意图.*`、`图2-2_UAV辅助通信应用场景.*`、`图2-3_强化学习智能体环境交互框架.*` |
| **第 3 章（协议 I：扩展需求观测）** | `environment/{env,uavs,user_equipments,comm_model,request_types}.py`、`marl_models/{attention_mappo,offload_mappo,vanilla_mappo,joint_mappo,ippo_baseline,uncoordinated_greedy_baseline,random_baseline,uniform_baseline}/`、`run_canonical_experiment.py`、`run_hierarchical_mappo_experiment.py`、`analyze_canonical_suite.py`、`scripts/analysis/run_chapter34_energy_free.py`、`docs/figures/scripts/generate_chapter34_figures_20261002.py` | `docs/figures/data/chapter34_20261004/{canonical_baseline_analysis.json,bottleneck_review.json,pilot_analysis.json,confirmation_paired_comparison.json,confirmation_training.csv,confirmation_units.csv}` | `latex/docs/chap03.tex`、`docs/figures/图3-1_系统与自然覆盖示意图.*` 至 `图3-9_接纳率与接纳后质量配对比较.*` |
| **第 4 章（协议 I：MC-ITR 二元字典序分配）** | `utils/{chapter4,assignment_search,snapshot_evaluator,canonical_experiment,formal_protocol}.py`、`run_chapter4_suite.py`、`scripts/analysis/run_chapter34_energy_free.py`、`docs/figures/scripts/generate_chapter34_figures_20261002.py` | `docs/figures/data/chapter34_20261004/{chapter4_analysis.json,chapter4_units.csv,bridge_analysis.json}` | `latex/docs/chap04.tex`、`docs/figures/图4-1_批量算力共享与迁移外部影响示意.*`、`图4-2_MC-ITR算法流程图.*`、`图4-3_主配置质量时延耗时对比.*`、`图4-4_相对Iterative-Base配对差值.*` 至 `图4-8_候选轨迹快照桥接对比.*` |
| **第 5 章（协议 II：跨时隙持续 EDF 排队控制）** | `environment/queued_service.py`、`marl_models/{queued_joint_rl,queued_joint_training,queued_planner_distillation}.py`、`utils/{queued_joint_rl,queued_joint_training,queued_planner_distillation,queued_quality_amortization,queued_waiting_rescue,queued_arrival_adaptation}.py`（及上游特征依赖模块）、`scripts/analysis/{queued_joint_rl_experiment,review_queued_joint_rl}.py`、`docs/figures/scripts/generate_queued_joint_thesis_figures.py` | `docs/figures/data/queued_joint_rl_20261106/`、`results/chapter5_dynamic_service_20261028/{queued_joint_rl_v2_20261106,queued_long_value_20261102,queued_score_control_20261104}/` | `latex/docs/chap05.tex`、`latex/docs/tables/ch5_dynamic_{main,effects,cost}.tex`、`docs/figures/ch5_dynamic_01_protocol.*` 至 `ch5_dynamic_06_decision_cost.*` |
| **文献与全面审计** | `scripts/literature/verify_thesis_references.py` | `docs/literature/`、`output/thesis_full_audit_20261109/`、`output/thesis_writing_20261004/` | — |

### 2. `recent/` 历史归档目录使用规范（AI 严禁主动读取）

- **归档范围**：`recent/` 内按原相对路径存放了 405 项历史迭代资产，包括：
  1. 早期含能耗/强制接纳/三元字典序说明文档（`recent/CLAUDE.md`、`recent/wodexuexi_AI_GUIDE.md`、`recent/CANONICAL_SEMANTICS_SPEC.md`、`recent/CHAPTER4_PROTOCOL.md`、`recent/attention mappo项目.md`）；
  2. 已废弃的 `SF-HLM` / `dynamic_service` 过渡版环境、模型、脚本与测试（`recent/environment/dynamic_service*.py`、`recent/marl_models/dynamic_service*.py`、`recent/scripts/analysis/dynamic_service*.py` 等）；
  3. 第 5 章早期的中间探索实验目录（`recent/results/chapter5_dynamic_service_20261028/` 下的 38 个早期实验子目录）及 2026 年 6–9 月旧版含能耗实验结果（`recent/results/baseline_matrix_*`、`recent/results/final_force_admission` 等）；
  4. 旧版未使用的图件与绘图脚本（`recent/docs/figures/图5-*`、`recent/docs/figures/图4-3_主配置质量能耗耗时对比.*`、`recent/latex/docs/figures/`、中大模板 VOC2007 示例图 `recent/latex/image/{appendix1,chap03,chap04}` 等）。
- **强制红线**：
  - **在常规论文修改、审稿、画图、测试或代码维护任务中，严禁读取、搜索或引用 `recent/` 目录下的任何文件**，以免把旧版能耗公式、强制接纳或 `SF-HLM` 口径重新混入当前工作区；
  - **仅当用户明确要求找回某个旧版文件、对比历史版本或查阅早期实验记录时**，才按原相对路径到 `recent/<原相对路径>` 中查找或恢复。

## 文件位置与生成物规则

创建或修改文件前必须遵循本节；用户明确指定位置时遵从用户，不为常规归类重复询问。

1. **创建前先查找**：搜索已有同用途文件，优先更新；不要另建 `final_v2`、`new` 等重复版本。
2. **根目录默认禁止新增文件或目录**：现有入口仅为兼容保留。新脚本、报告、数据、图片、日志按下表归类。确需新增根目录项时，必须有用户明确要求，并同步更新 `scripts/maintenance/root_allowlist.py`；不得为了让检查通过自动扩充白名单。
3. **按用途归类**：实验指标 JSON 属于 `results/`，任务验收 JSON 属于任务的 `validation/`，长期报告的配套表格或 JSON 与报告同目录。
4. **长期资料与任务产物分开**：长期项目文档放 `docs/`，一次性交付放 `output/`，可丢弃的中间文件放 `.tmp/`。不要默认生成任务未要求且无长期价值的总结报告。
5. **一个任务一个输出目录**：采用 `output/<主题>_YYYYMMDD/`，后续修改复用该目录；同日独立任务重名时增加 `_02` 等序号。根据需要创建 `deliverables/`、`validation/`、`assets/`，不创建空目录凑结构。已有验证目录按历史兼容保留。
6. **路径不依赖启动位置**：仓库内部路径从 `paths.py` 的 `REPO_ROOT` 推导，禁止写死本机或服务器仓库绝对路径。外部输入可通过参数指定；保留旧外部输入默认值时说明兼容原因。Shell 脚本先确定自身与仓库位置。
7. **移动必须修复引用**：更新代码、现行文档链接及命令；历史日志、源码锁、实验清单保持原始内容，通过迁移清单解释旧路径。不得因整理而重新生成或覆盖科研结果。
8. **交付前检查**：核对本任务新增文件的位置、重复文件与失效引用，运行 `python3 scripts/maintenance/check_layout.py`，提供成果链接。该检查只强制检查根目录白名单，不替代子目录归类与引用检查。不删除其他任务的文件。

| 目录 | 唯一职责与落盘规则 |
|---|---|
| 根目录 | 项目说明、AI 指令、配置、现有 Python/Shell 入口及实验协议；允许项见白名单 |
| `environment/`、`marl_models/`、`utils/` | 仿真、模型及共享逻辑 |
| `scripts/analysis/` | 新增的可复用实验分析脚本 |
| `scripts/literature/` | 新增的可复用文献处理脚本；现有根目录入口保留 |
| `scripts/maintenance/` | 仓库检查、维护脚本及配置 |
| `tools/` | 独立转换、导出工具；不要与 scripts 重复实现同一功能 |
| `tests/` | 测试及验证代码 |
| `docs/plans/` | 工作计划，优先更新已有同主题计划 |
| `docs/reports/` | 长期审计、修复、验收报告及配套表格、JSON |
| `docs/provenance/` | 历史工程证据、迁移清单；历史事实不随当前状态改写 |
| `docs/literature/extracted/` | 文献提取文本 |
| `docs/literature/analysis/` | 文献分析资料及结果 |
| `docs/figures/` | 正式论文图及源数据，绘图脚本沿用 `docs/figures/scripts/` |
| `latex/` | 论文源码及既有编译产物；保持编译路径 |
| `output/<主题>_YYYYMMDD/deliverables/` | PPT、文档等任务最终交付物 |
| `output/<主题>_YYYYMMDD/validation/` | 任务验证日志、检查结果 |
| `output/<主题>_YYYYMMDD/assets/` | 仅供本任务使用的素材 |
| `.tmp/<主题>/` | 可重新生成的预览、临时提取、一次性探针 |
| `results/`、`saved_models/`、`train_logs/` | 实验结果、模型检查点、训练日志；沿用实验约定的子目录 |
| `recent/`、`logs/`、`tasks/` | 历史归档与兼容目录（`recent/` 存放已归档的旧版文件，常规工作严禁混用） |

## Superpowers 工作流程

本项目参考 [Superpowers](https://github.com/obra/superpowers) 工作流程，按下述适用范围执行，不将完整开发流程套用于所有任务。以下适配规则优先于本项目引用的通用工作流程建议，但不覆盖系统、平台权限或用户当前明确设置的批准要求。

### 自主执行与批准边界

- 只读检查、解释、审阅和运行已有验证不需要设计审批。只要求审阅或建议时不修改文件。
- 新增功能、修改代码行为：用户明确要求实现或修复时，视为已授权在该请求范围内设计并实施。只有方案存在实质歧义，或会改变范围、外部依赖、关键科研结论、持久化数据或不可逆影响时，才先呈现设计并获得批准；已批准且范围未变的设计不重复确认。用户要求提纲或计划先行时，在批准后实施；用户已批准实施时，不再为执行方式或重复展示同一方案额外暂停。
- 先从现有文件、上下文和已确认的选择中解决疑问。只有缺失信息会实质改变正确性、范围、关键科研事实或授权时才提问；一般实现细节遵循项目惯例并说明必要假设，不要求用户代做技术选型。
- 保留具体 Skill 的明确批准门槛，例如首次无法确定绘图后端、研究核心主张或证据存在实质歧义、持久化秘密、外部投递或未授权的范围扩展。用户直接要求重写或重构时，在该范围内执行，不再为同一结构变更重复批准。不得以自主执行为由猜测实验结果或绕过批准。
- 预期的 TDD RED 失败不是阻塞。意外失败先在已批准范围内诊断和修复；超出设计、权限或缺少不可推断的用户信息时，仅暂停受影响步骤并报告原因，继续可独立完成的工作。
- 提交、推送、合并、创建 PR、丢弃修改和清理工作树由独立授权控制，不因任务完成而自动执行。保留破坏性操作的明确确认；不撤销他人改动，不清理非本任务创建的工作区。

### 核心工作流程

1. **brainstorming** - 对存在实质设计歧义、范围扩展或高影响取舍的新增功能和行为修改，读取上下文、呈现设计并获取用户批准。范围明确的实现或修复直接进入实施；只读任务不进入此流程。
2. **writing-plans** - 对批准后的多步骤实现列出文件路径、修改内容和验证步骤；详细程度与风险相称。计划不是用户要求的最终成果时，计划完成后继续已获批准的实施，不停在执行方式选择上。
3. **test-driven-development** - 在可行且与风险相称时，代码功能和修复使用 RED-GREEN-REFACTOR：先验证测试因预期原因失败，最小实现后确认通过，再重构。遗留代码、随机实验或缺少正式测试套件时，采用针对性回归、烟雾测试或实验级验证并说明覆盖范围。纯文档和指令修改使用内容、引用路径及格式检查，不制造无关代码测试。提交不是测试循环的默认步骤。
4. **requesting-code-review** - 在有意义的检查点根据计划审查并修复问题。关键问题阻止依赖它的步骤或最终交付，但不禁止诊断修复和独立工作。
5. **finishing-a-development-branch** - 仅在用户要求分支集成或清理时使用；普通任务完成后保留工作区并交付结果，不强制提供 Git 操作菜单。

## 项目概述

中山大学硕士论文项目：面向时限计算服务的无人机集群需求感知轨迹规划、批量字典序任务分配与跨时隙动态排队协同控制。系统由 5 架 UAV、100 个 UE、1 个 MBS 在 700×700m 区域运行，采用无能耗（energy-free）建模口径。

## 运行环境与工作边界规范（核心原则）

本项目采用**“本机写作与开发 + 远程服务器跑实验”**的双机协同架构，严禁在本机执行重载实验训练：

### 1. 本机环境 (Local Host)
- **核心职责**：代码编写、静态代码审计、Git 版本管理、LaTeX 论文编写与编译排版（`cd latex && make main` 或 `xelatex`）、轻量测试与图表查看。
- **限制与约束**：
  - **严禁在本机启动长时间或大规模 MARL 训练实验**；
  - 本机仅负责轻量静态检查、纯逻辑分析与文档/LaTeX 编译。

### 2. 远程实验服务器 (Remote Server: 100.69.44.85)
- **核心职责**：所有 MARL 模型训练、基线算法比对、消融实验、敏感性分析、以及生成全量实验数据（`results/`）。
- **连接方式**：`ssh 100.69.44.85`（端口 2222，用户 `PengYanghan`）。
- **工作目录**：`~/Lunwen/wodexuexi`

## 核心架构与章节对应

1. **第三章（自然覆盖下需求感知轨迹控制）**：
   - 环境与模型：`environment/env.py`、`marl_models/attention_mappo/`、`marl_models/offload_mappo/`。
   - 上层观测模式：`UPPER_UE_OBSERVATION_MODE = "associated_plus_uncovered_in_sense"`（保持 30 槽与 $100\,\mathrm{m}$ 物理覆盖半径不变，按距离补入 $460\,\mathrm{m}$ 感知范围内未覆盖 UE）。
   - 共享奖励：`reward = W_FAIR*jfi - W_LAT*norm_latency + W_DSR*dsr - 0.5*mbs_Load`（无能耗项、无离线率项）。
2. **第四章（固定快照二元字典序重评估 MC-ITR）**：
   - 实现：`utils/chapter4.py`、`utils/assignment_search.py`、`scripts/analysis/run_chapter34_energy_free.py`。
   - 严格二元字典序目标：$F(\mathbf{a}) = (F_1(\mathbf{a}), F_2(\mathbf{a}))$，第一层为超时请求数，第二层为平均归一化时延。
3. **第五章（跨时隙持续排队中的规划知识蒸馏与反事实辅助 PPO）**：
   - 实现：`environment/queued_service.py`、`marl_models/queued_joint_rl.py`、`utils/queued_joint_rl.py`、`scripts/analysis/queued_joint_rl_experiment.py`。
   - 核心机制：释放时间工作守恒串联排队近似（Stage 1 通信释放时刻 + Stage 2 节点先占式 EDF 计算）、候选等变双分支联合策略（$\mathbf{C}_{\mathrm{upper}}(s,k)\in\mathbb{R}^4$ 期望运动反馈）、两轮 DAgger 规划蒸馏、共同随机数（CRN）三步反事实辅助 PPO 微调、零在线前瞻与神经引导 Top-6 MPC。
