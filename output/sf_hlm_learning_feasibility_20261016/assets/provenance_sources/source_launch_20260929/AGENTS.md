# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

### 使用方法

- 按上面的适用范围选择流程，不因 Skill 名称匹配而扩大任务范围。
- 当前任务已确认的设计、术语、后端和输出要求持续有效，除非用户改变要求或新信息造成实质冲突。
- 必需的 Skill 资源缺失时报告具体路径，不声称已经加载；继续不依赖该资源的工作，受影响部分标记未完成，不跳过伦理、证据核验或明确的批准规则。

### 调用方式

优先使用当前平台实际提供的 Skill 加载方式；没有 `Skill` 工具时，读取可用的 `SKILL.md` 并遵循其指令。以下名称用于识别流程，不代表本会话一定提供同名工具或已安装相应 Skill：
- `/superpowers:brainstorming` - 头脑风暴和设计
- `/superpowers:writing-plans` - 编写实现计划
- `/superpowers:test-driven-development` - 测试驱动开发
- `/superpowers:requesting-code-review` - 请求代码审查
- `/superpowers:finishing-a-development-branch` - 完成开发分支

没有子代理工具时，普通代码审查可用显式自审替代，并注明不是独立审查。用户明确要求独立审查时不得以自审冒充完成；继续其他已授权工作并报告该审查缺口。不要为适配工具名称擅自安装插件、修改全局配置或创建新的用户任务。

### 完成条件

- 完成表示请求范围内的工作和相应验证已完成，并提供可访问的结果及影响使用的限制；除非用户明确要求的交付物就是计划、大纲、脚本或启动状态，不能把它们当作最终产物。
- 检查发现的问题先在授权范围内修复。验证无法执行时，报告已完成部分、未验证部分和具体阻塞，不声称全部通过；无关的历史失败单独记录，不擅自扩展修复范围。
- 长时间实验区分“已启动”“运行中”“已完成且验证”。用户只要求启动时核验进程和日志即可交付启动状态；用户要求实验结果时需等待并核验结果。未来提醒或持续监控仅在用户明确要求时使用平台支持的自动化，并遵循其授权规则；不承诺不存在的后台跟进。

## 项目概述

中山大学硕士论文项目：基于 Attention-MAPPO 的多无人机 MEC 轨迹与任务卸载协同优化方法。系统由 5 架 UAV、100 个 UE、1 个 MBS 在 700×700m 区域运行。使用 Python 3.9 + PyTorch 2.6 实现。

## 运行环境与工作边界规范（核心原则）

本项目采用**“本机写作与开发 + 远程服务器跑实验”**的双机协同架构，严禁在本机执行重载实验训练：

### 1. 本机环境 (Local Host: macOS Darwin arm64)
- **核心职责**：代码编写、静态代码审计、Git 版本管理、LaTeX 论文编写与编译排版（`cd latex && make main` 等）、轻量测试与图表查看。
- **限制与约束**：
  - 本机为 Apple Silicon (Darwin arm64) 架构，仓库内置的 `.venv` 为 Linux x86_64 二进制，**严禁在本机直接尝试运行该 `.venv/bin/python`**；
  - **严禁在本机启动长时间或大规模 MARL 训练实验**；
  - 本机仅负责无 PyTorch 依赖的轻量静态检查、纯逻辑分析与文档/LaTeX 编译。

### 2. 远程实验服务器 (Remote Server: 100.69.44.85)
- **核心职责**：所有 MARL 模型训练、基线算法比对、消融实验、敏感性分析、以及生成全量实验数据（`results/`）。
- **连接方式**：`ssh 100.69.44.85`（通过本机 `~/.ssh/config` 配置的免密别名直连，端口 2222，用户 `PengYanghan`）。
- **工作目录**：`~/Lunwen/wodexuexi`
- **运行环境**：服务器端专用 Conda/Python 环境（如 `/home/PengYanghan/miniconda3/envs/drone/bin/python`）。
- **数据与代码同步流程**：
  1. 本机完成代码或脚本修改后，通过 `git push` 推送至远程仓库；
  2. SSH 登录远程服务器拉取最新代码（`git pull`），并在远程后台启动实验（使用 `nohup` 或 shell 脚本）；
  3. 实验完成后，将远程生成的 `results/` 与图表数据同步回本机（通过 `scp` / `rsync` / git），供本机 LaTeX 编译与论文撰写引用。

## 常用命令

### Python 实验运行（必须在远程服务器 100.69.44.85 上执行）

```bash
# 1. 本机通过 SSH 登录远程实验服务器
ssh 100.69.44.85
cd ~/Lunwen/wodexuexi

# 2. 激活服务器端运行环境
source .venv/bin/activate
# 或使用服务器端 Conda:
# conda activate drone

# 运行单次分层 MAPPO 训练（默认 attention_mappo 上层 + constrained_attention_offload_mappo 下层）
python run_hierarchical_mappo_experiment.py --seed 42 --episodes 200

# 运行 joint MAPPO baseline 训练
python run_joint_end_to_end_mappo_experiment.py --seed 42

# 评估 joint MAPPO baseline
python evaluate_joint_end_to_end_mappo.py

# 运行完整实验套件（3 training seeds × 10 workload seeds）
bash run_linear_v2_full.sh

# 运行 DSR-priority 配置实验
bash run_linear_v2_dsr_strong.sh

# 运行补充实验（vanilla vs attention、attention 可视化、joint MAPPO）
bash run_remaining_three_experiments.sh

# 统计分析（paired t-test、Wilcoxon、95% CI）
python analyze_experiment_statistics.py

# 可视化
python visualize.py --mode snapshot    # 单帧快照 PNG
python visualize.py --mode animate     # 仿真动画 GIF
python visualize.py --mode results     # 实验指标柱状图
```

### LaTeX 论文编译

```bash
cd latex
make main      # 编译论文正文 main.pdf
make pre       # 编译答辩 PPT pre.pdf
make all       # 编译所有文档
make clean     # 清理中间文件
make cleanall  # 清理所有生成文件（含 PDF）
```

## 核心架构：分层双层 MARL

系统采用上下两层决策结构，而非单一联合动作空间：

### 上层：轨迹控制（attention-MAPPO）
- 每架 UAV 为一个智能体，输出连续 2D 动作（方向向量）
- 使用多头交叉注意力（8 heads, hidden_dim=64）建模 UAV-UAV 和 UAV-UE 关系
- 关键文件：`marl_models/attention_mappo/` (attention_mappo.py, agents.py)
- 共享注意力模块：`marl_models/attention.py`（CrossAttentionExtractor, AttentionActorBase, AttentionCriticBase）

### 下层：请求级卸载（constrained attention offload MAPPO）
- 每架 UAV 对至多 30 个请求做离散卸载决策：0=本地, 1=MBS, 2+i=协作 UAV i
- 质量感知动作 mask 屏蔽不可行动作，Lagrange 约束机制处理 DSR/MBS 负载权衡
- 关键文件：`marl_models/offload_mappo/offload_mappo.py`, `marl_models/offload_policy.py`

### Baseline 模型
- `marl_models/vanilla_mappo/` — 无注意力的 MLP 版 MAPPO（消融上层 attention）
- `marl_models/joint_mappo/` — 端到端联合 MAPPO（单 actor 输出轨迹+卸载动作）
- `marl_models/uncoordinated_greedy_baseline/` — 非学习贪心+启发式基线

### 共享基础设施
- `marl_models/buffer_and_helpers.py` — AttentionRolloutBuffer、DiscreteOffloadRolloutBuffer、JointMAPPOBuffer
- `marl_models/utils.py` — get_model() 工厂函数、save_models()
- `marl_models/base_model.py` — 模型基类

## 环境仿真

- `environment/env.py` — 主仿真环境类 Env（step/reset/渲染）
- `environment/uavs.py` — UAV 模型（飞行能耗、计算、通信、WPT、缓存）
- `environment/user_equipments.py` — UE 模型（电池动态、请求生成）
- `environment/comm_model.py` — 自由空间路径损耗信道模型
- `environment/request_types.py` — 服务和内容请求定义

## 配置体系

所有参数集中在 `config.py`，通过 `import config` 在全局共享。实验脚本在运行时可覆盖 config 变量（如 `apply_lower_ablation()` 修改 OFFLOAD_MASK_MODE 等）。路径管理统一通过 `paths.py` 的 `REPO_ROOT` 和 `results_path()`。

## 关键设计约定

- **单位转换**：请求/文件大小以 bytes 记录，无线链路速率以 bit/s，传输时延统一为 `T_tx = 8 * D / R`
- **观测维度**：上层 `OBS_DIM_SINGLE = SELF_OBS_DIM + MAX_UAV_NEIGHBORS * NEIGHBOR_OBS_DIM + MAX_ASSOCIATED_UES * UE_OBS_DIM`；下层 `OFFLOAD_OBS_DIM_SINGLE = 5 + MAX_OFFLOAD_REQUESTS_PER_UAV * OFFLOAD_REQUEST_FEATURE_DIM`
- **奖励函数**：线性形式 `reward = W_FAIR*jfi - W_LAT*norm_latency - W_ENERGY*norm_energy - W_OFFLINE*offline_rate + W_DSR*dsr`，其中 latency 和 energy 做了归一化
- **Lagrange 约束**：下层使用可学习的 lambda_dsr 和 lambda_mbs 拉格朗日乘子，约束 DSR≥target 且 MBS 负载≤ceiling
- **模型命名**：`get_model()` 工厂函数根据字符串名称（如 "constrained_attention_offload_mappo"）返回对应模型实例

## 实验结论与论文证据边界（截至 2026-09-28）

本节记录已完成实验的**结论及其适用范围**，供后续实验和写作查证；`PASS` 只说明对应任务通过验收，不自动证明论文主张。当前论文主口径是自然覆盖接纳：未进入 UAV 100 m 服务半径的请求不被强制处理，DSR = 满足截止时间的服务请求数 / 全部实际生成的服务请求数。训练种子是独立训练重复；同一检查点的工作负载、评估 episode 和请求不能充作独立训练重复。不同源码摘要、请求接纳规则和评估协议下的绝对指标不得直接拼接。UE 电池低于阈值会把潜在服务请求转为能量请求，故实际生成服务请求数具有轻微策略依赖性；配对外生随机流不等于分母完全相同，以潜在请求为固定分母的稳健性检验已于 09-28 完成，结果见下表；它不改变原预定主指标。

### 结果位置、章节关系与实验走向

- 本项目有主工作区和 `coverage-optimization/wodexuexi` 独立实验工作树；在当前机器运行 `git worktree list` 可定位各自根目录。下表的相对路径必须相对于标明的工作树读取，不能把两个工作树中同名相对路径视为同一文件。旧全量、第四章、下层/奖励 pilot 及瓶颈诊断在主工作区；覆盖观测 pilot、独立确认和 09-28 整套方法对比在 `coverage-optimization/wodexuexi` 工作树。结果尚未合并为单一工作区，不应因为某工作树缺少另一方结果就判为实验未完成。
- 拟采用的研究章节是：第三章研究上层轨迹控制与覆盖需求观测；第四章研究固定轨迹/历史快照下的任务卸载，已实现的方法为 MC-ITR；第五章研究完整分层策略的端到端表现。第四章 MC-ITR **没有接入**第五章学习式下层，不可把第五章写成“上层 + MC-ITR”的性能验证。当前实现和冻结的第四章协议均未采用拍卖/竞价算法；所收录的拍卖论文只是文献。`latex/main.tex` 已启用独立的第四、五章；现行 LaTeX 仍需随补充实验更新。
- 研究路线按证据依次变化：旧强制接纳口径不可用于自然覆盖结论 → 修正自然覆盖后发现原完整双注意力不及下层无注意力强基线 → 下层注意力拆分和简单奖励调整未给出稳定改善 → 瓶颈诊断发现未覆盖是主要失败来源 → 扩展上层预动作观测的 pilot 与独立确认显示 100 UE 主场景 DSR 改善 → 09-28 整套方法对比进一步支持其相对现有学习方法的 DSR 优势。每一步的数值、实验单位和不能外推的边界见下表。

| 实验阶段与证据 | 已完成结果 | 结论与边界 |
|---|---|---|
| 2026-06 至 07 旧强制接纳基线、消融、DSR 优先与敏感性；`results/baseline_matrix_force_admission_20260626_resume/`、`results/final_force_admission/evaluation_20260630/`、`results/sensitivity_force_admission/`、`docs/reports/defense_experiment_audit_force_admission_full.md` | 旧协议将未覆盖请求强制关联，处理率为 100%；扩展基线中 proposed DSR 约 0.3762、Joint 0.4163、Uniform 0.4529；旧 DSR 优先配置 DSR 0.3524，低于原配置的 0.3762。旧敏感性下 proposed 在多数设置高于 Vanilla，但启发式常有更高 DSR。 | 这些是**历史强制接纳口径**，可说明旧设计的权衡与修复动机，不能支持现行自然覆盖系统的性能、最优性或摘要结论；旧奖励与公平性排名也不能直接移植。 |
| 2026-09-20 自然覆盖全量首版；`results/canonical_all_20260920/marl/` | 1650 评估任务完成；100 UE 主配置 hierarchical DSR 0.2459、lower_no_attention 0.2548，前者低约 0.89 个百分点；hierarchical 比 vanilla_hierarchical 高约 2.43 个百分点。 | 首版已被 09-23 修正版取代；保留为趋势复核，不作为最新数值引用。 |
| 2026-09-23 自然覆盖修正版全量；`results/canonical_all_20260923_fixed/marl/{status,analysis}.json` | 5 训练种子 × 10 工作负载，主配置 hierarchical DSR 0.2428；Vanilla 0.2186、IPPO 0.2204、Joint 0.2123、lower_no_attention **0.2609**。hierarchical−Vanilla +2.42 个百分点（配对交叉 bootstrap 95% CI +1.60 至 +3.27）；hierarchical−lower_no_attention **−1.81 个百分点**（−2.40 至 −1.17）。去质量 mask、去惩罚相对完整模型的 DSR 差异区间跨 0。 | 分层框架相对若干基线有收益，但**不能声称双层都加注意力最优，也不能把下层注意力或软约束写成已证实提高 DSR**；后续以 lower_no_attention 为强基线。 |
| 同一修正版的 UE 数量与算力敏感性；`results/canonical_all_20260923_fixed/marl/analysis.json` | hierarchical 在 60/80/100/120/140 UE 的 DSR 依次约 0.3567/0.2902/0.2428/0.2032/0.1761；60、80 UE 时不高于 Vanilla。算力 0.6→1.4 倍时 DSR 约 0.1963→0.2706，各设置高于 Vanilla。 | 负载增大时服务率下降，注意力上层的优势依场景而变；该网格**各设置重新训练**，不是固定模型跨场景泛化。 |
| 第四章固定快照优化；主工作区 `results/canonical_all_20260920/chapter4/{status,analysis}.json` | 410 评估单元完成；主设置 MC-ITR 每快照平均 deadline misses 2.055，优于单轮贪心 2.243、迭代基线 2.228、启发式 2.908；平均分配能耗归一化值 2.128，高于迭代基线 1.872；平均搜索耗时约 2.574 s，且不含共享 Greedy 初始化。 | 固定几何与请求下的字典序卸载求解器证据，**不是**长时序 MARL 策略的 DSR、全局最优或 1 s 时隙内可实时执行的证明；能耗和完整计算开销必须同时报告。 |
| 2026-09-24 下层注意力拆分 pilot；`results/lower_attention_pilot_20260924/`、`output/lower_attention_20260924/validation/effect_review.json` | 5×5 配对；完整下层/无注意力/仅 actor 注意力/仅 critic 注意力的 DSR 约 0.2267/0.2439/0.2393/0.2461；各关键配对区间均跨 0。 | 未证明下层注意力有正增益，也未证明其必然无效；不要把单个最高均值当成架构选择的确证。 |
| 2026-09-24 奖励权重 pilot 与 09-26 固定分母奖励 pilot；`results/reward_priority_pilot_20260924/`、`output/reward_denominator_20260926/validation/completion_review.json` | 5×5 奖励权重试验：基线 DSR 0.2567，DSR 权重 5 为 0.2537，能耗参考值调整为 0.2586；固定分母奖励比基线低 0.26 个百分点，95% CI −2.17 至 +1.55。 | 简单调权和分母替换未给出稳定 DSR 改善；不应以“再调奖励”作为主要创新路线。 |
| 2026-09-27 瓶颈诊断；`output/bottleneck_diagnostic_20260927/validation/bottleneck_review.json` | 25 个完整评估单元中 DSR 0.2459，未覆盖率 48.14%，未覆盖占全部失败约 63.84%；50 个冻结快照中，双步质量掩码局部搜索使成功数 362→441，即这些快照 +5.13 个百分点，同时分配能耗 +22.86%。 | 未覆盖是描述性主要失败来源；局部搜索显示卸载层仍有空间，但不是可部署学习策略、长时序增益或上界。覆盖与接纳改善后仍须检查被接纳请求的质量。 |
| 2026-09-27 覆盖观测 G4 探索 pilot；独立 `coverage-optimization/wodexuexi` 工作树下 `results/coverage_observation_g4_20260927/pilot_paired_comparison.json` | 5×5 配对，10 次训练、50 组评估、300 个评估 episode；候选（上层可见感知范围内未覆盖 UE）相对 lower_no_attention：DSR 0.2367→0.2696，+3.30 个百分点，交叉 bootstrap 95% CI +1.29 至 +5.33。 | 探索性发现，不独立支撑论文结论；试验只改上层预动作观测，未改变 100 m 物理接纳半径。 |
| 2026-09-27 覆盖观测独立确认；同一独立工作树下 `results/coverage_observation_confirmation_20260927/confirmation_paired_comparison.json` | 新的 10 个训练种子 × 5 个新工作负载，20 次训练、100 组评估、600 个 episode；候选与强基线 DSR 0.2444→0.2793，**+3.49 个百分点**；训练种子配对 t 95% CI +2.44 至 +4.54，10/10 种子、49/50 配对单元为正。接纳率 0.4996→0.6189，而已接纳请求的截止期满足率 0.4919→0.4537；UAV 能耗均值约降低 6.37%。 | 这是**100 UE 主场景、相对 lower_no_attention 的独立确认**，尚未覆盖所有基线、负载/算力设置，也未证明实际部署可获取预接纳请求信息。DSR 增益主要来自接纳增加，同时已接纳请求质量下降；pilot 与确认的分解机制不同。只改 `UPPER_UE_OBSERVATION_MODE`，仍须审计 460 m 感知下 UE 位置/请求/电量的可获得性、时序和通信代价。源码有内容摘要，但非正式提交锁定发布版。 |
| 2026-09-28 整套方法全量对比；独立工作树 `docs/plans/whole_system_comparison_20260928.md`、`results/whole_system_comparison_20260928/whole_system_paired_comparison.json`、`output/whole_system_comparison_20260928/validation/` | 预先固定的 100 UE 主场景，10 个新训练种子 × 5 个新工作负载；60 次学习式训练、325 个评估任务、1950 个评估 episode，`remote_completion.json` 和 `raw_evaluation_audit.json` 均为 PASS。候选 DSR 0.2798；lower_no_attention 0.2459（差 +3.39 个百分点，训练种子配对 t 95% CI +2.13 至 +4.66）、hierarchical 0.2433、vanilla_hierarchical 0.2153、IPPO 0.2162、Joint 0.2074；候选相对五个预先指定学习对手的 DSR 差值均为正且 Holm 校正双侧 p<0.05。最强非学习参照 Uniform DSR 0.2153，其余四种也已评估。相对 lower_no_attention，候选接纳率 0.5269→0.6248，而接纳后按时完成率 0.4703→0.4491（该次指标区间跨 0）。 | 支持**在该 100 UE 仿真设置、现有方法各自原配置下，候选整套方法的 DSR 优于五个指定学习对手**；非学习方法没有训练种子，不能伪装成 10 次独立训练重复。候选独享 460 m 内未接纳 UE 的位置、当前请求特征和电量观测；尚未证明同信息预算下的架构优势、信息的部署可得性、跨负载/算力稳健性或所有指标均优。此轮 `purpose=pilot` 且仅有源码内容摘要，不称正式提交锁定发布版。 |

| 2026-09-28 固定潜在服务请求分母复核；独立工作树 `scripts/analysis/potential_dsr_robustness.py`、`results/potential_dsr_20260928/potential_dsr_analysis.json`、`output/potential_dsr_20260928/validation/validation.json` | 冻结源码哈希与 325 个原评估单元均验收通过；原 DSR 从计数精确复现。六个学习方法各自 50 单元的共同潜在服务请求数均为 9,526,870；候选实际服务请求 9,525,735、强基线 9,526,728。换成共同固定分母后，候选相对 lower_no_attention 的 DSR 差值 +3.390 个百分点，训练种子配对 95% t CI +2.125 至 +4.655；10/10 种子为正，相对五个预定学习对手的 Holm 校正双侧检验仍均通过。 | 轻微分母内生性未解释原主场景 DSR 增益；这是保留原成功数的替代分母敏感性分析，**不是**让低电量 UE 继续生成服务请求的反事实环境运行，也不改变原预定主指标。 |

其他历史记录：`results/baseline_matrix_force_admission_20260624/` 是后来续跑矩阵的未完成前段，`results/baseline_matrix_force_admission_20260624_smoke/` 和 `results/baseline_matrix_audit_20260624_smoke/` 是流程验收；`results/sensitivity_force_admission/effective_efficiency_convergence/summary.json` 只给旧协议 3 种子终点统计，不能单凭它宣称稳定收敛。`results/experiment_review_20260921/summary.json` 记录正式矩阵中途失败的诊断，后续已用完成版复核；这些记录不另算独立正面证据。先前中断的远程覆盖观测副本未完成，也不计入 G4 pilot 或独立确认。

**当前研究判断**：论文中心应从“上下层注意力组件堆叠”转向“自然覆盖约束下，轨迹策略能否看见尚未接纳的服务需求，以及覆盖扩展与接纳后服务质量如何共同决定端到端 DSR”。上下层分解可保留为系统框架；下层无注意力模型是当前可复现的强基线。09-28 全量对比已经完成“相对现有完整学习方法是否提高 DSR”的主场景检验，无需把同一基线矩阵再当成待完成任务。论证必须同时报告 `DSR = 接纳率 × 已接纳请求截止期满足率`（汇总时使用一致的请求计数口径）、UAV 能耗、公平性、MBS 依赖和计算开销。当前只支持**该仿真主场景、方法各自原配置下的端到端 DSR 优势**，不支持“信息预算相同的架构优势”“所有指标全面最优”或现实部署有效性。下一轮优先做信息可得性与同信息预算对照（仅位置、请求延迟/缺失、通信时延和能耗），再做 UE/算力变化下的配对复验，并在第五章轨迹产生的同一快照上检验第四章 MC-ITR 与在线学习下层的质量和计算代价关系；任何改变观测物理假设的试验都应单独标为新协议。

**写作同步状态**：主工作区的 `latex/docs/abstract.tex`、`chap01.tex`、`chap03.tex`、新建的 `chap04.tex`/`chap05.tex`、`chap06.tex` 与 `latex/main.tex` 已改为截至 09-28 的自然覆盖证据口径；旧强制接纳数值和“未覆盖请求按最近 UAV 关联”已从这些现行正文文件移除，第四、五章已启用。原第三章系统示意图会使所有 UE 看起来都已接入，已从现行正文移除。该 LaTeX 是证据边界草稿，并非最终定稿；固定潜在请求分母复核已写入现行正文，同信息预算、信息代价和跨场景实验仍未完成。`latex/docs/chap02.tex` 的文献陈述、模板封面身份字段以及旧图文件尚须另行核验，不能因正文已改就宣称整篇论文完成审阅。不得重新引入旧强制接纳口径或把第四章快照结果写成第五章在线增益。

## 输出目录

- `results/` — 实验输出（JSON 指标、报告）
- `saved_models/` — 模型检查点（按实验名/seed/episode 组织）
- `train_logs/` — 训练日志
- `docs/figures/` — 论文图表（PDF/PNG），生成脚本在 `docs/figures/scripts/`

## 多模态文件处理

- 仅当任务需要理解图片、音频、视频或 PDF 的内容时触发；提及扩展名、检查路径、重命名或审阅含有示例路径的文本不触发内容分析。
- 先检查当前平台、模型和工具的实际能力，不沿用其他模型的模态限制。适用且已授权的 `understand_image`、`understand_audio`、`understand_video`、`understand_pdf` 可优先使用，但不假定它们存在。
- 指定 MCP 不可用时，可使用平台已提供且适用的原生理解能力，或在本地进行文本提取、OCR、转写、渲染与检查；遵循 PDF、文档等相关 Skill 和平台权限。不得将不支持的二进制内容直接传入文本模型，也不得为降级擅自安装工具。
- 本地处理不等于授权外传。不得把私有论文、敏感数据或审稿材料上传到新的外部服务；外部处理需要明确授权并满足隐私要求。
- 无可用的授权处理路径时，报告具体限制，继续独立工作，并仅请求受影响部分所需的文本、文件或权限。不凭文件名猜测内容。
- 如实说明实际使用的分析方式和识别不确定性；只有实际调用并成功得到 MCP 结果时，才可写“已通过 MCP 分析完成”。

## 注意事项

- 无 requirements.txt，依赖安装在 `.venv/` 中手动管理
- 已有 `tests/` 单元、回归及烟雾验证；本机只运行无 PyTorch 依赖的轻量检查，训练与完整实验验证在远程服务器执行
- 实验脚本使用 `nohup` 后台运行，通过 `--max_workers` 控制并行度
- 环境变量 `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` 用于限制线程数
