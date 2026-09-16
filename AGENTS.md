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
- 无正式测试套件，验证通过烟雾测试和实验级评估完成
- 实验脚本使用 `nohup` 后台运行，通过 `--max_workers` 控制并行度
- 环境变量 `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` 用于限制线程数
