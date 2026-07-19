# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Superpowers 工作流程

本项目使用 [Superpowers](https://github.com/obra/superpowers) 工作流程。在执行任何任务之前，请遵循以下流程：

### 核心工作流程

1. **brainstorming** - 在写代码之前激活。通过问题细化想法，探索替代方案，呈现设计并获取用户批准。
2. **writing-plans** - 使用批准的设计激活。将工作分解为小任务（每个 2-5 分钟），每个任务有确切的文件路径、完整代码和验证步骤。
3. **test-driven-development** - 在实现期间激活。强制执行 RED-GREEN-REFACTOR：先写失败的测试，看它失败，写最少的代码，看它通过，提交。
4. **requesting-code-review** - 在任务之间激活。根据计划审查，按严重性报告问题。关键问题阻止进度。
5. **finishing-a-development-branch** - 当任务完成时激活。验证测试，呈现选项（合并/PR/保留/丢弃），清理工作区。

### 使用方法

- 在开始任何创造性工作（创建功能、构建组件、添加功能或修改行为）之前，使用 `brainstorming` skill
- 在设计批准后，使用 `writing-plans` skill 创建实现计划
- 在实现期间，使用 `test-driven-development` skill 确保代码质量
- 在任务之间，使用 `requesting-code-review` skill 进行代码审查
- 当任务完成时，使用 `finishing-a-development-branch` skill 完成工作

### 调用方式

使用 `Skill` 工具调用这些 skills：
- `/superpowers:brainstorming` - 头脑风暴和设计
- `/superpowers:writing-plans` - 编写实现计划
- `/superpowers:test-driven-development` - 测试驱动开发
- `/superpowers:requesting-code-review` - 请求代码审查
- `/superpowers:finishing-a-development-branch` - 完成开发分支

## 项目概述

中山大学硕士论文项目：基于 Attention-MAPPO 的多无人机 MEC 轨迹与任务卸载协同优化方法。系统由 5 架 UAV、100 个 UE、1 个 MBS 在 700×700m 区域运行。使用 Python 3.9 + PyTorch 2.6 实现。

## 常用命令

### Python 实验运行

```bash
# 激活虚拟环境
source .venv/bin/activate

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

## 多模态文件自动处理（MCP）

### ⚠️ 关键规则：遇到多模态文件必须调用 MCP

**当检测到用户消息中包含以下任何文件时，必须立即调用对应的 MCP 工具，不要尝试自己处理：**

### 触发文件类型

| 文件类型 | 扩展名 | 必须调用的工具 |
|----------|--------|----------------|
| 图片 | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp` | `understand_image` |
| 音频 | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac` | `understand_audio` |
| 视频 | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.flv` | `understand_video` |
| PDF | `.pdf` | `understand_pdf` |

### 强制执行规则

1. **检测到多模态文件 → 必须调用 MCP 工具**，不要尝试用 Read 工具或其他方式自己处理
2. **绝对不要直接发送图片/音频/视频/PDF 给主模型**（mimo-v2.5-pro 不支持多模态输入，会报错）
3. **MCP 工具调用完成后**，将结果整合到回复中继续对话

### 调用示例

```
用户：分析这个PDF /path/to/paper.pdf
你：[调用 understand_pdf] → 返回结果 → 整合到回复中
```

```
用户：描述这张图片 /path/to/image.png
你：[调用 understand_image] → 返回结果 → 整合到回复中
```

### 回复格式

> 已通过 MCP 分析完成，结果如下：
> [MCP 返回的识别内容]

## 注意事项

- 无 requirements.txt，依赖安装在 `.venv/` 中手动管理
- 无正式测试套件，验证通过烟雾测试和实验级评估完成
- 实验脚本使用 `nohup` 后台运行，通过 `--max_workers` 控制并行度
- 环境变量 `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` 用于限制线程数
