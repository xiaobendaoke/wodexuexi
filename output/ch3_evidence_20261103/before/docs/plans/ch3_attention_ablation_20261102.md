# 第三章注意力机制消融方案（2×2 因子预注册）

2026-11-02。用户授权按 2×2 因子补齐"同等信息条件下的注意力增益"，并加入参数量匹配对照与注意力权重机制图。本文件是**预注册协议**，不是实验结论；训练在远程实验服务器执行，本机仅编辑、核验与论文编译。协议先冻结再开跑，禁止事后挑选方法或指标。

## 背景与待解问题

第三章现行独立确认实验 `upper_uncovered_observation_independent_confirmation_v1`（`results/coverage_observation_confirmation_20260927/`）只比较两种观测模式 `covered` 与 `covered_plus_uncovered`，两臂的上层网络与下层均为**同一架构**（`config_differences` 只含 `UPPER_UE_OBSERVATION_MODE`）。因此该实验只能证明"需求信息扩展"有价值，**不能分离注意力结构的独立贡献**。正文表 `tab:ch3-protocol` 的"对比方法唯一差异维度"当前也只登记观测一轴。

已有旁证（旧队列，未在确认协议下配对复现）：`results/canonical_all_20260923_fixed/marl/evaluation_units.csv` 中，同为 covered 观测、同为无注意力下层、仅上层不同的两方法：

- `lower_no_attention`（注意力上层）DSR **0.2609**
- `no_attention`（vanilla 上层）DSR **0.2257**

5 个训练种子全部为正，均值 **+3.52 个百分点**，配对 t 95% 区间 **[+2.85, +4.20] 个百分点**。该对照属旧队列（种子 42/84/126/168/210，负载 42…420），既未在确认协议下配对复现，也未在扩展观测下检验。

## 目标与判定规则

- **主目标**：在扩展观测 `covered_plus_uncovered` 下，估计"注意力上层 − vanilla 上层"的 DSR 配对差值（第四格 − 第三格）。
- **主判定**：主差值均值大于零，且以 10 个训练种子为单位的双侧配对 t 95% 区间不含零，crossed bootstrap 敏感性区间方向一致，才称"同等信息下的注意力增益得到确认"；否则报告为未确认或不明确。
- **次目标（描述性，不做选择性显著性主张）**：信息主效应、注意力主效应（covered 与 uncovered 两列）、交互项，以及 `DSR = α·η` 分解。

## 因子设计

四格共用同一无注意力下层、同一奖励、同一 325 维输入；仅上层架构与观测模式变化：

|  | 已覆盖观测 `covered` | 扩展观测 `covered_plus_uncovered` |
|---|---|---|
| **注意力上层** | `lower_no_attention`（已有） | `lower_mlp_upper_uncovered_obs`（已有） |
| **vanilla 上层** | `no_attention`（待补） | `vanilla_lower_mlp_upper_uncovered_obs`（待补） |

- 已有两格来自 `results/coverage_observation_confirmation_20260927/`，本次**复用其冻结检查点**，不重训、不覆盖。
- 待补两格用**同一批训练种子与工作负载**新训，构成完整 2×2。
- 方法名已存在于 runner：`no_attention` 在 `METHODS`，`vanilla_lower_mlp_upper_uncovered_obs` 在 `BATCH_METHODS`，两者都无需新增定义，仅需纳入协议套件。

可拆解的量：

- 信息主效应 = `lower_mlp_upper_uncovered_obs` − `lower_no_attention`（注意力行）与 `vanilla_lower_mlp_upper_uncovered_obs` − `no_attention`（vanilla 行）。
- 注意力主效应 = `lower_no_attention` − `no_attention`（covered 列）与 `lower_mlp_upper_uncovered_obs` − `vanilla_lower_mlp_upper_uncovered_obs`（uncovered 列）。
- 交互项 = (uncovered 列注意力增益) − (covered 列注意力增益)。

## 参数量匹配对照

注意力上层（`CrossAttentionExtractor`，8 头、`d_model=64`）与 vanilla MLP 上层参数量不同。为排除"注意力更好只是因为参数更多"：

- 读 `marl_models/vanilla_mappo/` 结构，报告两种上层的可训练参数量。
- 新增宽度对齐的 vanilla 变体，使参数量与注意力上层相当，注册为 `no_attention_capmatched`（covered）与 `vanilla_lower_mlp_upper_uncovered_capmatched`（uncovered）。
- 若参数量匹配后注意力增益消失或大幅缩小，则按"容量而非结构"解释。

## 预注册参数

- 训练种子：`7013, 7283, 7541, 7823, 8089, 8353, 8627, 8887, 9151, 9419`（10 个，与确认协议一致）。
- 工作负载种子：`2203, 2243, 2287, 2333, 2377`（5 个）。
- 训练回合 200，评估回合 6，单回合步数 1000；rollout 1000、PPO epochs 8、batch 256、checkpoint 每 20 回合，取最终预定点。
- 统计单元：训练种子（先在各训练种子内对 5 个工作负载取平均，再跨 10 个种子统计）。
- 训练量：基础 2 方法 ×10 = **20 次训练**；含容量匹配 4 方法 ×10 = **40 次训练**，另加对应评估。

## 统计口径

- 单元内按计数求比例、跨单元等权平均；`DSR_i = α_i η_i`，跨单元直接对 DSR 等权平均。
- 主对比与各因子对比分别给配对 t 95% 区间与 crossed bootstrap 区间；多重对比用 Holm 校正。
- 同时把 `DSR` 分解为接纳项与条件完成项，判断注意力改善的是接纳还是完成质量。

## 混淆控制

- 观测可得性（460 m 内 UE 位置/请求/电量）与注意力轴**正交**；本次沿用确认协议既有假设，不新增或放宽物理假设。若单独审计信息代价，另立协议。
- 同下层、同奖励、同观测维度、同种子/负载；`config_differences` 只允许出现 `UPPER_UE_OBSERVATION_MODE` 与上层模型名。

## 机制图

在评估阶段加轻量导出钩子，dump 注意力权重；绘图脚本对比"高优先级/未覆盖热点槽位"与其余槽位的权重分布，支撑 3.3.3 节现有定性声明。图形输出按 `docs/figures/` 规范，源数据单独归档。

## 结果导向的写作分支

1. **注意力显著为正**：保留 Attention-MAPPO 命名，3.4 新增"同等信息下的架构消融"小节；`tab:ch3-protocol` 的差异维度由单轴改双轴。
2. **不显著或为负**：注意力降级为实现细节，创新点重心移到"需求可见性 → 空间部署与接纳"，改写本章小结归因（现为"结合多头交叉注意力网络"）。
3. **存在交互**：写成"注意力在信息稀缺时无差别、信息丰富时被激活"。

## 验证门槛

- 回归测试：方法注册、观测维度 325、checkpoint 契约不跨方法串用。
- smoke：每方法 2×8 步可起。
- 分析脚本用已有确认数据自检，必须复现 cell4 − cell2 = **+3.49 个百分点**，否则不采信。
- 训练与评估只在远程执行；本机只跑无 PyTorch 依赖的静态检查与论文编译。

## 代码位置与边界

- 本机原主工作树 `我的论文/wodexuexi`（分支 `双层注意力MAPPO多智能体强化学习`）的 `run_canonical_experiment.py` **不含覆盖观测方法**（无 `lower_mlp_upper_uncovered_obs` / `BATCH_METHODS`）；本机也未检出 `coverage-optimization/wodexuexi` 工作树（`git worktree list` 仅有主工作树）。
- 覆盖观测代码的唯一本机副本是封存快照 `output/sf_hlm_learning_feasibility_20261016/assets/provenance_sources/source_launch_20260929/`。实现改动应在**覆盖优化工作树（远程）**进行，或先由该快照恢复本地工作树；**不得把改动写进封存快照**。
- 本文件不覆盖、不改写任何封存证据与历史结果；不提交或推送 Git，除非另行授权。

## 证据与输出位置

- 新协议与套件：`docs/plans/ch3_attention_ablation_20261102.json`（实现时随代码落在实验工作树）。
- 新结果：`results/ch3_attention_ablation_20261102/`。
- 验收日志：`output/ch3_attention_ablation_20261102/validation/`。
- 论文改动：`latex/docs/chap03.tex` 3.4 节与 `tab:ch3-protocol`。
