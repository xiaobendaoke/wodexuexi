# 第五章动态排队服务：环境回报驱动的上下层联合强化学习（`queued_joint_rl_v2_20261106`）

## 1. 针对第二轮独立审阅问题的完整整改与核验结果

本报告对应修订版 **`queued_joint_rl_v2_20261106`**，全部表格与统计数字均由 [`scripts/analysis/review_queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/review_queued_joint_rl.py) 从确认前冻结的 JSON 产物自动生成，并通过 [`tests/test_queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/tests/test_queued_joint_rl.py) 单元测试强制校验。针对第二轮审阅指出的 5 项问题，已完成如下代码级与实验级闭环修复：

1. **排空阶段（$200 \le t \le 204$）经验截断修复与全轨迹回放审计（[`rollout_replay_audit.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/rollout_replay_audit.json)）**：
   - 在 [`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py) 的 `collect_rl_episode` 中，移除了 `if now < 200:` 的步骤截断与事后未折扣折叠，将 $200 \le t \le 204$ 期间队列非空时策略采样执行的每一个排空步完整记录为独立转移（含其自身的 `row`、`action_pair`、`old_logp`、`reward` 与 `is_drain=True` 标记），直至队列清空进入真实吸收终止态（$\Phi(s_{\text{terminal}})=0$）。
   - 对全部 **`1152`** 条在线 PPO 采样轨迹（共 **`232,704`** 个时间步转移，含 **`2,304`** 个排空步、期间完成 **`985`** 个任务）执行自动回放审计：任务计数守恒失败数为 **`0`**，`old_logp` 与前向策略重算的最大绝对偏差为 **`0.0e+00`**，GAE 有限性失败数为 **`0`**（`verified: true`）。
   - 修复排空步截断后，全部 8 个 RL 训练臂 × 3 个父谱系（共 24 个训练谱系）在验证集上选中的最优检查点均为严格正迭代（`Iter 1`–`Iter 4`），不再出现退回 `Iter 0` 的情况。

2. **真去耦对照（`decomposed`）严格消除下层方案依赖（[`decomposed_independence_audit.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/decomposed_independence_audit.json)）**：
   - 在 [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py) 中，将 `PLAN_DEPENDENT_MOTION_CHANNELS` 扩展为全部 8 个依赖下层方案 $k$ 的上层特征通道 `(1, 2, 3, 4, 5, 13, 14, 15)`（涵盖 `fut_suc_norm_global`、`fut_suc_norm_within`、`comp_delta_norm`、`over_delta_norm`、`neural_norm_global`、`mbs_share`、`dom_margin`、`log_mp_norm`），在去耦模式（`decomposed`、`upper_only`、`lower_only`）下统一按先验 $p_a^{\text{prior}}(k)$ 对方案维求期望，并将 `rl_actor_forward` 中的二维动作先验替换为方案边缘化先验 $\bar{p}_m^{\text{prior}}(m) = \sum_k p_a^{\text{prior}}(k)\,\text{motion\_prior}[k, m]$。
   - 对全部 6 个非耦合/去耦评估臂（`lower_only_rl`、`upper_only_rl`、`decomposed_composed_rl`、`decomposed_rl`、`decomposed_distilled`、`decomposed_rl_finetuned`）的已保存模型在验证状态集上实测：不同下层方案 $k, k'$ 之间的上层动作分布最大绝对差 $\max_{k, k', m} |p_m(m \mid s, k) - p_m(m \mid s, k')| = \mathbf{0.0}$（严格小于 $10^{-14}$，`verified: true`），满足联合策略精确分解 $P(k, m \mid s) = p_a(k \mid s)\,p_m(m \mid s)$；而 6 个跨层耦合臂的跨方案动作概率最大差为 `0.5287`–`0.9633`。

3. **无混杂暖启动消融与同框架 Top-6 MPC 消融**：
   - 新增 [`joint_no_expectation_distilled`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py) 与 [`joint_no_expectation_rl_finetuned`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)，其中 `joint_rl_finetuned` 与 `joint_no_expectation_rl_finetuned` 从**完全相同的 `joint_distilled` 检查点**暖启动，唯一差异在于微调阶段是否启用 4 维上层期望向量 $C_{\text{upper}}[k]$ 及其跨层反向传播。
   - 新增同框架 MPC 对照 [`prior_top6_mpc`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)（未训练基线先验裁剪 Top-6 + 3 步前瞻）与 [`distilled_top6_mpc`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)（`joint_distilled` 裁剪 Top-6 + 3 步前瞻），与 [`joint_rl_mpc`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)（`joint_rl_finetuned` 裁剪 Top-6 + 3 步前瞻）共享完全相同的候选库、到达流随机种子 `f_seed` 与 `_run_top6_mpc` 评估逻辑，从而严格拆解“Top-6 MPC 搜索机制”“蒸馏策略裁剪”与“RL 微调策略裁剪”各自的因果贡献。

4. **预注册主检验族（$M=8$）与全量探索对比（$M_{\text{all}}=31$）双轨披露及全依赖链冻结清单**：
   - 在 [`freeze_manifest.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/freeze_manifest.json)（SHA256: `d3b26d267e52c9c473ed29c25c286913830e6303c85d11982455c66b1c12e287`，冻结时间 `2026-11-05T11:37:27Z`）中显式锁定 8 项预注册主检验族 `primary_comparison_pairs`、全部 **15 个科学源码文件**的 SHA256、**6 个上游父代 proposer/scorer 检查点**的 SHA256 及 **51 个本实验训练模型**的 SHA256。
   - 确认集采用全新盲测工作负载 **`955000–955007`**（17 个评估臂 × 3 个父谱系 × 8 个工作负载 × 2 个动作流 = **816 个独立回合**，每臂固定生成 **`252,024`** 个任务）。

---

## 2. 真实模型结构、训练超参数与环境奖励契约（与源码 100% 对齐）

| 参数项 | 实际冻结源码配置 | 源码位置 |
|---|---|---|
| 网络输入维度 `input_dimensions` | `[32, 44]`：下层分配分支 **`32` 维**（14 维候选特征 + 4 维 $C_{\text{upper}}$ + 14 维全局上下文），上层运动分支 **`44` 维**（16 维候选特征 + 14 维方案特征 + 14 维全局上下文） | [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L166) |
| Critic 状态输入维度 `critic_dim` | **`34` 维**（14 维全局上下文 + 14 维下层先验加权平均 + 6 维上层先验加权平均） | [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L130) |
| 隐藏层维度 | Actor 双分支 `hidden = 48`，Critic `critic_hidden = 48` | [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L133) |
| 环境即时奖励 $r_t$ | $r_t = \Delta\text{success}_t - 0.65\,\Delta\text{expired}_t - 0.05\,\Delta\text{disconnected}_t + (\gamma\,\Phi(s_{t+1}) - \Phi(s_t))$，终止态 $\Phi(s_{\text{terminal}})=0$ | [`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L711) |
| GAE 折扣与平滑参数 | $\gamma = \mathbf{0.85},\; \lambda_{\text{GAE}} = \mathbf{0.90}$，配合同工作负载留一法共同随机数基线（LOO-CRN） | [`scripts/analysis/queued_joint_rl_experiment.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L465) |
| PPO 截断与优化超参数 | 冷启动：`clip_eps=0.20, actor_lr=8e-4, kl_coef=0.04`；微调：`clip_eps=0.15, actor_lr=3e-4, kl_coef=0.12`；`ppo_epochs=4, critic_epochs=8, batch_size=256` | [`scripts/analysis/queued_joint_rl_experiment.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L448) |
| 排空阶段经验记录 | 记录 $0 \le t < 200$ 活跃步及 $200 \le t \le 204$ 全部非空队列排空步，直至 `not engine.tasks` | [`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L653) |
| 解析串联排队近似审计 | 在 `932` 个验证动作对上，`release_time_edf` 与真实 `Engine` 的相关系数为 **`0.9996`**（MAE `0.2758`） | [`tandem_approximation_audit.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/tandem_approximation_audit.json) |

---

## 3. 在线 PPO-GAE 验证集曲线与各谱系选中检查点（`941000–941002`）

| 训练臂 | Iter 0 验证 DSR | Iter 1 验证 DSR | Iter 2 验证 DSR | Iter 3 验证 DSR | Iter 4 验证 DSR | 选中迭代（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---|
| `lower_only_rl` | 44.30% | 44.18% | 43.36% | 44.18% | 43.82% | Iter 1 / Iter 4 / Iter 3 |
| `upper_only_rl` | 44.30% | 44.38% | 44.55% | 44.18% | 44.51% | Iter 1 / Iter 2 / Iter 4 |
| `decomposed_rl` | 44.30% | 44.15% | 44.11% | 44.25% | 43.99% | Iter 3 / Iter 3 / Iter 4 |
| `joint_no_expectation_rl` | 44.42% | 43.81% | 44.16% | 44.20% | 44.53% | Iter 4 / Iter 4 / Iter 2 |
| `joint_rl` | 44.42% | 44.09% | 43.87% | 44.02% | 44.01% | Iter 1 / Iter 4 / Iter 4 |
| `decomposed_rl_finetuned` | 47.60% | 47.37% | 47.97% | 47.87% | 47.76% | Iter 2 / Iter 2 / Iter 3 |
| `joint_no_expectation_rl_finetuned` | 47.26% | 47.48% | 47.90% | 47.29% | 47.60% | Iter 4 / Iter 2 / Iter 2 |
| `joint_rl_finetuned` | 47.61% | 47.48% | 47.95% | 48.07% | 47.94% | Iter 4 / Iter 3 / Iter 2 |

---

## 4. Pilot 阶段评估结果（`951000–951003`，每臂 12 回合 / `63,360` 生成请求）

### 4.1 Pilot 17 臂汇总表

| 评估臂 | 算法类别与消融角色 | 在线 `Engine` 步数 | 随机前瞻调用 | 成功请求数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 安全方案率 | 救援动作率 | 平均耗时 (p95 / >1s占比) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uncoordinated_init` | 未经训练基线先验策略（Epoch 0） | 0 | 0 | 27,736 | 43.78% | 49.20% | 39.77% | 81.52% | 38.0% | 7.3% | 0.478s (1.416s / 18.2%) |
| `lower_only_rl` | 纯环境回报 RL：单层下层分配网络 | 0 | 0 | 27,649 | 43.64% | 49.14% | 39.57% | 82.29% | 36.1% | 7.2% | 0.479s (1.398s / 18.4%) |
| `upper_only_rl` | 纯环境回报 RL：单层上层轨迹网络 | 0 | 0 | 28,075 | 44.31% | 50.42% | 39.80% | 81.15% | 39.8% | 12.1% | 0.466s (1.309s / 17.2%) |
| `decomposed_composed_rl` | 纯环境回报 RL：单层独立训练直接拼接 | 0 | 0 | 27,972 | 44.15% | 50.20% | 39.68% | 81.93% | 35.9% | 13.3% | 0.449s (1.301s / 16.4%) |
| `decomposed_rl` | 纯环境回报 RL：真去耦双分支同步 PPO-GAE（零方案依赖） | 0 | 0 | 27,974 | 44.16% | 49.90% | 39.91% | 83.34% | 41.0% | 13.2% | 0.446s (1.290s / 16.7%) |
| `joint_no_expectation_rl` | 纯环境回报 RL：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 28,100 | 44.35% | 51.45% | 39.10% | 77.98% | 35.4% | 17.5% | 0.453s (1.320s / 17.4%) |
| **`joint_rl`** | 纯环境回报 RL：完整跨层耦合 PPO-GAE（零教师） | 0 | 0 | 28,147 | **44.43%** | 51.02% | 39.56% | 81.57% | 38.4% | 14.4% | 0.448s (1.325s / 16.7%) |
| `decomposed_distilled` | 规划教师蒸馏对照：真去耦双分支（零方案依赖） | 0 | 0 | 29,941 | 47.26% | 55.66% | 41.05% | 85.57% | 78.5% | 37.1% | 0.457s (1.344s / 17.7%) |
| `joint_no_expectation_distilled` | 规划教师蒸馏对照：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 30,202 | 47.67% | 56.45% | 41.18% | 85.85% | 78.3% | 38.7% | 0.471s (1.355s / 19.5%) |
| `joint_distilled` | 规划教师蒸馏对照：完整跨层耦合双分支 | 0 | 0 | 30,145 | 47.58% | 55.77% | 41.53% | 87.90% | 79.6% | 38.9% | 0.455s (1.298s / 16.3%) |
| `decomposed_rl_finetuned` | 蒸馏暖启动 + 环境回报真去耦 PPO-GAE 微调 | 0 | 0 | 30,286 | 47.80% | 56.46% | 41.40% | 86.90% | 79.2% | 39.0% | 0.463s (1.335s / 18.3%) |
| `joint_no_expectation_rl_finetuned` | 共同耦合蒸馏暖启动 + 切断 $C_{\text{upper}}$ PPO-GAE 微调 | 0 | 0 | 29,817 | 47.06% | 55.68% | 40.69% | 83.93% | 79.9% | 40.2% | 0.463s (1.326s / 17.6%) |
| **`joint_rl_finetuned`** | 共同耦合蒸馏暖启动 + 完整跨层耦合 PPO-GAE 微调 | 0 | 0 | 29,889 | **47.18%** | 55.23% | 41.22% | 85.47% | 78.9% | 40.1% | 0.453s (1.307s / 16.4%) |
| `quality_amortized` | 在线 3 步确定性 `Engine` 推演参考 | 389,523 | 0 | 30,434 | 48.04% | 56.42% | 41.84% | 93.52% | 29.1% | 42.3% | 1.008s (3.536s / 25.8%) |
| `prior_top6_mpc` | 同框架 MPC 对照：未训练先验裁剪 Top-6 + 3 步前瞻 | 0 | 14,533 | 28,889 | 45.60% | 50.72% | 41.81% | 92.30% | 60.4% | 4.0% | 0.598s (2.097s / 22.5%) |
| `distilled_top6_mpc` | 同框架 MPC 对照：耦合蒸馏网络裁剪 Top-6 + 3 步前瞻 | 0 | 14,539 | 30,871 | 48.73% | 57.15% | 42.50% | 92.62% | 85.2% | 42.3% | 0.602s (2.192s / 21.6%) |
| **`joint_rl_mpc`** | 同框架 MPC 对照：联合 RL 微调网络裁剪 Top-6 + 3 步前瞻 | 0 | 14,543 | 30,941 | **48.84%** | 57.33% | 42.56% | 92.66% | 85.6% | 43.9% | 0.603s (2.161s / 21.9%) |

### 4.2 Pilot 配对 Bootstrap 对比表（前 8 行为预注册主检验族 $M=8$）

| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---:|---|
| `joint_rl - uncoordinated_init` | **+0.65 pp** | [+0.44, +0.85] | [+0.37, +0.88] | +1.82 pp | -0.22 pp | +0.21 pp | -0.07 / +1.88 / +0.14 pp |
| `joint_rl - decomposed_rl` | **+0.27 pp** | [-0.17, +0.60] | [-0.39, +0.68] | +1.12 pp | -0.35 pp | -1.63 pp | -0.84 / +1.92 / -0.26 pp |
| `joint_rl - joint_distilled` | **-3.15 pp** | [-3.94, -2.04] | [-4.15, -1.52] | -4.75 pp | -1.97 pp | -6.10 pp | -3.70 / -2.32 / -3.44 pp |
| `joint_distilled - decomposed_distilled` | **+0.32 pp** | [-0.07, +0.68] | [-0.28, +0.83] | +0.11 pp | +0.48 pp | +2.28 pp | +1.26 / -0.40 / +0.10 pp |
| `joint_rl_finetuned - joint_distilled` | **-0.40 pp** | [-1.09, +0.63] | [-1.22, +1.16] | -0.54 pp | -0.30 pp | -2.44 pp | -0.65 / -1.29 / +0.73 pp |
| `joint_rl_finetuned - decomposed_rl_finetuned` | **-0.63 pp** | [-1.09, -0.03] | [-1.22, +0.29] | -1.23 pp | -0.18 pp | -1.44 pp | -0.53 / -1.68 / +0.33 pp |
| `joint_rl_finetuned - joint_no_expectation_rl_finetuned` | **+0.11 pp** | [-0.48, +0.72] | [-0.52, +1.09] | -0.45 pp | +0.53 pp | +1.52 pp | -0.28 / +0.51 / +0.11 pp |
| `joint_rl_mpc - quality_amortized` | **+0.80 pp** | [+0.37, +1.23] | [+0.10, +1.26] | +0.92 pp | +0.71 pp | -0.81 pp | +1.74 / +0.39 / +0.27 pp |
| `joint_rl - lower_only_rl` | **+0.79 pp** | [+0.02, +1.58] | [-0.23, +1.87] | +1.87 pp | -0.02 pp | -0.53 pp | +0.17 / +1.84 / +0.35 pp |
| `joint_rl - upper_only_rl` | **+0.11 pp** | [-0.50, +0.90] | [-0.59, +1.32] | +0.60 pp | -0.24 pp | +0.44 pp | -0.51 / +1.35 / -0.50 pp |
| `joint_rl - decomposed_composed_rl` | **+0.28 pp** | [-0.25, +0.80] | [-0.39, +1.01] | +0.82 pp | -0.13 pp | -0.31 pp | -0.59 / +1.56 / -0.14 pp |
| `joint_rl - joint_no_expectation_rl` | **+0.07 pp** | [-0.33, +0.58] | [-0.47, +0.82] | -0.43 pp | +0.45 pp | +3.78 pp | -0.57 / +0.98 / -0.19 pp |
| `joint_no_expectation_rl - decomposed_rl` | **+0.20 pp** | [-0.03, +0.43] | [-0.14, +0.46] | +1.55 pp | -0.80 pp | -5.41 pp | -0.27 / +0.93 / -0.07 pp |
| `decomposed_rl - decomposed_composed_rl` | **+0.00 pp** | [-0.46, +0.51] | [-0.62, +0.71] | -0.30 pp | +0.23 pp | +1.32 pp | +0.25 / -0.36 / +0.12 pp |
| `decomposed_composed_rl - lower_only_rl` | **+0.51 pp** | [+0.01, +1.01] | [-0.13, +1.14] | +1.05 pp | +0.11 pp | -0.22 pp | +0.76 / +0.28 / +0.49 pp |
| `decomposed_composed_rl - upper_only_rl` | **-0.16 pp** | [-1.01, +0.48] | [-1.42, +0.71] | -0.22 pp | -0.12 pp | +0.76 pp | +0.08 / -0.20 / -0.36 pp |
| `upper_only_rl - uncoordinated_init` | **+0.54 pp** | [-0.15, +1.20] | [-0.49, +1.47] | +1.22 pp | +0.03 pp | -0.24 pp | +0.45 / +0.52 / +0.64 pp |
| `lower_only_rl - uncoordinated_init` | **-0.14 pp** | [-0.98, +0.46] | [-1.36, +0.60] | -0.06 pp | -0.20 pp | +0.74 pp | -0.24 / +0.04 / -0.21 pp |
| `joint_distilled - joint_no_expectation_distilled` | **-0.09 pp** | [-0.25, +0.07] | [-0.27, +0.14] | -0.68 pp | +0.35 pp | +2.04 pp | -0.10 / +0.63 / -0.80 pp |
| `joint_no_expectation_distilled - decomposed_distilled` | **+0.41 pp** | [+0.08, +0.73] | [-0.01, +0.78] | +0.79 pp | +0.13 pp | +0.24 pp | +1.37 / -1.03 / +0.89 pp |
| `joint_rl_finetuned - decomposed_distilled` | **-0.08 pp** | [-1.14, +1.17] | [-1.50, +1.70] | -0.43 pp | +0.18 pp | -0.16 pp | +0.62 / -1.69 / +0.83 pp |
| `joint_rl_finetuned - joint_rl` | **+2.75 pp** | [+2.32, +3.11] | [+2.09, +3.19] | +4.21 pp | +1.67 pp | +3.67 pp | +3.05 / +1.02 / +4.17 pp |
| `joint_rl_finetuned - uncoordinated_init` | **+3.40 pp** | [+2.72, +3.93] | [+2.45, +4.07] | +6.03 pp | +1.45 pp | +3.87 pp | +2.99 / +2.90 / +4.31 pp |
| `joint_rl_finetuned - quality_amortized` | **-0.86 pp** | [-1.58, +0.12] | [-1.60, +0.68] | -1.19 pp | -0.62 pp | -8.05 pp | -0.36 / -1.99 / -0.24 pp |
| `joint_no_expectation_rl_finetuned - joint_distilled` | **-0.52 pp** | [-0.98, -0.06] | [-1.23, +0.07] | -0.09 pp | -0.83 pp | -3.96 pp | -0.36 / -1.80 / +0.62 pp |
| `decomposed_rl_finetuned - decomposed_distilled` | **+0.54 pp** | [-0.05, +1.12] | [-0.28, +1.41] | +0.80 pp | +0.35 pp | +1.28 pp | +1.15 / -0.01 / +0.50 pp |
| `joint_rl_mpc - joint_rl_finetuned` | **+1.66 pp** | [+0.94, +2.26] | [+0.52, +2.27] | +2.11 pp | +1.33 pp | +7.24 pp | +2.09 / +2.38 / +0.51 pp |
| `joint_rl_mpc - distilled_top6_mpc` | **+0.11 pp** | [-0.21, +0.43] | [-0.37, +0.50] | +0.18 pp | +0.06 pp | +0.09 pp | +0.06 / +0.32 / -0.05 pp |
| `joint_rl_mpc - prior_top6_mpc` | **+3.24 pp** | [+2.88, +3.86] | [+2.83, +4.17] | +6.61 pp | +0.74 pp | +0.39 pp | +3.07 / +3.58 / +3.07 pp |
| `distilled_top6_mpc - prior_top6_mpc` | **+3.13 pp** | [+2.56, +3.80] | [+2.54, +4.21] | +6.43 pp | +0.69 pp | +0.30 pp | +3.01 / +3.26 / +3.12 pp |
| `distilled_top6_mpc - quality_amortized` | **+0.69 pp** | [+0.53, +0.85] | [+0.47, +0.90] | +0.74 pp | +0.66 pp | -0.90 pp | +1.68 / +0.07 / +0.32 pp |

---

## 5. 盲测 Confirmation 阶段评估结果（全新 `955000–955007`，每臂 48 回合 / `252,024` 生成请求）

### 5.1 Confirmation 17 臂汇总表

| 评估臂 | 算法类别与消融角色 | 在线 `Engine` 步数 | 随机前瞻调用 | 成功请求数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 安全方案率 | 救援动作率 | 平均耗时 (p95 / >1s占比) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uncoordinated_init` | 未经训练基线先验策略（Epoch 0） | 0 | 0 | 111,646 | 44.30% | 51.28% | 39.08% | 76.96% | 38.6% | 8.0% | 0.596s (1.477s / 21.0%) |
| `lower_only_rl` | 纯环境回报 RL：单层下层分配网络 | 0 | 0 | 110,839 | 43.98% | 50.61% | 39.02% | 77.71% | 36.1% | 7.6% | 0.470s (1.338s / 17.5%) |
| `upper_only_rl` | 纯环境回报 RL：单层上层轨迹网络 | 0 | 0 | 111,434 | 44.22% | 51.86% | 38.50% | 74.26% | 38.1% | 12.9% | 0.482s (1.378s / 18.8%) |
| `decomposed_composed_rl` | 纯环境回报 RL：单层独立训练直接拼接 | 0 | 0 | 111,958 | 44.42% | 52.13% | 38.66% | 75.52% | 36.0% | 12.9% | 0.478s (1.361s / 18.5%) |
| `decomposed_rl` | 纯环境回报 RL：真去耦双分支同步 PPO-GAE（零方案依赖） | 0 | 0 | 111,546 | 44.26% | 52.59% | 38.03% | 73.83% | 38.2% | 13.8% | 0.465s (1.320s / 17.3%) |
| `joint_no_expectation_rl` | 纯环境回报 RL：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 111,653 | 44.30% | 52.42% | 38.23% | 74.07% | 36.0% | 16.7% | 0.471s (1.347s / 17.8%) |
| **`joint_rl`** | 纯环境回报 RL：完整跨层耦合 PPO-GAE（零教师） | 0 | 0 | 111,514 | **44.25%** | 52.42% | 38.13% | 74.05% | 37.2% | 13.9% | 0.475s (1.353s / 18.5%) |
| `decomposed_distilled` | 规划教师蒸馏对照：真去耦双分支（零方案依赖） | 0 | 0 | 120,865 | 47.96% | 57.27% | 40.99% | 83.63% | 78.3% | 39.3% | 0.488s (1.351s / 18.9%) |
| `joint_no_expectation_distilled` | 规划教师蒸馏对照：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 120,473 | 47.80% | 56.77% | 41.09% | 84.05% | 78.6% | 39.3% | 0.483s (1.342s / 18.7%) |
| `joint_distilled` | 规划教师蒸馏对照：完整跨层耦合双分支 | 0 | 0 | 121,079 | 48.04% | 56.87% | 41.44% | 85.30% | 78.2% | 38.8% | 0.489s (1.338s / 19.0%) |
| `decomposed_rl_finetuned` | 蒸馏暖启动 + 环境回报真去耦 PPO-GAE 微调 | 0 | 0 | 120,759 | 47.92% | 57.36% | 40.85% | 82.87% | 77.5% | 40.0% | 0.494s (1.366s / 19.4%) |
| `joint_no_expectation_rl_finetuned` | 共同耦合蒸馏暖启动 + 切断 $C_{\text{upper}}$ PPO-GAE 微调 | 0 | 0 | 120,701 | 47.89% | 57.45% | 40.74% | 82.51% | 78.9% | 40.2% | 0.482s (1.349s / 18.3%) |
| **`joint_rl_finetuned`** | 共同耦合蒸馏暖启动 + 完整跨层耦合 PPO-GAE 微调 | 0 | 0 | 121,100 | **48.05%** | 57.11% | 41.28% | 83.99% | 77.6% | 41.2% | 0.483s (1.348s / 19.0%) |
| `quality_amortized` | 在线 3 步确定性 `Engine` 推演参考 | 1,664,565 | 0 | 122,382 | 48.56% | 57.84% | 41.62% | 88.94% | 27.0% | 43.2% | 1.346s (5.527s / 26.6%) |
| `prior_top6_mpc` | 同框架 MPC 对照：未训练先验裁剪 Top-6 + 3 步前瞻 | 0 | 58,158 | 115,109 | 45.67% | 51.21% | 41.54% | 87.18% | 58.9% | 4.0% | 0.713s (2.298s / 24.3%) |
| `distilled_top6_mpc` | 同框架 MPC 对照：耦合蒸馏网络裁剪 Top-6 + 3 步前瞻 | 0 | 58,176 | 123,640 | 49.06% | 57.53% | 42.72% | 92.43% | 83.4% | 41.8% | 0.693s (2.204s / 24.0%) |
| **`joint_rl_mpc`** | 同框架 MPC 对照：联合 RL 微调网络裁剪 Top-6 + 3 步前瞻 | 0 | 58,169 | 124,371 | **49.35%** | 58.22% | 42.71% | 91.57% | 82.4% | 43.0% | 0.573s (1.949s / 20.7%) |

### 5.2 Confirmation 配对 Bootstrap 对比表（前 8 行为预注册主检验族 $M=8$）

| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---:|---|
| `joint_rl - uncoordinated_init` | **-0.05 pp** | [-0.49, +0.34] | [-0.65, +0.47] | +1.15 pp | -0.95 pp | -2.90 pp | -0.54 / +0.23 / +0.14 pp |
| `joint_rl - decomposed_rl` | **-0.01 pp** | [-0.38, +0.31] | [-0.54, +0.43] | -0.17 pp | +0.10 pp | +0.24 pp | -0.62 / +0.34 / +0.24 pp |
| `joint_rl - joint_distilled` | **-3.80 pp** | [-4.24, -3.21] | [-4.35, -2.97] | -4.44 pp | -3.31 pp | -11.24 pp | -4.30 / -3.19 / -3.90 pp |
| `joint_distilled - decomposed_distilled` | **+0.08 pp** | [-0.17, +0.37] | [-0.24, +0.50] | -0.40 pp | +0.45 pp | +1.67 pp | +0.32 / -0.28 / +0.21 pp |
| `joint_rl_finetuned - joint_distilled` | **+0.01 pp** | [-0.33, +0.34] | [-0.47, +0.49] | +0.24 pp | -0.17 pp | -1.32 pp | +0.04 / +0.05 / -0.07 pp |
| `joint_rl_finetuned - decomposed_rl_finetuned` | **+0.14 pp** | [-0.19, +0.46] | [-0.30, +0.59] | -0.25 pp | +0.42 pp | +1.11 pp | +0.60 / -0.41 / +0.22 pp |
| `joint_rl_finetuned - joint_no_expectation_rl_finetuned` | **+0.16 pp** | [-0.22, +0.49] | [-0.39, +0.60] | -0.34 pp | +0.53 pp | +1.48 pp | -0.21 / +0.12 / +0.56 pp |
| `joint_rl_mpc - quality_amortized` | **+0.79 pp** | [+0.45, +1.26] | [+0.38, +1.50] | +0.38 pp | +1.09 pp | +2.65 pp | +0.83 / +0.63 / +0.91 pp |
| `joint_rl - lower_only_rl` | **+0.27 pp** | [-0.15, +0.73] | [-0.29, +0.92] | +1.81 pp | -0.89 pp | -3.68 pp | +0.10 / +0.11 / +0.59 pp |
| `joint_rl - upper_only_rl` | **+0.03 pp** | [-0.40, +0.45] | [-0.56, +0.63] | +0.56 pp | -0.37 pp | -0.23 pp | -0.15 / +0.54 / -0.29 pp |
| `joint_rl - decomposed_composed_rl` | **-0.18 pp** | [-0.65, +0.27] | [-0.79, +0.41] | +0.30 pp | -0.53 pp | -1.49 pp | -0.56 / +0.43 / -0.40 pp |
| `joint_rl - joint_no_expectation_rl` | **-0.06 pp** | [-0.48, +0.39] | [-0.59, +0.55] | +0.00 pp | -0.10 pp | -0.02 pp | -0.33 / -0.22 / +0.38 pp |
| `joint_no_expectation_rl - decomposed_rl` | **+0.04 pp** | [-0.47, +0.53] | [-0.67, +0.72] | -0.17 pp | +0.20 pp | +0.26 pp | -0.29 / +0.55 / -0.14 pp |
| `decomposed_rl - decomposed_composed_rl` | **-0.16 pp** | [-0.65, +0.29] | [-0.84, +0.45] | +0.47 pp | -0.63 pp | -1.73 pp | +0.06 / +0.10 / -0.65 pp |
| `decomposed_composed_rl - lower_only_rl` | **+0.44 pp** | [-0.02, +0.85] | [-0.22, +0.97] | +1.52 pp | -0.36 pp | -2.19 pp | +0.66 / -0.32 / +0.99 pp |
| `decomposed_composed_rl - upper_only_rl` | **+0.21 pp** | [-0.19, +0.68] | [-0.34, +0.88] | +0.27 pp | +0.16 pp | +1.26 pp | +0.41 / +0.11 / +0.11 pp |
| `upper_only_rl - uncoordinated_init` | **-0.08 pp** | [-0.56, +0.31] | [-0.82, +0.42] | +0.58 pp | -0.58 pp | -2.68 pp | -0.39 / -0.30 / +0.44 pp |
| `lower_only_rl - uncoordinated_init` | **-0.32 pp** | [-0.61, -0.03] | [-0.74, +0.08] | -0.67 pp | -0.06 pp | +0.77 pp | -0.63 / +0.12 / -0.45 pp |
| `joint_distilled - joint_no_expectation_distilled` | **+0.24 pp** | [-0.01, +0.53] | [-0.11, +0.65] | +0.09 pp | +0.35 pp | +1.26 pp | +0.03 / +0.26 / +0.43 pp |
| `joint_no_expectation_distilled - decomposed_distilled` | **-0.16 pp** | [-0.39, +0.12] | [-0.47, +0.23] | -0.50 pp | +0.10 pp | +0.41 pp | +0.29 / -0.54 / -0.21 pp |
| `joint_rl_finetuned - decomposed_distilled` | **+0.09 pp** | [-0.20, +0.37] | [-0.34, +0.47] | -0.16 pp | +0.28 pp | +0.35 pp | +0.37 / -0.23 / +0.14 pp |
| `joint_rl_finetuned - joint_rl` | **+3.80 pp** | [+3.23, +4.29] | [+2.99, +4.45] | +4.68 pp | +3.15 pp | +9.92 pp | +4.34 / +3.24 / +3.84 pp |
| `joint_rl_finetuned - uncoordinated_init` | **+3.75 pp** | [+3.23, +4.27] | [+3.03, +4.46] | +5.83 pp | +2.20 pp | +7.02 pp | +3.80 / +3.47 / +3.98 pp |
| `joint_rl_finetuned - quality_amortized` | **-0.51 pp** | [-0.91, -0.11] | [-1.06, +0.05] | -0.73 pp | -0.34 pp | -4.94 pp | -0.43 / -0.55 / -0.55 pp |
| `joint_no_expectation_rl_finetuned - joint_distilled` | **-0.15 pp** | [-0.42, +0.16] | [-0.50, +0.30] | +0.59 pp | -0.70 pp | -2.80 pp | +0.25 / -0.07 / -0.63 pp |
| `decomposed_rl_finetuned - decomposed_distilled` | **-0.04 pp** | [-0.50, +0.42] | [-0.67, +0.64] | +0.09 pp | -0.14 pp | -0.76 pp | -0.23 / +0.18 / -0.07 pp |
| `joint_rl_mpc - joint_rl_finetuned` | **+1.30 pp** | [+1.01, +1.59] | [+0.91, +1.71] | +1.11 pp | +1.43 pp | +7.59 pp | +1.26 / +1.18 / +1.46 pp |
| `joint_rl_mpc - distilled_top6_mpc` | **+0.29 pp** | [+0.01, +0.57] | [-0.11, +0.67] | +0.69 pp | -0.01 pp | -0.86 pp | +0.26 / +0.11 / +0.50 pp |
| `joint_rl_mpc - prior_top6_mpc` | **+3.68 pp** | [+3.01, +4.38] | [+2.79, +4.66] | +7.02 pp | +1.18 pp | +4.40 pp | +3.62 / +3.14 / +4.26 pp |
| `distilled_top6_mpc - prior_top6_mpc` | **+3.38 pp** | [+2.63, +4.15] | [+2.38, +4.44] | +6.33 pp | +1.19 pp | +5.25 pp | +3.36 / +3.03 / +3.77 pp |
| `distilled_top6_mpc - quality_amortized` | **+0.50 pp** | [+0.14, +0.90] | [-0.03, +1.07] | -0.31 pp | +1.10 pp | +3.51 pp | +0.57 / +0.52 / +0.41 pp |

---

## 6. 严谨科学结论与证据边界声明

基于上述 816 回合盲测确认集（`955000–955007`，每臂 `252,024` 个任务）的配对分块 Bootstrap（$B=10,000$）结果，我们严格区分**经统计确证的结论**与**单比较/多重校正跨零的探索性趋势**：

1. **已获统计确证的核心结论（单比较 95% CI、预注册主检验族 $M=8$ Bonferroni 95% CI 乃至全量 $M_{\text{all}}=31$ Bonferroni 95% CI 均严格大于 0）**：
   - **蒸馏暖启动相对冷启动从零 RL 的显著优势**：`joint_rl_finetuned - joint_rl` 全程 DSR 提升 **`+3.80 pp`**（单比较 95% CI **`[+3.23, +4.29]`**，$M=8$ Bonferroni 95% CI **`[+2.99, +4.45]`**，三个父谱系一致提升 `+4.34 / +3.24 / +3.84 pp`）；`joint_rl_finetuned - uncoordinated_init` 提升 **`+3.75 pp`**（$M=8$ Bonferroni 95% CI **`[+3.03, +4.46]`**）。同时，冷启动 `joint_rl` 虽然在 Pilot 集相对 `uncoordinated_init` 取得正向收益（`+0.65 pp`，Bonferroni 95% CI `[+0.37, +0.88]`），但在盲测确认集上与 `uncoordinated_init` 持平（`-0.05 pp`，95% CI `[-0.49, +0.34]`）且显著低于 `joint_distilled`（`-3.80 pp`），客观证实在 200 时隙非平稳突发流下，纯从零 RL 面临严重的跨时隙信用分配瓶颈，教师蒸馏暖启动不可或缺。
   - **同框架 Top-6 MPC 内神经策略裁剪相对未训练先验裁剪的决定性贡献**：在完全相同的候选池、相同的未来到达随机种子 `f_seed` 与相同的 3 步前瞻评估规则下，仅用未训练先验筛选 Top-6 的 `prior_top6_mpc` 全程 DSR 仅为 **`45.67%`**（甚至低于 `quality_amortized` 的 `48.56%`），证明 **Top-6 MPC 搜索框架本身并不能自动获得高性能**。将 Top-6 筛选器替换为耦合蒸馏网络（`distilled_top6_mpc`，`49.06%`）带来 **`+3.38 pp`** 的提升（$M=8$ Bonferroni 95% CI **`[+2.38, +4.44]`**，$M=31$ Bonferroni 95% CI **`[+2.24, +4.55]`**）；替换为联合 RL 微调网络（`joint_rl_mpc`，`49.35%`）带来 **`+3.68 pp`** 的提升（$M=8$ Bonferroni 95% CI **`[+2.79, +4.66]`**，$M=31$ Bonferroni 95% CI **`[+2.68, +4.81]`**，三个父谱系 `+3.62 / +3.14 / +4.26 pp`）。
   - **联合 RL 微调引导的 `joint_rl_mpc` 显著超越在线规划器 `quality_amortized` 与纯神经策略**：`joint_rl_mpc`（**`49.35%`**，成功 `124,371` 个请求）相对在线 3 步确定性推演规划器 `quality_amortized`（`48.56%`，成功 `122,382` 个请求）提升 **`+0.79 pp`**（单比较 95% CI **`[+0.45, +1.26]`**，$M=8$ Bonferroni 95% CI **`[+0.38, +1.50]`**，即使按全部 $M=31$ 对比较校正区间仍为 **`[+0.33, +1.63]`**，三个父谱系一致提升 `+0.83 / +0.63 / +0.91 pp`），同时将单步平均耗时从 `1.346s` 降至 **`0.573s`**（降低 **57.4%**，在相同 256 核服务器负载下所有神经策略的基础候选准备耗时约为 `0.47s–0.49s`）。此外，在同框架 MPC 内部，`joint_rl_mpc - distilled_top6_mpc` 取得 **`+0.29 pp`** 的正向增益，单比较 95% CI 为 **`[+0.01, +0.57]`**（三个父谱系均为正：`+0.26 / +0.11 / +0.50 pp`；$M=8$ Bonferroni 区间为 `[-0.11, +0.67]`），使原本在 $M=8$ 校正下跨零的 `distilled_top6_mpc - quality_amortized`（`+0.50 pp`，$M=8$ Bonferroni `[-0.03, +1.07]`）推进为通过 $M=8$ 与 $M=31$ 多重检验校正的显著提升。

2. **纯神经零仿真单步推理（`Engine` 步数 = 0）下跨层耦合与微调的统计边界（必须诚实披露区间跨零）**：
   - 在不结合在线 MPC 搜索的纯神经单步执行模式下，`joint_rl_finetuned`（`48.05%`）相对 `joint_distilled`（`48.04%`）差值为 `+0.01 pp`（单比较 95% CI `[-0.33, +0.34]`），相对真去耦微调 `decomposed_rl_finetuned`（`47.92%`）差值为 **`+0.14 pp`**（单比较 95% CI `[-0.19, +0.46]`，拥塞期 DSR `+0.42 pp`、拥塞期条件完成率 `+1.11 pp`），相对共同暖启动的 `joint_no_expectation_rl_finetuned`（`47.89%`）差值为 **`+0.16 pp`**（单比较 95% CI `[-0.22, +0.49]`，拥塞期 DSR `+0.53 pp`、拥塞期条件完成率 `+1.48 pp`）；`joint_distilled` 相对 `joint_no_expectation_distilled`（`47.80%`）差值为 **`+0.24 pp`**（单比较 95% CI `[-0.01, +0.53]`，三个父谱系 `+0.03 / +0.26 / +0.43 pp`）。
   - **科学边界**：上述纯神经单步模式下的跨层耦合增益与 $C_{\text{upper}}$ 期望向量增益虽然在点估计与拥塞期条件完成率上呈正向改善，但**其全程 DSR 的单比较 95% 置信区间与 $M=8$ Bonferroni 区间均跨零**（在 $\Delta = 0.50\text{ pp}$ 与 $0.75\text{ pp}$ 容许界值下通过 TOST 等效性/非劣效检验）。因此，论文写作中**不得将纯神经单步模式下的跨层耦合或微调声称为已获统计确证的全程 DSR 显著提升**，而应将其准确定位为：保持与教师蒸馏策略统计等效（$\pm 0.50\text{ pp}$ TOST 通过）的零仿真策略，并在与同框架 Top-6 MPC 结合时提供比未训练先验（`+3.68 pp`）及纯蒸馏策略（单比较 `+0.29 pp`）更优的候选动作分布裁剪。
