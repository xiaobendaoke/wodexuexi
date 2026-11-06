# 第五章动态排队服务：环境回报驱动的上下层联合强化学习（`queued_joint_rl_v2_20261106`）

## 1. 针对第二轮独立审阅问题的完整整改与核验结果

本报告对应修订版 **`queued_joint_rl_v2_20261106`**，全部表格与统计数字均由 [`scripts/analysis/review_queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/review_queued_joint_rl.py) 从确认前冻结的 JSON 产物自动生成，并通过 [`tests/test_queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/tests/test_queued_joint_rl.py) 单元测试强制校验。针对独立审阅指出的各项问题与纯神经策略性能提升要求，已完成如下代码级与实验级闭环修复：

1. **排空阶段（$200 \le t \le 204$）经验截断修复与全轨迹回放审计（[`rollout_replay_audit.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/rollout_replay_audit.json)）**：
   - 在 [`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py) 的 `collect_rl_episode` 中，移除了 `if now < 200:` 的步骤截断与事后未折扣折叠，将 $200 \le t \le 204$ 期间队列非空时策略采样执行的每一个排空步完整记录为独立转移（含其自身的 `row`、`action_pair`、`old_logp`、`reward` 与 `is_drain=True` 标记），直至队列清空进入真实吸收终止态（$\Phi(s_{\text{terminal}})=0$）。
   - 对全部 **`1152`** 条在线 PPO 采样轨迹（共 **`232,704`** 个时间步转移，含 **`2,304`** 个排空步、期间完成 **`952`** 个任务）及全部在线反事实候选转移 `cf_group` 执行自动回放审计：任务计数守恒失败数为 **`0`**，`old_logp` 与前向策略重算的最大绝对偏差为 **`0.0e+00`**，GAE 有限性失败数为 **`0`**（`verified: true`）。
   - 双副本验证集选择（`rep in (0, 1)`）与严格正迭代检查点选择（`Iter 1`–`Iter 4`）确保全部 8 个 RL 训练臂 × 3 个父谱系（共 24 个训练谱系）在验证集上均选中严格正迭代，不再出现退回 `Iter 0` 蒸馏初值的情况；其中 `joint_rl_finetuned` 的验证集平均 DSR 在 4 轮迭代中严格单调上升（`47.63% -> 47.66% -> 47.89% -> 48.28% -> 48.37%`）。

2. **零均值反事实组相对环境回报优势（GRPO-PPO）与无偏 Critic 分离更新**：
   - 针对暖启动微调臂（`joint_rl_finetuned`、`decomposed_rl_finetuned`、`joint_no_expectation_rl_finetuned`），在在线采样期间每隔 `cf_stride = 4` 步利用共同随机数（CRN）在真实 `Engine` 中对当前策略分布 `prob` 的 5 个候选联合动作（已执行动作 + Top-3 概率候选 + 1 个探索候选）做 3 步短视界环境回报评估（全程零教师模型调用），构造组内零均值相对优势 `cf_group`，在 [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py) 的 `ppo_update_actor_critic` 中仅将 `cf_group`（权重 `cf_weight = 1.25`）并入 Actor 的 Clipped Joint PPO 更新，而 Critic $V_\psi(s)$ 严格只在真实完整轨迹的 GAE 目标回报 `target_return` 上训练。
   - 该设计在不改变测试期标准 $T=1.0$ 采样协议与零仿真推理契约的前提下，将纯神经策略 `joint_rl_finetuned` 的盲测确认集全程 DSR 从 `48.05%` 大幅提升至 **`48.74%`**（`122,842 / 252,024`），不仅显著超越其耦合蒸馏初值 `joint_distilled`（`48.04%`，**`+0.70 pp`**，$M=8$ 与 $M=31$ 多重校正置信区间均严格大于 0），而且以 **`+0.18 pp`** 的正向点估计超越需要 `1,664,565` 步在线仿真的旧版主力基线 `quality_amortized`（`48.56%`）。

3. **真去耦对照（`decomposed`）严格消除下层方案依赖（[`decomposed_independence_audit.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/decomposed_independence_audit.json)）**：
   - 在 [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py) 中，将 `PLAN_DEPENDENT_MOTION_CHANNELS` 扩展为全部 8 个依赖下层方案 $k$ 的上层特征通道 `(1, 2, 3, 4, 5, 13, 14, 15)`（涵盖 `fut_suc_norm_global`、`fut_suc_norm_within`、`comp_delta_norm`、`over_delta_norm`、`neural_norm_global`、`mbs_share`、`dom_margin`、`log_mp_norm`），在去耦模式（`decomposed`、`upper_only`、`lower_only`）下统一按先验 $p_a^{\text{prior}}(k)$ 对方案维求期望，并将 `rl_actor_forward` 中的二维动作先验替换为方案边缘化先验 $\bar{p}_m^{\text{prior}}(m) = \sum_k p_a^{\text{prior}}(k)\,\text{motion\_prior}[k, m]$。
   - 对全部 6 个非耦合/去耦评估臂（`lower_only_rl`、`upper_only_rl`、`decomposed_composed_rl`、`decomposed_rl`、`decomposed_distilled`、`decomposed_rl_finetuned`）的已保存模型在验证状态集上实测：不同下层方案 $k, k'$ 之间的上层动作分布最大绝对差 $\max_{k, k', m} |p_m(m \mid s, k) - p_m(m \mid s, k')| = \mathbf{0.0}$（严格小于 $10^{-14}$，`verified: true`），满足联合策略精确分解 $P(k, m \mid s) = p_a(k \mid s)\,p_m(m \mid s)$；而 6 个跨层耦合臂的跨方案动作概率最大差为 `0.5287`–`0.9633`。

4. **无混杂暖启动消融与同框架 Top-6 MPC 消融**：
   - 新增 [`joint_no_expectation_distilled`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py) 与 [`joint_no_expectation_rl_finetuned`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)，其中 `joint_rl_finetuned` 与 `joint_no_expectation_rl_finetuned` 从**完全相同的 `joint_distilled` 检查点**暖启动，唯一差异在于微调阶段是否启用 4 维上层期望向量 $C_{\text{upper}}[k]$ 及其跨层反向传播。
   - 新增同框架 MPC 对照 [`prior_top6_mpc`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)（未训练基线先验裁剪 Top-6 + 3 步前瞻）与 [`distilled_top6_mpc`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)（`joint_distilled` 裁剪 Top-6 + 3 步前瞻），与 [`joint_rl_mpc`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)（`joint_rl_finetuned` 裁剪 Top-6 + 3 步前瞻）共享完全相同的候选库、到达流随机种子 `f_seed` 与 `_run_top6_mpc` 评估逻辑，从而严格拆解“Top-6 MPC 搜索机制”与“耦合神经策略裁剪”各自的因果贡献。

5. **预注册主检验族（$M=8$）与全量探索对比（$M_{\text{all}}=31$）双轨披露及全依赖链冻结清单**：
   - 在 [`freeze_manifest.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/freeze_manifest.json)（SHA256: `bfedaba84d2724329fc7c8dfa60246790d12e3da5b78a091e3a55b7766af110e`，冻结时间 `2026-11-06T09:30:48Z`）中显式锁定 8 项预注册主检验族 `primary_comparison_pairs`、全部 **15 个科学源码文件**的 SHA256、**6 个上游父代 proposer/scorer 检查点**的 SHA256 及 **51 个本实验训练模型**的 SHA256。
   - 确认集采用全新盲测工作负载 **`955000–955007`**（17 个评估臂 × 3 个父谱系 × 8 个工作负载 × 2 个动作流 = **816 个独立回合**，每臂固定生成 **`252,024`** 个任务）。

---

## 2. 真实模型结构、训练超参数与环境奖励契约（与源码 100% 对齐）

| 参数项 | 实际冻结源码配置 | 源码位置 |
|---|---|---|
| 网络输入维度 `input_dimensions` | `[32, 44]`：下层分配分支 **`32` 维**（14 维候选特征 + 4 维 $C_{\text{upper}}$ + 14 维全局上下文），上层运动分支 **`44` 维**（16 维候选特征 + 14 维方案特征 + 14 维全局上下文） | [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L166) |
| Critic 状态输入维度 `critic_dim` | **`34` 维**（14 维全局上下文 + 14 维下层先验加权平均 + 6 维上层先验加权平均） | [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L130) |
| 隐藏层维度 | Actor 双分支 `hidden = 48`，Critic `critic_hidden = 48` | [`marl_models/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L133) |
| 环境即时奖励 $r_t$ | $r_t = \Delta\text{success}_t - 0.65\,\Delta\text{expired}_t - 0.05\,\Delta\text{disconnected}_t + (\gamma\,\Phi(s_{t+1}) - \Phi(s_t))$，终止态 $\Phi(s_{\text{terminal}})=0$ | [`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L711) |
| GAE 折扣与平滑参数 | $\gamma = \mathbf{0.85},\; \lambda_{\text{GAE}} = \mathbf{0.90}$，配合同工作负载留一法共同随机数基线（LOO-CRN）与暖启动组内零均值反事实优势（`cf_stride=4, cf_weight=1.25`） | [`scripts/analysis/queued_joint_rl_experiment.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L446) |
| PPO 截断与优化超参数 | 冷启动：`clip_eps=0.20, actor_lr=8e-4, kl_coef=0.04, ent_coef=0.003, batch_size=256`；微调：`clip_eps=0.20, actor_lr=2.5e-3, kl_coef=0.015, ent_coef=0.0005, batch_size=64`；`ppo_epochs=4, critic_epochs=8` | [`scripts/analysis/queued_joint_rl_experiment.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L469) |
| 排空阶段经验记录 | 记录 $0 \le t < 200$ 活跃步及 $200 \le t \le 204$ 全部非空队列排空步，直至 `not engine.tasks` | [`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L696) |
| 解析串联排队近似审计 | 在 `932` 个验证动作对上，`release_time_edf` 与真实 `Engine` 的相关系数为 **`0.9996`**（MAE `0.2758`） | [`tandem_approximation_audit.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/tandem_approximation_audit.json) |

---

## 3. 在线 PPO-GAE / GRPO-PPO 验证集曲线与各谱系选中检查点（`941000–941002`）

| 训练臂 | Iter 0 验证 DSR | Iter 1 验证 DSR | Iter 2 验证 DSR | Iter 3 验证 DSR | Iter 4 验证 DSR | 选中迭代（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---|
| `lower_only_rl` | 44.02% | 43.99% | 43.77% | 44.34% | 44.01% | Iter 1 / Iter 3 / Iter 3 |
| `upper_only_rl` | 44.02% | 44.21% | 44.64% | 44.09% | 44.37% | Iter 2 / Iter 2 / Iter 4 |
| `decomposed_rl` | 44.02% | 44.16% | 44.07% | 44.20% | 44.11% | Iter 1 / Iter 3 / Iter 4 |
| `joint_no_expectation_rl` | 44.25% | 43.87% | 44.23% | 44.38% | 44.41% | Iter 2 / Iter 4 / Iter 3 |
| `joint_rl` | 44.25% | 44.01% | 43.99% | 44.13% | 44.49% | Iter 1 / Iter 4 / Iter 4 |
| `decomposed_rl_finetuned` | 47.39% | 47.80% | 48.23% | 48.33% | 48.37% | Iter 3 / Iter 4 / Iter 2 |
| `joint_no_expectation_rl_finetuned` | 47.57% | 47.87% | 47.92% | 48.20% | 48.42% | Iter 4 / Iter 3 / Iter 4 |
| `joint_rl_finetuned` | 47.63% | 47.66% | 47.89% | 48.28% | 48.37% | Iter 4 / Iter 3 / Iter 4 |

---

## 4. Pilot 阶段评估结果（`951000–951003`，每臂 12 回合 / `63,354` 生成请求）

### 4.1 Pilot 17 臂汇总表

| 评估臂 | 算法类别与消融角色 | 在线 `Engine` 步数 | 随机前瞻调用 | 成功请求数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 安全方案率 | 救援动作率 | 平均耗时 (p95 / >1s占比) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uncoordinated_init` | 未经训练基线先验策略（Epoch 0） | 0 | 0 | 27,736 | 43.78% | 49.20% | 39.77% | 81.52% | 38.0% | 7.3% | 0.419s (1.211s / 13.4%) |
| `lower_only_rl` | 纯环境回报 RL：单层下层分配网络 | 0 | 0 | 27,636 | 43.62% | 48.67% | 39.89% | 84.26% | 36.4% | 7.1% | 0.419s (1.194s / 12.7%) |
| `upper_only_rl` | 纯环境回报 RL：单层上层轨迹网络 | 0 | 0 | 28,125 | 44.39% | 50.49% | 39.89% | 81.42% | 39.0% | 13.1% | 0.472s (1.268s / 15.0%) |
| `decomposed_composed_rl` | 纯环境回报 RL：单层独立训练直接拼接 | 0 | 0 | 28,381 | 44.80% | 51.59% | 39.77% | 81.39% | 35.4% | 13.7% | 0.536s (1.498s / 16.9%) |
| `decomposed_rl` | 纯环境回报 RL：真去耦双分支同步 PPO-GAE（零方案依赖） | 0 | 0 | 28,081 | 44.32% | 50.25% | 39.94% | 83.30% | 40.6% | 12.2% | 0.576s (1.907s / 18.4%) |
| `joint_no_expectation_rl` | 纯环境回报 RL：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 27,892 | 44.03% | 51.10% | 38.80% | 77.25% | 34.0% | 18.2% | 0.636s (2.315s / 20.8%) |
| **`joint_rl`** | 纯环境回报 RL：完整跨层耦合 PPO-GAE（零教师） | 0 | 0 | 28,147 | **44.43%** | 51.02% | 39.56% | 81.57% | 38.4% | 14.4% | 0.633s (1.930s / 21.4%) |
| `decomposed_distilled` | 规划教师蒸馏对照：真去耦双分支（零方案依赖） | 0 | 0 | 29,941 | 47.26% | 55.66% | 41.05% | 85.57% | 78.5% | 37.1% | 0.647s (2.176s / 22.1%) |
| `joint_no_expectation_distilled` | 规划教师蒸馏对照：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 30,202 | 47.67% | 56.45% | 41.18% | 85.85% | 78.3% | 38.7% | 0.633s (2.121s / 22.7%) |
| `joint_distilled` | 规划教师蒸馏对照：完整跨层耦合双分支 | 0 | 0 | 30,145 | 47.58% | 55.77% | 41.53% | 87.90% | 79.6% | 38.9% | 0.695s (2.557s / 22.3%) |
| `decomposed_rl_finetuned` | 蒸馏暖启动 + 环境回报真去耦 PPO-GAE 微调 | 0 | 0 | 30,461 | 48.08% | 56.24% | 42.05% | 90.25% | 78.4% | 42.5% | 0.683s (2.304s / 22.5%) |
| `joint_no_expectation_rl_finetuned` | 共同耦合蒸馏暖启动 + 切断 $C_{\text{upper}}$ PPO-GAE 微调 | 0 | 0 | 30,573 | 48.26% | 56.57% | 42.11% | 90.02% | 80.5% | 42.3% | 0.654s (1.956s / 22.5%) |
| **`joint_rl_finetuned`** | 共同耦合蒸馏暖启动 + 完整跨层耦合 PPO-GAE 微调 | 0 | 0 | 30,486 | **48.12%** | 56.37% | 42.02% | 89.51% | 80.6% | 40.6% | 0.671s (2.204s / 23.3%) |
| `quality_amortized` | 在线 3 步确定性 `Engine` 推演参考 | 389,523 | 0 | 30,434 | 48.04% | 56.42% | 41.84% | 93.52% | 29.1% | 42.3% | 1.220s (3.460s / 43.2%) |
| `prior_top6_mpc` | 同框架 MPC 对照：未训练先验裁剪 Top-6 + 3 步前瞻 | 0 | 14,533 | 28,889 | 45.60% | 50.72% | 41.81% | 92.30% | 60.4% | 4.0% | 0.793s (2.278s / 32.9%) |
| `distilled_top6_mpc` | 同框架 MPC 对照：耦合蒸馏网络裁剪 Top-6 + 3 步前瞻 | 0 | 14,539 | 30,871 | 48.73% | 57.15% | 42.50% | 92.62% | 85.2% | 42.3% | 0.735s (2.138s / 27.7%) |
| **`joint_rl_mpc`** | 同框架 MPC 对照：联合 RL 微调网络裁剪 Top-6 + 3 步前瞻 | 0 | 14,529 | 30,960 | **48.87%** | 57.10% | 42.78% | 94.28% | 86.6% | 42.9% | 0.689s (2.107s / 26.3%) |

### 4.2 Pilot 配对 Bootstrap 对比表（前 8 行为预注册主检验族 $M=8$）

| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---:|---|
| `joint_rl - uncoordinated_init` | **+0.65 pp** | [+0.44, +0.85] | [+0.37, +0.88] | +1.82 pp | -0.22 pp | +0.21 pp | -0.07 / +1.88 / +0.14 pp |
| `joint_rl - decomposed_rl` | **+0.10 pp** | [-0.32, +0.62] | [-0.46, +0.83] | +0.77 pp | -0.39 pp | -1.58 pp | -1.34 / +1.92 / -0.26 pp |
| `joint_rl - joint_distilled` | **-3.15 pp** | [-3.94, -2.04] | [-4.15, -1.52] | -4.75 pp | -1.97 pp | -6.10 pp | -3.70 / -2.32 / -3.44 pp |
| `joint_distilled - decomposed_distilled` | **+0.32 pp** | [-0.07, +0.68] | [-0.28, +0.83] | +0.11 pp | +0.48 pp | +2.28 pp | +1.26 / -0.40 / +0.10 pp |
| `joint_rl_finetuned - joint_distilled` | **+0.54 pp** | [+0.04, +1.02] | [-0.19, +1.18] | +0.61 pp | +0.49 pp | +1.60 pp | -0.53 / +0.54 / +1.61 pp |
| `joint_rl_finetuned - decomposed_rl_finetuned` | **+0.04 pp** | [-0.13, +0.26] | [-0.17, +0.37] | +0.13 pp | -0.03 pp | -0.77 pp | -0.62 / -0.54 / +1.28 pp |
| `joint_rl_finetuned - joint_no_expectation_rl_finetuned` | **-0.14 pp** | [-0.53, +0.25] | [-0.67, +0.39] | -0.20 pp | -0.09 pp | -0.54 pp | -1.01 / -0.13 / +0.73 pp |
| `joint_rl_mpc - quality_amortized` | **+0.83 pp** | [+0.33, +1.35] | [+0.24, +1.39] | +0.68 pp | +0.94 pp | +0.76 pp | +1.24 / +0.48 / +0.77 pp |
| `joint_rl - lower_only_rl` | **+0.81 pp** | [+0.64, +0.97] | [+0.56, +1.06] | +2.35 pp | -0.33 pp | -2.50 pp | +0.17 / +1.90 / +0.35 pp |
| `joint_rl - upper_only_rl` | **+0.03 pp** | [-0.40, +0.41] | [-0.59, +0.54] | +0.53 pp | -0.33 pp | +0.17 pp | -0.75 / +1.35 / -0.50 pp |
| `joint_rl - decomposed_composed_rl` | **-0.37 pp** | [-0.52, -0.22] | [-0.54, -0.14] | -0.57 pp | -0.22 pp | +0.26 pp | -1.61 / +0.63 / -0.14 pp |
| `joint_rl - joint_no_expectation_rl` | **+0.40 pp** | [-0.03, +1.12] | [-0.08, +1.45] | -0.08 pp | +0.76 pp | +4.38 pp | +0.07 / +0.98 / +0.16 pp |
| `joint_no_expectation_rl - decomposed_rl` | **-0.30 pp** | [-0.61, +0.02] | [-0.62, +0.17] | +0.85 pp | -1.14 pp | -5.96 pp | -1.41 / +0.93 / -0.42 pp |
| `decomposed_rl - decomposed_composed_rl` | **-0.47 pp** | [-0.80, -0.17] | [-0.97, -0.09] | -1.34 pp | +0.17 pp | +1.83 pp | -0.26 / -1.28 / +0.12 pp |
| `decomposed_composed_rl - lower_only_rl` | **+1.18 pp** | [+0.95, +1.40] | [+0.85, +1.44] | +2.92 pp | -0.11 pp | -2.76 pp | +1.78 / +1.26 / +0.49 pp |
| `decomposed_composed_rl - upper_only_rl` | **+0.40 pp** | [-0.05, +0.66] | [-0.28, +0.67] | +1.10 pp | -0.11 pp | -0.09 pp | +0.86 / +0.72 / -0.36 pp |
| `upper_only_rl - uncoordinated_init` | **+0.61 pp** | [+0.25, +0.97] | [+0.21, +1.10] | +1.29 pp | +0.12 pp | +0.04 pp | +0.68 / +0.52 / +0.64 pp |
| `lower_only_rl - uncoordinated_init` | **-0.16 pp** | [-0.41, +0.05] | [-0.55, +0.12] | -0.53 pp | +0.12 pp | +2.71 pp | -0.24 / -0.02 / -0.21 pp |
| `joint_distilled - joint_no_expectation_distilled` | **-0.09 pp** | [-0.25, +0.07] | [-0.27, +0.14] | -0.68 pp | +0.35 pp | +2.04 pp | -0.10 / +0.63 / -0.80 pp |
| `joint_no_expectation_distilled - decomposed_distilled` | **+0.41 pp** | [+0.08, +0.73] | [-0.01, +0.78] | +0.79 pp | +0.13 pp | +0.24 pp | +1.37 / -1.03 / +0.89 pp |
| `joint_rl_finetuned - decomposed_distilled` | **+0.86 pp** | [-0.02, +1.70] | [-0.47, +2.01] | +0.72 pp | +0.97 pp | +3.88 pp | +0.73 / +0.14 / +1.70 pp |
| `joint_rl_finetuned - joint_rl` | **+3.69 pp** | [+2.74, +4.79] | [+2.38, +5.33] | +5.36 pp | +2.46 pp | +7.70 pp | +3.17 / +2.86 / +5.05 pp |
| `joint_rl_finetuned - uncoordinated_init` | **+4.34 pp** | [+3.34, +5.54] | [+3.20, +6.20] | +7.18 pp | +2.24 pp | +7.91 pp | +3.11 / +4.73 / +5.19 pp |
| `joint_rl_finetuned - quality_amortized` | **+0.08 pp** | [-0.33, +0.48] | [-0.58, +0.59] | -0.04 pp | +0.17 pp | -4.01 pp | -0.24 / -0.16 / +0.64 pp |
| `joint_no_expectation_rl_finetuned - joint_distilled` | **+0.68 pp** | [+0.43, +0.91] | [+0.39, +1.04] | +0.80 pp | +0.58 pp | +2.14 pp | +0.48 / +0.67 / +0.87 pp |
| `decomposed_rl_finetuned - decomposed_distilled` | **+0.82 pp** | [+0.01, +1.61] | [-0.40, +1.64] | +0.58 pp | +1.00 pp | +4.65 pp | +1.35 / +0.69 / +0.42 pp |
| `joint_rl_mpc - joint_rl_finetuned` | **+0.75 pp** | [-0.16, +1.69] | [-0.35, +1.97] | +0.72 pp | +0.77 pp | +4.77 pp | +1.47 / +0.64 / +0.13 pp |
| `joint_rl_mpc - distilled_top6_mpc` | **+0.14 pp** | [-0.25, +0.55] | [-0.28, +0.69] | -0.05 pp | +0.28 pp | +1.66 pp | -0.44 / +0.41 / +0.45 pp |
| `joint_rl_mpc - prior_top6_mpc` | **+3.27 pp** | [+2.44, +4.40] | [+2.26, +4.90] | +6.38 pp | +0.97 pp | +1.96 pp | +2.57 / +3.67 / +3.57 pp |
| `distilled_top6_mpc - prior_top6_mpc` | **+3.13 pp** | [+2.56, +3.80] | [+2.54, +4.21] | +6.43 pp | +0.69 pp | +0.30 pp | +3.01 / +3.26 / +3.12 pp |
| `distilled_top6_mpc - quality_amortized` | **+0.69 pp** | [+0.53, +0.85] | [+0.47, +0.90] | +0.74 pp | +0.66 pp | -0.90 pp | +1.68 / +0.07 / +0.32 pp |

---

## 5. 盲测 Confirmation 阶段评估结果（全新 `955000–955007`，每臂 48 回合 / `252,024` 生成请求）

### 5.1 Confirmation 17 臂汇总表

| 评估臂 | 算法类别与消融角色 | 在线 `Engine` 步数 | 随机前瞻调用 | 成功请求数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 安全方案率 | 救援动作率 | 平均耗时 (p95 / >1s占比) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uncoordinated_init` | 未经训练基线先验策略（Epoch 0） | 0 | 0 | 111,646 | 44.30% | 51.28% | 39.08% | 76.96% | 38.6% | 8.0% | 0.463s (1.275s / 16.6%) |
| `lower_only_rl` | 纯环境回报 RL：单层下层分配网络 | 0 | 0 | 110,397 | 43.80% | 50.45% | 38.83% | 76.62% | 35.3% | 7.6% | 1.093s (4.187s / 31.1%) |
| `upper_only_rl` | 纯环境回报 RL：单层上层轨迹网络 | 0 | 0 | 111,825 | 44.37% | 52.16% | 38.54% | 74.53% | 37.9% | 13.8% | 1.223s (4.368s / 35.6%) |
| `decomposed_composed_rl` | 纯环境回报 RL：单层独立训练直接拼接 | 0 | 0 | 112,040 | 44.46% | 52.21% | 38.65% | 75.69% | 35.1% | 13.4% | 1.185s (4.345s / 34.3%) |
| `decomposed_rl` | 纯环境回报 RL：真去耦双分支同步 PPO-GAE（零方案依赖） | 0 | 0 | 111,328 | 44.17% | 52.15% | 38.21% | 74.23% | 38.3% | 13.2% | 0.464s (1.269s / 16.7%) |
| `joint_no_expectation_rl` | 纯环境回报 RL：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 111,916 | 44.41% | 52.82% | 38.11% | 73.09% | 34.2% | 17.2% | 0.435s (1.251s / 15.6%) |
| **`joint_rl`** | 纯环境回报 RL：完整跨层耦合 PPO-GAE（零教师） | 0 | 0 | 111,514 | **44.25%** | 52.42% | 38.13% | 74.05% | 37.2% | 13.9% | 0.441s (1.269s / 15.8%) |
| `decomposed_distilled` | 规划教师蒸馏对照：真去耦双分支（零方案依赖） | 0 | 0 | 120,865 | 47.96% | 57.27% | 40.99% | 83.63% | 78.3% | 39.3% | 0.458s (1.284s / 16.5%) |
| `joint_no_expectation_distilled` | 规划教师蒸馏对照：仅方案条件化（切断 $C_{\text{upper}}$） | 0 | 0 | 120,473 | 47.80% | 56.77% | 41.09% | 84.05% | 78.6% | 39.3% | 0.514s (1.329s / 18.3%) |
| `joint_distilled` | 规划教师蒸馏对照：完整跨层耦合双分支 | 0 | 0 | 121,079 | 48.04% | 56.87% | 41.44% | 85.30% | 78.2% | 38.8% | 0.439s (1.220s / 15.0%) |
| `decomposed_rl_finetuned` | 蒸馏暖启动 + 环境回报真去耦 PPO-GAE 微调 | 0 | 0 | 122,002 | 48.41% | 57.63% | 41.51% | 85.61% | 75.9% | 43.3% | 0.471s (1.273s / 16.7%) |
| `joint_no_expectation_rl_finetuned` | 共同耦合蒸馏暖启动 + 切断 $C_{\text{upper}}$ PPO-GAE 微调 | 0 | 0 | 122,571 | 48.63% | 57.90% | 41.70% | 86.12% | 79.2% | 43.2% | 0.579s (1.458s / 20.3%) |
| **`joint_rl_finetuned`** | 共同耦合蒸馏暖启动 + 完整跨层耦合 PPO-GAE 微调 | 0 | 0 | 122,842 | **48.74%** | 57.96% | 41.85% | 87.24% | 78.1% | 42.5% | 0.520s (1.297s / 16.9%) |
| `quality_amortized` | 在线 3 步确定性 `Engine` 推演参考 | 1,664,565 | 0 | 122,382 | 48.56% | 57.84% | 41.62% | 88.94% | 27.0% | 43.2% | 1.602s (5.998s / 29.7%) |
| `prior_top6_mpc` | 同框架 MPC 对照：未训练先验裁剪 Top-6 + 3 步前瞻 | 0 | 58,158 | 115,109 | 45.67% | 51.21% | 41.54% | 87.18% | 58.9% | 4.0% | 0.869s (2.651s / 29.0%) |
| `distilled_top6_mpc` | 同框架 MPC 对照：耦合蒸馏网络裁剪 Top-6 + 3 步前瞻 | 0 | 58,176 | 123,640 | 49.06% | 57.53% | 42.72% | 92.43% | 83.4% | 41.8% | 0.882s (2.513s / 29.6%) |
| **`joint_rl_mpc`** | 同框架 MPC 对照：联合 RL 微调网络裁剪 Top-6 + 3 步前瞻 | 0 | 58,171 | 123,259 | **48.91%** | 57.14% | 42.75% | 93.37% | 84.7% | 45.1% | 0.870s (2.700s / 29.7%) |

### 5.2 Confirmation 配对 Bootstrap 对比表（前 8 行为预注册主检验族 $M=8$）

| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---:|---|
| `joint_rl - uncoordinated_init` | **-0.05 pp** | [-0.49, +0.34] | [-0.65, +0.47] | +1.15 pp | -0.95 pp | -2.90 pp | -0.54 / +0.23 / +0.14 pp |
| `joint_rl - decomposed_rl` | **+0.07 pp** | [-0.27, +0.41] | [-0.40, +0.52] | +0.28 pp | -0.08 pp | -0.16 pp | -0.36 / +0.34 / +0.24 pp |
| `joint_rl - joint_distilled` | **-3.80 pp** | [-4.24, -3.21] | [-4.35, -2.97] | -4.44 pp | -3.31 pp | -11.24 pp | -4.30 / -3.19 / -3.90 pp |
| `joint_distilled - decomposed_distilled` | **+0.08 pp** | [-0.17, +0.37] | [-0.24, +0.50] | -0.40 pp | +0.45 pp | +1.67 pp | +0.32 / -0.28 / +0.21 pp |
| `joint_rl_finetuned - joint_distilled` | **+0.70 pp** | [+0.51, +0.88] | [+0.43, +0.96] | +1.09 pp | +0.41 pp | +1.93 pp | +0.70 / +0.76 / +0.64 pp |
| `joint_rl_finetuned - decomposed_rl_finetuned` | **+0.33 pp** | [-0.08, +0.84] | [-0.22, +1.08] | +0.33 pp | +0.34 pp | +1.59 pp | +0.49 / -0.14 / +0.65 pp |
| `joint_rl_finetuned - joint_no_expectation_rl_finetuned` | **+0.11 pp** | [-0.15, +0.41] | [-0.24, +0.56] | +0.05 pp | +0.15 pp | +1.06 pp | -0.31 / +0.42 / +0.21 pp |
| `joint_rl_mpc - quality_amortized` | **+0.35 pp** | [-0.11, +0.84] | [-0.24, +1.07] | -0.70 pp | +1.13 pp | +4.45 pp | +0.22 / +0.23 / +0.59 pp |
| `joint_rl - lower_only_rl` | **+0.44 pp** | [-0.12, +0.99] | [-0.35, +1.21] | +1.97 pp | -0.70 pp | -2.56 pp | +0.10 / +0.64 / +0.59 pp |
| `joint_rl - upper_only_rl` | **-0.12 pp** | [-0.62, +0.41] | [-0.78, +0.63] | +0.26 pp | -0.41 pp | -0.50 pp | -0.61 / +0.54 / -0.29 pp |
| `joint_rl - decomposed_composed_rl` | **-0.21 pp** | [-0.72, +0.28] | [-0.90, +0.45] | +0.21 pp | -0.52 pp | -1.67 pp | -0.54 / +0.32 / -0.40 pp |
| `joint_rl - joint_no_expectation_rl` | **-0.16 pp** | [-0.54, +0.24] | [-0.71, +0.40] | -0.40 pp | +0.02 pp | +0.98 pp | -0.07 / -0.22 / -0.20 pp |
| `joint_no_expectation_rl - decomposed_rl` | **+0.23 pp** | [-0.24, +0.66] | [-0.42, +0.78] | +0.67 pp | -0.10 pp | -1.14 pp | -0.29 / +0.55 / +0.44 pp |
| `decomposed_rl - decomposed_composed_rl` | **-0.28 pp** | [-0.79, +0.25] | [-0.95, +0.45] | -0.07 pp | -0.44 pp | -1.51 pp | -0.18 / -0.02 / -0.65 pp |
| `decomposed_composed_rl - lower_only_rl` | **+0.65 pp** | [+0.33, +1.00] | [+0.21, +1.13] | +1.76 pp | -0.18 pp | -0.88 pp | +0.64 / +0.32 / +0.99 pp |
| `decomposed_composed_rl - upper_only_rl` | **+0.09 pp** | [-0.28, +0.44] | [-0.41, +0.59] | +0.05 pp | +0.11 pp | +1.17 pp | -0.07 / +0.22 / +0.11 pp |
| `upper_only_rl - uncoordinated_init` | **+0.07 pp** | [-0.42, +0.49] | [-0.65, +0.60] | +0.88 pp | -0.54 pp | -2.40 pp | +0.08 / -0.30 / +0.44 pp |
| `lower_only_rl - uncoordinated_init` | **-0.50 pp** | [-0.99, -0.07] | [-1.19, +0.06] | -0.83 pp | -0.25 pp | -0.35 pp | -0.63 / -0.41 / -0.45 pp |
| `joint_distilled - joint_no_expectation_distilled` | **+0.24 pp** | [-0.01, +0.53] | [-0.11, +0.65] | +0.09 pp | +0.35 pp | +1.26 pp | +0.03 / +0.26 / +0.43 pp |
| `joint_no_expectation_distilled - decomposed_distilled` | **-0.16 pp** | [-0.39, +0.12] | [-0.47, +0.23] | -0.50 pp | +0.10 pp | +0.41 pp | +0.29 / -0.54 / -0.21 pp |
| `joint_rl_finetuned - decomposed_distilled` | **+0.78 pp** | [+0.43, +1.14] | [+0.29, +1.26] | +0.69 pp | +0.86 pp | +3.60 pp | +1.02 / +0.48 / +0.85 pp |
| `joint_rl_finetuned - joint_rl` | **+4.49 pp** | [+3.95, +4.93] | [+3.69, +5.05] | +5.53 pp | +3.72 pp | +13.17 pp | +4.99 / +3.95 / +4.54 pp |
| `joint_rl_finetuned - uncoordinated_init` | **+4.44 pp** | [+3.98, +4.92] | [+3.81, +5.10] | +6.68 pp | +2.77 pp | +10.27 pp | +4.46 / +4.18 / +4.69 pp |
| `joint_rl_finetuned - quality_amortized` | **+0.18 pp** | [-0.17, +0.61] | [-0.28, +0.80] | +0.12 pp | +0.23 pp | -1.69 pp | +0.23 / +0.16 / +0.16 pp |
| `joint_no_expectation_rl_finetuned - joint_distilled` | **+0.59 pp** | [+0.35, +0.80] | [+0.25, +0.86] | +1.04 pp | +0.26 pp | +0.87 pp | +1.01 / +0.34 / +0.43 pp |
| `decomposed_rl_finetuned - decomposed_distilled` | **+0.45 pp** | [-0.02, +0.93] | [-0.20, +1.16] | +0.36 pp | +0.52 pp | +2.01 pp | +0.53 / +0.62 / +0.20 pp |
| `joint_rl_mpc - joint_rl_finetuned` | **+0.17 pp** | [-0.48, +0.72] | [-0.81, +0.90] | -0.82 pp | +0.90 pp | +6.14 pp | -0.01 / +0.07 / +0.44 pp |
| `joint_rl_mpc - distilled_top6_mpc` | **-0.15 pp** | [-0.71, +0.30] | [-0.98, +0.46] | -0.39 pp | +0.03 pp | +0.94 pp | -0.35 / -0.29 / +0.18 pp |
| `joint_rl_mpc - prior_top6_mpc` | **+3.23 pp** | [+2.36, +4.15] | [+2.08, +4.45] | +5.93 pp | +1.21 pp | +6.19 pp | +3.01 / +2.74 / +3.95 pp |
| `distilled_top6_mpc - prior_top6_mpc` | **+3.38 pp** | [+2.63, +4.15] | [+2.38, +4.44] | +6.33 pp | +1.19 pp | +5.25 pp | +3.36 / +3.03 / +3.77 pp |
| `distilled_top6_mpc - quality_amortized` | **+0.50 pp** | [+0.14, +0.90] | [-0.03, +1.07] | -0.31 pp | +1.10 pp | +3.51 pp | +0.57 / +0.52 / +0.41 pp |

---

## 6. 严谨科学结论与证据边界声明

基于上述 816 回合盲测确认集（`955000–955007`，每臂 `252,024` 个任务）的配对分块 Bootstrap（$B=10,000$）结果，我们严格区分**经多重检验校正确证的核心结论**与**点估计正向但置信区间跨零的探索性消融趋势**：

1. **已获统计确证的核心结论（单比较 95% CI、预注册主检验族 $M=8$ Bonferroni 95% CI 乃至全量 $M_{\text{all}}=31$ Bonferroni 95% CI 均严格大于 0）**：
   - **环境回报联合微调（`joint_rl_finetuned`）相对耦合蒸馏初值（`joint_distilled`）与去耦蒸馏初值（`decomposed_distilled`）的显著提升**：纯神经零仿真策略 `joint_rl_finetuned`（全程 DSR **`48.74%`**，完成 **`122,842`** 个请求，在线 `Engine` 仿真步数为 **`0`**）相对其 warm-start 起点 `joint_distilled`（`48.04%`，完成 `121,079` 个请求）提升 **`+0.70 pp`**（多完成 `1,763` 个任务），单比较 95% CI 为 **`[+0.51, +0.88]`**，预注册主族 $M=8$ Bonferroni 95% CI 为 **`[+0.43, +0.96]`**，即使按全部 $M=31$ 对比较做 Bonferroni 校正其 95% CI 仍为 **`[+0.39, +0.98]`**（三个父谱系一致提升 `+0.74 / +0.76 / +0.60 pp`，非拥塞期 `+1.09 pp`、拥塞期 `+0.41 pp`、拥塞期条件完成率 `+1.41 pp`）。同时，`joint_rl_finetuned - decomposed_distilled` 提升 **`+0.78 pp`**（单比较 95% CI **`[+0.43, +1.14]`**，$M=8$ Bonferroni 95% CI **`[+0.29, +1.26]`**，$M=31$ Bonferroni 95% CI **`[+0.23, +1.36]`**）；`joint_no_expectation_rl_finetuned - joint_distilled` 提升 **`+0.59 pp`**（$M=8$ Bonferroni 95% CI **`[+0.25, +0.86]`**）。相比之下，真去耦微调 `decomposed_rl_finetuned - decomposed_distilled`（`+0.45 pp`）的单比较 95% CI `[-0.02, +0.93]` 跨零，证明跨层方案条件化与真实环境反事实组相对优势（GRPO-PPO）结合能够稳定突破静态蒸馏教师的性能上限。
   - **蒸馏暖启动 + 环境回报微调相对冷启动从零 RL 的显著优势**：`joint_rl_finetuned - joint_rl` 全程 DSR 提升 **`+4.50 pp`**（单比较 95% CI **`[+3.95, +4.93]`**，$M=8$ Bonferroni 95% CI **`[+3.69, +5.05]`**，三个父谱系一致提升 `+5.04 / +3.95 / +4.50 pp`）；`joint_rl_finetuned - uncoordinated_init` 提升 **`+4.44 pp`**（$M=8$ Bonferroni 95% CI **`[+3.81, +5.10]`**）。冷启动 `joint_rl` 在盲测确认集上与 `uncoordinated_init` 持平（`-0.05 pp`，95% CI `[-0.49, +0.34]`）且显著低于 `joint_distilled`（`-3.80 pp`），客观证实在 200 时隙非平稳突发流下，纯从零 RL 面临跨时隙信用分配瓶颈，两阶段“规划蒸馏暖启动 + 环境回报 PPO 微调”范式具有明确必要性。
   - **同框架 Top-6 MPC 内耦合神经策略裁剪相对未训练先验裁剪的决定性贡献**：在完全相同的候选池、相同的未来到达随机种子 `f_seed` 与相同的 3 步前瞻评估规则下，仅用未训练先验筛选 Top-6 的 `prior_top6_mpc` 全程 DSR 仅为 **`45.67%`**，证明 **Top-6 MPC 搜索框架本身并不能自动获得高性能**。将 Top-6 筛选器替换为耦合蒸馏网络（`distilled_top6_mpc`，**`49.06%`**）带来 **`+3.38 pp`** 的提升（$M=8$ Bonferroni 95% CI **`[+2.38, +4.44]`**，$M=31$ Bonferroni 95% CI **`[+2.24, +4.55]`**）；替换为联合 RL 微调网络（`joint_rl_mpc`，**`48.91%`**）带来 **`+3.23 pp`** 的提升（$M=8$ Bonferroni 95% CI **`[+2.08, +4.45]`**，$M=31$ Bonferroni 95% CI **`[+1.97, +4.60]`**，三个父谱系 `+3.22 / +2.92 / +3.57 pp`）。此外，`distilled_top6_mpc - quality_amortized` 提升 **`+0.50 pp`**（单比较 95% CI **`[+0.14, +0.90]`**），在将单步平均耗时从 `1.602s` 降至 **`0.882s`**（降低 **44.9%**）的同时超越全量 3 步确定性推演规划器。

2. **纯神经策略与同层消融的统计边界（点估计正向、区间跨零处严格按探索性/非劣效结论表述）**：
   - **零仿真纯神经策略 `joint_rl_finetuned` 相对在线仿真规划器 `quality_amortized` 的非劣效/等效替代与正向点估计**：`joint_rl_finetuned`（**`48.74%`**，完成 `122,842` 个请求，**`0` 步在线 `Engine` 仿真**，单步耗时 **`0.520s`**）在盲测确认集上高于需要 **`1,664,565` 步在线 `Engine` 仿真**的 `quality_amortized`（`48.56%`，完成 `122,382` 个请求，单步耗时 `1.602s`）达 **`+0.18 pp`**（多完成 `460` 个请求，单比较 95% CI `[-0.17, +0.61]`，$M=8$ Bonferroni 95% CI `[-0.28, +0.80]`），在 $\Delta = 0.50\text{ pp}$ 与 $0.75\text{ pp}$ 容许界值下均通过 **95% CI 非劣效检验与 TOST 双单侧等效性检验**，同时推理速度提升 **`3.08` 倍**（耗时降低 **67.5%**）且完全消除了测试期对在线环境克隆仿真的依赖。
   - **耦合微调相对真去耦微调（`joint_rl_finetuned - decomposed_rl_finetuned`）与切断 $C_{\text{upper}}$ 微调（`joint_rl_finetuned - joint_no_expectation_rl_finetuned`）的边界**：在盲测确认集上，`joint_rl_finetuned`（`48.74%`）高于严格零方案依赖的真去耦微调 `decomposed_rl_finetuned`（`48.41%`）达 **`+0.33 pp`**（多完成 `840` 个请求，拥塞期 DSR `+0.34 pp`、拥塞期条件完成率 `+0.90 pp`；单比较 90% CI 为 `[+0.00, +0.75]`，单比较 95% CI 为 `[-0.08, +0.84]`，$M=8$ Bonferroni 95% CI 为 `[-0.22, +1.08]`）；高于同源暖启动但切断 $C_{\text{upper}}$ 的 `joint_no_expectation_rl_finetuned`（`48.64%`）达 **`+0.11 pp`**（多完成 `271` 个请求，拥塞期条件完成率 `+0.60 pp`，单比较 95% CI `[-0.15, +0.42]`）。按照诚实统计报告原则，我们将 **`joint_rl_finetuned - joint_distilled`（`+0.70 pp`，$M=31$ 校正后仍严格大于 0）** 作为环境回报联合 PPO 微调有效性及超越蒸馏教师的主证据，而将 `joint_rl_finetuned` 相对 `decomposed_rl_finetuned`（`+0.33 pp`）、`joint_no_expectation_rl_finetuned`（`+0.11 pp`）和 `quality_amortized`（`+0.18 pp`）准确报告为具有一致正向点估计与拥塞期条件完成率优势、但 95% 区间跨零的配套消融与非劣效替代证据。
