# 对比算法实现与论文实验方案

## 一、修改目标

本文目标不是简单堆叠 baseline 数量，而是构造一组能支撑论文第五章叙事的对比实验。参考《基于深度强化学习的无人机辅助通信资源分配算法研究_刘潇.pdf》的实验组织方式，建议采用如下结构：

1. 给出统一仿真参数表和算法参数表。
2. 用 3-5 类典型算法与本文方法对比，包括非学习方法、经典 DRL 方法、MARL 方法和本文关键消融。
3. 结果图表围绕收敛性、轨迹行为、主指标对比和参数敏感性展开。
4. 每个 baseline 都要回答一个明确问题，而不是只为了增加算法数量。

因此，本方案将原始计划中的 5 个 baseline 重构为“论文主对比 + 关键消融 + 可选扩展”的三层体系，便于 Claude Code 分阶段实现，也便于后续写入论文。

---

## 二、最终推荐对比体系

### 2.1 主对比算法清单

| 编号 | 方法名称 | 类型 | 是否建议进主表 | 核心对比问题 | 实现状态 |
|------|----------|------|----------------|--------------|----------|
| 1 | Random | 非学习随机策略 | 是 | 学习方法是否明显优于随机下界 | 需新增轻量实现 |
| 2 | Greedy/TSP + Heuristic Offload | 启发式方法 | 是 | 传统轨迹/贪心策略能达到什么水平 | 可复用 greedy 思路，需整理为统一 baseline |
| 3 | DQN-Trajectory + Heuristic Offload | 经典 DRL | 建议是 | 离散动作 DRL 在连续轨迹问题中的局限 | 可选新增，注意边界 |
| 4 | IPPO | MARL 消融 | 是 | 无 CTDE/独立学习是否弱于协作训练 | 需新增 |
| 5 | Vanilla MAPPO | MARL 消融 | 是 | 去掉 Attention 后性能如何 | 已存在，需接入统一实验 |
| 6 | Joint MAPPO | 结构消融 | 是 | 直接联合动作空间是否难训 | 已存在，需接入统一实验 |
| 7 | Proposed Hierarchical Attention-MAPPO | 本文方法 | 是 | 本文完整双层协同框架效果 | 已存在 |

### 2.2 论文中的推荐表述

建议在论文中将这些方法分成三组描述：

1. **非学习基线**：Random、Greedy/TSP。
2. **经典深度强化学习基线**：DQN-Trajectory。
3. **多智能体强化学习基线与消融**：IPPO、Vanilla MAPPO、Joint MAPPO、Proposed。

这样写比“5 个 baseline 每个消融一个组件”更自然。因为 Random 和 TSP 并不是严格消融组件，它们更适合作为非学习下界；Vanilla MAPPO、IPPO、Joint MAPPO 才是和本文结构强相关的消融。

---

## 三、重要修正点

### 3.1 删除原计划中的重复和错误

原计划需要修正以下问题：

1. `4.4 Vanilla MAPPO` 段落后半部分混入了 DQN 内容，后面 `4.5 DQN Baseline` 又重复了一次 DQN，需要删除混入部分。
2. `Vanilla MAPPO` 已经存在于 `marl_models/vanilla_mappo/`，不应写成从零新增，应写成“复用现有实现并接入统一实验脚本”。
3. `Joint MAPPO` 已经存在于 `marl_models/joint_mappo/`，应纳入主对比，因为它能证明“直接联合轨迹+卸载动作空间”的训练难度。
4. “IPPO 和 Vanilla MAPPO 使用 CTDE 训练框架”这句话不准确。Vanilla MAPPO 使用 CTDE；IPPO 应是 independent learning，不使用 centralized critic。
5. DQN 的动作离散化索引应按 `N_DISTANCES` 拆分，而不是按 `N_DIRECTIONS` 拆分。
6. DQN 不建议完整承担“轨迹+请求级卸载”双层联合决策，否则动作空间会过大且不公平。建议定位为 `DQN-Trajectory + Heuristic Offload`。

### 3.2 参考文献表述修正

以下文献方向可以保留，但写作时要谨慎：

| 用途 | 推荐引用 | 说明 |
|------|----------|------|
| Random baseline | Li et al., Sensors 2022, DOI: `10.3390/s22103854` | 文中使用 RANDOM/LOCAL 作为 benchmark，可支撑随机卸载下界 |
| TSP/传统轨迹 | Lu et al., Applied Intelligence 2024, DOI: `10.1007/s10489-024-05339-8` | 需核对原文中 TSP 表述，避免题名或算法名写错 |
| IPPO/MAPPO 对比 | Chen et al., Drones 2026, DOI: `10.3390/drones10020116` | 可支撑 IPPO、standard MAPPO、改进 MAPPO 的对比逻辑 |
| Vanilla/Pure MAPPO | Bin Li et al., IEEE IoT-J 2024, DOI: `10.1109/JIOT.2023.3300718` | 可支撑 Pure-MAPPO / MAPPO 类 baseline |
| DQN trajectory/off-policy baseline | Zhang et al., Drones 2024, DOI: `10.3390/drones8090485` | 可支撑 DQN-COTO 类离散动作 baseline |

Claude Code 不需要联网查文献，只需要在实现计划中保留 DOI 和用途；最终论文写作前再人工核对 BibTeX。

---

## 四、各 baseline 的最终设计

### 4.1 Random Baseline

#### 论文定位

Random 用于提供绝对下界，说明本文方法不是依靠环境奖励设计自然得到提升，而是真正学习到了有效轨迹和卸载策略。

#### 决策逻辑

轨迹：

```text
对每架 UAV，在每个 time slot 随机采样 2D 方向向量。
若方向向量范数过小，则重新采样或置为悬停。
执行前按环境已有边界/碰撞逻辑处理。
```

卸载：

```text
对每个有效请求，在动作 mask 允许的动作集合中随机选择。
如果不接入 mask，则至少保证动作编号合法。
建议使用 mask-aware random，避免大量非法动作导致对比失真。
```

#### 推荐实现文件

```text
marl_models/random_baseline/
├── __init__.py
└── random_baseline.py
```

#### 关键接口

```python
class RandomBaseline(MARLModel):
    def select_trajectory_actions(self, obs):
        """Return shape: (NUM_UAVS, ACTION_DIM)."""

    def select_offload_actions(self, offload_obs, action_masks=None):
        """Return shape compatible with Env.step offload actions."""
```

#### 实现注意

1. 不需要训练、保存、加载模型。
2. 需要支持 seed，保证多次评估可复现。
3. 输出 metrics 格式必须与其他实验一致。

---

### 4.2 Greedy/TSP + Heuristic Offload Baseline

#### 论文定位

该方法代表传统启发式轨迹规划。它回答的问题是：如果 UAV 只根据 UE 空间位置移动，而不学习长期奖励，系统能达到什么性能。

#### 命名建议

论文里可写为：

```text
Greedy-TSP + Heuristic Offloading
```

代码里可写为：

```text
greedy_tsp_baseline
```

#### 轨迹策略

不要实现完整 NP-hard TSP 求解器，使用最近邻贪心即可，论文中说明为 greedy nearest-neighbor TSP heuristic。

推荐逻辑：

```text
1. 每架 UAV 维护一个目标 UE。
2. 若当前目标不存在、已接近或离开服务集合，则从未覆盖/低电量/有请求 UE 中选择最近者。
3. UAV 朝目标 UE 移动，方向向量归一化为环境动作。
4. 多 UAV 目标选择时尽量避免全部 UAV 追同一个 UE，可按 UAV id 轮转或分区选择。
```

#### 卸载策略

建议不要写成“永远本地执行”，因为这会在缓存未命中或本地拥塞时过弱。更合理的 heuristic：

```text
1. 若本地 UAV 能满足 deadline，则 local。
2. 否则在可行协作 UAV 中选择估计 latency 最小者。
3. 若协作不可行或超时，则 MBS。
4. 若启用了 action mask，则只在 mask 允许动作中选择。
```

如果实现时间有限，可以先做 local-first heuristic，但文档中要说明这是一个弱启发式基线。

#### 推荐实现文件

```text
marl_models/greedy_tsp_baseline/
├── __init__.py
├── greedy_tsp_baseline.py
└── trajectory_planner.py
```

#### 实现注意

1. 不需要训练。
2. 需要和 Random 一样走统一评估脚本。
3. 需要记录轨迹，便于画轨迹对比图。

---

### 4.3 DQN-Trajectory + Heuristic Offload Baseline

#### 论文定位

DQN 用于仿照参考论文中“经典 DRL baseline”的写法，但要明确它只学习离散化轨迹动作，卸载仍采用启发式策略。

推荐论文名称：

```text
DQN-Trajectory
```

不要写成完整的 `DQN joint trajectory-offloading`，除非后续真的实现了可训练且动作空间合理的联合 DQN。

#### 动作空间

建议先使用较小动作空间，降低训练风险：

```text
N_DIRECTIONS = 8 或 16
N_DISTANCES = 2 或 3
N_ACTIONS = N_DIRECTIONS * N_DISTANCES + 1  # +1 表示 hover
```

原计划的 `36 × 5 = 180` 个动作也可以，但对当前环境和训练时长压力更大，不建议第一版采用。

#### 动作离散化修正

正确拆分应为：

```python
angle_idx = action_idx // N_DISTANCES
dist_idx = action_idx % N_DISTANCES
```

如果包含 hover，则建议：

```python
if action_idx == 0:
    return np.zeros(2, dtype=np.float32)
shifted = action_idx - 1
angle_idx = shifted // N_DISTANCES
dist_idx = shifted % N_DISTANCES
```

#### 推荐实现文件

```text
marl_models/dqn_baseline/
├── __init__.py
├── dqn_baseline.py
├── dqn_network.py
├── replay_buffer.py
└── discretization.py
```

#### 训练方式

建议参数共享，即所有 UAV 共享同一个 Q 网络：

```text
transition = (local_obs_i, discrete_action_i, shared_reward, next_local_obs_i, done)
```

奖励可以先使用系统 reward，而不是单独设计 per-agent reward，保证和其他方法指标一致。

#### 实现注意

1. DQN 是可选增强项。如果时间紧，优先实现 Random、Greedy、IPPO，并复用 Vanilla MAPPO/Joint MAPPO。
2. DQN 只作为经典 DRL 对比，不作为本文核心消融。
3. 如果 DQN 不稳定，论文里可以只报告收敛曲线和最终指标，但不要过度解释。

---

### 4.4 IPPO Baseline

#### 论文定位

IPPO 用于消融 CTDE 和多智能体协作。它回答的问题是：如果每架 UAV 独立学习，只依赖自己的局部 critic，性能是否下降。

#### 与 Vanilla MAPPO 的区别

| 项目 | IPPO | Vanilla MAPPO |
|------|------|---------------|
| Actor | 局部观测 | 局部观测 |
| Critic | 局部 critic `V_i(o_i)` | 集中式 critic `V(s)` |
| CTDE | 否 | 是 |
| Attention | 否 | 否 |
| 参数共享 | 可共享，也可每 agent 独立 | 通常共享 |

#### 推荐实现策略

为了降低代码量，优先复用 `marl_models/vanilla_mappo/agents.py` 中的 MLP actor/critic 结构，但 critic 输入改为单 UAV 局部观测，而不是全局 state。

推荐文件：

```text
marl_models/ippo_baseline/
├── __init__.py
├── ippo_baseline.py
└── agents.py
```

#### 训练逻辑

```text
1. 每架 UAV 使用局部观测 o_i 采样动作 a_i。
2. 环境返回共享系统 reward。
3. 每个 agent 的 buffer 存储 (o_i, a_i, log_prob_i, reward, done, value_i)。
4. 使用局部 critic 计算 GAE。
5. 使用 PPO clipped objective 更新 actor 和 local critic。
```

#### 实现注意

1. IPPO 不要使用 centralized critic。
2. 可以参数共享 actor/critic，这仍然可以称为 IPPO，只要 critic 输入不是全局 state。
3. 评估时输出格式与 MAPPO 保持一致。

---

### 4.5 Vanilla MAPPO Baseline

#### 论文定位

Vanilla MAPPO 是本文最重要的 attention 消融。它回答的问题是：不使用 attention，仅使用 MLP 表征时，多 UAV 协作能力和性能会如何变化。

#### 当前代码状态

仓库中已经存在：

```text
marl_models/vanilla_mappo/
├── __init__.py
├── agents.py
└── vanilla_mappo.py
```

因此不需要新增模型主体，只需要：

1. 确认 `marl_models/utils.py` 已支持 `vanilla_mappo`。
2. 在统一实验脚本中加入 `trajectory_model_name="vanilla_mappo"`。
3. 保持 PPO 超参数、训练 seed、workload seed 与 proposed 方法一致。

#### 推荐实验设置

```text
Upper: vanilla_mappo
Lower: constrained_attention_offload_mappo
```

这个设置只消融上层 trajectory attention。如果还要消融下层 offload attention，可以使用已有 `no_attention_offload_mappo` 作为下层消融，但建议放到 ablation 表，不放主对比表。

---

### 4.6 Joint MAPPO Baseline

#### 论文定位

Joint MAPPO 用于证明双层分解的必要性。它回答的问题是：如果不拆分上层轨迹和下层请求级卸载，而是直接端到端联合学习，训练难度和最终性能如何。

#### 当前代码状态

仓库中已经存在：

```text
marl_models/joint_mappo/
├── __init__.py
├── agents.py
└── joint_mappo.py
```

同时已有：

```text
run_joint_end_to_end_mappo_experiment.py
evaluate_joint_end_to_end_mappo.py
```

#### 推荐处理

1. 不新增 Joint MAPPO 模型。
2. 检查现有 runner 的结果输出字段是否与其他 baseline 一致。
3. 如果字段不一致，新增一个转换/汇总层，不要大改模型。
4. 在论文中强调它是 structural baseline，不是 attention 消融。

---

### 4.7 Proposed Hierarchical Attention-MAPPO

#### 论文定位

本文完整方法：

```text
Upper: attention_mappo
Lower: constrained_attention_offload_mappo
Components: attention encoder + quality-aware action mask + Lagrange constraints
```

建议在主表中叫：

```text
H-Attention-MAPPO (Proposed)
```

在消融表中再拆：

```text
Proposed w/o upper attention
Proposed w/o lower attention
Proposed w/o quality mask
Proposed w/o Lagrange
Joint MAPPO
```

---

## 五、统一实验与评估设计

### 5.1 推荐新增统一入口

新增一个统一 baseline 入口，而不是为每个 baseline 写完全独立的脚本。

推荐文件：

```text
run_baseline_comparison_experiment.py
evaluate_baseline_comparison.py
recent/run_baseline_comparison_full.sh
```

### 5.2 统一命令示例

```bash
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline dqn_trajectory --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 200 --eval_episodes 10 --seed 42
```

建议统一使用：

```text
--train_episodes
--eval_episodes
```

非学习方法忽略 `--train_episodes`，只使用 `--eval_episodes`。

### 5.3 统一结果目录

建议所有 baseline 输出到：

```text
results/baseline_comparison/
├── random/
├── greedy_tsp/
├── dqn_trajectory/
├── ippo/
├── vanilla_mappo/
├── joint_mappo/
└── proposed/
```

每个目录下至少包含：

```text
metrics_seed_<seed>.json
training_curve_seed_<seed>.json       # 非学习 baseline 可不含
trajectory_seed_<seed>.json           # 用于轨迹图
config_seed_<seed>.json
```

### 5.4 统一 JSON 指标字段

所有方法必须输出相同字段，便于统计：

```json
{
  "method": "proposed",
  "seed": 42,
  "workload_seed": 1001,
  "reward": -1451.7,
  "latency": 0.0,
  "energy": 0.0,
  "fairness": 0.0,
  "deadline_satisfaction_rate": 0.0,
  "offline_rate": 0.0,
  "offloading_ratio_local": 0.0,
  "offloading_ratio_cooperative": 0.0,
  "offloading_ratio_mbs": 0.0,
  "mbs_load_ratio": 0.0,
  "collision_count": 0,
  "boundary_violation_count": 0
}
```

---

## 六、论文图表规划

### 6.1 主指标对比表

建议表格字段：

| Method | Reward | Latency | Energy | Fairness | DSR | Offline Rate | MBS Load |
|--------|--------|---------|--------|----------|-----|--------------|----------|
| Random | | | | | | | |
| Greedy-TSP | | | | | | | |
| DQN-Trajectory | | | | | | | |
| IPPO | | | | | | | |
| Vanilla MAPPO | | | | | | | |
| Joint MAPPO | | | | | | | |
| Proposed | | | | | | | |

每个值建议写：

```text
mean ± 95% CI
```

### 6.2 收敛曲线图

参考 PDF 的写法，建议画：

```text
x-axis: training episodes
y-axis: episode reward 或 moving-average reward
methods: DQN-Trajectory, IPPO, Vanilla MAPPO, Joint MAPPO, Proposed
```

Random 和 Greedy 没有训练曲线，不放入收敛图。

### 6.3 轨迹对比图

建议选同一个 seed 和 workload seed，画：

```text
Random
Greedy-TSP
DQN-Trajectory
Vanilla MAPPO
Proposed
```

Joint MAPPO/IPPO 可选，避免图太拥挤。

### 6.4 参数敏感性图

仿照 PDF 中“不同 UAV 数量/带宽”的实验形式，建议选择 2 个最贴合本文的参数：

1. UAV 数量：`NUM_UAVS = 3, 5, 7`
2. UE 数量或 workload 强度：`NUM_UES = 60, 100, 140` 或请求到达率分档

如果修改环境参数成本较高，优先做 workload 强度，因为它通常比改 UAV 数量更少触发维度改动。

---

## 七、实现顺序

### 阶段 0：清理计划与确认现有代码

目标：

1. 确认 `vanilla_mappo`、`joint_mappo`、`uncoordinated_greedy` 的当前可运行状态。
2. 不重写已有模型。
3. 先统一结果字段。

验证：

```bash
python -m py_compile marl_models/utils.py
python -m py_compile run_hierarchical_mappo_experiment.py
python -m py_compile run_joint_end_to_end_mappo_experiment.py
```

### 阶段 1：统一评估数据结构

新增：

```text
utils/baseline_metrics.py
```

职责：

1. 统一 metrics 字段。
2. 提供 `save_metrics_json()`。
3. 提供 `aggregate_metrics()`。

验证：

```bash
python -m py_compile utils/baseline_metrics.py
```

### 阶段 2：实现 Random baseline

新增：

```text
marl_models/random_baseline/__init__.py
marl_models/random_baseline/random_baseline.py
```

修改：

```text
marl_models/utils.py
run_baseline_comparison_experiment.py
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 2 --seed 42
```

### 阶段 3：实现 Greedy-TSP baseline

新增：

```text
marl_models/greedy_tsp_baseline/__init__.py
marl_models/greedy_tsp_baseline/trajectory_planner.py
marl_models/greedy_tsp_baseline/greedy_tsp_baseline.py
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 2 --seed 42
```

### 阶段 4：接入 Vanilla MAPPO 和 Proposed

不新增模型，只接入统一脚本。

验证：

```bash
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 2 --eval_episodes 1 --seed 42
```

### 阶段 5：接入 Joint MAPPO

优先复用：

```text
run_joint_end_to_end_mappo_experiment.py
evaluate_joint_end_to_end_mappo.py
```

如输出不一致，新增转换函数，不要重写模型。

验证：

```bash
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### 阶段 6：实现 IPPO

新增：

```text
marl_models/ippo_baseline/__init__.py
marl_models/ippo_baseline/agents.py
marl_models/ippo_baseline/ippo_baseline.py
```

修改：

```text
marl_models/utils.py
run_baseline_comparison_experiment.py
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42
```

### 阶段 7：可选实现 DQN-Trajectory

只有在前面 baseline 都跑通后再做。

新增：

```text
marl_models/dqn_baseline/__init__.py
marl_models/dqn_baseline/discretization.py
marl_models/dqn_baseline/replay_buffer.py
marl_models/dqn_baseline/dqn_network.py
marl_models/dqn_baseline/dqn_baseline.py
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline dqn_trajectory --train_episodes 5 --eval_episodes 1 --seed 42
```

---

## 八、建议的 Claude Code 任务拆分

可以把下面内容直接交给 Claude Code 执行。

### Task 1：统一 baseline runner 和 metrics

目标：

1. 新增 `utils/baseline_metrics.py`。
2. 新增 `run_baseline_comparison_experiment.py`。
3. 先支持 `--baseline proposed` 和 `--baseline vanilla_mappo`。
4. 输出统一 JSON。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 2 --eval_episodes 1 --seed 42
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### Task 2：新增 Random baseline

目标：

1. 新增 `marl_models/random_baseline/`。
2. 接入 `get_model()`。
3. 接入 runner。
4. 使用 mask-aware random offload。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 3 --seed 42
```

### Task 3：新增 Greedy-TSP baseline

目标：

1. 新增最近邻轨迹规划器。
2. 新增 heuristic offload。
3. 输出轨迹 JSON。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 3 --seed 42
```

### Task 4：接入 Joint MAPPO

目标：

1. 复用已有 `joint_mappo`。
2. 统一输出字段。
3. 不重写已有 joint 模型。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### Task 5：新增 IPPO

目标：

1. 局部 actor + 局部 critic。
2. 不使用 centralized critic。
3. 复用 PPO 更新风格和现有 buffer 思路。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42
```

### Task 6：可选新增 DQN-Trajectory

目标：

1. 参数共享 Q 网络。
2. 离散轨迹动作。
3. 卸载使用 heuristic。
4. 先用小动作空间，稳定后再扩大。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline dqn_trajectory --train_episodes 5 --eval_episodes 1 --seed 42
```

---

## 九、风险与取舍

### 9.1 最大风险：DQN 动作空间不公平

DQN 只能处理离散动作，而本文轨迹是连续动作，下层卸载又是请求级离散动作。如果强行让 DQN 同时决策全部内容，会导致动作空间巨大，训练很可能不可用。

处理建议：

```text
DQN 只学习轨迹，卸载使用同一个 heuristic offload。
论文中明确写作 DQN-Trajectory baseline。
```

### 9.2 最大工作量：IPPO

IPPO 需要新增训练逻辑，但它对论文价值高，因为它能说明 CTDE 和协作训练的重要性。

处理建议：

```text
先实现局部 critic + 参数共享 actor/critic 的版本。
不要先做每个 UAV 完全独立网络，否则代码和实验管理更复杂。
```

### 9.3 最容易复用：Vanilla MAPPO 和 Joint MAPPO

这两个已经在仓库里存在，应优先接入统一评估，而不是重写。

处理建议：

```text
先跑通 existing models，再新增模型。
```

---

## 十、最终推荐优先级

如果时间充足：

```text
Random
Greedy-TSP
DQN-Trajectory
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

如果时间紧张，最低可接受主对比：

```text
Random
Greedy-TSP
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

如果再压缩，论文也仍然能成立：

```text
Greedy-TSP
Vanilla MAPPO
Joint MAPPO
Proposed
```

但建议至少保留 Random，因为它对展示“学习有效性”很直观。

---

## 十一、论文第五章建议结构

```text
5.1 实验设置
    5.1.1 仿真环境参数
    5.1.2 训练参数与评价指标
    5.1.3 对比算法说明

5.2 收敛性能分析
    对比 DQN/IPPO/Vanilla MAPPO/Joint MAPPO/Proposed 的 reward 曲线。

5.3 主性能对比
    表格展示 reward、latency、energy、fairness、DSR、offline rate、MBS load。

5.4 轨迹与卸载行为分析
    展示不同方法的 UAV 轨迹和 offloading ratio。

5.5 消融实验
    Vanilla MAPPO、no lower attention、no mask、no Lagrange、Joint MAPPO。

5.6 参数敏感性分析
    分析 UAV 数量、UE 数量或 workload 强度变化下的性能。
```

这样组织最接近参考 PDF 的论文形式，同时也能突出本文自己的贡献：attention、双层分解、mask、Lagrange 约束。
