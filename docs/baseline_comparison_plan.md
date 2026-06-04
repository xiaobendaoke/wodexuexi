# 对比算法实现与论文实验方案

## 一、方案定位

本方案用于指导后续对比算法实现、实验脚本整理和论文第五章写作。当前建议将以下论文作为**主要对比实验参考**：

```text
T. Du, X. Gui, and T. Sheng,
"Multiagent Deep Reinforcement Learning-Based Hierarchical Scheduling in Heterogeneous UAV-Enabled Vehicular Networks,"
IEEE Internet of Things Journal, vol. 12, no. 24, pp. 54938-54954, 2025.
DOI: 10.1109/JIOT.2025.3621756
```

选择这篇论文作为主参考的原因：

1. 它同样面向 UAV-enabled MEC/IoV 场景中的任务调度与卸载问题。
2. 它明确采用 dual-layer / multilayer hierarchical scheduling 叙事。
3. 它使用 MADRL、CTDE 和 MAPPO 系列方法作为主算法框架。
4. 它的 baseline 设计非常适合本文参考：general MAPPO、self-interested PPO、random policy、uniform policy。
5. 它的实验图表组织清晰，包括学习率敏感性、收敛曲线、不同 agent 数量、时延、公平性和任务数量变化。

但需要注意：该论文的“双层”是 `OU -> RU -> MeNB` 的两阶段任务传输和卸载；本文的“双层”是 `上层 UAV 轨迹控制 -> 下层请求级任务卸载`。因此本文不能直接照搬其系统结构，只应借鉴其**分层调度叙事、baseline 组织方式和实验图表逻辑**。

建议论文中这样表述：

```text
受异构 UAV 分层任务调度思想启发，本文同样采用分层决策框架。不同的是，本文面向多 UAV MEC 中轨迹控制与请求级卸载强耦合问题，将决策层划分为上层 UAV 轨迹协同控制和下层请求级任务卸载，从而降低联合动作空间复杂度并提升策略可解释性。
```

---

## 二、主参考论文与本文方案的对应关系

| 主参考论文 MAHHV | 含义 | 本文可对应的 baseline/组件 |
|------------------|------|----------------------------|
| MAHHV | MAPPO + GRU + hierarchical scheduling，本文方法 | Proposed H-Attention-MAPPO |
| GMAPPO | general MAPPO，MLP actor/critic，无 GRU | Vanilla MAPPO / General MAPPO |
| Self-Interested PPO | 每个 RU 独立 PPO，去掉 centralized cooperation | IPPO |
| Random Policy | 随机任务接收与卸载 | Random baseline |
| Uniform Policy | 均匀分配任务到计算节点 | Uniform / Load-balanced heuristic baseline |
| Different number of RUs | 不同 agent 数量敏感性 | 不同 UAV 数量敏感性 |
| Different number of vehicles | 不同任务规模敏感性 | 不同 UE 数量或 workload 强度敏感性 |
| Load fairness | 任务负载均衡 | Jain fairness / MBS load / offloading ratio |

因此，本文主 baseline 不建议再以 DQN 为核心。DQN 可作为经典 DRL 补充实验，但不是最贴合本文双层 MAPPO 贡献的主对比方法。

---

## 三、最终推荐 baseline 体系

### 3.1 主对比方法

| 编号 | 方法名称 | 类型 | 论文作用 | 实现状态 | 优先级 |
|------|----------|------|----------|----------|--------|
| 1 | Random Policy | 非学习随机策略 | 绝对下界，验证学习是否有效 | 需新增 | 高 |
| 2 | Uniform / Load-Balanced Heuristic | 非学习均衡策略 | 对齐 MAHHV 的 Uniform Policy，验证简单负载均衡效果 | 需新增 | 高 |
| 3 | Greedy-TSP + Heuristic Offload | 传统启发式 | 展示位置驱动轨迹启发式的效果 | 可选新增 | 中 |
| 4 | IPPO / Self-Interested PPO | 独立 PPO | 消融 CTDE 和多智能体协作 | 需新增 | 高 |
| 5 | Vanilla MAPPO / General MAPPO | 标准 MAPPO | 消融 Attention，类似 MAHHV 的 GMAPPO | 已存在，需接入统一实验 | 高 |
| 6 | Joint MAPPO | 结构消融 | 验证直接联合轨迹+卸载动作空间的训练难度 | 已存在，需统一评估 | 高 |
| 7 | Proposed H-Attention-MAPPO | 本文方法 | 完整双层协同方法 | 已存在 | 高 |

### 3.2 可选扩展方法

| 方法名称 | 类型 | 是否建议主表展示 | 说明 |
|----------|------|------------------|------|
| DQN-Trajectory + Heuristic Offload | 经典 DRL | 不建议放主表，除非时间充足 | 可作为附加经典 DRL baseline，但和 MAPPO 分层框架贴合度较低 |
| Uncoordinated Greedy | 已有启发式 | 可并入 Uniform/Greedy 类 | 仓库已有 `uncoordinated_greedy`，建议优先检查能否复用 |
| No lower attention / No mask / No Lagrange | 组件消融 | 放消融表，不放主对比表 | 用于证明下层 attention、mask、Lagrange 的作用 |

### 3.3 最终论文推荐主表

建议第五章主结果表包含：

```text
Random
Uniform
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

如果版面允许，再加入：

```text
Greedy-TSP
```

DQN 建议作为补充实验或附录，不建议作为主线必做项。

---

## 四、各 baseline 的详细设计

### 4.1 Random Policy

#### 论文定位

Random Policy 对齐 MAHHV 论文中的 `Random Policy`，用于提供绝对性能下界。它说明在同样环境和指标下，随机轨迹与随机卸载无法获得稳定的低时延、高公平性和高 DSR。

#### 决策逻辑

轨迹动作：

```text
对每架 UAV 随机采样二维动作 a_i = [dx, dy]。
动作范围与现有环境一致，建议采样后归一化到合法动作范围。
若动作范数过小，可视为 hover。
```

卸载动作：

```text
对每个有效请求，在合法动作集合中随机选择。
优先使用 action mask，避免随机产生大量物理不可行动作。
如果某请求无有效 mask，则 fallback 到 local 或 MBS。
```

#### 推荐新增文件

```text
marl_models/random_baseline/
├── __init__.py
└── random_baseline.py
```

#### 接口要求

```python
class RandomBaseline(MARLModel):
    def get_action(self, obs):
        """Return trajectory actions with shape (NUM_UAVS, ACTION_DIM)."""

    def get_offload_action(self, offload_obs, action_masks=None):
        """Return request-level offload actions compatible with Env.step()."""
```

#### 验证命令

```bash
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 3 --seed 42
```

---

### 4.2 Uniform / Load-Balanced Heuristic Policy

#### 论文定位

Uniform Policy 是本方案相对原计划最重要的新增 baseline。它直接对齐 MAHHV 论文中的 `Uniform Policy`，用于说明：即使不学习，只做均匀负载分配，也能改善部分公平性指标，但难以综合优化时延、能耗、DSR 和 MBS load。

#### 命名建议

论文中建议命名：

```text
Uniform Policy
```

代码中建议命名：

```text
uniform_baseline
```

如果实现加入了 latency-aware 或 mask-aware 逻辑，论文中可写：

```text
Uniform Load-Balanced Heuristic
```

#### 轨迹策略

Uniform 本身主要是卸载/分配策略。为了公平对比，需要给它一个简单、固定、可解释的轨迹策略。建议使用以下两种之一：

方案 A：固定巡航轨迹

```text
每架 UAV 在初始区域附近按圆形或网格路径巡航。
不同 UAV 分配不同巡航中心，避免重复覆盖。
```

方案 B：复用 Greedy/Uncoordinated 轨迹

```text
轨迹采用已有 uncoordinated_greedy 或 nearest-UE 移动策略。
卸载采用 uniform load balancing。
```

建议第一版采用方案 B，减少新增代码量。

#### 卸载策略

核心思想是均衡 local / cooperative UAV / MBS 的任务负载。推荐逻辑：

```text
1. 对每个有效请求，获得 mask 允许的目标集合。
2. 为每个目标维护当前 episode 或当前 time slot 的已分配任务数。
3. 在允许目标中选择当前负载最小的目标。
4. 若多个目标负载相同，优先级为 local -> cooperative UAV -> MBS。
5. 若目标估计 latency 明显超过 deadline，可跳过该目标。
```

该策略应体现“均匀分配”而不是“随机分配”。它与 Random 的区别是：Random 不考虑负载；Uniform 显式考虑负载均衡。

#### 推荐新增文件

```text
marl_models/uniform_baseline/
├── __init__.py
└── uniform_baseline.py
```

#### 验证命令

```bash
python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 3 --seed 42
```

---

### 4.3 Greedy-TSP + Heuristic Offload

#### 论文定位

Greedy-TSP 不是 MAHHV 主参考论文中的 baseline，但它适合 UAV 轨迹优化论文，用于展示传统位置驱动启发式轨迹策略的表现。

如果时间有限，该方法优先级低于 Random、Uniform、IPPO、Vanilla MAPPO 和 Joint MAPPO。

#### 轨迹策略

不实现完整 TSP 求解器，只实现 nearest-neighbor TSP heuristic：

```text
1. 每架 UAV 根据当前 UE/request 分布选择一个目标点。
2. 优先选择未覆盖、有请求、低电量或高 priority 的 UE。
3. 朝目标点移动，动作归一化到环境允许范围。
4. 使用 UAV id 或区域划分避免所有 UAV 选择同一目标。
```

#### 卸载策略

可复用 Uniform 的 mask-aware load balancing，或采用 local-first heuristic：

```text
local feasible -> local
else best cooperative UAV -> cooperative
else MBS
```

#### 推荐新增文件

```text
marl_models/greedy_tsp_baseline/
├── __init__.py
├── trajectory_planner.py
└── greedy_tsp_baseline.py
```

#### 验证命令

```bash
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 3 --seed 42
```

---

### 4.4 IPPO / Self-Interested PPO

#### 论文定位

IPPO 对齐 MAHHV 论文中的 `Self-Interested PPO`。它用于证明 centralized training 和多智能体协作的必要性。

论文中建议写法：

```text
Self-Interested PPO (IPPO): each UAV independently learns its policy using local observations and a local critic, without centralized value estimation.
```

#### 与本文方法的差异

| 项目 | IPPO | Proposed |
|------|------|----------|
| 训练范式 | independent learning | CTDE |
| Critic 输入 | local observation | global state / joint information |
| Agent 协作 | 隐式，弱协作 | centralized critic + shared reward |
| Attention | 无 | 有 |
| 决策结构 | 可复用双层流程，但各 agent 独立学习 | 双层协同 MAPPO |

#### 推荐实现策略

优先复用现有 `vanilla_mappo` 的 MLP actor/critic 结构，但将 critic 输入限制为单 agent 局部观测。

推荐新增文件：

```text
marl_models/ippo_baseline/
├── __init__.py
├── agents.py
└── ippo_baseline.py
```

#### 训练逻辑

```text
1. 每个 UAV 根据自己的局部观测 o_i 输出轨迹动作。
2. 不使用全局 state 训练 critic。
3. reward 可暂时使用共享系统 reward，以保证指标一致。
4. 每个 agent 使用本地 value 计算 GAE。
5. 使用 PPO clipped objective 更新 actor 和 local critic。
```

#### 注意事项

1. IPPO 不要使用 centralized critic，否则会变成 MAPPO。
2. 可以参数共享 actor/critic；关键是 critic 输入必须是 local observation。
3. 如果实现成本太高，可第一版只做上层 IPPO + 下层 heuristic/offload model，并在文档中说明。

#### 验证命令

```bash
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42
```

---

### 4.5 Vanilla MAPPO / General MAPPO

#### 论文定位

Vanilla MAPPO 对齐 MAHHV 论文中的 `GMAPPO`。它是本文最重要的 attention 消融，用于证明 attention 编码器对多 UAV 协同轨迹和覆盖关系建模的贡献。

论文中建议命名：

```text
General MAPPO / Vanilla MAPPO
```

#### 当前代码状态

仓库中已经存在：

```text
marl_models/vanilla_mappo/
├── __init__.py
├── agents.py
└── vanilla_mappo.py
```

`marl_models/utils.py` 中也已经支持：

```python
if model_name == "vanilla_mappo":
    return VanillaMAPPO(...)
```

#### 推荐实验设置

主对比中建议使用：

```text
Upper: vanilla_mappo
Lower: constrained_attention_offload_mappo
```

这样只消融上层 attention，其他下层机制保持与 Proposed 一致，便于解释。

下层消融另放在 ablation 表：

```text
no lower attention
no mask
no Lagrange
```

#### 验证命令

```bash
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

---

### 4.6 Joint MAPPO

#### 论文定位

Joint MAPPO 不是 MAHHV 论文中的 baseline，但它对本文非常重要。它用于证明：如果不进行上层轨迹与下层卸载分解，而直接联合建模，动作空间复杂度和训练难度会明显上升。

论文中建议定位：

```text
Structural baseline for validating hierarchical decomposition.
```

#### 当前代码状态

仓库中已经存在：

```text
marl_models/joint_mappo/
├── __init__.py
├── agents.py
└── joint_mappo.py
```

已有脚本：

```text
run_joint_end_to_end_mappo_experiment.py
evaluate_joint_end_to_end_mappo.py
```

#### 推荐处理

1. 不重写 Joint MAPPO 模型。
2. 优先检查现有 runner 是否可正常输出 reward、latency、energy、fairness、DSR、offloading ratio 等字段。
3. 若输出字段不统一，新增转换函数，而不是大改模型。
4. 在主结果表中保留 Joint MAPPO，因为它和本文“分层双层”贡献直接相关。

#### 验证命令

```bash
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

---

### 4.7 Proposed H-Attention-MAPPO

#### 论文定位

本文完整方法：

```text
Upper: attention_mappo
Lower: constrained_attention_offload_mappo
Components: attention encoder + request-level action mask + Lagrange constraints
```

论文中建议命名：

```text
H-Attention-MAPPO (Proposed)
```

或：

```text
Hierarchical Attention-MAPPO
```

#### 主对比目的

证明本文方法在以下指标上优于 baseline：

```text
reward
latency
energy
fairness
deadline satisfaction rate
offline rate
MBS load ratio
offloading distribution
```

---

### 4.8 DQN-Trajectory + Heuristic Offload（可选）

#### 论文定位

DQN 不再作为主线 baseline。它只作为经典 DRL 补充实验，用于回答：离散动作 Q-learning 类方法在连续 UAV 轨迹控制中是否存在训练和表达能力限制。

#### 推荐降级原因

1. MAHHV 主参考论文没有使用 DQN，而是使用 GMAPPO、PPO、Random、Uniform。
2. 本文核心贡献是 MAPPO/Attention/双层分解，DQN 对应关系较弱。
3. 若 DQN 同时决策轨迹和请求级卸载，动作空间会过大且不公平。

#### 如果实现，建议边界

```text
DQN 只学习上层离散化轨迹。
下层卸载复用 Uniform 或 heuristic offload。
动作空间先用 8 或 16 个方向，2 或 3 个距离档，加 hover。
```

---

## 五、统一实验脚本设计

### 5.1 推荐新增统一入口

为了避免每个 baseline 一个脚本导致维护困难，建议新增：

```text
run_baseline_comparison_experiment.py
evaluate_baseline_comparison.py
recent/run_baseline_comparison_full.sh
utils/baseline_metrics.py
```

### 5.2 命令行接口

统一使用：

```text
--baseline
--train_episodes
--eval_episodes
--seed
--workload_seed
--output_dir
```

推荐命令：

```bash
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 200 --eval_episodes 10 --seed 42
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 200 --eval_episodes 10 --seed 42
```

非学习 baseline 忽略 `--train_episodes`。

### 5.3 统一结果目录

```text
results/baseline_comparison/
├── random/
├── uniform/
├── greedy_tsp/
├── ippo/
├── vanilla_mappo/
├── joint_mappo/
└── proposed/
```

每个方法至少输出：

```text
metrics_seed_<seed>_workload_<workload_seed>.json
training_curve_seed_<seed>.json       # 非学习方法可不输出
trajectory_seed_<seed>.json           # 用于轨迹图
config_seed_<seed>.json
```

### 5.4 统一 JSON 字段

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

### 6.1 对齐 MAHHV 的图表结构

MAHHV 论文的实验结构包括：

```text
Fig. 4: learning rate sensitivity
Fig. 5: convergence of different algorithms
Fig. 6: different number of RUs
Fig. 7: latency analysis
Fig. 8: load fairness comparison
Fig. 9: different number of vehicles
Fig. 10: task load changes in one episode
```

本文可对应设计为：

| MAHHV 图表 | 本文对应图表 |
|------------|--------------|
| learning rate sensitivity | PPO learning rate 或 reward weight sensitivity |
| convergence of algorithms | IPPO / Vanilla MAPPO / Joint MAPPO / Proposed 收敛曲线 |
| different number of RUs | 不同 UAV 数量 |
| latency analysis | 平均时延、deadline satisfaction rate |
| load fairness comparison | Jain fairness、MBS load ratio |
| different number of vehicles | 不同 UE 数量或 workload 强度 |
| task load changes | offloading ratio、每 UAV/MBS 负载分布 |

### 6.2 主结果表

建议主结果表：

| Method | Reward | Latency | Energy | Fairness | DSR | Offline Rate | MBS Load |
|--------|--------|---------|--------|----------|-----|--------------|----------|
| Random | | | | | | | |
| Uniform | | | | | | | |
| IPPO | | | | | | | |
| Vanilla MAPPO | | | | | | | |
| Joint MAPPO | | | | | | | |
| Proposed | | | | | | | |

可选加入 Greedy-TSP。

每个数值建议写：

```text
mean ± 95% CI
```

### 6.3 收敛曲线

```text
x-axis: training episodes
y-axis: moving-average reward
methods: IPPO, Vanilla MAPPO, Joint MAPPO, Proposed
```

Random 和 Uniform 没有训练过程，不放入收敛曲线。

### 6.4 负载与卸载分布图

建议增加一张类似 MAHHV Fig. 10 的图：

```text
每种方法在同一 episode 中的 local / cooperative / MBS 卸载比例。
或每架 UAV 与 MBS 的任务处理数量热力图/柱状图。
```

这张图非常适合解释本文的 Lagrange 和 mask 如何影响 MBS load 与协作卸载。

### 6.5 参数敏感性图

优先选择：

```text
NUM_UAVS = 3, 5, 7
workload intensity = low, medium, high
```

如果改 `NUM_UAVS` 会牵涉观测维度和模型结构，优先做 workload intensity。

---

## 七、实现顺序

### 阶段 0：确认现有模型状态

目标：

1. 检查 `vanilla_mappo` 是否可训练。
2. 检查 `joint_mappo` 是否可训练和评估。
3. 检查 `uncoordinated_greedy` 是否能复用为 Greedy/Uniform 的轨迹部分。

验证：

```bash
python -m py_compile marl_models/utils.py
python -m py_compile run_hierarchical_mappo_experiment.py
python -m py_compile run_joint_end_to_end_mappo_experiment.py
```

### 阶段 1：统一 metrics 与 runner

新增：

```text
utils/baseline_metrics.py
run_baseline_comparison_experiment.py
```

先支持：

```text
proposed
vanilla_mappo
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 2 --eval_episodes 1 --seed 42
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### 阶段 2：实现 Random Policy

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
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 3 --seed 42
```

### 阶段 3：实现 Uniform Policy

新增：

```text
marl_models/uniform_baseline/__init__.py
marl_models/uniform_baseline/uniform_baseline.py
```

目标：

```text
mask-aware load-balanced offloading
simple shared trajectory policy
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 3 --seed 42
```

### 阶段 4：接入 Joint MAPPO

不新增模型，只统一入口和输出字段。

验证：

```bash
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### 阶段 5：实现 IPPO

新增：

```text
marl_models/ippo_baseline/__init__.py
marl_models/ippo_baseline/agents.py
marl_models/ippo_baseline/ippo_baseline.py
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42
```

### 阶段 6：可选实现 Greedy-TSP

新增：

```text
marl_models/greedy_tsp_baseline/__init__.py
marl_models/greedy_tsp_baseline/trajectory_planner.py
marl_models/greedy_tsp_baseline/greedy_tsp_baseline.py
```

验证：

```bash
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 3 --seed 42
```

### 阶段 7：可选实现 DQN-Trajectory

只有前面全部跑通后再做。

---

## 八、建议给 Claude Code 的任务拆分

### Task 1：统一 baseline runner 和 metrics

目标：

1. 新增 `utils/baseline_metrics.py`。
2. 新增 `run_baseline_comparison_experiment.py`。
3. 支持 `proposed` 和 `vanilla_mappo`。
4. 输出统一 JSON。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 2 --eval_episodes 1 --seed 42
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### Task 2：新增 Random Policy

目标：

1. 新增 `marl_models/random_baseline/`。
2. 使用 mask-aware random offload。
3. 接入统一 runner。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 3 --seed 42
```

### Task 3：新增 Uniform Policy

目标：

1. 新增 `marl_models/uniform_baseline/`。
2. 实现 mask-aware load-balanced offload。
3. 轨迹先复用简单 greedy 或固定策略。
4. 输出和其他 baseline 一致的 metrics。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 3 --seed 42
```

### Task 4：接入 Joint MAPPO

目标：

1. 复用已有 `joint_mappo`。
2. 统一结果字段。
3. 不重写模型。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42
```

### Task 5：新增 IPPO / Self-Interested PPO

目标：

1. 局部 actor + 局部 critic。
2. 不使用 centralized critic。
3. 复用 PPO buffer/update 风格。
4. 输出统一 metrics。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42
```

### Task 6：可选 Greedy-TSP

目标：

1. 最近邻轨迹规划。
2. 可复用 Uniform 或 local-first offload。
3. 输出轨迹 JSON。

完成标准：

```bash
python run_baseline_comparison_experiment.py --baseline greedy_tsp --eval_episodes 3 --seed 42
```

---

## 九、关键风险与处理

### 9.1 Uniform 的定义要清楚

Uniform 不是随机。它应该显式均衡任务负载。否则它和 Random 的实验意义会重叠。

建议定义：

```text
Uniform chooses the least-loaded feasible target among local UAV, cooperative UAVs, and MBS for each request.
```

### 9.2 IPPO 不要误写成 CTDE

IPPO/Self-Interested PPO 的关键是 decentralized training 或 local critic。它不能使用 centralized critic，否则就和 MAPPO 混淆。

### 9.3 Vanilla MAPPO 不要重写

仓库已经有 `vanilla_mappo`，只需要接入统一 runner 和评估输出。

### 9.4 DQN 不再作为主线

DQN 与 MAHHV 主参考论文不一致，也不是本文最关键的消融。除非时间充足，否则不要优先做。

### 9.5 修改 UAV 数量可能牵涉维度

如果 `NUM_UAVS` 改变会导致模型输入输出维度、checkpoint 或 buffer 复杂修改，参数敏感性优先做 workload intensity，而不是 UAV 数量。

---

## 十、最终优先级

### 必做主线

```text
Random
Uniform
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

### 建议补充

```text
Greedy-TSP
```

### 时间充足再做

```text
DQN-Trajectory
```

---

## 十一、论文第五章建议结构

```text
5.1 实验设置
    5.1.1 仿真环境参数
    5.1.2 训练参数与评价指标
    5.1.3 对比算法说明

5.2 收敛性能分析
    对比 IPPO、Vanilla MAPPO、Joint MAPPO 和 Proposed。

5.3 主性能对比
    对比 Random、Uniform、IPPO、Vanilla MAPPO、Joint MAPPO 和 Proposed。

5.4 轨迹与卸载行为分析
    展示不同方法的 UAV 轨迹、local/cooperative/MBS 卸载比例和 MBS load。

5.5 负载公平性分析
    仿照 MAHHV 的 load fairness 图，分析 Jain fairness、UAV 负载和 MBS 负载。

5.6 消融实验
    分析 no upper attention、no lower attention、no mask、no Lagrange、Joint MAPPO。

5.7 参数敏感性分析
    分析 workload intensity、UE 数量或 UAV 数量变化下的性能。
```

该结构同时满足两个目标：一方面借鉴 MAHHV 论文的分层调度和 baseline 设计，另一方面保留本文自己的创新叙事：轨迹-卸载双层分解、attention 建模、质量感知 mask 和 Lagrange 约束。
