# Baseline 对比实验方案实施计划

## 概述

按照 `docs/baseline_comparison_plan.md` 方案，为论文第五章构建统一 baseline 对比实验体系。核心目标：新增 `Random`、`Uniform`、`IPPO` 三个 baseline，并统一现有 `proposed`、`vanilla_mappo`、`joint_mappo` 的评估入口。

主对比方法：

```text
Random
Uniform
IPPO / Self-Interested PPO
Vanilla MAPPO / General MAPPO
Joint MAPPO
Proposed H-Attention-MAPPO
```

关键原则：

1. 不重写已有 `attention_mappo`、`vanilla_mappo`、`joint_mappo`、`offload_mappo` 模型。
2. `proposed` 和 `vanilla_mappo` 训练逻辑优先复用 `train_hierarchical_mappo()`。
3. `joint_mappo` 训练逻辑优先复用 `run_joint_end_to_end_mappo_experiment.py` 中已有流程。
4. Random/Uniform 不学习，只实现轨迹动作和 request-level offload 动作选择。
5. 所有方法输出统一 metrics JSON，并额外输出 trajectory JSON，方便后续画图。

---

## 阶段 0：确认现有模型状态

**目标**：验证现有模型和脚本可正常编译。

**验证命令**：

```bash
python -m py_compile marl_models/utils.py
python -m py_compile run_hierarchical_mappo_experiment.py
python -m py_compile run_joint_end_to_end_mappo_experiment.py
python -m py_compile evaluate_joint_end_to_end_mappo.py
```

**预期**：全部通过。

---

## 阶段 1：统一 baseline metrics 与 runner

### Task 1.1：新增 `utils/baseline_metrics.py`

**职责**：提供统一评估、聚合和 JSON 输出工具。

**关键接口**：

```python
SUMMARY_METRIC_NAMES = [
    "reward",
    "latency",
    "energy",
    "fairness",
    "offline_rate",
    "deadline_satisfaction_rate",
    "offloading_ratio_local",
    "offloading_ratio_cooperative",
    "offloading_ratio_mbs",
    "mbs_load_ratio",
    "service_offloads_local",
    "service_offloads_cooperative",
    "service_offloads_mbs",
]


def run_single_episode(
    env,
    trajectory_model=None,
    offload_model=None,
    joint_model=None,
    baseline_policy=None,
    exploration=False,
    record_trajectory=False,
) -> tuple[dict[str, float], dict[str, object]]:
    """Run one eval episode and optionally return trajectory records."""


def aggregate_metric_dicts(entries: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return mean/std/95% CI for each metric."""


def save_baseline_metrics(method_name, seed, workload_seed, episode_metrics, output_dir):
    """Save raw episode metrics and aggregated metrics."""


def save_trajectory_json(method_name, seed, workload_seed, trajectory_records, output_dir):
    """Save trajectory records for chapter-5 trajectory plots."""
```

**参考实现**：

```text
evaluate_joint_end_to_end_mappo.py 中的 run_single_episode 和 aggregate_metric_dicts
environment/env.py 中 metrics 字段
```

### Task 1.2：新增 `run_baseline_comparison_experiment.py`

**职责**：统一 baseline 训练/评估入口。

**命令行接口**：

```bash
python run_baseline_comparison_experiment.py \
    --baseline <method> \
    --train_episodes <N> \
    --eval_episodes <N> \
    --seed <S> \
    --workload_seed <W> \
    --output_dir results/baseline_comparison
```

**支持的 baseline**：

```text
proposed       -> attention_mappo + constrained_attention_offload_mappo
vanilla_mappo  -> vanilla_mappo + constrained_attention_offload_mappo
joint_mappo    -> joint_mappo
random         -> RandomBaseline
uniform        -> UniformBaseline
ippo           -> IPPOBaseline
```

**重要实现约束**：

1. `proposed` 不要重写训练循环，应调用：

```python
train_hierarchical_mappo(
    num_episodes=args.train_episodes,
    seed=args.seed,
    trajectory_model_name="attention_mappo",
    lower_ablation="full",
)
```

2. `vanilla_mappo` 不要重写训练循环，应调用：

```python
train_hierarchical_mappo(
    num_episodes=args.train_episodes,
    seed=args.seed,
    trajectory_model_name="vanilla_mappo",
    lower_ablation="full",
)
```

3. `joint_mappo` 优先复用现有 joint runner 的训练/评估函数。如果原脚本没有可直接 import 的函数，再做最小封装，不重写模型。

4. Random/Uniform 不需要训练；`--train_episodes` 对它们忽略。

**输出结构**：

```text
results/baseline_comparison/<method>/
├── metrics_seed_<S>_workload_<W>.json
├── training_curve_seed_<S>.json          # 非学习方法可不输出
├── trajectory_seed_<S>_workload_<W>.json # 用于轨迹图
└── config_seed_<S>_workload_<W>.json
```

---

## 阶段 2：实现非学习 baseline 的统一接口

### Task 2.1：新增 `marl_models/baseline_policy_utils.py`

**目的**：避免 Random/Uniform 使用和现有 `OffloadMAPPO` 不一致的接口。

**推荐接口**：

```python
def valid_slot(mask_row: np.ndarray) -> bool:
    return bool(np.sum(mask_row) > 0.0)


def default_offload_action() -> int:
    return int(config.OFFLOAD_ACTION_LOCAL)


def legal_actions_from_mask(mask_row: np.ndarray) -> np.ndarray:
    legal = np.where(mask_row > 0.0)[0]
    return legal.astype(np.int64)


def select_random_offload_actions(action_masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return shape: (NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV)."""


def select_uniform_load_balanced_offload_actions(action_masks: np.ndarray) -> np.ndarray:
    """Return shape: (NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV)."""
```

**必须处理空 mask**：

```python
if len(legal) == 0:
    offload_actions[uav_idx, req_idx] = config.OFFLOAD_ACTION_LOCAL
    continue
```

原因：`Env.get_offloading_obs_and_masks()` 对 padded request slots 使用全 0 mask。不能对空数组调用 `np.random.choice()` 或 `np.argmin()`。

### Task 2.2：非学习 baseline offload 调用方式

Random/Uniform 可以实现以下两种之一，推荐方案 A。

**方案 A：实现 baseline 自己的 offload 方法，由 runner 调用**

```python
class RandomBaseline(MARLModel):
    def select_actions(self, observations: np.ndarray, exploration: bool = False) -> np.ndarray:
        ...

    def select_offload_actions(self, offload_obs: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        return select_random_offload_actions(action_masks, self.rng)
```

runner 中：

```python
offload_obs, offload_masks = env.get_offloading_obs_and_masks()
offload_actions = baseline_policy.select_offload_actions(offload_obs, offload_masks)
next_obs, rewards, metrics = env.step(traj_actions, offloading_actions=offload_actions)
```

**方案 B：兼容 `OffloadMAPPO.get_action_and_value()` 形式**

```python
def get_action_and_value(self, obs, state=None, masks=None, exploration=False):
    actions = self.select_offload_actions(obs, masks)
    log_probs = np.zeros((self.num_agents,), dtype=np.float32)
    values = np.zeros((self.num_agents,), dtype=np.float32)
    return actions, log_probs, values
```

如果采用方案 B，仍然要保留 `select_offload_actions()`，便于阅读和测试。

---

## 阶段 3：实现 Random Policy

### Task 3.1：新增 `marl_models/random_baseline/`

**文件结构**：

```text
marl_models/random_baseline/
├── __init__.py
└── random_baseline.py
```

**类设计**：`RandomBaseline(MARLModel)`

**关键方法**：

```python
class RandomBaseline(MARLModel):
    def __init__(...):
        super().__init__(...)
        self.rng = np.random.default_rng()

    def select_actions(self, observations: np.ndarray, exploration: bool = False) -> np.ndarray:
        actions = self.rng.uniform(-1.0, 1.0, size=(self.num_agents, self.action_dim))
        return actions.astype(np.float32)

    def select_offload_actions(self, offload_obs: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        return select_random_offload_actions(action_masks, self.rng)

    def update(self, batch):
        return {}

    def reset(self):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass
```

**注册到 `marl_models/utils.py`**：

```python
if model_name == "random_baseline":
    from marl_models.random_baseline.random_baseline import RandomBaseline
    return RandomBaseline(
        model_name=model_name,
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device=device,
    )
```

**验证命令**：

```bash
python -m py_compile marl_models/baseline_policy_utils.py
python -m py_compile marl_models/random_baseline/random_baseline.py
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 1 --seed 42 --workload_seed 1001
```

---

## 阶段 4：实现 Uniform / Load-Balanced Policy

### Task 4.1：新增 `marl_models/uniform_baseline/`

**文件结构**：

```text
marl_models/uniform_baseline/
├── __init__.py
└── uniform_baseline.py
```

**类设计**：`UniformBaseline(MARLModel)`

### 轨迹策略

第一版使用现有 `UncoordinatedGreedyModel` 的中心飞行策略，命名时不要称为真正 TSP 或 coverage-aware greedy。

论文/代码注释建议称为：

```text
center-seeking fixed trajectory + load-balanced offloading
```

轨迹代码可复用：

```python
self.target_pos = np.array([[config.AREA_WIDTH / 2.0, config.AREA_HEIGHT / 2.0]], dtype=np.float32)
self.area_dims = np.array([[config.AREA_WIDTH, config.AREA_HEIGHT]], dtype=np.float32)
```

```python
def select_actions(self, observations, exploration=False):
    current_pos = observations[:, :2] * self.area_dims
    delta = self.target_pos - current_pos
    distances = np.linalg.norm(delta, axis=1, keepdims=True)
    actions = self.rng.uniform(-0.1, 0.1, size=(self.num_agents, self.action_dim)).astype(np.float32)
    mask = (distances > 10.0).flatten()
    if np.any(mask):
        actions[mask] = delta[mask] / np.maximum(distances[mask], config.EPSILON)
    return np.clip(actions, -1.0, 1.0).astype(np.float32)
```

### 卸载策略

核心：mask-aware load-balanced，而不是 random。

**必须实现显式 tie-break**：

```text
负载最小时按 local -> cooperative UAV -> MBS 优先。
```

不能直接用 `legal[np.argmin(load_counts[legal])]`，因为 action index 顺序可能是 local=0, MBS=1, cooperative=2...，这会导致 tie 时优先 MBS。

推荐 helper：

```python
def offload_priority(action: int) -> tuple[int, int]:
    if action == config.OFFLOAD_ACTION_LOCAL:
        return (0, action)
    if action >= config.OFFLOAD_ACTION_COOPERATIVE_START:
        return (1, action)
    if action == config.OFFLOAD_ACTION_MBS:
        return (2, action)
    return (3, action)
```

选择逻辑：

```python
candidates = sorted(
    legal,
    key=lambda a: (load_counts[int(a)], offload_priority(int(a))),
)
chosen = int(candidates[0])
```

**空 mask 处理**：

```python
if len(legal) == 0:
    offload_actions[uav_idx, req_idx] = config.OFFLOAD_ACTION_LOCAL
    continue
```

**注册到 `marl_models/utils.py`**：

```python
if model_name == "uniform_baseline":
    from marl_models.uniform_baseline.uniform_baseline import UniformBaseline
    return UniformBaseline(...)
```

**验证命令**：

```bash
python -m py_compile marl_models/uniform_baseline/uniform_baseline.py
python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 1 --seed 42 --workload_seed 1001
```

---

## 阶段 5：接入 Proposed 和 Vanilla MAPPO

### Task 5.1：统一 runner 中支持 `proposed`

**必须复用已有函数**：

```python
from run_hierarchical_mappo_experiment import train_hierarchical_mappo
```

训练：

```python
result = train_hierarchical_mappo(
    num_episodes=args.train_episodes,
    seed=args.seed,
    trajectory_model_name="attention_mappo",
    lower_ablation="full",
)
```

评估：

```text
加载 result["trajectory_model_dir"] 和 result["offload_model_dir"]，使用统一 run_single_episode()。
```

如果 `train_episodes <= 0`，允许直接用随机初始化模型评估，但输出中必须标注 `trained=false`。

### Task 5.2：统一 runner 中支持 `vanilla_mappo`

训练：

```python
result = train_hierarchical_mappo(
    num_episodes=args.train_episodes,
    seed=args.seed,
    trajectory_model_name="vanilla_mappo",
    lower_ablation="full",
)
```

其他逻辑与 proposed 一致。

**验证命令**：

```bash
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
```

---

## 阶段 6：接入 Joint MAPPO

### Task 6.1：统一 runner 中支持 `joint_mappo`

**实现要点**：

1. 复用已有 `JointMAPPO` 模型。
2. 优先复用 `run_joint_end_to_end_mappo_experiment.py` 中的训练逻辑。
3. 如果原脚本缺少可 import 函数，可做最小封装，例如新增 `train_joint_mappo(...)`，不要重写模型。
4. 评估时使用 `joint_model.get_action_and_value(joint_obs, masks=offload_masks, exploration=False)`，并传入：

```python
env.step(trajectory_actions, offloading_actions=offload_actions)
```

5. 输出统一 JSON。

**验证命令**：

```bash
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
```

---

## 阶段 7：实现 IPPO / Self-Interested PPO

### Task 7.1：新增 `marl_models/ippo_baseline/`

**文件结构**：

```text
marl_models/ippo_baseline/
├── __init__.py
├── agents.py
└── ippo_baseline.py
```

**类设计**：`IPPOBaseline(MARLModel)`

**关键区别**：

| 项目 | IPPO | Vanilla MAPPO |
|------|------|---------------|
| Critic 输入 | 单 agent 局部观测 `(batch, obs_dim)` | 全局 state `(batch, N*obs_dim)` |
| 训练范式 | independent learning | CTDE |
| Attention | 无 | 无 |
| Actor | 可复用 MLP actor | MLP actor |

### Actor/Critic

Actor 可复用 `VanillaActorNetwork`。

Critic 必须是 local critic：

```python
class IPPOCriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.critic(obs).squeeze(-1)
```

### Buffer/update 约束

现有 `AttentionRolloutBuffer` 可以复用，但 `IPPOBaseline.update()` 必须忽略 `batch["states"]`，只使用 `batch["obs"]` 训练 critic。

推荐 update 逻辑：

```python
flat_obs = obs_batch.view(-1, self.obs_dim)
flat_actions = actions_batch.view(-1, self.action_dim)
flat_old_log_probs = old_log_probs_batch.view(-1)
flat_advantages = advantages_batch.view(-1)
flat_returns = returns_batch.view(-1)
flat_old_values = old_values_batch.view(-1)

values = self.critic(flat_obs)
```

`get_action_and_value(obs, state)` 中可以接受 `state` 参数以兼容 runner，但不要用于 critic：

```python
def get_action_and_value(self, obs, state=None):
    obs_tensor = torch.from_numpy(obs).float().to(self.device)
    dist = self.actor(obs_tensor)
    actions = dist.sample()
    log_probs = dist.log_prob(actions).sum(dim=-1)
    values = self.critic(obs_tensor)
    return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
```

这样可以继续复用 `train_hierarchical_mappo()` 的上层 PPO buffer/update 结构，只要 `get_model("ippo_baseline")` 返回该模型，并在 runner 中调用：

```python
train_hierarchical_mappo(
    num_episodes=args.train_episodes,
    seed=args.seed,
    trajectory_model_name="ippo_baseline",
    lower_ablation="full",
)
```

**注册到 `marl_models/utils.py`**：

```python
if model_name == "ippo_baseline":
    from marl_models.ippo_baseline.ippo_baseline import IPPOBaseline
    return IPPOBaseline(...)
```

**验证命令**：

```bash
python -m py_compile marl_models/ippo_baseline/agents.py
python -m py_compile marl_models/ippo_baseline/ippo_baseline.py
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
```

---

## 阶段 8（可选）：Greedy-TSP

如果时间允许，实现：

```text
marl_models/greedy_tsp_baseline/
├── __init__.py
├── trajectory_planner.py
└── greedy_tsp_baseline.py
```

轨迹策略：nearest-neighbor heuristic，优先选择未覆盖/有请求/低电量 UE。

卸载策略：复用 Uniform 的 load-balanced 逻辑。

该阶段不是主线，优先级低于 Random、Uniform、IPPO、Vanilla MAPPO、Joint MAPPO、Proposed。

---

## 文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `utils/baseline_metrics.py` | 统一指标收集、聚合和 JSON 输出 |
| `run_baseline_comparison_experiment.py` | 统一 baseline 训练/评估入口 |
| `marl_models/baseline_policy_utils.py` | Random/Uniform offload action helper |
| `marl_models/random_baseline/__init__.py` | Random baseline 包 |
| `marl_models/random_baseline/random_baseline.py` | Random baseline 实现 |
| `marl_models/uniform_baseline/__init__.py` | Uniform baseline 包 |
| `marl_models/uniform_baseline/uniform_baseline.py` | Uniform baseline 实现 |
| `marl_models/ippo_baseline/__init__.py` | IPPO baseline 包 |
| `marl_models/ippo_baseline/agents.py` | IPPO Actor/Critic 网络 |
| `marl_models/ippo_baseline/ippo_baseline.py` | IPPO baseline 实现 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `marl_models/utils.py` | 在 `get_model()` 中注册 random/uniform/ippo |
| `run_joint_end_to_end_mappo_experiment.py` | 如有必要，抽出可 import 的 `train_joint_mappo()`，不改变原 CLI 行为 |

---

## 总体验证命令

```bash
# 编译检查
python -m py_compile marl_models/utils.py
python -m py_compile utils/baseline_metrics.py
python -m py_compile run_baseline_comparison_experiment.py
python -m py_compile marl_models/baseline_policy_utils.py
python -m py_compile marl_models/random_baseline/random_baseline.py
python -m py_compile marl_models/uniform_baseline/uniform_baseline.py
python -m py_compile marl_models/ippo_baseline/agents.py
python -m py_compile marl_models/ippo_baseline/ippo_baseline.py

# 非学习 baseline smoke test
python run_baseline_comparison_experiment.py --baseline random --eval_episodes 1 --seed 42 --workload_seed 1001
python run_baseline_comparison_experiment.py --baseline uniform --eval_episodes 1 --seed 42 --workload_seed 1001

# 现有学习 baseline smoke test
python run_baseline_comparison_experiment.py --baseline proposed --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
python run_baseline_comparison_experiment.py --baseline vanilla_mappo --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
python run_baseline_comparison_experiment.py --baseline joint_mappo --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001

# 新学习 baseline smoke test
python run_baseline_comparison_experiment.py --baseline ippo --train_episodes 2 --eval_episodes 1 --seed 42 --workload_seed 1001
```

---

## 实施顺序

1. 阶段 1：统一 metrics 和 runner，先支持 `proposed`、`vanilla_mappo`。
2. 阶段 2：新增非学习 offload helper，处理空 mask 和 tie-break。
3. 阶段 3：实现 Random。
4. 阶段 4：实现 Uniform。
5. 阶段 5：确认 proposed/vanilla 复用 `train_hierarchical_mappo()`。
6. 阶段 6：接入 Joint MAPPO。
7. 阶段 7：实现 IPPO。
8. 阶段 8：可选 Greedy-TSP。

---

## 关键设计约束

1. 所有 baseline 必须兼容 `MARLModel.select_actions()`。
2. Random/Uniform 必须额外提供 `select_offload_actions(offload_obs, action_masks)`，或者兼容 `get_action_and_value(..., masks=...)`。
3. `action_masks` 的 shape 是 `(NUM_UAVS, MAX_OFFLOAD_REQUESTS_PER_UAV, OFFLOAD_NUM_ACTIONS)`。
4. padded request slot 的 mask 全 0，必须跳过或返回默认 local action。
5. Uniform 必须显式负载均衡，不能退化成 random。
6. Uniform tie-break 必须是 local -> cooperative UAV -> MBS。
7. IPPO critic 必须只使用局部观测，不能使用全局 state。
8. `proposed`、`vanilla_mappo`、`joint_mappo` 必须复用已有训练逻辑，避免重复实现 PPO。
9. 所有方法输出统一 metrics JSON。
10. 所有评估输出 trajectory JSON，方便后续生成轨迹图。
