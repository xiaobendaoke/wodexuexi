# 三项补充实验实施方案

本方案设计三项补充实验：上层 vanilla/no-attention MAPPO 消融、attention 权重可视化、完整端到端联合 MAPPO baseline，并明确需要修改的代码、实验分组和论文使用边界。

## 1. 实验一：上层 vanilla/no-attention MAPPO 对比

### 目的

证明上层 `attention-MAPPO` 的收益不是只来自 PPO 本身，而是来自 attention-based 状态编码对邻居 UAV 和覆盖 UE 信息的建模。

### 推荐结论口径

- 主比较：`attention_mappo__heuristic` vs `vanilla_mappo__heuristic`
- 辅助比较：`uncoordinated_greedy__heuristic` vs `vanilla_mappo__heuristic` vs `attention_mappo__heuristic`
- 若 `attention_mappo` 显著提升 fairness/reward，可在论文中说 attention 状态编码相对无 attention MAPPO 有增益。
- 若不显著，只说 attention-MAPPO 相对启发式有优势，vanilla 对比不支持强结论。

### 代码修改方案

新增上层无 attention 模型：

- `marl_models/vanilla_mappo/agents.py`
  - `VanillaActorNetwork`
    - 输入：单 UAV 的 `OBS_DIM_SINGLE`
    - 结构：MLP + `Normal(mean, std)` 连续动作输出
    - 输出动作维度：`ACTION_DIM=2`
  - `VanillaCriticNetwork`
    - 推荐使用 centralized state MLP：输入 `NUM_UAVS * OBS_DIM_SINGLE`
    - 输出 `NUM_UAVS` 个 value
    - 这样和 attention-MAPPO 一样属于 CTDE，但去掉 attention。

- `marl_models/vanilla_mappo/vanilla_mappo.py`
  - 接口对齐 `AttentionMAPPO`
  - 实现：
    - `select_actions(observations, exploration)`
    - `get_action_and_value(obs, state)`
    - `update(batch)`
    - `save(directory)` 保存 `vanilla_mappo.pth`
    - `load(directory)` 读取 `vanilla_mappo.pth`
  - `update()` 与 `AttentionMAPPO` 保持 PPO clipped objective 一致。
  - 区别：actor 用局部 MLP，critic 用 `batch["states"]` 的 centralized MLP，而不是 attention critic。

- `marl_models/utils.py`
  - 注册：`model_name == "vanilla_mappo"`

- `config.py`
  - `MODEL` 注释中加入 `vanilla_mappo`

- `recent/run_joint_trajectory_offload_experiment.py`
  - 无需大改，现有 `--combos LABEL:TRAJECTORY_MODEL:OFFLOAD_POLICY` 可直接支持，只要 `get_model("vanilla_mappo")` 可用。

### 实验设计

训练：

- 训练 seeds：`42, 84, 126`
- episode：建议与主实验一致 `200`
- mode：`upper_only`
- offloading：heuristic

命令模板：

```bash
python run_hierarchical_mappo_experiment.py \
  --mode upper_only \
  --trajectory_model vanilla_mappo \
  --num_episodes 200 \
  --seed 42 \
  --timestamp linear_v2_upper_vanilla_20260525_seed42_200ep
```

评估组合：

```text
uncoordinated_greedy__heuristic
vanilla_mappo__heuristic
attention_mappo__heuristic
```

评估设置：

- workload seeds：`42 84 126 168 210 252 294 336 378 420`
- 每个 workload seed：`6 episodes`
- 统计单元：`training_seed x workload_seed`，即 `N=30`
- 指标：Reward, Energy, DSR, Fairness, Offline%, Local/Coop/MBS, MBS load
- 统计：paired t-test + Wilcoxon + 95% CI

论文中新增位置：

- 第 5.2 节后增加“上层 attention 消融”小节。
- 表述重点：attention 相对 vanilla MAPPO 是否改善 fairness/reward。

## 2. 实验二：attention weight 可视化

### 目的

给 Attention-MAPPO 提供机制解释：模型在轨迹决策时是否关注邻近 UAV、覆盖 UE、deadline/priority/电池状态更关键的对象。

### 推荐结论口径

这项实验主要是解释性证据，不作为强统计性能结论。

可以写：

- attention 权重显示模型会对部分邻居 UAV 和高优先级/紧 deadline UE 分配更高关注。
- 该结果说明 attention 模块具有可解释的状态筛选行为。
- 不要写成“证明 attention 一定导致性能提升”，性能提升仍应来自实验一。

### 代码修改方案

上层 attention 权重：

- 修改 `marl_models/attention.py`
  - 为 `CrossAttentionExtractor` 增加不影响训练的权重提取方法，例如：
    - `compute_attention_weights(self_embedding, target_embeddings, mask=None)`
  - 该方法手动计算：
    - `Q = query_layer(...)`
    - `K = key_layer(...)`
    - `scores = QK^T / sqrt(d)`
    - mask padding
    - softmax
    - 返回 shape：`batch x heads x targets`
  - 不改变现有 `forward()`，避免影响训练结果。

- 修改 `AttentionActorBase`
  - 新增 `extract_attention_weights(obs_flat)`
  - 返回：
    - `neighbor_weights`
    - `ue_weights`
    - `neighbor_mask`
    - `ue_mask`
    - 可选：解析出的 `own_state / neighbor_states / ue_states`

- 修改 `marl_models/attention_mappo/attention_mappo.py`
  - 新增 `extract_attention_weights(observations)`，调用 actor 的提取方法。

下层 request attention 权重：

- 修改 `marl_models/offload_mappo/offload_mappo.py`
  - `OffloadActor` 增加 `extract_request_attention_weights(obs)`
  - 使用 `nn.MultiheadAttention(..., need_weights=True, average_attn_weights=False)`
  - 返回 request slot 间 attention，shape 约为 `batch x heads x request x request`
  - 不改变原 forward 的 `need_weights=False`，避免训练开销增加。

新增可视化脚本：

- `scripts/visualize_attention_weights.py` 或 `recent/visualize_attention_weights.py`
  - 参数：
    - `--trajectory_model_dir`
    - `--lower_model_dir` 可选
    - `--seed 42`
    - `--steps 1000`
    - `--sample_interval 20`
    - `--output_dir results/attention_visualization/...`
  - 固定 workload seed，加载 attention-MAPPO 模型，rollout 若干 step。
  - 保存：
    - `attention_samples.json`
    - `neighbor_attention_heatmap.png`
    - `ue_attention_by_priority_deadline.png`
    - 可选 `request_attention_heatmap.png`

### 可视化图建议

- 图 A：某一固定 workload 下，5 架 UAV 对邻居 UAV 的平均 attention heatmap。
- 图 B：UE attention 权重与 deadline/priority 的散点图或分组箱线图。
- 图 C：下层 request attention 对不同 request slot 的关注热力图。

论文中新增位置：

- 第 3.2 节末尾或第 5.2 节后。
- 标题可为“Attention 权重可解释性分析”。

## 3. 实验三：完整端到端联合 MAPPO baseline

### 目的

回应“为什么不直接做端到端联合 MAPPO”的质疑。该实验不是为了保证一定超过双层方法，而是提供直接 baseline 或训练困难证据。

### 关键定义

真正端到端 baseline 必须满足：

- 单一 MAPPO 模型；
- 同一策略同时输出：
  - UAV 连续轨迹动作 `N x 2`
  - 请求级离散卸载动作 `N x K`
- 不能只是把已经训练好的上层和下层模型组合评估。

### 代码修改方案

新增模型：

- `marl_models/joint_mappo/agents.py`
  - `JointActorNetwork`
    - 输入：每个 UAV 的联合观测
      - 上层 obs：`OBS_DIM_SINGLE`
      - 下层 obs：`OFFLOAD_OBS_DIM_SINGLE`
      - 拼接为 `JOINT_OBS_DIM_SINGLE`
    - 输出：
      - trajectory `Normal(mean, std)`，维度 `ACTION_DIM=2`
      - offload logits，shape `MAX_OFFLOAD_REQUESTS_PER_UAV x OFFLOAD_NUM_ACTIONS`
  - `JointCriticNetwork`
    - centralized MLP 或 attention critic
    - 输入所有 UAV 的联合观测
    - 输出每个 UAV 的 value

- `marl_models/joint_mappo/joint_mappo.py`
  - `get_action_and_value(joint_obs, masks)` 返回：
    - `trajectory_actions`
    - `offload_actions`
    - `joint_log_probs`
    - `values`
  - log prob：
    - continuous trajectory log prob
    - 加上有效 request slots 的 discrete action log prob
  - `update(batch)` 同样使用 PPO clipped objective。

新增 buffer：

- 在 `marl_models/buffer_and_helpers.py` 增加 `JointMAPPOBuffer`
  - 存：
    - joint observations
    - trajectory actions
    - offload actions
    - offload masks
    - joint log probs
    - rewards
    - values
    - dones
  - 计算 GAE 和 mini-batch。

新增训练脚本：

- 推荐新增 `run_joint_end_to_end_mappo_experiment.py`
  - 不建议硬塞进 `run_hierarchical_mappo_experiment.py`，避免破坏现有双层流程。
  - 每一步流程：
    1. `traj_obs = env._get_obs()` 或使用 reset/step 返回的 obs
    2. `offload_obs, offload_masks = env.get_offloading_obs_and_masks()`
    3. 拼接 joint obs
    4. joint actor 同时输出 trajectory 和 offload actions
    5. `env.step(traj_actions, offloading_actions=offload_actions)`
    6. 使用系统级 reward 更新单一 joint MAPPO
  - 保存：`saved_models/joint_mappo_<timestamp>/final/joint_mappo.pth`
  - summary 写入：`results/reports/joint_e2e_mappo_summary_<timestamp>.json`

新增评估脚本：

- 推荐新增 `evaluate_joint_end_to_end_mappo.py`
  - 复用 `recent/run_joint_trajectory_offload_experiment.py` 中的：
    - workload seed 逻辑
    - `Log`
    - `build_statistics`
    - `build_delta_vs_reference`
  - 但不要假装它是 trajectory + lower combo；单独输出 `joint_end_to_end_mappo`。

注册：

- `marl_models/utils.py`
  - 注册 `joint_mappo`

### 实验设计

先做 smoke test：

- `1 episode`，seed 42，检查 shape、mask、保存、评估能跑通。

再做小规模可行性：

- `20 episodes x 1 seed`
- 看 reward 是否非 NaN、fallback 是否为 0、MBS/DSR 是否有合理值。

正式实验：

- training seeds：`42, 84, 126`
- episode：建议 `200`，与主实验一致
- workload seeds：`42 84 126 168 210 252 294 336 378 420`
- 每个 workload seed：`6 episodes`
- 统计单元：`N=30`

比较对象：

```text
uncoordinated_greedy__heuristic
full_hierarchical_marl
joint_end_to_end_mappo
```

主要指标：

- Reward
- Energy
- DSR
- Fairness
- MBS load
- Offline rate
- 训练稳定性：NaN 次数、fallback 次数、reward 方差
- 参数量与单步推理耗时，可作为复杂度补充

论文使用方式：

- 若 joint E2E 明显更差：说明端到端联合动作学习存在样本效率和 credit assignment 难度。
- 若 joint E2E 接近但训练更慢：说明双层方法在可解释性和训练效率上更有优势。
- 若 joint E2E 更好：不能忽略，应如实报告，并将双层方法定位为更可解释/模块化而非性能绝对最优。
- 若训练不稳定或失败：可以作为 negative result，但必须报告训练设置和失败现象，不能只说“复杂度高”。

## 4. 推荐实施顺序

1. 先做实验一：上层 vanilla MAPPO。
   - 代码改动最小。
   - 对论文题目“Attention-MAPPO”支撑最直接。

2. 再做实验二：attention 可视化。
   - 风险低。
   - 能补机制解释图。

3. 最后做实验三：端到端联合 MAPPO。
   - 工程量最大。
   - 最容易出现 shape、mask、训练不稳定问题。
   - 建议先 smoke test，再决定是否跑正式 3 seeds。

## 5. 最终产物清单

代码产物：

- `marl_models/vanilla_mappo/`
- `marl_models/joint_mappo/`
- `run_joint_end_to_end_mappo_experiment.py`
- `evaluate_joint_end_to_end_mappo.py`
- `scripts/visualize_attention_weights.py`
- `marl_models/utils.py` 注册更新
- `config.py` 模型名注释更新

实验结果产物：

- `saved_models/vanilla_mappo_*`
- `saved_models/joint_mappo_*`
- `results/joint_experiments/*vanilla_attention_ablation*`
- `results/attention_visualization/*`
- `results/reports/joint_e2e_mappo_summary_*.json`

论文产物：

- 上层 attention 消融表
- attention 权重可视化图
- 端到端联合 MAPPO baseline 表或 negative result 描述

## 6. 风险与边界

- vanilla MAPPO 必须和 attention-MAPPO 使用相同 PPO 超参数、相同 seeds、相同 workload，否则不能做配对统计。
- attention 可视化只能解释模型关注模式，不单独证明性能提升。
- 端到端联合 MAPPO 不能复用上下层已训练模型，否则不是真正 end-to-end baseline。
- 若端到端实验训练成本过高，应至少保留 smoke test 和失败日志，论文中谨慎写为“后续工作”或“初步尝试”。
