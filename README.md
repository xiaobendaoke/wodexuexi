# Quality-aware Constrained Hierarchical MARL for Multi-UAV MEC

这个仓库讲的是一个很明确的故事：

> 多无人机 MEC 系统里，服务质量不是只靠 UAV 飞得好就能解决，也不是只靠请求卸载策略聪明就够了。UAV 轨迹决定链路、覆盖、邻居协作机会和 MBS 回传条件；请求级卸载又反过来决定时延、能耗、deadline satisfaction 和 MBS 负载。直接把轨迹和所有请求卸载揉成一个巨大联合动作空间很难训练、也很难解释。因此，本项目把问题拆成上下两层 MARL：上层管 UAV 轨迹，下层管服务请求卸载。

本文主方法是 **Quality-aware Constrained Hierarchical MARL**：

- 上层：`attention_mappo` 控制多 UAV 轨迹。
- 下层：`constrained_attention_offload_mappo` 对每个服务请求选择 `local UAV / cooperative UAV / MBS`。
- 质量感知动作 mask：避免明显不可行的协作 UAV 或过慢 MBS 动作。
- Lagrange 约束：显式控制 deadline satisfaction 和 MBS load 的权衡。
- `oracle_guided` 不再是主方法，只作为 teacher/reference baseline。

## 为什么这样拆

在多 UAV MEC 场景里，轨迹控制和请求卸载是强耦合的：

- UAV 位置影响 UE-UAV 链路速率。
- UAV 位置影响是否存在合适的协作 UAV。
- UAV-MBS 回传质量影响 MBS fallback 是否值得用。
- 卸载决策影响请求时延、UAV 队列压力、能耗和 MBS 负载。

如果上层策略只优化轨迹，系统可能覆盖得很好，但服务请求仍然大量回落到 MBS。  
如果下层策略只优化卸载，它只能被动适应当前 UAV 分布，无法创造更好的协作机会。  
所以这里采用双层结构：**上层塑造服务环境，下层在当前环境里做请求级质量控制**。

## 方法结构

每个环境 step 的流程是：

1. 环境生成当前 UAV/UE/request 状态。
2. 下层调用 `env.get_offloading_obs_and_masks()` 得到请求级观测和动作 mask。
3. 下层 MAPPO 为每架 UAV 的服务请求 slot 输出离散卸载动作。
4. 环境执行请求处理，统计 latency、energy、DSR、MBS load 等指标。
5. 上层 attention-MAPPO 输出 UAV 轨迹动作，更新下一时刻位置。

下层动作空间固定为：

| 动作 | 含义 |
| --- | --- |
| `0` | local UAV 执行 |
| `1` | cooperative UAV 执行 |
| `2` | MBS 执行 |

下层消融设计：

| 方法 | Attention | Quality mask | Lagrange |
| --- | --- | --- | --- |
| `full` | yes | yes | yes |
| `no_mask` | yes | no | yes |
| `no_lagrange` | yes | yes | no |
| `no_attention` | no | yes | yes |

## 仓库结构

```text
.
├── run_hierarchical_mappo_experiment.py      # 上下层 MAPPO 训练入口
├── run_joint_trajectory_offload_experiment.py # 联合评估入口
├── analyze_experiment_statistics.py          # paired unit 统计分析
├── config.py                                 # 环境、奖励、约束与 PPO 配置
├── environment/                              # 多 UAV MEC 仿真环境
├── marl_models/
│   ├── attention_mappo/                      # 上层 attention-MAPPO
│   ├── offload_mappo/                        # 下层 constrained offload MAPPO
│   ├── uncoordinated_greedy_baseline/        # 必要轨迹基线
│   ├── attention.py                          # attention 模块
│   ├── buffer_and_helpers.py                 # PPO buffer 与工具
│   ├── offload_policy.py                     # oracle-guided teacher/reference 加载接口
│   └── utils.py                              # 精简后的模型工厂
├── utils/                                    # 日志、绘图和对比工具
└── docs/
    ├── PAPER_DRAFT.md                        # 论文草稿
    ├── doc_attention.md
    └── system_model.jpg
```

仓库已经清理掉旧的 MADDPG / MATD3 / MASAC / classifier training pipeline / 历史实验结果，只保留双层 MAPPO 主线和必要基线。

## 训练

### 完整双层 MARL

```powershell
python run_hierarchical_mappo_experiment.py `
  --mode full_hierarchical `
  --lower_ablation full `
  --seed 42 `
  --num_episodes 50 `
  --timestamp full_hmarl_seed42
```

### 只训练上层

```powershell
python run_hierarchical_mappo_experiment.py `
  --mode upper_only `
  --trajectory_model attention_mappo `
  --seed 42 `
  --num_episodes 50 `
  --timestamp upper_attention_seed42
```

### 固定上层训练下层

```powershell
python run_hierarchical_mappo_experiment.py `
  --mode lower_only_fixed_upper `
  --trajectory_model attention_mappo `
  --trajectory_model_dir saved_models/attention_mappo_xxx/final `
  --lower_ablation full `
  --seed 42 `
  --num_episodes 50 `
  --timestamp lower_attention_seed42
```

下层消融通过 `--lower_ablation` 切换：

```powershell
--lower_ablation full
--lower_ablation no_mask
--lower_ablation no_lagrange
--lower_ablation no_attention
```

## 评估

主表建议比较六组：

| 组合 | 作用 |
| --- | --- |
| `uncoordinated_greedy + heuristic` | 基础参考 |
| `attention_mappo + heuristic` | 只看上层 attention-MAPPO 贡献 |
| `attention_mappo + oracle_guided` | teacher/reference baseline |
| `uncoordinated_greedy + lower_mappo` | 下层 MARL 在弱上层分布下的表现 |
| `attention_mappo + lower_mappo` | 分别训练的上下层组合 |
| `full_hierarchical_marl` | 主方法 |

示例：

```powershell
python run_joint_trajectory_offload_experiment.py `
  --hmarl_main_table `
  --trajectory_run_root . `
  --training_seeds 42 84 126 `
  --seeds 42 84 126 168 210 252 294 336 378 420 `
  --episodes_per_seed 6 `
  --steps_per_episode 1000 `
  --lower_model_dirs `
    uncoordinated_greedy__lower_mappo@42=saved_models/offload_mappo_lower_uncoord_seed42/final `
    attention_mappo__lower_mappo@42=saved_models/offload_mappo_lower_attention_seed42/final `
    full_hierarchical_marl@42=saved_models/offload_mappo_full_hmarl_seed42/final
```

如果要评估 `oracle_guided` teacher/reference，需要显式提供外部 checkpoint：

```powershell
--surrogate_checkpoint path/to/offload_policy_surrogate_runtime.pt
```

本仓库不再内置旧 classifier checkpoint。

## 统计口径

推荐实验设置：

- training seeds：`42, 84, 126`
- workload seeds：`42, 84, 126, 168, 210, 252, 294, 336, 378, 420`
- 每个 workload seed：`6` episodes
- 每个 episode：`1000` steps

统计单位是：

```text
(training_seed, workload_seed) 的 episode mean
```

报告指标：

- reward
- latency
- energy
- deadline satisfaction rate
- fairness
- offline rate
- local / cooperative / MBS ratio
- MBS load ratio
- lower actor loss、critic loss、entropy
- `lambda_dsr`、`lambda_mbs`
- constraint penalty
- cooperative/MBS masked count

统计脚本：

```powershell
python analyze_experiment_statistics.py `
  results/joint_experiments/<exp_name>/joint_experiment_summary.json `
  --reference uncoordinated_greedy__heuristic `
  --output_md results/joint_experiments/<exp_name>/statistics.md
```

## 这篇工作的论文故事

这项工作不是声称“第一次联合优化轨迹和卸载”。已有工作已经大量讨论 UAV-MEC 中的轨迹、卸载、资源分配、缓存和服务放置联合优化。

这个仓库的核心定位是：

1. **用双层 MARL 拆解复杂动作空间**  
   上层只处理 UAV 轨迹，下层只处理请求级卸载，避免端到端联合动作爆炸。

2. **让下层卸载从 classifier 回到纯 RL**  
   旧的 oracle-guided 分类器降级为 teacher/reference，主方法改为 constrained attention offload MAPPO。

3. **把服务质量写进动作可行性和约束里**  
   quality-aware mask 负责屏蔽坏动作，Lagrange 约束负责调节 DSR 与 MBS load。

4. **用成对统计证明贡献来自哪里**  
   主表拆分上层、下层、teacher/reference 和 full hierarchical MARL；消融表拆分 mask、Lagrange 和 attention。

一句话版本：

> 本文提出一种面向多 UAV MEC 的质量感知约束式双层 MARL 框架：上层 attention-MAPPO 学习 UAV 协同轨迹，下层 constrained attention offload MAPPO 学习请求级卸载，并通过质量 mask 与 Lagrange 约束控制 deadline satisfaction 和 MBS load 的权衡。

## 当前状态

- 代码已精简到双层 MAPPO 主线。
- 旧实验结果、旧 checkpoint、旧 dataset 已删除。
- `docs/PAPER_DRAFT.md` 保留论文草稿。
- 后续需要重新跑主表和消融表，生成新的 results。
