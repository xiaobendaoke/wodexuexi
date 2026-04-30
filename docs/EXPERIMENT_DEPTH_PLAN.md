# 补充实验执行方案

这份方案对应三个目标：证明 SC-OGO 的机制来源，验证尺度泛化，补上复杂度与部署分析。当前脚本先做运行时消融，也就是固定已训练的 surrogate checkpoint，只改变 SC-OGO 在线重排序项。

## 1. SC-OGO 组件消融

运行：

```powershell
.\.venv\Scripts\python.exe run_sc_ogo_ablation.py `
  --surrogate_checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/sc_ogo_component_ablation.json `
  --seeds 42 84 126 168 210 `
  --episodes_per_seed 4 `
  --steps_per_episode 100
```

消融口径：

| 版本 | 代码口径 | 论文中回答的问题 |
| --- | --- | --- |
| `classifier_only` | `SERVICE_OFFLOAD_POLICY=learned` | 没有 SC-OGO 重排序时 surrogate 本身能做到什么 |
| `sc_ogo_full` | 默认 SC-OGO | 完整安全约束重排序效果 |
| `sc_ogo_no_mbs_penalty` | `SC_OGO_MBS_WEIGHT=0` | MBS 压缩是否来自显式 fallback penalty |
| `sc_ogo_no_deadline_margin` | 关闭 margin 与 hard deadline switch | DSR 安全性是否来自 deadline safety margin |
| `sc_ogo_no_queue_pressure` | `SC_OGO_QUEUE_WEIGHT=0` | 队列压力感知是否有贡献 |
| `sc_ogo_no_coop_term` | `SC_OGO_COOP_WEIGHT=0` | 协作路径调节项是否有贡献 |

读结果时重点看：

- `delta_vs_classifier_only`：证明 SC-OGO 重排序的增量价值。
- `delta_vs_sc_ogo_full`：证明去掉某一项后指标如何变化。
- 如果 `no_mbs_penalty` 让 `mbs_load_ratio` 或 `offloading_ratio_mbs` 上升，说明 MBS 压缩确实来自显式 MBS fallback 代价。
- 如果 `no_deadline_margin` 让 DSR 下降或 DSR 波动变大，说明 safety margin 对 deadline 安全性有贡献。
- 如果 `no_queue_pressure` 改变 local/coop/MBS 结构，说明队列感知在改变请求流向。

更强版本：训练期 oracle 消融。这个需要重新采集数据和训练 checkpoint，例如把 `OFFLOAD_ORACLE_MBS_LOAD_WEIGHT`、`OFFLOAD_ORACLE_QUEUE_WEIGHT` 或 `OFFLOAD_ORACLE_COOP_QUEUE_RELIEF_BONUS` 置零后分别采集数据、训练分类器、再跑同样的运行时对比。运行时消融适合先产机制证据，训练期消融适合顶刊增强。

## 2. 尺度泛化实验

运行：

```powershell
.\.venv\Scripts\python.exe run_scale_generalization.py `
  --surrogate_checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/scale_generalization_runtime.json `
  --seeds 42 84 126 168 `
  --episodes_per_seed 3 `
  --steps_per_episode 100
```

默认是一因子扫描：

| 因子 | 默认取值 |
| --- | --- |
| UAV 数量 | `3, 5, 7` |
| UE 数量 | `50, 100, 150` |
| 热点数量 | `1, 2, 3, 4` |
| backhaul bandwidth | `120 kHz, 350 kHz, 750 kHz, 1.5 MHz, 5 MHz` |

这个脚本使用静态 UAV 位置和零移动动作，主要隔离下层卸载策略的尺度泛化。若要验证完整 `attention_mappo + sc_ogo` 的尺度泛化，改变 UAV 数量时通常需要为每个 UAV 数量重新训练或准备兼容的上层模型。

论文中建议报告：

- 每个尺度下 `SC-OGO - heuristic` 的 DSR、MBS load、MBS ratio、energy delta。
- 横轴用 UAV/UE/hotspot/backhaul，纵轴用 MBS load delta 与 DSR delta。
- 最好画成两张图：一张 MBS load，一张 DSR；或者做 DSR-MBS load Pareto-style scatter。

## 3. 复杂度与部署分析

运行：

```powershell
.\.venv\Scripts\python.exe analyze_complexity_deployment.py `
  --checkpoint saved_offload_policies/offload_policy_surrogate_runtime.pt `
  --output results/reports/complexity_deployment_analysis.md `
  --json_output results/reports/complexity_deployment_analysis.json
```

论文中建议这样写：

> 上层策略每个 time slot 只输出 `2U` 维连续运动动作。在默认 `U=5` 时，上层动作维度为 `10`。请求级卸载不并入上层动作，而是在服务请求发生时由下层策略输出 `local/cooperative/MBS` 三分类结果。因此，若一个 time slot 内存在 `R_t` 个服务请求，单层枚举式请求卸载会产生 `3^{R_t}` 种请求去向组合，而本文框架保持上层动作接口固定，并将请求相关离散决策转化为每请求一次轻量推理。

注意不要写成“全局最优复杂度最低”。更稳妥的说法是：本文降低了端到端动作空间随请求数量动态膨胀的问题，并提供了更适合在线部署的分解式推理流程。
