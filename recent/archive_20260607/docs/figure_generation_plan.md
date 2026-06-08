# 论文第五章图表生成方案

## 一、方案目标

本文图表方案服务于第五章实验结果展示，目标是形成一条清晰的论文叙事链：

```text
1. 本文方法是否优于 baseline？
2. 本文方法为什么更好？
3. attention、双层分解、mask、Lagrange 等组件分别起什么作用？
4. 在 workload 变化时，本文方法是否仍然稳定？
```

本方案主要参考以下两类论文：

1. **主实验逻辑参考**：

```text
T. Du, X. Gui, and T. Sheng,
"Multiagent Deep Reinforcement Learning-Based Hierarchical Scheduling in Heterogeneous UAV-Enabled Vehicular Networks,"
IEEE Internet of Things Journal, vol. 12, no. 24, pp. 54938-54954, 2025.
DOI: 10.1109/JIOT.2025.3621756
```

重点借鉴：

```text
GMAPPO / Self-Interested PPO / Random / Uniform baseline 设计
收敛曲线
不同 agent 数量敏感性
时延分析
负载公平性分析
任务规模敏感性
任务负载分布可视化
```

2. **图表呈现风格参考**：

```text
刘潇，《基于深度强化学习的无人机辅助通信资源分配算法研究》
```

重点借鉴：

```text
不同算法轨迹图
不同 DRL 算法收敛图
参数敏感性图
二维/三维场景展示图
```

注意：刘潇论文适合作为画图风格参考，但 baseline 主逻辑应以 MAHHV 论文为主。

---

## 二、推荐图表总览

### 2.1 最低必做图表

如果时间紧，至少完成以下图表：

| 编号 | 图表 | 类型 | 主要目的 |
|------|------|------|----------|
| 表 5-1 | 主性能对比表 | 表格 | 展示不同算法的综合性能 |
| 图 5-1 | 训练收敛曲线 | 折线图 | 证明学习型算法能收敛，Proposed 更稳定 |
| 图 5-2 | 主指标柱状图 | 多子图柱状图 | 直观展示 reward、latency、energy、fairness、DSR、MBS load |
| 图 5-3 | 卸载比例堆叠柱状图 | 堆叠柱状图 | 分析 local/cooperative/MBS 卸载行为 |
| 图 5-4 | UAV 轨迹对比图 | 2D 轨迹图 | 展示不同策略的空间行为 |
| 图 5-5 | 组件消融图 | 多子图柱状图 | 证明 attention、mask、Lagrange、hierarchical decomposition 的作用 |

### 2.2 建议补充图表

| 编号 | 图表 | 类型 | 主要目的 |
|------|------|------|----------|
| 图 5-6 | 负载公平性热力图 | heatmap | 对齐 MAHHV 的 load fairness 分析 |
| 图 5-7 | workload 强度敏感性图 | 折线图 | 验证不同任务压力下的稳定性 |
| 图 5-8 | DSR-priority trade-off 图 | 折线图/柱状图 | 解释 DSR、能耗、MBS load 的权衡 |

### 2.3 可选图表

| 编号 | 图表 | 类型 | 主要目的 |
|------|------|------|----------|
| 图 5-9 | 不同 UAV 数量敏感性 | 折线图 | 对齐 MAHHV 的 different number of RUs |
| 图 5-10 | action mask 效果图 | 柱状图 | 展示 mask 屏蔽低质量动作的比例 |
| 图 5-11 | Lagrange 乘子变化曲线 | 折线图 | 展示约束调节过程 |
| 图 5-12 | attention 权重可视化 | heatmap | 展示 attention 的可解释性 |

---

## 三、主结果表设计

### 表 5-1：不同算法综合性能对比

#### 对比方法

```text
Random
Uniform
IPPO / Self-Interested PPO
Vanilla MAPPO / General MAPPO
Joint MAPPO
Proposed H-Attention-MAPPO
```

可选加入：

```text
Greedy-TSP
```

#### 指标

```text
Reward
Average Latency
Energy
Jain Fairness
Deadline Satisfaction Rate (DSR)
Offline Rate
MBS Load Ratio
Local Offloading Ratio
Cooperative Offloading Ratio
MBS Offloading Ratio
```

#### 表格格式

每个指标建议写：

```text
mean ± 95% CI
```

如果做统计检验，可额外标注：

```text
p-value vs Proposed
paired t-test / Wilcoxon
```

#### 数据来源

```text
results/baseline_comparison/<method>/metrics_seed_<seed>_workload_<workload_seed>.json
```

#### Claude Code 任务

新增或修改：

```text
scripts/generate_chapter5_tables.py
utils/baseline_metrics.py
```

输出：

```text
results/baseline_comparison/tables/table_main_comparison.csv
results/baseline_comparison/tables/table_main_comparison.tex
```

---

## 四、必须生成的图

### 图 5-1：训练收敛曲线

#### 目的

证明学习型算法能收敛，并比较不同方法的收敛速度、最终 reward 和稳定性。

#### 对比方法

```text
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

可选加入：

```text
DQN-Trajectory
```

不建议加入：

```text
Random
Uniform
Greedy-TSP
```

因为它们没有训练过程。

#### 横轴与纵轴

```text
x-axis: Training episode
y-axis: Moving-average episode reward
```

#### 画法

```text
折线图
多 seed 平均曲线
95% CI 阴影
moving average window = 10 或 20
```

#### 输出文件

```text
docs/figures/fig_convergence_comparison.pdf
docs/figures/fig_convergence_comparison.png
latex/docs/figures/fig_convergence_comparison.pdf
```

#### 参考

```text
MAHHV Fig. 5: Convergence of different algorithms
刘潇论文：不同 DRL 算法收敛曲线
```

---

### 图 5-2：主性能指标柱状图

#### 目的

直观展示 Proposed 在核心指标上的整体优势。

#### 对比方法

```text
Random
Uniform
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

#### 子图设计

建议 2 x 3 布局：

```text
(a) Reward
(b) Average latency
(c) Energy
(d) Jain fairness
(e) DSR
(f) MBS load ratio
```

#### 画法

```text
柱状图 + error bar
error bar = 95% CI
统一颜色映射
```

#### 指标方向

```text
Reward: higher is better
Latency: lower is better
Energy: lower is better
Fairness: higher is better
DSR: higher is better
MBS load ratio: lower is better
```

#### 输出文件

```text
docs/figures/fig_main_metric_bars.pdf
docs/figures/fig_main_metric_bars.png
latex/docs/figures/fig_main_metric_bars.pdf
```

---

### 图 5-3：卸载比例堆叠柱状图

#### 目的

解释不同算法的任务卸载行为，展示 local、cooperative UAV 和 MBS 三类执行路径的比例。

#### 对比方法

```text
Random
Uniform
IPPO
Vanilla MAPPO
Joint MAPPO
Proposed
```

#### 指标

```text
Local execution ratio
Cooperative UAV execution ratio
MBS execution ratio
```

#### 画法

```text
堆叠柱状图
每个方法一根柱子
三种颜色分别表示 local / cooperative / MBS
总和应为 1 或 100%
```

#### 论文解释重点

```text
1. Proposed 是否减少 MBS 依赖。
2. Proposed 是否更合理利用 cooperative UAV。
3. Uniform 是否只是机械均衡。
4. Joint MAPPO 是否卸载分布不稳定。
5. Lagrange 和 mask 是否改变 MBS load ratio。
```

#### 输出文件

```text
docs/figures/fig_offloading_ratio_stacked.pdf
docs/figures/fig_offloading_ratio_stacked.png
latex/docs/figures/fig_offloading_ratio_stacked.pdf
```

---

### 图 5-4：UAV 轨迹对比图

#### 目的

展示不同算法的空间行为，说明 Proposed 是否能主动靠近高请求区域、改善覆盖并减少无效移动。

#### 对比方法

为避免图过于拥挤，建议只选：

```text
Random
Uniform 或 Greedy-TSP
Vanilla MAPPO
Joint MAPPO
Proposed
```

#### 图中元素

```text
UE positions: 灰色小点
High-request UEs: 深色点或更大点
UAV initial positions: 空心圆
UAV final positions: 叉号
UAV trajectories: 不同颜色折线
MBS position: 三角形或星形
Coverage radius: 可选，浅色圆
```

#### 画法

```text
2D 轨迹图优先
固定同一个 seed 和 workload seed
每种方法一张子图或 2 x 3 布局
坐标范围与环境区域一致，如 700 x 700 m
```

#### 不建议

```text
如果 UAV 高度固定，不优先画 3D 图。
3D 图可读性通常不如 2D 图。
```

#### 输出文件

```text
docs/figures/fig_trajectory_comparison.pdf
docs/figures/fig_trajectory_comparison.png
latex/docs/figures/fig_trajectory_comparison.pdf
```

#### 参考

```text
刘潇论文中的 2D/3D UAV 轨迹图
DQN-COTO 类 UAV trajectory map
```

---

### 图 5-5：组件消融对比图

#### 目的

证明本文各创新组件的作用。

#### 对比方法

```text
Proposed
w/o upper attention
w/o lower attention
w/o quality mask
w/o Lagrange
Joint MAPPO
```

#### 指标

建议做 1 x 5 或 2 x 3 子图：

```text
Reward
Energy
DSR
MBS load ratio
Fairness
Offloading ratio, optional
```

#### 画法

```text
柱状图 + error bar
不要优先用雷达图，柱状图更适合论文精确比较
```

#### 论文解释重点

```text
upper attention: 影响 UAV 协同轨迹和覆盖关系建模
lower attention: 影响请求级卸载决策
quality mask: 减少低质量/不可行动作
Lagrange: 调节 DSR 与 MBS load trade-off
Joint MAPPO: 证明双层分解优于直接联合学习
```

#### 输出文件

```text
docs/figures/fig_ablation_components.pdf
docs/figures/fig_ablation_components.png
latex/docs/figures/fig_ablation_components.pdf
```

---

## 五、建议生成的图

### 图 5-6：负载公平性热力图

#### 目的

对齐 MAHHV 的 load fairness 分析，展示不同算法是否会造成 UAV 或 MBS 过载。

#### 对比方法

建议重点展示：

```text
Uniform
Vanilla MAPPO
Joint MAPPO
Proposed
```

#### 指标

可选两种：

```text
每个 UAV/MBS 处理任务数量
每个 UAV/MBS 负载比例
```

#### 画法 A：heatmap

```text
x-axis: UAV_1 ... UAV_N, MBS
y-axis: methods
color: processed task count or load ratio
```

#### 画法 B：节点负载柱状图

```text
x-axis: nodes
y-axis: processed tasks
bar groups: methods
```

推荐使用 heatmap。

#### 输出文件

```text
docs/figures/fig_load_fairness_heatmap.pdf
docs/figures/fig_load_fairness_heatmap.png
latex/docs/figures/fig_load_fairness_heatmap.pdf
```

---

### 图 5-7：Workload 强度敏感性图

#### 目的

验证不同任务压力下 Proposed 的稳定性。

#### 横轴

优先选择：

```text
Low workload
Medium workload
High workload
```

或：

```text
request intensity = 0.5x, 1.0x, 1.5x
```

#### 纵轴

建议 1 x 3 子图：

```text
(a) DSR
(b) Average latency
(c) MBS load ratio
```

可选加入：

```text
Reward
Energy
```

#### 对比方法

```text
Uniform
Vanilla MAPPO
Joint MAPPO
Proposed
```

#### 画法

```text
折线图
每个方法一条线
error bar 或 CI 阴影
```

#### 注意

优先做 workload intensity，而不是 UAV 数量，因为修改 `NUM_UAVS` 可能影响模型输入输出维度、buffer 和 checkpoint。

#### 输出文件

```text
docs/figures/fig_workload_sensitivity.pdf
docs/figures/fig_workload_sensitivity.png
latex/docs/figures/fig_workload_sensitivity.pdf
```

---

### 图 5-8：DSR-priority trade-off 图

#### 目的

解释 DSR、能耗、MBS load 和 reward 之间的权衡，尤其适合本文已有 DSR-priority 配置叙事。

#### 横轴

```text
Default
DSR-priority
Strong DSR-priority
```

或：

```text
不同 W_DSR / OFFLOAD_DSR_TARGET / Lagrange 配置
```

#### 纵轴

建议 2 x 2 子图：

```text
(a) DSR
(b) Energy
(c) MBS load ratio
(d) Reward
```

#### 画法

```text
折线图或分组柱状图
```

#### 论文解释重点

```text
1. 提高 DSR 权重是否提升 DSR。
2. 提升 DSR 是否增加能耗或 MBS load。
3. Proposed 是否能在 DSR 与能耗之间保持可调节 trade-off。
```

#### 输出文件

```text
docs/figures/fig_dsr_priority_tradeoff.pdf
docs/figures/fig_dsr_priority_tradeoff.png
latex/docs/figures/fig_dsr_priority_tradeoff.pdf
```

---

## 六、可选图

### 图 5-9：不同 UAV 数量敏感性

#### 目的

对齐 MAHHV 的 different number of RUs 实验。

#### 横轴

```text
NUM_UAVS = 3, 5, 7
```

#### 纵轴

```text
Reward
Fairness
Latency
DSR
```

#### 对比方法

```text
Vanilla MAPPO
Proposed
```

#### 风险

如果修改 `NUM_UAVS` 会导致观测维度、模型结构和已有 checkpoint 不兼容，则不要优先做该图。

---

### 图 5-10：Action mask 效果图

#### 目的

展示 quality-aware action mask 实际屏蔽了多少低质量动作。

#### 指标

```text
Cooperative actions masked ratio
MBS actions masked ratio
Invalid/low-quality action ratio
```

#### 对比

```text
with mask
without mask
```

#### 输出文件

```text
docs/figures/fig_action_mask_effect.pdf
latex/docs/figures/fig_action_mask_effect.pdf
```

---

### 图 5-11：Lagrange 乘子变化曲线

#### 目的

展示约束机制在训练过程中如何动态调节 DSR 和 MBS load。

#### 横轴与纵轴

```text
x-axis: training episode
y-axis: lambda_dsr / lambda_mbs
```

#### 输出文件

```text
docs/figures/fig_lagrange_lambda_curve.pdf
latex/docs/figures/fig_lagrange_lambda_curve.pdf
```

---

### 图 5-12：Attention 权重可视化

#### 目的

展示 attention 模块关注哪些 UAV 或 UE，增强可解释性。

#### 画法

```text
UAV-UAV attention matrix heatmap
UAV-UE attention weights heatmap
```

#### 输出文件

```text
docs/figures/fig_attention_weights_heatmap.pdf
latex/docs/figures/fig_attention_weights_heatmap.pdf
```

---

## 七、建议的章节排布

```text
5.2 收敛性能分析
    图 5-1：训练收敛曲线

5.3 综合性能对比
    表 5-1：主性能对比表
    图 5-2：主指标柱状图

5.4 轨迹与卸载行为分析
    图 5-3：卸载比例堆叠柱状图
    图 5-4：UAV 轨迹对比图

5.5 负载公平性分析
    图 5-6：负载公平性热力图

5.6 消融实验
    图 5-5：组件消融图
    可选：图 5-10、图 5-11、图 5-12

5.7 参数敏感性分析
    图 5-7：workload 强度敏感性图
    图 5-8：DSR-priority trade-off 图
    可选：图 5-9：不同 UAV 数量敏感性
```

---

## 八、Claude Code 实现任务拆分

### Task 1：统一结果读取与统计工具

目标：

```text
读取 results/baseline_comparison/ 下所有方法的 metrics JSON。
按 method 聚合 training seeds 和 workload seeds。
计算 mean、std、95% CI。
```

建议新增：

```text
scripts/plot_chapter5_figures.py
utils/baseline_metrics.py
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --list-available
```

---

### Task 2：生成主表和主指标柱状图

目标：

```text
生成 table_main_comparison.csv/.tex
生成 fig_main_metric_bars.pdf/.png
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --fig main_metrics
```

---

### Task 3：生成收敛曲线

目标：

```text
读取 training_curve_seed_<seed>.json。
绘制 IPPO、Vanilla MAPPO、Joint MAPPO、Proposed 的 moving-average reward。
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --fig convergence
```

---

### Task 4：生成卸载比例和负载公平性图

目标：

```text
生成 offloading stacked bar。
生成 load fairness heatmap。
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --fig offloading
python scripts/plot_chapter5_figures.py --fig load_fairness
```

---

### Task 5：生成轨迹对比图

目标：

```text
读取 trajectory_seed_<seed>.json。
绘制典型 workload 下不同算法的 2D UAV 轨迹。
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --fig trajectory --seed 42 --workload_seed 1001
```

---

### Task 6：生成消融图

目标：

```text
读取 ablation 结果。
绘制 Proposed、w/o upper attention、w/o lower attention、w/o mask、w/o Lagrange、Joint MAPPO 对比。
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --fig ablation
```

---

### Task 7：生成参数敏感性图

目标：

```text
生成 workload sensitivity。
生成 DSR-priority trade-off。
```

完成标准：

```bash
python scripts/plot_chapter5_figures.py --fig workload_sensitivity
python scripts/plot_chapter5_figures.py --fig dsr_tradeoff
```

---

## 九、输出规范

所有图建议同时输出：

```text
PDF: 用于 LaTeX 论文
PNG: 用于 README、快速查看和答辩 PPT
```

输出目录：

```text
docs/figures/
latex/docs/figures/
results/baseline_comparison/figures/
```

命名规范：

```text
fig_convergence_comparison.pdf
fig_main_metric_bars.pdf
fig_offloading_ratio_stacked.pdf
fig_trajectory_comparison.pdf
fig_ablation_components.pdf
fig_load_fairness_heatmap.pdf
fig_workload_sensitivity.pdf
fig_dsr_priority_tradeoff.pdf
```

图中文字建议：

```text
英文坐标轴和图例，便于论文统一。
中文说明放在论文正文和图题中。
```

颜色建议：

```text
Random: gray
Uniform: blue-gray
IPPO: orange
Vanilla MAPPO: blue
Joint MAPPO: purple
Proposed: red 或 dark green
```

注意：同一算法在所有图中必须使用同一颜色。

---

## 十、最终优先级

### 必做

```text
表 5-1：主性能对比表
图 5-1：训练收敛曲线
图 5-2：主指标柱状图
图 5-3：卸载比例堆叠柱状图
图 5-4：UAV 轨迹对比图
图 5-5：组件消融图
```

### 强烈建议

```text
图 5-6：负载公平性热力图
图 5-7：workload 强度敏感性图
图 5-8：DSR-priority trade-off 图
```

### 可选

```text
图 5-9：不同 UAV 数量敏感性
图 5-10：Action mask 效果图
图 5-11：Lagrange 乘子变化曲线
图 5-12：Attention 权重可视化
```
