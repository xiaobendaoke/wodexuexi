# 面向毕业论文的项目说明

## 论文定位

本项目建议围绕下面这个题目来组织：

**面向多无人机移动边缘计算的双层协同优化：轨迹控制与学习式任务卸载**

核心思想不是把所有模块都混在一起同时优化，而是把系统拆成两层：

- 上层：UAV 轨迹控制
- 下层：请求级任务卸载

其中：

- 上层负责决定无人机如何移动，从而影响覆盖关系、链路质量、邻居可用性和系统负载分布。
- 下层负责在每个服务请求到来时，决定该请求由本地 UAV、协作 UAV 还是 MBS 处理。

这两层通过系统状态耦合：

- 轨迹影响链路与协作机会
- 卸载影响时延、MBS 负载和计算资源使用

因此，这篇论文要讲的不是“某一个算法绝对最优”，而是：

**多无人机 MEC 的系统性能来源于轨迹控制与任务卸载的协同优化。**

## 推荐论文结构

建议把全文按三部分实验来组织：

1. 上层轨迹控制器筛选
2. 下层卸载策略筛选
3. 上下层联合实验

对应的主问题分别是：

1. 哪个轨迹控制器最适合作为双层框架的上层控制器？
2. 哪种卸载策略最适合作为双层框架的下层策略？
3. 当上层和下层组合后，系统整体性能如何变化？

## 上层实验

上层建议先只比较：

- 主算法：`attention_mappo`
- 强基线：`uncoordinated_greedy`

推荐命令：

```powershell
.\.venv\Scripts\python.exe run_all_experiments.py --skip_offload --models attention_mappo uncoordinated_greedy --train_episodes 300 --test_episodes 50 --name upper_compare_1
```

重点指标：

- `deadline_satisfaction_rate`
- `latency`
- `energy`
- `fairness`

判断顺序建议：

1. 先看 `deadline_satisfaction_rate`
2. 再看 `latency`
3. `energy` 和 `fairness` 作为补充

这一部分的目标不是强行证明 MARL 一定赢，而是给最终双层框架选一个**合理的上层控制器**。

推荐论文写法：

- 如果 `attention_mappo` 更好：写成“学习型轨迹控制器优于强启发式基线”
- 如果 `uncoordinated_greedy` 更好：写成“强启发式基线在当前设置下仍具有优势，但 `attention_mappo` 作为可学习控制器更适合双层扩展框架”

## 下层实验

下层建议比较三类策略：

- `heuristic_offloading`：传统规则基线
- `surrogate_baseline`：启发式蒸馏基线
- `rich_reduced_runtime_policy`：主学习策略

推荐命令：

```powershell
.\.venv\Scripts\python.exe run_all_experiments.py --skip_rl --name lower_learned_1
```

如果先想把主流程稳定跑完：

```powershell
.\.venv\Scripts\python.exe run_all_experiments.py --skip_rl --skip_advanced_validity --name lower_learned_1
```

重点结果文件：

- `results/full_offload_experiments/lower_learned_1_offload/experiment_summary.json`
- `results/full_runs/lower_learned_1/reports/offload_policy_validity_report.json`

重点指标分两类：

- 离线泛化：`macro_f1`、`cross_scenario_split`
- 系统级效果：`latency`、`deadline_satisfaction_rate`、`mbs_load_ratio`

这一部分要注意角色定位：

- `surrogate` 不是论文主方法，它主要说明“神经网络可以复现启发式规则”
- `rich_reduced` 才是你应该重点讲的学习式卸载策略

## 联合实验

分开跑上层和下层之后，最后再做组合实验。

推荐比较：

- `attention_mappo + heuristic`
- `attention_mappo + surrogate`
- `attention_mappo + rich_reduced`

如果已经有上层和下层实验结果，可以使用：

```powershell
.\.venv\Scripts\python.exe run_joint_trajectory_offload_experiment.py `
  --name joint_attn_mappo_1 `
  --trajectory_run_root results\full_runs\full_run_1 `
  --trajectory_model attention_mappo `
  --offload_experiment_root results\full_offload_experiments\full_run_1_offload `
  --policies heuristic surrogate rich_reduced `
  --seeds 42 84 126 168 `
  --episodes_per_seed 8
```

联合实验的目标不是再筛选单层方法，而是回答：

**在固定上层轨迹控制器后，不同下层卸载策略会给系统带来什么差异？**

## 论文里的主叙事

建议把全文讲成下面这条逻辑链：

1. 多无人机 MEC 的性能不仅由 UAV 轨迹决定，也由任务卸载方式决定。
2. 仅优化轨迹、固定卸载规则，无法充分利用动态链路和协作资源。
3. 因此本文提出双层协同优化框架：上层控制轨迹，下层控制请求级卸载。
4. 通过分阶段实验筛选上下层策略，并通过联合实验验证组合效果。

一句话版本：

**本文关注的不是单一控制模块，而是多无人机 MEC 中轨迹控制与任务卸载的协同优化机制。**

## 结果解释原则

如果实验结果很好，可以写：

- 学习型轨迹控制器提升了系统服务质量
- 学习式卸载策略进一步改善了时延、负载或截止满足率
- 双层联合优于单纯规则式方案

如果实验结果没有全面超过启发式，也仍然可以成立：

- 你证明了双层框架是可运行的
- 你区分了“启发式蒸馏”和“真正学习式卸载”
- 你识别出当前系统中上层学习和下层学习分别面临的难点

这同样是一个有效的研究结论。

## 当前推荐的正文定位

正文建议这样写：

- 上层主算法：`attention_mappo`
- 上层强基线：`uncoordinated_greedy`
- 下层主学习策略：`rich_reduced_runtime_policy`
- 下层蒸馏基线：`surrogate_baseline`
- 下层规则基线：`heuristic_offloading`

## 建议的章节安排

1. 绪论
2. 系统模型与问题描述
3. 双层协同优化框架
4. 上层轨迹控制方法
5. 下层学习式卸载策略
6. 实验设计与结果分析
7. 总结与展望

## 配套文档

- 论文草图见 [docs/PAPER_DRAFT.md](/C:/Users/woshinibaba/Documents/研究生/论文/毕业论文/多无人机/docs/PAPER_DRAFT.md)
- 原始项目说明见 [README.md](/C:/Users/woshinibaba/Documents/研究生/论文/毕业论文/多无人机/README.md)
