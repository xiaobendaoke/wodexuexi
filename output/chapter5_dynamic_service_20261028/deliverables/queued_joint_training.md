# 第五章动态持久队列：上下层耦合规划教师蒸馏与九臂消融实验报告（`queued_joint_training_20261105` 审计勘误版）

## 1. 算法定位、审计问题闭环与核心实验结论

> **算法本质定位说明**：本阶段实验（`queued_joint_training_20261105`）完成的是**面向双分支耦合架构的两轮 DAgger 规划教师蒸馏（Two-Round DAgger Coupled Policy Teacher Distillation）**。训练过程从仿真环境采集状态，利用离线 3 步确定性 `Engine` 推演与排队解析评分构造温度 $\tau = 0.015$ 的指数加权软教师分布 `target_joint`，并通过最小化双分支联合分布与教师分布之间的交叉熵（含经由 4 维期望向量 $C_{\text{upper}}$ 的跨层反向传播）拟合策略参数。本阶段作为真正环境回报驱动联合强化学习（见 [`queued_joint_rl.md`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/chapter5_dynamic_service_20261028/deliverables/queued_joint_rl.md)）的**共同暖启动初始化与强蒸馏对照基线**。

全部 **6 个纯神经网络评估臂** 在线闭环推理期间均严格满足 **`deterministic_engine_steps = 0`** 与 **`forecast_calls = 0`**（零在线 `Engine.from_observation` 拷贝、零在线多步推演、零离散赢家标记特征），并在盲测确认集（`953000–953007`，9 臂 × 48 回合 = 432 个仿真回合，每臂 252,000 个生成任务）上完成了闭环验证。

### 核心定量结论（确认集 `953000–953007`，每臂 48 回合 / 252,000 生成请求）

| 评估臂 | 算法角色与归因定位 | 在线 `Engine` 步数 | 随机前瞻调用 | 成功请求数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 平均决策耗时 (p95 / >1s占比) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `uncoordinated_init` | 未经训练的基线先验策略（Epoch 0） | 0 | 0 | 111,590 | **44.28%** | 49.21% | 40.56% | 84.89% | 0.401s (1.130s / 10.5%) |
| `lower_only_trained` | 单层分配蒸馏网络（冻结基线轨迹先验） | 0 | 0 | 111,908 | **44.41%** | 48.94% | 40.99% | 85.40% | 0.395s (1.125s / 9.9%) |
| `upper_only_trained` | 单层轨迹蒸馏网络（冻结基线分配先验） | 0 | 0 | 115,369 | **45.78%** | 54.83% | 38.96% | 74.77% | 0.432s (1.213s / 14.8%) |
| `decomposed_composed` | 独立蒸馏单层网络直接拼接（零重训） | 0 | 0 | 116,767 | **46.34%** | 55.26% | 39.61% | 76.36% | 0.432s (1.204s / 14.9%) |
| `decomposed_simultaneous` | 同数据集、同候选池、同联合教师去耦蒸馏 | 0 | 0 | 118,766 | **47.13%** | 54.70% | 41.42% | 86.39% | 0.409s (1.125s / 10.7%) |
| **`joint_trained`** | **双分支跨层耦合规划蒸馏网络（零在线仿真）** | **0** | **0** | **119,774** | **47.53%** | **55.05%** | **41.86%** | **90.37%** | **0.340s (1.007s / 5.2%)** |
| `old_plan` | 在线 3 步随机到达采样 MPC 基线 | 0 | 387,148 | 112,178 | **44.52%** | 47.90% | 41.96% | 93.22% | 1.533s (5.712s / 30.7%) |
| `quality_amortized` | 在线 3 步确定性 `Engine` 推演教师参考 | 1,545,285 | 0 | 119,954 | **47.60%** | 55.39% | 41.73% | 92.89% | 1.061s (3.986s / 25.2%) |
| **`joint_neural_mpc`** | **基于 `joint_trained` 裁剪 Top-6 的混合 MPC** | **0** | **58,153** | **121,107** | **48.06%** | **55.42%** | **42.51%** | **94.49%** | **0.523s (1.663s / 20.0%)** |

1. **单层蒸馏与直接拼接的定量表现及统计显著性边界**：
   - 仅蒸馏下层分配网络（`lower_only_trained`）将安全分配方案选择率从 `40.6%` 提升至 `82.7%`，拥塞期 DSR 由 `40.56%` 升至 `40.99%`（单比较 95% CI `[+0.07%, +0.75%]`，但 $M=7$ Bonferroni 区间 `[-0.06%, +0.85%]` 跨零），全程 DSR 差值为 `+0.13 pp`（单比较 95% CI `[-0.18%, +0.47%]` 跨零，未确认全程 DSR 单独提升）。
   - 仅蒸馏上层轨迹网络（`upper_only_trained`）将非拥塞期 DSR 推高 `+5.62 pp`（`49.21% -> 54.83%`），全程 DSR 达 `45.78%`（`+1.50 pp`，Bonferroni 95% CI `[+0.79%, +2.38%]`）。需要注意，其全局 `context` 输入实际上包含 `pressure`、`is_congested` 与队列任务计数；其在拥塞期仍高频选择救援动作（`55.6%`）并导致拥塞期条件完成率下降 `-10.09 pp`（`84.89% -> 74.77%`），是因为其离线教师目标 `q_upper` 未计入下层队列溢出与拥塞期救援惩罚。
   - 将独立蒸馏的单层网络直接拼接（`decomposed_composed`）取得 `46.34%` DSR，显著超越两个单层网络（较 `lower_only_trained` `+1.93 pp`，Bonferroni 95% CI `[+1.33%, +2.65%]`；较 `upper_only_trained` `+0.55 pp`，Bonferroni 95% CI `[+0.09%, +1.07%]`）。
2. **联合蒸馏相对直接拼接与同目标去耦的归因边界**：
   - **联合蒸馏相对直接拼接（`joint_trained - decomposed_composed`）**：全程 DSR 净增 **`+1.19` 个百分点**（单比较 95% CI `[+0.41%, +1.84%]`，Bonferroni 95% CI `[+0.08%, +1.99%]`，3 个父谱系分别为 `+1.29 / +1.02 / +1.27 pp`），正向收益经族水平校正得到确认；其中拥塞期 DSR 净增 `+2.26 pp`，拥塞期条件完成率提升 `+14.00 pp`。
   - **同目标去耦蒸馏相对直接拼接（`decomposed_simultaneous - decomposed_composed`）**：全程 DSR 差值 `+0.79 pp`，单比较 95% CI `[+0.15%, +1.40%]` 为正，但 Bonferroni 95% CI `[-0.09%, +1.62%]` 跨零；拥塞期条件完成率提升 `+10.03 pp`。
   - **耦合特征包相对同目标去耦（`joint_trained - decomposed_simultaneous`）**：在离线验证集上，耦合特征方案将相对教师 KL 从 `2.1664` 降至 `1.2929`、Top-1 遗憾从 `1.0809` 降至 `0.3418`；在闭环确认集上，**全程条件完成率提升 `+2.44 pp`**（Bonferroni 95% CI `[+0.89%, +4.67%]`），**拥塞期条件完成率提升 `+3.98 pp`**（Bonferroni 95% CI `[+1.17%, +8.16%]`），**拥塞期 DSR 提升 `+0.44 pp`**（Bonferroni 95% CI `[+0.06%, +0.77%]`）。但**全程 DSR 差值 `+0.40 pp` 的单比较 95% CI `[-0.32%, +0.94%]` 与 Bonferroni 95% CI `[-0.66%, +1.08%]` 均跨零**，且 3 个父谱系差值为 **`-0.01 / +0.06 / +1.15 pp`**（均值提升主要来自第 3 个父谱系 Parent 2）。因此，**显式耦合提高全程 DSR 在本轮蒸馏实验中尚未得到统计确认**。
   - **因果混杂说明**：在本轮蒸馏实现中，`joint` 与 `decomposed_simultaneous` 同时改变了下层方案条件化、跨方案特征平均、8 个运动特征通道屏蔽（含全局 `is_congested` 与 `is_congested * is_rescue`）、4 维期望向量 $C_{\text{upper}}$ 及跨层反传，且教师和输入特征显式编码了 `-5.0 * is_congested * is_rescue`。因此本轮只能检验整套耦合特征与教师蒸馏方案，尚不能单独分离 $C_{\text{upper}}$ 或声称网络无监督自主发现节流规律（该解混杂与去除手工惩罚在后续真正联合强化学习 [`queued_joint_rl.md`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/chapter5_dynamic_service_20261028/deliverables/queued_joint_rl.md) 中完成）。
3. **计算开销摊销、非劣效检验（TOST）与混合 MPC 边界**：
   - **非劣效与等效性边界**：`joint_trained`（`47.53%` DSR）相对在线 3 步确定性 `Engine` 推演参考 `quality_amortized`（`47.60%` DSR）的点估计比值为 `99.85%`，差值为 `-0.07 pp`，90% CI 为 `[-0.53%, +0.39%]`，单比较 95% CI 为 `[-0.62%, +0.47%]`，Bonferroni 95% CI 为 `[-0.85%, +0.65%]`。按统计学规范，“置信区间跨零”不直接等于等效：若预设等效/非劣效容许界值 $\Delta = 0.50\text{ pp}$，当前区间下界（`-0.62 pp`）尚未通过检验（允许最大约 `0.62 pp` 的单比较 DSR 损失）；仅在容许界值设为 $\Delta = 0.75\text{ pp}$ 时通过 90% CI 双单侧等效检验（TOST）。
   - **在线决策耗时边界**：在 256 进程并发评估下，`joint_trained` 将平均决策耗时从 `quality_amortized` 的 `1.061s` 降至 `0.340s`（加速 3.1 倍），`p95` 从 `3.986s` 降至 `1.007s`，但仍有 **`5.21%` 的时隙耗时超过 1.0 秒**（`joint_neural_mpc` 平均 `0.523s`、`p95 = 1.663s`、**`20.00%` 超过 1.0 秒**）。由于仿真环境未将计算超时折算为物理控制延迟，当前结果支持“平均计算加速”，不能声称“硬实时 1s 闭环已验证”。
   - **混合 MPC 归因说明**：`joint_neural_mpc`（`48.06%` DSR）相对 `joint_trained`（`+0.53 pp`）和 `quality_amortized`（`+0.46 pp`）在单比较 95% CI 下为正（`[+0.10%, +0.99%]` 与 `[+0.05%, +0.88%]`），但在 $M=7$ Bonferroni 校正后区间分别为 `[-0.04%, +1.17%]` 与 `[-0.08%, +1.06%]`（均跨零）。此外，`joint_neural_mpc` 与 `quality_amortized` / `old_plan` 的候选池门控（`unguarded` vs `gated` vs `none`）、前瞻采样与打分公式存在差异，其增益属于“神经候选裁剪 + 3 步随机前瞻组合策略”的整体效果，不能单独归因于 Top-6 裁剪。

---

## 2. 算法实现细节与串联排队解析近似说明

### 2.1 串联排队解析近似（`_lindley_plan_motion_eval`）及其物理反例说明

在 [`utils/queued_joint_training.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py#L260-L396) 中，特征提取函数 [`extract_actor_row`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py#L504-L658) 调用 [`_lindley_plan_motion_eval`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_training.py#L260-L396) 在零 `Engine` 实例化下估计上传、中继与节点 CPU 队列的完成数与溢出数。

- **近似性质与反例机制**：该函数按静态 `(deadline, uid)` 顺序单遍更新标量 `c_free[tgt] = max(c_free[tgt], t_tr_done) + dt_cpu` 与 `u_free[src]`，属于**快速串联排队解析近似（Fast Tandem-Queue Analytical Approximation）**，而非精确的动态就绪 EDF 仿真。
- **双任务跨阶段反例验证**（见 [`review_queued_joint_training.py:lindley_counterexample_audit`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/review_queued_joint_training.py#L215-L248)）：当单架 UAV 上存在 Task 1（`deadline = 1.4s`，尚在 `upload` 阶段，需 `0.7s` 上传 + `0.5s` 计算）与 Task 2（`deadline = 1.5s`，已在 `compute` 阶段就绪，需 `0.6s` 计算）时，真实 [`Engine`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L208) 会在 `[0.0, 0.6s]` 空闲窗口先完成已就绪的 Task 2，随后在 `[0.7, 1.2s]` 完成 Task 1（真实成功 2 个、违约 0 个）；而静态单遍近似因先处理截止期较早的 Task 1 并将 `c_free[0]` 推至 `1.2s`，误将 Task 2 排在 `1.2 + 0.6 = 1.8s > 1.5s`（预测成功 1 个、违约 1 个）。此外，真实 `Engine` 中同 UAV 下不同 UE 的上传任务并发平分带宽，而静态近似按串行累加。后续联合强化学习模块（[`utils/queued_joint_rl.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py)）已引入考虑就绪时间（Release-Time）与并发上传平分带宽的改进多阶段排队近似并消除该反例。

### 2.2 双分支耦合架构与有限差分梯度检验

在 [`marl_models/queued_joint_training.py`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_training.py) 中，联合策略分解为：
$$P_\theta(k, m \mid s) = p_a(k \mid s; \theta) \cdot p_m(m \mid s, k; \theta)$$
- **上层轨迹分支输入 `xm`（42 维）**：拼接候选运动特征 $M[k, m] \in \mathbb{R}^{16}$、下层分配方案特征 $A[k] \in \mathbb{R}^{14}$ 与全局上下文特征 $c \in \mathbb{R}^{12}$。
- **下层分配分支输入 `xa`（30 维）**：拼接分配方案特征 $A[k] \in \mathbb{R}^{14}$、4 维上层策略期望耦合向量 $C_{\text{upper}}[k] = \sum_{m=1}^{M} p_m(m \mid s, k; \theta_{\text{motion}}) \cdot \Phi_{\text{couple}}[k, m, 0:4]$ 以及全局上下文特征 $c \in \mathbb{R}^{12}$（$14 + 4 + 12 = 30$ 维）。
- 单元测试 [`test_finite_difference_gradients_all_modes_and_tensors`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/tests/test_queued_joint_training.py#L94-L140) 验证了全部 4 种训练模式的所有参数张量（`w, W, b, v`）中心有限差分误差 `< 1e-8`。

---

## 3. 实验结果详表

### 3.1 离线验证集拟合收敛曲线（工作负载 `941000–941002`）

数据源：[`results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/training_curves.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/training_curves.json)

| 训练分支 | 训练目标与耦合模式 | Epoch 0 Val KL | Epoch 32 Val KL | Epoch 64 Val KL | Epoch 0 Top-1 Regret | Epoch 64 Top-1 Regret | Epoch 0 最优匹配概率 | Epoch 64 最优匹配概率 | Epoch 64 成对排序准确率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lower_only_trained` | 下层边际软价值目标（冻结上层） | 1.1723 | 0.4101 | **0.4418** | 0.9663 | **0.2212** | 41.01% | **82.68%** | **84.70%** |
| `upper_only_trained` | 上层非耦合覆盖目标（冻结下层） | 2.5042 | 0.9353 | **0.9292** | 2.3012 | **0.9048** | 5.54% | **21.46%** | **75.63%** |
| `decomposed_simultaneous` | 联合教师 `target_joint`（屏蔽跨层通道） | 4.1956 | 2.1523 | **2.1664** | 2.5593 | **1.0809** | 7.14% | **34.18%** | **80.48%** |
| **`joint_trained`** | **联合教师 `target_joint`（启用 $C_{\text{upper}}$ 与耦合通道）** | **4.1956** | **1.2521** | **1.2929** | **2.5593** | **0.3418** | **7.14%** | **63.68%** | **88.10%** |

### 3.2 Pilot 闭环检查点演化曲线（工作负载 `951000–951003`）

数据源：[`results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/pilot_analysis.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/pilot_analysis.json)

| 检查点轮次 | 在线 `Engine` 步数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 安全分配选择率 | 救援轨迹选择率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch 0 (`uncoordinated_init`) | 0 | 44.15% | 49.63% | 40.11% | 83.57% | 39.07% | 8.25% |
| Epoch 1 | 0 | 44.73% | 50.36% | 40.56% | 83.95% | 45.63% | 12.21% |
| Epoch 2 | 0 | 46.12% | 52.68% | 41.27% | 82.87% | 57.51% | 27.06% |
| Epoch 4 | 0 | 48.05% | 55.49% | 42.54% | 86.50% | 80.24% | 42.12% |
| Epoch 8 | 0 | 47.65% | 55.44% | 41.89% | 89.14% | 83.42% | 39.85% |
| Epoch 16 | 0 | 47.56% | 55.13% | 41.97% | 89.14% | 83.13% | 36.96% |
| Epoch 32 (Round 1 结束) | 0 | 47.71% | 55.83% | 41.71% | 89.08% | 80.94% | 36.55% |
| **Epoch 64 (Round 2 DAgger 结束)** | **0** | **48.02%** | **56.07%** | **42.06%** | **90.40%** | **81.85%** | **41.05%** |

### 3.3 确认集配对统计检验（`953000–953007`，`B = 10,000` 次重采样，由脚本自动生成）

数据源：[`results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/confirmation_analysis.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/confirmation_analysis.json)（下表所有数值均由 [`format_markdown_comparison_table`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/review_queued_joint_training.py#L251-L289) 从 JSON 直接导出并经自动化单元测试校验）

| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |
|---|---:|---:|---:|---:|---:|---:|---|
| `joint_trained - uncoordinated_init` | **+3.25 pp** | [+2.51, +3.74] | [+2.15, +3.87] | +5.84 pp | +1.30 pp | +5.49 pp | +3.45 / +3.07 / +3.22 pp |
| `joint_trained - lower_only_trained` | **+3.12 pp** | [+2.51, +3.56] | [+2.23, +3.65] | +6.10 pp | +0.87 pp | +4.95 pp | +3.19 / +2.83 / +3.34 pp |
| `joint_trained - upper_only_trained` | **+1.75 pp** | [+1.03, +2.32] | [+0.74, +2.44] | +0.22 pp | +2.90 pp | +15.58 pp | +1.67 / +1.68 / +1.90 pp |
| `joint_trained - decomposed_composed` | **+1.19 pp** | [+0.41, +1.84] | [+0.08, +1.99] | -0.22 pp | +2.26 pp | +14.00 pp | +1.29 / +1.02 / +1.27 pp |
| `joint_trained - decomposed_simultaneous` | **+0.40 pp** | [-0.32, +0.94] | [-0.66, +1.08] | +0.35 pp | +0.44 pp | +3.98 pp | -0.01 / +0.06 / +1.15 pp |
| `joint_trained - old_plan` | **+3.01 pp** | [+2.37, +3.66] | [+2.13, +3.88] | +7.15 pp | -0.10 pp | -2.84 pp | +2.94 / +2.84 / +3.26 pp |
| `joint_trained - quality_amortized` | **-0.07 pp** | [-0.62, +0.47] | [-0.85, +0.65] | -0.34 pp | +0.13 pp | -2.51 pp | -0.34 / +0.12 / +0.00 pp |
| `decomposed_composed - lower_only_trained` | **+1.93 pp** | [+1.47, +2.46] | [+1.33, +2.65] | +6.32 pp | -1.38 pp | -9.05 pp | +1.90 / +1.82 / +2.07 pp |
| `decomposed_composed - upper_only_trained` | **+0.55 pp** | [+0.21, +0.92] | [+0.09, +1.07] | +0.43 pp | +0.65 pp | +1.58 pp | +0.38 / +0.66 / +0.63 pp |
| `decomposed_composed - uncoordinated_init` | **+2.05 pp** | [+1.46, +2.70] | [+1.27, +2.96] | +6.05 pp | -0.96 pp | -8.51 pp | +2.16 / +2.06 / +1.95 pp |
| `decomposed_simultaneous - decomposed_composed` | **+0.79 pp** | [+0.15, +1.40] | [-0.09, +1.62] | -0.56 pp | +1.81 pp | +10.03 pp | +1.30 / +0.95 / +0.12 pp |
| `upper_only_trained - uncoordinated_init` | **+1.50 pp** | [+0.97, +2.12] | [+0.79, +2.38] | +5.62 pp | -1.61 pp | -10.09 pp | +1.78 / +1.40 / +1.32 pp |
| `lower_only_trained - uncoordinated_init` | **+0.13 pp** | [-0.18, +0.47] | [-0.28, +0.62] | -0.27 pp | +0.42 pp | +0.54 pp | +0.25 / +0.24 / -0.12 pp |
| `joint_neural_mpc - joint_trained` | **+0.53 pp** | [+0.10, +0.99] | [-0.04, +1.17] | +0.37 pp | +0.65 pp | +4.11 pp | +0.62 / +0.65 / +0.31 pp |
| `joint_neural_mpc - quality_amortized` | **+0.46 pp** | [+0.05, +0.88] | [-0.08, +1.06] | +0.03 pp | +0.78 pp | +1.60 pp | +0.28 / +0.77 / +0.32 pp |
| `joint_neural_mpc - decomposed_composed` | **+1.72 pp** | [+0.97, +2.44] | [+0.70, +2.67] | +0.15 pp | +2.91 pp | +18.12 pp | +1.91 / +1.67 / +1.59 pp |
| `joint_neural_mpc - old_plan` | **+3.54 pp** | [+2.99, +4.06] | [+2.79, +4.26] | +7.52 pp | +0.55 pp | +1.27 pp | +3.57 / +3.49 / +3.57 pp |

---

## 4. 章节承接边界与证据冻结说明

1. **章节组织边界**：本实验中的 `upper_only_trained` 与 `lower_only_trained` 是将第三章“未覆盖需求感知轨迹规划”与第四章“防级联违约安全分配”的核心思想投射到第五章持久队列候选动作空间上的单层神经网络蒸馏分支，并未直接复用第三章连续动作 Attention-MAPPO 或第四章固定快照字典序求解器 MC-ITR 源码，因此在论文正文中定位为“持久队列候选空间下的单层思想投射对照”，不替代原章节算法本身的验证。
2. **证据冻结与审计边界说明**：本轮结果目录 [`protocol.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/protocol.json) 已补充绑定当前源码、模型检查点与分析脚本的 SHA256 哈希摘要。由于本轮运行器 [`episode_worker`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_training.py#L251-L257) 支持断点恢复且未在确认集启动前单独写入带外部时间戳的预冻结清单，本轮对 `953000–953007` 的表述严格限定为“使用独立于训练（`940000–940011`）、验证（`941000–941002`）和 Pilot（`951000–951003`）的确认工作负载号段 `953000–953007`”。在后续真正联合强化学习实验（`queued_joint_rl_20261106`）中，强制执行确认前哈希冻结协议（`freeze_manifest.json`）。
