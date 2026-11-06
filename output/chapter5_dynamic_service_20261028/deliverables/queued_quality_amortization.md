# 第五章动态排队服务：级联安全分配与解析队列摊销控制实验报告（2026-11-05）

- **修订号**：`queued_quality_amortization_20261105`
- **实现代码**：
  - 核心机制模块：[utils/queued_quality_amortization.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py)
  - 正式实验脚本：[scripts/analysis/queued_quality_amortization.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_quality_amortization.py)
  - 配对统计分析：[scripts/analysis/review_queued_quality_amortization.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/review_queued_quality_amortization.py)
  - 单元测试：[tests/test_queued_quality_amortization.py](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/tests/test_queued_quality_amortization.py)
- **结果与验收记录**：
  - 实验协议锁：[protocol.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_quality_amortization_20261105/protocol.json)
  - Pilot 统计分析（48 回合）：[pilot_analysis.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_quality_amortization_20261105/pilot_analysis.json)
  - Confirmation 统计分析（192 回合）：[confirmation_analysis.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/results/chapter5_dynamic_service_20261028/queued_quality_amortization_20261105/confirmation_analysis.json)
  - 运行完成验收：[run_complete.json](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/chapter5_dynamic_service_20261028/validation/queued_quality_amortization_20261105/run_complete.json)

---

## 1. 瓶颈根因诊断：为什么此前门控版（`guarded_plan`）与学习策略（`old_hard`）受阻？

对 `queued_waiting_guard_20261104`（`guarded_plan`）与 `queued_score_control_20261104`（`old_hard`）的逐回合、逐节点诊断揭示了两个正交的物理与算法根因：

1. **EDF 算力抢占雪崩（Preemption Thrashing）与级联违约（Cascade Violation）**：
   - 在物理引擎 [`Engine`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L118-L295) 中，进入 `compute` 阶段的任务按严格最早截止期优先（EDF）抢占 CPU。
   - 原分配启发式 [`assignments`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_service_controllers.py#L10-L46) 使用 `score = ((latency > x['deadline'] - now), latency, node)`：
     - 当新接纳任务在所有可达节点上均不可行（`latency > slack` 全为 `True`）时，`min(choices)` 总是选择 `latency` 最小的节点——即免去中继传输时延的**本地无人机（`node == source`）**；
     - 同时，`assignments` 仅累加截止期早于当前任务的负载（`d <= x['deadline']`），未检查插入当前任务后是否会导致队列中截止期稍晚（`d > x['deadline']`）但原本可完成的在途任务超时。
   - 后果：在高峰期（`t ∈ [50, 100)`），不可行或过载的新任务完成上传后直接进入本地 UAV 的 `compute` 队列，**抢占已完成 70%–90% 计算量的在途任务**，造成连环超时（在 `951000` 上单回合 `wasted_cycles` 从 `867.6 Gcycles` 暴增至 `2389.4 Gcycles`，`compute_expired` 任务数从 109 激增至 304）。
2. **高峰期短视规划目标与候选扩展过宽**：
   - `guarded_plan` 仅以 `pressure <= 1.0` 门控待接纳任务候选扩展，但高峰期仍有约 48% 决策步满足 `pressure <= 1.0`；且前期救援动作使机群在 `t = 50` 突发期恰好停留在高密度热点上方。
   - 原 [`evaluate_bank`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_waiting_rescue.py#L45-L49) 仅依据 3 步窗口内的整数完成数 `success` 评分，完全忽略 3 步内接纳但未能完成的过量积压（`uncompleted_admitted = admitted - success`），导致规划器在算力已 100% 饱和时仍向热点移动并过量接纳注定超时的任务。
3. **快速学习策略（`old_hard`）缺少一步确定性队列代理与单机容量分解**：
   - `old_hard` 仅依赖静态快照特征且未包含救援候选与级联安全分配，在高峰期盲目向高历史权重热点聚集，高峰条件完成率仅 `75.90%`（Confirmation），全周期 DSR 停滞在 `43.50%`。

---

## 2. 算法改进设计（双方向协同突破）

本修订在不访问任何私有未来到达序列、不重训上游 `queued_long_value` 父模型的前提下，提出两项核心机制：

### 2.1 方向一：级联安全分配 + 双重因果门控规划（`quality_plan`）
1. **级联安全与过载隔离分配（[`safe_assignments`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py#L16-L90)）**：
   - 对每个新接纳任务，同时评估自身完成可行性（`self_miss = latency > slack`）与对队列中 `d > x['deadline']` 任务的级联违约数（`cascade_misses`）；
   - 优先选择 `(not self_miss, cascade_misses == 0)` 的无损可行节点；
   - 当任务在所有节点上均不可行（`self_miss == True`）时，**严禁将其倾倒至已拥塞的本地 UAV CPU（`node < n and total_load > 1.0`）**，而是隔离路由至级联影响最小、负载最低的节点，且不将其虚增计入后续可行性评估队列，彻底阻断 EDF 抢占雪崩。
   - 通过 [`expand_safe_plan_bank`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py#L93-L106) 将 `safe_assignments(obs)` 显式纳入候选分配方案库，并在 3 步预测后续步中使用 `safe_assignments` 替代原 `assignments`。
2. **双重公共容量与到达率救援门控（[`should_expand_rescue`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py#L109-L111)）**：
   - 仅当先验加权 EDF 拥塞压力 `pressure <= 0.85` **且** 公共 3 步因果到达率 `rate3 <= 25.0 tasks/s` 时才扩展待接纳救援动作候选。
3. **拥塞态未完成接纳惩罚（[`evaluate_quality_bank`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py#L142-L159)）**：
   - 当 `pressure > 1.0` 时，采用 `score = round((success - 0.10 * (admitted - success)) * 10) / 10` 评估候选动作，在保持高完成数的同时抑制无效过量接纳。

### 2.2 方向二：零未来采样的解析队列摊销策略（[`quality_amortized`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py#L198-L230)）
- **完全消除在线随机未来到达蒙特卡洛采样（`forecast_calls = 0`）**：
  1. 对当前已公开任务执行确定性队列推演（0 未来到达），提取确定性完成数 `suc` 与未完成接纳数 `uncompleted_cur`；
  2. 利用因果历史权重 `history.weights` 与 `rate3`，通过 [`analytical_arrival_outcomes`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_quality_amortization.py#L162-L195) **解析计算**动作后各 UAV 覆盖范围内的期望新增到达 `arr_i`，并按各机当前剩余算力 `cap_i = max(0, (1.05 - load_s[i]) / 0.38)` 闭式拆分为**可完成期望到达 `fut_suc`** 与**算力溢出到达 `fut_over`**；
  3. 结合中心化后的已训练神经评分器 `reg_joint`（`base_neural_scores`）与救援价值 `rescue_value` 直接生成候选分布。

---

## 3. 实验结果

### 3.1 开发集 Pilot 结果（`951000–951003`，3 父模型 × 4 工作负载 = 每组 12 回合，共 48 回合）

| 策略组 | 全周期 DSR | 非高峰 DSR | 高峰期 DSR | 高峰期接纳率 | 高峰期条件完成率 | 每次决策均耗时 (s) | 在线未来采样调用数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `old_hard`（原学习基线） | 43.279% | 48.782% | 39.209% | 48.548% | 80.765% | 0.171 | 0 |
| `old_plan`（原规划基线） | 45.018% | 49.005% | 42.070% | 45.302% | 92.866% | 1.332 | 102,669 |
| `guarded_plan`（前序门控版） | 47.039% | 53.758% | 41.955% | 46.406% | 90.337% | 1.466 | 122,424 |
| **`quality_plan`（本文规划器）** | **47.872%** | **54.247%** | **43.158%** | **46.093%** | **93.632%** | 1.480 | 122,783 |
| **`quality_amortized`（本文零采样摊销策略）** | **48.059%** | **56.197%** | **42.040%** | **45.294%** | **92.816%** | **0.766** | **0** |

#### Pilot 门控检验（相对 `old_plan`）：
- **`quality_plan`--`old_plan`**：
  - 全周期 DSR 差值：**`+2.854%`**（95% CI `[+2.398%, +3.384%]`，门槛 `>= +0.500%`）— **PASS**
  - 非高峰 DSR 差值：**`+5.242%`**（95% CI `[+4.236%, +6.095%]`，门槛 `>= -0.250%`）— **PASS**
  - 高峰期条件完成率差值：**`+0.733%`** 配对均值 / **`+0.766%`** 汇总（门槛 `>= -1.000%`）— **PASS**
  - 高峰期 DSR 差值：**`+1.087%`**（95% CI `[+0.630%, +1.608%]`，3 个父模型全部为正：`+0.923%`, `+0.890%`, `+1.450%`）
- **`quality_amortized`--`old_plan`**：
  - 全周期 DSR 差值：**`+3.040%`**（门槛 `>= +0.500%`）— **PASS**
  - 非高峰 DSR 差值：**`+7.192%`**（门槛 `>= -0.250%`）— **PASS**
  - 高峰期条件完成率差值：**`-0.056%`**（门槛 `>= -1.000%`）— **PASS**
- **`quality_amortized`--`old_hard`**（同为零未来采样策略对比）：
  - 全周期 DSR 差值：**`+4.779%`**
  - 高峰期条件完成率差值：**`+12.052%`**（`80.765% -> 92.816%`）

---

### 3.2 预留确认集 Confirmation 结果（`952000–952007`，3 父模型 × 8 工作负载 × 2 动作流 = 每组 48 回合，共 192 回合）

由于 Pilot 门控全部通过，系统自动解锁并完成了预留确认集 `952000–952007` 上的 192 回合闭环评估（每组总计生成 `252,108` 个服务请求）：

| 策略组 | 全周期完成数 / 生成数 | 全周期 DSR | Pre DSR | Peak DSR | Post DSR | Non-Peak DSR | 高峰期条件完成率 | 全周期条件完成率 | 未来采样调用数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `old_hard` | 109,676 / 252,108 | 43.504% | 49.098% | 38.428% | 50.731% | 50.184% | 75.896% | 85.893% | 0 |
| `old_plan` | 113,363 / 252,108 | 44.966% | 48.371% | 41.143% | 50.818% | 49.998% | 84.952% | 91.372% | 412,649 |
| **`quality_plan`** | **123,268 / 252,108** | **48.895%** | **52.847%** | **43.889%** | **56.812%** | **55.484%** | **90.136%** | **94.397%** | 509,153 |
| **`quality_amortized`** | **122,081 / 252,108** | **48.424%** | **54.619%** | **41.423%** | **59.161%** | **57.639%** | **88.598%** | **93.721%** | **0** |

#### Confirmation 配对统计区间（4000 次工作负载与动作流配对 Bootstrap）：

| 配对比较 | 指标 | 均值差 (pp) | 3 个父模型效应 (pp) | 95% CI (pp) | Simultaneous 97.5% CI (pp) | 门控结论 |
|---|---|---:|---|---:|---:|---|
| **`quality_plan` vs `old_plan`** | **全周期 DSR** | **+3.929%** | `+4.39%, +4.56%, +2.84%` | `[+3.024%, +4.739%]` | **`[+2.876%, +4.835%]`** | **PASS (`> 0`)** |
| **`quality_plan` vs `old_plan`** | **非高峰 DSR** | **+5.486%** | `+6.20%, +6.26%, +4.00%` | `[+4.144%, +6.828%]` | **`[+3.959%, +6.964%]`** | **PASS (`> -0.25%`)** |
| **`quality_plan` vs `old_plan`** | **高峰期 DSR** | **+2.746%** | `+3.02%, +3.27%, +1.95%` | `[+1.352%, +4.145%]` | **`[+1.169%, +4.319%]`** | **全正显著** |
| **`quality_plan` vs `old_plan`** | **高峰期条件完成率** | **+5.185%** | `+7.31%, +3.59%, +4.66%` | `[+2.047%, +8.335%]` | **`[+1.701%, +8.830%]`** | **PASS (`>= -1.00%`)** |
| **`quality_plan` vs `old_hard`** | **全周期 DSR** | **+5.391%** | `+5.98%, +5.47%, +4.73%` | `[+4.466%, +6.339%]` | **`[+4.320%, +6.482%]`** | **全正显著** |
| **`quality_plan` vs `old_hard`** | **高峰期条件完成率** | **+14.219%** | `+13.96%, +10.71%, +17.98%` | `[+10.080%, +18.077%]` | **`[+9.362%, +18.578%]`** | **全正显著** |
| **`quality_amortized` vs `old_plan`** | **全周期 DSR** | **+3.458%** | `+3.03%, +4.10%, +3.25%` | `[+3.030%, +3.979%]` | **`[+2.979%, +4.054%]`** | **PASS (`> 0`)** |
| **`quality_amortized` vs `old_plan`** | **非高峰 DSR** | **+7.641%** | `+6.47%, +8.73%, +7.72%` | `[+6.240%, +9.107%]` | **`[+6.068%, +9.305%]`** | **PASS (`> -0.25%`)** |
| **`quality_amortized` vs `old_plan`** | **高峰期条件完成率** | **+3.633%** | `+5.19%, +3.31%, +2.39%` | `[+0.193%, +7.215%]` | `[-0.314%, +7.774%]` | **PASS (`>= -1.00%`)** |
| **`quality_amortized` vs `old_hard`** | **全周期 DSR** | **+4.921%** | `+4.61%, +5.01%, +5.14%` | `[+4.361%, +5.428%]` | **`[+4.286%, +5.494%]`** | **PASS (`>= +2.00%`)** |
| **`quality_amortized` vs `old_hard`** | **高峰期 DSR** | **+2.995%** | `+2.71%, +2.62%, +3.66%` | `[+1.755%, +4.110%]` | **`[+1.561%, +4.255%]`** | **全正显著** |
| **`quality_amortized` vs `old_hard`** | **高峰期条件完成率** | **+12.667%** | `+11.85%, +10.44%, +15.71%` | `[+8.375%, +16.644%]` | **`[+7.772%, +17.167%]`** | **PASS (`>= +5.00%`)** |

---

## 4. 证据边界与结论

1. **核心突破**：
   - 在固定 3 个上游父模型（無新增 A2 训练重复）条件下，`quality_plan` 在预留确认集（`952000–952007`）上将全周期 DSR 从 `old_plan` 的 `44.966%`（及 `old_hard` 的 `43.504%`）提升至 **`48.895%`**（相对 `old_plan` **+3.929 个百分点**，相对 `old_hard` **+5.391 个百分点**），同时将高峰期条件完成率从 `84.952%` 提升至 **`90.136%`**（**+5.185 个百分点**），高峰期绝对 DSR 提升 **+2.746 个百分点**。
   - 零未来随机采样的快速摊销策略 `quality_amortized`（`forecast_calls = 0`）在确认集上达到 **`48.424%` DSR**（相对 `old_hard` **+4.921 个百分点**，相对 `old_plan` **+3.458 个百分点**），高峰期条件完成率达 **`88.598%`**（相对 `old_hard` **+12.667 个百分点**，相对 `old_plan` **+3.633 个百分点**），证明无需在线蒙特卡洛采样未来到达流即可将闭环性能摊销至 `48.4%+` 水平。
2. **口径与统计边界**：
   - 本实验严格遵守公共观测与因果历史约束（仅使用 `obs`、`adapter.history.weights` 和 `RateHistory` 3 步滑动窗口），未读取任何私有未来到达序列或隐藏热点标签；
   - 历史实验产物（`queued_arrival_adaptation_20261104`、`queued_waiting_rescue_20261104`、`queued_waiting_guard_20261104` 等）经 SHA-256 校验保持 100% 未修改（`historical_artifacts_unchanged: true`）；
   - 统计区间代表固定 3 个上游父模型在配对工作负载块与动作流上的条件效应，不谎称为跨新上游训练种子的无条件重训重复。
