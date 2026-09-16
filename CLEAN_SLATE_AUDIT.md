# CLEAN_SLATE_AUDIT.md

多 UAV MEC 协同优化系统：Clean-Slate 重构前最终只读审计。

> 本阶段严格只读。未修改任何 source code，未自动修复，未训练，未覆盖历史结果，未删除文件。
> 仓库内全部历史实验、checkpoint、figure、table、aggregate result 一律视为 `HISTORICAL_ONLY`，仅可用于 debugging / regression comparison。
> 最终论文 canonical results 必须在修复后的统一环境中重新训练、重新评估、重新统计。
> 静态审计在 `LOCAL_RUNTIME_COMPATIBLE = NO` 条件下完成；未在本机执行 PyTorch 或环境步进。

---

## 1. Repository state

| 项 | 值 |
| --- | --- |
| Branch | `双层注意力MAPPO多智能体强化学习` |
| HEAD | `fcbf7f721e09923f37306f135afd694cd5039936` |
| OS | Darwin 25.6.0 (`uname -a`: `root:xnu-12377.161.14~5/RELEASE_ARM64_T8142`) |
| CPU architecture | `arm64` (Apple Silicon) |
| Host Python | `/usr/bin/python3`，Python 3.9.6 |
| Repo `.venv` Python | `.venv/bin/python`：`ELF 64-bit LSB pie executable, x86-64`，`home = /home/PengYanghan/miniconda3/bin`，`version = 3.9.12` |
| Host torch | 未安装（`ModuleNotFoundError: No module named 'torch'`） |
| Repo `.venv` torch | 无法执行（`exec format error`） |

Git status（审计开始时）：

```text
 M .DS_Store
 M .tmp/.DS_Store
 m .tmp/nature-skills-inspect
 M AGENTS.md
 M docs/.DS_Store
 M latex/.DS_Store
 M results/.DS_Store
?? CLEAN_SLATE_AUDIT.md
?? docs/figures/prism_source_data/
?? latex/main_compressed.pdf
?? output/
?? tools/
```

无 Python source 处于已修改状态。

---

## 2. Runtime compatibility

```text
LOCAL_RUNTIME_COMPATIBLE = NO
```

仓库 `.venv` 是 Linux x86_64 解释器，不能在 Darwin arm64 上执行。按本阶段规则：不修复、不污染原 `.venv`。静态代码审计继续完成。本机无法运行本审计第 8 节所列测试。

---

## 3. Verified P0 bugs

下列问题已由源码路径闭合，不依赖运行时数值实验。它们改变 environment / algorithm semantics，且要求 canonical retraining。

### ENV-01 — action 与当期 flight energy / reward 错位；terminal action 飞行能耗丢失

* 判定：`VERIFIED`
* 文件 / 函数：`environment/env.py` `Env.step()`、`Env._apply_actions_to_env()`、`Env._get_rewards_and_metrics()`；`environment/uavs.py` `UAV.update_energy_consumption()`、`UAV.reset_for_next_step()`、`UAV.update_position()`
* 真实顺序（`Env.step()`）：

```text
1. request processing     env.py:124-128  process_requests()     在当前位置 p_t
2. energy update          env.py:140      update_energy_consumption()
3. reward calculation     env.py:142      _get_rewards_and_metrics()
4. GDSF (every 50 steps)  env.py:147-150
5. UE movement            env.py:153-154
6. UAV slot reset         env.py:157-158  清空 energy / collision / boundary
7. movement               env.py:160      _apply_actions_to_env(actions)  应用 a_t
8. next obs               env.py:162      含 generate_request + association
```

飞行能耗使用的是上一时步留下的 `_dist_moved`：

```python
# environment/uavs.py:904-907
time_moving: float = self._dist_moved / config.UAV_SPEED
time_hovering: float = config.TIME_SLOT_DURATION - time_moving
fly_energy: float = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
self._energy_current_slot += fly_energy
```

`reset_for_next_step()` 不清空 `_dist_moved`，但会在 `apply_actions` 之前清掉 `collision_violation` / `boundary_violation` / `_energy_current_slot`。随后 `_apply_actions_to_env()` 才写入本期位移与碰撞标记。因此：

* 时步 \(t\) 的 reward / `metrics["energy"]` 结算的是 \(a_{t-1}\) 的飞行与碰撞，不是 \(a_t\)。
* episode 第 1 步 `_dist_moved = 0`，飞行项被算成纯悬停。
* episode 最后一步 \(T=1000\)：`run_hierarchical_mappo_experiment.py:255` 在 `step()` 返回后置 `done`；`a_T` 已写入位置，但再也没有下一次 `update_energy_consumption()`。terminal action 的 flight energy 与碰撞惩罚丢失。

* Training impact：破坏 \((s_t,a_t,r_t,s_{t+1})\) 对齐；PPO 用延迟 1 步且含上一步运动结果的奖励更新轨迹策略。
* Evaluation impact：每回合系统少计 1 个时隙飞行能耗；能量、能效、奖励均系统性偏差。
* 改变 environment semantics：是。
* Canonical retraining：是。

### ENV-02 — collision correction 后位移可超过 \(v_{\max}\tau\)，且 `time_hovering` 可为负

* 判定：`VERIFIED`
* 文件 / 函数：`environment/env.py` `Env._apply_actions_to_env()`；`environment/uavs.py` `UAV.update_energy_consumption()`；论文约束 `latex/docs/chap03.tex` 式 (constraint-speed)

提议位移先被限制在 `max_dist = UAV_SPEED * TIME_SLOT_DURATION`（15 m）。随后碰撞排斥直接改 `next_positions`，只再做边界 clip，没有二次速度投影：

```python
# environment/env.py:348, 388-405
max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION
...
overlap: float = config.MIN_UAV_SEPARATION - dist
next_positions[i] += direction * overlap * 0.5
next_positions[j] -= direction * overlap * 0.5
...
final_positions = np.clip(next_positions, [min_boundary_gap, min_boundary_gap], [...])
```

`MIN_UAV_SEPARATION = 200` m。两机提议位置一旦进入 200 m 内，单次排斥即可把每机再推数十米。`update_position()` 用欧氏位移写 `_dist_moved`，能量公式不做 `time_hovering` 截断。因此可以同时出现：

* `distance > UAV_SPEED * TIME_SLOT_DURATION`
* `time_hovering < 0`，负悬停功率从飞行能耗中倒扣

这与论文硬约束 \(\|\mathbf{q}_{n+1}-\mathbf{q}_n\| \le v_{\max}\tau\) 不一致。正文只写“通过动作幅值限制、碰撞检测与位置修正近似执行”。

* Training impact：策略可利用排斥“瞬移”，并因负悬停而得到虚假低能耗。
* Evaluation impact：运动学与能耗非物理。
* 改变 environment semantics：是。
* Canonical retraining：是。

### ENV-03 — 跨 UAV cooperative 写入的 `_working_cache` 可被目标 UAV 自己覆盖

* 判定：`VERIFIED`
* 文件 / 函数：`environment/env.py` `Env.step()`；`environment/uavs.py` `UAV.process_requests()`、`_try_add_file_to_cache()`、`_process_service_request()` / `_process_content_request()`

```python
# env.py:124-128  按 UAV id 升序
for uav_idx, uav in enumerate(self._uavs):
    uav.process_requests(...)

# uavs.py:361
self._working_cache = self.cache.copy()

# uavs.py:79-88
if used_space + FILE_SIZES[file_id] <= UAV_STORAGE_CAPACITY[uav.id]:
    uav._working_cache[file_id] = True
```

UAV \(i\) 把请求协作给 UAV \(j>i\) 时，会改 `uavs[j]._working_cache`。随后 UAV \(j\) 进入 `process_requests()` 第一行，用自己的 `cache.copy()` 覆盖该写入。反向协作 \(j\to i\)（\(i\) 已处理完）则保留。缓存演化依赖 Python 列表下标，不是对称物理过程。

命中判定用的是 `target_uav.cache` 而不是 `_working_cache`（`uavs.py:781, 801, 832, 850`）。同一时隙内新写入的 working cache 既可能被后处理 UAV 抹掉，也不会让同槽后续请求立刻命中。

* Training / evaluation impact：协作缓存收益非对称、顺序依赖。
* 改变 environment semantics：是。
* Canonical retraining：是。

---

## 4. Verified P1 issues

### ENV-04 — heuristic 与 learned/external 使用不同 request order

* 判定：`VERIFIED`
* 文件 / 函数：`environment/uavs.py` `UAV.process_requests()`；`environment/env.py` `Env.step()`、`get_offloading_obs_and_masks()`

| 路径 | 顺序 |
| --- | --- |
| Heuristic（`offload_actions is None`） | `np.random.permutation(len(current_covered_ues))`：服务 / 内容 / 能量请求混洗 |
| Learned / external | 服务请求按 UE–UAV 二维距离升序，然后才是非服务请求 |
| Lower-visible observation | 同样按距离取前 `MAX_OFFLOAD_REQUESTS_PER_UAV` 个服务 UE |
| Global UAV processing | 固定 `UAV 0 → 1 → 2 → 3 → 4` |

`UncoordinatedGreedyModel` 只输出轨迹，卸载仍走 heuristic 分支，因此基线对比混入“随机顺序 vs 距离优先”。由于 C2 队列在循环中变化，顺序会改变时延、能耗和卸载统计。

* Training / evaluation impact：基线与学习策略的 intra-slot 调度协议不对等。
* 改变 environment semantics：是（对比实验协议）。
* Canonical retraining：若统一顺序或改为批处理共享，是。

### ENV-05 — C2 processor-sharing 随 sequential processing 动态变化

见第 5 节。此处只确认行为。

```text
C2_BEHAVIOR_VERIFIED = YES
```

### ENV-06 — UE–UAV 带宽为 `BANDWIDTH_EDGE / num_associated_ues`

* 判定：`VERIFIED`
* 文件 / 函数：`environment/comm_model.py` `calculate_ue_uav_rate()`；调用点 `uavs.py:388-391`、`env.py:206-209`

```python
bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues
```

`num_associated_ues = len(uav.current_covered_ues)`，含内容 / 能量 / 强制接入的服务 UE，且 association 没有 `MAX_ASSOCIATED_UES` 硬上限。

* 依赖 trajectory / association：是。覆盖集合一变，接入带宽和信道增益都变。
* 固定 Ch4 snapshot 下是否随 offloading assignment 改变：否。时隙内带宽不随 Local / MBS / Coop 改变。
* 与 C2 的区别：接入带宽是 association 静态切片；C2 是循环内动态 \(F_j/n_j\)。二者不是同一种 coupling。
* Canonical retraining：当前实现与公式一致，单独此项不强制重训。

### ENV-07 — 论文统计的能量边界

* 判定：`VERIFIED`
* 文件 / 函数：`env.py` `_get_rewards_and_metrics()` `total_energy = sum(uav.energy)`；累加点见 `uavs.py`

| 分量 | 进入 `uav._energy_current_slot` / `total_energy` |
| --- | --- |
| UAV flight (move) | 是（受 ENV-01/02 污染） |
| UAV hover | 是（可为负，ENV-02） |
| WPT transmit `P_wpt * τ`（覆盖内存在能量请求时整槽） | 是 |
| UAV computing `K_cpu C f^2`（本地或协作目标机） | 是 |
| UAV comm TX/RX、backhaul TX/RX | 是 |
| MBS computing energy | 否 |
| UE static / TX / RX / harvested energy | 否（只改 UE 电池；通过 `offline_rate` 间接进奖励） |

论文真正统计的是 **UAV fleet 侧总能耗** \(E_{total}=\sum_i(E_{fly}+E_{comp}+E_{comm}+E_{wpt})\)，与 `chap03.tex` 式 (total-energy) 一致。MBS 与 UE 不在该边界内。

### ENV-08 — `offline_rate` 统计对象

* 判定：`VERIFIED`
* 文件 / 函数：`env.py:465-466`；阈值 `config.UE_CRITICAL_THRESHOLD = 0.1 * 500 = 50 J`

```python
offline_count: int = sum(1 for ue in self._ues if ue.battery_level < config.UE_CRITICAL_THRESHOLD)
offline_rate: float = offline_count / config.NUM_UES
```

统计对象是全体 UE（基数 100），不是 UAV。低电量 UE 在下一时隙 `generate_request()` 中改为能量请求。

### LOWER-01 — factorized request log-probability 是均值不是联合和

* 判定：`VERIFIED`
* 文件 / 函数：`marl_models/offload_mappo/offload_mappo.py` `get_action_and_value()` / `update()`；`marl_models/joint_mappo/joint_mappo.py` 卸载分支相同

当前实现是 **`sum(log π_m) / valid_count`**，不是 `sum(log π_m)`：

```python
action_log_probs = dist.log_prob(actions) * slot_valid
denom = slot_valid.sum(dim=-1).clamp_min(1.0)
log_probs = action_log_probs.sum(dim=-1) / denom
ratio = torch.exp(new_log_probs - old_log_probs)
```

PPO ratio 的实际形式：

\[
\rho = \left(\prod_{m\in\mathcal{V}} \frac{\pi_\theta(a_m\mid s)}{\pi_{\theta_{\mathrm{old}}}(a_m\mid s)}\right)^{1/|\mathcal{V}|}
\]

即有效请求重要性比率的几何平均，而不是联合策略比率 \(\prod_m \pi_\theta/\pi_{\mathrm{old}}\)。`valid_slots` 来自 mask 行和 > 0。Joint MAPPO 对卸载头使用同一均值，再与轨迹 `sum(log π)` 相加。

这不一定是实现错误，但若论文写成标准因子化联合 MAPPO，则理论与代码不一致。改公式必须重训。

### LOWER-02 — lower energy 项包含 trajectory flight energy

* 判定：`VERIFIED`
* 文件 / 函数：`run_hierarchical_mappo_experiment.py` `lower_rewards_from_metrics()`；论文 `chap03.tex` 式 (lower-reward)

```python
energy_term = float(metrics["energy"]) / (config.NUM_UAVS * config.OFFLOAD_ENERGY_NORM_REF + config.EPSILON)
...
- config.OFFLOAD_REWARD_ENERGY_WEIGHT * energy_term
```

`metrics["energy"]` 是 ENV-07 的 UAV fleet 总能耗，含飞行 / 悬停 / WPT / 计算 / 通信，且飞行项还错位一期。下层动作不能控制轨迹。论文公式同样使用 \(E/(N\cdot E_{ref}^{offload})\)，因此代码与现有正文一致，但 credit assignment 把外生飞行能耗灌进卸载策略。若修复期剥离飞行项，必须改论文公式并重训。

### LOWER-03 — lower reward 范围

* 判定：`VERIFIED`

```text
LOWER_REWARD_SCOPE = GLOBAL_TEAM
```

```python
return [float(lower_reward)] * len(system_rewards), diagnostics
```

全体下层 agent 拿同一标量。这不是自动 bug。含义：全局合作、无 per-UAV 归因，存在 credit dilution / lazy agent。上层基础奖励也是全局标量，但额外按机减去 collision / boundary penalty；下层连碰撞都没有个体项。

### LOWER-04 — request truncation 与 fallback

* 判定：`VERIFIED`

流转：

1. `generated_service_requests`：全体 UE 本槽服务请求（低电量者改为能量请求，不计入）。
2. `lower-visible requests`：每机按距离排序后截断到 `MAX_OFFLOAD_REQUESTS_PER_UAV`（当前 = 30）。association 本身无 30 上限，超出部分对下层不可见。
3. `learned decisions`：`service_action_cursor < 30` 时消费 `offload_actions[cursor]`。协作目标不在 `neighbors` 时 fallback 到 Local 或 MBS（`uavs.py:466-469`）。非法离散值 fallback 到 heuristic。
4. `heuristic fallback`：第 31 个及以后的服务请求走 `_select_service_offloading_target()`；旧 classifier 路径失败同样回退 heuristic。

观测构造与执行截断都按距离，但 heuristic-only 整表随机，截断协议仍不对齐。

### LOWER-05 — cooperative action 指向特定 UAV

* 判定：`VERIFIED`（分层 MAPPO / 环境执行路径）
* 文件：`config.py:186-189`；`env.py` mask `OFFLOAD_ACTION_COOP_BASE + neighbor.id`；`uavs.py:_resolve_external_service_offload_action()`

动作是 `{0: Local, 1: MBS, 2+j: Coop-UAV-j}`，\(j\in\{0,\ldots,N-1\}\)，共 \(N+2=7\)。不是抽象 Coop 类。旧三分类 classifier（`offload_policy.py` 的 `OFFLOAD_TARGET_COOPERATIVE=1`）仍是抽象类，且只在 `SERVICE_OFFLOAD_POLICY=="learned"` 时使用；分层 MAPPO 不走该路径。

### 其它 P1（缓存 / 关联，非单独编号）

* 执行命中看 `cache`，写入看 `_working_cache`，同槽后请求不能立刻命中刚拉取的文件。
* `_associate_ues_to_uavs()` 不限制每机关联数；观测与下层动作只看见最近 30 个，带宽却按全部关联 UE 均分。
* `get_offloading_obs_and_masks()` 用初始队列估计协作质量；`process_requests()` 执行时队列已随循环改变。

---

## 5. C2 behavior vs scientific-model distinction

代码行为与科学模型必须分开。

### 行为（只描述实现）

```text
C2_BEHAVIOR_VERIFIED = YES
```

证据：

```python
# uavs.py:68-73
computing_capacity_per_request = UAV_COMPUTING_CAPACITY[uav.id] / uav._current_service_request_count
latency = cpu_cycles / computing_capacity_per_request
energy = K_CPU * cpu_cycles * (computing_capacity_per_request ** 2)
```

```python
# uavs.py:431-436  Optimistic relief
if current_req.is_service and best_target_idx != OFFLOAD_TARGET_LOCAL:
    self._current_service_request_count = max(0, self._current_service_request_count - 1)
    if best_target_idx == OFFLOAD_TARGET_COOPERATIVE and best_target_uav is not None:
        best_target_uav._current_service_request_count += 1
```

时隙开始 `calculate_initial_load()` 把关联服务请求全部计入 \(n_j\)。之后每处理一个外迁请求，\(n_j\) 立刻减 1；协作则立刻增加目标机 \(n_k\)。本地执行不减 \(n_j\)。因此 \(F_j/n_j\) 随 Python 顺序变化，且本地 / 外迁更新规则不对称。

### 科学模型门

```text
C2_MODEL_JUSTIFIED = NO
```

| 过关条件 | 事实 | 结果 |
| --- | --- | --- |
| 1. 明确 intra-slot sequential admission/service semantics | 请求在 `_get_obs()` 时同时生成，无到达时间戳、无子时隙时钟 | 不满足 |
| 2. queue update 与该语义一致 | 外迁立刻减队列，本地完成不减；协作立刻加邻居 | 不满足 |
| 3. processor share 生效时点有物理定义 | 生效点是 `for idx in shuffled_indices` 的迭代瞬间 | 不满足 |
| 4. execution order 不是 list/index 偶然产物 | UAV 按下标；heuristic 随机；learned 按距离 | 不满足 |
| 5. 论文可明确描述该系统模型 | `chap03.tex:107` 写“当多个服务请求同时由同一 UAV 处理时，计算资源在请求之间共享”，未写循环快照 / optimistic relief | 不满足 |

不能把 C2 写成已经过关的连续处理器共享模型。修复前必须先选定批处理等额共享或显式微观排队，再改代码与正文。

---

## 6. Cache information model

四类状态：

1. **actual cache** `uav.cache`：布尔向量，物理命中与上层观测使用的真实缓存。
2. **`_working_cache`**：时隙内尝试写入的草稿；`update_ema_and_cache()` 将其 copy 回 `cache`。
3. **cache belief** `_get_belief_probability(file_id, neighbor_id)`：由 Zipf rank 与邻居容量构造的 sigmoid，不读邻居真实 cache。
4. **GDSF periodic refresh**：`T_CACHE_UPDATE_INTERVAL = 50`。用 `_ema_scores / FILE_SIZES` 重装整个 cache。

问答：

* Heuristic / policy **本地**决策：actual cache（`self.cache[req_id]`）。
* Heuristic / policy **协作**决策：belief，不查询邻居 actual cache。
* Environment **执行**：actual cache 判命中；未命中则真实回传并写入 `_working_cache`。不对 belief 采样。
* GDSF：每 50 step 一次；另有每步 reactive `cache = working_cache.copy()`。
* 若构建冻结 snapshot 卸载评测，至少冻结：`cache`、`_working_cache`、`_ema_scores`、`_freq_counts`、位置、association、`_current_service_request_count`。仓库当前没有独立 Ch4 frozen-snapshot evaluator；`utils/plot_snapshots.py` 只画可视化。历史结果不可当作该评测器已经存在。

ENV-03 使 working-cache 跨机更新在后处理 UAV 上无效。

---

## 7. FORCE_SERVICE_ADMISSION

```text
FORCE_SERVICE_ADMISSION_CANONICAL = UNRESOLVED
```

| 来源 | 值 |
| --- | --- |
| `config.py:81` 默认 | `False` |
| `run_hierarchical_mappo_experiment.py` CLI | `store_true`，不加 flag 则为 False |
| `run_baseline_matrix_v2.py` / `run_baseline_comparison_experiment.py` | 默认 False，除非显式 flag |
| `run_sensitivity_full.py`、force-admission 脚本 / runbook | `True` |
| 历史矩阵 `results/baseline_matrix_force_admission_*`、`results/final_force_admission/evaluation_20260630` | `true` |
| 论文 `abstract.tex`、`chap01.tex`、`chap03.tex:18`、`chap06.tex` | 描述的是 True：覆盖外服务请求关联最近 UAV，真实距离算链路，不保证 DSR |

语义：

* `False`：覆盖外服务请求不接入，`assigned=False`，时延罚 `NON_SERVED_LATENCY_PENALTY=20`，DSR 计 0。内容 / 能量请求同样不强制接入。
* `True`：仅服务请求强制接入最近 UAV；`processed_request_ratio` 可到 1；100 m 覆盖半径不再是服务准入硬约束，仍是信道距离的物理输入。

没有充分科学依据在本阶段替作者选定 TRUE 或 FALSE。canonical 重训前必须先统一 config 默认值、训练脚本、评估协议和正文。历史 force-admission 数字全部 `HISTORICAL_ONLY`。

---

## 8. Required code fixes

在作者裁定 FORCE 与 C2 模型之前，下列修复规格已经可以从代码闭合。本阶段不实施。

1. **FIX-ENV-01**：重排 `Env.step()`，使 \(a_t\) 的位移、飞行能耗、碰撞 / 越界与当期 reward/metrics 对齐；保证 terminal step 计入 \(a_T\)。
2. **FIX-ENV-02**：碰撞修正后投影到 \(\|\Delta\mathbf{q}\|\le v_{\max}\tau\)；`time_hovering = max(0, τ - t_moving)`。
3. **FIX-ENV-03**：所有 UAV 在处理请求前并行初始化 `_working_cache`；禁止后处理 UAV 覆盖先到协作写入。明确同槽命中读 `cache` 还是 `working_cache`。
4. **FIX-ENV-04**：heuristic 与 learned 使用同一 request order；UAV 循环顺序不得成为物理状态源。
5. **FIX-C2**：按第 5 节选定的科学模型改 queue / share 更新；禁止继续用“循环副作用”充当处理器共享。
6. **FIX-LOWER-01**：在正文与代码之间二选一：标准 \(\sum\log\pi_m\)，或显式写成 geometric-mean multi-action PPO。
7. **FIX-LOWER-02**：若下层只控制卸载，能量项应去掉飞行 / 悬停（并同步改 `eq:lower-reward`）；若保留全局 \(E\)，必须在正文写明这是 team energy 而非卸载因果能耗。
8. **FIX-FORCE**：config、CLI 默认、训练 / 评估脚本、LaTeX 使用同一 canonical 值。
9. **FIX-ASSOC-CAP**（建议）：明确关联集合是否硬截断到 30；带宽分母与可见请求集合必须定义一致。

---

## 9. Required tests

本机当前不能跑。修复阶段需要在兼容 runtime（Linux x86_64 原环境，或新建 Darwin arm64 隔离环境，禁止改原 `.venv`）上至少覆盖：

1. `test_env_temporal_alignment`：\(a_t\) 的飞行能耗与碰撞出现在 step \(t\)；第 \(T\) 步能耗非零且计入。
2. `test_collision_kinematics_boundary`：对心碰撞后位移 \(\le v_{\max}\tau\)，`time_hovering >= 0`。
3. `test_cooperative_cache_symmetry`：UAV0→UAV4 与 UAV4→UAV0 的 working-cache 提交对称。
4. `test_request_order_parity`：heuristic 与 external 路径同一关联集合、同一处理顺序。
5. `test_c2_share_schedule`：按选定模型断言 \(n_j\) 与 \(F_j/n_j\) 的更新时刻。
6. `test_offload_logprob_contract`：锁定 `sum` 或 `mean`，并检查 PPO ratio 数值。
7. `test_lower_energy_terms`：下层 reward 是否含 / 不含飞行能耗，与规格一致。
8. `test_force_admission_semantics`：True/False 下 uncovered 服务请求的 assigned、latency penalty、DSR 分母。
9. `test_bandwidth_invariant_to_offload`：冻结 association 后改卸载动作，UE–UAV 带宽不变。

---

## 10. Retraining impact

* 全部历史 checkpoint / JSON / 图 / 表：`HISTORICAL_ONLY`。
* ENV-01/02/03 任一修复都会改变 MDP 与能耗，**必须** canonical retraining。
* 统一 request order、C2 批处理化、lower energy 剥离、FORCE 切换，同样必须重训。
* LOWER-01 若只改论文表述、不改代码，可不因该项单独重训；若改 logprob 合同，必须重训。
* 重训范围：分层主模型、下层消融、Joint / Vanilla / IPPO / greedy 基线、敏感性、统计检验与论文图。旧数字不得直接写入终稿。

---

## 11. Unresolved questions

1. `FORCE_SERVICE_ADMISSION_CANONICAL`：正文与正式 force-admission 实验为 True；`config.py` 与无 flag 训练入口为 False。本审计不代选。
2. C2 科学模型：批处理等额共享 \(f=F_i/N_i^{\mathrm{assigned}}\)，还是带时间戳的微观排队。当前实现不能直接当作已论证模型。
3. Lower PPO 合同：几何平均是否升格为论文方法，还是改回联合 log-prob。
4. Lower energy：保留论文现写的全局 \(E\)，还是改为卸载因果能耗。
5. 兼容实验 runtime：本机 Darwin arm64 不能用现有 `.venv`；canonical 重训应在可复现的 Linux x86_64 环境，或全新隔离 arm64 环境。
6. 关联集合硬上限与带宽分母是否应为同一集合。

---

## 12. Recommended repair order

1. 裁定未决项：FORCE canonical 值；C2 模型；lower logprob 合同；lower energy 边界。
2. 准备兼容 runtime（不改原 `.venv`）。
3. RED 测试：ENV-01/02/03，再补 ENV-04 与选定后的 C2。
4. 修环境物理层：时序、碰撞投影、缓存对称。
5. 修 C2 / request order，使执行顺序不再是 Python 下标事故。
6. 对齐 FORCE 与论文。
7. 修下层 MAPPO 合同与奖励（若裁定要求改代码）。
8. GREEN 测试与短烟雾评估。
9. 丢弃历史数字，canonical 重训 / 重评 / 重统计。

---

```text
C2_BEHAVIOR_VERIFIED = YES
C2_MODEL_JUSTIFIED = NO
FORCE_SERVICE_ADMISSION_CANONICAL = UNRESOLVED
LOCAL_RUNTIME_COMPATIBLE = NO
STATUS = BLOCKED
```

阻塞原因（任一未解除则不能开始统一修复与 canonical 重训）：

1. FORCE 的论文口径尚未由作者裁定。
2. C2 只有代码行为，没有可写入论文的正当系统模型。
3. 本宿主不能运行验证测试，也不能用原 `.venv` 做修复后回归。

P0 代码缺陷（ENV-01/02/03）规格已闭合，但不解除上述三门。本阶段 STOP，不开始修代码。
