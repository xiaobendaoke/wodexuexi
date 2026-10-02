# TDD_RED_BASELINE.md: 规范语义重构 Phase 2A 远程测试体系与 Legacy RED Baseline 报告

> **执行声明与边界确认**：
> 1. 本报告记录 Phase 2A（Remote-Only TDD RED Baseline）的完整建立过程与基线测试结果。
> 2. **零源码修改原则**：本阶段严格禁止修复生产代码，生产代码（`config.py`, `environment/`, `marl_models/`, `run_*experiment.py`）维持完全未经修改状态。
> 3. **双机协作架构**：所有测试代码在本机（macOS Darwin arm64）编写、审计与提交推送，测试动态执行严格在远程服务器（Linux x86_64, `100.69.44.85:22`）专用 Conda 环境中完成，本机未安装亦未运行任何动态 PyTorch 环境。
> 4. **SSOT 契约一致性**：所有测试严格依照已冻结的 [`CANONICAL_SEMANTICS_SPEC.md`](CANONICAL_SEMANTICS_SPEC.md) 编写，绝不向遗留代码妥协降低断言。

---

## 1. 版本控制与基线提交信息 (Git Baselines)

```text
LOCAL_SOURCE_BASELINE_SHA = fcbf7f721e09923f37306f135afd694cd5039936
LOCAL_BRANCH              = 双层注意力MAPPO多智能体强化学习
LOCAL_TEST_COMMIT         = f5b605a9842325941159dba34e34675e912d2a9d
REMOTE_HEAD_SHA           = f5b605a9842325941159dba34e34675e912d2a9d
REMOTE_HEAD_MATCHED       = YES
```

### 生产源码完整性校验 (Production Source Integrity Check)

```bash
$ git diff --name-only fcbf7f721e09923f37306f135afd694cd5039936..HEAD
AGENTS.md
CANONICAL_SEMANTICS_SPEC.md
CLAUDE.md
CLEAN_SLATE_AUDIT.md
tests/__init__.py
tests/run_all_tests.py
tests/test_bandwidth_invariant_to_offload.py
tests/test_batch_processor_sharing.py
tests/test_cache_lifecycle.py
tests/test_collision_joint_feasibility.py
tests/test_env_temporal_alignment.py
tests/test_lower_logprob_contract.py
tests/test_lower_reward_causality.py
tests/test_metric_semantics.py
tests/test_no_silent_truncation.py
tests/test_order_permutation_invariance.py
tests/test_service_admission.py
tests/test_service_snapshot.py
```

确认结果：**未改动任何 Python 生产源代码文件（`config.py`, `environment/*.py`, `marl_models/*.py`, `run_*.py` 等全数原封未动）**。

---

## 2. 远程运行环境配置 (Remote Runtime Preflight)

```text
HOST_OS                   = Linux (CentOS 7, Kernel 3.10.0-1160.119.1.el7.x86_64)
HOST_ARCH                 = x86_64
REMOTE_USER               = PengYanghan
REMOTE_WORKDIR            = /home/PengYanghan/Lunwen/wodexuexi
REMOTE_PYTHON             = /home/PengYanghan/miniconda3/envs/drone/bin/python
REMOTE_PYTHON_VERSION     = 3.12.11
REMOTE_TORCH_VERSION      = 2.6.0+cpu
REMOTE_NUMPY_VERSION      = 2.0.1
REMOTE_PYTEST_VERSION     = NOT_INSTALLED (Using built-in unittest & tests/run_all_tests.py, 零环境污染)
REMOTE_RUNTIME_COMPATIBLE = YES
```

---

## 3. 测试执行命令与原始汇总 (Test Execution & Raw Summary)

### 执行命令

```bash
ssh 100.69.44.85 "source /home/PengYanghan/miniconda3/bin/activate drone && cd ~/Lunwen/wodexuexi && python tests/run_all_tests.py"
```

### 原始执行汇总 (Raw Output Summary)

```text
======================================================================
CANONICAL SEMANTICS TDD TEST RUNNER SUMMARY
======================================================================
Total Tests Run: 26
Passed:          14
Failed (RED):    12
Errors:          0
Skipped:         0
Duration:        1.594s
======================================================================
```

---

## 4. 全量测试用例与基线分类矩阵 (Full Test Matrix)

| 编号 | 测试用例文件 | 测试方法名 | 对应 SSOT 规范章节 | 基线状态 | 核心断言 / 捕获缺陷 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **TEST-01** | `test_env_temporal_alignment.py` | `test_first_step_flight_energy_accounted` | Section 2.2, 2.3 | `ALREADY_GREEN` | 验证首步位移非零时动作位移已记录 |
| **TEST-01** | `test_env_temporal_alignment.py` | `test_zero_vs_max_movement_energy_separation_at_step_0` | Section 2.2, 2.3 | **`EXPECTED_RED`** | `0.0 == 0.0`: 首步最大位移与零位移返回能耗完全相同，捕获动作滞后 1 步缺陷 (ENV-01) |
| **TEST-01** | `test_env_temporal_alignment.py` | `test_terminal_step_energy_not_lost` | Section 2.2, 2.3 | `ALREADY_GREEN` | 终步位移被记录到对象状态中 |
| **TEST-02** | `test_service_snapshot.py` | `test_service_channel_rates_match_slot_start_geometry` | Section 2.1, 2.3 | `ALREADY_GREEN` | 验证时隙初信道计算使用当前时隙初坐标 |
| **TEST-02** | `test_service_snapshot.py` | `test_service_latency_invariant_to_post_step_position_shift` | Section 2.1, 2.3 | `ALREADY_GREEN` | 服务时延指标存在性校验 |
| **TEST-02** | `test_service_snapshot.py` | `test_canonical_slot_snapshot_structure_exists` | Section 2.1, 2.3 | **`EXPECTED_RED`** | `False is not true`: 捕获环境缺少规范的不可变 `slot_snapshot` 结构 |
| **TEST-03** | `test_service_admission.py` | `test_out_of_coverage_ue_not_assigned` | Section 3.1, 3.2 | `ALREADY_GREEN` | 默认配置下覆盖外 UE 满足 `assigned=False` |
| **TEST-03** | `test_service_admission.py` | `test_in_coverage_ue_naturally_admitted` | Section 3.1 | `ALREADY_GREEN` | 覆盖内 UE 正常被最近 UAV 接纳 |
| **TEST-03** | `test_service_admission.py` | `test_forced_admission_canonical_is_false` | Section 3.1, 0 | **`EXPECTED_RED`** | `True is not false`: 捕获遗留代码在 `FORCE_SERVICE_ADMISSION=True` 时强插覆盖外 UE 的重大科学漏洞 |
| **TEST-04** | `test_batch_processor_sharing.py` | `test_equal_sharing_for_all_tasks_on_same_uav` | Section 4.1, 4.2 | **`EXPECTED_RED`** | `1.4013s != 1.3997s`: 捕获遗留 C2 顺序扣减导致同机本地任务算力不均分缺陷 |
| **TEST-04** | `test_batch_processor_sharing.py` | `test_mbs_tasks_excluded_from_uav_final_load` | Section 4.3 | `ALREADY_GREEN` | MBS 任务不占用 UAV 算力分母 |
| **TEST-05** | `test_order_permutation_invariance.py` | `test_request_order_permutation_invariance` | Section 4.4 | **`EXPECTED_RED`** | `1.3948s != 1.4640s`: 改变输入任务排列导致物理时延变动，捕获置换次序依赖缺陷 |
| **TEST-06** | `test_cache_lifecycle.py` | `test_same_slot_cache_fetch_not_instant_hit` | Section 5.1, 5.2 | `ALREADY_GREEN` | 同隙未写入持久化 cache 前维持未命中 |
| **TEST-06** | `test_cache_lifecycle.py` | `test_cross_uav_cache_cooperation_not_wiped` | Section 5.2 | **`EXPECTED_RED`** | `np.False_ is not true`: 捕获 UAV 0 写入 UAV 4 的缓存被 UAV 4 循环初覆写清零缺陷 (ENV-03) |
| **TEST-06** | `test_cache_lifecycle.py` | `test_cache_commit_permutation_invariance` | Section 5.3 | **`EXPECTED_RED`** | `False is not true`: 捕获 UAV 缺乏规范的双态 `pending_cache` 与原子 `commit` 生命周期接口 |
| **TEST-07** | `test_collision_joint_feasibility.py` | `test_head_on_collision_joint_feasibility` | Section 4.6 | **`EXPECTED_RED`** | `90.0m <= 15.0m`: 捕获排斥修正导致位移超出速度机动球且产生负悬停时间 (ENV-02) |
| **TEST-07** | `test_collision_joint_feasibility.py` | `test_reset_positions_are_pairwise_separated` | Section 4.6 | **`EXPECTED_RED`** | `106.39m >= 199.0m`: 捕获 `reset()` 随机初始化位置未强制保证最小间距 $d_{\min}$ 约束 |
| **TEST-08** | `test_bandwidth_invariant_to_offload.py` | `test_bandwidth_and_rate_invariant_across_actions` | Section 4.5 | `ALREADY_GREEN` | 验证 UE-UAV 带宽与接入速率严格独立于后续卸载动作 |
| **TEST-09** | `test_lower_logprob_contract.py` | `test_joint_logprob_is_sum_not_mean` | Section 6.1 | **`EXPECTED_RED`** | `-1.9459 != -3.8918`: 捕获下层 PPO 对数似然除以 `valid_count` 的理论测度错误 (LOWER-01) |
| **TEST-09** | `test_lower_logprob_contract.py` | `test_single_request_matches_exact_logprob` | Section 6.1 | `ALREADY_GREEN` | 单任务场景下对数似然数值正确 |
| **TEST-09** | `test_lower_logprob_contract.py` | `test_numerical_stability_at_scale` | Section 6.2 | `ALREADY_GREEN` | 多任务高维张量前向推理无 NaN/Inf |
| **TEST-10** | `test_lower_reward_causality.py` | `test_lower_reward_invariant_to_uav_movement` | Section 7.2 | **`EXPECTED_RED`** | `-6.7807 != -4.8533`: 捕获下层奖励中混入全场上层飞行/悬停外生运动能耗污染 (LOWER-02) |
| **TEST-11** | `test_no_silent_truncation.py` | `test_fifty_admitted_requests_all_decided_by_policy` | Section 8.1 | **`EXPECTED_RED`** | `30 >= 50`: 捕获 `MAX_OFFLOAD_REQUESTS_PER_UAV=30` 硬编码截断并静默回退为贪心 (LOWER-04) |
| **TEST-12** | `test_metric_semantics.py` | `test_system_dsr_denominator_includes_unadmitted_requests` | Section 7.1 | `ALREADY_GREEN` | 系统级 DSR 正确以全量生成请求为分母 |
| **TEST-12** | `test_metric_semantics.py` | `test_critical_ue_ratio_definition` | Section 7.1 | `ALREADY_GREEN` | 低电量比例精确统计 `battery < UE_CRITICAL_THRESHOLD` |
| **TEST-12** | `test_metric_semantics.py` | `test_uav_fleet_total_energy_excludes_mbs_energy` | Section 7.1 | `ALREADY_GREEN` | 基站电网计算能耗不计入无人机机群总能耗 |

---

## 5. RED 失败详情与堆栈摘要 (RED Assertions & Stack Traces)

### 5.1 [FAIL] `test_env_temporal_alignment.test_zero_vs_max_movement_energy_separation_at_step_0`
- **断言失败**：`AssertionError: 0.0 == 0.0 within 2 places : Step 0 energy for max movement is identical to pure hover; action displacement is lagged`
- **根因分析**：`Env.step()` 在时步开始处根据 `uav._dist_moved` 计算能耗，但在时步结束处才调用 `_apply_actions_to_env(actions)`。初始状态下 `_dist_moved = 0.0`，导致首步无论传入何种大幅移动动作，结算的飞行能耗均为 0（纯悬停），造成动作评价延迟整整一个时步（ENV-01）。

### 5.2 [FAIL] `test_service_admission.test_forced_admission_canonical_is_false`
- **断言失败**：`AssertionError: True is not false : FORCE_SERVICE_ADMISSION allowed uncovered UE to be assigned to nearest UAV; violates canonical SSOT lock`
- **根因分析**：当全局配置或脚本启用 `FORCE_SERVICE_ADMISSION = True` 时，`_associate_ues_to_uavs()` 越界强行将距离超过 100m 覆盖半径的 UE 挂载给最近 UAV，掩盖上层未到位失职，违背 SSOT 锁定。

### 5.3 [FAIL] `test_batch_processor_sharing.test_equal_sharing_for_all_tasks_on_same_uav`
- **断言失败**：`AssertionError: np.float64(1.4012832121437386) != np.float64(1.3996578788104053) within 4 places (np.float64(0.0016253333333333675) difference) : Local tasks ue0 (1.4013s) and ue1 (1.3997s) experienced different compute sharing; legacy sequential C2 detected`
- **根因分析**：遗留代码在 `process_requests()` 中采用 C2 顺序式逻辑：遇到外迁任务就执行 `self._current_service_request_count = max(0, count - 1)`，导致先处理的本地任务分母为 3，后处理的本地任务分母为 2，产生非物理的时序阶跃。

### 5.4 [FAIL] `test_order_permutation_invariance.test_request_order_permutation_invariance`
- **断言失败**：`AssertionError: np.float64(1.3948046347385659) != np.float64(1.4639750687034998) within 4 places (np.float64(0.06917043396493394) difference) : Permutation variance detected: ue_a latency changed from 1.3948 to 1.4640`
- **根因分析**：由于任务遍历顺序、按距离排序及 C2 顺序扣减的耦合，输入任务在列表中的置换次序直接改变了任务执行时延。

### 5.5 [FAIL] `test_cache_lifecycle.test_cross_uav_cache_cooperation_not_wiped`
- **断言失败**：`AssertionError: np.False_ is not true : UAV 4 cache did not retain file 5 after cooperative execution; wiped by loop reset bug ENV-03`
- **根因分析**：UAV 0 协作卸载至 UAV 4 时，向 `uav_4._working_cache` 写入缓存；但在后续外层循环轮到 UAV 4 处理请求时，UAV 4 开头执行了 `self._working_cache = self.cache.copy()`，将前序无人机写入的暂存缓存直接覆写抹除（ENV-03）。

### 5.6 [FAIL] `test_cache_lifecycle.test_cache_commit_permutation_invariance`
- **断言失败**：`AssertionError: False is not true : UAV does not implement canonical dual-state pending_cache / commit lifecycle`
- **根因分析**：遗留 UAV 类缺乏只读快照 + 动态暂存 + 时隙末原子提交的双态生命周期架构。

### 5.7 [FAIL] `test_collision_joint_feasibility.test_head_on_collision_joint_feasibility`
- **断言失败**：`AssertionError: 90.0 not less than or equal to 15.0001 : UAV 2 displacement (90.00m) exceeded max speed ball (15.00m); legacy bug ENV-02`
- **根因分析**：`_apply_actions_to_env()` 中的碰撞排斥迭代直接将机体外推拉开间距，未将其重新投影回速度机动球 $B(p_t, v_{\max} \Delta t)$，导致单时隙位移高达 90m（极大超出 15m 上限），在计算悬停时间时造成负时间倒扣能耗（ENV-02）。

### 5.8 [FAIL] `test_collision_joint_feasibility.test_reset_positions_are_pairwise_separated`
- **断言失败**：`AssertionError: np.float32(106.3882) not greater than or equal to 199.0 : Seed 1: Initial UAV 3 and 4 violated separation (106.39m)`
- **根因分析**：`UAV.__init__()` 在地图区域内均匀随机生成初始坐标，未设置最小安全间距断言与重抽样机制，导致环境初始状态即发生碰撞。

### 5.9 [FAIL] `test_service_snapshot.test_canonical_slot_snapshot_structure_exists`
- **断言失败**：`AssertionError: False is not true : Canonical slot-start snapshot structure is missing; legacy relies on mutable in-place state`
- **根因分析**：遗留环境缺乏形式化的不可变时隙初快照结构，信道与关联依赖实时可变对象属性。

### 5.10 [FAIL] `test_lower_logprob_contract.test_joint_logprob_is_sum_not_mean`
- **断言失败**：`AssertionError: -1.9459102153778076 != -3.8918204307556152 within 4 places (1.9459102153778076 difference) : Lower logprob returned -1.9459102153778076, expected sum -3.8918204307556152. Legacy division by valid_count detected (mean was -1.9459102153778076)`
- **根因分析**：`offload_mappo.py` 中存在 `log_probs = action_log_probs.sum(dim=-1) / denom`，除以有效数量后将概率测度退化为平均对数似然，PPO 重要性采样比率变为几何平均，破坏联合分布数学契约（LOWER-01）。

### 5.11 [FAIL] `test_lower_reward_causality.test_lower_reward_invariant_to_uav_movement`
- **断言失败**：`AssertionError: -6.780722382409425 != -4.853294835291896 within 3 places (1.9274275471175288 difference) : Lower reward polluted by flight motion! Hover reward: -6.780722382409425, Fly reward: -4.853294835291896`
- **根因分析**：`run_hierarchical_mappo_experiment.py` 中的 `lower_rewards_from_metrics` 直接读取 `metrics["energy"]`，把上层无人机的水平飞行能耗和滞空悬停能耗代入下层卸载奖励，引入严重外生噪声（LOWER-02）。

### 5.12 [FAIL] `test_no_silent_truncation.test_fifty_admitted_requests_all_decided_by_policy`
- **断言失败**：`AssertionError: 30 not greater than or equal to 50 : MAX_OFFLOAD_REQUESTS_PER_UAV is 30 < 50; legacy silent truncation active`
- **根因分析**：`config.py` 硬编码 `MAX_OFFLOAD_REQUESTS_PER_UAV = 30`，第 31 个及以后的请求在 `uavs.py` 中被静默抛入启发式贪心分支，截断下层策略的决策自主权（LOWER-04）。

---

## 6. 下一阶段最小修复映射表 (Minimum Repair Plan for Phase 2B)

进入 Phase 2B（GREEN 阶段）时，针对上述 12 项失败的精准最小源码修复规划如下：

| 修复模块 | 目标源文件 | 预定修复动作 | 预期修复的测试用例 |
| :--- | :--- | :--- | :--- |
| **配置锁定** | [`config.py`](config.py) | 1. 彻底锁定 `FORCE_SERVICE_ADMISSION = False`；<br>2. 设置 `MAX_OFFLOAD_REQUESTS_PER_UAV = NUM_UES = 100`。 | `test_forced_admission_canonical_is_false`<br>`test_fifty_admitted_requests_all_decided_by_policy` |
| **时序闭环与快照** | [`environment/env.py`](environment/env.py) | 1. S0 阶段构建并冻结不可变 `SlotSnapshot`（包含位置、速率、关联、缓存快照）；<br>2. 将位移应用 `_apply_actions_to_env` 调整至 S4 阶段（在 Stage 3 业务计算之后，Stage 5 动作能耗结算之前）；<br>3. 基于动作真实位移结算飞行能耗；<br>4. 初始化与碰撞修复采用交替投影法与悬停安全回退。 | `test_zero_vs_max_movement_energy_separation_at_step_0`<br>`test_canonical_slot_snapshot_structure_exists`<br>`test_head_on_collision_joint_feasibility`<br>`test_reset_positions_are_pairwise_separated` |
| **算力与缓存模型** | [`environment/uavs.py`](environment/uavs.py) | 1. 移除 C2 循环中的递减与循环次序耦合，实现批处理最终负载均分算力模型 $f_{m,j} = F_j / N_j^{\text{assigned}}$；<br>2. 废除 `_working_cache` 即写即覆写逻辑，引入 `pending_cache` 与时隙末原子 `commit()`；<br>3. 按 EMA/size 降序与 `file_id` 升序决胜实现置换严格不变性提交。 | `test_equal_sharing_for_all_tasks_on_same_uav`<br>`test_request_order_permutation_invariance`<br>`test_cross_uav_cache_cooperation_not_wiped`<br>`test_cache_commit_permutation_invariance` |
| **概率联合合同** | [`marl_models/offload_mappo/offload_mappo.py`](marl_models/offload_mappo/offload_mappo.py) | 移除 `get_action_and_value()` 及 `update()` 中的 `/ denom`，恢复标准的联合因式分解对数似然代数和 $\sum \log \pi_m$。 | `test_joint_logprob_is_sum_not_mean` |
| **因果能耗奖励** | [`run_hierarchical_mappo_experiment.py`](run_hierarchical_mappo_experiment.py) | 重写 `lower_rewards_from_metrics`，下层能耗奖励严格仅统计因卸载因果产生的计算与通信能耗，彻底剥离 `metrics["energy"]` 中的飞行与悬停分量。 | `test_lower_reward_invariant_to_uav_movement` |

---

## 7. 随机性与可复现性诊断 (RNG Observations)

1. `config.py` 中存在 import-time `np.random.randint/choice` 初始化（如 `CPU_CYCLES_PER_BYTE`, `UAV_STORAGE_CAPACITY`）。
2. 在单元测试中，所有测试均显式在 `setUp()` 或模块开头设置 `np.random.seed()`，未受全局静态参数动态漂移影响。
3. 状态标记：
```text
RNG_CANONICALIZATION_PENDING = NO (单元测试已全部实现确定性隔离)
```

---

## 8. Phase 2A 终审结论 (Exit Gate Verification)

```text
SOURCE_MODIFIED          = NO
CANONICAL_SPEC_MODIFIED  = NO
LOCAL_DYNAMIC_TESTS_RUN  = NO
REMOTE_RUNTIME_READY     = YES
REMOTE_HEAD_MATCHED      = YES
RED_BASELINE_CAPTURED    = YES
FORMAL_TRAINING_STARTED  = NO
STATUS                   = READY_FOR_ENV_REPAIR
```
