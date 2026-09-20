# 仿真环境复核与修复（2026-09-19）

## 范围与计划

用户授权检查并修复环境，说明可运行的实验。本轮聚焦环境物理状态转移、请求执行、缓存与运动约束，以及远程运行环境；不启动正式训练，不改写已冻结科学规范，不自动提交或推送。

基线 HEAD：`4b9652a5ae1e12cc4b34902a2c7569e2fbe96335`。本机已有文档迁移、暂存修改及未跟踪文件，全部保留。远程主仓库同 HEAD，检查时工作区干净。

1. 在远程重新运行现有测试，区分环境失败与下层策略已知失败。
2. `tests/test_global_service_batch.py`：补充跨 UAV 多源协作、目标端外迁和全局执行顺序的反例；先验证失败。
3. `environment/env.py`、`environment/uavs.py`：在全局范围先解析全部任务，再统计最终执行负载，最后统一执行。
4. 检查碰撞回退、缓存提交及快照边界；对确认问题增加回归并修复。
5. 在远程隔离目录验证，保存结果、源码指纹与实验适用边界；不覆盖远程主仓库和历史结果。

远程验证目录：`/home/PengYanghan/Lunwen/env-repair-20260919-kBaVvp`。

## 基线验证

远程运行 `python -m unittest discover -s tests -v`：26 项，23 通过、3 失败。失败是已知下层联合概率取平均、下层奖励含飞行能耗、请求容量 30。现有 21 项环境测试通过，但没有覆盖多源跨 UAV 最终负载；不能据此推断环境完全正确。

## 最终结果

**本轮发现的环境实现问题已修复，并在远程隔离副本验证通过。环境可用于功能验证和探索性实验；整个项目尚未达到正式论文实验就绪状态。**

### 修复内容

| 问题 | 修复 | 证据 |
|---|---|---|
| 每架 UAV 分别解析、执行，目标端本地负载覆盖已有协作负载，且先执行任务看不到后续协作任务 | 全场先解析全部服务任务，再统一统计 Local + Coop-in 最终负载，最后执行 | 新增 3 个生产 `Env.step` 跨 UAV 回归。修复前同一目标端 3 任务计算能耗 658.26 J，公式期望 146.28 J；修复后吻合。目标端自身转发 MBS 时也正确保留 incoming tasks |
| 下层观测的服务上下文使用上一时隙 `_uav_mbs_rate` | 在生成当前下层观测前更新所有 UAV 的当前回传速率 | 构造陈旧值 1 bit/s，修复后使用当前几何对应的 10,206,194 bit/s |
| 服务卸载到 MBS/协作节点后，源 UAV 无服务文件传输却获得该文件缓存 | 删除源端虚假 service cache 写入，保留真正执行端发生缓存未命中时的写入 | MBS 和 cooperative 两条路径均有回归；协作执行端仍正确缓存 |
| 碰撞回退只扫描一次，恢复某对 UAV 后可能使之前检查的 UAV 对重新碰撞 | 冲突集合闭包传播悬停回退，并核验最终位置、速度、边界 | 新增三机级联回退反例；新增非法初始布局与非有限动作拒绝测试 |
| 周期 GDSF 同分排序未显式保证 ID 决胜 | 使用 score 降序、file_id 升序的排序 | 同分文件按 ID 装载回归通过；本项为确定性加固，原版本该测试也通过 |
| 原确定性脚本以容差比较却宣称逐值完全一致，且接入计数键名错误会默认为零 | 改为数组精确比较、标量精确相等和直接读取真实计数键 | 双次、每次 2 回合 × 5 步确定性检查通过 |

保留已冻结的总体模型、请求容量和策略算法。本轮没有改动 SSOT、PPO 目标或下层奖励。修复涉及环境真实行为变化，历史模型/数据不能据此重新标为正式结果。

两项旧测试使用了物理不合法布局（UAV 间距 50m；多架 UAV 重合在区域外），已改为保持原验证目的的合法布局，没有放宽新物理约束。

### 验证结果与范围

| 检查 | 本次结果 |
|---|---|
| 本机语法检查 | 9 个新增/修改 Python 文件均通过；没有在本机导入 PyTorch 或运行训练 |
| 修改格式 | `git diff --check` 通过 |
| 远程完整测试 | **35 项：32 通过、3 已知下层失败、0 错误** |
| 环境子集 | **30/30 通过**（原 21 项 + 新 9 项） |
| 严格确定性 | 两次运行、每次 2 回合 × 5 步，通过 |
| 连续环境集成 | 2 个种子 × 4 条路径 × 100 步 = **800 步通过** |
| 连续运行检查内容 | 观测/奖励/指标有限性；接入与执行计数；总能耗与分项一致；缓存容量；速度、边界和最小间距 |
| 上层接口与更新 | Attention-MAPPO、Vanilla-MAPPO 各进行 1 次环境交互和优化器更新，损失、参数、梯度均为有限值 |
| 正式训练 | **未启动**；上述单步更新仅为兼容性 smoke，未生成正式 checkpoint |
| 代码审阅 | 完成显式自审；不是独立代理审查 |

800 步覆盖 heuristic / local / MBS / cooperative 控制路径，属于有限规模集成验证，不是算法性能对比，也不证明任意规模、任意参数或完整训练收敛。外部下层接口的 30 请求容量仍在，不能将这些 smoke 当作满容量策略验证。

运行环境由本次集成脚本实际返回：Python **3.12.11**，PyTorch **2.6.0+cpu**，NumPy **2.0.1**，CUDA 不可用。现有依赖足以完成上述测试；没有安装或升级依赖。额外 `pip check` 因后述 SSH 断线未完成，不声称整个 Conda 环境的所有包均无依赖冲突。

### 仍然存在的项目限制

1. `test_joint_logprob_is_sum_not_mean` 仍失败：下层联合 log-prob 使用平均而非求和。
2. `test_lower_reward_invariant_to_uav_movement` 仍失败：下层奖励包含飞行相关能耗；奖励和约束信号的 admitted-domain 一致性仍需专门修复。
3. `test_fifty_admitted_requests_all_decided_by_policy` 仍失败：容量仍为 30，不能保证全部请求由 learned lower 决策。
4. Joint-MAPPO 的概率和动作接口仍需单独修复验证，不能由上层两种模型的 smoke 推断其正确。
5. 全局 RNG、场景回放、硬件 realization 配对与正式结果 provenance 尚未完成 canonical freeze。本轮相同种子的可重复 smoke 不等于不同算法之间已完成场景配对。
6. 第四章 MC-ITR / Exact-Small 的完整评估与优化设施未由本轮交付或验证。

### 可以运行什么实验

| 实验 | 当前适用性 | 边界 |
|---|---|---|
| 环境正确性回归、确定性、缓存/时延/能耗/碰撞行为检查 | 可运行 | 本轮已经运行，入口见下 |
| 本地/MBS/协作卸载控制的探索性仿真 | 可运行 | 高负载时必须处理/披露外部动作容量；不得称正式 learned-lower 对比 |
| Attention-MAPPO 与 Vanilla-MAPPO 上层轨迹预实验，固定启发式卸载 | 可开始小规模预实验 | 模型交互及单步更新通过，尚未验证完整长训练；正式对比仍需场景与配置冻结 |
| 完整上下层联合学习、下层训练与相关消融 | 暂不用于正式结果 | 先修复上述概率、奖励、容量，并单独验证 Joint-MAPPO |
| 第四章固定轨迹优化、MC-ITR / Exact-Small | 环境可作为开发基础 | 仍需优化器、纯快照评估器及等价性验证 |
| 5 训练种子 × 10 workload 种子的最终论文实验 | 尚不就绪 | 需完成下层修复、场景配对、配置/源码冻结和原始输出隔离 |

远程已验证副本中的复核命令（不要在本机执行）：

```bash
cd /home/PengYanghan/Lunwen/env-repair-20260919-kBaVvp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/PengYanghan/miniconda3/envs/drone/bin/python -m unittest discover -s tests -v
/home/PengYanghan/miniconda3/envs/drone/bin/python tests/deterministic_smoke.py
/home/PengYanghan/miniconda3/envs/drone/bin/python tests/environment_validation_smoke.py
```

完整测试命令仍预期报告上述 3 项下层失败，不应隐藏或将其改为跳过。

### 源码与证据位置

修复保存在本机当前工作区，未提交、未推送；远程主仓库未部署修改，已验证代码位于上述隔离目录。上传时逐文件核对 SHA-256，一致结果保存在本机 `output/env_repair_20260919/source_hashes.json`。

已验证核心源码：

```text
environment/env.py
b264ade1f8b2ab10142aa2f9d76b9af682523a0740ca91c45e6f7e459f9bdba0
environment/uavs.py
6e5e2c9a307cc5229554fdbca38589f343dad7849fe432bcf258c10d754d0a72
config.py（未修改）
cc0042a3ec0b8f2e8b9ab81a7ae0c2ed275694962166d4be580c262c7b9d3743
```

远程隔离目录保留 `env_batch_red.log`、`env_boundaries_red.log`、`env_collision_red.log`、`env_full_after.log`、`env_deterministic.log`、`env_integration.log`。本报告依据本次实际返回的工具输出整理；原始日志尚未下载成功，不将本地摘要冒充原始日志。

### 交付时连接限制

完成测试后，在回收日志阶段 SSH 复用连接断开（`Broken pipe`）。新建端口 22 连接返回 `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`；按项目文档尝试 2222 返回 connection refused。未修改 SSH 配置、密钥或认证设置。

因此需要先恢复远程登录，才能再次复核或启动任何远程实验。连接问题不改变已经返回的测试通过证据，但不能宣称交付时新 SSH 会话可正常使用。
