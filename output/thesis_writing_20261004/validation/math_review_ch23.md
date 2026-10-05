# 第二、三章关键推导独立核验

核验日期：2026-10-04。核验对象为本次扩写后的 `latex/docs/chap02.tex` 与 `latex/docs/chap03.tex`。

本核验由负责第五章的代理完成，第二、三章由其他代理撰写。因此，这是章节作者以外的独立核验，不是第二、三章作者自审，也不等同于外部专家同行评议。核验采用逐式代数检查、概率对象和单位检查、实现对照及 TeX 静态检查；未在本机运行 PyTorch、训练或新的环境实验。主稿编译由主代理统一负责。

## 核验结论

在所核范围内，有限时域策略梯度、带终止掩码的 GAE、原始正态采样及对数概率、实际前向每头注意力缩放、服务三路径时延及 UAV 能耗公式均与其明确假设和代码相符。未发现需要改变实验结论或算法行为的错误。

核验修复一处明确缺失的数学假设、一处 TeX 指令遗漏，并调整主代理提供的两处第三章排版溢出；具体如下。

| 位置 | 核验发现 | 修复及影响 |
|---|---|---|
| 第二章 `eq:ch2-attention-variance` 前提 | 仅有零均值、单位方差和维度间独立，不足以推出点积方差为键维度；同维度的查询与键还需满足相应独立或矩条件 | 明确补充同一维度内 `q_i` 与 `k_i` 相互独立，以及不同维度变量对相互独立。保留原结论及“实际特征未必独立同分布”的证据边界 |
| 第三章 `eq:ch3-cache-score` | 第二个间距指令写为字面量 `qquad`，缺反斜线 | 改为 `\qquad`，缓存 EMA 系数及公式不变 |
| 第三章 `eq:ch3-normal-policy` | 主代理首轮日志显示长等式溢出约 35.71 pt | 用 `aligned` 将采样定义和对数概率分为两行，数学含义不变 |
| 第三章 `tab:upper-confirmation` | 主代理首轮日志显示表宽溢出约 6.26 pt | 在该表局部将 `\tabcolsep` 设为 5 pt，数字与统计口径不变 |

主代理先前发现并修复的缓存信念式 `form-feed + rac` 控制字符，不是本核验新增发现；本次复核其现式为正确的 `\frac{S_v^{\mathrm{cache}}}{\overline B}`。主代理修复的第五章两处 `+gamma` 同样不计入第二、三章独立核验成果。

## 逐项公式与实现对照

### 1. 有限时域策略梯度

核验 `eq:ch2-trajectory-density`、`eq:ch2-reward-to-go-gradient`、`eq:ch2-policy-gradient` 和 `eq:ch2-baseline-unbiased`。

目标是从时隙零折扣的期望总回报。固定初始分布和环境转移、不直接依赖策略参数、允许交换求导与积分时，轨迹得分为各时隙策略得分之和。动作前奖励的得分期望为零，删除这些项后仍需保留外部的 `gamma^t`；正文正确保留了该项。当前时隙起算的回报与时隙零起算的目标没有混淆。

动作前信息的基线与当前采样动作无关时，条件得分期望为零，减去该基线不改变理想梯度期望。正文还说明 actor 更新不能把基线网络的参数导数混入该证明，部分可观测条件不能直接冒充完整状态。这里的理想梯度推导没有被写成 PPO 有限样本更新或神经网络训练的单调性能保证。

### 2. 有限窗口 GAE、真实终止与采样截断

核验 `eq:ch2-nstep-advantage`、`eq:ch2-gae-mixture`、`eq:ch2-gae-coefficient`、`eq:ch2-masked-td`、`eq:ch2-masked-gae`、`eq:ch2-gae-products` 及第三章 `eq:gae`。

有限窗口混合最后一项吸收剩余权重，权重和为一；逐项收集 TD 残差后系数为 `(gamma lambda)^l`。带连接掩码的展开式使用此前连接掩码的乘积，空乘积为一，首个断点之后的残差不会泄漏到当前优势。

第二章区分 bootstrap 掩码与递推连接掩码：真实终止均为零，仅采样截断可为一和零。第三章对应当前代码的有限 1000 步任务，使用同一 `1-done` 关闭 bootstrap 和跨回合递推，正文没有把这套特定实现泛化为所有时间限制的处理规则。

实现对照：`marl_models/buffer_and_helpers.py` 中 `AttentionRolloutBuffer.compute_returns_and_advantages`，以及 `run_hierarchical_mappo_experiment.py` 的 `done = episode_step >= STEPS_PER_EPISODE`。实现先以未归一化优势和旧值形成 critic 回报，再归一化 actor 优势；第三章描述顺序正确。

### 3. 正态策略的采样对象、概率比率与熵

核验 `eq:ch2-gaussian-logprob`、`eq:ch2-log-ratio`、`eq:ch2-ppo-piecewise`、`eq:ch3-normal-policy`、`eq:ppo-clip` 及 `eq:ch3-value-clipping`。

二维独立正态的对数概率按两个坐标求和，方差项和归一化常数正确。均值经过 tanh，不等于整个采样变量经过 tanh；标准差来自限制在 `[-20,2]` 的可学习对数标准差。原始随机变量保留在缓冲区，执行变量逐坐标裁剪，再经径向位移与几何修复。多对一执行映射没有被当作正态密度，也没有错误添加压缩正态的雅可比项。

实现对照：`marl_models/attention_mappo/agents.py:ActorNetwork.forward`、`marl_models/attention_mappo/attention_mappo.py:get_action_and_value/update`，以及 `run_hierarchical_mappo_experiment.py` 的 `traj_actions_raw` 与 `traj_actions` 分离保存路径。

PPO 分段目标对负优势使用上界形式 `A max(rho,1-epsilon)`，符号方向正确；正文明确裁剪目标并非硬概率比率或 KL 约束。critic 损失包含 `1/2` 系数与新旧裁剪值平方误差的逐样本最大值，符合实现。代码 `dist.entropy().mean()` 在动作坐标上也取平均，而 `log_prob(...).sum(-1)` 求和；第三章明确给出这一不同约定，未把平均熵写成联合正态总熵。

### 4. 实际注意力缩放与方差前提

核验 `eq:ch2-scaled-attention`、`eq:ch2-attention-variance`、`eq:ch2-multihead` 及第三章注意力权重式。

上层总嵌入维度 64，头数 8，实际拆头后的 query/key 最后一维为 8。`marl_models/attention.py:CrossAttentionExtractor.forward` 调用 `F.scaled_dot_product_attention`，没有传入自定义 scale，因此实际缩放为 `1/sqrt(8)`。源码中的旧 `self.scale = ATTN_HIDDEN_DIM**(-0.5)` 字段没有进入该实际前向调用，不能用这一遗留字段推断实际缩放。

默认缩放及 bool 掩码的语义已核对 [PyTorch 2.6 官方 SDPA 文档](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)：默认尺度采用 query 最后一维，True 表示参与注意力。第三章采用每头 `sqrt(8)` 与有效槽掩码正确；全空上下文经过投影可能仍含偏置的限制也已在正文说明。

对方差推导，补充独立假设后有乘积均值为零、乘积二阶矩为一，各维度乘积独立，故方差相加得到键维度。原假设不足的明确反例是每维令 `q_i=k_i` 为独立标准正态：查询和键各自零均值、单位方差，维度间仍独立，但每个平方项方差为 2，点积方差为 `2d_k`。本次只补足前提，不把真实网络特征假定为满足该理想化模型。

### 5. 最终负载、三路径服务时延与分配能耗

核验第三章 `eq:ch3-final-load`、`eq:ch3-service-three-paths`、`eq:ch3-dynamic-compute-energy`、`eq:ch3-deadline-load-budget`、`eq:ch3-assignment-energy-paths` 和 `eq:ch3-full-fleet-energy`。

输入和文件以 bytes 计，链路速率以 bit/s 计，传输时延均为 `8 size/rate`。同节点本地与迁入服务共同形成最终负载，计算份额为 `F_j/N_j`。三路径都先经过来源 UAV 上传；协作路径额外包含机间转发及目标缓存缺失回源；MBS 路径使用固定 MBS 算力，不按 MBS 任务数均分。缓存使用步首冻结状态，不能以预测命中概率替代实际路径中的二值状态。

计算功率乘计算时长消去一个频率因子，得到 `kappa C_m (F_j/N_j)^2`，周期、每秒周期和焦耳的单位一致。协作机队能耗计入来源发送和目标接收两端；MBS 只计来源 UAV 的回传发射，不计地面计算能耗。三路径共同的来源 UAV 输入接收能耗属于完整机队能耗，排除在目标敏感的分配能耗之外；UE 上传能耗属于 UE 电池。这些口径没有混用。

实现对照：`environment/uavs.py:_tx_latency_bytes/_get_computing_latency_and_energy/_process_service_request`，`environment/uavs.py:UAV.execute_request_batch` 以及 `environment/env.py:Env.step` 的全批目标解析后执行顺序。

### 6. 机动、补能与完整能耗

核验第三章 `eq:ch3-radial-action`、`eq:ch3-wpt-harvest`、`eq:ch3-wpt-fleet-cost` 与 `eq:ch3-motion-energy`。

径向动作式对应逐坐标裁剪后的向量，数值保护导致小范围近似线性而非严格线性，正文措辞正确。飞行/悬停结算使用实际修复位移，以移动距离除最大速度得到移动时长；代码额外把数值位移限制在合法速度范围，正文按满足速度约束的条件推导。存在至少一个关联能量请求时 UAV 支付一次 WPT 发射功率乘时隙长，并未按能量用户数重复计 UAV 发射成本。正文没有从该简化能耗模型推导未实现的剩余 UAV 电池约束。

实现对照：`environment/env.py:_apply_actions_to_env`、`environment/uavs.py:update_energy_consumption` 和 `config.py` 对应功率、时间、采集增益参数。

## 检查边界与后续集成

- 本次只修复能够由代数、源码或 TeX 指令直接确定的问题；未改变科研数据、统计口径、实验协议或章节范围。
- 已对第二、三章进行控制字符、花括号、环境配对和重复标签静态检查。最终主稿编译与两处 overfull 是否消失，需以主代理对更新后源码的编译日志确认；本核验未独立编译整稿。
- 本次没有重新评估环境、检验统计假设或保证理论最优；公式正确性不自动证明实际部署有效性或学习收敛。
