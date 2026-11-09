# 全论文数学公式逐条核查、理论溯源与完整推导报告
**（维度 1：全量 140 个编译编号公式 + 第四章 3 个命题的合理性、来源、量纲、推导与代码对齐审计）**

- **审计日期**：2026-11-09
- **审计范围**：[`latex/docs/chap01.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap01.tex) 至 [`latex/docs/chap05.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex) 中的全部 **140 个编译编号公式**（含 132 个带 `\label{eq:...}` 的公式与第二章 `align` 环境中未加 `\nonumber` 的 8 个中间推导行），以及第四章的 **3 个理论命题**。

---

## 一、 核心审计结论总览（Executive Summary of Formula Audit）

1. **总体数学正确性**：论文绝大部分公式（140 个编号公式中的 **126 个**）在代数结构、概率测度论基础、量纲一致性及与代码的对应关系上完全正确。
2. **发现的 6 项核心公式问题（必须在定稿前修正）**：
   - **[F-ERR-1] 跨章节公式冲突（第二章式 2.66–2.67 vs 第五章式 5.6–5.8）**：第二章 [`chap02.tex:L629-L637`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex#L629-L637) 将上层期望反馈 $\mathbf{C}_{\mathrm{upper}}[k,m](s)$ 定义为 $4\times 6$ 标量打分矩阵，并在基础 logit $z_k^{\mathrm{base}}(s)$ 外部做线性加权 $z_k(s)=z_k^{\mathrm{base}}(s)+\eta_{\mathrm{fb}}\sum_m p_{\theta_m}(m\mid s,k)\mathbf{C}_{\mathrm{upper}}[k,m](s)$；而第五章 [`chap05.tex:L150-L163`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L150-L163) 式 (5.7)–(5.8) 与实际代码 [`marl_models/queued_joint_rl.py:L126-L135`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L126-L135) 将 $\mathbf{C}_{\mathrm{upper}}(s,k)=\sum_{m=1}^M p_{\theta_m}(m\mid s,k)\boldsymbol{\phi}_{km}(s)\in\mathbb{R}^4$ 定义为 4 维期望特征向量，并拼接在分配分支 MLP 输入内部 $z_k^{\mathrm{alloc}}=g_{\theta_a}([\mathbf{a}_k,\mathbf{C}_{\mathrm{upper}}(s,k),\mathbf{c}])+\beta_a\log p_a^0(k\mid s)$。
   - **[F-ERR-2] 第五章释放时间近似公式（式 5.4–5.5）严重遗漏代码中的串行排队递推与截断逻辑**：正文 [`chap05.tex:L125-L134`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L125-L134) 将式 (5.4) 写为无排队等待的独立比值相加 $\tilde{r}_{n,j}=t + b_n^{\mathrm{up,rem}}/\hat{R}^{\mathrm{edge}} + \mathbf{1}\{j\neq u_n\}b_n^{\mathrm{fwd,rem}}/\hat{R}^{\mathrm{inter/bh}}$；而 [`utils/queued_joint_rl.py:L181-L300`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L181-L300) 实际执行的是按截止期排序后的**多级串行单服务台递推**（含同 `(src, ue)` 对上传串行空闲时刻 `ue_up_free`、离开覆盖时 `t_up_done > 0.75s -> 10.0s` 硬截断、回传与机间全局串行空闲时刻 `t_bh_free / t_inter_free`、`waiting` 队列在 $t\ge 1.0\,\mathrm{s}$ 的虚拟注入，以及式 5.5 窗口外完工任务的 `0.85 / 0.15` 折算系数）。
   - **[F-ERR-3] 第五章式 (5.16) KL 散度参考策略 $\pi_{\mathrm{ref}}$ 的文字定义与代码矛盾**：[`chap05.tex:L260`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L260) 声称冷启动 RL 中 $\pi_{\mathrm{ref}}$ 为固定初始先验策略，而 [`scripts/analysis/queued_joint_rl_experiment.py:L394-L398`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L394-L398) 对冷启动组在每轮迭代将 `ref_model` 更新为上一轮策略 `prev_model`（移动锚点）。
   - **[F-ERR-4] 第三章式 (3.26) 注意力缩放因子 $1/\sqrt{d_h}=1/\sqrt{8}$ 与 `get_attention_weights()` 代码不一致**：[`marl_models/attention.py:L52,L114`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/attention.py#L52) 中 `self.scale = config.ATTN_HIDDEN_DIM ** -0.5` 误用了 $1/\sqrt{64}$。
   - **[F-ERR-5] 第三章式 (3.31) 下层拉格朗日乘子更新与式 (2.53) 联合动作对数概率在代码中的畸变**：[`marl_models/offload_mappo/offload_mappo.py:L216-L221`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/offload_mappo/offload_mappo.py#L216-L221) 在求回合均值前对每步违例取了 `max(0.0, violation)`，导致乘子 $\lambda_{\mathrm{dsr}},\lambda_{\mathrm{mbs}}$ 永远无法下降；且 `L253, L291` 对可变活跃请求数 $A_{u,t}\in[0,100]$ 直接求和 $\log\pi$ 未做归一化。
   - **[F-ERR-6] 第二章 8 处 `align` 推导中间行漏加 `\nonumber`**：[`chap02.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex) 中的式 `(2.20)`, `(2.25)`, `(2.28)`, `(2.31)`, `(2.32)`, `(2.40)`, `(2.61)`, `(2.64)` 均为多行等式推导的左端或中间过渡行，未加 `\nonumber` 导致编译出 8 个多余的空标签公式编号。

---

## 二、 全文 140 个编译编号公式逐条核查总表（Complete Inventory Table）

### 2.1 第一章（共 1 个公式）与第二章（共 71 个编译编号，含 63 个标签公式 + 8 个未加 `\nonumber` 中间行）

| 编译编号 | LaTeX 标签与位置 | 公式核心内容 | 理论/物理来源 | 量纲与数学合理性 | 代码对齐状态 |
|---|---|---|---|---|---|
| `(1.1)` | `eq:ch1-dsr-factorization` (`chap01.tex:L29`) | $\mathrm{DSR}=\frac{S}{G}=\frac{A}{G}\cdot\frac{S}{A}\ (G>0,A>0)$ | 条件概率/计数分解恒等式 | 无量纲比率 $\in[0,1]$，完全合理 | ✅ 与全库 `S/G = (A/G)*(S/A)` 严格一致 |
| `(2.1)` | `eq:ch2-swarm-motion` (`chap02.tex:L23`) | $\tilde{\mathbf{p}}_u(t+1)=\mathbf{p}_u(t)+\Delta\mathbf{p}_u(t),\ \|\Delta\mathbf{p}_u(t)\|_2\le v_{\max}\tau$ | 离散时间二维运动学 | 位置 $\mathrm{m}$，步长 $15\,\mathrm{m/s}\times 1\,\mathrm{s}=15\,\mathrm{m}$ | ✅ 对齐 `uavs.py` 与 `queued_service.py` |
| `(2.2)` | `eq:ch2-joint-geometry` (`chap02.tex:L32`) | $\mathcal{P}_{\mathrm{feas}}=\{\mathbf{P}\in[x_{\min},x_{\max}]^U\times[y_{\min},y_{\max}]^U:\|\mathbf{p}_u-\mathbf{p}_v\|_2\ge d_{\min}\}$ | 集群边界与防碰撞几何约束 | 坐标 $\mathrm{m}$，$d_{\min}=200\,\mathrm{m}$ | ✅ 对齐 `env.py` 与 `queued_service.py` |
| `(2.3)` | `eq:ch2-sensing-neighbors` (`chap02.tex:L46`) | $\mathcal{N}_u^{\mathrm{sense}}(t)=\{v\neq u:\|\mathbf{p}_u(t)-\mathbf{p}_v(t)\|_2\le R_s\}$ | 邻机感知与协作通信图 | 距离 $\mathrm{m}$，$R_s=460\,\mathrm{m}$ | ✅ 对齐 `config.UAV_SENSING_RANGE = 460.0` |
| `(2.4)` | `eq:ch2-cycles` (`chap02.tex:L83`) | $C_m=\omega_m L_m$ | MEC 线性计算负载模型 (Mao et al., 2017) | $\mathrm{cycles} = (\mathrm{cycles/byte})\times\mathrm{byte}$ | ✅ 对齐 `request_types.py` |
| `(2.5)` | `eq:ch2-distance` (`chap02.tex:L93`) | $d_{ij}(t)=\sqrt{\|\mathbf{p}_i(t)-\mathbf{p}_j(t)\|_2^2+(z_i-z_j)^2}$ | 三维欧氏距离 | $\mathrm{m}$ | ✅ 对齐 `comm_model.py:L25` |
| `(2.6)` | `eq:ch2-link-rate` (`chap02.tex:L100`) | $g_{ij}(t)=\frac{G_0 g_0}{d_{ij}(t)^2+\epsilon_0},\ R_{ij}(t)=B_{ij}\log_2(1+\frac{P_i g_{ij}(t)}{\sigma^2})$ | 自由空间路径损耗 + Shannon 容量 (Zeng et al., 2016) | $g_{ij}$ 无量纲（$\epsilon_0=10^{-6}\,\mathrm{m}^2$），$R_{ij}$ 为 $\mathrm{bit/s}$ | ⚠️ 公式与代码一致，但回传仰角 $8.65^\circ<45^\circ$ 且需声明跨机正交频分与 TDMA 等效分摊 |
| `(2.7)` | `eq:ch2-byte-bit` (`chap02.tex:L105`) | $T_{\mathrm{tx}}=\frac{8D}{R}$ | 字节–比特传输时间换算 | $\frac{8\,\mathrm{bit/byte}\times\mathrm{byte}}{\mathrm{bit/s}}=\mathrm{s}$ | ✅ 对齐 `uavs.py` 与 `queued_service.py` |
| `(2.8)` | `eq:ch2-shared-computation` (`chap02.tex:L126`) | $f_{m,j}=\frac{F_j}{N_j(\mathbf{b})},\ T_{m,j}^{\mathrm{cmp}}=\frac{C_m}{f_{m,j}}=\frac{C_m N_j(\mathbf{b})}{F_j}$ | 处理器共享（Processor Sharing）等分模型 | $\frac{\mathrm{cycles}}{\mathrm{cycles/s}}=\mathrm{s}$ | ✅ 对齐 `uavs.py:L820-L885`（协议 I） |
| `(2.9)` | `eq:ch2-cache-capacity` (`chap02.tex:L136`) | $\sum_s B_s h_{u,s}(t)\le S_u^{\mathrm{cache}},\ h_{u,s}(t)\in\{0,1\}$ | 0-1 背包容量约束 | $\mathrm{byte}\le\mathrm{byte}$ | ✅ 对齐 `uavs.py` 缓存容量检查 |
| `(2.10)` | `eq:ch2-jain` (`chap02.tex:L150`) | $J=\frac{(\sum_{m=1}^M x_m)^2}{M\sum_{m=1}^M x_m^2}\in[\frac{1}{M},1]$ | Jain 公平性指数 (Jain et al., 1984) | 无量纲；全零时约定 $J=0$ | ✅ 对齐 `env.py` Jain 计算 |
| `(2.11)` | `eq:ch2-markov-property` (`chap02.tex:L175`) | $P(s_{t+1},r_t\mid s_0,a_0,\dots,s_t,a_t)=P(s_{t+1},r_t\mid s_t,a_t)$ | 马尔可夫决策过程（MDP）定义 (Sutton & Barto, 2018) | 条件概率测度 | ✅ 基础定义 |
| `(2.12)` | `eq:ch2-return` (`chap02.tex:L190`) | $G_t=\sum_{l=0}^{T-t}\gamma^l r_{t+l}$ | 有限时域折扣回报 | 与标量奖励同量纲 | ✅ 基础定义 |
| `(2.13)`–`(2.16)` | `eq:ch2-state-value` 至 `eq:ch2-bellman-v` (`chap02.tex:L198-L213`) | $V_t^\pi(s), Q_t^\pi(s,a)$ 定义与一步 Bellman 期望方程 | 动态规划与 Bellman 方程 (Bellman, 1957; Sutton & Barto, 2018) | 期望算子线性展开，完全严格 | ✅ 见第三节推导链 3 |
| `(2.17)`–`(2.18)` | `eq:ch2-advantage`, `eq:ch2-advantage-zero` (`chap02.tex:L217-L225`) | $A_t^\pi(s,a)=Q_t^\pi(s,a)-V_t^\pi(s),\ \mathbb{E}_{a\sim\pi}[A_t^\pi(s,a)]=0$ | 优势函数定义与零均值性质 | 由重期望定律直接成立 | ✅ 见第三节推导链 3 |
| `(2.19)`–`(2.21)` | `eq:ch2-bellman-optimal`, `eq:ch2-bellman-contraction` (`chap02.tex:L232-L244`) | $(\mathcal{T}^*V)(s)$ 定义及 $\|\mathcal{T}^*V_1-\mathcal{T}^*V_2\|_\infty\le\gamma\|V_1-V_2\|_\infty$ | Banach 不动点与最优 Bellman 算子压缩性 | 利用 $|\max_a f-\max_a g|\le\max_a|f-g|$ 严格成立；**注：`(2.20)` 为中间行漏加 `\nonumber`** | ⚠️ 给 `(2.20)` 补 `\nonumber` |
| `(2.22)`–`(2.30)` | `eq:ch2-objective` 至 `eq:ch2-policy-gradient` (`chap02.tex:L256-L306`) | 轨迹密度 $p_\theta(\tau)$、对数似然梯度恒等式、得分函数零期望与因果 Reward-to-Go 策略梯度 | REINFORCE 与策略梯度定理 (Williams, 1992; Sutton et al., 1999) | 严格成立；**注：`(2.25)`, `(2.28)` 为中间行漏加 `\nonumber`** | ⚠️ 给 `(2.25)`, `(2.28)` 补 `\nonumber` |
| `(2.31)`–`(2.34)` | `eq:ch2-baseline-unbiased`, `eq:ch2-advantage-gradient` (`chap02.tex:L315-L329`) | 状态基线 $b(s_t)$ 不改变策略梯度期望证明及优势梯度式 | 基线方差缩减定理 (Williams, 1992) | 严格成立；**注：`(2.31)`, `(2.32)` 为中间行漏加 `\nonumber`** | ⚠️ 给 `(2.31)`, `(2.32)` 补 `\nonumber` |
| `(2.35)` | `eq:ch2-value-loss` (`chap02.tex:L333`) | $L_V(\phi)=\frac{1}{2}\mathbb{E}[(V_\phi(s_t)-\widehat{V}_t)^2]$ | Critic 均方误差损失 | 标量平方误差 | ✅ 对齐 Critic MSE 损失 |
| `(2.36)`–`(2.38)` | `eq:ch2-td-error` 至 `eq:ch2-nstep-advantage` (`chap02.tex:L346-L360`) | $\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$ 及 $n$ 步优势望远镜求和 $\widehat{A}_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$ | 多步 TD 与望远镜相消恒等式 (Sutton & Barto, 2018) | 相邻项 $\pm\gamma^l V_\phi(s_{t+l})$ 精确抵消 | ✅ 见第三节推导链 4 |
| `(2.39)`–`(2.42)` | `eq:ch2-gae-mixture` 至 `eq:ch2-gae-residual-sum` (`chap02.tex:L369-L391`) | GAE 几何加权定义、双重求和交换次序导出 $(\gamma\lambda)^l\delta_{t+l}$ | 广义优势估计 GAE (Schulman et al., 2016) | 几何级数 $\sum_{n=l+1}^\infty(1-\lambda)\lambda^{n-1}=\lambda^l$ 严格成立；**注：`(2.40)` 漏加 `\nonumber`** | ⚠️ 给 `(2.40)` 补 `\nonumber` |
| `(2.43)`–`(2.45)` | `eq:ch2-masked-td` 至 `eq:ch2-gae-products` (`chap02.tex:L400-L413`) | 带非终止掩码 $m_t=1-d_t$ 的 TD 残差与反向递推 $\widehat{A}_t=\delta_t+\gamma\lambda m_t\widehat{A}_{t+1}$ | 有限缓冲区带终止掩码 GAE | 展开递推即得连乘积 $\prod_{j=t}^{t+l-1}m_j$ | ✅ 对齐 `buffer_and_helpers.py` |
| `(2.46)`–`(2.50)` | `eq:ch2-importance` 至 `eq:ch2-entropy` (`chap02.tex:L425-L471`) | 重要性比率 $\rho_t(\theta)$、PPO-Clip 目标及其分段等价形式、策略熵 $\mathcal{H}$ | 近端策略优化 PPO (Schulman et al., 2017) | 分段分析 $\widehat{A}_t>0$ 与 $\widehat{A}_t<0$ 完全严密 | ✅ 对齐 PPO 实现 |
| `(2.51)` | `eq:ch2-gaussian-logprob` (`chap02.tex:L478`) | $\log\pi_\theta(\tilde{\mathbf{a}}\mid o)=-\frac{1}{2}\sum_{j=1}^d\left[\frac{(\tilde{a}_j-\mu_{\theta,j}(o))^2}{\sigma_j^2}+2\log\sigma_j+\log(2\pi)\right]$ | 对角高斯分布对数密度 | 实数对数密度 | ✅ 对齐 `attention_mappo/agents.py` |
| `(2.52)` | `eq:ch2-masked-policy` (`chap02.tex:L489`) | $\pi_\theta(c\mid o,M)=\frac{M_c\exp(z_c(o))}{\sum_{c'}M_{c'}\exp(z_{c'}(o))}$ | 掩码分类分布（Masked Categorical） | 概率归一化 $\sum_c\pi_\theta(c)=1$ | ✅ 对齐 `offload_policy.py`（置 $-\infty$ / $-10^{10}$） |
| `(2.53)` | `eq:ch2-factorized-policy` (`chap02.tex:L507`) | $\log\pi_\theta(\mathbf{a}_t\mid o_t)=\sum_{r\in\mathcal{A}_t}\log\pi_\theta(a_{t,r}\mid o_t,M_{t,r})$ | 活跃槽位条件独立因式分解 | 数学成立，但需注意活跃槽位数 $|\mathcal{A}_t|$ 随步剧烈变化 | ⚠️ 代码 `offload_mappo.py` 未除以 $|\mathcal{A}_t|$，高负载下比率膨胀 |
| `(2.54)`–`(2.58)` | `eq:ch2-effective-transition` 至 `eq:ch2-mappo-loss` (`chap02.tex:L517-L562`) | 多智能体有效转移核、联合观测 Critic $V_\phi^u(\mathbf{o}_t)$、共享参数总梯度与 MAPPO 损失 | CTDE 与 MAPPO (Lowe et al., 2017; Yu et al., 2022) | 推导严谨，准确区分个体输出集中 Critic 与标量团队 Critic | ✅ 对齐 `attention_mappo.py` |
| `(2.59)`–`(2.62)` | `eq:ch2-scaled-attention` 至 `eq:ch2-multihead` (`chap02.tex:L576-L596`) | 缩放点积注意力、点积方差 $\mathrm{Var}(q^\top k)=d_k$ 证明与多头拼接投影 | Transformer 注意力 (Vaswani et al., 2017) | 独立零均值单位方差分量下 $\sum_{r=1}^{d_k}\mathrm{Var}(q_r k_r)=d_k$ 严格成立；**注：`(2.61)` 漏加 `\nonumber`** | ⚠️ `attention.py:L52` 的 `get_attention_weights` 误用 $d_{\mathrm{model}}$ 而非 $d_k$ |
| `(2.63)`–`(2.65)` | `eq:ch2-permutation-invariance`, `eq:ch2-message-passing` (`chap02.tex:L607-L625`) | 置换不变性 $\sum_i\alpha(q,k_{\sigma(i)})v_{\sigma(i)}=\sum_j\alpha(q,k_j)v_j$ 与局部消息聚合 | DeepSets / 集合置换不变性 (Zaheer et al., 2017) | 双射换元 $j=\sigma(i)$ 严格成立；**注：`(2.64)` 漏加 `\nonumber`** | ✅ 理论推导正确 |
| `(2.66)`–`(2.67)` | `eq:ch2-dual-branch-factorization`, `eq:ch2-expectation-feedback` (`chap02.tex:L629-L637`) | $\pi_\theta(k,m\mid s)=p_{\theta_a}(k\mid s)p_{\theta_m}(m\mid s,k)$ 及 $z_k(s)=z_k^{\mathrm{base}}(s)+\eta_{\mathrm{fb}}\sum_m p_{\theta_m}(m\mid s,k)\mathbf{C}_{\mathrm{upper}}[k,m](s)$ | 链式法则 + 期望反馈打分 | ❌ **式 (2.67) 与第五章式 (5.7)–(5.8) 及代码严重冲突！** | ❌ 代码中 $\mathbf{C}_{\mathrm{upper}}(s,k)\in\mathbb{R}^4$ 是 4 维向量且拼在 MLP 内部，无 $\eta_{\mathrm{fb}}$ |
| `(2.68)`–`(2.70)` | `eq:ch2-batch-externality` 至 `eq:ch2-matching-cost` (`chap02.tex:L647-L673`) | 单请求迁移对同节点任务时延差分 $\mp C_r/F_j$、二元字典序判据（含容差 $\epsilon_{\mathrm{tie}}$） | 处理器共享差分代数与字典序优化 | 代数严格正确，但需注明第四章确定性实现取 $\epsilon_{\mathrm{tie}}=0$ | ⚠️ 统一 $\epsilon_{\mathrm{tie}}=0$ 说明 |
| `(2.71)` | `eq:ch2-paired-difference` (`chap02.tex:L709`) | $d_{s,w}=Y_{\mathrm{cand}}(s,w)-Y_{\mathrm{base}}(s,w),\ \bar{d}_s=\frac{1}{W}\sum_{w=1}^W d_{s,w}$ | 配对实验设计块均值 | 线性统计量 | ✅ 对齐第三、四章统计分析 |

---

### 2.2 第三章（共 33 个公式）、第四章（共 17 个公式）与第五章（共 18 个公式）

| 编译编号 | LaTeX 标签与位置 | 公式核心内容 | 理论/物理来源 | 量纲与数学合理性 | 代码对齐状态 |
|---|---|---|---|---|---|
| `(3.1)`–`(3.4)` | `eq:current-dsr-decomp` 至 `eq:ch3-paired-factor-decomp` (`chap03.tex:L23-L49`) | $G_t=U_t+A_t,\ A_t=S_t+L_t$，跨单元均值协方差项 $\mathrm{Cov}(\alpha_i,\eta_i)$，以及配对差精确双线性分解 $\alpha_1\eta_1-\alpha_0\eta_0=\bar{\eta}\Delta\alpha+\bar{\alpha}\Delta\eta$ | 恒等式代数与双线性差分恒等式 | 展开 $\frac{\eta_1+\eta_0}{2}(\alpha_1-\alpha_0)+\frac{\alpha_1+\alpha_0}{2}(\eta_1-\eta_0)=\alpha_1\eta_1-\alpha_0\eta_0$ **恒等精确成立** | ✅ 推导极为漂亮严密 |
| `(3.5)` | `eq:ch3-natural-association` (`chap03.tex:L84`) | $\mathcal{U}_m^{\mathrm{cov}}(t)=\{u:\|\mathbf{p}_u(t)-\mathbf{q}_m(t)\|_2\le R_c\},\ u_m(t)=\arg\min_{u\in\mathcal{U}_m^{\mathrm{cov}}(t)}\|\mathbf{p}_u(t)-\mathbf{q}_m(t)\|_2$ | 最近邻自然覆盖关联规则 | 距离 $\mathrm{m}$，$R_c=100\,\mathrm{m}$ | ✅ 对齐 `env.py` 关联逻辑 |
| `(3.6)`–`(3.8)` | `eq:ch3-radial-action` 至 `eq:ch3-two-action-timing` (`chap03.tex:L110-L145`) | 径向裁剪 $\mathbf{a}_u^{\mathrm{clip}}=\tilde{\mathbf{a}}_u/\max(1,\|\tilde{\mathbf{a}}_u\|_2)$，位移步长 $v_{\max}\tau=15\,\mathrm{m}$，先在 $\mathbf{p}_u(t)$ 结算卸载再移动至 $\mathbf{p}_u(t+1)$ | 欧氏球投影与两阶段时序合同 | 投影后 $\|\mathbf{a}_u^{\mathrm{clip}}\|_2\le 1$，位移 $\le 15\,\mathrm{m}$ | ✅ 对齐 `env.py:L242-L260` |
| `(3.9)`–`(3.11)` | `eq:ch3-zipf` 至 `eq:ch3-static-uav-specs` (`chap03.tex:L163-L184`) | Zipf 分布 $p_k=\frac{(k+k_0)^{-\beta}}{\sum_{j=1}^N(j+k_0)^{-\beta}}$，$C_m=\omega_s L_m$，五机算力 $[75,60,70,85,50]\,\mathrm{GHz}$ 与缓存 $[158,171,148,134,162]\,\mathrm{MB}$ | Zipf 定律与异构边缘节点设定 | $\sum_{k=1}^N p_k=1$（代码中 `k_0 = 0, beta = 0.8`） | ✅ 对齐 `config.py` 与 `user_equipments.py` |
| `(3.12)`–`(3.15)` | `eq:channel-gain` 至 `eq:ch3-final-load` (`chap03.tex:L191-L228`) | 信道增益 $g_{i,j}$、边缘接入速率 $R_{m,u}^{\mathrm{edge}}$（除以 $n_u^{\mathrm{assoc}}$）、机间/回传/内容速率与最终批量负载 $N_j(\mathbf{b})$ | 自由空间信道与处理器共享 | 量纲均为 $\mathrm{bit/s}$，但需说明机间/回传在协议 I 中未按并发数分摊 | ✅ 对齐 `comm_model.py` 与 `uavs.py` |
| `(3.16)`–`(3.18)` | `eq:ch3-service-three-paths` 至 `eq:ch3-content-three-paths` (`chap03.tex:L232-L266`) | 服务请求三路径时延（本地、协作 UAV $v$、MBS）、剩余计算时间预算 $\tau_{m,j}^{\mathrm{rem}}$ 与内容请求三路径时延 | 串联传输 + 回源 + 计算时延 | 单位均为 $\mathrm{s}$；当 $\tau_{m,j}^{\mathrm{rem}}>0$ 时 $T_{m,j}\le D_m\iff N_j(\mathbf{b})\le\frac{F_j\tau_{m,j}^{\mathrm{rem}}}{C_m}$ | ✅ 对齐 `uavs.py:L820-L885` |
| `(3.19)`–`(3.21)` | `eq:ch3-cache-score` 至 `eq:jfi` (`chap03.tex:L272-L299`) | GDSF 缓存评分 $S_{u,k}=\mathrm{freq}_{u,k}\frac{\mathrm{cost}_k}{\max(B_k,1)}$、邻机缓存容量比信念 $\widehat{h}_{v,s}$、累计达标覆盖率 $c_m(t)$ 与 Jain 指数 | GDSF 缓存替换算法 (Cherkasova, 1998) 与 Jain 指数 | 量纲合理 | ✅ 对齐 `uavs.py` 与 `env.py` |
| `(3.22)`–`(3.23)` | `eq:ch3-ue-slot`, `eq:ch3-lower-obs-spec` (`chap03.tex:L345-L396`) | 上层 7 维用户槽特征（含是否覆盖标记 $\mathbf{1}[m\in\mathcal{M}_u^{\mathrm{assoc}}]$）与下层 1405 维卸载观测（$5+100\times 14$） | 状态归一化特征工程 | 各分量均作归一化或截断保护 | ✅ 对齐 `env.py` 观测拼装 |
| `(3.24)`–`(3.25)` | `eq:upper-reward`, `eq:ch3-final-reward` (`chap03.tex:L405-L419`) | 上层局部奖励 $r_u$、未关联惩罚时延 $T^{\mathrm{pen}}=20\,\mathrm{s}$ 及上层总奖励 $r_{u,t}^{\mathrm{upper}}=0.5 r_t^{\mathrm{global}}+0.5 r_u-p_u^{\mathrm{viol}}$ | 团队–个体混合奖励塑形 | 标量无量纲奖励 | ✅ 对齐 `env.py`（注：评估脚本算表 3.5 时延指标时分母只除了服务请求数） |
| `(3.26)`–`(3.30)` | `eq:ch3-cross-attention` 至 `eq:ch3-value-clipping` (`chap03.tex:L429-L500`) | 多头交叉注意力（$d_h=8$）、高斯原始动作对数似然、GAE、PPO-Clip 与带 Value Clipping 的 Critic 损失 | Vaswani et al. (2017); Schulman et al. (2016, 2017) | 公式完全正确 | ⚠️ `attention.py:L52` 的 `get_attention_weights` 缩放写错为 $1/\sqrt{64}$ |
| `(3.31)` | `eq:ch3-lower-reward-lagrange` (`chap03.tex:L544`) | 下层奖励与自适应惩罚乘子更新 $\lambda_{\mathrm{dsr}}\leftarrow\mathrm{clip}(\lambda_{\mathrm{dsr}}+\alpha_\lambda\overline{[g_{\mathrm{dsr},t}]_+},0,\lambda_{\max})$ | 拉格朗日对偶启发式惩罚 | ⚠️ **由于对每步先取 $[\cdot]_+=\max(0,\cdot)$，步均值恒 $\ge 0$，导致乘子单调不减（棘轮效应）！** | ✅ 正文式 (3.31) 如实写出了代码中的 $\overline{[g_t]_+}$，但需指出单调不减缺陷 |
| `(3.32)` | `eq:ch3-slot-selection` (`chap03.tex:L581`) | 候选扩展观测对已关联用户与未关联紧急用户的优先级排序元组 $k_m(u)$ 及前 $K_{\mathrm{UE}}=30$ 截断 | 确定性 Top-$K$ 槽位选择规则 | 排序键良定义 | ✅ 对齐 `env.py` 候选观测构造 |
| `(3.33)` | `eq:ch3-seed-paired-ci` (`chap03.tex:L647`) | 种子级配对均值差 $\bar{d}=\frac{1}{S}\sum_{s=1}^S d_s$ 与 $t_{0.975,S-1}$ 置信区间 | 配对样本 Student's $t$ 检验 | $S=10,\ df=9,\ t_{0.975,9}=2.2622$ | ✅ 对齐 `confirmation_paired_comparison.json` |
| `(4.1)`–`(4.5)` | `eq:ch4-assignment-space` 至 `eq:compute-latency-energy` (`chap04.tex:L24-L75`) | 乘积分配空间 $\mathcal{A}=\prod_{m\in\mathcal{M}}\mathcal{C}_m$、传输时延 $8L_m/R$、三路径真实缓存时延与最终负载计算时延 $C_m N_j(\mathbf{a})/F_j$ | 离散组合空间与处理器共享模型 | 量纲严格为 $\mathrm{s}$ | ✅ 对齐 `utils/chapter4.py` 与 `snapshot_evaluator.py` |
| `(4.6)` | `eq:ch4-migration-latency-externality` (`chap04.tex:L84`) | $i\to j$ 单请求迁移下：$\Delta T_k^{\mathrm{cmp}}=-C_k/F_i\ (k\in\mathcal{M}_i\setminus\{m\})$，$\Delta T_\ell^{\mathrm{cmp}}=+C_\ell/F_j\ (\ell\in\mathcal{M}_j)$ | 线性算力分母差分性质 | $\frac{C_k(n_i-1)}{F_i}-\frac{C_k n_i}{F_i}=-\frac{C_k}{F_i}$，严格精确 | ✅ 见第三节推导链 7 |
| `(4.7)`–`(4.9)` | `eq:chapter4-lex-objective` 至 `eq:ch4-marginal-deadline-count` (`chap04.tex:L133-L165`) | 二元字典序目标 $F(\mathbf{a})=(F_1(\mathbf{a}),F_2(\mathbf{a}))$、边际归一化时延分解 $\Delta F_2$ 与边际超时数分解 $\Delta F_1$ | 多准则字典序组合优化 | 代数分解完全精确 | ✅ 对齐 `utils/chapter4.py:L16-L36` |
| `(4.10)`–`(4.11)` | `eq:ch4-own-cost`, `eq:ch4-one-move-neighborhood` (`chap04.tex:L191-L212`) | 单请求自身代价 $f_m(\mathbf{a})$ 与单步迁移邻域 $\mathcal{N}_1(\mathbf{a})$ | 1-Exchange 局部搜索邻域 | 集合定义清晰 | ✅ 对齐 `utils/assignment_search.py` |
| `(4.12)`–`(4.13)` | `eq:ch4-strict-descent`, `eq:ch4-conditional-local-optimum` (`chap04.tex:L252-L270`) | 严格字典序下降链 $F(\mathbf{a}^{r+1})<_{\mathrm{lex}}F(\mathbf{a}^r)$ 与无改善退出时的 $\mathcal{N}_1$ 弱局部最优性 | 偏序关系与局部搜索收敛性 | 命题 1–3 数学证明严密 | ✅ 见第三节推导链 7 |
| `(4.14)`–`(4.17)` | `eq:ch4-neighbor-count` 至 `eq:ch4-workload-pairing` (`chap04.tex:L300-L367`) | 每轮邻域规模 $H=\sum_m(|\mathcal{C}_m|-1)\le 5K$、总评价次数 $Q_{\mathrm{eval}}=1+RH+r_{\mathrm{acc}}$、精确穷举解 $\mathbf{a}^{\mathrm{ex}}$ 与工作负载配对均值差 | 算法复杂度计数与配对统计 | 计数恒等式精确成立 | ✅ 对齐 `utils/assignment_search.py` |
| `(5.1)`–`(5.3)` | `eq:ch5-arrival` 至 `eq:ch5-edf` (`chap05.tex:L48-L87`) | 突发调制时变泊松到达 $\lambda_i(t)=\lambda_0 w_i(1+A_b\exp(-\cdots))$、跨时隙排空计数恒等式与节点抢占式 EDF 微分/子步推进方程 | 非齐次泊松过程与抢占式 EDF 调度 (Liu & Layland, 1973) | 量纲一致（$\mathrm{cycles}$ 与 $\mathrm{s}$） | ✅ 对齐 `environment/queued_service.py:L94-L115, L254-L320` |
| `(5.4)`–`(5.5)` | `eq:ch5-release-time-stage1`, `eq:ch5-release-time-stage2` (`chap05.tex:L125-L134`) | 第一阶段 CPU 释放时刻估计 $\tilde{r}_{n,j}(s,k,m)$ 与第二阶段节点事件驱动 EDF 推进 | 串联排队释放时间近似 | ❌ **式 (5.4) 忽略了代码中真实运行的上传/回传/机间串行等待递推、离开覆盖截断及 `waiting` 虚拟注入** | ❌ 严重偏离 `utils/queued_joint_rl.py:L87-L309`（见 **[F-ERR-2]**） |
| `(5.6)`–`(5.8)` | `eq:ch5-motion-policy` 至 `eq:ch5-joint-policy` (`chap05.tex:L144-L163`) | 运动条件分布 $p_{\theta_m}(m\mid s,k)$、4 维期望运动反馈向量 $\mathbf{C}_{\mathrm{upper}}(s,k)=\sum_m p_{\theta_m}(m\mid s,k)\boldsymbol{\phi}_{km}$ 与分配分支分布 $p_{\theta_a}(k\mid s)$ | 链式法则与期望特征耦合 | 数学良定义且可微（梯度经 $\mathbf{C}_{\mathrm{upper}}$ 回传至 $\theta_m$） | ✅ 与 `marl_models/queued_joint_rl.py:L110-L150` **100% 一致**（但与第二章式 2.67 冲突） |
| `(5.9)` | `eq:ch5-teacher` (`chap05.tex:L173`) | 规划教师玻尔兹曼分布 $q_T(k,m\mid s)\propto (p_a^0 p_m^0)^{\beta_T}\exp(J_T/\tau_T)$ 与蒸馏 KL 损失 | 策略蒸馏与 DAgger (Rusu et al., 2016; Ross et al., 2011) | 概率分布良定义 | ✅ 对齐 `utils/queued_joint_rl.py` 教师分布生成 |
| `(5.10)`–`(5.12)` | `eq:ch5-real-reward` 至 `eq:ch5-gae` (`chap05.tex:L195-L215`) | 动作前后势差塑形奖励 $r_t^{\mathrm{env}}=1.6\Delta C_t-1.8\Delta O_t+0.08\tilde{R}_{\mathrm{rescue}}-0.10\tilde{M}_t+0.35(\Phi_{\mathrm{post}}(s_t^+)-\Phi_{\mathrm{pre}}(s_t))$、势函数定义与带排空吸收态的 GAE | 势函数启发式稠密奖励塑形 + GAE | ⚠️ 未乘 $\gamma$ 且 $\Phi_{\mathrm{post}}$ 含新覆盖 `waiting` 任务，故不满足 Ng et al. (1999) 严格策略不变性（正文已如实声明为启发式） | ✅ 与 `utils/queued_joint_rl.py:L556-L645` **逐系数完全一致** |
| `(5.13)` | `eq:ch5-loo` (`chap05.tex:L221`) | 留一均值基线 $b_{\mathrm{LOO}}^{(-i)}(s)=\frac{1}{G-1}\sum_{j\neq i}Q(s,u^{(j)})$ | Leave-One-Out 多样本基线 (Kool et al., 2019) | 独立同分布采样下 $b_{\mathrm{LOO}}^{(-i)}$ 与第 $i$ 个动作条件独立，故梯度期望严格无偏 | ✅ 见第三节推导链 8 |
| `(5.14)`–`(5.16)` | `eq:ch5-cf-score` 至 `eq:ch5-ppo` (`chap05.tex:L233-L259`) | CRN 三步反事实分支评分 $Q_{\mathrm{cf}}(s,k,m)$、组内标准化优势 $\hat{A}_{\mathrm{cf}}$ 与混合 PPO+反事实+KL 目标 $\mathcal{L}(\theta)$ | 共同随机数（CRN）+ 组相对优势 (Shao et al., 2024 GRPO) | 数学结构合理，有效降低突发到达外生方差 | ⚠️ 正文 L260 对冷启动 RL 的 $\pi_{\mathrm{ref}}$ 描述与代码 `prev_model` 不符（见 **[F-ERR-3]**） |
| `(5.17)`–`(5.18)` | `eq:ch5-entropy-calibration`, `eq:ch5-mpc-score` (`chap05.tex:L319-L334`) | 归一化熵 $\bar{H}(s)$ 自适应温度先验融合 $\widetilde{p}(k,m)$ 与 Top-6 MPC 综合评分 $J(s,k,m)$ | 熵自适应模型预测控制（MPC） | $\bar{H}(s)\in[0,1]$，$\alpha(s)\in[0.35,0.85]$，凸组合后重新归一化为合法分布 | ✅ 与 `queued_joint_rl_experiment.py:L220-L295` **100% 精确一致** |

---

## 三、 核心公式链的完整数学推导与证明（Step-by-Step Mathematical Derivations）

应用户要求（*“所有的公式你需要给出正确的推导步骤，来帮我看看能否推导正确”*），以下对全文 **8 条核心公式推导链**给出不跳步的完整数学推导，并标出每一处推导成立的前提条件与代码对齐检验。

---

### 推导链 1：端到端 DSR 计数分解与配对差分的精确双线性分解（式 1.1、式 3.1–3.4、式 5.2）

1. **单评估单元计数恒等式（式 1.1、式 3.1–3.2）**：
   设某评估单元实际生成的服务请求总数为 $G>0$，其中进入 $100\,\mathrm{m}$ 物理覆盖半径被自然接纳的请求数为 $A$，未覆盖直接损失数为 $U$；在已接纳的 $A$ 个请求中，于截止期 $D_m$ 前完成的成功数为 $S$，接纳后超时数为 $L$。由互斥完备划分有：
   $$G = U + A,\qquad A = S + L.$$
   定义自然接纳率 $\alpha = A/G$，接纳后条件完成率 $\eta = S/A$（当 $A>0$ 时），则端到端截止期满足率（DSR）恒满足：
   $$\mathrm{DSR} = \frac{S}{G} = \frac{A}{G}\cdot\frac{S}{A} = \alpha\eta.$$
   **推导核验**：恒等成立（当 $A=0$ 时必有 $S=0$，两边均为 $0$）。

2. **跨评估单元求平均时的协方差修正项（式 3.3）**：
   设共有 $N$ 个评估单元，第 $i$ 个单元的接纳率为 $\alpha_i$、条件完成率为 $\eta_i$，算术平均分别为 $\bar{\alpha}=\frac{1}{N}\sum_{i=1}^N \alpha_i$ 与 $\bar{\eta}=\frac{1}{N}\sum_{i=1}^N \eta_i$。展开样本协方差定义：
   $$\mathrm{Cov}(\alpha_i,\eta_i) = \frac{1}{N}\sum_{i=1}^N (\alpha_i-\bar{\alpha})(\eta_i-\bar{\eta}) = \frac{1}{N}\sum_{i=1}^N \alpha_i\eta_i - \bar{\alpha}\bar{\eta}.$$
   移项即得式 (3.3)：
   $$\overline{\mathrm{DSR}} = \frac{1}{N}\sum_{i=1}^N \alpha_i\eta_i = \bar{\alpha}\bar{\eta} + \mathrm{Cov}(\alpha_i,\eta_i).$$
   **推导核验**：完全正确！这解释了为什么表 3.5 中算术均值之积 $\bar{\alpha}\bar{\eta} = 0.5751\times 0.4786 = 0.27524$ 与算术平均 $\overline{\mathrm{DSR}} = 0.27351$ 之间存在 $-0.00173$ 的负协方差项（因为高突发负载单元接纳率高但条件完成率低，$\mathrm{Cov}(\alpha_i,\eta_i)<0$）。

3. **配对评估单元上 DSR 差值的精确双线性分解（式 3.4）**：
   对同一配对单元下的候选方法 $1$（$(\alpha_1,\eta_1)$）与基线方法 $0$（$(\alpha_0,\eta_0)$），定义均值 $\bar{\alpha}=\frac{\alpha_1+\alpha_0}{2}$，$\bar{\eta}=\frac{\eta_1+\eta_0}{2}$，差值 $\Delta\alpha=\alpha_1-\alpha_0$，$\Delta\eta=\eta_1-\eta_0$。将右端展开：
   $$\begin{aligned}
   \bar{\eta}\Delta\alpha + \bar{\alpha}\Delta\eta
   &= \frac{\eta_1+\eta_0}{2}(\alpha_1-\alpha_0) + \frac{\alpha_1+\alpha_0}{2}(\eta_1-\eta_0) \\
   &= \frac{1}{2}\left(\alpha_1\eta_1 - \alpha_0\eta_1 + \alpha_1\eta_0 - \alpha_0\eta_0 + \alpha_1\eta_1 - \alpha_1\eta_0 + \alpha_0\eta_1 - \alpha_0\eta_0\right) \\
   &= \alpha_1\eta_1 - \alpha_0\eta_0 = \Delta(\mathrm{DSR}).
   \end{aligned}$$
   **推导核验**：二阶交叉项 $\pm\frac{1}{2}\alpha_1\eta_0$ 与 $\mp\frac{1}{2}\alpha_0\eta_1$ 精确相消，**无需任何一阶泰勒近似即恒等成立**！

---

### 推导链 2：有限时域 MDP Bellman 方程、最优算子 $\gamma$-压缩性与策略梯度定理（式 2.12–2.35）

1. **有限时域折扣回报递推与 Bellman 期望方程（式 2.12–2.18）**：
   由式 (2.12) $G_t = \sum_{l=0}^{T-t}\gamma^l r_{t+l} = r_t + \gamma\sum_{k=0}^{T-(t+1)}\gamma^k r_{t+1+k} = r_t + \gamma G_{t+1}$（约定终点边界 $G_{T+1}\equiv 0,\ V_{T+1}^\pi\equiv 0$）。
   对动作值函数 $Q_t^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]$ 应用线性性与马尔可夫性质 (2.11)：
   $$\begin{aligned}
   Q_t^\pi(s,a)
   &= \mathbb{E}_\pi\left[r_t + \gamma G_{t+1}\mid s_t=s,a_t=a\right] \\
   &= \mathbb{E}\left[r_t + \gamma\mathbb{E}_\pi[G_{t+1}\mid s_{t+1}]\;\middle|\; s_t=s,a_t=a\right] \\
   &= \mathbb{E}_{s_{t+1},r_t\sim P(\cdot\mid s,a)}\left[r_t + \gamma V_{t+1}^\pi(s_{t+1})\right],
   \end{aligned}$$
   此即式 (2.15)。再对 $a\sim\pi(\cdot\mid s)$ 求期望：
   $$V_t^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t=s] = \sum_a \pi(a\mid s)\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a] = \mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[Q_t^\pi(s,a)\right],$$
   此即式 (2.16)。进而对优势函数 $A_t^\pi(s,a)=Q_t^\pi(s,a)-V_t^\pi(s)$ 求条件期望：
   $$\mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[A_t^\pi(s,a)\right] = \mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[Q_t^\pi(s,a)\right] - V_t^\pi(s) = V_t^\pi(s) - V_t^\pi(s) = 0,$$
   此即式 (2.18)。**推导完全正确。**

2. **Bellman 最优算子在无穷范数 $\|\cdot\|_\infty$ 下的 $\gamma$-压缩性证明（式 2.19–2.21）**：
   对任意有界函数 $V_1, V_2: \mathcal{S}\to\mathbb{R}$ 及任意状态 $s\in\mathcal{S}$，由式 (2.19) 定义 $(\mathcal{T}^*V)(s)=\max_a \mathbb{E}[r_t+\gamma V(s_{t+1})\mid s_t=s,a_t=a]$。利用基本不等式 $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a)-g(a)|$ 以及三角不等式 $|\mathbb{E}[X]|\le\mathbb{E}[|X|]$，有：
   $$\begin{aligned}
   \left|(\mathcal{T}^*V_1)(s) - (\mathcal{T}^*V_2)(s)\right|
   &\le \max_a \left|\mathbb{E}\left[r_t+\gamma V_1(s_{t+1}) - (r_t+\gamma V_2(s_{t+1}))\;\middle|\; s,a\right]\right| \\
   &\le \gamma \max_a \mathbb{E}\left[\left|V_1(s_{t+1})-V_2(s_{t+1})\right|\;\middle|\; s,a\right] \\
   &\le \gamma \max_a \mathbb{E}\left[\|V_1-V_2\|_\infty\;\middle|\; s,a\right] = \gamma \|V_1-V_2\|_\infty.
   \end{aligned}$$
   对左端关于 $s\in\mathcal{S}$ 取上确界即得式 (2.21)：
   $$\|\mathcal{T}^*V_1 - \mathcal{T}^*V_2\|_\infty \le \gamma \|V_1-V_2\|_\infty.$$
   **推导核验**：严密无误（仅需给 `chap02.tex:L240` 第一行补加 `\nonumber` 以消除空标签编号 `(2.20)`）。

3. **轨迹策略梯度定理、得分函数零期望与因果 Reward-to-Go 化简（式 2.22–2.30）**：
   - **第一步（轨迹密度对数求导，式 2.23–2.24）**：
     轨迹 $\tau=(s_0,a_0,r_0,\dots,s_T,a_T,r_T)$ 的联合密度为
     $$p_\theta(\tau) = \rho_0(s_0)\prod_{t=0}^T \pi_\theta(a_t\mid s_t)P(s_{t+1},r_t\mid s_t,a_t).$$
     取自然对数将乘积化为求和：
     $$\log p_\theta(\tau) = \log\rho_0(s_0) + \sum_{t=0}^T \log\pi_\theta(a_t\mid s_t) + \sum_{t=0}^T \log P(s_{t+1},r_t\mid s_t,a_t).$$
     由于初始分布 $\rho_0$ 与环境转移核 $P$ 均不依赖策略参数 $\theta$，对 $\theta$ 求梯度后这两项恒为零，得式 (2.24)：
     $$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^T \nabla_\theta \log\pi_\theta(a_t\mid s_t).$$
   - **第二步（对数导数技巧导出全回报梯度，式 2.25–2.26）**：
     在正则性条件（积分与求导可交换）下，利用 $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)$：
     $$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau)G(\tau)\,\mathrm{d}\tau = \int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)G(\tau)\,\mathrm{d}\tau = \mathbb{E}_{\tau\sim p_\theta}\left[\left(\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t\mid s_t)\right)\left(\sum_{t'=0}^T \gamma^{t'}r_{t'}\right)\right].$$
   - **第三步（得分函数条件零期望与因果截断 $t'<t$ 项消去，式 2.27–2.29）**：
     对任意给定状态 $s_t$，因 $\int \pi_\theta(a_t\mid s_t)\,\mathrm{d}a_t = 1$，两边对 $\theta$ 求导得式 (2.27)：
     $$\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\;\middle|\; s_t\right] = \int \pi_\theta(a_t\mid s_t)\frac{\nabla_\theta\pi_\theta(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}\,\mathrm{d}a_t = \nabla_\theta \int \pi_\theta(a_t\mid s_t)\,\mathrm{d}a_t = \nabla_\theta 1 = \mathbf{0}.$$
     因此，对于任何先于时刻 $t$ 的历史奖励 $r_{t'}$（$t'<t$），它在给定截止到 $s_t$ 的历史 $\mathcal{H}_t=(s_0,a_0,r_0,\dots,s_t)$ 下是确定常数（与 $a_t$ 条件独立）。由重期望定律：
     $$\mathbb{E}_{\tau}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\gamma^{t'}r_{t'}\right] = \mathbb{E}_{\mathcal{H}_t}\left[\gamma^{t'}r_{t'}\,\underbrace{\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\;\middle|\;\mathcal{H}_t\right]}_{=\mathbf{0}}\right] = \mathbf{0}.$$
     故双重求和中所有 $t'<t$ 的过去奖励项严格为零，仅保留 $t'\ge t$ 的未来奖励项 $\sum_{t'=t}^T \gamma^{t'}r_{t'} = \gamma^t \sum_{l=0}^{T-t}\gamma^l r_{t+l} = \gamma^t G_t$，得式 (2.29)–(2.30)：
     $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^T \gamma^t G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].$$
   - **第四步（状态依赖基线 $b(s_t)$ 的无偏性，式 2.31–2.34）**：
     因 $b(s_t)$ 不依赖动作 $a_t$，同理有：
     $$\mathbb{E}_{s_t,a_t}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)\right] = \mathbb{E}_{s_t}\left[b(s_t)\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\;\middle|\; s_t\right]\right] = \mathbf{0}.$$
     取 $b(s_t)=V_t^\pi(s_t)$ 并对 $G_t$ 先取关于 $(s_t,a_t)$ 的条件期望 $\mathbb{E}[G_t\mid s_t,a_t]=Q_t^\pi(s_t,a_t)$，即得式 (2.34)：
     $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^T \gamma^t A_t^\pi(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].$$
   **推导核验**：全链条 100% 严密正确（仅需给 `chap02.tex` 中间过渡行 `(2.25)`, `(2.28)`, `(2.31)`, `(2.32)` 补加 `\nonumber`）。

---

### 推导链 3：多步 TD 望远镜求和与 GAE 广义优势估计推导（式 2.36–2.45、式 3.28、式 5.12）

1. **$n$ 步优势函数的望远镜消去推导（式 2.37–2.38）**：
   将一步 TD 残差 $\delta_{t+l} = r_{t+l} + \gamma V_\phi(s_{t+l+1}) - V_\phi(s_{t+l})$ 代入 $\sum_{l=0}^{n-1}\gamma^l \delta_{t+l}$：
   $$\begin{aligned}
   \sum_{l=0}^{n-1}\gamma^l \delta_{t+l}
   &= \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \sum_{l=0}^{n-1}\gamma^{l+1}V_\phi(s_{t+l+1}) - \sum_{l=0}^{n-1}\gamma^l V_\phi(s_{t+l}) \\
   &= \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \left(\sum_{k=1}^{n}\gamma^k V_\phi(s_{t+k}) - \sum_{l=0}^{n-1}\gamma^l V_\phi(s_{t+l})\right) \\
   &= \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t) = \widehat{V}_t^{(n)} - V_\phi(s_t) = \widehat{A}_t^{(n)}.
   \end{aligned}$$
   **推导核验**：严格成立。

2. **GAE 指数加权双重求和化简（式 2.39–2.42）**：
   对于 $\lambda\in[0,1)$，无穷视野下 GAE 定义为 $n$ 步优势的几何加权平均（归一化因子为 $\sum_{n=1}^\infty \lambda^{n-1} = \frac{1}{1-\lambda}$）：
   $$\widehat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}\widehat{A}_t^{(n)} = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}.$$
   由于双重求和的指标域为 $\{(n,l)\in\mathbb{Z}^2: n\ge 1,\ 0\le l\le n-1\} = \{(n,l)\in\mathbb{Z}^2: l\ge 0,\ n\ge l+1\}$，由 Fubini/Tonelli 定理交换求和次序：
   $$\begin{aligned}
   \widehat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
   &= \sum_{l=0}^\infty \gamma^l \delta_{t+l} \left[(1-\lambda)\sum_{n=l+1}^\infty \lambda^{n-1}\right] \\
   &= \sum_{l=0}^\infty \gamma^l \delta_{t+l} \left[(1-\lambda)\lambda^l \sum_{k=0}^\infty \lambda^k\right] \\
   &= \sum_{l=0}^\infty \gamma^l \delta_{t+l} \left[(1-\lambda)\frac{\lambda^l}{1-\lambda}\right] = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}.
   \end{aligned}$$
   在有限窗口 $T$ 且引入非终止掩码 $m_j = 1 - d_j \in \{0,1\}$ 时，单步残差为 $\delta_t = r_t + \gamma m_t V_\phi(s_{t+1}) - V_\phi(s_t)$，反向递推为：
   $$\widehat{A}_t = \delta_t + \gamma\lambda m_t \widehat{A}_{t+1} = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l \left(\prod_{j=t}^{t+l-1} m_j\right)\delta_{t+l}.$$
   **推导核验**：式 (2.39)–(2.45) 推导完全严密（仅需给 `(2.40)` 补 `\nonumber`）。
   **下标边界核查**：注意在 [`chap03.tex:L471`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex#L471) 式 (3.28) 与 [`chap05.tex:L210`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L210) 式 (5.12) 中，若时隙下标取 $t\in\{1,\dots,T\}$，求和上限写为 $T-t$（最后一项为 $\delta_T$），对应 $s_{T+1}$ 为终止/截断边界；若按代码中 $0$-indexed（$t\in\{0,\dots,T-1\}$），求和上限为 $T-1-t$（如式 2.42 与 2.45）。建议在式 (3.28) 和 (5.12) 处注明 $t\in\{1,\dots,T\}$ 以与第二章 $t\in\{0,\dots,T-1\}$ 区分。

---

### 推导链 4：PPO 裁剪目标的分段等价性与缩放点积注意力方差推导（式 2.48–2.49、式 2.59–2.63）

1. **PPO-Clip 目标分段形式推导（式 2.48–2.49）**：
   设 $L_t^{\mathrm{CLIP}}(\theta) = \min\left(\rho_t(\theta)\widehat{A}_t,\ \mathrm{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon)\widehat{A}_t\right)$。
   - **情形 1（$\widehat{A}_t \ge 0$）**：因乘数 $\widehat{A}_t \ge 0$ 保序，有 $\mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\widehat{A}_t = \max((1-\varepsilon)\widehat{A}_t, \min(\rho_t\widehat{A}_t, (1+\varepsilon)\widehat{A}_t))$。
     当 $\rho_t < 1-\varepsilon$ 时，$\rho_t\widehat{A}_t < (1-\varepsilon)\widehat{A}_t$，故外层 $\min(\rho_t\widehat{A}_t, (1-\varepsilon)\widehat{A}_t) = \rho_t\widehat{A}_t = \min(\rho_t, 1+\varepsilon)\widehat{A}_t$；
     当 $\rho_t \ge 1-\varepsilon$ 时，$\mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) = \min(\rho_t, 1+\varepsilon)$，外层 $\min(\rho_t, \min(\rho_t, 1+\varepsilon))\widehat{A}_t = \min(\rho_t, 1+\varepsilon)\widehat{A}_t$。
     故当 $\widehat{A}_t \ge 0$ 时，恒有 $L_t^{\mathrm{CLIP}}(\theta) = \min(\rho_t(\theta), 1+\varepsilon)\widehat{A}_t$。
   - **情形 2（$\widehat{A}_t < 0$）**：因乘数 $\widehat{A}_t < 0$ 反序，当 $\rho_t > 1+\varepsilon$ 时，$\rho_t\widehat{A}_t < (1+\varepsilon)\widehat{A}_t$，故外层 $\min$ 取 $\rho_t\widehat{A}_t = \max(\rho_t, 1-\varepsilon)\widehat{A}_t$；当 $\rho_t \le 1+\varepsilon$ 时，$\mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) = \max(\rho_t, 1-\varepsilon) \ge \rho_t$，乘以负数 $\widehat{A}_t<0$ 后有 $\max(\rho_t, 1-\varepsilon)\widehat{A}_t \le \rho_t\widehat{A}_t$，故外层 $\min$ 取 $\max(\rho_t(\theta), 1-\varepsilon)\widehat{A}_t$。
   合并两种情形即得式 (2.49)：
   $$L_t^{\mathrm{CLIP}}(\theta) = \begin{cases} \min(\rho_t(\theta), 1+\varepsilon)\widehat{A}_t, & \widehat{A}_t \ge 0, \\ \max(\rho_t(\theta), 1-\varepsilon)\widehat{A}_t, & \widehat{A}_t < 0. \end{cases}$$
   **推导核验**：100% 精确等价！

2. **缩放点积注意力方差 $\mathrm{Var}(q^\top k)=d_k$ 推导（式 2.59–2.60）**：
   设 $q, k \in \mathbb{R}^{d_k}$ 的各分量相互独立，且 $\mathbb{E}[q_r]=\mathbb{E}[k_r]=0$，$\mathrm{Var}(q_r)=\mathbb{E}[q_r^2]=1$，$\mathrm{Var}(k_r)=\mathbb{E}[k_r^2]=1$。则对每一项乘积 $q_r k_r$：
   $$\mathbb{E}[q_r k_r] = \mathbb{E}[q_r]\mathbb{E}[k_r] = 0,\qquad \mathrm{Var}(q_r k_r) = \mathbb{E}[q_r^2 k_r^2] - (\mathbb{E}[q_r k_r])^2 = \mathbb{E}[q_r^2]\mathbb{E}[k_r^2] - 0 = 1\times 1 = 1.$$
   由独立随机变量之和的方差可加性：
   $$\mathbb{E}[q^\top k] = \sum_{r=1}^{d_k}\mathbb{E}[q_r k_r] = 0,\qquad \mathrm{Var}(q^\top k) = \sum_{r=1}^{d_k}\mathrm{Var}(q_r k_r) = d_k.$$
   除以 $\sqrt{d_k}$ 后，$\mathrm{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \frac{1}{d_k}\mathrm{Var}(q^\top k) = 1$。
   **推导核验**：数学推导完全正确。这也直接证明了为什么 [`marl_models/attention.py:L52,L114`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/attention.py#L52) 中 `get_attention_weights()` 使用 $1/\sqrt{d_{\mathrm{model}}}=1/\sqrt{64}$ 是错误的——因为每个注意力头的点积是在 $d_h=8$ 维子空间上求和（$\mathrm{Var}(q_h^\top k_h)=d_h=8$），必须除以 $\sqrt{d_h}=\sqrt{8}$ 才能使单头点积方差归一！

---

### 推导链 5：第四章批量算力均分下的迁移外部性与二元字典序局部搜索命题证明（式 4.1–4.16 及命题 1–3）

1. **单请求迁移 $i\to j$ 对同节点未迁移任务的时延外部性（式 2.68、式 4.6）**：
   设迁移前分配为 $\mathbf{a}$，请求 $m$ 位于 UAV $i$（$d_m(a_m)=i$，节点负载 $N_i(\mathbf{a})=n_i \ge 1$），迁移后分配为 $\mathbf{a}'=\mathbf{a}^{m\to c}$，请求 $m$ 迁至 UAV $j$（$d_m(c)=j\neq i$，迁移前节点负载 $N_j(\mathbf{a})=n_j \ge 0$）。
   迁移后各节点负载变为：$N_i(\mathbf{a}') = n_i - 1$，$N_j(\mathbf{a}') = n_j + 1$，其余节点 $v\notin\{i,j\}$ 负载 $N_v(\mathbf{a}')=N_v(\mathbf{a})$ 不变。
   - 对留在源 UAV $i$ 上的任意其他请求 $k\in\mathcal{M}_i(\mathbf{a})\setminus\{m\}$（此时必有 $n_i\ge 2$，故 $N_i(\mathbf{a}')=n_i-1\ge 1$），其上传、转发与缓存回源项均不依赖 $\mathbf{a}$，仅计算时延由式 (4.5) 发生变化：
     $$\Delta T_k(\mathbf{a}\to\mathbf{a}') = \Delta T_k^{\mathrm{cmp}} = \frac{C_k(n_i-1)}{F_i} - \frac{C_k n_i}{F_i} = -\frac{C_k}{F_i} < 0.$$
   - 对原已在目标 UAV $j$ 上的任意请求 $\ell\in\mathcal{M}_j(\mathbf{a})$，同理仅计算时延发生变化：
     $$\Delta T_\ell(\mathbf{a}\to\mathbf{a}') = \Delta T_\ell^{\mathrm{cmp}} = \frac{C_\ell(n_j+1)}{F_j} - \frac{C_\ell n_j}{F_j} = +\frac{C_\ell}{F_j} > 0.$$
   **推导核验**：式 (4.6) 代数恒等成立。

2. **边际归一化时延 $\Delta F_2$ 与边际超时数 $\Delta F_1$ 分解（式 4.8–4.9）**：
   当 $K=|\mathcal{M}|>0$ 且 $D_r \ge 10^{-9}\,\mathrm{s}$ 时，$F_2(\mathbf{a}) = \frac{1}{K}\sum_{r\in\mathcal{M}}\frac{T_r(\mathbf{a})}{D_r}$。由于在集合 $\mathcal{H}=\{m\}\cup(\mathcal{M}_i(\mathbf{a})\setminus\{m\})\cup\mathcal{M}_j(\mathbf{a})$ 之外的所有请求 $r\notin\mathcal{H}$ 满足 $\Delta T_r = 0$，将求和限制在互不相交的三个子集 $\{m\}$、$\mathcal{M}_i\setminus\{m\}$、$\mathcal{M}_j$ 上，立得式 (4.8)：
   $$\Delta F_2 = F_2(\mathbf{a}') - F_2(\mathbf{a}) = \frac{1}{K}\left[\frac{\Delta T_m}{D_m} - \sum_{k\in\mathcal{M}_i\setminus\{m\}}\frac{C_k}{F_i D_k} + \sum_{\ell\in\mathcal{M}_j}\frac{C_\ell}{F_j D_\ell}\right].$$
   同理，第一分量 $F_1(\mathbf{a})=\sum_{r\in\mathcal{M}}\mathbf{1}[T_r(\mathbf{a})>D_r]$ 在 $r\notin\mathcal{H}$ 处差分恒为零，故仅需对 $k\in\mathcal{H}$ 求和，得式 (4.9)：
   $$\Delta F_1 = F_1(\mathbf{a}') - F_1(\mathbf{a}) = \sum_{k\in\mathcal{H}}\left(\mathbf{1}[T_k(\mathbf{a})+\Delta T_k > D_k] - \mathbf{1}[T_k(\mathbf{a}) > D_k]\right).$$
   **推导核验**：100% 精确！

3. **第四章命题 1、命题 2、命题 3 的形式化证明核验**：
   - **命题 1（严格字典序下降与对贪心初值的非劣性，式 4.12）**：
     二元字典序关系 $\mathbf{u} <_{\mathrm{lex}} \mathbf{v} \iff (u_1 < v_1) \lor (u_1 = v_1 \land u_2 < v_2)$ 是 $\mathbb{R}^2$ 上的**严格全序（因此是严格偏序，满足非自反性 $\neg(\mathbf{u}<_{\mathrm{lex}}\mathbf{u})$ 与传递性 $\mathbf{u}<_{\mathrm{lex}}\mathbf{v} \land \mathbf{v}<_{\mathrm{lex}}\mathbf{w} \implies \mathbf{u}<_{\mathrm{lex}}\mathbf{w}$）**。算法 4.1 仅当复核后满足 $F(\mathbf{a}^{r+1}) <_{\mathrm{lex}} F(\mathbf{a}^r)$ 时才更新解，故由归纳法与传递性，对任意接受步 $r\ge 1$ 有 $F(\mathbf{a}^r) <_{\mathrm{lex}} F(\mathbf{a}^0)$；若 $r=0$ 则 $F(\mathbf{a}^0)=F(\mathbf{a}^0)$，故恒有 $F(\mathbf{a}^r) \le_{\mathrm{lex}} F(\mathbf{a}^0)$。**证毕。**
   - **命题 2（有限次接受终止性）**：
     因每个请求 $m\in\mathcal{M}$ 的硬候选集满足 $1\le |\mathcal{C}_m|\le 6$ 且 $|\mathcal{M}|=K<\infty$，乘积空间 $|\mathcal{A}|=\prod_{m=1}^K |\mathcal{C}_m| \le 6^K < \infty$ 是有限集。在冻结快照与确定性评价器下，$F:\mathcal{A}\to\mathbb{Z}_{\ge 0}\times\mathbb{R}_{\ge 0}$ 是单值确定映射。若接受序列 $\mathbf{a}^0, \mathbf{a}^1, \dots, \mathbf{a}^r$ 中存在 $0\le p < q \le r$ 使得 $\mathbf{a}^p = \mathbf{a}^q$，则确定性给出 $F(\mathbf{a}^p)=F(\mathbf{a}^q)$；但由命题 1 及 $<_{\mathrm{lex}}$ 的传递性有 $F(\mathbf{a}^q) <_{\mathrm{lex}} F(\mathbf{a}^p)$，与 $<_{\mathrm{lex}}$ 的非自反性矛盾！故接受序列中各分配两两互异，接受次数 $r_{\mathrm{acc}} \le |\mathcal{A}| - 1 < \infty$。引入有限轮数预算 $R_{\max}<\infty$ 后，外层循环至多执行 $\min(R_{\max}, |\mathcal{A}|)$ 轮即终止。**证毕。**
   - **命题 3（“邻域无改善”出口认证 $\mathcal{N}_1$ 弱局部最优性，式 4.13）**：
     当算法在第 $r$ 轮从第 239 行返回时，$\mathrm{best}=\varnothing$。由于第 232–237 行完整遍历了 $\mathcal{N}_1(\mathbf{a}^*)$ 中的每一个邻居 $\mathbf{a}'=\mathbf{a}^{*\,m\to c}$ 且未触发 $\mathbf{v}' <_{\mathrm{lex}} \mathbf{v}$，故对一切 $\mathbf{a}'\in\mathcal{N}_1(\mathbf{a}^*)$ 均有 $\neg(F(\mathbf{a}') <_{\mathrm{lex}} F(\mathbf{a}^*))$。由 $\mathbb{R}^2$ 上字典序的全序三分律（Trichotomy Law），$\neg(F(\mathbf{a}') <_{\mathrm{lex}} F(\mathbf{a}^*)) \iff F(\mathbf{a}^*) \le_{\mathrm{lex}} F(\mathbf{a}')$。**证毕。**

---

### 推导链 6：第五章两阶段释放时间串联排队近似与源码对照（式 5.3–5.5）

1. **真实环境 `Engine` 的抢占式 EDF 演化（式 5.3）**：
   在真实环境 [`environment/queued_service.py:L254-L320`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/environment/queued_service.py#L254-L320) 中，每个宏步 $\Delta t = 1.0\,\mathrm{s}$ 被均分为 $N_{\mathrm{sub}}=20$ 个步长 $\delta\tau = 0.05\,\mathrm{s}$ 的微步。在每个微步内，若任务 $n$ 是计算节点 $j$ 就绪队列 $\mathcal{C}_j(\tau)$ 中截止期最早（平局按 `uid` 最小）的任务 $n_j^*(\tau)$，则其剩余周期减少 $F_j\delta\tau$：
   $$c_n^{\mathrm{rem}}(\tau+\delta\tau) = \max\left(0,\ c_n^{\mathrm{rem}}(\tau) - F_j\delta\tau\,\mathbf{1}\{n=n_j^*(\tau)\}\right),$$
   且代码第 275–295 行实现了微步内的**工作守恒余量顺延（Work-Conserving Rollover）**：若排在队首的任务 $n_1$ 在微步中途耗尽剩余周期（仅用去 $\delta\tau_1 = c_{n_1}^{\mathrm{rem}} / F_j < \delta\tau$），则剩余微步时间 $\delta\tau - \delta\tau_1$ 立即在同一微步内顺延分配给下一个最早截止任务 $n_2$。因此式 (5.3) 与真实 `Engine` 严格对应。

2. **解析近似公式（式 5.4–5.5）与 [`utils/queued_joint_rl.py:L87-L309`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L87-L309) 的精确数学差异推导**：
   - 正文式 (5.4) 写为：
     $$\tilde{r}_{n,j}(s,k,m) = t + \frac{b_n^{\mathrm{up,rem}}}{\hat{R}_{n,u_n}^{\mathrm{edge}}(s,m)} + \mathbf{1}\{j\neq u_n\}\frac{b_n^{\mathrm{fwd,rem}}}{\hat{R}_{u_n,j}^{\mathrm{inter/bh}}(s,m)}.$$
   - 然而，在 [`utils/queued_joint_rl.py:L181-L300`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L181-L300) 的 `_predict_plan_move_metrics_v2` 中，设按相对截止期裕度 $s_n = d_n - t$ 升序排列的任务序列为 $n_1, n_2, \dots$，代码实际执行的递推方程组为（设当前时隙起点相对时间为 $0$）：
     1. **同 `(src, ue)` 对上传串行递推与离开覆盖截断（代码第 200–211 行）**：
        $$\tilde{t}_{n}^{\mathrm{up,raw}} = t_{u_n,\mathrm{ue}_n}^{\mathrm{up,free}} + \frac{b_n^{\mathrm{up,rem}}}{\hat{R}_{n,u_n}^{\mathrm{edge}}(s,m)},$$
        $$\tilde{t}_n^{\mathrm{up}} = \begin{cases} 10.0, & \text{若 } d_n^{(0)}\le R_c \land d_n^{(0.5)}\le R_c \land d_n^{(1.0)}>R_c \land \tilde{t}_n^{\mathrm{up,raw}}>0.75, \\ \tilde{t}_n^{\mathrm{up,raw}}, & \text{其他情形}, \end{cases}$$
        $$t_{u_n,\mathrm{ue}_n}^{\mathrm{up,free}} \leftarrow \tilde{t}_n^{\mathrm{up}}.$$
     2. **回传与机间转发全局串行排队递推（代码第 213–240 行）**：
        - 若 $j = u_n$（本地执行）：$\tilde{r}_{n,j} = \tilde{t}_n^{\mathrm{up}}$；
        - 若 $j = \mathrm{MBS}$（回传卸载）：
          $$\tilde{r}_{n,\mathrm{MBS}} = \max\left(t_{\mathrm{bh}}^{\mathrm{free}},\ \tilde{t}_n^{\mathrm{up}}\right) + \frac{b_n^{\mathrm{fwd,rem}}}{\hat{R}_{u_n,\mathrm{MBS}}^{\mathrm{bh}}(s,m)},\qquad t_{\mathrm{bh}}^{\mathrm{free}} \leftarrow \tilde{r}_{n,\mathrm{MBS}};$$
        - 若 $j \in \mathcal{U}\setminus\{u_n\}$（机间协作转发）：
          $$\tilde{r}_{n,j} = \max\left(t_{\mathrm{inter}}^{\mathrm{free}},\ \tilde{t}_n^{\mathrm{up}}\right) + \frac{b_n^{\mathrm{fwd,rem}}}{\hat{R}_{u_n,j}^{\mathrm{inter}}(s,m)},\qquad t_{\mathrm{inter}}^{\mathrm{free}} \leftarrow \tilde{r}_{n,j}.$$
     3. **第二阶段窗口外完工任务的 `0.85 / 0.15` 折算汇总（代码第 139–145 行 `_simulate_node_release_edf`）**：
        设任务 $n$ 在节点 $j$ 事件驱动 EDF 下的近似完工相对时间为 $\tilde{\tau}_n^{\mathrm{done}}$，相对截止期为 $s_n = d_n - t$，三步窗口截止期为 $s_n^{\mathrm{win}} = \min(3.0, s_n)$。则单任务对 $(\widehat{C}, \widehat{O})$ 的贡献并非二值 $\{0,1\}$，而是分段函数：
        $$(\hat{c}_n, \hat{o}_n) = \begin{cases} (1.0,\ 0.0), & \text{若 } \tilde{\tau}_n^{\mathrm{done}} \le s_n^{\mathrm{win}}, \\ (0.85,\ 0.15), & \text{若 } s_n^{\mathrm{win}} < \tilde{\tau}_n^{\mathrm{done}} \le s_n, \\ (0.0,\ 1.0), & \text{若 } \tilde{\tau}_n^{\mathrm{done}} > s_n \text{（违约过期）}. \end{cases}$$
   **审计结论**：必须将上述递推方程补充至第 5 章 5.2.1 节式 (5.4)–(5.5) 中，才能使论文公式与实际运行的特征提取器完全一致。

---

### 推导链 7：第五章双分支策略耦合梯度、留一基线（LOO）无偏性与熵校准 MPC（式 5.6–5.18）

1. **双分支联合策略因式分解与跨分支梯度传导推导（式 5.6–5.8 vs 第二章式 2.66–2.67）**：
   在第五章式 (5.6)–(5.8) 与 [`marl_models/queued_joint_rl.py:L110-L150`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L110-L150) 中：
   - 给定方案 $k\in\{1,\dots,K\}$，运动分支以 $\mathbf{x}_{km}^{\mathrm{mot}} = [\mathbf{m}_{km}, \mathbf{a}_k, \mathbf{c}] \in \mathbb{R}^{16+14+14=44}$ 为输入，输出条件运动概率：
     $$z_{km}^{\mathrm{mot}}(\theta_m) = g_{\theta_m}(\mathbf{x}_{km}^{\mathrm{mot}}) + \beta_m \log p_m^0(m\mid s,k),\qquad p_{\theta_m}(m\mid s,k) = \frac{\exp(z_{km}^{\mathrm{mot}}(\theta_m))}{\sum_{m'=1}^M \exp(z_{km'}^{\mathrm{mot}}(\theta_m))}.$$
   - 随后计算方案 $k$ 下核心交互特征 $\boldsymbol{\phi}_{km}\in\mathbb{R}^4$ 的条件数学期望（式 5.7）：
     $$\mathbf{C}_{\mathrm{upper}}(s,k;\theta_m) = \sum_{m=1}^M p_{\theta_m}(m\mid s,k)\,\boldsymbol{\phi}_{km}(s) \in \mathbb{R}^4.$$
   - 分配分支以 $\mathbf{x}_k^{\mathrm{alloc}}(\theta_m) = [\mathbf{a}_k, \mathbf{C}_{\mathrm{upper}}(s,k;\theta_m), \mathbf{c}] \in \mathbb{R}^{14+4+14=32}$ 为输入，输出边际分配概率（式 5.8）：
     $$z_k^{\mathrm{alloc}}(\theta_a, \theta_m) = g_{\theta_a}(\mathbf{x}_k^{\mathrm{alloc}}(\theta_m)) + \beta_a \log p_a^0(k\mid s),\qquad p_{\theta_a,\theta_m}(k\mid s) = \frac{\exp(z_k^{\mathrm{alloc}})}{\sum_{k'=1}^K \exp(z_{k'}^{\mathrm{alloc}})}.$$
   - **联合分布归一性验证**：
     $$\sum_{k=1}^K \sum_{m=1}^M \pi_\theta(k,m\mid s) = \sum_{k=1}^K p_{\theta_a,\theta_m}(k\mid s)\underbrace{\sum_{m=1}^M p_{\theta_m}(m\mid s,k)}_{=1} = \sum_{k=1}^K p_{\theta_a,\theta_m}(k\mid s) = 1.$$
   - **跨层梯度回传链式法则推导**：
     当计算分配对数似然 $\log p_{\theta_a,\theta_m}(k^*\mid s)$ 对运动网络参数 $\theta_m$ 的梯度时，由于代码中**没有对 `c_upper` 调用 `.detach()`**（见 [`marl_models/queued_joint_rl.py:L130`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/marl_models/queued_joint_rl.py#L130) `c_upper = torch.einsum("bkm,bkmd->bkd", motion_probs, phi)`），由多元复合函数求导链式法则：
     $$\frac{\partial \log p(k^*\mid s)}{\partial \theta_m} = \sum_{k=1}^K \left(\mathbf{1}\{k=k^*\} - p(k\mid s)\right)\left(\frac{\partial g_{\theta_a}(\mathbf{x}_k^{\mathrm{alloc}})}{\partial \mathbf{C}_{\mathrm{upper}}(s,k)}\right)^\top \sum_{m=1}^M \boldsymbol{\phi}_{km}(s)\frac{\partial p_{\theta_m}(m\mid s,k)}{\partial \theta_m}.$$
     这证明了分配分支的损失确实通过 $\mathbf{C}_{\mathrm{upper}}(s,k)$ 产生了对运动分支参数 $\theta_m$ 的非零解析梯度！同时也再次确认：**第二章式 (2.67) 写的外部标量线性相加 $z_k = z_k^{\mathrm{base}} + \eta_{\mathrm{fb}}\sum_m p_m \mathbf{C}_{\mathrm{upper}}[k,m]$ 是错误的旧草稿公式，必须修正为上述式 (5.7)–(5.8) 的形式。**

2. **留一均值基线（Leave-One-Out Baseline，式 5.13）无偏性证明**：
   设在状态 $s$ 下独立同分布采样 $G\ge 2$ 个动作 $u^{(1)},\dots,u^{(G)}\stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta(\cdot\mid s)$，第 $i$ 个样本的留一基线定义为：
   $$b_{\mathrm{LOO}}^{(-i)}(s) = \frac{1}{G-1}\sum_{j\neq i} Q(s, u^{(j)}).$$
   因为当 $j\neq i$ 时，$u^{(j)}$ 与 $u^{(i)}$ 在给定 $s$ 下相互独立，故 $b_{\mathrm{LOO}}^{(-i)}(s)$ 与 $u^{(i)}$ 条件独立。对第 $i$ 项梯度修正项求条件期望：
   $$\begin{aligned}
   \mathbb{E}_{u^{(1)},\dots,u^{(G)}\stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta}\left[b_{\mathrm{LOO}}^{(-i)}(s)\nabla_\theta\log\pi_\theta(u^{(i)}\mid s)\;\middle|\; s\right]
   &= \mathbb{E}_{u^{(-i)}}\left[b_{\mathrm{LOO}}^{(-i)}(s)\right]\cdot\underbrace{\mathbb{E}_{u^{(i)}\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(u^{(i)}\mid s)\;\middle|\; s\right]}_{=\mathbf{0}} \\
   &= \mathbf{0}.
   \end{aligned}$$
   **推导核验**：严格无偏！同时需注意：在第五章实际实现（式 5.15 及 [`utils/queued_joint_rl.py:L748-L775`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/utils/queued_joint_rl.py#L748-L775)）中，候选集 $\mathcal{B}_t$ 是由“当前采样动作 + 最优动作 + 高先验 Top-$B$ 确定性无放回子集”构成的，且除以了组内经验标准差 $\sqrt{\frac{1}{|\mathcal{B}_t|}\sum (Q_{\mathrm{cf}}-\bar{Q}_{\mathrm{cf}})^2 + 10^{-6}}$（即 Shao et al., 2024 的 GRPO 组标准化形式）。因为确定性子集不满足 i.i.d. 采样且分母标准差依赖全体组内样本，正文 [`chap05.tex:L226-L249`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex#L226-L249) 专门区分了“经典 i.i.d. LOO 理论原型（式 5.13）”与“本章确定性候选组相对标准化辅助优势（式 5.15）”，并明确指出式 (5.15) 是有偏但低方差的辅助引导信号——这一理论界定非常严谨准确！

3. **神经引导 Top-6 MPC 的熵自适应提议先验校准推导（式 5.17–5.18）**：
   对于大小 $|\mathcal{K}\times\mathcal{M}| = KM = 4\times 6 = 24$ 的离散联合动作分布 $\pi_\theta(k,m\mid s)$，其香农熵满足经典不等式：
   $$0 \le -\sum_{k=1}^K \sum_{m=1}^M \pi_\theta(k,m\mid s)\log\pi_\theta(k,m\mid s) \le \log(KM) = \log 24,$$
   故归一化熵 $\bar{H}(s) = \frac{-\sum_{k,m}\pi_\theta(k,m\mid s)\log\pi_\theta(k,m\mid s)}{\log(KM)} \in [0, 1]$。
   进而：
   $$\alpha(s) = \mathrm{clip}\left(1 - 0.55\bar{H}(s),\ 0.35,\ 0.85\right) \in [0.45, 0.85] \subset [0.35, 0.85],$$
   $$T(s) = 1 + 0.6\bar{H}(s) \in [1.0, 1.6].$$
   由于 $p^{\mathrm{T}}(k,m) \propto \pi_\theta(k,m\mid s)^{1/T(s)} > 0$ 且 $\sum_{k,m} p^{\mathrm{T}}(k,m) = 1$，先验分布 $p^0(k,m)=p_a^0(k\mid s)p_m^0(m\mid s,k)$ 亦满足 $\sum_{k,m} p^0(k,m)=1$，二者以权重 $\alpha(s)$ 与 $1-\alpha(s)$ 作凸组合后再归一化得到的 $\widetilde{p}(k,m)$ 恒为 $\mathcal{K}\times\mathcal{M}$ 上的严格正概率分布。
   **推导核验**：式 (5.17)–(5.18) 数学性质完全良定义，且与 [`scripts/analysis/queued_joint_rl_experiment.py:L231-L265`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/scripts/analysis/queued_joint_rl_experiment.py#L231-L265) 逐行吻合。
