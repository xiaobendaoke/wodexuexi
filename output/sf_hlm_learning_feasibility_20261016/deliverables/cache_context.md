# 缓存先验与冷暖状态混合：只读机制诊断

关键事实：400个开发快照中100个来自step0，1980条可校准合作边的真实邻居命中率为0。其余300状态的11,479条边，按原状态权重的真实命中率为93.6962%。每状态原权重和均为1，因此冷启动占该预测指标25%，虽然只占边数约14.7%。不能把该加权预测分布当成完整1000步回合的时间分布。

| 分组 | 状态 | 合作边 | 原状态权重 | 真命中率 % | full学习Brier | 原先验Brier | half Brier | cold_guard_half Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 400 | 13459 | 400.0 | 70.2722 | 0.08490829 | 0.26970973 | 0.11001092 | 0.05315455 |
| step0 | 100 | 1980 | 100.0 | 0.0000 | 0.08369609 | 0.90970186 | 0.22742547 | 0.00000000 |
| later | 300 | 11479 | 300.0 | 93.6962 | 0.08531236 | 0.05637902 | 0.07087274 | 0.07087274 |

同样的log loss也完整保留，不能把Brier一种损失的排序称为全部预测性能排序：

| 分组 | full学习Log loss | 原先验 | own代理 | half | cold_guard_half |
|---|---:|---:|---:|---:|---:|
| all | 0.27680279 | 4.30374869 | 3.12921936 | 0.50106985 | 0.34180921 |
| step0 | 0.31501485 | 15.76154879 | 0.00000000 | 0.63704252 | 0.00000000 |
| later | 0.26406544 | 0.48448199 | 4.17229248 | 0.45574562 | 0.45574562 |

原full模型的整体Brier优于half，但后续状态Brier反而差于原先验和half。它在冷启动组降低了严重先验误差，不能据整体均值宣布在实际长回合中更准。冷暖混合及采样权重是一个有依据的目标失配解释，尚不是经干预验证的失败原因。

新固定参照cold_guard_half只读现有公开池：至少一个本地缓存option可见且所有可见本地缓存均miss时，将可校准合作边的half概率置0，其余不变。该规则触发100个状态，当前样本中没有触发后续状态，也没有把实际命中的邻居边置0。其整体Brier=0.05315455，低于full模型0.08490829。规则不是邻居真缓存访问；未见状态仍可能误触发，必须独立实测，不能由本样本的零误触发外推。

以上规则只完成概率诊断，没有新增匹配或计时，不能称其提高了DSR或解决费用反转。后续训练或任务试验必须保留它及其他同信息强规则对照，不能只与旧half比较。

实际加载源码核验：

- neighbor_cache_prior：`/home/PengYanghan/Lunwen/batch-offload-20260928/source_launch_20260929/environment/uavs.py:57`。
- uav_initial_state：`/home/PengYanghan/Lunwen/batch-offload-20260928/source_launch_20260929/environment/uavs.py:94`。
- pending_cache_commit：`/home/PengYanghan/Lunwen/batch-offload-20260928/source_launch_20260929/environment/uavs.py:339`。
- periodic_cache_update：`/home/PengYanghan/Lunwen/batch-offload-20260928/source_launch_20260929/environment/uavs.py:984`。
- popularity_initialization：`/home/PengYanghan/Lunwen/batch-offload-20260928/source_launch_20260929/environment/user_equipments.py:36`。

冻结先验为sigmoid(γ×(容量/平均文件大小−流行度rank))，该函数不读取实际缓存、填充程度或时间。UAV初始缓存为全0；时隙末提交pending缓存，并按实际环境50步周期执行GDSF。服务类别的流行度rank映射在环境初始化时由随机流打乱，因此不能把服务类别编号简单当成跨新工作负载不变的流行度。实际函数原文、行号及源码摘要已保存，未用另一工作树的相似文件代替。

未将UE/服务ID、时间索引或私有邻居缓存加入特征；全局及各source的公开可见缓存数量、比例、缺失标记逐状态保存。全部原预测总体指标复现，通过独立的公开option计数及标量加权损失自审。审计为主代理自审，不是独立审查。

协议SHA `6b106fcfdffcea7e8384f1581f53a7ef1bb91d8cb59deb82beff346879aa0fb1`；诊断SHA `3fc85d268cc6252ef113acf6d5cd5e88d3f36e44ec21c78cc6508ad62852a061`。

下一项可检验假说是把可观察的冷态规则与暖态概率校准分开，修正混合目标；它必须另冻协议并与冷态规则参照比较，先验证暖态预测，再验证实际任务/费用和新负载闭环。本报告不把这一假说当成已有效的方法。原十块确认、论文和历史失败不改写，持续目标未完成。

[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_context_20261018/diagnostic_v1/analysis.json) · [实际源码清单](../../../results/sf_hlm_learning_feasibility_20261016/cache_context_20261018/diagnostic_v1/source_inventory.json) · [逐状态公开上下文](../../../results/sf_hlm_learning_feasibility_20261016/cache_context_20261018/diagnostic_v1/states.json) · [原始审计](../validation/cache_context_20261018/raw_record_audit.json)
