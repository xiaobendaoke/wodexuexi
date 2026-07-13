# 主张-证据矩阵（2026-07-02）

| 位置 | 主张 | 证据 | 状态 | 边界/备注 |
|---|---|---|---|---|
| 摘要/贡献 | 所有 service request 均被系统接入并尝试处理 | Force-admission eval + sensitivity audit: processed_request_ratio=100%, generated=processed | supported | 注意：不等同于 deadline 全满足 |
| 摘要/结论 | 本文方法取得最高服务公平性 | 扩展基线 proposed fairness=0.9452，高于 IPPO/Vanilla/Joint/Random/Uniform | supported | 限定在 force-admission 扩展基线矩阵 |
| 摘要/结论 | 本文方法取得最低 UAV 侧能耗 | 扩展基线 proposed energy=49.98M，低于所有扩展基线 | supported | Energy 为 UAV 侧统计口径 |
| 摘要/结论 | 本文方法取得最低 MBS load | 扩展基线 proposed mbs_load=3.59%，低于所有扩展基线 | supported | MBS load 分母为 generated service requests |
| 实验结论 | 本文方法不是 DSR 单项最优 | Proposed DSR=0.3762 < Joint 0.4163 < Uniform 0.4529 | supported | 应主动承认 |
| 主配置 | 完整双层是能耗/公平性折中点 | main_full: energy=49.98M, fairness=0.9452, DSR=0.3762；upper-only DSR=0.4957但能耗=74.65M | supported | 避免说完整双层所有指标最优 |
| 消融 | Lagrange 抑制 MBS 依赖最关键 | no_lagrange MBS load=36.48% vs lower_full 4.76% | supported | 软约束，不是严格约束保证 |
| DSR-priority | 服务质量偏好调节存在边界 | DSR-priority DSR=0.3524 < base 0.3762, MBS load=23.05% > base 3.59% | supported | 负面结果要讲清楚 |
| 敏感性 | 负载/算力变化下保持折中优势 | UE 和 CPU sensitivity：Proposed 多数情况下 DSR>Vanilla，公平性/EEE优势明显 | supported | 不声称 DSR 高于 heuristic |
| 轨迹图 | 固定 seed 下本文方法移动距离较低 | 图5-6 trajectory data/figure | supported as visualization | 不作为约束严格满足证明 |
| 引言/贡献 | 报告 95% CI、paired t-test、Wilcoxon | 部分扩展基线/审计材料支持；正文表格未全部展示 | needs wording | 建议改成范围限定 |
