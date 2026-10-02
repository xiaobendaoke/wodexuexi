# 答辩高风险问题库（2026-07-02）

## 1. 为什么 force-admission 后还说 DSR 只有 0.3762？
**标准回答**：force-admission 保证的是所有 service request 被系统接入并尝试处理，不保证全部满足 deadline。DSR 的分母仍是全部生成的 service request，分子是满足 deadline 的请求，因此处理率 100% 和 DSR 0.3762 不矛盾。

## 2. 为什么本文方法不是 DSR 最高，还能说有优势？
**标准回答**：本文方法的目标不是单独最大化 DSR，而是在全服务接入下平衡能耗、公平性和 MBS 依赖。扩展基线中 Uniform 和 Joint MAPPO 的 DSR 更高，但 Proposed 的公平性最高、UAV 侧能耗最低、MBS load 最低，因此优势是综合折中。

## 3. force-admission 会不会人为美化结果？
**标准回答**：它只改变接入集合，把未自然覆盖的 service request 接入最近 UAV；链路速率、时延和能耗仍按真实距离计算。因此它提高的是处理链路覆盖，不会把远距离请求自动判定为 deadline 满足。

## 4. MBS load 约束是否严格满足？
**标准回答**：不是严格逐时隙约束，而是 Lagrange 软约束。实验结论应表述为“抑制/降低 MBS 依赖”。消融中 no_lagrange 的 MBS load 升至 36.48%，说明该机制对抑制 MBS 依赖有效。

## 5. 为什么蓝色 UAV 或部分轨迹贴边？
**标准回答**：轨迹图是固定 seed 的单 episode 行为示例。贴边来自边界裁剪、动态请求分布和策略在该场景下的空间选择；它不作为物理最优轨迹证明。综合结论看多 seed/workload 的统计矩阵。

## 6. 为什么 DSR-priority 没有提高 DSR？
**标准回答**：这正是偏好调节边界。提高 DSR target 或放宽 MBS 目标会改变策略对 MBS 的依赖，但在泛化评估中不一定稳定提升 deadline 满足率，因此本文将其解释为 trade-off 边界，而不是性能提升实验。

## 7. 为什么 attention 辅助实验里 vanilla 某些指标更好？
**标准回答**：attention 不是保证所有单项指标提升的模块，它改变的是轨迹协同和负载分配偏好。正文已将该实验定位为边界分析，正式主结论仍以 force-admission 扩展矩阵为准。

## 8. 处理率 100% 是否意味着所有服务都成功？
**标准回答**：不是。处理率 100% 表示所有 service request 都进入 local/cooperative/MBS 的处理链路；服务成功或 deadline 满足由 DSR 衡量。

## 9. 为什么保留启发式策略 DSR 更高的结果？
**标准回答**：这是为了完整展示权衡。启发式在某些负载下 DSR 高，但通常在公平性、能效或 MBS 依赖方面不占优。本文不回避单项指标差异，而强调多目标折中。

## 10. 现在能否用于答辩？
**标准回答**：可以，但建议先修正 DSR 口径残留和少数表述/排版问题。实验主链条已经完整：主配置、消融、扩展基线、DSR-priority 和敏感性均有 force-admission 结果与审计支撑。
