# 硕士答辩前论文全链路审核报告（2026-07-02）

## 总体判断

**结论：建议小修后答辩。** 当前 force-admission 实验链条已经基本可守：正式结果完整，处理率审计通过，正文主故事已经从“DSR 最优”调整为“全服务接入下的能耗、公平性和 MBS 依赖折中”。但第 3 章 DSR 统计口径仍有一处旧表述残留，必须在答辩前修正；若不修，老师追问 DSR 分母时容易暴露口径冲突。

投稿基础判断：具备整理为中文期刊或硕士论文成果的基础；若按 SCI/CCF 会议标准，仍需要更强的新颖性包装、更多统计检验展示和可复现材料。

## 数据源总表

| 模块 | 数据源 | 样本/统计单元 | 用途 |
|---|---|---:|---|
| 主配置/消融/DSR-priority | results/final_force_admission/evaluation_20260630/variant_statistics.json | N=30 | 正式 force-admission 变体结果 |
| 扩展基线矩阵 | results/baseline_matrix_force_admission_20260626_resume/statistics_workload_balanced.json | workload-balanced N=10 | 跨算法主结论 |
| UE 数量敏感性 | results/sensitivity_force_admission/ue_count/summary.json | 每点每算法 N=30 | 正式鲁棒性结果 |
| UAV 算力敏感性 | results/sensitivity_force_admission/uav_cpu_scale/summary.json | 每点每算法 N=30 | 正式鲁棒性结果 |
| 处理率/分母审计 | docs/defense_experiment_audit_force_admission_eval_full.md; docs/defense_experiment_audit_force_admission_sensitivity.md | 审计报告 | 证明 processed=generated 与分母一致 |

## 核心数据一致性结论

- 完整双层 Proposed：DSR `0.3762`，Energy `49.98M`，Fairness `0.9452`，MBS load `3.59%`，Processed ratio `100.00%`。
- 扩展基线中 Proposed 的公平性最高、UAV 侧能耗最低、MBS load 最低；DSR 低于 Joint MAPPO `0.4163` 和 Uniform `0.4529`，正文已基本按折中优势表述。
- force-admission eval 与 sensitivity 审计均显示 generated/processed 无 mismatch，处理率为 100%；这只能证明“接入并尝试处理”，不证明 deadline 全满足。
- 图 5-16/5-17 已来自 `results/sensitivity_force_admission`，每个变量点每算法 N=30。

## 实验严谨性审核

- **可守部分**：主配置、消融、DSR-priority、扩展基线和敏感性实验均已切到 force-admission 口径；扩展基线覆盖 random/uniform/IPPO/vanilla/joint/proposed，答辩对比面足够。
- **边界部分**：attention 上层辅助实验和学习率曲线不属于 force-admission 主矩阵，应保持“辅助/边界分析”定位。
- **负面结果处理**：DSR-priority 未提升 DSR、启发式/Uniform 在 DSR 单项上可能更高，这些已被解释为多目标权衡，不应隐藏。
- **轨迹图风险**：图 5-6 是固定 seed 行为示例，只能说明移动模式和距离，不可作为所有服务质量或运动约束严格满足的证据。

## 论文问题分级摘要

| 等级 | 数量 | 说明 |
|---|---:|---|
| 必须修改 | 1 | 见问题清单 TSV |
| 建议修改 | 6 | 见问题清单 TSV |
| 可保留但需备答 | 3 | 见问题清单 TSV |

## 必须修改项

1. **DSR 口径冲突**：`latex/docs/chap03.tex:200` 仍写“主配置表中的 DSR 为各时隙 DSR 的 episode 平均”，但当前主配置表采用的是 request-weighted DSR。建议统一为正式 force-admission 结果均优先采用 request-weighted DSR，step-mean 仅用于训练诊断。

## 建议修改项

- 将 `chap01.tex:71` 的“95% CI、paired t-test 和 Wilcoxon 检验”改成更克制的范围表述，避免老师要求全文每个表都给检验。
- 将 `chap06.tex:21` 的“显著降低 MBS 依赖”改成“明显降低”或补充统计检验证据。
- 将图 5-11 继续明确为过程性学习率参考，不纳入正式 force-admission 鲁棒性主结论。
- 最终提交前处理编译日志中的字体、bicaption、overfull 和 float warning。

## 答辩可守结论

可以答辩，但建议先完成上述小修。答辩时主线应固定为：force-admission 保证所有 service request 被接入并尝试处理；DSR 衡量 deadline 满足率；本文方法的优势不是 DSR 单项第一，而是在全服务接入前提下取得最高公平性、最低能耗、最低 MBS load，并在敏感性实验中保持较稳定的能效--公平性--服务质量折中。
