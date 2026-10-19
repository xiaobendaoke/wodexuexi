# 本轮变更与证据索引

实施：2026-10-19；任务书：[写作方案](../../../docs/plans/chapter5_cost_aware_rewrite_20261018.md)。

| 主张/术语 | 证据 | 论文位置 |
|---|---|---|
| 解析缓存混合＋能耗定价（mixture_price），计费DSR | 冻结20块summary.json、formal/report.md | 第五章5.3—5.6，摘要、绪论、结论 |
| 完整分配、合法域、配额守恒、标量成本条件指派 | 封存源码cache_mixture/priced_runtime/calibrated_matching | 第五章5.2—5.4 |
| 能耗与MBS代价、冷启动及未加价区间跨零 | summary.json全部八比较 | 第五章5.6 |
| 原SF-HLM学习配额变体，未计等待DSR | 原十块确认，不与新20块合并 | 附录A—B，正文动机及边界 |
| 旧逐次费用事后评分，状态不回写 | 原延迟复核 | 正文动机，附录B |
| 原固定九格与原PPO推导 | 当前改稿前第五章 | 附录A—B，不能用于新方法跨场景结论 |

术语锁定：计费DSR=实际当次完整软件延迟进入请求反馈后的S/G；名义费用10 ms只用于预测评分；预测分配能耗与环境记录完整机队能耗分开；不把SF-HLM单独用作新解析方法名称。

新旧label：原共享模型eq:ch5-assignment-feasibility/transfer-parts/edge-coefficients/mbs-coefficients/sharing-derivation/prediction保留正文；原学习和旧结果label统一增加legacy:前缀移附录；正文新方法使用cost:前缀。旧图字节及原始科研证据保留。

主证据根：`results/sf_hlm_learning_feasibility_20261016/framework_confirmation_20261018/confirmation_saturated120_v1/`；解释与导出根：`output/sf_hlm_learning_feasibility_20261016/validation/framework_confirmation_20261018/evidence_saturated_v1/formal/`。源码一致性见本任务validation/source_identity.json。哈希只作溯源，本次检查为自审。

## 实际交付与检查

- [论文PDF](../../../latex/main.pdf)：163页；六章正文105页，不含前置页、参考文献和附录。第五章正文页87—101；摘要、绪论与第六章同步。
- [第五章源码](../../../latex/docs/chap05.tex)、[补充附录源码](../../../latex/docs/chap05_supplement.tex)。原PPO推导与教师训练进入附录A，原十块/九格/诊断进入附录B，新确认补充统计及来源为附录C。
- [新图脚本](../../../docs/figures/scripts/generate_ch5_cost_aware_schematics.py)生成两幅概念图；[主配对图](../../../docs/figures/ch5_cost_aware_paired.pdf)与已核验原图逐字节一致，[160条源数据](../../../docs/figures/data/ch5_cost_aware/framework_paid_dsr_source.csv)保留全部八参照与20块。

主结论：计费DSR相对G与同框架二值价格提高约1.028和0.262个百分点。相对G机队能耗均值增加约1.87%，MBS依赖上升；与未加价及冷启动近似参照的区间跨零，未证明价格提高DSR、精确信念不可替代或PPO独有增量。新方法跨负载/算力、持续队列、真实控制通信和软件CPU物理能耗仍未确认。

正文与附录的旧label分配见[内容检查](../validation/content_checks.json)和[任务内差异](../validation/task_only.diff)。第二章将既有外接术语原文接入chap02.tex，仅修正水平覆盖距离、MBS非均分、当前需求/槽截断、计费定义及学习变体的附录指代；原外接文件保留。符号表将实测软件时间记为`c_t^sw`，与优势递推连接掩码`c_t`分开。绪论停止接入含五个未定义文献键及旧主结论的外接草稿，原文件保留；沿用既有可用引用，不新增或虚构书目。这不是全文文献核验。

[编译日志](../validation/compile.log)、[PDF检查](../validation/pdf_checks.json)、[全文在用图来源](../validation/figure_provenance.json)、[关键词逐处归属检查](../validation/keyword_review.txt)、[图形QA](../validation/figures/visual_qa.json)保留本轮自审证据。新图源码预检仅提示没有TIFF；本任务以可编辑SVG与矢量PDF交付，不需要TIFF，PNG仅供预览。两新图最终PDF字形均不低于8 pt；碰撞0错误、0警告；单流程画布的多面板对齐为不适用。主配对图保留原QA及字节比对。

未解决的历史问题：[布局日志](../validation/layout.log)仍报根目录“修改总结报告.md”不在白名单；封面身份字段及模板封面既有12.52 pt盒宽警告保留。未更改模板、题目、封面、致谢或成果；未执行Git提交、推送或合并。图形和内容检查为主代理自审，不能替代作者逐项确认及独立同行审查。
