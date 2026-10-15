# 原论文存档的数据、来源与复核范围

日期：2026-10-15；存档名称：可用论文1号；分支：codex/usable-thesis-1。

## 来源

正文、参考文献、配图与10月14日费用分析来自原写作工作区当前文件。基础提交为c7d87d827740b1a2ae466745a37a3ede27e30200，纸面修改尚未提交，因此另保存逐文件SHA256。

第三章覆盖观测与第五章SF-HLM统计从实验归档提交a8bcf9a8731edd68b7bc483e1c959c576807e0d6按Git对象导出，不复制实验工作树的未跟踪草案。第四章及自然覆盖历史参照从写作分支基础提交导出。

`source_origins.json`列出来源；`snapshot_manifest.json`记录当前存档文件指纹。`original_main.pdf`保留创建存档时原PDF原件；latex/main.pdf是在独立目录重新编译的交付PDF。

## 数据索引

| results子目录 | 用途与边界 |
|---|---|
| coverage_observation_confirmation_20260927 | 第三章10训练种子×5负载独立确认、块/单元表格与训练曲线 |
| coverage_observation_g4_20260927 | 第三章探索pilot，与确认分开解释 |
| whole_system_comparison_20260928 | 自然覆盖整套学习方法主场景比较 |
| potential_dsr_20260928 | 共同潜在请求分母稳健性，不是改变请求生成的反事实 |
| canonical_all_20260920/chapter4 | 第四章快照优化统计与410评估单元表 |
| canonical_all_20260923_fixed/marl | 修正版自然覆盖参照与敏感性统计 |
| bridge_20260928/full | 第四—五章共享轨迹固定快照桥接；不是在线接入收益 |
| lower_attention_pilot_20260924、reward_priority_pilot_20260924、reward_denominator_pilot_20260926 | 负面/不确定开发结果，不能当独立确认 |
| sf_hlm_v2_20261001 | 第五章开发、独立确认、计时、MBS与九格稳健性分析，原协议保留 |
| sf_hlm_random_baselines_20261004 | 强随机/逐请求随机等补充比较 |
| thesis_revision_20261014 | 完整30组逐请求gzip账本、900回合汇总、软件费用和能耗分析 |

40幅图的源数据另在docs/figures/data/；它们是对应结果的精简副本或派生表，不计为新实验。

## 原研究源码与外部大文件

两个tar.gz分别保存写作基础提交与SF-HLM实验提交的环境、模型、共享逻辑、运行/分析脚本、测试与关键协议。它们属于不同源码合同，不能无说明混为一次实验；本次不重新训练。

未全量复制的内容包括：历史检查点、所有旧训练/确认原始回合、未跟踪的大压缩包，以及10月14日单次候选探索的约1.76GB完整输出。它们没有被删除。

原位置包括：

- 本机原写作目录：`../wodexuexi/`，10月14日费用原始数据已完整复制到本存档。
- 本机实验工作树：`/Users/wangpengfei/.codex/worktrees/coverage-optimization/wodexuexi/`。
- 服务器旧主实验：`/home/PengYanghan/Lunwen/wodexuexi/`。
- SF-HLM确认归档：`/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/`；具体原文件路径与哈希保留在results/thesis_revision_20261014/groups/*/*/result.json及协议中。
- Git实验分支：`codex/sf-hlm-results-20261003`，提交a8bcf9a8731edd68b7bc483e1c959c576807e0d6。

无需上述外部位置即可编译论文、读取保留统计、重绘图和复核费用评分数据；完整重训/重放旧策略仍需相应模型、原始回合和服务器环境。本次不声称已经打包全部重训依赖。

## 原件、迁移与历史记录

底稿来自10月7日审阅包，然后以当前正文与图源更新。审阅包原根README/PACKAGE_MANIFEST及evidence移到本目录的historical_review_bundle，原条目通过source_origins.json的entry定位。旧PACKAGE_MANIFEST只描述旧审阅包，不能用来验收当前快照。

原数据JSON中的服务器绝对路径、源码摘要和历史PASS原样保留。它们是来源证据，不表示当前机器存在该路径，也不代表本次验收。当前验收另在output/usable_thesis_1_20261015/validation/。

所有DSR需保持原口径；同种子负载/回合/请求不是新的训练重复。第三章信息收益不等于注意力独立增益；第四章快照优化不等于在线部署；第五章学习小增量和费用风险均保留。

## 独立编译时识别出的未核验草稿

当前写作工作区的chap01/chap02另外输入了三个尚未进入已验收PDF的片段，新增五个未定义引文键，独立编译会变为163页。存档正文移除这三处input调用及指向该草稿1.6节的一句悬空说明，其余已完成的正文修改与费用分析保留；完整当前章节及三个片段保存在unverified_drafts/latex/docs/，不纳入有效正文或证据结论。原工作区文件未改。
