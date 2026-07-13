# 修后复核与投稿可行性判断（2026-07-02）

## 复核结论

本轮已完成上一版全链路审核中列为“必须修改”和主要“建议修改”的口径问题修正。修正后，当前论文版本可用于硕士答辩和校内正式提交；若目标是期刊或会议正式投稿，当前稿件具备投稿基础，但不建议以学位论文形态原封不动投稿，需要进一步压缩成期刊论文结构并强化创新性表述、统计检验呈现和可复现材料。

综合判断：

- 硕士答辩/校内提交：可以正式使用。
- 中文期刊投稿：小修和期刊化改写后具备投稿基础。
- SCI/CCF 论文投稿：具备实验基础，但仍需重构论文篇幅、相关工作定位、方法贡献边界和统计显著性呈现后再投。

## 已完成修改

| 位置 | 原风险 | 修后状态 |
|---|---|---|
| `latex/docs/chap03.tex` 指标定义 | DSR 表述仍区分主配置 step-mean 与扩展矩阵 request-weighted，容易和正式 force-admission 表格口径冲突 | 已统一为正式结果优先报告 request-weighted DSR，step/episode DSR 仅作训练和内部诊断 |
| `latex/docs/chap01.tex` 贡献表述 | “均值、95% CI、paired t-test 和 Wilcoxon”表述过满，和部分图表实际呈现不完全一致 | 已改为报告均值和波动范围，配对统计检验作为扩展基线审计补充依据 |
| `latex/docs/chap03.tex` 图 5-11 | 学习率图可能被误读为正式 force-admission 敏感性结论 | 已在正文和图注中定位为“过程性参考” |
| `latex/docs/chap06.tex` 结论 | “显著降低 MBS 依赖”若未逐项给出 p 值，投稿时表述偏强 | 已改为“大幅降低 MBS 依赖”，与均值差异证据匹配 |

## 修后检查结果

### 1. 编译检查

已运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

结果：`latex/main.pdf` 成功生成，大小约 35 MB。日志未检出 `LaTeX Error`、`Fatal`、`Undefined reference`、`Citation undefined` 或 missing figure 类硬错误。

剩余低风险 warning：

- `sysuthesis` 提示 LiShu 字体需求；
- `bicaption` 提示 main language 未设置；
- 第 21 行存在约 12.52 pt 的 overfull hbox；
- 第 454 行存在一个 float 略大，超出约 6.95 pt；
- 少量 underfull hbox。

这些属于模板和版面微调问题，不构成实验或结论硬伤。正式提交前建议目视检查 PDF 中标题、图表和算法页是否美观。

### 2. 旧口径与危险表述检查

已检索以下风险表述：

`主配置表中的 DSR`、`step-mean`、`95% 置信区间、paired t-test`、`显著降低 MBS`、`辅助旧口径`、`暂不纳入`、`原始自然覆盖`、`离线率为 0`、`所有指标最优`、`DSR 最优`、`完全满足所有`、`自然覆盖所有服务`、`后续补跑`

结果：正式核心文件中未发现残留。

### 3. 数据链检查

正式数据源保持一致：

- 主配置/消融/DSR-priority：`results/final_force_admission/evaluation_20260630/variant_statistics.json`
- 扩展基线：`results/baseline_matrix_force_admission_20260626_resume/statistics_workload_balanced.json`
- 敏感性：`results/sensitivity_force_admission/*/summary.json`

对正式结果和敏感性 raw 记录检查 `processed_request_ratio`，未发现不等于 1.0 的记录。当前论文“全服务接入并尝试处理”的口径可守，但仍必须坚持：processed request ratio = 100% 不等于 DSR = 100%，也不等于所有 deadline 均满足。

## 仍需备答的问题

1. 为什么本文方法不是 DSR 最高？
   答：force-admission 保证所有 service request 接入处理链路，但 deadline 满足率仍受链路距离、计算资源和 MBS 软约束影响。本文方法目标不是单独最大化 DSR，而是在 DSR、能耗、公平性和 MBS 依赖之间取得折中。

2. 为什么采用 force-admission？
   答：原始自然覆盖口径会让“服务是否被接入”和“接入后是否按时完成”混在一起。force-admission 将所有 service request 纳入处理链路，再用 request-weighted DSR 衡量 deadline 满足比例，更适合比较卸载与资源调度策略。

3. 处理率 100% 是否等于服务全部成功？
   答：不是。处理率表示请求被接入并尝试执行；DSR 表示满足 deadline 的比例；MBS load 表示依赖远端节点的比例，三者分母和含义不同。

4. MBS 负载约束是否严格满足？
   答：本文采用 Lagrange 软约束抑制 MBS 依赖，不声明每个 episode 都严格低于目标；正式结论表述为降低或抑制 MBS load。

5. 轨迹图能否证明所有约束满足？
   答：不能。轨迹图是固定 seed 的行为示例，用于展示移动模式和飞行距离；综合性能和服务指标以多 seed、多 workload 的统计结果为准。

## 投稿判断

从硕士论文角度看，当前版本已经消除了主要硬伤：指标口径统一、force-admission 数据链闭合、旧自然覆盖口径不再作为正式结论、图表和正文主张基本一致。因此，可以进入正式答辩/校内提交阶段。

从正式投稿角度看，当前工作可以作为论文投稿的实验基础，但还需要做“期刊化重写”：压缩系统模型推导和章节式叙述，突出 1--2 个核心创新点，减少学位论文式过程记录，把敏感性和消融整合成更紧凑的结果叙事，并补充统计显著性或置信区间呈现。若不做这些改写，直接投稿的主要风险不是数据不可守，而是稿件结构冗长、贡献边界不够集中。
