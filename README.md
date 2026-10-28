# 可用论文2号

当前论文的独立复制存档，建立日期：2026-10-19。参照“可用论文1号”的文件结构，保存当前工作区的论文、配图及支撑资料；按用户随后授权，作为独立Git仓库保存到分支`codex/usable-thesis-2`并推送至原GitHub仓库；原工作区与1号分支不动。

入口：[论文PDF](latex/main.pdf)、[LaTeX主文件](latex/main.tex)、[第五章](latex/docs/chap05.tex)、[补充附录](latex/docs/chap05_supplement.tex)。本次复制的PDF为163页，六章正文105页。

第五章采用计及决策开销的解析缓存混合与能耗定价方法，以新20块计费闭环确认为主证据；原学习配额变体、旧十块、九格及事后费用评分分别保留在附录。两套方法与三种费用协议不能混用。

2026-10-28论文收束：A阶段写作、B阶段开发补证及C0固定预算开发已完成。C0未通过质量与完整时间联合要求，本轮停止C1—C3，不启动新训练或可选6000回合迁移复验。原20块主确认、B阶段开发解释与C0停止依据分别保留，当前版本与核验见[收束记录](docs/reports/chapter5_closure_20261028.md)。上文163页为10月19日复制时的历史页数。

- `latex/`：当前正文、参考文献、模板、图片资产与PDF。
- `docs/figures/`：配图、图源数据及绘图脚本，包含新第五章概念图和完整八参照配对图。
- `results/`：继承1号的第三、四章及原第五章证据，另完整复制本机`sf_hlm_learning_feasibility_20261016`资料。
- `output/sf_hlm_learning_feasibility_20261016/`：新确认报告、审计和封存源码。
- `output/chapter5_cost_aware_rewrite_20261018/`：本轮论文改写与验证记录。
- `docs/provenance/usable_thesis_1_20261015/`：1号的历史来源记录，保留历史身份，不作为2号当前验收。
- `docs/provenance/usable_thesis_2_20261019/`、`output/usable_thesis_2_20261019/validation/`：本次复制清单、指纹与独立编译验证。

编译：`make -C latex main`。需要XeLaTeX、BibTeX和latexmk。未复制Git管理文件、Python环境、无关的新仿真原型、缓存和未被正文引用的大型`latex/docs/figures`旧目录。封面身份、致谢与成果占位按原稿保留。

本分支完整保留本次复制的本机资料，采用分批提交/上传，未删减新确认原始数据。复制仅保留本机已有资料，不表示全部历史训练日志、模型及重放依赖均已从服务器迁回。旧协议和JSON中的历史绝对路径保持原样，不代表当前路径；请结合对应来源说明阅读。未运行新训练、评估或统计分析。
