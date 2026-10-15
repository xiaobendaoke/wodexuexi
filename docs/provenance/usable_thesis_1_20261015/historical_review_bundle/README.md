# 无人机集群硕士论文审阅包（更新：2026-10-07）

入口为 latex/main.pdf：143总页，六章有效正文121页。
题目：《面向时限服务的无人机集群协同轨迹规划与计算任务分配方法研究》。
学校模板与六章结构保留，专业及个人信息仍待作者填写。学院源码字段已同步为航空航天学院；现有专业硕士封面未单列院系。

## 编辑与编译

保持本目录层级，在 latex/ 中运行 `make main`，需要 XeLaTeX、BibTeX 和 latexmk。
源码入口为 latex/main.tex，六章正文位于 latex/docs/。本次使用 TeX Live2026。
全部在用图位于 docs/figures/，编译无需训练或下载模型。

## 科学图与数据

39幅实际插图均有矢量PDF、可编辑文字SVG和600dpi PNG。
图源在 docs/figures/scripts/，紧凑原始与派生数据在 docs/figures/data/。
图5-6展示预选单回合的五机轨迹、实际航程和离散机间距，UE散点仅为初始位置。
图5-19展示全部十个确认块的三分支执行比例，不设新统计检验。
重绘需要 Python、matplotlib、numpy、Pillow、PyMuPDF及可用中文字体。
图源优先读取包内 tools/figure_qa/，不依赖另行安装nature技能。

## 证据与验收

169个编号公式、18表、3算法、28条实际参考文献。原数学环境和既有实验数值保留。
本轮入口为 output/thesis_writing_20261004/validation/README.md 与 formalization_20261007.json。
acceptance.json及10-05报告保留为历史验收，当前正文及PDF已正式化改写并适度压缩。
包含来源指纹、独立公式及新数据复现、两段全页视觉报告和全部图形核验。
现行算法依据归档提交 a8bcf9a8731edd68b7bc483e1c959c576807e0d6，分支未合并。
evidence/sf_hlm_archive/ 提供选定算法和协议源码，仅供对照；本包不是完整训练环境。
本轮未重训或新增回放，原模型和大型历史回合仍保存在原服务器／仓库。
所有结论限于任务层规划与原仿真、观测信息条件，MBS使用、学习增量和调用代价仍保留。

文件清单及SHA256见 PACKAGE_MANIFEST.json。ZIP本身的指纹和解包编译核验位于工作区
output/thesis_writing_20261004/validation/package_formalization_20261007.json，该文件不进入ZIP以避免自引用。
未提交、推送或合并；本稿供作者审阅删改。tools/figure_qa/ 的出处见其 PROVENANCE.json。
