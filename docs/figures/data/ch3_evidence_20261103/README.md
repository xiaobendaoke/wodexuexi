# 第三章新增证据图源数据

图形脚本：`docs/figures/scripts/generate_ch3_evidence_20261103.py`。
本目录 CSV 均由原实验服务器生成，仅转换为百分点显示，不在本机重新估计统计量。

- `factorial_cell_means.csv`：四组原尺度指标均值。
- `factorial_effects.csv`：四个 DSR 对照，百分点与 95% 配对 t 区间，n=10 训练种子。
- `paired_seed_differences.csv`：主比较 DSR 与共同接纳集合完成率差，序号依冻结种子表顺序。
- `admission_decomposition.csv`：历史与新队列分别计算的精确代数项，未合并队列；不是因果中介估计。
- `boundary_effects.csv`：12 个冻结策略 B-A 结果，百分点与 95% 配对 t 区间，n=3 训练种子，探索性。
- `figure_provenance.json`：输入、脚本、字体 SHA256、运行命令及后端。
- `*.alignment.json`：每幅最终图的服务器布局审计。
- `font_provenance.json`：沿用旧轨迹图宋体字形来源；本次所有中文标签通过字形覆盖检查。

服务器独立写作目录：`/home/PengYanghan/Lunwen/ch3-writeup-20261103/`。
使用已有 drone 环境执行，设定 `LD_LIBRARY_PATH` 和 `OPENBLAS_NUM_THREADS=1`，不修改训练环境或旧证据。
运行参数由 `figure_provenance.json` 记录。

PDF/SVG/PNG 位于 `docs/figures/ch3_{factorial_evidence,admission_diagnostic,boundary_evidence}_20261103.*`。
历史图3-9源图保留；论文中使用新命名的诊断图，编号由 LaTeX 自动更新。

验收例外：通用 Nature 静态检查只接受无衬线字体，本论文沿用宋体与衬线数字，以服从既有学位论文风格。该条记为有理由的字体规则例外，不把原始静态检查伪报为全通过。源 PDF 实测全部文字 9 pt；最终论文嵌入字号另行检查。PDF 为排版原稿，PNG 为 600 dpi 预览，不生成不需要的期刊投稿 TIFF。代数分解两队列的柱条为恒等式的描述性均值，不作两个独立实验之间的推断比较；区间显示在其余配对对照与种子汇总中。
