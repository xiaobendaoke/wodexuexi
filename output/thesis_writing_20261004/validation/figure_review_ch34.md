# 第三、四章旧图最终验收（2026-10-04）

本次覆盖正文实际引用的17幅图（第三章9幅、第四章8幅），输出冻结于正式目录 `docs/figures/`。只调整绘图排版和明确标注的构造示意，没有重跑训练或修改科研结果。最后一次导出仅为图3-1，将构造位置 `(430,470)` 调整为 `(400,600)`，五架示意UAV均满足200m最小间距；全部实验源文件仍逐字节一致。

## 验收结果

- 默认 `audit_figure_collisions.py`：17/17，零FAIL、零WARN。所有图的逐幅JSON位于 `validation/figures/`。
- 多面板alignment硬门禁：10幅PASS；其余7幅单面板为NOT APPLICABLE。两脚本均在最终draw后先进行严格alignment，再导出，并以 `check=True` 运行默认碰撞审计。没有关闭审计规则或放宽碰撞阈值。
- 全部17幅PDF通过 `audit_pdf_text.py --min-pt 5`，最小导出字形为6.3pt，常规文字至少8pt。按研究生模板146.6mm正文宽度及当前17处include宽度估计，最小正文中字形约5.221pt；该估计未替代整稿实际渲染。
- 全部图包含PDF、可编辑文字SVG、600dpi PNG。没有复制到 `latex/docs/figures/`。所有最终图文件SHA256已记录在 `figures_ch34_validation.json`，可用于核对最终整稿与页图预览是否同步。
- 使用不存在的 `--coverage-root .tmp/no_external_result_tree` 成功重新生成全部默认17幅图，证明所用六个小源文件足够，不依赖原覆盖工作树。六文件共641148字节，SHA256与原源文件及manifest均一致。
- 已观察全部17幅图的联系表，并检查流程、架构、共享公式与桥接图的独立600dpi预览。长方法tick改为图内短码及图外完整键；图例移出绘图区；去掉会遮住全部重复点的冗余均值注记；保留每个原始重复、均值、误差和单位。大感知圆弧实际裁为轴内线段，避免不可见PDF clip路径造成误报。构造几何对照保持同一组UE位置。
- `python3 -m py_compile` 和 `python3 scripts/maintenance/check_layout.py` 通过。未在本机使用PyTorch、Linux仓库虚拟环境或开展训练。

## 数据与统计口径

独立子代理复核见 `figure_data_review_ch34.md`：确认实验的100单元是两方法各10训练种子×5负载；第四章410单元是9正常配置×4方法×10负载加5方法×10负载Exact-Small；桥接是50任务×8快照=400。确认图按训练种子内5负载平均，第四章主图按10工作负载，桥接差值按10训练种子汇总。训练阴影为种子间±1 SD，敏感性误差为逐项95% t区间，第四章差图为5000次配对bootstrap区间。未把快照或请求充作独立训练重复。

## 通用静态预检的人工判读

两脚本的 `validate_figure.py` 原始JSON均已保留。该工具唯一FAIL为FONT-FAMILY：其规则只接受Arial/Helvetica/sans-serif，不能识别学校模板使用的宋体serif。这是学位模板与通用期刊默认值的差异，实际PDF嵌入Songti SC及字号已核对；未修改工具来强制得到PASS。PNG无TIFF、非89/183mm默认宽度、RNG三个WARN分别对应本任务指定的PDF/SVG/600dpi PNG、当前学位模板include宽度，以及已明示的构造示意/散点x方向抖动/实测配对重抽样；没有合成实验结果。

## 复现入口

```sh
.tmp/figure_env/bin/python docs/figures/scripts/generate_chapter34_figures_20261002.py
.tmp/figure_env/bin/python docs/figures/scripts/generate_2026_revision_figures.py
```

两脚本优先读取 `docs/figures/data/chapter34_20261004/`。原始路径、分支/提交、SHA256、构造参数和最后一次几何修正均见该目录 `source_manifest.json`。第二脚本默认只生成现行图3-2、3-6、3-7；过时第五章图仅在显式 `--include-historical-ch5` 时生成，该兼容路径不在此次验收范围。

本代理已冻结图输出，不再变更。最终整稿重新编译、全页渲染及页数核验由主代理负责。
