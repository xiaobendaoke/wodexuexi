# 《中山大学航空航天学院学位论文撰写模板（2025年12月更新）.doc》逆向提取、LaTeX 结构化校准与 PDF 叠印验证报告

- **权威基准原件**：`《中山大学航空航天学院学位论文撰写模板（2025年12月更新）.doc》`（通过本机 Word COM 无损导出 OOXML `.docx` 与 25 页原生矢量 PDF）
- **校准目标工程**：[`latex/thesis-layout.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex)、[`latex/docs/`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs) 及编译产物 [`latex/main.pdf`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/main.pdf)（共 131 页）
- **校准边界遵循说明**：严格执行**零先验盲测**、**内容与样式分离**、**修改必复测**原则；遵循用户明确指示，**保留论文现有章节组织与内容结构**（如 [`chap06.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap06.tex) 结论章组织、[`abstract.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex) 条目表述与 [`main.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/main.tex) 前置目录取舍），聚焦全部**纯排版几何参数**（纸张版心、页眉双线、字体字号、基线纵坐标 `y_base`、行距 `dy`、左右缩进、表格线宽、双语题注间距、文献引用胶水与零盒模型告警）的像素级对齐。

---

## 一、第一阶段：Word 模板底层参数提取结果与初始 LaTeX PDF 逐项实测偏差总表

### 1.1 Word 全部 18 个分节（Section 1–18）纸张、边距与网格真值

解压解析 `word/document.xml`、`word/styles.xml`、`word/header*.xml` 与 `word/footer*.xml`（换算基准：`1 twip = 0.05 bp`，`1 cm = 28.3464567 bp`），全部 18 个分节共享统一页面几何：

| 全局几何项 | Word XML 原始值（twips） | 公制换算（cm） | PDF 精确点值（bp） | 校准前 `thesis-layout.tex` | 校准后 [`thesis-layout.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L24-L37) | 复测误差 `Δ` |
|---|---|---|---|---|---|---|
| 纸张宽 × 高 | `11906 × 16838` | `21.0009 × 29.7004` | `595.30 × 841.90` | `a4paper` (`595.28 × 841.89`) | `paperwidth=595.30bp, paperheight=841.90bp` | **`0.00 bp`** |
| 上页边距 `top` | `2155` | `3.8012` | `107.75` | `3.80cm` (`107.72 bp`) | `top=107.75bp`（正文首行基线 `123.77 bp`） | **`0.00 bp`** |
| 下页边距 `bottom` | `1701` | `3.0004` | `85.05` | `3.00cm` (`85.04 bp`) | `bottom=85.05bp` | **`0.00 bp`** |
| 左 / 右边距 `left / right` | `1701 / 1701` | `3.0004 / 3.0004` | `85.05 / 85.05` | `3.00cm` (`85.04 bp`) | `left=85.05bp, right=85.05bp` (`425.20 bp`) | **`0.00 bp`** |
| 页眉距顶端 `header` | `1701` | `3.0004` | `85.05` | 中文 `94.13 bp` / 英文 `93.41 bp` | 中文 `94.13 bp` / 英文 `93.41 bp` | **`0.00 bp`** |
| 页脚距底端 `footer` | `1304` | `2.3001` | `65.20` | 页码基线 `774.65 bp` | 页码基线 `774.65 bp`（TNR `9.0 bp`） | **`0.00 bp`** |

### 1.2 Word 分节页眉双线矢量几何真值（5 类分节样式）

Word 页眉双线采用 `thickThinMediumGap`（线间空隙恒为 `0.72 bp`）：`sz=18` 对应上粗 `1.44 bp` + 下细 `0.72 bp`，`sz=24` 对应上粗 `2.16 bp` + 下细 `0.72 bp`。原生导出 PDF 的上下边缘坐标与校准后 `main.pdf` 实测完全重合：

| 分节模块 | Word 模板页 | 上粗线顶边..底边（线宽） | 下细线顶边..底边（线宽） | `main.pdf` 实测中心线与线宽 | 边缘误差 `Δ` |
|---|---|---|---|---|---|
| 中文摘要（`摘要`） | WP9 | `100.70..102.14 bp` (`1.44 bp`) | `102.86..103.58 bp` (`0.72 bp`) | `yc = 101.42 / 103.22 bp` (`1.44 / 0.72 bp`) | **`0.00 bp`** |
| 英文摘要（`Abstract`） | WP10 | `96.35..97.79 bp` (`1.44 bp`) | `98.51..99.23 bp` (`0.72 bp`) | `yc = 97.07 / 98.87 bp` (`1.44 / 0.72 bp`) | **`0.00 bp`** |
| 中文目录 / 正文各章 / 参考文献 / 附录 / 致谢 | WP11, 15–20, 22–25 | `97.70..99.86 bp` (`2.16 bp`) | `100.58..101.30 bp` (`0.72 bp`) | `yc = 98.78 / 100.94 bp` (`2.16 / 0.72 bp`) | **`0.00 bp`** |
| 英文目录（`Contents`） | WP12 | `96.35..98.51 bp` (`2.16 bp`) | `99.23..99.95 bp` (`0.72 bp`) | `yc = 97.43 / 99.59 bp` (`2.16 / 0.72 bp`) | **`0.00 bp`** |
| 末章（`结 论` / WP21） | WP21 | `97.70..99.14 bp` (`1.44 bp`) | `99.86..100.58 bp` (`0.72 bp`) | `yc = 98.42 / 100.22 bp` (`1.44 / 0.72 bp`) | **`0.00 bp`** |

### 1.3 校准前初始 `main.pdf` 实测偏差与修复对照总表

| # | 模块 | 检测项 | Word 原生 PDF 真值 | 校准前 `main.pdf` 实测 | 初始偏差 | 校准后 `main.pdf` 实测 | 复测误差 |
|---|---|---|---|---|---|---|---|
| 1 | 页面几何 | 纸张尺寸与版心 `x0..x1` | `595.30×841.90`, `85.05..510.25` | `595.28×841.89`, `85.04..510.24` | `-0.02 / -0.01 bp` | `595.30×841.90`, `85.05..510.25` | **`0.00 bp`** |
| 2 | 封面 P1 | 密级/编号/学位/题目/三行信息/日期 | `119.81, 135.41, 286.85, 393.05, 566.81..621.29, 738.77` | 正常标题下对齐，但 4+4 行长标题下底部弹性串扰 | 弹性串扰隐患 | 双区定高弹性盒锁定：`738.77 bp`（2 行与 4+4 行均锁定） | **`0.00 bp`** |
| 3 | 中文扉页 P3 | 标题首行 + 三行信息 + 答辩委员会 5 行 + 日期 | `163.73`；三行 `316.49, 345.29, 374.09`；委员会 `437.45, 466.37`；日期 `660.65` | 标题 `162.84`；三行 `319.50..377.10`；委员会 `439.69..468.54` | **`-0.89 / +3.01 / +2.24 bp`** | `163.73`；三行 `316.49..374.09`；委员会 `437.45..466.37`；日期 `660.65` | **`0.00 bp`** |
| 4 | 声明页 P5/P7 | 原创性声明与授权声明标题/正文/落款 | 标题 `133.13`，首行 `207.65`，`dy=33.22` | `133.13`，`207.64`，`dy=33.21` | `-0.01 bp` | `133.13`，`207.64`，`dy=33.21` | **`-0.01 bp`** |
| 5 | 中文摘要 | 同页首段与条目列表右边界一致性 | 全页统一 `x1 = 508.08 bp` | 首段 `508.07 bp`，列表突出至 `510.24 bp` | **`+2.17 bp` 右缘台阶** | 全页段落与列表统一 `x1 = 508.09 bp` | **`+0.01 bp`** |
| 6 | 英文摘要 | 标题与首行基线、`labelsep`、句号后空格 | 标题 `155.21`，首行 `200.69`；`labelsep=1.40 bp` | `labelsep=1.60 bp`；4 处 `.First,` 缺空格 | `+0.20 bp` + 缺空格 | `155.21 -> 200.69 bp`；`labelsep=1.40 bp`；空格已补齐 | **`0.00 bp`** |
| 7 | 中文目录 | `目 录` 标题宽度、前导章间距、后置连续章间距 | 宽 `45.60 bp`；`183.05 -> 201.65 -> 219.77`；后置 `dy=19.44 bp` | 宽 `44.98 bp`；后置 `dy=18.48 bp` | **`-0.62 bp` / `-0.96 bp`** | 宽 `45.60 bp`；`183.06 -> 201.66 -> 219.78`；后置 `dy=19.44 bp` | **`0.00~0.01 bp`** |
| 8 | 英文目录 | `Contents` 字体字距与行距 | 原生 `Times New Roman 12bp`（`LetterSpace=0`），`dy=16.56 bp` | 误设 `LetterSpace=3.78, WordSpace=1.18` 撑散字母 | **`+0.45 bp/字` 失真** | 移除伪字距拉伸，纯正 `TNR 12bp`，`dy=16.56 bp` | **`0.00 bp`** |
| 9 | 一至三级标题 | 章首 `145.37->189.89`，`section` 与连续 `section->subsection` | `145.37->189.89`；连续 `1.2->1.2.1` `dy=30.00 bp` | 连续 `1.2->1.2.1` `dy=29.12 bp` | **`-0.88 bp`** | `145.38->189.90`；连续 `1.2->1.2.1` `dy=30.00 bp` | **`0.00~0.01 bp`** |
| 10 | 四级标题 | `subsubsection` 段前/段后 `dy` 与序号间距 | 段前 `0` 行、段后 `0` 行，固定值 `20 bp`（`dy=20.04 bp`），`aftername=6.84 bp` | `beforeskip=8.90bp, afterskip=8.90bp`（`dy=28.90 bp`） | **`+8.86 bp` 偏大** | 精确补偿 `fixskip` 后实测前后 `dy = 20.04 / 20.04 bp` | **`0.00 bp`** |
| 11 | 双语图题（31 幅） | `图X-Y` -> `Fig.X-Y` 中英首行基线差 `dy` | `13.92 bp`（`10.5 bp` 正体居中，无空格标签） | 单行 `13.92 bp`，但 **14 处双行英文题注骤缩至 `11.24 bp`** | **`-2.68 bp` 挤压跳变** | 全部 31 幅图题（单行与多行）恒为 `13.91~13.92 bp` | **`0.00~0.01 bp`** |
| 12 | 双语表题（24 张） | `表X-Y` -> `TableX-Y` 中英首行基线差 `dy` | `16.32 bp`（`10.5 bp` 正体居中，无空格标签） | 单行 `16.32 bp`，但 **7 处双行英文题注骤缩至 `13.64 bp`** | **`-2.68 bp` 挤压跳变** | 全部 24 张表题（单行与多行）恒为 `16.32 bp` | **`0.00 bp`** |
| 13 | 三线表与越界 | 24 张表线宽（`1.44 / 0.96 bp`）与版心边界 | `1.44 / 0.96 bp`，`x in [85.05..510.25]` | `表4-7` 右越界至 `594.77 bp`，且含 2 处 `式(??)` | **`+84.53 bp` 越界 + `??`** | 24 张表全部收于 `[85.05..510.25]`，`0` 处 `??` | **`0.00 bp`** |
| 14 | 文献引用间距 | `\cite` / `\inlinecite` 前防断行与后防多余胶水 | 紧贴文字，无行首孤立 `[n]`，无多余空白 | `]` 后跟汉字泄露 `CJKecglue`（`4.19~6.13 bp` 空隙） | **`+4.19~6.13 bp` 空隙** | `\unskip\nobreak\kern\z@` + 后置胶水阻断，`0` 孤立行首引用 | **`0.00 bp`** |
| 15 | 结论、文献、附录、致谢 | 标题基线与首行基线 | 结论/附录/致谢 `145.37->186.89`；文献 `145.37->184.97`（缩进 `108.00`） | 文献首条 `183.68 bp`（`-1.29 bp`） | **`-1.29 bp`** | 结论/附录/致谢 `145.37->186.89`；文献 `145.37->184.97`（`108.01`） | **`0.00~0.01 bp`** |
| 16 | 全文字体与标记卫生 | 禁用局部补丁、非标字体及 `Overfull/Underfull` 警告 | `0` 局部补丁、`0` 非标字体、`0` 盒告警 | 1 处 `Courier New`（`\texttt`）、11 处局部 `\tabcolsep`、2 处 `\indent`、2 处 `Underfull \hbox` | 违规局部补丁与告警 | **`0` 局部补丁、`0` 非标字体、`0` `Overfull`、`0` `Underfull`** | **`0`** |

---

## 二、第二阶段：全局宏包结构化重构与正文标记净化实施摘要

### 2.1 全局配置层重构：[`latex/thesis-layout.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex)

1. **精确 `bp` 版心几何与字体统一**（[L1-L45](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L1-L45)）：采用 `paperwidth=595.30bp, paperheight=841.90bp, top=107.75bp, bottom=85.05bp, left=85.05bp, right=85.05bp`；移除英文目录 `LetterSpace=3.78` 伪字距拉伸；统一 `SimSun` 声明消除重复子集警告。
2. **封面与中文扉页双区定高弹性盒**（[L901-L1092](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L901-L1092)）：将标题区封装为定高弹性盒，彻底隔离下半页信息栏、答辩委员会 5 行下划线与底部日期坐标，消除中文扉页 `+3.01 bp / +2.24 bp` 下移漂移，并保证 1–4 行任意长度中英文题目绝不溢页。
3. **一至四级标题与连续标题补偿**（[L244-L310](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L244-L310)）：补齐 `section -> subsection` 连续标题 `+0.88 bp` 补偿（实测 `dy = 30.00 bp`），校准 `subsubsection` 在 `fixskip=true` 下的精确基线补偿（实测前后 `dy = 20.04 bp`，`aftername = 6.84 bp`）。
4. **双语图表题注盒对称化与语义化三线表环境**（[L339-L445](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L339-L445)）：显式声明 `bicaption` 语言并统一单行/多行题注顶底支柱，彻底根治 21 处双行英文题注 `-2.68 bp` 基线骤缩；提供全局 `sysutabular*` 与 `sysutabularx` 环境统一管理满宽三线表。
5. **中英文目录跨层间距状态机与后置章校准**（[L500-L535](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L500-L535)、[L1206-L1348](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L1206-L1348)）：精确复刻中文目录首部 `摘 要 -> Abstract -> 第1章`（`183.06 -> 201.66 -> 219.78 bp`）、`节 -> 章`（`17.58 bp`）与后置连续章 `结 论 -> 参考文献 -> 附录 -> 致谢`（`19.44 bp`）三类间距；校准 `\conclusionchapter`、`参考文献` 首条 `[1]`（`184.97 bp`）及 `acknowledgements` 首行（`186.89 bp`）。
6. **文献引用胶水精确阻断**（[L1222-L1258](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/thesis-layout.tex#L1222-L1258)）：在 `\NAT@citesuper` 与 `\NAT@citenum` 中通过 `\unskip\nobreak\kern\z@` 与后置零宽汉字锚点消除 `]` 后多余中英间距并杜绝行首孤立 `[序号]`。

### 2.2 正文标记净化范围（零局部排版硬编码）

- [`latex/docs/abstract.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/abstract.tex)：仅补齐英文摘要 4 处英文句号后缺失的空格（`. First,`），保留原有条目结构与文字。
- [`latex/docs/achievements.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/achievements.tex)：移除 2 处手写 `\indent`，统一交由 `achievements` 环境控制缩进。
- [`latex/docs/chap02.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap02.tex)、[`chap03.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap03.tex)、[`chap04.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap04.tex)、[`chap05.tex`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/latex/docs/chap05.tex)：将全部局部 `\setlength{\tabcolsep}` 替换为全局 `sysutabular*` / `sysutabularx` 环境；将 `chap05.tex` 唯一 1 处 `\texttt{...}` 改为标准正体 Times New Roman 排印；在长行内公式断点补充 `\allowbreak`，实现全书 131 页 `Overfull \hbox = 0`、`Underfull \hbox = 0`。

---

## 三、第三阶段：PDF 矢量自动断言与 300 DPI 半透明叠印校准验证

### 3.1 103 项全量矢量坐标与排版卫生自动断言结果

机器可读完整断言数据见 [`validation/vector_assertions.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/vector_assertions.json)：

- **总断言数**：`103` 项（容差阈值 `<= 0.10 bp`）
- **通过率**：**`103 / 103`（100% 通过）**
- **全书最大坐标绝对误差 `max |Δ|`**：**`0.04 bp`**（约 `0.014 mm`，远优于 `0.10 bp` 精度目标）
- **编译日志与字体扫描**：
  - `Overfull \hbox / \vbox` 警告：**`0`**
  - `Underfull \hbox` 警告：**`0`**
  - 未定义引用 `??` 或 `Rerun` 警告：**`0`**
  - 正文非标准字体 span（`Courier / Arial / YaHei / KaiTi / FangSong`）：**`0`**（全书 38 个插图源文件亦全部仅含 `SimSun`、`Times New Roman` 及标准数学符号字体）
  - 行首孤立 `[n]` 引用：**`0`**
  - `latex/docs/**/*.tex` 局部排版硬编码补丁：**`0`**

### 3.2 极限边界隔离编译探针验证（4+4 行长标题 / 实名模式 / 四级标题 / 图表目录 / 附录条目）

在不污染 `latex/` 源码的前提下，于 `.tmp/` 隔离构建探针测试集，实测结果存档于 [`validation/edge_case_probes.json`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/edge_case_probes.json)：

| 边界探针场景 | 检测指标 | Word 基准真值 | LaTeX 探针实测 | 差值 `Δ` | 结论 |
|---|---|---|---|---|---|
| 4 行中文 + 4 行英文极限长标题 + `\sysublindfalse` | 封面 P1 落款日期基线 & P2 空白偶页行数 | `738.77 bp`, `0` 行 | `738.77 bp`, `0` 行 | **`0.00 bp`** | 通过（零溢页，实名正常渲染） |
| 4 行中文 + 4 行英文极限长标题 + `\sysublindfalse` | 中文扉页 P3 落款日期基线 & P4 空白偶页行数 | `660.65 bp`, `0` 行 | `660.66 bp`, `0` 行 | **`+0.01 bp`** | 通过（零溢页） |
| 四级标题 `\subsubsection` (`1.1.1.1`) | 段前基线差 `dy_before` / 段后基线差 `dy_after` | `20.04 / 20.04 bp` | `20.04 / 20.04 bp` | **`0.00 / 0.00 bp`** | 通过 |
| 图目录 `\listoffigures` / 表目录 `\listoftables` | 标题基线 -> 首条基线 | `133.73 -> 180.53 / 174.65 bp` | `133.73 -> 180.53` / `133.74 -> 174.65 bp` | **`0.00~0.01 bp`** | 通过 |
| 附录学术成果列表 `theachievements` | 首行 `x0` / 续行悬挂缩进 `x0` / 行距 `dy` | `85.08 / 112.32 / 19.44 bp` | `85.05 / 112.29 / 19.44 bp` | **`-0.03 / 0.00 bp`** | 通过 |

### 3.3 16 组 300 DPI 半透明叠印校准图索引（Cyan = Word 原生 PDF，Magenta = LaTeX `main.pdf`，深藏青色 = 1:1 重合）

全部 16 张叠印图位于 [`validation/overlays/`](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays)：

1. [01_cover_WP01_vs_MP01.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/01_cover_WP01_vs_MP01.png) — 封面（校名图、密级编号、学位名、标题首行、三行下划线与底部日期重合）
2. [02_cn_inner_title_WP03_vs_MP03.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/02_cn_inner_title_WP03_vs_MP03.png) — 中文扉页（标题首行、全部 8 条信息/委员会下划线与落款日期完全重合）
3. [03_decl_orig_WP05_vs_MP05.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/03_decl_orig_WP05_vs_MP05.png) — 论文原创性声明页
4. [04_decl_auth_WP07_vs_MP07.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/04_decl_auth_WP07_vs_MP07.png) — 学位论文使用授权声明页
5. [05_cn_abstract_WP09_vs_MP09.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/05_cn_abstract_WP09_vs_MP09.png) — 中文摘要页（34 字网格、右缘 `508.08 bp` 及 `1.44/0.72 bp` 页眉双线重合）
6. [06_en_abstract_WP10_vs_MP11.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/06_en_abstract_WP10_vs_MP11.png) — 英文摘要页
7. [07_cn_toc_WP11_vs_MP13.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/07_cn_toc_WP11_vs_MP13.png) — 中文目录页（标题、各级缩进、跨层行距与引导点线逐行重合）
8. [08_en_contents_WP12_vs_MP16.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/08_en_contents_WP12_vs_MP16.png) — 英文 `Contents` 页（原生 `Times New Roman 12bp` 字距与 `16.56 bp` 行网重合）
9. [09_chap1_open_WP15_vs_MP23.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/09_chap1_open_WP15_vs_MP23.png) — 第 1 章章首页与二级标题 `1.1`
10. [10_headings_list_WP17_vs_MP27.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/10_headings_list_WP17_vs_MP27.png) — 正文行网与 `（1）` 编号列表页
11. [11_bicaption_fig_WP16_vs_MP28.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/11_bicaption_fig_WP16_vs_MP28.png) — 双语图题页（`dy = 13.92 bp`）
12. [12_three_line_table_WP19_vs_MP29.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/12_three_line_table_WP19_vs_MP29.png) — 双语表题与三线表页（`dy = 16.32 bp`，线宽 `1.44/0.96 bp`）
13. [13_conclusion_WP21_vs_MP123.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/13_conclusion_WP21_vs_MP123.png) — 结论章首页（`1.44/0.72 bp` 双线与标题基线重合）
14. [14_references_WP22_vs_MP125.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/14_references_WP22_vs_MP125.png) — 参考文献首页（`184.97 bp` 首条基线与 `108.00 bp` 悬挂缩进重合）
15. [15_appendix_WP23_vs_MP130.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/15_appendix_WP23_vs_MP130.png) — 附录首页
16. [16_acknowledgements_WP25_vs_MP131.png](file:///c:/Users/woshinibaba/Documents/学习/我的项目/我的论文/wodexuexi/output/word_latex_calibration_20261111/validation/overlays/16_acknowledgements_WP25_vs_MP131.png) — 致谢页（标题、正文四行网格与页码完全重合）

---

## 四、可复现编译与检查命令

1. **完整四步 LaTeX 编译（在 `latex/` 目录下执行）**：
   ```powershell
   Set-Location latex
   xelatex -halt-on-error -interaction=nonstopmode main.tex
   bibtex main
   xelatex -halt-on-error -interaction=nonstopmode main.tex
   xelatex -halt-on-error -interaction=nonstopmode main.tex
   Set-Location ..
   ```
2. **仓库目录规范检查（在仓库根目录下执行）**：
   ```powershell
   python scripts/maintenance/check_layout.py
   ```
