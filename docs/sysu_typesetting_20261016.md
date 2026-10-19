# SYSU 专业硕士论文排版与编译记录

日期：2026-10-16。入口为 `latex/main.tex`，交付为 `latex/main.pdf`。

## 模板来源

- 仓库：[SYSU-SCC/sysu-thesis](https://github.com/SYSU-SCC/sysu-thesis)。
- 固定版本：`33eb453bc7455bf0b42744844d8dc7e7e4a1d969`；文档类标记日期为 `2026/06/15`。
- 同步文档类、顺序编码 BibTeX 样式、已有的 BibLaTeX 数字制配套样式及新增的 inline 配套文件；校徽 PDF 与上游一致，无需替换。
- 上游顺序编码样式基于 `gbt7714-2025-numeric.bst`，标记版本为 `2026-07-20 v3.0.1`。实际构建采用 BibTeX + natbib，不混用 BibLaTeX。
- 上游原有许可证保留。来源提交信息及差异保存在 `output/sysu_typesetting_20261016/upstream/`。

## 排版调整

- 保留 `degree=master`、`degree-type=professional` 及航空航天学院元数据。姓名、专业、导师、学号与专业学位类型显示待填信息；英文信息保留原占位符。研究生封面的展示字段沿用模板，未自行增加本科封面的学号、院系栏。
- 中文使用模板原生 Fandol 方案，西文明确指定 Times New Roman / Arial，数学使用 XITS。macOS 预设曾因本机 XeTeX 无法解析 `Kaiti SC Regular` 而失败，未安装或复制系统字体。
- 上游双语图题设置修复了原先的主语言未设置警告；继承上游页眉高度、摘要关键词分隔符与公式编号逻辑。
- `latex/thesis-layout.tex` 缩减封面委员会表格的空白末列，解决原先约 12.52 pt 的超宽。
- 英文摘要移除题名区的空行，将题名与个人信息间隔从 0.5 cm 收为 0.3 cm；关键词前移除额外空白段落，并移除原摘要中关键词的手动断行。摘要文字、字号和正文行距均保留，关键词随正文落在 PDF 第 8 页。
- 最终 PDF 为 149 页，原件为 151 页；消除了仅有英文关键词的孤页及其后随附的空白页。保留双面模板生成的第 2、4、18 页空白页。
- 正文图表、公式与标题经页面缩略图全篇筛查，未发现需要进一步改写章节源码的排版问题。

文档类与上游仅有一处项目改动：将“缺少隶书字体”的提示延迟至实际调用 `\lishu` 时。当前专业硕士封面不调用该命令，因此无需为未使用的字体产生加载警告。其余封面和摘要适配集中在独立配置文件。BibLaTeX `.bbx` 中仅额外规整了上游混合空格和制表符的缩进，不改变样式逻辑。

## 本机编译和预览

环境：MacTeX / TeX Live 2026，XeLaTeX、BibTeX、latexmk；同时需要系统已有的 Times New Roman 和 Arial。无需项目 Linux `.venv`。

在项目根目录运行：

```sh
make -C latex
```

等效命令为 `latexmk -cd -xelatex latex/main.tex`。`latex/latexmkrc` 固定错误即停止的 XeLaTeX 构建方式，latexmk 自动处理引文和交叉引用。`make -C latex clean` 仅清理中间文件；`cleanall` 会同时删除 PDF。Makefile 已移除本项目不存在的演示文稿、模板指南和安装目标。

填写正式完成日期时，将 `latex/sysusetup.tex` 中的 `\newcommand\thesiscompletiondate{}` 改为例如 `\newcommand\thesiscompletiondate{2026-12-01}`；留空则封面显示“完成年月待填写”，不会将编译日期当作完成日期。其余个人信息修改同文件的 `\sysusetup` 字段。

## 验证与保留事项

- 六章正文、符号表、致谢、科研成果及参考文献数据库的 SHA256 与修改前一致；中英文摘要的正文源码逐字一致。
- 正文保留 40 个图环境、22 个表环境，参考文献列表为 31 条。图源、结果数据和来源归档均未修改；未引入未核验草稿，未运行训练、仿真、提交或推送。
- 最终日志：编译成功，0 个超宽/超高、0 个缺字、0 个未定义引用或引文，0 条 LaTeX `Warning`。保留 12 条 `Underfull hbox` 行间疏松诊断，相关页面已视觉检查，未通过降低诊断阈值掩盖。
- BibTeX 保留 4 条真实元数据提示：`lillicrap2016continuous`、`wang2025matching`、`schulman2015high`、`schulman2017proximal` 缺 `urldate`。此次没有据排版任务编造访问日期，也没有修改原文献数据库。
- 致谢内容仍是原简短待完善文本，科研成果页仍为空；正式提交前需要作者补齐。
- `git diff --check` 与 `python3 scripts/maintenance/check_layout.py` 均通过。

原 PDF、原编译日志和源文件指纹位于 `output/sysu_typesetting_20261016/baseline/`；最终诊断及完整性结果位于同级 `validation.json`、`final_main.log`、`final_main.blg` 和 `layout_check.txt`；149 页逐页 PNG、8 张全篇联系表及重点页面大图位于 `final_review/`。

审阅图由 `scripts/maintenance/render_thesis_review.py` 生成，依赖 `pypdfium2` 和 Pillow。当前捆绑 Poppler 缺少可工作的 CJK 映射，故使用 PDFium 完成中文页面渲染；该问题不影响论文编译和字体嵌入。
