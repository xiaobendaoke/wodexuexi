# 九篇相关工作文献的独立字段与主张核验

核验日期：2026-10-04。核验对象为当前 `latex/reference.bib` 及正文引用，正文归述集中于 `latex/docs/chap01.tex` 的第 40、56、58、71、73、87 行附近。本报告由负责第五章的代理核验第一章作者的归述，属于作者以外的独立核验。只写核验报告，未修改 BibTeX 或正文。

采用 nature-ref-verifier 的字段比对流程。复用 `validation/reference_queries.json` 的 Crossref 响应，针对失败条目补查 DOI 注册元数据，另查出版方、作者机构仓储和本地已有原文提取。查询失败、搜索缓存、摘要和完整原文片段分开标记；有出版社链接并不等于成功取得全文。

## 结论与修复建议

九篇文献的身份均可确认。当前归述没有发现张冠李戴或把本论文结果错误归给其他论文的情形。可以取得的出版方摘要或原文片段支持对应概括；Hu 文使用作者机构接收稿摘要补核，Tao 文的本次在线出版方正文未能载入，仅能用本地已有提取文本的摘要核对。这两项不能记作本次已取得出版方全文。

1. **必须修正 `zhao2024robust` 的末页**：当前 BibTeX 为 `4775--4785`，IEEE 出版方和 Crossref 均为 **4775--4786**。题名宜同时按正式出版题名写 `A Multiagent DRL Approach`；原 `multi-agent` 属拼写形式差异，不构成另一篇论文。[IEEE 正式记录](https://doi.org/10.1109/JIOT.2023.3300718)
2. **`wang2026ccah` 已正式见刊**：核验初始条目写“网络优先出版，正式卷期页码未确定”，当前官网 HTML 元数据与第 7 期目录均明确 **2026, 48(7): 3071–3083**。主代理已在本核验过程中补齐卷期页码并删除旧 note，最新 BibTeX 与现值一致。不要再退回只有首发信息的版本。[出版方文章页](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT260054)，[官网第 7 期目录](https://jeit.ac.cn/)
3. **建议收紧第一章第 40 行的共同概括**：将“轨迹与资源分配的联合优化研究也体现了这一联系”改为“**UAV 位置或轨迹与资源分配的联合优化研究也体现了这一联系**”。Hu 文优化时序轨迹，Yu 文摘要明确是 UAV 位置与通信、计算和任务拆分的联合优化；没有必要让 Yu 文承担动态轨迹设计的归述。[Yu 文 IEEE 摘要](https://doi.org/10.1109/JIOT.2020.2965898)
4. `fan2021uav` 可补 `number={11}`，但该字段不是当前主要错误。`354` 是文章号，当前 `pages={354}` 可保留以适配既有参考文献样式，不能扩展为虚构的页码区间。[Springer 文章页](https://link.springer.com/article/10.1007/s10462-025-11340-5)

## 字段比对

题名的大小写、UAV/MEC 的保护花括号及中英文空格不计为实体差异。下表作者顺序均按现条目与实际来源核对。

| 引用键 | 作者身份与顺序 | 期刊、正式年份与卷期页/文章号 | 当前字段判断 |
|---|---|---|---|
| `hu2019uav` | Xiaoyan Hu；Kai-Kit Wong；Kun Yang；Zhongbin Zheng | IEEE Transactions on Wireless Communications，2019，18(10):4738–4752，DOI 10.1109/TWC.2019.2928539 | 主代理修正后的作者及其余字段与 Crossref、UCL 接收稿记录一致 |
| `yu2020joint` | Zhe Yu；Yanmin Gong；Shimin Gong；Yuanxiong Guo | IEEE Internet of Things Journal，2020，7(4):3147–3159，DOI 10.1109/JIOT.2020.2965898 | 当前字段一致；本次补查 Crossref 已成功，不能继续把先前连接失败记作实体不存在 |
| `he2019uav` | Xinwang Yuan；Zhidong Xie；Xin Tan | Sensors，2022，22(10)，文章号 3854，DOI 10.3390/s22103854 | 当前身份、字段正确；键名保留历史年份不改变实际出版年 2022 |
| `zhao2024robust` | Bin Li；Rongrong Yang；Lei Liu；Junyi Wang；Ning Zhang；Mianxiong Dong | IEEE Internet of Things Journal，2024，11(3):4775–4786，DOI 10.1109/JIOT.2023.3300718 | 作者、刊年、卷期和 DOI 正确，末页需 4785→4786；早期上线于 2023 不要求把正式卷年改为 2023 |
| `tao2026online` | Youwei Tao；Zhufang Kuang；Fen Hou；Anfeng Liu | IEEE Transactions on Cognitive Communications and Networking，2026，12:4159–4173，DOI 10.1109/TCCN.2025.3613487 | 当前字段与注册元数据一致；不能用 DOI 含 2025 推翻正式卷年 2026；旧审计中的 4168 已被现条目更正 |
| `zeng2026uav` | 曾耀平；李怀；陈世森；李金丁 | 浙江大学学报（工学版），2026，60(8):1686–1696，DOI 10.3785/j.issn.1008-973X.2026.08.008 | 当前字段与出版方一致 |
| `li2026sensing` | 李侍阳；朱晓荣 | 电子与信息学报，2026，48(5):2166–2177，DOI 10.11999/JEIT251284 | 当前字段与出版方一致 |
| `wang2026ccah` | 王义君；王雅出；SHAHD Batool；缪瑞新 | 电子与信息学报，2026，48(7):3071–3083，DOI 10.11999/JEIT260054 | 主代理在本次核验中更新后，字段一致；外国作者整名保护写法与官网一致 |
| `fan2021uav` | Mohamad A. Hady；Siyi Hu；Mahardhika Pratama；Zehong Cao；Ryszard Kowalczyk | Artificial Intelligence Review，2025，58(11)，文章号 354，DOI 10.1007/s10462-025-11340-5 | 现作者、题名、年份、卷号与文章号一致，缺期号为可补项；不是旧键暗示的 2021 UAV 论文 |

## 逐篇主张、来源与取得范围

### `hu2019uav`

当前正文只用其支撑位置/轨迹与资源安排存在耦合，没有为其编造本论文的自然接纳或配额机制。作者机构摘要描述 UAV 既可计算，也可中继任务给接入点；联合安排计算、带宽与轨迹，以满足任务和因果等约束并降低加权能耗，支持这一背景概括。[UCL 作者机构接收稿记录与摘要](https://discovery.ucl.ac.uk/id/eprint/10085354/)

字段依据为既有 [Crossref DOI 记录](https://api.crossref.org/works/10.1109/TWC.2019.2928539)。[IEEE 文章页](https://ieeexplore.ieee.org/document/8767017/) 本次只返回极少页面内容，未成功载入摘要或正式全文；作者机构记录明确标注 accepted version。**本次为注册元数据加机构接收稿摘要核验，不是 IEEE 正式全文核验。**

### `yu2020joint`

IEEE 摘要明确 UAV 与边缘云协作，优化 UAV 位置、通信/计算资源和任务拆分，目标涉及用户服务时延与 UAV 能耗；这支持资源与节点位置相互影响的背景。它不直接支持把该文概括为动态轨迹优化，所以提出前述一处措辞建议。[IEEE 摘要及正式元数据](https://doi.org/10.1109/JIOT.2020.2965898)

本次 [Crossref DOI 响应](https://api.crossref.org/works/10.1109/JIOT.2020.2965898) 成功返回四位作者、卷期、页码与 2020 年。仅核对摘要和元数据，未声称全文通读。

### `he2019uav`

正文“基站、UAV 与用户的多层 Stackelberg 交互，以效用与响应选择卸载策略”由出版方摘要、贡献小节及三层模型支持。该文的资源定价、利益主体与本论文的协作两模块有明显区别，正文已经作了这种区分。[MDPI 原文及元数据](https://www.mdpi.com/1424-8220/22/10/3854)

本次搜索返回的出版方文本包含摘要、贡献、领导者/追随者子博弈及算法片段；另以 [Crossref DOI 记录](https://api.crossref.org/works/10.3390/s22103854) 复核字段。**属于原文相关小节核验，不是全篇逐式验证。**

### `zhao2024robust`

正文的通信/计算不确定性、不完整 CSI、复杂度估计误差、轨迹/任务划分/资源联合设计以及 Beta-MAPPO 均在 IEEE 摘要中有直接对应。正文没有把该文的鲁棒设计或有界 Beta 策略冒充本项目已实现的自然接纳与正态裁剪，归述成立。[IEEE 正式摘要与页码](https://doi.org/10.1109/JIOT.2023.3300718)

补核 [作者预印本摘要](https://arxiv.org/abs/2308.12756) 及本地 `docs/literature/extracted/Robust_Computation_Offloading_and_Trajectory_Optimization_for_Multi-UAV-Assisted_MEC_A_Multiagent_DRL_Approach.md` 的摘要、贡献片段。IEEE 搜索结果取得完整摘要和元数据，直接 open 仍只返回少量页面；**未取得可宣称全面核读的 IEEE 正式全文**。末页修订有出版方及 Crossref 双重依据。

### `tao2026online`

本地已有提取文本的摘要和贡献描述在线服务选择、卸载、CPU 频率、轨迹与悬停安排，并采用 Lyapunov 与 DRL 的分层方法。正文对应概括与这些片段一致，没有声称该文研究本论文的五架 UAV 自然覆盖接纳。来源为 `docs/literature/extracted/Tao 等 - 2026 - Online Service Selection and Task Offloading in UAV-Assisted MEC via Deep Reinforcement Learning.md`；这里只读摘要和相关贡献，未重新核读全部提取文本。

字段与 [Crossref DOI 注册记录](https://api.crossref.org/works/10.1109/TCCN.2025.3613487) 一致。[IEEE 正式入口](https://ieeexplore.ieee.org/document/11176805) 和 DOI 页面本次未成功载入摘要/全文，搜索返回的第三方摘要不能冒充取得了出版方原文。**结论为本地已有原文提取支持、在线注册元数据确认；本次出版方在线内容复核存在访问缺口。** 保留当前概括是合理的，但不应把本报告标成九篇全部成功全文核验。

### `zeng2026uav`

出版方摘要明确 UAV 为领导者、UE 为追随者，以联盟形成处理追随者层、精确势博弈处理领导者层，支持正文的部署/卸载、领导者/追随者、联盟与势博弈归述。正文未把该文均衡性质直接继承给本论文，范围合适。[出版方摘要和书目信息](https://www.zjujournals.com/eng/CN/10.3785/j.issn.1008-973X.2026.08.008)

另外直接读取官网 HTML 的 `citation_authors/volume/issue/firstpage/lastpage/doi`，与 BibTeX 一致。**仅核对摘要和元数据，未核读其均衡证明全文。** Crossref 404 不足以否定中文 DOI。

### `li2026sensing`

出版方摘要明确单 UAV 与多个计算节点协作，优化带宽、功率、算力、节点选择、数据量与压缩率，并分为四子问题交替求解，支持正文“巡检感知与 AI 任务、多维资源分解和交替优化”的概括。[出版方摘要](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT251284)

官网 HTML `dc.source` 与各 `citation_*` 字段明确 2026、48(5)、2166–2177、李侍阳/朱晓荣；[同一出版方检索列表](https://jeit.ac.cn/search?q=Ai+Jia-qiu) 亦列这些正式字段。正文未把其单 UAV 巡检结果当作本论文多 UAV 端到端结果。**摘要与字段核验，未完整验证四子问题数学过程。**

### `wang2026ccah`

出版方摘要直接对应微云—边缘—本地三层、UAV 预制缓存、细粒度部分卸载、UAV 选择与 CCAH-MVO 的动态时延/能耗权衡。正文据此作为程序获取和节点选择背景，并说明其部分卸载与本论文不可拆分请求不同，归述成立。[出版方摘要与算法片段](https://jeit.ac.cn/article/doi/10.11999/JEIT260054)

正式出版状态以当前直接读取的官网文章 HTML 元数据及官网第 7 期目录为准：`citation_year=2026`、`citation_volume=48`、`citation_issue=7`、`citation_firstpage=3071`、`citation_lastpage=3083`；当前 `citation_date/citation_online_date=2026-07-10`。本地旧提取是网络首发版，注明首发 2026-04-28，不能用旧版没有卷期的封面推翻最新正式信息。[官网当前目录](https://jeit.ac.cn/)

本地首发提取还提供 [CNKI 首发永久入口](https://link.cnki.net/urlid/11.4494.TN.20260426.1819.002)。本次中文检索曾按题名查询 CNKI/万方，但未取得新的有效返回；不将这些未载入页面计为额外核验来源。**本次摘要、原文算法片段与正式字段核验，未全篇核读。**

### `fan2021uav`

Springer 可访问的 OA 正文第 4.2.1 节及其子节讨论车联网、移动、无线、计算机和 IoT 网络的 MARL 资源分配，支持正文“方法分类与应用条件”的概括。随后信息条件和独立重复的三点是本论文据此展开的研究审视方式，不应重新写成该综述逐字提出了本论文统计协议。[Springer OA 原文](https://link.springer.com/article/10.1007/s10462-025-11340-5)

作者列表、2025、卷 58、文章号 354 与现字段相符；Crossref 另给第 11 期，可补。**核对摘要、指定正文片段与元数据，不是全文逐项事实复核。**

## 验证局限

旧审计报告曾把文献替换或修正后的引用键保留为旧作者/年份。键名不是出版身份，实际题名、作者、DOI 和正文点名须一致；本次检查针对现条目，不沿用旧审计的“完全匹配”标签。

IEEE 多个直连页面及 PDF 入口未能有效载入，官方搜索缓存成功返回 Yu、Li 的摘要及书目信息；这些可支持摘要层面的概括，不能证明全文任何未读取的模型细节。Hu 的作者机构接收稿摘要与正式注册身份对应，但不是成功取得出版社全文；Tao 的现归述依本地已有提取文本核对，在线出版方内容仍有限制。

本报告只审阅所列九篇及其当前正文归述，不保证整份未引用 BibTeX 库都完成核验，也不把所引论文的性能、最优性、均衡或收敛结论移植为本论文的结果。
