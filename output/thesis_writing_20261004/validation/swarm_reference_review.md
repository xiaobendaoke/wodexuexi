# 无人机集群叙事新增引文核验

核验日期：2026-10-04。范围为本次叙事重构新增的两篇文献；未将本记录作为既有全部引文的重新核验。两条 DOI 均经 Crossref HTTP API 成功解析，作者、顺序、标题、期刊、年份、卷期及页码或文章号与第二来源一致。

| BibTeX 标识 | 核验后的书目信息 | 第二来源 | 结果 |
|---|---|---|---|
| `chung2018swarm` | Chung S.-J.; Paranjape A. A.; Dames P.; Shen S.; Kumar V. A survey on aerial swarm robotics. IEEE Transactions on Robotics, 2018, 34(4): 837–855. DOI: 10.1109/TRO.2018.2857475. | [作者机构记录及摘要](https://research.monash.edu/en/publications/a-survey-on-aerial-swarm-robotics/) | Verified |
| `skaltsis2023allocation` | Skaltsis G. M.; Shin H.-S.; Tsourdos A. A review of task allocation methods for UAVs. Journal of Intelligent & Robotic Systems, 2023, 109(4), article 76. DOI: 10.1007/s10846-023-02011-0. | [Springer 正式出版页及摘要](https://link.springer.com/article/10.1007/s10846-023-02011-0/) | Verified |

## 正文归述及证据边界

- `chung2018swarm` 用于绪论“集群协同任务规划与空地服务需求”“集群协同覆盖与需求驱动的空间规划”，以及第二章“任务分工、轨迹规划与飞行执行的层次关系”：航空集群研究包括轨迹生成、任务分配、分布式感知，且需考虑空中平台运动特性。上述范围由机构记录中的原文摘要直接支持。本次未声称通读该综述全文，也未将其视为本文算法性能的证据。
- `skaltsis2023allocation` 用于绪论“协同计算任务分配与模型驱动资源优化”，以及第二章“任务分工、轨迹规划与飞行执行的层次关系”：任务分配文献涉及不同任务与约束，评价关注复杂度、最优性、可扩展性，以及通信方式和不确定性。上述归述由出版方摘要直接支持。本文请求级计算分配、集中下层和信息条件属于本项目定义，未归于该综述的独创结论。
- 应急临时空地服务是作者已确认的应用动机。当前仿真未建模灾情、搜救目标或巡检流程，不据新增综述推导现实应急部署效果。
- 2023 条目以正式文章号 `76` 记录，不将其伪装为页码范围；为兼容现有 BibTeX 样式，`pages` 与 `eid` 均保存该文章号。2023 年 11 月在线发布、12 月卷期归属一致，不构成年份冲突。

## 查询记录

Crossref 查询地址：[Chung 2018](https://api.crossref.org/works/10.1109/TRO.2018.2857475)、[Skaltsis 2023](https://api.crossref.org/works/10.1007/s10846-023-02011-0)。浏览器检索工具对 Crossref 端点返回不可访问；随后用标准库 HTTP 请求成功取得同一端点的出版方登记元数据。失败响应未计入通过依据。

结构化字段、对应正文位置与来源状态见同目录 [swarm_reference_review.json](swarm_reference_review.json)。本次 BibTeX 仅新增上述两条，保留原条目。
