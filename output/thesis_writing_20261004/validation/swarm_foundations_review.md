# 第二、六章集群叙事修订验证

本次修改仅涉及 `latex/docs/chap02.tex`、`latex/docs/chap06.tex`；本记录保存于既有任务验收目录。

## 内容变化

- 第二章增加集群任务分工、轨迹规划与飞行执行的层次关系；二维运动、联合离散几何和感知邻机三个公式；服务空间与机载资源耦合；分散上层与集中下层的协调组织。既有强化学习推导保留。
- 第六章按需求感知空间部署、跨机资源分工和 SF-HLM 分层决策重述贡献，保留原始性能数值、学习增量和统计边界。
- 展望对应当前证据缺口：动力学可执行轨迹、信息延迟/缺失与协调代价、平台失效及有限续航；不声称已有编队稳定性、连续避碰或真实应急应用验证。

## 代码与文献来源

- `config.py`：五机、100 m 高度、15 m/s、1 s、700 m 场地、100 m 服务半径、460 m 感知、200 m 最小间距。
- `environment/env.py::_apply_actions_to_env`、`_finalize_joint_positions`：边界50 m、有限冲突修复、悬停回退、位移预算和离散状态验证。
- `environment/uavs.py::set_neighbors`：感知邻机按实际欧氏距离筛选；固定高度下等于水平距离。
- [Chung 等集群综述](https://research.monash.edu/en/publications/a-survey-on-aerial-swarm-robotics/)：实际读取作者机构摘要，支撑任务分配、轨迹规划和飞行协调的研究定位。
- [Skaltsis 等任务分配综述](https://link.springer.com/article/10.1007/s10846-023-02011-0)：实际读取出版社摘要及元数据，支撑无人机任务分配的一般研究定位。引文键 `chung2018swarm`、`skaltsis2023allocation` 由绪论修订任务同步加入文献库。

## 静态检查

- latex/docs/chap02.tex: environments balanced; 107 unique local labels; 35203 characters.
- latex/docs/chap06.tex: environments balanced; 8 unique local labels; 5051 characters.
- Existing sampled RL/system equations retained; all 22 checked chapter 6 evidence values retained.
- New labels unique across all latex/docs sources.
- Nine main scenario geometry and timing constants verified by AST without importing config or PyTorch.

LaTeX 全稿编译、页数与逐页视觉检查由主任务在各章节集成后统一执行，本记录不将它们声明为已经通过。
