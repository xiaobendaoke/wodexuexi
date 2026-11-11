# 面向时限计算服务的无人机辅助边缘计算协同控制（去能耗定稿版）

本仓库为中山大学硕士学位论文《面向时限计算服务的无人机集群轨迹规划、批量任务分配与动态排队协同控制研究》的官方代码、实验数据与 LaTeX 论文源码仓库。

---

## 一、核心研究口径（去能耗版）

全系统由 5 架旋翼无人机（UAV）、100 个地面移动终端（UE）与 1 个宏基站（MBS）在 $700\,\mathrm{m}\times 700\,\mathrm{m}$ 区域内协同运行。全文聚焦通信传输、边缘计算与截止期服务完成质量，**不包含飞行推进能耗、计算/通信能耗、无线携能传输（WPT）或终端电量退网机制**，核心主评价指标为**截止时间满足率（Deadline Satisfaction Rate, DSR）**：
$$\mathrm{DSR} = \frac{S}{G}$$
其中 $G$ 为全部实际生成的服务请求总数，$S$ 为在截止时间前完成通信与计算的服务请求总数。

---

## 二、论文各章与代码/数据对应关系

| 章节 | 研究协议与核心方法 | 核心代码实现 | 正式图源数据与绘图脚本 | 论文源码与图表 |
|---|---|---|---|---|
| **第 2 章** | MEC 三层架构、UAV 辅助通信与强化学习交互基础 | — | `docs/figures/scripts/generate_foundation_figures.py` | `latex/docs/chap02.tex`、`docs/figures/图2-1`~`图2-3` |
| **第 3 章** | **协议 I（单时隙批量结算）**：下层消融诊断与上层扩展未覆盖需求观测（`associated_plus_uncovered_in_sense`） | `environment/env.py`、`marl_models/attention_mappo/`、`marl_models/offload_mappo/`、`scripts/analysis/run_chapter34_energy_free.py` | `docs/figures/data/chapter34_20261004/`、`docs/figures/scripts/generate_chapter34_figures_20261002.py` | `latex/docs/chap03.tex`、`docs/figures/图3-1`~`图3-9` |
| **第 4 章** | **协议 I（冻结快照重评估）**：严格二元字典序目标 $F(\mathbf{a})=(F_1(\mathbf{a}),F_2(\mathbf{a}))$ 与移动一致迭代重分配（MC-ITR） | `utils/chapter4.py`、`utils/assignment_search.py`、`utils/snapshot_evaluator.py`、`scripts/analysis/run_chapter34_energy_free.py` | `docs/figures/data/chapter34_20261004/`、`docs/figures/scripts/generate_chapter34_figures_20261002.py` | `latex/docs/chap04.tex`、`docs/figures/图4-1`~`图4-8` |
| **第 5 章** | **协议 II（跨时隙持续 EDF 排队）**：释放时间串联排队近似、候选等变双分支联合策略（$\mathbf{C}_{\mathrm{upper}}$）、两轮规划蒸馏、反事实辅助 PPO 微调与 Top-6 MPC | `environment/queued_service.py`、`marl_models/queued_joint_rl.py`、`utils/queued_joint_rl.py`、`scripts/analysis/queued_joint_rl_experiment.py` | `docs/figures/data/queued_joint_rl_20261106/`、`results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/`、`docs/figures/scripts/generate_queued_joint_thesis_figures.py` | `latex/docs/chap05.tex`、`latex/docs/tables/ch5_dynamic_*.tex`、`docs/figures/ch5_dynamic_01`~`06` |

---

## 三、常用命令

### 1. 编译 LaTeX 学位论文
```bash
cd latex
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
编译产物为 `latex/main.pdf`。

### 2. 重新生成论文正式图表
```bash
python docs/figures/scripts/generate_foundation_figures.py
python docs/figures/scripts/generate_chapter34_figures_20261002.py
python docs/figures/scripts/generate_queued_joint_thesis_figures.py
```

### 3. 运行仓库布局检查与单元测试
```bash
python scripts/maintenance/check_layout.py
python tests/run_all_tests.py
```

---

## 四、`recent/` 历史归档目录说明

- 所有早期研究阶段的历史文件（包括旧版带能耗/强制接纳的实验脚本与说明文档、早期 `SF-HLM` 过渡版代码、第 5 章中间探索实验结果、未使用的旧图及 LaTeX 模板示例图）已按**原相对路径**完整归档至根目录下的 [`recent/`](./recent/README.md) 文件夹中（详见 `docs/provenance/RECENT_ARCHIVE_MANIFEST_20261109.json`）。
- 常规论文修订与代码维护请仅使用当前主工作区中的文件，切勿混用 `recent/` 中的旧版口径。如需恢复或查阅某项历史文件，请参见 [`recent/README.md`](./recent/README.md)。
