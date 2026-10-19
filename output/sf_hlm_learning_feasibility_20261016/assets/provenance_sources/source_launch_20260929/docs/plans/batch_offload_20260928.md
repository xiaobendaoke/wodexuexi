# 批量卸载协调：实现与冻结实验协议

用户批准：保留分层框架和现有主场景；分阶段投入；优先整套方法与章节衔接。本文是实现协议，不是实验结论。禁止自动启动 pilot/confirmation，不提交或推送，不修改论文性能结论。

## 实现与信息边界

- A `lower_mlp_upper_uncovered_obs`：当前 Attention 上层、逐请求 MLP 下层。
- B `lower_batch_upper_uncovered_obs`：逐请求输入增加全批有效请求特征均值及有效数/30。
- C `lower_sequential_upper_uncovered_obs`：B 加本 UAV 前序分配到各动作的请求数/30及已归一化请求大小之和/30。
- D `vanilla_lower_mlp_upper_uncovered_obs`：与 A 完全相同的观测、下层和奖励，仅上层 Vanilla。
- 确认配套方法为 `vanilla_lower_batch_upper_uncovered_obs` 或 `vanilla_lower_sequential_upper_uncovered_obs`。

B/C 参数形状相同；B 前缀通道恒零。顺序沿用环境现有请求顺序，静态质量 mask 不变，不增加全局负载或未来信息。C 更新时按保存动作重建前缀，累加条件 log-prob；padding 不参与均值、计数、前缀和策略损失。critic、隐藏层宽度与学习率沿用无注意力下层。新方法标识写入既有 checkpoint method/config/source/checksum 契约；历史契约不放宽，历史检查点用其冻结源码重放。

主要改动在 `marl_models/offload_mappo/offload_mappo.py`、模型工厂与现有 canonical runner。环境与物理模型不改。新增脚本统一位于 `scripts/analysis/`，测试位于 `tests/`。

## 可执行入口

从对应源码仓库根目录运行。内部输出路径由 `paths.REPO_ROOT` 推导；下面相对命令约定当前目录为根目录，外部冻结源码与数据目录通过参数传入。所有含 PyTorch、环境重放和训练的命令只在远程 Linux 执行。

生成固定清单（不会训练）：

```sh
python3 -m scripts.analysis.batch_offload_protocol pilot
python3 -m scripts.analysis.batch_offload_protocol smoke
```

依赖既有服务器环境及 `run_remote_python.sh`，无需安装软件。独立验证目录中运行：

```sh
bash run_remote_python.sh -m unittest tests.test_batch_offload tests.test_batch_offload_protocol tests.test_canonical_runner -v
bash run_remote_python.sh run_formal_suite.py --plan-file docs/plans/batch_offload_smoke_20260928.json --output .tmp/batch_offload/smoke --workers 1 --threads 1
```

历史低成本诊断：

```sh
bash run_remote_python.sh -m scripts.analysis.batch_offload_diagnostic_suite --source-root /path/to/original/frozen/source --suite-root /path/to/completed/suite --output /path/to/diagnostic --workers 4
```

默认只显示计划，追加 `--run` 才执行。完整诊断重放原 10×5 配对各前 2 回合，每回合预先固定步骤 0/250/500/750，共 400 快照；保留空快照。逐步核对原请求计数、成功数、分配能耗及外生轨迹哈希。按源 UAV 的决策前请求数 1–5、6–10、11–20、21+ 描述失败和共享目标情况，不把请求当独立训练重复。协作预估是 best-neighbor proxy，不能当成所选目标的准确预测；截断估计另行标识。训练最后20回合与前20回合的 DSR 仅描述趋势，不替代 held-out 收敛分析。

## Pilot、晋级和确认

冻结清单见同目录 `batch_offload_pilot_20260928.json`。A/B/C/D 各5新种子，200×1000步；每检查点5工作负载×6回合。20训练、100评估、600评估回合。使用最后检查点。种子命名空间：训练620001–620005、评估720001–720005、训练负载820001–820005；确认分别630001–630010、730001–730005、830001–830010。smoke 使用独立的610001/710001/810001。

获得独立的启动授权后才能运行：

```sh
bash run_remote_python.sh run_formal_suite.py --plan-file docs/plans/batch_offload_pilot_20260928.json --output results/batch_offload_20260928/pilot --workers 1 --threads 1
bash run_remote_python.sh -m scripts.analysis.batch_offload_analysis results/batch_offload_20260928/pilot
```

B/C 相对 A 的 DSR 均值差≥0.01、至少4/5训练种子为正、能耗均值≤1.05倍、完整 `policy_step` p95<1秒，才晋级。两者均满足且 C 相对 B 增益<0.005时选 B，否则选 C。D 不参与候选筛选。未晋级就停止，保留 A。

计时包含上下层推理、下层观测/mask构造与数据转换；CUDA 时同步。包括冷启动样本，不含环境 step、磁盘保存、训练更新。采用同主机、运行环境、线程和顺序作业；p95 是当前硬件软件条件下经验值，不是实时硬保证。

确认清单只能从完整、原始文件校验通过的 pilot 生成：

```sh
bash run_remote_python.sh -m scripts.analysis.batch_offload_protocol confirmation --pilot-root results/batch_offload_20260928/pilot
```

生成不启动训练。确认固定4方法×10训练种子，5工作负载×6评估回合，40训练、200评估、1200回合。主比较是 Attention+晋级下层减 A，以及 Attention+晋级下层减 Vanilla+同晋级下层。两个训练种子配对 t 检验用 Holm 校正；同时报告配对交叉 bootstrap 区间。Vanilla 下层改进及交互项是次级结果。脚本强制完整矩阵、原始校验、外生配对和一致计时元数据；smoke 不能晋级。替换主方法与支持上层架构优势是两项独立判定。

## 第四章桥接与交付条件

确认后，用同一确认源码及完成矩阵执行 `batch_offload_diagnostic.py`，通过 `--reference-method lower_mlp_upper_uncovered_obs --compare-method <晋级方法> --mc-itr`，按相同50配对、2回合、8快照协议重放 A 轨迹。额外方法只在冻结快照评估，不回写环境。输出原下层、晋级下层的下层推理时间及字典序目标；MC-ITR 记录含启发式和共享贪心初值的完整调用耗时。不能把下层推理时间标为完整策略延迟，也不能把快照成功数差累加到长期 DSR。

验证覆盖空/单/满容量请求、无效槽、非法动作、前缀因果、采样与重算 log-prob 一致、有限梯度、全空批更新、顺序策略断点续跑一致、检查点跨方法拒绝、历史 runner 回归、晋级阈值与四组 smoke。交付做显式自审，不冒充独立审查。任务验收日志在 `output/batch_offload_20260928/validation/`，实验数据在 `results/batch_offload_20260928/`。未完成 pilot/confirmation 时不得写入论文新性能结论。

## 实施验收（2026-09-29，延续原任务目录）

已实现全部策略与实验入口，20项远程回归通过；四组各2×8步训练及1×8步评估的 smoke 通过。三方法固定快照桥接入口另以1个快照通过验证，不能作为性能证据。本机与远程验证源码摘要为 `5f2ed5f7009672d67528447e2f2bd40a132c5fce176be28553d5e848889e749f`。无提交、推送或 pilot/confirmation 启动。

原整套方法10×5配对的100个历史评估回合重放全部通过，覆盖400快照、6938个快照服务请求。诊断汇总位于 `results/batch_offload_20260928/diagnostic/summary.json`：50万次 UAV 观测中82.07%发生30槽截断；可见未覆盖服务需求出现次数中44.18%未进入该 UAV 的观测槽（出现次数含跨UAV和跨时隙重复，不能解释为44.18%的独立用户或全系统请求不可见）。按源UAV决策前请求数，1–5、6–10、11–20组超时率为38.53%、61.91%、77.69%；21+组只有22个请求，不外推趋势。此为描述性关联。未截断的非协作时延估计并非普遍低估，不能认定乐观负载估计是统一原因。最后20个训练回合相对前20回合的训练DSR平均+0.266个百分点，5/10种子为正，不能据此证明已收敛。

验收文件：`output/batch_offload_20260928/validation/validation.json`。主工作区目录检查因既有 `.zcode`/`.zcodeignore` 失败，覆盖工作树因既有迁移前根目录项失败；本任务没有新增根目录项，没有扩大白名单或删除旧文件。细节保存在同目录 `layout.json`。
