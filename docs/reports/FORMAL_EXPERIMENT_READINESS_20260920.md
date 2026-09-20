# 正式实验运行准备（2026-09-20）

> 后续复核更正：本报告仅覆盖下列 MARL 矩阵，不表示第四章 MC-ITR / Exact-Small 已完成。旧 v2 发布版存在导入顺序相关的 C++ 动态库加载隐患。新的验证与准备操作应统一使用 `bash run_remote_python.sh ...`；第四章目标确认及补全状态见 `COMPLETE_REPAIR_PLAN_20260920.md`。以下验收数据是当时版本的真实记录，不表示后续源码自动通过相同凭证。

## 本次目标

让冻结协议中的主对比、消融、非学习基线、敏感性分析及统计汇总能够完整运行。此前仅单模型短验证通过，不能代表旧全量脚本已经可用。新流程替换缺失的旧调度器，不加载历史强制接入检查点。

## 正式矩阵

训练场景种子单独固定为 2026，与全部评估场景种子不重合，避免在训练用过的前几个场景回合上评估。固定策略训练种子 42、84、126、168、210；评估工作负载种子 42、84、126、168、210、252、294、336、378、420。学习方法每种子训练 200 回合，每回合 1000 步；每个检查点在每个评估场景运行 6 回合。PPO rollout=1000、epochs=8、batch_size=256。

主矩阵包含 10 个学习配置：完整分层、Vanilla 上层+注意力下层、IPPO 上层+注意力下层、Joint、仅 Attention 上层、仅 Vanilla 上层、仅去除下层注意力、两层去注意力、去质量 mask、去自适应惩罚。5 个非学习配置为 Greedy、Local、MBS、Random、Uniform，仅重评，不重复伪造训练种子。

敏感性沿用仓库已有网格：UE 数量 60/80/100/120/140；算力倍率 0.6/0.8/1.0/1.2/1.4。每个设置对完整分层、Vanilla 分层和 Greedy 分别运行。学习方法按设置重训，避免容量/观测维度变化时复用不兼容模型；这衡量各设置下重新训练后的性能，不等同于固定模型的迁移实验。

全量共 150 次训练、1650 个评估任务。包含全部训练原始曲线，不会将仅有最终检查点的测试结果称为收敛曲线。相同基准设置在敏感性网格中保留独立输出，便于完整追溯。

## 正式数据规则

正式入口要求对应的源码快照、运行时版本和 PASS 验收文件相符。快照复制当前源码，记录逐文件摘要和原 Git 状态，验收后把源码设为只读；不需要擅自提交或推送用户工作区。正式运行没有关闭验证门槛。

每个训练任务保存模型和优化器、随机状态、惩罚系数、配置及源码摘要。最新回合可恢复，每 20 回合保留中间检查点，删除的仅是该任务上一个非保留检查点。评估没有更新模型。任务进度和完成文件带摘要，再次运行同一命令跳过已验收任务；改变配置或篡改结果会拒绝继续。未完成评估从新 attempt 重跑，未完成训练从最近完整回合恢复。

每个工作负载内先合并评估回合，DSR 使用成功/生成请求总数，能效使用成功请求总数/机群能耗总数。主效应为同一训练种子、同一场景的配对差值，95% CI 对训练种子与工作负载两维重采样；评估回合不当成独立训练重复。非学习基线不伪造 5 份独立模型。原始单位数据、效应量、置信区间及训练曲线导出为 JSON/CSV。

## 使用方式

服务器 Python：`/home/PengYanghan/miniconda3/envs/drone/bin/python`。

准备新的源码发布目录：

```bash
cd /home/PengYanghan/Lunwen/wodexuexi
bash run_remote_python.sh prepare_formal_release.py \
  --output /home/PengYanghan/Lunwen/canonical-releases/一个新的版本目录
```

该命令会真正执行验收，失败不生成正式运行凭证。已准备版本及实测结果将在本报告末尾记录。

从通过验收的发布目录运行全部正式实验：

```bash
cd /home/PengYanghan/Lunwen/canonical-releases/20260920-formal-v2
nohup bash run_linear_v2_full.sh /home/PengYanghan/Lunwen/results/canonical_all_20260920 \
  --workers 4 --threads 4 > /home/PengYanghan/Lunwen/canonical_all_20260920.log 2>&1 < /dev/null &
```

中断后重跑完全相同命令即可恢复。`--suite main` 或 `--suite sensitivity` 可在 `run_formal_suite.py` 上选择子集。`--smoke` 仅用于小规模流程验收，输出明确标注 verification，不可混入论文正式数据。`--plan-only --output plan.json` 只导出完整矩阵。

旧 DSR-strong 脚本缺少原配置生成器，且不属于冻结矩阵；现在会明确报错，不猜测或替换它的科学配置。旧强制接入敏感性 CLI 已转向 canonical 敏感性入口。历史数据和历史模型仍保留，不作为新协议正式结果。

## 资源与边界

检查时服务器可用内存约 815 GiB、目标磁盘可用约 26 TiB、逻辑 CPU 256。当前 Python 使用 CPU 版 PyTorch。默认调度 1 个任务，可指定 4；线程和 worker 有界，启动时检查 CPU、可用内存和磁盘余量。全量运行耗时不能由几步 smoke 推定，因此额外执行完整 1000 步、8 个 PPO epoch 的容量验收；实际耗时会单独记录。

本任务交付可运行的正式实验流程，不自动启动上述 150 次完整训练。测试数据均与正式输出目录隔离。

## 最终验收与可运行状态

- **READY**：远程发布版 `/home/PengYanghan/Lunwen/canonical-releases/20260920-formal-v2`。
- 最终源码摘要：`6e82e76eab8d708dc6ab32406ffa3bb2e7add1ba2b0f2b1da6a0fc543fb55d8a`。87 个被追溯文件在本机、发布版和服务器工作目录中一致。
- 最终发布版 55 项测试全部通过，无错误、失败或跳过；30 个单方法短验证通过。
- 最终版全分支缩小矩阵：45 个方法/设置组，30 个训练任务 + 45 个评估任务，共 75 个任务通过；生成 45 个评估单位及 238 组指标对比。因 smoke 只有一个训练种子，明确不给出伪造的多种子置信区间。
- 实际正式门槛验收：最终发布版 Joint 使用 `--purpose formal`、只读源码凭证，完成 1000 步、8 个 PPO epoch、batch size 256，并保存检查点。该单回合是门槛验收，不是完整 5×10 正式统计结果。
- 大容量验收：140 UE 的分层方法完成 1000 步、8 个 PPO epoch、batch size 256，耗时约 100 秒，损失有限。该测试使用 v1 的相同模型实现；v2 随后仅修改协议默认训练场景种子及其测试，未更改模型/环境实现。
- 原工作目录 `/home/PengYanghan/Lunwen/wodexuexi` 已同步源码与凭证，正式门槛检查通过。修改前备份位于 `/home/PengYanghan/Lunwen/before-formal-readiness-20260920.tar.gz`。
- 完整计划已生成，包含 165 个方法/训练种子组，150 次训练和 1650 个评估任务。本任务未启动整个长训练矩阵。
- 原始验收记录：`output/formal_readiness_20260920/`；正式矩阵：`output/formal_readiness_20260920/formal-plan.json`。

### 原目录直接启动

已完成准备，无需再次生成发布版或提交 Git，也可直接使用原工作目录：

```bash
ssh 100.69.44.85
cd /home/PengYanghan/Lunwen/wodexuexi
nohup bash run_linear_v2_full.sh /home/PengYanghan/Lunwen/results/canonical_all_20260920 \
  --workers 4 --threads 4 > /home/PengYanghan/Lunwen/canonical_all_20260920.log 2>&1 < /dev/null &
```

如需边开发边训练，请使用上文只读发布目录的启动方式；修改原目录源码后原凭证会失效，需要重新 prepare。日志可查看上述 `.log`，每个任务的 `state.json` 和 `worker.log` 保存运行状态与错误。全部完成后输出根目录生成 `status.json`、`analysis.json`、`evaluation_units.csv` 和 `training_curves.csv`。
