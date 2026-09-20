# 环境修复与远程验收（2026-09-20）

最终状态：部署后的完整验收 **PASS**（49/49 测试，14/14 短训练）。本机与远程 73 个源码文件一致，最终源码摘要为 `214cba350eb3d724d8681faf08fea1dd5e39c231fa3cbe707697ae4e658ce8dd`。验收文件：`output/env_repair_20260920/deployed-validation-20260920.json`。

## 修复范围

- 环境：全机队统一批处理计算负载、时隙通信状态对齐、缓存生命周期、碰撞回退闭包、输入校验。
- 下层策略：联合动作 log-prob 与 entropy 求和；空请求不更新 actor；Joint 保存用于概率计算的原始移动动作，执行时再裁剪。
- 奖励与容量：下层仅统计已接纳服务及卸载决策引起的能耗；空域奖励中性；每架 UAV 可表示全部 100 个 UE 请求，无静默容量截断；质量 mask 与硬约束分离。
- 复现：硬件静态随机源、UE 请求随机源和移动随机源独立；记录训练/场景种子、配置和源码摘要；阻止旧语义检查点混用。
- 入口：`run_canonical_experiment.py` 支持完整或末尾不足一段的 rollout、模型更新、回合边界恢复、独立场景评估及独立输出目录。

## 使用位置与运行环境

本机代码与远程 `/home/PengYanghan/Lunwen/wodexuexi` 同步。远程修改前确认工作树干净，25 个修复文件已备份并同步。备份：`/home/PengYanghan/Lunwen/before-full-repair-20260920.tar.gz`。没有执行提交、推送、历史结果清理或大规模训练。

Python：`/home/PengYanghan/miniconda3/envs/drone/bin/python`。服务器使用 CPU 版 PyTorch 2.6.0，CUDA 不可用；Python 3.12.11、NumPy 2.0.1。`pip check` 无依赖冲突。已保存依赖快照（该快照记录当前安装状态，并非跨平台安装保证）。本机未运行 Linux `.venv` 或 PyTorch 训练。

## 验证证据

- 49 项测试：无失败、错误或跳过。覆盖 0/1/30/50/100 请求、概率一致性、奖励因果性、缓存/碰撞/时序、随机源隔离、保存加载与旧模型拒绝。
- 回合边界断点续跑：与连续训练逐回合结果和 actor 参数完全一致。
- 7 种方法 × 2 个训练种子 × 8 步，共 14 次短训练；均完成采样、rollout、GAE、更新和保存，损失有限。同一场景下外生请求与移动轨迹摘要一致。
- 4 种卸载路径 × 2 个种子 × 100 步，共 800 步仿真；检查观测、奖励、能耗、缓存容量及运动约束。上层 Attention/Vanilla 各完成一次额外优化器更新。
- 确定性测试：两个相同种子的仿真逐项精确一致。
- 旧分层与 Joint 入口各完成 1 回合 × 4 步，输出模型、有限损失和汇总文件。旧入口仍标注追溯不完整，正式实验使用新入口。
- 本机 73 个 Python 文件语法检查及差异空白检查通过。

原始记录位于 `output/env_repair_20260920/`。隔离目录含一个历史诊断脚本，导致其全源码摘要与正式工作目录不同；因此部署后在实际工作目录重新执行完整验收，不将隔离目录摘要冒充最终源码摘要。

## 可运行实验

1. 上层 Attention-MAPPO 与 Vanilla-MAPPO（启发式卸载）的对比。
2. 完整分层方法与端到端 Joint MAPPO 的对比。
3. 无注意力、无质量 mask、无自适应惩罚消融。
4. 多训练种子重复、固定外生场景配对、独立 workload seed 泛化评估。
5. 请求容量、时隙一致性及其他已有回归测试。

这些是已验证的运行能力，短验收不证明算法收敛、性能优势或论文统计结论。尚未运行正式多种子长训练。旧检查点不可直接用于新语义的下层/Joint 训练。当前 CPU 环境可以运行实验，但没有验证 GPU 路径或大规模训练耗时。

## 启动示例（服务器执行）

```bash
ssh 100.69.44.85
cd /home/PengYanghan/Lunwen/wodexuexi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/PengYanghan/miniconda3/envs/drone/bin/python run_canonical_experiment.py \
  --method hierarchical --training-seed 7 --workload-seed 19 \
  --episodes 1 --steps 8 --rollout 4 --ppo-epochs 1 --batch-size 2 \
  --purpose pilot --output results/canonical/my_first_pilot
```

输出目录必须未使用。`--method` 可取 `upper_attention`、`upper_vanilla`、`hierarchical`、`no_attention`、`no_quality_mask`、`no_penalty`、`joint`。恢复时使用原参数、原目录及 `--resume`，增加 `--episodes` 为目标总回合数。评估使用新目录及 `--evaluate-checkpoint` 指向 `checkpoints/episode_XXXXX`，可变更 workload seed。

`--purpose formal` 要求源码已提交且工作树干净，以及匹配源码摘要的 PASS 验收文件。当前修复保留为未提交修改；提交/推送仍需用户单独授权，未绕过该正式运行条件。
