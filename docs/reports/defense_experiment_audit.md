# 答辩前实验与评价标准审计备忘

## 评价指标口径

- DSR 有两种口径：主配置表使用每步 DSR 的 episode 平均，当前正式矩阵结果和新增审计输出使用 request-weighted DSR。
- request-weighted DSR = 满足 deadline 的服务请求总数 / 生成服务请求总数，更适合作为请求级服务质量结论。
- fairness 基于 UE 的历史 service_coverage 计算 Jain 指数，衡量服务机会分布均衡性，不等同于 deadline 满足率。
- offloading ratio 以已处理服务请求为分母；MBS load ratio 以全部生成服务请求为分母，两者不能混用。
- EEE = deadline-satisfied service requests / UAV total energy，用于表达单位能耗下有效完成服务请求的能力。

## 图 5-6 与 GIF 的答辩解释

- 图 5-6 和 `fig_system_animation.gif` 均使用固定训练随机种子 42、工作负载随机种子 42 的单个 episode 轨迹。
- 该图用于展示空间移动模式和移动代价，不用于证明总体性能最优。
- 空心圆表示轨迹记录中的首个位置，实心方块表示最后记录位置；轨迹记录发生在 `env.step()` 之后，不是 reset 初始位置。
- 蓝色 UAV 长时间贴近 `y=50` 或 `x=650`，是边界缓冲裁剪下形成的策略行为，不是绘图错位。
- 对该固定 proposed episode 的审计结果：累计飞行距离约 39.7 km；服务请求处理率约 55.8%；DSR 约 0.221。

## 约束解释

- 区域边界使用 `[R_c/2, W-R_c/2]`，在当前配置下即 `[50,650]`。
- DSR 和 MBS load 是 Lagrange 软约束目标，用于引导训练，不是逐时隙硬可行性保证。
- 速度与安全间距通过动作幅值限制、碰撞检测和位置修正近似执行；固定轨迹中存在少量修正后步长或间距偏离，因此正文不应声称所有运动约束逐步严格满足。
- 对外表述建议：本文方法通过软约束惩罚降低约束违反倾向，并在多目标之间形成折中。

## 推荐答辩回答

**问：为什么蓝色 UAV 一直贴着下面/右边？**

这是固定 seed 下策略学到的边界行为。环境为了避免覆盖区域过度越界，将 UAV 水平位置裁剪在 `[50,650]` 范围内，因此贴近 `y=50` 或 `x=650` 仍处于合法边界缓冲区内。图 5-6 只用于展示该 episode 的空间移动模式和移动代价，不作为所有服务指标最优的证明。

**问：是不是所有服务都完成了？**

不是。本文的服务质量用 DSR、processed request ratio 和 deadline_satisfied_per_processed 等指标量化。以图 5-6 对应的固定 proposed episode 为例，生成服务请求约 39976 个，处理约 22294 个，处理率约 55.8%，满足 deadline 的比例约 0.221。因此本文强调的是多目标折中，而不是所有服务全部完成。

**问：MBS 负载约束是否严格满足 0.03？**

不是逐时隙硬约束。`0.03` 是训练中的软约束目标，通过 Lagrange 惩罚抑制 MBS 依赖。部分实验结果会高于该目标，因此论文中应表述为降低或约束 MBS 依赖倾向，而不是严格满足硬上限。

**问：DSR 的分母是什么？**

请求加权 DSR 的分母是全部生成服务请求，分子是其中被分配执行且满足自身 deadline 的请求。主配置表还报告了每步 DSR 的 episode 平均，因此论文中需要明确两种口径；最终服务质量解释优先采用当前正式矩阵结果和新增审计输出中的 request-weighted DSR。

**问：为什么 proposed 不是所有指标最好？**

本文方法定位为多目标折中。它的优势集中在公平性、离线稳定性、能耗/有效能效以及相对受控的 MBS 依赖；在 DSR、时延或单项 reward 上不必然支配所有基线。论文结论应避免“所有指标最优”，改为“在能耗、服务质量和公平性之间形成更稳健的折中”。

## 非覆盖审计试跑

为避免覆盖既有正式结果，审计试跑统一写入按日期命名的新目录，不使用 v3/v6 等容易混淆的版本号。

Smoke run:

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python run_baseline_matrix_v2.py \
  --output_dir results/baseline_matrix_audit_20260624_smoke \
  --methods proposed random uniform \
  --training_seeds 42 \
  --workload_seeds 42 \
  --train_episodes 5 \
  --eval_episodes 2
```

Smoke audit:

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python scripts/audit_baseline_matrix_results.py \
  --input_dir results/baseline_matrix_audit_20260624_smoke \
  --output docs/defense_experiment_audit_smoke.md
```

正式非覆盖审计矩阵:

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python run_baseline_matrix_v2.py \
  --output_dir results/baseline_matrix_audit_20260624 \
  --methods ippo vanilla_mappo joint_mappo proposed random uniform \
  --training_seeds 42 84 126 \
  --workload_seeds 42 84 126 168 210 252 294 336 378 420 \
  --train_episodes 200 \
  --eval_episodes 6
```

审计脚本会显式检查 generated、processed、deadline-satisfied 三段服务链，以及 MBS load 和 MBS offloading ratio 的分母是否混用。矩阵结果默认不保存 UAV 位置，因此若输入目录没有 trajectory JSON，轨迹异常核查会标记为不可用，而不是伪造轨迹结论。
