# Force-Admission 正式替换运行说明

本文最终实验口径采用 `FORCE_SERVICE_ADMISSION=True`：当 service request 不在任一 UAV 自然覆盖范围内时，系统将其接入最近 UAV，并继续通过本地、协作 UAV 或 MBS 链路尝试处理。该机制保证请求被接入并尝试处理，不保证满足 deadline。

## 已可直接使用的正式结果

- 扩展基线矩阵：`results/baseline_matrix_force_admission_20260626_resume`
- 正式统计文件：`statistics_workload_balanced.json`
- 审计文件：`docs/defense_experiment_audit_force_admission_full.md`
- 关键口径：所有方法 `processed_request_ratio = 1.0`；Proposed 的 request-weighted DSR 为 `0.3762`，energy 为 `49.98M`，fairness 为 `0.9452`，MBS load ratio 为 `0.0359`。

## 立即重生成可替换图表

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python scripts/plot_baseline_matrix_v2_figures.py --skip_sensitivity
/home/PengYanghan/miniconda3/envs/drone/bin/python docs/figures/scripts/generate_force_admission_trajectory_static.py
```

第一条命令用 force-admission 矩阵重画图 5-14 和图 5-15。第二条命令用 force-admission 轨迹重画图 5-6 的四个静态子图，并同步保存到 `docs/figures/` 和 `latex/docs/figures/`。

## 需要补跑后才能作为正式 force-admission 结论的实验

### 主配置与分层贡献

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode unco_heuristic --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode att_heuristic --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode unco_lower --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode att_lower --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode full_hmarl --num_episodes 200 --seed 42 --force_service_admission
```

正式论文需要对 seeds `42 84 126` 全部执行，并按 10 个 workload seeds、每个 6 个 episode 评估汇总。上面是单 seed 命令模板。

### 下层消融

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode att_lower --lower_ablation full --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode att_lower --lower_ablation no_mask --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode att_lower --lower_ablation no_lagrange --num_episodes 200 --seed 42 --force_service_admission
/home/PengYanghan/miniconda3/envs/drone/bin/python run_hierarchical_mappo_experiment.py --mode att_lower --lower_ablation no_attention --num_episodes 200 --seed 42 --force_service_admission
```

### 敏感性实验

```bash
/home/PengYanghan/miniconda3/envs/drone/bin/python run_sensitivity_full.py --experiment all --force_service_admission
```

输出目录应统一改为 `results/sensitivity_force_admission/...` 后再作为正文图 5-16/5-17 的来源；在该结果完成前，正文中的敏感性图仍应标注为旧口径辅助实验，或暂不作为 force-admission 主结论。

## 审计要求

每个正式结果目录至少检查：

- `processed_request_ratio = 1.0`
- `forced_service_admissions + naturally_covered_service_requests = service_requests_generated`
- `dsr_request_weighted <= processed_request_ratio`
- MBS offloading ratio 以 processed requests 为分母，MBS load ratio 以 generated requests 为分母

答辩标准表述：force-admission 保证服务请求被系统接入并尝试处理；DSR 表示其中满足 deadline 的比例，因此 DSR 不应被解释为服务覆盖率。
