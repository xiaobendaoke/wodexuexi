# Candidate Figure Catalog

| File | Type | Recommendation | Caption suggestion |
| --- | --- | --- | --- |
| `fig_framework.png` | 端到端系统流程图 | 主文强烈建议保留 | 双层协同优化框架：上层负责 UAV 轨迹与协同控制，下层负责请求级卸载决策。 |
| `fig_pareto_tradeoff_scatter.png` | Pareto-style 权衡图 | 主文建议保留 | 不同策略与场景在 MBS 依赖与服务成功率之间形成权衡边界。 |
| `fig_training_convergence.png` | 训练收敛曲线 | 主文或附录 | 上层控制器训练过程的 reward、latency、energy 和 DSR rolling mean。 |
| `fig_paired_seed_slope.png` | 配对种子改进图 | 主文建议保留 | 同一随机种子下，oracle-guided 策略相对启发式的指标变化。 |
| `fig_uav_hotspot_fallback_heatmap.png` | UAV 轨迹与热点覆盖热图 | 当前为 trace/示意混合，谨慎使用 | 空间 trace 或配置示意展示 UAV 轨迹、UE 热点与 MBS fallback 位置。 |
| `fig_request_destination_stack.png` | 请求去向结构图 | 主文建议保留 | 下层卸载策略改变 local、cooperative UAV 与 MBS 的请求流向占比。 |
| `fig_offload_classifier_quality.png` | 下层分类质量图 | 主文或附录强烈建议 | oracle-guided 卸载器与 oracle 标签的一致性及概率校准表现。 |
| `fig_hyperparameter_sensitivity_template.png` | 超参数敏感性图 | 附录 | 若存在 MBS penalty 扫描结果则展示真实敏感性；否则生成待扫描模板。 |
