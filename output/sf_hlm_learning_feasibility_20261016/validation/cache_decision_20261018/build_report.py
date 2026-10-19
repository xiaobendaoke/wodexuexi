"""Render audited decision-relevance diagnostics on the remote server."""
import argparse,json,hashlib,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);o=p.parse_args();read=lambda p:json.loads(p.read_text());s=read(o.results/'analysis.json');a=read(o.validation/'raw_record_audit.json');assert a['status']=='PASS' and a['analysis_sha256']==hashlib.sha256((o.results/'analysis.json').read_bytes()).hexdigest()
lines=['# 预测改善为何未转化为任务增量：决策相关诊断','','复用400个冻结开发状态、13,459条合作边及原留出预测；没有训练、匹配、物理环境运行或计时。截止期敏感边定义为在已选配额下“命中可按时、未命中超时”，仅由公开分解、负载、deadline和共同固定延迟判断。私有缓存只作已审计标签。','',
'| 子集 | 边数 | 学习Brier | cold_prior Brier | cold_half Brier |','|---|---:|---:|---:|---:|']
keys=[('全部','all'),('未触发公开冷态规则','non_guarded'),('10 ms敏感并集','sensitive_union_10'),('20 ms敏感并集','sensitive_union_20'),('学习已选且10 ms敏感','selected_sensitive_10_cache_regime_probability_price'),('cold_half已选且10 ms敏感','selected_sensitive_10_cold_half_proxy_price')]
for label,key in keys:
 g=s['groups'][key];lines.append(f"| {label} | {g['rows']} | {g['methods']['learned']['brier']:.6f} | {g['methods']['cold_guard_prior']['brier']:.6f} | {g['methods']['cold_guard_half']['brier']:.6f} |")
lines+=['','关键边和两种方法所选边上，学习预测损失仍较低；不能把任务失败简单归因为平均预测精度不足。子集保留原状态权重，未重新归一成每个子集状态等权；完整五块与log loss保留于JSON。敏感性基于已选配额，属于描述性分析。','',
'| 学习−cold_half分解 | 状态数 | 共同10 ms贡献 pp | 共同20 ms贡献 pp |','|---|---:|---:|---:|']
for key in ('same_assignment','same_quota_different_assignment','different_quota','step0','later'):
 g=s['decision_decomposition']['cold_half_proxy_price'][key];lines.append(f"| {key} | {g['states']} | {g['contribution_pp']['10']:+.6f} | {g['contribution_pp']['20']:+.6f} |")
x=s['cross_scores_by_block']['cold_half_proxy_price'];lower=sum(v['learned']['learned_assignment_lower_priced_score_states'] for v in x.values());expected=statistics.mean(v['learned']['expected_success_delta_10ms_pp'] for v in x.values())
lines+=['',f"三种分配类别精确加回原比较，冷态/后续分类也分别加回。学习评分器在{lower}个状态给cold_half已选分配更高的自身带价评分；按学习评分器，预期成功差的块均值为{expected:+.6f} pp。有限候选和硬能耗筛选是可检验假说，尚无干预证明；不能将交叉评分当可部署的事后择优收益。",'',
'实际加载源码在UAV上下文生成时以`config.FILE_SIZES[req_id]`构造`service_file_size`，缓存替换优先级也除以文件大小；配置由固定随机源生成服务文件大小。旧公开rows字段只有cycles/deadline/options/size/source/ue_id，18维特征中的size是请求数据大小。源码中字段存在不等于现行协议已通信该字段，也不保证它有额外预测价值。未将服务ID或私有邻居缓存加入模型。全部公开miss−hit差>0.5秒，预定较小分箱为空并如实保留。','',
f"原始自审通过{a['sensitivity_scalar_checks']:,}次标量敏感性判断、44组及五块损失、全部交叉评分与来源摘要。此为主代理自审，五个历史上层块只作开发描述；无独立确认、原生费用或闭环结论，持续目标尚未完成。",'',
f"协议SHA `{a['protocol_sha256']}`；统计SHA `{a['analysis_sha256']}`。",'',
'[完整诊断](../../../results/sf_hlm_learning_feasibility_20261016/cache_decision_20261018/diagnostic_v1/analysis.json) · [实际源码清点](../../../results/sf_hlm_learning_feasibility_20261016/cache_decision_20261018/diagnostic_v1/source_inventory.json) · [原始自审](../validation/cache_decision_20261018/raw_record_audit.json)']
out=o.validation.parent.parent/'deliverables/cache_decision.md';out.write_text('\n'.join(lines)+'\n');print(out)
