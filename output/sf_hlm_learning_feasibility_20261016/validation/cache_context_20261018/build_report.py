import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);a=parser.parse_args();m=json.loads(a.manifest.read_text());root=Path(m['out']);v=Path(m['validation']);s=json.loads((root/'analysis.json').read_text());audit=json.loads((v/'raw_record_audit.json').read_text());source=json.loads((root/'source_inventory.json').read_text());assert audit['status']=='PASS'
lines=['# 缓存先验与冷暖状态混合：只读机制诊断','','关键事实：400个开发快照中100个来自step0，1980条可校准合作边的真实邻居命中率为0。其余300状态的11,479条边，按原状态权重的真实命中率为93.6962%。每状态原权重和均为1，因此冷启动占该预测指标25%，虽然只占边数约14.7%。不能把该加权预测分布当成完整1000步回合的时间分布。','',
'| 分组 | 状态 | 合作边 | 原状态权重 | 真命中率 % | full学习Brier | 原先验Brier | half Brier | cold_guard_half Brier |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in ('all','step0','later'):
 x=s[name];g=x['methods'];lines.append(f"| {name} | {x['states']} | {x['rows']} | {x['original_weight']:.1f} | {100*x['hit_rate']:.4f} | {g['learned_full']['brier']:.8f} | {g['prior']['brier']:.8f} | {g['half_proxy']['brier']:.8f} | {g['cold_guard_half']['brier']:.8f} |")
lines+=['','同样的log loss也完整保留，不能把Brier一种损失的排序称为全部预测性能排序：','',
'| 分组 | full学习Log loss | 原先验 | own代理 | half | cold_guard_half |','|---|---:|---:|---:|---:|---:|']
for name in ('all','step0','later'):
 g=s[name]['methods'];lines.append('| '+name+' | '+' | '.join(f"{g[k]['log_loss']:.8f}" for k in ('learned_full','prior','own_proxy','half_proxy','cold_guard_half'))+' |')
lines+=['',
'原full模型的整体Brier优于half，但后续状态Brier反而差于原先验和half。它在冷启动组降低了严重先验误差，不能据整体均值宣布在实际长回合中更准。冷暖混合及采样权重是一个有依据的目标失配解释，尚不是经干预验证的失败原因。', '',
'新固定参照cold_guard_half只读现有公开池：至少一个本地缓存option可见且所有可见本地缓存均miss时，将可校准合作边的half概率置0，其余不变。该规则触发100个状态，当前样本中没有触发后续状态，也没有把实际命中的邻居边置0。其整体Brier=0.05315455，低于full模型0.08490829。规则不是邻居真缓存访问；未见状态仍可能误触发，必须独立实测，不能由本样本的零误触发外推。', '',
'以上规则只完成概率诊断，没有新增匹配或计时，不能称其提高了DSR或解决费用反转。后续训练或任务试验必须保留它及其他同信息强规则对照，不能只与旧half比较。','',
'实际加载源码核验：','']
for name in ('neighbor_cache_prior','uav_initial_state','pending_cache_commit','periodic_cache_update','popularity_initialization'):
 x=source['excerpts'][name];lines.append(f"- {name}：`{x['path']}:{x['first_line']}`。")
lines+=['',
'冻结先验为sigmoid(γ×(容量/平均文件大小−流行度rank))，该函数不读取实际缓存、填充程度或时间。UAV初始缓存为全0；时隙末提交pending缓存，并按实际环境50步周期执行GDSF。服务类别的流行度rank映射在环境初始化时由随机流打乱，因此不能把服务类别编号简单当成跨新工作负载不变的流行度。实际函数原文、行号及源码摘要已保存，未用另一工作树的相似文件代替。','',
'未将UE/服务ID、时间索引或私有邻居缓存加入特征；全局及各source的公开可见缓存数量、比例、缺失标记逐状态保存。全部原预测总体指标复现，通过独立的公开option计数及标量加权损失自审。审计为主代理自审，不是独立审查。','',
f"协议SHA `{audit['protocol_sha256']}`；诊断SHA `{audit['analysis_sha256']}`。",'',
'下一项可检验假说是把可观察的冷态规则与暖态概率校准分开，修正混合目标；它必须另冻协议并与冷态规则参照比较，先验证暖态预测，再验证实际任务/费用和新负载闭环。本报告不把这一假说当成已有效的方法。原十块确认、论文和历史失败不改写，持续目标未完成。','',
'[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_context_20261018/diagnostic_v1/analysis.json) · [实际源码清单](../../../results/sf_hlm_learning_feasibility_20261016/cache_context_20261018/diagnostic_v1/source_inventory.json) · [逐状态公开上下文](../../../results/sf_hlm_learning_feasibility_20261016/cache_context_20261018/diagnostic_v1/states.json) · [原始审计](../validation/cache_context_20261018/raw_record_audit.json)']
out=v.parent.parent/'deliverables/cache_context.md';out.write_text('\n'.join(lines)+'\n');print(out)
