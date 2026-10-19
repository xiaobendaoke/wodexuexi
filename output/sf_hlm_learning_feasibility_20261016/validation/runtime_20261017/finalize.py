"""Remote formatting of a validated runtime result; goal remains active."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def effect(d):return f"{d['mean']*100:+.3f} [{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}]"


ap=argparse.ArgumentParser()
for key in ('results_root','validation_root','report'):ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
a=ap.parse_args();r=a.results_root;v=a.validation_root;s=read(r/'analysis/summary.json');p=read(r/'protocol.json');audit=read(v/'raw_record_audit.json')
assert audit['status']=='PASS' and audit['summary_sha256']==sha(r/'analysis/summary.json')
all_=s['subsets']['all'];groups=all_['groups'];pairs=all_['comparisons'];candidates=p['candidates']
best=max(candidates,key=lambda m:groups[m]['fee_rate']);d=s['decision']['candidates'][best]
headline=('至少一学习候选通过相对加速贪心的开发费用门槛，仍须新负载闭环与独立确认。' if any(x['practical'] for x in s['decision']['candidates'].values()) else
          '公共计算/稀疏推理加速未使四个学习候选达到相对加速贪心的开发费用门槛。')
lead='**'+headline+'持续目标尚未完成。**\n\n'
text=a.report.read_text();assert '**主要配对结果。**' not in text;text=lead+text
text+='\n**主要配对结果。**\n\n| 比较 | 费用后差及95% t区间/pp | 正向块 |\n|---|---|---:|\n'
for first,second in [(m,'fast_G') for m in candidates]+[(m,s['decision']['best_fast_nonlearning']) for m in candidates]+[('fast_R8','fast_G'),('fast_R16','fast_G')]:
    stat=pairs[first+'-'+second]['fee'];text+=f"| {first} − {second} | {effect(stat)} | {stat['positive']}/5 |\n"
text+='\n| 原始方法 | 加速实现 | 完整均值时间变化/ms | 费用后变化及95% t区间/pp |\n|---|---|---|---|\n'
for base in ('G','L16','L8','R16','R8'):
    old=groups['legacy_'+base];new=groups['fast_'+base];stat=pairs['fast_'+base+'-legacy_'+base]['fee']
    text+=f"| {base} | fast | {old['mean_seconds']*1000:.2f} → {new['mean_seconds']*1000:.2f} | {effect(stat)} |\n"
for base in ('L16','L8'):
    old=groups['fast_'+base];new=groups['sparse_'+base];stat=pairs['sparse_'+base+'-fast_'+base]['fee']
    text+=f"| fast_{base} | sparse | {old['mean_seconds']*1000:.2f} → {new['mean_seconds']*1000:.2f} | {effect(stat)} |\n"
text+='\n新旧原始成功率差全部为零；本轮分配不变，仅测得软件实现对事后费用评分的影响。公共加速也给予非学习对照，因此不能把相对旧慢随机的提升归为学习特有贡献。稀疏网络是相同权重方程的另一实现，隐藏浮点值可以不同；实测图逐位一致、logits在冻结容差内、全部动作和最终分配一致。\n'
text+='\n**状态来源与取舍。**\n\n| 来源 | 候选 | 相对fast_G/pp | 相对最佳加速非学习/pp |\n|---|---|---:|---:|\n'
non=s['decision']['best_fast_nonlearning']
for source in ('greedy','learned'):
    ss=s['subsets'][source]
    for m in candidates:
        text+=f"| {source} | {m} | {ss['comparisons'][m+'-fast_G']['fee']['mean']*100:+.3f} | {ss['comparisons'][m+'-'+non]['fee']['mean']*100:+.3f} |\n"
text+='\n| 方法 | MBS/已接纳请求/% | 公平性 | 稀疏网络调用/决策 | 密集回退/决策 |\n|---|---:|---:|---:|---:|\n'
for m in ['fast_G','fast_R8','fast_R16',*candidates]:
    g=groups[m];text+=f"| {m} | {g['mbs_share']*100:.3f} | {g['fairness']:.5f} | {g['sparse_forwards']:.3f} | {g['dense_fallbacks']:.3f} |\n"
text+='\n实际机队能耗与原分配一致，其相对fast_G比值见主表；未估计软件执行的机载耗电，也不把运行加速换算为续航收益。消息内容、物理观测、接纳和原环境不变；全部基线均按各自完整软件时间逐请求计费。\n'
text+='\n**验证与下一步。**\n\n'
text+=f"- 正式400状态、11200分支实例、33600个正式时间完整；{s['graph_checks']}次图张量核验、稀疏logits最大绝对差{s['max_sparse_logit_error']:.3g}，全部分配复现原控制。缓存每调用清空，输入矩阵和特征准备包含在计时内。\n"
text+='- 本机20项轻量测试通过、5项Linux数值测试按边界跳过；服务器6项对应测试全部通过。8状态烟雾仅验证兼容性。原始费用、块级统计和闸门由另一实现复算通过；这是同一主代理自审，不称独立代理审查。\n'
text+='- 所有时间样本和预定方法均保留。五块为统计单位，25/400状态或33600次时间均不能增加独立n。配对t区间是开发探索性区间，另给精确双侧符号翻转；未作新的确认性显著性主张。\n'
text+='- 本轮结果不替换原确认或完整回合DSR。费用仍是事后截止期评分，未回写软件等待；下一步必须根据候选闸门推进新负载闭环或新的成本感知决策实验，不能因本轮失败就把持续目标改成负结果整理。\n'
text+='- 仓库布局检查仍因既存根目录`修改总结报告.md`失败，本轮未修改该文件或白名单。\n'
text+='\n可复核文件：[协议](../../../results/sf_hlm_learning_feasibility_20261016/runtime_20261017/development_v1/protocol.json)、[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/runtime_20261017/development_v1/analysis/summary.json)、[逐状态汇总](../../../results/sf_hlm_learning_feasibility_20261016/runtime_20261017/development_v1/analysis/states.json)、[复算验收](../validation/runtime_20261017/raw_record_audit.json)、[执行计划](../../../docs/plans/sf_hlm_learning_feasibility_20261016.md)。\n'
a.report.write_text(text)
files=[r/'protocol.json',r/'status.json',r/'analysis/summary.json',r/'analysis/states.json',a.report,v/'raw_record_audit.json']
receipt=dict(status='COMPLETED_AND_VALIDATED',stage='runtime development comparison',goal_complete=False,
    states=400,branch_instances=11200,timing_samples=33600,independent_confirmation=False,independent_agent_review=False,
    decision=s['decision'],raw_record_audit=audit,files={str(path):sha(path) for path in files})
(v/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],best_learning=best,practical=d['practical'],goal_complete=False)))
