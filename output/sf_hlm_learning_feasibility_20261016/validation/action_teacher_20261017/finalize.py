"""Report finite one-transfer space without presenting it as deployment gain."""
import argparse
import hashlib
import json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

ap=argparse.ArgumentParser()
for key in ('results_root','validation_root','report'):ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
a=ap.parse_args();r=a.results_root;v=a.validation_root;s=read(r/'analysis/summary.json');audit=read(v/'raw_record_audit.json')
assert audit['status']=='PASS' and audit['summary_sha256']==sha(r/'analysis/summary.json')
lines=['**一步配额调整的有限动作空间诊断：存在值得训练廉价排序器的空间**','',
    f"400个复用开发状态、{s['branches']:,}个分支、{s['unique_physical_outcomes']:,}个不同物理分配；离线匹配{s['offline_matching_calls']:,}次。没有训练，也没有测量新排序器完整软件费用。",'',
    '分配先用纯观测信息生成并封存，再退出加速上下文，在原私有快照副本上逐分配评分；模型将来只能读公开输入，不能读这些私有标签。一步分支最多两次匹配，保留原质量mask、预测能耗上限与预测最好保留。','',
    '| 共同费用假设/ms | STOP−G/pp | 随机一步−G/pp | 风险启发式−G/pp | 公开预测枚举择优−G/pp | 知道真实结果的枚举择优−G/pp及95% t区间 |',
    '|---:|---:|---:|---:|---:|---|']
for cost,cs in s['subsets']['all']['comparisons'].items():
    o=cs['oracle-G'];vals=[cs[k]['mean']*100 for k in ('stop-G','random_one-G','heuristic_one-G','predicted_best-G')]
    lines.append(f"| {float(cost):g} | {vals[0]:+.3f} | {vals[1]:+.3f} | {vals[2]:+.3f} | {vals[3]:+.3f} | {o['mean']*100:+.3f} [{o['ci95'][0]*100:+.3f}, {o['ci95'][1]*100:+.3f}] |")
lines += ['',
    '**这些不是实测部署收益。** 四列成本均为共同增加的假设，不含尚未拟合排序器的真实额外费用；枚举全部候选也没有按在线路径计费。经验择优只能说明指定状态与有限动作集合的即时选择空间，不能外推闭环，也不能当成10 ms安全线。','',
    '随机一步是所有合法非STOP动作的精确均匀期望，包含匹配/能耗拒绝后保留旧分配的动作，没有挑最好随机种子。公开预测枚举择优只读预测目标，但需枚举全部动作，列为离线参照。真实结果择优更额外知道物理执行结果，两者都不是训练好的策略。','',
    '**复核。** 未加速原引擎重新生成全部22,193个公开分支，核对观察输入、动作集合、预测目标、配额/分配、能耗与预算；各状态G物理结果复现历史请求和指标。另一实现重算所有逐请求费用假设、控制期望、块级差、区间及精确符号翻转，全部通过。主代理自审，不称独立代理审查。','',
    '统计单位仍是五个历史训练块；状态、动作与费用情景不能增加独立n。区间为探索性配对t区间；例如经验择优五块同向时精确双侧符号翻转仍为0.0625。原始质量/费用假设的正结果不充作独立确认。','',
    '**下一步。** 原协议的空间筛选已通过，进入预先冻结的廉价排序器留出探针。公开特征与私有目标分开，按块/负载同时隔离，再检查原生完整费用、新负载闭环和独立确认。持续目标尚未完成。','',
    '可复核文件：[协议](../../../results/sf_hlm_learning_feasibility_20261016/action_teacher_20261017/oracle_v1/protocol.json)、[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/action_teacher_20261017/oracle_v1/analysis/summary.json)、[逐状态摘要](../../../results/sf_hlm_learning_feasibility_20261016/action_teacher_20261017/oracle_v1/analysis/states.json)、[原引擎复算验收](../validation/action_teacher_20261017/raw_record_audit.json)、[执行计划](../../../docs/plans/sf_hlm_learning_feasibility_20261016.md)。','',
    '布局检查仍只有既存根目录`修改总结报告.md`白名单问题，本轮未修改该文件或白名单；原论文及原确认未改写。']
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')
files=[r/'protocol.json',r/'status.json',r/'analysis/summary.json',r/'analysis/states.json',v/'raw_record_audit.json',a.report]
receipt=dict(status='COMPLETED_AND_VALIDATED',goal_complete=False,decision=s['decision'],states=400,
    branches=s['branches'],unique_physical_outcomes=s['unique_physical_outcomes'],raw_record_audit=audit,
    files={str(p):sha(p) for p in files})
(v/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],space_for_cheap_probe=s['decision']['space_for_cheap_probe'],goal_complete=False)))
