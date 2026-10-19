"""Write the failed learning-increment decision and retain all controls."""
import argparse
import hashlib
import json
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser()
for k in ('results','validation','report'):ap.add_argument('--'+k,type=Path,required=True)
a=ap.parse_args();r=a.results;s=read(r/'analysis/summary.json');pool=read(r/'analysis/pool_summary.json');audit=read(a.validation/'raw_record_audit.json');extra=read(r/'analysis/audit_supplement.json')
assert audit['status']=='PASS' and audit['summary_sha256']==sha(r/'analysis/summary.json') and audit['supplement_sha256']==sha(r/'analysis/audit_supplement.json')
prep=read(r/'prepared.json')
lines=['**少量匹配后的请求风险校准：有框架收益，尚未建立学习增量**','',
    f"400个复用开发状态，固定池至多G/STOP/8个动作、最多9次条件匹配；去重后{prep['unique_candidate_assignments']:,}个候选分配，四费用条件共{prep['rows']:,}个请求标签行。25个同时留块/留负载折、50个固定模型完成。",'',
    '初始公开候选池在10 ms共同费用假设下：知道实际结果的择优相对G +1.115 pp，费用感知预测规则+0.723 pp。该容量门槛通过后才拟合请求成功概率与预测残差两个HGB。逐请求概率相加决定候选，训练和测试状态、块、负载均按合同隔离。','',
    '| 共同费用假设/ms | 原预测字典序−G/pp | 费用感知预测规则−G/pp | 请求概率模型−G/pp | 请求残差模型−G/pp |',
    '|---:|---:|---:|---:|---:|']
for cost,z in s['subsets']['all'].items():
    vals=[z['comparisons'][n+'-G']['mean']*100 for n in ('predicted_lex','predicted_fee','hgb_probability','hgb_residual')]
    lines.append(f"| {float(cost):g} | "+' | '.join(f'{v:+.3f}' for v in vals)+' |')
lines+=['','主10 ms条件的预定比较：','','| 比较 | 差/pp | 95% t区间/pp | 正向块 |','|---|---:|---|---:|']
for name in ('hgb_probability-G','hgb_residual-G','hgb_probability-predicted_fee','hgb_residual-predicted_fee'):
    z=s['subsets']['all']['10.0']['comparisons'][name]
    lines.append(f"| {name} | {z['mean']*100:+.3f} | [{z['ci95'][0]*100:+.3f}, {z['ci95'][1]*100:+.3f}] | {z['positive']}/5 |")
lines+=['',
    '**闸门决定。** 两个模型相对G均超过+0.5 pp且5/5块正向，但相对最强同池非学习规则未达到预定+0.1 pp：概率模型仅+0.077 pp，残差模型−0.069 pp。两者均不晋级为已通过的学习候选，不调整原门槛。概率模型相对原预测字典序的较大收益，也不能全部归为学习，因为简单费用感知预测规则已经恢复大部分收益。','',
    '**费用边界。** 表中是所有路径共同增加相同费用的假设，未实测新池的9次匹配、特征与推理完整时间。不能把+0.799 pp写成计全实测费用后的部署增益，也不由此推断新负载闭环。原G/强随机仍须按各自最省路径比较。','',
    '**原始执行能耗与MBS。** 下表是主10 ms选择对应的五块即时分配结果；能耗比为每块候选均值/该块G均值，MBS比例以已接纳请求为分母。不是闭环能耗或通信审计。','',
    '| 方法 | 五块能耗比G范围 | 五块MBS已接纳比例范围 |','|---|---|---|']
energy=extra['energy_and_mbs']['all']['10.0']
for name in ('G','predicted_lex','predicted_fee','hgb_probability','hgb_residual'):
    ratios=[z['energy_ratio_to_G'][name] for z in energy.values()];mbs=[z['mbs_admitted_fraction'][name]*100 for z in energy.values()]
    lines.append(f'| {name} | {min(ratios):.4f}–{max(ratios):.4f} | {min(mbs):.2f}%–{max(mbs):.2f}% |')
lines+=['',
    f"**复核。** 重构全部公开提案和去重池、{audit['feature_rows']:,}行特征/标签与权重，重放50个序列化模型的全部留出概率与选择；固定首折两个模型重拟合一致。另一实现复算所有计数、块级区间、符号翻转与闸门，并补存概率误差、即时能耗/MBS。主代理自审，不称独立代理审查。原验证脚本重复解压NPZ导致低效，已保存v1源码/日志/停止收据；v2一次载入数组，验证逻辑和实验不变。",'',
    '统计单位仍为五个历史训练块；请求、费用情景和25个交叉折都不是独立重复。原始五块同向的精确双侧符号翻转p仍为0.0625；t区间仅作开发描述。没有独立确认，持续目标尚未完成。','',
    '后续依据请求概率误差诊断，再决定是否把校准放入条件匹配本身；不会只对本轮模型调参直到过线。任何更改匹配目标的新方法都须另冻协议、支付全部费用，并与同信息非学习对照比较。','',
    '可复核文件：[协议](../../../results/sf_hlm_learning_feasibility_20261016/request_risk_20261017/development_v1/protocol.json)、[池容量](../../../results/sf_hlm_learning_feasibility_20261016/request_risk_20261017/development_v1/analysis/pool_summary.json)、[完整留出统计](../../../results/sf_hlm_learning_feasibility_20261016/request_risk_20261017/development_v1/analysis/summary.json)、[能耗/MBS及概率误差](../../../results/sf_hlm_learning_feasibility_20261016/request_risk_20261017/development_v1/analysis/audit_supplement.json)、[复算验收](../validation/request_risk_20261017/raw_record_audit.json)。','',
    '根目录`修改总结报告.md`既存白名单问题保留；原十块确认、论文及Git发布均未更改。']
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')
files=[r/'protocol.json',r/'analysis/summary.json',r/'analysis/pool_summary.json',r/'analysis/audit_supplement.json',a.validation/'raw_record_audit.json',a.report]
out=dict(status='COMPLETED_AND_VALIDATED',states=400,models=50,goal_complete=False,decision=s['decision'],files={str(p):sha(p) for p in files})
(a.validation/'completion.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],goal_complete=False)))
