"""Recompute readiness contrasts from the audited parent block summaries."""
import argparse,json,hashlib,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);o=p.parse_args();read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();protocol=read(o.results/'protocol.json');s=read(o.results/'analysis.json');done=read(o.results/'completion.json')
assert done['analysis_sha256']==sha(o.results/'analysis.json') and done['protocol_sha256']==sha(o.results/'protocol.json')
for path,digest in protocol['input_files'].items():assert sha(path)==digest
source_path=next(Path(x) for x in protocol['input_files'] if x.endswith('/development_v1/analysis/summary.json'));parent=read(source_path);primary=protocol['primary'];assert primary=='mixture_price' and s['posthoc_secondary_control_nomination']
from scipy.stats import t
checks=0
for mode,x in parent['modes'].items():
 for control,wanted in s['modes'][mode]['comparisons'].items():
  for metric in ('dsr','potential_dsr','raw_dsr','shadow_fee_dsr','mbs_share','mean_fairness','mean_seconds','p95_seconds'):
   values=[row[primary][metric]-row[control][metric] for row in x['blocks'].values()];assert len(values)==5;mean=math.fsum(values)/5;sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/4);radius=float(t.ppf(.975,4))*sd/math.sqrt(5);r=wanted[metric]
   assert all(math.isclose(v,u,abs_tol=1e-12) for v,u in zip(values,r['values'])) and math.isclose(mean,r['mean'],abs_tol=1e-12) and math.isclose(sd,r['historical_block_sd'],abs_tol=1e-12) and math.isclose(mean-radius,r['ci95'][0],abs_tol=1e-12) and math.isclose(mean+radius,r['ci95'][1],abs_tol=1e-12) and r['positive']==sum(v>0 for v in values);checks+=1
  ratios=[row[primary]['fleet_energy']/row[control]['fleet_energy'] for row in x['blocks'].values()];assert ratios==wanted['fleet_energy_ratio']['values'] and math.isclose(math.fsum(ratios)/5,wanted['fleet_energy_ratio']['mean'],abs_tol=1e-12)
g=s['modes']['measured']['comparisons']['G'];e=g['dsr'];energy=g['fleet_energy_ratio']['mean'];on_time=all(x[primary]['p95_seconds']<1 for x in parent['modes']['measured']['blocks'].values());ready=e['mean']>=.005 and e['positive']>=4 and energy<=1.05 and on_time;assert ready==s['readiness']['practical_development_signal_vs_G'] and math.isclose(float(t.ppf(.975,9))*e['historical_block_sd']/math.sqrt(10),s['readiness']['projected_10_new_block_ci_halfwidth'],abs_tol=1e-12)
report=dict(status='PASS',block_contrasts_recomputed=checks,source_summary_sha256=sha(source_path),analysis_sha256=sha(o.results/'analysis.json'),protocol_sha256=sha(o.results/'protocol.json'),training_source_inventory_sha256=sha(o.validation/'training_source_inventory.json'),audit_script_sha256=sha(__file__),independent_confirmation=False,audit_type='primary agent self-audit; audited parent blocks, scalar moments and all seven method contrasts')
(o.validation/'raw_record_audit.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# 解析混合规则的闭环正结果：独立确认准备','','这是从已完成开发实验的次要对照中事后提名的框架候选，不是新的确认结果，也不改写学习独立增量的失败。固定候选为mixture_price，未依据本轮比较转选其它规则。原开发包含五历史训练块、三个此前未用于拟合的负载、各1000步；完整逐次软件延迟已在环境指标/奖励/下一观测前注入，真实网络、持续队列和软件CPU物理能耗仍不在该协议中。','',
'| 参照 | 候选−参照：计费闭环DSR pp | 五块描述性95% t区间 pp | 正向块 | 候选/参照能耗 |','|---|---:|---|---:|---:|']
for control,x in s['modes']['measured']['comparisons'].items():
 r=x['dsr'];lines.append(f"| {control} | {100*r['mean']:+.5f} | [{100*r['ci95'][0]:+.5f},{100*r['ci95'][1]:+.5f}] | {r['positive']}/5 | {x['fleet_energy_ratio']['mean']:.5f} |")
lines+=['',f"相对G的开发实用性条件满足：{ready}；能耗/G={energy:.5f}，所有块逐次p95<1秒={on_time}。候选没有高于未加价mixture的均值，不能把正结果归结为软价格必然有益；相对同框架binary的区间跨0，不宣称组件优势已确认，更不宣称胜过全部方法。",'',
f"在保持开发块间标准差不变的假设下，10个新块的候选−G描述区间半宽预计约{100*s['readiness']['projected_10_new_block_ci_halfwidth']:.3f} pp；这不是功效保证，不假定增加episode即可减少独立块方差。",'',
'后续若推进，仍需10个或更多预先固定的新训练块、新负载、冻结样本量与候选、确认前的主比较/多重检验及时间/能耗条件。原十确认块不得用于这次新主张确认。实际归档followup入口含foundation训练函数和checkpoint/source/config合同验证，可作为实现参考；本节没有启动新训练或确认。原论文定位仍待单独批准。','',
'完整学习增量门槛仍保留，本轮框架正结果不算学习路线成功。五历史块只是开发描述，持续目标未完成。','',f"协议SHA `{report['protocol_sha256']}`；统计SHA `{report['analysis_sha256']}`。",'',
'[完整比较](../../../results/sf_hlm_learning_feasibility_20261016/framework_readiness_20261018/diagnostic_v1/analysis.json) · [原始自审](../validation/framework_readiness_20261018/raw_record_audit.json) · [归档训练入口清点](../validation/framework_readiness_20261018/training_source_inventory.json)']
out=o.validation.parent.parent/'deliverables/framework_readiness.md';out.write_text('\n'.join(lines)+'\n');print(json.dumps(report));print(out)
