"""Render the verified same-pool conditional learning test on the server."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);o=p.parse_args();read=lambda p:json.loads(p.read_text());s=read(o.results/'analysis/summary.json');audit=read(o.validation/'raw_record_audit.json');assert audit['status']=='PASS' and audit['summary_sha256']==hashlib.sha256((o.results/'analysis/summary.json').read_bytes()).hexdigest()
lines=['# 同一cold_half可行候选池：条件学习重排检验','','保留第34节25个模型及18维输入。候选完全来自公开cold_half匹配，学习只评分选择；3种无标签重排也使用相同候选池与固定0软价格。旧14方法保留，总共18方法、14个无标签对照。线上最多10次匹配，本轮复用已封存公开池作共同延迟预筛；原生实现仍须当次重建池、支付全部特征/推理/重排费用。','',
f"同状态候选跨训练折完全相同，重复核对{s['pool_duplicate_checks']:,}次，原cold_half选择400/400复现。主方法相对原cold_half改变{s['learning_choices_differ_from_cold_half']}个选择。{audit['fresh_physical_outcomes_replayed']}个新物理结果分别重放核验，旧14方法全部报告指标复现。",'',
'| 方法 | 0 ms DSR % | 10 ms DSR % | 20 ms DSR % | 30 ms DSR % | 机队能耗/G |','|---|---:|---:|---:|---:|---:|']
for m,g in s['groups'].items():lines.append('| '+m+' | '+' | '.join(f"{100*g['rate'][k]:.5f}" for k in ('0','10','20','30'))+f" | {g['energy_ratio_to_G']:.6f} |")
lines+=['','| 共同延迟 ms | 最好无标签对照 | 学习−该对照 pp | 描述性95% t区间 pp | 正向块 | 学习−G pp |','|---|---|---:|---|---:|---:|']
for k,x in s['contrasts'].items():
 e=x['vs_best'];lines.append(f"| {k} | {x['best_nonlearning']} | {100*e['mean']:+.5f} | [{100*e['ci95'][0]:+.5f},{100*e['ci95'][1]:+.5f}] | {e['positive']}/5 | {100*x['vs_G']['mean']:+.5f} |")
lines+=['',f"预定10/20 ms联合筛选通过：{s['decision']['common_delay_screen_pass']}。相对最强无标签≥+0.1 pp且至少4/5块正、20 ms比G≥+0.5 pp、实际机队能耗/G≤1.05，联合判定，不能挑单项替代。",'',
f"原直接学习自身评分不足的{s['lower_priced_score_states']}个状态中，{s['lower_priced_score_control_action_absent']}个状态的cold_half所选动作未出现在直接学习接受的候选池。该记录描述有限候选差异；不能直接称硬能耗guard已被证明是唯一原因。",'',
'首次汇总在400状态完成后因builtin float与numpy.float32混用而失败；从封存JSON转为Python float重算，没有重新选择动作。首版自审因旧汇总还含calls字段而中止；保留脚本和日志，改为逐项比较双方报告的rate/energy/MBS/fairness四字段后通过。所有物理重放仅用于自审，不改原结果或门槛。','',
'五个历史上层训练块只是开发描述；25折、400状态及分支不能当独立训练重复。共同延迟不是实际软件费用，没有闭环或独立确认结论。原十块确认、论文与历史失败不动，持续目标未完成。','',
f"协议SHA `{audit['protocol_sha256']}`；统计SHA `{audit['summary_sha256']}`。",'',
'[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_guarded_rerank_20261018/development_v1/analysis/summary.json) · [原始自审](../validation/cache_guarded_rerank_20261018/raw_record_audit.json)']
if not s['decision']['common_delay_screen_pass']:lines+=['','本轮筛选失败，按协议不进入原生计时、闭环或确认，不追加重排权重求正。']
out=o.validation.parent.parent/'deliverables/cache_guarded_rerank.md';out.write_text('\n'.join(lines)+'\n');print(out)
