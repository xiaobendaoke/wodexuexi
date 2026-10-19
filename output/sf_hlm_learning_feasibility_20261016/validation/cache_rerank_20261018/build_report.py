import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);a=parser.parse_args();launch=json.loads(a.launch.read_text());root=Path(launch['args']['results_root']);v=Path(launch['args']['validation_root']);s=json.loads((root/'analysis/summary.json').read_text());audit=json.loads((v/'raw_record_audit.json').read_text());assert audit['status']=='PASS'
lines=['# 固定等权匹配池、学习重评分：开发筛选','','本候选保留half代理的全部条件匹配身份分配，让同一full_logit_context冻结概率模型只在候选中选择。不重训模型，不变提案、最多10次匹配或能耗目标。每折价格仅使用256训练状态的公开候选与学习期望分数；16留出状态不标价格。','',
'共同固定延迟0/10/20/30 ms用于筛选，全部方法加相同值；这不是各方法的实测软件费用，不能据此宣称费用反转已解决。若筛选通过仍须原生全费用与闭环。','',
'| 方法 | 0 ms DSR % | 10 ms DSR % | 20 ms DSR % | 30 ms DSR % | 真实机队能耗/G | MBS占已接纳 % | 公平性 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,g in s['groups'].items():lines.append('| '+name+' | '+' | '.join(f"{g['rate'][k]*100:.4f}" for k in ('0','10','20','30'))+f" | {g['energy_ratio_to_G']:.5f} | {100*g['mbs_share']:.4f} | {g['fairness']:.6f} |")
lines+=['','| 共同延迟 ms | 最好无标签对照 | 学习−该对照 pp | 描述性95% t区间 pp | 正向历史块 | 学习−G pp |','|---|---|---:|---|---:|---:|']
for key,x in s['contrasts'].items():
 z=x['vs_best'];lines.append(f"| {key} | {x['best_nonlearning']} | {100*z['mean']:+.5f} | [{100*z['ci95'][0]:+.5f},{100*z['ci95'][1]:+.5f}] | {z['positive']}/5 | {100*x['vs_G']['mean']:+.5f} |")
failures=[]
for key in ('10','20'):
 z=s['contrasts'][key]['vs_best']
 if z['mean']<.001:failures.append(key+' ms相对最好无标签对照不足+0.1 pp')
 if z['positive']<4:failures.append(key+' ms正向块不足4/5')
if s['contrasts']['20']['vs_G']['mean']<.005:failures.append('20 ms相对G不足+0.5 pp')
if s['decision']['energy_ratio_to_G']>1.05:failures.append('机队能耗/G超过1.05')
assert bool(failures)!=s['decision']['common_delay_screen_pass']
lines+=['','筛选判断：'+('通过，仅可晋级原生计时。' if not failures else '未通过，不晋级原生计时、闭环或独立确认。原因：'+'；'.join(failures)+'。'),'',
'五个历史上层块仅作开发描述；400状态、25折、边和多延迟不能充作独立训练重复。均按块先累加成功/生成计数再平均，随机对照在同一状态的5个重复先平均。不得挑只通过的延迟口径代替预定联合门槛。','',
f"原始审计复算{audit['public_price_candidates_recomputed']:,}个训练期候选、全部400个公开选择与共同延迟计数；{audit['physical_states_replayed']}个新分配状态重新物理执行，{audit['physical_states_reused_after_exact_match']}个在同状态/完全同分配核验后复用已审计结果。全部half候选池逐项相同，预算和训练隔离通过。审计为主代理自审，不是独立审查。",'',
f"协议SHA `{audit['protocol_sha256']}`；统计SHA `{audit['summary_sha256']}`。",'',
'本轮失败说明该固定池与概率重评分组合未达到门槛，不证明所有学习方法无效。旧结果、原十块确认和论文正文保持不动，持续目标未完成。','',
'[主统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_rerank_20261018/development_v1/analysis/summary.json) · [原始审计](../validation/cache_rerank_20261018/raw_record_audit.json)']
out=v.parent.parent/'deliverables/cache_rerank.md';out.write_text('\n'.join(lines)+'\n');print(out)
