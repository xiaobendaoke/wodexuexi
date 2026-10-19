"""Render audited conditional-cache task evidence on the experiment server."""
import argparse,json,statistics
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);opt=parser.parse_args();a=json.loads(opt.launch.read_text())['args'];root=Path(a['calibration_root']).parent;v=Path(a['validation_root']);read=lambda p:json.loads(p.read_text());price=read(v/'price_calibration_audit.json');common=read(root/'common_delay_v1/analysis/summary.json');ca=read(v/'common_delay_raw_audit.json');assert price['status']==ca['status']=='PASS'
lines=['# 条件缓存匹配：强规则对照与费用筛选','','本轮14方法中前11个是无标签对照；新学习模型保留原18维特征，只在公开全miss规则未触发的训练状态拟合。原三种概率控制、三种对应cold规则和新学习方法使用同一公开修复128、配额提案、原mask/guard、最多10次匹配及名义10 ms。私有邻居缓存只作离线训练标签和物理评分，不传给选择器。','',
f"25折仅在各自256训练状态标定公开能耗价格，模型自身拟合其中192非guard状态。全部{price['public_candidates_recomputed']:,}个公开候选与最小价格复核通过；原三目标{price['label_free_control_cases_exactly_unchanged']:,}个公开案例和价格逐项复现旧轮。",'',
'## 共同固定延迟筛选','','下表给所有方法共同加固定0/10/20/30 ms，不能解释为各自实测软件费用。400状态/8,800分支的公开决定均先封存，再物理评分；旧方法当场重算动作以核对冻结选择，只有同状态完全同分配的结果才复用。','',
'| 方法 | 0 ms DSR % | 10 ms DSR % | 20 ms DSR % | 30 ms DSR % | 机队能耗/G | MBS占已接纳 % | 公平性 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
g=common['subsets']['all']['groups'];d=common['decision']
for name,x in g.items():lines.append('| '+name+' | '+' | '.join(f"{100*x['rate'][k]:.4f}" for k in ('0','10','20','30'))+f" | {x['energy_ratio_to_G']:.5f} | {100*x['mbs_share']:.4f} | {x['fairness']:.6f} |")
lines+=['','| 共同延迟 ms | 最好无标签对照 | 学习−该对照 pp | 描述性95% t区间 pp | 正向块 | 学习−G pp |','|---|---|---:|---|---:|---:|']
for key,x in common['subsets']['all']['contrasts'].items():
 e=x['vs_best'];lines.append(f"| {key} | {x['best_nonlearning']} | {100*e['mean']:+.5f} | [{100*e['ci95'][0]:+.5f},{100*e['ci95'][1]:+.5f}] | {e['positive']}/5 | {100*x['vs_G']['mean']:+.5f} |")
lines+=['',f"预定共同延迟联合筛选：{d['common_delay_screen_pass']}。要求10及20 ms均比11个无标签最好者至少+0.1 pp、至少4/5块正，20 ms比G至少+0.5 pp，真实机队能耗/G≤1.05。不能挑单个通过口径替代联合门槛。",'',
f"公开guard触发{ca['guarded_states']}状态；7,200个旧公开分支逐项复现，原始审计核验{ca['counts']['physical_requests']:,}条唯一物理请求，{ca['counts']['fresh_physical_replays']}个新分配结果再次物理执行，其余完全同分配结果与已审计参照比较。共同延迟数据没有正式软件计时样本。",'',
f"共同延迟协议SHA `{ca['protocol_sha256']}`；统计SHA `{ca['summary_sha256']}`。",'']
native=root/'development_v1/analysis/summary.json';native_audit=v/'native_raw_audit.json'
if native.exists() and native_audit.exists():
 n=read(native);na=read(native_audit);assert na['status']=='PASS';groups=n['subsets']['all']['groups'];decision=n['decision'];lines+=['## 原生逐次软件费用','','每调用清动态缓存，重做prepare/修复/公开guard/特征/模型/矩阵/匹配/价格/反馈/上层/映射与逻辑JSON；模型/价格预加载。公开guard触发时跳过推理，非学习只支付自己的必要计算。计费逐次加入该次请求的实际物理时延后判定deadline，再平均3次计时与随机重复，不能先平均时间。真实网络、持续队列与软件CPU物理能耗不在该软件费用口径。','',
 '| 方法 | 原始DSR % | 计实测软件费用DSR % | 均值 ms | 逐次p95 ms | 最大 ms | 机队能耗/G | MBS占已接纳 % | 公平性 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,x in groups.items():lines.append(f"| {name} | {100*x['raw_rate']:.4f} | {100*x['fee_rate']:.4f} | {1000*x['pooled_mean_seconds']:.3f} | {1000*x['pooled_p95_seconds']:.3f} | {1000*x['max_seconds']:.3f} | {x['energy_ratio_to_G']:.5f} | {100*x['mbs_share']:.4f} | {x['fairness']:.6f} |")
 lines+=['','| 方法 | 共同接纳率 % | 原始接纳后成功率 % | 计实测费用接纳后成功率 % |','|---|---:|---:|---:|']
 blocks=list(n['subsets']['all']['blocks'].values());admission=statistics.mean(b['admitted']/b['generated'] for b in blocks)
 for name in groups:
  raw_quality=statistics.mean(b['methods'][name]['raw_rate']*b['generated']/b['admitted'] for b in blocks);fee_quality=statistics.mean(b['methods'][name]['fee_rate']*b['generated']/b['admitted'] for b in blocks)
  lines.append(f'| {name} | {100*admission:.4f} | {100*raw_quality:.4f} | {100*fee_quality:.4f} |')
 lines+=['','每块精确满足DSR=接纳率×接纳后成功率；以上是五块均值，不能将两个均值相乘替代主DSR。','']
 for label,key in [('学习−G','gain_vs_G'),('学习−最好无标签'+decision['best_nonlearning'],'learning_increment')]:
  x=decision[key];lines+=['',f"{label}：{100*x['mean']:+.5f} pp，五历史块描述性95% t区间[{100*x['ci95'][0]:+.5f},{100*x['ci95'][1]:+.5f}]，{x['positive']}/5正。"]
 lines+=['',f"原生联合门槛：{decision['native_development_signal']}；实际机队能耗/G={decision['energy_ratio_to_G']:.5f}，各块逐次p95<1秒={decision['all_block_p95_below_second']}。仍按相对G+0.5 pp/至少4正、相对最强11无标签+0.1 pp、能耗≤1.05与时限判定。",'',
 f"8状态同源烟雾和400状态/26,400次正式时间复核通过，完整随机次序/同CPU区间不重叠、模型/价格/来源、原物理请求及实际费用统计一致。原生协议SHA `{na['protocol_sha256']}`；统计SHA `{na['summary_sha256']}`。",'']
else:lines+=['原生全软件费用尚未完成并通过审计；本报告不将共同延迟当实际费用。' if d['common_delay_screen_pass'] else '共同延迟筛选失败，按协议不晋级原生计时、闭环或独立确认。','']
lines+=['五个历史上层训练块仅作开发描述，400状态/25折/分支/时间重复不能充作独立训练重复。固定状态未回写环境，尚不证明长回合收益；规则在未见批次可能误触发。原十块确认、论文正文、历史失败与Git发布未改动。持续目标未完成，最终仍须全新负载闭环及独立确认。','',
'[价格审计](../validation/cache_regime_20261018/price_calibration_audit.json) · [共同延迟统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_regime_20261018/common_delay_v1/analysis/summary.json) · [共同延迟原始审计](../validation/cache_regime_20261018/common_delay_raw_audit.json)']
if native.exists() and native_audit.exists():lines+=['[原生统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_regime_20261018/development_v1/analysis/summary.json) · [原生原始审计](../validation/cache_regime_20261018/native_raw_audit.json)']
out=v.parent.parent/'deliverables/cache_regime_matching.md';out.write_text('\n'.join(lines)+'\n');print(out)
