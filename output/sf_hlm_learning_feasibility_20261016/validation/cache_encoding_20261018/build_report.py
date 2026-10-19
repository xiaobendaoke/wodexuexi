"""Render audited remote evidence without changing results or decision rules."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);a=parser.parse_args()
m=json.loads(a.manifest.read_text());v=Path(m['args']['validation_root']);root=Path(m['args']['results_root']).parent
read=lambda p:json.loads(p.read_text())
s=read(root/'training_v1/analysis/summary.json');audit=read(v/'raw_record_audit.json');assert audit['status']=='PASS'
lines=['# 缓存概率编码修订：开发证据', '',
'本轮整体修订18维输入中的两列：保留完整先验logit，并与可见本地缓存值交互；其余16列、标签、权重、25折划分、C=1逻辑回归和三个无标签控制不变。不将收益单独归于取消截断。', '',
'400个历史开发状态、13,459条可校准合作边；留上层块且留负载，每折256训练、16测试、128排除。五个历史上层训练块是描述汇总层级；25折共享训练数据，边/状态/时间重复不作为独立训练重复。', '',
'| 概率预测器 | 状态/块加权Brier | Log loss |','|---|---:|---:|']
for name,g in s['groups'].items():lines.append(f"| {name} | {g['brier']:.8f} | {g['log_loss']:.8f} |")
c=s['encoding_vs_clipped'];lines+=['',f"相对旧clipped学习模型的Brier减小{c['mean_improvement']:.8f}，五块描述性95% t区间[{c['ci95'][0]:.8f},{c['ci95'][1]:.8f}]，{c['positive']}/5块改善。预测筛选通过。这不是DSR百分点，也不是任务增量或闭环证据。", '',
'原始审计已重建全部公开特征、验证只变两列及全部静态编码元数据，重做25个固定配方以复现预测，并复算按块指标；标签沿用已审计且逐元素相同的旧数组。旧公开逆分解540条精度失败保留，直接标签物理核验不向策略暴露私有分量。', '',
f"训练协议SHA `{s['protocol_sha256']}`；主统计SHA `{audit['summary_sha256']}`。审计为主代理自审，不是独立审查。", '']
native=root/'development_v1/analysis/summary.json'
if native.exists() and (v/'native_raw_audit.json').exists():
 n=read(native);na=read(v/'native_raw_audit.json');assert na['status']=='PASS';g=n['subsets']['all']['groups'];d=n['decision']
 lines+=['## 同状态原生软件费用筛选','',
 '全部11方法在本轮共同恢复状态、清缓存、准备输入、修复、特征/模型/匹配/价格、反馈、上层与逻辑序列化；模型/价格预加载。每次实测时间直接用于该次逐请求截止期判定后汇总，不能用平均时间代替。真实网络、排队与软件CPU物理能耗不在本软件费用口径。', '',
 '| 方法 | 原始DSR % | 计软件费用DSR % | 均值 ms | 逐次p95 ms | 最大 ms | 机队能耗/G | MBS占已接纳 % | 公平性 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,z in g.items():lines.append(f"| {name} | {100*z['raw_rate']:.4f} | {100*z['fee_rate']:.4f} | {1000*z['pooled_mean_seconds']:.3f} | {1000*z['pooled_p95_seconds']:.3f} | {1000*z['max_seconds']:.3f} | {z['energy_ratio_to_G']:.5f} | {100*z['mbs_share']:.4f} | {z['fairness']:.6f} |")
 lines+=['','| 方法 | 共同接纳率 % | 原始接纳后成功率 % | 计软件费用接纳后成功率 % |','|---|---:|---:|---:|']
 import statistics
 for name in g:
  bb=list(n['subsets']['all']['blocks'].values());admission=statistics.mean(b['admitted']/b['generated'] for b in bb);raw=statistics.mean(b['methods'][name]['raw_rate']*b['generated']/b['admitted'] for b in bb);fee=statistics.mean(b['methods'][name]['fee_rate']*b['generated']/b['admitted'] for b in bb)
  lines.append(f'| {name} | {100*admission:.4f} | {100*raw:.4f} | {100*fee:.4f} |')
 lines+=['','上述各块精确满足DSR=接纳率×接纳后成功率；表中五块均值的乘积不替代主指标。', '']
 for title,key in [('学习−G','gain_vs_G'),('学习−最好非学习'+d['best_nonlearning'],'learning_increment')]:
  z=d[key];lines.append(f"- {title}：{100*z['mean']:+.4f} pp，描述性95% t区间[{100*z['ci95'][0]:+.4f},{100*z['ci95'][1]:+.4f}]，{z['positive']}/5块正向。")
 failures=[]
 if d['gain_vs_G']['mean']<.005:failures.append('相对G增益不足0.5 pp')
 if d['gain_vs_G']['positive']<4:failures.append('相对G正向块不足4/5')
 if d['learning_increment']['mean']<.001:failures.append('相对最强无标签控制增量不足0.1 pp')
 if d['energy_ratio_to_G']>1.05:failures.append('实际机队能耗/G超过1.05')
 if not d['all_block_p95_below_second']:failures.append('有块逐次p95达到1秒')
 assert bool(failures)!=d['native_development_signal']
 lines+=[f"- 实际机队能耗/G={d['energy_ratio_to_G']:.5f}；各块逐次p95<1秒：{d['all_block_p95_below_second']}。",f"- 冻结的联合晋级判断：{d['native_development_signal']}。保持相对G+0.5 pp、至少4正块、相对8无标签最好者+0.1 pp、能耗≤1.05及p95<1秒门槛；不因结果改阈值。", '',
 '本轮唯一主方法修订为整体编码与对应模型/训练期价格；三个无标签概率控制全部公开候选/最小价格与旧轮逐项相同，主软件时间在本轮重新实测。第27节clipped结果保留为历史参照，不把跨轮时钟差解释成编码因果收益。', '',
 f"原生协议SHA `{n['protocol_sha256']}`；主统计SHA `{na['summary_sha256']}`。8状态烟雾、全部400状态/7,600分支/22,800时间及物理请求/旧分支/计数/同CPU时钟审计通过。固定状态反事实不回写环境，尚不能外推长期收益。", '']
 lines+=['未晋级原因：'+'；'.join(failures)+'。本候选不进入新负载闭环或独立确认。' if failures else '仅晋级前瞻闭环设计，尚未获得闭环或独立确认结论。','']
 raw=n['subsets']['all']['comparisons'][d['best_nonlearning']]['raw'];lines+=[f"与同一最强无标签对照的原始成功率差为{100*raw['mean']:+.4f} pp（描述性95% t区间[{100*raw['ci95'][0]:+.4f},{100*raw['ci95'][1]:+.4f}]）。它和逐次计费差分别保留，不混称同一指标。", '',
 '| 原状态来源 | 学习−G计费差 pp | 本来源最好无标签控制 | 学习−该控制计费差 pp | 机队能耗/G |','|---|---:|---|---:|---:|']
 for source in ('greedy','learned'):
  z=n['subsets'][source];best=max(('G','R8','R16','binary_price','mixture','mixture_price','own_proxy_price','half_proxy_price'),key=lambda method:z['groups'][method]['fee_rate'])
  lines.append(f"| {source} | {100*z['comparisons']['G']['fee']['mean']:+.4f} | {best} | {100*z['comparisons'][best]['fee']['mean']:+.4f} | {z['groups']['cache_probability_price']['energy_ratio_to_G']:.5f} |")
 lines+=['','来源分组仅作开发描述，不替代预定的共同400状态主判断。','']
 diagnostic=root/'development_v1/analysis/decision_diagnostic.json'
 if diagnostic.exists():
  z=read(diagnostic);assert z['source_summary_sha256']==na['summary_sha256'] and not z['models_retrained']
  lines+=['## 失败后的只读诊断','','诊断未改分配、时间、模型或晋级判断。','',
   '| 决策类别 | 状态数 | 原始差贡献 pp | 计费差贡献 pp |','|---|---:|---:|---:|']
  for name,x in z['decision_categories'].items():lines.append(f"| {name} | {x['states']} | {x['raw_contribution_pp']:+.5f} | {x['fee_contribution_pp']:+.5f} |")
  lines+=['','类别贡献先在各块以共同生成计数归一再平均，精确加回本轮主差。','',
   '| 所选可校准合作边来自 | 边数 | 覆盖状态 | 学习Brier | half Brier |','|---|---:|---:|---:|---:|']
  for name,x in z['selected_cooperative_edges'].items():lines.append(f"| {name} | {x['rows']} | {x['states']} | {x['learned_brier']:.6f} | {x['half_brier']:.6f} |")
  lines+=['','选中边保留原状态权重；选择仅定义描述子群，不回流训练。','',
   '| 交叉评分器 | 预测学习−half成功差 pp | 真物理同加10 ms差 pp | 学习分配的该价格目标更低的状态数 |','|---|---:|---:|---:|']
  for name in ('cache_probability_price','half_proxy_price'):
   bb=[x[name] for x in z['cross_scores_by_block'].values()];expected=statistics.mean(x['expected_success_delta_pp'] for x in bb);nominal=statistics.mean(x['actual_nominal_delta_pp'] for x in bb);lower=sum(x['primary_has_lower_priced_score_states'] for x in bb)
   lines.append(f'| {name} | {expected:+.5f} | {nominal:+.5f} | {lower} |')
  lines+=['','交叉评分只使用公开矩阵和已封存分配，不额外匹配；上述统一名义10 ms与真实软件费用主口径分开。预测误差、能耗价格及候选池都可能影响分配，不能只凭Brier判定任务效果。','']
  if z['reference_changes_by_block']:
   rr=list(z['reference_changes_by_block'].values());lines+=[f"相对第27节clipped系统：原始成功率差{statistics.mean(x['raw_delta_pp'] for x in rr):+.5f} pp，{sum(x['assignment_changed_states'] for x in rr)}/400状态改变分配，{sum(x['quota_changed_states'] for x in rr)}/400改变配额。这里比较整体编码及对应训练价格，跨轮计时不作为同一次配对时钟。",'']
else:lines+=['条件匹配的训练期价格/原生计时尚待完成与审计；当前仅可报告预测筛选。','']
lines+=['若任务门槛通过，仍须全新负载闭环及独立留出确认；原十块确认、本轮以前的失败、论文正文和Git发布未改动。持续目标未完成。', '',
'[训练统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_encoding_20261018/training_v1/analysis/summary.json) · [训练审计](../validation/cache_encoding_20261018/raw_record_audit.json)']
if native.exists() and (v/'native_raw_audit.json').exists():lines.append('[任务统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_encoding_20261018/development_v1/analysis/summary.json) · [原生审计](../validation/cache_encoding_20261018/native_raw_audit.json) · [价格审计](../validation/cache_encoding_20261018/price_calibration_audit.json)')
dest=v.parent.parent/'deliverables/cache_encoding.md';dest.write_text('\n'.join(lines)+'\n');print(dest)
