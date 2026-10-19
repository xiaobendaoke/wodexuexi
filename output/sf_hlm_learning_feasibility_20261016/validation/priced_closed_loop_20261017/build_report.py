from pathlib import Path
import json,statistics,math,hashlib,itertools
from scipy.stats import t
a={"release_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/priced_closed_loop_a3e5f7128a06","feasibility_release":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3","parent_launch":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3/parent_launch.json","old_training":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/calibrated_matching_20261017/training_v1","old_price_calibration":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/calibration_v1","native_results":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_runtime_20261017/logging_development_v2","native_validation":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_runtime_20261017","fit_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/final_fit_v1","smoke_results":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/smoke_v1","results_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/development_v1","validation_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_closed_loop_20261017","resource_module":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001/utils/sf_hlm_resources.py","resources":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/train_logs/resources"}
root=Path(a['results_root']);v=Path(a['validation_root']);s=json.loads((root/'analysis/summary.json').read_text());audit=json.loads((v/'raw_record_audit.json').read_text());assert audit['status']=='PASS'
def effect(values):
 mean=statistics.mean(values);radius=float(t.ppf(.975,4))*statistics.stdev(values)/math.sqrt(5)
 return dict(mean=mean,ci95=[mean-radius,mean+radius],values=values,positive=sum(x>0 for x in values),n=5)
secondary={}
for mode,x in s['modes'].items():
 secondary[mode]={}
 for name in ['mixture','mixture_price','binary_price','R8','R16','L16']:
  bs=x['blocks'];d=effect([b[name]['dsr']-b['G']['dsr'] for b in bs.values()])
  secondary[mode][name]=dict(vs_G=d,energy_ratio_to_G=statistics.mean(b[name]['fleet_energy']/b['G']['fleet_energy'] for b in bs.values()))
paired=[]
for block in range(620001,620006):
 for load in [2026101701,2026101702,2026101703]:
  for method in s['modes']['zero']['groups']:
   rr=[json.loads((root/'tasks'/f'{method}_{block}_{load}_{mode}'/'result.json').read_text()) for mode in ['zero','measured']]
   paired.append(dict(method=method,block=block,workload=load,actions_equal=rr[0]['action_hash']==rr[1]['action_hash']))
extra=dict(secondary_development_only=True,not_a_changed_primary_endpoint=True,comparisons=secondary,mode_action_pairs=paired,all_mode_action_pairs_equal=all(x['actions_equal'] for x in paired))
(root/'analysis/secondary_diagnostics.json').write_text(json.dumps(extra,indent=2)+'\n')
lines=['# 实测软件延迟反馈的闭环开发：完整结果','','本轮240个完整回合、240,000步已完成并通过原始记录自审；学习候选未过预定门槛，不晋级独立确认。固定状态诊断中的正结果没有在新负载完整回合中复现。','','5个历史上层开发块×3个新负载×8方法×zero/measured两模式。最终HGB仅在旧400状态/98,260行拟合，全部新负载共用该单一校准器。区间为五历史块的描述性配对t区间，不能视为独立下层训练确认；240回合与数百万请求不增加独立训练n。','','| 方法 | zero DSR % | measured DSR % | measured均值ms | 逐次池化p95 ms | 最大ms | 能耗比G | MBS/接纳 % | JFI |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,y in s['modes']['measured']['groups'].items():
 ts=audit['pooled_timing']['measured'][name];bs=s['modes']['measured']['blocks'];er=statistics.mean(b[name]['fleet_energy']/b['G']['fleet_energy'] for b in bs.values())
 lines.append(f"| {name} | {s['modes']['zero']['groups'][name]['dsr']*100:.3f} | {y['dsr']*100:.3f} | {ts['mean_seconds']*1000:.3f} | {ts['pooled_p95_seconds']*1000:.3f} | {ts['max_seconds']*1000:.3f} | {er:.4f} | {y['mbs_share']*100:.3f} | {y['mean_fairness']:.5f} |")
lines+=['','DSR为先按每块3负载累加请求计数，再平均5块；MBS份额和JFI为块均值。p95明确采用15,000个逐次时间的池化分位数，与块p95均值区分。能耗比为五块能耗总和分别除以对应G再取均值。','','预定主结果（measured环境DSR）：','']
x=s['modes']['measured']
for ref in ['G',x['best_nonlearning'],'binary_price','L16']:
 d=x['learning_differences'][ref];lines.append(f"- 学习−{ref}：{d['mean']*100:+.3f}个百分点，描述性95% t区间[{d['ci95'][0]*100:+.3f},{d['ci95'][1]*100:+.3f}]，{d['positive']}/5块正向。")
lines+=['',f"学习机队能耗比G={x['energy_ratio_vs_G']:.4f}。相对G至少+0.5个百分点、相对最强非学习至少+0.1个百分点、能耗比≤1.05三个门槛均未通过。逐次p95<1秒的时间门槛通过；24万次实测没有一次超过1秒。这不能改变质量/能耗未过门槛的结论。",'','解析混合对照是有效的机制参照：']
for name in ['mixture','mixture_price']:
 row=secondary['measured'][name];d=row['vs_G'];lines.append(f"- {name}−G：{d['mean']*100:+.3f}个百分点，描述性95% t区间[{d['ci95'][0]*100:+.3f},{d['ci95'][1]*100:+.3f}]，{d['positive']}/5块正向；能耗比G={row['energy_ratio_to_G']:.4f}。")
lines+=['','这些是本轮预定对照的开发结果，不将解析方法改称学习增量，也不据此宣称已完成独立确认。学习即使在zero模式也低于最好解析混合约0.352个百分点，说明失败不只来自额外软件耗时。离线价格只控制旧训练状态的公开预测预算，没有保证闭环能耗预算。','',f"本轮zero与measured的{len(paired)}个对应方法/块/负载动作序列全部相同：{extra['all_mode_action_pairs_equal']}。延迟已进入环境奖励/公平性/指标更新，但这些配对轨迹的动作未发生变化；不能声称验证了延迟诱发的自适应决策或持续排队。实际服务请求数与潜在服务请求数本轮相同，固定潜在分母复核精确重现主DSR。",'','核验范围：','',f"- 从压缩原始记录复算{audit['counts']['steps']:,}步、{audit['counts']['requests']:,}条已接纳请求，逐次核对raw_latency+injected_seconds≤deadline、身份/去向/MBS/分母及聚合统计。",'- 源码、模型、价格与原输入摘要一致；外生随机流及潜在分母配对，同CPU观察区间无重叠，所有记录保留。','- 解析分解无回退；原15项远程测试、32烟雾回合、16个zero重复与32个首状态完整环境一致性均通过。','- 审查类型为主代理自审，不冒充独立审查。','','[主统计](../../../results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/development_v1/analysis/summary.json) · [次要诊断](../../../results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/development_v1/analysis/secondary_diagnostics.json) · [原始记录复核](../validation/priced_closed_loop_20261017/raw_record_audit.json) · [启动前复核](../validation/priced_closed_loop_20261017/preflight_audit.json)','','下一步只在新的冻结协议下检验缓存命中概率校准，保留本轮失败和解析强对照；原十块确认与论文正文尚未更改。真实网络信令、持续排队与软件执行的物理能耗仍未建模。仓库布局检查仅有既有根目录修改总结报告.md白名单问题，本次不动。']
report=Path(a['validation_root']).parent.parent/'deliverables/priced_closed_loop.md'
report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(report=str(report),secondary=secondary,all_mode_action_pairs_equal=extra['all_mode_action_pairs_equal'])))
