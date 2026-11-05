"""Generate a readable report only after the frozen server experiment finishes."""
import argparse
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
state=json.loads((a.root/'status.json').read_text())
if state.get('status')!='COMPLETE':raise SystemExit('Evidence is not complete; no scientific report generated.')
sys.path.insert(0,str(a.source))
from ch3_analysis import effect
plan=json.loads(a.protocol.read_text());report=json.loads((a.root/'analysis.json').read_text())
if report['source_hash']!=state['source_hash']:raise ValueError('Mixed evidence source')
primary=report['comparisons']['primary']['DSR']
lines=['# 第三章同信息补证结果','',f"协议 `{plan['protocol']}`，源码 `{state['source_hash']}`。",'',
       '## 同信息架构主比较','',
       f"B−D DSR差值 {100*primary['mean_difference']:+.4f} 个百分点，训练种子t 95%区间 {[round(100*x,4) for x in primary['t_ci95']]}，crossed bootstrap区间 {[round(100*x,4) for x in primary['crossed_bootstrap_ci95']]}。",
       f"预定主门槛通过：{primary['criterion_met']}。上层Actor与Critic同时替换，下层同架构分别训练。",'',
       '## 各格主场景结果','', '| 格点 | DSR/% | 接纳率/% | 条件完成率/% |','|---|---:|---:|---:|']
for cell,values in report['means'].items():lines.append(f"| {cell} | {100*values['DSR']:.4f} | {100*values['alpha']:.4f} | {100*values['eta']:.4f} |")
cap=a.root/'capacity_analysis.json'
if cap.exists():
 c=json.loads(cap.read_text());v=c['comparisons']['capacity']['DSR']
 lines+=['',f"固定序列容量比较B−E：{100*v['mean_difference']:+.4f}个百分点，联合门槛通过：{c['capacity_criterion_met']}。宽度及剩余参数差异见冻结发布记录。"]
else:lines+=['','主门槛未通过，容量阶段按预注册规则未启动；不据此宣称注意力无效。']
lines+=['','## 请求级诊断','', '以下按每个种子×负载单元分别列共同接纳与各自额外接纳集合，未将请求数当作独立训练重复。各组条件完成率描述不同条件集合，不识别因果中介。','', '| 训练种子 | 负载 | 集合 | 计数 | 左侧成功数 | 右侧成功数 |','|---|---|---|---:|---:|---:|']
diagnostics=json.loads((a.root/'request_diagnostics.json').read_text())
for row in diagnostics['units']:
 for group,counts in row['groups'].items():
  lines.append(f"| {row['training_seed']} | {row['workload_seed']} | {group} | {counts['count']:.0f} | {counts.get('left_successful',0):.0f} | {counts.get('right_successful',0):.0f} |")
lines+=['','完整任务难度、接入负载、最终执行节点负载和通信/计算时延的分组和、有效样本数见request_diagnostics.json。不可把差值机械归为拥塞。','', '## 冻结策略边界','', '仅前三训练种子、三个边界负载，每负载两个回合。所有结果为探索性敏感性；有不适用单元的场景不按剩余单元生成效果。','', '| 单因素设置 | B−A DSR/百分点 | t 95%区间（描述性） | 状态 |','|---|---:|---|---|']
boundary=json.loads((a.root/'boundary_analysis.json').read_text());lookup={};invalid=set()
for row in boundary['units']:
 path=Path(row['path']);setting,method,seed,workload=path.parts[-4:]
 if row['status']!='PASS':invalid.add(setting);continue
 lookup[setting,method,int(seed.removeprefix('seed_')),int(workload.removeprefix('workload_'))]=row
for setting in plan['boundary']['settings']:
 name=setting['name']
 if name in invalid:lines.append(f'| {name} | — | — | 接口不适用，未删单元平均 |');continue
 matrix=[]
 for seed in plan['training_seeds'][:3]:
  matrix.append([lookup[name,plan['cells']['B'],seed,w]['DSR']-lookup[name,plan['cells']['A'],seed,w]['DSR'] for w in plan['boundary_workload_seeds']])
 v=effect(matrix,plan)
 lines.append(f"| {name} | {100*v['mean_difference']:+.4f} | {[round(100*x,4) for x in v['t_ci95']]} | 描述性 |")
lines+=['','## 解释边界','', '原24.44%→27.93%确认、这次新架构确认、冻结策略边界三者不混合。信令能耗、机载平台与连续队列未被本次补证验证。新结果按独立小节报告，不覆盖历史主表。']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(a.output)
