"""Render the audited zero-price intervention on the experiment server."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);o=p.parse_args()
a=json.loads(o.launch.read_text())['args'];root=Path(a['results_root']);v=Path(a['validation_root'])
read=lambda p:json.loads(p.read_text());s=read(root/'analysis/summary.json');audit=read(v/'raw_record_audit.json')
assert audit['status']=='PASS' and audit['summary_sha256']==hashlib.sha256((root/'analysis/summary.json').read_bytes()).hexdigest()
lines=['# 条件缓存学习：同候选池零软价格消融','','保持25个冻结模型、18维输入、公开冷态规则、配额提案、最多10次匹配和硬能耗guard，只把学习方法的软价格固定为0；不再声称满足原训练期软价格能耗约束。全部400状态重新生成公开候选池，原价格选择400/400复现，决定封存后才作物理评分。','',
f"零价改变{audit['parent_choices_changed']}/400个分配；{audit['physical_states_replayed']}个新结果重新物理执行、{audit['physical_states_reused_after_exact_match']}个同状态同分配结果核验复用。0/10/20/30 ms下的成功数与带价学习完全相同；真实机队能耗/G={s['groups']['cache_regime_zero_price']['energy_ratio_to_G']:.6f}。",'',
'| 共同延迟 ms | 最好无标签对照 | 零价学习−该对照 pp | 描述性95% t区间 pp | 正向块 | 零价学习−G pp |','|---|---|---:|---|---:|---:|']
for k,x in s['contrasts'].items():
 e=x['vs_best'];lines.append(f"| {k} | {x['best_nonlearning']} | {100*e['mean']:+.5f} | [{100*e['ci95'][0]:+.5f},{100*e['ci95'][1]:+.5f}] | {e['positive']}/5 | {100*x['vs_G']['mean']:+.5f} |")
lines+=['',f"预定联合筛选通过：{s['decision']['common_delay_screen_pass']}。失败按协议保留，不启动原生费用、闭环或独立确认，也不追加价格网格求正。共同固定延迟不是实测软件费用；本轮没有新的正式时间样本。",'',
'本消融未支持“软价格抑制了成功数”的解释，适用范围仅为这组冻结状态、模型和有限候选池。五个历史上层块、400状态和25交叉折不能混作独立训练重复；此为主代理自审，尚未解决持续研究目标。原十块确认及论文不改。','',
f"协议SHA `{audit['protocol_sha256']}`；统计SHA `{audit['summary_sha256']}`。",'',
'[原始审计](../validation/cache_regime_zero_20261018/raw_record_audit.json) · [共同延迟统计](../../../results/sf_hlm_learning_feasibility_20261016/cache_regime_zero_20261018/development_v1/analysis/summary.json)']
out=v.parent.parent/'deliverables/cache_regime_zero.md';out.write_text('\n'.join(lines)+'\n');print(out)
