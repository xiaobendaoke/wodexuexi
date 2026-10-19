"""Verify the completed confirmation evidence chain and export bounded claims.

This is a separate reporting release; the frozen timed evaluator is unchanged.
All data validation, arithmetic and formatting run on the experiment server.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

METHODS=('G','R10','R16','binary_price','mixture','mixture_price','cold_prior_price','cold_half_proxy_price','HGB_probability_price')
FILES=('paths.py','scripts/analysis/sf_hlm_framework_evidence.py','tests/test_sf_hlm_framework_evidence.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
RATE_KEYS=('dsr','potential_dsr','raw_dsr','admission_rate','accepted_quality','mbs_share','mean_fairness')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def entry_path(data):
    root=Path(data.get('release',data.get('frozen_release','')));name=data.get('entry_script','scripts/analysis/sf_hlm_framework_evaluation.py')
    if name not in ('scripts/analysis/sf_hlm_framework_evaluation.py','scripts/analysis/sf_hlm_framework_saturated.py'):raise ValueError('Unregistered evaluation entry')
    return root/name
def remote():
    if sys.platform!='linux':raise RuntimeError('All scientific evidence validation and reporting run remotely')
def close(left,right):return math.isclose(float(left),float(right),abs_tol=1e-13,rel_tol=1e-13)


def check_summary(p,s):
    remote();smoke=p['runner_smoke'];blocks=p['blocks'];primary='mixture_price'
    if p['methods']!=list(METHODS) or p['primary']!=primary or p['primary_comparisons']!=[[primary,'G'],[primary,'binary_price']]:raise ValueError('Registered reporting family changed')
    if set(s['groups'])!=set(METHODS) or set(s['blocks'])!=set(map(str,blocks)) or s['runner_smoke']!=smoke or s['episodes']!=len(p['tasks']) or s['steps']!=len(p['tasks'])*p['steps']:raise ValueError('Evidence population is incomplete')
    if not s['exogenous_streams_paired'] or not s['fixed_potential_denominator_paired'] or not s['dsr_decomposition_verified_per_block'] or not s['fixed_development_hgb_model'] or s['quota_ppo_models_evaluated']!=0:raise ValueError('Statistical or model scope changed')
    if any(s['decision'][name] for name in ('unique_learning_increment_proven','old_sf_hlm_contract_satisfied','goal_complete')):raise ValueError('Framework analysis overclaims the separate learning objective')
    for block in blocks:
        row=s['blocks'][str(block)]
        if set(row)!=set(METHODS):raise ValueError('Comparator missing inside a block')
        for method in METHODS:
            z=row[method]
            if z['generated']<=0 or z['potential']<=0 or not 0<=z['success']<=z['raw_success']<=z['admitted']<=z['generated']<=z['potential'] or z['shadow_fee_success']!=z['success']:raise ValueError('Invalid block request accounting')
            if not close(z['dsr'],z['success']/z['generated']) or not close(z['potential_dsr'],z['success']/z['potential']) or not close(z['raw_dsr'],z['raw_success']/z['generated']) or not close(z['admission_rate'],z['admitted']/z['generated']):raise ValueError('Block rate does not reproduce counts')
            if z['admitted'] and (not close(z['accepted_quality'],z['success']/z['admitted']) or not close(z['dsr'],z['admission_rate']*z['accepted_quality'])):raise ValueError('Block DSR decomposition differs')
            if z['episodes']!=len(p['workloads'])*p['episodes_per_workload'] or z['timing_samples']!=z['episodes']*p['steps']:raise ValueError('Block call/episode population differs')
    for method in METHODS:
        for key in RATE_KEYS:
            values=[s['blocks'][str(block)][method][key] for block in blocks];expected=None if any(x is None for x in values) else statistics.mean(values);actual=s['groups'][method][key]
            if expected is None and actual is not None or expected is not None and (actual is None or not close(actual,expected)):raise ValueError('Reported group macro rate differs')
    if smoke:
        if s['primary_family'] or s['comparisons'] or s['decision']['confirmation_gate_evaluated'] or s['decision']['framework_confirmation_signal']:raise ValueError('Smoke data cannot support inference')
        return dict(classification='COMPATIBILITY_ONLY',confirmation_claim_permitted=False)
    if len(blocks)!=20 or len(p['tasks'])!=2700 or p['steps']!=1000 or len(p['workloads'])!=5 or p['episodes_per_workload']!=3 or not s['decision']['confirmation_gate_evaluated']:raise ValueError('Formal confirmation population or inference is incomplete')
    if set(s['comparisons'])!=set(METHODS)-{primary} or set(s['primary_family'])!={'G','binary_price'}:raise ValueError('Required controls or primary comparison missing')
    for control in set(METHODS)-{primary}:
        diff=s['comparisons'][control]['dsr'];values=[s['blocks'][str(block)][primary]['dsr']-s['blocks'][str(block)][control]['dsr'] for block in blocks]
        if diff['n']!=20 or len(diff['values'])!=20 or any(not close(x,y) for x,y in zip(diff['values'],values)) or not close(diff['mean'],statistics.mean(values)) or diff['positive']!=sum(x>0 for x in values):raise ValueError('Paired block comparison does not reproduce')
        ratios=[s['blocks'][str(block)][primary]['fleet_energy']/s['blocks'][str(block)][control]['fleet_energy'] for block in blocks]
        if not close(s['comparisons'][control]['fleet_energy_ratio']['mean'],statistics.mean(ratios)):raise ValueError('Energy comparison differs from block ratios')
    family=s['primary_family'];ordered=sorted(family,key=lambda k:family[k]['t_p_two_sided']);adjusted={};previous=0.
    for i,key in enumerate(ordered):previous=min(1.,max(previous,(2-i)*family[key]['t_p_two_sided']));adjusted[key]=previous
    for key in family:
        if not close(family[key]['holm_p'],adjusted[key]) or any(family[key][field]!=s['comparisons'][key]['dsr'][field] for field in ('mean','ci95','values','n','positive','t_p_two_sided','exact_signflip_p')) or family[key]['exact_signflip_p'] is None:raise ValueError('Frozen two-comparison test family differs')
    g=p['gates'];checks=dict(mean_thresholds_pass=family['G']['mean']>=g['mean_gain_G'] and family['binary_price']['mean']>=g['mean_gain_binary'],positive_block_thresholds_pass=all(row['positive']>=g['positive_blocks'] for row in family.values()),primary_holm_family_pass=all(value<g['holm_alpha'] for value in adjusted.values()),energy_mean_ratio_pass=s['comparisons']['G']['fleet_energy_ratio']['mean']<=g['energy_ratio_G'],all_block_p95_pass=all(row[primary]['p95_seconds']<g['per_block_p95_seconds'] for row in s['blocks'].values()))
    if any(s['decision'][key]!=value for key,value in checks.items()) or s['decision']['framework_confirmation_signal']!=all(checks.values()):raise ValueError('Reported promotion decision does not reproduce frozen conditions')
    return dict(classification='CONFIRMED_FRAMEWORK_CANDIDATE' if all(checks.values()) else 'COMPLETED_WITHOUT_FRAMEWORK_PROMOTION',confirmation_claim_permitted=all(checks.values()),recomputed_conditions=checks)


def render(p,s,decision):
    remote()
    if p['runner_smoke']:
        return '\n'.join(('# 独立烟雾证据链核验','',f"已核对{s['episodes']}回合、{s['steps']}步的完成与原始自审回执。",'', '本输出仅验证报告器与已封存烟雾兼容，不报告方法性能差异，不作科学推断，不代表正式确认、学习增量或持续目标完成。',''))
    label='通过预定框架晋级条件' if decision['confirmation_claim_permitted'] else '未通过预定框架晋级条件'
    lines=['# 全新基座上的计全费用闭环确认','',f'固定候选mixture_price：{label}。结果保留全部预定方法与比较，不改候选或追加样本。','',
        '20个新上层训练基座×5个新负载×3回合；每方法300回合，每回合1000步，9方法共2700回合、270万步。推断单位是20个新训练块；固定开发HGB不是20个新训练下层模型。块内先合并15回合成功/生成计数，再求率和块配对差。','',
        '本次硬件运行条件：'+p.get('hardware_regime','原冻结的八物理核隔离计时条件')+'。本次所有软件费用均实测，不与其它硬件并发条件的开发费用拼接。','',
        '| 方法 | 计费DSR % | 潜在分母DSR % | 同轨迹原物理DSR % | 接纳率 % | 接纳后质量 % | MBS/接纳 % | 公平性 | 平均调用 ms | 合并逐次p95 ms |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value,scale=1.):return '未定义' if value is None else f'{value*scale:.5f}'
    for method in METHODS:
        z=s['groups'][method];values=[number(z[k],100.) for k in RATE_KEYS[:-1]]+[number(z['mean_fairness']),number(z['mean_seconds'],1000.),number(z['pooled_p95_seconds'],1000.)];lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines+=['','各率是20个块率的均值；接纳率与接纳后质量的跨块均值不相乘。同轨迹原物理DSR使用相同动作和状态中、注入当次软件延迟前的成功数，不是另一次零费用闭环。表中p95合并全部调用；晋级使用每个训练块各15,000次完整调用的p95。','',
        '| 主候选−参照 | 块均差 pp | 95%配对t区间 pp | 正向块 | 双侧t p | Holm p | 精确符号翻转p |','|---|---:|---|---:|---:|---:|---:|']
    for method in ('G','binary_price'):
        z=s['primary_family'][method];ci=', '.join(number(v,100.) for v in z['ci95']);lines.append(f"| {method} | {number(z['mean'],100.)} | [{ci}] | {z['positive']}/20 | {z['t_p_two_sided']:.8g} | {z['holm_p']:.8g} | {z['exact_signflip_p']:.8g} |")
    lines+=['','精确符号翻转枚举2^20种块符号模式，是独立块配对效应对称性假设下的敏感性分析，不是无假设因果随机化检验。Holm仅校正两个预定主比较；其余比较是完整描述，不作为另一个确认检验族。','',
        '| 所有参照 | 候选−参照计费DSR pp | 描述性95% t区间 pp | 正向块 | 候选/参照平均块能耗比 |','|---|---:|---|---:|---:|']
    for method in METHODS:
        if method=='mixture_price':continue
        z=s['comparisons'][method];e=z['dsr'];ci=', '.join(number(v,100.) for v in e['ci95']);lines.append(f"| {method} | {number(e['mean'],100.)} | [{ci}] | {e['positive']}/20 | {number(z['fleet_energy_ratio']['mean'])} |")
    g=p['gates'];lines+=['',f"预定条件：相对G均值≥{100*g['mean_gain_G']:.3f} pp，相对binary均值≥{100*g['mean_gain_binary']:.3f} pp；两比较各至少{g['positive_blocks']}/20块正且Holm双侧p<{g['holm_alpha']}；候选/G平均块能耗比≤{g['energy_ratio_G']}，各块逐次p95<{g['per_block_p95_seconds']}秒。",'','晋级条件逐项复算：','']+[f"- {key}: {value}" for key,value in decision['recomputed_conditions'].items()]
    lines+=['','全部块/方法的原计数、能耗、MBS、公平性和完整时间摘要见block_metrics.csv；源分析保留潜在分母和全部质量取舍的配对结果。均值达到实用阈值不代表真实效应区间下界也达到该阈值。','',
        '全量270万步完成原始计数/阈值与计时核验；预定180回合、18万步完成物理Env重放。重放使用封存动作、运动dtype和原软件费用，不重新计时，不替换原费用。二者覆盖范围不同。','',
        '结论边界：本轮只检验固定解析下层在全新轨迹基座上的框架实用性及两个预定比较。它不证明PPO独有增量、软价格相对未加价混合有效、胜过所有方法或原两个强对照各+1 pp合同通过。实测软件延迟在当前请求指标/奖励/下一观测前注入；连续队列、真实网络和软件CPU物理能耗不在此仿真合同中。原论文改稿仍需独立授权。','',
        '源码、原始回执、主张覆盖与文件摘要见evidence_index.json。证据完整与方法晋级分开记录，整个持续目标不由本报告自动判定完成。','']
    return '\n'.join(lines)


def export(a):
    remote()
    if a.output.exists():raise FileExistsError('Immutable evidence export namespace')
    data=read(a.run_manifest);runner_root=Path(data.get('release',data.get('frozen_release','')));args=argparse.Namespace(**{k:Path(v) for k,v in data['args'].items()});m=load('framework_evidence_frozen_runner',entry_path(data));p=read(args.results_root/'protocol.json');args.runner_smoke=p['runner_smoke']
    if args.runner_smoke!=a.smoke or runner_root!=args.release_root:raise ValueError('Evidence smoke/source identity differs')
    f,fa,fp=m.verify_runner(args,p);m.prerequisites(args);done=read(args.results_root/'completion.json');audit=read(args.results_root/'raw_audit.json');s=read(args.results_root/'analysis/summary.json');direct=read(args.direct_audit)
    if done['status']!='COMPLETED' or done['runner_smoke']!=a.smoke or done['formal_confirmation_completed']!=(not a.smoke) or done['protocol_sha256']!=sha(args.results_root/'protocol.json') or done['summary_sha256']!=sha(args.results_root/'analysis/summary.json') or done['raw_audit_sha256']!=sha(args.results_root/'raw_audit.json') or audit['status']!='PASS' or audit['runner_smoke']!=a.smoke or audit['summary_sha256']!=done['summary_sha256'] or audit['protocol_sha256']!=done['protocol_sha256']:raise ValueError('Completed raw audit evidence chain missing or stale')
    expected_replays=len(p['physical_replay_tasks'])
    if audit['episodes']!=len(p['tasks']) or audit['all_steps_threshold_and_clock_checked']!=len(p['tasks'])*p['steps'] or audit['physical_replay_episodes']!=expected_replays or audit['physical_replay_steps']!=expected_replays*p['steps'] or audit['replay_policy_clocks_remeasured'] or audit['model_or_price_refit']:raise ValueError('Raw audit coverage differs from its actual design')
    files={str(path):sha(path) for path in (a.run_manifest,args.results_root/'protocol.json',args.results_root/'completion.json',args.results_root/'raw_audit.json',args.results_root/'analysis/summary.json',args.results_root/'collection_complete.json',args.direct_audit,args.price_audit)};paired={};hgb=set();configs={};basis={row['block']:row for row in direct['blocks']};replay={tuple(row['task']):row for row in audit['physical_replays']}
    if set(replay)!=set(map(tuple,p['physical_replay_tasks'])):raise ValueError('Actual physical replay sample changed')
    for task in p['tasks']:
        record=m.verified_episode(args,p,task);key=tuple(task[1:4]);value=(record['exogenous_hash'],record['totals']['potential'])
        if key in paired and paired[key]!=value:raise ValueError('Episode source streams are not paired')
        paired[key]=value;hgb.add(record['hgb_fingerprint'])
        if record['upper_fingerprint']!=p['upper_fingerprints'][str(task[1])] or record['checkpoint']!=basis[task[1]]['checkpoint']:raise ValueError('Episode basis fingerprint or actual source checkpoint differs')
        if task[1] in configs and configs[task[1]]!=record['config_sha256']:raise ValueError('Episode configuration differs within an actual basis block')
        configs[task[1]]=record['config_sha256']
        if tuple(task) in replay and (replay[tuple(task)]['steps_sha256']!=record['steps_sha256'] or replay[tuple(task)]['steps']!=p['steps']):raise ValueError('Physical replay refers to a different sealed trajectory')
        path=m.task_folder(args,task);files[str(path/'result.json')]=sha(path/'result.json');files[str(path/'steps.json.gz')]=record['steps_sha256']
    if len(hgb)!=1:raise ValueError('More than one fixed HGB evaluated')
    for row in direct['blocks']:
        checkpoint=Path(row['checkpoint']);canonical=checkpoint.parent.parent;receipt=canonical.parent/'result.json'
        for path,digest in ((canonical/'manifest.json',row['canonical_manifest_sha256']),(checkpoint/'contract.json',row['checkpoint_contract_sha256']),(receipt,row['source_receipt_sha256'])):
            if sha(path)!=digest:raise ValueError('Direct training evidence changed after audit')
            files[str(path)]=digest
    decision=check_summary(p,s);own={name:sha(a.release_root/name) for name in FILES};write(a.output/'protocol.json',dict(name='sf_hlm_framework_evidence_export_20261018_v1',files=own,input_files=files,runner_release=str(runner_root),smoke=a.smoke,no_model_or_timer_changes=True))
    (a.output/'report.md').write_text(render(p,s,decision))
    fields=('block','method',*s['blocks'][str(p['blocks'][0])][METHODS[0]].keys())
    with (a.output/'block_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for block in p['blocks']:
            for method in METHODS:writer.writerow(dict(block=block,method=method,**s['blocks'][str(block)][method]))
    index=dict(status='VERIFIED_EVIDENCE',scientific_outcome=decision,smoke=a.smoke,formal_confirmation_completed=not a.smoke,goal_complete=False,unique_learning_increment_proven=False,
        coverage=dict(fresh_training_blocks=direct['foundations'],actual_training_seed_manifests=direct['actual_seed_and_recipe_manifests_checked'],upper_states_perturb_reloaded=direct['upper_state_perturb_reload_checked'],episodes=len(p['tasks']),all_accounting_steps=audit['all_steps_threshold_and_clock_checked'],physical_replay_episodes=expected_replays,physical_replay_steps=audit['physical_replay_steps'],same_fixed_hgb_fingerprint=next(iter(hgb)),all_nine_methods_retained=True),
        scope='Framework confirmation; independent upper blocks, one fixed development HGB, no newly trained lower PPO; self-audit, not an independent reviewer',source_summary= str(args.results_root/'analysis/summary.json'),input_files=files)
    write(a.output/'evidence_index.json',index)
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('Evidence source changed while exporting')
    write(a.output/'completion.json',dict(status='COMPLETED',smoke=a.smoke,protocol_sha256=sha(a.output/'protocol.json'),files={name:sha(a.output/name) for name in ('report.md','block_metrics.csv','evidence_index.json')},scientific_outcome=decision,formal_confirmation_completed=not a.smoke,goal_complete=False));print(json.dumps(dict(status='COMPLETED',smoke=a.smoke,classification=decision['classification'],episodes=len(p['tasks']))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--release-root',type=Path,default=REPO_ROOT);p.add_argument('--run-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();remote()
    # Resource management remains the same archived server implementation.
    data=read(a.run_manifest);m=load('framework_evidence_resource_entry',entry_path(data));args=argparse.Namespace(**{k:Path(v) for k,v in data['args'].items()});f,fa,fp=m.foundation(args);old=f.parents(fa)
    with f.resources(fa).CpuLease(old.resources,task=['framework_evidence_export',a.smoke],kind='environment'):export(a)


if __name__=='__main__':main()
