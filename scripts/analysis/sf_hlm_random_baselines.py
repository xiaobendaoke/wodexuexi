"""Supplement frozen SF-HLM confirmation with two explicit random policies.

All simulation and numerical analysis run on the remote Linux server. Existing
confirmation releases, protocols, checkpoints and results remain immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

NEW_METHODS = ('full_random', 'upper_random_offload')
PARENT_METHODS = ('learned', 'heuristic', 'random', 'greedy', 'original_frozen', 'original_adapted')
FIELDS = ('phase', 'method', 'block', 'workload', 'episode')
PATH_ARGS = ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results',
    'resources', 'adaptation_root', 'followup_root', 'confirmation_root', 'results_root',
    'release_root', 'delivery_root')
FILES = ('scripts/analysis/sf_hlm_random_baselines.py', 'tests/test_sf_hlm_random_baselines.py', 'paths.py')
COUNTERS = {'generated':'generated_service_count', 'admitted':'admitted_service_count',
    'success':'deadline_satisfied_count', 'energy':'energy', 'mbs':'service_offloads_mbs', 'fairness':'fairness'}


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def registered_tasks(blocks,workloads,episodes):
    tasks=[(phase,method,b,w,e) for phase in ('quality','timing')
        for b in blocks for w in workloads for e in range(episodes) for method in NEW_METHODS]
    if len(set(tasks)) != len(tasks):raise ValueError('Duplicate population')
    return tasks


def eligible(tasks,completed,active):
    timing_ready=all(t in completed for t in tasks if t[0]=='quality')
    return [t for t in tasks if t not in completed and t not in active
        and (t[0]!='timing' or timing_ready)]


def random_seeds(parent_hash,block,workload,episode):
    def seed(kind):
        data=json.dumps([parent_hash,'plain_random_v1',block,workload,episode,kind],separators=(',',':')).encode()
        return int.from_bytes(hashlib.sha256(data).digest()[:4],'big')
    return seed('motion'),seed('offload')


def request_index(env,masks):
    import numpy as np
    requests=[]
    for owner,uav in enumerate(env.uavs):
        service=sorted((ue for ue in uav.current_covered_ues if ue.current_request.is_service),
            key=lambda ue:float(np.linalg.norm(uav.pos[:2]-ue.pos[:2])))
        if len(service)>masks.shape[1]:raise ValueError('Request truncation')
        requests.extend((owner,slot,ue.id) for slot,ue in enumerate(service))
    requests.sort(key=lambda r:r[2])
    if len({r[2] for r in requests})!=len(requests):raise ValueError('Duplicate admission')
    return tuple(requests)


def sample_offload(requests,masks,rng):
    import numpy as np
    actions=np.zeros(masks.shape[:2],np.int64);assignment=[];legal=[]
    for owner,slot,_ in requests:
        choices=np.flatnonzero(masks[owner,slot]>.5)
        if not len(choices):raise ValueError('No legal action for an admitted request')
        action=int(rng.choice(choices));actions[owner,slot]=action
        assignment.append(action);legal.append(choices.tolist())
    return actions,tuple(assignment),legal


def random_policy(env,obs,method,models,motion_rng,offload_rng):
    import config
    import numpy as np
    if method not in NEW_METHODS:raise ValueError('Unknown random policy')
    started=time.perf_counter()
    _,masks=env.get_offloading_obs_and_masks();requests=request_index(env,masks)
    actions,assignment,legal=sample_offload(requests,masks,offload_rng)
    motion=(motion_rng.uniform(-1.,1.,size=(config.NUM_UAVS,config.ACTION_DIM)).astype(np.float32)
        if method=='full_random' else models['upper'].select_actions(obs,exploration=False))
    motion=np.clip(motion,-1,1)
    return motion,actions,dict(request_ids=[r[2] for r in requests],assignment=assignment,
        request_slots=[list(r[:2]) for r in requests],legal_actions=legal,executed_motion=motion.tolist(),
        diagnostic=dict(matching_calls=0,matching_budget=0,repair_evaluations=0,guard_applied=False),
        policy_seconds=time.perf_counter()-started)


def assert_random_step(metrics,info,potential):
    if any(metrics[k] for k in ('forced_service_admissions','service_fallback_count','service_predict_exception_fallback_count')):
        raise ValueError('Admission/fallback protocol violation')
    if not 0<=metrics['deadline_satisfied_count']<=metrics['admitted_service_count']<=metrics['generated_service_count']<=potential:
        raise ValueError('Invalid request counts')
    if metrics['admitted_service_count']!=len(info['request_ids']):raise ValueError('Lost admitted request')
    if metrics['service_offloads_mbs']!=info['assignment'].count(1):raise ValueError('MBS execution differs')
    if any(a not in legal for a,legal in zip(info['assignment'],info['legal_actions'])):raise ValueError('Mask violation')
    if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite environment')
    if info['diagnostic']['matching_calls'] or info['diagnostic']['guard_applied']:raise ValueError('Random policy used search')


def backend(args):return load('random_baseline_followup',args.followup_root/'scripts/analysis/sf_hlm_followup.py')


def parent_args(args):
    a=copy.copy(args);a.results_root=args.confirmation_root;a.seed=0;a.family='complete';a.phase='confirmation_eval'
    return a


def make_protocol(args):
    parent=read(args.confirmation_root/'protocol.json')
    if parent['smoke'] or read(args.confirmation_root/'status.json')['phase']!='CONFIRMATION_COMPLETE':
        raise ValueError('Completed frozen confirmation required')
    blocks=parent['confirmation_blocks'][:1] if args.smoke else parent['confirmation_blocks']
    workloads=parent['confirmation_workloads'][:1] if args.smoke else parent['confirmation_workloads']
    episodes=1 if args.smoke else 6
    return dict(version='sf_hlm_random_baselines_20261004_v1',smoke=args.smoke,
        source_hash=parent['source_hash'],parent_protocol_sha256=sha(args.confirmation_root/'protocol.json'),
        parent_analysis_files={f:sha(args.confirmation_root/'analysis'/f) for f in
            ('confirmation.json','confirmation_eval.json','confirmation_timing.json')},
        files={f:sha(args.release_root/f) for f in FILES},blocks=blocks,workloads=workloads,
        episodes=episodes,steps=8 if args.smoke else 1000,tasks=[list(t) for t in registered_tasks(blocks,workloads,episodes)],
        new_methods=list(NEW_METHODS),reused_methods=list(PARENT_METHODS),
        primary_comparisons=['learned-full_random','learned-upper_random_offload'],
        secondary_comparison='upper_random_offload-full_random',
        control_definitions=dict(full_random='IID Uniform[-1,1]^2 motion and uniform per-request quality-mask-legal offload.',
            upper_random_offload='Same frozen trained upper policy, uniform per-request quality-mask-legal offload.',
            random='Parent control: trained upper + common greedy/repair + random quota search + conditional matching + best retention.'),
        common_rules='Same natural admission, physical configuration, initial-state/exogenous streams, quality mask, and Env step.',
        removed_components=['greedy initialization','ordinary repair','conditional matching','quota search','best retention','predicted allocation-energy guard'],
        energy_guard_is_method_specific_not_a_physical_constraint=True,all_admitted_requests_keep_a_destination=True,
        rng_contract='Separate local motion/offload generators; seeds contain parent protocol and block/workload/episode, never method/phase/concurrency.',
        inference_contract='Aggregate request counts over five workloads and six episodes within each of ten trained-model blocks. Full-random block IDs are paired Monte Carlo replicas, not training seeds.',
        statistical_scope='Supplementary frozen-policy evaluation on previously used confirmation workloads, conditional on this workload set; no new independent training confirmation.',
        p95_condition='Existing parent reserved-core full-policy times + new baseline reserved-core full-policy times; environment step excluded.',
        no_retraining=True,natural_admission=True,quality_mask_unchanged=True,no_new_deployment_claim=True)


def verify_releases(args,p):
    if sha(args.confirmation_root/'protocol.json')!=p['parent_protocol_sha256']:raise ValueError('Parent protocol changed')
    parent=read(args.confirmation_root/'protocol.json')
    for files,root in [(p['files'],args.release_root),(parent['files'],args.followup_root),
            (parent['extension_files'],args.extension_root),(parent['adaptation_files'],args.adaptation_root)]:
        for f,h in files.items():
            if sha(root/f)!=h:raise ValueError('Frozen source changed: '+str(root/f))
    for f,h in p['parent_analysis_files'].items():
        if sha(args.confirmation_root/'analysis'/f)!=h:raise ValueError('Parent analysis changed')


def folder(args,task):
    phase,method,b,w,e=task
    return args.results_root/phase/method/f'block_{b}_workload_{w}'/f'episode_{e:05d}'


def task_contract(args,p,task):
    return dict(zip(FIELDS,task),protocol_sha256=sha(args.results_root/'protocol.json'),
        parent_protocol_sha256=p['parent_protocol_sha256'],source_hash=p['source_hash'])


def verify(args,p,task):
    path=folder(args,task)
    if not (path/'result.json').exists():return False
    result=read(path/'result.json')
    if result['status']!='PASS' or result['contract']!=task_contract(args,p,task) or result['steps']!=p['steps']:
        raise ValueError('Partial/mismatched episode receipt')
    if set(result['files'])!={'episode.json'}:raise ValueError('Missing raw episode')
    for f,h in result['files'].items():
        if sha(path/f)!=h:raise ValueError('Raw episode changed')
    return True


def worker(args):
    import numpy as np
    p=read(args.results_root/'protocol.json');task=tuple(getattr(args,k) for k in FIELDS)
    if list(task) not in p['tasks']:raise ValueError('Unregistered task')
    verify_releases(args,p)
    a=parent_args(args);m=backend(a);r=m.runner(a)
    resource=args.resources/'isolated' if args.phase=='timing' else args.resources
    with r.res.task_lock(args.results_root/'task_locks'/('_'.join(map(str,task))+'.lock')):
        if verify(args,p,task):return
        with r.res.CpuLease(resource,task=['random_baselines',*task],kind='environment') as lease:
            old,_=m.initialize(a,r);parent=read(args.confirmation_root/'protocol.json')
            models,source,check,_=m.basis(a,r,old,parent)
            fingerprint=r.base.policy_fingerprint([models['upper'],models['lower']])
            env,obs=r.episode_env(args.workload,args.episode)
            motion_seed,offload_seed=random_seeds(p['parent_protocol_sha256'],args.block,args.workload,args.episode)
            motion_rng=np.random.default_rng(motion_seed);offload_rng=np.random.default_rng(offload_seed)
            from utils.canonical_experiment import digest,exogenous_frame
            import config
            records=[];trace=hashlib.sha256();physical=hashlib.sha256();total=dict.fromkeys((*COUNTERS,'potential'),0.)
            # Same tiny non-mutating model warm-up in quality/timing; restore all
            # random streams afterwards. Timing excludes checkpoint loading.
            models['upper'].select_actions(obs,exploration=False)
            for step in range(p['steps']):
                trace.update(digest(exogenous_frame(env)).encode())
                potential=sum(ue.latent_request[0]<config.NUM_SERVICES for ue in env.ues)
                motion,actions,info=random_policy(env,obs,args.method,models,motion_rng,offload_rng)
                obs,rewards,metrics=env.step(motion,offloading_actions=actions);obs=np.asarray(obs,np.float32)
                assert_random_step(metrics,info,potential)
                total['potential']+=potential
                for k,n in COUNTERS.items():total[k]+=float(metrics[n])
                record=dict(step=step,potential=potential,metrics=metrics,team_reward=float(np.mean(rewards)),**info)
                physical.update(digest({k:v for k,v in record.items() if k!='policy_seconds'}).encode())
                records.append(record)
            if fingerprint!=r.base.policy_fingerprint([models['upper'],models['lower']]):raise ValueError('Frozen models changed')
            path=folder(args,task);path.mkdir(parents=True,exist_ok=True)
            row=dict(summary=total,steps=records,exogenous_trace_hash=trace.hexdigest(),deterministic_trace_hash=physical.hexdigest())
            if args.phase=='timing':
                quality=read(folder(args,('quality',*task[1:]))/'episode.json')
                if any(row[k]!=quality[k] for k in ('summary','exogenous_trace_hash','deterministic_trace_hash')):
                    raise ValueError('Timing changed physical outcomes or random actions')
            r.res.atomic_json(path/'episode.json',row)
            r.res.atomic_json(path/'result.json',dict(status='PASS',contract=task_contract(args,p,task),steps=p['steps'],
                files={'episode.json':sha(path/'episode.json')},model_fingerprint=fingerprint,
                basis_checkpoint_sha256=sha(check/'contract.json'),config_hash=source['config_hash'],
                motion_seed=motion_seed,offload_seed=offload_seed,cpu=lease.cpu,threads=1))


def command(args,task):
    cmd=[sys.executable,str(args.release_root/FILES[0]),'worker']
    for k in PATH_ARGS:cmd+=['--'+k.replace('_','-'),str(getattr(args,k))]
    for k,v in zip(FIELDS,task):cmd+=['--'+k,str(v)]
    return cmd


def live_workers(args,tasks,res):
    found={}
    for f in Path('/proc').glob('[0-9]*/cmdline'):
        try:words=f.read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeError):continue
        if words[1:3]!=[str(args.release_root/FILES[0]),'worker']:continue
        flags=dict(zip(words[3::2],words[4::2]))
        if flags.get('--results-root')!=str(args.results_root):continue
        task=tuple(flags['--'+k] if k in FIELDS[:2] else int(flags['--'+k]) for k in FIELDS)
        pid=int(f.parent.name);identity=res.process_identity(pid)
        if identity is None:continue
        if task not in tasks or task in found:raise ValueError('Duplicate/unregistered worker')
        found[task]=(pid,identity)
    return found


def execute(args,p,task,res,owner=None):
    if owner:
        while res.process_identity(owner[0])==owner[1]:time.sleep(2)
        if verify(args,p,task):return task
    log=args.results_root/'job_logs'/('_'.join(map(str,task))+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:subprocess.run(command(args,task),cwd=args.source_root,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not verify(args,p,task):raise ValueError('No verified worker result')
    return task


def sign_flip(values):
    observed=abs(sum(values));count=0
    for signs in itertools.product((-1,1),repeat=len(values)):
        count+=abs(sum(a*b for a,b in zip(values,signs)))>=observed-1e-15
    return count/(2**len(values))


def aggregate(args,p,r):
    import numpy as np
    r.base.np=np;units={};raw={};traces={};goldens={};audit=Counter()
    q=read(args.confirmation_root/'analysis/confirmation_eval.json')
    timing=read(args.confirmation_root/'analysis/confirmation_timing.json')
    for t0 in p['tasks']:
        t=tuple(t0)
        if not verify(args,p,t):raise ValueError('Incomplete population')
        path=folder(args,t);row=read(path/'episode.json');s=row['summary'];receipt=read(path/'result.json')
        raw[str(path/'episode.json')]=receipt['files']['episode.json']
        if len(row['steps'])!=p['steps']:raise ValueError('Partial raw episode')
        key=(t[3],t[4]);trace=(row['exogenous_trace_hash'],s['potential'])
        if traces.setdefault(key,trace)!=trace:raise ValueError('Exogenous streams differ')
        if not p['smoke']:
            if key not in goldens:
                a=parent_args(args);a.phase='confirmation_eval';a.method='learned';a.block=p['blocks'][0];a.workload=t[3];a.episode=t[4]
                m=backend(a);m.verified(m.folder(a));golden=read(m.folder(a)/'episode.json')
                goldens[key]=(golden['exogenous_trace_hash'],golden['summary']['potential'])
            if trace!=goldens[key]:raise ValueError('Parent exogenous pairing changed')
            if receipt['basis_checkpoint_sha256']!=read(args.confirmation_root/'confirmation_eval/learned'/f'block_{t[2]}_workload_{t[3]}'/f'episode_{t[4]:05d}'/'result.json')['contract']['basis_checkpoint_sha256']:
                raise ValueError('Different trained upper basis')
        u=units.setdefault((t[0],t[1],t[2]),dict(total=dict.fromkeys((*COUNTERS,'potential'),0.),times=[],steps=0))
        if t[0]=='quality':
            for k,n in COUNTERS.items():
                actual=sum(x['metrics'][n] for x in row['steps'])
                if not math.isclose(actual,s[k],rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Summary count differs')
            if sum(x['potential'] for x in row['steps'])!=s['potential']:raise ValueError('Potential count differs')
            for x in row['steps']:assert_random_step(x['metrics'],x,x['potential'])
            audit['quality_steps']+=p['steps']
        else:
            quality=read(folder(args,('quality',*t[1:]))/'episode.json')
            if any(row[k]!=quality[k] for k in ('summary','exogenous_trace_hash','deterministic_trace_hash')):raise ValueError('Timing replay differs')
            audit['isolated_timing_steps']+=p['steps']
        for k,v in s.items():u['total'][k]+=v
        u['times'].extend(x['policy_seconds'] for x in row['steps']);u['steps']+=p['steps']
    metrics={}
    for (phase,method,b),u in units.items():
        s=u['total'];metrics[phase,method,b]=dict(dsr=s['success']/max(s['generated'],1),potential_dsr=s['success']/max(s['potential'],1),
            admission=s['admitted']/max(s['generated'],1),conditional=s['success']/max(s['admitted'],1),energy=s['energy'],
            energy_per_step=s['energy']/u['steps'],mbs_share=s['mbs']/max(s['admitted'],1),fairness=s['fairness']/u['steps'],
            p95=float(np.quantile(u['times'],.95)),matching_calls=0.)
    bm={}
    for method in NEW_METHODS:
        for b in p['blocks']:
            value=metrics['quality',method,b].copy();value['p95']=metrics['timing',method,b]['p95'];bm[f'{method}:{b}']=value
    if not p['smoke']:
        for method in PARENT_METHODS:
            for b in p['blocks']:
                value=q['block_metrics'][f'{method}:{b}'].copy();value['p95']=timing['block_metrics'][f'{method}:{b}']['p95']
                value['energy_per_step']=value['energy']/(len(p['workloads'])*p['episodes']*p['steps']);bm[f'{method}:{b}']=value
    names=(*PARENT_METHODS,*NEW_METHODS) if not p['smoke'] else NEW_METHODS
    common=('dsr','potential_dsr','admission','conditional','energy','energy_per_step','mbs_share','fairness','p95')
    groups={m:{k:float(np.mean([bm[f'{m}:{b}'][k] for b in p['blocks']])) for k in common} for m in names}
    comparisons={}
    if not p['smoke']:
        for other in NEW_METHODS:
            values=[bm[f'learned:{b}']['dsr']-bm[f'{other}:{b}']['dsr'] for b in p['blocks']]
            comparisons['learned-'+other]=dict(**r.base.paired(values),exact_sign_flip_p=sign_flip(values),
                positive_blocks=sum(v>0 for v in values),energy_ratio=float(np.mean([bm[f'learned:{b}']['energy']/bm[f'{other}:{b}']['energy'] for b in p['blocks']])))
        r.holm(comparisons,list(comparisons))
    values=[bm[f'upper_random_offload:{b}']['dsr']-bm[f'full_random:{b}']['dsr'] for b in p['blocks']]
    secondary=r.base.paired(values)
    result=dict(status='PASS',smoke=p['smoke'],groups=groups,block_metrics=bm,new_primary_comparisons=comparisons,
        secondary_upper_effect=secondary,parent_comparisons=q['comparisons'],raw_files=raw,audit=dict(audit),
        new_quality_episodes=sum(t[0]=='quality' for t in p['tasks']),new_timing_episodes=sum(t[0]=='timing' for t in p['tasks']),
        parent_analysis_files=p['parent_analysis_files'],not_new_independent_training_confirmation=True,
        random_has_no_training_seeds=True,statistical_scope=p['statistical_scope'],quality_mask_preserved=True,
        per_block_product_identity_checked=all(math.isclose(v['dsr'],v['admission']*v['conditional'],rel_tol=1e-10) for v in bm.values()))
    r.res.atomic_json(args.results_root/'analysis/summary.json',result)
    if not p['smoke']:write_report(args,p,result)
    r.res.atomic_json(args.delivery_root/'validation/completion_review.json',dict(status='PASS',tasks=len(p['tasks']),
        **dict(audit),raw_files=len(raw),parent_source_unchanged=True,quality_timing_identical=True,
        mask_and_admission_checked=True,independent_new_training=False,smoke=p['smoke']))


def write_report(args,p,result):
    labels=dict(learned='学习配额搜索',heuristic='启发式配额搜索',random='配额随机搜索（共享强框架）',greedy='贪心＋普通修复',
        original_frozen='原冻结学习下层＋修复',original_adapted='原适配学习下层＋修复',full_random='全随机（质量 mask 内）',upper_random_offload='相同学习上层＋逐请求随机卸载')
    lines=['# SF-HLM 随机基线补充比较（2026-10-04）','',
        '复用已完成确认实验的10套冻结基座与学习搜索模型，5个工作负载×6回合×1000步。新增两种随机策略各300个质量回合与300个隔离计时回合；其余六方法引用同源码、同配置、同负载的冻结确认结果。此次没有重新训练，是在已用确认负载上的补充评价。','',
        '全随机：每步每架UAV的二维动作分别均匀采样于[-1,1]，由原环境执行速度与边界规则；每个实际接纳请求在共同质量mask允许的动作中均匀采样。相同上层＋随机卸载：使用对应块的冻结学习式上层，卸载同上。两者不调用贪心初始化、普通修复、条件匹配、配额搜索或最优保留，也不应用这些优化方法特有的1.05倍预测分配能耗筛选。环境、自然接纳、质量mask与物理参数均相同；本实验不是移除质量mask后的随机策略。','',
        '| 方法 | DSR | 接纳率 | 接纳后完成率 | UAV能耗/步 | MBS占比 | 公平性 | 各块隔离p95均值(ms) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,g in result['groups'].items():
        lines.append(f"| {labels[name]} | {g['dsr']*100:.3f}% | {g['admission']*100:.3f}% | {g['conditional']*100:.3f}% | {g['energy_per_step']:.1f} | {g['mbs_share']*100:.2f}% | {g['fairness']:.4f} | {g['p95']*1000:.2f} |")
    lines+=['','DSR、接纳率和条件完成率先按块合并请求计数，再跨块平均。DSR=接纳率×条件完成率在每个块内成立，表中三个跨块均值不能直接相乘。能耗仅为UAV总能耗，不包括MBS或真实通信部署开销。','',
        '| 新登记比较 | DSR差值(百分点) | 95%训练块配对t区间 | 正向块数 | Holm p | 精确符号翻转p |',
        '|---|---:|---|---:|---:|---:|']
    for name,c in result['new_primary_comparisons'].items():
        ci=c['ci95'];lines.append(f"| 学习－{labels[name.removeprefix('learned-')]} | {c['mean']*100:+.3f} | [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}] | {c['positive_blocks']}/10 | {c['holm_p']:.6g} | {c['exact_sign_flip_p']:.6g} |")
    lines+=['','两个新增学习对照属于预先登记的新Holm族。原学习－配额随机搜索等检验保留原确认族及其证据边界，不重新加入新族选择性改变旧检验。','',
        '全随机没有训练种子。其10个块是与10套学习模型配对的独立策略随机流重复；工作负载与回合不充作独立训练重复。区间反映给定这5个工作负载的模型/随机策略变化，不证明对所有场景泛化。冻结相同上层参数也不意味着长期几何完全相同：卸载改变能量、缓存和后续观测，轨迹可随闭环状态分化。','',
        '全随机对照支持完整系统相对随机决策的比较；相同上层＋逐请求随机卸载支持完整下层优化包的比较；配额随机搜索才用于辨别共享强框架之上学习搜索的额外贡献。前两种优势不能替代第三种的机制证据。','',
        f"验收：{result['new_quality_episodes']}个质量回合、{result['new_timing_episodes']}个隔离计时回合；{result['audit']['quality_steps']}个新增质量步逐项核对计数/接纳/动作合法性。全部计时重放与质量回合的随机动作和物理指标相同；源码、模型指纹及共同外生流核验通过。所有预定回合保留，无按结果排除。"]
    path=args.delivery_root/'deliverables/analysis.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')


def pipeline(args):
    a=parent_args(args);r=backend(a).runner(a);res=r.res
    profile=read(args.resources/'profile.json');isolated=read(args.resources/'isolated/profile.json')
    os.sched_setaffinity(0,set(profile['cpus'][:profile['slots']]))
    p=make_protocol(args);file=args.results_root/'protocol.json'
    if file.exists() and read(file)!=p:raise ValueError('Frozen protocol differs')
    verify_releases(args,p);res.atomic_json(file,p);tasks=[tuple(t) for t in p['tasks']]
    with res.task_lock(args.results_root/'controller.lock'):
        adopted=live_workers(args,tasks,res);completed={t for t in tasks if t not in adopted and verify(args,p,t)}
        futures={};failures=[]
        def status(phase):res.atomic_json(args.results_root/'status.json',dict(phase=phase,pid=os.getpid(),expected=len(tasks),completed=len(completed),
            completed_by_phase=dict(Counter(t[0] for t in completed)),active_by_phase=dict(Counter(t[0] for t in futures.values())),
            failures=failures,unix=time.time(),no_retraining=True,smoke=p['smoke'],scientific_success_claim=False))
        with ThreadPoolExecutor(max_workers=min(profile['slots'],args.workers)) as pool:
            for task,owner in adopted.items():futures[pool.submit(execute,args,p,task,res,owner)]=task
            while len(completed)<len(tasks):
                ready=[] if failures else eligible(tasks,completed,set(futures.values()))
                capacity=min(isolated['slots'] if ready and ready[0][0]=='timing' else profile['slots'],args.workers)
                for task in ready[:max(0,capacity-len(futures))]:futures[pool.submit(execute,args,p,task,res)]=task
                status('RUNNING' if not failures else 'FAILED_DRAINING')
                if not futures:status('FAILED');raise RuntimeError('No runnable task; inspect logs')
                finished,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    task=futures.pop(future)
                    try:completed.add(future.result())
                    except Exception as exc:failures.append(dict(task=task,error=str(exc)))
        if failures:status('FAILED');raise RuntimeError('Worker failed')
        status('AGGREGATING');aggregate(args,p,r);verify_releases(args,p);status('COMPLETE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('pipeline','worker','analyze'))
    for k in PATH_ARGS:parser.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--workers',type=int,default=192)
    for k in FIELDS:parser.add_argument('--'+k,type=str if k in FIELDS[:2] else int)
    args=parser.parse_args()
    for k in PATH_ARGS:setattr(args,k,getattr(args,k).expanduser().resolve())
    if sys.platform!='linux':raise RuntimeError('Simulation, matching and statistics are remote Linux only')
    if args.workers<1:raise ValueError('Workers must be positive')
    if args.mode=='worker':worker(args)
    elif args.mode=='pipeline':pipeline(args)
    else:
        a=parent_args(args);p=read(args.results_root/'protocol.json');verify_releases(args,p)
        aggregate(args,p,backend(a).runner(a))


if __name__=='__main__':main()
