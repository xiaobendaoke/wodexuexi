"""Frozen diagnostics of the incremental current-plan conditioning advantage."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import copy
import gzip
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
import traceback
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from environment.queued_service import Engine,Protocol
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_service_controllers import assignments
from utils.queued_service_learning import VirtualQueue,state_features
from utils.queued_service_shared_budget import BudgetAdapter
from utils.queued_service_conditioning import conditioning_value,value_fit_by_phase
from utils.queued_service_diagnostics import probability_metrics,commit_sequence,mc_returns
from scripts.analysis import queued_service_shared_budget as trial
from scripts.analysis.queued_service_business import read,raw_write
from scripts.analysis.queued_service_validation import START,atomic,sha,now,core_order
from scripts.analysis.queued_service_coupling import cooperative_plan

DIAGNOSTIC='conditioning_diagnostic_20261101'
MODES=('native','swap','greedy_motion','model_actual','model_source')
SOURCE_FILES=trial.SOURCE_FILES+('utils/queued_service_conditioning.py','scripts/analysis/diagnose_queued_service_conditioning.py',
    'tests/test_queued_service_conditioning.py')


def contract():
    files={name:sha(REPO_ROOT/name) for name in SOURCE_FILES}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files


def locations(task):
    out,val=trial.locations(task);suffix=Path(DIAGNOSTIC)/contract()[0][:12]
    return out/suffix,val/suffix


def jobs():
    result=[dict(kind='episode',method=m,block=b,workload=w,mode=mode)
        for m,b,w,mode in itertools.product(trial.core.METHODS,range(3),trial.core.EVALUATION,MODES)]
    result += [dict(kind='credit',method=m,block=b,round=r) for m,b,r in itertools.product(trial.core.METHODS,range(3),range(1,7))]
    return result


def identity(spec):return '_'.join(f'{k}-{spec[k]}' for k in sorted(spec))


def case_path(out,spec):return out/'cases'/(identity(spec)+'.json.gz')


def model_at(task,method,block,cp,torch):
    from marl_models.queued_service_shared_budget import BudgetPolicy
    out,_=trial.locations(task);path=trial.core.model_path(out,method,block,cp)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['source_contract']==trial.contract()[0] and saved['method']==method and saved['block']==block and saved['checkpoint']==cp
    ref=trial.core.training_jobs(method,block,cp+1)[0] if cp<6 else trial.core.evaluation_jobs(method,block,6)[0]
    raw=trial.core.result_path(out,ref);manifest=read(out/'analysis.json')['raw_sha256']
    assert sha(raw)==manifest[raw.relative_to(out).as_posix()] and read(raw)['checkpoint_sha256']==sha(path)
    model=BudgetPolicy(method);model.load_state_dict(saved['model']);model.eval();return model,sha(path)


def choose(model,obs,adapter,seed,mode,torch):
    native=model.method
    try:
        if mode=='swap':model.method='decomposed' if native=='joint' else 'joint'
        row,plan,motion=model.act(obs,adapter,seed)
    finally:model.method=native
    if mode=='greedy_motion':motion=adapter.plans[max(range(len(adapter.plans)),key=lambda i:row['motion_probs'][i])]
    if mode in ('model_actual','model_source'):
        prediction_plan=plan if mode=='model_actual' else {x['uid']:x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
        best=None;zero=[(0.,0.)]*5
        for i,candidate in enumerate(adapter.plans):
            forecast=Engine.from_observation(obs);forecast.assign(prediction_plan);forecast.advance(candidate)
            forecast.admit();forecast.assign(assignments(forecast.observe(),'queue'));forecast.advance(zero)
            pending=sum(x.remaining_cycles/forecast.p.capacity[x.target if x.target is not None else 5] for x in forecast.tasks.values())
            score=(forecast.counts['success'],-forecast.counts['expired'],-pending,adapter.history._history_score(obs,candidate))
            if best is None or score>best[0]:best=(score,i)
        motion=adapter.plans[best[1]]
    return row,plan,motion


def sample_plan(model,obs,weights,seed,torch):
    from marl_models.queued_service_shared_budget import tensor
    prefix=VirtualQueue(obs);plan={};generator=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for x in obs['tasks']:
            if x['source'] is None or x['target'] is not None:continue
            state,features,mask,prior=prefix.features(x['uid'],weights)
            dist=model.allocation(tensor([state]),tensor([features]),tensor([mask],torch.bool),tensor([prior]))
            node=int(torch.multinomial(dist.probs[0],1,generator=generator));plan[x['uid']]=node;prefix.assign(x['uid'],node)
    return plan


def motion_prob(model,bank,torch,state=None,features=None):
    from marl_models.queued_service_shared_budget import tensor
    with torch.no_grad():
        p=model.motion(tensor([bank['state'] if state is None else state]),tensor([bank['features'] if features is None else features]),
            tensor([bank['mask']],torch.bool)).probs[0]
    return p[:sum(bank['mask'])].tolist()


def payoff_matrix(engine,tape,plans,actions):
    zero=[(0.,0.)]*5;base=engine.counts['success'];initial=len(engine.tasks);result={}
    for scale in ('one_step','six_steps'):
        values={};denominators=[]
        for name,plan in plans.items():
            vector=[]
            for first in actions:
                sequence=[first]+[zero]*5 if scale=='one_step' else commit_sequence(engine,first)
                f=copy.deepcopy(engine);offset=0;generated=initial
                while offset<6 or f.tasks:
                    assert offset<=12
                    if offset:
                        arrivals=tape.slot(int(engine.t)+offset) if offset<6 else ();generated+=len(arrivals)
                        f.add(arrivals);new=f.admit();route={uid:f.tasks[uid].source for uid in new}
                    else:route=plan
                    f.assign(route);f.advance(sequence[offset] if offset<6 else zero);offset+=1
                f.audit();vector.append(f.counts['success']-base);denominators.append(generated)
            values[name]=vector
        assert len(set(denominators))==1
        sampled=[values[name] for name in ('actual','sample1','sample2')]
        result[scale]=dict(values=values,at_risk=denominators[0],on_policy=conditioning_value(sampled),
            extreme=conditioning_value([values['local'],values['cooperative']]))
    return result


def probe(engine,tape,obs,adapter,plan,models,spec,torch):
    import numpy as np
    from marl_models.queued_service_shared_budget import normalized_candidates,tensor
    plans=dict(actual=plan)
    for i in (1,2):
        seed=int.from_bytes(hashlib.sha256(f'{DIAGNOSTIC}/{spec}/{obs["time"]}/{i}'.encode()).digest()[:8],'big')%(2**63-1)
        plans[f'sample{i}']=sample_plan(models[6],obs,adapter.history.weights,seed,torch)
    plans['local']={x['uid']:x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    plans['cooperative']=cooperative_plan(obs)
    banks={'pre':adapter.motion(obs)}
    for name,route in plans.items():
        prefix=VirtualQueue(obs)
        for uid,node in route.items():prefix.assign(uid,node)
        banks[name]=adapter.motion(prefix.obs)
    k=len(adapter.plans);probabilities={};normal={}
    for name,bank in banks.items():
        normal[name]=normalized_candidates(tensor([bank['features']]),tensor([bank['mask']],torch.bool))[0,:k].numpy()
    for cp,model in models.items():
        p={name:motion_prob(model,bank,torch) for name,bank in banks.items()}
        p['pre_context_post_features']=motion_prob(model,banks['actual'],torch,state=banks['pre']['state'])
        p['post_context_pre_features']=motion_prob(model,banks['pre'],torch,state=banks['actual']['state'])
        probabilities[str(cp)]=p
    norms={}
    for name in ('sample1','sample2','local','cooperative'):
        raw=np.asarray(banks[name]['features'][:k])-np.asarray(banks['actual']['features'][:k])
        norms[name]=dict(raw_rms=float(np.sqrt(np.mean(raw*raw))),normalized_rms=float(np.sqrt(np.mean((normal[name]-normal['actual'])**2))))
    distinct=len({tuple(sorted(plans[name].items())) for name in ('actual','sample1','sample2')})
    native='actual' if spec['method']=='joint' else 'pre';p=probabilities['6']
    return dict(time=obs['time'],candidates=k,new_tasks=len(plans['actual']),sampled_plan_distinct=distinct,
        eligible=len(plans['actual'])>0 and k>1 and distinct>1,
        pre_new_count=banks['pre']['state'][74]*40,post_new_count=banks['actual']['state'][74]*40,
        pre_state=banks['pre']['state'],post_state=banks['actual']['state'],plans=plans,
        probabilities=probabilities,input_change=norms,
        native_vs_uniform=probability_metrics(p[native],[1/k]*k),
        pre_vs_post=probability_metrics(p['pre'],p['actual']),
        sampled_plan_tv=statistics.mean(probability_metrics(p['actual'],p[name])['tv'] for name in ('sample1','sample2')),
        initial_final_tv=probability_metrics(probabilities['0'][native],p[native]),
        immediate_completion_feature=[banks['actual']['features'][i][74]*20 for i in range(k)],
        payoff=payoff_matrix(engine,tape,plans,adapter.plans),
        boundary='Future arrivals only settle fixed-action matrices; equal-weight three sampled plans, finite candidate bank and fixed continuation, not a long-term policy bound')


def episode_case(spec,task):
    torch=trial.core.configure();out,val=locations(task);parent_out,_=trial.locations(task)
    original=dict(kind='eval',method=spec['method'],block=spec['block'],checkpoint=6,profile='burst',workload=spec['workload'])
    raw_path=trial.core.result_path(parent_out,original);expected=read(raw_path)
    assert sha(raw_path)==read(parent_out/'analysis.json')['raw_sha256'][raw_path.relative_to(parent_out).as_posix()]
    model,cp_sha=model_at(task,spec['method'],spec['block'],6,torch);assert cp_sha==expected['checkpoint_sha256']
    models={0:model_at(task,spec['method'],spec['block'],0,torch)[0],6:model} if spec['mode']=='native' and not spec.get('smoke') else {}
    tape=RecoveryScenario(spec['workload'],'burst','normal',.6);assert tape.arrival_fingerprint==expected['arrival_fingerprint']
    e=Engine(Protocol(),START,tape.ue_positions);adapter=BudgetAdapter();rows=[];probes=[];values=[];times=[];verified=0
    progress=val/'progress'/(identity(spec)+'.json')
    future_mc=mc_returns([r['success']/20 for r in expected['rows']]);start=time.perf_counter()
    while e.t<200 or e.tasks:
        t=int(e.t);assert t<=204;e.add(tape.slot(t));e.admit();obs=e.observe();began=time.perf_counter()
        row,plan,motion=choose(model,obs,adapter,trial.action_seed(original,t),spec['mode'],torch);times.append(time.perf_counter()-began)
        if spec['mode']=='native':values.append(dict(time=t,prediction=row['value'],target=future_mc[t]))
        if models and t in (40,75,100,150):probes.append(probe(e,tape,obs,adapter,plan,models,spec,torch))
        e.assign(plan);actual=e.advance(motion)
        if spec['mode']=='native':
            old=expected['rows'][t]
            assert all(actual[key]==old[key] for key in ('generated','completed','expired','admitted','active','success'))
            assert plan=={int(uid):node for uid,node in old['assignment'].items()}
            assert all(abs(a-b)<1e-9 for u,v in zip(motion,old['motion']) for a,b in zip(u,v));verified+=1
        rows.append(actual)
        if t%20==0:atomic(progress,dict(stage='running',time=time.time(),step=t,cpu=sorted(os.sched_getaffinity(0))))
    return dict(spec=spec,diagnostic_contract=contract()[0],parent_contract=trial.contract()[0],checkpoint_sha256=cp_sha,
        parent_raw_sha256=sha(raw_path),fingerprint=tape.arrival_fingerprint,replay_steps_exact=verified,
        summary=e.summary(),audit=e.audit(),rows=rows,values=values,probes=probes,
        decision_seconds=statistics.mean(times),seconds=time.perf_counter()-start,cpu=sorted(os.sched_getaffinity(0)))


def cosine(a,b,np):
    d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(np.dot(a,b)/d) if d>1e-20 else None


def credit_case(spec,task):
    torch=trial.core.configure();import numpy as np
    from marl_models.queued_service_shared_budget import tensor
    method=spec['method'];block=spec['block'];cp=spec['round']-1;model,cp_sha=model_at(task,method,block,cp,torch)
    source_out,_=trial.locations(task);manifest=read(source_out/'analysis.json');episodes=[];refs=[]
    for job in trial.core.training_jobs(method,block,cp+1):
        raw_path=trial.core.result_path(source_out,job);raw=read(raw_path)
        assert sha(raw_path)==manifest['raw_sha256'][raw_path.relative_to(source_out).as_posix()] and raw['checkpoint_sha256']==cp_sha
        path=task/'.tmp'/trial.core.TASK/trial.REVISION/'experience'/(trial.core.job_id(job)+'.pkl.gz')
        assert sha(path)==raw['experience_sha256'];rows=pickle.loads(gzip.decompress(path.read_bytes()));episodes.append(rows)
        returns=mc_returns([r['reward'] for r in rows])
        assert all(abs(r['returns']-v)<1e-9 and abs(r['advantage']-(v-r['value']))<1e-9 for r,v in zip(rows,returns))
        refs.append(dict(raw=raw_path.relative_to(source_out).as_posix(),raw_sha256=sha(raw_path),experience_sha256=sha(path)))
    flat=[r for rows in episodes for r in rows];adv=tensor([r['advantage'] for r in flat]);scale=adv.std(unbiased=False).clamp_min(.05)
    normalized=(adv-adv.mean())/scale;params=list(model.motion.parameters());grads={k:[] for k in ('mc','immediate_exact','future_residual','entropy')}
    replay_error=0.;pred_error=[];regrets=[];ties=0;cursor=0
    for rows in episodes:
        dist=model.motion(tensor([r['motion_state'] for r in rows]),tensor([r['motion_features'] for r in rows]),tensor([r['motion_mask'] for r in rows],torch.bool))
        log=dist.log_prob(torch.tensor([r['motion_action'] for r in rows]));old=tensor([r['motion_log'] for r in rows])
        replay_error=max(replay_error,float((log-old).abs().max().detach()));assert replay_error<.002
        q=tensor([[feature[74] for feature in r['motion_features']] for r in rows]);reward=tensor([r['reward'] for r in rows])
        chosen=q.gather(1,torch.tensor([r['motion_action'] for r in rows]).unsqueeze(1)).squeeze(1)
        error=(chosen-reward).abs();pred_error+=error.tolist()
        if method=='joint':assert float(error.max())<1e-7
        a=normalized[cursor:cursor+len(rows)];cursor+=len(rows)
        loss=dict(mc=-(log*a).mean(),immediate_exact=-(dist.probs*q).sum(-1).mean()/scale,
            future_residual=-(log*(a-reward/scale)).mean(),entropy=-.001*dist.entropy().mean())
        for name in loss:
            values=torch.autograd.grad(loss[name],params,retain_graph=name!='entropy')
            grads[name].append(torch.cat([v.flatten() for v in values]).detach().numpy().astype(float))
        for r in rows:
            valid=[f[74] for f,m in zip(r['motion_features'],r['motion_mask']) if m]
            k=len(valid);p=r['motion_probs'][:k]
            regrets.append((max(valid)-sum(x*y for x,y in zip(valid,p)))*20);ties+=max(valid)-min(valid)<1e-10
    vectors=np.asarray(grads['mc']);mean=vectors.mean(0);immediate=np.asarray(grads['immediate_exact']).mean(0)
    residual=np.asarray(grads['future_residual']).mean(0);entropy=np.asarray(grads['entropy']).mean(0)
    leave=[cosine(v,(vectors.sum(0)-v)/7,np) for v in vectors]
    return dict(spec=spec,diagnostic_contract=contract()[0],parent_contract=trial.contract()[0],source=refs,checkpoint_sha256=cp_sha,
        episodes=8,steps=len(flat),max_sample_log_error=replay_error,mc_advantage_std=float(scale),
        candidate_short_reward_tie_fraction=ties/len(flat),feature_reward_prediction_mae=statistics.mean(pred_error),
        feature_reward_prediction_max=max(pred_error),expected_short_reward_regret_tasks=statistics.mean(regrets),
        motion_gradient=dict(mc_norm=float(np.linalg.norm(mean)),immediate_model_norm=float(np.linalg.norm(immediate)),future_residual_norm=float(np.linalg.norm(residual)),
            entropy_norm=float(np.linalg.norm(entropy)),mc_vs_immediate_cosine=cosine(mean,immediate,np),
            immediate_to_mc_norm=float(np.linalg.norm(immediate)/max(np.linalg.norm(mean),1e-30)),
            entropy_to_mc_norm=float(np.linalg.norm(entropy)/max(np.linalg.norm(mean),1e-30)),
            episode_coherence=float(np.linalg.norm(mean)/max(np.sqrt(np.mean(np.sum(vectors*vectors,1))),1e-30)),
            leave_one_episode_out_cosines=leave),
        boundary='Immediate model gradient is exact for the joint plan-conditioned known one-step reward; decomposed source-default features are a surrogate. Different short and full horizons do not imply MC is incorrect.',
        cpu=sorted(os.sched_getaffinity(0)))


def worker(payload):
    task_string,spec=payload;task=Path(task_string);out,val=locations(task);path=case_path(out,spec)
    if path.exists():
        d=read(path);assert d['spec']==spec and d['diagnostic_contract']==contract()[0];return dict(cpu=d['cpu'])
    progress=val/'progress'/(identity(spec)+'.json')
    atomic(progress,dict(stage='running',time=time.time(),cpu=sorted(os.sched_getaffinity(0))))
    d=episode_case(spec,task) if spec['kind']=='episode' else credit_case(spec,task)
    raw_write(path,d);atomic(progress,dict(stage='completed',time=time.time(),cpu=d['cpu']));return dict(cpu=d['cpu'])


def validate(task):
    trial.core.configure();os.environ['QUEUED_TORCH_TESTS']='1';stream=io.StringIO()
    names=['tests.test_queued_service_conditioning','tests.test_queued_service_shared_budget','tests.test_queued_service_diagnostics',
        'tests.test_queued_service_learning','tests.test_queued_service','tests.test_queued_service_business','tests.test_queued_service_business_refinement']
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names));_,val=locations(task)
    atomic(val/'tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,skipped=len(result.skipped),
        diagnostic_contract=contract()[0],parent_contract=trial.contract()[0],log=stream.getvalue(),independent_review=False,time=now()))
    print(stream.getvalue(),flush=True);return result.wasSuccessful()


def run(task):
    import fcntl
    trial.core.configure();out,val=locations(task);val.mkdir(parents=True,exist_ok=True)
    lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest,files=contract();assert trial.contract()[0]=='15ea668d0efc613157c85b6b761b1741c7e35f4a371f918a5785586210c61257'
    tests=read(val/'tests.json');assert tests['status']=='PASS' and tests['skipped']==0 and tests['diagnostic_contract']==digest
    atomic(out/'protocol.json',dict(diagnostic_contract=digest,parent_contract=trial.contract()[0],source_files=files,jobs=jobs(),
        times=[40,75,100,150],no_training=True,opportunity='Equal-weight three sampled plans; offline fixed-future, finite candidates and continuation; not a global policy bound'))
    protected=val/'protected_before.json'
    if not protected.exists():
        previous={}
        for root in (task/'results'/trial.core.TASK,task/'output'/trial.core.TASK,task/'saved_models'/trial.core.TASK):
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and not p.is_relative_to(out) and not p.is_relative_to(val) and p.suffix not in ('.lock','.tmp'):
                        previous[p.relative_to(task).as_posix()]=sha(p)
        atomic(protected,previous)
    for method in trial.core.METHODS:worker((str(task),dict(kind='episode',method=method,block=0,workload=48,mode='native',smoke=True)))
    atomic(val/'smoke.json',dict(status='PASS_TWO_EXACT_COMPLETE_REPLAYS',diagnostic_contract=digest,time=now()))
    cpus=core_order();queue=jobs();pending={};done=0;used=set();peak=0;last=0.
    atomic(val/'launch.json',dict(pid=os.getpid(),source=str(REPO_ROOT),diagnostic_contract=digest,started=now(),jobs=276,cpus=cpus,no_training=True))
    try:
        with ProcessPoolExecutor(max_workers=len(cpus),mp_context=multiprocessing.get_context('fork'),initializer=trial.core.worker_init,initargs=(cpus,)) as pool:
            while queue or pending:
                while queue and len(pending)<len(cpus):
                    s=queue.pop(0);pending[pool.submit(worker,(str(task),s))]=s
                finished,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for f in finished:
                    d=f.result();pending.pop(f);done+=1;used.update(d['cpu'])
                if time.time()-last>5 or not pending:
                    active=[read(p) for p in (val/'progress').glob('*.json')];count=sum(d['stage']=='running' for d in active);peak=max(peak,count)
                    record=dict(status='RUNNING',updated=now(),diagnostic_contract=digest,completed=done,total=276,active=count,submitted=len(pending),queued=len(queue),used_cpus=sorted(used),peak_active=peak)
                    atomic(val/'status.json',record)
                    with (val/'resources.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
                    last=time.time()
        analyze(task)
        old=read(protected);changed=[name for name,h in old.items() if not (task/name).exists() or sha(task/name)!=h];assert not changed,changed
        final=dict(status='PASS_FROZEN_CONDITIONING_DIAGNOSTIC',diagnostic_contract=digest,parent_contract=trial.contract()[0],completed=now(),jobs=276,
            used_cpus=sorted(used),peak_active=peak,protected_files=len(old),changed_old_files=[],no_training=True)
        atomic(val/'completion.json',final);atomic(val/'status.json',final)
    except BaseException:
        atomic(val/'status.json',dict(status='FAILED',diagnostic_contract=digest,error=traceback.format_exc(),updated=now()));raise


def analyze(task):
    out,val=locations(task);cases=[read(case_path(out,s)) for s in jobs()];digest=contract()[0]
    assert len(cases)==276 and all(d['spec']==s and d['diagnostic_contract']==digest for d,s in zip(cases,jobs()))
    episodes=[d for d in cases if d['spec']['kind']=='episode'];credit=[d for d in cases if d['spec']['kind']=='credit'];groups=defaultdict(list)
    for d in episodes:
        c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0) and d['rows'][-1]['active']==0
        assert sum(r['success'] for r in d['rows'])==c.get('success',0) and d['audit']['status']=='PASS'
        s=d['spec'];groups[(s['method'],s['block'],s['mode'])].append(d)
        if s['mode']=='native':assert d['replay_steps_exact']==len(d['rows'])
    summary=[]
    for (m,b,mode),ds in sorted(groups.items()):
        g=sum(d['summary']['counts']['generated'] for d in ds);a=sum(d['summary']['counts'].get('admitted',0) for d in ds);s=sum(d['summary']['counts'].get('success',0) for d in ds)
        summary.append(dict(method=m,block=b,mode=mode,episodes=len(ds),generated=g,admitted=a,success=s,dsr=s/g,admission=a/g,conditional=s/max(a,1)))
    probes=[dict(method=d['spec']['method'],block=d['spec']['block'],workload=d['spec']['workload'],**p) for d in episodes for p in d['probes']];assert len(probes)==192
    value=[]
    for method in trial.core.METHODS:
        target=[];prediction=[];phases=[]
        for d in episodes:
            if d['spec']['method']==method and d['spec']['mode']=='native':
                for r in d['values']:target.append(r['target']);prediction.append(r['prediction']);phases.append((d['spec']['block'],r['time']))
        value.append(dict(method=method,**value_fit_by_phase(target,prediction,phases)))
    result=dict(status='PASS_FROZEN_DIAGNOSTIC_RAW_AUDIT',diagnostic_contract=digest,parent_contract=trial.contract()[0],completed=now(),
        summary=summary,probes=probes,credit=credit,value=value,
        raw_sha256={case_path(out,s).relative_to(out).as_posix():sha(case_path(out,s)) for s in jobs()},
        no_training=True,independent_confirmation=False)
    atomic(out/'analysis.json',result);report(task,result);return result


def report(task,data):
    import csv
    out,val=locations(task);deliver=task/'output'/trial.core.TASK/'deliverables';deliver.mkdir(parents=True,exist_ok=True)
    with (deliver/'queued_conditioning_summary.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(data['summary'][0]));writer.writeheader();writer.writerows(data['summary'])
    lines=['# 当前分配前缀条件化为何未稳定胜出','',f"完成：{data['completed']}；冻结15ea668d检查点，不训练、不调负载。",'',
        '## 对照实际检验什么','',
        'joint与decomposed都共同学习未来团队回报，都看见历史真实计算/传输积压。当前差别仅为同一宏时隙的新分配是否立即进入运动输入。后者是计划无关的因子化联合策略，不是没有队列反馈的独立控制器。',
        '', 'joint以post摘要替换pre摘要，没有保留两个摘要的显式旁路。新接纳未分配任务计数在post归零，因此不是严格的信息追加结构；不能由理论最优联合策略包含分解策略，直接推断这两个有限特征网络必然具有嵌套关系。','',
        '## 冻结输入和规划替换','', '| 方法 | 解码 | DSR | 相对native |', '|---|---|---:|---:|']
    for method in trial.core.METHODS:
        baseline=statistics.mean(r['dsr'] for r in data['summary'] if r['method']==method and r['mode']=='native')
        for mode in MODES:
            dsr=statistics.mean(r['dsr'] for r in data['summary'] if r['method']==method and r['mode']==mode)
            lines.append(f'| {method} | {mode} | {100*dsr:.3f}% | {100*(dsr-baseline):+.3f}个百分点 |')
    lines+=['','swap仅在冻结权重下交换pre/post运动输入，分配采样保持本算法；包含输入分布变化，不等于重新训练的消融。model_actual/model_source均采用当前已知任务的两秒模型，真实执行均为采样分配；两者只差预测时是否使用当前实际分配。没有未来到达进入这些控制器。','',
        '## 条件信息的局部机会值','', '| 状态动作尺度 | 状态 | 三计划有条件收益 | 平均机会（任务） | 极端计划有条件收益 |', '|---|---:|---:|---:|---:|']
    for scale in ('one_step','six_steps'):
        qs=[p['payoff'][scale] for p in data['probes']]
        lines.append(f"| {scale} | {len(qs)} | {sum(q['on_policy']['information_gain']>1e-9 for q in qs)} | {statistics.mean(q['on_policy']['information_gain'] for q in qs):.6f} | {sum(q['extreme']['information_gain']>1e-9 for q in qs)} |")
    lines+=['','机会值=每种计划各选最佳候选的平均完成数−所有计划共用一条最佳候选的平均完成数。三个计划为实际样本加两个额外同策略样本，重复样本保留权重；极端计划另列。固定未来5批仅结算预先构造的动作，并排空；这是有限候选、固定继续策略、有限未来样本的离线诊断，不是长期策略收益上界，记录可能重复。','',
        '## 价值是否区分同一时刻的状态','', '| 方法 | MC总体EV | 同时间去均值EV | MC RMSE | 跨回合时间基线RMSE |', '|---|---:|---:|---:|---:|']
    for r in data['value']:lines.append(f"| {r['method']} | {r['overall_ev']:.4f} | {r['within_phase_ev']:.4f} | {r['rmse']:.3f} | {r['other_episode_time_baseline_rmse']:.3f} |")
    lines+=['','native全轨迹原动作/计数精确重现后，才用相同轨迹剩余MC作为目标。去均值按训练种子和同一时间对8个复用开发工作负载分组。时间基线使用其余回合的未来结果，只是离线诊断，未进入策略。','',
        '## 运动学习信号','', '| 方法 | 批次 | 一秒完成数候选全平比例 | MC梯度跨回合一致度 | 即时模型梯度/MC梯度范数 | MC与即时方向余弦 |', '|---|---:|---:|---:|---:|---:|']
    for method in trial.core.METHODS:
        ds=[d for d in data['credit'] if d['spec']['method']==method]
        cs=[d['motion_gradient']['mc_vs_immediate_cosine'] for d in ds if d['motion_gradient']['mc_vs_immediate_cosine'] is not None]
        lines.append(f"| {method} | {len(ds)} | {statistics.mean(d['candidate_short_reward_tie_fraction'] for d in ds):.4f} | {statistics.mean(d['motion_gradient']['episode_coherence'] for d in ds):.4f} | {statistics.mean(d['motion_gradient']['immediate_to_mc_norm'] for d in ds):.4f} | {statistics.mean(cs) if cs else 0.:.4f} |")
    lines+=['','joint的候选预测即时奖励与实际采样即时奖励逐项核对。decomposed默认来源预测只是替代模型，不能视为当前实际分配下的精确奖励。MC与即时梯度的目标窗口不同，方向不一致不证明MC错误；没有优化器或训练被创建。','',
        '- [闭环逐格CSV](queued_conditioning_summary.csv)',
        f'- [完整分析]({os.path.relpath(out/"analysis.json",deliver).replace(os.sep,"/")})',
        f'- [冻结协议]({os.path.relpath(out/"protocol.json",deliver).replace(os.sep,"/")})',
        f'- [完成验收]({os.path.relpath(val/"completion.json",deliver).replace(os.sep,"/")})',
        f'- [测试]({os.path.relpath(val/"tests.json",deliver).replace(os.sep,"/")})','',
        '原结果字段名保留，不把本轮重新命名成更强或更弱方法。3个初始化种子、8复用工作负载，状态/请求不作独立训练重复，显式自审。',
        '',f'诊断合同：`{contract()[0]}`。']
    (deliver/'queued_conditioning_diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('package','validate','run','analyze'))
    p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--archive',type=Path,default=REPO_ROOT/'.tmp/queued_conditioning/release.zip')
    args=p.parse_args();task=args.task.resolve()
    if args.command=='package':
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name in SOURCE_FILES:z.write(REPO_ROOT/name,name)
        print(json.dumps(dict(diagnostic_contract=contract()[0],archive_sha256=sha(args.archive),files=len(SOURCE_FILES))))
    elif args.command=='validate':raise SystemExit(0 if validate(task) else 1)
    elif args.command=='run':run(task)
    else:analyze(task)


if __name__=='__main__':main()
