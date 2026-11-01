"""Frozen queue-policy diagnostics. No optimizer, training or parameter writes."""
import argparse
from collections import Counter,defaultdict
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
from environment.queued_service import Engine,Protocol,distance
from utils.queued_service_controllers import Controller,assignments
from utils.queued_service_learning import LearningAdapter,VirtualQueue,finite_gae
from utils.queued_service_diagnostics import commit_sequence,probability_metrics,payoff_metrics,mc_returns,explained_variance
from utils.queued_service_business_refinement import RecoveryScenario
from scripts.analysis.queued_service_validation import START,atomic,core_order,sha,now
from scripts.analysis.queued_service_business import read,raw_write
from scripts.analysis.queued_service_coupling import cooperative_plan
from scripts.analysis import queued_service_joint_short as parent

DIAGNOSTIC='diagnostic_20261031'
MODES=('factual','uniform_motion','greedy_motion','queue_alloc','history_motion','model_motion')
SOURCE_FILES=parent.SOURCES+('utils/queued_service_diagnostics.py','scripts/analysis/diagnose_queued_service_joint.py',
    'tests/test_queued_service_diagnostics.py')


def contract():
    files={name:sha(REPO_ROOT/name) for name in SOURCE_FILES}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files


def locations(task):
    out,val=parent.locations(task);suffix=Path(DIAGNOSTIC)/contract()[0][:12]
    return out/suffix,val/suffix


def jobs():
    episodes=[dict(kind='episode',method=m,block=b,workload=w,mode=mode)
        for m,b,w,mode in itertools.product(parent.METHODS,range(3),parent.EVALUATION,MODES)]
    episodes += [dict(kind='episode',method='history_queue',block=0,workload=w,mode='queue_history') for w in parent.EVALUATION]
    credit=[dict(kind='credit',method=m,block=b,round=r) for m,b,r in itertools.product(parent.METHODS,range(3),range(1,7))]
    return episodes+credit


def identity(spec):return '_'.join(f'{k}-{spec[k]}' for k in sorted(spec))


def case_path(out,spec):return out/'cases'/(identity(spec)+'.json.gz')


def load_model(task,method,block,checkpoint,torch):
    from marl_models.queued_service_joint import JointPolicy
    out,_=parent.locations(task);path=parent.model_path(out,method,block,checkpoint)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['source_contract']==parent.contract()[0] and saved['checkpoint']==checkpoint
    assert saved['method']==method and saved['block']==block
    reference=parent.training_jobs(method,block,checkpoint+1)[0] if checkpoint<6 else parent.evaluation_jobs(method,block,6)[0]
    raw=parent.result_path(out,reference);manifest=read(out/'analysis.json')['raw_sha256']
    assert sha(raw)==manifest[raw.relative_to(out).as_posix()]
    assert read(raw)['checkpoint_sha256']==sha(path)
    model=JointPolicy(method);model.load_state_dict(saved['model']);model.eval()
    return model,sha(path)


def source_episode(spec,task):
    original=dict(kind='eval',method=spec['method'],block=spec['block'],checkpoint=6,profile='burst',workload=spec['workload'])
    path=parent.result_path(parent.locations(task)[0],original);data=read(path)
    assert data['source_contract']==parent.contract()[0] and data['job']==original
    digest=sha(path);out,_=parent.locations(task)
    assert digest==read(out/'analysis.json')['raw_sha256'][path.relative_to(out).as_posix()]
    return original,data,digest


def motion_distribution(model,bank,torch):
    from marl_models.queued_service_joint import tensor
    with torch.no_grad():
        dist=model.motion(tensor([bank['state']]),tensor([bank['features']]),tensor([bank['mask']],torch.bool))
    return dist.probs[0]


def choose(model,obs,adapter,seed,mode,torch):
    row,learned,motion=model.act(obs,adapter,seed)
    plan=assignments(obs,'queue') if mode=='queue_alloc' else learned
    if mode=='queue_alloc':
        prefix=VirtualQueue(obs)
        for uid,node in plan.items():prefix.assign(uid,node)
        bank=adapter.motion(prefix.obs if model.method=='joint' else obs)
        probabilities=motion_distribution(model,bank,torch)
    else:probabilities=torch.tensor(row['motion_probs'])
    generator=torch.Generator().manual_seed(seed)
    for token in row['allocations']:torch.multinomial(torch.tensor(token['probs']),1,generator=generator)
    if mode=='uniform_motion':probabilities=torch.tensor(row['motion_mask'],dtype=torch.float32);probabilities/=probabilities.sum()
    index=int(torch.multinomial(probabilities,1,generator=generator))
    if mode=='factual':assert index==row['motion_action']
    if mode=='greedy_motion':index=int(probabilities.argmax())
    if mode=='history_motion':index=0
    if mode=='model_motion':
        best=None;zero=[(0.,0.)]*5
        for i,candidate in enumerate(adapter.plans):
            forecast=Engine.from_observation(obs);forecast.assign(plan);forecast.advance(candidate)
            forecast.admit();forecast.assign(assignments(forecast.observe(),'queue'));forecast.advance(zero)
            pending=sum(z.remaining_cycles/forecast.p.capacity[z.target if z.target is not None else 5] for z in forecast.tasks.values())
            score=(forecast.counts['success'],-forecast.counts['expired'],-pending,adapter.history._history_score(obs,candidate))
            if best is None or score>best[0]:best=(score,i)
        index=best[1]
    return row,plan,adapter.plans[index]


def payoff(engine,tape,plans,bank):
    zero=[(0.,0.)]*5;output={};base=engine.counts['success'];initial=len(engine.tasks)
    for mode in ('one_step','six_steps'):
        values={};denominators=[]
        for name in ('local','cooperative'):
            counts=[]
            for first in bank:
                sequence=[first]+[zero]*5 if mode=='one_step' else commit_sequence(engine,first)
                f=copy.deepcopy(engine);offset=0;generated=initial
                while offset<6 or f.tasks:
                    assert offset<=12
                    if offset:
                        arrivals=tape.slot(int(engine.t)+offset) if offset<6 else ()
                        generated+=len(arrivals);f.add(arrivals);new=f.admit();route={uid:f.tasks[uid].source for uid in new}
                    else:route=plans[name]
                    f.assign(route);f.advance(sequence[offset] if offset<6 else zero);offset+=1
                f.audit();counts.append(f.counts['success']-base);denominators.append(generated)
            values[name]=counts
        assert len(set(denominators))==1
        output[mode]=dict(values=values,at_risk=denominators[0],**payoff_metrics(values['local'],values['cooperative']))
    return output


def probe(engine,tape,obs,adapter,actual,models,torch):
    import numpy as np
    local={x['uid']:x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    plans=dict(local=local,cooperative=cooperative_plan(obs),queue=assignments(obs,'queue'),factual=actual)
    banks={};prefixes={}
    for name,plan in plans.items():
        prefix=VirtualQueue(obs)
        for uid,node in plan.items():prefix.assign(uid,node)
        prefixes[name]=prefix.obs;banks[name]=adapter.motion(prefix.obs)
    banks['no_prefix']=adapter.motion(obs)
    k=len(adapter.plans);changed=sum(plans['local'][uid]!=plans['cooperative'][uid] for uid in local)
    probabilities={};metrics={}
    for cp,model in models.items():
        dist={name:motion_distribution(model,banks[name] if model.method=='joint' else banks['no_prefix'],torch)[:k].tolist() for name in banks}
        probabilities[str(cp)]=dist
        metrics[str(cp)]=dict(local_vs_cooperative=probability_metrics(dist['local'],dist['cooperative']),
            factual_vs_no_prefix=probability_metrics(dist['factual'],dist['no_prefix']),
            factual_vs_uniform=probability_metrics(dist['factual'],[1/k]*k))
    difference=np.asarray(banks['local']['features'][:k])-np.asarray(banks['cooperative']['features'][:k])
    centered=difference-difference.mean(0,keepdims=True)
    saturation=Counter();p=obs['protocol'];coop=prefixes['cooperative'];radio=Engine(p,obs['positions'],obs['ues'])
    for action in adapter.plans:
        mid=[(a[0]+d[0]/2,a[1]+d[1]/2) for a,d in zip(obs['positions'],action)]
        for x in coop['tasks']:
            a,b=x['source'],x['target']
            if a is None or b is None or a==b or x['remaining_transfer']<=0:continue
            if b!=5 and distance(mid[a],mid[b])>p.inter_range:saturation['disconnected']+=1;continue
            end=p.mbs if b==5 else (*mid[b],p.altitude);kind='backhaul' if b==5 else 'inter'
            saturation['connected']+=1
            saturation['cap_hit']+=radio.rate(kind,(*mid[a],p.altitude),end)>=x['remaining_transfer']
    return dict(time=obs['time'],candidates=k,new_tasks=len(local),changed_assignments=changed,eligible=changed>0 and k>1,
        probability=probabilities,metrics=metrics,
        initial_vs_final=probability_metrics(probabilities['0']['factual'],probabilities['6']['factual']),
        feature_change_max=float(np.abs(difference).max()),candidate_centered_feature_change_max=float(np.abs(centered).max()),
        progress_saturation=dict(saturation),payoff=payoff(engine,tape,plans,adapter.plans),
        offline_future_used_for_fixed_action_evaluation_only=True)


def episode_case(spec,task):
    torch=parent.configure();out,val=locations(task);digest=contract()[0]
    original,expected,raw_sha=source_episode(spec,task)
    model=None;models={};cp_sha=None;controller=None
    if spec['method'] in parent.METHODS:
        model,cp_sha=load_model(task,spec['method'],spec['block'],6,torch);adapter=LearningAdapter()
        if spec['mode']=='factual' and not spec.get('smoke'):
            models={0:load_model(task,spec['method'],spec['block'],0,torch)[0],6:model}
    else:controller=Controller('history_queue')
    assert cp_sha==expected['checkpoint_sha256']
    tape=RecoveryScenario(spec['workload'],'burst','normal',.6);engine=Engine(Protocol(),START,tape.ue_positions)
    assert tape.arrival_fingerprint==expected['arrival_fingerprint']
    rows=[];probes=[];decisions=[];start=time.perf_counter();verified=0
    def heartbeat(stage,step):atomic(val/'progress'/(identity(spec)+'.json'),dict(stage=stage,time=time.time(),pid=os.getpid(),cpu=sorted(os.sched_getaffinity(0)),step=step))
    while engine.t<200 or engine.tasks:
        t=int(engine.t);assert t<=204;engine.add(tape.slot(t));engine.admit();obs=engine.observe();began=time.perf_counter()
        if model:sample,plan,motion=choose(model,obs,adapter,parent.action_seed(original,t),spec['mode'],torch)
        else:plan,motion=controller.act(obs)
        decisions.append(time.perf_counter()-began)
        if models and t in (40,75,100,150):probes.append(probe(engine,tape,obs,adapter,plan,models,torch))
        engine.assign(plan);row=engine.advance(motion)
        if spec['mode'] in ('factual','queue_history'):
            old=expected['rows'][t]
            assert all(row[key]==old[key] for key in ('generated','completed','expired','admitted','active','success')),(spec,t)
            assert plan=={int(k):v for k,v in old['assignment'].items()}
            assert all(abs(a-b)<1e-9 for d,e in zip(motion,old['motion']) for a,b in zip(d,e))
            verified+=1
        rows.append(row)
        if t%20==0:heartbeat('running',t)
    audit=engine.audit();summary=engine.summary();terminal=[];route=Counter()
    for z in engine.finished.values():
        fate='success' if z.stage=='completed' else z.expired_stage
        terminal.append([z.uid,fate,z.source,z.target])
        if z.target is not None:route['mbs' if z.target==5 else 'local' if z.target==z.source else 'cooperative']+=1
    summary['cpu_wait_seconds_total']=sum(z.time_in['compute_wait'] for z in engine.finished.values())
    return dict(spec=spec,diagnostic_contract=digest,parent_contract=parent.contract()[0],parent_raw_sha256=raw_sha,
        checkpoint_sha256=cp_sha,replay_steps_exact=verified,summary=summary,audit=audit,rows=rows,probes=probes,
        arrival_fingerprint=tape.arrival_fingerprint,
        terminal=terminal,route_class=dict(route),decision_seconds=statistics.mean(decisions),
        decision_over_one_second=sum(x>1 for x in decisions),seconds=time.perf_counter()-start,cpu=sorted(os.sched_getaffinity(0)))


def cosine(a,b,np):
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denominator) if denominator>1e-20 else None


def credit_case(spec,task):
    torch=parent.configure();import numpy as np
    from marl_models.queued_service_joint import tensor
    out,_=parent.locations(task);round_=spec['round'];method=spec['method'];block=spec['block']
    model,old_sha=load_model(task,method,block,round_-1,torch);after,new_sha=load_model(task,method,block,round_,torch)
    episodes=[];source=[];raws=[]
    for job in parent.training_jobs(method,block,round_):
        raw_path=parent.result_path(out,job);data=read(raw_path)
        assert data['job']==job and data['source_contract']==parent.contract()[0]
        assert sha(raw_path)==read(out/'analysis.json')['raw_sha256'][raw_path.relative_to(out).as_posix()]
        path=task/'.tmp'/parent.TASK/parent.REVISION/'experience'/(parent.job_id(job)+'.pkl.gz')
        assert sha(path)==data['experience_sha256'] and data['checkpoint_sha256']==old_sha
        rows=pickle.loads(gzip.decompress(path.read_bytes()));episodes.append(rows);raws.append(data)
        source.append(dict(raw=str(raw_path.relative_to(task)),raw_sha256=sha(raw_path),experience_sha256=sha(path)))
        adv,ret=finite_gae([r['reward'] for r in rows],[r['value'] for r in rows])
        assert max(abs(a-r['advantage']) for a,r in zip(adv,rows))<1e-9
        assert max(abs(a-r['returns']) for a,r in zip(ret,rows))<1e-9
        assert math.isclose(sum(r['reward'] for r in rows)*20,data['summary']['counts']['success'],abs_tol=1e-7)
    flat=[r for rows in episodes for r in rows];mc=[mc_returns([r['reward'] for r in rows]) for rows in episodes]
    actual=tensor([r['advantage'] for r in flat]);mc_adv=tensor([v-r['value'] for rows,vs in zip(episodes,mc) for r,v in zip(rows,vs)])
    normalized={name:(a-a.mean())/a.std(unbiased=False).clamp_min(.05) for name,a in [('gae',actual),('mc',mc_adv)]}
    total=len(flat);valid_count=sum(bool(r['allocations']) for r in flat)
    parameters=list(model.allocation.parameters())+list(model.motion.parameters())
    split=sum(p.numel() for p in model.allocation.parameters());gradients={name:[] for name in ('gae','mc','entropy')}
    predictions=[];post_values=[];max_log_error=0.;motion_tv=[];allocation_tv=[];prior_tv=[];clip_a=[];clip_m=[];cursor=0
    for rows in episodes:
        z=model.replay(rows);valid=z['allocation_count']>0
        max_log_error=max(max_log_error,float((z['allocation_log']-tensor([r['allocation_log'] for r in rows])).abs().max().detach()),
            float((z['motion_log']-tensor([r['motion_log'] for r in rows])).abs().max().detach()))
        assert max_log_error<.002
        losses={}
        for name in ('gae','mc'):
            a=normalized[name][cursor:cursor+len(rows)]
            losses[name]=-(z['allocation_log'][valid]*a[valid]).sum()/(valid_count/8)-(z['motion_log']*a).sum()/(total/8)
        losses['entropy']=-.001*(z['allocation_entropy'].mean()+z['motion_entropy'].mean())
        for name in ('gae','mc','entropy'):
            pieces=torch.autograd.grad(losses[name],parameters,allow_unused=True,retain_graph=name!='entropy')
            vector=torch.cat([g.flatten() if g is not None else torch.zeros_like(p).flatten() for g,p in zip(pieces,parameters)]).detach().numpy().astype(float)
            gradients[name].append(vector)
        with torch.no_grad():
            new=after.replay(rows)
            predictions+=z['value'].detach().tolist();post_values+=new['value'].tolist()
            clip_a += ((new['allocation_log']-z['allocation_log']).exp()[valid].sub(1).abs()>.2).tolist()
            clip_m += ((new['motion_log']-z['motion_log']).exp().sub(1).abs()>.2).tolist()
            for row in rows:
                old=torch.tensor(row['motion_probs']);mask=torch.tensor(row['motion_mask'])
                p=after.motion(tensor([row['motion_state']]),tensor([row['motion_features']]),tensor([row['motion_mask']],torch.bool)).probs[0]
                motion_tv.append(float((p-old).abs().sum()/2))
                for token in row['allocations']:
                    q=after.allocation(tensor([token['state']]),tensor([token['features']]),tensor([token['mask']],torch.bool),tensor([token['prior']])).probs[0]
                    prior=torch.softmax(torch.tensor(token['prior']).masked_fill(~torch.tensor(token['mask']),-1e9),0)
                    allocation_tv.append(float((q-torch.tensor(token['probs'])).abs().sum()/2));prior_tv.append(float((q-prior).abs().sum()/2))
        cursor+=len(rows)
    metrics={}
    for branch,sl in [('allocation',slice(None,split)),('motion',slice(split,None))]:
        vectors=np.asarray(gradients['gae'])[:,sl];mean=vectors.mean(0);mc_mean=np.asarray(gradients['mc'])[:,sl].mean(0)
        entropy=np.asarray(gradients['entropy'])[:,sl].mean(0)
        leave=[cosine(v,(vectors.sum(0)-v)/7,np) for v in vectors]
        metrics[branch]=dict(reward_gradient_norm=float(np.linalg.norm(mean)),entropy_gradient_norm=float(np.linalg.norm(entropy)),
            entropy_to_reward_norm=float(np.linalg.norm(entropy)/max(np.linalg.norm(mean),1e-30)),
            episode_gradient_coherence=float(np.linalg.norm(mean)/max(np.sqrt(np.mean(np.sum(vectors*vectors,1))),1e-30)),
            leave_one_episode_out_cosines=leave,gae_vs_mc_gradient_cosine=cosine(mean,mc_mean,np))
    gae_target=[r['returns'] for r in flat];mc_target=[v for vs in mc for v in vs]
    time_baseline=[statistics.mean(mc[j][t] if t<len(mc[j]) else 0 for j in range(8) if j!=i) for i,rows in enumerate(episodes) for t in range(len(rows))]
    def rmse(a,b):return math.sqrt(statistics.mean((x-y)**2 for x,y in zip(a,b)))
    return dict(spec=spec,diagnostic_contract=contract()[0],parent_contract=parent.contract()[0],source=source,
        checkpoints=dict(before=old_sha,after=new_sha),episodes=8,steps=total,max_sample_replay_log_error=max_log_error,
        advantage=dict(mean=float(actual.mean()),std=float(actual.std(unbiased=False)),positive_fraction=float((actual>0).float().mean())),
        critic=dict(gae_ev_before=explained_variance(gae_target,predictions),gae_ev_after=explained_variance(gae_target,post_values),
            mc_ev_before=explained_variance(mc_target,predictions),mc_ev_after=explained_variance(mc_target,post_values),
            gae_rmse_before=rmse(gae_target,predictions),gae_rmse_after=rmse(gae_target,post_values),
            mc_rmse_before=rmse(mc_target,predictions),mc_rmse_after=rmse(mc_target,post_values),
            independent_episode_time_baseline_mc_rmse=rmse(mc_target,time_baseline),
            prediction_mean=statistics.mean(predictions),gae_target_mean=statistics.mean(gae_target),mc_target_mean=statistics.mean(mc_target)),
        gradient=metrics,policy_change=dict(motion_tv_mean=statistics.mean(motion_tv),allocation_tv_mean=statistics.mean(allocation_tv),
            allocation_vs_prior_tv_mean=statistics.mean(prior_tv),post_update_allocation_clip_fraction=statistics.mean(clip_a),
            post_update_motion_clip_fraction=statistics.mean(clip_m)),cpu=sorted(os.sched_getaffinity(0)))


def worker(payload):
    task_string,spec=payload;task=Path(task_string);out,val=locations(task);path=case_path(out,spec)
    if path.exists():
        previous=read(path);assert previous['diagnostic_contract']==contract()[0] and previous['spec']==spec
        return dict(spec=spec,cpu=previous['cpu'])
    progress=val/'progress'/(identity(spec)+'.json')
    atomic(progress,dict(stage='running',time=time.time(),pid=os.getpid(),cpu=sorted(os.sched_getaffinity(0))))
    result=episode_case(spec,task) if spec['kind']=='episode' else credit_case(spec,task)
    raw_write(path,result);atomic(progress,dict(stage='completed',time=time.time(),pid=os.getpid(),cpu=result['cpu']))
    return dict(spec=spec,cpu=result['cpu'])


def validate(task):
    parent.configure();os.environ['QUEUED_TORCH_TESTS']='1';stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_queued_service_diagnostics','tests.test_queued_service_learning',
        'tests.test_queued_service','tests.test_queued_service_business','tests.test_queued_service_business_refinement'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);_,val=locations(task)
    atomic(val/'tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,skipped=len(result.skipped),
        diagnostic_contract=contract()[0],parent_contract=parent.contract()[0],time=now(),log=stream.getvalue(),independent_review=False))
    print(stream.getvalue(),flush=True);return result.wasSuccessful()


def run(task):
    import fcntl
    parent.configure();out,val=locations(task);val.mkdir(parents=True,exist_ok=True)
    lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest,files=contract();assert parent.contract()[0]=='f55f9b56a5144f2de962ac6a42186297ecbb08a20e671ef7ab864d3e2960ebd0'
    tests=read(val/'tests.json');assert tests['status']=='PASS' and tests['skipped']==0 and tests['diagnostic_contract']==digest
    definition=dict(diagnostic_contract=digest,source_files=files,parent_contract=parent.contract()[0],jobs=jobs(),
        fixed_probe_times=[40,75,100,150],no_training=True,independent_confirmation=False,
        future_input='Only fixed-action payoff evaluation uses common future arrivals; policies and known-task model controller never see them')
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==definition
    else:atomic(out/'protocol.json',definition)
    protect=val/'protected_before.json'
    if not protect.exists():
        previous={}
        for root in (task/'results'/parent.TASK,task/'output'/parent.TASK,task/'saved_models'/parent.TASK):
            if root.exists():
                for path in root.rglob('*'):
                    if path.is_file() and not path.is_relative_to(out) and not path.is_relative_to(val) and path.suffix not in ('.tmp','.lock'):
                        previous[str(path.relative_to(task))]=sha(path)
        atomic(protect,previous)
    # Gate all diagnostic interpretation on exact reproduction of real episodes.
    for method in parent.METHODS:
        smoke=dict(kind='episode',method=method,block=0,workload=48,mode='factual',smoke=True)
        worker((str(task),smoke))
    atomic(val/'smoke.json',dict(status='PASS_TWO_FULL_FACTUAL_REPRODUCTIONS',diagnostic_contract=digest,time=now()))
    cpus=core_order();schedule=jobs();assert len(schedule)==332
    atomic(val/'launch.json',dict(status='RUNNING',started=now(),pid=os.getpid(),cpus=cpus,diagnostic_contract=digest,source=str(REPO_ROOT),jobs=len(schedule),no_training=True))
    pending={};queue=list(schedule);completed=0;used=set();peak=0;last=0.
    try:
        with ProcessPoolExecutor(max_workers=len(cpus),mp_context=multiprocessing.get_context('fork'),initializer=parent.worker_init,initargs=(cpus,)) as pool:
            while queue or pending:
                while queue and len(pending)<len(cpus):
                    spec=queue.pop(0);pending[pool.submit(worker,(str(task),spec))]=spec
                finished,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for f in finished:
                    result=f.result();pending.pop(f);completed+=1;used.update(result['cpu'])
                if time.time()-last>5 or not pending:
                    active=[read(p) for p in (val/'progress').glob('*.json') if p.is_file()]
                    active=[r for r in active if r['stage']=='running'];peak=max(peak,len(active))
                    record=dict(status='RUNNING',updated=now(),diagnostic_contract=digest,completed=completed,total=len(schedule),
                        active=len(active),submitted=len(pending),queued=len(queue),used_cpus=sorted(used),peak_active=peak)
                    atomic(val/'status.json',record)
                    with (val/'resources.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
                    last=time.time()
        analyze(task)
        previous=read(protect);changed=[name for name,h in previous.items() if not (task/name).exists() or sha(task/name)!=h]
        assert not changed,changed
        final=dict(status='PASS_COMPLETED_FROZEN_DIAGNOSTICS',diagnostic_contract=digest,parent_contract=parent.contract()[0],
            completed=now(),jobs=332,used_cpus=sorted(used),peak_active=peak,protected_files=len(previous),changed_old_files=changed,no_training=True)
        atomic(val/'completion.json',final);atomic(val/'status.json',final)
    except BaseException:
        atomic(val/'status.json',dict(status='FAILED',error=traceback.format_exc(),diagnostic_contract=digest,updated=now()));raise


def analyze(task):
    out,val=locations(task);cases=[read(case_path(out,s)) for s in jobs()];digest=contract()[0]
    assert all(d['spec']==s and d['diagnostic_contract']==digest for d,s in zip(cases,jobs()))
    episode=[d for d in cases if d['spec']['kind']=='episode'];credit=[d for d in cases if d['spec']['kind']=='credit']
    fingerprints={};hashes={}
    for d in episode:
        c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0) and d['rows'][-1]['active']==0
        workload=d['spec']['workload']
        assert workload not in fingerprints or fingerprints[workload]==d['arrival_fingerprint'];fingerprints[workload]=d['arrival_fingerprint']
        assert sum(r['success'] for r in d['rows'])==c.get('success',0) and d['audit']['status']=='PASS'
        assert len(d['terminal'])==c['generated'] and len({r[0] for r in d['terminal']})==c['generated']
        assert sum(r[1]=='success' for r in d['terminal'])==c.get('success',0)
        if d['spec']['mode'] in ('factual','queue_history'):assert d['replay_steps_exact']==len(d['rows'])
    for spec in jobs():hashes[case_path(out,spec).relative_to(out).as_posix()]=sha(case_path(out,spec))
    groups=defaultdict(list)
    for d in episode:
        s=d['spec'];groups[(s['method'],s['block'],s['mode'])].append(d)
    summary=[]
    for (method,block,mode),ds in sorted(groups.items()):
        g=sum(d['summary']['counts']['generated'] for d in ds);a=sum(d['summary']['counts'].get('admitted',0) for d in ds);success=sum(d['summary']['counts'].get('success',0) for d in ds)
        summary.append(dict(method=method,block=block,mode=mode,episodes=len(ds),generated=g,admitted=a,success=success,
            dsr=success/g,admission=a/g,conditional=success/max(a,1),decision_seconds=statistics.mean(d['decision_seconds'] for d in ds),
            expired_waiting=sum(d['summary']['expiration_stage'].get('waiting',0) for d in ds),
            expired_upload=sum(d['summary']['expiration_stage'].get('upload',0) for d in ds),
            expired_transfer=sum(d['summary']['expiration_stage'].get('transfer',0) for d in ds),
            expired_compute=sum(d['summary']['expiration_stage'].get('compute',0) for d in ds)))
    probes=[dict(method=d['spec']['method'],block=d['spec']['block'],workload=d['spec']['workload'],**p) for d in episode for p in d['probes']]
    assert len(probes)==192
    for p in probes:
        for value in p['payoff'].values():
            assert {k:value[k] for k in ('local_range','coop_range','interaction_range','strict_reversal')}==payoff_metrics(value['values']['local'],value['values']['cooperative'])
    causal=[]
    reference={d['spec']['workload']:dict((r[0],r[1]) for r in d['terminal']) for d in episode if d['spec']['mode']=='queue_history'}
    for d in episode:
        if d['spec']['mode']!='factual':continue
        ref=reference[d['spec']['workload']];terminal=dict((r[0],r[1]) for r in d['terminal']);assert ref.keys()==terminal.keys()
        lost=Counter(terminal[uid] for uid in ref if ref[uid]=='success' and terminal[uid]!='success')
        gained=Counter(ref[uid] for uid in ref if terminal[uid]=='success' and ref[uid]!='success')
        assert sum(gained.values())-sum(lost.values())==sum(v=='success' for v in terminal.values())-sum(v=='success' for v in ref.values())
        causal.append(dict(spec=d['spec'],lost_success_by_terminal_stage=dict(lost),gained_success_from_reference_stage=dict(gained),
            boundary='paired outcome changes, descriptive stages not unique causal attribution'))
    result=dict(status='PASS_DIAGNOSTIC_RAW_AUDIT',diagnostic_contract=digest,parent_contract=parent.contract()[0],completed=now(),
        episode_summary=summary,probes=probes,credit=credit,task_outcome_changes=causal,raw_sha256=hashes,
        factual_episodes=48,probe_states=192,credit_batches=36,training=False,independent_confirmation=False)
    atomic(out/'analysis.json',result);make_report(task,result);return result


def make_report(task,data):
    import csv
    out,val=locations(task);deliver=task/'output'/parent.TASK/'deliverables';deliver.mkdir(parents=True,exist_ok=True)
    rows=data['episode_summary']
    with (deliver/'queued_joint_diagnosis_summary.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 联合短试冻结模型诊断','',f"完成：{data['completed']}。不训练、不更改检查点或负载；这是开发诊断，不是独立确认。",'',
        '## 闭环分支替换','',
        '| 方法 | 解码/替换方式 | 突发DSR | 相对原采样 |','|---|---|---:|---:|']
    for method in parent.METHODS:
        factual=statistics.mean(r['dsr'] for r in rows if r['method']==method and r['mode']=='factual')
        for mode in MODES:
            value=statistics.mean(r['dsr'] for r in rows if r['method']==method and r['mode']==mode)
            lines.append(f'| {method} | {mode} | {100*value:.3f}% | {100*(value-factual):+.3f}个百分点 |')
    lines+=['','模型运动只预测当前已知任务2秒，不读未来任务；诊断解码不替换原采样主结果。队列分配+历史运动仅跑8条确定性轨迹，不充当3次训练重复。','',
        '## 分配对轨迹概率的影响','',
        '| 方法 | 有效状态 | 本地/协作TV均值 | 最大TV | 原运动归一化熵 | 相对均匀运动TV | 0→6检查点TV |','|---|---:|---:|---:|---:|---:|---:|']
    for method in parent.METHODS:
        ps=[p for p in data['probes'] if p['method']==method and p['eligible']]
        tv=[p['metrics']['6']['local_vs_cooperative']['tv'] for p in ps]
        lines.append(f"| {method} | {len(ps)} | {statistics.mean(tv):.6g} | {max(tv):.6g} | {statistics.mean(p['metrics']['6']['factual_vs_uniform']['entropy_fraction'] for p in ps):.6f} | {statistics.mean(p['metrics']['6']['factual_vs_uniform']['tv'] for p in ps):.6g} | {statistics.mean(p['initial_vs_final']['tv'] for p in ps):.6g} |")
    connected=sum(p['progress_saturation'].get('connected',0) for p in data['probes']);capped=sum(p['progress_saturation'].get('cap_hit',0) for p in data['probes'])
    lines+=['',f"协作前缀下，当前进度特征所评估的连通链路×候选中，{capped}/{connected}（{100*capped/max(connected,1):.2f}%）触及min(1,rate/remaining_bits)上限。它使用单链路全池速率，是特征近似，不等于实际共享带宽下都能在1秒传完。",'',
        '## 固定动作完成矩阵','',
        '| 动作尺度 | 状态 | 有候选完成差异 | 有分配×轨迹交互 | 严格优劣反转 |','|---|---:|---:|---:|---:|']
    for mode in ('one_step','six_steps'):
        ps=[p for p in data['probes'] if p['eligible']]
        values=[p['payoff'][mode] for p in ps]
        lines.append(f"| {mode} | {len(values)} | {sum(v['local_range']>0 or v['coop_range']>0 for v in values)} | {sum(v['interaction_range']>0 for v in values)} | {sum(v['strict_reversal'] for v in values)} |")
    lines+=['','这张矩阵用相同未来5批任务评估预先固定动作，再排空；未来输入只用于离线反事实结算，没有进入候选构造或策略。六步保持方向逐段检查安全，悬停始终保留悬停。它不是可部署控制器、平均联合收益或全局上界。','',
        '## 价值和信用分配','',
        '| 方法 | 更新批次 | critic对MC的更新后EV | 对MC的更新后RMSE | 跨回合时间基线RMSE | 分配GAE/MC梯度余弦 | 运动GAE/MC梯度余弦 |','|---|---:|---:|---:|---:|---:|---:|']
    for method in parent.METHODS:
        xs=[x for x in data['credit'] if x['spec']['method']==method]
        mean=lambda f:statistics.mean(f(x) for x in xs)
        lines.append(f"| {method} | {len(xs)} | {mean(lambda x:x['critic']['mc_ev_after']):.4f} | {mean(lambda x:x['critic']['mc_rmse_after']):.3f} | {mean(lambda x:x['critic']['independent_episode_time_baseline_mc_rmse']):.3f} | {mean(lambda x:x['gradient']['allocation']['gae_vs_mc_gradient_cosine']):.3f} | {mean(lambda x:x['gradient']['motion']['gae_vs_mc_gradient_cosine']):.3f} |")
    lines+=['','MC为完整回合成功数/20的未来总和；GAE lambda=.95不是MC目标。critic对MC误差大本身不能判定GAE实现错误，需要同时看其GAE目标、重放误差和梯度方向。跨回合时间基线为其余7个独立输入回合的同一时刻MC均值，只作离线诊断；没有训练替代critic。梯度为更新前冻结模型上的估计，GAE/MC方向不一致不等于替换MC必然提高DSR。','',
        '## 文件和限制','',
        '- [闭环逐格CSV](queued_joint_diagnosis_summary.csv)',
        f'- [完整诊断分析]({os.path.relpath(out/"analysis.json",deliver).replace(os.sep,"/")})',
        f'- [完成记录]({os.path.relpath(val/"completion.json",deliver).replace(os.sep,"/")})',
        f'- [服务器测试]({os.path.relpath(val/"tests.json",deliver).replace(os.sep,"/")})','',
        '原始任务终局阶段对照保存在分析中，只是描述性变化，不能把过期阶段当作唯一因果归因。3个初始化种子、8个复用开发工作负载；状态和请求不充当独立训练重复。显式自审，不冒称独立审查。',
        '',f'父科学源码合同：`{parent.contract()[0]}`；诊断合同：`{contract()[0]}`。']
    (deliver/'queued_joint_diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('package','validate','run','analyze'))
    p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--archive',type=Path,default=REPO_ROOT/'.tmp/queued_joint_diagnosis/release.zip')
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
