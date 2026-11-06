"""Server experiment: root-centered advantage vs time-centered absolute Q.

Labels use frozen A2 roots and the fixed matched three-second continuation.
No actor/critic update and no true future or identifiers in predictor inputs.
"""
import argparse
import copy
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
import numpy as np
from utils.queued_root_advantage import fit_models,predict,tie_weights,bootstrap_blocks

TASK='chapter5_dynamic_service_20261028'
REV='queued_root_advantage_20261103'
WORKLOADS={'train':tuple(range(12000,12012)),'dev':tuple(range(12100,12104)),'test':tuple(range(12200,12208))}
REPLICAS={'train':tuple(range(7000,7008)),'dev':tuple(range(7100,7116)),'test':tuple(range(7200,7232))}
METHODS=('root_centered','time_centered_absolute','root_shuffled','original_q','short_window','uniform')

def result_dir(root):return Path(root)/'results'/TASK/REV
def validation_dir(root):return Path(root)/'output'/TASK/'validation'/REV
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def feature_path(root,b,w,now):return result_dir(root)/'features'/f'{b}_{w}_{now}.json'
def state_path(root,b,w,now):return result_dir(root)/'states'/f'{b}_{w}_{now}.pkl.gz'
def label_path(root,b,w,now,i,j,rep):return result_dir(root)/'labels'/f'{b}_{w}_{now}_{i}_{j}_{rep}.json'

def feature_record(snap,split):
    row=snap['row'];choices=[tuple(x) for x in snap['choices']];assert 1<=len(choices)<=4
    x=np.array([row['terminal_features'][i,j] for i,j in choices],float)
    assert x.shape==(len(choices),99) and np.isfinite(x).all()
    return dict(b=snap['b'],w=snap['w'],time=snap['time'],split=split,choices=choices,x=x.tolist(),
        p=[float(row['short'][i,j]) for i,j in choices],old_tail=[float(row['tail'][i,j]) for i,j in choices],
        retained_mass=float(row['retained_mass']),plan_count=len(row['plans']),motion_count=len(row['actions']),
        input_scope='Public single-forecast terminal features only; identifiers are metadata, never predictor features.')

def export_old(root,b,w,now):
    from scripts.analysis import queued_joint_mpo_experiment as t
    path=t.validation(root)/'candidate_states'/f'{b}_{w}_{now}.pkl.gz'
    snap=t.read(str(path));d=feature_record(snap,'old_train');d['parent_snapshot_sha256']=sha(path)
    write(feature_path(root,b,w,now),d)
    return [dict(b=b,w=w,time=now,split='old_train')]

def collect(root,b,w,split):
    from scripts.analysis import queued_joint_mpo_experiment as t
    from utils.queued_long_value import LongAdapter
    model=t.parent(root,b);tape=t.RecoveryScenario(w,'burst','normal',.6)
    e=t.Engine(t.Protocol(),t.START,tape.ue_positions);adapter=LongAdapter(True);records=[]
    tail=t.read(str(t.out(root)/'tail'/f'{b}.pkl.gz'))
    while e.t<=175:
        now=int(e.t);e.add(tape.slot(now));e.admit();obs=e.observe()
        if now in (75,175):
            local=t.LearningAdapter();local.history=copy.deepcopy(adapter.history);local.last_time=adapter.last_time
            row=t.bank(model,obs,local,t.seed(t.REV,'bank',b,w,now))
            t.teacher(row,obs,local.history,t.seed(t.REV,'forecast',b,w,now),tail)
            choices=sorted(set((i,j) for i in (0,len(row['plans'])-1) for j in (0,len(row['actions'])-1)))
            assert 1<=len(choices)<=4
            full=dict(time=e.t,positions=e.positions,ues=e.ues,tasks=[asdict(x) for x in sorted(e.tasks.values(),key=lambda z:(z.deadline,z.uid))],protocol=e.p)
            later=copy.deepcopy(adapter);later.prepare(obs)
            snap=dict(engine=t.Engine.from_observation(full),adapter=later,row=row,choices=choices,b=b,w=w,time=now)
            t.dump(state_path(root,b,w,now),snap)
            record=feature_record(snap,split);record['snapshot_sha256']=sha(state_path(root,b,w,now))
            write(feature_path(root,b,w,now),record)
            records.append(dict(b=b,w=w,time=now,split=split))
        _,plan,motion=model.act(obs,adapter,t.seed(t.REV,'parent',b,w,now));e.assign(plan);e.advance(motion)
    e.audit();return records

def label(root,b,w,now,i,j,rep,split):
    from scripts.analysis import queued_joint_mpo_experiment as t
    from utils.queued_long_value import remaining_future
    from utils.queued_service_controllers import assignments
    snap=t.read(str(state_path(root,b,w,now)));e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter'])
    row=snap['row'];model=t.parent(root,b);future=remaining_future(w,e.ues,now,rep)
    root_ids=set(e.tasks);prefix=None
    e.assign(row['plans'][i]);e.advance(row['actions'][j])
    while e.t<200 or e.tasks:
        tick=int(e.t)
        if tick==now+3:prefix=int(e.counts['success'])
        e.add(copy.deepcopy(future.get(tick,())));e.admit();obs=e.observe()
        if tick<now+3:
            adapter.history._update_history(obs);adapter.last_time=obs['time']
            plan=assignments(obs);motion=[(0.,0.)]*len(e.positions)
        else:_,plan,motion=model.act(obs,adapter,t.seed(t.REV,'candidate_tail',b,w,now,rep,tick))
        e.assign(plan);e.advance(motion)
    audit=e.audit();c=e.counts
    assert prefix is not None and c['generated']==c['success']+c['expired']
    record=dict(b=b,w=w,time=now,i=i,j=j,replica=rep,split=split,mode='matched',
        success=c['success'],generated=c['generated'],expired=c['expired'],actual_prefix=prefix,actual_tail=c['success']-prefix,
        root_success=sum(t.uid in root_ids and t.stage=='completed' for t in e.finished.values()),audit=audit)
    write(label_path(root,b,w,now,i,j,rep),record)
    return dict(file=str(label_path(root,b,w,now,i,j,rep).relative_to(root)))

def init_worker(cpus):os.sched_setaffinity(0,{cpus[(mp.current_process()._identity[-1]-1)%len(cpus)]})
def timed(job):
    kind,args=job;wall=time.time();cpu=time.process_time()
    try:
        result=globals()[kind](*args)
        return dict(ok=True,kind=kind,result=result,start=wall,end=time.time(),cpu_seconds=time.process_time()-cpu,
                    pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),threads=1)
    except Exception:
        import traceback
        return dict(ok=False,kind=kind,args=list(args[1:]),error=traceback.format_exc(),start=wall,end=time.time(),cpu_seconds=time.process_time()-cpu)

def protection(root):
    from scripts.analysis import queued_candidate_reliability as prior
    result=prior.protection(root)
    for folder in (prior.result_dir(root),Path(root)/'source_candidate_reliability_20261103'):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:result[str(p.relative_to(root))]=sha(p)
    return result

def load_features(root):return sorted([readj(p) for p in (result_dir(root)/'features').glob('*.json')],key=lambda x:(x['b'],x['w'],x['time']))

def outcomes(root,f):
    choices=[tuple(x) for x in f['choices']];split=f['split'];b,w,now=f['b'],f['w'],f['time']
    if split=='old_train':
        folder=Path(root)/'results'/TASK/'queued_candidate_reliability_20261103'/'labels'
        reps=range(5000,5032)
        records=[[readj(folder/f'{b}_{w}_{now}_{i}_{j}_{rep}_matched.json') for rep in reps] for i,j in choices]
    else:
        reps=REPLICAS[split];records=[[readj(label_path(root,b,w,now,i,j,rep)) for rep in reps] for i,j in choices]
    y=np.array([[r['success'] for r in rs] for rs in records],float)
    for k in range(y.shape[1]):assert len({rs[k]['generated'] for rs in records})==1
    for rs in records:
        for r in rs:assert r['generated']==r['success']+r['expired'] and r['audit']['status']=='PASS'
    return y

def freeze_models(root):
    r=result_dir(root);v=validation_dir(root);features=load_features(root)
    train=[f for f in features if f['split'] in ('old_train','train')];assert len(train)==96
    rows=[];inputs={}
    for f in train:
        y=outcomes(root,f).mean(1);rows.append(dict(x=f['x'],p=f['p'],y=y,time=f['time'],key=f"{f['b']}_{f['w']}_{f['time']}"))
        inputs[f"{f['b']}_{f['w']}_{f['time']}"]=hashlib.sha256(y.tobytes()).hexdigest()
    wall=time.perf_counter();cpu=time.process_time();model=fit_models(rows,alpha=1.)
    write(r/'models.json',model)
    write(v/'fit_frozen.json',dict(status='FROZEN',epoch=time.time(),model_sha256=sha(r/'models.json'),train_roots=96,
        training_target_hashes=inputs,dev_or_test_labels_read=False,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu))
    return model

def method_scores(model,f):
    x=np.array(f['x']);p=np.array(f['p']);scores={name:predict(model,name,x,p,f['time']) for name in ('root_centered','time_centered_absolute','root_shuffled')}
    scores.update(original_q=p+np.array(f['old_tail']),short_window=p,uniform=np.zeros(len(p)))
    return scores

def evaluate(root,model,split,write_outputs=True):
    from scripts.analysis.queued_candidate_reliability import crossfit_selection,paired_statistics
    fs=[f for f in load_features(root) if f['split']==split];assert len(fs)==len(WORKLOADS[split])*6
    wids=WORKLOADS[split];R=len(REPLICAS[split]);gains={m:np.zeros((len(wids),3,2,R)) for m in METHODS}
    truth=[];preds={m:[] for m in METHODS};moments={m:[] for m in METHODS};state_rows=[];cross=[]
    for f in fs:
        y=outcomes(root,f);scores=method_scores(model,f);base=y.mean(0);wi=wids.index(f['w']);ti=(75,175).index(f['time'])
        stat=paired_statistics(y,scores['root_centered'])
        row=dict(b=f['b'],w=f['w'],time=f['time'],split=split,candidate_count=len(f['choices']),mean_pair_se=float(np.mean([x['se'] for x in stat['pairs']])) if stat['pairs'] else None,gains={},top_weights={})
        for m,s in scores.items():
            weights=tie_weights(s);g=weights@y-base;gains[m][wi,f['b'],ti]=g
            row['gains'][m]=float(g.mean());row['top_weights'][m]=weights.tolist();preds[m].extend(s-s.mean())
            pc=s-s.mean();tc=y.mean(1)-y.mean();moments[m].append([np.mean((pc-tc)**2),np.mean(tc**2),np.mean(pc**2),np.mean(pc*tc)])
        truth.extend(y.mean(1)-y.mean())
        if R>=32:cross.append(crossfit_selection(y,16))
        state_rows.append(row)
    truth=np.array(truth);methods={}
    for m in METHODS:
        pred=np.array(preds[m]);summary=bootstrap_blocks(gains[m].mean(axis=(1,2)))
        mse,t2,p2,cov=np.mean(moments[m],0)
        summary.update(within_root_ev=float(1-mse/max(t2,1e-12)),
            centered_mse=float(mse),centered_correlation=float(cov/np.sqrt(p2*t2)) if p2>1e-24 and t2>1e-24 else None,
            by_lineage=gains[m].mean(axis=(0,2,3)).tolist(),by_time={str(t):float(gains[m][:,:,i].mean()) for i,t in enumerate((75,175))})
        methods[m]=summary
    comparisons={m:bootstrap_blocks((gains['root_centered']-gains[m]).mean(axis=(1,2))) for m in ('uniform','time_centered_absolute','original_q','root_shuffled')}
    oracle_gain=float(np.mean([x['heldout_selection_gain_mean'] for x in cross])) if cross else None
    gate=all(comparisons[m]['simultaneous98_333'][0]>0 for m in ('uniform','time_centered_absolute','original_q')) and min(methods['root_centered']['by_lineage'])>0 and (oracle_gain is None or oracle_gain>0)
    result=dict(status='COMPLETE',split=split,roots=len(fs),workloads=list(wids),replicas=R,methods=methods,comparisons=comparisons,
        centered_supervision_gate=bool(gate),crossfit_label_gain16=oracle_gain,state_rows=state_rows,
        scope='New workloads/roots, same three inherited parent policies, four fixed diagnostic candidates and matched continuation. Not new MARL training or closed-loop DSR.')
    if write_outputs:write(result_dir(root)/f'analysis_{split}.json',result)
    return result


def smoke(root):
    from scripts.analysis import queued_joint_mpo_experiment as t
    old=t.validation(root)/'candidate_states'/'0_11100_175.pkl.gz'
    expected=Path(root)/'results'/TASK/'queued_candidate_reliability_20261103'/'labels'/'0_11100_175_0_0_5000_matched.json'
    original_state,original_label=globals()['state_path'],globals()['label_path']
    target=validation_dir(root)/'smoke_label.json'
    started=time.perf_counter();cpu=time.process_time()
    globals()['state_path']=lambda *args:old
    globals()['label_path']=lambda *args:target
    try:label(root,0,11100,175,0,0,5000,'smoke')
    finally:globals()['state_path'],globals()['label_path']=original_state,original_label
    actual=readj(target);reference=readj(expected)
    keys=('success','generated','expired','actual_prefix','actual_tail','root_success')
    assert all(actual[k]==reference[k] for k in keys),keys
    features=feature_record(t.read(str(old)),'smoke')
    write(validation_dir(root)/'smoke.json',dict(status='PASS',old_matched_label_exact_replay=True,
        replay_fields=list(keys),parent_state_sha256=sha(old),reference_label_sha256=sha(expected),feature_dim=len(features['x'][0]),
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu))
    print('SMOKE PASS',flush=True)


def run(root):
    if sys.platform!='linux':raise RuntimeError('Physical work and fitting run on server only')
    r=result_dir(root);v=validation_dir(root)
    if r.exists():raise FileExistsError(r)
    r.mkdir(parents=True);v.mkdir(parents=True,exist_ok=True)
    lock=v/'run.lock'
    with lock.open('x') as h:h.write(str(os.getpid()))
    try:
        before=protection(root);write(v/'protection_before.json',before)
        files=['utils/queued_root_advantage.py','scripts/analysis/queued_root_advantage_experiment.py','tests/test_queued_root_advantage.py']
        source={name:sha(REPO_ROOT/name) for name in files};cpus=sorted(os.sched_getaffinity(0))
        write(r/'protocol.json',dict(revision=REV,source=source,workloads={k:list(x) for k,x in WORKLOADS.items()},replicas={k:list(x) for k,x in REPLICAS.items()},
            feature_dim=99,alpha=1.,old_train_roots=24,new_train_roots=72,dev_roots=24,test_roots=48,new_branch_budget_upper=9984,
            allowed_cpus=cpus,methods=list(METHODS),train_only_scaling=True,source_root=str(REPO_ROOT)))
        pending={};scheduled=set();timings=[];started=time.perf_counter();own=time.process_time()
        with ProcessPoolExecutor(max_workers=len(cpus),mp_context=mp.get_context('fork'),initializer=init_worker,initargs=(cpus,)) as pool:
            def submit(kind,args):pending[pool.submit(timed,(kind,args))]=(kind,args)
            for b in range(3):
                for w in range(11100,11104):
                    for now in (75,175):submit('export_old',(str(root),b,w,now))
            for split,ws in WORKLOADS.items():
                for b in range(3):
                    for w in ws:submit('collect',(str(root),b,w,split))
            # Poll atomic feature files so t75 labels can use spare CPUs while
            # the same real collector continues its trajectory to t175.
            while pending or len(scheduled)<144:
                for split,ws in WORKLOADS.items():
                    for now in (75,175):
                        for b in range(3):
                            for w in ws:
                                key=(b,w,now)
                                if key in scheduled or not feature_path(root,*key).exists():continue
                                f=readj(feature_path(root,*key));assert f['split']==split;scheduled.add(key)
                                for rep in REPLICAS[split]:
                                    for i,j in f['choices']:submit('label',(str(root),b,w,now,i,j,rep,split))
                ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in ready:
                    kind,args=pending.pop(future);rec=future.result();timings.append(rec)
                    if not rec['ok']:
                        write(v/'failure.json',rec)
                        for item in pending:item.cancel()
                        raise RuntimeError(rec['error'])
                if timings and len(timings)//256>getattr(run,'printed',0):
                    run.printed=len(timings)//256
                    print('completed',len(timings),'pending',len(pending),'roots',len(scheduled),'wall',round(time.perf_counter()-started,1),flush=True)
        wall=time.perf_counter()-started;own=time.process_time()-own;cpu=sum(x['cpu_seconds'] for x in timings);active=peak=0
        for _,change in sorted([(x['start'],1) for x in timings]+[(x['end'],-1) for x in timings]):active+=change;peak=max(peak,active)
        expected=sum(len(f['choices'])*len(REPLICAS[f['split']]) for f in load_features(root) if f['split']!='old_train')
        assert len(scheduled)==144 and len(list((r/'labels').glob('*.json')))==expected<=9984
        resource=dict(tasks=len(timings),workers=len(cpus),threads=1,allowed_cpus=cpus,wall_seconds=wall,worker_cpu_seconds=cpu,coordinator_cpu_seconds=own,
            equivalent_cpus=(cpu+own)/wall,utilization=(cpu+own)/wall/len(cpus),peak_active=peak,failures=0,retries=0,records=timings)
        write(v/'resources.json',resource)
        model=freeze_models(root)
        frozen=readj(v/'fit_frozen.json');write(v/'evaluation_read_started.json',dict(epoch=time.time(),frozen_model_sha256=frozen['model_sha256']))
        assert sha(r/'models.json')==frozen['model_sha256']
        dev=evaluate(root,model,'dev');test=evaluate(root,model,'test')
        assert protection(root)==before;write(v/'protection_after.json',before)
        assert sha(r/'models.json')==frozen['model_sha256']
        write(v/'server_review.json',dict(status='PASS',new_branches=expected,branch_budget_upper=9984,features=168,new_collectors=72,train_roots=96,dev_roots=24,test_roots=48,
            model_sealed_before_evaluation=True,protected_files=len(before),source_unchanged=all(sha(REPO_ROOT/p)==d for p,d in source.items()),
            source=source,gate=test['centered_supervision_gate'],training_scope='Three fixed analytic supervised fits; no actor/critic or MARL update.',independent_review=False))
        print('COMPLETE',json.dumps(dict(gate=test['centered_supervision_gate'],root_gain=test['methods']['root_centered']['mean'],
            comparisons={k:v['simultaneous98_333'] for k,v in test['comparisons'].items()},equivalent_cpus=resource['equivalent_cpus'],wall_seconds=wall)),flush=True)
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:raise ValueError('Read metadata/analysis files to inspect a completed run; --run explicitly creates a new revision')
    run(args.task_root.resolve())

if __name__=='__main__':main()
