"""Train-only probability calibration, frozen-Q residual PPO, paired validation."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_motion_trust_experiment as previous
from scripts.analysis import queued_long_value_unfreeze as episode
from utils.queued_long_value import remaining_future
from utils.queued_service_diagnostics import mc_returns
from utils.queued_probability_scale import calibrate,probabilities

base=previous.base;study=previous.study
REV='queued_probability_decoupling_20261102'
ARMS=('raw','raw_residual','decoupled')
MODES=('raw_initial','calibrated_initial','raw','raw_residual','decoupled')
TEST=tuple(range(9600,9632));RANK=tuple(range(9700,9704));TIMES=(75,175);FUTURES=tuple(range(200,208))
SMOKE=False


def out(task):return Path(task)/'results'/base.TASK/REV/('smoke' if SMOKE else '')


def checkpoint(task,b,arm,v):return out(task)/'models'/arm/str(b)/f'{v}.pt'


def factory(arm):
    from marl_models.queued_probability_decoupling import DecoupledMotion
    m=study.factory()
    if arm!='raw':m.motion=DecoupledMotion(m.motion,arm=='decoupled',1.)
    return m


@lru_cache(maxsize=16)
def model_at(task,b,arm,v):
    import torch
    m=factory(arm);m.load_state_dict(torch.load(checkpoint(task,b,arm,v),map_location='cpu',weights_only=False)['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


def calibration(job):
    task,b=job
    import torch
    from marl_models.queued_motion_trust import MotionBatch
    root=Path(task)/'results'/base.TASK/'queued_motion_trust_20261102';m=study.model_at(task,b,2);rows=[];owners=[];sources={}
    old=json.loads((root/'protocol.json').read_text());assert old['parent_sha'][str(b)]==base.sha(study.checkpoint(task,b,2))
    for w in range(9000,9008):
        name=f'legacy_0_{b}_{w}_train';p=root/'experience'/f'{name}.pkl.gz';record=json.loads((root/'episodes'/f'{name}.json').read_text());assert base.sha(p)==record['experience_sha'];sources[p.relative_to(task).as_posix()]=base.sha(p)
        for row in base.read(p):
            if row['decision_time']<200 and int(row['decision_time'])%5==0:rows.append(row);owners.append(w)
    assert len(rows)==320;batch=MotionBatch(m,rows)
    with torch.no_grad():q=m.motion.values(*batch.x).tolist()
    data=[dict(q=values,mask=r['motion_mask'],time=r['decision_time'],workload=w) for r,values,w in zip(rows,q,owners)]
    result=calibrate(data,.5);assert result['calibrated'],result
    def describe(normalize,t):
        ps=[probabilities(r['q'],r['mask'],t,normalize) for r in data]
        return dict(mean_entropy=sum(-sum(p*math.log(p) for p in pi if p>0)/math.log(sum(r['mask'])) if sum(r['mask'])>1 else 0. for pi,r in zip(ps,data))/len(data),
            saturated=sum(max(p)>=.999 for p in ps),states=len(data))
    result.update(block=b,rows=data,input_sha=sources,raw=describe(False,.5),calibrated_policy=describe(True,result['temperature']),cpu=sorted(os.sched_getaffinity(0)),future_labels_used=False)
    base.write(out(task)/'calibration'/f'{b}.json',result);return dict(cpu=result['cpu'])


def initialize(task,blocks=range(3)):
    import torch
    from marl_models.queued_probability_decoupling import DecoupledMotion
    for b in blocks:
        original=copy.deepcopy(study.model_at(str(task),b,2));t=json.loads((out(task)/'calibration'/f'{b}.json').read_text())['temperature']
        torch.manual_seed(base.seed(REV,'residual_initialization',b));template=DecoupledMotion(copy.deepcopy(original.motion),False,.5)
        for arm in ARMS:
            m=copy.deepcopy(original)
            if arm!='raw':
                m.motion=copy.deepcopy(template);m.motion.normalize=arm=='decoupled';m.motion.policy_temperature.fill_(t if arm=='decoupled' else .5)
            p=checkpoint(task,b,arm,0);p.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=m.state_dict()),p)


def configure():
    episode.REV=REV;episode.out=out;episode.checkpoint=checkpoint;episode.model_at=model_at


def run_case(job):configure();return episode.run_case(job)


def mode_model(task,b,mode):
    if mode in ('initial','calibrated_initial'):return model_at(task,b,'decoupled',0)
    if mode=='raw_initial':return model_at(task,b,'raw',0)
    return model_at(task,b,mode,1 if SMOKE else 2)


def fit(job):
    task,b,arm,v,ws=job
    import torch
    from marl_models.queued_motion_trust import update_motion
    m=copy.deepcopy(model_at(task,b,arm,v-1));rows=[]
    for w in ws:
        name=f'{arm}_{v-1}_{b}_{w}_train';p=out(task)/'experience'/f'{name}.pkl.gz';ep=json.loads((out(task)/'episodes'/f'{name}.json').read_text());assert base.sha(p)==ep['experience_sha']
        selected=base.read(p);targets=mc_returns([r['reward'] for r in selected]);assert all(abs(r['returns']-target)<1e-9 for r,target in zip(selected,targets));rows+=selected
    value_before=copy.deepcopy(m.motion.value_scorer.state_dict()) if arm!='raw' else None
    saved=torch.load(checkpoint(task,b,arm,v-1),map_location='cpu',weights_only=False);torch.manual_seed(base.seed(REV,'fit',b,v))
    d=update_motion(m,rows,optimizer_states=saved.get('optimizer_states'));states=d.pop('_optimizer_states')
    if value_before is not None:
        assert all(torch.equal(value,m.motion.value_scorer.state_dict()[name]) for name,value in value_before.items())
        assert all(p.grad is None for p in m.motion.value_scorer.parameters());d['value_scorer_frozen']=True
    dest=checkpoint(task,b,arm,v);dest.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=m.state_dict(),optimizer_states=states),dest)
    base.write(out(task)/'updates'/f'{arm}_{v}_{b}.json',dict(block=b,arm=arm,version=v,workloads=list(ws),episodes=len(ws),rows=len(rows),metrics=d,checkpoint_sha=base.sha(dest),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def collect(job):
    # Reuse the already corrected collector; its reference alias 'initial' maps
    # to calibrated_initial, and every frozen scoring policy sees the same bank.
    previous.out=out;previous.mode_model=mode_model;previous.TIMES=TIMES;previous.MODES=MODES
    return previous.collect(job)


@lru_cache(maxsize=8)
def state_at(task,b,w,t):return base.read(out(task)/'states'/f'{b}_{w}_{t}.pkl.gz')


def label(job):
    task,b,w,t,r,chunk=job;snap=state_at(task,b,w,t);m=mode_model(task,b,'calibrated_initial');future=remaining_future(w,snap['engine'].ues,t,r)
    digest=hashlib.sha256(pickle.dumps(future,5)).hexdigest();indexes=list(range(4*chunk,min(4*chunk+4,len(snap['pre']['actions']))));values=[];counts=[]
    for i in indexes:
        e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter']);initial=len(e.tasks);e.assign(snap['plan']);e.advance(snap['pre']['actions'][i])
        while e.t<200 or e.tasks:
            now=int(e.t);assert now<=204;e.add(copy.deepcopy(future.get(now,())));e.admit()
            _,plan,move=m.act(e.observe(),adapter,base.seed(REV,'rank_tail',b,w,t,r,now));e.assign(plan);e.advance(move)
        e.audit();c=dict(e.counts);assert initial+sum(len(xs) for at,xs in future.items() if t<at<200)==c['generated']==c['success']+c['expired'];values.append(c['success']);counts.append(c)
    base.write(out(task)/'labels'/f'{b}_{w}_{t}_{r}_{chunk}.json',dict(block=b,workload=w,time=t,replica=r,indexes=indexes,values=values,counts=counts,future_sha=digest,cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def main():
    global SMOKE
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();task=a.task;SMOKE=a.smoke
    import torch,fcntl
    torch.set_num_threads(1);out(task).mkdir(parents=True,exist_ok=True);lock=open(out(task)/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if SMOKE:
        calibration((str(task),0));initialize(task,[0])
        for arm in ARMS:run_case((str(task),0,arm,0,9499,'train',True));fit((str(task),0,arm,1,[9499]))
        collect((str(task),0,RANK[0]));label((str(task),0,RANK[0],TIMES[-1],FUTURES[0],0));print('SMOKE PASS',flush=True);return
    assert not (out(task)/'protocol.json').exists(),'Existing run must not be overwritten'
    cpus=sorted(os.sched_getaffinity(0));files={p.relative_to(REPO_ROOT).as_posix():base.sha(p) for d in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/d).glob('*.py')}
    parents={str(b):base.sha(study.checkpoint(task,b,2)) for b in range(3)}
    base.write(out(task)/'protocol.json',dict(revision=REV,source_sha=files,parent_sha=parents,train=list(range(9500,9596)),test=TEST,rank=RANK,times=TIMES,futures=FUTURES,arms=ARMS,
        calibration_workloads=list(range(9000,9008)),calibration_source_revision='queued_motion_trust_20261102',calibration_actor_version=0,target_entropy=.5,
        allocation='frozen',features=119,rounds=2,episodes_per_round=48,full_kl_limit=.02,step_kl_limit=.0002,
        ranking_reference='calibrated_initial frozen continuation only; common Q for all scoring policies, not each learned own Q',
        gate='Decoupled must beat calibrated_initial, raw_initial, raw PPO and raw_residual in every lineage; >16/32 paired positives and all leave-one-out gains positive. Ranking incremental gain over calibrated_initial bootstrap lower >0 each lineage.',
        cpu=cpus,independent_confirmation=False,scope='Combined mapping-plus-residual test; raw_residual and decoupled have matched policy capacity. Raw PPO differs in trainable architecture; no pure architecture superiority claim.'))
    original=base.out;base.out=out
    try:
        base.parallel(task,'calibration',calibration,[(str(task),b) for b in range(3)],cpus);initialize(task)
        for v in (1,2):
            ws=list(range(9500+(v-1)*48,9500+v*48))
            base.parallel(task,f'train_{v}',run_case,[(str(task),b,arm,v-1,w,'train',True) for b in range(3) for arm in ARMS for w in ws],cpus)
            base.parallel(task,f'update_{v}',fit,[(str(task),b,arm,v,ws) for b in range(3) for arm in ARMS],cpus)
        jobs=[]
        for b in range(3):
            for mode in MODES:
                arm='raw' if mode=='raw_initial' else 'decoupled' if mode=='calibrated_initial' else mode
                jobs.extend((str(task),b,arm,0 if mode.endswith('initial') else 2,w,mode,False) for w in TEST)
        base.parallel(task,'evaluation',run_case,jobs,cpus)
        base.parallel(task,'rank_collection',collect,[(str(task),b,w) for b in range(3) for w in RANK],cpus)
        jobs=[]
        for t in TIMES:
            for b in range(3):
                for w in RANK:
                    k=len(state_at(str(task),b,w,t)['pre']['actions']);jobs.extend((str(task),b,w,t,r,c) for r in FUTURES for c in range(math.ceil(k/4)))
        base.parallel(task,'independent_futures',label,jobs,cpus)
    finally:base.out=original
    from scripts.analysis.analyze_queued_probability_decoupling import analyze
    analyze(task)
    assert all(base.sha(REPO_ROOT/name)==digest for name,digest in files.items())
    assert all(base.sha(study.checkpoint(task,int(b),2))==digest for b,digest in parents.items())
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
