"""Frozen repair-policy diagnosis; no optimizer, parameter update or new training.

Future requests are read ONLY by explicitly labelled counterfactual admission
diagnostics after the policy has acted. They are never passed to a controller.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_feasible_credit_20261030'
DIAG='diagnostic_20261031'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def locations(args):
    root=args.task/'results'/TASK/REVISION
    return root,root/DIAG,args.task/'output'/TASK/'validation'/REVISION/DIAG
def jobs():
    return [dict(method=m,block=b,workload=w,mode=mode,id=f'{m}_{b}_{w}_{mode}')
        for m in ('repair_joint','feasible_trajectory') for b in range(3) for w in range(2) for mode in ('replay','uniform')]
def initialize(args):
    args.source=args.source.resolve();args.task=args.task.resolve();sys.path.insert(0,str(args.source))
    os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    root,out,val=locations(args);p=json.loads((root/'protocol.json').read_text())
    for name,d in p['source'].items():
        if sha(args.source/name)!=d:raise RuntimeError(f'frozen source changed: {name}')
    args.contract=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();args.script_sha=sha(__file__)
    return root,out,val

def worker(args):
    root,out,val=initialize(args)
    from scripts.analysis.dynamic_service_experiment import configure_runtime
    torch=configure_runtime()
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    from marl_models.dynamic_service_repair import RepairedPolicy,RepairedCritic,normalized_plan
    from marl_models.dynamic_service_hmarl import tensor
    from utils.dynamic_service_baselines import matching_search
    from utils.dynamic_service_feasible_motion import motion_library
    from utils.dynamic_service_joint_rollout import causal_returns,slot_surrogate
    from utils.dynamic_service_scenarios import Scenario,seed_for
    job=jobs()[args.index];path=out/'cases'/(job['id']+'.json')
    if path.exists():
        d=json.loads(path.read_text());assert d['script_sha256']==args.script_sha;return
    models={};saved={};model_hashes={}
    for cp in (0,20,40):
        file=root/'models/pilot'/str(job['block'])/job['method']/f'{cp:04}.pt'
        s=torch.load(file,map_location='cpu',weights_only=False);assert s['contract']==args.contract and s['episode']==cp
        a=RepairedPolicy(job['method']);a.load_state_dict(s['actor']);a.eval();models[cp]=a;saved[cp]=s;model_hashes[str(cp)]=sha(file)
    actor=models[40];critic=RepairedCritic();critic.load_state_dict(saved[40]['critic']);critic.eval()
    original=json.loads((root/'evaluation/pilot'/str(job['block'])/job['method']/'40'/f'D_{job["workload"]}_0.json').read_text())
    assert original['checkpoint_sha256']==model_hashes['40']
    env=DynamicServiceEnvV2(Scenario(original['workload_seed'],'D'))
    hidden=actor.initial_state();ch=critic.initial_state();alternatives={cp:models[cp].initial_state() for cp in (0,20)}
    rows=[];probes=[];separability=[];totals=defaultdict(int)
    def probs(a,state,bank,plan):
        with torch.no_grad():
            feature=tensor(bank['features'],a.device);context=a.context(torch.cat((state.flatten(),tensor(plan,a.device))))
            conditioned=a.score_scale*(a.candidate(feature)*context).sum(-1)/(128**.5)
            bias=a.bias(feature).squeeze(-1);mask=torch.as_tensor(bank['mask'],device=a.device)
            p=torch.softmax((conditioned+bias).masked_fill(~mask,-torch.inf),-1).numpy()
        return p,conditioned.numpy(),bias.numpy()
    def tv(p,q):return float(np.abs(p-q).sum()/2)
    def kl(p,q):return float((p*np.log(np.maximum(p,1e-30)/np.maximum(q,1e-30))).sum())
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else None
    def clone(e):
        x=copy.copy(e)
        for name in ('positions','feedback','motion','generated_per_ue','success_per_ue'):setattr(x,name,getattr(e,name).copy())
        return x
    for t in range(1000):
        obs=env.observe();batch=env.batch();hb=hidden.numpy().copy();cb=ch.numpy().copy()
        seed=seed_for(f'repair/evaluation/actions/{job["block"]}/{job["workload"]}/0',t)
        sample=actor.act(obs,batch,hidden,seed=seed,fixed_assignment=matching_search(batch) if job['method']=='feasible_trajectory' else None)
        if job['mode']=='replay':
            prediction=critic.predict(obs,batch,ch,sample,t);ch=torch.tensor(prediction['critic_hidden'])
            with torch.no_grad():
                for cp in alternatives:alternatives[cp]=models[cp].encoder(obs,alternatives[cp])
            if t%20==0:
                bank=sample['bank'];mask=bank['mask'];p=sample['motion_probs'].astype(float);uniform=mask.astype(float)/mask.sum()
                state=tensor(sample['hidden'],actor.device)
                plan=normalized_plan(batch,sample['assignment']) if actor.use_plan else np.zeros(18,np.float32)
                p2,context,bias=probs(actor,state,bank,plan);np.testing.assert_allclose(p,p2,atol=3e-6)
                coverage=bank['features'][:,40].astype(float)*100;best=float(coverage[mask].max())
                d=dict(t=t,legal=int(mask.sum()),entropy=float(-(p*np.log(np.maximum(p,1e-30))).sum()),
                    entropy_fraction=float(-(p*np.log(np.maximum(p,1e-30))).sum()/np.log(mask.sum())) if mask.sum()>1 else 1.,
                    kl_uniform=kl(p,uniform),tv_uniform=tv(p,uniform),max_probability=float(p.max()),
                    expected_covered=float(p@coverage),uniform_covered=float(uniform@coverage),best_covered=best,
                    probability_on_best_coverage=float(p[mask & (coverage>=best-1e-6)].sum()),
                    context_logit_std=float(context[mask].std()),bias_logit_std=float(bias[mask].std()),score_scale=float(actor.score_scale.detach()),
                    selected_index=sample['motion_index'])
                for cp in alternatives:
                    oldp,_,_=probs(models[cp],alternatives[cp],bank,plan)
                    d[f'tv_checkpoint_{cp}']=tv(p,oldp);d[f'kl_checkpoint_{cp}']=kl(p,oldp)
                    if len(batch) and actor.learn_allocation:
                        with torch.no_grad():
                            olddist=models[cp].allocation.distribution_context(batch,models[cp].allocation.encode_context(batch,alternatives[cp]),tensor(sample['risk'],actor.device))
                        q=olddist.probs.numpy();pp=sample['allocation_probs']
                        d[f'allocation_conditional_kl_checkpoint_{cp}']=float((pp*np.log(np.maximum(pp,1e-30)/np.maximum(q,1e-30))).sum())
                no_demand=dict(obs,ue=obs['ue'].copy());no_demand['ue'][:,:,2:]=0
                no_feedback=dict(obs,feedback=np.zeros_like(obs['feedback']))
                with torch.no_grad():
                    no_state=actor.encoder(no_demand,tensor(hb,actor.device));fb_state=actor.encoder(no_feedback,tensor(hb,actor.device))
                    reset_state=actor.encoder(obs,actor.initial_state())
                bank0=motion_library(no_demand)
                for name,s,b,pl in [('no_demand',no_state,bank0,plan),('no_encoder_demand',no_state,bank,plan),
                    ('no_candidate_demand',state,bank0,plan),('no_feedback',fb_state,bank,plan),
                    ('reset_history',reset_state,bank,plan),('no_plan',state,bank,np.zeros(18,np.float32))]:
                    q,_,_=probs(actor,s,b,pl);d[name+'_tv']=tv(p,q)
                if t<999:
                    # Future information is a diagnostic oracle ONLY, never policy input.
                    requests=env.tape.slot(t+1);ids=np.asarray([r.ue for r in requests],int)
                    after=env.positions[None]+bank['actions']*15
                    distance=np.linalg.norm(after[:,:,None]-env.ue_positions[ids][None,None],axis=-1)
                    counts=(distance<=100).any(1).sum(1).astype(float)
                    d.update(oracle_next_uniform_admitted=float(uniform@counts),oracle_next_policy_admitted=float(p@counts),
                        oracle_next_best_admitted=float(counts[mask].max()),oracle_next_min_admitted=float(counts[mask].min()),
                        oracle_next_selected_centered=float(counts[sample['motion_index']]-p@counts))
                probes.append(d)
            if t in (100,400,700) and len(batch):
                left=clone(env);right=clone(env)
                a=left.step(sample['actions'],batch.source);b=right.step(sample['actions'],np.full(len(batch),5,dtype=np.int64))
                separability.append(dict(t=t,current_success_source=a['success'],current_success_mbs=b['success'],
                    next_positions_identical=bool(np.array_equal(left.positions,right.positions)),
                    next_capacity_identical=bool(np.array_equal(left.capacity,right.capacity)),
                    next_batch_tx_identical=bool(np.array_equal(left.batch().tx,right.batch().tx)),
                    next_batch_ids_identical=bool(np.array_equal(left.batch().ids,right.batch().ids)),
                    next_feedback_identical=bool(np.array_equal(left.feedback,right.feedback))))
        else:
            gm=torch.Generator().manual_seed((int(seed)+1000003)%(2**63-1))
            index=int(torch.multinomial(torch.tensor(sample['bank']['mask'],dtype=torch.float32),1,generator=gm))
            sample['actions']=sample['bank']['actions'][index].copy()
        hidden=torch.tensor(sample['hidden']);result=env.step(sample['actions'],sample['assignment'])
        if job['mode']=='replay':
            expected=original['rows'][t]
            for key in ('generated','admitted','success','uncovered','deadline_misses','mbs','counts','assignment'):assert result[key]==expected[key]
            np.testing.assert_allclose(env.positions,expected['positions'],atol=1e-7,rtol=0)
            rows.append(dict(observation=obs,batch=batch,hidden_before=hb,critic_hidden_before=cb,**sample,**prediction,reward=result['reward'],time=t))
        for key in ('generated','admitted','success','uncovered','deadline_misses','mbs'):totals[key]+=result[key]
        if t%100==0:atomic(val/'progress'/(job['id']+'.json'),dict(pid=os.getpid(),t=t,time=time.time()))
    gradients=[];credit={}
    if rows:
        targets=causal_returns(rows);adv=np.asarray([t['motion_target']-r['motion_value'] for r,t in zip(rows,targets)])
        mc=np.cumsum(np.asarray([r['reward'] for r in rows])[::-1])[::-1]-np.asarray([r['reward'] for r in rows])
        values=np.asarray([r['motion_value'] for r in rows]);lam=np.asarray([t['motion_target'] for t in targets])
        aa=np.concatenate([r['reward']-r['batch_values'] for r in rows]) if actor.learn_allocation else np.empty(0)
        scale=max(float(np.std(np.r_[aa,adv])),.05)
        credit=dict(advantage_mean=float(adv.mean()),advantage_std=float(adv.std()),reward_std=float(np.std([r['reward'] for r in rows])),
            value_lambda_rmse=float(np.sqrt(np.mean((values-lam)**2))),value_mc_rmse=float(np.sqrt(np.mean((values-mc)**2))),
            lambda_target_mean=float(lam.mean()),value_mean=float(values.mean()),common_scale=scale,
            oracle_one_step_admission_advantage_correlation=corr([adv[d['t']] for d in probes],[d['oracle_next_selected_centered'] for d in probes]))
        named=list(actor.named_parameters());parameters=[p for _,p in named]
        for start in (100,400,700):
            burn=start-16;h=tensor(rows[burn]['hidden_before'],actor.device)
            with torch.no_grad():
                for i in range(burn,start):h=actor.encoder(rows[i]['observation'],h)
            alloc=[];move=[];entropy=[]
            for i in range(start,start+16):
                r=rows[i];z=actor.replay(r['observation'],r['batch'],h,r);h=z['hidden']
                move.append(slot_surrogate(z['motion_log'],tensor(r['motion_log'],actor.device),tensor(adv[i]/scale,actor.device)))
                entropy.append(-.001*z['motion_entropy'])
                if actor.learn_allocation and len(r['batch']):
                    a=(r['reward']-r['batch_values'])/scale
                    alloc.append(slot_surrogate(z['allocation_log'],tensor(r['allocation_log'],actor.device),tensor(a,actor.device))-.001*z['allocation_entropy'].sum())
            losses=dict(motion=torch.stack(move).mean(),motion_entropy=torch.stack(entropy).mean())
            if alloc:losses['allocation']=torch.stack(alloc).mean()
            gs={k:torch.autograd.grad(v,parameters,allow_unused=True,retain_graph=True) for k,v in losses.items()}
            groupstats={}
            for group,prefix in [('all',''),('encoder','encoder.'),('context','context.'),('candidate','candidate.'),('bias','bias.'),('scale','score_scale')]:
                groupstats[group]={k:float(torch.sqrt(sum((g.square().sum() for (name,_),g in zip(named,grads) if g is not None and name.startswith(prefix)),torch.tensor(0.)))) for k,grads in gs.items()}
            combined=[sum((g[i] for g in gs.values() if g[i] is not None),torch.zeros_like(p)) for i,p in enumerate(parameters)]
            norm=float(torch.sqrt(sum(g.square().sum() for g in combined)))
            gradients.append(dict(start=start,groups=groupstats,combined_norm=norm,global_clip_multiplier=min(1,.5/max(norm,1e-30))))
    d=dict(**job,contract=args.contract,script_sha256=args.script_sha,checkpoint_sha256=model_hashes,totals=dict(totals),
        dsr=totals['success']/totals['generated'],admission=totals['admitted']/totals['generated'],probes=probes,credit=credit,gradients=gradients,
        physical_separability=separability,parameter_updates=False,oracle_not_policy_input=True)
    atomic(path,d);print(json.dumps(dict(id=job['id'],status='PASS',dsr=d['dsr'],probes=len(probes))),flush=True)

def analyze(args):
    root,out,val=initialize(args);cases=[json.loads((out/'cases'/(j['id']+'.json')).read_text()) for j in jobs()]
    import statistics
    summary={}
    for method in ('repair_joint','feasible_trajectory'):
        ds=[d for d in cases if d['method']==method and d['mode']=='replay'];ps=[p for d in ds for p in d['probes']]
        numeric={k:statistics.mean(p[k] for p in ps if k in p) for k in ps[0] if isinstance(ps[0][k],(int,float))}
        summary[method]=dict(mean_probe=numeric,replay_dsr=statistics.mean(d['dsr'] for d in ds),
            uniform_dsr=statistics.mean(d['dsr'] for d in cases if d['method']==method and d['mode']=='uniform'),
            replay_admission=statistics.mean(d['admission'] for d in ds),
            uniform_admission=statistics.mean(d['admission'] for d in cases if d['method']==method and d['mode']=='uniform'),
            credit=[d['credit'] for d in ds],gradients=[g for d in ds for g in d['gradients']],
            separability=[s for d in ds for s in d['physical_separability']])
    atomic(out/'analysis.json',dict(status='PASS_FROZEN_REPAIR_DIAGNOSIS',contract=args.contract,script_sha256=args.script_sha,
        cases=len(cases),exact_replay_steps=12000,summary=summary,training=False,
        limitations='Frozen interventions and same-data model probes, 3 development seeds x 2 existing tapes; future used only as labelled diagnostic oracle; not retrained ablations or independent confirmation'))

def run(args):
    root,out,val=initialize(args);val.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protected=val/'protected_before.json'
    if not protected.exists():atomic(protected,{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and DIAG not in p.relative_to(root).parts and p.suffix not in ('.lock','.tmp')})
    pending=jobs();active=[];done=[];failed=[];cpus=list(range(16,16+min(args.workers,24)))
    while pending or active:
        busy={x['cpu'] for x in active}
        for cpu in cpus:
            if not pending or cpu in busy:continue
            job=pending.pop(0);log=(val/(job['id']+'.log')).open('a')
            cmd=['taskset','-c',str(cpu),sys.executable,str(Path(__file__).resolve()),'worker','--source',str(args.source),'--task',str(args.task),'--index',str(jobs().index(job))]
            p=subprocess.Popen(cmd,env=os.environ,cwd=args.source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(job=job,proc=p,log=log,cpu=cpu))
        for x in active[:]:
            rc=x['proc'].poll()
            if rc is None:continue
            x['log'].close();active.remove(x)
            (done if rc==0 else failed).append(x['job']['id'])
        atomic(val/'status.json',dict(time=time.time(),completed=len(done),total=24,failed=failed,active=[dict(id=x['job']['id'],pid=x['proc'].pid,cpu=x['cpu']) for x in active]))
        if active:time.sleep(3)
    if not failed:analyze(args)
    prior=json.loads(protected.read_text());changed=[name for name,d in prior.items() if sha(root/name)!=d]
    atomic(val/'completion.json',dict(status='PASS' if not failed and not changed else 'FAIL',cases=len(done),failed=failed,
        protected_files=len(prior),changed_files=changed,script_sha256=args.script_sha,training=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('worker','run','analyze'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--index',type=int);args=p.parse_args()
    if args.command=='worker':worker(args)
    elif args.command=='run':run(args)
    else:analyze(args)
if __name__=='__main__':main()
