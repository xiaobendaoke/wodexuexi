"""Finite one-transfer teacher space, with public decisions sealed before truth.

This stage only collects counterfactual labels and an optimistic capacity screen;
common added software costs are assumptions, never measured deployment fees.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

FILES=('paths.py','scripts/analysis/sf_hlm_action_teacher.py','tests/test_sf_hlm_action_teacher.py',
       'docs/plans/sf_hlm_learning_feasibility_20261016.md')
COST_MS=(0.,5.,10.,20.)
CONTROLS=('G','stop','random_one','heuristic_one','predicted_best','oracle')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def normal(x):return json.loads(json.dumps(x,default=serial,allow_nan=False))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=serial,allow_nan=False)+'\n');tmp.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def seed_for(*x):return int.from_bytes(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).digest()[:4],'big')


def clone_branch(base):
    child=copy.copy(base);child.current=base.current.copy();child.seen=set(base.seen);child.history=copy.deepcopy(base.history)
    child.best=copy.deepcopy(base.best);child.current_predicted=copy.deepcopy(base.current_predicted)
    child.assignment=tuple(base.assignment);child.current_assignment=tuple(base.current_assignment)
    if getattr(base,'_mask_cache',None) is not None:child._mask_cache=base._mask_cache.copy()
    return child


def oracle_action(scores):return max(sorted(scores),key=lambda a:scores[a])
def random_score(scores,actions):return sum(scores[a] for a in actions)/len(actions) if actions else scores[0]


def signature(base):
    state=dict(current=base.current,seen=sorted(base.seen),history=base.history,assignment=base.assignment,
        current_assignment=base.current_assignment,best=base.best,current_predicted=base.current_predicted,calls=base.calls,done=base.done)
    return hashlib.sha256(json.dumps(normal(state),sort_keys=True).encode()).hexdigest()


def archive(a):return load('action_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def folder(a,identity):return a.results_root/'tasks'/('oracle_'+'_'.join(map(str,identity)))


def init(a,f):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Preserve old protocol namespace')
    parent=read(a.runtime_results/'protocol.json');p0=read(a.state_root/'protocol.json')
    if parent['smoke'] or read(a.runtime_results/'status.json')['status']!='COMPLETED':raise ValueError('Runtime screen must be complete before new collection')
    identities=p0['identities']
    if a.smoke:identities=[i for i in identities if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    inputs=parent['input_files'].copy();inputs[str(a.runtime_results/'protocol.json')]=sha(a.runtime_results/'protocol.json')
    inputs.update({str(a.runtime_release/rel):digest for rel,digest in parent['files'].items()})
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Frozen input/source changed: '+path)
    p=dict(name='sf_hlm_action_teacher_20261017_v1',smoke=a.smoke,purpose='verification' if a.smoke else 'development',
        identities=identities,states=[s for s in parent['states'] if s['identity'] in identities],
        jobs=[list(z) for z in sorted({tuple(i[:2]) for i in identities})],input_files=inputs,
        files={rel:sha(a.release_root/rel) for rel in FILES},cost_hypotheses_ms=list(COST_MS),
        controls=list(CONTROLS),online_matching_limit=2,initial_feature_budget=16,repair_budget=128,energy_ratio=1.05,
        no_model_training=True,no_actual_policy_cost_measurement=True,state_delay_applied=False,
        public_before_private=True,uniform_random='exact expected score over all legal nonSTOP moves, including rejected moves',
        prediction_retention='original predicted best objective[:2] and energy cap retained',
        capacity_gate=dict(gain_vs_G=.005,positive_blocks=4),goal_complete=False,
        boundary='finite-state finite-action immediate capacity; assumed common costs do not establish actual deployment gain')
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',states=len(identities),updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=len(identities),protocol_sha256=sha(a.results_root/'protocol.json'))),flush=True)


def verify(a,p,external=False):
    for rel,digest in p['files'].items():
        if sha(a.release_root/rel)!=digest:raise ValueError('New frozen source changed')
    if external:
        for path,digest in p['input_files'].items():
            if sha(path)!=digest:raise ValueError('Ancestor input changed: '+path)


def completed(a,p,identity):
    path=folder(a,identity)/'result.json'
    if not path.exists():return False
    rec=read(path)
    if rec['status']!='PASS' or rec['identity']!=identity or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Changed oracle task identity')
    for name,digest in rec['files'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Changed oracle task file')
    return True


def collect(a,p,f,identity,runtime,fast):
    import numpy as np
    from utils.canonical_experiment import digest
    from utils.sf_hlm_matching import allocation_counts
    from utils.snapshot_evaluator import evaluate
    _,_,runner,models,lower,basis=runtime;record=next(s for s in p['states'] if s['identity']==identity)
    for key in ('state','benchmark'):
        if sha(record[key])!=record[key+'_sha256']:raise ValueError('State/reference changed')
    meta=read(record['state']);old=read(record['benchmark']);inp=read(meta['input'])
    for path,checksum in meta['files'].items():
        if sha(path)!=checksum:raise ValueError('State bytes changed')
    env,obs=pickle.loads(Path(meta['observed']).read_bytes());_,_,requests,rows=runner.base.prepare(env)
    expected=inp['rows']
    for row in expected:row['options']={int(k):v for k,v in row['options'].items()}
    if digest(rows)!=digest(expected) or [q[2] for q in requests]!=inp['request_ids']:raise ValueError('Observed reconstruction mismatch')
    fp=runner.base.policy_fingerprint([models['upper'],lower])
    if fp!=old['model_fingerprint']:raise ValueError('Frozen model fingerprint changed')
    old_g=next(q for q in old['results'] if q['method']=='G')
    with fast.FastRuntime() as rt:
        greedy,_=rt.greedy(rows);g,gd=rt.common_repair(rows,greedy,128)
        if list(g)!=old_g['assignment']:raise ValueError('Public G assignment changed')
        base=rt.search_class(rows,num_uavs=5,matching_budget=16,base_repair_budget=128)
        original_signature=signature(base);mask=base.action_mask();actions=[int(x) for x in np.flatnonzero(mask[1:])+1]
        candidates=[dict(action=-1,transfer=None,assignment=list(g),predicted=normal(gd['chosen_predicted']),matching_calls=0,quota_reached=True)]
        for action in [0,*actions]:
            branch=clone_branch(base)
            if not branch.done:branch.step(action)
            candidates.append(dict(action=action,transfer=None if action==0 else list(base.actions[action]),
                assignment=list(branch.assignment),predicted=normal(branch.best),matching_calls=branch.calls,
                quota_reached=True if action==0 else bool(branch.history[-1]['feasible']) and branch.current.tolist()!=base.current.tolist(),
                executed_counts=allocation_counts(rows,branch.assignment,5).tolist(),incumbent_counts=branch.current.tolist()))
            if branch.calls>2 or branch.best['energy']>branch.cap+1e-7:raise ValueError('One-transfer branch budget/energy guard')
            if signature(base)!=original_signature:raise ValueError('Branch mutated the initial public search state')
        problem=base.problem;load=np.r_[np.maximum(base.current[:-1],1),1]
        risk=(((problem['base']+problem['work']*load[None,:])>problem['deadline'][:,None])&problem['legal']).sum(0)
        heuristic=min(actions,key=lambda k:(-risk[base.actions[k][0]],risk[base.actions[k][1]],base.actions[k][2],k)) if actions else 0
        predicted=min(candidates,key=lambda c:(tuple(c['predicted']['objective'][:2]),c['action']))['action']
        public=dict(identity=identity,rows=rows,request_ids=inp['request_ids'],initial_counts=base.current.tolist(),initial_assignment=list(base.assignment),
            greedy_assignment=list(g),initial_predicted=normal(base.best),reference_predicted=normal(base.reference),
            energy_cap=base.cap,legal_nonstop_actions=actions,heuristic_action=heuristic,predicted_best_action=predicted,
            candidates=candidates,offline_matching_calls=base.calls+len(actions),basis=basis,model_fingerprint=fp,
            no_private_outcome_fields=True)
    # No private execution occurs until all public candidates are sealed and patches restored.
    td=folder(a,identity);write(td/'public.json',public);public_hash=sha(td/'public.json')
    snapshot=pickle.loads(Path(meta['snapshot']).read_bytes())
    if snapshot.snapshot_hash!=inp['snapshot_hash'] or not np.array_equal(np.asarray(snapshot.motion),np.asarray(inp['motion'])):raise ValueError('Private snapshot identity differs')
    assignments={hashlib.sha256(json.dumps(c['assignment']).encode()).hexdigest():c['assignment'] for c in candidates}
    gkey=hashlib.sha256(json.dumps(list(g)).encode()).hexdigest();order=[gkey,*[k for k in sorted(assignments) if k!=gkey]]
    if len(order)>1:
        tail=np.random.default_rng(seed_for('action_teacher_order',identity)).permutation(order[1:]).tolist();order=[gkey,*tail]
    outcomes={}
    for key in order:
        actual=evaluate(snapshot,assignments[key]);metrics=actual['metrics']
        if int(metrics['generated_service_count'])!=old['generated'] or len(actual['requests'])!=old['admitted']:raise ValueError('Physical denominator changed')
        if key==gkey:
            if actual['requests']!=old_g['requests']:raise ValueError('Original G physical requests differ')
            for field,value in old_g['metrics'].items():
                if not math.isclose(float(metrics[field]),float(value),rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Original G physical metric differs')
        outcomes[key]=dict(assignment=assignments[key],requests=actual['requests'],metrics=metrics)
    if sha(td/'public.json')!=public_hash or normal(public)!=read(td/'public.json') or snapshot.snapshot_hash!=inp['snapshot_hash']:
        raise ValueError('Private evaluation mutated public or snapshot inputs')
    branch_map={str(c['action']):hashlib.sha256(json.dumps(c['assignment']).encode()).hexdigest() for c in candidates}
    write(td/'labels.json',dict(identity=identity,generated=old['generated'],admitted=old['admitted'],public_sha256=public_hash,
        outcomes=outcomes,action_outcome=branch_map,evaluation_order=order,snapshot_hash=snapshot.snapshot_hash,
        state_sha256=record['state_sha256'],private_truth_for_labels_only=True))
    if fp!=runner.base.policy_fingerprint([models['upper'],lower]):raise ValueError('Frozen model changed')
    write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=sha(a.results_root/'protocol.json'),
        files={name:sha(td/name) for name in ('public.json','labels.json')},branches=len(candidates),unique_physical_outcomes=len(outcomes),offline_matching_calls=public['offline_matching_calls']))


def worker(a,p,f):
    verify(a,p);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unknown source/block')
    parent=f.parent_args(a);res=load('action_worker_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(parent.resources,task=['action_teacher',*job],kind='environment'):
            runtime=f.boot(a,job[1]);fast=load('action_fast_runtime',a.runtime_release/'utils/sf_hlm_fast_runtime.py')
            for n,identity in enumerate(i for i in p['identities'] if i[:2]==job):
                if not completed(a,p,identity):collect(a,p,f,identity,runtime,fast)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=identity,updated_unix=time.time()))


def analyze(a,p,f):
    core=f.core(argparse.Namespace(release_root=a.feasibility_release));states=[];branches=unique=solves=0
    for identity in p['identities']:
        if not completed(a,p,identity):raise ValueError('Incomplete action space')
        td=folder(a,identity);public=read(td/'public.json');labels=read(td/'labels.json');rec=read(td/'result.json')
        branches+=rec['branches'];unique+=rec['unique_physical_outcomes'];solves+=rec['offline_matching_calls'];values={}
        for delay in COST_MS:
            score={int(action):core.fee_count(labels['outcomes'][key]['requests'],delay/1000) for action,key in labels['action_outcome'].items()}
            best=oracle_action(score)
            values[str(delay)]=dict(G=score[-1],stop=score[0],random_one=random_score(score,public['legal_nonstop_actions']),
                heuristic_one=score[public['heuristic_action']],predicted_best=score[public['predicted_best_action']],oracle=score[best],oracle_action=best)
        states.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],values=values))
    subsets={}
    for source in ('all','greedy','learned'):
        sr=[s for s in states if source=='all' or s['identity'][0]==source];blocks={}
        for block in sorted({s['identity'][1] for s in sr}):
            br=[s for s in sr if s['identity'][1]==block];den=sum(s['generated'] for s in br)
            blocks[str(block)]=dict(generated=den,states=len(br),rates={str(delay):{m:sum(s['values'][str(delay)][m] for s in br)/den for m in CONTROLS} for delay in COST_MS})
        groups={str(delay):{m:f.paired([b['rates'][str(delay)][m] for b in blocks.values()]) for m in CONTROLS} for delay in COST_MS}
        comparisons={str(delay):{m+'-G':f.paired([b['rates'][str(delay)][m]-b['rates'][str(delay)]['G'] for b in blocks.values()]) for m in CONTROLS if m!='G'} for delay in COST_MS}
        subsets[source]=dict(blocks=blocks,groups=groups,comparisons=comparisons)
    capacity={str(c):subsets['all']['comparisons'][str(c)]['oracle-G']['mean']>=.005 and subsets['all']['comparisons'][str(c)]['oracle-G']['positive']>=4 for c in COST_MS}
    result=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS',protocol_sha256=sha(a.results_root/'protocol.json'),states=len(states),branches=branches,
        unique_physical_outcomes=unique,offline_matching_calls=solves,subsets=subsets,
        decision=dict(capacity_by_cost=capacity,space_for_cheap_probe=any(capacity.values()),goal_complete=False,not_actual_cost_benefit=True),
        no_training=True,state_delay_applied=False,no_actual_policy_cost_measurement=True)
    write(a.results_root/'analysis/summary.json',result);write(a.results_root/'analysis/states.json',states)
    write(a.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(states),updated_unix=time.time()))
    print(json.dumps(dict(status=result['status'],states=len(states),decision=result['decision'])),flush=True)


def command(a,job):
    cmd=[sys.executable,str(a.release_root/FILES[1]),'worker']
    for key in ('release_root','runtime_release','runtime_results','feasibility_release','parent_launch','state_root','results_root','validation_root'):
        cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
    return cmd+['--job',json.dumps(job,separators=(',',':'))]


def pipeline(a,p,f):
    parent=f.parent_args(a);profile=read(parent.resources/'profile.json');os.sched_setaffinity(0,set(profile['cpus'][:profile['slots']]))
    res=load('action_controller_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(a.results_root/'controller.lock'):
        verify(a,p,external=True)
        def run(job):
            path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as log:out=subprocess.run(command(a,job),cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
            if out.returncode:raise RuntimeError('Oracle worker failed: '+str(job))
        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures=[pool.submit(run,j) for j in p['jobs']]
                for future in as_completed(futures):
                    future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed=sum(completed(a,p,i) for i in p['identities']),total=len(p['identities']),updated_unix=time.time()))
            analyze(a,p,f)
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('init','pipeline','worker','analyze'))
    for key in ('release_root','runtime_release','runtime_results','feasibility_release','parent_launch','state_root','results_root','validation_root'):
        ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('All matching, physics and scientific analysis run remotely')
    f=archive(a)
    if a.mode=='init':init(a,f)
    else:
        p=read(a.results_root/'protocol.json')
        if a.mode=='worker':worker(a,p,f)
        elif a.mode=='pipeline':pipeline(a,p,f)
        else:verify(a,p);analyze(a,p,f)


if __name__=='__main__':main()
