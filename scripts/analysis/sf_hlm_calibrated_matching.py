"""Cross-fit request-success matching; fresh frozen-state physical evaluation.

Training, matching and analysis are remote only. Common-time replay precedes
native timing and cannot establish deployability or closed-loop improvement.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

METHODS=('G','old_pool_fee','binary','smooth','probability')
COSTS=(0.,5.,10.,20.)
FILES=('paths.py','scripts/analysis/sf_hlm_calibrated_matching.py','utils/sf_hlm_calibrated_matching.py',
    'utils/sf_hlm_request_risk.py','utils/sf_hlm_action_ranker.py','utils/sf_hlm_selector.py',
    'tests/test_sf_hlm_calibrated_matching.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARG_PATHS=('release_root','risk_results','risk_validation','teacher_results','runtime_release','runtime_results',
    'feasibility_release','parent_launch','state_root','training_root','results_root','validation_root','resource_module','resources')

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
def modules(a):
    return (load('calibrated_public_solver',a.release_root/'utils/sf_hlm_calibrated_matching.py'),
        load('calibrated_public_pool',a.release_root/'utils/sf_hlm_request_risk.py'),
        load('calibrated_public_proposals',a.release_root/'utils/sf_hlm_action_ranker.py'))
def helper(a):return load('calibrated_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def folds(a):return load('calibrated_folds',a.release_root/'utils/sf_hlm_selector.py').folds(read(a.training_root/'protocol.json')['identities'])
def fold_dir(a,b,w):return a.training_root/'folds'/f'block_{b}_workload_{w}'
def task_dir(a,identity):return a.results_root/'tasks'/('matched_'+'_'.join(map(str,identity)))
def verify_source(a,p):
    for file,digest in p['files'].items():
        if sha(a.release_root/file)!=digest:raise ValueError('Frozen calibrated source changed')

def prepare(a):
    import numpy as np
    from utils.sf_hlm_matching import allocation_counts
    if (a.training_root/'protocol.json').exists():raise FileExistsError('Immutable training namespace')
    audit=read(a.risk_validation/'raw_record_audit.json');rp=read(a.risk_results/'protocol.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=sha(a.risk_results/'analysis/summary.json'):raise ValueError('Unverified parent request data')
    cm,pool,rank=modules(a);arc=np.load(a.risk_results/'training_data.npz');data={k:arc[k] for k in arc.files};layout=read(a.risk_results/'layout.json')
    p=dict(name='sf_hlm_quota_local_success_matching_20261017_v1',identities=rp['identities'],schema=cm.SCHEMA,
        columns=list(cm.LOCAL_COLUMNS),features=[rp['features'][i] for i in cm.LOCAL_COLUMNS],hgb=rp['hgb'],
        files={file:sha(a.release_root/file) for file in FILES},risk_protocol_sha256=sha(a.risk_results/'protocol.json'),
        risk_audit_sha256=sha(a.risk_validation/'raw_record_audit.json'),risk_data_sha256=sha(a.risk_results/'training_data.npz'),
        split='25 leave-block-and-workload-out folds',nominal_cost_ms=10.,common_replay_cost_ms=list(COSTS),
        objective='learned expected request successes in the conditional matching objective',energy_cap_ratio=1.05,
        independent_confirmation=False,goal_complete=False)
    write(a.training_root/'protocol.json',p);checked=0
    # Check all original request rows, not just a synthetic shape example.
    for si,state in enumerate(layout):
        td=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,state['identity'])));pub=read(td/'public.json');safe=rank.public_payload(pub)
        for cost,info in state['costs'].items():
            for c in info['candidates']:
                counts=allocation_counts(safe['rows'],c['assignment'],5);xx,index,_,_=cm.edge_features(safe['rows'],counts,float(cost))
                lookup={tuple(pair):value for pair,value in zip(index,xx)};selected=[]
                for i,(row,action) in enumerate(zip(safe['rows'],c['assignment'])):
                    target=row['options'][action]['target'];selected.append(lookup[i,5 if target=='mbs' else int(target)])
                if selected:
                    expected=data['x'][c['start']:c['stop']][:,cm.LOCAL_COLUMNS]
                    if not np.allclose(np.asarray(selected),expected,rtol=0,atol=1e-12):raise ValueError('Local edge features differ from training representation')
                checked+=c['stop']-c['start']
    output=a.training_root/'training_data.npz';np.savez_compressed(output,x=data['x'][:,cm.LOCAL_COLUMNS],y=data['y_actual'],weight=data['weight'],state_index=data['state_index'])
    write(a.training_root/'layout.json',layout);write(a.training_root/'prepared.json',dict(status='PASS',states=400,rows=len(data['x']),feature_parity_rows=checked,
        data_sha256=sha(output),layout_sha256=sha(a.training_root/'layout.json')))
    write(a.training_root/'status.json',dict(status='PREPARED',states=400,updated_unix=time.time()));print(json.dumps(dict(status='PREPARED',feature_parity_rows=checked)),flush=True)

def verify_training(a,p):
    verify_source(a,p);prep=read(a.training_root/'prepared.json')
    if sha(a.training_root/'training_data.npz')!=prep['data_sha256'] or sha(a.training_root/'layout.json')!=prep['layout_sha256']:raise ValueError('Training data changed')

def fit(a):
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    p=read(a.training_root/'protocol.json');verify_training(a,p);fold=next(f for f in folds(a) if f['block']==a.block and f['workload']==a.workload)
    folder=fold_dir(a,a.block,a.workload)
    if (folder/'result.json').exists():raise FileExistsError('Completed calibrated model is immutable')
    arc=np.load(a.training_root/'training_data.npz');data={k:arc[k] for k in arc.files};train=np.isin(data['state_index'],fold['train']);test=np.isin(data['state_index'],fold['test'])
    model=HistGradientBoostingClassifier(**p['hgb']).fit(data['x'][train],data['y'][train],sample_weight=data['weight'][train])
    versions=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__)
    bundle=dict(model=model,schema=p['schema'],features=p['features'],protocol_sha256=sha(a.training_root/'protocol.json'),fold=fold,versions=versions)
    folder.mkdir(parents=True,exist_ok=True);path=folder/'model.pkl';path.write_bytes(pickle.dumps(bundle,protocol=5));restored=pickle.loads(path.read_bytes())['model']
    pp=model.predict_proba(data['x'][test]);again=restored.predict_proba(data['x'][test])
    if not np.array_equal(pp,again):raise ValueError('Model serialization differs')
    np.savez_compressed(folder/'test_probabilities.npz',row_indices=np.flatnonzero(test),probability=pp[:,list(model.classes_).index(1.)])
    write(folder/'result.json',dict(status='PASS',fold=fold,protocol_sha256=sha(a.training_root/'protocol.json'),versions=versions,
        files={name:sha(folder/name) for name in ('model.pkl','test_probabilities.npz')},cpu=sorted(os.sched_getaffinity(0))))
    print(json.dumps(dict(status='FITTED',block=a.block,workload=a.workload)),flush=True)

def command(a,mode):
    cmd=[sys.executable,str(a.release_root/FILES[1]),mode]
    for key in ARG_PATHS:cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
    if a.training_release is not None:cmd+=['--training-release',str(a.training_release)]
    if a.g_anchored_energy:cmd+=['--g-anchored-energy']
    return cmd

def train(a):
    p=read(a.training_root/'protocol.json');verify_training(a,p);res=load('calibrated_training_resources',a.resource_module)
    def run(fold):
        cmd=command(a,'fit')+['--block',str(fold['block']),'--workload',str(fold['workload'])]
        path=a.training_root/'worker_logs'/f"{fold['block']}_{fold['workload']}.log";path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:out=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=a.release_root)
        if out.returncode:raise RuntimeError('Calibrated training failed '+str(fold['block'])+'/'+str(fold['workload']))
    with res.task_lock(a.training_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                jobs=[pool.submit(run,fold) for fold in folds(a)]
                for n,future in enumerate(as_completed(jobs),1):future.result();write(a.training_root/'status.json',dict(status='RUNNING',completed_folds=n,total_folds=25,updated_unix=time.time()))
            files={}
            for fold in folds(a):
                folder=fold_dir(a,fold['block'],fold['workload']);record=read(folder/'result.json')
                if record['fold']!=fold or record['status']!='PASS':raise ValueError('Training fold receipt differs')
                files[str(folder/'result.json')]=sha(folder/'result.json')
                for name,digest in record['files'].items():
                    if sha(folder/name)!=digest:raise ValueError('Training model changed')
                    files[str(folder/name)]=digest
            write(a.training_root/'completion.json',dict(status='COMPLETED',models=25,files=files,protocol_sha256=sha(a.training_root/'protocol.json')))
            write(a.training_root/'status.json',dict(status='COMPLETED',models=25,updated_unix=time.time()))
        except BaseException as error:
            write(a.training_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise

def init_eval(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable evaluation namespace')
    training=read(a.training_root/'completion.json');tp=read(a.training_root/'protocol.json')
    source_args=argparse.Namespace(**vars(a));source_args.release_root=a.training_release or a.release_root
    verify_training(source_args,tp)
    if training['status']!='COMPLETED':raise ValueError('Training incomplete')
    for file,digest in training['files'].items():
        if sha(file)!=digest:raise ValueError('Trained model artifact changed')
    parent=read(a.runtime_results/'protocol.json');identities=tp['identities']
    if a.smoke:identities=[i for i in identities if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    states=[s for s in parent['states'] if s['identity'] in identities]
    if len(states)!=len(identities):raise ValueError('Evaluation state population differs')
    p=dict(name='sf_hlm_calibrated_matching_replay_20261017_v1',smoke=a.smoke,identities=identities,states=states,
        jobs=[list(z) for z in sorted({tuple(i[:2]) for i in identities})],methods=list(METHODS),cost_ms=list(COSTS),nominal_cost_ms=10.,
        files={file:sha(a.release_root/file) for file in FILES},training_source_root=str(source_args.release_root),training_files=tp['files'],
        training_protocol_sha256=sha(a.training_root/'protocol.json'),training_completion_sha256=sha(a.training_root/'completion.json'),
        teacher_protocol_sha256=sha(a.teacher_results/'protocol.json'),max_conditional_matching_calls=10,
        selection='same public 9 quota set, conditional expected success + predicted energy cap; old_pool_fee retains archived branch guard',
        g_anchored_energy=a.g_anchored_energy,energy_cap_formula='min(legacy_cap,1.05*predicted_energy(G))' if a.g_anchored_energy else 'legacy_cap',
        truth_boundary='public decisions sealed before restoring private physics',no_native_costs=True,independent_confirmation=False,goal_complete=False)
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',states=len(states),updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',smoke=a.smoke,states=len(states))),flush=True)

def completed(a,p,identity):
    path=task_dir(a,identity)/'result.json'
    if not path.exists():return False
    rec=read(path)
    if rec['status']!='PASS' or rec['identity']!=identity or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Calibrated task identity changed')
    for name,digest in rec['files'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Calibrated task file changed')
    return True

def collect(a,p,identity,runtime,f,cm,pool,rank,fast):
    import numpy as np
    from utils.sf_hlm_matching import problem_from_rows
    from utils.snapshot_evaluator import evaluate
    source=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)));parent=read(source/'result.json');pub=read(source/'public.json')
    if parent['identity']!=identity or parent['status']!='PASS' or parent['protocol_sha256']!=p['teacher_protocol_sha256']:raise ValueError('Parent public identity changed')
    for name,digest in parent['files'].items():
        if sha(source/name)!=digest:raise ValueError('Parent action data changed')
    safe=rank.public_payload(pub);rows=safe['rows'];ids,fx=rank.features(safe,10.);proposals=pool.proposal_actions(ids,fx);quotas=cm.quota_set(safe['initial_counts'],proposals)
    model_path=fold_dir(a,identity[1],identity[2])/'model.pkl';bundle=pickle.loads(model_path.read_bytes());fold=bundle['fold']
    tc=read(a.training_root/'completion.json')
    if sha(a.training_root/'completion.json')!=p['training_completion_sha256'] or sha(model_path)!=tc['files'][str(model_path)]:raise ValueError('Frozen model bytes changed')
    if fold['block']!=identity[1] or fold['workload']!=identity[2] or bundle['schema']!=cm.SCHEMA or bundle['protocol_sha256']!=p['training_protocol_sha256']:raise ValueError('Wrong held-out conditional model')
    model=bundle['model'];_,_,runner,models,lower,basis=runtime;fp=runner.base.policy_fingerprint([models['upper'],lower])
    if fp!=pub['model_fingerprint']:raise ValueError('Frozen foundation changed')
    original_pool=pool.unique_pool(pub['candidates'],proposals);old_scores=[sum(t+.010<=r['deadline'] for t,r in zip(c['predicted']['latencies'],rows)) for c in original_pool]
    old_action=pool.choose([c['action'] for c in original_pool],old_scores);old_chosen=next(c for c in original_pool if c['action']==old_action)
    candidates={};selected={'G':dict(action=-1,assignment=safe['greedy_assignment'],predicted=next(c for c in pub['candidates'] if c['action']==-1)['predicted'],matching_calls=0),
        'old_pool_fee':dict(action=old_action,assignment=old_chosen['assignment'],predicted=old_chosen['predicted'],matching_calls=1+len(proposals))}
    with fast.FastRuntime() as rt:
        problem=problem_from_rows(rows,5)
        effective_cap=min(safe['energy_cap'],1.05*rt.assess(rows,safe['greedy_assignment'])['energy']) if p.get('g_anchored_energy',False) else safe['energy_cap']
        for objective in ('binary','smooth','probability'):
            scored=[];eligible=[]
            g=safe['greedy_assignment'];gd=rt.assess(rows,g);score=cm.score_assignment(rows,g,10.,objective,model,problem=problem)
            scored.append(dict(action=-1,assignment=list(g),predicted=normal(gd),expected_success=score,accepted=True,reason='G_fallback'));eligible.append(scored[-1])
            for action,counts in quotas:
                result=cm.solve(rows,counts,10.,objective,model,assess=rt.assess,problem=problem)
                if result is None:scored.append(dict(action=action,counts=counts,accepted=False,reason='infeasible'));continue
                accept=result['predicted']['energy']<=effective_cap+1e-7
                item=normal(dict(action=action,**result,accepted=accept,reason='accepted' if accept else 'predicted_energy_cap'))
                scored.append(item)
                if accept:eligible.append(item)
            best=cm.select([c['expected_success'] for c in eligible],[c['predicted']['energy'] for c in eligible]);chosen=eligible[best]
            candidates[objective]=scored;selected[objective]=dict(action=chosen['action'],assignment=chosen['assignment'],predicted=chosen['predicted'],
                expected_success=chosen['expected_success'],matching_calls=1+len(quotas))
    # No private result exists in the decision-generation context.
    td=task_dir(a,identity);public=dict(identity=identity,public_parent_sha256=sha(source/'public.json'),model_sha256=sha(model_path),
        proposals=proposals,quotas=quotas,legacy_energy_cap=safe['energy_cap'],energy_cap=effective_cap,selected=selected,candidates=candidates,
        prediction_contract='new success objective; archived predicted best retention intentionally replaced, energy cap shared',
        model_training_excludes_block=identity[1],model_training_excludes_workload=identity[2],model_fingerprint=fp)
    write(td/'public.json',public);public_hash=sha(td/'public.json')
    entry=next(s for s in p['states'] if s['identity']==identity)
    if sha(entry['state'])!=entry['state_sha256'] or sha(entry['benchmark'])!=entry['benchmark_sha256']:raise ValueError('State source changed')
    meta=read(entry['state']);benchmark=read(entry['benchmark'])
    for path,digest in meta['files'].items():
        if sha(path)!=digest:raise ValueError('Private frozen state changed')
    snapshot=pickle.loads(Path(meta['snapshot']).read_bytes());assignments={hashlib.sha256(json.dumps(c['assignment']).encode()).hexdigest():c['assignment'] for c in selected.values()};outcomes={};old_g=next(c for c in benchmark['results'] if c['method']=='G')
    for key in sorted(assignments):
        result=evaluate(snapshot,assignments[key]);metrics=result['metrics']
        if int(metrics['generated_service_count'])!=benchmark['generated'] or len(result['requests'])!=benchmark['admitted']:raise ValueError('Physical denominator changed')
        if assignments[key]==selected['G']['assignment']:
            if result['requests']!=old_g['requests']:raise ValueError('G physical replay differs')
            for field,value in old_g['metrics'].items():
                if not math.isclose(metrics[field],value,rel_tol=1e-10,abs_tol=1e-8):raise ValueError('G physical metric differs')
        outcomes[key]=dict(assignment=assignments[key],requests=result['requests'],metrics=metrics)
    if sha(td/'public.json')!=public_hash or fp!=runner.base.policy_fingerprint([models['upper'],lower]):raise ValueError('Truth mutated public/model')
    write(td/'labels.json',dict(identity=identity,generated=benchmark['generated'],admitted=benchmark['admitted'],public_sha256=public_hash,
        outcomes=outcomes,methods={name:hashlib.sha256(json.dumps(c['assignment']).encode()).hexdigest() for name,c in selected.items()},private_truth_for_labels_only=True))
    write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=sha(a.results_root/'protocol.json'),files={name:sha(td/name) for name in ('public.json','labels.json')},physical_outcomes=len(outcomes)))

def worker(a,p):
    f=helper(a);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unexpected worker identity')
    res=load('calibrated_collection_resources',a.resource_module)
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(a.resources,task=['calibrated_matching',*job],kind='environment'):
            runtime=f.boot(a,job[1]);cm,pool,rank=modules(a);fast=load('calibrated_fast_runtime',a.runtime_release/'utils/sf_hlm_fast_runtime.py')
            for n,identity in enumerate(i for i in p['identities'] if i[:2]==job):
                if not completed(a,p,identity):collect(a,p,identity,runtime,f,cm,pool,rank,fast)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=identity,updated_unix=time.time()))

def aggregate(a,p):
    f=helper(a);states=[]
    for identity in p['identities']:
        if not completed(a,p,identity):raise ValueError('Incomplete calibrated evaluation')
        td=task_dir(a,identity);pub=read(td/'public.json');label=read(td/'labels.json');values={}
        for cost in COSTS:
            values[str(cost)]={name:sum(r['latency']+cost/1000<=r['deadline'] for r in label['outcomes'][label['methods'][name]]['requests']) for name in METHODS}
        states.append(dict(identity=identity,generated=label['generated'],admitted=label['admitted'],values=values,
            metrics={name:label['outcomes'][label['methods'][name]]['metrics'] for name in METHODS},
            matching_calls={name:pub['selected'][name]['matching_calls'] for name in METHODS}))
    subsets={}
    for source in ('all','greedy','learned'):
        sr=[s for s in states if source=='all' or s['identity'][0]==source];by_cost={}
        for cost in COSTS:
            blocks={}
            for block in sorted({s['identity'][1] for s in sr}):
                rows=[s for s in sr if s['identity'][1]==block];den=sum(s['generated'] for s in rows);admitted=sum(s['admitted'] for s in rows)
                metrics={name:{key:sum(s['metrics'][name][key] for s in rows)/len(rows) for key in ('uav_fleet_total_energy','fairness','mbs_load_ratio')} for name in METHODS}
                energy_g=metrics['G']['uav_fleet_total_energy']
                blocks[str(block)]=dict(generated=den,rates={name:sum(s['values'][str(cost)][name] for s in rows)/den for name in METHODS},
                    metrics=metrics,energy_ratio_to_G={name:metrics[name]['uav_fleet_total_energy']/energy_g for name in METHODS},
                    mbs_admitted_fraction={name:sum(s['metrics'][name]['service_offloads_mbs'] for s in rows)/max(admitted,1) for name in METHODS})
            comparisons={name+'-G':f.paired([b['rates'][name]-b['rates']['G'] for b in blocks.values()]) for name in METHODS if name!='G'}
            for ref in ('old_pool_fee','binary','smooth'):comparisons['probability-'+ref]=f.paired([b['rates']['probability']-b['rates'][ref] for b in blocks.values()])
            by_cost[str(cost)]=dict(blocks=blocks,groups={name:f.paired([b['rates'][name] for b in blocks.values()]) for name in METHODS},comparisons=comparisons)
        subsets[source]=by_cost
    main=subsets['all']['10.0'];best=max(('G','old_pool_fee','binary','smooth'),key=lambda n:main['groups'][n]['mean']);gain=main['comparisons']['probability-G'];increment=main['comparisons']['probability-'+best] if best!='G' else gain
    energy_ratio=sum(b['energy_ratio_to_G']['probability'] for b in main['blocks'].values())/len(main['blocks'])
    passed=gain['mean']>=.005 and gain['positive']>=4 and increment['mean']>=.001 and energy_ratio<=1.05
    result=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS_DEVELOPMENT_REPLAY',states=len(states),protocol_sha256=sha(a.results_root/'protocol.json'),subsets=subsets,
        decision=dict(gain_vs_G=gain,best_nonlearning=best,learning_increment=increment,mean_block_energy_ratio=energy_ratio,
            optimistic_signal=passed,goal_complete=False,native_costs_not_measured=True,independent_confirmation=False))
    write(a.results_root/'analysis/summary.json',result);write(a.results_root/'analysis/states.json',states)
    write(a.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(states),updated_unix=time.time()));print(json.dumps(result['decision']),flush=True)

def evaluate_pipeline(a):
    p=read(a.results_root/'protocol.json');verify_source(a,p);res=load('calibrated_eval_resources',a.resource_module)
    def run(job):
        cmd=command(a,'worker')+['--job',json.dumps(job,separators=(',',':'))]
        path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:out=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=a.release_root)
        if out.returncode:raise RuntimeError('Calibrated collection failed '+str(job))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                pending=[pool.submit(run,job) for job in p['jobs']]
                for future in as_completed(pending):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed=sum(completed(a,p,i) for i in p['identities']),total=len(p['identities']),updated_unix=time.time()))
            with res.CpuLease(a.resources,task='calibrated_aggregate'):aggregate(a,p)
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('prepare','fit','train','init-eval','evaluate','worker','aggregate'))
    for key in ARG_PATHS:ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--training-release',type=Path);ap.add_argument('--g-anchored-energy',action='store_true')
    ap.add_argument('--block',type=int);ap.add_argument('--workload',type=int);ap.add_argument('--job');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Training and scientific computations run remotely')
    res=load('calibrated_main_resources',a.resource_module)
    if a.mode=='train':train(a)
    elif a.mode=='evaluate':evaluate_pipeline(a)
    elif a.mode=='worker':worker(a,read(a.results_root/'protocol.json'))
    else:
        with res.CpuLease(a.resources,task=['calibrated_matching',a.mode,a.block,a.workload]):
            if a.mode=='prepare':prepare(a)
            elif a.mode=='fit':fit(a)
            elif a.mode=='init-eval':init_eval(a)
            else:aggregate(a,read(a.results_root/'protocol.json'))

if __name__=='__main__':main()
