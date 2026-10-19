"""Training-only public energy-price calibration and held-out snapshot replay."""
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

FILES=('paths.py','scripts/analysis/sf_hlm_energy_price.py','utils/sf_hlm_energy_price.py','utils/sf_hlm_calibrated_matching.py',
    'utils/sf_hlm_action_ranker.py','utils/sf_hlm_selector.py','tests/test_sf_hlm_energy_price.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=('release_root','base_results','gcap_results','base_validation','teacher_results','training_root','calibration_root',
    'results_root','feasibility_release','parent_launch','validation_root','resource_module','resources')
METHODS=('G','old_pool_fee','binary','smooth','gcap_binary','gcap_smooth','binary_price','smooth_price','probability_price')
NONLEARNING=METHODS[:-1]
COSTS=(0.,5.,10.,20.)
OBJECTIVES=('binary','smooth','probability')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=serial,allow_nan=False)+'\n');tmp.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def helpers(a):
    return (load('price_rule',a.release_root/'utils/sf_hlm_energy_price.py'),load('price_matching',a.release_root/'utils/sf_hlm_calibrated_matching.py'),load('price_observed_rows',a.release_root/'utils/sf_hlm_action_ranker.py'))
def archive(a):return load('price_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def folded(a,b,w):return a.calibration_root/'folds'/f'block_{b}_workload_{w}'
def fold_list(a):return load('price_splits',a.release_root/'utils/sf_hlm_selector.py').folds(read(a.calibration_root/'protocol.json')['identities'])
def task_dir(root,identity):return Path(root)/'tasks'/('matched_'+'_'.join(map(str,identity)))
def output_dir(a,identity):return a.results_root/'tasks'/('priced_'+'_'.join(map(str,identity)))
def source(a,p):
    for file,digest in p['files'].items():
        if sha(a.release_root/file)!=digest:raise ValueError('Frozen price source changed')

def prepare(a):
    if (a.calibration_root/'protocol.json').exists():raise FileExistsError('Immutable calibration namespace')
    base=read(a.base_results/'protocol.json');gc=read(a.gcap_results/'protocol.json')
    if base['smoke'] or base.get('g_anchored_energy',False) or not gc.get('g_anchored_energy',False):raise ValueError('Wrong reference protocol')
    for root,label in ((a.base_results,'legacy_cap'),(a.gcap_results,'gcap')):
        audit=read(a.base_validation/(label+'_raw_record_audit.json'))
        if audit['status']!='PASS' or audit['summary_sha256']!=sha(root/'analysis/summary.json'):raise ValueError('Unverified reference results')
    tp=read(a.training_root/'protocol.json');completion=read(a.training_root/'completion.json');inputs={}
    for path in (a.base_results/'protocol.json',a.gcap_results/'protocol.json',a.training_root/'protocol.json',a.training_root/'completion.json'):
        inputs[str(path)]=sha(path)
    inputs.update(completion['files'])
    for identity in base['identities']:
        for root in (a.base_results,a.gcap_results):
            td=task_dir(root,identity);rec=read(td/'result.json')
            if rec['identity']!=identity or rec['status']!='PASS' or rec['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Reference state receipt')
            if sha(td/'public.json')!=rec['files']['public.json']:raise ValueError('Reference public data changed')
            inputs[str(td/'public.json')]=sha(td/'public.json')
        td=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)));rec=read(td/'result.json')
        if sha(td/'public.json')!=rec['files']['public.json']:raise ValueError('Observed rows changed')
        inputs[str(td/'public.json')]=sha(td/'public.json')
    for file,digest in inputs.items():
        if sha(file)!=digest:raise ValueError('Price calibration source changed')
    p=dict(name='sf_hlm_training_public_energy_price_20261017_v1',identities=base['identities'],states=base['states'],
        files={file:sha(a.release_root/file) for file in FILES},input_files=inputs,training_protocol_sha256=sha(a.training_root/'protocol.json'),
        split='25 leave-one-block-and-workload folds, 256 calibration states/16 test states/128 excluded',
        target=1.05,nominal_cost_ms=10.,model_training_new=False,private_physics_for_calibration=False,
        calibration='minimum nonnegative public score/energy crossing satisfying each of four training block budgets',
        methods=list(METHODS),nonlearning=list(NONLEARNING),cost_ms=list(COSTS),independent_confirmation=False,goal_complete=False)
    write(a.calibration_root/'protocol.json',p);write(a.calibration_root/'status.json',dict(status='PREPARED',states=400,updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',inputs=len(inputs),new_model_training=False)),flush=True)

def summarize_candidates(candidates):
    return [dict(action=c['action'],assignment=c['assignment'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates if c['accepted']]

def calibrate_fold(a):
    p=read(a.calibration_root/'protocol.json');source(a,p);rule,cm,rank=helpers(a);fold=next(f for f in fold_list(a) if f['block']==a.block and f['workload']==a.workload)
    folder=folded(a,a.block,a.workload)
    if (folder/'result.json').exists():raise FileExistsError('Completed price calibration is immutable')
    model_path=a.training_root/'folds'/f'block_{a.block}_workload_{a.workload}'/'model.pkl'
    if sha(model_path)!=p['input_files'][str(model_path)]:raise ValueError('Frozen request model changed')
    bundle=pickle.loads(model_path.read_bytes())
    if bundle['fold']!=fold or bundle['protocol_sha256']!=p['training_protocol_sha256']:raise ValueError('Calibration model sees held-out identities')
    model=bundle['model'];cases={key:[] for key in OBJECTIVES}
    for si in fold['train']:
        identity=p['identities'][si];td=task_dir(a.base_results,identity);pub=read(td/'public.json')
        observed=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)))/'public.json'
        if sha(td/'public.json')!=p['input_files'][str(td/'public.json')] or sha(observed)!=p['input_files'][str(observed)]:raise ValueError('Public training source changed')
        safe=rank.public_payload(read(observed));rows=safe['rows'];g=safe['greedy_assignment'];cap=pub['energy_cap']
        for objective in ('binary','smooth'):
            cases[objective].append(dict(state_index=si,identity=identity,block=identity[1],candidates=summarize_candidates(pub['candidates'][objective])))
        # All probability candidates are regenerated with THIS fold's model.
        from utils.deadline_repair import assess
        from utils.sf_hlm_matching import problem_from_rows
        problem=problem_from_rows(rows,5);gd=assess(rows,g);score=cm.score_assignment(rows,g,10.,'probability',model,problem=problem)
        candidates=[dict(action=-1,assignment=g,score=score,energy=gd['energy'])];rejected=[]
        for action,counts in pub['quotas']:
            result=cm.solve(rows,counts,10.,'probability',model,problem=problem)
            if result is None:rejected.append(dict(action=action,reason='infeasible'));continue
            if result['predicted']['energy']>cap+1e-7:rejected.append(dict(action=action,reason='legacy_predicted_energy_cap'));continue
            candidates.append(dict(action=action,assignment=result['assignment'],score=result['expected_success'],energy=result['predicted']['energy']))
        cases['probability'].append(dict(state_index=si,identity=identity,block=identity[1],candidates=candidates,rejected=rejected))
    prices={objective:rule.calibrate(rows,p['target']) for objective,rows in cases.items()}
    write(folder/'cases.json',cases);write(folder/'prices.json',dict(fold=fold,model_sha256=sha(model_path),protocol_sha256=sha(a.calibration_root/'protocol.json'),prices=prices))
    write(folder/'result.json',dict(status='PASS',fold=fold,model_sha256=sha(model_path),protocol_sha256=sha(a.calibration_root/'protocol.json'),
        files={file:sha(folder/file) for file in ('cases.json','prices.json')},cpu=sorted(os.sched_getaffinity(0)),no_private_calibration_labels=True))
    print(json.dumps(dict(status='CALIBRATED',block=a.block,workload=a.workload,prices={key:row['price'] for key,row in prices.items()})),flush=True)

def command(a,mode):
    cmd=[sys.executable,str(a.release_root/FILES[1]),mode]
    for key in ARGS:cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
    return cmd

def calibrate_pipeline(a):
    p=read(a.calibration_root/'protocol.json');source(a,p);res=load('price_calibration_resources',a.resource_module)
    def run(fold):
        cmd=command(a,'fold')+['--block',str(fold['block']),'--workload',str(fold['workload'])]
        path=a.calibration_root/'worker_logs'/f"{fold['block']}_{fold['workload']}.log";path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:out=subprocess.run(cmd,cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
        if out.returncode:raise RuntimeError('Price calibration failed '+str(fold['block'])+'/'+str(fold['workload']))
    with res.task_lock(a.calibration_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                pending=[pool.submit(run,f) for f in fold_list(a)]
                for n,future in enumerate(as_completed(pending),1):future.result();write(a.calibration_root/'status.json',dict(status='RUNNING',completed_folds=n,total_folds=25,updated_unix=time.time()))
            files={}
            for fold in fold_list(a):
                folder=folded(a,fold['block'],fold['workload']);record=read(folder/'result.json')
                if record['status']!='PASS' or record['fold']!=fold or record['protocol_sha256']!=sha(a.calibration_root/'protocol.json'):raise ValueError('Calibration receipt mismatch')
                files[str(folder/'result.json')]=sha(folder/'result.json')
                for name,digest in record['files'].items():
                    if sha(folder/name)!=digest:raise ValueError('Calibration artifact changed')
                    files[str(folder/name)]=digest
            write(a.calibration_root/'completion.json',dict(status='COMPLETED',folds=25,files=files,protocol_sha256=sha(a.calibration_root/'protocol.json'),private_labels_used=False))
            write(a.calibration_root/'status.json',dict(status='COMPLETED',folds=25,updated_unix=time.time()))
        except BaseException as error:
            write(a.calibration_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise

def init_eval(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable price evaluation namespace')
    cp=read(a.calibration_root/'protocol.json');source(a,cp);completion=read(a.calibration_root/'completion.json')
    if completion['status']!='COMPLETED' or completion['protocol_sha256']!=sha(a.calibration_root/'protocol.json'):raise ValueError('Calibration incomplete')
    for file,digest in completion['files'].items():
        if sha(file)!=digest:raise ValueError('Frozen price artifact changed')
    identities=cp['identities']
    if a.smoke:identities=[i for i in identities if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    p=dict(name='sf_hlm_energy_price_replay_20261017_v1',smoke=a.smoke,identities=identities,
        states=[s for s in cp['states'] if s['identity'] in identities],jobs=[list(z) for z in sorted({tuple(i[:2]) for i in identities})],
        files=cp['files'],calibration_protocol_sha256=sha(a.calibration_root/'protocol.json'),calibration_completion_sha256=sha(a.calibration_root/'completion.json'),
        methods=list(METHODS),nonlearning=list(NONLEARNING),cost_ms=list(COSTS),nominal_cost_ms=10.,
        offline_public_computation_reused=True,online_matching_fees_still_required=True,native_costs_measured=False,goal_complete=False,independent_confirmation=False)
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',states=len(identities),updated_unix=time.time()))

def completed(a,p,identity):
    path=output_dir(a,identity)/'result.json'
    if not path.exists():return False
    rec=read(path)
    if rec['status']!='PASS' or rec['identity']!=identity or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Price evaluation receipt mismatch')
    for file,digest in rec['files'].items():
        if sha(path.parent/file)!=digest:raise ValueError('Price evaluation artifact changed')
    return True

def collect(a,p,identity,runtime):
    from utils.snapshot_evaluator import evaluate
    rule,_,_=helpers(a);cp=read(a.calibration_root/'protocol.json');td=task_dir(a.base_results,identity);gd=task_dir(a.gcap_results,identity)
    for folder in (td,gd):
        if sha(folder/'public.json')!=cp['input_files'][str(folder/'public.json')]:raise ValueError('Held-out public reference changed')
    pub=read(td/'public.json');gc=read(gd/'public.json');folder=folded(a,identity[1],identity[2]);rec=read(folder/'result.json')
    if sha(folder/'result.json')!=read(a.calibration_root/'completion.json')['files'][str(folder/'result.json')]:raise ValueError('Frozen price receipt changed')
    for file,digest in rec['files'].items():
        if sha(folder/file)!=digest:raise ValueError('Frozen prices changed')
    prices=read(folder/'prices.json');fold=prices['fold'];test_index=cp['identities'].index(identity)
    if prices['protocol_sha256']!=sha(a.calibration_root/'protocol.json') or rec['protocol_sha256']!=prices['protocol_sha256']:raise ValueError('Price protocol changed')
    if test_index not in fold['test'] or test_index in fold['train'] or prices['model_sha256']!=pub['model_sha256']:raise ValueError('Price or model sees held-out identity')
    chosen={name:pub['selected'][name] for name in ('G','old_pool_fee','binary','smooth')}
    chosen.update(gcap_binary=gc['selected']['binary'],gcap_smooth=gc['selected']['smooth']);decision={}
    for objective in OBJECTIVES:
        candidates=[c for c in pub['candidates'][objective] if c['accepted']];simple=summarize_candidates(candidates);parameters=prices['prices'][objective]
        index=rule.choose(simple,parameters['price'],parameters['normalizer']);c=candidates[index];name=objective+'_price'
        chosen[name]=dict(action=c['action'],assignment=c['assignment'],predicted=c['predicted'],matching_calls=pub['selected'][objective]['matching_calls'])
        decision[name]=dict(candidate_index=index,price=parameters['price'],normalizer=parameters['normalizer'],expected_success=c['expected_success'],
            utility=c['expected_success']-parameters['price']*c['predicted']['energy']/parameters['normalizer'])
    _,_,runner,models,lower,basis=runtime;fp=runner.base.policy_fingerprint([models['upper'],lower])
    if fp!=pub['model_fingerprint']:raise ValueError('Foundation fingerprint changed')
    out=output_dir(a,identity);write(out/'public.json',dict(identity=identity,base_public_sha256=sha(td/'public.json'),gcap_public_sha256=sha(gd/'public.json'),
        price_receipt_sha256=sha(folder/'result.json'),price_sha256=sha(folder/'prices.json'),selected=chosen,decision=decision,model_fingerprint=fp,
        online_candidates_must_be_recomputed=True,no_actual_outcomes_used_for_decision=True))
    public_hash=sha(out/'public.json');entry=next(s for s in p['states'] if s['identity']==identity)
    if sha(entry['state'])!=entry['state_sha256']:raise ValueError('Frozen snapshot manifest changed')
    meta=read(entry['state'])
    for path,digest in meta['files'].items():
        if sha(path)!=digest:raise ValueError('Frozen snapshot changed')
    snapshot=pickle.loads(Path(meta['snapshot']).read_bytes());assignments={hashlib.sha256(json.dumps(c['assignment']).encode()).hexdigest():c['assignment'] for c in chosen.values()};outcomes={};base_labels=read(td/'labels.json');gcap_labels=read(gd/'labels.json')
    for key,assignment in sorted(assignments.items()):
        actual=evaluate(snapshot,assignment)
        if int(actual['metrics']['generated_service_count'])!=base_labels['generated'] or len(actual['requests'])!=base_labels['admitted']:raise ValueError('Physical denominator changed')
        outcomes[key]=dict(assignment=assignment,requests=actual['requests'],metrics=actual['metrics'])
    for method in ('G','old_pool_fee','binary','smooth','gcap_binary','gcap_smooth'):
        source_labels=gcap_labels if method.startswith('gcap_') else base_labels;source_name=method.removeprefix('gcap_');key=hashlib.sha256(json.dumps(chosen[method]['assignment']).encode()).hexdigest();old=source_labels['outcomes'][source_labels['methods'][source_name]]
        if outcomes[key]['requests']!=old['requests']:raise ValueError('Archived physical reference differs')
        for name,value in old['metrics'].items():
            if not math.isclose(outcomes[key]['metrics'][name],value,rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Archived physical metric differs')
    if sha(out/'public.json')!=public_hash or fp!=runner.base.policy_fingerprint([models['upper'],lower]):raise ValueError('Private scoring mutated public model')
    write(out/'labels.json',dict(identity=identity,generated=base_labels['generated'],admitted=base_labels['admitted'],public_sha256=public_hash,outcomes=outcomes,
        methods={name:hashlib.sha256(json.dumps(c['assignment']).encode()).hexdigest() for name,c in chosen.items()},private_truth_for_labels_only=True))
    write(out/'result.json',dict(status='PASS',identity=identity,protocol_sha256=sha(a.results_root/'protocol.json'),files={file:sha(out/file) for file in ('public.json','labels.json')},physical_outcomes=len(outcomes)))

def worker(a,p):
    f=archive(a);res=load('price_eval_resources',a.resource_module);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unknown source/block worker')
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(a.resources,task=['energy_price',*job],kind='environment'):
            runtime=f.boot(a,job[1])
            for n,identity in enumerate(i for i in p['identities'] if i[:2]==job):
                if not completed(a,p,identity):collect(a,p,identity,runtime)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=identity,updated_unix=time.time()))

def aggregate(a,p):
    f=archive(a);states=[]
    for identity in p['identities']:
        if not completed(a,p,identity):raise ValueError('Incomplete priced evaluation')
        folder=output_dir(a,identity);pub=read(folder/'public.json');lab=read(folder/'labels.json');outcomes={name:lab['outcomes'][lab['methods'][name]] for name in METHODS}
        states.append(dict(identity=identity,generated=lab['generated'],admitted=lab['admitted'],values={str(c):{name:sum(q['latency']+c/1000<=q['deadline'] for q in item['requests']) for name,item in outcomes.items()} for c in COSTS},
            metrics={name:item['metrics'] for name,item in outcomes.items()},matching_calls={name:pub['selected'][name]['matching_calls'] for name in METHODS}))
    subsets={}
    for source_name in ('all','greedy','learned'):
        rows=[s for s in states if source_name=='all' or s['identity'][0]==source_name];by_cost={}
        for cost in COSTS:
            blocks={}
            for block in sorted({s['identity'][1] for s in rows}):
                ss=[s for s in rows if s['identity'][1]==block];den=sum(s['generated'] for s in ss);admitted=sum(s['admitted'] for s in ss)
                metrics={name:{key:sum(s['metrics'][name][key] for s in ss)/len(ss) for key in ('uav_fleet_total_energy','fairness','mbs_load_ratio')} for name in METHODS}
                blocks[str(block)]=dict(generated=den,rates={name:sum(s['values'][str(cost)][name] for s in ss)/den for name in METHODS},metrics=metrics,
                    energy_ratio_to_G={name:metrics[name]['uav_fleet_total_energy']/metrics['G']['uav_fleet_total_energy'] for name in METHODS},
                    mbs_admitted_fraction={name:sum(s['metrics'][name]['service_offloads_mbs'] for s in ss)/max(admitted,1) for name in METHODS})
            comparisons={name+'-G':f.paired([b['rates'][name]-b['rates']['G'] for b in blocks.values()]) for name in METHODS if name!='G'}
            for ref in NONLEARNING:
                if ref!='G':comparisons['probability_price-'+ref]=f.paired([b['rates']['probability_price']-b['rates'][ref] for b in blocks.values()])
            by_cost[str(cost)]=dict(blocks=blocks,groups={name:f.paired([b['rates'][name] for b in blocks.values()]) for name in METHODS},comparisons=comparisons)
        subsets[source_name]=by_cost
    main=subsets['all']['10.0'];best=max(NONLEARNING,key=lambda n:main['groups'][n]['mean']);gain=main['comparisons']['probability_price-G'];inc=main['comparisons']['probability_price-'+best] if best!='G' else gain
    energy=sum(b['energy_ratio_to_G']['probability_price'] for b in main['blocks'].values())/len(main['blocks']);passed=gain['mean']>=.005 and gain['positive']>=4 and inc['mean']>=.001 and energy<=1.05
    out=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS_DEVELOPMENT_REPLAY',states=len(states),protocol_sha256=sha(a.results_root/'protocol.json'),subsets=subsets,
        decision=dict(gain_vs_G=gain,best_nonlearning=best,learning_increment=inc,mean_block_energy_ratio=energy,optimistic_signal=passed,goal_complete=False,native_costs_not_measured=True,independent_confirmation=False))
    write(a.results_root/'analysis/summary.json',out);write(a.results_root/'analysis/states.json',states);write(a.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(states),updated_unix=time.time()));print(json.dumps(out['decision']),flush=True)

def evaluate_pipeline(a):
    p=read(a.results_root/'protocol.json');source(a,p);res=load('price_collection_resources',a.resource_module)
    def run(job):
        cmd=command(a,'worker')+['--job',json.dumps(job,separators=(',',':'))];path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:out=subprocess.run(cmd,cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
        if out.returncode:raise RuntimeError('Price collection failed '+str(job))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                pending=[pool.submit(run,j) for j in p['jobs']]
                for future in as_completed(pending):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed=sum(completed(a,p,i) for i in p['identities']),total=len(p['identities']),updated_unix=time.time()))
            with res.CpuLease(a.resources,task='price_aggregate'):aggregate(a,p)
        except BaseException as error:write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('prepare','fold','calibrate','init-eval','evaluate','worker','aggregate'))
    for key in ARGS:ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--block',type=int);ap.add_argument('--workload',type=int);ap.add_argument('--job');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Calibration, matching, physics and statistics run remotely')
    res=load('price_main_resources',a.resource_module)
    if a.mode=='calibrate':calibrate_pipeline(a)
    elif a.mode=='evaluate':evaluate_pipeline(a)
    elif a.mode=='worker':worker(a,read(a.results_root/'protocol.json'))
    else:
        with res.CpuLease(a.resources,task=['energy_price',a.mode,a.block,a.workload]):
            if a.mode=='prepare':prepare(a)
            elif a.mode=='fold':calibrate_fold(a)
            elif a.mode=='init-eval':init_eval(a)
            else:aggregate(a,read(a.results_root/'protocol.json'))

if __name__=='__main__':main()
