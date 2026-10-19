"""Paid candidate-pool capacity screen and fixed cross-fit request calibration.

Remote only. Replay cost scenarios are assumptions, not native software time.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

MODELS=('hgb_probability','hgb_residual')
COSTS=(0.,5.,10.,20.)
FILES=('paths.py','scripts/analysis/sf_hlm_request_risk.py','utils/sf_hlm_request_risk.py',
    'utils/sf_hlm_action_ranker.py','utils/sf_hlm_selector.py','tests/test_sf_hlm_request_risk.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=serial)+'\n');tmp.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def module(a):return load('request_risk_public',a.release_root/'utils/sf_hlm_request_risk.py')
def stats_module(a):return load('request_risk_statistics',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def folded(a,b,w):return a.results_root/'folds'/f'block_{b}_workload_{w}'
def folds(a,layout):return load('request_risk_folds',a.release_root/'utils/sf_hlm_selector.py').folds([s['identity'] for s in layout])

def block_scores(layout,rows,names,helper):
    out={}
    for source in ('all','greedy','learned'):
        by_cost={};indices=[i for i,s in enumerate(layout) if source=='all' or s['identity'][0]==source]
        for cost in COSTS:
            blocks={}
            for block in sorted({layout[i]['identity'][1] for i in indices}):
                ids=[i for i in indices if layout[i]['identity'][1]==block];den=sum(layout[i]['generated'] for i in ids)
                total={n:sum(rows[str(i)][str(cost)][n] for i in ids) for n in names}
                blocks[str(block)]=dict(generated=den,rates={k:v/den for k,v in total.items()})
            comparisons={n+'-G':helper.paired([b['rates'][n]-b['rates']['G'] for b in blocks.values()]) for n in names if n!='G'}
            for name in names:
                if name in MODELS or name=='oracle':
                    for ref in ('predicted_fee','predicted_lex','uniform_pool'):
                        comparisons[name+'-'+ref]=helper.paired([b['rates'][name]-b['rates'][ref] for b in blocks.values()])
            by_cost[str(cost)]=dict(blocks=blocks,groups={n:helper.paired([b['rates'][n] for b in blocks.values()]) for n in names},comparisons=comparisons)
        out[source]=by_cost
    return out

def prepare(a):
    import numpy as np
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Preserve completed development datasets')
    parent=read(a.teacher_results/'protocol.json');audit=read(a.teacher_validation/'raw_record_audit.json')
    if parent['smoke'] or audit['status']!='PASS' or audit['summary_sha256']!=sha(a.teacher_results/'analysis/summary.json'):raise ValueError('Unverified teacher collection')
    m=module(a);rank=load('request_proposal_features',a.release_root/'utils/sf_hlm_action_ranker.py')
    hp=dict(max_iter=64,learning_rate=.1,max_leaf_nodes=8,max_depth=3,min_samples_leaf=64,l2_regularization=.1,early_stopping=False,random_state=20261017)
    protocol=dict(name='sf_hlm_paid_candidate_request_risk_20261017_v1',identities=parent['identities'],models=list(MODELS),schema=m.SCHEMA,
        features=list(m.FEATURES),dimension=len(m.FEATURES),cost_ms=list(COSTS),main_cost_ms=10.,hgb=hp,teacher_results=str(a.teacher_results),
        teacher_protocol_sha256=sha(a.teacher_results/'protocol.json'),teacher_summary_sha256=sha(a.teacher_results/'analysis/summary.json'),
        teacher_audit_sha256=sha(a.teacher_validation/'raw_record_audit.json'),files={f:sha(a.release_root/f) for f in FILES},
        proposal='4 distinct pairs ranked by rescue-loss/safe/small amount/action, then fixed seed 20261017 sample 4 from remaining sorted legal actions',
        online_matching_limit=9,weight='One total weight per nonempty state over deduplicated assignments, requests and four cost scenarios',
        split='25 leave-one-block-and-workload folds, 256 train 16 test 128 excluded',no_native_costs=True,independent_confirmation=False,goal_complete=False)
    write(a.results_root/'protocol.json',protocol)
    xrows=[];ya=[];yp=[];weights=[];state_indices=[];layout=[];inputs={};offset=0;pool_rows={};matches=0;physical=0
    for si,identity in enumerate(protocol['identities']):
        td=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)));pub=read(td/'public.json');label=read(td/'labels.json');receipt=read(td/'result.json')
        if receipt['status']!='PASS' or receipt['identity']!=identity or receipt['protocol_sha256']!=protocol['teacher_protocol_sha256']:raise ValueError('Parent task identity')
        for file,digest in receipt['files'].items():
            if sha(td/file)!=digest:raise ValueError('Parent input changed')
            inputs[str(td/file)]=digest
        safe=rank.public_payload(pub);action_ids,proposal_x=rank.features(safe,10.)
        proposed=m.proposal_actions(action_ids,proposal_x);pool=m.unique_pool(pub['candidates'],proposed)
        if len(proposed)>8:raise ValueError('Online matching limit')
        matches+=1+len(proposed);physical+=len(pool);cost_layout={};state_scores={}
        lex=min(pool,key=lambda c:(tuple(c['predicted']['objective'][:2]),c['action']))['action']
        for cost in COSTS:
            candidates=[];actual={};predicted={}
            for c in pool:
                action=c['action'];requests=label['outcomes'][label['action_outcome'][str(action)]]['requests']
                if len(requests)!=len(safe['rows']):raise ValueError('Request label count')
                for row,assigned,request in zip(safe['rows'],c['assignment'],requests):
                    if row['ue_id']!=request['ue_id'] or row['source']!=request['source'] or assigned!=request['action']:raise ValueError('Request alignment')
                x,p0=m.candidate_features(safe['rows'],c['assignment'],cost,safe['energy_cap'])
                target=np.asarray([r['latency']+cost/1000<=r['deadline'] for r in requests],dtype=np.float64)
                if not np.array_equal(p0,np.asarray([t+cost/1000<=r['deadline'] for t,r in zip(c['predicted']['latencies'],safe['rows'])],dtype=np.float64)):raise ValueError('Public prediction reconstruction')
                actual[action]=int(target.sum());predicted[action]=int(p0.sum());n=len(x)
                candidates.append(dict(action=action,start=offset,stop=offset+n,assignment=c['assignment'],success=int(target.sum()),predicted_success=int(p0.sum()),
                    metrics=label['outcomes'][label['action_outcome'][str(action)]]['metrics']))
                if n:
                    xrows.append(x);ya.extend(target);yp.extend(p0);weights.extend([m.state_weight(len(pool),n,len(COSTS))]*n);state_indices.extend([si]*n)
                offset+=n
            ids=[c['action'] for c in pool];fee=m.choose(ids,[predicted[k] for k in ids]);oracle=m.choose(ids,[actual[k] for k in ids])
            controls=dict(G=actual[-1],predicted_lex=actual[lex],predicted_fee=actual[fee],uniform_pool=sum(actual.values())/len(actual),oracle=actual[oracle])
            cost_layout[str(cost)]=dict(candidates=candidates,controls=controls,control_actions=dict(G=-1,predicted_lex=lex,predicted_fee=fee,oracle=oracle))
            state_scores[str(cost)]=controls
        pool_rows[str(si)]=state_scores
        layout.append(dict(identity=identity,generated=label['generated'],admitted=label['admitted'],proposals=proposed,matching_calls=1+len(proposed),costs=cost_layout))
    x=np.concatenate(xrows) if xrows else np.empty((0,len(m.FEATURES)))
    state_index=np.asarray(state_indices,dtype=np.int64);weight=np.asarray(weights,dtype=np.float64)
    for si,state in enumerate(layout):
        if state['admitted'] and not np.isclose(weight[state_index==si].sum(),1.):raise ValueError('State training weight')
    path=a.results_root/'training_data.npz';np.savez_compressed(path,x=x,y_actual=np.asarray(ya),y_predictor=np.asarray(yp),weight=weight,state_index=state_index)
    write(a.results_root/'layout.json',layout);write(a.results_root/'inputs.json',inputs)
    write(a.results_root/'prepared.json',dict(status='PASS',states=len(layout),rows=offset,unique_candidate_assignments=physical,matching_calls_for_online_pools=matches,
        data_sha256=sha(path),layout_sha256=sha(a.results_root/'layout.json'),input_manifest_sha256=sha(a.results_root/'inputs.json')))
    scored=block_scores(layout,pool_rows,list(pool_rows['0']['10.0']),stats_module(a));main=scored['all']['10.0']['comparisons']
    capacity=main['oracle-G']['mean']>=.005 and main['oracle-G']['positive']>=4 and main['oracle-predicted_fee']['mean']>=.001
    write(a.results_root/'analysis/pool_summary.json',dict(status='PASS_CAPACITY_REPLAY',states=len(layout),subsets=scored,
        decision=dict(fit_capacity_gate=capacity,goal_complete=False,no_native_costs=True),protocol_sha256=sha(a.results_root/'protocol.json')))
    write(a.results_root/'status.json',dict(status='PREPARED' if capacity else 'STOPPED_POOL_CAPACITY',states=len(layout),rows=offset,updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',rows=offset,fit_capacity_gate=capacity,oracle_vs_G=main['oracle-G'],oracle_vs_predicted_fee=main['oracle-predicted_fee'])),flush=True)

def verify(a,p):
    for file,digest in p['files'].items():
        if sha(a.release_root/file)!=digest:raise ValueError('Frozen risk source changed')
    prep=read(a.results_root/'prepared.json')
    for file,key in [('training_data.npz','data_sha256'),('layout.json','layout_sha256'),('inputs.json','input_manifest_sha256')]:
        if sha(a.results_root/file)!=prep[key]:raise ValueError('Prepared risk data changed')

def fit(a):
    import numpy as np
    import sklearn
    import scipy
    from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
    p=read(a.results_root/'protocol.json');verify(a,p)
    if not read(a.results_root/'analysis/pool_summary.json')['decision']['fit_capacity_gate']:raise ValueError('Pool failed capacity gate')
    layout=read(a.results_root/'layout.json');fold=next(f for f in folds(a,layout) if f['block']==a.block and f['workload']==a.workload)
    folder=folded(a,a.block,a.workload)
    if (folder/'result.json').exists():raise FileExistsError('Preserve fitted risk fold')
    data=np.load(a.results_root/'training_data.npz');x=data['x'];train=np.isin(data['state_index'],fold['train']);test=np.isin(data['state_index'],fold['test'])
    if set(data['state_index'][train])!={i for i in fold['train'] if layout[i]['admitted']}:raise ValueError('Training membership')
    m=module(a);models={};versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__);files={};all_predictions={}
    for name in MODELS:
        if name=='hgb_probability':
            model=HistGradientBoostingClassifier(**p['hgb']).fit(x[train],data['y_actual'][train],sample_weight=data['weight'][train])
        else:model=HistGradientBoostingRegressor(**p['hgb']).fit(x[train],data['y_actual'][train]-data['y_predictor'][train],sample_weight=data['weight'][train])
        bundle=dict(model=model,name=name,protocol_sha256=sha(a.results_root/'protocol.json'),schema=m.SCHEMA,features=list(m.FEATURES),versions=versions,train_states=fold['train'],test_states=fold['test'])
        folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.pkl');path.write_bytes(pickle.dumps(bundle,protocol=5));files[path.name]=sha(path)
        restored=pickle.loads(path.read_bytes())['model'];raw=model.predict_proba(x[test])[:,list(model.classes_).index(1.)] if name=='hgb_probability' else model.predict(x[test])
        again=restored.predict_proba(x[test])[:,list(restored.classes_).index(1.)] if name=='hgb_probability' else restored.predict(x[test])
        if not np.array_equal(raw,again):raise ValueError('Serialized risk model prediction')
        probabilities=raw if name=='hgb_probability' else np.clip(raw+data['y_predictor'][test],0.,1.)
        test_indices=np.flatnonzero(test);test_map=np.full(len(x),-1,dtype=np.int64);test_map[test_indices]=np.arange(len(test_indices))
        all_predictions[name]=probabilities
        for si in fold['test']:
            state=layout[si];out=models.setdefault(str(si),{})
            for cost in COSTS:
                info=state['costs'][str(cost)];scores=[]
                for c in info['candidates']:
                    indices=test_map[c['start']:c['stop']]
                    if len(indices) and np.any(indices<0):raise ValueError('Cross-fold request lookup')
                    scores.append(float(probabilities[indices].sum()))
                ids=[c['action'] for c in info['candidates']];action=m.choose(ids,scores);chosen=next(c for c in info['candidates'] if c['action']==action)
                out.setdefault(str(cost),dict(controls=info['controls'],models={}))['models'][name]=dict(action=action,success=chosen['success'],scores=scores)
    np.savez_compressed(folder/'test_probabilities.npz',row_indices=np.flatnonzero(test),**all_predictions);files['test_probabilities.npz']=sha(folder/'test_probabilities.npz')
    write(folder/'predictions.json',models);files['predictions.json']=sha(folder/'predictions.json')
    write(folder/'result.json',dict(status='PASS',fold=fold,versions=versions,protocol_sha256=sha(a.results_root/'protocol.json'),files=files,training_rows=int(train.sum()),testing_rows=int(test.sum()),cpu=sorted(os.sched_getaffinity(0))))
    print(json.dumps(dict(status='FITTED',block=a.block,workload=a.workload)),flush=True)

def aggregate(a):
    p=read(a.results_root/'protocol.json');verify(a,p);layout=read(a.results_root/'layout.json');rows={};scores={}
    for fold in folds(a,layout):
        folder=folded(a,fold['block'],fold['workload']);rec=read(folder/'result.json')
        if rec['status']!='PASS' or rec['fold']!=fold or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Risk fold receipt')
        for file,digest in rec['files'].items():
            if sha(folder/file)!=digest:raise ValueError('Risk fold artifact changed')
        for si,record in read(folder/'predictions.json').items():
            if si in rows or int(si) not in fold['test']:raise ValueError('Risk held-out membership')
            rows[si]=record;scores[si]={cost:dict(**value['controls'],**{name:item['success'] for name,item in value['models'].items()}) for cost,value in record.items()}
    if len(rows)!=400:raise ValueError('Incomplete risk OOF')
    names=list(scores['0']['10.0']);subsets=block_scores(layout,scores,names,stats_module(a));main=subsets['all']['10.0'];gates={}
    refs=('G','predicted_fee','predicted_lex','uniform_pool');best=max(refs,key=lambda n:main['groups'][n]['mean'])
    for name in MODELS:
        g=main['comparisons'][name+'-G'];increment=main['comparisons'][name+'-'+best] if best!='G' else g
        gates[name]=dict(gain_vs_G=g,nonlearning_reference=best,increment=increment,
            optimistic_signal=g['mean']>=.005 and g['positive']>=4 and increment['mean']>=.001)
    out=dict(status='PASS_DEVELOPMENT_REPLAY',states=400,folds=25,models=50,protocol_sha256=sha(a.results_root/'protocol.json'),subsets=subsets,
        decision=dict(candidates=gates,goal_complete=False,native_costs_not_measured=True,independent_confirmation=False))
    write(a.results_root/'analysis/summary.json',out);write(a.results_root/'analysis/predictions.json',rows)
    write(a.results_root/'status.json',dict(status='COMPLETED',states=400,models=50,updated_unix=time.time()));print(json.dumps(out['decision']),flush=True)

def pipeline(a):
    p=read(a.results_root/'protocol.json');verify(a,p);layout=read(a.results_root/'layout.json');resource=load('risk_controller_resources',a.resource_module)
    if not read(a.results_root/'analysis/pool_summary.json')['decision']['fit_capacity_gate']:raise ValueError('No fit: pool failed capacity gate')
    def run(fold):
        cmd=[sys.executable,str(a.release_root/FILES[1]),'fit']
        for key in ('release_root','teacher_results','teacher_validation','results_root','validation_root','feasibility_release','resource_module','resources'):
            cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
        cmd+=['--block',str(fold['block']),'--workload',str(fold['workload'])]
        path=a.results_root/'worker_logs'/f"{fold['block']}_{fold['workload']}.log";path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:ret=subprocess.run(cmd,cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:raise RuntimeError('Risk fitting failed: '+str(fold['block'])+'/'+str(fold['workload']))
    with resource.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                pending=[pool.submit(run,f) for f in folds(a,layout)]
                for n,future in enumerate(as_completed(pending),1):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed_folds=n,total_folds=25,updated_unix=time.time()))
            with resource.CpuLease(a.resources,task='request_risk_aggregate'):aggregate(a)
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('prepare','fit','pipeline','aggregate'))
    for key in ('release_root','teacher_results','teacher_validation','results_root','validation_root','feasibility_release','resource_module','resources'):
        ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--block',type=int);ap.add_argument('--workload',type=int);a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Numerical processing, model fitting and statistics are remote-only')
    resource=load('request_risk_resources',a.resource_module)
    if a.mode=='pipeline':pipeline(a)
    else:
        with resource.CpuLease(a.resources,task=['request_risk',a.mode,a.block,a.workload]):
            if a.mode=='prepare':prepare(a)
            elif a.mode=='fit':fit(a)
            else:aggregate(a)

if __name__=='__main__':main()
