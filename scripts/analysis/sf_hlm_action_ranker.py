"""Fixed cheap ranking probes with two-axis development cross-fitting.

All fitting and scientific analysis run remotely. Cost conditions here are
common added-time assumptions; native full-policy timing is a later stage.
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
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

MODELS=('ridge_actual','hgb_actual','hgb_predictor')
FILES=('paths.py','scripts/analysis/sf_hlm_action_ranker.py','utils/sf_hlm_action_ranker.py','utils/sf_hlm_selector.py',
       'tests/test_sf_hlm_action_ranker.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


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
def feature_module(a):return load('action_ranker_inputs',a.release_root/'utils/sf_hlm_action_ranker.py')
def fold_folder(a,block,workload):return a.results_root/'folds'/f'block_{block}_workload_{workload}'


def prepare(a):
    import numpy as np
    f=feature_module(a);parent=read(a.teacher_results/'protocol.json');audit=read(a.teacher_validation/'raw_record_audit.json')
    if parent['smoke'] or audit['status']!='PASS' or audit['summary_sha256']!=sha(a.teacher_results/'analysis/summary.json'):raise ValueError('Teacher collection not verified')
    if (a.results_root/'protocol.json').exists():raise FileExistsError('New fitting namespace required')
    p=dict(name='sf_hlm_action_ranker_20261017_v1',models=list(MODELS),schema=f.SCHEMA,dimension=128,
        identities=parent['identities'],cost_ms=parent['cost_hypotheses_ms'],main_cost_ms=10.,teacher_results=str(a.teacher_results),
        teacher_protocol_sha256=sha(a.teacher_results/'protocol.json'),teacher_summary_sha256=sha(a.teacher_results/'analysis/summary.json'),
        teacher_audit_sha256=sha(a.teacher_validation/'raw_record_audit.json'),files={rel:sha(a.release_root/rel) for rel in FILES},
        ridge_alpha=10.,hgb=dict(max_iter=50,learning_rate=.1,max_leaf_nodes=8,max_depth=3,min_samples_leaf=32,l2_regularization=10.,early_stopping=False,random_state=20261017),
        weight='Each state has total weight one over all candidate actions and cost conditions',
        split='25 leave-one-block-and-workload folds; 256 training states,16 testing,128 excluded',
        feature_boundary='Public rows, initial/G allocation and quota only; no queried candidate matching or private outcome in features',
        no_native_cost_measurement=True,independent_confirmation=False,goal_complete=False)
    write(a.results_root/'protocol.json',p)
    xs=[];ys=[];yps=[];weights=[];state_indices=[];layout=[];inputs={}
    offset=0
    for si,identity in enumerate(p['identities']):
        td=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)))
        rec=read(td/'result.json');public=read(td/'public.json');labels=read(td/'labels.json')
        if rec['identity']!=identity or rec['status']!='PASS' or rec['protocol_sha256']!=p['teacher_protocol_sha256']:raise ValueError('Teacher receipt identity')
        for name,digest in rec['files'].items():
            if sha(td/name)!=digest:raise ValueError('Teacher record changed')
            inputs[str(td/name)]=digest
        safe=f.public_payload(public);per_state={};candidates={c['action']:c for c in public['candidates']}
        for c in p['cost_ms']:
            ids,x=f.features(safe,c);greq=labels['outcomes'][labels['action_outcome']['-1']]['requests']
            gscore=sum(r['latency']+c/1000<=r['deadline'] for r in greq)
            gd=candidates[-1]['predicted']['latencies'];deadlines=[r['deadline'] for r in safe['rows']]
            gp=sum(lat+c/1000<=d for lat,d in zip(gd,deadlines));ya=[];yp=[];absolute={}
            for action in ids:
                req=labels['outcomes'][labels['action_outcome'][str(action)]]['requests']
                score=sum(r['latency']+c/1000<=r['deadline'] for r in req);absolute[action]=score;ya.append(score-gscore)
                latency=candidates[action]['predicted']['latencies'];yp.append(sum(lat+c/1000<=d for lat,d in zip(latency,deadlines))-gp)
            rule=f.analytic_choice(ids,x,len(safe['rows']));choices=safe['legal_nonstop_actions']
            controls=dict(G=gscore,stop=absolute[0],random_one=sum(absolute[k] for k in choices)/len(choices) if choices else absolute[0],
                heuristic_one=absolute[public['heuristic_action']],rule_cost=absolute[rule],
                predicted_best_offline=absolute[public['predicted_best_action']],oracle=max(absolute.values()))
            per_state[str(c)]=dict(start=offset,stop=offset+len(ids),actions=ids,controls=controls,rule_action=rule)
            xs.append(x);ys.extend(ya);yps.extend(yp);weights.extend([f.training_weight(len(ids),len(p['cost_ms']))]*len(ids))
            state_indices.extend([si]*len(ids));offset+=len(ids)
        layout.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],costs=per_state))
    x=np.concatenate(xs);data=a.results_root/'training_data.npz'
    np.savez_compressed(data,x=x,y_actual=np.asarray(ys,dtype=np.float64),y_predictor=np.asarray(yps,dtype=np.float64),
        weight=np.asarray(weights,dtype=np.float64),state_index=np.asarray(state_indices,dtype=np.int64))
    if x.shape!=(offset,128):raise ValueError('Feature population')
    for si in range(400):
        if not np.isclose(np.asarray(weights)[np.asarray(state_indices)==si].sum(),1.):raise ValueError('State weights are not normalized')
    write(a.results_root/'layout.json',layout);write(a.results_root/'inputs.json',inputs)
    write(a.results_root/'prepared.json',dict(status='PASS',rows=offset,states=400,data_sha256=sha(data),layout_sha256=sha(a.results_root/'layout.json'),input_manifest_sha256=sha(a.results_root/'inputs.json')))
    write(a.results_root/'status.json',dict(status='PREPARED',states=400,rows=offset,updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=400,rows=offset)),flush=True)


def verify(a,p):
    for rel,digest in p['files'].items():
        if sha(a.release_root/rel)!=digest:raise ValueError('Frozen fitting source changed')
    prep=read(a.results_root/'prepared.json')
    for file,key in [('training_data.npz','data_sha256'),('layout.json','layout_sha256'),('inputs.json','input_manifest_sha256')]:
        if sha(a.results_root/file)!=prep[key]:raise ValueError('Prepared dataset changed')


def fit_fold(a):
    import numpy as np
    import sklearn
    import scipy
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    p=read(a.results_root/'protocol.json');verify(a,p);f=feature_module(a)
    selector=load('ranker_fold_contract',a.release_root/'utils/sf_hlm_selector.py');layout=read(a.results_root/'layout.json')
    fold=next(z for z in selector.folds([s['identity'] for s in layout]) if z['block']==a.block and z['workload']==a.workload)
    folder=fold_folder(a,a.block,a.workload)
    if (folder/'result.json').exists():raise FileExistsError('Completed fitted models are immutable')
    archive=np.load(a.results_root/'training_data.npz');x=archive['x'];state_index=archive['state_index'];sample_weight=archive['weight']
    train=np.isin(state_index,fold['train']);test=np.isin(state_index,fold['test'])
    if set(state_index[train])!=set(fold['train']) or set(state_index[test])!=set(fold['test']):raise ValueError('Fold membership mismatch')
    versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    predictions={str(i):{str(c):dict(controls=layout[i]['costs'][str(c)]['controls'],models={}) for c in p['cost_ms']} for i in fold['test']}
    files={}
    for name in MODELS:
        target=archive['y_predictor'] if name.endswith('predictor') else archive['y_actual']
        scaler=None
        if name.startswith('ridge'):
            scaler=StandardScaler().fit(x[train],sample_weight=sample_weight[train])
            model=Ridge(alpha=10.).fit(scaler.transform(x[train]),target[train],sample_weight=sample_weight[train])
        else:model=HistGradientBoostingRegressor(**p['hgb']).fit(x[train],target[train],sample_weight=sample_weight[train])
        bundle=dict(model=model,scaler=scaler,name=name,schema=f.SCHEMA,protocol_sha256=sha(a.results_root/'protocol.json'),
            versions=versions,nominal_cost_ms=10.,train_states=fold['train'],test_states=fold['test'])
        folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.pkl');path.write_bytes(pickle.dumps(bundle,protocol=5));files[path.name]=sha(path)
        restored=pickle.loads(path.read_bytes())
        for si in fold['test']:
            for cost in p['cost_ms']:
                record=layout[si]['costs'][str(cost)];xx=x[record['start']:record['stop']]
                pp=model.predict(scaler.transform(xx) if scaler is not None else xx)
                again=restored['model'].predict(restored['scaler'].transform(xx) if restored['scaler'] is not None else xx)
                if not np.array_equal(pp,again):raise ValueError('Serialized inference differs')
                action=f.choose(record['actions'],pp);row=record['actions'].index(action)
                success=record['controls']['G']+archive['y_actual'][record['start']+row]
                predictions[str(si)][str(cost)]['models'][name]=dict(action=action,success=float(success),predicted=[float(v) for v in pp])
    write(folder/'predictions.json',predictions);files['predictions.json']=sha(folder/'predictions.json')
    write(folder/'result.json',dict(status='PASS',protocol_sha256=sha(a.results_root/'protocol.json'),fold=fold,versions=versions,
        training_rows=int(train.sum()),testing_rows=int(test.sum()),cpu=sorted(os.sched_getaffinity(0)),files=files))
    print(json.dumps(dict(status='FITTED',block=a.block,workload=a.workload)),flush=True)


def aggregate(a):
    p=read(a.results_root/'protocol.json');verify(a,p);layout=read(a.results_root/'layout.json')
    selector=load('ranker_score_contract',a.release_root/'utils/sf_hlm_selector.py');rows={};folds=list(selector.folds([s['identity'] for s in layout]))
    helper=load('ranker_feasibility_stats',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
    for fold in folds:
        folder=fold_folder(a,fold['block'],fold['workload']);rec=read(folder/'result.json')
        if rec['status']!='PASS' or rec['fold']!=fold or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Fold result mismatch')
        for file,digest in rec['files'].items():
            if sha(folder/file)!=digest:raise ValueError('Fitted artifact changed')
        pred=read(folder/'predictions.json')
        if set(map(int,pred))!=set(fold['test']):raise ValueError('Test prediction membership')
        for index,row in pred.items():
            if index in rows:raise ValueError('Duplicate held-out state')
            rows[index]=row
    if set(map(int,rows))!=set(range(400)):raise ValueError('Incomplete held-out population')
    subsets={};controls=list(layout[0]['costs'][str(p['cost_ms'][0])]['controls']);names=[*controls,*MODELS]
    for source in ('all','greedy','learned'):
        indices=[i for i,s in enumerate(layout) if source=='all' or s['identity'][0]==source];by_cost={}
        for cost in p['cost_ms']:
            blocks={}
            for block in sorted({layout[i]['identity'][1] for i in indices}):
                ids=[i for i in indices if layout[i]['identity'][1]==block];den=sum(layout[i]['generated'] for i in ids)
                totals={name:sum(rows[str(i)][str(cost)]['controls'][name] if name in controls else rows[str(i)][str(cost)]['models'][name]['success'] for i in ids) for name in names}
                blocks[str(block)]=dict(generated=den,rates={name:value/den for name,value in totals.items()})
            comparisons={name+'-G':helper.paired([b['rates'][name]-b['rates']['G'] for b in blocks.values()]) for name in names if name!='G'}
            for name in MODELS:
                for ref in ('rule_cost','random_one','heuristic_one'):
                    comparisons[name+'-'+ref]=helper.paired([b['rates'][name]-b['rates'][ref] for b in blocks.values()])
            comparisons['hgb_actual-hgb_predictor']=helper.paired([b['rates']['hgb_actual']-b['rates']['hgb_predictor'] for b in blocks.values()])
            by_cost[str(cost)]=dict(blocks=blocks,groups={name:helper.paired([b['rates'][name] for b in blocks.values()]) for name in names},comparisons=comparisons)
        subsets[source]=by_cost
    main=subsets['all']['10.0'];gates={name:dict(gain=main['comparisons'][name+'-G'],optimistic_signal=main['comparisons'][name+'-G']['mean']>=.005 and main['comparisons'][name+'-G']['positive']>=4) for name in MODELS}
    out=dict(status='PASS_DEVELOPMENT_REPLAY',protocol_sha256=sha(a.results_root/'protocol.json'),states=400,folds=25,models=75,
        subsets=subsets,decision=dict(candidates=gates,native_costs_not_measured=True,goal_complete=False,independent_confirmation=False))
    write(a.results_root/'analysis/summary.json',out);write(a.results_root/'analysis/predictions.json',rows)
    write(a.results_root/'status.json',dict(status='COMPLETED',states=400,models=75,updated_unix=time.time()))
    print(json.dumps(out['decision']),flush=True)


def pipeline(a):
    p=read(a.results_root/'protocol.json');verify(a,p);selector=load('ranker_fold_setup',a.release_root/'utils/sf_hlm_selector.py')
    folds=list(selector.folds(p['identities']));res=load('ranker_resources',a.resource_module)
    def run(fold):
        command=[sys.executable,str(a.release_root/FILES[1]),'fit']
        for key in ('release_root','teacher_results','teacher_validation','results_root','validation_root','feasibility_release','resource_module','resources'):
            command+=['--'+key.replace('_','-'),str(getattr(a,key))]
        command+=['--block',str(fold['block']),'--workload',str(fold['workload'])]
        path=a.results_root/'worker_logs'/f"{fold['block']}_{fold['workload']}.log";path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:ret=subprocess.run(command,cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:raise RuntimeError('Fold fitting failed: '+str(fold['block'])+'/'+str(fold['workload']))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(run,fold) for fold in folds]
                for n,future in enumerate(as_completed(futures),1):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed_folds=n,total_folds=25,updated_unix=time.time()))
            with res.CpuLease(a.resources,task='ranker_aggregate'):aggregate(a)
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('prepare','fit','pipeline','aggregate'))
    for key in ('release_root','teacher_results','teacher_validation','results_root','validation_root','feasibility_release','resource_module','resources'):
        ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--block',type=int);ap.add_argument('--workload',type=int);a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Remote model fitting and statistics only')
    res=load('ranker_runtime_resources',a.resource_module)
    if a.mode=='pipeline':pipeline(a)
    else:
        with res.CpuLease(a.resources,task=['action_ranker',a.mode,a.block,a.workload]):
            if a.mode=='prepare':prepare(a)
            elif a.mode=='fit':fit_fold(a)
            else:aggregate(a)


if __name__=='__main__':main()
