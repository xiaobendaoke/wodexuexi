"""Cache-hit target screening on archived development states, remote only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
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

FILES=('paths.py','scripts/analysis/sf_hlm_cache_calibration.py','utils/sf_hlm_cache_calibration.py',
    'utils/sf_hlm_cache_mixture.py','utils/sf_hlm_selector.py','tests/test_sf_hlm_cache_calibration.py',
    'docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=('release_root','feasibility_release','parent_launch','old_calibration','closed_loop','closed_validation',
    'precision_diagnostic','results_root','validation_root','resources','resource_module')
METHODS=('learned','prior','own_proxy','half_proxy')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    raise TypeError(type(x).__name__)
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n');temp.replace(p)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def res(a):return load('cache_calibration_resources',a.resource_module)
def helpers(a):
    return load('utils.sf_hlm_cache_calibration',a.release_root/'utils/sf_hlm_cache_calibration.py'),load('utils.sf_hlm_cache_mixture',a.release_root/'utils/sf_hlm_cache_mixture.py')
def verify(a,p):
    for name,digest in p['files'].items():
        if sha(a.release_root/name)!=digest:raise ValueError('Frozen cache-calibration source changed')
def splits(a,p):return load('cache_calibration_folds',a.release_root/'utils/sf_hlm_selector.py').folds(p['identities'])
def folded(a,b,w):return a.results_root/'folds'/f'block_{b}_workload_{w}'


def prepare(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable cache target namespace')
    audit=read(a.closed_validation/'raw_record_audit.json');summary=read(a.closed_loop/'analysis/summary.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=sha(a.closed_loop/'analysis/summary.json') or summary['decision']['closed_loop_development_signal']:raise ValueError('Expected audited failed precursor')
    precision=read(a.precision_diagnostic)
    if precision['models_trained']!=0 or precision['checks']['direct_failures'] or not precision['checks']['inverse_failures']:raise ValueError('Expected separate public-inverse precision diagnosis')
    old=read(a.old_calibration/'protocol.json');inputs={str(p):sha(p) for p in (a.old_calibration/'protocol.json',a.closed_validation/'raw_record_audit.json',a.closed_loop/'analysis/summary.json',a.parent_launch,a.precision_diagnostic)}
    states=[]
    for state in old['states']:
        if sha(state['state'])!=state['state_sha256']:raise ValueError('Original state metadata changed')
        meta=read(state['state']);identity=state['identity'];suffix='oracle_'+'_'.join(map(str,identity))+'/public.json'
        pub=next(Path(p) for p in old['input_files'] if p.endswith(suffix));receipt=read(pub.parent/'result.json')
        for name in ('public.json','labels.json'):
            path=pub.parent/name
            if sha(path)!=receipt['files'][name]:raise ValueError('Old public/physical record changed')
            inputs[str(path)]=receipt['files'][name]
        if sha(meta['observed'])!=meta['files'][meta['observed']]:raise ValueError('Private label source changed')
        inputs[state['state']]=state['state_sha256'];inputs[meta['observed']]=meta['files'][meta['observed']]
        states.append(dict(identity=identity,observed=meta['observed'],public=str(pub),labels=str(pub.parent/'labels.json')))
    cc,_=helpers(a)
    p=dict(name='sf_hlm_public_cache_hit_calibration_20261018_label_physics_v3',files={f:sha(a.release_root/f) for f in FILES},input_files=inputs,
        identities=old['identities'],states=states,features=cc.FEATURES,model=cc.MODEL,methods=METHODS,
        training_states=400,split='25 leave-block-and-workload-out folds; 256 train/16 test/128 excluded',
        state_weight=1.,log_loss_clip=1e-12,calibration_bins=10,cache_truth_labels_only=True,
        eligibility='public cooperative decomposition, original 0<p<1, miss-hit base >1e-10',
        physical_parity_tolerance=1e-6,label_validation='independent direct physical components on evaluator side only',
        public_inverse_precision_failure_retained=True,no_additional_private_policy_inputs=True,
        new_closed_loop_data_used_for_training=False,independent_confirmation=False,goal_complete=False)
    write(a.results_root/'protocol.json',p)
    f=load('cache_calibration_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py');f.boot(a,620001);cc,mix=helpers(a)
    import config
    from environment import comm_model as comms
    import numpy as np
    powers=dict(backhaul_rx=float(config.UAV_BACKHAUL_RX_POWER),backhaul_tx=float(config.UAV_BACKHAUL_TX_POWER),uav_tx=float(config.UAV_COMM_TX_POWER),uav_rx=float(config.UAV_COMM_RX_POWER))
    def tx(size,rate):return float(size)*float(config.BITS_PER_BYTE)/max(float(rate),float(config.EPSILON))
    table=[];targets=[];weights=[];indices=[];baselines={m:[] for m in METHODS[1:]};layout=[];counts=Counter();bad=[];checked=0;max_residual=0.;max_inverse_residual=0.;residuals=[]
    for si,state in enumerate(states):
        public=read(state['public']);labels=read(state['labels']);rows=public['rows']
        for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
        env,obs=pickle.loads(Path(state['observed']).read_bytes());ues={u.id:u for u in env.ues};uavs={u.id:u for u in env.uavs}
        parts=[mix.decode_row(row,powers) for row in rows];truth={};direct={};local=[]
        for i,(row,options) in enumerate(zip(rows,parts)):
            ue=ues[row['ue_id']]
            if not ue.current_request.is_service:raise ValueError('Private label request identity changed')
            for action,part in options.items():
                counts['all_public_edges']+=1
                if not cc.eligible(part):counts['publicly_ineligible_edges']+=1;continue
                target=int(row['options'][action]['target']);hit=float(bool(uavs[target].cache[ue.current_request.req_id]));truth[i,action]=hit
                # Independent evaluator-only validation, never sent to features.
                source=uavs[row['source']];destination=uavs[target]
                upload=tx(row['size'],comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos,source.pos),max(len(source.current_covered_ues),1)))
                hop=tx(row['size'],comms.calculate_uav_uav_rate(comms.calculate_channel_gain(source.pos,destination.pos)))
                fetch=tx(config.FILE_SIZES[ue.current_request.req_id],comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(destination.pos,config.MBS_POS)))
                direct[i,action]=(upload+hop,upload+hop+fetch)
                local.append((cc.cache_features(row,action),hit,cc.controls(row,action)));counts['eligible_edges']+=1
        for oi,outcome in enumerate(labels['outcomes'].values()):
            assignment=outcome['assignment'];target_counts=Counter(row['options'][action]['target'] for row,action in zip(rows,assignment));requests=outcome['requests']
            if len(requests)!=len(rows):raise ValueError('Physical label population differs')
            for i,(row,action,request) in enumerate(zip(rows,assignment,requests)):
                if request['ue_id']!=row['ue_id']:raise ValueError('Physical request identity differs')
                if (i,action) not in truth:continue
                part=parts[i][action];target=row['options'][action]['target'];compute=part['work']*target_counts[target]
                inverse=(part['hit_base'] if truth[i,action] else part['miss_base'])+compute
                pred=direct[i,action][0 if truth[i,action] else 1]+compute
                residual=abs(float(request['latency'])-pred);inverse_residual=abs(float(request['latency'])-inverse)
                max_residual=max(max_residual,residual);max_inverse_residual=max(max_inverse_residual,inverse_residual);checked+=1
                counts['public_inverse_parity_failures']+=int(inverse_residual>1e-6*(1+abs(float(request['latency']))))
                residuals.append([si,oi,i,action,part['p'],truth[i,action],float(request['latency']),inverse,pred,inverse_residual,residual])
                if residual>1e-6*(1+abs(float(request['latency']))):
                    if len(bad)<30:bad.append(dict(identity=state['identity'],row=i,action=action,actual=request['latency'],branch_prediction=pred,residual=residual,cache_hit=truth[i,action]))
                    counts['physical_parity_failures']+=1
        start=len(table)
        for x,y,control in local:
            table.append(x);targets.append(y);weights.append(1/len(local));indices.append(si)
            for method in baselines:baselines[method].append(control[method])
        layout.append(dict(identity=state['identity'],start=start,stop=len(table),eligible_edges=len(local)))
    residual_path=a.results_root/'physical_residuals.npz';np.savez_compressed(residual_path,values=np.asarray(residuals),columns=np.asarray(['state_index','outcome_index','row','action','prior','cache_hit','actual','public_inverse','direct','inverse_residual','direct_residual']))
    write(a.results_root/'preparation_audit.json',dict(status='FAIL' if bad else 'PASS_LABEL_PHYSICS_PUBLIC_INVERSE_APPROXIMATE',counts=dict(counts),physical_branch_checks=checked,max_physical_residual=max_residual,
        max_public_inverse_residual=max_inverse_residual,all_residuals_sha256=sha(residual_path),public_inverse_parity_passed=False,private_components_used_only_for_label_validation=True,failures=bad,powers=powers))
    if bad:raise ValueError('Cache truth and physical branch equations disagree; no samples dropped')
    path=a.results_root/'training_data.npz';np.savez_compressed(path,x=np.asarray(table,dtype=np.float64).reshape(-1,18),y=np.asarray(targets,dtype=np.float64),weight=np.asarray(weights),state_index=np.asarray(indices),**{m:np.asarray(v) for m,v in baselines.items()})
    write(a.results_root/'layout.json',layout);write(a.results_root/'prepared.json',dict(status='PASS',rows=len(table),source_states=400,eligible_states=sum(x['eligible_edges']>0 for x in layout),data_sha256=sha(path),layout_sha256=sha(a.results_root/'layout.json'),preparation_audit_sha256=sha(a.results_root/'preparation_audit.json')))
    print(json.dumps(dict(status='PREPARED',rows=len(table),physical_checks=checked,max_residual=max_residual)),flush=True)


def fit(a,p):
    import numpy as np
    verify(a,p);prepared=read(a.results_root/'prepared.json')
    if sha(a.results_root/'training_data.npz')!=prepared['data_sha256']:raise ValueError('Frozen cache data changed')
    fold=next(f for f in splits(a,p) if f['block']==a.block and f['workload']==a.workload);out=folded(a,a.block,a.workload)
    if (out/'result.json').exists():raise FileExistsError('Cache fold already completed')
    data=np.load(a.results_root/'training_data.npz');train=np.isin(data['state_index'],fold['train']);test=np.isin(data['state_index'],fold['test']);cc,_=helpers(a)
    mode=p.get('feature_mode','clipped')
    if tuple(p['features'])!=cc.feature_names(mode):raise ValueError('Training feature schema differs from encoding')
    model,diagnostic=cc.fit_model(data['x'][train],data['y'][train],data['weight'][train]);model.public_cache_feature_mode=mode
    pred=model.predict_proba(data['x'][test])[:,1]
    out.mkdir(parents=True,exist_ok=True);path=out/'model.pkl';path.write_bytes(pickle.dumps(dict(model=model,fold=fold,features=p['features'],feature_mode=mode,protocol_sha256=sha(a.results_root/'protocol.json'),diagnostic=diagnostic),protocol=5))
    if not np.array_equal(pred,pickle.loads(path.read_bytes())['model'].predict_proba(data['x'][test])[:,1]):raise ValueError('Cache model serialization differs')
    if diagnostic['kind']=='weighted_standardized_logistic' and max(diagnostic['iterations'])>=p['model']['max_iter']:raise ValueError('Fixed cache optimizer did not converge')
    np.savez_compressed(out/'test_predictions.npz',row_indices=np.flatnonzero(test),probability=pred)
    write(out/'result.json',dict(status='PASS',fold=fold,diagnostic=diagnostic,train_rows=int(train.sum()),test_rows=int(test.sum()),
        protocol_sha256=sha(a.results_root/'protocol.json'),files={name:sha(out/name) for name in ('model.pkl','test_predictions.npz')},cpu=sorted(os.sched_getaffinity(0))))


def analyze(a,p,*,write_result=True):
    import numpy as np
    from scipy.stats import t
    verify(a,p);data=np.load(a.results_root/'training_data.npz');prob=np.full(len(data['y']),np.nan);seen=np.zeros(len(prob),int)
    for fold in splits(a,p):
        out=folded(a,fold['block'],fold['workload']);receipt=read(out/'result.json');pred=np.load(out/'test_predictions.npz')
        if receipt['fold']!=fold or receipt['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Wrong cache fold')
        for name,digest in receipt['files'].items():
            if sha(out/name)!=digest:raise ValueError('Cache prediction changed')
        ii=pred['row_indices'];prob[ii]=pred['probability'];seen[ii]+=1
    if not np.all(seen==1) or not np.isfinite(prob).all():raise ValueError('Missing or duplicate held-out rows')
    predictions=dict(learned=prob,**{m:data[m] for m in METHODS[1:]});states=[]
    for si,identity in enumerate(p['identities']):
        take=data['state_index']==si
        if not take.any():continue
        yy=data['y'][take];metrics={}
        for name,pp in predictions.items():
            zz=pp[take];clipped=np.clip(zz,1e-12,1-1e-12)
            metrics[name]=dict(brier=float(np.mean((zz-yy)**2)),log_loss=float(np.mean(-yy*np.log(clipped)-(1-yy)*np.log(1-clipped))))
        states.append(dict(identity=identity,edges=int(take.sum()),methods=metrics))
    blocks={}
    for block in sorted({x['identity'][1] for x in states}):
        rows=[x for x in states if x['identity'][1]==block];blocks[str(block)]={m:{metric:statistics.mean(x['methods'][m][metric] for x in rows) for metric in ('brier','log_loss')} for m in METHODS}
    groups={m:{metric:statistics.mean(x[m][metric] for x in blocks.values()) for metric in ('brier','log_loss')} for m in METHODS};best=min(METHODS[1:],key=lambda m:groups[m]['brier'])
    differences={}
    for control in METHODS[1:]:
        values=[x[control]['brier']-x['learned']['brier'] for x in blocks.values()];mean=statistics.mean(values);radius=float(t.ppf(.975,4))*statistics.stdev(values)/5**.5
        differences[control]=dict(mean_improvement=mean,ci95=[mean-radius,mean+radius],values=values,positive=sum(x>0 for x in values),historical_blocks=5)
    calibration={}
    for method,pp in predictions.items():
        boxes=[]
        for j in range(10):
            take=(pp>=j/10)&(pp<(j+1)/10 if j<9 else pp<=1);w=data['weight'][take]
            boxes.append(dict(bin=j,rows=int(take.sum()),weight=float(w.sum()),mean_prediction=float(np.average(pp[take],weights=w)) if len(w) else None,hit_rate=float(np.average(data['y'][take],weights=w)) if len(w) else None))
        calibration[method]=boxes
    go=differences[best]['mean_improvement']>0 and differences[best]['positive']>=4
    summary=dict(status='CACHE_TARGET_DEVELOPMENT_SCREEN',source_states=400,eligible_states=len(states),eligible_edges=len(prob),blocks=blocks,groups=groups,
        best_label_free_control=best,brier_improvements=differences,calibration=calibration,
        decision=dict(cache_hit_target_promising_development_only=go,matching_improvement_tested=False,closed_loop_improvement_tested=False,goal_complete=False),
        protocol_sha256=sha(a.results_root/'protocol.json'),descriptive_only=True,independent_confirmation=False)
    if write_result:
        write(a.results_root/'analysis/summary.json',summary);write(a.results_root/'analysis/state_metrics.json',states)
        print(json.dumps(dict(status='ANALYZED',best_control=best,groups=groups,decision=summary['decision'])),flush=True)
    return summary,states


def command(a,mode):
    out=[sys.executable,str(a.release_root/FILES[1]),mode]
    for name in ARGS:out+=['--'+name.replace('_','-'),str(getattr(a,name))]
    return out


def pipeline(a,p):
    verify(a,p);r=res(a)
    def execute(fold):
        cmd=command(a,'fit')+['--block',str(fold['block']),'--workload',str(fold['workload'])];path=a.results_root/'worker_logs'/f"{fold['block']}_{fold['workload']}.log";path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:result=subprocess.run(cmd,cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Cache fold failed: '+str(fold['block'])+'/'+str(fold['workload']))
    with r.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                jobs=[pool.submit(execute,f) for f in splits(a,p)]
                for n,future in enumerate(as_completed(jobs),1):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed_folds=n,total=25,updated_unix=time.time()))
            result=subprocess.run(command(a,'analysis'),cwd=a.release_root)
            if result.returncode:raise RuntimeError('Cache analysis failed')
            write(a.results_root/'completion.json',dict(status='COMPLETED',models=25,protocol_sha256=sha(a.results_root/'protocol.json'),summary_sha256=sha(a.results_root/'analysis/summary.json')))
            write(a.results_root/'status.json',dict(status='COMPLETED',models=25,updated_unix=time.time()))
        except BaseException as error:write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','fit','train','analysis'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    parser.add_argument('--block',type=int);parser.add_argument('--workload',type=int);a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Cache labels/training/statistics run remotely')
    r=res(a)
    if a.mode=='prepare':
        with r.CpuLease(a.resources,task=['cache_calibration_prepare'],kind='environment'):prepare(a)
    else:
        p=read(a.results_root/'protocol.json');verify(a,p)
        if a.mode=='train':pipeline(a,p)
        else:
            with r.CpuLease(a.resources,task=['cache_calibration',a.mode,a.block,a.workload],kind='environment'):
                if a.mode=='fit':fit(a,p)
                else:analyze(a,p)


if __name__=='__main__':main()
