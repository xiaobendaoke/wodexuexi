"""Prespecified 19-dimensional public fetch-gap conditional calibration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import sf_hlm_cache_calibration as b

METHODS=('learned','legacy_learned','prior','own_proxy','half_proxy','cold_guard_prior','cold_guard_own','cold_guard_half')
FILES=('paths.py','scripts/analysis/sf_hlm_cache_gap.py','scripts/analysis/sf_hlm_cache_calibration.py','utils/sf_hlm_cache_gap.py','utils/sf_hlm_cache_calibration.py','utils/sf_hlm_cache_context.py','utils/sf_hlm_cache_mixture.py','utils/sf_hlm_selector.py','tests/test_sf_hlm_cache_gap.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=b.ARGS+('source_training','source_validation','decision_results','decision_validation','public_price_calibration')
b.FILES=FILES;b.ARGS=ARGS;b.METHODS=METHODS


def prepare(a):
    import numpy as np
    from utils.sf_hlm_cache_gap import features,FEATURES,MODE,RULE
    from utils.sf_hlm_cache_calibration import MODEL,eligible
    from utils.sf_hlm_cache_mixture import decode_row
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable public-gap namespace')
    old=b.read(a.source_training/'protocol.json');audit=b.read(a.source_validation/'raw_record_audit.json');prepared=b.read(a.source_training/'prepared.json');da=b.read(a.decision_validation/'raw_record_audit.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=b.sha(a.source_training/'analysis/summary.json') or old['feature_mode']!='full_logit_context' or old['context_rule']!=RULE:raise ValueError('Wrong audited predecessor')
    if da['status']!='PASS' or da['analysis_sha256']!=b.sha(a.decision_results/'analysis.json'):raise ValueError('Unverified fixed decision masks')
    if b.sha(a.source_training/'training_data.npz')!=prepared['data_sha256']:raise ValueError('Frozen predecessor arrays changed')
    z=np.load(a.source_training/'training_data.npz');data={k:z[k].copy() for k in z.files};x=[];inputs={str(p):b.sha(p) for p in (a.source_training/'protocol.json',a.source_training/'prepared.json',a.source_training/'training_data.npz',a.source_training/'layout.json',a.source_training/'analysis/summary.json',a.source_validation/'raw_record_audit.json',a.decision_results/'protocol.json',a.decision_results/'analysis.json',a.decision_results/'edge_context.npz',a.decision_validation/'raw_record_audit.json')}
    powers=None
    for fold in (a.public_price_calibration/'folds').iterdir():
        path=fold/'prices.json';part=b.read(path)['powers'];inputs[str(path)]=b.sha(path)
        if powers is not None and powers!=part:raise ValueError('Public powers depend on fold')
        powers=part
    legacy=np.full(len(data['y']),np.nan);seen=np.zeros(len(legacy),int)
    for fold in (a.source_training/'folds').iterdir():
        for name,digest in b.read(fold/'result.json')['files'].items():
            if b.sha(fold/name)!=digest:raise ValueError('Original model/prediction changed')
            inputs[str(fold/name)]=digest
        inputs[str(fold/'result.json')]=b.sha(fold/'result.json');z=np.load(fold/'test_predictions.npz');legacy[z['row_indices']]=z['probability'];seen[z['row_indices']]+=1
    if not np.all(seen==1):raise ValueError('Legacy prediction coverage differs')
    for si,state in enumerate(old['states']):
        path=Path(state['public']);inputs[str(path)]=b.sha(path);rows=b.read(path)['rows'];ii=np.flatnonzero(data['state_index']==si);local=[]
        for row in rows:
            row['options']={int(k):v for k,v in row['options'].items()}
            for ac,part in decode_row(row,powers).items():
                if eligible(part):local.append(features(row,ac,powers))
        arr=np.asarray(local)
        if arr.shape!=(len(ii),19) or not np.array_equal(arr[:,:18],data['x'][ii]):raise ValueError('Original eighteen features/edge order changed')
        x.extend(local)
    data['x']=np.asarray(x);data['legacy_learned']=legacy
    if not np.isfinite(data['x']).all() or data['x'].shape!=(13459,19):raise ValueError('Unexpected public gap population')
    p=dict(name='sf_hlm_public_fetch_gap19_conditional_calibration_20261018_v1',files={f:b.sha(a.release_root/f) for f in FILES},input_files=inputs,states=old['states'],identities=old['identities'],features=FEATURES,feature_mode=MODE,context_rule=RULE,model=MODEL,methods=METHODS,split=old['split'],training_states=400,guarded_states=100,training_filter='not public guarded state',original_eighteen_fields_labels_weights_splits_unchanged=True,new_feature='log1p(public miss_base - hit_base)',public_powers=powers,prediction_improvement_threshold_vs_legacy=.001,independent_confirmation=False,goal_complete=False)
    b.write(a.results_root/'protocol.json',p);np.savez_compressed(a.results_root/'training_data.npz',**data);b.write(a.results_root/'layout.json',b.read(a.source_training/'layout.json'));b.write(a.results_root/'prepared.json',dict(status='PASS',rows=len(legacy),dimensions=19,data_sha256=b.sha(a.results_root/'training_data.npz')));print(json.dumps(dict(status='PREPARED',rows=len(legacy),dimensions=19)),flush=True)


def fit(a,p):
    import numpy as np
    from utils.sf_hlm_cache_calibration import fit_model
    from utils.sf_hlm_cache_gap import validate_bundle,MODE,RULE
    b.verify(a,p);prepared=b.read(a.results_root/'prepared.json')
    if b.sha(a.results_root/'training_data.npz')!=prepared['data_sha256']:raise ValueError('Prepared gap data changed')
    z=np.load(a.results_root/'training_data.npz');d={k:z[k] for k in z.files};fold=next(x for x in b.splits(a,p) if x['block']==a.block and x['workload']==a.workload);train=np.isin(d['state_index'],fold['train'])&~d['guarded'];test=np.isin(d['state_index'],fold['test']);warm_test=test&~d['guarded'];out=b.folded(a,a.block,a.workload)
    if (out/'result.json').exists():raise FileExistsError('Fold already completed')
    if len(set(d['state_index'][train].tolist()))!=192:raise ValueError('Conditional training split differs')
    model,diagnostic=fit_model(d['x'][train],d['y'][train],d['weight'][train]);model.public_cache_feature_mode=MODE;model.public_cache_context_rule=RULE
    prediction=np.zeros(int(test.sum()));prediction[~d['guarded'][test]]=model.predict_proba(d['x'][warm_test])[:,1]
    bundle=dict(model=model,fold=fold,features=p['features'],feature_mode=MODE,context_rule=RULE,training_state_indices=sorted(set(d['state_index'][train].tolist())),diagnostic=diagnostic,protocol_sha256=b.sha(a.results_root/'protocol.json'),deployment_requires_distinct_gap19_adapter=True);validate_bundle(bundle)
    if diagnostic['kind']=='weighted_standardized_logistic' and max(diagnostic['iterations'])>=1000:raise ValueError('Optimizer did not converge')
    out.mkdir(parents=True,exist_ok=True);(out/'model.pkl').write_bytes(pickle.dumps(bundle,protocol=5));restored=pickle.loads((out/'model.pkl').read_bytes());validate_bundle(restored)
    if not np.array_equal(restored['model'].predict_proba(d['x'][warm_test])[:,1],prediction[~d['guarded'][test]]):raise ValueError('Serialization changed predictions')
    np.savez_compressed(out/'test_predictions.npz',row_indices=np.flatnonzero(test),probability=prediction,guarded=d['guarded'][test]);b.write(out/'result.json',dict(status='PASS',fold=fold,diagnostic=diagnostic,train_states=192,train_rows=int(train.sum()),test_rows=int(test.sum()),protocol_sha256=b.sha(a.results_root/'protocol.json'),files={n:b.sha(out/n) for n in ('model.pkl','test_predictions.npz')}))


def analyze(a,p):
    import numpy as np
    summary,states=b.analyze(a,p,write_result=False);z=np.load(a.results_root/'training_data.npz');d={k:z[k] for k in z.files};pred=np.full(len(d['y']),np.nan)
    for fold in (a.results_root/'folds').iterdir():
        v=np.load(fold/'test_predictions.npz');pred[v['row_indices']]=v['probability']
    masks={'guarded':d['guarded'],'non_guarded':~d['guarded']};edge=np.load(a.decision_results/'edge_context.npz')
    if not np.array_equal(edge['state_index'],d['state_index']):raise ValueError('Fixed sensitive masks changed population')
    masks.update({k:edge[k] for k in ('sensitive_union_10','sensitive_union_20')});subsets={}
    probs=dict(learned=pred,**{m:d[m] for m in METHODS[1:]})
    for name,mask in masks.items():
        blocks={}
        for block in map(str,range(620001,620006)):
            take=mask&np.asarray([str(p['identities'][int(i)][1])==block for i in d['state_index']]);w=d['weight'][take];y=d['y'][take];metrics={}
            for m,prob in probs.items():
                pp=prob[take];c=np.clip(pp,1e-12,1-1e-12);metrics[m]=dict(brier=float(np.average((pp-y)**2,weights=w)),log_loss=float(np.average(-y*np.log(c)-(1-y)*np.log1p(-c),weights=w)))
            blocks[block]=metrics
        subsets[name]=dict(blocks=blocks,groups={m:{k:statistics.mean(v[m][k] for v in blocks.values()) for k in ('brier','log_loss')} for m in METHODS})
    best=min(METHODS[2:],key=lambda m:summary['groups'][m]['brier']);e=summary['brier_improvements']['legacy_learned'];control=summary['brier_improvements'][best];go=e['mean_improvement']>=.001 and e['positive']>=4 and control['mean_improvement']>0 and control['positive']>=4
    summary.update(status='PUBLIC_FETCH_GAP19_CONDITIONAL_PREDICTION_SCREEN',subsets=subsets,best_label_free_control=best,legacy_eighteen_dimensional_improvement=e)
    summary['decision'].update(cache_hit_target_promising_development_only=go,public_gap_feature_promising_development_only=go,matching_improvement_tested=False,closed_loop_improvement_tested=False,goal_complete=False)
    b.write(a.results_root/'analysis/summary.json',summary);b.write(a.results_root/'analysis/state_metrics.json',states);print(json.dumps(dict(groups=summary['groups'],legacy_improvement=e,decision=summary['decision'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','fit','train','analysis'))
    for name in ARGS:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--block',type=int);p.add_argument('--workload',type=int);a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('All feature construction, fitting and statistics run remotely')
    if a.mode=='train':b.pipeline(a,b.read(a.results_root/'protocol.json'))
    else:
        with b.res(a).CpuLease(a.resources,task=['cache_gap19',a.mode,a.block,a.workload],kind='environment'):
            if a.mode=='prepare':prepare(a)
            else:
                protocol=b.read(a.results_root/'protocol.json')
                if a.mode=='fit':fit(a,protocol)
                else:analyze(a,protocol)


if __name__=='__main__':main()
