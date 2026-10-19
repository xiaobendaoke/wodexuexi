"""One prespecified public-rule/conditional-cache calibration experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import sf_hlm_cache_calibration as base

RULE='all_visible_miss_zero_v1'
METHODS=('learned','prior','own_proxy','half_proxy','cold_guard_prior','cold_guard_own','cold_guard_half')
FILES=('paths.py','scripts/analysis/sf_hlm_cache_regime.py','scripts/analysis/sf_hlm_cache_calibration.py',
    'utils/sf_hlm_cache_calibration.py','utils/sf_hlm_cache_context.py','utils/sf_hlm_cache_mixture.py',
    'utils/sf_hlm_selector.py','tests/test_sf_hlm_cache_calibration.py','tests/test_sf_hlm_cache_context.py',
    'docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=base.ARGS+('source_training','source_validation','context_results','context_validation')
base.FILES=FILES;base.ARGS=ARGS;base.METHODS=METHODS


def prepare(a):
    import numpy as np
    from utils.sf_hlm_cache_calibration import MODEL,feature_names
    from utils.sf_hlm_cache_context import public_context
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable conditional-calibration namespace')
    old=base.read(a.source_training/'protocol.json');prep=base.read(a.source_training/'prepared.json');audit=base.read(a.source_validation/'raw_record_audit.json');context=base.read(a.context_results/'analysis.json');context_audit=base.read(a.context_validation/'raw_record_audit.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=base.sha(a.source_training/'analysis/summary.json') or context_audit['status']!='PASS' or context_audit['analysis_sha256']!=base.sha(a.context_results/'analysis.json'):raise ValueError('Precursors are not audited')
    if old['model']!=MODEL or old['feature_mode']!='full_logit_context' or context['models_retrained'] or context['matching_called']:raise ValueError('Unexpected precursor recipe')
    archive=np.load(a.source_training/'training_data.npz');data={k:archive[k].copy() for k in archive.files}
    if base.sha(a.source_training/'training_data.npz')!=prep['data_sha256'] or len(data['y'])!=13459:raise ValueError('Original arrays changed')
    source_contexts=base.read(a.context_results/'states.json');guard_states=[];inputs={str(p):base.sha(p) for p in (a.source_training/'protocol.json',a.source_training/'prepared.json',a.source_training/'training_data.npz',a.source_training/'layout.json',a.source_training/'analysis/summary.json',a.source_validation/'raw_record_audit.json',a.context_results/'protocol.json',a.context_results/'analysis.json',a.context_results/'states.json',a.context_validation/'raw_record_audit.json')}
    for i,state in enumerate(old['states']):
        inputs[state['public']]=base.sha(state['public']);observed=public_context(base.read(state['public'])['rows'])
        if source_contexts[i]['identity']!=old['identities'][i] or source_contexts[i]['context']!=observed:raise ValueError('Public rule reconstruction differs')
        guard_states.append(observed['global']['all_visible_miss'])
    data['guarded']=np.asarray([guard_states[int(i)] for i in data['state_index']],dtype=bool)
    for output,source in (('cold_guard_prior','prior'),('cold_guard_own','own_proxy'),('cold_guard_half','half_proxy')):
        data[output]=data[source].copy();data[output][data['guarded']]=0.
    for key in archive.files:
        if not np.array_equal(data[key],archive[key]):raise ValueError('Original feature/label/weight/control changed')
    for folder in (a.source_training/'folds').iterdir():
        if folder.is_dir():
            receipt=base.read(folder/'result.json');inputs[str(folder/'result.json')]=base.sha(folder/'result.json')
            for name,digest in receipt['files'].items():
                if base.sha(folder/name)!=digest:raise ValueError('Legacy reference changed')
                inputs[str(folder/name)]=digest
    p=dict(name='sf_hlm_public_guard_conditional_cache_calibration_20261018_v1',files={f:base.sha(a.release_root/f) for f in FILES},input_files=inputs,
        identities=old['identities'],states=old['states'],features=feature_names('full_logit_context'),feature_mode='full_logit_context',context_rule=RULE,
        model=MODEL,methods=METHODS,training_states=400,guarded_states=sum(guard_states),training_filter='not public guarded state',
        split=old['split'],labels_features_weights_unchanged=True,private_cache_used_for_guard=False,independent_confirmation=False,goal_complete=False)
    base.write(a.results_root/'protocol.json',p);np.savez_compressed(a.results_root/'training_data.npz',**data);base.write(a.results_root/'layout.json',base.read(a.source_training/'layout.json'))
    base.write(a.results_root/'prepared.json',dict(status='PASS',rows=len(data['y']),source_states=400,eligible_states=400,guarded_states=sum(guard_states),data_sha256=base.sha(a.results_root/'training_data.npz'),layout_sha256=base.sha(a.results_root/'layout.json'),source_prepared_sha256=base.sha(a.source_training/'prepared.json')))
    print(json.dumps(dict(status='PREPARED',states=400,guarded_states=sum(guard_states),feature_dimensions=18)),flush=True)


def fit(a,p):
    import numpy as np
    from utils import sf_hlm_cache_calibration as cc
    base.verify(a,p);prepared=base.read(a.results_root/'prepared.json')
    if base.sha(a.results_root/'training_data.npz')!=prepared['data_sha256']:raise ValueError('Frozen fitting data changed')
    archive=np.load(a.results_root/'training_data.npz');data={k:archive[k] for k in archive.files};fold=next(f for f in base.splits(a,p) if f['block']==a.block and f['workload']==a.workload);out=base.folded(a,a.block,a.workload)
    if (out/'result.json').exists():raise FileExistsError('Conditional fold already completed')
    all_train=np.isin(data['state_index'],fold['train']);train=all_train&~data['guarded'];test=np.isin(data['state_index'],fold['test']);warm_test=test&~data['guarded'];train_states=sorted(set(data['state_index'][train].tolist()))
    if len(train_states)!=192 or p['context_rule']!=RULE:raise ValueError('Unexpected public fitting population')
    model,diagnostic=cc.fit_model(data['x'][train],data['y'][train],data['weight'][train]);model.public_cache_feature_mode=p['feature_mode'];model.public_cache_context_rule=RULE
    pred=np.zeros(int(test.sum()),dtype=float);test_guard=data['guarded'][test];pred[~test_guard]=model.predict_proba(data['x'][warm_test])[:,1]
    bundle=dict(model=model,fold=fold,features=p['features'],feature_mode=p['feature_mode'],context_rule=RULE,training_state_indices=train_states,protocol_sha256=base.sha(a.results_root/'protocol.json'),diagnostic=diagnostic,deployment_requires_explicit_context_adapter=True)
    cc.validate_bundle(bundle,supported_context_rules=('none',RULE));out.mkdir(parents=True,exist_ok=True);path=out/'model.pkl';path.write_bytes(pickle.dumps(bundle,protocol=5));restored=pickle.loads(path.read_bytes())
    if not np.array_equal(pred[~test_guard],restored['model'].predict_proba(data['x'][warm_test])[:,1]):raise ValueError('Serialization changed non-guarded prediction')
    if diagnostic['kind']=='weighted_standardized_logistic' and max(diagnostic['iterations'])>=p['model']['max_iter']:raise ValueError('Fixed optimizer failed to converge')
    np.savez_compressed(out/'test_predictions.npz',row_indices=np.flatnonzero(test),probability=pred,guarded=test_guard)
    base.write(out/'result.json',dict(status='PASS',fold=fold,diagnostic=diagnostic,train_rows=int(train.sum()),excluded_guarded_train_rows=int((all_train&data['guarded']).sum()),effective_training_states=len(train_states),test_rows=int(test.sum()),test_guarded_rows=int(test_guard.sum()),
        protocol_sha256=base.sha(a.results_root/'protocol.json'),files={n:base.sha(out/n) for n in ('model.pkl','test_predictions.npz')}))


def analyze(a,p):
    import numpy as np
    from scipy.stats import t
    summary,states=base.analyze(a,p,write_result=False);archive=np.load(a.results_root/'training_data.npz');data={k:archive[k] for k in archive.files};pred=np.full(len(data['y']),np.nan);legacy=pred.copy();seen=np.zeros(len(pred),int)
    for fold in base.splits(a,p):
        name=f"block_{fold['block']}_workload_{fold['workload']}";left=np.load(a.results_root/'folds'/name/'test_predictions.npz');right=np.load(a.source_training/'folds'/name/'test_predictions.npz');ii=left['row_indices']
        if not np.array_equal(ii,right['row_indices']) or not np.array_equal(left['guarded'],data['guarded'][ii]):raise ValueError('Prediction populations differ')
        pred[ii]=left['probability'];legacy[ii]=right['probability'];seen[ii]+=1
    if not np.all(seen==1) or not np.all(pred[data['guarded']]==0):raise ValueError('Public gate or coverage differs')
    predictions=dict(learned=pred,legacy_mixed=legacy,**{k:data[k] for k in METHODS[1:]});subsets={}
    for name,mask in (('guarded',data['guarded']),('non_guarded',~data['guarded']),('step0',np.asarray([p['identities'][int(i)][4]==0 for i in data['state_index']])),('later',np.asarray([p['identities'][int(i)][4]!=0 for i in data['state_index']]))):
        blocks={}
        for block in range(620001,620006):
            take=mask&np.asarray([p['identities'][int(i)][1]==block for i in data['state_index']]);yy=data['y'][take];ww=data['weight'][take];metrics={}
            for method,pp in predictions.items():
                pr=pp[take];clip=np.clip(pr,1e-12,1-1e-12);metrics[method]=dict(brier=float(np.average((pr-yy)**2,weights=ww)),log_loss=float(np.average(-yy*np.log(clip)-(1-yy)*np.log1p(-clip),weights=ww)))
            blocks[str(block)]=metrics
        groups={method:{metric:statistics.mean(b[method][metric] for b in blocks.values()) for metric in ('brier','log_loss')} for method in predictions};subsets[name]=dict(blocks=blocks,groups=groups)
    values=[b['prior']['brier']-b['learned']['brier'] for b in subsets['non_guarded']['blocks'].values()];mean=statistics.mean(values);radius=float(t.ppf(.975,4))*statistics.stdev(values)/5**.5;effect=dict(mean_improvement=mean,ci95=[mean-radius,mean+radius],values=values,positive=sum(v>0 for v in values),n=5)
    global_go=summary['decision']['cache_hit_target_promising_development_only'];warm_go=mean>0 and effect['positive']>=4;go=global_go and warm_go
    summary.update(status='CONDITIONAL_PUBLIC_CACHE_CALIBRATION_DEVELOPMENT_SCREEN',context_rule=RULE,subsets=subsets,non_guarded_vs_prior=effect)
    summary['decision'].update(cache_hit_target_promising_development_only=go,conditional_target_promising_development_only=go,beats_six_label_free_controls=global_go,beats_non_guarded_prior=warm_go,matching_improvement_tested=False,closed_loop_improvement_tested=False,goal_complete=False)
    base.write(a.results_root/'analysis/summary.json',summary);base.write(a.results_root/'analysis/state_metrics.json',states);print(json.dumps(dict(groups=summary['groups'],non_guarded_vs_prior=effect,decision=summary['decision'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','fit','train','analysis'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--block',type=int);parser.add_argument('--workload',type=int);a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Data preparation, fitting and statistics run remotely')
    if a.mode=='train':base.pipeline(a,base.read(a.results_root/'protocol.json'))
    else:
        with base.res(a).CpuLease(a.resources,task=['cache_regime',a.mode,a.block,a.workload],kind='environment'):
            if a.mode=='prepare':prepare(a)
            else:
                p=base.read(a.results_root/'protocol.json');base.verify(a,p)
                if a.mode=='fit':fit(a,p)
                else:analyze(a,p)


if __name__=='__main__':main()
