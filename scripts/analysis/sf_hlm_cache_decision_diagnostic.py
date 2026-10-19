"""Read-only held-out cache errors at actual deadline-relevant decisions."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

METHODS=('cache_regime_probability_price','cold_half_proxy_price','cold_prior_price')
VARIANTS=('learned','cold_half','cold_prior')
FILES=('paths.py','scripts/analysis/sf_hlm_cache_decision_diagnostic.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def run(a,out):
    import numpy as np
    from scripts.analysis import sf_hlm_cache_matching as b
    from utils.sf_hlm_cache_calibration import cache_features,eligible,validate_bundle
    from utils.sf_hlm_cache_mixture import decode_row
    from utils.sf_hlm_cache_matching import CacheWeightKernel
    if (out/'protocol.json').exists():raise FileExistsError('Immutable diagnostic namespace')
    source=a.source_results;p=b.read(source/'protocol.json');s=b.read(source/'analysis/summary.json')
    audit=b.read(a.source_validation/'common_delay_raw_audit.json');tp=b.read(a.cache_training/'protocol.json')
    ca=b.read(a.cache_validation/'raw_record_audit.json');prepared=b.read(a.cache_training/'prepared.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=b.sha(source/'analysis/summary.json') or s['decision']['common_delay_screen_pass']:raise ValueError('Expected audited failed source')
    if ca['status']!='PASS' or ca['summary_sha256']!=b.sha(a.cache_training/'analysis/summary.json') or b.sha(a.cache_training/'training_data.npz')!=prepared['data_sha256']:raise ValueError('Prediction data not audited')
    if tp['identities']!=p['identities']:raise ValueError('State populations differ')
    for path,digest in {**p['input_files'],**p['calibration_files']}.items():
        if b.sha(path)!=digest:raise ValueError('Frozen matching input changed')
    f=b.archive(a);f.boot(a,620001);b.modules(a)
    import config
    from environment import uavs,user_equipments
    own=Path(__file__).resolve().parents[2]
    inputs={str(x):b.sha(x) for x in (source/'protocol.json',source/'analysis/summary.json',a.source_validation/'common_delay_raw_audit.json',a.cache_training/'protocol.json',a.cache_training/'prepared.json',a.cache_training/'training_data.npz',a.cache_training/'analysis/summary.json',a.cache_validation/'raw_record_audit.json')}
    sources={str(Path(m.__file__).resolve()):b.sha(m.__file__) for m in (uavs,user_equipments,config)}
    inputs.update(sources)
    for state in p['states']:
        td=source/'tasks'/('cache_'+'_'.join(map(str,state['identity'])))
        for x in (td/'result.json',td/'public.json',td/'labels.json',Path(state['public_rows'])):inputs[str(x)]=b.sha(x)
    for fold in (a.cache_training/'folds').iterdir():
        for name,digest in b.read(fold/'result.json')['files'].items():
            if b.sha(fold/name)!=digest:raise ValueError('Prediction fold changed')
        for name in ('model.pkl','test_predictions.npz','result.json'):inputs[str(fold/name)]=b.sha(fold/name)
        price=a.calibration_root/'folds'/fold.name/'prices.json';inputs[str(price)]=b.sha(price)
    protocol=dict(name='sf_hlm_cache_decision_relevance_diagnostic_20261018_v1',files={x:b.sha(own/x) for x in FILES},input_files=inputs,methods=METHODS,states=400,
        delays_seconds=[.010,.020],original_state_weights=True,models_retrained=False,matching_called=False,physical_simulation_called=False,native_timing_called=False,independent_confirmation=False)
    b.write(out/'protocol.json',protocol)
    excerpts={}
    for mod in (uavs,user_equipments,config):
        lines=Path(mod.__file__).read_text().splitlines();wanted=set()
        for i,line in enumerate(lines):
            if any(token in line for token in ('service_file_size','FILE_SIZES','def gdsf_cache_update')):
                wanted.update(range(max(0,i-3),min(len(lines),i+5)))
        excerpts[str(Path(mod.__file__).resolve())]=[dict(line=i+1,text=lines[i]) for i in sorted(wanted)]
    b.write(out/'source_inventory.json',dict(files=sources,excerpts=excerpts,old_feature_names=tp['features'],note='Source availability only; no new online feature or communication contract is claimed'))
    archive=np.load(a.cache_training/'training_data.npz');d={k:archive[k] for k in archive.files};n=len(d['y'])
    learned=np.full(n,np.nan);seen=np.zeros(n,int)
    for fold in (a.cache_training/'folds').iterdir():
        z=np.load(fold/'test_predictions.npz');ii=z['row_indices'];learned[ii]=z['probability'];seen[ii]+=1
    if not np.all(seen==1):raise ValueError('OOF population coverage changed')
    pp=dict(learned=learned,**{name:d[name] for name in ('prior','own_proxy','half_proxy','cold_guard_prior','cold_guard_own','cold_guard_half')})
    sensitive={m:{str(ms):np.zeros(n,bool) for ms in (10,20)} for m in METHODS};selected={m:np.zeros(n,bool) for m in METHODS}
    gaps=np.zeros(n);row_numbers=np.zeros(n,int);actions=np.zeros(n,int);states=[];models={};all_option_fields=set();all_row_fields=set();checks=0
    for si,entry in enumerate(p['states']):
        identity=entry['identity'];td=source/'tasks'/('cache_'+'_'.join(map(str,identity)));pub=b.read(td/'public.json');labels=b.read(td/'labels.json');receipt=b.read(td/'result.json')
        for name,digest in receipt['files'].items():
            if b.sha(td/name)!=digest:raise ValueError('Sealed source record changed')
        if pub['locks']['cache_model_sha256']!=b.sha(a.cache_training/'folds'/b.fold_name(identity[1],identity[2])/'model.pkl'):raise ValueError('Selection model differs from OOF model')
        rows=b.read(entry['public_rows'])['rows']
        for row in rows:
            row['options']={int(k):v for k,v in row['options'].items()};all_row_fields.update(row)
            for opt in row['options'].values():all_option_fields.update(opt)
        refs={m:next(x for x in pub['selections'] if x['method']==m) for m in METHODS}
        counts={m:Counter(row['options'][int(ac)]['target'] for row,ac in zip(rows,refs[m]['assignment'])) for m in METHODS}
        indices=np.flatnonzero(d['state_index']==si);k=0;name=b.fold_name(identity[1],identity[2]);price=b.read(a.calibration_root/'folds'/name/'prices.json')
        for ri,row in enumerate(rows):
            for action,part in decode_row(row,price['powers']).items():
                if not eligible(part):continue
                j=int(indices[k]);k+=1
                if not np.array_equal(np.asarray(cache_features(row,action,mode='full_logit_context')),d['x'][j]):raise ValueError('Edge alignment differs')
                row_numbers[j]=ri;actions[j]=action;gaps[j]=part['miss_base']-part['hit_base']
                for method in METHODS:
                    q=counts[method][row['options'][action]['target']]
                    selected[method][j]=int(refs[method]['assignment'][ri])==action
                    for ms in (10,20):sensitive[method][str(ms)][j]=q>0 and part['hit_base']+part['work']*q+ms/1000<=row['deadline']<part['miss_base']+part['work']*q+ms/1000
        if k!=len(indices):raise ValueError('Eligibility population differs')
        if name not in models:
            bundle=pickle.loads((a.cache_training/'folds'/name/'model.pkl').read_bytes());validate_bundle(bundle,supported_context_rules=('none','all_visible_miss_zero_v1'));models[name]=bundle['model']
        quotas={m:[counts[m][j] for j in range(5)]+[counts[m]['mbs']] for m in METHODS};cross={}
        for variant in VARIANTS:
            kernel=CacheWeightKernel(rows,price['powers'],variant,models[name]);matrices=kernel.probabilities(list(quotas.values()));cross[variant]={}
            for m in METHODS:
                matrix=matrices[tuple(quotas[m])];targets=[5 if row['options'][int(ac)]['target']=='mbs' else row['options'][int(ac)]['target'] for row,ac in zip(rows,refs[m]['assignment'])]
                value=float(sum(matrix[i,j] for i,j in enumerate(targets)));energy=refs[m]['diagnostic']['chosen_predicted']['energy'];param=price['prices'][variant]
                cross[variant][m]=dict(expected_success_10ms=value,priced_score=value-param['price']*energy/param['normalizer'])
                if m==METHODS[VARIANTS.index(variant)]:
                    if not math.isclose(value,refs[m]['diagnostic']['expected_success'],abs_tol=1e-10,rel_tol=0):raise ValueError('Sealed self-score differs')
                    checks+=1
        for j in indices:
            ri=int(row_numbers[j]);ac=int(actions[j])
            if kernel.parts[ri][ac]['p']!=float(d['cold_guard_prior'][j]):raise ValueError('Guarded prior OOF differs')
        # Both assignments remain the sealed choices. No oracle selects a method here.
        actual={m:{str(ms):sum(x['latency']+ms/1000<=x['deadline'] for x in labels['outcomes'][labels['mapping'][m+':0']]['requests']) for ms in (0,10,20,30)} for m in METHODS}
        categories={m:'same_assignment' if refs[m]['assignment']==refs[METHODS[0]]['assignment'] else 'same_quota_different_assignment' if counts[m]==counts[METHODS[0]] else 'different_quota' for m in METHODS[1:]}
        states.append(dict(identity=identity,generated=labels['generated'],actual=actual,cross=cross,categories=categories,temperature='step0' if identity[4]==0 else 'later'))
    blocks=sorted({str(x['identity'][1]) for x in states});den={k:sum(x['generated'] for x in states if str(x['identity'][1])==k) for k in blocks}
    def measures(mask):
        if not np.any(mask):return dict(rows=0,states=0,original_weight=0.,hit_rate=None,methods={})
        w=d['weight'][mask];y=d['y'][mask];metrics={}
        for name,prob in pp.items():
            p1=prob[mask];clipped=np.clip(p1,1e-12,1-1e-12);metrics[name]=dict(brier=float(np.average((p1-y)**2,weights=w)),log_loss=float(np.average(-y*np.log(clipped)-(1-y)*np.log1p(-clipped),weights=w)))
        return dict(rows=int(mask.sum()),states=len(set(d['state_index'][mask].tolist())),original_weight=float(w.sum()),hit_rate=float(np.average(y,weights=w)),methods=metrics)
    union={str(ms):np.logical_or.reduce([sensitive[m][str(ms)] for m in METHODS]) for ms in (10,20)}
    masks={'all':np.ones(n,bool),'non_guarded':~d['guarded'],'guarded':d['guarded']}
    for m in METHODS:
        masks['selected_'+m]=selected[m]
        for ms in (10,20):
            masks['sensitive_'+str(ms)+'_'+m]=sensitive[m][str(ms)];masks['selected_sensitive_'+str(ms)+'_'+m]=selected[m]&sensitive[m][str(ms)]
    for ms in (10,20):masks['sensitive_union_'+str(ms)]=union[str(ms)]
    own={'own_missing':d['x'][:,3]==0,'own_miss':(d['x'][:,3]==1)&(d['x'][:,2]==0),'own_hit':(d['x'][:,3]==1)&(d['x'][:,2]==1)}
    for name,mask in own.items():
        masks[name]=mask
        for ms in (10,20):masks[name+'_sensitive_'+str(ms)]=mask&union[str(ms)]
    for lo,hi in ((0.,.01),(.01,.05),(.05,.1),(.1,.5),(.5,np.inf)):
        mask=(gaps>lo)&(gaps<=hi);key=f'fetch_gap_({lo},{hi}]';masks[key]=mask
        for ms in (10,20):masks[key+'_sensitive_'+str(ms)]=mask&union[str(ms)]
    metric_groups={name:measures(mask) for name,mask in masks.items()}
    by_block={block:{name:measures(mask&np.asarray([str(p['identities'][int(i)][1])==block for i in d['state_index']])) for name,mask in masks.items()} for block in blocks}
    original=b.read(a.cache_training/'analysis/summary.json')
    for name,metrics in metric_groups['all']['methods'].items():
        for key,value in metrics.items():
            if not math.isclose(value,original['groups'][name][key],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Original prediction loss does not reproduce')
    decompositions={};cross_blocks={}
    for control in METHODS[1:]:
        parts={}
        for group in ('same_assignment','same_quota_different_assignment','different_quota','step0','later'):
            rr=[x for x in states if x['categories'][control]==group or x['temperature']==group]
            parts[group]=dict(states=len(rr),contribution_pp={str(ms):statistics.mean(sum(x['actual'][METHODS[0]][str(ms)]-x['actual'][control][str(ms)] for x in rr if str(x['identity'][1])==block)/den[block]*100 for block in blocks) for ms in (0,10,20,30)})
        for ms in (0,10,20,30):
            wanted=(s['subsets']['all']['groups'][METHODS[0]]['rate'][str(ms)]-s['subsets']['all']['groups'][control]['rate'][str(ms)])*100
            for groups in (('same_assignment','same_quota_different_assignment','different_quota'),('step0','later')):
                if not math.isclose(sum(parts[k]['contribution_pp'][str(ms)] for k in groups),wanted,abs_tol=1e-10,rel_tol=0):raise ValueError('Contributions do not add back')
        decompositions[control]=parts;cross_blocks[control]={}
        for block in blocks:
            rr=[x for x in states if str(x['identity'][1])==block];cross_blocks[control][block]={}
            for variant in VARIANTS:
                cross_blocks[control][block][variant]=dict(expected_success_delta_10ms_pp=sum(x['cross'][variant][METHODS[0]]['expected_success_10ms']-x['cross'][variant][control]['expected_success_10ms'] for x in rr)/den[block]*100,
                    actual_delta_10ms_pp=sum(x['actual'][METHODS[0]]['10']-x['actual'][control]['10'] for x in rr)/den[block]*100,
                    learned_assignment_lower_priced_score_states=sum(x['cross'][variant][METHODS[0]]['priced_score']<x['cross'][variant][control]['priced_score']-1e-10 for x in rr))
    edge_data=dict(state_index=d['state_index'],row=row_numbers,action=actions,public_fetch_gap=gaps,**{'sensitive_union_'+k:v for k,v in union.items()})
    for m in METHODS:
        edge_data['selected_'+m]=selected[m]
        for ms in (10,20):edge_data['sensitive_'+str(ms)+'_'+m]=sensitive[m][str(ms)]
    np.savez_compressed(out/'edge_context.npz',**edge_data)
    result=dict(status='READ_ONLY_DEVELOPMENT_DIAGNOSTIC',protocol_sha256=b.sha(out/'protocol.json'),source_summary_sha256=b.sha(source/'analysis/summary.json'),source_inventory_sha256=b.sha(out/'source_inventory.json'),edge_context_sha256=b.sha(out/'edge_context.npz'),rows=n,states=400,groups=metric_groups,blocks=by_block,
        decision_decomposition=decompositions,cross_scores_by_block=cross_blocks,sealed_self_scores_reproduced=checks,decision_decomposition_adds_back=True,public_row_fields=sorted(all_row_fields),public_option_fields=sorted(all_option_fields),models_retrained=False,matching_called=False,goal_complete=False,independent_confirmation=False,descriptive_only=True)
    for path,digest in inputs.items():
        if b.sha(path)!=digest:raise ValueError('Diagnostic source changed')
    b.write(out/'states.json',states);b.write(out/'analysis.json',result);b.write(out/'completion.json',dict(status='COMPLETED_READ_ONLY',analysis_sha256=b.sha(out/'analysis.json'),states_sha256=b.sha(out/'states.json'),rows=n,states=400))
    print(json.dumps(dict(groups={k:metric_groups[k] for k in ('all','non_guarded','sensitive_union_10','sensitive_union_20')},decision_decomposition=decompositions,cross_scores_by_block=cross_blocks)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);opt=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('All numerical analysis runs on the experiment server')
    a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(opt.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
    from scripts.analysis import sf_hlm_cache_matching as b
    with b.resource(a).CpuLease(a.resources,task=['cache_decision_diagnostic'],kind='environment'):run(a,opt.out)


if __name__=='__main__':main()
