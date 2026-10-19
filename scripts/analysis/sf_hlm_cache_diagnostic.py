"""Read-only encoding/decision diagnostics after cache-matching screening."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

PRIMARY='cache_probability_price'
CONTROL='half_proxy_price'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def run(a,out):
    import numpy as np
    sys.path.insert(0,str(a.release_root))
    from utils.sf_hlm_cache_calibration import eligible,cache_features,validate_bundle
    from utils.sf_hlm_cache_mixture import decode_row
    from utils.sf_hlm_cache_matching import CacheWeightKernel
    p=read(a.results_root/'protocol.json');summary=read(a.results_root/'analysis/summary.json');audit=read(a.validation_root/'native_raw_audit.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=sha(a.results_root/'analysis/summary.json'):raise ValueError('Unverified matching results')
    if p['smoke'] or summary['decision']['native_development_signal']:raise ValueError('Expected failed development increment screen')
    tp=read(a.cache_training/'protocol.json');data=np.load(a.cache_training/'training_data.npz');cache_audit=read(a.cache_validation/'raw_record_audit.json')
    if cache_audit['status']!='PASS' or cache_audit['summary_sha256']!=sha(a.cache_training/'analysis/summary.json'):raise ValueError('Unverified prediction source')
    predictions=np.full(len(data['y']),np.nan);seen=np.zeros(len(predictions),int)
    for path in (a.cache_training/'folds').glob('*/test_predictions.npz'):
        z=np.load(path);ii=z['row_indices'];predictions[ii]=z['probability'];seen[ii]+=1
    if not np.all(seen==1):raise ValueError('Prediction population changed')
    if tp['identities']!=p['identities']:raise ValueError('State order differs')
    mode=tp.get('feature_mode','clipped');full=np.log(data['prior'])-np.log1p(-data['prior'])
    encoded=np.clip(full,-12.,12.)/12 if mode=='clipped' else full/12
    if not np.allclose(encoded,data['x'][:,0],rtol=0,atol=1e-14):raise ValueError('Frozen encoding differs')
    full=np.log(data['prior'])-np.log1p(-data['prior']);saturated=np.abs(full)>12
    weights=data['weight'];truth=data['y'];half=data['half_proxy']
    def measures(mask):
        ww=weights[mask]
        return dict(rows=int(mask.sum()),original_state_weight=float(ww.sum()),states=int(len(set(data['state_index'][mask].tolist()))),
            hit_rate=float(np.average(truth[mask],weights=ww)) if len(ww) else None,
            mean_prediction=float(np.average(predictions[mask],weights=ww)) if len(ww) else None,
            mean_half_proxy=float(np.average(half[mask],weights=ww)) if len(ww) else None,
            learned_brier=float(np.average((predictions[mask]-truth[mask])**2,weights=ww)) if len(ww) else None,
            half_brier=float(np.average((half[mask]-truth[mask])**2,weights=ww)) if len(ww) else None)
    bins=[-np.inf,-12.,12.,18.,24.,30.,36.,48.,np.inf];strata={'own_invisible':data['x'][:,3]==0,'own_miss':(data['x'][:,3]==1)&(data['x'][:,2]==0),'own_hit':(data['x'][:,3]==1)&(data['x'][:,2]==1)}
    encoding=dict(rows=len(full),saturated_rows=int(saturated.sum()),saturated_row_fraction=float(saturated.mean()),
        saturated_original_weight_fraction=float(weights[saturated].sum()/weights.sum()),full_logit_range=[float(full.min()),float(full.max())],
        saturated=measures(saturated),unsaturated=measures(~saturated),bins={})
    for name,condition in strata.items():
        encoding['bins'][name]=[dict(bin=f'[{lo},{hi})',**measures(condition&(full>=lo)&(full<hi))) for lo,hi in zip(bins,bins[1:])]
    sensitive={PRIMARY:np.zeros(len(full),bool),CONTROL:np.zeros(len(full),bool)};selected={name:np.zeros(len(full),bool) for name in sensitive};categories={name:dict(states=0,raw={},fee={}) for name in ('same_assignment','same_quota_different_assignment','different_quota')};temperatures={name:dict(states=0,raw={},fee={}) for name in ('cold_step0','later_step')};den={};details=[];models={};reference=None;reference_summary=None
    if getattr(a,'reference_launch',None):
        ra=read(a.reference_launch)['args'];reference=Path(ra['results_root']);reference_summary=read(reference/'analysis/summary.json');reference_audit=read(Path(ra['validation_root'])/'native_raw_audit.json')
        if reference_audit['status']!='PASS' or reference_audit['summary_sha256']!=sha(reference/'analysis/summary.json') or read(reference/'protocol.json')['identities']!=p['identities']:raise ValueError('Reference is not an audited identical state population')
    for si,entry in enumerate(p['states']):
        identity=entry['identity'];td=a.results_root/'tasks'/('cache_'+'_'.join(map(str,identity)));pub=read(td/'public.json');timing=read(td/'timing.json');labels=read(td/'labels.json')
        for name,digest in read(td/'result.json')['files'].items():
            if sha(td/name)!=digest:raise ValueError('Audited native record changed')
        if identity!=tp['identities'][si]:raise ValueError('Diagnostic state mismatch')
        rows=read(entry['public_rows'])['rows']
        for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
        refs={name:next(z for z in pub['selections'] if z['method']==name) for name in sensitive}
        counts={name:Counter(row['options'][int(action)]['target'] for row,action in zip(rows,ref['assignment'])) for name,ref in refs.items()}
        category='same_assignment' if refs[PRIMARY]['assignment']==refs[CONTROL]['assignment'] else 'same_quota_different_assignment' if counts[PRIMARY]==counts[CONTROL] else 'different_quota'
        scores={}
        for name,ref in refs.items():
            rr=labels['outcomes'][labels['mapping'][name+':0']]['requests'];ts=next(z['software_seconds'] for z in timing['records'] if z['method']==name and z['repeat']==0)
            scores[name]=dict(raw=sum(r['latency']<=r['deadline'] for r in rr),nominal_10ms=sum(r['latency']+.010<=r['deadline'] for r in rr),fee=statistics.mean(sum(r['latency']+t<=r['deadline'] for r in rr) for t in ts))
        block=str(identity[1]);den[block]=den.get(block,0)+labels['generated'];temperature='cold_step0' if identity[4]==0 else 'later_step'
        for group in (categories[category],temperatures[temperature]):
            group['states']+=1
            for metric in ('raw','fee'):group[metric][block]=group[metric].get(block,0)+scores[PRIMARY][metric]-scores[CONTROL][metric]
        indices=np.flatnonzero(data['state_index']==si);k=0;fold_name=f'block_{identity[1]}_workload_{identity[2]}';price=read(a.calibration_root/'folds'/fold_name/'prices.json');pw=price['powers']
        for ri,row in enumerate(rows):
            for action,part in decode_row(row,pw).items():
                if not eligible(part):continue
                j=int(indices[k]);k+=1
                if not np.array_equal(np.asarray(cache_features(row,action,mode=mode)),data['x'][j]):raise ValueError('Diagnostic edge alignment differs')
                target=row['options'][action]['target']
                for name in sensitive:
                    q=counts[name][target]
                    sensitive[name][j]=q>0 and part['hit_base']+part['work']*q+.010<=row['deadline']<part['miss_base']+part['work']*q+.010
                    selected[name][j]=int(refs[name]['assignment'][ri])==action
        if k!=len(indices):raise ValueError('Diagnostic eligibility differs')
        cross={}
        if fold_name not in models:
            path=a.cache_training/'folds'/fold_name/'model.pkl';bundle=pickle.loads(path.read_bytes())
            if sha(path)!=pub['locks']['cache_model_sha256'] or validate_bundle(bundle)!=mode:raise ValueError('Diagnostic scorer changed')
            models[fold_name]=bundle['model']
        quotas={name:[counts[name][j] for j in range(5)]+[counts[name]['mbs']] for name in sensitive}
        for scorer,variant in ((PRIMARY,'learned'),(CONTROL,'half_proxy')):
            kernel=CacheWeightKernel(rows,pw,variant,models[fold_name]);matrices=kernel.probabilities(list(quotas.values()));parameters=price['prices'][variant];cross[scorer]={}
            for name,ref in refs.items():
                matrix=matrices[tuple(quotas[name])];targets=[5 if row['options'][int(action)]['target']=='mbs' else row['options'][int(action)]['target'] for row,action in zip(rows,ref['assignment'])];value=float(sum(matrix[i,j] for i,j in enumerate(targets)));energy=ref['diagnostic']['chosen_predicted']['energy']
                cross[scorer][name]=dict(expected_success_10ms=value,public_energy=energy,priced_score=value-parameters['price']*energy/parameters['normalizer'])
                if name==scorer and not math.isclose(value,ref['diagnostic']['expected_success'],abs_tol=1e-10,rel_tol=0):raise ValueError('Cross-score does not reproduce sealed selection')
        old_change=None
        if reference is not None:
            rd=reference/'tasks'/td.name;old_pub=read(rd/'public.json');old_labels=read(rd/'labels.json')
            for name,digest in read(rd/'result.json')['files'].items():
                if sha(rd/name)!=digest:raise ValueError('Reference native record changed')
            old_ref=next(z for z in old_pub['selections'] if z['method']==PRIMARY);old_requests=old_labels['outcomes'][old_labels['mapping'][PRIMARY+':0']]['requests'];old_counts=Counter(row['options'][int(action)]['target'] for row,action in zip(rows,old_ref['assignment']))
            old_change=dict(assignment_changed=old_ref['assignment']!=refs[PRIMARY]['assignment'],quota_changed=old_counts!=counts[PRIMARY],raw_delta=scores[PRIMARY]['raw']-sum(r['latency']<=r['deadline'] for r in old_requests))
        details.append(dict(identity=identity,category=category,temperature=temperature,
            changed_requests=sum(x!=y for x,y in zip(refs[PRIMARY]['assignment'],refs[CONTROL]['assignment'])),
            raw_delta=scores[PRIMARY]['raw']-scores[CONTROL]['raw'],fee_delta=scores[PRIMARY]['fee']-scores[CONTROL]['fee'],nominal_delta=scores[PRIMARY]['nominal_10ms']-scores[CONTROL]['nominal_10ms'],cross_scores=cross,reference_change=old_change))
    def contribution(groups):
        return {name:dict(states=g['states'],**{metric+'_contribution_pp':statistics.mean(g[metric].get(block,0)/value*100 for block,value in den.items()) for metric in ('raw','fee')}) for name,g in groups.items()}
    category_output=contribution(categories);temperature_output=contribution(temperatures)
    cross_blocks={};old_blocks={}
    for block,denominator in den.items():
        rr=[d for d in details if str(d['identity'][1])==block];cross_blocks[block]={}
        for scorer in sensitive:
            cross_blocks[block][scorer]=dict(expected_success_delta_pp=sum(d['cross_scores'][scorer][PRIMARY]['expected_success_10ms']-d['cross_scores'][scorer][CONTROL]['expected_success_10ms'] for d in rr)/denominator*100,
                actual_nominal_delta_pp=sum(d['nominal_delta'] for d in rr)/denominator*100,
                primary_has_lower_priced_score_states=sum(d['cross_scores'][scorer][PRIMARY]['priced_score']<d['cross_scores'][scorer][CONTROL]['priced_score']-1e-10 for d in rr))
        if reference is not None:old_blocks[block]=dict(raw_delta_pp=sum(d['reference_change']['raw_delta'] for d in rr)/denominator*100,assignment_changed_states=sum(d['reference_change']['assignment_changed'] for d in rr),quota_changed_states=sum(d['reference_change']['quota_changed'] for d in rr))
    if reference is not None:
        difference=(summary['subsets']['all']['groups'][PRIMARY]['raw_rate']-reference_summary['subsets']['all']['groups'][PRIMARY]['raw_rate'])*100
        if not math.isclose(statistics.mean(d['raw_delta_pp'] for d in old_blocks.values()),difference,abs_tol=1e-10,rel_tol=0):raise ValueError('Reference raw decomposition differs')
    for metric in ('raw','fee'):
        wanted=summary['subsets']['all']['comparisons'][CONTROL][metric]['mean']*100
        for groups in (category_output,temperature_output):
            if not math.isclose(sum(g[metric+'_contribution_pp'] for g in groups.values()),wanted,rel_tol=0,abs_tol=1e-10):raise ValueError('Decision decomposition does not add back')
    result=dict(status='READ_ONLY_DEVELOPMENT_DIAGNOSTIC',source_summary_sha256=sha(a.results_root/'analysis/summary.json'),source_audit_sha256=sha(a.validation_root/'native_raw_audit.json'),
        script_sha256=sha(__file__),feature_mode=mode,encoding=encoding,decision_categories=category_output,temperature_categories=temperature_output,
        selected_cooperative_edges={name:measures(mask) for name,mask in selected.items()},cross_scores_by_block=cross_blocks,reference_changes_by_block=old_blocks,
        decision_sensitive={name:dict(all=measures(mask),saturated=measures(mask&saturated),unsaturated=measures(mask&~saturated)) for name,mask in sensitive.items()},
        decision_decomposition_adds_to_primary_result=True,original_state_weights_retained=True,models_retrained=False,independent_confirmation=False,changes_stage27_gate=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Immutable diagnostic output')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');out.with_name(out.stem+'_states.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],saturated_row_fraction=encoding['saturated_row_fraction'],full_logit_range=encoding['full_logit_range'],
        decision_categories=category_output,decision_sensitive=result['decision_sensitive'],selected_cooperative_edges=result['selected_cooperative_edges'],cross_scores_by_block=cross_blocks,reference_changes_by_block=old_blocks)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reference-launch',type=Path);args=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Diagnostic numerical analysis runs remotely')
    a=argparse.Namespace(**{key:Path(value) for key,value in read(args.launch)['args'].items()})
    a.reference_launch=args.reference_launch
    r=load('cache_diagnostic_resources',a.resource_module)
    with r.CpuLease(a.resources,task=['cache_encoding_decision_diagnostic'],kind='environment'):run(a,args.out)


if __name__=='__main__':main()
