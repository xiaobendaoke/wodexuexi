"""Read-only public batch-cache context and actually loaded source inventory."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys

OWN_FILES=('paths.py','scripts/analysis/sf_hlm_cache_context.py','utils/sf_hlm_cache_context.py',
    'utils/sf_hlm_cache_calibration.py','tests/test_sf_hlm_cache_context.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);opt=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Source loading, data construction and statistics run remotely')
    a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(opt.launch.read_text())['args'].items()});own=Path(__file__).resolve().parents[2];sys.path.insert(0,str(a.release_root))
    from scripts.analysis import sf_hlm_cache_matching as b
    with b.resource(a).CpuLease(a.resources,task=['cache_context_diagnostic'],kind='environment'):
        if (opt.out/'protocol.json').exists():raise FileExistsError('Immutable cache context diagnostic namespace')
        summary=b.read(a.source_results/'analysis/summary.json');audit=b.read(a.source_validation/'native_raw_audit.json');tp=b.read(a.cache_training/'protocol.json');prepared=b.read(a.cache_training/'prepared.json')
        if audit['status']!='PASS' or audit['summary_sha256']!=b.sha(a.source_results/'analysis/summary.json') or summary['decision']['native_development_signal']:raise ValueError('Expected audited failed direct matching')
        if b.sha(a.cache_training/'training_data.npz')!=prepared['data_sha256']:raise ValueError('Audited cache label data changed')
        f=b.archive(a);f.boot(a,620001);b.modules(a)
        # The context helper comes from this small immutable diagnostic release.
        context_module=b.load('cache_context_public_helper',own/'utils/sf_hlm_cache_context.py')
        import numpy as np
        import config
        from environment import uavs,user_equipments,env
        sources={str(Path(m.__file__).resolve()):b.sha(m.__file__) for m in (uavs,user_equipments,env,config)}
        excerpts={}
        for name,func in (('neighbor_cache_prior',uavs._get_belief_probability),('uav_initial_state',uavs.UAV.__init__),('pending_cache_commit',uavs.UAV.commit_pending_cache),('periodic_cache_update',uavs.UAV.gdsf_cache_update),('popularity_initialization',user_equipments.UE.initialize_ue_class)):
            text,line=inspect.getsourcelines(func);excerpts[name]=dict(path=str(Path(inspect.getsourcefile(func)).resolve()),first_line=line,source=''.join(text))
        env_lines=Path(env.__file__).read_text().splitlines();excerpts['env_lifecycle_calls']=[dict(line=i+1,text=line) for i,line in enumerate(env_lines) if any(t in line for t in ('initialize_ue_class','gdsf_cache_update','T_CACHE_UPDATE_INTERVAL','update_ema_and_cache'))]
        inputs={str(p):b.sha(p) for p in (a.source_results/'analysis/summary.json',a.source_validation/'native_raw_audit.json',a.cache_training/'protocol.json',a.cache_training/'prepared.json',a.cache_training/'training_data.npz',opt.launch)};inputs.update(sources)
        for state in tp['states']:inputs[state['public']]=b.sha(state['public'])
        for path in (a.cache_training/'folds').glob('*/test_predictions.npz'):inputs[str(path)]=b.sha(path)
        protocol=dict(name='sf_hlm_public_batch_cache_context_diagnostic_20261018_v1',files={n:b.sha(own/n) for n in OWN_FILES},input_files=inputs,states=400,source_feature_mode=tp['feature_mode'],models_retrained=False,matching_called=False,native_timing_called=False,independent_confirmation=False)
        b.write(opt.out/'protocol.json',protocol);b.write(opt.out/'source_inventory.json',dict(files=sources,excerpts=excerpts,constants=dict(cache_update_interval=int(config.T_CACHE_UPDATE_INTERVAL),probability_gamma=float(config.PROB_GAMMA),average_file_size=float(config.AVG_FILE_SIZE))))
        data=np.load(a.cache_training/'training_data.npz');pred=np.full(len(data['y']),np.nan);seen=np.zeros(len(pred),int)
        for path in (a.cache_training/'folds').glob('*/test_predictions.npz'):
            z=np.load(path);ii=z['row_indices'];pred[ii]=z['probability'];seen[ii]+=1
        if not np.all(seen==1):raise ValueError('Held-out prediction coverage differs')
        contexts=[];fractions=np.full(len(pred),np.nan);source_fractions=fractions.copy();cold=np.zeros(len(pred),bool)
        owners=np.argmax(data['x'][:,8:13],axis=1)
        for si,state in enumerate(tp['states']):
            rows=b.read(state['public'])['rows'];context=context_module.public_context(rows);mask=data['state_index']==si;global_group=context['global'];fraction=global_group['fraction']
            if fraction is not None:fractions[mask]=fraction
            cold[mask]=global_group['all_visible_miss']
            for source in range(5):
                fraction=context['sources'][str(source)]['fraction']
                if fraction is not None:source_fractions[mask&(owners==source)]=fraction
            contexts.append(dict(identity=tp['identities'][si],context=context,eligible_edges=int(mask.sum())))
        guarded=data['half_proxy'].copy();guarded[cold]=0.;predictions=dict(learned_full=pred,prior=data['prior'],own_proxy=data['own_proxy'],half_proxy=data['half_proxy'],cold_guard_half=guarded);yy=data['y'];weights=data['weight']
        def measures(mask):
            ww=weights[mask];metrics={}
            for name,pp in predictions.items():
                pr=pp[mask];clipped=np.clip(pr,1e-12,1-1e-12);metrics[name]=dict(brier=float(np.average((pr-yy[mask])**2,weights=ww)) if len(ww) else None,log_loss=float(np.average(-yy[mask]*np.log(clipped)-(1-yy[mask])*np.log1p(-clipped),weights=ww)) if len(ww) else None)
            return dict(rows=int(mask.sum()),states=len(set(data['state_index'][mask].tolist())),original_weight=float(ww.sum()),hit_rate=float(np.average(yy[mask],weights=ww)) if len(ww) else None,methods=metrics)
        def bins(values):
            out={'none_visible':measures(np.isnan(values)),'all_visible_miss':measures(values==0)}
            for lo,hi in ((0.,.25),(.25,.5),(.5,.75),(.75,1.)):out[f'({lo},{hi}]']=measures((values>lo)&(values<=hi))
            return out
        zero=np.asarray([tp['identities'][int(i)][4]==0 for i in data['state_index']]);blocks={}
        for block in range(620001,620006):blocks[str(block)]=measures(np.asarray([tp['identities'][int(i)][1]==block for i in data['state_index']]))
        comparisons={name:f.paired([x['methods'][name]['brier']-x['methods']['cold_guard_half']['brier'] for x in blocks.values()]) for name in predictions if name!='cold_guard_half'}
        result=dict(status='READ_ONLY_PUBLIC_CONTEXT_DIAGNOSTIC',source_inventory_sha256=b.sha(opt.out/'source_inventory.json'),protocol_sha256=b.sha(opt.out/'protocol.json'),input_files_verified=True,all=measures(np.ones(len(pred),bool)),step0=measures(zero),later=measures(~zero),global_bins=bins(fractions),source_bins=bins(source_fractions),blocks=blocks,cold_guard_brier_improvement=comparisons,guarded_states=len(set(data['state_index'][cold].tolist())),guarded_later_states=len(set(data['state_index'][cold&~zero].tolist())),guarded_edges_with_true_neighbor_hit=int((cold&(yy==1)).sum()),models_retrained=False,matching_called=False,native_timing_called=False,goal_complete=False,independent_confirmation=False,descriptive_only=True)
        old_summary=b.read(a.cache_training/'analysis/summary.json')
        for name in ('learned_full','prior','own_proxy','half_proxy'):
            for metric,value in result['all']['methods'][name].items():
                if not math.isclose(value,old_summary['groups']['learned' if name=='learned_full' else name][metric],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Original prediction statistics do not reproduce')
        for path,digest in inputs.items():
            if b.sha(path)!=digest:raise ValueError('Diagnostic input changed during execution')
        b.write(opt.out/'analysis.json',result);b.write(opt.out/'states.json',contexts)
        b.write(opt.out/'completion.json',dict(status='COMPLETED_READ_ONLY',analysis_sha256=b.sha(opt.out/'analysis.json'),states=400,edges=len(pred)))
        print(json.dumps({key:result[key] for key in ('status','all','step0','later','guarded_states','guarded_later_states','guarded_edges_with_true_neighbor_hit')}),flush=True)


if __name__=='__main__':main()
