"""One prospective cache-encoding revision; reuse frozen target and fit recipe."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import sf_hlm_cache_calibration as base

MODE='full_logit_context'
FILES=('paths.py','scripts/analysis/sf_hlm_cache_encoding.py','scripts/analysis/sf_hlm_cache_calibration.py',
    'utils/sf_hlm_cache_calibration.py','utils/sf_hlm_cache_mixture.py','utils/sf_hlm_selector.py',
    'tests/test_sf_hlm_cache_calibration.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=base.ARGS+('source_training','source_validation','diagnostic_results')
base.FILES=FILES
base.ARGS=ARGS


def prepare(a):
    import numpy as np
    from utils.sf_hlm_cache_calibration import feature_names,cache_features,eligible,MODEL
    from utils.sf_hlm_cache_mixture import decode_row
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable encoding namespace')
    old=base.read(a.source_training/'protocol.json');prep=base.read(a.source_training/'prepared.json');audit=base.read(a.source_validation/'raw_record_audit.json');diag=base.read(a.diagnostic_results)
    if audit['status']!='PASS' or audit['summary_sha256']!=base.sha(a.source_training/'analysis/summary.json'):raise ValueError('Old predictor not audited')
    if diag['status']!='READ_ONLY_DEVELOPMENT_DIAGNOSTIC' or diag['models_retrained'] or diag['changes_stage27_gate']:raise ValueError('Expected read-only predecessor diagnostic')
    if old['model']!=MODEL or old['training_states']!=400 or prep['rows']!=13459:raise ValueError('Unexpected frozen training recipe')
    source=a.source_training/'training_data.npz'
    if base.sha(source)!=prep['data_sha256']:raise ValueError('Old data changed')
    inputs={str(p):base.sha(p) for p in (source,a.source_training/'protocol.json',a.source_training/'prepared.json',a.source_training/'layout.json',
        a.source_training/'preparation_audit.json',a.source_training/'analysis/summary.json',a.source_validation/'raw_record_audit.json',a.diagnostic_results)}
    for folder in (a.source_training/'folds').iterdir():
        if not folder.is_dir():continue
        rec=base.read(folder/'result.json')
        if rec['status']!='PASS' or rec['protocol_sha256']!=base.sha(a.source_training/'protocol.json'):raise ValueError('Old model receipt differs')
        inputs[str(folder/'result.json')]=base.sha(folder/'result.json')
        for name,digest in rec['files'].items():
            if base.sha(folder/name)!=digest:raise ValueError('Old model/prediction changed')
            inputs[str(folder/name)]=digest
    arc=np.load(source);data={k:arc[k].copy() for k in arc.files};original=data['x'].copy()
    phi=np.asarray([(math.log(float(p))-math.log1p(-float(p)))/12 for p in data['prior']],dtype=np.float64)
    data['x'][:,0]=phi;data['x'][:,4]=phi*original[:,2]
    if not np.array_equal(original[:,[j for j in range(18) if j not in (0,4)]],data['x'][:,[j for j in range(18) if j not in (0,4)]]):raise ValueError('Non-encoding inputs changed')
    if not np.isfinite(data['x']).all():raise ValueError('Nonfinite full logit')
    pw=base.read(a.source_training/'preparation_audit.json')['powers'];checked=0
    for si,state in enumerate(old['states']):
        path=state['public'];inputs[path]=base.sha(path);rows=base.read(path)['rows'];expected=[]
        for row in rows:
            row['options']={int(k):v for k,v in row['options'].items()}
            for action,part in decode_row(row,pw).items():
                if eligible(part):expected.append(cache_features(row,action,mode=MODE))
        mask=data['state_index']==si
        if not np.array_equal(np.asarray(expected,dtype=np.float64),data['x'][mask]):raise ValueError('Deployment encoding differs from derived data')
        checked+=len(expected)
    p=dict(name='sf_hlm_full_logit_context_encoding_20261018_v1',files={f:base.sha(a.release_root/f) for f in FILES},input_files=inputs,
        identities=old['identities'],states=old['states'],features=feature_names(MODE),feature_mode=MODE,model=MODEL,methods=base.METHODS,
        training_states=400,source_rows=13459,changed_columns=[0,4],source_protocol_sha256=base.sha(a.source_training/'protocol.json'),
        split=old['split'],state_weight=1.,log_loss_clip=1e-12,calibration_bins=10,cache_truth_labels_only=True,
        labels_and_weights_unchanged=True,new_physical_components_used_for_policy=False,public_inverse_precision_failure_retained=True,
        independent_confirmation=False,goal_complete=False)
    base.write(a.results_root/'protocol.json',p);np.savez_compressed(a.results_root/'training_data.npz',**data)
    base.write(a.results_root/'layout.json',base.read(a.source_training/'layout.json'))
    for key in data:
        if key!='x' and not np.array_equal(data[key],arc[key]):raise ValueError('Target/weight/split/control changed')
    base.write(a.results_root/'prepared.json',dict(status='PASS',rows=len(data['y']),source_states=400,eligible_states=400,feature_rows_checked=checked,
        changed_columns=[0,4],data_sha256=base.sha(a.results_root/'training_data.npz'),layout_sha256=base.sha(a.results_root/'layout.json'),
        source_preparation_audit_sha256=base.sha(a.source_training/'preparation_audit.json'),labels_weights_controls_splits_unchanged=True))
    print(json.dumps(dict(status='PREPARED',rows=len(data['y']),feature_rows_checked=checked,mode=MODE)),flush=True)


def analyze(a,p):
    import numpy as np
    from scipy.stats import t
    summary,states=base.analyze(a,p,write_result=False)
    data=np.load(a.results_root/'training_data.npz');old_p=base.read(a.source_training/'protocol.json');new=np.full(len(data['y']),np.nan);old=np.full(len(new),np.nan);seen=np.zeros(len(new),int)
    for fold in base.splits(a,p):
        name=f"block_{fold['block']}_workload_{fold['workload']}";left=np.load(a.results_root/'folds'/name/'test_predictions.npz');right=np.load(a.source_training/'folds'/name/'test_predictions.npz')
        if not np.array_equal(left['row_indices'],right['row_indices']):raise ValueError('Paired prediction rows changed')
        ii=left['row_indices'];new[ii]=left['probability'];old[ii]=right['probability'];seen[ii]+=1
    if not np.all(seen==1) or p['identities']!=old_p['identities']:raise ValueError('Encoding prediction population differs')
    comparisons={};blocks={}
    for block in range(620001,620006):
        ss=[i for i,identity in enumerate(p['identities']) if identity[1]==block];mask=np.isin(data['state_index'],ss);yy=data['y'][mask];ww=data['weight'][mask]
        blocks[str(block)]=dict(old_clipped=float(np.average((old[mask]-yy)**2,weights=ww)),new_full_context=float(np.average((new[mask]-yy)**2,weights=ww)))
    values=[x['old_clipped']-x['new_full_context'] for x in blocks.values()];mean=statistics.mean(values);radius=float(t.ppf(.975,4))*statistics.stdev(values)/5**.5
    comparison=dict(mean_improvement=mean,ci95=[mean-radius,mean+radius],values=values,positive=sum(v>0 for v in values),n=5)
    old_summary=base.read(a.source_training/'analysis/summary.json')
    for method in base.METHODS[1:]:
        for metric in ('brier','log_loss'):
            if not math.isclose(summary['groups'][method][metric],old_summary['groups'][method][metric],abs_tol=1e-12):raise ValueError('Label-free comparator changed')
    vs_controls=summary['decision']['cache_hit_target_promising_development_only'];go=vs_controls and mean>0 and comparison['positive']>=4
    summary.update(status='CACHE_ENCODING_DEVELOPMENT_SCREEN',feature_mode=MODE,changed_columns=[0,4],paired_clipped_blocks=blocks,
        encoding_vs_clipped=comparison,source_summary_sha256=base.sha(a.source_training/'analysis/summary.json'))
    summary['decision'].update(cache_hit_target_promising_development_only=go,encoding_promising_development_only=go,
        beats_label_free_prediction_screen=vs_controls,beats_clipped_prediction_screen=mean>0 and comparison['positive']>=4,
        matching_improvement_tested=False,closed_loop_improvement_tested=False,goal_complete=False)
    base.write(a.results_root/'analysis/summary.json',summary);base.write(a.results_root/'analysis/state_metrics.json',states)
    print(json.dumps(dict(status='ANALYZED',groups=summary['groups'],encoding_vs_clipped=comparison,decision=summary['decision'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','fit','train','analysis'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--block',type=int);parser.add_argument('--workload',type=int);a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Data construction, fitting and statistics run remotely')
    resource=base.res(a)
    if a.mode=='train':base.pipeline(a,base.read(a.results_root/'protocol.json'))
    else:
        with resource.CpuLease(a.resources,task=['cache_encoding',a.mode,a.block,a.workload],kind='environment'):
            if a.mode=='prepare':prepare(a)
            else:
                p=base.read(a.results_root/'protocol.json');base.verify(a,p)
                if a.mode=='fit':base.fit(a,p)
                else:analyze(a,p)


if __name__=='__main__':main()
