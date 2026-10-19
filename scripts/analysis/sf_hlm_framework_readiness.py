"""Read-only promotion readiness for a previously positive analytical control."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

PRIMARY='mixture_price'
FILES=('paths.py','scripts/analysis/sf_hlm_framework_readiness.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);o=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('All statistics run on the experiment server')
    a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()})
    spec=importlib.util.spec_from_file_location('framework_readiness_closed',a.release_root/'scripts/analysis/sf_hlm_priced_closed_loop.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    with b.resources(a).CpuLease(a.resources,task=['framework_readiness'],kind='environment'):
        from scipy.stats import t
        if (o.out/'protocol.json').exists():raise FileExistsError('Immutable readiness diagnostic namespace')
        root=a.results_root;sp=b.read(root/'protocol.json');done=b.read(root/'completion.json');s=b.read(root/'analysis/summary.json');audit=b.read(a.validation_root/'raw_record_audit.json')
        if done['status']!='COMPLETED' or audit['status']!='PASS' or done['summary_sha256']!=b.sha(root/'analysis/summary.json') or audit['summary_sha256']!=b.sha(root/'analysis/summary.json'):raise ValueError('Closed-loop source not audited')
        if not sp['real_delay_in_state'] or sp['smoke'] or s['independent_confirmation'] or not s['exogenous_streams_paired']:raise ValueError('Wrong closed-loop source contract')
        own=Path(__file__).resolve().parents[2];inputs={str(x):b.sha(x) for x in (o.launch,root/'protocol.json',root/'completion.json',root/'analysis/summary.json',a.validation_root/'raw_record_audit.json',a.fit_root/'protocol.json',a.fit_root/'prices.json',a.fit_root/'completion.json')}
        for name,digest in sp['files'].items():
            if b.sha(a.release_root/name)!=digest:raise ValueError('Closed-loop source changed')
            inputs[str(a.release_root/name)]=digest
        for name,digest in sp['frozen_fit_files'].items():
            if b.sha(a.fit_root/name)!=digest:raise ValueError('Frozen full-fit price/model changed')
            inputs[str(a.fit_root/name)]=digest
        protocol=dict(name='sf_hlm_framework_positive_control_readiness_20261018_v1',files={f:b.sha(own/f) for f in FILES},input_files=inputs,primary=PRIMARY,posthoc_secondary_control_nomination=True,original_learning_failures_preserved=True,new_training_started=False,new_confirmation_started=False,original_ten_confirmation_blocks_used=False,independent_confirmation=False)
        b.write(o.out/'protocol.json',protocol);modes={};checks=0
        def effect(values):
            mean=statistics.mean(values);sd=statistics.stdev(values);radius=float(t.ppf(.975,len(values)-1))*sd/math.sqrt(len(values))
            return dict(mean=mean,ci95=[mean-radius,mean+radius],values=values,n=len(values),positive=sum(x>0 for x in values),historical_block_sd=sd)
        for mode,source in s['modes'].items():
            blocks=source['blocks'];comparisons={}
            for control in sp['methods']:
                if control==PRIMARY:continue
                row={}
                for metric in ('dsr','potential_dsr','raw_dsr','shadow_fee_dsr','mbs_share','mean_fairness','mean_seconds','p95_seconds'):
                    row[metric]=effect([x[PRIMARY][metric]-x[control][metric] for x in blocks.values()])
                row['fleet_energy_ratio']=dict(mean=statistics.mean(x[PRIMARY]['fleet_energy']/x[control]['fleet_energy'] for x in blocks.values()),values=[x[PRIMARY]['fleet_energy']/x[control]['fleet_energy'] for x in blocks.values()]);comparisons[control]=row
                original=source['learning_differences']
                if control in original:
                    observed=row['dsr']['values'];expected=[x-y for x,y in zip(original[control]['values'],original[PRIMARY]['values'])]
                    if not all(math.isclose(x,y,rel_tol=0,abs_tol=1e-12) for x,y in zip(observed,expected)):raise ValueError('Learning contrast identity does not reproduce')
                    checks+=1
            modes[mode]=dict(groups=source['groups'],comparisons=comparisons)
        measured=modes['measured'];gain=measured['comparisons']['G']['dsr'];energy=measured['comparisons']['G']['fleet_energy_ratio']['mean'];p95=all(x[PRIMARY]['p95_seconds']<1. for x in s['modes']['measured']['blocks'].values());ready=gain['mean']>=.005 and gain['positive']>=4 and energy<=1.05 and p95
        halfwidth=float(t.ppf(.975,9))*gain['historical_block_sd']/math.sqrt(10)
        result=dict(status='POSTHOC_ANALYTICAL_FRAMEWORK_READINESS_DIAGNOSTIC',protocol_sha256=b.sha(o.out/'protocol.json'),source_summary_sha256=b.sha(root/'analysis/summary.json'),modes=modes,original_contrast_identities_reproduced=checks,readiness=dict(practical_development_signal_vs_G=ready,measured_energy_ratio_G=energy,all_block_p95_below_second=p95,projected_10_new_block_ci_halfwidth=halfwidth,projection_assumption='unchanged between-block development variance; no assumed gain from additional episodes',independent_confirmation=False,new_training_started=False,unique_learning_increment_rescued=False,goal_complete=False),posthoc_secondary_control_nomination=True,all_original_learning_failures_preserved=True,descriptive_only=True)
        for path,digest in inputs.items():
            if b.sha(path)!=digest:raise ValueError('Readiness input changed')
        b.write(o.out/'analysis.json',result);b.write(o.out/'completion.json',dict(status='COMPLETED_READ_ONLY',analysis_sha256=b.sha(o.out/'analysis.json'),protocol_sha256=b.sha(o.out/'protocol.json')))
        print(json.dumps(dict(readiness=result['readiness'],measured_comparisons={k:v['dsr'] for k,v in measured['comparisons'].items()})),flush=True)


if __name__=='__main__':main()
