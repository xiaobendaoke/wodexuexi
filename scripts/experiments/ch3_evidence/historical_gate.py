"""Run only on the experimental Linux server; validate original raw confirmation."""
import argparse
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--results',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    sys.path.insert(0,a.source)
    import numpy as np
    from analyze_canonical_suite import read_unit
    from utils.canonical_experiment import source_manifest
    directory=Path(a.results);suite=json.loads((directory/'suite.json').read_text())
    expected=json.loads(Path(a.reference).read_text());actual=source_manifest()['source_hash']
    if actual != expected['source_hash'] or suite['source_hash'] != actual:
        raise ValueError('Original source lock does not match archived confirmation')
    plan=suite['plan']; data={};traces={}
    for method in plan['variants'][0]['methods']:
        rows=[]
        for seed in plan['training_seeds']:
            for workload in plan['workload_seeds']:
                values, trace=read_unit(directory/'main'/method/f'seed_{seed}'/f'workload_{workload}',6,actual)
                if workload in traces and traces[workload] != trace:raise ValueError('Unpaired exogenous trace')
                traces[workload]=trace;rows.append(values)
        data[method]=rows
    baseline=data['lower_no_attention'];candidate=data['lower_mlp_upper_uncovered_obs']
    metrics={}
    for key in ('DSR','Admission fraction','Deadline satisfaction among admitted'):
        b=np.array([r[key] for r in baseline]);c=np.array([r[key] for r in candidate])
        for label,value in (('baseline_mean',b.mean()),('candidate_mean',c.mean()),('mean_difference',(c-b).mean())):
            target=expected['comparisons'][key][label]
            if abs(value-target)>1e-6:raise AssertionError((key,label,value,target))
        metrics[key]={'baseline_mean':float(b.mean()),'candidate_mean':float(c.mean()),'difference':float((c-b).mean())}
    ab=np.array([r['Admission fraction'] for r in baseline]);ac=np.array([r['Admission fraction'] for r in candidate])
    eb=np.array([r['Deadline satisfaction among admitted'] for r in baseline]);ec=np.array([r['Deadline satisfaction among admitted'] for r in candidate])
    parts={'admission_component':float(((ac-ab)*(ec+eb)/2).mean()),'conditional_success_component':float(((ec-eb)*(ac+ab)/2).mean())}
    for k,v in parts.items():
        if abs(v-expected['dsr_identity_decomposition'][k])>1e-6:raise AssertionError((k,v))
    report={'status':'PASS','source_hash':actual,'original_results':a.results,'metrics':metrics,'decomposition':parts,'tolerance':1e-6,'evaluation_units':100,'episodes':600}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
