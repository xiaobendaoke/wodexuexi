"""Remote scientific aggregation; refuses partial matrices and changed artifacts."""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
from utils.canonical_experiment import write_json


def verify_job(path, source_hash, episodes, steps):
    path=Path(path);done=json.loads((path/'completed.json').read_text())
    if done['status']!='PASS' or done['source_hash']!=source_hash:raise ValueError(f'Incomplete/mixed job: {path}')
    for name,expected in done['hashes'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Changed artifact: {path/name}')
    raw=[json.loads(p.read_text()) for p in sorted((path/'episodes').glob('*.json'))]
    if len(raw)!=episodes or any(len(ep['steps'])!=steps for ep in raw):raise ValueError(f'Partial episodes: {path}')
    if [ep['episode'] for ep in raw]!=list(range(episodes)):raise ValueError('Episode identity mismatch')
    return raw


def unit(path, source_hash, episodes, steps):
    raw=verify_job(path,source_hash,episodes,steps)
    metrics=[row['metrics'] for ep in raw for row in ep['steps']]
    g=sum(m['generated_service_count'] for m in metrics);a=sum(m['admitted_service_count'] for m in metrics)
    s=sum(m['deadline_satisfied_count'] for m in metrics)
    if not 0<=s<=a<=g or g==0 or a==0:raise ValueError('Invalid service count bounds')
    return dict(DSR=s/g,alpha=a/g,eta=s/a,generated=g,admitted=a,successful=s,
                energy=sum(m['uav_fleet_total_energy'] for m in metrics)/episodes,
                fairness=float(np.mean([m['fairness'] for m in metrics]))),[ep['exogenous_trace_hash'] for ep in raw]


def effect(values, plan):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<2 or not np.isfinite(x).all():raise ValueError('Invalid paired matrix')
    means=x.mean(axis=1);se=means.std(ddof=1)/np.sqrt(len(means));mean=float(means.mean())
    half=float(t.ppf(.975,len(means)-1)*se)
    rng=np.random.default_rng(plan['bootstrap_seed']);n=plan['bootstrap_draws']
    ti=rng.integers(0,x.shape[0],size=(n,x.shape[0]));wi=rng.integers(0,x.shape[1],size=(n,x.shape[1]))
    boot=x[ti[:,:,None],wi[:,None,:]].mean(axis=(1,2));ci=np.quantile(boot,[.025,.975]).tolist()
    return dict(mean_difference=mean,t_ci95=[mean-half,mean+half],crossed_bootstrap_ci95=ci,
                positive_training_seeds=int((means>0).sum()),seed_mean_differences=means.tolist(),
                criterion_met=bool(mean>0 and mean-half>0 and ci[0]>0))


def analyze(root, plan, source_hash, capacity=False):
    root=Path(root);cells=dict(plan['cells'])
    if capacity:cells['E']=plan['capacity_method']
    data={};traces={};rows=[]
    for cell,method in cells.items():
        blocks=[]
        for seed in plan['training_seeds']:
            block=[]
            for workload in plan['workload_seeds']:
                path=root/'main'/method/f'seed_{seed}'/f'workload_{workload}'
                row,trace=unit(path,source_hash,plan['eval_episodes'],plan['steps'])
                if workload in traces and traces[workload]!=trace:raise ValueError('Unpaired evaluation exogenous streams')
                traces[workload]=trace;block.append(row);rows.append(dict(cell=cell,method=method,training_seed=seed,workload_seed=workload,**row))
            blocks.append(block)
        data[cell]=blocks
    def matrix(cell,key):return np.array([[r[key] for r in b] for b in data[cell]])
    comparisons={}
    for name,c,b in [('primary','B','D'),('covered_architecture','A','C'),('attention_information','B','A'),('mlp_information','D','C')]+([('capacity','B','E')] if capacity else []):
        comparisons[name]={key:effect(matrix(c,key)-matrix(b,key),plan) for key in ('DSR','alpha','eta')}
        ac,ab,ec,eb=[matrix(cell,key) for cell,key in [(c,'alpha'),(b,'alpha'),(c,'eta'),(b,'eta')]]
        parts=[float(((ac-ab)*(ec+eb)/2).mean()),float(((ec-eb)*(ac+ab)/2).mean())]
        if abs(sum(parts)-comparisons[name]['DSR']['mean_difference'])>1e-12:raise AssertionError('DSR decomposition failure')
        comparisons[name]['decomposition']=dict(admission_component=parts[0],conditional_component=parts[1],interpretation='algebraic, not causal')
    comparisons['interaction']=effect((matrix('B','DSR')-matrix('D','DSR'))-(matrix('A','DSR')-matrix('C','DSR')),plan)
    primary=comparisons['primary']['DSR']['criterion_met']
    report=dict(status='PASS',source_hash=source_hash,protocol=plan['protocol'],comparisons=comparisons,
        primary_criterion_met=primary,capacity_criterion_met=(primary and comparisons['capacity']['DSR']['criterion_met']) if capacity else None,
        statistical_unit=plan['statistical_unit'],means={c:{k:float(matrix(c,k).mean()) for k in ('DSR','alpha','eta','energy','fairness')} for c in data},
        pooled={c:sum(r['successful'] for b in blocks for r in b)/sum(r['generated'] for b in blocks for r in b) for c,blocks in data.items()},
        secondary_inference='descriptive, no selective significance claims',units=len(rows))
    write_json(root/('capacity_analysis.json' if capacity else 'analysis.json'),report)
    with (root/'evaluation_units.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def diagnostic_pair(left,right):
    """Bounded-memory merge per episode; common request identity includes signature."""
    groups=defaultdict(lambda:defaultdict(float))
    for lfile,rfile in zip(sorted((Path(left)/'requests').glob('episode_*.gz')),sorted((Path(right)/'requests').glob('episode_*.gz')),strict=True):
        def read(path):
            with gzip.open(path,'rt') as f:
                records=[json.loads(line) for line in f]
            result={(r['workload_seed'],r['episode'],r['slot'],r['ue']):r for r in records}
            if len(result)!=len(records):raise ValueError('Duplicate request identifiers')
            return result
        l,r=read(lfile),read(rfile)
        for key in l.keys()|r.keys():
            a,b=l.get(key),r.get(key)
            if a is None or b is None:group='generated_only_left' if b is None else 'generated_only_right'
            else:
                if any(a[k]!=b[k] for k in ('request_id','size','deadline','priority','cycles')):raise ValueError('Common request signature mismatch')
                group=('both_admitted' if a['admitted'] and b['admitted'] else 'left_only_admitted' if a['admitted'] else 'right_only_admitted' if b['admitted'] else 'both_unadmitted')
            groups[group]['count']+=1
            for side,row in (('left',a),('right',b)):
                if row is None:continue
                groups[group][side+'_generated']+=1;groups[group][side+'_admitted']+=int(row['admitted']);groups[group][side+'_successful']+=int(row['success'])
                for name in ('size','cycles','deadline','cache_hit','associated_users','final_node_tasks','upload_latency','forward_latency','fetch_latency','compute_latency','latency'):
                    if row.get(name) is not None:
                        groups[group][side+'_'+name+'_sum']+=float(row[name]);groups[group][side+'_'+name+'_n']+=1
    return {k:dict(v) for k,v in groups.items()}
