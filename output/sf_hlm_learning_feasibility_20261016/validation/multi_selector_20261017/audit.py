"""Second implementation: verify public features, refit folds, raw-score OOF.

Run only on the remote experiment server. This is a main-agent audit, not an
independent agent or an independent confirmation experiment.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11):raise AssertionError((a,b))


def main(a):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from scipy.stats import t
    r=a.results_root;p=read(r/'protocol.json');states=read(r/'states.json');folds=read(r/'folds.json')
    summary=read(r/'summary.json');inputs=read(r/'inputs.json');methods=p['methods'];hvals=p['overhead_ms']
    assert len(states)==len(inputs)==400 and len(folds)==25
    assert summary['protocol_sha256']==sha(r/'protocol.json')
    assert sha(Path(p['parent_results'])/'protocol.json')==p['parent_protocol_sha256']
    ids=[s['identity'] for s in states]
    assert ids==p['identities'] and len(set(map(tuple,ids)))==400
    scores=[];raw=[];energy=[];independent_x=[];policies=[]
    branch_instances=timing_samples=0
    for s,receipt in zip(states,inputs):
        assert s['identity']==receipt['identity']
        for key in ('benchmark','state','input'):assert sha(receipt[key])==receipt[key+'_sha256']
        meta=read(receipt['state']);obs=read(receipt['input'])['rows'];bench=read(receipt['benchmark'])
        assert meta['files'][meta['input']]==receipt['input_sha256']
        assert len(obs)==s['admitted'] and bench['generated']==s['generated']
        # Independent vector construction: numpy quantiles, explicit whitelist.
        series=[[] for _ in range(10)];src=np.zeros(5);dst=np.zeros(6);legals=[0,0]
        for row in obs:
            src[int(row['source'])]+=1;d=row['deadline'];opts=row['options']
            lat={str(k):o['base']+o['work'] for k,o in opts.items()};best=min(lat.values())
            cache=[];caps=[]
            for o in opts.values():
                dst[5 if o['target']=='mbs' else int(o['target'])]+=1
                cache.append(o['cache_belief']);caps.append(math.log1p(o['capacity']))
            vals=[d,math.log1p(row['size']),math.log1p(row['cycles']),len(opts),min(10.,best/d),
                  max(-10.,min(10.,d-best)),min(10.,lat['0']/d) if '0' in lat else 10.,
                  min(10.,lat['1']/d) if '1' in lat else 10.,sum(cache)/len(cache),sum(caps)/len(caps)]
            for series_,v in zip(series,vals):series_.append(v)
            legals[0]+=('0' in lat);legals[1]+=('1' in lat)
        if obs:
            v=[float(len(obs))]
            for arr in series:v += [min(arr),float(np.quantile(arr,.25)),sum(arr)/len(arr),float(np.quantile(arr,.75)),max(arr)]
            src/=len(obs);dst/=dst.sum()
            v += [float(src.max()),float((src**2).sum()),float((src>0).mean()),float(dst.max()),float((dst**2).sum())]
            v += [sum(x>=q for x in series[4])/len(obs) for q in (.75,.9,1.)]
            v += [sum(x<=q for x in series[5])/len(obs) for q in (.02,.05)]
            v += [n/len(obs) for n in legals]
        else:v=[0.]*63
        assert len(v)==len(s['features'])==63
        for x,y in zip(v,s['features']):close(x,y)
        independent_x.append(v);fs={};rs={};es={}
        for m in methods:
            trials=[z for z in bench['results'] if z['method']==m]
            assert len(trials)==(5 if m.startswith('R') else 1)
            branch_instances+=len(trials);timing_samples+=sum(len(z['software_seconds']) for z in trials)
            fs[m]={}
            for h in hvals:
                fs[m][str(h)]=sum(sum(sum(z['latency']+cost+h/1000<=z['deadline'] for z in trial['requests'])
                        for cost in trial['software_seconds'])/len(trial['software_seconds']) for trial in trials)/len(trials)
                close(fs[m][str(h)],s['methods'][m]['fee'][str(h)])
            rs[m]=sum(z['raw_success'] for z in trials)/len(trials)
            es[m]=sum(z['metrics']['uav_fleet_total_energy'] for z in trials)/len(trials)
        scores.append(fs);raw.append(rs);energy.append(es)
        policies.append({m:{m:1.} for m in methods})
    # Use the recorded feature bytes after independently validating their values;
    # floating-point summation differences must not change a tree split.
    x=np.asarray([s['features'] for s in states]);seen=[];model_count=0
    for f in folds:
        train=[i for i,z in enumerate(ids) if z[1]!=f['block'] and z[2]!=f['workload']]
        test=[i for i,z in enumerate(ids) if z[1]==f['block'] and z[2]==f['workload']]
        assert f['train']==train and f['test']==test and len(train)==256 and len(test)==16
        seen+=test
        for name,record in f['models'].items():
            arms=(methods if name.endswith('_all') else methods[:5])[1:]
            y=np.asarray([[scores[i][m]['0.0']-scores[i]['G']['0.0'] for m in arms] for i in range(400)])
            reg=(make_pipeline(StandardScaler(),Ridge(alpha=10.)) if name.startswith('ridge') else
                 DecisionTreeRegressor(max_depth=2,min_samples_leaf=16,random_state=20261017))
            reg.fit(x[train],y[train]);pred=reg.predict(x);chosen=[];model_count+=1
            for i,values in enumerate(pred):
                benefits={'G':0.,**dict(zip(arms,values))}
                chosen.append('G' if not states[i]['admitted'] else max(methods,key=lambda m:benefits.get(m,0.)))
            count=Counter(chosen[i] for i in train);probs={m:count[m]/256 for m in methods}
            for m in methods:close(probs[m],record['train_proportions'][m])
            for i in test:
                assert chosen[i]==f['predictions'][name][str(i)]['chosen']
                for j,m in enumerate(arms):close(pred[i,j],f['predictions'][name][str(i)]['predicted'][m])
                policies[i][name]={chosen[i]:1.};policies[i][name+'_random_mix']=probs
                if name.endswith('_all'):
                    policies[i][name+'_replace_L_with_R']={{'L8':'R8','L16':'R16'}.get(chosen[i],chosen[i]):1.}
        blocks=sorted({ids[i][1] for i in train})
        values={m:sum(sum(scores[i][m]['0.0'] for i in train if ids[i][1]==b)/sum(states[i]['generated'] for i in train if ids[i][1]==b) for b in blocks)/4 for m in methods}
        best=max(methods,key=lambda m:values[m]);assert best==f['best_constant']
        for i in test:
            policies[i]['fold_best_constant']={best:1.}
            policies[i]['rule_R8_G']={'R8' if states[i]['admitted']>=16 else 'G':1.}
    assert sorted(seen)==list(range(400)) and model_count==100
    for a_,b_ in zip(policies,states):
        assert a_.keys()==b_['policies'].keys()
        for key,prob in a_.items():
            for m in methods:close(prob.get(m,0.),b_['policies'][key].get(m,0.))
    checked=0
    for subset,ss in summary['subsets'].items():
        subset_ids=[i for i,z in enumerate(ids) if subset=='all' or z[0]==subset]
        block_values={}
        for b,record in ss['blocks'].items():
            indices=[i for i in subset_ids if ids[i][1]==int(b)];den=sum(states[i]['generated'] for i in indices)
            assert den==record['generated']
            block_values[b]={}
            for name,hs in record['policies'].items():
                block_values[b][name]={}
                for h,reported in hs.items():
                    if name.startswith('oracle'):
                        arms=methods if name=='oracle_all' else methods[:5]
                        total=sum(max(scores[i][m][h] for m in arms) for i in indices)
                    else:
                        fee='0.0' if name in methods else h
                        total=sum(sum(prob*scores[i][m][fee] for m,prob in policies[i][name].items()) for i in indices)
                        er=sum(sum(prob*energy[i][m] for m,prob in policies[i][name].items()) for i in indices)
                        close(er,reported['energy'])
                        rawrate=sum(sum(prob*raw[i][m] for m,prob in policies[i][name].items()) for i in indices)/den
                        close(rawrate,reported['raw_rate'])
                        regret=sum(max(scores[i][m][h] for m in methods)-sum(prob*scores[i][m][fee] for m,prob in policies[i][name].items()) for i in indices)
                        close(regret,reported['oracle_loss'])
                    rate=total/den;close(rate,reported['fee_rate']);block_values[b][name][h]=rate;checked+=1
        for name,hs in ss['groups'].items():
            for h,stat in hs.items():
                vals=[v[name][h] for v in block_values.values()];close(stat['mean'],sum(vals)/5)
        for pair,hs in ss['comparisons'].items():
            first,second=pair.split('-')
            for h,stat in hs.items():
                vals=[v[first][h]-v[second]['0.0' if second=='G' else h] for v in block_values.values()]
                mean=sum(vals)/5;radius=float(t.ppf(.975,4))*statistics.stdev(vals)/math.sqrt(5)
                for v,w in zip(vals,stat['values']):close(v,w)
                close(mean,stat['mean']);close(mean-radius,stat['ci95'][0]);close(mean+radius,stat['ci95'][1])
                exact=sum(abs(sum(s*v for s,v in zip(signs,vals))/5)>=abs(mean)-1e-14 for signs in itertools.product([-1,1],repeat=5))/32
                close(exact,stat['exact_signflip_p'])
    verdict=dict(status='PASS',states=400,input_hashes=1200,features_per_state=63,models_refit=model_count,
        folds=25,branch_instances=branch_instances,timing_samples=timing_samples,block_scores_checked=checked,
        checks=['observed-only feature reconstruction','raw per-request fee scores','two-axis split','refit predictions',
                'train-only random proportions','all policies','energy/raw/regret','block rates and comparisons/CI/signflip'],
        independent_agent_review=False,independent_confirmation=False,summary_sha256=sha(r/'summary.json'))
    a.output.write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results-root',type=Path,required=True)
    ap.add_argument('--resource-module',type=Path,required=True);ap.add_argument('--resources',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Remote Linux only')
    spec=importlib.util.spec_from_file_location('resources',a.resource_module);resources=importlib.util.module_from_spec(spec);spec.loader.exec_module(resources)
    with resources.CpuLease(a.resources,task='sf_hlm_selector_raw_audit'):main(a)
