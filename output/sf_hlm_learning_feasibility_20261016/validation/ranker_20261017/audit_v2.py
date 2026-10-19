"""Raw-record and serialized-model self-audit; execute on the remote server."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy import stats

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def paired(values):
    x=np.asarray(values);mean=float(x.mean());se=float(stats.sem(x));delta=float(stats.t.ppf(.975,len(x)-1)*se)
    p=sum(abs(float(np.mean(x*np.asarray(s))))>=abs(mean)-1e-15 for s in itertools.product((-1,1),repeat=len(x)))/2**len(x)
    return dict(mean=mean,ci95=[mean-delta,mean+delta],values=list(x),n=len(x),positive=int((x>0).sum()),exact_signflip_p=p)
def close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(int,float)):assert np.isclose(a,b,rtol=1e-10,atol=1e-12),(a,b)
    else:assert a==b,(a,b)

ap=argparse.ArgumentParser()
for k in ('release','results','validation'):ap.add_argument('--'+k,type=Path,required=True)
a=ap.parse_args();r=a.results;p=read(r/'protocol.json');layout=read(r/'layout.json');archive=np.load(r/'training_data.npz');data={k:archive[k] for k in archive.files}
sys.path.insert(0,str(a.release));f=load('audit_ranker_features',a.release/'utils/sf_hlm_action_ranker.py')
for rel,digest in p['files'].items():assert sha(a.release/rel)==digest
prep=read(r/'prepared.json')
for file,key in [('training_data.npz','data_sha256'),('layout.json','layout_sha256'),('inputs.json','input_manifest_sha256')]:assert sha(r/file)==prep[key]
for file,digest in read(r/'inputs.json').items():assert sha(file)==digest
teacher=Path(p['teacher_results']);all_expected=[];controls={};feature_rows=0
for si,state in enumerate(layout):
    td=teacher/'tasks'/('oracle_'+'_'.join(map(str,state['identity'])))
    pub=read(td/'public.json');labels=read(td/'labels.json');safe=f.public_payload(pub)
    for c in p['cost_ms']:
        info=state['costs'][str(c)];ids,x=f.features(safe,c);sl=slice(info['start'],info['stop'])
        assert ids==info['actions'] and np.array_equal(data['x'][sl],x)
        # An arbitrary private/identity payload cannot change the public whitelist.
        dirty=dict(pub,identity=['other'],latency=-1,cache_truth={'hidden':True})
        dirty['candidates']=[{'private':'discarded'}]
        assert np.array_equal(f.features(dirty,c)[1],x)
        values={k:sum(q['latency']+c/1000<=q['deadline'] for q in labels['outcomes'][labels['action_outcome'][str(k)]]['requests']) for k in ids}
        pred={v['action']:sum(t+c/1000<=row['deadline'] for t,row in zip(v['predicted']['latencies'],pub['rows'])) for v in pub['candidates']}
        assert np.array_equal(data['y_actual'][sl],[values[k]-values[-1] for k in ids])
        assert np.array_equal(data['y_predictor'][sl],[pred[k]-pred[-1] for k in ids])
        assert np.all(data['state_index'][sl]==si)
        assert np.allclose(data['weight'][sl],1/(len(ids)*len(p['cost_ms'])),rtol=0,atol=1e-16)
        choices=pub['legal_nonstop_actions'];rule=f.analytic_choice(ids,x,len(pub['rows']))
        ctrl=dict(G=values[-1],stop=values[0],random_one=sum(values[k] for k in choices)/len(choices) if choices else values[0],
            heuristic_one=values[pub['heuristic_action']],rule_cost=values[rule],predicted_best_offline=values[pub['predicted_best_action']],oracle=max(values.values()))
        close(ctrl,info['controls']);controls[si,c]=ctrl;feature_rows+=len(ids)
    assert np.isclose(data['weight'][data['state_index']==si].sum(),1.)

rows={};checked_models=0;refitted=0
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
for block in sorted({s['identity'][1] for s in layout}):
    for workload in sorted({s['identity'][2] for s in layout}):
        folder=r/'folds'/f'block_{block}_workload_{workload}';rec=read(folder/'result.json');fold=rec['fold']
        train=[i for i,s in enumerate(layout) if s['identity'][1]!=block and s['identity'][2]!=workload]
        test=[i for i,s in enumerate(layout) if s['identity'][1]==block and s['identity'][2]==workload]
        assert train==fold['train'] and test==fold['test'] and len(train)==256 and len(test)==16
        for file,digest in rec['files'].items():assert sha(folder/file)==digest
        preds=read(folder/'predictions.json');assert set(map(int,preds))==set(test)
        mask=np.isin(data['state_index'],train)
        for name in p['models']:
            bundle=pickle.loads((folder/(name+'.pkl')).read_bytes());assert bundle['train_states']==train and bundle['test_states']==test
            assert bundle['protocol_sha256']==sha(r/'protocol.json') and bundle['schema']==p['schema']
            model=bundle['model'];scaler=bundle['scaler']
            if (block,workload)==(620001,780001):
                y=data['y_predictor'] if name.endswith('predictor') else data['y_actual']
                if name.startswith('ridge'):
                    sc=StandardScaler().fit(data['x'][mask],sample_weight=data['weight'][mask]);assert np.array_equal(sc.mean_,scaler.mean_)
                    fitted=Ridge(alpha=p['ridge_alpha']).fit(sc.transform(data['x'][mask]),y[mask],sample_weight=data['weight'][mask])
                else:fitted=HistGradientBoostingRegressor(**p['hgb']).fit(data['x'][mask],y[mask],sample_weight=data['weight'][mask])
                refitted+=1
            for si in test:
                for cost in p['cost_ms']:
                    info=layout[si]['costs'][str(cost)];xx=data['x'][info['start']:info['stop']]
                    xx=scaler.transform(xx) if scaler is not None else xx;pp=model.predict(xx)
                    if (block,workload)==(620001,780001):assert np.array_equal(pp,fitted.predict(xx))
                    got=preds[str(si)][str(cost)]['models'][name];assert np.array_equal(pp,got['predicted'])
                    scores={int(k):float(v) for k,v in zip(info['actions'],pp)};scores[-1]=0.
                    choice=max(sorted(scores),key=lambda k:scores[k]);assert choice==got['action']
                    offset=info['start']+info['actions'].index(choice)
                    assert got['success']==controls[si,cost]['G']+data['y_actual'][offset]
            checked_models+=1
        for key,row in preds.items():assert key not in rows;rows[key]=row
assert len(rows)==400;close(rows,read(r/'analysis/predictions.json'))
summary=read(r/'analysis/summary.json');comparisons=0
names=[*layout[0]['costs']['0.0']['controls'],*p['models']]
for source in ('all','greedy','learned'):
    for cost in p['cost_ms']:
        expected={};indices=[i for i,s in enumerate(layout) if source=='all' or s['identity'][0]==source]
        for block in sorted({layout[i]['identity'][1] for i in indices}):
            ids=[i for i in indices if layout[i]['identity'][1]==block];den=sum(layout[i]['generated'] for i in ids)
            totals={n:sum(controls[i,cost][n] if n not in p['models'] else rows[str(i)][str(cost)]['models'][n]['success'] for i in ids) for n in names}
            expected[str(block)]=dict(generated=den,rates={k:v/den for k,v in totals.items()})
        saved=summary['subsets'][source][str(cost)];close(expected,saved['blocks'])
        for name,value in saved['comparisons'].items():
            left,right=name.split('-');close(paired([v['rates'][left]-v['rates'][right] for v in expected.values()]),value);comparisons+=1
        for name,value in saved['groups'].items():close(paired([v['rates'][name] for v in expected.values()]),value)
for name in p['models']:
    g=summary['subsets']['all']['10.0']['comparisons'][name+'-G']
    assert summary['decision']['candidates'][name]['optimistic_signal']==(g['mean']>=.005 and g['positive']>=4)
out=dict(status='PASS',states=400,feature_rows=feature_rows,models_checked=checked_models,models_refitted=refitted,
    comparisons= comparisons,summary_sha256=sha(r/'analysis/summary.json'),audit_type='primary-agent self-audit; alternate raw scoring and split reconstruction; first fold refitted',goal_complete=False)
a.validation.mkdir(parents=True,exist_ok=True);(a.validation/'raw_record_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
