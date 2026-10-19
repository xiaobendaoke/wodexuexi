"""Reconstruct paid pool, request features/labels and every held-out inference.

Remote self-audit. Supplemental probability diagnostics and energy reporting do
not change the frozen selections or failed promotion gates.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from scipy import stats

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def paired(x):
    x=np.asarray(x,dtype=float);mean=float(x.mean());d=float(stats.t.ppf(.975,len(x)-1)*stats.sem(x))
    return dict(mean=mean,ci95=[mean-d,mean+d],values=x.tolist(),n=len(x),positive=int((x>0).sum()),exact_signflip_p=sum(abs(np.mean(x*np.asarray(s)))>=abs(mean)-1e-15 for s in itertools.product((-1,1),repeat=len(x)))/2**len(x))
def close(x,y):
    if isinstance(x,dict):
        assert set(x)==set(y)
        for k in x:close(x[k],y[k])
    elif isinstance(x,list):
        assert len(x)==len(y)
        for a,b in zip(x,y):close(a,b)
    elif isinstance(x,(int,float)):assert np.isclose(x,y,rtol=1e-10,atol=1e-12),(x,y)
    else:assert x==y,(x,y)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

ap=argparse.ArgumentParser()
for k in ('release','results','validation'):ap.add_argument('--'+k,type=Path,required=True)
a=ap.parse_args();r=a.results;p=read(r/'protocol.json');layout=read(r/'layout.json');data=np.load(r/'training_data.npz');v=a.validation
sys.path.insert(0,str(a.release));m=load('audited_request_features',a.release/'utils/sf_hlm_request_risk.py');rank=load('audited_proposal_inputs',a.release/'utils/sf_hlm_action_ranker.py')
for file,digest in p['files'].items():assert sha(a.release/file)==digest
prep=read(r/'prepared.json')
for file,key in [('training_data.npz','data_sha256'),('layout.json','layout_sha256'),('inputs.json','input_manifest_sha256')]:assert sha(r/file)==prep[key]
for file,digest in read(r/'inputs.json').items():assert sha(file)==digest
teacher=Path(p['teacher_results']);costs=p['cost_ms'];controls={};all_choices={};feature_rows=0
tokens=rank.actions()
for si,state in enumerate(layout):
    td=teacher/'tasks'/('oracle_'+'_'.join(map(str,state['identity'])));pub=read(td/'public.json');label=read(td/'labels.json')
    safe=rank.public_payload(pub);ids,fx=rank.features(safe,10.);index={action:i for i,action in enumerate(ids)}
    positive=sorted(k for k in ids if k>0);ordered=sorted(positive,key=lambda k:(-(fx[index[k],118]-fx[index[k],119]),-fx[index[k],120],tokens[k][2],k))
    proposals=[];pairs=set()
    for k in ordered:
        if tokens[k][:2] in pairs:continue
        pairs.add(tokens[k][:2]);proposals.append(k)
        if len(proposals)==4:break
    rest=[k for k in positive if k not in proposals];proposals+=random.Random(20261017).sample(rest,min(4,len(rest)))
    assert proposals==state['proposals'] and state['matching_calls']==1+len(proposals)<=9
    table={c['action']:c for c in pub['candidates']};pool=[];seen=set()
    for action in [-1,0,*sorted(proposals)]:
        c=table[action];key=tuple(c['assignment'])
        if key in seen:continue
        seen.add(key);pool.append(c)
    lex=min(pool,key=lambda c:(tuple(c['predicted']['objective'][:2]),c['action']))['action']
    for cost in costs:
        saved=state['costs'][str(cost)];assert [c['action'] for c in pool]==[c['action'] for c in saved['candidates']]
        actual={};predicted={}
        for c,meta in zip(pool,saved['candidates']):
            sl=slice(meta['start'],meta['stop']);x,p0=m.candidate_features(safe['rows'],c['assignment'],cost,safe['energy_cap']);assert np.array_equal(x,data['x'][sl])
            assert np.array_equal(p0,data['y_predictor'][sl]);actual_requests=label['outcomes'][label['action_outcome'][str(c['action'])]]['requests']
            ya=np.asarray([q['latency']+cost/1000<=q['deadline'] for q in actual_requests],dtype=float);assert np.array_equal(ya,data['y_actual'][sl])
            assert np.all(data['state_index'][sl]==si)
            if len(x):assert np.allclose(data['weight'][sl],1/(len(pool)*len(x)*len(costs)),rtol=0,atol=1e-16)
            actual[c['action']]=int(ya.sum());predicted[c['action']]=int(p0.sum());assert meta['success']==int(ya.sum()) and meta['predicted_success']==int(p0.sum())
            close(meta['metrics'],label['outcomes'][label['action_outcome'][str(c['action'])]]['metrics']);feature_rows+=len(x)
        fee=max(sorted(predicted),key=lambda k:predicted[k]);oracle=max(sorted(actual),key=lambda k:actual[k])
        ctrl=dict(G=actual[-1],predicted_fee=actual[fee],predicted_lex=actual[lex],uniform_pool=sum(actual.values())/len(actual),oracle=actual[oracle])
        close(ctrl,saved['controls']);controls[si,cost]=ctrl
    if state['admitted']:assert np.isclose(data['weight'][data['state_index']==si].sum(),1.)
    if si%40==0:write(v/'audit_progress.json',dict(stage='raw_features',states=si+1,total=400))

predictions={};briers={name:np.zeros(len(data['x'])) for name in (*p['models'],'predictor')};all_proba={name:np.zeros(len(data['x'])) for name in p['models']};models_checked=0;refitted=0
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
for block in sorted({s['identity'][1] for s in layout}):
    for workload in sorted({s['identity'][2] for s in layout}):
        folder=r/'folds'/f'block_{block}_workload_{workload}';rec=read(folder/'result.json');fold=rec['fold']
        train=[i for i,s in enumerate(layout) if s['identity'][1]!=block and s['identity'][2]!=workload];test=[i for i,s in enumerate(layout) if s['identity'][1]==block and s['identity'][2]==workload]
        assert fold['train']==train and fold['test']==test and len(train)==256 and len(test)==16
        for file,digest in rec['files'].items():assert sha(folder/file)==digest
        indices=np.flatnonzero(np.isin(data['state_index'],test));saved=np.load(folder/'test_probabilities.npz');assert np.array_equal(indices,saved['row_indices']);mask=np.isin(data['state_index'],train)
        rows=read(folder/'predictions.json');assert set(map(int,rows))==set(test)
        index=np.full(len(data['x']),-1,dtype=int);index[indices]=np.arange(len(indices))
        for name in p['models']:
            bundle=pickle.loads((folder/(name+'.pkl')).read_bytes());assert bundle['train_states']==train and bundle['test_states']==test and bundle['protocol_sha256']==sha(r/'protocol.json')
            assert bundle['schema']==p['schema'] and bundle['features']==p['features'];model=bundle['model'];xx=data['x'][indices]
            raw=model.predict_proba(xx)[:,list(model.classes_).index(1.)] if name=='hgb_probability' else model.predict(xx)
            if (block,workload)==(620001,780001):
                if name=='hgb_probability':again=HistGradientBoostingClassifier(**p['hgb']).fit(data['x'][mask],data['y_actual'][mask],sample_weight=data['weight'][mask]);rr=again.predict_proba(xx)[:,list(again.classes_).index(1.)]
                else:again=HistGradientBoostingRegressor(**p['hgb']).fit(data['x'][mask],data['y_actual'][mask]-data['y_predictor'][mask],sample_weight=data['weight'][mask]);rr=again.predict(xx)
                assert np.array_equal(raw,rr);refitted+=1
            proba=raw if name=='hgb_probability' else np.clip(raw+data['y_predictor'][indices],0,1)
            assert np.array_equal(proba,saved[name]);all_proba[name][indices]=proba
            for si in test:
                for cost in costs:
                    info=layout[si]['costs'][str(cost)];scores=[float(proba[index[c['start']:c['stop']]].sum()) for c in info['candidates']]
                    actions=[c['action'] for c in info['candidates']];choice=max(sorted(zip(actions,scores)),key=lambda q:q[1])[0]
                    got=rows[str(si)][str(cost)]['models'][name];assert got['action']==choice;assert np.array_equal(got['scores'],scores)
                    assert got['success']==next(c for c in info['candidates'] if c['action']==choice)['success']
            models_checked+=1
        for key,row in rows.items():assert key not in predictions;predictions[key]=row
        write(v/'audit_progress.json',dict(stage='models',models=models_checked,total=50))
close(predictions,read(r/'analysis/predictions.json'))

def reconstruct(model_names):
    names=[*controls[0,10.],*model_names];out={};energy={};calibration={}
    for source in ('all','greedy','learned'):
        out[source]={};energy[source]={};calibration[source]={}
        indices=[i for i,s in enumerate(layout) if source=='all' or s['identity'][0]==source]
        for cost in costs:
            blocks={};extras={};diagnostics={}
            for block in sorted({layout[i]['identity'][1] for i in indices}):
                ids=[i for i in indices if layout[i]['identity'][1]==block];den=sum(layout[i]['generated'] for i in ids);totals={n:0. for n in names};energies={n:[] for n in names if n!='uniform_pool'};mbs={n:0. for n in energies};admitted=sum(layout[i]['admitted'] for i in ids)
                err={n:[] for n in (*model_names,'predictor')}
                for si in ids:
                    info=layout[si]['costs'][str(cost)];cand={c['action']:c for c in info['candidates']}
                    for n in names:
                        if n in model_names:
                            item=predictions[str(si)][str(cost)]['models'][n];totals[n]+=item['success'];action=item['action']
                        else:totals[n]+=controls[si,cost][n];action=info['control_actions'].get(n)
                        if n!='uniform_pool':
                            cc=cand[action];energies[n].append(cc['metrics']['uav_fleet_total_energy']);mbs[n]+=cc['metrics']['service_offloads_mbs']
                    for n in err:
                        e=[]
                        for c in info['candidates']:
                            sl=slice(c['start'],c['stop']);pr=data['y_predictor'][sl] if n=='predictor' else all_proba[n][sl]
                            if len(pr):e.append(float(np.mean((pr-data['y_actual'][sl])**2)))
                        if e:err[n].append(float(np.mean(e)))
                blocks[str(block)]=dict(generated=den,rates={n:v/den for n,v in totals.items()})
                extras[str(block)]=dict(mean_fleet_energy={n:float(np.mean(z)) for n,z in energies.items()},energy_ratio_to_G={n:float(np.mean(z)/np.mean(energies['G'])) for n,z in energies.items()},mbs_admitted_fraction={n:z/max(admitted,1) for n,z in mbs.items()})
                diagnostics[str(block)]={n:float(np.mean(z)) for n,z in err.items() if z}
            comps={n+'-G':paired([b['rates'][n]-b['rates']['G'] for b in blocks.values()]) for n in names if n!='G'}
            for name in names:
                if name in model_names or name=='oracle':
                    for ref in ('predicted_fee','predicted_lex','uniform_pool'):comps[name+'-'+ref]=paired([b['rates'][name]-b['rates'][ref] for b in blocks.values()])
            out[source][str(cost)]=dict(blocks=blocks,groups={n:paired([b['rates'][n] for b in blocks.values()]) for n in names},comparisons=comps)
            energy[source][str(cost)]=extras;calibration[source][str(cost)]=diagnostics
    return out,energy,calibration
pool,_,_=reconstruct([]);close(pool,read(r/'analysis/pool_summary.json')['subsets'])
full,energy,calibration=reconstruct(p['models']);summary=read(r/'analysis/summary.json');close(full,summary['subsets'])
main=full['all']['10.0'];best=max(('G','predicted_fee','predicted_lex','uniform_pool'),key=lambda n:main['groups'][n]['mean'])
for name in p['models']:
    gain=main['comparisons'][name+'-G'];inc=main['comparisons'][name+'-'+best] if best!='G' else gain
    assert summary['decision']['candidates'][name]['optimistic_signal']==(gain['mean']>=.005 and gain['positive']>=4 and inc['mean']>=.001)
write(r/'analysis/audit_supplement.json',dict(energy_and_mbs=energy,request_brier_by_block=calibration,
    diagnostic_only=True,decision_unchanged=True,goal_complete=False,statistical_unit='five historical blocks, not per-request records'))
out=dict(status='PASS',states=400,feature_rows=feature_rows,models_checked=models_checked,models_refitted=refitted,summary_sha256=sha(r/'analysis/summary.json'),
    supplement_sha256=sha(r/'analysis/audit_supplement.json'),audit_type='primary-agent self-audit; alternate pool/scoring reconstruction; first fold refitted',goal_complete=False)
write(v/'raw_record_audit.json',out);print(json.dumps(out))
