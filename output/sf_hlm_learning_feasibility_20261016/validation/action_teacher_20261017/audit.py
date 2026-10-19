"""Remote audit: recompute every public branch with the unaccelerated engine."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x):return json.loads(json.dumps(x,default=lambda v:v.tolist(),allow_nan=False))
def close(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-11),(x,y)


def audit(a):
    import numpy as np
    from scipy.stats import t
    from utils.sf_hlm_v2 import CompleteQuotaSearch
    from utils.sf_hlm_search import common_repair
    from utils.deadline_repair_online import greedy_assignment
    from utils.sf_hlm_matching import allocation_counts
    p=read(a.results_root/'protocol.json');summary=read(a.results_root/'analysis/summary.json')
    assert not p['smoke'] and len(p['identities'])==400 and summary['protocol_sha256']==sha(a.results_root/'protocol.json')
    launch=read(a.output.parent/'oracle_launch.json')['command'];release=Path(launch[launch.index('--release-root')+1])
    for rel,h in p['files'].items():assert sha(release/rel)==h,rel
    for path,h in p['input_files'].items():assert sha(path)==h,path
    states=[];branches=unique=solves=public_replays=0
    for state in p['states']:
        identity=state['identity'];folder=a.results_root/'tasks'/('oracle_'+'_'.join(map(str,identity)))
        receipt=read(folder/'result.json');public=read(folder/'public.json');labels=read(folder/'labels.json')
        assert receipt['status']=='PASS' and receipt['protocol_sha256']==sha(a.results_root/'protocol.json')
        assert receipt['identity']==public['identity']==labels['identity']==identity
        for name,h in receipt['files'].items():assert sha(folder/name)==h
        assert labels['public_sha256']==sha(folder/'public.json')
        assert sha(state['state'])==state['state_sha256']==labels['state_sha256']
        assert sha(state['benchmark'])==state['benchmark_sha256']
        meta=read(state['state']);inp=read(meta['input']);old=read(state['benchmark']);g_ref=next(z for z in old['results'] if z['method']=='G')
        assert public['rows']==inp['rows'] and public['request_ids']==inp['request_ids']
        assert labels['snapshot_hash']==inp['snapshot_hash']
        assert labels['generated']==old['generated']==meta['generated'] and labels['admitted']==old['admitted']==meta['admitted']
        rows=copy.deepcopy(public['rows'])
        for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
        greedy,_=greedy_assignment(rows);g,gd=common_repair(rows,greedy,128)
        assert list(g)==public['greedy_assignment']==g_ref['assignment']
        base=CompleteQuotaSearch(rows,num_uavs=5,matching_budget=16,base_repair_budget=128)
        actions=[int(x) for x in np.flatnonzero(base.action_mask()[1:])+1]
        assert actions==public['legal_nonstop_actions'];assert base.current.tolist()==public['initial_counts']
        assert list(base.assignment)==public['initial_assignment'] and norm(base.best)==public['initial_predicted']
        candidates={c['action']:c for c in public['candidates']};assert set(candidates)=={-1,0,*actions}
        assert len(candidates)==receipt['branches'];assert public['offline_matching_calls']==base.calls+len(actions)
        for action,record in candidates.items():
            if action==-1:assignment=list(g);predicted=gd['chosen_predicted'];calls=0
            else:
                child=copy.deepcopy(base)
                if not child.done:child.step(action)
                assignment=list(child.assignment);predicted=child.best;calls=child.calls
            assert assignment==record['assignment'] and norm(predicted)==record['predicted'] and calls==record['matching_calls']
            if action>0:assert list(base.actions[action])==record['transfer']
            assert calls<=2 and predicted['energy']<=base.cap+1e-7
            public_replays+=1
            key=hashlib.sha256(json.dumps(assignment).encode()).hexdigest()
            assert labels['action_outcome'][str(action)]==key
            out=labels['outcomes'][key];assert out['assignment']==assignment
            requests=out['requests'];metrics=out['metrics']
            assert len(requests)==len(rows) and [x['ue_id'] for x in requests]==public['request_ids']
            assert [x['action'] for x in requests]==assignment
            for request,row in zip(requests,rows):
                assert request['deadline']==row['deadline'] and math.isfinite(request['latency']) and request['latency']>=0.
                assert request['success']==(request['latency']<=request['deadline'])
            close(sum(x['success'] for x in requests),metrics['deadline_satisfied_count'])
            close(metrics['generated_service_count'],labels['generated']);close(metrics['admitted_service_count'],labels['admitted'])
            if action==-1:
                assert requests==g_ref['requests']
                for k,v in g_ref['metrics'].items():close(v,metrics[k])
        problem=base.problem;load=np.r_[np.maximum(base.current[:-1],1),1]
        risk=(((problem['base']+problem['work']*load[None,:])>problem['deadline'][:,None])&problem['legal']).sum(0)
        h=min(actions,key=lambda k:(-risk[base.actions[k][0]],risk[base.actions[k][1]],base.actions[k][2],k)) if actions else 0
        assert h==public['heuristic_action']
        pred=min(candidates,key=lambda k:(tuple(candidates[k]['predicted']['objective'][:2]),k));assert pred==public['predicted_best_action']
        values={}
        for c in p['cost_hypotheses_ms']:
            counts={a:sum(q['latency']+c/1000<=q['deadline'] for q in labels['outcomes'][labels['action_outcome'][str(a)]]['requests']) for a in candidates}
            best=min(k for k,v in counts.items() if v==max(counts.values()))
            values[str(c)]=dict(G=counts[-1],stop=counts[0],random_one=sum(counts[k] for k in actions)/len(actions) if actions else counts[0],
                heuristic_one=counts[h],predicted_best=counts[pred],oracle=counts[best],oracle_action=best)
        states.append(dict(identity=identity,generated=labels['generated'],values=values))
        branches+=receipt['branches'];unique+=receipt['unique_physical_outcomes'];solves+=receipt['offline_matching_calls']
    stats=0
    for subset,record in summary['subsets'].items():
        sr=[s for s in states if subset=='all' or s['identity'][0]==subset];blocks={}
        for b,reported in record['blocks'].items():
            br=[s for s in sr if s['identity'][1]==int(b)];den=sum(s['generated'] for s in br);assert den==reported['generated']
            blocks[b]={str(c):{m:sum(s['values'][str(c)][m] for s in br)/den for m in p['controls']} for c in p['cost_hypotheses_ms']}
            for c,ms in blocks[b].items():
                for m,v in ms.items():close(v,reported['rates'][c][m])
        for c,comparisons in record['comparisons'].items():
            for pair,reported in comparisons.items():
                first,second=pair.split('-');values=[b[c][first]-b[c][second] for b in blocks.values()]
                mean=sum(values)/5;radius=float(t.ppf(.975,4))*statistics.stdev(values)/math.sqrt(5)
                for v,w in zip(values,reported['values']):close(v,w)
                close(mean,reported['mean']);close(mean-radius,reported['ci95'][0]);close(mean+radius,reported['ci95'][1])
                exact=sum(abs(sum(s*v for s,v in zip(signs,values))/5)>=abs(mean)-1e-14 for signs in itertools.product([-1,1],repeat=5))/32
                close(exact,reported['exact_signflip_p']);assert reported['positive']==sum(v>0 for v in values);stats+=1
    for c,flag in summary['decision']['capacity_by_cost'].items():
        d=summary['subsets']['all']['comparisons'][c]['oracle-G'];assert flag==(d['mean']>=.005 and d['positive']>=4)
    assert branches==summary['branches'] and unique==summary['unique_physical_outcomes'] and solves==summary['offline_matching_calls']
    verdict=dict(status='PASS',states=400,branches=branches,public_branches_replayed_with_original_engine=public_replays,
        unique_physical_outcomes=unique,offline_matching_calls=solves,statistical_comparisons_checked=stats,
        public_inputs_match_original_observation=True,original_G_physical_metrics_reproduced=True,
        actual_policy_costs_not_measured=True,independent_agent_review=False,independent_confirmation=False,
        summary_sha256=sha(a.results_root/'analysis/summary.json'))
    a.output.write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resource-module',type=Path,required=True);ap.add_argument('--resources',type=Path,required=True);a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Remote numerical audit only')
    import importlib.util
    spec=importlib.util.spec_from_file_location('resources',a.resource_module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with m.CpuLease(a.resources,task='action_teacher_audit'):audit(a)
