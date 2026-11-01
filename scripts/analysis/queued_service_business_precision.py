"""Recheck every observed business-workload reversal at fixed initial state."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from environment.queued_service import Engine,Protocol
from utils.queued_service_business import BusinessScenario
from scripts.analysis.queued_service_coupling import cooperative_plan,motion_sequence
from scripts.analysis.queued_service_business import START,read,locations,path_for,atomic,contract,sha


def recheck(args):
    probe,dt,task=args;s=probe['spec'];out,_=locations(task);parent=read(path_for(out,s))
    tape=BusinessScenario(s['workload'],s['profile'],s['sla'],s['scale']);e=Engine(Protocol(),START,tape.ue_positions)
    for t in range(probe['time']):
        e.add(tape.slot(t));e.admit();r=parent['rows'][t]
        e.assign({int(k):v for k,v in r['assignment'].items()});got=e.advance(r['motion'])
        assert all(got[k]==r[k] for k in ('generated','completed','expired','admitted','active'))
    e.add(tape.slot(probe['time']));e.admit();obs=e.observe()
    local={x['uid']:x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    coop=cooperative_plan(obs);sequence=motion_sequence(e,parent['rows'][probe['time']]['motion'])
    baseline=e.counts['success'];initial_n=len(e.tasks);e.p=replace(e.p,dt=dt);zero=[(0.,0.)]*5;cases={}
    for moving,routing in itertools.product((False,True),('source','cooperative')):
        f=copy.deepcopy(e);offset=0;generated=initial_n
        while offset<6 or f.tasks:
            assert offset<=12
            if offset:
                arrivals=tape.slot(probe['time']+offset) if offset<6 else ();generated+=len(arrivals)
                f.add(arrivals);new=f.admit();route={u:f.tasks[u].source for u in new}
            else:route=local if routing=='source' else coop
            f.assign(route);f.advance(sequence[offset] if moving and offset<6 else zero);offset+=1
        f.audit();cases[f"{'move' if moving else 'hold'}_{routing}"]=dict(success=f.counts['success']-baseline,generated=generated)
    result=dict(spec=s,time=probe['time'],dt=dt,cases=cases,original=probe['cases'],identical=cases==probe['cases'],
                source_contract=contract()[0],script_sha256=sha(Path(__file__)))
    name=f"{s['profile']}_w{s['workload']}_t{probe['time']}_dt{dt}.json"
    atomic(out/'reversal_precision'/name,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True);a=p.parse_args();task=a.task.resolve()
    out,val=locations(task);data=read(out/'analysis.json');reversals=[x for x in data['holdout']['interventions'] if x['eligible'] and x['reversal']]
    jobs=[(x,dt,task) for x in reversals for dt in (.1,.05,.025)]
    atomic(val/'reversal_precision/manifest.json',dict(source_contract=contract()[0],script_sha256=sha(Path(__file__)),
        selection='All observed holdout reversal states, numerical verification only',cases=[dict(spec=x['spec'],time=x['time']) for x in reversals],steps=[.1,.05,.025]))
    if jobs:
        import os
        with ProcessPoolExecutor(max_workers=min(len(jobs),len(os.sched_getaffinity(0)))) as pool:results=list(pool.map(recheck,jobs))
    else:results=[]
    result=dict(status='PASS_BUSINESS_REVERSAL_PRECISION' if all(x['identical'] for x in results) else 'NUMERICAL_SENSITIVITY_FOUND',
        reversal_states=len(reversals),runs=len(results),all_counts_identical=all(x['identical'] for x in results),
        source_contract=contract()[0],script_sha256=sha(Path(__file__)),results=results)
    atomic(out/'reversal_precision/analysis.json',result);atomic(val/'reversal_precision/completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
