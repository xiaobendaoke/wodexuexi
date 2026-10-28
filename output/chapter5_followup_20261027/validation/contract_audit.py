"""Additional recorded-candidate/guard audit; execute on server after timing."""
import argparse
import gzip
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('results',type=Path)
a=parser.parse_args()
if sys.platform!='linux':raise RuntimeError('Recorded scientific checks execute on server')
records=[json.loads(p.read_text()) for p in (a.results/'fixed_states').glob('*/public_decisions.json')]
assert len(records)==1228
same=guard=parity=0
for record in records:
    decisions=record['decisions']
    same+=decisions['B0']['assignment']==decisions['Blambda']['assignment']
    for method in ['B0','M0','Blambda','Mlambda']:
        decision=decisions[method];diagnostic=decision['diagnostic']
        assert diagnostic['matching_calls']<=10
        assert decision['candidates'] and decision['candidates'][0]['action']==-1
        for candidate in decision['candidates']:
            if candidate['action']!=-1:
                assert candidate['predicted']['energy']<=diagnostic['common_energy_cap']+1e-7
                guard+=1
    assert [z['quota'] for z in record['attempts']['B0']]==[z['quota'] for z in record['attempts']['M0']]
    parity+=1
assert same==1228
raw_guards=raw_steps=0
for path in (a.results/'tasks').glob('*/*/steps.json.gz'):
    with gzip.open(path,'rt') as stream:steps=json.load(stream)
    for step in steps:
        raw_steps+=1
        diagnostic=step['info']['diagnostic']
        if path.parent.name!='G':
            assert diagnostic['matching_budget']==10
            assert diagnostic['matching_calls']<=10
            assert diagnostic['chosen_predicted']['energy']<=diagnostic['common_energy_cap']+1e-7
            raw_guards+=1
assert raw_steps==345000 and raw_guards==270000
result=dict(status='PASS',states=len(records),binary_price_assignment_unchanged=same,
    common_proposal_quota_sequences=parity,accepted_candidate_guard_checks=guard,
    all_recorded_steps=raw_steps,closed_loop_non_G_guard_checks=raw_guards)
(a.results/'analysis/candidate_contract_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
