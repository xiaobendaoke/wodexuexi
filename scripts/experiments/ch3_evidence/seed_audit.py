"""Fail closed if any proposed seed occurs as a historical seed or path token."""
import argparse
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--history',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.protocol.read_text())
seeds=set(plan['training_seeds']+plan['workload_seeds']+plan['boundary_workload_seeds']+plan['smoke_training_seeds']+[plan['training_workload_seed'],plan['smoke_workload_seed'],plan['bootstrap_seed']])
pattern=re.compile(r'(?<!\d)('+ '|'.join(map(str,sorted(seeds))) +r')(?!\d)')
conflicts=[];scanned=0;errors=[]
for root in a.history:
    if not root.exists():errors.append(str(root));continue
    for path in root.rglob('*'):
        if 'ch3-evidence-20261103' in str(path) or not path.is_file():continue
        if pattern.search(str(path)):conflicts.append(str(path))
        if path.suffix.lower() not in ('.json','.md','.csv','.yaml','.toml') or path.stat().st_size>2000000:continue
        try:
            if pattern.search(path.read_text(errors='replace')):conflicts.append(str(path))
            scanned+=1
            if scanned % 1000 == 0: print(f'scanned={scanned} root={root}', flush=True)
        except OSError as exc:errors.append(str(exc))
report=dict(status='PASS' if not conflicts and not errors else 'FAIL',seeds=sorted(seeds),files_scanned=scanned,history_roots=list(map(str,a.history)),conflicts=sorted(set(conflicts)),errors=errors)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if report['status']!='PASS':raise SystemExit(1)
