"""One authorized pilot launch followed by analysis; never starts confirmation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--state',type=Path,required=True)
args=p.parse_args()
source=args.source.resolve();output=args.output.resolve()
sys.path.insert(0,str(source))
from utils.canonical_experiment import source_manifest
manifest=source_manifest()
def record(phase,**extra):
    value={'phase':phase,'pid':os.getpid(),'source_hash':manifest['source_hash'],
           'output':str(output),'time_unix':time.time(),'confirmation_started':False,**extra}
    temporary=args.state.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(args.state)
if output.exists():raise FileExistsError('Refusing duplicate launch: '+str(output))
record('PILOT_RUNNING')
try:
    subprocess.run([sys.executable,'run_formal_suite.py','--plan-file','docs/plans/batch_offload_pilot_20260928.json',
                    '--output',str(output),'--workers','1','--threads','1'],cwd=source,check=True)
    record('PILOT_ANALYZING')
    subprocess.run([sys.executable,'-m','scripts.analysis.batch_offload_analysis',str(output)],cwd=source,check=True)
    report=json.loads((output/'batch_offload_analysis.json').read_text())
    record('PILOT_COMPLETE',selected_for_confirmation=report['selected_for_confirmation'])
except Exception as error:
    record('FAILED',error=str(error))
    raise
