"""Launch the already validated fixed experiment chain, without Git operations."""
import json
import os
from pathlib import Path
import subprocess

base=Path('/home/PengYanghan/Lunwen/ch3-evidence-20261103')
release=base/'release_v2';validation=base/'validation_v2';root=base/'results_v2'
lock=json.loads((validation/'release.json').read_text())
if lock['status']!='PASS':raise RuntimeError('Preflight is not complete')
if (validation/'launch.json').exists():raise RuntimeError('Existing launch must be inspected, never duplicate')
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/PengYanghan/miniconda3/envs/drone/lib',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
cmd=['/home/PengYanghan/miniconda3/envs/drone/bin/python','ch3_controller.py','--mode','run','--protocol',str(base/'incoming/ch3_evidence_20261103_v2.json'),'--root',str(root),'--validation',str(validation)]
with (base/'controller_v2.log').open('xb') as log:
    process=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,source_hash=lock['source_hash'],command=cmd,cwd=str(release),log=str(base/'controller_v2.log'),status='LAUNCHED_NOT_COMPLETED')
(validation/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
