from pathlib import Path
import json,subprocess,os,time
v=Path(__file__).parent;d=json.loads((v/'release_manifest.json').read_text());a=d['args'];root=Path(d['release']);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH='/home/PengYanghan/miniconda3/envs/drone/lib',PYTHONPATH=str(root)+':'+str(Path(d['base'])/'extension_release_v2_20261001'))
for mode in ('selftest','init','run'):
 cmd=['/home/PengYanghan/miniconda3/envs/drone/bin/python',str(root/'scripts/analysis/sf_hlm_framework_evaluation.py'),mode]
 for key,value in a.items():cmd+=['--'+key.replace('_','-'),value]
 cmd+=['--runner-smoke'];start=time.time()
 with (v/(mode+'.log')).open('ab') as log:code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 (v/(mode+'_exit.json')).write_text(json.dumps(dict(mode=mode,returncode=code,started=start,ended=time.time(),formal_evaluation_started=False),indent=2)+'\n')
 if code:raise SystemExit(code)
