from pathlib import Path
import json,subprocess,os,time,hashlib
v=Path(__file__).parent;settings=json.loads((v/'controller_inputs.json').read_text());manifest=Path(settings['formal_manifest']);d=json.loads(manifest.read_text());root=Path(d['release']);report=Path(settings['evidence_release']);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH='/home/PengYanghan/miniconda3/envs/drone/lib',PYTHONPATH=str(root)+':'+d['args']['legacy_release']+':'+'/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001')
for mode in ('init','run','evidence'):
 if mode=='evidence':cmd=['/home/PengYanghan/miniconda3/envs/drone/bin/python',str(report/'scripts/analysis/sf_hlm_framework_evidence.py'),'--release-root',str(report),'--run-manifest',str(manifest),'--output',settings['evidence_output']]
 else:
  cmd=['/home/PengYanghan/miniconda3/envs/drone/bin/python',str(root/d['entry_script']),mode]
  for key,value in d['args'].items():cmd+=['--'+key.replace('_','-'),value]
 start=time.time()
 with (v/(mode+'.log')).open('ab') as log:code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 (v/(mode+'_exit.json')).write_text(json.dumps(dict(mode=mode,returncode=code,started_unix=start,ended_unix=time.time(),run_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),indent=2)+'\n')
 if code:raise SystemExit(code)
