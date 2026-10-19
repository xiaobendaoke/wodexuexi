from pathlib import Path
import json,subprocess,os,time,sys,importlib.util,argparse
v=Path(__file__).parent;d=json.loads((v/'release_manifest.json').read_text());root=Path(d['release']);runner=Path(d['runner_release']);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH='/home/PengYanghan/miniconda3/envs/drone/lib',PYTHONPATH=str(root)+':'+str(runner)+':'+str(Path(d['base'])/'extension_release_v2_20261001'));sys.path[:0]=env['PYTHONPATH'].split(':')
sp=importlib.util.spec_from_file_location('evidence_controller_runner',runner/'scripts/analysis/sf_hlm_framework_evaluation.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);a=argparse.Namespace(**{k:Path(x) for k,x in json.loads(Path(d['run_manifest']).read_text())['args'].items()});f,fa,fp=m.foundation(a);old=f.parents(fa)
with f.resources(fa).CpuLease(old.resources,task=['framework_evidence_boundary_tests'],kind='environment'):
 with (v/'selftests.log').open('ab') as log:code=subprocess.run(['/home/PengYanghan/miniconda3/envs/drone/bin/python','-m','unittest','discover','-s',str(root/'tests'),'-p','test_sf_hlm_framework_evidence.py','-v'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
(v/'selftests_exit.json').write_text(json.dumps(dict(returncode=code,files=d['files'],ended_unix=time.time()),indent=2)+'\n')
if code:raise SystemExit(code)
cmd=['/home/PengYanghan/miniconda3/envs/drone/bin/python',str(root/'scripts/analysis/sf_hlm_framework_evidence.py'),'--release-root',str(root),'--run-manifest',d['run_manifest'],'--output',d['smoke_output'],'--smoke']
with (v/'smoke.log').open('ab') as log:code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
(v/'smoke_exit.json').write_text(json.dumps(dict(returncode=code,ended_unix=time.time(),formal_confirmation_completed=False),indent=2)+'\n');raise SystemExit(code)
