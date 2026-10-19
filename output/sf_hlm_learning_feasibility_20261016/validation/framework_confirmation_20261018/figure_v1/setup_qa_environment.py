from pathlib import Path
import os,json,subprocess,time,sys
v=Path(__file__).parent;d=json.loads((v/'release_manifest.json').read_text());qa=Path(d['qa_environment']);py='/home/PengYanghan/miniconda3/envs/drone/bin/python';env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/PengYanghan/miniconda3/envs/drone/lib',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PIP_REQUIRE_VIRTUALENV='true',PIP_CONFIG_FILE='/dev/null',PIP_INDEX_URL='https://pypi.org/simple',PIP_EXTRA_INDEX_URL='',PIP_DISABLE_PIP_VERSION_CHECK='1')
with (v/'qa_environment_setup.log').open('ab') as log:
 subprocess.run([py,'-m','venv','--system-site-packages',str(qa)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
 subprocess.run([str(qa/'bin/python'),'-m','pip','install','--only-binary=:all:','--timeout','20','--retries','1','PyMuPDF>=1.24.9'],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
 probe=subprocess.run([str(qa/'bin/python'),'-c',"import importlib.metadata,json,sys;print(json.dumps(dict(prefix=sys.prefix,matplotlib=importlib.metadata.version('matplotlib'),pymupdf=importlib.metadata.version('PyMuPDF'))))"],check=True,env=env,capture_output=True,text=True)
(v/'qa_environment.json').write_text(json.dumps(dict(status='PASS',environment=json.loads(probe.stdout),experimental_conda_environment_modified=False,setup_after_all_timed_workers_ended=True,completed_unix=time.time()),indent=2)+'\n')
