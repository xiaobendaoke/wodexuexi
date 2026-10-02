from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, subprocess, time, sys, os
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/PengYanghan/Lunwen/canonical-releases/20260920-ch4-55631fc')
def write(name,data):
    target=ROOT/name
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    temp.replace(target)
def execute(label,arguments):
    command=['bash',str(SOURCE/'run_remote_python.sh'),*arguments]
    write(label+'-state.json',{'status':'RUNNING','command':command,'started':time.time()})
    started=time.monotonic()
    with (ROOT/(label+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
    state={'status':'PASS' if result.returncode==0 else 'FAIL','exit_code':result.returncode,'seconds':time.monotonic()-started,'command':command}
    write(label+'-state.json',state)
    print(json.dumps({'step':label,**{k:v for k,v in state.items() if k!='command'}}),flush=True)
    if result.returncode: raise RuntimeError(label+' failed')
    return state
COMMON=['run_canonical_experiment.py','--training-seed','42','--steps','1000','--rollout','1000','--ppo-epochs','8','--batch-size','256','--threads','4','--purpose','formal','--source-lock','source-lock.json','--validation-report','validation.json']
def model_probe(method):
    target=ROOT/method
    train=[*COMMON,'--method',method,'--workload-seed','2026','--output',str(target)]
    states=[execute(method+'-train',[*train,'--episodes','1'])]
    if method=='hierarchical':
        states.append(execute(method+'-resume',[*train,'--episodes','2','--resume']))
    checkpoint=target/json.loads((target/'latest.json').read_text())['checkpoint']
    states.append(execute(method+'-evaluate',[*COMMON,'--method',method,'--workload-seed','42','--episodes','1','--evaluate-checkpoint',str(checkpoint),'--output',str(ROOT/(method+'-evaluation'))]))
    episodes=[]
    for directory in (target,ROOT/(method+'-evaluation')):
        for path in sorted((directory/'episodes').glob('*.json')):
            r=json.loads(path.read_text())
            episodes.append({'path':str(path),'steps':len(r['steps']),'final_losses':r['final_losses'],'max_fallbacks':max(s['metrics']['service_fallback_count'] for s in r['steps'])})
    manifest=json.loads((target/'manifest.json').read_text())
    return {'status':'PASS','steps':states,'episodes':episodes,'git_sha':manifest['source']['git_sha']}
def chapter4_probe():
    code="from run_chapter4_suite import plan,run; p=plan('main'); p.update(workloads=[42],episodes=1,steps=2,snapshot_steps=[0,1]); r=run(p,"+repr(str(ROOT/'chapter4'))+",source_lock='source-lock.json',validation_report='validation.json'); print(r['status'])"
    state=execute('chapter4',['-c',code])
    records=[]
    for path in sorted((ROOT/'chapter4/main/workload_42').glob('episode_*.json')):
        r=json.loads(path.read_text())
        records.append({'snapshot':path.name,'admitted_requests':len(r['requests']),'mc_itr_termination':r['methods']['mc_itr']['termination'],'mc_itr_objective':r['methods']['mc_itr']['objective']})
    return {'status':'PASS','step':state,'snapshots':records}
write('plan.json',{'purpose':'startup verification only; shortened runs passing the formal source gate, not thesis results','source':str(SOURCE),'hierarchical':'1 x 1000 train, resume to 2 episodes, 1 x 1000 evaluation','joint':'1 x 1000 train and 1 x 1000 evaluation','chapter4':'100 UE, 2 full natural-admission snapshots','ppo_epochs':8,'batch_size':256,'threads_per_job':4})
write('status.json',{'status':'RUNNING','pid':os.getpid(),'started':time.time()})
results={}
with ThreadPoolExecutor(max_workers=3) as pool:
    futures={pool.submit(model_probe,'hierarchical'):'hierarchical',pool.submit(model_probe,'joint'):'joint',pool.submit(chapter4_probe):'chapter4'}
    for future in as_completed(futures):
        name=futures[future]
        try: results[name]=future.result()
        except Exception as error: results[name]={'status':'FAIL','error':repr(error)}
        write('progress.json',results)
write('status.json',{'status':'PASS' if all(r['status']=='PASS' for r in results.values()) else 'FAIL','results':results,'finished':time.time()})
sys.exit(0 if all(r['status']=='PASS' for r in results.values()) else 1)
