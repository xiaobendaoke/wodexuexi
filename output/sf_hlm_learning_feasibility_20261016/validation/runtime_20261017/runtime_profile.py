"""Remote diagnostic only: ordinary full-path times plus separate cProfile."""
import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import pstats
import subprocess
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def main(a):
    if sys.platform!='linux':raise RuntimeError('Remote only')
    f=module('profile_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
    parent=f.parent_args(a);res=module('profile_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    if a.results_root.exists():raise FileExistsError('Preserve earlier diagnostics')
    p=read(a.state_root/'protocol.json');identities=[i for i in p['identities'] if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    if a.block is not None:identities=[i for i in identities if i[1]==a.block]
    assert len(identities)==(8 if a.block is None else 4)
    protocol=dict(identities=identities,methods=['G','L16','L8','R16'],plain_repeats=3,profile_repeats=1,
        source=sha(Path(__file__)),parent_protocol_sha256=sha(a.state_root/'protocol.json'),
        profiler_times_are_not_formal_costs=True,no_training=True,no_selection_from_probe=True)
    write(a.results_root/'protocol.json',protocol);records=[]
    if a.block is None:
        for block in (620001,620002):
            command=[sys.executable,str(Path(__file__).resolve()),'--feasibility-release',str(a.feasibility_release),
                '--parent-launch',str(a.parent_launch),'--state-root',str(a.state_root),
                '--results-root',str(a.results_root/f'block_{block}'),'--block',str(block)]
            subprocess.run(command,check=True)
            records += read(a.results_root/f'block_{block}'/'summary.json')['records']
        write(a.results_root/'summary.json',dict(status='PASS_DIAGNOSTIC',records=records,plain_samples=96,profile_calls=32,
            interpretation='Two independent block processes; profiler times not mixed with plain costs'))
        write(a.results_root/'status.json',dict(status='COMPLETED',updated_unix=time.time()))
        print(json.dumps(dict(status='PASS_DIAGNOSTIC',records=len(records))),flush=True);return
    with res.CpuLease(parent.resources/'isolated',task='sf_hlm_runtime_profile'):
        runtime=None;last_block=None;c=f.core(argparse.Namespace(release_root=a.feasibility_release))
        for identity in identities:
            if identity[1]!=last_block:runtime=f.boot(a,identity[1]);last_block=identity[1]
            _,_,runner,models,lower,basis=runtime
            sf=f.state_dir(argparse.Namespace(results_root=a.state_root),identity);meta=read(sf/'state.json')
            for path,digest in meta['files'].items():assert sha(path)==digest
            blob=Path(meta['observed']).read_bytes()
            specs=[s for s in c.branch_specs(identity,5) if s['method'] in protocol['methods'] and s['repeat']==0]
            for spec in specs:
                env,obs=pickle.loads(blob);_,_,ref,_=c.complete_policy(env,obs,spec,models,lower,runner)
                times=[]
                for _ in range(3):
                    env,obs=pickle.loads(blob);_,_,info,_=c.complete_policy(env,obs,spec,models,lower,runner)
                    assert info['assignment']==ref['assignment'];times.append(info['policy_seconds'])
                env,obs=pickle.loads(blob);pr=cProfile.Profile();pr.enable()
                _,_,info,_=c.complete_policy(env,obs,spec,models,lower,runner);pr.disable()
                assert info['assignment']==ref['assignment']
                folder=a.results_root/('profile_'+'_'.join(map(str,identity))+'_'+spec['method']);folder.mkdir()
                pr.dump_stats(str(folder/'calls.prof'))
                stats=pstats.Stats(pr);functions=[]
                for (file,line,name),(primitive,calls,self_s,cumulative,callers) in stats.stats.items():
                    functions.append(dict(file=file,line=line,function=name,primitive_calls=primitive,calls=calls,self_seconds=self_s,cumulative_seconds=cumulative))
                functions.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
                row=dict(identity=identity,method=spec['method'],cpu=sorted(os.sched_getaffinity(0)),plain_seconds=times,
                    matching_calls=info['diagnostic']['matching_calls'],base_evaluations=info['diagnostic']['base_evaluations'],
                    assignment=info['assignment'],functions=functions,basis=basis)
                write(folder/'summary.json',row);records.append(row)
        write(a.results_root/'summary.json',dict(status='PASS_DIAGNOSTIC',records=records,plain_samples=len(records)*3,profile_calls=len(records),interpretation='cProfile separate from ordinary full-policy times'))
    write(a.results_root/'status.json',dict(status='COMPLETED',updated_unix=time.time()))
    print(json.dumps(dict(status='PASS_DIAGNOSTIC',records=len(records))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('feasibility_release','parent_launch','state_root','results_root'):ap.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    ap.add_argument('--block',type=int,choices=(620001,620002))
    main(ap.parse_args())
