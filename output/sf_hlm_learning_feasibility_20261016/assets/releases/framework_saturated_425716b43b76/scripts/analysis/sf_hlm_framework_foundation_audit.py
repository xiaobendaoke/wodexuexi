"""Direct evidence for fresh independent foundation training, remote only."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def collect_modules(model,torch):
    found={};seen=set()
    def visit(obj,path,root=False):
        if id(obj) in seen:return
        seen.add(id(obj))
        if isinstance(obj,torch.nn.Module):found[path]=obj;return
        if isinstance(obj,dict):
            for k,value in obj.items():visit(value,path+'/'+str(k))
        elif isinstance(obj,(tuple,list)):
            for i,value in enumerate(obj):visit(value,path+'/'+str(i))
        elif root or type(obj).__module__.startswith('marl_models'):
            for k,value in vars(obj).items():visit(value,path+'/'+k)
    visit(model,'upper',True)
    if not found:raise ValueError('No upper network modules inspected')
    return found


def run(a):
    import numpy as np
    import torch
    f=load('framework_direct_foundation_contract',a.foundation_release/'scripts/analysis/sf_hlm_framework_confirmation.py');fa=argparse.Namespace(**{k:Path(v) for k,v in f.read(a.foundation_launch)['args'].items()});p=f.read(fa.results_root/'protocol.json');fa.smoke=p['smoke']
    if fa.smoke!=a.smoke:raise ValueError('Smoke does not count as formal fresh training')
    done=f.read(fa.results_root/'completion.json');prior=f.read(fa.results_root/'foundation_audit.json');f.verify(fa,p,True)
    if done['status']!='COMPLETED' or prior['status']!='PASS' or done['foundation_audit_sha256']!=f.sha(fa.results_root/'foundation_audit.json') or done['protocol_sha256']!=f.sha(fa.results_root/'protocol.json'):raise ValueError('Every registered foundation must be complete before direct audit')
    if not a.smoke and p['confirmation_blocks']!=list(f.BLOCKS):raise ValueError('New foundation population differs')
    old=f.original_args(fa,p,p['confirmation_blocks'][0]);r=f.followup(fa);runner=r.runner(old);original,rollout=r.initialize(old,runner);out=[];episodes_checked=0;steps_checked=0;all_fingerprints=[]
    for block in p['confirmation_blocks']:
        args=f.original_args(fa,p,block);path=r.folder(args);receipt=r.verified(path,full=True);canonical=path/'canonical';manifest=f.read(canonical/'manifest.json');settings=manifest['settings'];latest=f.read(canonical/'latest.json');check=canonical/latest['checkpoint']
        expected=dict(method=runner.base.UPPER,training_seed=block,workload_seed=p['training_workloads'][str(block)],steps=p['steps'],rollout=1000,ppo_epochs=8,batch_size=256,threads=1,purpose='verification' if a.smoke else 'pilot',ue_count=100,compute_scale=1.,checkpoint_every=20)
        if settings!=expected:raise ValueError('Actual canonical training seed or numerical recipe differs')
        final=p['foundation_episodes']-1
        if latest['completed_episodes']!=p['foundation_episodes'] or check.name!=f'episode_{final:05d}':raise ValueError('Wrong last checkpoint selected')
        episode_paths=sorted((canonical/'episodes').glob('*.json'))
        if [x.name for x in episode_paths]!=[f'{i:05d}.json' for i in range(p['foundation_episodes'])]:raise ValueError('Training repetitions missing or duplicated')
        for i,episode_path in enumerate(episode_paths):
            ep=f.read(episode_path)
            if ep['episode']!=i or len(ep['steps'])!=p['steps'] or [s['step'] for s in ep['steps']]!=list(range(p['steps'])):raise ValueError('Canonical training episode or step sequence incomplete')
            for step in ep['steps']:
                if step['metrics']['forced_service_admissions']!=0 or not math.isfinite(float(step['team_reward'])) or not math.isfinite(float(step['lower_reward'])):raise ValueError('Natural admission or finite training reward violated')
            episodes_checked+=1;steps_checked+=len(ep['steps'])
        models,source,loaded,state=r.basis(args,runner,original,p)
        if loaded!=check:raise ValueError('Actual basis consumer selected another checkpoint')
        upper=models['upper'];fingerprint=runner.base.policy_fingerprint([upper]);modules=collect_modules(upper,torch);saved={name:{k:value.detach().clone() for k,value in module.state_dict().items() if torch.is_tensor(value)} for name,module in modules.items()};tensor_count=0
        with torch.no_grad():
            for values in saved.values():
                for value in values.values():
                    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():raise ValueError('Nonfinite upper network state')
                    tensor_count+=1
            for module in modules.values():
                for value in module.state_dict().values():
                    if torch.is_tensor(value) and value.numel() and value.is_floating_point():value.add_(1.)
        if fingerprint==runner.base.policy_fingerprint([upper]):raise ValueError('Reload test did not perturb upper network state')
        upper.load(str(check/'upper'));restored=collect_modules(upper,torch)
        if saved.keys()!=restored.keys():raise ValueError('Upper module structure changed on reload')
        for name,values in saved.items():
            actual=restored[name].state_dict()
            if any(not torch.equal(value,actual[k]) for k,value in values.items()):raise ValueError('Upper checkpoint loader did not restore its entire inspected state')
        if fingerprint!=runner.base.policy_fingerprint([upper]):raise ValueError('Upper fingerprint did not restore')
        env,obs=runner.episode_env(args.workload,0);motion=upper.select_actions(obs,exploration=False)
        if np.asarray(motion).shape!=(5,2) or not np.isfinite(motion).all():raise ValueError('Fresh upper inference invalid')
        all_fingerprints.append(fingerprint);out.append(dict(block=block,training_workload=args.workload,canonical_manifest_sha256=f.sha(canonical/'manifest.json'),settings=settings,checkpoint=str(check),checkpoint_contract_sha256=f.sha(check/'contract.json'),upper_fingerprint=fingerprint,upper_modules_checked=len(modules),upper_state_tensors_checked=tensor_count,all_upper_state_restored_after_perturbation=True,source_hash=source['source_hash'],config_hash=source['config_hash'],source_receipt_sha256=f.sha(path/'result.json')))
        f.write(a.out.with_name(a.out.stem+'_progress.json'),dict(status='AUDITING',foundations=len(out),expected=len(p['confirmation_blocks']),training_episodes_checked=episodes_checked,training_steps_checked=steps_checked))
    if len(set(all_fingerprints))!=len(p['confirmation_blocks']):raise ValueError('Fresh foundation upper weights are not distinct')
    report=dict(status='PASS',smoke=a.smoke,foundations=len(out),blocks=out,training_episodes_checked=episodes_checked,training_steps_checked=steps_checked,distinct_upper_weight_fingerprints=len(set(all_fingerprints)),actual_seed_and_recipe_manifests_checked=True,all_steps_natural_admission_and_finite_rewards=True,last_checkpoint_only=True,upper_state_perturb_reload_checked=True,protocol_sha256=f.sha(fa.results_root/'protocol.json'),foundation_completion_sha256=f.sha(fa.results_root/'completion.json'),audit_script_sha256=f.sha(__file__),formal_confirmation_completed=False,audit_type='primary agent self-audit; direct canonical manifests, every training step, source load and in-memory upper state restoration')
    f.write(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='blocks'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--foundation-release',type=Path,required=True);p.add_argument('--foundation-launch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Training and network audits run remotely')
    if a.out.exists():raise FileExistsError('Completed direct foundation audit is immutable')
    f=load('framework_direct_audit_resources',a.foundation_release/'scripts/analysis/sf_hlm_framework_confirmation.py');fa=argparse.Namespace(**{k:Path(v) for k,v in f.read(a.foundation_launch)['args'].items()});old=f.parents(fa)
    with f.resources(fa).CpuLease(old.resources,task=['framework_direct_foundation_audit',a.smoke],kind='environment'):run(a)


if __name__=='__main__':main()
