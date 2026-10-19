"""Load the new smoke basis with the actual archived consumer contract."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);o=p.parse_args();a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()});a.smoke=True;sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_framework_confirmation as c
protocol=c.read(a.results_root/'protocol.json');c.verify(a,protocol,True);old=c.original_args(a,protocol,c.SMOKE_BLOCK);f=c.followup(a)
with c.resources(a).CpuLease(old.resources,task=['framework_fresh_basis_preflight'],kind='environment'):
 import numpy as np
 r=f.runner(old);original,rollout=f.initialize(old,r);models,source,checkpoint,state=f.basis(old,r,original,protocol);lock=c.read(checkpoint/'contract.json')
 assert source['config_hash']==lock['config_hash'] and source['source_hash']==lock['source_hash']==protocol['source_hash']
 env,obs=r.episode_env(c.SMOKE_LOAD,0);before=r.base.policy_fingerprint([models['upper']]);left=models['upper'].select_actions(obs,exploration=False);right=models['upper'].select_actions(obs,exploration=False)
 assert np.asarray(left).shape==(5,2) and np.isfinite(left).all() and np.array_equal(left,right) and before==r.base.policy_fingerprint([models['upper']])
 report=dict(status='PASS',smoke=True,new_block=c.SMOKE_BLOCK,checkpoint=str(checkpoint),checkpoint_contract_sha256=c.sha(checkpoint/'contract.json'),protocol_sha256=c.sha(a.results_root/'protocol.json'),source_hash=source['source_hash'],config_hash=source['config_hash'],upper_observation_shape=list(np.asarray(obs).shape),upper_action_shape=list(np.asarray(left).shape),deterministic_action_and_unchanged_fingerprint=True,actual_archived_basis_consumer_loaded=True,formal_nine_method_adapter_smoke_completed=False,formal_confirmation_completed=False)
 c.write(a.validation_root/'basis_preflight.json',report);print(json.dumps(report),flush=True)
