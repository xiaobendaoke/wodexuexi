"""Remote fixed-upper lower-training pilot; frozen historical inputs are read-only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

UPPERS=('lower_mlp_upper_uncovered_obs','vanilla_lower_mlp_upper_uncovered_obs')
KINDS=('original','flat','shared','greedy')
EXTENSION_FILES=('utils/deadline_repair.py','utils/deadline_repair_online.py',
                 'utils/node_offload_observation.py','marl_models/node_offload_mappo.py',
                 'scripts/analysis/node_offload_experiment.py','tests/test_node_offload.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_default(value):
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'item'):return value.item()
    raise TypeError('Unsupported JSON value: '+type(value).__name__)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=json_default)+'\n');tmp.replace(path)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def model_hash(model):
    h=hashlib.sha256()
    for name in ('actor','critic'):
        for key,tensor in sorted(getattr(model,name).state_dict().items()):
            h.update((name+'/'+key).encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def job_path(root,phase,upper,kind,seed,workload):
    suffix=f'block_{seed}' if phase=='train' else f'seed_{seed}_workload_{workload}'
    return root/('training' if phase=='train' else 'evaluation')/upper/kind/suffix


def setup(args):
    if platform.system()!='Linux':raise RuntimeError('Simulation/training belongs on remote Linux')
    sys.path.insert(0,str(args.source_root.resolve()))
    import numpy as np
    import torch
    import config
    from utils.canonical_experiment import source_manifest,config_snapshot,digest
    from utils.formal_protocol import verify_completed_job
    from utils.logger import refresh_derived_config_fields
    from run_canonical_experiment import configure_method
    protocol=json.loads((args.output/'protocol.json').read_text())
    key=[args.upper,args.kind,args.seed,args.workload]
    if key not in protocol['train_jobs' if args.phase=='train' else 'eval_jobs']:raise ValueError('Undeclared job')
    for rel,value in protocol['extension_files'].items():
        if sha(args.extension_root/rel)!=value:raise ValueError('Extension changed: '+rel)
    if sha(args.suite_root/'suite.json')!=protocol['suite_sha256']:raise ValueError('Source suite changed')
    if source_manifest()['source_hash']!=protocol['source_hash']:raise ValueError('Frozen source changed')
    suite=json.loads((args.suite_root/'suite.json').read_text());plan=suite['plan']
    v=plan['variants'][0];config.NUM_UES=v['ue_count']
    config.UAV_COMPUTING_CAPACITY=config.UAV_COMPUTING_CAPACITY*v['compute_scale']
    refresh_derived_config_fields()
    config.PPO_EPOCHS=plan['ppo_epochs'];config.PPO_BATCH_SIZE=plan['batch_size'];config.PPO_ROLLOUT_LENGTH=plan['rollout']
    torch.set_num_threads(1);np.random.seed(protocol['lower_seeds'][str(args.seed)])
    torch.manual_seed(protocol['lower_seeds'][str(args.seed)])
    models=configure_method(args.upper)
    base=args.suite_root/'main'/args.upper/f'seed_{args.seed}'
    training=verify_completed_job(base/'train',protocol['source_hash'])
    latest=json.loads((training/'latest.json').read_text())
    if latest['completed_episodes']!=200:raise ValueError('Upper is not final prespecified checkpoint')
    checkpoint=training/latest['checkpoint'];old_contract=json.loads((checkpoint/'contract.json').read_text())
    if old_contract['method']!=args.upper or old_contract['source_hash']!=protocol['source_hash'] or old_contract['config_hash']!=digest(config_snapshot()):
        raise ValueError('Historical checkpoint/config mismatch')
    for rel,value in old_contract['files'].items():
        if sha(checkpoint/rel)!=value:raise ValueError('Historical checkpoint changed')
    # Both historical models load before any new lower initialization.
    for name,model in models.items():model.load(str(checkpoint/name))
    upper=models['upper']
    for name in ('actor','critic'):
        getattr(upper,name).eval()
        for parameter in getattr(upper,name).parameters():parameter.requires_grad_(False)
    frozen_hash=model_hash(upper)
    load_module('utils.deadline_repair',args.extension_root/'utils/deadline_repair.py')
    online=load_module('utils.deadline_repair_online',args.extension_root/'utils/deadline_repair_online.py')
    features=load_module('utils.node_offload_observation',args.extension_root/'utils/node_offload_observation.py')
    node=load_module('marl_models.node_offload_mappo',args.extension_root/'marl_models/node_offload_mappo.py')
    config.PPO_EPOCHS=protocol['ppo_epochs'];config.PPO_BATCH_SIZE=protocol['batch_size'];config.PPO_ROLLOUT_LENGTH=protocol['train_steps']
    lower=models['lower']
    if args.kind in ('flat','shared'):
        # Align lower initialization seeds after historical load; actor architectures differ.
        torch.manual_seed(protocol['lower_seeds'][str(args.seed)])
        lower=node.NodeOffloadMAPPO(args.kind,config.NUM_UAVS,config.OFFLOAD_OBS_DIM_SINGLE,
                                  config.MAX_OFFLOAD_REQUESTS_PER_UAV,'cpu')
    cfg_hash=digest(config_snapshot())
    contract=dict(protocol_sha256=sha(args.output/'protocol.json'),source_hash=protocol['source_hash'],
                  upper=args.upper,upper_seed=args.seed,lower_seed=protocol['lower_seeds'][str(args.seed)],
                  kind=args.kind,upper_contract_sha256=sha(checkpoint/'contract.json'),upper_parameter_sha256=frozen_hash,
                  config_hash=cfg_hash,feature_contract=features.FEATURE_CONTRACT)
    return protocol,models,upper,lower,online,features,contract


def policy(models,upper,lower,online,features,env,obs,kind,exploration):
    import numpy as np
    from run_canonical_experiment import policy_step
    import config
    started=time.perf_counter();network_seconds=0.;feature_seconds=0.;initial_evaluations=0
    if kind=='original':
        t=time.perf_counter();motion,original,entries=policy_step(models,env,obs,exploration=False)
        network_seconds+=time.perf_counter()-t
        lower_obs=entries['lower'][0];masks=entries['lower'][2];logp=values=None;packed=None
    else:
        t=time.perf_counter();lower_obs,masks=env.get_offloading_obs_and_masks()
        motion=upper.select_actions(obs,exploration=False);network_seconds+=time.perf_counter()-t
        original=np.zeros(masks.shape[:2],dtype=np.int64);logp=values=packed=None
    t=time.perf_counter();requests,rows=online.prepare(env,masks);prepare_seconds=time.perf_counter()-t
    t=time.perf_counter()
    if kind in ('flat','shared'):
        packed=features.prepare_observation(env,lower_obs,masks,requests,rows)
        feature_seconds=time.perf_counter()-t
        t=time.perf_counter();original,logp,values=lower.get_action_and_value(packed,masks=masks,exploration=exploration)
        network_seconds+=time.perf_counter()-t;initialization_seconds=0.
    elif kind=='greedy':
        initial,initial_evaluations=online.greedy_assignment(rows)
        for request,action in zip(requests,initial):original[request[0],request[1]]=action
        initialization_seconds=time.perf_counter()-t
    else:initialization_seconds=0.
    t=time.perf_counter();chosen,diag=online.apply(requests,rows,original,'ordinary',budget=128)
    repair_seconds=time.perf_counter()-t
    raw=[int(original[r[0],r[1]]) for r in requests];executed=[int(chosen[r[0],r[1]]) for r in requests]
    if any(not masks[r[0],r[1],a] for r,a in zip(requests,executed)):raise ValueError('Illegal action')
    upload=len(json.dumps({'rows':rows,'proposals':raw},separators=(',',':')).encode())
    download=len(json.dumps({'ue_ids':[r[2] for r in requests],'actions':executed},separators=(',',':')).encode())
    info=dict(request_ue_ids=[r[2] for r in requests],original_proposals=raw,executed_actions=executed,
              repair=diag,network_seconds=network_seconds,prepare_seconds=prepare_seconds,
              feature_seconds=feature_seconds,initialization_seconds=initialization_seconds,
              initialization_evaluations=initial_evaluations,repair_seconds=repair_seconds,
              lower_network_called=kind!='greedy',logical_upload_bytes=upload,logical_download_bytes=download)
    info['policy_seconds']=time.perf_counter()-started
    return motion,chosen,(packed,original,masks,logp,values),info


def save_checkpoint(path,lower,contract,next_episode,rho_dsr,rho_mbs):
    import numpy as np
    import torch
    path.mkdir(parents=True,exist_ok=True);lower.save(str(path))
    rng=np.random.get_state()
    write(path/'state.json',dict(next_episode=next_episode,rho_dsr=rho_dsr,rho_mbs=rho_mbs,
          numpy_rng=[rng[0],rng[1].tolist(),int(rng[2]),int(rng[3]),float(rng[4])]))
    torch.save(torch.get_rng_state(),path/'torch_rng.pt')
    files={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.name!='contract.json'}
    write(path/'contract.json',dict(contract,files=files,next_episode=next_episode))


def verify_checkpoint(path,contract):
    d=json.loads((path/'contract.json').read_text())
    if any(d.get(k)!=v for k,v in contract.items()):raise ValueError('New lower checkpoint incompatible')
    for rel,value in d['files'].items():
        if sha(path/rel)!=value:raise ValueError('New lower checkpoint changed')
    return d


def job(args):
    import numpy as np
    import torch
    protocol,models,upper,lower,online,features,contract=setup(args)
    import config
    from environment.env import Env
    from marl_models.buffer_and_helpers import DiscreteOffloadRolloutBuffer
    from run_hierarchical_mappo_experiment import lower_rewards_from_metrics,update_on_policy_model
    from utils.canonical_experiment import exogenous_frame,digest,json_value
    out=job_path(args.output,args.phase,args.upper,args.kind,args.seed,args.workload)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    out.mkdir(parents=True,exist_ok=True)
    train=args.phase=='train';total_episodes=protocol['train_episodes' if train else 'eval_episodes']
    steps=protocol['train_steps'] if train else protocol['steps']
    start=0;rho_dsr=rho_mbs=0.;checkpoint=None
    if train and args.resume and (out/'latest.json').exists():
        latest=json.loads((out/'latest.json').read_text());checkpoint=out/latest['checkpoint']
        verify_checkpoint(checkpoint,contract);lower.load(str(checkpoint))
        state=json.loads((checkpoint/'state.json').read_text());start=state['next_episode']
        rho_dsr=state['rho_dsr'];rho_mbs=state['rho_mbs'];rng=state['numpy_rng']
        np.random.set_state((rng[0],np.asarray(rng[1],dtype=np.uint32),rng[2],rng[3],rng[4]))
        torch.set_rng_state(torch.load(checkpoint/'torch_rng.pt',weights_only=True))
    elif train and (out/'latest.json').exists():raise ValueError('Partial training exists: use --resume')
    if not train and args.kind in ('flat','shared'):
        train_dir=job_path(args.output,'train',args.upper,args.kind,args.seed,0)
        latest=json.loads((train_dir/'latest.json').read_text());checkpoint=train_dir/latest['checkpoint']
        c=verify_checkpoint(checkpoint,contract)
        if c['next_episode']!=protocol['train_episodes']:raise ValueError('Not final new lower checkpoint')
        lower.load(str(checkpoint))
    reference=None
    if protocol['smoke'] and not train and args.kind in ('original','greedy'):
        mode='ordinary' if args.kind=='original' else 'greedy_ordinary'
        ref=args.control_root/args.upper/mode/f'seed_{args.seed}_workload_{args.workload}'
        d=json.loads((ref/'result.json').read_text())
        if d['status']!='PASS' or d['checkpoint_contract_sha256']!=contract['upper_contract_sha256']:raise ValueError('Regression reference mismatch')
        for rel,value in d['episode_files'].items():
            if sha(ref/rel)!=value:raise ValueError('Regression reference changed')
        reference=json.loads((ref/'episodes/00000.json').read_text())['steps']
    env=Env(scenario_seed=args.workload)
    for _ in range(start):env.reset()
    summaries=[]
    for episode in range(total_episodes):
        if episode<start:
            summaries.append(json.loads((out/'episodes'/f'{episode:05d}.json').read_text())['summary']);continue
        obs=np.asarray(env.reset(),dtype=np.float32);upper.reset();lower.reset()
        buffer=DiscreteOffloadRolloutBuffer(config.NUM_UAVS,lower.obs_dim,config.MAX_OFFLOAD_REQUESTS_PER_UAV,
                 config.OFFLOAD_NUM_ACTIONS,steps,'cpu') if train else None
        trace=hashlib.sha256();records=[];violations=[];totals={k:0. for k in ['generated','admitted','success','potential','total_uav_energy','assignment_energy','mbs_count','fairness','changed_actions','lower_reward']}
        timings=[];action_hash=hashlib.sha256()
        for step in range(steps):
            trace.update(digest(exogenous_frame(env)).encode())
            potential=sum(ue.latent_request[0]<config.NUM_SERVICES for ue in env.ues)
            motion,executed,entry,info=policy(models,upper,lower,online,features,env,obs,args.kind,train)
            next_obs,rewards,metrics=env.step(np.clip(motion,-1,1),offloading_actions=executed)
            if metrics['service_fallback_count'] or metrics['forced_service_admissions']:raise ValueError('Unexpected fallback/forced admission')
            if metrics['admitted_service_count']!=len(info['request_ue_ids']) or metrics['service_offloads_mbs']!=info['executed_actions'].count(1):raise ValueError('Executed action mismatch')
            if not np.isfinite(next_obs).all() or not all(math.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite simulation')
            lower_rewards,diag=lower_rewards_from_metrics(metrics,rewards,rho_dsr,rho_mbs);violations.append(diag)
            done=step+1==steps
            if train:
                # Buffer gets sampled network proposals and their own log probabilities.
                buffer.add(*entry[:4],lower_rewards,done,entry[4])
            if reference is not None:
                old=reference[step]
                if info['original_proposals']!=old['original_proposals'] or info['executed_actions']!=old['executed_actions']:raise ValueError('Historical actions regression')
                for name in ['generated_service_count','admitted_service_count','deadline_satisfied_count','energy','lower_assignment_energy','fairness']:
                    if not np.isclose(metrics[name],old['metrics'][name],rtol=1e-10,atol=1e-10):raise ValueError('Historical metric regression: '+name)
            mapping={'generated':'generated_service_count','admitted':'admitted_service_count','success':'deadline_satisfied_count',
                     'total_uav_energy':'energy','assignment_energy':'lower_assignment_energy','mbs_count':'service_offloads_mbs','fairness':'fairness'}
            for key,name in mapping.items():totals[key]+=float(metrics[name])
            totals['potential']+=int(potential);totals['changed_actions']+=int(info['repair']['changed_actions']);totals['lower_reward']+=float(lower_rewards[0])
            timings.append(info['policy_seconds']);action_hash.update(digest([info['original_proposals'],info['executed_actions']]).encode())
            if not train:records.append(dict(step=step,metrics=json_value(metrics),potential_service_count=int(potential),**info))
            obs=np.asarray(next_obs,dtype=np.float32)
            if step%250==0:print(json.dumps(dict(phase=args.phase,episode=episode,step=step)),flush=True)
        losses={}
        if train:
            losses=update_on_policy_model(lower,buffer,np.zeros(config.NUM_UAVS,np.float32))
            if not all(math.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite PPO update')
            rho_dsr=float(np.clip(rho_dsr+config.OFFLOAD_LAGRANGE_LR*statistics.mean(d['dsr_violation'] for d in violations),0,config.OFFLOAD_LAGRANGE_MAX))
            rho_mbs=float(np.clip(rho_mbs+config.OFFLOAD_LAGRANGE_LR*statistics.mean(d['mbs_load_violation'] for d in violations),0,config.OFFLOAD_LAGRANGE_MAX))
        if model_hash(upper)!=contract['upper_parameter_sha256']:raise ValueError('Frozen upper changed')
        summary=dict(episode=episode,**totals,dsr=totals['success']/max(totals['generated'],1),
            admission_rate=totals['admitted']/max(totals['generated'],1),conditional_success=totals['success']/max(totals['admitted'],1),
            fairness_mean=totals['fairness']/steps,policy_p95_seconds=sorted(timings)[math.ceil(.95*len(timings))-1],
            exogenous_trace_hash=trace.hexdigest(),action_trace_hash=action_hash.hexdigest(),losses=losses,rho_dsr=rho_dsr,rho_mbs=rho_mbs,
            upper_parameter_sha256=contract['upper_parameter_sha256'])
        write(out/'episodes'/f'{episode:05d}.json',dict(summary=summary,steps=records));summaries.append(summary)
        if train and ((episode+1)%20==0 or episode+1==total_episodes or protocol['smoke']):
            checkpoint=out/'checkpoints'/f'episode_{episode:05d}'
            save_checkpoint(checkpoint,lower,contract,episode+1,rho_dsr,rho_mbs)
            write(out/'latest.json',dict(checkpoint=str(checkpoint.relative_to(out)),completed_episodes=episode+1))
        write(out/'progress.json',dict(phase=args.phase,completed_episodes=episode+1,total_episodes=total_episodes,pid=os.getpid(),updated_unix=time.time()))
        print(json.dumps(dict(phase=args.phase,completed_episodes=episode+1,dsr=summary['dsr'],losses=losses)),flush=True)
    result=dict(status='PASS',phase=args.phase,**contract,workload=args.workload,episodes=summaries,
        checkpoint=str(checkpoint) if checkpoint else None,upper_frozen=True,historical_control_verified=reference is not None,
        parameter_counts={name:sum(p.numel() for p in getattr(lower,name).parameters()) for name in ('actor','critic')} if args.kind!='greedy' else {},
        episode_files={str(p.relative_to(out)):sha(p) for p in sorted((out/'episodes').glob('*.json'))})
    write(out/'result.json',result);print(json.dumps(dict(status='PASS',phase=args.phase,upper=args.upper,kind=args.kind,seed=args.seed,workload=args.workload)),flush=True)


def analyse(root,protocol):
    """Prespecified seed-block metrics, not a confirmation or automatic retraining."""
    from scipy import stats
    blocks={};groups={};times={};records=[]
    traces={}
    for upper,kind,seed,workload in protocol['eval_jobs']:
        path=job_path(root,'eval',upper,kind,seed,workload);d=json.loads((path/'result.json').read_text())
        if d['status']!='PASS' or not d['upper_frozen']:raise ValueError('Failed evaluation')
        if d['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Evaluation protocol mismatch')
        records.append(d)
        for e in d['episodes']:
            key=(workload,e['episode']);fingerprint=(e['exogenous_trace_hash'],e['potential'])
            if key in traces and traces[key]!=fingerprint:raise ValueError('Exogenous pairing failed')
            traces[key]=fingerprint
            key=(upper,kind,seed);blocks.setdefault(key,[]).append(e)
            ep_path=path/'episodes'/f"{e['episode']:05d}.json"
            if sha(ep_path)!=d['episode_files'][str(ep_path.relative_to(path))]:raise ValueError('Episode checksum mismatch')
            episode_data=json.loads(ep_path.read_text());steps=episode_data['steps']
            if episode_data['summary']!=e or len(steps)!=protocol['steps']:raise ValueError('Episode record mismatch')
            totals={f:0. for f in ['generated','admitted','success','potential','total_uav_energy','mbs_count']}
            for index,q in enumerate(steps):
                m=q['metrics'];diag=q['repair'];n=len(q['request_ue_ids'])
                if q['step']!=index or m['service_fallback_count'] or m['forced_service_admissions']:raise ValueError('Invalid execution record')
                if n!=m['admitted_service_count'] or n!=len(q['executed_actions']) or n!=len(q['original_proposals']):raise ValueError('Lost admitted request')
                if m['generated_service_count']!=n+m['unadmitted_service_count'] or m['processed_service_count']!=n:raise ValueError('Count identity failed')
                if any(type(a)is not int or not 0<=a<=6 for a in q['executed_actions']):raise ValueError('Saved action out of range')
                if m['service_offloads_mbs']!=q['executed_actions'].count(1):raise ValueError('Saved execution mismatch')
                if q['lower_network_called']!=(kind!='greedy'):raise ValueError('Greedy invoked lower network')
                if diag['evaluations']>128 or diag['chosen_predicted_energy']>1.05*diag['original_predicted_energy']+1e-7:raise ValueError('Repair budget/guard mismatch')
                if diag['changed_actions']!=sum(a!=b for a,b in zip(q['original_proposals'],q['executed_actions'])):raise ValueError('Changed-action count mismatch')
                if not 0<=m['deadline_satisfied_count']<=n<=m['generated_service_count']<=q['potential_service_count']:raise ValueError('Invalid service counts')
                for f,mf in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('total_uav_energy','energy'),('mbs_count','service_offloads_mbs')]:totals[f]+=m[mf]
                totals['potential']+=q['potential_service_count']
            if not all(math.isclose(e[f],v,rel_tol=1e-9,abs_tol=1e-7) for f,v in totals.items()):raise ValueError('Summary count mismatch')
            times.setdefault((upper,kind),[]).extend(q['policy_seconds'] for q in steps)
    units={}
    for key,es in blocks.items():
        totals={f:sum(e[f] for e in es) for f in ['generated','admitted','success','potential','total_uav_energy','mbs_count']}
        units[key]=dict(dsr=totals['success']/totals['generated'],potential_dsr=totals['success']/totals['potential'],
            admission_rate=totals['admitted']/totals['generated'],conditional_success=totals['success']/totals['admitted'],
            energy=totals['total_uav_energy']/len(es),mbs_share=totals['mbs_count']/totals['admitted'],fairness=statistics.mean(e['fairness_mean'] for e in es))
    for upper in UPPERS:
        for kind in KINDS:
            us=[units[upper,kind,s] for s in protocol['upper_seeds']]
            ts=sorted(times[upper,kind]);groups[upper+'/'+kind]={f:statistics.mean(u[f] for u in us) for f in us[0]}
            groups[upper+'/'+kind].update(policy_p95_seconds=ts[math.ceil(.95*len(ts))-1],policy_mean_seconds=statistics.mean(ts))
    comparisons={};gates={}
    if not protocol['smoke']:
        def paired(xs):
            average=statistics.mean(xs)
            if statistics.stdev(xs)==0:
                ci=[average,average];p=1. if average==0 else 0.;t=None
            else:
                res=stats.ttest_1samp(xs,0);ci=list(res.confidence_interval());p=float(res.pvalue);t=float(res.statistic)
            flips=[abs(statistics.mean(s*x for s,x in zip(signs,xs))) for signs in itertools.product((-1,1),repeat=len(xs))]
            return dict(mean=average,ci95=ci,p_two_sided=p,t=t,df=4,
                        exact_sign_flip_p=sum(v>=abs(average)-1e-12 for v in flips)/len(flips),
                        positive_blocks=sum(x>0 for x in xs),block_differences=xs)
        for upper in UPPERS:
            family=[]
            for new,old in [('shared','flat'),('shared','greedy'),('flat','greedy')]:
                key=upper+'/'+new+'-'+old
                comparisons[key]=paired([units[upper,new,s]['dsr']-units[upper,old,s]['dsr'] for s in protocol['upper_seeds']]);family.append(comparisons[key])
            running=0.
            for i,item in enumerate(sorted(family,key=lambda v:v['p_two_sided'])):
                running=max(running,min(1.,(len(family)-i)*item['p_two_sided']));item['holm_p']=running
            for kind in ['flat','shared']:
                diffs={old:[units[upper,kind,s]['dsr']-units[upper,old,s]['dsr'] for s in protocol['upper_seeds']] for old in ['original','greedy']}
                energy=statistics.mean(units[upper,kind,s]['energy']/units[upper,'greedy',s]['energy'] for s in protocol['upper_seeds'])
                gates[upper+'/'+kind]=dict(dsr_mean_differences={k:statistics.mean(v) for k,v in diffs.items()},
                    positive_blocks={k:sum(x>0 for x in v) for k,v in diffs.items()},mean_energy_ratio=energy,
                    pass_screen=all(statistics.mean(v)>=.01 and sum(x>0 for x in v)>=4 for v in diffs.values()) and energy<=1.05 and groups[upper+'/'+kind]['policy_p95_seconds']<1.)
    write(root/'analysis.json',dict(status='PASS',groups=groups,comparisons=comparisons,promotion_gates=gates,
        independent_blocks=len(protocol['upper_seeds']),new_upper_training=False,independent_confirmation=False,
        parameter_counts={d['upper']+'/'+d['kind']:d['parameter_counts'] for d in records},
        raw_episode_hashes_and_count_identities_verified=True,
        limitations=protocol['limitations'],next_stage_started=False))
    write(root/'summary.json',dict(status='PASS',train_jobs=len(protocol['train_jobs']),eval_jobs=len(records),
          episodes=sum(len(d['episodes']) for d in records),exogenous_pairing_verified=True,upper_frozen=True))


def suite(args):
    if platform.system()!='Linux':raise RuntimeError('Run on remote Linux')
    old=json.loads((args.suite_root/'suite.json').read_text())
    seeds=[620001] if args.smoke else list(range(620001,620006))
    workloads=[750001] if args.smoke else list(range(760001,760006))
    lower_seeds={str(s):660001+s-620001 for s in seeds};training_workloads={str(s):860001+s-620001 for s in seeds}
    train_jobs=[[u,k,s,training_workloads[str(s)]] for u in UPPERS for k in ['flat','shared'] for s in seeds]
    eval_jobs=[[u,k,s,w] for w in workloads for u in UPPERS for k in KINDS for s in seeds]
    protocol=dict(version='node_offload_fixed_upper_v1',purpose='development_screening',smoke=args.smoke,
        source_hash=old['source_hash'],suite_sha256=sha(args.suite_root/'suite.json'),
        extension_files={rel:sha(args.extension_root/rel) for rel in EXTENSION_FILES},
        upper_seeds=seeds,lower_seeds=lower_seeds,training_workloads=training_workloads,eval_workloads=workloads,
        train_jobs=train_jobs,eval_jobs=eval_jobs,train_episodes=2 if args.smoke else 200,
        eval_episodes=1 if args.smoke else 2,steps=100 if args.smoke else 1000,train_steps=8 if args.smoke else 1000,
        ppo_epochs=1 if args.smoke else 8,batch_size=2 if args.smoke else 256,
        train_workers=args.train_workers,eval_workers=args.eval_workers,threads=1,repair_budget=128,
        feature_contract='source_local_action_aligned_12_v1',upper_training=False,checkpoint_selection='last prespecified only',
        training_action_contract='sample proposal log-probability; repaired execution reward; fixed repair mapping',
        actor_comparison='same node input; two hidden layers width128; unequal parameter counts reported',
        limitations=['Ideal instantaneous lossless uncharged coordination.','Initializer-relative 1.05 energy guards have different absolute feasible sets.',
            'Frozen upper policies can follow different state-dependent trajectories.','Original lower is historical reference, not a matched retraining information ablation.',
            'Five upper/lower blocks; no independent confirmation.','Concurrent software timing excludes physical signaling.',
            'No automatic next stage, joint retraining, or test-based checkpoint choice.'])
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Protocol already exists with different content')
    else:write(protocol_path,protocol)
    complete={'train':0,'eval':0};failures=[]
    def status(phase):write(args.output/'status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),
        expected_train_jobs=len(train_jobs),expected_eval_jobs=len(eval_jobs),completed_train_jobs=complete['train'],
        completed_eval_jobs=complete['eval'],failures=failures,new_upper_training=False))
    def run(phase,item):
        upper,kind,seed,workload=item;out=job_path(args.output,phase,*item)
        if (out/'result.json').exists():
            d=json.loads((out/'result.json').read_text())
            if d['status']!='PASS' or d['protocol_sha256']!=sha(protocol_path):raise ValueError('Existing result mismatch')
            for rel,h in d['episode_files'].items():
                if sha(out/rel)!=h:raise ValueError('Existing episode changed')
            return out
        out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--job','--phase',phase,'--source-root',str(args.source_root),
            '--suite-root',str(args.suite_root),'--extension-root',str(args.extension_root),'--output',str(args.output),
            '--upper',upper,'--kind',kind,'--seed',str(seed),'--workload',str(workload)]
        if args.control_root:cmd+=['--control-root',str(args.control_root)]
        if args.resume:cmd+=['--resume']
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
        log=out/('resume.log' if args.resume else 'run.log')
        with log.open('a' if args.resume else 'w') as f:subprocess.run(cmd,cwd=args.source_root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        return out
    for phase,jobs,workers in [('train',train_jobs,args.train_workers),('eval',eval_jobs,args.eval_workers)]:
        status('TRAINING' if phase=='train' else 'EVALUATING')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(run,phase,item):item for item in jobs}
            for f in as_completed(futures):
                try:f.result();complete[phase]+=1
                except Exception as e:failures.append(dict(phase=phase,job=futures[f],error=str(e)))
                status('TRAINING' if phase=='train' else 'EVALUATING')
                print(json.dumps(dict(phase=phase,completed=complete[phase],failures=len(failures),total=len(jobs))),flush=True)
        if failures:status('FAILED');raise RuntimeError('Failed tasks retained; dependent phase not started')
    status('ANALYSING')
    try:analyse(args.output,protocol)
    except Exception as e:
        failures.append(dict(phase='analysis',error=str(e)));status('FAILED_ANALYSIS');raise
    status('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','suite-root','extension-root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-root',type=Path);p.add_argument('--job',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--phase',choices=['train','eval'])
    p.add_argument('--upper',choices=UPPERS);p.add_argument('--kind',choices=KINDS)
    p.add_argument('--seed',type=int);p.add_argument('--workload',type=int)
    p.add_argument('--train-workers',type=int,default=20);p.add_argument('--eval-workers',type=int,default=5)
    args=p.parse_args()
    if args.train_workers not in range(1,21) or args.eval_workers not in range(1,6):p.error('Worker limits exceeded')
    if args.job:
        if not all(getattr(args,k) is not None for k in ['phase','upper','kind','seed','workload']):p.error('Missing job identity')
        job(args)
    else:suite(args)


if __name__=='__main__':main()
