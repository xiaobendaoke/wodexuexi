"""Bounded, read-only diagnostic replays of the frozen chapter-five release.

Numerical work is server-only. Diagnostic interventions never replace formal
evaluations, checkpoints, or the frozen protocol. No optimizer is constructed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = 'chapter5_dynamic_service_20261028'
MODES = ('full_replay', 'matching_replay', 'stochastic_upper_matching',
         'reset_memory_matching', 'zero_feedback_matching', 'history_learned_lower')


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_imports(args):
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(args.extension))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from utils.dynamic_service_parallel import install_environment_cache
    install_environment_cache()
    return torch


def model(args, block):
    import torch
    from scripts.analysis.dynamic_service_experiment import models
    path = args.task/'results'/TASK/'models/dev'/str(block)/'full.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['episode'] == 400 and not saved['smoke']
    upper, lower = models('full')
    upper.load_state_dict(saved['upper']); lower.load_state_dict(saved['lower'])
    upper.eval(); lower.eval()
    return upper, lower, digest(path)


def trajectory(args):
    import numpy as np
    import torch
    from environment.dynamic_service import DynamicServiceEnv
    from utils.dynamic_service_baselines import HistoryController, matching_search
    from utils.dynamic_service_scenarios import Scenario, seed_for
    upper, lower, sha = model(args, args.block)
    # Sampling stream is diagnostic-only; workloads are the original dev tapes.
    torch.manual_seed(seed_for('diagnostic/20261030/sampling', args.block*3+args.workload))
    tape = Scenario(seed_for(f'dev/evaluation/{args.workload}', 0), 'D')
    env = DynamicServiceEnv(tape, 1.)
    hidden = upper.initial_state(); controller = HistoryController()
    root = args.task/'results'/TASK/'evaluation/dev'
    expected_method = dict(full_replay='full', matching_replay='matching_lower', history='history').get(args.mode)
    expected_path = root/str(0 if args.mode == 'history' else args.block)/str(expected_method)/f'D_1.0_{args.workload}_0.json'
    expected = json.loads(expected_path.read_text()) if expected_method else None
    totals = dict(generated=0, admitted=0, success=0, mbs=0)
    metrics = defaultdict(list); windows = [dict(generated=0, admitted=0, success=0) for _ in range(10)]
    trace = []; mismatch = []; actor_means = []; mean_trace=[]; demand_probes=[]
    hook = upper.actor.register_forward_hook(lambda module, inputs, result: actor_means.append(result.detach().cpu().numpy()))
    for t in range(1000):
        before = env.positions.copy(); obs = env.observe(); batch = env.batch()
        controller.update(obs, before)
        learned_upper = args.mode not in ('history', 'hover', 'history_learned_lower')
        if learned_upper:
            hidden_before=hidden.clone()
            policy_obs = obs
            if args.mode == 'zero_feedback_matching':
                policy_obs = dict(obs, feedback=np.zeros_like(obs['feedback']))
            if args.mode == 'reset_memory_matching':
                hidden = upper.initial_state()
            action = upper.act(policy_obs, hidden, stochastic=args.mode == 'stochastic_upper_matching')
            hidden = action['hidden']; actions = action['actions']
            means = actor_means.pop()
            mean_trace.append(means)
            metrics['mean_action_norm'].extend(np.linalg.norm(means, axis=1).tolist())
            metrics['raw_component_saturation'].extend((np.abs(action['raw']) >= 1).ravel().tolist())
            if args.detail and t%100==0:
                no_demand=dict(obs,ue=obs['ue'].copy());no_demand['ue'][:,:,2:]=0
                no_feedback=dict(obs,feedback=np.zeros_like(obs['feedback']))
                with torch.no_grad():
                    base=upper(obs,hidden_before)[0].mean
                    demand=upper(no_demand,hidden_before)[0].mean
                    memory=upper(obs,upper.initial_state())[0].mean
                    feedback=upper(no_feedback,hidden_before)[0].mean
                demand_probes.append(dict(t=t,mean_action=base.tolist(),
                    zero_current_demand_delta=float((demand-base).norm(dim=1).mean()),
                    zero_history_delta=float((memory-base).norm(dim=1).mean()),
                    zero_feedback_delta=float((feedback-base).norm(dim=1).mean())))
                actor_means.clear()
        else:
            actions = np.zeros((5, 2)) if args.mode == 'hover' else controller.actions(before)
        assignment = lower.act(batch)['assignment'] if args.mode in ('full_replay', 'history_learned_lower') else matching_search(batch)
        # Replay original fallback events for exact baselines. Added diagnostic
        # instrumentation is excluded from the formal one-second timing rule.
        timeout = bool(expected['rows'][t]['decision_timeout']) if expected else False
        out = env.step(actions, assignment, decision_timeout=timeout)
        if timeout:
            hidden = upper.initial_state()
        if expected:
            row = expected['rows'][t]
            if any(out[k] != row[k] for k in ('generated', 'admitted', 'success', 'mbs', 'assignment')):
                mismatch.append(t)
        for key in totals:
            totals[key] += out[key]
        for key in windows[t//100]:
            windows[t//100][key] += out[key]
        executed = env.positions-before
        norm = np.linalg.norm(actions, axis=1, keepdims=True)
        nominal = actions/np.maximum(norm, 1e-12)*np.minimum(norm, 1)*15
        if timeout:
            nominal *= 0
        desired = np.linalg.norm(nominal, axis=1); actual = np.linalg.norm(executed, axis=1)
        metrics['intended_step_m'].extend(desired.tolist())
        metrics['actual_step_m'].extend(actual.tolist())
        metrics['projection_residual_m'].extend(np.linalg.norm(executed-nominal, axis=1).tolist())
        metrics['hover_fraction'].extend((actual < .1).tolist())
        metrics['boundary_fraction'].extend(((env.positions < 51) | (env.positions > 649)).any(axis=1).tolist())
        outward=((before<51)&(actions<0))|((before>649)&(actions>0))
        metrics['outward_on_boundary_fraction'].extend(outward.any(axis=1).tolist())
        requests = tape.slot(t)
        active = [q.ue for q in requests]
        near = np.linalg.norm(env.ue_positions[active, None]-before[None], axis=-1).min(axis=1)
        metrics['active_request_nearest_uav_m'].extend(near.tolist())
        # Ground-truth center is measured only here; never given to the actor.
        metrics['hotspot_nearest_uav_m'].append(float(np.linalg.norm(before-np.array(tape.center(t)), axis=1).min()))
        if t % 25 == 0:
            trace.append(dict(t=t, positions=before.tolist(), admitted=out['admitted'], success=out['success']))
    hook.remove()
    trace.append(dict(t=1000, positions=env.positions.tolist()))
    if mismatch:
        raise AssertionError(f'baseline replay mismatch: {args.mode} {args.block} {args.workload}: {mismatch[:5]}')
    return dict(mode=args.mode, block=args.block, workload=args.workload, repeat=0, totals=totals,
        dsr=totals['success']/totals['generated'], admission=totals['admitted']/totals['generated'],
        conditional=totals['success']/max(totals['admitted'], 1), metrics={k:float(np.mean(x)) for k,x in metrics.items()},
        std=upper.log_std.clamp(-3, 1).exp().detach().tolist(), checkpoint_sha256=sha,
        request_fingerprint=tape.fingerprint(), trace=trace, windows=windows,
        mean_action_per_uav=np.mean(mean_trace,axis=0).tolist() if mean_trace else None,
        mean_action_temporal_std_per_uav=np.std(mean_trace,axis=0).tolist() if mean_trace else None,
        demand_probes=demand_probes,
        replay_exact=not mismatch if expected else None, diagnostic_only=True)


def lower_audit(args):
    import numpy as np
    import torch
    from environment.dynamic_service import Batch, evaluate
    from scripts.analysis.dynamic_service_experiment import models
    upper, lower, sha = model(args, args.block)
    records=[]; root=args.task/'results'/TASK
    warm = None
    migration = args.task/'saved_models'/TASK/'pre_resource_migration_20261029'
    files = sorted(migration.glob(f'train_dev_{args.block}_full_episode*.pt'))
    if files:
        saved=torch.load(files[-1],map_location='cpu',weights_only=False)
        _,warm=models('full');warm.load_state_dict(saved['lower']);warm.eval()
    torch.manual_seed(18300+args.block)
    for f in sorted((root/'evaluation/dev'/str(args.block)/'full').glob('D_1.0_*.json')):
        data=json.loads(f.read_text())
        for snapshot in data.get('snapshots',[]):
            batch=Batch(**{k:np.asarray(x) for k,x in snapshot['batch'].items()})
            n=len(batch)
            if not n:
                continue
            encoded=lower.encode(batch);chosen=[];entropy=[];mbs_prob=[];miss_avoidable=0;mbs_override=0
            for i in range(n):
                with torch.no_grad():
                    dist,value=lower.distribution(batch,encoded,chosen,i)
                target=int(dist.logits.argmax());chosen.append(target)
                entropy.append(float(dist.entropy()));mbs_prob.append(float(dist.probs[5]))
                counts=np.bincount(chosen[:-1],minlength=6)
                latency=batch.tx[i]+batch.cycles[i]*(counts+1)/batch.capacity
                feasible=(latency<=batch.deadline[i])&batch.legal[i]
                if latency[target]>batch.deadline[i] and feasible.any():
                    miss_avoidable+=1
                    mbs_override+=target==5
            actual=evaluate(batch,chosen)
            assert actual['success']==snapshot['controls']['full']['success']
            failed=~actual['satisfied'];targets=np.array(chosen)
            original_dynamic=lower.dynamic_features
            lower.dynamic_features=lambda b,c,i: torch.zeros((6,4),device=lower.device)
            zero=lower.act(batch)['assignment'];lower.dynamic_features=original_dynamic
            sampled=[evaluate(batch,lower.act(batch,stochastic=True)['assignment'])['success'] for _ in range(8)]
            warm_result=evaluate(batch,warm.act(batch)['assignment'])['success'] if warm else None
            detail={}
            if args.detail:
                rescue=targets.copy();rescued=actual['success'];one_move_best=rescued
                # Offline counterfactual only. No repaired action is ever sent
                # to an environment or reported as learned-policy performance.
                for i in np.flatnonzero(targets==5):
                    for node in np.flatnonzero(batch.legal[i,:5]):
                        trial=targets.copy();trial[i]=node
                        one_move_best=max(one_move_best,evaluate(batch,trial)['success'])
                    best=None
                    for node in np.flatnonzero(batch.legal[i,:5]):
                        trial=rescue.copy();trial[i]=node;quality=evaluate(batch,trial)['success']
                        if quality>rescued and (best is None or quality>best[0]):best=(quality,trial)
                    if best:rescued,rescue=best
                detail=dict(mbs_transmission_alone_late=int(((targets==5)&(batch.tx[:,5]>batch.deadline)).sum()),
                    mbs_singleton_already_late=int(((targets==5)&(batch.tx[:,5]+batch.cycles/batch.capacity[5]>batch.deadline)).sum()),
                    mbs_one_move_best_success=one_move_best,mbs_pass_rescue_success=rescued,
                    minimum_mbs_tx=float(batch.tx[:,5].min()),median_mbs_tx=float(np.median(batch.tx[:,5])))
            records.append(dict(workload=data['workload'],repeat=data['repeat'],t=snapshot['t'],requests=n,
                full_success=actual['success'],recorded_controls={k:x['success'] for k,x in snapshot['controls'].items()},
                counts=actual['counts'].tolist(),failures_by_node=np.bincount(targets[failed],minlength=6).tolist(),
                transmissions_alone_late=int((batch.tx[np.arange(n),targets]>batch.deadline).sum()),
                mean_predicted_value=float(value),
                mean_entropy=float(np.mean(entropy)),mean_mbs_probability=float(np.mean(mbs_prob)),
                avoidable_current_request_misses=miss_avoidable,avoidable_mbs_current_misses=int(mbs_override),
                zero_dynamic_success=evaluate(batch,zero)['success'],zero_dynamic_assignment_changes=int((zero!=targets).sum()),
                sampled_success_mean=float(np.mean(sampled)),warm_checkpoint_success=warm_result,**detail))
    return dict(mode=args.mode,block=args.block,checkpoint_sha256=sha,records=records,
        warm_checkpoint=str(files[-1]) if files else None,diagnostic_only=True)


def parameter_and_log_audit(args):
    import numpy as np
    from collections import defaultdict
    root=args.task/'results'/TASK;out=[]
    for block in range(5):
        upper,lower,sha=model(args,block)
        phases=defaultdict(lambda:dict(generated=0,admitted=0,success=0,episodes=0))
        losses=[]
        for f in sorted((root/'training/dev'/str(block)/'full').glob('[0-9][0-9][0-9][0-9].json')):
            d=json.loads(f.read_text());e=d['episode'];phase='warm' if e<=50 else ('upper' if e<=200 else 'alternating')
            key=f'{phase}/{d["kind"]}/{d["scale"]}';q=phases[key]
            for k in ('generated','admitted','success'):q[k]+=d[k]
            q['episodes']+=1
            if d['lower_loss'] is not None:losses.append(dict(episode=e,loss=d['lower_loss']))
        parameters={}
        for name,policy in [('upper',upper),('lower',lower)]:
            parameters[name]=dict(finite=all(bool(np.isfinite(x.detach().numpy()).all()) for x in policy.parameters()),
                                  max_abs=max(float(x.detach().abs().max()) for x in policy.parameters()))
        out.append(dict(block=block,checkpoint_sha256=sha,parameters=parameters,phases=dict(phases),lower_loss_series=losses))
    return dict(mode=args.mode,blocks=out,diagnostic_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--extension',type=Path,required=True)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--mode',choices=MODES+('hover','history','lower_audit','log_audit'))
    parser.add_argument('--block',type=int,default=0)
    parser.add_argument('--workload',type=int,default=0)
    parser.add_argument('--detail',action='store_true',help='Additional offline sensitivity measurements')
    args=parser.parse_args()
    if sys.platform!='linux':parser.error('numerical diagnostics are server-only')
    args.source,args.extension,args.task=(x.resolve() for x in (args.source,args.extension,args.task))
    root=args.task/'results'/TASK/'diagnostic_20261030'
    if args.mode:
        numerical_imports(args)
        result=lower_audit(args) if args.mode=='lower_audit' else (parameter_and_log_audit(args) if args.mode=='log_audit' else trajectory(args))
        atomic(root/('detail' if args.detail else 'cases')/f'{args.mode}_{args.block}_{args.workload}.json',result)
        return
    sys.path.insert(0,str(args.source))
    from scripts.analysis.dynamic_service_experiment import protocol
    expected=json.loads((args.task/'results'/TASK/'protocol.json').read_text())
    contract=hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()
    # protocol.json stores the full definition, not just its digest.
    if hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()!=contract:
        raise RuntimeError('frozen scientific source changed')
    if not 1<=args.workers<=32:parser.error('workers must be 1..32')
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    validation=args.task/'output'/TASK/'validation/diagnostic_20261030'
    validation.mkdir(parents=True,exist_ok=True)
    lock=(validation/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=[(mode,b,w) for mode in MODES for b in range(5) for w in range(3)]
    cases += [(mode,0,w) for mode in ('hover','history') for w in range(3)]
    cases += [('lower_audit',b,0) for b in range(5)]+[('log_audit',0,0)]
    # One hardware thread from each physical core; leave the first eight cores.
    seen=set();cpus=[]
    for cpu in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((p/'physical_package_id').read_text(),(p/'core_id').read_text())
        if key not in seen:seen.add(key);cpus.append(cpu)
    cpus=cpus[8:8+args.workers]
    manifest=dict(started=datetime.now(timezone(timedelta(hours=8))).isoformat(),script_sha256=digest(Path(__file__)),
        scientific_contract=contract,cases=list(cases),cpus=cpus,diagnostic_only=True,training=False,
        scope='three prespecified existing dev workloads, repeat zero; frozen snapshots from all primary episodes')
    atomic(root/'manifest.json',manifest)
    active=[];done=[];started=time.time()
    while cases or active:
        for item in active[:]:
            code=item['process'].poll()
            if code is not None:
                item['log'].close();active.remove(item)
                if code:raise RuntimeError(f'diagnostic failed: {item["case"]}; other cases preserved')
                done.append(item['case'])
        free=[cpu for cpu in cpus if cpu not in {x['cpu'] for x in active}]
        while cases and free:
            mode,b,w=cases.pop(0);cpu=free.pop(0);name=f'{mode}_{b}_{w}'
            log=(validation/(name+'.log')).open('a')
            cmd=['taskset','-c',str(cpu),sys.executable,str(Path(__file__).resolve()),'--source',str(args.source),
                 '--extension',str(args.extension),'--task',str(args.task),'--mode',mode,'--block',str(b),'--workload',str(w)]
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            active.append(dict(process=process,log=log,case=(mode,b,w),cpu=cpu))
        atomic(validation/'status.json',dict(elapsed=time.time()-started,completed=len(done),total=len(manifest['cases']),
             active=[dict(pid=x['process'].pid,case=x['case']) for x in active],diagnostic_only=True))
        time.sleep(2)
    summary=dict(status='COMPLETE',manifest=manifest,cases=[json.loads(f.read_text()) for f in sorted((root/'cases').glob('*.json'))])
    atomic(root/'summary.json',summary)


if __name__=='__main__':
    main()
