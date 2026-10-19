"""Stage 0a/0b only: frozen snapshots, prefix truncation and measured software fees.

No fitting, optimizer updates, confirmation selection or delayed state mutation.
Numerical imports, model execution and scientific aggregation are Linux-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

FILES = ('scripts/analysis/sf_hlm_learning_feasibility.py', 'utils/sf_hlm_feasibility.py',
         'tests/test_sf_hlm_learning_feasibility.py', 'paths.py',
         'docs/plans/sf_hlm_learning_feasibility_20261016.md')
SOURCES = ('learned', 'greedy')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def serial(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=serial, indent=2) + '\n')
    temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def core(args): return load('sf_feasibility_core', args.release_root/'utils/sf_hlm_feasibility.py')


def parent_args(args):
    launch = read(args.parent_launch); command = launch['command']; data = {}
    for i, value in enumerate(command[:-1]):
        if value.startswith('--') and value not in ('--workers', '--timing-workers'):
            data[value[2:].replace('-', '_')] = Path(command[i+1])
    required = ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results', 'resources',
                'adaptation_root', 'followup_root', 'parent_results', 'results_root', 'release_root')
    if any(k not in data for k in required): raise ValueError('Incomplete parent launch')
    return argparse.Namespace(**data)


def task_id(task): return '_'.join(map(str, task))


def task_dir(args, task): return args.results_root/'tasks'/task_id(task)


def state_dir(args, identity):
    source, block, workload, episode, step = identity
    return args.results_root/'states'/source/f'block_{block}_workload_{workload}'/f'episode_{episode:05d}'/f'step_{step:05d}'


def parent_episode(a, source, block, workload, episode):
    phase, method = ('capture', 'old_learned') if source == 'learned' else ('evaluate', 'greedy')
    return a.results_root/phase/method/f'block_{block}_workload_{workload}'/f'episode_{episode:05d}'


def inventory(args):
    if sys.platform != 'linux': raise RuntimeError('Remote archive inventory required')
    a = parent_args(args); old = read(a.results_root/'protocol.json'); c = core(args)
    if old['smoke'] or old['blocks'] != list(range(620001, 620006)) or old['workloads'] != list(range(780001, 780006)):
        raise ValueError('Unexpected development split')
    if old['steps'] != 1000 or old['snapshot_steps'] != [0, 250, 500, 750]: raise ValueError('Changed sampling contract')
    if sha(a.output/'protocol.json') != old['inputs']['old_protocol'] or sha(a.parent_results/'protocol.json') != old['inputs']['parent_protocol']:
        raise ValueError('Changed ancestor protocol')
    for rel, checksum in old['files'].items():
        if sha(a.release_root/rel) != checksum: raise ValueError('Changed parent release: '+rel)
    protocols = [a.results_root/'protocol.json', a.output/'protocol.json', a.parent_results/'protocol.json']
    input_files = {str(q): sha(q) for q in protocols}; episodes = []; states = []
    for b in old['blocks']:
        checkpoint = a.output/f'lower_training/complete/learned/block_{b}/checkpoint/quota_v2.pt'
        if sha(checkpoint) != old['inputs'][f'lower_{b}']: raise ValueError('Changed frozen quota model')
        input_files[str(checkpoint)] = sha(checkpoint)
        for w, e in itertools.product(old['workloads'], range(2)):
            for source in SOURCES:
                folder = parent_episode(a, source, b, w, e); receipt = read(folder/'result.json')
                expected_task = ['capture' if source == 'learned' else 'evaluate',
                                 'old_learned' if source == 'learned' else 'greedy', b, w, e, -1]
                if receipt['status'] != 'PASS' or receipt['task'] != expected_task or receipt['steps'] != 1000 or receipt['protocol_sha256'] != sha(a.results_root/'protocol.json'):
                    raise ValueError('Invalid parent episode: '+str(folder))
                checked = 0
                for rel, checksum in receipt['files'].items():
                    if sha(folder/rel) != checksum: raise ValueError('Parent input hash differs: '+str(folder/rel))
                    checked += 1
                input_files[str(folder/'result.json')] = sha(folder/'result.json')
                ep = read(folder/'episode.json')
                if len(ep['steps']) != 1000: raise ValueError('Partial parent trajectory')
                if source == 'learned' and receipt.get('snapshots') != 4: raise ValueError('Missing parent snapshots')
                for step in old['snapshot_steps']:
                    row = ep['steps'][step]
                    if row['step'] != step: raise ValueError('Parent step order changed')
                    states.append(dict(identity=[source, b, w, e, step],
                        generated=int(row['metrics']['generated_service_count']),
                        admitted=int(row['metrics']['admitted_service_count']),
                        reuse=source == 'learned', reference_episode=str(folder/'episode.json')))
                episodes.append(dict(source=source, block=b, workload=w, episode=e,
                                     receipt=str(folder/'result.json'), verified_files=checked))
    c.require_population([tuple(q['identity']) for q in states],
        itertools.product(SOURCES, old['blocks'], old['workloads'], range(2), old['snapshot_steps']))
    result = dict(status='PASS', stage='0a', blocks=old['blocks'], workloads=old['workloads'],
        episodes=2, steps=1000, snapshot_steps=old['snapshot_steps'], states=states,
        reused_snapshots=200, new_greedy_snapshots=200, source_episodes=episodes,
        input_files=input_files, parent_release_files={str(a.release_root/rel): h for rel, h in old['files'].items()},
        actual_request_latency_in_parent_snapshots=False,
        required_reconstruction='Evaluate private snapshot copies to retain per-request latency/deadline.',
        no_confirmation_outcomes_used=True, no_training=True, parent_launch_sha256=sha(args.parent_launch))
    write(args.validation_root/'inventory.json', result)
    print(json.dumps(dict(status='PASS', reused=200, new=200, verified_episodes=len(episodes)), ensure_ascii=False))
    return result


def create_protocol(args):
    inv = read(args.validation_root/'inventory.json'); a = parent_args(args); c = core(args)
    if inv['status'] != 'PASS' or sha(args.parent_launch) != inv['parent_launch_sha256']: raise ValueError('Missing valid inventory')
    blocks, workloads, episodes, points = inv['blocks'], inv['workloads'], 2, inv['snapshot_steps']
    if args.smoke:
        blocks, workloads, episodes, points = blocks[:2], workloads[:1], 1, [0, 250]
    identities = [list(q) for q in itertools.product(SOURCES, blocks, workloads, range(episodes), points)]
    captures = [list(q) for q in itertools.product(['capture'], SOURCES, blocks, workloads, range(episodes))]
    benchmarks = [['benchmark', *q] for q in identities]
    p = dict(version='sf_hlm_learning_feasibility_20261016_v1', purpose='verification' if args.smoke else 'development',
        smoke=args.smoke, identities=identities, tasks=captures+benchmarks, blocks=blocks, workloads=workloads,
        episodes=episodes, points=points, steps=1000, random_repeats=5, timing_repeats=3,
        methods=list(c.METHODS), fixed_rule=dict(feature='admitted_batch_size', threshold=16, yes='L8', no='G'),
        software_overhead_ms=[0, .5, 1, 2, 5], feature_budget=16, actual_caps=[8, 16], repair_budget=128,
        gates=dict(practical_gain=.005, learning_gain=.001, raw_loss=.001, positive_blocks=4,
                   actual_energy_ratio=1.05, software_p95=1.),
        inventory_sha256=sha(args.validation_root/'inventory.json'), parent_launch_sha256=sha(args.parent_launch),
        input_files=inv['input_files'], parent_release_files=inv['parent_release_files'],
        release_files={rel: sha(args.release_root/rel) for rel in FILES},
        state_delay_applied=False, training=False, confirmation_results_used=False,
        inference_unit='five historical training blocks; snapshots/repetitions are subsamples',
        timing_contract='Full prepare+upper+initialization+repair+search+mapping+serialization; no snapshot restore or physics.',
        timing_noise='Three measured samples retained; score each fee before averaging; no outlier exclusions.',
        hard_failure='Any hash/action/prefix/identity mismatch stops dispatch; no outcome-dependent reruns.',
        physical_protocol='original frozen SF-HLM batch environment; no waiting or continuous queue added')
    path = args.results_root/'protocol.json'
    if path.exists() and read(path) != p: raise ValueError('Existing protocol differs; preserve it and use a new namespace')
    write(path, p); return p


def verify_release(args, p, external=False):
    for rel, checksum in p['release_files'].items():
        if sha(args.release_root/rel) != checksum: raise ValueError('Frozen new release changed: '+rel)
    if sha(args.parent_launch) != p['parent_launch_sha256']: raise ValueError('Launch input changed')
    if external:
        for path, checksum in {**p['input_files'], **p['parent_release_files']}.items():
            if sha(path) != checksum: raise ValueError('Frozen external input changed: '+path)


def receipt(args, task, **extra):
    folder = task_dir(args, task)
    files = {str(f.relative_to(folder)): sha(f) for f in folder.rglob('*')
             if f.is_file() and f.name != 'result.json' and not f.name.endswith('.tmp')}
    write(folder/'result.json', dict(status='PASS', task=task, protocol_sha256=sha(args.results_root/'protocol.json'),
                                   files=files, **extra))


def completed(args, task):
    path = task_dir(args, task)/'result.json'
    if not path.exists(): return False
    d = read(path)
    if d['status'] != 'PASS' or d['task'] != task or d['protocol_sha256'] != sha(args.results_root/'protocol.json'):
        raise ValueError('Changed task identity')
    for rel, checksum in d['files'].items():
        if sha(path.parent/rel) != checksum: raise ValueError('Changed task output')
    for name, checksum in d.get('state_files', {}).items():
        if sha(name) != checksum: raise ValueError('Changed state output/input')
    return True


def boot(args, block):
    a = parent_args(args); a.block=block; a.seed=0; a.family='complete'
    old_runner = load('feasibility_parent', a.release_root/'scripts/analysis/sf_hlm_learning_improvement.py')
    followup, runner, old = old_runner.initialize_numerical(a)
    models, lower, basis = old_runner.basis(a, followup, runner, old)
    return a, old_runner, runner, models, lower, basis


def capture_task(args, p, task, runtime):
    import numpy as np
    from utils.snapshot_evaluator import capture
    from utils.canonical_experiment import digest, exogenous_frame
    a, old_runner, runner, models, lower, basis = runtime
    _, source, block, workload, episode = task
    parent = parent_episode(a, source, block, workload, episode)
    if sha(parent/'result.json') != p['input_files'][str(parent/'result.json')]:
        raise ValueError('Parent receipt changed after inventory')
    expected_files = read(parent/'result.json')['files']
    if sha(parent/'episode.json') != expected_files['episode.json']:
        raise ValueError('Parent episode changed after inventory')
    original = read(parent/'episode.json'); manifests = {}; collected = []
    if source == 'greedy':
        env, obs = runner.episode_env(workload, episode); external = hashlib.sha256()
    for step in range(p['steps'] if source == 'greedy' else 0):
        external.update(digest(exogenous_frame(env)).encode())
        if step in p['points']: observed = pickle.dumps((env, obs), protocol=pickle.HIGHEST_PROTOCOL)
        motion, actions, info, _, _, rows = runner.policy(env, obs, 'greedy', models, family='complete', lower=lower)
        if step in p['points']:
            folder = state_dir(args, [source, block, workload, episode, step]); folder.mkdir(parents=True, exist_ok=True)
            snapshot = capture(env, motion)
            (folder/'snapshot.bin').write_bytes(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
            (folder/'observed.bin').write_bytes(observed)
            write(folder/'input.json', dict(rows=rows, motion=motion, request_ids=info['request_ids'],
                                            snapshot_hash=snapshot.snapshot_hash, policy=info))
        obs, _, metrics = env.step(motion, offloading_actions=actions); obs=np.asarray(obs, np.float32)
        before = original['steps'][step]
        if metrics != before['metrics'] or list(info['assignment']) != before['assignment'] or info['request_ids'] != before['request_ids']:
            raise ValueError('Greedy trajectory replay differs at '+str([block, workload, episode, step]))
    if source == 'greedy' and external.hexdigest() != original['exogenous_trace_hash']: raise ValueError('Greedy exogenous trace changed')
    for step in p['points']:
        identity=[source, block, workload, episode, step]; dest=state_dir(args, identity)
        src=parent/f'step_{step:05d}' if source == 'learned' else dest
        if source == 'learned':
            for name in ('snapshot.bin','observed.bin','input.json'):
                if sha(src/name) != expected_files[str((src/name).relative_to(parent))]:
                    raise ValueError('Original snapshot changed after inventory')
        before=original['steps'][step]; inp=read(src/'input.json')
        if inp['policy']['assignment'] != before['assignment'] or inp['request_ids'] != before['request_ids']:
            raise ValueError('Snapshot reference differs')
        files={str(src/name):sha(src/name) for name in ('snapshot.bin','observed.bin','input.json')}
        meta=dict(identity=identity, files=files, snapshot=str(src/'snapshot.bin'), observed=str(src/'observed.bin'),
            input=str(src/'input.json'), generated=int(before['metrics']['generated_service_count']),
            admitted=int(before['metrics']['admitted_service_count']), potential=int(before['potential']),
            reference_metrics=before['metrics'], reference_assignment=before['assignment'], basis=basis,
            reference_episode=str(parent/'episode.json'), reference_episode_sha256=sha(parent/'episode.json'))
        write(dest/'state.json', meta); manifests.update(files); manifests[str(dest/'state.json')]=sha(dest/'state.json');collected.append(identity)
    receipt(args, task, state_files=manifests, identities=collected, basis=basis)


def validate_prefix(c, rows, lower, num_uavs, identity):
    import numpy as np
    from utils.sf_hlm_v2 import search
    checks=[]
    for kind in ('learned','random','heuristic'):
        seed=c.seed_for('prefix_validation',identity)
        full, fd, fr=c.truncated_search(rows,num_uavs=num_uavs,kind=kind,model=lower,rng=np.random.default_rng(seed),cap=16,trace=True)
        short, sd, sr=c.truncated_search(rows,num_uavs=num_uavs,kind=kind,model=lower,rng=np.random.default_rng(seed),cap=8,trace=True)
        reference, rd, rr=search(rows,num_uavs=num_uavs,kind=kind,family='complete',model=lower,rng=np.random.default_rng(seed),budget=16)
        if full != reference or [q['action'] for q in fr] != [q['action'] for q in rr]: raise ValueError('16-call adapter differs from frozen search')
        for key in ('matching_calls','base_evaluations','common_energy_cap','chosen_predicted','termination'):
            if fd[key] != rd[key]: raise ValueError('Frozen search diagnostic differs: '+key)
        if sd['prefix'] != fd['prefix'][:len(sd['prefix'])]: raise ValueError('Truncated prefix changed')
        if not rows:
            if short or sd['matching_calls'] != 0: raise ValueError('Empty batch lost')
        elif sd['matching_calls'] >= 8:
            if list(short) != fd['prefix'][len(sd['prefix'])-1]['assignment']: raise ValueError('Wrong retained prefix assignment')
        elif short != full: raise ValueError('Early STOP differs')
        checks.append(dict(kind=kind, full_calls=fd['matching_calls'], short_calls=sd['matching_calls'],
                           prefix_actions=len(sr), reference_actions=len(rr)))
    return checks


def benchmark_task(args, p, task, runtime):
    import numpy as np
    import config
    from utils.canonical_experiment import digest
    from utils.snapshot_evaluator import evaluate
    _, _, runner, models, lower, basis=runtime; c=core(args); identity=task[1:]
    meta=read(state_dir(args,identity)/'state.json')
    for name,checksum in meta['files'].items():
        if sha(name)!=checksum: raise ValueError('State bytes changed')
    observed=Path(meta['observed']).read_bytes(); snapshot=pickle.loads(Path(meta['snapshot']).read_bytes()); inp=read(meta['input'])
    expected_rows=inp['rows']
    for row in expected_rows:row['options']={int(k):v for k,v in row['options'].items()}
    env,obs=pickle.loads(observed);_,_,requests,rows=runner.base.prepare(env)
    if digest(rows)!=digest(expected_rows) or [q[2] for q in requests]!=inp['request_ids'] or snapshot.snapshot_hash!=inp['snapshot_hash']:
        raise ValueError('Observed or physical state restoration differs')
    fingerprint=runner.base.policy_fingerprint([models['upper'],lower])
    prefix=validate_prefix(c,rows,lower,config.NUM_UAVS,identity)
    specs=c.branch_specs(identity,p['random_repeats']); original_reference=None; results={}
    # One equal warmup for every algorithm/repetition, never counted as a formal sample.
    for spec in specs:
        env,obs=pickle.loads(observed); c.complete_policy(env,obs,spec,models,lower,runner)
    for iteration in range(p['timing_repeats']):
        order=np.random.default_rng(c.seed_for('formal_order',identity,iteration)).permutation(len(specs))
        for index in order:
            spec=specs[int(index)];env,obs=pickle.loads(observed)
            motion,_,info,actual_rows=c.complete_policy(env,obs,spec,models,lower,runner)
            if digest(actual_rows)!=digest(expected_rows) or info['request_ids']!=inp['request_ids'] or not np.allclose(motion,snapshot.motion,rtol=0,atol=1e-7):
                raise ValueError('Formal state/action identity differs')
            d=info['diagnostic']
            if d['matching_calls']>spec['cap'] or d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7: raise ValueError('Online guard violated')
            key=(spec['method'],spec['repeat']);row=results.get(key)
            if row is None:
                actual=evaluate(snapshot,info['assignment']);metrics=actual['metrics'];requests_out=actual['requests']
                if int(metrics['generated_service_count'])!=meta['generated'] or len(requests_out)!=meta['admitted']: raise ValueError('State denominator changed')
                raw=c.fee_count(requests_out,0.)
                if raw!=int(metrics['deadline_satisfied_count']):raise ValueError('Zero-fee score changed')
                if spec['method']==('L16' if identity[0]=='learned' else 'G'):
                    if list(info['assignment'])!=meta['reference_assignment']:raise ValueError('Original policy assignment changed')
                    for field,value in meta['reference_metrics'].items():
                        if not math.isclose(float(metrics[field]),float(value),rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Original snapshot metric changed: '+field)
                    original_reference=True
                row=dict(**spec,assignment=list(info['assignment']),requests=requests_out,raw_success=raw,
                    generated=meta['generated'],admitted=meta['admitted'],metrics=metrics,diagnostic=d,
                    software_seconds=[],timing_iteration=[],selected_order=[])
                results[key]=row
            elif row['assignment']!=list(info['assignment']):raise ValueError('Timing repetition changed allocation')
            row['software_seconds'].append(info['policy_seconds']);row['timing_iteration'].append(iteration);row['selected_order'].append(int(index))
    if not original_reference or fingerprint!=runner.base.policy_fingerprint([models['upper'],lower]):raise ValueError('Frozen policy/replay validation failed')
    for row in results.values():
        if len(row['software_seconds'])!=p['timing_repeats']:raise ValueError('Incomplete timing population')
        row['fee_success']=c.mean_fee_count(row['requests'],row['software_seconds'])
    payload=dict(identity=identity,state_sha256=sha(state_dir(args,identity)/'state.json'),basis=basis,
        generated=meta['generated'],admitted=meta['admitted'],potential=meta['potential'],
        prefix_checks=prefix,original_reference_exact=True,model_fingerprint=fingerprint,
        cpu=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),results=list(results.values()),
        timing_samples=len(specs)*p['timing_repeats'],state_delay_applied=False)
    write(task_dir(args,task)/'benchmark.json',payload)
    receipt(args,task,state_files={str(state_dir(args,identity)/'state.json'):payload['state_sha256']},
            identity=identity,timing_samples=payload['timing_samples'])


def worker(args):
    if sys.platform!='linux':raise RuntimeError('Numerical work only on experiment server')
    p=read(args.results_root/'protocol.json');task=json.loads(args.task)
    if task not in p['tasks']:raise ValueError('Unregistered task')
    verify_release(args,p);a=parent_args(args)
    resource=a.resources/('isolated' if task[0]=='benchmark' else '')
    res=load('feasibility_resources',a.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(args.results_root/'locks'/(task_id(task)+'.lock')):
        if completed(args,task):return
        with res.CpuLease(resource,task=['feasibility',*task],kind='environment'):
            runtime=boot(args,task[2])
            if task[0]=='capture':capture_task(args,p,task,runtime)
            elif task[0]=='benchmark':benchmark_task(args,p,task,runtime)
            else:raise ValueError('Unknown phase')


def command(args, task):
    return [sys.executable,str(args.release_root/FILES[0]),'worker','--release-root',str(args.release_root),
            '--parent-launch',str(args.parent_launch),'--results-root',str(args.results_root),
            '--validation-root',str(args.validation_root),'--deliverables',str(args.deliverables),
            '--task',json.dumps(task,separators=(',',':'))]


def dispatch(args,p,phase):
    tasks=[q for q in p['tasks'] if q[0]==phase];waiting=[q for q in tasks if not completed(args,q)]
    limit=args.timing_workers if phase=='benchmark' else args.capture_workers
    def run(task):
        folder=args.results_root/'worker_logs';folder.mkdir(parents=True,exist_ok=True)
        with (folder/(task_id(task)+'.log')).open('a') as log:
            result=subprocess.run(command(args,task),stdout=log,stderr=subprocess.STDOUT,cwd=args.release_root)
        if result.returncode or not completed(args,task):raise RuntimeError('Worker failed: '+task_id(task))
        return task
    with ThreadPoolExecutor(max_workers=limit) as pool:
        futures={pool.submit(run,q):q for q in waiting};done=len(tasks)-len(waiting)
        try:
            for future in as_completed(futures):
                future.result();done+=1
                write(args.results_root/'status.json',dict(status='RUNNING',phase=phase,completed=done,total=len(tasks),pid=os.getpid(),updated_unix=time.time()))
        except BaseException:
            for future in futures:future.cancel()
            raise


def paired(values):
    from scipy.stats import t
    n=len(values);mean=statistics.mean(values)
    radius=float(t.ppf(.975,n-1))*statistics.stdev(values)/math.sqrt(n) if n>1 else 0.
    exact=sum(abs(statistics.mean(s*v for s,v in zip(signs,values)))>=abs(mean)-1e-14
              for signs in itertools.product((-1,1),repeat=n))/(2**n)
    return dict(mean=mean,ci95=[mean-radius,mean+radius] if n>1 else None,values=values,n=n,
                positive=sum(v>0 for v in values),exact_signflip_p=exact)


def analyze(args,p):
    if sys.platform!='linux':raise RuntimeError('Scientific aggregation only on experiment server')
    c=core(args);rows=[]
    for task in p['tasks']:
        if not completed(args,task):raise ValueError('Partial population cannot be analyzed')
        if task[0]=='benchmark':rows.append(read(task_dir(args,task)/'benchmark.json'))
    c.require_population([tuple(q['identity']) for q in rows],map(tuple,p['identities']))
    selections=('oracle_all','oracle_nonlearning','oracle_L8_G','oracle_L16_G','rule_L8_G')
    block_data={}; per_state=[]
    for state in rows:
        means={};expected=c.branch_specs(state['identity'],p['random_repeats'])
        c.require_population([(q['method'],q['repeat']) for q in state['results']],[(q['method'],q['repeat']) for q in expected])
        for method in c.METHODS:
            reps=[q for q in state['results'] if q['method']==method]
            for q in reps:
                if len(q['software_seconds'])!=p['timing_repeats'] or not math.isclose(q['fee_success'],c.mean_fee_count(q['requests'],q['software_seconds']),abs_tol=1e-12):
                    raise ValueError('Fee score or repetition changed')
            means[method]=dict(fee=statistics.mean(q['fee_success'] for q in reps),raw=statistics.mean(q['raw_success'] for q in reps),
                energy=statistics.mean(q['metrics']['uav_fleet_total_energy'] for q in reps),
                allocation_energy=statistics.mean(q['metrics']['lower_assignment_energy'] for q in reps),
                calls=statistics.mean(q['diagnostic']['matching_calls'] for q in reps),
                mbs=statistics.mean(q['metrics']['service_offloads_mbs'] for q in reps),
                fairness=statistics.mean(q['metrics']['fairness'] for q in reps),
                times=[t for q in reps for t in q['software_seconds']])
        source,b,*_=state['identity'];per_state.append(dict(identity=state['identity'],generated=state['generated'],admitted=state['admitted'],methods=means))
        for subset in ('all',source):
            block=block_data.setdefault((subset,b),dict(generated=0,admitted=0,states=0,methods={}))
            block['generated']+=state['generated'];block['admitted']+=state['admitted'];block['states']+=1
            values={m:v['fee'] for m,v in means.items()}
            for name,group in [('oracle_all',c.METHODS),('oracle_nonlearning',c.NONLEARNING),('oracle_L8_G',('L8','G')),('oracle_L16_G',('L16','G'))]:
                values[name]=max(means[m]['fee'] for m in group)
            values['rule_L8_G']=means['L8' if state['admitted']>=16 else 'G']['fee']
            for overhead in p['software_overhead_ms']:
                fee={m:statistics.mean(c.mean_fee_count(q['requests'],[t+overhead/1000 for t in q['software_seconds']])
                    for q in state['results'] if q['method']==m) for m in c.METHODS}
                values[f'oracle_L8_G_overhead_{overhead}ms']=max(fee['L8'],fee['G'])
                values[f'oracle_L16_G_overhead_{overhead}ms']=max(fee['L16'],fee['G'])
            for m,v in values.items():
                item=block['methods'].setdefault(m,dict(success=0.,raw=0.,energy=0.,calls=0.,mbs=0.,fairness=0.,times=[]))
                item['success']+=v
                if m in means:
                    for field in ('raw','energy','calls','mbs','fairness'):item[field]+=means[m][field]
                    item['times'].extend(means[m]['times'])
    subsets={}
    for subset in ('all',*SOURCES):
        blocks={str(b):block_data[subset,b] for b in p['blocks']};groups={}
        for b,block in blocks.items():
            for m,v in block['methods'].items():
                v['fee_rate']=v['success']/block['generated'] if block['generated'] else 0.
                if m in c.METHODS:
                    v['raw_rate']=v['raw']/block['generated'] if block['generated'] else 0.
                    v['mean_seconds']=statistics.mean(v['times']);v['p95_seconds']=float(__import__('numpy').quantile(v['times'],.95))
                    v['energy_per_state']=v['energy']/block['states'];v['calls_per_state']=v['calls']/block['states']
                    v['fairness_per_state']=v['fairness']/block['states'];v['mbs_share']=v['mbs']/block['admitted'] if block['admitted'] else 0.
                    del v['times']
        for m in next(iter(blocks.values()))['methods']:
            groups[m]=paired([q['methods'][m]['fee_rate'] for q in blocks.values()])
        pairs={m+'-G':paired([q['methods'][m]['fee_rate']-q['methods']['G']['fee_rate'] for q in blocks.values()]) for m in groups if m!='G'}
        for a,b in [('L8','L16'),('L8','R8'),('L8','R16'),('L8','H8'),('L8','H16'),
                    ('L16','R16'),('L16','H8'),('L16','H16'),('oracle_all','oracle_nonlearning')]:
            pairs[a+'-'+b]=paired([q['methods'][a]['fee_rate']-q['methods'][b]['fee_rate'] for q in blocks.values()])
        pairs['raw_L8-L16']=paired([q['methods']['L8']['raw_rate']-q['methods']['L16']['raw_rate'] for q in blocks.values()])
        subsets[subset]=dict(blocks=blocks,groups=groups,comparisons=pairs)
    combined=subsets['all'];g=p['gates'];best_nonlearning=max(c.NONLEARNING,key=lambda m:combined['groups'][m]['mean'])
    budget={}
    for method in ('L8','R8','H8'):
        diff=combined['comparisons'][method+'-G'];blocks=combined['blocks']
        er=statistics.mean(q['methods'][method]['energy']/max(q['methods']['G']['energy'],1e-12) for q in blocks.values())
        raw_loss=statistics.mean(q['methods'][method]['raw_rate']-q['methods']['L16']['raw_rate'] for q in blocks.values())
        within_time=all(q['methods'][method]['p95_seconds']<g['software_p95'] for q in blocks.values())
        budget[method]=dict(practical=diff['mean']>=g['practical_gain'] and diff['positive']>=g['positive_blocks'] and er<=g['actual_energy_ratio'] and within_time,
            difference_vs_G=diff,actual_energy_ratio=er,raw_difference_vs_L16=raw_loss,
            raw_preservation_description=raw_loss>=-g['raw_loss'],software_within_original_gate=within_time)
    choice=combined['comparisons']['oracle_all-oracle_nonlearning']['mean']
    gate_space={m:combined['comparisons']['oracle_'+m+'_G-G']['mean'] for m in ('L8','L16')}
    decision=dict(best_nonlearning=best_nonlearning,budget=budget,learning_oracle_increment=choice,
        binary_oracle_gains=gate_space,gate_probe_space=choice>=g['learning_gain'] and max(gate_space.values())>=g['practical_gain'],
        no_automatic_next_stage=True,not_independent_confirmation=True,
        timing_condition='reserved physical CPU cores; per-call scores averaged after thresholding')
    result=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS',protocol_sha256=sha(args.results_root/'protocol.json'),
        states=len(rows),training_blocks=len(p['blocks']),branch_instances=sum(len(q['results']) for q in rows),
        timing_samples=sum(q['timing_samples'] for q in rows),subsets=subsets,decision=decision,
        state_delay_applied=False,no_training=True,interpretation='Selected-state immediate fee scoring, not episode DSR or deployment evidence.')
    write(args.results_root/'analysis/summary.json',result);write(args.results_root/'analysis/state_summary.json',per_state)
    lines=['**学习增量同状态费用诊断：阶段0a/0b**','',result['interpretation'],'',
           f"状态 {len(rows)}；历史开发块 {len(p['blocks'])}；分支实例 {result['branch_instances']}；计时样本 {result['timing_samples']}。",'',
           '| 方法 | 计费后采样状态成功率/% | 原始成功率/% | 相对G/pp | 平均软件时间/ms | 块p95均值/ms | 匹配次数/状态 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in c.METHODS:
        bs=[q['methods'][m] for q in combined['blocks'].values()]
        delta=0. if m=='G' else combined['comparisons'][m+'-G']['mean']*100
        lines.append(f"| {m} | {combined['groups'][m]['mean']*100:.4f} | {statistics.mean(q['raw_rate'] for q in bs)*100:.4f} | {delta:+.4f} | {statistics.mean(q['mean_seconds'] for q in bs)*1000:.3f} | {statistics.mean(q['p95_seconds'] for q in bs)*1000:.3f} | {statistics.mean(q['calls_per_state'] for q in bs):.3f} |")
    lines+=['','所有对照使用同状态、相同生成分母；重复先平均成功数，不挑随机种子，不用平均时间代替逐次评分。',
        '原始分配、前缀及冻结模型验证逐状态通过。两种状态来源与块级差、区间、精确符号翻转敏感性见summary.json。',
        '固定规则只用预先冻结的批次大小16阈值。规则结果使用分支费用，尚未实测门控封装开销，因此是乐观诊断，不能晋级为部署结果。','',
        '开发闸门（实用阈值不是已有性能保证）：','', '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
        '本轮没有拟合门控、更新RL、使用原确认结果选候选或执行延迟状态回写。下一阶段须人工审阅闸门；本运行器不自动启动。','']
    args.deliverables.mkdir(parents=True,exist_ok=True);(args.deliverables/'analysis.md').write_text('\n'.join(lines))
    write(args.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(rows),stage='0b',updated_unix=time.time()))
    return result


def pipeline(args):
    if sys.platform!='linux':raise RuntimeError('Remote execution required')
    a=parent_args(args);profile=read(a.resources/'profile.json');os.sched_setaffinity(0,set(profile['cpus'][:profile['slots']]))
    res=load('feasibility_controller_resources',a.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(args.results_root/'controller.lock'):
        p=read(args.results_root/'protocol.json');verify_release(args,p,external=True)
        write(args.validation_root/'execution_environment.json',dict(python=sys.version,uname=list(os.uname()),
            resource_profile=profile,isolated_profile=read(a.resources/'isolated/profile.json'),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],pid=os.getpid(),started_unix=time.time()))
        try:
            for phase in ('capture','benchmark'):dispatch(args,p,phase)
            analyze(args,p)
        except BaseException as error:
            write(args.results_root/'status.json',dict(status='FAILED',error=str(error),pid=os.getpid(),updated_unix=time.time()))
            raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('inventory','init','pipeline','worker','analyze','status'))
    p.add_argument('--release-root',type=Path,default=REPO_ROOT)
    p.add_argument('--parent-launch',type=Path,required=True)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--validation-root',type=Path,required=True)
    p.add_argument('--deliverables',type=Path,required=True)
    p.add_argument('--task');p.add_argument('--smoke',action='store_true')
    p.add_argument('--capture-workers',type=int,default=16);p.add_argument('--timing-workers',type=int,default=8)
    return p


def main():
    args=parser().parse_args()
    for key in ('release_root','parent_launch','results_root','validation_root','deliverables'):setattr(args,key,getattr(args,key).resolve())
    if args.capture_workers<1 or not 1<=args.timing_workers<=8:raise ValueError('Invalid concurrency')
    if args.mode=='inventory':inventory(args)
    elif args.mode=='init':create_protocol(args)
    elif args.mode=='worker':worker(args)
    elif args.mode=='pipeline':pipeline(args)
    elif args.mode=='analyze':
        p=read(args.results_root/'protocol.json');verify_release(args,p);analyze(args,p)
    else:print(json.dumps(read(args.results_root/'status.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
