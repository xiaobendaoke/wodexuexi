"""Chapter 5 prospective development: run numerical modes on the SSH server only.

Frozen releases are external, read-only dependencies. This runner changes no
training weights, historical evidence, physical semantics or native policy code.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = list(range(620001, 620006))
SCENARIOS = {'center': (100, 1.), 'ue60': (60, 1.), 'ue140': (140, 1.),
             'cpu06': (100, .6), 'cpu14': (100, 1.4)}
TRANSFER = ['G', 'binary_price', 'mixture_price', 'cold_half_proxy_price']
COMPONENTS = ['B0', 'M0', 'Blambda', 'Mlambda']
COUNTERS = {'generated': 'generated_service_count', 'admitted': 'admitted_service_count',
            'success': 'deadline_satisfied_count', 'energy': 'uav_fleet_total_energy',
            'mbs': 'service_offloads_mbs', 'fairness': 'fairness'}


def read(p):
    return json.loads(Path(p).read_text())


def normal(x):
    return json.loads(json.dumps(x, default=lambda z: z.tolist() if hasattr(z, 'tolist') else str(z), allow_nan=False))


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(normal(x), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(normal(x), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def source(a):
    return a.archive / 'output/sf_hlm_learning_feasibility_20261016/assets/releases'


def boot(a, block):
    releases = source(a)
    f = load('stage_b_archive', releases / 'd448ad95f9a3/scripts/analysis/sf_hlm_learning_feasibility.py')
    args = argparse.Namespace(parent_launch=releases / 'd448ad95f9a3/parent_launch.json')
    runtime = f.boot(args, block)
    # Load only the exact numerical modules from the final sealed release.
    final = releases / 'framework_saturated_425716b43b76'
    sys.path.insert(0, str(final))
    from scripts.analysis import sf_hlm_framework_evaluation as evaluation
    fast = evaluation.public_modules(argparse.Namespace(release_root=final))
    core = f.core(argparse.Namespace(release_root=releases / 'd448ad95f9a3'))
    pricepath = a.archive / 'results/sf_hlm_learning_feasibility_20261016/framework_confirmation_20261018/calibration_v1/prices.json'
    prices = read(pricepath)
    return f, runtime, fast, core, prices


def protocol(a):
    p = read(a.output / 'protocol.json')
    if p['runner_sha256'] != sha(__file__): raise ValueError('Runner source changed after freeze')
    for path, h in p['inputs'].items():
        if sha(path) != h: raise ValueError('Frozen input changed: ' + path)
    return p


def initialize(a):
    if (a.output / 'protocol.json').exists(): raise FileExistsError('Immutable run namespace')
    candidates = [19271001, 19271002, 19271003]
    # Scan metadata, not millions of raw timesteps. Includes task keys, training
    # seeds, protocol inputs, original launch commands and seed-specific folders.
    excluded = {'tasks', 'states', 'snapshots', 'dataset', 'dataset_episode', 'canonical',
                '__pycache__', '.git', 'worker_logs', 'job_logs', 'step_records', 'assets',
                'capture', 'evaluate', 'folds', 'logs', 'train_logs', 'figures', 'evidence_saturated_v1'}
    scanned = {}; collisions = []
    for root in [a.archive / 'results', a.archive / 'output', a.archive.parent / 'batch-offload-20260928/pilot']:
        for folder, dirs, files in os.walk(root):
            for s in candidates:
                if str(s) in str(folder): collisions.append({'seed': s, 'path': folder})
            dirs[:] = [d for d in dirs if d not in excluded and not d.startswith(('episode_', 'step_', 'block_', 'seed_', 'attempt_'))
                       and len(Path(folder).relative_to(root).parts)<7]
            for name in files:
                if name not in ('protocol.json', 'selected.json', 'parent_launch.json') and not (('launch' in name or 'manifest' in name or 'seed' in name) and name.endswith('.json')): continue
                path = Path(folder) / name
                if path.stat().st_size > 30_000_000: continue
                text = path.read_text(errors='replace'); scanned[str(path)] = sha(path)
                for s in candidates:
                    if str(s) in text: collisions.append({'seed': s, 'path': str(path)})
    write(a.output / 'seed_audit.json', dict(scanned=scanned, candidates=candidates, collisions=collisions,
          scope='Historical protocol/launch/manifest/seed metadata and folder names; raw step/task bodies excluded',
          purpose='new development workloads only; no new training or confirmation seeds'))
    if collisions: raise ValueError('Candidate seed appears in historical metadata')
    # Verify each actual development model by booting later; freeze all checkpoint
    # files now, together with complete source releases and ancestral protocols.
    roots = [source(a) / 'framework_saturated_425716b43b76', source(a) / 'd448ad95f9a3',
             a.archive / 'extension_release_v2_20261001', a.archive / 'followup_release_20261002',
             a.archive / 'upper_adaptation_release', a.archive / 'learning_improvement_release_20261003_951dea3274',
             a.archive.parent / 'batch-offload-20260928/source_launch_20260929']
    inputs = {}
    for root in roots:
        if not root.exists(): raise FileNotFoundError(root)
        for path in root.rglob('*.py'): inputs[str(path)] = sha(path)
    inputs[str(a.output / 'seed_audit.json')] = sha(a.output / 'seed_audit.json')
    models = {}
    pilot = a.archive.parent / 'batch-offload-20260928/pilot/main/lower_mlp_upper_uncovered_obs'
    for b in BLOCKS:
        training = pilot / f'seed_{b}/train'
        completed = read(training / 'completed.json')
        attempt = training / completed['attempt']; latest = read(attempt / 'latest.json')
        checkpoint = attempt / latest['checkpoint']
        models[str(b)] = str(checkpoint)
        for path in checkpoint.rglob('*'):
            if path.is_file(): inputs[str(path)] = sha(path)
    pricepath = a.archive / 'results/sf_hlm_learning_feasibility_20261016/framework_confirmation_20261018/calibration_v1/prices.json'
    inputs[str(pricepath)] = sha(pricepath)
    prices = read(pricepath); common = {k: prices['prices']['mixture'][k] for k in ('price', 'normalizer')}
    cpus = []; seen = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        t = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((t/'physical_package_id').read_text().strip(), (t/'core_id').read_text().strip())
        if key not in seen: cpus.append(cpu); seen.add(key)
    cpus = cpus[64:64+a.workers]
    if len(cpus) != a.workers: raise ValueError('Insufficient distinct physical cores')
    samples = {}
    for b in BLOCKS:
        for w in candidates:
            for method in ['G', 'mixture_price']:
                key = f'{b}_{w}_{method}'
                rng = random.Random('chapter5_stage_b_20261027_' + key)
                samples[key] = [i*25 + rng.randrange(25) for i in range(40)]
    bundles = [[b,w,s] for b in BLOCKS for w in candidates for s in SCENARIOS]
    random.Random(2026102709).shuffle(bundles)
    p = dict(name='chapter5_stage_b_development_20261027_v1', runner_sha256=sha(__file__), inputs=inputs,
        blocks=BLOCKS, workloads=candidates, models=models, scenarios=SCENARIOS, common_price=common,
        original_prices={k: {q: v[q] for q in ('price', 'normalizer')} for k,v in prices['prices'].items()},
        transfer_methods=TRANSFER, components=COMPONENTS, bundles=bundles, samples=samples, steps=1000,
        primary='S/G from per-block pooled counts, then paired differences; five historical upper blocks',
        independent_n=5, new_training=False, independent_confirmation=False, workers=a.workers, cpus=cpus,
        timing='New regime: up to specified single-thread workers on distinct cores, full native call; logs/snapshots outside timer; not old saturated120 timing',
        sampling='one preselected random state per 25-step stratum, 40 per episode; step0 separate unless already sampled',
        b1_episodes=30, b1_main_states=1200, b3_episodes=300, b2_closed_loop_episodes=60,
        reuse='Mlambda equals original mixture_price; reuse 15 center episodes; 345 unique closed-loop episodes',
        comparisons=['M0-B0','Mlambda-Blambda','Blambda-B0','Mlambda-M0','(Mlambda-Blambda)-(M0-B0)'],
        failure='Retain all outcomes. Infrastructure failures may restart same bundle in new attempt with old logs retained; no seed replacement',
        resource_decision='Complete five-block development first. Full 6000-episode transfer is conditional; do not start automatically',
        smoke={'block':BLOCKS[0], 'workload':candidates[0], 'steps':16, 'all_scenarios':True})
    write(a.output / 'protocol.json', p)
    write(a.output / 'input_inventory.json', dict(models=models, frozen_files=len(inputs), common_price=common,
        formal_raw={'legal_edges':'replay recoverable; original raw stores only public_rows_hash',
            'belief_and_branch_inputs':'replay recoverable', 'all_candidate_allocations':'replay recoverable',
            'rejected_candidates':'replay recoverable', 'per_call_time':'present', 'per_request_outcomes':'present'},
        historical_400_states='available; not a substitute for long-episode distribution',
        b1_gap='long-episode distribution under current G and mixture_price trajectories',
        b2_question='common-price scoring AND within-quota assignment module, not probability formula alone',
        b3_question='zero retuning transfer on five-axis cross; no interaction claim'))
    print(json.dumps({'status':'PREPARED','frozen_files':len(inputs),'common_price':common,'bundles':len(bundles)}), flush=True)


def methods(s):
    return TRANSFER + (['B0','M0','Blambda'] if s == 'center' else [])


def make_policy(rt, method, prices):
    from utils.sf_hlm_priced_runtime import PricedPolicy
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy
    effective = copy.deepcopy(prices)
    kind = 'native_G' if method == 'G' else method
    if method in COMPONENTS:
        module = 'binary' if method.startswith('B') else 'mixture'
        kind = module + '_price'
        effective['prices'][module] = copy.deepcopy(prices['prices']['mixture'])
        if method.endswith('0'): effective['prices'][module]['price'] = 0.
    cls = CacheMatchingPolicy if method == 'cold_half_proxy_price' else PricedPolicy
    return cls(rt, None, None, effective, prices['powers']) if cls is CacheMatchingPolicy else cls(rt, None, effective, prices['powers']), kind


def configure_scenario(p, name):
    import config
    import numpy as np
    baseline = np.asarray(config.UAV_COMPUTING_CAPACITY).copy()
    config.NUM_UES = int(p['scenarios'][name][0])
    config.UAV_COMPUTING_CAPACITY = baseline * p['scenarios'][name][1]
    # Frozen observation and offload slots remain exactly as trained (30/100).
    # Do not recompute config-derived dimensions after changing population.
    return baseline


def taskdir(a,b,w,s,m,smoke=False):
    return a.output / ('smoke' if smoke else 'tasks') / f'{b}_{w}_{s}' / m


def episode(a,p,b,w,s,m, runtime, fast, core, prices, smoke=False):
    import numpy as np
    import config
    from utils.canonical_experiment import exogenous_frame,config_snapshot
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    _,_,runner,models,lower,basis = runtime
    dest = taskdir(a,b,w,s,m,smoke)
    if (dest/'result.json').exists():
        receipt = read(dest/'result.json')
        if receipt['steps_sha256'] != sha(dest/'steps.json.gz'): raise ValueError('Raw receipt mismatch')
        return receipt
    if dest.exists(): raise FileExistsError('Partial episode preserved: '+str(dest))
    dest.mkdir(parents=True)
    env,obs = runner.episode_env(w,0)
    fp = runner.base.policy_fingerprint([models['upper']])
    records=[]; totals={k:0. for k in COUNTERS}; exog=hashlib.sha256(); times=[]
    selected = p['samples'].get(f'{b}_{w}_{m}',[]) if s=='center' else []
    nsteps = p['smoke']['steps'] if smoke else p['steps']
    with fast.FastRuntime() as rt:
        policy,kind=make_policy(rt,m,prices); rt.patch(core,'truncated_search',policy.run)
        def spec(step):
            return dict(kind=kind,cap=0 if m=='G' else 10,seed=int(digest(['stage_b',b,w,step])[:8],16))
        warm_env,warm_obs=pickle.loads(pickle.dumps((env,obs),protocol=5))
        rt.complete_policy(core,warm_env,warm_obs,spec(0),models,None,runner)
        with SoftwareServiceDelay(env,'measured') as hook:
            for step in range(nsteps):
                frame=digest(exogenous_frame(env)); exog.update(frame.encode())
                sample = s=='center' and m in ('G','mixture_price') and (step in selected or step==0)
                observed = pickle.dumps((env,obs),protocol=5) if sample else None
                start=time.monotonic_ns()
                motion,actions,info,rows=rt.complete_policy(core,env,obs,spec(step),models,None,runner)
                end=time.monotonic_ns()
                hook.arm(info['request_ids'],info['policy_seconds'])
                nxt,rewards,metrics=env.step(motion,offloading_actions=actions); outcome=hook.finish()
                runner.base.assert_step(metrics,info)
                assert outcome['success']==int(metrics['deadline_satisfied_count'])
                assert len(outcome['requests'])==len(rows)==len(info['assignment'])
                assert info['policy_seconds'] <= (end-start)/1e9 + 1e-6
                for k,metric in COUNTERS.items():totals[k]+=float(metrics[metric])
                for r in outcome['requests']:
                    assert math.isfinite(r['latency']) and r['latency']>=0
                if m!='G':assert info['diagnostic']['matching_calls']<=10
                entry=dict(step=step,exogenous_hash=frame,metrics=normal(metrics),requests=outcome['requests'],
                    info=normal(info),motion=motion.tolist(),window_ns=[start,end],injected_seconds=outcome['injected_seconds'])
                records.append(entry); times.append(info['policy_seconds'])
                if sample:
                    sd=dest/'samples'/f'{step:04d}'; sd.mkdir(parents=True)
                    (sd/'observed.bin').write_bytes(observed)
                    write(sd/'public.json',dict(block=b,workload=w,source=m,step=step,rows=rows,
                        sample_role='main' if step in selected else 'cold_only',stratum=step//25,
                        design_weight=25 if step in selected else 0,request_type_public=False,
                        protocol_sha256=sha(a.output/'protocol.json'),outcome=entry))
                obs=np.asarray(nxt,np.float32)
                if (step+1)%100==0:write(dest/'progress.json',dict(step=step+1,total=nsteps,unix=time.time()))
    assert fp==runner.base.policy_fingerprint([models['upper']])
    with gzip.open(dest/'steps.json.gz','wt') as stream:json.dump(normal(records),stream,separators=(',',':'),allow_nan=False)
    receipt=dict(status='PASS',block=b,workload=w,scenario=s,method=m,steps=nsteps,totals=totals,
        upper_fingerprint=fp,basis=basis,config=normal(config_snapshot()),exogenous_hash=exog.hexdigest(),
        mean_seconds=float(np.mean(times)),p95_seconds=float(np.quantile(times,.95)),
        steps_sha256=sha(dest/'steps.json.gz'),protocol_sha256=sha(a.output/'protocol.json'),
        samples=len(list((dest/'samples').glob('*'))),cpu=sorted(os.sched_getaffinity(0)))
    write(dest/'result.json',receipt)
    print(json.dumps({'episode':'PASS','b':b,'w':w,'s':s,'m':m,'seconds':sum(times)}),flush=True)
    return receipt


def worker(a):
    p=protocol(a); b,w,s=json.loads(a.bundle); os.sched_setaffinity(0,{a.cpu})
    f,runtime,fast,core,prices=boot(a,b)
    import config
    from utils.canonical_experiment import config_snapshot
    before=normal(config_snapshot()); configure_scenario(p,s); after=normal(config_snapshot())
    changed=[k for k in before if before[k]!=after[k]]
    assert set(changed)<= {'NUM_UES','UAV_COMPUTING_CAPACITY'}, changed
    receipts=[]; order=methods(s).copy(); random.Random(f'{b}_{w}_{s}').shuffle(order)
    for method in order:receipts.append(episode(a,p,b,w,s,method,runtime,fast,core,prices,a.smoke))
    assert len({r['exogenous_hash'] for r in receipts})==1,'Unpaired exogenous workload'
    assert len({r['upper_fingerprint'] for r in receipts})==1
    write(a.output/('smoke_bundles' if a.smoke else 'bundles')/f'{b}_{w}_{s}.json',dict(status='PASS',changed=changed,
        methods=order,observation_slots=config.MAX_ASSOCIATED_UES,offload_slots=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
        fingerprints={r['method']:r['upper_fingerprint'] for r in receipts},exogenous_hash=receipts[0]['exogenous_hash']))


def command(a,mode):
    return [sys.executable,str(Path(__file__).resolve()),mode,'--archive',str(a.archive),'--output',str(a.output)]


def controller(a,smoke=False):
    p=protocol(a)
    if not smoke:
        checks=read(a.output/'smoke_complete.json')
        assert checks['status']=='PASS' and checks['runner_sha256']==sha(__file__)
    bundles=[[BLOCKS[0],p['workloads'][0],s] for s in SCENARIOS] if smoke else p['bundles']
    # Each thread leases its specified unique CPU through a controller-owned queue.
    import queue
    slots=queue.Queue()
    for cpu in p['cpus']:slots.put(cpu)
    def execute(bundle):
        cpu=slots.get()
        try:
            log=a.output/'logs'/('smoke_' if smoke else '')/('_'.join(map(str,bundle))+'.log')
            log.parent.mkdir(parents=True,exist_ok=True)
            cmd=command(a,'worker')+['--bundle',json.dumps(bundle),'--cpu',str(cpu)]+(['--smoke'] if smoke else [])
            with log.open('ab') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
            return bundle
        finally:slots.put(cpu)
    with ThreadPoolExecutor(max_workers=min(p['workers'],len(bundles))) as pool:
        futures=[pool.submit(execute,b) for b in bundles]
        for i,future in enumerate(as_completed(futures)):
            bundle=future.result();write(a.output/('smoke_status.json' if smoke else 'status.json'),dict(status='RUNNING',completed=i+1,total=len(bundles),last=bundle,unix=time.time()))
    write(a.output/('smoke_complete.json' if smoke else 'collection_complete.json'),dict(status='PASS',bundles=len(bundles),runner_sha256=sha(__file__),protocol_sha256=sha(a.output/'protocol.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['init','smoke','run','worker'])
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--bundle');p.add_argument('--cpu',type=int);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific execution only on the experiment server')
    {'init':initialize,'smoke':lambda a:controller(a,True),'run':controller,'worker':worker}[a.mode](a)


if __name__=='__main__':main()
