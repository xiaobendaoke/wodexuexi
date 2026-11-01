"""Bounded queue-aware joint PPO pilot, isolated from all frozen protocols."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
import traceback
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from environment.queued_service import Engine, Protocol
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_service_controllers import Controller
from utils.queued_service_learning import LearningAdapter, finite_gae
from scripts.analysis.queued_service_validation import START, atomic, core_order, sha, now
from scripts.analysis.queued_service_business import read, raw_write
from scripts.analysis.queued_service_business_refinement import SOURCES as BUSINESS_SOURCES

TASK = 'chapter5_dynamic_service_20261028'
REVISION = 'queued_joint_short_20261031'
METHODS = ('joint', 'decomposed')
CONTROLS = ('history_queue', 'history_balance', 'rolling_joint')
PROFILES = ('light', 'steady', 'burst', 'overload')
EVALUATION = tuple(range(48, 56))
SOURCES = tuple(dict.fromkeys(BUSINESS_SOURCES+('utils/queued_service_learning.py',
    'marl_models/queued_service_joint.py', 'scripts/analysis/queued_service_joint_short.py',
    'tests/test_queued_service_learning.py')))


def contract():
    files = {name: sha(REPO_ROOT/name) for name in SOURCES}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def protocol():
    digest, files = contract()
    return dict(revision=REVISION, source_contract=digest, source_files=files,
        business_contract='42409b2fd4f63d2ff6a6d1a1a9f2c08008b800758ceee415866cb5425e44dcfb',
        physics=asdict(Protocol()), input_MB=[1.5, 3], cycles_per_bit=[500, 1000], deadline=[2, 3],
        rates=dict(light=7.5, steady=30, burst=[15, 60, 15], overload=90), peak_window=[50, 100], horizon=200,
        methods=list(METHODS), controls=list(CONTROLS), central_team_prototype=True,
        shared_information='current public task telemetry, sensed UE and observed demand history; no future tape',
        difference='joint motion conditions on sampled allocation prefix; decomposed motion sees pre-allocation public queues',
        allocation='EDF task order; six target candidates; legal mask; fixed -2*estimated_seconds logit prior plus learned residual',
        motion='same <=16 safe history-based fleet trajectories; no teacher; initial residual gain .05',
        training=dict(blocks=3, model_seeds=[17001, 17002, 17003], updates=6, complete_episodes_per_update=8,
            training_episodes=288, workload_id='1000+1000*block+8*(round-1)+replica', profile='burst'),
        reward='new on-time completions /20, both branches share finite-episode future reward; full drain; no batch-only reward',
        credit=dict(gamma=1., gae_lambda=.95, terminal_value=0., critic='public pre-action state only'),
        optimizer=dict(lr=.0003, epochs=2, minibatch=128, clip=.2, entropy=.001, grad_clip_per_branch=.5,
            critic_weight=.5, allocation_ratio='whole sampled task prefix probability',
            clipping='separate allocation and motion branches; not a full joint trust-region guarantee',
            actor_kl_skip='sampled prefix-conditional allocation KL plus motion KL mean > .05; critic still updates'),
        evaluation=dict(workloads=list(EVALUATION), profiles=list(PROFILES), checkpoints=[0, 3, 6],
            learned_episodes=576, control_episodes=96, main_decoder='sampled; no best checkpoint selection',
            role='development only; not independent confirmation'),
        gate='exploratory: all three seed burst DSR gains over own initialization positive and mean >=1pp; report strong controls regardless',
        parallel='all allowed logical CPUs, one thread per pinned worker; useful training and evaluation jobs only',
        independent_confirmation=False, automatic_expansion=False)


def locations(task):
    return task/'results'/TASK/REVISION, task/'output'/TASK/'validation'/REVISION


def configure():
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', MKL_THREADING_LAYER='GNU', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
    import torch
    torch.set_num_threads(1)
    return torch


def worker_init(cpus):
    identity = multiprocessing.current_process()._identity
    index = (identity[-1]-1) if identity else 0
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    configure()


def job_id(job):
    return '_'.join(str(job[k]) for k in ('kind', 'method', 'block', 'checkpoint', 'profile', 'workload'))


def result_path(out, job):
    return out/'episodes'/(job_id(job)+'.json.gz')


def model_path(out, method, block, checkpoint):
    return out/'models'/str(block)/method/f'{checkpoint:04}.pt'


def save_model(path, data, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp'); torch.save(data, temp); os.replace(temp, path)


def action_seed(job, step):
    # Matching streams across methods; block repeats are not new workload replicas.
    label = f'{REVISION}/{job["kind"]}/{job["block"]}/{job["workload"]}/{step}'
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'big') % (2**63-1)


def episode(payload):
    task_string, job, digest = payload; task = Path(task_string); out, val = locations(task)
    target = result_path(out, job); training = job['kind'] == 'train'; torch = configure()
    checkpoint = model_path(out, job['method'], job['block'], job['checkpoint']) if job['method'] in METHODS else None
    cp_sha = sha(checkpoint) if checkpoint else None
    experience = task/'.tmp'/TASK/REVISION/'experience'/(job_id(job)+'.pkl.gz')
    progress = val/'progress'/(job_id(job)+'.json')
    def heartbeat(stage, **extra):
        atomic(progress, dict(stage=stage, pid=os.getpid(), cpu=sorted(os.sched_getaffinity(0)), time=time.time(), job=job, **extra))
    if target.exists():
        saved = read(target)
        assert saved['source_contract'] == digest and saved['job'] == job and saved['checkpoint_sha256'] == cp_sha
        if training: assert experience.exists() and sha(experience) == saved['experience_sha256']
        heartbeat('completed', resumed=True)
        return dict(job=job, path=str(target), cpu=saved['cpu'], resumed=True)
    heartbeat('running', step=0); begin = time.perf_counter()
    tape = RecoveryScenario(job['workload'], job['profile'], 'normal', .6 if job['profile'] == 'burst' else .75)
    engine = Engine(Protocol(), START, tape.ue_positions)
    model = None; adapter = None; controller = None
    if checkpoint:
        from marl_models.queued_service_joint import JointPolicy
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['source_contract'] == digest and saved['checkpoint'] == job['checkpoint']
        assert saved['method'] == job['method'] and saved['block'] == job['block']
        model = JointPolicy(job['method']); model.load_state_dict(saved['model']); model.eval(); adapter = LearningAdapter()
    else: controller = Controller(job['method'])
    rows = []; decisions = []; experience_rows = []; cohort = []; route_counts = [0]*6
    while engine.t < 200 or engine.tasks:
        step = int(engine.t); assert step <= 204
        engine.add(tape.slot(step)); engine.admit(); obs = engine.observe()
        start = time.perf_counter(); before = engine.counts['success']
        if model:
            row, plan, motion = model.act(obs, adapter, action_seed(job, step))
        else: plan, motion = controller.act(obs)
        elapsed = time.perf_counter()-start; decisions.append(elapsed)
        for node in plan.values(): route_counts[node] += 1
        engine.assign(plan); result = engine.advance(motion)
        result.update(step=step, decision_seconds=elapsed, assignment=plan, motion=motion)
        rows.append(result)
        if training:
            row.update(reward=(engine.counts['success']-before)/20)
            experience_rows.append(row)
        if engine.t == 100: cohort = [x.uid for x in engine.tasks.values() if x.source is not None]
        if step % 20 == 0: heartbeat('running', step=step)
    audit = engine.audit(); summary = engine.summary()
    summary.update(cpu_wait_seconds_total=sum(x.time_in['compute_wait'] for x in engine.finished.values()),
        burst_cohort_size=len(cohort), burst_cohort_completed=sum(engine.finished[uid].stage == 'completed' for uid in cohort))
    def inventory(row): return sum(row['stages'][k] for k in ('upload', 'transfer', 'compute'))
    summary['backlog'] = {name: statistics.mean(inventory(r) for r in rows[a:b])
        for name, a, b in (('before', 30, 50), ('during', 50, 100), ('late', 180, 200))}
    assert summary['counts']['generated'] == summary['counts'].get('success', 0)+summary['counts'].get('expired', 0)
    experience_sha = None
    if training:
        adv, returns = finite_gae([r['reward'] for r in experience_rows], [r['value'] for r in experience_rows])
        for row, advantage, value in zip(experience_rows, adv, returns):
            row.update(advantage=advantage, returns=value); row.pop('history_weights')
        experience.parent.mkdir(parents=True, exist_ok=True)
        temp = experience.with_name(experience.name+'.tmp')
        temp.write_bytes(gzip.compress(pickle.dumps(experience_rows, protocol=5), compresslevel=1, mtime=0)); os.replace(temp, experience)
        experience_sha = sha(experience)
    raw_write(target, dict(source_contract=digest, job=job, checkpoint_sha256=cp_sha, experience_sha256=experience_sha,
        arrival_fingerprint=tape.arrival_fingerprint, fingerprint=tape.fingerprint, audit=audit, summary=summary,
        rows=rows, route_counts=route_counts, decision=dict(mean=statistics.mean(decisions), maximum=max(decisions),
        over_one_second=sum(x > 1 for x in decisions)), seconds=time.perf_counter()-begin, cpu=sorted(os.sched_getaffinity(0))))
    heartbeat('completed', step=len(rows)); return dict(job=job, path=str(target), cpu=sorted(os.sched_getaffinity(0)), resumed=False)


def evaluation_jobs(method, block=0, checkpoint=6):
    return [dict(kind='eval', method=method, block=block, checkpoint=checkpoint, profile=p, workload=w)
        for p in PROFILES for w in EVALUATION]


def training_jobs(method, block, round_):
    return [dict(kind='train', method=method, block=block, checkpoint=round_-1, profile='burst',
        workload=1000+1000*block+8*(round_-1)+replica) for replica in range(8)]


def validate(task):
    configure(); os.environ['QUEUED_TORCH_TESTS'] = '1'; stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_queued_service_learning', 'tests.test_queued_service',
        'tests.test_queued_service_business', 'tests.test_queued_service_business_refinement'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    _, val = locations(task)
    atomic(val/'tests.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        skipped=len(result.skipped), source_contract=contract()[0], source=str(REPO_ROOT), time=now(), log=stream.getvalue(), independent_review=False))
    print(stream.getvalue(), flush=True); return result.wasSuccessful()


def initialize(task, torch, digest):
    from marl_models.queued_service_joint import JointPolicy
    out, _ = locations(task); learners = {}
    for block in range(3):
        for method in METHODS:
            torch.manual_seed(17001+block); model = JointPolicy(method)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            latest = out/'models'/str(block)/method/'latest.pt'; round_ = 0
            if latest.exists():
                saved = torch.load(latest, map_location='cpu', weights_only=False)
                assert saved['source_contract'] == digest and saved['method'] == method and saved['block'] == block
                model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); round_ = saved['checkpoint']
                torch.set_rng_state(saved['torch_rng'])
                save_model(model_path(out, method, block, round_), saved, torch)
            else:
                saved = dict(source_contract=digest, method=method, block=block, checkpoint=0, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(), record=None)
                save_model(latest, saved, torch); save_model(model_path(out, method, block, 0), saved, torch)
            learners[(method, block)] = dict(model=model, optimizer=optimizer, round=round_, completed=[], rng=saved['torch_rng'])
    return learners


def protected_files(task):
    result = {}
    for root in (task/'results'/TASK, task/'output'/TASK, task/'saved_models'/TASK):
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and REVISION not in path.relative_to(root).parts and path.suffix not in ('.tmp', '.lock'):
                    result[str(path.relative_to(task))] = sha(path)
    return result


def run(task):
    import fcntl
    torch = configure(); from marl_models.queued_service_joint import update
    out, val = locations(task); val.mkdir(parents=True, exist_ok=True)
    lock = (val/'controller.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    definition = protocol(); digest = definition['source_contract']
    tests = read(val/'tests.json'); assert tests['status'] == 'PASS' and tests['skipped'] == 0 and tests['source_contract'] == digest
    if (out/'protocol.json').exists(): assert read(out/'protocol.json') == definition
    else: atomic(out/'protocol.json', definition)
    protected = val/'protected_before.json'
    if not protected.exists(): atomic(protected, protected_files(task))
    learners = initialize(task, torch, digest)
    # Complete real-horizon smoke checks before scheduling the pilot.
    for method in METHODS:
        episode((str(task), dict(kind='smoke', method=method, block=0, checkpoint=0, profile='burst', workload=56), digest))
    atomic(val/'smoke.json', dict(status='PASS_FULL_200_SECOND_DRAINED_EPISODES', methods=list(METHODS), source_contract=digest, time=now()))
    cpus = core_order(); assert len(cpus) == len(os.sched_getaffinity(0))
    queue = []; scheduled = set(); done = set(); pending = {}; used = set(); peak = 0
    def enqueue(jobs, front=False):
        fresh = [job for job in jobs if job_id(job) not in scheduled]
        scheduled.update(job_id(job) for job in fresh)
        if front: queue[:0] = fresh
        else: queue.extend(fresh)
    for (method, block), learner in learners.items():
        if learner['round'] < 6: enqueue(training_jobs(method, block, learner['round']+1), True)
        for cp in (0, 3, 6):
            if cp <= learner['round']: enqueue(evaluation_jobs(method, block, cp))
    for method in CONTROLS: enqueue(evaluation_jobs(method))
    atomic(val/'launch.json', dict(status='RUNNING', pid=os.getpid(), source=str(REPO_ROOT), source_contract=digest,
        started=now(), cpus=cpus, logical_cpus=len(cpus), expected_main_jobs=960, training_episodes=288,
        learned_evaluations=576, controls=96, source_immutable=True, automatic_expansion=False))
    resource_file = val/'resources.jsonl'; last = 0.
    try:
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context('fork'), initializer=worker_init, initargs=(cpus,)) as pool:
            while queue or pending:
                while queue and len(pending) < len(cpus):
                    job = queue.pop(0); future = pool.submit(episode, (str(task), job, digest)); pending[future] = job
                finished, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future); record = future.result(); done.add(job_id(job)); used.update(record['cpu'])
                    if job['kind'] != 'train': continue
                    learner = learners[(job['method'], job['block'])]; learner['completed'].append(job)
                    if len(learner['completed']) != 8: continue
                    batch = []; source_paths = []
                    for item in sorted(learner['completed'], key=job_id):
                        path = task/'.tmp'/TASK/REVISION/'experience'/(job_id(item)+'.pkl.gz')
                        assert sha(path) == read(result_path(out, item))['experience_sha256']
                        batch.extend(pickle.loads(gzip.decompress(path.read_bytes()))); source_paths.append(str(result_path(out, item).relative_to(out)))
                    torch.set_rng_state(learner['rng']); started = time.perf_counter()
                    metrics = update(learner['model'], learner['optimizer'], batch); learner['rng'] = torch.get_rng_state()
                    learner['round'] += 1; cp = learner['round']; method, block = job['method'], job['block']
                    training = dict(source_contract=digest, method=method, block=block, checkpoint=cp, episodes=8,
                        physical_steps=len(batch), experience_files=source_paths, update=metrics, seconds=time.perf_counter()-started, time=now())
                    saved = dict(source_contract=digest, method=method, block=block, checkpoint=cp, model=learner['model'].state_dict(),
                        optimizer=learner['optimizer'].state_dict(), torch_rng=learner['rng'], record=training)
                    save_model(model_path(out, method, block, cp), saved, torch)
                    save_model(out/'models'/str(block)/method/'latest.pt', saved, torch)
                    atomic(out/'training'/str(block)/method/f'{cp:04}.json', training)
                    print(json.dumps(dict(stage='update_completed', method=method, block=block, checkpoint=cp, **metrics)), flush=True)
                    learner['completed'] = []
                    if cp < 6: enqueue(training_jobs(method, block, cp+1), True)
                    if cp in (3, 6): enqueue(evaluation_jobs(method, block, cp))
                if time.time()-last >= 5 or not pending:
                    active = []
                    for path in (val/'progress').glob('*.json'):
                        r = read(path)
                        if r['stage'] == 'running' and time.time()-r['time'] < 120: active.append(dict(pid=r['pid'], cpu=r['cpu'], job=r['job'], step=r.get('step')))
                    peak = max(peak, len(active)); sample = dict(time=now(), active=len(active), submitted=len(pending), completed=len(done), queued=len(queue))
                    with resource_file.open('a') as handle: handle.write(json.dumps(sample)+'\n')
                    atomic(val/'status.json', dict(status='RUNNING', source_contract=digest, updated=now(), **sample,
                        active_jobs=active, rounds={f'{m}/{b}': x['round'] for (m, b), x in learners.items()}, used_cpus=sorted(used), peak_active=peak))
                    last = time.time()
        assert all(x['round'] == 6 for x in learners.values())
        result = analyze(task)
        old = read(protected); changed = [name for name, h in old.items() if not (task/name).exists() or sha(task/name) != h]
        completion = dict(status='PASS_SHORT_TRIAL_COMPLETED' if not changed else 'FAIL_PROTECTED_FILES',
            source_contract=digest, completed=now(), main_jobs=960, used_cpus=sorted(used), peak_active=peak,
            learning_gate=result['learning_gate'], protected_files=len(old), changed_old_files=changed, automatic_expansion=False)
        atomic(val/'completion.json', completion); atomic(val/'status.json', completion)
    except BaseException:
        atomic(val/'status.json', dict(status='FAILED', source_contract=digest, updated=now(), error=traceback.format_exc(), automatic_expansion=False))
        raise


def aggregate(records):
    g = sum(x['summary']['counts']['generated'] for x in records)
    a = sum(x['summary']['counts'].get('admitted', 0) for x in records)
    s = sum(x['summary']['counts'].get('success', 0) for x in records)
    cohort = sum(x['summary']['burst_cohort_size'] for x in records)
    return dict(episodes=len(records), generated=g, admitted=a, success=s, dsr=s/g, admission=a/g, conditional=s/max(a, 1),
        cpu_wait_per_admitted=sum(x['summary']['cpu_wait_seconds_total'] for x in records)/max(a, 1),
        service_crossed=sum(x['summary']['counts'].get('service_crossed', 0) for x in records)/max(a, 1),
        cohort_size=cohort, cohort_completion=sum(x['summary']['burst_cohort_completed'] for x in records)/cohort if cohort else None,
        mean_decision_seconds=statistics.mean(x['decision']['mean'] for x in records),
        decisions_over_one_second=sum(x['decision']['over_one_second'] for x in records))


def analyze(task):
    import math
    out, val = locations(task); digest = contract()[0]; groups = defaultdict(list); hashes = {}; tapes = {}; updates = {}
    jobs = [j for m in METHODS for b in range(3) for cp in (0, 3, 6) for j in evaluation_jobs(m, b, cp)]
    jobs += [j for m in CONTROLS for j in evaluation_jobs(m)]
    jobs += [j for m in METHODS for b in range(3) for round_ in range(1, 7) for j in training_jobs(m, b, round_)]
    assert len(jobs) == 960 and len({job_id(j) for j in jobs}) == 960
    for job in jobs:
        path = result_path(out, job); d = read(path); assert d['source_contract'] == digest and d['job'] == job and d['audit']['status'] == 'PASS'
        c = d['summary']['counts']; assert c['generated'] == c.get('success', 0)+c.get('expired', 0)
        assert d['rows'][-1]['active'] == 0 and sum(x['success'] for x in d['rows']) == c.get('success', 0)
        assert sum(d['summary']['expiration_stage'].values()) == c.get('expired', 0)
        key = (job['profile'], job['workload']); assert key not in tapes or tapes[key] == d['arrival_fingerprint']; tapes[key] = d['arrival_fingerprint']
        if job['method'] in METHODS: assert d['checkpoint_sha256'] == sha(model_path(out, job['method'], job['block'], job['checkpoint']))
        hashes[str(path.relative_to(out))] = sha(path)
        if job['kind'] == 'eval': groups[(job['profile'], job['method'], job['block'], job['checkpoint'])].append(d)
    summary = []
    for (profile, method, block, cp), records in sorted(groups.items()):
        r = dict(profile=profile, method=method, block=block, checkpoint=cp, **aggregate(records))
        assert math.isclose(r['dsr'], r['admission']*r['conditional'], abs_tol=1e-12); summary.append(r)
    for m in METHODS:
        for b in range(3):
            ds = [read(out/'training'/str(b)/m/f'{cp:04}.json') for cp in range(1, 7)]
            assert all(d['source_contract'] == digest and d['checkpoint'] == i for i, d in enumerate(ds, 1))
            updates[f'{m}/{b}'] = dict(rounds=6, allocation_parameter_change=sum(d['update']['allocation_parameter_change'] for d in ds),
                motion_parameter_change=sum(d['update']['motion_parameter_change'] for d in ds), actor_steps=sum(d['update']['actor_steps'] for d in ds),
                max_joint_conditional_kl=max(d['update']['joint_conditional_kl'] for d in ds))
    def lookup(method, block, checkpoint):
        return next(r for r in summary if r['profile'] == 'burst' and r['method'] == method and r['block'] == block and r['checkpoint'] == checkpoint)
    gains = {m: [lookup(m, b, 6)['dsr']-lookup(m, b, 0)['dsr'] for b in range(3)] for m in METHODS}
    comparison = [lookup('joint', b, 6)['dsr']-lookup('decomposed', b, 6)['dsr'] for b in range(3)]
    gate = {m: all(v > 0 for v in xs) and statistics.mean(xs) >= .01 for m, xs in gains.items()}
    result = dict(status='PASS_RAW_EXPERIMENT_AUDIT', source_contract=digest, summary=summary, updates=updates,
        gain_over_initialization=gains, joint_minus_decomposed=comparison, learning_gate=gate, raw_sha256=hashes,
        completed=now(), independent_confirmation=False, automatic_expansion=False,
        limitations='Central team residual PPO pilot; 3 initializations; 8 shared development workloads; no significance or deployment claim')
    atomic(out/'analysis.json', result)
    import csv
    deliver = task/'output'/TASK/'deliverables'; deliver.mkdir(parents=True, exist_ok=True)
    with (deliver/'queued_joint_short_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ['# 跨时隙联合PPO短试', '', f'完成时间：{now()}；本次为开发短试，不是独立确认。', '',
        '固定2–3秒期限、15→60→15/s突发和原硬件。集中式团队策略依次分配任务并选择可行轨迹，两分支共用未来完成回报。分配具有明确的队列时延先验，未使用辅助教师。', '',
        '| 方法 | 初始化突发DSR | 第3次更新 | 第6次更新 | 相对初始化变化 | 探索门槛 |', '|---|---:|---:|---:|---:|---|']
    for m in METHODS:
        means = [statistics.mean(lookup(m, b, cp)['dsr'] for b in range(3)) for cp in (0, 3, 6)]
        lines.append(f'| {m} | {100*means[0]:.2f}% | {100*means[1]:.2f}% | {100*means[2]:.2f}% | {100*statistics.mean(gains[m]):+.2f}个百分点 | {gate[m]} |')
    lines += ['', f'联合减分解的三个初始化种子差值（百分点）：{[round(100*v, 3) for v in comparison]}。', '',
        '| 非学习强对照 | 突发DSR |', '|---|---:|']
    for m in CONTROLS: lines.append(f"| {m} | {100*lookup(m, 0, 6)['dsr']:.2f}% |")
    lines += ['', '3个初始化种子为训练重复，不能把8个工作负载或请求当作独立训练重复。末检查点是主结果，未挑最好检查点；其他负载档、接纳后质量、排队、计算成本及分支变化见逐格CSV和原始分析。', '',
        '分支clip不是完整联合信任域保证；KL为采样前缀上的条件KL估计。超时未施加部署回退，不能认定实时可用。未自动扩训或改写业务参数。显式自审，不冒称独立审查。', '',
        '- [逐格CSV](queued_joint_short_summary.csv)',
        f'- [原始分析](../../../results/{TASK}/{REVISION}/analysis.json)',
        f'- [冻结协议](../../../results/{TASK}/{REVISION}/protocol.json)',
        f'- [完成验收](../validation/{REVISION}/completion.json)',
        f'- [服务器测试](../validation/{REVISION}/tests.json)', '', f'科学源码合同：`{digest}`。']
    (deliver/'queued_joint_short.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'package', 'run', 'analyze', 'protocol'))
    parser.add_argument('--task', type=Path, default=REPO_ROOT)
    parser.add_argument('--archive', type=Path, default=REPO_ROOT/'.tmp/queued_joint_short/release.zip')
    args = parser.parse_args(); task = args.task.resolve()
    if args.command == 'protocol': print(json.dumps(protocol(), ensure_ascii=False, indent=2))
    elif args.command == 'package':
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCES: archive.write(REPO_ROOT/name, name)
        print(json.dumps(dict(source_contract=contract()[0], archive_sha256=sha(args.archive), files=len(SOURCES))))
    elif args.command == 'validate': raise SystemExit(0 if validate(task) else 1)
    elif args.command == 'analyze': analyze(task)
    else: run(task)


if __name__ == '__main__': main()
