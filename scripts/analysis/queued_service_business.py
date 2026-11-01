"""Freeze, run and audit business-scale queue calibration; no RL training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from environment.queued_service import Engine, Protocol
from utils.queued_service_business import BusinessScenario, SLA, PROFILES, BASIS, NAMESPACE
from utils.queued_service_controllers import Controller, METHODS
from scripts.analysis.queued_service_coupling import cooperative_plan, motion_sequence
from scripts.analysis.queued_service_validation import SOURCES as OLD_SOURCES, START, atomic, core_order, init_worker, cpu_sample

TASK = 'chapter5_dynamic_service_20261028'
REVISION = 'business_calibration_20261031'
BIBLIOGRAPHY = f'output/{TASK}/deliverables/queued_business_sources.json'
SOURCES = OLD_SOURCES+('scripts/analysis/queued_service_coupling.py', 'utils/queued_service_business.py',
    'scripts/analysis/queued_service_business.py', 'tests/test_queued_service_business.py', BIBLIOGRAPHY)


def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locations(task): return task/'results'/TASK/REVISION, task/'output'/TASK/'validation'/REVISION


def contract():
    files = {x: sha(REPO_ROOT/x) for x in SOURCES}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def read(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()).decode()) if str(path).endswith('.gz') else json.loads(Path(path).read_text(encoding='utf-8'))


def raw_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), compresslevel=6, mtime=0))
    os.replace(temp, path)


def key(s): return f"{s['profile']}_{s['sla']}_r{s['scale']}_{s['method']}_w{s['workload']}_dt{s['dt']}"
def path_for(out, s): return out/s['phase']/(key(s)+'.json.gz')


def specs(phase, sla=None, scale=None):
    if phase == 'screen':
        return [dict(phase=phase, profile=p, sla=d, scale=r, method=m, workload=w, dt=.1, horizon=200)
                for p, d, r, m, w in itertools.product(PROFILES, SLA, (.75, 1., 1.25), METHODS, range(6))]
    if phase == 'pilot':
        return [dict(phase=phase, profile=p, sla='normal', scale=1., method=m, workload=0, dt=.1, horizon=120)
                for p, m in itertools.product(('burst', 'overload'), METHODS)]
    return [dict(phase=phase, profile=p, sla=sla, scale=scale, method=m, workload=w, dt=dt, horizon=200)
            for p, m, w, dt in itertools.product(PROFILES, METHODS, range(6, 12) if phase == 'holdout' else (6, 7),
                                                   (.1,) if phase == 'holdout' else (.05, .025))]


def protocol():
    return dict(namespace=NAMESPACE, physics=asdict(Protocol()), basis=BASIS, sla=SLA,
        rates=dict(light=10, steady=40, burst='20; 100 during [50,100); then 20', overload=120),
        rate_scales=[.75, 1., 1.25], primary_before_results=dict(sla='normal', scale=1.),
        methods=list(METHODS), horizon=200, screen_jobs=648, holdout_jobs=72, fixed_action_precision_jobs=48,
        workload_blocks=dict(screen=list(range(6)), holdout=list(range(6, 12))),
        intervention=dict(times=[40,75,100,150], profiles=list(PROFILES), workloads=list(range(6,12)), cases=96,
                          initial_routes=['source','cooperative'], planned_motion_slots=6, arrival_batches=6, drain=True),
        selection_rule='All three controls pass burst screens; nearest declared normal/rate1 by abs(rate-1)+0.5*abs(sla_index-1); tie SLA normal/tight/relaxed, rate 1/.75/1.25',
        screen=dict(dsr=[.05,.8], conditional=[.4,.95], service_crossed_min=.1, cpu_wait_per_admitted_min=.05,
                    transfer_nonempty_min=.05, mbs_fraction_max=.8, late_backlog='<=1.2*before+1 task',
                    burst_boundary_cohort_completion_min=.5),
        assumptions='Synthetic second-scale image-event workload; discrete Poisson batches; no real application trace or model profiling; fixed original hardware',
        no_rl=True, automatic_training=False)


def validate(task):
    stream = io.StringIO(); suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_queued_service_business','tests.test_queued_service'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite); _, val = locations(task)
    atomic(val/'tests.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        source_contract=contract()[0], log=stream.getvalue(), independent_review=False, time=now()))
    print(stream.getvalue(), flush=True); return result.wasSuccessful()


def intervention(initial, scenario, obs, motion):
    """Fixed initial state, fixed future tape; actions use no future information."""
    zero = [(0.,0.)]*initial.n
    local = {x['uid']: x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    coop = cooperative_plan(obs); sequence = motion_sequence(initial, motion)
    baseline = initial.counts['success']; start = int(initial.t); results = {}
    for moving, routing in itertools.product((False,True), ('source','cooperative')):
        e = copy.deepcopy(initial); offset = 0; generated = len(e.tasks)
        while offset < 6 or e.tasks:
            if offset > 12: raise RuntimeError('intervention tail failed')
            if offset:
                arrivals = scenario.slot(start+offset) if offset < 6 else (); generated += len(arrivals)
                e.add(arrivals); new = e.admit(); plan = {uid: e.tasks[uid].source for uid in new}
            else: plan = local if routing == 'source' else coop
            e.assign(plan); e.advance(sequence[offset] if moving and offset < 6 else zero); offset += 1
        e.audit(); results[f"{'move' if moving else 'hold'}_{routing}"] = dict(success=e.counts['success']-baseline, generated=generated)
    j = {k:v['success'] for k,v in results.items()}
    a = j['hold_cooperative']-j['hold_source']; b = j['move_cooperative']-j['move_source']
    assert len({v['generated'] for v in results.values()}) == 1
    changed = sum(local[u] != coop[u] for u in local); length = sum(math.hypot(*d) for ds in sequence for d in ds)
    return dict(time=start, cases=results, interaction=a-b, reversal=a*b < 0, changed_assignments=changed,
        motion_length=length, eligible=changed>0 and length>1e-6, arrival_fingerprint=scenario.arrival_fingerprint)


def episode(s, task):
    out, val = locations(task); started = time.monotonic(); cpu = next(iter(os.sched_getaffinity(0)))
    hb = val/'progress'/(s['phase']+'_'+key(s)+'.json')
    def heartbeat(step, stage='running'):
        atomic(hb, dict(stage=stage, step=step, pid=os.getpid(), cpu=cpu, updated=now()))
    heartbeat(0)
    tape = BusinessScenario(s['workload'], s['profile'], s['sla'], s['scale'], s['horizon'])
    from dataclasses import replace
    e = Engine(replace(Protocol(), dt=s['dt']), START, tape.ue_positions); controller = Controller(s['method'])
    rows = []; times = []; probes = []; cohort = []; replay = None
    if s['phase'] == 'precision':
        parent = dict(s, phase='holdout', dt=.1); parent_path = path_for(out, parent)
        old = read(parent_path); assert old['source_contract'] == contract()[0]
        replay = old['rows']; parent_sha = sha(parent_path)
    while int(e.t) < tape.horizon or e.tasks:
        t = int(e.t)
        if t >= tape.horizon+6: raise RuntimeError('main tail failed')
        e.add(tape.slot(t)); e.admit(); decision_start = time.monotonic(); obs = e.observe()
        if replay is None: route, motion = controller.act(obs)
        else:
            record = replay[t] if t < len(replay) else dict(assignment={}, motion=[(0.,0.)]*5)
            route = {int(k):v for k,v in record['assignment'].items()}; motion = record['motion']
            assert set(route) == {z.uid for z in e.tasks.values() if z.source is not None and z.target is None}
        elapsed = time.monotonic()-decision_start
        if s['phase'] == 'holdout' and s['method'] == 'history_queue' and t in (40,75,100,150):
            probes.append(intervention(e, tape, obs, motion))
        commit_start = time.monotonic(); e.assign(route); elapsed += time.monotonic()-commit_start; times.append(elapsed)
        row = e.advance(motion); row.update(assignment=route, motion=motion, decision_seconds=elapsed)
        if e.t == 100: cohort = [z.uid for z in e.tasks.values() if z.source is not None]
        rows.append(row)
        if t%10 == 0: heartbeat(t+1)
    summary = e.summary(); admitted = e.counts['admitted']; finished = list(e.finished.values())
    wait_total = sum(z.time_in['compute_wait'] for z in finished)
    summary.update(cpu_wait_seconds_total=wait_total, cpu_wait_per_admitted=wait_total/max(admitted,1),
        burst_cohort_size=len(cohort), burst_cohort_completed=sum(e.finished[u].stage=='completed' for u in cohort),
        input_cycles_total=sum(z.cycles for z in finished), input_bits_total=sum(z.bits for z in finished),
        expired_served_cycles=sum(z.served_cycles for z in finished if z.stage=='expired'))
    def backlog(r): return sum(r['stages'][x] for x in ('upload','transfer','compute'))
    summary['backlog'] = {k:statistics.mean(backlog(x) for x in rows[a:b]) for k,a,b in
        (('before',30,50),('during',80,100),('late',s['horizon']-20,s['horizon']))}
    summary['tail_seconds'] = e.t-tape.horizon
    result = dict(spec=s, source_contract=contract()[0], scenario=tape.statistics,
        arrival_fingerprint=tape.arrival_fingerprint, fingerprint=tape.fingerprint,
        summary=summary, audit=e.audit(), rows=rows, interventions=probes,
        decision=dict(mean=statistics.mean(times), p95=sorted(times)[int(.95*(len(times)-1))], max=max(times),
                      over_one_second=sum(x>1 for x in times), forecast_calls=controller.forecast_calls,
                      scope='observation construction, action and assignment validation; diagnostic interventions excluded'),
        seconds=time.monotonic()-started, cpu=cpu, pid=os.getpid())
    if replay is not None: result['fixed_action_parent_sha256'] = parent_sha
    path = path_for(out,s)
    if path.exists(): raise FileExistsError(path)
    raw_write(path,result); heartbeat(len(rows),'completed')
    return dict(file=str(path.relative_to(out)), sha256=sha(path), cpu=cpu, seconds=result['seconds'])


def batch(schedule, task, workers):
    out,val = locations(task); digest = contract()[0]; done=[]; pending=[]; failed=[]
    for s in schedule:
        path = path_for(out,s)
        if path.exists():
            d=read(path); assert d['spec']==s and d['source_contract']==digest
            done.append(dict(file=str(path.relative_to(out)),sha256=sha(path),cpu=d['cpu'],resumed=True))
        else: pending.append(s)
    if not pending: return done
    cpus=core_order()[:min(workers,len(pending))]; last=cpu_sample(); peak=0
    with ProcessPoolExecutor(max_workers=len(cpus),initializer=init_worker,initargs=(cpus,val)) as pool:
        futures={pool.submit(episode,s,task):s for s in pending}
        while futures:
            ready,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for f in ready:
                s=futures.pop(f)
                try: done.append(f.result())
                except Exception: failed.append(dict(spec=s,traceback=traceback.format_exc()))
            active=[]
            for p in (val/'progress').glob('*.json'):
                d=read(p)
                if d['stage']=='running': active.append(dict(job=p.stem,**d))
            peak=max(peak,len(active)); current=cpu_sample()
            utilization={cpu:1-(v[1]-last[cpu][1])/max(v[0]-last[cpu][0],1) for cpu,v in current.items()};last=current
            with (val/'resources.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(time=now(),active=len(active),cpu_busy=utilization))+'\n')
            atomic(val/'status.json',dict(stage=schedule[0]['phase'],updated=now(),pid=os.getpid(),completed=len(done),
                total=len(schedule),failed=failed,active=active,peak_active=peak,worker_cpus=cpus,source_contract=digest))
    if failed:
        atomic(val/'failures.json',failed);raise RuntimeError(f'{len(failed)} business jobs failed')
    return done


def aggregate(schedule, task):
    out,_=locations(task); groups={}; fingerprints={}; hashes={}; probes=[]
    for s in schedule:
        path=path_for(out,s);d=read(path);assert d['source_contract']==contract()[0] and d['spec']==s
        c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0)
        assert d['rows'][-1]['active']==0 and d['audit']['status']=='PASS'
        assert sum(x['success'] for x in d['rows'])==c.get('success',0)
        assert sum(d['summary']['expiration_stage'].values())==c.get('expired',0)
        pair=(s['profile'],s['scale'],s['workload'])
        assert pair not in fingerprints or fingerprints[pair]==d['arrival_fingerprint'];fingerprints[pair]=d['arrival_fingerprint']
        group=(s['profile'],s['sla'],s['scale'],s['method'])
        groups.setdefault(group,[]).append(d);hashes[str(path.relative_to(out))]=sha(path)
        probes.extend(dict(spec=s,**x) for x in d['interventions'])
    rows=[]
    for key_, ds in sorted(groups.items()):
        p,sla,scale,m=key_;summaries=[d['summary'] for d in ds];n=len(ds)
        g=sum(x['counts']['generated'] for x in summaries);a=sum(x['counts'].get('admitted',0) for x in summaries)
        success=sum(x['counts'].get('success',0) for x in summaries);duration=sum(x['elapsed'] for x in summaries)
        cohort=sum(x['burst_cohort_size'] for x in summaries);cohort_done=sum(x['burst_cohort_completed'] for x in summaries)
        r=dict(profile=p,sla=sla,scale=scale,method=m,replicates=n,generated=g,admitted=a,success=success,
            dsr=success/g,admission=a/g,conditional=success/max(a,1),
            cpu_wait_per_admitted=sum(x['cpu_wait_seconds_total'] for x in summaries)/max(a,1),
            service_crossed=sum(x['counts'].get('service_crossed',0) for x in summaries)/max(a,1),
            transfer_nonempty=sum(x['nonempty_fraction']['transfer']*x['elapsed'] for x in summaries)/duration,
            mbs_fraction=sum(x['assignment_by_node'].get('5',x['assignment_by_node'].get(5,0)) for x in summaries)/max(a,1),
            backlog={k:statistics.mean(x['backlog'][k] for x in summaries) for k in ('before','during','late')},
            cohort_size=cohort,cohort_completion=cohort_done/cohort if cohort else None,
            expiration_stage={k:sum(x['expiration_stage'].get(k,0) for x in summaries) for k in ('waiting','upload','transfer','compute')},
            cpu_utilization=[sum(x['cpu_utilization'][i]*x['elapsed'] for x in summaries)/duration for i in range(6)],
            decision_mean=sum(d['decision']['mean']*len(d['rows']) for d in ds)/sum(len(d['rows']) for d in ds),
            decision_over_one_second=sum(d['decision']['over_one_second'] for d in ds),
            dsr_by_workload={str(d['spec']['workload']):d['summary']['dsr'] for d in ds})
        assert math.isclose(r['dsr'],r['admission']*r['conditional'],abs_tol=1e-12)
        r['criteria']=dict(dsr=.05<=r['dsr']<=.8,conditional=.4<=r['conditional']<=.95,
            crossed=r['service_crossed']>=.1,cpu_wait=r['cpu_wait_per_admitted']>=.05,
            transfer=r['transfer_nonempty']>=.05,mbs=r['mbs_fraction']<.8,
            recovery=r['backlog']['late']<=1.2*r['backlog']['before']+1.,
            cohort=r['cohort_completion'] is not None and r['cohort_completion']>=.5)
        r['gate']=all(r['criteria'].values());r['strict_recovery']=r['backlog']['late']<=1.2*r['backlog']['before']+1e-12
        rows.append(r)
    return dict(summary=rows,raw_sha256=hashes,interventions=probes,source_contract=contract()[0])


def select(screen, task):
    candidates=[]
    for sla,scale in itertools.product(SLA,(.75,1.,1.25)):
        rows=[x for x in screen['summary'] if x['profile']=='burst' and x['sla']==sla and x['scale']==scale]
        assert len(rows)==3
        candidates.append(dict(sla=sla,scale=scale,passed=all(x['gate'] for x in rows),
            cost=abs(scale-1)+.5*abs(list(SLA).index(sla)-1),
            failures={x['method']:[k for k,v in x['criteria'].items() if not v] for x in rows}))
    eligible=[x for x in candidates if x['passed']]
    eligible.sort(key=lambda x:(x['cost'],('normal','tight','relaxed').index(x['sla']),(1.,.75,1.25).index(x['scale'])))
    chosen=eligible[0] if eligible else dict(sla='normal',scale=1.,passed=False,cost=0.,diagnostic_fallback=True)
    result=dict(status='FROZEN_CALIBRATION_CANDIDATE' if eligible else 'NO_SETTING_PASSED_DIAGNOSTIC_ONLY',
                selected=chosen,candidates=candidates,source_contract=contract()[0],selection_uses_rl=False)
    out,val=locations(task);path=out/'selection.json'
    if path.exists():assert read(path)==result
    else:atomic(path,result)
    atomic(val/'selection_frozen.json',dict(**result,selected_before_new_workloads=True))
    return chosen


def protect(task,val):
    path=val/'protected_before.json'
    if path.exists():return read(path)
    files={}
    for base in (task/'results'/TASK,task/'output'/TASK):
        for p in base.rglob('*'):
            if p.is_file() and REVISION not in p.relative_to(base).parts and p.suffix not in ('.lock','.tmp'):
                files[str(p.relative_to(task))]=sha(p)
    atomic(path,files);return files


def analyze(task):
    out,val=locations(task);screen=aggregate(specs('screen'),task);chosen=read(out/'selection.json')['selected']
    holdout=aggregate(specs('holdout',chosen['sla'],chosen['scale']),task)
    precision=[];precision_hashes={}
    for s in specs('precision',chosen['sla'],chosen['scale']):
        path=path_for(out,s);d=read(path);parent_path=path_for(out,dict(s,phase='holdout',dt=.1));parent=read(parent_path)
        assert d['fixed_action_parent_sha256']==sha(parent_path)
        assert d['source_contract']==contract()[0] and d['arrival_fingerprint']==parent['arrival_fingerprint']
        assert d['summary']['counts']['generated']==parent['summary']['counts']['generated']
        precision.append(dict(spec=s,dsr_change=d['summary']['dsr']-parent['summary']['dsr'],
            success_change=d['summary']['counts'].get('success',0)-parent['summary']['counts'].get('success',0)))
        precision_hashes[str(path.relative_to(out))]=sha(path)
    old=read(val/'protected_before.json');changed=[p for p,h in old.items() if not (task/p).exists() or sha(task/p)!=h]
    probes=holdout['interventions'];valid=[x for x in probes if x['eligible']]
    assert len(probes)==96
    records=[read(path_for(out,s)) for s in specs('screen')]
    result=dict(status='PASS_COMPLETED_BUSINESS_CALIBRATION' if not changed else 'FAIL_OLD_FILES_CHANGED',
        source_contract=contract()[0],screen=screen,holdout=holdout,precision=precision,precision_raw_sha256=precision_hashes,
        max_abs_fixed_action_dsr_change=max(abs(x['dsr_change']) for x in precision),
        selection=read(out/'selection.json'),interventions=dict(total=len(probes),eligible=len(valid),
            nonzero=sum(x['interaction']!=0 for x in valid),reversals=sum(x['reversal'] for x in valid)),
        actual_screen_cpus=sorted({d['cpu'] for d in records}),main_generated_tasks=sum(d['summary']['counts']['generated'] for d in records),
        protected_files=len(old),changed_old_files=changed,completed=now(),no_rl=True,
        limitations='Synthetic batch arrivals and literature-scale task demand; new-workload checks are not independent training repeats; local interventions are not optimized-policy gains')
    atomic(out/'analysis.json',result)
    atomic(val/'completion.json',{k:v for k,v in result.items() if k not in ('screen','holdout','precision','precision_raw_sha256','actual_screen_cpus')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('validate','pilot','run','analyze','package'))
    parser.add_argument('--task',type=Path,default=REPO_ROOT);parser.add_argument('--workers',type=int,default=256)
    parser.add_argument('--output',type=Path);args=parser.parse_args();task=args.task.resolve();out,val=locations(task)
    if args.command=='package':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
            for s in SOURCES:z.write(REPO_ROOT/s,s)
        print(json.dumps(dict(source_contract=contract()[0],zip_sha256=sha(args.output))));return
    if args.command=='validate':raise SystemExit(0 if validate(task) else 1)
    if args.command=='analyze':analyze(task);return
    import fcntl
    val.mkdir(parents=True,exist_ok=True);lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tests=read(val/'tests.json');assert tests['status']=='PASS' and tests['source_contract']==contract()[0]
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    protect(task,val)
    if args.command=='pilot':
        result=batch(specs('pilot'),task,6);atomic(val/'pilot_summary.json',dict(completed=now(),jobs=result));return
    digest,files=contract();proto=json.loads(json.dumps(dict(protocol(),source_contract=digest,source_files=files)))
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==proto
    else:atomic(out/'protocol.json',proto)
    atomic(val/'launch.json',dict(started=now(),pid=os.getpid(),source=str(REPO_ROOT),source_contract=digest,
        workers=args.workers,cpus=core_order(),screen_jobs=648,holdout_jobs=72,precision_jobs=48,no_rl=True))
    batch(specs('screen'),task,args.workers);screen=aggregate(specs('screen'),task);atomic(out/'screen_analysis.json',screen)
    chosen=select(screen,task)
    batch(specs('holdout',chosen['sla'],chosen['scale']),task,args.workers)
    batch(specs('precision',chosen['sla'],chosen['scale']),task,args.workers)
    result=analyze(task);print(json.dumps({k:v for k,v in result.items() if k not in ('screen','holdout','precision','precision_raw_sha256','actual_screen_cpus')}),flush=True)


if __name__=='__main__':main()
