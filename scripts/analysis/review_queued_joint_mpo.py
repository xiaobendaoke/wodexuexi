"""Independent arithmetic/replay audit of the bounded joint-MPO run.

Server mode replays all saved candidate probabilities and sampled complete
physical episodes. Local mode verifies transported raw counters and statistics.
This is explicit self-review, not independent human/agent review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def metadata(root, trial):
    records = [json.loads(p.read_text()) for p in (trial.out(root)/'episodes').rglob('*.json')]
    assert len(records) == 1608, len(records)
    counts = {kind: sum(r['kind'] == kind for r in records) for kind in ('tail', 'train', 'test')}
    assert counts == dict(tail=264, train=768, test=576), counts
    for r in records:
        c = r['counts']
        assert c['generated'] == c.get('success', 0)+c.get('expired', 0)
        assert sum(r['successes']) == c.get('success', 0) and r['audit']['status'] == 'PASS'
        assert r['slots'] in range(200, 205)
    tests = [r for r in records if r['kind'] == 'test']
    for w in range(11400,11416):
        group = [r for r in tests if r['workload'] == w]
        assert len(group) == 36
        assert len({r['fingerprint'] for r in group}) == len({r['counts']['generated'] for r in group}) == 1
    saved = json.loads((trial.out(root)/'analysis.json').read_text())
    # Recompute without rewriting the scientific analysis file.
    for arm, summaries in saved['summary'].items():
        for b, summary in enumerate(summaries):
            rs = [r for r in tests if r['arm'] == arm and r['b'] == b]
            total = sum(r['counts']['generated'] for r in rs); success = sum(r['counts']['success'] for r in rs)
            assert len(rs) == 16 and summary['generated'] == total and summary['success'] == success
            assert abs(summary['dsr']-success/total) < 1e-12
    import numpy as np
    rng=np.random.default_rng(20261102)
    bykey={(r['arm'],r['b'],r['workload']):r['counts'] for r in tests}
    n=np.array([bykey['joint',0,w]['generated'] for w in range(11400,11416)])
    for arm in trial.METHODS:
        if arm=='joint':continue
        difference=np.array([[bykey['joint',b,w]['success']-bykey[arm,b,w]['success'] for w in range(11400,11416)] for b in range(3)])
        d=saved['comparisons'][arm];np.testing.assert_allclose(difference.sum(1)/n.sum(),d['by_lineage'],atol=1e-12)
        bootstrap=[]
        for _ in range(4000):
            bs=rng.integers(0,3,3);ws=rng.integers(0,16,16)
            bootstrap.append(float(np.mean(difference[bs][:,ws].sum(1)/n[ws].sum())))
        np.testing.assert_allclose(np.quantile(bootstrap,[.025,.975]),d['crossed_bootstrap95'],atol=1e-12)
    protocol=json.loads((trial.out(root)/'protocol.json').read_text())
    for relative in ('utils/queued_joint_mpo.py','marl_models/queued_joint_mpo.py','scripts/analysis/queued_joint_mpo_experiment.py','tests/test_queued_joint_mpo.py'):
        # On server, compare the selected frozen source; locally compare workspace.
        source=trial.REPO_ROOT/relative
        assert trial.sha(source)==protocol['source'][relative],relative
    resources = []
    for p in (trial.validation(root)/'resources').glob('*.json'):
        r = json.loads(p.read_text()); cpu = r['worker_cpu_seconds']+r.get('coordinator_cpu_seconds',0.)
        assert abs(r['equivalent_cpus']-cpu/r['wall_seconds']) < 1e-8
        assert abs(r['utilization']-r['equivalent_cpus']/len(r['allowed_cpus'])) < 1e-10
        if r.get('records'):
            assert abs(sum(x['cpu_seconds'] for x in r['records'])-r['worker_cpu_seconds']) < 1e-6
            events=sorted([(x['start'],1) for x in r['records']]+[(x['end'],-1) for x in r['records']]); active=peak=0
            for _,change in events: active+=change;peak=max(peak,active)
            if 'peak_active' in r: assert peak==r['peak_active']
        else: peak=None
        resources.append(dict(stage=p.stem,wall_seconds=r['wall_seconds'],equivalent_cpus=r['equivalent_cpus'],utilization=r['utilization'],peak_active=peak))
    return dict(status='PASS', episodes=counts, unique_counting=True, common_test_inputs=True, resource_accounting=resources,
                independent_review=False, scope='Raw counter/statistic/resource arithmetic; no local Torch or physical rerun.')


def audit_case(args):
    import numpy as np
    from scripts.analysis import queued_joint_mpo_experiment as t
    from marl_models.queued_joint_mpo import policy
    root,arm,b,version,w = args
    rows=t.read(str(t.out(root)/'experience'/f'train_{arm}_{version-1}'/f'{b}_{w}.pkl.gz'))
    old=t.read(str(t.theta_path(root,arm,b,version-1)))['theta']; new=t.read(str(t.theta_path(root,arm,b,version)))['theta']
    error=0.; kls=[]; masses=[]; coupling=[]; short_ranges=[]; tail_ranges=[]
    for row in rows:
        p=policy(row,old,t.mode_for(arm)); q=policy(row,new,t.mode_for(arm))
        error=max(error,float(np.max(abs(p-row['probabilities'])))); assert error<1e-10
        assert np.isfinite(q).all() and abs(q.sum()-1)<1e-12
        assert len(row['plans'])==len({tuple(sorted(x.items())) for x in row['plans']})
        prior,mass=t.restricted_prior(row['log_masses']); np.testing.assert_allclose(prior,row['prior'],atol=1e-12)
        assert abs(mass-row['retained_mass'])<1e-10 and mass<=1+1e-5
        kls.append(t.divergence(p,q)); masses.append(mass)
        pm=q/q.sum(1,keepdims=True); coupling.append(float(np.max(np.ptp(pm,axis=0))))
        short_ranges.append(float(np.ptp(row['short'])));tail_ranges.append(float(np.ptp(row['tail'])))
    return dict(arm=arm,b=b,version=version,w=w,rows=len(rows),replay_error=error,kl_sum=sum(kls),
                retained_mass_mean=float(np.mean(masses)),conditional_motion_range_mean=float(np.mean(coupling)),
                short_range_mean=float(np.mean(short_ranges)),tail_range_mean=float(np.mean(tail_ranges)))


def physical_replay(args):
    """Redirect writes to an isolated audit subtree; compare every success slot."""
    from scripts.analysis import queued_joint_mpo_experiment as t
    root,b,w,arm,version,kind=args
    destination=t.out(root); validation=t.validation(root)
    original_write, original_dump = t.write,t.dump
    def audit_path(path):
        return validation/'physical_replay'/Path(path).relative_to(destination)
    t.write=lambda path,data: original_write(audit_path(path),data)
    t.dump=lambda path,data: original_dump(audit_path(path),data)
    try: result=t.episode(args)
    finally: t.write,t.dump=original_write,original_dump
    actual=result['record']; expected=json.loads((destination/result['file']).read_text())
    for key in ('counts','fingerprint','parent_sha','slots','successes'): assert actual[key]==expected[key],key
    if kind=='train':
        replay=t.read(str(validation/'physical_replay'/'experience'/f'train_{arm}_{version}'/f'{b}_{w}.pkl.gz'))
        source=t.read(str(destination/'experience'/f'train_{arm}_{version}'/f'{b}_{w}.pkl.gz'))
        import numpy as np
        for a,brow in zip(replay,source):
            for key in ('a','m','prior','short','tail','terminal_features','probabilities'): np.testing.assert_allclose(a[key],brow[key],rtol=0,atol=1e-10)
            assert a['plans']==brow['plans'] and a['actions']==brow['actions']
    return dict(arm=arm,workload=w,slots=actual['slots'],status='PASS')


def candidate_collect(args):
    import copy
    from dataclasses import asdict
    from scripts.analysis import queued_joint_mpo_experiment as t
    from utils.queued_long_value import LongAdapter
    root,b,w=args; model=t.parent(root,b); tape=t.RecoveryScenario(w,'burst','normal',.6)
    e=t.Engine(t.Protocol(),t.START,tape.ue_positions);adapter=LongAdapter(True);states=[]
    tail=t.read(str(t.out(root)/'tail'/f'{b}.pkl.gz'))
    while e.t<=175:
        now=int(e.t);e.add(tape.slot(now));e.admit();obs=e.observe()
        if now in (75,175):
            local=t.LearningAdapter();local.history=copy.deepcopy(adapter.history);local.last_time=adapter.last_time
            row=t.bank(model,obs,local,t.seed(t.REV,'bank',b,w,now))
            t.teacher(row,obs,local.history,t.seed(t.REV,'forecast',b,w,now),tail)
            choices=sorted(set((i,j) for i in (0,len(row['plans'])-1) for j in (0,len(row['actions'])-1)))
            full=dict(time=e.t,positions=e.positions,ues=e.ues,tasks=[asdict(x) for x in sorted(e.tasks.values(),key=lambda z:(z.deadline,z.uid))],protocol=e.p)
            compact=t.Engine.from_observation(full)
            # Parent continuation history already incorporates this root slot.
            later=copy.deepcopy(adapter);later.prepare(obs)
            snap=dict(engine=compact,adapter=later,row=row,choices=choices,b=b,w=w,time=now)
            path=t.validation(root)/'candidate_states'/f'{b}_{w}_{now}.pkl.gz';t.dump(path,snap)
            states.append(dict(b=b,w=w,time=now,choices=choices,path=str(path),sha=t.sha(path)))
        _,plan,motion=model.act(obs,adapter,t.seed(t.REV,'parent',b,w,now));e.assign(plan);e.advance(motion)
    return states


def candidate_label(args):
    import copy
    from scripts.analysis import queued_joint_mpo_experiment as t
    from utils.queued_long_value import remaining_future
    root,b,w,now,i,j,replica=args
    snap=t.read(str(t.validation(root)/'candidate_states'/f'{b}_{w}_{now}.pkl.gz'))
    e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter']);model=t.parent(root,b)
    future=remaining_future(w,e.ues,now,replica)
    tail=t.read(str(t.out(root)/'tail'/f'{b}.pkl.gz'))
    row=snap['row'];e.assign(row['plans'][i]);e.advance(row['actions'][j]);prefix=None;actual_v=None
    while e.t<200 or e.tasks:
        tick=int(e.t)
        if tick==now+3:
            prefix=e.counts['success'];actual_v=t.tail_predict(tail,t.value_features(e.observe(),adapter.history),tick,not e.tasks and tick>=200)
        e.add(copy.deepcopy(future.get(tick,())));e.admit();obs=e.observe()
        _,plan,motion=model.act(obs,adapter,t.seed(t.REV,'candidate_tail',b,w,now,replica,tick))
        e.assign(plan);e.advance(motion)
    e.audit();c=e.counts;assert c['generated']==c['success']+c['expired'] and prefix is not None
    record=dict(b=b,w=w,time=now,i=i,j=j,replica=replica,success=c['success'],generated=c['generated'],expired=c['expired'],
        actual_prefix=prefix,actual_tail=c['success']-prefix,actual_endpoint_value=actual_v,
        predicted_prefix=float(row['short'][i,j]),predicted_tail=float(row['tail'][i,j]))
    t.write(t.out(root)/'candidate_validation'/f'{b}_{w}_{now}_{i}_{j}_{replica}.json',record)
    return record


def candidate_summary(records):
    import numpy as np
    from collections import defaultdict
    groups=defaultdict(list)
    for r in records:groups[r['b'],r['w'],r['time']].append(r)
    true=[];pred=[];tail=[];vp=[];half0=[];half1=[];states=[]
    for key,rows in groups.items():
        actions=sorted({(r['i'],r['j']) for r in rows});ys=[];ps=[];ts=[];vs=[];h0=[];h1=[]
        for i,j in actions:
            rs=[r for r in rows if (r['i'],r['j'])==(i,j)];assert len(rs)==4
            ys.append(np.mean([r['success'] for r in rs]));ps.append(rs[0]['predicted_prefix']+rs[0]['predicted_tail'])
            ts.append(np.mean([r['actual_tail'] for r in rs]));vs.append(np.mean([r['actual_endpoint_value'] for r in rs]))
            h0.append(np.mean([r['success'] for r in rs if r['replica']<402]));h1.append(np.mean([r['success'] for r in rs if r['replica']>=402]))
        for destination,values in ((true,ys),(pred,ps),(tail,ts),(vp,vs),(half0,h0),(half1,h1)):
            v=np.asarray(values);destination.extend(v-v.mean())
        states.append(dict(b=key[0],w=key[1],time=key[2],actions=actions,actual_q=ys,predicted_q=ps,
                           actual_tail=ts,actual_endpoint_value=vs,half0=h0,half1=h1))
    def ev(y,p):return float(1-np.var(np.asarray(y)-np.asarray(p))/max(np.var(y),1e-12))
    return dict(branches=len(records),states=len(groups),full_q_within_state_ev=ev(true,pred),actual_endpoint_tail_within_state_ev=ev(tail,vp),
        split_half_correlation=float(np.corrcoef(half0,half1)[0,1]) if np.std(half0)*np.std(half1)>0 else None,
        scope='A2 continuation, 4 held-out workloads x 3 inherited lineages x 2 times x <=4 actions x 4 conditional futures. Not new independent training or an online oracle.',rows=states)


def audit_pipeline(pool,root,t):
    """As soon as one prefix is ready, fill spare CPUs with its real branches."""
    import time
    from concurrent.futures import wait,FIRST_COMPLETED
    pending={};timings=[];labels=[];replays=[];states=[];started=time.perf_counter()
    def submit(kind,args):pending[pool.submit(t.timed_job,(kind,(args,)))]=(kind,args)
    for b in range(3):
        for w in range(11100,11104):submit('candidate_collect',(str(root),b,w))
    for args in [(str(root),0,11200,'joint',0,'train'),(str(root),1,11300,'flat',1,'train'),(str(root),2,11400,'model_tail',2,'test')]:submit('physical_replay',args)
    while pending:
        ready,_=wait(pending,return_when=FIRST_COMPLETED)
        for future in ready:
            kind,args=pending.pop(future);r=future.result();r['kind']=kind;timings.append(r)
            if not r['ok']:
                t.write(t.validation(root)/'pipeline_failure.json',r);raise RuntimeError(r['error'])
            if kind=='candidate_collect':
                states.extend(r['result'])
                for s in r['result']:
                    for i,j in s['choices']:
                        for replica in range(400,404):submit('candidate_label',(str(root),s['b'],s['w'],s['time'],i,j,replica))
            elif kind=='candidate_label':labels.append(r['result'])
            else:replays.append(r['result'])
            if len(timings)%32==0:print('audit pipeline',len(timings),'finished',len(pending),'pending',flush=True)
    elapsed=time.perf_counter()-started;cpu=sum(r['cpu_seconds'] for r in timings)
    events=sorted([(r['start'],1) for r in timings]+[(r['end'],-1) for r in timings]);active=peak=0
    for _,delta in events:active+=delta;peak=max(peak,active)
    cpus=sorted(os.sched_getaffinity(0))
    t.write(t.validation(root)/'resources'/'audit_pipeline.json',dict(tasks=len(timings),allowed_cpus=cpus,workers=len(cpus),threads=1,
        wall_seconds=elapsed,worker_cpu_seconds=cpu,equivalent_cpus=cpu/elapsed,utilization=cpu/elapsed/len(cpus),peak_active=peak,failures=0,retries=0,records=timings))
    t.write(t.out(root)/'candidate_state_index.json',states)
    t.write(t.out(root)/'candidate_validation.json',candidate_summary(labels))
    return replays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--server',action='store_true');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.source:sys.path.insert(0,str(args.source))
    from scripts.analysis import queued_joint_mpo_experiment as t
    root=args.task_root
    if args.finalize:
        computed=json.loads((t.validation(root)/'audit_compute.json').read_text())
        assert json.loads((t.validation(root)/'protection_before.json').read_text())==t.protect(root)
        result=metadata(root,t);result.update(computed);t.write(t.validation(root)/'server_review.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='resource_accounting'},indent=2));return
    if args.server:
        import numpy as np
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        cpus=sorted(os.sched_getaffinity(0))
        # Register audit jobs before fork, retaining the frozen runner unchanged.
        t.audit_case=audit_case;t.physical_replay=physical_replay;t.candidate_collect=candidate_collect;t.candidate_label=candidate_label
        with ProcessPoolExecutor(max_workers=len(cpus),mp_context=mp.get_context('fork'),initializer=t.init_worker,initargs=(cpus,)) as pool:
            jobs=[(str(root),arm,b,v,w) for arm in t.ARMS for b in range(3) for v in (1,2) for w in range(11200+(v-1)*100,11216+(v-1)*100)]
            stats=t.parallel(pool,root,'audit_replay','audit_case',jobs)
            replay=audit_pipeline(pool,root,t)
        for arm in t.ARMS:
            for b in range(3):
                for version in (1,2):
                    d=json.loads((t.out(root)/'updates'/f'{version}_{arm}_{b}.json').read_text());rs=[r for r in stats if r['arm']==arm and r['b']==b and r['version']==version]
                    assert abs(sum(r['kl_sum'] for r in rs)/sum(r['rows'] for r in rs)-d['final']['kl'])<1e-10
                    ck=t.read(str(t.theta_path(root,arm,b,version)));assert ck['adam_step']==len(d['accepted'])
                    assert np.isfinite(ck['theta']).all() and np.isfinite(ck['adam_m']).all() and np.isfinite(ck['adam_v']).all()
        before=json.loads((t.validation(root)/'protection_before.json').read_text()); assert before==t.protect(root)
        computed=dict(scope='All saved candidate probability/full KL replay, all raw counters, 3 complete physical replays including short-window and tail reconstruction, parent/source protection. Explicit self-review.',
            replay_rows=sum(r['rows'] for r in stats),max_replay_error=max(r['replay_error'] for r in stats),physical_replays=replay,protected_files=len(before))
        t.write(t.validation(root)/'audit_compute.json',computed)
        t.write(t.out(root)/'mechanism_diagnostics.json',dict(rows=stats))
        if not (t.out(root)/'analysis.json').exists():
            print('AUDIT_COMPUTE_COMPLETE; main test still running; finalize after analysis',flush=True);return
        result=metadata(root,t);result.update(computed)
        t.write(t.validation(root)/'server_review.json',result)
    else:
        manifest_path=t.validation(root)/'transport_manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            for name,digest in manifest.items():assert t.sha(root/name)==digest,name
        result=metadata(root,t);t.write(t.validation(root)/'local_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='resource_accounting'},indent=2))


if __name__=='__main__':main()
