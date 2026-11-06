"""Bounded closed-loop development experiment and locked formal-test runner.

All physical/Torch work is POSIX-server-only. Preparing/finalizing configuration
never instantiates a formal-test scenario. No automatic full experiment launch.
"""
import argparse,copy,gzip,hashlib,heapq,json,multiprocessing as mp,os,pickle,sys,time
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_joint_mpo_experiment as t
from utils.queued_closed_loop import (ARMS,BASELINES,design,validate_job,relative_scores,
    pack_rows,packed_statistics,batch_improve,paired_summary,select_method)
from marl_models.queued_joint_mpo import policy,transactional_step
TASK=t.TASK
REVISIONS={'pilot':'queued_closed_loop_pilot_20261103','formal':'queued_closed_loop_formal_20261103'}
SCORER_REV='queued_root_advantage_20261103'

def out(root,phase):return Path(root)/'results'/TASK/REVISIONS[phase]
def val(root,phase):return Path(root)/'output'/TASK/'validation'/REVISIONS[phase]
def write(path,data):t.write(path,data)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def theta_path(root,phase,arm,run,version):return out(root,phase)/'models'/arm/str(run)/f'{version}.pkl.gz'
def mode(arm):return 'decomposed' if arm.endswith('decomposed') else 'joint'

def source_hashes():
    folders=('environment','utils','marl_models','scripts/analysis','tests')
    return {str(p.relative_to(REPO_ROOT)).replace('\\','/'):t.sha(p) for folder in folders for p in (REPO_ROOT/folder).rglob('*.py')}

def protected(root):
    paths=[]
    for rev in (t.REV,SCORER_REV,'queued_candidate_reliability_20261103'):
        for p in (Path(root)/'results'/TASK/rev).rglob('*'):
            if p.is_file():paths.append(p)
    paths += [t.parent_path(root,b) for b in range(3)]
    return {str(p.relative_to(root)):t.sha(p) for p in sorted(set(paths))}

def prepare(root):
    root=Path(root).resolve()
    for phase in ('pilot','formal'):
        p=out(root,phase)/'reservation.json';record=design(phase)
        if p.exists():assert load(p)==record,'existing reservation differs'
        else:write(p,record)
    report=dict(status='RESERVED',formal_outcomes_generated=False,formal_outcomes_read=False,
        formal_test=list(design('formal')['evaluation_workloads']),pilot_episodes=1080,smoke_episodes_limit=2,
        no_automatic_formal_execution=True,selection_uses='pilot training and development only')
    write(val(root,'pilot')/'split_reservation.json',report)
    return report

@lru_cache(maxsize=4)
def scorer(root):
    p=Path(root)/'results'/TASK/SCORER_REV/'models.json'
    assert t.sha(p)=='23ef618e1379697ba6d69088964d9be97a2709f1c6f8e450aa1ab576e42179c1'
    return load(p)

@lru_cache(maxsize=8)
def cached(path):return t.read(path)

def score_bank(row,obs,history,forecast_seed,root,parent,teacher):
    arrivals=t.arrival_prediction(obs,history,forecast_seed,3)
    shape=(len(row['plans']),len(row['actions']));p=np.zeros(shape);x=np.zeros(shape+(99,));empty=np.zeros(shape,bool)
    for i,plan in enumerate(row['plans']):
        for j,motion in enumerate(row['actions']):
            z=t.forecast_window(obs,history,plan,motion,arrivals,3)
            p[i,j]=z['success'];x[i,j]=z['terminal'];empty[i,j]=z['terminal_empty']
    if teacher=='short':q=p-p.mean()
    elif teacher in ('relative','shuffled'):
        q=relative_scores(scorer(str(root)),x,p,empty,'root_centered' if teacher=='relative' else 'root_shuffled')
    elif teacher=='old':
        tail=cached(str(t.out(root)/'tail'/f'{parent}.pkl.gz'))
        q=p+np.array([[t.tail_predict(tail,x[i,j],obs['time']+3,bool(empty[i,j])) for j in range(shape[1])] for i in range(shape[0])]);q-=q.mean()
    else:raise ValueError('unknown public teacher')
    if not np.isfinite(q).all():raise AssertionError('nonfinite public score')
    row.update(q=q,short=p,terminal_features=x,terminal_empty=empty)
    return row

def teacher_for(arm):
    if arm in ('plan_short','short_joint','short_decomposed'):return 'short'
    if arm in ('plan_relative','relative_joint','relative_decomposed'):return 'relative'
    if arm=='shuffled_joint':return 'shuffled'
    if arm=='plan_old':return 'old'
    raise ValueError('method has no teacher')

def episode(root,phase,run_id,parent,workload,replica,arm,version,kind,smoke=False):
    spec=design(phase)
    if smoke:
        if workload!=909999 or kind not in ('training','evaluation'):raise ValueError('invalid smoke namespace')
    else:validate_job(spec,kind,run_id,workload)
    if phase=='formal' and not (val(root,phase)/'execution_authorized.json').exists():raise PermissionError('formal execution is locked')
    model=t.parent(root,parent);tape=t.RecoveryScenario(workload,'burst','normal',.6)
    e=t.Engine(t.Protocol(),t.START,tape.ue_positions)
    if arm=='parent':
        from utils.queued_long_value import LongAdapter
        adapter=LongAdapter(True)
    else:adapter=t.LearningAdapter()
    theta=None
    if arm in ARMS and version>0:theta=cached(str(theta_path(root,phase,arm,run_id,version)))['theta']
    samples=[];decisions=[];successes=[];flight=0.;sizes=[];initial_obs=None
    while e.t<200 or e.tasks:
        now=int(e.t);assert now<=204
        e.add(tape.slot(now));e.admit();obs=e.observe();begin=time.perf_counter()
        if initial_obs is None:initial_obs=value_fingerprint(obs)
        if arm=='parent':
            _,plan,motion=model.act(obs,adapter,t.seed(REVISIONS[phase],'parent',parent,workload,replica,now))
        else:
            row=t.bank(model,obs,adapter,t.seed(REVISIONS[phase],'bank',parent,workload,replica,now))
            local=np.zeros(row['a'].shape[-1]+row['m'].shape[-1]) if theta is None else theta
            prob=policy(row,local,mode(arm));sizes.append(prob.size)
            if arm.startswith('plan_') or (kind=='training' and now in t.TIMES):
                score_bank(row,obs,adapter.history,t.seed(REVISIONS[phase],'forecast',parent,workload,replica,now),root,parent,teacher_for(arm))
            if arm.startswith('plan_'):
                prob=prob*(row['q']>=row['q'].max()-1e-12);prob/=prob.sum()
            rng=np.random.default_rng(t.seed(REVISIONS[phase],'action',parent,workload,replica,now))
            choice=int(rng.choice(prob.size,p=prob.ravel()));i,j=np.unravel_index(choice,prob.shape)
            plan,motion=row['plans'][i],row['actions'][j]
            if kind=='training' and now in t.TIMES:
                row.update(time=now,parent=parent,workload=workload,chosen=[int(i),int(j)],old_prob=prob)
                samples.append(row)
        decisions.append(time.perf_counter()-begin);flight+=sum(float(np.hypot(*v)) for v in motion)
        e.assign(plan);transition=e.advance(motion);successes.append(transition['success'])
    audit=e.audit();counts=dict(e.counts)
    assert not e.tasks and counts['generated']==counts.get('success',0)+counts.get('expired',0)
    assert sum(successes)==counts.get('success',0)
    record=dict(phase=phase,run=run_id,parent=parent,workload=workload,replica=replica,arm=arm,version=version,kind=kind,
        counts=counts,audit=audit,metrics=e.summary(),fingerprint=tape.fingerprint,initial_fingerprint=initial_obs,
        parent_sha=t.sha(t.parent_path(root,parent)),decision_mean=float(np.mean(decisions)),decision_max=max(decisions),
        over_second=sum(x>1 for x in decisions),slots=len(decisions),successes=successes,training_rows=len(samples),
        flight_distance_m=flight,candidate_count_mean=float(np.mean(sizes)) if sizes else None,
        joule_energy=None,energy_note='Frozen queue engine has no energy accounting; flight distance is not converted to joules.')
    base=val(root,phase)/'smoke' if smoke else out(root,phase)
    path=base/'episodes'/f'{kind}_{arm}_{version}'/f'{run_id}_{parent}_{workload}_{replica}.json'
    if path.exists():raise FileExistsError('refuse episode overwrite')
    write(path,record)
    if samples:t.dump(base/'experience'/arm/f'{run_id}_{version}_{workload}.pkl.gz',samples)
    return dict(file=str(path.relative_to(root)),record={k:record[k] for k in ('kind','arm','run','parent','workload','replica','training_rows')})

def value_fingerprint(obs):
    z=dict(time=obs['time'],positions=obs['positions'],ues=obs['ues'],tasks=obs['tasks'])
    return hashlib.sha256(pickle.dumps(z,5)).hexdigest()

def reduce_fit(path,theta,previous):return packed_statistics(cached(path),np.asarray(theta),np.asarray(previous))

def initialize_worker(cpus):
    os.sched_setaffinity(0,{cpus[(mp.current_process()._identity[-1]-1)%len(cpus)]})

def timed(payload):
    key,kind,args=payload;start=time.time();cpu=time.process_time()
    try:
        result=episode(*args) if kind=='episode' else reduce_fit(*args)
        return dict(key=key,kind=kind,ok=True,result=result,start=start,end=time.time(),cpu_seconds=time.process_time()-cpu,
            pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),threads=len(os.listdir('/proc/self/task')))
    except Exception:
        import traceback
        return dict(key=key,kind=kind,ok=False,error=traceback.format_exc(),start=start,end=time.time(),cpu_seconds=time.process_time()-cpu,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)))

class Pipeline:
    """Bounded priority queue: fitting/training first; real evaluations fill gaps."""
    def __init__(self,root,phase,cpus):
        self.root=root;self.phase=phase;self.cpus=cpus;self.serial=0;self.pending=[];self.active={};self.done={};self.records=[]
        self.wall=time.perf_counter();self.cpu=time.process_time();self.last=0.;self.pool=ProcessPoolExecutor(max_workers=len(cpus),initializer=initialize_worker,initargs=(cpus,))
    def add(self,key,kind,args,priority):
        if key in self.done or key in self.active.values() or any(x[2][0]==key for x in self.pending):raise ValueError('duplicate scheduled task')
        self.serial+=1;heapq.heappush(self.pending,(priority,self.serial,(key,kind,args)))
        return key
    def fill(self):
        while self.pending and len(self.active)<len(self.cpus):
            _,_,payload=heapq.heappop(self.pending);self.active[self.pool.submit(timed,payload)]=payload[0]
    def until(self,keys):
        target=set(keys)
        while not target.issubset(self.done):
            if time.perf_counter()-self.wall>(2700 if self.phase=='pilot' else 21600):raise TimeoutError('dispatch wall budget exceeded; no automatic continuation')
            self.fill()
            if not self.active:raise AssertionError('missing dependency task')
            ready,_=wait(self.active,timeout=2,return_when=FIRST_COMPLETED)
            for future in ready:
                key=self.active.pop(future);r=future.result();self.records.append({k:v for k,v in r.items() if k!='result'})
                if not r['ok']:raise RuntimeError(r['error'])
                self.done[key]=r['result']
            if time.perf_counter()-self.last>15:
                self.last=time.perf_counter();self.live()
        return [self.done[k] for k in keys]
    def live(self):
        z=dict(epoch=time.time(),completed=len(self.done),active=len(self.active),pending=len(self.pending),wall_seconds=time.perf_counter()-self.wall,
            completed_worker_cpu_seconds=sum(r['cpu_seconds'] for r in self.records))
        write(val(self.root,self.phase)/'live.json',z);print('pipeline',z,flush=True)
    def finish(self):
        self.until([x[2][0] for x in self.pending]+list(self.active.values()));self.pool.shutdown(wait=True)
        elapsed=time.perf_counter()-self.wall;own=time.process_time()-self.cpu;used=sum(r['cpu_seconds'] for r in self.records)
        active=peak=0
        for _,change in sorted([(x['start'],1) for x in self.records]+[(x['end'],-1) for x in self.records]):active+=change;peak=max(peak,active)
        report=dict(status='COMPLETE',tasks=len(self.records),wall_seconds=elapsed,worker_cpu_seconds=used,coordinator_cpu_seconds=own,
            equivalent_cpus=(used+own)/elapsed,utilization=(used+own)/elapsed/len(self.cpus),allowed_cpus=self.cpus,workers=len(self.cpus),peak_active=peak,
            failures=sum(not r['ok'] for r in self.records),retries=0,records=self.records,
            scheduling='At most one active job per allowed CPU; high-priority cached vectorized state shards for fitting, dependency-ready training, then genuine evaluation episodes. No duplicated physics or padding work.')
        write(val(self.root,self.phase)/'resources.json',report);return report


def fit_round(pipe,root,phase,version):
    spec=design(phase);sessions=[]
    for arm in ARMS:
        for run in spec['runs']:
            rows=[]
            for w in run['training'][version-1]:rows+=cached(str(out(root,phase)/'experience'/arm/f'{run["id"]}_{version-1}_{w}.pkl.gz'))
            previous=cached(str(theta_path(root,phase,arm,run['id'],version-1)))
            theta=np.asarray(previous['theta']).copy();packed=pack_rows(rows,theta,mode(arm))
            for r,pred in zip(rows,packed['old']):
                k,m=r['m'].shape[:2];assert np.max(np.abs(pred[:k,:m]-r['old_prob']))<1e-12
            target,estep=batch_improve(packed['old'],packed['q'],spec['estep_kl']);packed['target']=target
            paths=[]
            for start in range(0,len(rows),8):
                path=out(root,phase)/'fit_shards'/f'{version}_{arm}_{run["id"]}_{start}.pkl.gz'
                t.dump(path,{k:a[start:start+8] for k,a in packed.items()});paths.append(str(path))
            state=previous.get('optimizer',dict(m=np.zeros_like(theta),v=np.zeros_like(theta),step=0))
            sessions.append(dict(arm=arm,run=run['id'],theta=theta,initial=theta.copy(),state=state,paths=paths,estep=estep,accepted=[],rejected=0,n=len(rows)))
    counter=0
    def evaluate(candidates,previous,indices=None):
        nonlocal counter
        counter+=1;keys=[];owners=[]
        for i in range(len(sessions)) if indices is None else indices:
            for j,path in enumerate(sessions[i]['paths']):
                key=f'fit:{version}:{counter}:{i}:{j}';pipe.add(key,'fit',(path,candidates[i],previous[i]),0);keys.append(key);owners.append(i)
        results=pipe.until(keys);sums={}
        for i,d in zip(owners,results):
            if i not in sums:sums[i]={k:np.asarray(v).copy() if k=='gradient' else v for k,v in d.items()}
            else:
                for k,v in d.items():sums[i][k]+=v
        for d in sums.values():
            for k in d:
                if k!='n':d[k]/=d['n']
        return sums
    current=evaluate([s['theta'] for s in sessions],[s['theta'] for s in sessions])
    for step in range(spec['fit_steps']):
        original=[s['theta'].copy() for s in sessions];proposals=[]
        for i,s in enumerate(sessions):
            state=copy.deepcopy(s['state']);state['step']+=1;g=current[i]['gradient']
            state['m']=.9*state['m']+.1*g;state['v']=.999*state['v']+.001*g*g
            direction=state['m']/(1-.9**state['step'])/(np.sqrt(state['v']/(1-.999**state['step']))+1e-8)
            proposals.append((state,direction))
        pending=set(range(len(sessions)))
        for scale in (8.,4.,2.,1.,.5,.25,.125,.0625,.03125,.015625):
            candidates=[original[i]-.03*scale*proposals[i][1] if i in pending else sessions[i]['theta'] for i in range(len(sessions))]
            checks=evaluate(candidates,original,sorted(pending))
            for i in sorted(pending.copy()):
                d=checks[i]
                if np.isfinite(d['loss']) and d['loss']<=current[i]['loss']+1e-12 and d['kl']<=spec['cumulative_kl']+1e-10 and d['delta']<=spec['per_step_kl']+1e-10:
                    sessions[i]['theta']=candidates[i];sessions[i]['state']=proposals[i][0]
                    sessions[i]['accepted'].append(dict(step=step,scale=scale,kl=d['kl'],delta=d['delta'],loss=d['loss']))
                    pending.remove(i)
            if not pending:break
        for i in pending:sessions[i]['rejected']+=1
        current=evaluate([s['theta'] for s in sessions],[s['theta'] for s in sessions])
        print('fit',phase,version,step,'accepted',len(sessions)-len(pending),'/',len(sessions),flush=True)
    for i,s in enumerate(sessions):
        t.dump(theta_path(root,phase,s['arm'],s['run'],version),dict(theta=s['theta'],optimizer=s['state']))
        cut=len(s['theta'])-44
        record=dict(arm=s['arm'],run=s['run'],version=version,rows=s['n'],estep=s['estep'],accepted=s['accepted'],rejected=s['rejected'],
            final={k:v for k,v in current[i].items() if k!='gradient'},allocation_change=float(np.linalg.norm((s['theta']-s['initial'])[:cut])),
            motion_change=float(np.linalg.norm((s['theta']-s['initial'])[cut:])),optimizer_step=s['state']['step'])
        write(out(root,phase)/'updates'/f'{version}_{s["arm"]}_{s["run"]}.json',record)

def evaluation_jobs(pipe,root,phase,baselines,version):
    spec=design(phase);keys=[]
    entries=[(-1,b) for b in range(3)] if baselines else [(r['id'],r['parent']) for r in spec['runs']]
    for arm in BASELINES if baselines else ARMS:
        for rid,b in entries:
            for w in spec['evaluation_workloads']:
                for rep in spec['action_replicas']:
                    key=f'eval:{arm}:{rid}:{b}:{w}:{rep}';pipe.add(key,'episode',(str(root),phase,rid,b,w,rep,arm,version,'evaluation'),3);keys.append(key)
    return keys

def analyze(root,phase,write_outputs=True):
    spec=design(phase);folder=out(root,phase);records=[load(p) for p in (folder/'episodes').glob('evaluation_*/*.json')]
    assert len(records)==spec['maximum_evaluation_episodes'],len(records)
    bykey={(r['arm'],r['run'],r['parent'],r['workload'],r['replica']):r for r in records}
    assert len(bykey)==len(records);S=len(spec['runs'])//3;W=len(spec['evaluation_workloads']);R=len(spec['action_replicas'])
    successes={m:np.zeros((3,S,W,R)) for m in ARMS+BASELINES};generated={m:np.zeros((3,S,W,R)) for m in ARMS+BASELINES};summary={}
    fingerprint={}
    for arm in ARMS+BASELINES:
        ds=[];flight=[];slots=[];times=[];metrics=[]
        for run in spec['runs']:
            b=run['parent'];rr=run['repeat'];rid=run['id'] if arm in ARMS else -1
            good=total=0
            for wi,w in enumerate(spec['evaluation_workloads']):
                for ri,rep in enumerate(spec['action_replicas']):
                    r=bykey[arm,rid,b,w,rep];assert r['audit']['status']=='PASS'
                    c=r['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0)
                    successes[arm][b,rr,wi,ri]=c.get('success',0);generated[arm][b,rr,wi,ri]=c['generated']
                    good+=c.get('success',0);total+=c['generated'];times.append(r['decision_mean']);flight.append(r['flight_distance_m']);slots.append(r['slots']);metrics.append(r['metrics'])
                    pair=(w,rep);finger=(r['fingerprint'],r['initial_fingerprint'],c['generated'])
                    if pair in fingerprint:assert fingerprint[pair]==finger,'unpaired evaluation input'
                    fingerprint[pair]=finger
            ds.append(good/total)
        summary[arm]=dict(mean_dsr=float(np.mean(ds)),by_training_run=ds,by_parent=np.asarray(ds).reshape(3,S).mean(1).tolist(),
            decision_mean=float(np.mean(times)),flight_distance_mean_m=float(np.mean(flight)),
            physical_evaluation_episodes=sum(r['arm']==arm for r in records),replicated_baseline_for_pairing_only=arm in BASELINES)
    assert all(np.array_equal(generated[m],generated['initial']) for m in successes)
    comparisons={}
    for a in ('relative_joint','short_joint'):
        for b in ARMS+BASELINES:
            if a!=b:comparisons[a+'--'+b]=paired_summary(successes[a],successes[b],generated[a])
    updates=[load(p) for p in (folder/'updates').glob('*.json')]
    assert len(updates)==len(ARMS)*len(spec['runs'])*spec['rounds']
    mech=all(u['final']['kl']<=.02000001 and abs(u['final']['kl']-u['final']['akl']-u['final']['mkl'])<1e-8 for u in updates)
    mech=mech and all(u['allocation_change']>0 and u['motion_change']>0 for u in updates if u['arm'].endswith('joint'))
    selection=select_method(comparisons,bool(mech)) if phase=='pilot' else load(out(root,'formal')/'frozen_config.json')['pilot_selection']
    result=dict(status='COMPLETE',phase=phase,summary=summary,comparisons=comparisons,mechanism_gate=bool(mech),selection=selection,
        counts=dict(training_episodes=spec['maximum_training_episodes'],evaluation_episodes=len(records),fresh_residual_training_runs=len(spec['runs']),fixed_inherited_parents=3),
        formal_test_read=phase=='formal',scope='Closed-loop finite-bank residual policies with a frozen A2 proposer and pooled frozen scorer. No new independent A2 training or scorer-training repeats. Energy-free dynamic queue protocol, separate from legacy thesis results.')
    if phase=='formal':
        selected=selection['selected_method'];pairs=[selected+'--initial',selected+'--'+selected.replace('joint','decomposed'),'relative_joint--short_joint','relative_joint--shuffled_joint']
        result['primary_family']={k:comparisons[k] for k in pairs};result['primary_interval']='Bonferroni simultaneous 98.75% across four predeclared claims'
    if write_outputs:write(folder/'analysis.json',result)
    return result

def freeze_formal(root):
    root=Path(root).resolve();pilot=load(out(root,'pilot')/'analysis.json');spec=load(out(root,'formal')/'reservation.json')
    assert pilot['phase']=='pilot' and pilot['formal_test_read'] is False
    if (out(root,'formal')/'episodes').exists() or (val(root,'formal')/'execution_authorized.json').exists():raise RuntimeError('formal outcomes/access already exist')
    selection=pilot['selection'];selected=selection['selected_method']
    config=dict(status='FROZEN_GO' if selected else 'FROZEN_NO_GO',phase='formal',specification=spec,pilot_selection=selection,
        pilot_analysis_sha256=t.sha(out(root,'pilot')/'analysis.json'),sealed_scorer_sha256=t.sha(Path(root)/'results'/TASK/SCORER_REV/'models.json'),
        source=source_hashes(),epoch=time.time(),formal_outcomes_generated=False,formal_outcomes_read=False,execute_automatically=False,
        formal_primary_family=[(selected or 'NONE')+'--initial',(selected or 'NONE')+'--'+(selected or 'NONE').replace('joint','decomposed'),'relative_joint--short_joint','relative_joint--shuffled_joint'],
        statistics='Fixed-parent-stratified crossed training-repeat/workload paired bootstrap, 4000 draws; four Bonferroni simultaneous 98.75% primary intervals. Primary DSR uses counts aggregated within each trained policy then equal policy/parent weighting. Action replicas and requests are not training repeats.',
        decision_boundary='NO_GO blocks formal execution. GO freezes method and all configurations before fresh formal training/test. Only an explicit separate --execute-formal invocation can run the full experiment.')
    path=out(root,'formal')/'frozen_config.json'
    if path.exists():raise FileExistsError('frozen formal configuration is immutable')
    write(path,config);write(val(root,'pilot')/'formal_freeze.json',dict(status=config['status'],config_sha256=t.sha(path),formal_test_access=False))
    return config

def _run(root,phase,execute_formal=False):
    if os.name!='posix' or not hasattr(os,'sched_getaffinity'):raise RuntimeError('physical work requires the server')
    root=Path(root).resolve();prepare(root);spec=design(phase);folder=out(root,phase);v=val(root,phase)
    if (folder/'protocol.json').exists():raise FileExistsError('existing scientific run; no automatic restart or overwrite')
    if phase=='formal':
        config=load(folder/'frozen_config.json')
        if not execute_formal or config['status']!='FROZEN_GO':raise PermissionError('formal stage locked/no-go')
        assert config['source']==source_hashes(),'formal source differs from freeze'
        if load(val(root,'pilot')/'server_review.json')['status']!='PASS':raise PermissionError('pilot arithmetic/provenance review has not passed')
        write(v/'execution_authorized.json',dict(epoch=time.time(),config_sha256=t.sha(folder/'frozen_config.json'),explicit_flag=True))
    cpus=sorted(os.sched_getaffinity(0));before=protected(root);write(v/'protection_before.json',before)
    contract=dict(revision=REVISIONS[phase],specification=spec,source=source_hashes(),parents={str(b):t.sha(t.parent_path(root,b)) for b in range(3)},
        frozen_scorer_sha256=t.sha(Path(root)/'results'/TASK/SCORER_REV/'models.json'),allowed_cpus=cpus,
        continuation='Offline scorer labels used candidate + two EDF/hover seconds + A2 suffix. Deployed control replans every second. This is a fixed-reference model-guided score, not exact current-policy Q and not a monotonic policy-improvement guarantee.',
        empty_boundary='When all forecast terminal states are empty, residual scores are zero and only short-window success remains. Mixed-empty relative scores retain their fitted contrast meaning.',
        energy='Energy-free frozen queue engine; report actual flight distance, no invented joules.',formal_test_locked=phase=='pilot')
    write(folder/'protocol.json',contract);pipe=Pipeline(root,phase,cpus)
    for arm in ARMS:
        for run in spec['runs']:t.dump(theta_path(root,phase,arm,run['id'],0),dict(theta=np.zeros(256),optimizer=dict(m=np.zeros(256),v=np.zeros(256),step=0)))
    baseline=evaluation_jobs(pipe,root,phase,True,0) if phase=='pilot' else []
    for version in range(1,spec['rounds']+1):
        keys=[]
        for arm in ARMS:
            for run in spec['runs']:
                for w in run['training'][version-1]:
                    key=f'train:{version}:{arm}:{run["id"]}:{w}'
                    pipe.add(key,'episode',(str(root),phase,run['id'],run['parent'],w,0,arm,version-1,'training'),1);keys.append(key)
        pipe.until(keys);fit_round(pipe,root,phase,version)
        write(v/f'round_{version}_complete.json',dict(status='COMPLETE',epoch=time.time()))
    write(v/'final_models_sealed.json',dict(epoch=time.time(),models={str(theta_path(root,phase,a,r['id'],spec['rounds']).relative_to(root)):t.sha(theta_path(root,phase,a,r['id'],spec['rounds'])) for a in ARMS for r in spec['runs']},formal_test_not_yet_generated=phase=='formal'))
    if phase=='formal':baseline=evaluation_jobs(pipe,root,phase,True,0)
    learned=evaluation_jobs(pipe,root,phase,False,spec['rounds']);pipe.until(learned+baseline);resources=pipe.finish()
    assert len([r for r in resources['records'] if r['kind']=='episode'])==spec['maximum_training_episodes']+spec['maximum_evaluation_episodes']
    after=protected(root);assert before==after;write(v/'protection_after.json',dict(status='PASS',files=len(after),hashes=after))
    result=analyze(root,phase)
    if phase=='pilot':freeze_formal(root)
    write(v/'run_complete.json',dict(status='COMPLETE',epoch=time.time(),analysis_sha256=t.sha(folder/'analysis.json'),formal_started=phase=='formal'))
    print('COMPLETE',phase,result['selection'],flush=True)

def run(root,phase,execute_formal=False):
    if os.name!='posix' or not hasattr(os,'sched_getaffinity'):raise RuntimeError('server only')
    lock=val(root,phase)/'run.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:return _run(root,phase,execute_formal)
    except Exception:
        import traceback
        write(val(root,phase)/'run_failed.json',dict(epoch=time.time(),error=traceback.format_exc(),no_automatic_retry=True))
        raise
    finally:lock.unlink(missing_ok=True)

def smoke(root):
    if os.name!='posix':raise RuntimeError('server smoke only')
    root=Path(root).resolve();v=val(root,'pilot');begin=time.time()
    results=[episode(str(root),'pilot',0,0,909999,0,'relative_joint',0,'training',True),
        episode(str(root),'pilot',-1,0,909999,1,'initial',0,'evaluation',True)]
    write(v/'smoke.json',dict(status='PASS',episodes=2,workload=909999,wall_seconds=time.time()-begin,results=results))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--phase',choices=('pilot','formal'),default='pilot')
    p.add_argument('--mode',choices=('prepare','smoke','run','analyze','freeze'),required=True);p.add_argument('--execute-formal',action='store_true');args=p.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(args.task_root),indent=2))
    elif args.mode=='smoke':smoke(args.task_root)
    elif args.mode=='run':run(args.task_root,args.phase,args.execute_formal)
    elif args.mode=='analyze':analyze(args.task_root,args.phase)
    else:freeze_formal(args.task_root)

if __name__=='__main__':main()
