"""Public candidate construction and training-only conditional future sampling."""
import bisect
import math
import random
import statistics
from functools import lru_cache
from environment.queued_service import Engine,Task,distance,safe_path
from utils.dynamic_service_scenarios import Scenario
from utils.queued_service_business import seed_for,poisson,BASIS


@lru_cache(maxsize=16)
def geometry(workload):return Scenario(seed_for('geometry',workload),'D')


def conditional_suffix(workload,ues,start,replica):
    """Independent marked-Poisson suffix, conditional on fixed latent hotspot path.

    Private simulation input only. Fixed geometry/phase/direction are never sent
    to the candidate generator or policies as future information.
    """
    geo=geometry(workload)
    assert tuple(ues[i] for i in range(100))==geo.ue_positions
    result=[]
    for offset in range(1,6):
        t=start+offset;rate=60. if 50<=t<100 else 15.
        if t>=200:result.append([]);continue
        def rng(stream):return random.Random(seed_for(f'joint_opportunity_20261102/{workload}/{start}/{replica}/{stream}',t))
        count=poisson(150.,rng('count'));thin=rng('thin');ue_rng=rng('ue');attrs=rng('attributes')
        center=geo.center(t);cumulative=[];total=0.
        for xy in geo.ue_positions:
            total+=.2+.8*math.exp(-sum((xy[k]-center[k])**2 for k in range(2))/20000);cumulative.append(total)
        row=[]
        for i in range(count):
            gate=thin.random();ue=bisect.bisect_left(cumulative,ue_rng.random()*total)
            size=attrs.randint(*BASIS['input_bytes']);density=attrs.randint(*BASIS['cycles_per_bit']);mark=attrs.random()
            if gate>=rate/150.:continue
            row.append(Task(10**9+replica*10**6+t*1000+i,ue,float(t),t+2+mark,float(8*size),float(8*size*density)))
        result.append(row)
    return result


def expanded_bank(obs,original):
    p=obs['protocol'];positions=obs['positions'];n=len(positions);bank=[[tuple(d) for d in move] for move in original]
    def toward(i,targets):
        total=sum(w for _,w in targets)
        if not total:return (0.,0.)
        target=tuple(sum(x[k]*w for x,w in targets)/total for k in range(2))
        delta=tuple(target[k]-positions[i][k] for k in range(2));scale=min(1.,p.speed/max(math.hypot(*delta),1e-12))
        return tuple(x*scale for x in delta)
    upload=[];forward=[]
    for i in range(n):
        tasks=[x for x in obs['tasks'] if x['source']==i and x['remaining_upload']>0]
        upload.append(toward(i,[(obs['ues'][x['ue']],x['remaining_upload']/max(x['deadline']-obs['time'],.1)) for x in tasks]))
        tasks=[x for x in obs['tasks'] if x['source']==i and x['target'] not in (None,i) and x['remaining_transfer']>0]
        targets=[(p.mbs if x['target']==n else positions[x['target']],x['remaining_transfer']/max(x['deadline']-obs['time'],.1)) for x in tasks]
        forward.append((sum(w for _,w in targets),i,toward(i,targets)))
    proposals=[upload]
    for i,d in enumerate(upload):
        move=[(0.,0.)]*n;move[i]=d;proposals.append(move)
    for _,i,d in sorted(forward,reverse=True)[:2]:
        move=[(0.,0.)]*n;move[i]=d;proposals.append(move)
    for move in proposals:
        for fraction in (1.,.5,.25):
            action=[(dx*fraction,dy*fraction) for dx,dy in move]
            if safe_path(positions,action,p):
                if action not in bank:bank.append(action)
                break
    assert bank[:len(original)]==original and len(bank)<=16
    assert all(safe_path(positions,a,p) for a in bank)
    return bank


def motion_plan(obs,motion):
    """EDF-aware routing estimate with motion-integrated link rates, current masks."""
    p=obs['protocol'];pos=obs['positions'];n=len(pos);radio=Engine(p,pos,obs['ues']);work=[[] for _ in p.capacity]
    inter=sum(x['remaining_transfer'] for x in obs['tasks'] if x['target'] not in (None,x['source'],n))
    back=sum(x['remaining_transfer'] for x in obs['tasks'] if x['target']==n)
    for x in obs['tasks']:
        if x['target'] is not None:work[x['target']].append((x['deadline'],x['remaining_cycles']))
    def rate(kind,source,target):
        values=[]
        for k in range(10):
            f=(k+.5)/10;a=tuple(pos[source][j]+f*motion[source][j] for j in range(2))
            if kind=='uplink':
                b=obs['ues'][target];ok=distance(a,b)<=p.coverage
                values.append(radio.rate(kind,(*a,p.altitude),(*b,0.)) if ok else 0.)
            else:
                b=p.mbs if target==n else (*tuple(pos[target][j]+f*motion[target][j] for j in range(2)),p.altitude)
                ok=target==n or distance(a,b)<=p.inter_range
                values.append(radio.rate(kind,(*a,p.altitude),b) if ok else 0.)
        return max(statistics.mean(values),1e-9)
    plan={}
    for x in sorted(obs['tasks'],key=lambda x:(x['deadline'],x['uid'])):
        source=x['source']
        if source is None or x['target'] is not None:continue
        active={z['ue'] for z in obs['tasks'] if z['source']==source and z['remaining_upload']>0}
        upload=x['remaining_upload']*max(1,len(active))/rate('uplink',source,x['ue'])
        choices=[]
        for node in range(n+1):
            if node not in (source,n) and distance(pos[source],pos[node])>p.inter_range:continue
            transfer=0. if node==source else ((back if node==n else inter)+x['bits'])/rate('backhaul' if node==n else 'inter',source,node)
            cpu=(x['remaining_cycles']+sum(c for d,c in work[node] if d<=x['deadline']))/p.capacity[node]
            latency=upload+transfer+cpu;choices.append(((latency>x['deadline']-obs['time'],latency,node),node))
        node=min(choices)[1];plan[x['uid']]=node;work[node].append((x['deadline'],x['remaining_cycles']))
        if node!=source:
            if node==n:back+=x['bits']
            else:inter+=x['bits']
    return plan


def crossfit(futures,old_count):
    """Two-fold selector evaluation. Rectangles: future x fixed-plan x motion."""
    assert len(futures)>=2 and len(futures)%2==0
    r=len(futures[0]);k=len(futures[0][0]);assert 0<old_count<=k
    assert all(len(m)==r and all(len(row)==k for row in m) for m in futures)
    result=[];half=len(futures)//2
    for train_ids,test_ids in ((range(half),range(half,len(futures))),(range(half,len(futures)),range(half))):
        mean=lambda ids:[[statistics.mean(futures[f][a][u] for f in ids) for u in range(k)] for a in range(r)]
        train=mean(train_ids);test=mean(test_ids)
        def best(cols):return max(((a,u) for a in range(r) for u in cols),key=lambda z:train[z[0]][z[1]])
        a,u=best(range(k));oa,ou=best(range(old_count));nu=max(range(k),key=lambda j:train[0][j])
        common_u=max(range(k),key=lambda j:statistics.mean(train[i][j] for i in range(r)))
        common_a=max(range(r),key=lambda i:statistics.mean(train[i]))
        motion_gain=statistics.mean(test[i][max(range(k),key=lambda j:train[i][j])]-test[i][common_u] for i in range(r))
        allocation_gain=statistics.mean(test[max(range(r),key=lambda i:train[i][j])][j]-test[common_a][j] for j in range(k))
        result.append(dict(capacity_gain=test[a][u]-test[oa][ou],joint_vs_native=test[a][u]-test[0][nu],
                           motion_conditioning=motion_gain,allocation_conditioning=allocation_gain))
    return {name:statistics.mean(x[name] for x in result) for name in result[0]}
