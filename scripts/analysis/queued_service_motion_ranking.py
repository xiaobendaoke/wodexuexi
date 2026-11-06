"""Frozen-allocation, paired six-step motion-ranking experiment (server only)."""
import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from environment.queued_service import Engine, Protocol
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_service_shared_budget import BudgetAdapter
from utils.queued_service_controllers import Controller
from scripts.analysis.queued_service_validation import START

TASK = 'chapter5_dynamic_service_20261028'
REV = 'queued_motion_ranking_20261101'
PARENT = 'queued_plan_residual_20261101'
TIMES = (30, 45, 55, 65, 75, 85, 95, 105, 120, 145, 170, 190)
TRAIN = tuple(range(6000, 6008))
VALID = tuple(range(6008, 6016))
TEST = tuple(range(6100, 6124))
CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'big') % (2**63-1)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(pickle.dumps(value, protocol=5), compresslevel=1, mtime=0))


def read(path):
    return pickle.loads(gzip.decompress(path.read_bytes()))


def out(task):
    return Path(task)/'results'/TASK/REV


def checkpoint(task, block):
    return Path(task)/'results'/TASK/PARENT/'models'/str(block)/'joint'/'0006.pt'


def init(cpus):
    os.sched_setaffinity(0, {cpus[(mp.current_process()._identity[-1]-1) % len(cpus)]})
    import torch
    torch.set_num_threads(1)


def model_at(task, block, learned=False):
    import torch
    from marl_models.queued_service_plan_residual import ResidualPolicy
    key = (str(task), block, learned)
    if key not in CACHE:
        saved = torch.load(checkpoint(task, block), map_location='cpu', weights_only=False)
        assert saved['source_contract'] == 'f5287d76c8f3f2603d4a92f3927be7bc831a4e264e9c55145695e6073fd9ab87'
        model = ResidualPolicy('joint'); model.load_state_dict(saved['model'])
        if learned:
            model.motion.load_state_dict(torch.load(out(task)/'models'/f'{block}.pt', map_location='cpu', weights_only=True))
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval(); CACHE[key] = model
    return CACHE[key]


def distribution(model, pre, post):
    from marl_models.queued_service_joint import tensor
    return model.motion(tensor([pre['state']]), tensor([pre['features']]), tensor([pre['mask']], __import__('torch').bool),
                        tensor([post['state']]), tensor([post['features']]), True)


def banks(adapter, obs, plan):
    from utils.queued_service_learning import VirtualQueue
    prefix = VirtualQueue(obs)
    for uid, target in plan.items(): prefix.assign(uid, target)
    pre, post = adapter.motion(obs), adapter.motion(prefix.obs)
    assert pre['actions'] == post['actions'] and pre['mask'] == post['mask']
    return pre, post


def episode(payload):
    task, block, workload, mode, collect = payload
    import torch
    from scripts.analysis.diagnose_queued_service_conditioning import choose
    model = model_at(task, block, mode.startswith('learned'))
    tape = RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = Engine(Protocol(), START, tape.ue_positions); adapter = BudgetAdapter()
    snapshots = []; durations = []; rows = []; started = time.monotonic()
    while engine.t < 200 or engine.tasks:
        t = int(engine.t); assert t <= 204
        engine.add(tape.slot(t)); engine.admit(); obs = engine.observe()
        begin = time.monotonic()
        action_seed = seed(REV, block, workload, t)
        if mode == 'model':
            row, plan, motion = choose(model, obs, adapter, action_seed, 'model_actual', torch)
        else:
            row, plan, motion = model.act(obs, adapter, action_seed)
            if mode == 'uniform':
                g = torch.Generator().manual_seed(seed(REV, 'uniform', block, workload, t))
                motion = adapter.plans[int(torch.randint(len(adapter.plans), (1,), generator=g))]
            elif mode == 'learned_argmax':
                motion = adapter.plans[max(range(len(adapter.plans)), key=lambda i: row['motion_probs'][i])]
        durations.append(time.monotonic()-begin)
        if collect and t in TIMES:
            pre, post = banks(adapter, obs, plan)
            snapshots.append(dict(engine=copy.deepcopy(engine), adapter=copy.deepcopy(adapter), plan=plan,
                                  pre=pre, post=post, time=t, block=block, workload=workload))
        engine.assign(plan); result = engine.advance(motion); rows.append(result)
    audit = engine.audit(); summary = engine.summary(); c = summary['counts']
    assert c['generated'] == c['success']+c['expired'] and not engine.tasks
    assert sum(r['success'] for r in rows) == c['success']
    name = f'{block}_{workload}_{mode}'
    record = dict(block=block, workload=workload, mode=mode, counts=dict(c), summary=summary,
                  fingerprint=tape.fingerprint, audit=audit, rows=rows, allocation_sha=sha(checkpoint(task, block)),
                  decision_seconds=statistics.mean(durations), max_decision_seconds=max(durations),
                  over_one_second=sum(x>1 for x in durations), cpu=sorted(os.sched_getaffinity(0)), seconds=time.monotonic()-started)
    write(out(task)/'episodes'/f'{name}.json', record)
    if collect: dump(out(task)/'states'/f'{block}_{workload}.pkl.gz', snapshots)
    return dict(cpu=record['cpu'], file=f'episodes/{name}.json')


def label(payload):
    task, block, workload, index, replica = payload
    snap = read(out(task)/'states'/f'{block}_{workload}.pkl.gz')[index]
    model = model_at(task, block); tape = RecoveryScenario(workload, 'burst', 'normal', .6)
    frozen = hashlib.sha256(pickle.dumps(snap, protocol=5)).hexdigest()
    values = []; fingerprints = []
    for candidate in snap['pre']['actions']:
        e = copy.deepcopy(snap['engine']); adapter = copy.deepcopy(snap['adapter'])
        base = e.counts['success']; e.assign(snap['plan']); e.advance(candidate)
        # Six arrival slots including the current one, followed by deadline drain.
        # Only the first action is intervened on; suffix replans every second.
        for offset in range(1, 9):
            t = int(e.t)
            e.add(tape.slot(t) if offset < 6 else ()); e.admit()
            obs = e.observe()
            _, plan, motion = model.act(obs, adapter, seed(REV, 'suffix', block, workload, snap['time'], replica, offset))
            e.assign(plan); e.advance(motion)
        assert not e.tasks
        e.audit(); values.append(e.counts['success']-base); fingerprints.append(e.counts['generated'])
    assert len(set(fingerprints)) == 1
    assert hashlib.sha256(pickle.dumps(snap, protocol=5)).hexdigest() == frozen
    record = dict(block=block, workload=workload, time=snap['time'], index=index, replica=replica,
                  values=values, generated=fingerprints[0], cpu=sorted(os.sched_getaffinity(0)))
    write(out(task)/'labels'/f'{block}_{workload}_{index}_{replica}.json', record)
    return dict(cpu=record['cpu'])


def labelled(task, block, workloads):
    result = []
    for w in workloads:
        snapshots = read(out(task)/'states'/f'{block}_{w}.pkl.gz')
        for i, snap in enumerate(snapshots):
            labels = [json.loads((out(task)/'labels'/f'{block}_{w}_{i}_{r}.json').read_text())['values'] for r in range(3)]
            result.append((snap, [statistics.mean(xs) for xs in zip(*labels)]))
    return result


def ranking(model, data):
    import torch
    rows = []
    with torch.no_grad():
        for snap, q in data:
            p = distribution(model, snap['pre'], snap['post']).probs[0, :len(q)].tolist()
            expected = sum(a*b for a,b in zip(p,q)); best = max(range(len(q)), key=lambda i:p[i])
            rows.append(dict(workload=snap['workload'], time=snap['time'], gain=expected-statistics.mean(q),
                             regret=max(q)-expected, argmax_gain=q[best]-statistics.mean(q), spread=max(q)-min(q), probabilities=p, values=q))
    return dict(gain=statistics.mean(r['gain'] for r in rows), regret=statistics.mean(r['regret'] for r in rows),
                argmax_gain=statistics.mean(r['argmax_gain'] for r in rows), informative=sum(r['spread']>0 for r in rows), rows=rows)


def fit(payload):
    task, block = payload
    import torch
    torch.manual_seed(seed(REV, 'fit', block))
    model = copy.deepcopy(model_at(task, block)); model.motion.train()
    for p in model.motion.parameters(): p.requires_grad_(True)
    before = {k:v.clone() for k,v in model.allocation.state_dict().items()}
    data = labelled(task, block, TRAIN); valid = labelled(task, block, VALID)
    initial = dict(train=ranking(model, data), validation=ranking(model, valid))
    optimizer = torch.optim.Adam(model.motion.parameters(), lr=3e-4); losses=[]
    # Fixed final epoch; validation never selects a checkpoint or hyperparameter.
    for epoch in range(40):
        for ids in torch.randperm(len(data)).split(32):
            terms=[]
            for i in ids.tolist():
                snap,q = data[i]; dist=distribution(model,snap['pre'],snap['post'])
                target=torch.softmax(torch.tensor(q)/.5,dim=0)
                terms.append(-(target*dist.logits[0,:len(q)]).sum())
            loss=torch.stack(terms).mean(); assert bool(torch.isfinite(loss))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.motion.parameters(),.5);optimizer.step();losses.append(float(loss))
    assert all(torch.equal(v, before[k]) for k,v in model.allocation.state_dict().items())
    target=out(task)/'models'/f'{block}.pt';target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.motion.state_dict(),target)
    record=dict(block=block, epochs=40, optimizer_steps=len(losses), loss_first=losses[0], loss_last=losses[-1],
                initial=initial, final=dict(train=ranking(model,data),validation=ranking(model,valid)),
                allocation_unchanged=True, motion_sha=sha(target),cpu=sorted(os.sched_getaffinity(0)))
    write(out(task)/'fit'/f'{block}.json',record);return dict(cpu=record['cpu'])


def parallel(task, stage, function, jobs, cpus):
    started=time.time(); results=[]
    with ProcessPoolExecutor(max_workers=len(cpus),mp_context=mp.get_context('fork'),initializer=init,initargs=(cpus,)) as pool:
        futures=[pool.submit(function,j) for j in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%32==0: print(stage,len(results),'/',len(jobs),flush=True)
    record=dict(stage=stage,jobs=len(jobs),completed=len(results),workers=len(cpus),allowed_cpus=cpus,
                observed_cpus=sorted({c for r in results for c in r['cpu']}),seconds=time.time()-started)
    write(out(task)/'execution'/f'{stage}.json',record)


def report(task):
    records=[json.loads(p.read_text()) for p in (out(task)/'episodes').glob('*.json') if int(p.stem.split('_')[1]) in TEST]
    modes=('native','uniform','learned','learned_argmax','model'); summary={}; paired={}
    for mode in modes:
        summary[mode]=[]
        for block in range(3):
            rows=[r for r in records if r['mode']==mode and r['block']==block];assert len(rows)==len(TEST)
            g=sum(r['counts']['generated'] for r in rows);s=sum(r['counts']['success'] for r in rows);a=sum(r['counts']['admitted'] for r in rows)
            summary[mode].append(dict(block=block,generated=g,success=s,dsr=s/g,admission=a/g,conditional=s/max(a,1),
                                      decision_seconds=statistics.mean(r['decision_seconds'] for r in rows)))
    for block in range(3):
        for w in TEST:
            rows=[r for r in records if r['block']==block and r['workload']==w]
            assert len({r['fingerprint'] for r in rows})==1 and len({r['counts']['generated'] for r in rows})==1
    for mode in modes[1:]:paired[mode]=[summary[mode][b]['dsr']-summary['native'][b]['dsr'] for b in range(3)]
    fits=[json.loads((out(task)/'fit'/f'{b}.json').read_text()) for b in range(3)]
    result=dict(status='COMPLETE',summary=summary,paired_dsr_gain=paired,
                ranking_gain=[f['final']['validation']['gain']-f['initial']['validation']['gain'] for f in fits],
                allocation_unchanged=all(f['allocation_unchanged'] for f in fits), independent_confirmation=False,
                scope='Frozen-allocation model-assisted ranking learning; three development training initializations, not joint-RL confirmation')
    write(out(task)/'analysis.json',result)
    lines=['# 冻结分配的多步运动排序实验','', '三训练初始化；开发机制实验，未进行联合RL独立确认。','',
           '|运动|三个种子DSR（%）|相对原运动（百分点）|','|---|---|---|']
    for m in modes:lines.append('|'+m+'|'+'/'.join(f'{r["dsr"]*100:.4f}' for r in summary[m])+'|'+('/'.join(f'{v*100:+.4f}' for v in paired[m]) if m in paired else '—')+'|')
    lines+=['', '验证排序期望收益变化（任务/状态）：'+str(result['ranking_gain']), '',
            '标签为六批到达加尾部排空；每候选采用同一外生任务后缀和三个策略随机后缀。不是三个独立未来到达样本，也不是长期价值上界。',
            '只干预首步，后续每秒按同一冻结策略重规划。训练40轮，温度0.5，末轮交付；验证不选模型。未来结果仅作训练/诊断标签。',
            '闭环使用新工作负载6100–6123和完整200秒到达及排空。分配参数冻结，具体分配可随运动引起的状态变化而不同。',
            '全部原始状态、候选标签、拟合记录与闭环记录保存在服务器；CPU使用见execution记录。显式自审。']
    dest=Path(task)/'output'/TASK/'deliverables'/'queued_motion_ranking.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args();task=args.task
    cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>1
    if args.smoke:
        import torch
        torch.set_num_threads(1)
        original=sha(checkpoint(task,0))
        episode((str(task),0,TRAIN[0],'native',True))
        label((str(task),0,TRAIN[0],6,0))
        path=out(task)/'labels'/f'0_{TRAIN[0]}_6_0.json'; first=path.read_bytes()
        label((str(task),0,TRAIN[0],6,0));assert path.read_bytes()==first
        assert sha(checkpoint(task,0))==original
        write(out(task)/'smoke.json',dict(status='PASS',paired_label_replay=True,parent_unchanged=True))
        print('SMOKE PASS',flush=True);return
    manifest={str(p.relative_to(REPO_ROOT)):sha(p) for folder in ('environment','marl_models','utils','scripts/analysis') for p in (REPO_ROOT/folder).glob('*.py')}
    protocol=dict(revision=REV,source_sha=manifest,parent_checkpoints={str(b):sha(checkpoint(task,b)) for b in range(3)},
                  train=TRAIN,validation=VALID,test=TEST,times=TIMES,horizon=6,policy_suffix_replicas=3,
                  business='Frozen RecoveryScenario burst .6 normal; Protocol unchanged',candidate='Frozen Controller.candidates',
                  optimizer=dict(epochs=40,batch=32,lr=.0003,soft_target_temperature=.5),cpu=cpus,
                  acceptance='Report all outcomes; ranking improvement and new-workload closed-loop DSR positive per block; no tuning or automatic budget expansion')
    write(out(task)/'protocol.json',protocol)
    parallel(task,'collect',episode,[(str(task),b,w,'native',True) for b in range(3) for w in TRAIN+VALID],cpus)
    parallel(task,'labels',label,[(str(task),b,w,i,r) for b in range(3) for w in TRAIN+VALID for i in range(len(TIMES)) for r in range(3)],cpus)
    parallel(task,'fit',fit,[(str(task),b) for b in range(3)],cpus)
    parallel(task,'evaluation',episode,[(str(task),b,w,m,False) for b in range(3) for w in TEST for m in ('native','uniform','learned','learned_argmax','model')],cpus)
    report(task)
    assert all(sha(REPO_ROOT/p)==h for p,h in manifest.items())
    assert all(sha(checkpoint(task,b))==protocol['parent_checkpoints'][str(b)] for b in range(3))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
