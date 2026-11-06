"""Independent-future expected ranking gain and paired closed-loop arithmetic."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics as s
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_motion_trust_experiment as trial


def correlation(a,b):
    a=[x-s.mean(a) for x in a];b=[x-s.mean(b) for x in b];den=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/den if den else None


def ranking(rows,workloads,seed):
    rng=random.Random(seed);draws=[]
    for _ in range(2000):
        values=[]
        for w in rng.choices(workloads,k=len(workloads)):
            future_ids=rng.choices(range(8),k=8)
            values.extend(s.mean(row['gains'][r] for r in future_ids) for row in rows if row['workload']==w)
        draws.append(s.mean(values))
    draws.sort();corr=[r['split_correlation'] for r in rows if r['split_correlation'] is not None]
    return dict(expected_gain=s.mean(s.mean(r['gains']) for r in rows),bootstrap_95=[draws[49],draws[1949]],
        split_correlation=s.mean(corr) if corr else None,crossfit_selector_gain=s.mean(r['crossfit_selector_gain'] for r in rows),
        predicted_argmax_gain=s.mean(r['argmax_gain'] for r in rows),states=len(rows),
        phase_gains={str(t):s.mean(s.mean(r['gains']) for r in rows if r['time']==t) for t in trial.TIMES})


def analyze(task):
    out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'));ev=[load(p) for p in (out/'episodes').glob('*.json') if not load(p)['training']]
    assert len(ev)==288;summary={};contrasts={};robustness={}
    for mode in trial.MODES:
        summary[mode]=[]
        for b in range(3):
            selected=[r for r in ev if r['mode']==mode and r['block']==b];assert len(selected)==32
            counts={key:sum(r['counts'][key] for r in selected) for key in ('success','generated','admitted')}
            summary[mode].append(dict(block=b,**counts,dsr=counts['success']/counts['generated']))
    for ref in ('initial','legacy'):
        contrasts[ref]=[summary['trust'][b]['dsr']-summary[ref][b]['dsr'] for b in range(3)];robustness[ref]=[]
        for b in range(3):
            def dsr(mode,omit=None):
                c=[r['counts'] for r in ev if r['mode']==mode and r['block']==b and r['workload']!=omit]
                return sum(x['success'] for x in c)/sum(x['generated'] for x in c)
            positive=0
            for w in trial.TEST:
                matched=[r for r in ev if r['block']==b and r['workload']==w];assert len({r['fingerprint'] for r in matched})==len({r['counts']['generated'] for r in matched})==1
                a=next(r for r in matched if r['mode']=='trust');z=next(r for r in matched if r['mode']==ref);positive+=a['counts']['success']>z['counts']['success']
            robustness[ref].append(dict(positive=positive,min_leave_one_out=min(dsr('trust',w)-dsr(ref,w) for w in trial.TEST)))
    rows=[];branches=0
    for b in range(3):
        for w in trial.RANK:
            for t in trial.TIMES:
                snap=trial.state_at(str(task),b,w,t);k=len(snap['pre']['actions'])
                for mode in trial.MODES:
                    q=[];p=snap['probabilities'][mode]
                    for r in trial.FUTURES:
                        values=[]
                        for c in range(math.ceil(k/4)):values+=load(out/'labels'/f'{b}_{w}_{t}_{r}_{mode}_{c}.json')['values']
                        assert len(values)==k;branches+=k;q.append(values)
                    a=[s.mean(xs) for xs in zip(*q[:4])];z=[s.mean(xs) for xs in zip(*q[4:])]
                    gain=((z[max(range(k),key=a.__getitem__)]-s.mean(z))+(a[max(range(k),key=z.__getitem__)]-s.mean(a)))/2
                    gains=[sum((pp-1/k)*v for pp,v in zip(p,values)) for values in q]
                    argmax=max(range(k),key=p.__getitem__)
                    rows.append(dict(block=b,workload=w,time=t,mode=mode,probabilities=p,values=q,gains=gains,split_correlation=correlation(a,z),
                        crossfit_selector_gain=gain,argmax_gain=s.mean(values[argmax]-s.mean(values) for values in q)))
    rank={mode:[ranking([r for r in rows if r['mode']==mode and r['block']==b],list(trial.RANK),trial.base.seed(trial.REV,'bootstrap',b)) for b in range(3)] for mode in trial.MODES}
    rankgate=all(r['bootstrap_95'][0]>0 and r['split_correlation'] is not None and r['split_correlation']>0 and r['crossfit_selector_gain']>0 for r in rank['trust'])
    dsrgate=all(d>0 for ds in contrasts.values() for d in ds) and all(r['positive']>16 and r['min_leave_one_out']>0 for rs in robustness.values() for r in rs)
    updates=[load(p) for p in (out/'updates').glob('*.json')];assert len(updates)==12
    repair=all(d['metrics']['measured_full_kl']<=.020001 for d in updates if d['arm']=='trust')
    result=dict(status='COMPLETE',summary=summary,trust_contrasts=contrasts,robustness=robustness,ranking=rank,ranking_rows=rows,full_remaining_branches=branches,
        repair_kl_gate=repair,ranking_gate=rankgate,dsr_gate=dsrgate,independent_confirmation=False,
        scope='48 common states, 8 independent futures; each method uses its own frozen continuation policy. Future labels are never fitted. Bootstrap resamples workloads and shared future IDs across times; conditional developmental uncertainty with only four workload clusters.')
    trial.base.write(out/'analysis.json',result)
    lines=['# 运动步长修复与独立未来排序验证','', '|方法|平均DSR|逐谱系DSR|','|---|---:|---|']
    for mode,rs in summary.items():lines.append(f'|{mode}|{s.mean(r["dsr"] for r in rs)*100:.4f}%|'+ '/'.join(f'{r["dsr"]*100:.4f}%' for r in rs)+'|')
    lines+=['',f'完整批KL门槛：{repair}；排序门槛：{rankgate}；DSR稳健性门槛：{dsrgate}。',
        '三个既有训练谱系；开发结果，不是新训练种子独立确认。分配、业务、119维候选特征与候选生成规则冻结。',
        '每训练臂每谱系两轮×48完整回合；每个候选仅干预首步，后续分别恢复相应方法自己的末策略，完整运行至200秒并排空。',
        '未来100–107全部未进入训练，排序状态9100–9103与训练9000–9095、DSR测试9200–9231隔离。4/4分半仅作目标一致性诊断，网络预测从未据其选择或拟合。',
        '当前评分器经过PPO，输出是策略评分而非校准Q；报告的是其概率/排序相对独立长期完成标签的表现。',
        '每次Adam候选步后检查全批旧策略KL≤.02及相对上个已接受策略KL≤.0002；最多12次折半，失败回退参数和Adam状态。两次连续全失败只停止actor，critic继续。实际接受/拒绝和推理成本单列。',
        'KL仅约束保存状态上的条件运动分布，不是完整轨迹或未见状态的严格信赖域；多次全批验收有额外计算成本。']
    for mode,rs in rank.items():
        lines+=['',mode+'：']
        for b,r in enumerate(rs):lines.append(f'- 谱系{b}：期望相对均匀收益{r["expected_gain"]:.4f}任务/状态，条件bootstrap95% {r["bootstrap_95"]}，分半相关{r["split_correlation"]}，交叉选择收益{r["crossfit_selector_gain"]:.4f}。')
    dest=Path(task)/'output'/trial.base.TASK/'deliverables'/'queued_motion_trust.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=REPO_ROOT);a=p.parse_args();analyze(a.task)
