"""Separate sampling calibration from learned gains, under common future values."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics as s
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_probability_decoupling as trial
from scripts.analysis.analyze_queued_motion_trust import correlation
from utils.queued_probability_scale import entropy


def bootstrap(rows,seed):
    rng=random.Random(seed);draws=[]
    for _ in range(2000):
        values=[]
        for w in rng.choices(list(trial.RANK),k=len(trial.RANK)):
            ids=rng.choices(range(8),k=8)
            values.extend(s.mean(row['gains'][r] for r in ids) for row in rows if row['workload']==w)
        draws.append(s.mean(values))
    draws.sort();return dict(gain=s.mean(s.mean(r['gains']) for r in rows),bootstrap_95=[draws[49],draws[1949]])


def analyze(task):
    out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    episodes=[load(p) for p in (out/'episodes').glob('*.json')];ev=[r for r in episodes if not r['training']];assert len(ev)==480
    summary={};contrasts={};robustness={}
    for mode in trial.MODES:
        summary[mode]=[]
        for b in range(3):
            rows=[r for r in ev if r['mode']==mode and r['block']==b];assert len(rows)==32
            c={k:sum(r['counts'][k] for r in rows) for k in ('generated','success','admitted')};summary[mode].append(dict(block=b,**c,dsr=c['success']/c['generated']))
    for ref in trial.MODES[:-1]:
        contrasts[ref]=[summary['decoupled'][b]['dsr']-summary[ref][b]['dsr'] for b in range(3)];robustness[ref]=[]
        for b in range(3):
            def dsr(mode,omit=None):
                rows=[r['counts'] for r in ev if r['mode']==mode and r['block']==b and r['workload']!=omit]
                return sum(r['success'] for r in rows)/sum(r['generated'] for r in rows)
            positive=0
            for w in trial.TEST:
                matched=[r for r in ev if r['block']==b and r['workload']==w];assert len({r['fingerprint'] for r in matched})==len({r['counts']['generated'] for r in matched})==1
                positive+=next(r['counts']['success'] for r in matched if r['mode']=='decoupled')>next(r['counts']['success'] for r in matched if r['mode']==ref)
            robustness[ref].append(dict(positive=positive,min_leave_one_out=min(dsr('decoupled',w)-dsr(ref,w) for w in trial.TEST)))
    rank_rows=[];branches=0;target=[];probability={mode:[] for mode in trial.MODES}
    for b in range(3):
        for w in trial.RANK:
            for t in trial.TIMES:
                snap=trial.state_at(str(task),b,w,t);k=len(snap['pre']['actions']);q=[]
                for r in trial.FUTURES:
                    values=[]
                    for c in range(math.ceil(k/4)):values+=load(out/'labels'/f'{b}_{w}_{t}_{r}_{c}.json')['values']
                    assert len(values)==k;branches+=k;q.append(values)
                a=[s.mean(xs) for xs in zip(*q[:4])];z=[s.mean(xs) for xs in zip(*q[4:])]
                target.append(dict(block=b,workload=w,time=t,split_correlation=correlation(a,z),crossfit_gain=((z[max(range(k),key=a.__getitem__)]-s.mean(z))+(a[max(range(k),key=z.__getitem__)]-s.mean(a)))/2))
                reference=snap['probabilities']['calibrated_initial']
                for mode in trial.MODES:
                    p=snap['probabilities'][mode];gains=[sum((pp-rr)*(v-s.mean(values)) for pp,rr,v in zip(p,reference,values)) for values in q]
                    rank_rows.append(dict(block=b,workload=w,time=t,mode=mode,probabilities=p,values=q,gains=gains))
                    probability[mode].append(dict(block=b,workload=w,time=t,pmax=max(p),entropy=entropy(p,k),argmax=max(range(k),key=p.__getitem__),tv_from_calibrated=sum(abs(x-y) for x,y in zip(p,reference))/2))
    ranking={mode:[bootstrap([r for r in rank_rows if r['mode']==mode and r['block']==b],trial.base.seed(trial.REV,'bootstrap',b)) for b in range(3)] for mode in trial.MODES}
    dsr_gate=all(x>0 for ds in contrasts.values() for x in ds) and all(r['positive']>16 and r['min_leave_one_out']>0 for rs in robustness.values() for r in rs)
    rank_gate=all(r['bootstrap_95'][0]>0 for r in ranking['decoupled'])
    calibration=[load(out/'calibration'/f'{b}.json') for b in range(3)]
    mechanism=all(c['calibrated'] and c['calibrated_policy']['saturated']<c['raw']['saturated'] for c in calibration)
    result=dict(status='COMPLETE',summary=summary,contrasts=contrasts,robustness=robustness,ranking=ranking,ranking_rows=rank_rows,target_agreement=target,probability=probability,
        full_remaining_branches=branches,mechanism_gate=mechanism,dsr_gate=dsr_gate,ranking_gate=rank_gate,independent_confirmation=False,
        scope='24 peak/post-peak fixed states, 8 held-out futures, one calibrated_initial continuation Q shared by all scoring policies; not learned own Q. Four workload clusters, existing three training lineages, conditional developmental bootstrap.')
    trial.base.write(out/'analysis.json',result)
    lines=['# 价值尺度与策略概率解耦验证','', '|方法|平均DSR|逐谱系DSR|','|---|---:|---|']
    for mode,rs in summary.items():lines.append(f'|{mode}|{s.mean(r["dsr"] for r in rs)*100:.4f}%|'+ '/'.join(f'{r["dsr"]*100:.4f}%' for r in rs)+'|')
    lines+=['',f'校准机制门槛：{mechanism}；DSR稳健性门槛：{dsr_gate}；共同参考排序增益门槛：{rank_gate}。',
        'raw_initial=原概率冻结；calibrated_initial=归一化Q、训练状态内熵目标0.5校准温度、残差为零；raw=原评分器PPO；raw_residual=冻结原Q先验+新策略残差PPO；decoupled=冻结归一化Q先验+同结构策略残差PPO。',
        '后两者残差网络、初始化、可训练容量相同，区别为Q到先验概率的映射；raw的可训练架构不同，不将该比较称纯架构优势。',
        '分配、业务、候选银行及119维输入冻结，继续使用上一轮共同A2初始化，不按测试选起点。864训练、480评估、18更新；只有PPO运动梯度，价值教师不产生梯度且逐参数保护。',
        '温度仅从上一轮原初始化训练经验9000–9007，每5秒采样，320状态/谱系校准；只使用已有特征和Q分数，不使用未来标签或回报筛选。没有额外校准环境交互。',
        '训练9500–9595，DSR测试9600–9631，排序9700–9703的75/175秒，未来200–207。排序只测有限共同参考Q的单步改进；完整闭环DSR另行评价。',
        '保持完整批KL≤.02、单步KL≤.0002与Adam回退。实际接受步数和额外校准计算单列，不称总CPU算力相同。三个既有训练谱系的开发实验，不是独立确认；不自动追加训练或解冻分配。']
    for b,c in enumerate(calibration):lines.append(f'\n谱系{b}：温度{c["temperature"]:.6f}，原/校准熵{c["raw"]["mean_entropy"]:.4f}/{c["calibrated_policy"]["mean_entropy"]:.4f}，高集中状态{c["raw"]["saturated"]}/{c["calibrated_policy"]["saturated"]}（各320状态）。')
    for b,r in enumerate(ranking['decoupled']):lines.append(f'\n谱系{b}，相对校准冻结策略的共同Q增益：{r["gain"]:.6f}任务/状态，条件bootstrap95% {r["bootstrap_95"]}。')
    dest=Path(task)/'output'/trial.base.TASK/'deliverables'/'queued_probability_decoupling.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=REPO_ROOT);a=p.parse_args();analyze(a.task)
