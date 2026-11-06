"""Cross-score the same independent Q matrices without any additional rollout."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def probability_diagnosis(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_motion_trust_experiment as trial
    out=trial.out(task);path=out/'analysis.json';data=json.loads(path.read_text(encoding='utf-8'));rows=data['ranking_rows'];groups={};changes={}
    for mode in trial.MODES:
        groups[mode]={}
        for t in trial.TIMES:
            selected=[r for r in rows if r['mode']==mode and r['time']==t];valid=[r for r in selected if len(r['probabilities'])>1]
            entropies=[-sum(p*math.log(p) for p in r['probabilities'] if p>0)/math.log(len(r['probabilities'])) for r in valid]
            groups[mode][str(t)]=dict(states=len(selected),multiple_candidates=len(valid),pmax_ge_0999=sum(max(r['probabilities'])>=.999 for r in valid),
                pmax_ge_099=sum(max(r['probabilities'])>=.99 for r in valid),mean_pmax=statistics.mean(max(r['probabilities']) for r in valid) if valid else None,
                mean_normalized_entropy=statistics.mean(entropies) if entropies else None)
    for t in trial.TIMES:
        pairs=[]
        for row in rows:
            if row['mode']!='trust' or row['time']!=t:continue
            initial=next(x for x in rows if x['mode']=='initial' and x['block']==row['block'] and x['workload']==row['workload'] and x['time']==t)
            p=row['probabilities'];q=initial['probabilities'];assert len(p)==len(q)
            pairs.append((sum(abs(a-b) for a,b in zip(p,q))/2,max(range(len(p)),key=p.__getitem__)!=max(range(len(q)),key=q.__getitem__)))
        changes[str(t)]=dict(states=len(pairs),mean_tv=statistics.mean(x[0] for x in pairs),changed_argmax=sum(x[1] for x in pairs))
    result=dict(status='COMPLETE',analysis_sha=hashlib.sha256(path.read_bytes()).hexdigest(),diagnostic_source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),groups=groups,trust_minus_initial=changes,
        scope='Posthoc probability descriptions on 48 fixed states only; no training, new rollouts, causal temperature intervention, or claim that these states represent all training occupancy.')
    trial.base.write(out/'probability_diagnosis.json',result);print(json.dumps(result,indent=2))


def diagnose(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_motion_trust_experiment as trial
    from scripts.analysis.analyze_queued_motion_trust import ranking
    out=trial.out(task);data=json.loads((out/'analysis.json').read_text());rows=data['ranking_rows'];matrix={};contrasts={}
    for continuation in trial.MODES:
        matrix[continuation]={}
        for scoring in trial.MODES:
            values=[]
            for b in range(3):
                gains=[]
                for row in rows:
                    if row['mode']!=continuation or row['block']!=b:continue
                    probs=next(x['probabilities'] for x in rows if x['block']==b and x['workload']==row['workload'] and x['time']==row['time'] and x['mode']==scoring)
                    gains.extend(sum(p*(v-statistics.mean(q)) for p,v in zip(probs,q)) for q in row['values'])
                values.append(statistics.mean(gains))
            matrix[continuation][scoring]=values
    for continuation in ('initial','trust'):
        summaries=[]
        for b in range(3):
            selected=[]
            for row in rows:
                if row['mode']!=continuation or row['block']!=b:continue
                matched={x['mode']:x['probabilities'] for x in rows if x['block']==b and x['workload']==row['workload'] and x['time']==row['time']}
                delta=[p-z for p,z in zip(matched['trust'],matched['initial'])]
                gains=[sum(d*(v-statistics.mean(q)) for d,v in zip(delta,q)) for q in row['values']]
                selected.append(dict(row,gains=gains))
            r=ranking(selected,list(trial.RANK),trial.base.seed(trial.REV,'cross_score_bootstrap',b))
            summaries.append(dict(expected_gain=r['expected_gain'],bootstrap_95=r['bootstrap_95'],phase_gains=r['phase_gains']))
        contrasts[continuation]=summaries
    result=dict(status='COMPLETE',matrix_mean_gain_over_uniform=matrix,trust_minus_initial=contrasts,training=False,extra_rollouts=0,
        scope='Posthoc fixed-state cross-scoring on common continuation Q, probabilities centered on each candidate return mean. No model selection, policy improvement theorem claim or independent training replication.')
    trial.base.write(out/'cross_policy_ranking.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--probabilities-only',action='store_true');a=p.parse_args()
    (probability_diagnosis if a.probabilities_only else diagnose)(a.task,a.source)
