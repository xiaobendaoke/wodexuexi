"""A public-prediction-only energy price calibrated on training strata.

Neither actual successes nor measured energy is accepted as an optimization
input. The held-out physical and software-cost checks remain necessary.
"""
from __future__ import annotations
import math

CONTRACT='minimum_public_energy_price_per_training_block_v1'

def clean_candidates(candidates):
    out=[]
    for c in candidates:
        score=float(c['score']);energy=float(c['energy'])
        if not math.isfinite(score) or not math.isfinite(energy) or energy<0:raise ValueError('Invalid public candidate')
        out.append(dict(action=int(c['action']),score=score,energy=energy))
    if not out or out[0]['action']!=-1:raise ValueError('G must be the first candidate')
    return out

def choose(candidates,price,normalizer):
    if not math.isfinite(price) or price<0 or not math.isfinite(normalizer) or normalizer<=0:raise ValueError('Invalid frozen price')
    rows=clean_candidates(candidates)
    return min(range(len(rows)),key=lambda i:(-(rows[i]['score']-price*rows[i]['energy']/normalizer),rows[i]['energy'],i))

def calibrate(cases,target=1.05):
    """Smallest nonnegative candidate-price crossing satisfying every block."""
    if not cases or not math.isfinite(target) or target<1:raise ValueError('Invalid training budget')
    rows=[dict(block=str(c['block']),candidates=clean_candidates(c['candidates'])) for c in cases]
    normalizer=max(sum(c['candidates'][0]['energy'] for c in rows)/len(rows),1e-12)
    crossings={0.}
    for case in rows:
        for i,left in enumerate(case['candidates']):
            for right in case['candidates'][i+1:]:
                delta=left['energy']-right['energy']
                if delta==0:continue
                value=(left['score']-right['score'])*normalizer/delta
                if math.isfinite(value) and value>=0:crossings.add(value)
    points=sorted(crossings)
    def evaluate(price):
        groups={};selected=[]
        for case in rows:
            index=choose(case['candidates'],price,normalizer);selected.append(index)
            group=groups.setdefault(case['block'],dict(chosen=0.,G=0.))
            group['chosen']+=case['candidates'][index]['energy'];group['G']+=case['candidates'][0]['energy']
        ratios={key:(z['chosen']/z['G'] if z['G'] else 1. if z['chosen']==0 else math.inf) for key,z in groups.items()}
        return all(value<=target+1e-12 for value in ratios.values()),ratios,selected
    ok,ratios,selected=evaluate(0.)
    if ok:price=0.;index=None
    else:
        def rightward(value):return value+max(abs(value),1.)*1e-12
        low=0;high=len(points)-1
        if not evaluate(rightward(points[high]))[0]:raise RuntimeError('G fallback did not yield a finite feasible price')
        while low<high:
            middle=(low+high)//2
            if evaluate(rightward(points[middle]))[0]:high=middle
            else:low=middle+1
        index=low;price=rightward(points[index]);ok,ratios,selected=evaluate(price)
        if index>0 and evaluate(rightward(points[index-1]))[0]:raise RuntimeError('Price was not the first feasible crossing')
    if not ok:raise RuntimeError('Calibrated training constraint is not satisfied')
    return dict(contract=CONTRACT,price=price,normalizer=normalizer,target=target,states=len(rows),
        blocks=sorted({r['block'] for r in rows}),training_energy_ratio_by_block=ratios,
        crossing_count=len(points),first_feasible_crossing_index=index,selected_candidate_indices=selected,
        no_private_labels_used=True)
