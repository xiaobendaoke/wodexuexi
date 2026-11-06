"""Post-run matched-state ranking check; does not modify or select checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import statistics
import sys


def case(payload):
    task,block=payload
    import torch
    torch.set_num_threads(1);os.sched_setaffinity(0,{block})
    from scripts.analysis import queued_service_motion_ppo as trial
    from scripts.analysis import queued_service_motion_ranking as base
    data=base.labelled(task,block,base.VALID)
    warm=base.ranking(trial.model_at(task,block,0),data)
    final=base.ranking(trial.model_at(task,block,trial.ROUNDS),data)
    old=json.loads((base.out(task)/'fit'/f'{block}.json').read_text())['final']['validation']['gain']
    assert abs(warm['gain']-old)<1e-6
    return dict(block=block,warm=warm,final=final,change=final['gain']-warm['gain'],cpu=sorted(os.sched_getaffinity(0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args();sys.path.insert(0,str(args.source))
    from scripts.analysis import queued_service_motion_ppo as trial
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(case,[(str(args.task),b) for b in range(3)]))
    trial.base.write(trial.out(args.task)/'ranking_review.json',dict(status='PASS_RECOMPUTATION',rows=rows,
        mean_change=statistics.mean(r['change'] for r in rows),
        boundary='Matched previous validation states and original frozen-policy continuation labels; off-policy diagnostic, not current-policy Q or a model-selection criterion'))
    print('Matched ranking changes',[r['change'] for r in rows])
