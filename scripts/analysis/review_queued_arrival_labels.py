"""Read-only corrected teacher and learned-score error checks on old recovery roots."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_arrival_adaptation as experiment
from scripts.analysis.review_queued_recovery_audit import paired_summary
from utils.queued_score_control import controlled_policy


def review(root):
    root=Path(root);folder=experiment.out(root);records=[]
    for path in sorted((folder/'diagnostic_labels_compact').glob('*/*.pkl.gz')):
        labels=experiment.saved(path)
        old_root=experiment.load(experiment.recovery.out(root)/'roots'/path.relative_to(folder/'diagnostic_labels_compact').with_suffix('').with_suffix('.json'))
        diagnostic=experiment.saved(folder/'recovery_diagnostics'/path.relative_to(folder/'diagnostic_labels_compact'))
        anchor=np.array(old_root['anchor']);truth=labels['realized'];values={}
        for name in ('slow','fast'):
            teacher=labels[name].mean(0)
            probability,_=controlled_policy(anchor,teacher,old_root['control']['pressure'],'gated')
            best=teacher>=teacher.max()-1e-12
            i,j=np.triu_indices(teacher.size,1);dq=teacher.ravel()[i]-teacher.ravel()[j];dy=truth.ravel()[i]-truth.ravel()[j]
            informative=abs(dy)>1e-12;weight=abs(dy[informative])
            correct=np.where(abs(dq[informative])<=1e-12,.5,np.sign(dq[informative])==np.sign(dy[informative]))
            values[name+'/teacher_true_gain']=float(np.sum((probability-anchor)*truth))
            values[name+'/teacher_regret']=float(truth.max()-truth[best].mean())
            values[name+'/teacher_weighted_accuracy']=float(np.sum(weight*correct)/weight.sum()) if weight.size else .5
            gains=diagnostic['methods'][name+'_rates_joint']['gains']
            values[name+'/score_approximation_error']=gains['predicted']-gains[name]
            values[name+'/forecast_gain_error']=gains[name]-gains['realized']
        records.append(dict(run=labels['run'],workload=labels['workload'],time=labels['time'],behavior=labels['behavior'],**values))
    assert len(records)==96
    result=dict(records=records,groups={str(now):{key:paired_summary(records,key,now) for key in values}
        for now in experiment.recovery.TIMES},scope='Posthoc diagnostic on two old workload blocks, fixed roots and public H3/EDF labels. No new model selection or training.')
    experiment.write(folder/'corrected_label_diagnostics.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True)
    review(parser.parse_args().task_root)
