"""Pure statistics for the incremental value of same-slot plan conditioning."""
from collections import defaultdict
import math
import statistics
from utils.queued_service_diagnostics import explained_variance


def conditioning_value(matrix):
    if not matrix or not matrix[0] or any(len(row)!=len(matrix[0]) for row in matrix):
        raise ValueError('nonempty rectangular payoff matrix required')
    if any(not math.isfinite(x) for row in matrix for x in row):raise ValueError('finite payoffs required')
    r=len(matrix);k=len(matrix[0]);rowmean=[statistics.mean(row) for row in matrix]
    colmean=[statistics.mean(matrix[i][j] for i in range(r)) for j in range(k)];grand=statistics.mean(rowmean)
    conditional=statistics.mean(max(row) for row in matrix);common=max(colmean)
    common_optimal=set(range(k))
    for row in matrix:common_optimal&={j for j,value in enumerate(row) if value==max(row)}
    return dict(conditional_best=conditional,common_best=common,information_gain=max(0.,conditional-common),
        common_optimum_exists=bool(common_optimal),routing_variance=statistics.mean((x-grand)**2 for x in rowmean),
        motion_variance=statistics.mean((x-grand)**2 for x in colmean),
        interaction_variance=statistics.mean((matrix[i][j]-rowmean[i]-colmean[j]+grand)**2 for i in range(r) for j in range(k)),
        total_variance=statistics.mean((x-grand)**2 for row in matrix for x in row))


def value_fit_by_phase(target,prediction,keys):
    if not target or len(target)!=len(prediction) or len(target)!=len(keys):raise ValueError('aligned values and phases required')
    groups=defaultdict(list)
    for i,key in enumerate(keys):groups[key].append(i)
    residual_target=[];residual_prediction=[];baseline_errors=[]
    for ids in groups.values():
        if len(ids)<2:continue
        mean_t=statistics.mean(target[i] for i in ids);mean_p=statistics.mean(prediction[i] for i in ids)
        for i in ids:
            residual_target.append(target[i]-mean_t);residual_prediction.append(prediction[i]-mean_p)
            baseline=(sum(target[j] for j in ids)-target[i])/(len(ids)-1)
            baseline_errors.append((baseline-target[i])**2)
    return dict(overall_ev=explained_variance(target,prediction),
        within_phase_ev=explained_variance(residual_target,residual_prediction) if residual_target else None,
        rmse=math.sqrt(statistics.mean((a-b)**2 for a,b in zip(target,prediction))),
        other_episode_time_baseline_rmse=math.sqrt(statistics.mean(baseline_errors)) if baseline_errors else None,
        within_phase_states=len(residual_target),total_states=len(target),phases=len(groups))
