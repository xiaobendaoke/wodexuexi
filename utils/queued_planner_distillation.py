"""Public-planner imitation targets and workload-separated development design."""
import numpy as np

REVISION = 'queued_planner_distillation_20261103'
ARMS = ('mlp_joint', 'mlp_decomposed', 'linear_joint')
BASELINES = ('initial', 'plan_short')


def specification():
    runs = []
    for i in range(6):
        runs.append(dict(id=i, parent=i//2, repeat=i % 2,
                         training=[list(range(940000+i*12+j*6, 940000+i*12+(j+1)*6)) for j in range(2)],
                         holdout=[list(range(941000+i*4+j*2, 941000+i*4+(j+1)*2)) for j in range(2)]))
    return dict(revision=REVISION, runs=runs, arms=list(ARMS), baselines=list(BASELINES),
                rounds=2, hidden=32, epochs=4, batch_size=8, learning_rate=.003,
                label_times=list(range(0, 200, 2)), evaluation_workloads=list(range(911000, 911008)),
                action_replicas=[0, 1], horizon=3,
                maximum_training_episodes=216, maximum_holdout_episodes=72,
                maximum_evaluation_episodes=384, maximum_physical_episodes=672,
                scope='Development only; evaluation workloads already read. No formal access, GO decision or automatic launch.',
                continuation='Candidate action, then two public EDF/hover slots. Teacher is a fixed-reference short forecast, not current-policy Q.',
                independent_unit='New residual training repeat conditional on one of three inherited fixed parents; no new A2 training.')


def validate_workload(spec, split, run_id, workload):
    if split == 'evaluation':
        allowed = spec['evaluation_workloads']
    elif split in ('training', 'holdout'):
        run = next((r for r in spec['runs'] if r['id'] == run_id), None)
        if run is None:
            raise ValueError('unknown residual training repeat')
        allowed = [w for group in run[split] for w in group]
    else:
        raise ValueError('unknown data split')
    if workload not in allowed:
        raise ValueError('workload outside reserved development split')


def validate_episode(spec, arm, run_id, parent, workload, replica, version, split):
    validate_workload(spec, split, run_id, workload)
    if arm not in ARMS+BASELINES or parent not in range(3) or replica not in spec['action_replicas']:
        raise ValueError('invalid development arm, parent or action replica')
    if arm in BASELINES:
        if split != 'evaluation' or run_id != -1 or version != 0:
            raise ValueError('baselines are unique evaluation-only physical episodes')
        return
    run = next((r for r in spec['runs'] if r['id'] == run_id), None)
    if run is None or run['parent'] != parent:
        raise ValueError('actor does not belong to this fixed parent')
    if split == 'evaluation':
        if version != spec['rounds']:
            raise ValueError('only fixed final actors may be evaluated')
    elif (version not in range(spec['rounds']) or replica != 0
          or workload not in run[split][version]):
        raise ValueError('wrong aggregation round or unbudgeted action replica')


def _scores_prior(scores, prior):
    q, prior = np.asarray(scores, float), np.asarray(prior, float)
    if (q.ndim != 2 or not q.size or prior.shape != (q.shape[0],)
            or not np.isfinite(q).all() or not np.isfinite(prior).all()
            or np.any(prior <= 0) or not np.isclose(prior.sum(), 1.)):
        raise ValueError('finite nonempty scores and positive normalized plan prior required')
    return q, prior


def planner_target(scores, prior):
    """Exact best-set teacher; ties follow the INITIAL restricted prior.

    Independent of the student's current probability or a candidate index.
    This bypasses the old KL-constrained E-step; it has no improvement theorem.
    """
    q, prior = _scores_prior(scores, prior)
    best = q >= q.max()-1e-12
    target = np.broadcast_to(prior[:, None]/q.shape[1], q.shape)*best
    return target/target.sum()


def imitation_metrics(scores, probabilities, prior):
    q, prior = _scores_prior(scores, prior)
    p = np.asarray(probabilities, float)
    if (p.shape != q.shape or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1.)):
        raise ValueError('normalized finite actor probabilities required')
    target = planner_target(q, prior)
    best = q >= q.max()-1e-12
    initial = np.broadcast_to(prior[:, None]/q.shape[1], q.shape)
    expected = float(np.sum(p*q))
    base = float(np.sum(initial*q))
    available = float(q.max()-base)
    active = target > 0
    kl = float(np.sum(target[active]*(np.log(target[active])-np.log(p[active].clip(1e-300)))))
    return dict(teacher_kl=kl, score_regret=float(q.max()-expected),
                optimal_probability=float(p[best].sum()), all_tied=bool(best.all()),
                available_planning_gain=available, expected_score_gain=expected-base,
                retained_planning_gain=(expected-base)/available if available > 1e-12 else None)


def check_rows(training, holdout=()):
    if not training:
        raise ValueError('nonempty training rows required')
    for rows, split in ((training, 'training'), (holdout, 'holdout')):
        if any(r.get('split') != split for r in rows):
            raise ValueError('row split mismatch; holdout cannot enter optimization')
    if {r['workload'] for r in training} & {r['workload'] for r in holdout}:
        raise ValueError('entire workloads must be separated, not individual states')


def summarize_imitation(rows, model, mode='joint'):
    from marl_models.queued_planner_distillation import policy
    if not rows:
        raise ValueError('nonempty diagnostic split required')
    records = [imitation_metrics(r['q'], policy(r, model, mode), r['prior']) for r in rows]
    # Report all tied states. The gain ratio is a ratio of sums, not the mean
    # of unstable state-wise ratios; no low-opportunity state is discarded.
    available = sum(z['available_planning_gain'] for z in records)
    return dict(states=len(rows), workloads=len({r['workload'] for r in rows}),
                teacher_kl=float(np.mean([z['teacher_kl'] for z in records])),
                score_regret=float(np.mean([z['score_regret'] for z in records])),
                optimal_probability=float(np.mean([z['optimal_probability'] for z in records])),
                all_tied_states=sum(z['all_tied'] for z in records),
                retained_planning_gain=sum(z['expected_score_gain'] for z in records)/available if available > 1e-12 else None,
                scope='Fixed public teacher on saved states; not actual DSR or a formal GO gate.')
