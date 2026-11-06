"""Read-only telemetry, paired first-action factors and remainder accounting."""
import numpy as np

REVISION = 'queued_recovery_audit_20261104'
TIMES = (80,100,110,150)
BRANCHES = ('anchor','allocation','motion','full')


def action_factors(anchor, proposed, seeds):
    anchor,proposed = (np.asarray(x,float) for x in (anchor,proposed))
    if anchor.shape != proposed.shape or anchor.ndim != 2 or not anchor.size:
        raise ValueError('compatible candidate distributions required')
    for probability in (anchor,proposed):
        if not np.isfinite(probability).all() or np.any(probability < 0) or not np.isclose(probability.sum(),1.):
            raise ValueError('normalized finite candidate probabilities required')
    mappings = []
    for seed in seeds:
        # Reinitialize so both categorical draws use the same uniform variate.
        first = np.unravel_index(np.random.default_rng(seed).choice(anchor.size,p=anchor.ravel()),anchor.shape)
        second = np.unravel_index(np.random.default_rng(seed).choice(proposed.size,p=proposed.ravel()),proposed.shape)
        mappings.append({name:tuple(map(int,pair)) for name,pair in zip(BRANCHES,
            (first,(second[0],first[1]),(first[0],second[1]),second))})
    unique = sorted({pair for draw in mappings for pair in draw.values()})
    return mappings,unique


def telemetry(engine):
    positions = np.asarray(engine.positions,float)
    covered = [int(uid) for uid,xy in sorted(engine.ues.items())
               if np.any(np.linalg.norm(positions-np.asarray(xy),axis=1) <= engine.p.coverage+1e-9)]
    counts = {stage:sum(x.stage == stage for x in engine.tasks.values()) for stage in ('waiting','upload','transfer','compute')}
    work = [sum(x.remaining_cycles for x in engine.tasks.values() if x.target == node)/capacity
            for node,capacity in enumerate(engine.p.capacity)]
    return dict(time=float(engine.t),positions=positions.tolist(),covered_ues=covered,
                inventory=counts,node_work_seconds=work)


def remainder_cohorts(tasks, root_ids, now):
    tasks = list(tasks);root_ids = set(root_ids)
    groups = dict(root=[x for x in tasks if x.uid in root_ids],
        future=[x for x in tasks if x.uid not in root_ids],
        future_late=[x for x in tasks if x.uid not in root_ids and x.arrival >= now+3],
        future_post=[x for x in tasks if x.uid not in root_ids and x.arrival >= 100])
    return {name:dict(generated=len(xs),admitted=sum(x.source is not None for x in xs),
        success=sum(x.stage == 'completed' for x in xs),expired=sum(x.stage == 'expired' for x in xs))
        for name,xs in groups.items()}


def factorial_effects(values):
    a,allocation,motion,full = (float(values[name]) for name in BRANCHES)
    return dict(total=full-a,allocation=allocation-a,motion=motion-a,
                interaction=full-allocation-motion+a)


def gain_layers(predicted, public, oracle_public, realized):
    return dict(predicted=float(predicted),public=float(public),oracle_public=float(oracle_public),
        realized=float(realized),approximation_error=float(predicted-public),
        prediction_error=float(public-oracle_public),visibility_error=float(oracle_public-realized))
