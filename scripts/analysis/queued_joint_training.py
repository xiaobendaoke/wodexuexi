"""Server-only runner for upper-lower joint neural training and unconfounded ablation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

if __name__ == '__main__':
    for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[env_key] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from marl_models.queued_joint_training import (
    compose_decomposed_actor,
    fit_joint_actor,
    initialize_joint_actor,
)
from scripts.analysis import queued_quality_amortization as qa_exp
from scripts.analysis import queued_waiting_rescue as prototype
from scripts.analysis.queued_waiting_rescue import quality_details
from scripts.analysis.review_queued_joint_training import (
    CONFIRMATION_WORKLOADS,
    PILOT_WORKLOADS,
    checkpoint_curve_review,
    stage_review,
)
from utils.queued_arrival_adaptation import RateHistory
from utils.queued_joint_training import (
    ALL_EVAL_ARMS,
    REVISION,
    TRAINING_ARMS,
    build_joint_bank,
    compute_offline_teacher_targets,
    extract_actor_row,
    mode_for_arm,
    neural_actor_step,
    prepare_slot_base,
)
from utils.queued_taskset_closed_loop import cohort_statistics

t = prototype.t

# Dedicated training and validation workloads (strictly disjoint from Pilot 951000-951003
# and blind Confirmation 953000-953007)
TRAIN_R1_WORKLOADS = (940000, 940001, 940002, 940003, 940004, 940005)
TRAIN_R2_WORKLOADS = (940006, 940007, 940008, 940009, 940010, 940011)
VAL_WORKLOADS = (941000, 941001, 941002)
SAMPLE_STRIDE = 4
CHECKPOINT_EPOCHS = (0, 1, 2, 4, 8, 16, 32, 64)


def out(root):
    return Path(root) / 'results/chapter5_dynamic_service_20261028' / REVISION


def val(root):
    return Path(root) / 'output/chapter5_dynamic_service_20261028/validation' / REVISION


def model_path(root, arm, parent):
    return out(root) / 'models' / arm / f'{parent}.pkl.gz'


def ckpt_model_path(root, epoch, parent):
    return out(root) / 'models' / 'checkpoints' / f'epoch_{epoch}' / f'{parent}.pkl.gz'


def collect_trajectory_worker(payload):
    """Roll out one training/validation episode and collect shared state & teacher targets."""
    root, split, round_idx, parent, workload = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2)

    r1_model = None
    if round_idx == 2:
        r1_path = out(root) / 'models_r1' / 'joint_trained' / f'{parent}.pkl.gz'
        r1_model = prototype.previous.saved(r1_path)

    samples = {arm: [] for arm in TRAINING_ARMS}
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        assert now <= 204
        engine.add(tape.slot(now))
        engine.admit()
        obs = engine.observe()
        temporal.update(obs)
        base_state = prepare_slot_base(
            root, prototype, proposer, score_model, obs, adapter, parent, workload, 0, now
        )

        if now < 200 and (now % SAMPLE_STRIDE == 0 or (40 <= now < 100 and now % 2 == 0)):
            b_shared = build_joint_bank(
                root, prototype, proposer, score_model, obs, adapter, temporal,
                parent, workload, 0, now,
                include_safe_plan=True, rescue_mode='unguarded', precomputed_base=base_state,
            )
            row_shared = extract_actor_row(obs, b_shared, adapter, temporal)
            targets_all = compute_offline_teacher_targets(obs, b_shared, adapter)
            compact_row = {k: row_shared[k] for k in ('a', 'm', 'context', 'prior', 'motion_prior')}
            for arm in TRAINING_ARMS:
                mode = mode_for_arm(arm)
                target_dist, val_mat = targets_all[mode]
                samples[arm].append(dict(
                    time=now,
                    workload=workload,
                    parent=parent,
                    row=compact_row,
                    target=target_dist,
                    value=val_mat,
                ))

        if round_idx == 2 and r1_model is not None:
            step = neural_actor_step(
                root, prototype, proposer, score_model, obs, adapter, temporal,
                'joint_trained', r1_model, parent, workload, 0, now,
                precomputed_base=base_state,
            )
            engine.assign(step['plan'])
            engine.advance(step['motion'])
        else:
            # Round 1: 50% uncoordinated_init + 50% uniform exploration over shared bank
            mix_rng = np.random.default_rng(t.seed(REVISION, 'explore_mix', parent, workload, now))
            if mix_rng.random() < 0.5:
                step = neural_actor_step(
                    root, prototype, proposer, score_model, obs, adapter, temporal,
                    'uncoordinated_init', None, parent, workload, 0, now,
                    precomputed_base=base_state,
                )
                engine.assign(step['plan'])
                engine.advance(step['motion'])
            else:
                b_exp = build_joint_bank(
                    root, prototype, proposer, score_model, obs, adapter, temporal,
                    parent, workload, 0, now,
                    include_safe_plan=True, rescue_mode='unguarded', precomputed_base=base_state,
                )
                p_idx = int(mix_rng.integers(0, len(b_exp['plans'])))
                m_idx = int(mix_rng.integers(0, len(b_exp['actions'])))
                engine.assign(b_exp['plans'][p_idx])
                engine.advance(b_exp['actions'][m_idx])

    assert not engine.tasks and engine.audit()['status'] == 'PASS'
    dest = out(root) / 'datasets' / f'{split}_r{round_idx}_{parent}_{workload}.pkl.gz'
    prototype.previous.dump(dest, samples)
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(split=split, round=round_idx, parent=parent, workload=workload, states=len(samples['joint_trained'])),
    )


def fit_lineage_worker(payload):
    root, round_idx, parent, arm = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    mode = mode_for_arm(arm)

    train_samples = []
    for w in TRAIN_R1_WORKLOADS:
        data = prototype.previous.saved(out(root) / 'datasets' / f'train_r1_{parent}_{w}.pkl.gz')
        train_samples.extend(data[arm])
    if round_idx == 2:
        for w in TRAIN_R2_WORKLOADS:
            data = prototype.previous.saved(out(root) / 'datasets' / f'train_r2_{parent}_{w}.pkl.gz')
            train_samples.extend(data[arm])

    val_samples = []
    for w in VAL_WORKLOADS:
        data = prototype.previous.saved(out(root) / 'datasets' / f'val_r1_{parent}_{w}.pkl.gz')
        val_samples.extend(data[arm])

    train_rows = [s['row'] for s in train_samples]
    train_targets = [s['target'] for s in train_samples]
    train_values = [s['value'] for s in train_samples]
    val_rows = [s['row'] for s in val_samples]
    val_targets = [s['target'] for s in val_samples]
    val_values = [s['value'] for s in val_samples]

    if round_idx == 1:
        init_model = initialize_joint_actor(train_rows[0], hidden=48, seed=t.seed(REVISION, 'init', parent, arm))
        trained, info, ckpts = fit_joint_actor(
            train_rows, train_targets, train_values,
            val_rows, val_targets, val_values,
            init_model, mode=mode, epochs=32, batch_size=16,
            learning_rate=0.008, seed=t.seed(REVISION, 'fit_r1', parent, arm),
            checkpoint_epochs=(0, 1, 2, 4, 8, 16, 32), use_warmup=True,
        )
        prototype.previous.dump(out(root) / 'models_r1' / arm / f'{parent}.pkl.gz', trained)
        if arm == 'joint_trained':
            for ep, m_ep in ckpts.items():
                prototype.previous.dump(ckpt_model_path(root, ep, parent), m_ep)
    else:
        r1_model = prototype.previous.saved(out(root) / 'models_r1' / arm / f'{parent}.pkl.gz')
        trained, info, ckpts = fit_joint_actor(
            train_rows, train_targets, train_values,
            val_rows, val_targets, val_values,
            r1_model, mode=mode, epochs=32, batch_size=16,
            learning_rate=0.010, seed=t.seed(REVISION, 'fit_r2', parent, arm),
            checkpoint_epochs=(0, 16, 32), use_warmup=False,
        )
        for c in info['curve']:
            c['epoch'] += 32
        prototype.previous.dump(model_path(root, arm, parent), trained)
        if arm == 'joint_trained':
            prototype.previous.dump(ckpt_model_path(root, 64, parent), ckpts[32])

    prototype.write(
        out(root) / 'curves' / f'r{round_idx}_{arm}_{parent}.json',
        dict(round=round_idx, arm=arm, parent=parent, info=info),
    )
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(round=round_idx, arm=arm, parent=parent, final_train_kl=info['curve'][-1]['train']['kl']),
    )


def assemble_composed_models(root):
    """Assemble `decomposed_composed` directly from `lower_only_trained` + `upper_only_trained`."""
    root = Path(root)
    for parent in range(3):
        lower_m = prototype.previous.saved(model_path(root, 'lower_only_trained', parent))
        upper_m = prototype.previous.saved(model_path(root, 'upper_only_trained', parent))
        composed = compose_decomposed_actor(lower_m, upper_m)
        prototype.previous.dump(model_path(root, 'decomposed_composed', parent), composed)


def episode_worker(payload):
    root, stage, arm, parent, workload, replica, ckpt_epoch = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    allowed = PILOT_WORKLOADS if stage in ('pilot', 'pilot_ckpt') else CONFIRMATION_WORKLOADS
    assert workload in allowed, (stage, workload)
    if stage == 'pilot_ckpt':
        dest = out(root) / 'pilot_checkpoints' / f'epoch_{ckpt_epoch}' / f'{parent}_{workload}_{replica}.json'
    else:
        dest = out(root) / stage / arm / f'{parent}_{workload}_{replica}.json'
    if dest.exists():
        existing = json.loads(dest.read_text(encoding='utf-8'))
        return dict(
            seconds=0.0,
            cpu_seconds=0.0,
            result=dict(stage=stage, arm=arm, ckpt_epoch=ckpt_epoch, success=existing['counts'].get('success', 0)),
        )

    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2)

    if ckpt_epoch is not None:
        model = prototype.previous.saved(ckpt_model_path(root, ckpt_epoch, parent))
    elif arm in TRAINING_ARMS or arm == 'decomposed_composed':
        model = prototype.previous.saved(model_path(root, arm, parent))
    elif arm == 'joint_neural_mpc':
        model = prototype.previous.saved(model_path(root, 'joint_trained', parent))
    else:
        model = None

    successes = []
    trajectory = []
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        assert now <= 204
        engine.add(tape.slot(now))
        admitted = engine.admit()
        obs = engine.observe()
        temporal.update(obs)
        step = neural_actor_step(
            root, prototype, proposer, score_model, obs, adapter, temporal,
            arm, model, parent, workload, replica, now,
        )
        trajectory.append(dict(
            time=now, positions=list(engine.positions), motion=step['motion'],
            admitted=len(admitted), original_candidates=step['original_candidates'],
            candidates=step['candidates'], selected_new=step['selected_new'],
            metadata=step['metadata'], decision_seconds=step['decision_seconds'],
            deterministic_engine_steps=step['deterministic_engine_steps'],
            forecast_calls=step['forecast_calls'],
        ))
        engine.assign(step['plan'])
        successes.append(engine.advance(step['motion'])['success'])

    audit = engine.audit()
    counts = dict(engine.counts)
    assert not engine.tasks and audit['status'] == 'PASS'
    assert sum(successes) == counts.get('success', 0)
    assert counts['generated'] == counts.get('success', 0) + counts.get('expired', 0)
    record = dict(
        stage=stage, arm=arm, parent=parent, workload=workload, replica=replica,
        ckpt_epoch=ckpt_epoch, counts=counts,
        cohorts=cohort_statistics(engine.finished.values()),
        quality=quality_details(list(engine.finished.values())),
        successes=successes, trajectory=trajectory, fingerprint=tape.fingerprint, audit=audit,
    )
    if stage == 'pilot_ckpt':
        dest = out(root) / 'pilot_checkpoints' / f'epoch_{ckpt_epoch}' / f'{parent}_{workload}_{replica}.json'
    else:
        dest = out(root) / stage / arm / f'{parent}_{workload}_{replica}.json'
    prototype.write(dest, record)
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(stage=stage, arm=arm, ckpt_epoch=ckpt_epoch, success=counts.get('success', 0)),
    )


def dispatch_fn(root, label, worker_fn, jobs):
    cpus = sorted(os.sched_getaffinity(0))
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(
        max_workers=min(len(cpus), len(jobs)),
        mp_context=mp.get_context('spawn'),
        initializer=prototype.previous.validation.init_worker,
        initargs=(cpus,),
    ) as pool:
        futures = [pool.submit(worker_fn, (str(root),) + tuple(job)) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 12 == 0 or len(records) == len(jobs):
                print(label, len(records), '/', len(jobs), flush=True)
    seconds = time.perf_counter() - started
    cpu = sum(row['cpu_seconds'] for row in records)
    prototype.write(val(root) / (label + '_complete.json'), dict(
        tasks=len(jobs), wall_seconds=seconds,
        worker_cpu_seconds=cpu, equivalent_cpus=cpu / max(seconds, 1e-9),
        utilization=cpu / max(seconds, 1e-9) / len(cpus), records=records,
    ))


def copy_ckpt_from_arm(root, arm, epoch):
    src_folder = out(root) / 'pilot' / arm
    for source in sorted(src_folder.glob('*.json')):
        record = json.loads(source.read_text(encoding='utf-8'))
        record['ckpt_epoch'] = epoch
        dest = out(root) / 'pilot_checkpoints' / f'epoch_{epoch}' / source.name
        prototype.write(dest, record)


def aggregate_training_curves(root):
    curves = {}
    for arm in TRAINING_ARMS:
        arm_epochs = {}
        for parent in range(3):
            r1 = json.loads((out(root) / 'curves' / f'r1_{arm}_{parent}.json').read_text(encoding='utf-8'))['info']['curve']
            r2 = json.loads((out(root) / 'curves' / f'r2_{arm}_{parent}.json').read_text(encoding='utf-8'))['info']['curve']
            combined = r1 + [c for c in r2 if c['epoch'] > 32]
            for item in combined:
                ep = item['epoch']
                arm_epochs.setdefault(ep, []).append(item)
        summary_list = []
        for ep in sorted(arm_epochs):
            items = arm_epochs[ep]
            summary_list.append(dict(
                epoch=ep,
                train={
                    k: float(np.mean([x['train'][k] for x in items]))
                    for k in ('kl', 'top1_regret', 'optimal_probability', 'expected_value', 'pairwise_accuracy')
                },
                validation={
                    k: float(np.mean([x['validation'][k] for x in items]))
                    for k in ('kl', 'top1_regret', 'optimal_probability', 'expected_value', 'pairwise_accuracy')
                },
            ))
        curves[arm] = summary_list
    prototype.write(out(root) / 'training_curves.json', curves)
    return curves


def protected(root):
    return qa_exp.protected(root) | {
        str(path.relative_to(root)): prototype.t.sha(path)
        for path in qa_exp.out(root).rglob('*') if path.is_file()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert str(root).startswith('/home/PengYanghan/Lunwen/chapter5-dynamic-service-20261028')

    before = protected(root)
    if not (out(root) / 'protocol.json').exists():
        prototype.write(out(root) / 'protocol.json', dict(
            revision=REVISION,
            training_arms=list(TRAINING_ARMS),
            evaluated_arms=list(ALL_EVAL_ARMS),
            train_r1_workloads=list(TRAIN_R1_WORKLOADS),
            train_r2_workloads=list(TRAIN_R2_WORKLOADS),
            val_workloads=list(VAL_WORKLOADS),
            pilot_workloads=list(PILOT_WORKLOADS),
            confirmation_workloads=list(CONFIRMATION_WORKLOADS),
            checkpoint_epochs=list(CHECKPOINT_EPOCHS),
            reused_baselines={},
            online_simulation_audit='All neural arms verified with deterministic_engine_steps == 0 and forecast_calls == 0.',
        ))
    if not (val(root) / 'historical_hashes_before.json').exists():
        prototype.write(val(root) / 'historical_hashes_before.json', before)

    if not (val(root) / 'collect_r1_complete.json').exists():
        # 1. Round 1 trajectory & teacher target collection (18 train + 9 val = 27 episodes)
        r1_collect_jobs = (
            [('train', 1, p, w) for p in range(3) for w in TRAIN_R1_WORKLOADS]
            + [('val', 1, p, w) for p in range(3) for w in VAL_WORKLOADS]
        )
        dispatch_fn(root, 'collect_r1', collect_trajectory_worker, r1_collect_jobs)

    if not (val(root) / 'fit_r1_complete.json').exists():
        # 2. Round 1 neural fitting (3 parents x 4 training arms = 12 fits)
        r1_fit_jobs = [(1, p, arm) for p in range(3) for arm in TRAINING_ARMS]
        dispatch_fn(root, 'fit_r1', fit_lineage_worker, r1_fit_jobs)

    if not (val(root) / 'collect_r2_complete.json').exists():
        # 3. Round 2 on-policy trajectory collection under R1 joint_trained (18 episodes)
        r2_collect_jobs = [('train', 2, p, w) for p in range(3) for w in TRAIN_R2_WORKLOADS]
        dispatch_fn(root, 'collect_r2', collect_trajectory_worker, r2_collect_jobs)

    if not (val(root) / 'fit_r2_complete.json').exists():
        # 4. Round 2 neural fitting (3 parents x 4 training arms = 12 fits) + assemble decomposed_composed
        r2_fit_jobs = [(2, p, arm) for p in range(3) for arm in TRAINING_ARMS]
        dispatch_fn(root, 'fit_r2', fit_lineage_worker, r2_fit_jobs)
        assemble_composed_models(root)
        aggregate_training_curves(root)

    # 5. Pilot evaluation (all 9 arms freshly evaluated on 951000-951003, 108 episodes)
    if not (val(root) / 'pilot_complete.json').exists():
        pilot_jobs = [
            ('pilot', arm, p, w, 0, None)
            for arm in ALL_EVAL_ARMS
            for p in range(3)
            for w in PILOT_WORKLOADS
        ]
        ckpt_jobs = [
            ('pilot_ckpt', 'joint_trained', p, w, 0, ep)
            for ep in (1, 2, 4, 8, 16, 32)
            for p in range(3)
            for w in PILOT_WORKLOADS
        ]
        dispatch_fn(root, 'pilot', episode_worker, pilot_jobs + ckpt_jobs)
        copy_ckpt_from_arm(root, 'uncoordinated_init', 0)
        copy_ckpt_from_arm(root, 'joint_trained', 64)

    pilot = stage_review(root, 'pilot', out(root))
    pilot['checkpoint_curve'] = checkpoint_curve_review(out(root))
    for p_path in (out(root) / 'pilot_analysis.json', val(root) / 'pilot_gate.json'):
        if p_path.exists():
            p_path.unlink()
        prototype.write(p_path, pilot)

    confirmation = dict(triggered=False, reason='pilot gate failed')
    if pilot['gate_passed']:
        # 6. Blind Confirmation evaluation on brand-new 953000-953007 (9 arms x 48 = 432 episodes)
        if not (val(root) / 'confirmation_complete.json').exists():
            conf_jobs = [
                ('confirmation', arm, p, w, r, None)
                for arm in ALL_EVAL_ARMS
                for p in range(3)
                for w in CONFIRMATION_WORKLOADS
                for r in (0, 1)
            ]
            dispatch_fn(root, 'confirmation', episode_worker, conf_jobs)
        confirmation = stage_review(root, 'confirmation', out(root))
        for c_path in (out(root) / 'confirmation_analysis.json', val(root) / 'confirmation_gate.json'):
            if c_path.exists():
                c_path.unlink()
            prototype.write(c_path, confirmation)

    after = protected(root)
    assert before == after
    rc_path = val(root) / 'run_complete.json'
    if rc_path.exists():
        rc_path.unlink()
    prototype.write(rc_path, dict(
        status='PASS',
        revision=REVISION,
        pilot_gate_passed=pilot['gate_passed'],
        confirmation_triggered=bool(pilot['gate_passed']),
        confirmation_gate_passed=confirmation.get('gate_passed'),
        historical_files=len(after),
        historical_artifacts_unchanged=True,
    ))


if __name__ == '__main__':
    main()
