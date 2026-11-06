"""Server-only runner for true environment-reward Joint RL (PPO-GAE) and unconfounded causal audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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

from environment.queued_service import Engine
from marl_models.queued_joint_rl import (
    compose_decomposed_rl_actor,
    fit_clean_distilled_actor,
    initialize_rl_actor_critic,
    ppo_update_actor_critic,
    warmstart_from_distilled_actor,
)
from scripts.analysis import queued_quality_amortization as qa_exp
from scripts.analysis import queued_waiting_rescue as prototype
from scripts.analysis.queued_waiting_rescue import quality_details
from scripts.analysis.review_queued_joint_rl import (
    CONFIRMATION_WORKLOADS,
    PILOT_WORKLOADS,
    PRIMARY_COMPARISON_PAIRS,
    RL_SOURCE_FILES,
    audit_rl_rollouts,
    audit_true_decomposed_independence,
    rl_checkpoint_curve_review,
    stage_review,
    verify_freeze_manifest,
)
from utils.queued_arrival_adaptation import RateHistory
from utils.queued_joint_rl import (
    RL_EVAL_ARMS,
    RL_REVISION,
    RL_TRAINING_ARMS,
    _release_time_tandem_queue_eval,
    collect_rl_episode,
    extract_clean_actor_row,
    rl_actor_step,
    rl_mode_for_arm,
)
from utils.queued_joint_training import (
    _lindley_plan_motion_eval,
    build_joint_bank,
    compute_offline_teacher_targets,
    prepare_slot_base,
)
from utils.queued_quality_amortization import safe_assignments
from utils.queued_taskset_closed_loop import cohort_statistics

t = prototype.t

# Dedicated training and validation workloads (strictly disjoint from Pilot 951000-951003
# and blind Confirmation 955000-955007)
TRAIN_R1_WORKLOADS = (940000, 940001, 940002, 940003, 940004, 940005)
TRAIN_R2_WORKLOADS = (940006, 940007, 940008, 940009, 940010, 940011)
VAL_WORKLOADS = (941000, 941001, 941002)
SAMPLE_STRIDE = 4
PPO_ITERATIONS = (1, 2, 3, 4)
ROLLOUT_REPS = (0, 1)
DISTILLED_ARMS = ('decomposed_distilled', 'joint_no_expectation_distilled', 'joint_distilled')
SCRATCH_RL_ARMS = (
    'lower_only_rl',
    'upper_only_rl',
    'decomposed_rl',
    'joint_no_expectation_rl',
    'joint_rl',
)
FINETUNED_RL_ARMS = (
    'decomposed_rl_finetuned',
    'joint_no_expectation_rl_finetuned',
    'joint_rl_finetuned',
)


def utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def out(root):
    return Path(root) / 'results/chapter5_dynamic_service_20261028' / RL_REVISION


def val(root):
    return Path(root) / 'output/chapter5_dynamic_service_20261028/validation' / RL_REVISION


def model_path(root, arm, parent):
    return out(root) / 'models' / arm / f'{parent}.pkl.gz'


def iter_model_path(root, iteration, arm, parent):
    return out(root) / 'models_iter' / f'iter_{iteration}' / arm / f'{parent}.pkl.gz'


def ckpt_model_path(root, iteration, parent):
    return out(root) / 'models' / 'checkpoints' / f'iter_{iteration}' / f'{parent}.pkl.gz'


def collect_clean_distill_worker(payload):
    """Collect clean physical states and teacher targets ONLY for the two distilled baselines."""
    root, split, round_idx, parent, workload = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    dest = out(root) / 'distill_datasets' / f'{split}_r{round_idx}_{parent}_{workload}.pkl.gz'
    if dest.exists():
        data = prototype.previous.saved(dest)
        return dict(
            seconds=0.0,
            cpu_seconds=0.0,
            result=dict(
                split=split, round=round_idx, parent=parent, workload=workload,
                states=len(data['joint_distilled']),
            ),
        )

    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2)

    r1_model = None
    if round_idx == 2:
        r1_path = out(root) / 'distill_models_r1' / 'joint_distilled' / f'{parent}.pkl.gz'
        r1_model = prototype.previous.saved(r1_path)

    samples = {arm: [] for arm in DISTILLED_ARMS}
    approx_comparisons = []

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
            row_clean = extract_clean_actor_row(obs, b_shared, adapter, temporal)
            targets_all = compute_offline_teacher_targets(obs, b_shared, adapter)
            compact_row = {k: row_clean[k] for k in ('a', 'm', 'context', 'prior', 'motion_prior')}
            for arm in DISTILLED_ARMS:
                t_mode = 'decomposed' if arm == 'decomposed_distilled' else 'joint'
                target_dist, _ = targets_all[t_mode]
                samples[arm].append(dict(
                    time=now,
                    workload=workload,
                    parent=parent,
                    row=compact_row,
                    target=target_dist,
                ))

            # Quantify tandem-queue approximation vs old static Lindley on validation states
            if split == 'val' and now % 16 == 0:
                admitted = sorted(
                    [z for z in obs['tasks'] if z['source'] is not None],
                    key=lambda z: (z['deadline'], z['uid']),
                )
                waiting = sorted(
                    [z for z in obs['tasks'] if z['source'] is None],
                    key=lambda z: (z['deadline'], z['uid']),
                )
                zero = [(0.0, 0.0)] * len(obs['positions'])
                for pi, plan in enumerate(b_shared['plans'][:2]):
                    for mi, motion in enumerate(b_shared['actions'][:4]):
                        c_new, _ = _release_time_tandem_queue_eval(obs, admitted, waiting, plan, motion)
                        c_old, _ = _lindley_plan_motion_eval(obs, admitted, waiting, plan, motion)
                        e = Engine.from_observation(obs)
                        e.assign(plan)
                        e.advance(motion)
                        for _ in range(2):
                            e.admit()
                            pub = e.observe()
                            e.assign(safe_assignments(pub))
                            e.advance(zero)
                        approx_comparisons.append(dict(
                            parent=int(parent),
                            workload=int(workload),
                            time=int(now),
                            plan_idx=int(pi),
                            motion_idx=int(mi),
                            engine_success=float(e.counts['success']),
                            release_edf_pred=float(c_new),
                            static_lindley_pred=float(c_old),
                        ))

        if round_idx == 2 and r1_model is not None:
            step = rl_actor_step(
                root, prototype, proposer, score_model, obs, adapter, temporal,
                'joint_distilled', r1_model, parent, workload, 0, now,
                precomputed_base=base_state,
            )
            engine.assign(step['plan'])
            engine.advance(step['motion'])
        else:
            mix_rng = np.random.default_rng(t.seed(RL_REVISION, 'distill_explore', parent, workload, now))
            if mix_rng.random() < 0.5:
                step = rl_actor_step(
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
    samples['_approx_comparisons'] = approx_comparisons
    prototype.previous.dump(dest, samples)
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(
            split=split, round=round_idx, parent=parent, workload=workload,
            states=len(samples['joint_distilled']),
        ),
    )


def fit_clean_distill_worker(payload):
    """Fit `decomposed_distilled` and `joint_distilled` on clean physical features."""
    root, round_idx, parent, arm = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    mode = rl_mode_for_arm(arm)

    train_samples = []
    for w in TRAIN_R1_WORKLOADS:
        data = prototype.previous.saved(out(root) / 'distill_datasets' / f'train_r1_{parent}_{w}.pkl.gz')
        train_samples.extend(data[arm])
    if round_idx == 2:
        for w in TRAIN_R2_WORKLOADS:
            data = prototype.previous.saved(out(root) / 'distill_datasets' / f'train_r2_{parent}_{w}.pkl.gz')
            train_samples.extend(data[arm])

    val_samples = []
    for w in VAL_WORKLOADS:
        data = prototype.previous.saved(out(root) / 'distill_datasets' / f'val_r1_{parent}_{w}.pkl.gz')
        val_samples.extend(data[arm])

    train_rows = [s['row'] for s in train_samples]
    train_targets = [s['target'] for s in train_samples]
    val_rows = [s['row'] for s in val_samples]
    val_targets = [s['target'] for s in val_samples]

    if round_idx == 1:
        init_m = initialize_rl_actor_critic(
            train_rows[0], hidden=48, critic_hidden=48, seed=t.seed(RL_REVISION, 'init_distill', parent, arm)
        )
        trained, info = fit_clean_distilled_actor(
            train_rows, train_targets, val_rows, val_targets,
            init_m, mode=mode, epochs=32, batch_size=16,
            learning_rate=0.008, seed=t.seed(RL_REVISION, 'fit_distill_r1', parent, arm),
        )
        prototype.previous.dump(out(root) / 'distill_models_r1' / arm / f'{parent}.pkl.gz', trained)
    else:
        r1_m = prototype.previous.saved(out(root) / 'distill_models_r1' / arm / f'{parent}.pkl.gz')
        trained, info = fit_clean_distilled_actor(
            train_rows, train_targets, val_rows, val_targets,
            r1_m, mode=mode, epochs=32, batch_size=16,
            learning_rate=0.008, seed=t.seed(RL_REVISION, 'fit_distill_r2', parent, arm),
        )
        prototype.previous.dump(model_path(root, arm, parent), trained)

    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(round=round_idx, arm=arm, parent=parent, best_val_kl=info['best_val_kl']),
    )


def summarize_tandem_approximation_audit(root):
    root = Path(root)
    records = []
    for p in range(3):
        for w in VAL_WORKLOADS:
            data = prototype.previous.saved(out(root) / 'distill_datasets' / f'val_r1_{p}_{w}.pkl.gz')
            records.extend(data.get('_approx_comparisons', []))
    if not records:
        return {}
    eng = np.array([r['engine_success'] for r in records], dtype=np.float64)
    new_p = np.array([r['release_edf_pred'] for r in records], dtype=np.float64)
    old_p = np.array([r['static_lindley_pred'] for r in records], dtype=np.float64)
    audit = dict(
        evaluated_pairs=len(records),
        release_time_edf=dict(
            mae=float(np.mean(np.abs(new_p - eng))),
            rmse=float(np.sqrt(np.mean((new_p - eng) ** 2))),
            correlation=float(np.corrcoef(new_p, eng)[0, 1]) if np.std(new_p) > 1e-9 else 1.0,
        ),
        static_single_pass_lindley=dict(
            mae=float(np.mean(np.abs(old_p - eng))),
            rmse=float(np.sqrt(np.mean((old_p - eng) ** 2))),
            correlation=float(np.corrcoef(old_p, eng)[0, 1]) if np.std(old_p) > 1e-9 else 1.0,
        ),
    )
    prototype.write(out(root) / 'tandem_approximation_audit.json', audit)
    return audit


def initialize_rl_lineages(root):
    """Initialize Iteration 0 checkpoints for all 7 RL training arms."""
    root = Path(root)
    sample_data = prototype.previous.saved(
        out(root) / 'distill_datasets' / f'val_r1_0_{VAL_WORKLOADS[0]}.pkl.gz'
    )
    sample_row = sample_data['joint_distilled'][0]['row']

    for parent in range(3):
        for arm in SCRATCH_RL_ARMS:
            p0 = iter_model_path(root, 0, arm, parent)
            if not p0.exists():
                init_m = initialize_rl_actor_critic(
                    sample_row, hidden=48, critic_hidden=48,
                    seed=t.seed(RL_REVISION, 'rl_init_scratch', parent, arm),
                )
                prototype.previous.dump(p0, init_m)
                if arm == 'joint_rl':
                    prototype.previous.dump(ckpt_model_path(root, 0, parent), init_m)

        for arm, src_distilled in (
            ('decomposed_rl_finetuned', 'decomposed_distilled'),
            # Both `joint_no_expectation_rl_finetuned` and `joint_rl_finetuned` warm-start from the
            # EXACT SAME `joint_distilled` checkpoint so their comparison strictly isolates C_upper[k]
            # and cross-layer backpropagation during RL fine-tuning.
            ('joint_no_expectation_rl_finetuned', 'joint_distilled'),
            ('joint_rl_finetuned', 'joint_distilled'),
        ):
            p0 = iter_model_path(root, 0, arm, parent)
            if not p0.exists():
                dist_m = prototype.previous.saved(model_path(root, src_distilled, parent))
                warm_m = warmstart_from_distilled_actor(
                    dist_m, sample_row, critic_hidden=48,
                    seed=t.seed(RL_REVISION, 'rl_init_warm', parent, arm),
                )
                prototype.previous.dump(p0, warm_m)


def rl_rollout_worker(payload):
    """Collect one on-policy RL trajectory driven strictly by real Engine environment rewards."""
    root, iteration, arm, parent, workload, rollout_rep = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    dest = out(root) / 'rl_rollouts' / f'iter_{iteration}' / arm / f'{parent}_{workload}_{rollout_rep}.pkl.gz'
    if dest.exists():
        saved_ep = prototype.previous.saved(dest)
        return dict(
            seconds=0.0,
            cpu_seconds=0.0,
            result=dict(
                iteration=iteration, arm=arm, parent=parent,
                workload=workload, rollout_rep=rollout_rep,
                dsr=saved_ep['summary']['dsr'],
            ),
        )

    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2)
    mode = rl_mode_for_arm(arm)

    prev_model = prototype.previous.saved(iter_model_path(root, iteration - 1, arm, parent))
    if arm in FINETUNED_RL_ARMS:
        ref_model = prototype.previous.saved(iter_model_path(root, 0, arm, parent))
    else:
        ref_model = prev_model

    steps, summary = collect_rl_episode(
        root,
        prototype,
        proposer,
        score_model,
        tape,
        prev_model,
        mode=mode,
        parent=parent,
        workload=workload,
        iteration=iteration,
        rollout_rep=rollout_rep,
        ref_model=ref_model,
        gamma=0.85,
        lambda_expired=0.65,
        cf_stride=4 if arm in FINETUNED_RL_ARMS else 0,
    )
    prototype.previous.dump(dest, dict(steps=steps, summary=summary))
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(
            iteration=iteration, arm=arm, parent=parent,
            workload=workload, rollout_rep=rollout_rep,
            dsr=summary['dsr'],
        ),
    )


def rl_ppo_fit_worker(payload):
    """Run Clipped Joint PPO + Critic GAE update on collected on-policy episodes."""
    root, iteration, arm, parent = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    dest = iter_model_path(root, iteration, arm, parent)
    stat_dest = out(root) / 'rl_curves' / f'iter_{iteration}_{arm}_{parent}.json'
    if dest.exists() and stat_dest.exists():
        info = json.loads(stat_dest.read_text(encoding='utf-8'))
        return dict(
            seconds=0.0,
            cpu_seconds=0.0,
            result=dict(
                iteration=iteration, arm=arm, parent=parent,
                mean_episode_reward=info['stats']['mean_episode_reward'],
            ),
        )

    workloads = TRAIN_R1_WORKLOADS if iteration in (1, 3) else TRAIN_R2_WORKLOADS
    episodes = []
    rollout_dsrs = []
    for w in workloads:
        for rep in ROLLOUT_REPS:
            ep_data = prototype.previous.saved(
                out(root) / 'rl_rollouts' / f'iter_{iteration}' / arm / f'{parent}_{w}_{rep}.pkl.gz'
            )
            episodes.append(ep_data['steps'])
            rollout_dsrs.append(float(ep_data['summary']['dsr']))

    prev_model = prototype.previous.saved(iter_model_path(root, iteration - 1, arm, parent))
    mode = rl_mode_for_arm(arm)

    if arm in FINETUNED_RL_ARMS:
        updated, stats = ppo_update_actor_critic(
            episodes,
            prev_model,
            mode=mode,
            ppo_epochs=4,
            critic_epochs=8,
            batch_size=64,
            actor_lr=0.0025,
            critic_lr=0.008,
            clip_eps=0.20,
            ent_coef=0.0005,
            kl_coef=0.015,
            gamma=0.85,
            gae_lambda=0.90,
            cf_weight=1.25,
            seed=t.seed(RL_REVISION, 'ppo_fit', iteration, arm, parent),
        )
    else:
        updated, stats = ppo_update_actor_critic(
            episodes,
            prev_model,
            mode=mode,
            ppo_epochs=6,
            critic_epochs=10,
            batch_size=32,
            actor_lr=0.0035,
            critic_lr=0.008,
            clip_eps=0.20,
            ent_coef=0.002,
            kl_coef=0.015,
            gamma=0.85,
            gae_lambda=0.90,
            cf_weight=0.0,
            seed=t.seed(RL_REVISION, 'ppo_fit', iteration, arm, parent),
        )

    stats['mean_rollout_dsr'] = float(np.mean(rollout_dsrs)) if rollout_dsrs else 0.0
    prototype.previous.dump(dest, updated)
    if arm == 'joint_rl':
        prototype.previous.dump(ckpt_model_path(root, iteration, parent), updated)
    prototype.write(stat_dest, dict(iteration=iteration, arm=arm, parent=parent, stats=stats))
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(
            iteration=iteration, arm=arm, parent=parent,
            mean_episode_reward=stats['mean_episode_reward'],
            mean_rollout_dsr=stats['mean_rollout_dsr'],
        ),
    )


def rl_val_worker(payload):
    """Evaluate an iteration checkpoint on held-out VAL_WORKLOADS across 2 replicas using pure closed-loop Engine DSR."""
    root, iteration, arm, parent, workload = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    dest = out(root) / 'rl_val' / f'iter_{iteration}' / arm / f'{parent}_{workload}.json'
    if dest.exists():
        existing = json.loads(dest.read_text(encoding='utf-8'))
        return dict(
            seconds=0.0,
            cpu_seconds=0.0,
            result=dict(iteration=iteration, arm=arm, parent=parent, workload=workload, dsr=existing['dsr']),
        )

    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2)
    model = prototype.previous.saved(iter_model_path(root, iteration, arm, parent))

    total_counts = {}
    rep_dsrs = []
    for rep in (0, 1):
        engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
        adapter = t.LearningAdapter()
        temporal = RateHistory()
        while engine.t < 200 or engine.tasks:
            now = int(engine.t)
            assert now <= 204
            engine.add(tape.slot(now))
            engine.admit()
            obs = engine.observe()
            temporal.update(obs)
            step = rl_actor_step(
                root, prototype, proposer, score_model, obs, adapter, temporal,
                arm, model, parent, workload, rep, now,
            )
            engine.assign(step['plan'])
            engine.advance(step['motion'])

        audit = engine.audit()
        assert not engine.tasks and audit['status'] == 'PASS'
        rep_dsrs.append(float(engine.counts['success'] / max(1, engine.counts['generated'])))
        for k_cnt, v_cnt in engine.counts.items():
            total_counts[k_cnt] = total_counts.get(k_cnt, 0) + int(v_cnt)

    dsr = float(np.mean(rep_dsrs))
    rec = dict(
        iteration=iteration, arm=arm, parent=parent, workload=workload,
        counts=total_counts, dsr=dsr,
    )
    prototype.write(dest, rec)
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(iteration=iteration, arm=arm, parent=parent, workload=workload, dsr=dsr),
    )


def select_and_assemble_rl_models(root):
    """Select best validation checkpoint per (arm, parent), assemble `decomposed_composed_rl`, and write curves."""
    root = Path(root)
    curves = {}
    selection_summary = {}

    for arm in RL_TRAINING_ARMS:
        arm_curve = []
        selection_summary[arm] = {}
        candidate_iters = (1, 2, 3, 4)
        for parent in range(3):
            best_iter = None
            best_key = None
            for it in candidate_iters:
                val_dsrs = []
                for w in VAL_WORKLOADS:
                    v_rec = json.loads(
                        (out(root) / 'rl_val' / f'iter_{it}' / arm / f'{parent}_{w}.json').read_text(encoding='utf-8')
                    )
                    val_dsrs.append(float(v_rec['dsr']))
                mean_val_dsr = float(np.mean(val_dsrs))
                key = (round(mean_val_dsr, 6), int(it))
                if best_key is None or key > best_key:
                    best_key = key
                    best_iter = it

            chosen_model = prototype.previous.saved(iter_model_path(root, best_iter, arm, parent))
            prototype.previous.dump(model_path(root, arm, parent), chosen_model)
            selection_summary[arm][str(parent)] = dict(
                selected_iteration=int(best_iter),
                val_dsr=float(best_key[0]),
            )

        for it in (0, 1, 2, 3, 4):
            val_dsrs = []
            for p in range(3):
                for w in VAL_WORKLOADS:
                    v_rec = json.loads(
                        (out(root) / 'rl_val' / f'iter_{it}' / arm / f'{p}_{w}.json').read_text(encoding='utf-8')
                    )
                    val_dsrs.append(float(v_rec['dsr']))
            entry = dict(iteration=it, validation_dsr=float(np.mean(val_dsrs)))
            if it > 0:
                it_stats = [
                    json.loads((out(root) / 'rl_curves' / f'iter_{it}_{arm}_{p}.json').read_text(encoding='utf-8'))['stats']
                    for p in range(3)
                ]
                entry['train_episode_reward'] = float(np.mean([s['mean_episode_reward'] for s in it_stats]))
                entry['train_rollout_dsr'] = float(np.mean([s['mean_rollout_dsr'] for s in it_stats]))
                entry['actor_loss'] = float(np.mean([s['actor_loss'] for s in it_stats]))
                entry['critic_loss'] = float(np.mean([s['critic_loss'] for s in it_stats]))
            arm_curve.append(entry)
        curves[arm] = arm_curve

    for parent in range(3):
        lower_m = prototype.previous.saved(model_path(root, 'lower_only_rl', parent))
        upper_m = prototype.previous.saved(model_path(root, 'upper_only_rl', parent))
        composed = compose_decomposed_rl_actor(lower_m, upper_m)
        prototype.previous.dump(model_path(root, 'decomposed_composed_rl', parent), composed)

    prototype.write(out(root) / 'rl_training_curves.json', dict(curves=curves, selected=selection_summary))
    return curves


def write_freeze_manifest(root):
    """Write pre-confirmation `freeze_manifest.json` binding UTC time, full dependency source SHA256s, parent checkpoints, and model SHA256s."""
    root = Path(root)
    code_root = Path(__file__).resolve().parents[2]
    manifest_path = out(root) / 'freeze_manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding='utf-8')), t.sha(manifest_path)

    source_sha256 = {
        rel_path: t.sha((code_root / rel_path) if (code_root / rel_path).exists() else (root / rel_path))
        for rel_path in RL_SOURCE_FILES
    }
    parent_checkpoint_sha256 = {}
    for p in range(3):
        p_proposer = Path(t.parent_path(root, p))
        if p_proposer.exists():
            parent_checkpoint_sha256[str(p_proposer.relative_to(root)).replace('\\', '/')] = t.sha(p_proposer)
        p_scorer = Path(prototype.previous.old.score_path(str(root), 'reg', 'joint', p * 2))
        if p_scorer.exists():
            parent_checkpoint_sha256[str(p_scorer.relative_to(root)).replace('\\', '/')] = t.sha(p_scorer)

    models_root = out(root) / 'models'
    model_sha256 = {
        str(p.relative_to(out(root))).replace('\\', '/'): t.sha(p)
        for p in sorted(models_root.rglob('*.pkl.gz'))
    }
    manifest = dict(
        revision=RL_REVISION,
        frozen_at_utc=utc_now_iso(),
        confirmation_workloads=list(CONFIRMATION_WORKLOADS),
        evaluated_arms=list(RL_EVAL_ARMS),
        primary_comparison_pairs=[f'{a}--{b}' for a, b in PRIMARY_COMPARISON_PAIRS],
        source_sha256=source_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        model_sha256=model_sha256,
    )
    prototype.write(manifest_path, manifest)
    return manifest, t.sha(manifest_path)


def episode_worker(payload):
    """Run one closed-loop evaluation episode for Pilot or Blind Confirmation."""
    root, stage, arm, parent, workload, replica, ckpt_iter, freeze_sha, frozen_at_utc = payload
    started = time.perf_counter()
    cpu = time.process_time()
    root = Path(root)
    allowed = PILOT_WORKLOADS if stage in ('pilot', 'pilot_ckpt') else CONFIRMATION_WORKLOADS
    assert workload in allowed, (stage, workload)

    if stage == 'pilot_ckpt':
        dest = out(root) / 'pilot_checkpoints' / f'iter_{ckpt_iter}' / f'{parent}_{workload}_{replica}.json'
    else:
        dest = out(root) / stage / arm / f'{parent}_{workload}_{replica}.json'

    if dest.exists():
        existing = json.loads(dest.read_text(encoding='utf-8'))
        valid_resume = True
        if stage == 'confirmation':
            if existing.get('freeze_manifest_sha256') != freeze_sha:
                valid_resume = False
            if str(existing.get('started_at_utc', '')) < str(frozen_at_utc):
                valid_resume = False
        if valid_resume:
            return dict(
                seconds=0.0,
                cpu_seconds=0.0,
                result=dict(stage=stage, arm=arm, ckpt_iter=ckpt_iter, success=existing['counts'].get('success', 0)),
            )

    started_at_utc = utc_now_iso()
    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2)

    if ckpt_iter is not None:
        model = prototype.previous.saved(ckpt_model_path(root, ckpt_iter, parent))
    elif arm in (
        'lower_only_rl',
        'upper_only_rl',
        'decomposed_composed_rl',
        'decomposed_rl',
        'joint_no_expectation_rl',
        'joint_rl',
        'decomposed_distilled',
        'joint_no_expectation_distilled',
        'joint_distilled',
        'decomposed_rl_finetuned',
        'joint_no_expectation_rl_finetuned',
        'joint_rl_finetuned',
    ):
        model = prototype.previous.saved(model_path(root, arm, parent))
    elif arm == 'distilled_top6_mpc':
        model = prototype.previous.saved(model_path(root, 'joint_distilled', parent))
    elif arm == 'joint_rl_mpc':
        model = prototype.previous.saved(model_path(root, 'joint_rl_finetuned', parent))
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
        step = rl_actor_step(
            root, prototype, proposer, score_model, obs, adapter, temporal,
            arm if ckpt_iter is None else 'joint_rl', model, parent, workload, replica, now,
        )
        trajectory.append(dict(
            time=now,
            positions=list(engine.positions),
            motion=step['motion'],
            admitted=len(admitted),
            original_candidates=step['original_candidates'],
            candidates=step['candidates'],
            selected_new=step['selected_new'],
            metadata=step['metadata'],
            decision_seconds=step['decision_seconds'],
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
    completed_at_utc = utc_now_iso()

    record = dict(
        stage=stage,
        arm=arm,
        parent=parent,
        workload=workload,
        replica=replica,
        ckpt_iter=ckpt_iter,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        freeze_manifest_sha256=freeze_sha,
        counts=counts,
        cohorts=cohort_statistics(engine.finished.values()),
        quality=quality_details(list(engine.finished.values())),
        successes=successes,
        trajectory=trajectory,
        fingerprint=tape.fingerprint,
        audit=audit,
    )
    prototype.write(dest, record)
    return dict(
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        result=dict(stage=stage, arm=arm, ckpt_iter=ckpt_iter, success=counts.get('success', 0)),
    )


def copy_ckpt_from_arm(root, arm, iteration):
    src_folder = out(root) / 'pilot' / arm
    for source in sorted(src_folder.glob('*.json')):
        record = json.loads(source.read_text(encoding='utf-8'))
        record['ckpt_iter'] = iteration
        dest = out(root) / 'pilot_checkpoints' / f'iter_{iteration}' / source.name
        prototype.write(dest, record)


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
            if len(records) % 16 == 0 or len(records) == len(jobs):
                print(label, len(records), '/', len(jobs), flush=True)
    seconds = time.perf_counter() - started
    cpu = sum(row['cpu_seconds'] for row in records)
    prototype.write(val(root) / (label + '_complete.json'), dict(
        tasks=len(jobs),
        wall_seconds=seconds,
        worker_cpu_seconds=cpu,
        equivalent_cpus=cpu / max(seconds, 1e-9),
        utilization=cpu / max(seconds, 1e-9) / len(cpus),
        records=records,
    ))


def protected(root):
    v1_dir = Path(root) / 'results/chapter5_dynamic_service_20261028/queued_joint_rl_20261106'
    v1_files = {
        str(path.relative_to(root)): prototype.t.sha(path)
        for path in v1_dir.rglob('*')
        if path.is_file() and 'rl_rollouts' not in path.parts and 'distill_datasets' not in path.parts
    } if v1_dir.exists() else {}
    return qa_exp.protected(root) | {
        str(path.relative_to(root)): prototype.t.sha(path)
        for path in qa_exp.out(root).rglob('*') if path.is_file()
    } | v1_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert str(root).startswith('/home/PengYanghan/Lunwen/chapter5-dynamic-service-20261028')

    before = protected(root)
    if not (val(root) / 'historical_hashes_before.json').exists():
        prototype.write(val(root) / 'historical_hashes_before.json', before)

    # Stage 1: Clean-feature teacher distillation for warm-start & control baselines (`decomposed_distilled`, `joint_distilled`)
    if not (val(root) / 'distill_collect_r1_complete.json').exists():
        r1_jobs = (
            [('train', 1, p, w) for p in range(3) for w in TRAIN_R1_WORKLOADS]
            + [('val', 1, p, w) for p in range(3) for w in VAL_WORKLOADS]
        )
        dispatch_fn(root, 'distill_collect_r1', collect_clean_distill_worker, r1_jobs)

    if not (val(root) / 'distill_fit_r1_complete.json').exists():
        r1_fit_jobs = [(1, p, arm) for p in range(3) for arm in DISTILLED_ARMS]
        dispatch_fn(root, 'distill_fit_r1', fit_clean_distill_worker, r1_fit_jobs)

    if not (val(root) / 'distill_collect_r2_complete.json').exists():
        r2_jobs = [('train', 2, p, w) for p in range(3) for w in TRAIN_R2_WORKLOADS]
        dispatch_fn(root, 'distill_collect_r2', collect_clean_distill_worker, r2_jobs)

    if not (val(root) / 'distill_fit_r2_complete.json').exists():
        r2_fit_jobs = [(2, p, arm) for p in range(3) for arm in DISTILLED_ARMS]
        dispatch_fn(root, 'distill_fit_r2', fit_clean_distill_worker, r2_fit_jobs)

    approx_audit = summarize_tandem_approximation_audit(root)

    # Stage 2: Genuine Environment-Reward Joint RL (On-Policy Clipped PPO + Critic GAE, ZERO teacher calls)
    initialize_rl_lineages(root)
    if not (val(root) / 'rl_val_iter_0_complete.json').exists():
        val0_jobs = [(0, arm, p, w) for arm in RL_TRAINING_ARMS for p in range(3) for w in VAL_WORKLOADS]
        dispatch_fn(root, 'rl_val_iter_0', rl_val_worker, val0_jobs)

    for it in PPO_ITERATIONS:
        workloads = TRAIN_R1_WORKLOADS if it in (1, 3) else TRAIN_R2_WORKLOADS
        if not (val(root) / f'rl_rollout_iter_{it}_complete.json').exists():
            rollout_jobs = [
                (it, arm, p, w, rep)
                for arm in RL_TRAINING_ARMS
                for p in range(3)
                for w in workloads
                for rep in ROLLOUT_REPS
            ]
            dispatch_fn(root, f'rl_rollout_iter_{it}', rl_rollout_worker, rollout_jobs)

        if not (val(root) / f'rl_fit_iter_{it}_complete.json').exists():
            fit_jobs = [(it, arm, p) for arm in RL_TRAINING_ARMS for p in range(3)]
            dispatch_fn(root, f'rl_fit_iter_{it}', rl_ppo_fit_worker, fit_jobs)

        if not (val(root) / f'rl_val_iter_{it}_complete.json').exists():
            val_jobs = [(it, arm, p, w) for arm in RL_TRAINING_ARMS for p in range(3) for w in VAL_WORKLOADS]
            dispatch_fn(root, f'rl_val_iter_{it}', rl_val_worker, val_jobs)

    select_and_assemble_rl_models(root)

    rollout_audit = audit_rl_rollouts(out(root), prototype.previous.saved)
    assert rollout_audit['verified'], rollout_audit
    ra_path = out(root) / 'rollout_replay_audit.json'
    if ra_path.exists():
        ra_path.unlink()
    prototype.write(ra_path, rollout_audit)

    val_sample_payload = prototype.previous.saved(out(root) / 'distill_datasets/val_r1_0_941000.pkl.gz')
    sample_rows = [item['row'] for item in val_sample_payload['joint_distilled'][:30]]
    decomp_audit = audit_true_decomposed_independence(
        out(root), prototype.previous.saved, sample_rows
    )
    assert decomp_audit['verified'], decomp_audit
    da_path = out(root) / 'decomposed_independence_audit.json'
    if da_path.exists():
        da_path.unlink()
    prototype.write(da_path, decomp_audit)

    # Stage 3: Pilot Evaluation on 951000-951003
    if not (val(root) / 'pilot_complete.json').exists():
        pilot_jobs = [
            ('pilot', arm, p, w, 0, None, None, None)
            for arm in RL_EVAL_ARMS
            for p in range(3)
            for w in PILOT_WORKLOADS
        ]
        ckpt_jobs = [
            ('pilot_ckpt', 'joint_rl', p, w, 0, it, None, None)
            for it in (1, 2, 3, 4)
            for p in range(3)
            for w in PILOT_WORKLOADS
        ]
        dispatch_fn(root, 'pilot', episode_worker, pilot_jobs + ckpt_jobs)
        copy_ckpt_from_arm(root, 'uncoordinated_init', 0)

    pilot = stage_review(root, 'pilot', out(root))
    pilot['checkpoint_curve'] = rl_checkpoint_curve_review(out(root))
    pilot['tandem_approximation_audit'] = approx_audit
    pilot['rollout_replay_audit'] = rollout_audit
    pilot['decomposed_independence_audit'] = decomp_audit
    for p_path in (out(root) / 'pilot_analysis.json', val(root) / 'pilot_gate.json'):
        if p_path.exists():
            p_path.unlink()
        prototype.write(p_path, pilot)

    # Stage 4: Pre-Confirmation Freeze & Blind Confirmation on brand-new 955000-955007
    manifest, freeze_sha = write_freeze_manifest(root)
    frozen_at_utc = manifest['frozen_at_utc']

    prot_path = out(root) / 'protocol.json'
    if prot_path.exists():
        prot_path.unlink()
    prototype.write(prot_path, dict(
        revision=RL_REVISION,
        algorithm_classification='on_policy_clipped_joint_ppo_with_critic_and_gae',
        rl_training_arms=list(RL_TRAINING_ARMS),
        distilled_baseline_arms=list(DISTILLED_ARMS),
        evaluated_arms=list(RL_EVAL_ARMS),
        train_r1_workloads=list(TRAIN_R1_WORKLOADS),
        train_r2_workloads=list(TRAIN_R2_WORKLOADS),
        val_workloads=list(VAL_WORKLOADS),
        pilot_workloads=list(PILOT_WORKLOADS),
        confirmation_workloads=list(CONFIRMATION_WORKLOADS),
        ppo_iterations=list(PPO_ITERATIONS),
        rollout_replicas_per_workload=len(ROLLOUT_REPS),
        zero_teacher_calls_in_rl=True,
        clean_physical_features_no_handcoded_penalty=True,
        unconfounded_plan_dependent_masking=True,
        true_decomposed_plan_independent_verified=decomp_audit['verified'],
        post200_drain_steps_recorded_verified=rollout_audit['verified'],
        freeze_manifest_sha256=freeze_sha,
        frozen_at_utc=frozen_at_utc,
        tandem_approximation_audit=approx_audit,
        rollout_replay_audit=rollout_audit,
        decomposed_independence_audit=decomp_audit,
    ))

    confirmation = dict(triggered=False, reason='pilot gate failed')
    if pilot['gate_passed']:
        if not (val(root) / 'confirmation_complete.json').exists():
            conf_jobs = [
                ('confirmation', arm, p, w, r, None, freeze_sha, frozen_at_utc)
                for arm in RL_EVAL_ARMS
                for p in range(3)
                for w in CONFIRMATION_WORKLOADS
                for r in (0, 1)
            ]
            dispatch_fn(root, 'confirmation', episode_worker, conf_jobs)
        confirmation = stage_review(root, 'confirmation', out(root))
        confirmation['tandem_approximation_audit'] = approx_audit
        confirmation['rollout_replay_audit'] = rollout_audit
        confirmation['decomposed_independence_audit'] = decomp_audit
        for c_path in (out(root) / 'confirmation_analysis.json', val(root) / 'confirmation_gate.json'):
            if c_path.exists():
                c_path.unlink()
            prototype.write(c_path, confirmation)

    freeze_audit = verify_freeze_manifest(root, out(root))
    assert freeze_audit['verified'], freeze_audit

    after = protected(root)
    assert before == after
    rc_path = val(root) / 'run_complete.json'
    if rc_path.exists():
        rc_path.unlink()
    prototype.write(rc_path, dict(
        status='PASS',
        revision=RL_REVISION,
        pilot_gate_passed=pilot['gate_passed'],
        confirmation_triggered=bool(pilot['gate_passed']),
        confirmation_gate_passed=confirmation.get('gate_passed'),
        freeze_manifest_verified=freeze_audit['verified'],
        rollout_replay_audit_verified=rollout_audit['verified'],
        decomposed_independence_audit_verified=decomp_audit['verified'],
        freeze_manifest_sha256=freeze_sha,
        historical_files=len(after),
        historical_artifacts_unchanged=True,
    ))


if __name__ == '__main__':
    main()
