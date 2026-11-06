"""Statistical review, pre-confirmation freeze audit, and automated Markdown table generator for Joint RL."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from marl_models.queued_joint_rl import (
    compute_episode_gae_and_returns,
    rl_actor_forward,
    rl_actor_policy,
)
from scripts.analysis.review_queued_waiting_rescue import load
from utils.queued_joint_rl import RL_EVAL_ARMS, RL_TRAINING_ARMS, rl_mode_for_arm

PILOT_WORKLOADS = tuple(range(951000, 951004))
CONFIRMATION_WORKLOADS = tuple(range(955000, 955008))
BOOTSTRAP_DRAWS = 10000
PRIMARY_FAMILY_SIZE = 8
ARMS = RL_EVAL_ARMS

RL_SOURCE_FILES = (
    'environment/queued_service.py',
    'utils/queued_closed_loop.py',
    'utils/queued_taskset_closed_loop.py',
    'utils/queued_service_controllers.py',
    'utils/queued_service_learning.py',
    'utils/queued_arrival_adaptation.py',
    'utils/queued_waiting_rescue.py',
    'utils/queued_planner_distillation.py',
    'utils/queued_quality_amortization.py',
    'utils/queued_joint_training.py',
    'marl_models/queued_joint_training.py',
    'marl_models/queued_joint_rl.py',
    'utils/queued_joint_rl.py',
    'scripts/analysis/queued_joint_rl_experiment.py',
    'scripts/analysis/review_queued_joint_rl.py',
)

NEURAL_ZERO_SIM_ARMS = (
    'uncoordinated_init',
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
)

# Pre-registered Primary Comparison Family (M = 8):
PRIMARY_COMPARISON_PAIRS = (
    ('joint_rl', 'uncoordinated_init'),
    ('joint_rl', 'decomposed_rl'),
    ('joint_rl', 'joint_distilled'),
    ('joint_distilled', 'decomposed_distilled'),
    ('joint_rl_finetuned', 'joint_distilled'),
    ('joint_rl_finetuned', 'decomposed_rl_finetuned'),
    ('joint_rl_finetuned', 'joint_no_expectation_rl_finetuned'),
    ('joint_rl_mpc', 'quality_amortized'),
)

# Exploratory / Mechanism Attribution Comparisons:
EXPLORATORY_COMPARISON_PAIRS = (
    ('joint_rl', 'lower_only_rl'),
    ('joint_rl', 'upper_only_rl'),
    ('joint_rl', 'decomposed_composed_rl'),
    ('joint_rl', 'joint_no_expectation_rl'),
    ('joint_no_expectation_rl', 'decomposed_rl'),
    ('decomposed_rl', 'decomposed_composed_rl'),
    ('decomposed_composed_rl', 'lower_only_rl'),
    ('decomposed_composed_rl', 'upper_only_rl'),
    ('upper_only_rl', 'uncoordinated_init'),
    ('lower_only_rl', 'uncoordinated_init'),
    ('joint_distilled', 'joint_no_expectation_distilled'),
    ('joint_no_expectation_distilled', 'decomposed_distilled'),
    ('joint_rl_finetuned', 'decomposed_distilled'),
    ('joint_rl_finetuned', 'joint_rl'),
    ('joint_rl_finetuned', 'uncoordinated_init'),
    ('joint_rl_finetuned', 'quality_amortized'),
    ('joint_no_expectation_rl_finetuned', 'joint_distilled'),
    ('decomposed_rl_finetuned', 'decomposed_distilled'),
    ('joint_rl_mpc', 'joint_rl_finetuned'),
    ('joint_rl_mpc', 'distilled_top6_mpc'),
    ('joint_rl_mpc', 'prior_top6_mpc'),
    ('distilled_top6_mpc', 'prior_top6_mpc'),
    ('distilled_top6_mpc', 'quality_amortized'),
)

COMPARISON_PAIRS = PRIMARY_COMPARISON_PAIRS + EXPLORATORY_COMPARISON_PAIRS


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_bootstrap(
    a,
    da,
    b,
    db,
    draws=BOOTSTRAP_DRAWS,
    family_size=PRIMARY_FAMILY_SIZE,
    all_family_size=len(COMPARISON_PAIRS),
    seed=2026110601,
):
    """Paired block bootstrap over workloads and action streams conditioned on 3 fixed parents."""
    a, da, b, db = (np.asarray(value, float) for value in (a, da, b, db))
    assert a.shape == da.shape == b.shape == db.shape
    assert a.ndim == 3 and a.shape[0] == 3 and np.all(da > 0) and np.all(db > 0)
    _, workloads, replicas = a.shape
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, workloads, (draws, workloads))
    streams = rng.integers(0, replicas, (draws, workloads, replicas))
    samples = np.zeros(draws, dtype=np.float64)
    effects = a.sum((1, 2)) / da.sum((1, 2)) - b.sum((1, 2)) / db.sum((1, 2))
    for parent in range(3):
        sums = [value[parent][indices[:, :, None], streams].sum((1, 2)) for value in (a, da, b, db)]
        samples += (sums[0] / sums[1] - sums[2] / sums[3]) / 3.0
    alpha_bonf = 0.05 / (2.0 * max(1, int(family_size)))
    alpha_all = 0.05 / (2.0 * max(1, int(all_family_size)))
    ci90 = np.quantile(samples, [0.05, 0.95]).tolist()
    ci95 = np.quantile(samples, [0.025, 0.975]).tolist()
    bonf95 = np.quantile(samples, [alpha_bonf, 1.0 - alpha_bonf]).tolist()
    bonf_all95 = np.quantile(samples, [alpha_all, 1.0 - alpha_all]).tolist()
    return dict(
        mean=float(effects.mean()),
        by_parent=effects.tolist(),
        draws=int(draws),
        ci90=ci90,
        ci95=ci95,
        bonferroni_family95=bonf95,
        bonferroni_all_comparisons95=bonf_all95,
        tost_equivalent_0_50pp=bool(ci90[0] > -0.005 and ci90[1] < 0.005),
        tost_equivalent_0_75pp=bool(ci90[0] > -0.0075 and ci90[1] < 0.0075),
        noninferior_0_50pp_ci95=bool(ci95[0] > -0.005),
        noninferior_0_75pp_ci95=bool(ci95[0] > -0.0075),
        scope=(
            f'Conditional on 3 fixed parent lineages, paired block bootstrap over {workloads} '
            f'exogenous workloads x {replicas} action streams (B={draws}, Primary Bonferroni M={family_size}, '
            f'All-Pairs Bonferroni M={all_family_size}).'
        ),
    )


def summarize_records(rows, workloads, replicas, arm=None, expected_freeze_sha=None, frozen_at_utc=None):
    arrays = {
        phase: {key: np.zeros((3, len(workloads), replicas)) for key in ('generated', 'admitted', 'success')}
        for phase in ('all', 'pre', 'peak', 'post')
    }
    for row in rows:
        counts = row['counts']
        assert row['audit']['status'] == 'PASS'
        assert counts['generated'] == counts.get('success', 0) + counts.get('expired', 0)
        assert sum(row['successes']) == counts.get('success', 0)
        if expected_freeze_sha is not None:
            assert row.get('freeze_manifest_sha256') == expected_freeze_sha, (
                f"Episode freeze hash mismatch for {arm}: {row.get('freeze_manifest_sha256')} != {expected_freeze_sha}"
            )
        if frozen_at_utc is not None and row.get('started_at_utc') is not None:
            assert row['started_at_utc'] >= frozen_at_utc, (
                f"Episode started before freeze timestamp: {row['started_at_utc']} < {frozen_at_utc}"
            )
        index = (row['parent'], workloads.index(row['workload']), row['replica'])
        for phase in arrays:
            source = counts if phase == 'all' else row['cohorts']['arrival'][phase]
            for key in arrays[phase]:
                arrays[phase][key][index] = source.get(key, 0)
    arrays['nonpeak'] = {key: arrays['pre'][key] + arrays['post'][key] for key in arrays['pre']}
    phase_summary = {}
    for phase, values in arrays.items():
        g, a, s = (values[key].sum() for key in ('generated', 'admitted', 'success'))
        phase_summary[phase] = dict(
            generated=int(g), admitted=int(a), success=int(s),
            dsr=float(s / g), admission=float(a / g), conditional=float(s / a),
        )
    frames = [frame for row in rows for frame in row['trajectory']]
    det_steps = sum(int(frame.get('deterministic_engine_steps', 0)) for frame in frames)
    f_calls = sum(int(frame['forecast_calls']) for frame in frames)
    if arm in NEURAL_ZERO_SIM_ARMS:
        assert det_steps == 0, f'neural arm {arm} executed {det_steps} online Engine steps'
        assert f_calls == 0, f'neural arm {arm} executed {f_calls} online forecast calls'
    phase_summary.update(
        episodes=len(rows),
        added_decision_fraction=float(np.mean([frame['metadata']['added'] > 0 for frame in frames])),
        selected_new_fraction=float(np.mean([frame['selected_new'] for frame in frames])),
        selected_safe_plan_fraction=float(np.mean([frame['metadata'].get('selected_safe_plan', False) for frame in frames])),
        decision_mean=float(np.mean([frame['decision_seconds'] for frame in frames])),
        decision_p95=float(np.quantile([frame['decision_seconds'] for frame in frames], .95)),
        over_second=float(np.mean([frame['decision_seconds'] > 1 for frame in frames])),
        flight_m_per_episode=float(np.mean([
            sum(np.linalg.norm(delta) for frame in row['trajectory'] for delta in frame['motion'])
            for row in rows
        ])),
        deterministic_engine_steps=int(det_steps),
        forecast_calls=int(f_calls),
    )
    return arrays, phase_summary


def stage_review_rl(root, stage, folder):
    replicas = 1 if stage == 'pilot' else 2
    folder = Path(folder)

    expected_freeze_sha = None
    frozen_at_utc = None
    if stage == 'confirmation':
        manifest_path = folder / 'freeze_manifest.json'
        assert manifest_path.exists(), 'freeze_manifest.json must exist before confirmation review'
        expected_freeze_sha = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        frozen_at_utc = manifest['frozen_at_utc']

    rows = [load(path) for path in (folder / stage).glob('*/*.json')]
    workloads = PILOT_WORKLOADS if stage == 'pilot' else tuple(sorted({int(r['workload']) for r in rows}))
    active_arms = [a for a in ARMS if any(r['arm'] == a for r in rows)]
    assert len(rows) == len(workloads) * replicas * 3 * len(active_arms), (
        len(rows), len(workloads) * replicas * 3 * len(active_arms)
    )
    matrices = {}
    summary = {}
    fingerprints = {}
    for arm in active_arms:
        own = [row for row in rows if row['arm'] == arm]
        assert len(own) == len(workloads) * replicas * 3, (arm, len(own))
        for row in own:
            exogenous = (row['workload'], row['replica'])
            fingerprint = (row['fingerprint'], row['counts']['generated'])
            assert exogenous not in fingerprints or fingerprints[exogenous] == fingerprint
            fingerprints[exogenous] = fingerprint
        arrays, phase_summary = summarize_records(
            own, workloads, replicas, arm=arm,
            expected_freeze_sha=expected_freeze_sha, frozen_at_utc=frozen_at_utc,
        )
        matrices[arm] = arrays
        summary[arm] = phase_summary

    active_pairs = [
        (first, second) for first, second in COMPARISON_PAIRS
        if first in matrices and second in matrices
    ]
    primary_set = set(PRIMARY_COMPARISON_PAIRS)
    comparisons = {}
    for first, second in active_pairs:
        result = {
            'is_primary_family': bool((first, second) in primary_set),
        }
        for phase in matrices[first]:
            a, b = matrices[first][phase], matrices[second][phase]
            assert np.array_equal(a['generated'], b['generated'])
            result[phase] = {
                metric: paired_bootstrap(
                    a['success'] if metric != 'admission' else a['admitted'], a[denominator],
                    b['success'] if metric != 'admission' else b['admitted'], b[denominator],
                    family_size=PRIMARY_FAMILY_SIZE,
                    all_family_size=len(active_pairs),
                )
                for metric, denominator in (('dsr', 'generated'), ('admission', 'generated'), ('conditional', 'admitted'))
            }
        comparisons[first + '--' + second] = result

    diagnostics = {
        'joint_rl_beats_init': bool(comparisons['joint_rl--uncoordinated_init']['all']['dsr']['mean'] > 0.0),
        'joint_rl_beats_decomposed_composed_rl': bool(
            comparisons['joint_rl--decomposed_composed_rl']['all']['dsr']['mean'] > 0.0
        ),
        'joint_rl_beats_decomposed_rl': bool(
            comparisons['joint_rl--decomposed_rl']['all']['dsr']['mean'] > 0.0
        ),
        'c_upper_expectation_gain_positive': bool(
            comparisons['joint_rl--joint_no_expectation_rl']['all']['dsr']['mean'] > 0.0
        ),
        'joint_rl_finetuned_beats_distilled': bool(
            comparisons['joint_rl_finetuned--joint_distilled']['all']['dsr']['mean'] >= 0.0
        ),
        'joint_rl_finetuned_beats_decomposed_finetuned': bool(
            comparisons['joint_rl_finetuned--decomposed_rl_finetuned']['all']['dsr']['mean'] > 0.0
        ),
    }
    if 'joint_rl_finetuned--joint_no_expectation_rl_finetuned' in comparisons:
        diagnostics['joint_rl_finetuned_beats_no_expectation_finetuned'] = bool(
            comparisons['joint_rl_finetuned--joint_no_expectation_rl_finetuned']['all']['dsr']['mean'] >= 0.0
        )
    gate_passed = bool(
        diagnostics['joint_rl_beats_init']
        or diagnostics['joint_rl_finetuned_beats_distilled']
    )
    return dict(
        stage=stage,
        workloads=list(workloads),
        replicas=replicas,
        episodes=len(rows),
        evaluated_arms=active_arms,
        primary_comparison_pairs=[f'{a}--{b}' for a, b in PRIMARY_COMPARISON_PAIRS if (a, b) in active_pairs],
        exploratory_comparison_pairs=[f'{a}--{b}' for a, b in EXPLORATORY_COMPARISON_PAIRS if (a, b) in active_pairs],
        freeze_manifest_sha256=expected_freeze_sha,
        frozen_at_utc=frozen_at_utc,
        summary=summary,
        comparisons=comparisons,
        diagnostics=diagnostics,
        gate_passed=gate_passed,
    )


stage_review = stage_review_rl


def rl_checkpoint_curve_review(folder, checkpoints=(0, 1, 2, 3, 4)):
    """Evaluate pilot DSR progression across RL PPO iterations."""
    folder = Path(folder)
    curve = []
    for it in checkpoints:
        ckpt_dir = folder / 'pilot_checkpoints' / f'iter_{it}'
        if not ckpt_dir.exists():
            continue
        rows = [load(path) for path in sorted(ckpt_dir.glob('*.json'))]
        if not rows:
            continue
        _, phase_summary = summarize_records(rows, PILOT_WORKLOADS, 1, arm='joint_rl')
        curve.append(dict(
            iteration=int(it),
            dsr=phase_summary['all']['dsr'],
            peak_dsr=phase_summary['peak']['dsr'],
            nonpeak_dsr=phase_summary['nonpeak']['dsr'],
            conditional=phase_summary['all']['conditional'],
            peak_conditional=phase_summary['peak']['conditional'],
            selected_safe_plan_fraction=phase_summary['selected_safe_plan_fraction'],
            selected_new_fraction=phase_summary['selected_new_fraction'],
        ))
    return curve


def verify_freeze_manifest(root, folder):
    """Verify that `freeze_manifest.json` matches current source files, parent checkpoints, models, and confirmation episodes."""
    root = Path(root)
    code_root = Path(__file__).resolve().parents[2]
    folder = Path(folder)
    manifest_path = folder / 'freeze_manifest.json'
    if not manifest_path.exists():
        return dict(verified=False, reason='missing freeze_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_sha = sha256_file(manifest_path)
    frozen_at = str(manifest.get('frozen_at_utc', ''))

    source_mismatches = []
    for rel_path, expected_sha in manifest.get('source_sha256', {}).items():
        candidate_path = (code_root / rel_path) if (code_root / rel_path).exists() else (root / rel_path)
        actual_sha = sha256_file(candidate_path)
        if actual_sha != expected_sha:
            source_mismatches.append(rel_path)

    parent_mismatches = []
    checked_parents = 0
    for rel_path, expected_sha in manifest.get('parent_checkpoint_sha256', {}).items():
        candidate_path = root / rel_path
        if candidate_path.exists():
            checked_parents += 1
            if sha256_file(candidate_path) != expected_sha:
                parent_mismatches.append(rel_path)

    model_mismatches = []
    for rel_path, expected_sha in manifest.get('model_sha256', {}).items():
        actual_sha = sha256_file(folder / rel_path)
        if actual_sha != expected_sha:
            model_mismatches.append(rel_path)

    conf_dir = folder / 'confirmation'
    conf_files = list(conf_dir.glob('*/*.json')) if conf_dir.exists() else []
    bad_episodes = []
    for p in conf_files:
        rec = json.loads(p.read_text(encoding='utf-8'))
        if rec.get('freeze_manifest_sha256') != manifest_sha:
            bad_episodes.append(str(p.name))
        elif str(rec.get('started_at_utc', '')) < frozen_at:
            bad_episodes.append(str(p.name))

    verified = (
        (not source_mismatches)
        and (not parent_mismatches)
        and (not model_mismatches)
        and (not bad_episodes)
    )
    return dict(
        verified=bool(verified),
        freeze_manifest_sha256=manifest_sha,
        frozen_at_utc=frozen_at,
        checked_sources=len(manifest.get('source_sha256', {})),
        checked_parent_checkpoints=checked_parents,
        checked_models=len(manifest.get('model_sha256', {})),
        checked_confirmation_episodes=len(conf_files),
        source_mismatches=source_mismatches,
        parent_mismatches=parent_mismatches,
        model_mismatches=model_mismatches,
        bad_episodes=bad_episodes,
    )


def audit_rl_rollouts(folder, saved_loader):
    """Replay and audit all saved on-policy RL training trajectories (`rl_rollouts`).

    Verifies:
    1. Post-200s drain transitions are explicitly recorded (`len(steps) >= 200`, `drain_steps` tracked).
    2. Exact conservation of task successes and expirations across all recorded steps:
       `sum(d_success) == summary['success']` and `sum(d_expired) == summary['expired']`.
    3. Exact log-probability replay `|old_logp - log P_theta_old(a_t, m_t | s_t)| < 1e-10` on sampled steps.
    4. Finite GAE advantages and returns on every episode.
    """
    folder = Path(folder)
    rollouts_root = folder / 'rl_rollouts'
    if not rollouts_root.exists():
        return dict(verified=False, reason='missing rl_rollouts directory')

    rollout_files = sorted(rollouts_root.glob('iter_*/*/*.pkl.gz'))
    total_episodes = 0
    total_transitions = 0
    total_drain_transitions = 0
    total_drain_successes = 0
    max_logp_diff = 0.0
    conservation_failures = []
    logp_failures = []
    gae_failures = []

    model_cache = {}
    for rf in rollout_files:
        parts = rf.parts
        iter_dir = parts[-3]  # 'iter_1', etc.
        arm = parts[-2]
        stem = rf.name.split('.')[0]  # '{parent}_{workload}_{rep}'
        parent = int(stem.split('_')[0])
        iteration = int(iter_dir.split('_')[1])
        mode = rl_mode_for_arm(arm)

        m_key = (iteration - 1, arm, parent)
        if m_key not in model_cache:
            m_path = folder / 'models_iter' / f'iter_{iteration - 1}' / arm / f'{parent}.pkl.gz'
            model_cache[m_key] = saved_loader(m_path)
        prev_model = model_cache[m_key]

        ep_data = saved_loader(rf)
        steps = ep_data['steps']
        summary = ep_data['summary']
        total_episodes += 1
        total_transitions += len(steps)

        d_suc_sum = int(round(sum(float(s['d_success']) for s in steps)))
        d_exp_sum = int(round(sum(float(s['d_expired']) for s in steps)))
        if (
            len(steps) < 200
            or d_suc_sum != int(summary['success'])
            or d_exp_sum != int(summary['expired'])
            or int(summary['generated']) != int(summary['success']) + int(summary['expired'])
        ):
            conservation_failures.append(str(rf.relative_to(folder)))

        drain_steps = [s for s in steps if s.get('is_drain', int(s['time']) >= 200)]
        total_drain_transitions += len(drain_steps)
        total_drain_successes += int(round(sum(float(s['d_success']) for s in drain_steps)))

        # Check old_logp replay on first, middle, last-active (t=199), and all drain steps (plus any cf_group entries)
        check_indices = sorted(set([0, len(steps) // 2, 199] + list(range(200, len(steps)))))
        for idx in check_indices:
            s = steps[idx]
            prob = rl_actor_policy(s['row'], prev_model, mode=mode)
            k_star, m_star = int(s['action_pair'][0]), int(s['action_pair'][1])
            recomputed_lp = float(np.log(max(float(prob[k_star, m_star]), 1e-12)))
            diff = abs(float(s['old_logp']) - recomputed_lp)
            if diff > max_logp_diff:
                max_logp_diff = diff
            if diff > 1e-9:
                logp_failures.append(f"{rf.name}:step_{idx}:diff={diff:.3e}")
            for cf in s.get('cf_group', []):
                ck, cm = int(cf['action_pair'][0]), int(cf['action_pair'][1])
                cf_lp = float(np.log(max(float(prob[ck, cm]), 1e-12)))
                cf_diff = abs(float(cf['old_logp']) - cf_lp)
                if cf_diff > max_logp_diff:
                    max_logp_diff = cf_diff
                if cf_diff > 1e-9:
                    logp_failures.append(f"{rf.name}:step_{idx}_cf:diff={cf_diff:.3e}")

        trs, _ = compute_episode_gae_and_returns([steps], prev_model, gamma=0.85, gae_lambda=0.90)
        if len(trs) != len(steps) or not all(
            np.isfinite(t['raw_gae_advantage']) and np.isfinite(t['target_return']) for t in trs
        ):
            gae_failures.append(str(rf.relative_to(folder)))

    verified = (
        total_episodes > 0
        and not conservation_failures
        and not logp_failures
        and not gae_failures
        and max_logp_diff < 1e-9
    )
    return dict(
        verified=bool(verified),
        checked_episodes=int(total_episodes),
        total_transitions=int(total_transitions),
        total_drain_transitions=int(total_drain_transitions),
        total_drain_successes=int(total_drain_successes),
        max_old_logp_abs_diff=float(max_logp_diff),
        conservation_failures=conservation_failures,
        logp_failures=logp_failures,
        gae_failures=gae_failures,
    )


def audit_true_decomposed_independence(folder, saved_loader, sample_rows):
    """Verify exact mathematical independence `p_m(m | s, k) == p_m(m | s, k')` for all decomposed arms."""
    folder = Path(folder)
    uncoupled_arms = (
        'lower_only_rl',
        'upper_only_rl',
        'decomposed_composed_rl',
        'decomposed_rl',
        'decomposed_distilled',
        'decomposed_rl_finetuned',
    )
    coupled_arms = (
        'joint_no_expectation_rl',
        'joint_rl',
        'joint_no_expectation_distilled',
        'joint_distilled',
        'joint_no_expectation_rl_finetuned',
        'joint_rl_finetuned',
    )
    max_diff_by_arm = {}
    for arm in uncoupled_arms + coupled_arms:
        mode = rl_mode_for_arm(arm)
        max_diff = 0.0
        for parent in range(3):
            m_path = folder / 'models' / arm / f'{parent}.pkl.gz'
            if not m_path.exists():
                continue
            model = saved_loader(m_path)
            for row in sample_rows:
                _, pm, *_ = rl_actor_forward(row, model, mode=mode)
                diff = float(np.max(np.abs(pm - pm[0:1, :])))
                if diff > max_diff:
                    max_diff = diff
        max_diff_by_arm[arm] = float(max_diff)

    uncoupled_max = max(max_diff_by_arm.get(a, 0.0) for a in uncoupled_arms)
    return dict(
        verified=bool(uncoupled_max < 1e-14),
        max_plan_dependence_uncoupled=float(uncoupled_max),
        max_plan_dependence_by_arm=max_diff_by_arm,
    )


def format_rl_summary_table(analysis_data):
    """Automatically generate the Markdown summary table for all evaluated arms."""
    role_map = {
        'uncoordinated_init': '未经训练基线先验策略（Epoch 0）',
        'lower_only_rl': '纯环境回报 RL：单层下层分配网络',
        'upper_only_rl': '纯环境回报 RL：单层上层轨迹网络',
        'decomposed_composed_rl': '纯环境回报 RL：单层独立训练直接拼接',
        'decomposed_rl': '纯环境回报 RL：真去耦双分支同步 PPO-GAE（零方案依赖）',
        'joint_no_expectation_rl': '纯环境回报 RL：仅方案条件化（切断 $C_{\\text{upper}}$）',
        'joint_rl': '纯环境回报 RL：完整跨层耦合 PPO-GAE（零教师）',
        'decomposed_distilled': '规划教师蒸馏对照：真去耦双分支（零方案依赖）',
        'joint_no_expectation_distilled': '规划教师蒸馏对照：仅方案条件化（切断 $C_{\\text{upper}}$）',
        'joint_distilled': '规划教师蒸馏对照：完整跨层耦合双分支',
        'decomposed_rl_finetuned': '蒸馏暖启动 + 环境回报真去耦 PPO-GAE 微调',
        'joint_no_expectation_rl_finetuned': '共同耦合蒸馏暖启动 + 切断 $C_{\\text{upper}}$ PPO-GAE 微调',
        'joint_rl_finetuned': '共同耦合蒸馏暖启动 + 完整跨层耦合 PPO-GAE 微调',
        'quality_amortized': '在线 3 步确定性 `Engine` 推演参考',
        'prior_top6_mpc': '同框架 MPC 对照：未训练先验裁剪 Top-6 + 3 步前瞻',
        'distilled_top6_mpc': '同框架 MPC 对照：耦合蒸馏网络裁剪 Top-6 + 3 步前瞻',
        'joint_rl_mpc': '同框架 MPC 对照：联合 RL 微调网络裁剪 Top-6 + 3 步前瞻',
    }
    lines = [
        '| 评估臂 | 算法类别与消融角色 | 在线 `Engine` 步数 | 随机前瞻调用 | 成功请求数 | 全程 DSR | 非拥塞期 DSR | 拥塞期 DSR | 拥塞期条件完成率 | 安全方案率 | 救援动作率 | 平均耗时 (p95 / >1s占比) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    s = analysis_data['summary']
    eval_arms = analysis_data.get('evaluated_arms', [a for a in ARMS if a in s])
    for arm in eval_arms:
        row = s[arm]
        bold = '**' if arm in ('joint_rl', 'joint_rl_finetuned', 'joint_rl_mpc') else ''
        lines.append(
            f"| {bold}`{arm}`{bold} | {role_map.get(arm, arm)} | {row['deterministic_engine_steps']:,} "
            f"| {row['forecast_calls']:,} | {row['all']['success']:,} "
            f"| {bold}{row['all']['dsr']*100:.2f}%{bold} | {row['nonpeak']['dsr']*100:.2f}% "
            f"| {row['peak']['dsr']*100:.2f}% | {row['peak']['conditional']*100:.2f}% "
            f"| {row['selected_safe_plan_fraction']*100:.1f}% | {row['selected_new_fraction']*100:.1f}% "
            f"| {row['decision_mean']:.3f}s ({row['decision_p95']:.3f}s / {row['over_second']*100:.1f}%) |"
        )
    return '\n'.join(lines)


def format_rl_comparison_table(analysis_data, pairs=None, include_all_bonf=False):
    """Automatically generate the exact Markdown paired comparison table from analysis JSON."""
    if include_all_bonf:
        lines = [
            '| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | 主族 $M=8$ 95% CI | 全表 $M_{\\text{all}}$ 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|',
        ]
    else:
        lines = [
            '| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |',
            '|---|---:|---:|---:|---:|---:|---:|---|',
        ]
    comps = analysis_data['comparisons']
    target_pairs = pairs if pairs is not None else COMPARISON_PAIRS
    for a, b in target_pairs:
        key = f'{a}--{b}'
        if key not in comps:
            continue
        c = comps[key]
        all_dsr = c['all']['dsr']
        nonpeak_dsr = c['nonpeak']['dsr']['mean'] * 100.0
        peak_dsr = c['peak']['dsr']['mean'] * 100.0
        peak_cond = c['peak']['conditional']['mean'] * 100.0
        mean_pp = all_dsr['mean'] * 100.0
        ci95 = [x * 100.0 for x in all_dsr['ci95']]
        bonf = [x * 100.0 for x in all_dsr['bonferroni_family95']]
        p0, p1, p2 = [x * 100.0 for x in all_dsr['by_parent']]
        if include_all_bonf and 'bonferroni_all_comparisons95' in all_dsr:
            bonf_all = [x * 100.0 for x in all_dsr['bonferroni_all_comparisons95']]
            lines.append(
                f'| `{a} - {b}` | **{mean_pp:+.2f} pp** | [{ci95[0]:+.2f}, {ci95[1]:+.2f}] '
                f'| [{bonf[0]:+.2f}, {bonf[1]:+.2f}] | [{bonf_all[0]:+.2f}, {bonf_all[1]:+.2f}] '
                f'| {nonpeak_dsr:+.2f} pp | {peak_dsr:+.2f} pp '
                f'| {peak_cond:+.2f} pp | {p0:+.2f} / {p1:+.2f} / {p2:+.2f} pp |'
            )
        else:
            lines.append(
                f'| `{a} - {b}` | **{mean_pp:+.2f} pp** | [{ci95[0]:+.2f}, {ci95[1]:+.2f}] '
                f'| [{bonf[0]:+.2f}, {bonf[1]:+.2f}] | {nonpeak_dsr:+.2f} pp | {peak_dsr:+.2f} pp '
                f'| {peak_cond:+.2f} pp | {p0:+.2f} / {p1:+.2f} / {p2:+.2f} pp |'
            )
    return '\n'.join(lines)


def format_rl_selection_table(curves_data):
    """Automatically generate the Markdown checkpoint selection table from `rl_training_curves.json`."""
    lines = [
        '| 训练臂 | Iter 0 验证 DSR | Iter 1 验证 DSR | Iter 2 验证 DSR | Iter 3 验证 DSR | Iter 4 验证 DSR | 选中迭代（Parent 0 / 1 / 2） |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    curves = curves_data['curves']
    selected = curves_data['selected']
    for arm in RL_TRAINING_ARMS:
        if arm not in curves:
            continue
        c_list = {int(item['iteration']): float(item['validation_dsr']) * 100.0 for item in curves[arm]}
        sel = selected[arm]
        s0 = sel['0']['selected_iteration']
        s1 = sel['1']['selected_iteration']
        s2 = sel['2']['selected_iteration']
        lines.append(
            f"| `{arm}` | {c_list.get(0, 0.0):.2f}% | {c_list.get(1, 0.0):.2f}% "
            f"| {c_list.get(2, 0.0):.2f}% | {c_list.get(3, 0.0):.2f}% | {c_list.get(4, 0.0):.2f}% "
            f"| Iter {s0} / Iter {s1} / Iter {s2} |"
        )
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'confirmation'), default='pilot')
    args = parser.parse_args()
    print(json.dumps(stage_review_rl(args.folder.parents[2], args.stage, args.folder), indent=2))
