"""Count-based paired analyses for upper-lower joint neural training and unconfounded ablation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.analysis.review_queued_waiting_rescue import load
from utils.queued_joint_training import ALL_EVAL_ARMS, NEURAL_EVAL_ARMS

PILOT_WORKLOADS = tuple(range(951000, 951004))
CONFIRMATION_WORKLOADS = tuple(range(953000, 953008))
BOOTSTRAP_DRAWS = 10000
PRIMARY_FAMILY_SIZE = 7
ARMS = ALL_EVAL_ARMS


def paired_bootstrap(a, da, b, db, draws=BOOTSTRAP_DRAWS, family_size=PRIMARY_FAMILY_SIZE, seed=2026110501):
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
    ci95 = np.quantile(samples, [0.025, 0.975]).tolist()
    ci90 = np.quantile(samples, [0.05, 0.95]).tolist()
    bonf95 = np.quantile(samples, [alpha_bonf, 1.0 - alpha_bonf]).tolist()
    return dict(
        mean=float(effects.mean()),
        by_parent=effects.tolist(),
        draws=int(draws),
        ci90=ci90,
        ci95=ci95,
        simultaneous97_5=np.quantile(samples, [0.0125, 0.9875]).tolist(),
        bonferroni_family95=bonf95,
        tost_equivalent_0_50pp=bool(ci90[0] > -0.005 and ci90[1] < 0.005),
        tost_equivalent_0_75pp=bool(ci90[0] > -0.0075 and ci90[1] < 0.0075),
        noninferior_0_50pp_ci95=bool(ci95[0] > -0.005),
        noninferior_0_75pp_ci95=bool(ci95[0] > -0.0075),
        scope=(
            f'Conditional on 3 fixed parent lineages, paired block bootstrap over {workloads} '
            f'exogenous workloads x {replicas} action streams (B={draws}, Bonferroni M={family_size}).'
        ),
    )


def summarize_records(rows, workloads, replicas, arm=None):
    arrays = {
        phase: {key: np.zeros((3, len(workloads), replicas)) for key in ('generated', 'admitted', 'success')}
        for phase in ('all', 'pre', 'peak', 'post')
    }
    for row in rows:
        counts = row['counts']
        assert row['audit']['status'] == 'PASS'
        assert counts['generated'] == counts.get('success', 0) + counts.get('expired', 0)
        assert sum(row['successes']) == counts.get('success', 0)
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
    if arm in NEURAL_EVAL_ARMS:
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


def stage_review(root, stage, folder):
    workloads = PILOT_WORKLOADS if stage == 'pilot' else CONFIRMATION_WORKLOADS
    replicas = 1 if stage == 'pilot' else 2
    folder = Path(folder)
    rows = [load(path) for path in (folder / stage).glob('*/*.json')]
    assert len(rows) == len(workloads) * replicas * 3 * len(ARMS), (len(rows), len(workloads) * replicas * 3 * len(ARMS))
    matrices = {}
    summary = {}
    fingerprints = {}
    for arm in ARMS:
        own = [row for row in rows if row['arm'] == arm]
        assert len(own) == len(workloads) * replicas * 3, (arm, len(own))
        for row in own:
            exogenous = (row['workload'], row['replica'])
            fingerprint = (row['fingerprint'], row['counts']['generated'])
            assert exogenous not in fingerprints or fingerprints[exogenous] == fingerprint
            fingerprints[exogenous] = fingerprint
        arrays, phase_summary = summarize_records(own, workloads, replicas, arm=arm)
        matrices[arm] = arrays
        summary[arm] = phase_summary

    pairs = (
        ('joint_trained', 'uncoordinated_init'),
        ('joint_trained', 'lower_only_trained'),
        ('joint_trained', 'upper_only_trained'),
        ('joint_trained', 'decomposed_composed'),
        ('joint_trained', 'decomposed_simultaneous'),
        ('joint_trained', 'old_plan'),
        ('joint_trained', 'quality_amortized'),
        ('decomposed_composed', 'lower_only_trained'),
        ('decomposed_composed', 'upper_only_trained'),
        ('decomposed_composed', 'uncoordinated_init'),
        ('decomposed_simultaneous', 'decomposed_composed'),
        ('decomposed_simultaneous', 'uncoordinated_init'),
        ('upper_only_trained', 'uncoordinated_init'),
        ('lower_only_trained', 'uncoordinated_init'),
        ('joint_neural_mpc', 'joint_trained'),
        ('joint_neural_mpc', 'decomposed_composed'),
        ('joint_neural_mpc', 'old_plan'),
        ('joint_neural_mpc', 'quality_amortized'),
    )
    comparisons = {}
    for first, second in pairs:
        result = {}
        for phase in matrices[first]:
            a, b = matrices[first][phase], matrices[second][phase]
            assert np.array_equal(a['generated'], b['generated'])
            result[phase] = {
                metric: paired_bootstrap(
                    a['success'] if metric != 'admission' else a['admitted'], a[denominator],
                    b['success'] if metric != 'admission' else b['admitted'], b[denominator],
                )
                for metric, denominator in (('dsr', 'generated'), ('admission', 'generated'), ('conditional', 'admitted'))
            }
        comparisons[first + '--' + second] = result

    gates = {
        'joint_beats_init': bool(comparisons['joint_trained--uncoordinated_init']['all']['dsr']['mean'] >= .025),
        'joint_beats_lower_only': bool(comparisons['joint_trained--lower_only_trained']['all']['dsr']['mean'] >= .01),
        'joint_beats_upper_only': bool(comparisons['joint_trained--upper_only_trained']['all']['dsr']['mean'] >= .01),
        'joint_beats_decomposed_composed': bool(comparisons['joint_trained--decomposed_composed']['all']['dsr']['mean'] > 0.0),
        'joint_beats_decomposed_simultaneous': bool(comparisons['joint_trained--decomposed_simultaneous']['all']['dsr']['mean'] > 0.0),
        'joint_beats_old_plan': bool(comparisons['joint_trained--old_plan']['all']['dsr']['mean'] >= .005),
        'lower_only_improves_peak_cond': bool(
            comparisons['lower_only_trained--uncoordinated_init']['peak']['conditional']['mean'] > 0.0
            and comparisons['decomposed_composed--upper_only_trained']['peak']['conditional']['mean'] >= .015
        ),
        'upper_only_improves_nonpeak_dsr': bool(
            comparisons['upper_only_trained--uncoordinated_init']['nonpeak']['dsr']['mean'] >= .025
        ),
    }
    return dict(
        stage=stage,
        workloads=list(workloads),
        replicas=replicas,
        episodes=len(rows),
        summary=summary,
        comparisons=comparisons,
        gates=gates,
        gate_passed=bool(all(gates.values())),
    )


def checkpoint_curve_review(folder):
    folder = Path(folder)
    ckpt_dir = folder / 'pilot_checkpoints'
    if not ckpt_dir.exists():
        return {}
    workloads = PILOT_WORKLOADS
    curve = []
    for sub in sorted(ckpt_dir.iterdir(), key=lambda p: int(p.name.removeprefix('epoch_'))):
        if not sub.is_dir():
            continue
        epoch = int(sub.name.removeprefix('epoch_'))
        rows = [load(p) for p in sub.glob('*.json')]
        if len(rows) != len(workloads) * 3:
            continue
        _, summary = summarize_records(rows, workloads, 1, arm='joint_trained')
        curve.append(dict(
            epoch=epoch,
            all_dsr=summary['all']['dsr'],
            nonpeak_dsr=summary['nonpeak']['dsr'],
            peak_dsr=summary['peak']['dsr'],
            peak_conditional=summary['peak']['conditional'],
            selected_safe_plan_fraction=summary['selected_safe_plan_fraction'],
            selected_new_fraction=summary['selected_new_fraction'],
            deterministic_engine_steps=summary['deterministic_engine_steps'],
            forecast_calls=summary['forecast_calls'],
        ))
    return curve


def lindley_counterexample_audit():
    """Reproduce the 1-UAV, 2-task cross-stage counterexample showing static Lindley recursion
    is an approximation (reserving CPU before upload finishes) rather than exact work-conserving EDF.
    """
    from environment.queued_service import Engine, Protocol, Task
    from utils.queued_joint_training import _lindley_plan_motion_eval

    proto = Protocol(capacity=(100e9, 200e9), uplink_bps=10e6, inter_bps=10e6, backhaul_bps=10e6)
    engine = Engine(proto, [(200.0, 200.0)], {0: (200.0, 200.0), 1: (200.0, 200.0)})
    # Task 1: earlier deadline (1.4s), still in 'upload' (needs 0.7s upload + 0.5s CPU -> finishes at 1.2s)
    # Task 2: later deadline (1.5s), already in 'compute' (0.0s upload + 0.6s CPU -> runs during [0.0, 0.6] while Task 1 uploads!)
    t1 = Task(1, 0, 0.0, 1.4, 7e6, 50e9)
    t2 = Task(2, 1, 0.0, 1.5, 0.0, 60e9)
    engine.add([t1, t2])
    engine.admit()
    engine.assign({1: 0, 2: 0})

    obs = engine.observe()
    admitted = sorted([t for t in obs['tasks'] if t['source'] is not None], key=lambda z: (z['deadline'], z['uid']))
    lindley_comp, lindley_over = _lindley_plan_motion_eval(obs, admitted, [], {1: 0, 2: 0}, [(0.0, 0.0)])

    for _ in range(2):
        engine.advance([(0.0, 0.0)])
    audit = engine.audit()
    return dict(
        engine_success=int(engine.counts['success']),
        engine_expired=int(engine.counts['expired']),
        engine_audit=audit['status'],
        static_lindley_predicted_success=float(lindley_comp),
        static_lindley_predicted_overflow=float(lindley_over),
    )


def format_markdown_comparison_table(confirmation_data):
    """Generate the exact Markdown comparison table directly from confirmation_analysis.json without manual transcription."""
    ordered_pairs = (
        ('joint_trained', 'uncoordinated_init'),
        ('joint_trained', 'lower_only_trained'),
        ('joint_trained', 'upper_only_trained'),
        ('joint_trained', 'decomposed_composed'),
        ('joint_trained', 'decomposed_simultaneous'),
        ('joint_trained', 'old_plan'),
        ('joint_trained', 'quality_amortized'),
        ('decomposed_composed', 'lower_only_trained'),
        ('decomposed_composed', 'upper_only_trained'),
        ('decomposed_composed', 'uncoordinated_init'),
        ('decomposed_simultaneous', 'decomposed_composed'),
        ('upper_only_trained', 'uncoordinated_init'),
        ('lower_only_trained', 'uncoordinated_init'),
        ('joint_neural_mpc', 'joint_trained'),
        ('joint_neural_mpc', 'quality_amortized'),
        ('joint_neural_mpc', 'decomposed_composed'),
        ('joint_neural_mpc', 'old_plan'),
    )
    lines = [
        '| 对比对（方法 A − 方法 B） | 全程 DSR 差值 | 单比较 95% CI | Bonferroni 族 95% CI | 非拥塞期 DSR 差值 | 拥塞期 DSR 差值 | 拥塞期条件完成率差值 | 3 个父谱系差值（Parent 0 / 1 / 2） |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    comps = confirmation_data['comparisons']
    for a, b in ordered_pairs:
        key = f'{a}--{b}'
        c = comps[key]
        all_dsr = c['all']['dsr']
        nonpeak_dsr = c['nonpeak']['dsr']['mean'] * 100.0
        peak_dsr = c['peak']['dsr']['mean'] * 100.0
        peak_cond = c['peak']['conditional']['mean'] * 100.0
        mean_pp = all_dsr['mean'] * 100.0
        ci95 = [x * 100.0 for x in all_dsr['ci95']]
        bonf = [x * 100.0 for x in all_dsr['bonferroni_family95']]
        p0, p1, p2 = [x * 100.0 for x in all_dsr['by_parent']]
        lines.append(
            f'| `{a} - {b}` | **{mean_pp:+.2f} pp** | [{ci95[0]:+.2f}, {ci95[1]:+.2f}] '
            f'| [{bonf[0]:+.2f}, {bonf[1]:+.2f}] | {nonpeak_dsr:+.2f} pp | {peak_dsr:+.2f} pp '
            f'| {peak_cond:+.2f} pp | {p0:+.2f} / {p1:+.2f} / {p2:+.2f} pp |'
        )
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'confirmation'), default='pilot')
    args = parser.parse_args()
    print(json.dumps(stage_review(args.folder.parents[2], args.stage, args.folder), indent=2))
