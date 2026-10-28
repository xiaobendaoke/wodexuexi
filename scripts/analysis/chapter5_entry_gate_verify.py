"""Server engineering gate: native full/skip parity and one-G-only execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch

import chapter5_stage_b as b

ROOT = Path(__file__).resolve().parents[2]


def close(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for k in left:
            close(left[k], right[k])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            close(x, y)
    elif isinstance(left, float):
        assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10), (left, right)
    else:
        assert left == right, (left, right)


def worker(a):
    p = b.read(a.output/'protocol.json')
    os.sched_setaffinity(0, {p['cpus'][p['blocks'].index(a.block)]})
    _, runtime, fast, core, prices = b.boot(a, a.block)
    import config
    import numpy as np
    from utils.sf_hlm_priced_runtime import PricedPolicy
    from utils.sf_hlm_matching import feedback
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    from chapter5_entry_gate_policy import EntryGatePolicy, feature_pair
    from chapter5_gate_features import public_features
    rule = dict(mode='C', n0=0, q0=1.01, fallback_full=False)
    checked = []
    states = [s for s in p['states'] if s['identity'][0] == a.block]
    for s in states:
        path = Path(s['folder'])/'public.json'
        assert b.sha(path) == s['public_sha256']
        rows = b.read(path)['rows']
        for row in rows:
            row['options'] = {int(k): v for k, v in row['options'].items()}
        with fast.FastRuntime() as rt:
            native = PricedPolicy(rt, None, prices, prices['powers'])
            expected = native.run(rows, num_uavs=5, kind='mixture_price', cap=10, trace=True)
            rt.clear()
            gated = EntryGatePolicy(rt, prices, rule, force='full')
            with patch.object(rt, 'common_repair', wraps=rt.common_repair) as repair:
                actual = gated.run(rows, num_uavs=5, kind='mixture_price', cap=10, trace=True)
                assert repair.call_count == 1
            close(b.normal(actual), b.normal(expected))
            close(b.normal(feedback(rows, actual[0], 5, energy_reference=config.OFFLOAD_ENERGY_NORM_REF)),
                  b.normal(feedback(rows, expected[0], 5, energy_reference=config.OFFLOAD_ENERGY_NORM_REF)))
            rt.clear()
            expected_g = native.run(rows, num_uavs=5, kind='native_G', trace=True)
            rt.clear()
            skipped = EntryGatePolicy(rt, prices, rule, force='skip')
            with patch.object(rt, 'search_class', side_effect=AssertionError('Skip entered complex initialization')), \
                 patch.object(rt, 'common_repair', wraps=rt.common_repair) as repair:
                actual_g = skipped.run(rows, num_uavs=5, kind='mixture_price', cap=10, trace=True)
                assert repair.call_count == 1
            close(b.normal(actual_g), b.normal(expected_g))
            n, q = feature_pair(rows, actual_g[1])
            reference = public_features(rows, actual_g[0])
            assert n == reference['batch_size']
            assert q == reference['G_pred_success_fraction']
            assert not skipped.last_trace['entered']
        checked.append(dict(identity=s['identity'], status='PASS', matching_calls=actual[1]['matching_calls']))
    _, _, runner, models, _, _ = runtime
    blob = (Path(states[0]['folder'])/'observed.bin').read_bytes()
    full_checks = []
    for force in ('full', 'skip'):
        results = []
        for gated in (False, True):
            env, obs = pickle.loads(blob)
            with fast.FastRuntime() as rt:
                impl = EntryGatePolicy(rt, prices, rule, force=force) if gated else PricedPolicy(rt, None, prices, prices['powers'])
                rt.patch(core, 'truncated_search', impl.run)
                kind = 'mixture_price' if gated or force == 'full' else 'native_G'
                motion, actions, info, rows = rt.complete_policy(core, env, obs,
                    dict(kind=kind, cap=10, seed=123), models, None, runner)
                if gated:
                    assert info['policy_seconds'] >= impl.last_trace['g_seconds']+impl.last_trace['gate_seconds']
                with SoftwareServiceDelay(env, 'measured') as hook:
                    hook.arm(info['request_ids'], .0123)
                    nxt, reward, metrics = env.step(motion, offloading_actions=actions)
                    outcome = hook.finish()
                results.append((motion, actions, info, rows, nxt, reward, metrics, outcome))
        left, right = results
        assert np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1])
        for key in ('assignment', 'request_ids', 'diagnostic'):
            close(b.normal(left[2][key]), b.normal(right[2][key]))
        close(b.normal(left[3:]), b.normal(right[3:]))
        full_checks.append(dict(force=force, status='PASS', common_injected_seconds=.0123,
                                gate_cost_inside_complete_timer=True))
    with fast.FastRuntime() as rt:
        impl = EntryGatePolicy(rt, prices, rule, force='skip')
        with patch.object(rt, 'search_class', side_effect=AssertionError('Empty batch initialized search')):
            action, diagnostic, _ = impl.run([], num_uavs=5, kind='mixture_price', cap=10)
        assert len(action) == 0 and diagnostic['matching_calls'] == 0
    b.write(a.output/'blocks'/f'{a.block}.json', dict(status='PASS', states=len(checked),
        checks=checked, full_call_checks=full_checks, empty_batch='PASS'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--c0', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--block', type=int)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Engineering numerical checks execute on server')
    if a.block:
        return worker(a)
    if a.output.exists():
        raise FileExistsError('Preserve previous verification attempts')
    c0 = b.read(a.c0/'protocol.json')
    files = ['scripts/analysis/chapter5_entry_gate_policy.py', 'scripts/analysis/chapter5_entry_gate_verify.py',
             'scripts/analysis/chapter5_stage_b.py', 'scripts/analysis/chapter5_gate_features.py']
    inputs = dict(c0['inputs'])
    for path in (a.c0/'protocol.json', a.selection/'rules.json'):
        inputs[str(path)] = b.sha(path)
    for path, digest in inputs.items():
        assert b.sha(path) == digest
    b.write(a.output/'protocol.json', dict(name='entry_gate_native_parity', states=c0['states'],
        blocks=c0['blocks'], cpus=c0['cpus'][:5], inputs=inputs,
        files={f: b.sha(ROOT/f) for f in files}, no_performance_inference_from_common_fee_tests=True))
    logs = a.output/'logs'
    logs.mkdir(parents=True)
    def launch(block):
        cmd = [sys.executable, str(Path(__file__).resolve()), '--archive', str(a.archive),
               '--development', str(a.development), '--c0', str(a.c0), '--selection', str(a.selection),
               '--output', str(a.output), '--block', str(block)]
        with (logs/f'{block}.log').open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return b.read(a.output/'blocks'/f'{block}.json')
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [f.result() for f in as_completed([pool.submit(launch, x) for x in c0['blocks']])]
    assert sum(r['states'] for r in results) == 1200
    for path, digest in inputs.items():
        assert b.sha(path) == digest
    b.write(a.output/'completion.json', dict(status='PASS', states=1200, blocks=5,
        common_fee_full_call_checks=10, frozen_inputs_unchanged=True,
        full_branch_candidate_action_feedback_parity=True, skip_branch_G_parity=True,
        G_repair_calls_per_gate=1, numerical_public_feature_parity=True,
        timing_performance_claim=False))
    print(json.dumps(b.read(a.output/'completion.json')), flush=True)


if __name__ == '__main__':
    main()
