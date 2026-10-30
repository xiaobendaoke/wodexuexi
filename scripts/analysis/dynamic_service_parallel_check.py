"""Server-only equivalence and throughput checks for execution extensions."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

EXTENSION = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graphs', type=int, default=64)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('numerical validation is server-only')
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(0, str(EXTENSION))
    os.environ.update(MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from environment.dynamic_service import DynamicServiceEnv, evaluate, prefix_batch
    from utils.dynamic_service_scenarios import Scenario
    from utils.dynamic_service_parallel import ParallelLowerUpdater, install_environment_cache
    from marl_models.dynamic_service_hmarl import LowerPolicy, update_lower
    physical, available = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((topology/'physical_package_id').read_text(), (topology/'core_id').read_text())
        if key not in physical:
            physical.add(key); available.append(cpu)
    worker_cpus = available[-9:-1]
    os.sched_setaffinity(0, {available[-1]})
    evidence = dict(status='RUNNING', scientific_claims=False, torch=torch.__version__,
                    worker_cpus=worker_cpus, comparisons=[], benchmark=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(evidence, indent=2))

    def trace(policy, count):
        env = DynamicServiceEnv(Scenario(81, 'D'))
        items = []
        for _ in range(count):
            batch = env.batch()
            action = policy.act(batch, stochastic=True)
            stats = env.step(np.zeros((5, 2)), action['assignment'])
            items.append(dict(batch=batch, **action, reward=stats['success']/max(len(batch), 1)))
        return items

    # Caching must preserve observations, stochastic actions, counters, geometry,
    # and the parent's random stream, not just the final service percentage.
    torch.manual_seed(7)
    cache_policy = LowerPolicy()
    def rollout():
        env = DynamicServiceEnv(Scenario(81, 'D'))
        torch.manual_seed(1234)
        rows = []
        started = time.perf_counter()
        for step in range(40):
            observation, batch = env.observe(), env.batch()
            action = cache_policy.act(batch, stochastic=True)
            stats = env.step(np.full((5, 2), .05), action['assignment'])
            rows.append((observation, batch, action, stats, env.positions.copy()))
        return rows, torch.get_rng_state(), time.perf_counter()-started
    baseline_rows, baseline_rng, baseline_time = rollout()
    restore = install_environment_cache()
    try:
        cached_rows, cached_rng, cached_time = rollout()
    finally:
        restore()
    assert torch.equal(baseline_rng, cached_rng)
    for before, after in zip(baseline_rows, cached_rows):
        for key in before[0]:
            np.testing.assert_array_equal(before[0][key], after[0][key])
        for key in before[1].__dataclass_fields__:
            np.testing.assert_array_equal(getattr(before[1], key), getattr(after[1], key))
        for key in before[2]:
            np.testing.assert_array_equal(before[2][key], after[2][key])
        assert before[3] == after[3]
        np.testing.assert_array_equal(before[4], after[4])
    evidence['environment_cache'] = dict(status='PASS_EXACT', baseline_seconds=baseline_time,
                                       cached_seconds=cached_time, steps=40)
    save()
    print(json.dumps(evidence['environment_cache']), flush=True)

    for mode in ('autoregressive', 'parallel', 'independent'):
        torch.manual_seed(7)
        base = LowerPolicy(mode)
        items = trace(base, 19)
        empty = prefix_batch(items[0]['batch'], 0)
        items.insert(3, dict(batch=empty, **base.act(empty), reward=0.))
        serial, parallel = copy.deepcopy(base), copy.deepcopy(base)
        serial_opt = torch.optim.Adam(serial.parameters(), lr=3e-4)
        parallel_opt = torch.optim.Adam(parallel.parameters(), lr=3e-4)
        state = torch.get_rng_state().clone()
        serial_loss = update_lower(serial, serial_opt, items)
        serial_rng = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        updater = ParallelLowerUpdater(parallel, worker_cpus[:2])
        try:
            parallel_loss = updater.update(parallel_opt, items)
            differences = {name: float((parameter-other).abs().max())
                           for (name, parameter), other in zip(serial.named_parameters(), parallel.parameters())}
            evidence['last_parameter_differences'] = differences
            save()
            print(json.dumps(dict(mode=mode, largest_differences=sorted(differences.items(), key=lambda x: -x[1])[:5])), flush=True)
            maximum = 0.
            for (name, a), b in zip(serial.named_parameters(), parallel.parameters()):
                maximum = max(maximum, float((a-b).abs().max()))
                # The final scalar bias adds the same constant to every legal
                # action logit. It is an unidentifiable softmax gauge; Adam can
                # amplify roundoff of its theoretically-zero gradient. Report
                # its raw difference, but test normalized probabilities below.
                if name != 'score.2.bias':
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5, msg=lambda message: name+': '+message)
            assert serial_opt.state_dict()['state'].keys() == parallel_opt.state_dict()['state'].keys()
            for index, serial_state in serial_opt.state_dict()['state'].items():
                other = parallel_opt.state_dict()['state'][index]
                for key in serial_state:
                    torch.testing.assert_close(serial_state[key], other[key], atol=3e-7, rtol=3e-5)
            assert torch.equal(serial_rng, torch.get_rng_state())
            assert updater.last_timing['optimizer_steps'] == 4*math.ceil(19/8)
            assert abs(serial_loss-parallel_loss) < 3e-6
            maximum_probability_difference, maximum_value_difference = 0., 0.
            for item in items:
                if not len(item['batch']):
                    continue
                with torch.no_grad():
                    a_log, a_value, _ = serial.replay(item['batch'], item['assignment'])
                    b_log, b_value, _ = parallel.replay(item['batch'], item['assignment'])
                torch.testing.assert_close(a_log, b_log, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(a_value, b_value, atol=3e-6, rtol=3e-5)
                maximum_probability_difference = max(maximum_probability_difference, float((a_log.exp()-b_log.exp()).abs().max()))
                maximum_value_difference = max(maximum_value_difference, float((a_value-b_value).abs().max()))
                state = torch.get_rng_state()
                a_action = serial.act(item['batch'], stochastic=True)
                torch.set_rng_state(state)
                b_action = parallel.act(item['batch'], stochastic=True)
                np.testing.assert_array_equal(a_action['assignment'], b_action['assignment'])
            row = dict(mode=mode, status='PASS', maximum_parameter_difference=maximum,
                       maximum_identifiable_parameter_difference=max(v for k,v in differences.items() if k != 'score.2.bias'),
                       softmax_common_bias_difference=differences['score.2.bias'],
                       maximum_probability_difference=maximum_probability_difference,
                       maximum_value_difference=maximum_value_difference,
                       serial_loss=serial_loss, parallel_loss=parallel_loss,
                       optimizer_steps=updater.last_timing['optimizer_steps'], rng_identical=True,
                       sampled_assignments_identical=True,
                       equivalence='functional/numerical; softmax-common bias gauge explicitly reported, not bitwise weights')
            evidence['comparisons'].append(row)
            save(); print(json.dumps(row), flush=True)
        finally:
            updater.close()

    torch.manual_seed(7)
    base = LowerPolicy()
    items = trace(base, args.graphs)
    initial = copy.deepcopy(base.state_dict())
    serial_times = []
    for _ in range(2):
        model = copy.deepcopy(base)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        started = time.perf_counter()
        update_lower(model, optimizer, items)
        serial_times.append(time.perf_counter()-started)
    evidence['serial_update_seconds'] = serial_times
    for workers in (1, 2, 4, 8):
        model = copy.deepcopy(base)
        updater = ParallelLowerUpdater(model, worker_cpus[:workers])
        try:
            timings = []
            for _ in range(2):
                model.load_state_dict(initial)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                started = time.perf_counter()
                updater.update(optimizer, items)
                timings.append(time.perf_counter()-started)
            row = dict(workers=workers, seconds=timings, startup_seconds=updater.startup_seconds,
                       speedup=float(np.median(serial_times)/np.median(timings)))
            evidence['benchmark'].append(row)
            save(); print(json.dumps(row), flush=True)
        finally:
            updater.close()
    best = min(evidence['benchmark'], key=lambda row: np.median(row['seconds']))
    # Select the smallest count within 5% of the measured best throughput.
    threshold = np.median(best['seconds'])*1.05
    acceptable = [r for r in evidence['benchmark'] if np.median(r['seconds']) <= threshold]
    evidence['selected_workers'] = min(r['workers'] for r in acceptable) if best['speedup'] > 1.05 else 0
    evidence['status'] = 'PASS'
    save()
    print(json.dumps(dict(status='PASS', selected_workers=evidence['selected_workers'])), flush=True)


if __name__ == '__main__':
    main()
