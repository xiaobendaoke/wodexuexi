"""Run independent bridge jobs in disjoint shards and aggregate after completion."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location('frozen_bridge_runner', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load bridge runner: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    if args.shards < 1 or (not args.aggregate_only and
                           (args.shard_index is None or not 0 <= args.shard_index < args.shards)):
        raise ValueError('Invalid shard selection')
    runner_path = args.runner.resolve()
    suite_root = args.suite_root.resolve()
    output_root = args.output_root.resolve()
    runner = load_runner(runner_path)
    suite = json.loads((suite_root / 'suite.json').read_text())
    if suite['source_hash'] != runner.source_manifest()['source_hash']:
        raise ValueError('Frozen source changed')
    plan = suite['plan']
    seeds = plan['training_seeds']
    workloads = plan['workload_seeds']
    jobs = [(seed, workload) for seed in seeds for workload in workloads]
    if len(jobs) != 50:
        raise ValueError(f'Expected 50 paired jobs, found {len(jobs)}')
    if args.aggregate_only:
        paths = [output_root / f'seed_{seed}_workload_{workload}.json' for seed, workload in jobs]
        result = runner.aggregate(paths, plan, seeds, workloads,
                                  expected_per_job=2 * len(runner.FULL_SNAPSHOT_STEPS))
        runner.write_json(output_root / 'analysis.json', result)
        print(json.dumps({'status': result['status'], 'jobs': result['evaluation_jobs'],
                          'snapshots': result['snapshots']}, ensure_ascii=False), flush=True)
        return
    script_hash = runner.file_sha(runner_path)
    selected = [(index, pair) for index, pair in enumerate(jobs)
                if index % args.shards == args.shard_index]
    if args.limit is not None:
        selected = selected[:args.limit]
    for index, (seed, workload) in selected:
        path = output_root / f'seed_{seed}_workload_{workload}.json'
        reused = path.exists()
        runner.run_job(suite_root, output_root, plan, seed, workload, 2,
                       runner.FULL_SNAPSHOT_STEPS, script_hash)
        print(f'{"REUSED" if reused else "PASS"} shard={args.shard_index} '
              f'job={index + 1}/50 seed={seed} workload={workload}', flush=True)


if __name__ == '__main__':
    main()
