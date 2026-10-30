"""Run an immutable scientific entrypoint with verified execution extensions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

EXTENSION = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gradient-cpus', default='')
    parser.add_argument('--runtime-record', type=Path, required=True)
    args, scientific_arguments = parser.parse_known_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    # Both roots are namespace-package contributors; scientific files continue
    # to resolve from the frozen source, this new module from the extension.
    sys.path.insert(0, str(EXTENSION))
    from utils.dynamic_service_parallel import ParallelLowerUpdater, install_environment_cache
    os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    import torch
    import marl_models.dynamic_service_hmarl as scientific_model
    cpus = tuple(int(c) for c in args.gradient_cpus.split(',') if c)
    runtime = dict(version='execution_parallel_v1', scientific_source=str(source), gradient_cpus=cpus,
                   extension_sha256={p.relative_to(EXTENSION).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (EXTENSION/'utils/dynamic_service_parallel.py', Path(__file__))},
                   parent_pid=os.getpid(), parent_cpus=sorted(os.sched_getaffinity(0)))
    args.runtime_record.parent.mkdir(parents=True, exist_ok=True)
    def save_runtime():
        temporary = args.runtime_record.with_suffix('.tmp')
        temporary.write_text(json.dumps(runtime, indent=2))
        os.replace(temporary, args.runtime_record)
    save_runtime()
    pool = None
    original_update = scientific_model.update_lower

    def update_lower(policy, optimizer, transitions):
        nonlocal pool
        if not cpus:
            return original_update(policy, optimizer, transitions)
        if pool is None:
            pool = ParallelLowerUpdater(policy, cpus)
            runtime['gradient_workers'] = pool.identities
            runtime['pool_startup_seconds'] = pool.startup_seconds
        loss = pool.update(optimizer, transitions)
        runtime['last_update'] = pool.last_timing
        save_runtime()
        return loss

    scientific_model.update_lower = update_lower
    restore = install_environment_cache()
    original_save = torch.save

    def save(obj, *positionals, **keywords):
        if isinstance(obj, dict) and {'upper', 'lower', 'protocol', 'episode'}.issubset(obj):
            obj['execution_runtime'] = dict(runtime)
            if 'last_record' in obj:
                obj['last_record']['execution_runtime'] = dict(runtime)
        return original_save(obj, *positionals, **keywords)
    torch.save = save
    spec = importlib.util.spec_from_file_location('frozen_dynamic_service_entrypoint', source/'scripts/analysis/dynamic_service_experiment.py')
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    # The entrypoint derives REPO_ROOT from its own frozen source. Extension
    # files never enter or rewrite the original scientific source contract.
    sys.argv = [str(source/'scripts/analysis/dynamic_service_experiment.py'), *scientific_arguments]
    try:
        entrypoint.main()
    finally:
        if pool is not None:
            pool.close()
        restore()
        torch.save = original_save


if __name__ == '__main__':
    main()
