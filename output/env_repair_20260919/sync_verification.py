"""Synchronize repair files to the isolated verification directory only."""
import hashlib, io, json, pathlib, shlex, subprocess, tarfile
root = pathlib.Path(__file__).resolve().parents[2]
remote = '/home/PengYanghan/Lunwen/env-repair-20260919-kBaVvp'
names = ['config.py', 'environment/env.py', 'environment/uavs.py', 'environment/user_equipments.py', 'marl_models/buffer_and_helpers.py', 'marl_models/joint_mappo/joint_mappo.py', 'marl_models/offload_mappo/offload_mappo.py', 'run_hierarchical_mappo_experiment.py', 'run_joint_end_to_end_mappo_experiment.py', 'run_canonical_experiment.py', 'utils/logger.py', 'utils/canonical_experiment.py', 'tests/test_global_service_batch.py', 'tests/test_environment_boundaries.py', 'tests/test_collision_fallback.py', 'tests/test_cache_lifecycle.py', 'tests/test_metric_semantics.py', 'tests/deterministic_smoke.py', 'tests/environment_validation_smoke.py', 'tests/test_lower_reward_causality.py', 'tests/test_policy_update_contracts.py', 'tests/test_lower_accounting.py', 'tests/test_scenario_reproducibility.py', 'tests/test_canonical_runner.py', 'tests/full_stack_validation.py']
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as bundle:
    for name in names:
        bundle.add(root / name, arcname=name)
subprocess.run(['ssh', '-o', 'BatchMode=yes', '100.69.44.85',
                'tar xzf - -C ' + shlex.quote(remote)], input=archive.getvalue(), check=True)
script = 'import pathlib,hashlib,json; root=pathlib.Path(' + repr(remote) + '); print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ' + repr(names) + '}))'
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '100.69.44.85',
                         '/home/PengYanghan/miniconda3/envs/drone/bin/python -c ' + shlex.quote(script)],
                        capture_output=True, text=True, check=True)
expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
assert json.loads(result.stdout) == expected
(root / 'output/env_repair_20260919/full_repair_source_hashes.json').write_text(json.dumps(expected, indent=2) + '\n')
print('Source hashes match for all transferred files.')
