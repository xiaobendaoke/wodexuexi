"""Frozen experiment matrix and verified-source release contract."""
import hashlib
import json
from pathlib import Path
import platform

TRAIN_SEEDS = [42, 84, 126, 168, 210]
WORKLOAD_SEEDS = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]
LEARNED = ['hierarchical', 'vanilla_hierarchical', 'ippo', 'joint',
           'upper_attention', 'upper_vanilla', 'lower_no_attention',
           'no_attention', 'no_quality_mask', 'no_penalty']
NONLEARNED = ['greedy', 'local', 'mbs', 'random', 'uniform']


def verify_completed_job(directory, source_hash):
    """Read a completed job only when its files and source match this suite."""
    directory = Path(directory)
    done = json.loads((directory / 'completed.json').read_text())
    if done.get('status') != 'PASS':
        raise ValueError(f'Incomplete result: {directory}')
    for relative, expected in done['hashes'].items():
        path = directory / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Altered result: {path}')
    attempt = directory / done['attempt']
    manifest = json.loads((attempt / 'manifest.json').read_text())
    if manifest['source']['source_hash'] != source_hash:
        raise ValueError(f'Mixed source versions: {directory}')
    return attempt


def verify_release(lock_path, report_path, source):
    if not lock_path or not report_path:
        raise ValueError('Formal runs require a verified source release and validation report')
    lock = json.loads(Path(lock_path).read_text())
    report_bytes = Path(report_path).read_bytes()
    report = json.loads(report_bytes)
    if (lock.get('source_hash') != source['source_hash']
            or lock.get('files') != source['files']
            or lock.get('validation_sha256') != hashlib.sha256(report_bytes).hexdigest()
            or report.get('source_hash') != source['source_hash']
            or report.get('status') != 'PASS'
            or lock.get('git_clean') is not True
            or not lock.get('git_sha')):
        raise ValueError('Release source or validation evidence changed')
    import numpy as np
    import torch
    runtime = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)}
    if lock.get('runtime') != runtime:
        raise ValueError('Runtime changed since release validation; prepare a new release')
    return lock


def matrix(suite='all', smoke=False):
    """Sensitivity is retrained at each setting; no incompatible checkpoint reuse."""
    variants = []
    if suite in ('all', 'main'):
        variants.append({'name': 'main', 'ue_count': 100, 'compute_scale': 1.0,
                         'methods': LEARNED + NONLEARNED})
    if suite in ('all', 'sensitivity'):
        for count in [60, 80, 100, 120, 140]:
            variants.append({'name': f'ue_{count}', 'ue_count': count, 'compute_scale': 1.0,
                             'methods': ['hierarchical', 'vanilla_hierarchical', 'greedy']})
        for scale in [.6, .8, 1., 1.2, 1.4]:
            variants.append({'name': f'compute_{scale:g}', 'ue_count': 100, 'compute_scale': scale,
                             'methods': ['hierarchical', 'vanilla_hierarchical', 'greedy']})
    if not variants:
        raise ValueError('Unknown suite')
    return {'protocol': 'canonical_formal_v1', 'purpose': 'verification' if smoke else 'formal',
            'training_seeds': [42] if smoke else TRAIN_SEEDS,
            'workload_seeds': [294] if smoke else WORKLOAD_SEEDS,
            'training_workload_seed': 2026, 'train_episodes': 1 if smoke else 200,
            'eval_episodes': 1 if smoke else 6, 'steps': 4 if smoke else 1000,
            'rollout': 4 if smoke else 1000, 'ppo_epochs': 1 if smoke else 8,
            'batch_size': 2 if smoke else 256, 'checkpoint_every': 20,
            'variants': variants,
            'sensitivity_interpretation': 'retrain per setting; fixed pretrained transfer is a separate experiment',
            'statistical_unit': 'training seed after averaging paired workloads; evaluation episodes are not independent training replicates'}
