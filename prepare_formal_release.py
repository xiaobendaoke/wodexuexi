"""Create and validate an isolated source snapshot from a clean Git checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from utils.canonical_experiment import ROOT, source_manifest, write_json


def prepare(destination):
    if platform.system() != 'Linux':
        raise RuntimeError('Prepare and validate on the remote Linux server')
    import numpy as np
    import torch
    source = source_manifest()
    if source['git_clean'] is not True or not source['git_sha']:
        raise RuntimeError('Formal release requires a clean, committed Git checkout')
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    for relative in source['files']:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    report = dest / 'validation.json'
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (dest / 'validation.log').open('w') as log:
        subprocess.run([sys.executable, 'tests/full_stack_validation.py', '--report', str(report)],
                       cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    validation = json.loads(report.read_text())
    if validation['status'] != 'PASS' or validation['source_hash'] != source['source_hash']:
        raise RuntimeError('Release validation did not attest the copied source')
    lock = {**source, 'validation_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)},
            'source_directory': str(ROOT), 'release_directory': str(dest)}
    write_json(dest / 'source-lock.json', lock)
    for relative in source['files']:
        (dest / relative).chmod(0o444)
    (dest / 'source-lock.json').chmod(0o444)
    (dest / 'validation.json').chmod(0o444)
    print(json.dumps({'status': 'READY', 'release': str(dest), 'source_hash': source['source_hash']}))
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
