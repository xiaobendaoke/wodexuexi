"""Task-specific dispatch evidence: cleared development seeds only.

Imports the unchanged, previously tested frozen release. Confirmation is never
dispatched by this entrypoint; the original global collision report is retained.
"""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def candidates(seed_for, split, blocks):
    seeds = set()
    for block in range(blocks):
        seeds.add(seed_for(f'{split}/train', block))
        seeds.update(seed_for(f'{split}/workload/{block}', e) for e in range(400))
    seeds.update(seed_for(f'{split}/evaluation/{w}', e) for w in range(5) for e in range(6))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    source, task = args.source.resolve(), args.task.resolve()
    if not source.is_relative_to(task) or source.name != 'source_b4b20295':
        raise RuntimeError('unexpected frozen source')
    sys.path.insert(0, str(source))
    from scripts.analysis.dynamic_service_experiment import atomic_json, protocol, source_contract, METHODS
    from scripts.analysis.dynamic_service_pipeline import cores, run_jobs
    from utils.dynamic_service_scenarios import seed_for
    validation = task/'output/chapter5_dynamic_service_20261028/validation'
    output = task/'results/chapter5_dynamic_service_20261028'
    handle = (validation/'execution/controller.lock').open('a+')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle.seek(0); handle.truncate(); handle.write(str(os.getpid())); handle.flush()
    tests = json.loads((validation/'runtime_tests_b4b20295.json').read_text())
    if tests['status'] != 'PASS' or tests['tests'] != 15 or tests['source'] != source_contract():
        raise RuntimeError('frozen release no longer matches tested source')
    report_path = validation/'seed_audit.json'
    raw = report_path.read_bytes()
    audit = json.loads(raw)
    dev, confirm = candidates(seed_for, 'dev', 5), candidates(seed_for, 'confirm', 10)
    if len(dev) != 2035 or len(confirm) != 4040 or dev.intersection(confirm):
        raise RuntimeError('internal seed collision or incomplete seed registry')
    if audit['candidate_count'] != len(dev|confirm) or audit['excluded_large'] or audit['scanned'] <= 0:
        raise RuntimeError('incomplete historical seed audit')
    collisions = {seed for row in audit['collisions'] for seed in row['seeds']}
    if dev.intersection(collisions) or not collisions.issubset(confirm):
        raise RuntimeError('development not cleared by historical audit')
    if audit['status'] not in ('PASS', 'REQUIRES_REVIEW'):
        raise RuntimeError('unrecognized audit status')
    definition = protocol()
    contract = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU')
    cpu_ids = cores()[:5]
    if len(cpu_ids) != 5:
        raise RuntimeError('five distinct physical cores unavailable')
    atomic_json(validation/'development_seed_clearance.json', dict(
        status='PASS_DEVELOPMENT_ONLY', development_candidates=len(dev),
        original_audit_status=audit['status'], original_audit_sha256=hashlib.sha256(raw).hexdigest(),
        historical_files_scanned=audit['scanned'], development_collisions=[],
        confirmation_collisions=audit['collisions'], confirmation_dispatch=False,
        protocol=contract, source=definition['source']))
    atomic_json(validation/'development_launch.json', dict(
        pid=os.getpid(), started=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        status='DEVELOPMENT_RUNNING', protocol=contract, source=str(source), cpus=cpu_ids,
        command=sys.argv, dispatcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        confirmation_dispatch=False, scientific_claims=False))
    common = dict(contract=contract, stage='dev_train_full', validation=validation)
    for methods in (('full', 'independent'), ('no_memory', 'no_feedback', 'parallel')):
        common['stage'] = f'dev_train_{methods[0]}'
        jobs = []
        for block in range(5):
            for method in methods:
                jobs.append((f'train_dev_{block}_{method}',
                             ['train', '--split', 'dev', '--block', str(block), '--method', method],
                             output/'models/dev'/str(block)/(method+'.pt')))
        run_jobs(jobs, output, cpu_ids, 5, common)
    primary, other = [], []
    for kind in 'ABCD':
        for scale in (.6, 1., 1.4):
            for method in METHODS:
                for block in range(1 if method in ('history', 'mpc') else 5):
                    for workload in range(5):
                        for repeat in range(6):
                            identity = f'eval_dev_{block}_{method}_{kind}_{scale}_{workload}_{repeat}'
                            job = (identity, ['evaluate', '--split', 'dev', '--block', str(block), '--method', method,
                                             '--kind', kind, '--scale', str(scale), '--workload', str(workload),
                                             '--repeat', str(repeat)],
                                   output/'evaluation/dev'/str(block)/method/f'{kind}_{scale}_{workload}_{repeat}.json')
                            (primary if kind == 'D' and scale == 1. else other).append(job)
    common['stage'] = 'dev_primary_evaluation'
    run_jobs(primary, output, cpu_ids, 5, common)
    analyze = [sys.executable, str(source/'scripts/analysis/dynamic_service_experiment.py'),
               'analyze', '--split', 'dev', '--output', str(output)]
    subprocess.run(analyze+['--primary-only'], check=True, cwd=source)
    gate = json.loads((output/'development_gate.json').read_text())
    if gate['status'] != 'PASS':
        atomic_json(validation/'execution/final_status.json', dict(status='STOPPED_DEVELOPMENT_GATE',
                    confirmation_dispatch=False, next_step='diagnose development results'))
        return
    common['stage'] = 'dev_remaining_cells'
    run_jobs(other, output, cpu_ids, 5, common)
    subprocess.run(analyze, check=True, cwd=source)
    atomic_json(validation/'execution/final_status.json', dict(status='DEVELOPMENT_COMPLETE_CONFIRMATION_HELD',
                confirmation_dispatch=False, next_step='resolve unused confirmation workload seed collision'))


if __name__ == '__main__':
    main()
