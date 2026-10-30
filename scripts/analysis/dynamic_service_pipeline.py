"""Resumable bounded server dispatch; stop on failed validation/development gate."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import REPO_ROOT
ROOT = REPO_ROOT
from scripts.analysis.dynamic_service_experiment import TASK, TRAINED, METHODS, atomic_json, protocol
from utils.dynamic_service_scenarios import seed_for


def audit_seeds(roots):
    """Audit historical metadata including hidden smoke folders, not raw rows.

    Read seed-valued JSON keys, including large raw files and hidden smoke
    directories. Do not mistake request sizes, cycles, or durations for seeds.
    """
    candidates = set()
    for split, blocks in (('dev', 5), ('confirm', 10)):
        for b in range(blocks):
            candidates.add(seed_for(f'{split}/train', b))
            for e in range(400):
                candidates.add(seed_for(f'{split}/workload/{b}', e))
        for w in range(5):
            for e in range(6):
                candidates.add(seed_for(f'{split}/evaluation/{w}', e))
    collisions, scanned = [], 0
    needle = re.compile(rb'"[^"\r\n]{0,100}seed[^"\r\n]{0,100}"\s*:\s*([0-9]+|\[[0-9,\s]{0,16000}\])', re.I)
    seed_tokens = re.compile(rb'seed', re.I)
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob('*.json'):
            if TASK in str(path) or 'chapter5-dynamic-service-20261028' in str(path):
                continue
            hits, carry = set(), b''
            with path.open('rb') as stream:
                while True:
                    part = stream.read(1024*1024)
                    raw = carry+part
                    boundary = len(raw)-20000 if part else len(raw)
                    # Anchor the expensive key parser only at literal seed
                    # tokens, not every quote in multi-megabyte metric arrays.
                    for token in seed_tokens.finditer(raw):
                        left = raw.rfind(b'"', max(0, token.start()-101), token.start()+1)
                        if left < 0 or left >= boundary:
                            continue
                        match = needle.match(raw, left)
                        if match and match.start() <= token.start() < match.end():
                            hits.update(candidates.intersection(int(x) for x in re.findall(rb'[0-9]+', match.group(1))))
                    if not part:
                        break
                    carry = raw[max(boundary, 0):]
            scanned += 1
            if scanned % 1000 == 0:
                print(json.dumps(dict(stage='SEED_AUDIT', scanned=scanned, collisions=len(collisions))), flush=True)
            if hits:
                collisions.append(dict(path=str(path), seeds=sorted(hits)))
    return dict(status='PASS' if not collisions else 'REQUIRES_REVIEW',
                scanned=scanned, candidate_count=len(candidates), collisions=collisions,
                excluded_large=[])


def cores():
    physical, result = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity = ((topology/'physical_package_id').read_text(), (topology/'core_id').read_text())
        if identity not in physical:
            physical.add(identity); result.append(cpu)
    return result[8:]


def run_jobs(jobs, output, cpus, parallel, common):
    pending, active = list(jobs), []
    while pending or active:
        occupied = {x[2] for x in active}
        for cpu in cpus[:parallel]:
            if not pending or cpu in occupied:
                continue
            identity, arguments, expected = pending.pop(0)
            if expected.exists():
                if expected.suffix == '.json':
                    record = json.loads(expected.read_text())
                    if record.get('protocol') != common['contract']:
                        raise RuntimeError('cannot reuse mixed protocol result')
                    continue
                # Train command itself verifies complete/resumable checkpoint.
            log = common['validation']/'execution'/f'{identity}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open('a')
            command = ['taskset', '-c', str(cpu), sys.executable, str(ROOT/'scripts/analysis/dynamic_service_experiment.py'),
                       *arguments, '--output', str(output)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            active.append((process, handle, cpu, identity))
        for item in active[:]:
            process, handle, cpu, identity = item
            code = process.poll()
            if code is not None:
                handle.close(); active.remove(item)
                if code:
                    # Stop new dispatch; let independent in-flight jobs finish.
                    for running, file, _, _ in active:
                        running.wait(); file.close()
                    raise RuntimeError(f'{identity} failed ({code}); inspect its execution log')
        atomic_json(common['validation']/'execution/status.json', dict(stage=common['stage'], pending=len(pending),
                    active=[dict(pid=x[0].pid, cpu=x[2], job=x[3]) for x in active], time=time.time()))
        if active:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/TASK)
    parser.add_argument('--validation', type=Path, default=ROOT/'output'/TASK/'validation')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--history-root', type=Path, action='append', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('server only')
    import fcntl
    controller_lock = args.validation/'execution/controller.lock'
    controller_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = controller_lock.open('a+')
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_handle.seek(0); lock_handle.truncate(); lock_handle.write(str(os.getpid())); lock_handle.flush()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU')
    definition = protocol()
    contract = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    seed_audit = audit_seeds(args.history_root)
    atomic_json(args.validation/'seed_audit.json', seed_audit)
    if args.audit_only:
        print(json.dumps(dict(status=seed_audit['status'], scanned=seed_audit['scanned'],
                              collisions=seed_audit['collisions'][:10], excluded_large=len(seed_audit['excluded_large']))))
        return
    if seed_audit['status'] != 'PASS':
        raise RuntimeError('historical seed audit needs review before any training')
    cpus = cores()
    if not cpus or not 1 <= args.workers <= min(5, len(cpus)):
        parser.error('workers must fit at most five distinct available physical cores')
    subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_dynamic_service_scenarios',
                    'tests.test_dynamic_service_runtime'], cwd=ROOT, check=True)
    common = dict(contract=contract, stage='validation', validation=args.validation)
    atomic_json(args.validation/'launch_contract.json', dict(protocol=contract, cpus=cpus[:args.workers],
                workers=args.workers, source=definition['source'], python=sys.executable,
                audit_roots=[str(p) for p in args.history_root]))
    for split, blocks in (('dev', 5), ('confirm', 10)):
        # Full models first. Mechanism controls share their frozen counterpart.
        for methods in (('full', 'independent'), ('no_memory', 'no_feedback', 'parallel')):
            common['stage'] = f'{split}_train_{methods[0]}'
            jobs = []
            for b in range(blocks):
                for method in methods:
                    jobs.append((f'train_{split}_{b}_{method}', ['train', '--split', split, '--block', str(b), '--method', method],
                                 args.output/'models'/split/str(b)/(method+'.pt')))
            run_jobs(jobs, args.output, cpus, args.workers, common)
        common['stage'] = f'{split}_primary_evaluation'
        primary_jobs, other_jobs = [], []
        for kind in 'ABCD':
            for scale in (.6, 1., 1.4):
                for method in METHODS:
                    for b in range(1 if method in ('history', 'mpc') else blocks):
                        for w in range(5):
                            for e in range(6):
                                identity = f'eval_{split}_{b}_{method}_{kind}_{scale}_{w}_{e}'
                                job = (identity, ['evaluate', '--split', split, '--block', str(b), '--method', method,
                                                 '--kind', kind, '--scale', str(scale), '--workload', str(w), '--repeat', str(e)],
                                       args.output/'evaluation'/split/str(b)/method/f'{kind}_{scale}_{w}_{e}.json')
                                (primary_jobs if kind == 'D' and scale == 1. else other_jobs).append(job)
        run_jobs(primary_jobs, args.output, cpus, args.workers, common)
        analysis_command = [sys.executable, str(ROOT/'scripts/analysis/dynamic_service_experiment.py'),
                            'analyze', '--split', split, '--output', str(args.output)]
        subprocess.run(analysis_command+['--primary-only'], check=True)
        if split == 'dev':
            gate = json.loads((args.output/'development_gate.json').read_text())
            if gate['status'] != 'PASS':
                atomic_json(args.validation/'execution/final_status.json', dict(status='STOPPED_DEVELOPMENT_GATE',
                            next_step='diagnose development results; confirmation has not started'))
                return
        common['stage'] = f'{split}_remaining_cells'
        run_jobs(other_jobs, args.output, cpus, args.workers, common)
        subprocess.run(analysis_command, check=True)
    atomic_json(args.validation/'execution/final_status.json', dict(status='EXPERIMENTS_COMPLETE',
                remaining='independent result audit, figures, chapter draft, compilation'))


if __name__ == '__main__':
    main()
