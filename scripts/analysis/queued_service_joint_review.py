"""Verify synchronized pilot metadata without running Torch on the local host."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis.queued_service_validation import atomic, now, sha
from scripts.analysis.queued_service_joint_short import (
    TASK, REVISION, METHODS, CONTROLS, locations, contract, protocol,
    evaluation_jobs, training_jobs, result_path)


def review(task):
    out, val = locations(task)
    analysis = json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    definition = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert definition == json.loads(json.dumps(protocol()))
    digest, files = contract(); assert digest == analysis['source_contract'] == definition['source_contract']
    completion = json.loads((val/'completion.json').read_text(encoding='utf-8'))
    tests = json.loads((val/'tests.json').read_text(encoding='utf-8'))
    assert completion['status'] == 'PASS_SHORT_TRIAL_COMPLETED' and completion['main_jobs'] == 960
    assert completion['source_contract'] == tests['source_contract'] == digest
    assert tests['status'] == 'PASS' and tests['tests'] == 49 and tests['skipped'] == 0
    assert not completion['changed_old_files'] and len(completion['used_cpus']) == 256
    assert completion['peak_active'] == 256 and not completion['automatic_expansion']
    jobs = [j for m in METHODS for b in range(3) for cp in (0, 3, 6) for j in evaluation_jobs(m, b, cp)]
    jobs += [j for m in CONTROLS for j in evaluation_jobs(m)]
    jobs += [j for m in METHODS for b in range(3) for r in range(1, 7) for j in training_jobs(m, b, r)]
    expected = {result_path(out, job).relative_to(out).as_posix() for job in jobs}
    assert len(expected) == 960 and expected == set(analysis['raw_sha256'])
    assert all(re.fullmatch(r'[0-9a-f]{64}', h) for h in analysis['raw_sha256'].values())
    summary = analysis['summary']; assert len(summary) == 84
    for r in summary:
        assert r['episodes'] == 8 and 0 <= r['success'] <= r['admitted'] <= r['generated']
        for key, expected_value in [('dsr', r['success']/r['generated']), ('admission', r['admitted']/r['generated']),
                ('conditional', r['success']/max(r['admitted'], 1))]:
            assert math.isclose(r[key], expected_value, abs_tol=1e-12)
        assert math.isclose(r['dsr'], r['admission']*r['conditional'], abs_tol=1e-12)
    updates = []
    for method in METHODS:
        for block in range(3):
            for cp in range(1, 7):
                record = json.loads((out/'training'/str(block)/method/f'{cp:04}.json').read_text(encoding='utf-8'))
                assert record['source_contract'] == digest and record['method'] == method and record['block'] == block and record['checkpoint'] == cp
                assert record['episodes'] == 8 and 1600 <= record['physical_steps'] <= 1640
                assert len(record['experience_files']) == 8
                assert set(record['experience_files']) == {result_path(out, j).relative_to(out).as_posix() for j in training_jobs(method, block, cp)}
                metric = record['update']; assert all(math.isfinite(v) for v in metric.values())
                assert metric['actor_steps'] > 0 and metric['allocation_parameter_change'] > 0 and metric['motion_parameter_change'] > 0
                updates.append(record)
    def lookup(method, block, cp):
        return next(r for r in summary if r['profile'] == 'burst' and r['method'] == method and r['block'] == block and r['checkpoint'] == cp)
    gains = {m: [lookup(m, b, 6)['dsr']-lookup(m, b, 0)['dsr'] for b in range(3)] for m in METHODS}
    comparison = [lookup('joint', b, 6)['dsr']-lookup('decomposed', b, 6)['dsr'] for b in range(3)]
    gates = {m: all(v > 0 for v in xs) and sum(xs)/3 >= .01 for m, xs in gains.items()}
    assert gains == analysis['gain_over_initialization'] and comparison == analysis['joint_minus_decomposed']
    assert gates == analysis['learning_gate'] == completion['learning_gate']
    deliver = task/'output'/TASK/'deliverables'
    with (deliver/'queued_joint_short_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        csvrows = list(csv.DictReader(handle))
    assert len(csvrows) == len(summary)
    for actual, expected_row in zip(csvrows, summary):
        for key, value in expected_row.items():
            if value is None: assert actual[key] == ''
            elif isinstance(value, (int, float)): assert math.isclose(float(actual[key]), value, abs_tol=1e-12)
            else: assert actual[key] == value
    report = deliver/'queued_joint_short.md'; links = []
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
        if '://' not in link and not link.startswith('#'):
            assert (report.parent/link.split('#')[0]).resolve().exists(), link; links.append(link)
    layout = subprocess.run([sys.executable, str(REPO_ROOT/'scripts/maintenance/check_layout.py')], cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONIOENCODING='utf-8'), capture_output=True)
    layout_log = layout.stdout.decode('utf-8')
    assert layout.returncode == 1 and len([s for s in layout_log.splitlines() if 'unexpected root files' in s]) == 2
    result = dict(status='PASS_LOCAL_METADATA_REVIEW_WITH_EXISTING_LAYOUT_ISSUES', checked_at=now(), source_contract=digest,
        source_files=files, server_tests=49, checked_training_update_records=len(updates), csv_cells=len(csvrows),
        server_audited_raw_episodes=960, local_raw_episode_audit=False,
        raw_boundary='Raw episodes, experience and model checkpoints remain on server; this verifies transferred metadata and source contract, not a second local raw-count audit',
        all_branches_changed=True, finite_updates=True, learning_gate=gates, completed=completion['completed'],
        logical_cpus=256, peak_active=256, old_files_preserved=completion['protected_files'], changed_old_files=[],
        report_links_checked=len(links), layout_exit_code=layout.returncode, layout_log=layout_log,
        independent_review=False, automatic_expansion=False)
    atomic(val/'local_review.json', result)
    atomic(val/'server_status_snapshot.json', dict(observed_via='Authenticated metadata archive; final server status', **completion))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files', 'layout_log')}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--task', type=Path, default=REPO_ROOT)
    review(parser.parse_args().task.resolve())
