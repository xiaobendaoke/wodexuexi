#!/usr/bin/env bash
# Run on the remote server after the frozen pilot suite starts.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$task_root/source"
result_root="$task_root/results/pilot"
validation_root="$task_root/validation"

while [[ ! -f "$result_root/status.json" ]]; do
    sleep 30
done

status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$result_root/status.json")"
if [[ "$status" != PASS ]]; then
    python3 - "$validation_root/postprocess_status.json" "$status" <<'PY'
import json,sys
open(sys.argv[1], 'w').write(json.dumps({'status': 'FAILED', 'suite_status': sys.argv[2]}, indent=2) + '\n')
PY
    exit 1
fi

cd "$source_root"
PYTHONPATH="$source_root" bash run_remote_python.sh "$validation_root/analyze_refinement.py" "$result_root"
python3 - "$result_root" "$validation_root/postprocess_status.json" <<'PY'
import json,sys
from pathlib import Path
result, output = (Path(value) for value in sys.argv[1:])
suite = json.loads((result / 'suite.json').read_text())
paired = json.loads((result / 'observation_refinement_paired_comparison.json').read_text())
training_jobs = list(result.glob('main/*/seed_*/train/completed.json'))
evaluation_jobs = list(result.glob('main/*/seed_*/workload_*/completed.json'))
if (len(training_jobs), len(evaluation_jobs)) != (15, 75):
    raise SystemExit('Incomplete training/evaluation job markers')
if paired['status'] != 'PASS' or paired['training_source_hash'] != suite['source_hash']:
    raise SystemExit('Paired result/source mismatch')
record = {'status': 'PASS', 'protocol': suite['plan']['protocol'],
          'source_hash': suite['source_hash'], 'training_jobs': len(training_jobs),
          'evaluation_jobs': len(evaluation_jobs),
          'selected_for_confirmation': paired['pilot_selected_for_independent_confirmation']}
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
PY
