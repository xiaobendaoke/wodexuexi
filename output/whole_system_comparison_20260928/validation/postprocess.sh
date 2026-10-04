#!/usr/bin/env bash
# Post-process the isolated remote comparison after the full suite finishes.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$task_root/source"
result_root="$task_root/results/main"
validation_root="$task_root/validation"
while [[ ! -f "$result_root/status.json" ]]; do
  sleep 60
done
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$result_root/status.json")"
if [[ "$status" != PASS ]]; then
  printf 'Full suite ended with %s; inspect results/main/status.json and validation/full.log\n' "$status" > "$validation_root/postprocess_status.txt"
  exit 1
fi
cd "$source_root"
bash run_remote_python.sh -m scripts.analysis.whole_system_comparison "$result_root" > "$validation_root/postprocess.log" 2>&1
python3 - "$result_root" "$validation_root" <<'PY'
import json
import sys
from pathlib import Path

result, validation = (Path(arg) for arg in sys.argv[1:])
suite = json.loads((result / 'suite.json').read_text())
analysis = json.loads((result / 'analysis.json').read_text())
paired = json.loads((result / 'whole_system_paired_comparison.json').read_text())
if not (analysis['status'] == paired['status'] == 'PASS'):
    raise SystemExit('Analysis did not pass')
if len(analysis['units']) != 325:
    raise SystemExit('Unexpected number of evaluation units')
markers = list(result.rglob('completed.json'))
if len(markers) != 385:
    raise SystemExit(f'Expected 385 completed markers, found {len(markers)}')
episodes = list(result.rglob('episodes/*.json'))
if len(episodes) != 60 * 200 + 325 * 6:
    raise SystemExit(f'Unexpected episode count: {len(episodes)}')
record = {'status': 'PASS', 'source_hash': suite['source_hash'],
          'protocol': suite['plan']['protocol'], 'training_jobs': 60,
          'evaluation_jobs': 325, 'completed_markers': len(markers),
          'episode_json_files': len(episodes),
          'decision_rule_met': paired['decision_rule_met']}
(validation / 'completion.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
PY
