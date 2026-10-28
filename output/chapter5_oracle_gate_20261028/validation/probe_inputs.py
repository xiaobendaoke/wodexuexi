"""Read-only schema inspection. No outcome aggregation or optimization."""
from pathlib import Path
import gzip
import json
import sys

root = Path(sys.argv[1])
for name in ('protocol.json', 'labels_complete.json', 'fixed_complete.json'):
    obj = json.loads((root / name).read_text())
    print(name, {k: (len(obj[k]) if k == 'states' and isinstance(obj[k], list) else obj[k])
                 for k in ('name', 'status', 'blocks', 'states') if k in obj})
paths = sorted((root / 'fixed_labels').glob('*.json.gz'))
print('label_files', [p.name for p in paths])
with gzip.open(paths[0], 'rt') as f:
    data = json.load(f)
state = data[0]
print('state_keys', list(state))
print('identity_schema', state['identity'])
print('method_names', list(state['methods']))
print('method_fields', list(state['methods']['G']))
print('request_fields', list(state['methods']['G']['requests'][0]))
summary = json.loads((root / 'analysis/summary.json').read_text())
print('summary_keys', list(summary))
print('fixed_metric_keys', list(summary['fixed_snapshot']['G']))
