#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="${1:?server Python}"
cd "$root"
"$py" -m unittest discover -s tests -v > output/validation/tests.txt 2>&1
"$py" output/validation/validate_figure.py scripts/analysis/chapter5_oracle_gate_figure.py --backend python --json > output/validation/figure_source_audit.json
for name in oracle_tradeoff oracle_gain_distribution public_feature_bins; do
  "$py" output/validation/audit_pdf_text.py "results/attempt_01/figures/$name.pdf" --json > "output/validation/$name.text.json"
  "$py" output/validation/audit_figure_collisions.py "results/attempt_01/figures/$name.pdf" --json-out "output/validation/$name.collision.json" --overlay-pdf "output/validation/$name.collision.pdf"
done
echo ORACLE_FIGURE_QA_COMPLETE
