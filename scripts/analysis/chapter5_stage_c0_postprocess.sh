#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
archive="${1:?historical archive}"
development="${2:?stage B development}"
results="${3:?C0 results}"
py="${4:?server Python}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$py")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
validation="$root/output/validation"
while [[ ! -f "$results/fixed_complete.json" ]]; do
  if ! pgrep -u "$(id -u)" -f '[c]hapter5_stage_c0.py fixed' >/dev/null; then
    echo 'Fixed timing ended without a completion receipt; keep all raw records.' >&2
    exit 1
  fi
  sleep 2
done
"$py" "$root/scripts/analysis/chapter5_stage_c0_analysis.py" labels --archive "$archive" --development "$development" --output "$results" > "$validation/labels.log" 2>&1
"$py" "$root/scripts/analysis/chapter5_stage_c0_analysis.py" analyze --archive "$archive" --development "$development" --output "$results" > "$validation/analysis.log" 2>&1
"$py" "$root/scripts/analysis/chapter5_stage_c0_figure.py" --results "$results" --output "$results/figures" --qa-tools "$validation" > "$validation/figure.log" 2>&1
"$py" "$validation/validate_figure.py" "$root/scripts/analysis/chapter5_stage_c0_figure.py" --backend python --json > "$validation/figure_source_audit.json"
"$py" "$validation/audit_pdf_text.py" "$results/figures/c0_tradeoff.pdf" --json > "$validation/figure_text_audit.json"
echo C0_POSTPROCESS_COMPLETE
