#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
archive="${1:?pass external historical archive root}"
results="${2:?pass frozen result namespace}"
py="${3:?pass server Python executable}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$py")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
validation="$task_root/output/validation"
mkdir -p "$validation"
while [[ ! -f "$results/collection_complete.json" ]]; do
  if ! pgrep -u "$(id -u)" -f '[c]hapter5_stage_b.py run' >/dev/null; then
    echo 'Collector ended without completion receipt; retain all partial data.' >&2
    exit 1
  fi
  sleep 2
done
sha256sum "$task_root/scripts/analysis/"*.py "$task_root/tests/test_chapter5_stage_b.py" > "$validation/final_source_sha256.txt"
"$py" "$task_root/tests/test_chapter5_stage_b.py" --archive "$archive" --output "$results" > "$validation/targeted_tests.txt" 2>&1
"$py" "$task_root/scripts/analysis/chapter5_stage_b_analysis.py" closed-loop --archive "$archive" --output "$results" > "$validation/closed_loop_analysis.txt" 2>&1
"$py" "$task_root/scripts/analysis/chapter5_stage_b_analysis.py" fixed --archive "$archive" --output "$results" > "$validation/fixed_analysis.txt" 2>&1
"$py" "$task_root/scripts/analysis/chapter5_stage_b_analysis.py" mechanism --archive "$archive" --output "$results" > "$validation/mechanism_analysis.txt" 2>&1
"$py" "$task_root/scripts/analysis/chapter5_stage_b_figure.py" --results "$results" --output "$results/figures" --qa-tools "$validation" > "$validation/figure_render.txt" 2>&1
"$py" "$validation/validate_figure.py" "$task_root/scripts/analysis/chapter5_stage_b_figure.py" --backend python --json > "$validation/figure_source_audit.json"
for name in stage_b_transfer stage_b_calibration; do
  "$py" "$validation/audit_pdf_text.py" "$results/figures/$name.pdf" --json > "$validation/$name.text_audit.json"
  "$py" "$validation/audit_figure_collisions.py" "$results/figures/$name.pdf" --json-out "$validation/$name.collision.json" --overlay-pdf "$validation/$name.collision.pdf"
done
echo 'POSTPROCESS_COMPLETE'
