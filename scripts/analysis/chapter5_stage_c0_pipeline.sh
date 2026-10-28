#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
archive="${1:?historical archive}"
development="${2:?stage B development}"
results="${3:?C0 frozen results}"
py="${4:?server Python}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$py")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$py" "$root/scripts/analysis/chapter5_stage_c0.py" closed --archive "$archive" --development "$development" --output "$results"
"$py" "$root/scripts/analysis/chapter5_stage_c0.py" fixed --archive "$archive" --development "$development" --output "$results"
echo C0_TIMING_COLLECTION_COMPLETE
