#!/usr/bin/env bash
# Run Chapter 3/5 MARL and Chapter 4 controlled assignment experiments.
set -euo pipefail
cd "$(dirname "$0")"
OUTPUT="${1:?请指定新的全论文实验结果目录}"
shift
bash run_remote_python.sh run_formal_suite.py --suite all --output "$OUTPUT/marl" "$@"
bash run_remote_python.sh run_chapter4_suite.py --suite all --output "$OUTPUT/chapter4" "$@"
