#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Linux ]]; then
  echo "请在远程 Linux 服务器运行。" >&2
  exit 1
fi
exec bash run_remote_python.sh run_formal_suite.py --suite all --output "${1:?请指定新的结果目录，例如 /home/PengYanghan/Lunwen/results/canonical_all}" "${@:2}"
