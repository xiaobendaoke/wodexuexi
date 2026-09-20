#!/usr/bin/env bash
# Start every experiment with the selected environment's C++ runtime.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
  echo '实验与 PyTorch 验证必须在远程 Linux 服务器执行。' >&2
  exit 1
fi
PYTHON="${EXPERIMENT_PYTHON:-/home/PengYanghan/miniconda3/envs/drone/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python 不可执行：$PYTHON" >&2
  exit 1
fi
RUNTIME_PREFIX="$(cd "$(dirname "$PYTHON")/.." && pwd)"
export LD_LIBRARY_PATH="$RUNTIME_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "$PYTHON" "$@"
