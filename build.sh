#!/usr/bin/env bash
# Build libzepp_cuda.so (CMake) and the zepp._C python extension (in place).
# Usage: source env/setup.sh && ./build.sh [--jobs N] [--skip-cmake]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS=16; SKIP_CMAKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --skip-cmake) SKIP_CMAKE=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
: "${NVSHMEM_HOME:?NVSHMEM_HOME is not set (source env/setup.sh)}"
: "${CUDA_HOME:?CUDA_HOME is not set (source env/setup.sh)}"

mkdir -p "$ROOT/build"
cd "$ROOT/build"
if [[ $SKIP_CMAKE -eq 0 || ! -f CMakeCache.txt ]]; then
  cmake .. -DNVSHMEM_HOME="$NVSHMEM_HOME" -DCMAKE_INSTALL_PREFIX="$ROOT/python/zepp"
fi
make -j"$JOBS"
make install >/dev/null
cd "$ROOT"
MAX_JOBS="$JOBS" python setup.py build_ext --inplace
echo "build complete: python/zepp/lib/libzepp_cuda.so + $(ls python/zepp/_C*.so)"
