# Example environment for the SGLang integration (integrations/sglang); adapt the module loads to your system.
# Source it (do not execute): `source env/setup_sglang.sh`.
#
# Pins (2026-09): SGLang v0.5.3 -> torch 2.8.0 (cu129 wheels), so the extension is built with the
# CUDA 12.9 toolchain; NVSHMEM 3.2.5 (CUDA 12.4 build, no libcudart dependency) over libfabric/CXI;
# gcc 12. nccl/2.24.3 is loaded BEFORE nvshmem: the nvshmem module's unversioned nccl dependency
# would otherwise pull the CUDA 13 build. libfabric/1.20.1 first (the node image's libfabric 2.3.1
# crashes NVSHMEM 3.2.5 at init).
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1
module unload nccl nvshmem cudatoolkit 2>/dev/null
module load libfabric/1.20.1 gcc-native/12.3 cudatoolkit/12.9 nccl/2.24.3 nvshmem/3.2.5-1 \
            conda/Miniforge3-25.11.0-1 >/dev/null 2>&1

# Python environment: the SGLang env (torch 2.8.0 + sglang 0.5.3 + sgl-kernel, see integrations/sglang/README.md).
: "${ZEPP_CONDA_ENV:?set ZEPP_CONDA_ENV to the path of the SGLang conda env (integrations/sglang/README.md, Install)}"
conda activate "$ZEPP_CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Every cache off the (small) home directory: ZEPP_CACHE_DIR, default $SCRATCH/zepp_cache.
_zepp_c="${ZEPP_CACHE_DIR:-${SCRATCH:-$HOME/.cache}/zepp_cache}"
export PIP_CACHE_DIR="$_zepp_c/pip" XDG_CACHE_HOME="$_zepp_c/xdg" HF_HOME="$_zepp_c/hf"
export TRITON_CACHE_DIR="$_zepp_c/triton" TORCH_EXTENSIONS_DIR="$_zepp_c/torch_ext"
export FLASHINFER_WORKSPACE_BASE="$_zepp_c/flashinfer"
unset _zepp_c

# Toolchain: CUDA 12.9 from the HPC SDK 25.5; drop any CUDA 13 entries the modules add.
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9
_zepp_filter() { echo "${1:-}" | tr ':' '\n' | grep -vE '/cuda/13|/math_libs/13|/usr/local/cuda-1[3-9]|nccl/2\.29' | grep -v '^$' | paste -sd: ; }
export PATH="$CUDA_HOME/bin:$(_zepp_filter "$PATH")"
export CPATH="$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$(_zepp_filter "${CPATH:-}")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_MATH_LIBS/lib64:$(_zepp_filter "$LD_LIBRARY_PATH")"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_MATH_LIBS/lib64:$(_zepp_filter "${LIBRARY_PATH:-}")"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=gcc CXX=g++ CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"
unset -f _zepp_filter
: "${NVSHMEM_HOME:?nvshmem module did not export NVSHMEM_HOME}"

export ZEPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ZEPP_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
