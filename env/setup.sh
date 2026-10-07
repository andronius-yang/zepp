# Example environment for building and running zepp (NVSHMEM over libfabric); adapt the module loads to your system.
# Source it (do not execute): `source env/setup.sh`.
#
# Pins (2026-09): CUDA 12.4 toolchain, NVSHMEM 3.2.5 over libfabric/CXI, gcc 12.
# nccl/2.24.3 provides the libfabric NCCL network plugin torch's NCCL loads at run time.
# libfabric/1.20.1 must be loaded first: the libfabric 2.3.1 shipped with the
# node image crashes NVSHMEM 3.2.5 during initialization.
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1
module unload nccl nvshmem cudatoolkit 2>/dev/null
module load libfabric/1.20.1 gcc-native/12.3 cudatoolkit/12.4 nvshmem/3.2.5-1 nccl/2.24.3 \
            conda/Miniforge3-25.11.0-1 >/dev/null 2>&1

# Python environment: a conda env with PyTorch 2.x built for CUDA 12 (see README, "Build").
: "${ZEPP_CONDA_ENV:?set ZEPP_CONDA_ENV to the path of a conda env with PyTorch 2.x (CUDA 12)}"
conda activate "$ZEPP_CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Toolchain: pin CUDA 12.4 explicitly. The cudatoolkit module has drifted to export a
# newer CUDA (13.x) on CPATH / PATH / LD_LIBRARY_PATH; those entries are removed so
# nvcc, the headers and the runtime all come from the same 12.4 install.
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4   # cusparse/cublas headers torch needs
_zepp_filter() { echo "${1:-}" | tr ':' '\n' | grep -vE 'hpc_sdk/Linux_x86_64/2[5-9]\.|/usr/local/cuda-1[3-9]' | grep -v '^$' | paste -sd: ; }
export PATH="$CUDA_HOME/bin:$(_zepp_filter "$PATH")"
export CPATH="$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$(_zepp_filter "$CPATH")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_MATH_LIBS/lib64:$(_zepp_filter "$LD_LIBRARY_PATH")"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_MATH_LIBS/lib64:$(_zepp_filter "${LIBRARY_PATH:-}")"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=gcc CXX=g++ CUDAHOSTCXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"
unset -f _zepp_filter
# NVSHMEM_HOME is exported by the nvshmem module.
: "${NVSHMEM_HOME:?nvshmem module did not export NVSHMEM_HOME}"

export ZEPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ZEPP_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
