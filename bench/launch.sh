#!/usr/bin/env bash
# torchrun launcher: one instance per node (under srun) or standalone on one node.
#   bench/launch.sh bench/replay.py --model qwen3 --budget-mib 4 --comm-strategy overlap --swap 0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

# the layer's launch environment
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export NCCL_IB_TIMEOUT=23

# knobs that must be known before CUDA / NVSHMEM initialise
strategy=overlap; model=""; budget=""; swap=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --comm-strategy) strategy="${args[i+1]}" ;;
    --model) model="${args[i+1]}" ;;
    --budget-mib) budget="${args[i+1]}" ;;
    --swap) swap="${args[i+1]}" ;;
  esac
done
if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]]; then
  if [[ "$strategy" == "direct" ]]; then export CUDA_DEVICE_MAX_CONNECTIONS=8; else export CUDA_DEVICE_MAX_CONNECTIONS=24; fi
fi
if [[ -z "${NVSHMEM_SYMMETRIC_SIZE:-}" && -n "$model" && -n "$budget" ]]; then
  export NVSHMEM_SYMMETRIC_SIZE="$(python -m zepp.heap --model "$model" --budget-mib "$budget" --swap "$swap")"
fi

nproc_per_node=$(nvidia-smi -L | wc -l)
nnodes=${SLURM_NNODES:-1}
node_rank=${SLURM_NODEID:-0}
port=${MASTER_PORT:-23456}
if [[ "$nnodes" -gt 1 ]]; then
  master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  export NVSHMEM_REMOTE_TRANSPORT=libfabric
  export NVSHMEM_LIBFABRIC_PROVIDER=cxi
else
  master=127.0.0.1
fi
exec torchrun --node_rank="$node_rank" --nproc_per_node="$nproc_per_node" --nnodes="$nnodes" \
  --rdzv_endpoint="$master:$port" "$@"
