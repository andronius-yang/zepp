# Launch environment of the zepp SGLang backend. Source it on every node right before
# `python -m sglang.launch_server` (after env/setup_sglang.sh). Needs ZEPP_CONFIG.
: "${ZEPP_CONFIG:?set ZEPP_CONFIG to the calibrate.py config JSON}"
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export NCCL_IB_TIMEOUT=23
_strategy=$(python -c "import json;print(json.load(open('$ZEPP_CONFIG')).get('comm_strategy','overlap'))")
if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]]; then
  if [[ "$_strategy" == "direct" ]]; then export CUDA_DEVICE_MAX_CONNECTIONS=8; else export CUDA_DEVICE_MAX_CONNECTIONS=24; fi
fi
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-$(python -m zepp.heap --config "$ZEPP_CONFIG")}"
if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
  export NVSHMEM_REMOTE_TRANSPORT=libfabric
  export NVSHMEM_LIBFABRIC_PROVIDER=cxi
fi
unset _strategy
