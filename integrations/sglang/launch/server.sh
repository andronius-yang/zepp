#!/usr/bin/env bash
# server.sh <baseline|ours> <model_dir> [extra sglang args...]
# One invocation per node (under `srun --nodes=N --ntasks-per-node=1`), or
# standalone on one node. tp = dp = ep = GPUs per node * nodes (DP attention, attention TP = 1).
#   baseline  stock SGLang (a2a backend "none", CUDA graphs disabled): with CALIBRATE=1 it records the
#             routing calibrate.py reads
#   ours      --moe-a2a-backend zepp (needs ZEPP_CALIB_DIR)
# Environment: SMAX (tokens per rank per step, default 2048), MAXRR (max running requests per
# rank, default 128), CTX_LEN (--context-length, default the model's), PORT (30000), MEMFRAC
# (--mem-fraction-static), CALIBRATE=1 (baseline records per-token expert routing into
# $SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR).
# SGLang divides --max-running-requests by the DP size, so it is passed as MAXRR * TP. Every rank
# still sizes its request-to-token table for the global count, (MAXRR * TP) x (CTX_LEN + 4) int32
# entries, so bound CTX_LEN when MAXRR is large.
set -euo pipefail
MODE=$1; MODEL=$2; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NN=${SLURM_NNODES:-1}; TP=$(($(nvidia-smi -L | wc -l) * NN)); SMAX=${SMAX:-2048}; MAXRR=${MAXRR:-128}; PORT=${PORT:-30000}
args=(--model-path "$MODEL" --trust-remote-code --host 0.0.0.0 --port "$PORT"
      --tp-size "$TP" --dp-size "$TP" --enable-dp-attention --ep-size "$TP"
      --attention-backend flashinfer --chunked-prefill-size $((SMAX * TP)) --max-running-requests $((MAXRR * TP))
      --disable-overlap-schedule)
[[ -n "${MEMFRAC:-}" ]] && args+=(--mem-fraction-static "$MEMFRAC")
[[ -n "${CTX_LEN:-}" ]] && args+=(--context-length "$CTX_LEN")
if [[ "$NN" -gt 1 ]]; then
  head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  args+=(--nnodes "$NN" --node-rank "${SLURM_NODEID:-0}" --dist-init-addr "$head:5000")
fi
case "$MODE" in
  baseline)       args+=(--moe-a2a-backend none --disable-cuda-graph) ;;
  ours)
    : "${ZEPP_CALIB_DIR:?set ZEPP_CALIB_DIR (calibrate.py --out)}"
    export ZEPP_CONFIG="${ZEPP_CONFIG:-$ZEPP_CALIB_DIR/zepp_config.json}"
    source "$HERE/env_launch.sh"
    red=$(python -c "import json;c=json.load(open('$ZEPP_CONFIG'));print(c['ranks']*c['redundant_slots_per_rank'])")
    args+=(--moe-a2a-backend zepp --disable-cuda-graph --ep-num-redundant-experts "$red"
           --init-expert-location "$ZEPP_CALIB_DIR/expert_location.json") ;;
  *) echo "mode must be baseline | ours"; exit 2 ;;
esac
if [[ "${CALIBRATE:-0}" == "1" ]]; then
  export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR:?}"
  args+=(--expert-distribution-recorder-mode per_token)
fi
echo "[server.sh] $MODE nodes=$NN tp=$TP conn=${CUDA_DEVICE_MAX_CONNECTIONS:-unset} heap=${NVSHMEM_SYMMETRIC_SIZE:-unset}: python -m sglang.launch_server ${args[*]} $*"
exec python -m sglang.launch_server "${args[@]}" "$@"
