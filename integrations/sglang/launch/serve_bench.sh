#!/usr/bin/env bash
# serve_bench.sh <url> <out_prefix> [num_prompts] [concurrency list]
# Serving benchmark against a running server: sharegpt prompts, fixed seed, one run per concurrency
# level; JSONL results in <out_prefix>_c<conc>.jsonl.
set -euo pipefail
URL=$1; OUT=$2; NP=${3:-256}; CONCS=${4:-"16 64"}
: "${HF_HOME:?}"   # dataset cache off the home directory
DS=${SHAREGPT_PATH:-$HF_HOME/ShareGPT_V3_unfiltered_cleaned_split.json}
# BENCH_MODE=prefill: long prompts, short outputs (random dataset: input lengths in [ISL/2, ISL], ISL default
# 2048; output lengths in [OSL/2, OSL], OSL default 32)
if [[ "${BENCH_MODE:-sharegpt}" == "prefill" ]]; then
  DSARGS=(--dataset-name random --random-input-len "${ISL:-2048}" --random-output-len "${OSL:-32}" --random-range-ratio 0.5)
else
  DSARGS=(--dataset-name sharegpt --dataset-path "$DS")
fi
for c in $CONCS; do
  python -m sglang.bench_serving --backend sglang --base-url "$URL" "${DSARGS[@]}" \
    --num-prompts "$NP" --max-concurrency "$c" --request-rate inf --seed 1 --warmup-requests 8 \
    --output-file "${OUT}_c${c}.jsonl" 2>&1 | grep -E "Successful|Benchmark duration|Request throughput|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Median E2E|Concurrency" | sed "s/^/[c=$c] /"
done
