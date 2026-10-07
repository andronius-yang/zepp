# Zepp

This repository contains reference code for the paper: "Zepp: Accelerating Distributed MoE Serving under
Relaxed Balance Constraints"

Authors: Chang Chen, Andrew Yang, Tiancheng Chen, Jiangfei Duan, Xinwei Qiang, Zhongkai Yu, Xiang Fu, Yufei Ding

## Layout

```
python/zepp     MoE layer: placement, routing, planning, expert swap, communication
src
 |- dispatch    dispatch fused with the first expert GEMM
 |- combine     second expert GEMM fused with the combine
 |- direct      unfused all-to-all path
 |- dwire       GPU-issued dispatch wire
 |- planner     routing, swap-decision and swap-lane kernels
 |- core        runtime support
include         headers
bench           benchmark driver
data/traces     sample routing traces
examples        usage examples
tests           unit tests
scripts         benchmark sweep
env             example environment
integrations    SGLang integration
```

## Build

Requirements: CUDA 12.4, NVSHMEM 3.2, gcc 12, Python 3.11 with PyTorch 2.6 (CUDA 12.4).

```
git submodule update --init
export ZEPP_CONDA_ENV=/path/to/conda/env
source env/setup.sh                 # adapt the module loads to your system
./build.sh
```

## Run

```
srun -N <nodes> --ntasks-per-node=1 bench/launch.sh bench/replay.py --model qwen3 --budget-mib 4 \
     --comm-strategy overlap --swap 1 --router-c 0.25
```

Runs one layer on sampled routing and prints its latency.

- `--model`: `qwen3` or `k2`
- `--budget-mib`: send budget per rank, in MiB
- `--comm-strategy`: `overlap` (communication fused into the GEMMs) or `direct`
- `--swap`: `1` enables expert swaps
- `--router-c`: routing slack C
- `--gpu-plan`: `1` (default) plans and issues the layer on the GPU; `0` plans on the host
- `--check`: `1` compares the output with PyTorch

## Using the layer

```python
import torch
import torch.distributed as dist
from zepp import EPMoE, EPMoEConfig, SHAPES

cfg = EPMoEConfig(shape=SHAPES["qwen3"], ranks=dist.get_world_size(),
                  ranks_per_node=torch.cuda.device_count(), max_tokens_per_rank=512,
                  comm_strategy="overlap", swap=False, router_c=0.25)
layer = EPMoE(cfg, dist.group.WORLD, sizing_routing=ids_sample, pool_routing=ids_pool)  # collective
layer.load_weights(w1_of=lambda e: W1[e], w2_of=lambda e: W2[e])
layer.prime(topk_ids, topk_weights)                   # once: capture graphs, warm the wire
plan = layer.prepare(topk_ids, topk_weights)          # per step: loads, (swap), route, metadata
y = layer.forward(x, plan)                            # dispatch_gemm -> activation -> gemm_combine
```

`sizing_routing` is a representative batch the buffer capacities are derived from; `pool_routing` is
the routing window the placement is planned from.

## SGLang

`integrations/sglang` runs the layer as an MoE backend of SGLang; see `integrations/sglang/README.md`.

## License

Apache-2.0. The CUTLASS-based grouped GEMM kernels and the build/binding skeleton derive from
ByteDance Flux (Apache-2.0); see `LICENSE` and `NOTICE`.
