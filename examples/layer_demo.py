"""Layer API smoke test on one node: build EPMoE on random routing, run a step, compare with a
plain PyTorch MoE.

    bench/launch.sh examples/layer_demo.py [--comm-strategy overlap|direct] [--swap 0|1]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from zepp import EPMoE, EPMoEConfig, SHAPES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm-strategy", choices=("overlap", "direct"), default="overlap")
    ap.add_argument("--swap", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=256, help="tokens per rank (multiple of top-k)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    shape = SHAPES["qwen3"]
    cfg = EPMoEConfig(shape=shape, ranks=world, ranks_per_node=world, max_tokens_per_rank=a.tokens,
                      comm_strategy=a.comm_strategy, swap=bool(a.swap), router_c=0.25)

    g = torch.Generator().manual_seed(1234)
    E, K, H, F = shape.num_experts, shape.topk, shape.hidden, shape.ffn_hidden
    # routing: every rank's tokens pick k distinct experts, skewed towards low ids
    weights = torch.linspace(4.0, 1.0, E)
    ids_all = torch.stack([torch.multinomial(weights, K, replacement=False, generator=g)
                           for _ in range(world * a.tokens)]).view(world, a.tokens, K).int()
    probs_all = torch.softmax(torch.rand(world, a.tokens, K, generator=g), dim=-1)
    ids_own, probs_own = ids_all[rank], probs_all[rank]

    w1 = {e: ((torch.rand(F, H, generator=g) - 0.5) * 0.02).bfloat16() for e in range(E)}
    w2 = {e: ((torch.rand(H, F, generator=g) - 0.5) * 0.02).bfloat16() for e in range(E)}
    x = ((torch.rand(a.tokens, H, generator=g) - 0.5) * 2).bfloat16().cuda()

    layer = EPMoE(cfg, dist.group.WORLD, sizing_routing=ids_all, pool_routing=ids_all,
                  probs_own=probs_own)
    layer.load_weights(lambda e: w1[e], lambda e: w2[e])
    layer.prime(ids_own.cuda(), probs_own.cuda())
    plan = layer.prepare(ids_own.cuda(), probs_own.cuda())
    y = layer.forward(x, plan)
    torch.cuda.synchronize()

    # reference: dense top-k MoE on this rank's tokens
    ref = torch.zeros(a.tokens, H, dtype=torch.float32, device="cuda")
    for t in range(a.tokens):
        for j in range(K):
            e = int(ids_own[t, j])
            h = torch.nn.functional.gelu(x[t].float() @ w1[e].cuda().float().t())
            ref[t] += float(probs_own[t, j]) * (h @ w2[e].cuda().float().t())
    err = (y.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    ok = err <= 2e-2 * max(scale, 1e-3)
    print(f"[demo] rank {rank}: {cfg.comm_strategy} swap={int(cfg.swap)} max|err| = {err:.4g} "
          f"(ref max {scale:.3g}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
