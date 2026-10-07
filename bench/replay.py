"""Replay benchmark: one MoE layer (dispatch + expert GEMMs + combine) on real LiveCodeBench
routing, timed per iteration with the layer's own bracket marks.

Launch one process per GPU with bench/launch.sh (torchrun + Slurm), e.g. on 4 nodes:
  srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 bench/launch.sh bench/replay.py \\
       --model qwen3 --budget-mib 4 --comm-strategy overlap --swap 0 --router-c 0.25 --out runs/x.csv
"""
import argparse
import csv
import os
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traces  # noqa: E402

from zepp import SHAPES, EPMoE, EPMoEConfig  # noqa: E402
from zepp._ext import C  # noqa: E402
from zepp.config import tokens_per_rank  # noqa: E402
from zepp.layer import activation  # noqa: E402

METRICS = ("plan_comm_ms", "place_ms", "plan_ms", "dispatch_ms", "act_ms", "combine_ms", "e2e_ms", "total_ms")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(SHAPES), required=True)
    ap.add_argument("--budget-mib", type=float, required=True, help="pre-top-k send budget per rank (MiB)")
    ap.add_argument("--comm-strategy", choices=("overlap", "direct"), default="overlap")
    ap.add_argument("--act", choices=("gelu", "swiglu"), default="gelu",
                    help="expert activation (swiglu: gate and up projections, as in the models)")
    ap.add_argument("--swap", type=int, default=0, help="band-triggered expert swap on/off")
    ap.add_argument("--router-c", type=float, default=0.25, help="routing slack C")
    ap.add_argument("--gpu-plan", type=int, default=1,
                    help="1: planning and communication issued on the GPU (overlap on >= 2 nodes); 0: planned on the host")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--isolated", type=int, default=1, help="device sync + barrier before every timed iteration")
    ap.add_argument("--check", type=int, default=0, help="every timed iteration on a fresh random payload vs a torch reference")
    ap.add_argument("--out", type=str, default="", help="CSV of per-rank per-iteration metrics (rank 0 writes)")
    return ap.parse_args()


_W = {}


def expert_w1(e, ffn, H, dtype):
    k = ("w1", e)
    if k not in _W:
        g = torch.Generator().manual_seed(10007 + e)
        _W[k] = ((torch.rand((ffn, H), generator=g) * 0.02) - 0.01).to(dtype)
    return _W[k]


def expert_w2(e, ffn, H, dtype):
    k = ("w2", e)
    if k not in _W:
        g = torch.Generator().manual_seed(20011 + e)
        _W[k] = ((torch.rand((H, ffn), generator=g) * 0.02) - 0.01).to(dtype)
    return _W[k]


@torch.no_grad()
def torch_reference(x, topk_own, probs_own, shape, dtype):
    H = shape.hidden
    S, K = topk_own.shape
    y = torch.zeros(S, H, dtype=torch.float32, device="cuda")
    e_flat = topk_own.reshape(-1).long()
    t_flat = torch.arange(S, device=e_flat.device).repeat_interleave(K)
    p_flat = probs_own.reshape(-1).float().cuda()
    for e in torch.unique(e_flat).tolist():
        sel = (e_flat == e).nonzero(as_tuple=True)[0]
        rows = t_flat[sel]
        w1 = expert_w1(e, shape.ffn1, H, dtype).cuda()
        w2 = expert_w2(e, shape.ffn_hidden, H, dtype).cuda()
        part = activation(x[rows] @ w1.t(), shape) @ w2.t()
        y.index_add_(0, rows, part.float() * p_flat[sel].unsqueeze(1))
    return y.to(dtype)


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    L = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    group = dist.group.WORLD
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    for n in range(world // L):
        assert len(set(hosts[n * L:(n + 1) * L])) == 1, f"ranks are not node-major: {hosts}"

    import dataclasses
    shape = dataclasses.replace(SHAPES[args.model], act=args.act)
    dtype = torch.bfloat16
    T = tokens_per_rank(args.budget_mib, shape)
    ts = traces.load_slice(args.model)
    rows = traces.sample_batch(ts, world, L, args.budget_mib, shape.chunk_bytes)
    topk_all = torch.tensor(rows, dtype=torch.int32).view(world, T, shape.topk)
    pool = ts.pool_rows[: (len(ts.pool_rows) // world) * world]
    pool_topk = torch.tensor(pool, dtype=torch.int32).view(world, -1, shape.topk)
    gen_p = torch.Generator().manual_seed(777)
    probs_all = torch.rand((world * T, shape.topk), generator=gen_p) + 0.5
    probs_own = probs_all[rank * T:(rank + 1) * T]
    topk_own = topk_all[rank].cuda()
    probs_own_d = probs_own.cuda()
    gen_x = torch.Generator(device="cuda").manual_seed(4242 + rank)
    x = ((torch.rand((T, shape.hidden), device="cuda", generator=gen_x) * 0.02) - 0.01).to(dtype)

    cfg = EPMoEConfig(shape=shape, ranks=world, ranks_per_node=L, max_tokens_per_rank=T,
                      comm_strategy=args.comm_strategy, swap=bool(args.swap), router_c=args.router_c,
                      gpu_plan=bool(args.gpu_plan))
    t0 = time.perf_counter()
    layer = EPMoE(cfg, group, sizing_routing=topk_all, pool_routing=pool_topk, probs_own=probs_own, dtype=dtype)
    layer.load_weights(lambda e: expert_w1(e, shape.ffn1, shape.hidden, dtype),
                       lambda e: expert_w2(e, shape.ffn_hidden, shape.hidden, dtype))
    layer.prime(topk_own, probs_own_d)
    if rank == 0:
        c = layer.caps
        print(f"[setup] {args.model} ranks={world} tokens/rank={T} strategy={args.comm_strategy} swap={args.swap} "
              f"C={args.router_c} | recv_cap {c.recv_cap} dispatch recv/stage/relay {c.dispatch_recv}/"
              f"{c.dispatch_stage}/{c.dispatch_relay} combine send/conv/wire {c.combine_send}/{c.combine_conv}/"
              f"{c.combine_wire} pair {c.pair_cap} | placement {layer.placement.stats} | act={args.act} | "
              f"gpu_plan={int(layer.gpu_plan)} | "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

    total = args.warmup + args.iters
    names = ("iter_start", "plan_comm", "place", "plan", "dispatch", "act", "combine")
    ev = {n: [torch.cuda.Event(enable_timing=True) for _ in range(total)] for n in names}
    bad_total, checked = 0, 0
    for i in range(total):
        layer.prep()
        if args.check:
            # a new payload every iteration: a stale-delivery wire bug cannot pass on static data
            gen_x.manual_seed(4242 + rank + 1000 * (i + 1))
            x = ((torch.rand((T, shape.hidden), device="cuda", generator=gen_x) * 0.02) - 0.01).to(dtype)
        if args.isolated:
            torch.cuda.synchronize()
            dist.barrier()
        marks = lambda name, i=i: ev[name][i].record()  # noqa: E731
        ev["iter_start"][i].record()
        plan = layer.prepare(topk_own, probs_own_d, marks)
        y = layer.forward(x, plan, marks)
        if args.check and i >= args.warmup:
            torch.cuda.synchronize()
            ref = torch_reference(x, topk_own, probs_own_d, shape, dtype)
            bad_total += int((~torch.isclose(y.float(), ref.float(), atol=1e-2, rtol=1.5e-2)).any(dim=1).sum())
            checked += 1
    torch.cuda.synchronize()
    times = {m: [] for m in METRICS}
    for i in range(args.warmup, total):
        e = {n: ev[n][i] for n in names}
        times["plan_comm_ms"].append(e["iter_start"].elapsed_time(e["plan_comm"]))
        times["place_ms"].append(e["plan_comm"].elapsed_time(e["place"]))
        times["plan_ms"].append(e["place"].elapsed_time(e["plan"]))
        times["dispatch_ms"].append(e["plan"].elapsed_time(e["dispatch"]))
        times["act_ms"].append(e["dispatch"].elapsed_time(e["act"]))
        times["combine_ms"].append(e["act"].elapsed_time(e["combine"]))
        times["e2e_ms"].append(e["plan"].elapsed_time(e["combine"]))
        times["total_ms"].append(e["iter_start"].elapsed_time(e["combine"]))
    gathered = [None] * world
    dist.all_gather_object(gathered, times)
    if rank == 0:
        per_iter_max = [max(gathered[r]["total_ms"][i] for r in range(world)) for i in range(args.iters)]
        summary = statistics.median(per_iter_max)
        means = {m: statistics.mean(max(gathered[r][m][i] for r in range(world)) for i in range(args.iters)) for m in METRICS}
        print(f"[result] {args.model} nodes={world // L} b{args.budget_mib:g} {args.comm_strategy} swap={args.swap}: "
              f"total_ms iter_max_median = {summary:.3f} | rank-max means " +
              " ".join(f"{m[:-3]} {v:.3f}" for m, v in means.items()), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            new = not os.path.exists(args.out)
            with open(args.out, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["model", "nodes", "ranks", "budget_mib", "comm_strategy", "swap", "router_c",
                                "rank", "iter", "metric", "value_ms"])
                for r in range(world):
                    for m in METRICS:
                        for i, v in enumerate(gathered[r][m]):
                            w.writerow([args.model, world // L, world, args.budget_mib, args.comm_strategy,
                                        args.swap, args.router_c, r, i, m, f"{v:.4f}"])
                w.writerow([args.model, world // L, world, args.budget_mib, args.comm_strategy, args.swap,
                            args.router_c, -1, -1, "total_ms_iter_max_median", f"{summary:.4f}"])

    if args.check:
        ok = bad_total == 0
        flag = torch.tensor([0 if ok else 1], device="cuda")
        dist.all_reduce(flag)
        for r in range(world):
            if r == rank:
                print(f"[check] rank {rank}: {'PASS' if ok else 'FAIL'} bad rows {bad_total}/{checked * T} over {checked} iterations "
                      f"(fresh payload each)", flush=True)
            dist.barrier()
        assert int(flag) == 0, "correctness check failed"
    dist.barrier()
    torch.cuda.synchronize()
    if getattr(layer, "gpu_plan", False) and layer.shared.layer_graph:
        # a communicator whose collectives were captured in layer graphs hangs in its finalize at teardown: drop the
        # graphs and leave without the process-group teardown
        layer.shared._graphs.clear()
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
