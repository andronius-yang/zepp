"""Layer-level test of the serving path against a PyTorch reference (4 ranks per node, one or
more nodes): several layers on one SharedComm, a different token count per rank and per step
(buckets, padding, zero-token ranks), SwiGLU, per-step gate weights, capacity growth mid-run
(--caps-scale < 1), swap on/off with a load shift (--skew-flip). The serving path runs the overlap
strategy on two or more nodes.

  bench/launch.sh examples/serving_check.py --steps 60 --layers 3
  bench/launch.sh examples/serving_check.py --caps-scale 0.1            # forces growth
  bench/launch.sh examples/serving_check.py --swap 1 --skew-flip 1      # swaps happen

With the overlap strategy every step's layers run as one forward (SharedComm.begin_forward /
end_forward / forward_check): the buffer demands are checked once per forward on the device, and a
forward that exceeded them (or the forced abort, --force-abort <layer>) is recovered (growth,
re-prime) and run again. --out-hash 1 prints a digest of every output, to compare two runs bit for bit
(e.g. --caps-scale 0.5 with growth and redo against a run that never grows).
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from zepp import EPMoEConfig, LayerState, ModelShape, SharedComm
from zepp.placement import demand_histogram, solve_placement
from zepp.routing import Capacities, compute_capacities
from zepp.swap import swap_orbit

SHAPES = {"small": ModelShape(32, 8, 1024, 512, act="swiglu"), "tiny": ModelShape(16, 8, 1024, 512, act="swiglu"),
          "qwen3-30b": ModelShape(128, 8, 2048, 768, act="swiglu"),
          "qwen3-235b": ModelShape(128, 8, 4096, 1536, act="swiglu")}


def expert_probs(G, seed, hot_frac=0.25, hot_mass=0.7):
    """Skewed expert popularity: a random quarter of the experts carries most of the mass."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(G, generator=g)
    p = torch.full((G,), (1 - hot_mass) / (G - int(G * hot_frac)))
    p[perm[:int(G * hot_frac)]] = hot_mass / int(G * hot_frac)
    return p / p.sum()


def sample_routing(p, n, K, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.multinomial(p.expand(n, -1), K, replacement=False, generator=g).int()


class RecordedRouting:
    """Real routing for the harness (--routing-dumps): this rank's rows of an SGLang expert-distribution recording
    (per_token mode, one file per rank, suffix _<rank>.pt; records in forward-pass order with forward_mode and
    topk_ids_of_layer [model layers, T, K]). mode "decode" keeps the decode forwards (forward_mode 2), "all" every
    forward with tokens. Each harness step takes the next n rows of the rank's stream (consecutive recorded steps,
    wrapping), so the per-step popularity and its drift are the recording's."""

    def __init__(self, dumps, rank, model_layers, mode="decode"):
        import glob
        files = [f for f in glob.glob(os.path.join(dumps, "*.pt")) if f.endswith(f"_{rank}.pt")]
        assert len(files) == 1, f"rank {rank}: {len(files)} dump files under {dumps}"
        data = torch.load(files[0], map_location="cpu", weights_only=False)
        recs = sorted(data["records"], key=lambda r: r["forward_pass_id"])
        keep = [r for r in recs if r["topk_ids_of_layer"].shape[1] > 0 and (mode == "all" or r["forward_mode"] == 2)]
        assert keep, f"rank {rank}: no {mode} records with tokens in {files[0]}"
        self.rows = [torch.cat([r["topk_ids_of_layer"][ml] for r in keep], 0).int() for ml in model_layers]
        self.n_rows = self.rows[0].shape[0]
        self.n_steps = len(keep)
        self.cursor = 0

    def take(self, n):
        """[layer] -> int32 [n, K] (the same token rows for every layer: one model forward)"""
        idx = (torch.arange(n) + self.cursor) % self.n_rows
        self.cursor = (self.cursor + n) % self.n_rows
        return [r[idx] for r in self.rows]


def make_weights(shape, layer, device):
    g = torch.Generator(device="cpu").manual_seed(10007 + 1000 * layer)
    G, H, f1, f = shape.num_experts, shape.hidden, shape.ffn1, shape.ffn_hidden
    w1 = ((torch.rand(G, f1, H, generator=g) * 2 - 1) * 0.05).to(device)
    w2 = ((torch.rand(G, H, f, generator=g) * 2 - 1) * 0.05).to(device)
    return w1, w2


def reference(x, ids, w, W1, W2, shape):
    """fp32 SwiGLU MoE on this rank's tokens (all experts' weights are known to every rank)."""
    n, K = ids.shape
    f = shape.ffn_hidden
    xf = x.float()
    y = torch.zeros(n, shape.hidden, dtype=torch.float32, device=x.device)
    for e in range(shape.num_experts):
        rows, ks = (ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        h = xf[rows] @ W1[e].float().t()
        a = torch.nn.functional.silu(h[:, :f]) * h[:, f:]
        y.index_add_(0, rows, (a @ W2[e].float().t()) * w[rows, ks].float().unsqueeze(1))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--swap", type=int, default=0)
    ap.add_argument("--caps-scale", type=float, default=1.0)
    ap.add_argument("--smax", type=int, default=256)
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--router-c", type=float, default=0.25)
    ap.add_argument("--skew-flip", type=int, default=0, help="shift the expert popularity every --flip-every steps")
    ap.add_argument("--flip-every", type=int, default=20, help="period of the popularity shift (steps)")
    ap.add_argument("--token-mode", default="cycle", choices=["cycle", "full"],
                    help="cycle: uniform / random / zero-token rank / full batches in turn; full: every rank at --smax")
    ap.add_argument("--ref", type=int, default=1, help="check every output against the torch reference (0: timing runs)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expect-heap-error", type=int, default=0)
    ap.add_argument("--starve-rank", type=int, default=-1,
                    help="no token of any rank picks this rank's home experts (zero-row rank; with --swap 1 "
                         "exercises the gated slot commit on a rank whose GEMMs compute nothing)")
    ap.add_argument("--swap-cost-max", type=float, default=0.0,
                    help="layer-steps with moves must take <= this x the median without moves (0 = report only; "
                         "the bracket holds the whole layer-step, so this is not the lane's cost alone)")
    ap.add_argument("--shared-weights", type=int, default=0,
                    help="every layer uses layer 0's weights (many layers in memory; the reference stays exact)")
    ap.add_argument("--graphs", type=int, default=0,
                    help="capture one CUDA graph per (layer, bucket) after the warm-up and replay them "
                         "(SharedComm layer_graphs; overlap strategy). 0: every step eager")
    ap.add_argument("--out-hash", type=int, default=0,
                    help="print a per-rank digest of every layer output (bitwise comparison of two runs)")
    ap.add_argument("--routing-dumps", default="",
                    help="real routing: an SGLang per_token expert-distribution dump directory recorded at this rank "
                         "count (RecordedRouting); replaces the synthetic popularity")
    ap.add_argument("--routing-mode", default="decode", choices=["decode", "all"])
    ap.add_argument("--calib", default="",
                    help="zepp_config.json of a serving calibration (or its directory): placements and capacities "
                         "of the model layers below (capacities x --caps-scale)")
    ap.add_argument("--model-layers", default="",
                    help="comma list of model layers behind the harness layers (default: evenly spaced over the "
                         "calibration's / 48 layers)")
    ap.add_argument("--force-abort", type=int, default=-1,
                    help="deferred verdict: abort the first forward once at this MoE layer ordinal (exercises the "
                         "recovery and the redo of the forward; -1 = off)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % 4))
    torch.cuda.set_device(local)
    dev = torch.device("cuda")
    group = dist.group.WORLD
    shape = SHAPES[a.shape]
    L = min(W, 4)
    cfg = EPMoEConfig(shape=shape, ranks=W, ranks_per_node=L, max_tokens_per_rank=a.smax,
                      comm_strategy="overlap", swap=bool(a.swap), router_c=a.router_c)
    G, K, nlp = shape.num_experts, shape.topk, cfg.slots_per_rank
    t0 = time.time()

    recorded = None
    calib = None
    if a.calib:
        import json
        cpath = os.path.join(a.calib, "zepp_config.json") if os.path.isdir(a.calib) else a.calib
        calib = json.load(open(cpath))
        assert calib["ranks"] == W and calib["ranks_per_node"] == L, "calibration recorded for another rank count"
    n_model = calib["num_layers"] if calib is not None else 48
    model_layers = ([int(v) for v in a.model_layers.split(",")] if a.model_layers else
                    [(l * n_model) // a.layers for l in range(a.layers)])
    assert len(model_layers) == a.layers, "--model-layers needs one entry per harness layer"
    if a.routing_dumps:
        recorded = RecordedRouting(a.routing_dumps, rank, model_layers, a.routing_mode)
        if rank == 0:
            print(f"[check] recorded routing {a.routing_dumps} ({a.routing_mode}): {recorded.n_steps} forwards, "
                  f"{recorded.n_rows} rows on rank 0, model layers {model_layers}", flush=True)
    # per-layer expert popularity (identical on every rank), placement from a pool window, capacities
    probs = [expert_probs(G, 100 + l) for l in range(a.layers)]
    probs_alt = [expert_probs(G, 500 + l) for l in range(a.layers)]
    if a.starve_rank >= 0:                      # one node: hosted == home shard (no replication room)
        home = G // W
        for pl in probs + probs_alt:
            pl[a.starve_rank * home:(a.starve_rank + 1) * home] = 0
            pl /= pl.sum()
    placements, caps_layers = [], []
    for l in range(a.layers):
        pool = torch.stack([sample_routing(probs[l], 2048, K, 1000 + 31 * l + r) for r in range(W)])
        hist = demand_histogram(pool.to(dev), L, G).cpu()
        pl = solve_placement(hist, L, nlp)
        placements.append(pl)
        sizing = torch.stack([sample_routing(probs[l], a.smax, K, 2000 + 31 * l + r) for r in range(W)])
        pls = [pl]
        if cfg.swap:
            load_g = torch.bincount(sizing.reshape(-1).long(), minlength=G)
            pls += swap_orbit(load_g, pl, L, nlp, cfg.router_c)
        caps_layers.append(compute_capacities(sizing, pls, nlp, L, cfg.router_c, group))
    f = {k: max(int(a.caps_scale * max(getattr(c, k) for c in caps_layers)), 1)
         for k in ("recv_cap", "dispatch_recv", "dispatch_stage", "dispatch_relay", "combine_send", "combine_conv",
                   "combine_wire", "pair_cap")}
    if calib is not None:
        # the serving calibration: its placements (as the serving runtime builds them) and its capacities
        from zepp.placement import Placement, rebuild_l2p
        placements = []
        for ml in model_layers:
            p2l = torch.tensor(calib["p2l"][str(ml)], dtype=torch.int32)
            placements.append(Placement(p2l, rebuild_l2p(p2l, G, W), torch.bincount(p2l[p2l >= 0].long(), minlength=G).int(), {}))
        f = {k: max(int(a.caps_scale * calib["caps"][k]), 1) for k in f}
    caps0 = Capacities(**f)
    if rank == 0:
        print(f"[check] W={W} shape={a.shape} S_max={a.smax} swap={a.swap} caps0={caps0}", flush=True)

    try:
        shared = SharedComm(cfg, group, caps0, dtype=torch.bfloat16, force_abort=a.force_abort,
                            layer_graphs=bool(a.graphs))
    except RuntimeError as e:
        if a.expect_heap_error:
            print(f"[rank {rank}] EXPECTED HEAP ERROR: {str(e)[:160]}", flush=True)
            dist.barrier()
            return
        raise
    weights, layers = [], []
    for l in range(a.layers):
        W1, W2 = weights[0] if (a.shared_weights and l > 0) else make_weights(shape, l, dev)
        weights.append((W1, W2))
        st = LayerState(l, cfg, placements[l], rank, dtype=torch.bfloat16)
        p2l = placements[l].p2l
        for j in range(nlp):
            e = int(p2l[rank * nlp + j])
            if e >= 0:
                st.slots.w1[1 + j].copy_(W1[e].to(torch.bfloat16))
                st.slots.w2[1 + j].copy_(W2[e].to(torch.bfloat16))
        layers.append(st)
    torch.cuda.synchronize()
    shared.prime(layers[0])
    if rank == 0:
        print(f"[check] setup {time.time() - t0:.1f}s, buckets {shared.buckets}, growths at prime {shared.growths}", flush=True)
    # the serving warm-up (every kernel loaded, every bucket run once on synthetic tokens); the checks
    # below also prove it leaves the model state unchanged (weights, placement, swap epoch)
    epoch0 = shared.lane.epoch if shared.lane is not None else None
    p2l0 = [st.p2l.clone() for st in layers]
    wu = shared.warmup(layers[0])
    torch.cuda.synchronize()
    assert all(torch.equal(st.p2l, p) for st, p in zip(layers, p2l0)), "warm-up changed a placement"
    assert shared.lane is None or shared.lane.epoch == epoch0, "warm-up changed the swap epoch"
    assert shared.steps == 0 and shared.swap_moves == 0, (shared.steps, shared.swap_moves)
    for l, st in enumerate(layers):
        p2l = placements[l].p2l
        for j in range(nlp):
            e = int(p2l[rank * nlp + j])
            if e >= 0:
                assert torch.equal(st.slots.w1[1 + j], weights[l][0][e].to(torch.bfloat16)), \
                    f"rank {rank}: warm-up changed W1 slot {j} of layer {l}"
                assert torch.equal(st.slots.w2[1 + j], weights[l][1][e].to(torch.bfloat16)), \
                    f"rank {rank}: warm-up changed W2 slot {j} of layer {l}"
    if rank == 0:
        print(f"[check] warm-up {wu}", flush=True)
    if a.graphs:
        ng = shared.capture_graphs(layers)
        if rank == 0:
            print(f"[check] layer graphs captured: {ng}", flush=True)

    bad_rows = 0
    max_err = 0.0
    rows_checked = 0
    epochs = []
    step_evs = []                                   # (start, end, moved, step) per layer-step
    digest = hashlib.sha256()
    slot_checks = 0
    for step in range(a.steps):
        rs = np.random.default_rng(a.seed * 7919 + step)
        mode = step % 4 if a.token_mode == "cycle" else 3
        if mode == 0:
            n_list = [int(rs.integers(1, a.smax + 1))] * W
        elif mode == 1:
            n_list = [int(v) for v in rs.integers(0, a.smax + 1, size=W)]
        elif mode == 2:
            n_list = [int(v) for v in rs.integers(1, a.smax + 1, size=W)]
            n_list[int(rs.integers(0, W))] = 0
        else:
            n_list = [a.smax] * W
        n = n_list[rank]
        gx = torch.Generator(device="cpu").manual_seed(a.seed * 104729 + 1000 * step + rank)
        x = (torch.randn(n, shape.hidden, generator=gx)).to(dev).to(torch.bfloat16)
        w = torch.rand(n, K, generator=gx) + 0.5
        w = (w / w.sum(1, keepdim=True)).to(dev)
        flip = bool(a.skew_flip) and (step // a.flip_every) % 2 == 1
        ids_l = []
        rec_rows = recorded.take(n) if (recorded is not None and n) else None
        for l in range(a.layers):
            p = probs_alt[l] if flip else probs[l]
            if rec_rows is not None:
                ids_l.append(rec_rows[l].to(dev))
                continue
            ids_l.append(sample_routing(p, n, K, a.seed * 31337 + 1000 * step + 100 * l + rank).to(dev) if n else
                         torch.zeros(0, K, dtype=torch.int32, device=dev))

        def run_pass():
            """the step's layers, in order (one model forward); per layer (y, ev0, ev1, moved)"""
            outs = []
            for l in range(a.layers):
                moves_before = shared.swap_moves
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                y = shared.step(layers[l], x, ids_l[l], w, n_list)
                ev1.record()
                outs.append((y, ev0, ev1, shared.swap_moves > moves_before))
                if shared.lane is not None and not shared.deferred:
                    epochs.append(shared.lane.epoch)
            return outs

        # deferred verdict: the step's layers are one forward, checked once at its end and run again after a
        # recovery (growth, re-prime) when it aborted; otherwise begin/end/check are no-ops
        moves_fwd = shared.swap_moves
        shared.begin_forward()
        outs = run_pass()
        shared.end_forward()
        for attempt in range(9):
            abort = shared.forward_check()
            if abort is None:
                break
            assert attempt < 8, "the forward still aborts after 8 recoveries"
            if rank == 0:
                print(f"[check] step {step}: {abort}; recover and run the forward again", flush=True)
            shared.recover(abort, layers[0])
            shared.begin_forward()
            outs = run_pass()
            shared.end_forward()
        if shared.deferred and shared.lane is not None:
            epochs.append(shared.lane.epoch)
        moved_fwd = shared.swap_moves > moves_fwd
        for l, (y, ev0, ev1, moved) in enumerate(outs):
            if shared.deferred:
                moved = moved_fwd                    # the swap counter is read once per forward
            step_evs.append((ev0, ev1, moved, step))
            if moved:
                # every hosted slot of this layer now holds its expert (the commit landed, no stale slot;
                # on a zero-row rank this exercises the gated commit of a skipped GEMM)
                torch.cuda.synchronize()
                st_l = layers[l]
                p2l_now = st_l.p2l.cpu()               # the device table the GEMMs route by (authoritative)
                for j in range(nlp):
                    e = int(p2l_now[rank * nlp + j])
                    if e >= 0:
                        assert torch.equal(st_l.slots.w1[1 + j], weights[l][0][e].to(torch.bfloat16)), \
                            f"rank {rank} step {step} layer {l}: W1 slot {j} != expert {e} after swap"
                        assert torch.equal(st_l.slots.w2[1 + j], weights[l][1][e].to(torch.bfloat16)), \
                            f"rank {rank} step {step} layer {l}: W2 slot {j} != expert {e} after swap"
                        slot_checks += 1
            if a.out_hash:
                digest.update(y.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            if n and a.ref:
                ref = reference(x, ids_l[l], w, weights[l][0], weights[l][1], shape)
                yf = y.float()
                ok = torch.isclose(yf, ref, atol=1e-2, rtol=2e-2).all(dim=1)
                bad_rows += int((~ok).sum())
                rows_checked += n
                max_err = max(max_err, float((yf - ref).abs().max()))
        if rank == 0 and (step % 10 == 0 or step == a.steps - 1):
            print(f"[check] step {step} n={n_list} bucket={shared.bucket_of(max(n_list))} growths={shared.growths} "
                  f"swap_moves={shared.swap_moves} bad_rows={bad_rows} max_err={max_err:.4f}", flush=True)
    if epochs and shared.deferred:
        # per forward: the redo passes and the re-prime's warm-up steps advance it too, never back
        assert all(e1 > e0 for e0, e1 in zip(epochs, epochs[1:])), "the shared swap epoch must only grow"
    elif epochs:
        assert epochs == list(range(epochs[0], epochs[0] + len(epochs))), "shared swap epoch must advance once per layer-step"
    torch.cuda.synchronize()
    moved_ms = sorted(e0.elapsed_time(e1) for (e0, e1, mv, s) in step_evs if mv and s >= 20)
    quiet_ms = sorted(e0.elapsed_time(e1) for (e0, e1, mv, s) in step_evs if not mv and s >= 20)
    med = lambda v: v[len(v) // 2] if v else float("nan")
    lane_mode = shared.lane.mode if shared.lane is not None else "none"
    print(f"[rank {rank}] layer-step ms: no-move median {med(quiet_ms):.3f} (n={len(quiet_ms)}), "
          f"move median {med(moved_ms):.3f} (n={len(moved_ms)}), slot checks {slot_checks}, lane {lane_mode}", flush=True)
    if a.swap and a.swap_cost_max > 0 and len(moved_ms) >= 5 and len(quiet_ms) >= 5:
        assert med(moved_ms) <= a.swap_cost_max * med(quiet_ms), \
            f"rank {rank}: swap steps cost {med(moved_ms):.3f} ms vs {med(quiet_ms):.3f} without moves"
    flags = torch.tensor([bad_rows, shared.growths, shared.swap_moves], dtype=torch.int64, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    ok = int(flags[0]) == 0
    print(f"[rank {rank}] rows {rows_checked} bad {bad_rows} max_err {max_err:.4f} growths {shared.growths} "
          f"gen {shared.comm.generation} swap_moves {shared.swap_moves} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if a.caps_scale < 1.0:
        assert int(flags[1]) > 0, "expected at least one capacity growth with scaled-down capacities"
    if rank == 0:
        print(f"[check] {'PASS' if ok else 'FAIL'} total bad rows {int(flags[0])}, growths {int(flags[1]) // W}, "
              f"swap moves {int(flags[2])}, redos {shared.redos}, graph replays {getattr(shared, 'graph_replays', 0)}, "
              f"{time.time() - t0:.1f}s; caps {shared.caps}",
              flush=True)
    if a.out_hash:
        print(f"[rank {rank}] output digest {digest.hexdigest()}", flush=True)
    dist.barrier()
    if a.graphs:
        # a communicator whose collectives were captured in graphs hangs in ncclCommFinalize at teardown: drop the
        # graphs, then leave without the process-group teardown
        shared._graphs.clear()
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if ok else 1)
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
