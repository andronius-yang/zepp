"""Calibration: SGLang expert-distribution dumps (mode per_token, recorded by the stock server,
launch/server.sh baseline with CALIBRATE=1) -> per-layer placement, buffer capacities and the two
launch files of the zepp backend.

  python -m zepp_sglang.calibrate --dumps /path/to/dumps --model /path/to/model --ranks 16 \
      --ranks-per-node 4 --smax 2048 --swap 0 --router-c 0.25 --out /path/out

Writes out/zepp_config.json (ZEPP_CONFIG) and out/expert_location.json (--init-expert-location).
Each rank's dump holds the tokens of its own attention DP shard, so the per-rank attribution of the
pool and sizing shapes is exact.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from zepp import EPMoEConfig, ModelShape
from zepp.placement import demand_histogram, solve_placement
from zepp.routing import compute_capacities
from zepp.swap import swap_orbit


def load_topk(dumps, forward_modes=None):
    """{layer: {rank: int32 [N, K]}} from the per_token dump files (one per rank, suffix _<rank>.pt;
    records[i]["topk_ids_of_layer"] is [L, T, K], -1 padded). A rank's file holds the tokens that
    rank's attention DP shard routed, so the attribution is exact. forward_modes: keep only the records
    of these SGLang forward modes (1 extend / prefill, 2 decode); None keeps every record."""
    per_layer = {}
    if dumps in ("none", ""):
        return per_layer                                       # uniform routing for every layer
    files = sorted(glob.glob(os.path.join(dumps, "*.pt")))
    assert files, f"no .pt dumps under {dumps}"
    for f in files:
        stem = os.path.basename(f)[:-3]
        try:
            rank = int(stem.rsplit("_", 1)[1])
        except ValueError:
            rank = 0
        data = torch.load(f, map_location="cpu", weights_only=False)
        for rec in data["records"]:
            if forward_modes is not None and rec.get("forward_mode") not in forward_modes:
                continue
            t = rec["topk_ids_of_layer"]
            for l in range(t.shape[0]):
                rows = t[l]
                rows = rows[(rows >= 0).all(dim=1)]
                if rows.numel():
                    per_layer.setdefault(l, {}).setdefault(rank, []).append(rows.int())
    return {l: {r: torch.cat(v, 0) for r, v in by_rank.items()} for l, by_rank in per_layer.items()}


def sample_by_rank(rows_by_rank, R, S, seed):
    """[R, S, K]: rank r's rows sampled with replacement from its own records (union if none)."""
    g = torch.Generator().manual_seed(seed)
    union = torch.cat(list(rows_by_rank.values()), 0)
    out = []
    for r in range(R):
        rows = rows_by_rank.get(r, union)
        idx = torch.randint(0, rows.shape[0], (S,), generator=g)
        out.append(rows[idx])
    return torch.stack(out)


def sample(rows, R, S, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, rows.shape[0], (R * S,), generator=g)
    return rows[idx].view(R, S, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="per_token dump dir, or 'none' for uniform routing")
    ap.add_argument("--model", required=True, help="HF model dir (config.json)")
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--smax", type=int, default=2048)
    ap.add_argument("--swap", type=int, default=0)
    ap.add_argument("--router-c", type=float, default=0.25)
    ap.add_argument("--redundant", type=int, default=2)
    ap.add_argument("--pool-tokens", type=int, default=4096, help="tokens per rank in the placement window")
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--forward-mode", default="all", choices=["all", "decode", "prefill"],
                    help="calibrate on the decode (or prefill) forwards of the recording only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hf = json.load(open(os.path.join(a.model, "config.json")))
    shape = ModelShape(hf["num_experts"], hf["num_experts_per_tok"], hf["hidden_size"], hf["moe_intermediate_size"],
                       act="swiglu" if hf.get("hidden_act", "silu") == "silu" else "gelu")
    cfg = EPMoEConfig(shape=shape, ranks=a.ranks, ranks_per_node=a.ranks_per_node, max_tokens_per_rank=a.smax,
                      comm_strategy="overlap", swap=bool(a.swap), router_c=a.router_c,
                      redundant_slots_per_rank=a.redundant)
    R, L, nlp, G, K = cfg.ranks, cfg.ranks_per_node, cfg.slots_per_rank, shape.num_experts, shape.topk
    per_layer = load_topk(a.dumps, {"all": None, "decode": {2}, "prefill": {1}}[a.forward_mode])
    n_layers = hf["num_hidden_layers"]
    print(f"[calibrate] {len(per_layer)} layers with routing, "
          f"{sum(v.shape[0] for by in per_layer.values() for v in by.values())} token rows")
    p2l, phys2log, caps_all = {}, [], []
    fields = ("recv_cap", "dispatch_recv", "dispatch_stage", "dispatch_relay", "combine_send", "combine_conv",
              "combine_wire", "pair_cap")
    for l in range(n_layers):
        by_rank = per_layer.get(l)
        if by_rank is None:                                    # dense layer or no data: uniform pool
            g = torch.Generator().manual_seed(77 + l)
            by_rank = {0: torch.stack([torch.randperm(G, generator=g)[:K] for _ in range(4096)]).int()}
        pool = sample_by_rank(by_rank, R, a.pool_tokens, 1000 + l)
        pl = solve_placement(demand_histogram(pool, L, G), L, nlp)
        sizing = sample_by_rank(by_rank, R, a.smax, 2000 + l)
        pls = [pl]
        if cfg.swap:
            pls += swap_orbit(torch.bincount(sizing.reshape(-1).long(), minlength=G), pl, L, nlp, cfg.router_c)
        caps_all.append(compute_capacities(sizing, pls, nlp, L, cfg.router_c, None))
        p2l[l] = pl.p2l.tolist()
        home = G // R
        m = [e if e >= 0 else (s // nlp) * home + (s % nlp) % home for s, e in enumerate(p2l[l])]
        phys2log.append(m)
        if l % 10 == 0:
            print(f"[calibrate] layer {l}: replicas max {int(pl.lcnts.max())}, caps {caps_all[-1]}", flush=True)
    pad = a.smax * K                                           # pad rows of a zero-token rank
    caps = {f: int(np.ceil(a.factor * max(getattr(c, f) for c in caps_all))) for f in fields}
    for f in ("recv_cap", "dispatch_recv", "combine_send", "pair_cap"):
        caps[f] += pad
    os.makedirs(a.out, exist_ok=True)
    conf = dict(shape=dict(num_experts=G, topk=K, hidden=shape.hidden, ffn_hidden=shape.ffn_hidden, act=shape.act),
                ranks=R, ranks_per_node=L, max_tokens_per_rank=a.smax, comm_strategy="overlap", swap=bool(a.swap),
                router_c=a.router_c, redundant_slots_per_rank=a.redundant, caps=caps, num_layers=n_layers,
                p2l={str(l): v for l, v in p2l.items()}, source=dict(dumps=os.path.abspath(a.dumps), factor=a.factor,
                                                                   forward_mode=a.forward_mode,
                                                                   pad_allowance=pad))
    json.dump(conf, open(os.path.join(a.out, "zepp_config.json"), "w"))
    json.dump({"physical_to_logical_map": phys2log}, open(os.path.join(a.out, "expert_location.json"), "w"))
    print(f"[calibrate] caps {caps}\n[calibrate] wrote {a.out}/zepp_config.json and expert_location.json "
          f"(--ep-num-redundant-experts {R * a.redundant})")


if __name__ == "__main__":
    main()
