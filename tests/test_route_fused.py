"""The fused router (C.route_fused, the planner's router) against the reference router (C.route), on one GPU.

Random placements planned from skewed demand, random per-rank expert choices; every rank of the topology is routed
by both paths. Checked per rank:
  - the budget kernels: release + extra per (expert, replica) is invariant under the vacate pass (a move takes one
    release at its source and one extra at its target and returns one of each the other way), so after both paths
    the workspace sums must be equal to each other and to the host reference (routing.vacate_budgets);
  - every copy routed to a replica of its expert, no ticket overflow (stats[2] == 0);
  - the send row: send[:SK] == phys, send[SK:] == the gate weight bits.
Checked over all ranks: every replica's global fill inside [Lb, U] for both paths, and the remote (token, node)
pairs of the two paths (relaxed tickets: equal in distribution, printed).

    python tests/test_route_fused.py [--cases 300] [--seed 1]
"""
import argparse

import torch

from zepp._ext import C
from zepp.config import EPMoEConfig, ModelShape
from zepp.placement import demand_histogram, solve_placement
from zepp.routing import c_rational, instance_phys_of_rank, route_tables, vacate_budgets

TOPOLOGIES = [(16, 4, 128, 8), (8, 4, 32, 8), (32, 4, 128, 8), (64, 4, 128, 8), (16, 4, 128, 16), (4, 4, 128, 8)]
K_MAX_REP = 32


def skewed(G, gen, hot=0.25, mass=0.7):
    perm = torch.randperm(G, generator=gen)
    p = torch.full((G,), (1 - mass) / (G - int(G * hot)))
    p[perm[:int(G * hot)]] = mass / int(G * hot)
    return p


def ws_views(ws, G, R):
    o = G * R + 2 * G * K_MAX_REP + 2 * G + G * K_MAX_REP * R + G * K_MAX_REP
    rel = ws[o:o + G * K_MAX_REP].view(G, K_MAX_REP)
    ext = ws[o + G * K_MAX_REP:o + 2 * G * K_MAX_REP].view(G, K_MAX_REP)
    return rel, ext


def remote_pairs(phys, nlp, L, home):
    node = (phys.long() // nlp) // L
    pairs = 0
    for s in range(phys.shape[0]):
        pairs += len(set(node[s].tolist()) - {home})
    return pairs


def run_case(R, L, G, K, gen, stats):
    S = int(torch.randint(1, 257, (1,), generator=gen))
    cfg = EPMoEConfig(shape=ModelShape(G, K, 64, 32, act="swiglu"), ranks=R, ranks_per_node=L, max_tokens_per_rank=256)
    nlp = cfg.slots_per_rank
    c_num, c_den = c_rational(cfg.router_c)
    p = skewed(G, gen)
    pool = torch.multinomial(p, R * 512 * K, replacement=True, generator=gen).view(R, 512, K).int()
    pl = solve_placement(demand_histogram(pool, L, G).cpu(), L, nlp)
    if int(torch.randint(0, 2, (1,), generator=gen)):
        p = skewed(G, gen)                                  # demand shifted away from the planned one
    topk = torch.stack([torch.stack([torch.multinomial(p, K, replacement=False, generator=gen) for _ in range(S)])
                        for _ in range(R)]).int()           # [R, S, K]
    d = torch.stack([torch.bincount(topk[r].reshape(-1).long(), minlength=G) for r in range(R)]).int()
    ipr = instance_phys_of_rank(pl.l2p, pl.lcnts, nlp, R)
    tab = route_tables(d.long(), ipr, L, c_num, c_den)
    release, extra = vacate_budgets(d.long(), tab)
    budget_ref = (release + extra)                          # [G, R, Cmax]
    Cmax = tab["Cmax"]
    l2p, lcnts, p2l = pl.l2p.cuda(), pl.lcnts.cuda(), pl.p2l.cuda().long()
    dd = d.cuda()
    nws = C.route_workspace_ints(G, R)
    fills = {"route": torch.zeros(R * nlp, dtype=torch.int64), "fused": torch.zeros(R * nlp, dtype=torch.int64)}
    for i in range(R):
        tk = topk[i].cuda().contiguous()
        probs = torch.rand(S, K, generator=gen).cuda()
        ws_a = torch.full((nws,), -7, dtype=torch.int32, device="cuda")
        ws_b = torch.full((nws,), -7, dtype=torch.int32, device="cuda")
        send = torch.full((2 * S * K,), -7, dtype=torch.int32, device="cuda")
        ph_a, st_a = C.route(tk, dd, l2p, lcnts, i, nlp, L, c_num, c_den, ws_a)
        ph_b, st_b = C.route_fused(tk, dd, l2p, lcnts, i, nlp, L, c_num, c_den, ws_b, probs.view(-1), send)
        torch.cuda.synchronize()
        for name, ph, st, ws in (("route", ph_a, st_a, ws_a), ("fused", ph_b, st_b, ws_b)):
            assert int(st[2]) == 0, f"{name}: ticket overflow R={R} G={G} rank={i}"
            assert bool(p2l[ph.long()].eq(tk.long()).all()), f"{name}: copy routed off its expert R={R} rank={i}"
            rel, ext = ws_views(ws, G, R)
            got = (rel + ext)[:, :Cmax].cpu().long()
            ref = budget_ref[:, i, :]
            cnt = tab["c"]
            mask = torch.arange(Cmax).unsqueeze(0) < cnt.unsqueeze(1)
            assert torch.equal(got[mask], ref[mask]), f"{name}: budgets differ from the host reference R={R} rank={i}"
            fills[name] += torch.bincount(ph.reshape(-1).long().cpu(), minlength=R * nlp)
            stats[name + "_moves"] += int(st[1])
            stats[name + "_remote"] += remote_pairs(ph.cpu(), nlp, L, i // L)
        assert torch.equal(send[:S * K], ph_b.reshape(-1)), f"send row phys differs R={R} rank={i}"
        assert torch.equal(send[S * K:], probs.view(-1).view(torch.int32)), f"send row weights differ R={R} rank={i}"
    U, Lb = tab["U"], tab["Lb"]
    for name, f in fills.items():
        for g in range(G):
            for r in range(R):
                ph = int(ipr[g, r])
                if ph < 0:
                    continue
                assert int(Lb[g]) <= int(f[ph]) <= int(U[g]), \
                    f"{name}: replica fill {int(f[ph])} outside [{int(Lb[g])}, {int(U[g])}] R={R} g={g} r={r}"
    stats["cases"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(a.seed)
    stats = dict(cases=0, route_moves=0, fused_moves=0, route_remote=0, fused_remote=0)
    per = max(1, a.cases // len(TOPOLOGIES))
    for (R, L, G, K) in TOPOLOGIES:
        n = max(1, per * 16 // R)
        for _ in range(n):
            run_case(R, L, G, K, gen, stats)
        print(f"R={R} L={L} G={G} K={K}: {n} cases ok ({stats})", flush=True)
    rr, fr = stats["route_remote"], stats["fused_remote"]
    print(f"PASS: {stats['cases']} cases; budgets equal to the host reference on both paths, send rows exact, "
          f"bounds held; remote (token, node) pairs route {rr} fused {fr} ({(fr - rr) / max(rr, 1) * 100:+.2f}%), "
          f"vacate moves route {stats['route_moves']} fused {stats['fused_moves']}", flush=True)


if __name__ == "__main__":
    main()
