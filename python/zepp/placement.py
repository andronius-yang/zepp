"""Load-aware replicated expert placement.

The placement is a pure integer function of the per-node demand histogram hist[NN, G]
(experts x nodes), so every rank computes the identical result without communication:

  counts   replicate argmax(load / c) greedily, at most one instance per node;
  nodes    experts in (share desc, id asc) order take their instances one at a time on the
           best node by (affinity hist[node, e] desc, accumulated node share asc, node id
           asc) among nodes with free slots not yet hosting the expert; leftover slots are
           backfilled with the highest-share non-hosting experts;
  ranks    within a node, instances sorted (share desc, id asc) snake over the ranks;
  tables   each rank's hosted experts in ascending id occupy its slots 0..n-1.

Tables: p2l[R * slots] = expert of each physical slot (-1 empty), l2p[G, R] = physical slots
hosting each expert in ascending order (-1 pad), lcnts[G] = replica count.
"""
import heapq
from dataclasses import dataclass

import numpy as np
import torch

SHARE_BITS = 20  # integer per-instance share scale: (load << 20) // c


@dataclass
class Placement:
    p2l: torch.Tensor      # int32 [R * slots_per_rank]
    l2p: torch.Tensor      # int32 [G, R]
    lcnts: torch.Tensor    # int32 [G]
    stats: dict

    @property
    def num_experts(self) -> int:
        return int(self.lcnts.numel())

    def clone(self) -> "Placement":
        return Placement(self.p2l.clone(), self.l2p.clone(), self.lcnts.clone(), dict(self.stats))


def demand_histogram(topk_all: torch.Tensor, ranks_per_node: int, num_experts: int) -> torch.Tensor:
    """Per-node expert demand [NN, G] from routing [R, S, K] (rank-major rows)."""
    R, S, K = topk_all.shape
    NN = R // ranks_per_node
    tk = topk_all.long()
    ent_g = tk.reshape(-1)
    ent_tok = torch.arange(R * S, device=tk.device, dtype=torch.int64).repeat_interleave(K)
    ent_node = ent_tok // (ranks_per_node * S)
    h = torch.zeros(NN * num_experts, dtype=torch.int64, device=tk.device)
    h.index_add_(0, ent_node * num_experts + ent_g, torch.ones_like(ent_g))
    return h.view(NN, num_experts)


def _seg_prefix(sorted_vals, sorted_keys):
    csum = torch.cumsum(sorted_vals, 0) - sorted_vals
    N = sorted_vals.numel()
    if N == 0:
        return csum
    idx = torch.arange(N, dtype=torch.int64)
    newgrp = torch.ones(N, dtype=torch.bool)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    base = torch.cummax(torch.where(newgrp, csum, torch.zeros_like(csum)), 0).values
    return csum - base


def _run_ordinal(sorted_keys):
    N = sorted_keys.numel()
    idx = torch.arange(N, dtype=torch.int64)
    if N == 0:
        return idx
    newgrp = torch.ones(N, dtype=torch.bool)
    newgrp[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = torch.cummax(torch.where(newgrp, idx, torch.zeros_like(idx)), 0).values
    return idx - starts


def replica_counts(load_e: torch.Tensor, nodes: int, total_slots: int) -> torch.Tensor:
    G = int(load_e.numel())
    assert total_slots >= G, "fewer slots than experts"
    lo = load_e.tolist()
    c = [1] * G
    cap = min(nodes, total_slots)
    heap = [((-(lo[g] << SHARE_BITS), g)) for g in range(G)]
    heapq.heapify(heap)
    extra = total_slots - G
    while extra > 0 and heap:
        _, g = heapq.heappop(heap)
        if c[g] >= cap:
            continue
        c[g] += 1
        extra -= 1
        if c[g] < cap:
            heapq.heappush(heap, (-((lo[g] << SHARE_BITS) // c[g]), g))
    return torch.tensor(c, dtype=torch.int64)


def assign_nodes(hist: torch.Tensor, c: torch.Tensor, slots_per_node: int):
    """-> (ion [G, NN] bool instance-on-node, primary [G], spilled)."""
    NN, G = hist.shape
    load = hist.sum(0)
    share = (load << SHARE_BITS) // c
    order = torch.argsort(share * G + (G - 1 - torch.arange(G, dtype=torch.int64)),
                          descending=True, stable=True).tolist()
    aff_l = hist.t().tolist()
    c_l = c.tolist()
    share_l = share.tolist()
    free = [slots_per_node] * NN
    loadv = [0] * NN
    pairs_g, pairs_u = [], []
    primary_l = [0] * G
    spilled = 0
    for g in order:
        affs = aff_l[g]
        sg = share_l[g]
        hosted = []
        for _i in range(c_l[g]):
            bu, ba, bl = -1, -1, 0
            for u in range(NN):
                if free[u] <= 0 or u in hosted:
                    continue
                a = affs[u]
                if a > ba or (a == ba and (bu < 0 or loadv[u] < bl)):
                    bu, ba, bl = u, a, loadv[u]
            if bu < 0:
                spilled += c_l[g] - _i
                break
            hosted.append(bu)
            free[bu] -= 1
            loadv[bu] += sg
        for u in hosted:
            pairs_g.append(g)
            pairs_u.append(u)
        primary_l[g] = hosted[0] if hosted else -1
    assert all(p >= 0 for p in primary_l), "expert with no instance"
    ion = torch.zeros(G, NN, dtype=torch.bool)
    ion[torch.tensor(pairs_g, dtype=torch.int64), torch.tensor(pairs_u, dtype=torch.int64)] = True
    primary = torch.tensor(primary_l, dtype=torch.int64)
    if sum(free) > 0:  # backfill leftover slots: highest share first, per node
        node_free = torch.tensor(free, dtype=torch.int64)
        smax = int(share.clamp(max=1 << 32).max()) + 2
        share_c = share.clamp(max=1 << 32)
        cand = (~ion) & (node_free > 0).unsqueeze(0)
        gc_, uc_ = cand.nonzero(as_tuple=True)
        if gc_.numel():
            okey = (uc_ * smax + (smax - 1 - share_c[gc_])) * G + gc_
            o = torch.argsort(okey, stable=True)
            u_s = uc_[o]
            pre = _seg_prefix(torch.ones_like(u_s), u_s)
            fit = pre < node_free[u_s]
            ion[gc_[o[fit]], u_s[fit]] = True
    assert bool((ion.any(dim=1)).all()), "expert with no instance"
    return ion, primary, spilled


def assign_ranks(ion: torch.Tensor, load: torch.Tensor, c: torch.Tensor, ranks_per_node: int):
    G, NN = ion.shape
    gg, uu = ion.nonzero(as_tuple=True)
    share = (load << SHARE_BITS) // c
    smax = int(share.max()) + 2
    okey = (uu * smax + (smax - 1 - share[gg])) * G + gg
    order = torch.argsort(okey, stable=True)
    u_s = uu[order]
    pos = _run_ordinal(u_s)
    lane = pos % ranks_per_node
    down = (pos // ranks_per_node) % 2 == 1
    lane = torch.where(down, ranks_per_node - 1 - lane, lane)
    return gg[order], u_s * ranks_per_node + lane


def slot_tables(g_flat: torch.Tensor, r_flat: torch.Tensor, G: int, R: int, slots_per_rank: int):
    """(expert, rank) pairs -> (p2l, l2p, lcnts) under the canonical slot recipe."""
    nlp = slots_per_rank
    N = int(g_flat.numel())
    p2l = torch.full((R * nlp,), -1, dtype=torch.int32)
    l2p = torch.full((G, R), -1, dtype=torch.int32)
    lcnts = torch.zeros(G, dtype=torch.int64)
    lcnts.index_add_(0, g_flat, torch.ones(N, dtype=torch.int64))
    key_s = torch.sort(g_flat * R + r_flat).values
    assert N == 0 or bool((key_s[1:] != key_s[:-1]).all()), "duplicate host"
    o_r = torch.argsort(r_flat * G + g_flat, stable=True)
    r_s = r_flat[o_r]
    ordn = _run_ordinal(r_s)
    assert N == 0 or bool((ordn < nlp).all()), "rank over its slot count"
    phys_s = r_s * nlp + ordn
    p2l[phys_s] = g_flat[o_r].to(torch.int32)
    phys = torch.empty(N, dtype=torch.int64)
    phys[o_r] = phys_s
    o_g = torch.argsort(g_flat * (R * nlp) + phys, stable=True)
    g_s = g_flat[o_g]
    j = _run_ordinal(g_s)
    l2p[g_s, j] = phys[o_g].to(torch.int32)
    return p2l, l2p, lcnts.to(torch.int32)


def solve_placement(hist: torch.Tensor, ranks_per_node: int, slots_per_rank: int) -> Placement:
    """hist [NN, G] (host int64) -> Placement. Deterministic; identical on every rank."""
    NN, G = hist.shape
    R = NN * ranks_per_node
    hist = hist.long()
    load = hist.sum(0)
    c = replica_counts(load, NN, R * slots_per_rank)
    ion, primary, spilled = assign_nodes(hist, c, ranks_per_node * slots_per_rank)
    c_real = ion.long().sum(1)
    g_flat, r_flat = assign_ranks(ion, load, c_real.clamp(min=1), ranks_per_node)
    p2l, l2p, lcnts = slot_tables(g_flat, r_flat, G, R, slots_per_rank)
    remote_rows = int((hist * (~ion).t().long()).sum())
    return Placement(p2l, l2p, lcnts, {
        "replicas": int(c_real.sum()) - G,
        "spilled": spilled,
        "c_max": int(c_real.max()),
        "remote_rows_predicted": remote_rows,
    })


def rebuild_l2p(p2l: torch.Tensor, num_experts: int, ranks: int) -> torch.Tensor:
    """l2p from p2l (ascending physical slot per expert)."""
    p = p2l.cpu().numpy().astype(np.int64)
    phys = np.nonzero(p >= 0)[0]
    e = p[phys]
    order = np.argsort(e, kind="stable")          # stable: slots of one expert stay ascending
    e, phys = e[order], phys[order]
    col = np.arange(e.size) - np.searchsorted(e, e, side="left")
    l2p = np.full((num_experts, ranks), -1, dtype=np.int32)
    l2p[e, col] = phys
    return torch.from_numpy(l2p)
