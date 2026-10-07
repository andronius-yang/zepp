"""Capacity-constrained replica routing: host reference and buffer capacities.

For every expert e with global demand D_e served by c_e replicas the balanced reference load
is q_e = D_e / c_e; the router keeps every replica inside [floor((1-C) q_e), ceil((1+C) q_e)]
(paper eq. 2) while preserving each source's expert selections.

Algorithm (rotation water-fill): R rounds p = 0..R-1; in round p source rank i visits rank
tgt(i, p) = ((u_i + p // L) mod NN) * L + ((l_i + p) mod L) (itself, then its node, then the
remote nodes in rotation order) and takes min(remaining demand, replica room, unassigned minus
the other replicas' lower-bound deficits). The tables are a pure integer function of the
gathered demand histogram d[R, G] and the placement; a rank then materializes only its own
entries. A per-token vacate pass afterwards moves entries off remote nodes when the release /
extra budgets on both replicas allow it (fewer touched nodes per token, bounds preserved).

The device kernel (`zepp._C.route`) implements the same tables; this module is the reference
used to size buffers and to check the kernel.
"""
import math
from dataclasses import dataclass

import torch

from . import constants


def c_rational(C: float):
    """Routing slack C as an exact rational (num, den); C must be a dyadic rational."""
    assert math.isfinite(C) and C >= 0, f"router_c must be finite and >= 0: {C}"
    den = 1 << 16
    num = int(round(C * den))
    assert abs(num / den - C) < 1e-12, f"router_c {C} is not a dyadic rational"
    g = math.gcd(num, den) if num else den
    return num // g, den // g


def _ceil_div(a, b):
    return -((-a) // b)


def rotation_round(i, j, L, NN):
    u_i, l_i = i // L, i % L
    u_j, l_j = j // L, j % L
    return ((u_j - u_i) % NN) * L + ((l_j - l_i) % L)


def instance_phys_of_rank(l2p, lcnts, nlp, R):
    """l2p [G, Cmax] + lcnts [G] -> ipr [G, R] int64 (phys slot or -1)."""
    G, Cm = l2p.shape
    dev = l2p.device
    l2p_l = l2p.long()
    valid = (torch.arange(Cm, device=dev).unsqueeze(0) < lcnts.long().unsqueeze(1)) & (l2p_l >= 0)
    g_idx, j_idx = valid.nonzero(as_tuple=True)
    phys = l2p_l[g_idx, j_idx]
    ipr = torch.full((G, R), -1, dtype=torch.int64, device=dev)
    ipr[g_idx, phys // nlp] = phys
    return ipr


def replica_table(ipr):
    G, R = ipr.shape
    hosted = ipr >= 0
    c = hosted.long().sum(1)
    Cmax = max(int(c.max()), 1)
    ordn = hosted.long().cumsum(1) - 1
    rep = torch.full((G, Cmax), -1, dtype=torch.int64, device=ipr.device)
    g_idx, r_idx = hosted.nonzero(as_tuple=True)
    rep[g_idx, ordn[g_idx, r_idx]] = r_idx
    return rep, c


def route_bounds_per_expert(D, c, C_num, C_den):
    cc = c.clamp(min=1)
    U = _ceil_div((C_den + C_num) * D, C_den * cc)
    Lb = ((C_den - C_num) * D) // (C_den * cc)
    U = torch.where(c > 0, U, torch.zeros_like(U))
    Lb = torch.where(c > 0, Lb, torch.zeros_like(Lb))
    return U, Lb


def route_tables(d, ipr, L, C_num, C_den):
    """Deterministic table pass. d [R, G] int64, ipr [G, R] -> dict(M [G, R, Cmax] rows from
    source i to replica jj of e, rep, c, D, U, Lb, fill, Cmax)."""
    R, G = d.shape
    NN = R // L
    dev = d.device
    d = d.long()
    rep, c = replica_table(ipr)
    Cmax = rep.shape[1]
    D = d.sum(0)
    assert bool((c[D > 0] >= 1).all()), "demand for an expert with no replica"
    U, Lb = route_bounds_per_expert(D, c, C_num, C_den)
    valid = rep >= 0
    fill = torch.zeros(G, Cmax, dtype=torch.int64, device=dev)
    left = d.t().contiguous()
    unassigned = D.clone()
    M = torch.zeros(G, R, Cmax, dtype=torch.int64, device=dev)
    g_ar = torch.arange(G, device=dev, dtype=torch.int64)
    for p in range(R):
        du, dl = p // L, p % L
        for jj in range(Cmax):
            j = rep[:, jj]
            v = valid[:, jj]
            jj_safe = torch.where(v, j, torch.zeros_like(j))
            u_j, l_j = jj_safe // L, jj_safe % L
            i = ((u_j - du) % NN) * L + ((l_j - dl) % L)
            want = left[g_ar, i]
            deficit = (Lb.unsqueeze(1) - fill).clamp(min=0) * valid.long()
            def_tot = deficit.sum(1)
            def_j = deficit[:, jj]
            room = U - fill[:, jj]
            take = torch.minimum(torch.minimum(want, room), unassigned - (def_tot - def_j))
            take = torch.where(v, take.clamp(min=0), torch.zeros_like(take))
            M[g_ar, i, jj] = take
            fill[:, jj] += take
            left[g_ar, i] -= take
            unassigned -= take
    assert bool((unassigned == 0).all()), "stranded demand"
    assert bool((left == 0).all())
    return dict(M=M, rep=rep, c=c, D=D, U=U, Lb=Lb, fill=fill, Cmax=Cmax)


def visit_order(rep, R, L):
    G, Cmax = rep.shape
    NN = R // L
    dev = rep.device
    i = torch.arange(R, device=dev, dtype=torch.int64).view(R, 1, 1)
    j = rep.unsqueeze(0)
    valid = j >= 0
    j_safe = torch.where(valid, j, torch.zeros_like(j))
    pr = rotation_round(i, j_safe, L, NN)
    pr = torch.where(valid, pr, torch.full_like(pr, R + 1))
    order = torch.argsort(pr, dim=2, stable=True)
    return order, torch.gather(pr, 2, order)


def demand_of(topk_all, G):
    R, S, K = topk_all.shape
    dev = topk_all.device
    flat = topk_all.long().reshape(R, S * K)
    d = torch.zeros(R * G, dtype=torch.int64, device=dev)
    d.index_add_(0, (torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1) * G + flat).reshape(-1),
                 torch.ones(R * S * K, dtype=torch.int64, device=dev))
    return d.view(R, G)


def route_water_fill(topk_all, p2l, l2p, lcnts, nlp, L, C_num, C_den):
    """Reference of the table route for ALL ranks: [R, S, K] expert ids -> ([R, S, K] int32
    physical slots, tables dict incl. d)."""
    R, S, K = topk_all.shape
    dev = topk_all.device
    G = int(lcnts.numel())
    topk = topk_all.long()
    ipr = instance_phys_of_rank(l2p.to(dev), lcnts.to(dev), nlp, R)
    flat = topk.reshape(R, S * K)
    assert bool(((flat >= 0) & (flat < G)).all()), "expert ids out of range"
    d = demand_of(topk_all, G)
    tab = route_tables(d, ipr, L, C_num, C_den)
    M, rep, Cmax = tab["M"], tab["rep"], tab["Cmax"]
    order, _ = visit_order(rep, R, L)
    M_vis = torch.gather(M.permute(1, 0, 2), 2, order)
    cum = torch.cumsum(M_vis, dim=2)
    rep_vis = torch.gather(rep.unsqueeze(0).expand(R, G, Cmax), 2, order)
    ent_src = torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1).expand(R, S * K).reshape(-1)
    ent_g = flat.reshape(-1)
    key = ent_src * G + ent_g
    o = torch.argsort(key, stable=True)
    ks = key[o]
    idx = torch.arange(ks.numel(), device=dev, dtype=torch.int64)
    newgrp = torch.ones_like(ks, dtype=torch.bool)
    if ks.numel() > 1:
        newgrp[1:] = ks[1:] != ks[:-1]
    starts = torch.cummax(torch.where(newgrp, idx, torch.zeros_like(idx)), 0).values
    ordn = idx - starts
    src_o, g_o = ent_src[o], ent_g[o]
    cum_e = cum[src_o, g_o]
    seg = (ordn.unsqueeze(1) >= cum_e).sum(1)
    assert bool((seg < Cmax).all()), "ordinal beyond the segment table"
    j_o = rep_vis[src_o, g_o, seg]
    assert bool((j_o >= 0).all())
    phys_flat = torch.empty(R * S * K, dtype=torch.int64, device=dev)
    phys_flat[o] = ipr[g_o, j_o]
    phys = phys_flat.view(R, S, K)
    assert bool(p2l.to(dev).long()[phys].eq(topk).all()), "conservation violated"
    tab["d"] = d
    tab["ipr"] = ipr
    return phys.to(torch.int32), tab


def _lr_split(total, weights):
    W = sum(weights)
    if total <= 0 or W <= 0:
        return [0] * len(weights)
    base = [total * w // W for w in weights]
    rem = total - sum(base)
    frac = [(total * w) % W for w in weights]
    order = sorted(range(len(weights)), key=lambda k: (-frac[k], k))
    for k in order[:rem]:
        base[k] += 1
    return base


def vacate_budgets(d, tab):
    """release / extra [G, R, Cmax]: rows source i may move off / onto replica jj."""
    R, G = d.shape
    M, fill, U, Lb, c = tab["M"], tab["fill"], tab["U"], tab["Lb"], tab["c"]
    Cmax = tab["Cmax"]
    release = torch.zeros(G, R, Cmax, dtype=torch.int64)
    extra = torch.zeros(G, R, Cmax, dtype=torch.int64)
    d_l = d.t().tolist()
    M_l, fill_l = M.tolist(), fill.tolist()
    U_l, Lb_l, c_l = U.tolist(), Lb.tolist(), c.tolist()
    for g in range(G):
        for jj in range(c_l[g]):
            slack_lo = fill_l[g][jj] - Lb_l[g]
            if slack_lo > 0:
                w = [M_l[g][i][jj] for i in range(R)]
                rel = _lr_split(slack_lo, w)
                for i in range(R):
                    release[g, i, jj] = min(rel[i], w[i])
            slack_hi = U_l[g] - fill_l[g][jj]
            if slack_hi > 0:
                ex = _lr_split(slack_hi, d_l[g])
                for i in range(R):
                    extra[g, i, jj] = ex[i]
    return release, extra


def reference_route(topk_all, placement, nlp, L, C: float):
    """Deterministic reference of the device router for ALL ranks (water-fill + vacate).
    Returns (phys [R, S, K] int32, info) with info = dict(tables, release, extra, d)."""
    C_num, C_den = c_rational(C)
    R, S, K = topk_all.shape
    p2l, l2p, lcnts = placement.p2l, placement.l2p, placement.lcnts
    phys0, tab = route_water_fill(topk_all, p2l, l2p, lcnts, nlp, L, C_num, C_den)
    d = tab["d"]
    rep, Cmax = tab["rep"], tab["Cmax"]
    release, extra = vacate_budgets(d.cpu(), tab)
    ipr = instance_phys_of_rank(l2p.cpu(), lcnts.cpu(), nlp, R)
    rep_l, ipr_l = rep.tolist(), ipr.tolist()
    topk = topk_all.long().cpu()
    phys = phys0.long().cpu().clone()
    for i in range(R):
        home = i // L
        rel = release[:, i].tolist()
        ext = extra[:, i].tolist()
        tk = topk[i].tolist()
        ph = phys[i].tolist()
        for s in range(S):
            node_of = [ph[s][k] // nlp // L for k in range(K)]
            while True:
                nodes = {}
                for k in range(K):
                    if node_of[k] != home:
                        nodes.setdefault(node_of[k], []).append(k)
                if len(nodes) < 1:
                    break
                done_any = False
                for n in sorted(nodes, key=lambda x: (len(nodes[x]), x)):
                    targets = [home] + [m for m in nodes if m != n]
                    plan = []
                    ok = True
                    for k in nodes[n]:
                        g = tk[s][k]
                        jo = next(jj for jj in range(Cmax)
                                  if rep_l[g][jj] >= 0 and ipr_l[g][rep_l[g][jj]] == ph[s][k])
                        if rel[g][jo] <= 0:
                            ok = False
                            break
                        tgt = None
                        for m in targets:
                            for jj in range(Cmax):
                                j = rep_l[g][jj]
                                if j < 0:
                                    break
                                if j // L == m and ext[g][jj] > 0:
                                    tgt = jj
                                    break
                            if tgt is not None:
                                break
                        if tgt is None:
                            ok = False
                            break
                        plan.append((k, g, jo, tgt))
                    if not ok:
                        continue
                    for k, g, jo, jt in plan:
                        rel[g][jo] -= 1
                        ext[g][jo] += 1
                        ext[g][jt] -= 1
                        rel[g][jt] += 1
                        ph[s][k] = ipr_l[g][rep_l[g][jt]]
                        node_of[k] = rep_l[g][jt] // L
                    done_any = True
                    break
                if not done_any:
                    break
        phys[i] = torch.tensor(ph)
    assert bool(p2l.cpu().long()[phys].eq(topk).all()), "conservation violated"
    return phys.to(torch.int32), dict(tables=tab, release=release, extra=extra, d=d, ipr=ipr)


# --------------------------------------------------------------------------------------
# virtual routing metadata and buffer capacities
# --------------------------------------------------------------------------------------

def virtual_route(phys, nlp, gpe):
    """Physical slots -> virtual GEMM group ids (pad group first: v = owner*gpe + 1 + slot)."""
    p = phys.long()
    return ((p // nlp) * gpe + 1 + p % nlp).int()


def meta_from_virtual_route(vce, R, S, gpe, nn, L):
    """(splits, scatter_index, splits_per_source, unique_counts) of a virtual routing
    vce [R*S, K] — the host reference of the ops' derive."""
    dev = vce.device
    ntokens = R * S
    kg = vce.shape[1]
    E_virt = R * gpe
    vce_flat = vce.long().reshape(-1)
    scatter_index = vce_flat.argsort(stable=True).argsort().int().view(ntokens, kg)
    splits = torch.bincount(vce_flat, minlength=E_virt).int()
    home = torch.arange(ntokens, device=dev, dtype=torch.int64) // S
    src_of_copy = home.repeat_interleave(kg)
    sps = torch.bincount(src_of_copy * E_virt + vce_flat, minlength=R * E_virt).view(R, E_virt)
    owner = vce.long() // gpe
    flags = torch.zeros(ntokens, R, dtype=torch.bool, device=dev)
    flags.scatter_(1, owner, True)
    u_mat = flags.view(R, S, R).sum(1)
    U_mat = flags.view(ntokens, nn, L).any(dim=2).view(R, S, nn).sum(1)
    uc = torch.cat([u_mat, U_mat], dim=1)
    return splits, scatter_index, sps.int(), uc.int()


def _chunk_bound(U_mat, L, n, m, k):
    total = int(U_mat[n * L:(n + 1) * L, m].sum())
    return (total // L) * k + min(k, total % L)


def dispatch_demands(sps, uc, W, L, gpe):
    """Exact dispatch buffer demands (rows) of one routing: recv (node-union regions),
    per-relay staging, per-round relay staging."""
    nn = W // L
    u = uc[:, :W].long()
    U = uc[:, W:].long()
    m_per_rank = sps.long().view(W, W, gpe).sum(2).sum(0)

    def region_rows(s, d):
        if nn > 1 and s // L != d // L:
            return int(U[s, d // L])
        return int(u[s, d])

    max_recv = int(m_per_rank.max())
    for d in range(W):
        max_recv = max(max_recv, sum(region_rows(s, d) for s in range(W)))
    max_stage = max_relay = 0
    if nn > 1:
        for n in range(nn):
            for k in range(L):
                srows = sum(_chunk_bound(U, L, ns, n, k + 1) - _chunk_bound(U, L, ns, n, k)
                            for ns in range(nn) if ns != n)
                max_stage = max(max_stage, srows)
                per_round = [_chunk_bound(U, L, n, (n - dn + nn) % nn, k + 1)
                             - _chunk_bound(U, L, n, (n - dn + nn) % nn, k) for dn in range(1, nn)]
                max_relay = max(max_relay, constants.RELAY_SLOTS * max(per_round))
    return dict(recv=max(max_recv, 1), stage=max(max_stage, 1), relay=max(max_relay, 1))


def combine_demands(sps, uc, W, L, gpe):
    """Exact combine buffer demands (rows): send panel, gateway staging, convergence, wire."""
    nn = W // L
    chunks = sps.long().view(W, W, gpe).sum(2)
    U = uc[:, W:].long()
    rs_send = int(chunks.sum(0).max())
    rs_stage = rs_conv = rs_wire = 0
    if nn > 1:
        for gn in range(nn):
            for gl in range(L):
                rs_stage = max(rs_stage, sum(int(chunks[h][ns * L + gl]) for ns in range(nn) if ns != gn
                                             for h in range(gn * L, (gn + 1) * L)))
        for n2 in range(nn):
            for dl in range(L):
                rs_conv = max(rs_conv, sum(int(chunks[tn * L + dl][n2 * L + ls]) for tn in range(nn)
                                           if tn != n2 for ls in range(L)))
                rs_wire = max(rs_wire, sum(int(U[tn * L + dl][n2]) for tn in range(nn) if tn != n2))
    return dict(send=max(rs_send, 1), stage=max(rs_stage, 1), conv=max(rs_conv, 1), wire=max(rs_wire, 1))


@dataclass
class Capacities:
    recv_cap: int          # rows this rank may receive/compute (dispatch recv == combine send)
    dispatch_recv: int     # dispatch recv panel rows
    dispatch_stage: int    # gateway staging rows
    dispatch_relay: int    # per-round relay staging rows
    combine_send: int
    combine_conv: int
    combine_wire: int
    pair_cap: int          # rows on any (source, destination) pair (direct strategy staging)


def compute_capacities(topk_all, placements, nlp, L, C: float, group=None) -> Capacities:
    """Provable buffer capacities from the reference route on every placement the layer may
    route on (the resident placement and, with swap enabled, its reachable swap orbit)."""
    R, S, K = topk_all.shape
    W = R
    nn = W // L
    gpe = nlp + 1
    recv_ub = pair_ub = 0
    ext_dst_max = 0
    ext_pair = torch.zeros(W, W, dtype=torch.int64)
    disp = dict(recv=0, stage=0, relay=0)
    comb = dict(send=0, stage=0, conv=0, wire=0)
    for pl in placements:
        phys, info = reference_route(topk_all, pl, nlp, L, C)
        tab, extra = info["tables"], info["extra"]
        rep = tab["rep"]
        hosted = rep >= 0
        rep_c = rep.clamp(min=0)
        r_ub = torch.zeros(W, dtype=torch.int64)
        r_ub.index_add_(0, rep_c.reshape(-1), (tab["U"].unsqueeze(1) * hosted.long()).reshape(-1))
        idx = (torch.arange(W).view(1, W, 1) * W + rep_c.unsqueeze(1)).expand_as(tab["M"])
        p_ub = torch.zeros(W * W, dtype=torch.int64)
        p_ub.index_add_(0, idx.reshape(-1), ((tab["M"] + extra) * hosted.unsqueeze(1).long()).reshape(-1))
        e_dst = torch.zeros(W, dtype=torch.int64)
        e_dst.index_add_(0, rep_c.unsqueeze(1).expand_as(extra).reshape(-1),
                         (extra * hosted.unsqueeze(1).long()).reshape(-1))
        e_pair = torch.zeros(W * W, dtype=torch.int64)
        e_pair.index_add_(0, idx.reshape(-1), (extra * hosted.unsqueeze(1).long()).reshape(-1))
        recv_ub = max(recv_ub, int(r_ub.max()))
        pair_ub = max(pair_ub, int(p_ub.max()))
        ext_dst_max = max(ext_dst_max, int(e_dst.max()))
        ext_pair = torch.maximum(ext_pair, e_pair.view(W, W))
        vce = virtual_route(phys.view(R * S, K), nlp, gpe)
        _, _, sps, uc = meta_from_virtual_route(vce, R, S, gpe, nn, L)
        dd = dispatch_demands(sps, uc, W, L, gpe)
        cd = combine_demands(sps, uc, W, L, gpe)
        disp = {k: max(disp[k], dd[k]) for k in disp}
        comb = {k: max(comb[k], cd[k]) for k in comb}
    cushion = ext_dst_max + constants.CAPACITY_CUSHION_PER_RANK * W
    recv_cap = recv_ub + constants.CAPACITY_CUSHION_PER_RANK * W
    E = ext_pair.view(nn, L, nn, L).sum(3).sum(1)
    E.fill_diagonal_(0)
    cushion_relay = constants.RELAY_SLOTS * ((int(E.max()) + L - 1) // L + 1) + constants.CAPACITY_CUSHION_PER_RANK * W
    vals = torch.tensor([
        recv_cap,
        max(disp["recv"] + cushion, recv_cap),
        disp["stage"] + cushion,
        disp["relay"] + cushion_relay,
        max(comb["send"] + cushion, recv_cap),
        comb["conv"] + cushion,
        comb["wire"] + cushion,
        pair_ub + cushion,
    ], dtype=torch.int64)
    if group is not None and torch.distributed.is_initialized():
        v = vals.cuda()
        torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.MAX, group=group)
        vals = v.cpu()
    v = [int(x) for x in vals]
    return Capacities(*v)


# ---- dispatch plan block (sort_util.h DispatchPlan): numpy reference of the device plan kernel ----
DP_HEADER = 8
DP_ERR = dict(room=1, pieces=2, unique=4, union=8, recv=16, send=32, stage=64, relay=128)


def dispatch_plan_layout(L, nn):
    """Word offsets of the dispatch plan block: seg_off, round0, tn[0], tn stride, ns[0], ns stride, words."""
    nseg = L + nn - 1
    r0 = DP_HEADER + nseg + 1
    tn0 = r0 + 3 * L
    tns = 6 + 8 * L
    ns0 = tn0 + nn * tns
    nss = 3 + L
    return dict(seg=DP_HEADER, nseg=nseg, round0=r0, tn0=tn0, tns=tns, ns0=ns0, nss=nss, words=ns0 + nn * nss)


def _waterfill(V, L):
    """lb_minmove water-fill of one n -> m stream: bounds, kept / imported rows per relay, pieces
    (sl, j_lo, j_hi, k, off) in the device kernels' order (sources ascending, a source's kept rows
    first), and whether it fitted."""
    tot = sum(V)
    cap = [tot // L + (1 if k < tot % L else 0) for k in range(L)]
    keep = [min(V[k], cap[k]) for k in range(L)]
    room = [cap[k] - keep[k] for k in range(L)]
    off = list(keep)
    pieces, imp, ok = [], 0, True
    for sl in range(L):
        if keep[sl] > 0:
            pieces.append((sl, 0, keep[sl], sl, 0))
        j = keep[sl]
        while j < V[sl]:
            while imp < L and room[imp] == 0:
                imp += 1
            if imp >= L:
                ok = False
                break
            take = min(V[sl] - j, room[imp])
            pieces.append((sl, j, j + take, imp, off[imp]))
            off[imp] += take
            room[imp] -= take
            j += take
        if not ok:
            break
    bd = [0]
    for k in range(L):
        bd.append(bd[-1] + off[k])
    return bd, keep, [off[k] - keep[k] for k in range(L)], pieces, ok


def dispatch_plan_ref(sps, uc, W, L, gpe, rank, relay_slots, copies_per_rank, max_recv, max_stage, max_relay):
    """The dispatch plan block of `rank` (list of int64 words) from splits_per_source [W, W*gpe] and
    unique_counts [W, W + nn]: every size and offset the dispatch wire reads."""
    import numpy as np
    sps = np.asarray(sps, dtype=np.int64)
    uc = np.asarray(uc, dtype=np.int64)
    nn = W // L
    my_node, my_lr = rank // L, rank % L
    lay = dispatch_plan_layout(L, nn)
    P = [0] * lay["words"]
    u = uc[:, :W]
    U = uc[:, W:]
    err = 0

    def region(s, d):
        return int(U[s, d // L]) if s // L != d // L else int(u[s, d])

    def recv_off_of_u(s, d):
        return sum(region(sq, d) for sq in range(s))

    chunks = sps.reshape(W, W, gpe).sum(2)
    for s in range(W):
        for d in range(W):
            cv, uv = int(chunks[s, d]), int(u[s, d])
            if not (0 <= uv <= cv and (uv > 0) == (cv > 0)):
                err |= DP_ERR["unique"]
            if uv > int(U[s, d // L]):
                err |= DP_ERR["union"]
        for n in range(nn):
            if int(U[s, n]) > int(u[s, n * L:(n + 1) * L].sum()):
                err |= DP_ERR["union"]
    max_col = max(sum(region(s, d) for s in range(W)) for d in range(W))
    seg = [0]
    for n in range(nn):
        if n == my_node:
            for dl in range(L):
                seg.append(seg[-1] + int(u[rank, n * L + dl]))
        else:
            seg.append(seg[-1] + int(U[rank, n]))
    nseg = lay["nseg"]
    P[lay["seg"]:lay["seg"] + nseg + 1] = seg
    for dlg in range(L):
        d = my_node * L + dlg
        r = lay["round0"] + 3 * dlg
        P[r:r + 3] = [int(u[rank, d]), seg[my_node + dlg], recv_off_of_u(rank, d)]
    if nn > 1:
        wf = {}
        for n in range(nn):
            for m in range(nn):
                if n != m:
                    wf[(n, m)] = _waterfill([int(U[n * L + k, m]) for k in range(L)], L)
                    if not wf[(n, m)][4]:
                        err |= DP_ERR["room"]
                    if len(wf[(n, m)][3]) > 2 * L:
                        err |= DP_ERR["pieces"]

        def rows_of(n, m, k):
            return 0 if n == m else wf[(n, m)][0][k + 1] - wf[(n, m)][0][k]

        def peer_seg_base(plr, tn):
            prank = my_node * L + plr
            a = 0
            for n in range(tn):
                a += int(u[prank, n * L:(n + 1) * L].sum()) if n == my_node else int(U[prank, n])
            return a

        max_stage_rows = max(sum(rows_of(ns, n, k) for ns in range(nn) if ns != n)
                             for n in range(nn) for k in range(L))
        max_relay_rows = max(max(relay_slots * rows_of(n, (n - dn + nn) % nn, k) for dn in range(1, nn))
                             for n in range(nn) for k in range(L))
        if max_stage_rows > max_stage:
            err |= DP_ERR["stage"]
        if max_relay_rows > max_relay:
            err |= DP_ERR["relay"]
        for tn in range(nn):
            if tn == my_node:
                continue
            bd, keep, imp, pieces, _ = wf[(my_node, tn)]
            r = lay["tn0"] + tn * lay["tns"]
            own_only = imp[my_lr] == 0
            P[r + 0], P[r + 1] = bd[my_lr], bd[my_lr + 1]
            P[r + 2] = 1 if own_only else 0
            P[r + 3] = seg[tn if tn < my_node else tn + L - 1]
            P[r + 4] = sum(rows_of(n, tn, my_lr) for n in range(my_node) if n != tn)
            pcs = []
            if not own_only:
                if keep[my_lr] > 0:
                    pcs.append((my_lr, 0, 0, keep[my_lr]))
                for (sl, j_lo, j_hi, k, o) in pieces:
                    if k == my_lr and sl != my_lr:
                        pcs.append((sl, j_lo, o, j_hi - j_lo))
            for q, (sl, src, dst, rows) in enumerate(pcs):
                P[r + 6 + 4 * q:r + 10 + 4 * q] = [sl, peer_seg_base(sl, tn) + src, dst, rows]
            P[r + 5] = len(pcs)
        for ns in range(nn):
            if ns == my_node:
                continue
            bd = wf[(ns, my_node)][0]
            r = lay["ns0"] + ns * lay["nss"]
            P[r + 0], P[r + 1] = bd[my_lr], bd[my_lr + 1]
            P[r + 2] = sum(rows_of(n, my_node, my_lr) for n in range(ns) if n != my_node)
            for dlg in range(L):
                P[r + 3 + dlg] = recv_off_of_u(ns * L, my_node * L + dlg)
    if max_col > max_recv:
        err |= DP_ERR["recv"]
    if seg[nseg] > copies_per_rank:
        err |= DP_ERR["send"]
    P[0] = int(chunks[:, rank].sum())
    P[1] = seg[nseg]
    P[2] = int(u[rank, rank])
    P[3] = seg[my_node + my_lr]
    P[4] = recv_off_of_u(rank, rank)
    P[5] = err
    return P

# --------------------------------------------------------------------------------------
# combine plan block (host reference of the combine op's device plan kernels)
# --------------------------------------------------------------------------------------

COMBINE_PLAN_HEADER = ("m_this_ep", "own_total", "conv_total", "wire_total", "rem_total", "remote_rows",
                       "max_send", "max_conv", "max_wire", "col_min", "col_max", "seq")
COMBINE_PLAN_HEADER_WORDS = 16


def combine_plan_layout(W, NN, E):
    """Word offsets of the combine plan block (CombinePlanLayout in args/gemm_combine.h)."""
    off, o = {}, COMBINE_PLAN_HEADER_WORDS
    for name, n in (("send_off", W + 1), ("send_rows", W), ("dst_off", W), ("lane_off", W + 1), ("conv_dst", W),
                    ("wire_seg", NN), ("wire_rows", NN), ("node_base", E * (NN + 1))):
        off[name] = o
        o += n
    off["total"] = o
    return off


def combine_plan_ref(sps, uc, W, E, L, NN, rank, seq=1):
    """Reference of the combine plan block of `rank` and its bucket lane table, from the routing counts
    sps [W, W*E] (cnt[h][e]: copies of home h for expert e) and uc [W, W + NN] (U[h][n] = uc[h][W + n]:
    tokens of home h with a copy on node n). Returns (block int64 [layout total], lanes int32 [2W + 1]).

    C[s][d] = sum over owner s's experts of cnt[d][e] (rows s returns to home d); the receive image is
    compressed: cp[s][d] = C[s][d] when s and d share a node, U[d][node(s)] when s is remote with d's local
    rank (one pre-reduced row per token and node), else 0."""
    import numpy as np
    cnt = np.asarray(sps, dtype=np.int64).reshape(W, W * E)
    U = np.asarray(uc, dtype=np.int64).reshape(W, W + NN)[:, W:]
    C = cnt.reshape(W, W, E).sum(2).T
    node, lr = np.arange(W) // L, np.arange(W) % L
    my_node, my_lr = rank // L, rank % L
    same_node = node[:, None] == node[None, :]
    same_lr = lr[:, None] == lr[None, :]
    cp = np.where(same_node, C, np.where(same_lr, U.T[node, :], 0))
    lay = combine_plan_layout(W, NN, E)
    blk = np.zeros(lay["total"], dtype=np.int64)
    rows_to = C[rank]
    blk[lay["send_off"]:lay["send_off"] + W + 1] = np.concatenate([[0], np.cumsum(rows_to)])
    blk[lay["send_rows"]:lay["send_rows"] + W] = rows_to
    blk[lay["dst_off"]:lay["dst_off"] + W] = cp[:rank].sum(0)
    lane_off = np.concatenate([[0], np.cumsum(cp[:, rank])])
    blk[lay["lane_off"]:lay["lane_off"] + W + 1] = lane_off
    mine = slice(my_node * L, (my_node + 1) * L)            # owners on my node
    remote = [tn for tn in range(NN) if tn != my_node]
    for tn in remote:
        for dl in range(L):
            d = tn * L + dl
            # gateway (my_node, dl) stacks target nodes ascending, then my node's ranks by local rank
            acc = sum(int(C[mine, t2 * L + dl].sum()) for t2 in remote if t2 < tn)
            blk[lay["conv_dst"] + d] = acc + int(C[my_node * L:my_node * L + my_lr, d].sum())
    wire_rows = [int(U[tn * L + my_lr, my_node]) if tn != my_node else 0 for tn in range(NN)]
    blk[lay["wire_rows"]:lay["wire_rows"] + NN] = wire_rows
    acc = 0
    for seg, tn in enumerate(remote):
        blk[lay["wire_seg"] + seg] = acc
        acc += wire_rows[tn]
    blk[lay["wire_seg"] + NN - 1] = acc
    hn = node                                                 # home node of each home rank
    for i in range(E):
        col = cnt[:, rank * E + i]
        for n2 in range(NN + 1):
            blk[lay["node_base"] + i * (NN + 1) + n2] = int(col[hn < n2].sum())
    conv_g = np.zeros(W, dtype=np.int64)
    wire_g = np.zeros(W, dtype=np.int64)
    for n2 in range(NN):
        for dl in range(L):
            others = [tn * L + dl for tn in range(NN) if tn != n2]
            conv_g[n2 * L + dl] = C[n2 * L:(n2 + 1) * L][:, others].sum()
            wire_g[n2 * L + dl] = U[others, n2].sum()
    colsum = C.sum(0)
    hdr = dict(m_this_ep=rows_to.sum(), own_total=C[mine, rank].sum(),
               conv_total=sum(int(cnt[tn * L + my_lr, my_node * L * E:(my_node + 1) * L * E].sum()) for tn in remote),
               wire_total=sum(wire_rows), rem_total=sum(int(U[rank, m]) for m in remote),
               remote_rows=rows_to[node != my_node].sum(), max_send=C.sum(1).max(), max_conv=conv_g.max(),
               max_wire=wire_g.max(), col_min=colsum.min(), col_max=colsum.max(), seq=seq)
    for k, name in enumerate(COMBINE_PLAN_HEADER):
        blk[k] = int(hdr[name])
    # bucket receiver lanes: chain position = own node first in (my_lr - lr) order, then the same-lr
    # remote lanes in (my_node - node) order; other sources 0
    chain = np.zeros(W, dtype=np.int64)
    for s in range(W):
        if node[s] == my_node:
            chain[s] = (my_lr - lr[s]) % L
        elif lr[s] == my_lr:
            chain[s] = L - 1 + (my_node - node[s]) % NN
    lanes = np.concatenate([lane_off, chain]).astype(np.int32)
    return blk, lanes


def msplit_tables_ref(node_base, E, NN, waves, gate, reorder, n_wgate):
    """Reference of the combine's per-forward GEMM problem tables (a2av_msplit_tables): node_base
    [E, NN + 1] (exclusive home-node row prefix per local expert), waves = [(lo, hi)] home-node ranges,
    gate[e] = weight-signal index or -1. Problem order is wave-outer, experts ascending inside a wave, or
    with `reorder` ungated experts first then gated ones. Returns (tables int32 [4 P + n_waves] =
    rows | offset | expert | wave | non-empty experts per wave, wgate int32 [n_wgate])."""
    import numpy as np
    nb = np.asarray(node_base, dtype=np.int64).reshape(E, NN + 1)
    nw = len(waves)
    order = [e for e in range(E) if not reorder or gate[e] < 0] + [e for e in range(E) if reorder and gate[e] >= 0]
    pos = {e: k for k, e in enumerate(order)}
    P = nw * E
    out = np.zeros(4 * P + nw, dtype=np.int64)
    for w, (lo, hi) in enumerate(waves):
        for e in range(E):
            ip = w * E + pos[e]
            rows = nb[e, hi] - nb[e, lo]
            out[ip], out[P + ip], out[2 * P + ip], out[3 * P + ip] = rows, nb[e, lo], e, w
            out[4 * P + w] += rows > 0
    wg = np.zeros(n_wgate, dtype=np.int64)
    if nw > 0:
        for w in range(nw):
            for e in range(E):
                wg[w * E + pos[e]] = gate[e]
    else:
        for i in range(n_wgate):
            wg[i] = gate[i % E]
    return out.astype(np.int32), wg.astype(np.int32)
