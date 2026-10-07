"""The combine op's device plan kernels against the numpy references in routing.py.

- C.combine_plan (a2av_combine_plan_block: the plan block every rank's combine forward reads, plus the
  bucket receiver's lane table) against routing.combine_plan_ref, bitwise,
  for every rank (a sample of ranks at W >= 32);
- C.msplit_tables (a2av_msplit_tables: the msplit GEMM problem tables and the weight-gate map) against
  routing.msplit_tables_ref, from the device block's node_base, with and without gated experts.

The reference checks need no GPU and run first (alone with --cpu): combine_plan_ref against the independent
host formulas of routing.combine_demands, and the cross-rank identities the combine's data movement relies
on (every sender's put offsets tile its receiver's panel exactly).

Shapes: W in {4, 16, 32, 64} ranks (L = 4), gpe in {5, 11}, K = 8; routings: uniform random, everything on
one rank's slots (zero-row ranks, remote nodes with no rows), one hot expert, two experts at the ends.

    python tests/test_plan_device.py [--seed 1] [--cpu]      (or: pytest tests/test_plan_device.py)
"""
import argparse

import numpy as np
import torch

from zepp.routing import (COMBINE_PLAN_HEADER, combine_demands, combine_plan_layout, combine_plan_ref,
                           meta_from_virtual_route, msplit_tables_ref, virtual_route)


def routings(W, S, K, gpe, rng, seed):
    nlp = gpe - 1
    E = W * gpe
    yield "uniform", virtual_route(torch.from_numpy(rng.integers(0, W * nlp, size=(W * S, K))), nlp, gpe)
    phys = torch.randint(0, nlp, (W * S, K), generator=torch.Generator().manual_seed(seed + W + S))
    yield "one-rank", virtual_route(phys, nlp, gpe)
    yield "one-expert", torch.full((W * S, K), 1, dtype=torch.int32)
    yield "two-experts", (torch.from_numpy(rng.integers(0, 2, size=(W * S, K))) * (E - 1)).int()


def header(blk):
    return {name: int(blk[k]) for k, name in enumerate(COMBINE_PLAN_HEADER)}


def check_tiles(intervals, total, what):
    """[start, start + rows) intervals (rows > 0) must tile [0, total) exactly."""
    iv = sorted((s, r) for s, r in intervals if r > 0)
    pos = 0
    for s, r in iv:
        assert s == pos, f"{what}: gap or overlap at {pos} (next interval starts {s})"
        pos += r
    assert pos == total, f"{what}: intervals cover {pos} of {total}"


def cpu_checks(sps, uc, W, L, gpe, S, K, tag):
    """combine_plan_ref of every rank: agreement with routing.combine_demands and the cross-rank identities."""
    NN = W // L
    lay = combine_plan_layout(W, NN, gpe)
    blks = [combine_plan_ref(sps, uc, W, gpe, L, NN, r)[0] for r in range(W)]
    hdr = [header(b) for b in blks]
    cd = combine_demands(sps, uc, W, L, gpe)
    cnt = sps.long().numpy()
    for r, (b, h) in enumerate(zip(blks, hdr)):
        assert h["m_this_ep"] == int(cnt[:, r * gpe:(r + 1) * gpe].sum()), (tag, r)
        assert h["col_min"] == h["col_max"] == S * K, (tag, r, h)
        assert max(h["max_send"], 1) == cd["send"], (tag, r, h, cd)
        if NN > 1:
            assert max(h["max_conv"], 1) == cd["conv"] and max(h["max_wire"], 1) == cd["wire"], (tag, r, h, cd)
        assert b[lay["lane_off"] + W] == h["own_total"] + h["rem_total"] <= S * K, (tag, r)
        assert b[lay["send_off"] + W] == h["m_this_ep"], (tag, r)
        nb = b[lay["node_base"]:lay["node_base"] + gpe * (NN + 1)].reshape(gpe, NN + 1)
        assert int(nb[:, NN].sum()) == h["m_this_ep"], (tag, r)
        mine = r // L
        assert h["remote_rows"] == h["m_this_ep"] - int((nb[:, mine + 1] - nb[:, mine]).sum()), (tag, r)
        segs = [(int(b[lay["wire_seg"] + k]), 0) for k in range(NN - 1)]
        rows = [int(b[lay["wire_rows"] + tn]) for tn in range(NN) if tn != mine]
        check_tiles([(s, n) for (s, _), n in zip(segs, rows)], h["wire_total"], f"{tag} wire panel of rank {r}")
    # receive image of home d: sender s puts its lane at its dst_off[d], the receiver reads lane s at lane_off[s]
    for d in range(W):
        lane = blks[d][lay["lane_off"]:lay["lane_off"] + W + 1]
        for s in range(W):
            assert blks[s][lay["dst_off"] + d] == lane[s], (tag, "dst_off vs lane_off", s, d)
        check_tiles([(int(lane[s]), int(lane[s + 1] - lane[s])) for s in range(W)], int(lane[W]),
                    f"{tag} receive image of rank {d}")
    # convergence panel of gateway g = (n, l): the senders of node n stack their rows for every remote
    # target (tn, l) at their conv_dst; together they tile the gateway's conv_total exactly
    for g in range(W):
        n, l = g // L, g % L
        iv = []
        for s in range(n * L, (n + 1) * L):
            for tn in range(NN):
                if tn != n:
                    iv.append((int(blks[s][lay["conv_dst"] + tn * L + l]), int(blks[s][lay["send_rows"] + tn * L + l])))
        check_tiles(iv, hdr[g]["conv_total"], f"{tag} convergence panel of gateway {g}")
    return blks


def device_checks(sps, uc, W, L, gpe, ranks, rng, tag, stats):
    from zepp._ext import C
    NN = W // L
    lay = combine_plan_layout(W, NN, gpe)
    sps_d, uc_d = sps.int().cuda().contiguous(), uc.int().cuda().contiguous()
    for r in ranks:
        seq = int(rng.integers(1, 1 << 40))
        blk, lanes = C.combine_plan(sps_d, uc_d, W, L, gpe, r, seq)
        rb, rl = combine_plan_ref(sps, uc, W, gpe, L, NN, r, seq)
        blk, lanes = blk.cpu().numpy(), lanes.cpu().numpy()
        if not np.array_equal(blk, rb):
            bad = np.nonzero(blk != rb)[0]
            raise AssertionError(f"{tag} rank {r}: plan block differs in {bad.size} words, first {bad[:5].tolist()}: "
                                 f"device {blk[bad[:5]].tolist()} ref {rb[bad[:5]].tolist()} (layout {lay})")
        assert np.array_equal(lanes, rl), (tag, r, "lanes", lanes.tolist(), rl.tolist())
        stats["blocks"] += 1
        # msplit tables from the device node_base: own node last, remote nodes in ring order
        nb = torch.from_numpy(blk[lay["node_base"]:lay["node_base"] + gpe * (NN + 1)]).cuda()
        me = r // L
        waves = [((me + 1 + i) % NN, (me + 1 + i) % NN + 1) for i in range(NN - 1)] + [(me, me + 1)]
        for reorder in (False, True):
            gate = [int(x) if x < gpe - 1 else -1 for x in rng.integers(0, 2 * gpe, size=gpe)]
            if not reorder:
                gate = [-1] * gpe if rng.integers(0, 2) else gate
            for wv in (waves, []):
                n_wgate = (len(wv) if wv else 1) * gpe
                tab, wg = C.msplit_tables(nb, gpe, NN, [a for a, _ in wv], [b for _, b in wv], gate,
                                          reorder and bool(wv), n_wgate)
                rt, rw = msplit_tables_ref(nb.cpu().numpy(), gpe, NN, wv, gate, reorder and bool(wv), n_wgate)
                assert np.array_equal(tab.cpu().numpy(), rt), (tag, r, wv, gate, reorder, "tables")
                assert np.array_equal(wg.cpu().numpy(), rw), (tag, r, wv, gate, reorder, "wgate")
                stats["msplit"] += 1


def run(seed=1, cpu=False):
    rng = np.random.default_rng(seed)
    stats = {"cases": 0, "blocks": 0, "msplit": 0}
    K = 8
    for W in (4, 16, 32, 64):
        L = 4
        for gpe in (5, 11):
            for S in (8, 64):
                for name, vce in routings(W, S, K, gpe, rng, seed):
                    tag = f"{name} W{W} gpe{gpe} S{S}"
                    _, _, sps, uc = meta_from_virtual_route(vce, W, S, gpe, W // L, L)
                    cpu_checks(sps, uc, W, L, gpe, S, K, tag)
                    stats["cases"] += 1
                    if cpu:
                        continue
                    ranks = range(W) if W <= 16 else sorted({0, 1, L - 1, L, W // 2, W - 1,
                                                             *rng.integers(0, W, size=4).tolist()})
                    device_checks(sps, uc, W, L, gpe, ranks, rng, tag, stats)
    if cpu:
        print(f"test_plan_device (cpu): {stats['cases']} routings, combine_plan_ref agrees with combine_demands "
              f"and every cross-rank tiling identity holds")
    else:
        torch.cuda.synchronize()
        print(f"test_plan_device: {stats['cases']} routings, {stats['blocks']} plan blocks and {stats['msplit']} "
              f"msplit / weight-gate tables bitwise identical to the references")


def test_plan_reference_cpu():
    run(cpu=True)


def test_plan_device():
    run(cpu=not torch.cuda.is_available())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="reference checks only (no GPU)")
    a = ap.parse_args()
    run(a.seed, a.cpu)
