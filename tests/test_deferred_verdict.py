"""The single-GPU pieces of the deferred capacity verdict (src/core/verdict.h).

1. The demands kernel with a verdict block: the violation mask is ORed in (sticky), the first violating layer
   latches its ordinal and demands (a later one does not), the test-abort bit, and once the block is set the
   step's counts, splits and dispatch plan block are zeroed and the dead word is 1; a clean step with a clean
   block changes nothing.
2. The scale fold equals torch's in-place multiply bit for bit (bf16, fp16), honours the decision word and the
   device row count, and sets the rows-bound assert bit.
3. The msplit tables: the device wave-adapt decision equals the host rule of gemm_combine.cc; with decision 0
   the tables describe the single-gate GEMM in the wave layout (wave 0 holds every row of each expert, the
   other waves are empty); the device gate map gives the same tables as the host gate list.
4. lane_arm in `always` mode: without a swap every gate word is raised, the schedule is the identity, the front
   count is gpe and the gate map is -1; with a swap the words of the default mode plus the front count and the
   gate map. pad_rebuild in save mode keeps the overwritten table and flags whether it rebuilt.

One GPU, no process group:  python tests/test_deferred_verdict.py [--seed 1]
"""
import argparse

import numpy as np
import torch

from zepp._ext import C

LAY = None


def demands_case(rng, W=8, L=4, gpe=5, K=8, S=64):
    E = W * gpe
    ids = torch.from_numpy(rng.integers(0, E, size=(W * S, K))).int().cuda()
    splits, _scatter, sps, uc = C.routed_meta(ids, W, L, gpe)
    big = [1 << 40] * 8
    ref = C.a2av_demands(sps, uc, W, L, gpe, 2, False, big)
    assert int(ref[7]) == 0
    words = LAY["words"]

    def run(v, caps, layer, force=False):
        s1, u1, sp1 = sps.clone(), uc.clone(), splits.clone()
        plan = torch.full((37,), 7, dtype=torch.int64, device="cuda")
        out = C.a2av_demands(s1, u1, W, L, gpe, 2, False, caps, verdict=v, layer=layer, force=force, splits=sp1,
                             plan=plan)
        torch.cuda.synchronize()
        return out, s1, u1, sp1, plan

    # clean step, clean block: nothing changes
    v = torch.zeros(words, dtype=torch.int64, device="cuda")
    out, s1, u1, sp1, plan = run(v, big, 0)
    assert torch.equal(out, ref) and int(v.abs().sum()) == 0
    assert torch.equal(s1, sps) and torch.equal(u1, uc) and torch.equal(sp1, splits) and bool((plan == 7).all())
    # violating step (recv_cap one row short): mask, latch, demands, dead, zeroed counts
    tight = list(big)
    tight[0] = int(ref[0]) - 1
    out, s1, u1, sp1, plan = run(v, tight, 1)
    m = int(out[7])
    assert m & 1, m
    assert int(v[LAY["mask"]]) == m and int(v[LAY["latch"]]) == 2 and int(v[LAY["dead"]]) == 1
    d0 = LAY["demands"]
    assert v[d0:d0 + 7].tolist() == ref[:7].tolist()
    assert int(s1.abs().sum()) == 0 and int(u1.abs().sum()) == 0 and int(sp1.abs().sum()) == 0
    # every word zeroed except the header's sequence word (word 6, DispatchPlan::kSeq), which is kept
    assert int(plan[6]) == 7 and int(plan.abs().sum()) == 7, plan.tolist()
    # a later step of the same forward: degenerate even though it fits, the latch keeps the first layer
    out, s1, u1, sp1, plan = run(v, big, 2)
    assert int(v[LAY["mask"]]) == m and int(v[LAY["latch"]]) == 2
    assert int(s1.abs().sum()) == 0 and int(sp1.abs().sum()) == 0
    # the test-abort bit on a clean block
    v.zero_()
    out, s1, u1, sp1, plan = run(v, big, 3, force=True)
    assert int(v[LAY["mask"]]) == LAY["force_bit"] and int(v[LAY["latch"]]) == 4 and int(v[LAY["dead"]]) == 1
    assert int(s1.abs().sum()) == 0
    return 4


def fold_case(rng):
    n = 0
    for dt in (torch.bfloat16, torch.float16):
        R, K, M = 97, 264, 61
        x = (torch.randn(R, K, device="cuda") * 3).to(dt)
        sc = torch.rand(R, device="cuda") * 2
        rows = torch.tensor([M], dtype=torch.int64, device="cuda")
        ref = x.clone()
        ref[:M].mul_(sc[:M].unsqueeze(1))
        y = x.clone()
        C.fold_scales(y, sc, rows)
        assert torch.equal(y, ref), dt
        dec = torch.zeros(1, dtype=torch.int32, device="cuda")
        y = x.clone()
        C.fold_scales(y, sc, rows, dec=dec)
        assert torch.equal(y, x)
        dec.fill_(1)
        C.fold_scales(y, sc, rows, dec=dec)
        assert torch.equal(y, ref)
        v = torch.zeros(LAY["words"], dtype=torch.int64, device="cuda")
        C.fold_scales(y, None, torch.tensor([R + 1], dtype=torch.int64, device="cuda"), fold=False, verdict=v)
        torch.cuda.synchronize()
        assert int(v[LAY["mask"]]) == LAY["error_bit"] and int(v[LAY["err"]]) == (1 << 8)
        n += 3
    return n


def waves_of(NN, my_node):
    """gemm_combine.cc forward_impl: ring runs of one node each, own node last."""
    w = [(n, n + 1) for n in range(my_node + 1, NN)] + [(n, n + 1) for n in range(0, my_node)]
    return w + [(my_node, my_node + 1)]


def msplit_case(rng):
    n = 0
    for NN, E in ((2, 5), (4, 9), (8, 3)):
        for my_node in (0, NN - 1):
            counts = rng.integers(0, 6, size=(E, NN))
            counts[rng.integers(0, E)] = 0                      # an empty expert
            nb = np.concatenate([np.zeros((E, 1), dtype=np.int64), np.cumsum(counts, 1)], 1)
            node_base = torch.from_numpy(nb.reshape(-1).astype(np.int64)).cuda()
            waves = waves_of(NN, my_node)
            lo, hi = [a for a, _ in waves], [b for _, b in waves]
            nw = len(waves)
            gate = [-1] * E
            for e in rng.choice(E, size=min(2, E), replace=False):
                gate[int(e)] = int(e) - 1 if e > 0 else -1
            gate_dev = torch.tensor(gate, dtype=torch.int32, device="cuda")
            host_t, host_w = C.msplit_tables(node_base, E, NN, lo, hi, gate, True, nw * E)
            dev_t, dev_w = C.msplit_tables(node_base, E, NN, lo, hi, [-1] * E, True, nw * E, gate_dev=gate_dev)
            assert torch.equal(host_t, dev_t) and torch.equal(host_w, dev_w), "device gate map"
            remote = int(sum(counts[:, m].sum() for m in range(NN) if m != my_node))
            for ratio in (0, 3, 48):
                reread, rowb = 7 * 1000, 10
                dec = torch.full((1,), -5, dtype=torch.int32, device="cuda")
                rr = torch.tensor([remote], dtype=torch.int64, device="cuda")
                t, w = C.msplit_tables(node_base, E, NN, lo, hi, gate, True, nw * E, dec=dec, remote_rows=rr,
                                       adapt_reread=reread, adapt_ratio=ratio, adapt_row_bytes=rowb)
                want = 0 if (ratio > 0 and reread > ratio * remote * rowb) else 1
                assert int(dec) == want, (ratio, remote, int(dec))
                assert torch.equal(w, host_w)
                if want == 1:
                    assert torch.equal(t, host_t)
                    continue
                np_ = nw * E
                tt = t.cpu().numpy()
                wave_M, wave_off, eid, grp, ne = (tt[:np_], tt[np_:2 * np_], tt[2 * np_:3 * np_],
                                                  tt[3 * np_:4 * np_], tt[4 * np_:])
                ht = host_t.cpu().numpy()
                assert np.array_equal(eid, ht[2 * np_:3 * np_]) and np.array_equal(grp, ht[3 * np_:4 * np_])
                for ip in range(np_):
                    e = int(eid[ip])
                    want_m = int(nb[e, NN]) if grp[ip] == 0 else 0
                    assert wave_M[ip] == want_m and wave_off[ip] == 0, (ip, e, wave_M[ip])
                assert ne[0] == int((nb[:, NN] > 0).sum()) and not ne[1:].any()
                n += 1
            n += 1
    return n


def lane_case(rng, R=8, L=4, nlp=6, G=40, cap=2):
    gpe = nlp + 1
    n_out = C.swap_decide_out_ints(R, G, nlp, cap)
    stag = torch.arange(1, 2 * cap + 1, dtype=torch.int64, device="cuda") * 4096
    n = 0
    for rank in (0, 5):
        for moves in ([], [(2, rank ^ 1, 4, 11)], [(0, rank ^ 1, 1, 3), (5, rank ^ 2, 2, 9)]):
            blk = torch.zeros(n_out, dtype=torch.int64, device="cuda")
            if moves:
                blk[0], blk[1] = 1, len(moves)
                blk[3 + rank] = len(moves)
                base = 3 + R + rank * cap * 4
                for i, mv in enumerate(moves):
                    blk[base + 4 * i:base + 4 * i + 4] = torch.tensor(mv)
            epoch = 17
            for always in (False, True):
                gate = torch.full((2 * gpe,), -3, dtype=torch.int64, device="cuda")
                sched = torch.full((gpe + 1,), -3, dtype=torch.int32, device="cuda")
                ovr = [torch.zeros(gpe, dtype=torch.int64, device="cuda") for _ in (0, 1)]
                gmap = torch.full((gpe,), -9, dtype=torch.int32, device="cuda")
                C.lane_arm(blk, R, rank, nlp, cap, gate, sched, ovr[0], ovr[1], stag, epoch, always=always,
                           gmap=gmap if always else None)
                torch.cuda.synchronize()
                if not always and not moves:
                    assert bool((gate == -3).all()) and bool((sched == -3).all()), "default mode, no move"
                    continue
                changed = {m[0] for m in moves}
                f = d = 0
                exp_sched = [0] * gpe
                exp_sched[0] = f
                f += 1
                for j in range(nlp):
                    if j in changed:
                        exp_sched[1 + j] = (1 << 30) | d
                        d += 1
                    else:
                        exp_sched[1 + j] = f
                        f += 1
                assert sched[:gpe].tolist() == exp_sched
                assert int(sched[gpe]) == (f if always else -3)
                for j in range(nlp):
                    want = -3 if j in changed else epoch
                    assert int(gate[1 + j]) == want and int(gate[gpe + 1 + j]) == want
                for i, mv in enumerate(moves):
                    assert int(ovr[0][1 + mv[0]]) == int(stag[i]) and int(ovr[1][1 + mv[0]]) == int(stag[cap + i])
                if always:
                    assert gmap.tolist() == [-1] + [j if j in changed else -1 for j in range(nlp)]
                n += 1
            # pad_rebuild save mode
            p2l = torch.from_numpy(rng.permutation(np.arange(-1, R * nlp - 1)) % G).long().cuda()
            lcnts = torch.bincount(p2l[p2l >= 0], minlength=G).int()
            pad = torch.from_numpy(rng.integers(0, G, size=64)).int().cuda()
            prev = torch.full_like(pad, -1)
            flag = torch.full((1,), -1, dtype=torch.int32, device="cuda")
            old = pad.clone()
            ref = pad.clone()
            C.pad_rebuild(blk, p2l, lcnts, rank, nlp, G // R, ref)
            C.pad_rebuild(blk, p2l, lcnts, rank, nlp, G // R, pad, pad_prev=prev, flag=flag)
            torch.cuda.synchronize()
            assert torch.equal(pad, ref)
            if moves:
                assert int(flag) == 1 and torch.equal(prev, old)
            else:
                assert int(flag) == 0 and bool((prev == -1).all()) and torch.equal(pad, old)
            n += 1
    return n


def main():
    global LAY
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    LAY = C.verdict_layout()
    counts = dict(demands=demands_case(rng), fold=fold_case(rng), msplit=msplit_case(rng), lane=lane_case(rng))
    print(f"PASS: deferred-verdict kernels {counts}")


def test_deferred_verdict():
    main()


if __name__ == "__main__":
    main()
