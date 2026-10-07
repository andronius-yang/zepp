"""The device lane's single-GPU kernels (lane_arm, pad_rebuild) against the host reference (expect_arm) and the
host pad lists.

Decisions come from the device swap decision (swap_decide) on skewed random loads, as in test_swap_decide.py.
For every step with a swap and every rank: lane_arm must write the gate words of the unchanged slots (both
matrices), the moved-last schedule encoding (bit 30 marks an incoming slot, the low bits rank the slot within its
class) and the overrides of the incoming slots, and touch nothing else; pad_rebuild must write the table
LayerState._refresh_pads builds from the new placement. Steps without a swap must leave every buffer untouched.
The pushes and commits need node peers and are covered by the serving harness with the torch reference
(examples/serving_check.py --ref 1, device decision).

    python tests/test_lane_device.py [--cases 300] [--seed 1]
"""
import argparse

import numpy as np
import torch

from zepp import constants
from zepp._ext import C
from zepp.config import EPMoEConfig, ModelShape
from zepp.placement import Placement, demand_histogram, solve_placement
from zepp.serving import LayerState

TOPOLOGIES = [(4, 4, 32), (8, 4, 32), (16, 4, 128), (32, 4, 128), (16, 4, 384)]
SENT = -7


def skewed(G, gen, hot=0.25, mass=0.7):
    perm = torch.randperm(G, generator=gen)
    p = torch.full((G,), (1 - mass) / (G - int(G * hot)))
    p[perm[:int(G * hot)]] = mass / int(G * hot)
    return p


def expect_arm(moves_r, nlp, cap, epoch, stag_addr):
    gpe = nlp + 1
    gate = np.full(2 * gpe, SENT, dtype=np.int64)
    sched = np.full(gpe, SENT, dtype=np.int32)
    ovr = [np.full(gpe, SENT, dtype=np.int64) for _ in (0, 1)]
    if not moves_r:
        return gate, sched, ovr
    changed = {dj for (dj, _sr, _ss, _e) in moves_r}
    for i, (dj, _sr, _ss, _e) in enumerate(moves_r):
        ovr[0][1 + dj] = stag_addr[0][i]
        ovr[1][1 + dj] = stag_addr[1][i]
    f = d = 0
    for s in range(gpe):
        if s >= 1 and (s - 1) in changed:
            sched[s] = (1 << 30) | d
            d += 1
        else:
            sched[s] = f
            f += 1
            if s >= 1:
                gate[s] = epoch
                gate[gpe + s] = epoch
    return gate, sched, ovr


def run_sequence(R, L, G, steps, gen, stats):
    K, S_max = 8, 256
    cfg = EPMoEConfig(shape=ModelShape(G, K, 64, 32, act="swiglu"), ranks=R, ranks_per_node=L, max_tokens_per_rank=S_max)
    nlp, cap = cfg.slots_per_rank, constants.SWAP_MAX_MOVES
    p = skewed(G, gen)
    pool = torch.multinomial(p, R * 512 * K, replacement=True, generator=gen).view(R, 512, K).int()
    placement = solve_placement(demand_histogram(pool, L, G).cpu(), L, nlp)
    states = LayerState(0, cfg, placement, 0, dtype=torch.bfloat16)
    out = torch.zeros(C.swap_decide_out_ints(R, G, nlp, cap), dtype=torch.int64, device="cuda")
    stag_addr = [[(k + 1) * 10 ** 9 + i * 4096 for i in range(cap)] for k in (0, 1)]
    stag_dev = torch.tensor(stag_addr, dtype=torch.int64, device="cuda")
    for step in range(steps):
        if step % 3 == 2:
            p = skewed(G, gen)
        S_b = int(torch.randint(1, S_max // K + 1, (1,), generator=gen)) * K
        n_tokens = [int(v) for v in torch.randint(0, S_b + 1, (R,), generator=gen)]
        loads = torch.zeros(R, G, dtype=torch.int32)
        states.refresh_pads_if_needed()
        for r in range(R):
            real = torch.multinomial(p, n_tokens[r] * K, replacement=True, generator=gen) if n_tokens[r] else \
                torch.zeros(0, dtype=torch.long)
            pad = states._pad_lists[r]
            npad = (S_b - n_tokens[r]) * K
            pad_ids = torch.from_numpy(pad[(np.arange(n_tokens[r] * K, n_tokens[r] * K + npad)) % len(pad)])
            loads[r] = torch.bincount(torch.cat([real, pad_ids.long()]), minlength=G).int()
        C.swap_decide(loads.cuda(), torch.tensor(n_tokens, dtype=torch.int64, device="cuda"), S_b, K, states.p2l,
                      states.l2p, states.lcnts, L, nlp, float(cfg.router_c), cap, 32, out)
        blk = out.cpu().numpy()
        rounds = int(blk[0])
        assert int(blk[2]) == 0
        off_mv = 3 + R
        mv = blk[off_mv:off_mv + R * cap * 4].reshape(R, cap, 4)
        epoch = 1000 + step
        for r in range(R):
            gate = torch.full((2 * (nlp + 1),), SENT, dtype=torch.int64, device="cuda")
            sched = torch.full((nlp + 1,), SENT, dtype=torch.int32, device="cuda")
            ovr = [torch.full((nlp + 1,), SENT, dtype=torch.int64, device="cuda") for _ in (0, 1)]
            C.lane_arm(out, R, r, nlp, cap, gate, sched, ovr[0], ovr[1], stag_dev, epoch)
            moves_r = [tuple(int(v) for v in mv[r, i]) for i in range(int(blk[3 + r]))] if rounds else []
            eg, es, eo = expect_arm(moves_r, nlp, cap, epoch, stag_addr)
            assert np.array_equal(gate.cpu().numpy(), eg), f"gate R={R} G={G} step={step} rank={r}"
            assert np.array_equal(sched.cpu().numpy(), es), f"sched R={R} G={G} step={step} rank={r}"
            assert np.array_equal(ovr[0].cpu().numpy(), eo[0]) and np.array_equal(ovr[1].cpu().numpy(), eo[1]), \
                f"override R={R} G={G} step={step} rank={r}"
            pad = torch.full((S_max * K,), SENT, dtype=torch.int32, device="cuda")
            C.pad_rebuild(out, states.p2l, states.lcnts, r, nlp, G // R, pad)
            if rounds:
                p2l_new = blk[off_mv + R * cap * 4:off_mv + R * cap * 4 + R * nlp]
                lc = states.placement.lcnts.numpy().astype(np.int64)
                lst = np.asarray(states._pad_experts(p2l_new, lc, r), dtype=np.int32)
                want = lst[np.arange(S_max * K) % len(lst)]
                assert np.array_equal(pad.cpu().numpy(), want), f"pad table R={R} G={G} step={step} rank={r}"
                stats["arm_checks"] += 1
            else:
                assert bool((pad == SENT).all()), "pad_rebuild wrote without a swap"
                stats["quiet_checks"] += 1
        if rounds:
            p2l_new = torch.from_numpy(blk[off_mv + R * cap * 4:off_mv + R * cap * 4 + R * nlp].astype(np.int32))
            states.set_placement_host(Placement(p2l_new, None, states.placement.lcnts, states.placement.stats))
            stats["swaps"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(a.seed)
    stats = dict(swaps=0, arm_checks=0, quiet_checks=0)
    per = a.cases // len(TOPOLOGIES)
    for (R, L, G) in TOPOLOGIES:
        for _ in range(max(1, per // 10)):
            run_sequence(R, L, G, 10, gen, stats)
        print(f"R={R} L={L} G={G}: ok ({stats})", flush=True)
    print(f"PASS: lane_arm and pad_rebuild match the host reference and the host pad lists on {stats['swaps']} swap "
          f"steps ({stats['arm_checks']} rank checks) and leave every buffer untouched on {stats['quiet_checks']} "
          f"quiet rank checks", flush=True)


if __name__ == "__main__":
    main()
