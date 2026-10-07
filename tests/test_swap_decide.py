"""The device swap decision (swap_decide kernel) against the host decision, on one GPU.

For random placements (planned from skewed demand, as the layer does), random gathered loads and random
per-rank token counts (pad rows included), both paths must produce the same loads after the pad correction,
the same placement, the same l2p table and the same per-rank pull lists; the placement then carries over to
the next case of the same sequence, so multi-step orbits are exercised.

    python tests/test_swap_decide.py [--cases 400] [--seed 1]
"""
import argparse

import numpy as np
import torch

from zepp import constants
from zepp._ext import C
from zepp.config import EPMoEConfig, ModelShape
from zepp.placement import demand_histogram, rebuild_l2p, solve_placement
from zepp.serving import LayerState
from zepp.swap import decide_swaps, net_moves

TOPOLOGIES = [(4, 4, 32), (8, 4, 32), (16, 4, 128), (32, 4, 128), (64, 4, 128), (16, 4, 384), (4, 4, 128)]


def skewed(G, gen, hot=0.25, mass=0.7):
    perm = torch.randperm(G, generator=gen)
    p = torch.full((G,), (1 - mass) / (G - int(G * hot)))
    p[perm[:int(G * hot)]] = mass / int(G * hot)
    return p


def run_sequence(R, L, G, steps, gen, stats):
    K, S_max = 8, 256
    cfg = EPMoEConfig(shape=ModelShape(G, K, 64, 32, act="swiglu"), ranks=R, ranks_per_node=L, max_tokens_per_rank=S_max)
    nlp = cfg.slots_per_rank
    p = skewed(G, gen)
    pool = torch.multinomial(p, R * 512 * K, replacement=True, generator=gen).view(R, 512, K).int()
    placement = solve_placement(demand_histogram(pool, L, G).cpu(), L, nlp)
    state = LayerState(0, cfg, placement, 0, dtype=torch.bfloat16)
    n_out = C.swap_decide_out_ints(R, G, nlp, constants.SWAP_MAX_MOVES)
    out = torch.zeros(n_out, dtype=torch.int64, device="cuda")
    for step in range(steps):
        state.refresh_pads_if_needed()
        if step % 3 == 2:                                  # shift the demand: swaps become likely
            p = skewed(G, gen)
        S_b = int(torch.randint(1, S_max // K + 1, (1,), generator=gen)) * K
        n_tokens = [int(v) for v in torch.randint(0, S_b + 1, (R,), generator=gen)]
        loads = torch.zeros(R, G, dtype=torch.int32)
        for r in range(R):
            real = torch.multinomial(p, n_tokens[r] * K, replacement=True, generator=gen) if n_tokens[r] else \
                torch.zeros(0, dtype=torch.long)
            pad = state._pad_lists[r]
            npad = (S_b - n_tokens[r]) * K
            pad_ids = torch.from_numpy(pad[(np.arange(n_tokens[r] * K, n_tokens[r] * K + npad)) % len(pad)])
            ids = torch.cat([real, pad_ids.long()])
            loads[r] = torch.bincount(ids, minlength=G).int()
        # host
        load_g = loads.long().sum(0) - state.pad_hist(n_tokens, S_b)
        ref_pl, _ = decide_swaps(load_g, state.placement, L, nlp, cfg.router_c)
        ref_moves = net_moves(state.placement.p2l, ref_pl.p2l, L, nlp) if ref_pl is not None else [[] for _ in range(R)]
        # device
        C.swap_decide(loads.cuda(), torch.tensor(n_tokens, dtype=torch.int64, device="cuda"), S_b, K, state.p2l,
                      state.l2p, state.lcnts, L, nlp, float(cfg.router_c), constants.SWAP_MAX_MOVES, 32, out)
        blk = out.cpu().numpy()
        cap = constants.SWAP_MAX_MOVES
        rounds, err = int(blk[0]), int(blk[2])
        off_mv, off_p2l = 3 + R, 3 + R + R * cap * 4
        dev_load = blk[off_p2l + R * nlp:off_p2l + R * nlp + G]
        assert err == 0, (R, G, step, err)
        assert np.array_equal(dev_load, load_g.numpy()), f"pad-corrected loads differ R={R} G={G} step={step}"
        assert (rounds > 0) == (ref_pl is not None), f"trigger differs R={R} G={G} step={step} dev rounds {rounds}"
        if ref_pl is None:
            assert torch.equal(state.p2l.cpu().int(), state.placement.p2l.int()), "tables changed without a swap"
            stats["quiet"] += 1
            continue
        dev_p2l = torch.from_numpy(blk[off_p2l:off_p2l + R * nlp].astype(np.int32))
        assert torch.equal(dev_p2l, ref_pl.p2l.int()), f"placement differs R={R} G={G} step={step}"
        assert torch.equal(state.p2l.cpu().int(), ref_pl.p2l.int()), "device p2l table differs"
        assert torch.equal(state.l2p.cpu(), rebuild_l2p(ref_pl.p2l, G, R)), "device l2p table differs"
        mv = blk[off_mv:off_p2l].reshape(R, cap, 4)
        dev_moves = [[tuple(int(v) for v in mv[r, i]) for i in range(int(blk[3 + r]))] for r in range(R)]
        assert dev_moves == [list(m) for m in ref_moves], f"moves differ R={R} G={G} step={step}"
        state.set_placement_host(ref_pl)                  # both paths agree: carry the placement forward
        stats["swaps"] += 1
        stats["moves"] += sum(len(m) for m in ref_moves)
        stats["rounds"] = max(stats["rounds"], rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=420)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(a.seed)
    stats = dict(quiet=0, swaps=0, moves=0, rounds=0)
    per = a.cases // len(TOPOLOGIES)
    for (R, L, G) in TOPOLOGIES:
        for seq in range(max(1, per // 10)):
            run_sequence(R, L, G, 10, gen, stats)
        print(f"R={R} L={L} G={G}: ok ({stats})", flush=True)
    print(f"PASS: {stats['swaps']} swap decisions ({stats['moves']} moves, up to {stats['rounds']} rounds) and "
          f"{stats['quiet']} in-band steps identical on host and device", flush=True)


if __name__ == "__main__":
    main()
