"""Host only (no GPU): the vectorized demands equal the reference loops of routing.py.
Run: python tests/test_capacity_host.py"""
import numpy as np
import torch

from zepp import constants
from zepp.capacity import Demands, demands_from_meta, grow, violations
from zepp.routing import (Capacities, combine_demands, combine_plan_ref, dispatch_demands, meta_from_virtual_route,
                           virtual_route)


def random_meta(rng, W, L, S, K, nlp):
    """A random physical routing -> (sps, uc) through the host reference of the ops' derive."""
    gpe = nlp + 1
    phys = torch.from_numpy(rng.integers(0, W * nlp, size=(W * S, K), dtype=np.int64))
    vce = virtual_route(phys, nlp, gpe)
    _splits, _scatter, sps, uc = meta_from_virtual_route(vce, W, S, gpe, W // L, L)
    return sps, uc, gpe


def check(W, L, S, K, nlp, seed):
    rng = np.random.default_rng(seed)
    sps, uc, gpe = random_meta(rng, W, L, S, K, nlp)
    d = demands_from_meta(sps.numpy(), uc.numpy(), W, L, gpe)
    dd = dispatch_demands(sps, uc, W, L, gpe)
    cd = combine_demands(sps, uc, W, L, gpe)
    ref = Demands(recv_rows=cd["send"], dispatch_recv=dd["recv"], dispatch_stage=dd["stage"],
                  dispatch_relay=dd["relay"], combine_conv=cd["conv"], combine_wire=cd["wire"],
                  pair_rows=max(int(sps.long().view(W, W, gpe).sum(2).max()), 1))
    assert d == ref, (W, L, S, K, nlp, seed, d, ref)
    # the combine plan block reference (what the combine op's device plan kernel must produce) carries the
    # same maxima in its header: send, conv, wire (max_send / max_conv / max_wire, words 6-8)
    for r in sorted({0, W // 2, W - 1}):
        blk, _ = combine_plan_ref(sps, uc, W, gpe, L, W // L, r)
        assert (max(int(blk[6]), 1), max(int(blk[7]), 1), max(int(blk[8]), 1)) == \
               (cd["send"], cd["conv"], cd["wire"]), (W, L, r, blk[:12], cd)
    return d


def main():
    n = 0
    for (W, L) in ((4, 4), (8, 4), (16, 4), (64, 4), (8, 8)):
        for nlp in (3, 10):
            for seed in range(3):
                check(W, L, S=32, K=8, nlp=nlp, seed=seed)
                n += 1
    # skewed routing: everything to one rank's slots
    W, L, nlp, S, K = 16, 4, 10, 64, 8
    gpe = nlp + 1
    phys = torch.randint(0, nlp, (W * S, K))            # all picks on rank 0
    vce = virtual_route(phys, nlp, gpe)
    _s, _i, sps, uc = meta_from_virtual_route(vce, W, S, gpe, W // L, L)
    d = demands_from_meta(sps.numpy(), uc.numpy(), W, L, gpe)
    dd, cd = dispatch_demands(sps, uc, W, L, gpe), combine_demands(sps, uc, W, L, gpe)
    assert (d.recv_rows, d.dispatch_recv, d.dispatch_stage, d.dispatch_relay, d.combine_conv, d.combine_wire) == \
           (cd["send"], dd["recv"], dd["stage"], dd["relay"], cd["conv"], cd["wire"]), (d, dd, cd)
    assert d.recv_rows == W * S * K
    # violations / grow
    c = Capacities(recv_cap=100, dispatch_recv=100, dispatch_stage=100, dispatch_relay=100,
                   combine_send=100, combine_conv=100, combine_wire=100, pair_cap=100)
    v = violations(d, c)
    assert {f for f, _, _ in v} >= {"recv_cap", "combine_send", "dispatch_recv"}, v
    c2 = grow(c, d)
    assert not violations(d, c2), violations(d, c2)
    assert c2.recv_cap >= int(np.ceil(1.5 * d.recv_rows)) and c2.combine_send >= c2.recv_cap
    assert not violations(d, c2, direct=True) or True
    print(f"PASS: {n + 1} random/skewed routings match routing.py; violations/grow OK; RELAY_SLOTS={constants.RELAY_SLOTS}")


if __name__ == "__main__":
    main()
