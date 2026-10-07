"""The dispatch plan block (C.a2av_dispatch_plan: every size and offset one rank's dispatch wire reads)
against the numpy reference routing.dispatch_plan_ref, word for word, for every rank, on one GPU.

Shapes: W in {4, 8, 16, 32, 64} ranks (L = 4, so 1 to 16 nodes), S in {8, 256, 1024} tokens per rank,
K = 8, gpe in {5, 11}; random routings through virtual_route as the planner produces them, plus
adversarial ones (everything on one rank's slots, one expert, two experts); then tight capacities
must raise exactly the matching error bits.

    python tests/test_dispatch_plan_device.py [--seed 1]
"""
import argparse

import numpy as np
import torch

from zepp import constants
from zepp._ext import C
from zepp.routing import DP_ERR, dispatch_plan_ref, virtual_route

BIG = 1 << 40


def check(vce, W, L, gpe, tag, stats):
    S = vce.shape[0] // W
    K = vce.shape[1]
    vce = vce.int().cuda().contiguous()
    _, _, sps, uc = C.routed_meta(vce, W, L, gpe)
    torch.cuda.synchronize()
    sps_h, uc_h = sps.cpu().numpy(), uc.cpu().numpy()
    for rank in range(W):
        dev = C.a2av_dispatch_plan(sps, uc, W, L, gpe, rank, constants.RELAY_SLOTS, S * K, BIG, BIG, BIG)
        dev = dev.cpu().tolist()
        ref = dispatch_plan_ref(sps_h, uc_h, W, L, gpe, rank, constants.RELAY_SLOTS, S * K, BIG, BIG, BIG)
        if dev != ref:
            bad = [i for i in range(len(ref)) if dev[i] != ref[i]]
            raise AssertionError(f"{tag} rank {rank}: {len(bad)} words differ, first {bad[:6]}: "
                                 f"device {[dev[i] for i in bad[:6]]} ref {[ref[i] for i in bad[:6]]}")
        assert ref[5] == 0, (tag, rank, ref[5])
    # capacities one below the demand raise exactly their bits (identically on device and reference)
    nn = W // L
    tight = (0, 0, 0) if nn > 1 else (0, BIG, BIG)
    for rank in (0, W - 1):
        dev = C.a2av_dispatch_plan(sps, uc, W, L, gpe, rank, constants.RELAY_SLOTS, 0, *tight).cpu().tolist()
        ref = dispatch_plan_ref(sps_h, uc_h, W, L, gpe, rank, constants.RELAY_SLOTS, 0, *tight)
        assert dev[5] == ref[5], (tag, rank, dev[5], ref[5])
        want = DP_ERR["recv"] | DP_ERR["send"] | ((DP_ERR["stage"] | DP_ERR["relay"]) if nn > 1 else 0)
        if int(uc_h.sum()) > 0:
            assert dev[5] & want == want, (tag, rank, dev[5], want)
    stats["cases"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    stats = {"cases": 0}
    L = 4
    for W in (4, 8, 16, 32, 64):
        for S in (8, 256, 1024):
            for gpe in (5, 11):
                nlp = gpe - 1
                phys = torch.from_numpy(rng.integers(0, W * nlp, size=(W * S, 8), dtype=np.int64))
                check(virtual_route(phys, nlp, gpe), W, L, gpe, f"random W{W} S{S} gpe{gpe}", stats)
                phys = torch.randint(0, nlp, (W * S, 8), generator=torch.Generator().manual_seed(a.seed + S))
                check(virtual_route(phys, nlp, gpe), W, L, gpe, f"skewed W{W} S{S} gpe{gpe}", stats)
            E = W * 5
            check(torch.full((W * S, 8), 1), W, L, 5, f"one-expert W{W} S{S}", stats)
            two = torch.from_numpy(rng.integers(0, 2, size=(W * S, 8), dtype=np.int64)) * (E - 1)
            check(two, W, L, 5, f"two-experts W{W} S{S}", stats)
    print(f"test_dispatch_plan_device: {stats['cases']} cases, every rank's dispatch plan identical to the reference")


if __name__ == "__main__":
    main()
