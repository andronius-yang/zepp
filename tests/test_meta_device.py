"""The device routed-metadata chain (C.routed_meta: splits, stable scatter_index, splits_per_source,
unique_counts) against the host reference routing.meta_from_virtual_route, bitwise, and the device
demands (C.a2av_demands) against capacity.demands_from_meta, on one GPU.

Shapes: W in {4, 16, 32, 64} ranks (L = 4), S in {8, 256, 512, 1024} tokens per rank, K = 8 (and 6:
tiles of whole tokens that do not divide the tile), gpe in {5, 11}; random physical routings through
virtual_route as the planner produces them, plus adversarial routings (one expert, two experts,
duplicate picks inside a token).

    python tests/test_meta_device.py [--seed 1]
"""
import argparse

import numpy as np
import torch

from zepp import constants
from zepp._ext import C
from zepp.capacity import Demands, demands_from_meta
from zepp.routing import meta_from_virtual_route, virtual_route


def check(vce, W, L, gpe, tag, stats):
    S = vce.shape[0] // W
    vce = vce.int().cuda().contiguous()
    ref = meta_from_virtual_route(vce, W, S, gpe, W // L, L)
    dev = C.routed_meta(vce, W, L, gpe)
    torch.cuda.synchronize()
    for name, a, b in zip(("splits", "scatter_index", "sps", "uc"), dev, ref):
        a, b = a.cpu(), b.cpu()
        assert a.shape == b.shape, (tag, name, a.shape, b.shape)
        if not torch.equal(a, b):
            bad = (a != b).nonzero()
            raise AssertionError(f"{tag}: {name} differs at {bad.numel()} entries, first {bad[:5].tolist()}: "
                                 f"device {a.flatten()[bad[:5, 0] if a.dim() == 1 else 0].tolist()}")
    sps, uc = dev[2], dev[3]
    dh = demands_from_meta(sps.cpu().numpy(), uc.cpu().numpy(), W, L, gpe)
    loose = [1 << 40] * 8
    dd = C.a2av_demands(sps, uc, W, L, gpe, constants.RELAY_SLOTS, False, loose).cpu().tolist()
    assert Demands(*dd[:7]) == dh, (tag, dd, dh)
    assert dd[7] == 0, (tag, dd)
    # every capacity one short of its demand -> every bit of the mask (direct: bits 0 and 7)
    tight = [dh.recv_rows - 1, dh.dispatch_recv - 1, dh.dispatch_stage - 1, dh.dispatch_relay - 1,
             dh.recv_rows - 1, dh.combine_conv - 1, dh.combine_wire - 1, dh.pair_rows - 1]
    dt = C.a2av_demands(sps, uc, W, L, gpe, constants.RELAY_SLOTS, False, tight).cpu().tolist()
    assert dt[7] == 0x7F, (tag, dt)
    dx = C.a2av_demands(sps, uc, W, L, gpe, constants.RELAY_SLOTS, True, tight).cpu().tolist()
    assert dx[7] == 0x81, (tag, dx)
    stats["cases"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    stats = {"cases": 0}
    for W in (4, 16, 32, 64):
        L = 4
        for S in (8, 256, 512, 1024):
            for gpe in (5, 11):
                nlp = gpe - 1
                for K in (8, 6):
                    if K == 6 and S != 512:
                        continue
                    phys = torch.from_numpy(rng.integers(0, W * nlp, size=(W * S, K), dtype=np.int64))
                    check(virtual_route(phys, nlp, gpe), W, L, gpe, f"random W{W} S{S} K{K} gpe{gpe}", stats)
                # skewed: everything on rank 0's slots, with duplicate picks inside tokens
                phys = torch.randint(0, nlp, (W * S, 8), generator=torch.Generator().manual_seed(a.seed + S))
                check(virtual_route(phys, nlp, gpe), W, L, gpe, f"skewed W{W} S{S} gpe{gpe}", stats)
            # one expert, two experts
            E = W * gpe
            check(torch.full((W * S, 8), 1), W, L, gpe, f"one-expert W{W} S{S}", stats)
            two = torch.from_numpy(rng.integers(0, 2, size=(W * S, 8), dtype=np.int64)) * (E - 1)
            check(two, W, L, gpe, f"two-experts W{W} S{S}", stats)
    print(f"test_meta_device: {stats['cases']} cases bitwise identical (meta + demands)")


if __name__ == "__main__":
    main()
