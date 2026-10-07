"""Symmetric-heap size for the launcher (NVSHMEM reads NVSHMEM_SYMMETRIC_SIZE before init).
Usage: python -m zepp.heap --model qwen3 --budget-mib 4 --swap 0  -> prints e.g. 6G"""
import argparse
import math

from . import constants
from .config import SHAPES, tokens_per_rank


def heap_gib(shape, tokens_per_rank: int, swap: bool) -> int:
    max_row_bytes = tokens_per_rank * shape.topk * shape.chunk_bytes
    k = constants.HEAP_ROW_MULTIPLIER[bool(swap)]
    g = math.ceil((k * max_row_bytes + (1 << 30)) / (1 << 30))
    return max(constants.HEAP_MIN_GIB, min(constants.HEAP_MAX_GIB, g))


def heap_gib_from_caps(shape, S_max: int, caps, swap: bool, direct: bool, W: int, headroom: float = 3.0) -> int:
    """Symmetric-heap size for the serving path: the bytes both fused ops allocate for `caps`
    (dispatch: send + recv + stage + relay panels; combine: send + recv + conv + wire panels),
    the swap staging and the direct wires, the growable terms times `headroom`, plus 1 GiB."""
    row = shape.chunk_bytes
    fixed = 2 * S_max * shape.topk * row                                    # dispatch send + combine recv
    grow = (caps.dispatch_recv + caps.dispatch_stage + caps.dispatch_relay
            + caps.combine_send + caps.combine_conv + caps.combine_wire) * row
    if direct:
        grow += 2 * (caps.pair_cap * W * row + caps.pair_cap * W * 4)       # two panels per All2AllSingle
    if swap:
        fixed += constants.SWAP_MAX_MOVES * (shape.ffn1 * shape.hidden + shape.hidden * shape.ffn_hidden) * 2
    total = fixed + headroom * grow + (1 << 30)
    return max(constants.HEAP_MIN_GIB, math.ceil(total / (1 << 30)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SHAPES))
    ap.add_argument("--budget-mib", type=float)
    ap.add_argument("--swap", type=int, default=0)
    ap.add_argument("--config", help="serving: a calibrate.py config JSON (capacity-based sizing)")
    ap.add_argument("--headroom", type=float, default=3.0)
    a = ap.parse_args()
    if a.config:
        import json
        from .config import ModelShape
        from .routing import Capacities
        c = json.load(open(a.config))
        sh = c["shape"]
        shape = ModelShape(sh["num_experts"], sh["topk"], sh["hidden"], sh["ffn_hidden"], act=sh.get("act", "swiglu"))
        print(f"{heap_gib_from_caps(shape, c['max_tokens_per_rank'], Capacities(**c['caps']), bool(c.get('swap')), c.get('comm_strategy') == 'direct', c['ranks'], a.headroom)}G")
    else:
        shape = SHAPES[a.model]
        print(f"{heap_gib(shape, tokens_per_rank(a.budget_mib, shape), bool(a.swap))}G")
