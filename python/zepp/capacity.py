"""Exact per-step buffer demands and the capacity guarantee of the serving path.

Every symmetric buffer the ops touch is bounded by a function of two small matrices that the
dispatch op's `derive_routed_meta` builds identically on every rank: the per-source splits
`sps [W, E_virt]` and the unique counts `uc [W, W + NN]`. Its demands kernel evaluates those
functions on the device before the wire and the GEMMs launch (`demands_from_meta` is the host
reference, the same expressions), so an overflow can never be reached; `violations` names the
buffers a step would overflow and `grow` gives the capacities to resize the ops to. There is no
alternate compute path.
"""
from dataclasses import dataclass, fields

import numpy as np

from . import constants
from .routing import Capacities


@dataclass(frozen=True)
class Demands:
    recv_rows: int        # max over ranks of rows to compute (out_buf, scale_buf, combine send panel)
    dispatch_recv: int    # max over destinations of the node-union receive regions
    dispatch_stage: int   # rows staged at one gateway relay
    dispatch_relay: int   # per-round relay staging (RELAY_SLOTS slots)
    combine_conv: int     # convergence panel rows
    combine_wire: int     # wire panel rows
    pair_rows: int        # rows on one (source, destination) pair (direct strategy)


def demands_from_meta(sps, uc, W: int, L: int, gpe: int) -> Demands:
    """sps: [W, W*gpe] int, uc: [W, W + W//L] int (numpy arrays or CPU tensors)."""
    sps = np.asarray(sps, dtype=np.int64)
    uc = np.asarray(uc, dtype=np.int64)
    NN = W // L
    chunks = sps.reshape(W, W, gpe).sum(2)                 # rows from source s to destination d
    u, U = uc[:, :W], uc[:, W:]                            # unique tokens s->d, s->node n
    m_per_rank = chunks.sum(0)
    recv_rows = int(m_per_rank.max())
    if NN == 1:
        return Demands(max(recv_rows, 1), max(recv_rows, 1), 1, 1, 1, 1, max(int(chunks.max()), 1))
    node = np.arange(W) // L
    same = node[:, None] == node[None, :]
    region = np.where(same, u, U[:, node])                 # U[s, node(d)] across nodes, u[s, d] inside
    dispatch_recv = max(recv_rows, int(region.sum(0).max()))
    T = U.reshape(NN, L, NN).sum(1)                        # union rows of source node ns -> node n
    k = np.arange(L)
    cr = (T // L)[:, :, None] + (k[None, None, :] < (T % L)[:, :, None])   # chunk of relay k
    off = ~np.eye(NN, dtype=bool)
    stage = int((cr * off[:, :, None]).sum(0).max())       # sum over ns != n at (n, k)
    relay = constants.RELAY_SLOTS * int(np.where(off[:, :, None], cr, 0).max())
    C4 = chunks.reshape(NN, L, NN, L)                      # [tn, dl, n2, ls]
    conv = int(np.where(off[:, None, :], C4.sum(3), 0).sum(0).max())
    wire = int(np.where(off[:, None, :], U.reshape(NN, L, NN), 0).sum(0).max())
    return Demands(max(recv_rows, 1), max(dispatch_recv, 1), max(stage, 1), max(relay, 1),
                   max(conv, 1), max(wire, 1), max(int(chunks.max()), 1))


# demand field -> capacity field(s) it must fit in
_FIT = (("recv_rows", "recv_cap"), ("recv_rows", "combine_send"), ("dispatch_recv", "dispatch_recv"),
        ("dispatch_stage", "dispatch_stage"), ("dispatch_relay", "dispatch_relay"),
        ("combine_conv", "combine_conv"), ("combine_wire", "combine_wire"))
_FIT_DIRECT = (("recv_rows", "recv_cap"), ("pair_rows", "pair_cap"))


def violations(d: Demands, c: Capacities, direct: bool = False):
    """[(capacity field, demand, capacity)] for every buffer the step would overflow."""
    out = []
    for df, cf in (_FIT_DIRECT if direct else _FIT):
        need, have = getattr(d, df), getattr(c, cf)
        if need > have:
            out.append((cf, need, have))
    return out


def grow(c: Capacities, d: Demands, factor: float = 1.5, direct: bool = False) -> Capacities:
    """Capacities that fit `d` with headroom: per field max(current, ceil(factor * demand))."""
    new = {f.name: getattr(c, f.name) for f in fields(c)}
    for df, cf in (_FIT_DIRECT if direct else _FIT):
        new[cf] = max(new[cf], int(np.ceil(factor * getattr(d, df))))
    new["combine_send"] = max(new["combine_send"], new["recv_cap"])
    new["dispatch_recv"] = max(new["dispatch_recv"], new["recv_cap"])
    return Capacities(**new)


class CapacityExceeded(Exception):
    """A step's exact demands exceed the current capacities (raised identically on all ranks,
    before anything is launched). The serving path grows the buffers and re-plans the step."""

    def __init__(self, viol, demands: Demands):
        super().__init__("capacity exceeded: " + ", ".join(f"{f} needs {n} > {h}" for f, n, h in viol))
        self.violations = viol
        self.demands = demands
