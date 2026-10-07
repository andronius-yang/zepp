"""Band-triggered intra-node expert swap, scheduled across the two GEMMs.

Decision (host, per iteration, from the gathered loads d[R, G] and the current placement):
  reference load  L_r = sum over hosted slots of D_e // c_e (the fair share the router is
                  bound to within (1 +- C)); a node is OUT of band iff
                  max_r L_r > (1 + C) * mean_r L_r over its ranks (paper eq. 3);
  orbit           only out-of-band nodes act: pair heaviest <-> lightest rank, exchange the
                  one (slot, slot) pair that most reduces the pair maximum, repeat until the
                  node is back in band (cap: SWAP_MAX_MOVES slots per rank);
  moves           the composed permutation as per-rank pull lists (dst slot, src rank, src
                  slot) -- every expert has at most one instance per node, so the source is
                  unique.

Execution (NVLink, never inter-node): the pushes of matrix W1 start with the dispatch GEMM
and the pushes of W2 with the combine GEMM (the ops write a GEMM-start mark on their stream
that the movement streams wait on with a zero-SM stream memop). Each rank copies its outgoing
slots into the destination's staging and stores a landed signal; then waits its own incoming
signals, copies staging -> slot and raises the slot's readiness signal. The fused GEMMs gate
the swapped-in slots' tiles on those signals, so the swap benefit lands in the same iteration.
"""
import ctypes

import numpy as np
import torch

from . import constants
from ._ext import C
from .placement import Placement, rebuild_l2p

CU_STREAM_WAIT_VALUE_GEQ = 0x0
CU_STREAM_WAIT_VALUE_FLUSH = 0x4


# ------------------------------------------------------------------ decision (host)

def rank_loads(load_g, p2l, lcnts, R, nlp):
    p2l_l = p2l.long()
    valid = p2l_l >= 0
    e = p2l_l.clamp(min=0)
    share = torch.where(valid, load_g.long()[e] // lcnts.long().clamp(min=1)[e], torch.zeros_like(e))
    return share.view(R, nlp).sum(1)


def node_band(L_r, L, C):
    """(out_of_band bool [NN], max/mean ratio [NN]) of the per-node reference loads."""
    lr = np.asarray(L_r, dtype=np.int64).reshape(-1, L)
    tot = lr.sum(1)
    mx = lr.max(1)
    out = mx.astype(np.float64) * L > (1.0 + C) * tot.astype(np.float64)
    ratio = np.where(tot > 0, mx * L / np.maximum(tot, 1), 1.0)
    return out, ratio


def _swap_round(p2l_np, lg, lc, L, nlp, w_slot, L_r, per_pair, C):
    """One greedy round: for each out-of-band node pair heaviest<->lightest ranks and pick
    the exchange that most reduces the pair max (tie: smallest expert ids). Returns
    [(r_h, s_h, e_h, r_l, s_l, e_l)] with global slot ids."""
    R = p2l_np.shape[0] // nlp
    NN = R // L
    G = lg.shape[0]
    lr = L_r.reshape(NN, L)
    key = (-lr) * L + np.arange(L)[None, :]
    order = np.argsort(key, axis=1, kind="stable")
    P = L // 2
    H = (np.arange(NN)[:, None] * L + order[:, :P]).reshape(-1)
    Lo = (np.arange(NN)[:, None] * L + order[:, L - 1:L - 1 - P:-1]).reshape(-1)
    Q = H.shape[0]
    lrH, lrL = L_r[H], L_r[Lo]
    pair_ok = (lrH - lrL > 1)
    node_out, _ = node_band(L_r, L, C)
    pair_ok &= node_out[H // L]
    hs = H[:, None] * nlp + np.arange(nlp)[None, :]
    ls = Lo[:, None] * nlp + np.arange(nlp)[None, :]
    e_hs, e_ls = p2l_np[hs], p2l_np[ls]
    v_hs, v_ls = e_hs >= 0, e_ls >= 0
    w_hs, w_ls = w_slot[hs], w_slot[ls]
    BIG = np.int64(1) << 60
    k_hs = np.lexsort((np.where(v_hs, e_hs, BIG), np.where(v_hs, -w_hs, BIG)), axis=1)
    k_ls = np.lexsort((np.where(v_ls, e_ls, BIG), np.where(v_ls, w_ls, BIG)), axis=1)
    K = min(8, nlp)
    qi = np.arange(Q)[:, None]
    hs8, ls8 = hs[qi, k_hs[:, :K]], ls[qi, k_ls[:, :K]]
    eh8, el8 = p2l_np[hs8], p2l_np[ls8]
    vh8, vl8 = eh8 >= 0, el8 >= 0
    wh8, wl8 = w_slot[hs8], w_slot[ls8]
    eh_in_l = (eh8[:, :, None] == e_ls[:, None, :]).any(-1)
    el_in_h = (el8[:, :, None] == e_hs[:, None, :]).any(-1)
    new_max = np.maximum(lrH[:, None, None] - wh8[:, :, None] + wl8[:, None, :],
                         lrL[:, None, None] - wl8[:, None, :] + wh8[:, :, None])
    gain = np.maximum(lrH, lrL)[:, None, None] - new_max
    ok = (vh8[:, :, None] & vl8[:, None, :] & ~eh_in_l[:, :, None] & ~el_in_h[:, None, :]
          & (eh8[:, :, None] != el8[:, None, :]) & (wh8[:, :, None] > wl8[:, None, :]) & pair_ok[:, None, None])
    ok &= gain >= 1
    tie = eh8[:, :, None] * G + el8[:, None, :]
    sel = np.where(ok, gain * (G * G + 1) - tie, -BIG)
    flat = sel.reshape(Q, -1)
    best = flat.argmax(1)
    has = flat[np.arange(Q), best] > -BIG
    a, b = best // K, best % K
    swaps = []
    for q in np.nonzero(has)[0]:
        swaps.append((int(H[q]), int(hs8[q, a[q]]), int(eh8[q, a[q]]), int(Lo[q]), int(ls8[q, b[q]]), int(el8[q, b[q]])))
    lrH, lrL = lrH.copy(), lrL.copy()
    L_w = L_r.copy()
    for _m in range(1, per_pair):
        if not has.any():
            break
        qs = np.nonzero(has)[0]
        d = wh8[qs, a[qs]] - wl8[qs, b[qs]]
        lrH[qs] -= d
        lrL[qs] += d
        ok[qs, a[qs], :] = False
        ok[qs, :, b[qs]] = False
        L_w[H[qs]] = lrH[qs]
        L_w[Lo[qs]] = lrL[qs]
        node_out, _ = node_band(L_w, L, C)
        ok &= node_out[H // L][:, None, None]
        new_max = np.maximum(lrH[:, None, None] - wh8[:, :, None] + wl8[:, None, :],
                             lrL[:, None, None] - wl8[:, None, :] + wh8[:, :, None])
        gain = np.maximum(lrH, lrL)[:, None, None] - new_max
        sel = np.where(ok & (gain >= 1), gain * (G * G + 1) - tie, -BIG)
        flat = sel.reshape(Q, -1)
        best = flat.argmax(1)
        has = flat[np.arange(Q), best] > -BIG
        a, b = best // K, best % K
        for q in np.nonzero(has)[0]:
            swaps.append((int(H[q]), int(hs8[q, a[q]]), int(eh8[q, a[q]]), int(Lo[q]), int(ls8[q, b[q]]), int(el8[q, b[q]])))
    return swaps


def decide_swaps(load_g, placement: Placement, L, nlp, C,
                 cap=constants.SWAP_MAX_MOVES, per_pair=constants.SWAP_PAIR_MOVES, max_rounds=32):
    """Band-triggered orbit to the shortest swap sequence that puts every node back in band.
    Returns (new_placement or None, rounds). The band test costs microseconds; the orbit runs
    only for out-of-band nodes."""
    R = placement.p2l.numel() // nlp
    a0 = np.asarray(placement.p2l.numpy(), dtype=np.int64)
    lg = np.asarray(load_g.numpy(), dtype=np.int64)
    lc = np.maximum(np.asarray(placement.lcnts.numpy(), dtype=np.int64), 1)
    valid = a0 >= 0
    e_all = np.where(valid, a0, 0)
    w_slot = np.where(valid, lg[e_all] // lc[e_all], 0)
    L_r = w_slot.reshape(R, nlp).sum(1)
    if not node_band(L_r, L, C)[0].any():
        return None, 0
    cur = a0.copy()
    rounds = 0
    for _ in range(max_rounds):
        swaps = _swap_round(cur, lg, lc, L, nlp, w_slot, L_r, per_pair, C)
        if not swaps:
            break
        nxt = cur.copy()
        for (_rh, sh, eh, _rl, sl, el) in swaps:
            nxt[sh] = el
            nxt[sl] = eh
        if int((nxt != a0).reshape(R, nlp).sum(1).max()) > cap:
            if per_pair > 1:
                swaps = _swap_round(cur, lg, lc, L, nlp, w_slot, L_r, 1, C)
                if not swaps:
                    break
                nxt = cur.copy()
                for (_rh, sh, eh, _rl, sl, el) in swaps:
                    nxt[sh] = el
                    nxt[sl] = eh
                if int((nxt != a0).reshape(R, nlp).sum(1).max()) > cap:
                    break
            else:
                break
        cur = nxt
        rounds += 1
        for (rh, sh, _eh, rl, sl, _el) in swaps:
            for sx in (sh, sl):
                e = cur[sx]
                w_new = lg[e] // lc[e] if e >= 0 else 0
                L_r[sx // nlp] += w_new - w_slot[sx]
                w_slot[sx] = w_new
    if rounds == 0:
        return None, 0
    p2l_f = torch.from_numpy(cur.astype(np.int32))
    return Placement(p2l_f, rebuild_l2p(p2l_f, lg.shape[0], R), placement.lcnts, placement.stats), rounds


def net_moves(p2l0, p2l_f, L, nlp):
    """Composed intra-node permutation as per-rank pull lists:
    moves[r] = sorted [(dst_slot_local, src_rank, src_slot_local, expert)]."""
    R = p2l0.numel() // nlp
    a0, af = p2l0.tolist(), p2l_f.tolist()
    loc0 = {}
    for phys, e in enumerate(a0):
        if e >= 0:
            r = phys // nlp
            loc0[(r // L, e)] = (r, phys % nlp)
    moves = [[] for _ in range(R)]
    for phys, e in enumerate(af):
        if e >= 0 and a0[phys] != e:
            r = phys // nlp
            sr, ss = loc0[(r // L, e)]
            moves[r].append((phys % nlp, sr, ss, e))
    for m in moves:
        m.sort()
    return moves


def swap_orbit(load_g, placement: Placement, L, nlp, C, max_rounds=8):
    """Setup-time fixed-point iteration of the runtime swap sequence on a fixed demand: the
    list of successive placements (sizing envelope for swap-enabled layers)."""
    out = []
    cur = placement
    seen = {bytes(cur.p2l.numpy().tobytes())}
    for _ in range(max_rounds):
        nxt, rounds = decide_swaps(load_g, cur, L, nlp, C, max_rounds=1)
        if nxt is None:
            break
        key = bytes(nxt.p2l.numpy().tobytes())
        if key in seen:
            break
        seen.add(key)
        out.append(nxt)
        cur = nxt
    return out


# ------------------------------------------------------------------ weights + movement (device)

def _libcuda():
    lib = ctypes.CDLL("libcuda.so.1")
    fn = getattr(lib, "cuStreamWaitValue64_v2", None) or lib.cuStreamWaitValue64
    fn.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_uint]
    fn.restype = ctypes.c_int
    wr = getattr(lib, "cuStreamWriteValue64_v2", None) or lib.cuStreamWriteValue64
    wr.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_uint]
    wr.restype = ctypes.c_int
    return fn, wr


class WeightSlots:
    """Expert weights in per-rank slot storage with a per-slot readiness signal: the fused
    GEMMs read the slots directly and gate a swapped slot's tiles on its signal. Slot 0 is the
    empty pad group. w1 [gpe, ffn1, H] (ffn1 = 2 * ffn with SwiGLU), w2 [gpe, H, ffn]."""

    def __init__(self, rank, cfg, dtype, device="cuda", w1=None, w2=None):
        """w1 / w2: existing slot storage to adopt (e.g. a serving framework's parameters with
        the pad group at index 0); allocated zero when not given."""
        shape = cfg.shape
        self.rank = rank
        self.nlp = cfg.slots_per_rank
        self.gpe = cfg.groups_per_rank
        self.ffn, self.ffn1, self.H, self.dtype = shape.ffn_hidden, shape.ffn1, shape.hidden, dtype
        if w1 is None:
            w1 = torch.zeros(self.gpe, self.ffn1, self.H, dtype=dtype, device=device)
        if w2 is None:
            w2 = torch.zeros(self.gpe, self.H, self.ffn, dtype=dtype, device=device)
        assert w1.shape == (self.gpe, self.ffn1, self.H) and w1.is_contiguous() and w1.dtype == dtype, tuple(w1.shape)
        assert w2.shape == (self.gpe, self.H, self.ffn) and w2.is_contiguous() and w2.dtype == dtype, tuple(w2.shape)
        self.w1, self.w2 = w1, w2
        self.sig1 = torch.zeros(self.gpe, dtype=torch.int64, device=w1.device)
        self.sig2 = torch.zeros(self.gpe, dtype=torch.int64, device=w1.device)

    def fill(self, p2l_host, w1_of, w2_of):
        for j in range(self.nlp):
            e = int(p2l_host[self.rank * self.nlp + j])
            if e < 0:
                self.w1[1 + j].zero_()
                self.w2[1 + j].zero_()
                continue                        # unassigned redundant slot
            self.w1[1 + j].copy_(w1_of(e).to(self.dtype))
            self.w2[1 + j].copy_(w2_of(e).to(self.dtype))
        torch.cuda.synchronize()

    def gemm_weights(self):
        return self.w1, self.w2

    def slots(self, k):
        return self.w1 if k == 0 else self.w2

    def signals(self, k):
        return self.sig1 if k == 0 else self.sig2


class SwapLane:
    """Executes the composed intra-node permutation. Two lane modes (`mode`):
    "overlap" (benchmark): SWAP_STREAMS side streams, the W1 phase gated on the dispatch GEMM's start
    mark and the W2 phase on the combine GEMM's; "staged" (serving, enable_device): device kernels read
    the swap decision's result block, the sender pushes into the destination's staging and raises the
    destination's per-slot GATE word, the receiver's GEMM reads the moved expert from the staging (weight
    pointer override, moved-last order, tile gates spin only until the push lands) and the staging -> slot
    commit runs after the GEMM."""

    def __init__(self, shape, dtype, rank, L, nlp, pg, cap=constants.SWAP_MAX_MOVES,
                 n_streams=constants.SWAP_STREAMS):
        """One lane per process: the staging (symmetric), the epoch and the movement streams
        are shared by every layer; `bind` selects the layer's slots/signals before `prepare`."""
        self.slots = None
        self.rank = rank
        self.L = L
        self.nlp = nlp
        self.cap = cap
        self.ffn, self.ffn1, self.H = shape.ffn_hidden, shape.ffn1, shape.hidden
        self.local_rank = rank % L
        self.epoch = 0
        # device step state (core/step_state.h): the device lane's kernels and gate kwargs read the step's epoch from
        # the step slot (-1 = "use the slot"); the host epoch stays as its shadow
        self.step_device = False
        self.w_stream = torch.cuda.Stream()
        self._xstreams = [torch.cuda.Stream() for _ in range(n_streams - 1)]
        self._ev_push = [torch.cuda.Event() for _ in range(n_streams)]
        self._ev_join = [torch.cuda.Event() for _ in range(n_streams)]
        self.ev_done = torch.cuda.Event()
        self.ev_pre = torch.cuda.Event()
        self._ev_ps = [torch.cuda.Event(enable_timing=True) for _ in (0, 1)]
        self._ev_pe = [torch.cuda.Event(enable_timing=True) for _ in (0, 1)]
        self._idx_all = torch.arange(1, nlp + 1, device="cuda")
        self._slot_idx = [torch.tensor([1 + j], device="cuda") for j in range(nlp)]
        self.move_bytes_this_iter = 0
        self._issued = False
        self._n_issued = 0
        self._phase_done = [False, False]
        self._in, self._out = [], []
        self._w_waited = False
        self._stag_w1_all = C.create_tensor_list([cap * self.ffn1, self.H], dtype, pg)
        self._stag_w2_all = C.create_tensor_list([cap * self.H, self.ffn], dtype, pg)
        self._xsig_all = C.create_tensor_list([2 * cap], torch.int64, pg, False, True)
        assert len(self._stag_w1_all) == L, f"node-local view count {len(self._stag_w1_all)} != {L}"
        self._xsig = self._xsig_all[self.local_rank]
        # staged lane: peer-writable per-slot gate words (matrix k at [k*gpe, (k+1)*gpe)), the per-expert
        # weight pointer override tables the GEMMs read, the staging addresses they are pointed at, and
        # the moved-last schedule encoding (pinned mirror, parity double-buffered, one async H2D per use)
        self.mode = "overlap"
        self.gpe = nlp + 1
        self._gate_all = C.create_tensor_list([2 * self.gpe], torch.int64, pg, False, True)
        self._gate = self._gate_all[self.local_rank]
        self._ovr = [torch.zeros(self.gpe, dtype=torch.int64, device="cuda") for _ in (0, 1)]
        self._stag_addr = [[self._stag(k, self.local_rank, i).data_ptr() for i in range(cap)] for k in (0, 1)]
        # [gpe] schedule encoding, then (deferred verdict) the front-class count
        self._sched_dev = torch.empty(self.gpe + 1, dtype=torch.int32, device="cuda")
        self._dev = False                              # device lane (enable_device): arm / push / commit kernels
        self._keep_pin = [torch.empty(nlp, dtype=torch.int64, pin_memory=True) for _ in (0, 1)]
        self._keep_dev = torch.empty(nlp, dtype=torch.int64, device="cuda")
        self._keep_par = 0
        self._cu_wait, self._cu_write = _libcuda()
        self._wait_flags = CU_STREAM_WAIT_VALUE_GEQ | CU_STREAM_WAIT_VALUE_FLUSH
        if self._cu_wait(torch.cuda.current_stream().cuda_stream, self._xsig.data_ptr(), 0, self._wait_flags) != 0:
            self._wait_flags = CU_STREAM_WAIT_VALUE_GEQ
        self.ev_pre.record(torch.cuda.current_stream())
        cs = torch.cuda.current_stream().cuda_stream
        peer_probe = self._xsig_all[(self.local_rank + 1) % L]
        self._write_ok = (self._cu_write(cs, self._xsig.data_ptr(), 0, 0) == 0
                          and self._cu_write(cs, peer_probe.data_ptr(), 0, 0) == 0)
        torch.cuda.synchronize()
        self._dispatch_op = self._combine_op = None
        self._mark_dispatch = self._mark_combine = None

    def warmup(self, slots: WeightSlots):
        """Run every primitive of a phase once, to self, while the device is idle: with lazy module
        loading a kernel's first launch loads its module, and a load issued while a persistent gated
        GEMM occupies the device never completes (the same reason the combine op preloads its kernels).
        Contents of slot 1 are copied out and back unchanged; the signals are left at zero."""
        word = torch.zeros(2, dtype=torch.int64, device="cuda")
        if self.mode == "staged":                     # every primitive of the staged lane, on the forward stream
            st = torch.cuda.current_stream()
            for k in (0, 1):
                src = slots.slots(k)[1]
                self._stag(k, self.local_rank, 0).copy_(src)
                if self._write_ok:
                    self._cu_write(st.cuda_stream, word.data_ptr(), 1, 0)
                else:
                    word[0:1].fill_(1)
                C.stream_wait_geq(word, 0, 1, st.cuda_stream)
                src.copy_(self._stag(k, self.local_rank, 0))
            torch.cuda.synchronize()
            return
        streams = [self.w_stream] + self._xstreams
        for st in streams:
            st.wait_event(self.ev_pre)
        for k in (0, 1):
            src = slots.slots(k)[1]
            for i, st in enumerate(streams):
                with torch.cuda.stream(st):
                    self._stag(k, self.local_rank, 0).copy_(src)          # push path (to own staging)
                    self._cu_write(st.cuda_stream, word.data_ptr(), 1 + i, 0)
                    self._wait_geq(st, word, 0, 1)
                    src.copy_(self._stag(k, self.local_rank, 0))          # pull path
                    self._ev_push[i].record(st)
            for st in streams:
                for e in self._ev_push:
                    st.wait_event(e)
        with torch.cuda.stream(self.w_stream):
            self.ev_done.record()
        torch.cuda.synchronize()

    def warmup_swap_path(self):
        """Launch once, on scratch tensors, the torch fill kernels behind the lanes' host-issued word writes
        (index fills of a gate-word or slot-signal table, the single-word fill that replaces a stream memory
        write where those are unavailable), so no swap step loads code beside a spinning GEMM. Touches no
        lane state; collective-free."""
        idx = torch.ones(1, dtype=torch.int64, device="cuda")
        gate = torch.zeros(2, self.gpe, dtype=torch.int64, device="cuda")
        gate.index_fill_(1, idx, 1)                    # gate-word table [2, gpe]
        sig = torch.zeros(self.gpe, dtype=torch.int64, device="cuda")
        sig.index_fill_(0, idx, 1)                     # overlap lane: slot signals
        sig[0:1].fill_(1)                              # single-word writes without stream memops
        torch.cuda.synchronize()

    def dry_gate_kwargs(self):
        """Weight-gate kwargs for both GEMMs that exercise the gated GEMM path of a swap step without a
        swap: one slot (the last) is scheduled in the deferred class, every gate word is already raised,
        and no weight pointer is overridden, so the GEMMs compute exactly what an ungated step does."""
        if getattr(self, "_dry", None) is None:
            gpe = self.gpe
            word = torch.ones(gpe, dtype=torch.int64, device="cuda")
            ovr = [torch.zeros(gpe, dtype=torch.int64, device="cuda") for _ in (0, 1)]
            order = list(range(gpe - 1)) + [1 << 30]
            sched = torch.tensor(order, dtype=torch.int32, device="cuda")
            gate_of = [-1] * gpe
            gate_of[gpe - 1] = gpe - 2                 # index into word[1:]
            self._dry = (dict(weight_signal=word[1:], weight_signal_epoch=1, weight_gate_group_start=1,
                              weight_ptr_override=ovr[0], sched_expert_order=sched, sched_n_front=gpe - 1),
                         dict(weight_signal=word[1:], weight_signal_epoch=1, gate_of_expert=gate_of,
                              weight_ptr_override=ovr[1]))
        return self._dry

    def snapshot(self):
        """Host counters and device words of the lane (warm-up: restored afterwards). The device words
        are this rank's own gate / exchange words, the override tables and the push counters."""
        host = {k: getattr(self, k) for k in (
            "epoch", "slots", "_keep_par", "_in", "_out", "_phase_done", "_issued", "_n_issued",
            "_w_waited", "move_bytes_this_iter")}
        snap = dict(host={k: (list(v) if isinstance(v, list) else v) for k, v in host.items()})
        dev = dict(gate=self._gate, xsig=self._xsig, ovr0=self._ovr[0], ovr1=self._ovr[1], sched=self._sched_dev)
        if self._dev:
            dev["push_ctr"] = self._push_ctr
            snap["host"].update(_blk=self._blk, _dev_rounds=self._dev_rounds)
        snap["dev"] = {k: (t, t.clone()) for k, t in dev.items()}
        return snap

    def restore(self, snap):
        """Undo a snapshot's changes. Every rank must be quiesced (device synchronized and past a barrier):
        peers write this rank's gate and exchange words."""
        for k, v in snap["host"].items():
            setattr(self, k, v)
        for t, saved in snap["dev"].values():
            t.copy_(saved)

    def bind(self, slots: WeightSlots):
        """Select the layer whose slots the next prepare/phases move."""
        assert slots.ffn1 == self.ffn1 and slots.ffn == self.ffn and slots.H == self.H
        self.slots = slots

    def attach_ops(self, dispatch_op, combine_op):
        self._dispatch_op, self._combine_op = dispatch_op, combine_op
        self._mark_dispatch = dispatch_op.gemm_start_mark()
        self._mark_combine = combine_op.gemm_start_mark()

    def detach_ops(self):
        """Drop the op references (their start-mark tensors die with the ops)."""
        self._dispatch_op = self._combine_op = None
        self._mark_dispatch = self._mark_combine = None

    def _stag(self, mat, local_rank, idx):
        if mat == 0:
            return self._stag_w1_all[local_rank].view(self.cap, self.ffn1, self.H)[idx]
        return self._stag_w2_all[local_rank].view(self.cap, self.H, self.ffn)[idx]

    def prepare(self, all_moves):
        """Arm one iteration's exchange from the replicated move lists (may be empty)."""
        self._in = list(all_moves[self.rank])
        assert len(self._in) <= self.cap, f"incoming {len(self._in)} > staging cap {self.cap}"
        self._out = []
        node = self.rank // self.L
        for r in range(node * self.L, (node + 1) * self.L):     # pulls only happen within the node
            for idx, (dj, sr, ss, _e) in enumerate(all_moves[r]):
                if sr == self.rank:
                    self._out.append((ss, r, idx, dj))
        self.move_bytes_this_iter = 0
        self._issued = False
        self._n_issued = 0
        self._phase_done = [False, False]
        self._w_waited = False
        self.epoch += 1
        changed = {dj for (dj, _sr, _ss, _e) in self._in}
        self.ev_pre.record(torch.cuda.current_stream())        # the side-stream phases start after this point
        keep = self._keep_index(changed) if changed else self._idx_all
        self.slots.sig1.index_fill_(0, keep, self.epoch)
        self.slots.sig2.index_fill_(0, keep, self.epoch)

    def _keep_index(self, changed):
        """Device index of the slots not changed this step (1-based), uploaded from a pinned buffer."""
        keep = [1 + j for j in range(self.nlp) if j not in changed]
        pin = self._keep_pin[self._keep_par]
        self._keep_par ^= 1
        n = len(keep)
        pin.numpy()[:n] = keep
        self._keep_dev[:n].copy_(pin[:n], non_blocking=True)
        return self._keep_dev[:n]

    def active(self):
        return bool(self._in or self._out)

    def _wait_geq(self, st, word, index, value):
        """Block stream `st` until word[index] >= value (a stream memory operation, zero SMs)."""
        rc = self._cu_wait(st.cuda_stream, word.data_ptr() + 8 * index, value, self._wait_flags)
        assert rc == 0, f"cuStreamWaitValue64 rc={rc}"

    def _wait_mark(self, streams, mark):
        for st in streams:
            self._wait_geq(st, mark, 0, self.epoch)

    def _phase(self, k, pre_mark):
        assert not self._phase_done[k]
        streams = [self.w_stream] + self._xstreams
        N = len(streams)
        if not self._w_waited:
            for st in streams:
                st.wait_event(self.ev_pre)
            self._w_waited = True
        self._wait_mark(streams, pre_mark)
        with torch.cuda.stream(self.w_stream):
            self._ev_ps[k].record()
        slots = self.slots.slots(k)
        sigs = self.slots.signals(k)
        for i, (ss, dr, idx, _dj) in enumerate(self._out):     # push outgoing
            st = streams[i % N]
            with torch.cuda.stream(st):
                dst_local = dr % self.L
                self._stag(k, dst_local, idx).copy_(slots[1 + ss])
                peer_sig = self._xsig_all[dst_local]
                if self._write_ok:
                    rc = self._cu_write(st.cuda_stream, peer_sig.data_ptr() + 8 * (2 * idx + k), self.epoch, 0)
                    assert rc == 0, f"cuStreamWriteValue64 rc={rc}"
                else:
                    peer_sig[2 * idx + k:2 * idx + k + 1].fill_(self.epoch)
        for si, st in enumerate(streams):                      # all pushes precede all pulls
            self._ev_push[si].record(st)
        for st in streams:
            for si in range(N):
                st.wait_event(self._ev_push[si])
        for idx, (dj, _sr, _ss, _e) in enumerate(self._in):   # pull incoming, raise gates
            st = streams[idx % N]
            with torch.cuda.stream(st):
                self._wait_geq(st, self._xsig, 2 * idx + k, self.epoch)
                slot = slots[1 + dj]
                slot.copy_(self._stag(k, self.local_rank, idx))
                if self._write_ok:
                    rc = self._cu_write(st.cuda_stream, sigs.data_ptr() + 8 * (1 + dj), self.epoch, 0)
                    assert rc == 0, f"cuStreamWriteValue64 rc={rc}"
                else:
                    sigs.index_fill_(0, self._slot_idx[dj], self.epoch)
                self.move_bytes_this_iter += 2 * slot.numel() * slot.element_size()
        for si, st in enumerate(streams[1:], start=1):
            self._ev_join[si].record(st)
            self.w_stream.wait_event(self._ev_join[si])
        with torch.cuda.stream(self.w_stream):
            self._ev_pe[k].record()
        self._phase_done[k] = True
        self._n_issued += 1
        if self._n_issued == 2:
            with torch.cuda.stream(self.w_stream):
                self.ev_done.record()
            self._issued = True

    # -- staged lane on the device (serving) ----------------------------------------------
    def enable_device(self, R):
        """Arming, pushes and commits become device kernels (lane_device.cu) that read the swap_decide
        result block on the device; the host reads only this rank's pulls from the block it already has
        after the step's planning sync (for the GEMM kwargs)."""
        assert self.mode == "staged", "the device lane implements the staged mode"
        esz = self._stag_w1_all[0].element_size()
        self._dev = True
        self._R = R
        self._slot_bytes = [self.ffn1 * self.H * esz, self.H * self.ffn * esz]
        self._stag_addr_dev = torch.tensor(self._stag_addr, dtype=torch.int64, device="cuda")      # [2, cap]
        self._peer_stag = [torch.tensor([t.data_ptr() for t in views], dtype=torch.int64, device="cuda")
                           for views in (self._stag_w1_all, self._stag_w2_all)]
        self._peer_gate = torch.tensor([t.data_ptr() for t in self._gate_all], dtype=torch.int64, device="cuda")
        self._push_ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
        # deferred verdict: the combine's per-expert gate map, written by the arm kernel every step
        self._gmap = torch.full((self.gpe,), -1, dtype=torch.int32, device="cuda")
        self._blk = None
        self._dev_rounds = 0
        # the expert-weight pushes run on a side stream forked right where they are issued (before the dispatch op,
        # so ahead of GEMM 1 in any shared hardware queue); the forward stream joins them before the first commit
        self._push_stream = torch.cuda.Stream()
        self._push_fork, self._push_done = torch.cuda.Event(), torch.cuda.Event()
        self._push_pending = False

    def _ep(self):
        return -1 if self.step_device else self.epoch

    def arm_device(self, blk, always=False):
        """Right after swap_decide (same stream): gate words of the unchanged slots, moved-last order and
        overrides of this rank's pulls, all written by one kernel from the device block.

        always (deferred verdict): the host never learns the moves (no per-layer readback), so the step is
        armed for the gated GEMM path whatever the block holds: without an incoming slot every gate word is
        raised and the schedule is the identity, which computes exactly what the ungated path does; the
        front-class count and the combine's gate map are written on the device too."""
        self.epoch += 1
        self._blk = blk
        self._in, self._out = [], []
        self._dev_rounds = None if always else 0     # None: unknown on the host
        self.move_bytes_this_iter = 0
        self._phase_done = [False, False]
        C.lane_arm(blk, self._R, self.rank, self.nlp, self.cap, self._gate, self._sched_dev, self._ovr[0],
                   self._ovr[1], self._stag_addr_dev, self._ep(), always=always, gmap=self._gmap if always else None)

    def read_device(self, rounds, incoming):
        """After the planning sync: whether any rank swapped this step, and this rank's pulls."""
        self._dev_rounds = rounds
        self._in = incoming

    def staged_phase_before(self, k):
        """On the current (forward) stream right before the GEMM of matrix k. Before the dispatch GEMM, push
        every outgoing slot of both matrices into the destination's staging and raise the destination's gate
        word for that slot (lane_push_kernel); before the combine GEMM nothing is pushed: a W2 push right
        before the combine GEMM would make a receiver's W2-gated combine GEMM depend on the sender's whole
        dispatch phase, which can in turn need the receiver (gateway forwards), a cross-rank wait cycle. The
        arm kernel already pointed this rank's override table at the staging of every incoming slot. Returns
        the gate kwargs for the GEMM (None when nothing arrives): its tiles for incoming slots come last and
        spin on the gate word, i.e. directly on the peer's push, and read the expert from the staging. The
        receiver neither copies nor waits before the GEMM; the sender pays only its push copy."""
        if self._dev_rounds is None:
            # deferred verdict: push (the kernel returns at once without a swap) and take the gated path, with
            # the device-written schedule count (sched_n_front < 0) and gate map
            gpe = self.gpe
            self._push_device((0, 1) if k == 0 else ())
            gate = self._gate[k * gpe:(k + 1) * gpe]
            if k == 0:
                return dict(weight_signal=gate[1:], weight_signal_epoch=self._ep(), weight_gate_group_start=1,
                            weight_ptr_override=self._ovr[0], sched_expert_order=self._sched_dev, sched_n_front=-1)
            return dict(weight_signal=gate[1:], weight_signal_epoch=self._ep(), gate_of_expert=[],
                        weight_ptr_override=self._ovr[1], gate_map=self._gmap)
        if not self._dev_rounds:
            return None
        gpe = self.gpe
        self._push_device((0, 1) if k == 0 else ())
        if not self._in:
            return None
        gate = self._gate[k * gpe:(k + 1) * gpe]
        if k == 0:
            return dict(weight_signal=gate[1:], weight_signal_epoch=self._ep(), weight_gate_group_start=1,
                        weight_ptr_override=self._ovr[0], sched_expert_order=self._sched_dev,
                        sched_n_front=gpe - len(self._in))
        gate_of = [-1] * gpe
        for (dj, _sr, _ss, _e) in self._in:
            gate_of[1 + dj] = dj                       # index into gate[1:]
        return dict(weight_signal=gate[1:], weight_signal_epoch=self._ep(), gate_of_expert=gate_of,
                    weight_ptr_override=self._ovr[1])

    def _push_device(self, kks):
        """The pushes of matrices kks (lane_push_kernel) on the side stream, forked from the current stream."""
        if not kks:
            return
        side = self._push_stream
        self._push_fork.record(torch.cuda.current_stream())
        side.wait_event(self._push_fork)
        with torch.cuda.stream(side):
            for kk in kks:
                C.lane_push(self._blk, self._R, self.L, self.rank, self.nlp, self.cap, kk, self.slots.slots(kk),
                            self._slot_bytes[kk], self._peer_stag[kk], self._peer_gate, self._ep(),
                            self._push_ctr[kk:kk + 1])
        self._push_done.record(side)
        self._push_pending = True

    def join_push(self):
        """The forward stream waits for the side-stream pushes of this step (no-op when joined)."""
        if getattr(self, "_push_pending", False):
            torch.cuda.current_stream().wait_event(self._push_done)
            self._push_pending = False

    def commit_after(self, k):
        """After the GEMM of matrix k: land every incoming expert in its slot and drop the override
        (lane_commit_kernel). Gated on the slot's word: an expert without rows schedules no tile and a rank
        without rows skips the GEMM, so the push may not have been waited for yet."""
        self.join_push()                             # a commit may overwrite a slot one of the pushes reads
        if self._dev_rounds is None:
            # deferred verdict: the kernel returns at once when this rank receives no slot
            stag = (self._stag_w1_all if k == 0 else self._stag_w2_all)[self.local_rank].data_ptr()
            C.lane_commit(self._blk, self._R, self.rank, self.nlp, self.cap, k, self._gate, self.slots.slots(k),
                          self._slot_bytes[k], stag, self._ovr[k], self._ep())
            self._phase_done[k] = True
            return
        if not self._in:
            return
        stag = (self._stag_w1_all if k == 0 else self._stag_w2_all)[self.local_rank].data_ptr()
        C.lane_commit(self._blk, self._R, self.rank, self.nlp, self.cap, k, self._gate, self.slots.slots(k),
                      self._slot_bytes[k], stag, self._ovr[k], self._ep())
        self.move_bytes_this_iter += 2 * self._slot_bytes[k] * len(self._in)
        self._phase_done[k] = True

    # -- hooks around the two GEMMs -----------------------------------------------------
    def before_dispatch(self):
        """Arm the dispatch GEMM's start mark (call right before the dispatch op is enqueued)."""
        if self.active():
            self._dispatch_op.set_gemm_start_mark(self.epoch)

    def dispatch_gate_kwargs(self):
        return dict(weight_signal=self.slots.sig1[1:], weight_signal_epoch=self.epoch,
                    weight_gate_group_start=1)

    def after_dispatch(self):
        """Issue the W1 phase (must be the very next enqueue after the dispatch op)."""
        if self.active():
            self._phase(0, self._mark_dispatch)

    def before_combine(self):
        if self.active():
            self._combine_op.set_gemm_start_mark(self.epoch)

    def combine_gate_kwargs(self):
        if not self._in:
            return None
        gate = [-1] * (self.nlp + 1)
        for (dj, _sr, _ss, _e) in self._in:
            gate[1 + dj] = dj
        return dict(weight_signal=self.slots.sig2[1:], weight_signal_epoch=self.epoch,
                    gate_of_expert=gate)

    def after_combine(self):
        """Issue the W2 phase, then join the exchange into the current stream."""
        if self.active():
            self._phase(1, self._mark_combine)
        if self._issued:
            torch.cuda.current_stream().wait_event(self.ev_done)


def symmetric_w2_slots(cfg, dtype, group, rank):
    """W2 slot storage on the symmetric heap, so node peers can read a moved expert's W2 (the dual3 lane pulls it):
    (this rank's [gpe, H, ffn] view, every node peer's view). Collective."""
    shape, gpe, L = cfg.shape, cfg.groups_per_rank, cfg.ranks_per_node
    views = C.create_tensor_list([gpe * shape.hidden, shape.ffn_hidden], dtype, group)
    views[rank % L].zero_()                        # only this rank's storage (a peer may already be filling its own)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    return views[rank % L].view(gpe, shape.hidden, shape.ffn_hidden), views


class Dual3Lane(SwapLane):
    """The 3D swap schedule of the overlapped lane, on the device (gpu_plan with swap): every step is driven by the
    swap decision's result block, with no host read of the moves and no host-issued copy.

    W1 of every moved slot is pushed by its sender under the sender's dispatch GEMM (dual3_push_w1); W2 is pulled by
    its receiver under the receiver's combine GEMM from the sender's slot (dual3_pull_w2; the W2 slots live on the
    symmetric heap, symmetric_w2_slots). The GEMMs read the moved experts from the staging behind the per-slot gate
    words with the moved-last order (the staged lane's arming, overrides and commits). A W2 pull keeps the combine
    GEMM's gates on the receiver's own progress: a sender-side W2 push issued after the sender's dispatch GEMM would
    tie the receiver's resident gated GEMM to the sender's data-dependent waits.

    The phase kernels are launched on side streams behind already-enqueued forward-stream work and spin on their
    GEMM's start mark themselves; no stream wait or event join is parked on a side stream (the process has more
    streams than hardware queues, so a parked wait can block the queue that carries the forward stream's later mark
    write). The commits join on words: the W1 commit waits until this rank's pushes raised the receivers' gates,
    the W2 commit until the receivers acknowledged their pulls. Host epochs: eager layer-steps only."""

    def __init__(self, shape, dtype, rank, L, nlp, pg, **kw):
        super().__init__(shape, dtype, rank, L, nlp, pg, **kw)
        self._pg = pg
        self._d3_slots = None
        self._d3_peer_w2 = None

    def enable_device(self, R):
        super().enable_device(R)
        L, cap = self.L, self.cap
        acks = C.create_tensor_list([L * cap], torch.int64, self._pg, False, True)
        self._d3_ack = acks[self.local_rank]
        self._d3_ack.zero_()
        self._d3_peer_ack = torch.tensor([t.data_ptr() for t in acks], dtype=torch.int64, device="cuda")
        self._d3_stag_w2 = self._stag_w2_all[self.local_rank].data_ptr()
        self._d3_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        self._d3_fork = [torch.cuda.Event(), torch.cuda.Event()]
        self._d3_ctr = torch.zeros(2, dtype=torch.int32, device="cuda")

    def bind(self, slots: WeightSlots):
        super().bind(slots)
        if slots is not self._d3_slots:
            peers = getattr(slots, "w2_peers", None)
            assert peers is not None, "the dual3 lane needs W2 slots on the symmetric heap (symmetric_w2_slots)"
            self._d3_peer_w2 = torch.tensor([v.data_ptr() for v in peers], dtype=torch.int64, device="cuda")
            self._d3_slots = slots

    def staged_phase_before(self, k):
        if self._dev_rounds is not None:              # a host-checked step (warm-up): the staged lane's path
            return super().staged_phase_before(k)
        gpe = self.gpe
        cur = torch.cuda.current_stream()
        side = self._d3_streams[k]
        self._d3_fork[k].record(cur)
        side.wait_event(self._d3_fork[k])
        if k == 0:
            self._dispatch_op.set_gemm_start_mark(self.epoch)
            with torch.cuda.stream(side):
                C.dual3_push_w1(self._blk, self._R, self.L, self.rank, self.nlp, self.cap, self.slots.slots(0),
                                self._slot_bytes[0], self._peer_stag[0], self._peer_gate, self._mark_dispatch,
                                self.epoch, self._d3_ctr[0:1])
            gate = self._gate[0:gpe]
            return dict(weight_signal=gate[1:], weight_signal_epoch=self.epoch, weight_gate_group_start=1,
                        weight_ptr_override=self._ovr[0], sched_expert_order=self._sched_dev, sched_n_front=-1)
        self._combine_op.set_gemm_start_mark(self.epoch)
        with torch.cuda.stream(side):
            C.dual3_pull_w2(self._blk, self._R, self.L, self.rank, self.nlp, self.cap, self._d3_peer_w2,
                            self._slot_bytes[1], self._d3_stag_w2, self._gate, self._d3_peer_ack, self._mark_combine,
                            self.epoch, self._d3_ctr[1:2])
        gate = self._gate[gpe:2 * gpe]
        return dict(weight_signal=gate[1:], weight_signal_epoch=self.epoch, gate_of_expert=[],
                    weight_ptr_override=self._ovr[1], gate_map=self._gmap)

    def commit_after(self, k):
        if self._dev_rounds is None:
            # the commit overwrites the slots this rank gave away: their readers must be done
            C.dual3_wait(self._blk, self._R, self.L, self.rank, self.cap,
                         self._peer_gate if k == 0 else self._d3_ack, k == 0, self.epoch)
        return super().commit_after(k)
