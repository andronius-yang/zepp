"""`direct` communication strategy: a plain one-sided all-to-all wire (NVSHMEM), then the
expert GEMMs, with no overlap between the two. Same planner, same placement and routing."""
import gc

import torch
import torch.distributed as dist

from .. import constants
from .._ext import C
from ..capacity import CapacityExceeded, Demands, violations


def _counts(ids, size):
    out = torch.zeros(size, dtype=torch.int64, device=ids.device)
    return out.index_add_(0, ids.reshape(-1), torch.ones_like(ids.reshape(-1)))


def direct_layout(ent_tok, ent_phys, rank, nlp, R):
    """Sync-free layout of the direct wire from the (token, physical slot) entries of all
    ranks in destination-major order: splits, receiver segments, placement scatter."""
    dev = ent_tok.device
    N = ent_tok.shape[1]
    RN = R * N
    my_tok = ent_tok[rank]
    my_phys = ent_phys[rank]
    in_splits = _counts(torch.div(my_phys, nlp, rounding_mode="floor"), R).to(torch.int32)
    dest_all = torch.div(ent_phys, nlp, rounding_mode="floor")
    mine = dest_all == rank
    out_splits = mine.sum(dim=1).to(torch.int32)
    minef = mine.reshape(-1)
    loc_key = torch.where(minef, ent_phys.reshape(-1) - rank * nlp, torch.full((1,), nlp, dtype=torch.int64, device=dev))
    seg_full = _counts(loc_key, nlp + 1)
    seg_rows = seg_full[:nlp]
    seg_start = torch.zeros(nlp, dtype=torch.int64, device=dev)
    seg_start[1:] = torch.cumsum(seg_rows, dim=0)[:-1]
    sorder = torch.argsort(loc_key, stable=True)
    arr_idx = minef.long().cumsum(0) - 1
    idx = torch.where(minef[sorder], arr_idx[sorder], torch.full((1,), RN, dtype=torch.int64, device=dev))
    place_pad = torch.empty(RN + 1, dtype=torch.int64, device=dev)
    place_pad.scatter_(0, idx, torch.arange(RN, device=dev, dtype=torch.int64))
    src_ids = torch.arange(R, device=dev, dtype=torch.int64).unsqueeze(1)
    pair_rows = _counts((src_ids * R + dest_all).reshape(-1), R * R)
    recv_all = _counts(dest_all.reshape(-1), R)               # rows every destination receives
    return dict(my_tok=my_tok, in_splits=in_splits, out_splits=out_splits, place_slots_pad=place_pad,
                seg_rows=seg_rows, seg_start=seg_start, n_recv_dev=minef.sum().reshape(1),
                pair_max=pair_rows.max(), recv_max=recv_all.max())


class DirectComm:
    def __init__(self, cfg, group, caps, probs_own=None, dtype=torch.bfloat16, check_capacity=False):
        """probs_own: default gate weights [S, K] when plan_meta is not given per-step ones;
        check_capacity: raise CapacityExceeded (identically on all ranks) instead of asserting."""
        shape = cfg.shape
        self.cfg = cfg
        self.group = group
        self.rank = group.rank()
        self.W = group.size()
        self.L = cfg.ranks_per_node
        self.S, self.K, self.H = cfg.max_tokens_per_rank, shape.topk, shape.hidden
        self.ffn, self.ffn1 = shape.ffn_hidden, shape.ffn1
        self.nlp = cfg.slots_per_rank
        self.gpe = cfg.groups_per_rank
        self.N = self.S * self.K
        self.dtype = dtype
        self.check_capacity = check_capacity
        self.num_comm_sm = constants.DIRECT_COMM_SMS
        dev = torch.device("cuda")
        self._gemm = C.GemmOnly(dtype, dtype, dtype)
        self._tok_t = torch.arange(self.S, device=dev, dtype=torch.int64).repeat_interleave(self.K)
        self._k_t = torch.arange(self.K, device=dev, dtype=torch.int64).repeat(self.S)
        if probs_own is None:
            probs_own = torch.ones(self.S, self.K)
        self._probs_own_flat = probs_own.reshape(-1).float().contiguous().to(dev)
        self._blob_pin = torch.empty(2 * self.nlp + 2 * self.W + 3, dtype=torch.int64, pin_memory=True)
        S, H, N = self.S, self.H, self.N
        self.send_buf = torch.empty(N, H, dtype=dtype, device=dev)
        self.wsend_buf = torch.empty(N, dtype=torch.float32, device=dev)
        self.comb_recv_buf = torch.empty(N, H, dtype=dtype, device=dev)
        self.stage_buf = torch.empty(N, H, dtype=dtype, device=dev)
        self.final_out = torch.zeros(S, H, dtype=dtype, device=dev)
        self._in_splits = torch.empty(self.W, dtype=torch.int32, device=dev)
        self._out_splits = torch.empty(self.W, dtype=torch.int32, device=dev)
        self.n_recv = 0
        self.S_b = self.S
        self.N_b = self.N
        self._send_row_index = self._place_slots = self._comb_dst = self._pentry = None
        self._segments = []
        self.plan_overlap = 0
        self.last_demands = None
        self.generation = 0
        self.w1 = self.w2 = None
        self._build(caps)

    # -- symmetric resources ------------------------------------------------------------
    def _build(self, caps):
        """Collective: the two all-to-all wires (symmetric heap) and the capacity-sized buffers."""
        self.caps = caps
        self.recv_cap = int(caps.recv_cap)
        self.max_split = int(caps.pair_cap)
        dev, dtype, H, cap = torch.device("cuda"), self.dtype, self.H, self.recv_cap
        self._a2a_hidden = C.All2AllSingle(self.group, self.max_split, H, self.L, dtype)
        self._a2a_probs = C.All2AllSingle(self.group, self.max_split, 1, self.L, torch.float32)
        self.recv_buf = torch.empty(cap, H, dtype=dtype, device=dev)
        self.wrecv_buf = torch.empty(cap, dtype=torch.float32, device=dev)
        self.hidden_buf = torch.zeros(cap, H, dtype=dtype, device=dev)
        self.weights_buf = torch.zeros(cap, dtype=torch.float32, device=dev)
        self.out_buf = torch.zeros(cap, self.ffn1, dtype=dtype, device=dev)
        self.comb_hidden_buf = torch.zeros(cap, H, dtype=dtype, device=dev)
        self.comb_send_buf = torch.empty(cap, H, dtype=dtype, device=dev)
        self.generation += 1

    def resize(self, caps):
        """Collective: rebuild the two all-to-all wires and the capacity-sized buffers for `caps`
        (the wires hold no streams or epoch state; the caller quiesces every rank first)."""
        self._a2a_hidden = self._a2a_probs = None
        self.recv_buf = self.wrecv_buf = self.hidden_buf = self.weights_buf = None
        self.out_buf = self.comb_hidden_buf = self.comb_send_buf = None
        self._place_slots = None
        gc.collect()
        dist.barrier(group=self.group)
        self._build(caps)

    def set_weights(self, w1_slots, w2_slots):
        assert w1_slots.shape == (self.gpe, self.ffn1, self.H), tuple(w1_slots.shape)
        assert w2_slots.shape == (self.gpe, self.H, self.ffn), tuple(w2_slots.shape)
        self.w1 = w1_slots.contiguous()
        self.w2 = w2_slots.contiguous()

    def prep(self, used_only=False):
        self.out_buf.zero_()

    def prime(self, planner):
        return

    def plan_meta(self, plan, probs_own=None):
        """Virtual routing -> wire layout; one batched pinned D2H is the host sync. probs_own:
        this step's gate weights [S_b*K] (default: the constructor's)."""
        K, R, nlp = self.K, self.W, self.nlp
        ntok = plan.vce.shape[0]
        assert ntok % R == 0
        S_b = ntok // R
        N_b = S_b * K
        vce = plan.vce.view(R, N_b).long()
        phys_all = (vce // self.gpe) * nlp + (vce % self.gpe) - 1
        tok_exp = self._tok_t[:N_b].expand(R, N_b)
        k_exp = self._k_t[:N_b].expand(R, N_b)
        order = torch.argsort(phys_all * (S_b + 1) + tok_exp, dim=1, stable=True)
        ent_tok = torch.gather(tok_exp, 1, order)
        ent_phys = torch.gather(phys_all, 1, order)
        lay = direct_layout(ent_tok, ent_phys, self.rank, nlp, R)
        my_tok = lay["my_tok"]
        my_k = torch.gather(k_exp, 1, order)[self.rank]
        ent_flat = my_tok * K + my_k
        blob = self._blob_pin
        blob.copy_(torch.cat([lay["seg_rows"], lay["seg_start"], lay["in_splits"].long(), lay["out_splits"].long(),
                              lay["pair_max"].reshape(1), lay["recv_max"].reshape(1), lay["n_recv_dev"].long()]))
        seg_rows_h = blob[:nlp].tolist()
        seg_start_h = blob[nlp:2 * nlp].tolist()
        pair_max, recv_max, n_recv = int(blob[-3]), int(blob[-2]), int(blob[-1])
        if self.check_capacity:
            d = Demands(recv_rows=max(recv_max, 1), dispatch_recv=1, dispatch_stage=1, dispatch_relay=1,
                        combine_conv=1, combine_wire=1, pair_rows=max(pair_max, 1))
            self.last_demands = d
            v = violations(d, self.caps, direct=True)
            if v:
                raise CapacityExceeded(v, d)
        else:
            assert pair_max <= self.max_split, f"pair rows {pair_max} exceed the wire staging {self.max_split}"
            assert n_recv <= self.recv_cap, f"recv overflow: {n_recv} > recv_cap {self.recv_cap}"
        probs = self._probs_own_flat if probs_own is None else probs_own.reshape(-1)
        self.n_recv = n_recv
        self.S_b, self.N_b = S_b, N_b
        self._send_row_index = my_tok
        self._place_slots = lay["place_slots_pad"][:n_recv]
        self._comb_dst = ent_flat
        self._pentry = probs[ent_flat]
        self._in_splits.copy_(lay["in_splits"])
        self._out_splits.copy_(lay["out_splits"])
        self._segments = [(p, seg_start_h[p], seg_start_h[p] + seg_rows_h[p]) for p in range(nlp) if seg_rows_h[p] > 0]

    def issue_combine_meta_late(self, plan):
        return

    def dispatch_gemm(self, x, gate_kwargs=None, w1=None):
        w1 = self.w1 if w1 is None else w1
        N_b = self.N_b
        torch.index_select(x, 0, self._send_row_index, out=self.send_buf[:N_b])
        self.wsend_buf[:N_b].copy_(self._pentry)
        n = self.n_recv
        self._a2a_hidden.forward(self.send_buf[:N_b], self.recv_buf[:n], self._in_splits, self._out_splits, self.num_comm_sm)
        self._a2a_probs.forward(self.wsend_buf[:N_b].view(-1, 1), self.wrecv_buf[:n].view(-1, 1), self._in_splits,
                                self._out_splits, self.num_comm_sm)
        self.hidden_buf[:n].index_copy_(0, self._place_slots, self.recv_buf[:n])
        self.weights_buf[:n].index_copy_(0, self._place_slots, self.wrecv_buf[:n])
        for p, start, end in self._segments:
            self._gemm.forward(self.hidden_buf[start:end], w1[1 + p], output_buf=self.out_buf[start:end],
                               fast_accum=False)
        return self.out_buf[:n]

    def gemm_combine(self, h, gate_kwargs=None, w2=None):
        w2 = self.w2 if w2 is None else w2
        n, N_b, S_b = self.n_recv, self.N_b, self.S_b
        for p, start, end in self._segments:
            self._gemm.forward(h[start:end], w2[1 + p], output_buf=self.comb_hidden_buf[start:end],
                               fast_accum=False)
        rows = self.comb_hidden_buf[:n].index_select(0, self._place_slots)
        scale = self.weights_buf[:n].index_select(0, self._place_slots)
        self.comb_send_buf[:n] = (rows.float() * scale.unsqueeze(1)).to(self.dtype)
        self._a2a_hidden.forward(self.comb_send_buf[:n], self.comb_recv_buf[:N_b], self._out_splits, self._in_splits,
                                 self.num_comm_sm)
        self.stage_buf[:N_b].index_copy_(0, self._comb_dst, self.comb_recv_buf[:N_b])
        self.final_out[:S_b].copy_(self.stage_buf[:N_b].view(S_b, self.K, self.H).sum(1))
        return self.final_out[:S_b]
