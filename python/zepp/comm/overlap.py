"""`overlap` communication strategy: the all-to-all dispatch is fused into grouped GEMM 1 and
grouped GEMM 2 is fused with the all-to-all combine. Tiles of the GEMMs spin on per-source
arrival signals, so communication and computation overlap at tile granularity; inter-node
traffic is staged per round through node gateways with peer pulls over NVLink.
"""
import torch

from .. import constants
from .._ext import C
from ..capacity import CapacityExceeded, Demands, violations


class OverlapComm:
    def __init__(self, cfg, group, caps, dtype=torch.bfloat16, check_capacity=False):
        """check_capacity: evaluate the exact buffer demands of every plan against `caps` before
        anything is launched and raise CapacityExceeded (identically on all ranks) instead of
        letting the ops trip their own checks mid-forward. The serving path turns it on."""
        shape = cfg.shape
        self.cfg = cfg
        self.group = group
        self.rank = group.rank()
        self.W = group.size()
        self.L = cfg.ranks_per_node
        self.nnodes = cfg.nodes
        self.gpe = cfg.groups_per_rank
        self.E_virt = cfg.virtual_experts
        self.H, self.ffn, self.ffn1, self.K = shape.hidden, shape.ffn_hidden, shape.ffn1, shape.topk
        self.dtype = dtype
        self.sm_margin = cfg.sm_margin
        self.ntokens = cfg.ranks * cfg.max_tokens_per_rank
        self.ep_start = self.rank * self.gpe
        self.check_capacity = check_capacity
        budget_mib = cfg.max_tokens_per_rank * shape.chunk_bytes / (1 << 20)
        # the combine-metadata derive overlaps the dispatch GEMM on a side stream (plan_overlap 2)
        # up to PLAN_OVERLAP_MAX_MIB per rank; above that it runs inline in the plan bracket
        self.plan_overlap = 2 if budget_mib <= constants.PLAN_OVERLAP_MAX_MIB else 0
        self._env = C.DistEnv(tp_group=group, nnodes=self.nnodes, ep_group=group)
        self._scale_graphs = {}          # (ntok, probs ptr, generation) -> CUDAGraph
        self._scale_graph_broken = False
        self._meta_ev = torch.cuda.Event()
        self._plan_ev = torch.cuda.Event()
        self._main_stream = torch.cuda.current_stream()
        self._sd = self._scd = self._sps = self._uc = None
        self._combine_kwargs = None
        self._m_this = 0
        # rows the GEMM buffers of the step are sliced to: the true row count on a host-checked step, its
        # upper bound min(recv_cap, copies of the step) on a deferred-verdict step (the count stays on the
        # device; the GEMMs, the pack and the scale fold read it there, rows past it are never read)
        self._m_cap = 0
        self._deferred_step = False
        self.last_demands = None
        self.generation = 0
        self.w1 = self.w2 = None
        self._build_ops(caps)
        # the combine's host unique_counts argument on a deferred-verdict step: shape only (the device plan
        # block drives), so the step's counts are never read on the host
        self._uc_stub = torch.zeros(self.W, self.nnodes, dtype=torch.int32)
        # the combine derive's side stream is created by the combine op (non-blocking, default priority):
        # torch.cuda.Stream() would hand out a pool stream shared round-robin with NCCL and the swap lane
        self._meta_stream = torch.cuda.ExternalStream(self.combine_op.meta_stream())

    # -- symmetric resources ------------------------------------------------------------
    def _build_ops(self, caps):
        """Collective: allocates the two fused ops (symmetric heap) for `caps`."""
        self.caps = caps
        self.recv_cap = int(caps.recv_cap)
        moe_args = C.MoeArguments(max_ntokens=self.ntokens, hidden=self.H, ffn_hidden=self.ffn1,
                                  nexperts=self.E_virt, topk=self.K, input_dtype=self.dtype, output_dtype=self.dtype)
        dispatch_opts = C.DispatchOptions(max_recv_rows=caps.dispatch_recv, max_stage_rows=caps.dispatch_stage,
                                          max_relay_rows=caps.dispatch_relay)
        self.dispatch_op = C.DispatchGemmOp(self._env, moe_args, dispatch_opts)
        combine_opts = C.CombineOptions(max_send_rows=caps.combine_send, max_conv_rows=caps.combine_conv,
                                        max_wire_rows=caps.combine_wire)
        self.combine_op = C.GemmCombineOp(self.group, self.E_virt, self.ntokens * self.K, self.H, self.K, self.dtype,
                                          self.W, self.nnodes, combine_opts)
        self.out_buf = torch.zeros(self.recv_cap, self.ffn1, dtype=self.dtype, device="cuda")
        self.scale_buf = torch.zeros(self.recv_cap + 1, dtype=torch.float32, device="cuda")
        self.generation += 1

    def resize(self, caps):
        """Collective: grow the ops' capacity-sized symmetric panels in place (streams, events,
        signals and the run epoch are kept; nothing is re-primed). The caller quiesces every
        rank first (device sync + barrier). The scale graphs are dropped because scale_buf moves."""
        self.dispatch_op.resize_capacities(C.DispatchOptions(
            max_recv_rows=caps.dispatch_recv, max_stage_rows=caps.dispatch_stage, max_relay_rows=caps.dispatch_relay))
        self.combine_op.resize_capacities(C.CombineOptions(
            max_send_rows=caps.combine_send, max_conv_rows=caps.combine_conv, max_wire_rows=caps.combine_wire))
        self.caps = caps
        self.recv_cap = int(caps.recv_cap)
        self.out_buf = torch.zeros(self.recv_cap, self.ffn1, dtype=self.dtype, device="cuda")
        self.scale_buf = torch.zeros(self.recv_cap + 1, dtype=torch.float32, device="cuda")
        self._scale_graphs.clear()
        self.generation += 1

    def set_weights(self, w1_slots, w2_slots):
        """Per-group weights [gpe, ffn1, H] / [gpe, H, ffn]; group 0 is the empty pad group."""
        assert w1_slots.shape == (self.gpe, self.ffn1, self.H), tuple(w1_slots.shape)
        assert w2_slots.shape == (self.gpe, self.H, self.ffn), tuple(w2_slots.shape)
        self.w1 = w1_slots.contiguous()
        self.w2 = w2_slots.contiguous()

    # -- plan bracket -----------------------------------------------------------------
    def plan_meta(self, plan, probs_own=None, verdict=None):
        """Derive the dispatch metadata in-op on the device (its pinned D2H event sync is the one honest
        host sync of the iteration), then check the step's exact buffer demands (evaluated on the device,
        landed with the same sync).

        verdict = (layer ordinal, abort flag): a deferred-verdict step (a serving forward is open,
        core/verdict.h). The demands are checked on the device against the process's verdict block, which
        makes this and every later layer of the forward degenerate on a violation; nothing raises here and no
        count is read on the host (the caller checks the block once, at the end of the forward)."""
        if verdict is not None:
            c = self.caps
            caps_vec = [c.recv_cap, c.dispatch_recv, c.dispatch_stage, c.dispatch_relay, c.combine_send,
                        c.combine_conv, c.combine_wire, c.pair_cap, constants.RELAY_SLOTS,
                        int(verdict[0]), int(bool(verdict[1]))]
            sd, scd, sps, uc, _dem, sps_dev, uc_dev = self.dispatch_op.derive_routed_meta(plan.vce, caps_vec, False)
            self._sps_dev, self._uc_dev = sps_dev, uc_dev
            self._sd, self._scd, self._sps, self._uc = sd, scd, sps, uc
            self._deferred_step = True
            self._m_cap = min(self.recv_cap, int(plan.vce.numel()))   # a shape, not a count
            self.last_demands = None
            if self.plan_overlap == 2:
                self._plan_ev.record(torch.cuda.current_stream())
            else:
                self._derive_combine(plan)
            return
        self._deferred_step = False
        # the device counts also feed the combine's device tables
        c = self.caps
        caps_vec = [c.recv_cap, c.dispatch_recv, c.dispatch_stage, c.dispatch_relay, c.combine_send,
                    c.combine_conv, c.combine_wire, c.pair_cap, constants.RELAY_SLOTS]
        sd, scd, sps, uc, dem, sps_dev, uc_dev = self.dispatch_op.derive_routed_meta(plan.vce, caps_vec, False)
        self._sps_dev, self._uc_dev = sps_dev, uc_dev
        self._sd, self._scd, self._sps, self._uc = sd, scd, sps, uc
        self._m_this = int(sps[:, self.ep_start:self.ep_start + self.gpe].sum())
        self._m_cap = self._m_this
        if self.check_capacity:
            d = Demands(*[int(v) for v in dem[:7].tolist()])
            mask = int(dem[7])
            self.last_demands = d
            if mask:
                raise CapacityExceeded(violations(d, self.caps), d)
        else:
            assert self._m_this <= self.recv_cap, f"recv overflow: {self._m_this} > recv_cap {self.recv_cap}"
        if self.plan_overlap == 2:
            self._plan_ev.record(torch.cuda.current_stream())
        else:
            self._derive_combine(plan)

    def _derive_combine(self, plan):
        if self.nnodes == 1:
            uc_combine = None
        elif self._deferred_step:
            uc_combine = self._uc_stub
        else:
            uc_combine = self._uc[:, self.W:].contiguous()
        # device counts from plan_meta; the routing ids are the expert of every copy
        extra = {"sps_dev": self._sps_dev, "uc_dev": self._uc_dev, "routing_ids": plan.vce.view(-1)}
        if self.plan_overlap and not plan.stable:
            plan.vce.record_stream(torch.cuda.current_stream())
        meta = self.combine_op.derive_combine_meta(self._sd, self._scd.view(-1), self._sps, unique_counts=uc_combine,
                                                  **extra)
        if self.plan_overlap:
            for t in meta:
                t.record_stream(self._main_stream)
        combine_kwargs = {"splits_per_source": self._sps, "pack_index": meta[0], "reduce_index": meta[1]}
        if uc_combine is not None:
            combine_kwargs["wire_csr"] = [meta[2], meta[3]]
            combine_kwargs["reduce_csr"] = [meta[4], meta[5]]
            combine_kwargs["unique_counts"] = uc_combine
        self._combine_kwargs = combine_kwargs
        self._build_scale(plan)

    def _scale_compute(self, plan):
        sd_long = self._sd.long()
        m_start = sd_long[:self.ep_start].sum()
        idx = self._scd.view(-1).long() - m_start
        valid = (idx >= 0) & (idx < self.recv_cap)
        idx = torch.where(valid, idx, torch.full_like(idx, self.recv_cap))
        self.scale_buf.zero_()
        self.scale_buf.scatter_(0, idx, plan.probs_all.view(-1))

    def _scale_key(self, plan):
        return (plan.vce.shape[0], plan.probs_all.data_ptr(), self.generation)

    def _build_scale(self, plan):
        """Replay the captured scale graph of this (bucket, op generation) if there is one,
        else compute eagerly (graphs are captured only at prime time, never mid-step)."""
        if not self._scale_graph_broken and plan.stable and not torch.cuda.is_current_stream_capturing():
            g = self._scale_graphs.get(self._scale_key(plan))
            if g is not None:
                g.replay()
                return
        self._scale_compute(plan)

    def _capture_scale_graph(self, plan):
        key = self._scale_key(plan)
        if self._scale_graph_broken or key in self._scale_graphs or not plan.stable:
            return
        try:
            for _ in range(2):
                self._scale_compute(plan)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._scale_compute(plan)
            g.replay()
            torch.cuda.synchronize()
            self._scale_graphs[key] = g
        except Exception as e:  # noqa: BLE001
            self._scale_graph_broken = True
            if self.rank == 0:
                print(f"[zepp] scale graph capture failed ({type(e).__name__}: {e}); eager", flush=True)

    def prime(self, planner):
        """Setup-time scale-graph capture on the planner's persistent buffers (call right after
        a plan_meta of that planner's plan, with every rank quiesced)."""
        if self._sd is None:
            return
        from ..planner import Plan
        self._main_stream = torch.cuda.current_stream()
        self._capture_scale_graph(Plan(planner._vce_buf, planner._probs_all_buf, stable=True))

    # -- the two fused layers ----------------------------------------------------------
    def dispatch_gemm(self, x, gate_kwargs=None, w1=None):
        w1 = self.w1 if w1 is None else w1
        self.dispatch_op.forward(inputs_shard=x, weights=w1, splits_gpu=self._sd, scatter_index=self._scd,
                                 outputs_buf=self.out_buf[:self._m_cap], fast_accum=False, sm_margin=self.sm_margin,
                                 splits_per_source=self._sps, unique_counts=self._uc, **(gate_kwargs or {}))
        return self.out_buf[:self._m_cap]

    def issue_combine_meta_late(self, plan):
        """plan_overlap 2: the combine metadata derive runs now, while the GPU executes the dispatch
        GEMM; its kernels ride the side stream on the sm_margin headroom."""
        if self.plan_overlap != 2:
            return
        self._main_stream = torch.cuda.current_stream()
        self._meta_stream.wait_event(self._plan_ev)
        with torch.cuda.stream(self._meta_stream):
            self._derive_combine(plan)
        self._meta_ev.record(self._meta_stream)
        # the combine's msplit tables and pack inverse fork from here onto a side stream (they need the metadata
        # only; in a layer graph they run while the dispatch GEMM computes)
        self.combine_op.set_prep_fork(self._meta_ev.cuda_event)

    def activation(self, h, shape, out):
        """SwiGLU between the GEMMs; on a deferred-verdict step over the rows this rank computes only (a device word
        of the dispatch plan block, not the host's shape bound min(recv_cap, R*S*K), which grows with the batch; rows
        past them are never read: the combine's problem sizes, scale fold and pack are bounded on the device)."""
        if self._deferred_step and shape.act == "swiglu" and h.dtype == torch.bfloat16:
            C.silu_mul_bounded(h, out, shape.ffn_hidden, self.dispatch_op.rows_dev())
            return out
        from ..layer import activation
        return activation(h, shape, out=out)

    def gemm_combine(self, h, gate_kwargs=None, w2=None):
        w2 = self.w2 if w2 is None else w2
        if self.plan_overlap:
            torch.cuda.current_stream().wait_event(self._meta_ev)
        if gate_kwargs is not None:
            self.combine_op.set_weight_gate(**gate_kwargs)
        return self.combine_op.forward(h, w2, self._sd, self._scd.view(-1),
                                       output_vec_scale=self.scale_buf[:self._m_cap], fast_accum=False,
                                       sm_margin=self.sm_margin, bias=None, **self._combine_kwargs)

    def prep(self, used_only=False):
        """Per-iteration hygiene. The benchmark zeroes the whole GEMM output buffer outside its timed window;
        serving (used_only, after plan_meta) zeroes only this step's rows: rows the GEMM does not write, e.g.
        the pad group's, are read as zero downstream, and rows past them are never read (a deferred-verdict
        step zeroes up to the row bound)."""
        if used_only and self._deferred_step:
            C.zero_rows_bounded(self.out_buf[:self._m_cap], self.dispatch_op.rows_dev())
        else:
            (self.out_buf[:self._m_cap] if used_only else self.out_buf).zero_()
        self.dispatch_op.clear_buffers()
