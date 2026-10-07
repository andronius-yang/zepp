"""EPMoE: the expert-parallel MoE layer.

    layer = EPMoE(cfg, group, sizing_routing=...)   # collective
    layer.place(pool_routing)                        # load-aware replicated placement
    layer.load_weights(w1_of, w2_of)                 # expert -> [ffn, H] / [H, ffn]
    plan = layer.prepare(topk_ids, topk_weights)     # per iteration: loads, (swap), route
    y = layer.forward(x, plan)                       # dispatch_gemm -> act -> gemm_combine
"""
import torch

from ._ext import ensure_shm
from .comm import DirectComm, OverlapComm
from .config import EPMoEConfig
from .placement import Placement, demand_histogram, solve_placement
from .planner import Planner
from .routing import compute_capacities
from .swap import SwapLane, WeightSlots, decide_swaps, net_moves, swap_orbit, symmetric_w2_slots


def activation(h, shape, out=None):
    """The expert activation between the two GEMMs: GELU on [m, ffn], or SwiGLU on [m, 2*ffn]
    (gate rows first, then up) -> [m, ffn]."""
    if shape.act == "gelu":
        return torch.nn.functional.gelu(h)
    ffn = shape.ffn_hidden
    gate = torch.nn.functional.silu(h[:, :ffn])
    return torch.mul(gate, h[:, ffn:], out=out)


class Marks:
    """Optional bracket marks for benchmarking (a callable per boundary)."""
    def __call__(self, name):
        pass


class EPMoE(torch.nn.Module):
    def __init__(self, cfg: EPMoEConfig, group, sizing_routing, pool_routing, probs_own=None,
                 dtype=torch.bfloat16):
        """sizing_routing: [R, S, K] expert ids of a representative batch (sizes the buffers
        from the provable routing bounds); pool_routing: [R, S', K] ids the placement is
        planned from; probs_own: this rank's gate weights for the direct strategy's static
        buffers (any [S, K]; the values are re-set by prepare)."""
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.rank = group.rank()
        self.dtype = dtype
        shape = cfg.shape
        R, L, nlp = cfg.ranks, cfg.ranks_per_node, cfg.slots_per_rank
        assert group.size() == R
        self.device = torch.device("cuda")
        ensure_shm(group)                      # NVSHMEM over the layer's group (once per process)
        # placement from the pool
        hist = demand_histogram(pool_routing.to(self.device), L, shape.num_experts).cpu()
        self.placement = solve_placement(hist, L, nlp)
        # sizing: the resident placement and, with swap on, its reachable orbit on the sizing batch
        sizing_cpu = sizing_routing.cpu()
        placements = [self.placement]
        if cfg.swap:
            load_g = torch.bincount(sizing_cpu.reshape(-1).long(), minlength=shape.num_experts)
            placements += swap_orbit(load_g, self.placement, L, nlp, cfg.router_c)
        self.caps = compute_capacities(sizing_cpu, placements, nlp, L, cfg.router_c, group)
        self.redos = 0
        # gpu_plan (overlap on >= 2 nodes): the layer-step runs on the serving runtime, planned and issued on the GPU
        # (SharedComm + LayerState of one layer, one deferred-verdict forward per call); otherwise the host-planned path
        self.gpu_plan = bool(cfg.gpu_plan) and cfg.comm_strategy == "overlap" and cfg.nodes >= 2
        if self.gpu_plan:
            from .serving import LayerState, SharedComm
            w2 = w2_peers = None
            if cfg.swap:
                w2, w2_peers = symmetric_w2_slots(cfg, dtype, group, self.rank)
            self.shared = SharedComm(cfg, group, self.caps, dtype, layer_graphs=False if cfg.swap else None,
                                     swap_lane="dual3")
            self.layer_state = LayerState(0, cfg, self.placement, self.rank, dtype=dtype, w2=w2)
            self.layer_state.slots.w2_peers = w2_peers
            self.slots = self.layer_state.slots
            self.planner = self.comm = self.swap_lane = None
            self._step_in = None
            return
        self.planner = Planner(cfg, self.placement, self.rank, group, self.device)
        if cfg.comm_strategy == "overlap":
            self.comm = OverlapComm(cfg, group, self.caps, dtype)
        else:
            assert not cfg.swap, "swap is defined on the overlap strategy"
            if probs_own is None:
                probs_own = torch.ones(cfg.max_tokens_per_rank, shape.topk)
            self.comm = DirectComm(cfg, group, self.caps, probs_own, dtype)
        self.slots = None
        self.swap_lane = None
        if cfg.swap:
            self.slots = WeightSlots(self.rank, cfg, dtype)
            self.swap_lane = SwapLane(shape, dtype, self.rank, L, nlp, group)
            self.swap_lane.bind(self.slots)
            self.swap_lane.attach_ops(self.comm.dispatch_op, self.comm.combine_op)
        self._loads_host = None

    # -- weights ------------------------------------------------------------------------
    def load_weights(self, w1_of, w2_of):
        """w1_of(e) -> [ffn, H], w2_of(e) -> [H, ffn] (host or device tensors); called for every
        expert this rank hosts (unassigned redundant slots stay zero)."""
        cfg, shape = self.cfg, self.cfg.shape
        nlp, gpe = cfg.slots_per_rank, cfg.groups_per_rank
        if self.gpu_plan:
            self.layer_state.slots.fill(self.placement.p2l, w1_of, w2_of)
            return
        if self.slots is not None:
            self.slots.fill(self.placement.p2l, w1_of, w2_of)
            w1, w2 = self.slots.gemm_weights()
        else:
            w1 = torch.zeros(gpe, shape.ffn1, shape.hidden, dtype=self.dtype, device=self.device)
            w2 = torch.zeros(gpe, shape.hidden, shape.ffn_hidden, dtype=self.dtype, device=self.device)
            for i in range(nlp):
                e = int(self.placement.p2l[self.rank * nlp + i])
                if e < 0:
                    continue                    # unassigned redundant slot (never routed to)
                w1[1 + i].copy_(w1_of(e).to(self.dtype))
                w2[1 + i].copy_(w2_of(e).to(self.dtype))
        self.comm.set_weights(w1, w2)

    # -- setup ------------------------------------------------------------------------
    def prime(self, topk_ids, topk_weights):
        """Capture the plan/scale graphs with every rank quiesced; run once before timing."""
        if self.gpu_plan:
            # every kernel loaded and every bucket run once (warm-up), then the layer graphs of the small buckets
            self.shared.prime(self.layer_state)
            self.shared.warmup(self.layer_state)
            if self.shared.layer_graph:
                self.shared.capture_graphs([self.layer_state])
            torch.cuda.synchronize()
            torch.distributed.barrier(group=self.group)
            return None
        self.planner.set_routing(topk_ids, topk_weights)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self.group)
        self.planner.prime()
        loads = self.planner.exchange_loads()
        plan = self.planner.plan(loads)
        self.comm.plan_meta(plan, self.planner._probs_own)
        self.comm.prime(self.planner)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self.group)
        return plan

    # -- per iteration ----------------------------------------------------------------------
    def prepare(self, topk_ids, topk_weights, marks=Marks()):
        """Exchange loads -> (swap decision) -> route -> op metadata. Returns the Plan."""
        cfg = self.cfg
        if self.gpu_plan:
            # planning runs inside forward's layer-step on the GPU
            self._step_in = (topk_ids, topk_weights)
            for name in ("plan_comm", "place", "plan"):
                marks(name)
            return None
        self.planner.set_routing(topk_ids, topk_weights)
        loads = self.planner.exchange_loads()
        marks("plan_comm")
        if self.swap_lane is not None:
            load_g = loads.cpu().long().sum(0)               # waits for the in-flight all_gather
            new_pl, _rounds = decide_swaps(load_g, self.placement, cfg.ranks_per_node, cfg.slots_per_rank,
                                           cfg.router_c)
            moves = [[] for _ in range(cfg.ranks)]
            if new_pl is not None:
                moves = net_moves(self.placement.p2l, new_pl.p2l, cfg.ranks_per_node, cfg.slots_per_rank)
                self.placement = new_pl
                self.planner.apply_placement(new_pl)
            self.swap_lane.prepare(moves)
        marks("place")
        plan = self.planner.plan(loads)
        self.comm.plan_meta(plan, self.planner._probs_own)
        marks("plan")
        return plan

    def forward(self, x, plan, marks=Marks()):
        if self.gpu_plan:
            return self._forward_gpu(x, marks)
        lane = self.swap_lane
        gate = lane.dispatch_gate_kwargs() if lane is not None else None
        if lane is not None:
            lane.before_dispatch()
        h = self.comm.dispatch_gemm(x, gate)
        if lane is not None:
            lane.after_dispatch()                # must be the very next enqueue after dispatch
        self.comm.issue_combine_meta_late(plan)
        marks("dispatch")
        h = activation(h, self.cfg.shape)
        marks("act")
        gate1 = None
        if lane is not None:
            lane.before_combine()
            gate1 = lane.combine_gate_kwargs()
        y = self.comm.gemm_combine(h, gate1)
        if lane is not None:
            lane.after_combine()
        marks("combine")
        return y

    def _forward_gpu(self, x, marks):
        """One forward of this layer on the GPU path: the capacity verdict is read once, after the forward (a
        forward that exceeded the buffers grows them and runs again)."""
        sh, st = self.shared, self.layer_state
        ids, w = self._step_in
        n = [int(x.shape[0])] * self.cfg.ranks
        sh.begin_forward()
        y = sh.step(st, x, ids, w, n)
        sh.end_forward()
        for name in ("dispatch", "act", "combine"):
            marks(name)
        abort = sh.forward_check()
        while abort is not None:
            self.redos += 1
            sh.recover(abort, st)
            sh.begin_forward()
            y = sh.step(st, x, ids, w, n)
            sh.end_forward()
            abort = sh.forward_check()
        return y

    @property
    def swap_moves(self):
        return self.shared.swap_moves if self.gpu_plan else 0

    def prep(self):
        if self.gpu_plan:
            return
        self.comm.prep()
