"""Per-iteration planner: routes this rank's tokens to physical expert slots and exchanges
the routing with every rank.

Per iteration (all timed):
  1. loads       d[R, G] = all_gather of this rank's per-expert token counts (plan_comm);
  2. route       the device kernel builds the water-fill tables from d and the current
                 placement (identical on every rank) and assigns this rank's entries;
  3. exchange    ONE all_gather of (physical slots | gate weights) for all ranks;
  4. tail        virtual GEMM group ids for the fused ops (pad group first), replayed as a
                 CUDA graph on persistent buffers.
Nothing carries over between iterations except buffers.
"""
import torch
import torch.distributed as dist

from ._ext import C
from .routing import c_rational


class Plan:
    """One iteration's routing products."""
    __slots__ = ("vce", "probs_all", "stable")

    def __init__(self, vce, probs_all, stable=False):
        self.vce = vce              # [R*S, K] int32 virtual group ids (device)
        self.probs_all = probs_all  # [R*S, K] fp32 gate weights (device)
        self.stable = stable        # True: lives in the planner's persistent buffers


class Planner:
    def __init__(self, cfg, placement, rank, group, device, use_graph=True, S=None, ws=None):
        """placement: the tables the planner routes on (None: the tables are passed to every
        plan() call); S: tokens per rank (default cfg.max_tokens_per_rank); ws: a shared route
        workspace (its size does not depend on S)."""
        shape = cfg.shape
        self.cfg = cfg
        self.rank = rank
        self.group = group
        self.device = device
        self.R = cfg.ranks
        self.L = cfg.ranks_per_node
        self.NN = cfg.nodes
        self.G = shape.num_experts
        self.K = shape.topk
        self.S = S or cfg.max_tokens_per_rank
        self.nlp = cfg.slots_per_rank
        self.gpe = cfg.groups_per_rank
        self.c_num, self.c_den = c_rational(cfg.router_c)
        self.use_graph = use_graph
        S, K, R = self.S, self.K, self.R
        # device copies of the placement tables (updated in place by the swap lane)
        self.p2l = self.l2p = self.lcnts = None
        if placement is not None:
            assert int(placement.lcnts.max()) <= 32, "route kernel: > 32 replicas of one expert"
            self.p2l = placement.p2l.long().to(device)
            self.l2p = placement.l2p.to(device)
            self.lcnts = placement.lcnts.to(device)
            self._pin_p2l = [torch.empty(R * self.nlp, dtype=torch.long).pin_memory() for _ in range(2)]
            self._pin_l2p = [torch.empty_like(placement.l2p).pin_memory() for _ in range(2)]
        self._pin_parity = 0
        # routing inputs (this rank), persistent
        self._topk_own = torch.zeros(S, K, dtype=torch.int32, device=device)
        self._topk_own_flat = torch.zeros(S * K, dtype=torch.int64, device=device)
        self._probs_own = torch.zeros(S, K, dtype=torch.float32, device=device)
        self._d_own = torch.zeros(self.G, dtype=torch.int32, device=device)
        self._ones = torch.ones(S * K, dtype=torch.int32, device=device)
        self._ws = ws if ws is not None else torch.empty(C.route_workspace_ints(self.G, R), dtype=torch.int32, device=device)
        # fused exchange buffers: [phys (S*K) | probs fp32-bitcast (S*K)] int32 per rank
        self._xchg_send = torch.empty(2 * S * K, dtype=torch.int32, device=device)
        self._xchg_gather = torch.empty(R * 2 * S * K, dtype=torch.int32, device=device)
        # persistent tail buffers
        self._vce_buf = torch.empty(R * S, K, dtype=torch.int32, device=device)
        self._probs_all_buf = torch.empty(R * S, K, dtype=torch.float32, device=device)
        self._tail_graph = None
        self._tail_graph_broken = False
        self.loads_buf = torch.zeros(R, self.G, dtype=torch.int32, device=device)

    # -- placement tables ------------------------------------------------------------
    def apply_placement(self, placement):
        """Push new host tables into the device tables (two non-blocking copies from pinned
        mirrors, double-buffered by parity). lcnts is invariant under swaps."""
        k = self._pin_parity
        self._pin_parity ^= 1
        self._pin_p2l[k].copy_(placement.p2l.long())
        self._pin_l2p[k].copy_(placement.l2p)
        self.p2l.copy_(self._pin_p2l[k], non_blocking=True)
        self.l2p.copy_(self._pin_l2p[k], non_blocking=True)

    # -- routing input ---------------------------------------------------------------
    def set_routing(self, topk_ids, topk_weights):
        """This rank's gating output for the next plan: [S, K] expert ids and weights."""
        assert topk_ids.shape == (self.S, self.K), (tuple(topk_ids.shape), (self.S, self.K))
        if topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32:
            # elementwise kernels instead of copy-engine copies (a scheduling bubble each inside a layer graph);
            # integer adds of zero copy the bits exactly (the weights through an int32 view)
            torch.add(topk_ids, 0, out=self._topk_own)
            torch.add(self._topk_own.reshape(-1), 0, out=self._topk_own_flat)
            torch.add(topk_weights.contiguous().view(torch.int32), 0, out=self._probs_own.view(torch.int32))
            return
        self._topk_own.copy_(topk_ids.to(torch.int32))
        self._topk_own_flat.copy_(self._topk_own.reshape(-1).long())
        self._probs_own.copy_(topk_weights.to(torch.float32))

    def local_loads(self):
        """d[G] for this rank (sync-free index_add)."""
        self._d_own.zero_()
        self._d_own.index_add_(0, self._topk_own_flat, self._ones)
        return self._d_own

    def exchange_loads(self):
        dist.all_gather_into_tensor(self.loads_buf, self.local_loads(), group=self.group)
        return self.loads_buf

    # -- plan -------------------------------------------------------------------------
    def plan(self, loads, tables=None) -> Plan:
        """tables: (l2p, lcnts) device tables to route on (default: the planner's own)."""
        l2p, lcnts = tables if tables is not None else (self.l2p, self.lcnts)
        # tables (+ counters zeroed), budgets per expert, route + vacate + the send row: three launches
        phys_own, _stats = C.route_fused(self._topk_own, loads, l2p, lcnts, self.rank, self.nlp, self.L,
                                         self.c_num, self.c_den, self._ws, self._probs_own.view(-1),
                                         self._xchg_send)
        dist.all_gather_into_tensor(self._xchg_gather, self._xchg_send, group=self.group)
        # inside a layer-graph capture the tail is recorded eagerly (no nested replay)
        if self.use_graph and not self._tail_graph_broken and not torch.cuda.is_current_stream_capturing():
            if self._tail_graph is None:
                self._capture_tail_graph()
            if self._tail_graph is not None:
                self._tail_graph.replay()
                return Plan(self._vce_buf, self._probs_all_buf, stable=True)
        self._tail_compute()
        return Plan(self._vce_buf, self._probs_all_buf, stable=True)

    def _tail_compute(self):
        # gate weights and physical -> virtual group ids (pad group first) of every rank, one kernel
        C.planner_tail(self._xchg_gather, self._probs_all_buf, self._vce_buf, self.R, self.S * self.K, self.nlp,
                       self.gpe)

    def _capture_tail_graph(self):
        try:
            for _ in range(2):
                self._tail_compute()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._tail_compute()
            g.replay()
            torch.cuda.synchronize()
            self._tail_graph = g
        except Exception as e:  # noqa: BLE001 - eager fallback
            self._tail_graph = None
            self._tail_graph_broken = True
            if self.rank == 0:
                print(f"[zepp] plan tail graph capture failed ({type(e).__name__}: {e}); eager", flush=True)

    def prime(self):
        """Capture the tail graph at setup, with every rank quiesced (a first-use capture inside
        an iteration can deadlock against peers' in-flight collectives)."""
        if self.use_graph and not self._tail_graph_broken and self._tail_graph is None:
            self._capture_tail_graph()

    def plan_from_route(self, phys_all) -> Plan:
        """Plan from a given full routing [R*S, K] (the reference route): gathers the gate
        weights from all ranks. Used for the correctness check, not on the timed path."""
        p = phys_all.view(self.R * self.S, self.K).long().to(self.device)
        vce = ((p // self.nlp) * self.gpe + 1 + p % self.nlp).int()
        out = torch.empty(self.R * self.S * self.K, dtype=torch.float32, device=self.device)
        dist.all_gather_into_tensor(out, self._probs_own.view(-1), group=self.group)
        return Plan(vce, out.view(self.R * self.S, self.K))
