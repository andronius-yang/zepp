"""Serving path: one process-wide `SharedComm` (the symmetric buffers and fused ops, one planner
per token bucket, the swap staging) serves every MoE layer of a model; each layer keeps its own
`LayerState` (placement tables, weight slots, readiness signals).

Per layer-step the token count varies per rank, so every rank pads to the same bucket (the
smallest bucket holding the largest rank's count; all ranks know all counts, no communication).
Pad rows carry real expert ids of this rank's own non-replicated slots with zero gate weight:
they route locally, cost only GEMM rows, and are subtracted before the swap band test.

Capacity guarantee (no fallback): the exact buffer demands of every step are checked before
anything is launched; when a step needs more than the buffers hold, every rank grows the
buffers collectively (grow the panels in place, same routing, same step) and continues on the same path.

Deferred verdict (overlap strategy): the check moves to the device and to the end of the
forward. The caller brackets a model forward with begin_forward() / end_forward() and calls
forward_check() before using its outputs. Inside the bracket no layer-step reads a routing count on
the host: the demands kernel of every layer ORs its violations into a process-wide device block
(core/verdict.h), and from the first violating layer on the forward is degenerate (zero rows
everywhere, no swap; nothing overflows, nothing hangs). forward_check() reads the block once,
confirms that every rank holds the same verdict and, on a violation, returns it; the caller then
calls recover() (collective: grow the buffers, re-prime with the warm-up) and runs the same forward
again. The swaps the aborted pass decided before the violating layer are kept, so the second pass
does not decide them again for those layers (redo_skip). SharedComm(force_abort=<MoE layer ordinal>)
sets the verdict once at that layer of a forward (test hook for the redo path).
"""
import time

import numpy as np
import torch
import torch.distributed as dist

from . import constants
from ._ext import C, ensure_shm
from .capacity import CapacityExceeded, Demands, grow, violations
from .comm import DirectComm, OverlapComm
from .layer import activation
from .placement import Placement
from .planner import Planner
from .swap import Dual3Lane, SwapLane, WeightSlots


def token_buckets(cfg):
    """K, 2K, 4K, ... and max_tokens_per_rank (all multiples of top-k)."""
    K, S_max = cfg.shape.topk, cfg.max_tokens_per_rank
    out, s = [], K
    while s < S_max:
        out.append(s)
        s *= 2
    out.append(S_max)
    return out


class WarmupSkip(Exception):
    """A warm-up step whose exact buffer demands exceed the capacities (raised identically on every
    rank before anything is launched): the warm-up skips it instead of growing the buffers."""


class ForwardAbort:
    """The deferred verdict of an aborted forward (SharedComm.forward_check), identical on every rank:
    the device mask, the MoE layer ordinal that tripped first, its demands, and whether the buffers must
    grow (capacity) or the abort came from the test hook (forced)."""
    __slots__ = ("mask", "layer", "demands", "capacity", "forced")

    def __init__(self, mask, layer, demands, capacity, forced):
        self.mask, self.layer, self.demands, self.capacity, self.forced = mask, layer, demands, capacity, forced

    def __repr__(self):
        return (f"ForwardAbort(layer={self.layer}, mask={self.mask:#x}, capacity={self.capacity}, "
                f"forced={self.forced}, demands={self.demands})")


class LayerState:
    """Per-layer state: placement tables (device + pinned mirrors), weight slots, pad tables."""

    def __init__(self, layer_id, cfg, placement: Placement, rank, dtype=torch.bfloat16, w1=None, w2=None,
                 device="cuda"):
        self.layer_id = layer_id
        self.cfg = cfg
        self.rank = rank
        self.R, self.L, self.nlp = cfg.ranks, cfg.ranks_per_node, cfg.slots_per_rank
        self.G, self.K, self.S_max = cfg.shape.num_experts, cfg.shape.topk, cfg.max_tokens_per_rank
        self.device = torch.device(device)
        assert int(placement.lcnts.max()) <= 32, "route kernel: > 32 replicas of one expert"
        self.placement = placement
        self.p2l = placement.p2l.long().to(self.device)
        self.l2p = placement.l2p.to(self.device)
        self.lcnts = placement.lcnts.to(self.device)
        self.slots = WeightSlots(rank, cfg, dtype, device=device, w1=w1, w2=w2)
        self._pad_lists = None
        self.pad_table = None
        self._refresh_pads()
        # deferred verdict: the pad table a device rebuild overwrote, and whether this layer's last step
        # rebuilt it (a forward that runs again after an abort pads like the aborted pass did)
        self.pad_prev = torch.empty_like(self.pad_table)
        self.pad_flag = torch.zeros(1, dtype=torch.int32, device=self.device)

    def tables(self):
        return self.l2p, self.lcnts

    def set_placement_host(self, placement: Placement):
        """Host mirror only: the device swap decision already rewrote p2l / l2p in place."""
        self.placement = placement
        self._pad_dirty = True

    # -- padding -------------------------------------------------------------------------
    def _pad_experts(self, p2l, lcnts, r):
        hosted = [int(e) for e in p2l[r * self.nlp:(r + 1) * self.nlp] if e >= 0]
        single = [e for e in hosted if lcnts[e] == 1]
        if single:
            return single
        if hosted:
            return hosted
        home = self.G // self.R
        return list(range(r * home, (r + 1) * home))

    def _refresh_pads(self):
        p2l = self.placement.p2l.numpy().astype(np.int64)
        lcnts = self.placement.lcnts.numpy().astype(np.int64)
        self._pad_lists = [np.asarray(self._pad_experts(p2l, lcnts, r), dtype=np.int64) for r in range(self.R)]
        mine = self._pad_lists[self.rank]
        idx = np.arange(self.S_max * self.K) % len(mine)
        table = torch.from_numpy(mine[idx].astype(np.int32)).view(self.S_max, self.K)
        if self.pad_table is None:
            self.pad_table = table.to(self.device)
        else:
            # in place: captured layer graphs hold this table's address (their device pad rebuild writes it)
            self.pad_table.copy_(table)
        self._pad_dirty = False

    def rebuild_pads_device(self, blk, save=False):
        """Device decision: the next step's pad table from the rewritten p2l, built on the device right
        after swap_decide (no-op without a swap). save=True (deferred verdict) keeps the table it
        overwrites in pad_prev and records in pad_flag whether it rebuilt."""
        C.pad_rebuild(blk, self.p2l, self.lcnts, self.rank, self.nlp, self.G // self.R, self.pad_table,
                      pad_prev=self.pad_prev if save else None, flag=self.pad_flag if save else None)

    def pad_rows(self, n, S_b, aborted_pass=False):
        """Pad expert ids of rows n..S_b-1; aborted_pass: the table the aborted pass of this forward used."""
        if aborted_pass:
            return torch.where(self.pad_flag.bool(), self.pad_prev[n:S_b], self.pad_table[n:S_b])
        return self.pad_table[n:S_b]

    def refresh_pads_if_needed(self):
        if self._pad_dirty:
            self._refresh_pads()

    def pad_hist(self, n_tokens, S_b):
        """[G] rows the pad tokens of every rank add to the gathered loads this step (closed form:
        rank r's pads are rows n_r..S_b-1 of its cyclic pad table)."""
        hist = np.zeros(self.G, dtype=np.int64)
        K = self.K
        for r, lst in enumerate(self._pad_lists):
            lo, hi = int(n_tokens[r]) * K, S_b * K
            if hi <= lo:
                continue
            m = len(lst)
            i = np.arange(m)
            cnt = (hi - 1 - i) // m - (lo - 1 - i) // m
            np.add.at(hist, lst, cnt)
        return torch.from_numpy(hist)


class SharedComm:
    """Process-wide communication state shared by every layer (collective constructor).

    Serving runs the overlap strategy on two or more nodes; other configurations are refused at construction.

    layer_graphs: capture_graphs() records one CUDA graph per (layer, bucket) and the steps of an open forward
    replay them (None or True). False: every step eager."""

    def __init__(self, cfg, group, caps, dtype=torch.bfloat16, use_graph=True, log=print, force_abort=-1,
                 layer_graphs=None, swap_lane="staged"):
        """swap_lane: "staged" (both matrices pushed before the dispatch GEMM, the serving default) or "dual3" (the 3D
        schedule on the device, Dual3Lane: W1 under the dispatch GEMM, W2 under the combine GEMM; eager layer-steps,
        W2 slots on the symmetric heap)."""
        assert swap_lane in ("staged", "dual3"), swap_lane
        shape = cfg.shape
        self.cfg, self.group, self.dtype = cfg, group, dtype
        self.rank, self.W, self.L = group.rank(), group.size(), cfg.ranks_per_node
        assert self.W == cfg.ranks
        self.K, self.H, self.ffn, self.ffn1 = shape.topk, shape.hidden, shape.ffn_hidden, shape.ffn1
        self.G, self.nlp, self.S_max = shape.num_experts, cfg.slots_per_rank, cfg.max_tokens_per_rank
        self.direct = cfg.comm_strategy == "direct"
        if self.direct or cfg.nodes < 2:
            raise NotImplementedError("serving supports comm_strategy='overlap' on two or more nodes "
                                      f"(got {cfg.comm_strategy!r} on {cfg.nodes} node(s))")
        self.log = log
        dev = torch.device("cuda")
        ensure_shm(group)
        # symmetric allocations that outlive rebuilds come first (the ops are the heap tail)
        self.lane = None
        if cfg.swap:
            assert not self.direct, "swap is defined on the overlap strategy"
            lane_cls = Dual3Lane if swap_lane == "dual3" else SwapLane
            self.lane = lane_cls(shape, dtype, self.rank, self.L, self.nlp, group)
        if self.direct:
            self.comm = DirectComm(cfg, group, caps, dtype=dtype, check_capacity=True)
        else:
            self.comm = OverlapComm(cfg, group, caps, dtype, check_capacity=True)
        if self.lane is not None:
            self.lane.attach_ops(self.comm.dispatch_op, self.comm.combine_op)
        self.buckets = token_buckets(cfg)
        self._ws = torch.empty(C.route_workspace_ints(self.G, self.W), dtype=torch.int32, device=dev)
        # the layer's two all-gathers (loads, routing exchange) run on a dedicated communicator (same ranks), eager
        # and inside captured layer graphs alike, never mixed with the framework's own collectives on `group`
        self.plan_group = dist.new_group(ranks=dist.get_process_group_ranks(group))
        self.planners = {S_b: Planner(cfg, None, self.rank, self.plan_group, dev, use_graph=use_graph, S=S_b,
                                      ws=self._ws)
                         for S_b in self.buckets}
        self.x_buf = torch.zeros(self.S_max, self.H, dtype=dtype, device=dev)
        self.ids_buf = torch.zeros(self.S_max, self.K, dtype=torch.int32, device=dev)
        self.w_buf = torch.zeros(self.S_max, self.K, dtype=torch.float32, device=dev)
        self.act_buf = torch.zeros(self.comm.recv_cap, self.ffn, dtype=dtype, device=dev)
        self.growths = 0
        self.swap_moves = 0
        self.steps = 0
        self._reprime = False
        self._warmup_mode = False                   # warm-up steps: no swap, capacity overflow skips the step
        self._warmup_gate = None                    # warm-up: gate kwargs of a gated step without a swap
        self.warmed = False
        # swap lane (staged, on the device): arming, pushes, commits and the next step's pad table are kernels reading
        # the decision block; the GEMM reads the moved expert from the staging behind its tile gates, the commit runs
        # after the GEMM. Swap decision: the swap_decide kernel (tables rewritten in place, the result block read after
        # the step's planning sync on a host-checked step)
        if self.lane is not None:
            assert constants.SWAP_PAIR_MOVES == 1, "the device swap decision implements one exchange per pair per round"
            n_out = C.swap_decide_out_ints(self.W, self.G, self.nlp, constants.SWAP_MAX_MOVES)
            self._sd_out = torch.zeros(n_out, dtype=torch.int64, device=dev)
            self._sd_pin = torch.zeros(n_out, dtype=torch.int64, pin_memory=True)
            self._sd_ev = torch.cuda.Event()
            self._nt_pin = torch.zeros(self.W, dtype=torch.int64, pin_memory=True)
            self._nt_dev = torch.zeros(self.W, dtype=torch.int64, device=dev)
            self.lane.mode = "staged"
            self.lane.enable_device(self.W)
        # device step state (core/step_state.h, overlap strategy): a head kernel per layer-step advances the run
        # ids and the swap epoch on the device; the GEMM tile / weight gates and the device lane read them there.
        # The host counters (ops' run ids, lane epoch) are synced into the device ones on an idle device.
        self.step_state = not self.direct
        self._step_synced = False
        if self.step_state and self.lane is not None:
            self.lane.step_device = True
        dual3 = self.lane is not None and swap_lane == "dual3"
        if dual3:
            self.lane.step_device = False            # the dual3 lane runs on host epochs (eager layer-steps)
            assert not layer_graphs, "the dual3 lane runs eager layer-steps (no layer graphs)"
            layer_graphs = False
        # one CUDA graph per (layer, bucket): needs the device step state and the deferred verdict (overlap strategy;
        # no host value or host thread inside a layer-step)
        self.layer_graph = not self.direct if layer_graphs is None else bool(layer_graphs)
        if self.layer_graph:
            assert self.step_state and not self.direct, "layer graphs need the overlap strategy"
        self._graphs = {}                            # (layer_id, S_b) -> (CUDAGraph, y [S_b, H])
        self.graph_replays = 0
        # the host stays at most HOST_AHEAD_LAYERS layer graphs ahead of the GPU
        self._ahead_k = constants.HOST_AHEAD_LAYERS
        self._ahead_ev = [torch.cuda.Event() for _ in range(self._ahead_k)]
        self._ahead_i = 0
        self._graph_pool = None
        self._graph_gen = -1                         # comm.generation the graphs were captured for
        self._graph_layers = None
        self._capturing = False
        # deferred capacity verdict (module docstring): the forward window and the end-of-forward readback
        self.deferred = not self.direct
        self.redos = 0
        self._fwd_open = False
        self._fwd_ended = False
        self._fwd_layer = 0                         # MoE layer ordinal within the open forward
        self._fwd_state = None                      # the forward's first layer (re-prime after an abort)
        self._redo_skip = -1                        # second pass: the ordinals whose swap the aborted pass kept
        self._nt_key = None
        self._force_abort = int(force_abort)        # test hook: abort the first forward once at this MoE layer
        self._force_done = False
        if self.deferred:
            self._vlay = C.verdict_layout()
            self._vblock = C.verdict_tensor()                    # the device block (core/verdict.h)
            nw = int(self._vlay["words"])
            # rows: this rank's block | block (all-reduced max) | -block (all-reduced max = -min) | counters
            self._vdev = torch.zeros(4, nw, dtype=torch.int64, device=dev)
            self._vpin = torch.zeros(4, nw, dtype=torch.int64, pin_memory=True)
            self._vev = torch.cuda.Event()
            self._moves_acc = torch.zeros(1, dtype=torch.int64, device=dev)   # swap moves of the forward

    @property
    def caps(self):
        return self.comm.caps

    def report(self):
        return dict(steps=self.steps, growths=self.growths, generation=self.comm.generation,
                    swap_moves=self.swap_moves, caps=self.comm.caps, redos=self.redos)

    # -- deferred capacity verdict (once per forward) ------------------------------------------
    def begin_forward(self):
        """Open a model forward (no-op with the direct strategy): zero the device verdict block on the
        current stream; the layer-steps until end_forward() check their demands on the device only."""
        if not self.deferred:
            return
        assert not self._fwd_open, "begin_forward: a forward is already open"
        C.verdict_begin()
        self._fwd_open = True
        self._fwd_ended = False
        self._fwd_layer = 0
        self._moves_acc.zero_()

    def end_forward(self):
        """Close the forward: one D2H of the verdict block (plus its max and min over the ranks, one
        all-reduce) and of the forward's swap counter, behind the forward on the current stream."""
        if not self.deferred or not self._fwd_open:
            return
        C.verdict_end()
        self._fwd_open = False
        v = self._vdev
        v[0].copy_(self._vblock)
        v[1].copy_(self._vblock)
        torch.neg(self._vblock, out=v[2])
        v[3, 0:1].copy_(self._moves_acc)
        dist.all_reduce(v[1:3], op=dist.ReduceOp.MAX, group=self.group)
        self._vpin.copy_(v, non_blocking=True)
        self._vev.record()
        self._fwd_ended = True

    def abandon_forward(self):
        """Disarm after an exception inside the forward (the block is left as it is)."""
        if self.deferred and self._fwd_open:
            C.verdict_end()
            self._fwd_open = False
            self._fwd_ended = False

    def forward_check(self):
        """The verdict of the forward closed by end_forward(): None when it stands, else a ForwardAbort
        (the forward must run again after recover()). Waits for the forward to finish (one event sync).
        Identical on every rank (it comes from the replicated counts); confirmed here, and any disagreement
        raises on every rank."""
        if not self.deferred or not self._fwd_ended:
            return None
        self._fwd_ended = False
        self._vev.synchronize()
        lay = self._vlay
        loc = self._vpin[0].tolist()
        mx = self._vpin[1].tolist()
        mn = [-x for x in self._vpin[2].tolist()]
        if mx != mn:
            raise RuntimeError(f"zepp: ranks disagree on the forward's capacity verdict (max {mx}, min {mn})")
        assert loc == mx, (loc, mx)
        self.swap_moves += int(self._vpin[3, 0])
        mask = loc[lay["mask"]]
        if mask == 0:
            self._redo_skip = -1
            return None
        if mask & lay["error_bit"]:
            self._redo_skip = -1
            raise RuntimeError(f"zepp: device assert in the forward (verdict mask {mask:#x}, error bits "
                               f"{loc[lay['err']]:#x}; core/verdict.h): inconsistent routing counts")
        d0 = lay["demands"]
        return ForwardAbort(mask, loc[lay["latch"]] - 1, Demands(*loc[d0:d0 + 7]),
                            capacity=bool(mask & lay["capacity_bits"]), forced=bool(mask & lay["force_bit"]))

    def recover(self, abort, layer=None):
        """Collective, between forwards, after forward_check() returned `abort`: quiesce the wire, clear the
        verdict, grow the buffers to the latched demands (identical on every rank), re-prime with the
        warm-up, and mark the ordinals whose swap decision the aborted pass already applied. The caller then
        runs the same forward again (begin_forward ... end_forward, forward_check). layer: the LayerState the
        warm-up runs on (default: the forward's first layer)."""
        layer = layer if layer is not None else self._fwd_state
        assert layer is not None, "recover: no layer-step ran in the forward"
        C.verdict_reset()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        if abort.capacity:
            v = violations(abort.demands, self.comm.caps, direct=self.direct)
            self._grow(CapacityExceeded(v, abort.demands), layer, where=f"MoE layer {abort.layer} of the forward")
        elif self.rank == 0:
            self.log(f"[zepp] forward aborted at MoE layer {abort.layer} (forced abort); running it again",
                     flush=True)
        self._reprime = False
        self.redos += 1
        self.warmup(layer, keep_lane_epoch=True)
        if self.layer_graph and self._graph_layers and self._graph_gen != self.comm.generation:
            self.capture_graphs(self._graph_layers)  # the buffers moved: recapture (collective)
        self._redo_skip = max(self._redo_skip, abort.layer)

    def _take_force(self, ordinal):
        if self._force_abort == ordinal and not self._force_done:
            self._force_done = True
            return True
        return False

    # -- setup -----------------------------------------------------------------------------
    def bucket_of(self, n_max):
        """Tokens per rank of this step: the coarse power-of-two bucket."""
        n_max = int(n_max)
        if n_max > self.S_max:
            raise RuntimeError(f"{n_max} tokens on one rank exceed max_tokens_per_rank {self.S_max}: bound the "
                               "per-rank tokens per step (prefill chunk, running requests) to the configured ceiling")
        for s in self.buckets:
            if s >= n_max:
                return s
        raise AssertionError

    def prime(self, layer: LayerState):
        """Route every bucket once (pad rows only, all local) with every rank quiesced: captures the
        planners' tail graphs and the per-bucket scale graphs; repeats after any growth."""
        for _ in range(8):
            self._reprime = False
            for S_b in self.buckets:
                planner = self.planners[S_b]
                planner.set_routing(layer.pad_table[:S_b], self.w_buf[:S_b])
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                planner.prime()
                loads = planner.exchange_loads()
                plan = planner.plan(loads, layer.tables())
                self._plan_meta_checked(plan, planner, layer)
                if not self._reprime:
                    self.comm.prime(planner)
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            if not self._reprime:
                return
        raise RuntimeError("capacities did not settle while priming")

    # -- capacity guarantee ------------------------------------------------------------------
    def _plan_meta_checked(self, plan, planner, layer):
        probs = planner._probs_own.view(-1)
        for _ in range(16):
            try:
                self.comm.plan_meta(plan, probs)
                return
            except CapacityExceeded as e:
                if self._warmup_mode:                # identical on every rank: all skip the step
                    raise WarmupSkip(str(e)) from None
                self._grow(e, layer)
        raise RuntimeError("capacity growth did not converge")

    def _grow(self, exc: CapacityExceeded, layer: LayerState, where=None):
        """Collective, decided identically on every rank from the replicated demands."""
        new = grow(self.comm.caps, exc.demands, direct=self.direct)
        fields = ("recv_cap", "dispatch_recv", "dispatch_stage", "dispatch_relay", "combine_send", "combine_conv",
                  "combine_wire", "pair_cap")
        vec = torch.tensor([getattr(new, f) for f in fields], dtype=torch.int64, device="cuda")
        ref = vec.clone()
        dist.all_reduce(vec, op=dist.ReduceOp.MAX, group=self.group)
        assert bool(torch.equal(vec, ref)), "ranks disagree on the grown capacities"
        self.growths += 1
        if self.rank == 0:
            self.log(f"[zepp] capacity growth #{self.growths} at step {self.steps} "
                     f"{where if where is not None else f'layer {layer.layer_id}'}: {exc}; new {new}", flush=True)
        torch.cuda.synchronize()                     # every inbound put has landed on every rank
        dist.barrier(group=self.group)
        self.comm.resize(new)                        # in place: streams, signals, epoch untouched
        self.act_buf = torch.zeros(self.comm.recv_cap, self.ffn, dtype=self.dtype, device="cuda")
        self._reprime = True

    # -- one layer-step --------------------------------------------------------------------------
    def warmup_lane(self, layer: LayerState):
        """Preload the swap lane's kernels on the layer's slots (call once, before the first forward)."""
        if self.lane is not None and not getattr(self, "_lane_warm", False):
            self.lane.warmup(layer.slots)
            C.lane_device_preload()
            self._lane_warm = True

    def _swap_band_c(self, kept=False):
        """Band tolerance C of the swap decision; the warm-up steps pass a band no load leaves (no swap), and
        so does a layer whose swap the aborted pass of this forward already applied (kept)."""
        return 1e30 if (self._warmup_mode or kept) else float(self.cfg.router_c)

    # -- warm-up (once, before serving) ------------------------------------------------------
    def warmup(self, layer: LayerState, keep_lane_epoch=False):
        """Load, before serving, everything the steady state can launch, so that no first launch (a lazy
        code load, during which the GPU takes no new work from any thread while a kernel is resident)
        happens beside a spinning GEMM. Collective: every rank calls it at the same point with its own
        state of the same layer, before the first real step. Steps, in order:
          1. C.preload_all(): every kernel of libzepp_cuda.so, the GEMM of every registered op included
             (the ops' constructors already loaded these and primed every NVSHMEM primitive variant);
          2. prime(): every bucket's routing on pad rows, capturing the planners' tail graphs (otherwise
             captured inside a bucket's first step) and the scale graphs (otherwise eager);
          3. the swap lane's primitives and the torch kernels only a step with moves launches (scratch);
          4. per power-of-two bucket, full layer steps (routing, dispatch GEMM, activation, combine GEMM,
             wire) on synthetic tokens: every rank full; every rank full with the swap step's gated GEMM
             path (gates already raised, no weight redirected); one rank with zero tokens while the
             others send to every rank; the same with nothing sent to the zero rank; and once every
             rank with zero tokens (signal-only wire everywhere).
        Model state is unchanged: the warm-up steps cannot decide a swap (the band test is disabled), so
        no weight moves and the placement tables stay; the swap lane's epoch, its device words, the
        layer's tables and the step counters are restored from a snapshot after a full quiesce; a warm-up
        step whose buffer demands exceed the capacities is skipped (identically on every rank) instead
        of growing the buffers. The operators' run ids keep advancing (they must stay monotonic), and with
        keep_lane_epoch (a re-prime between forwards) so does the swap lane's epoch (never rewound).
        Returns a summary dict."""
        t0 = time.time()
        # a warm-up reached from inside a forward (SGLang's fallback at the first MoE layer when the loader hook
        # did not fire): its steps are host-checked, so the deferred window closes around it and reopens after
        reopen = self.deferred and self._fwd_open
        if reopen:
            assert self._fwd_layer == 0, "warm-up inside a forward after its first MoE layer"
            self.abandon_forward()
        try:
            return self._warmup(layer, t0, keep_lane_epoch)
        finally:
            if reopen:
                self.begin_forward()

    def _warm_deferred_ops(self, layer):
        """The torch kernels only a deferred-verdict forward launches (the aborted pass's pad select, the swap
        counter, the end-of-forward readback): loaded here, on an idle device, never first beside a spinner."""
        if not self.deferred:
            return
        S = min(self.S_max, self.K)
        layer.pad_rows(0, S, aborted_pass=True)
        acc = torch.zeros_like(self._moves_acc)
        if self.lane is not None:
            acc.add_(self._sd_out[1:2])
        v = torch.zeros_like(self._vdev)
        v[0].copy_(self._vblock)
        torch.neg(self._vblock, out=v[2])
        torch.cuda.synchronize()

    def _warmup(self, layer, t0, keep_lane_epoch=False):
        n_kernels = int(C.preload_all())
        self._warm_activation()
        self._warm_deferred_ops(layer)
        self.prime(layer)
        self.warmup_lane(layer)
        layer.refresh_pads_if_needed()
        dry_gate = None
        if self.lane is not None:
            self.lane.warmup_swap_path()
            dry_gate = self.lane.dry_gate_kwargs()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        snap = self._warmup_snapshot(layer)
        ran, skipped = [], []
        self._warmup_mode = True
        try:
            for i, S_b in enumerate(self.buckets):
                z = i % self.W
                zero = [S_b] * self.W
                zero[z] = 0
                cases = [("full", [S_b] * self.W, None, None)]
                if dry_gate is not None:
                    cases.append(("full_gated", [S_b] * self.W, None, dry_gate))
                if self.W > 1:
                    cases += [("zero_rank", zero, None, None), ("zero_rank_isolated", zero, z, None)]
                if i == 0:
                    cases.append(("all_zero", [0] * self.W, None, None))
                for name, n_list, avoid, gate in cases:
                    x, ids, w = self._warmup_inputs(layer, n_list, avoid)
                    self._warmup_gate = gate
                    try:
                        self.step(layer, x, ids, w, n_tokens=n_list)
                        ran.append((S_b, name))
                    except WarmupSkip as e:
                        skipped.append((S_b, name, str(e)))
                    finally:
                        self._warmup_gate = None
        finally:
            self._warmup_mode = False
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        self._warmup_restore(layer, snap, keep_lane_epoch)
        self._step_synced = False                    # the restore may rewind the lane epoch: re-sync before the next head
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        self.warmed = True
        out = dict(kernels=n_kernels, steps=len(ran), skipped=skipped, buckets=list(self.buckets),
                   growths=self.growths, seconds=round(time.time() - t0, 2))
        if self.rank == 0:
            self.log(f"[zepp] warm-up: {n_kernels} kernels loaded, {len(ran)} layer-steps over buckets "
                     f"{self.buckets}, {len(skipped)} skipped for capacity "
                     f"{[(b, n) for b, n, _ in skipped]}, growths {self.growths}, {out['seconds']} s", flush=True)
        return out

    def _warm_activation(self):
        """The activation's kernel variants, launched once before any layer step: strided gate / up halves (two
        or more rows) and the contiguous single-row path. A first launch is a lazy code load, which must not
        happen beside a spinning GEMM (the GPU would take no new work until the GEMM ends, and the GEMM waits
        on that work); one row is a real decode case."""
        shape = self.cfg.shape
        width = 2 * shape.ffn_hidden if shape.act == "swiglu" else shape.ffn_hidden
        for m in (1, 2, 3):
            h = torch.zeros(m, width, dtype=self.dtype, device="cuda")
            if shape.act == "swiglu":
                activation(h, shape, out=self.act_buf[:m])
            else:
                activation(h, shape)
        torch.cuda.synchronize()

    def _warmup_inputs(self, layer, n_list, avoid=None):
        """This rank's synthetic tokens of a warm-up step: zero activations, gate weights 1/K, and token i
        routed (all K copies) to the pad experts of destination rank dests[(rank + i) % len(dests)] (its
        single-replica hosted experts, so the copies land on that rank): once a bucket has W tokens every
        (source, destination) lane carries rows, and every rank receives its fair share of copies.
        dests: every rank, or every rank but `avoid` (the zero-token rank then receives nothing)."""
        n, K = int(n_list[self.rank]), self.K
        dev = torch.device("cuda")
        x = torch.zeros(n, self.H, dtype=self.dtype, device=dev)
        w = torch.full((n, K), 1.0 / K, dtype=torch.float32, device=dev)
        dests = np.asarray([d for d in range(self.W) if d != avoid], dtype=np.int64)
        ids = np.zeros((n, K), dtype=np.int32)
        if n:
            rows = np.arange(n)
            dest = dests[(self.rank + rows) % len(dests)]
            for d in np.unique(dest):
                r = rows[dest == d]
                lst = np.asarray(layer._pad_lists[int(d)], dtype=np.int64)
                ids[r] = lst[(r[:, None] * K + np.arange(K)[None, :]) % len(lst)]
        return x, torch.from_numpy(ids).to(dev), w

    def _warmup_snapshot(self, layer):
        snap = dict(steps=self.steps, swap_moves=self.swap_moves,
                    placement=layer.placement, pad_dirty=layer._pad_dirty,
                    pad_lists=layer._pad_lists,
                    dev=[(t, t.clone()) for t in (layer.p2l, layer.l2p, layer.lcnts, layer.pad_table,
                                                  layer.slots.sig1, layer.slots.sig2)])
        if self.lane is not None:
            snap["lane"] = self.lane.snapshot()
        return snap

    def _warmup_restore(self, layer, snap, keep_lane_epoch=False):
        """Every rank quiesced (device synchronized, past a barrier)."""
        if self.swap_moves != snap["swap_moves"]:
            raise RuntimeError("zepp warm-up: a swap was decided during the warm-up (weights moved)")
        for t, saved in snap["dev"]:
            t.copy_(saved)
        layer.placement, layer._pad_dirty = snap["placement"], snap["pad_dirty"]
        layer._pad_lists = snap["pad_lists"]
        self.steps = snap["steps"]
        if self.lane is not None:
            epoch = self.lane.epoch
            self.lane.restore(snap["lane"])
            if keep_lane_epoch:
                # the warm-up decides no swap, so it raised no gate word: the advanced epoch stays valid, and a
                # rewound one could meet a word already at a higher value
                self.lane.epoch = epoch

    def _read_swap_decision(self):
        """Parse the device decision's result block (the tables and the pad table are already rewritten on the
        device) and hand this rank's pulls to the device lane (its kernels read the other ranks' lists themselves)."""
        self._sd_ev.synchronize()
        blk = self._sd_pin.numpy()
        W, cap = self.W, constants.SWAP_MAX_MOVES
        rounds, total, err = int(blk[0]), int(blk[1]), int(blk[2])
        assert err == 0, f"device swap decision error flags {err} (1: source not found, 2: over the staging cap)"
        off_mv, off_p2l = 3 + W, 3 + W + W * cap * 4
        incoming = []
        if rounds > 0:
            cnt = blk[3:3 + W]
            mv = blk[off_mv:off_p2l].reshape(W, cap, 4)
            incoming = [tuple(m) for m in mv[self.rank, :int(cnt[self.rank])].tolist()]
            self.swap_moves += total
        self.lane.read_device(rounds, incoming)

    def step(self, layer: LayerState, x, topk_ids, topk_weights, n_tokens=None, bucket_max=None):
        """x [n, H] this rank's real tokens, topk_ids [n, K] LOGICAL expert ids, topk_weights [n, K].
        n_tokens: exact per-rank token counts (host list, identical on every rank); or bucket_max:
        an upper bound of every rank's count that is identical on every rank (a serving framework's
        synchronized batch sizes) -- the exact counts are then gathered only when swap needs them.
        Returns y [n, H]."""
        try:
            return self._step(layer, x, topk_ids, topk_weights, n_tokens, bucket_max)
        except BaseException:
            # the head may have advanced the device counters for a step whose ops did not run: re-sync first
            self._step_synced = False
            raise

    def step_slot(self, layer):
        return layer.layer_id % 256

    def sync_step_counters(self):
        """Make the host counters (ops' run ids, lane epoch) and the device step counters equal, idle device. Never
        backwards: both sides take the larger value (a capture advances the host only, a graph replay the device;
        every consumer compares with >=, so a jump forward is safe and a rewind is not)."""
        if not self.step_state:
            return
        torch.cuda.synchronize()
        dev = [int(v) for v in C.step_counters_read()]
        d = max(int(self.comm.dispatch_op.run_id()), dev[0])
        r = max(int(self.comm.combine_op.run_id()), dev[1])
        e = max(int(self.lane.epoch) if self.lane is not None else 0, dev[2])
        self.comm.dispatch_op.advance_run_id(d - int(self.comm.dispatch_op.run_id()))
        self.comm.combine_op.advance_run_id(r - int(self.comm.combine_op.run_id()))
        if self.lane is not None:
            self.lane.epoch = e
        C.step_sync_counters(d, r, e)
        torch.cuda.synchronize()
        self._step_synced = True

    def _step(self, layer: LayerState, x, topk_ids, topk_weights, n_tokens, bucket_max):
        n = int(x.shape[0])
        self.warmup_lane(layer)
        if self.step_state and not self._step_synced:
            self.sync_step_counters()
        y = self._graph_step(layer, x, topk_ids, topk_weights, n_tokens)
        if y is not None:
            return y
        # deferred-verdict step (a forward is open): no routing count is read on the host below; `kept`: the
        # aborted pass of this forward already applied this layer's swap decision (pads as it did, no decision)
        dfr = self.deferred and self._fwd_open
        ordinal, kept = -1, False
        if dfr:
            ordinal = self._fwd_layer
            self._fwd_layer += 1
            if ordinal == 0:
                self._fwd_state = layer
            kept = ordinal <= self._redo_skip
        nt_dev = None
        if n_tokens is not None:
            n_tokens = [int(v) for v in n_tokens]
            assert n == n_tokens[self.rank], (n, n_tokens)
            S_b = self.bucket_of(max(n_tokens))
        else:
            assert bucket_max is not None and n <= int(bucket_max), (n, bucket_max)
            S_b = self.bucket_of(int(bucket_max))
            if self.lane is not None:                # the pad histogram needs every rank's real count
                cnt = torch.full((1,), n, dtype=torch.int64, device="cuda")
                allc = torch.empty(self.W, dtype=torch.int64, device="cuda")
                dist.all_gather_into_tensor(allc, cnt, group=self.group)
                if dfr:
                    nt_dev = allc                    # the device decision reads them there
                else:
                    n_tokens = allc.tolist()
        layer.refresh_pads_if_needed()
        xb, ids, w = self.x_buf[:S_b], self.ids_buf[:S_b], self.w_buf[:S_b]
        if n:
            xb[:n].copy_(x)
            ids[:n].copy_(topk_ids)
            w[:n].copy_(topk_weights)
        if n < S_b:
            xb[n:].zero_()
            ids[n:].copy_(layer.pad_rows(n, S_b, aborted_pass=kept))
            w[n:].zero_()
        lane = self.lane
        if lane is not None and nt_dev is None:
            nt_dev = self._upload_counts(n_tokens, dfr, ordinal)
        y = self._step_body(layer, S_b, n_tokens, nt_dev, dfr, ordinal, kept)
        self.steps += 1
        if self.step_state:
            C.step_release()
        if self._reprime and not dfr:                # a growth happened in this step: recapture graphs
            self.prime(layer)
        return y[:n]

    def _graph_step(self, layer, x, topk_ids, topk_weights, n_tokens):
        """Replay of the layer's captured graph (capture_graphs): staged inputs and pads, count upload, one replay,
        then the host bookkeeping the body does when it runs eagerly. None = run the step eagerly."""
        if not (self.layer_graph and self._graphs and self.deferred and self._fwd_open) or n_tokens is None:
            return None
        if self._capturing or self._graph_gen != self.comm.generation:
            return None
        ordinal = self._fwd_layer
        if ordinal <= self._redo_skip:               # the redo of an aborted forward runs eagerly
            return None
        n_tokens = [int(v) for v in n_tokens]
        S_b = self.bucket_of(max(n_tokens))
        entry = self._graphs.get((layer.layer_id, S_b))
        if entry is None:
            return None
        graph, y = entry
        n = int(x.shape[0])
        self._fwd_layer += 1
        if ordinal == 0:
            self._fwd_state = layer
        layer.refresh_pads_if_needed()
        xb, ids, w = self.x_buf[:S_b], self.ids_buf[:S_b], self.w_buf[:S_b]
        # one kernel: copy-engine copies between kernels leave the GPU idle for their scheduling latency
        C.graph_stage_in(x.contiguous(), topk_ids.contiguous(), topk_weights.float().contiguous(),
                         layer.pad_rows(n, S_b).contiguous(), xb, ids, w)
        if self.lane is not None:
            self._upload_counts(n_tokens, True, ordinal)
        # wait for the replay k layers back before this one, so the launch queue never holds more than k layers of work
        ev = self._ahead_ev[self._ahead_i % self._ahead_k]
        if self._ahead_i >= self._ahead_k:
            ev.synchronize()
        graph.replay()
        ev.record()
        self._ahead_i += 1
        self.graph_replays += 1
        # what the captured body did on the host when it was recorded, once per replay
        self.comm.dispatch_op.advance_run_id(1)
        self.comm.combine_op.advance_run_id(1)
        if self.lane is not None:
            self.lane.epoch += 1
        self.steps += 1
        # y is this graph's own output (alive with the graph, so no other graph's capture reused its memory) and is
        # rewritten only by the next replay of the same (layer, bucket), a forward later
        return y[:n]

    def capture_graphs(self, layers):
        """Collective, idle device, after the warm-up: capture one graph per (layer, power-of-two bucket) of the
        layer-step body (every rank, same order: the graphs hold NCCL and NVSHMEM collectives). `layers` = the
        model's MoE LayerStates in forward order (the index is the layer's ordinal in a forward)."""
        if not self.layer_graph:
            return 0
        import time
        t0 = time.time()
        self._graph_layers = list(layers)
        self._graphs.clear()
        # the caching allocator releases a pool once its last graph is gone and asserts on a capture into it: every
        # capture pass (incl. the recapture after a growth) takes a new pool
        self._graph_pool = torch.cuda.graph_pool_handle()
        self.sync_step_counters()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        self._capturing = True
        n = 0
        # buckets above GRAPH_MAX_BUCKET stay eager
        buckets = [b for b in self.buckets if b <= constants.GRAPH_MAX_BUCKET]
        try:
            for S_b in buckets:
                n_tokens = [S_b] * self.W
                for ordinal, layer in enumerate(layers):
                    self.begin_forward()
                    self._fwd_layer = ordinal
                    if self.lane is not None:
                        self._upload_counts(n_tokens, True, 0)   # outside the graph, as in a replay
                        torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self._graph_pool, capture_error_mode="thread_local"):
                        y = self._step_body(layer, S_b, n_tokens, self._nt_dev if self.lane is not None else None,
                                            True, ordinal, False)
                    if self.step_state:
                        C.step_release()
                    self.abandon_forward()
                    self._graphs[(layer.layer_id, S_b)] = (g, y)
                    n += 1
        finally:
            self._capturing = False
            self._nt_key = None
        self._graph_gen = self.comm.generation
        # the capture advanced the host counters only: bring the device ones up (never backwards)
        self.sync_step_counters()
        dist.barrier(group=self.group)
        if self.rank == 0:
            self.log(f"[zepp] layer graphs: {n} captured ({len(layers)} layers x {len(buckets)} buckets) in "
                     f"{time.time() - t0:.1f} s", flush=True)
        return n

    def _upload_counts(self, n_tokens, dfr, ordinal):
        """The per-rank token counts on the device (the swap decision's pad histogram reads them there)."""
        if dfr:
            # one upload per forward (every layer of a forward has the same counts): without the per-layer
            # planning sync, rewriting the pinned buffer could race the previous upload
            key = tuple(n_tokens)
            if key != self._nt_key:
                assert ordinal == 0, "per-rank token counts changed inside a forward"
                self._nt_pin.numpy()[:] = n_tokens
                self._nt_dev.copy_(self._nt_pin, non_blocking=True)
                self._nt_key = key
        else:
            self._nt_pin.numpy()[:] = n_tokens
            self._nt_dev.copy_(self._nt_pin, non_blocking=True)
            self._nt_key = None
        return self._nt_dev

    def _step_body(self, layer, S_b, n_tokens, nt_dev, dfr, ordinal, kept):
        """The layer-step from the bucket buffers on (everything a captured layer graph holds): the head of the
        device step state, routing + both exchanges, the swap decision and lane, planning, dispatch, activation,
        combine. Returns the [S_b, H] output."""
        cfg = self.cfg
        if self.step_state:
            C.step_head(self.step_slot(layer), 1 if self.lane is not None else 0)
        xb, ids, w = self.x_buf[:S_b], self.ids_buf[:S_b], self.w_buf[:S_b]
        planner = self.planners[S_b]
        planner.set_routing(ids, w)
        loads = planner.exchange_loads()
        lane = self.lane
        if lane is not None:
            C.swap_decide(loads, nt_dev, S_b, self.K, layer.p2l, layer.l2p, layer.lcnts, self.L, self.nlp,
                          self._swap_band_c(kept), constants.SWAP_MAX_MOVES, 32, self._sd_out)
            if dfr:
                self._moves_acc.add_(self._sd_out[1:2])      # read once, at the end of the forward
            else:
                self._sd_pin.copy_(self._sd_out, non_blocking=True)
                self._sd_ev.record()
            lane.bind(layer.slots)
            lane.arm_device(self._sd_out, always=dfr)
            if not kept:
                layer.rebuild_pads_device(self._sd_out, save=dfr)
        plan = planner.plan(loads, layer.tables())
        if dfr:
            self.comm.plan_meta(plan, planner._probs_own.view(-1), verdict=(ordinal, self._take_force(ordinal)))
        else:
            self._plan_meta_checked(plan, planner, layer)
        if lane is not None and not dfr:
            # the planning step above already synchronized with the device; the block has landed
            self._read_swap_decision()
        comm = self.comm
        comm.prep(used_only=True)
        gate = None
        if lane is not None:
            gate = lane.staged_phase_before(0)       # push W1 out + point the GEMM at the incoming staging
            if gate is None and self._warmup_gate is not None:
                gate = self._warmup_gate[0]          # warm-up: the gated GEMM path without a swap
        h = comm.dispatch_gemm(xb, gate, w1=layer.slots.w1)
        comm.issue_combine_meta_late(plan)
        if lane is not None:
            lane.commit_after(0)                     # staging -> slot, off the gated path
        m = h.shape[0]
        if cfg.shape.act == "swiglu":
            a = (comm.activation(h, cfg.shape, self.act_buf[:m]) if hasattr(comm, "activation")
                 else activation(h, cfg.shape, out=self.act_buf[:m]))
        else:
            a = activation(h, cfg.shape)
        gate1 = None
        if lane is not None:
            gate1 = lane.staged_phase_before(1)
            if gate1 is None and self._warmup_gate is not None:
                gate1 = self._warmup_gate[1]
        y = comm.gemm_combine(a, gate1, w2=layer.slots.w2)
        if lane is not None:
            lane.join_push()                         # side-stream weight pushes joined inside the layer-step
            lane.commit_after(1)
        return y


class ServingMoE:
    """One layer's view of the shared communication state."""

    def __init__(self, shared: SharedComm, layer: LayerState):
        self.shared, self.layer = shared, layer

    def forward(self, x, topk_ids, topk_weights, n_tokens):
        return self.shared.step(self.layer, x, topk_ids, topk_weights, n_tokens)
