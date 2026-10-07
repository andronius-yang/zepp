"""Process-wide zepp state inside an SGLang scheduler process.

`ZEPP_CONFIG` names the JSON written by `calibrate.py`: the model shape, the layer settings, the
buffer capacities and every layer's physical-slot -> expert table (`p2l`). The first MoE layer
that is constructed builds the `SharedComm` (NVSHMEM heap, fused ops, planners), before SGLang
profiles memory for its KV pool; every layer then adds its own `LayerState`.

Once the model's weights are loaded, `warmup` runs zepp's one-time warm-up (SharedComm.warmup:
every kernel and NVSHMEM primitive the steady state can launch is loaded, every bucket primed and run
once on synthetic tokens; model state unchanged) and, with the overlap strategy, captures the
per-layer CUDA graphs, before SGLang profiles memory.

`run_forward` is the forward of SGLang's TpModelWorker (the patch calls it in place of
model_runner.forward). With the overlap strategy the MoE layers check their buffer demands on the
device only and the forward's verdict is read once, here: on a violation every rank grows the
buffers and the same forward runs again (python/zepp/serving.py, module docstring). Otherwise it is
model_runner.forward. `run_forward_sample` (what the patch calls for every batch but verification
and prefill-only batches) also enqueues the sampling before it waits for the verdict.
"""
import json
import os

import torch

from zepp import EPMoEConfig, ModelShape, SharedComm
from zepp.placement import Placement, rebuild_l2p
from zepp.routing import Capacities

_CFG = None
_SHARED = None


def config():
    global _CFG
    if _CFG is None:
        path = os.environ.get("ZEPP_CONFIG")
        if not path:
            raise RuntimeError("ZEPP_CONFIG is not set (path of the calibrate.py output JSON)")
        with open(path) as f:
            _CFG = json.load(f)
    return _CFG


def layer_config() -> EPMoEConfig:
    c = config()
    sh = c["shape"]
    shape = ModelShape(sh["num_experts"], sh["topk"], sh["hidden"], sh["ffn_hidden"], act=sh.get("act", "swiglu"))
    return EPMoEConfig(shape=shape, ranks=c["ranks"], ranks_per_node=c["ranks_per_node"],
                       max_tokens_per_rank=c["max_tokens_per_rank"], comm_strategy=c.get("comm_strategy", "overlap"),
                       swap=bool(c.get("swap", False)),
                       router_c=float(c.get("router_c", 0.25)),
                       redundant_slots_per_rank=int(c.get("redundant_slots_per_rank", 2)))


def capacities() -> Capacities:
    return Capacities(**config()["caps"])


def placement(layer_id: int) -> Placement:
    c = config()
    p2l = torch.tensor(c["p2l"][str(layer_id)] if isinstance(c["p2l"], dict) else c["p2l"][layer_id], dtype=torch.int32)
    G, R = c["shape"]["num_experts"], c["ranks"]
    l2p = rebuild_l2p(p2l, G, R)
    lcnts = torch.bincount(p2l[p2l >= 0].long(), minlength=G).int()
    return Placement(p2l, l2p, lcnts, {})


_COUNTS = None
_FIRST_LAYER = None
_PREFETCH = None        # run_forward: the counts of the current forward, gathered before it started


def exact_counts(forward_batch, n_real: int, group, layer_id: int):
    """Exact per-rank real token counts of this forward pass: one all-gather per pass, taken when the
    model's first MoE layer runs and reused by the later layers (SGLang's synchronized
    `global_num_tokens_cpu` are padded sizes; `id(forward_batch)` is not a safe key, ids are reused).
    Under run_forward with the deferred verdict the gather happens before the forward (no host read inside
    it)."""
    global _COUNTS, _FIRST_LAYER
    if _PREFETCH is not None:
        assert _PREFETCH[group.rank()] == n_real, (_PREFETCH, n_real, layer_id)
        return _PREFETCH
    if _FIRST_LAYER is None or layer_id < _FIRST_LAYER:
        _FIRST_LAYER = layer_id
    if _COUNTS is None or layer_id == _FIRST_LAYER:
        W = group.size()
        cnt = torch.full((1,), n_real, dtype=torch.int64, device="cuda")
        allc = torch.empty(W, dtype=torch.int64, device="cuda")
        torch.distributed.all_gather_into_tensor(allc, cnt, group=group)
        _COUNTS = [int(v) for v in allc.tolist()]
    assert _COUNTS[group.rank()] == n_real, (_COUNTS, n_real, layer_id)
    return _COUNTS


_LAYER_STATES = {}      # id(ZeppFusedMoE) -> its LayerState, every zepp layer constructed in this process
_LOADED = set()         # ids of the zepp layers whose weights SGLang's loader has finished
_WARMED = False


def register_layer(module, state):
    _LAYER_STATES[id(module)] = state


def layer_loaded(module):
    """SGLang's loader finished `module` (process_weights_after_loading runs on every module once all
    weights are loaded): warm up when every zepp layer of the model is done. Every rank loads the same
    model in the same module order, so every rank reaches the (collective) warm-up together."""
    _LOADED.add(id(module))
    if _LAYER_STATES and set(_LAYER_STATES) <= _LOADED:
        warmup(_LAYER_STATES[id(module)])


def warmed() -> bool:
    return _WARMED


def warmup(state):
    """zepp's one-time warm-up on the process-wide SharedComm (collective; see SharedComm.warmup). Runs
    after the weights are loaded (or, if the loader hook did not fire, at the first MoE forward, before
    the layer's first real step)."""
    global _WARMED
    if _WARMED or _SHARED is None:
        return
    _WARMED = True
    # the loader runs under SGLang's default dtype / device contexts; zepp's host tables expect the
    # defaults of a forward pass
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.no_grad(), torch.device("cpu"):
            _SHARED.warmup(state)
            if _SHARED.layer_graph:
                # one graph per (MoE layer, bucket), the layers in forward order (collective)
                _SHARED.capture_graphs(sorted(_LAYER_STATES.values(), key=lambda st: st.layer_id))
    finally:
        torch.set_default_dtype(prev)
    torch.cuda.empty_cache()                   # the warm-up's temporaries must not count against the KV pool


def _prefetch_counts(forward_batch, group):
    """The forward's per-rank real token counts (the layer's `n`: num_token_non_padded_cpu), known before the
    forward so that no MoE layer reads them back on the host inside it. They are the scheduler's
    (ForwardBatch.global_num_tokens_cpu, the real token count of every attention-DP rank, gathered by SGLang's MLP
    sync for this batch and padded only later, inside model_runner.forward), so no all-gather and no host wait of
    its own; a batch without them gathers them."""
    global _PREFETCH
    _PREFETCH = None
    n = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if n is None:
        return
    g = getattr(forward_batch, "global_num_tokens_cpu", None)
    if g is not None and len(g) == group.size():
        host = [int(v) for v in g]
        assert host[group.rank()] == int(n), (host, int(n), group.rank())
        _PREFETCH = host
        return
    cnt = torch.full((1,), int(n), dtype=torch.int64, device="cuda")
    allc = torch.empty(group.size(), dtype=torch.int64, device="cuda")
    torch.distributed.all_gather_into_tensor(allc, cnt, group=group)
    _PREFETCH = [int(v) for v in allc.tolist()]


MAX_REDO = 8


def run_forward(model_runner, forward_batch, pp_proxy_tensors=None):
    """model_runner.forward with zepp's capacity verdict deferred to the end of the forward (collective).

    begin_forward; model_runner.forward; end_forward; forward_check (waits for the forward). On a violation:
    quiesce the wire, grow the buffers (identical on every rank), re-prime with the warm-up, and run the same
    forward_batch again through model_runner._forward_raw: forward_pass_id is not advanced, the expert
    distribution recorder records nothing (its hooks are disabled), and the EPLB manager is not stepped, so the
    pass counts once. The forward's KV writes are recomputed by the second pass. Returns what
    model_runner.forward returns."""
    global _PREFETCH
    sh = _SHARED
    if sh is None or not sh.deferred:
        return model_runner.forward(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder

    _prefetch_counts(forward_batch, sh.group)
    try:
        sh.begin_forward()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        sh.end_forward()
        for _ in range(MAX_REDO):
            abort = sh.forward_check()
            if abort is None:
                return out
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)   # the warm-up's host tables expect the forward defaults
            try:
                with torch.no_grad(), torch.device("cpu"):
                    sh.recover(abort)
            finally:
                torch.set_default_dtype(prev)
            torch.cuda.empty_cache()                 # warm-up temporaries (the device is idle here)
            sh.begin_forward()
            with get_global_expert_distribution_recorder().disable_this_region():
                out = model_runner._forward_raw(forward_batch, False, pp_proxy_tensors, False, 1)
            sh.end_forward()
        raise RuntimeError(f"zepp: the forward still exceeds the buffers after {MAX_REDO} growths")
    except BaseException:
        sh.abandon_forward()
        raise
    finally:
        _PREFETCH = None


def run_forward_sample(model_runner, forward_batch, pp_proxy_tensors, sample):
    """forward_batch_generation's forward + sampling for zepp (the SGLang hook calls it when present): returns
    ((logits_output, can_run_cuda_graph), next_token_ids). `sample(logits_output)` enqueues the sampling of the
    forward BEFORE waiting for its capacity verdict (forward_check), so the GPU runs the sampling while the host
    waits instead of idling until the host has checked the verdict and launched the sampling. The verdict is still
    read before anything uses the sampled ids (forward_batch_generation returns only after the check); a forward
    that aborts runs again (forward and sampling), its first sampled ids are discarded. Without the deferred
    verdict (direct strategy): run_forward, then the sampling."""
    global _PREFETCH
    sh = _SHARED
    if sh is None or not sh.deferred:
        out = run_forward(model_runner, forward_batch, pp_proxy_tensors)
        return out, sample(out[0])
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder

    _prefetch_counts(forward_batch, sh.group)
    try:
        sh.begin_forward()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        sh.end_forward()
        nxt = sample(out[0])                      # stream-ordered after the forward; not read before the check
        for _ in range(MAX_REDO):
            abort = sh.forward_check()
            if abort is None:
                return out, nxt
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                with torch.no_grad(), torch.device("cpu"):
                    sh.recover(abort)
            finally:
                torch.set_default_dtype(prev)
            torch.cuda.empty_cache()
            sh.begin_forward()
            with get_global_expert_distribution_recorder().disable_this_region():
                out = model_runner._forward_raw(forward_batch, False, pp_proxy_tensors, False, 1)
            sh.end_forward()
            nxt = sample(out[0])
        raise RuntimeError(f"zepp: the forward still exceeds the buffers after {MAX_REDO} growths")
    except BaseException:
        sh.abandon_forward()
        raise
    finally:
        _PREFETCH = None


def shared(group, dtype=torch.bfloat16) -> SharedComm:
    """The process-wide SharedComm (collective on first call)."""
    global _SHARED
    if _SHARED is None:
        cfg = layer_config()
        assert group.size() == cfg.ranks, (group.size(), cfg.ranks)
        # SGLang constructs the model with CUDA as the default device; zepp's host-side buffers
        # (pinned mirrors, planner staging) rely on the CPU default, so pin it for the construction.
        with torch.device("cpu"):
            _SHARED = SharedComm(cfg, group, capacities(), dtype=dtype)
    return _SHARED
