"""`ZeppFusedMoE`: SGLang's FusedMoE with the zepp layer as the execution path.

SGLang's machinery is reused for what it does well: the parameter tensors (physical expert
slots, contiguous per rank) and the checkpoint loader that fills every physical replica of a
logical expert through the expert-location metadata. zepp provides placement, routing, the
fused dispatch/combine and the swap. The parameters are views into the layer's slot storage
(pad group 0 at index 0), so the weights exist once.
"""
import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

from zepp import LayerState

from . import runtime


class ZeppFusedMoE(FusedMoE):
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, layer_id, params_dtype=None,
                 quant_config=None, prefix="", **kwargs):
        assert quant_config is None, "zepp runs bf16 experts only"
        params_dtype = params_dtype or torch.get_default_dtype()
        super().__init__(num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,
                         layer_id=layer_id, top_k=top_k, params_dtype=params_dtype, quant_config=None, prefix=prefix,
                         **kwargs)
        cfg = runtime.layer_config()
        shape = cfg.shape
        assert (shape.num_experts, shape.topk, shape.hidden, shape.ffn_hidden) == \
               (num_experts - cfg.ranks * cfg.redundant_slots_per_rank, top_k, hidden_size, intermediate_size), \
            "ZEPP_CONFIG shape does not match the model"
        assert self.num_local_experts == cfg.slots_per_rank, (self.num_local_experts, cfg.slots_per_rank)
        assert self.moe_tp_size == 1, "zepp needs ep_size == tp_size"
        gpe = cfg.groups_per_rank
        dev = self.w13_weight.device
        # slot storage with the pad group at index 0; SGLang's parameters become the [1:] views
        self._w1_base = torch.zeros(gpe, shape.ffn1, hidden_size, dtype=params_dtype, device=dev)
        self._w2_base = torch.zeros(gpe, hidden_size, shape.ffn_hidden, dtype=params_dtype, device=dev)
        for name, base in (("w13_weight", self._w1_base), ("w2_weight", self._w2_base)):
            p = getattr(self, name)
            assert tuple(p.shape) == tuple(base[1:].shape), (name, tuple(p.shape), tuple(base.shape))
            p.data = base[1:]                     # same Parameter object (keeps weight_loader & attrs), our storage
        torch.cuda.empty_cache()                  # the replaced storage must not stay reserved (KV pool sizing)
        group = get_tp_group().device_group
        self._shared = runtime.shared(group, params_dtype)          # collective on the first layer
        with torch.device("cpu"):                                   # host tables and pinned mirrors
            self._state = LayerState(layer_id, cfg, runtime.placement(layer_id), group.rank(), dtype=params_dtype,
                                     w1=self._w1_base, w2=self._w2_base, device=dev)
        self._checked = False
        runtime.register_layer(self, self._state)
        self._hook_after_load()

    def _hook_after_load(self):
        """Warm up zepp once the model is loaded: SGLang's loader calls quant_method.process_weights_after_loading
        on every module after every weight is in place; the last zepp layer to pass triggers the warm-up."""
        qm = getattr(self, "quant_method", None)
        if qm is None:
            return
        orig = qm.process_weights_after_loading

        def after_loading(layer, _orig=orig):
            _orig(layer)
            runtime.layer_loaded(layer)
        qm.process_weights_after_loading = after_loading

    def _check_aliasing(self):
        """The loader wrote in place into our slot storage; recover once if SGLang re-wrapped it."""
        for name, base in (("w13_weight", self._w1_base), ("w2_weight", self._w2_base)):
            p = getattr(self, name)
            if p.data_ptr() != base[1:].data_ptr():
                base[1:].copy_(p.data)
                p.data = base[1:]
        self._checked = True

    def forward(self, hidden_states, topk_idx, topk_weights, forward_batch):
        """hidden_states [n_pad, H]: this rank's tokens (under DP attention SGLang pads every rank's batch
        to the longest rank; padded rows carry top-k id -1)."""
        if not self._checked:
            self._check_aliasing()
        if not runtime.warmed():                     # the loader hook did not fire: warm up before the first step
            runtime.warmup(self._state)
        n_pad = int(hidden_states.shape[0])
        n = getattr(forward_batch, "num_token_non_padded_cpu", None)
        n = n_pad if n is None else int(n)
        assert n <= n_pad, (n, n_pad)
        shared, group = self._shared, self._shared.group
        counts = runtime.exact_counts(forward_batch, n, group, self.layer_id)
        H = hidden_states.shape[1]
        # the exact per-rank counts of this pass (gathered once, at the first MoE layer): the layer needs
        # them for the swap decision's pad correction, so it never gathers them again per layer
        y = shared.step(self._state, hidden_states[:n], topk_idx[:n].to(torch.int32), topk_weights[:n],
                        n_tokens=counts)
        if n != n_pad:
            out = torch.zeros(n_pad, H, dtype=y.dtype, device=y.device)
            out[:n] = y
            y = out
        return y

    def report(self):
        return self._shared.report()
