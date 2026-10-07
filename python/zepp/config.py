"""Model shapes and the layer configuration (the three public knobs)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    num_experts: int
    topk: int
    hidden: int
    ffn_hidden: int
    act: str = "gelu"               # "gelu": one ffn matrix; "swiglu": gate|up (N = 2 * ffn_hidden), silu(gate) * up

    def __post_init__(self):
        assert self.act in ("gelu", "swiglu"), self.act

    @property
    def ffn1(self) -> int:
        """N of the first GEMM (rows of a w1 slot): 2 * ffn_hidden with SwiGLU."""
        return self.ffn_hidden * (2 if self.act == "swiglu" else 1)

    @property
    def chunk_bytes(self) -> int:
        """Bytes of one bf16 token row on the wire."""
        return self.hidden * 2


def tokens_per_rank(budget_mib: float, shape: ModelShape) -> int:
    """Tokens per rank for a pre-top-k send budget of `budget_mib` MiB per rank, rounded to a
    multiple of top-k (at least top-k)."""
    budget_bytes = budget_mib * (1 << 20)
    return max(shape.topk, round(budget_bytes / (shape.chunk_bytes * shape.topk)) * shape.topk)


SHAPES = {
    "qwen3": ModelShape(num_experts=128, topk=8, hidden=4096, ffn_hidden=1536),
    "k2": ModelShape(num_experts=384, topk=8, hidden=7168, ffn_hidden=2048),
}


@dataclass(frozen=True)
class EPMoEConfig:
    shape: ModelShape
    ranks: int                      # expert-parallel world size
    ranks_per_node: int
    max_tokens_per_rank: int        # tokens per rank per forward (sizes the buffers)
    comm_strategy: str = "overlap"  # "overlap": dispatch/combine fused into the GEMMs; "direct": plain all-to-all + GEMM
    swap: bool = False              # band-triggered intra-node expert swap, scheduled across the two GEMMs
    router_c: float = 0.25          # capacity slack C of the routing constraint (paper eq. 2)
    redundant_slots_per_rank: int = 2
    sm_margin: int = 8              # SMs left free for the communication kernels
    gpu_plan: bool = True           # overlap on >= 2 nodes: planning and communication issued on the GPU (no host
                                    # wait inside a layer-step; with swap, the 3D swap schedule on the GPU); False:
                                    # the host plans (swap decision, metadata readback) and issues the swap copies

    def __post_init__(self):
        assert self.comm_strategy in ("overlap", "direct"), self.comm_strategy
        assert self.shape.num_experts % self.ranks == 0, "num_experts must divide by ranks"
        assert self.ranks % self.ranks_per_node == 0
        assert 0.0 <= self.router_c <= 1.0
        assert self.max_tokens_per_rank % self.shape.topk == 0

    @property
    def nodes(self) -> int:
        return self.ranks // self.ranks_per_node

    @property
    def slots_per_rank(self) -> int:
        """Physical expert slots per rank: the home shard plus the redundant slots."""
        return self.shape.num_experts // self.ranks + self.redundant_slots_per_rank

    @property
    def groups_per_rank(self) -> int:
        """Virtual GEMM groups per rank: the slots plus one always-empty pad group at
        local index 0 (the fused ops' weight gate needs the gated groups to start at 1)."""
        return self.slots_per_rank + 1

    @property
    def virtual_experts(self) -> int:
        return self.ranks * self.groups_per_rank
