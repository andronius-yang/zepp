"""zepp: expert-parallel MoE layer for multi-node A100 systems."""
__version__ = "0.1.0"

from .config import SHAPES, EPMoEConfig, ModelShape  # noqa: F401
from .layer import EPMoE  # noqa: F401
from .serving import LayerState, ServingMoE, SharedComm  # noqa: F401
