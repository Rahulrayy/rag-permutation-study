"""Pruner arms. Importing this package populates the registry in `base`.

Add an arm: one module here, one Pruner subclass with a `name`, one import below.
That is the whole cost of a ninth arm.
"""

from .base import Pruner, get_pruner, registered_arms, validate_selection
from .full import Full
from .llm_pruner import LLMPruner
from .llmlingua2 import LLMLingua2
from .loo_oracle import LOOOracle
from .nocontext import NoContext
from .placebo_pos import PlaceboPositional
from .provence import Provence
from .random_drop import RandomDrop
from .rerank_topk import RerankTopK

__all__ = [
    "Pruner",
    "get_pruner",
    "registered_arms",
    "validate_selection",
    "Full",
    "LLMPruner",
    "LLMLingua2",
    "LOOOracle",
    "NoContext",
    "PlaceboPositional",
    "Provence",
    "RandomDrop",
    "RerankTopK",
]
