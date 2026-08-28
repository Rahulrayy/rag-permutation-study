"""Ask the generator which chunks to keep. Strong, expensive, common in practice.

Watch for two failure modes when implementing: the model returning more than
`budget` indices (validate_selection will catch it), and the selection prompt
itself being order-sensitive -- which would make this arm selection, not just its
answer, a function of the ordering it was shown. Log the ordering used for
selection; it is a confound worth reporting.

STATUS: stubbed (week 2).
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner


class LLMPruner(Pruner):
    name = "llm_pruner"

    def __init__(self, generator=None, max_new_tokens: int = 64) -> None:
        self.generator = generator
        self.max_new_tokens = max_new_tokens

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        raise NotImplementedError("LLM-as-pruner not implemented (week 2)")
