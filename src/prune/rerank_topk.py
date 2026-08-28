"""Cross-encoder rerank, keep top-k. The baseline everyone should beat.

Default denominator arm for OAE (see configs/main.yaml -> metrics.baseline_arm).

STATUS: stubbed (week 2).
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner, validate_selection


class RerankTopK(Pruner):
    name = "rerank_topk"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = model
        self._encoder = None

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        raise NotImplementedError("cross-encoder rerank not implemented (week 2)")
