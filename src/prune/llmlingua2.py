"""LLMLingua-2: token-level compression. Published method, different family.

Granularity mismatch is the trap: this compresses *within* chunks rather than
selecting whole ones, so a keep-k budget is not directly comparable. Report
against **input-token count**, not k (plan Sec. 4.3), and decide before the run
how a token-compressed context is permuted -- the honest choice is to permute the
surviving chunk-level units, not individual tokens.

STATUS: stubbed (week 2).
"""

from __future__ import annotations

from typing import Sequence

from ..chunks import Chunk
from .base import Pruner


class LLMLingua2(Pruner):
    name = "llmlingua2"

    def __init__(self, rate: float = 0.33) -> None:
        self.rate = rate

    def select(self, query: str, chunks: Sequence[Chunk], budget: int) -> list[int]:
        raise NotImplementedError("LLMLingua-2 not implemented (week 2)")
