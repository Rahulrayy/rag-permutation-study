"""Shared fixtures."""

import pytest

from src.chunks import Chunk


@pytest.fixture
def chunks():
    return [
        Chunk(idx=i, title=f"title{i}", text=f"body of paragraph {i}", rank=i,
              is_gold=i in (2, 5))
        for i in range(10)
    ]
