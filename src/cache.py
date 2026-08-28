"""Content-hash cache for generations.

Non-negotiable per plan Sec. 5: you will rerun the analysis twenty times and you
should never pay for the same generation twice. Keyed on
``sha256(model, prompt, decode_params)`` so the key is a pure function of what
determines the output -- never on query id, arm or permutation index, which would
miss the many cells that resolve to an identical prompt.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cache_key(model: str, prompt: str, decode_params: dict[str, Any]) -> str:
    """Stable content hash. Sorted keys so dict ordering can't change the key."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "decode": decode_params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachedGeneration:
    text: str
    # Log-prob of the reference answer sequence, when scored rather than decoded.
    # This is why the primary generator is local: hosted APIs generally don't
    # expose it, and the LOO oracle needs it (plan Sec. 4.2).
    answer_logprob: float | None = None
    meta: dict[str, Any] | None = None


class GenerationCache:
    """SQLite-backed. Safe to open from multiple processes; writes are small."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                key            TEXT PRIMARY KEY,
                model          TEXT NOT NULL,
                text           TEXT NOT NULL,
                answer_logprob REAL,
                meta           TEXT,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> CachedGeneration | None:
        row = self._conn.execute(
            "SELECT text, answer_logprob, meta FROM generations WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return CachedGeneration(
            text=row[0],
            answer_logprob=row[1],
            meta=json.loads(row[2]) if row[2] else None,
        )

    def put(self, key: str, model: str, gen: CachedGeneration) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO generations "
            "(key, model, text, answer_logprob, meta) VALUES (?, ?, ?, ?, ?)",
            (
                key,
                model,
                gen.text,
                gen.answer_logprob,
                json.dumps(gen.meta) if gen.meta else None,
            ),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GenerationCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
