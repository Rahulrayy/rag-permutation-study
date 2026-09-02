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


def artifact_key(kind: str, params: dict[str, Any]) -> str:
    """Stable content hash for a non-generation artifact.

    Same discipline as `cache_key`: the key must be a pure function of
    everything that determines the output, and of nothing else. `params` should
    therefore carry the checkpoint, every setting that changes the result, and
    the *content* being processed -- never a chunk index or a query id, which
    identify a slot rather than an input. Keying on a slot is what silently
    invalidated the llmlingua2 arm (ANALYSIS_PLAN Sec. 9, 2026-09-01).
    """
    payload = json.dumps({"kind": kind, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ArtifactCache:
    """Disk cache for expensive work that is not a generation.

    Selections and compressions are as costly to recompute as generations and,
    unlike them, were only ever memoized in the process. That is invisible until
    a long run is interrupted, at which point every encoder pass is re-paid: on
    the 2Wiki config, where Provence and LLMLingua-2 run on CPU to leave the GPU
    to generation, that is ~90 minutes per restart.

    Deliberately a separate table in the same SQLite file, so one path in the
    config carries both and a run's whole history moves as one artifact.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                key        TEXT PRIMARY KEY,
                kind       TEXT NOT NULL,
                payload    TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT payload FROM artifacts WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, kind: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts (key, kind, payload) VALUES (?, ?, ?)",
            (key, kind, json.dumps(value, ensure_ascii=False)),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ArtifactCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


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

    def created_at(self, key: str) -> str | None:
        """When this key was first generated, as SQLite's UTC timestamp.

        Only the determinism audit needs this, and it needs it to say something
        the identical-rate alone cannot: that the answers it re-issued today
        were paid for on *earlier* days. A same-session 50/50 and a cross-day
        50/50 print the same number and mean very different things.
        """
        row = self._conn.execute(
            "SELECT created_at FROM generations WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

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
