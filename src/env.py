"""Read a local ``.env`` into the process environment.

A key cannot live in ``configs/*.yaml`` -- those are committed, and this repo is
public -- and an ``export`` does not survive a new terminal on Windows, which is
where this project runs. A gitignored ``.env`` is what is left.

Hand-rolled rather than taking python-dotenv: this is thirty lines against a new
dependency, and every addition to this environment is another chance for pip to
resolve a PyPI torch over the cu128 build (HANDOFF Sec. 6). The format supported
is the intersection everyone agrees on -- ``KEY=value``, ``#`` comments, blank
lines, an optional ``export`` prefix, one optional pair of surrounding quotes.
No interpolation, no multi-line values.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repo root: this file is ``<root>/src/env.py``.
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

#: Paths already applied, so a long run cannot pick up a mid-flight edit to the
#: file and change credentials under a job that is already half paid for.
_loaded: set[Path] = set()


def load_env(
    path: str | Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Apply ``KEY=VALUE`` lines from ``path`` to ``os.environ``.

    Returns only what it actually set, so a caller can log "loaded GROQ_API_KEY
    from .env" without ever holding the value.

    A missing file is not an error: the variable may be exported in the shell,
    and the caller that needs it is responsible for saying so clearly if it is
    absent from both.

    An existing environment variable wins unless ``override`` is set. A key
    exported deliberately in the current shell should beat a stale file, not
    silently lose to it.
    """
    env_path = (Path(path) if path is not None else DEFAULT_ENV_PATH).resolve()
    if env_path in _loaded and not override:
        return {}
    if not env_path.is_file():
        # Deliberately not marked loaded: creating the file and retrying in the
        # same session (a notebook, say) should work.
        return {}

    applied: dict[str, str] = {}
    # utf-8-sig: PowerShell's `>` and Notepad both write a BOM by default, and a
    # BOM glued to the first key name turns a correct file into "GROQ_API_KEY is
    # not set" with nothing on screen to explain why.
    text = env_path.read_text(encoding="utf-8-sig")

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(
                f"{env_path}:{lineno}: expected KEY=VALUE, got {raw.strip()!r}"
            )

        key = key.strip()
        value = value.strip()
        # Strip one matching pair of quotes: a key pasted with the quotes from a
        # docs snippet would otherwise carry them into the Authorization header
        # and come back as an opaque 401.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        if key in os.environ and not override:
            continue
        os.environ[key] = value
        applied[key] = value

    _loaded.add(env_path)
    return applied
