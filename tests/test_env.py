"""The .env loader. A key that silently fails to load reads as 'not set', which
points at the wrong problem, so the parsing edge cases are worth pinning."""

import pytest

from src import env
from src.env import load_env


@pytest.fixture(autouse=True)
def _fresh_loader_state(monkeypatch):
    """Each test gets its own idempotence set, or the first test to load a path
    would suppress every later one."""
    monkeypatch.setattr(env, "_loaded", set())


def _write(tmp_path, body):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_key_value_pairs(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = _write(tmp_path, "GROQ_API_KEY=gsk_secret\n")
    assert load_env(path) == {"GROQ_API_KEY": "gsk_secret"}


def test_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = _write(tmp_path, "# a comment\n\n  \nGROQ_API_KEY=abc\n")
    assert load_env(path) == {"GROQ_API_KEY": "abc"}


def test_tolerates_an_export_prefix(tmp_path, monkeypatch):
    """Copying the line straight out of the README should work."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = _write(tmp_path, "export GROQ_API_KEY=abc\n")
    assert load_env(path) == {"GROQ_API_KEY": "abc"}


@pytest.mark.parametrize("body", ['K="abc"\n', "K='abc'\n"])
def test_strips_one_pair_of_quotes(tmp_path, monkeypatch, body):
    """A key pasted with its quotes would otherwise reach the Authorization
    header with them attached and come back as an opaque 401."""
    monkeypatch.delenv("K", raising=False)
    assert load_env(_write(tmp_path, body)) == {"K": "abc"}


def test_survives_a_utf8_bom(tmp_path, monkeypatch):
    """PowerShell's `>` and Notepad both write a BOM. Glued to the first key
    name it produces 'not set' with nothing on screen to explain why."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=abc\n", encoding="utf-8-sig")
    assert load_env(path) == {"GROQ_API_KEY": "abc"}


def test_an_exported_variable_beats_the_file(tmp_path, monkeypatch):
    """A key set deliberately in the current shell must not lose to a stale
    file that the user has forgotten about."""
    monkeypatch.setenv("GROQ_API_KEY", "from_shell")
    load_env(_write(tmp_path, "GROQ_API_KEY=from_file\n"))
    import os

    assert os.environ["GROQ_API_KEY"] == "from_shell"


def test_override_reverses_that(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from_shell")
    assert load_env(_write(tmp_path, "GROQ_API_KEY=from_file\n"), override=True) == {
        "GROQ_API_KEY": "from_file"
    }


def test_a_missing_file_is_not_an_error(tmp_path):
    """The variable may be exported instead. Saying so is the caller's job."""
    assert load_env(tmp_path / "nope.env") == {}


def test_a_missing_file_does_not_poison_a_later_load(tmp_path, monkeypatch):
    """Creating the file and retrying in the same session must work."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = tmp_path / ".env"
    assert load_env(path) == {}
    path.write_text("GROQ_API_KEY=abc\n", encoding="utf-8")
    assert load_env(path) == {"GROQ_API_KEY": "abc"}


def test_second_load_is_a_no_op(tmp_path, monkeypatch):
    """Re-reading mid-run would let an edit swap credentials under a job that is
    already half paid for."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = _write(tmp_path, "GROQ_API_KEY=abc\n")
    assert load_env(path) == {"GROQ_API_KEY": "abc"}
    assert load_env(path) == {}


def test_a_malformed_line_names_the_file_and_line(tmp_path):
    """Failing loudly beats a key that is quietly absent."""
    path = _write(tmp_path, "GROQ_API_KEY\n")
    with pytest.raises(ValueError, match=r"\.env:1: expected KEY=VALUE"):
        load_env(path)


def test_an_unfilled_placeholder_reads_as_absent(tmp_path, monkeypatch):
    """The shipped .env has `GROQ_API_KEY=` with no value. That must reach
    GroqGenerator as falsy so it raises its own clear message."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert load_env(_write(tmp_path, "GROQ_API_KEY=\n")) == {"GROQ_API_KEY": ""}
    import os

    assert not os.environ["GROQ_API_KEY"]
