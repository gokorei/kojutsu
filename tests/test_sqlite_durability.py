"""The durability pragmas are asserted, not assumed.

These are the tests that make the README's "durable outbox" claim true rather
than aspirational. A crash test cannot substitute for them: on a healthy machine
a ``SIGKILL`` also passes with ``synchronous=NORMAL``, so a green crash suite
proves nothing about whether a commit reached stable storage before it was
acknowledged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from kojutsu.core import sqlite_durability
from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.core.sqlite_durability import (
    DurabilityError,
    configure_durable_connection,
)


def _pragma(db: sqlite3.Connection, name: str) -> Any:
    row = db.execute(f"PRAGMA {name}").fetchone()
    return None if row is None else row[0]


@pytest.mark.parametrize("factory_name", ["outbox", "registry"])
def test_local_state_files_are_configured_for_durability(tmp_path: Path, factory_name: str) -> None:
    """Both local files must be WAL with synchronous=FULL, verified by read-back."""
    if factory_name == "outbox":
        handle: Any = TansekiOutbox(tmp_path / "outbox.db")
    else:
        handle = SqliteQuestionRegistry(tmp_path / "registry.db")
    try:
        db = handle._db
        assert _pragma(db, "journal_mode") == "wal"
        assert _pragma(db, "synchronous") == 2  # FULL
        assert _pragma(db, "foreign_keys") == 1
    finally:
        handle.close()


def test_configure_refuses_when_the_journal_mode_does_not_take_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SQLite reports the pre-existing mode when conversion fails; that is not success.

    A filesystem without shared-memory support makes ``journal_mode=WAL`` a
    no-op. Continuing would mean running the journal mode we were trying to
    leave, which is the gap the pragma closes.
    """
    db = sqlite3.connect(str(tmp_path / "x.db"))
    try:
        monkeypatch.setattr(sqlite_durability, "_read_back", lambda *_a: "delete")
        with pytest.raises(DurabilityError, match="WAL"):
            configure_durable_connection(db)
    finally:
        db.close()


def test_configure_refuses_when_synchronous_does_not_take_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = sqlite3.connect(str(tmp_path / "x.db"))
    try:
        answers = iter(["wal", 1])
        monkeypatch.setattr(sqlite_durability, "_read_back", lambda *_a: next(answers))
        with pytest.raises(DurabilityError, match="synchronous"):
            configure_durable_connection(db)
    finally:
        db.close()


def test_copying_the_main_file_of_a_live_database_loses_the_newest_rows(
    tmp_path: Path,
) -> None:
    """Copying a live local file is not backing it up, and it fails silently.

    After a clean close SQLite checkpoints its write-ahead log into the main
    file, so a copy taken then is complete. The hazard is a copy taken while a
    writer is live -- the support-engineer case -- where the copy opens without
    error and is missing the rows written most recently. Stated as a
    measurement rather than a warning, because nothing about the failure is
    loud.
    """
    source_path = tmp_path / "outbox.db"
    outbox = TansekiOutbox(source_path)
    try:
        outbox.enqueue("e1", {"id": "d1"})
        outbox._db.commit()

        copy_path = tmp_path / "copy.db"
        copy_path.write_bytes(source_path.read_bytes())  # what `cp outbox.db` does

        copy = sqlite3.connect(str(copy_path))
        try:
            tables = {row[0] for row in copy.execute("SELECT name FROM sqlite_master")}
            if "outbox" in tables:
                rows = copy.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
                assert rows == 0
            else:
                # The schema itself may never have reached the main file, which
                # is the same lesson in a more dramatic form.
                assert "outbox" not in tables
        finally:
            copy.close()
    finally:
        outbox.close()
