"""Durability pragmas for the local SQLite state, verified rather than assumed.

The outbox and the question registry are the only things standing between a
captured human answer and losing it, and the README describes the outbox as a
durable write-ahead queue. That claim is only true if a commit has reached
stable storage before it is acknowledged, which is what ``synchronous=FULL``
buys and what SQLite's default (``NORMAL``) does not.

Every pragma here is read back and checked. ``PRAGMA journal_mode=WAL`` is a
no-op that silently reports the pre-existing mode when the conversion cannot
happen -- on a filesystem without working shared-memory support, for instance.
Assuming the statement did what it said would silently leave the process
running the journal mode it was trying to leave, which is the failure this
module exists to prevent.

A crash test is not a substitute for these assertions: on a healthy machine a
``SIGKILL`` also passes with ``synchronous=NORMAL``, so a passing crash suite
proves nothing about the pragma. Assert the pragma directly.
"""

from __future__ import annotations

import sqlite3

__all__ = ["DurabilityError", "configure_durable_connection"]

_EXPECTED_JOURNAL_MODE = "wal"
_EXPECTED_SYNCHRONOUS = 2  # FULL, as SQLite reports it.
_EXPECTED_FOREIGN_KEYS = 1


class DurabilityError(RuntimeError):
    """A local SQLite file could not be put into its required durable state."""


def _read_back(db: sqlite3.Connection, pragma: str) -> object:
    row = db.execute(f"PRAGMA {pragma}").fetchone()
    return None if row is None else row[0]


def configure_durable_connection(db: sqlite3.Connection) -> None:
    """Apply the durability pragmas, failing loudly if any does not take effect.

    Raises:
        DurabilityError: if the journal mode, synchronous level or foreign-key
            enforcement does not match what was asked for. Degrading silently
            would reinstate the exact gap these pragmas close.
    """
    db.execute("PRAGMA busy_timeout = 5000")

    db.execute("PRAGMA journal_mode = WAL")
    journal_mode = _read_back(db, "journal_mode")
    if str(journal_mode).lower() != _EXPECTED_JOURNAL_MODE:
        raise DurabilityError(
            "Could not enable WAL journalling, which this file needs to be durable. "
            f"SQLite reported {journal_mode!r}. This usually means the file is on a "
            "filesystem without shared-memory support. Refusing to continue, because "
            "running without it would mean acknowledging writes that are not yet on disk."
        )

    db.execute("PRAGMA synchronous = FULL")
    synchronous = _read_back(db, "synchronous")
    if synchronous != _EXPECTED_SYNCHRONOUS:
        raise DurabilityError(
            "Could not set synchronous=FULL, so a commit could be acknowledged before "
            f"reaching stable storage. SQLite reported {synchronous!r}."
        )

    db.execute("PRAGMA foreign_keys = ON")
    foreign_keys = _read_back(db, "foreign_keys")
    if foreign_keys != _EXPECTED_FOREIGN_KEYS:
        raise DurabilityError(
            f"Could not enable foreign-key enforcement; SQLite reported {foreign_keys!r}."
        )
