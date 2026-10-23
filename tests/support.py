"""White-box SQLite probes for tests: the one boundary for raw store access.

Schema/migration, timing, and census tests legitimately need SQL the public
API refuses to run: invalid states, tampered checksums, expired leases,
pragma introspection. Rather than reaching into store privates
(``registry._db``, ``outbox._lock``) from every test file, they go through
here. Each helper opens its OWN short-lived connection to the store file --
identified by the store's public ``.path`` -- so production classes keep no
test-only surface, connection lifecycle lives in one place, and the SQL text
that remains in tests is the assertion, not a lifecycle hack.

All writers commit before returning. Stores run one statement per method on
their own connection, so anything committed here is visible to the next
public call; nothing here ever observes uncommitted store state.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def store_db(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection to a store file. Always closed."""
    db = sqlite3.connect(str(Path(path).expanduser()))
    try:
        yield db
    finally:
        db.close()


def query_one(path: str | Path, sql: str, params: Sequence[object] = ()) -> tuple | None:
    """One row, or ``None``. Read-only probes for counts and statuses."""
    with store_db(path) as db:
        return db.execute(sql, tuple(params)).fetchone()


def query_all(path: str | Path, sql: str, params: Sequence[object] = ()) -> list[tuple]:
    """Every matching row, as plain tuples."""
    with store_db(path) as db:
        return db.execute(sql, tuple(params)).fetchall()


def query_value(path: str | Path, sql: str, params: Sequence[object] = ()) -> object:
    """The first column of the first row. Status and count probes."""
    with store_db(path) as db:
        row = db.execute(sql, tuple(params)).fetchone()
        return row[0] if row else None


def execute_write(path: str | Path, sql: str, params: Sequence[object] = ()) -> int:
    """Run one write and commit. Returns the affected row count."""
    with store_db(path) as db:
        cursor = db.execute(sql, tuple(params))
        db.commit()
        return cursor.rowcount


def executemany_write(path: str | Path, sql: str, rows: Sequence[Sequence[object]]) -> None:
    """Run one write over many rows and commit."""
    with store_db(path) as db:
        db.executemany(sql, [tuple(row) for row in rows])
        db.commit()


def table_count(path: str | Path, table: str) -> int:
    """Row count for a known store table."""
    row = query_one(path, f"SELECT COUNT(*) FROM {table}")
    return int(row[0]) if row else 0


def user_version(path: str | Path) -> int:
    """The schema version stamp."""
    with store_db(path) as db:
        row = db.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0


def set_user_version(path: str | Path, version: int) -> None:
    """Stamp a schema version (migration refusal tests)."""
    execute_write(path, f"PRAGMA user_version = {int(version)}")


def table_columns(path: str | Path, table: str) -> set[str]:
    """Column names for a known store table."""
    return {str(row[1]) for row in query_all(path, f"PRAGMA table_info({table})")}


def index_names(path: str | Path, table: str) -> set[str]:
    """Index names for a known store table."""
    return {str(row[1]) for row in query_all(path, f"PRAGMA index_list({table})")}
