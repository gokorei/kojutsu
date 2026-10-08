"""Durable outbox so Tanseki writes survive store outages."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import random
import sqlite3
import stat
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from kojutsu.core.sqlite_durability import configure_durable_connection
from kojutsu.integrations.tanseki import TansekiError, TansekiPermanentError, TansekiWriter

DEFAULT_RETENTION_DAYS = 30
DEFAULT_RELAY_LEASE_SECONDS = 120.0

#: Most rows the outbox will hold. Ten thousand undelivered records is far past
#: anything a real outage produces -- one row is a human writing a comment on a
#: pull request -- so reaching this bound means something is wrong that an
#: operator must see now, not a throttle that would otherwise have absorbed the
#: queue. It is deliberately sized to be reachable only by a prolonged outage.
DEFAULT_MAX_ENTRIES = 10_000

#: Most bytes of queued content the outbox will hold. This is the disk half of
#: the bound, and the spool must never be the reason a host runs out of space,
#: so it sits at a fraction of a plausible filesystem rather than at the edge
#: of one. It is separate from the row count because the two bound different
#: things: a few very large captures and a very large number of small ones
#: reach the same disk pressure at completely different row counts, and one
#: number cannot be right for both.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

SCHEMA_VERSION = 2

_CREATE_OUTBOX_TABLE = """
CREATE TABLE outbox (
    entry_id          TEXT PRIMARY KEY,
    payload           TEXT NOT NULL,
    pending_payload   TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    attempts          INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,
    next_attempt_at   TEXT,
    dead_lettered_at  TEXT,
    lease_token       TEXT,
    lease_expires_at  TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
)
"""

_ITEM_COLUMNS = """
entry_id, payload, attempts, last_error, status, next_attempt_at,
dead_lettered_at, created_at, updated_at, lease_token, lease_expires_at
"""

_REQUIRED_COLUMNS = {
    "entry_id",
    "payload",
    "pending_payload",
    "status",
    "attempts",
    "last_error",
    "next_attempt_at",
    "dead_lettered_at",
    "lease_token",
    "lease_expires_at",
    "created_at",
    "updated_at",
}

#: Bytes a row costs the outbox, measured as the UTF-8 length of the text it
#: stores. Bare ``length()`` counts characters rather than bytes, which
#: understates the cost of precisely the records most likely to be large, and a
#: byte quota that undercounts is a quota that stops bounding the thing it
#: bounds. SQLite's own file size is deliberately *not* used instead: that
#: depends on page size, free pages, fragmentation and WAL state, so a bound
#: built on it would refuse at a different point on every filesystem and an
#: operator could not act on the number.
_ROW_BYTES = """
    length(CAST(entry_id AS BLOB))
    + length(CAST(payload AS BLOB))
    + COALESCE(length(CAST(pending_payload AS BLOB)), 0)
    + COALESCE(length(CAST(last_error AS BLOB)), 0)
"""


class OutboxOwnershipError(RuntimeError):
    """The local outbox is already owned by another process."""


@dataclass(frozen=True)
class OutboxQuotaUsage:
    """What the quota currently holds, and what it would allow.

    Exposed so an operator surface can report the bound without a caller having
    to know how the outbox measures a row, and so a refusal carries the same
    numbers the quota would have reported on its own.
    """

    entries: int
    bytes: int
    max_entries: int
    max_bytes: int

    @property
    def full(self) -> bool:
        """True when the next entry would be refused."""
        return self.entries >= self.max_entries or self.bytes >= self.max_bytes


class OutboxQuotaError(RuntimeError):
    """The outbox is full and refused to store another entry.

    A distinct type rather than a return value, because ``enqueue``'s ``False``
    means "your entry is durable and an in-flight lease owns delivery". Every
    existing caller reads ``False`` as success, so reporting a refusal through
    it would tell a producer its knowledge is captured when nothing was written
    at all -- the one outcome this queue exists to make impossible.
    """

    def __init__(
        self,
        entry_id: str,
        *,
        limit_name: str,
        limit: int,
        observed: int,
        usage: OutboxQuotaUsage,
    ) -> None:
        self.entry_id = entry_id
        self.limit_name = limit_name
        self.limit = limit
        self.observed = observed
        self.usage = usage
        super().__init__(self._detail())

    def _detail(self) -> str:
        if self.limit_name == "max_entries":
            measured = f"{self.observed} entries already held, limit {self.limit}"
        else:
            measured = f"{self.observed} bytes of queued content, limit {self.limit}"
        return (
            f"Refusing to store {self.entry_id!r}: outbox quota {self.limit_name} is full "
            f"({measured}). Nothing was deleted to make room, and nothing already "
            "queued was altered. Restore the store and run `kojutsu relay` to drain "
            "the queue. Writes that cannot be delivered count against this quota until "
            "an operator acts on them: list them with `kojutsu outbox-dead-letters`, "
            "remove expired ones with `kojutsu outbox-cleanup --older-than-days N`, "
            "and inspect what is queued with `kojutsu outbox`. If a queue this deep "
            "is normal for this deployment, raise max_entries/max_bytes where the outbox "
            "is constructed rather than leaving the bound off."
        )


@dataclass(frozen=True)
class OutboxItem:
    entry_id: str
    payload: dict[str, Any]
    attempts: int
    last_error: str | None
    status: str = "pending"
    next_attempt_at: str | None = None
    dead_lettered_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    lease_token: str | None = None
    lease_expires_at: str | None = None

    @property
    def state(self) -> str:
        if self.status == "pending":
            return "captured-locally"
        if self.status == "dead_letter":
            return "delivery-failed"
        return "delivery-retrying"


@dataclass(frozen=True)
class RelayResult:
    sent: int
    failed: int
    dead_lettered: int = 0


class CorruptOutboxPayloadError(ValueError):
    """A stored payload that is not valid JSON and can never be delivered."""

    def __init__(self, entry_id: str, raw: str) -> None:
        self.entry_id = entry_id
        self.raw = raw
        super().__init__(f"outbox entry {entry_id!r} has corrupt payload")


@dataclass
class _OwnerRecord:
    pid: int
    references: int
    descriptor: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _text_bytes(text: str) -> int:
    """Bytes a stored string costs, agreeing with how ``_ROW_BYTES`` measures a row."""
    return len(text.encode("utf-8"))


def _positive_limit(name: str, value: int) -> int:
    # ``bool`` is an ``int``, so without this a stray ``True`` would be a
    # one-entry quota, accepted without the author ever meaning to ask for one.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _secure_open_flags() -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _table_exists(db: sqlite3.Connection, table: str) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _validate_schema(db: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in db.execute("PRAGMA table_info(outbox)")}
    if not _REQUIRED_COLUMNS.issubset(columns):
        raise RuntimeError("Outbox migration produced an invalid schema")
    indexes = {str(row[1]) for row in db.execute("PRAGMA index_list(outbox)")}
    required_indexes = {"outbox_status_idx", "outbox_due_idx", "outbox_lease_idx"}
    if not required_indexes.issubset(indexes):
        raise RuntimeError("Outbox migration produced invalid indexes")


def _migrate_schema(db: sqlite3.Connection) -> None:
    try:
        db.execute("BEGIN IMMEDIATE")
        version_row = db.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0]) if version_row else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError("Outbox schema is newer than this application")
        if version == SCHEMA_VERSION:
            _validate_schema(db)
            db.commit()
            return
        if not _table_exists(db, "outbox"):
            db.execute(_CREATE_OUTBOX_TABLE)
        else:
            columns = {str(row[1]) for row in db.execute("PRAGMA table_info(outbox)")}
            if not {"entry_id", "payload"}.issubset(columns):
                raise RuntimeError("Outbox migration cannot repair an invalid base schema")
            for name, declaration in (
                ("pending_payload", "TEXT"),
                ("status", "TEXT NOT NULL DEFAULT 'pending'"),
                ("attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("last_error", "TEXT"),
                ("next_attempt_at", "TEXT"),
                ("dead_lettered_at", "TEXT"),
                ("lease_token", "TEXT"),
                ("lease_expires_at", "TEXT"),
                ("created_at", "TEXT NOT NULL DEFAULT ''"),
                ("updated_at", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE outbox ADD COLUMN {name} {declaration}")
        db.execute("CREATE INDEX IF NOT EXISTS outbox_status_idx ON outbox (status)")
        db.execute("CREATE INDEX IF NOT EXISTS outbox_due_idx ON outbox (status, next_attempt_at)")
        db.execute("CREATE INDEX IF NOT EXISTS outbox_lease_idx ON outbox (lease_expires_at)")
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        _validate_schema(db)
        db.commit()
    except BaseException:
        db.rollback()
        raise


class _OwnerRegistry:
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _owners: ClassVar[dict[Path, _OwnerRecord]] = {}

    @classmethod
    def acquire(cls, path: Path) -> _OwnerRecord:
        key = path.resolve(strict=False)
        pid = os.getpid()
        with cls._lock:
            current = cls._owners.get(key)
            if current is not None and current.pid == pid:
                current.references += 1
                return current
            descriptor = os.open(path, _secure_open_flags(), 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                os.close(descriptor)
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise OutboxOwnershipError(
                        f"Outbox is already owned by another process: {path}"
                    ) from None
                raise
            record = _OwnerRecord(pid=pid, references=1, descriptor=descriptor)
            cls._owners[key] = record
            return record

    @classmethod
    def release(cls, path: Path, record: _OwnerRecord) -> None:
        key = path.resolve(strict=False)
        with cls._lock:
            current = cls._owners.get(key)
            if current is None or current is not record:
                return
            current.references -= 1
            if current.references > 0:
                return
            cls._owners.pop(key, None)
            try:
                fcntl.flock(current.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(current.descriptor)


class TansekiOutbox:
    """Single-owner SQLite outbox for Tanseki writes.

    Delivery retries are unbounded on purpose. A transient failure means the
    store was unavailable, not that the capture was unwanted, and the only copy
    of a human's answer is the row sitting in this queue. Dead-lettering on an
    attempt count would convert a temporary outage into permanent loss of
    knowledge, which is the one outcome this project exists to prevent.

    This is deliberately different from the question registry, which *does*
    cap attempts and moves a claim to a terminal ``failed`` state. A question
    claim is a lease on work that can be re-derived from the PR; a captured
    answer cannot. The two ceilings are not the same decision.

    The attempt count is still recorded and still reported by
    ``kojutsu outbox``, so an operator can see a delivery that has been
    failing for days and act on it. The difference is that kojutsu will not
    make that decision silently, and will not throw the knowledge away first.

    Growth, by contrast, is bounded, and the bound is applied by refusing.
    While a store is unreachable every capture is still durable, so a long
    outage grows the spool; eventually that growth becomes the outage itself,
    and it surfaces as an opaque failure somewhere unrelated to the queue
    rather than as a report about the queue. So the outbox caps what it holds
    and declines to exceed the cap. It never evicts. A quota that deletes the
    oldest row to make room for the newest is not a bound, it is a policy for
    losing knowledge, and it applies that policy silently in the one component
    whose reason to exist is that nothing is lost. Refusing is louder and it is
    the honest failure: the caller learns its record was not captured, and can
    retry once someone has restored the store or cleared the queue.

    That is the opposite of the retry ceiling above, and the reason is that the
    two numbers measure different costs. A retry consumes no disk and so can be
    unbounded honestly; a row consumes disk, and cannot.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        retention_days: int | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        """Open the outbox, bounded to ``max_entries`` rows and ``max_bytes`` of content.

        Both limits are always enforced and neither has an unlimited mode. An
        operator who needs more room sets a larger number, which stays visible
        in configuration; "unbounded" is the absence of that decision, which is
        the condition the quota exists to close. Omission is bounded too: the
        defaults are finite, so an outbox constructed without saying anything
        still gets a bound.

        The two limits are independent rather than one setting that scales into
        the other, because they answer different questions. The row count
        bounds how many undelivered records of human knowledge are at risk,
        which is the quantity an operator can reason about. The byte count
        bounds what the spool costs on disk, which is what actually breaks a
        host. A few large captures and a great many small ones reach the same
        disk pressure at completely different row counts, so a single number
        would be wrong for one of them no matter which it was.
        """
        if retention_days is not None and retention_days < 0:
            raise ValueError("retention_days must be non-negative")
        self.max_entries = _positive_limit("max_entries", max_entries)
        self.max_bytes = _positive_limit("max_bytes", max_bytes)
        self.path = Path(path).expanduser()
        parent_created = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent_created:
            self.path.parent.chmod(0o700)
        self.owner_lock_path = self.path.with_name(f"{self.path.name}.owner.lock")
        self.retention_days = retention_days
        self._closed = False
        self._lock = threading.RLock()
        self._owner = _OwnerRegistry.acquire(self.owner_lock_path)
        try:
            self._secure_database_file()
            self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
            configure_durable_connection(self._db)
            _migrate_schema(self._db)
            self.cleanup_dead_letters()
        except BaseException:
            try:
                if "_db" in self.__dict__:
                    self._db.close()
            finally:
                _OwnerRegistry.release(self.owner_lock_path, self._owner)
            raise

    def _secure_database_file(self) -> None:
        descriptor = os.open(self.path, _secure_open_flags(), 0o600)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OutboxOwnershipError("Outbox path must be a regular file")
            if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
                raise OutboxOwnershipError("Outbox file must be owned by the current user")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._db.close()
            finally:
                _OwnerRegistry.release(self.owner_lock_path, self._owner)

    def __enter__(self) -> TansekiOutbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _occupied(self) -> tuple[int, int]:
        """Return rows held and bytes of queued content, as the caller sees them.

        The caller must already hold ``self._lock``. Inside ``enqueue`` this runs
        in the open write transaction, so the totals it returns include this
        transaction's own uncommitted work -- which is what makes the quota a
        bound rather than a check that two callers can both pass.

        Counted by scanning rather than from a cached counter, because a counter
        drifts the moment anything else moves the total: ``mark_sent`` deletes
        rows, cleanup deletes rows, and a process killed between the write and
        the counter's increment leaves the two permanently disagreeing. A
        drifted counter refuses at a limit that is not the real occupancy, and
        an operator cannot act on a number they cannot reproduce. A capture is
        one scan per human decision, which is not a hot path.
        """
        row = self._db.execute(
            f"SELECT COUNT(*), COALESCE(SUM({_ROW_BYTES}), 0) FROM outbox"
        ).fetchone()
        return int(row[0]), int(row[1])

    def _admit(
        self,
        entry_id: str,
        usage: OutboxQuotaUsage,
        *,
        projected_bytes: int,
        adds_entry: bool,
    ) -> None:
        """Refuse an enqueue that would take the outbox past its bound.

        ``projected_bytes`` is what the outbox would hold *after* this enqueue,
        and ``adds_entry`` says whether it would be a row the outbox does not
        already hold. Both matter for duplicates: an entry that is already
        stored is charged only the difference its new content makes, because a
        retry storm must not be able to exhaust the quota by repeating one fact,
        and a producer re-confirming something already durable must not be told
        "not stored" when it is in fact stored -- that ambiguity is what makes a
        webhook retry and post the comment twice.

        Nothing is deleted here. The bound is applied by declining, never by
        making room.
        """
        if adds_entry and usage.entries >= usage.max_entries:
            raise OutboxQuotaError(
                entry_id,
                limit_name="max_entries",
                limit=usage.max_entries,
                observed=usage.entries,
                usage=usage,
            )
        if projected_bytes > usage.max_bytes:
            raise OutboxQuotaError(
                entry_id,
                limit_name="max_bytes",
                limit=usage.max_bytes,
                observed=projected_bytes,
                usage=usage,
            )

    def quota_usage(self) -> OutboxQuotaUsage:
        """Report what the quota holds now, for an operator surface or a health check."""
        with self._lock:
            entries, queued_bytes = self._occupied()
        return OutboxQuotaUsage(
            entries=entries,
            bytes=queued_bytes,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )

    def enqueue(self, entry_id: str, payload: dict[str, Any]) -> bool:
        """Durably replace an unleased entry and preserve active in-flight deliveries.

        Returns ``False`` only when an active delivery lease owns the entry. The
        payload is still durable in that case, staged as ``pending_payload``, so
        ``False`` is a concurrency notice about delivery and never a failure to
        store.

        Raises:
            OutboxQuotaError: if the entry would take the outbox past
                ``max_entries`` or ``max_bytes``. Nothing is written, and
                nothing already stored is deleted or evicted to make room. This
                raises rather than returning ``False`` precisely because every
                existing caller reads ``False`` as "your entry is safe", and a
                refusal reported that way would be a lost record announced as a
                success.
        """
        if not entry_id:
            raise ValueError("entry_id must not be empty")
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    f"""
                    SELECT lease_token, lease_expires_at, payload,
                           COALESCE(length(CAST(pending_payload AS BLOB)), 0), {_ROW_BYTES}
                    FROM outbox WHERE entry_id = ?
                    """,
                    (entry_id,),
                ).fetchone()
                active = (
                    row is not None and row[0] is not None and (row[1] is None or str(row[1]) > now)
                )
                occupied_entries, occupied_bytes = self._occupied()
                usage = OutboxQuotaUsage(
                    entries=occupied_entries,
                    bytes=occupied_bytes,
                    max_entries=self.max_entries,
                    max_bytes=self.max_bytes,
                )
                payload_text = json.dumps(payload)
                payload_bytes = _text_bytes(payload_text)
                if active:
                    # The lease keeps delivering the row's current payload, so the
                    # staged replacement is the only content that changes and no
                    # row is added -- this can only be refused by the byte bound.
                    # Re-sending exactly what is already in flight stages nothing:
                    # that duplicate is a second copy of a payload the outbox
                    # already holds, and refusing a producer's re-confirmation of
                    # something durable is how a webhook ends up retrying and
                    # posting the same comment twice.
                    redundant = str(row[2]) == payload_text
                    self._admit(
                        entry_id,
                        usage,
                        projected_bytes=(
                            occupied_bytes - int(row[3]) + (0 if redundant else payload_bytes)
                        ),
                        adds_entry=False,
                    )
                    if redundant:
                        self._db.execute(
                            "UPDATE outbox SET updated_at = ? WHERE entry_id = ?",
                            (now, entry_id),
                        )
                    else:
                        self._db.execute(
                            """
                            UPDATE outbox SET pending_payload = ?, updated_at = ?
                            WHERE entry_id = ?
                            """,
                            (payload_text, now, entry_id),
                        )
                else:
                    # Replacing a row subtracts what that row costs today, so
                    # re-sending an entry that is already stored is charged only
                    # for what actually changes on disk. The upsert clears
                    # pending_payload and last_error, so entry_id and payload are
                    # the whole of the new cost.
                    self._admit(
                        entry_id,
                        usage,
                        projected_bytes=(
                            occupied_bytes
                            - (int(row[4]) if row is not None else 0)
                            + _text_bytes(entry_id)
                            + payload_bytes
                        ),
                        adds_entry=row is None,
                    )
                    self._db.execute(
                        """
                        INSERT INTO outbox (
                            entry_id, payload, pending_payload, status, attempts, last_error,
                            next_attempt_at, dead_lettered_at, lease_token, lease_expires_at,
                            created_at, updated_at
                        ) VALUES (?, ?, NULL, 'pending', 0, NULL, NULL, NULL, NULL, NULL, ?, ?)
                        ON CONFLICT(entry_id) DO UPDATE SET
                            payload = excluded.payload, pending_payload = NULL,
                            status = 'pending', attempts = 0, last_error = NULL,
                            next_attempt_at = NULL, dead_lettered_at = NULL,
                            lease_token = NULL, lease_expires_at = NULL,
                            updated_at = excluded.updated_at
                        """,
                        (entry_id, payload_text, now, now),
                    )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            return not active

    def pending(self, limit: int = 100) -> list[OutboxItem]:
        """Return active items, including retry items awaiting a due time."""
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT {_ITEM_COLUMNS}
                FROM outbox WHERE status IN ('pending', 'retrying')
                ORDER BY created_at ASC LIMIT ?
                """,
                (self._bounded_limit(limit),),
            ).fetchall()
        return self._convert_rows(rows)

    def ready(self, limit: int = 100) -> list[OutboxItem]:
        """Return due, unclaimed active items."""
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT {_ITEM_COLUMNS}
                FROM outbox
                WHERE (
                    status = 'pending'
                    OR (status = 'retrying' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                )
                AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                ORDER BY created_at ASC LIMIT ?
                """,
                (_now(), _now(), self._bounded_limit(limit)),
            ).fetchall()
        return self._convert_rows(rows)

    def claim_due(
        self, *, limit: int = 100, lease_seconds: float = DEFAULT_RELAY_LEASE_SECONDS
    ) -> list[OutboxItem]:
        """Atomically claim due items for one relay batch."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        bounded_limit = self._bounded_limit(limit)
        if bounded_limit == 0:
            return []
        now = datetime.now(UTC)
        token = uuid.uuid4().hex
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                rows = self._db.execute(
                    f"""
                    SELECT {_ITEM_COLUMNS}
                    FROM outbox
                    WHERE (
                        status = 'pending'
                        OR (
                            status = 'retrying'
                            AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                        )
                    )
                    AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                    ORDER BY created_at ASC LIMIT ?
                    """,
                    (now.isoformat(), now.isoformat(), bounded_limit),
                ).fetchall()
                entry_ids = [str(row[0]) for row in rows]
                if entry_ids:
                    placeholders = ", ".join("?" for _ in entry_ids)
                    self._db.execute(
                        f"""
                        UPDATE outbox SET lease_token = ?, lease_expires_at = ?, updated_at = ?
                        WHERE entry_id IN ({placeholders})
                        """,
                        (token, lease_expires_at, now.isoformat(), *entry_ids),
                    )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        claimed_at = len(_ITEM_COLUMNS.split(",")) - 2
        claimed_rows = [(*row[:claimed_at], token, lease_expires_at) for row in rows]
        return self._convert_rows(claimed_rows)

    def claim_entry(
        self, entry_id: str, *, lease_seconds: float = DEFAULT_RELAY_LEASE_SECONDS
    ) -> OutboxItem | None:
        """Claim one due item when it is not owned by another relay."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        token = uuid.uuid4().hex
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    f"""
                    SELECT {_ITEM_COLUMNS}
                    FROM outbox
                    WHERE entry_id = ?
                      AND (
                        status = 'pending'
                        OR (
                            status = 'retrying'
                            AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                        )
                      )
                      AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                    """,
                    (entry_id, now.isoformat(), now.isoformat()),
                ).fetchone()
                if row is None:
                    self._db.rollback()
                    return None
                self._db.execute(
                    """
                    UPDATE outbox SET lease_token = ?, lease_expires_at = ?, updated_at = ?
                    WHERE entry_id = ?
                    """,
                    (token, lease_expires_at, now.isoformat(), entry_id),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        claimed_at = len(_ITEM_COLUMNS.split(",")) - 2
        claimed_row = (*row[:claimed_at], token, lease_expires_at)
        try:
            return self._item(claimed_row)
        except CorruptOutboxPayloadError as exc:
            self._quarantine_corrupt(exc.entry_id, exc.raw)
            return None

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        return max(0, min(limit, 1_000))

    def _quarantine_corrupt(self, entry_id: str, raw: str) -> None:
        """Move an undeliverable payload to dead_letter so it cannot starve valid rows."""
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    """
                    UPDATE outbox
                    SET status = 'dead_letter', attempts = attempts + 1,
                        last_error = ?, dead_lettered_at = ?,
                        lease_token = NULL, lease_expires_at = NULL,
                        next_attempt_at = NULL, updated_at = ?
                    WHERE entry_id = ? AND status != 'dead_letter'
                    """,
                    (f"corrupt payload (not valid JSON, {len(raw)} chars)", now, now, entry_id),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    def _convert_rows(self, rows: list[tuple[Any, ...]]) -> list[OutboxItem]:
        items: list[OutboxItem] = []
        for row in rows:
            try:
                items.append(self._item(row))
            except CorruptOutboxPayloadError as exc:
                self._quarantine_corrupt(exc.entry_id, exc.raw)
                continue
        return items

    @staticmethod
    def _item(row: tuple[Any, ...]) -> OutboxItem:
        try:
            payload = json.loads(str(row[1]))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CorruptOutboxPayloadError(str(row[0]), str(row[1])) from exc
        if not isinstance(payload, dict):
            raise CorruptOutboxPayloadError(str(row[0]), str(row[1]))
        return OutboxItem(
            entry_id=str(row[0]),
            payload=payload,
            attempts=int(row[2]),
            last_error=str(row[3]) if row[3] is not None else None,
            status=str(row[4]),
            next_attempt_at=str(row[5]) if row[5] is not None else None,
            dead_lettered_at=str(row[6]) if row[6] is not None else None,
            created_at=str(row[7]),
            updated_at=str(row[8]),
            lease_token=str(row[9]) if row[9] is not None else None,
            lease_expires_at=str(row[10]) if row[10] is not None else None,
        )

    def pending_count(self) -> int:
        with self._lock:
            (count,) = self._db.execute(
                "SELECT COUNT(*) FROM outbox WHERE status IN ('pending', 'retrying')"
            ).fetchone()
        return int(count)

    def captured_count(self) -> int:
        with self._lock:
            (count,) = self._db.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = 'pending'"
            ).fetchone()
        return int(count)

    def delivery_failed_count(self) -> int:
        with self._lock:
            (count,) = self._db.execute(
                "SELECT COUNT(*) FROM outbox WHERE status IN ('retrying', 'dead_letter')"
            ).fetchone()
        return int(count)

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) FROM outbox GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "retrying": 0, "dead_letter": 0}
        counts.update({str(status): int(count) for status, count in rows})
        return counts

    def dead_letters(self, limit: int = 100) -> list[OutboxItem]:
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT {_ITEM_COLUMNS}
                FROM outbox WHERE status = 'dead_letter'
                ORDER BY dead_lettered_at ASC, created_at ASC LIMIT ?
                """,
                (self._bounded_limit(limit),),
            ).fetchall()
        items: list[OutboxItem] = []
        for row in rows:
            try:
                items.append(self._item(row))
            except CorruptOutboxPayloadError:
                # Already quarantined; surface with empty payload so operators can see it.
                items.append(
                    OutboxItem(
                        entry_id=str(row[0]),
                        payload={},
                        attempts=int(row[2]),
                        last_error=str(row[3]) if row[3] is not None else "corrupt payload",
                        status=str(row[4]),
                        next_attempt_at=str(row[5]) if row[5] is not None else None,
                        dead_lettered_at=str(row[6]) if row[6] is not None else None,
                        created_at=str(row[7]),
                        updated_at=str(row[8]),
                        lease_token=str(row[9]) if row[9] is not None else None,
                        lease_expires_at=str(row[10]) if row[10] is not None else None,
                    )
                )
        return items

    def mark_sent(self, entry_id: str, *, lease_token: str) -> bool:
        """Delete a delivered entry, but only while its lease is still held.

        The lease is required, not optional: a leaseless delete removes whatever
        row happens to carry the id, including one a *different* relay claimed
        after this caller's lease expired. Every production caller holds a
        lease from ``claim_due``/``claim_entry``; there is no unclaimed path
        that needs deleting, so there is no escape hatch either.
        """
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT pending_payload FROM outbox WHERE entry_id = ? AND lease_token = ?",
                    (entry_id, lease_token),
                ).fetchone()
                if row is None:
                    self._db.rollback()
                    return False
                if row[0] is None:
                    cursor = self._db.execute(
                        "DELETE FROM outbox WHERE entry_id = ? AND lease_token = ?",
                        (entry_id, lease_token),
                    )
                else:
                    cursor = self._db.execute(
                        """
                        UPDATE outbox
                        SET payload = ?, pending_payload = NULL, status = 'pending',
                            attempts = 0, last_error = NULL, next_attempt_at = NULL,
                            dead_lettered_at = NULL, lease_token = NULL,
                            lease_expires_at = NULL, updated_at = ?
                        WHERE entry_id = ? AND lease_token = ?
                        """,
                        (row[0], _now(), entry_id, lease_token),
                    )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def mark_failed(
        self,
        entry_id: str,
        error: str | Exception,
        *,
        retry_after: float | None = None,
        lease_token: str | None = None,
    ) -> str | None:
        now = datetime.now(UTC)
        permanent = isinstance(error, TansekiPermanentError)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if lease_token is None:
                    row = self._db.execute(
                        "SELECT attempts, pending_payload FROM outbox WHERE entry_id = ?",
                        (entry_id,),
                    ).fetchone()
                else:
                    row = self._db.execute(
                        """
                        SELECT attempts, pending_payload FROM outbox
                        WHERE entry_id = ? AND lease_token = ?
                        """,
                        (entry_id, lease_token),
                    ).fetchone()
                if row is None:
                    self._db.rollback()
                    return None
                if row[1] is not None:
                    cursor = self._db.execute(
                        """
                        UPDATE outbox
                        SET payload = ?, pending_payload = NULL, status = 'pending', attempts = 0,
                            last_error = ?, next_attempt_at = NULL, dead_lettered_at = NULL,
                            lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                        WHERE entry_id = ?
                        """,
                        (row[1], str(error), now.isoformat(), entry_id),
                    )
                    # Lease-scoped callers must not clobber a row whose lease
                    # changed between SELECT and UPDATE (stale worker vs new owner).
                    if lease_token is not None and cursor.rowcount == 0:
                        self._db.rollback()
                        return None
                    self._db.commit()
                    return "pending"
                attempts = int(row[0]) + 1
                if permanent:
                    status = "dead_letter"
                    next_attempt = None
                    dead_lettered_at = now.isoformat()
                else:
                    status = "retrying"
                    if retry_after is None:
                        delay = min(300.0, 2 ** min(attempts - 1, 8))
                        delay *= random.uniform(0.5, 1.5)  # noqa: S311 - backoff jitter, not cryptographic
                    else:
                        delay = min(300.0, max(0.0, retry_after))
                    next_attempt = (now + timedelta(seconds=delay)).isoformat()
                    dead_lettered_at = None
                parameters: list[Any] = [
                    attempts,
                    status,
                    str(error),
                    next_attempt,
                    dead_lettered_at,
                    now.isoformat(),
                    entry_id,
                ]
                where = "entry_id = ?"
                if lease_token is not None:
                    parameters.append(lease_token)
                    where += " AND lease_token = ?"
                cursor = self._db.execute(
                    f"""
                    UPDATE outbox
                    SET attempts = ?, status = ?, last_error = ?, next_attempt_at = ?,
                        dead_lettered_at = ?, lease_token = NULL, lease_expires_at = NULL,
                        updated_at = ?
                    WHERE {where}
                    """,
                    parameters,
                )
                if cursor.rowcount == 0:
                    # Lease lost or row deleted between SELECT and UPDATE.
                    self._db.rollback()
                    return None
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            else:
                return status

    def release_claim(self, entry_id: str, *, lease_token: str) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT pending_payload FROM outbox WHERE entry_id = ? AND lease_token = ?",
                    (entry_id, lease_token),
                ).fetchone()
                if row is None:
                    self._db.rollback()
                    return False
                if row[0] is None:
                    self._db.execute(
                        """
                        UPDATE outbox SET lease_token = NULL, lease_expires_at = NULL
                        WHERE entry_id = ? AND lease_token = ?
                        """,
                        (entry_id, lease_token),
                    )
                else:
                    self._db.execute(
                        """
                        UPDATE outbox
                        SET payload = ?, pending_payload = NULL, status = 'pending', attempts = 0,
                            next_attempt_at = NULL, dead_lettered_at = NULL,
                            lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                        WHERE entry_id = ? AND lease_token = ?
                        """,
                        (row[0], _now(), entry_id, lease_token),
                    )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            else:
                return True

    def requeue_dead_letter(self, entry_id: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE outbox
                    SET status = 'pending', attempts = 0, last_error = NULL,
                        pending_payload = NULL, next_attempt_at = NULL, dead_lettered_at = NULL,
                        lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE entry_id = ? AND status = 'dead_letter'
                    """,
                    (now, entry_id),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def delete_dead_letter(self, entry_id: str) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    "DELETE FROM outbox WHERE entry_id = ? AND status = 'dead_letter'",
                    (entry_id,),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def cleanup_dead_letters(self, retention_days: int | None = None) -> int:
        effective_retention = self.retention_days if retention_days is None else retention_days
        if effective_retention is None:
            return 0
        if effective_retention < 0:
            raise ValueError("retention_days must be non-negative")
        cutoff = (datetime.now(UTC) - timedelta(days=effective_retention)).isoformat()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    DELETE FROM outbox
                    WHERE status = 'dead_letter'
                      AND lease_token IS NULL
                      AND lease_expires_at IS NULL
                      AND COALESCE(dead_lettered_at, updated_at) <= ?
                    """,
                    (cutoff,),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
            else:
                return max(0, cursor.rowcount)


def _deliver(
    outbox: TansekiOutbox,
    client: TansekiWriter,
    item: OutboxItem,
    on_error: Callable[[str, str], None] | None,
) -> tuple[bool, bool]:
    if item.lease_token is None:
        return False, False
    try:
        client.upsert_document(item.payload)
    except TansekiError as exc:
        status = outbox.mark_failed(
            item.entry_id,
            exc,
            retry_after=getattr(exc, "retry_after", None),
            lease_token=item.lease_token,
        )
        if on_error is not None:
            on_error(item.entry_id, str(exc))
        return True, status == "dead_letter"
    except BaseException:
        outbox.release_claim(item.entry_id, lease_token=item.lease_token)
        raise
    outbox.mark_sent(item.entry_id, lease_token=item.lease_token)
    return False, False


def relay(
    outbox: TansekiOutbox,
    client: TansekiWriter,
    *,
    limit: int = 100,
    lease_seconds: float = DEFAULT_RELAY_LEASE_SECONDS,
    on_error: Callable[[str, str], None] | None = None,
) -> RelayResult:
    """Drain a leased batch without letting poison rows starve valid rows.

    The quota does not apply here, and that is deliberate rather than an
    oversight: the relay never enqueues, so it can only be refused by nothing
    and can only free a slot. A full outbox is therefore the outage that was
    already happening with the spool now bounded -- not a new one -- and the
    relay's job in that state is exactly what it always was, keep trying the
    rows it holds and hand each one back to the store as it recovers.

    What it does *not* do is evict to make room for a retry. ``mark_failed``
    leaves the row in place, so a retry storm against a dead store keeps exactly
    one row per record and no more, and a dead letter keeps its row and its
    bytes until an operator removes it. Both are charged against the quota,
    which is the point: an undelivered record occupies the outbox whether or not
    it is still being retried.

    The batch is claimed with :meth:`TansekiOutbox.claim_due` in a single
    transaction, not discovered with ``ready`` and claimed row by row. Two
    relays racing the old way both read the same due rows and only one won each
    claim -- safe but wasteful, one transaction per row plus the discovery read.
    The batch claim holds the write lock from selection through lease assignment,
    so the second relay never sees the first relay's rows at all.
    """
    sent = 0
    failed = 0
    dead_lettered = 0
    for item in outbox.claim_due(limit=limit, lease_seconds=lease_seconds):
        item_failed, item_dead_lettered = _deliver(outbox, client, item, on_error)
        if item_dead_lettered:
            dead_lettered += 1
        if item_failed:
            failed += 1
        else:
            sent += 1
    return RelayResult(sent=sent, failed=failed, dead_lettered=dead_lettered)


def retry_dead_letter(
    outbox: TansekiOutbox,
    client: TansekiWriter,
    entry_id: str,
    *,
    lease_seconds: float = DEFAULT_RELAY_LEASE_SECONDS,
    on_error: Callable[[str, str], None] | None = None,
) -> RelayResult:
    """Requeue and immediately deliver one dead-lettered entry."""
    if not outbox.requeue_dead_letter(entry_id):
        return RelayResult(sent=0, failed=0)
    item = outbox.claim_entry(entry_id, lease_seconds=lease_seconds)
    if item is None:
        return RelayResult(sent=0, failed=0)
    item_failed, item_dead_lettered = _deliver(outbox, client, item, on_error)
    return RelayResult(
        sent=0 if item_failed else 1,
        failed=1 if item_failed else 0,
        dead_lettered=1 if item_dead_lettered else 0,
    )
