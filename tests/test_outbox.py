"""Tests for the durable outbox and relay."""

from __future__ import annotations

import multiprocessing
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty

import pytest

import kojutsu.core.outbox as outbox_module
from kojutsu.core.outbox import (
    OutboxOwnershipError,
    RelayResult,
    TansekiOutbox,
    relay,
    retry_dead_letter,
)
from kojutsu.integrations.tanseki import TansekiError, TansekiPermanentError


class _OkClient:
    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert_document(self, payload: dict) -> dict:
        self.upserts.append(payload)
        return {"revision": "r1"}


class _FailClient:
    def upsert_document(self, payload: dict) -> dict:
        raise TansekiError("store unavailable")


def _open_outbox_in_child(path: str, result: multiprocessing.Queue[str]) -> None:
    try:
        with TansekiOutbox(path):
            result.put("opened")
    except OutboxOwnershipError:
        result.put("owned")


def test_enqueue_and_pending(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1", "content": "x"})
        outbox.enqueue("e2", {"id": "d2", "content": "y"})
        assert outbox.pending_count() == 2
        items = outbox.pending()
        assert {i.entry_id for i in items} == {"e1", "e2"}
        assert items[0].attempts == 0


def test_enqueue_is_idempotent_by_entry_id(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1", "content": "old"})
        outbox.enqueue("e1", {"id": "d1", "content": "new"})
        assert outbox.pending_count() == 1
        assert outbox.pending()[0].payload["content"] == "new"


def test_relay_sends_and_clears(tmp_path: Path) -> None:
    client = _OkClient()
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1", "content": "x"})
        result = relay(outbox, client)
        assert result.sent == 1
        assert result.failed == 0
        assert outbox.pending_count() == 0
    assert client.upserts[0]["id"] == "d1"


def test_relay_records_failures(tmp_path: Path) -> None:
    errors: list[tuple[str, str]] = []
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1", "content": "x"})
        result = relay(outbox, _FailClient(), on_error=lambda eid, err: errors.append((eid, err)))
        assert result.sent == 0
        assert result.failed == 1
        assert outbox.pending_count() == 1
        item = outbox.pending()[0]
        assert item.attempts == 1
        assert item.last_error == "store unavailable"
    assert errors == [("e1", "store unavailable")]


def test_concurrent_relays_partition_rows_without_double_delivery(tmp_path: Path) -> None:
    """Two relays racing the same rows deliver each exactly once.

    The batch claim takes every due row in one transaction, so the relay that
    loses the race finds nothing to do -- no second discovery read, no
    per-row claim contention. What must never happen, under any interleaving,
    is both relays delivering the same row.
    """
    started = threading.Event()
    release = threading.Event()

    class SlowClient(_OkClient):
        def upsert_document(self, payload: dict) -> dict:
            if payload["id"] == "first":
                started.set()
                release.wait(timeout=2)
            return super().upsert_document(payload)

    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("first", {"id": "first"})
        outbox.enqueue("second", {"id": "second"})
        slow_client = SlowClient()
        second_client = _OkClient()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(relay, outbox, slow_client)
            try:
                assert started.wait(timeout=2)
                second_result = executor.submit(relay, outbox, second_client).result(timeout=2)
            finally:
                release.set()
            first_result = first_future.result(timeout=2)
        assert first_result == RelayResult(sent=2, failed=0)
        assert second_result == RelayResult(sent=0, failed=0)
        delivered = [payload["id"] for payload in slow_client.upserts] + [
            payload["id"] for payload in second_client.upserts
        ]
        assert sorted(delivered) == ["first", "second"]
        assert outbox.pending_count() == 0


def test_enqueue_replacement_resets_attempts_after_retry_and_dead_letter(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("retry", {"id": "retry", "content": "old"})
        outbox.mark_failed("retry", TansekiError("temporary"), retry_after=0)
        assert outbox.pending()[0].attempts == 1

        outbox.enqueue("retry", {"id": "retry", "content": "replacement"})
        assert outbox.pending()[0].attempts == 0
        assert outbox.pending()[0].payload["content"] == "replacement"
        assert (
            outbox.mark_failed("retry", TansekiError("temporary again"), retry_after=0)
            == "retrying"
        )
        assert outbox.status_counts()["retrying"] == 1

        outbox.mark_failed("retry", TansekiPermanentError("invalid"), lease_token=None)
        assert outbox.status_counts()["dead_letter"] == 1
        outbox.enqueue("dead", {"id": "dead"})
        outbox.mark_failed("dead", TansekiPermanentError("invalid"))
        outbox.enqueue("dead", {"id": "dead", "content": "replacement"})
        assert outbox.pending()[0].attempts == 0
        assert outbox.pending()[0].payload["content"] == "replacement"


def test_mark_sent_removes_row(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        claimed = outbox.claim_due(limit=10, lease_seconds=60)
        assert outbox.mark_sent("e1", lease_token=claimed[0].lease_token) is True
        assert outbox.pending_count() == 0


def test_mark_sent_without_a_lease_is_refused(tmp_path: Path) -> None:
    """A leaseless delete removes whatever row carries the id -- including one
    another relay claimed -- so the lease is required, not optional."""
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        claimed = outbox.claim_due(limit=10, lease_seconds=60)
        assert outbox.mark_sent("e1", lease_token="stale-or-foreign") is False
        assert outbox.pending_count() == 1
        assert claimed[0].lease_token is not None


def test_transient_failure_is_scheduled_and_permanent_failure_is_dead_lettered(
    tmp_path: Path,
) -> None:
    class _TransientFailureError(TansekiError):
        pass

    class _PermanentFailureError(TansekiPermanentError):
        pass

    class _Client:
        def __init__(self) -> None:
            self.error: Exception = _TransientFailureError("temporary")

        def upsert_document(self, payload: dict) -> dict:
            raise self.error

    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("retry", {"id": "retry"})
        client = _Client()
        assert relay(outbox, client).failed == 1
        assert outbox.ready() == []
        assert outbox.status_counts()["retrying"] == 1
        outbox._db.execute("UPDATE outbox SET next_attempt_at = '2000-01-01T00:00:00+00:00'")
        outbox._db.commit()
        assert len(outbox.ready()) == 1
        outbox._db.execute("DELETE FROM outbox WHERE entry_id = 'retry'")
        outbox._db.commit()

        outbox.enqueue("dead", {"id": "dead"})
        client.error = _PermanentFailureError("invalid")
        result = relay(outbox, client)
        assert result.dead_lettered == 1
        assert outbox.status_counts()["dead_letter"] == 1
        assert outbox.pending_count() == 0


def test_transient_failures_never_dead_letter_from_attempt_count(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("retry", {"id": "retry"})
        for _ in range(3):
            assert (
                outbox.mark_failed("retry", TansekiError("temporary"), retry_after=0) == "retrying"
            )

        assert outbox.status_counts() == {"pending": 0, "retrying": 1, "dead_letter": 0}


def test_claim_lease_prevents_duplicate_relay_and_stale_completion(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        claimed = outbox.claim_due(limit=10, lease_seconds=60)
        assert [item.entry_id for item in claimed] == ["e1"]
        assert claimed[0].lease_token is not None
        assert outbox.claim_due(limit=10, lease_seconds=60) == []

        assert outbox.enqueue("e1", {"id": "d1", "content": "replacement"}) is False
        active = outbox.pending()[0]
        assert active.payload == {"id": "d1"}
        assert outbox.mark_sent("e1", lease_token=claimed[0].lease_token) is True
        assert outbox.pending()[0].payload["content"] == "replacement"

        replacement = outbox.claim_due(limit=10, lease_seconds=60)
        assert outbox.mark_sent("e1", lease_token=replacement[0].lease_token) is True
        assert outbox.pending_count() == 0


def test_newer_enqueue_survives_permanent_failure_of_active_payload(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1", "content": "old"})
        claim = outbox.claim_entry("e1")
        assert claim is not None and claim.lease_token is not None
        assert outbox.enqueue("e1", {"id": "d1", "content": "new"}) is False

        assert (
            outbox.mark_failed(
                "e1", TansekiPermanentError("old payload invalid"), lease_token=claim.lease_token
            )
            == "pending"
        )
        item = outbox.pending()[0]
        assert item.payload["content"] == "new"
        assert outbox.status_counts()["dead_letter"] == 0


def test_dead_letter_can_be_listed_requeued_and_deleted(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        outbox.mark_failed("e1", TansekiPermanentError("invalid payload"))

        dead = outbox.dead_letters()
        assert outbox.captured_count() == 0
        assert outbox.delivery_failed_count() == 1
        assert [item.entry_id for item in dead] == ["e1"]
        assert dead[0].state == "delivery-failed"
        assert outbox.requeue_dead_letter("e1") is True
        assert outbox.pending()[0].attempts == 0
        assert outbox.pending()[0].state == "captured-locally"

        outbox.mark_failed("e1", TansekiPermanentError("invalid payload"))
        assert outbox.delete_dead_letter("e1") is True
        assert outbox.dead_letters() == []


def test_retry_dead_letter_delivers_exact_entry(tmp_path: Path) -> None:
    client = _OkClient()
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("older", {"id": "older"})
        outbox.enqueue("target", {"id": "target"})
        outbox.mark_failed("target", TansekiPermanentError("retry me"))

        result = retry_dead_letter(outbox, client, "target")

        assert result == RelayResult(sent=1, failed=0)
        assert [payload["id"] for payload in client.upserts] == ["target"]
        assert [item.entry_id for item in outbox.pending()] == ["older"]


def test_dead_letter_retention_requires_explicit_safe_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "outbox.db"
    with TansekiOutbox(path) as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        outbox.mark_failed("e1", TansekiPermanentError("invalid payload"))
        expired = (datetime.now(UTC) - timedelta(days=31)).isoformat()
        outbox._db.execute(
            "UPDATE outbox SET dead_lettered_at = ?, updated_at = ?", (expired, expired)
        )
        outbox._db.commit()

    with TansekiOutbox(path) as reopened:
        assert [item.entry_id for item in reopened.dead_letters()] == ["e1"]
        assert reopened.cleanup_dead_letters() == 0
        assert reopened.cleanup_dead_letters(retention_days=30) == 1
        assert reopened.dead_letters() == []


def test_dead_letter_cleanup_preserves_an_entry_with_a_lease(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("leased", {"id": "leased"})
        outbox.mark_failed("leased", TansekiPermanentError("invalid"))
        expired = (datetime.now(UTC) - timedelta(days=31)).isoformat()
        outbox._db.execute(
            """
            UPDATE outbox SET dead_lettered_at = ?, updated_at = ?,
                lease_token = 'operator-lease', lease_expires_at = ?
            """,
            (expired, expired, expired),
        )
        outbox._db.commit()

        assert outbox.cleanup_dead_letters(retention_days=30) == 0
        assert [item.entry_id for item in outbox.dead_letters()] == ["leased"]


def test_outbox_migration_is_versioned_and_adds_pending_payload(tmp_path: Path) -> None:
    path = tmp_path / "outbox.db"
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            """
            CREATE TABLE outbox (
                entry_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    with TansekiOutbox(path) as outbox:
        columns = {row[1] for row in outbox._db.execute("PRAGMA table_info(outbox)")}
        assert "pending_payload" in columns
        assert outbox._db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_outbox_migration_failure_rolls_back_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "outbox.db"
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            """
            CREATE TABLE outbox (
                entry_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    original_validate = outbox_module._validate_schema

    def fail_validation(db: sqlite3.Connection) -> None:
        raise RuntimeError("migration interrupted")

    monkeypatch.setattr(outbox_module, "_validate_schema", fail_validation)
    with pytest.raises(RuntimeError, match="migration interrupted"):
        TansekiOutbox(path)

    with closing(sqlite3.connect(path)) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(outbox)")}
        assert "pending_payload" not in columns
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0

    monkeypatch.setattr(outbox_module, "_validate_schema", original_validate)
    with TansekiOutbox(path) as outbox:
        assert outbox._db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_outbox_and_owner_lock_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "outbox.db"
    with TansekiOutbox(path) as outbox:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(outbox.owner_lock_path.stat().st_mode) == 0o600


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(), reason="requires process spawning"
)
def test_outbox_rejects_another_process(tmp_path: Path) -> None:
    path = tmp_path / "outbox.db"
    context = multiprocessing.get_context("spawn")
    result: multiprocessing.Queue[str] = context.Queue()
    with TansekiOutbox(path):
        process = context.Process(target=_open_outbox_in_child, args=(str(path), result))
        process.start()
        try:
            assert result.get(timeout=5) == "owned"
        except Empty:
            pytest.fail("child process did not report its outbox ownership result")
        finally:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
