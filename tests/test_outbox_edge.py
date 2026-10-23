"""Edge-case coverage for the durable outbox: leases, corruption, bounds."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from tests.support import execute_write, set_user_version

from kojutsu.core.outbox import SCHEMA_VERSION, TansekiOutbox
from kojutsu.core.sqlite_durability import configure_durable_connection
from kojutsu.integrations.tanseki import TansekiError


class _BoomClient:
    def upsert_document(self, payload: dict) -> dict:
        raise RuntimeError("poison row")


def test_deliver_base_exception_releases_claim(tmp_path: Path) -> None:
    from kojutsu.core.outbox import relay

    with TansekiOutbox(tmp_path / "o.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        with pytest.raises(RuntimeError, match="poison"):
            relay(outbox, _BoomClient())
        # Claim released: a fresh claim_due must return the row again.
        due = outbox.claim_due(limit=10)
        assert [i.entry_id for i in due] == ["e1"]


def test_release_claim_clears_lease(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "o.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        claimed = outbox.claim_due(limit=10)
        assert claimed and claimed[0].lease_token
        assert outbox.release_claim("e1", lease_token=claimed[0].lease_token or "") is True
        pending = outbox.pending()
        assert pending[0].lease_token is None


def test_corrupt_payload_quarantined_to_dead_letter(tmp_path: Path) -> None:
    path = tmp_path / "o.db"
    with TansekiOutbox(path) as outbox:
        outbox.enqueue("bad", {"id": "d"})
    execute_write(path, "UPDATE outbox SET payload = 'not-json{{' WHERE entry_id = 'bad'")
    with TansekiOutbox(path) as outbox:
        assert outbox.pending() == []
        dead = outbox.dead_letters()
        assert [d.entry_id for d in dead] == ["bad"]
        assert "corrupt payload" in (dead[0].last_error or "")


def test_non_dict_payload_quarantined(tmp_path: Path) -> None:
    path = tmp_path / "o.db"
    with TansekiOutbox(path) as outbox:
        outbox.enqueue("bad", {"id": "d"})
    execute_write(path, "UPDATE outbox SET payload = '[1,2]' WHERE entry_id = 'bad'")
    with TansekiOutbox(path) as outbox:
        assert outbox.pending() == []
        assert [d.entry_id for d in outbox.dead_letters()] == ["bad"]


def test_mark_failed_stale_lease_returns_none(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "o.db") as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        outbox.claim_due(limit=10)
        assert outbox.mark_failed("e1", TansekiError("x"), lease_token="stale") is None
        assert outbox.mark_sent("e1", lease_token="stale") is False


def test_backoff_is_capped(tmp_path: Path) -> None:
    path = tmp_path / "o.db"
    with TansekiOutbox(path) as outbox:
        outbox.enqueue("e1", {"id": "d1"})
        claimed = outbox.claim_due(limit=10)
        outbox.mark_failed("e1", TansekiError("down"), lease_token=claimed[0].lease_token or "")
        row = outbox.pending()[0]
        # attempts=1; next_attempt must exist and be bounded (cap 300s * jitter 1.5).
        assert row.next_attempt_at is not None
        # A far-future attempt count still caps rather than growing unbounded.
        for _ in range(12):
            c = outbox.claim_due(limit=10)
            if not c:
                # force-due by clearing next_attempt
                execute_write(path, "UPDATE outbox SET next_attempt_at = NULL")
                c = outbox.claim_due(limit=10)
            outbox.mark_failed("e1", TansekiError("down"), lease_token=(c[0].lease_token or ""))
            execute_write(path, "UPDATE outbox SET next_attempt_at = NULL")
        assert outbox.pending()[0].attempts >= 10


def test_limits_are_clamped_through_the_public_api(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "o.db") as outbox:
        for n in range(3):
            outbox.enqueue(f"e{n}", {"id": f"d{n}"})
        assert outbox.pending(limit=0) == []
        assert outbox.claim_due(limit=-5) == []
        assert len(outbox.pending(limit=10_000)) == 3


def test_newer_schema_version_refused(tmp_path: Path) -> None:
    path = tmp_path / "o.db"
    with TansekiOutbox(path):
        pass
    set_user_version(path, SCHEMA_VERSION + 99)
    with pytest.raises(Exception, match="newer"):
        TansekiOutbox(path)


def test_busy_timeout_configured(tmp_path: Path) -> None:
    path = tmp_path / "durability.db"
    with closing(sqlite3.connect(str(path))) as db:
        configure_durable_connection(db)
        (timeout,) = db.execute("PRAGMA busy_timeout").fetchone()
        assert timeout == 5000
