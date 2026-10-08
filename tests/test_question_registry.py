"""Tests for SQLite registry schema and lifecycle deduplication."""

import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import kojutsu.core.question_registry as question_registry_module
from kojutsu.core.question_registry import SCHEMA_VERSION, SqliteQuestionRegistry


def _record(registry: SqliteQuestionRegistry, question_id: str, *, repo: str, pr_number: int):
    """Record one pending question, the fixture every queue test starts from."""
    registry.record_question(
        question_id=question_id,
        github_comment_id=1000 + int(question_id.rsplit("q", 1)[-1] or 0),
        repo=repo,
        pr_number=pr_number,
        pr_url=f"https://github.com/{repo}/pull/{pr_number}",
        question_text="Why?",
        question_category="design_decision",
    )


def _v3_questions_table() -> str:
    """The ``questions`` table exactly as schema v3 left it."""
    return """
    CREATE TABLE questions (
        question_id       TEXT PRIMARY KEY,
        identifier        INTEGER,
        repo              TEXT,
        pr_number         INTEGER,
        pr_url            TEXT,
        question_text     TEXT,
        category          TEXT,
        jira_ticket_key   TEXT,
        session_id        TEXT,
        status            TEXT NOT NULL DEFAULT 'pending',
        error_message     TEXT,
        answered          INTEGER NOT NULL DEFAULT 0,
        answer_comment_id INTEGER,
        question_author   TEXT,
        created_at        TEXT NOT NULL
    );
    """


def test_pr_state_events_allow_repeated_action_and_reject_replay(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    assert not registry.pr_state_change_seen("org/repo", 1, "closed", "delivery-1")
    registry.mark_pr_state_change("org/repo", 1, "closed", "delivery-1")
    assert registry.pr_state_change_seen("org/repo", 1, "closed", "delivery-1")
    assert not registry.pr_state_change_seen("org/repo", 1, "closed", "delivery-2")
    registry.mark_pr_state_change("org/repo", 1, "closed", "delivery-2")
    registry.close()


def test_webhook_delivery_claim_is_durable(tmp_path) -> None:
    path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(path)
    claim = registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1")
    assert claim not in {"duplicate", "conflict"}
    assert registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1") == "active"
    assert registry.complete_delivery("delivery-1", claim)
    registry.close()

    reopened = SqliteQuestionRegistry(path)
    assert (
        reopened.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1") == "duplicate"
    )
    reopened.close()


def test_old_pr_state_schema_is_migrated(tmp_path) -> None:
    path = tmp_path / "registry.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE pr_state_changes (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (repo, pr_number, action)
        );
        CREATE INDEX pr_state_changes_lookup_idx
            ON pr_state_changes (repo, pr_number, action);
        """
    )
    connection.execute(
        "INSERT INTO pr_state_changes VALUES ('org/repo', 1, 'closed', '2023-01-01')"
    )
    connection.commit()
    connection.close()

    registry = SqliteQuestionRegistry(path)
    assert registry.pr_state_change_seen("org/repo", 1, "closed")
    assert not registry.pr_state_change_seen("org/repo", 1, "closed", "new-delivery")
    columns = {row[1] for row in registry._db.execute("PRAGMA table_info(pr_state_changes)")}
    assert {"event_id", "status", "claim_token"}.issubset(columns)
    assert registry._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert registry._db.execute("SELECT COUNT(*) FROM pr_state_changes").fetchone()[0] == 1
    assert not registry._db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pr_state_changes_legacy'"
    ).fetchone()
    registry.close()


def test_delivery_failure_is_retryable_and_identity_conflicts_fail_closed(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")

    first_claim = registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1")
    assert first_claim not in {"duplicate", "conflict"}
    assert registry.release_delivery("delivery-1", first_claim, "temporary failure")
    second_claim = registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1")
    assert second_claim not in {"duplicate", "conflict"}
    assert (
        registry.claim_delivery("delivery-1", "other/repo", "pull_request", "hash-2") == "conflict"
    )
    assert registry.complete_delivery("delivery-1", second_claim)


def test_expired_delivery_lease_can_be_reclaimed(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    first_claim = registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1")
    assert first_claim not in {"duplicate", "conflict"}
    registry._db.execute(
        "UPDATE webhook_deliveries SET lease_expires_at = ? WHERE delivery_id = ?",
        ("2000-01-01T00:00:00+00:00", "delivery-1"),
    )
    registry._db.commit()

    second_claim = registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1")
    assert second_claim not in {"duplicate", "conflict"}
    assert not registry.complete_delivery("delivery-1", first_claim)
    assert not registry.release_delivery("delivery-1", first_claim, "stale failure")
    assert registry.complete_delivery("delivery-1", second_claim)
    registry.close()


def test_different_answer_ids_cannot_atomically_claim_the_same_question(tmp_path) -> None:
    path = tmp_path / "registry.db"
    first = SqliteQuestionRegistry(path)
    first.record_question(
        question_id="q1",
        github_comment_id=100,
        repo="org/repo",
        pr_number=1,
        pr_url="https://github.com/org/repo/pull/1",
        question_text="Why?",
        question_category="design_decision",
    )
    first.close()
    registries = [SqliteQuestionRegistry(path), SqliteQuestionRegistry(path)]

    def claim(registry: SqliteQuestionRegistry, answer_id: int) -> str | None:
        return registry.claim_answer(
            parent_comment_id=100,
            answer_comment_id=answer_id,
            repo="org/repo",
            pr_number=1,
            question_id="q1",
            entry_id=f"answer-{answer_id}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(claim, registry, answer_id)
            for registry, answer_id in zip(registries, (201, 202), strict=True)
        ]
        results = [future.result() for future in futures]
    for registry in registries:
        registry.close()

    assert sum(result is not None for result in results) == 1


def test_old_webhook_delivery_schema_is_migrated_as_completed(tmp_path) -> None:
    path = tmp_path / "registry.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE webhook_deliveries (
            delivery_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            event TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO webhook_deliveries VALUES ('delivery-1', 'org/repo', 'pull_request', '2023-01-01')"
    )
    connection.commit()
    connection.close()

    registry = SqliteQuestionRegistry(path)
    assert (
        registry.claim_delivery("delivery-1", "org/repo", "pull_request", "hash-1") == "duplicate"
    )
    registry.close()


def test_stale_answer_token_cannot_finalize_reclaimed_capture(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    registry.record_question(
        question_id="q1",
        github_comment_id=100,
        repo="org/repo",
        pr_number=1,
        pr_url="https://github.com/org/repo/pull/1",
        question_text="Why?",
        question_category="design_decision",
    )
    first_claim = registry.claim_answer(
        parent_comment_id=100,
        answer_comment_id=201,
        repo="org/repo",
        pr_number=1,
        question_id="q1",
        entry_id="answer-201",
    )
    assert first_claim is not None
    registry._db.execute(
        "UPDATE answer_captures SET lease_expires_at = ? WHERE answer_comment_id = ?",
        ("2000-01-01T00:00:00+00:00", 201),
    )
    registry._db.commit()

    second_claim = registry.claim_answer(
        parent_comment_id=100,
        answer_comment_id=201,
        repo="org/repo",
        pr_number=1,
        question_id="q1",
        entry_id="answer-201",
    )
    assert second_claim is not None
    assert second_claim != first_claim
    assert not registry.complete_answer(201, first_claim)
    assert not registry.release_answer(201, first_claim)
    assert registry.complete_answer(201, second_claim)
    assert registry.is_question_answered(100)
    registry.close()


def test_cleanup_deliveries_is_bounded_and_preserves_active_claims(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    old = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    future = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    rows = [
        ("old-completed", "completed", old, old, None),
        ("recent-completed", "completed", recent, recent, None),
        ("active-processing", "processing", None, old, future),
        ("expired-processing", "processing", None, old, old),
        ("processing-without-lease", "processing", None, old, None),
        ("retryable", "retryable", None, old, old),
    ]
    registry._db.executemany(
        """
        INSERT INTO webhook_deliveries (
            delivery_id, repo, event, payload_hash, status, completed_at,
            created_at, lease_expires_at
        ) VALUES (?, 'org/repo', 'pull_request', 'hash', ?, ?, ?, ?)
        """,
        rows,
    )
    registry._db.commit()

    assert registry.cleanup_deliveries(retention_days=30, limit=2) == 2
    assert registry.cleanup_deliveries(retention_days=30, limit=2) == 1
    assert registry.cleanup_deliveries(retention_days=30) == 0
    remaining = {
        row[0] for row in registry._db.execute("SELECT delivery_id FROM webhook_deliveries")
    }
    assert remaining == {
        "recent-completed",
        "active-processing",
        "processing-without-lease",
    }
    indexes = {row[1] for row in registry._db.execute("PRAGMA index_list(webhook_deliveries)")}
    assert "webhook_deliveries_completed_idx" in indexes
    with pytest.raises(ValueError):
        registry.cleanup_deliveries(retention_days=-1)
    registry.close()


def test_delivery_cleanup_preserves_answer_dedupe_state(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    registry.record_question(
        question_id="q1",
        github_comment_id=100,
        repo="org/repo",
        pr_number=1,
        pr_url="https://github.com/org/repo/pull/1",
        question_text="Why?",
        question_category="design_decision",
    )
    claim = registry.claim_answer(
        parent_comment_id=100,
        answer_comment_id=201,
        repo="org/repo",
        pr_number=1,
        question_id="q1",
        entry_id="answer-201",
    )
    assert claim is not None
    assert registry.complete_answer(201, claim)

    assert registry.cleanup_deliveries(retention_days=0) == 0
    assert registry.answer_comment_seen(201)
    assert registry.is_question_answered(100)
    registry.close()


def test_migration_failure_rolls_back_closes_and_retries(tmp_path, monkeypatch) -> None:
    path = tmp_path / "registry.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE pr_state_changes (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (repo, pr_number, action)
        );
        INSERT INTO pr_state_changes VALUES ('org/repo', 1, 'closed', '2023-01-01');
        """
    )
    connection.close()

    original_connect = question_registry_module.sqlite3.connect
    original_migrate = question_registry_module._migrate_schema
    opened: list[sqlite3.Connection] = []

    def tracked_connect(*args, **kwargs):
        db = original_connect(*args, **kwargs)
        opened.append(db)
        return db

    def fail_migration(db: sqlite3.Connection) -> None:
        db.execute("BEGIN IMMEDIATE")
        db.execute("ALTER TABLE pr_state_changes RENAME TO pr_state_changes_legacy")
        raise RuntimeError("migration interrupted")

    monkeypatch.setattr(question_registry_module.sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(question_registry_module, "_migrate_schema", fail_migration)
    with pytest.raises(RuntimeError, match="migration interrupted"):
        SqliteQuestionRegistry(path)

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")

    monkeypatch.setattr(question_registry_module.sqlite3, "connect", original_connect)
    monkeypatch.setattr(question_registry_module, "_migrate_schema", original_migrate)
    registry = SqliteQuestionRegistry(path)
    assert registry.pr_state_change_seen("org/repo", 1, "closed")
    assert registry._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    registry.close()


def test_list_questions_filters_by_status_and_bounds_the_page(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    _record(registry, "q2", repo="org/repo", pr_number=2)
    _record(registry, "q3", repo="other/repo", pr_number=1)
    registry.mark_question_answered(1001, 9001)

    pending = registry.list_questions(status="pending")
    assert {row["question_id"] for row in pending} == {"q2", "q3"}
    assert registry.list_questions(status="answered")[0]["question_id"] == "q1"

    assert {row["question_id"] for row in registry.list_questions(repo="org/repo")} == {"q1", "q2"}
    assert [row["question_id"] for row in registry.list_questions(repo="other/repo")] == ["q3"]
    assert [row["question_id"] for row in registry.list_questions(pr_number=1)] == ["q1", "q3"]

    # Filters combine conjunctively rather than widening each other.
    assert registry.list_questions(status="pending", repo="org/repo", pr_number=1) == []
    assert [row["question_id"] for row in registry.list_questions(repo="org/repo", limit=1)] == [
        "q1"
    ]
    assert registry.list_questions(limit=0) == []
    assert registry.list_questions(limit=10_000) == registry.list_questions()

    assert registry.count_questions() == 3
    assert registry.count_questions(status="pending") == 2
    assert registry.count_questions(repo="org/repo") == 2
    registry.close()


def test_list_questions_order_is_total_so_paging_never_skips_or_repeats(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    for index in range(5):
        _record(registry, f"q{index}", repo="org/repo", pr_number=index + 1)
    # Same repo and PR as q0, so only question_id can break that tie.
    _record(registry, "q9", repo="org/repo", pr_number=1)

    first = [row["question_id"] for row in registry.list_questions(status="pending", limit=100)]
    second = [row["question_id"] for row in registry.list_questions(status="pending", limit=100)]
    assert first == second, "repeated identical queries must order identically"
    assert len(set(first)) == len(first)
    # pr_number 1 holds q0 and q9; the primary key breaks that tie, so the pair
    # must be adjacent rather than separated by a later PR.
    assert first.index("q0") < first.index("q9")
    assert first[:2] == ["q0", "q9"]

    # Walk the queue a page at a time, retiring each page as a worker would:
    # no gaps, no repeats, and the order the full listing reports.
    walked: list[str] = []
    while True:
        page = registry.list_questions(status="pending", limit=2)
        if not page:
            break
        walked.extend(row["question_id"] for row in page)
        registry._db.execute(
            "DELETE FROM questions WHERE question_id IN (?, ?)",
            tuple(row["question_id"] for row in page),
        )
        registry._db.commit()
    assert walked == first
    assert len(set(walked)) == 6
    assert registry.list_questions(status="pending") == []
    registry.close()


def test_list_questions_rejects_unbounded_or_unknown_filters(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    with pytest.raises(ValueError, match="non-negative"):
        registry.list_questions(limit=-1)
    with pytest.raises(ValueError, match="non-negative"):
        registry.list_questions(limit=True)
    with pytest.raises(ValueError, match="Unknown question status"):
        registry.list_questions(status="not-a-status")
    with pytest.raises(ValueError, match="Unknown question status"):
        registry.count_questions(status="not-a-status")
    registry.close()


def test_enumeration_uses_the_work_index_rather_than_scanning(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    plan = " ".join(
        str(row[3])
        for row in registry._db.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT question_id FROM questions
            WHERE status = 'pending' AND repo = 'org/repo' AND pr_number = 1
            """
        )
    )
    assert "questions_work_idx" in plan
    assert "SCAN questions" not in plan
    registry.close()


def test_concurrent_claimants_win_exactly_one_question(tmp_path) -> None:
    path = tmp_path / "registry.db"
    seed = SqliteQuestionRegistry(path)
    _record(seed, "q1", repo="org/repo", pr_number=1)
    seed.close()

    registries = [SqliteQuestionRegistry(path) for _ in range(4)]

    def claim(index: int) -> str | None:
        return registries[index].claim_question_for_answer("q1", f"worker-{index}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(claim, index) for index in range(4)]
        results = [future.result() for future in futures]
    tokens = [token for token in results if token is not None]
    for registry in registries:
        registry.close()

    assert len(results) == 4, "all four claimants must actually have run"
    assert len(tokens) == 1
    assert len(set(tokens)) == 1
    winner = SqliteQuestionRegistry(path)
    claimed = winner.list_questions(status="claimed")
    assert len(claimed) == 1
    assert claimed[0]["question_id"] == "q1"
    assert claimed[0]["attempts"] == 1
    assert claimed[0]["assignee"] in {f"worker-{index}" for index in range(4)}
    # The winning token is the only one stored against the question.
    assert (
        winner._db.execute(
            "SELECT claim_token FROM questions WHERE question_id = ?", ("q1",)
        ).fetchone()[0]
        == tokens[0]
    )
    winner.close()


def test_live_claim_refuses_a_second_claimant_and_rejects_bad_claims(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    first = registry.claim_question_for_answer("q1", "worker-a")
    assert first is not None
    assert registry.claim_question_for_answer("q1", "worker-b") is None
    assert registry.claim_question_for_answer("missing", "worker-b") is None

    for bad in ("", "   "):
        with pytest.raises(ValueError, match="assignee"):
            registry.claim_question_for_answer("q1", bad)
    with pytest.raises(ValueError, match="ttl"):
        registry.claim_question_for_answer("q1", "worker-b", ttl=timedelta(0))
    with pytest.raises(ValueError, match="max_attempts"):
        registry.claim_question_for_answer("q1", "worker-b", max_attempts=0)

    assert registry.list_questions(status="claimed")[0]["attempts"] == 1
    registry.close()


def test_expired_question_lease_is_stealable_and_counts_the_attempt(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    stale = registry.claim_question_for_answer("q1", "worker-a")
    assert stale is not None
    registry._db.execute(
        "UPDATE questions SET lease_expires_at = ? WHERE question_id = ?",
        ("2000-01-01T00:00:00+00:00", "q1"),
    )
    registry._db.commit()

    fresh = registry.claim_question_for_answer("q1", "worker-b")
    assert fresh is not None
    assert fresh != stale

    claimed = registry.list_questions(status="claimed")[0]
    assert claimed["attempts"] == 2
    assert claimed["assignee"] == "worker-b"
    # The dead worker must not be able to release the live owner's claim.
    assert not registry.release_question("q1", stale, "dead worker cleaning up")
    assert registry.list_questions(status="claimed")[0]["assignee"] == "worker-b"
    assert (
        registry._db.execute(
            "SELECT claim_token FROM questions WHERE question_id = ?", ("q1",)
        ).fetchone()[0]
        == fresh
    )
    registry.close()


def test_claim_stops_at_the_attempt_ceiling_and_dead_letters_the_question(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)

    for attempt in (1, 2):
        token = registry.claim_question_for_answer("q1", f"worker-{attempt}", max_attempts=2)
        assert token is not None
        assert registry.release_question("q1", token, f"attempt {attempt} failed")

    assert registry.claim_question_for_answer("q1", "worker-3", max_attempts=2) is None
    assert registry.list_questions(status="failed")[0]["attempts"] == 2
    assert registry.claim_question_for_answer("q1", "worker-4", max_attempts=2) is None
    assert registry.list_questions(status="pending") == []
    registry.close()


def test_terminal_questions_are_never_claimable(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    _record(registry, "q2", repo="org/repo", pr_number=2)
    _record(registry, "q3", repo="org/repo", pr_number=3)
    registry.mark_question_answered(1001, 9001)
    registry.update_question_status(1002, "superseded")
    registry.update_question_status(1003, "failed", "gave up")

    for question_id in ("q1", "q2", "q3"):
        assert registry.claim_question_for_answer(question_id, "worker") is None
    assert registry.list_questions(status="pending") == []
    assert registry.count_questions() == 3

    with pytest.raises(ValueError, match="Unknown question status"):
        registry.update_question_status(1002, "invented")

    other = SqliteQuestionRegistry(tmp_path / "other.db")
    with pytest.raises(ValueError, match="Unknown question status"):
        other.record_question(
            question_id="q9",
            github_comment_id=1,
            repo="org/repo",
            pr_number=9,
            pr_url="https://github.com/org/repo/pull/9",
            question_text="Why?",
            question_category="design_decision",
            status="invented",
        )
    other.close()
    registry.close()


def test_release_question_requeues_and_preserves_the_attempt_count(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    first = registry.claim_question_for_answer("q1", "worker-a")
    assert first is not None

    assert not registry.release_question("q1", "not-the-token", "wrong owner")
    with pytest.raises(ValueError, match="reason"):
        registry.release_question("q1", first, "")
    assert registry.release_question("q1", first, "github unavailable")

    requeued = registry.list_questions(status="pending")[0]
    assert requeued["question_id"] == "q1"
    assert requeued["attempts"] == 1, "releasing is not a reset"
    assert requeued["assignee"] is None
    assert requeued["lease_expires_at"] is None
    assert requeued["last_error"] == "github unavailable"
    assert requeued["last_attempt_at"] is not None
    assert (
        registry._db.execute(
            "SELECT claim_token FROM questions WHERE question_id = ?", ("q1",)
        ).fetchone()[0]
        is None
    )

    second = registry.claim_question_for_answer("q1", "worker-b")
    assert second is not None and second != first
    assert registry.list_questions(status="claimed")[0]["attempts"] == 2
    # A released question can no longer be released twice.
    assert not registry.release_question("q1", first, "double release")
    registry.close()


def test_recording_a_question_never_walks_an_answered_one_back(tmp_path) -> None:
    """Re-running `ask` on the same question id must not re-open the dedupe gate."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    registry.mark_question_answered(1001, 9001)
    assert registry.is_question_answered(1001)

    _record(registry, "q1", repo="org/repo", pr_number=1)
    assert registry.is_question_answered(1001)
    answered = registry.list_questions(status="answered")[0]
    assert answered["answer_comment_id"] == 9001
    assert answered["attempts"] == 0
    # The disagreement the migration refuses to tolerate must not be creatable.
    assert (
        registry._db.execute(
            "SELECT COUNT(*) FROM questions WHERE answered = 1 AND status <> 'answered'"
        ).fetchone()[0]
        == 0
    )
    registry.close()


def test_status_vocabulary_is_enforced_by_the_database(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record(registry, "q1", repo="org/repo", pr_number=1)
    with pytest.raises(sqlite3.IntegrityError, match="documented vocabulary"):
        registry._db.execute("UPDATE questions SET status = 'invented' WHERE question_id = 'q1'")
    with pytest.raises(sqlite3.IntegrityError, match="documented vocabulary"):
        registry._db.execute(
            "INSERT INTO questions (question_id, status, created_at) "
            "VALUES ('q2', 'invented', '2026-01-01')"
        )
    assert registry.list_questions(status="pending")[0]["status"] == "pending"
    registry.close()


def _write_v3_registry(path, *, questions, captures=()) -> None:
    """Write a registry file exactly as schema v3 left it, then stop at version 3."""
    connection = sqlite3.connect(path)
    connection.executescript(
        _v3_questions_table()
        + """
        CREATE TABLE answer_captures (
            answer_comment_id INTEGER PRIMARY KEY,
            parent_comment_id INTEGER NOT NULL,
            question_id      TEXT NOT NULL,
            entry_id         TEXT NOT NULL UNIQUE,
            status           TEXT NOT NULL,
            claimed_at       TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            last_error       TEXT,
            completed_at     TEXT,
            claim_token      TEXT
        );
        """
    )
    connection.executemany(
        """
        INSERT INTO questions (
            question_id, identifier, repo, pr_number, pr_url, question_text, category,
            jira_ticket_key, session_id, status, error_message, answered,
            answer_comment_id, question_author, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        questions,
    )
    connection.executemany(
        """
        INSERT INTO answer_captures (
            answer_comment_id, parent_comment_id, question_id, entry_id, status,
            claimed_at, lease_expires_at, last_error, completed_at, claim_token
        ) VALUES (?, ?, ?, ?, 'completed', '2023-01-01T00:00:00+00:00',
                  '2023-01-01T00:00:00+00:00', NULL, '2023-01-01T00:00:00+00:00', NULL)
        """,
        captures,
    )
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()


def test_question_answered_before_the_migration_still_reads_as_answered(tmp_path) -> None:
    path = tmp_path / "registry.db"
    _write_v3_registry(
        path,
        questions=[
            (
                "q1",
                100,
                "org/repo",
                1,
                "https://github.com/org/repo/pull/1",
                "Why?",
                "design_decision",
                None,
                "session-1",
                "answered",
                None,
                1,
                9001,
                "asker",
                "2023-01-01T00:00:00+00:00",
            ),
            (
                "q2",
                101,
                "org/repo",
                1,
                "https://github.com/org/repo/pull/1",
                "How?",
                "design_decision",
                None,
                "session-1",
                "pending",
                None,
                0,
                None,
                "asker",
                "2023-01-01T00:00:00+00:00",
            ),
        ],
        captures=[(9001, 100, "q1", "answer-9001")],
    )

    registry = SqliteQuestionRegistry(path)
    assert registry._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert registry.is_question_answered(100)
    assert not registry.is_question_answered(101)

    answered = registry.list_questions(status="answered")
    assert [row["question_id"] for row in answered] == ["q1"]
    assert answered[0]["answer_comment_id"] == 9001
    assert answered[0]["answered_at"] == "2023-01-01T00:00:00+00:00"
    assert answered[0]["updated_at"] == "2023-01-01T00:00:00+00:00"

    # The captured entry the answer produced is untouched by the migration.
    capture = registry._db.execute(
        "SELECT entry_id, status, completed_at FROM answer_captures WHERE answer_comment_id = 9001"
    ).fetchone()
    assert capture == ("answer-9001", "completed", "2023-01-01T00:00:00+00:00")
    registry.close()


def test_migration_repairs_a_legacy_status_answered_disagreement(tmp_path) -> None:
    """A pre-v4 row whose flag and status disagree is promoted, never discarded."""
    path = tmp_path / "registry.db"
    _write_v3_registry(
        path,
        questions=[
            (
                "q1",
                100,
                "org/repo",
                1,
                "https://github.com/org/repo/pull/1",
                "Why?",
                "design_decision",
                None,
                None,
                "pending",
                None,
                1,
                9001,
                "asker",
                "2023-01-01T00:00:00+00:00",
            ),
        ],
        captures=[(9001, 100, "q1", "answer-9001")],
    )

    registry = SqliteQuestionRegistry(path)
    assert registry.is_question_answered(100), "the answered flag is honoured, not discarded"
    assert [row["status"] for row in registry.list_questions()] == ["answered"]
    registry.close()


def test_migration_is_checksummed_forward_only_and_preserves_existing_rows(tmp_path) -> None:
    path = tmp_path / "registry.db"
    original = [
        (
            f"q{index}",
            100 + index,
            "org/repo",
            index + 1,
            f"https://github.com/org/repo/pull/{index + 1}",
            f"Question {index}",
            "design_decision",
            f"KEY-{index}",
            "session-1",
            "pending",
            "some error",
            0,
            None,
            "asker",
            "2023-01-01T00:00:00+00:00",
        )
        for index in range(4)
    ]
    _write_v3_registry(path, questions=original)

    registry = SqliteQuestionRegistry(path)
    rows = registry._db.execute(
        "SELECT question_id, question_text, created_at, error_message, jira_ticket_key "
        "FROM questions ORDER BY question_id"
    ).fetchall()
    assert rows == [
        (
            f"q{index}",
            f"Question {index}",
            "2023-01-01T00:00:00+00:00",
            "some error",
            f"KEY-{index}",
        )
        for index in range(4)
    ], "existing rows must be preserved, not rewritten"

    recorded = registry._db.execute(
        "SELECT version, checksum FROM registry_migrations ORDER BY version"
    ).fetchall()
    # Derived from the ledger rather than spelled out, so a new migration is
    # covered by this assertion the moment it is registered instead of needing
    # this test edited -- which is the failure mode a hardcoded list invites.
    assert recorded == [
        (version, question_registry_module._migration_checksum(statements))
        for version, statements in sorted(question_registry_module._LEDGERED_MIGRATIONS.items())
    ]
    registry.close()

    # Reopening a migrated registry is a no-op: the ledger prevents a replay.
    reopened = SqliteQuestionRegistry(path)
    assert reopened.count_questions() == 4
    assert reopened._db.execute("SELECT COUNT(*) FROM registry_migrations").fetchone()[0] == len(
        question_registry_module._LEDGERED_MIGRATIONS
    )
    reopened.close()


def test_rewritten_migration_history_is_refused_rather_than_replayed(tmp_path) -> None:
    path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(path)
    registry._db.execute("UPDATE registry_migrations SET checksum = 'tampered'")
    registry._db.commit()
    registry.close()

    with pytest.raises(RuntimeError, match="does not match"):
        SqliteQuestionRegistry(path)

    # A missing ledger row is equally refused: the history cannot be reconstructed
    # by assumption, because the applied statements may not have been these.
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM registry_migrations")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="not recorded in the ledger"):
        SqliteQuestionRegistry(path)


def test_registry_newer_than_this_application_is_refused(tmp_path) -> None:
    path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(path)
    registry._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    registry.close()
    with pytest.raises(RuntimeError, match="newer than this application"):
        SqliteQuestionRegistry(path)


# --- schema v6: declared decision rationale ----------------------------------
#
# A rationale is a claim by an agent about its own work, stored in its own table
# so it is never interleaved with an answer a reviewer concluded. These tests
# cover the two properties the table has to have that a bare CREATE TABLE does
# not: that one declaration cannot become two records, and that the claim
# protocol behaves here exactly as it does for every other capture table.


def _claim_rationale(registry: SqliteQuestionRegistry, entry_id: str = "rationale-v1-a", **kwargs):
    """Take a claim on one rationale, defaulting the anchor fields."""
    fields = {
        "entry_id": entry_id,
        "repo": "org/repo",
        "pr_number": 42,
        "branch": "feat/x",
        "declared_by": "opencode",
        "declared_model": "opencode/model",
        "source": "declared",
        "revision": 1,
        "revises": None,
        "rationale_text": "Chose exponential backoff because the API rate-limits on 429.",
    }
    fields.update(kwargs)
    return registry.claim_rationale(**fields)  # type: ignore[arg-type]


def test_a_fresh_registry_reaches_the_current_schema_with_the_rationale_table(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")

    tables = {
        row[0]
        for row in registry._db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    indexes = {str(row[1]) for row in registry._db.execute("PRAGMA index_list(rationale_captures)")}
    version = registry._db.execute("PRAGMA user_version").fetchone()[0]
    registry.close()

    assert "rationale_captures" in tables
    assert {"rationale_captures_lookup_idx", "rationale_captures_revision_unique_idx"} <= indexes
    assert version == SCHEMA_VERSION


def test_the_migration_refuses_a_non_idempotent_statement() -> None:
    """The statement guard is the ledger's main defence and a new table must not widen it.

    Everything else in the ledger can be re-derived from the recorded checksums;
    this is the one place that decides what a migration is even allowed to say.
    """
    connection = question_registry_module.sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="not idempotent"):
            question_registry_module._apply_statement(connection, "CREATE TABLE nope (a TEXT)")
    finally:
        connection.close()


def test_racing_claims_on_one_declaration_yield_exactly_one_record(tmp_path) -> None:
    """Two deliveries of the same declaration must not store the same statement twice.

    This is the failure ``semantic_review_event_id`` was written for: GitHub
    re-serialises and re-delivers the same review under a new delivery id, and
    keying on the delivery rather than the content stored it twice.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")

    first = _claim_rationale(registry)
    assert first is not None
    second = _claim_rationale(registry)
    count = registry._db.execute("SELECT COUNT(*) FROM rationale_captures").fetchone()[0]
    registry.close()

    assert second is None, "a second claim on a live lease must be refused"
    assert count == 1


def test_a_second_declaration_is_a_revision_and_never_replaces_the_first(tmp_path) -> None:
    """Earlier revisions are kept: an early intent is worth having precisely
    because it is usually the one a later revision contradicts."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")

    first_token = _claim_rationale(registry, entry_id="rationale-v1-a")
    assert first_token is not None
    assert registry.complete_rationale("rationale-v1-a", first_token) is True
    second_token = _claim_rationale(
        registry,
        entry_id="rationale-v1-b",
        revision=2,
        revises="rationale-v1-a",
        rationale_text="Dropped the jitter; it was hiding the real latency.",
    )
    assert second_token is not None
    assert registry.complete_rationale("rationale-v1-b", second_token) is True

    listed = registry.list_rationales(repo="org/repo")
    registry.close()

    assert [row["revision"] for row in listed] == [1, 2], "revisions must read in order"
    assert listed[0]["rationale_text"].startswith("Chose exponential backoff")
    assert listed[1]["revises"] == "rationale-v1-a"


def test_an_expired_lease_is_reclaimable_and_stale_tokens_stop_working(tmp_path) -> None:
    """A crashed collector must not be able to finalise work someone else took over."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")

    stale = _claim_rationale(registry)
    assert stale is not None
    registry._db.execute(
        "UPDATE rationale_captures SET lease_expires_at = '2000-01-01T00:00:00+00:00'"
    )
    registry._db.commit()

    fresh = _claim_rationale(registry)
    assert fresh is not None
    stale_completes = registry.complete_rationale("rationale-v1-a", stale)
    fresh_completes = registry.complete_rationale("rationale-v1-a", fresh)
    registry.close()

    assert fresh != stale
    assert stale_completes is False, (
        "the previous holder's token must stop working once the lease was reclaimed, "
        "or a crashed collector could finalise work it no longer holds"
    )
    assert fresh_completes is True


def test_a_released_claim_is_retryable_and_keeps_the_error(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")

    token = _claim_rationale(registry)
    assert token is not None
    assert registry.release_rationale("rationale-v1-a", token, "store unreachable") is True
    row = registry._db.execute(
        "SELECT status, last_error FROM rationale_captures WHERE entry_id = 'rationale-v1-a'"
    ).fetchone()
    registry.close()

    assert row[0] == "retryable"
    assert row[1] == "store unreachable", "a failure an operator must act on has to survive"


def test_the_claim_token_never_appears_in_an_enumeration(tmp_path) -> None:
    """A claim token is the capability to release somebody else's claim.

    ``list_rationales`` is read by far more callers than the one worker that
    already holds a token, so the token stays in the claim response only.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _claim_rationale(registry)

    listed = registry.list_rationales(repo="org/repo")
    registry.close()

    assert listed and "claim_token" not in listed[0]


def test_a_derived_id_that_now_describes_something_else_is_refused(tmp_path) -> None:
    """Refuse rather than overwrite: the derivation has moved.

    Overwriting here would replace a stored declaration with a different one
    because of an id collision, which is precisely the silent loss the golden
    identity test exists to prevent reaching production in the first place.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    first = _claim_rationale(registry)
    assert first is not None
    assert registry.complete_rationale("rationale-v1-a", first) is True

    conflicting = _claim_rationale(registry, rationale_text="Something else entirely.")
    stored = registry._db.execute(
        "SELECT rationale_text FROM rationale_captures WHERE entry_id = 'rationale-v1-a'"
    ).fetchone()[0]
    registry.close()

    assert conflicting is None
    assert stored == "Chose exponential backoff because the API rate-limits on 429."


def test_rationale_enumeration_is_bounded_and_filters_by_anchor(tmp_path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    for revision in range(1, 4):
        token = _claim_rationale(
            registry,
            entry_id=f"rationale-v1-{revision}",
            revision=revision,
            revises=None if revision == 1 else "rationale-v1-1",
        )
        assert token is not None
        assert registry.complete_rationale(f"rationale-v1-{revision}", token) is True

    assert len(registry.list_rationales(repo="org/repo", limit=2)) == 2
    assert registry.list_rationales(repo="org/repo", declared_by="nobody") == []
    with pytest.raises(ValueError, match="limit"):
        registry.list_rationales(repo="org/repo", limit=0)
    with pytest.raises(ValueError, match="limit"):
        registry.list_rationales(repo="org/repo", limit=10_000)
    registry.close()


# --- schema v7: the commit a question was asked about --------------------------
#
# A question is a claim about a specific diff, and the column that says which one is
# what lets a reader notice when the answer no longer describes the code. It is also
# the one field on this row that is *absent* by design: nothing about it can be
# reconstructed, and a plausible substitute would be believed. These tests cover the
# three ways that could go wrong — inventing a value, letting a later one overwrite
# it, and skipping the checksum that catches a rewritten migration.


def _v6_column_names() -> tuple[str, ...]:
    """The ``questions`` columns a schema v6 registry has, read off a real one.

    Built by replaying the recorded steps through the application's own
    ``_apply_statement`` rather than by hand-writing the DDL, so it cannot drift
    from what the migrations actually produce. The v6 table is not spelled out here
    on purpose: a hand-written copy would be a second statement of the schema that
    goes stale silently, and the property this test needs — that v6 had no anchor
    column — is better asserted against the file than against a fixture.
    """
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_v3_questions_table())
        for version, statements in sorted(question_registry_module._LEDGERED_MIGRATIONS.items()):
            # Pinned at 7, not at SCHEMA_VERSION: v6 is the schema *before* the
            # anchor migration, frozen. Bumping the current version must not
            # silently redefine what this fixture builds.
            if version >= 7:
                continue
            for statement in statements:
                question_registry_module._apply_statement(connection, statement)
        return tuple(
            str(row[1]) for row in connection.execute("PRAGMA table_info(questions)").fetchall()
        )
    finally:
        connection.close()


def _write_v6_registry(path: Path, *, questions: tuple) -> None:
    """Write a registry file exactly as schema v6 left it, then stop at version 6.

    A v6 file is a v3 file with every step before v7 applied and its checksums
    recorded, and that is how this builds one — through the same guarded statement
    applier the application uses, so nothing is asserted here that the migration
    path would not itself produce. v6's ``rationale_captures`` is the reason the
    steps cannot be skipped: it appears in no ``_SCHEMA`` and exists only because
    migration v6 created it, so a file that omits it is a v5 file wearing a v6
    version number, and the validation below will (correctly) refuse to open it.
    """
    connection = sqlite3.connect(path)
    connection.executescript(_v3_questions_table())
    connection.executemany(
        """
        INSERT INTO questions (
            question_id, identifier, repo, pr_number, pr_url, question_text, category,
            jira_ticket_key, session_id, status, error_message, answered,
            answer_comment_id, question_author, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        questions,
    )
    applied = [
        (version, statements)
        for version, statements in sorted(question_registry_module._LEDGERED_MIGRATIONS.items())
        # Pinned at 7: a v6 file is every step before the anchor migration.
        if version < 7
    ]
    for _version, statements in applied:
        for statement in statements:
            question_registry_module._apply_statement(connection, statement)
    connection.executescript(
        """
        CREATE TABLE registry_migrations (
            version INTEGER PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO registry_migrations (version, checksum, applied_at) VALUES (?, ?, ?)",
        [
            (version, question_registry_module._migration_checksum(statements), "2023-01-01")
            for version, statements in applied
        ],
    )
    connection.execute("PRAGMA user_version = 6")
    connection.commit()
    connection.close()


def test_a_v6_registry_gains_the_anchor_without_being_backfilled(tmp_path: Path) -> None:
    """The absence is the migration's content, so it is what gets asserted.

    A ``DEFAULT`` here would invent a commit for every question already stored, and
    deriving one from the pull request's current head would invent a different and
    far more convincing lie. Both are invisible to the reader afterwards: a sha in
    this column is taken as an anchor, and nothing downstream can tell a recorded
    fact from a reconstruction. So a pre-v7 row must come out the other side with
    ``NULL``, which is the truth — the question was asked and nobody recorded what
    against.
    """
    path = tmp_path / "registry.db"
    _write_v6_registry(
        path,
        questions=(
            (
                "q1",
                100,
                "org/repo",
                1,
                "https://github.com/org/repo/pull/1",
                "Why?",
                "design_decision",
                None,
                "session-1",
                "pending",
                None,
                0,
                None,
                "asker",
                "2023-01-01T00:00:00+00:00",
            ),
        ),
    )
    assert "head_sha" not in _v6_column_names(), (
        "the precondition: if a v6 registry already had the column this migration "
        "would be a no-op and nothing here would be tested"
    )

    registry = SqliteQuestionRegistry(path)
    columns = {row[1] for row in registry._db.execute("PRAGMA table_info(questions)")}
    rows = registry._db.execute(
        "SELECT question_id, question_text, created_at, head_sha FROM questions"
    ).fetchall()
    ledger = registry._db.execute(
        "SELECT version, checksum FROM registry_migrations ORDER BY version"
    ).fetchall()
    registry.close()

    assert "head_sha" in columns, "the column is the deliverable"
    assert rows == [("q1", "Why?", "2023-01-01T00:00:00+00:00", None)], (
        "the existing row must survive untouched and with no invented anchor"
    )
    assert ledger == [
        (version, question_registry_module._migration_checksum(statements))
        for version, statements in sorted(question_registry_module._LEDGERED_MIGRATIONS.items())
    ], "v7 is recorded in the ledger alongside every step before it"


def test_reopening_a_v7_registry_verifies_rather_than_reapplies(tmp_path: Path) -> None:
    """A second open must re-derive nothing, and must still check what was recorded.

    The ledger is the only thing standing between a rewritten migration and a silent
    half-apply: the statements are skipped because the version is present, so a
    mismatch has to be caught by comparing checksums rather than by the statements
    failing. Reopening is the first thing that happens after an interrupted process,
    which is why it is the moment the check matters.
    """
    path = tmp_path / "registry.db"
    _write_v6_registry(
        path,
        questions=(
            (
                "q1",
                100,
                "org/repo",
                1,
                "https://github.com/org/repo/pull/1",
                "Why?",
                "design_decision",
                None,
                "session-1",
                "pending",
                None,
                0,
                None,
                "asker",
                "2023-01-01T00:00:00+00:00",
            ),
        ),
    )
    SqliteQuestionRegistry(path).close()

    reopened = SqliteQuestionRegistry(path)
    versions = [row[0] for row in reopened._db.execute("SELECT version FROM registry_migrations")]
    count = reopened.count_questions()
    # A second open would raise the ADD COLUMN again if the ledger were not doing
    # its job, so the fact that this line is reached at all is part of the assertion.
    reopened.close()

    assert versions == sorted(question_registry_module._LEDGERED_MIGRATIONS)
    assert count == 1


def test_v7_is_refused_when_its_recorded_checksum_disagrees(tmp_path: Path) -> None:
    """The newest step specifically, since the ledger check is what catches a rewrite.

    Rewriting a migration's statements while leaving its version number alone would
    otherwise be invisible: the statements are skipped on a second open precisely
    because the version is present, so nothing would fail except a registry that had
    been migrated by one build and opened by another. The generic tamper test cannot
    show that for v7 in particular — it falsifies every row, so the first
    disagreement it reports is v4's. This falsifies one, and only the version under
    test may be the one that refuses.
    """
    path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(path)
    registry._db.execute("UPDATE registry_migrations SET checksum = 'tampered' WHERE version = 7")
    registry._db.commit()
    registry.close()

    with pytest.raises(RuntimeError, match=r"migration 7 .* does not match"):
        SqliteQuestionRegistry(path)


def _record_anchored(
    registry: SqliteQuestionRegistry,
    question_id: str,
    *,
    head_sha: str | None,
    comment_id: int | None = None,
    repo: str = "org/repo",
    pr_number: int = 1,
) -> None:
    """Record one pending question against a head, or against none."""
    registry.record_question(
        question_id=question_id,
        github_comment_id=comment_id if comment_id is not None else 1000,
        repo=repo,
        pr_number=pr_number,
        pr_url=f"https://github.com/{repo}/pull/{pr_number}",
        question_text="Why?",
        question_category="design_decision",
        head_sha=head_sha,
    )


def test_a_recorded_question_carries_the_head_it_was_asked_against(tmp_path: Path) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record_anchored(registry, "q1", head_sha="a" * 40, comment_id=100)

    by_comment = registry.get_question_by_comment_id(100)
    by_id = registry.get_question_by_id("q1")
    registry.close()

    assert by_comment is not None and by_comment["head_sha"] == "a" * 40
    assert by_id is not None and by_id["head_sha"] == "a" * 40


def test_a_question_with_no_recorded_head_reads_as_absent(tmp_path: Path) -> None:
    """``None`` is a reportable answer, so it has to survive the round trip.

    A question asked by a caller with no head to give, and a question that predates
    v7, both land here. Collapsing either into a placeholder string would put a
    value in the column that reads as a commit.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _record_anchored(registry, "q1", head_sha=None, comment_id=100)

    row = registry.get_question_by_comment_id(100)
    registry.close()

    assert row is not None
    assert row["head_sha"] is None


def test_a_re_ask_never_relabels_the_anchor_a_comment_was_posted_against(tmp_path: Path) -> None:
    """First head recorded wins, and an absent one never erases it.

    Question ids are derived from the pull request and the question text, so the same
    question re-asked after a push lands on the same row. Last-write-wins would put
    the *later* head on a comment the reviewer has been answering since the earlier
    one — an anchor that is a plausible sha and the wrong one, and it overwrites the
    drift a reader needs in order to see that the record is stale. The recorded head
    is a property of the ask, not of the most recent run that happened to repeat it.
    """
    path = tmp_path / "registry.db"
    registry = SqliteQuestionRegistry(path)
    _record_anchored(registry, "q1", head_sha="a" * 40, comment_id=100)
    # A push, then `ask` again against the new head, then a run whose caller had no
    # head to record at all.
    _record_anchored(registry, "q1", head_sha="b" * 40, comment_id=100)
    _record_anchored(registry, "q1", head_sha=None, comment_id=100)

    row = registry.get_question_by_comment_id(100)
    registry.close()

    assert row is not None
    assert row["head_sha"] == "a" * 40


def test_the_registry_file_and_its_directory_are_owner_only(tmp_path: Path) -> None:
    """The registry holds question text, author logins and Jira keys in plaintext.

    The outbox has had this assertion since its own hardening; the registry never got
    one, so a regression that widened either mode would have been caught on one side
    of the seam and not the other. The mode is asserted rather than the umask, because
    what matters is the bytes on disk after the process is gone, not how they got
    there -- a permissive umask is the easy case and this is the one that survives it.
    """
    nested = tmp_path / "state" / "registry.db"

    with SqliteQuestionRegistry(nested) as registry:
        _record(registry, "q1", repo="acme/widgets", pr_number=7)

    assert stat.S_IMODE(nested.stat().st_mode) == 0o600
    assert stat.S_IMODE(nested.parent.stat().st_mode) == 0o700


def test_a_registry_that_is_already_too_permissive_is_tightened(tmp_path: Path) -> None:
    """An existing world-readable file is narrowed, not merely left alone.

    The common way this state becomes exposed is a file created by an older build or
    by hand, so the check that matters is the one on an existing file. Opening with
    ``O_CREAT`` and a mode does not touch the mode of a file that is already there --
    which is why the constructor fchmods the descriptor rather than relying on the
    open mode alone.
    """
    path = tmp_path / "registry.db"
    path.write_bytes(b"")
    path.chmod(0o666)

    with SqliteQuestionRegistry(path):
        pass

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_state_and_permissions_survive_a_close_and_reopen(tmp_path: Path) -> None:
    """Criterion 3: the data is still there, and still not readable by anyone else.

    Reopening is where a permission fix would be lost -- a second constructor that
    opens the existing file without re-applying the mode would leave the file exactly
    as the first one left it and quietly stop enforcing anything.
    """
    path = tmp_path / "registry.db"

    with SqliteQuestionRegistry(path) as registry:
        _record(registry, "q1", repo="acme/widgets", pr_number=7)
        _record(registry, "q2", repo="acme/widgets", pr_number=8)

    with SqliteQuestionRegistry(path) as reopened:
        listed = reopened.list_questions(repo="acme/widgets", status="pending")
        assert {q["question_id"] for q in listed} == {"q1", "q2"}

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
