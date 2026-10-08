"""Local registry for Kojutsu's ingestion state.

This is deliberately **not** the Tanseki knowledge store. It holds operational
state that the capture pipeline needs before/while a decision exists:

- posted questions and their GitHub-comment mapping,
- answer dedupe (a question is answered once; a comment is processed once),
- PR state-change dedupe,
- review sessions.

It is backed by SQLite (stdlib), so no external state service is required.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import stat
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from kojutsu.core.sqlite_durability import configure_durable_connection
from kojutsu.models import ReviewSession

if TYPE_CHECKING:
    from kojutsu.config import Settings

from kojutsu.core.registry_identity import (
    ANSWER_IDENTITY_DOMAIN,
    ANSWER_IDENTITY_VERSION,
    CLARIFICATION_IDENTITY_DOMAIN,
    CLARIFICATION_IDENTITY_VERSION,
    EVALUATION_IDENTITY_DOMAIN,
    EVALUATION_IDENTITY_VERSION,
    RATIONALE_IDENTITY_DOMAIN,
    RATIONALE_IDENTITY_VERSION,
    stable_answer_entry_id,
    stable_clarification_entry_id,
    stable_evaluation_entry_id,
    stable_rationale_entry_id,
)
from kojutsu.core.registry_schema import (
    _LEDGERED_MIGRATIONS,
    _QUESTION_LIST_COLUMNS,
    _SCHEMA,
    ANSWER_LEASE,
    DEFAULT_MAX_QUESTION_ATTEMPTS,
    DELIVERY_LEASE,
    DELIVERY_RETENTION_DAYS,
    MAX_QUESTION_ERROR_LENGTH,
    MAX_QUESTION_LIST_LIMIT,
    PR_EVENT_LEASE,
    QUESTION_LEASE,
    QUESTION_STATUSES,
    RATIONALE_LEASE,
    SCHEMA_VERSION,
    TERMINAL_QUESTION_STATUSES,
    _apply_statement,
    _bounded_question_limit,
    _migrate_schema,
    _migration_checksum,
    _now,
    _question_row,
)

#: Names that moved to ``registry_identity`` / ``registry_schema`` but stay
#: importable here. The split is organisational, not a new API: every existing
#: ``from kojutsu.core.question_registry import X`` keeps working, and this
#: list is the surface that promise covers.
__all__ = [
    "ANSWER_IDENTITY_DOMAIN",
    "ANSWER_IDENTITY_VERSION",
    "CLARIFICATION_IDENTITY_DOMAIN",
    "CLARIFICATION_IDENTITY_VERSION",
    "EVALUATION_IDENTITY_DOMAIN",
    "EVALUATION_IDENTITY_VERSION",
    "QUESTION_STATUSES",
    "RATIONALE_IDENTITY_DOMAIN",
    "RATIONALE_IDENTITY_VERSION",
    "SCHEMA_VERSION",
    "TERMINAL_QUESTION_STATUSES",
    "_LEDGERED_MIGRATIONS",
    "DeliveryClaim",
    "QuestionRegistry",
    "SqliteQuestionRegistry",
    "_apply_statement",
    "_migrate_schema",
    "_migration_checksum",
    "build_registry",
    "stable_answer_entry_id",
    "stable_clarification_entry_id",
    "stable_evaluation_entry_id",
    "stable_rationale_entry_id",
]

DeliveryClaim = Literal["duplicate", "active", "conflict"] | str


class QuestionRegistry(Protocol):
    """Ingestion-state operations the capture pipeline depends on."""

    def __enter__(self) -> QuestionRegistry: ...

    def __exit__(self, *exc: object) -> None: ...

    def record_question(
        self,
        *,
        question_id: str,
        github_comment_id: int,
        repo: str,
        pr_number: int,
        pr_url: str,
        question_text: str,
        question_category: str,
        jira_ticket_key: str | None = None,
        session_id: str | None = None,
        question_author: str | None = None,
        head_sha: str | None = None,
        error_message: str | None = None,
        status: str = "pending",
    ) -> None: ...

    def get_question_by_comment_id(self, github_comment_id: int) -> dict[str, Any] | None: ...

    def get_question_by_id(self, question_id: str) -> dict[str, Any] | None: ...

    def list_questions(
        self,
        *,
        status: str | None = None,
        repo: str | None = None,
        pr_number: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def count_questions(
        self,
        *,
        status: str | None = None,
        repo: str | None = None,
        pr_number: int | None = None,
    ) -> int: ...

    def claim_question_for_answer(
        self,
        question_id: str,
        assignee: str,
        *,
        ttl: timedelta = QUESTION_LEASE,
        max_attempts: int = DEFAULT_MAX_QUESTION_ATTEMPTS,
    ) -> str | None: ...

    def release_question(self, question_id: str, claim_token: str, reason: str) -> bool: ...

    def claim_answer(
        self,
        *,
        parent_comment_id: int,
        answer_comment_id: int,
        repo: str,
        pr_number: int,
        question_id: str,
        entry_id: str,
    ) -> str | None: ...

    def complete_answer(self, answer_comment_id: int, claim_token: str) -> bool: ...

    def release_answer(self, answer_comment_id: int, claim_token: str) -> bool: ...

    def update_question_status(
        self, github_comment_id: int, status: str, error_message: str | None = None
    ) -> None: ...

    def is_question_answered(self, github_comment_id: int) -> bool: ...

    def mark_question_answered(self, github_comment_id: int, answer_comment_id: int) -> None: ...

    def answer_comment_seen(self, answer_comment_id: int) -> bool: ...

    def pr_state_change_seen(
        self, repo: str, pr_number: int, action: str, event_id: str | None = None
    ) -> bool: ...

    def mark_pr_state_change(
        self, repo: str, pr_number: int, action: str, event_id: str | None = None
    ) -> None: ...

    def claim_pr_state_change(
        self, repo: str, pr_number: int, action: str, event_id: str | None = None
    ) -> str | None: ...

    def claim_review_capture(
        self,
        *,
        review_event_id: str,
        repo: str,
        pr_number: int,
        review_id: int,
        comment_id: int | None,
        kind: str,
        author: str | None,
    ) -> str | None: ...

    def supersede_questions_for_pr(
        self, repo: str, pr_number: int, head_sha: str | None = None
    ) -> int: ...

    def complete_review_capture(self, review_event_id: str, claim_token: str) -> bool: ...

    def release_review_capture(
        self, review_event_id: str, claim_token: str, error: str
    ) -> bool: ...

    def claim_rationale(
        self,
        *,
        entry_id: str,
        repo: str,
        pr_number: int | None,
        branch: str,
        declared_by: str,
        declared_model: str | None,
        source: str,
        revision: int,
        revises: str | None,
        rationale_text: str,
    ) -> str | None: ...

    def complete_rationale(self, entry_id: str, claim_token: str) -> bool: ...

    def release_rationale(self, entry_id: str, claim_token: str, error: str) -> bool: ...

    def list_rationales(
        self,
        *,
        repo: str,
        pr_number: int | None = None,
        declared_by: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def complete_pr_state_change(self, event_id: str, claim_token: str) -> bool: ...

    def release_pr_state_change(self, event_id: str, claim_token: str, error: str) -> bool: ...

    def get_backfill_receipt(self, *, repo: str, pr_number: int) -> dict[str, Any] | None: ...

    def record_backfill_receipt(
        self,
        *,
        repo: str,
        pr_number: int,
        observed_updated_at: datetime,
        window_since: datetime,
        window_until: datetime | None,
        authorized_associations: frozenset[str] | None,
        finished: bool,
    ) -> None: ...

    def record_session(self, session: ReviewSession) -> None: ...

    def claim_delivery(
        self, delivery_id: str, repo: str, event: str, payload_hash: str
    ) -> DeliveryClaim: ...

    def complete_delivery(self, delivery_id: str, claim_token: str) -> bool: ...

    def release_delivery(self, delivery_id: str, claim_token: str, error: str) -> bool: ...

    def cleanup_deliveries(
        self, retention_days: int = DELIVERY_RETENTION_DAYS, *, limit: int = 1_000
    ) -> int: ...


class SqliteQuestionRegistry:
    """SQLite-backed :class:`QuestionRegistry`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        parent_created = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent_created:
            self.path.parent.chmod(0o700)
        self._secure_database_file()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        self._lock = threading.RLock()
        try:
            configure_durable_connection(self._db)
            self._db.executescript(_SCHEMA)
            _migrate_schema(self._db)
        except Exception:
            self._db.close()
            raise

    def _secure_database_file(self) -> None:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise PermissionError("Question registry path must be a regular file")
            if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
                raise PermissionError("Question registry file must be owned by the current user")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def cleanup_deliveries(
        self, retention_days: int = DELIVERY_RETENTION_DAYS, *, limit: int = 1_000
    ) -> int:
        if isinstance(retention_days, bool) or not isinstance(retention_days, int):
            raise ValueError("retention_days must be a non-negative integer")
        if retention_days < 0:
            raise ValueError("retention_days must be non-negative")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be a non-negative integer")
        bounded_limit = max(0, min(limit, 10_000))
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                rows = self._db.execute(
                    """
                    SELECT delivery_id
                    FROM webhook_deliveries
                    WHERE (
                        status = 'completed'
                        AND completed_at IS NOT NULL
                        AND completed_at <= ?
                    ) OR (
                        status = 'retryable'
                        AND COALESCE(lease_expires_at, created_at) <= ?
                    ) OR (
                        status = 'processing'
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= ?
                    )
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (cutoff, cutoff, cutoff, bounded_limit),
                ).fetchall()
                if rows:
                    placeholders = ", ".join("?" for _ in rows)
                    self._db.execute(
                        f"DELETE FROM webhook_deliveries WHERE delivery_id IN ({placeholders})",
                        tuple(str(row[0]) for row in rows),
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            return len(rows)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> SqliteQuestionRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record_question(
        self,
        *,
        question_id: str,
        github_comment_id: int,
        repo: str,
        pr_number: int,
        pr_url: str,
        question_text: str,
        question_category: str,
        jira_ticket_key: str | None = None,
        session_id: str | None = None,
        question_author: str | None = None,
        head_sha: str | None = None,
        error_message: str | None = None,
        status: str = "pending",
    ) -> None:
        """Record that a question comment exists, idempotently.

        Re-recording a question refreshes its comment mapping, but never walks an
        ``answered`` question back to ``pending``. Re-running ``ask`` against the
        same question id is an ordinary operation, and if it cleared the terminal
        state then the answer-dedupe gate would let the same answer be captured a
        second time. Queue fields (assignee, attempts, lease) are left alone: this
        records that the question was posted, not that anybody is working on it.

        ``head_sha`` is the commit the question was asked *about*, and it is
        written once. Question ids are derived from the pull request and the
        question text, so the same logical question re-asked after a push collides
        on the primary key -- and a later head is not a correction to the earlier
        one, it is a different commit. Last-write-wins would relabel the anchor of
        a comment that was posted against the first of them, erasing the very drift
        the column exists to expose; an absent value must not erase a recorded one
        either, for the same reason ``question_author`` is coalesced below. So the
        first head anybody recorded is the head the reviewer is answering.
        """
        if status not in QUESTION_STATUSES:
            raise ValueError(f"Unknown question status: {status!r}")
        now = _now()
        with self._lock:
            self._db.execute(
                """
                INSERT INTO questions (
                    question_id, identifier, repo, pr_number, pr_url, question_text,
                    category, jira_ticket_key, session_id, question_author, head_sha, status,
                    error_message, answered, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(question_id) DO UPDATE SET
                    identifier = excluded.identifier,
                    status = CASE
                        WHEN questions.status = 'answered' THEN 'answered'
                        ELSE excluded.status
                    END,
                    answered = CASE
                        WHEN questions.status = 'answered' THEN 1
                        ELSE questions.answered
                    END,
                    error_message = CASE
                        WHEN questions.status = 'answered' THEN questions.error_message
                        ELSE excluded.error_message
                    END,
                    question_author = COALESCE(excluded.question_author, questions.question_author),
                    head_sha = COALESCE(questions.head_sha, excluded.head_sha),
                    updated_at = excluded.updated_at
                """,
                (
                    question_id,
                    github_comment_id,
                    repo,
                    pr_number,
                    pr_url,
                    question_text,
                    question_category,
                    jira_ticket_key,
                    session_id,
                    question_author,
                    head_sha,
                    status,
                    error_message,
                    now,
                    now,
                ),
            )
            self._db.commit()

    def get_question_by_comment_id(self, github_comment_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT question_id, question_text, category, pr_url, jira_ticket_key,
                       session_id, question_author, head_sha
                FROM questions WHERE identifier = ?
                """,
                (github_comment_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "question_id": row[0],
            "question_text": row[1],
            "category": row[2],
            "pr_url": row[3],
            "jira_ticket_key": row[4],
            "session_id": row[5],
            "question_author": row[6],
            "head_sha": row[7],
        }

    def get_question_by_id(self, question_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT identifier, question_text, category, pr_url, jira_ticket_key,
                       session_id, question_author, head_sha
                FROM questions WHERE question_id = ?
                """,
                (question_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "parent_comment_id": row[0],
            "question_text": row[1],
            "category": row[2],
            "pr_url": row[3],
            "jira_ticket_key": row[4],
            "session_id": row[5],
            "question_author": row[6],
            "head_sha": row[7],
        }

    def _question_filter(
        self, status: str | None, repo: str | None, pr_number: int | None
    ) -> tuple[str, list[Any]]:
        """Build the shared WHERE clause for the ``questions`` read side."""
        if status is not None and status not in QUESTION_STATUSES:
            raise ValueError(f"Unknown question status: {status!r}")
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (("status", status), ("repo", repo), ("pr_number", pr_number)):
            if value is None:
                continue
            clauses.append(f"{column} = ?")
            parameters.append(value)
        if not clauses:
            return "", []
        return f" WHERE {' AND '.join(clauses)}", parameters

    def list_questions(
        self,
        *,
        status: str | None = None,
        repo: str | None = None,
        pr_number: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return outstanding capture work, bounded and in a deterministic order.

        This is the enumeration surface the registry never had: without it a
        worker can only reach a question it already knows the id of, which is not
        a queue. Every filter is optional and combines conjunctively.

        ``repo`` matches exactly, not case-insensitively, so that the
        ``questions_work_idx`` index stays usable. GitHub treats owner/repo
        case-insensitively, so a differently-cased filter legitimately matches
        nothing here; that fails toward showing less work rather than mixing
        another repository's questions into a page.

        The order is ``(status, repo, pr_number, question_id)``. ``question_id``
        is the primary key, so the order is total: paging with a limit over the
        same filter cannot skip or repeat a row. Returns an empty list rather
        than raising when a bounded limit of zero is requested.
        """
        where, parameters = self._question_filter(status, repo, pr_number)
        bounded_limit = _bounded_question_limit(limit)
        if bounded_limit == 0:
            return []
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT {", ".join(_QUESTION_LIST_COLUMNS)}
                FROM questions{where}
                ORDER BY status, repo, pr_number, question_id
                LIMIT ?
                """,
                (*parameters, bounded_limit),
            ).fetchall()
        return [_question_row(row) for row in rows]

    def count_questions(
        self,
        *,
        status: str | None = None,
        repo: str | None = None,
        pr_number: int | None = None,
    ) -> int:
        """Count the questions matching the same filters as :meth:`list_questions`.

        Separated from the listing because the listing is bounded by design, so a
        caller cannot infer the size of a queue from it. Counting a large queue
        with ``list_questions`` would read up to a page and report a capped
        number, which reads exactly like a queue that is not growing.
        """
        where, parameters = self._question_filter(status, repo, pr_number)
        with self._lock:
            row = self._db.execute(f"SELECT COUNT(*) FROM questions{where}", parameters).fetchone()
        return int(row[0]) if row else 0

    def claim_question_for_answer(
        self,
        question_id: str,
        assignee: str,
        *,
        ttl: timedelta = QUESTION_LEASE,
        max_attempts: int = DEFAULT_MAX_QUESTION_ATTEMPTS,
    ) -> str | None:
        """Take a lease on one question, returning a claim token or ``None``.

        ``None`` means "do not do this work", and covers every refusal: the
        question does not exist, is already answered, is terminal, is held by a
        live claim, or has exhausted ``max_attempts``. Callers must treat it as a
        normal outcome rather than an error, because under concurrency it is the
        expected result for all but one racing worker.

        A claim is stealable once its lease expires, so a worker that dies
        mid-question cannot strand it. A ``claimed`` row with no lease at all is
        also treated as stealable: a claim that cannot be shown to be live is not
        honoured, which keeps a half-written claim from blocking the queue
        forever.

        Reaching ``max_attempts`` moves the question to ``failed`` and refuses, so
        exhaustion is a visible terminal state in :meth:`list_questions` rather
        than a question that silently declines to be claimed forever.
        """
        if not isinstance(assignee, str) or not assignee.strip():
            raise ValueError("assignee must be a non-empty string")
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise ValueError("ttl must be a positive timedelta")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        now = datetime.now(UTC)
        lease_expires_at = (now + ttl).isoformat()
        now_text = now.isoformat()
        claim_token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """
                    SELECT status, attempts, claim_token, lease_expires_at
                    FROM questions WHERE question_id = ?
                    """,
                    (question_id,),
                ).fetchone()
                if row is None:
                    self._db.rollback()
                    return None
                status = str(row[0])
                attempts = int(row[1])
                if status in TERMINAL_QUESTION_STATUSES or status not in ("pending", "claimed"):
                    self._db.rollback()
                    return None
                if status == "claimed" and row[3] is not None and str(row[3]) > now_text:
                    self._db.rollback()
                    return None
                if attempts >= max_attempts:
                    self._db.execute(
                        """
                        UPDATE questions
                        SET status = 'failed', claim_token = NULL, claimed_at = NULL,
                            lease_expires_at = NULL, updated_at = ?
                        WHERE question_id = ?
                        """,
                        (now_text, question_id),
                    )
                    self._db.commit()
                    return None
                self._db.execute(
                    """
                    UPDATE questions
                    SET status = 'claimed', assignee = ?, claim_token = ?, claimed_at = ?,
                        lease_expires_at = ?, attempts = ?, last_attempt_at = ?,
                        last_error = NULL, updated_at = ?
                    WHERE question_id = ?
                    """,
                    (
                        assignee.strip(),
                        claim_token,
                        now_text,
                        lease_expires_at,
                        attempts + 1,
                        now_text,
                        now_text,
                        question_id,
                    ),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return claim_token

    def release_question(self, question_id: str, claim_token: str, reason: str) -> bool:
        """Hand a claimed question back to the queue after a failed attempt.

        Only the holder of the current claim token can release, and only a
        ``claimed`` question can be released, so a worker whose lease was already
        stolen cannot clobber the new owner's work. ``attempts`` is deliberately
        preserved: releasing is not a reset, and a question that keeps failing
        must still reach its ceiling instead of looping forever.

        Returns ``False`` when the token does not match the live claim, which
        includes the case where the question has since been answered.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE questions
                    SET status = 'pending', assignee = NULL, claim_token = NULL,
                        claimed_at = NULL, lease_expires_at = NULL, last_error = ?,
                        updated_at = ?
                    WHERE question_id = ? AND status = 'claimed' AND claim_token = ?
                    """,
                    (reason[:MAX_QUESTION_ERROR_LENGTH], now, question_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def claim_review_capture(
        self,
        *,
        review_event_id: str,
        repo: str,
        pr_number: int,
        review_id: int,
        comment_id: int | None,
        kind: str,
        author: str | None,
    ) -> str | None:
        """Take a lease on one review record, or ``None`` if it is already captured.

        Deduplication has two independent keys here. ``review_event_id`` is the
        semantic identity, so a redelivered or re-serialised event collapses to one
        record. ``comment_id`` is separately unique-indexed, which closes the gap
        where the same inline comment arrives under two different review payload
        shapes: the second insert hits the unique index and is refused rather than
        producing a duplicate entry.
        """
        if not review_event_id:
            raise ValueError("review_event_id must not be empty")
        now = datetime.now(UTC)
        lease = now + PR_EVENT_LEASE
        claim_token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """
                    SELECT repo, pr_number, review_id, comment_id, kind, status, lease_expires_at
                    FROM review_captures WHERE review_event_id = ?
                    """,
                    (review_event_id,),
                ).fetchone()
                if row is not None:
                    if (row[0], row[1], row[2], row[3], row[4]) != (
                        repo,
                        pr_number,
                        review_id,
                        comment_id,
                        kind,
                    ):
                        self._db.rollback()
                        return None
                    if row[5] == "completed":
                        self._db.rollback()
                        return None
                    if row[5] == "processing" and row[6] and row[6] > now.isoformat():
                        self._db.rollback()
                        return None
                    self._db.execute(
                        """
                        UPDATE review_captures
                        SET status = 'processing', claimed_at = ?, lease_expires_at = ?,
                            last_error = NULL, completed_at = NULL, claim_token = ?
                        WHERE review_event_id = ?
                        """,
                        (now.isoformat(), lease.isoformat(), claim_token, review_event_id),
                    )
                else:
                    self._db.execute(
                        """
                        INSERT INTO review_captures (
                            review_event_id, repo, pr_number, review_id, comment_id, kind,
                            author, status, claimed_at, lease_expires_at, last_error,
                            completed_at, claim_token
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?, ?, NULL, NULL, ?)
                        """,
                        (
                            review_event_id,
                            repo,
                            pr_number,
                            review_id,
                            comment_id,
                            kind,
                            author,
                            now.isoformat(),
                            lease.isoformat(),
                            claim_token,
                        ),
                    )
                self._db.commit()
            except sqlite3.IntegrityError:
                # A comment already captured under a different event identity.
                self._db.rollback()
                return None
            except Exception:
                self._db.rollback()
                raise
        return claim_token

    def claim_rationale(
        self,
        *,
        entry_id: str,
        repo: str,
        pr_number: int | None,
        branch: str,
        declared_by: str,
        declared_model: str | None,
        source: str,
        revision: int,
        revises: str | None,
        rationale_text: str,
    ) -> str | None:
        """Take a lease on one declared rationale, or ``None`` if already captured.

        Deduplication has two independent keys, mirroring
        :meth:`claim_review_capture`. ``entry_id`` is the semantic identity derived
        from the anchor, so a re-delivery collapses to one record. The unique index
        over ``(repo, pr_number, branch, declared_by, revision)`` closes the gap
        where the same declaration arrives under two different derived ids: the
        second insert hits the index and is refused rather than storing the same
        statement twice.

        A claim whose lease has expired may be taken again, and the previous
        holder's ``claim_token`` stops working at that moment, so a crashed
        collector cannot be completed by a caller that no longer holds the work.
        """
        if not entry_id:
            raise ValueError("entry_id must not be empty")
        now = datetime.now(UTC)
        lease = now + RATIONALE_LEASE
        claim_token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """
                    SELECT repo, pr_number, branch, declared_by, source, revision, revises,
                           status, lease_expires_at
                    FROM rationale_captures WHERE entry_id = ?
                    """,
                    (entry_id,),
                ).fetchone()
                if row is not None:
                    if (row[0], row[1], row[2], row[3], row[4], row[5], row[6]) != (
                        repo,
                        pr_number,
                        branch,
                        declared_by,
                        source,
                        revision,
                        revises,
                    ):
                        # The same id now describes something else, which means the
                        # derivation moved. Refuse rather than overwrite a stored
                        # record with a different one.
                        self._db.rollback()
                        return None
                    if row[7] == "completed":
                        self._db.rollback()
                        return None
                    if row[7] == "processing" and row[8] and row[8] > now.isoformat():
                        self._db.rollback()
                        return None
                    self._db.execute(
                        """
                        UPDATE rationale_captures
                        SET status = 'processing', claimed_at = ?, lease_expires_at = ?,
                            last_error = NULL, completed_at = NULL, claim_token = ?
                        WHERE entry_id = ?
                        """,
                        (now.isoformat(), lease.isoformat(), claim_token, entry_id),
                    )
                else:
                    self._db.execute(
                        """
                        INSERT INTO rationale_captures (
                            entry_id, repo, pr_number, branch, declared_by, declared_model,
                            source, revision, revises, rationale_text, status, claimed_at,
                            lease_expires_at, last_error, completed_at, claim_token
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing', ?, ?, NULL, NULL, ?)
                        """,
                        (
                            entry_id,
                            repo,
                            pr_number,
                            branch,
                            declared_by,
                            declared_model,
                            source,
                            revision,
                            revises,
                            rationale_text,
                            now.isoformat(),
                            lease.isoformat(),
                            claim_token,
                        ),
                    )
                self._db.commit()
            except sqlite3.IntegrityError:
                # A declaration already captured under a different derived id.
                self._db.rollback()
                return None
            except Exception:
                self._db.rollback()
                raise
        return claim_token

    def complete_rationale(self, entry_id: str, claim_token: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE rationale_captures
                    SET status = 'completed', completed_at = ?, lease_expires_at = NULL,
                        last_error = NULL
                    WHERE entry_id = ? AND status = 'processing' AND claim_token = ?
                    """,
                    (now, entry_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def release_rationale(self, entry_id: str, claim_token: str, error: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE rationale_captures
                    SET status = 'retryable', lease_expires_at = ?, last_error = ?
                    WHERE entry_id = ? AND status = 'processing' AND claim_token = ?
                    """,
                    (now, error[:1000], entry_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def list_rationales(
        self,
        *,
        repo: str,
        pr_number: int | None = None,
        declared_by: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Enumerate stored rationales, oldest revision first, without the token.

        ``claim_token`` is the capability to release or finalise somebody else's
        claim, so it is excluded here for the same reason it is excluded from
        :meth:`list_questions`: an enumeration surface is read by far more callers
        than the single worker that already holds one. Whether a claim is held is
        reported by ``status`` and ``lease_expires_at``.

        Ordering is by revision, so a reader sees the sequence a declaration went
        through rather than only its final word. Earlier revisions are not
        discarded: an intent captured at the start of implementation is worth
        keeping precisely because it is often the one a later revision contradicts.
        """
        if limit < 1 or limit > MAX_QUESTION_LIST_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_QUESTION_LIST_LIMIT}")
        clauses = ["repo = ?"]
        params: list[Any] = [repo]
        if pr_number is not None:
            clauses.append("pr_number = ?")
            params.append(pr_number)
        if declared_by is not None:
            clauses.append("declared_by = ?")
            params.append(declared_by)
        params.append(limit)
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT entry_id, repo, pr_number, branch, declared_by, declared_model, source,
                       revision, revises, rationale_text, status, completed_at
                FROM rationale_captures
                WHERE {" AND ".join(clauses)}
                ORDER BY revision ASC, entry_id ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [
            {
                "entry_id": str(row[0]),
                "repo": str(row[1]),
                "pr_number": row[2],
                "branch": row[3],
                "declared_by": str(row[4]),
                "declared_model": row[5],
                "source": str(row[6]),
                "revision": int(row[7]),
                "revises": row[8],
                "rationale_text": str(row[9]),
                "status": str(row[10]),
                "completed_at": row[11],
            }
            for row in rows
        ]

    def supersede_questions_for_pr(
        self, repo: str, pr_number: int, head_sha: str | None = None
    ) -> int:
        """Mark this PR's outstanding questions as asked about an older revision.

        Only ``pending`` questions are superseded. A question already answered
        describes a decision someone actually made about a specific diff, and
        rewriting that record's state would erase real evidence to tidy a queue.
        A claimed question is likewise left alone: a worker is mid-flight on it, and
        silently invalidating its claim is worse than leaving it outstanding.

        ``head_sha`` is the commit that *overtook* the question, so it belongs in
        the message and not in the row's ``head_sha``, which is the commit the
        question was asked about. Writing this one over that one would destroy the
        only record of what the question was aimed at, in the exact situation where
        a reader most needs it -- which is also the situation this function exists
        to make visible.

        Returns the number of questions superseded, so the caller can report
        honestly rather than claiming work it did not do.
        """
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE questions
                    SET status = 'superseded', last_error = ?, updated_at = ?
                    WHERE repo = ? AND pr_number = ? AND status = 'pending'
                    """,
                    (
                        f"superseded by a new commit: {head_sha}"
                        if head_sha
                        else "superseded by a new commit",
                        now,
                        repo,
                        pr_number,
                    ),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount

    def complete_review_capture(self, review_event_id: str, claim_token: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE review_captures
                    SET status = 'completed', completed_at = ?, lease_expires_at = NULL,
                        last_error = NULL
                    WHERE review_event_id = ? AND status = 'processing' AND claim_token = ?
                    """,
                    (now, review_event_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def release_review_capture(self, review_event_id: str, claim_token: str, error: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE review_captures
                    SET status = 'retryable', lease_expires_at = ?, last_error = ?
                    WHERE review_event_id = ? AND status = 'processing' AND claim_token = ?
                    """,
                    (now, error[:1000], review_event_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def claim_answer(
        self,
        *,
        parent_comment_id: int,
        answer_comment_id: int,
        repo: str,
        pr_number: int,
        question_id: str,
        entry_id: str,
    ) -> str | None:
        now = datetime.now(UTC)
        lease = now + ANSWER_LEASE
        claim_token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                question = self._db.execute(
                    """
                    SELECT answered, repo, pr_number
                    FROM questions WHERE question_id = ? AND identifier = ?
                    """,
                    (question_id, parent_comment_id),
                ).fetchone()
                if (
                    question is None
                    or question[0]
                    or not isinstance(question[1], str)
                    or question[1].casefold() != repo.casefold()
                    or question[2] != pr_number
                ):
                    self._db.rollback()
                    return None
                row = self._db.execute(
                    """
                    SELECT parent_comment_id, question_id, entry_id, status, lease_expires_at
                    FROM answer_captures WHERE answer_comment_id = ?
                    """,
                    (answer_comment_id,),
                ).fetchone()
                if row is not None:
                    if (
                        row[0] != parent_comment_id
                        or row[1] != question_id
                        or row[2] != entry_id
                        or row[3] == "completed"
                    ):
                        self._db.rollback()
                        return None
                    if row[3] == "processing" and row[4] and row[4] > now.isoformat():
                        self._db.rollback()
                        return None
                    self._db.execute(
                        """
                        UPDATE answer_captures
                        SET status = 'processing', claimed_at = ?, lease_expires_at = ?,
                            last_error = NULL, completed_at = NULL, claim_token = ?
                        WHERE answer_comment_id = ?
                        """,
                        (
                            now.isoformat(),
                            lease.isoformat(),
                            claim_token,
                            answer_comment_id,
                        ),
                    )
                else:
                    self._db.execute(
                        """
                        INSERT INTO answer_captures (
                            answer_comment_id, parent_comment_id, question_id, entry_id, status,
                            claimed_at, lease_expires_at, last_error, completed_at, claim_token
                        ) VALUES (?, ?, ?, ?, 'processing', ?, ?, NULL, NULL, ?)
                        """,
                        (
                            answer_comment_id,
                            parent_comment_id,
                            question_id,
                            entry_id,
                            now.isoformat(),
                            lease.isoformat(),
                            claim_token,
                        ),
                    )
                self._db.commit()
            except sqlite3.IntegrityError:
                self._db.rollback()
                return None
            except Exception:
                self._db.rollback()
                raise
        return claim_token

    def complete_answer(self, answer_comment_id: int, claim_token: str) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """
                    SELECT parent_comment_id
                    FROM answer_captures
                    WHERE answer_comment_id = ?
                      AND status = 'processing'
                      AND claim_token = ?
                    """,
                    (answer_comment_id, claim_token),
                ).fetchone()
                if row is None:
                    self._db.rollback()
                    return False
                now = _now()
                updated = self._db.execute(
                    """
                    UPDATE answer_captures
                    SET status = 'completed', completed_at = ?, lease_expires_at = ?
                    WHERE answer_comment_id = ?
                      AND status = 'processing'
                      AND claim_token = ?
                    """,
                    (now, now, answer_comment_id, claim_token),
                )
                if updated.rowcount != 1:
                    self._db.rollback()
                    return False
                # The dedupe gate. Keyed on `status` rather than the legacy
                # `answered` column so the queue and the gate can never disagree
                # about whether an answer was already captured; rowcount == 1 is
                # what makes this safe under concurrency, since a second writer
                # for the same parent finds no un-answered row to update.
                answered = self._db.execute(
                    """
                    UPDATE questions
                    SET answered = 1, answer_comment_id = ?, status = 'answered',
                        answered_at = ?, claim_token = NULL, claimed_at = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE identifier = ? AND status <> 'answered'
                    """,
                    (answer_comment_id, now, now, row[0]),
                )
                if answered.rowcount != 1:
                    self._db.rollback()
                    return False
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return True

    def release_answer(self, answer_comment_id: int, claim_token: str) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    DELETE FROM answer_captures
                    WHERE answer_comment_id = ?
                      AND status = 'processing'
                      AND claim_token = ?
                    """,
                    (answer_comment_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def update_question_status(
        self, github_comment_id: int, status: str, error_message: str | None = None
    ) -> None:
        """Move a question to another lifecycle state.

        The status is validated here as well as by the database trigger, so a
        caller gets a named error instead of an opaque ``RAISE(ABORT)`` from
        SQLite. The trigger remains the authority: this check is for legibility,
        not for enforcement.
        """
        if status not in QUESTION_STATUSES:
            raise ValueError(f"Unknown question status: {status!r}")
        now = _now()
        with self._lock:
            self._db.execute(
                """
                UPDATE questions
                SET status = ?, error_message = ?, updated_at = ?
                WHERE identifier = ?
                """,
                (status, error_message, now, github_comment_id),
            )
            self._db.commit()

    def is_question_answered(self, github_comment_id: int) -> bool:
        """Whether this question is answered.

        Reads ``status``, which is the single source of truth for the lifecycle.
        The legacy ``answered`` column mirrors it and is kept only for databases
        written by older versions.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM questions WHERE identifier = ?", (github_comment_id,)
            ).fetchone()
        return bool(row and row[0] == "answered")

    def mark_question_answered(self, github_comment_id: int, answer_comment_id: int) -> None:
        now = _now()
        with self._lock:
            self._db.execute(
                """
                UPDATE questions
                SET answered = 1, answer_comment_id = ?, status = 'answered',
                    answered_at = ?, claim_token = NULL, claimed_at = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE identifier = ?
                """,
                (answer_comment_id, now, now, github_comment_id),
            )
            self._db.commit()

    def answer_comment_seen(self, answer_comment_id: int) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM questions WHERE answer_comment_id = ?", (answer_comment_id,)
            ).fetchone()
        return row is not None

    def pr_state_change_seen(
        self, repo: str, pr_number: int, action: str, event_id: str | None = None
    ) -> bool:
        with self._lock:
            if event_id is None:
                row = self._db.execute(
                    """
                    SELECT 1 FROM pr_state_changes
                    WHERE repo = ? AND pr_number = ? AND action = ?
                    LIMIT 1
                    """,
                    (repo, pr_number, action),
                ).fetchone()
            else:
                row = self._db.execute(
                    "SELECT 1 FROM pr_state_changes WHERE event_id = ?", (event_id,)
                ).fetchone()
        return row is not None

    def mark_pr_state_change(
        self, repo: str, pr_number: int, action: str, event_id: str | None = None
    ) -> None:
        key = event_id or f"legacy:{repo}:{pr_number}:{action}"
        now = _now()
        with self._lock:
            if event_id is None:
                existing = self._db.execute(
                    """
                    SELECT 1 FROM pr_state_changes
                    WHERE repo = ? AND pr_number = ? AND action = ?
                    LIMIT 1
                    """,
                    (repo, pr_number, action),
                ).fetchone()
                if existing is not None:
                    return
            self._db.execute(
                """
                INSERT OR IGNORE INTO pr_state_changes (
                    event_id, repo, pr_number, action, created_at, status,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, 'completed', ?)
                """,
                (key, repo, pr_number, action, now, now),
            )
            self._db.commit()

    def claim_pr_state_change(
        self, repo: str, pr_number: int, action: str, event_id: str | None = None
    ) -> str | None:
        key = event_id or f"legacy:{repo}:{pr_number}:{action}"
        now = datetime.now(UTC)
        lease = now + PR_EVENT_LEASE
        claim_token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """
                    SELECT repo, pr_number, action, status, lease_expires_at
                    FROM pr_state_changes WHERE event_id = ?
                    """,
                    (key,),
                ).fetchone()
                if row is not None:
                    if row[0] != repo or row[1] != pr_number or row[2] != action:
                        self._db.rollback()
                        return None
                    if row[3] == "completed":
                        self._db.rollback()
                        return None
                    if row[3] == "processing" and row[4] and row[4] > now.isoformat():
                        self._db.rollback()
                        return None
                    self._db.execute(
                        """
                        UPDATE pr_state_changes
                        SET status = 'processing', claimed_at = ?, lease_expires_at = ?,
                            last_error = NULL, completed_at = NULL, claim_token = ?
                        WHERE event_id = ?
                        """,
                        (now.isoformat(), lease.isoformat(), claim_token, key),
                    )
                else:
                    self._db.execute(
                        """
                        INSERT INTO pr_state_changes (
                            event_id, repo, pr_number, action, created_at, status,
                            claimed_at, lease_expires_at, last_error, completed_at, claim_token
                        ) VALUES (?, ?, ?, ?, ?, 'processing', ?, ?, NULL, NULL, ?)
                        """,
                        (
                            key,
                            repo,
                            pr_number,
                            action,
                            now.isoformat(),
                            now.isoformat(),
                            lease.isoformat(),
                            claim_token,
                        ),
                    )
                self._db.commit()
            except sqlite3.IntegrityError:
                self._db.rollback()
                return None
            except Exception:
                self._db.rollback()
                raise
        return claim_token

    def complete_pr_state_change(self, event_id: str, claim_token: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE pr_state_changes
                    SET status = 'completed', completed_at = ?, lease_expires_at = NULL,
                        last_error = NULL
                    WHERE event_id = ? AND status = 'processing' AND claim_token = ?
                    """,
                    (now, event_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def release_pr_state_change(self, event_id: str, claim_token: str, error: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE pr_state_changes
                    SET status = 'retryable', lease_expires_at = ?, last_error = ?
                    WHERE event_id = ? AND status = 'processing' AND claim_token = ?
                    """,
                    (now, error[:1000], event_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    @staticmethod
    def _backfill_repo_key(repo: str) -> str:
        """Receipts key on identity, not spelling: two casings of one repository
        cannot hold two receipts that disagree about whether it was finished."""
        return repo.casefold()

    @staticmethod
    def _backfill_associations_key(
        authorized_associations: frozenset[str] | None,
    ) -> str | None:
        if authorized_associations is None:
            return None
        return ",".join(sorted(authorized_associations))

    def get_backfill_receipt(self, *, repo: str, pr_number: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT observed_updated_at, window_since, window_until, associations,
                       finished, recorded_at
                FROM backfill_receipts WHERE repo = ? AND pr_number = ?
                """,
                (self._backfill_repo_key(repo), int(pr_number)),
            ).fetchone()
        if row is None:
            return None
        return {
            "observed_updated_at": row[0],
            "window_since": row[1],
            "window_until": row[2],
            "associations": row[3],
            "finished": bool(row[4]),
            "recorded_at": row[5],
        }

    def record_backfill_receipt(
        self,
        *,
        repo: str,
        pr_number: int,
        observed_updated_at: datetime,
        window_since: datetime,
        window_until: datetime | None,
        authorized_associations: frozenset[str] | None,
        finished: bool,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO backfill_receipts
                    (repo, pr_number, observed_updated_at, window_since, window_until,
                     associations, finished, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (repo, pr_number) DO UPDATE SET
                    observed_updated_at = excluded.observed_updated_at,
                    window_since = excluded.window_since,
                    window_until = excluded.window_until,
                    associations = excluded.associations,
                    finished = excluded.finished,
                    recorded_at = excluded.recorded_at
                """,
                (
                    self._backfill_repo_key(repo),
                    int(pr_number),
                    observed_updated_at.isoformat(),
                    window_since.isoformat(),
                    window_until.isoformat() if window_until is not None else None,
                    self._backfill_associations_key(authorized_associations),
                    1 if finished else 0,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._db.commit()

    def claim_delivery(
        self, delivery_id: str, repo: str, event: str, payload_hash: str
    ) -> DeliveryClaim:
        now = datetime.now(UTC)
        lease = now + DELIVERY_LEASE
        claim_token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """
                    SELECT repo, event, payload_hash, status, lease_expires_at
                    FROM webhook_deliveries WHERE delivery_id = ?
                    """,
                    (delivery_id,),
                ).fetchone()
                if row is not None:
                    if row[0] != repo or row[1] != event:
                        self._db.rollback()
                        return "conflict"
                    if row[2] not in {"", payload_hash}:
                        self._db.rollback()
                        return "conflict"
                    if row[3] == "completed":
                        self._db.rollback()
                        return "duplicate"
                    if row[3] == "processing" and row[4] and row[4] > now.isoformat():
                        self._db.rollback()
                        return "active"
                    self._db.execute(
                        """
                        UPDATE webhook_deliveries
                        SET status = 'processing', claimed_at = ?, lease_expires_at = ?,
                            last_error = NULL, completed_at = NULL, claim_token = ?
                        WHERE delivery_id = ?
                        """,
                        (now.isoformat(), lease.isoformat(), claim_token, delivery_id),
                    )
                else:
                    self._db.execute(
                        """
                        INSERT INTO webhook_deliveries (
                            delivery_id, repo, event, payload_hash, status, claimed_at,
                            lease_expires_at, last_error, completed_at, claim_token, created_at
                        ) VALUES (?, ?, ?, ?, 'processing', ?, ?, NULL, NULL, ?, ?)
                        """,
                        (
                            delivery_id,
                            repo,
                            event,
                            payload_hash,
                            now.isoformat(),
                            lease.isoformat(),
                            claim_token,
                            now.isoformat(),
                        ),
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return claim_token

    def complete_delivery(self, delivery_id: str, claim_token: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE webhook_deliveries
                    SET status = 'completed', completed_at = ?, lease_expires_at = NULL,
                        last_error = NULL
                    WHERE delivery_id = ?
                      AND status = 'processing'
                      AND claim_token = ?
                    """,
                    (now, delivery_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def release_delivery(self, delivery_id: str, claim_token: str, error: str) -> bool:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute(
                    """
                    UPDATE webhook_deliveries
                    SET status = 'retryable', lease_expires_at = ?, last_error = ?
                    WHERE delivery_id = ?
                      AND status = 'processing'
                      AND claim_token = ?
                    """,
                    (now, error[:1000], delivery_id, claim_token),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            else:
                return cursor.rowcount == 1

    def record_session(self, session: ReviewSession) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, context_url, context_id, scope, metadata, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.context_url,
                    session.context_id,
                    session.scope,
                    json.dumps(session.metadata),
                    session.created_at.isoformat(),
                    session.created_by,
                ),
            )
            self._db.commit()


def build_registry(settings: Settings) -> QuestionRegistry:
    """Build the local question registry from configuration."""
    return SqliteQuestionRegistry(settings.kojutsu_registry_path)
