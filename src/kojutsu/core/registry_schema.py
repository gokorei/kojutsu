"""SQLite schema, migrations and validation for the question registry.

Split out of ``question_registry``: version, status vocabulary, table
definitions, the forward-only ledgered migrations and the validators.
The store class owns connections and queries; this module owns what the
database must look like.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
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
CREATE INDEX IF NOT EXISTS questions_identifier_idx ON questions (identifier);

CREATE TABLE IF NOT EXISTS pr_state_changes (
    event_id       TEXT PRIMARY KEY,
    repo           TEXT NOT NULL,
    pr_number      INTEGER NOT NULL,
    action         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'completed',
    claimed_at     TEXT,
    lease_expires_at TEXT,
    last_error     TEXT,
    completed_at   TEXT,
    claim_token    TEXT
);
CREATE INDEX IF NOT EXISTS pr_state_changes_lookup_idx
    ON pr_state_changes (repo, pr_number, action);

CREATE TABLE IF NOT EXISTS review_captures (
    review_event_id   TEXT PRIMARY KEY,
    repo              TEXT NOT NULL,
    pr_number         INTEGER NOT NULL,
    review_id         INTEGER NOT NULL,
    comment_id        INTEGER,
    kind              TEXT NOT NULL,
    author            TEXT,
    status            TEXT NOT NULL DEFAULT 'completed',
    claimed_at        TEXT,
    lease_expires_at  TEXT,
    last_error        TEXT,
    completed_at      TEXT,
    claim_token       TEXT
);
CREATE INDEX IF NOT EXISTS review_captures_lookup_idx
    ON review_captures (repo, pr_number, review_id);
CREATE UNIQUE INDEX IF NOT EXISTS review_captures_comment_unique_idx
    ON review_captures (comment_id) WHERE comment_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS answer_captures (
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
CREATE INDEX IF NOT EXISTS answer_captures_question_idx
    ON answer_captures (parent_comment_id, status);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    delivery_id      TEXT PRIMARY KEY,
    repo             TEXT NOT NULL,
    event            TEXT NOT NULL,
    payload_hash     TEXT NOT NULL,
    status           TEXT NOT NULL,
    claimed_at       TEXT,
    lease_expires_at TEXT,
    last_error       TEXT,
    completed_at     TEXT,
    claim_token      TEXT,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    context_url TEXT,
    context_id  TEXT,
    scope       TEXT,
    metadata    TEXT,
    created_at  TEXT,
    created_by  TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA_VERSION = 7
DELIVERY_LEASE = timedelta(minutes=5)
ANSWER_LEASE = timedelta(minutes=5)
PR_EVENT_LEASE = timedelta(minutes=5)
QUESTION_LEASE = timedelta(minutes=5)
RATIONALE_LEASE = timedelta(minutes=5)
DELIVERY_RETENTION_DAYS = 30

#: The closed vocabulary for ``questions.status``.
#:
#: ``pending``   outstanding, claimable by a worker.
#: ``claimed``   held by a live lease; reclaimable only once that lease expires.
#: ``answered``  terminal success. This is the single source of truth for the
#:               answer-dedupe gate -- see :meth:`SqliteQuestionRegistry.
#:               is_question_answered`.
#: ``failed``    terminal failure, reached when ``attempts`` reaches the ceiling.
#: ``superseded``terminal, replaced by a newer question set for the same PR.
#:
#: Terminal states are ``answered``, ``failed`` and ``superseded``. Nothing may
#: move a question out of one, so an enumeration of ``pending`` is a complete
#: description of outstanding work.
QUESTION_STATUSES: tuple[str, ...] = (
    "pending",
    "claimed",
    "answered",
    "failed",
    "superseded",
)
TERMINAL_QUESTION_STATUSES: frozenset[str] = frozenset({"answered", "failed", "superseded"})

#: How many times a question may be claimed before it is dead-lettered. Bounded
#: so a permanently unanswerable question cannot be retried forever.
DEFAULT_MAX_QUESTION_ATTEMPTS = 8

#: Upper bound on a single ``list_questions`` page, so an unattended worker
#: cannot ask the registry for an unbounded amount of work.
MAX_QUESTION_LIST_LIMIT = 1_000

#: Upper bound on a stored failure reason, matching the outbox's bound on
#: ``last_error`` so a hostile upstream message cannot bloat a row.
MAX_QUESTION_ERROR_LENGTH = 1_000


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in db.execute(f"PRAGMA table_info({table})")}


def _table_exists(db: sqlite3.Connection, table: str) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


#: The projection :meth:`SqliteQuestionRegistry.list_questions` selects, in column
#: order. It sits beside the mapper that consumes it so the two cannot drift.
#:
#: ``claim_token`` is deliberately absent. A claim token is the capability to
#: release or finalize somebody else's claim, and an enumeration surface is read
#: by far more callers -- including the CLI -- than the single worker that already
#: holds the token. Whether a claim is held is reported by ``status``,
#: ``assignee`` and ``lease_expires_at``; the token itself stays in the claim
#: response.
_QUESTION_LIST_COLUMNS = (
    "question_id",
    "identifier",
    "repo",
    "pr_number",
    "pr_url",
    "question_text",
    "category",
    "jira_ticket_key",
    "session_id",
    "status",
    "assignee",
    "attempts",
    "last_attempt_at",
    "claimed_at",
    "lease_expires_at",
    "last_error",
    "answer_comment_id",
    "answered_at",
    "created_at",
    "updated_at",
)


def _question_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(_QUESTION_LIST_COLUMNS, row, strict=True))


def _bounded_question_limit(limit: int) -> int:
    """Clamp a requested page size, rejecting anything that is not an integer.

    A bool is rejected explicitly because it is an ``int`` subclass, and
    ``True`` silently meaning "one row" is the kind of coercion that turns a
    paging bug into a truncated result rather than an error.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a non-negative integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return min(limit, MAX_QUESTION_LIST_LIMIT)


def _create_pr_state_changes(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE pr_state_changes (
            event_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            claimed_at TEXT,
            lease_expires_at TEXT,
            last_error TEXT,
            completed_at TEXT,
            claim_token TEXT
        )
        """
    )


def _insert_legacy_pr_state_changes(db: sqlite3.Connection, table: str) -> None:
    db.execute(
        f"""
        INSERT OR IGNORE INTO pr_state_changes (event_id, repo, pr_number, action, created_at)
        SELECT 'legacy:' || repo || ':' || pr_number || ':' || action,
               repo, pr_number, action, created_at
        FROM {table}
        """
    )


def _validate_pr_state_changes(db: sqlite3.Connection) -> None:
    required = {
        "event_id",
        "repo",
        "pr_number",
        "action",
        "created_at",
        "status",
        "claimed_at",
        "lease_expires_at",
        "last_error",
        "completed_at",
        "claim_token",
    }
    if not required.issubset(_table_columns(db, "pr_state_changes")):
        raise RuntimeError("PR state migration produced an invalid schema")
    if db.execute("SELECT COUNT(*) FROM pr_state_changes WHERE event_id IS NULL").fetchone()[0]:
        raise RuntimeError("PR state migration produced rows without event identities")


def _migrate_pr_state_changes(db: sqlite3.Connection) -> None:
    current_columns = _table_columns(db, "pr_state_changes")
    legacy_exists = _table_exists(db, "pr_state_changes_legacy")
    db.execute("DROP INDEX IF EXISTS pr_state_changes_lookup_idx")
    db.execute("DROP INDEX IF EXISTS pr_state_changes_status_idx")
    if legacy_exists:
        if not current_columns:
            _create_pr_state_changes(db)
        elif "event_id" not in current_columns:
            db.execute("ALTER TABLE pr_state_changes RENAME TO pr_state_changes_partial")
            _create_pr_state_changes(db)
            partial_columns = _table_columns(db, "pr_state_changes_partial")
            if "event_id" in partial_columns:
                db.execute(
                    """
                    INSERT OR IGNORE INTO pr_state_changes (event_id, repo, pr_number, action, created_at)
                    SELECT event_id, repo, pr_number, action, created_at
                    FROM pr_state_changes_partial
                    """
                )
            else:
                _insert_legacy_pr_state_changes(db, "pr_state_changes_partial")
            db.execute("DROP TABLE pr_state_changes_partial")
        _insert_legacy_pr_state_changes(db, "pr_state_changes_legacy")
        missing = db.execute(
            """
            SELECT COUNT(*)
            FROM pr_state_changes_legacy AS legacy
            LEFT JOIN pr_state_changes AS current
              ON current.event_id = 'legacy:' || legacy.repo || ':' || legacy.pr_number || ':' || legacy.action
            WHERE current.event_id IS NULL
            """
        ).fetchone()[0]
        if missing:
            raise RuntimeError("PR state migration lost legacy rows")
        _validate_pr_state_changes(db)
        db.execute("DROP TABLE pr_state_changes_legacy")
        return
    if not current_columns:
        _create_pr_state_changes(db)
    elif "event_id" not in current_columns:
        legacy_count = db.execute("SELECT COUNT(*) FROM pr_state_changes").fetchone()[0]
        db.execute("ALTER TABLE pr_state_changes RENAME TO pr_state_changes_legacy")
        _create_pr_state_changes(db)
        _insert_legacy_pr_state_changes(db, "pr_state_changes_legacy")
        migrated_count = db.execute("SELECT COUNT(*) FROM pr_state_changes").fetchone()[0]
        if migrated_count != legacy_count:
            raise RuntimeError("PR state migration changed the legacy row count")
        _validate_pr_state_changes(db)
        db.execute("DROP TABLE pr_state_changes_legacy")


def _ensure_columns(db: sqlite3.Connection, table: str, additions: dict[str, str]) -> None:
    columns = _table_columns(db, table)
    if not columns:
        raise RuntimeError(f"Required table {table} is missing")
    for column, definition in additions.items():
        if column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


_STATUS_LITERALS_SQL = ", ".join(f"'{status}'" for status in QUESTION_STATUSES)

#: The status vocabulary is enforced by triggers rather than a table-level CHECK.
#:
#: SQLite cannot add a CHECK constraint to an existing table, and the only way to
#: get one is to rebuild the table -- which would rewrite every stored question.
#: A ``BEFORE INSERT``/``BEFORE UPDATE OF status`` trigger enforces exactly the
#: same rule while leaving all existing rows untouched, which is the property
#: this migration is required to preserve. Rows that already violate the
#: vocabulary are repaired by the migration itself and re-checked by
#: :func:`_validate_schema`.
_STATUS_VOCABULARY_TRIGGER_SQL = (
    "CREATE TRIGGER IF NOT EXISTS questions_status_vocabulary_insert\n"
    "BEFORE INSERT ON questions\n"
    f"WHEN NEW.status NOT IN ({_STATUS_LITERALS_SQL})\n"
    "BEGIN\n"
    "  SELECT RAISE(ABORT, 'questions.status is outside the documented vocabulary');\n"
    "END"
)
_STATUS_VOCABULARY_UPDATE_TRIGGER_SQL = (
    "CREATE TRIGGER IF NOT EXISTS questions_status_vocabulary_update\n"
    "BEFORE UPDATE OF status ON questions\n"
    f"WHEN NEW.status NOT IN ({_STATUS_LITERALS_SQL})\n"
    "BEGIN\n"
    "  SELECT RAISE(ABORT, 'questions.status is outside the documented vocabulary');\n"
    "END"
)

#: Schema v4: turn ``questions`` into a findable work queue.
#:
#: This tuple is the migration. It is both what gets executed and what gets
#: checksummed, so the recorded history cannot drift away from the statements the
#: application would actually run. Every statement must be idempotent, because a
#: registry may already be part-way through when the process is interrupted.
#:
#: Nothing here deletes or rebuilds ``questions``. Columns are added, indexes and
#: triggers are created, and two targeted ``UPDATE`` statements repair rows whose
#: ``status`` and legacy ``answered`` flag disagree. That repair is information-
#: preserving: it only ever promotes a row to the state its own ``answered`` flag
#: already asserts, so no recorded answer can be lost or duplicated.
_MIGRATION_V4: tuple[str, ...] = (
    "ALTER TABLE questions ADD COLUMN assignee TEXT",
    "ALTER TABLE questions ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE questions ADD COLUMN last_attempt_at TEXT",
    "ALTER TABLE questions ADD COLUMN answered_at TEXT",
    "ALTER TABLE questions ADD COLUMN updated_at TEXT",
    "ALTER TABLE questions ADD COLUMN claimed_at TEXT",
    "ALTER TABLE questions ADD COLUMN lease_expires_at TEXT",
    "ALTER TABLE questions ADD COLUMN last_error TEXT",
    "ALTER TABLE questions ADD COLUMN claim_token TEXT",
    "CREATE INDEX IF NOT EXISTS questions_work_idx "
    "ON questions (status, repo, pr_number, question_id)",
    "UPDATE questions SET status = 'answered' WHERE answered = 1 AND status <> 'answered'",
    "UPDATE questions SET answered_at = created_at WHERE answered = 1 AND answered_at IS NULL",
    "UPDATE questions SET updated_at = created_at WHERE updated_at IS NULL",
    _STATUS_VOCABULARY_TRIGGER_SQL,
    _STATUS_VOCABULARY_UPDATE_TRIGGER_SQL,
)

#: The data repairs in :data:`_MIGRATION_V4`, in normalised form.
#:
#: Each one is written so that re-running it matches no row, which is what makes a
#: partially applied migration safe to resume. Listing them explicitly means a
#: future repair has to be acknowledged as a repair instead of slipping through a
#: general-purpose statement guard.
_REPAIR_STATEMENTS: frozenset[str] = frozenset(
    " ".join(statement.split())
    for statement in _MIGRATION_V4
    if statement.upper().startswith("UPDATE ")
)

#: Schema v5: give review evidence its own capture table.
#:
#: Review verdicts and inline review comments are not PR lifecycle transitions, so
#: they do not belong in ``pr_state_changes``. Keeping them apart is what stops a
#: tool's own output being later quoted as a reviewer's decision: the two kinds of
#: record are never interleaved in one table where a reader could confuse them.
#:
#: The claim columns mirror the other three claim tables, and ``comment_id`` is
#: uniquely indexed so one inline comment can never produce two records even if two
#: deliveries of it race.
_MIGRATION_V5: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS review_captures ("
    "review_event_id TEXT PRIMARY KEY, "
    "repo TEXT NOT NULL, "
    "pr_number INTEGER NOT NULL, "
    "review_id INTEGER NOT NULL, "
    "comment_id INTEGER, "
    "kind TEXT NOT NULL, "
    "author TEXT, "
    "status TEXT NOT NULL DEFAULT 'completed', "
    "claimed_at TEXT, "
    "lease_expires_at TEXT, "
    "last_error TEXT, "
    "completed_at TEXT, "
    "claim_token TEXT)",
    "CREATE INDEX IF NOT EXISTS review_captures_lookup_idx "
    "ON review_captures (repo, pr_number, review_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS review_captures_comment_unique_idx "
    "ON review_captures (comment_id) WHERE comment_id IS NOT NULL",
)

#: Schema v6: give declared decision rationale its own capture table.
#:
#: A rationale is not an answer to a registered question and not a PR lifecycle
#: transition, so it belongs in neither ``questions`` nor ``pr_state_changes``.
#: Keeping it apart is the same reasoning as v5: a reader who can see what an
#: agent said it *decided* and what a reviewer *concluded* in one interleaved
#: table can confuse the two, and the confusion is exactly the one this
#: programme exists to prevent. The two kinds of record are never neighbours.
#:
#: ``entry_id`` is derived from the semantic anchor -- repo, pr number or branch,
#: declaring principal, revision -- and not from the delivery, so a re-delivery
#: collapses to one record the way :func:`semantic_review_event_id` collapses a
#: re-delivered review. It is the primary key, which is what makes the second
#: racing claim fail rather than produce a duplicate.
#:
#: The claim columns mirror the other three claim tables so the pattern stays
#: universal, and ``revision`` makes an appended declaration a revision rather
#: than a replacement: intent captured early goes stale as the work proceeds, and
#: a later and worse rationale must not be able to destroy an earlier one.
_MIGRATION_V6: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS rationale_captures ("
    "entry_id TEXT PRIMARY KEY, "
    "repo TEXT NOT NULL, "
    "pr_number INTEGER, "
    "branch TEXT, "
    "declared_by TEXT NOT NULL, "
    "declared_model TEXT, "
    "source TEXT NOT NULL DEFAULT 'declared', "
    "revision INTEGER NOT NULL DEFAULT 1, "
    "revises TEXT, "
    "rationale_text TEXT NOT NULL, "
    "status TEXT NOT NULL DEFAULT 'completed', "
    "claimed_at TEXT, "
    "lease_expires_at TEXT, "
    "last_error TEXT, "
    "completed_at TEXT, "
    "claim_token TEXT)",
    "CREATE INDEX IF NOT EXISTS rationale_captures_lookup_idx "
    "ON rationale_captures (repo, pr_number, revision)",
    "CREATE UNIQUE INDEX IF NOT EXISTS rationale_captures_revision_unique_idx "
    "ON rationale_captures (repo, COALESCE(pr_number, -1), branch, declared_by, revision)",
)

#: Schema v7: remember the commit a question was asked about.
#:
#: A question is a claim about a specific diff. Without the commit it was aimed at,
#: the row records only that somebody asked something, and a reader holding it after
#: the branch has moved cannot tell a still-valid question from one overtaken by
#: thirty commits. The registry already knows to *notice* that movement --
#: :meth:`SqliteQuestionRegistry.supersede_questions_for_pr` exists for it -- but
#: until now it could only write the superseding commit into a message meant for a
#: human. Noticing the head moved and being able to say what the question was about
#: are two halves of one capability, and only the first was here.
#:
#: **No default and no backfill**, which is the whole content of this migration. An
#: ``ADD COLUMN`` with a default would invent a value for every question already
#: stored; deriving one from the pull request's current head would invent a
#: different, far worse value. A question asked three weeks ago was asked about
#: whatever the head was *then*, and today's head is not a fact about it. Those rows
#: read ``NULL``, which is the truth: the question was asked and nobody recorded what
#: against. A wrong anchor is worse than a missing one, because a wrong one is
#: believed -- a reader who finds a sha in this column has no way to tell a recorded
#: fact from a plausible reconstruction, and will treat a guess as an anchor. An
#: absent value at least advertises that the store does not know.
#:
#: The same reasoning governs what an answer carries. An answer arrives in a later
#: ``issue_comment`` delivery, long after the ask, so the head on that delivery is
#: generally not the head the question was about -- and that gap is exactly what
#: ``supersede_questions_for_pr`` exists to notice. Storing the arrival-time head
#: would therefore be wrong in precisely the case the system is built to flag, which
#: is why this column is the ask-time head and why it has to exist before an answer
#: can be anchored at all.
#:
#: Nothing here rebuilds or rewrites ``questions``. One column is added, and the
#: existing ``ALTER TABLE ... ADD COLUMN`` handling makes a re-run a no-op.
_MIGRATION_V7: tuple[str, ...] = ("ALTER TABLE questions ADD COLUMN head_sha TEXT",)

#: Migrations recorded in the ``registry_migrations`` ledger, and therefore
#: checksum-verified on every open. Extend this as versions are added.
_LEDGERED_MIGRATIONS: dict[int, tuple[str, ...]] = {
    4: _MIGRATION_V4,
    5: _MIGRATION_V5,
    6: _MIGRATION_V6,
    7: _MIGRATION_V7,
}


def _migration_checksum(statements: tuple[str, ...]) -> str:
    """Checksum a migration's statements, ignoring incidental whitespace."""
    canonical = "\n".join(" ".join(statement.split()) for statement in statements)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _apply_statement(db: sqlite3.Connection, statement: str) -> None:
    """Apply one migration statement, skipping it when the schema already has it.

    Re-running a partially applied migration has to be safe, so a statement that
    would otherwise fail on the second attempt is detected and skipped. Schema
    statements are guarded mechanically; data repairs are guarded by the explicit
    :data:`_REPAIR_STATEMENTS` allowlist, because "re-running this is a no-op" is
    a property of the statement's own WHERE clause and cannot be inferred from
    its shape. Anything unrecognised is refused rather than guessed at.
    """
    normalised = " ".join(statement.split())
    head = normalised.upper()
    if head.startswith("ALTER TABLE"):
        target, separator, addition = normalised.partition(" ADD COLUMN ")
        if not separator:
            raise RuntimeError(f"Migration statement is not a column addition: {normalised}")
        table = target[len("ALTER TABLE ") :].strip()
        if addition.split()[0] in _table_columns(db, table):
            return
    elif head.startswith("CREATE "):
        if "IF NOT EXISTS" not in head:
            raise RuntimeError(f"Migration statement is not idempotent: {normalised}")
    elif normalised in _REPAIR_STATEMENTS:
        pass
    else:
        raise RuntimeError(f"Migration statement is not a supported form: {normalised}")
    db.execute(statement)


def _ensure_ledger_table(db: sqlite3.Connection) -> None:
    """Create the migration ledger, which is infrastructure rather than a step.

    The ledger is deliberately not part of any version's checksummed statement
    list: a checksum exists to detect that the *content* of a migration was
    rewritten, and the table that records those checksums is not itself content
    that can be rewritten without being noticed.
    """
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS registry_migrations (
            version INTEGER PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _record_migration(db: sqlite3.Connection, version: int, statements: tuple[str, ...]) -> None:
    """Apply one forward-only migration step and record its checksum.

    A step already present in the ledger is not re-run, but its checksum is still
    compared: a registry whose recorded history disagrees with this application
    has been rewritten behind our back, and continuing from it could silently
    half-apply a migration. That is refused rather than repaired.
    """
    checksum = _migration_checksum(statements)
    _ensure_ledger_table(db)
    row = db.execute(
        "SELECT checksum FROM registry_migrations WHERE version = ?", (version,)
    ).fetchone()
    if row is not None:
        if str(row[0]) != checksum:
            raise RuntimeError(
                f"Registry migration {version} was recorded with checksum {row[0]}, but this "
                f"application implements it as {checksum}. The applied history does not match."
            )
        return
    for statement in statements:
        _apply_statement(db, statement)
    db.execute(
        "INSERT INTO registry_migrations (version, checksum, applied_at) VALUES (?, ?, ?)",
        (version, checksum, _now()),
    )


def _validate_migration_ledger(db: sqlite3.Connection) -> None:
    """Fail closed when the recorded migration history is missing or rewritten."""
    if not _table_exists(db, "registry_migrations"):
        raise RuntimeError("Registry migration ledger is missing")
    recorded = {
        int(row[0]): str(row[1])
        for row in db.execute("SELECT version, checksum FROM registry_migrations")
    }
    for version, statements in _LEDGERED_MIGRATIONS.items():
        expected = _migration_checksum(statements)
        if version not in recorded:
            raise RuntimeError(f"Registry migration {version} is not recorded in the ledger")
        if recorded[version] != expected:
            raise RuntimeError(
                f"Registry migration {version} was recorded with checksum {recorded[version]}, "
                f"but this application implements it as {expected}. The applied history does not "
                "match."
            )
    for version in recorded:
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"Registry migration {version} is newer than this application")


def _validate_question_state(db: sqlite3.Connection) -> None:
    """Assert the ``status``/``answered`` invariant the dedupe gate depends on.

    ``status`` is the single source of truth for whether a question is answered.
    ``answered`` is retained as a migration-compatible mirror. If the two ever
    disagree, the answer-dedupe gate and the enumeration would tell different
    stories about the same question, so a disagreement is a hard failure rather
    than a value to guess at.
    """
    placeholders = ", ".join(f"'{status}'" for status in QUESTION_STATUSES)
    if db.execute(
        f"SELECT COUNT(*) FROM questions WHERE status NOT IN ({placeholders})"
    ).fetchone()[0]:
        raise RuntimeError("Registry contains a question status outside the documented vocabulary")
    if db.execute(
        "SELECT COUNT(*) FROM questions WHERE answered = 1 AND status <> 'answered'"
    ).fetchone()[0]:
        raise RuntimeError("Registry contains questions whose answered flag and status disagree")
    if db.execute(
        "SELECT COUNT(*) FROM questions WHERE status = 'answered' AND answered = 0"
    ).fetchone()[0]:
        raise RuntimeError("Registry contains answered questions with no mirrored answered flag")
    if db.execute("SELECT COUNT(*) FROM questions WHERE updated_at IS NULL").fetchone()[0]:
        raise RuntimeError("Registry contains questions with no last-touched timestamp")


def _validate_schema(db: sqlite3.Connection) -> None:
    required_columns = {
        "questions": {
            "question_id",
            "identifier",
            "question_author",
            "created_at",
            "updated_at",
            "answered_at",
            "status",
            "assignee",
            "attempts",
            "last_attempt_at",
            "claimed_at",
            "lease_expires_at",
            "last_error",
            "claim_token",
            # Nullable by design and by migration: a question predating v7 has no
            # commit recorded against it, and a column that refused ``NULL`` would
            # force one to be invented.
            "head_sha",
        },
        "registry_migrations": {"version", "checksum", "applied_at"},
        "pr_state_changes": {
            "event_id",
            "repo",
            "pr_number",
            "action",
            "status",
            "claim_token",
            "completed_at",
        },
        "answer_captures": {
            "answer_comment_id",
            "parent_comment_id",
            "question_id",
            "entry_id",
            "status",
            "claimed_at",
            "lease_expires_at",
            "claim_token",
            "completed_at",
        },
        "webhook_deliveries": {
            "delivery_id",
            "repo",
            "event",
            "payload_hash",
            "status",
            "claimed_at",
            "lease_expires_at",
            "claim_token",
            "completed_at",
            "created_at",
        },
        "sessions": {"session_id"},
        "review_captures": {
            "review_event_id",
            "repo",
            "pr_number",
            "review_id",
            "comment_id",
            "kind",
            "author",
            "status",
            "claimed_at",
            "lease_expires_at",
            "last_error",
            "completed_at",
            "claim_token",
        },
        "rationale_captures": {
            "entry_id",
            "repo",
            "pr_number",
            "branch",
            "declared_by",
            "declared_model",
            "source",
            "revision",
            "revises",
            "rationale_text",
            "status",
            "claimed_at",
            "lease_expires_at",
            "last_error",
            "completed_at",
            "claim_token",
        },
    }
    for table, required in required_columns.items():
        if not _table_exists(db, table) or not required.issubset(_table_columns(db, table)):
            raise RuntimeError(f"Registry migration produced an invalid {table} schema")
    _validate_pr_state_changes(db)
    _validate_migration_ledger(db)
    _validate_question_state(db)
    if _table_exists(db, "pr_state_changes_legacy"):
        raise RuntimeError("Legacy PR state table was not removed")
    required_indexes = {
        "answer_captures": {
            "answer_captures_question_unique_idx",
            "answer_captures_question_idx",
        },
        "pr_state_changes": {
            "pr_state_changes_lookup_idx",
            "pr_state_changes_status_idx",
        },
        "webhook_deliveries": {
            "webhook_deliveries_status_idx",
            "webhook_deliveries_completed_idx",
        },
        # Enumeration of outstanding work must not degrade into a full table scan.
        "questions": {"questions_work_idx"},
        "review_captures": {
            "review_captures_lookup_idx",
            "review_captures_comment_unique_idx",
        },
        "rationale_captures": {
            "rationale_captures_lookup_idx",
            "rationale_captures_revision_unique_idx",
        },
    }
    for table, required in required_indexes.items():
        indexes = {str(row[1]) for row in db.execute(f"PRAGMA index_list({table})")}
        if not required.issubset(indexes):
            raise RuntimeError(f"Registry migration produced invalid indexes for {table}")
    if db.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT question_id
            FROM answer_captures
            GROUP BY question_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]:
        raise RuntimeError("Registry migration left duplicate answer captures")


def _migrate_schema_transaction(db: sqlite3.Connection) -> None:
    version_row = db.execute("PRAGMA user_version").fetchone()
    version = int(version_row[0]) if version_row else 0
    if version > SCHEMA_VERSION:
        raise RuntimeError("Registry schema is newer than this application")
    if version == SCHEMA_VERSION:
        _validate_schema(db)
        return
    _ensure_columns(
        db,
        "pr_state_changes",
        {
            "status": "TEXT NOT NULL DEFAULT 'completed'",
            "claimed_at": "TEXT",
            "lease_expires_at": "TEXT",
            "last_error": "TEXT",
            "completed_at": "TEXT",
            "claim_token": "TEXT",
        },
    )
    _migrate_pr_state_changes(db)
    _ensure_columns(db, "questions", {"question_author": "TEXT"})
    _ensure_columns(
        db,
        "answer_captures",
        {
            "status": "TEXT NOT NULL DEFAULT 'completed'",
            "claimed_at": "TEXT",
            "lease_expires_at": "TEXT",
            "last_error": "TEXT",
            "completed_at": "TEXT",
            "claim_token": "TEXT",
        },
    )
    db.execute(
        """
        DELETE FROM answer_captures
        WHERE rowid NOT IN (
            SELECT COALESCE(
                MIN(CASE WHEN status = 'completed' THEN rowid END),
                MIN(rowid)
            )
            FROM answer_captures
            GROUP BY question_id
        )
        """
    )
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS answer_captures_question_unique_idx "
        "ON answer_captures (question_id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS answer_captures_question_idx "
        "ON answer_captures (parent_comment_id, status)"
    )
    _ensure_columns(
        db,
        "webhook_deliveries",
        {
            "payload_hash": "TEXT NOT NULL DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'completed'",
            "claimed_at": "TEXT",
            "lease_expires_at": "TEXT",
            "last_error": "TEXT",
            "completed_at": "TEXT",
            "claim_token": "TEXT",
        },
    )
    db.execute(
        """
        UPDATE webhook_deliveries
        SET status = 'completed', completed_at = COALESCE(completed_at, created_at),
            lease_expires_at = NULL
        WHERE completed_at IS NULL AND status = 'completed'
        """
    )
    db.execute(
        """
        UPDATE pr_state_changes
        SET status = 'completed', completed_at = COALESCE(completed_at, created_at),
            lease_expires_at = NULL, last_error = NULL
        WHERE completed_at IS NULL AND status = 'completed'
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS pr_state_changes_lookup_idx "
        "ON pr_state_changes (repo, pr_number, action)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS pr_state_changes_status_idx "
        "ON pr_state_changes (status, lease_expires_at)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS webhook_deliveries_status_idx "
        "ON webhook_deliveries (status, lease_expires_at)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS webhook_deliveries_completed_idx "
        "ON webhook_deliveries (status, completed_at)"
    )
    _record_migration(db, 4, _MIGRATION_V4)
    _record_migration(db, 5, _MIGRATION_V5)
    _record_migration(db, 6, _MIGRATION_V6)
    _record_migration(db, 7, _MIGRATION_V7)
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    _validate_schema(db)


def _migrate_schema(db: sqlite3.Connection) -> None:
    try:
        db.execute("BEGIN IMMEDIATE")
        _migrate_schema_transaction(db)
        db.commit()
    except Exception:
        db.rollback()
        raise
