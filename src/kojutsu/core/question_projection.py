"""Project terminal decision requests into the knowledge store.

The question registry holds the richest dataset in this project and no other
process can read it. Every decision request ever raised is there with its
lifecycle, and ``list_questions`` already exists as the enumeration — so this
module is a read of that data, not a new write path into the registry, and
nothing here can strand a claim or block the capture path.

Three decisions shape it, and each is a position rather than an implementation
detail.

**Terminal states only.** A ``pending`` or ``claimed`` question is outstanding
work: operational, transient, and re-written every time a worker claims or
releases it. Projecting it would put a live queue into the knowledge store,
where it reads as a set of open decisions rather than as requests this process
is still waiting on, and it would put a rewrite-per-lease onto a delivery path
built for immutable records. What survives is the interesting part: a request
that was answered, one that hit the attempt ceiling, and one the code overtook
before a human replied. ``created_at`` and ``answered_at`` are both stored, so
the wait is computable without ever persisting a transient state.

**The model is the allowlist.** :class:`~kojutsu.models.QuestionRecord` has no
``claim_token`` field and no ``last_error`` field, so this module cannot publish
either — not because it filters them out, but because there is nowhere for them
to be read from. A deny-list would make the safe behaviour depend on remembering
to update a blocklist, and its failure is silent: a document merely lacks a key
and nothing reports it. ``claim_token`` is the more important of the two: it is a
capability, and Tanseki is readable over MCP by agents.

**Nothing is deleted and nothing is re-derived.** A question that is still
outstanding is simply absent from the store, and the registry remains the
authority on whether it exists. That is the opposite failure to the census, and
it is the right one here — see ``docs/decisions/001-registry-through-tanseki.md`` in
the consuming project for why the same seam carries the other direction.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from kojutsu.models import QuestionRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome, KnowledgeSink


class _QuestionSource(Protocol):
    """The one registry method this projection needs.

    Narrower than :class:`~kojutsu.core.question_registry.QuestionRegistry` on
    purpose. A projection that takes the whole registry claims to need the claim,
    the lease and the context manager, when it only ever reads a list — and
    asserting that against a double is how a read ends up coupled to a write path
    it has no business touching.
    """

    def list_questions(
        self,
        *,
        status: str | None = ...,
        repo: str | None = ...,
        pr_number: int | None = ...,
        limit: int = ...,
    ) -> list[dict[str, Any]]: ...


#: The statuses a record is written for. Mirrors the model's own closed
#: vocabulary, and is asserted to agree with it rather than imported across the
#: module boundary that would pull SQLite into a pure model.
TERMINAL_STATUSES: tuple[str, ...] = ("answered", "failed", "superseded")

#: Rows read per call. The registry's own page limit is enforced downstream, so
#: this is a bound on work per projection rather than on the registry.
DEFAULT_PROJECTION_LIMIT = 200

_REGISTRY_ROW_KEYS = (
    "question_id",
    "repo",
    "pr_number",
    "pr_url",
    "question_text",
    "category",
    "jira_ticket_key",
    "session_id",
    "question_author",
    "assignee",
    "attempts",
    "answer_comment_id",
    "created_at",
    "answered_at",
    "updated_at",
)


def _as_datetime(value: object) -> datetime | None:
    """Read a registry timestamp, which arrives as ISO-8601 text.

    The registry is a schema migration away from storing these natively, so the
    string is the contract today. A value that will not parse becomes ``None``
    rather than raising: a question with an unreadable timestamp is still a
    question, and losing it to a format change would be worse than recording it
    without that one field.
    """
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def to_question_record(row: Mapping[str, Any]) -> QuestionRecord:
    """Build a record from a registry row, reading only the projected keys.

    The projection names the columns explicitly rather than splatting the row, so
    a column added to ``_QUESTION_LIST_COLUMNS`` later is not published until it
    is added here. The model is the second half of that guarantee: it has no
    field to put an unlisted column into.
    """
    return QuestionRecord(
        question_id=str(row.get("question_id") or ""),
        repo=str(row.get("repo") or ""),
        pr_number=row.get("pr_number"),
        pr_url=row.get("pr_url"),
        question_text=str(row.get("question_text") or ""),
        status=str(row.get("status") or ""),
        category=row.get("category"),
        jira_ticket_key=row.get("jira_ticket_key"),
        session_id=row.get("session_id"),
        question_author=row.get("question_author"),
        assignee=row.get("assignee"),
        attempts=int(row.get("attempts") or 0),
        answer_comment_id=row.get("answer_comment_id"),
        created_at=_as_datetime(row.get("created_at")),
        answered_at=_as_datetime(row.get("answered_at")),
        updated_at=_as_datetime(row.get("updated_at")),
    )


def projectable_rows(registry: _QuestionSource) -> list[dict[str, Any]]:
    """Every terminal question in the registry, one status at a time.

    Read per status rather than with a single ``IN`` clause because
    ``list_questions`` is the registry's published enumeration and the status
    filter is part of its contract; going around it would be a private query
    against a schema this module does not own.
    """
    rows: list[dict[str, Any]] = []
    for status in TERMINAL_STATUSES:
        rows.extend(registry.list_questions(status=status, limit=DEFAULT_PROJECTION_LIMIT))
    return [row for row in rows if row.get("question_id")]


def project_terminal_questions(
    registry: _QuestionSource,
    sink: KnowledgeSink,
    *,
    limit: int = DEFAULT_PROJECTION_LIMIT,
) -> list[KnowledgeDeliveryOutcome]:
    """Write every terminal question to the store, and return what happened.

    Idempotent by document id: the id is derived from ``(repo, pr, question_id)``
    and never from the status, so re-projecting an unchanged question is an
    upsert of the same document rather than a second record. That is what makes
    this safe to run on a schedule and safe to re-run after a failure, which in
    turn is why it does not need the registry to tell it what changed.

    A row that will not build a record is skipped rather than raising. One
    malformed row must not stop the rest of a queue from being recorded, and a
    question that cannot be projected is a question whose *projection* is wrong,
    not a question that should be dropped from the registry — the registry is
    unaffected either way, because this module never writes to it.
    """
    outcomes: list[KnowledgeDeliveryOutcome] = []
    for row in projectable_rows(registry)[: max(0, limit)]:
        try:
            record = to_question_record(row)
        except (ValueError, TypeError):
            continue
        outcome = sink.store(record)
        if outcome is not None:
            outcomes.append(outcome)
    return outcomes


__all__ = [
    "DEFAULT_PROJECTION_LIMIT",
    "TERMINAL_STATUSES",
    "project_terminal_questions",
    "projectable_rows",
    "to_question_record",
]
