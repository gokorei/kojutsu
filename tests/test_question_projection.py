"""The registry projected into the knowledge store as decision requests.

The properties here are the ones the projection's design rests on, and each of
them is a way the change could have been made wrong in a way nothing would
report:

- a claim token or an internal error reaching the store, which is a capability
  and a leak respectively
- an outstanding question being projected, which puts a live work queue into a
  knowledge store
- a question reading as evidence, when nobody stated anything
- a status with no age attached, so a reader assumes a stale status is current
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryOutcome,
    KnowledgeDeliveryStatus,
    to_payload,
)
from kojutsu.core.question_projection import (
    project_terminal_questions,
    to_question_record,
)
from kojutsu.core.tanseki_mapping import (
    build_question_frontmatter,
    question_document_id,
)
from kojutsu.models import QuestionRecord

AT = datetime(2026, 3, 4, 10, 0, tzinfo=UTC)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "question_id": "q-1",
        "repo": "org/repo",
        "pr_number": 42,
        "pr_url": "https://github.com/org/repo/pull/42",
        "question_text": "Why exponential backoff?",
        "category": "design_decision",
        "jira_ticket_key": "PROJ-7",
        "session_id": "session-1",
        "status": "answered",
        "question_author": "opencode",
        "assignee": "worker-1",
        "attempts": 2,
        "answer_comment_id": 555,
        "created_at": "2026-03-01T09:00:00+00:00",
        "answered_at": "2026-03-04T10:00:00+00:00",
        "updated_at": "2026-03-04T10:00:00+00:00",
        # Deliberately present on the row. The projection must not publish it.
        "claim_token": "super-secret-capability-value",
        "last_error": "ValueError: could not reach the store at /Users/someone/secret/path",
    }
    row.update(overrides)
    return row


class _Sink:
    def __init__(self) -> None:
        self.records: list[QuestionRecord] = []

    def store(self, entry: object) -> KnowledgeDeliveryOutcome:
        assert isinstance(entry, QuestionRecord)
        self.records.append(entry)
        return KnowledgeDeliveryOutcome(
            entry_id=entry.question_id, status=KnowledgeDeliveryStatus.DELIVERED
        )


class _Registry:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.reads: list[str | None] = []

    def list_questions(
        self,
        *,
        status: str | None = None,
        repo: str | None = None,
        pr_number: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        self.reads.append(status)
        return [row for row in self._rows if status is None or row.get("status") == status]


# -- the allowlist, which is the whole reason this is a model ------------------


def test_a_claim_token_in_the_registry_row_never_reaches_the_store() -> None:
    """It is a capability, and Tanseki is readable over MCP by agents.

    ``claim_token`` is the ``WHERE`` guard on releasing a claim, so anyone holding
    it can release a claim somebody else holds. Publishing it would hand that to
    every reader of the store.

    The record is asserted to have no such field, which is stronger than
    asserting the payload does not contain the value: there is nowhere for it to
    be read from in the first place.
    """
    record = to_question_record(_row())
    assert "claim_token" not in QuestionRecord.model_fields
    assert "claim_token" not in record.model_dump()
    assert "super-secret-capability-value" not in str(record)


def test_an_internal_error_message_never_reaches_the_store() -> None:
    """``last_error`` carries paths and exception detail, with no analytic value."""
    record = to_question_record(_row())
    assert "last_error" not in QuestionRecord.model_fields
    assert "secret" not in str(record)


def test_the_written_payload_carries_neither_nor_anything_else_unlisted() -> None:
    """The whole payload, not just the two fields under test."""
    record = to_question_record(_row())
    payload = to_payload(record)
    rendered = str(payload)
    assert "super-secret-capability-value" not in rendered
    assert "secret" not in rendered
    assert payload["id"] == "org/repo/pr-42/question/q-1"


def test_a_column_added_to_the_registry_later_is_not_published_by_default() -> None:
    """The projection names what it reads rather than splatting the row.

    A deny-list would make the safe behaviour depend on remembering to update a
    blocklist, and its failure is silent: a document merely lacks a key and
    nothing reports it.
    """
    record = to_question_record(_row(some_future_column="a new piece of state"))
    assert "some_future_column" not in record.model_dump()
    assert "a new piece of state" not in str(record)


# -- terminal states only ------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "claimed"])
def test_an_outstanding_question_cannot_be_projected(status: str) -> None:
    """Outstanding work is operational, and belongs in the registry.

    Projecting it would put a live work queue into the knowledge store, where it
    reads as a set of open decisions rather than as requests this process is
    still waiting on — and the same document id would be rewritten on every claim
    and release, which is a write rate the delivery path is not built for.
    """
    with pytest.raises(ValueError, match="terminal"):
        to_question_record(_row(status=status))


@pytest.mark.parametrize("status", ["answered", "failed", "superseded"])
def test_a_terminal_question_is_projected(status: str) -> None:
    assert to_question_record(_row(status=status)).status == status


def test_an_unknown_status_is_refused_rather_than_projected() -> None:
    with pytest.raises(ValueError, match="terminal"):
        to_question_record(_row(status="something_new"))


# -- a question is never evidence ---------------------------------------------


def test_a_question_carries_no_capture_source_and_no_independence() -> None:
    """Nobody stated anything, so it must sit below every evidence threshold.

    A reader who asked for independent evidence would rather see nothing than
    see a request and assume it had been answered — the same resting place a
    rationale gets, and for the same reason.
    """
    frontmatter = build_question_frontmatter(to_question_record(_row()))
    assert "independence" not in frontmatter
    assert "capture_source" not in frontmatter
    assert "independence_reason" not in frontmatter


def test_the_evidence_axis_is_absent_from_the_type_rather_than_validated() -> None:
    """There is no field to set wrongly, so there is nothing to guard.

    An earlier draft had a validator checking the record carried no capture source
    and no independence. It was dead code: the model has neither field, and a
    guard asserting an absence the type already enforces is a second place for
    the claim to live and drift.
    """
    assert "capture_source" not in QuestionRecord.model_fields
    assert "independence" not in QuestionRecord.model_fields
    assert "independence_reason" not in QuestionRecord.model_fields


# -- eventual consistency, made visible ---------------------------------------


def test_a_status_carries_the_time_it_was_observed() -> None:
    """A reader must be able to see how stale a status is.

    The projection is eventually consistent. A status with no age attached is a
    claim about *now* made by a document about the past, and a reader who cannot
    see the gap will compute a wait that includes time the store never saw.
    """
    frontmatter = build_question_frontmatter(to_question_record(_row()))
    assert frontmatter["status_as_of"] == "2026-03-04T10:00:00+00:00"
    assert frontmatter["question_status"] == "answered"


def test_an_unreadable_timestamp_does_not_lose_the_question() -> None:
    """A format change is not a reason to drop a record."""
    record = to_question_record(
        _row(updated_at="not a timestamp", answered_at=None, created_at=None)
    )
    assert record.question_id == "q-1"
    assert record.updated_at is None
    assert build_question_frontmatter(record)["status_as_of"]


# -- identity and idempotence --------------------------------------------------


def test_the_document_id_is_derived_from_the_change_not_the_status() -> None:
    """So re-projecting an unchanged question upserts rather than duplicates."""
    answered = to_question_record(_row(status="answered"))
    assert question_document_id(answered) == "org/repo/pr-42/question/q-1"


def test_a_question_never_lands_in_the_answer_namespace() -> None:
    """Same reason a rationale lives under ``rationale/``."""
    doc_id = question_document_id(to_question_record(_row()))
    assert "/question/" in doc_id
    assert doc_id.count("/") == 4  # owner, repo, pr-N, question, id


def test_re_projection_is_idempotent_by_document_id() -> None:
    """Running twice writes the same documents, not twice as many.

    Idempotence here is the *store's* upsert keyed on document id, not a dedupe
    in this module: two rows for the same question produce the same id, and the
    second upsert overwrites the first. That is what makes the projection safe to
    run on a schedule and safe to re-run after a failure, which is in turn why it
    does not need the registry to tell it what changed.
    """
    registry = _Registry([_row(), _row(question_id="q-2")])
    sink = _Sink()
    assert len(project_terminal_questions(registry, sink)) == 2
    assert len(project_terminal_questions(registry, sink)) == 2
    assert [record.question_id for record in sink.records] == ["q-1", "q-2"] * 2


def test_two_rows_for_the_same_question_produce_the_same_document_id() -> None:
    """The property the store-side upsert depends on, stated directly."""
    first = question_document_id(to_question_record(_row(status="answered")))
    later = question_document_id(to_question_record(_row(status="superseded", updated_at=None)))
    assert first == later == "org/repo/pr-42/question/q-1"


def test_a_malformed_row_does_not_stop_the_rest_of_the_queue() -> None:
    """One bad row must not mean the rest go unrecorded."""
    registry = _Registry([_row(question_id="", status="answered"), _row(question_id="q-2")])
    outcomes = project_terminal_questions(registry, _Sink())
    assert len(outcomes) == 1


def test_outstanding_questions_are_never_read_by_the_projection() -> None:
    """The registry is read per terminal status, so pending is not even fetched."""
    registry = _Registry([_row(status="pending")])
    assert project_terminal_questions(registry, _Sink()) == []
    assert "pending" not in registry.reads


# -- the cost the ticket asked to be measured ----------------------------------


def test_the_sweep_reports_the_queue_depth_it_left_behind() -> None:
    """A question is re-derivable; a captured answer is the only copy of a human's
    words. If the sweep queues work, it is spending the valuable thing to store
    the derivable one — so the number that decides whether the sweep keeps running
    is the one it reports.
    """
    from kojutsu.runtime import QuestionProjectionResult

    clean = QuestionProjectionResult(questions_projected=3, outbox_pending_after=0)
    assert clean.backpressure is False
    assert int(clean) == 3

    pressured = QuestionProjectionResult(questions_projected=3, outbox_pending_after=12)
    assert pressured.backpressure is True


def test_a_projection_result_is_not_a_deletion_report() -> None:
    """It must not report a truthy "total deleted" of zero for 300 writes.

    Reusing MaintenanceResult would have done exactly that, and a sweep that
    wrote three hundred documents reporting "total deleted: 0" is the sort of
    small lie that makes an operational surface untrustworthy.
    """
    from kojutsu.runtime import MaintenanceResult, QuestionProjectionResult

    result = QuestionProjectionResult(questions_projected=300, outbox_pending_after=0)
    assert not isinstance(result, MaintenanceResult)
    assert "deleted" not in set(result.__dataclass_fields__)

    # Truthiness answers the same question MaintenanceResult's does — did this do
    # anything — rather than defaulting to True for every instance.
    assert bool(result) is True
    assert bool(QuestionProjectionResult(questions_projected=0, outbox_pending_after=0)) is False
