"""Tests for the census write path: a delivery that was processed and captured nothing.

The record itself is pinned in ``test_census_capture.py`` — what it may say, and
what it is forbidden to say. What is pinned here is the narrower thing the store
cannot check for itself: **which deliveries produce one.** Every test below is a
boundary of that decision, because both of its failure modes are silent. An
observation written for a delivery that was never processed claims a silence that
did not happen; one withheld from a delivery that really was silent is invisible,
and the corpus quietly becomes a count of what it chose to write down.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from kojutsu import runtime as runtime_module
from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryOutcome,
    KnowledgeDeliveryStatus,
    StorableRecord,
    to_payload,
)
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.core.tanseki_mapping import CENSUS_NAMESPACE
from kojutsu.models import CensusRecord, KnowledgeEntry
from kojutsu.webhook import app

client = TestClient(app)
WEBHOOK_SECRET = "test-secret"
DELIVERY_ID = "11111111-1111-4111-8111-111111111111"
OTHER_DELIVERY_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def allow_local_webhook_repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "org/repo")


class RecordingSink:
    """Records every record the write path stores, whatever kind it is.

    Deliberately not split by kind: the question these tests ask is "what reached
    the store", and a fake that only kept knowledge entries would answer it by
    dropping the observation on the floor and passing.
    """

    def __init__(self, status: KnowledgeDeliveryStatus = KnowledgeDeliveryStatus.DELIVERED) -> None:
        self.records: list[StorableRecord] = []
        self.status = status

    def store(self, record: StorableRecord) -> KnowledgeDeliveryOutcome:
        self.records.append(record)
        return KnowledgeDeliveryOutcome(entry_id=record.entry_id, status=self.status)

    @property
    def observations(self) -> list[CensusRecord]:
        return [r for r in self.records if isinstance(r, CensusRecord)]

    @property
    def captures(self) -> list[KnowledgeEntry]:
        return [r for r in self.records if isinstance(r, KnowledgeEntry)]


class FakeRuntime:
    def __init__(self, registry: SqliteQuestionRegistry, sink: RecordingSink) -> None:
        self.registry = registry
        self.sink = sink


def _signed_headers(body: bytes, event: str, delivery_id: str | None = None) -> dict[str, str]:
    digest = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": event,
        "X-Hub-Signature-256": f"sha256={digest}",
    }
    headers["X-GitHub-Delivery"] = delivery_id or DELIVERY_ID
    return headers


def _post_signed(payload: dict, event: str, delivery_id: str | None = None) -> httpx.Response:
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook/github",
        content=body,
        headers=_signed_headers(body, event, delivery_id),
    )


def _webhook_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: KnowledgeDeliveryStatus = KnowledgeDeliveryStatus.DELIVERED,
) -> tuple[SqliteQuestionRegistry, RecordingSink]:
    registry = SqliteQuestionRegistry(tmp_path / "r.db")
    sink = RecordingSink(status)
    monkeypatch.setattr(runtime_module, "_runtime", FakeRuntime(registry, sink))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", WEBHOOK_SECRET)
    return registry, sink


def _pull_request_payload(action: str, number: int = 5, sha: str = "abc123") -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": number,
            "title": "Add retry",
            "state": "open",
            "user": {"login": "author"},
            "head": {"sha": sha},
        },
        "repository": {"full_name": "org/repo"},
    }


def _review_payload(
    *,
    number: int = 5,
    review_id: int = 900,
    state: str = "changes_requested",
    body: str = "The retry loop swallows the error.",
    association: str | None = "MEMBER",
    comments: list[dict] | None = None,
) -> dict:
    return {
        "action": "submitted",
        "review": {
            "id": review_id,
            "state": state,
            "body": body,
            "user": {"login": "reviewer"},
            "submitted_at": "2023-01-01T00:00:00Z",
            "author_association": association,
        },
        "pull_request": {
            "number": number,
            "title": "Add retry",
            "state": "open",
            "user": {"login": "author"},
            "head": {"sha": "abc123"},
        },
        "repository": {"full_name": "org/repo"},
        "comments": comments
        if comments is not None
        else [
            {
                "id": 901,
                "body": "This loses the original error.",
                "user": {"login": "reviewer"},
                "created_at": "2023-01-01T00:00:00Z",
                "author_association": association,
                "path": "src/retry.py",
                "line": 42,
            }
        ],
    }


# --- the pull request event path (A919PRD5) ----------------------------------


def test_a_pull_request_that_captured_writes_no_observation(tmp_path, monkeypatch) -> None:
    """The capture is the record. An observation beside it would claim silence.

    The census exists to make the denominator visible, and a change that opened
    quietly and is discussed tomorrow legitimately has both. What it must never
    have is an observation for an event that *did* capture: a count over census
    documents that includes it would report a change nobody engaged with as one
    that was never engaged with.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    response = _post_signed(_pull_request_payload("opened"), "pull_request")

    assert response.status_code == 200
    assert response.json() == {"status": "processed", "stored": True, "delivery": "delivered"}
    assert len(sink.captures) == 1
    assert sink.observations == []


def test_a_pull_request_that_captured_nothing_writes_an_observation(tmp_path, monkeypatch) -> None:
    """The observation names the delivery it came from and when it was seen.

    Both are what make the claim checkable rather than assertable: a reader can
    re-fetch the delivery behind the document, and can tell the system's clock
    from the change's own history. Neither may be inferred afterwards.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    response = _post_signed(_pull_request_payload("synchronize"), "pull_request")

    assert response.status_code == 200
    assert response.json()["census"] == "delivered"
    assert sink.captures == []
    assert len(sink.observations) == 1
    observation = sink.observations[0]
    assert observation.repo == "org/repo"
    assert observation.pr_number == 5
    assert observation.action == "synchronize"
    assert observation.delivery_id == DELIVERY_ID
    assert observation.observed_at is not None
    assert observation.change_author_account == "author"
    assert observation.head_sha == "abc123"
    # The document is keyed on the change, so a count over census documents is a
    # count over changes rather than over the pushes that produced them.
    assert str(to_payload(observation)["id"]) == "org/repo/pr-5/census/synchronize"


def test_a_change_observed_silently_and_later_captured_has_both(tmp_path, monkeypatch) -> None:
    """Both is the honest state, and the two never collapse into one another.

    The census is a statement about one delivery, not a verdict on the change. A
    change that moved once with nobody reviewing it and was reviewed the next day
    is exactly the case where a reader who conflated the two would conclude the
    corpus is healthier than it is — so the observation and the capture have to
    stay separately addressable, which they are because they are separate
    documents in separate namespaces.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    _post_signed(_pull_request_payload("synchronize"), "pull_request")
    _post_signed(_review_payload(), "pull_request_review", OTHER_DELIVERY_ID)

    observation_ids = {str(to_payload(record)["id"]) for record in sink.observations}
    capture_ids = {str(to_payload(record)["id"]) for record in sink.captures}
    assert observation_ids == {"org/repo/pr-5/census/synchronize"}
    assert len(sink.captures) == 2, "the review verdict and its inline comment"
    # The captures sit outside the census namespace, so "this change has both" is
    # answerable from the corpus rather than something a reader has to infer.
    assert all(f"/{CENSUS_NAMESPACE}/" not in document_id for document_id in capture_ids)


def test_a_repository_outside_the_allowlist_writes_neither_record(tmp_path, monkeypatch) -> None:
    """Refused before any writer runs, on every event type.

    Verified rather than assumed, because an observation is a claim that a
    delivery was examined: a census record for a repository the operator never
    configured would assert a collection that does not exist.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "org/allowed")
    synchronize = _pull_request_payload("synchronize")
    synchronize["repository"]["full_name"] = "other/repo"
    review = _review_payload()
    review["repository"]["full_name"] = "other/repo"

    synchronize_response = _post_signed(synchronize, "pull_request")
    review_response = _post_signed(review, "pull_request_review", OTHER_DELIVERY_ID)

    assert synchronize_response.status_code == 403
    assert review_response.status_code == 403
    assert sink.records == []


def test_a_duplicate_delivery_writes_a_second_of_neither(tmp_path, monkeypatch) -> None:
    """GitHub redelivers, and often. The second pass must change nothing at all.

    Not only the count: a second pass that re-wrote the observation would replace
    the delivery the document names with the newer one, so the corpus would end up
    pointing at a delivery that never carried the event being claimed.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    first = _post_signed(_pull_request_payload("synchronize"), "pull_request")
    second = _post_signed(_pull_request_payload("synchronize"), "pull_request")

    assert first.json()["census"] == "delivered"
    assert second.json() == {"status": "duplicate", "stored": False}
    assert len(sink.observations) == 1
    assert sink.observations[0].delivery_id == DELIVERY_ID


def test_a_redelivered_push_is_one_document_not_one_per_push(tmp_path, monkeypatch) -> None:
    """The key is the change and the action, so ten pushes are one observation.

    Delivery-keyed, each push would be its own document and the census would be
    measuring traffic. Keyed this way the corpus answers "how many changes did we
    observe with nothing", which is the only question the record is honest enough
    to ask. The sink is asked once per delivery and returns the same identity each
    time, so the store collapses them.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    for index in range(3):
        payload = _pull_request_payload("synchronize", sha=f"sha{index}")
        _post_signed(payload, "pull_request", f"{index:08d}-1111-4111-8111-111111111111")

    assert len(sink.observations) == 3, "the sink is asked once per delivery"
    assert len({observation.entry_id for observation in sink.observations}) == 1
    document_ids = {str(to_payload(observation)["id"]) for observation in sink.observations}
    assert document_ids == {"org/repo/pr-5/census/synchronize"}


def test_an_observation_that_dead_letters_is_surfaced_and_holds_the_delivery(
    tmp_path, monkeypatch
) -> None:
    """A missing observation is a missing claim, so the delivery stays retryable.

    The operator sees 503 and re-delivers; completing the delivery instead would
    leave the corpus claiming a silence it never recorded, and the only record of
    the gap would be a delivery status nobody reads.
    """
    registry, _ = _webhook_runtime(
        tmp_path, monkeypatch, status=KnowledgeDeliveryStatus.DEAD_LETTERED
    )

    response = _post_signed(_pull_request_payload("synchronize"), "pull_request")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert (
        registry._db.execute(
            "SELECT status FROM webhook_deliveries WHERE delivery_id = ?", (DELIVERY_ID,)
        ).fetchone()[0]
        == "retryable"
    )


# --- the review event path (64RA7RKJ) ----------------------------------------


def test_a_review_that_captured_nothing_writes_an_observation(tmp_path, monkeypatch) -> None:
    """A note with no verdict and no inline comment says nothing, and that is data.

    It is also the least dramatic uncaptured event there is, which is why it is the
    one that gets left out of a census built by whoever remembered the loud cases.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)
    payload = _review_payload(state="commented", body="", comments=[])

    response = _post_signed(payload, "pull_request_review")

    assert response.status_code == 200
    assert response.json()["stored"] is False
    assert response.json()["census"] == "delivered"
    assert sink.captures == []
    assert len(sink.observations) == 1
    assert sink.observations[0].action == "submitted"
    assert sink.observations[0].pr_number == 5


def test_a_declined_review_event_records_no_reason(tmp_path, monkeypatch) -> None:
    """The case most likely to be "improved" with a reason code, so it is pinned.

    Kojutsu does know why nothing was captured here: it declined the capture.
    That is a fact about its own configuration and nothing about what the reviewer
    meant, and writing it into the corpus would dress configuration up as a finding
    about a person. The absence has to hold at the frontmatter, where a later writer
    would put it, and in the body, which is the claim a reader reads.

    **The gate used to be the association one, and that route is closed.** Admission
    is unrestricted now, so a ``CONTRIBUTOR`` reviewer is captured like anyone else
    and the webhook has no setting that could narrow it back -- the review path's
    override belongs to the backfill run, which is the only caller that passes one.
    The property is therefore re-pinned against the gate the webhook *can* still
    reach, an account GitHub could not attribute, because the mechanism under test is
    the census entry and not which gate produced the empty result.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)
    payload = _review_payload(association="MEMBER")
    payload["review"]["user"] = {"login": ""}

    _post_signed(payload, "pull_request_review")

    assert sink.captures == [], "the capture is still declined"
    assert len(sink.observations) == 1
    payload_for_store = to_payload(sink.observations[0])
    frontmatter = payload_for_store["frontmatter"]
    assert isinstance(frontmatter, dict)
    assert "reason" not in frontmatter
    assert "census_reason" not in frontmatter
    assert "independence" not in frontmatter
    # The body is the claim a reader actually reads, so it must not narrate the
    # decline either: the authorisation decision is kojutsu's own and stays out
    # of the corpus.
    content = str(payload_for_store["content"])
    assert "refus" not in content.casefold()
    assert "does not say" in content


def test_a_review_observation_that_dead_letters_holds_the_delivery(tmp_path, monkeypatch) -> None:
    """The review path surfaces its dead letters too, not just the capture path.

    A delivery completed as though it were recorded leaves the corpus one silence
    short of what it processed, and there is no later pass that would notice: the
    next delivery for that review is a re-delivery, and a re-delivery records
    nothing. The 503 is the only place that gap is still visible.
    """
    registry, _ = _webhook_runtime(
        tmp_path, monkeypatch, status=KnowledgeDeliveryStatus.DEAD_LETTERED
    )
    # A review with nothing to say, so the observation is what gets written and it is
    # the observation that dead-letters. The fixture used to reach the same place by
    # being refused: a CONTRIBUTOR reviewer was dropped before anything was attempted,
    # and admission is unrestricted now.
    payload = _review_payload(body="", comments=[])

    response = _post_signed(payload, "pull_request_review")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert (
        registry._db.execute(
            "SELECT status FROM webhook_deliveries WHERE delivery_id = ?", (DELIVERY_ID,)
        ).fetchone()[0]
        == "retryable"
    )


def test_a_review_that_captured_a_verdict_writes_no_observation(tmp_path, monkeypatch) -> None:
    """Something was said about the change, so there is nothing to observe."""
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    response = _post_signed(_review_payload(), "pull_request_review")

    assert response.status_code == 200
    assert response.json()["records"] == 2
    assert response.json()["stored"] is True
    assert sink.observations == []


def test_a_reserialised_review_writes_no_observation(tmp_path, monkeypatch) -> None:
    """A review redelivered under a fresh delivery id is a duplicate, not silence.

    The delivery claim cannot catch this one — GitHub re-serialises with a new id —
    and the semantic review id can, which is why the capture path keys on it. The
    observation has to key on the same distinction: it would otherwise claim a
    delivery captured nothing while the capture for that very review sits in the
    store.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    first = _post_signed(_review_payload(), "pull_request_review")
    second = _post_signed(_review_payload(), "pull_request_review", OTHER_DELIVERY_ID)

    assert first.json()["records"] == 2
    assert second.json()["records"] == 0
    assert len(sink.captures) == 2
    assert sink.observations == []


def test_an_edited_review_action_is_never_processed_and_so_never_observed(
    tmp_path, monkeypatch
) -> None:
    """An ignored action was not processed, so it observed nothing.

    This is the rule most likely to be broken by a later "let's record what we
    skipped" change: the refusal below is a decision not to process the delivery,
    and an observation for it would assert an examination that never happened.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)
    payload = _review_payload()
    payload["action"] = "edited"

    response = _post_signed(payload, "pull_request_review")

    assert response.json() == {"status": "ignored", "action": "edited"}
    assert sink.records == []


def test_a_review_observation_is_a_different_document_from_a_pull_request_one(
    tmp_path, monkeypatch
) -> None:
    """Two observations about one change are two facts, so they are two documents.

    They share the change and differ only in the event, and the document key is the
    only thing that says which — a field naming the event kind would have to be
    kept correct by every future writer for a reader who has not appeared yet.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    _post_signed(_pull_request_payload("synchronize"), "pull_request")
    _post_signed(
        _review_payload(state="commented", body="", comments=[]),
        "pull_request_review",
        OTHER_DELIVERY_ID,
    )

    document_ids = sorted(str(to_payload(record)["id"]) for record in sink.observations)
    assert document_ids == ["org/repo/pr-5/census/submitted", "org/repo/pr-5/census/synchronize"]


def test_a_review_observation_is_the_same_shape_as_a_pull_request_one(
    tmp_path, monkeypatch
) -> None:
    """One record kind, one namespace, one counting rule — checked by inspection.

    A dashboard filtering on ``record_kind: census`` has to count these two the
    same way, and it can only do that while they are the same record. So the
    payloads are compared key for key: everything but the change the delivery was
    about must be identical, and a field one path carries and the other does not is
    exactly the split that would make the count a lie.
    """
    _, sink = _webhook_runtime(tmp_path, monkeypatch)

    _post_signed(_pull_request_payload("synchronize"), "pull_request")
    _post_signed(
        _review_payload(state="commented", body="", comments=[]),
        "pull_request_review",
        OTHER_DELIVERY_ID,
    )

    payloads = [to_payload(observation) for observation in sink.observations]
    frontmatters = []
    for payload in payloads:
        frontmatter = payload["frontmatter"]
        assert isinstance(frontmatter, dict)
        assert frontmatter["record_kind"] == "census"
        assert frontmatter["tags"] == ["census"]
        assert frontmatter["capture_source"] == "webhook"
        assert frontmatter["observed_at"]
        assert frontmatter["delivery_id"]
        assert str(payload["path"]).endswith(".md")
        frontmatters.append(frontmatter)

    # Same keys, and the same values under every key that does not name the event.
    # A field one path carries and the other does not is exactly the split that
    # would make the count a lie, so the key sets are compared as sets rather than
    # as a subset check.
    per_event = {"action", "title", "census_id", "delivery_id", "observed_at"}
    assert set(frontmatters[0]) == set(frontmatters[1])
    assert {k: v for k, v in frontmatters[0].items() if k not in per_event} == {
        k: v for k, v in frontmatters[1].items() if k not in per_event
    }
