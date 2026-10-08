"""Tests for the census record: an observation that captured nothing.

A census record exists to make the denominator of the corpus visible. Every test
below is a boundary of the claim it makes, because the claim is narrow: *this
delivery was processed and nothing was captured from it*. It is not a claim about
the change, and not a claim about anyone's intent. Those two overreaches are the
ones worth pinning, because each is individually easy and looks like a feature.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kojutsu.core.answer_collector import RECORD_KINDS, RecordKind
from kojutsu.core.knowledge_sink import to_payload
from kojutsu.core.tanseki_mapping import (
    CENSUS_NAMESPACE,
    build_census_frontmatter,
    census_document_id,
    to_census_upsert_payload,
)
from kojutsu.models import CaptureSource, CensusRecord

OBSERVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _record(**overrides: object) -> CensusRecord:
    fields: dict[str, object] = {
        "entry_id": "census-org-repo-pr-42-opened",
        "repo": "org/repo",
        "pr_number": 42,
        "pr_url": "https://example.test/org/repo/pull/42",
        "action": "opened",
        "change_author_account": "octocat",
        "head_sha": "abc123",
        "observed_at": OBSERVED_AT,
        "delivery_id": "delivery-1",
    }
    fields.update(overrides)
    return CensusRecord(**fields)


def test_a_census_record_is_its_own_record_kind() -> None:
    """It is a closed vocabulary, so a future writer cannot invent a second spelling."""
    assert RecordKind.CENSUS.value == "census"
    assert "census" in RECORD_KINDS


def test_a_census_record_lives_in_its_own_namespace() -> None:
    """It must never land where a reader would take it for a decision."""
    assert CENSUS_NAMESPACE == "census"
    assert f"/{CENSUS_NAMESPACE}/" in census_document_id(_record())


def test_a_census_record_keeps_the_webhook_capture_source() -> None:
    """A real signed delivery that happened to yield nothing.

    Not a fourth value: how the delivery arrived and what it yielded are two
    questions, and ``webhook`` answers the first accurately. This is also why the
    anchor rules needed no new rule for this record — the delivery id is its anchor.
    """
    assert _record().capture_source is CaptureSource.WEBHOOK


def test_a_census_record_is_never_counted_as_a_capture() -> None:
    """It reached the store through a delivery and captured nothing."""
    assert _record().is_captured is False


def test_a_census_record_without_a_delivery_id_is_refused() -> None:
    """The claim "we processed this and kept nothing" is the easiest to invent.

    A single extra document would let a reader believe a change was examined when
    nothing happened, so the record cannot exist without naming the delivery it
    claims to have come from.
    """
    with pytest.raises(ValidationError, match="delivery_id is required"):
        _record(delivery_id=None)

    with pytest.raises(ValidationError, match="delivery_id is required"):
        _record(delivery_id="   ")


def test_a_census_record_carries_no_reason_and_no_independence() -> None:
    """Absence is the mechanism: each reason would be a guess about someone's intent.

    No comment posted, no authorised reviewer, a malformed marker, a change nobody
    reviewed — all plausible, all hypotheses about another person's behaviour. A
    field that existed and stayed empty would be a place for a later writer to put
    one, which is why the model has no such field to leave empty.
    """
    frontmatter = build_census_frontmatter(_record())

    assert "reason" not in frontmatter
    assert "census_reason" not in frontmatter
    assert "independence" not in frontmatter
    assert "independence_reason" not in frontmatter


def test_a_census_record_states_the_observation_and_not_a_conclusion() -> None:
    """The body is the claim, so it is written out rather than left to be inferred.

    "No knowledge" would be a claim about the change; an empty answer body would
    read as a capture that found nothing to say. Both are more flattering, and
    neither is what happened.
    """
    payload = to_census_upsert_payload(_record())
    body = str(payload["content"])

    assert "nothing was captured" in body
    assert "not about the change" in body
    assert "does not say" in body


def test_a_census_record_says_it_is_about_one_event_not_the_whole_change() -> None:
    """A change opened quietly and reviewed with a capture legitimately has both.

    Without this sentence a reader compares a count of census records against a
    count of captures and concludes the corpus is in better shape than it is.
    """
    body = str(to_census_upsert_payload(_record())["content"])

    assert "statement about one event" in body
    assert "quietly" in body


def test_the_census_document_is_keyed_on_the_change_and_not_on_the_delivery() -> None:
    """Two deliveries of the same action are two observations of one fact.

    Keying on the delivery would accumulate a record per redelivery, and a count
    over census documents would then be a count over traffic rather than over the
    changes the corpus has seen.
    """
    first = census_document_id(_record(delivery_id="delivery-1"))
    second = census_document_id(_record(delivery_id="delivery-2"))

    assert first == second
    assert first == "org/repo/pr-42/census/opened"


def test_a_census_document_id_separates_actions_on_the_same_change() -> None:
    """A change observed opened and then reviewed with no capture is two facts."""
    opened = census_document_id(_record(action="opened"))
    synchronize = census_document_id(_record(action="synchronize"))

    assert opened != synchronize


def test_a_census_record_is_dispatched_to_its_own_renderer() -> None:
    """Dispatch is on type, so a census record cannot be written as knowledge.

    Falling through to the entry renderer would store an answer nobody gave, which
    is the specific failure the separate model exists to prevent.
    """
    payload = to_payload(_record())

    assert payload["id"] == "org/repo/pr-42/census/opened"
    assert payload["frontmatter"]["record_kind"] == "census"
    assert payload["frontmatter"]["delivery_id"] == "delivery-1"
    assert payload["frontmatter"]["capture_source"] == "webhook"


def test_a_census_record_carries_its_anchors_into_frontmatter() -> None:
    """As checkable as any capture: the reader can re-fetch the delivery."""
    frontmatter = build_census_frontmatter(_record())

    assert frontmatter["repo"] == "org/repo"
    assert frontmatter["pr"] == 42
    assert frontmatter["delivery_id"] == "delivery-1"
    assert frontmatter["observed_at"] == OBSERVED_AT.isoformat()
    assert frontmatter["head_sha"] == "abc123"


def test_a_census_record_without_a_change_author_says_unknown_rather_than_inventing_one() -> None:
    """The author of a change is a login, not a person, and absence stays absence."""
    frontmatter = build_census_frontmatter(_record(change_author_account=None))

    assert "change_author_account" not in frontmatter
    assert frontmatter["author"] != ""
