"""Tests for the Tanseki capture sink."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from kojutsu.core.knowledge_sink import KnowledgeDeliveryStatus, TansekiKnowledgeSink
from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.tanseki_mapping import document_id
from kojutsu.integrations.tanseki import (
    TansekiAuthenticationError,
    TansekiClient,
    TansekiError,
    TansekiPermanentError,
    TansekiUnavailableError,
)
from kojutsu.models import KnowledgeEntry, QuestionCategory


def make_entry() -> KnowledgeEntry:
    return KnowledgeEntry(
        entry_id="e1",
        question_text="Why?",
        answer_text="Because.",
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        metadata={"repo": "org/repo", "pr_number": 1},
    )


class FakeWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[dict] = []

    def upsert_document(self, payload: dict) -> dict:
        if self.fail:
            raise TansekiError("down")
        self.payloads.append(payload)
        return {}


def test_tanseki_sink_sends_and_clears(tmp_path: Path) -> None:
    writer = FakeWriter()
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        sink = TansekiKnowledgeSink(writer, outbox)
        outcome = sink.store(make_entry())
        assert outcome.status is KnowledgeDeliveryStatus.DELIVERED
        assert outbox.pending_count() == 0
    assert writer.payloads[0]["id"] == document_id(make_entry())


def test_tanseki_sink_never_returns_none(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        assert TansekiKnowledgeSink(FakeWriter(), outbox).store(make_entry()) is not None


def test_tanseki_sink_keeps_entry_when_store_down(tmp_path: Path) -> None:
    before = datetime.now(UTC)
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        sink = TansekiKnowledgeSink(FakeWriter(fail=True), outbox)
        outcome = sink.store(make_entry())
        assert outcome.status is KnowledgeDeliveryStatus.QUEUED
        assert outcome.retryable
        assert outbox.pending_count() == 1
        item = outbox.pending()[0]
        assert item.payload["id"] == document_id(make_entry())
        assert item.state == "delivery-retrying"
        assert item.next_attempt_at is not None
        assert datetime.fromisoformat(item.next_attempt_at) > before
        assert outbox.ready() == []


def test_tanseki_sink_honors_explicit_zero_retry_after(tmp_path: Path) -> None:
    class RetryAfterWriter(FakeWriter):
        def upsert_document(self, payload: dict) -> dict:
            raise TansekiUnavailableError("down", retry_after=0)

    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        sink = TansekiKnowledgeSink(RetryAfterWriter(), outbox)
        sink.store(make_entry())
        assert outbox.ready()[0].entry_id == "e1"
        assert outbox.pending()[0].next_attempt_at is not None


def test_tanseki_sink_dead_letters_permanent_delivery_failure(tmp_path: Path) -> None:
    class PermanentFailureWriter(FakeWriter):
        def upsert_document(self, payload: dict) -> dict:
            raise TansekiPermanentError("rejected")

    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        sink = TansekiKnowledgeSink(PermanentFailureWriter(), outbox)
        outcome = sink.store(make_entry())
        assert outcome.status is KnowledgeDeliveryStatus.DEAD_LETTERED
        assert outcome.retryable
        assert outbox.pending_count() == 0
        assert outbox.status_counts()["dead_letter"] == 1
        assert outbox.dead_letters()[0].state == "delivery-failed"


@pytest.mark.parametrize("status_code", [401, 403, 408, 425])
def test_tanseki_auth_and_transport_statuses_remain_retryable(
    tmp_path: Path, status_code: int
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status_code, headers={"Retry-After": "0"})
    )
    with (
        httpx.Client(base_url="https://tanseki.test/v1", transport=transport) as http,
        TansekiOutbox(tmp_path / "outbox.db") as outbox,
    ):
        client = TansekiClient("https://tanseki.test", client=http, max_retries=0)
        outcome = TansekiKnowledgeSink(client, outbox).store(make_entry())
        item = outbox.pending()[0]

    assert outcome.status is KnowledgeDeliveryStatus.QUEUED
    assert item.status == "retrying"
    assert item.last_error is not None
    assert str(status_code) in item.last_error


def test_tanseki_authentication_error_is_operator_actionable() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401))
    with httpx.Client(base_url="https://tanseki.test/v1", transport=transport) as http:
        client = TansekiClient("https://tanseki.test", client=http, max_retries=0)
        with pytest.raises(TansekiAuthenticationError) as captured:
            client.upsert_document({"id": "d1", "content": "x"})

    assert captured.value.operator_action_required is True


def test_tanseki_validation_response_dead_letters(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(422))
    with (
        httpx.Client(base_url="https://tanseki.test/v1", transport=transport) as http,
        TansekiOutbox(tmp_path / "outbox.db") as outbox,
    ):
        client = TansekiClient("https://tanseki.test", client=http, max_retries=0)
        outcome = TansekiKnowledgeSink(client, outbox).store(make_entry())

    assert outcome.status is KnowledgeDeliveryStatus.DEAD_LETTERED
