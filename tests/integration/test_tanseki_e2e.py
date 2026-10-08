"""Cross-seam end-to-end tests: capture -> outbox -> Tanseki -> retrieve.

Runs against an in-process fake of the Tanseki ``/v1`` API (httpx.MockTransport),
so it exercises the real client, mapping, outbox, sink, and relay without a
running store. Graph edges are derived server-side by Tanseki's ``EdgeDeriver``,
so they are not exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from kojutsu.core.knowledge_sink import TansekiKnowledgeSink
from kojutsu.core.outbox import TansekiOutbox, relay
from kojutsu.core.tanseki_mapping import document_id
from kojutsu.integrations.tanseki import TansekiClient
from kojutsu.models import KnowledgeEntry, QuestionCategory

BASE = "https://tanseki.test/v1"


class FakeTanseki:
    """Minimal in-memory Tanseki /v1 implementation."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def client(self, *, fail: bool = False) -> TansekiClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if fail:
                return httpx.Response(503, headers={"Retry-After": "0"})
            return self._handle(request)

        http = httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))
        return TansekiClient("https://tanseki.test", api_key="test-key", client=http, max_retries=0)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/health":
            return httpx.Response(200, json={"status": "ok"})

        if path == "/v1/documents:get":
            body = json.loads(request.read() or b"{}")
            doc = self.docs.get(body.get("id"))
            return httpx.Response(200, json=doc) if doc else httpx.Response(404)

        if path == "/v1/documents:upsert":
            body = json.loads(request.read())
            doc_id = body["id"]
            self.docs[doc_id] = {
                "id": doc_id,
                "collection": body.get("collection", "kojutsu"),
                "path": body.get("path", ""),
                "content": body.get("content", ""),
                "contentHash": "hash",
                "revision": "r1",
                "updatedAt": "2026-01-01T00:00:00Z",
                "deleted": False,
                "frontmatter": body.get("frontmatter", {}),
            }
            return httpx.Response(200, json={"id": doc_id, "revision": "r1", "created": True})

        if path == "/v1/documents:delete":
            body = json.loads(request.read() or b"{}")
            remaining = self.docs.pop(body.get("id"), None)
            return httpx.Response(200 if remaining else 404, json={})

        if path == "/v1/documents:traverse":
            return httpx.Response(200, json={"ids": []})

        if path == "/v1/search":
            query = (request.url.params.get("q") or "").lower()
            limit = int(request.url.params.get("limit", 10))
            hits = [
                {"id": i, "score": 1.0, "snippet": d.get("content", "")[:60]}
                for i, d in self.docs.items()
                if query in d.get("content", "").lower()
            ]
            page = hits[:limit]
            return httpx.Response(
                200,
                json={
                    "hits": page,
                    "limit": limit,
                    "offset": 0,
                    "hasMore": len(hits) > limit,
                },
            )

        return httpx.Response(404)


def make_entry() -> KnowledgeEntry:
    return KnowledgeEntry(
        entry_id="e1",
        question_text="Why this approach?",
        answer_text="Because X.",
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        tags=["design"],
        metadata={"repo": "org/repo", "pr_number": 42},
    )


def test_capture_store_retrieve_round_trip(tmp_path: Path) -> None:
    store = FakeTanseki()
    entry = make_entry()

    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        sink = TansekiKnowledgeSink(store.client(), outbox)
        sink.store(entry)
        assert outbox.pending_count() == 0  # delivered immediately

    client = store.client()
    assert client.health() is True

    hits = client.search("Because X")
    assert [h.id for h in hits] == [document_id(entry)]

    doc = client.get_document(document_id(entry))
    assert doc is not None
    assert "Because X" in doc.content
    assert doc.frontmatter["repo"] == "org/repo"
    # Read back through the store, so this pins the whole seam: the value that goes
    # out is typed and the value that comes back is the same type rather than its
    # spelling.
    assert doc.frontmatter["pr"] == 42
    assert isinstance(doc.frontmatter["pr"], int)


def test_outage_then_relay_recovers(tmp_path: Path) -> None:
    store = FakeTanseki()
    entry = make_entry()

    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        # Tanseki is down: the write is queued, not lost.
        TansekiKnowledgeSink(store.client(fail=True), outbox).store(entry)
        assert outbox.pending_count() == 1

        # Tanseki recovers: the relay drains the outbox.
        result = relay(outbox, store.client())
        assert result.sent == 1
        assert outbox.pending_count() == 0

    assert document_id(entry) in store.docs
