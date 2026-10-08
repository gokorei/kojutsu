"""Tests for the minimal dev console (fake client, no network)."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kojutsu.config import Settings
from kojutsu.core.tanseki_mapping import (
    to_clarification_upsert_payload,
    to_evaluation_upsert_payload,
    to_rationale_upsert_payload,
    to_upsert_payload,
)
from kojutsu.dev_console import MAX_DASHBOARD_ANSWER_CHARS, MAX_DOCUMENT_BYTES, create_console_app
from kojutsu.integrations.tanseki import (
    MAX_LIST_RESULTS,
    TansekiDocument,
    TansekiError,
    TansekiHit,
)
from kojutsu.models import (
    STRUCTURE_INFERRED_BY,
    CaptureSource,
    ClarificationEntry,
    EvaluationEntry,
    EvaluationTarget,
    KnowledgeEntry,
    QuestionCategory,
    RationaleEntry,
    RecordStructure,
)

DOC_ID = "org/repo/pr-1/e1"


class StubTanseki:
    """Records calls and returns canned Tanseki payloads."""

    def __init__(self) -> None:
        self.closed = False
        self.search_calls: list[tuple[str, dict[str, str] | None, int]] = []
        self.list_calls: list[int] = []
        self.list_error: bool = False

    def health(self) -> bool:
        return True

    def count(self, collection: str | None = None) -> int:
        return 3

    def search(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        frontmatter: dict[str, str] | None = None,
        limit: int = 10,
        collection: str | None = None,
    ) -> list[TansekiHit]:
        self.search_calls.append((query, frontmatter, limit))
        return [TansekiHit(id=DOC_ID, score=1.5, snippet="...tokens...")]

    def get_document(self, doc_id: str, collection: str | None = None) -> TansekiDocument | None:
        if doc_id != DOC_ID:
            return None
        return TansekiDocument(
            id=doc_id,
            path=f"{doc_id}.md",
            collection="kojutsu",
            content="body: tokens over sessions",
            revision="r1",
            frontmatter={"repo": "org/repo"},
        )

    def list_documents(
        self, *, limit: int = MAX_LIST_RESULTS, collection: str | None = None
    ) -> list[str]:
        self.list_calls.append(limit)
        if self.list_error:
            raise TansekiError("host=internal.example token=super-secret")
        return [DOC_ID]

    def close(self) -> None:
        self.closed = True


def make_client(
    tmp_path: Path, *, tanseki_url: str = "http://tanseki.test"
) -> tuple[TestClient, StubTanseki]:
    settings = Settings(
        tanseki_url=tanseki_url,
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
        kojutsu_registry_path=str(tmp_path / "registry.db"),
    )
    stub = StubTanseki()
    app = create_console_app(settings, client_factory=lambda _settings: stub)
    return TestClient(app), stub


def test_index_renders_html(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Kojutsu dev console" in response.text


def test_status_reports_health_count_and_outbox(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    payload = client.get("/api/status").json()
    assert payload["tanseki_reachable"] is True
    assert payload["document_count"] == 3
    assert payload["outbox_pending"] == 0
    assert payload["outbox_captured_locally"] == 0
    assert payload["outbox_delivery_failed"] == 0
    assert payload["tanseki_url"] == "http://tanseki.test"


def test_search_forwards_filters_and_limit(tmp_path: Path) -> None:
    client, stub = make_client(tmp_path)
    payload = client.get(
        "/api/search",
        params={"q": "tokens", "repo": "org/repo", "jira": "ABC-1", "limit": 5},
    ).json()
    assert [h["id"] for h in payload["hits"]] == [DOC_ID]
    assert stub.search_calls == [("tokens", {"repo": "org/repo", "jira": "ABC-1"}, 5)]


def test_document_round_trip_and_missing(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    doc = client.get("/api/document", params={"id": DOC_ID}).json()
    assert doc["content"].startswith("body")
    assert doc["frontmatter"] == {"repo": "org/repo"}
    assert client.get("/api/document", params={"id": "nope"}).status_code == 404


def test_search_without_tanseki_url_is_503(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, tanseki_url="")
    assert client.get("/api/search", params={"q": "x"}).status_code == 503


def test_knowledge_aggregates_documents_into_captures(tmp_path: Path) -> None:
    client, stub = make_client(tmp_path)
    payload = client.get("/api/knowledge", params={"limit": 25}).json()

    assert stub.list_calls == [25]
    assert payload["total"] == 3
    assert payload["captured"] == 1
    assert payload["truncated"] is True
    assert payload["collection"] == Settings().tanseki_collection
    assert payload["generated_at"]
    capture = payload["captures"][0]
    assert capture["id"] == DOC_ID
    assert capture["repo"] == "org/repo"
    assert capture["author"] == "unknown"
    assert capture["category"] == "uncategorized"
    assert capture["capture_source"] == ""
    assert capture["structure"] == "anchored"
    assert capture["agent_authored"] is False
    assert capture["answer_truncated"] is False
    assert stub.closed is True


def test_knowledge_exposes_agent_provenance_and_hides_the_marker(tmp_path: Path) -> None:
    class AgentCaptureClient(StubTanseki):
        def get_document(self, doc_id: str, collection: str | None = None):
            return TansekiDocument(
                id=doc_id,
                path=f"{doc_id}.md",
                collection="kojutsu",
                content=(
                    "## Answer\n"
                    "<!-- kojutsu:agent:opencode -->\n\n"
                    "The cost is one missing audit row.\n"
                ),
                frontmatter={
                    "title": "Why is the marker hidden?",
                    "author": "opencode",
                    "answered_by_agent": "opencode",
                    "comment_author": "acme",
                    "capture_source": "webhook",
                    "tags": ["agent_authored"],
                    "category": "trade_off",
                    "repo": "org/repo",
                    "pr": "7",
                },
            )

    settings = Settings(
        tanseki_url="http://tanseki.test",
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
    )
    app = create_console_app(settings, client_factory=lambda _settings: AgentCaptureClient())
    capture = TestClient(app).get("/api/knowledge").json()["captures"][0]

    assert capture["answered_by_agent"] == "opencode"
    assert capture["comment_author"] == "acme"
    assert capture["capture_source"] == "webhook"
    assert capture["agent_authored"] is True
    assert capture["pr"] == 7
    assert "kojutsu:agent" not in capture["answer"]
    assert capture["answer"] == "The cost is one missing audit row."


def test_knowledge_without_tanseki_url_is_503(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, tanseki_url="")
    assert client.get("/api/knowledge").status_code == 503


def test_knowledge_rejects_limit_above_shared_maximum(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.get("/api/knowledge", params={"limit": MAX_LIST_RESULTS + 1}).status_code == 422


def test_knowledge_errors_do_not_expose_internal_details(tmp_path: Path) -> None:
    client, stub = make_client(tmp_path)
    stub.list_error = True
    response = client.get("/api/knowledge")

    assert response.status_code == 502
    assert "super-secret" not in response.text
    assert "internal.example" not in response.text


def test_knowledge_truncates_long_answers(tmp_path: Path) -> None:
    class LongAnswerClient(StubTanseki):
        def get_document(self, doc_id: str, collection: str | None = None):
            return TansekiDocument(
                id=doc_id,
                path=f"{doc_id}.md",
                collection="kojutsu",
                content=f"## Answer\n{'a' * (MAX_DASHBOARD_ANSWER_CHARS + 500)}",
            )

    settings = Settings(
        tanseki_url="http://tanseki.test",
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
    )
    app = create_console_app(settings, client_factory=lambda _settings: LongAnswerClient())
    capture = TestClient(app).get("/api/knowledge").json()["captures"][0]

    assert len(capture["answer"]) == MAX_DASHBOARD_ANSWER_CHARS
    assert capture["answer_truncated"] is True


def test_dashboard_serves_the_html_that_polls_the_endpoint() -> None:
    response = TestClient(create_console_app(Settings())).get("/dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/api/knowledge" in response.text
    assert "setInterval" in response.text


def test_dashboard_reports_a_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kojutsu.dev_console.DASHBOARD_PATH", tmp_path / "absent.html")
    response = TestClient(create_console_app(Settings())).get("/dashboard")

    assert response.status_code == 503
    assert "checkout" in response.text


def test_public_bind_requires_explicit_opt_in_and_token(tmp_path: Path) -> None:
    settings = Settings(tanseki_outbox_path=str(tmp_path / "outbox.db"))

    with pytest.raises(ValueError, match="--allow-insecure-bind"):
        create_console_app(settings, bind_host="0.0.0.0")
    with pytest.raises(ValueError, match="DEV_CONSOLE_TOKEN"):
        create_console_app(settings, bind_host="0.0.0.0", allow_insecure_bind=True)


def test_public_bind_requires_authentication(tmp_path: Path) -> None:
    settings = Settings(tanseki_outbox_path=str(tmp_path / "outbox.db"))
    app = create_console_app(
        settings,
        bind_host="0.0.0.0",
        allow_insecure_bind=True,
        access_token="secret-token",
    )
    client = TestClient(app)

    assert client.get("/").status_code == 401
    assert client.get("/", auth=("kojutsu", "wrong")).status_code == 401
    assert client.get("/", auth=("kojutsu", "secret-token")).status_code == 200


def test_index_escapes_quotes_for_html_attributes(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    page = client.get("/").text
    assert "/[&<>\"']/g" in page
    assert "'\"': '&quot;'" in page
    assert "&#39;" in page


def test_search_rejects_limit_above_shared_maximum(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.get("/api/search", params={"q": "x", "limit": 51}).status_code == 422


def test_tanseki_errors_do_not_expose_internal_details(tmp_path: Path) -> None:
    class SecretErrorClient(StubTanseki):
        def health(self) -> bool:
            raise TansekiError("token=super-secret")

        def search(self, *args, **kwargs):
            raise TansekiError("host=internal.example token=super-secret")

        def get_document(self, doc_id: str, collection: str | None = None):
            raise TansekiError("token=super-secret")

    settings = Settings(
        tanseki_url="http://tanseki.test",
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
    )
    app = create_console_app(settings, client_factory=lambda _settings: SecretErrorClient())
    client = TestClient(app)

    status = client.get("/api/status")
    search = client.get("/api/search", params={"q": "x"})
    document = client.get("/api/document", params={"id": DOC_ID})

    assert status.status_code == 200
    assert search.status_code == 502
    assert document.status_code == 502
    for response in (status, search, document):
        assert "super-secret" not in response.text
        assert "internal.example" not in response.text


def test_oversized_document_is_rejected_without_returning_content(tmp_path: Path) -> None:
    class OversizedClient(StubTanseki):
        def get_document(self, doc_id: str, collection: str | None = None):
            return TansekiDocument(
                id=doc_id,
                path=f"{doc_id}.md",
                collection="kojutsu",
                content="x" * (MAX_DOCUMENT_BYTES + 1),
            )

    settings = Settings(
        tanseki_url="http://tanseki.test",
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
    )
    app = create_console_app(settings, client_factory=lambda _settings: OversizedClient())
    response = TestClient(app).get("/api/document", params={"id": DOC_ID})

    assert response.status_code == 413
    assert "x" * 100 not in response.text
    assert "maximum size" in response.text


def test_document_response_limit_includes_metadata(tmp_path: Path) -> None:
    class LargeMetadataClient(StubTanseki):
        def get_document(self, doc_id: str, collection: str | None = None):
            return TansekiDocument(
                id=doc_id,
                path=f"{doc_id}.md",
                collection="kojutsu",
                content="x" * (MAX_DOCUMENT_BYTES - 1_000),
                frontmatter={"large": "y" * 10_000},
            )

    settings = Settings(
        tanseki_url="http://tanseki.test",
        tanseki_outbox_path=str(tmp_path / "outbox.db"),
    )
    app = create_console_app(settings, client_factory=lambda _settings: LargeMetadataClient())
    response = TestClient(app).get("/api/document", params={"id": DOC_ID})

    assert response.status_code == 413
    assert "maximum size" in response.text


class RationaleClient:
    """An Tanseki client serving one stored rationale and one ordinary capture.

    Both at once, because the failure this guards against is a rationale rendered
    as a capture: in a store holding only rationales it would look fine.
    """

    def __init__(self) -> None:
        self._docs = {
            "pilot/repo/pr-1/rationale/rationale-v1-abc": _doc(
                "pilot/repo/pr-1/rationale/rationale-v1-abc",
                "## Reason\n\narchitecture: used a lease token because a timestamp cannot "
                "tell held from expired.\n\n## Attribution\nDeclared by: kojutsu-pilot\n",
                {
                    "tags": ["rationale", "rationale_declared"],
                    "repo": "pilot/repo",
                    "pr": 1,
                    "declared_by": "kojutsu-pilot",
                    "declared_by_model": "opencode/model",
                    "rationale_source": "declared",
                    "rationale_revision": 1,
                    "capture_source": "asserted",
                },
            ),
            "pilot/repo/pr-1/q-1": _doc(
                "pilot/repo/pr-1/q-1",
                "## Answer\n\nThe cost is one missing audit row.",
                {"tags": ["agent_authored"], "category": "trade_off", "repo": "pilot/repo"},
            ),
            **_evaluation_docs(),
        }

    def count(self) -> int:
        return len(self._docs)

    def list_documents(self, limit: int = 100) -> list[str]:
        return list(self._docs)[:limit]

    def get_document(self, doc_id: str) -> Any:
        return self._docs.get(doc_id)


def _doc(doc_id: str, content: str, frontmatter: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=doc_id,
        content=content,
        frontmatter=frontmatter,
        updated_at="2026-03-04T12:00:00Z",
        revision=1,
    )


def test_a_rationale_is_served_as_a_claim_not_a_capture(tmp_path: Path) -> None:
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: RationaleClient())
    captures = TestClient(app).get("/api/knowledge").json()["captures"]

    rationale = next(c for c in captures if c["kind"] == "rationale")
    capture = next(c for c in captures if c["kind"] == "capture")

    assert rationale["capture_source"] == "asserted"
    assert rationale["author"] == "kojutsu-pilot"
    assert rationale["rationale_model"] == "opencode/model"
    assert rationale["rationale_revision"] == 1
    # The reason is shown; the unverifiable attribution is not appended to it.
    assert "lease token" in rationale["answer"]
    assert "Declared by" not in rationale["answer"]
    # And it is never given a question category, because it has none.
    assert rationale["category"] == "declared"

    # The ordinary capture is unaffected by the rationale handling.
    assert capture["category"] == "trade_off"
    assert capture["capture_source"] == ""


def test_a_rationale_never_borrows_a_capture_category(tmp_path: Path) -> None:
    """Regression: a rationale flattened as a capture showed as an empty
    'uncategorized' row, which is the one thing it must never look like."""
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: RationaleClient())
    rationale = next(
        c
        for c in TestClient(app).get("/api/knowledge").json()["captures"]
        if "/rationale/" in c["id"]
    )

    assert rationale["category"] != "uncategorized"
    assert rationale["answer"].strip()
    assert rationale["kind"] == "rationale"


# --- structure: the payload states it, for every record ----------------------
#
# The dashboard draws a badge and a count from this field, and a reader asking "was
# this pairing established or guessed?" must never have to infer the answer from a
# missing key. "Always present" is the whole requirement: a row whose structure is
# absent and a row whose structure is anchored have to be the same field for any
# filter on it to work.


class StructureClient:
    """Three records: inferred, unlabelled (pre-dating the axis), and unreadable."""

    def __init__(self) -> None:
        self._docs = {
            "org/repo/pr-1/guessed": _doc(
                "org/repo/pr-1/guessed",
                "## Answer\n\nProbably the audit row.",
                {
                    "category": "trade_off",
                    "repo": "org/repo",
                    "pr": 1,
                    "capture_source": "webhook",
                    "structure": "inferred",
                    "structure_inferred_by_model": "anthropic/claude-opus-5",
                },
            ),
            "org/repo/pr-1/older": _doc(
                "org/repo/pr-1/older",
                "## Answer\n\nIt was the audit row.",
                {"category": "edge_case", "repo": "org/repo", "pr": 1},
            ),
            "org/repo/pr-1/garbled": _doc(
                "org/repo/pr-1/garbled",
                "## Answer\n\nSomething.",
                {"category": "edge_case", "repo": "org/repo", "pr": 1, "structure": "guessed"},
            ),
        }

    def count(self) -> int:
        return len(self._docs)

    def list_documents(self, limit: int = 100) -> list[str]:
        return list(self._docs)[:limit]

    def get_document(self, doc_id: str) -> Any:
        return self._docs.get(doc_id)


def test_every_capture_in_the_payload_carries_a_structure(tmp_path: Path) -> None:
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: StructureClient())
    captures = TestClient(app).get("/api/knowledge").json()["captures"]

    assert captures, "the fixture must serve records for this to mean anything"
    for capture in captures:
        assert "structure" in capture, (
            "an absent key and an anchored value are indistinguishable to a filter, "
            f"so the field must be present on every row: {capture['id']}"
        )
        assert isinstance(capture["structure"], str) and capture["structure"]


def test_the_payload_states_an_inferred_pairing_and_resolves_the_default(
    tmp_path: Path,
) -> None:
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: StructureClient())
    by_id = {c["id"]: c for c in TestClient(app).get("/api/knowledge").json()["captures"]}

    assert by_id["org/repo/pr-1/guessed"]["structure"] == "inferred"
    # A record written before the axis existed carries no key and resolves to the
    # documented default, which is the truth: nothing could have inferred a pairing.
    assert by_id["org/repo/pr-1/older"]["structure"] == "anchored"
    # A value this code cannot read is reported as such rather than resolved to
    # anchored, which would be a claim about the store made by the reader.
    assert by_id["org/repo/pr-1/garbled"]["structure"] == "unknown"


def test_a_rationale_row_states_a_structure_too(tmp_path: Path) -> None:
    """The other kind of record is not exempt.

    A rationale has no pairing to have inferred, so it resolves to anchored -- but
    it resolves *explicitly*, because a badge that appears for one kind of row and
    silently not for another is how a reader starts guessing.
    """
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: RationaleClient())
    rationale = next(
        c
        for c in TestClient(app).get("/api/knowledge").json()["captures"]
        if c["kind"] == "rationale"
    )

    assert rationale["structure"] == "anchored"
    # And it is not hiding a reconstruction behind that value: the same payload
    # carries the rationale's own source.
    assert rationale["category"] == "declared"


# --- three kinds in one store -----------------------------------------------------
#
# Every kind is tested against a store holding only itself and an ordinary capture,
# which is enough to catch a record being flattened into the wrong shape. It is not
# enough to catch a *count*: three kinds each rendering as a plausible row is
# indistinguishable, on a dashboard, from two of them being folded into the third.
# The payload is what the KPI row counts, so the count is asserted here, on a store
# built from the real mappers rather than from hand-written frontmatter -- a fixture
# that spells the keys the writers do not is a fixture that agrees with itself.
#
# The documents are built with ``to_*_upsert_payload`` on purpose. A hand-written
# frontmatter block would let this test keep passing after a writer changed what it
# emits, which is the drift that made the rationale's revision and model go missing
# in the first place.

ANSWERED_AT = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
INFERRED_MODEL = "anthropic/claude-opus-5"


def _anchored_answer() -> dict[str, Any]:
    return to_upsert_payload(
        KnowledgeEntry(
            entry_id="answered",
            question_text="Why is the audit row written on release rather than on exit?",
            answer_text="Because a process killed mid-run writes no row, and that "
            "absence is the only signal a crash leaves behind.",
            category=QuestionCategory.TRADE_OFF,
            capture_source=CaptureSource.WEBHOOK,
            capture_delivery_id="49c14d2e-83f1-4852-b372-b1f11f4841fb",
            captured_at=ANSWERED_AT,
            answered_at=ANSWERED_AT,
            author="opencode",
            tags=["agent_authored"],
            metadata={"repo": "pilot/repo", "pr_number": 1, "comment_author": "davy"},
        )
    )


def _inferred_answer() -> dict[str, Any]:
    return to_upsert_payload(
        KnowledgeEntry(
            entry_id="guessed",
            question_text="Why is the ledger append-only?",
            answer_text="Probably so a row cannot be edited after the fact.",
            category=QuestionCategory.DESIGN_DECISION,
            capture_source=CaptureSource.WEBHOOK,
            capture_delivery_id="49c14d2e-83f1-4852-b372-b1f11f4841fb",
            captured_at=ANSWERED_AT,
            answered_at=ANSWERED_AT,
            author="opencode",
            tags=["agent_authored"],
            structure=RecordStructure.INFERRED,
            metadata={
                "repo": "pilot/repo",
                "pr_number": 1,
                STRUCTURE_INFERRED_BY: INFERRED_MODEL,
            },
        )
    )


def _rationale() -> dict[str, Any]:
    return to_rationale_upsert_payload(
        RationaleEntry(
            entry_id="rationale-v1-abc",
            repo="pilot/repo",
            pr_number=1,
            declared_by="kojutsu-pilot",
            declared_model="opencode/model",
            rationale_text="Used a lease token because a timestamp cannot tell held from expired.",
            declared_at=ANSWERED_AT,
        )
    )


def _clarification() -> dict[str, Any]:
    return to_clarification_upsert_payload(
        ClarificationEntry(
            entry_id="clarification-1",
            repo="pilot/repo",
            pr_number=1,
            statement="The narrow window is the accepted cost for v0.1, not an oversight.",
            author="davy",
            author_association="OWNER",
            github_comment_id=5,
            capture_source=CaptureSource.COLLECT,
            captured_at=ANSWERED_AT,
            declared_at=ANSWERED_AT,
        )
    )


class ThreeKindClient:
    """A store holding an anchored answer, an inferred one, a rationale and a
    clarification -- every kind the dashboard claims to distinguish, at once.

    Built by handing the console the documents the writers would have written, so
    the test fails if the read path stops understanding the store rather than only
    if it stops understanding itself.
    """

    def __init__(self) -> None:
        self._docs = {
            payload["id"]: _doc(payload["id"], payload["content"], payload["frontmatter"])
            for payload in (_anchored_answer(), _inferred_answer(), _rationale(), _clarification())
        }

    def add(self, payload: dict[str, Any]) -> None:
        """Put one more document in the store, as the writers would have written it."""
        self._docs[payload["id"]] = _doc(payload["id"], payload["content"], payload["frontmatter"])

    def count(self) -> int:
        return len(self._docs)

    def list_documents(self, limit: int = 100) -> list[str]:
        return list(self._docs)[:limit]

    def get_document(self, doc_id: str) -> Any:
        return self._docs.get(doc_id)


def _three_kinds(tmp_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: ThreeKindClient())
    payload = TestClient(app).get("/api/knowledge").json()
    return {c["id"]: c for c in payload["captures"]}, payload


def test_a_mixed_store_counts_each_kind_separately(tmp_path: Path) -> None:
    """No kind is folded into another, and the counts sum to the store.

    The sum is the part that catches a fold. Each kind rendering as a plausible row
    is not enough: on a dashboard the only evidence that a clarification was counted
    as a clarification is that the clarification total is one, and a total that
    happens to be right for the wrong reason is the failure.
    """
    by_id, payload = _three_kinds(tmp_path)
    kinds = Counter(c["kind"] for c in by_id.values())

    assert kinds == {"capture": 2, "rationale": 1, "clarification": 1}, (
        "a kind was folded into another; the dashboard's per-kind KPIs read these "
        f"counts and would report the store as something it is not: {by_id}"
    )
    assert sum(kinds.values()) == payload["total"] == len(by_id), (
        "the payload lost a record on the way out, so the kind totals cannot sum to the store"
    )
    for kind in ("capture", "rationale", "clarification"):
        assert kinds[kind], f"the {kind} kind is empty in a store that holds one"


def test_a_mixed_store_keeps_the_two_quoting_kinds_off_the_question_axis(
    tmp_path: Path,
) -> None:
    """A rationale and a clarification have no question, so neither gets a category.

    Both are otherwise the same row as an answer, and an empty "uncategorized" bucket
    on a question axis is a conclusion with nothing behind it -- the exact shape the
    rationale regression first hit and the clarification one would have hit next.
    """
    by_id, _ = _three_kinds(tmp_path)

    clarification = next(c for c in by_id.values() if c["kind"] == "clarification")
    rationale = next(c for c in by_id.values() if c["kind"] == "rationale")
    captures = [c for c in by_id.values() if c["kind"] == "capture"]

    assert clarification["category"] == "", "a clarification has no question to categorise"
    assert rationale["category"] != "uncategorized"
    # And the two that do have questions keep theirs, so the axis is not emptied.
    assert all(c["category"] != "uncategorized" for c in captures)
    assert len({c["category"] for c in captures}) == 2


def test_a_mixed_store_names_the_model_that_inferred_the_pairing(tmp_path: Path) -> None:
    """The row badge is completed from the payload, so the payload must carry it.

    "inferred" alone says a guess happened and not who to ask about it. The badge
    renders the model in the row's own text rather than behind a hover, so a field
    that stopped arriving here would quietly reduce the badge to a shrug.
    """
    by_id, _ = _three_kinds(tmp_path)
    inferred = next(c for c in by_id.values() if c["structure"] == "inferred")

    assert inferred["structure_inferred_by"] == INFERRED_MODEL
    # Present on every row, not only the inferred one, for the reason ``structure``
    # is: absent and empty have to be the same field for a badge to be decidable.
    for capture in by_id.values():
        assert "structure_inferred_by" in capture
        assert isinstance(capture["structure_inferred_by"], str)


def test_a_mixed_store_reports_one_unreadable_structure_apart_from_the_inferred(
    tmp_path: Path,
) -> None:
    """A value this build cannot read is not an inference, and is not merged with one.

    The dashboard counts and badges the two apart, because they are different claims:
    one is a writer saying it guessed, the other is a store this page cannot
    interpret. Merged, a rename would be reported as the readers' caution.

    The corrupt value is written onto a real mapped document rather than modelled.
    That is deliberate: no writer can produce one, because the write path only emits
    values the axis defines. A renamed or hand-edited value is something the *store*
    can come to hold, which is exactly why the read path has to survive it.
    """
    payload = _anchored_answer()
    client = ThreeKindClient()
    client.add(
        {
            **payload,
            "id": "pilot/repo/pr-1/garbled",
            "frontmatter": {**payload["frontmatter"], "structure": "guessed"},
        }
    )
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: client)
    by_id = {c["id"]: c for c in TestClient(app).get("/api/knowledge").json()["captures"]}

    assert by_id["pilot/repo/pr-1/garbled"]["structure"] == "unknown"
    assert by_id["pilot/repo/pr-1/garbled"]["structure_inferred_by"] == ""
    # The inferred record is unaffected: resolved to unknown for the garbled one,
    # which is the truth about it, and still inferred for the one that said so.
    # Resolving the garbled value to anchored would be a claim about the store made
    # by the reader rather than by its writer.
    assert by_id["pilot/repo/pr-1/guessed"]["structure"] == "inferred"
    assert by_id["pilot/repo/pr-1/guessed"]["structure_inferred_by"] == INFERRED_MODEL


def _evaluation_doc() -> Any:
    """A stored measurement, built through the real payload rather than imitated.

    A hand-written fixture here would test a document shape nobody writes, and the
    three failures this file already records were each one where the reader and the
    writer disagreed about a shape. The entry is built by the same ``to_*_upsert_payload``
    the harness stores it with, so a change to the storage shape breaks this rather
    than passing it.
    """
    entry = EvaluationEntry(
        entry_id="evaluation-v1-" + "a" * 64,
        repo="pilot/repo",
        pr_number=1,
        subject="opencode/model",
        target=EvaluationTarget.MODEL,
        measurement="thread-classifier-accuracy/marker-blind",
        result_text="recall 0.44-0.78, precision 0.667-0.875",
        scope="one thread, one repository, one reviewer, one model",
    )
    payload = to_evaluation_upsert_payload(entry)
    return _doc(payload["id"], payload["content"], payload["frontmatter"])


def _evaluation_docs() -> dict[str, Any]:
    return {_evaluation_doc().id: _evaluation_doc()}


def test_a_measurement_is_served_as_a_measurement_not_an_empty_capture(tmp_path: Path) -> None:
    """The third record kind that lost its shape on the way to the reader.

    Found by storing a measurement in the same collection the dashboard reads and
    looking at the page, which is the only way it shows up: every endpoint returned
    200 and the whole suite was green while the document rendered as an
    ``uncategorized`` row with no text — a precision figure sitting beside eleven
    real records with nothing on it to say what was measured.
    """
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: RationaleClient())
    captures = TestClient(app).get("/api/knowledge").json()["captures"]

    evaluation = next(c for c in captures if c["kind"] == "evaluation")

    assert evaluation["capture_source"] == "asserted"
    assert evaluation["evaluated_model"] == "opencode/model"
    assert evaluation["evaluation_target"] == "model"
    assert evaluation["category"] == "thread-classifier-accuracy/marker-blind"
    assert "recall 0.44-0.78" in evaluation["answer"]
    # The limits are not optional decoration on a row of figures: a measurement read
    # without its scope is a claim about the classifier rather than about one thread.
    assert "one thread" in evaluation["evaluation_scope"]
    # It has no pairing, so it must not claim a structure — `anchored` here would let
    # a filter asking which pairings were real count it.
    assert evaluation["structure"] == "unknown"
    assert evaluation["structure_stated"] is False


def test_a_measurement_never_becomes_uncategorized(tmp_path: Path) -> None:
    """The specific shape of the bug: an empty row on a question axis.

    Asserted over every kind rather than for the measurement alone, because the same
    flattening has now produced this for a rationale and a clarification too, and the
    invariant is about the reader.
    """
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "o.db")
    )
    captures = (
        TestClient(create_console_app(settings, client_factory=lambda _s: RationaleClient()))
        .get("/api/knowledge")
        .json()["captures"]
    )

    kinds = Counter(c["kind"] for c in captures)
    assert kinds["capture"] == 1, "the fixture should hold exactly one real capture"
    for record in captures:
        assert record["category"] != "uncategorized", record["kind"]
