"""Tests for the minimal dev console (fake client, no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kojutsu import dev_console
from kojutsu.config import Settings
from kojutsu.core.answer_collector import RECORD_KINDS, RecordKind
from kojutsu.core.tanseki_mapping import (
    RECORD_KIND_KEY,
    build_census_content,
    build_census_frontmatter,
    build_content,
    build_frontmatter,
    build_question_content,
    build_question_frontmatter,
    build_rationale_content,
    build_rationale_frontmatter,
    census_document_id,
    document_id,
    question_document_id,
    rationale_document_id,
)
from kojutsu.dev_console import MAX_DASHBOARD_ANSWER_CHARS, MAX_DOCUMENT_BYTES, create_console_app
from kojutsu.integrations.tanseki import (
    MAX_LIST_RESULTS,
    TansekiDocument,
    TansekiError,
    TansekiHit,
)
from kojutsu.models import (
    CaptureSource,
    CensusRecord,
    KnowledgeEntry,
    QuestionCategory,
    QuestionRecord,
    RationaleEntry,
)

OBSERVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)

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
            frontmatter={"repo": "org/repo", RECORD_KIND_KEY: RecordKind.ANSWER.value},
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
    assert doc["frontmatter"] == {"repo": "org/repo", RECORD_KIND_KEY: RecordKind.ANSWER.value}
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
                    RECORD_KIND_KEY: RecordKind.ANSWER.value,
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


class CorpusClient:
    """An Tanseki client serving one document of every record kind at once.

    Built from the writers themselves — ``build_census_content``,
    ``build_question_content``, ``build_rationale_content`` — rather than from
    hand-written Markdown. That is the whole point: a fixture written by hand keeps
    passing after the writer changes its body, and the console keeps reading a
    section that no longer exists. Deriving the fixture from the writer means a
    change to a document's shape fails *here*, where the failure is about the
    console's reader, rather than in production where it is an empty row.

    Every kind together for the same reason the rationale fixture was: the failure
    this guards against is one kind absorbing another, and in a corpus holding only
    the absorbed kind there is nothing to absorb it into.
    """

    def __init__(self) -> None:
        self._docs: dict[str, Any] = {}

        def add(doc_id: str, content: str, frontmatter: dict[str, Any]) -> None:
            self._docs[doc_id] = _doc(doc_id, content, frontmatter)

        # A stated reason. Not a RecordKind, so recognised by its namespace.
        rationale = RationaleEntry(
            entry_id="rationale-v1-abc",
            repo="pilot/repo",
            pr_number=1,
            declared_by="kojutsu-pilot",
            declared_model="opencode/model",
            rationale_text=(
                "architecture: used a lease token because a timestamp cannot tell held from expired."
            ),
        )
        add(
            rationale_document_id(rationale),
            build_rationale_content(rationale),
            build_rationale_frontmatter(rationale),
        )

        # An observation. A RecordKind, and structurally identified besides.
        census = CensusRecord(
            entry_id="census-opened",
            repo="pilot/repo",
            pr_number=2,
            pr_url="https://example.test/pilot/repo/pull/2",
            action="opened",
            change_author_account="octocat",
            head_sha="abc123",
            observed_at=OBSERVED_AT,
            delivery_id="delivery-1",
        )
        add(
            census_document_id(census),
            build_census_content(census),
            build_census_frontmatter(census),
        )

        # A decision request. No record_kind key at all, so identified by namespace.
        question = QuestionRecord(
            question_id="q-7",
            repo="pilot/repo",
            pr_number=3,
            pr_url="https://example.test/pilot/repo/pull/3",
            question_text="Should the outbox row be deleted or marked?",
            status="answered",
            category="data_model",
            question_author="octocat",
            attempts=2,
            created_at=OBSERVED_AT,
            updated_at=OBSERVED_AT,
        )
        add(
            question_document_id(question),
            build_question_content(question),
            build_question_frontmatter(question),
        )

        # Every capture kind, one document each, all with the same answer text and
        # category. A check run reads on its own like a review verdict, which is
        # exactly why it is in this corpus: it is the case that fails silently.
        for kind in RecordKind:
            if kind is RecordKind.CENSUS:
                # Written by its own writer below, and it is not a capture.
                continue
            entry = KnowledgeEntry(
                entry_id=f"{kind.value}-1",
                question_text=f"Why {kind.value}?",
                answer_text="Because of the change.",
                category=QuestionCategory.TRADE_OFF,
                capture_source=CaptureSource.WEBHOOK,
                captured_at=OBSERVED_AT,
                capture_delivery_id=f"delivery-{kind.value}",
                answered_at=OBSERVED_AT,
                metadata={
                    "repo": "pilot/repo",
                    "pr_number": 10,
                    RECORD_KIND_KEY: kind.value,
                    "answered_by_agent": "opencode",
                    "comment_author": "acme",
                },
            )
            add(document_id(entry), build_content(entry), build_frontmatter(entry))

        # A backfilled answer: the same shape, a weaker guarantee. It still has to
        # name what it read, which is what makes it a record rather than a guess.
        backfilled = KnowledgeEntry(
            entry_id="backfilled-1",
            question_text="Why now?",
            answer_text="Reconstructed from history.",
            category=QuestionCategory.DESIGN_DECISION,
            capture_source=CaptureSource.BACKFILLED,
            captured_at=OBSERVED_AT,
            answered_at=OBSERVED_AT,
            metadata={
                "repo": "pilot/repo",
                "pr_number": 11,
                RECORD_KIND_KEY: RecordKind.ANSWER.value,
                "answered_by_agent": "opencode",
                "github_comment_id": 4242,
            },
        )
        add(document_id(backfilled), build_content(backfilled), build_frontmatter(backfilled))

        # A document written before the store recorded kinds at all.
        add(
            "pilot/repo/pr-12/legacy-1",
            "## Answer\n\nWritten before kinds were recorded.\n",
            {
                "title": "An older answer",
                "author": "opencode",
                "capture_source": CaptureSource.COLLECT.value,
                "category": "trade_off",
                "repo": "pilot/repo",
                "pr": "12",
                "answered_at": OBSERVED_AT.isoformat(),
            },
        )
        # A kind the store's vocabulary has never contained. Nothing can vouch that
        # this will not happen; that it is visible when it does is the point.
        add(
            "pilot/repo/pr-13/from-the-future-1",
            "## Answer\n\nA kind nobody here has decided how to read.\n",
            {
                "title": "An attestation",
                "record_kind": "verification_attestation",
                "author": "opencode",
                "capture_source": CaptureSource.WEBHOOK.value,
                "repo": "pilot/repo",
                "pr": "13",
                "answered_at": OBSERVED_AT.isoformat(),
            },
        )

    def count(self) -> int:
        return len(self._docs)

    def list_documents(self, limit: int = 100) -> list[str]:
        return list(self._docs)[:limit]

    def get_document(self, doc_id: str) -> Any:
        return self._docs.get(doc_id)


# Kept as its own name because the two tests that use it are about a rationale
# specifically; it is the whole corpus either way.
RationaleClient = CorpusClient


def _doc(doc_id: str, content: str, frontmatter: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=doc_id,
        content=content,
        frontmatter=frontmatter,
        updated_at="2026-03-04T12:00:00Z",
        revision=1,
    )


def _records(app: Any) -> dict[str, dict[str, Any]]:
    """The knowledge payload keyed by id, so a test can name the record it means."""
    payload = TestClient(app).get("/api/knowledge").json()
    return {r["id"]: r for r in payload["captures"]}


def _corpus(tmp_path: Path) -> dict[str, dict[str, Any]]:
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    return _records(create_console_app(settings, client_factory=lambda _s: CorpusClient()))


def test_a_rationale_is_served_as_a_claim_not_a_capture(tmp_path: Path) -> None:
    records = _corpus(tmp_path)
    rationale = records["pilot/repo/pr-1/rationale/rationale-v1-abc"]

    assert rationale["kind"] == "rationale"
    assert rationale["capture_source"] == "asserted"
    assert rationale["author"] == "kojutsu-pilot"
    assert rationale["rationale_model"] == "opencode/model"
    assert rationale["rationale_revision"] == 1
    # The reason is shown; the unverifiable attribution is not appended to it.
    assert "lease token" in rationale["answer"]
    assert "Declared by" not in rationale["answer"]
    # And it is never given a question category, because it has none.
    assert rationale["category"] == "declared"

    # An ordinary capture in the same corpus is unaffected by the handling above.
    capture = records["pilot/repo/pr-10/answer-1"]
    assert capture["kind"] == RecordKind.ANSWER.value
    assert capture["category"] == "trade_off"
    assert capture["capture_source"] == "webhook"


def test_a_rationale_never_borrows_a_capture_category(tmp_path: Path) -> None:
    """Regression: a rationale flattened as a capture showed as an empty
    'uncategorized' row, which is the one thing it must never look like."""
    records = _corpus(tmp_path)
    rationale = next(r for r in records.values() if "/rationale/" in r["id"])

    assert rationale["category"] != "uncategorized"
    assert rationale["answer"].strip()
    assert rationale["kind"] == "rationale"


def test_every_record_kind_is_reported_apart(tmp_path: Path) -> None:
    """The closed vocabulary, one kind per record, and nothing folded together.

    Requirement 1 made concrete: ``kind`` distinguishes the kinds rather than
    splitting them into "capture or not". Driven by the writers' own vocabulary
    rather than a list written here, so a kind added to the store and not to this
    test fails instead of passing unnoticed — the check-run-is-a-review failure is
    the one that would otherwise be invisible.
    """
    records = _corpus(tmp_path)

    for kind in RecordKind:
        if kind is RecordKind.CENSUS:
            continue
        record = records[f"pilot/repo/pr-10/{kind.value}-1"]
        assert record["kind"] == kind.value, kind
        # Same text, same category, same author — the only thing that separates
        # them is the kind, which is the whole claim.
        assert record["answer"] == "Because of the change."

    assert records["pilot/repo/pr-1/rationale/rationale-v1-abc"]["kind"] == "rationale"
    assert records["pilot/repo/pr-2/census/opened"]["kind"] == RecordKind.CENSUS.value
    assert records["pilot/repo/pr-3/question/q-7"]["kind"] == "question"

    # Every record carries exactly one kind, and the kinds that appear are the whole
    # vocabulary: every RecordKind, the two that are separate models, and the two
    # ways a record fails to be classified. Nothing is missing, and nothing is
    # counted as a kind it is not.
    kinds = [r["kind"] for r in records.values()]
    # The closed set the console can report: every RecordKind, the four that are
    # separate models and name themselves by namespace or tag rather than by key,
    # and the two ways classification fails.
    reported = set(RECORD_KINDS) | {
        # ``capture`` is the console's word for a record no writer labelled but whose
        # own contents settle it. It is not a RecordKind because nothing writes it --
        # which is the whole point of it.
        "capture",
        "rationale",
        "question",
        "census",
        "evaluation",
        "clarification",
        "undeclared",
        "unrecognised",
    }
    assert set(kinds) <= reported, (
        f"a record was reported as a kind outside the closed vocabulary: "
        f"{sorted(set(kinds) - reported)}"
    )
    # This corpus exists to exercise the interesting ones, so assert it still does.
    # ``undeclared`` is covered directly above rather than by this corpus, because a
    # keyless record that also states a category is a capture on the evidence of the
    # document and belongs in the capture column.
    for expected in ("capture", "answer", "rationale", "question", "census", "unrecognised"):
        assert expected in kinds, f"the corpus no longer covers {expected}"
    assert len(kinds) == len(records)


def test_a_census_record_is_never_a_capture(tmp_path: Path) -> None:
    """It reached the store through a delivery and captured nothing.

    Requirement 3, at the boundary the record exists to protect. Every field a
    capture reader would look for is asserted absent rather than empty: an absent
    key is what stops a renderer falling back to one.
    """
    census = _corpus(tmp_path)["pilot/repo/pr-2/census/opened"]

    assert census["kind"] == RecordKind.CENSUS.value
    # Nobody answered anything, so nothing may name an answerer. The ``author``
    # field carries the login that opened the change, which is a fact about the
    # change rather than a claim that this person asserted the observation.
    assert census["answered_by_agent"] == ""
    assert census["comment_author"] == ""
    # And no category, because a category would be a judgement about the change.
    assert census["category"] == ""
    # The observation itself is intact and checkable.
    assert census["observation_delivery_id"] == "delivery-1"
    assert census["observation_action"] == "opened"
    assert census["answered_at"] == OBSERVED_AT.isoformat()
    assert "produced no knowledge record" in census["answer"]


def test_a_question_is_never_a_capture_that_produced_nothing(tmp_path: Path) -> None:
    """Requirement 4. It is a request, so it carries no answer and no source.

    Both absences are mechanisms in the writer, and flattening the record would
    overwrite them with defaults: a question that acquired a ``capture_source``
    would pass an evidence-only query it was written specifically to fail.
    """
    question = _corpus(tmp_path)["pilot/repo/pr-3/question/q-7"]

    assert question["kind"] == "question"
    assert question["capture_source"] == ""
    assert question["structure"] in {"unknown", "anchored"}
    assert "independence" not in question
    assert question["question_id"] == "q-7"
    assert question["question_status"] == "answered"
    assert "outbox row" in question["answer"]
    # And no answerer: nobody answered anything here, so nothing names one.
    assert question["answered_by_agent"] == ""
    assert question["capture_source"] == ""


def test_a_backfilled_capture_is_distinguishable_from_a_witnessed_one(tmp_path: Path) -> None:
    """Requirement 7. Same shape, weaker guarantee, and a reader has to be able to tell.

    ``reconstructed`` is a flag rather than a comparison the dashboard makes,
    because the dashboard would otherwise have to know that ``backfilled`` is the
    source with the different meaning.
    """
    records = _corpus(tmp_path)

    backfilled = records["pilot/repo/pr-11/backfilled-1"]
    witnessed = records["pilot/repo/pr-10/answer-1"]
    assert backfilled["capture_source"] == CaptureSource.BACKFILLED.value
    assert backfilled["backfilled"] is True
    assert witnessed["capture_source"] == CaptureSource.WEBHOOK.value
    assert witnessed["backfilled"] is False


def test_a_keyless_record_with_a_category_is_a_capture_on_evidence(tmp_path: Path) -> None:
    """Absence of ``record_kind`` is resolved from what the document says, not guessed.

    The store began writing ``record_kind`` partway through its life, so older
    documents carry no key. This one sits in the flat answer namespace and states a
    retrospective question category -- and only something that answers a question can
    have one, so it is a capture by what the document states rather than by anything
    this code supplied. Reporting it as ``undeclared`` would understate the corpus;
    reporting it as an answer would be inventing a pairing.
    """
    legacy = _corpus(tmp_path)["pilot/repo/pr-12/legacy-1"]

    assert legacy["kind"] == "capture"
    assert legacy["declared_kind"] == ""
    assert legacy["backfilled"] is False


def test_a_keyless_record_with_nothing_to_go_on_is_undeclared() -> None:
    """No key, no category, no namespace: the document does not say, and neither do we.

    This is the case defaulting to ``capture`` would hide. A record in an unknown
    namespace carrying no category could be anything, and counting it as a capture is
    how a store starts reporting itself as something it is not.
    """
    assert dev_console.record_kind_of({}, [], "some/repo/pr-9/something-else") == "undeclared"


def test_a_kind_this_console_has_never_heard_of_is_reported_unaccounted(tmp_path: Path) -> None:
    """Requirement 6, from the outside: not folded into a capture, and named.

    ``declared_kind`` keeps the value the document actually wrote, because that is
    what somebody has to go and handle. Reporting the record as itself would be a
    claim that this console understands it, and it does not.
    """
    future = _corpus(tmp_path)["pilot/repo/pr-13/from-the-future-1"]

    assert future["kind"] == "unrecognised"
    assert future["declared_kind"] == "verification_attestation"


def test_a_kind_this_build_cannot_resolve_is_reported_rather_than_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a kind to the writer must not break the page it is rendered on.

    The tempting alternative to a fallback is a ``KeyError``, which is louder. It
    is also worse: one unhandled kind takes down every other record in the store,
    so the failure that hides the most is the cheapest one to ship. The record is
    counted as unaccounted, and the vocabulary it came from is named.
    """
    monkeypatch.setattr(
        dev_console, "RECORD_KINDS", frozenset(RECORD_KINDS - {RecordKind.REVIEW_VERDICT.value})
    )
    settings = Settings(
        tanseki_url="http://tanseki.test", tanseki_outbox_path=str(tmp_path / "outbox.db")
    )
    app = create_console_app(settings, client_factory=lambda _s: CorpusClient())

    records = _records(app)

    # The endpoint answered, and every other record is still there with its kind.
    assert records["pilot/repo/pr-10/answer-1"]["kind"] == RecordKind.ANSWER.value
    assert records["pilot/repo/pr-2/census/opened"]["kind"] == RecordKind.CENSUS.value
    # The one whose reader was removed is the only thing that changed.
    verdict = records["pilot/repo/pr-10/review_verdict-1"]
    assert verdict["kind"] == "unrecognised"
    assert verdict["declared_kind"] == RecordKind.REVIEW_VERDICT.value


def test_every_kind_the_store_can_write_resolves_to_itself(tmp_path: Path) -> None:
    """The vocabulary and the reader are kept in step, and this is where.

    Without it the previous test is the only thing standing between a new record
    kind and a corpus that reports it as unaccounted -- which is correct, but is
    meant to be noticed once rather than discovered once.
    """
    records = _corpus(tmp_path)

    for kind in RECORD_KINDS:
        if kind == RecordKind.CENSUS.value:
            continue  # written by its own writer, and not a capture
        assert records[f"pilot/repo/pr-10/{kind}-1"]["kind"] == kind, (
            f"{kind} resolves to a different kind than the one the writer declared; "
            "the classification has fallen out of step with the vocabulary and "
            "nothing else would notice"
        )


def test_every_record_carries_one_time_across_kinds(tmp_path: Path) -> None:
    """Observations and questions are dated, so they reach the timeline.

    A per-kind time field would leave them undated, which drops them off the
    chart without saying so — the same silent absorption, in a chart rather than
    in a total.
    """
    records = _corpus(tmp_path)

    assert records["pilot/repo/pr-2/census/opened"]["answered_at"] == OBSERVED_AT.isoformat()
    assert records["pilot/repo/pr-3/question/q-7"]["answered_at"] == OBSERVED_AT.isoformat()
    assert records["pilot/repo/pr-1/rationale/rationale-v1-abc"]["answered_at"]
    assert all(r["answered_at"] for r in records.values())
