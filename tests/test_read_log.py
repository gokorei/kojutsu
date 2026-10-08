"""Tests for the read log: what a read leaves behind, and what it must not.

A read log is a behavioural record of who asked what, minus the part kojutsu
cannot know: who asked. The tests below are mostly about the two ways that can
go wrong — storing something the store should not contain, and recording a read
in a way that reads as more than it is.

Assertions are on the **parsed** event, never on a rendered substring. The read
path draws a CSPRNG fence nonce per response, so a substring can match a
document id by coincidence; `docs/design-review/read-path.md` explains why, and
`tests/test_mcp_server.py` works the same way.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kojutsu.config import Settings
from kojutsu.core.read_log import (
    MAX_RECORDED_QUERY_CHARS,
    READ_EVENT_SCHEMA,
    ReadAccounting,
    ReadOutcome,
    new_event,
    record_read,
)
from kojutsu.integrations.tanseki import TansekiDocument, TansekiError, TansekiHit

READ_LOG_MODULE = Path(__file__).resolve().parents[1] / "src" / "kojutsu" / "core" / "read_log.py"
SERVER_SOURCE = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"

#: Pinned so a field cannot be added by a side effect of whatever a caller
#: happened to pass, and so a consumer can rely on every key being present.
EXPECTED_KEYS = {
    "schema",
    "recorded_at",
    "tool",
    "outcome",
    "query",
    "query_truncated",
    "caller_claims",
    "result_count",
    "excluded_count",
    "excluded_reasons",
    "truncated",
    "error_code",
}

#: No event key, and no key inside a caller's claims, may read as an identity.
#: A stdio server has one trust domain and no caller identity, so a field named
#: like one would be the `answered_by_model` problem: a self-assertion wearing the
#: grammar of a fact.
IDENTITY_WORDS = ("agent", "principal", "actor", "user", "identity", "session", "attribut", "who")

#: A body no stored record in this repository contains, so finding it in a log
#: line can only mean kojutsu wrote document content into its own telemetry.
BODY_MARKER = "the operator approved rotating the deploy key on friday"


def load_server():
    import importlib

    return importlib.reload(importlib.import_module("mcp_server.server"))


@pytest.fixture
def server():
    return load_server()


def load_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def log_path(tmp_path: Path) -> Path:
    return tmp_path / "read-log.jsonl"


def configure(
    server, fake, tmp_path: Path, *, allowed: str = "org/repo", tanseki: bool = True, **extra
):
    extra.setdefault("read_log_path", str(log_path(tmp_path)))
    server.configure(
        settings=Settings(
            tanseki_url="http://injected.test" if tanseki else "",
            github_webhook_allowed_repositories=allowed,
            **extra,
        ),
        client_factory=lambda _settings: fake,
    )


def make_docs() -> dict[str, TansekiDocument]:
    return {
        "d1": TansekiDocument(
            id="d1",
            path="d1.md",
            collection="kojutsu",
            content=BODY_MARKER,
            frontmatter={"repo": "org/repo", "pr": "1", "category": "design_decision"},
        ),
        "d2": TansekiDocument(
            id="d2",
            path="d2.md",
            collection="kojutsu",
            content="# Other",
            frontmatter={"repo": "other/x", "category": "trade_off"},
        ),
    }


class FakeTanseki:
    """A store that fails loudly if anything writes to it."""

    def __init__(self, docs: dict[str, TansekiDocument]) -> None:
        self.docs = docs
        self.calls: list[str] = []
        self.closed = False

    def _record(self, name: str) -> None:
        self.calls.append(name)

    def search(
        self, query: str, *, tags=None, frontmatter=None, limit: int = 10
    ) -> list[TansekiHit]:
        self._record("search")
        filters = frontmatter or {}
        return [
            TansekiHit(id=key, score=1.0)
            for key, document in self.docs.items()
            if all(str(document.frontmatter.get(k)) == str(v) for k, v in filters.items())
        ][:limit]

    def get_document(self, doc_id: str) -> TansekiDocument | None:
        self._record("get_document")
        return self.docs.get(doc_id)

    def get_documents(self, doc_ids: list[str]) -> list[TansekiDocument | None]:
        self._record("get_documents")
        return [self.docs.get(doc_id) for doc_id in doc_ids]

    def upsert_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the read path must never write to the knowledge store")

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        raise AssertionError("the read path must never write to the knowledge store")

    def close(self) -> None:
        self.closed = True


class UnavailableTanseki(FakeTanseki):
    def search(self, *_args, **_kwargs) -> list[TansekiHit]:
        self._record("search")
        raise TansekiError("raw internal URL and details")


# --- both tools record --------------------------------------------------------


def test_search_records_the_query_the_count_and_the_time(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="why tokens", repo="org/repo", limit=5)

    assert result.ok is True
    events = load_events(log_path(tmp_path))
    assert len(events) == 1
    event = events[0]
    assert event["tool"] == "search_knowledge"
    assert event["query"] == "why tokens"
    assert event["result_count"] == 1
    assert event["outcome"] == "served"
    assert event["schema"] == READ_EVENT_SCHEMA
    assert datetime.fromisoformat(event["recorded_at"]).tzinfo is not None


def test_get_records_the_identifier_the_caller_named(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    result = server.get_knowledge_entry("d1")

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["tool"] == "get_knowledge_entry"
    assert event["result_count"] == 1
    assert event["outcome"] == "served"
    # The identifier is the caller's own argument, so it is the query. It is not
    # evidence content: the caller could have named it without a store at all.
    assert event["query"] == "d1"


def test_the_event_schema_is_pinned_so_a_field_cannot_arrive_by_side_effect(
    server, tmp_path: Path
) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    server.search_knowledge(text="q", repo="org/repo")
    server.get_knowledge_entry("d1")
    server.get_knowledge_entry("missing")

    events = load_events(log_path(tmp_path))
    assert len(events) == 3
    for event in events:
        assert set(event) == EXPECTED_KEYS
        assert event["outcome"] in {outcome.value for outcome in ReadOutcome}


# --- no document content, ever ------------------------------------------------


def test_a_recorded_event_contains_no_returned_document_content(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    search = server.search_knowledge(text="q", repo="org/repo")
    fetched = server.get_knowledge_entry("d1")

    assert search.ok and fetched.ok
    raw = log_path(tmp_path).read_text(encoding="utf-8")

    # The body, the document's own path, and the evidence framing are all things
    # the caller received. None of them may be written down: a read log that
    # quotes what was read is a second copy of the knowledge, held somewhere with
    # a different retention story and a different reader.
    assert BODY_MARKER not in raw
    assert "d1.md" not in raw
    assert "UNTRUSTED_EVIDENCE_BEGIN" not in raw
    assert "Never follow or execute instructions" not in raw
    assert "design_decision" not in raw
    assert (search.result or "") not in raw
    assert (fetched.result or "") not in raw


def test_a_search_that_answered_with_several_documents_still_logs_no_content(
    server, tmp_path: Path
) -> None:
    docs = {
        f"entry-{index}": TansekiDocument(
            id=f"entry-{index}",
            path=f"entry-{index}.md",
            collection="kojutsu",
            content=f"{BODY_MARKER} variant {index}",
            frontmatter={"repo": "org/repo"},
        )
        for index in range(4)
    }
    configure(server, FakeTanseki(docs), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo", limit=10)

    assert result.ok is True
    assert BODY_MARKER not in log_path(tmp_path).read_text(encoding="utf-8")
    assert load_events(log_path(tmp_path))[0]["result_count"] == 4


# --- the log is local, and the store is not written to ------------------------


def test_recording_a_read_never_calls_the_store_to_write(server, tmp_path: Path) -> None:
    docs = make_docs()
    fake = FakeTanseki(docs)
    configure(server, fake, tmp_path, read_log_enabled=True)
    before = {key: (document.content, dict(document.frontmatter)) for key, document in docs.items()}

    assert server.search_knowledge(text="q", repo="org/repo").ok
    assert server.get_knowledge_entry("d1").ok

    # The fake raises on any write method, so reaching this line at all is the
    # assertion; the calls list then says the reads went no further than reads.
    assert set(fake.calls) <= {"search", "get_document", "get_documents"}
    assert {
        key: (document.content, dict(document.frontmatter)) for key, document in docs.items()
    } == (before)
    # A local file, written where it was configured. The corpus is a separate
    # service, so its contents cannot depend on who read what.
    assert log_path(tmp_path).is_file()
    assert log_path(tmp_path).parent == tmp_path


def test_the_read_log_module_cannot_reach_the_knowledge_store() -> None:
    """Structural, not behavioural: the module is not even given the means.

    A behavioural test proves today's code does not write. This proves the next
    change cannot write by accident, which is the property the read server's
    read-only tests are built on.
    """
    source = READ_LOG_MODULE.read_text(encoding="utf-8")

    assert "import kojutsu" not in source
    assert "from kojutsu" not in source
    for forbidden in (
        "TansekiClient",
        "TansekiKnowledgeSink",
        "knowledge_sink",
        "TansekiOutbox",
        "claim_rationale",
        "post_issue_comment",
        "capture_server",
    ):
        assert forbidden not in source, (
            f"the read log names {forbidden!r}. It is a local file for behavioural "
            "records; if it needs the knowledge store it has stopped being a log "
            "of reads and started being a write path."
        )


# --- a filter is a claim, not an identity -------------------------------------


def test_a_stated_filter_is_recorded_as_a_claim(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    # Authorisation folds case, so this read is allowed; the claim is what the
    # caller typed, which is not the same string as the allowlist entry.
    configure(
        server,
        fake,
        tmp_path,
        allowed="Org/Repo",
        read_log_enabled=True,
    )

    result = server.search_knowledge(
        text="q", repo="Org/Repo", jira_ticket_key="PROJ-7", min_independence="independent"
    )

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["caller_claims"] == {
        "repo": "Org/Repo",
        "jira_ticket_key": "PROJ-7",
        "min_independence": "independent",
        "limit": "10",
    }


def test_no_field_in_an_event_reads_as_a_caller_identity(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    server.search_knowledge(text="q", repo="org/repo")
    event = load_events(log_path(tmp_path))[0]

    for key in (*event, *event["caller_claims"]):
        assert not any(word in key.lower() for word in IDENTITY_WORDS), (
            f"{key!r} reads as an identity. This server is stdio with a single trust "
            "domain, so it cannot know who called; what a caller said about its read "
            "lives in caller_claims and a name in it is a self-assertion."
        )


def test_a_get_states_no_filter_rather_than_borrowing_the_stored_one(
    server, tmp_path: Path
) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    assert server.get_knowledge_entry("d1").ok

    # The document says which repository it belongs to, but the caller never
    # claimed a scope, and recording the stored value under a claim field would
    # be kojutsu's own statement wearing the caller's name.
    assert load_events(log_path(tmp_path))[0]["caller_claims"] == {}


# --- refusals are first class -------------------------------------------------


def test_a_denied_read_is_recorded_as_denied(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="other/x")

    assert result.code == "repository_not_authorized"
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "denied"
    assert event["error_code"] == "repository_not_authorized"
    assert event["result_count"] == 0
    assert fake.calls == []


def test_a_get_against_an_unauthorized_document_is_recorded_as_denied(
    server, tmp_path: Path
) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, allowed="org/repo", read_log_enabled=True)

    result = server.get_knowledge_entry("d2")

    assert result.code == "repository_not_authorized"
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "denied"
    assert event["result_count"] == 0


def test_a_rejected_read_is_recorded_as_rejected_not_as_a_denial(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q")

    assert result.code == "repository_required"
    event = load_events(log_path(tmp_path))[0]
    # A missing argument and a refused scope are both absences from the log when
    # they are not written, and they are different events: one is kojutsu
    # declining, the other is the caller sending something kojutsu would not
    # look at.
    assert event["outcome"] == "rejected"
    assert event["error_code"] == "repository_required"
    assert event["query"] == "q"


def test_an_unconfigured_store_is_recorded_as_its_own_outcome(server, tmp_path: Path) -> None:
    fake = FakeTanseki({})
    configure(server, fake, tmp_path, tanseki=False, read_log_enabled=True)

    search_result = server.search_knowledge(text="q", repo="org/repo")
    get_result = server.get_knowledge_entry("d1")

    assert search_result.code == "tanseki_not_configured"
    assert get_result.code == "tanseki_not_configured"
    events = load_events(log_path(tmp_path))
    assert [event["outcome"] for event in events] == ["unconfigured", "unconfigured"]
    assert [event["error_code"] for event in events] == ["tanseki_not_configured"] * 2


def test_a_store_failure_is_recorded_as_failed(server, tmp_path: Path) -> None:
    fake = UnavailableTanseki({})
    configure(server, fake, tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo")

    assert result.code == "tanseki_unavailable"
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "failed"
    assert event["error_code"] == "tanseki_unavailable"
    # The store's own error text stays in its own hands: it can name an internal
    # URL, and a behavioural record is the wrong place for that.
    assert "raw internal" not in log_path(tmp_path).read_text(encoding="utf-8")


def test_a_get_that_the_store_failed_is_recorded_as_failed(server, tmp_path: Path) -> None:
    class UnavailableGetTanseki(FakeTanseki):
        def get_document(self, doc_id: str) -> TansekiDocument | None:
            self._record("get_document")
            raise TansekiError("raw internal URL and details")

    configure(server, UnavailableGetTanseki(make_docs()), tmp_path, read_log_enabled=True)

    result = server.get_knowledge_entry("d1")

    assert result.code == "tanseki_unavailable"
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "failed"
    assert event["error_code"] == "tanseki_unavailable"
    assert event["result_count"] == 0


def test_a_result_kojutsu_refused_to_emit_is_recorded_as_failed(
    server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_document: TansekiDocument) -> str:
        raise server.EvidenceFramingError("refusing to emit a result")

    monkeypatch.setattr(server, "_render_tanseki_document", refuse)
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo")

    assert result.code == "evidence_framing_failed"
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "failed"
    # Nothing was emitted, so nothing was delivered. An event claiming one here
    # would report a read the caller never received.
    assert event["result_count"] == 0


def test_an_absent_entry_is_recorded_as_a_read_that_found_nothing(server, tmp_path: Path) -> None:
    fake = FakeTanseki(make_docs())
    configure(server, fake, tmp_path, read_log_enabled=True)

    result = server.get_knowledge_entry("missing")

    assert result.code == "not_found"
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "no_results"
    assert event["error_code"] == "not_found"


def test_a_search_that_matched_nothing_is_recorded_as_no_results(server, tmp_path: Path) -> None:
    configure(server, FakeTanseki({}), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo")

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "no_results"
    assert event["result_count"] == 0
    assert event["excluded_count"] == 0
    assert event["truncated"] is False
    assert event["error_code"] is None


# --- a bounded answer is visible as bounded in the log too --------------------


def test_a_budget_exhausted_search_is_not_recorded_as_an_empty_one(server, tmp_path: Path) -> None:
    """The event must not repeat the defect the answer was fixed for.

    ``search_knowledge`` refuses to report "no knowledge entries found" when the
    budget dropped everything, and an event that recorded that read as an empty
    one would put the same false claim in the operator's log.
    """
    oversized = TansekiDocument(
        id="d1",
        path="d1.md",
        collection="kojutsu",
        content="é" * (server.MAX_SEARCH_RESPONSE_CHARS // 2),
        frontmatter={"repo": "org/repo"},
    )
    configure(server, FakeTanseki({"d1": oversized}), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo")

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "budget_exhausted"
    assert event["result_count"] == 0
    assert event["excluded_count"] == 1
    assert event["excluded_reasons"] == ["response_budget"]
    assert event["truncated"] is True


def test_a_partial_answer_records_what_it_left_out(server, tmp_path: Path) -> None:
    docs = {
        f"entry-{index}": TansekiDocument(
            id=f"entry-{index}",
            path=f"entry-{index}.md",
            collection="kojutsu",
            content="z" * 30_000,
            frontmatter={"repo": "org/repo"},
        )
        for index in range(8)
    }
    configure(server, FakeTanseki(docs), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo", limit=50)

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "served"
    assert event["result_count"] > 0
    assert event["truncated"] is True
    assert event["excluded_count"] > 0
    assert event["excluded_reasons"] == ["response_budget"]


def test_exclusions_the_caller_or_the_store_caused_are_recorded_as_their_own_reasons(
    server, tmp_path: Path
) -> None:
    class LeakyTanseki(FakeTanseki):
        def search(self, *_args, **_kwargs) -> list[TansekiHit]:
            self._record("search")
            return [TansekiHit(id="d1", score=1.0), TansekiHit(id="d2", score=0.9)]

    configure(server, LeakyTanseki(make_docs()), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo")

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["result_count"] == 1
    assert event["excluded_count"] == 1
    assert event["excluded_reasons"] == ["cross_repository"]


def test_records_excluded_by_the_callers_own_threshold_are_named_in_the_log(
    server, tmp_path: Path
) -> None:
    docs = {
        key: TansekiDocument(
            id=document.id,
            path=document.path,
            collection=document.collection,
            content=document.content,
            frontmatter=document.frontmatter,
        )
        for key, document in make_docs().items()
    }
    docs["d1"] = TansekiDocument(
        id="d1",
        path="d1.md",
        collection="kojutsu",
        content=BODY_MARKER,
        frontmatter={"repo": "org/repo", "independence": "self_certified"},
    )
    configure(server, FakeTanseki(docs), tmp_path, read_log_enabled=True)

    result = server.search_knowledge(text="q", repo="org/repo", min_independence="independent")

    assert result.ok is True
    event = load_events(log_path(tmp_path))[0]
    assert event["outcome"] == "no_results"
    assert event["result_count"] == 0
    # Not a silent empty: a search that found something and showed none of it has
    # to say so, or it reads as a store that held nothing.
    assert event["excluded_count"] == 1
    assert event["excluded_reasons"] == ["below_min_independence"]


# --- the recorded query is bounded -------------------------------------------


def test_a_long_query_is_bounded_and_says_it_was_bounded(server, tmp_path: Path) -> None:
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)
    long_query = "needle " * (MAX_RECORDED_QUERY_CHARS // 2)

    assert server.search_knowledge(text=long_query, repo="org/repo").ok

    event = load_events(log_path(tmp_path))[0]
    assert event["query_truncated"] is True
    assert event["query"].startswith("needle ")
    assert len(event["query"]) < len(long_query)
    assert "truncated" in event["query"]


def test_a_short_query_is_recorded_whole_and_claims_no_truncation(server, tmp_path: Path) -> None:
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)

    assert server.search_knowledge(text="why tokens", repo="org/repo").ok

    event = load_events(log_path(tmp_path))[0]
    assert event["query"] == "why tokens"
    assert event["query_truncated"] is False


def test_a_search_that_sent_no_query_records_no_query(server, tmp_path: Path) -> None:
    """A repository-only search is a real read, and absence of a query is not a gap.

    Recording the empty string rather than a placeholder keeps the field honest:
    a consumer can tell "the caller asked for no text" from any other value, and
    cannot mistake a placeholder for something somebody searched for.
    """
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)

    assert server.search_knowledge(repo="org/repo").ok

    event = load_events(log_path(tmp_path))[0]
    assert event["query"] == ""
    assert event["query_truncated"] is False
    assert event["result_count"] == 1


# --- the log is not a file this process may write to --------------------------


def test_a_log_path_that_is_not_a_regular_file_is_refused(
    server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A read log holds what somebody asked for, so it does not follow a path out.

    A symlink at the log path, or a directory, is somewhere those questions would
    end up owned by whoever put it there. The read is served either way: refusing
    to log is not a reason to refuse to read.
    """
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_text("", encoding="utf-8")
    link = tmp_path / "read-log.jsonl"
    link.symlink_to(elsewhere)
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)

    with caplog.at_level(logging.WARNING, logger="kojutsu.core.read_log"):
        result = server.search_knowledge(text="q", repo="org/repo")

    assert result.ok is True
    assert elsewhere.read_text(encoding="utf-8") == ""
    assert not list(tmp_path.glob(".read-log.jsonl.*"))
    assert any("ReadLogError" in record.message for record in caplog.records)


def test_a_log_owned_by_someone_else_is_refused(
    server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)
    log_path(tmp_path).write_text("", encoding="utf-8")
    monkeypatch.setattr("os.geteuid", lambda: log_path(tmp_path).stat().st_uid + 1, raising=False)

    with caplog.at_level(logging.WARNING, logger="kojutsu.core.read_log"):
        result = server.search_knowledge(text="q", repo="org/repo")

    assert result.ok is True
    assert log_path(tmp_path).read_text(encoding="utf-8") == ""
    assert any("ReadLogError" in record.message for record in caplog.records)


def test_the_server_configures_logging_before_it_serves_stdio(
    server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retention announcement is only "not silent" if something is listening.

    stdout carries the protocol, so the log goes to stderr, and a stdio server
    that never installs a handler would drop the one line an operator needs when
    the log shrinks.
    """
    order: list[str] = []
    monkeypatch.setattr(
        "kojutsu.logging_config.configure_logging", lambda *a, **k: order.append("logging")
    )
    monkeypatch.setattr(server, "configure_logging", lambda *a, **k: order.append("logging"))
    monkeypatch.setattr(server.server, "run", lambda **kwargs: order.append("run"))

    server.main()

    assert order == ["logging", "run"]


def test_starting_the_server_says_where_the_log_is_and_what_it_keeps(
    server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound nobody can see is not a bound, and a disabled log nobody can see
    looks like a log with no history.

    The line is the place where the start date becomes visible, so it states what
    is kept rather than only that something is written.
    """
    said: list[str] = []
    monkeypatch.setattr(server, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(server.server, "run", lambda **kwargs: None)
    monkeypatch.setattr(server.logger, "info", lambda message, *args: said.append(message % args))
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)

    server.main()

    assert said and "recording read events" in said[0]
    assert str(log_path(tmp_path)) in said[0]
    assert "newest 10000 kept" in said[0]
    assert "older than 7 day(s) dropped" in said[0]


def test_starting_the_server_says_plainly_that_nothing_is_being_recorded(
    server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said: list[str] = []
    monkeypatch.setattr(server, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(server.server, "run", lambda **kwargs: None)
    monkeypatch.setattr(server.logger, "info", lambda message, *args: said.append(message % args))
    configure(server, FakeTanseki(make_docs()), tmp_path)

    server.main()

    assert said == [
        "read event recording is off (READ_LOG_ENABLED is false): reads leave "
        "no trace and past ones are not recoverable"
    ]


# --- off by default -----------------------------------------------------------


def test_recording_is_off_by_default_so_the_history_does_not_exist(server, tmp_path: Path) -> None:
    """A read log that is on by default is a decision nobody made.

    Enabling it fixes a start date. Reads that already happened are gone, and no
    setting can bring them back, so the default stays off and the log starts
    where somebody asked for it to start.
    """
    configure(server, FakeTanseki(make_docs()), tmp_path)

    assert Settings().read_log_enabled is False
    assert server.search_knowledge(text="q", repo="org/repo").ok
    assert server.get_knowledge_entry("d1").ok
    assert not log_path(tmp_path).exists()
    assert not (tmp_path / "read-log.jsonl").exists()


# --- a failed log write does not fail the read --------------------------------


def test_a_log_that_cannot_be_written_does_not_fail_the_read(
    server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    configure(
        server,
        FakeTanseki(make_docs()),
        tmp_path,
        read_log_enabled=True,
        read_log_path=str(blocker / "x.jsonl"),
    )

    with caplog.at_level(logging.WARNING, logger="kojutsu.core.read_log"):
        result = server.search_knowledge(text="q", repo="org/repo")

    assert result.ok is True
    assert BODY_MARKER in (result.result or "")
    assert any("Could not record" in record.message for record in caplog.records)


def test_a_query_holding_characters_that_would_change_a_line_are_recorded_exactly(
    server, tmp_path: Path
) -> None:
    """Two properties, one cause: the line is escaped rather than written raw.

    A right-to-left override inside a query could otherwise make a log line
    *display* as something it is not, and a lone surrogate is a character no
    UTF-8 file can hold at all. Escaping the whole payload handles both, and
    parsing it back returns what the caller actually sent.
    """
    query = "orphan \ud800 and ‮reversed"  # noqa: PLE2502 - fixture data for the escaping test
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)

    assert server.search_knowledge(text=query, repo="org/repo").ok

    raw = log_path(tmp_path).read_text(encoding="utf-8")
    assert raw.isascii()
    assert load_events(log_path(tmp_path))[0]["query"] == query


# --- retention ----------------------------------------------------------------


def test_entries_past_the_count_bound_are_removed_and_announced(
    server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    configure(
        server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True, read_log_max_entries=2
    )

    with caplog.at_level(logging.INFO, logger="kojutsu.core.read_log"):
        for index in range(4):
            assert server.search_knowledge(text=f"q{index}", repo="org/repo").ok

    events = load_events(log_path(tmp_path))
    assert [event["query"] for event in events] == ["q2", "q3"]
    # No silent deletion: a behavioural record that shrinks without a word leaves
    # an operator reasoning about a gap they cannot see.
    removals = [
        record for record in caplog.records if "read log retention removed" in record.message
    ]
    assert len(removals) == 2
    assert "removed 1 of 3" in removals[0].message


def test_entries_past_the_age_bound_are_removed_and_announced(
    server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)
    stale = datetime.now(UTC) - timedelta(days=30)
    for index in range(2):
        record_read(
            path=log_path(tmp_path),
            tool="search_knowledge",
            code="ok",
            query=f"old{index}",
            max_entries=100,
            max_age_days=30,
            now=stale,
        )
    # A wider bound first, so the stale entries survive to be dropped by the
    # narrower one rather than by the seeding call.
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)
    assert len(load_events(log_path(tmp_path))) == 2

    with caplog.at_level(logging.INFO, logger="kojutsu.core.read_log"):
        assert server.search_knowledge(text="fresh", repo="org/repo").ok

    events = load_events(log_path(tmp_path))
    assert [event["query"] for event in events] == ["fresh"]
    assert any("read log retention removed 2 of 3" in record.message for record in caplog.records)


def test_an_undatable_line_is_removed_rather_than_kept_for_ever(
    server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    configure(server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True)
    with log_path(tmp_path).open("a", encoding="utf-8") as stream:
        stream.write('{"schema": "kojutsu.read-event.v1"}\n')
        stream.write("\n")

    with caplog.at_level(logging.INFO, logger="kojutsu.core.read_log"):
        assert server.search_knowledge(text="q", repo="org/repo").ok

    events = load_events(log_path(tmp_path))
    assert [event["query"] for event in events] == ["q"]
    assert any("read log retention removed 2 of 3" in record.message for record in caplog.records)


def test_the_retention_bound_can_be_widened_but_never_removed() -> None:
    """A bound that can be switched off is a preference.

    An operator who wants a longer history sets a larger number. There is no
    setting that leaves a behavioural record growing for ever, because that is
    the outcome the bound exists to prevent and it is one environment variable
    away.
    """
    with pytest.raises(ValidationError):
        Settings(read_log_max_age_days=0)
    with pytest.raises(ValidationError):
        Settings(read_log_max_entries=0)

    with pytest.raises(ValueError):
        record_read(
            path="/dev/null", tool="search_knowledge", code="ok", max_entries=0, max_age_days=1
        )
    with pytest.raises(ValueError):
        record_read(
            path="/dev/null", tool="search_knowledge", code="ok", max_entries=1, max_age_days=0
        )


def test_retention_does_not_delete_the_file_itself(server, tmp_path: Path) -> None:
    configure(
        server, FakeTanseki(make_docs()), tmp_path, read_log_enabled=True, read_log_max_entries=1
    )

    for index in range(3):
        assert server.search_knowledge(text=f"q{index}", repo="org/repo").ok

    assert log_path(tmp_path).is_file()
    assert len(load_events(log_path(tmp_path))) == 1


# --- the read surface did not grow -------------------------------------------


def test_the_read_server_still_exposes_no_write_surface() -> None:
    """The two properties `tests/test_mcp_server.py` pins, from this side.

    This change adds imports to the read server, which is exactly the kind of
    edit that could quietly give it a write path, so the property is asserted
    here too rather than only in the file that motivated it.
    """
    import asyncio

    server = load_server()
    tools = asyncio.run(server.server.list_tools())

    assert {tool.name for tool in tools} == {
        "search_knowledge",
        "get_knowledge_entry",
        "traverse_knowledge",
        "list_knowledge",
    }
    assert all(tool.annotations is not None and tool.annotations.read_only_hint for tool in tools)

    source = SERVER_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "post_issue_comment",
        "rationale_comment_body_as_agent",
        "claim_rationale",
        "KnowledgeSink",
        "TansekiOutbox",
        "capture_server",
    ):
        assert forbidden not in source


# --- the accounting the read path fills in ------------------------------------


def test_an_unclassified_error_code_is_still_recorded_as_something() -> None:
    """A future code must leave a trace rather than fall out of the log.

    The cost of the fallback is that a new error code reads as a failure until
    someone classifies it, which is the cheaper mistake to make: a read that is
    recorded under the wrong outcome is fixable, and a read that left no record
    is not.
    """
    assert new_event(
        tool="search_knowledge", code="a_code_nobody_wrote_down", query="q"
    ).outcome == (ReadOutcome.FAILED)
    assert ReadAccounting().outcome == ReadOutcome.NO_RESULTS
