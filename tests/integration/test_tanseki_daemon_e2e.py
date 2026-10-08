"""Cross-seam end-to-end tests against a real tanseki-daemon.

``test_tanseki_e2e.py`` runs the same round trip against an in-process fake of the
Tanseki ``/v1`` API. That fake exercises the client, mapping, outbox, sink and relay
faithfully, and it proves nothing whatever about the store: it answers from a dict
the test itself filled in, so it cannot derive an edge from frontmatter, cannot
refuse an over-cap listing, and cannot distinguish a filter the server applied
from a filter nobody applied at all. Those are exactly the claims in the
"Guarantees Kojutsu relies on" section of ``docs/tanseki-seam.md`` that live on
the far side of the process boundary, so they are asserted here and only here.

Skipped unless a daemon answers ``/v1/health`` at ``TANSEKI_URL``. The probe runs at
collection time, before ``tests/conftest.py`` isolates the environment and closes
the sockets, and the skip reason names what was missing and how to start a
daemon.
"""

from __future__ import annotations

import os
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import yaml

from kojutsu.core.knowledge_sink import TansekiKnowledgeSink, to_payload
from kojutsu.core.outbox import TansekiOutbox, relay
from kojutsu.core.tanseki_mapping import (
    FILES_KEY,
    build_frontmatter,
    document_id,
    document_path,
)
from kojutsu.integrations.tanseki import TansekiClient
from kojutsu.models import KnowledgeEntry, QuestionCategory

#: The daemon under test. Read at collection time because ``env_isolate`` deletes
#: ``TANSEKI_URL`` before any fixture runs, and a probe that ran afterwards would
#: always report the daemon absent.
_TANSEKI_URL = os.environ.get("TANSEKI_URL", "").strip().rstrip("/")
_TANSEKI_API_KEY = os.environ.get("TANSEKI_API_KEY", "").strip()
_TANSEKI_COLLECTION = os.environ.get("TANSEKI_COLLECTION", "").strip()

#: Proxies are disabled for the probe because a proxy in the runner's environment
#: would answer for a daemon bound to loopback, and the skip would then describe
#: the proxy's opinion rather than this machine's.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

#: Short, because a daemon that is up answers immediately and a skip nobody waits
#: for is a skip everybody stops reading.
_PROBE_TIMEOUT_SECONDS = 2.0

#: How long an eventually-consistent store is given to derive an edge or index a
#: document. Bounded and re-checked rather than slept once, so a store that never
#: derives the edge fails in seconds with the fact rather than after a fixed wait.
_SETTLE_SECONDS = 15.0

#: Namespace prefix for the collections these tests write to. Read at collection
#: time for the same reason as ``_TANSEKI_URL``, and used only as a prefix: several
#: tests below assert an exact count, so a collection shared between them would make
#: that count a function of test ordering rather than of the seam. The per-test
#: suffix is what keeps them independent.
_COLLECTION_PREFIX = _TANSEKI_COLLECTION or f"kojutsu-e2e-{uuid.uuid4().hex[:8]}"

#: Captured at import, before ``env_isolate`` replaces them with functions that
#: refuse every connection. Restoring them is the only thing this module does that
#: the rest of the suite must not: here the socket is the subject.
_REAL_SOCKET_CONNECT = socket.socket.connect
_REAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection


def _probe_daemon() -> tuple[bool, str]:
    """Ask the daemon whether it is there, and say plainly why not when it is not."""
    if not _TANSEKI_URL:
        return False, (
            "TANSEKI_URL is not set, so there is no tanseki-daemon to run against. Start one "
            "and export TANSEKI_URL -- docs/tanseki-quickstart.md has the command, "
            "scripts/dev-e2e.sh starts a throwaway one, and the tanseki-e2e job in CI "
            "starts its own."
        )
    try:
        with _OPENER.open(f"{_TANSEKI_URL}/v1/health", timeout=_PROBE_TIMEOUT_SECONDS) as response:
            if 200 <= response.status < 300:
                return True, f"tanseki-daemon is reachable at {_TANSEKI_URL}"
            return False, (
                f"{_TANSEKI_URL}/v1/health answered {response.status}, so whatever is "
                f"listening there is not a healthy daemon. See docs/tanseki-quickstart.md."
            )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, (
            f"no tanseki-daemon answered {_TANSEKI_URL}/v1/health ({exc}). Start one and export "
            f"TANSEKI_URL -- docs/tanseki-quickstart.md, or scripts/dev-e2e.sh for a throwaway "
            f"one; the tanseki-e2e job in CI starts its own."
        )


_REACHABLE, _SKIP_REASON = _probe_daemon()

#: The gate for this whole module. Skipping rather than failing is the right
#: default: the daemon is another project's build, and a developer with no checkout
#: of it should still get a green suite. It is a skip that says what was missing,
#: so it cannot be mistaken for a pass.
requires_daemon = pytest.mark.skipif(not _REACHABLE, reason=_SKIP_REASON)


def _assert_eventually(
    predicate: Callable[[], bool],
    description: str,
    timeout: float = _SETTLE_SECONDS,
) -> None:
    """Assert a predicate becomes true, re-checked until a deadline.

    Edges and the search index are the store's to rebuild, and a store that does it
    asynchronously is not slow, it is correct. Re-checking until a bound is what
    separates "not yet" from "never", and the failure says which of the two it was.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{description} did not hold within {timeout:.0f}s against a live daemon "
                f"at {_TANSEKI_URL}"
            )
        time.sleep(0.25)


@pytest.fixture
def daemon(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    env_isolate: None,
) -> TansekiClient:
    """A client on its own collection, with the suite's socket denier lifted.

    Depending on ``env_isolate`` rather than hoping autouse ordering is favourable:
    the denials have to be in place before this reopens the socket, or the daemon
    the suite deliberately isolates from is one the daemon tests never reach.

    One collection per test, not per session, because several tests here assert an
    exact count and a shared collection would make that count a function of test
    ordering. ``TANSEKI_COLLECTION`` is honoured as the prefix only, so naming a
    namespace cannot quietly merge the tests into one.
    """
    monkeypatch.setattr(socket.socket, "connect", _REAL_SOCKET_CONNECT)
    monkeypatch.setattr(socket.socket, "connect_ex", _REAL_SOCKET_CONNECT_EX)
    monkeypatch.setattr(socket, "create_connection", _REAL_CREATE_CONNECTION)
    suffix = f"{request.node.name}-{uuid.uuid4().hex[:6]}"
    return TansekiClient(
        _TANSEKI_URL,
        api_key=_TANSEKI_API_KEY,
        collection=f"{_COLLECTION_PREFIX}-{suffix}",
        timeout=10.0,
    )


def make_entry(
    entry_id: str,
    *,
    repo: str = "org/repo",
    pr_number: int = 42,
    jira_ticket_key: str | None = None,
    tags: list[str] | None = None,
    files: list[str] | None = None,
    answer: str = "Because X.",
) -> KnowledgeEntry:
    metadata: dict[str, object] = {"repo": repo, "pr_number": pr_number}
    if jira_ticket_key is not None:
        metadata["jira_ticket_key"] = jira_ticket_key
    if files is not None:
        metadata[FILES_KEY] = files
    return KnowledgeEntry(
        entry_id=entry_id,
        question_text=f"Why {entry_id}?",
        answer_text=answer,
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        tags=tags if tags is not None else ["design"],
        metadata=metadata,
    )


def store(client: TansekiClient, outbox_path: Path, *entries: KnowledgeEntry) -> None:
    """Write entries through the real sink and outbox, not straight to the client.

    The delivery path is part of what crosses the seam, so a test that bypassed it
    would check the HTTP calls and not the system.
    """
    with TansekiOutbox(outbox_path) as outbox:
        sink = TansekiKnowledgeSink(client, outbox)
        for entry in entries:
            outcome = sink.store(entry)
            assert outcome is not None and outcome.delivered, outcome


def _stored_frontmatter(content: str) -> dict[str, object]:
    """The frontmatter block as the store hands it back, parsed from the content.

    Parsed rather than read off the JSON ``frontmatter`` field, because the two are
    separate claims and only the content proves the block survived the round trip
    as part of the document rather than as a side-channel that happened to agree.
    """
    assert content.startswith("---\n"), "the stored document has no frontmatter block"
    block = content.split("---\n", 2)[1]
    parsed = yaml.safe_load(block)
    assert isinstance(parsed, dict)
    return parsed


@requires_daemon
def test_capture_round_trip_preserves_the_canonical_document(
    daemon: TansekiClient, tmp_path: Path
) -> None:
    """The record that comes back is the record that was sent."""
    entry = make_entry("round-trip")
    store(daemon, tmp_path / "outbox.db", entry)

    doc_id = document_id(entry)
    doc = daemon.get_document(doc_id)
    assert doc is not None, f"{doc_id} was written but is not readable back"
    assert doc.path == document_path(entry)
    assert doc.revision

    # Every key the mapper emitted, compared rather than spot-checked: a store that
    # dropped one would still return a document that looks right in a listing.
    stored = _stored_frontmatter(doc.content)
    expected = build_frontmatter(entry)
    # ``updated_at`` is compared as an instant, not as text: the store
    # canonicalises the field's text form by contract (``+00:00`` sent,
    # ``Z`` stored -- Tanseki decision AH454A1G, commit 0426e39), so a string
    # comparison would fail on the spelling while the instant survived. Parsing
    # both sides means an offset silently applied as a shift still fails.
    assert datetime.fromisoformat(str(stored["updated_at"])) == datetime.fromisoformat(
        str(expected["updated_at"])
    )
    assert {k: v for k, v in stored.items() if k != "updated_at"} == {
        k: v for k, v in expected.items() if k != "updated_at"
    }
    assert entry.answer_text in doc.content


@requires_daemon
def test_listing_total_and_ids_describe_the_collection(
    daemon: TansekiClient, tmp_path: Path
) -> None:
    """``GET /v1/documents`` answers the total and enumerates the collection.

    The bounded listing is what ``kojutsu status`` and the knowledge dashboard
    read, and the fake never implemented the endpoint at all, so both the total and
    the shape of the array were unproven until a real store answered.
    """
    assert daemon.count() == 0
    entries = (make_entry("listed-a"), make_entry("listed-b"), make_entry("listed-c"))
    store(daemon, tmp_path / "outbox.db", *entries)

    assert daemon.count() == 3
    assert set(daemon.list_documents()) == {document_id(entry) for entry in entries}


@requires_daemon
def test_edges_are_derived_from_shared_frontmatter(daemon: TansekiClient, tmp_path: Path) -> None:
    """``repo``, ``pr`` and ``jira`` become edges the store derived, not ones sent.

    Kojutsu writes no edges, and that is only safe if the store derives them.
    That is a claim about the two sides agreeing on which frontmatter keys mean an
    edge, and it is the one thing the fake cannot make at all: it has no deriver to
    disagree with.
    """
    shared = make_entry("edge-a", repo="org/shared", pr_number=7, jira_ticket_key="ABC-1")
    sibling = make_entry("edge-b", repo="org/shared", pr_number=7, jira_ticket_key="ABC-1")
    # Differs from ``sibling`` in exactly one key, so an empty jira traversal here is
    # a statement about the key rather than about the store being empty.
    no_jira = make_entry("edge-c", repo="org/shared", pr_number=7)
    stranger = make_entry("edge-d", repo="org/other", pr_number=9, jira_ticket_key="ABC-2")
    store(daemon, tmp_path / "outbox.db", shared, sibling, no_jira, stranger)

    shared_id = document_id(shared)
    sibling_id = document_id(sibling)
    no_jira_id = document_id(no_jira)
    stranger_id = document_id(stranger)

    for relation in ("repo", "pr"):
        _assert_eventually(
            lambda rel=relation: sibling_id in daemon.traverse(shared_id, rel),
            f"traversing {relation} from {shared_id} reaches {sibling_id}",
        )
        assert stranger_id not in daemon.traverse(shared_id, relation)

    _assert_eventually(
        lambda: sibling_id in daemon.traverse(shared_id, "jira"),
        f"traversing jira from {shared_id} reaches {sibling_id}",
    )
    assert no_jira_id not in daemon.traverse(shared_id, "jira")
    assert daemon.traverse(no_jira_id, "jira") == []


@requires_daemon
def test_files_edge_resolves_against_a_document_written_afterwards(
    daemon: TansekiClient, tmp_path: Path
) -> None:
    """A ``files`` reference stays dangling until something writes that path.

    The seam document says the edge resolves against a document that already exists,
    which is a claim about ordering: the reference is written first, the target
    arrives afterwards, and the edge has to light up anyway because edges are
    rebuilt from documents rather than computed once at write time.
    """
    target = make_entry("files-target", repo="org/files", pr_number=3)
    store(daemon, tmp_path / "outbox.db", target)

    referring = make_entry(
        "files-source",
        repo="org/files",
        pr_number=4,
        files=[document_path(target)],
    )
    store(daemon, tmp_path / "outbox.db", referring)

    _assert_eventually(
        lambda: document_id(target) in daemon.traverse(document_id(referring), "files"),
        f"the files edge from {document_id(referring)} reaches {document_id(target)}",
    )


@requires_daemon
def test_search_filters_exclude_documents_the_query_matched(
    daemon: TansekiClient, tmp_path: Path
) -> None:
    """``tags`` and ``fm=`` are honoured by the store, not by the caller.

    A client filtering locally would answer identically, so the assertion that
    distinguishes them is exclusion: the query alone matches both documents, and
    only a constraint the server holds can drop one.
    """
    query = "seamfilterprobe"
    shared_answer = f"{query} is a token both documents carry verbatim."
    tagged = make_entry(
        "filter-tagged",
        repo="org/filtered",
        pr_number=11,
        tags=["seam-filter-probe"],
        answer=shared_answer,
    )
    untagged = make_entry(
        "filter-plain",
        repo="org/other-repo",
        pr_number=12,
        answer=shared_answer,
    )
    store(daemon, tmp_path / "outbox.db", tagged, untagged)

    tagged_id = document_id(tagged)
    untagged_id = document_id(untagged)
    both = {tagged_id, untagged_id}

    def unfiltered() -> set[str]:
        return {hit.id for hit in daemon.search(query, limit=10)}

    _assert_eventually(lambda: unfiltered() == both, "the query alone matches both documents")
    assert {hit.id for hit in daemon.search(query, tags=["seam-filter-probe"], limit=10)} == {
        tagged_id
    }
    assert {
        hit.id for hit in daemon.search(query, frontmatter={"repo": "org/other-repo"}, limit=10)
    } == {untagged_id}


@requires_daemon
def test_absent_documents_answer_not_found(daemon: TansekiClient) -> None:
    """The three 404 shapes stay distinct: ``None``, ``False``, ``[]``.

    They are not interchangeable. A ``:get`` that raised on 404 would turn an empty
    result into an error, and a ``:delete`` that returned ``True`` would report
    removing something that was never there.
    """
    absent = "org/repo/pr-0/never-written"
    assert daemon.get_document(absent) is None
    assert daemon.delete_document(absent, message="e2e", author="e2e") is False
    assert daemon.traverse(absent, "pr") == []


@requires_daemon
def test_deleting_a_document_removes_it_from_the_store(
    daemon: TansekiClient, tmp_path: Path
) -> None:
    """A delete is a store-side tombstone: the count and the listing both move."""
    entry = make_entry("to-delete", repo="org/deleted", pr_number=5)
    store(daemon, tmp_path / "outbox.db", entry)
    doc_id = document_id(entry)
    assert daemon.count() == 1

    assert daemon.delete_document(doc_id, message="e2e delete", author="e2e") is True
    assert daemon.get_document(doc_id) is None
    assert daemon.count() == 0
    assert doc_id not in daemon.list_documents()


@requires_daemon
def test_replaying_a_delivery_leaves_one_document(daemon: TansekiClient, tmp_path: Path) -> None:
    """Idempotency by document id, with the ``Idempotency-Key`` the client sends.

    Replay is the ordinary case rather than the exceptional one -- the outbox exists
    to retry after an outage. Whether the store deduplicates by id, by key, or on
    content is Tanseki's business; what Kojutsu relies on is that a replay leaves
    one document, which is the only part of this the fake could assert and only
    because it held a dict keyed on the id.
    """
    entry = make_entry("replayed", repo="org/replayed", pr_number=6)
    doc_id = document_id(entry)
    store(daemon, tmp_path / "outbox.db", entry)

    with TansekiOutbox(tmp_path / "replay.db") as outbox:
        assert outbox.enqueue(entry.entry_id, to_payload(entry))
        result = relay(outbox, daemon)
    assert result.sent == 1

    assert daemon.count() == 1
    assert daemon.list_documents() == [doc_id]


@requires_daemon
def test_listing_above_the_store_cap_is_refused(daemon: TansekiClient) -> None:
    """The 400 that makes ``MAX_LIST_RESULTS`` necessary.

    Issued raw, because the client clamps the limit and would hide the very
    behaviour its cap exists to avoid. This pins a reason stated in
    ``integrations/tanseki.py`` rather than a promise made by ``docs/tanseki-seam.md``,
    which documents only ``{total}`` -- so a failure here is a finding about the
    client's comment, not a regression in Kojutsu's behaviour.
    """
    headers = {"Accept": "application/json"}
    if _TANSEKI_API_KEY:
        headers["X-API-Key"] = _TANSEKI_API_KEY
    with httpx.Client(base_url=f"{_TANSEKI_URL}/v1", headers=headers, timeout=10.0) as raw:
        response = raw.get("/documents", params={"limit": 501, "collection": daemon.collection})
    assert response.status_code == 400, response.text
