"""Tests for the traversal and listing read tools (``traverse_knowledge``, ``list_knowledge``).

Both surfaces hand a caller something that could be mistaken for knowledge, so most
of what follows is about what each one may *not* imply: a bounded walk must not read
as the whole graph, an empty walk must not read as an empty store, a missing anchor
must not read as a graph with no neighbours, and a listing row must not read as
verified evidence.

The store fake is the one ``tests/test_mcp_server.py`` already uses, extended rather
than replaced. A second fake for the same seam is how two suites end up disagreeing
about the contract.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest

from kojutsu.integrations.tanseki import (
    MAX_LIST_RESULTS,
    TansekiClient,
    TansekiDocument,
    TansekiUnavailableError,
    bound_list_limit,
)

# ``tests/`` is on sys.path under pytest's default import mode, and this module is a
# peer of ``test_mcp_server`` rather than a replacement for it.
from test_mcp_server import FakeTanseki, assert_error, configure_server

SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"


def load_server():
    return importlib.reload(importlib.import_module("mcp_server.server"))


@pytest.fixture
def server():
    return load_server()


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> TansekiClient:
    """An Tanseki client over a mocked transport, for the store's own HTTP contract."""
    return TansekiClient(
        "https://tanseki.test",
        api_key="secret-key",
        collection="kojutsu",
        max_retries=0,
        client=httpx.Client(
            base_url="https://tanseki.test/v1", transport=httpx.MockTransport(handler)
        ),
    )


class GraphTanseki(FakeTanseki):
    """The existing MCP fake plus the two endpoints the new tools use.

    Edges are supplied as a graph and expanded here rather than handed back pre-baked,
    because the cases worth testing -- a cycle, a high-fan-out hub, a node reachable
    by two routes -- are properties of a graph and not of a canned list. A fake that
    only ever returns one fixed answer cannot fail the way a real walk fails.
    """

    def __init__(
        self,
        docs: dict[str, TansekiDocument],
        *,
        edges: dict[str, dict[str, list[str]]] | None = None,
        ordered_ids: list[str] | None = None,
        total: int | None = None,
    ) -> None:
        super().__init__(docs)
        self.edges = edges or {}
        self.ordered_ids = ordered_ids
        self.total = len(docs) if total is None else total
        self.traversals: list[tuple[str, str, int]] = []
        self.listed_with: int | None = None

    def traverse(self, doc_id: str, rel: str, *, depth: int = 3, **_: Any) -> list[str]:
        self.traversals.append((doc_id, rel, depth))
        return walk(self.edges, doc_id, rel, depth)

    def list_documents(self, *, limit: int = MAX_LIST_RESULTS, **_: Any) -> list[str]:
        self.listed_with = limit
        return list(self.ordered_ids if self.ordered_ids is not None else self.docs)[:limit]

    def count(self) -> int:
        return self.total


def walk(edges: dict[str, dict[str, list[str]]], start: str, rel: str, depth: int) -> list[str]:
    """Breadth-first walk with a visited set, so a cyclic graph still terminates here.

    The visited set is the fake's, not kojutsu's: this stands in for the store's
    walk, and a store that looped forever would hang the suite rather than fail it.
    What kojutsu must survive is the *output* of a walk over a cyclic graph, which
    is what the tests below assert.
    """
    seen = {start}
    reached: list[str] = []
    frontier = deque([start])
    for _ in range(max(0, depth)):
        following: deque[str] = deque()
        while frontier:
            for neighbour in edges.get(frontier.popleft(), {}).get(rel, []):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                reached.append(neighbour)
                following.append(neighbour)
        frontier = following
    return reached


class RawWalk(GraphTanseki):
    """A store that returns a hand-written id list, for the defensive cases.

    Used only where the store's own walk would never produce the input: an anchor
    returned as its own neighbour, or the same id once per route.
    """

    def __init__(self, docs: dict[str, TansekiDocument], raw: list[str]) -> None:
        super().__init__(docs)
        self.raw = raw

    def traverse(self, doc_id: str, rel: str, *, depth: int = 3, **_: Any) -> list[str]:
        self.traversals.append((doc_id, rel, depth))
        return list(self.raw)


def doc(
    doc_id: str,
    repo: str = "org/repo",
    *,
    chars: int | None = None,
    **frontmatter: Any,
) -> TansekiDocument:
    """A well-anchored capture, so a test opts into an anomaly rather than around one."""
    payload: dict[str, Any] = {
        "repo": repo,
        "pr": "7",
        "capture_source": "webhook",
        "delivery_id": f"delivery-{doc_id}",
        "captured_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(frontmatter)
    body = "body" if chars is None else "é" * chars
    return TansekiDocument(
        id=doc_id,
        path=f"{doc_id}.md",
        collection="kojutsu",
        content=f"# {doc_id}\n{body}",
        updated_at="2026-01-01T00:00:00+00:00",
        frontmatter=payload,
    )


def graph_docs(node_ids: Iterable[str], repo: str = "org/repo") -> dict[str, TansekiDocument]:
    return {node_id: doc(node_id, repo=repo) for node_id in node_ids}


def blocks(rendered: str) -> list[dict[str, Any]]:
    """The JSON payloads in a rendered answer, in order.

    Parsed rather than substring-matched for the reason ``read-path.md`` gives: the
    fence carries a random nonce, and a hex nonce can contain a short document id by
    coincidence often enough to make a substring assertion lie.
    """
    return [json.loads(line) for line in rendered.splitlines() if line.startswith("{")]


def returned_ids(rendered: str) -> list[str]:
    return [block["provenance"]["document_id"] for block in blocks(rendered)]


# --- happy paths --------------------------------------------------------------


def test_traverse_serves_related_documents_as_untrusted_evidence(server) -> None:
    fake = GraphTanseki(graph_docs(["anchor", "a1", "a2"]), edges={"anchor": {"pr": ["a1", "a2"]}})
    configure_server(server, fake)

    result = server.traverse_knowledge("anchor", repo="org/repo")

    assert result.ok is True
    assert returned_ids(result.result or "") == ["a1", "a2"]
    assert all(block["trust"] == "untrusted" for block in blocks(result.result or ""))
    # The bounds the caller chose reach the store rather than being applied after it:
    # a walk asked for one hop deep must not be sent deeper.
    assert fake.traversals == [("anchor", "pr", 1)]
    assert fake.closed is True


def test_traverse_sends_the_relation_and_depth_it_was_asked_for(server) -> None:
    fake = GraphTanseki(graph_docs(["anchor", "b1"]), edges={"anchor": {"jira": ["b1"]}})
    configure_server(server, fake)

    result = server.traverse_knowledge("anchor", repo="org/repo", rel="jira", depth=3)

    assert result.ok is True
    assert returned_ids(result.result or "") == ["b1"]
    assert fake.traversals == [("anchor", "jira", 3)]


def test_list_serves_index_rows_and_never_a_document_body(server) -> None:
    fake = GraphTanseki(
        {"d1": doc("d1", category="design_decision", record_kind="answer")}, total=1
    )
    configure_server(server, fake)

    result = server.list_knowledge(repo="org/repo")

    assert result.ok is True
    rows = blocks(result.result or "")
    assert [row["document_id"] for row in rows] == ["d1"]
    assert rows[0]["pr"] == "7"
    assert rows[0]["record_kind"] == "answer"
    assert rows[0]["structure"] == "anchored"
    # The distinction the whole surface rests on: a body would make this a search
    # that cannot be narrowed, and get_knowledge_entry is where a body is served.
    assert "content" not in rows[0]
    assert "body" not in (result.result or "")
    assert fake.listed_with == 20
    assert fake.closed is True


def test_list_covers_the_kinds_a_caller_cannot_search_for(server) -> None:
    """Census records and projected questions are the reason a listing exists at all."""
    fake = GraphTanseki(
        {
            "org/repo/pr-7/census/closed": doc("org/repo/pr-7/census/closed", record_kind="census"),
            "org/repo/pr-7/question/q1": doc(
                "org/repo/pr-7/question/q1", record_kind="question", question_status="answered"
            ),
        },
        total=2,
    )
    configure_server(server, fake)

    rows = blocks(server.list_knowledge(repo="org/repo").result or "")

    assert {row["record_kind"] for row in rows} == {"census", "question"}
    assert all(
        row["question_status"] == "answered" for row in rows if row["record_kind"] == "question"
    )


# --- input bounds -------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 4, "1", True, None, 1.0])
def test_traverse_rejects_a_depth_outside_its_stated_window(server, bad) -> None:
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    assert_error(server.traverse_knowledge("anchor", repo="org/repo", depth=bad), "invalid_input")
    assert "depth" in (server.traverse_knowledge("anchor", repo="org/repo", depth=bad).error or "")
    assert fake.traversals == []


@pytest.mark.parametrize("bad", [0, "3", True, False, None])
def test_traverse_rejects_a_fan_out_outside_its_stated_window(server, bad) -> None:
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    assert_error(server.traverse_knowledge("anchor", repo="org/repo", fan_out=bad), "invalid_input")
    assert fake.traversals == []


@pytest.mark.parametrize("bad", [0, 51, "10", True, None])
def test_traverse_rejects_a_limit_outside_its_stated_window(server, bad) -> None:
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    assert_error(server.traverse_knowledge("anchor", repo="org/repo", limit=bad), "invalid_input")
    assert fake.traversals == []


def test_a_string_bound_is_never_coerced_into_the_integer_it_spells(server) -> None:
    """``int("3")`` is 3, so a bound that parses its input invents a request.

    The caller asked for something kojutsu would have refused as an integer. The
    two ways that could go wrong are a wrong value and a right value for the wrong
    reason; both are the same defect, so both are refused the same way.
    """
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    for field in ("depth", "fan_out", "limit"):
        result = server.traverse_knowledge("anchor", repo="org/repo", **{field: "3"})
        assert_error(result, "invalid_input")
        assert field in (result.error or "")
    assert fake.traversals == []


def test_a_true_is_not_accepted_where_an_integer_is_stated(server) -> None:
    """``True`` is an ``int`` in Python and a JSON client's default for every flag."""
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    for field in ("depth", "fan_out", "limit"):
        assert_error(
            server.traverse_knowledge("anchor", repo="org/repo", **{field: True}),
            "invalid_input",
        )
    assert fake.traversals == []


def test_a_string_booleanness_filter_is_rejected_rather_than_becoming_truthy(server) -> None:
    """``bool("false")`` is True, so coercing it would *tighten* the filter.

    A caller who sent the string ``"false"`` asked for a looser answer and would get
    a stricter one, silently. That is the failure direction that loses evidence
    without the caller ever knowing a filter ran at all.
    """
    fake = GraphTanseki(graph_docs(["anchor"]), edges={"anchor": {"pr": []}})
    configure_server(server, fake)

    assert_error(
        server.traverse_knowledge("anchor", repo="org/repo", anchored_only="false"),
        "invalid_input",
    )
    assert fake.traversals == []


@pytest.mark.parametrize("rel", ["", "PR", "author", 3, None, "repo "])
def test_traverse_rejects_a_relation_tanseki_does_not_derive(server, rel) -> None:
    """An unknown relation walks to nothing, which reads as "no related knowledge".

    The store cannot distinguish a relation it does not have from a relation along
    which nothing is related, so the value is refused at the edge rather than turned
    into a claim about the store that nothing checked.
    """
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    assert_error(server.traverse_knowledge("anchor", repo="org/repo", rel=rel), "invalid_input")
    assert fake.traversals == []


def test_traverse_requires_an_explicit_repository(server) -> None:
    fake = GraphTanseki(graph_docs(["anchor"]))
    configure_server(server, fake)

    assert_error(server.traverse_knowledge("anchor"), "repository_required")
    assert_error(server.traverse_knowledge("anchor", repo="   "), "repository_required")
    assert fake.traversals == []


@pytest.mark.parametrize("bad", [0, -1, "20", True, None])
def test_list_rejects_a_limit_it_cannot_state(server, bad) -> None:
    fake = GraphTanseki({"d1": doc("d1")})
    configure_server(server, fake)

    assert_error(server.list_knowledge(repo="org/repo", limit=bad), "invalid_input")
    assert fake.listed_with is None


def test_list_requires_an_explicit_repository(server) -> None:
    fake = GraphTanseki({"d1": doc("d1")})
    configure_server(server, fake)

    assert_error(server.list_knowledge(), "repository_required")
    assert fake.listed_with is None


def test_a_listing_can_never_express_a_request_the_store_would_refuse(server) -> None:
    """Tanseki answers ``GET /v1/documents`` with a 400 above 500, so the tool stays below it."""
    assert MAX_LIST_RESULTS == 500
    fake = GraphTanseki({"d1": doc("d1")})
    configure_server(server, fake)

    assert_error(server.list_knowledge(repo="org/repo", limit=MAX_LIST_RESULTS), "invalid_input")
    assert_error(
        server.list_knowledge(repo="org/repo", limit=MAX_LIST_RESULTS + 1), "invalid_input"
    )
    assert fake.listed_with is None
    assert server.MAX_LIST_TOOL_LIMIT < MAX_LIST_RESULTS


def test_the_client_holds_its_own_listing_request_inside_the_store_cap() -> None:
    """The cap is the store's precondition, so the client enforces it, not each caller.

    Otherwise the first surface to pass a large limit receives a 400 it reports as its
    own failure, and kojutsu's unstated precondition reads as a caller's mistake.
    """
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("limit"))
        return httpx.Response(200, json={"total": 0, "documents": []})

    assert make_client(handler).list_documents(limit=MAX_LIST_RESULTS + 1) == []
    assert seen == [str(MAX_LIST_RESULTS)]


@pytest.mark.parametrize("bad", [0, -1, "3", True, None, 1.0])
def test_the_listing_bound_refuses_a_non_integer_outright(bad) -> None:
    with pytest.raises(ValueError):
        bound_list_limit(bad)


# --- provenance is re-checked on every record either surface returns -----------


def test_traverse_flags_an_anchorless_capture_and_still_serves_it(server) -> None:
    fake = RawWalk(graph_docs(["anchor", "bad"]), ["bad"])
    fake.docs["bad"] = TansekiDocument(
        id="bad",
        path="bad.md",
        collection="kojutsu",
        content="body",
        # Claims a signed delivery and carries no delivery id to check it against.
        frontmatter={"repo": "org/repo", "pr": "7", "capture_source": "webhook"},
    )
    configure_server(server, fake)

    result = server.traverse_knowledge("anchor", repo="org/repo")

    assert result.ok is True
    provenance = blocks(result.result or "")[0]["provenance"]
    # Served *and* flagged. Discarding it would lose real stored content; serving it
    # unflagged is the defect the re-check exists to prevent.
    assert returned_ids(result.result or "") == ["bad"]
    assert "capture_anchor_missing" in provenance["provenance_anomalies"]


def test_traverse_flags_an_independence_verdict_without_a_capture(server) -> None:
    fake = RawWalk(graph_docs(["anchor", "q"]), ["q"])
    fake.docs["q"] = doc("q", capture_source="asserted", independence="independent")
    configure_server(server, fake)

    provenance = blocks(server.traverse_knowledge("anchor", repo="org/repo").result or "")[0][
        "provenance"
    ]

    assert "independence_without_capture" in provenance["provenance_anomalies"]


def test_a_traversal_crosses_record_kinds_and_checks_each_one(server) -> None:
    """A pr hop lands on every kind written for that change, and each is re-checked.

    Crossing kinds is the point of the surface, and it is also where an inferred
    pairing could be mistaken for a conversation somebody had, so each document is
    held to the same standard a search would have held it to.
    """
    fake = RawWalk(graph_docs(["anchor", "answer", "verdict"]), ["answer", "verdict"])
    fake.docs["answer"] = doc("answer", record_kind="answer")
    fake.docs["verdict"] = doc("verdict", record_kind="review_verdict")
    configure_server(server, fake)

    provenance = {
        block["provenance"]["document_id"]: block["provenance"]
        for block in blocks(server.traverse_knowledge("anchor", repo="org/repo").result or "")
    }

    assert set(provenance) == {"answer", "verdict"}
    assert "provenance_anomalies" not in provenance["answer"]
    assert "provenance_anomalies" not in provenance["verdict"]


def test_list_rows_carry_the_same_provenance_check_as_a_rendered_record(server) -> None:
    """A row is not a body, but a reader concludes from it just as firmly.

    A row saying ``capture_source: webhook`` with nothing behind it and no flag makes
    the same false claim the evidence block refuses to make.
    """
    fake = GraphTanseki(
        {
            "bad": TansekiDocument(
                id="bad",
                path="bad.md",
                collection="kojutsu",
                content="body",
                frontmatter={"repo": "org/repo", "pr": "7", "capture_source": "webhook"},
            )
        },
        total=1,
    )
    configure_server(server, fake)

    row = blocks(server.list_knowledge(repo="org/repo").result or "")[0]

    assert row["capture_source"] == "webhook"
    assert "capture_anchor_missing" in row["provenance_anomalies"]


def test_list_rows_flag_an_independence_verdict_without_a_capture(server) -> None:
    fake = GraphTanseki(
        {"q": doc("q", capture_source="asserted", independence="independent")}, total=1
    )
    configure_server(server, fake)

    row = blocks(server.list_knowledge(repo="org/repo").result or "")[0]

    assert "independence_without_capture" in row["provenance_anomalies"]


def test_a_list_row_bounds_each_value(server) -> None:
    """Fifty rows of 2 000 characters is not an index, and cannot be skimmed."""
    fake = GraphTanseki({"d1": doc("d1", title="t" * 5_000)}, total=1)
    configure_server(server, fake)

    row = blocks(server.list_knowledge(repo="org/repo").result or "")[0]

    assert row["title"].endswith("[truncated]")
    assert len(row["title"]) <= server.MAX_SUMMARY_VALUE_CHARS + len(" [truncated]")
    assert server.MAX_SUMMARY_VALUE_CHARS < server.MAX_METADATA_VALUE_CHARS


def test_a_clean_row_carries_no_anomaly_field_at_all(server) -> None:
    """A check that fires on every row is a check that reports nothing."""
    fake = GraphTanseki({"d1": doc("d1")}, total=1)
    configure_server(server, fake)

    row = blocks(server.list_knowledge(repo="org/repo").result or "")[0]

    assert "provenance_anomalies" not in row
    assert row["structure"] == "anchored"


# --- a cycle terminates and the answer stays bounded --------------------------


def test_a_traversal_through_a_cycle_terminates_and_fetches_each_node_once(server) -> None:
    """a -> b -> c -> a, walked at the maximum depth.

    Kojutsu has no recursion here at all: one store request, a flat id list, then
    deduplication and a cap. So the property is not "it did not hang" -- it cannot --
    but that a cycle cannot spend fan-out twice on one document, and that the answer
    is a set of documents rather than of routes.
    """
    fake = GraphTanseki(
        graph_docs(["a", "b", "c"]),
        edges={"a": {"pr": ["b"]}, "b": {"pr": ["c"]}, "c": {"pr": ["a", "b"]}},
    )
    configure_server(server, fake)

    result = server.traverse_knowledge("a", repo="org/repo", depth=3)

    assert result.ok is True
    assert fake.traversals == [("a", "pr", 3)]
    # Reached by two routes in the graph, fetched and answered exactly once.
    assert fake.fetched == ["b", "c"]
    assert returned_ids(result.result or "") == ["b", "c"]
    assert "not the whole graph" in (result.result or "")


def test_a_cycle_cannot_serve_the_anchor_as_its_own_neighbour(server) -> None:
    """A walk that re-enters its anchor must not invent a second document."""
    fake = RawWalk(graph_docs(["a", "b"]), ["a", "b", "a"])

    configure_server(server, fake)

    ids = returned_ids(server.traverse_knowledge("a", repo="org/repo").result or "")

    assert ids == ["b"]


def test_a_repeated_id_never_appears_twice_in_the_answer(server) -> None:
    fake = RawWalk(graph_docs(["a", "x", "y"]), ["x", "x", "y", "x"])
    configure_server(server, fake)

    assert returned_ids(server.traverse_knowledge("a", repo="org/repo").result or "") == [
        "x",
        "y",
    ]


def test_a_hub_with_many_neighbours_never_multiplies_into_an_answer(server) -> None:
    """The case the bound exists for: one node with a very large fan-out.

    Sixty neighbours at the maximum fan-out is still twenty-five documents fetched and
    ten rendered, and both shortfalls are named rather than absorbed into the answer.
    """
    neighbours = [f"n{index}" for index in range(60)]
    fake = GraphTanseki(graph_docs(["hub", *neighbours]), edges={"hub": {"pr": neighbours}})
    configure_server(server, fake)

    result = server.traverse_knowledge("hub", repo="org/repo", fan_out=25, limit=3)

    assert len(returned_ids(result.result or "")) == 3
    assert fake.fetched == neighbours[:25]
    rendered = result.result or ""
    assert "35 further related document id(s) were beyond the fan_out bound of 25" in rendered
    assert "22 usable related entries were within fan_out but beyond the limit of 3" in rendered


def test_a_walk_that_fetches_nothing_fetches_nothing(server) -> None:
    """The empty-graph case must not cost a document fetch to discover."""
    fake = GraphTanseki(graph_docs(["a"]))
    configure_server(server, fake)

    result = server.traverse_knowledge("a", repo="org/repo")

    assert result.ok is True
    assert fake.fetched == []
    assert fake.traversals == [("a", "pr", 1)]


# --- exclusions are counted and named, never dropped --------------------------


def test_a_walk_that_only_reaches_another_repository_says_so(server) -> None:
    """A jira edge crosses repositories by design, which makes this check load-bearing."""
    fake = GraphTanseki(
        {"a": doc("a"), "far": doc("far", repo="other/x")}, edges={"a": {"jira": ["far"]}}
    )
    configure_server(server, fake)

    result = server.traverse_knowledge("a", repo="org/repo", rel="jira")

    assert result.ok is True
    assert "1 belonged to another repository" in (result.result or "")
    assert "far" not in (result.result or "")


def test_a_document_that_vanished_is_named_rather_than_dropped(server) -> None:
    class Vanishing(GraphTanseki):
        def get_documents(self, doc_ids: list[str]) -> list[TansekiDocument | None]:
            self.fetched.extend(doc_ids)
            return [None] * len(doc_ids)

    fake = Vanishing(graph_docs(["a", "gone"]), edges={"a": {"pr": ["gone"]}})
    configure_server(server, fake)

    rendered = server.traverse_knowledge("a", repo="org/repo").result or ""

    # Distinct from an empty graph, and distinct from a cross-repository discard:
    # the store did hold this and could not serve it.
    assert "reached 1 related document(s)" in rendered
    assert "1 could no longer be retrieved from the store" in rendered


def test_a_walk_filtered_to_nothing_reports_the_filter_not_an_empty_graph(server) -> None:
    fake = GraphTanseki(
        graph_docs(["a"]),
        edges={"a": {"pr": ["i"]}},
    )
    fake.docs["i"] = doc(
        "i", capture_source="asserted", structure="inferred", structure_inferred_by="model-x"
    )
    configure_server(server, fake)

    rendered = server.traverse_knowledge("a", repo="org/repo", anchored_only=True).result or ""

    assert "1 were not anchored pairings" in rendered
    assert "No usable knowledge entries remain." in rendered
    assert "No related knowledge entries found" not in rendered


def test_a_walk_below_the_independence_threshold_reports_it(server) -> None:
    fake = GraphTanseki(graph_docs(["a", "weak"]), edges={"a": {"pr": ["weak"]}})
    fake.docs["weak"] = doc("weak", independence="self_certified")
    configure_server(server, fake)

    rendered = (
        server.traverse_knowledge("a", repo="org/repo", min_independence="independent").result or ""
    )

    assert "1 were below the requested independence level" in rendered


def test_a_walk_with_no_neighbours_says_the_entry_exists_and_the_walk_was_empty(server) -> None:
    """A real entry with no edges is not a missing entry."""
    fake = GraphTanseki(graph_docs(["a"]))
    configure_server(server, fake)

    rendered = server.traverse_knowledge("a", repo="org/repo").result or ""

    # Reporting this as a missing entry would send the caller hunting a typo that does
    # not exist, which is the one thing a traversal can most easily get wrong.
    assert "The entry exists and the traversal completed" in rendered
    assert "No related knowledge entries found" in rendered


def test_a_missing_anchor_is_not_reported_as_an_empty_graph(server) -> None:
    """``:traverse`` answers a missing document with an empty id list, like no edges."""
    fake = GraphTanseki(graph_docs(["a"]))
    configure_server(server, fake)

    result = server.traverse_knowledge("nope", repo="org/repo")

    assert_error(result, "not_found")
    # The walk is not even attempted: an anchor that is not there has no neighbourhood.
    assert fake.traversals == []


def test_an_unauthorized_anchor_is_denied_before_the_store_is_asked_to_walk(server) -> None:
    fake = GraphTanseki({"far": doc("far", repo="other/x")})
    configure_server(server, fake)

    result = server.traverse_knowledge("far", repo="org/repo")

    assert_error(result, "repository_not_authorized")
    assert fake.traversals == []


def test_a_response_budget_that_carries_nothing_names_the_entries_it_dropped(server) -> None:
    """One document can outgrow the whole budget, and must not become an empty walk."""
    fake = RawWalk(graph_docs(["a"]), ["big"])
    fake.docs["big"] = doc("big", chars=server.MAX_SEARCH_RESPONSE_CHARS + 1)
    configure_server(server, fake)

    result = server.traverse_knowledge("a", repo="org/repo")

    assert result.ok is True
    rendered = result.result or ""
    assert "excluded by the" in rendered
    assert "Matching entries: big" in rendered
    assert "This is a bounded answer, not an empty one" in rendered


def test_the_exclusions_are_counted_rather_than_dropped_on_the_floor(server) -> None:
    """The read log is told about every exclusion, so the count cannot disagree with it."""
    from kojutsu.core.read_log import ExclusionReason, ReadAccounting
    from kojutsu.models import Independence

    fake = GraphTanseki(
        {
            "a": doc("a"),
            "far": doc("far", repo="other/x", independence="independent"),
            "weak": doc("weak", independence="self_certified"),
            "a2": doc("a2", independence="independent"),
        },
        edges={"a": {"jira": ["far", "weak", "a2"]}},
    )
    configure_server(server, fake)
    accounting = ReadAccounting()

    server._traverse_tanseki(
        fake,
        anchor="a",
        rel="jira",
        repo="org/repo",
        depth=1,
        fan_out=25,
        limit=10,
        min_independence=Independence.INDEPENDENT,
        accounting=accounting,
    )

    assert accounting.delivered == 1
    assert accounting.excluded == {
        ExclusionReason.CROSS_REPOSITORY: 1,
        ExclusionReason.BELOW_MIN_INDEPENDENCE: 1,
    }
    assert accounting.excluded_count == 2


# --- a listing must stay a bounded listing ------------------------------------


def test_a_listing_larger_than_the_window_does_not_pretend_to_be_complete(server) -> None:
    fake = GraphTanseki(graph_docs(["d1", "d2"]), ordered_ids=["d1", "d2"], total=900)
    configure_server(server, fake)

    rendered = server.list_knowledge(repo="org/repo", limit=2).result or ""

    assert "The store holds 900 document(s) in total" in rendered
    assert "898 document(s) were never listed and this is not the collection" in rendered
    assert "in the store's enumeration order rather than by relevance" in rendered


def test_a_listing_cannot_report_a_repository_empty_when_the_window_missed_it(server) -> None:
    fake = GraphTanseki({"other": doc("other", repo="other/x")}, ordered_ids=["other"], total=5_000)
    configure_server(server, fake)

    rendered = server.list_knowledge(repo="org/repo").result or ""

    assert "none belonged to org/repo within that window" in rendered
    assert "This is a bounded listing, not an empty repository." in rendered
    assert "can fall outside the window" in rendered


def test_a_complete_listing_says_so_rather_than_hedging(server) -> None:
    fake = GraphTanseki({"d1": doc("d1")}, ordered_ids=["d1"], total=1)
    configure_server(server, fake)

    rendered = server.list_knowledge(repo="org/repo").result or ""

    assert "the complete enumeration at this window" in rendered
    assert "cannot say whether it is complete" not in rendered


def test_a_listing_survives_an_unavailable_document_total_rather_than_refusing(server) -> None:
    class NoCount(GraphTanseki):
        def count(self) -> int:
            raise TansekiUnavailableError("store down")

    fake = NoCount(graph_docs(["d1"]), ordered_ids=["d1"], total=1)
    configure_server(server, fake)

    result = server.list_knowledge(repo="org/repo")

    # Not worth refusing a read the store would otherwise serve, but the gap in what
    # can be claimed about it has to be stated rather than assumed.
    assert result.ok is True
    assert "cannot say whether it is complete" in (result.result or "")
    assert [row["document_id"] for row in blocks(result.result or "")] == ["d1"]


def test_a_listing_never_asks_the_store_for_more_than_its_own_window(server) -> None:
    fake = GraphTanseki(graph_docs(["d1"]), ordered_ids=["d1"], total=1)
    configure_server(server, fake)

    server.list_knowledge(repo="org/repo", limit=5)

    assert fake.listed_with == 5


def test_a_listing_has_no_query_and_no_filters_so_it_cannot_become_a_search(server) -> None:
    """A listing that grew a query would be a second search surface with weaker bounds."""
    import inspect

    parameters = inspect.signature(server.list_knowledge).parameters

    assert list(parameters) == ["repo", "limit"]


def test_a_listing_reports_documents_that_vanished_rather_than_dropping_them(server) -> None:
    class Vanishing(GraphTanseki):
        def get_documents(self, doc_ids: list[str]) -> list[TansekiDocument | None]:
            self.fetched.extend(doc_ids)
            return [None] * len(doc_ids)

    fake = Vanishing(graph_docs(["gone"]), ordered_ids=["gone"], total=1)
    configure_server(server, fake)

    rendered = server.list_knowledge(repo="org/repo").result or ""

    assert "none belonged to org/repo within that window" in rendered
    assert "This is a bounded listing, not an empty repository." in rendered


# --- the surface itself --------------------------------------------------------


def test_the_new_tools_are_registered_and_read_only(server) -> None:
    tools = asyncio.run(server.server.list_tools())
    names = {tool.name for tool in tools}

    assert {"traverse_knowledge", "list_knowledge"} <= names
    for tool in tools:
        assert tool.annotations is not None, f"{tool.name} declares no annotations at all"
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False


def test_the_new_tool_descriptions_state_their_bounds(server) -> None:
    """A bound the caller cannot read is a bound they cannot rely on."""
    tools = {tool.name: tool for tool in asyncio.run(server.server.list_tools())}

    traverse = tools["traverse_knowledge"].description or ""
    assert f"1-{server.MAX_TRAVERSE_DEPTH}" in traverse
    assert f"1-{server.MAX_TRAVERSE_FAN_OUT}" in traverse
    assert f"1-{server.MAX_SEARCH_LIMIT}" in traverse

    listing = tools["list_knowledge"].description or ""
    assert f"1-{server.MAX_LIST_TOOL_LIMIT}" in listing
    assert str(MAX_LIST_RESULTS) in listing


def test_the_read_server_still_imports_no_write_path(server) -> None:
    source = SERVER_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "post_issue_comment",
        "claim_rationale",
        "KnowledgeSink",
        "TansekiOutbox",
        "capture_server",
    ):
        assert forbidden not in source
