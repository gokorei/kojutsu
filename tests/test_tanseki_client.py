"""Tests for the Tanseki client against a mocked HTTP transport."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from kojutsu.integrations.tanseki import (
    DEFAULT_COLLECTION,
    MAX_LIST_RESULTS,
    MAX_RETRIES_CEILING,
    MAX_RETRY_DELAY_SECONDS,
    MAX_SEARCH_RESULTS,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_JITTER_RATIO,
    TansekiAuthenticationError,
    TansekiClient,
    TansekiDocument,
    TansekiError,
    TansekiPermanentError,
    TansekiResponseError,
    TansekiUnavailableError,
    _retry_delay_seconds,
)


def make_client(
    handler,
    *,
    max_retries: int = 0,
    collection: str = "kojutsu",
    sleep: Any = None,
    clock: Any = None,
    jitter: Any = None,
) -> TansekiClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="https://tanseki.test/v1", transport=transport)
    extra: dict[str, Any] = {}
    if sleep is not None:
        extra["sleep"] = sleep
    if clock is not None:
        extra["clock"] = clock
    if jitter is not None:
        extra["jitter"] = jitter
    return TansekiClient(
        "https://tanseki.test",
        api_key="secret-key",
        collection=collection,
        client=http,
        max_retries=max_retries,
        **extra,
    )


class Sleeps(list):
    """A stand-in sleeper that records what it was asked to wait for."""

    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def test_remote_http_is_rejected_and_loopback_http_is_accepted() -> None:
    with pytest.raises(TansekiPermanentError, match="HTTPS"):
        TansekiClient("http://tanseki.example.com")

    with TansekiClient("http://127.0.0.1:8000") as client:
        assert client.collection == DEFAULT_COLLECTION


def test_health_ok() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"status": "ok"}))
    assert client.health() is True


def test_health_unavailable() -> None:
    client = make_client(lambda request: httpx.Response(503))
    assert client.health() is False


def test_count_requires_a_non_negative_integer_total() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"total": 2}))
    assert client.count() == 2

    for payload in ({"total": -1}, {"total": True}, {"total": "2"}, {}):
        client = make_client(lambda request, payload=payload: httpx.Response(200, json=payload))
        with pytest.raises(TansekiResponseError):
            client.count()


def test_count_sends_bounded_collection_listing_request() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["limit"] = request.url.params.get("limit", "")
        captured["collection"] = request.url.params.get("collection", "")
        return httpx.Response(200, json={"total": 0})

    client = make_client(handler)
    client.count()

    assert captured == {
        "path": "/v1/documents",
        "limit": "1",
        "collection": "kojutsu",
    }


def test_list_documents_stays_within_tansekis_listing_cap() -> None:
    """Tanseki answers ``GET /v1/documents`` with a 400 above 500."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        limit = request.url.params.get("limit", "")
        seen.append(limit)
        return httpx.Response(200, json={"total": 0, "documents": []})

    client = make_client(handler)
    assert client.list_documents() == []
    assert seen == [str(MAX_LIST_RESULTS)]
    assert MAX_LIST_RESULTS == 500


def test_list_documents_reads_ids_from_the_documented_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total": 3,
                "limit": 3,
                "offset": 0,
                "hasMore": False,
                "documents": [
                    {"id": "org/repo/pr-1/a", "collection": "kojutsu", "path": "a.md"},
                    "org/repo/pr-1/b",
                    {"path": "c.md"},
                ],
            },
        )

    assert make_client(handler).list_documents(limit=3) == ["org/repo/pr-1/a", "org/repo/pr-1/b"]


def test_list_documents_rejects_an_envelope_without_an_array() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"total": 2})

    with pytest.raises(TansekiResponseError, match="no document array"):
        make_client(handler).list_documents()


def test_get_document_sends_auth_and_collection() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["api_key"] = request.headers.get("X-API-Key", "")
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "id": "kojutsu/org/repo/pr-1/abc",
                "path": "kojutsu/org/repo/pr-1/abc.md",
                "collection": "kojutsu",
                "content": "# Q\nA",
                "contentHash": "deadbeef",
                "revision": "r1",
                "frontmatter": {"repo": "org/repo"},
            },
        )

    client = make_client(handler)
    doc = client.get_document("kojutsu/org/repo/pr-1/abc")

    assert isinstance(doc, TansekiDocument)
    assert doc.frontmatter["repo"] == "org/repo"
    assert captured["api_key"] == "secret-key"
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/documents:get")
    assert '"collection":"kojutsu"' in captured["body"]


@pytest.mark.parametrize(
    ("document_id", "collection"),
    [
        ("another-id", "kojutsu"),
        ("requested-id", "another-collection"),
    ],
)
def test_get_document_rejects_mismatched_identity_or_collection(
    document_id: str, collection: str
) -> None:
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={
                "id": document_id,
                "path": "document.md",
                "collection": collection,
                "content": "answer",
            },
        )
    )

    with pytest.raises(TansekiResponseError, match=r"identity|collection"):
        client.get_document("requested-id")


def test_get_document_requires_returned_collection() -> None:
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"id": "d1", "path": "d1.md", "content": "answer"},
        )
    )

    with pytest.raises(TansekiResponseError, match="collection"):
        client.get_document("d1")


def test_get_documents_validates_each_returned_document() -> None:
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"id": "d1", "path": "d1.md", "collection": "other", "content": "answer"},
        )
    )

    with pytest.raises(TansekiResponseError, match="collection"):
        client.get_documents(["d1", "d2"], max_workers=1)


def test_get_document_missing_returns_none() -> None:
    client = make_client(lambda request: httpx.Response(404))
    assert client.get_document("nope") is None


def test_search_parses_hits_and_tags() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "hits": [{"id": "d1", "score": 1.5, "snippet": "..."}],
                "limit": 5,
                "offset": 0,
                "hasMore": False,
            },
        )

    client = make_client(handler)
    hits = client.search("auth", tags=["design"], limit=5)

    assert len(hits) == 1
    assert hits[0].id == "d1"
    assert hits[0].score == 1.5
    assert "tags=design" in captured["url"]


def test_search_accepts_the_tanseki_pagination_envelope_without_total() -> None:
    """Tanseki's SearchResponse is {hits, limit, offset, hasMore} and has no total."""
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={
                "hits": [{"id": "d1", "score": 1.0}, {"id": "d2", "score": 0.5}],
                "limit": 2,
                "offset": 0,
                "hasMore": True,
            },
        )
    )
    hits = client.search("x", limit=2)
    assert [hit.id for hit in hits] == ["d1", "d2"]


def test_search_rejects_a_malformed_present_envelope_field() -> None:
    for key, value in (("offset", -1), ("limit", "5"), ("hasMore", "false"), ("total", "1")):
        client = make_client(
            lambda request, key=key, value=value: httpx.Response(
                200, json={"hits": [{"id": "d1", "score": 1}], key: value}
            )
        )
        with pytest.raises(TansekiResponseError, match=key):
            client.search("x")


def test_search_validates_every_hit() -> None:
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"hits": [{"id": "d1", "score": "1"}], "limit": 5, "hasMore": False},
        )
    )
    with pytest.raises(TansekiResponseError, match="score"):
        client.search("x")


def test_search_rejects_total_smaller_than_returned_hits() -> None:
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"hits": [{"id": "d1", "score": 1}], "total": 0},
        )
    )
    with pytest.raises(TansekiResponseError, match="total"):
        client.search("x")


def test_search_sends_frontmatter_filters() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"hits": [], "limit": 5, "offset": 0, "hasMore": False})

    client = make_client(handler)
    client.search("x", frontmatter={"repo": "org/repo", "jira": "ABC-1"})
    assert "fm=repo" in captured["url"]
    assert "fm=jira" in captured["url"]


def test_upsert_document_rejects_blank_or_untyped_ids() -> None:
    """Falsy is not the check: ``''``, whitespace, ``0`` and ``False`` are all
    distinct facts and none of them is a document id."""
    client = make_client(lambda request: httpx.Response(200, json={"revision": "r1"}))

    for bad_id in ("", "   ", 0, False, None):
        with pytest.raises(TansekiError, match="non-blank string 'id'"):
            client.upsert_document({"id": bad_id, "content": "x"})


def test_upsert_document_puts_payload_with_forced_collection() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"revision": "r2", "created": True})

    client = make_client(handler, collection="configured")
    result = client.upsert_document(
        {"id": "d1", "path": "d1.md", "content": "x", "collection": "attacker"}
    )

    assert captured["method"] == "POST"
    assert captured["path"].endswith("/documents:upsert")
    assert result["revision"] == "r2"
    assert captured["body"] == {
        "id": "d1",
        "path": "d1.md",
        "content": "x",
        "collection": "configured",
    }


def test_delete_document_reports_missing() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(404)

    client = make_client(handler)
    assert client.delete_document("d1", message="m", author="a") is False
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/documents:delete")


def test_traverse_returns_ids() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"ids": ["a", "b"]})

    client = make_client(handler)
    assert client.traverse("d1", "links-to", depth=2) == ["a", "b"]
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/documents:traverse")


def test_transient_failure_raises_unavailable() -> None:
    client = make_client(lambda request: httpx.Response(503), max_retries=0)
    with pytest.raises(TansekiUnavailableError):
        client.get_document("d1")


def _sequence(responses: list[httpx.Response]):
    """Serve ``responses`` in order, then repeat the last one forever."""
    remaining = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return handler


def test_retry_after_seconds_form_is_honoured_over_computed_backoff() -> None:
    """A store that names a wait is believed exactly, and not jittered."""
    sleeps = Sleeps()
    client = make_client(
        _sequence(
            [
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json={"hits": [], "hasMore": False}),
            ]
        ),
        max_retries=2,
        sleep=sleeps,
        jitter=lambda: 0.5,
    )

    assert client.search("x") == []
    assert sleeps == [7.0]


def test_retry_after_http_date_form_is_read_against_the_injected_clock() -> None:
    """The date form is the one a proxy rewrites a header into; it used to be ignored."""
    sleeps = Sleeps()
    client = make_client(
        _sequence(
            [
                httpx.Response(
                    429,
                    headers={"Retry-After": "Wed, 21 Oct 2015 07:28:30 GMT"},
                ),
                httpx.Response(200, json={"hits": [], "hasMore": False}),
            ]
        ),
        max_retries=2,
        sleep=sleeps,
        clock=lambda: datetime(2015, 10, 21, 7, 28, 0, tzinfo=UTC).timestamp(),
    )

    assert client.search("x") == []
    assert sleeps == [30.0]


def test_retry_after_date_in_the_past_means_now_rather_than_falling_back() -> None:
    """The window Tanseki named has already passed; backing off would be inventing one."""
    sleeps = Sleeps()
    client = make_client(
        _sequence(
            [
                httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:27:00 GMT"}),
                httpx.Response(200, json={"hits": [], "hasMore": False}),
            ]
        ),
        max_retries=2,
        sleep=sleeps,
        clock=lambda: datetime(2015, 10, 21, 7, 29, 0, tzinfo=UTC).timestamp(),
    )

    assert client.search("x") == []
    assert sleeps == [0.0]


def test_retry_after_is_capped_and_a_malformed_header_is_ignored() -> None:
    """Neither an absurd header nor an unparseable one becomes an unbounded wait."""
    sleeps = Sleeps()
    capped = make_client(
        _sequence(
            [
                httpx.Response(429, headers={"Retry-After": "99999"}),
                httpx.Response(200, json={"hits": [], "hasMore": False}),
            ]
        ),
        max_retries=1,
        sleep=sleeps,
    )
    assert capped.search("x") == []
    assert sleeps == [MAX_RETRY_DELAY_SECONDS]

    garbage = Sleeps()
    malformed = make_client(
        _sequence(
            [
                httpx.Response(429, headers={"Retry-After": "soon-ish"}),
                httpx.Response(200, json={"hits": [], "hasMore": False}),
            ]
        ),
        max_retries=1,
        sleep=garbage,
        jitter=lambda: 0.0,
    )
    assert malformed.search("x") == []
    assert garbage == [RETRY_BASE_DELAY_SECONDS * (1 - RETRY_JITTER_RATIO)]


def test_computed_backoff_grows_and_never_leaves_the_ceiling() -> None:
    sleeps = Sleeps()
    client = make_client(
        lambda request: httpx.Response(503),
        max_retries=6,
        sleep=sleeps,
        jitter=lambda: 0.5,
    )

    with pytest.raises(TansekiUnavailableError):
        client.get_document("d1")

    assert len(sleeps) == 6
    assert sleeps == sorted(sleeps), "a later attempt must not wait less than an earlier one"
    assert all(0.0 < value <= MAX_RETRY_DELAY_SECONDS for value in sleeps)


@pytest.mark.parametrize("jitter", [0.0, 0.5, 0.999999])
def test_jitter_cannot_produce_a_negative_delay_or_one_past_the_ceiling(jitter: float) -> None:
    for attempt in range(12):
        for retry_after in (None, -5.0, 0.0, 1.0, MAX_RETRY_DELAY_SECONDS, 1e9):
            delay = _retry_delay_seconds(attempt=attempt, retry_after=retry_after, jitter=jitter)
            assert 0.0 <= delay <= MAX_RETRY_DELAY_SECONDS


def test_jitter_spreads_retries_that_would_otherwise_be_synchronised() -> None:
    """Two clients failing at the same instant must not wake at the same instant."""
    delays = [
        _retry_delay_seconds(attempt=2, retry_after=None, jitter=fraction)
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.999999)
    ]

    assert len(set(delays)) == len(delays)
    assert delays == sorted(delays)
    window = RETRY_BASE_DELAY_SECONDS * 4
    assert delays[0] == pytest.approx(window * (1 - RETRY_JITTER_RATIO))
    assert delays[-1] == pytest.approx(window)


def test_attempts_are_bounded_whatever_the_configuration_asks_for() -> None:
    """Unbounded retry against a rate-limited store is how the client becomes the outage."""
    requests: list[httpx.Request] = []
    sleeps = Sleeps()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = make_client(handler, max_retries=10_000, sleep=sleeps)

    with pytest.raises(TansekiUnavailableError):
        client.get_document("d1")

    assert client.max_retries == MAX_RETRIES_CEILING
    assert len(requests) == MAX_RETRIES_CEILING + 1
    assert len(sleeps) == MAX_RETRIES_CEILING


def test_retry_after_is_reread_per_attempt_rather_than_carried_forward() -> None:
    """A delay belongs to the response that asked for it."""
    sleeps = Sleeps()
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "9"}),
            httpx.Response(200, json={"hits": [], "hasMore": False}),
        ]
    )
    client = make_client(
        lambda request: next(responses),
        max_retries=3,
        sleep=sleeps,
        jitter=lambda: 0.0,
    )

    assert client.search("x") == []
    assert sleeps == [9.0]


def test_retry_after_survives_into_the_raised_error_for_the_caller() -> None:
    client = make_client(
        lambda request: httpx.Response(429, headers={"Retry-After": "12"}),
        max_retries=0,
    )

    with pytest.raises(TansekiUnavailableError) as raised:
        client.get_document("d1")

    assert raised.value.retry_after == 12.0


def test_permanent_auth_failure_is_normalized() -> None:
    client = make_client(lambda request: httpx.Response(401))
    with pytest.raises(TansekiAuthenticationError):
        client.upsert_document({"id": "d1", "content": "x"})


def test_invalid_response_shape_is_normalized() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(TansekiResponseError):
        client.search("x")


def test_search_limit_is_capped() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["limit"] = request.url.params.get("limit", "")
        return httpx.Response(200, json={"hits": [{"id": "d1", "score": 1}], "hasMore": False})

    client = make_client(handler)
    client.search("x", limit=1000)

    assert captured["limit"] == str(MAX_SEARCH_RESULTS)
    with pytest.raises(ValueError, match="at least 1"):
        client.search("x", limit=0)


def test_search_documents_bounds_search_and_document_requests() -> None:
    request_counts = {"search": 0, "get": 0}
    search_limit = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal search_limit
        if request.url.path.endswith("/search"):
            request_counts["search"] += 1
            search_limit = int(request.url.params["limit"])
            hits = [{"id": f"d{index}", "score": 1.0} for index in range(120)]
            return httpx.Response(
                200,
                json={
                    "hits": hits,
                    "limit": MAX_SEARCH_RESULTS,
                    "offset": 0,
                    "hasMore": True,
                },
            )
        if request.url.path.endswith("/documents:get"):
            request_counts["get"] += 1
            document_id = json.loads(request.read())["id"]
            return httpx.Response(
                200,
                json={
                    "id": document_id,
                    "path": f"{document_id}.md",
                    "collection": "kojutsu",
                    "content": "answer",
                    "frontmatter": {},
                },
            )
        return httpx.Response(404)

    client = make_client(handler)
    documents = client.search_documents("x", limit=1000)

    assert request_counts == {"search": 1, "get": MAX_SEARCH_RESULTS}
    assert search_limit == MAX_SEARCH_RESULTS
    assert len(documents) == MAX_SEARCH_RESULTS
