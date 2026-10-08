"""Tests for the read-only pull request date range, and for what it refuses.

Every test here runs against :class:`RecordingTransport` and never the network.
The recording is not incidental: the ticket's central claim is that this seam
cannot write to GitHub, and a claim about a property of a run is only worth
anything if the run is what is inspected. So the transport records the method of
every request that passes through it, and
:func:`test_a_backfill_run_issues_no_mutating_request` fails the build if any
method other than ``GET`` appears. That test is the reason to believe the rest of
this file's claim to read-only; without it, read-only is a comment.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from kojutsu.cli import app
from kojutsu.config import Settings
from kojutsu.core.backfill import run_backfill
from kojutsu.core.question_registry import (
    SqliteQuestionRegistry,
    stable_answer_entry_id,
)
from kojutsu.integrations.github import (
    DEFAULT_SEARCH_PACE_SECONDS,
    MAX_RANGE_SPAN_DAYS,
    SEARCH_RESULT_CAP,
    GitHubClient,
    GitHubRangeError,
    GitHubRateLimitError,
    GitHubRepositoryNotAllowedError,
    answer_comment_body,
    kojutsu_comment_body,
    rationale_comment_body_as_agent,
    validate_pull_request_range,
)

#: HTTP verbs that change state on the forge. Listing them rather than naming the
#: safe ones is deliberate: a new verb added to this seam fails the test, whereas
#: an allow-list of read verbs would quietly accept one nobody thought about.
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

REPO = "org/repo"
SINCE = "2024-03-10"
UNTIL = "2024-03-20"


def pr_hit(number: int, created: str, **overrides: Any) -> dict[str, Any]:
    """One search hit, shaped the way ``GET /search/issues`` returns it.

    Built deliberately without ``head`` or ``base``: a search hit is an issue
    object and does not carry the diff-side fields a pull request does, so a
    fixture that included them would be testing a shape GitHub never sends.
    """
    hit: dict[str, Any] = {
        "number": number,
        "title": f"PR {number}",
        "state": "closed",
        "user": {"login": "dev"},
        "created_at": created,
        "closed_at": created,
        "updated_at": created,
    }
    hit.update(overrides)
    return hit


def comment_hit(comment_id: int, body: str, author: str, association: str) -> dict[str, Any]:
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": author},
        "created_at": "2024-03-12T10:00:00Z",
        "author_association": association,
    }


class RecordingTransport:
    """A stub that answers from a fixture and remembers every method it saw.

    ``mutations`` is the whole point: a review can read this class and see that a
    write is *possible*, because the transport is happy to answer a POST. The
    tests below assert none is ever asked to.
    """

    def __init__(
        self,
        *,
        hits: list[dict[str, Any]] | None = None,
        total_count: int | None = None,
        incomplete_results: bool = False,
        comments: dict[int, list[dict[str, Any]]] | None = None,
        reviews: dict[int, list[dict[str, Any]]] | None = None,
        review_comments: dict[int, list[dict[str, Any]]] | None = None,
        page_of: Any = None,
        rate_limit_on: str | None = None,
        fail_comment_prs: frozenset[int] = frozenset(),
    ) -> None:
        self.hits = list(hits or [])
        self.total_count = len(self.hits) if total_count is None else total_count
        self.incomplete_results = incomplete_results
        self.comments = comments or {}
        self.reviews = reviews or {}
        self.review_comments = review_comments or {}
        #: Callable ``(page, per_page) -> list[hit]`` overriding the default
        #: slicing, for fixtures that need a page to come back empty or short.
        self.page_of = page_of
        self.rate_limit_on = rate_limit_on
        self.fail_comment_prs = fail_comment_prs
        self.calls: list[tuple[str, str]] = []
        self.search_params_seen: list[dict[str, str]] = []

    @property
    def mutations(self) -> list[tuple[str, str]]:
        """Every recorded request that would change state if it were real."""
        return [call for call in self.calls if call[0] in MUTATING_METHODS]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/search/issues":
            return self._search(request)
        # Issue comments and inline review comments are different objects on
        # different paths, and both end in "/comments". Routing them apart by
        # suffix alone would answer a review request with the thread, which is
        # the kind of fixture that lets a real bug pass.
        if "/issues/" in path and path.endswith("/comments"):
            number = int(path.split("/issues/")[1].split("/")[0])
            if number in self.fail_comment_prs:
                return httpx.Response(500, json={"message": "boom"})
            return httpx.Response(200, json=self._listing(request, self.comments.get(number, [])))
        if "/pulls/" in path and path.endswith("/comments"):
            number = int(path.split("/pulls/")[1].split("/")[0])
            return httpx.Response(
                200, json=self._listing(request, self.review_comments.get(number, []))
            )
        if "/pulls/" in path and path.endswith("/reviews"):
            number = int(path.split("/pulls/")[1].split("/")[0])
            return httpx.Response(200, json=self._listing(request, self.reviews.get(number, [])))
        if path.startswith("/repos/"):
            return httpx.Response(404, json={"message": "not found"})
        return httpx.Response(404, json={"message": f"unstubbed {path}"})

    @staticmethod
    def _listing(request: httpx.Request, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serve one page the way the forge does: short final page ends the walk.

        A fixture that returned the whole list on every page would send the
        client paging to the 100-page ceiling on any full first page -- which is
        precisely the truncation behaviour under test, not the common case.
        """
        params = dict(request.url.params)
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "100"))
        start = (page - 1) * per_page
        return items[start : start + per_page]

    def _search(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.search_params_seen.append(params)
        if self.rate_limit_on == "search":
            return httpx.Response(
                403,
                json={"message": "API rate limit exceeded for user ID 1."},
                headers={
                    "x-ratelimit-limit": "30",
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-resource": "search",
                    "x-ratelimit-reset": "0",
                },
            )
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "30"))
        if self.page_of is not None:
            items = list(self.page_of(page, per_page))
        else:
            start = (page - 1) * per_page
            items = self.hits[start : start + per_page]
        return httpx.Response(
            200,
            json={
                "total_count": self.total_count,
                "incomplete_results": self.incomplete_results,
                "items": items,
            },
        )


def patch_http(monkeypatch: pytest.MonkeyPatch, transport: RecordingTransport) -> None:
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(transport))

    monkeypatch.setattr(httpx, "Client", factory)


def make_client(**kwargs: Any) -> GitHubClient:
    """A client that does not sleep, so pacing is asserted rather than waited out."""
    kwargs.setdefault("search_pace_seconds", 0.0)
    kwargs.setdefault("sleep", lambda _seconds: None)
    return GitHubClient("tok", **kwargs)


def settings_for(*repositories: str) -> Settings:
    return Settings(github_webhook_allowed_repositories=",".join(repositories))


#: Pull requests on both sides of both boundaries, plus one either side of the
#: whole window. The two extreme instants of each boundary day are the cases that
#: separate an inclusive range from a half-open one, and a boundary test that
#: only used midnight would pass against a reader that was off by a day.
BOUNDARY_HITS = [
    pr_hit(1, "2024-03-09T23:59:59Z"),  # one second before the window
    pr_hit(2, "2024-03-10T00:00:00Z"),  # first instant of the since day
    pr_hit(3, "2024-03-10T23:59:59Z"),  # last instant of the since day
    pr_hit(4, "2024-03-15T12:00:00Z"),  # comfortably inside
    pr_hit(5, "2024-03-20T00:00:00Z"),  # first instant of the until day
    pr_hit(6, "2024-03-20T23:59:59Z"),  # last instant of the until day
    pr_hit(7, "2024-03-21T00:00:00Z"),  # one second after the window
]

IN_RANGE_NUMBERS = [2, 3, 4, 5, 6]


def _page(first: int, count: int) -> list[dict[str, Any]]:
    """``count`` search hits numbered from ``first``, all created mid-range."""
    return [pr_hit(number, "2024-03-15T00:00:00Z") for number in range(first, first + count)]


# -- boundaries --------------------------------------------------------------


def test_only_pull_requests_created_inside_the_range_are_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert result.numbers == IN_RANGE_NUMBERS
    assert result.out_of_range == 2
    # The index returned everything it was asked for and the range filter is what
    # removed two, so this is a complete read of the index -- not a short one.
    assert result.truncated is False
    assert result.hits_seen == len(BOUNDARY_HITS)
    assert result.total_count == len(BOUNDARY_HITS)


def test_the_range_is_inclusive_on_both_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both boundary days are fully in range, not just their midnight.

    A range expressed as a pair of days means days. ``since=2024-03-10`` covers
    all of the tenth, and a reader that stopped at its first instant would drop a
    whole day of pull requests without saying so.
    """
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    kept = {pull.number: pull.created_at for pull in result.pull_requests}
    assert kept[2].date() == date(2024, 3, 10)
    assert kept[6].date() == date(2024, 3, 20)
    # The last instant of the final day is inside; the first instant of the next
    # day is not. One second separates them, so this is not a rounding artefact.
    assert kept[6] == datetime(2024, 3, 20, 23, 59, 59, tzinfo=UTC)
    assert 7 not in kept


def test_a_single_day_range_keeps_that_day(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=SINCE, settings=settings_for(REPO)
    )

    assert result.numbers == [2, 3]
    assert result.out_of_range == 5


def test_the_query_carries_the_range_and_pull_request_qualifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert transport.search_params_seen[0]["q"] == (f"repo:{REPO} is:pr created:{SINCE}..{UNTIL}")
    # A total order is what makes a paginated walk over a shifting index
    # repeatable. Without it, page two can repeat or skip an item from page one.
    assert transport.search_params_seen[0]["sort"] == "created"
    assert transport.search_params_seen[0]["order"] == "asc"


def test_a_hit_with_no_creation_date_is_excluded_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    undated = pr_hit(9, "2024-03-12T00:00:00Z")
    del undated["created_at"]
    transport = RecordingTransport(hits=[pr_hit(4, "2024-03-15T12:00:00Z"), undated])
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert result.numbers == [4]
    # Counted, not silently dropped: a hit that cannot be shown to be in range is
    # a hole in the coverage, and a hole nobody counted is a hole nobody sees.
    assert result.undated == 1
    assert result.out_of_range == 0


def test_hits_map_onto_the_existing_pull_request_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search hit is a ``GitHubPullRequest``, not a new parallel model.

    Asserted by type and by field, because a second shape would mean a second set
    of rules for what a pull request is -- and the two would drift.
    """
    from kojutsu.integrations.github_models import GitHubPullRequest

    transport = RecordingTransport(
        hits=[pr_hit(4, "2024-03-15T12:00:00Z", title="Rotate the credential")]
    )
    patch_http(monkeypatch, transport)

    pull = (
        make_client()
        .search_pull_requests("org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO))
        .pull_requests[0]
    )

    assert isinstance(pull, GitHubPullRequest)
    assert pull.number == 4
    assert pull.title == "Rotate the credential"
    assert pull.state == "closed"
    assert pull.user is not None and pull.user.login == "dev"
    assert pull.created_at == datetime(2024, 3, 15, 12, tzinfo=UTC)
    # A search hit carries no diff-side fields, and a merged pull request reports
    # no merge through this endpoint, so both stay honestly absent.
    assert pull.head is None
    assert pull.merged_at is None


# -- the cap, and reporting a short read -------------------------------------


def test_a_range_past_the_cap_returns_the_cap_and_reports_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap is enforced here, because GitHub enforces it silently."""
    everything = [pr_hit(number, "2024-03-15T00:00:00Z") for number in range(1, 1201)]
    transport = RecordingTransport(hits=everything)
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, limit=5000, settings=settings_for(REPO)
    )

    assert len(result.pull_requests) == SEARCH_RESULT_CAP
    assert result.truncated is True
    assert result.total_count == 1200
    assert result.hits_seen == SEARCH_RESULT_CAP
    # Ten pages of a hundred is the whole of the cap; the walk stops there
    # rather than issuing a request that cannot return anything new.
    assert result.pages_fetched == SEARCH_RESULT_CAP // 100


def test_a_caller_limit_below_the_cap_also_reports_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(
        hits=[pr_hit(number, "2024-03-15T00:00:00Z") for number in range(1, 8)]
    )
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, limit=2, settings=settings_for(REPO)
    )

    assert result.numbers == [1, 2]
    assert result.truncated is True
    assert result.total_count == 7


def test_github_saying_it_gave_up_is_reported_as_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``incomplete_results`` is GitHub admitting it could not finish counting."""
    transport = RecordingTransport(
        hits=[pr_hit(4, "2024-03-15T12:00:00Z")], incomplete_results=True
    )
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert result.numbers == [4]
    assert result.truncated is True
    assert result.hits_seen == result.total_count


def test_a_fully_enumerated_range_is_not_reported_as_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert result.truncated is False


# -- pagination --------------------------------------------------------------


def test_a_page_that_is_empty_mid_sequence_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty page ends the walk instead of looping or raising.

    GitHub returns one when the index shifts under a paginated read. Asking again
    cannot help, so the walk stops -- and reports the shortfall, because stopping
    early is exactly the case ``truncated`` exists to describe.
    """
    first_page = _page(1, 100)
    pages: dict[int, list[dict[str, Any]]] = {1: first_page, 2: [], 3: first_page}
    transport = RecordingTransport(
        hits=[],
        total_count=250,
        page_of=lambda page, _per_page: pages.get(page, []),
    )
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, limit=200, settings=settings_for(REPO)
    )

    assert result.numbers == list(range(1, 101))
    assert result.pages_fetched == 2
    # The third page was never asked for; page two was simply empty. Continuing
    # would either loop on an empty page or walk off the end of the index.
    assert [params["page"] for params in transport.search_params_seen] == ["1", "2"]
    assert result.truncated is True


def test_pages_walk_in_order_and_stop_on_a_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages: dict[int, list[dict[str, Any]]] = {1: _page(1, 100), 2: _page(101, 5)}
    transport = RecordingTransport(
        hits=[],
        total_count=105,
        page_of=lambda page, _per_page: pages.get(page, []),
    )
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, limit=200, settings=settings_for(REPO)
    )

    assert result.numbers == list(range(1, 106))
    assert result.pages_fetched == 2
    # Every hit the index offered was retrieved, so this is a whole read.
    assert result.truncated is False
    assert [params["page"] for params in transport.search_params_seen] == ["1", "2"]


def test_a_caller_limit_below_one_page_still_asks_but_keeps_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``total_count`` is the one thing a caller asking for nothing cannot get.

    The request is still issued, because "how many are there" is the fact a
    planning run needs, and the result says it truncated rather than returning an
    empty list that reads as an empty range.
    """
    transport = RecordingTransport(hits=BOUNDARY_HITS, total_count=len(BOUNDARY_HITS))
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, limit=0, settings=settings_for(REPO)
    )

    assert result.pull_requests == []
    assert result.total_count == len(BOUNDARY_HITS)
    assert result.truncated is True


def test_an_empty_first_page_is_an_empty_range_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=[], total_count=0)
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert result.pull_requests == []
    assert result.pages_fetched == 1
    assert result.truncated is False


def test_requests_are_paced_against_the_search_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search allows 30 requests a minute, and a naive walk ignores that.

    Ten pages is a third of the budget. Unpaced, a real backfill of a real range
    is a 403 partway through -- the kind of failure that looks like an empty
    range rather than like a limit.
    """
    slept: list[float] = []
    transport = RecordingTransport(
        hits=[pr_hit(number, "2024-03-15T00:00:00Z") for number in range(1, 1201)]
    )
    patch_http(monkeypatch, transport)

    client = GitHubClient(
        "tok",
        search_pace_seconds=DEFAULT_SEARCH_PACE_SECONDS,
        sleep=slept.append,
    )
    result = client.search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, limit=5000, settings=settings_for(REPO)
    )

    assert result.pages_fetched == 10
    # One wait fewer than the number of requests: the first request is not late.
    assert len(slept) == 9
    assert all(0 < pause <= DEFAULT_SEARCH_PACE_SECONDS for pause in slept)
    assert pytest.approx(2.0) == DEFAULT_SEARCH_PACE_SECONDS


def test_no_wait_is_taken_before_the_first_request(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    transport = RecordingTransport(hits=[pr_hit(4, "2024-03-15T12:00:00Z")])
    patch_http(monkeypatch, transport)

    GitHubClient("tok", search_pace_seconds=5.0, sleep=slept.append).search_pull_requests(
        "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
    )

    assert slept == []


# -- refusals ----------------------------------------------------------------


def test_an_inverted_range_is_refused_naming_both_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRangeError) as excinfo:
        make_client().search_pull_requests(
            "org", "repo", since=UNTIL, until=SINCE, settings=settings_for(REPO)
        )

    message = str(excinfo.value)
    assert "since" in message
    assert "until" in message
    # An inverted range matches nothing on a perfectly healthy index, so a caller
    # reading "zero results" would conclude the range was empty.
    assert "inverted" in message
    assert transport.calls == []


@pytest.mark.parametrize(
    ("since", "until", "faulty_field"),
    [
        ("not-a-date", UNTIL, "since"),
        (SINCE, "2024-13-45", "until"),
        (SINCE, "", "until"),
        ("2024-03-10T00:00:00Z", UNTIL, "since"),
        ("20240310", UNTIL, "since"),
    ],
)
def test_a_malformed_range_is_refused_naming_the_field_at_fault(
    monkeypatch: pytest.MonkeyPatch,
    since: str,
    until: str,
    faulty_field: str,
) -> None:
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRangeError) as excinfo:
        make_client().search_pull_requests(
            "org", "repo", since=since, until=until, settings=settings_for(REPO)
        )

    message = str(excinfo.value)
    assert message.startswith(faulty_field)
    assert "YYYY-MM-DD" in message
    assert transport.calls == []


def test_a_range_wider_than_the_stated_bound_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRangeError) as excinfo:
        make_client().search_pull_requests(
            "org",
            "repo",
            since="2020-01-01",
            until="2024-12-31",
            settings=settings_for(REPO),
        )

    message = str(excinfo.value)
    assert str(MAX_RANGE_SPAN_DAYS) in message
    assert str(SEARCH_RESULT_CAP) in message
    assert transport.calls == []


def test_a_range_exactly_at_the_bound_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)
    start = date(2023, 1, 1)
    end = date.fromordinal(start.toordinal() + MAX_RANGE_SPAN_DAYS)

    result = make_client().search_pull_requests(
        "org",
        "repo",
        since=start.isoformat(),
        until=end.isoformat(),
        settings=settings_for(REPO),
    )

    assert result.pull_requests == []
    assert transport.calls[0][1] == "/search/issues"


def test_a_repository_outside_the_allowlist_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRepositoryNotAllowedError) as excinfo:
        make_client().search_pull_requests(
            "other", "elsewhere", since=SINCE, until=UNTIL, settings=settings_for(REPO)
        )

    assert "GITHUB_WEBHOOK_ALLOWED_REPOSITORIES" in str(excinfo.value)
    # The point of checking first: an allowlist enforced after the request has
    # been sent has already told GitHub what the caller was looking for.
    assert transport.calls == []


def test_an_empty_allowlist_denies_every_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRepositoryNotAllowedError):
        make_client().search_pull_requests(
            "org", "repo", since=SINCE, until=UNTIL, settings=settings_for("")
        )

    assert transport.calls == []


def test_a_wildcard_allowlist_does_not_authorise_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wildcard narrows to nothing. It is never authorisation."""
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRepositoryNotAllowedError):
        make_client().search_pull_requests(
            "org", "repo", since=SINCE, until=UNTIL, settings=settings_for("*")
        )

    assert transport.calls == []


def test_the_allowlist_folds_case_the_way_github_defines_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    result = make_client().search_pull_requests(
        "Org", "Repo", since=SINCE, until=UNTIL, settings=settings_for("ORG/REPO")
    )

    assert result.pull_requests == []
    assert transport.calls[0][1] == "/search/issues"


def test_the_allowlist_check_comes_before_the_range_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authorisation is about whether a caller may ask at all.

    Refusing an unauthorised caller should not also tell them which half of their
    range was wrong.
    """
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRepositoryNotAllowedError):
        make_client().search_pull_requests(
            "other",
            "elsewhere",
            since="nonsense",
            until="also-nonsense",
            settings=settings_for(REPO),
        )


def test_the_range_validator_is_usable_on_its_own() -> None:
    start, end = validate_pull_request_range("2024-03-10", date(2024, 3, 20))
    assert (start, end) == (date(2024, 3, 10), date(2024, 3, 20))

    with pytest.raises(GitHubRangeError, match="until"):
        validate_pull_request_range("2024-03-10", "nonsense")


# -- rate limiting -----------------------------------------------------------


def test_a_rate_limited_search_is_surfaced_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 on the search budget is raised, not absorbed into a short range.

    Swallowing it would produce the worst outcome available: a run that reports
    "no pull requests in this range" when the range is full of them. Nothing
    downstream could tell that from a genuinely empty range.
    """
    transport = RecordingTransport(hits=BOUNDARY_HITS, rate_limit_on="search")
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRateLimitError) as excinfo:
        make_client().search_pull_requests(
            "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
        )

    error = excinfo.value
    assert error.retryable is True
    assert error.resource == "search"
    assert "search" in str(error)


def test_a_rate_limit_is_not_mistaken_for_a_scope_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare 403 must not become "your token is too broad"."""
    from kojutsu.integrations.github import GitHubTokenScopeError

    transport = RecordingTransport(rate_limit_on="search")
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRateLimitError):
        make_client().search_pull_requests(
            "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
        )

    # The scope assertion is not reached at all, so it cannot have fired.
    assert not issubclass(GitHubRateLimitError, GitHubTokenScopeError)


def test_a_permissions_forbidden_is_not_reported_as_a_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 that is not a rate limit stays a plain status error.

    The discrimination matters in both directions. Mistaking a permissions
    failure for a rate limit tells an operator to wait for a window that will
    never help, and mistaking a rate limit for a permissions failure tells them to
    re-scope a token that was already correct.
    """
    calls: list[str] = []

    def forbidden(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(403, json={"message": "Resource not accessible by integration"})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda *a, **k: real_client(transport=httpx.MockTransport(forbidden))
    )

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        make_client().search_pull_requests(
            "org", "repo", since=SINCE, until=UNTIL, settings=settings_for(REPO)
        )

    assert isinstance(excinfo.value, httpx.HTTPStatusError)
    assert not isinstance(excinfo.value, GitHubRateLimitError)
    assert excinfo.value.response.status_code == 403
    assert calls == ["GET"]


def test_a_rate_limited_backfill_run_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=BOUNDARY_HITS, rate_limit_on="search")
    patch_http(monkeypatch, transport)

    with pytest.raises(GitHubRateLimitError):
        run_backfill(
            make_client(),
            registry,
            _sink(),
            repository=REPO,
            since=SINCE,
            until=UNTIL,
            settings=settings_for(REPO),
        )

    assert transport.mutations == []


# -- the run, and its record -------------------------------------------------


class _Sink:
    """A knowledge sink that keeps what it was given, in order."""

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def store(self, entry: Any) -> None:
        self.entries.append(entry)


def _sink() -> _Sink:
    return _Sink()


def _question_and_answer() -> dict[int, list[dict[str, Any]]]:
    return {
        2: [
            comment_hit(500, kojutsu_comment_body("q1", "Why here?"), "bot", "NONE"),
            comment_hit(
                501,
                answer_comment_body("q1", "Because the index was the bottleneck."),
                "dev",
                "MEMBER",
            ),
        ],
    }


def _recorded_question(registry: SqliteQuestionRegistry) -> None:
    registry.record_question(
        question_id="q1",
        github_comment_id=500,
        repo=REPO,
        pr_number=2,
        pr_url=f"https://github.com/{REPO}/pull/2",
        question_text="Why here?",
        question_category="design_decision",
        question_author="bot",
    )


def test_a_backfill_run_reads_a_range_and_collects_from_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments=_question_and_answer())
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    # The two out-of-range hits are never even read, which is the point of the
    # range: a backfill over a quarter does not fetch a quarter.
    assert coverage.pull_requests == 5
    assert coverage.prs_read == 5
    assert coverage.out_of_range == 2
    assert coverage.comments_read == 2
    assert coverage.answers_captured == 1
    assert coverage.complete is True
    assert coverage.since == SINCE
    assert coverage.until == UNTIL
    assert [entry.metadata["github_comment_id"] for entry in sink.entries] == [501]
    # One search page, then per pull request: the thread, and the review list.
    # Inline review comments are only fetched for a pull request that actually has
    # reviews, so a pull request with none costs two calls rather than three.
    assert len(transport.calls) == 11
    assert transport.calls.count(("GET", "/repos/org/repo/pulls/2/reviews")) == 1
    assert all(method == "GET" for method, _ in transport.calls)


def test_a_backfill_run_issues_no_mutating_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The read-only claim, checked rather than asserted.

    A full run -- enumerate the range, read every pull request's comments, store
    what the collectors find -- with every HTTP method recorded. If this path ever
    grows a POST, this fails. That is worth more than any number of comments
    saying it does not have one, because a comment cannot fail a build.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    comments = _question_and_answer()
    comments[3] = [
        comment_hit(
            600,
            rationale_comment_body_as_agent(
                "Picked the interval over polling.", "opencode", branch="feat/backoff"
            ),
            "kojutsu-bot",
            "COLLABORATOR",
        )
    ]
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments=comments)
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert transport.calls, "the run issued no requests at all, so nothing was proved"
    assert transport.mutations == []
    assert {method for method, _path in transport.calls} == {"GET"}
    # The collectors really did run; an empty run would pass the assertion above
    # without having exercised the path that could have written.
    assert coverage.answers_captured == 1
    assert coverage.rationales_captured == 1


def test_the_same_range_run_twice_stores_nothing_new(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resumable and idempotent, because identity is the comment.

    The second run re-reads the whole range and stores nothing. This is not a
    special case in the backfill: it is the registry claim the webhook relies on,
    which is the only reason a half-finished run is safe to finish by running the
    same range again.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments=_question_and_answer())
    patch_http(monkeypatch, transport)
    client = make_client()

    first = run_backfill(
        client,
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )
    second = run_backfill(
        client,
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert first.answers_captured == 1
    assert second.answers_captured == 0
    # Both runs report the same coverage. The second reports no *new* records,
    # not a different range: an operator re-running after a failure needs to see
    # that the range was still whole.
    assert second.pull_requests == first.pull_requests
    assert second.comments_read == first.comments_read
    assert len(sink.entries) == 1
    assert sink.entries[0].entry_id == stable_answer_entry_id(REPO, 2, 501)


def test_an_overlapping_range_does_not_duplicate_the_pull_request_it_shares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two ranges that overlap by one pull request agree about that overlap."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments=_question_and_answer())
    patch_http(monkeypatch, transport)
    client = make_client()

    run_backfill(
        client,
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until="2024-03-14",
        settings=settings_for(REPO),
    )
    run_backfill(
        client,
        registry,
        sink,
        repository=REPO,
        since="2024-03-14",
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert len(sink.entries) == 1


def test_a_pull_request_that_cannot_be_read_does_not_end_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run that dies on the twelfth of forty is a run that cannot be resumed."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    transport = RecordingTransport(
        hits=BOUNDARY_HITS, comments=_question_and_answer(), fail_comment_prs=frozenset({3})
    )
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert coverage.prs_read == 4
    assert coverage.prs_failed == 1
    assert coverage.answers_captured == 1
    # An incomplete run is not a complete one, and says so by name.
    assert coverage.complete is False
    assert "org/repo#3" in coverage.errors[0]
    assert "could not be read" in coverage.summary()


def test_a_pull_request_with_nothing_on_it_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments={})
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        _sink(),
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert coverage.prs_failed == 0
    assert coverage.comments_read == 0
    assert coverage.complete is True


def test_the_coverage_record_says_when_a_range_was_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bounded answer must not read as a complete one."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(
        hits=[pr_hit(number, "2024-03-15T00:00:00Z") for number in range(1, 8)],
        comments={},
    )
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        _sink(),
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        limit=2,
        settings=settings_for(REPO),
    )

    assert coverage.truncated is True
    assert coverage.complete is False
    assert "INCOMPLETE RANGE" in coverage.summary()
    assert coverage.as_dict()["search_total_count"] == 7
    assert coverage.as_dict()["truncated"] is True
    # JSON-serialisable, so a --report file is not a second thing to go wrong.
    assert json.loads(json.dumps(coverage.as_dict())) == coverage.as_dict()


def test_a_pull_request_past_the_comment_page_cap_is_reported_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ten thousand comments read is a floor, not a total, when the walk stopped
    at the page ceiling: the coverage must say so rather than report the count
    as complete."""
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(
        hits=[pr_hit(7, "2024-03-15T00:00:00Z")],
        comments={
            7: [
                comment_hit(10_000 + index, "just a comment", "dev", "MEMBER")
                for index in range(10_000)
            ]
        },
    )
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        _sink(),
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert coverage.prs_read == 1
    assert coverage.prs_failed == 0
    assert coverage.comments_read == 10_000
    assert coverage.prs_truncated == 1
    assert coverage.complete is False
    assert "page cap" in coverage.summary()
    assert coverage.as_dict()["prs_truncated"] == 1
    comment_calls = [call for call in transport.calls if call[1].endswith("/comments")]
    assert len(comment_calls) == 100, "the walk stops at the ceiling, not past it"


def test_the_coverage_record_reports_hits_the_index_and_data_disagree_about(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments={})
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        _sink(),
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    # The index matched seven; two of them are not in the range by their own
    # creation date. That is a different fact from a truncated read and the
    # record has to keep them apart.
    assert coverage.out_of_range == 2
    assert coverage.truncated is False
    assert "own creation date" in coverage.summary()


def test_an_undated_hit_is_reported_rather_than_collected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    undated = pr_hit(9, "2024-03-12T00:00:00Z")
    del undated["created_at"]
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=[pr_hit(4, "2024-03-15T12:00:00Z"), undated], comments={})
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        _sink(),
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert coverage.pull_requests == 1
    assert coverage.undated == 1
    assert "no creation date" in coverage.summary()


def test_a_repository_that_is_not_owner_slash_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=[])
    patch_http(monkeypatch, transport)

    with pytest.raises(ValueError, match="owner/name"):
        run_backfill(
            make_client(),
            registry,
            _sink(),
            repository="just-a-name",
            since=SINCE,
            until=UNTIL,
            settings=settings_for(REPO),
        )


# -- the CLI surface ---------------------------------------------------------


class _FakeRuntime:
    def __init__(self, registry: SqliteQuestionRegistry, sink: _Sink) -> None:
        self.registry = registry
        self.sink = sink

    def __enter__(self) -> _FakeRuntime:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _invoke_backfill(
    monkeypatch: pytest.MonkeyPatch,
    registry: SqliteQuestionRegistry,
    sink: _Sink,
    transport: RecordingTransport,
    *args: str,
) -> Any:
    from kojutsu import cli_backfill as cli_module

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    patch_http(monkeypatch, transport)
    monkeypatch.setattr(cli_module, "build_runtime", lambda _settings: _FakeRuntime(registry, sink))
    return CliRunner().invoke(
        app, ["backfill", "--repo", REPO, "--since", SINCE, "--until", UNTIL, *args]
    )


def test_the_backfill_command_collects_and_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments=_question_and_answer())

    result = _invoke_backfill(monkeypatch, registry, sink, transport)

    assert result.exit_code == 0
    assert "1 answer(s)" in result.output
    assert "INCOMPLETE RANGE" not in result.output
    assert transport.mutations == []


def test_the_backfill_command_writes_a_coverage_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    sink = _sink()
    transport = RecordingTransport(hits=BOUNDARY_HITS, comments={})
    report = tmp_path / "coverage.json"

    result = _invoke_backfill(monkeypatch, registry, sink, transport, "--report", str(report))

    assert result.exit_code == 0
    recorded = json.loads(report.read_text(encoding="utf-8"))
    assert recorded["repository"] == REPO
    assert recorded["since"] == SINCE
    assert recorded["until"] == UNTIL
    assert recorded["pull_requests"] == 5
    assert recorded["truncated"] is False
    assert recorded["complete"] is True


def test_a_truncated_backfill_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A truncation is a result to act on, not a success.

    Exit 0 is what a script looks for, and a script that re-runs on a short read
    is how a partial range gets mistaken for a whole one.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    sink = _sink()
    transport = RecordingTransport(
        hits=[pr_hit(number, "2024-03-15T00:00:00Z") for number in range(1, 8)], comments={}
    )

    result = _invoke_backfill(monkeypatch, registry, sink, transport, "--limit", "2")

    assert result.exit_code == 2
    assert "INCOMPLETE RANGE" in result.output


def test_the_backfill_command_refuses_an_inverted_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kojutsu import cli_backfill as cli_module

    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    sink = _sink()
    transport = RecordingTransport(hits=BOUNDARY_HITS)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    patch_http(monkeypatch, transport)
    monkeypatch.setattr(cli_module, "build_runtime", lambda _settings: _FakeRuntime(registry, sink))

    result = CliRunner().invoke(
        app, ["backfill", "--repo", REPO, "--since", UNTIL, "--until", SINCE]
    )

    assert result.exit_code == 1
    assert "since" in result.output
    assert transport.calls == []


def test_the_backfill_command_refuses_a_repository_outside_the_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kojutsu import cli_backfill as cli_module

    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=[])
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", "some/other")
    patch_http(monkeypatch, transport)
    monkeypatch.setattr(
        cli_module, "build_runtime", lambda _settings: _FakeRuntime(registry, _sink())
    )

    result = CliRunner().invoke(
        app, ["backfill", "--repo", REPO, "--since", SINCE, "--until", UNTIL]
    )

    assert result.exit_code == 1
    assert "GITHUB_WEBHOOK_ALLOWED_REPOSITORIES" in result.output
    assert transport.calls == []


def test_the_backfill_command_names_a_rate_limit_as_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kojutsu import cli_backfill as cli_module

    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    transport = RecordingTransport(hits=[], rate_limit_on="search")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    patch_http(monkeypatch, transport)
    monkeypatch.setattr(
        cli_module, "build_runtime", lambda _settings: _FakeRuntime(registry, _sink())
    )

    result = CliRunner().invoke(
        app, ["backfill", "--repo", REPO, "--since", SINCE, "--until", UNTIL]
    )

    assert result.exit_code == 1
    assert "retryable" in result.output
    assert "rate limit" in result.output


def test_the_backfill_command_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = CliRunner().invoke(
        app, ["backfill", "--repo", REPO, "--since", SINCE, "--until", UNTIL]
    )

    assert result.exit_code == 1
    assert "GITHUB_TOKEN" in result.output


def test_the_backfill_command_has_no_apply_flag() -> None:
    """There is no ``--apply``, and that is the control.

    An ingestion path that can write needs a flag, a review, and an audit of who
    passed it. This one has none of those, so there is nothing to review and
    nothing to audit. Checked against the command's own parameters rather than its
    help text, because the help text *mentions* the flag in order to say it is
    absent -- and a test that greps for a word the documentation uses is a test
    that will be "fixed" by deleting the sentence.
    """
    command = get_command(app).commands["backfill"]
    option_names = {option for parameter in command.params for option in parameter.opts}

    assert "--apply" not in option_names
    assert "--dry-run" not in option_names
    assert option_names == {"--repo", "--since", "--until", "--limit", "-n", "--report"}


def _one_review() -> dict[int, list[dict[str, Any]]]:
    """A pull request carrying a review verdict and nothing else."""
    return {
        2: [
            {
                "id": 9001,
                "state": "APPROVED",
                "body": "Ship it: the retry is bounded and the test pins it.",
                "user": {"login": "reviewer"},
                "submitted_at": "2024-05-02T10:00:00Z",
                "author_association": "MEMBER",
            }
        ]
    }


def test_a_backfill_run_captures_a_review_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pull request with a review on it is the case the suite never ran.

    Every other backfill test here fetched ``/reviews`` and got an empty list back, so
    the loop body that consumes a review never executed. That is why a return-type
    change could break review capture completely while the suite stayed green:
    ``process_review_event_outcome`` began returning ``ReviewCaptureResult`` instead
    of a list, ``backfill.py`` kept calling ``len()`` on it, and every backfill died
    with ``TypeError: object of type 'ReviewCaptureResult' has no len()`` on the first
    real review it met. Found by backfilling a real repository, not by any test.

    The assertion is that a review is stored, not merely that nothing raised: a
    ``len`` that returned zero would satisfy a smoke test just as comfortably.
    """
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _recorded_question(registry)
    sink = _sink()
    transport = RecordingTransport(
        hits=BOUNDARY_HITS, comments=_question_and_answer(), reviews=_one_review()
    )
    patch_http(monkeypatch, transport)

    coverage = run_backfill(
        make_client(),
        registry,
        sink,
        repository=REPO,
        since=SINCE,
        until=UNTIL,
        settings=settings_for(REPO),
    )

    assert transport.calls.count(("GET", "/repos/org/repo/pulls/2/reviews")) == 1
    assert coverage.reviews_captured == 1, "a review verdict was fetched and then dropped"
    review_records = [
        entry for entry in sink.entries if "review_state_approved" in (entry.tags or [])
    ]
    assert len(review_records) == 1
    assert "Ship it" in review_records[0].answer_text
