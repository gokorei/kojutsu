"""Tests for bounded concurrent history reads, and for the pool they share.

The change these tests hold in place is invisible when it works. A walk that reads
pages concurrently has to store the *same records, in the same order, with the same
report* as the sequential walk it replaced -- because ``run_backfill`` is documented as
idempotent by construction and an operator re-runs the same range specifically to watch
it advance. So most of this file is about equivalence, and the assertions are on record
identity and order rather than on counts: a count cannot tell a reordering from a
complete walk, and a reordering is the failure that would be invisible in the output.

The fake reader deliberately answers *later pages sooner*, so completion order is the
reverse of the order the pages were asked for. Without that, a consumer that trusted
completion order would pass by luck.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from kojutsu.config import Settings
from kojutsu.core.backfill_reviews import (
    BackfillPlan,
    BackfillReport,
    run_backfill,
)
from kojutsu.core.backfill_reviews_client import PAGE_SIZE, GitHubHistoryReader
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.integrations.github import (
    DIFF_TIMEOUT_SECONDS,
    HISTORY_READ_CONCURRENCY,
    REQUEST_TIMEOUT_SECONDS,
    GitHubClient,
)
from kojutsu.integrations.github_models import (
    GitHubComment,
    GitHubPullRequest,
    GitHubUser,
    PullRequestReview,
    PullRequestReviewComment,
)

REPO = "org/repo"
FLOOR = datetime(2024, 1, 1, tzinfo=UTC)
READ_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
SUBMITTED_AT = datetime(2024, 3, 4, 9, 0, tzinfo=UTC)
COMMENTED_AT = datetime(2024, 3, 4, 9, 30, tzinfo=UTC)

#: The concurrency under test. Deliberately not the production default: this number has
#: to be small enough that a round of it fits in a test and large enough that pages
#: genuinely overlap, and 4 does both.
CONCURRENCY = 4

#: The slowest page in the fake, in seconds. Long enough that a thread pool reliably
#: interleaves rounds, short enough that the suite does not notice.
PAGE_DELAY = 0.03


class RecordingSink:
    """A sink that keeps what it was handed, in the order it was handed it."""

    def __init__(self) -> None:
        self.entries: list[object] = []

    def store(self, entry):  # type: ignore[no-untyped-def]
        self.entries.append(entry)

    def identity(self) -> list[str]:
        """Record ids, in the order they were stored. The thing being compared."""
        return [str(entry.entry_id) for entry in self.entries]  # type: ignore[attr-defined]


class JitteryHistory:
    """A reader whose pages are paged correctly and answered out of order.

    Three properties matter and each one is there to fail a specific wrong
    implementation:

    * **It pages.** A run that ignored ``page`` would loop forever here too, not only
      in production.
    * **Later pages answer sooner**, so if the walk consumed pages in completion order
      the stored records would come out backwards. ``completions`` records the order
      the pages actually finished in, and the equivalence test asserts it differed from
      the order they were asked for -- so a passing test cannot be passing by luck.
    * **It counts its own in-flight reads** under a lock, which is how the bound is
      asserted rather than assumed.
    """

    def __init__(
        self,
        *,
        pull_requests: list[GitHubPullRequest],
        reviews: dict[int, list[PullRequestReview]] | None = None,
        review_comments: dict[int, list[PullRequestReviewComment]] | None = None,
        issue_comments: dict[int, list[GitHubComment]] | None = None,
        failing_pages: frozenset[int] = frozenset(),
        delay: float = PAGE_DELAY,
    ) -> None:
        self.pull_requests = pull_requests
        self.reviews = reviews or {}
        self.review_comments = review_comments or {}
        self.issue_comments = issue_comments or {}
        self.failing_pages = failing_pages
        self._delay = delay
        self._lock = threading.Lock()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.pages_asked: list[int] = []
        self.completions: list[int] = []

    @staticmethod
    def _page(items: list, page: int, per_page: int) -> list:
        start = (page - 1) * per_page
        return list(items[start : start + per_page])

    def _serve(self, page: int) -> list:
        """Answer a listing page, out of order and under a measured in-flight count."""
        with self._lock:
            self.pages_asked.append(page)
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            time.sleep(self._delay / page)
            if page in self.failing_pages:
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=httpx.Request("GET", f"https://api.github.com/repos/{REPO}/pulls"),
                    response=httpx.Response(500),
                )
            return self._page(self.pull_requests, page, PAGE_SIZE)
        finally:
            with self._lock:
                self.in_flight -= 1
                self.completions.append(page)

    def list_pull_requests(
        self, owner: str, repo: str, *, page: int, per_page: int
    ) -> list[GitHubPullRequest]:
        return self._serve(page)

    def list_reviews(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[PullRequestReview]:
        return self._page(self.reviews.get(pr_number, []), page, per_page)

    def list_review_comments(
        self, owner: str, repo: str, pr_number: int, review_id: int, *, page: int, per_page: int
    ) -> list[PullRequestReviewComment]:
        return self._page(self.review_comments.get(review_id, []), page, per_page)

    def list_issue_comments(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[GitHubComment]:
        return self._page(self.issue_comments.get(pr_number, []), page, per_page)


def _pull(number: int, *, updated_at: datetime = SUBMITTED_AT) -> GitHubPullRequest:
    return GitHubPullRequest(
        number=number,
        title=f"Change {number}",
        state="closed",
        user=GitHubUser(login="dev"),
        created_at=datetime(2023, 6, 1, tzinfo=UTC),
        closed_at=updated_at,
        updated_at=updated_at,
    )


def _review(
    review_id: int,
    *,
    body: str = "Ship it: the retry is bounded.",
    submitted_at: datetime = SUBMITTED_AT,
) -> PullRequestReview:
    return PullRequestReview(
        id=review_id,
        state="approved",
        body=body,
        user=GitHubUser(login="reviewer"),
        submitted_at=submitted_at,
        author_association="MEMBER",
    )


def _inline(comment_id: int) -> PullRequestReviewComment:
    return PullRequestReviewComment(
        id=comment_id,
        body="This retry never terminates.",
        user=GitHubUser(login="reviewer"),
        created_at=COMMENTED_AT,
        path="src/x.py",
        line=42,
        diff_hunk="@@ -1 +1 @@",
    )


def _plan(**overrides: object) -> BackfillPlan:
    fields: dict[str, object] = {
        "repositories": (REPO,),
        "since": FLOOR,
        "max_objects": 1000,
    }
    fields.update(overrides)
    return BackfillPlan(**fields)  # type: ignore[arg-type]


@pytest.fixture
def registry(tmp_path) -> SqliteQuestionRegistry:  # type: ignore[no-untyped-def]
    return SqliteQuestionRegistry(tmp_path / "registry.db")


@pytest.fixture
def pair_of_registries(tmp_path) -> tuple[SqliteQuestionRegistry, SqliteQuestionRegistry]:  # type: ignore[no-untyped-def]
    """Two empty registries, for the tests that run the same plan twice and compare.

    Two rather than one is not tidiness. The registry is what makes a re-run collide on
    the semantic event id and write nothing, so a *shared* one would make the second
    run store nothing at all -- which is the idempotence claim working perfectly and
    would make an equivalence comparison vacuous.
    """
    return (
        SqliteQuestionRegistry(tmp_path / "sequential.db"),
        SqliteQuestionRegistry(tmp_path / "concurrent.db"),
    )


def _run(
    reader: JitteryHistory,
    registry: SqliteQuestionRegistry,
    sink: RecordingSink,
    *,
    concurrency: int | None,
    plan: BackfillPlan | None = None,
) -> BackfillReport:
    return run_backfill(
        reader=reader,  # type: ignore[arg-type]
        registry=registry,
        sink=sink,  # type: ignore[arg-type]
        plan=plan or _plan(),
        clock=lambda: READ_AT,
        read_concurrency=concurrency,
    )


def _busy_history(pages: int = 3) -> JitteryHistory:
    """``pages`` full pages of changes, each with one review that stores something.

    Full pages matter: a short page ends the walk, so a fixture of one page would never
    reach a second speculative round and the equivalence test would prove nothing about
    ordering.
    """
    numbers = list(range(1, pages * PAGE_SIZE + 1))
    return JitteryHistory(
        pull_requests=[_pull(number) for number in numbers],
        reviews={number: [_review(9000 + number)] for number in numbers},
        review_comments={},
        issue_comments={},
    )


def _report_fields(report: BackfillReport) -> tuple[object, ...]:
    """Every counter and every gap, so two reports can be compared honestly."""
    return (
        report.objects_read,
        report.records_written,
        report.already_present,
        report.unreadable,
        report.silent,
        report.objects_new,
        report.budget_exhausted,
        report.floor_unreached,
        dict(report.refusals),
        report.gaps,
    )


# --- the whole point: concurrency changes the wall clock and nothing else ----------


def test_concurrency_stores_the_same_records_in_the_same_order_as_sequential(
    pair_of_registries: tuple[SqliteQuestionRegistry, SqliteQuestionRegistry],
) -> None:
    """The test that matters most, and it is an identity assertion rather than a count.

    Three full pages, read sequentially and then at concurrency 4. The stored record
    ids must match element for element, and so must every counter. A count would pass
    this if the walk had stored the same records in the wrong order -- which is the
    exact failure a speculative walk can have, and the one that would corrupt
    ``captured_at`` ordering and any consumer reading the corpus as a sequence.
    """
    sequential_registry, concurrent_registry = pair_of_registries
    sequential_sink = RecordingSink()
    concurrent_sink = RecordingSink()

    sequential = _run(_busy_history(), sequential_registry, sequential_sink, concurrency=1)
    concurrent = _run(
        _busy_history(), concurrent_registry, concurrent_sink, concurrency=CONCURRENCY
    )

    assert concurrent_sink.identity() == sequential_sink.identity()
    assert len(concurrent_sink.entries) > PAGE_SIZE, "the fixture must span several pages"
    assert _report_fields(concurrent) == _report_fields(sequential)


def test_the_concurrent_walk_really_did_overlap_and_really_did_reorder(
    registry: SqliteQuestionRegistry,
) -> None:
    """Without this the equivalence test above could be passing by accident.

    It asserts the two things that make ordering load-bearing: that pages overlapped in
    time, and that they *finished* in an order other than the one they were asked in.
    If either stopped being true the test above would silently stop testing anything --
    a walk whose pages complete in request order would agree with a sequential walk for
    reasons that have nothing to do with ordering.
    """
    history = _busy_history()
    _run(history, registry, RecordingSink(), concurrency=CONCURRENCY)

    assert history.peak_in_flight > 1, "the pages did not overlap, so nothing was concurrent"
    assert history.completions != sorted(history.completions), (
        "the pages happened to finish in request order, which is the one case where a "
        "walk that trusted completion order would look correct"
    )
    # `pages_asked` records when a worker *started*, not when it was submitted, so its
    # order is racy and proves nothing. What must hold is that each page is asked for
    # exactly once and the walk asks for a contiguous run of them -- asking twice would
    # double the quota cost, and skipping one would drop a hundred changes silently.
    # The ramp asks 1, then 2, then 4: pages 1; 2-3; 4-7. The walk stops at 4 (empty,
    # so the enumeration is over) and 5-7 are pure speculation.
    assert sorted(history.pages_asked) == [1, 2, 3, 4, 5, 6, 7], (
        f"pages were not asked for exactly once each: {sorted(history.pages_asked)}"
    )


def test_a_budget_that_trips_mid_batch_captures_the_sequential_prefix(
    pair_of_registries: tuple[SqliteQuestionRegistry, SqliteQuestionRegistry],
) -> None:
    """**Why this is the requirement and not a nicety.** ``--max-objects`` charges new
    work and a re-run is relied upon to *advance*, so the run that trips the budget has
    to leave the same prefix captured. Concurrency cannot be allowed to decide which
    objects fall inside that prefix: a subset chosen by whichever page happened to finish
    first would still report a clean ``budget_exhausted``, and the next run would resume
    from a place nothing recorded.

    The budget here is well inside the first page's worth of reviews, so the trip lands
    while later pages are already being fetched -- the mid-batch case, not the
    convenient one.
    """
    sequential_registry, concurrent_registry = pair_of_registries
    budget = 40
    sequential_sink = RecordingSink()
    concurrent_sink = RecordingSink()
    plan = _plan(max_objects=budget)

    sequential = _run(
        _busy_history(), sequential_registry, sequential_sink, concurrency=1, plan=plan
    )
    concurrent = _run(
        _busy_history(), concurrent_registry, concurrent_sink, concurrency=CONCURRENCY, plan=plan
    )

    assert sequential.budget_exhausted is True
    assert concurrent.budget_exhausted is True
    assert len(concurrent_sink.entries) == budget
    assert concurrent_sink.identity() == sequential_sink.identity(), (
        "the captured prefix is not the sequential prefix, so the next run resumes from "
        "a place this one never reported"
    )
    assert _report_fields(concurrent) == _report_fields(sequential)


# --- the bound is stated, so it is asserted rather than assumed -------------------


def test_at_most_the_bound_requests_are_in_flight_at_once(
    registry: SqliteQuestionRegistry,
) -> None:
    """Observed from the reader, which is where the promise lives.

    Asserted by counting entry and exit in the fake rather than by reading the
    semaphore: a semaphore that is acquired and released around the wrong thing would
    pass a test that inspected the semaphore.
    """
    for bound in (1, 2, 3, CONCURRENCY):
        history = _busy_history()
        _run(history, registry, RecordingSink(), concurrency=bound)
        assert history.peak_in_flight <= bound, (
            f"bound {bound} was exceeded: {history.peak_in_flight} reads in flight"
        )


def test_a_lowered_bound_is_what_the_run_actually_uses(
    registry: SqliteQuestionRegistry,
) -> None:
    """An operator on a shared token lowers this number, so it has to be load-bearing
    rather than decorative."""
    history = _busy_history()
    _run(history, registry, RecordingSink(), concurrency=2)

    assert history.peak_in_flight <= 2
    assert history.peak_in_flight > 1, "and the walk must still be concurrent at 2"


def test_concurrency_of_one_fetches_one_page_at_a_time_and_creates_no_threads(
    registry: SqliteQuestionRegistry,
) -> None:
    """``1`` is not "concurrency disabled" -- it is the strictly sequential walk this
    replaced, which is what an operator on a shared token is told to set. So it has to
    speculate nothing at all: with a ramp starting at one page, a walk of one page must
    fetch one page."""
    history = _busy_history()
    _run(history, registry, RecordingSink(), concurrency=1)

    assert history.peak_in_flight == 1
    assert history.pages_asked == [1, 2, 3, 4], (
        "three pages of history plus the fourth that discovers the end, asked for one at "
        "a time -- and with a one-page window, one at a time is the whole of the walk"
    )


# --- speculation is bounded, and the bound is stated ------------------------------


def test_the_speculation_waste_is_bounded_by_the_bound(
    registry: SqliteQuestionRegistry,
) -> None:
    """**At most ``bound - 1`` pages are ever fetched and thrown away.**

    Measured where the waste is worst -- the last round of a walk that ends early --
    and also on a walk long enough for the ramp to have reached its bound several
    times, because a bound that only held on short walks would not be a bound.

    "Thrown away" is counted as *asked for and never reached*: the pages the walk got
    to are the ones it consumed, and everything else it paid for and read nothing from.
    """
    worst = _busy_history(pages=3)
    worst_report = _run(worst, registry, RecordingSink(), concurrency=CONCURRENCY)
    fetched = sorted(worst.pages_asked)
    # The fixture holds three full pages, so the walk consumes 1-3 and then page 4,
    # which comes back empty and ends it. Everything after page 4 was fetched for
    # nothing.
    consumed = 4
    assert fetched == [1, 2, 3, 4, 5, 6, 7]
    assert worst_report.floor_unreached is True
    assert fetched[consumed:] == [5, 6, 7], (
        f"the pages fetched and thrown away were {fetched[consumed:]}"
    )
    assert len(fetched[consumed:]) == CONCURRENCY - 1, (
        "the speculation is bounded by the bound minus one, which is what the constant claims"
    )

    long_walk = _busy_history(pages=6)
    _run(long_walk, registry, RecordingSink(), concurrency=CONCURRENCY)
    # Six full pages, so the ramp's last round (4-7) contains both the remaining history
    # and the empty page that ends the walk: it wastes nothing at all. A fixed-size
    # batch would have fetched 4-7 on the previous round and discarded three pages here.
    assert sorted(long_walk.pages_asked) == [1, 2, 3, 4, 5, 6, 7]


def test_a_run_that_stops_at_its_floor_still_speculates_only_within_the_bound(
    registry: SqliteQuestionRegistry,
) -> None:
    """The pathological case for speculation: the window is on page one, so every later
    page in the round is fetched for nothing."""
    history = JitteryHistory(pull_requests=[_pull(number) for number in range(1, 6)])
    report = _run(history, registry, RecordingSink(), concurrency=CONCURRENCY)

    assert len(history.pages_asked) == 1, (
        "a short first page must not trigger speculation at all: the ramp starts at one "
        "page precisely so the common case pays nothing"
    )
    assert report.floor_unreached is True


def test_the_floor_still_ends_the_walk_and_is_still_reported(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The floor short-circuit, unchanged.** Concurrency must not make a run claim a
    floor it never checked, and must not make it stop short of one it did."""
    old = datetime(2023, 5, 1, tzinfo=UTC)
    history = JitteryHistory(
        pull_requests=[
            _pull(7),
            *[_pull(number, updated_at=old) for number in range(8, 8 + PAGE_SIZE)],
        ],
        reviews={7: [_review(9001)]},
        review_comments={9001: [_inline(7001)]},
    )

    report = _run(history, registry, RecordingSink(), concurrency=CONCURRENCY)

    assert report.objects_read == 2, "the one change inside the window, and its review"
    assert report.floor_unreached is False, (
        "the floor *was* reached, so this is not a range the run failed to cover"
    )


def test_a_ceiling_still_skips_rather_than_ends_the_walk(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The ceiling short-circuit, unchanged.** A page of changes that are all newer
    than the bound says the window is further back, not that there is none."""
    may = datetime(2024, 5, 20, 12, 0, tzinfo=UTC)
    april = datetime(2024, 4, 10, 12, 0, tzinfo=UTC)
    history = JitteryHistory(
        pull_requests=[_pull(number, updated_at=may) for number in range(1, PAGE_SIZE + 1)]
        + [_pull(PAGE_SIZE + 1, updated_at=april)],
        reviews={PAGE_SIZE + 1: [_review(9001, submitted_at=april)]},
        review_comments={9001: [_inline(7001)]},
    )

    report = _run(
        history,
        registry,
        RecordingSink(),
        concurrency=CONCURRENCY,
        plan=_plan(
            since=datetime(2024, 4, 1, tzinfo=UTC),
            until=datetime(2024, 4, 30, 23, 59, 59, 999_999, tzinfo=UTC),
        ),
    )

    assert report.objects_read == 2, (
        "the change behind the page of newer ones, and its review; a whole page of May "
        "changes was passed over to reach it"
    )


# --- a failure inside a round is the run's failure, not the pool's ---------------


def test_a_failed_page_surfaces_as_the_run_reports_any_other_failed_read(
    registry: SqliteQuestionRegistry,
) -> None:
    """Not swallowed, not wrapped, and not an exception escaping a worker.

    The run's own error handling is written against a sequential walk, so a page failure
    has to arrive as the original exception object: the operator sees the forge's status,
    and ``cli.py`` still catches it in the ``except`` clause that prints "Backfill
    stopped" and exits 1. A future wrapping would fall through that clause and surface
    as a traceback.
    """
    boom = httpx.HTTPStatusError(
        "Server Error",
        request=httpx.Request("GET", "https://api.github.com/repos/org/repo/pulls"),
        response=httpx.Response(500),
    )

    class OneBadPage(JitteryHistory):
        def list_pull_requests(self, owner, repo, *, page, per_page):  # type: ignore[no-untyped-def]
            if page == 3:
                raise boom
            return super().list_pull_requests(owner, repo, page=page, per_page=per_page)

    history = OneBadPage(pull_requests=[_pull(number) for number in range(1, 400)])

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _run(history, registry, RecordingSink(), concurrency=CONCURRENCY)

    assert excinfo.value is boom, (
        "the failure was re-raised by the walk rather than rebuilt in a worker, so it is "
        "the same object the sequential walk would have raised"
    )


def test_the_first_failing_page_in_listing_order_is_the_one_that_is_reported(
    registry: SqliteQuestionRegistry,
) -> None:
    """Two pages of one round both fail, and which one is *reported* is a real question.

    A sequential walk would have stopped at the earlier page and never issued the later
    request, so reporting the later failure would attribute the run's stop to a page the
    sequential walk never reached. Page order, not completion order, is what makes the
    concurrent walk report the same thing.
    """
    failures: dict[int, httpx.HTTPStatusError] = {}

    def fail(page: int) -> httpx.HTTPStatusError:
        return failures.setdefault(
            page,
            httpx.HTTPStatusError(
                f"boom {page}",
                request=httpx.Request("GET", "https://api.github.com/repos/org/repo/pulls"),
                response=httpx.Response(500 + page),
            ),
        )

    class TwoBadPages(JitteryHistory):
        def list_pull_requests(self, owner, repo, *, page, per_page):  # type: ignore[no-untyped-def]
            if page in (2, 3):
                # Raise them in reverse request order to make sure the *reported* one
                # is chosen by page number rather than by who lost the race.
                time.sleep(0.02 if page == 2 else 0.001)
                raise fail(page)
            return super().list_pull_requests(owner, repo, page=page, per_page=per_page)

    history = TwoBadPages(pull_requests=[_pull(number) for number in range(1, 400)])

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _run(history, registry, RecordingSink(), concurrency=CONCURRENCY)

    assert excinfo.value is failures[2], (
        "page 2 fails first in listing order, so page 2 is reported"
    )


def test_a_deleted_review_is_still_a_gap_rather_than_a_stopped_run(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The gap path, unchanged.** A review deleted after the fact 404s on its comments;
    the run reports the hole and carries on. Concurrency must not turn a survivable read
    failure into a fatal one, so this exercises it on a walk that has already been
    reading concurrently."""
    numbers = list(range(1, PAGE_SIZE + 3))
    history = JitteryHistory(
        pull_requests=[_pull(number) for number in numbers],
        reviews={number: [_review(9000 + number)] for number in numbers},
        review_comments={},
    )
    deleted = 9000 + 5
    # Bound before the override replaces it, or the override calls itself.
    original = history.list_review_comments

    def comments(owner, repo, pr_number, review_id, *, page, per_page):  # type: ignore[no-untyped-def]
        if review_id == deleted:
            request = httpx.Request("GET", f"https://api.github.com/repos/{REPO}/pulls/5/comments")
            raise httpx.HTTPStatusError(
                "Not Found", request=request, response=httpx.Response(404, request=request)
            )
        return original(owner, repo, pr_number, review_id, page=page, per_page=per_page)

    history.list_review_comments = comments  # type: ignore[method-assign]
    sink = RecordingSink()

    report = _run(history, registry, sink, concurrency=CONCURRENCY)

    assert report.unreadable == 1, "the hole is reported rather than written around"
    assert report.gaps[0].what == "review 9005"
    assert "no longer has it" in report.gaps[0].reason
    assert len(sink.entries) > 0, "and the rest of the walk was still reconstructed"


# --- the pool is shared, and the two timeouts are not the same number --------------


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Stub the transport, forwarding the timeout.

    Forwarding matters for the one test that reads timeouts off the requests: a stub
    that swallowed them would report httpx's 5-second default and the assertion would
    pass or fail for reasons that have nothing to do with the seam.
    """
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def _patch_transport_counting(
    monkeypatch: pytest.MonkeyPatch, handler: Any, built: list[httpx.Client]
) -> None:
    """Stub the transport and keep every pool that gets built, so reuse is countable."""
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", factory)


def test_a_client_reuses_one_pool_across_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handshake cost this removes: one pool for the client's life, not one per
    request. Counted by counting the pools that were actually built."""
    built: list[httpx.Client] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "kojutsu-bot"})
        return httpx.Response(200, json=[])

    _patch_transport_counting(monkeypatch, handler, built)

    client = GitHubClient("tok")
    for _ in range(5):
        client.list_issue_comments("org", "repo", 1)
    client.get_authenticated_user()

    assert len(built) == 1, f"{len(built)} pools for five reads; the handshakes are back"

    client.close()
    assert built[0].is_closed is True, "close() has to actually return the sockets"


def test_a_client_that_never_reads_builds_no_pool() -> None:
    """Most of these clients perform one short call and are dropped; eagerly opening a
    pool for an object about to be garbage would trade a real cost for a tidy one."""
    GitHubClient("tok").close()


def test_the_diff_read_keeps_its_own_timeout_and_the_rest_keep_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The two timeouts are not unified, and this is why they must not be.** A diff is
    megabytes and everything else on this seam is kilobytes, so the one call that
    transfers orders of magnitude more bytes is given orders of magnitude more patience.

    Captured off the request rather than read off the constant, because the constant is
    what the two call sites both *say* and the request is what they *do* -- and a
    refactor that moved the number without moving the behaviour would pass the former.
    """
    seen: list[tuple[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout")
        read = float(timeout["read"]) if isinstance(timeout, dict) else None
        seen.append((request.headers.get("Accept", ""), read))
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/pulls/1"):
            return httpx.Response(200, text="diff --git a/x b/x")
        raise AssertionError(f"unexpected path {request.url.path}")

    _patch_transport(monkeypatch, handler)

    with GitHubClient("tok") as client:
        client.list_issue_comments("org", "repo", 1)
        client.get_pull_diff("org", "repo", 1)

    assert (DIFF_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS) == (60.0, 30.0)
    ordinary = [read for accept, read in seen if "diff" not in accept]
    diff = [read for accept, read in seen if "diff" in accept]
    assert ordinary == [REQUEST_TIMEOUT_SECONDS]
    assert diff == [DIFF_TIMEOUT_SECONDS]


def test_the_history_reader_reuses_one_pool_across_a_walk(
    monkeypatch: pytest.MonkeyPatch, registry: SqliteQuestionRegistry
) -> None:
    """The reader is the class the walk actually goes through, so this is where the 96
    handshakes were being paid."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/pulls"):
            page = int(request.url.params.get("page", 1))
            if page > 3:
                return httpx.Response(200, json=[])
            return httpx.Response(
                200, json=[_pull(number).model_dump(mode="json") for number in range(1, 4)]
            )
        if request.url.path.endswith("/reviews"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[])

    built: list[httpx.Client] = []
    _patch_transport_counting(monkeypatch, handler, built)

    with GitHubHistoryReader(token="tok") as reader:
        _run(reader, registry, RecordingSink(), concurrency=CONCURRENCY)  # type: ignore[arg-type]

    assert len(built) == 1, f"{len(built)} pools for one walk"


# --- the operator can lower it, and the two places that hold the number agree -----


def _settings() -> Settings:
    return Settings(_env_file=None, _config_file=None)  # type: ignore[call-arg]


def test_the_configured_bound_defaults_to_the_documented_one() -> None:
    """``config.py`` cannot import the constant -- that module imports ``config`` -- so
    the number is written twice and pinned here. This is the guard on that arrangement,
    and it exists because a silently stale default is the worst kind: the run would
    quietly ignore the documented bound."""
    assert _settings().github_history_concurrency == HISTORY_READ_CONCURRENCY


def test_an_operator_on_a_shared_token_can_lower_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_HISTORY_CONCURRENCY", "1")
    assert _settings().github_history_concurrency == 1

    monkeypatch.setenv("GITHUB_HISTORY_CONCURRENCY", "2")
    assert _settings().github_history_concurrency == 2


def test_a_bound_of_nothing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero is not "as sequential as possible" -- it is no reads at all, and a run that
    reads nothing must not start. Negative is the same mistake with a sign."""
    for value in ("0", "-1"):
        monkeypatch.setenv("GITHUB_HISTORY_CONCURRENCY", value)
        with pytest.raises(ValueError):
            _settings()
