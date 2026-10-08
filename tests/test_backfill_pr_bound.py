"""Single-PR bound for backfill-reviews: walk nothing, keep the accounting.

Covers the ticket's acceptance criteria: a ``--pr`` run reads only the named
pull requests' own pages and no listing pages; the window still applies and is
reported rather than widened; a PR past the page ceiling is reported through
``tally.gap`` with the existing reason; a PR with no reviews is an empty range
rather than an error; and the report keeps the same shape as a ranged run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from kojutsu.cli_backfill import _parse_pr_numbers
from kojutsu.core.backfill_reviews import (
    _PAGE_CEILING_REASON,
    BackfillPlan,
    BackfillReport,
    build_plan,
    run_backfill,
)
from kojutsu.core.backfill_reviews_client import PAGE_SIZE
from kojutsu.core.question_registry import SqliteQuestionRegistry
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


class RecordingSink:
    """A sink that keeps what it was handed, in the order it was handed it."""

    def __init__(self) -> None:
        self.entries: list[object] = []

    def store(self, entry):  # type: ignore[no-untyped-def]
        self.entries.append(entry)


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


class SinglePrHistory:
    """A reader serving named pull requests and their pages, counting listing reads."""

    def __init__(
        self,
        *,
        pulls: dict[int, GitHubPullRequest],
        reviews: dict[int, list[PullRequestReview]] | None = None,
        review_comments: dict[int, list[PullRequestReviewComment]] | None = None,
        issue_comments: dict[int, list[GitHubComment]] | None = None,
    ) -> None:
        self.pulls = pulls
        self.reviews = reviews or {}
        self.review_comments = review_comments or {}
        self.issue_comments = issue_comments or {}
        self.listing_calls = 0
        self.fetched_prs: list[int] = []

    @staticmethod
    def _page(items: list, page: int, per_page: int) -> list:
        start = (page - 1) * per_page
        return list(items[start : start + per_page])

    def get_pull_request(self, owner: str, repo: str, pr_number: int):  # type: ignore[no-untyped-def]
        self.fetched_prs.append(pr_number)
        return self.pulls.get(pr_number)

    def list_pull_requests(self, owner: str, repo: str, *, page: int, per_page: int):  # type: ignore[no-untyped-def]
        self.listing_calls += 1
        return []

    def list_reviews(self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int):  # type: ignore[no-untyped-def]
        return self._page(self.reviews.get(pr_number, []), page, per_page)

    def list_review_comments(
        self, owner: str, repo: str, pr_number: int, review_id: int, *, page: int, per_page: int
    ):  # type: ignore[no-untyped-def]
        return self._page(self.review_comments.get(review_id, []), page, per_page)

    def list_issue_comments(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ):  # type: ignore[no-untyped-def]
        return self._page(self.issue_comments.get(pr_number, []), page, per_page)


def _plan(pr_numbers: tuple[int, ...] | None, **overrides: Any) -> BackfillPlan:
    fields: dict[str, Any] = {
        "repositories": (REPO,),
        "since": FLOOR,
        "max_objects": 1000,
        "pr_numbers": pr_numbers,
    }
    fields.update(overrides)
    return BackfillPlan(**fields)


def _run(
    reader: SinglePrHistory, plan: BackfillPlan, tmp_path
) -> tuple[BackfillReport, RecordingSink]:
    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    sink = RecordingSink()
    try:
        report = run_backfill(
            reader=reader,  # type: ignore[arg-type]
            registry=registry,
            sink=sink,  # type: ignore[arg-type]
            plan=plan,
            clock=lambda: READ_AT,
            read_concurrency=1,
        )
    finally:
        registry.close()
    return report, sink


def test_pr_bound_reads_no_listing_pages(tmp_path) -> None:
    """The walk fetches the named PR and its pages, and never lists."""
    reader = SinglePrHistory(
        pulls={2829: _pull(2829)},
        reviews={2829: [_review(1)]},
    )
    report, sink = _run(reader, _plan((2829,)), tmp_path)

    assert reader.listing_calls == 0
    assert reader.fetched_prs == [2829]
    assert report.objects_read >= 1
    assert len(sink.entries) > 0


def test_pr_bound_with_since_narrows_rather_than_widens(tmp_path) -> None:
    """A review outside the window is skipped; the window is reported unchanged."""
    old = _review(
        1, body="Old verdict from before the floor.", submitted_at=datetime(2023, 1, 2, tzinfo=UTC)
    )
    new = _review(2, body="New verdict inside the window.", submitted_at=SUBMITTED_AT)
    reader = SinglePrHistory(
        pulls={7: _pull(7)},
        reviews={7: [old, new]},
    )
    plan = _plan((7,))
    report, sink = _run(reader, plan, tmp_path)

    assert plan.since == FLOOR
    assert report.plan.since == FLOOR
    # The pull container plus the one in-window review; the out-of-window
    # review is skipped before it is ever examined.
    assert report.objects_read == 2
    rendered = " ".join(str(entry) for entry in sink.entries)
    assert "before the floor" not in rendered
    assert "inside the window" in rendered


def test_pr_before_floor_is_skipped_not_walked(tmp_path) -> None:
    """A PR updated before the floor contributes nothing and reads no reviews."""
    stale = _pull(9, updated_at=datetime(2023, 1, 2, tzinfo=UTC))
    reader = SinglePrHistory(
        pulls={9: stale},
        reviews={9: [_review(1)]},
    )
    report, sink = _run(reader, _plan((9,)), tmp_path)

    assert report.objects_read == 0
    assert len(sink.entries) == 0


def test_missing_pr_is_a_gap_not_an_error(tmp_path) -> None:
    """A number the forge has nothing for is reported, not raised."""
    reader = SinglePrHistory(pulls={})
    report, sink = _run(reader, _plan((4242,)), tmp_path)

    assert len(sink.entries) == 0
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.pr_number == 4242
    assert gap.repository == REPO


def test_pr_with_no_reviews_is_an_empty_range(tmp_path) -> None:
    """An existing PR with nothing to say stores nothing and fails nothing."""
    reader = SinglePrHistory(pulls={11: _pull(11)}, reviews={})
    report, sink = _run(reader, _plan((11,)), tmp_path)

    assert len(sink.entries) == 0
    assert report.unreadable == 0
    assert report.budget_exhausted is False
    assert report.objects_read == 1


def test_pr_past_the_page_ceiling_reports_a_gap(tmp_path, monkeypatch) -> None:
    """Reviews past the ceiling are a named gap with the existing reason."""
    import kojutsu.core.backfill_reviews as backfill_module

    monkeypatch.setattr(backfill_module, "MAX_PAGES_PER_OBJECT", 2)
    many = [_review(1000 + i) for i in range(3 * PAGE_SIZE)]
    reader = SinglePrHistory(pulls={5: _pull(5)}, reviews={5: many})
    report, _sink = _run(reader, _plan((5,)), tmp_path)

    reasons = [gap.reason for gap in report.gaps]
    assert any(reason == _PAGE_CEILING_REASON for reason in reasons)


def test_report_keeps_the_same_shape(tmp_path) -> None:
    """Every counter a ranged run reports is present on a --pr run too."""
    reader = SinglePrHistory(
        pulls={2829: _pull(2829)},
        reviews={2829: [_review(1)]},
    )
    report, _sink = _run(reader, _plan((2829,)), tmp_path)

    assert isinstance(report.objects_read, int)
    assert isinstance(report.records_written, int)
    assert isinstance(report.already_present, int)
    assert isinstance(report.unreadable, int)
    assert isinstance(report.silent, int)
    assert isinstance(report.objects_new, int)
    assert isinstance(report.budget_exhausted, bool)
    assert isinstance(report.refusals, dict)
    assert isinstance(report.gaps, tuple)


def test_build_plan_validates_pr_numbers() -> None:
    """Non-numbers, non-positive numbers, and empties are refused."""

    class FakeSettings:
        github_webhook_allowed_repositories = REPO

    settings = FakeSettings()
    plan = build_plan(
        settings=settings,  # type: ignore[arg-type]
        repositories=[REPO],
        since="2024-01-01",
        max_objects=10,
        pr_numbers=[2829, "2830"],
    )
    assert plan.pr_numbers == (2829, 2830)

    for bad in (["abc"], [0], [-3], [True], []):
        with pytest.raises(ValueError, match="--pr"):
            build_plan(
                settings=settings,  # type: ignore[arg-type]
                repositories=[REPO],
                since="2024-01-01",
                max_objects=10,
                pr_numbers=bad,  # type: ignore[arg-type]
            )


def test_parse_pr_numbers_accepts_repeatable_and_comma_separated() -> None:
    assert _parse_pr_numbers(None) is None
    assert _parse_pr_numbers([]) is None
    assert _parse_pr_numbers(["2829"]) == [2829]
    assert _parse_pr_numbers(["2829", "2830, 2831"]) == [2829, 2830, 2831]
    with pytest.raises(ValueError, match="--pr"):
        _parse_pr_numbers(["nope"])
    with pytest.raises(ValueError, match="--pr"):
        _parse_pr_numbers(["0"])
