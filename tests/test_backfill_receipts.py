"""Pass-end receipts for backfill-reviews: skip finished, unchanged PRs.

A run that finished a pull request records the pass end -- the PR number, the
listing timestamp it saw, and the window it ran under. The next run lists (the
cheap part, ~20 requests for a quarter), compares each listing timestamp
against its receipt, and skips the review/comment reads (the expensive part)
for finished PRs nothing moved on.

What these tests pin, in the order the strategy was argued:

- a finished, unchanged PR costs no review/comment reads on the next run;
- a PR the listing shows as touched is re-read, and stores only what is
  missing;
- a PR a stopped run was inside has no finished receipt, so it is re-read;
- a receipt written under one window does not excuse a run under another;
- a page-ceiling truncation is not a finished pass;
- an empty-bodied review still finishes the pass: the evidence is reaching
  the end of the walk, never the timestamps of what was stored.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kojutsu.core.backfill_reviews import BackfillPlan, run_backfill
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
COMMENTED_AT = datetime(2024, 3, 4, 9, 5, tzinfo=UTC)


class RecordingSink:
    """A sink that keeps what it was handed, in the order it was handed it."""

    def __init__(self) -> None:
        self.entries: list[object] = []

    def store(self, entry):  # type: ignore[no-untyped-def]
        self.entries.append(entry)


class ReceiptHistory:
    """A reader over fixed data that records which pages it was asked for."""

    def __init__(
        self,
        *,
        pull_requests: list[GitHubPullRequest] | None = None,
        reviews: dict[int, list[PullRequestReview]] | None = None,
        review_comments: dict[int, list[PullRequestReviewComment]] | None = None,
        issue_comments: dict[int, list[GitHubComment]] | None = None,
    ) -> None:
        self.pull_requests = pull_requests or []
        self.reviews = reviews or {}
        self.review_comments = review_comments or {}
        self.issue_comments = issue_comments or {}
        self.calls: list[str] = []

    @staticmethod
    def _page(items: list, page: int, per_page: int) -> list:
        start = (page - 1) * per_page
        return list(items[start : start + per_page])

    def list_pull_requests(self, owner: str, repo: str, *, page: int, per_page: int):  # type: ignore[no-untyped-def]
        self.calls.append("pulls")
        return self._page(self.pull_requests, page, per_page)

    def list_reviews(self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int):  # type: ignore[no-untyped-def]
        self.calls.append(f"reviews:{pr_number}")
        return self._page(self.reviews.get(pr_number, []), page, per_page)

    def list_review_comments(
        self, owner: str, repo: str, pr_number: int, review_id: int, *, page: int, per_page: int
    ):  # type: ignore[no-untyped-def]
        self.calls.append(f"review-comments:{review_id}")
        return self._page(self.review_comments.get(review_id, []), page, per_page)

    def list_issue_comments(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ):  # type: ignore[no-untyped-def]
        self.calls.append(f"issue-comments:{pr_number}")
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


def _inline(comment_id: int = 7001) -> PullRequestReviewComment:
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
        "max_objects": 100,
    }
    fields.update(overrides)
    return BackfillPlan(**fields)  # type: ignore[arg-type]


def _run(
    reader: ReceiptHistory,
    registry: SqliteQuestionRegistry,
    sink: RecordingSink,
    plan: BackfillPlan,
):  # type: ignore[no-untyped-def]
    return run_backfill(
        reader=reader,  # type: ignore[arg-type]
        registry=registry,
        sink=sink,  # type: ignore[arg-type]
        plan=plan,
        clock=lambda: READ_AT,
        read_concurrency=1,
    )


def _registry(tmp_path: Path) -> SqliteQuestionRegistry:
    return SqliteQuestionRegistry(tmp_path / "registry.db")


def test_second_run_skips_finished_unchanged_pull_requests(tmp_path: Path) -> None:
    """The listing still runs; the review and comment reads do not."""
    history = ReceiptHistory(
        pull_requests=[_pull(7), _pull(8)],
        reviews={7: [_review(9001)], 8: [_review(9002)]},
        review_comments={9001: [_inline(7001)], 9002: [_inline(7002)]},
        issue_comments={7: [], 8: []},
    )
    registry = _registry(tmp_path)
    sink = RecordingSink()

    first = _run(history, registry, sink, _plan())
    assert first.skipped_finished == 0, "no receipt exists yet, so nothing is skipped"
    assert first.records_written > 0

    calls_before = list(history.calls)
    second = _run(history, registry, sink, _plan())

    assert second.skipped_finished == 2
    assert second.records_written == 0
    assert second.already_present == 0, "nothing re-read means nothing to collide on"
    fresh = history.calls[len(calls_before) :]
    assert "pulls" in fresh, "the listing is the evidence the skip is decided on"
    assert not [call for call in fresh if call.startswith("reviews:")], fresh
    assert not [call for call in fresh if call.startswith("issue-comments:")], fresh


def test_touched_pull_request_is_re_read_and_stores_only_what_is_missing(
    tmp_path: Path,
) -> None:
    """A moved listing timestamp re-opens the pull request; identity dedupes it."""
    history = ReceiptHistory(
        pull_requests=[_pull(7), _pull(8)],
        reviews={7: [_review(9001)], 8: [_review(9002)]},
        review_comments={9001: [_inline(7001)], 9002: [_inline(7002)]},
        issue_comments={7: [], 8: []},
    )
    registry = _registry(tmp_path)
    sink = RecordingSink()
    _run(history, registry, sink, _plan())

    moved_at = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
    history.pull_requests = [_pull(7), _pull(8, updated_at=moved_at)]
    history.reviews[8].append(
        _review(9003, body="Follow-up: bound the retry.", submitted_at=moved_at)
    )
    history.review_comments[9003] = [_inline(7003)]

    second = _run(history, registry, sink, _plan())

    assert second.skipped_finished == 1, "the untouched change is still skipped"
    assert "reviews:8" in history.calls
    assert second.objects_new == 1, "only the follow-up review is new work"
    assert second.already_present >= 1, "the old review collides instead of duplicating"

    third = _run(history, registry, sink, _plan())
    assert third.skipped_finished == 2, "the re-read left a fresh receipt behind it"
    assert third.records_written == 0


def test_interrupted_pull_request_is_re_read(tmp_path: Path) -> None:
    """The pull request a stopped run was inside has no finished receipt."""
    history = ReceiptHistory(
        pull_requests=[_pull(7), _pull(8)],
        reviews={7: [_review(9001)], 8: [_review(9002)]},
        review_comments={9001: [_inline(7001)], 9002: [_inline(7002)]},
        issue_comments={7: [], 8: []},
    )
    registry = _registry(tmp_path)
    sink = RecordingSink()

    first = _run(history, registry, sink, _plan(max_objects=1))
    assert first.budget_exhausted is True
    assert registry.get_backfill_receipt(repo=REPO, pr_number=7)["finished"] is True
    assert registry.get_backfill_receipt(repo=REPO, pr_number=8) is None

    calls_before = list(history.calls)
    second = _run(history, registry, sink, _plan(max_objects=100))

    fresh = history.calls[len(calls_before) :]
    assert "reviews:8" in fresh, "the unfinished change is read again"
    assert "reviews:7" not in fresh, "the finished change is not"
    assert second.skipped_finished == 1
    assert second.budget_exhausted is False


def test_receipt_written_under_one_window_does_not_excuse_another(
    tmp_path: Path,
) -> None:
    """A wider window re-lists reviews the narrower pass skipped over."""
    history = ReceiptHistory(
        pull_requests=[_pull(7)],
        reviews={7: [_review(9001)]},
        review_comments={9001: [_inline(7001)]},
        issue_comments={7: []},
    )
    registry = _registry(tmp_path)
    sink = RecordingSink()
    _run(history, registry, sink, _plan())

    calls_before = list(history.calls)
    second = _run(history, registry, sink, _plan(since=datetime(2023, 1, 1, tzinfo=UTC)))

    fresh = history.calls[len(calls_before) :]
    assert "reviews:7" in fresh
    assert second.skipped_finished == 0
    assert second.already_present >= 1


def test_page_ceiling_leaves_no_finished_receipt(tmp_path: Path, monkeypatch) -> None:
    """A truncated walk records the attempt, never the finish."""
    import kojutsu.core.backfill_reviews as backfill_module
    from kojutsu.core.backfill_reviews_client import PAGE_SIZE

    monkeypatch.setattr(backfill_module, "MAX_PAGES_PER_OBJECT", 1)
    history = ReceiptHistory(
        pull_requests=[_pull(7)],
        reviews={7: [_review(9000 + i) for i in range(2 * PAGE_SIZE)]},
        review_comments={},
        issue_comments={7: []},
    )
    registry = _registry(tmp_path)
    sink = RecordingSink()
    _run(history, registry, sink, _plan(max_objects=10_000))

    receipt = registry.get_backfill_receipt(repo=REPO, pr_number=7)
    assert receipt is not None
    assert receipt["finished"] is False

    calls_before = list(history.calls)
    _run(history, registry, sink, _plan(max_objects=10_000))
    assert "reviews:7" in history.calls[len(calls_before) :]


def test_empty_bodied_review_still_finishes_the_pass(tmp_path: Path) -> None:
    """Nothing stored is not nothing read: reaching the end is the evidence."""
    history = ReceiptHistory(
        pull_requests=[_pull(7)],
        reviews={7: [_review(9001, body="")]},
        review_comments={9001: []},
        issue_comments={7: []},
    )
    registry = _registry(tmp_path)
    sink = RecordingSink()
    first = _run(history, registry, sink, _plan())

    assert first.records_written == 0
    assert registry.get_backfill_receipt(repo=REPO, pr_number=7)["finished"] is True

    calls_before = list(history.calls)
    second = _run(history, registry, sink, _plan())

    assert second.skipped_finished == 1
    fresh = history.calls[len(calls_before) :]
    assert not [call for call in fresh if call.startswith("reviews:")], fresh


def test_receipt_is_overwritten_by_the_latest_pass(tmp_path: Path) -> None:
    """A re-read replaces the receipt; a stale finished flag cannot linger."""
    registry = _registry(tmp_path)
    observed = datetime(2024, 3, 4, 9, 0, tzinfo=UTC)
    registry.record_backfill_receipt(
        repo=REPO,
        pr_number=7,
        observed_updated_at=observed,
        window_since=FLOOR,
        window_until=None,
        authorized_associations=None,
        finished=True,
    )
    later = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
    registry.record_backfill_receipt(
        repo=REPO,
        pr_number=7,
        observed_updated_at=later,
        window_since=FLOOR,
        window_until=None,
        authorized_associations=None,
        finished=False,
    )

    receipt = registry.get_backfill_receipt(repo=REPO, pr_number=7)
    assert receipt is not None
    assert receipt["finished"] is False
    assert receipt["observed_updated_at"] == later.isoformat()


def test_receipt_keys_on_identity_not_spelling(tmp_path: Path) -> None:
    """Two casings of one repository cannot hold disagreeing receipts."""
    registry = _registry(tmp_path)
    registry.record_backfill_receipt(
        repo="Org/Repo",
        pr_number=7,
        observed_updated_at=SUBMITTED_AT,
        window_since=FLOOR,
        window_until=None,
        authorized_associations=None,
        finished=True,
    )

    receipt = registry.get_backfill_receipt(repo="org/repo", pr_number=7)
    assert receipt is not None
    assert receipt["finished"] is True
