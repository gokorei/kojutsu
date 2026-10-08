"""Tests for the bounded, resumable backfill over repository history.

Every test here is a boundary of what a reconstruction may claim, because that is
the whole risk of the feature. A backfill reads the forge long after the events it
reads, and the failure mode is not a crash: it is a corpus that quietly claims more
than it saw, answers to questions nobody asked, anchored to commits that were never
reviewed, and a hole where a deleted review used to be with nothing recording that
it is a hole.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from kojutsu.cli import app
from kojutsu.config import Settings
from kojutsu.core.answer_collector import (
    REVIEW_KIND_INLINE,
    REVIEW_KIND_VERDICT,
    process_review_event_outcome,
    semantic_review_event_id,
    stable_review_entry_id,
)
from kojutsu.core.backfill_reviews import (
    MAX_OBJECTS_PER_RUN,
    BackfillPlan,
    BackfillReadSink,
    UnanchorableReadError,
    build_plan,
    run_backfill,
)
from kojutsu.core.backfill_reviews_client import (
    MAX_PAGES_PER_OBJECT,
    PAGE_SIZE,
    GitHubHistoryReader,
    RateLimitedError,
)
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.integrations.github import (
    answer_comment_body,
    kojutsu_comment_body,
)
from kojutsu.integrations.github_models import (
    GitHubComment,
    GitHubPullRequest,
    GitHubUser,
    PullRequestReview,
    PullRequestReviewComment,
)
from kojutsu.models import (
    REVIEW_ID_KEY,
    CaptureSource,
    KnowledgeEntry,
    QuestionCategory,
    capture_anchor_gaps,
)

runner = CliRunner()

_ANSI_CODE = re.compile(r"\x1b\[[0-9;]*m")


def _operator_text(output: str) -> str:
    """The text an operator reads, without terminal styling.

    Typer highlights `--options` when it believes the output is a terminal --
    which, on CI, is always, because `GITHUB_ACTIONS` forces terminal mode --
    and its highlighter styles a long option in fragments (`-`, `-max`,
    `-objects`), so the plain option name is not a substring of the styled
    output. These tests assert on what the operator reads, not on the styling,
    so they compare against the unstyled text.
    """
    return _ANSI_CODE.sub("", output)


REPO = "org/repo"
FLOOR = datetime(2024, 1, 1, tzinfo=UTC)
READ_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)

SUBMITTED_AT = datetime(2024, 3, 4, 9, 0, tzinfo=UTC)
COMMENTED_AT = datetime(2024, 3, 4, 9, 30, tzinfo=UTC)


class RecordingSink:
    """A sink that keeps what it was handed, in the order it was handed it."""

    def __init__(self) -> None:
        self.entries: list[KnowledgeEntry] = []

    def store(self, entry: KnowledgeEntry) -> None:
        self.entries.append(entry)

    def kinds(self) -> list[str]:
        return [str(entry.metadata.get("record_kind")) for entry in self.entries]


def _not_found(path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://api.github.com/{path}")
    return httpx.HTTPStatusError(
        "Not Found", request=request, response=httpx.Response(404, request=request)
    )


class FakeHistory:
    """A reader over a fixed history: no socket, no clock, no credentials.

    Paged exactly as the real reader pages, so a run that ignored ``page`` would
    loop forever here too rather than only in production.
    """

    def __init__(
        self,
        *,
        pull_requests: list[GitHubPullRequest] | None = None,
        reviews: dict[int, list[PullRequestReview]] | None = None,
        review_comments: dict[int, list[PullRequestReviewComment]] | None = None,
        issue_comments: dict[int, list[GitHubComment]] | None = None,
        deleted_reviews: frozenset[int] = frozenset(),
    ) -> None:
        self.pull_requests = pull_requests or []
        self.reviews = reviews or {}
        self.review_comments = review_comments or {}
        self.issue_comments = issue_comments or {}
        self.deleted_reviews = deleted_reviews
        self.calls: list[str] = []

    @staticmethod
    def _page(items: list, page: int, per_page: int) -> list:
        start = (page - 1) * per_page
        return list(items[start : start + per_page])

    def list_pull_requests(
        self, owner: str, repo: str, *, page: int, per_page: int
    ) -> list[GitHubPullRequest]:
        self.calls.append("pulls")
        return self._page(self.pull_requests, page, per_page)

    def list_reviews(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[PullRequestReview]:
        self.calls.append(f"reviews:{pr_number}")
        return self._page(self.reviews.get(pr_number, []), page, per_page)

    def list_review_comments(
        self, owner: str, repo: str, pr_number: int, review_id: int, *, page: int, per_page: int
    ) -> list[PullRequestReviewComment]:
        if review_id in self.deleted_reviews:
            raise _not_found(f"repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/comments")
        self.calls.append(f"review-comments:{review_id}")
        return self._page(self.review_comments.get(review_id, []), page, per_page)

    def list_issue_comments(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[GitHubComment]:
        self.calls.append(f"issue-comments:{pr_number}")
        return self._page(self.issue_comments.get(pr_number, []), page, per_page)


@pytest.fixture
def registry(tmp_path: Path) -> SqliteQuestionRegistry:
    return SqliteQuestionRegistry(tmp_path / "registry.db")


def _plan(**overrides: object) -> BackfillPlan:
    fields: dict[str, object] = {
        "repositories": (REPO,),
        "since": FLOOR,
        "max_objects": 100,
    }
    fields.update(overrides)
    return BackfillPlan(**fields)  # type: ignore[arg-type]


def _run(
    reader: FakeHistory,
    registry: SqliteQuestionRegistry,
    sink: RecordingSink,
    plan: BackfillPlan | None = None,
) -> object:
    return run_backfill(
        reader=reader,
        registry=registry,
        sink=sink,
        plan=plan or _plan(),
        clock=lambda: READ_AT,
    )


def _pull(
    number: int = 7,
    *,
    created_at: datetime = datetime(2023, 6, 1, tzinfo=UTC),
    updated_at: datetime = SUBMITTED_AT,
    author: str = "dev",
    state: str = "closed",
) -> GitHubPullRequest:
    return GitHubPullRequest(
        number=number,
        title=f"Change {number}",
        state=state,
        user=GitHubUser(login=author),
        created_at=created_at,
        closed_at=updated_at if state == "closed" else None,
        updated_at=updated_at,
    )


def _review(
    review_id: int = 9001,
    *,
    state: str = "approved",
    body: str = "Ship it: the retry is bounded.",
    author: str = "reviewer",
    submitted_at: datetime | None = SUBMITTED_AT,
    association: str | None = "MEMBER",
    account_type: str | None = None,
) -> PullRequestReview:
    return PullRequestReview(
        id=review_id,
        state=state,
        body=body,
        user=GitHubUser(login=author, type=account_type),
        submitted_at=submitted_at,
        author_association=association,
    )


def _inline(
    comment_id: int = 7001,
    *,
    body: str = "This retry never terminates.",
    path: str = "src/x.py",
    line: int = 42,
) -> PullRequestReviewComment:
    return PullRequestReviewComment(
        id=comment_id,
        body=body,
        user=GitHubUser(login="reviewer"),
        created_at=COMMENTED_AT,
        path=path,
        line=line,
        diff_hunk="@@ -1 +1 @@",
    )


def _issue_comment(
    comment_id: int,
    body: str,
    *,
    author: str = "dev",
    created_at: datetime = COMMENTED_AT,
    association: str | None = "MEMBER",
) -> GitHubComment:
    return GitHubComment(
        id=comment_id,
        body=body,
        user=GitHubUser(login=author),
        created_at=created_at,
        author_association=association,
    )


def _history_with_one_review(**review_kwargs: object) -> FakeHistory:
    return FakeHistory(
        pull_requests=[_pull()],
        reviews={7: [_review(**review_kwargs)]},  # type: ignore[arg-type]
        review_comments={9001: [_inline()]},
        issue_comments={7: []},
    )


# --- the stamp: the one place a backfill decides what a record claims ------------


def test_the_stamp_corrects_a_collector_that_hardcodes_its_delivery() -> None:
    """The collectors are built for deliveries and set the source themselves, so
    ``delivery_id=None`` alone would leave a record claiming a signed delivery that
    never happened. This is the correction, and it removes the delivery rather than
    nulling it: the storage layer writes any value it is given."""
    sink = RecordingSink()
    entry = KnowledgeEntry(
        entry_id="review-abc",
        question_text="Review verdict approved on PR #7",
        answer_text="Ship it.",
        category=QuestionCategory.DESIGN_DECISION,
        answered_at=SUBMITTED_AT,
        metadata={
            "repo": REPO,
            "pr_number": 7,
            REVIEW_ID_KEY: 9001,
            "delivery_id": "11111111-1111-1111-1111-111111111111",
        },
        capture_source=CaptureSource.WEBHOOK,
        captured_at=SUBMITTED_AT,
        capture_delivery_id="11111111-1111-1111-1111-111111111111",
    )

    BackfillReadSink(sink, clock=lambda: READ_AT).store(entry)

    assert len(sink.entries) == 1
    stamped = sink.entries[0]
    assert stamped.capture_source is CaptureSource.BACKFILLED
    assert stamped.capture_delivery_id is None
    assert "delivery_id" not in stamped.metadata
    assert stamped.captured_at == READ_AT
    assert stamped.answered_at == SUBMITTED_AT, "the event time is not the read time"
    assert stamped.metadata[REVIEW_ID_KEY] == 9001, "the read anchor survives the rewrite"


def test_a_record_with_no_read_anchor_is_refused_rather_than_stamped() -> None:
    """The stamp is not a place where the anchor rule gets relaxed, and nothing
    reaches the store when it cannot be anchored."""
    sink = RecordingSink()
    unanchorable = KnowledgeEntry(
        entry_id="no-anchor",
        question_text="Review verdict approved on PR #7",
        answer_text="Ship it.",
        category=QuestionCategory.DESIGN_DECISION,
        metadata={"repo": REPO, "pr_number": 7},
    )

    with pytest.raises(UnanchorableReadError, match="capture_read_anchor"):
        BackfillReadSink(sink, clock=lambda: READ_AT).store(unanchorable)

    assert sink.entries == []


def test_the_stored_document_says_backfilled_and_names_no_delivery(
    registry: SqliteQuestionRegistry,
) -> None:
    """The frontmatter is what a reader of the corpus actually sees, and it is
    written by a different module from the one that stamps the record."""
    from kojutsu.core.tanseki_mapping import build_frontmatter

    sink = RecordingSink()
    _run(_history_with_one_review(), registry, sink)

    frontmatter = build_frontmatter(sink.entries[0])
    assert frontmatter["capture_source"] == CaptureSource.BACKFILLED.value
    assert frontmatter["captured_at"] == READ_AT.isoformat()
    assert frontmatter["review_id"] == 9001, "the read anchor travels with the document"
    assert "delivery_id" not in frontmatter, (
        "there was no delivery, and a key carrying one would be read by any consumer "
        "that filters on it"
    )


# --- the bounds: a run that is not bounded must not start ------------------------


def test_the_command_refuses_to_run_without_a_date_floor_or_a_budget() -> None:
    """Both are required, so neither can default to a value this code chose."""
    without_floor = runner.invoke(app, ["backfill-reviews"])
    assert without_floor.exit_code != 0
    assert "--since" in _operator_text(without_floor.output)

    without_budget = runner.invoke(
        app, ["backfill-reviews", "--since", "2024-01-01", "--repo", REPO]
    )
    assert without_budget.exit_code != 0
    assert "--max-objects" in _operator_text(without_budget.output)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "github_webhook_allowed_repositories": REPO,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_a_plan_without_a_repository_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one repository"):
        build_plan(settings=_settings(), repositories=[], since="2024-01-01", max_objects=10)


def test_a_repository_outside_the_allowlist_is_refused() -> None:
    """A backfill reaches into history on its own initiative, so the same scope
    that authorises a delivery has to authorise this."""
    with pytest.raises(ValueError, match="not in GITHUB_WEBHOOK_ALLOWED_REPOSITORIES"):
        build_plan(
            settings=_settings(), repositories=["other/repo"], since="2024-01-01", max_objects=10
        )


def test_a_budget_of_nothing_or_beyond_the_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_plan(settings=_settings(), repositories=[REPO], since="2024-01-01", max_objects=0)

    with pytest.raises(ValueError, match="capped"):
        build_plan(
            settings=_settings(),
            repositories=[REPO],
            since="2024-01-01",
            max_objects=MAX_OBJECTS_PER_RUN + 1,
        )


def test_a_date_floor_is_required_parsed_and_never_in_the_future() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_plan(settings=_settings(), repositories=[REPO], since="last tuesday", max_objects=10)

    with pytest.raises(ValueError, match="in the future"):
        build_plan(
            settings=_settings(),
            repositories=[REPO],
            since=(datetime.now(UTC) + timedelta(days=2)).date().isoformat(),
            max_objects=10,
        )


def test_a_malformed_floor_is_refused_rather_than_read_as_the_beginning_of_time() -> None:
    """An unparsed date must not fall through to an unbounded run."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_plan(settings=_settings(), repositories=[REPO], since="", max_objects=10)


# --- what a reconstructed record is allowed to claim ----------------------------


def test_a_written_record_is_backfilled_and_satisfies_its_own_anchor_rule(
    registry: SqliteQuestionRegistry,
) -> None:
    """The model refuses to build a backfilled record without a read anchor, so
    these records existing at all is the proof the wiring is right."""
    sink = RecordingSink()

    _run(_history_with_one_review(), registry, sink)

    assert sink.entries, "the review should have produced a record"
    for entry in sink.entries:
        assert entry.capture_source is CaptureSource.BACKFILLED
        assert entry.capture_delivery_id is None, (
            "nothing was delivered to a backfill, and an id in this field would be read "
            "as a delivery by anyone filtering on it"
        )
        assert entry.captured_at == READ_AT, (
            "captured_at is the read time for this source, and it is the only evidence "
            "there is of when the object was read"
        )
        gaps = capture_anchor_gaps(
            capture_source=entry.capture_source,
            repo=entry.metadata.get("repo"),
            pr_number=entry.metadata.get("pr_number"),
            captured_at=entry.captured_at,
            delivery_id=entry.capture_delivery_id,
            comment_id=entry.metadata.get("github_comment_id"),
            review_id=entry.metadata.get(REVIEW_ID_KEY),
        )
        assert gaps == []


def test_a_written_record_names_the_review_it_read_and_keeps_answered_at(
    registry: SqliteQuestionRegistry,
) -> None:
    sink = RecordingSink()

    _run(_history_with_one_review(), registry, sink)

    assert sink.kinds() == [REVIEW_KIND_VERDICT, REVIEW_KIND_INLINE]
    for entry in sink.entries:
        assert entry.metadata[REVIEW_ID_KEY] == 9001
        assert entry.answered_at == SUBMITTED_AT, (
            "the event time is the review's own; restamping it with the read time would "
            "file a six-month-old decision under today"
        )


def test_no_head_sha_is_invented_for_a_historical_review(
    registry: SqliteQuestionRegistry,
) -> None:
    """A wrong anchor is checkable, and it is wrong.

    The pull request's current head is not the head this review was written
    against, so anchoring six-month-old evidence to a commit from this morning
    would assert a correspondence that never held.
    """
    pull = _pull()
    pull.head = {"ref": "feature/x", "sha": "aaaaaaa"}
    history = FakeHistory(
        pull_requests=[pull],
        reviews={7: [_review()]},
        review_comments={9001: [_inline()]},
        issue_comments={7: []},
    )
    sink = RecordingSink()

    _run(history, registry, sink)

    assert sink.entries
    for entry in sink.entries:
        assert entry.metadata.get("head_sha") is None


def test_a_review_with_no_submission_time_is_not_admitted_by_guessing(
    registry: SqliteQuestionRegistry,
) -> None:
    """An object that cannot be placed against an editorial boundary is a gap."""
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={7: [_review(submitted_at=None)]},
        review_comments={9001: [_inline()]},
        issue_comments={7: []},
    )

    report = _run(history, registry, sink)

    assert sink.entries == []
    assert report.objects_read == 1, "only the pull request itself was read"


# --- resumability comes from identity, not from a cursor ------------------------


def test_an_interrupted_backfill_re_run_writes_no_duplicate_records(
    registry: SqliteQuestionRegistry,
) -> None:
    """The first run is cut off by its budget; the second covers the same range.

    This is the whole of resumability. The first run finished the first change,
    so it left a receipt and the second run skips its reads outright; the
    second change has no receipt, so it is read and stored. Identity still
    guards the records -- nothing is written twice -- but the skip is what
    keeps the second run from paying for the first run's reads again.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=7), _pull(number=8)],
        reviews={7: [_review(review_id=9001)], 8: [_review(review_id=9002)]},
        review_comments={9001: [_inline(comment_id=7001)], 9002: [_inline(comment_id=7002)]},
        issue_comments={7: [], 8: []},
    )

    # One object's worth of budget: the first change's review stores a verdict and
    # an inline comment and is one object for it, and the second change's review
    # does not fit.
    first = _run(history, registry, sink, _plan(max_objects=1))
    assert first.budget_exhausted is True
    assert first.records_written == 2, "the verdict and its inline comment"
    assert first.objects_new == 1, "one object, however many records it stored"
    assert len(sink.entries) == 2
    assert "review-comments:9002" not in history.calls, (
        "the second change's review was listed and then refused: the listing is one "
        "request and happens before the budget is asked, the reconstruction is the work"
    )

    second = _run(history, registry, sink, _plan(max_objects=100))

    assert second.budget_exhausted is False
    assert second.records_written == 2, "only the change the first run never reached"
    assert second.skipped_finished == 1, "the finished change is skipped, not re-read"
    assert second.already_present == 0, "nothing re-read means nothing to collide on"
    entry_ids = [entry.entry_id for entry in sink.entries]
    assert len(entry_ids) == 4
    assert len(set(entry_ids)) == 4, "identity, not a cursor, is what stopped the duplicates"


def test_a_re_read_comment_collides_on_the_semantic_event_id(
    registry: SqliteQuestionRegistry,
) -> None:
    """Not on its content. The id is derived from the forge's object identity, so a
    comment read twice is one record even if the body is re-serialised."""
    first_sink = RecordingSink()
    second_sink = RecordingSink()
    history = _history_with_one_review()

    _run(history, registry, first_sink)
    # A wider window than the receipt was written under, so the receipt does
    # not apply and the review is genuinely re-read: this is the path that
    # proves the collision is on the forge-derived id rather than on a skip.
    wider = _plan(since=datetime(2023, 1, 1, tzinfo=UTC))
    report = _run(history, registry, second_sink, wider)

    assert second_sink.entries == [], "a second record for the same comment is the failure"
    assert report.records_written == 0
    assert report.already_present == 1
    inline = next(
        entry for entry in first_sink.entries if entry.metadata["record_kind"] == REVIEW_KIND_INLINE
    )
    expected = stable_review_entry_id(semantic_review_event_id(REPO, 7, 9001, comment_id=7001))
    assert inline.entry_id == expected, (
        "the collision has to be on the id the forge implies, or a re-read would only "
        "be deduplicated by accident"
    )


def test_a_change_reviewed_after_the_deployment_is_not_captured_twice(
    registry: SqliteQuestionRegistry,
) -> None:
    """Opened before the deployment, reviewed after it: the live path already wrote
    that record, and the backfill must not overwrite it with a weaker one.

    This is the case a cursor would get wrong and identity gets right. The change
    predates the floor on its ``created`` timestamp, so a backfill ordered by
    creation would skip it; ordered by update, it is found, collides, and the
    signed-delivery record stays exactly as the webhook wrote it.
    """
    live_sink = RecordingSink()
    live = process_review_event_outcome(
        repo=REPO,
        pr_number=7,
        review_id=9001,
        review_state="approved",
        review_body="Ship it: the retry is bounded.",
        review_author="reviewer",
        pr_author_account="dev",
        pr_opened_at=datetime(2023, 6, 1, tzinfo=UTC),
        review_submitted_at=SUBMITTED_AT,
        review_author_association="MEMBER",
        comments=[
            {
                "id": 7001,
                "body": "This retry never terminates.",
                "path": "src/x.py",
                "line": 42,
            }
        ],
        registry=registry,
        sink=live_sink,
        delivery_id="11111111-1111-1111-1111-111111111111",
    )
    assert len(live.outcomes) == 2
    assert all(entry.capture_source is CaptureSource.WEBHOOK for entry in live_sink.entries), (
        "the live path's records are witnessed, not reconstructed"
    )

    backfill_sink = RecordingSink()
    pull = _pull(created_at=datetime(2023, 6, 1, tzinfo=UTC), updated_at=SUBMITTED_AT)
    report = _run(
        FakeHistory(
            pull_requests=[pull],
            reviews={7: [_review()]},
            review_comments={9001: [_inline()]},
            issue_comments={7: []},
        ),
        registry,
        backfill_sink,
    )

    assert backfill_sink.entries == [], "the live record must not be re-captured"
    assert report.records_written == 0
    assert report.already_present == 1
    assert len(live_sink.entries) == 2
    assert [entry.capture_source for entry in live_sink.entries] == [
        CaptureSource.WEBHOOK,
        CaptureSource.WEBHOOK,
    ]


# --- what a backfill must not reconstruct ----------------------------------------


def test_a_deleted_review_produces_nothing_and_the_gap_is_reported(
    registry: SqliteQuestionRegistry,
) -> None:
    """A review withdrawn after the fact has no comments left to read.

    The verdict alone would still read perfectly well, which is why the comments are
    fetched first: the 404 is the only evidence that the object is gone, and a record
    for it would be a reconstruction of something the forge has withdrawn.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={7: [_review()]},
        review_comments={},
        issue_comments={7: []},
        deleted_reviews=frozenset({9001}),
    )

    report = _run(history, registry, sink)

    assert sink.entries == []
    assert report.unreadable == 1
    assert report.records_written == 0
    gap = report.gaps[0]
    assert gap.repository == REPO
    assert gap.pr_number == 7
    assert "review 9001" in gap.what
    assert "no longer has it" in gap.reason


def test_a_historical_comment_without_a_marker_is_not_recorded_as_an_answer(
    registry: SqliteQuestionRegistry,
) -> None:
    """Somebody talking to a colleague is not a conclusion about a decision.

    Kojutsu only started asking questions when it was installed, so a comment
    from before that carries no marker is not an answer to anything, and storing it
    as one would manufacture the question that was never asked.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={7: []},
        review_comments={},
        issue_comments={7: [_issue_comment(5001, "Looks good to me, shipping it.")]},
    )

    report = _run(history, registry, sink)

    assert sink.entries == []
    assert "answer" not in sink.kinds()
    assert report.records_written == 0
    assert report.silent == 1, "the comment was read, and it had nothing to say"
    assert report.unreadable == 0


def test_a_marked_comment_with_no_registered_question_is_still_not_an_answer(
    registry: SqliteQuestionRegistry,
) -> None:
    """The collector holds the line, not a filter written for the backfill.

    A comment carrying a Kojutsu answer marker, whose question the registry has
    never heard of, cannot reach the store by any route — which is what makes the
    previous test a property of the system rather than of this command.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={7: []},
        review_comments={},
        issue_comments={
            7: [_issue_comment(5002, answer_comment_body("q-never-asked", "Because X."))]
        },
    )

    report = _run(history, registry, sink)

    assert sink.entries == []
    assert report.records_written == 0
    assert report.silent == 1


def test_an_answer_to_a_registered_question_is_captured_and_marked_backfilled(
    registry: SqliteQuestionRegistry,
) -> None:
    """The positive case, so the refusal above is a gate and not a blanket denial.

    A question kojutsu did ask, answered in the ordinary way, is reconstructible:
    the marker is the proof a question was asked, and the record is still marked as
    a read rather than a delivery.
    """
    registry.record_question(
        question_id="q-1",
        github_comment_id=5000,
        repo=REPO,
        pr_number=7,
        pr_url="https://github.com/org/repo/pull/7",
        question_text="Why is the retry bounded?",
        question_category="design_decision",
        question_author="bot",
    )
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={7: []},
        review_comments={},
        issue_comments={
            7: [
                _issue_comment(5000, kojutsu_comment_body("q-1", "Why is the retry bounded?")),
                _issue_comment(5003, answer_comment_body("q-1", "Three attempts, then it stops.")),
            ]
        },
    )

    report = _run(history, registry, sink)

    assert report.records_written == 1
    assert [entry.metadata["record_kind"] for entry in sink.entries] == ["answer"]
    entry = sink.entries[0]
    assert entry.capture_source is CaptureSource.BACKFILLED
    assert entry.metadata["github_comment_id"] == 5003


# --- rate limits are honoured, never absorbed ------------------------------------


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real_client = httpx.Client

    def factory(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", factory)


def test_a_429_backs_off_and_resumes_rather_than_failing_the_run(
    monkeypatch: pytest.MonkeyPatch, registry: SqliteQuestionRegistry
) -> None:
    """A rate limit handled as a skip produces a corpus with a hole in it and no
    record that the hole exists, so the read is retried after the stated interval
    and the run continues from the same object."""
    calls: list[str] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path.endswith("/pulls") and calls.count(path) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        if path.endswith("/pulls"):
            return httpx.Response(200, json=[_pull().model_dump(mode="json")])
        if path.endswith("/reviews/9001/comments"):
            return httpx.Response(200, json=[_inline().model_dump(mode="json")])
        if path.endswith("/reviews"):
            return httpx.Response(200, json=[_review().model_dump(mode="json")])
        if path.endswith("/issues/7/comments"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected path {path}")

    _patch_transport(monkeypatch, handler)
    sink = RecordingSink()
    reader = GitHubHistoryReader(token="tok", sleep=slept.append)

    report = run_backfill(
        reader=reader, registry=registry, sink=sink, plan=_plan(), clock=lambda: READ_AT
    )

    assert slept == [7.0], "the interval the forge asked for, honoured rather than guessed"
    assert report.records_written == 2
    assert report.unreadable == 0


def test_a_rate_limit_that_never_clears_stops_the_run_instead_of_skipping_the_object(
    monkeypatch: pytest.MonkeyPatch, registry: SqliteQuestionRegistry
) -> None:
    """The alternative is a hole with nothing in it recording the hole. Failing
    loudly costs a re-run, and a re-run is safe because identity makes the work
    idempotent."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    _patch_transport(monkeypatch, handler)
    sink = RecordingSink()
    reader = GitHubHistoryReader(token="tok", sleep=lambda _seconds: None, rate_limit_retries=2)

    with pytest.raises(RateLimitedError):
        run_backfill(reader=reader, registry=registry, sink=sink, plan=_plan())

    assert sink.entries == []


# --- the run reports what it did ------------------------------------------------


def test_the_run_reports_every_count(registry: SqliteQuestionRegistry) -> None:
    """Including the two that are not flattering: what was already there, and what
    could not be reconstructed at all."""
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=7), _pull(number=8)],
        reviews={
            7: [_review(review_id=9001), _review(review_id=9002, association="NONE")],
            8: [],
        },
        review_comments={9001: [_inline()], 9002: [_inline(comment_id=7002)]},
        issue_comments={7: [_issue_comment(5001, "thanks")], 8: []},
    )

    report = _run(history, registry, sink)

    assert report.objects_read == 5, "two changes, two reviews, one comment"
    assert report.records_written == 4, (
        "two verdicts and two inline comments -- the NONE reviewer is stored too, "
        "because admission is unrestricted by default and the association is now a "
        "fact on the record rather than a gate in front of it"
    )
    assert report.already_present == 0
    assert report.unreadable == 0
    assert report.silent == 1, "the markerless issue comment was read and had nothing to capture"


def test_the_run_stops_at_its_budget_and_says_it_did(
    registry: SqliteQuestionRegistry,
) -> None:
    """Truncation is reported rather than silent: a run that reads three of a
    thousand pull requests and says nothing is indistinguishable from one that read
    three.

    The budget is spent by storing, so the cut-off is provoked by objects that would
    store something. A run over objects that would store nothing walks past all of
    them, which is the whole of the next test.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=number) for number in range(1, 6)],
        reviews={number: [_review(review_id=9000 + number)] for number in range(1, 6)},
        review_comments={},
        issue_comments={},
    )

    report = _run(history, registry, sink, _plan(max_objects=2))

    assert report.budget_exhausted is True
    assert report.objects_new == 2, "two reviews fit, and each is one object"
    assert report.objects_read == 5, "the two changes and their two reviews, plus the third"
    assert "review-comments:9003" not in history.calls, "whose review was never reconstructed"


# --- the budget is spent on new work, so a re-run advances ------------------------


def test_a_second_run_walks_past_what_the_first_stored(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The bug this replaces: a re-run of the range the output recommends made no
    progress at all.** Charged per object read, the walk re-read the same first page,
    spent the whole budget on objects the store already had, and stopped in the same
    place — so a repository with more history than one budget could hold could never
    be walked to its end, however many times it was asked.

    Two full pages of changes, each with one review that stores something, and a
    budget that fits neither page. The first run gets through part of page one; the
    second must get *past* the page the first one filled and reach page two.
    """
    sink = RecordingSink()
    page = PAGE_SIZE
    history = FakeHistory(
        pull_requests=[_pull(number=number) for number in range(1, page + 3)],
        reviews={number: [_review(review_id=9000 + number)] for number in range(1, page + 3)},
        review_comments={},
        issue_comments={},
    )

    first = _run(history, registry, sink, _plan(max_objects=60))
    assert first.budget_exhausted is True
    assert first.objects_new == 60
    first_numbers = {entry.metadata["pr_number"] for entry in sink.entries}
    assert max(first_numbers) < page, "the first run did not finish the page"

    second = _run(history, registry, sink, _plan(max_objects=60))

    assert second.objects_new == 42, "the 40 unread changes of page one, plus the 2 of page two"
    assert second.budget_exhausted is False, "the range was walked to its end"
    assert second.skipped_finished == 60, "the 60 the first run finished are skipped, not re-read"
    assert second.already_present == 0, "nothing re-read means nothing to collide on"
    reached = {entry.metadata["pr_number"] for entry in sink.entries} - first_numbers
    assert page + 1 in reached, (
        "the second run has to reach a change the first one never read, which is the "
        "only evidence that it advanced rather than repeating itself"
    )
    assert "pulls" in history.calls


def test_objects_that_store_nothing_are_not_charged_against_the_budget(
    registry: SqliteQuestionRegistry,
) -> None:
    """**Silences are most of what a walk meets, and they are what pinned it.**

    On the repository this was found on, 168 of the first 200 objects read had
    nothing to capture and 2 were already stored. Charging for duplicates alone
    would have freed two objects out of two hundred and left the walk exactly where
    it was: an object that stores nothing is not new work, and a run that spends its
    budget on reading is a run that cannot move.

    The five reviews are given an empty body to make them store nothing. They used to
    be made silent by an ``author_association`` of ``NONE``, which the default policy
    no longer refuses -- so the fixture would now prove the opposite of its own name
    and the walk would exhaust its budget on work it had not budgeted for.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=number) for number in range(1, 6)],
        reviews={number: [_review(review_id=9000 + number, body="")] for number in range(1, 6)},
        review_comments={},
        issue_comments={},
    )

    report = _run(history, registry, sink, _plan(max_objects=2))

    assert report.silent == 5, "every review was read and none of them could store anything"
    assert report.objects_new == 0
    assert report.budget_exhausted is False, "a budget spent on nothing is not a budget spent"
    assert report.objects_read == 10, "five changes and five reviews, all examined"


def test_a_re_run_over_a_fully_stored_range_is_not_reported_as_a_truncation(
    registry: SqliteQuestionRegistry,
) -> None:
    """Re-reading a range and storing nothing is what a completed backfill looks
    like on its second run. Calling that a truncation would make the number
    meaningless, and so would calling it success without saying what it cost."""
    sink = RecordingSink()
    history = _history_with_one_review()

    _run(history, registry, sink)
    second = _run(history, registry, sink)

    assert second.records_written == 0
    assert second.skipped_finished == 1, "the finished pull request is skipped, not re-read"
    assert second.already_present == 0
    assert second.budget_exhausted is False
    assert second.unreadable == 0


# --- a range bounded at both ends --------------------------------------------------


MAY = datetime(2024, 5, 20, 12, 0, tzinfo=UTC)
APRIL = datetime(2024, 4, 10, 12, 0, tzinfo=UTC)
MARCH = datetime(2024, 3, 20, 12, 0, tzinfo=UTC)
FEBRUARY = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)


def _three_months_of_history() -> FakeHistory:
    """One change per month, most recently updated first, as the forge lists them.

    Each review is dated inside its own change's month, because a review carries
    its own timestamp and the window is applied to that rather than to the change
    it hangs off.
    """
    return FakeHistory(
        pull_requests=[
            _pull(number=7, updated_at=MAY),
            _pull(number=8, updated_at=APRIL),
            _pull(number=9, updated_at=MARCH),
        ],
        reviews={
            7: [_review(review_id=9001, submitted_at=MAY)],
            8: [_review(review_id=9002, submitted_at=APRIL)],
        },
        review_comments={9001: [_inline(comment_id=7001)], 9002: [_inline(comment_id=7002)]},
        issue_comments={7: [], 8: [], 9: []},
    )


def test_a_ceiling_skips_the_too_recent_and_reads_the_window_behind_it(
    registry: SqliteQuestionRegistry,
) -> None:
    """**Without an upper bound the tool answers a different question than the one
    asked.** Enumeration starts at the newest change and descends, so a range with no
    ceiling necessarily includes everything up to now: asking for April on a busy
    repository returns the last page of its present, which is a few days, and the
    summary cannot tell the operator that.

    The newest change in this fixture is in May and the bound excludes it, so the
    May change must not be read while the April one behind it must be. A bound whose
    fixture happened to fit inside the window could not tell bounded from unbounded.
    """
    sink = RecordingSink()
    history = _three_months_of_history()

    report = _run(
        history,
        registry,
        sink,
        _plan(
            since=datetime(2024, 4, 1, tzinfo=UTC),
            until=datetime(2024, 4, 30, 23, 59, 59, 999_999, tzinfo=UTC),
        ),
    )

    assert report.objects_read == 2, "the April change and its review; May is out, March is down"
    assert report.records_written == 2
    assert "reviews:7" not in history.calls, "the May change's reviews were never listed"


def test_a_window_behind_a_whole_page_of_newer_changes_is_still_reached(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The ceiling skips; it does not end the walk.** A page whose every change is
    newer than the bound says the window is further back, not that there is none —
    and on a busy repository most windows are further back.

    Ending the walk there is the mistake this pair of tests exists for: it reported
    `objects-read: 0` for a range that exists, which is indistinguishable from an
    empty repository and is the same silent-truncation failure the command exists to
    prevent.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=number, updated_at=MAY) for number in range(1, PAGE_SIZE + 1)]
        + [_pull(number=PAGE_SIZE + 1, updated_at=APRIL)],
        reviews={PAGE_SIZE + 1: [_review(review_id=9001, submitted_at=APRIL)]},
        review_comments={9001: [_inline(comment_id=7001)]},
        issue_comments={},
    )

    report = _run(
        history,
        registry,
        sink,
        _plan(
            since=datetime(2024, 4, 1, tzinfo=UTC),
            until=datetime(2024, 4, 30, 23, 59, 59, 999_999, tzinfo=UTC),
        ),
    )

    assert report.objects_read == 2, "one whole page of newer changes was passed over"
    assert report.records_written == 2, "the change behind it, and its inline comment"
    assert [entry.metadata["pr_number"] for entry in sink.entries] == [PAGE_SIZE + 1, PAGE_SIZE + 1]


def test_the_same_fixture_reads_the_may_change_when_there_is_no_ceiling(
    registry: SqliteQuestionRegistry,
) -> None:
    """The control for the test above: with no ceiling the walk reads May as well.
    Without this pair, a ceiling that simply stopped the walk would satisfy the
    ceiling test too."""
    sink = RecordingSink()
    history = _three_months_of_history()

    report = _run(history, registry, sink, _plan(since=datetime(2024, 4, 1, tzinfo=UTC)))

    assert report.objects_read == 4, "two changes and their two reviews; March predates the floor"
    assert report.objects_new == 2, "a review is one object however many records it stores"
    assert report.records_written == 4, "a verdict and an inline comment each"
    assert "reviews:9" not in history.calls


def test_a_review_newer_than_the_ceiling_is_outside_the_era_and_is_not_read(
    registry: SqliteQuestionRegistry,
) -> None:
    """The change is inside the window; the review is not. Bounding the range is how
    an operator reconstructs one era, and a review from the next one is not part of
    it however convenient the change would make."""
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=7, updated_at=APRIL)],
        reviews={
            7: [
                _review(review_id=9001, submitted_at=datetime(2024, 4, 10, tzinfo=UTC)),
                _review(review_id=9002, submitted_at=datetime(2024, 4, 12, tzinfo=UTC)),
            ]
        },
        review_comments={9001: [_inline(comment_id=7001)], 9002: [_inline(comment_id=7002)]},
        issue_comments={7: []},
    )

    report = _run(
        history,
        registry,
        sink,
        _plan(
            since=datetime(2024, 4, 1, tzinfo=UTC),
            until=datetime(2024, 4, 11, 23, 59, 59, 999_999, tzinfo=UTC),
        ),
    )

    assert report.objects_read == 2, "the change and the one review inside the window"
    assert report.objects_new == 1
    assert report.records_written == 2, "the surviving review's verdict and its inline comment"
    assert "review-comments:9002" not in history.calls, "the later review was never read"


def test_a_ceiling_is_optional_and_an_inverted_or_malformed_one_is_refused() -> None:
    """Optional because "up to now" is a range someone can mean; refused because an
    inverted range matches nothing and reads exactly like an empty repository."""
    unbounded = build_plan(
        settings=_settings(), repositories=[REPO], since="2024-01-01", max_objects=10
    )
    assert unbounded.until is None

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_plan(
            settings=_settings(),
            repositories=[REPO],
            since="2024-01-01",
            max_objects=10,
            until="last tuesday",
        )

    with pytest.raises(ValueError, match="inverted"):
        build_plan(
            settings=_settings(),
            repositories=[REPO],
            since="2024-06-01",
            max_objects=10,
            until="2024-01-01",
        )


def test_a_ceiling_covers_the_whole_day_it_names() -> None:
    """``--until 2024-09-30`` is the whole of the 30th, the same inclusive reading
    the sibling ``backfill`` gives the same spelling. Carrying its midnight would
    silently drop everything written during the day the operator named."""
    plan = build_plan(
        settings=_settings(),
        repositories=[REPO],
        since="2024-01-01",
        max_objects=10,
        until="2024-09-30",
    )

    assert plan.until is not None
    assert plan.until.date().isoformat() == "2024-09-30"
    assert plan.until.hour == 23 and plan.until.microsecond == 999_999


def test_a_pull_request_predating_the_floor_is_never_reached(
    registry: SqliteQuestionRegistry,
) -> None:
    """Nothing after the floor is touched either: the listing is ascending by
    update, so the first entry that predates the floor ends the walk."""
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[
            _pull(number=7, updated_at=SUBMITTED_AT),
            _pull(number=8, updated_at=datetime(2023, 5, 1, tzinfo=UTC)),
        ],
        reviews={8: [_review(review_id=9999)]},
        review_comments={9999: [_inline(comment_id=7999)]},
        issue_comments={},
    )

    report = _run(history, registry, sink)

    assert report.objects_read == 1
    assert sink.entries == []
    assert "reviews:8" not in history.calls


def test_a_walk_that_reaches_the_page_ceiling_reports_it(
    registry: SqliteQuestionRegistry,
) -> None:
    """A list that quietly stopped short reads as a complete list.

    Ten thousand comments on one pull request is not a shape this corpus has, but
    the run that hit the ceiling must say so rather than report a conversation it
    only partly read.
    """
    sink = RecordingSink()
    ceiling = MAX_PAGES_PER_OBJECT * PAGE_SIZE
    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={},
        review_comments={},
        issue_comments={
            7: [_issue_comment(6000 + index, "no marker here") for index in range(ceiling)]
        },
    )

    report = _run(history, registry, sink, _plan(max_objects=ceiling * 2))

    assert report.unreadable == 1
    assert report.gaps[0].what == "issue comments"
    assert "page ceiling" in report.gaps[0].reason
    assert report.silent == ceiling


def test_the_run_reports_through_the_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The counts belong in the command's output, not only in a returned object."""
    from kojutsu import cli_backfill as cli_module

    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    sink = RecordingSink()

    class FakeRuntime:
        def __enter__(self) -> FakeRuntime:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    runtime = FakeRuntime()
    runtime.registry = registry  # type: ignore[attr-defined]
    runtime.sink = sink  # type: ignore[attr-defined]

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    monkeypatch.setattr(cli_module, "build_runtime", lambda _settings: runtime)
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: _history_with_one_review())
    monkeypatch.setattr(
        cli_module,
        "run_review_backfill",
        lambda **kwargs: run_backfill(**kwargs, clock=lambda: READ_AT),
    )

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "50"],
    )

    assert result.exit_code == 0, result.output
    assert "since=2024-01-01" in result.output
    assert "objects-read: 2" in result.output
    assert "records-written: 2" in result.output
    assert "already-present: 0" in result.output
    assert "unreadable: 0" in result.output


def test_the_command_says_on_stderr_that_unreadable_history_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It is the part of history that was never reconstructed and never will be,
    and it must not be the line a log tail drops."""
    from kojutsu import cli_backfill as cli_module

    class FakeRuntime:
        def __enter__(self) -> FakeRuntime:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    runtime = FakeRuntime()
    runtime.registry = SqliteQuestionRegistry(tmp_path / "registry.db")  # type: ignore[attr-defined]
    runtime.sink = RecordingSink()  # type: ignore[attr-defined]

    history = FakeHistory(
        pull_requests=[_pull()],
        reviews={7: [_review()]},
        review_comments={},
        issue_comments={7: []},
        deleted_reviews=frozenset({9001}),
    )

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    monkeypatch.setattr(cli_module, "build_runtime", lambda _settings: runtime)
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)
    monkeypatch.setattr(
        cli_module,
        "run_review_backfill",
        lambda **kwargs: run_backfill(**kwargs, clock=lambda: READ_AT),
    )

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "50"],
    )

    assert result.exit_code == 0, result.output
    assert "unreadable: 1" in result.output
    assert "gap: org/repo#7 review 9001" in result.output
    assert "not reconstructed" in result.stderr
    assert "will recover them" in result.stderr


def test_the_command_reports_a_stopped_run_rather_than_a_partial_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rate limit that outlives its backoff is an error, not a smaller result."""
    from kojutsu import cli_backfill as cli_module

    class FakeRuntime:
        def __enter__(self) -> FakeRuntime:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    runtime = FakeRuntime()
    runtime.registry = SqliteQuestionRegistry(tmp_path / "registry.db")  # type: ignore[attr-defined]
    runtime.sink = RecordingSink()  # type: ignore[attr-defined]

    def refuse(**_kwargs: object) -> None:
        raise RateLimitedError("GitHub rate-limited and the backoff did not clear it")

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    monkeypatch.setattr(cli_module, "build_runtime", lambda _settings: runtime)
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: FakeHistory())
    monkeypatch.setattr(cli_module, "run_review_backfill", refuse)

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "50"],
    )

    assert result.exit_code == 1
    assert "rate-limited" in result.stderr
    assert "records-written" not in result.stdout


def test_an_enumeration_that_ends_above_the_floor_says_the_range_is_not_covered(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The forge will not list a repository's history for ever.** On
    ``pingdotgg/t3code`` the pull request listing stops after about a thousand
    changes — back to early September — so a run asked for a July floor walks to the
    end of what it was given and stops, having read nothing from July.

    From inside the walk, "the repository has nothing older" and "the forge stopped
    listing" are the same event, so neither can be claimed. What can be said is that
    the floor was not reached, and a range silently missing its own floor is the
    same defect as a range silently cut short by the budget.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=7, updated_at=MAY), _pull(number=8, updated_at=APRIL)],
        reviews={7: [_review(review_id=9001, submitted_at=MAY)]},
        review_comments={9001: [_inline(comment_id=7001)]},
        issue_comments={7: []},
    )

    report = _run(history, registry, sink, _plan(since=FLOOR))

    assert report.objects_read == 3, "both changes and the one review they carry"
    assert report.budget_exhausted is False, "the budget is not what stopped it"
    assert report.floor_unreached is True
    assert report.unreadable == 0, "nothing was read and failed; the history was never listed"


def test_a_range_that_reaches_its_floor_is_not_reported_as_uncovered(
    registry: SqliteQuestionRegistry,
) -> None:
    """The other half of the pair above: a walk that meets the floor knows it has
    covered the range, and saying otherwise would make the word meaningless."""
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[
            _pull(number=7, updated_at=MAY),
            _pull(number=8, updated_at=FEBRUARY),
        ],
        reviews={7: [_review(review_id=9001, submitted_at=MAY)]},
        review_comments={9001: [_inline(comment_id=7001)]},
        issue_comments={7: []},
    )

    report = _run(history, registry, sink, _plan(since=datetime(2024, 3, 1, tzinfo=UTC)))

    assert report.objects_read == 2, "the change and its review; February is below the floor"
    assert report.floor_unreached is False


def test_the_command_names_a_range_it_could_not_cover_to_its_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kojutsu import cli_backfill as cli_module

    _cli_runtime(monkeypatch, SqliteQuestionRegistry(tmp_path / "registry.db"))
    history = FakeHistory(
        pull_requests=[_pull(number=7, updated_at=MAY)],
        reviews={7: [_review(review_id=9001, submitted_at=MAY)]},
        review_comments={9001: [_inline(comment_id=7001)]},
        issue_comments={7: []},
    )
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "50"],
    )

    assert result.exit_code == 0, "an unreachable floor is not a failed run"
    assert "NOT COVERED TO ITS FLOOR" in result.stderr
    assert "2024-01-01" in result.stderr


# --- what refused an object, which is not the same as nothing to say ---------------


def test_a_refused_object_names_the_gate_that_refused_it(
    registry: SqliteQuestionRegistry,
) -> None:
    """**One counter merged two facts with opposite implications.** An object with
    nothing capturable is an absence of review; an object a gate refused is review
    activity the corpus is discarding. On a repository where most reviewing is done
    by accounts outside the allowlist, `silent: 1769` could be read either way and
    the distinction is the whole question this ticket asks.

    The reason has to come from the collector rather than be inferred here: only the
    collector knows which gate fired, and a caller guessing at a list of rules that
    can change under it would be wrong in a way nothing would catch.

    The run is narrowed to ``MEMBER`` because the default admits everything and would
    refuse neither of the first two objects -- see
    ``test_a_review_outside_the_default_associations_stores_nothing``. Narrowing is
    still an operator decision and this is the test that says a narrowed run still
    reports its own refusals rather than letting them merge into `nothing to say`.
    """
    sink = RecordingSink()
    history = FakeHistory(
        pull_requests=[_pull(number=7)],
        reviews={
            7: [
                _review(review_id=9001, association="CONTRIBUTOR"),
                _review(review_id=9002, association="NONE"),
                _review(review_id=9003, association="MEMBER", body=""),
            ]
        },
        review_comments={},
        issue_comments={7: []},
    )

    report = _run(history, registry, sink, _plan(authorized_associations=frozenset({"MEMBER"})))

    assert report.refusals == {
        "author_association_not_authorised": 2,
        "nothing_capturable": 1,
    }, "the association gate and an empty body are different facts about different objects"
    assert report.silent == 3, "still one count of objects that stored nothing, for continuity"
    assert sink.entries == []


def test_an_object_that_stores_something_is_not_counted_as_refused(
    registry: SqliteQuestionRegistry,
) -> None:
    """The mix must be about refusals, not about everything read, or it answers
    nothing. A stored record and a duplicate both leave the mix untouched."""
    sink = RecordingSink()
    history = _history_with_one_review()

    first = _run(history, registry, sink)
    second = _run(history, registry, sink)

    assert first.refusals == {}
    assert second.refusals == {}, "a duplicate is an answer, not a refusal"


def test_the_summary_prints_which_gate_refused_what(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The count is only useful where an operator reads it, and the run summary is
    where the other counts already live."""
    from kojutsu import cli_backfill as cli_module

    _cli_runtime(monkeypatch, SqliteQuestionRegistry(tmp_path / "registry.db"))
    history = FakeHistory(
        pull_requests=[_pull(number=7)],
        reviews={7: [_review(review_id=9001, association="CONTRIBUTOR")]},
        review_comments={},
        issue_comments={7: []},
    )
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        [
            "backfill-reviews",
            "--repo",
            REPO,
            "--since",
            "2024-01-01",
            "--max-objects",
            "50",
            "--authorized-associations",
            "MEMBER",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "silent: 1" in result.stdout
    assert "refused:" in result.stdout
    assert "author_association_not_authorised" in result.stdout
    assert "admitting-author-associations: MEMBER" in result.stdout, (
        "the effective set has to be visible, because --authorized-associations "
        "replaces the policy rather than adding to it"
    )


def test_the_summary_says_all_when_the_policy_is_unrestricted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The line is printed on every run, so printing an empty value beside it would
    read as a bug in the tool rather than as the answer. ``all`` is the answer."""
    from kojutsu import cli_backfill as cli_module

    _cli_runtime(monkeypatch, SqliteQuestionRegistry(tmp_path / "registry.db"))
    history = FakeHistory(
        pull_requests=[_pull(number=7)],
        reviews={7: [_review(review_id=9001, association="CONTRIBUTOR")]},
        review_comments={},
        issue_comments={7: []},
    )
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "50"],
    )

    assert result.exit_code == 0, result.output
    assert "admitting-author-associations: all" in result.stdout
    assert "records-written: 1" in result.output, (
        "the CONTRIBUTOR review is stored; the flag no longer widens anything"
    )


def test_the_help_tells_the_operator_what_a_backfill_does_not_do() -> None:
    """Stated where a user meets it, rather than only where it was designed."""

    result = runner.invoke(app, ["backfill-reviews", "--help"])
    text = _operator_text(result.output)

    assert result.exit_code == 0
    assert "does not make the corpus representative" in text
    assert "policy decision" in text
    assert "Answers are not reconstructed" in text
    assert "no cursor file" in text
    assert "Pull requests: Read" in text
    assert "unreadable" in text
    assert "--until" in text, "the ceiling has to be discoverable to be used"
    assert "bounds new work, not reads" in text


# --- why a review stored nothing is named, not counted ------------------------------


def test_a_refused_review_is_counted_under_its_gate_and_not_as_silence_alone(
    registry: SqliteQuestionRegistry,
) -> None:
    """**One number cannot carry two facts with opposite implications.**

    A review with nothing capturable in it is an absence of review. A review refused
    by a gate is review activity the corpus is *discarding*, and on a repository
    where outsiders and bots do the reviewing that is most of what the walk meets.
    Before the split both were `silent`, so an operator reading the summary could not
    tell a quiet repository from a lossy one -- and a lossy backfill is
    indistinguishable from a complete one, which is the failure this whole module
    exists to prevent.

    So the refusal reason travels with the result and the tally groups by it. This is
    the association gate specifically, and it now fires only for a run that narrowed
    the policy on purpose -- which is exactly why it still has to be reported: a
    narrowing an operator cannot see in the summary is a narrowing they will believe
    is not happening.
    """
    sink = RecordingSink()
    report = _run(
        _history_with_one_review(association="NONE"),
        registry,
        sink,
        plan=_plan(authorized_associations=frozenset({"MEMBER"})),
    )

    assert report.records_written == 0
    assert report.silent == 1
    assert report.refusals == {"author_association_not_authorised": 1}, (
        "the gate that refused it must be visible; `silent` alone does not say whether "
        "there was nothing to say or whether something said it and was not allowed"
    )
    assert sink.entries == []


def test_two_different_gates_are_two_different_keys_and_not_one_bucket(
    registry: SqliteQuestionRegistry,
) -> None:
    """Grouping by cause is the point, so two causes must not collapse.

    A repository with a reviewer outside the policy an operator named and one whose
    reviewer GitHub could not attribute are different operational facts, and an
    operator tuning the association set needs to see which of the two is moving.

    The account gate is checked second, so reaching it needs an association that is
    itself admitted -- which is also what pins the order the two are checked in.
    """
    history = FakeHistory(
        pull_requests=[_pull(), _pull(number=8)],
        reviews={
            7: [_review(review_id=9001, association="NONE")],
            8: [_review(review_id=9002, association="MEMBER", author="")],
        },
        review_comments={9001: [_inline()], 9002: [_inline(comment_id=7002)]},
        issue_comments={7: [], 8: []},
    )
    report = _run(
        history,
        registry,
        RecordingSink(),
        plan=_plan(authorized_associations=frozenset({"MEMBER", "OWNER"})),
    )

    assert report.refusals == {
        "author_association_not_authorised": 1,
        "reviewer_account_missing": 1,
    }


def test_the_command_passes_the_ceiling_it_was_given_to_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The option has to reach the walk, not merely parse. A ceiling that is
    accepted and dropped is the same silent truncation with a nicer interface."""
    from kojutsu import cli_backfill as cli_module

    _cli_runtime(monkeypatch, SqliteQuestionRegistry(tmp_path / "registry.db"))
    history = _three_months_of_history()
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        [
            "backfill-reviews",
            "--repo",
            REPO,
            "--since",
            "2024-04-01",
            "--until",
            "2024-04-30",
            "--max-objects",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "since=2024-04-01 until=2024-04-30" in result.stdout
    assert "objects-read: 2" in result.stdout, "the May change is outside the range asked for"
    assert "records-written: 2" in result.stdout


def test_an_inverted_range_is_refused_by_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)

    result = runner.invoke(
        app,
        [
            "backfill-reviews",
            "--repo",
            REPO,
            "--since",
            "2024-06-01",
            "--until",
            "2024-01-01",
            "--max-objects",
            "50",
        ],
    )

    assert result.exit_code == 1
    assert "inverted" in result.stderr


def test_pull_requests_are_enumerated_most_recently_updated_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Ascending order reconstructs nothing, and reports it as an empty repository.**

    The walk stops on the first entry whose update predates the floor, because the
    reasoning is that everything after it is older. That reasoning needs the pages to
    get *older* as the walk proceeds. Ascending gives the opposite -- it starts at
    the repository's oldest pull request -- so on any long-lived repository the very
    first entry fails the floor and the run stops having read nothing.

    The summary then reads ``objects-read: 0``, which is exactly what a repository
    with nothing to say also produces. That is why this is asserted against the
    outgoing request rather than against the run: the fake reader in this file returns
    pre-ordered lists, so no test here can see the query at all, which is how the
    wrong direction survived in the first place.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=[])

    _patch_transport(monkeypatch, handler)
    reader = GitHubHistoryReader(token="tok")
    reader.list_pull_requests("fastapi", "fastapi", page=1, per_page=100)

    assert seen, "the reader must have made a request to be asserting anything"
    query = dict(httpx.URL(seen[0]).params)
    assert query["sort"] == "updated"
    assert query["direction"] == "desc", (
        "ascending order makes the floor's stop condition fire on the repository's "
        "oldest pull request, so the run reconstructs nothing"
    )


# --- what a truncated run says, and what it returns -------------------------------


def _cli_runtime(monkeypatch: pytest.MonkeyPatch, registry: SqliteQuestionRegistry) -> None:
    from kojutsu import cli_backfill as cli_module

    class FakeRuntime:
        def __enter__(self) -> FakeRuntime:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    runtime = FakeRuntime()
    runtime.registry = registry  # type: ignore[attr-defined]
    runtime.sink = RecordingSink()  # type: ignore[attr-defined]
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_WEBHOOK_ALLOWED_REPOSITORIES", REPO)
    monkeypatch.setattr(cli_module, "build_runtime", lambda _settings: runtime)
    monkeypatch.setattr(
        cli_module,
        "run_review_backfill",
        lambda **kwargs: run_backfill(**kwargs, clock=lambda: READ_AT),
    )


def test_a_budget_stop_is_reported_as_a_truncation_and_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**A truncated backfill is worse than none, because it is indistinguishable
    from a complete one.** The run reported objects read, records written and no
    error, and told the operator to re-run the same range — which, charged per
    object read, could not have helped.

    So it says it was truncated, on stderr where the exit code is not the only thing
    read, and it exits 2 as the sibling `backfill` does. An exit code of 0 is what a
    script is looking for.
    """
    from kojutsu import cli_backfill as cli_module

    _cli_runtime(monkeypatch, SqliteQuestionRegistry(tmp_path / "registry.db"))
    history = FakeHistory(
        pull_requests=[_pull(number=number) for number in range(1, 6)],
        reviews={number: [_review(review_id=9000 + number)] for number in range(1, 6)},
        review_comments={},
        issue_comments={},
    )
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "2"],
    )

    assert result.exit_code == 2, result.output
    assert "TRUNCATED" in result.stderr
    assert "not complete" in result.stderr
    assert "re-run the same range to continue" in result.stderr, (
        "the advice is only worth keeping because the re-run now advances"
    )
    assert "objects-read: 5" in result.stdout
    assert "new-objects: 2" in result.stdout


def test_a_range_whose_objects_are_all_stored_is_not_called_a_truncation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A completed range re-read is the normal outcome of running the same command
    twice, and it exits 0. Reporting it as truncated would make the word mean
    nothing."""
    from kojutsu import cli_backfill as cli_module

    registry = SqliteQuestionRegistry(tmp_path / "registry.db")
    _cli_runtime(monkeypatch, registry)
    history = _history_with_one_review()
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "50"],
    )

    assert result.exit_code == 0, result.output
    assert "TRUNCATED" not in result.output
    assert "walked to its end" in result.output
    assert "records-written: 2" in result.output


def test_a_truncation_blamed_on_the_page_ceiling_does_not_advise_another_identical_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The re-run advice is only true when the budget is what stopped the walk. A
    window that hit the page ceiling will hit it again identically, so advising the
    identical command would be the same false advice in a new place."""
    from kojutsu import cli_backfill as cli_module

    _cli_runtime(monkeypatch, SqliteQuestionRegistry(tmp_path / "registry.db"))
    ceiling = MAX_PAGES_PER_OBJECT * PAGE_SIZE
    history = FakeHistory(
        pull_requests=[_pull(), _pull(number=8)],
        # A review that stores, so the budget of one is spent, and a conversation
        # long enough to reach the page ceiling on the change that stored it.
        reviews={7: [_review(review_id=9001)]},
        review_comments={9001: [_inline(comment_id=7001)]},
        issue_comments={
            7: [_issue_comment(6000 + index, "no marker here") for index in range(ceiling)]
        },
    )
    monkeypatch.setattr(cli_module, "GitHubHistoryReader", lambda token: history)

    result = runner.invoke(
        app,
        ["backfill-reviews", "--repo", REPO, "--since", "2024-01-01", "--max-objects", "1"],
    )

    assert result.exit_code == 2, "a budget stop and a ceiling in one run is still a truncation"

    assert "page ceiling" in result.stdout
    assert "another identical run will hit" in result.stderr
    assert "re-run the same range to continue" not in result.stderr
    assert "author_association_not_authorised" not in result.stderr, (
        "the stored review was authorised, so no association refusal is reported; a "
        "refusal line here would mean the mix counts something other than refusals"
    )


def test_a_stored_review_is_not_a_refusal_and_leaves_the_bucket_empty(
    registry: SqliteQuestionRegistry,
) -> None:
    """The counter is about reviews that stored nothing.

    A review that stored its records has a `refusal` of `None`, so a run that stored
    everything reports an empty mapping rather than a row of zeroes -- which is the
    difference between "nothing was refused" and "nothing was counted".
    """
    report = _run(_history_with_one_review(), registry, RecordingSink())

    assert report.records_written == 2
    assert report.refusals == {}


# --- the association gate is a narrowing decision, and it is opt-in ---------------


@pytest.mark.parametrize("association", [None, "CONTRIBUTOR", "NONE", "FIRST_TIMER", "MEMBER"])
def test_the_default_admits_every_association(
    registry: SqliteQuestionRegistry, association: str | None
) -> None:
    """**The policy reversal, pinned. Read this before setting it back.**

    The default used to be ``{OWNER, MEMBER, COLLABORATOR}`` and this test asserted
    the opposite for every value outside it. It was measured, and it was the wrong
    filter for the thing people believed it was doing: over 400 stored review captures
    re-fetched from ``pingdotgg/t3code`` and ``fastapi/fastapi`` there were 278
    ``MEMBER``, 118 ``COLLABORATOR`` and **zero** ``CONTRIBUTOR`` or ``NONE`` — the
    gate had never refused a review — while on t3code's PR #2829 the *comments* it
    refused were 28 human ones to admit 21 bot ones. A bot is by definition not a
    member or collaborator of anything, so ``CONTRIBUTOR`` is where automation lands:
    widening to it admits every automated reviewer, widening to ``NONE`` admits only
    humans. One field cannot answer both questions at once.

    ``FIRST_TIMER`` and ``None`` are here because the set GitHub can send is not a
    closed one this code gets to enumerate: a value nobody has seen yet has to be
    admitted by a policy that means "all", not refused by a list that has to be
    edited the day GitHub adds another.
    """
    sink = RecordingSink()
    report = _run(
        _history_with_one_review(association=association),
        registry,
        sink,
    )

    assert report.records_written == 2
    assert report.refusals == {}
    verdicts = [e for e in sink.entries if e.metadata.get("record_kind") == "review_verdict"]
    assert verdicts[0].metadata["github_author_association"] == (
        association.upper() if association else None
    ), "the gate is gone; the fact it used to consume is still recorded"


def test_naming_an_association_narrows_the_run_to_exactly_those(
    registry: SqliteQuestionRegistry,
) -> None:
    """The override is still there, and it now means narrowing. Asserted in both
    directions across the pair of tests below, because a one-directional test passes
    just as happily against a gate that is always open or always shut -- which is the
    only way this could have been written and been wrong.
    """
    sink = RecordingSink()
    report = _run(
        _history_with_one_review(association="CONTRIBUTOR"),
        registry,
        sink,
        plan=_plan(authorized_associations=frozenset({"CONTRIBUTOR"})),
    )

    assert report.records_written == 2
    assert report.refusals == {}
    assert [e for e in sink.entries if e.metadata.get("record_kind")]


def test_an_empty_named_set_admits_nothing_rather_than_everything(
    registry: SqliteQuestionRegistry,
) -> None:
    """**``None`` and an empty set mean opposite things, and they used to be read as
    the same one.** The gate was written ``authorized_associations or DEFAULT``, so an
    operator narrowing to nothing got the widest possible policy instead of the
    narrowest, and nothing failed — the run simply stored everything and reported
    ``refusals={}``. That is the worst direction for this bug: an operator who meant
    "store nothing" got "store everything" with a green summary.
    """
    report = _run(
        _history_with_one_review(association="MEMBER"),
        registry,
        RecordingSink(),
        plan=_plan(authorized_associations=frozenset()),
    )

    assert report.records_written == 0
    assert report.refusals == {"author_association_not_authorised": 1}


def test_naming_an_association_replaces_the_set_rather_than_adding_to_it(
    registry: SqliteQuestionRegistry,
) -> None:
    """The named set is the whole set, which is what lets an operator narrow too.

    A union would read more naturally for the common case -- somebody widening the
    gate to reach outside contributors -- but it makes narrowing impossible, and
    narrowing is a real want: an operator who wants only the project's own members and
    collaborators has no way to say so.

    So the flag replaces, and the risk that creates is handled by saying what it
    admitted rather than by softening the semantics. ``--authorized-associations
    CONTRIBUTOR`` therefore refuses MEMBER, which would be a surprise if it were
    silent -- so the run summary prints the effective set.
    """
    sink = RecordingSink()
    report = _run(
        _history_with_one_review(association="MEMBER"),
        registry,
        sink,
        plan=_plan(authorized_associations=frozenset({"CONTRIBUTOR"})),
    )

    assert report.records_written == 0
    assert report.refusals == {"author_association_not_authorised": 1}
    assert sink.entries == []
    assert report.plan.authorized_associations == frozenset({"CONTRIBUTOR"}), (
        "the plan carries the effective set, so the run can report it rather than "
        "leaving the operator to remember what they typed"
    )


# --- machine authorship is recorded, not inferred by the reader -------------------


def test_a_review_by_an_application_account_records_that_it_was_one(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The association cannot answer this, which is why the flag exists.**

    The sampled reviewer accounts behind this field are mostly `CONTRIBUTOR` -- the
    same value an outside human receives -- so a consumer asked "was a machine one
    end of this" has nothing to read. Resolving it once, at write time, means the
    reader never has to know the answer lives in a login or in a payload field. This
    account is reported as ``Bot`` and also happens to carry the suffix, so it pins
    the preferred signal; the suffix-only case is the next test.
    """
    sink = RecordingSink()
    _run(
        _history_with_one_review(
            author="coderabbitai[bot]", association="MEMBER", account_type="Bot"
        ),
        registry,
        sink,
    )

    verdicts = [e for e in sink.entries if e.metadata.get("record_kind") == "review_verdict"]
    assert verdicts, "the fixture stores a review verdict"
    assert verdicts[0].metadata["reviewer_is_machine"] is True


def test_a_review_is_recalled_as_a_machine_from_the_naming_convention_when_the_payload_has_no_type(
    registry: SqliteQuestionRegistry,
) -> None:
    """**The fallback, and it is weaker, and it is why the type is parsed.**

    ``GitHubUser`` used to keep only ``login``, so every review this store had ever
    written was judged on a suffix while the function claimed to report what GitHub
    says. The two agreed on all eight accounts checked, which is exactly why the
    defect survived: agreement is not a source. Asserting the fallback here rather
    than only the preferred path is what stops a later reader assuming the flag is a
    platform report on records written before this change.
    """
    sink = RecordingSink()
    _run(_history_with_one_review(author="cursor[bot]", association="MEMBER"), registry, sink)

    verdicts = [e for e in sink.entries if e.metadata.get("record_kind") == "review_verdict"]
    assert verdicts[0].metadata["reviewer_is_machine"] is True


def test_the_reported_account_type_overrides_the_naming_convention(
    registry: SqliteQuestionRegistry,
) -> None:
    """GitHub's own ``user.type`` is believed over the suffix, in both directions.

    ``type: "User"`` on a ``[bot]`` login is the platform contradicting a naming
    convention, and the platform is the one that has to be believed: a docstring
    claiming "whether GitHub reports this account as an application" is only true if
    the code actually reads GitHub's answer. The residual cost is named in
    ``is_machine_account`` — an application whose login lacks the suffix is missed
    when no ``type`` arrives — but a convention overriding a report would be a larger
    one, because it would make the flag unreliable for exactly the accounts that
    matter most.
    """
    sink = RecordingSink()
    _run(
        _history_with_one_review(author="someone[bot]", association="MEMBER", account_type="User"),
        registry,
        sink,
    )

    verdicts = [e for e in sink.entries if e.metadata.get("record_kind") == "review_verdict"]
    assert verdicts[0].metadata["reviewer_is_machine"] is False


def test_a_review_by_a_human_records_that_it_was_not_one(
    registry: SqliteQuestionRegistry,
) -> None:
    """Recorded as ``False`` rather than omitted, so "human" and "not recorded" differ.

    Omitting the key for a human would make an absent flag ambiguous between a person
    and a collector that never decided, and the whole point of recording it is that a
    reader can weigh the two differently.
    """
    sink = RecordingSink()
    _run(_history_with_one_review(author="a-person"), registry, sink)

    verdicts = [e for e in sink.entries if e.metadata.get("record_kind") == "review_verdict"]
    assert verdicts[0].metadata["reviewer_is_machine"] is False
