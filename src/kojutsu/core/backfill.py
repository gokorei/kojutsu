"""Historical backfill: read a date range of pull requests and collect from it.

The webhook is the fast path. It sees a comment the moment it is written, and
everything Kojutsu has ever captured arrived that way. That is also its blind
spot: a range that predates the webhook, a webhook that was not registered, a
capture process that was down over a weekend. A backfill walks the history
instead, and finds comments the live path never saw.

Two things about this module are worth stating before the code, because both are
the reason it is shaped the way it is.

**It has no write path to GitHub.** Every request it issues is a ``GET``:
enumerate by date, then read each pull request's comments. Nothing here posts,
edits, or deletes anything on the forge, and there is no optional flag that turns
that on -- an ingestion path that can write is a different path with different
review, and the useful property here is that the capability is absent rather than
disabled. ``tests/test_github_range.py`` records every HTTP method that passes
through a stub transport during a full run and fails on any that mutates, so the
claim is checked rather than asserted in a docstring.

**Identity is the comment, not the run.** A capture is keyed on the comment it
came from -- ``stable_answer_entry_id(repo, pr_number, comment_id)`` for an
answer, ``stable_rationale_entry_id`` for a declaration -- and the registry
claims that key before anything is stored. So running an overlapping range twice
stores nothing the second time, and two ranges that overlap by one pull request
agree about it. That is not a special case handled here; it is the same dedupe the
webhook relies on, which is the only reason a resumable run is safe to resume.

The coverage record exists because of the failure this whole module is careful
about. "Collected 12 answers" does not say which twelve pull requests were looked
at, whether the range was truncated, or whether anything was skipped. An operator
comparing a backfill against what they expected to find cannot tell a complete
range from a truncated one, so the range, the truncation, the counts and the
shortfalls all travel together in :class:`BackfillCoverage`.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import date
from typing import Any

from kojutsu.config import Settings
from kojutsu.core.answer_collector import (
    capture_counts,
    process_comment_reply,
    process_review_event_outcome,
)
from kojutsu.core.clarification_collector import collect_clarifications
from kojutsu.core.knowledge_sink import KnowledgeSink
from kojutsu.core.question_registry import QuestionRegistry
from kojutsu.core.rationale_collector import process_rationale_comment_outcome
from kojutsu.integrations.github import (
    GitHubClient,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
    extract_question_id_from_comment_body,
    extract_rationale_claim,
    validate_pull_request_range,
)
from kojutsu.repo_name import split_repo

__all__ = ["BackfillCoverage", "run_backfill"]


@dataclass(frozen=True)
class BackfillCoverage:
    """What one backfill run looked at, and what it did not cover.

    Every field is something an operator would otherwise have to infer. The two
    that matter most are :attr:`truncated` and :attr:`search_total_count`: together
    they say whether the numbers below describe a whole range or the beginning of
    one, and a run that reports 12 captures without them is indistinguishable
    from a run that reported 12 out of several thousand.
    """

    repository: str
    since: str
    until: str
    #: Pull requests the range enumeration returned.
    pull_requests: int
    #: What GitHub's index said matched the range, which is not the count above
    #: whenever anything was capped, filtered, or not yet indexed.
    search_total_count: int
    #: The range is incomplete: capped, filtered, or GitHub's own partial count.
    truncated: bool
    #: Pull requests whose comments were read. Equal to ``pull_requests`` unless
    #: a read failed, in which case :attr:`prs_failed` says so.
    prs_read: int
    prs_failed: int
    #: Comments fetched across every pull request in the range.
    comments_read: int
    answers_captured: int
    rationales_captured: int
    #: Unprompted, trusted comments stored as quotations. Separate from the
    #: counts above because they are a different kind of record, and a run that
    #: reported them inside "answers" would be claiming a question was asked.
    clarifications_captured: int = 0
    #: Review verdicts stored, with their inline comments.
    reviews_captured: int = 0
    #: Hits the search index returned whose own creation date fell outside the
    #: range. The index and the data disagreeing is worth a number: it means the
    #: range filter, not the range, decided what was collected.
    out_of_range: int = 0
    #: Hits that stated no creation date, so could not be shown to be in range.
    undated: int = 0
    #: Search pages fetched, and the whole-run page count for the walk.
    search_pages: int = 0
    #: Errors from a pull request whose comments could not be read. Present so a
    #: partial run reports its gaps rather than looking like a short range.
    errors: tuple[str, ...] = ()
    #: Pull requests whose comment or review listing hit the page cap. Their
    #: ``comments_read`` counts are floors, not totals: comments past the cap
    #: were never fetched, so a run with a nonzero count here is a sample of
    #: those pull requests, not the whole of them.
    prs_truncated: int = 0

    @property
    def complete(self) -> bool:
        """Whether this run enumerated the whole range, with no unread pull requests."""
        return not self.truncated and self.prs_failed == 0 and self.prs_truncated == 0

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable record, for ``--report``."""
        return {
            "repository": self.repository,
            "since": self.since,
            "until": self.until,
            "pull_requests": self.pull_requests,
            "search_total_count": self.search_total_count,
            "truncated": self.truncated,
            "complete": self.complete,
            "prs_read": self.prs_read,
            "prs_failed": self.prs_failed,
            "comments_read": self.comments_read,
            "answers_captured": self.answers_captured,
            "rationales_captured": self.rationales_captured,
            "clarifications_captured": self.clarifications_captured,
            "reviews_captured": self.reviews_captured,
            "out_of_range": self.out_of_range,
            "undated": self.undated,
            "search_pages": self.search_pages,
            "errors": list(self.errors),
            "prs_truncated": self.prs_truncated,
        }

    def summary(self) -> str:
        """A one-screen account of the run, for an operator who has not opened it."""
        lines = [
            f"{self.repository} created {self.since}..{self.until}: "
            f"{self.pull_requests} pull request(s), "
            f"{self.comments_read} comment(s) read, "
            f"{self.answers_captured} answer(s), "
            f"{self.rationales_captured} rationale(s), "
            f"{self.clarifications_captured} clarification(s) and "
            f"{self.reviews_captured} review(s) stored."
        ]
        if self.truncated:
            lines.append(
                f"INCOMPLETE RANGE: GitHub's index reported {self.search_total_count} match(es) "
                f"and {self.pull_requests} were enumerated. This run is a sample of the range, "
                f"not the whole of it."
            )
        if self.out_of_range:
            lines.append(
                f"{self.out_of_range} search hit(s) the index matched were excluded by their "
                f"own creation date; the index and the data disagree about the range."
            )
        if self.undated:
            lines.append(
                f"{self.undated} search hit(s) stated no creation date and were excluded: "
                f"they cannot be shown to be inside the range."
            )
        if self.prs_failed:
            lines.append(
                f"{self.prs_failed} pull request(s) could not be read; re-run the range to "
                f"fill the gaps. Running an overlapping range again does not duplicate."
            )
        if self.prs_truncated:
            lines.append(
                f"{self.prs_truncated} pull request(s) hit the comment page cap; "
                f"their comment counts are floors. Narrow the range or raise the "
                f"cap to collect what is past it."
            )
        for error in self.errors:
            lines.append(f"  {error}")
        return "\n".join(lines)


def _capture_reviews(
    client: GitHubClient,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    *,
    owner: str,
    repo: str,
    repository: str,
    pr_number: int,
    pr_author_account: str | None,
) -> tuple[int, bool]:
    """Capture a PR's submitted reviews and their inline comments.

    This is the same collector the webhook calls, reached historically rather than
    by delivery. Reviews are worth reaching for: a verdict with an explanation is a
    person's judgement about the code, which is the thing this ledger exists to keep,
    and it lives in neither the issue-comment thread nor the answer pairs.

    The review body and the ``diff_hunk`` on each inline comment are attacker-
    reachable text, exactly like a diff, so they are recorded as quoted evidence and
    never treated as anything else.
    """
    reviews_result = client.list_pull_request_reviews(owner, repo, pr_number)
    reviews = reviews_result.items
    if not reviews:
        return 0, reviews_result.truncated
    inline_result = client.list_pull_request_review_comments(owner, repo, pr_number)
    inline = inline_result.items
    truncated = reviews_result.truncated or inline_result.truncated
    by_review: dict[int, list[dict[str, Any]]] = {}
    for comment in inline:
        if comment.pull_request_review_id is None:
            continue
        by_review.setdefault(comment.pull_request_review_id, []).append(
            {
                "id": comment.id,
                "body": comment.body,
                "path": comment.path,
                "line": comment.line,
                "original_line": comment.original_line,
                "side": comment.side,
                "diff_hunk": comment.diff_hunk,
                "commit_id": comment.commit_id,
                "in_reply_to_id": comment.in_reply_to_id,
            }
        )

    stored = 0
    for review in reviews:
        outcomes = process_review_event_outcome(
            repo=repository,
            pr_number=pr_number,
            review_id=review.id,
            review_state=review.state,
            review_body=review.body or "",
            review_author=review.user.login,
            pr_author_account=pr_author_account,
            review_submitted_at=review.submitted_at,
            review_author_association=review.author_association,
            review_author_type=review.user.type,
            comments=by_review.get(review.id, []),
            registry=registry,
            sink=sink,
        )
        written, _already_stored = capture_counts(outcomes)
        stored += written
    return stored, truncated


def _collect_for_pull_request(
    client: GitHubClient,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    *,
    owner: str,
    repo: str,
    repository: str,
    pr_number: int,
    branch: str,
    pr_author_account: str | None = None,
) -> tuple[int, int, int, int, int, bool]:
    """Run the ordinary collectors over one pull request.

    Returns ``(comments_read, answers, rationales, clarifications, reviews,
    truncated)``. The same calls the webhook makes, with the same gates: an
    answer must be an authorised, explicitly associated reply to a registered
    question, a rationale must be a declaration by an authorised account, a
    clarification must be an unprompted comment that is not already counted as
    one of the other two, and a review must be a submitted verdict. A pull
    request with nothing on it yields zeros and is not an error.
    """
    listed = client.list_issue_comments(owner, repo, pr_number)
    comments = listed.items
    truncated = listed.truncated
    comments.sort(key=lambda comment: comment.created_at)
    # Question markers resolved to real comments, so a reply can be attributed to
    # the question it answers. An unregistered question id resolves to nothing and
    # its replies are then not answers to anything -- which is correct, because
    # there is no record of what was asked.
    questions = {
        marker: comment
        for comment in comments
        if (marker := extract_question_id_from_comment_body(comment.body))
    }

    answers = 0
    rationales = 0
    for comment in comments:
        association = comment.author_association
        if extract_rationale_claim(comment.body) is not None:
            captured = process_rationale_comment_outcome(
                comment_body=comment.body,
                comment_id=comment.id,
                comment_author=comment.user.login,
                comment_created_at=comment.created_at,
                author_association=association,
                comment_author_type=comment.user.type,
                repo=repository,
                pr_number=pr_number,
                branch=branch,
                registry=registry,
                sink=sink,
            )
            if captured is not None:
                rationales += 1
        question_id = extract_answer_question_id_from_comment_body(comment.body)
        parent = questions.get(question_id) if question_id else None
        if not question_id or parent is None:
            continue
        created = process_comment_reply(
            new_comment_id=comment.id,
            new_comment_body=comment.body,
            new_comment_author=comment.user.login,
            new_comment_created_at=comment.created_at,
            parent_comment_id=parent.id,
            repo=repository,
            pr_number=pr_number,
            registry=registry,
            sink=sink,
            question_id=question_id,
            new_comment_author_association=association,
            new_comment_author_type=comment.user.type,
            parent_agent_claim=extract_agent_claim(parent.body),
        )
        if created:
            answers += 1

    # Clarifications last, so the registry already knows which comments are
    # questions and which are answers and can exclude them. A comment captured
    # under two headings would be counted twice and would read as "nobody asked"
    # about something that was in fact asked.
    clarifications = len(
        collect_clarifications(
            repo=repository,
            pr_number=pr_number,
            comments=comments,
            registry=registry,
            sink=sink,
        )
    )
    reviews, reviews_truncated = _capture_reviews(
        client,
        registry,
        sink,
        owner=owner,
        repo=repo,
        repository=repository,
        pr_number=pr_number,
        pr_author_account=pr_author_account,
    )
    return (
        len(comments),
        answers,
        rationales,
        clarifications,
        reviews,
        (truncated or reviews_truncated),
    )


def run_backfill(
    client: GitHubClient,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    *,
    repository: str,
    since: str | date,
    until: str | date,
    settings: Settings,
    limit: int = 100,
) -> BackfillCoverage:
    """Collect from every pull request created in ``since..until``.

    The range and the repository are validated before any request leaves, by the
    client: an inverted range and a repository outside the allowlist are both
    refused, and neither costs a request or tells GitHub what was being asked
    for.

    A pull request whose comments cannot be read does not abort the run. It is
    recorded in the coverage as a failure and named in the returned errors,
    because a run that dies on the twelfth of forty pull requests is a run that
    cannot be resumed from a known position -- and the resume is free here, since
    re-running an overlapping range stores nothing it stored before.

    Returns:
        BackfillCoverage: what was read, what was stored, and what was missed.
    """
    owner, repo = split_repo(repository)
    search = client.search_pull_requests(
        owner,
        repo,
        since=since,
        until=until,
        limit=limit,
        settings=settings,
    )

    prs_read = 0
    prs_failed = 0
    prs_truncated = 0
    comments_read = 0
    answers = 0
    rationales = 0
    clarifications = 0
    reviews = 0
    errors: list[str] = []

    for pull in search.pull_requests:
        head = pull.head or {}
        branch = str(head.get("ref", "") or "") if isinstance(head, dict) else ""
        try:
            (
                read,
                captured_answers,
                captured_rationales,
                captured_clarifications,
                captured_reviews,
                read_truncated,
            ) = _collect_for_pull_request(
                client,
                registry,
                sink,
                owner=owner,
                repo=repo,
                repository=repository,
                pr_number=pull.number,
                branch=branch,
                pr_author_account=(pull.user.login if pull.user is not None else None),
            )
        except Exception as exc:  # one unread PR must not end the run
            prs_failed += 1
            tb = traceback.format_exc(limit=3)
            # Bounded: last line carries the cause, full chain would flood coverage.
            tb_tail = "\n".join(tb.strip().splitlines()[-4:])
            errors.append(f"{repository}#{pull.number}: {type(exc).__name__}: {exc}\n{tb_tail}")
            continue
        prs_read += 1
        comments_read += read
        answers += captured_answers
        rationales += captured_rationales
        clarifications += captured_clarifications
        reviews += captured_reviews
        if read_truncated:
            prs_truncated += 1
            errors.append(
                f"{repository}#{pull.number}: comment listing hit the page cap; "
                "comments past the cap were not collected."
            )

    start, end = validate_pull_request_range(since, until)
    return BackfillCoverage(
        repository=repository,
        since=start.isoformat(),
        until=end.isoformat(),
        pull_requests=len(search.pull_requests),
        search_total_count=search.total_count,
        truncated=search.truncated,
        prs_read=prs_read,
        prs_failed=prs_failed,
        prs_truncated=prs_truncated,
        comments_read=comments_read,
        answers_captured=answers,
        rationales_captured=rationales,
        clarifications_captured=clarifications,
        reviews_captured=reviews,
        out_of_range=search.out_of_range,
        undated=search.undated,
        search_pages=search.pages_fetched,
        errors=tuple(errors),
    )
