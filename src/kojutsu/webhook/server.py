"""FastAPI app for GitHub webhooks: issue_comment events -> answer collection."""

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from kojutsu import allowlist
from kojutsu.config import Settings, get_settings
from kojutsu.core.answer_collector import (
    process_check_run_outcome,
    process_comment_reply_outcome,
    process_pr_state_change_outcome,
    process_review_event_outcome,
    record_census_observation_outcome,
    semantic_pr_event_id,
)
from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryOutcome,
    KnowledgeDeliveryStatus,
)
from kojutsu.core.outbox import OutboxOwnershipError, TansekiOutbox
from kojutsu.core.question_registry import DeliveryClaim
from kojutsu.core.rationale_collector import process_rationale_comment_outcome
from kojutsu.integrations.github import (
    KOJUTSU_RATIONALE_PREFIX,
    GitHubClient,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
    extract_question_id_from_comment_body,
)
from kojutsu.integrations.github_models import (
    CheckRunPayload,
    GitHubComment,
    IssueCommentPayload,
    PullRequestPayload,
    PullRequestReviewPayload,
)
from kojutsu.integrations.tanseki import TansekiClient
from kojutsu.integrations.webhook_client import redact_webhook_url, validate_webhook_url
from kojutsu.repo_name import split_repo
from kojutsu.runtime import get_runtime
from kojutsu.webhook.urls import default_webhook_url

logger = logging.getLogger(__name__)
router = APIRouter()

#: The result body every webhook delivery returns. Values are widened beyond
#: ``str | bool`` because review captures report a record count and a synchronize
#: reports a head sha, which is either a string or absent.
WebhookResult = dict[str, "str | bool | int | None"]

COMMENT_VISIBILITY_RETRIES = 3
COMMENT_VISIBILITY_DELAY_SECONDS = 0.05
MAX_WEBHOOK_BODY_BYTES = 1024 * 1024
MAX_COMMENT_CHARS = 65_000
WEBHOOK_RETRY_AFTER_SECONDS = 5
WEBHOOK_MAX_CONCURRENT_REQUESTS = 8
WEBHOOK_RATE_LIMIT = 120
WEBHOOK_RATE_WINDOW_SECONDS = 60.0

#: Webhook event types that reach the capture pipeline. ``pull_request_review``
#: is included because GitHub delivers a review and its inline comments in one
#: payload, so a single subscription covers both review evidence kinds rather than
#: leaving a second, separately-registered event to keep in step.
ACCEPTED_EVENTS = frozenset(
    {
        "issue_comment",
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
        # A check run is a machine report about a commit and is the nearest thing
        # the forge offers to an outcome signal. It is captured as its own record
        # kind, never as a review, and only once it has concluded.
        "check_run",
    }
)

#: Actions on each accepted event that produce a capture. ``synchronize`` is
#: included for ``pull_request`` because pushing a new commit invalidates the
#: questions asked about the previous diff.
ACCEPTED_PULL_REQUEST_ACTIONS = frozenset({"opened", "reopened", "closed", "synchronize"})
#: Actions on a review that produce a capture. ``edited`` is deliberately absent:
#: re-capturing an edited body would store the same reviewer's decision twice
#: under two ids, and the first version is the one that was actually submitted.
ACCEPTED_REVIEW_ACTIONS = frozenset({"submitted"})

#: Actions on a check run that produce a capture. ``rerequested`` and ``created``
#: are excluded: a run that has not concluded says nothing yet, and a re-run is a
#: new check run with a new id that will arrive under ``completed`` in its own
#: right. ``requested`` is excluded for the same reason — it is the start of a run
#: whose conclusion has not happened.
ACCEPTED_CHECK_RUN_ACTIONS = frozenset({"completed"})


class _WebhookProcessingGuard:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._starts: deque[float] = deque()

    def try_begin(self) -> bool:
        now = time.monotonic()
        with self._condition:
            cutoff = now - WEBHOOK_RATE_WINDOW_SECONDS
            while self._starts and self._starts[0] <= cutoff:
                self._starts.popleft()
            if (
                self._active >= WEBHOOK_MAX_CONCURRENT_REQUESTS
                or len(self._starts) >= WEBHOOK_RATE_LIMIT
            ):
                return False
            self._active += 1
            self._starts.append(now)
            return True

    def finish(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            if self._active == 0:
                self._condition.notify_all()

    def wait_for_drain(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


_webhook_processing = _WebhookProcessingGuard()


async def wait_for_webhook_processing(timeout_seconds: float = 5.0) -> bool:
    return await asyncio.to_thread(_webhook_processing.wait_for_drain, timeout_seconds)


@dataclass
class _DeliveryTracker:
    """The delivery claim held right now, for the failure paths.

    Every event branch claims through :func:`_take_delivery` and clears through
    ``_finish_delivery``/``_fail_delivery``. An unexpected exception mid-branch
    leaves the claim recorded here, so the outer handlers release exactly what
    is held -- never a completed delivery, never nothing dressed as something.
    """

    delivery_id: str | None = None
    token: str | None = None


def _take_delivery(
    delivery_id: str, repo: str, event: str, body: bytes, tracker: _DeliveryTracker
) -> str | None:
    """Claim a delivery, recording the claim for the failure paths.

    Returns the claim token, or ``None`` when the delivery is a duplicate and
    there is nothing to track.
    """
    claim = _claim_delivery_or_raise(delivery_id, repo, event, body)
    if claim is None:
        return None
    tracker.delivery_id = delivery_id
    tracker.token = claim
    return claim


def _finish_delivery(delivery_id: str, claim: str, tracker: _DeliveryTracker) -> None:
    """Complete a delivery and stop tracking its claim."""
    _complete_delivery(delivery_id, claim)
    tracker.delivery_id = None
    tracker.token = None


def _fail_delivery(
    delivery_id: str, claim: str, tracker: _DeliveryTracker, *, reason: str, detail: str
) -> NoReturn:
    """Release a delivery that cannot complete, and raise the retryable 503."""
    _release_delivery(delivery_id, claim, reason)
    tracker.delivery_id = None
    tracker.token = None
    raise HTTPException(
        status_code=503,
        detail=detail,
        headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
    )


def _verify_signature(payload_body: bytes, signature: str | None, secret: str) -> bool:
    """Verify HMAC signature for webhook payload."""
    if not secret or not signature:
        return False
    if not signature.startswith("sha256="):
        return False
    expected = (
        "sha256="
        + hmac.new(
            secret.encode(),
            payload_body,
            hashlib.sha256,
        ).hexdigest()
    )
    return hmac.compare_digest(expected, signature)


def _require_repository_allowlist(settings: Settings) -> frozenset[str]:
    """Refuse to serve unless a usable allowlist is configured.

    A malformed entry is an error rather than a skipped one: an entry that
    silently failed to parse would be a repository the operator believes is
    being collected and is not.
    """
    try:
        configured = allowlist.configured_repositories(settings)
    except allowlist.AllowlistError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not configured:
        raise HTTPException(
            status_code=503,
            detail=("No exact repositories are configured in GITHUB_WEBHOOK_ALLOWED_REPOSITORIES"),
        )
    return configured


def _configured_webhook_url(request: Request) -> str:
    lifecycle = getattr(request.app.state, "webhook_lifecycle", None)
    configured = getattr(lifecycle, "webhook_url", None)
    return validate_webhook_url(
        configured if isinstance(configured, str) and configured else default_webhook_url()
    )


def _validated_delivery_id(delivery_id: str | None) -> str:
    if not delivery_id:
        raise HTTPException(status_code=400, detail="A valid X-GitHub-Delivery is required")
    try:
        parsed = UUID(delivery_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="A valid X-GitHub-Delivery is required"
        ) from exc
    if str(parsed) != delivery_id:
        raise HTTPException(status_code=400, detail="A valid X-GitHub-Delivery is required")
    return delivery_id


async def _read_webhook_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
        if declared_size < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
        if declared_size > MAX_WEBHOOK_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Webhook payload is too large")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_WEBHOOK_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Webhook payload is too large")
    return bytes(body)


def _claim_delivery(
    delivery_id: str,
    repo: str,
    event: str,
    body: bytes,
) -> DeliveryClaim:
    runtime = get_runtime()
    payload_hash = hashlib.sha256(body).hexdigest()
    return runtime.registry.claim_delivery(delivery_id, repo, event, payload_hash)


def _complete_delivery(delivery_id: str, claim_token: str) -> None:
    if not get_runtime().registry.complete_delivery(delivery_id, claim_token):
        raise RuntimeError("Webhook delivery claim is no longer active")


def _release_delivery(delivery_id: str, claim_token: str, error: str) -> None:
    released = get_runtime().registry.release_delivery(delivery_id, claim_token, error)
    if not released:
        logger.error("Webhook delivery claim was not released: %s", delivery_id)
        raise RuntimeError("Webhook delivery claim is no longer active")


def _require_repository_allowed(settings: Settings, repo: str) -> None:
    """Refuse a delivery for a repository outside the allowlist.

    Shared by every event type on purpose. An event type that applied the check
    differently, or not at all, would be an authorization hole that no test of the
    other events would catch.
    """
    if not allowlist.repository_allowed(repo, settings):
        raise HTTPException(
            status_code=403,
            detail="Repository is not allowed by GITHUB_WEBHOOK_ALLOWED_REPOSITORIES",
        )


def _claim_delivery_or_raise(delivery_id: str, repo: str, event: str, body: bytes) -> str | None:
    """Claim a delivery, raising on conflict or contention.

    Returns ``None`` when the delivery is a duplicate, which is the one outcome the
    caller handles by returning rather than raising: GitHub redelivers, and a
    redelivery that already produced its record must not be reported as a failure.
    """
    claim = _claim_delivery(delivery_id, repo, event, body)
    if claim == "conflict":
        raise HTTPException(
            status_code=409, detail="Delivery identity does not match the original request"
        )
    if claim == "active":
        raise HTTPException(
            status_code=503,
            detail="Webhook delivery is already being processed; retry later",
            headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
        )
    if claim == "duplicate":
        return None
    return claim


def _release_delivery_after_failure(delivery_id: str, claim_token: str, error: str) -> None:
    try:
        _release_delivery(delivery_id, claim_token, error)
    except Exception as exc:
        logger.exception("Webhook delivery lease release failed for %s", delivery_id)
        raise HTTPException(
            status_code=503,
            detail="Webhook delivery lease could not be released; retry after the lease expires",
            headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
        ) from exc


def _process_rationale_comment(
    payload: IssueCommentPayload, delivery_id: str | None
) -> WebhookResult:
    """Capture a declared decision rationale posted on a change.

    Separate from the answer path because the two records answer different
    questions and have different trust. This one has no parent question to resolve,
    so there is no comment-visibility wait: a rationale is a standalone
    declaration, not a reply the collector has to find a question for.
    """
    runtime = get_runtime()
    outcome = process_rationale_comment_outcome(
        comment_body=payload.comment.body,
        comment_id=payload.comment.id,
        comment_author=payload.comment.user.login,
        comment_created_at=payload.comment.created_at,
        author_association=payload.comment.author_association,
        # The forge's own account kind, which the collector prefers over the login
        # suffix for the automation flag it records on every stored declaration.
        comment_author_type=payload.comment.user.type,
        repo=payload.repository.full_name,
        pr_number=payload.issue.number,
        # The branch is not in the issue-comment payload, so the marker is the only
        # place it can come from. The collector prefers the marker's value and
        # falls back to this, which is empty for a comment posted without one.
        branch="",
        registry=runtime.registry,
        sink=runtime.sink,
        delivery_id=delivery_id,
    )
    if outcome is None:
        # Most comments on a pull request are not rationales, and a comment from an
        # unauthorized association is refused rather than stored. Both are ordinary
        # outcomes, so neither is reported as a failure.
        return {"status": "ignored", "reason": "not_a_capturable_rationale"}
    return {"status": "processed", "stored": True, "delivery": outcome.status.value}


def _find_parent_comment(
    gh: GitHubClient,
    owner: str,
    repo_name: str,
    issue_number: int,
    comment_id: int,
    question_id: str,
) -> GitHubComment | None:
    """One visibility poll: list the comments and pick the question's own."""
    comments = gh.list_issue_comments(owner, repo_name, issue_number).items
    return next(
        (
            comment
            for comment in comments
            if comment.id != comment_id
            and extract_question_id_from_comment_body(comment.body) == question_id
        ),
        None,
    )


async def _await_parent_comment(
    token: str,
    owner: str,
    repo_name: str,
    issue_number: int,
    comment_id: int,
    question_id: str,
) -> GitHubComment | None:
    """Poll until the question comment is visible, without holding a thread.

    GitHub's list endpoint can lag the delivery that triggered it, so the first
    poll may not see the question comment yet. The waits between polls are
    ``asyncio.sleep``, not ``time.sleep``: this runs on the request path, and a
    blocking sleep here would pin a threadpool thread per delivery -- under
    burst (``WEBHOOK_MAX_CONCURRENT_REQUESTS=8`` times three retries) that
    starves the pool that ``asyncio.to_thread`` capture work itself needs.

    One client across the polls, closed by the block: the retries are waiting
    for GitHub to make a comment visible, and rehandshaking between them would
    add a round trip to a loop whose whole purpose is to spend as little time
    as possible between attempts.
    """
    with GitHubClient(token=token) as gh:
        parent: GitHubComment | None = None
        for attempt in range(COMMENT_VISIBILITY_RETRIES):
            parent = await asyncio.to_thread(
                _find_parent_comment,
                gh,
                owner,
                repo_name,
                issue_number,
                comment_id,
                question_id,
            )
            if parent is not None or attempt == COMMENT_VISIBILITY_RETRIES - 1:
                break
            await asyncio.sleep(COMMENT_VISIBILITY_DELAY_SECONDS)
        return parent


def _process_issue_comment(
    payload: IssueCommentPayload,
    settings: Settings,
    delivery_id: str | None,
    parent: GitHubComment | None,
) -> WebhookResult:
    if len(payload.comment.body) > MAX_COMMENT_CHARS:
        return {"status": "ignored", "reason": "comment_too_large"}

    # Precedence, not fallthrough. A comment carrying both markers is captured
    # once, as the thing it actually claims to be. Trying the rationale path and
    # then falling through to the answer path would store two records from one
    # comment, and the duplicate would be indistinguishable from two genuine
    # declarations -- the same failure semantic_review_event_id exists to prevent.
    if KOJUTSU_RATIONALE_PREFIX in payload.comment.body:
        return _process_rationale_comment(payload, delivery_id)

    question_id = extract_answer_question_id_from_comment_body(payload.comment.body)
    if not question_id:
        return {"status": "ignored", "reason": "missing_answer_marker"}

    if parent is None:
        return {"status": "pending", "reason": "question_comment_not_visible"}

    runtime = get_runtime()
    outcome = process_comment_reply_outcome(
        new_comment_id=payload.comment.id,
        new_comment_body=payload.comment.body,
        new_comment_author=payload.comment.user.login,
        new_comment_created_at=payload.comment.created_at,
        parent_comment_id=parent.id,
        new_comment_author_association=payload.comment.author_association,
        new_comment_author_type=payload.comment.user.type,
        repo=payload.repository.full_name,
        pr_number=payload.issue.number,
        registry=runtime.registry,
        sink=runtime.sink,
        question_id=question_id,
        delivery_id=delivery_id,
        parent_agent_claim=extract_agent_claim(parent.body),
    )
    result: WebhookResult = {"status": "processed", "stored": outcome is not None}
    if outcome is not None:
        result["delivery"] = outcome.status.value
    return result


def _process_check_run(payload: CheckRunPayload, delivery_id: str | None) -> WebhookResult:
    """Store a concluded check run, or say why it was not stored."""
    active = get_runtime()
    run = payload.check_run
    outcome = process_check_run_outcome(
        repo=payload.repository.full_name,
        check_run_id=run.id,
        check_name=run.name,
        check_status=run.status,
        check_conclusion=run.conclusion,
        head_sha=run.head_sha,
        pr_number=run.pr_number,
        registry=active.registry,
        sink=active.sink,
        delivery_id=delivery_id,
    )
    return {
        "status": "processed",
        "stored": outcome is not None,
        "conclusion": run.conclusion,
    }


def _head_sha(pull_request: Any) -> str | None:
    """The commit the pull request head pointed at, or nothing.

    Free: it is already on the payload. It is the one half of the code anchor
    that costs nothing, and it is what makes an ``answered`` record about a
    specific commit rather than about a moment.
    """
    head = getattr(pull_request, "head", None)
    if not isinstance(head, dict):
        return None
    sha = head.get("sha")
    return sha if isinstance(sha, str) and sha else None


def _changed_files(repo_full_name: str, pr_number: int, token: str) -> list[str] | None:
    """The files this change touches, or ``None`` when they could not be read.

    The webhook payload carries no file list, so this is one extra authenticated
    read per capture — the same shape as the comment lookup above, and worth the
    same cost, because without it nothing in the store can answer a question
    about a file. That is what makes a record about code rather than about a pull
    request, and it is the axis every reader actually works along.

    It is best-effort and silent on failure, deliberately. A file list is context,
    not provenance: the record is a signed delivery whether or not the paths could
    be read, and losing a review to a slow or rate-limited listing would trade a
    captured judgement for a file list.

    ``None`` and ``[]`` are different facts and stay different. ``None`` says the
    paths were not available; ``[]`` says the change touched no files, which is
    not a thing a real pull request does. Collapsing them would make a store
    outage indistinguishable from a genuinely file-free change.
    """
    try:
        owner, name = split_repo(repo_full_name)
    except ValueError:
        return None
    if not token:
        return None
    try:
        # Built for one read, so scoped to it. A client left to be dropped holds
        # its connection pool until the process exits, and this runs per review.
        with GitHubClient(token=token) as client:
            result = client.get_pr_files(owner, name, pr_number)
    except Exception as exc:  # a missing file list must not lose a review
        logger.warning(
            "Could not read changed files for %s#%d: %s: %s",
            repo_full_name,
            pr_number,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return None
    if result.truncated:
        logger.warning(
            "Changed files for %s#%d hit the page cap; the record's file context is partial.",
            repo_full_name,
            pr_number,
        )
    return result.items


def _record_census_observation(
    *,
    repo: str,
    pr_number: int,
    action: str,
    delivery_id: str | None,
    change_author_account: str | None = None,
    head_sha: str | None = None,
) -> KnowledgeDeliveryOutcome | None:
    """Record that a delivery was processed and captured nothing.

    One sink, one outbox: a census record is a delivery to the store like any
    other, and a second write path for it would be a second thing that can lose a
    record. The outcome is returned rather than swallowed so the caller can hold
    the delivery open when the observation itself could not be stored.
    """
    return record_census_observation_outcome(
        repo=repo,
        pr_number=pr_number,
        action=action,
        delivery_id=delivery_id,
        sink=get_runtime().sink,
        change_author_account=change_author_account,
        head_sha=head_sha,
    )


def _process_pull_request_review(
    payload: PullRequestReviewPayload, settings: Settings, delivery_id: str | None
) -> WebhookResult:
    """Capture a review verdict and the inline comments submitted with it.

    The review's own body is stored as quoted evidence. It arrives from the same
    untrusted surface as a pull request diff, and a review body is exactly where a
    prompt injection would be most tempting to place, so it is never treated as
    anything other than text to be recorded.
    """
    runtime = get_runtime()
    review = payload.review
    comments = [
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
        for comment in payload.comments
    ]
    capture = process_review_event_outcome(
        repo=payload.repository.full_name,
        pr_number=payload.pull_request.number,
        review_id=review.id,
        review_state=review.state,
        review_body=review.body or "",
        review_author=review.user.login,
        pr_author_account=(
            payload.pull_request.user.login if payload.pull_request.user is not None else None
        ),
        pr_opened_at=payload.pull_request.created_at,
        head_sha=_head_sha(payload.pull_request),
        files_changed=_changed_files(
            payload.repository.full_name, payload.pull_request.number, settings.github_token
        ),
        review_submitted_at=review.submitted_at,
        review_author_association=review.author_association,
        review_author_type=review.user.type,
        comments=comments,
        registry=runtime.registry,
        sink=runtime.sink,
        delivery_id=delivery_id,
    )
    # The one rule, and it is the whole rule: a delivery that was accepted and
    # stored nothing is an observation. An ignored action was never processed, and a
    # re-delivered review has already been processed once — neither may be written
    # down as silence, and only this point can tell the second apart from a review
    # that genuinely had nothing to say. No reason is recorded with it, not even the
    # one this system knows: an unauthorised reviewer is a fact about kojutsu's
    # authorisation policy, and a policy written into the corpus as a reason would
    # read as a finding about the reviewer.
    census: KnowledgeDeliveryOutcome | None = None
    if not capture.outcomes and not capture.already_stored:
        census = _record_census_observation(
            repo=payload.repository.full_name,
            pr_number=payload.pull_request.number,
            # The delivery's own action, so a review-event observation is a
            # different document from a pull-request one for the same change
            # without needing a field to say which it was.
            action=payload.action,
            delivery_id=delivery_id,
            change_author_account=(
                payload.pull_request.user.login if payload.pull_request.user is not None else None
            ),
            head_sha=_head_sha(payload.pull_request),
        )
    result: WebhookResult = {
        "status": "processed",
        "stored": bool(capture.outcomes),
        "records": len(capture.outcomes),
        "state": review.state,
    }
    if census is not None:
        result["census"] = census.status.value
    return result


def _process_pull_request_synchronize(
    payload: PullRequestPayload, delivery_id: str | None
) -> WebhookResult:
    """Record a new commit pushed to an open pull request.

    A synchronize is not a lifecycle transition and is not stored as one. It is
    recorded against the question set for the PR, marking the previous questions
    ``superseded`` because they were asked about code that has since changed, and
    recording the new head commit so a later reader can tell which revision the
    outstanding questions are about.

    It also captures nothing, by design rather than by accident: there is nobody in
    a push who said anything. That is what makes it an observation — the honest
    description of a change that moved ten times and was never discussed, which
    from outside a system that only writes down what it saw is indistinguishable
    from a change Kojutsu never saw at all. ``stored`` keeps reporting the
    question bookkeeping; the observation is reported under its own key because it
    is not a capture and must not be counted as one.
    """
    runtime = get_runtime()
    head = payload.pull_request.head or {}
    head_sha = head.get("sha") if isinstance(head, dict) else None
    superseded = runtime.registry.supersede_questions_for_pr(
        repo=payload.repository.full_name,
        pr_number=payload.pull_request.number,
        head_sha=head_sha if isinstance(head_sha, str) else None,
    )
    census = _record_census_observation(
        repo=payload.repository.full_name,
        pr_number=payload.pull_request.number,
        action=payload.action,
        delivery_id=delivery_id,
        change_author_account=(
            payload.pull_request.user.login if payload.pull_request.user is not None else None
        ),
        head_sha=head_sha if isinstance(head_sha, str) else None,
    )
    result: WebhookResult = {
        "status": "processed",
        "stored": True,
        "superseded": superseded,
        "head_sha": head_sha if isinstance(head_sha, str) else None,
    }
    if census is not None:
        result["census"] = census.status.value
    return result


def _process_pull_request(
    payload: PullRequestPayload, settings: Settings, delivery_id: str | None
) -> WebhookResult:
    if payload.action == "synchronize":
        return _process_pull_request_synchronize(payload, delivery_id)
    runtime = get_runtime()
    event_identity = semantic_pr_event_id(
        payload.repository.full_name,
        payload.pull_request.number,
        payload.action,
        payload.pull_request.state,
        payload.pull_request.closed_at,
        pr_merged_at=payload.pull_request.merged_at,
        pr_updated_at=payload.pull_request.updated_at,
    )
    outcome = process_pr_state_change_outcome(
        action=payload.action,
        pr_number=payload.pull_request.number,
        pr_title=payload.pull_request.title,
        pr_state=payload.pull_request.state,
        pr_closed_at=payload.pull_request.closed_at,
        repo=payload.repository.full_name,
        registry=runtime.registry,
        sink=runtime.sink,
        delivery_id=delivery_id,
        event_identity=event_identity,
        # Both of these arrive on every pull request payload and were previously
        # used only to derive the event identity, where they perturb a hash and
        # stop. Passing them through is what makes the record say how the change
        # ended rather than only that it did.
        pr_merged_at=payload.pull_request.merged_at,
        pr_opened_at=payload.pull_request.created_at,
        pr_author_account=(
            payload.pull_request.user.login if payload.pull_request.user is not None else None
        ),
        head_sha=_head_sha(payload.pull_request),
        files_changed=_changed_files(
            payload.repository.full_name, payload.pull_request.number, settings.github_token
        ),
    )
    result: WebhookResult = {"status": "processed", "stored": outcome is not None}
    if outcome is not None:
        result["delivery"] = outcome.status.value
    # No observation is written here, and the missing record is the point. A
    # lifecycle action either stores its transition or is a re-delivery of one that
    # is already stored, so ``None`` means "the record for this very event exists",
    # never "this change has nothing". Writing a census record from it would put a
    # "nothing was captured" document beside the capture for the same event, and a
    # count over census documents would then report changes that did produce
    # knowledge. Nothing is read from the registry to tell the two apart, because
    # there is nothing to tell: the one accepted action that captures nothing is
    # handled above.
    return result


@router.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
    x_github_delivery: str | None = Header(None, alias="X-GitHub-Delivery"),
) -> WebhookResult:
    if not _webhook_processing.try_begin():
        raise HTTPException(
            status_code=429,
            detail="Webhook processing capacity exceeded",
            headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
        )
    try:
        return await _github_webhook_impl(
            request,
            x_github_event=x_github_event,
            x_hub_signature_256=x_hub_signature_256,
            x_github_delivery=x_github_delivery,
        )
    finally:
        _webhook_processing.finish()


async def _handle_review_event(
    body: bytes,
    x_github_delivery: str | None,
    x_github_event: str,
    settings: Settings,
    tracker: _DeliveryTracker,
) -> WebhookResult:
    """Capture a submitted review or its inline comments."""
    review_payload = PullRequestReviewPayload.model_validate_json(body)
    if review_payload.action not in ACCEPTED_REVIEW_ACTIONS:
        return {"status": "ignored", "action": review_payload.action}
    _require_repository_allowed(settings, review_payload.repository.full_name)
    delivery_id = _validated_delivery_id(x_github_delivery)
    claim = _take_delivery(
        delivery_id, review_payload.repository.full_name, x_github_event, body, tracker
    )
    if claim is None:
        return {"status": "duplicate", "stored": False}
    result = await asyncio.to_thread(
        _process_pull_request_review, review_payload, settings, delivery_id
    )
    # An observation that could not be delivered is not a smaller event than
    # a capture that could not be: the claim "this delivery was processed
    # and captured nothing" is unverifiable while the record is missing, so
    # the delivery is held open for retry rather than completed as though
    # the silence had been written down.
    if result.get("census") == KnowledgeDeliveryStatus.DEAD_LETTERED.value:
        _fail_delivery(
            delivery_id,
            claim,
            tracker,
            reason="Census observation dead-lettered",
            detail="Observation was dead-lettered; operator action and webhook retry required",
        )
    _finish_delivery(delivery_id, claim, tracker)
    return result


async def _handle_check_run_event(
    body: bytes,
    x_github_delivery: str | None,
    x_github_event: str,
    settings: Settings,
    tracker: _DeliveryTracker,
) -> WebhookResult:
    """Capture a concluded check run."""
    check_payload = CheckRunPayload.model_validate_json(body)
    if check_payload.action not in ACCEPTED_CHECK_RUN_ACTIONS:
        return {"status": "ignored", "action": check_payload.action}
    _require_repository_allowed(settings, check_payload.repository.full_name)
    delivery_id = _validated_delivery_id(x_github_delivery)
    claim = _take_delivery(
        delivery_id, check_payload.repository.full_name, x_github_event, body, tracker
    )
    if claim is None:
        return {"status": "duplicate", "stored": False}
    result = await asyncio.to_thread(_process_check_run, check_payload, delivery_id)
    _finish_delivery(delivery_id, claim, tracker)
    return result


async def _handle_issue_comment_event(
    body: bytes,
    x_github_delivery: str | None,
    x_github_event: str,
    settings: Settings,
    tracker: _DeliveryTracker,
) -> WebhookResult:
    """Capture an answer or declaration posted as an issue comment."""
    payload = IssueCommentPayload.model_validate_json(body)
    if payload.action != "created":
        return {"status": "ignored", "action": payload.action}
    _require_repository_allowed(settings, payload.repository.full_name)
    if not settings.github_token:
        raise HTTPException(
            status_code=503,
            detail="GITHUB_TOKEN is required; retry the webhook delivery",
        )
    delivery_id = _validated_delivery_id(x_github_delivery)
    claim = _take_delivery(delivery_id, payload.repository.full_name, x_github_event, body, tracker)
    if claim is None:
        return {"status": "duplicate", "stored": False}
    # The visibility wait stays on the event loop (asyncio.sleep between
    # polls) rather than inside the worker thread: a blocking sleep in
    # ``to_thread`` pins a pool thread per delivery and starves capture
    # work under burst. The predicates mirror ``_process_issue_comment``'s
    # own ignore checks exactly, so a comment it would ignore issues no
    # GitHub read here either.
    parent = None
    if (
        len(payload.comment.body) <= MAX_COMMENT_CHARS
        and KOJUTSU_RATIONALE_PREFIX not in payload.comment.body
        and extract_answer_question_id_from_comment_body(payload.comment.body) is not None
    ):
        owner, repo_name = payload.repository.full_name.split("/", 1)
        parent = await _await_parent_comment(
            settings.github_token,
            owner,
            repo_name,
            payload.issue.number,
            payload.comment.id,
            extract_answer_question_id_from_comment_body(payload.comment.body) or "",
        )
    result = await asyncio.to_thread(_process_issue_comment, payload, settings, delivery_id, parent)
    if result.get("status") == "pending":
        _fail_delivery(
            delivery_id,
            claim,
            tracker,
            reason="Question comment is not visible yet",
            detail="Question comment is not visible yet; retry the webhook delivery",
        )
    if result.get("delivery") == KnowledgeDeliveryStatus.DEAD_LETTERED.value:
        _fail_delivery(
            delivery_id,
            claim,
            tracker,
            reason="Tanseki delivery dead-lettered",
            detail="Capture was dead-lettered; operator action and webhook retry required",
        )
    _finish_delivery(delivery_id, claim, tracker)
    return result


async def _handle_pull_request_event(
    body: bytes,
    x_github_delivery: str | None,
    x_github_event: str,
    settings: Settings,
    tracker: _DeliveryTracker,
) -> WebhookResult:
    """Capture a pull request lifecycle transition."""
    payload = PullRequestPayload.model_validate_json(body)
    if payload.action not in ACCEPTED_PULL_REQUEST_ACTIONS:
        return {"status": "ignored", "action": payload.action}
    _require_repository_allowed(settings, payload.repository.full_name)
    delivery_id = _validated_delivery_id(x_github_delivery)
    claim = _take_delivery(delivery_id, payload.repository.full_name, x_github_event, body, tracker)
    if claim is None:
        return {"status": "duplicate", "stored": False}
    result = await asyncio.to_thread(_process_pull_request, payload, settings, delivery_id)
    if result.get("delivery") == KnowledgeDeliveryStatus.DEAD_LETTERED.value:
        _fail_delivery(
            delivery_id,
            claim,
            tracker,
            reason="Tanseki delivery dead-lettered",
            detail="Capture was dead-lettered; operator action and webhook retry required",
        )
    if result.get("census") == KnowledgeDeliveryStatus.DEAD_LETTERED.value:
        _fail_delivery(
            delivery_id,
            claim,
            tracker,
            reason="Census observation dead-lettered",
            detail="Observation was dead-lettered; operator action and webhook retry required",
        )
    _finish_delivery(delivery_id, claim, tracker)
    return result


async def _github_webhook_impl(
    request: Request,
    x_github_event: str | None,
    x_hub_signature_256: str | None,
    x_github_delivery: str | None,
) -> WebhookResult:
    """Authenticate, validate and route one delivery. Nothing is stored here.

    **The character policy is deliberately not applied to the payload at this seam**,
    even though this is where the webhook body is parsed and it is the obvious place to
    put one. Two things above this function would break if it were:

    - The body is signed. :func:`_verify_signature` covers these exact bytes, so
      rewriting a character before the delivery claim invalidates the signature against
      itself and every delivery becomes a 401.
    - Markers are read verbatim and must stay that way. The ``kojutsu:`` grammar is what
      makes a comment a Kojutsu record, and
      :func:`kojutsu.integrations.github.normalise_captured_text` is explicit that a
      token whose punctuation had to be repaired in order to parse is not one this
      system wrote -- parsing fails closed on a mangled marker. Sanitising the payload
      before its markers are read would repair exactly those markers, and a forged
      marker would start being honoured.

    So the boundary where untrusted text becomes *stored* text is the collectors in
    :mod:`kojutsu.core.answer_collector`, which own the record schema and therefore own
    the place a removed character can be recorded. See
    :mod:`kojutsu.core.text_hygiene` for the policy and
    :func:`kojutsu.core.text_hygiene.describe_removals` for why it is measured against
    the payload rather than against the stored value.

    Every field this module reads is forwarded to one of those collectors, so nothing
    untrusted reaches storage through a path that does not pass through them. The one
    exception is worth naming: a rationale is forwarded to
    :mod:`kojutsu.core.rationale_collector`, which applies the extractors' narrower set
    and has nowhere to record what it removed.
    """
    body = await _read_webhook_body(request)
    settings = get_settings()
    if not settings.github_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook authentication is not configured")
    if not _verify_signature(body, x_hub_signature_256, settings.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event not in ACCEPTED_EVENTS:
        return {"status": "ignored", "event": x_github_event or "unknown"}

    # Every event is claimed through one path so that authorization and
    # deduplication cannot drift apart between event types. A review event
    # that silently skipped the allowlist would be a worse outcome than not
    # subscribing to reviews at all, because the operator would believe the
    # signal was covered.
    tracker = _DeliveryTracker()
    try:
        if x_github_event in ("pull_request_review", "pull_request_review_comment"):
            return await _handle_review_event(
                body, x_github_delivery, x_github_event, settings, tracker
            )
        if x_github_event == "check_run":
            return await _handle_check_run_event(
                body, x_github_delivery, x_github_event, settings, tracker
            )
        if x_github_event == "issue_comment":
            return await _handle_issue_comment_event(
                body, x_github_delivery, x_github_event, settings, tracker
            )
        return await _handle_pull_request_event(
            body, x_github_delivery, x_github_event, settings, tracker
        )
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid webhook payload") from exc
    except httpx.HTTPError as exc:
        if tracker.delivery_id is not None and tracker.token is not None:
            _release_delivery_after_failure(
                tracker.delivery_id, tracker.token, "GitHub is unavailable"
            )

        raise HTTPException(
            status_code=502,
            detail="GitHub is unavailable; retry the webhook delivery.",
            headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
        ) from exc
    except ValueError as exc:
        if tracker.delivery_id is not None and tracker.token is not None:
            _release_delivery_after_failure(tracker.delivery_id, tracker.token, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
        ) from exc
    except Exception as exc:
        if tracker.delivery_id is not None and tracker.token is not None:
            _release_delivery_after_failure(tracker.delivery_id, tracker.token, type(exc).__name__)
        logger.exception("Webhook processing failed", exc_info=exc)
        raise HTTPException(
            status_code=503,
            detail="Webhook processing failed; retry the delivery.",
            headers={"Retry-After": str(WEBHOOK_RETRY_AFTER_SECONDS)},
        ) from exc


@router.get("/webhook/health")
async def webhook_health() -> dict[str, str]:
    """Health check endpoint for webhook server."""
    return {"status": "healthy", "service": "kojutsu-webhook"}


def _sqlite_writable(path: str) -> bool:
    db_path = Path(path).expanduser()
    try:
        if not db_path.parent.exists():
            return False
        if not os.access(db_path.parent, os.W_OK):
            return False
        if not db_path.exists():
            return True
        db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=rw", uri=True, timeout=1)
        try:
            db.execute("PRAGMA query_only = ON")
            db.execute("PRAGMA schema_version").fetchone()
        finally:
            db.close()
        return os.access(db_path, os.W_OK)
    except (OSError, sqlite3.Error, ValueError):
        return False


def _webhook_ready_sync(settings: Settings) -> WebhookResult:
    if not settings.github_token:
        raise HTTPException(status_code=503, detail="GITHUB_TOKEN is not configured")
    if not settings.github_webhook_secret:
        raise HTTPException(status_code=503, detail="GITHUB_WEBHOOK_SECRET is not configured")
    _require_repository_allowlist(settings)
    if not settings.tanseki_enabled:
        raise HTTPException(status_code=503, detail="Tanseki is not configured")
    if not all(
        (
            _sqlite_writable(settings.kojutsu_registry_path),
            _sqlite_writable(settings.tanseki_outbox_path),
        )
    ):
        raise HTTPException(status_code=503, detail="Local SQLite storage is not writable")
    try:
        runtime = get_runtime()
        runtime_status = runtime.status()
    except OutboxOwnershipError as exc:
        raise HTTPException(status_code=503, detail="Outbox is owned by another process") from exc
    except Exception as exc:
        logger.warning("Webhook runtime readiness check failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Webhook runtime is unavailable") from exc
    if not runtime_status.get("tanseki_reachable", False):
        raise HTTPException(status_code=503, detail="Tanseki is unavailable")
    return {
        "status": "ready",
        "service": "kojutsu-webhook",
        "tanseki_reachable": True,
        "sqlite_writable": True,
        "runtime_ready": True,
        "outbox_owned": True,
    }


@router.get("/webhook/ready")
async def webhook_ready() -> WebhookResult:
    """Check whether the webhook service can persist captures."""
    return await asyncio.to_thread(_webhook_ready_sync, get_settings())


def _tanseki_reachable(settings: Settings) -> bool:
    with TansekiClient.from_settings(settings) as client:
        return client.health()


def _require_status_auth(secret: str, authorization: str | None) -> None:
    if not secret:
        raise HTTPException(status_code=503, detail="Status authentication is not configured")
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.casefold() != "bearer" or not secrets.compare_digest(token, secret):
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _webhook_status_sync(settings: Settings, webhook_url: str) -> dict[str, Any]:
    allowed_repositories = _require_repository_allowlist(settings)
    try:
        counts = get_runtime().outbox.status_counts()
    except Exception:
        with TansekiOutbox(settings.tanseki_outbox_path) as outbox:
            counts = outbox.status_counts()
    reachable: bool | None = None
    if settings.tanseki_enabled:
        with TansekiClient.from_settings(settings) as client:
            reachable = client.health()
    return {
        "status": "configured"
        if settings.github_token and settings.github_webhook_secret and allowed_repositories
        else "incomplete",
        "service": "kojutsu-webhook",
        "webhook_url": redact_webhook_url(webhook_url),
        "configured": bool(
            settings.github_token and settings.github_webhook_secret and allowed_repositories
        ),
        "allowed_repository_count": len(allowed_repositories),
        "tanseki_reachable": reachable,
        "outbox_pending": counts["pending"] + counts["retrying"],
        "outbox_retrying": counts["retrying"],
        "outbox_dead_letter": counts["dead_letter"],
    }


@router.get("/webhook/status")
async def webhook_status(
    request: Request,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict[str, Any]:
    """Get protected webhook status information without blocking the event loop."""
    settings = get_settings()
    _require_status_auth(settings.github_webhook_secret, authorization)
    return await asyncio.to_thread(_webhook_status_sync, settings, _configured_webhook_url(request))
