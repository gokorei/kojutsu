"""Reading repository history for a bounded backfill.

Three of the four reads a backfill needs are not on
:class:`~kojutsu.integrations.github.GitHubClient`: listing pull requests by
update time, listing a pull request's reviews, and listing the comments belonging
to one review. They are here rather than added there because
``docs/github-seam.md`` names ``integrations/github.py`` as the only module that
reaches the REST API, and its call table does not list these endpoints. The seam
needs the module and the document extended together; until then this is the second
place that opens a socket, and that is worth saying out loud rather than leaving a
reader to find.

**No diffs, no repository contents.** Review history needs ``Pull requests: Read``
and ``Issues: Read``, which is the documented minimum for this pipeline, and the
backfill stops there. ``docs/github-seam.md`` deliberately withholds ``Contents``,
and a reconstruction is the worst possible place to want it: a diff read at
reconstruction time is a diff of a *later* revision, so anchoring a historical
record to it would assert a correspondence that did not hold at the time. If a
caller finds itself wanting one, that is the signal to stop.

**A rate limit is honoured, never absorbed.** A 429 pauses the run for the interval
the forge asked for and resumes the same read. It is never caught and turned into a
skipped object, because a skip is indistinguishable from "there was nothing there"
and produces a corpus with a hole in it and no record that the hole exists. When
the bounded number of retries is exhausted the run fails loudly instead: the
re-run is safe precisely because identity, not a cursor, makes the work idempotent.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx

from kojutsu.integrations.github import (
    REQUEST_TIMEOUT_SECONDS,
    GitHubClient,
    GitHubPayloadError,
    github_backoff_delay,
)
from kojutsu.integrations.github_models import (
    GitHubComment,
    GitHubPullRequest,
    PullRequestReview,
    PullRequestReviewComment,
)

#: GitHub's own maximum. Larger pages are silently truncated to this, so asking
#: for more would mean issuing requests whose results are thrown away.
PAGE_SIZE = 100

#: A ceiling on pages walked for one object, matching what the rest of the client
#: already applies. History is unbounded in principle; a single pull request with
#: ten thousand comments is walked to its end and then reported as truncated
#: rather than read forever.
MAX_PAGES_PER_OBJECT = 100

DEFAULT_RATE_LIMIT_RETRIES = 5
MAX_RATE_LIMIT_BACKOFF_SECONDS = 60.0


class RateLimitedError(RuntimeError):
    """The forge rate-limited the run and the bounded backoff did not clear it.

    Deliberately fatal. The alternative -- treating the exhausted read as a
    skipped object -- is the failure this module exists to prevent, because the
    resulting corpus would carry a hole with nothing in it recording the hole.
    """


class HistoryReader(Protocol):
    """The four reads a backfill performs, and nothing else.

    A protocol rather than a concrete client so the command can be driven by a fake
    with no network, and so the set of reads is a thing a test can hold the
    implementation to: adding a read is a visible change to this interface rather
    than a new call site appearing inside the run loop.
    """

    def list_pull_requests(
        self, owner: str, repo: str, *, page: int, per_page: int
    ) -> list[GitHubPullRequest]: ...

    def list_reviews(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[PullRequestReview]: ...

    def list_review_comments(
        self, owner: str, repo: str, pr_number: int, review_id: int, *, page: int, per_page: int
    ) -> list[PullRequestReviewComment]: ...

    def list_issue_comments(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[GitHubComment]: ...

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest | None:
        """One pull request by number, or ``None`` when the forge has no such PR.

        The single-PR bound reads through here instead of the listing, so one
        PR costs one request rather than one page per hundred PRs updated since
        the window began. ``None`` (rather than an exception) is the not-found
        answer so the walk reports it as a gap, the same way it reports a pull
        request the listing named but the forge then could not serve.
        """
        ...


def is_missing_object(exc: BaseException) -> bool:
    """True when the read failed because the object is gone from the forge.

    A review deleted after the fact reads as a 404 on its comments, and that is
    the one read failure a backfill is expected to survive: nothing is written, and
    the gap is reported. Every other failure is fatal, because anything else is a
    question about this run rather than about history.
    """
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404


class GitHubHistoryReader:
    """The four history reads, over the same token and scope rules as capture.

    **One connection pool for the reader's whole life**, for the reason
    :class:`~kojutsu.integrations.github.GitHubClient` gives and not restated: a
    ``backfill-reviews`` run over ``pingdotgg/t3code`` issued 96 requests through this
    class, and every one of them used to pay a DNS lookup, a TCP handshake and a TLS
    negotiation before transferring a byte. Of those 96, 88 were the pull request
    listing and 114.2 of the run's 118.7 seconds were spent inside them, so the
    handshakes were a share of the largest single cost in the command.

    It is this class rather than the capture client that needed it, because this is the
    one the walk actually goes through -- ``HistoryReader`` is what
    :func:`~kojutsu.core.backfill_reviews.run_backfill` is handed, and
    :class:`~kojutsu.integrations.github.GitHubClient` is not on that path at all.

    Built lazily for the same reason and with the same thread-safety argument: the walk
    reads pages from a bounded thread pool, and ``httpx.Client`` is documented safe to
    use from several threads, so the first construction is guarded by a lock and
    nothing after it needs to be.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        sleep: Callable[[float], None] = time.sleep,
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        max_backoff_seconds: float = MAX_RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._sleep = sleep
        self._rate_limit_retries = max(rate_limit_retries, 0)
        self._max_backoff_seconds = max_backoff_seconds
        self._scope_checked = False
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    def http(self) -> httpx.Client:
        """The one pool this reader reads through, built on first use."""
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
            return self._client

    def close(self) -> None:
        """Close the connection pool. Idempotent, and safe if none was ever opened."""
        client = self._client
        if client is not None:
            client.close()

    def __enter__(self) -> GitHubHistoryReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str | int]) -> Sequence[Any]:
        """One paged GET, backing off on a 429 and never absorbing one.

        The scope check is applied to this module's own responses rather than left
        to :meth:`GitHubClient._verified`, which only sees the calls that go
        through that client: a run that reads reviews and never lists issue
        comments would otherwise never be scope-checked at all.

        The pool is *outside* the retry loop, deliberately. A retry after a 429 is the
        same read the run already decided to make, so it should reuse the connection
        rather than re-handshake -- and a run that had been re-handshaking on every
        retry was spending its backoff sleeping and then paying a fresh TLS
        negotiation on top.
        """
        attempt = 0
        client = self.http()
        while True:
            attempt += 1
            response = client.get(
                f"{self._base}/{path.lstrip('/')}", params=params, headers=self._headers
            )
            if response.status_code == 429:
                if attempt > self._rate_limit_retries:
                    raise RateLimitedError(
                        f"GitHub rate-limited {path} and {self._rate_limit_retries} backoff(s) "
                        "did not clear it; re-run the same range to continue"
                    ) from None
                self._sleep(self._backoff(response, attempt))
                continue
            response.raise_for_status()
            if not self._scope_checked:
                GitHubClient.assert_minimum_scope(response.headers.get("x-oauth-scopes"))
                self._scope_checked = True
            payload = response.json()
            if not isinstance(payload, list):
                raise GitHubPayloadError(f"GitHub returned an invalid list for {path}")
            return payload

    def _backoff(self, response: httpx.Response, attempt: int) -> float:
        """How long to wait before re-issuing this read; the shared policy.

        Delegates to :func:`kojutsu.integrations.github.github_backoff_delay`
        rather than keeping a second definition of "how long": the live capture
        path and the history path wait out the same forge, so they share one
        backoff. The retry loop around it stays here, because its contract --
        fail loudly with :class:`RateLimitedError` rather than return a short
        read -- belongs to the run, not to the seam.
        """
        return github_backoff_delay(response, attempt, max_backoff=self._max_backoff_seconds)

    def list_pull_requests(
        self, owner: str, repo: str, *, page: int, per_page: int
    ) -> list[GitHubPullRequest]:
        """One page of pull requests, most recently updated first.

        Descending by update time is what makes the date floor cheap and correct at
        the same time: walking forward the pages get older, so the first entry whose
        update predates the floor proves every later one does too, and the walk stops
        there.

        **Descending is load-bearing, and ascending reconstructed nothing at all.**
        Ascending starts at the repository's *oldest* pull request, whose update
        predates any floor a person would sensibly choose, so the walk stopped on its
        very first entry and reported zero objects read -- a summary line
        indistinguishable from a repository that genuinely had nothing to say. It
        cannot be worked around by passing an earlier floor either, because that
        admits a different era rather than the intended one.

        It is descending rather than a ``created`` ordering that admits a pull
        request opened long before the floor and reviewed after it, which is
        precisely the one a backfill exists to find.
        """
        payload = self._get(
            f"repos/{owner}/{repo}/pulls",
            {
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "page": page,
                "per_page": per_page,
            },
        )
        return [
            GitHubPullRequest.model_validate(entry) for entry in payload if isinstance(entry, dict)
        ]

    def list_reviews(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[PullRequestReview]:
        payload = self._get(
            f"repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            {"page": page, "per_page": per_page},
        )
        return [
            PullRequestReview.model_validate(entry) for entry in payload if isinstance(entry, dict)
        ]

    def list_review_comments(
        self, owner: str, repo: str, pr_number: int, review_id: int, *, page: int, per_page: int
    ) -> list[PullRequestReviewComment]:
        """The inline comments of one review.

        Scoped to the review rather than to the pull request, because that is the
        read whose failure carries meaning: a review deleted after the fact has no
        comments left to list, and the 404 is how the backfill learns the object it
        was about to reconstruct no longer exists. Reading them first, before the
        review is handed to a collector, is what keeps that from producing a
        half-record of something the forge has withdrawn.
        """
        payload = self._get(
            f"repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/comments",
            {"page": page, "per_page": per_page},
        )
        return [
            PullRequestReviewComment.model_validate(entry)
            for entry in payload
            if isinstance(entry, dict)
        ]

    def list_issue_comments(
        self, owner: str, repo: str, pr_number: int, *, page: int, per_page: int
    ) -> list[GitHubComment]:
        """Pull request comments, which the forge serves from the *issues* API.

        This is why the minimum token scope includes ``Issues: Read`` even though
        the backfill is only ever thinking about pull requests.
        """
        payload = self._get(
            f"repos/{owner}/{repo}/issues/{pr_number}/comments",
            {"page": page, "per_page": per_page},
        )
        return [GitHubComment.model_validate(entry) for entry in payload if isinstance(entry, dict)]

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest | None:
        """Fetch one pull request by number, or ``None`` when it is not there.

        A 404 is the forge's answer, not a failure of the run: the walk reports
        it as a gap and moves on, which is what makes ``--pr`` naming a
        deleted or mistyped number an empty range rather than an error.
        """
        response = self.http().get(
            f"{self._base}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers=self._headers,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if not self._scope_checked:
            GitHubClient.assert_minimum_scope(response.headers.get("x-oauth-scopes"))
            self._scope_checked = True
        payload = response.json()
        if not isinstance(payload, dict):
            raise GitHubPayloadError(f"GitHub returned an invalid pull request for {pr_number}")
        return GitHubPullRequest.model_validate(payload)
