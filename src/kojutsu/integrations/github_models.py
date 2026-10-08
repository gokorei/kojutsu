"""Pydantic models for GitHub API entities and webhook payloads."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GitHubUser(BaseModel):
    """GitHub user information.

    ``type`` is parsed rather than discarded because it is the platform's own answer
    to a question capture has to ask constantly: is this account a person or an
    application? GitHub sends it on every user object in every payload it has ever
    sent, and ``extra="ignore"`` was throwing it away, leaving
    :func:`kojutsu.core.answer_collector.is_machine_account` to answer from the login
    suffix alone while its docstring claimed to be reporting what GitHub says.

    That agreement is not a source, and it is what makes the gap worth closing: the
    two signals happened to match on all eight accounts checked, so the fallback was
    not demonstrably wrong, only not a claim. Optional rather than defaulted to
    ``"User"``, because a payload that carries no ``type`` is the forge declining to
    say and defaulting would record that silence as a person — the one thing a
    reader holding a provenance flag must not be misled about.
    """

    model_config = ConfigDict(extra="ignore")

    login: str
    #: ``"Bot"`` for an application, ``"User"`` for a person, ``"Organization"`` for
    #: an org. Compared case-folded and treated as authoritative when present, so
    #: see :func:`kojutsu.core.answer_collector.is_machine_account` before relying on
    #: it directly.
    type: str | None = None


class GitHubComment(BaseModel):
    """GitHub issue or review comment."""

    model_config = ConfigDict(extra="ignore")

    id: int
    body: str
    user: GitHubUser
    created_at: datetime
    author_association: str | None = None


class GitHubIssue(BaseModel):
    """GitHub issue metadata."""

    model_config = ConfigDict(extra="ignore")

    number: int


class GitHubRepository(BaseModel):
    """GitHub repository metadata."""

    model_config = ConfigDict(extra="ignore")

    full_name: str


class GitHubPullRequest(BaseModel):
    """GitHub pull request metadata.

    Every field here is populated straight from the payload field of the same
    name. Nothing is synthesised: a pull request is a record of somebody else's
    work, and a timestamp this model invented would be indistinguishable from one
    GitHub reported, which is the specific confusion a captured record exists to
    avoid.
    """

    model_config = ConfigDict(extra="ignore")

    number: int
    title: str = ""
    body: str | None = ""
    state: str = ""
    user: GitHubUser | None = None
    #: When the change was opened. Not the same fact as ``updated_at``, which
    #: moves on every push and comment, and not derivable from ``closed_at``,
    #: which is absent while the change is open.
    #:
    #: GitHub sends this on both the webhook payload and every search hit, and
    #: this model used to drop it, so a range query had nothing to filter on but
    #: the index's own answer. Optional rather than required because a hit that
    #: states no creation date cannot be shown to be inside a date range, and the
    #: reader that needs it has to be able to say so instead of raising.
    created_at: datetime | None = None
    closed_at: datetime | None = None
    merged_at: datetime | None = None
    updated_at: datetime | None = None
    head: dict | None = None
    base: dict | None = None


class IssueCommentPayload(BaseModel):
    """Payload for 'issue_comment' event."""

    model_config = ConfigDict(extra="ignore")

    action: str
    issue: GitHubIssue
    comment: GitHubComment
    repository: GitHubRepository


class PullRequestPayload(BaseModel):
    """Payload for 'pull_request' event."""

    model_config = ConfigDict(extra="ignore")

    action: str
    pull_request: GitHubPullRequest
    repository: GitHubRepository


class PullRequestReview(BaseModel):
    """A submitted review: a verdict plus the body explaining it."""

    model_config = ConfigDict(extra="ignore")

    id: int
    #: ``approved``, ``changes_requested``, ``commented`` or ``dismissed``. Kept as
    #: a plain string rather than an enum because GitHub may add values, and an
    #: unknown verdict must still be captured and shown, not dropped.
    state: str
    body: str | None = ""
    user: GitHubUser
    submitted_at: datetime | None = None
    author_association: str | None = None
    html_url: str | None = None


class PullRequestReviewComment(GitHubComment):
    """An inline comment anchored to a position in the diff.

    The anchor is the point of the record. An inline comment that does not say
    which line of which file it referred to is close to useless months later,
    because the line it discussed has usually moved on.
    """

    model_config = ConfigDict(extra="ignore")

    #: Repository-relative path of the file the comment is anchored to.
    path: str | None = None
    #: Line in the diff hunk the comment refers to, as GitHub reports it.
    line: int | None = None
    original_line: int | None = None
    #: Side of the diff: ``LEFT`` for the pre-image, ``RIGHT`` for the post-image.
    side: str | None = None
    #: ``diff_hunk`` is the surrounding patch text. It is the code under
    #: discussion, which makes it untrusted input like any other PR text.
    diff_hunk: str | None = None
    commit_id: str | None = None
    in_reply_to_id: int | None = None
    pull_request_review_id: int | None = None


class PullRequestReviewPayload(BaseModel):
    """Payload for 'pull_request_review' event.

    Covers both the review itself and the inline comments submitted with it, so
    the two cannot drift apart: GitHub delivers a review and its comments in one
    payload, and recording only the verdict would discard the text that explains
    it.
    """

    model_config = ConfigDict(extra="ignore")

    action: str
    review: PullRequestReview
    pull_request: GitHubPullRequest
    repository: GitHubRepository
    comments: list[PullRequestReviewComment] = []


class GitHubCheckRun(BaseModel):
    """One run of one check against one commit.

    Modelled rather than read as a raw dict because a check report is a machine
    report about a commit, and the only reason it reaches the store is so a
    reader can tell *that a check concluded this* — never *what that says about
    the change*. Keeping the conclusion as an opaque string the forge supplied is
    what stops this from becoming a judgement.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    #: The conclusion the check reported: ``success``, ``failure``,
    #: ``neutral``, ``cancelled``, ``timed_out``, ``action_required``,
    #: ``skipped``, ``stale``, or ``startup_failure``. Recorded verbatim.
    conclusion: str | None = None
    status: str = ""
    name: str = ""
    #: The commit the check ran against, which is what makes the report usable
    #: alongside ``head_sha`` on a review record.
    head_sha: str | None = None
    #: The pull request the check belongs to, when the check is on one. Absent for
    #: a check run on a branch, and a check with no pull request is ordinary.
    pull_requests: list[dict] = Field(default_factory=list)

    @property
    def pr_number(self) -> int | None:
        """The pull request number, or ``None`` when the check is not on one."""
        for entry in self.pull_requests:
            if not isinstance(entry, dict):
                continue
            number = entry.get("number")
            if isinstance(number, int) and not isinstance(number, bool) and number > 0:
                return number
        return None


class CheckRunPayload(BaseModel):
    """Payload for a ``check_run`` event."""

    model_config = ConfigDict(extra="ignore")

    action: str
    repository: GitHubRepository
    check_run: GitHubCheckRun
