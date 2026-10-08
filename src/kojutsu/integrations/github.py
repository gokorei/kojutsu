"""GitHub REST API client for PR diffs, comments, and branch parsing."""

import logging
import os
import random
import re
import threading
import time
import unicodedata
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit

import httpx

from kojutsu.allowlist import repository_allowed
from kojutsu.config import Settings
from kojutsu.integrations.github_models import (
    GitHubComment,
    GitHubPullRequest,
    PullRequestReview,
    PullRequestReviewComment,
)
from kojutsu.models import RationaleSource
from kojutsu.text_limits import (
    INVISIBLE_FORMATTING_CHARACTERS,
    MAX_AGENT_ID_CHARS,
    MAX_BRANCH_CHARS,
    MAX_MODEL_ID_CHARS,
)

logger = logging.getLogger(__name__)

KOJUTSU_MARKER_PREFIX = "<!-- kojutsu:question:"
KOJUTSU_MARKER_SUFFIX = " -->"
KOJUTSU_ANSWER_PREFIX = "<!-- kojutsu:answer:"
KOJUTSU_ANSWER_SUFFIX = " -->"
KOJUTSU_AGENT_PREFIX = "<!-- kojutsu:agent:"
KOJUTSU_AGENT_SUFFIX = " -->"
#: Marker for a stated decision rationale, in the same dialect as the agent
#: marker above. Reusing the vocabulary rather than inventing a second one is the
#: point: a reader of a comment body should not have to learn a new grammar to
#: tell a machine principal's declaration from its assertion of authorship.
KOJUTSU_RATIONALE_PREFIX = "<!-- kojutsu:rationale:"
KOJUTSU_RATIONALE_SUFFIX = " -->"
#: Key used for the declaration's revision position in the rationale marker.
REVISION_MARKER_KEY = "rev"
#: Key for how the machine obtained its reasoning: stated from its own session, or
#: inferred from a diff. The same class of self-assertion as the model half.
SOURCE_MARKER_KEY = "source"
#: Key for the change a declaration is about, so a collector reading the comment
#: can derive the same identity the writer did.
BRANCH_MARKER_KEY = "branch"
# Key used for the optional model half of an agent claim.
MODEL_MARKER_KEY = "model"

# Classic personal access token scopes that grant materially more than reading pull
# requests and posting comments on one repository. ``repo`` and ``public_repo`` cover
# every repository on the account, read and write, including file contents;
# ``admin:*`` scopes are administrative. A fine-grained token cannot produce any of
# these, which is why this check only ever fires for classic tokens.
_OVER_SCOPED_CLASSIC_SCOPES = frozenset(
    {
        "repo",
        "public_repo",
        "admin:org",
        "admin:repo_hook",
        "admin:org_hook",
        "admin:public_key",
        "manage_runners:org",
        "manage_runners:enterprise",
        "write:org",
        "write:packages",
        "delete_repo",
        "write:discussion",
        "gist",
    }
)


class GitHubIntegrationError(RuntimeError):
    """Base error for GitHub integration failures."""

    #: Whether issuing the identical request again could plausibly succeed. A
    #: configuration refusal and a malformed payload cannot be retried; an
    #: exhausted rate limit can, once the window rolls over. Callers that retry
    #: on transport errors need to know which of these they are holding, and
    #: deciding that by exception type at the call site is how a rate limit
    #: ends up silently swallowed.
    retryable = False


class GitHubTokenScopeError(GitHubIntegrationError):
    """GITHUB_TOKEN is scoped far more broadly than the pipeline requires.

    This is a configuration refusal rather than a transport failure, so it is not
    retryable: retrying the same credential cannot help.
    """


class GitHubRateLimitError(GitHubIntegrationError):
    """GitHub refused the request because a rate limit was exhausted.

    Retryable, and marked as such, because the request was correct and the window
    will reopen. The alternative -- letting ``raise_for_status`` produce a bare
    status error, or catching it and continuing -- is how a truncated read starts
    reading as a complete one. A rate limit that returns fewer results than
    reality holds is a silently wrong answer, so it is surfaced as an error rather
    than absorbed into an empty list.
    """

    retryable = True

    def __init__(
        self, message: str, *, resource: str = "", retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        #: Which limit was hit (``search`` or ``core``), when GitHub says.
        self.resource = resource
        #: Seconds GitHub asked us to wait, from ``Retry-After`` or the reset
        #: header, when it says. ``None`` rather than a guess.
        self.retry_after = retry_after


class GitHubPayloadError(GitHubIntegrationError):
    """The GitHub response did not have the expected shape."""


class GitHubRangeError(GitHubIntegrationError):
    """The requested pull request date range is not usable as written.

    A caller mistake, refused rather than answered. An inverted range returns
    zero results from an index that is working perfectly, and a caller reading
    "no pull requests" would conclude the range was empty rather than that the
    arguments were backwards. Every message names the field at fault, because
    "invalid range" sends the reader back to the call site to guess which half.
    """


class GitHubRepositoryNotAllowedError(GitHubIntegrationError):
    """The repository is outside the configured allowlist.

    Refused before any request is issued, for the same reason the webhook and the
    MCP server refuse one: authorisation that happens after the request has been
    sent is authorisation that has already leaked that the caller asked.
    """


def normalise_captured_text(value: str) -> str:
    """Return text a contributor wrote, in the one form Kojutsu keeps.

    Two problems, two independent reasons, and the same fix.

    **Identity and equality.** Third-party text arrives decomposed. ``é`` may reach
    this seam as one code point or as ``e`` plus a combining acute, which are the
    same characters to a reader and to the forge and different byte strings to
    everything downstream -- so the same sentence is stored twice, matches twice and
    counts twice, with no way to tell a reader that it is one sentence.

    **Display fidelity.** A bidirectional override or a zero-width space makes stored
    evidence render as something it is not. A record whose author reads as somebody
    else, or whose file path reads as a different file, is the failure this whole
    system exists to prevent, and it is invisible to exactly the reader it deceives.

    **The characters are removed and the record is kept.** Refusing the record is the
    obvious reading of "reject" and it is the wrong one: a hostile comment is
    evidence *of* a hostile comment, and dropping it destroys precisely what this
    project exists to preserve. The anchor survives -- the forge-issued comment,
    review or check id is stored alongside, so a reader can re-fetch the original
    with everything the author put there. The alternative of keeping the characters
    and marking the record instead was rejected because nothing in the record's
    schema can carry that mark: there is no field for it, and adding one would mean
    a frontmatter key whose only writer would be this function.

    Applied **once**, at the boundary, and that placement is the point. Every capture
    path reads text through either this function or one of the marker extractors
    below; normalising inside the derivations instead would mean the identity, the
    stored bytes and the reader's view of them were computed from three different
    values, which is a way of storing text this system cannot reproduce from its own
    input. One implementation, called where text enters, and nothing else to keep in
    step.

    **Marker grammar is read verbatim; a marker's value is normalised.** The
    ``kojutsu:`` grammar is what makes a comment a Kojutsu record, and a token
    whose own punctuation had to be repaired to parse is not one this system wrote --
    that fails closed. The value inside it is data: it names the question, and it
    becomes part of a document path, so an invisible character left in it is the
    spoofing this whole function exists to prevent. NFC is applied before the refused
    set is checked so the check runs against the canonical form a reader sees rather
    than against a decomposition that may be hiding one.
    """
    return "".join(
        character
        for character in unicodedata.normalize("NFC", value)
        if character not in INVISIBLE_FORMATTING_CHARACTERS
    )


def extract_jira_key_from_branch(branch_name: str) -> str | None:
    """
    Extract Jira ticket key from branch name.
    Expects format: fix|chore|feature|experiment/<JIRA-KEY>/optional-description
    """
    match = re.match(
        r"^(?:fix|chore|feature|experiment)[/\-]([A-Z][A-Z0-9]+-\d+)", branch_name, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    return None


def extract_jira_key_from_text(text: str) -> str | None:
    """Extract Jira ticket key from a block of text (e.g. PR description)."""
    if not text:
        return None
    # Matches common Jira patterns like ABC-123
    match = re.search(r"\b([A-Z][A-Z0-9]+-\d+)\b", text)
    if match:
        return match.group(1).upper()
    return None


def canonical_pr_url(repo: str, pr_number: int) -> str:
    """Return the canonical GitHub pull request URL."""
    return f"https://github.com/{repo}/pull/{pr_number}"


def parse_pr_identifier(pr_spec: str) -> tuple[str, int] | None:
    """
    Parse an exact PR URL or 'owner/repo#123' into (owner/repo, pr_number).
    Returns None if not parseable.
    """
    value = pr_spec.strip()
    m = re.fullmatch(r"([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#(\d+)", value)
    if m:
        return m.group(1), int(m.group(2))

    if "://" not in value or "?" in value or "#" in value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() not in {
        "github.com",
        "www.github.com",
    }:
        return None
    m = re.fullmatch(r"/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)/pull/(\d+)", parsed.path)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}", int(m.group(3))


def kojutsu_comment_body(question_id: str, question_text: str) -> str:
    """Build comment body with hidden marker for later matching."""
    marker = f"{KOJUTSU_MARKER_PREFIX}{question_id}{KOJUTSU_MARKER_SUFFIX}"
    return f"{marker}\n\n{question_text}"


def extract_question_id_from_comment_body(body: str) -> str | None:
    """Extract kojutsu question ID from a comment body, or None.

    The returned id is normalised: it is a value that becomes part of a document path
    and a registry lookup key, so an invisible character left in it is a question id
    that renders as another question. See :func:`normalise_captured_text`.
    """
    if KOJUTSU_MARKER_PREFIX not in body:
        return None
    start = body.find(KOJUTSU_MARKER_PREFIX) + len(KOJUTSU_MARKER_PREFIX)
    end = body.find(KOJUTSU_MARKER_SUFFIX, start)
    if end == -1:
        return None
    return normalise_captured_text(body[start:end].strip())


def extract_question_text_from_comment_body(body: str) -> str | None:
    """Extract the visible question text associated with a Kojutsu marker.

    The marker settles *whether* this is a Kojutsu question and is parsed without
    being repaired; the prose after it is somebody's own words and is normalised. See
    :func:`normalise_captured_text` for where that line falls.
    """
    question_id = extract_question_id_from_comment_body(body)
    if not question_id:
        return None
    marker_end = body.find(KOJUTSU_MARKER_SUFFIX)
    if marker_end == -1:
        return None
    return normalise_captured_text(body[marker_end + len(KOJUTSU_MARKER_SUFFIX) :].strip())


def answer_comment_body(question_id: str, answer_text: str) -> str:
    """Build an answer body with an explicit question association."""
    marker = f"{KOJUTSU_ANSWER_PREFIX}{question_id}{KOJUTSU_ANSWER_SUFFIX}"
    return f"{marker}\n\n{answer_text}"


def agent_comment_body(
    agent_id: str, model: str | None = None, source: RationaleSource | None = None
) -> str:
    """Build the marker that attributes a comment to a machine principal.

    The model half is what lets capture tell a second opinion apart from the same
    mind restating itself, so it is included whenever the caller knows it. Without
    it the record is captured honestly as a principal that did not say which model
    it was, which is a weaker claim rather than a wrong one.

    The source half records *how the reasoning was obtained* — stated from the
    agent's own session, or inferred from a diff. It rides in the same marker
    because it is the same kind of claim: an assertion by the author about its own
    output, verified by nothing. An answerer that reads a diff and states why the
    change looks the way it does is ``RECONSTRUCTED``, and a reader who cannot
    tell that from a declaration is reading a reconstruction as a recollection.

    Omitted rather than defaulted, so a comment written before this half existed
    carries no claim at all and is reported as unstated rather than guessed.
    """
    parts = [agent_id]
    if model:
        parts.append(f"{MODEL_MARKER_KEY}={model}")
    if source is not None:
        parts.append(f"{SOURCE_MARKER_KEY}={source.value}")
    return f"{KOJUTSU_AGENT_PREFIX}{' '.join(parts)}{KOJUTSU_AGENT_SUFFIX}"


def answer_comment_body_as_agent(
    question_id: str,
    answer_text: str,
    agent_id: str,
    model: str | None = None,
    source: RationaleSource | None = None,
) -> str:
    """Build a complete machine-authored answer: association, attribution, prose.

    The agent marker comes first so a reader skimming the comment sees who wrote it
    before reading what it says, and so the attribution is never separated from the
    text it describes by a length limit or a truncation.
    """
    return (
        f"{answer_comment_body(question_id, '')}\n"
        f"{agent_comment_body(agent_id, model, source)}\n\n"
        f"{answer_text.strip()}"
    )


@dataclass(frozen=True)
class AgentClaim:
    """A machine principal's self-declared identity, read from a comment body.

    All three halves are *assertions by the comment author*, not facts the platform
    verifies. GitHub proves who posted a comment; it can never prove which model
    drafted it, nor that a stated reason is the reason the agent actually had. What
    the platform does verify is that the posting account holds a trusted
    association, and the collector gates on that before this is read. So the claim
    is trustworthy exactly as far as the account making it.

    ``source`` defaults to ``UNKNOWN`` rather than to either real value. A comment
    posted before the key existed states nothing about how its reasoning was
    obtained, and defaulting would be the reader guessing at provenance the writer
    never claimed.
    """

    agent_id: str
    model: str | None = None
    source: RationaleSource = RationaleSource.UNKNOWN


def extract_agent_claim(body: str) -> AgentClaim | None:
    """Return the declared agent identity, with its optional model, or None.

    An agent that writes into a review thread declares which agent it is and,
    where it can, which model it is:

        <!-- kojutsu:agent:opencode -->
        <!-- kojutsu:agent:opencode model=opencode/model -->

    The model half is what makes a machine-authored answer weighable. Without it,
    a bot answering on two different models is indistinguishable from one bot
    answering on the same model, and neither can be told apart after the fact.

    The older marker without a model still parses, with the model recorded as
    unknown, so comments posted before this existed keep working and are honestly
    reported as not stating a model rather than silently treated as one.

    The declared agent id and model are normalised. Neither is hashed into an identity,
    so this cannot move a stored record; both are stored as the author of a record, and
    an invisible character in either is a machine principal that renders as a different
    one. The rationale marker is the exception, and the comment there says why.
    """
    if KOJUTSU_AGENT_PREFIX not in body:
        return None
    start = body.find(KOJUTSU_AGENT_PREFIX) + len(KOJUTSU_AGENT_PREFIX)
    end = body.find(KOJUTSU_AGENT_SUFFIX, start)
    if end == -1:
        return None
    tokens = body[start:end].split()
    if not tokens:
        return None

    agent_id = normalise_captured_text(tokens[0])
    if agent_id.startswith(MODEL_MARKER_KEY) or len(agent_id) > MAX_AGENT_ID_CHARS:
        # A leading `model=` means no agent was named, which is not a claim.
        return None

    model: str | None = None
    source = RationaleSource.UNKNOWN
    # Every token is read rather than stopping at the first match. An earlier
    # version stopped after ``model=``, which was correct while the model was the
    # only optional half and silently dropped everything after it once a third
    # key existed -- a claim written by the tool and quietly lost on the way back
    # in, which is the same silent-drop failure the frontmatter whitelist had.
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if key == MODEL_MARKER_KEY:
            candidate = normalise_captured_text(value.strip())
            if candidate and len(candidate) <= MAX_MODEL_ID_CHARS:
                model = candidate
        elif key == SOURCE_MARKER_KEY:
            try:
                source = RationaleSource(value.strip().casefold())
            except ValueError:
                # An unrecognised source is reported as unknown rather than guessed
                # at, for the same reason an unstated model is: the reader must not
                # invent provenance the writer did not claim.
                source = RationaleSource.UNKNOWN
    return AgentClaim(agent_id=agent_id, model=model, source=source)


def extract_agent_id_from_comment_body(body: str) -> str | None:
    """Return just the agent id from a comment body, or None if there is no claim."""
    claim = extract_agent_claim(body)
    return claim.agent_id if claim is not None else None


@dataclass(frozen=True)
class RationaleClaim:
    """A declaration of decision rationale, read from a comment body.

    Carries the same caveat as :class:`AgentClaim`, and it is worth restating here
    because a rationale is the case where the caveat is easiest to forget. The
    platform proves *who posted a comment* and nothing more. It cannot prove which
    model drafted the text, and it certainly cannot verify that the reason given is
    the reason the agent actually had. A declaration is an assertion by its author
    about its own reasoning, so it is trusted exactly as far as the account making
    it — and no further.

    ``revision`` is the declaration's position in the sequence for one change, not
    a version of the marker syntax. It exists so a second declaration is appended
    rather than overwriting the first.

    ``branch`` is present for one reason above all: **identity.** The capture tool
    claims with an anchor-derived entry id before it posts, so the collector that
    later reads the comment has to derive the same id or one declaration becomes
    two records. The branch is part of that anchor and the comment body is the only
    thing the collector has. It is also the honest thing to carry: a declaration
    that names no change is one nobody can find again.
    """

    agent_id: str
    model: str | None = None
    revision: int = 1
    branch: str = ""


def rationale_comment_body(
    agent_id: str,
    model: str | None = None,
    revision: int = 1,
    branch: str = "",
) -> str:
    """Build the marker that attributes a declared rationale to a machine principal.

    The revision is included because a declaration is appended, never replaced: a
    later rationale that silently overwrote an earlier one would destroy the record
    of what the agent thought when it started, which is usually the more
    interesting half.

    The branch is included so the collector derives the same identity the writer
    did. It is bounded, and it may be omitted — a marker written by hand or by an
    older build simply lacks it, and then derives a different id, which is correct
    because it is not the same fact as a capture-tool declaration.
    """
    if revision < 1:
        raise ValueError(f"rationale revision must be at least 1, got {revision!r}")
    if len(branch) > MAX_BRANCH_CHARS:
        raise ValueError(f"branch must be at most {MAX_BRANCH_CHARS} characters")
    parts = [agent_id]
    if model:
        parts.append(f"{MODEL_MARKER_KEY}={model}")
    if branch:
        parts.append(f"{BRANCH_MARKER_KEY}={branch}")
    parts.append(f"{REVISION_MARKER_KEY}={revision}")
    return f"{KOJUTSU_RATIONALE_PREFIX}{' '.join(parts)}{KOJUTSU_RATIONALE_SUFFIX}"


def rationale_comment_body_as_agent(
    rationale_text: str,
    agent_id: str,
    model: str | None = None,
    revision: int = 1,
    branch: str = "",
) -> str:
    """Build a complete declared rationale: attribution first, then the reason.

    The marker comes first for the same reason it does in
    :func:`answer_comment_body_as_agent`: a reader skimming the comment should
    meet who is claiming this before what it claims, and attribution should not be
    separable from the text it describes by a truncation.
    """
    return (
        f"{rationale_comment_body(agent_id, model, revision, branch)}\n\n{rationale_text.strip()}"
    )


def extract_rationale_claim(body: str) -> RationaleClaim | None:
    """Return the declared rationale identity, or ``None`` if there is no claim.

        <!-- kojutsu:rationale:opencode -->
        <!-- kojutsu:rationale:opencode model=opencode/model rev=2 -->
        <!-- kojutsu:rationale:opencode branch=feat/backoff rev=1 -->

    The model half lets a later reader tell a first-hand declaration apart from one
    made on a different model; the branch is what lets a *collector* derive the
    same identity the writer did, which is what stops one declaration becoming two
    records. Either may be absent: an agent that does not state its model is
    recorded as not stating one, which is a weaker claim rather than a wrong one,
    and a comment posted before either key existed still parses.

    The marker is space-separated, and a git ref cannot contain a space, so a
    branch cannot split into extra tokens. A hand-written marker that tries to will
    simply derive a different id rather than corrupt one, because the derivation is
    length-prefixed and bounded.

    **These values are deliberately not normalised**, unlike the agent marker's and
    unlike the prose. ``agent_id`` and ``branch`` are hashed by
    :func:`~kojutsu.core.question_registry.stable_rationale_entry_id`, and the
    declaration id is derived on the *writing* side too -- by the capture tool, before
    it posts -- so the collector has to derive exactly what the writer did. Normalising
    on this side alone would give the two halves of one declaration different ids when
    either value arrived decomposed or carried an invisible character, which is one
    declaration stored twice: the precise loss the branch field exists to prevent. It
    is the one place the boundary stops short of the value, and the reason is a
    counterpart on the far side of the comment rather than a local judgement.
    """
    if KOJUTSU_RATIONALE_PREFIX not in body:
        return None
    start = body.find(KOJUTSU_RATIONALE_PREFIX) + len(KOJUTSU_RATIONALE_PREFIX)
    end = body.find(KOJUTSU_RATIONALE_SUFFIX, start)
    if end == -1:
        return None
    tokens = body[start:end].split()
    if not tokens:
        return None

    agent_id = tokens[0]
    if agent_id.startswith(MODEL_MARKER_KEY) or len(agent_id) > MAX_AGENT_ID_CHARS:
        # A leading `model=` means no agent was named, which is not a claim.
        return None

    model: str | None = None
    revision = 1
    branch = ""
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if key == MODEL_MARKER_KEY:
            candidate = value.strip()
            if candidate and len(candidate) <= MAX_MODEL_ID_CHARS:
                model = candidate
        elif key == REVISION_MARKER_KEY:
            candidate = value.strip()
            # An unusable revision is reported as unstated rather than guessed at,
            # so a comment with a malformed rev is stored as revision 1 instead of
            # claiming a position in the sequence it never stated.
            if candidate.isdigit() and int(candidate) >= 1:
                revision = int(candidate)
        elif key == BRANCH_MARKER_KEY:
            candidate = value.strip()
            if candidate and len(candidate) <= MAX_BRANCH_CHARS:
                branch = candidate
    return RationaleClaim(agent_id=agent_id, model=model, revision=revision, branch=branch)


def extract_rationale_text_from_comment_body(body: str) -> str:
    """Return declared rationale text without its hidden attribution marker.

    The marker settles *what kind of record this is* and the prose is the reason
    somebody gave; the split is the same one as
    :func:`extract_question_text_from_comment_body`, and both are normalised through
    the one function it names.
    """
    marker_end = body.find(KOJUTSU_RATIONALE_SUFFIX)
    if body.lstrip().startswith(KOJUTSU_RATIONALE_PREFIX) and marker_end != -1:
        return normalise_captured_text(body[marker_end + len(KOJUTSU_RATIONALE_SUFFIX) :].strip())
    return normalise_captured_text(body.strip())


def extract_answer_question_id_from_comment_body(body: str) -> str | None:
    """Extract the question ID from an explicit answer marker.

    Normalised, for the reason given on :func:`extract_question_id_from_comment_body`:
    this value is the association a whole capture rests on, and it is the one a doctored
    comment would most want to point somewhere other than where it reads.
    """
    if KOJUTSU_ANSWER_PREFIX not in body:
        return None
    start = body.find(KOJUTSU_ANSWER_PREFIX) + len(KOJUTSU_ANSWER_PREFIX)
    end = body.find(KOJUTSU_ANSWER_SUFFIX, start)
    if end == -1:
        return None
    return normalise_captured_text(body[start:end].strip())


def extract_answer_text_from_comment_body(body: str) -> str:
    """Return answer text without its hidden association marker.

    Where a comment body becomes text Kojutsu keeps, which is why the
    normalisation happens in the extractors rather than in each writer: a caller that
    wanted different handling would have to re-implement the same character policy, and
    two implementations of it would disagree at some point in a way nothing reports. A
    body that is nothing but invisible characters arrives here as the empty string, and
    every caller already treats that as nothing to capture.
    """
    marker_end = body.find(KOJUTSU_ANSWER_SUFFIX)
    if body.lstrip().startswith(KOJUTSU_ANSWER_PREFIX) and marker_end != -1:
        return normalise_captured_text(body[marker_end + len(KOJUTSU_ANSWER_SUFFIX) :].strip())
    return normalise_captured_text(body.strip())


def new_question_id() -> str:
    """Generate a new unique question ID for kojutsu markers."""
    return str(uuid.uuid4())


#: The search API returns at most this many results, and it says nothing louder
#: than a 400 about it: a query matching more returns the first 1000 and a total
#: count nobody is obliged to read. A reader that ignores the total gets a short
#: list that looks exactly like a complete one, so the cap is enforced here and
#: the shortfall is reported rather than absorbed.
SEARCH_RESULT_CAP = 1000

#: Authenticated search is **30 requests per minute**. The core API's 5000/hour
#: does not apply, and the gap is the whole reason this is paced: a range needing
#: ten pages is twenty seconds of requests against a sixty-second budget, and
#: without a pace the seventh page is a 403. A range needing more than 30 pages is
#: over the result cap anyway, so the bound is self-enforcing.
SEARCH_REQUESTS_PER_MINUTE = 30

#: The floor between two search requests, derived from the limit above rather
#: than hard-coded, so the two cannot drift apart.
DEFAULT_SEARCH_PACE_SECONDS = 60.0 / SEARCH_REQUESTS_PER_MINUTE

#: How many requests this seam will have in flight at once.
#:
#: **Derived from the measurement that motivated it, not chosen.** A
#: ``backfill-reviews`` run over ``pingdotgg/t3code`` reaching back to a 3.5-month-old
#: floor was profiled at the socket: 96 requests, of which **88 were the pull request
#: listing, carrying 114.2 of the run's 118.7 seconds** -- 96% of the wall and none of
#: it capture work. It was the walk back to the window, at ~1.3s per 100-PR page,
#: fetched one page at a time with no dependency between neighbouring pages, so the
#: cost of asking for an older window was (PRs updated since the window / 100) x
#: 1.3s and the only thing bounding it was patience. Eight streams turn an 88-page
#: walk from ~115s into ~15s. What that costs is stated in full below rather than
#: left to be discovered by an operator whose other jobs started failing.
#:
#: **A semaphore bounds concurrency, which is not the same as bounding the rate, and
#: the gap between those two is the entire trade-off.** What a forge's abuse
#: detection looks at is how many connections one client opens at once, and eight is
#: unremarkable -- a small multiple of what a single interactive user holds open. It
#: does *not* bound requests per second, so the hourly budget is spent proportionally
#: faster. The core API allows 5000/hour ~= 1.4 req/s, and the measured sequential
#: listing already ran at 1/1.3s ~= 0.77 req/s, about **55% of that budget**: this
#: was never a frugal client, it was a slow one. At eight streams the same 88 requests
#: land in ~15s rather than ~115s. That is the same *count* -- concurrency does not
#: spend more quota on a walk of fixed length, it spends it faster -- but it spends
#: it inside a window short enough for a secondary limit to notice, and short enough
#: to starve everything else on the token. Both matter, because the token is shared:
#: it is one per-repository budget for the whole account, and the live capture path
#: spends from the same one.
#:
#: So this is a deliberate **trade of quota burst for latency**, and an operator on a
#: shared token who wants the old behaviour sets ``GITHUB_HISTORY_CONCURRENCY=1`` --
#: which is not "no concurrency" but the strictly sequential walk this replaced,
#: reachable without touching code.
#:
#: **This is not a replacement for :data:`DEFAULT_SEARCH_PACE_SECONDS`, and the two
#: constants are not two answers to one question.** Search is 30/minute, a *rate*
#: limit, so pacing is the only correct instrument there and concurrency cannot help
#: at all -- issuing the same pages faster just reaches the 403 sooner. This is the
#: core API's 5000/hour, two orders of magnitude larger per unit of time, where the
#: diagnosis was serialisation rather than a limit in reach. Dropping the search pace
#: because concurrency now exists would put back the 403 that
#: :data:`SEARCH_REQUESTS_PER_MINUTE` documents, one page into the walk.
HISTORY_READ_CONCURRENCY = 8

#: The per-request timeout for an ordinary read: metadata, listings, comments. Long
#: enough to survive a slow TLS handshake on a poor link, short enough that a hung
#: connection is a delay an operator notices rather than a hang they blame on the
#: forge.
REQUEST_TIMEOUT_SECONDS = 30.0

#: **The diff read gets twice the ordinary budget, and that asymmetry is the point.**
#: Every other read on this seam returns a page of JSON measured in kilobytes; a
#: unified diff on a large pull request is measured in megabytes and arrives
#: incrementally, so the one call that transfers orders of magnitude more bytes is
#: given orders of magnitude less patience to fail in. It is applied as a per-request
#: override rather than by building a second client, because the distinction is about
#: one call site's payload and not about a different connection policy -- and because
#: needing two pools is exactly what having one pool makes unnecessary.
#:
#: Its callers are the question generator, which cannot ask about a change without a
#: diff to ask about, and ``kojutsu ask``, which blocks on the same value. Unifying
#: the two timeouts would read as tidier and would quietly put a 30-second ceiling
#: back on the largest transfer this client makes.
DIFF_TIMEOUT_SECONDS = 60.0

#: How many times a request is retried after a rate limit (429, or a 403 the
#: rate-limit headers own) or a server error (5xx) before the response is
#: returned to the caller for normal error handling. Five, matching the history
#: reader's ``DEFAULT_RATE_LIMIT_RETRIES`` so the two paths give up at the same
#: point rather than one outlasting the other against the same outage.
DEFAULT_GITHUB_REQUEST_RETRIES = 5

#: Ceiling for one backoff wait. A forge that asks for an hour is asking this
#: process to stop, and honouring it literally would look like a hang.
MAX_GITHUB_BACKOFF_SECONDS = 60.0

#: ``per_page`` for search. Same 100 ceiling as the core API.
SEARCH_PAGE_SIZE = 100

#: How many pages one listing walk fetches before stopping. History is unbounded
#: in principle, so an unbounded walk is a run that never ends; the bound keeps
#: one pathological pull request from pinning a backfill forever. Reaching it
#: with a full last page reads as ``truncated`` on the result, because the only
#: way to know whether a 101st page exists would be a 101st read past the bound.
_MAX_LISTING_PAGES = 100


T = TypeVar("T")


@dataclass(frozen=True)
class PagedResult(Generic[T]):
    """One bounded page-walk: what was read, and whether the bound cut it short.

    A plain list cannot distinguish "this pull request had 30 comments" from
    "this pull request had 30,000 and the walk stopped at 10,000", and those two
    facts support completely different conclusions about coverage. Returning
    only the list is the mistake; ``truncated`` is the correction -- the same
    shape as :class:`PullRequestSearch`, which carries the same flag for the
    same reason.
    """

    items: list[T]
    truncated: bool


#: The widest range Kojutsu will enumerate, stated rather than discovered.
#:
#: This exists because of :data:`SEARCH_RESULT_CAP`, not despite it. A range
#: narrow enough to be a real unit of work is unlikely to exceed 1000 pull
#: requests; a range wide enough to do so is a caller mistake that is far cheaper
#: to name than to explain afterwards, because the failure mode of running it is a
#: truncated read that reports itself as complete. One year. A range wider than
#: this is refused with the bound in the message, so the fix is obvious.
MAX_RANGE_SPAN_DAYS = 365


def parse_range_boundary(value: str | date, field: str) -> date:
    """Parse one end of a range, or refuse it naming ``field``.

    ISO ``YYYY-MM-DD``, which is what the search grammar accepts and the only
    date form this seam accepts. A full timestamp is *not* accepted: the search
    query is a pair of dates, so accepting one and truncating it would mean the
    two ends of a range mean different kinds of thing, and the caller's
    ``since=2024-01-01T12:00:00Z`` would quietly become midnight.
    """
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise GitHubRangeError(f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}")
    text = value.strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise GitHubRangeError(
            f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}. "
            f"Both ends of the range are dates, not timestamps: the search "
            f"grammar compares days."
        ) from None
    # ``date.fromisoformat`` accepts other ISO shapes on 3.11. Round-tripping is
    # the check that the caller wrote the one form this seam documents.
    if parsed.isoformat() != text:
        raise GitHubRangeError(f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}")
    return parsed


def validate_pull_request_range(since: str | date, until: str | date) -> tuple[date, date]:
    """Validate an inclusive ``since..until`` range, or refuse it.

    Three refusals, in the order that makes the message most useful: each field is
    checked for being a date at all, then for the order, then for the width. All
    of it happens before a request is issued, because a range validated by the
    response is a range that has already cost a request and told GitHub what the
    caller was looking for.

    Raises:
        GitHubRangeError: naming ``since`` or ``until`` as the field at fault.
    """
    start = parse_range_boundary(since, "since")
    end = parse_range_boundary(until, "until")
    if start > end:
        raise GitHubRangeError(
            f"since ({start.isoformat()}) is after until ({end.isoformat()}): "
            f"the range is inverted. since and until are both at fault -- an "
            f"inverted range matches nothing, which is indistinguishable from an "
            f"empty one unless it is refused. Pass the earlier date as since."
        )
    span_days = (end - start).days
    if span_days > MAX_RANGE_SPAN_DAYS:
        raise GitHubRangeError(
            f"since..until spans {span_days} days, more than the {MAX_RANGE_SPAN_DAYS}-day "
            f"maximum. The GitHub search API caps at {SEARCH_RESULT_CAP} results with no "
            f"error, so a range this wide is a silently truncated read rather than a "
            f"historical ingest. Split it into ranges of at most {MAX_RANGE_SPAN_DAYS} days."
        )
    return start, end


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the wait GitHub asked for, or ``None`` if it did not say."""
    header = (response.headers.get("retry-after") or "").strip()
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            return None
    return None


def github_backoff_delay(
    response: httpx.Response, attempt: int, *, max_backoff: float = MAX_GITHUB_BACKOFF_SECONDS
) -> float:
    """How long to wait before retrying ``response``, the shared backoff policy.

    Prefer what the forge asked for via ``Retry-After``; fall back to doubling
    from one second. The forge's explicit ask is honoured exactly -- waking
    early from a stated interval just re-hits the limit, so jitter applies to
    the computed fallback only, where it keeps concurrent readers from waking
    in lockstep. Bounded throughout, so a large ``Retry-After`` cannot park the
    process. The history reader (:mod:`kojutsu.core.backfill_reviews_client`)
    delegates here rather than keeping its own copy, so there is one definition
    of "how long" for both the live path and the backfill path.
    """
    requested = _retry_after_seconds(response)
    if requested is not None:
        return max(0.0, min(requested, max_backoff))
    delay = float(2 ** max(attempt - 1, 0)) * random.uniform(0.5, 1.5)  # noqa: S311 - backoff jitter, not cryptographic
    return max(0.0, min(delay, max_backoff))


def _within_range(created_at: datetime | None, start: date, end: date) -> bool | None:
    """Whether ``created_at`` falls in ``[start, end]``, or ``None`` if unknowable.

    The comparison is re-applied here even though the query already asked for
    this range, because the query is answered by an index and the hit is data.
    They are two different claims: the index can be stale, wrong, or simply
    lenient about a boundary, and "the range is inclusive on both ends" is a
    property of *this* reader, so it is enforced here rather than trusted to
    arrive correct in a URL.

    Both ends are inclusive, which is what a caller naming ``since=2024-01-01``
    and ``until=2024-01-31`` means: thirty-one days, not thirty. Times within
    the boundary days count, so a pull request created at 23:59 on either day is
    inside the range -- the range is a pair of days, not two instants.

    ``None`` means the hit stated no creation date, so it cannot be shown to be
    in range. That is not the same as being out of it, and returning a definite
    answer either way would be the guess this exists to avoid.
    """
    if created_at is None:
        return None
    return start <= created_at.date() <= end


@dataclass(frozen=True)
class PullRequestSearch:
    """The result of one date-range enumeration, and what it did not cover.

    The two halves travel together on purpose. A list of pull requests on its own
    cannot distinguish "this range held three pull requests" from "this range held
    three thousand and the API gave me the first thousand", and those two facts
    support completely different conclusions about coverage. Returning only the
    list is the mistake; ``truncated`` is the correction.

    ``total_count`` is GitHub's own count of what the index says matches, which is
    not the same as ``len(pull_requests)`` for three separate reasons: a caller
    limit, the 1000-result cap, and results the search index has not caught up
    with. Keeping the number lets a caller compute the gap rather than be told it
    exists.

    ``out_of_range`` and ``undated`` count the hits that were retrieved and then
    not returned, for two different reasons worth telling apart: the index
    matched something whose own creation date falls outside the range (the index
    and the data disagree), or the hit stated no creation date at all (so nothing
    can be said either way). Neither is folded into ``truncated``, because a
    filter doing its job is not an incomplete read.

    There is deliberately no ``__iter__`` and no ``__len__``. Iterating the result
    would let a caller write ``for pull in search(...)`` and never see
    ``truncated``, which is the one field that decides whether the list is
    evidence of coverage. The list is named, and so is the caveat.
    """

    pull_requests: list[GitHubPullRequest]
    truncated: bool
    total_count: int
    #: Hits retrieved whose own creation date falls outside the requested range.
    out_of_range: int = 0
    #: Hits the index returned that stated no creation date, and so could not be
    #: shown to be inside the range. Counted rather than silently dropped.
    undated: int = 0
    #: Pages actually fetched. One more than the number of paced waits.
    pages_fetched: int = 0
    #: Hits the index actually returned, whether or not they were kept. Compared
    #: against ``total_count`` to decide ``truncated``.
    hits_seen: int = 0

    @property
    def numbers(self) -> list[int]:
        """The pull request numbers enumerated, in range order."""
        return [pull.number for pull in self.pull_requests]


class GitHubClient:
    """Client for GitHub API: PR details, diff, and issue comments.

    **One connection pool for the client's whole life, not one per request.** Every
    call used to open its own ``httpx.Client`` and close it on the way out, so each of
    those requests paid a DNS lookup, a TCP handshake and a TLS negotiation before
    transferring a single byte -- and one ``backfill-reviews`` run over
    ``pingdotgg/t3code`` issued 96 of them, which is 96 handshakes to read one
    repository's history. Against ``api.github.com`` that setup is comparable to the
    transfer it precedes, which is why the profiling that found this measured 96% of a
    118.7-second run inside the listing requests and about 3% outside HTTP entirely.

    Reuse is also what makes :data:`HISTORY_READ_CONCURRENCY` reachable at all:
    ``httpx.Client`` is documented safe to use from several threads, so one pool
    serves concurrent readers, whereas a pool per request would have made concurrency
    mean N simultaneous handshakes rather than N simultaneous transfers.

    Built **lazily**, because most of these clients perform one short call and are then
    dropped -- ``GitHubClient(token).get_pull_diff(...)`` is the shape at six call sites
    -- and eagerly opening a pool for an object that is about to become garbage trades a
    real cost for a tidy-looking one. The lock guards that first construction only;
    afterwards ``self._client`` is an unsynchronised attribute load, which is safe
    because the reference is published before the assignment that writes it completes.
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        *,
        search_pace_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rate_limit_retries: int = DEFAULT_GITHUB_REQUEST_RETRIES,
        max_backoff_seconds: float = MAX_GITHUB_BACKOFF_SECONDS,
    ) -> None:
        self._token = token
        self._base = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._authenticated_login: str | None = None
        self._scope_checked = False
        #: Minimum seconds between two *search* requests, and the sleep used to
        #: wait them out. Both injectable so a test can assert on the pacing
        #: without spending it: the arithmetic is the behaviour under test, and a
        #: test that actually sleeps two seconds per page to check it is a test
        #: nobody runs.
        self._search_pace_seconds = (
            DEFAULT_SEARCH_PACE_SECONDS if search_pace_seconds is None else search_pace_seconds
        )
        self._sleep = sleep
        #: How many rate-limit/server-error responses are absorbed with a
        #: backoff before the response is returned for normal error handling.
        #: Injectable alongside ``sleep`` so a test can exhaust the policy
        #: without spending it. ``0`` disables retries: the first response is
        #: returned as-is, which is how a caller opts a latency-sensitive read
        #: out of waiting.
        self._rate_limit_retries = max(rate_limit_retries, 0)
        self._max_backoff_seconds = max_backoff_seconds
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    def http(self) -> httpx.Client:
        """The one pool this client reads through, built on first use.

        Left open until :meth:`close`. Making that a call rather than a convention is
        what keeps it from being a leak: a client held for the length of a run should
        hand its sockets back when the run ends, and one built for a single call can
        simply be dropped, which closes its transport with it.
        """
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
            return self._client

    def close(self) -> None:
        """Close the connection pool. Safe to call twice, and safe on a client that
        never opened one.

        The attribute is deliberately not cleared, so a use after close raises httpx's
        own "client has been closed" error instead of quietly building a second pool
        behind the caller's back. Use-after-close is a bug and the failure ought to say
        so rather than appear as a mysteriously fresh connection.
        """
        client = self._client
        if client is not None:
            client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _url(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self._base}/{path}"

    def _verified(self, response: httpx.Response) -> httpx.Response:
        """Raise for a bad status and enforce token scope on the way past.

        The check lives here rather than in one method so it cannot be skipped by a
        code path that happens not to call the identity endpoint: every response the
        pipeline receives is inspected, including during a plan-only run.

        Rate limiting is recognised *before* ``raise_for_status`` and before the
        scope check. Both of those matter: a bare 403 from an exhausted search
        budget is indistinguishable from a permissions problem by status alone, and
        a 403 read as a scope problem produces a configuration error that no amount
        of waiting will fix, which is the worst possible advice for a caller
        holding a perfectly good token.
        """
        self._raise_for_rate_limit(response)
        response.raise_for_status()
        if not self._scope_checked:
            self.assert_minimum_scope(response.headers.get("x-oauth-scopes"))
            self._scope_checked = True
        return response

    @staticmethod
    def _raise_for_rate_limit(response: httpx.Response) -> None:
        """Raise :class:`GitHubRateLimitError` if this response is a rate limit.

        Detected on the headers GitHub documents for the purpose rather than on
        the body text alone, because the body is user-visible and the headers are
        not. A 403 with ``x-ratelimit-remaining: 0`` is unambiguous. The body
        match is a fallback for responses that omit the header, which a secondary
        or differently-versioned limit does.
        """
        if response.status_code not in {403, 429}:
            return
        remaining = (response.headers.get("x-ratelimit-remaining") or "").strip()
        if remaining not in {"", "0"}:
            return
        if remaining == "0":
            body_hint = ""
        else:
            try:
                body_hint = response.text.casefold()
            except Exception:  # a body we cannot read is not a signal
                body_hint = ""
            if "rate limit" not in body_hint and "secondary rate" not in body_hint:
                return
        resource = (response.headers.get("x-ratelimit-resource") or "").strip()
        retry_after = _retry_after_seconds(response)
        reset = (response.headers.get("x-ratelimit-reset") or "").strip()
        if retry_after is None and reset.isdigit():
            retry_after = max(0.0, float(reset) - time.time())
        window = f"the {resource} " if resource else ""
        wait = f" Retry after {retry_after:.0f}s." if retry_after is not None else ""
        raise GitHubRateLimitError(
            f"GitHub refused the request: {window}rate limit exhausted "
            f"(HTTP {response.status_code}).{wait} A rate-limited read returns "
            f"fewer results than reality holds, so it is raised rather than "
            f"treated as a short range.",
            resource=resource,
            retry_after=retry_after,
        )

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json: Any | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Send one request, absorbing rate limits and server errors with backoff.

        The single retry policy for this client: every ``get``/``list``/
        ``search``/``post`` below goes through here rather than calling the pool
        directly, so a 429 on the live webhook path is waited out instead of
        failing the capture the way a single-shot request would.

        Retryable is a 429, a 5xx on an idempotent ``GET`` (a gateway error may
        mean the request never reached the application, but for a ``POST`` it
        may equally mean the write applied and only the response was lost, so a
        blind ``POST`` retry can double-post -- posts therefore retry rate
        limits only), or a 403 the rate-limit headers own. Anything else,
        including a genuine 403/401/404, is returned immediately for
        :meth:`_verified` to raise on. When the bounded retries are exhausted
        the last response is returned rather than raised, so the caller raises
        the same taxed error it would have raised without retries and no new
        error type leaks out of this seam.
        """
        attempt = 0
        client = self.http()
        upper = method.upper()
        request_kwargs: dict[str, Any] = {}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        while True:
            attempt += 1
            response = client.request(
                upper, url, params=params, headers=headers, json=json, **request_kwargs
            )
            retryable = response.status_code == 429
            if response.status_code >= 500 and upper == "GET":
                retryable = True
            if not retryable:
                try:
                    self._raise_for_rate_limit(response)
                except GitHubRateLimitError:
                    retryable = True
            if not retryable or attempt > self._rate_limit_retries:
                return response
            self._sleep(
                github_backoff_delay(response, attempt, max_backoff=self._max_backoff_seconds)
            )

    @staticmethod
    def assert_minimum_scope(scopes_header: str | None) -> None:
        """Refuse a classic token that grants far more than Kojutsu needs.

        GitHub reports classic personal access token scopes in the ``x-oauth-scopes``
        response header, and leaves it empty for fine-grained tokens. A fine-grained
        token is therefore checked by construction, and this only ever fires on a
        classic one.

        The pipeline reads pull requests and posts issue comments. It never needs
        repository contents, and it never needs a scope covering every repository the
        account can reach. ``repo`` in particular carries read and write over code for
        all of them, so a credential holding it is not a read-and-comment credential
        no matter how it is used here.

        Set ``GITHUB_ALLOW_BROAD_SCOPES=true`` to downgrade this to a warning when
        migrating an existing setup; do not leave it set in production.
        """
        if scopes_header is None or not scopes_header.strip():
            return
        granted = {item.strip().casefold() for item in scopes_header.split(",") if item.strip()}
        offending = sorted(granted & _OVER_SCOPED_CLASSIC_SCOPES)
        if not offending:
            return
        message = (
            f"GITHUB_TOKEN carries over-scoped classic permissions: {', '.join(offending)}. "
            f"Kojutsu only needs to read pull requests and post issue comments on one "
            f"repository. Use a fine-grained token limited to that repository; see "
            f"docs/github-seam.md."
        )
        if os.getenv("GITHUB_ALLOW_BROAD_SCOPES", "").strip().casefold() in {"1", "true", "yes"}:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return
        raise GitHubTokenScopeError(message)

    def get_authenticated_user(self) -> str:
        """Return the login authenticated by this client."""
        if self._authenticated_login is not None:
            return self._authenticated_login
        response = self._request_with_retry("GET", self._url("user"), headers=self._headers)
        self._verified(response)
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("login"), str):
            raise GitHubPayloadError("GitHub returned an invalid authenticated user")
        login = str(payload["login"]).strip()
        if not login:
            raise GitHubPayloadError("GitHub returned an invalid authenticated user")
        self._authenticated_login = login
        return login

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Fetch PR metadata (title, body, head ref, etc.)."""
        r = self._request_with_retry(
            "GET",
            self._url(f"repos/{owner}/{repo}/pulls/{pr_number}"),
            headers=self._headers,
        )
        self._verified(r)
        return GitHubPullRequest.model_validate(r.json())

    def list_open_pull_requests(
        self, owner: str, repo: str, *, limit: int = 30
    ) -> list[GitHubPullRequest]:
        """List open pull requests, newest activity first.

        Bounded on the way out as well as on the request: GitHub's ``per_page``
        caps at 100 and silently returns that, so a caller asking for fewer than
        it sent would otherwise get a longer list than it asked for and have no
        way to tell.
        """
        page_size = min(max(limit, 1), 100)
        r = self._request_with_retry(
            "GET",
            self._url(f"repos/{owner}/{repo}/pulls"),
            params={
                "state": "open",
                "sort": "updated",
                "direction": "desc",
                "per_page": page_size,
            },
            headers=self._headers,
        )
        self._verified(r)
        payload = r.json()
        if not isinstance(payload, list):
            raise GitHubPayloadError("GitHub returned an invalid pull request list")
        pulls = [
            GitHubPullRequest.model_validate(entry) for entry in payload if isinstance(entry, dict)
        ]
        return pulls[: max(limit, 0)]

    def search_pull_requests(
        self,
        owner: str,
        repo: str,
        *,
        since: str | date,
        until: str | date,
        limit: int = SEARCH_PAGE_SIZE,
        settings: Settings,
    ) -> PullRequestSearch:
        """Enumerate the pull requests **created** within ``since..until``.

        This exists because :meth:`list_open_pull_requests` cannot answer the
        question. ``GET /pulls`` takes no date parameter and only ever sees
        ``state=open``, so anything already merged in a historical range is
        invisible to it -- exactly the pull requests a backfill exists to collect.
        The search API is the only GitHub surface that can enumerate a range, and
        the only one that indexes by creation date.

        **Read-only, by construction.** This method issues one HTTP verb, ``GET``,
        against a path that has no write form. There is no code path from here to a
        mutating request, and ``tests/test_github_range.py`` asserts it by
        recording every method issued through a stub transport and failing on any
        that mutates, rather than leaving the claim to review.

        **This endpoint is not a consistent read of the core API.** It indexes
        asynchronously, so a pull request opened seconds ago may simply not be
        there yet, and there is no way to ask whether it will be. That is
        immaterial for a historical ingest, where the range closed days ago and
        the index has long since caught up. It is disqualifying for a webhook, and
        it is the reason this is a separate method from
        :meth:`list_open_pull_requests` rather than a parameter on it: the two
        answer "what is open" and "what existed then", and only the first is
        safe to read the instant after a merge.

        Two limits are handled rather than assumed, because both fail quietly:

        * **30 requests per minute**, not the core API's 5000/hour. A naive
          page-through exhausts that on any range worth backfilling, so requests
          are paced against :data:`DEFAULT_SEARCH_PACE_SECONDS`.
        * **1000 results**, and no error past it. A wider range is silently
          incomplete, so the result carries ``truncated`` and the index's own
          ``total_count``; a short list is never returned as if it were whole.

        Ordering is pinned to ``created`` ascending. Without a total order, page
        two of a paginated walk can repeat an item from page one or skip one
        between them, and neither shows up in a list that looks complete.

        ``settings`` is required rather than optional so the allowlist cannot be
        skipped by omission. ``repository_allowed`` is called *before* the first
        request, reusing the one definition the webhook and the MCP server use --
        a second implementation here is exactly the drift
        ``docs/design-review/read-path.md`` documents, and a read path that
        authorised something the write path refused would capture a review it
        could not read back.
        """
        repository = f"{owner}/{repo}"
        if not repository_allowed(repository, settings):
            raise GitHubRepositoryNotAllowedError(
                f"{repository} is not in GITHUB_WEBHOOK_ALLOWED_REPOSITORIES. "
                f"Refusing before any request is issued: an allowlist enforced "
                f"after the request has been sent is not an allowlist."
            )
        start, end = validate_pull_request_range(since, until)

        # Bounded twice: by what the caller asked for, and by what the API will
        # return past a thousand. The cap is the one that matters and the one
        # GitHub does not enforce. A ``limit`` below one still fetches, because
        # ``total_count`` is worth having and it is the only way a caller asking
        # for nothing learns that the range is not empty.
        cap = min(max(limit, 0), SEARCH_RESULT_CAP)
        page_size = min(max(cap, 1), SEARCH_PAGE_SIZE)
        query = f"repo:{repository} is:pr created:{start.isoformat()}..{end.isoformat()}"

        collected: list[GitHubPullRequest] = []
        total_count = 0
        incomplete_results = False
        out_of_range = 0
        undated = 0
        hits_seen = 0
        page = 1
        pages_fetched = 0
        deadline: float | None = None

        while True:
            if deadline is not None:
                self._pace(deadline)
            request_started = time.monotonic()
            deadline = request_started + self._search_pace_seconds
            response = self._request_with_retry(
                "GET",
                self._url("search/issues"),
                params={
                    "q": query,
                    "per_page": page_size,
                    "page": page,
                    "sort": "created",
                    "order": "asc",
                },
                headers=self._headers,
            )
            self._verified(response)
            pages_fetched += 1
            items, total_count, incomplete_results = self._search_page(response)
            # An empty page ends the walk. GitHub can return one mid-sequence
            # when the index shifts under a paginated read, and the honest
            # response is to stop and report the shortfall -- not to keep
            # asking, which either loops or walks off the end of the index.
            if not items:
                break
            hits_seen += len(items)
            for entry in items:
                pull = GitHubPullRequest.model_validate(entry)
                kept = _within_range(pull.created_at, start, end)
                if kept is None:
                    # No creation date, so no way to show it is in range.
                    # Counting it is the alternative to pretending the filter
                    # was exact.
                    undated += 1
                    continue
                if kept:
                    collected.append(pull)
                else:
                    out_of_range += 1
            if len(collected) >= cap or len(items) < page_size:
                break
            page += 1
            if page > SEARCH_RESULT_CAP // page_size:
                # Off the end of what the cap can return, whatever the index
                # says is there.
                break

        return PullRequestSearch(
            pull_requests=collected[:cap],
            # Truncated is a claim about coverage, so it is measured against what
            # the index *offered*, not against what survived the date filter: a
            # filter rejecting a stale hit is the filter working, and calling that
            # an incomplete read would train an operator to ignore the flag.
            truncated=incomplete_results or hits_seen < total_count,
            total_count=total_count,
            out_of_range=out_of_range,
            undated=undated,
            pages_fetched=pages_fetched,
            hits_seen=hits_seen,
        )

    def _pace(self, deadline: float) -> None:
        """Sleep until ``deadline`` if the last request beat the pace.

        Deadline-based rather than ``sleep(interval)`` after each request, so a
        request that took longer than the interval does not buy credit for the
        next one. Ten pages of a slow connection would otherwise compress into
        a burst at the end, which is the thing being prevented.
        """
        remaining = deadline - time.monotonic()
        if remaining > 0:
            self._sleep(remaining)

    @staticmethod
    def _search_page(response: httpx.Response) -> tuple[list[dict], int, bool]:
        """Return ``(items, total_count, incomplete_results)`` from a search page."""
        payload = response.json()
        if not isinstance(payload, dict):
            raise GitHubPayloadError("GitHub returned an invalid pull request search response")
        items = payload.get("items")
        if not isinstance(items, list):
            raise GitHubPayloadError("GitHub returned an invalid pull request search response")
        if any(not isinstance(item, dict) for item in items):
            raise GitHubPayloadError("GitHub returned an invalid pull request search response")
        raw_total = payload.get("total_count", 0)
        total = raw_total if isinstance(raw_total, int) and raw_total >= 0 else 0
        # ``incomplete_results`` is GitHub saying out loud that it gave up
        # counting. That is the same fact as a silent cap, and it has to reach
        # the caller rather than be read as a complete count.
        incomplete = bool(payload.get("incomplete_results"))
        return items, total, incomplete

    def get_pull_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the unified diff for the PR.

        The one read on this seam that overrides the shared timeout, and the reason
        is payload size rather than a different connection policy: see
        :data:`DIFF_TIMEOUT_SECONDS`. It is a per-request argument precisely so the
        asymmetry survives sharing one pool -- had it been expressed as a second
        ``httpx.Client``, unifying the two timeouts later would have looked like
        tidying up an accident rather than like removing a deliberate difference.
        """
        r = self._request_with_retry(
            "GET",
            self._url(f"repos/{owner}/{repo}/pulls/{pr_number}"),
            headers={**self._headers, "Accept": "application/vnd.github.v3.diff"},
            timeout=DIFF_TIMEOUT_SECONDS,
        )
        self._verified(r)
        return r.text

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> PagedResult[str]:
        """Fetch list of changed file paths in the PR."""
        files: list[str] = []
        truncated = False
        page = 1
        while True:
            r = self._request_with_retry(
                "GET",
                self._url(f"repos/{owner}/{repo}/pulls/{pr_number}/files"),
                headers=self._headers,
                params={"per_page": 100, "page": page},
            )
            self._verified(r)
            payload = r.json()
            if not isinstance(payload, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("filename"), str)
                for item in payload
            ):
                raise GitHubPayloadError("GitHub returned an invalid pull request file list")
            files.extend(item["filename"] for item in payload)
            if len(payload) < 100:
                break
            if page >= _MAX_LISTING_PAGES:
                truncated = True
                break
            page += 1
        return PagedResult(items=files, truncated=truncated)

    def post_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> GitHubComment:
        """Post a comment on an issue or PR (same number). Returns the created comment."""
        r = self._request_with_retry(
            "POST",
            self._url(f"repos/{owner}/{repo}/issues/{issue_number}/comments"),
            headers=self._headers,
            json={"body": body},
        )
        self._verified(r)
        return GitHubComment.model_validate(r.json())

    def list_issue_comments(
        self, owner: str, repo: str, issue_number: int, *, per_page: int = 100
    ) -> PagedResult[GitHubComment]:
        """List all issue/PR comments across every GitHub page."""
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100")
        comments: list[GitHubComment] = []
        truncated = False
        page = 1
        while True:
            r = self._request_with_retry(
                "GET",
                self._url(f"repos/{owner}/{repo}/issues/{issue_number}/comments"),
                headers=self._headers,
                params={"per_page": per_page, "page": page},
            )
            self._verified(r)
            payload = r.json()
            if not isinstance(payload, list):
                raise GitHubPayloadError("GitHub returned an invalid comment list")
            comments.extend(GitHubComment.model_validate(comment) for comment in payload)
            if len(payload) < per_page:
                break
            if page >= _MAX_LISTING_PAGES:
                truncated = True
                break
            page += 1
        return PagedResult(items=comments, truncated=truncated)

    def list_pull_request_reviews(
        self, owner: str, repo: str, pr_number: int, *, per_page: int = 100
    ) -> PagedResult[PullRequestReview]:
        """List a PR's submitted reviews, across every page.

        Read-only, and separate from ``list_issue_comments`` because a review is a
        different object with a different anchor: it carries a verdict, and its
        inline comments are addressed to a line in a diff rather than to a thread.
        Backfilling a pull request needs both, and the webhook path that captures
        them live already exists -- what was missing was a way to reach it
        historically.
        """
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100")
        reviews: list[PullRequestReview] = []
        truncated = False
        page = 1
        while True:
            r = self._request_with_retry(
                "GET",
                self._url(f"repos/{owner}/{repo}/pulls/{pr_number}/reviews"),
                headers=self._headers,
                params={"per_page": per_page, "page": page},
            )
            self._verified(r)
            payload = r.json()
            if not isinstance(payload, list):
                raise GitHubPayloadError("GitHub returned an invalid review list")
            reviews.extend(PullRequestReview.model_validate(x) for x in payload)
            if len(payload) < per_page:
                break
            if page >= _MAX_LISTING_PAGES:
                truncated = True
                break
            page += 1
        return PagedResult(items=reviews, truncated=truncated)

    def list_pull_request_review_comments(
        self, owner: str, repo: str, pr_number: int, *, per_page: int = 100
    ) -> PagedResult[PullRequestReviewComment]:
        """List a PR's inline review comments, across every page.

        These are addressed to ``path``/``line``/``side`` and carry the surrounding
        ``diff_hunk``, which is untrusted input like any other PR text. The anchor
        is what makes an inline comment worth keeping months later, so the model
        keeps the location rather than only the prose.
        """
        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100")
        comments: list[PullRequestReviewComment] = []
        truncated = False
        page = 1
        while True:
            r = self._request_with_retry(
                "GET",
                self._url(f"repos/{owner}/{repo}/pulls/{pr_number}/comments"),
                headers=self._headers,
                params={"per_page": per_page, "page": page},
            )
            self._verified(r)
            payload = r.json()
            if not isinstance(payload, list):
                raise GitHubPayloadError("GitHub returned an invalid review comment list")
            comments.extend(PullRequestReviewComment.model_validate(x) for x in payload)
            if len(payload) < per_page:
                break
            if page >= _MAX_LISTING_PAGES:
                truncated = True
                break
            page += 1
        return PagedResult(items=comments, truncated=truncated)

    def _list_for_reconciliation(
        self, owner: str, repo: str, pr_number: int
    ) -> list[GitHubComment]:
        """Comments to reconcile markers against, warning when the read is short.

        A truncated listing can miss a marker past the cap, and a missed marker
        reconciles as "not posted yet" -- a duplicate question comment. The
        warning does not stop the posting (refusing to ask is worse than
        risking a duplicate), but an operator reading the log can tell a
        duplicate from a bug.
        """
        result = self.list_issue_comments(owner, repo, pr_number)
        if result.truncated:
            logger.warning(
                "Comment listing for %s#%d hit the %d-page cap; reconciling "
                "against a partial list, markers past the cap may double-post.",
                f"{owner}/{repo}",
                pr_number,
                _MAX_LISTING_PAGES,
            )
        return result.items

    def post_questions_as_pr_comments(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        questions: list[tuple[str, str]],
        *,
        existing_comments: list[GitHubComment] | None = None,
        on_comment: Callable[[str, str, GitHubComment], None] | None = None,
    ) -> list[GitHubComment]:
        """Post or reconcile each question using authenticated Kojutsu provenance."""
        if not questions:
            return []
        comments = (
            list(existing_comments)
            if existing_comments is not None
            else self._list_for_reconciliation(owner, repo, pr_number)
        )
        authenticated_login = self.get_authenticated_user().casefold()
        by_marker: dict[str, GitHubComment] = {}
        by_text: dict[str, GitHubComment] = {}
        for comment in comments:
            if comment.user.login.casefold() != authenticated_login:
                continue
            marker_id = extract_question_id_from_comment_body(comment.body)
            if marker_id and marker_id not in by_marker:
                by_marker[marker_id] = comment
            question_text = extract_question_text_from_comment_body(comment.body)
            if marker_id and question_text and question_text not in by_text:
                by_text[question_text] = comment

        reconciled: list[GitHubComment] = []
        for qid, text in questions:
            comment = by_marker.get(qid) or by_text.get(text)
            if comment is None:
                comment = self.post_issue_comment(
                    owner, repo, pr_number, kojutsu_comment_body(qid, text)
                )
                if comment.user.login.casefold() != authenticated_login:
                    raise GitHubPayloadError(
                        "GitHub comment provenance did not match the token owner"
                    )
                by_marker[qid] = comment
                by_text[text] = comment
            resolved_id = extract_question_id_from_comment_body(comment.body) or qid
            if on_comment is not None:
                on_comment(resolved_id, text, comment)
            reconciled.append(comment)
        return reconciled
