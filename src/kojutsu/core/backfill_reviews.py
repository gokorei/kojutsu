"""Reconstruct pre-deployment history as records marked ``capture_source: backfilled``.

Everything in the corpus so far was captured from the moment a webhook was
installed. This module reads the pull request reviews and comments that predate
that, and turns them into records — records that say, in their own provenance,
that they were read after the fact rather than witnessed.

**It writes no records of its own.** Every object it reads is handed to the same
collectors the webhook handler uses, with the same event shape, so there is one
writer per record kind. A second writer would not merely duplicate work: it would
be free to describe a review slightly differently, and a corpus with two
vocabularies for one fact cannot be filtered, counted or reasoned about. The
collectors also carry the gates that make a record defensible — the authorised
association check, the independence computation, the dedupe claim — and a writer
that skipped them would be skipping the parts that decide what the record is
worth.

What the collectors cannot do is mark a record as a read. They are built for
deliveries and hardcode that provenance, so :class:`BackfillReadSink` restamps
what they hand it. The stamp is applied here, by the runner, rather than left to
the caller: a backfill that could be pointed at an unstamped sink would be one
code path that produces records claiming a signed delivery that never happened.

**There is no cursor, and the reason is identity rather than discipline.** A
cursor records how far a previous run got. It drifts when a run is interrupted
between the write and the cursor update, it is lost when a machine is rebuilt, and
above all it cannot answer the only question that matters here — whether this run
saw the same object the last one did. The semantic event ids derive identity from
the forge's own object identity, so re-reading an object collides on the claim and
writes nothing. Re-running the same range is therefore not merely safe but free of
consequence, which is what makes an interrupted run resumable without a single byte
of bookkeeping. A cursor would add a second, weaker deduplication layer whose
failure mode is silent; this has none.

**Safe is not the same as useful, and the budget used to make the two disagree.**
Charged per object read, the budget pinned the walk to the first page of a busy
repository: enumeration starts at the most recently updated page every time, so a
second run re-read exactly the objects the first had stored, spent its whole
budget on objects it already held, and stopped in the same place. The advice to
re-run the same range was true of the writes and false about the progress. What
makes it true of both is charging for **new work**: an object the store already
holds costs nothing, and so does one that turns out to have nothing to capture.
Only an object that stores a record is charged, because only that is work this run
did that the last one did not.

Silences are the majority of what such a walk meets — on ``pingdotgg/t3code``, 168
of the first 200 objects read had nothing to capture and 2 were already held — so
charging for duplicates alone would leave the walk precisely where it was. An
object that stores nothing is not new work, and a budget charged for reading is a
budget charged against progress.

The reads that accounting stops bounding are not hidden by it. ``objects_read``
counts every object examined, not every object stored, and a re-run over an
already-stored range re-reads it; what bounds the reads is the page ceiling, not
``--max-objects``. That is the trade this makes, and ``--until`` is the other half
of it: bounding the top of the range is what turns a re-read from the only option
into a deliberate one.

**A date floor is a policy decision and is required, not defaulted.** Pre-deployment
history contains the pull requests nobody ever reviewed. Reconstructing those as
empty observations would assert a completeness that was never true — it would say
"we looked and there was nothing" about changes nobody discussed, and a reader
counting them would be counting a fabrication. The floor is therefore an editorial
statement by whoever runs the command, which is why the command will not choose
one, and why the floor in force is reported in the output.

**A ceiling is the same decision at the other end, and without one the floor is
only half a range.** The walk enumerates most recently updated first, so an
unbounded run necessarily includes everything up to the present moment: an
operator who asks for the last quarter of a busy repository gets the last page of
it, which is a few days, and the summary cannot tell them so. ``--until`` is the
one comparison that stops it — the first object newer than the ceiling ends the
walk, the mirror of the floor's — and it is optional because "up to now" is a
real answer. What it is not is a tuning knob: bounding the top of the range is how
an operator reconstructs one era rather than whatever happened to accumulate most
recently, and a quarter measured this way is the same quarter on the next run.

**Nothing here is representative, and the output says so.** A backfill removes the
deployment date from the picture and makes the corpus larger. It does not touch the
behavioural bias: a repository where nobody comments still produces nothing, and a
backfill over that repository faithfully reconstructs the fact that there was
nothing to reconstruct. That is a true statement about a biased sample, and it is
still a biased sample.

**The weaker guarantee is not fixable, only visible.** A backfilled record shows
what the forge says now. A comment edited since, a review body rewritten, a review
deleted outright are all indistinguishable from ones that were not. ``captured_at``
is the read time, which is the only evidence there is, and the id of the object
read is the only anchor — there was no delivery, and none is invented.

**A pull request lifecycle transition is not reconstructed.** A lifecycle record
names the change it read rather than a comment or a review, and
:func:`~kojutsu.models.capture_anchor_gaps` requires one of those two to anchor
a backfilled record; ``pr_number`` already carries the change. Rather than invent a
review or comment id to satisfy the rule, this run leaves those transitions alone
and the anchor rule stays the authority it was written to be. Reconstructing
lifecycle history needs a change to that rule, which is a change to the model
rather than to a writer.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kojutsu import allowlist
from kojutsu.config import Settings
from kojutsu.core.answer_collector import (
    capture_counts,
    process_comment_reply_outcome,
    process_review_event_outcome,
)
from kojutsu.core.backfill_reviews_client import (
    MAX_PAGES_PER_OBJECT,
    PAGE_SIZE,
    HistoryReader,
    is_missing_object,
)
from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome, KnowledgeSink, StorableRecord
from kojutsu.core.question_registry import QuestionRegistry
from kojutsu.integrations.github import (
    HISTORY_READ_CONCURRENCY,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
    extract_question_id_from_comment_body,
)
from kojutsu.models import (
    REVIEW_ID_KEY,
    CaptureSource,
    KnowledgeEntry,
    capture_anchor_gaps,
)
from kojutsu.repo_name import split_repo

#: The date format both bounds take. A date and not a timestamp: a bound is a
#: statement about which era of history is worth reconstructing, and a time of day
#: would imply a precision the policy does not have.
DATE_BOUND_FORMAT = "%Y-%m-%d"

#: The hard ceiling on new objects one run may store, whatever the operator asks
#: for. It exists because the run's cost is paid by a shared, per-repository token
#: and a shared rate limit: a runaway backfill is a denial of service against every
#: other user of the forge, including the capture path this system depends on.
#:
#: What it bounds is new work, not reads — see :class:`_Budget`. The forge reads a
#: run performs are bounded by the page ceilings instead, which is a weaker bound
#: than this one was and is stated as such rather than papered over.
MAX_OBJECTS_PER_RUN = 2_000

#: The key the collectors write the delivery under. Named here because the stamp
#: removes it, and a literal in the middle of a comprehension would be a place where
#: a reader has to go looking for which key is meant.
DELIVERY_ID_KEY = "delivery_id"

#: Reported rather than absorbed when a walk reaches the page ceiling. A list that
#: quietly stopped short reads as a complete list, and a review whose last thousand
#: comments were never read is otherwise indistinguishable from one that had none.
_PAGE_CEILING_REASON = (
    f"the read stopped at the {MAX_PAGES_PER_OBJECT * PAGE_SIZE}-object page ceiling"
)


class UnanchorableReadError(ValueError):
    """A record whose read anchor is missing, so nothing is written for it.

    Distinct from a validation failure because the runner reports it as a gap in
    history rather than as an error in the run: the object was read, and kojutsu
    cannot honestly reconstruct it, and both of those are facts the operator needs
    rather than a crash.
    """


@dataclass(frozen=True)
class BackfillPlan:
    """What one run is allowed to read. Every field is a bound, never a default."""

    #: In the operator's spelling, which is what the records carry; the semantic
    #: ids case-fold it, so two spellings of one repository cannot become two
    #: identities.
    repositories: tuple[str, ...]
    #: Inclusive floor, in UTC. A pull request is enumerated until the oldest
    #: update on a page predates it, and each review and comment is admitted only
    #: if its own timestamp is at or after it.
    since: datetime
    #: Objects this run may store, counted across every repository. A bound on new
    #: work: objects the store already holds, and objects that turn out to have
    #: nothing to capture, are not charged against it.
    max_objects: int
    #: Inclusive ceiling, in UTC, or ``None`` for "up to now". Optional because an
    #: unbounded top is a real answer; not defaulted because an unbounded top on a
    #: busy repository is the question the operator did not ask. Carried at the last
    #: instant of the named day, since ``--until 2026-09-30`` means the whole of
    #: the 30th rather than its midnight.
    until: datetime | None = None
    #: Specific pull requests to reconstruct, or ``None`` for the listing walk.
    #:
    #: When set, the run reads only these PRs' own pages and no listing pages
    #: at all: two to four requests per PR (reviews, issue comments, plus inline
    #: comments per review) instead of one page per hundred PRs updated since the
    #: window began. The window still applies -- reviews and comments outside it
    #: are skipped, and pulls outside it are skipped too -- and it is reported,
    #: never widened by this bound. Empty (a PR with no reviews) is an empty
    #: range, not an error.
    pr_numbers: tuple[int, ...] | None = None
    #: Association values this run treats as authorised, or ``None`` for the default,
    #: which is no restriction at all. Carried on the plan rather than read from global
    #: state so that a run states in one place what it was permitted to record.
    #:
    #: **Narrowing this is a decision about who gets a voice in the corpus.** The gate
    #: decides whether an account with no standing in the repository can cause records
    #: to be written, and on a public project the accounts doing most of the reviewing
    #: are frequently outside contributors and review bots. That used to argue for a
    #: narrow default; it argues the other way, because a bot is by definition not a
    #: member or collaborator of anything, so ``author_association`` sorts automation
    #: into the very value a narrow set excludes. Measured, on t3code PR #2829: the
    #: old default refused 28 human comments to admit 21 bot ones. The default is now
    #: unrestricted and automation is recorded on the record instead -- see
    #: :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` before changing this
    #: back.
    authorized_associations: frozenset[str] | None = None


def build_plan(
    *,
    settings: Settings,
    repositories: Sequence[str] | None,
    since: str,
    max_objects: int,
    until: str | None = None,
    authorized_associations: frozenset[str] | None = None,
    pr_numbers: Sequence[int | str] | None = None,
) -> BackfillPlan:
    """Validate the bounds and refuse anything that would read without limit.

    ``--repo``, ``--since`` and ``--max-objects`` are all required, and there is no
    default for any of them. That is the decision rather than an omission: each
    default would be a value chosen by this code about somebody else's repository,
    and a backfill is the one operation where "everything" is never the right
    default — history is unbounded, the token is per-repository, and the bill for a
    runaway run is paid by everyone else using the forge.

    ``--until`` is the exception, and it is a bound rather than a limit: unbounded
    above means "up to now", which is a range an operator can reasonably mean. It
    is validated with the same severity as the others anyway, because an inverted
    range matches nothing and an inverted range that matches nothing reads exactly
    like an empty repository.
    """
    requested = [str(item).strip() for item in (repositories or []) if str(item).strip()]
    if not requested:
        raise ValueError(
            "Name at least one repository with --repo. A backfill with no repository would "
            "read every repository the token can reach."
        )

    selected: list[str] = []
    seen: set[str] = set()
    for repository in requested:
        if not allowlist.is_valid_repository(repository):
            raise ValueError(f"Repository {repository!r} is not a valid owner/name value.")
        # The same allowlist the webhook authorises against, read through the same
        # definition. A backfill reaches into history rather than reacting to a
        # delivery, which makes it the more dangerous of the two: a live event has
        # to arrive to be captured, while a backfill goes looking.
        if not allowlist.repository_allowed(repository, settings):
            raise ValueError(
                f"Repository {repository!r} is not in GITHUB_WEBHOOK_ALLOWED_REPOSITORIES; "
                "a backfill reads only what capture is allowed to write."
            )
        folded = repository.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        selected.append(repository)

    if not isinstance(max_objects, int) or isinstance(max_objects, bool) or max_objects < 1:
        raise ValueError("--max-objects must be at least 1; a run that reads nothing is not a run.")
    if max_objects > MAX_OBJECTS_PER_RUN:
        raise ValueError(
            f"--max-objects is capped at {MAX_OBJECTS_PER_RUN}; the cap exists because the "
            "token and the rate limit are shared with every other user of the forge."
        )

    floor = _parse_bound(since, "since")
    if floor.date() > datetime.now(UTC).date():
        raise ValueError(
            f"--since {floor.strftime(DATE_BOUND_FORMAT)} is in the future. A floor past today "
            "reconstructs nothing, and a run that reports success while reading nothing is the "
            "failure this command exists to avoid."
        )

    ceiling: datetime | None = None
    ceiling_text = str(until or "").strip()
    if ceiling_text:
        parsed = _parse_bound(ceiling_text, "until")
        if floor.date() > parsed.date():
            raise ValueError(
                f"--since {floor.strftime(DATE_BOUND_FORMAT)} is after --until "
                f"{parsed.strftime(DATE_BOUND_FORMAT)}: the range is inverted, and an inverted "
                "range matches nothing, which is indistinguishable from an empty one unless it "
                "is refused. Pass the earlier date as --since."
            )
        # The last instant of the named day, not its midnight: ``--until
        # 2026-09-30`` is the whole of the 30th, the same inclusive reading the
        # sibling ``backfill`` gives the same spelling.
        ceiling = parsed.replace(hour=23, minute=59, second=59, microsecond=999_999)

    parsed_prs: tuple[int, ...] | None = None
    if pr_numbers is not None:
        seen_prs: set[int] = set()
        ordered: list[int] = []
        for raw in pr_numbers:
            try:
                number = int(str(raw).strip())
            except (TypeError, ValueError):
                raise ValueError(f"--pr must name pull request numbers, got {raw!r}.") from None
            if isinstance(raw, bool) or number < 1:
                raise ValueError(f"--pr must name pull request numbers, got {raw!r}.")
            if number not in seen_prs:
                seen_prs.add(number)
                ordered.append(number)
        if not ordered:
            raise ValueError("--pr names no pull request; omit it to walk the listing.")
        parsed_prs = tuple(ordered)

    return BackfillPlan(
        repositories=tuple(selected),
        since=floor,
        max_objects=max_objects,
        until=ceiling,
        authorized_associations=authorized_associations,
        pr_numbers=parsed_prs,
    )


def _parse_bound(text: str, field_name: str) -> datetime:
    """One end of the range as a UTC midnight, or a refusal naming which end failed.

    Both ends are parsed the same way and refused the same way. An end that does
    not parse is not a looser range — it is a command that will read whatever it is
    given while reporting a range the operator did not name — and naming the field
    is what tells them which half of what they typed was not a date.
    """
    try:
        return datetime.strptime(text, DATE_BOUND_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        if field_name == "since":
            raise ValueError(
                f"--since must be a date as YYYY-MM-DD, got {text!r}. It is required: kojutsu "
                "will not choose a floor for you, because the floor is an editorial decision "
                "about which era of history is worth reconstructing."
            ) from None
        raise ValueError(
            f"--until must be a date as YYYY-MM-DD, got {text!r}. Both ends of a range are "
            "dates, not timestamps: a ceiling of yesterday is a range, and a ceiling that is "
            "not a date at all is a command that will read up to whatever it was given."
        ) from None


class BackfillReadSink:
    """Restamp every record a collector writes as evidence of a read.

    The collectors cannot do this themselves: they were written for signed provider
    deliveries and set ``capture_source`` and ``capture_delivery_id`` from the
    delivery id they were handed, so ``delivery_id=None`` leaves a review claiming
    a delivery that never existed. Rather than a second writer, this wraps the one
    sink and rewrites the provenance of what passes through it.

    Three things are changed and nothing else. The source becomes
    :data:`~kojutsu.models.CaptureSource.BACKFILLED`, because the record was
    reconstructed by a read rather than witnessed. ``captured_at`` becomes the read
    time, which for this source is the only evidence there is and is what the
    anchor rule asks for. The delivery id is *removed* rather than nulled, because
    the storage layer writes any value it is given and a placeholder there would
    read as a delivery id to anyone filtering on it.

    A record with no read anchor is refused. The rule in
    :func:`~kojutsu.models.capture_anchor_gaps` is asked first so the failure
    can name what is missing, and the model is still asked afterwards because the
    model is the authority: this class is not a place where that rule gets
    relaxed, and a future reader who wants to relax it will find the refusal here
    rather than only in a model docstring.
    """

    def __init__(
        self,
        inner: KnowledgeSink,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._inner = inner
        self._clock = clock or (lambda: datetime.now(UTC))

    def store(self, entry: StorableRecord) -> KnowledgeDeliveryOutcome | None:
        """Stamp a captured entry, and pass anything else through untouched.

        Only entries are restamped. A census record could not arrive here — its
        anchor is a delivery, and a backfill has none — so there is nothing to
        stamp and no observation to write: a backfill's own silence is reported in
        the run's counts, because a record asserting "we looked and kept nothing"
        needs a delivery behind it and there is not one.
        """
        if isinstance(entry, KnowledgeEntry):
            entry = self._stamped(entry)
        return self._inner.store(entry)

    def _stamped(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        read_at = self._clock()
        metadata = {key: value for key, value in entry.metadata.items() if key != DELIVERY_ID_KEY}
        gaps = capture_anchor_gaps(
            capture_source=CaptureSource.BACKFILLED,
            repo=metadata.get("repo"),
            pr_number=metadata.get("pr_number"),
            captured_at=read_at,
            delivery_id=None,
            comment_id=metadata.get("github_comment_id"),
            check_id=metadata.get("check_id"),
            review_id=metadata.get(REVIEW_ID_KEY),
        )
        if gaps:
            raise UnanchorableReadError(
                f"record {entry.entry_id} cannot be stored as backfilled; missing: "
                + ", ".join(gaps)
            )
        # Rebuilt rather than copied. ``model_copy`` skips validation, and the
        # validation here is the entire point: a copy would let an unanchored
        # record through with a backfilled label on it and no complaint from
        # anything.
        return KnowledgeEntry(
            entry_id=entry.entry_id,
            session_id=entry.session_id,
            question_text=entry.question_text,
            answer_text=entry.answer_text,
            category=entry.category,
            context=entry.context,
            author=entry.author,
            answered_at=entry.answered_at,
            embedding=entry.embedding,
            tags=entry.tags,
            metadata=metadata,
            capture_source=CaptureSource.BACKFILLED,
            captured_at=read_at,
            capture_delivery_id=None,
        )


@dataclass(frozen=True)
class BackfillGap:
    """One object the run could not reconstruct, and why.

    The reason is kept because the count alone is not actionable: a 404 means the
    object is gone from the forge, a missing anchor means kojutsu will not
    reconstruct it, and only one of those is worth an operator's attention.
    """

    repository: str
    pr_number: int | None
    what: str
    reason: str


@dataclass(frozen=True)
class BackfillReport:
    """What one run did, and what it could not do.

    The counts are deliberately not four. ``unreadable`` is the one that matters
    most — it is the part of history that was never reconstructed and never will be
    — but a run also reads objects that legitimately had nothing to say: a review
    from an unauthorised association, a comment with no marker. Counting those as
    either written or unreadable would make a number lie, and counting them nowhere
    would make ``objects_read`` not add up to anything.

    ``objects_new`` is the count ``--max-objects`` was charged against, and it is
    not ``records_written``: one review can store a verdict and its inline comments
    and still be one object. It is reported separately because a run that stored
    nothing is the normal outcome of a re-run, and "stored nothing" and "stored
    nothing because the budget ran out" are different results that used to be
    indistinguishable from each other and from success.
    """

    plan: BackfillPlan
    objects_read: int = 0
    records_written: int = 0
    already_present: int = 0
    unreadable: int = 0
    silent: int = 0
    objects_new: int = 0
    budget_exhausted: bool = False
    #: Pull requests whose review/comment reads were skipped because a finished
    #: pass already covers them under this window (see
    #: :func:`_finished_and_unchanged`). Not read, not charged, not missing --
    #: the listing that proved them unchanged is the cheap part of the run, and
    #: this is the expensive part it avoided.
    skipped_finished: int = 0
    #: Objects read that stored nothing, grouped by the gate that refused them.
    #:
    #: This is the counter ``silent`` was, split by cause. The two have opposite
    #: implications — a review with nothing capturable is an absence of review, while
    #: a review refused by a policy is review activity the corpus is discarding — and
    #: one number cannot say which an operator is looking at. Empty for a run that
    #: stored or already held everything, which is the common case for a re-run.
    refusals: Mapping[str, int] = field(default_factory=dict)
    #: The enumeration ran out before it reached ``plan.since``, so the range was
    #: not covered to the floor the operator named. Not a gap: nothing was read and
    #: failed to reconstruct, and whether anything older exists is a question this
    #: walk cannot answer. False when the budget or the page ceiling ended it, where
    #: the answer is merely unknown rather than known to be no.
    floor_unreached: bool = False
    gaps: tuple[BackfillGap, ...] = field(default_factory=tuple)


class _ReadBudgetExhaustedError(Exception):
    """The run's budget for new work ran out. Not an error: the range is truncated.

    Not raised for a run that read a great deal and stored nothing, which is what a
    re-run over an already-stored range does, and is a normal outcome rather than a
    truncation.
    """


class _Budget:
    """The one place new work is charged, so the bound cannot be bypassed.

    **Charged for storing, not for reading.** An object the store already holds
    costs nothing, and neither does one that turns out to have nothing to capture,
    because neither is work this run did that the last one did not. Charged per
    read instead, the walk could not leave the first page of a busy repository: the
    same objects came back, the same budget went on them, and a re-run of the
    range the output recommended made no progress at all.

    The reservation is taken before the object is examined and released after,
    because whether an object is new is only knowable once a collector has seen it,
    and there is no probe that can answer it earlier. Taking it up front is what
    makes the bound exact rather than approximate: a run stores at most ``limit``
    new objects, and stops at the first object it examines once they are spent,
    whether that object turns out to be new or not.
    """

    def __init__(self, limit: int) -> None:
        self._remaining = limit

    def reserve(self) -> None:
        if self._remaining < 1:
            raise _ReadBudgetExhaustedError
        self._remaining -= 1

    def refund(self) -> None:
        """Give back a reservation for an object that stored nothing."""
        self._remaining += 1


class _Tally:
    """Mutable counters for the duration of one run."""

    def __init__(self) -> None:
        self.objects_read = 0
        self.records_written = 0
        self.already_present = 0
        self.silent = 0
        self.objects_new = 0
        self.skipped_finished = 0
        self.floor_unreached = False
        self.refusals: dict[str, int] = {}
        self.gaps: list[BackfillGap] = []

    def gap(self, repository: str, pr_number: int | None, what: str, reason: str) -> None:
        self.gaps.append(
            BackfillGap(repository=repository, pr_number=pr_number, what=what, reason=reason)
        )


def _normalise(timestamp: datetime) -> datetime:
    """UTC, whatever the forge called it. A naive time is read as UTC rather than
    as the reader's own zone, which would move objects across a bound."""
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)


def _before_floor(timestamp: datetime, plan: BackfillPlan) -> bool:
    """Whether a dated object predates the floor, which ends the walk."""
    return _normalise(timestamp) < plan.since


def _above_ceiling(timestamp: datetime, plan: BackfillPlan) -> bool:
    """Whether a dated object postdates the ceiling, which only skips it.

    Not the same shape as the floor, and the asymmetry is the enumeration's. The
    pages get *older* as the walk proceeds, so one object below the floor proves
    every later one is below it and ends the walk — while one object above the
    ceiling proves only that this page is too recent, and the window may be any
    number of pages further back. Ending the walk on it would report an empty range
    for every window in the past, which on a busy repository is most of them.
    """
    return plan.until is not None and _normalise(timestamp) > plan.until


def _within_window(timestamp: datetime | None, plan: BackfillPlan) -> bool:
    """Whether an object may be reconstructed inside this run's range.

    An object with no timestamp cannot be placed against a bound, and it is
    therefore not admitted. The alternative is to assume it is recent, which is a
    guess about which side of an editorial boundary a piece of history belongs on
    — and a guess that is invisible once written, because a record in the corpus
    cannot be told apart from one the policy actually admitted.

    The ceiling is compared rather than skipped, so a range bounded at both ends
    admits nothing from outside it. A review submitted after ``--until`` is outside
    the era the operator named even though its change sits inside it, and the era
    was the point.
    """
    if timestamp is None:
        return False
    return not (_before_floor(timestamp, plan) or _above_ceiling(timestamp, plan))


def _finished_and_unchanged(
    registry: QuestionRegistry,
    *,
    repository: str,
    pr_number: int,
    updated_at: datetime | None,
    plan: BackfillPlan,
) -> bool:
    """Whether a finished pass already covers this pull request under this window.

    True only for a receipt this code wrote: the pass finished the pull request,
    ran under the same window (floor, ceiling, admitted associations), and saw a
    listing timestamp at least as new as this listing's. The review and comment
    reads are then skipped entirely -- nothing on the pull request can have
    produced a storable object this run has not already seen, because any new
    review, comment, or push moves the listing timestamp and fails the recency
    check.

    Everything else reads: no receipt (including the pull request a stopped run
    was inside, which by construction has none), an unfinished pass, a window
    the receipt was not written under, a pull request with no listing timestamp
    to place, or a receipt value that does not parse. A receipt that cannot be
    understood must never skip work -- failing open spends reads, failing closed
    drops history.
    """
    if pr_number <= 0 or updated_at is None:
        return False
    try:
        receipt = registry.get_backfill_receipt(repo=repository, pr_number=pr_number)
    except Exception:
        return False
    if not receipt or not receipt.get("finished"):
        return False
    if receipt.get("window_since") != plan.since.isoformat():
        return False
    expected_until = plan.until.isoformat() if plan.until is not None else None
    if receipt.get("window_until") != expected_until:
        return False
    if receipt.get("associations") != _associations_key(plan.authorized_associations):
        return False
    try:
        observed = datetime.fromisoformat(str(receipt["observed_updated_at"]))
    except (TypeError, ValueError):
        return False
    return _normalise(observed) >= _normalise(updated_at)


def _associations_key(authorized_associations: frozenset[str] | None) -> str | None:
    """The receipt form of the admission policy: sorted CSV, or ``None`` for all."""
    if authorized_associations is None:
        return None
    return ",".join(sorted(authorized_associations))


def _pages(
    fetch: Callable[[int], Sequence[Any]], *, truncated: Callable[[], None]
) -> Iterator[Any]:
    """Walk one object's pages, bounded, and say when the bound was reached.

    The ceiling is inherited from the rest of the client rather than invented here.
    Reaching it is a fact about the run, though, and a fact nobody reported is a
    list that quietly stopped short — so the caller is handed a callback rather than
    left to infer truncation from a count it cannot reconcile.
    """
    for page in range(1, MAX_PAGES_PER_OBJECT + 1):
        batch = list(fetch(page))
        if not batch:
            return
        yield from batch
        if len(batch) < PAGE_SIZE:
            return
    truncated()


class _PageReads:
    """Bounded concurrent page reads for one run, handed back in the order asked.

    **Why the listing is the only thing read this way.** Profiling a
    ``backfill-reviews`` run over ``pingdotgg/t3code`` found 88 of its 96 requests and
    114.2 of its 118.7 seconds were the pull request listing: pages of 100 changes,
    each ~1.3s, each carrying no dependency on the page beside it. That is the entire
    cost of the command and it was being paid serially.

    The per-object reads are deliberately **left alone**, and the reason is that
    speculation would make them worse rather than faster. :func:`_pages` walks a
    review's inline comments or one change's conversation, and those lists are almost
    always empty or a single item -- so a speculative batch of four would turn one
    request into four, a ~4x increase in quota spent, to save latency on a read that
    was never the problem. The listing's pages are reliably full-length and the
    per-object pages are reliably not, and those two facts point in opposite
    directions. Concurrency here is for the walk that has pages to spare, not for the
    reads that do not.

    **Threads rather than an async client.** This is IO-bound, so the wall-clock result
    is the same either way, and threads are what leaves the ten synchronous call sites
    in ``integrations/github.py`` and every sync ``MockTransport`` test untouched. An
    ``httpx.AsyncClient`` would have required the whole seam to become async to make
    one walk faster.

    **The bound is enforced by a semaphore, and the pool is sized to match it.** They
    are two objects doing one job, and the redundancy is deliberate: if the pool were
    wider than the semaphore it would queue rather than wait on it, and if it were
    narrower the effective concurrency would be silently below the number this class
    documents. Sizing them together makes the documented bound the real one.

    **In-flight reads never exceed the bound, by construction rather than by argument.**
    The calling thread blocks inside :meth:`batch` for the whole of a speculative
    round and issues nothing of its own while it does, so the round's width is a total,
    not a component.

    **The speculation window ramps: 1, 2, 4, 8, 8, ... capped at the bound.** Round
    one is a single page, which is exactly what the sequential walk fetched, so a run
    whose floor is on the first page reads one page -- no threads, no pool, no wasted
    request. The width doubles per round, so a deep walk is at the bound from the
    fourth round on.

    **The waste this buys is at most ``bound - 1`` pages per repository per run**, and
    that is the whole price. A run that stops on the first entry of the round it
    speculated has fetched ``width - 1`` pages it would not otherwise have read; every
    earlier round ended because its last page was full, so it speculated nothing it did
    not use. With the default bound of 8 that is **at most 7 pages -- 700 pull requests
    -- fetched and discarded for a walk of any length**, against a walk that costs
    ~1.3s per 100. Measured on the 88-page walk that motivated all of this: 95 pages
    fetched, 7 discarded -- 8% more requests for a wall clock of 22.9s against 114.4s,
    with every report counter identical.

    **A failure inside a round is captured, not raised by the worker.** It is
    re-raised by the caller, in page order, so it is indistinguishable from the
    sequential walk having failed at the first page that failed -- which matters,
    because the first failing page is the one that decides what the run reports.
    """

    def __init__(self, *, concurrency: int) -> None:
        self._bound = max(1, concurrency)
        self._slots = threading.BoundedSemaphore(self._bound)
        self._pool: ThreadPoolExecutor | None = None
        self._window = 1

    @property
    def bound(self) -> int:
        """The most reads this run will have in flight at once."""
        return self._bound

    def next_window(self) -> int:
        """How many pages to fetch at once this round, doubling up to the bound.

        Read once per round, and the doubling is the ramp described on the class. It
        is capped at the bound so a lowered ``read_concurrency`` lowers the speculation
        with it rather than queueing pages behind a semaphore it cannot use.
        """
        width = min(self._window, self._bound)
        self._window = min(self._window * 2, self._bound)
        return width

    def batch(
        self,
        fetch: Callable[[int], Sequence[Any]],
        pages: Sequence[int],
    ) -> list[Sequence[Any] | Exception]:
        """Read every page in ``pages``, concurrently, and return them in that order.

        The return order is the point. Results are indexed by the page they were asked
        for, never by the order they happened to finish, because the walk consumes them
        in listing order and the forge's ``sort=updated`` guarantee only holds if the
        walk respects it.
        """
        if len(pages) == 1:
            # One page is what the sequential walk would have fetched, so this reads in
            # the caller's thread with no pool and no thread created at all.
            return [self._read(fetch, pages[0])]

        def read(page: int) -> tuple[int, Sequence[Any] | Exception]:
            return page, self._read(fetch, page)

        # Not ``with pool`` -- that would shut the executor down on the way out and the
        # next round would find it closed. The pool's lifetime is this run's, so it is
        # shut down by :meth:`close` and not per round.
        return [outcome for _page, outcome in self._ensure_pool().map(read, pages)]

    def _read(self, fetch: Callable[[int], Sequence[Any]], page: int) -> Sequence[Any] | Exception:
        """One page, holding a slot. A failure is returned rather than raised.

        Only ``Exception``: a ``KeyboardInterrupt`` or ``SystemExit`` raised inside a
        worker thread is not a failed page read, and swallowing one into a return value
        would turn an operator's interrupt into a reported fetch error.
        """
        try:
            with self._slots:
                return fetch(page)
        except Exception as exc:  # re-raised by the caller, in page order
            return exc

    def _ensure_pool(self) -> ThreadPoolExecutor:
        """The worker pool, built on the first round that needs more than one page."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._bound,
                thread_name_prefix="kojutsu-history",
            )
        return self._pool

    def close(self) -> None:
        """Stop the pool. Safe when no round ever needed one."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)


def _single_pull_requests(
    reader: HistoryReader,
    plan: BackfillPlan,
    tally: _Tally,
    repository: str,
    registry: QuestionRegistry,
) -> Iterator[Any]:
    """Yield the bound pull requests, reading no listing pages at all.

    Each number costs one ``get_pull_request`` read. The window still applies:
    a pull above the ceiling or below the floor is skipped, never widening the
    range the operator named -- and a PR the forge no longer has is reported
    as a gap, which is what makes a mistyped number an empty range rather than
    an error. ``floor_unreached`` is left alone: the listing never ran, so it
    cannot have run out.
    """
    pr_numbers = plan.pr_numbers
    if not pr_numbers:
        return
    owner, name = split_repo(repository)
    get_one = getattr(reader, "get_pull_request", None)
    for pr_number in pr_numbers:
        try:
            pull = get_one(owner, name, pr_number) if callable(get_one) else None
        except Exception as exc:
            if is_missing_object(exc):
                tally.gap(
                    repository,
                    pr_number,
                    "pull request",
                    f"the forge no longer has it ({type(exc).__name__})",
                )
                continue
            raise
        if pull is None:
            tally.gap(
                repository,
                pr_number,
                "pull request",
                "the forge has no such pull request",
            )
            continue
        updated_at = getattr(pull, "updated_at", None)
        if updated_at is not None and (
            _before_floor(updated_at, plan) or _above_ceiling(updated_at, plan)
        ):
            continue
        if _finished_and_unchanged(
            registry,
            repository=repository,
            pr_number=pr_number,
            updated_at=updated_at,
            plan=plan,
        ):
            # A finished pass under this window already saw everything this
            # pull request can store, and the listing timestamp proves nothing
            # moved since. The listing read that proved it is the cheap part;
            # the review and comment reads it avoids are not.
            tally.skipped_finished += 1
            continue
        tally.objects_read += 1
        yield pull


def _pull_requests(
    reader: HistoryReader,
    owner: str,
    name: str,
    plan: BackfillPlan,
    tally: _Tally,
    repository: str,
    registry: QuestionRegistry,
    *,
    reads: _PageReads,
) -> Iterator[Any]:
    """Yield the pull requests this run will look at, most recently updated first.

    Descending update order is what makes the floor cheap and complete at the same
    time: walking forward the pages get older, so the first entry whose update
    predates the floor proves every later one does too, and the walk stops there.
    It is also what admits a change opened long before the floor and reviewed after
    it — a ``created`` ordering would skip it entirely, and that is precisely the
    pull request a backfill exists to find.

    The ordering lives in :meth:`HistoryReader.list_pull_requests`, and it is not a
    default: ascending here would stop the walk on this repository's oldest pull
    request and report zero objects read, which reads as an empty repository rather
    than as a broken enumeration.

    The ceiling is where the two bounds part company, and the reason is the
    enumeration rather than the policy. Descending order means one object below
    the floor proves every later one is below it, so the floor ends the walk; one
    object above the ceiling proves only that this page is too recent, so the
    ceiling skips it and the walk continues to the next page, which is older. A
    range bounded at both ends therefore reads exactly the changes inside it, and
    a window entirely in the past is reached by walking back to it rather than
    being reported as an empty repository.

    **Pages are fetched ahead and consumed in listing order, and the second half is
    the requirement rather than the first.** :class:`_PageReads` hands back a whole
    speculative round; this function then walks it in page order and applies every
    short-circuit it always did -- the floor, the ceiling, the short page, the page
    ceiling -- at exactly the points it always applied them. Concurrency moves *when
    the bytes arrive*, never *which objects are admitted or in what order*.

    That is why the budget still truncates at the same place. ``--max-objects`` charges
    new work and a re-run is relied upon to *advance*, so a budget that trips must
    leave the same prefix captured and not an arbitrary subset: the budget is raised
    from this generator's own frame, on the caller's thread, while it yields -- so the
    prefix is the sequential prefix by construction. Fetching pages 4..7 before page 2
    has been examined changes what the run *read*; it cannot change what it *captured*.

    Nothing is charged here. A pull request is a container: it stores nothing
    itself, and its reviews and comments are each charged on their own account.
    Charging it as well would count the same work twice, which is the old bound's
    arithmetic rather than this one's.
    **Whether the floor was ever reached is reported, because the enumeration can
    end without it.** The forge will not list a repository's history for ever: on
    ``pingdotgg/t3code`` the pull request listing stops after about a thousand
    changes, which reaches back to early September and no further, so a run asked
    for a July floor walks to the end of what it was given and stops — having read
    nothing from July, and with no way to tell from inside the walk whether the
    repository has nothing older or the forge declined to say. An enumeration that
    ends above the floor is therefore named, because a range silently missing its
    own floor is the same defect as a range silently cut short by the budget.
    """
    if plan.pr_numbers is not None:
        yield from _single_pull_requests(reader, plan, tally, repository, registry)
        return
    next_page = 1
    while next_page <= MAX_PAGES_PER_OBJECT:
        window = reads.next_window()
        pages = tuple(range(next_page, min(next_page + window, MAX_PAGES_PER_OBJECT + 1)))
        outcomes = reads.batch(
            lambda page: reader.list_pull_requests(owner, name, page=page, per_page=PAGE_SIZE),
            pages,
        )
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                # Raised here, on the caller's thread, and in page order: a sequential
                # walk would have raised at the first page that failed, and the run's
                # own error reporting is written against that. Letting the worker's
                # exception escape the pool instead would make a failure on page 7 look
                # like a failure on page 1 whenever the rounds interleave.
                raise outcome
            batch = list(outcome)
            if not batch:
                tally.floor_unreached = True
                return
            for pull in batch:
                updated_at = getattr(pull, "updated_at", None)
                if updated_at is not None:
                    if _before_floor(updated_at, plan):
                        # Every later entry is older than this one, so the rest of the
                        # walk proves itself unnecessary.
                        return
                    if _above_ceiling(updated_at, plan):
                        # Too recent for the era named. This page is over, the next one
                        # is older, and the window may be on it — so this entry is
                        # skipped rather than the walk ended. A pull request with no
                        # update time cannot be placed against a bound at all and is
                        # walked rather than guessed at.
                        continue
                pr_number = int(getattr(pull, "number", 0) or 0)
                if _finished_and_unchanged(
                    registry,
                    repository=repository,
                    pr_number=pr_number,
                    updated_at=updated_at,
                    plan=plan,
                ):
                    tally.skipped_finished += 1
                    continue
                tally.objects_read += 1
                yield pull
            if len(batch) < PAGE_SIZE:
                tally.floor_unreached = True
                return
            next_page += 1
    tally.gap(repository, None, "pull requests", _PAGE_CEILING_REASON)


def _review_comments(
    reader: HistoryReader,
    owner: str,
    name: str,
    pr_number: int,
    review_id: int,
    *,
    truncated: Callable[[], None],
) -> list[Any]:
    return list(
        _pages(
            lambda page: reader.list_review_comments(
                owner, name, pr_number, review_id, page=page, per_page=PAGE_SIZE
            ),
            truncated=truncated,
        )
    )


def run_backfill(
    *,
    reader: HistoryReader,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    plan: BackfillPlan,
    clock: Callable[[], datetime] | None = None,
    read_concurrency: int | None = None,
) -> BackfillReport:
    """Reconstruct the history this plan allows, and report what it could not.

    Idempotent by construction, not by bookkeeping: every object is handed to a
    collector whose claim is keyed on the forge's own identity, so an object seen
    twice collides and writes nothing. That is what makes an interrupted run
    resumable, and it is why there is no cursor to drift, be lost, or disagree
    with the store about what it has already seen.

    And resumable is not the same as advancing, which is why the budget is charged
    for storing rather than for reading: a run over a range it has already walked
    reads it, stores nothing, and is not truncated by having done so.

    ``read_concurrency`` is the most forge reads this run will have in flight at
    once, ``None`` meaning :data:`~kojutsu.integrations.github.HISTORY_READ_CONCURRENCY`.
    It is a parameter rather than a global for the same reason ``search_pace_seconds``
    is: the behaviour under test is the bound itself, and a test that cannot set the
    bound to 1 is a test that cannot show concurrency changed nothing. **It bounds
    concurrency and not the request rate**, which is the trade the constant's own
    comment states in full -- an operator on a shared token who does not want the rate
    this implies sets it to 1 and gets the strictly sequential walk.
    """
    tally = _Tally()
    budget = _Budget(plan.max_objects)
    reads = _PageReads(
        concurrency=HISTORY_READ_CONCURRENCY if read_concurrency is None else read_concurrency
    )
    # Stamped here rather than by the caller: a backfill that could be pointed at
    # an unstamped sink would be one path that writes records claiming a signed
    # delivery that never happened.
    read_sink = BackfillReadSink(sink, clock=clock)
    budget_exhausted = False

    try:
        for repository in plan.repositories:
            owner, name = split_repo(repository)
            for pull in _pull_requests(
                reader, owner, name, plan, tally, repository, registry, reads=reads
            ):
                _reconstruct_pull_request(
                    reader=reader,
                    registry=registry,
                    sink=read_sink,
                    plan=plan,
                    repository=repository,
                    owner=owner,
                    name=name,
                    pull=pull,
                    budget=budget,
                    tally=tally,
                )
    except _ReadBudgetExhaustedError:
        # Not a failure, and not the run finishing. It did all the storing it was
        # allowed to do and stopped there, so the same range can be re-run at any
        # time and the re-run advances: what the store already holds is charged
        # nothing, so the budget is spent on new objects instead of on the ones
        # behind them. What a re-run costs is reads, and the summary says so.
        budget_exhausted = True
    finally:
        # Shut the worker pool even when the budget abandoned the listing generator
        # mid-round. ``_pull_requests`` is a generator, so the pages it was speculating
        # about are still in flight when the exception unwinds past it; without this
        # the run would return while threads were still writing to its reader.
        reads.close()

    return BackfillReport(
        plan=plan,
        objects_read=tally.objects_read,
        records_written=tally.records_written,
        already_present=tally.already_present,
        unreadable=len(tally.gaps),
        silent=tally.silent,
        objects_new=tally.objects_new,
        budget_exhausted=budget_exhausted,
        skipped_finished=tally.skipped_finished,
        floor_unreached=tally.floor_unreached,
        refusals=dict(sorted(tally.refusals.items(), key=lambda item: -item[1])),
        gaps=tuple(tally.gaps),
    )


def _reconstruct_pull_request(
    *,
    reader: HistoryReader,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    plan: BackfillPlan,
    repository: str,
    owner: str,
    name: str,
    pull: Any,
    budget: _Budget,
    tally: _Tally,
) -> None:
    pr_number = int(getattr(pull, "number", 0) or 0)
    if pr_number <= 0:
        tally.gap(
            repository,
            pr_number or None,
            "pull request",
            "GitHub reported no usable pull request number",
        )
        return

    # Whether every page this pull request needed was actually walked. A pass
    # that stops at the page ceiling read a prefix, not the pull request, and
    # recording it as finished would let the next run skip the unread tail --
    # so the flag starts true and any truncation clears it. An exception (a
    # missing object, an unreadable review, the budget running out) leaves
    # through the frame before the receipt below is written, which is what
    # makes the pull request a run stops inside re-readable: it has no
    # finished receipt, so the next run reads it and stores only what is
    # missing.
    finished = True

    def _mark_unfinished() -> None:
        nonlocal finished
        finished = False

    def _gap_unfinished(what: str) -> None:
        tally.gap(repository, pr_number, what, _PAGE_CEILING_REASON)
        _mark_unfinished()

    for review in _pages(
        lambda page: reader.list_reviews(owner, name, pr_number, page=page, per_page=PAGE_SIZE),
        truncated=lambda: _gap_unfinished("reviews"),
    ):
        submitted_at = getattr(review, "submitted_at", None)
        if not _within_window(submitted_at, plan):
            # Excluded by a bound, so never examined: an object the policy kept out
            # is not one this run read, and counting it would make the count
            # describe work that was not done.
            continue
        budget.reserve()
        _reconstruct_review(
            reader=reader,
            registry=registry,
            sink=sink,
            plan=plan,
            repository=repository,
            owner=owner,
            name=name,
            pr_number=pr_number,
            pull=pull,
            review=review,
            budget=budget,
            tally=tally,
            on_truncated=_mark_unfinished,
        )

    _reconstruct_issue_comments(
        reader=reader,
        registry=registry,
        sink=sink,
        plan=plan,
        repository=repository,
        owner=owner,
        name=name,
        pr_number=pr_number,
        budget=budget,
        tally=tally,
        on_truncated=_mark_unfinished,
    )

    # The pass end, in the pass's own terms: this pull request was walked to
    # its last page under this window, so a later run under the same window
    # whose listing timestamp is no newer can skip it. Stored comment
    # timestamps are deliberately not the evidence -- an empty-bodied review is
    # read and kept as nothing, and a pass can stop after one comment and
    # before another, so the oldest and newest stored comments do not prove
    # the middle was read. Only reaching this line proves it.
    observed_at = getattr(pull, "updated_at", None)
    if observed_at is not None:
        registry.record_backfill_receipt(
            repo=repository,
            pr_number=pr_number,
            observed_updated_at=_normalise(observed_at),
            window_since=plan.since,
            window_until=plan.until,
            authorized_associations=plan.authorized_associations,
            finished=finished,
        )


def _reconstruct_review(
    *,
    reader: HistoryReader,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    plan: BackfillPlan,
    repository: str,
    owner: str,
    name: str,
    pr_number: int,
    pull: Any,
    review: Any,
    budget: _Budget,
    tally: _Tally,
    on_truncated: Callable[[], None] | None = None,
) -> None:
    review_id = int(getattr(review, "id", 0) or 0)
    user = getattr(review, "user", None)
    login = getattr(user, "login", None)
    what = f"review {review_id}"

    def _comments_truncated() -> None:
        tally.gap(repository, pr_number, what, _PAGE_CEILING_REASON)
        if on_truncated is not None:
            on_truncated()

    # The inline comments are read before the review is handed to a collector,
    # which is the whole of requirement "a deleted object produces nothing". A
    # review deleted after the fact has no comments left to list, so the 404 is how
    # this learns the object is gone — and the verdict, which would still read
    # perfectly well on its own, is not written. A record for a review the forge has
    # withdrawn is not a reconstruction of anything.
    try:
        comments = _review_comments(
            reader,
            owner,
            name,
            pr_number,
            review_id,
            truncated=_comments_truncated,
        )
    except Exception as exc:
        if is_missing_object(exc):
            tally.gap(
                repository,
                pr_number,
                what,
                f"the forge no longer has it ({type(exc).__name__})",
            )
            # Read, and nothing reconstructed: the reservation was for work this
            # object would do, and it is not going to do any.
            budget.refund()
            return
        raise

    tally.objects_read += 1
    pull_author = getattr(getattr(pull, "user", None), "login", None)
    try:
        result = process_review_event_outcome(
            repo=repository,
            pr_number=pr_number,
            review_id=review_id,
            review_state=str(getattr(review, "state", "") or ""),
            review_body=str(getattr(review, "body", "") or ""),
            review_author=login,
            pr_author_account=pull_author,
            review_submitted_at=getattr(review, "submitted_at", None),
            review_author_association=getattr(review, "author_association", None),
            review_author_type=getattr(getattr(review, "user", None), "type", None),
            comments=[
                {
                    "id": getattr(comment, "id", None),
                    "body": getattr(comment, "body", "") or "",
                    "path": getattr(comment, "path", None),
                    "line": getattr(comment, "line", None),
                    "original_line": getattr(comment, "original_line", None),
                    "side": getattr(comment, "side", None),
                    "diff_hunk": getattr(comment, "diff_hunk", None),
                    "commit_id": getattr(comment, "commit_id", None),
                    "in_reply_to_id": getattr(comment, "in_reply_to_id", None),
                }
                for comment in comments
            ],
            registry=registry,
            sink=sink,
            # No delivery: nothing was delivered to a backfill, and the stamp on the
            # sink is what says so on the record.
            delivery_id=None,
            pr_opened_at=getattr(pull, "created_at", None),
            # No head, and no file list. The pull request's current head is not the
            # head this review was written against, and anchoring six-month-old
            # evidence to a commit from this morning is the specific wrongness the
            # anchor field must never carry: a wrong anchor is checkable, so it is
            # worse than a missing one. ``None`` means "not available", which is the
            # truth. The file list is one more authenticated read per change, against
            # a budget that history makes unbounded, so it is absent here for the same
            # reason and says so.
            head_sha=None,
            files_changed=None,
            # What the operator asked this run to admit. ``None`` is the default set,
            # and is passed through as ``None`` rather than resolved here so that the
            # default lives in exactly one place.
            authorized_associations=plan.authorized_associations,
        )
    except UnanchorableReadError as exc:
        # The collector released its claim on the way out, so nothing was written
        # and a re-run will try again. It is reported as a gap rather than raised:
        # the object was read, kojutsu will not reconstruct it, and an operator
        # needs that in the run's output rather than in a traceback.
        tally.gap(repository, pr_number, what, str(exc))
        budget.refund()
        return
    written, already = capture_counts(result)
    tally.records_written += written
    if written:
        tally.objects_new += 1
    if already:
        tally.already_present += 1
    if not written:
        # Read, and stored nothing. Not a gap and not a duplicate — the difference
        # between the two is the only thing that makes the other counts mean
        # anything. It is also not new work, so the reservation goes back: charging a
        # run for reading objects that will never store anything is what stopped this
        # walk advancing at all.
        if not already:
            tally.silent += 1
            # Which gate, because the reason decides what the number means. A review
            # with nothing capturable is an absence of review; one refused by
            # ``author_association`` is review activity the corpus is throwing away,
            # and those two are indistinguishable at this point unless they are told
            # apart. The association gate only fires for a run that narrowed the policy
            # deliberately -- the default admits every association -- so a run that
            # reports it is reporting its own configuration, not a gap in the forge.
            reason = str(getattr(result, "refusal", "") or "") or "unknown"
            tally.refusals[reason] = tally.refusals.get(reason, 0) + 1
        budget.refund()


def _reconstruct_issue_comments(
    *,
    reader: HistoryReader,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    plan: BackfillPlan,
    repository: str,
    owner: str,
    name: str,
    pr_number: int,
    budget: _Budget,
    tally: _Tally,
    on_truncated: Callable[[], None] | None = None,
) -> None:
    """Handle a pull request's conversation, and refuse to invent an answer.

    **An answer requires a question to have been asked, and kojutsu only started
    asking when it was installed.** So the honest result of a backfill over
    pre-deployment comments is that almost none of them are answers: a historical
    comment without a kojutsu marker is somebody talking to a colleague, and
    recording it as an answer would manufacture the question that was never asked
    and pair it with a conclusion that never referred to it.

    The rule is enforced by the collector rather than by a filter written here,
    which is the point: the collector already requires an explicit answer marker
    *and* a registered parent question, so a comment that carries neither cannot
    reach the store no matter which writer calls it.
    """

    def _conversation_truncated() -> None:
        tally.gap(repository, pr_number, "issue comments", _PAGE_CEILING_REASON)
        if on_truncated is not None:
            on_truncated()

    comments = list(
        _pages(
            lambda page: reader.list_issue_comments(
                owner, name, pr_number, page=page, per_page=PAGE_SIZE
            ),
            truncated=_conversation_truncated,
        )
    )
    # The parent question is resolved from the same read, exactly as the live path
    # does, because the comment body is the only place the association can come
    # from.
    questions = {
        marker: comment
        for comment in comments
        if (marker := extract_question_id_from_comment_body(getattr(comment, "body", "") or ""))
    }

    for comment in comments:
        created_at = getattr(comment, "created_at", None)
        if not _within_window(created_at, plan):
            continue
        budget.reserve()
        tally.objects_read += 1

        body = getattr(comment, "body", "") or ""
        question_id = extract_answer_question_id_from_comment_body(body)
        parent = questions.get(question_id) if question_id else None
        if question_id is None or parent is None:
            tally.silent += 1
            budget.refund()
            continue

        try:
            outcome = process_comment_reply_outcome(
                new_comment_id=int(getattr(comment, "id", 0) or 0),
                new_comment_body=body,
                new_comment_author=getattr(getattr(comment, "user", None), "login", None),
                new_comment_created_at=created_at,
                parent_comment_id=int(getattr(parent, "id", 0) or 0),
                repo=repository,
                pr_number=pr_number,
                registry=registry,
                sink=sink,
                question_id=question_id,
                new_comment_author_association=getattr(comment, "author_association", None),
                new_comment_author_type=getattr(getattr(comment, "user", None), "type", None),
                delivery_id=None,
                parent_agent_claim=extract_agent_claim(getattr(parent, "body", "") or ""),
            )
        except UnanchorableReadError as exc:
            # As above: the claim was released, nothing was written, and this is a
            # gap in what could be reconstructed rather than a failed run.
            tally.gap(repository, pr_number, f"comment {getattr(comment, 'id', '?')}", str(exc))
            budget.refund()
            continue
        if outcome is not None:
            tally.records_written += 1
            tally.objects_new += 1
            continue
        # The collector refuses for several reasons, and only one of them means the
        # record is already there. The registry is asked which, because a duplicate
        # and a refusal are different facts and reporting one as the other is how a
        # resumable run starts looking like it did work it did not do.
        if registry.answer_comment_seen(int(getattr(comment, "id", 0) or 0)):
            tally.already_present += 1
        else:
            tally.silent += 1
        budget.refund()
