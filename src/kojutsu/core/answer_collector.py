"""Process GitHub webhook payloads and store answers as knowledge entries."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from kojutsu import allowlist
from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryOutcome,
    KnowledgeDeliveryStatus,
    KnowledgeSink,
)
from kojutsu.core.question_registry import (
    QuestionRegistry,
    stable_answer_entry_id,
)
from kojutsu.core.tanseki_mapping import (
    CHANGE_AUTHOR_KEY as _CHANGE_AUTHOR_KEY,
)
from kojutsu.core.tanseki_mapping import (
    FILES_KEY as _FILES_KEY,
)
from kojutsu.core.tanseki_mapping import (
    HEAD_SHA_KEY as _HEAD_SHA_KEY,
)
from kojutsu.core.tanseki_mapping import (
    PR_MERGED_AT_KEY as _PR_MERGED_AT_KEY,
)
from kojutsu.core.tanseki_mapping import (
    PR_OPENED_AT_KEY as _PR_OPENED_AT_KEY,
)
from kojutsu.core.tanseki_mapping import (
    PR_OUTCOME_KEY as _PR_OUTCOME_KEY,
)
from kojutsu.core.tanseki_mapping import (
    RECORD_KIND_KEY as _RECORD_KIND_KEY,
)
from kojutsu.core.tanseki_mapping import (
    REVIEW_ID_KEY as _REVIEW_ID_KEY,
)
from kojutsu.core.text_hygiene import (
    SANITISATION_KEY,
    SanitisedText,
    combine_notes,
    describe_removals,
    sanitise,
)
from kojutsu.identity import identity_preimage
from kojutsu.integrations.github import (
    AgentClaim,
    extract_agent_claim,
    extract_answer_question_id_from_comment_body,
    extract_answer_text_from_comment_body,
)
from kojutsu.models import (
    CaptureSource,
    CensusRecord,
    Independence,
    KnowledgeEntry,
    QuestionCategory,
    RationaleSource,
    compute_independence,
)

MAX_ANSWER_CHARS = 65_000

#: The ``GitHubUser.type`` GitHub reports for an account acting as an application.
#: Spelled once because the string is the platform's, not ours, and a second
#: spelling is a second thing to keep in step with it. Compared case-insensitively,
#: because the platform's capitalisation is its own and a comparison that depends on
#: it would make the preferred signal quietly fall through to the convention.
GITHUB_BOT_ACCOUNT_TYPE = "Bot"

#: GitHub's naming convention for an account that acts on behalf of an application.
#:
#: **A convention, not a report, and the two were being conflated.** See
#: :func:`is_machine_account`, which now prefers the platform's own ``type`` and
#: falls back to this only when the payload carried none.
#:
#: Recorded rather than inferred downstream, because a consumer asking "was a machine
#: one end of this" should not have to know that the answer lives in a login suffix.
#: It is a marker of provenance, not a judgement: a bot review is still evidence that
#: a review happened, and the point of recording it is that a reader can weigh it.
BOT_ACCOUNT_SUFFIX = "[bot]"


def is_machine_account(login: str | None, *, account_type: str | None = None) -> bool:
    """Whether this account is an application, preferring GitHub's own report.

    **Two signals, in this order, and the order is the point.** GitHub sends
    ``user.type`` on every comment, review and webhook payload it has ever sent;
    :class:`~kojutsu.integrations.github_models.GitHubUser` parsed only ``login`` and
    discarded the rest via ``extra="ignore"``. So the platform has been asked which
    account this is and the answer thrown away, while the function's own docstring
    claimed it returned whether "GitHub reports this account as an application" —
    which was false, and false in the way that gets trusted later: the suffix
    happened to agree with GitHub on all eight accounts checked, so the claim was not
    demonstrably wrong, only not a claim at all. Agreement is not a source.

    With ``account_type`` present it is authoritative: ``"Bot"`` is a machine and
    anything else is not, whatever the login is called. With it absent the answer
    falls back to :data:`BOT_ACCOUNT_SUFFIX`, and that fallback is a naming
    convention, so two things are worth saying plainly rather than leaving to be
    discovered:

    - **An absent ``type`` is not a person.** It is the forge declining to say, and
      this answers it with the weaker signal while saying so in the return value's
      documentation rather than inventing certainty. Callers that have the payload
      should pass it; callers that have only a login should not pretend the two are
      the same question.
    - **Both residual errors point the same way and both cost the same thing:** an
      application recorded as a person. An app whose login lacks the suffix is
      missed when no ``type`` is supplied, and an account whose login carries the
      suffix while its ``type`` says otherwise is called a person because the
      platform was believed over the convention. Neither is fixable here, and both
      are the price of recording a boolean at all — which is still worth paying,
      because a reader who cannot see the flag has to infer automation from a name.

    Kept here, beside the automation key, rather than at the call sites so that
    the definition of "machine" is one edit rather than a search.
    """
    reported = (account_type or "").strip()
    if reported:
        return reported.casefold() == GITHUB_BOT_ACCOUNT_TYPE.casefold()
    return login is not None and login.endswith(BOT_ACCOUNT_SUFFIX)


#: The key a stored record carries automation under. One spelling for all four
#: capture paths, because a reader filtering on it cannot be expected to know which
#: writer produced the document, and two names for one fact is two filters, one of
#: which silently matches nothing.
COMMENT_AUTHOR_IS_MACHINE_KEY = "comment_author_is_machine"


#: The admission policy (which ``author_association`` values are admitted, and the
#: predicate applying the rule) lives in :mod:`kojutsu.allowlist` beside the
#: repository allowlist -- a single definition imported by the collectors, not one
#: per collector. Used here qualified as ``allowlist.*`` so this module defines
#: no policy name of its own: an alias here would be the same duplication spelled
#: differently, and the alias is what would be edited next.


#: Why a review event stored nothing. A closed set, because the reason travels with
#: the result and a tally groups by it: a reader gets names to reason about rather
#: than strings to match. The names describe the gate, not the intent — a refused
#: review is not evidence that its author had nothing to say.
#:
#: ``ASSOCIATION`` is **not reachable under the default policy** — see
#: :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` for the measurement and the reason — and it is kept
#: because narrowing is still a legitimate operator decision, not because it is
#: recommended. It is also the reason that is routinely mistaken for a bot filter,
#: and it is still not one: it consults no account name and no suffix, so a narrowed
#: run refuses bots and outsiders by exactly the same test, and reports the refusal
#: under this name rather than letting it disappear. The refusal is a fact about
#: this system's configuration and nothing about what the reviewer intended, so no
#: *reason* is recorded on the census entry either way.
REVIEW_REFUSAL_NO_CHANGE = "no_change_reference"
REVIEW_REFUSAL_ASSOCIATION = "author_association_not_authorised"
REVIEW_REFUSAL_NO_ACCOUNT = "reviewer_account_missing"
REVIEW_REFUSAL_NOTHING_CAPTURABLE = "nothing_capturable"


def _transition_marker(value: datetime | str | None) -> str | None:
    """Render a transition timestamp as the text the identity preimage carries.

    The identity below is a JSON array of strings, and the timestamps it folds in
    arrive from the payload as ``datetime`` objects. They were being handed to
    ``json.dumps`` directly, which raises ``TypeError: Object of type datetime is
    not JSON serializable`` — so *every* close and reopen whose payload carried a
    timestamp crashed the handler instead of producing an id. Only the all-``None``
    case had ever been exercised, which is why this survived: the function was
    tested on inputs the forge does not send, and the inputs the forge does send
    raised.

    Normalising to an ISO-8601 string keeps the preimage textual and the hash
    stable. A caller that already passes a string is passed through unchanged,
    so a legacy caller's ids do not move; and the ``None`` case still serialises
    as ``null`` exactly as before, so ids derived when no timestamp was available
    are byte-identical to the ones already stored.

    That leaves only the inputs that previously raised, which is what makes this
    safe to change at all: **no stored record can depend on a value that was never
    produced.** There is nothing to re-identify and no document to orphan.
    """
    if value is None or isinstance(value, str):
        return value
    return value.isoformat()


#: Domain label separating lifecycle-event identity from every other namespace.
#: Part of the hashed preimage, so it both keeps the namespaces apart and makes a
#: change to *how* the digest is computed produce a different value rather than one
#: that looks comparable and is not.
PR_EVENT_IDENTITY_DOMAIN = "kojutsu.pr_event.v1"


def semantic_pr_event_id(
    repo: str,
    pr_number: int,
    action: str,
    pr_state: str,
    pr_closed_at: datetime | str | None,
    *,
    pr_merged_at: datetime | str | None = None,
    pr_updated_at: datetime | str | None = None,
) -> str:
    """Stable identity for one pull request lifecycle transition, keyed on when it happened.

    A delivery id is not usable here: GitHub re-serialises and re-delivers the same
    transition under a new delivery id, which would store the transition twice. What
    the transition *is* -- which change, which action, the state it moved to, and the
    moment it moved -- is the same on every delivery, so keying on those collapses
    the re-deliveries onto one record.

    This is the derivation with the most room to be wrong by accident, because its
    fields have the same shape as another derivation's: five values, positionally
    significant, one of them an integer the forge issues. :func:`semantic_review_event_id`
    takes five too. Nothing in a bare JSON array says which kind of record asked for
    it, so before the domain label was folded in, a lifecycle transition and a review
    of the same change were one namespace -- and the ``pr-event:`` prefix was the only
    thing keeping them apart, which is a string convention rather than a property of
    the digest.
    """
    transition_marker = _transition_marker(pr_closed_at or pr_merged_at)
    if transition_marker is None and action in {"closed", "reopened"}:
        transition_marker = _transition_marker(pr_updated_at)
    identity = identity_preimage(
        PR_EVENT_IDENTITY_DOMAIN,
        (repo.casefold(), pr_number, action.casefold(), pr_state.casefold(), transition_marker),
    )
    return "pr-event:" + hashlib.sha256(identity).hexdigest()


MAX_REVIEW_CHARS = 65_000


#: The closed vocabulary of record kinds this module writes, dispatched through
#: :func:`_record_tags` so a writer cannot produce a record without one.
#:
#: A kind is a closed set because it is what a consumer filters on. An open
#: string field would let a future writer invent ``"pr_lifecycle_event"`` beside
#: ``"pr_lifecycle"``, and a dashboard filtering on the first would then show an
#: empty panel for a corpus that is full of records — the same class of quiet
#: failure as a tag typo, but with no way to spot it, because the value would be
#: a perfectly good string.
#:
#: ``REVIEW_VERDICT`` and ``INLINE_REVIEW_COMMENT`` keep the exact values the
#: store already holds. They are in every stored review record's ``record_kind``
#: and in the console's filters, so they are data, not vocabulary.
#:
#: Why the closed *set* is so small: these are the writers, and each is a
#: different thing to ask a reader to believe. An answer is a conclusion someone
#: reached, a verdict is a decision about someone else's change, an inline
#: comment is a note about a line of code that has since moved, a lifecycle
#: record is the write path reporting on itself, and a census record is the write
#: path reporting that it saw something and kept nothing. Nothing else is stored
#: as an entry, and nothing else should be: a stated rationale and a projected
#: question are different models with a different trust, dispatched separately at
#: :func:`kojutsu.core.knowledge_sink.to_payload`.
class RecordKind(StrEnum):
    ANSWER = "answer"
    REVIEW_VERDICT = "review_verdict"
    INLINE_REVIEW_COMMENT = "inline_review_comment"
    PR_LIFECYCLE = "pr_lifecycle"
    CHECK_RUN = "check_run"
    #: An observation that captured nothing. It is an ``answer_collector`` kind
    #: because it is written from the same place — the webhook handler, at the
    #: moment it learns what it stored — and because keeping it here is what stops
    #: it being invented with its own spelling later.
    CENSUS = "census"


RECORD_KINDS: frozenset[str] = frozenset(kind.value for kind in RecordKind)

#: Stable aliases for the two review kinds, kept because the registry claims
#: captures against them and callers name them. Same values, not a second list:
#: two spellings of one kind is how a closed vocabulary stops being closed.
REVIEW_KIND_VERDICT = RecordKind.REVIEW_VERDICT.value
REVIEW_KIND_INLINE = RecordKind.INLINE_REVIEW_COMMENT.value

#: Tags are a *projection* of the kind, not an independent claim about it. They
#: stay because the console and the dashboard filter on them today, and they stay
#: correct only because they are derived here rather than spelled at each call
#: site — a tag set that could drift from its kind is a second, quieter answer to
#: "what kind of record is this?".
RECORD_KIND_TAGS: dict[RecordKind, tuple[str, ...]] = {
    # An answer carries no tag of its own: the tag that does work here
    # (``agent_authored``) says who wrote it, which is orthogonal to what kind of
    # record it is.
    RecordKind.ANSWER: (),
    RecordKind.REVIEW_VERDICT: ("review",),
    RecordKind.INLINE_REVIEW_COMMENT: ("review", "inline_comment"),
    RecordKind.PR_LIFECYCLE: ("pr_state_change",),
    # A check conclusion is a machine report about a commit, kept in its own
    # kind so it can never be rendered where a reader would take it for a
    # review. ``check_state_<conclusion>`` is the *forge's* wording, not kojutsu's
    # judgement of it.
    RecordKind.CHECK_RUN: ("check",),
    # An observation carries ``census`` and nothing else, deliberately. The tag that
    # does the work is the kind itself: this record exists to be counted *apart* from
    # every capture, so any tag it shares with a capture is a tag a filter could
    # catch it by and read as knowledge.
    RecordKind.CENSUS: ("census",),
}

#: Review states that represent a decision about the code. ``commented`` is not
#: one: leaving a note with no verdict is feedback, not an approval.
REVIEW_VERDICT_STATES = frozenset({"approved", "changes_requested"})

#: How a closed pull request ended, as a closed vocabulary with three states
#: rather than one. GitHub reports a merge and an abandonment identically —
#: ``state`` is ``"closed"`` either way — so ``merged_at`` is the only thing that
#: separates them, and the record has to say which side of that line it is on.
#: A boolean cannot: it would have to collapse "closed without merging" and "we
#: were not told" into the same value, and those are different facts about
#: different records. So the third state is absence, written by leaving the key
#: out rather than by writing a null.
PR_OUTCOME_MERGED = "merged"
PR_OUTCOME_CLOSED_UNMERGED = "closed_unmerged"
PR_OUTCOMES: frozenset[str] = frozenset({PR_OUTCOME_MERGED, PR_OUTCOME_CLOSED_UNMERGED})

#: The login that opened the pull request, recorded under a name of its own.
#:
#: It cannot go in ``author``, which is who produced this particular record, and
#: it cannot go in ``comment_author``, which is the reviewer. It is the third
#: party, and it is the one every independence claim is computed *against*: a
#: verdict from the change's own author is not a check on that change. Naming it
#: ambiguously would leave a reader unable to reconstruct why a record was
#: labelled the way it was.
#:
#: The key name itself comes from :mod:`kojutsu.core.tanseki_mapping`, which owns
#: the storage contract. Spelling it here as well would be two spellings of one
#: key, and a document filtering on the wrong one would be missing rather than
#: failing.
CHANGE_AUTHOR_KEY = _CHANGE_AUTHOR_KEY
HEAD_SHA_KEY = _HEAD_SHA_KEY
FILES_KEY = _FILES_KEY

#: GitHub's own creation time for the change, under a name that says whose
#: timestamp it is. Never ``updated_at``: on a lifecycle record ``updated_at`` is
#: when the event happened (``answered_at``), and on a review record it is when
#: the review was submitted. Overloading either would report a fact about the
#: wrong event.
PR_OPENED_AT_KEY = _PR_OPENED_AT_KEY
PR_MERGED_AT_KEY = _PR_MERGED_AT_KEY
PR_OUTCOME_KEY = _PR_OUTCOME_KEY
RECORD_KIND_KEY = _RECORD_KIND_KEY
REVIEW_ID_KEY = _REVIEW_ID_KEY

REVIEW_CAPTURE_METADATA_TYPES = {"str", "int", "float", "bool", "type", "NoneType"}


def _require_kind(kind: str) -> RecordKind:
    """Resolve a written kind to its closed-vocabulary member, or refuse it.

    Every writer in this module routes its kind through here, which is what
    makes "a new writer cannot produce a record without a kind" true rather than
    a convention: the tag projection needs a member to look the tags up in, so a
    writer that skipped the enum would have to invent its tags instead, and the
    first thing a new writer does is copy the lines above it.
    """
    try:
        return RecordKind(kind)
    except ValueError:
        raise ValueError(
            f"record kind {kind!r} is not in the closed vocabulary "
            f"({', '.join(sorted(RECORD_KINDS))})"
        ) from None


def _record_tags(
    kind: RecordKind,
    *,
    review_state: str | None = None,
    action: str | None = None,
    check_conclusion: str | None = None,
    agent_authored: bool = False,
) -> list[str]:
    """Project a record kind onto the tags consumers already filter on.

    The base set comes from :data:`RECORD_KIND_TAGS` so it cannot drift from the
    kind; the two extras are *refinements of the kind itself* rather than
    separate facts, and are validated against it so they cannot be attached to a
    record that is not of that kind. ``agent_authored`` is the exception and is
    deliberately free-form: it answers "who wrote this", which no kind does.
    """
    tags = list(RECORD_KIND_TAGS[kind])
    if review_state is not None:
        if kind is not RecordKind.REVIEW_VERDICT:
            raise ValueError("a review state refines a verdict and nothing else")
        tags.append(f"review_state_{review_state}")
    if action is not None:
        if kind is not RecordKind.PR_LIFECYCLE:
            raise ValueError("a lifecycle action refines a lifecycle record and nothing else")
        tags.append(f"action_{action}")
    if check_conclusion is not None:
        if kind is not RecordKind.CHECK_RUN:
            raise ValueError("a check conclusion refines a check record and nothing else")
        tags.append(f"check_state_{check_conclusion}")
    if agent_authored:
        tags.append("agent_authored")
    return tags


def json_safe_metadata(value: object) -> object:
    """Reduce a value to something the Tanseki frontmatter writer can serialise.

    Review payloads carry values whose types are not known statically -- a diff
    hunk, an anchor, a commit id. Storing an arbitrary object would fail at the
    storage boundary, far from the code that accepted it, and would be a
    crash-on-read rather than a capture failure. Anything that is not a plain
    scalar is dropped instead, so an unserialisable field costs that one field
    rather than the whole record.
    """
    if value is None or type(value).__name__ in REVIEW_CAPTURE_METADATA_TYPES:
        return value
    return None


def _sanitised(value: object) -> SanitisedText | None:
    """Apply the stored-text character policy to a captured field that may be absent.

    ``None`` for a field that is not prose, so a caller can pass a payload field
    through without first deciding whether it is a string. That decision belongs
    here rather than at each call site because getting it wrong has two opposite
    failures -- coercing a non-string into one writes ``"None"`` into a stored field,
    and skipping a genuine string loses the sanitisation -- and only one of them
    raises.

    The policy itself is :func:`kojutsu.core.text_hygiene.sanitise`, which is applied
    at the point a field becomes stored text. See that module for why it is not
    applied to the marker grammar or to the signed delivery bytes.
    """
    return sanitise(value) if isinstance(value, str) else None


#: Domain label separating review-event identity from the lifecycle-event namespace
#: it shares a field layout with. See :func:`semantic_pr_event_id`.
REVIEW_EVENT_IDENTITY_DOMAIN = "kojutsu.review_event.v1"


def semantic_review_event_id(
    repo: str,
    pr_number: int,
    review_id: int,
    *,
    comment_id: int | None = None,
    submitted_at: datetime | None = None,
) -> str:
    """Stable identity for one review verdict or one inline review comment.

    A delivery id is not usable here: GitHub re-serialises and re-delivers the same
    review under a new delivery id, which would store the reviewer's decision
    twice. Keying on the provider's own review and comment ids collapses those to
    one record, the same way :func:`semantic_pr_event_id` does for lifecycle
    transitions.

    ``submitted_at`` is folded in through ``default=str``, which renders a
    ``datetime`` as ``str()`` rather than as ``isoformat()``. It is left that way
    deliberately: a stored review id is a stored review id, and changing the
    rendering would re-identify every review record captured with a timestamp.
    :func:`semantic_pr_event_id` renders the same kind of value differently, which
    is a genuine inconsistency in the corpus and is recorded here rather than fixed
    inside a hash.
    """
    identity = identity_preimage(
        REVIEW_EVENT_IDENTITY_DOMAIN,
        (repo.casefold(), pr_number, review_id, comment_id, submitted_at),
        default=str,
    )
    return "review:" + hashlib.sha256(identity).hexdigest()


#: Version of the review-entry identity derivation, for the reason spelled out on
#: ``RATIONALE_IDENTITY_VERSION``: the entry id becomes the Tanseki document id, so
#: moving the computation orphans documents that are already stored.
REVIEW_ENTRY_IDENTITY_VERSION = 1

#: Domain label separating review-entry identity from every other namespace. This
#: one was the sharpest gap: a check run and a review are both claimed in
#: ``review_captures`` and both used to hash a bare ``"<prefix>:<repo>:<id>"`` event
#: string, so ``stable_review_entry_id("check-run:org/repo:99")`` produced the very
#: digest ``process_check_run_outcome`` derives for that check run. Nothing in the
#: preimage said which kind of event it was, so the two kinds were one namespace
#: wearing two prefixes.
REVIEW_ENTRY_IDENTITY_DOMAIN = "kojutsu.review_entry.v1"


def stable_review_entry_id(review_event_id: str) -> str:
    """Durable identity for a captured review record.

    Takes the semantic event id and nothing else -- no repository, no change, no
    review. That is not an omission to fix here: the event id is *already* derived
    from the repository, the change, the review and the comment, so naming them
    again would add nothing except a second place for the same fact to be wrong. It
    is worth knowing that the guarantee is inherited rather than local, though: a
    caller that passes something other than a
    :func:`semantic_review_event_id` value gets an identity over exactly that string
    and no more, and the registry claim in ``review_captures`` is keyed on the same
    string.
    """
    return (
        f"review-v{REVIEW_ENTRY_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(REVIEW_ENTRY_IDENTITY_DOMAIN, (review_event_id,))
        ).hexdigest()
    )


def _store_outcome(
    sink: KnowledgeSink, entry: KnowledgeEntry | CensusRecord
) -> KnowledgeDeliveryOutcome:
    # Deliberately not :data:`~kojutsu.core.knowledge_sink.StorableRecord`: a
    # projected question has no ``entry_id`` — its identity is its ``question_id``
    # and the projection delivers it on its own path — so the attribute this reads
    # is not one every storable record has.
    result = sink.store(entry)
    if result is None:
        return KnowledgeDeliveryOutcome(
            entry_id=entry.entry_id,
            status=KnowledgeDeliveryStatus.QUEUED,
            detail="Sink returned no explicit delivery outcome; delivery is uncertain",
        )
    if not isinstance(result, KnowledgeDeliveryOutcome):
        raise TypeError("KnowledgeSink.store must return KnowledgeDeliveryOutcome or None")
    return result


#: Version of the census identity derivation, for the reason spelled out on
#: ``RATIONALE_IDENTITY_VERSION``.
CENSUS_IDENTITY_VERSION = 1

#: Domain label separating census identity from every other namespace.
CENSUS_IDENTITY_DOMAIN = "kojutsu.census.v1"


def stable_census_entry_id(repo: str, pr_number: int | None, action: str) -> str:
    """Durable identity for one observation.

    Derived from ``(repo, pr, action)`` — the same natural key
    :func:`~kojutsu.core.tanseki_mapping.census_document_id` keys the document on,
    and deliberately not from the delivery id. The delivery is the *anchor*: it is
    what a reader re-fetches to check the observation. Keying identity on it as
    well would make one event redelivered under two delivery ids two observations,
    and the store would then hold a document per redelivery of the same fact.
    """
    identity = identity_preimage(
        CENSUS_IDENTITY_DOMAIN,
        (repo.casefold(), pr_number if pr_number is not None else 0, action.casefold()),
    )
    return f"census-v{CENSUS_IDENTITY_VERSION}-" + hashlib.sha256(identity).hexdigest()


#: Version of the check-run identity derivation, for the reason spelled out on
#: ``RATIONALE_IDENTITY_VERSION``.
CHECK_IDENTITY_VERSION = 1

#: Domain label separating check-run identity from the review namespace it used to
#: share. See :data:`REVIEW_ENTRY_IDENTITY_DOMAIN` for the collision.
CHECK_IDENTITY_DOMAIN = "kojutsu.check.v1"


def stable_check_entry_id(check_event_id: str) -> str:
    """Durable identity for one captured check-run conclusion.

    Extracted from the writer and domain-separated rather than left as an inline
    digest, because a check run and a review of the same change are claimed in the
    same table under event ids of the same shape. Inlined, the two writers hashed
    the bare event string and produced the same digest for the same string; the only
    thing that told them apart afterwards was the ``check-`` against the ``review-``
    on the front, and a prefix is not a namespace.

    It takes the event id rather than ``(repo, check_run_id)`` because the event string
    already carries both, and re-deriving them here would be a second spelling of the
    same fact for the two halves of one record to disagree about.
    """
    return (
        f"check-v{CHECK_IDENTITY_VERSION}-"
        + hashlib.sha256(identity_preimage(CHECK_IDENTITY_DOMAIN, (check_event_id,))).hexdigest()
    )


def record_census_observation_outcome(
    *,
    repo: str,
    pr_number: int | None,
    action: str,
    delivery_id: str | None,
    sink: KnowledgeSink,
    change_author_account: str | None = None,
    head_sha: str | None = None,
    observed_at: datetime | None = None,
) -> KnowledgeDeliveryOutcome | None:
    """Store the observation that a delivery was processed and captured nothing.

    One call per delivery that reached a writer and stored nothing, and only from
    that writer: the question "did this delivery capture anything" is answered at
    the point of writing and nowhere else. It is answered *there* rather than by
    asking the store what the change already holds, because a census record that
    is suppressed because other records exist measures something other than what
    it says it measures — a change opened quietly and reviewed later with a capture
    has both.

    Refuses rather than guesses when the observation names nothing. Without a
    repository and a change there is no document to key on, and every malformed
    delivery would otherwise share one ``pr-0`` bucket, so a count over census
    documents would be a count over broken payloads. The same reasoning retires
    the no-delivery case, which :class:`~kojutsu.models.CensusRecord` refuses
    outright: "we looked at this and kept nothing" is the easiest record in the
    system to invent, and it is only evidence when a reader can re-fetch the
    delivery behind it.

    There is no registry claim, unlike every capture here. The record's identity is
    the document it becomes, and the outbox already keys on that identity, so a
    second dedup layer would be a second thing to keep in step with the document
    key — and a row that outlived its document would leave the claim refusing a
    record nothing ever wrote.
    """
    if not repo or not pr_number or not action:
        return None
    if not str(delivery_id or "").strip():
        return None
    record = CensusRecord(
        entry_id=stable_census_entry_id(repo, pr_number, action),
        repo=repo,
        pr_number=pr_number,
        pr_url=f"https://github.com/{repo}/pull/{pr_number}",
        # The delivery's own action, verbatim. This writer does not classify what
        # happened; a push and a review are told apart by the key, not by a field
        # that would have to be interpreted.
        action=action,
        change_author_account=change_author_account,
        head_sha=head_sha,
        observed_at=observed_at or datetime.now(UTC),
        delivery_id=delivery_id,
    )
    return _store_outcome(sink, record)


def process_pr_state_change(
    action: str,
    pr_number: int,
    pr_title: str,
    pr_state: str,
    pr_closed_at: datetime | None,
    repo: str,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    delivery_id: str | None = None,
    event_identity: str | None = None,
    pr_merged_at: datetime | None = None,
    pr_opened_at: datetime | None = None,
    pr_author_account: str | None = None,
    head_sha: str | None = None,
    files_changed: list[str] | None = None,
) -> bool:
    return (
        process_pr_state_change_outcome(
            action=action,
            pr_number=pr_number,
            pr_title=pr_title,
            pr_state=pr_state,
            pr_closed_at=pr_closed_at,
            repo=repo,
            registry=registry,
            sink=sink,
            delivery_id=delivery_id,
            event_identity=event_identity,
            pr_merged_at=pr_merged_at,
            pr_opened_at=pr_opened_at,
            pr_author_account=pr_author_account,
            head_sha=head_sha,
            files_changed=files_changed,
        )
        is not None
    )


@dataclass(frozen=True)
class ReviewCaptureResult:
    """What one review event stored, and whether it had already stored it.

    An empty :attr:`outcomes` does not mean the event had nothing to say. It means
    either that, or that every record it would have written is already in the
    store: GitHub re-serialises and re-delivers a review under a fresh delivery id,
    which the delivery claim cannot catch and the semantic review id can. The two
    are indistinguishable from the list alone, and only the first is an
    observation Kojutsu may record as silence — the second would claim a
    delivery captured nothing when the capture for that very review exists.

    So the fact travels with the result rather than being left to a reader to
    infer. A caller that only wants the outcomes keeps reading them as a list.
    """

    outcomes: list[KnowledgeDeliveryOutcome]
    already_stored: bool
    #: Why the event stored nothing, or ``None`` when it stored something.
    #:
    #: Added because the single "silent" tally merged facts with opposite
    #: implications: a review that carried nothing capturable is an absence of
    #: review, while a review refused by a policy is a presence of review being
    #: discarded, and a caller reading one counter cannot tell which it has. The
    #: reason is carried here rather than recomputed by the caller because only
    #: this function knows which gate refused — a caller inferring it would be
    #: guessing at a list of rules that can change under it.
    #:
    #: A named value rather than a free string so a tally can group by it and a
    #: reader gets a closed set to reason about. ``None`` on a stored event and on
    #: a duplicate, because a duplicate is not a refusal.
    refusal: str | None = None


def capture_counts(result: object) -> tuple[int, bool]:
    """How many records an event stored, and whether they were already there.

    ``process_review_event_outcome`` returns a result object whose ``outcomes`` and
    ``already_stored`` answer two different questions: did it write, and was this a
    duplicate rather than silence. The second is the one a backfill's resumability
    rests on, and it is exactly the answer a caller reading only the outcome count
    would get wrong.

    Lives here rather than in either backfill so that there is one way to read this
    return value. It was duplicated once already: ``core/backfill.py`` kept calling
    ``len()`` on the result and every backfill failed with a ``TypeError`` on the
    first review it met, while the other backfill -- written later, against the
    changed signature -- worked. Nothing in the build connects a return type to its
    callers, so a caller that reads it wrongly is only found by running it.

    Read by shape rather than by ``isinstance`` against the class, so that a caller
    holding either form keeps working: the collector and its callers are changed
    separately, and a run that began failing because the return type gained a field
    would turn somebody else's refactor into a broken backfill.
    """
    outcomes = getattr(result, "outcomes", None)
    if isinstance(outcomes, list):
        return len(outcomes), bool(getattr(result, "already_stored", False))
    if isinstance(result, list):
        return len(result), False
    return 0, False


@dataclass
class _ReviewContext:
    """Everything one review event's writes need, derived once up front.

    The nested writer this replaces closed over eighteen locals; a reader of
    the verdict loop could not tell which of them the write actually needed.
    Named fields say so, and they make the writer a module function its
    callers can exercise without staging a whole review event. Per-record
    values (event id, kind, texts, tags) stay per-call parameters — they
    differ on every write, so folding them in would make one context per
    record rather than one per event.
    """

    repo: str
    pr_number: int
    review_id: int
    review_author: str
    review_submitted_at: datetime | None
    delivery_id: str | None
    pr_opened_at: datetime | None
    head_sha: str | None
    files_changed: list[str] | None
    review_author_type: str | None
    association: str | None
    review_body: str
    review_body_field: SanitisedText
    state: str
    review_claim: AgentClaim | None
    reviewer_model: str | None
    pr_author: str | None
    independence: Independence
    independence_reason: str


def _review_context(
    *,
    repo: str,
    pr_number: int,
    review_id: int,
    review_author: str | None,
    review_submitted_at: datetime | None,
    delivery_id: str | None,
    pr_author_account: str | None,
    review_author_association: str | None,
    review_body: str,
    review_state: str,
    review_author_type: str | None,
    pr_opened_at: datetime | None,
    head_sha: str | None,
    files_changed: list[str] | None,
    authorized_associations: frozenset[str] | None,
) -> _ReviewContext | ReviewCaptureResult:
    """Run the admission gates, or return the refusal as a result.

    A refusal is a ``ReviewCaptureResult``, not an exception: an empty review,
    an unauthorised reviewer and a missing account are ordinary outcomes the
    caller reports, and the census path distinguishes them by name.
    """
    if not repo or pr_number == 0:
        return ReviewCaptureResult(
            outcomes=[], already_stored=False, refusal=REVIEW_REFUSAL_NO_CHANGE
        )
    association = review_author_association.strip().upper() if review_author_association else None
    # Under the default policy this admits everything, which is a decision and not an
    # omission: the set it used to apply refused 28 human comments on one change to
    # admit 21 bot ones, because ``author_association`` sorts automation into
    # CONTRIBUTOR. See :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` for the measurement. The gate
    # stays because an operator narrowing deliberately is legitimate, and when it
    # fires the refusal is reported under its own name rather than vanishing — it is
    # *not* a reason, and nothing below records one: kojutsu knows it declined the
    # capture, and that is a fact about its own configuration, not about anything the
    # reviewer intended.
    if not allowlist.association_admitted(association, authorized_associations):
        return ReviewCaptureResult(
            outcomes=[], already_stored=False, refusal=REVIEW_REFUSAL_ASSOCIATION
        )
    if not review_author:
        return ReviewCaptureResult(
            outcomes=[], already_stored=False, refusal=REVIEW_REFUSAL_NO_ACCOUNT
        )

    # Captured prose reaches this function through two doors, and both are closed here,
    # before anything reads the text, so the emptiness test, the length bound and the
    # stored body are three views of one value rather than three that can disagree --
    # a body of nothing but invisible controls has to be refused here, not stored as a
    # review whose visible content is empty.
    #
    # One door is a comment body, which arrives through the marker extractors in
    # :mod:`kojutsu.integrations.github`; those read the ``kojutsu:`` grammar verbatim
    # and normalise a marker's value, which is why they are not the place the wider
    # policy is applied and why the answer writer below does not use them either. The
    # other door is a payload field with no extractor behind it -- the review body, and
    # each inline comment's body, path and diff hunk, which are sanitised in the loop
    # below so the removals can still be reported. Both doors go through
    # :func:`sanitise`, whose refused set is a superset of the extractors', so nothing
    # the extractors would have removed survives the wider pass.
    review_body_field = sanitise(review_body)
    clean_body = review_body_field.text

    state = review_state.strip().lower()
    review_claim = extract_agent_claim(clean_body)
    # The author of the change is whoever opened the PR. A review from that same
    # account, by the same model, restates the change rather than checking it, so
    # the independence comparison needs the PR author rather than an absent value.
    pr_author = pr_author_account
    reviewer_model = review_claim.model if review_claim is not None else None
    # The change side of a review has no model marker and never will — a pull
    # request is a branch, not a sentence an agent wrote — so nothing is passed
    # as the asker model rather than assuming it equals the reviewer's. Assuming
    # two unknowns match would manufacture a weaker label in exactly the case
    # that needs scrutiny. ``compute_independence`` already answers this: an
    # unstated side yields ``SELF_CERTIFIED`` with the reason recorded, which is
    # the correct answer, so it is left alone.
    independence, independence_reason = compute_independence(
        asker_account=pr_author,
        asker_model=None,
        answerer_account=review_author,
        answerer_model=reviewer_model,
    )
    return _ReviewContext(
        repo=repo,
        pr_number=pr_number,
        review_id=review_id,
        review_author=review_author,
        review_submitted_at=review_submitted_at,
        delivery_id=delivery_id,
        pr_opened_at=pr_opened_at,
        head_sha=head_sha,
        files_changed=files_changed,
        review_author_type=review_author_type,
        association=association,
        review_body=clean_body,
        review_body_field=review_body_field,
        state=state,
        review_claim=review_claim,
        reviewer_model=reviewer_model,
        pr_author=pr_author,
        independence=independence,
        independence_reason=independence_reason,
    )


def _store_review_record(
    ctx: _ReviewContext,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    *,
    event_id: str,
    kind: RecordKind,
    comment_id: int | None,
    question_text: str,
    answer_text: str,
    category: QuestionCategory,
    tags: list[str],
    extra_metadata: dict[str, object],
    #: Prose the caller already sanitised, so the record's note can name every
    #: character that was removed from it rather than only those in the answer.
    #: One inline comment carries a body, a file path and a diff hunk, and three
    #: separate notes would leave a reader guessing whether they describe one
    #: attack or three.
    prose_fields: tuple[SanitisedText | None, ...] = (),
) -> tuple[bool, KnowledgeDeliveryOutcome | None]:
    """Claim, build and deliver one review record.

    Returns ``(attempted, outcome)``. The split matters downstream: an empty or
    oversized body is nothing to capture rather than a record already held, so
    it counts as neither an attempt nor a duplicate, while a refused claim is
    an attempt at something already stored. Collapsing the two would make a
    re-delivered review read as silence, and silence is what the census path
    records.
    """
    # Re-sanitised here rather than trusted, so this is the one place a review's
    # answer text becomes *stored* text and therefore the one place the policy is
    # applied to it. Idempotent, so a caller that already ran it costs nothing and
    # a caller that did not cannot get it wrong.
    answer = sanitise(answer_text)
    sanitisation_note = combine_notes(answer, *prose_fields)
    # An empty or oversized body is nothing to capture rather than a record
    # already held, so it is counted below as neither an attempt nor a
    # duplicate. Tested against the *sanitised* body, so a review whose entire
    # content is invisible controls is treated exactly as a review with no body
    # -- the same outcome the extractors already produce for a body of zero-width
    # spaces, rather than a second rule that disagrees with it.
    if not answer.text.strip() or len(answer.text) > MAX_REVIEW_CHARS:
        return False, None
    entry_id = stable_review_entry_id(event_id)
    claim_token = registry.claim_review_capture(
        review_event_id=event_id,
        repo=ctx.repo,
        pr_number=ctx.pr_number,
        review_id=ctx.review_id,
        comment_id=comment_id,
        kind=kind.value,
        author=ctx.review_author,
    )
    if claim_token is None:
        return True, None
    entry = KnowledgeEntry(
        entry_id=entry_id,
        session_id=None,
        question_text=question_text,
        answer_text=answer.text,
        category=category,
        context=None,
        # A declared agent is the author of the verdict, whoever submitted it.
        author=(ctx.review_claim.agent_id if ctx.review_claim is not None else None)
        or ctx.review_author,
        answered_at=ctx.review_submitted_at or datetime.now(UTC),
        embedding=None,
        tags=tags,
        metadata={
            "repo": ctx.repo,
            "pr_number": ctx.pr_number,
            "pr_url": f"https://github.com/{ctx.repo}/pull/{ctx.pr_number}",
            REVIEW_ID_KEY: ctx.review_id,
            "review_state": ctx.state,
            RECORD_KIND_KEY: kind.value,
            CHANGE_AUTHOR_KEY: ctx.pr_author,
            PR_OPENED_AT_KEY: ctx.pr_opened_at.isoformat() if ctx.pr_opened_at else None,
            HEAD_SHA_KEY: ctx.head_sha,
            FILES_KEY: list(ctx.files_changed) if ctx.files_changed else None,
            "comment_author": ctx.review_author,
            # Kept even though nothing gates on it any more. It was the gate, so
            # it is now the fact a reader weighs instead: which standing the forge
            # attributed to this account, recorded because the forge said it and
            # not because kojutsu acted on it. Dropping it with the filter would
            # have thrown away the only thing left that distinguishes a project
            # member from a drive-by commenter.
            "github_author_association": ctx.association,
            # Whether GitHub attributes this verdict to an application, resolved
            # once here so a reader asking "was a machine one end of this" never
            # has to know the answer lives in a login suffix or in a field
            # GitHub has been sending all along. The association cannot answer it:
            # the sampled reviewer accounts behind that field are mostly
            # CONTRIBUTOR, which is the same value an outside human gets, and
            # which is also where every automated reviewer lands.
            "reviewer_is_machine": is_machine_account(
                ctx.review_author, account_type=ctx.review_author_type
            ),
            "answered_by_agent": (
                ctx.review_claim.agent_id if ctx.review_claim is not None else None
            ),
            # Absent, not ``"unknown"``, when the reviewer stated no model:
            # a stored record in which a principal nobody named appears to
            # have declared a model called ``"unknown"`` is a document that
            # cannot be told apart from one where somebody really did. The
            # frontmatter extras skip ``None``, which is the honest way to
            # leave the key out.
            "answered_by_model": ctx.reviewer_model,
            "independence": ctx.independence.value,
            "independence_reason": ctx.independence_reason,
            "delivery_id": ctx.delivery_id,
            # What the character policy took out of this record's prose, in the
            # record's own words. Present only when something was removed, so its
            # absence is the claim that this record's text is byte-for-byte what
            # the reviewer wrote -- which is a fact worth being able to read, and
            # one that is otherwise indistinguishable from a record that was
            # quietly rewritten at the boundary. See
            # :data:`kojutsu.core.text_hygiene.SANITISATION_KEY`.
            **({SANITISATION_KEY: sanitisation_note} if sanitisation_note else {}),
            **extra_metadata,
        },
        capture_source=CaptureSource.WEBHOOK,
        captured_at=datetime.now(UTC),
        capture_delivery_id=ctx.delivery_id or event_id,
    )
    try:
        outcome = _store_outcome(sink, entry)
        if outcome.dead_lettered:
            if not registry.release_review_capture(event_id, claim_token, outcome.detail or "x"):
                raise RuntimeError("Failed to release dead-lettered review capture claim")
        elif not registry.complete_review_capture(event_id, claim_token):
            raise RuntimeError("Failed to complete review capture claim")
    except Exception as exc:
        registry.release_review_capture(event_id, claim_token, type(exc).__name__)
        raise
    return True, outcome


def _store_review_verdict(
    ctx: _ReviewContext,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
) -> tuple[bool, KnowledgeDeliveryOutcome | None]:
    """Store the review's own verdict, when it carries one."""
    if ctx.state not in REVIEW_VERDICT_STATES or not ctx.review_body.strip():
        return False, None
    return _store_review_record(
        ctx,
        registry,
        sink,
        event_id=semantic_review_event_id(
            ctx.repo, ctx.pr_number, ctx.review_id, submitted_at=ctx.review_submitted_at
        ),
        kind=RecordKind.REVIEW_VERDICT,
        comment_id=None,
        question_text=f"Review verdict {ctx.state} on PR #{ctx.pr_number}",
        answer_text=ctx.review_body,
        # A verdict is a judgement about the design, not a system event.
        category=QuestionCategory.DESIGN_DECISION,
        tags=_record_tags(RecordKind.REVIEW_VERDICT, review_state=ctx.state),
        extra_metadata={},
        # The body was sanitised above, before ``extract_agent_claim`` read it, so
        # the removals are handed over rather than recomputed -- recomputing from
        # the already-cleaned text would report nothing.
        prose_fields=(ctx.review_body_field,),
    )


def _store_inline_comments(
    ctx: _ReviewContext,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    comments: list[dict],
) -> tuple[int, list[KnowledgeDeliveryOutcome]]:
    """Store each inline comment anchored to the code it discusses."""
    claims_attempted = 0
    outcomes: list[KnowledgeDeliveryOutcome] = []
    for comment in comments:
        comment_id = comment.get("id")
        if not isinstance(comment_id, int):
            continue
        body = comment.get("body") or ""
        if not body.strip():
            continue
        # The path is somebody else's string and it is displayed on its own: in the
        # anchor below, in the record's own question text, and in the console. A
        # right-to-left override in it renders as a *different file*, which is the one
        # place a reviewer could point at a line they never commented on. Sanitised
        # before the anchor is built rather than after, so the two cannot disagree
        # about which file the record says it is about.
        path_field = _sanitised(comment.get("path"))
        hunk_field = _sanitised(comment.get("diff_hunk"))
        path = path_field.text if path_field is not None else comment.get("path")
        # Absent rather than empty, as it was before: a comment with no diff hunk and a
        # comment whose hunk was nothing but invisible controls are different facts.
        hunk = hunk_field.text or None if hunk_field is not None else None
        line = comment.get("line") or comment.get("original_line")
        anchor = f"{path}:{line}" if path and line is not None else None
        attempted, outcome = _store_review_record(
            ctx,
            registry,
            sink,
            event_id=semantic_review_event_id(
                ctx.repo, ctx.pr_number, ctx.review_id, comment_id=comment_id
            ),
            kind=RecordKind.INLINE_REVIEW_COMMENT,
            comment_id=comment_id,
            question_text=(
                f"Inline review comment on {anchor}" if anchor else "Inline review comment"
            ),
            answer_text=body,
            category=QuestionCategory.EDGE_CASE,
            tags=_record_tags(RecordKind.INLINE_REVIEW_COMMENT),
            # The path and the hunk, so one note covers all three fields. ``_store``
            # re-sanitises the body, which is free and keeps the answer text the
            # single place the policy is guaranteed to have been applied.
            prose_fields=(path_field, hunk_field),
            extra_metadata={
                "path": path,
                "line": line,
                "original_line": comment.get("original_line"),
                "side": comment.get("side"),
                "commit_id": comment.get("commit_id"),
                "diff_hunk": hunk,
                "in_reply_to_id": comment.get("in_reply_to_id"),
                "anchor": anchor,
            },
        )
        claims_attempted += 1 if attempted else 0
        if outcome is not None:
            outcomes.append(outcome)
    return claims_attempted, outcomes


def _review_capture_result(
    outcomes: list[KnowledgeDeliveryOutcome], claims_attempted: int
) -> ReviewCaptureResult:
    """Assemble the outcome, distinguishing silence from duplication."""
    return ReviewCaptureResult(
        outcomes=outcomes,
        # Every record this event could have written was refused because it is
        # already in the store: a re-serialised review, not silence. Recorded here
        # because this is the only point where the two are still tellable apart.
        already_stored=claims_attempted > 0 and not outcomes,
        # None when it stored something, and none for a duplicate: a record that is
        # already held is not a refusal, it is an answer. The silent case is the one
        # that needed naming — every gate above returned before this point, so
        # reaching here with nothing attempted means every candidate body was empty
        # or over the bound, which is an absence of capturable content rather than a
        # policy declining to record it.
        refusal=None if outcomes or claims_attempted else REVIEW_REFUSAL_NOTHING_CAPTURABLE,
    )


def process_review_event_outcome(
    *,
    repo: str,
    pr_number: int,
    review_id: int,
    review_state: str,
    review_body: str,
    review_author: str | None,
    #: The login that opened the PR, i.e. the author of the change under review.
    #: Used for two things, and they are not the same thing: it is an input to
    #: computing independence, and it is stored on the record under
    #: ``change_author_account``. It is recorded rather than derived from the
    #: resulting label because the label is lossy — ``INDEPENDENT`` collapses
    #: "a different account" and "a different account that is also the author of
    #: the change's parent issue", and only the raw login lets a reader tell
    #: those apart. Absent when GitHub reports no user, and then the key is
    #: simply not written.
    pr_author_account: str | None,
    review_submitted_at: datetime | None,
    review_author_association: str | None,
    comments: list[dict],
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    delivery_id: str | None = None,
    #: When the change under review was opened, as GitHub reports it. Distinct
    #: from when the review was submitted, which is what ``answered_at`` already
    #: holds; both are stored because a review months after the change and a
    #: review seconds after it are very different evidence.
    pr_opened_at: datetime | None = None,
    head_sha: str | None = None,
    files_changed: list[str] | None = None,
    #: The association values this run treats as authorised, overriding the default.
    #: ``None`` -- which is what every caller gets unless it deliberately passes
    #: something else -- means :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS`, i.e. **no restriction
    #: at all**. An explicit set means *only those*, which is how an operator
    #: narrows; the two are distinguished by identity rather than truthiness, so an
    #: empty set admits nothing and does not quietly fall back to the default. See
    #: :func:`association_admitted`, which is where that distinction is applied.
    #:
    #: **Narrowing remains an authorisation gate, and it is a decision rather than a
    #: setting to reach for.** It decides whether an account with no relationship to
    #: the repository can cause records to be written, and via the webhook any account
    #: that can comment on a change in an allow-listed repository reaches it. What
    #: changed is only the default: it used to be the narrow set, on evidence that
    #: has since been measured and found to be the wrong way round — see
    #: :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS`, which carries the numbers and should be read
    #: before this is set back to anything.
    authorized_associations: frozenset[str] | None = allowlist.ADMIT_ALL_ASSOCIATIONS,
    #: The account kind GitHub reported for the reviewer, which is what
    #: :func:`is_machine_account` prefers over the login. Optional because not every
    #: caller holds the payload — a login alone still yields an answer, from the
    #: naming convention, which is a weaker signal and is recorded as one.
    review_author_type: str | None = None,
) -> ReviewCaptureResult:
    """Capture a review verdict and its inline comments as review evidence.

    These are stored as their own record kind, never as PR lifecycle entries. A
    lifecycle entry says what the write path did; a review verdict is a person's
    judgement about the code. Conflating them would let a later reader quote a
    tool's own output as though it were a reviewer's decision.

    Every record carries the reviewing principal and an independence level, because
    a verdict from a different account is the cross-check this ledger exists to
    preserve, and a verdict from the same account and model checks nothing. The
    level is computed over (author of the change, reviewer) and the model each
    declared, using the same scale as answer capture, so a caller filtering on
    ``min_independence`` is not comparing two different vocabularies.

    Returns one outcome per record stored. Stored nothing is the normal result for
    an event that carries nothing capturable, and it is what a census record is
    written from — so the return distinguishes it from the other empty case, a
    re-delivered review whose records are already stored, which is a duplicate
    rather than silence. An unauthorised reviewer is no longer one of the empty
    cases under the default policy; under a narrowed one it still is, and it is
    still named.
    """
    resolved = _review_context(
        repo=repo,
        pr_number=pr_number,
        review_id=review_id,
        review_author=review_author,
        review_submitted_at=review_submitted_at,
        delivery_id=delivery_id,
        pr_author_account=pr_author_account,
        review_author_association=review_author_association,
        review_body=review_body,
        review_state=review_state,
        review_author_type=review_author_type,
        pr_opened_at=pr_opened_at,
        head_sha=head_sha,
        files_changed=files_changed,
        authorized_associations=authorized_associations,
    )
    if isinstance(resolved, ReviewCaptureResult):
        return resolved
    ctx = resolved

    outcomes: list[KnowledgeDeliveryOutcome] = []
    # How many records this event tried to write, whatever came of them. Counted at
    # the claim rather than at the store, so a record the claim refused still counts
    # as an attempt: the difference between "nothing to say" and "already said" is
    # exactly this number, and an observation is written for the first only.
    claims_attempted = 0

    # The verdict itself, when the review carries one.
    attempted, outcome = _store_review_verdict(ctx, registry, sink)
    claims_attempted += 1 if attempted else 0
    if outcome is not None:
        outcomes.append(outcome)

    # Each inline comment, anchored to the code it discusses.
    inline_attempted, inline_outcomes = _store_inline_comments(ctx, registry, sink, comments)
    claims_attempted += inline_attempted
    outcomes.extend(inline_outcomes)
    return _review_capture_result(outcomes, claims_attempted)


def process_comment_reply(
    new_comment_id: int,
    new_comment_body: str,
    new_comment_author: str | None,
    new_comment_created_at: datetime | None,
    parent_comment_id: int,
    repo: str,
    pr_number: int,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    question_id: str | None = None,
    new_comment_author_association: str | None = None,
    parent_agent_claim: AgentClaim | None = None,
    new_comment_author_type: str | None = None,
) -> bool:
    return (
        process_comment_reply_outcome(
            new_comment_id=new_comment_id,
            new_comment_body=new_comment_body,
            new_comment_author=new_comment_author,
            new_comment_created_at=new_comment_created_at,
            parent_comment_id=parent_comment_id,
            repo=repo,
            pr_number=pr_number,
            registry=registry,
            sink=sink,
            question_id=question_id,
            new_comment_author_association=new_comment_author_association,
            parent_agent_claim=parent_agent_claim,
            new_comment_author_type=new_comment_author_type,
        )
        is not None
    )


def _close_outcome(
    *, action: str, pr_state: str, pr_merged_at: datetime | None, pr_closed_at: datetime | None
) -> str | None:
    """How a closed change ended, from GitHub's two ways of saying "closed".

    ``state`` is ``"closed"`` for a merge and for an abandonment alike, so the
    only thing that separates them is ``merged_at``. A change that was never
    closed has no outcome at all, which is why this returns ``None`` rather than
    a third value: "merged" and "closed without merging" are both claims about a
    close event, and asserting either of them about an ``opened`` record would be
    asserting something GitHub did not say.
    """
    if action != "closed" and pr_state.strip().lower() != "closed":
        return None
    return PR_OUTCOME_MERGED if pr_merged_at is not None else PR_OUTCOME_CLOSED_UNMERGED


def process_pr_state_change_outcome(
    action: str,
    pr_number: int,
    pr_title: str,
    pr_state: str,
    pr_closed_at: datetime | None,
    repo: str,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    delivery_id: str | None = None,
    event_identity: str | None = None,
    pr_merged_at: datetime | None = None,
    pr_opened_at: datetime | None = None,
    pr_author_account: str | None = None,
    head_sha: str | None = None,
    files_changed: list[str] | None = None,
) -> KnowledgeDeliveryOutcome | None:
    """Store one deduplicated pull request lifecycle transition.

    ``answered_at`` is when the *transition* happened, which is a different fact
    for each action. A close is timestamped by the close; an open, by the open.
    It used to fall through to ``datetime.now(UTC)`` whenever no close time was
    supplied, which meant an ``opened`` record was stamped at the moment the
    webhook was processed — a fact about the write path, filed as a fact about
    the change, and one that differs from the truth by however long delivery took.
    ``captured_at`` already records the write path's own clock, so the two are
    not competing for the same field.
    """
    if not repo or pr_number == 0:
        return None
    if action not in ["opened", "closed", "reopened"]:
        return None

    # The title is somebody else's prose and becomes this record's question text, so it
    # goes through the same boundary as every other captured string -- and through the
    # *wider* pass rather than the marker extractors' narrower one, because there is
    # no marker here to leave alone and the refused set is a superset, so nothing the
    # narrower pass would have removed survives this one. Sanitising after the action
    # gate rather than before it keeps the guard on what the forge said.
    #
    # The removals are recorded, because a lifecycle record whose title was rewritten
    # is otherwise a record whose title somebody else wrote.
    title_field = sanitise(pr_title)
    pr_title = title_field.text
    title_note = title_field.note()

    event_id = event_identity or delivery_id
    if event_id is None:
        transition_time = pr_closed_at.isoformat() if pr_closed_at else "unknown"
        event_id = f"legacy:{repo}:{pr_number}:{action}:{transition_time}"
    claim_token = registry.claim_pr_state_change(repo, pr_number, action, event_id)
    if claim_token is None:
        return None

    # A reopen is neither an open nor a close, so the open time is not a fallback
    # for it: GitHub reports no reopen timestamp, and the one below is a
    # declaration that nothing better is known rather than a claim that the change
    # was opened then.
    transition_at = pr_closed_at or (pr_opened_at if action == "opened" else None)
    close_outcome = _close_outcome(
        action=action, pr_state=pr_state, pr_merged_at=pr_merged_at, pr_closed_at=pr_closed_at
    )
    entry = KnowledgeEntry(
        entry_id=f"pr-event-{event_id}",
        session_id=None,
        question_text=f"PR {action}: {pr_title}",
        answer_text=f"PR state changed to {pr_state}",
        category=QuestionCategory.SYSTEM_EVENT,
        context=None,
        author="system",
        answered_at=transition_at or datetime.now(UTC),
        embedding=None,
        tags=_record_tags(RecordKind.PR_LIFECYCLE, action=action),
        metadata={
            "repo": repo,
            "pr_number": pr_number,
            "pr_url": f"https://github.com/{repo}/pull/{pr_number}",
            RECORD_KIND_KEY: RecordKind.PR_LIFECYCLE.value,
            CHANGE_AUTHOR_KEY: pr_author_account,
            PR_OPENED_AT_KEY: pr_opened_at.isoformat() if pr_opened_at else None,
            # The merge is not stored in ``answered_at`` because a merge is a
            # different event from the close this record is about: the same pull
            # request can be closed at one time and merged at another, and a
            # reader sorting this record by time would place the merge at the
            # close. It is also not derivable from the outcome, which is a
            # claim rather than a time, so both endpoints are kept and the
            # arithmetic between them is left to whoever wants it.
            PR_OUTCOME_KEY: close_outcome,
            PR_MERGED_AT_KEY: pr_merged_at.isoformat() if pr_merged_at else None,
            HEAD_SHA_KEY: head_sha,
            FILES_KEY: list(files_changed) if files_changed else None,
            "delivery_id": delivery_id,
            **({SANITISATION_KEY: title_note} if title_note else {}),
        },
        # A lifecycle transition is a real provider delivery, so it claims
        # ``webhook`` and carries the delivery id that produced it.
        capture_source=CaptureSource.WEBHOOK,
        captured_at=datetime.now(UTC),
        capture_delivery_id=delivery_id or event_id,
    )

    try:
        outcome = _store_outcome(sink, entry)
        if outcome.dead_lettered:
            if not registry.release_pr_state_change(
                event_id, claim_token, outcome.detail or "delivery dead-lettered"
            ):
                raise RuntimeError("Failed to release dead-lettered PR capture claim")
            return outcome
        if not registry.complete_pr_state_change(event_id, claim_token):
            raise RuntimeError("Failed to complete PR capture claim")
    except Exception as exc:
        registry.release_pr_state_change(event_id, claim_token, type(exc).__name__)
        raise
    return outcome


@dataclass
class _AnswerContext:
    """Everything one answer's guards derive before anything is claimed.

    Same shape as :class:`_ReviewContext` for the same reason: the writer
    below closed over a dozen locals, and named fields say which of them the
    write needs.
    """

    author_association: str | None
    answer: SanitisedText
    sanitisation_note: str
    question: dict[str, Any]
    registered_question_id: str
    agent_id: str | None
    answered_by_model: str | None
    rationale_source: RationaleSource
    independence: Independence
    independence_reason: str


def _answer_context(
    *,
    new_comment_id: int,
    new_comment_body: str,
    new_comment_author: str | None,
    parent_comment_id: int,
    repo: str,
    pr_number: int,
    registry: QuestionRegistry,
    question_id: str | None,
    new_comment_author_association: str | None,
    parent_agent_claim: AgentClaim | None,
) -> _AnswerContext | None:
    """Run the admission gates, or ``None`` for an ordinary non-answer.

    ``None`` covers every refusal the same way: an unassociated comment, an
    empty body, a marker mismatch, a duplicate delivery and a question that is
    already answered are all comments this function was not asked to store,
    and the caller reports them identically.
    """
    author_association = (
        new_comment_author_association.strip().upper() if new_comment_author_association else None
    )
    if not allowlist.association_admitted(author_association, allowlist.ADMIT_ALL_ASSOCIATIONS):
        return None
    if not new_comment_body.strip():
        return None

    marker_question_id = extract_answer_question_id_from_comment_body(new_comment_body)
    requested_question_id = question_id or marker_question_id
    if marker_question_id and requested_question_id and marker_question_id != requested_question_id:
        return None
    answer_text = extract_answer_text_from_comment_body(new_comment_body)
    # Two passes, and neither is the other. The extractor is what *reads* the comment:
    # it parses the ``kojutsu:`` grammar verbatim, so a mangled marker fails closed
    # rather than being repaired into something this system did not write, and it
    # normalises the marker's value. :func:`sanitise` is what *stores* it, and it is
    # not repeated for the extractor's sake -- a second call site for one set of
    # characters is a second decision about them, and the two drift.
    #
    # The note is measured against the delivered comment rather than against
    # ``answer``, because ``answer`` has already been through the extractor's own
    # narrower pass and a note computed from it would name only the characters this
    # module took. See :func:`kojutsu.core.text_hygiene.describe_removals`, which is
    # the whole reason that function exists.
    answer = sanitise(answer_text)
    sanitisation_note = describe_removals(new_comment_body)
    # Tested after sanitising, so a comment whose whole content is invisible controls
    # lands on the extractor's own outcome -- nothing capturable -- rather than
    # producing a record that counts as knowledge and reads as nothing.
    if len(new_comment_body) > MAX_ANSWER_CHARS or not answer.text.strip():
        return None

    if registry.answer_comment_seen(new_comment_id):
        return None
    if registry.is_question_answered(parent_comment_id):
        return None

    question = registry.get_question_by_comment_id(parent_comment_id)
    if not question:
        return None
    registered_question_id = question.get("question_id")
    if requested_question_id and registered_question_id != requested_question_id:
        return None
    if not registered_question_id:
        return None
    agent_claim = extract_agent_claim(new_comment_body)
    agent_id = agent_claim.agent_id if agent_claim is not None else None
    answered_by_model = agent_claim.model if agent_claim is not None else None
    # How the reasoning was obtained, carried through from the author's own marker.
    # Absent on comments posted before the key existed, and then reported as
    # unknown rather than guessed at: a reader must not invent provenance the
    # writer never claimed, which is the same rule the unstated model follows.
    rationale_source = agent_claim.source if agent_claim is not None else RationaleSource.UNKNOWN
    # The posting account asked the question and is answering it. That is only
    # acceptable when the comment declares an agent identity: the record is then
    # explicitly attributed to a named machine principal rather than passing off a
    # self-assessment as independent review. A human answering their own question is
    # still refused, because it adds no independent evidence.
    same_account_as_asker = (
        bool(question.get("question_author"))
        and question.get("question_author") == new_comment_author
    )
    if same_account_as_asker and not agent_id:
        return None

    # Independence is computed here, from the two comments as they exist on the
    # forge, rather than from anything cached at question time. The asker's claim is
    # read off the parent comment body by the caller, so a later edit to either
    # comment cannot silently change the label on a record already stored.
    asker_claim: AgentClaim | None = parent_agent_claim
    independence, independence_reason = compute_independence(
        asker_account=question.get("question_author"),
        asker_model=asker_claim.model if asker_claim is not None else None,
        answerer_account=new_comment_author,
        answerer_model=answered_by_model,
    )
    return _AnswerContext(
        author_association=author_association,
        answer=answer,
        sanitisation_note=sanitisation_note,
        question=question,
        registered_question_id=registered_question_id,
        agent_id=agent_id,
        answered_by_model=answered_by_model,
        rationale_source=rationale_source,
        independence=independence,
        independence_reason=independence_reason,
    )


def _build_answer_entry(
    ctx: _AnswerContext,
    *,
    repo: str,
    pr_number: int,
    new_comment_id: int,
    new_comment_author: str | None,
    new_comment_created_at: datetime | None,
    new_comment_author_type: str | None,
    entry_id: str,
    category: QuestionCategory,
    delivery_id: str | None,
) -> KnowledgeEntry:
    """Build the answer record; claiming and delivery stay with the caller."""
    question = ctx.question
    return KnowledgeEntry(
        entry_id=entry_id,
        session_id=question.get("session_id"),
        question_text=question.get("question_text", ""),
        answer_text=ctx.answer.text,
        category=category,
        context=None,
        # A declared agent is the author of the answer, whoever posted the comment.
        # A reader must be able to see that this was machine-authored.
        author=ctx.agent_id or new_comment_author,
        answered_at=new_comment_created_at or datetime.now(UTC),
        embedding=None,
        tags=_record_tags(RecordKind.ANSWER, agent_authored=bool(ctx.agent_id)),
        metadata={
            "repo": repo,
            "pr_number": pr_number,
            "pr_url": question.get("pr_url", ""),
            "jira_ticket_key": question.get("jira_ticket_key"),
            "github_comment_id": new_comment_id,
            # What the forge said about this account's standing, kept because the gate
            # that used to consume it is gone and the fact is what a reader weighs
            # instead. An answer from a stranger is now stored on the strength of the
            # marker, the question and the independence level, and this is how anyone
            # reading it can tell that is what happened.
            "github_author_association": ctx.author_association,
            "question_id": ctx.registered_question_id,
            RECORD_KIND_KEY: RecordKind.ANSWER.value,
            "answered_by_agent": ctx.agent_id,
            # Absent, not ``"unknown"``, when the comment stated no model. See
            # the note in the review writer: a document in which a principal
            # nobody named appears to have declared a model called ``"unknown"``
            # cannot be told apart from one where somebody did.
            "answered_by_model": ctx.answered_by_model,
            # Recorded whether or not an agent claimed anything, so a reader can
            # tell an answer that states its own basis from one that does not.
            "rationale_source": ctx.rationale_source.value,
            "comment_author": new_comment_author,
            # Whether the posting account is an application. Recorded on every answer
            # because the association gate no longer filters automation out, and
            # refusing to write it down would leave a reader with a corpus that mixes
            # bot and human answers and no way to tell which is which except by
            # matching on a login. Provenance, not a judgement: a bot's answer is still
            # an answer and the point of recording this is that the reader can weigh
            # the two differently. Same signal and same key as the review, rationale
            # and clarification paths -- see :data:`COMMENT_AUTHOR_IS_MACHINE_KEY`.
            COMMENT_AUTHOR_IS_MACHINE_KEY: is_machine_account(
                new_comment_author, account_type=new_comment_author_type
            ),
            # Read off the registry row, not off this delivery. See the docstring.
            HEAD_SHA_KEY: question.get("head_sha"),
            "independence": ctx.independence.value,
            "independence_reason": ctx.independence_reason,
            # What the character policy took out of this answer, in the record's own
            # words. Present only when something was removed, so its absence is the
            # claim that this text is byte-for-byte what the contributor wrote -- which
            # is otherwise indistinguishable from a record that was quietly rewritten
            # at the boundary. See
            # :data:`kojutsu.core.text_hygiene.SANITISATION_KEY`.
            **({SANITISATION_KEY: ctx.sanitisation_note} if ctx.sanitisation_note else {}),
        },
        # This entry came from a real comment, so it is evidence rather than an
        # assertion. The channel records *how* it arrived: a signed provider
        # delivery, or an authenticated API read (``kojutsu collect``).
        capture_source=(CaptureSource.WEBHOOK if delivery_id else CaptureSource.COLLECT),
        captured_at=datetime.now(UTC),
        capture_delivery_id=delivery_id,
    )


def process_comment_reply_outcome(
    new_comment_id: int,
    new_comment_body: str,
    new_comment_author: str | None,
    new_comment_created_at: datetime | None,
    parent_comment_id: int,
    repo: str,
    pr_number: int,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    question_id: str | None = None,
    new_comment_author_association: str | None = None,
    delivery_id: str | None = None,
    parent_agent_claim: AgentClaim | None = None,
    #: The account kind GitHub reported for the commenter, which is what
    #: :func:`is_machine_account` prefers over the login. See that function for why a
    #: caller holding the payload should pass it and what a login alone can and cannot
    #: answer.
    new_comment_author_type: str | None = None,
) -> KnowledgeDeliveryOutcome | None:
    """Capture an explicitly associated answer to a question kojutsu asked.

    **Who may answer is no longer filtered by ``author_association``.** It used to
    be, hardcoded, which is why this function is one of the five sites the constant
    lived at and why one policy edit had to touch all of them. It now applies
    :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` — unrestricted — and records what the association
    was, because the fact outlived the rule. The path has no override parameter: it
    never had one, and the review path's flag is a backfill-run setting with no
    webhook-side configuration to hang it on. An operator who needs a narrower policy
    has it on the review and clarification paths and not here, which is a real
    asymmetry and is stated rather than hidden behind a knob nothing can set.

    **The record is anchored to the ask-time commit, not the head at arrival.**
    Those are two different facts and only the first describes what the answer is
    about: this function is reached from a later ``issue_comment`` delivery, by
    which time the branch may have moved on, and a question put against commit A
    and answered after the head reached B is a statement about A. Writing B instead
    would make the record wrong in exactly the case the system is built to notice --
    ``supersede_questions_for_pr`` exists to mark a question superseded when the
    head moves, so a moved head is tracked rather than incidental -- and it would do
    so invisibly, by overwriting the drift a reader would otherwise have seen.

    So the sha comes from the question's registry row, which
    :meth:`~kojutsu.core.question_registry.SqliteQuestionRegistry.record_question`
    wrote when the question was asked. It is absent rather than filled in for a
    question that predates schema v7, or one asked by a caller that had no head to
    record: there is no fallback to this delivery's head and none to the pull
    request's current head, because both are assertions about a different commit and
    a plausible sha on an answer reads as an anchor rather than a reconstruction.

    For the same reason the record carries no ``files``. The file list that would
    make it as useful as a review's is the list as of the ask-time head, this
    delivery has none, and obtaining one is a second forge call naming a commit only
    the question row knows. The review path already fetches and bounds ``files``
    (:func:`process_review_event_outcome`); a second fetch per pull request to fill
    this key would describe code the reviewer was not reading, filed beside an anchor
    claiming otherwise.
    """
    resolved = _answer_context(
        new_comment_id=new_comment_id,
        new_comment_body=new_comment_body,
        new_comment_author=new_comment_author,
        parent_comment_id=parent_comment_id,
        repo=repo,
        pr_number=pr_number,
        registry=registry,
        question_id=question_id,
        new_comment_author_association=new_comment_author_association,
        parent_agent_claim=parent_agent_claim,
    )
    if resolved is None:
        return None
    ctx = resolved

    entry_id = stable_answer_entry_id(repo, pr_number, new_comment_id)
    claim_token = registry.claim_answer(
        parent_comment_id=parent_comment_id,
        answer_comment_id=new_comment_id,
        repo=repo,
        pr_number=pr_number,
        question_id=ctx.registered_question_id,
        entry_id=entry_id,
    )
    if claim_token is None:
        return None

    category_str = ctx.question.get("category") or "design_decision"
    try:
        category = QuestionCategory(category_str)
    except ValueError:
        category = QuestionCategory.DESIGN_DECISION

    entry = _build_answer_entry(
        ctx,
        repo=repo,
        pr_number=pr_number,
        new_comment_id=new_comment_id,
        new_comment_author=new_comment_author,
        new_comment_created_at=new_comment_created_at,
        new_comment_author_type=new_comment_author_type,
        entry_id=entry_id,
        category=category,
        delivery_id=delivery_id,
    )

    try:
        outcome = _store_outcome(sink, entry)
        if outcome.dead_lettered:
            if not registry.release_answer(new_comment_id, claim_token):
                raise RuntimeError("Failed to release dead-lettered answer capture claim")
            return outcome
        if not registry.complete_answer(new_comment_id, claim_token):
            raise RuntimeError("Failed to complete answer capture claim")
    except Exception:
        registry.release_answer(new_comment_id, claim_token)
        raise
    return outcome


CHECK_RUN_KIND = "check_run"
MAX_CHECK_SUMMARY_CHARS = 20_000


def process_check_run_outcome(
    *,
    repo: str,
    check_run_id: int,
    check_name: str,
    check_status: str,
    check_conclusion: str | None,
    head_sha: str | None,
    pr_number: int | None,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    delivery_id: str | None = None,
) -> KnowledgeDeliveryOutcome | None:
    """Store one check run's conclusion as a machine report about a commit.

    A check run is the nearest thing GitHub offers to "this change turned out
    wrong", and it is worth having. What it is *not* is a verdict on the change:
    a red build says a check failed, and a consumer deciding what that means about
    the quality of the work is making an inference Kojutsu has no standing to
    make. So the conclusion is stored verbatim, in its own record kind, and the
    text says "concluded failure" rather than "failed" — the difference is the
    difference between a fact about a tool and a judgement about a person.

    Only a *concluded* run is stored. A run still in progress says nothing yet,
    and storing every transition would rewrite the same document several times per
    run on a delivery path built for immutable records.
    """
    if not repo or not check_run_id:
        return None
    conclusion = (check_conclusion or "").strip().lower()
    if not conclusion:
        return None

    # The forge's own id is the identity: a re-run is a new check run with a new
    # id, so this does not collapse a re-run into the run it replaced.
    event_id = f"check-run:{repo.casefold()}:{check_run_id}"
    # Sanitised here rather than at the model because this is the only writer in the
    # capture path that takes a bare string for a name it stores: the check name
    # becomes the record's author, its title and the first line of its body, so it is
    # displayed three times and is the field a forged check name would attack. A name
    # carrying a zero-width or bidirectional control would render in the console as
    # somebody else's.
    #
    # Recorded as sanitised, because an author field is a claim about who wrote
    # something and "the author's name had characters taken out of it" is not a claim
    # anybody made.
    check_name_field = sanitise(check_name)
    check_name = check_name_field.text
    check_name_note = check_name_field.note()
    claim_token = registry.claim_review_capture(
        review_event_id=event_id,
        repo=repo,
        pr_number=pr_number or 0,
        review_id=check_run_id,
        comment_id=None,
        kind=CHECK_RUN_KIND,
        author=check_name,
    )
    if claim_token is None:
        return None

    entry = KnowledgeEntry(
        entry_id=stable_check_entry_id(event_id),
        session_id=None,
        question_text=f"Check {check_name} concluded {conclusion}",
        answer_text=(
            f"The check {check_name!r} reported status {check_status or 'unknown'!r} "
            f"and conclusion {conclusion!r}."
            + (f" It ran against commit {head_sha}." if head_sha else "")
            + " A check reports on a commit. It is not a judgement about the change."
        ),
        category=QuestionCategory.SYSTEM_EVENT,
        context=None,
        # The check, not a person. Naming a tool as the author of a record is
        # honest; pretending the change's author asserted it is not.
        author=check_name or "check",
        answered_at=datetime.now(UTC),
        embedding=None,
        tags=_record_tags(RecordKind.CHECK_RUN, check_conclusion=conclusion),
        metadata={
            "repo": repo,
            "pr_number": pr_number,
            "pr_url": f"https://github.com/{repo}/pull/{pr_number}" if pr_number else None,
            RECORD_KIND_KEY: RecordKind.CHECK_RUN.value,
            HEAD_SHA_KEY: head_sha,
            "check_id": check_run_id,
            "check_name": check_name,
            "check_status": check_status,
            # Verbatim, and named as a report rather than a result.
            "check_conclusion": conclusion,
            "delivery_id": delivery_id,
            **({SANITISATION_KEY: check_name_note} if check_name_note else {}),
        },
        # A check run delivered by the forge is a real delivery and is checkable
        # against it, so the trust axis applies exactly as it does to a review.
        capture_source=CaptureSource.WEBHOOK,
        captured_at=datetime.now(UTC),
        capture_delivery_id=delivery_id or event_id,
    )
    try:
        outcome = _store_outcome(sink, entry)
        if outcome.dead_lettered:
            if not registry.release_review_capture(
                event_id, claim_token, outcome.detail or "check capture dead-lettered"
            ):
                raise RuntimeError("Failed to release dead-lettered check capture claim")
        elif not registry.complete_review_capture(event_id, claim_token):
            raise RuntimeError("Failed to complete check capture claim")
    except Exception as exc:
        registry.release_review_capture(event_id, claim_token, type(exc).__name__)
        raise
    return outcome
