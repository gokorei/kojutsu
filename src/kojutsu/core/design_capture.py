"""The recording half of the design phase: proposals in, a validated plan out.

:mod:`kojutsu.core.design_plan` gave a reconciled plan a shape and a validator. This
module is what puts the things around that plan into the ledger, and it exists
because of which step this ticket is most afraid of: reconciling several proposals
into one plan *"is where a single model judging other models is least trustworthy,
and it is exactly the step that decides what gets built."* A proposal that lives in a
chat transcript is evidence of nothing except that a transcript exists. A proposal
that is a **record** has a principal, a model, a moment, and an identity a later
reader can re-derive from the anchor rather than trusting.

**A proposal is a stated rationale, and reusing that model is the design rather than a
convenience.** A proposal *is* a stated reason for a design decision: a principal's
argued position, with its author and its model. :class:`kojutsu.models.RationaleEntry`
is exactly "one stated reason for a decision, as a record in its own right", and
reusing it buys the properties this phase cannot afford to re-derive:

- ``RationaleEntry`` refuses to be anything but ``ASSERTED``. A proposal is written
  by a model that has just read the repository, so it is exactly the record kind that
  must never be servable as capture evidence, and that refusal is a validator on the
  model rather than a convention in this module.
- ``declared_by`` / ``declared_model`` / ``revision`` / ``revises`` are the
  append-never-overwrite provenance a proposal needs, and they are already the
  provenance the reader of every other declaration is shown.
- ``claim_rationale`` gives dedupe, a lease, and complete-or-release **without a new
  registry table**, and the durable outbox plus
  :func:`kojutsu.core.knowledge_sink.to_payload` gives delivery, **without a new
  document renderer**. Both of those files belong to other phases. A new record kind
  would have to be added to :func:`to_payload`, to
  :func:`kojutsu.core.tanseki_mapping.rationale_document_id`'s neighbourhood and to
  a new schema migration to be reachable at all -- so a design record invented "for
  clarity" would be a record nothing delivers, which is worse than a record whose
  kind is shared.

What the design-specific facts need is a place, and the place is ``metadata``. See
:data:`DESIGN_ROLE_KEY` for what does and does not survive that boundary.

**What the independence check is worth, stated permanently because it is the whole
point of the module.** The ticket requires that the reconciler principal differs from
every proposer principal, and that the difference be *enforced*. It is enforced, in
:func:`reconcile_design_proposals`, by refusing before anything is read or written.
But an ``agent_id`` in this system is **self-declared and never verified**: it is
length-bounded and text-normalised by
:func:`kojutsu.integrations.github.extract_agent_claim`, and nothing checks it against
anything. So the check compares two strings the participants typed about themselves.

What that buys, precisely: it catches the accidental and the sloppy. A reconciler
that reuses the proposer's own configured principal -- the overwhelmingly likely way
this goes wrong, because one process runs several agents and one identifier is right
there -- is refused. A reconciler invoked under ``"Opencode"`` while a proposer
declared ``"opencode"`` is refused, because the comparison is case- and
whitespace-insensitive (see :func:`kojutsu.models.compute_independence` for the
rule, reused here rather than re-derived).

What it does not buy: anything about a determined self-approval. Two principals that
differ as strings may be one agent, and an agent that wants to grade its own homework
has only to spell its name differently. Nothing in this system could catch that: the
proposer and the reconciler are the same unverified self-declaration on both sides,
so a check strong enough to catch it would require an authenticated identity the
capture path does not have and cannot invent. So the honest claim is not "the
reconciler is independent" but **"the reconciler was not any of the proposers under
the names it declared"**, and that sentence is written into every reconciliation
record by :data:`INDEPENDENCE_LIMITATION_NOTE` -- not only into this docstring,
because the reader who is misled will be reading the record, not the source.

:mod:`kojutsu.core.rationale_collector` already says the related thing about a
rationale: the platform proves who posted a comment and never which model drafted it.
A design proposal is weaker than that, because there is no platform step at all.

**The topic rides in the branch column, and that is a real cost.** A design proposal
belongs to no pull request and no branch -- the plan it feeds is what creates
branches -- so ``pr_number`` is ``None`` and ``branch`` has no honest value to hold.
But the topic is what makes one proposal distinct from another, and
``rationale_captures`` carries its uniqueness index over
``(repo, COALESCE(pr_number, -1), branch, declared_by, revision)`` and nothing else.
Derive the id from the topic and claim with an empty branch and the *second* topic a
proposer works on is refused by that index as a duplicate of the first, silently,
which is the loss of a genuine proposal in exchange for a tidier field. So the topic
is stored in ``branch``, which is the one anchor column that can hold it, and the
consequence is named rather than left for a reader to infer from a document:
``branch`` on a design record is a topic, not a git ref, and there is nothing to
re-fetch. The alternative -- a new capture table keyed on the topic -- is a schema
migration in a file this phase does not own.

**Idempotency is a claim on an anchor, not a key somebody invented.** Both captures
derive an id from a semantic anchor that excludes the text (so a rephrasing is the
same record rather than an orphan plus a duplicate, per
:func:`kojutsu.core.question_registry.stable_rationale_entry_id`) and then claim it
through the registry before writing. Re-running either with the same inputs loses the
claim and stores nothing; the registry, not an API key, is what says so. The
precedent is ``claim_rationale`` / ``complete_rationale`` and this module follows
the whole shape around it, which means the tail of
:func:`capture_design_proposal` is a second copy of the tail of
:func:`kojutsu.core.rationale_collector.process_rationale_comment_outcome`.
Factoring it out means editing a file this phase does not own; if that tail is ever
changed, change both.

**The reconciliation records what it discarded, because that is the half a reader
needs.** The ticket is explicit that discarding a proposal is as much a decision
worth keeping as accepting one. So every discard is stored with its reason, twice:
as prose in ``rationale_text``, which reaches the stored document, and as structured
data in ``metadata``. What it deliberately does **not** record is which proposals
were *accepted*: the plan is that record, and a proposal that was read and
contributed nothing is indistinguishable here from one that was never read at all.
Saying so is better than a field that claims otherwise.

**Reading a reconciliation back, and the one thing that makes it a refusal.**
:func:`find_design_reconciliations` exists because
:func:`stable_design_reconciliation_id` is a derivation and a later invocation has no
principal and revision to derive from -- see that function for why. What it returns is
the *identity* of a recorded reconciliation: who reconciled, at which revision, and
**which plan**, read back out of the stored prose because ``rationale_captures`` has no
metadata column (see :data:`PLAN_DIGEST_LINE_PREFIX`).

That last field is what makes the lookup refuse rather than answer. A resume that
*spends* an approval has to know which plan it is spending it on, and two recorded
reconciliations for one topic is a real state -- a second revision, or a second
reconciler -- not a corruption. So
:func:`require_single_design_reconciliation` refuses anything but exactly one, naming
every candidate's principal, revision and digest, for the reason the module's own
identity argument gives: a silent choice here is indistinguishable afterwards from a
plan nobody approved being created.

**The character policy applies to model prose, for the same reason it does on the
rationale path.** A proposer has just read the repository and the reconciler has just
read the proposals, so both are quoting attacker-influenced text back at us, and the
removals are recorded under :data:`kojutsu.core.text_hygiene.SANITISATION_KEY` rather
than performed silently. It is *not* applied to a principal, a model or a topic:
those are hashed into the identity, so cleaning them here would let two principals
that differ only by an invisible glyph derive one id, and one of them would be lost.
They are **refused** instead, which is loud, loses nothing and re-identifies nothing.

**What this phase cannot do.** Nothing here makes a reconciler model choose well.
Recording the proposals and the reconciliation makes the choice auditable after the
fact, which is the available mitigation and not a guarantee -- and the reason the
independence constraint is enforced as a hard refusal rather than a confidence score
is that a score is exactly the kind of thing a determined self-approval would carry.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kojutsu.core.design_keys import (
    DESIGN_DISCARDED_KEY,
    DESIGN_PLAN_DIGEST_KEY,
    DESIGN_PROPOSAL_IDS_KEY,
    DESIGN_ROLE_KEY,
    DESIGN_ROLE_PROPOSAL,
    DESIGN_ROLE_RECONCILIATION,
)
from kojutsu.core.design_plan import DesignPlan, parse_design_plan
from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryOutcome,
    KnowledgeDeliveryStatus,
    KnowledgeSink,
)
from kojutsu.core.question_registry import (
    MAX_QUESTION_LIST_LIMIT,
    QuestionRegistry,
)
from kojutsu.core.rationale_collector import MAX_RATIONALE_CHARS
from kojutsu.core.text_hygiene import (
    REFUSED_RE,
    SANITISATION_KEY,
    describe_removals,
    sanitise,
)
from kojutsu.identity import identity_preimage
from kojutsu.models import (
    Independence,
    RationaleChannel,
    RationaleEntry,
    RationaleSource,
    compute_independence,
)
from kojutsu.text_limits import MAX_AGENT_ID_CHARS, MAX_BRANCH_CHARS, MAX_MODEL_ID_CHARS

#: Version of the design-proposal identity derivation. Bump this, and keep both
#: derivations, rather than editing :func:`stable_design_proposal_id` in place, for
#: the reason spelled out on
#: :data:`kojutsu.core.question_registry.RATIONALE_IDENTITY_VERSION`: the id is a
#: durable ``UNIQUE`` key in the registry and a path segment in a stored document,
#: so moving the computation re-identifies every stored record and orphans the
#: documents already in the store.
DESIGN_PROPOSAL_IDENTITY_VERSION = 1

#: Domain label separating a design proposal from every other namespace, and from a
#: reconciliation. Part of the hashed preimage, so it both keeps the namespaces
#: apart and makes a change to *how* the digest is computed produce a different value
#: rather than one that looks comparable and is not.
DESIGN_PROPOSAL_IDENTITY_DOMAIN = "kojutsu.design_proposal.v1"

#: Version and domain of the design-reconciliation derivation, for the same two
#: reasons as the proposal's.
DESIGN_RECONCILIATION_IDENTITY_VERSION = 1
DESIGN_RECONCILIATION_IDENTITY_DOMAIN = "kojutsu.design_reconciliation.v1"

#: The prefix every reconciliation's entry id carries, and the only thing that tells a
#: reconciliation apart from a proposal in a registry read.
#:
#: :data:`~kojutsu.core.design_keys.DESIGN_ROLE_KEY` would be the right discriminator
#: and is not available for this: ``rationale_captures`` stores no metadata, so the
#: role lives in the knowledge sink's payload and never reaches the local ledger at
#: all. Both record kinds are :class:`~kojutsu.models.RationaleEntry` on the same
#: repository with the same topic in the same column, so a lookup over a topic's
#: records finds both unless it looks at this.
#:
#: That is still true now that the role reaches the *stored document*: the frontmatter
#: projection lands in the knowledge store, not in the registry, so this prefix stays
#: load-bearing for every local read until ``rationale_captures`` grows a metadata
#: column. The day it does, the honest change is to read the column and delete the
#: prefix logic -- not to keep both and prefer the easier read.
#:
#: It is spelled here, derived from the version, rather than written out -- a literal
#: prefix beside a derivation whose version can be bumped is a prefix that will one day
#: not match. The consequence of the mismatch is the loud kind: a version bump without
#: this following it makes every lookup report no reconciliation, which refuses rather
#: than misbehaving, but the records are unreadable until the constant is updated.
RECONCILIATION_ID_PREFIX = f"design-reconciliation-v{DESIGN_RECONCILIATION_IDENTITY_VERSION}-"

#: The line :func:`_render_reconciliation` writes a reconciliation's plan digest on, and
#: the line :func:`find_design_reconciliations` reads it back from.
#:
#: **A contract between two functions in this file, and it is here because the registry
#: offers nowhere else to put the digest.** :data:`DESIGN_PLAN_DIGEST_KEY` holds it in
#: the record's metadata, which the durable outbox delivers to the knowledge store and
#: ``rationale_captures`` does not store -- so the only copy a local reader can reach is
#: the prose line. Making that line a named constant is what stops the two halves drifting:
#: the renderer cannot reword it without the reader breaking, and the reader cannot move
#: without the renderer breaking, because both compile against the same name.
#:
#: If a later phase gives ``rationale_captures`` a metadata column, this becomes a
#: fallback rather than the mechanism, and the honest change is to delete it -- not to
#: leave two sources for one fact and prefer the one that is easier to read.
PLAN_DIGEST_LINE_PREFIX = "Plan digest: "

#: What one recorded plan digest has to look like to be usable as an identity.
#:
#: Lowercase hex of exactly SHA-256's width, because that is what
#: :func:`design_plan_digest` returns and comparing a digest to a digest requires the
#: same alphabet. A recorded value that is not this is refused rather than normalised --
#: see :func:`_recorded_plan_digest`.
_PLAN_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

#: Named in the "not found" refusal so the message says where the read happened rather
#: than leaving an operator to guess which of the four local stores was consulted.
#:
#: It is the question registry and not the ticket store because that is the store
#: :func:`reconcile_design_proposals` claims into, and it is the store the capture half
#: writes through. An operator looking for a missing reconciliation has to be told to
#: look at the registry and not at the approvals ledger they were probably thinking of.
REGISTRY_STORE_DESCRIPTION = "the local question registry (KOJUTSU_REGISTRY_PATH)"

#: The sentence every reconciliation record carries about what its own independence
#: check is worth.
#:
#: It is in the record rather than only in this module's docstring because the reader
#: who would over-read the constraint is reading the record. "Reconciled by a
#: principal distinct from the proposers" reads as a verification, and it is not
#: one: both names are self-declarations. A record that carries the limitation is
#: auditable; one that carries only the result is a claim this system cannot back --
#: which is the failure :class:`kojutsu.models.RationaleEntry` refuses when it
#: refuses ``capture_source != ASSERTED``, and the same rule at one level up.
INDEPENDENCE_LIMITATION_NOTE = (
    "Independence is enforced over self-declared principals: it refuses a reconciler "
    "that reused a proposer's name, and it cannot refuse one that spelled the same "
    "name differently."
)

#: The most proposers one reconciliation may cover.
#:
#: A reviewability bound, on the same argument as
#: :data:`kojutsu.core.design_plan.MAX_PLAN_TICKETS`: the human gate reads the
#: reconciliation once and then acts on the plan without looking again, so the gate's
#: strength is the number of arguments one person can hold while approving. It also
#: bounds the record, since every proposer can contribute a discard line -- so
#: ``MAX_DESIGN_PROPOSALS * MAX_DISCARD_REASON_CHARS`` is the record's size and no
#: separate total cap is needed.
MAX_DESIGN_PROPOSALS = 16

#: Longest one reason for discarding a proposal.
#:
#: A sentence or two. Past this the reconciler is writing an argument nobody reads at
#: the gate, and the field whose whole job is to be the explanation stops being one.
MAX_DISCARD_REASON_CHARS = 1_000

#: Longest one design topic.
#:
#: :data:`kojutsu.integrations.github.MAX_BRANCH_CHARS` rather than a number chosen
#: here, because the topic is stored in the branch column and that column is that
#: bound's subject everywhere else it is written. A second bound for one field is how
#: two values for the same limit come to disagree.
MAX_DESIGN_TOPIC_CHARS = MAX_BRANCH_CHARS


class ReconcilerNotIndependentError(ValueError):
    """The reconciler is one of the things being reconciled.

    A ``ValueError`` for the reason every other refusal in this codebase is one --
    :class:`kojutsu.core.design_plan.DesignPlanError`,
    :class:`kojutsu.allowlist.AllowlistError`,
    :class:`kojutsu.core.answerer.AnswerSelectionError` -- and it is raised rather
    than returned because the ticket's instruction is that a constraint which cannot
    be satisfied must fail loudly rather than quietly reconcile anyway. A warning, a
    flag on an outcome, or a threshold would all be reconciling with one of the
    proposers while claiming not to.

    Every conflicting proposal is named, not the first one, for the same reason
    :class:`~kojutsu.core.design_plan.DesignPlanError` collects its problems: a
    caller that fixed one conflict and ran again would learn the rest one round at a
    time, and the reconciler principal is configuration, so the fix is one edit.

    The message includes the phrase :func:`kojutsu.models.compute_independence`
    returned, because whether the shared principal also shared the model is the
    difference between a misconfigured runner and one principal wearing two hats.
    """

    def __init__(self, conflicts: Sequence[tuple[str, str, str]]) -> None:
        self.conflicts: tuple[tuple[str, str, str], ...] = tuple(conflicts)
        named = "\n".join(
            f"  - reconciler {reconciler!r} is also the principal behind {label} ({reason})"
            for reconciler, label, reason in self.conflicts
        )
        super().__init__(
            f"Refusing to reconcile: the reconciler is one of the proposers "
            f"({len(self.conflicts)} conflict{'' if len(self.conflicts) == 1 else 's'})\n"
            f"{named}\n  - {INDEPENDENCE_LIMITATION_NOTE}"
        )


@dataclass(frozen=True)
class DesignProposal:
    """One proposer's identity for a design topic, as the anchor is built from it.

    Deliberately carries no proposal text. The anchor is ``(repo, topic, principal,
    revision)`` and the text is not in it, so a :class:`DesignProposal` is exactly the
    set of facts that decides *which* record this is -- nothing else a caller holds can
    change the identity, and nothing a reconciler later argues can be quietly folded
    into it. It is also what makes the handle safe: the id the reconciliation derives
    from a proposal is the id the capture derived from it, with no third spelling of the
    same three fields in between.

    ``revision`` exists for the reason it exists on
    :class:`kojutsu.models.RationaleEntry`: a proposer that revises its position
    appends rather than overwrites, because the first argument is often the more
    interesting half. It also means the idempotency claim is exactly "one record per
    principal, topic and revision" and no stronger; a caller that increments it on a
    retry creates a second record, which is visible in the record rather than silent.
    """

    principal: str
    model: str | None = None
    revision: int = 1


@dataclass(frozen=True)
class DiscardedProposal:
    """One proposal the reconciler did not take, and the reason it did not take it.

    The reason is required rather than optional for the argument
    :class:`kojutsu.core.design_plan.RejectedAlternative` makes: a discarded proposal
    recorded as a bare name is worth much less than it looks, because a reader cannot
    tell whether it was dropped as worse, as out of scope, or as redundant with a
    decision already in the plan -- and those three imply different next moves.

    Naming the :class:`DesignProposal` rather than its entry id is what lets the
    independence check see discarded proposals too, which matters: a reconciler that
    may discard a proposal without being checked against it could satisfy the
    constraint by throwing its own proposal away.
    """

    proposal: DesignProposal
    reason: str


@dataclass(frozen=True)
class ProposalCaptureOutcome:
    """What one proposal capture did, and the handle the reconciliation takes.

    ``proposal`` is returned rather than only consumed so the reconciliation cannot be
    handed a different principal than the one that was captured. Both sides derive the
    entry id from the same three fields; passing them separately would be two spellings
    of one fact, and the failure would be a reconciliation anchored on a proposal that
    was never captured.

    ``captured`` is ``False`` when the anchor was already claimed -- the ordinary
    outcome of a retry, and not an error. ``detail`` says so in words rather than
    leaving the caller to infer it from a boolean.
    """

    entry_id: str
    proposal: DesignProposal
    captured: bool
    delivery: KnowledgeDeliveryOutcome | None
    detail: str | None = None


@dataclass(frozen=True)
class ReconciliationOutcome:
    """What one reconciliation produced, recorded or not.

    ``plan`` and ``plan_sha256`` are returned whether or not the record was stored,
    because the caller holds a plan either way and the digest is how it tells whether
    the plan in hand is the one the ledger describes. That comparison is not made here
    -- see :func:`reconcile_design_proposals` for why it cannot be -- so the pair is
    the seam a later phase checks rather than a check performed for it.

    ``captured`` is ``False`` when a reconciliation already exists for this anchor.
    """

    entry_id: str
    plan: DesignPlan
    plan_sha256: str
    proposal_ids: tuple[str, ...]
    discarded: tuple[DiscardedProposal, ...]
    captured: bool
    delivery: KnowledgeDeliveryOutcome | None
    detail: str | None = None


def stable_design_proposal_id(*, repo: str, topic: str, declared_by: str, revision: int) -> str:
    """Return the durable identity of one proposer's proposal on one design topic.

    Derived from the semantic anchor -- which repository, which topic, which
    principal, which revision -- and never from the proposal text. That is
    :func:`kojutsu.core.question_registry.stable_rationale_entry_id`'s argument
    applied unchanged: a digest over the prose would make every rephrasing an
    unrelated record, silently orphaning the earlier one, which is the specific loss
    the revision model exists to prevent.

    ``topic`` and ``declared_by`` are hashed verbatim, which is also that derivation's
    rule. It is worth saying why the two are not folded: NFC and case folding would
    collapse two byte strings that are two records a reader can see as different, and
    the *independence* check is where case-insensitivity belongs -- it needs
    ``"Opencode"`` and ``"opencode"`` to be one party, while identity needs two
    different spellings of one proposal to be one record only if the caller says so
    with a revision.

    The digest lives under its own domain rather than the rationale one, so a design
    proposal and a change's declared rationale can never be the same record even when
    the repository, the principal and the revision line up. Both are stated reasons;
    conflating them would let a design argument be read as a statement about a change
    that does not exist yet.
    """
    if not topic.strip():
        raise ValueError("a design proposal must name the topic it is about")
    if revision < 1:
        raise ValueError(f"design proposal revision must be at least 1, got {revision!r}")
    return (
        f"design-proposal-v{DESIGN_PROPOSAL_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(
                DESIGN_PROPOSAL_IDENTITY_DOMAIN,
                (repo.casefold(), topic, declared_by, str(revision)),
            )
        ).hexdigest()
    )


def stable_design_reconciliation_id(
    *, repo: str, topic: str, reconciled_by: str, revision: int
) -> str:
    """Return the durable identity of one reconciliation on one design topic.

    The same relation as :func:`stable_design_proposal_id` -- a principal, a topic, a
    revision -- under a second domain. Two derivations rather than one is not
    duplication for its own sake: the caller must not be able to derive a proposal's
    id and a reconciliation's id onto the same registry row, and a shared domain with
    a different prefix would leave exactly that to be prevented by the prefix alone.

    **The proposal set is deliberately not in the anchor**, and this is worth stating
    plainly because the opposite looks obvious. It would be the better identity: a
    reconciliation identified by what it reconciled would get a fresh record whenever
    a proposer revised, with no revision counter for a caller to get wrong. It cannot
    be had, because ``rationale_captures`` carries its uniqueness index over the
    anchor columns and nothing else -- two reconciliations of one topic by one
    principal at revision 1 would derive different ids, hit that index, and the second
    would be refused silently as a duplicate of a record that reconciled a different
    set. A record that is missing because its anchor was cleverer than the table is
    worse than a revision a caller has to be deliberate about.

    So a second reconciliation of the same topic by the same principal takes
    ``revision=2``, exactly as a second declaration does, and each record states which
    proposals it covered so the two can be compared after the fact.
    """
    if not topic.strip():
        raise ValueError("a reconciliation must name the topic it reconciles")
    if revision < 1:
        raise ValueError(f"design revision must be at least 1, got {revision!r}")
    return (
        f"design-reconciliation-v{DESIGN_RECONCILIATION_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(
                DESIGN_RECONCILIATION_IDENTITY_DOMAIN,
                (repo.casefold(), topic, reconciled_by, str(revision)),
            )
        ).hexdigest()
    )


def design_plan_digest(plan: DesignPlan) -> str:
    """Return the digest that names one plan document, for cross-phase comparison.

    The plan carries no provenance on purpose -- see the
    :mod:`kojutsu.core.design_plan` module docstring -- so a reader of a reconciliation
    record cannot tell which plan it was made from unless the record says so. This is
    that "unless", and it exists for two consumers that cannot both be the writer of
    it: the reconciliation records it under :data:`DESIGN_PLAN_DIGEST_KEY`, and a
    later ticket-creation phase recomputes it over the plan it holds and compares,
    so tickets cannot be created from a plan no reconciliation ever described.

    The bytes are the plan's own canonical JSON serialisation, which is
    ``model_dump_json``: field order is declaration order, and the schema holds only
    strings, integers and one integer literal, so there is no float formatting or
    dictionary ordering to differ between runs. It is a fingerprint of *this document*,
    not a claim that two equal digests are the same decision -- a reconciler that
    produces a different plan for the same inputs gets a different digest, which is the
    point: see :func:`reconcile_design_proposals` on what happens to that second plan.
    """
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def capture_design_proposal(
    *,
    repo: str,
    topic: str,
    principal: str,
    model: str | None,
    text: str,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    proposed_at: datetime | None = None,
    revision: int = 1,
) -> ProposalCaptureOutcome:
    """Capture one proposer's proposal as a stated-reason record, or find it captured.

    The whole of this function is the shape
    :func:`kojutsu.core.rationale_collector.process_rationale_comment_outcome`
    documents: validate, sanitise, derive, claim, build, store, complete-or-release.
    Two details are specific to this path.

    **The channel is ``CAPTURE_SERVER`` and not ``FORGE_COMMENT``.** A proposal has
    no forge behind it: it is produced locally, and there is no pull request yet for a
    reader to find the argument on. Recording the wrong channel would claim a
    re-fetchable route that does not exist, which is the same over-claim as promoting
    a rationale to captured evidence.

    **The proposal text is refused, not truncated, when it is empty or oversized.**
    Empty after sanitising means the proposal's entire content was invisible, and a
    record that counts as an argument and reads as nothing is the outcome the character
    policy exists to prevent. Oversized is refused rather than cut, because a truncated
    proposal is a *different* argument from the one made and would be stored under the
    principal that made it.

    Returns the outcome rather than ``None`` for "already captured": a retry loop needs
    to know which id it was aiming at either way, and the proposal handle it returns is
    what :func:`reconcile_design_proposals` must be given.
    """
    _check_principal(principal)
    _check_model(model)
    _check_topic(topic)
    _check_revision(revision, "design proposal")

    field = sanitise(text)
    proposal_text = field.text
    note = field.note()
    if not proposal_text.strip():
        raise ValueError(
            "a design proposal whose whole content was removed by the character policy "
            "counts as an argument and reads as nothing, so it is not captured"
        )
    if len(proposal_text) > MAX_RATIONALE_CHARS:
        raise ValueError(
            f"a design proposal must be at most {MAX_RATIONALE_CHARS} characters, got "
            f"{len(proposal_text)}; it is refused rather than truncated, because a cut "
            f"argument stored under its own principal is a different one"
        )

    entry_id = stable_design_proposal_id(
        repo=repo, topic=topic, declared_by=principal, revision=revision
    )
    entry = RationaleEntry(
        entry_id=entry_id,
        repo=repo,
        # No pull request and no branch: a design proposal precedes both. The topic is
        # what it is about, and it is carried in ``branch`` because that is the anchor
        # column the registry's uniqueness index reads -- see the module docstring for
        # what that costs.
        pr_number=None,
        branch=topic,
        declared_by=principal,
        # A proposal that states no model is recorded as stating none. Guessing would
        # manufacture the field that makes two proposals comparable, and a reconciliation
        # reading a fabricated model is worse than one reading an honest absence.
        declared_model=model,
        rationale_text=proposal_text,
        source=RationaleSource.DECLARED,
        channel=RationaleChannel.CAPTURE_SERVER,
        revision=revision,
        revises=(
            None
            if revision == 1
            else stable_design_proposal_id(
                repo=repo, topic=topic, declared_by=principal, revision=revision - 1
            )
        ),
        declared_at=proposed_at or datetime.now(UTC),
        metadata={
            DESIGN_ROLE_KEY: DESIGN_ROLE_PROPOSAL,
            # The field's own note, which is measured against the text **as it
            # arrived** and so is exact here: unlike the comment path there is no
            # narrower extractor that removed characters silently before this
            # function saw them, which is the under-report
            # :func:`kojutsu.core.text_hygiene.describe_removals` exists to avoid.
            **({SANITISATION_KEY: note} if note else {}),
        },
    )

    token = registry.claim_rationale(
        entry_id=entry_id,
        repo=repo,
        pr_number=None,
        branch=topic,
        declared_by=principal,
        declared_model=model,
        source=RationaleSource.DECLARED.value,
        revision=revision,
        revises=entry.revises,
        rationale_text=proposal_text,
    )
    proposal = DesignProposal(principal=principal, model=model, revision=revision)
    if token is None:
        return ProposalCaptureOutcome(
            entry_id=entry_id,
            proposal=proposal,
            captured=False,
            delivery=None,
            detail=(
                "A proposal already captured for this repository, topic, principal and "
                "revision is not stored twice; re-running this capture is a no-op. The "
                "identity excludes the text, so a rephrasing is the same record and a "
                "genuinely new argument must take the next revision."
            ),
        )

    delivery = _store_claim(entry, registry=registry, sink=sink, entry_id=entry_id, token=token)
    return ProposalCaptureOutcome(
        entry_id=entry_id, proposal=proposal, captured=True, delivery=delivery
    )


def reconcile_design_proposals(
    *,
    repo: str,
    topic: str,
    reconciled_by: str,
    model: str | None,
    proposals: Sequence[DesignProposal],
    plan_payload: str | bytes | dict[str, Any],
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    discarded: Sequence[DiscardedProposal] = (),
    reconciled_at: datetime | None = None,
    revision: int = 1,
) -> ReconciliationOutcome:
    """Reconcile captured proposals into a validated plan, from outside, and record it.

    **The order of the refusals is the design.** Shape, then independence, then whether
    the proposals exist, then the plan. Independence comes second because it is the
    one failure that makes the call illegitimate rather than incomplete: an
    unentitled reconciler must not reach the payload at all, so the plan is never read
    by a principal that had no standing to reconcile it. The existence check comes
    before the parse because a reconciliation that names proposals nobody captured is
    the failure this whole phase exists to prevent -- and telling the caller that is
    more use than a second schema problem in the same run.

    **The plan is validated by :func:`~kojutsu.core.design_plan.parse_design_plan` and
    nothing is stored until it passes.** Not "validate and then apply what is valid":
    a reconciliation that stored its proposals' worth of partial plan would produce
    tickets from a document nobody approved, which is the outcome the ticket calls
    "a plan that generates malformed tickets is worse than no plan".

    **Every proposer is checked, including the discarded ones**, which is the detail
    that makes the constraint mean anything. A reconciler permitted to discard a
    proposal without being checked against it could satisfy the constraint by throwing
    its own proposal away -- so the check runs over ``proposals``, and
    ``discarded`` must name proposals within it or be refused as a decision about
    something nobody proposed.

    **What a second call with the same inputs does.** It loses the claim and stores
    nothing, which is the idempotency the ticket asks for. What it does *not* do is
    compare plans: if the reconciler is nondeterministic and produces a second, different
    plan for the same anchor, the stored reconciliation still describes the first, and
    the outcome reports ``captured=False`` beside the digest of the plan in hand.
    Detecting that would mean reading the stored record's prose back, and the registry
    exposes no read API for a record's content by id. So the caller has the digest and
    the choice: proceed, or take ``revision=2`` to record its own plan. Both are
    visible; neither is automatic.
    """
    _check_principal(reconciled_by)
    _check_model(model)
    _check_topic(topic)
    _check_revision(revision, "design reconciliation")
    if not proposals:
        raise ValueError(
            "a reconciliation needs at least one captured proposal; reconciling nothing "
            "produces a plan with no argument behind it, which is a wish list"
        )
    if len(proposals) > MAX_DESIGN_PROPOSALS:
        raise ValueError(
            f"a reconciliation may cover at most {MAX_DESIGN_PROPOSALS} proposals, got "
            f"{len(proposals)}; the human gate reads this record once, and a record "
            f"past the length anyone reads is a gate with the appearance of one"
        )

    # Keyed by the proposal rather than by its id, so a discard can be resolved to the
    # record it is about by value. Two equal proposals in the list are one proposal --
    # the identity is the same three fields either way -- and collapsing them here is
    # what keeps ``len(proposals)`` from overstating how many arguments were read.
    by_proposal = {
        proposal: stable_design_proposal_id(
            repo=repo, topic=topic, declared_by=proposal.principal, revision=proposal.revision
        )
        for proposal in proposals
    }
    phrases = _check_reconciler_independence(
        reconciled_by=reconciled_by,
        model=model,
        labelled=by_proposal,
    )
    _check_proposals_captured(
        registry=registry, repo=repo, topic=topic, ids=set(by_proposal.values())
    )

    plan = parse_design_plan(plan_payload)
    plan_sha256 = design_plan_digest(plan)

    discard_entries = _check_discarded(discarded=discarded, by_proposal=by_proposal)
    body = _render_reconciliation(
        topic=topic,
        reconciled_by=reconciled_by,
        model=model,
        plan=plan,
        plan_sha256=plan_sha256,
        proposal_ids=sorted(by_proposal.values()),
        phrases=phrases,
        discard_entries=discard_entries,
    )
    # Measured against the reasons **as they arrived**, not against the sanitised copies
    # :func:`_check_discarded` returns -- see
    # :func:`kojutsu.core.text_hygiene.describe_removals`, which is the whole reason that
    # function takes the payload rather than the stored value.
    note = describe_removals(*[discard.reason for _, discard, _ in discard_entries])

    entry_id = stable_design_reconciliation_id(
        repo=repo, topic=topic, reconciled_by=reconciled_by, revision=revision
    )
    entry = RationaleEntry(
        entry_id=entry_id,
        repo=repo,
        pr_number=None,
        branch=topic,
        declared_by=reconciled_by,
        declared_model=model,
        rationale_text=body,
        source=RationaleSource.DECLARED,
        channel=RationaleChannel.CAPTURE_SERVER,
        # A reconciliation is revision 1 by default and, when a second is wanted, is
        # revision 2 with the same anchor -- see
        # :func:`stable_design_reconciliation_id` for why the proposal set cannot be the
        # discriminator the registry would have to understand.
        revision=revision,
        revises=(
            None
            if revision == 1
            else stable_design_reconciliation_id(
                repo=repo, topic=topic, reconciled_by=reconciled_by, revision=revision - 1
            )
        ),
        declared_at=reconciled_at or datetime.now(UTC),
        metadata={
            DESIGN_ROLE_KEY: DESIGN_ROLE_RECONCILIATION,
            DESIGN_PLAN_DIGEST_KEY: plan_sha256,
            DESIGN_PROPOSAL_IDS_KEY: sorted(by_proposal.values()),
            DESIGN_DISCARDED_KEY: [
                {
                    "entry_id": identifier,
                    "principal": discard.proposal.principal,
                    "reason": reason,
                }
                for identifier, discard, reason in discard_entries
            ],
            **({SANITISATION_KEY: note} if note else {}),
        },
    )

    token = registry.claim_rationale(
        entry_id=entry_id,
        repo=repo,
        pr_number=None,
        branch=topic,
        declared_by=reconciled_by,
        declared_model=model,
        source=RationaleSource.DECLARED.value,
        revision=revision,
        revises=entry.revises,
        rationale_text=body,
    )
    proposal_ids = tuple(sorted(by_proposal.values()))
    discarded_proposals = tuple(discard for _, discard, _ in discard_entries)
    if token is None:
        return ReconciliationOutcome(
            entry_id=entry_id,
            plan=plan,
            plan_sha256=plan_sha256,
            proposal_ids=proposal_ids,
            discarded=discarded_proposals,
            captured=False,
            delivery=None,
            detail=(
                "A reconciliation already exists for this repository, topic, reconciler "
                "and revision, so this one is not recorded and the stored one stands. "
                f"The plan in hand digests to {plan_sha256}; nothing here compares it "
                "against the recorded plan, so a caller that needs its own plan "
                "recorded must take the next revision."
            ),
        )

    delivery = _store_claim(entry, registry=registry, sink=sink, entry_id=entry_id, token=token)
    return ReconciliationOutcome(
        entry_id=entry_id,
        plan=plan,
        plan_sha256=plan_sha256,
        proposal_ids=proposal_ids,
        discarded=discarded_proposals,
        captured=True,
        delivery=delivery,
    )


@dataclass(frozen=True)
class RecordedReconciliation:
    """One reconciliation as the ledger holds it, read back for a phase that acts on it.

    A reader of :func:`reconcile_design_proposals`'s outcome object holds the plan. A
    later, separate invocation does not: it holds a repository, a topic and an
    operator's instruction to approve. This is what it can recover.

    It is deliberately *not* a :class:`ReconciliationOutcome` and deliberately carries no
    plan. The plan is a document that lives somewhere else -- see
    :func:`find_design_reconciliations` for why -- and a type that could carry one would
    let a caller hand :func:`~kojutsu.core.ticket_drafts.create_tickets_from_plan` a
    ``ReconciliationOutcome`` whose ``plan`` was parsed from a file rather than from the
    reconciliation, which is the substitution the digest check exists to catch and which
    this type makes a type error instead of a runtime argument error.

    ``plan_sha256`` is the point of the whole record. It is the digest the *reconciler*
    recorded, read out of the stored prose, and comparing it against a plan found
    elsewhere is the one check that decides whether two independently durable things
    describe the same document.
    """

    entry_id: str
    reconciled_by: str
    model: str | None
    revision: int
    plan_sha256: str
    recorded_at: str | None = None


class DesignReconciliationNotFoundError(ValueError):
    """No reconciliation is recorded for this repository and topic.

    A refusal, never an empty answer with nothing to do. "I found no reconciliation"
    and "here is the one reconciliation" are the same return value, and the caller that
    proceeds on the empty one creates zero tickets and exits successfully -- which reads
    as a run that correctly decided there was no work. So the lookup refuses and the
    message says what it looked for and where, because the two causes an operator has
    are entirely different: they never ran the capture half, or they ran it against a
    different registry, a different instance, or a different topic spelling.
    """

    def __init__(self, *, repo: str, topic: str) -> None:
        super().__init__(
            f"No recorded reconciliation for repository {repo!r} on design topic {topic!r}. "
            "Looked for completed reconciliation records whose entry id carries the "
            f"{RECONCILIATION_ID_PREFIX!r} prefix in {REGISTRY_STORE_DESCRIPTION}, matching that "
            "repository and carrying that topic. Run the same command without --approve first: "
            "a reconciliation is written by the capture half, and this command only spends one."
        )


class AmbiguousReconciliationError(ValueError):
    """More than one reconciliation is recorded for this repository and topic.

    **Picking one is the failure this exists to prevent, and the parallel is not
    decorative.** A gateway that resolves two matching routes by choosing one is a
    routing bug nobody can see; a resume that resolves two recorded reconciliations by
    choosing one sends the tickets of the plan it happened to prefer under an approval
    read by somebody who was looking at the other. Both plans are well-formed, both
    reconciliations are complete records, and nothing downstream can tell afterwards
    which one was meant -- so the choice has to be refused rather than made.

    Every candidate is named with its principal, its revision and its plan digest,
    because any one of the three is what an operator needs to decide: the principal says
    who to ask, the revision says which sequence position, and the digest is what
    distinguishes two records by the same principal at the same revision number on a
    registry whose uniqueness index the reconciler could not satisfy. All three, because
    picking the wrong field to print would leave the operator solving for the other one.
    """

    def __init__(
        self,
        *,
        repo: str,
        topic: str,
        found: Sequence[RecordedReconciliation],
        requested: str | None = None,
    ) -> None:
        listed = "\n".join(
            f"  - {item.entry_id} by {item.reconciled_by!r} at revision {item.revision}, "
            f"plan digest {item.plan_sha256}"
            for item in found
        )
        head = (
            f"Refusing to proceed: {requested!r} is not one of the reconciliations "
            f"recorded for repository {repo!r} on design topic {topic!r}"
            if requested is not None
            else (
                f"Refusing to proceed: {len(found)} recorded reconciliations for repository "
                f"{repo!r} on design topic {topic!r}"
            )
        )
        tail = (
            "Name the one you mean with --reconciliation-id, which is the id the pause "
            "printed. It is checked against these records rather than trusted."
            if requested is None
            else "Nothing here is missing or malformed -- these are all complete records, and "
            "the id given names none of them, so approving would attach this approval to a "
            "plan nobody selected."
        )
        super().__init__(
            f"{head}.\n{listed}\n"
            "Nothing about them is broken: these are complete records, and choosing between "
            "them would create one plan's tickets under an approval read against the other.\n"
            f"{tail}"
        )


def find_design_reconciliations(
    *, repo: str, topic: str, registry: QuestionRegistry
) -> tuple[RecordedReconciliation, ...]:
    """Return every recorded reconciliation for one repository and topic, oldest first.

    **The lookup that :func:`stable_design_reconciliation_id` could not be.** The
    derivation answers "what is the id of the reconciliation of this topic by this
    principal at this revision", which is the wrong question for a phase that has to act
    on a reconciliation it did not write: a resume knows the repository and the topic and
    nothing else, because it may be run days later by somebody who did not do the
    capture. Deriving the id would require the resume to be told the principal and the
    revision as well -- which is not a lookup, it is asking the operator to reproduce the
    identity of a record they have never read, and a typo in a remembered revision
    silently selects nothing.

    **Why it lives here and not in the caller's module.** Three facts it needs are
    private to this file: the topic rides in the ``branch`` column (see the module
    docstring), only a ``completed`` claim is a proposal somebody can read (see
    :func:`_check_proposals_captured`), and the reconciliation id prefix is what tells a
    judgement apart from the arguments it judged when ``DESIGN_ROLE_KEY`` is unreachable
    from a registry read. A caller that re-derived those three rules would be able to
    disagree with this one silently, and the disagreement would read as "no
    reconciliation found" -- a successful run that creates nothing.

    **And the honest limit of reading the digest back out of the prose.**
    ``rationale_captures`` has no ``metadata`` column, so
    :data:`DESIGN_PLAN_DIGEST_KEY` -- which :func:`reconcile_design_proposals` writes and
    the knowledge sink delivers -- is not reachable through the registry at all. The
    digest is therefore recovered from :data:`PLAN_DIGEST_LINE_PREFIX`, one line of the
    rendered record that :func:`_render_reconciliation` writes. That is a coupling
    between a renderer and a reader, and it is why both live in this file: the renderer
    cannot drop or reword that line without breaking this function, and the next reader
    of this docstring knows it. A registry read API for record content is the real fix
    and belongs to another phase's table, not to this one.

    A row whose digest line is absent or unreadable **refuses the lookup** rather than
    being skipped. Skipping it would be indistinguishable from that reconciliation not
    existing, and the one thing this function must never produce is a confident "there is
    exactly one" over a set that silently lost a member.

    Ordering is by revision then id -- the same order
    :meth:`~kojutsu.core.question_registry.SqliteQuestionRegistry.list_rationales` returns,
    so a caller reading this list and one reading the registry see the same sequence.
    """
    found: list[RecordedReconciliation] = []
    for row in registry.list_rationales(repo=repo, limit=MAX_QUESTION_LIST_LIMIT):
        if str(row["branch"]) != topic:
            continue
        entry_id = str(row["entry_id"])
        if not entry_id.startswith(RECONCILIATION_ID_PREFIX):
            # A proposal on the same topic. Its absence from this result is the whole
            # reason the prefix check exists: see DESIGN_ROLE_KEY, which does not reach
            # a registry read, and the note there about who the prefix reaches.
            continue
        if str(row["status"]) != "completed":
            # An unfinished capture is a record nobody can read, so treating it as a
            # reconciliation would let a resume spend an approval on it.
            continue
        model = row["declared_model"]
        found.append(
            RecordedReconciliation(
                entry_id=entry_id,
                reconciled_by=str(row["declared_by"]),
                model=None if model is None else str(model),
                revision=int(row["revision"]),
                plan_sha256=_recorded_plan_digest(entry_id, str(row["rationale_text"])),
                recorded_at=None if row["completed_at"] is None else str(row["completed_at"]),
            )
        )
    return tuple(found)


def require_single_design_reconciliation(
    *,
    repo: str,
    topic: str,
    registry: QuestionRegistry,
    entry_id: str | None = None,
) -> RecordedReconciliation:
    """Return the one recorded reconciliation for a repository and topic, or refuse.

    The rule this encodes is that **a phase which spends an approval must not choose which
    reconciliation it is spending it on.** The cardinality is not tidiness: see
    :class:`AmbiguousReconciliationError` for what picking one silently would produce, and
    :class:`DesignReconciliationNotFoundError` for why an empty result is a refusal rather
    than a run with nothing to do.

    ``entry_id`` is how an operator who *has* decided gets to act on that decision. The
    ambiguity refusal is not a dead end -- naming the record the pause printed is a
    deliberate, checkable choice rather than a preference -- but it is also not the
    default, because "two reconciliations exist" is then a state every run against that
    topic is refused from, which is a louder failure than the question deserves. The id is
    verified against the records actually found rather than trusted, so it cannot name
    something outside this topic; and an id that is *not* among them is refused too,
    naming what is there, because an operator who has the wrong id is about to approve the
    wrong plan and the message is the only thing that will tell them.

    Both refusals are raised rather than returned because both are conditions no caller may
    proceed past, and a return value that means "do not continue" is one somebody will treat
    as "continue with what you have".
    """
    found = find_design_reconciliations(repo=repo, topic=topic, registry=registry)
    if entry_id is not None:
        for candidate in found:
            if candidate.entry_id == entry_id:
                return candidate
        raise AmbiguousReconciliationError(repo=repo, topic=topic, found=found, requested=entry_id)
    if len(found) != 1:
        if not found:
            raise DesignReconciliationNotFoundError(repo=repo, topic=topic)
        raise AmbiguousReconciliationError(repo=repo, topic=topic, found=found)
    only = found[0]
    return only


def _recorded_plan_digest(entry_id: str, rationale_text: str) -> str:
    """Read one recorded plan digest back out of a stored reconciliation's prose.

    The inverse of the one line :func:`_render_reconciliation` writes, and it fails
    closed on every shape the inverse could go wrong: no such line, a line whose payload
    is not a SHA-256 hex digest, and -- the case that would otherwise pass -- *two* such
    lines, which means the record is ambiguous about its own plan and there is nothing
    sensible to prefer.

    Every refusal names the ``entry_id``, because a reader holding two reconciliations
    and one error needs to know which record could not be read. Silently skipping it
    would let a resume conclude there is exactly one reconciliation when there are two.
    """
    seen: str | None = None
    for line in rationale_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(PLAN_DIGEST_LINE_PREFIX):
            continue
        if seen is not None:
            raise ValueError(
                f"reconciliation {entry_id!r} states its plan digest more than once, so it "
                "does not describe one plan and nothing here could choose between them"
            )
        seen = stripped[len(PLAN_DIGEST_LINE_PREFIX) :].strip()
    if seen is None:
        raise ValueError(
            f"reconciliation {entry_id!r} carries no line beginning "
            f"{PLAN_DIGEST_LINE_PREFIX!r}, so the plan it recorded cannot be identified. "
            "The record cannot be spent on an approval: an unidentified plan is not a "
            "checkable one."
        )
    if _PLAN_DIGEST_RE.fullmatch(seen) is None:
        raise ValueError(
            f"reconciliation {entry_id!r} states a plan digest of {seen!r}, which is not a "
            "SHA-256 hex digest. Refused rather than repaired, because a digest that had "
            "to be fixed up is a digest nothing can be compared against."
        )
    return seen


def check_reconciler_independence(
    reconciled_by: str, proposals: Sequence[DesignProposal]
) -> tuple[str, ...]:
    """Raise unless the reconciler is distinct from every proposer, else return the phrases.

    The enforced check, exposed so a caller can ask the question *before* it has
    produced a plan -- and so the answer is not only a side effect of writing a
    record. It is the same code :func:`reconcile_design_proposals` runs, not a second
    implementation of it, because two implementations of a constraint is how one of
    them ends up being the one that ships.

    The comparison is :func:`kojutsu.models.compute_independence`'s, required to reach
    :attr:`~kojutsu.models.Independence.INDEPENDENT`, which for a pair of accounts
    means *the two accounts differ*. Consulting it rather than writing
    ``reconciler != proposer`` is not decoration:

    - It brings this rule's normalisation with it -- ``strip().casefold()`` -- so
      ``"Opencode"`` reconciling ``"opencode"`` is caught. A plain ``!=`` would pass
      that, and it is precisely the sloppy case the constraint exists to stop.
    - It is the codebase's single definition of "are these two parties the same", so
      this module does not become a second, weaker copy of it.
    - Its returned phrase is what the refusal quotes and what the record stores, so
      the words a reader sees about independence are the words the rest of the system
      uses about independence.

    **And it is the limit of what is available here.** See the module docstring:
    both sides of the comparison are self-declared, so this catches the accidental and
    the sloppy and cannot catch a determined self-approval. The function returns the
    phrases rather than a boolean because a boolean is the kind of thing a caller
    stores as a verdict; the phrases name what was compared.
    """
    return _check_reconciler_independence(
        reconciled_by=reconciled_by,
        model=None,
        labelled={
            proposal: f"a proposal by {proposal.principal!r} at revision {proposal.revision}"
            for proposal in proposals
        },
    )


def _check_reconciler_independence(
    *, reconciled_by: str, model: str | None, labelled: dict[DesignProposal, str]
) -> tuple[str, ...]:
    """Compare the reconciler against every proposer and refuse a shared principal.

    ``labelled`` maps each proposal to the string a refusal should name it by, which is
    the derived entry id when there is a real anchor to derive one from and a
    description of the proposal when there is not. Passing the label in rather than
    deriving it here is what lets :func:`check_reconciler_independence` ask the same
    question before a repository and a topic exist, without this function inventing an
    id from two empty strings and quoting it as though it named a record.

    ``model`` is passed to the classifier even though the rule is about principals,
    because when the principals do collide the phrase distinguishes "same account,
    same model" from "same account, different models" -- which is the difference
    between one runner misconfigured with one identifier and one principal holding two
    hats. It cannot change the verdict: a differing account is ``INDEPENDENT`` whatever
    the models are.

    Returns the distinct phrases rather than a boolean, because a boolean is the shape a
    caller stores as a verdict. The phrases are the words the rest of the system uses
    about independence, and there is normally exactly one of them -- the case this
    accepts is by construction "the two accounts differ" -- which is itself the
    clearest statement of what the check does and does not look at.
    """
    conflicts: list[tuple[str, str, str]] = []
    phrases: list[str] = []
    for proposal, label in labelled.items():
        level, phrase = compute_independence(
            asker_account=proposal.principal,
            asker_model=proposal.model,
            answerer_account=reconciled_by,
            answerer_model=model,
        )
        phrases.append(phrase)
        if level is not Independence.INDEPENDENT:
            conflicts.append((reconciled_by, label, phrase))
    if conflicts:
        raise ReconcilerNotIndependentError(conflicts)
    return tuple(dict.fromkeys(phrases))


def _check_proposals_captured(
    *, registry: QuestionRegistry, repo: str, topic: str, ids: set[str]
) -> None:
    """Refuse to reconcile a proposal that was never captured as a record.

    "Proposals are evidence, not prose in a transcript" is the claim this whole phase
    rests on, and it is checkable in exactly one place: a reconciliation names the
    proposals it reconciled, so those proposals must exist in the ledger with the same
    topic and a completed claim. Without this, a caller that mistyped a revision, or
    reconciled before capturing, produces a record whose entire provenance points at
    records that do not exist -- and nothing downstream can tell, because the ids are
    well-formed.

    A completed claim is required rather than any row: a proposal whose claim is
    ``processing`` or ``retryable`` is a proposal that failed to store, and reconciling
    it would record an argument nobody will ever be able to read.

    The lookup is a bounded read, one per proposer, so the honest limit is that a
    repository holding more than
    :data:`~kojutsu.core.question_registry.MAX_QUESTION_LIST_LIMIT` rationales from one
    principal could have a captured proposal missed and refused here. The failure
    direction is deliberate: a loud refusal is recoverable, and the alternative --
    skipping the check -- is a silent reconciliation of nothing.
    """
    captured = {
        str(row["entry_id"])
        for row in registry.list_rationales(repo=repo, limit=MAX_QUESTION_LIST_LIMIT)
        if row["branch"] == topic and str(row["status"]) == "completed"
    }
    missing = sorted(ids - captured)
    if missing:
        raise ValueError(
            f"refusing to reconcile {len(missing)} proposal(s) that are not captured as "
            f"records for repository {repo!r} and topic {topic!r}: "
            + ", ".join(missing)
            + ". A reconciliation whose proposals are not in the ledger records an "
            "argument no reader can go back to."
        )


def _check_discarded(
    *, discarded: Sequence[DiscardedProposal], by_proposal: dict[DesignProposal, str]
) -> list[tuple[str, DiscardedProposal, str]]:
    """Return each discard as ``(entry_id, discard, stored_reason)``, or refuse it.

    Three refusals, all of them about a record that would be *false* rather than
    merely unhelpful:

    - a discard naming a proposal that is not in ``proposals`` -- a decision about
      something nobody proposed, which is what an invented disagreement looks like in
      the ledger;
    - a reason that is empty once the character policy has run, for the reason
      :func:`capture_design_proposal` refuses an empty proposal;
    - a reason past :data:`MAX_DISCARD_REASON_CHARS`, refused rather than truncated,
      because this is the sentence a future reader most often needs and a cut one
      argues the wrong way without saying so.

    The reason is sanitised here rather than by the caller so the same policy applies
    to every piece of prose this module stores, and the removals are measured against
    the text as it arrived by :func:`describe_removals` where the note is written. The
    *returned* reason is the sanitised one, so the prose and the metadata copy are the
    same string and cannot disagree about what was stored.
    """
    entries: list[tuple[str, DiscardedProposal, str]] = []
    for discard in discarded:
        identifier = by_proposal.get(discard.proposal)
        if identifier is None:
            raise ValueError(
                f"cannot discard a proposal by {discard.proposal.principal!r} at revision "
                f"{discard.proposal.revision}, because it is not among the proposals this "
                "reconciliation covers; a decision recorded against something nobody "
                "proposed is a false record of what was considered"
            )
        reason = sanitise(discard.reason).text.strip()
        if not reason:
            raise ValueError(
                "a proposal discarded without a reason is a decision with nothing "
                "recorded about it, which is the one thing this record exists to prevent"
            )
        if len(reason) > MAX_DISCARD_REASON_CHARS:
            raise ValueError(
                f"a discard reason must be at most {MAX_DISCARD_REASON_CHARS} characters, "
                f"got {len(reason)}; it is refused rather than truncated, because this is "
                f"the sentence a later reader most often needs and a cut one argues the "
                f"wrong way without saying so"
            )
        entries.append((identifier, discard, reason))
    return entries


def _render_reconciliation(
    *,
    topic: str,
    reconciled_by: str,
    model: str | None,
    plan: DesignPlan,
    plan_sha256: str,
    proposal_ids: Sequence[str],
    phrases: Sequence[str],
    discard_entries: list[tuple[str, DiscardedProposal, str]],
) -> str:
    """Render the reconciliation as the prose the stored document will show.

    Prose, not a serialisation, and deliberately: this is the half of the record a
    person reads at the gate, and it is the half that reaches the store -- the
    structured copy lives in ``metadata``, which
    :func:`kojutsu.core.tanseki_mapping.build_rationale_frontmatter` now projects
    onto the document's frontmatter. The two are written from the same sanitised
    reasons in the same call, so they cannot disagree.

    The limitation sentence from :data:`INDEPENDENCE_LIMITATION_NOTE` is in the
    rendered text and not only in the module docstring, which is the whole argument for
    it being a constant: the reader who would read "distinct principal" as "verified
    distinct" is reading this document, months later, and will not have the source.

    Sizes are bounded by construction rather than by a total cap: at most
    :data:`MAX_DESIGN_PROPOSALS` discard lines, each at most
    :data:`MAX_DISCARD_REASON_CHARS`, plus one bounded line per header field.

    **One line in here is load-bearing beyond display**, and it is
    :data:`PLAN_DIGEST_LINE_PREFIX`. The structured copy of the digest lives in
    ``metadata``, which this project's frontmatter projection does not copy and
    ``rationale_captures`` does not store -- so this line is the only copy a local reader
    can reach, and :func:`find_design_reconciliations` reads it back to answer "which
    plan did this record describe". It is therefore not cosmetic text that may be
    reworded for readability, and the renderer's only field-name guarantee in the whole
    phase is that it keeps emitting it.
    """
    lines = [
        f"# Reconciliation of {len(proposal_ids)} proposal(s)",
        "",
        f"Topic: {topic}",
        f"Reconciled by: {reconciled_by}",
        f"Reconciling model: {model or 'unknown'} (asserted by the author, not verified "
        f"by any platform)",
        f"Independence enforced: the reconciler principal differs from all "
        f"{len(proposal_ids)} proposer principal(s), compared as accounts "
        f"({', '.join(phrases)}).",
        # Spelled through the shared constant, because :func:`find_design_reconciliations`
        # reads this line back out of a stored record. See PLAN_DIGEST_LINE_PREFIX for why
        # the digest is recovered from prose at all.
        f"{PLAN_DIGEST_LINE_PREFIX}{plan_sha256}",
        f"Plan: {len(plan.goals)} goal(s), {len(plan.decisions)} decision(s), "
        f"{len(plan.tickets)} ticket draft(s)",
        "",
        "## Discarded proposals",
        "",
    ]
    if discard_entries:
        for identifier, discard, reason in discard_entries:
            lines.append(f"- {identifier} by {discard.proposal.principal}: {reason}")
    else:
        lines.append("None. Every proposal was read and none was discarded.")
    lines.extend(["", INDEPENDENCE_LIMITATION_NOTE, ""])
    return "\n".join(lines)


def _store_claim(
    entry: RationaleEntry,
    *,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    entry_id: str,
    token: str,
) -> KnowledgeDeliveryOutcome:
    """Store one claimed record, then complete or release the claim.

    A second copy of the tail of
    :func:`kojutsu.core.rationale_collector.process_rationale_comment_outcome`,
    kept as it is rather than factored out because the function to factor it out of
    belongs to another phase. If that tail is changed, change this one too; the two
    are the same shape and a divergence between them is a claim that never completes
    on one path.

    The three cases are the collector's, for its reasons: a sink returning nothing is
    *uncertain* rather than delivered, a dead-letter releases the claim so a later run
    can retry, and any other failure releases the claim with the exception's type name
    before re-raising -- a record left claimed by a caller that crashed is a record
    nobody can retry until its lease expires.
    """
    try:
        result = sink.store(entry)
        if result is None:
            outcome = KnowledgeDeliveryOutcome(
                entry_id=entry_id,
                status=KnowledgeDeliveryStatus.QUEUED,
                detail="Sink returned no explicit delivery outcome; delivery is uncertain",
            )
        elif not isinstance(result, KnowledgeDeliveryOutcome):
            raise TypeError("KnowledgeSink.store must return KnowledgeDeliveryOutcome or None")
        else:
            outcome = result
        if outcome.dead_lettered:
            if not registry.release_rationale(
                entry_id, token, outcome.detail or "delivery dead-lettered"
            ):
                raise RuntimeError("Failed to release dead-lettered rationale claim")
            return outcome
        if not registry.complete_rationale(entry_id, token):
            raise RuntimeError("Failed to complete rationale capture claim")
    except Exception as exc:
        registry.release_rationale(entry_id, token, type(exc).__name__)
        raise
    return outcome


def _check_principal(principal: str) -> None:
    """Refuse a principal that cannot be compared against another one.

    Three rules, each with a failure behind it. **Non-blank**: the independence
    constraint is a comparison between two principals, and a blank one would compare
    unequal to everything and so pass as independent -- the check would report its own
    absence of input as a result. **Bounded** at
    :data:`kojutsu.integrations.github.MAX_AGENT_ID_CHARS`, the same bound the marker
    parser applies, because this is a machine-authored identifier reaching a stored
    provenance field and a document path, and one definition of its length is worth
    more than a second. **No refused characters**, which is a refusal rather than a
    cleaning: the principal is hashed into the identity, so stripping an invisible
    glyph here would let two principals that differ only by an invisible glyph derive one
    id and lose one of the two records. Nothing here can re-fetch a principal to check
    it against, so there is no honest repair available and the caller is told instead.
    """
    if not isinstance(principal, str) or not principal.strip():
        raise ValueError(
            "a principal must name itself; a blank one compares unequal to everything "
            "and would pass the independence check by being absent"
        )
    if len(principal) > MAX_AGENT_ID_CHARS:
        raise ValueError(
            f"a principal must be at most {MAX_AGENT_ID_CHARS} characters, got {len(principal)}"
        )
    _refuse_refused_characters(principal, "principal")


def _check_model(model: str | None) -> None:
    """Bound an optional model name and refuse refused characters in it.

    A model name reaches a stored provenance field and the frontmatter a reader uses
    to decide what produced a proposal, so the bound is the marker parser's rather than
    a new one. The refused-character rule is the principal's, for the narrower reason
    :func:`_refuse_refused_characters` gives: a name that renders as another one is a
    name a reader will trust as the wrong thing, and it is read on the same line as the
    principal.

    ``None`` is accepted and means *unstated*, which is what the collector's records
    mean by it -- not a defect and never a guess.
    """
    if model is None:
        return
    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "a model is either stated or unstated; an empty one is a placeholder a "
            "reader cannot tell from a principal that named a model called nothing"
        )
    if len(model) > MAX_MODEL_ID_CHARS:
        raise ValueError(
            f"a model name must be at most {MAX_MODEL_ID_CHARS} characters, got {len(model)}"
        )
    _refuse_refused_characters(model, "model name")


def _check_topic(topic: str) -> None:
    """Refuse a topic that is blank, unbounded, or carrying a refused character.

    The topic is the identity half that has no column of its own -- see the module
    docstring -- so it is held to the branch bound it borrows and refused rather than
    cleaned for the reason every other hashed component is refused rather than
    cleaned. A cleaned topic would let two topics that differ only by an invisible
    glyph derive one id, and the second proposal would be stored as a duplicate of the
    first.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(
            "a design proposal and its reconciliation must both name the topic they are "
            "about; a blank topic is an anchor nothing can be derived from"
        )
    if len(topic) > MAX_DESIGN_TOPIC_CHARS:
        raise ValueError(
            f"a topic must be at most {MAX_DESIGN_TOPIC_CHARS} characters, got {len(topic)}"
        )
    _refuse_refused_characters(topic, "topic")


def _check_revision(revision: int, what: str) -> None:
    """Refuse a revision below one, for the reason the rationale derivation does.

    A revision is a position in a sequence, and position zero means "before the first
    one", which is not a proposal and not a reconciliation. Mirrors
    :func:`kojutsu.core.question_registry.stable_rationale_entry_id` rather than
    inventing a second spelling of the rule.
    """
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError(f"a {what} revision must be an integer, got {revision!r}")
    if revision < 1:
        raise ValueError(f"a {what} revision must be at least 1, got {revision!r}")


def _refuse_refused_characters(value: str, what: str) -> None:
    """Raise naming the code point if ``value`` carries a character the policy refuses.

    The collectors perform this check by *removing* the characters. Here it is a
    refusal, and the difference is the identity: a principal and a topic are hashed
    into the anchor, so two values differing only by a refused glyph are two records a
    reader can tell apart, and removing the glyph would give them one id and drop one
    of them.

    A model name is not hashed, so removing a character from it would be harmless in
    the way the answer path's agent id is harmless -- there, the entry id is keyed on
    the comment rather than the name. It is refused anyway, for the narrower reason
    that a principal and a model are read as one attribution line: a policy that
    cleaned one and refused the other would let a reader see two spellings of the same
    pair and have no way to tell it was this module.

    The character is named because a caller who cannot see which one will remove the
    wrong one, and a caller who wanted the value cleaned needs to know it was a
    decision rather than an oversight.
    """
    found = REFUSED_RE.search(value)
    if found is not None:
        raise ValueError(
            f"a {what} must not contain U+{ord(found.group()):04X}, which this project "
            f"refuses to store; it is refused rather than removed because a {what} is "
            f"part of a design record's identity or attribution, and removing a character "
            f"from it either merges two {what}s into one id or leaves two spellings of one "
            f"attribution with nothing to say so"
        )


#: The public surface, named rather than implied.
#:
#: Two later phases import from here, and both of them need to know which names are
#: contract. Everything else in this module -- the claim/store tail, the identity
#: checks, the renderer -- is an implementation detail that a maintainer is free to
#: restructure, and a later phase reaching past this list would be reaching past the
#: argument those functions carry with them.
__all__ = [
    "DESIGN_DISCARDED_KEY",
    "DESIGN_PLAN_DIGEST_KEY",
    "DESIGN_PROPOSAL_IDENTITY_DOMAIN",
    "DESIGN_PROPOSAL_IDENTITY_VERSION",
    "DESIGN_PROPOSAL_IDS_KEY",
    "DESIGN_RECONCILIATION_IDENTITY_DOMAIN",
    "DESIGN_RECONCILIATION_IDENTITY_VERSION",
    "DESIGN_ROLE_KEY",
    "DESIGN_ROLE_PROPOSAL",
    "DESIGN_ROLE_RECONCILIATION",
    "INDEPENDENCE_LIMITATION_NOTE",
    "MAX_DESIGN_PROPOSALS",
    "MAX_DESIGN_TOPIC_CHARS",
    "MAX_DISCARD_REASON_CHARS",
    "PLAN_DIGEST_LINE_PREFIX",
    "RECONCILIATION_ID_PREFIX",
    "REGISTRY_STORE_DESCRIPTION",
    "AmbiguousReconciliationError",
    "DesignProposal",
    "DesignReconciliationNotFoundError",
    "DiscardedProposal",
    "ProposalCaptureOutcome",
    "ReconcilerNotIndependentError",
    "ReconciliationOutcome",
    "RecordedReconciliation",
    "capture_design_proposal",
    "check_reconciler_independence",
    "design_plan_digest",
    "find_design_reconciliations",
    "reconcile_design_proposals",
    "require_single_design_reconciliation",
    "stable_design_proposal_id",
    "stable_design_reconciliation_id",
]
