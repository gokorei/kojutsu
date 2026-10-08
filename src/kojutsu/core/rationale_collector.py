"""Capture a declared decision rationale from a comment on the forge.

This is the reading half of the two routes a declaration can take. The capture
tool in ``mcp_server/capture_server.py`` posts a marked comment when a change
exists; this module turns that comment into a stored record, the way
:mod:`kojutsu.core.answer_collector` turns an answer comment into a
``KnowledgeEntry``.

**What is verified here, and what is not.** Nothing about who posted a comment is
verified any more. The posting account's association with the repository used to be
checked, against the same three values as an answer, and it was the only verification
this path performed. It is now recorded rather than enforced: see
:data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` for the measurement that
reversed the policy, and note what it did *not* change — the platform still proves
only who posted a comment. It cannot prove which model drafted it, and it certainly
cannot verify that the reason given is the reason the agent actually had. A rationale
is an assertion by its author about its own reasoning, and it is stored as
``asserted`` for that reason rather than because it arrived by an inconvenient route.
The corroboration that is still standing is that a declaration is only useful next to
the change it explains -- there is no asker for it to be independent of, because the
agent that wrote the code is the one explaining it -- and
:meth:`RationaleEntry._a_rationale_is_never_evidence` still refuses any capture
source but ``asserted``. That is testimony and never verification, and no account
standing ever made it otherwise.

**Why this is not the answer collector with different fields.** A rationale has no
parent question. There is no parent-comment lookup, no answer-dedupe gate, and no
independence comparison against an asker, because there is no asker to compare
against — the agent that wrote the code is the one explaining it, which is the
textbook self-certified case and is what ``Independence`` already says about it.
What this module shares with the other capture paths is the shape: gate, check for
a duplicate, claim, build, store, complete-or-release.

**Identity has to match the writer's.** The capture tool claims with an
anchor-derived entry id *before* it posts, because it has to dedupe before it
writes to the forge. So this module derives the id from the same anchor read out of
the comment marker. That is why the marker carries a branch. If the two derived
different ids, one declaration would become two records, and the duplicate would be
indistinguishable from two genuine declarations.

**The character policy applies here, and not because the comment path has one.**
:func:`kojutsu.core.text_hygiene.sanitise` is applied to the prose below, at the
same seam as :func:`kojutsu.core.answer_collector.process_comment_reply_outcome`,
and the argument for it here is not symmetry.

The argument that argues against is that this prose is machine-authored: an agent
writes it, so the attacker does not choose its bytes. That argument is about the
*probability* of an attack, and it assumes the injection did not work -- on the one
record kind where this system has the strongest reason to know it was tried, since
the author is by construction an agent that read attacker-controlled issue and pull
request text and is now writing prose about what it found there. A model asked to
explain a decision quotes the thing it is explaining. Exempting the one path where
the system can name the attacker as an input would be assuming its output is clean.

More to the point: the harm the policy prevents was never injection. It is a stored
document that renders as something it does not say, and that harm is a function of
where the text is *displayed*, not of who typed it. This prose is displayed --
:func:`kojutsu.core.tanseki_mapping._rationale_body` puts it under ``## Reason`` in
a Markdown document the vault renders, and the MCP read path serves the document to
agents. If anything the exposure is larger here than on the answer path, because
:func:`~kojutsu.core.tanseki_mapping.build_rationale_frontmatter` says in as many
words that a rationale is "the most persuasive unverified record the store could
hold". Deceiving a reader of this document is worth more than deceiving a reader of
an answer.

The over-filtering risk the policy invites on prose is answered by its construction
rather than by a judgement made here: the refused set is a closed enumeration, and
it explicitly preserves ZWJ, ZWNJ, the Arabic letter mark, the invisible plus and
the emoji tag block, which is everything a name or a family emoji is made of. So
there is no filter being loosened to let this path through, and no heuristic whose
threshold this path is arguing about.

**Where it runs, and the two places it must not.** After the extractors, never
before them. :func:`extract_rationale_claim` reads the ``kojutsu:`` grammar
verbatim so a marker whose own punctuation had to be repaired in order to parse
fails closed, and sanitising the payload first would repair exactly that
punctuation and defeat the check. The marker's *values* are left alone for a
separate reason: the entry id is derived from ``agent_id`` and ``branch`` on both
sides of the declaration, so normalising either here would give the writer and this
collector different ids for one comment -- one declaration stored twice, which is
the failure the branch field exists to prevent. The prose is not in that derivation
at all, so rewriting it cannot re-identify anything already stored. One
consequence is named rather than left implicit: an override inside a marker's own
``agent_id`` still reaches the stored author, because the writer side agrees to it
or the declaration is stored twice. That is a gap inherited from the identity
derivation, not a choice this policy makes.

**Where the note is recorded, and where it is not.** On the record's ``metadata``
under :data:`kojutsu.core.text_hygiene.SANITISATION_KEY`, measured against the
delivered comment rather than against the extracted prose -- the extractor has
already run its own narrower pass, so a note computed from its output would name
only the characters *this* module took, which is an under-report in the one field
whose whole job is to say what was done to the record. See
:func:`kojutsu.core.text_hygiene.describe_removals`.

What the note does and does not claim. It reaches the stored document --
``build_rationale_frontmatter`` reads it out of ``metadata`` -- and it measures the
*delivered comment* rather than the extracted prose, because the extractor has
already run its own narrower pass. So the key's absence means this module's pass
took nothing, not that the stored rationale is byte-for-byte its comment: the
extractor's removals are silent, and a clarification quoted from this record is
wording rather than bytes even when no key is present.
``docs/tanseki-seam.md`` states that boundary at the seam rather than leaving a
reader to infer a fidelity claim that does not exist.

A related asymmetry worth knowing about: the capture server writes the same record
directly, so a declaration can reach the same document by two routes. It completes
its claim before it stores, which means the collector below usually finds the claim
already closed and does not run -- the direct route is the one that survives, which
is why both apply the policy rather than only the collector.
"""

from __future__ import annotations

from datetime import UTC, datetime

# The admission policy is imported rather than declared. It used to be declared here
# *and* in ``core/answer_collector.py`` -- the same set twice -- which is how one
# policy edit had to be made twice and how a change to one of them would have left
# the two paths answering different questions. One constant and one predicate now, and
# the old name is deliberately not re-exported: an alias in a second module is the same
# duplication wearing a different spelling, and the alias is what would be edited next.
from kojutsu.allowlist import ADMIT_ALL_ASSOCIATIONS, association_admitted
from kojutsu.core.answer_collector import (
    COMMENT_AUTHOR_IS_MACHINE_KEY,
    is_machine_account,
)
from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryOutcome,
    KnowledgeDeliveryStatus,
    KnowledgeSink,
)
from kojutsu.core.question_registry import (
    QuestionRegistry,
    stable_rationale_entry_id,
)
from kojutsu.core.text_hygiene import SANITISATION_KEY, describe_removals, sanitise
from kojutsu.integrations.github import (
    extract_rationale_claim,
    extract_rationale_text_from_comment_body,
)
from kojutsu.models import RationaleChannel, RationaleEntry, RationaleSource

#: A declared reason is bounded. A comment body can hold far more than any single
#: decision rationale needs, and unbounded text in a knowledge store is the
#: manufactured-consensus failure with extra steps.
MAX_RATIONALE_CHARS = 8_000

#: A single change accumulates revisions, not pages. Well above the parser's own
#: per-item cap, because one comment may carry several declared decisions.
MAX_RATIONALE_ENTRIES = 8


def process_rationale_comment_outcome(
    *,
    comment_body: str,
    comment_id: int,
    comment_author: str | None,
    comment_created_at: datetime | None,
    author_association: str | None,
    repo: str,
    pr_number: int,
    branch: str,
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    delivery_id: str | None = None,
    #: The account kind GitHub reported for the commenter. Passed through to
    #: :func:`is_machine_account`, which prefers it over the login; see that function
    #: for what a login alone can and cannot answer.
    comment_author_type: str | None = None,
) -> KnowledgeDeliveryOutcome | None:
    """Capture one declared rationale comment as a stored record.

    Returns ``None`` when the comment is not capturable — no marker, an
    unparseable marker, a reason that is empty once the character policy has run, or a
    duplicate — and a delivery outcome when it was. ``None`` is the normal outcome for
    most comments on a pull request, so it is not an error and callers must not report
    it as one.

    Who posted the comment is no longer a reason to refuse it. That is the reversal
    described at the import below, and what is recorded in its place is what a reader
    needs instead: the association the forge attributed, and whether the account is an
    application.

    The character policy is applied to the prose and nothing else; see the module
    docstring for why it applies here and why the marker's values are not touched.
    """
    if len(comment_body) > MAX_RATIONALE_CHARS:
        return None
    claim = extract_rationale_claim(comment_body)
    if claim is None:
        return None
    if not association_admitted(author_association, ADMIT_ALL_ASSOCIATIONS):
        return None

    # Two passes, and neither is the other. The extractor is what *reads* the comment:
    # it parses the ``kojutsu:`` grammar verbatim, so a mangled marker fails closed
    # rather than being repaired into something this system did not write, and it
    # normalises a marker's value. :func:`sanitise` is what *stores* the prose, applied
    # here rather than inside the extractor because a removed character has to be
    # reported and a function returning a string has nowhere to put that. The refused
    # set is a superset of the extractor's, so nothing the narrower pass left behind
    # survives this one -- measured against ``tests/fixtures/forged-agent-report.md``,
    # what this seam adds is a BELL and two word joiners, which is what an answer
    # captured from the same comment already could not have held.
    #
    # The note is measured against the delivered comment, not against the extracted
    # prose: the extractor has already removed fourteen of the same characters and
    # handed back a bare string, so a note computed from it would name only what this
    # module took. That under-reports, and it under-reports in the one field whose
    # whole job is to say what was done to the record. See
    # :func:`kojutsu.core.text_hygiene.describe_removals`, which is the whole reason
    # that function exists.
    rationale_text = extract_rationale_text_from_comment_body(comment_body)
    reason_field = sanitise(rationale_text)
    rationale_text = reason_field.text
    sanitisation_note = describe_removals(comment_body)
    # Tested after sanitising, so a declaration whose whole content is invisible
    # controls lands on the extractor's own outcome -- nothing to capture -- rather
    # than producing a record that counts as a stated reason and reads as nothing.
    # Identical to the rule the answer and review writers apply, and for the same
    # reason: a second emptiness rule is a second thing to disagree with this one.
    if not rationale_text.strip() or len(rationale_text) > MAX_RATIONALE_CHARS:
        return None

    # The branch comes from the marker when the writer supplied one, because the
    # writer is the authority on which change it is about. A comment carrying
    # neither a branch nor one cannot be attached to a change, and a declaration
    # attached to nothing is one nobody will find again — so the caller's
    # repository-level fallback is used and the discrepancy is not papered over.
    anchor_branch = claim.branch or branch

    entry_id = stable_rationale_entry_id(
        repo=repo,
        pr_number=pr_number,
        branch=anchor_branch,
        declared_by=claim.agent_id,
        revision=claim.revision,
    )
    claim_token = registry.claim_rationale(
        entry_id=entry_id,
        repo=repo,
        pr_number=pr_number,
        branch=anchor_branch,
        declared_by=claim.agent_id,
        declared_model=claim.model,
        source=RationaleSource.DECLARED.value,
        revision=claim.revision,
        revises=(
            None
            if claim.revision == 1
            else stable_rationale_entry_id(
                repo=repo,
                pr_number=pr_number,
                branch=anchor_branch,
                declared_by=claim.agent_id,
                revision=claim.revision - 1,
            )
        ),
        rationale_text=rationale_text,
    )
    if claim_token is None:
        # Already captured, or claimed by a live lease. Either way this comment is
        # not a new fact, and a second record saying the same thing is the failure
        # semantic_review_event_id exists to prevent.
        return None

    entry = RationaleEntry(
        entry_id=entry_id,
        repo=repo,
        pr_number=pr_number,
        branch=anchor_branch,
        declared_by=claim.agent_id,
        # A comment that states no model is recorded as stating none. Guessing would
        # manufacture the provenance field that makes two declarations comparable.
        declared_model=claim.model,
        rationale_text=rationale_text,
        source=RationaleSource.DECLARED,
        channel=RationaleChannel.FORGE_COMMENT,
        revision=claim.revision,
        revises=(
            None
            if claim.revision == 1
            else stable_rationale_entry_id(
                repo=repo,
                pr_number=pr_number,
                branch=anchor_branch,
                declared_by=claim.agent_id,
                revision=claim.revision - 1,
            )
        ),
        declared_at=comment_created_at or datetime.now(UTC),
        metadata={
            "github_comment_id": comment_id,
            "comment_author": comment_author,
            # Kept, and now load-bearing in the way a gate used to be: with the
            # association no longer filtering anything, this is the record of what the
            # forge attributed to the account, and a reader weighing a declaration
            # weighs it with this beside it.
            "github_author_association": author_association,
            # Whether the posting account is an application. The capture tool posts
            # under its own identity, so on this path the flag is usually ``False`` --
            # but "usually" is why it is recorded rather than assumed, and it is the
            # same signal and the same key as the review, answer and clarification
            # paths so one reader serves all four. Provenance, not a judgement.
            COMMENT_AUTHOR_IS_MACHINE_KEY: is_machine_account(
                comment_author, account_type=comment_author_type
            ),
            "delivery_id": delivery_id,
            # What the character policy took out of this declaration, in the record's
            # own words, and absent when it took nothing -- so its absence is the claim
            # that this pass took nothing, not that the stored prose is byte-for-byte
            # the comment. ``build_rationale_frontmatter`` reads it out of ``metadata``,
            # so unlike an answer, a review or a check run it does reach the document.
            **({SANITISATION_KEY: sanitisation_note} if sanitisation_note else {}),
        },
    )

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
                entry_id, claim_token, outcome.detail or "delivery dead-lettered"
            ):
                raise RuntimeError("Failed to release dead-lettered rationale claim")
            return outcome
        if not registry.complete_rationale(entry_id, claim_token):
            raise RuntimeError("Failed to complete rationale capture claim")
    except Exception as exc:
        registry.release_rationale(entry_id, claim_token, type(exc).__name__)
        raise
    return outcome
