"""Capturing what somebody said when nobody asked them anything.

A pull request thread is mostly not Q&A. A PR owner explains why a narrow window is
the accepted cost. A reviewer concedes that a test couples to a private field. None
of it answers a Kojutsu question, so `collect` walks straight past it, and the
most useful prose in the thread never reaches the store.

This fills that gap. A comment is a candidate clarification when it is *not* a
registered question, *not* a recorded answer, and is long enough to be a statement. The
association gate that used to sit alongside those is gone, and what it was for is worth
stating because the reversal is a judgement call rather than a fact:

- **It was never a statement about trust, only about standing.** ``OWNER``/``MEMBER``/
  ``COLLABORATOR`` is GitHub saying the account has a relationship to the repository,
  and on a public project that is close to the *opposite* of a signal for automation:
  a bot is by definition not a member or collaborator of anything, so every automated
  reviewer lands on ``CONTRIBUTOR``. Measured on t3code's PR #2829, the default set
  refused 28 human comments and no bots to admit 21 of them. See
  :data:`kojutsu.allowlist.ADMIT_ALL_ASSOCIATIONS` for the numbers and the
  full argument; this path now applies that one definition rather than its own, which
  was already true of it and is the reason it is where the reversal shows up first.
- **The facts a reader needs are recorded instead.** ``github_author_association`` stays
  on the entry, and whether the account is an application is now recorded beside it,
  so "who said this" and "was a machine" are both readable without a reader having to
  match a login suffix. This record kind's author is *anyone who can comment*, so that
  distinction is the one that matters most here.
- **The override stays.** :attr:`ClarificationPolicy.associations` still narrows, and
  the narrowest setting is "only what I name" — see
  :func:`kojutsu.allowlist.association_admitted` for why an empty set is
  not the same as no restriction.

- **A comment that is already a question or an answer is excluded**, because those
  are already captured and a second record would double-count them under a heading
  that reads as "nobody asked". The registry records `answer_comment_id` and
  `parent_comment_id`, so this is a lookup rather than a guess.
- **The minimum length is a floor, not a quality bar.** It drops "+1" and "nit".
  It cannot tell a substantive statement from a terse one, and it is not asked to:
  whether a comment is *worth* capturing is an interpretive judgement, and the
  thread classifier (GT9QJEVN) is where that belongs. Until then this captures
  generously and lets the reader filter, which is the failure direction we want --
  an extra record is visible, a missing one is invisible.

What this module does not do is decide that a clarification is correct, valuable, or
even relevant. It records that an account said it, anchored to the comment id.
Everything beyond that is a reader's judgement.

**The quotation is stored verbatim — and "verbatim" has to be read precisely.** The
statement is the comment body with the display-deceptive characters removed, by
:func:`kojutsu.core.text_hygiene.sanitise`, the policy every other capture path applies
and applied here for the same reason the answer path does: this is the record kind
whose author is *any human who can comment on a pull request*, so it is the
least-trusted author in the system on the one path where the text is unprompted. What
was taken out is named under :data:`kojutsu.core.text_hygiene.SANITISATION_KEY` and
reaches the document through
:func:`kojutsu.core.tanseki_mapping.build_clarification_frontmatter`.

This is the one place the argument genuinely had to be made rather than inherited,
because a clarification is explicitly a quotation and a quote altered by a filter is
no longer evidence of what was said. What settles it is that the two are not actually
in conflict here, and the reason is a property of the refused set rather than a
judgement about this path:

- **The refused set contains no propositional content.** It is the C0/C1 controls,
  the directional overrides and isolates, a byte-order mark and a word joiner. Not
  one is a letter, a digit, or punctuation a human meant as prose. Sanitising a
  clarification therefore drops *no words at all* — where a summarised quotation
  loses meaning precisely because it drops words, this drops nothing a reader could
  have read in the first place.
- **The characters that do carry meaning are excluded from it, by name and by
  tripwire.** ZWJ, ZWNJ, the Arabic letter mark, the invisible plus and the emoji tag
  block live in :data:`~kojutsu.core.text_hygiene.PRESERVED_INVISIBLE_CHARACTERS`, and
  :mod:`kojutsu.core.text_hygiene` refuses to import if any of them reaches the
  refused set. So the cost is not "small", it is zero for honest content, and the
  direction of the trade is fixed by construction rather than argued per path.
- **A byte-faithful copy carrying an override is not a more faithful quotation.**
  The claim this record rests on is that a reader can re-fetch the comment and check
  the sentence against it — a check made by *reading*. U+202E renders the tail of a
  stored sentence reversed, so the two renderings of one set of bytes differ, and the
  reader has to guess which one the author meant. Keeping the override preserves the
  bytes and loses the sentence. The readers of this record are a Markdown renderer and
  an MCP client reading prose for an answer, and neither is looking for one.

**What a reader of a stored clarification is therefore entitled to.** Not the bytes:
the *wording*, plus an account of what was taken out of it, plus a re-fetchable
``github_comment_id`` to check the wording against. That is the whole of the anchor,
and it is why the removal is recorded here rather than deferred to the seam document —
a store that silently edited a quotation would be the dishonesty, and this one does not
do it. A document with no ``text_sanitisation`` key is byte-for-byte the comment, which
is a stronger statement than "it looked fine", and it is the only reason the stored
wording may be treated as the author's own.

**The counter-argument, recorded because it was seriously considered.** Keeping the
characters and stating plainly in ``docs/tanseki-seam.md`` that a clarification is not
sanitised would preserve byte fidelity at the price of a warning that every reader has
to act on correctly, forever, on the one record kind whose author can be anyone. That
was rejected: it hands the job of spotting homographic attacks in a document served to
an agent to whoever reads it. It also borrows the wrong precedent. A check run's
conclusion *is* stored verbatim, correctly, because a structured verdict has no
render-as-something-else surface in prose — and the very same path already sanitises
the check *name*, which does have one. The line drawn by this policy is renderability,
not authorship, and nothing about a human comment makes it the exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from kojutsu.allowlist import ADMIT_ALL_ASSOCIATIONS, association_admitted
from kojutsu.core.knowledge_sink import KnowledgeSink
from kojutsu.core.question_registry import (
    QuestionRegistry,
    stable_clarification_entry_id,
)
from kojutsu.core.tanseki_mapping import to_clarification_upsert_payload
from kojutsu.core.text_hygiene import SANITISATION_KEY, describe_removals, sanitise
from kojutsu.integrations.github import (
    extract_agent_claim,
    extract_question_id_from_comment_body,
)
from kojutsu.integrations.github_models import GitHubComment
from kojutsu.models import CaptureSource, ClarificationEntry

from .answer_collector import (
    COMMENT_AUTHOR_IS_MACHINE_KEY,
    is_machine_account,
)

#: Below this, a comment is almost certainly an acknowledgement rather than a
#: statement. Chosen to drop "+1", "LGTM", "done" and to keep anything with a
#: clause in it. It is a floor on length, not a claim about substance.
MIN_CLARIFICATION_BODY_CHARS = 40


@dataclass(frozen=True)
class ClarificationPolicy:
    """Which unprompted comments count. Overridable, and the default is stated.

    Exposed as a value rather than hard-coded so an operator can narrow it
    without editing the collector, and so a test can prove the default is what the
    docstring claims.

    ``associations=None`` — the default, and inherited rather than re-spelled —
    means **no restriction at all**. An explicit set means *only those*, which is
    how an operator who wants the old behaviour back asks for it. The two are
    distinguished by identity, not truthiness: an empty set admits nothing and does
    not quietly fall back to admitting everything, which is a distinction
    :func:`kojutsu.allowlist.association_admitted` exists to make once
    rather than five times.
    """

    associations: frozenset[str] | None = ADMIT_ALL_ASSOCIATIONS
    min_body_chars: int = MIN_CLARIFICATION_BODY_CHARS

    def admits(self, comment: GitHubComment) -> bool:
        """Whether this comment is stored at all.

        Two rules, and they answer different questions, so they are kept apart
        rather than folded into one length comparison.

        The floor is a judgement about *the comment as it arrived* — it drops "+1"
        and keeps anything with a clause in it — and it stays on the raw body for
        that reason. The second rule is about *the record*, so it runs on the text
        that would actually be stored. Without it a body that clears forty
        characters and is entirely display controls reaches
        :class:`~kojutsu.models.ClarificationEntry`, which refuses an empty
        statement: a hostile comment would become a ``ValueError`` out of a
        collector loop, which is the one way an attacker gets to decide when
        capture runs. This is the same rule the answer, review and rationale
        writers apply, and for the same reason — a second emptiness rule is a
        second thing to disagree with the first.
        """
        if not association_admitted(comment.author_association, self.associations):
            return False
        if len(comment.body.strip()) < self.min_body_chars:
            return False
        # ``sanitise`` is run twice per stored comment — here and again in
        # ``clarification_from_comment`` — which is two regex passes over a string
        # already bounded at ``MAX_RATIONALE_CHARS``-scale input. The gate and the
        # builder are separate public functions called by two callers, so sharing
        # the result would mean changing what ``admits`` returns for a caller that
        # has no use for it; the answer path already re-sanitises rather than trust.
        return bool(sanitise(comment.body).text.strip())


class _Sink(Protocol):
    def store(self, record: Any) -> Any: ...


def clarification_from_comment(
    comment: GitHubComment,
    *,
    repo: str,
    pr_number: int,
    capture_source: CaptureSource = CaptureSource.COLLECT,
    delivery_id: str | None = None,
    captured_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> ClarificationEntry:
    """Build the record for one quoted comment, with no interpretation added.

    The body's *wording* is stored as the author wrote it. A summarised quotation
    is no longer checkable against the comment it claims to come from, which would
    defeat the only anchor this record has. What is removed is the set of
    characters that cannot render as what they say — see the module docstring for
    why that is not a competing claim to fidelity, and note the cost of getting it
    wrong in each direction: keeping them costs a stored document that renders as
    something it does not say, and the removal is what makes the surviving wording
    quotable as the author's own.

    The note is measured against the body **as delivered**, not against the text
    that will be stored, so it stays exact rather than becoming an under-report the
    moment a narrower upstream pass is added to this seam. Nothing narrows it today,
    which is why the two would agree — see
    :func:`kojutsu.core.text_hygiene.describe_removals` for the case where they
    would not.

    ``metadata`` is for facts about *how this record was selected* rather than about
    the quotation -- the thread classifier passes the model that chose to hold the
    comment. Deliberately not the structure axis: a clarification's shape is a
    quotation with no question, which is established rather than inferred, so
    claiming an inference there would be a second meaning for one field.

    What the record does *not* do is judge the account. ``author_association`` is
    copied through as the forge reported it and the automation flag is resolved from
    GitHub's own ``user.type``, so both facts a reader needs are on the entry and
    neither of them is a decision this module made about who may speak.
    """
    # The marker is read from the body as it arrived, before anything is removed:
    # the ``kojutsu:`` grammar is parsed verbatim so a mangled marker fails closed
    # rather than being repaired into something this system did not write. Sanitising
    # first would repair exactly the punctuation that check exists to catch — which is
    # why this call sits below the extraction and not above it.
    claim = extract_agent_claim(comment.body)
    statement_field = sanitise(comment.body)
    sanitisation_note = describe_removals(comment.body)
    return ClarificationEntry(
        entry_id=stable_clarification_entry_id(
            repo=repo, pr_number=pr_number, github_comment_id=comment.id
        ),
        repo=repo,
        pr_number=pr_number,
        # The prose a reader renders, and nothing else from the body. The entry id is
        # derived from ``(repo, pr_number, comment.id)`` and never from the text, so
        # removing characters from the statement cannot move a document, orphan one,
        # or produce a second record of one comment.
        statement=statement_field.text,
        author=comment.user.login,
        author_association=(comment.author_association or "NONE").upper(),
        github_comment_id=comment.id,
        declared_at=comment.created_at,
        capture_source=capture_source,
        captured_at=captured_at or datetime.now(UTC),
        capture_delivery_id=delivery_id,
        # Present when the comment carries a marker, and never inferred from its
        # absence: a bot that declares nothing is recorded as declaring nothing.
        authored_by_agent=claim.agent_id if claim else None,
        authored_by_model=claim.model if claim else None,
        tags=["clarification", f"clarification_{capture_source.value}"],
        metadata={
            **(metadata or {}),
            # Whether the quoting account is an application, and the one flag on this
            # record kind that a reader genuinely needs: the author of a clarification
            # is anyone who can comment on a pull request, so "was a machine one end of
            # this" is unanswerable from the association -- which is the same value an
            # outside human gets. GitHub's own ``user.type`` is preferred over the
            # login suffix and both signals are described in
            # :func:`~kojutsu.core.answer_collector.is_machine_account`. Provenance, not
            # a judgement: a bot's clarification is still a clarification, and the point
            # of recording this is that a reader can weigh the two differently.
            COMMENT_AUTHOR_IS_MACHINE_KEY: is_machine_account(
                comment.user.login, account_type=comment.user.type
            ),
            # Absent when nothing was removed, and that absence is the claim that the
            # stored wording is the comment's own bytes. Present, it names every code
            # point taken and how many times, so a reader can tell sanitised evidence
            # from raw evidence without guessing which they are holding.
            **({SANITISATION_KEY: sanitisation_note} if sanitisation_note else {}),
        },
    )


def _excluded_comment_ids(registry: QuestionRegistry, repo: str, pr_number: int) -> set[int]:
    """Comment ids already accounted for as answers, so they are not stored twice.

    A recorded answer is already captured. Re-admitting it would put a second copy
    of the same text under a heading that reads as "nobody asked", which is
    precisely the misreading this record kind exists to prevent.

    Only answers come from the registry: ``list_questions`` projects a fixed column
    set that does not include the question's own comment id. Question comments are
    excluded by marker instead, in the caller, because the marker is what the
    capture path actually keys on and reading it from the body cannot disagree with
    what was posted.
    """
    excluded: set[int] = set()
    for row in registry.list_questions(repo=repo, pr_number=pr_number, limit=1000):
        value = row.get("answer_comment_id")
        if isinstance(value, int):
            excluded.add(value)
    return excluded


def collect_clarifications(
    *,
    repo: str,
    pr_number: int,
    comments: list[GitHubComment],
    registry: QuestionRegistry,
    sink: KnowledgeSink,
    policy: ClarificationPolicy | None = None,
    capture_source: CaptureSource = CaptureSource.COLLECT,
    delivery_id: str | None = None,
) -> list[ClarificationEntry]:
    """Store every unprompted, trusted comment in a thread as a clarification.

    Returns what was stored, so a caller can report coverage rather than inferring
    it from the store. Idempotent: identity is anchored on the comment id, so
    re-running over an overlapping range upserts rather than duplicating.
    """
    policy = policy or ClarificationPolicy()
    excluded = _excluded_comment_ids(registry, repo, pr_number)
    # A question's own comment is excluded by marker: it is the premise of an
    # answer, not a statement in its own right, and it is already stored with the
    # question it carries.
    excluded.update(
        comment.id for comment in comments if extract_question_id_from_comment_body(comment.body)
    )
    stored: list[ClarificationEntry] = []
    seen: set[int] = set()
    for comment in comments:
        if comment.id in excluded or comment.id in seen:
            continue
        if not policy.admits(comment):
            continue
        seen.add(comment.id)
        entry = clarification_from_comment(
            comment,
            repo=repo,
            pr_number=pr_number,
            capture_source=capture_source,
            delivery_id=delivery_id,
        )
        sink.store(entry)
        stored.append(entry)
    return stored


__all__ = [
    "ClarificationPolicy",
    "clarification_from_comment",
    "collect_clarifications",
    "to_clarification_upsert_payload",
]
