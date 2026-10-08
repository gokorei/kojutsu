"""Durable identity derivations for registry records.

Split out of ``question_registry`` so the store module holds the store:
SQLite state, leases and queries. These derivations are persisted
(registry ``UNIQUE`` keys, path-derived store ids), so changing one
re-identifies stored records -- version and pin, never silently patch.
"""

from __future__ import annotations

import hashlib

from kojutsu.identity import identity_preimage


def _length_prefixed(components: tuple[str, ...]) -> bytes:
    """Join identity components so no component can imitate a boundary.

    Delimiter-joining is ambiguous: ``("a|b", "c")`` and ``("a", "b|c")`` both
    serialise to ``a|b|c``, so two different records would share an id and the
    dedupe check would pass by accident. Prefixing each component with its own
    length removes the ambiguity without needing a separator that cannot occur in
    the data -- which matters because these components are attacker-influenced
    (a repository name, a branch name) and can contain any character.

    Retained for the rationale, clarification and evaluation derivations, which
    already hash through it. They are domain-separated and unambiguous already, so
    re-deriving them would re-identify stored records and orphan stored documents
    for no gain; new derivations use :func:`identity_preimage`, which is the same
    guarantee with one fewer encoding to hold in mind.
    """
    return b"".join(
        len(component.encode()).to_bytes(8, "big") + component.encode() for component in components
    )


#: Version of the answer identity derivation. Bump this, and keep both derivations,
#: rather than editing :func:`stable_answer_entry_id` in place -- for the reason
#: spelled out on :data:`RATIONALE_IDENTITY_VERSION`.
#:
#: This is 1 rather than 0 for a second reason: the preimage and the id prefix both
#: changed when the domain label was added, so an answer captured before and after
#: carry visibly different namespaces (``answer-<digest>`` against
#: ``answer-v1-<digest>``) rather than two values a reader cannot tell apart.
ANSWER_IDENTITY_VERSION = 1

#: Domain label separating answer identity from every other namespace, for the same
#: double duty as the labels below.
ANSWER_IDENTITY_DOMAIN = "kojutsu.answer.v1"


def stable_answer_entry_id(repo: str, pr_number: int, answer_comment_id: int) -> str:
    """Return the durable identity used for an answer capture.

    Derived from the comment the answer was read from, never from the answer text.
    A digest over the text would make every rephrasing an unrelated record and
    silently orphan the earlier one, which is the loss :func:`stable_rationale_entry_id`
    explains at length.

    The preimage is a JSON array behind this derivation's own domain label rather
    than the ``|``-joined string this used to build. That form was ambiguous:
    ``repo="a|1"`` with ``pr=2`` and ``repo="a"`` with ``pr="1|2"`` serialised
    identically, so two different answers derived one id and the second was dropped
    by the ``answer_captures`` unique index as a duplicate of a record nobody wrote.
    Only the forge's refusal to put a ``|`` in a repository name kept that from
    firing, and an encoding that is safe because the input happens to be is not an
    encoding.

    ``repo`` is hashed verbatim while every other derivation in this module casefolds
    it. Left as it is: casefolding here would re-identify every answer whose
    repository reached this function in mixed case, and whether that is the right
    behaviour is a question about which records should be the same record, not one
    to settle by editing a hash.
    """
    return (
        f"answer-v{ANSWER_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(
                ANSWER_IDENTITY_DOMAIN,
                (repo, pr_number, answer_comment_id),
            )
        ).hexdigest()
    )


#: Version of the rationale identity derivation. Bump this, and keep both
#: derivations, rather than editing :func:`stable_rationale_entry_id` in place.
#:
#: The rationale entry id is a durable ``UNIQUE`` key in the registry and the Tanseki
#: document id is path-derived from it, so changing the computation re-identifies
#: every stored record and orphans documents already in the store. That is a
#: migration with a story, never a patch -- see
#: ``docs/design-review/identity-and-limits.md``.
RATIONALE_IDENTITY_VERSION = 1

#: Domain label separating rationale identity from every other namespace in the
#: system. It is part of the hashed preimage, so it does double duty: it keeps
#: namespaces apart, and it means a future change to *how* the digest is computed
#: yields a different value rather than one that looks comparable and is not.
RATIONALE_IDENTITY_DOMAIN = "kojutsu.rationale.v1"


def stable_rationale_entry_id(
    *,
    repo: str,
    pr_number: int | None,
    branch: str,
    declared_by: str,
    revision: int,
) -> str:
    """Return the durable identity of one declared decision rationale.

    Derived from the *semantic anchor* of the declaration -- which change, which
    principal, which revision -- and never from the rationale text. That is a
    deliberate choice with a consequence: reworded text at the same position is
    the same record, and a genuinely new declaration must take the next revision.

    A digest over the text would make every rephrasing an unrelated record and
    silently orphan the earlier one, which is the specific loss the revision model
    exists to prevent. A declaration is appended, never overwritten, so a later
    and worse rationale cannot destroy an earlier and better one.

    ``branch`` is always part of the anchor even when ``pr_number`` is known,
    because a declaration is made against a change before the pull request it
    produces exists, and the branch is what is available at that moment.
    """
    if revision < 1:
        raise ValueError(f"rationale revision must be at least 1, got {revision!r}")
    return (
        "rationale-v1-"
        + hashlib.sha256(
            _length_prefixed(
                (
                    RATIONALE_IDENTITY_DOMAIN,
                    repo.casefold(),
                    "" if pr_number is None else str(int(pr_number)),
                    branch,
                    declared_by,
                    str(revision),
                )
            )
        ).hexdigest()
    )


#: Version of the clarification identity derivation. Versioned for the same reason
#: as :data:`RATIONALE_IDENTITY_VERSION`: the id is a durable key in the outbox and
#: the Tanseki document id is path-derived from it, so moving the computation
#: re-identifies every stored record and orphans the documents already in the store.
CLARIFICATION_IDENTITY_VERSION = 1

#: Domain label separating clarification identity from every other namespace. Part
#: of the hashed preimage, so it both keeps the namespaces apart and makes a future
#: change to *how* the digest is computed produce a different value rather than one
#: that looks comparable and is not.
CLARIFICATION_IDENTITY_DOMAIN = "kojutsu.clarification.v1"


def stable_clarification_entry_id(*, repo: str, pr_number: int, github_comment_id: int) -> str:
    """Return the durable identity of one captured human clarification.

    Derived from the comment the statement was read from, and never from its text.
    A clarification's whole value is that a reader can re-fetch the comment and see
    the statement quoted, so the comment is the only part of the record that can
    serve as its identity. Two consequences, both intended:

    - Reworded text on the same comment is the same record, so an edit to a comment
      updates one document instead of orphaning the one already stored. A digest
      over the text would make every rephrasing an unrelated record.
    - A different comment is a different record even when the text is identical, so
      "two people independently said this" stays countable. A text digest would
      collapse them into one and lose the agreement, which is the finding.

    ``repo`` and ``pr_number`` are part of the preimage so the id names *which*
    comment in *which* change, not a bare integer. That makes a mis-scoped caller
    produce a second, visible record rather than colliding with a legitimate one:
    a duplicate is noise a reader can see, and a collision is a stored record
    overwritten by someone else's text.

    A clarification with no comment id is refused rather than hashed, because
    without one there is nothing to re-fetch and the record is a typed claim.
    """
    if isinstance(github_comment_id, bool) or int(github_comment_id) < 1:
        raise ValueError(
            f"a clarification must name the comment it quotes, got {github_comment_id!r}"
        )
    if isinstance(pr_number, bool) or int(pr_number) < 1:
        raise ValueError(f"a clarification must name its pull request, got {pr_number!r}")
    return (
        "clarification-v1-"
        + hashlib.sha256(
            _length_prefixed(
                (
                    CLARIFICATION_IDENTITY_DOMAIN,
                    repo.casefold(),
                    str(int(pr_number)),
                    str(int(github_comment_id)),
                )
            )
        ).hexdigest()
    )


#: Version of the evaluation identity derivation. Versioned for the same reason as
#: :data:`RATIONALE_IDENTITY_VERSION` and :data:`CLARIFICATION_IDENTITY_VERSION`:
#: the Tanseki document id is path-derived from the entry id, so moving the
#: computation orphans whatever is already stored rather than updating it.
EVALUATION_IDENTITY_VERSION = 1

#: Domain label separating evaluation identity from every other namespace, for the
#: same double duty as the labels above.
EVALUATION_IDENTITY_DOMAIN = "kojutsu.evaluation.v1"


def stable_evaluation_entry_id(*, repo: str, pr_number: int, subject: str, measurement: str) -> str:
    """Return the durable identity of one measurement this system took of itself.

    Derived from what was measured and by what, and never from the numbers. A
    measurement is re-taken every time the harness runs, and the second run is the
    same question asked again rather than a new question: two documents for one
    measurement would make a reader's only recourse for "which of these two is
    current" be to guess from timestamps.

    That is why the digest deliberately excludes the result. A digest over the
    numbers would give every re-run a new id, the store would accumulate one
    document per harness invocation, and the *history* of the measurement -- which
    is the part worth keeping -- would be a set of near-identical rows rather than
    a series. The version of the harness belongs in ``metadata`` for the same
    reason: it is a fact about the reading, not about what was read.

    ``subject`` is what was measured, not what the measurement is about in
    general, and ``measurement`` names which question was asked of it. A new
    question about the same subject is a new record, which is the point: accuracy
    and refusal rate are different questions and averaging them is how a harness
    starts reporting a number nobody asked for.
    """
    if not subject.strip():
        raise ValueError("an evaluation must name what it measured")
    if not measurement.strip():
        raise ValueError("an evaluation must name the measurement taken")
    if isinstance(pr_number, bool) or int(pr_number) < 1:
        raise ValueError(f"an evaluation must name its pull request, got {pr_number!r}")
    return (
        "evaluation-v1-"
        + hashlib.sha256(
            _length_prefixed(
                (
                    EVALUATION_IDENTITY_DOMAIN,
                    repo.casefold(),
                    str(int(pr_number)),
                    subject,
                    measurement,
                )
            )
        ).hexdigest()
    )
