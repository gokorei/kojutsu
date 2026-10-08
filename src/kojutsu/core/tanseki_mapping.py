"""Map Kojutsu :class:`KnowledgeEntry` values to Tanseki documents.

Aligned with Tanseki's document/frontmatter contract
(``tanseki`` repo, ``docs/document-schema.md``):

- ``id`` is **path-derived**: ``<repo>/pr-<n>/<entry_id>`` (extension stripped).
- ``path`` is adapter-relative and unique per collection: ``<id>.md``.
- ``collection`` is set by the client (``kojutsu``).
- ``content`` is canonical Markdown **including** the frontmatter block.
- ``tags`` is a list; edge-relevant refs use ``repo``, ``pr``, ``jira`` (and
  optionally ``files``), matching Tanseki's ``EdgeDeriver``.

Edges are derived by the store from frontmatter and resolved ``[[wikilinks]]``,
so this module does not emit edges.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import yaml

from kojutsu import models as _models
from kojutsu.core.design_keys import (
    DESIGN_DISCARDED_KEY,
    DESIGN_PLAN_DIGEST_KEY,
    DESIGN_PROPOSAL_IDS_KEY,
    DESIGN_ROLE_KEY,
)
from kojutsu.core.text_hygiene import SANITISATION_KEY
from kojutsu.models import (
    STRUCTURE_INFERRED_BY,
    UNKNOWN_MODEL,
    CensusRecord,
    ClarificationEntry,
    EvaluationEntry,
    KnowledgeEntry,
    QuestionRecord,
    RationaleEntry,
    RecordStructure,
)

DEFAULT_COLLECTION = "kojutsu"

# Typed keys in Tanseki's Frontmatter; everything else is `extra`.
_KNOWN_KEYS = ("title", "author", "tags", "updated_at", "content_hash")
MAX_FRONTMATTER_TEXT_CHARS = 10_000
MAX_FRONTMATTER_TAGS = 100

#: Frontmatter keys describing a rationale, kept beside the entry's own list so a
#: reader looking for what a stored document claims about its own provenance finds
#: both in one place.
RATIONALE_FRONTMATTER_KEYS = ("rationale_source", "rationale_revision", "rationale_revises")

#: Names of the frontmatter keys describing *the change a record is about*, as
#: opposed to the record itself.
#:
#: These live here rather than beside the capture code that populates them because
#: this module owns the storage contract: a key is named in exactly one place, and
#: the writer imports the name from here rather than spelling it. Two spellings of
#: one key is a document that filters on one of them and silently misses the other.
CHANGE_AUTHOR_KEY = "change_author_account"
PR_OPENED_AT_KEY = "pr_opened_at"
PR_MERGED_AT_KEY = "pr_merged_at"
PR_OUTCOME_KEY = "pr_outcome"
RECORD_KIND_KEY = "record_kind"
#: Re-exported from ``models`` rather than spelled again. The key lives there
#: because ``capture_anchor_gaps`` needs it for backfilled records, and this module
#: already imports that one -- a second literal would be a second thing to keep
#: correct.
REVIEW_ID_KEY = _models.REVIEW_ID_KEY
HEAD_SHA_KEY = "head_sha"
FILES_KEY = "files"

#: The namespace a census record lives in. An observation is not a decision, so it
#: must never land among decisions -- the same reason a question gets its own.
CENSUS_NAMESPACE = "census"

#: The namespace a projected question lives in. A question is not an answer, so it
#: must never land where a reader would take it for one.
QUESTION_NAMESPACE = "question"

#: A change touching more files than this has its list truncated, and says so. The
#: bound exists because the value is unbounded in principle and frontmatter that
#: grows without limit is a document nobody can read. Truncation is recorded rather
#: than silent: a list quietly shortened reads as a complete description.
MAX_FRONTMATTER_FILES = 50

#: The same, for a clarification. There is no ``category`` here and that absence is
#: the point: a clarification has no question, so there is no retrospective
#: category for it to carry. What replaces it is the quote's own anchor — the
#: comment it was read from, and the account that posted it.
CLARIFICATION_FRONTMATTER_KEYS = (
    "github_comment_id",
    "github_author_association",
    # A clarification's author is anyone who can comment on a pull request, so on this
    # record kind in particular the association cannot say whether a machine wrote it.
    "comment_author_is_machine",
    "clarified_by_agent",
    "clarified_by_model",
)


class _FrontmatterDumper(yaml.SafeDumper):
    """Safe dumper that preserves every string as a YAML string."""


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style='"')


_FrontmatterDumper.add_representer(str, _represent_string)


def document_id(entry: KnowledgeEntry) -> str:
    """Path-derived document id (no extension), matching Tanseki's ``notes/a`` style."""
    repo = entry.metadata.get("repo") or "unknown"
    pr_number = entry.metadata.get("pr_number") or 0
    return f"{repo}/pr-{pr_number}/{entry.entry_id}"


def document_path(entry: KnowledgeEntry) -> str:
    """Adapter-relative Markdown path."""
    return f"{document_id(entry)}.md"


def _unrecorded(value: Any) -> bool:
    """Whether ``value`` is the *absence* of a fact rather than a fact of its own.

    Two shapes are absent, and for the same reason: nothing could be said about
    them. ``None`` is a collector that never learned the value; ``""`` is a slot
    somebody left blank. Either way a reader holding the document cannot tell
    "the field is blank" from "there was nothing to put in it", so the key is left
    out rather than written. That is the whole honest mechanism here -- a
    placeholder cannot be written instead, because a placeholder that passes the
    filter is then indistinguishable, to every reader downstream, from something a
    person actually stated.

    **An empty collection is deliberately not absent.** ``[]`` and ``{}`` are
    results rather than gaps: nothing matched, nothing was in the collection. They
    arrived here as the *string* ``"[]"``, which is a third thing and the worst of
    the three -- present, so it reads as recorded, but untyped, so it cannot be
    compared to anything, and a claim about a shape the document does not have.
    Collapsing them into ``None`` is what makes an outage look like an answer, and
    ``docs/tanseki-seam.md`` argues the two apart for ``files`` for exactly that
    reason; this makes the same call everywhere else a value can be a collection.
    ``None`` is "nobody looked"; ``[]`` is "nobody found anything". Only the second
    is a finding.

    Written as ``is None`` and ``== ""`` rather than as ``not value``, because
    truthiness is the bug this mapping is here to remove. ``0``, ``0.0`` and
    ``False`` are all falsy and all facts, and a projected question nobody retried
    carries ``attempts=0`` -- dropping that is dropping the observation, not
    tidying it.

    **There is no numeric branch here, on purpose.** ``bool`` is a subclass of
    ``int``, so any test that recognises a number without excluding ``bool`` first
    classifies ``True`` as ``1`` and then cannot tell a reviewer flag from a count.
    This function needs no numeric test -- ``0`` and ``False`` are both present --
    so it has nothing to get wrong. The first bound on a number's size belongs in
    :func:`_validate_frontmatter_value`, which is where a value is examined rather
    than admitted.
    """
    if value is None:
        return True
    return isinstance(value, str) and value == ""


def _write_extras(frontmatter: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge ``extra`` into ``frontmatter`` in place, **keeping each value's type**.

    Every record builder ends this way. It was six copies of one rule, which is six
    places to be wrong rather than one, and a rule about what counts as absent is
    exactly the kind that drifts: fixing it in one builder leaves the other five
    quietly disagreeing with it about the same key.

    The rule is :func:`_unrecorded`. This function adds only the part that was
    missing: what is written.

    **The type is the payload, and the coercion was the defect.** ``pr`` used to
    reach the store as ``"42"``. That is not a number, it is a number's spelling,
    and it costs a reader in two concrete ways. A document a person opens shows
    ``pr: "42"``, and every consumer that wants to do arithmetic -- a range, a
    median, a sort -- has to parse the value back before it can compare anything.
    The same held for ``reviewer_is_machine``, a real ``bool`` that arrived as
    ``"True"``/``"False"``, so a reader filtering on it had to know to compare
    against Python's capitalisation.

    **One capability does not come with this, and it was measured rather than
    assumed.** Against a real ``tanseki-daemon`` (main ``9ac4130``), a typed value
    is *stored* faithfully -- ``:get`` hands back ``pr: 42`` as a JSON number and
    ``pr_int: 42`` beside ``pr_str: "42"`` in the same block. But ``fm=`` equality
    filtering currently matches **nothing at all** for an extra, typed or not. One
    document, stored correctly, findable by free text, then filtered three ways:

    - ``plain: "hello"`` with ``fm=plain=hello`` -> 0 matched
    - ``pr_str: "42"`` with ``fm=pr_str=42`` -> 0 matched
    - ``pr_int: 42`` with ``fm=pr_int=42`` -> 0 matched
    - the same document with ``q=<body token>`` -> 1 matched

    So this is **not** a regression from sending types, and it is not a typed-value
    limitation: string extras fail identically. The document reaches the index --
    that is what the free-text hit proves -- but its frontmatter terms do not
    match a filter. ``LuceneLookup``'s own suite passes and covers exactly this
    case (``filters narrow by collection tag and frontmatter``, and
    ``typed frontmatter filters distinguish numbers and booleans``), so the
    ``Lookup`` is correct in isolation and the defect is upstream of it, in the
    path from a ``documents:upsert`` to an indexed document. Filed as Tanseki
    ``P4CCWK8T``.

    Two consequences worth stating rather than leaving for a reader to infer.
    Nothing here is filterable, so preserving a type buys no *queryable* behaviour
    yet -- the type is still worth preserving, because it is what a document shows
    a person and what any future consumer will compare against, and because the
    string form is lossy in a way the typed form is not. And ``tags`` is separately
    unfilterable by construction: it is a ``CONTRACT_KEY``, the typed ``values`` map
    must not contain it, and ``fm_tags`` is therefore never indexed at all.

    **What does not change is which keys exist.** This is the type of a value that
    was always going to be written, never the admission of a new one. The
    deliberately-textual values stay textual -- ``UNKNOWN_MODEL``,
    ``capture_source.value``, ``RecordStructure.value``, ``clarified_by_model`` --
    because a reader distinguishes "named no model" from "not recorded here" by
    reading a string, and an ``anchored`` record's ``structure`` key stays omitted
    so its bytes are unchanged from a document written before the axis existed.

    A document written before this landed still carries the string spelling, so the
    read path has to accept both: that is why
    :func:`kojutsu.models.capture_anchor_gaps` still tolerates a string number, and
    the tolerance must outlive this change rather than being tidied up with it,
    because the rows that need it are already written.
    """
    for key, value in extra.items():
        if not _unrecorded(value):
            frontmatter[key] = value


def build_frontmatter(entry: KnowledgeEntry) -> dict[str, Any]:
    """Flat frontmatter map: known keys + typed extras.

    ``repo``/``pr``/``jira`` are the keys Tanseki's edge deriver recognises.

    Provenance is always written, including for ``asserted`` entries. A reader must
    be able to tell, from the document alone, whether a record was captured from a
    signed provider delivery or merely typed in — otherwise a hand-written entry is
    indistinguishable from review evidence, which is the one thing this store cannot
    afford. The identifiers that make a capture checkable (``delivery_id``,
    ``question_id``, ``github_author_association``) are carried through here; they
    used to be dropped, which erased the audit trail at the storage boundary.
    """
    metadata = entry.metadata or {}
    frontmatter: dict[str, Any] = {
        "title": entry.question_text.strip()[:200],
        "author": entry.author or "unknown",
        "tags": list(entry.tags),
        "updated_at": entry.answered_at.isoformat(),
        "capture_source": entry.capture_source.value,
    }
    # ``files`` does not go through ``extra``, and the reason is no longer that a
    # list needs special handling -- ``_write_extras`` preserves a list perfectly
    # well now, so folding this in would not reproduce any defect. What is actually
    # load-bearing is the bound and the marker:
    #
    # - **Bounded** because the value is unbounded in principle, and frontmatter
    #   that grows without limit is a document nobody can read.
    # - **Truncated loudly** because a list quietly shortened reads as a complete
    #   description of a change. The marker is the last element rather than a
    #   separate key, so a reader cannot see the list without also seeing that it is
    #   incomplete.
    #
    # Tanseki's ``EdgeDeriver`` reads this key, so a change is reachable from the
    # files it touched, and that is the second reason a list is the right shape
    # rather than a joined string.
    #
    # **The two original reasons are both fixed, and neither is why the bypass is
    # still here.** The store half: Tanseki's ``WF6ENWNJ`` (main ``e419ddd``)
    # replaced the hand-rolled frontmatter subset with a real YAML parser, so
    # ``FrontmatterValue`` models a sequence and ``StoreApiDtos.toFrontmatter`` no
    # longer flattens an array to ``toString()`` -- a ``files`` list sent today
    # arrives as a list. The seam half: ``extra`` used to end in
    # ``frontmatter[key] = str(value)`` and now does not. So the bypass is no longer
    # a workaround for anything.
    #
    # **It stays anyway, because the rows already written cannot be repaired.**
    # Documents captured before ``WF6ENWNJ`` hold the repr as a scalar --
    # ``files: '["src/a.py", "src/b.py"]'`` as text, not as a list. Before that
    # change the store recovered them: ``EdgeDeriver.parseRefs`` took a scalar
    # string, stripped the brackets and split on commas, which is why the comma had
    # become load-bearing. ``WF6ENWNJ`` deleted that handling, and
    # ``Frontmatter.texts(key)`` is now ``values[key]?.leaves()?.mapNotNull { it.asText() }``
    # -- a ``TextValue`` holding the repr yields exactly one leaf, and ``asText()``
    # hands back the whole thing. So each historical document now derives a single
    # ``embeds`` edge whose target is the literal repr string, where it used to
    # derive the real paths. Removing this bypass without first re-capturing those
    # rows would convert a recoverable mess into a silent wrong answer, and the
    # store can no longer repair it: the parsing that made it recoverable is gone.
    #
    # So there are two honest end states, and this line should move only for one of
    # them: either the historical rows are re-captured (or migrated) so that no
    # stored document predates ``WF6ENWNJ``, or the bypass stays to keep those rows
    # readable. "It works now, so remove the workaround" is the one option that
    # loses data, and the reason is not visible from this side of the seam.
    #
    # One divergence from :func:`_unrecorded` is deliberate and pinned by a test:
    # ``files=[]`` is written as *absent* rather than as an empty list, because an
    # empty file list is not a thing a real pull request has. ``_unrecorded`` says
    # an empty collection is a result; for this key the honest reading is that the
    # reader was never given one. Reconciling the two is a decision about this key,
    # not a cleanup, and it belongs with the migration question above.
    files = metadata.get(FILES_KEY)
    if isinstance(files, list | tuple) and files:
        bounded = [str(path) for path in files[:MAX_FRONTMATTER_FILES]]
        if len(files) > MAX_FRONTMATTER_FILES:
            bounded.append(f"{len(files) - MAX_FRONTMATTER_FILES} more files not listed")
        frontmatter[FILES_KEY] = bounded
    extra = {
        "category": entry.category.value,
        "repo": metadata.get("repo"),
        "pr": metadata.get("pr_number"),
        "jira": metadata.get("jira_ticket_key"),
        "pr_url": metadata.get("pr_url"),
        "session_id": entry.session_id,
        "github_comment_id": metadata.get("github_comment_id"),
        "answered_at": entry.answered_at.isoformat(),
        # Provenance: what produced this record, when we saw it, and under which
        # provider delivery. Retained rather than discarded.
        "delivery_id": entry.capture_delivery_id or metadata.get("delivery_id"),
        "question_id": metadata.get("question_id"),
        "github_author_association": metadata.get("github_author_association"),
        "captured_at": entry.captured_at.isoformat() if entry.captured_at else None,
        # Attribution: who actually wrote the answer, and whether a machine did.
        "answered_by_agent": metadata.get("answered_by_agent"),
        "comment_author": metadata.get("comment_author"),
        # Whether the reviewer account is an application. Carried so a consumer can
        # separate machine review from human review without matching on a login
        # suffix, which is a naming convention rather than a field. Absent rather than
        # false when the collector did not decide, so "not recorded" and "recorded as
        # human" stay tellable apart.
        "reviewer_is_machine": metadata.get("reviewer_is_machine"),
        # The same question about the *commenting* account, which is what an answer, a
        # rationale-by-comment or an inline comment needs and ``reviewer_is_machine``
        # cannot answer. It replaces the association filter rather than supplementing
        # it: with admission unrestricted, this and ``github_author_association`` are
        # the two facts a reader weighs to tell a project's own review from a drive-by
        # or a bot, and a document that carried only one of them would read as though
        # the other had been decided and forgotten. Absent rather than false when the
        # collector did not decide, which is the same rule as the flag above -- and
        # ``False`` is a fact, so it is written.
        "comment_author_is_machine": metadata.get("comment_author_is_machine"),
        # How the reasoning was obtained: stated from the author's own session, or
        # inferred from the diff. Carried for machine-authored answers so a reader
        # can tell which they are holding.
        "rationale_source": metadata.get("rationale_source"),
        # Independence: how far this record is from checking itself. Carried with the
        # model it was derived from, because a level without its inputs is a claim
        # rather than a derivation.
        "answered_by_model": metadata.get("answered_by_model"),
        "independence": metadata.get("independence"),
        "independence_reason": metadata.get("independence_reason"),
        # Structure: whether the record's own question/answer pairing was
        # established or inferred by a model. Written *only* when it is not the
        # default, which is the opposite of the treatment ``capture_source`` gets
        # above and is deliberate. Adding the axis must not re-identify anything
        # already stored: a document written before it existed is a real pairing
        # by construction, so writing ``structure: anchored`` onto it would add a
        # claim the writer never made, and change the bytes of every stored
        # record to say so. Absence resolves to ``anchored`` on read, so the
        # value that is omitted is exactly the value a reader recovers.
        "structure": (
            entry.structure.value if entry.structure is not RecordStructure.ANCHORED else None
        ),
        STRUCTURE_INFERRED_BY: metadata.get(STRUCTURE_INFERRED_BY),
        # What kind of record this is, carried explicitly rather than left to be
        # inferred from the tag set. A record with no tags -- an answer written by
        # someone who stated no agent -- is otherwise indistinguishable from a
        # record of any other kind with no tags, and a reader left guessing cannot
        # tell a missing value from a missing record.
        RECORD_KIND_KEY: metadata.get(RECORD_KIND_KEY),
        # The forge's own identifier for a review. The one stable way to pair a
        # stored record with the thing it came from, and the only way to notice
        # that the review was later edited or deleted.
        REVIEW_ID_KEY: metadata.get(REVIEW_ID_KEY),
        # The change under review: who opened it, when it opened, how it ended.
        # Every one of these arrived on the webhook payload and was used to reach a
        # decision -- independence in the case of the author, a dedupe identity in
        # the case of the merge -- and then dropped, which left a store that could
        # not say whose change it was holding evidence about.
        CHANGE_AUTHOR_KEY: metadata.get(CHANGE_AUTHOR_KEY),
        PR_OPENED_AT_KEY: metadata.get(PR_OPENED_AT_KEY),
        PR_OUTCOME_KEY: metadata.get(PR_OUTCOME_KEY),
        PR_MERGED_AT_KEY: metadata.get(PR_MERGED_AT_KEY),
        # The commit the capture was taken against. An anchor, never a
        # verification: it says which commit this record is about, not that the
        # record is still true there, that the change was reviewed, or that the
        # two correspond to the same code.
        HEAD_SHA_KEY: metadata.get(HEAD_SHA_KEY),
        # A check run's own facts, so a reader can tell *which* check concluded
        # *what* about *which* commit. Recorded as the forge reported them; none of
        # these is kojutsu's reading of them.
        "check_id": metadata.get("check_id"),
        "check_name": metadata.get("check_name"),
        "check_status": metadata.get("check_status"),
        "check_conclusion": metadata.get("check_conclusion"),
        # What the hygiene pass took out of third-party text, if anything. Absent
        # when nothing was removed, and that absence is the claim: the stored text
        # is the contributor's byte for byte. Present, the note names each code
        # point and how many times, so a reader can tell sanitised evidence from
        # raw evidence without having to guess which they are looking at -- a
        # sanitisation that leaves no trace is indistinguishable from having not
        # happened, which is the opposite of what the ticket asked for.
        SANITISATION_KEY: metadata.get(SANITISATION_KEY),
    }
    _write_extras(frontmatter, extra)
    return frontmatter


#: The body marker for a question no human asked. On its own line, in capitals, and
#: above the question rather than in the frontmatter beside it.
#:
#: Frontmatter already says ``structure: inferred`` and names the model. The body
#: says it again because the frontmatter is a header a reader skims past and the
#: body is the part they quote, and a question quoted out of this document into a
#: ticket or a design note has to carry its own label with it. Frontmatter-only
#: labelling means the reconstruction becomes a question again the moment anyone
#: copies the prose -- which is the only way most of these records are ever read.
INFERRED_QUESTION_BODY_MARKER = "RECONSTRUCTED QUESTION — NOBODY ASKED THIS"


def _inferred_question_notice(entry: KnowledgeEntry) -> str:
    """The paragraph that has to travel with an inferred question, and the model.

    Names the model for the same reason the frontmatter key does: "a model guessed"
    is not weighable, and the reader deciding how much to trust a guessed pairing is
    deciding about a specific one. Both comment ids are given because the pairing is
    the claim, and a reader who wants to check it has to be able to fetch both ends.
    """
    model = entry.metadata.get(STRUCTURE_INFERRED_BY) or "an unnamed model"
    question_id = entry.metadata.get("inferred_question_comment_id")
    answer_id = entry.metadata.get("github_comment_id")
    ends = ", ".join(
        part
        for part in (
            f"question read out of comment {question_id}" if question_id else "",
            f"answer quoted from comment {answer_id}" if answer_id else "",
        )
        if part
    )
    notice = [
        f"> **Nobody asked this.** {model} inferred the pairing by reading the "
        "thread; the answer below is quoted verbatim from a real comment, which is "
        "what makes the question beside it persuasive without being true."
    ]
    if ends:
        notice.append(">")
        notice.append(f"> Inferred from: {ends}.")
    return "\n".join(notice) + "\n"


def _body(entry: KnowledgeEntry) -> str:
    question = entry.question_text.strip()
    if entry.structure is RecordStructure.INFERRED:
        # Only for an inferred record, and the anchored path through this function
        # is untouched: a golden test pins the bytes of a real anchored document, and
        # a record nobody guessed at must not gain a paragraph about guessing.
        return (
            f"# {INFERRED_QUESTION_BODY_MARKER}\n\n"
            f"{_inferred_question_notice(entry)}\n"
            f"## Question (reconstructed by "
            f"{entry.metadata.get(STRUCTURE_INFERRED_BY) or 'an unnamed model'})\n"
            f"{question}\n\n"
            f"## Answer (quoted from a real comment)\n{entry.answer_text.strip()}\n"
        )
    return f"# {question}\n\n## Question\n{question}\n\n## Answer\n{entry.answer_text.strip()}\n"


def rationale_document_id(rationale: RationaleEntry) -> str:
    """Path-derived document id for a rationale, kept out of the answer namespace.

    A rationale is not an answer to a question, so it must not land at
    ``<repo>/pr-<n>/<entry_id>`` where it would be indistinguishable from one. The
    ``rationale/`` path segment is what tells a reader of the store that this
    document is a stated reason rather than a conclusion.
    """
    pr_number = rationale.pr_number if rationale.pr_number is not None else 0
    return f"{rationale.repo}/pr-{pr_number}/rationale/{rationale.entry_id}"


def build_rationale_frontmatter(rationale: RationaleEntry) -> dict[str, Any]:
    """Flat frontmatter for a stated reason, carrying its provenance always.

    ``capture_source`` is written unconditionally, for the same reason it is on an
    entry: a reader must be able to tell, from the document alone, that this is a
    claim and not a capture. A rationale that could be read as evidence would be
    the most persuasive unverified record the store could hold, because it comes
    from the agent that did the work and therefore sounds authoritative.

    It carries no ``structure`` key, because a stated reason has no question and
    answer for a model to have paired. A document that says nothing here resolves
    to ``anchored`` on read, which is the truth for a reason someone declared; a
    reconstruction is already labelled by ``rationale_source`` on the same
    document.

    ``metadata`` is read here, and reading it is what makes the character policy's
    note and the forge comment id reach the document at all. It was not read before,
    and the cost was specific rather than general: a stored rationale was the only
    sanitised record in the store that could not say what had been taken out of it,
    so its silence looked like a fidelity claim and was not one — the extractors
    remove fourteen of the same characters silently on the way in. That is the
    defect ``docs/tanseki-seam.md`` had to state as a limit.

    ``github_comment_id`` belongs here, and the argument is that a rationale's only
    anchor is the comment it was declared in. Every other record kind either has a
    delivery id or a re-fetchable read id in its document, and this was the one that
    had neither while the id sat in the model: nothing in the stored document
    invited the comparison that is the entire reason to keep a self-asserted record
    at all. Absent rather than empty for a branch-only declaration, which was never
    posted anywhere and has no comment to name — and no channel test is needed to
    get that right, because the id is in ``metadata`` only when a comment was
    actually made.

    The design phase's own metadata is read here too -- role, plan digest, proposal
    ids, discards -- for the reason the absence rule gives rather than despite it:
    a proposal carries none of the reconciliation's keys, so ``metadata.get`` yields
    ``None`` and they stay absent. What reaches the document is what the record
    holds, and nothing else is invented on the way.
    """
    frontmatter: dict[str, Any] = {
        "title": f"Rationale {rationale.entry_id}"[:200],
        "author": rationale.declared_by,
        "tags": ["rationale", f"rationale_{rationale.source.value}"],
        "updated_at": rationale.declared_at.isoformat(),
        "capture_source": rationale.capture_source.value,
    }
    extra: dict[str, Any] = {
        "rationale_source": rationale.source.value,
        "rationale_revision": rationale.revision,
        "rationale_revises": rationale.revises,
        "repo": rationale.repo,
        "pr": rationale.pr_number,
        "branch": rationale.branch or None,
        "declared_by": rationale.declared_by,
        # Absent rather than a placeholder when no model was stated. Nothing in the
        # platform verifies which model drafted a comment, so writing "unknown"
        # would put a string in the field a reader uses to learn the model -- and a
        # reader cannot tell an honest absence from a stated unknown.
        "declared_by_model": rationale.declared_model,
        "declared_at": rationale.declared_at.isoformat(),
        # The forge comment this declaration was read from, so a reader holding the
        # document can re-fetch the original and check the wording against it. An
        # int from the forge, so there is no third-party text here to sanitise.
        # Spelled as the literal rather than given a constant beside
        # :data:`CHANGE_AUTHOR_KEY`, because the writer that populates it
        # (``rationale_collector``) names it as a literal and I do not own that file;
        # a constant here would be a second spelling of one key inside one module,
        # which is the drift the named keys above exist to prevent.
        "github_comment_id": rationale.metadata.get("github_comment_id"),
        # What the character policy took out of the prose, if anything. Absent when
        # nothing was, and that absence is now a real claim rather than an artefact
        # of this function not looking: the prose was sanitised before it was stored,
        # so a document with no key says the reader is holding the declaration's own
        # wording. Same key, same shape and same rule as every other record kind --
        # see :data:`kojutsu.core.text_hygiene.SANITISATION_KEY`, which is why it is
        # named there rather than spelled here.
        SANITISATION_KEY: rationale.metadata.get(SANITISATION_KEY),
        # Which of the two design record kinds this is, and -- on a reconciliation --
        # the plan it produced, the proposals it covered, and what it discarded.
        # Absent on any record that is not one: ``metadata.get`` returns ``None``
        # and :func:`_unrecorded` drops it, which is the existing absence rule and
        # not a new one. A proposal carries no digest, ids or discards, so those
        # keys are absent rather than empty -- there was nothing to put in them.
        #
        # ``[]`` is deliberately *not* absent: :func:`_unrecorded` treats an empty
        # collection as a result ("considered and discarded none") rather than a
        # gap ("no discard information"), so a reconciliation that kept every
        # proposal renders ``design_discarded: []`` and keeps that claim.
        #
        # None of these names ends in "tags", so the name-match bound in
        # :func:`_validate_frontmatter_value` does not touch them -- correct, but
        # worth saying, because a future ``design_tags`` would trip it silently.
        DESIGN_ROLE_KEY: rationale.metadata.get(DESIGN_ROLE_KEY),
        DESIGN_PLAN_DIGEST_KEY: rationale.metadata.get(DESIGN_PLAN_DIGEST_KEY),
        DESIGN_PROPOSAL_IDS_KEY: rationale.metadata.get(DESIGN_PROPOSAL_IDS_KEY),
        DESIGN_DISCARDED_KEY: rationale.metadata.get(DESIGN_DISCARDED_KEY),
    }
    _write_extras(frontmatter, extra)
    return frontmatter


def _rationale_body(rationale: RationaleEntry) -> str:
    """Render a rationale with the reason ahead of the attribution.

    The declared model is a *self-assertion* by the comment author, never
    something the platform verified, so it sits under the reason rather than
    above it: a reader skimming should meet the reason first and the claim about
    who produced it second, not the other way round.
    """
    lines = [
        f"# Rationale {rationale.revision}",
        "",
        f"## Reason\n{rationale.rationale_text.strip()}\n",
        "## Attribution",
        f"Declared by: {rationale.declared_by}",
        f"Declared model: {rationale.declared_model or UNKNOWN_MODEL} (asserted by the author, "
        "not verified by the platform)",
        f"Source: {rationale.source.value}",
    ]
    if rationale.revises:
        lines.append(f"Supersedes: {rationale.revises}")
    return "\n".join(lines) + "\n"


def build_rationale_content(rationale: RationaleEntry) -> str:
    """Canonical Markdown for a rationale: frontmatter block then body."""
    block = serialize_frontmatter(build_rationale_frontmatter(rationale))
    body = _rationale_body(rationale)
    return f"---\n{block}---\n{body}" if block else body


def to_rationale_upsert_payload(
    rationale: RationaleEntry,
    *,
    author: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the Tanseki upsert payload for a rationale."""
    document_id = rationale_document_id(rationale)
    return {
        "id": document_id,
        "path": f"{document_id}.md",
        "content": build_rationale_content(rationale),
        "frontmatter": build_rationale_frontmatter(rationale),
        "author": author or rationale.declared_by or "unknown",
        "message": message or f"Rationale {rationale.entry_id}",
    }


#: The tag that marks a document as a quotation rather than an answer. The read
#: path keys on this, exactly as a rationale is keyed on its own tag, because the
#: document is the only thing the store hands back: a tag in frontmatter is the one
#: piece of the record that survives the round trip without a second lookup.
CLARIFICATION_TAG = "clarification"


def clarification_document_id(clarification: ClarificationEntry) -> str:
    """Path-derived document id for a clarification, outside the answer namespace.

    A clarification answers no question, so it must not land at
    ``<repo>/pr-<n>/<entry_id>`` where it would be indistinguishable from an
    answer. The ``clarification/`` segment is what tells a reader browsing the
    store that this document is something a person volunteered, with no question
    anywhere near it.
    """
    return (
        f"{clarification.repo}/pr-{clarification.pr_number}"
        f"/{CLARIFICATION_TAG}/{clarification.entry_id}"
    )


def build_clarification_frontmatter(clarification: ClarificationEntry) -> dict[str, Any]:
    """Flat frontmatter for a quoted human statement, carrying its anchor always.

    ``capture_source`` is written unconditionally, for the same reason it is on an
    entry and on a rationale: a reader must be able to tell, from the document
    alone, what is behind it. Unlike a rationale's, it is written *with* its
    anchors, because unlike a rationale there is a delivery or a re-fetchable
    comment id behind this one.

    There is no ``category`` key, and none can be defaulted into being useful: a
    clarification has no question, so any of the six retrospective values would be
    a claim about a decision nobody was asked to make.

    ``metadata`` is read for one key, and reading it is not optional bookkeeping.
    A clarification is the record kind whose author can be anyone, and the
    character policy removes what its stored statement cannot render as
    (``clarification_collector``); a document that could not say so would make the
    key's absence mean "never needed sanitising" when it means "this mapper never
    looked", and those are different statements about the same quotation. The anchor
    argument in that module's docstring is what makes the surviving wording quotable
    at all, and a note that never reaches the store leaves the reader with the
    claim and none of the evidence for it.
    """
    tags = [CLARIFICATION_TAG, *clarification.tags]
    if clarification.is_agent_authored:
        tags.append("agent_authored")
    # Deduped because the kind tag is owned here and the collector also supplies it,
    # so a caller that names the kind in its own tags would otherwise store
    # `["clarification", "clarification", ...]` and any tag tally would count the
    # record twice. Order-preserving, so the kind tag stays first. A guard on
    # ``agent_authored`` alone left this open, which is the usual way a rule that
    # exists in two places drifts.
    tags = list(dict.fromkeys(tags))
    frontmatter: dict[str, Any] = {
        "title": f"Clarification {clarification.entry_id}"[:200],
        "author": clarification.author,
        "tags": tags,
        "updated_at": clarification.declared_at.isoformat(),
        "capture_source": clarification.capture_source.value,
    }
    extra: dict[str, Any] = {
        "repo": clarification.repo,
        "pr": clarification.pr_number,
        "github_comment_id": clarification.github_comment_id,
        "github_author_association": clarification.author_association,
        # Whether the quoting account is an application, read from the entry's metadata
        # rather than re-derived here for the same reason the forge's comment id is:
        # a rule that exists in two places is a rule that drifts.
        "comment_author_is_machine": clarification.metadata.get("comment_author_is_machine"),
        "clarified_by_agent": clarification.authored_by_agent,
        # A declared model is a self-assertion by the comment author, exactly as it
        # is for an answer, and is recorded as ``unknown`` rather than omitted so
        # a reader can tell "named no model" from "not recorded here".
        "clarified_by_model": clarification.authored_by_model or UNKNOWN_MODEL,
        "declared_at": clarification.declared_at.isoformat(),
        "captured_at": (
            clarification.captured_at.isoformat() if clarification.captured_at else None
        ),
        "delivery_id": clarification.capture_delivery_id,
        # What the character policy took out of the quotation, if anything. Absent
        # when nothing was, and the absence is the claim: the stored wording is the
        # comment's own bytes, which is the only reason it may be quoted as the
        # author's. Same key and same rule as every other sanitised record kind.
        SANITISATION_KEY: clarification.metadata.get(SANITISATION_KEY),
    }
    _write_extras(frontmatter, extra)
    # Written only when it is not the default, so an anchored clarification stays
    # byte-identical to one stored before the axis existed. Absence reads back as
    # anchored rather than as unknown, which is the same trade the axis makes on an
    # entry and for the same reason.
    if clarification.structure is not RecordStructure.ANCHORED:
        frontmatter["structure"] = clarification.structure.value
        # The inferring model lives in ``metadata`` on an entry, and a
        # clarification keeps that shape rather than growing a second one: a rule
        # that exists in two forms is a rule that drifts.
        frontmatter[STRUCTURE_INFERRED_BY] = clarification.metadata.get(STRUCTURE_INFERRED_BY)
    return frontmatter


def _clarification_body(clarification: ClarificationEntry) -> str:
    """Render a clarification with the statement ahead of the attribution.

    The quoted text comes first and the attribution after it, for the reason
    ``_rationale_body`` does the same: a name in the first line lends the claim more
    authority than it has earned. A reader skimming should meet the words the
    person actually wrote before meeting who wrote them, and should be able to see
    the words on their own — which is also the only way to check them against the
    comment they were quoted from.
    """
    lines = [
        "# Clarification",
        "",
        f"## Statement\n{clarification.statement.strip()}\n",
        "## Attribution",
        f"Author: {clarification.author}",
        f"Author association: {clarification.author_association}",
        f"Comment: {clarification.github_comment_id}",
    ]
    if clarification.authored_by_agent:
        lines.append(
            f"Authored by agent: {clarification.authored_by_agent} "
            "(declared in the comment, not verified by the platform)"
        )
    lines.append(
        f"Authored by model: {clarification.authored_by_model or UNKNOWN_MODEL} "
        "(asserted by the author, not verified by the platform)"
    )
    lines.append(f"Capture source: {clarification.capture_source.value}")
    return "\n".join(lines) + "\n"


def build_clarification_content(clarification: ClarificationEntry) -> str:
    """Canonical Markdown for a clarification: frontmatter block then body."""
    block = serialize_frontmatter(build_clarification_frontmatter(clarification))
    body = _clarification_body(clarification)
    return f"---\n{block}---\n{body}" if block else body


def to_clarification_upsert_payload(
    clarification: ClarificationEntry,
    *,
    author: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the Tanseki upsert payload for a clarification.

    Same delivery as every other record — the sink and the outbox are shared, and
    this adds no second path for a failure to be lost in.
    """
    document_id = clarification_document_id(clarification)
    return {
        "id": document_id,
        "path": f"{document_id}.md",
        "content": build_clarification_content(clarification),
        "frontmatter": build_clarification_frontmatter(clarification),
        "author": author or clarification.author or "unknown",
        "message": message or f"Clarification {clarification.entry_id}",
    }


#: The tag that marks a document as a measurement of this system rather than a
#: record of anything it was pointed at. The read path keys on this, exactly as a
#: rationale is keyed on its own tag, because a report about a model served beside
#: a review of a repository is a number a reader will attribute to the repository
#: unless the document says otherwise in the one place the store hands back whole.
EVALUATION_TAG = "evaluation"

#: The keys the read path has to surface for a stored measurement not to be
#: readable as a review record. ``evaluation_target`` is the load-bearing one: a
#: reader who can see that the document measures a model cannot mistake a good
#: score for a property of the code, and ``evaluated_model`` is the specificity
#: that makes the number weighable rather than a claim about models in general.
EVALUATION_FRONTMATTER_KEYS = (
    "evaluation_target",
    "evaluated_model",
    "evaluated_measurement",
    "evaluation_scope",
)


def _single_line(text: str) -> str:
    """Collapse prose to one line, because a frontmatter value must be one.

    The store rejects a frontmatter value containing a newline, and it is right to:
    a value with a blank line in it renders as a broken document in any YAML reader,
    and the eleven anchored records it sits beside are all single-line. The scope of
    a measurement is several paragraphs by nature, so the header carries a
    single-line form of it and :func:`_evaluation_body` carries the whole thing.
    Collapsing rather than truncating is what keeps the header's claim and the
    body's claim the same claim.
    """
    return " ".join(text.split())


def evaluation_document_id(entry: EvaluationEntry) -> str:
    """Path-derived document id for a measurement, outside every record namespace.

    Three namespaces now, and each one exists for the same reason: a document's
    path is the only part of it a reader meets before deciding what kind of thing
    they are holding. An answer lands at ``<repo>/pr-<n>/<entry_id>``, a
    clarification under ``clarification/``, a rationale under ``rationale/``, and
    a measurement under ``evaluation/``.

    The last is the one that would hurt most to get wrong. A precision figure
    filed beside the answers for a pull request looks, to every surface that
    groups by path, like a fact about that pull request -- and unlike a rationale
    or a clarification, nothing in the body of an answer says otherwise. A
    reader's trust in the eleven real records is the thing a stray number is most
    able to spend.
    """
    return f"{entry.repo}/pr-{entry.pr_number}/{EVALUATION_TAG}/{entry.entry_id}"


def build_evaluation_frontmatter(entry: EvaluationEntry) -> dict[str, Any]:
    """Flat frontmatter for a measurement, carrying its subject and its scope.

    ``capture_source`` is written unconditionally, for the reason it is on an
    entry, a rationale and a clarification: a reader must be able to tell from the
    document alone that this is a claim rather than a capture.

    It carries no ``category``, for the same reason a clarification carries none.
    A measurement has no question a category could be retrospective about, and the
    nearest fit in the vocabulary would be a false claim about the repository the
    harness was pointed at. What replaces it is ``evaluation_target``: what the
    document is a measurement *of*, which is the fact a reader needs and the one
    no question category can express.

    It carries no ``structure`` either, and the absence is meaningful rather than
    an omission. A document that states nothing here resolves to ``anchored`` on
    read, which is true of the report: the harness wrote both halves of it. The
    model-inferred material is *inside* the measurement, named by
    ``evaluated_model``, and labelling the document itself ``inferred`` would say
    the report's own question and answer were matched up by a model, which they
    were not.
    """
    frontmatter: dict[str, Any] = {
        "title": f"Evaluation {entry.measurement}"[:200],
        "author": "kojutsu-evaluation",
        "tags": [EVALUATION_TAG, entry.target.value],
        "updated_at": entry.measured_at.isoformat(),
        "capture_source": entry.capture_source.value,
    }
    extra: dict[str, Any] = {
        "repo": entry.repo,
        "pr": entry.pr_number,
        "evaluation_target": entry.target.value,
        "evaluated_model": entry.subject,
        "evaluated_measurement": entry.measurement,
        "measured_at": entry.measured_at.isoformat(),
        # The limits travel in the frontmatter as well as the body, because the
        # frontmatter is what a listing shows and a body is what a reader has to
        # open. A scope that only appears in the prose is a scope nobody meets
        # until they have already quoted the number. One line, because a
        # frontmatter value with a newline in it is rejected by the store; the
        # paragraphs are in the body under their own heading.
        "evaluation_scope": _single_line(entry.scope),
    }
    _write_extras(frontmatter, extra)
    return frontmatter


def _evaluation_body(entry: EvaluationEntry) -> str:
    """Render a measurement with its result ahead of its subject, and its limits loud.

    The numbers come before the attribution for the reason ``_rationale_body`` puts
    the reason first: a name lends a claim authority it has not earned, and a model
    id in the first line of a report reads as a certification. The subject and the
    limits follow, and the limits are a section rather than a closing line, so a
    reader who stops after the figures has still been told what they are.
    """
    return (
        f"# Evaluation {entry.measurement}\n\n"
        f"## Result\n{entry.result_text.strip()}\n\n"
        "## Subject\n"
        f"Measured: {entry.target.value}\n"
        f"Model: {entry.subject}\n"
        f"Measured at: {entry.measured_at.isoformat()}\n"
        f"Capture source: {entry.capture_source.value} (a number this system produced "
        "about a model it ran; nothing outside the process signed it)\n\n"
        f"## What these numbers can and cannot stand for\n{entry.scope.strip()}\n"
    )


def build_evaluation_content(entry: EvaluationEntry) -> str:
    """Canonical Markdown for a measurement: frontmatter block then body."""
    block = serialize_frontmatter(build_evaluation_frontmatter(entry))
    body = _evaluation_body(entry)
    return f"---\n{block}---\n{body}" if block else body


def to_evaluation_upsert_payload(
    entry: EvaluationEntry,
    *,
    author: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the Tanseki upsert payload for a measurement.

    The same delivery as every other record -- sink and outbox shared, no second
    path for a failure to be lost in -- with an id that is stable across re-runs, so
    measuring the same thing twice updates one document instead of accumulating
    one per harness invocation.
    """
    document_id = evaluation_document_id(entry)
    return {
        "id": document_id,
        "path": f"{document_id}.md",
        "content": build_evaluation_content(entry),
        "frontmatter": build_evaluation_frontmatter(entry),
        "author": author or "kojutsu-evaluation",
        "message": message or f"Evaluation {entry.measurement}",
    }


def question_document_id(question: QuestionRecord) -> str:
    """Path-derived id for a projected question, outside the answer namespace.

    A question is not an answer, so it must never land where a reader would take
    it for one — the same reason a rationale lives under ``rationale/``. The id is
    derived from ``(repo, pr, question_id)`` and never from the status, so
    re-projecting an unchanged question upserts the same document rather than
    creating a second one.
    """
    pr_number = question.pr_number if question.pr_number is not None else 0
    return f"{question.repo}/pr-{pr_number}/{QUESTION_NAMESPACE}/{question.question_id}"


def build_question_frontmatter(question: QuestionRecord) -> dict[str, Any]:
    """Frontmatter for a decision request.

    The status is written *with* the time it was observed, as ``status_as_of``,
    because the projection is eventually consistent: a reader that cannot see how
    stale a status is will assume it is current, and will compute a wait that
    includes time the store never saw. A status with no age attached is a claim
    about *now* made by a document about the past.

    No ``capture_source`` and no ``independence``, deliberately, and the omission
    is the mechanism rather than an oversight: a question is a request for a
    reason, so it carries neither, and that is what excludes it from an
    evidence-only query. Writing either field would be a way to make a request
    look like a checked conclusion.
    """
    frontmatter: dict[str, Any] = {
        "title": f"Decision request {question.question_id}"[:200],
        "author": question.question_author or "unknown",
        "tags": ["question", f"question_{question.status}"],
        "status_as_of": (
            question.updated_at or question.answered_at or datetime.now(UTC)
        ).isoformat(),
    }
    extra: dict[str, Any] = {
        "question_id": question.question_id,
        "question_status": question.status,
        "repo": question.repo,
        "pr": question.pr_number,
        "pr_url": question.pr_url,
        "category": question.category,
        "jira": question.jira_ticket_key,
        "session_id": question.session_id,
        "question_author": question.question_author,
        "assignee": question.assignee,
        "attempts": question.attempts,
        "answer_comment_id": question.answer_comment_id,
        "created_at": question.created_at.isoformat() if question.created_at else None,
        "answered_at": question.answered_at.isoformat() if question.answered_at else None,
    }
    _write_extras(frontmatter, extra)
    return frontmatter


def _question_body(question: QuestionRecord) -> str:
    return (
        f"# Decision request: {question.status}\n\n"
        f"## Request\n{question.question_text.strip()}\n\n"
        "## Attribution\n"
        f"Asked by: {question.question_author or 'not stated'}\n"
        f"Reached: {question.status} after {question.attempts} attempt(s)\n"
        "\nThis is a request for a reason, not a record of one. It carries no\n"
        "independence level and is excluded from evidence-only queries, because\n"
        "nobody stated anything here.\n"
    )


def build_question_content(question: QuestionRecord) -> str:
    block = serialize_frontmatter(build_question_frontmatter(question))
    body = _question_body(question)
    return f"---\n{block}---\n{body}" if block else body


def to_question_upsert_payload(
    question: QuestionRecord,
    *,
    author: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the Tanseki upsert payload for a projected decision request."""
    document_id = question_document_id(question)
    return {
        "id": document_id,
        "path": f"{document_id}.md",
        "content": build_question_content(question),
        "frontmatter": build_question_frontmatter(question),
        "author": author or question.question_author or "unknown",
        "message": message or f"Decision request {question.question_id} ({question.status})",
    }


def census_document_id(record: CensusRecord) -> str:
    """Path-derived id for an observation, keyed on the *change* rather than the event.

    ``(repo, pr, action)`` and never the delivery id, for the reason
    :class:`~kojutsu.models.CensusRecord` gives: the natural key of "this change
    was observed once with nothing captured" is the change, so two deliveries of the
    same action upsert one document instead of accumulating records. That is what
    makes a count over census documents a count over changes rather than a count over
    the traffic that produced them.
    """
    pr_number = record.pr_number if record.pr_number is not None else 0
    return f"{record.repo}/pr-{pr_number}/{CENSUS_NAMESPACE}/{record.action}"


def build_census_frontmatter(record: CensusRecord) -> dict[str, Any]:
    """Frontmatter for an observation.

    Carries ``capture_source`` and ``delivery_id`` so the observation is held to the
    same standard as a capture: a reader can re-fetch the delivery and see that it
    really was processed. Deliberately absent: ``independence`` (nobody was
    positioned to disagree with a non-event) and any reason for the absence (see the
    model). The omissions are the mechanism, the same way they are on a question —
    a field that existed and was left empty would be a place for a later writer to
    put a guess.
    """
    frontmatter: dict[str, Any] = {
        "title": f"Observed with nothing captured: {record.entry_id}"[:200],
        "author": record.change_author_account or UNKNOWN_MODEL,
        "tags": ["census"],
        "capture_source": record.capture_source.value,
    }
    extra: dict[str, Any] = {
        "census_id": record.entry_id,
        "repo": record.repo,
        "pr": record.pr_number,
        "pr_url": record.pr_url,
        "action": record.action,
        CHANGE_AUTHOR_KEY: record.change_author_account,
        HEAD_SHA_KEY: record.head_sha,
        RECORD_KIND_KEY: "census",
        "observed_at": record.observed_at.isoformat(),
        "delivery_id": record.delivery_id,
    }
    _write_extras(frontmatter, extra)
    return frontmatter


def _census_body(record: CensusRecord) -> str:
    # The wording is the claim, so it is written out rather than left to a reader to
    # infer from an absent answer. "No knowledge" would be a claim about the change;
    # an empty answer body would read as a capture that found nothing to say. Both
    # are more flattering, and neither is what happened: a delivery arrived, was
    # processed, and produced no record.
    return (
        "# Observed: nothing was captured\n\n"
        "A delivery for this change was processed and produced no knowledge record.\n"
        "That is a statement about one event, not about the change: a change opened\n"
        "quietly and reviewed later with a capture has both, and only this record\n"
        "exists because nothing was written at the time.\n\n"
        "## What this does not say\n"
        "It does not say nobody looked, and it does not say why nothing was captured.\n"
        "Kojutsu cannot know that — no comment, no authorised reviewer, a malformed\n"
        "marker, a change nobody reviewed are each a guess about intent, so none is\n"
        "recorded. Read the forge for the reason.\n\n"
        f"Delivery: {record.delivery_id}\n"
        f"Action: {record.action}\n"
        f"Observed at: {record.observed_at.isoformat()}\n"
    )


def build_census_content(record: CensusRecord) -> str:
    block = serialize_frontmatter(build_census_frontmatter(record))
    body = _census_body(record)
    return f"---\n{block}---\n{body}" if block else body


def to_census_upsert_payload(
    record: CensusRecord,
    *,
    author: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the Tanseki upsert payload for an observation."""
    doc_id = census_document_id(record)
    return {
        "id": doc_id,
        "path": f"{doc_id}.md",
        "content": build_census_content(record),
        "frontmatter": build_census_frontmatter(record),
        "author": author or record.change_author_account or UNKNOWN_MODEL,
        "message": message or f"Observed with nothing captured ({record.action})",
    }


def _validate_frontmatter_value(value: Any, path: str = "frontmatter") -> None:
    """Reject a frontmatter value this mapping cannot honestly write.

    Recursive, and deliberately so: the bound and the character checks apply to
    every value *wherever* it sits, so a list of long strings or a nested map is
    held to the same standard as a scalar. Validating only the top level would let
    a document be bounded per key and unbounded in aggregate, which is the shape of
    input that makes a store reject a write its caller believed was checked.

    **The text bound is per value, not per document.** ``MAX_FRONTMATTER_TEXT_CHARS``
    (10,000) applies to each string individually, so a document with thirty extras
    may legitimately total 300,000 characters and still pass here. This is a
    different axis from the store's, and neither bound implies the other:

    - Tanseki's ``MAX_FRONTMATTER_BYTES`` is 1 MiB over the *serialized block*
      (``RequestLimits.kt:17``), checked on its side of the seam.
    - This one is per value, and exists to stop one key's value being a document
      in its own right -- a 10,000-character ``pr_url`` is already a bug in the
      caller, not a fact worth storing.

    So a document can pass here and still be refused by the store, and that is not
    a contradiction: passing here means "nothing here is unreasonable in itself".
    It does not mean "the store will accept it", and it must not be read as a
    pre-flight check for the seam. What would *not* be acceptable is the reverse --
    a document refused for a limit the caller was never told about.

    **The one structural exception is ``tags``**, via ``path.endswith("tags")``.
    ``tags`` is a contract key with its own bound (``MAX_FRONTMATTER_TAGS``) because
    the store models it as a typed field rather than an extra. The match is on the
    key's *name* and is therefore loose: an extra called ``related_tags`` or
    ``pr_tags`` lands here too. That is deliberate and conservative -- a list-valued
    extra is bounded by the same reasoning as ``tags`` -- but it is a name match,
    not a key registry, and it is the sort of thing that surprises whoever next
    adds an extra. Everything else recurses with no per-key bound of its own.

    Unsupported types raise rather than being coerced. ``datetime``, ``date``,
    ``set`` and ``bytes`` were previously silently ``str()``-ed by the merge loop,
    and ``str(datetime)`` is a malformed non-ISO timestamp rather than a timestamp
    -- so the coercion produced something that looked recorded and was not. A loud
    failure is the better of the two. ``StrEnum`` is accepted, since it is an
    ``isinstance`` of ``str``.

    Non-finite floats (``nan``, ``inf``, ``-inf``) are rejected by name. They are
    ``float`` instances, so without this they would pass as numbers -- and then
    serialise as bare ``NaN``/``Infinity`` literals, which are not valid JSON, on
    the upsert body. Ordinary finite floats, including ``0.0``, negatives and
    very large magnitudes, are accepted unchanged.
    """
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(f"{path} contains a NUL character")
        if len(value) > MAX_FRONTMATTER_TEXT_CHARS:
            raise ValueError(f"{path} exceeds the {MAX_FRONTMATTER_TEXT_CHARS}-character limit")
        return
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} is a non-finite float; JSON has no literal for it")
        return
    if isinstance(value, (list, tuple)):
        if path.endswith("tags") and len(value) > MAX_FRONTMATTER_TAGS:
            raise ValueError(f"{path} exceeds the {MAX_FRONTMATTER_TAGS}-tag limit")
        for index, item in enumerate(value):
            _validate_frontmatter_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise ValueError(f"{path} contains an invalid key")
            _validate_frontmatter_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def serialize_frontmatter(frontmatter: dict[str, Any]) -> str:
    """Safely render canonical YAML with known keys first and sorted extras.

    The absence rule here is :func:`_unrecorded` -- the same predicate the six
    builders admit values through -- so a key cannot be dropped by the renderer for
    one reason and by its builder for another. That used to be two rules that
    happened to agree, which is how ``None``/``""``/``[]`` here and ``None``/``""``
    in the builders came to be written down separately in the first place.

    One thing is added on top, and it is a rendering decision rather than a second
    definition: an **empty collection on a known key is dropped**. ``tags`` is the
    only key that can be one, and an untagged document should carry no ``tags``
    line at all instead of a line asserting an empty list -- the typed keys have a
    shape in Tanseki's contract and a blank one is noise. Extras are deliberately
    not treated this way; see :func:`_write_extras` for why an empty list there is
    a result rather than a gap.
    """
    _validate_frontmatter_value(frontmatter)
    ordered: dict[str, Any] = {}
    for key in _KNOWN_KEYS:
        if key not in frontmatter:
            continue
        value = frontmatter[key]
        if _unrecorded(value) or (isinstance(value, list) and not value):
            continue
        ordered[key] = value
    for key in sorted(frontmatter):
        if key not in _KNOWN_KEYS:
            ordered[key] = frontmatter[key]
    if not ordered:
        return ""
    return yaml.dump(
        ordered,
        Dumper=_FrontmatterDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=MAX_FRONTMATTER_TEXT_CHARS,
    )


def build_content(entry: KnowledgeEntry) -> str:
    """Canonical Markdown: frontmatter block followed by the body."""
    block = serialize_frontmatter(build_frontmatter(entry))
    body = _body(entry)
    return f"---\n{block}---\n{body}" if block else body


def to_upsert_payload(
    entry: KnowledgeEntry,
    *,
    author: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the Tanseki upsert payload for an entry (collection added by the client)."""
    return {
        "id": document_id(entry),
        "path": document_path(entry),
        "content": build_content(entry),
        "frontmatter": build_frontmatter(entry),
        "author": author or entry.author or "unknown",
        "message": message or f"Capture decision {entry.entry_id}",
    }
