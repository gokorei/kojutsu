"""Domain models for knowledge capture."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: The forge's own identifier for a review, as it appears in ``metadata``.
#:
#: Spelled here rather than in ``core.tanseki_mapping``, which is otherwise the home
#: of the frontmatter vocabulary, because :func:`capture_anchor_gaps` needs it to
#: validate a backfilled record and ``tanseki_mapping`` already imports this module —
#: so the dependency cannot run the other way. ``tanseki_mapping`` re-exports it,
#: which keeps one spelling of the key rather than two that can drift.
REVIEW_ID_KEY = "review_id"


class CaptureSource(StrEnum):
    """How a knowledge entry came to exist.

    This is the trust axis of the whole system:

    - ``WEBHOOK`` — produced by the capture pipeline from a real, signed,
      allow-listed provider delivery. Checkable against the provider's own
      delivery id.
    - ``COLLECT`` — produced by an authenticated read of the provider API (the
      ``kojutsu collect`` path). Also real evidence, but its checkable anchor
      is the source comment id, which any reader can re-fetch.
    - ``ASSERTED`` — a caller typed it in. It may well be true, but nothing in the
      system verified any part of it.
    - ``BACKFILLED`` — reconstructed from history by an authenticated read that
      happened *after* the event, by a process that was not present for it. The
      strongest thing that can be said about such a record is that it matches the
      forge *now*; it does not claim to match the forge then, and a comment edited,
      rewritten or deleted since is indistinguishable from one that was not.

    The default is deliberately ``ASSERTED``: a record has to earn the right to be
    called captured. An unmarked record is never treated as evidence.

    **``BACKFILLED`` makes this axis answer two questions at once, deliberately.**
    The other three values are all about *how* the text was obtained;
    ``BACKFILLED`` is the only one that also says *when*. A second axis would keep
    each question to one thing, and was rejected for the schema cost — so the cost
    is paid here instead, and the mitigation is that the value's own name says
    ``backfilled``. A reader filtering on ``capture_source`` is filtering on both
    axes, and this docstring is where that is stated rather than left to be
    discovered by a reader who assumed otherwise.

    It is named ``BACKFILLED`` and not ``RECONSTRUCTED`` because
    :class:`RationaleSource` already has a ``RECONSTRUCTED`` meaning something
    quite different: a model *inferred* a reason from a diff. Two axes using one
    word for different things, over records both apply to, is precisely the
    confusion ``core.rationale_link`` exists to prevent.
    """

    WEBHOOK = "webhook"
    COLLECT = "collect"
    ASSERTED = "asserted"
    BACKFILLED = "backfilled"


def _is_positive_int(value: object, *, allow_string: bool) -> bool:
    """True for a positive integer, or the string spelling of one.

    ``allow_string`` exists because the two callers genuinely differ. The write
    path must reject a string ``"7"`` where an integer was required, and a test
    pins exactly that. The read path cannot: rows written before frontmatter
    extras kept their native types still carry the string spelling, so a document
    read back from the store may carry ``"7"`` where the model holds ``7`` --
    and those rows are already written, so the tolerance has to outlive the
    change. An int-only rule there would flag every legacy record as
    unanchored — a check that reports everything and therefore reports nothing.

    Booleans are rejected explicitly, since ``bool`` is an ``int`` and ``True``
    would otherwise pass as ``1``.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 1
    if allow_string and isinstance(value, str):
        return value.strip().isdigit() and int(value.strip()) >= 1
    return False


def capture_anchor_gaps(
    *,
    capture_source: CaptureSource,
    repo: object,
    pr_number: object,
    captured_at: object,
    delivery_id: object,
    comment_id: object,
    allow_string_numbers: bool = False,
    check_id: object = None,
    review_id: object = None,
) -> list[str]:
    """Return the anchors a captured record is missing, in a stable order.

    The one definition of "captured means checkable", used by the write-time
    validator and by the read path. It lives here rather than in either caller
    because a rule that exists in two places drifts, and the read path is the one
    that decides what a reader is allowed to conclude: a document in the store
    claiming a signed delivery with no delivery id is otherwise served as
    verified evidence, because the constructor check ran once in a different
    process and the store is a separate service.

    Pass ``allow_string_numbers=True`` only when checking a value read back from
    storage, where every number has been stringified on the way in.

    Returns an empty list for ``ASSERTED``, which is required to carry no anchors
    precisely because it claims no capture.

    ``check_id`` substitutes for ``pr_number`` and nothing else. A check run on a
    branch has no pull request, and the rule requiring one would force a choice
    between two bad answers: store a record whose pr_number is invented, or
    downgrade a genuine signed delivery to ``asserted`` because the fact it
    records happens not to be about a pull request. The check run's own id is the
    stronger anchor of the two — it is forge-issued, globally unique, and names
    the exact run — so accepting it *adds* a checkable anchor rather than
    weakening the requirement. Nothing else may substitute: a record with neither
    a pr_number nor a check_id is still missing what makes it checkable.

    ``review_id`` exists for ``BACKFILLED`` alone, and the rule it feeds is
    different in kind from every other one here. ``WEBHOOK`` is anchored to the
    delivery that produced it and ``COLLECT`` to the comment that was read; a
    backfilled record has no delivery at all — nothing was delivered, the system
    read history on its own initiative — so there is no id of that kind to ask for
    and asking would push the implementation toward inventing one. What stands
    behind it instead is *the read*: which forge object was read, and when.

    So ``BACKFILLED`` requires the repo, the change, ``captured_at`` and the id of
    whatever was read (``github_comment_id`` or ``review_id``), and specifically
    does **not** require ``delivery_id``. ``captured_at`` keeps its documented
    meaning — when *the system* observed the event — which for a backfill is when
    the read happened; that is the only evidence there is, so a record missing it
    cannot say what it is a reading of.

    A backfilled record therefore rests on a weaker guarantee than a webhook one:
    it shows what the forge says now, not what it said at the time. The anchor
    rule cannot make that stronger, only make the weakness visible.
    """
    if capture_source is CaptureSource.ASSERTED:
        return []

    missing: list[str] = []
    if not str(repo or "").strip():
        missing.append("metadata.repo")
    has_pr = _is_positive_int(pr_number, allow_string=allow_string_numbers)
    has_check = bool(str(check_id or "").strip())
    if not has_pr and not has_check:
        missing.append("metadata.pr_number")
    if captured_at is None:
        missing.append("captured_at")

    if capture_source is CaptureSource.WEBHOOK:
        if not str(delivery_id or "").strip():
            missing.append("capture_delivery_id")
    elif capture_source is CaptureSource.BACKFILLED:
        # Either forge object satisfies it: whichever one the backfill read. The
        # gap is named for the *read* rather than for either id, because a reader
        # who sees it should learn that the record cannot name what it read — not
        # that they should go looking for a delivery that never existed.
        read_comment = _is_positive_int(comment_id, allow_string=allow_string_numbers)
        read_review = _is_positive_int(review_id, allow_string=allow_string_numbers)
        if not read_comment and not read_review:
            missing.append("capture_read_anchor")
    elif not _is_positive_int(comment_id, allow_string=allow_string_numbers):
        missing.append("metadata.github_comment_id")
    return missing


class RecordStructure(StrEnum):
    """Whether a record's own shape was established, or inferred by a model.

    A second axis, and deliberately not an extension of :class:`CaptureSource`.
    That one answers *where did this text come from*; this one answers *was this
    structure established or inferred*. A record can need both answers at once,
    and the case where it matters is not exotic: a capture with a real delivery
    id and a real GitHub comment id, whose question/answer pairing a model
    matched up rather than anyone actually asking and answering. That record is
    completely checkable *as text* and unreadable *as a conversation*, and no
    value of ``CaptureSource`` can express the difference. Without this axis a
    reader holding an inferred pairing sees a row identical to a row where a
    human was asked and answered, and has no way to prefer the honest one.

    - ``ANCHORED`` — the pairing was established: a question was asked, an answer
      captured, and the record joins them on something a reader can check.
    - ``INFERRED`` — a model inferred the pairing. Allowed in the store, because
      an inferred record is still knowledge and refusing it would lose the
      record; not allowed to be silent, because it must name the model that
      inferred it.

    The default is ``ANCHORED``, the strict reading, and for the same reason
    ``CaptureSource`` defaults to ``ASSERTED``: both defaults put the work on the
    writer to earn the unusual label. Defaulting the other way is not a choice at
    all — every ordinary capture would then claim to be a guess and need a model
    name that does not exist — and defaulting to ``INFERRED`` on read would label
    every document written before this axis existed as an inference, which is
    false: nothing in the capture path could infer a pairing then.

    What the default does *not* do is detect an inference that was never
    labelled. Nothing can, and pretending otherwise would be the over-claim this
    whole module exists to avoid. What it does is make the declared case
    readable, so a record that says ``inferred`` is never served as a captured
    one.
    """

    ANCHORED = "anchored"
    INFERRED = "inferred"


#: Metadata key naming the model that inferred a record's structure. Named once
#: because three surfaces have to agree on the spelling: the write-time
#: validator, the frontmatter writer, and the read path. A key that exists in
#: two places is a key one of them will spell differently.
STRUCTURE_INFERRED_BY = "structure_inferred_by_model"


def structure_anchor_gaps(*, structure: RecordStructure, inferred_by_model: object) -> list[str]:
    """Return what an inferred record is missing: the model that inferred it.

    One definition for both paths, exactly as :func:`capture_anchor_gaps` is.
    The write path refuses construction; the read path flags a stored document
    whose provenance does not hold, because the check that ran in the writing
    process says nothing about what a separate service returns later.

    Returns an empty list for ``ANCHORED``, which is required to name nothing:
    a record that claims no inference has no inferrer to name.
    """
    if structure is not RecordStructure.INFERRED:
        return []
    if not str(inferred_by_model or "").strip():
        return [f"metadata.{STRUCTURE_INFERRED_BY}"]
    return []


def structure_of(value: object) -> RecordStructure | None:
    """Resolve a structure value that arrived from outside this process.

    Absence resolves to ``ANCHORED`` — the documented default, and the truth for
    every document written before the axis existed. That is not a guess made by
    the reader: the capture path could not have inferred a pairing, so those
    documents state nothing that this resolves wrongly.

    An unrecognised value resolves to ``None`` rather than to a default. A
    renamed or corrupted value must never be read as a confirmed pairing, so a
    caller has to treat ``None`` as *not anchored* and say so.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return RecordStructure.ANCHORED
    if not isinstance(value, str):
        return None
    try:
        return RecordStructure(value.strip().casefold())
    except ValueError:
        return None


class Independence(StrEnum):
    """How far a captured record is from being a check on itself.

    Ordered weakest to strongest, so ``min_independence`` is a comparison rather
    than a set membership test. The middle level exists because "same bot account,
    different model" is not the same thing as "same bot account, same model", and a
    boolean could not tell those apart after the fact.

    - ``SELF_CERTIFIED`` — the same account answered using the same model that
      produced the question. The record checks nothing; it restates.
    - ``MODEL_SEPARATED`` — the same account, but a different model. A real second
      opinion from a second mind, not from a second party.
    - ``INDEPENDENT`` — a different account. A second party, which is the strongest
      thing a comment thread can offer.

    None of these are claims that the reasoning is *correct*. They describe who was
    in a position to disagree, which is the only part a record can honestly attest.
    """

    SELF_CERTIFIED = "self_certified"
    MODEL_SEPARATED = "model_separated"
    INDEPENDENT = "independent"

    @property
    def rank(self) -> int:
        """Strength ordering, so a threshold comparison is meaningful."""
        return _INDEPENDENCE_RANK[self]


_INDEPENDENCE_RANK: dict[Independence, int] = {
    Independence.SELF_CERTIFIED: 0,
    Independence.MODEL_SEPARATED: 1,
    Independence.INDEPENDENT: 2,
}

#: Model value shown where a document has to *say* something about an unstated
#: model — a rationale's attribution line, which is prose rather than a queryable
#: field, and so has no way to leave the sentence out.
#:
#: This constant used to stand for the opposite rule as well, and the two
#: together meant the store could not say which was which. Captured records do
#: **not** use it: there, an unstated model leaves its key absent, because the
#: frontmatter extras skip ``None`` and that is the one mechanism in this codebase
#: that expresses absence honestly. Writing the string ``"unknown"`` into a
#: frontmatter key instead produced a document in which a principal nobody named
#: appeared to have declared a model called ``"unknown"`` — indistinguishable, to
#: any reader filtering on the key, from a principal who had genuinely stated that
#: model. A filter for a real model can no longer match it, but a filter asserting
#: "this record states no model" cannot distinguish the two either, which is the
#: whole point of the axis.
UNKNOWN_MODEL = "unknown"


class RationaleSource(StrEnum):
    """Where a stated reason for a decision came from.

    This is a separate axis from :class:`Independence`, and the two answer different
    questions. ``Independence`` describes *who was positioned to disagree* about a
    change. ``RationaleSource`` describes *how the reasoning was obtained* — which
    is a fact about the pipeline, not a judgement about the record's worth.

    - ``DECLARED`` — stated by the agent that performed the work, from its own
      session. First-hand, and still not a check on correctness: the author of a
      change explaining the author's change is the textbook self-certified case,
      so a declared rationale never raises independence.
    - ``RECONSTRUCTED`` — inferred by a model reading the diff, with no access to
      the session that produced it. The answerer is sandboxed with no tools, so it
      structurally cannot know; it infers. Honest about that, the record is still
      useful — it is a second opinion rather than a recollection — but it must never
      be read as a statement of intent by the implementer.

    The two are kept apart rather than merged because a reader who cannot tell
    which they are looking at is in a worse position than one with no record at
    all: a reconstructed answer can otherwise stand in for a declared reason, and
    the fact that they disagree is itself the finding worth having.

    ``UNKNOWN`` exists so a record written before this axis existed reports its
    absence rather than defaulting into one of the two, which would be a guess
    about provenance made by the reader rather than the writer.
    """

    DECLARED = "declared"
    RECONSTRUCTED = "reconstructed"
    UNKNOWN = "unknown"

    @property
    def is_first_hand(self) -> bool:
        """True only for a rationale the implementer itself stated."""
        return self is RationaleSource.DECLARED


def compute_independence(
    *,
    asker_account: str | None,
    asker_model: str | None,
    answerer_account: str | None,
    answerer_model: str | None,
) -> tuple[Independence, str]:
    """Classify a captured answer, and say why in a short auditable phrase.

    A different posting account is ``INDEPENDENT`` regardless of model: two
    parties are stronger evidence than two models, and a human on the same model
    as a bot is still a second party.

    Where the account is shared, the comparison is on model. An unstated model on
    either side is not treated as a match, because assuming two unknowns are the
    same would manufacture a worse label than admitting the gap.
    """
    asker = (asker_account or "").strip().casefold()
    answerer = (answerer_account or "").strip().casefold()
    if asker and answerer and asker != answerer:
        return Independence.INDEPENDENT, "different posting accounts"

    asked_by = (asker_model or "").strip().casefold()
    answered_by = (answerer_model or "").strip().casefold()
    if not asked_by or not answered_by:
        return Independence.SELF_CERTIFIED, "same account; model not stated by both parties"
    if asked_by != answered_by:
        return Independence.MODEL_SEPARATED, "same account, different models"
    return Independence.SELF_CERTIFIED, "same account, same model"


class QuestionCategory(StrEnum):
    """Category of a knowledge-capture question."""

    DESIGN_DECISION = "design_decision"
    TRADE_OFF = "trade_off"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    EDGE_CASE = "edge_case"
    DEPENDENCY = "dependency"
    SYSTEM_EVENT = "system_event"


class Question(BaseModel):
    """A single generated question with optional context."""

    id: str = Field(..., description="Unique ID for matching")
    text: str = Field(..., description="The question text")
    category: QuestionCategory = Field(..., description="Category of the question")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Contextual information (e.g. code snippets, ticket details)",
    )
    error_message: str | None = Field(default=None)
    status: str = Field(default="pending")


class ReviewSession(BaseModel):
    """Tracks a capture session (e.g. a CLI invocation for a PR)."""

    session_id: str
    context_url: str = Field(..., description="URL of the context (e.g. PR URL)")
    context_id: str = Field(..., description="ID of the context (e.g. PR number)")
    scope: str = Field(..., description="Scope of the session (e.g. repository name)")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = Field(default=None)

    @property
    def pr_url(self) -> str:
        return self.context_url

    @property
    def pr_number(self) -> int:
        try:
            return int(self.context_id)
        except (ValueError, TypeError):
            return 0

    @property
    def repo(self) -> str:
        return self.scope


class KnowledgeEntry(BaseModel):
    """A stored Q&A pair, with the provenance needed to judge how much it is worth.

    Provenance is first-class, not an afterthought in ``metadata``:

    - ``capture_source`` is the trust axis. It defaults to ``ASSERTED`` so that a
      record is never evidence until a capture has actually claimed it.
    - ``structure`` is a second, independent axis, answering whether the record's
      question/answer pairing was established or inferred by a model. Neither
      axis can substitute for the other: a checkable text can still be a guess
      at how it was put together, and a genuine conversation is still only
      ``asserted`` if nobody vouched for it.
    - ``captured_at`` is when *the system* observed the event, which is a different
      fact from ``answered_at`` (when the human wrote the answer).
    - ``capture_delivery_id`` is the provider's own delivery identifier, so a
      record can be traced back to the exact signed request that produced it.

    A ``WEBHOOK`` entry is rejected unless it carries the identifiers that make it
    checkable, because an unverifiable "captured" record is the failure mode this
    field exists to prevent. An ``INFERRED`` entry is rejected unless it names the
    model that inferred it, for the same reason one axis further along: an
    unattributed guess is indistinguishable from a capture.
    """

    entry_id: str
    session_id: str | None = None
    question_text: str
    answer_text: str
    category: QuestionCategory
    context: dict[str, Any] | None = None
    author: str | None = None
    answered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    capture_source: CaptureSource = CaptureSource.ASSERTED
    captured_at: datetime | None = None
    capture_delivery_id: str | None = None
    #: Whether this record's own shape was established or inferred. Separate from
    #: ``capture_source`` because that answers a different question, and the two
    #: must not be allowed to stand in for each other. See :class:`RecordStructure`.
    structure: RecordStructure = RecordStructure.ANCHORED

    @model_validator(mode="after")
    def _captured_entries_must_be_verifiable(self) -> Self:
        """Refuse to store a captured record that cannot prove it was captured.

        Without this, any caller can label a hand-written entry ``webhook`` and
        every downstream reader — the CLI, the console, an agent over MCP — will
        treat it as verified review evidence. Each captured channel is held to the
        anchor that makes it independently checkable: a delivery id for ``webhook``,
        a source comment id for ``collect``.
        """
        if self.capture_source is CaptureSource.ASSERTED:
            return self

        missing = capture_anchor_gaps(
            capture_source=self.capture_source,
            repo=self.metadata.get("repo"),
            pr_number=self.metadata.get("pr_number"),
            captured_at=self.captured_at,
            delivery_id=self.capture_delivery_id,
            comment_id=self.metadata.get("github_comment_id"),
            check_id=self.metadata.get("check_id"),
            review_id=self.metadata.get(REVIEW_ID_KEY),
        )
        if missing:
            raise ValueError(
                f"capture_source={self.capture_source.value} requires verifiable "
                "provenance; missing: " + ", ".join(missing)
            )
        return self

    @model_validator(mode="after")
    def _an_inferred_record_must_name_its_inferrer(self) -> Self:
        """Refuse an inferred record that does not say which model inferred it.

        ``inferred`` is a claim about one machine's judgement, and the only thing
        that makes it useful rather than a vague hedge is *which* machine. A
        reader deciding whether to weight a guessed pairing is deciding about a
        specific model, and the record has to let them. An ``inferred`` label
        with no model is an unattributed guess — and unattributed is exactly how
        an inference becomes indistinguishable from a captured one, which is the
        failure this axis exists to prevent.

        The record is not rejected for being inferred. It is rejected for being
        silently inferred, which is a different defect and a fixable one: the
        writer knows the model and simply did not say.
        """
        missing = structure_anchor_gaps(
            structure=self.structure,
            inferred_by_model=self.metadata.get(STRUCTURE_INFERRED_BY),
        )
        if missing:
            raise ValueError(
                f"structure={self.structure.value} requires the model that inferred it; "
                "missing: " + ", ".join(missing)
            )
        return self

    @property
    def is_captured(self) -> bool:
        """True for entries a real provider source produced.

        False for anything merely asserted, regardless of how much detail it
        carries. Detail is not provenance.

        ``BACKFILLED`` counts as captured but not as *witnessed*: the text came
        from a real authenticated read of the forge, months after the fact. It is
        checkable, and what it can be checked against is the forge as it stands
        today rather than as it stood when the event happened — which is why the
        source value rather than this property is what a reader should be
        filtering on.
        """
        return self.capture_source is not CaptureSource.ASSERTED

    @property
    def pr_url(self) -> str:
        return self.metadata.get("pr_url", "")

    @property
    def pr_number(self) -> int:
        return self.metadata.get("pr_number", 0)

    @property
    def repo(self) -> str:
        return self.metadata.get("repo", "")

    @property
    def jira_ticket_key(self) -> str | None:
        return self.metadata.get("jira_ticket_key")

    @property
    def question_category(self) -> QuestionCategory:
        return self.category


class QuestionRecord(BaseModel):
    """A decision request, projected into the knowledge store as a record.

    Deliberately its own model rather than a ``KnowledgeEntry`` with a category,
    for the same reasons a rationale is: a question is not a conclusion, and a
    record that asks "why did we choose this?" must never be readable as a record
    of the answer.

    **This model is the allowlist, and that is the point of it being a model.**
    The fields below are the *only* things a projected question can carry, so a
    column added to the registry later is not published by default — it has to be
    added here deliberately, where the omission is visible.

    Two columns are deliberately missing and must stay missing:

    - ``claim_token``. It is a ``secrets.token_urlsafe(32)`` capability and is the
      ``WHERE`` guard on releasing a claim, so anyone holding it can release a
      claim somebody else holds. Tanseki is readable over MCP by agents; a claim
      token in a document is a capability handed to every reader of the store.
    - ``last_error``. Internal exception detail with no analytical value, and the
      one field on a question row most likely to carry a path or a fragment of
      untrusted text.

    A deny-list would have been the wrong shape: it makes the safe behaviour
    depend on remembering to update a blocklist, and the failure is silent — a
    document is merely missing a key, and nothing reports it.
    """

    question_id: str
    repo: str
    pr_number: int | None = None
    pr_url: str | None = None
    question_text: str
    #: One of the closed registry statuses. Never ``pending`` or ``claimed``:
    #: outstanding work is operational and belongs in the registry, and a queue in
    #: a knowledge store is a queue pretending to be knowledge.
    status: str
    category: str | None = None
    jira_ticket_key: str | None = None
    session_id: str | None = None
    question_author: str | None = None
    assignee: str | None = None
    attempts: int = 0
    answer_comment_id: int | None = None
    created_at: datetime | None = None
    answered_at: datetime | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def _only_terminal_states_are_recorded(self) -> Self:
        """Refuse to project a question that is still outstanding.

        Projecting ``pending`` would put a live work queue into the knowledge
        store, where it reads as a set of open decisions a reader should act on
        rather than as a set of requests this process is still waiting on. It
        would also be a moving target: the same document id would be rewritten
        every time a worker claimed or released the question, which is a write
        rate the store's durability path was not built for.
        """
        if self.status not in _TERMINAL_QUESTION_STATUSES:
            raise ValueError(
                f"only a terminal question status is projected, got {self.status!r}; "
                f"terminal is one of {', '.join(sorted(_TERMINAL_QUESTION_STATUSES))}"
            )
        return self

    # There is deliberately no validator for the evidence axis. A question has
    # no ``capture_source`` and no ``independence`` field, so there is nothing to
    # validate and nothing that could be set wrongly — the absence *is* the
    # mechanism, and a guard asserting an absence the type system already
    # enforces would be a second place for the claim to live.


#: Terminal statuses, named here rather than imported from the registry: the
#: registry is the operational store and importing it would put a SQLite module
#: in the dependency graph of a pure model. The projection asserts the two agree.
_TERMINAL_QUESTION_STATUSES = frozenset({"answered", "failed", "superseded"})


class CensusRecord(BaseModel):
    """One observed event that produced no knowledge.

    Deliberately its own model rather than a ``KnowledgeEntry`` with a category, for
    the same reasons a question and a rationale are both their own: nothing was said
    here, so there is no author, no answer, and nothing for a reader to weigh. A
    census record says exactly one thing — *this delivery was processed and nothing
    was captured from it* — and that is a fact about the corpus rather than about
    anyone's thinking.

    The distinction it protects is the whole point of the record. "We looked at this
    change and found nothing worth keeping" and "we never saw this change" are the
    same observation from outside a system that only writes down what it saw, and
    computing a rate over a self-selected numerator is the failure this record exists
    to make impossible. **It makes the bias visible; it does not repair it.** The
    bias from "kojutsu only knows about changes that attracted attention" remains
    permanent for anything it did not witness.

    **The record is about an event, not about a change.** A change opened silently
    and reviewed the next day with a capture legitimately has one of each: the
    opening delivery was processed and yielded nothing, the review delivery was
    processed and yielded a review. A count of census records is therefore not a
    count of uncaptured changes, and the document id keys on ``(repo, pr, action)``
    rather than on the delivery so that two observations of the same fact upsert one
    document. A reader who wants "changes with no knowledge" has to ask for changes
    with no knowledge *record*, which is a different and more expensive question.

    **There is deliberately no reason field, and no independence level.** Kojutsu
    cannot know why nothing was captured — nobody commented, the reviewer was not
    authorised, the marker was malformed, the change was never reviewed — and each of
    those is a hypothesis about someone else's intent. An unauthorised reviewer is the
    tempting case, because the system does know it declined the capture; that is a
    fact about the system's configuration, not about the author's intent, and
    recording it in the corpus would dress policy up as fact. The absence is the
    mechanism: the reason is absent rather than defaulting to a value that implies a
    cause. A reader who wants the reason should read the forge.
    """

    entry_id: str
    #: The natural key is ``(repo, pr_number, action)`` — see the class docstring for
    #: why it is not the delivery id, which identifies the observation rather than
    #: the thing observed.
    repo: str
    pr_number: int | None = None
    pr_url: str | None = None
    #: The delivery's own action, verbatim from the forge. Kept rather than
    #: interpreted: this record does not classify what happened, it records that
    #: something was delivered and processed.
    action: str
    #: Who opened the change. A login, not a person and not a judgement — the same
    #: standing as ``change_author_account`` on a captured record.
    change_author_account: str | None = None
    head_sha: str | None = None
    #: When *the system* observed the event, which for this record is the only time
    #: there is. It is not when the change was opened.
    observed_at: datetime
    #: The forge's own delivery id. This is what makes the observation evidence
    #: about a delivery rather than a bare claim that one happened.
    delivery_id: str | None = None
    #: Permanently ``WEBHOOK``. A census record is produced by a genuine signed
    #: delivery that happened to yield no knowledge; how it arrived and what it
    #: yielded are different questions, and ``webhook`` answers the first accurately.
    #: It is also why the anchor rules need no new rule: the delivery id is the anchor.
    capture_source: CaptureSource = CaptureSource.WEBHOOK
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _an_observation_must_be_checkable(self) -> Self:
        """Refuse a census record that cannot name the delivery it came from.

        The claim "this was observed and nothing captured" is checkable only by
        re-fetching the delivery it names. Without the id it is an assertion, and an
        assertion about what kojutsu did *not* store is the easiest kind of record
        to invent — a single extra document would let a reader believe a change was
        examined when nothing happened. Construction is refused rather than the field
        cleared, because a census record with no delivery behind it is not a weaker
        census record, it is a false one.
        """
        if not str(self.delivery_id or "").strip():
            raise ValueError(
                "a census record claims an observation happened and must name the "
                "delivery it came from; delivery_id is required"
            )
        return self

    # There is deliberately no validator for independence or for a reason. A census
    # record carries neither field, so there is nothing that could be set wrongly —
    # the absence *is* the mechanism, and a guard asserting an absence the type
    # system already enforces would be a second place for the claim to live.

    @property
    def is_captured(self) -> bool:
        """False, always. It reached the store by a real delivery and captured nothing.

        Present so a caller can ask without special-casing, and so the answer is
        ``False`` for the obvious reason rather than by omission from a set.
        """
        return False


class RationaleChannel(StrEnum):
    """How a declaration reached the store.

    Deliberately separate from :class:`CaptureSource`. ``CaptureSource`` is the
    *trust* axis and a rationale is permanently ``ASSERTED`` on it, whichever route
    it took. This axis answers a different question — where the text came from —
    and the two must not be allowed to trade places.

    - ``FORGE_COMMENT`` — posted as a marked comment on the change, then collected.
      A human reading the pull request can see the reason, which is worth having
      and is the whole reason to prefer this route when a change exists.
    - ``CAPTURE_SERVER`` — declared straight through the capture tool, with no
      forge involved. This exists because an agent does not necessarily have a
      place to post: working from a local repository, there is nowhere to publish,
      and a rationale it could not record is a rationale nobody will read.

    Neither route makes the declaration evidence. That is the point of keeping them
    apart from the trust axis: a route that reaches the store more directly is not
    a route that verifies more.
    """

    FORGE_COMMENT = "forge_comment"
    CAPTURE_SERVER = "capture_server"


class RationaleEntry(BaseModel):
    """One stated reason for a decision, as a record in its own right.

    Deliberately a separate model rather than a field on :class:`KnowledgeEntry`.
    The two have different lifetimes, different trust, and different questions, and
    merging them would let a projection read as a conclusion:

    - A ``KnowledgeEntry`` answers a question that was asked. Its ``category`` is
      one of six retrospective values.
    - A rationale states a reason *before anyone asks*, and every existing category
      is retrospective. Adding one here would let a forward-looking claim carry the
      label of a decision already taken.

    It is also separate because of trust. ``capture_source`` on an entry can reach
    ``WEBHOOK``; a rationale has no provider delivery behind it and is
    permanently ``ASSERTED``. Folding one into the other would give it a field that
    says it can be evidence, and a reader who trusted that field would be wrong.

    There is deliberately no ``structure`` field here, though an entry has one. A
    rationale is a *statement*, with no question to pair an answer to, so there is
    no pairing for a model to have inferred. Whether the reasoning was declared or
    reconstructed is already a field on this model -- ``source`` -- and a second
    axis that could only ever hold one value would invite a reader to believe a
    rationale has an inferable structure to declare in the first place.

    Revisions are appended, never overwritten. A declaration made early goes stale
    as the work proceeds, so ``revision`` advances and ``revises`` points at what it
    supersedes. Overwriting would let a later, worse rationale destroy an earlier,
    better one with nothing left to show that anything was lost.
    """

    entry_id: str
    #: The semantic anchor the id is derived from, kept so a record can be
    #: re-derived and compared without trusting the stored id.
    repo: str
    branch: str = ""
    pr_number: int | None = None
    declared_by: str
    declared_model: str | None = None
    #: What the agent says it decided, and why. Free prose, bounded at the parser.
    rationale_text: str
    source: RationaleSource = RationaleSource.DECLARED
    #: Where the declaration came from. Recorded on the model rather than in
    #: ``metadata`` because it answers "how did this get in here?", and a bag key
    #: is where that fact goes to be dropped.
    channel: RationaleChannel = RationaleChannel.FORGE_COMMENT
    #: 1-based. Monotonic per (repo, branch, declared_by).
    revision: int = 1
    #: The entry id this revision supersedes, or None for the first.
    revises: str | None = None
    declared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Permanently ``ASSERTED``. There is no provider delivery behind a stated
    #: reason, and no anchor that would make one checkable, so a rationale is a
    #: claim about intent and never evidence about the code.
    capture_source: CaptureSource = CaptureSource.ASSERTED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _a_rationale_is_never_evidence(self) -> Self:
        """Refuse a rationale that claims to have been captured.

        The platform verifies who posted a comment, never which model drafted it or
        what it decided. A rationale that could claim ``webhook`` provenance would
        let an agent's own account of its work be served to a later reader as
        review evidence captured from a trusted thread -- an agent asserting its
        own reliability. Construction is refused rather than the field corrected,
        because a caller reaching for this is making a claim the system cannot back.
        """
        if self.capture_source is not CaptureSource.ASSERTED:
            raise ValueError(
                "a stated reason has no provider delivery behind it, so it cannot be "
                f"capture_source={self.capture_source.value}; rationale is always asserted"
            )
        return self

    @model_validator(mode="after")
    def _a_revision_must_name_what_it_supersedes(self) -> Self:
        """Refuse a revision that points nowhere, or a first revision that does.

        A revision whose target is missing has silently become a new record that
        claims to be an update, and a reader would read it as one. This mirrors
        ``select_questions`` in ``core/answerer.py``, which refuses to swap a named
        question for another: silently attaching a declaration to the nearest
        available record is how a system ends up stating something nobody said.
        """
        if self.revision < 1:
            raise ValueError(f"rationale revision must be at least 1, got {self.revision!r}")
        if self.revision == 1 and self.revises is not None:
            raise ValueError("the first revision of a rationale cannot supersede anything")
        if self.revision > 1 and not self.revises:
            raise ValueError(f"revision {self.revision} must name the entry id it supersedes")
        return self

    @property
    def is_first_hand(self) -> bool:
        """True when the principal that declared this also performed the work."""
        return self.source.is_first_hand


class ClarificationEntry(BaseModel):
    """One human statement in a review thread that answered no question.

    This is the content a PR owner volunteers — "this is deliberate, the narrow
    window is the accepted cost for v0.1" — and a reviewer concedes — "fair
    criticism, and it should be recorded as a known weakness of this test".
    Nobody asked for either. Often it is the most valuable sentence in the thread.

    :class:`KnowledgeEntry` cannot hold it. That model requires ``question_text``,
    and all six :class:`QuestionCategory` values are retrospective, so an
    unprompted statement forced into it is stored as a question that was never
    asked under a category that never applied — the exact mirror of the bug that
    made a rationale read as a conclusion. So this model has **no question text
    field at all** and no category. An empty string would be worse than an absent
    field, because a reader cannot then tell "no question was asked" (a fact about
    the record) from "the question text was lost in transit" (a defect in us).

    It is a separate model for the second reason, which is trust, and the trust
    runs the *opposite* way to a rationale's:

    - A :class:`RationaleEntry` is permanently ``ASSERTED``. Nothing was signed; a
      stated reason has no provider delivery behind it.
    - A clarification is a quotation from a real comment. The posting account, the
      author association and the comment id are things the forge actually
      verified, and the comment can be re-fetched by anyone. It is real evidence
      that simply has no question attached to it.

    So it is ``COLLECT`` or ``WEBHOOK`` and is held to :func:`capture_anchor_gaps`
    like any other captured record. ``ASSERTED`` is *refused* rather than
    defaulted: a record that quotes nobody is a typed claim, which is precisely
    what :class:`RationaleEntry` exists to be, and letting this one be written as
    an assertion would give the same text two different levels of trust.

    Revisions are deliberately not modelled. A comment is edited in place on the
    forge, so the stored text is the text as it stood at capture time and the live
    comment remains the record of truth. Identity is the comment, never the text —
    see :func:`kojutsu.core.question_registry.stable_clarification_entry_id`.

    What this record deliberately does **not** carry is an independence level.
    ``Independence`` answers "who was in a position to disagree about a change",
    and a clarification has no asker to compare against — nobody asked. Stating a
    level would be a guess made by the reader, so this record states none, which
    places it below every ``min_independence`` threshold. See
    ``docs/design-review/clarification.md``.
    """

    # D50SGP0Y deliberately did not define ``RecordStructure`` locally: it belongs to
    # sibling ticket 4797HFN6's axis, and defining it twice would have guaranteed a
    # merge conflict. The ``structure`` field below is attached from that axis at
    # merge time.
    #
    #: ``extra`` is forbidden, and that is the enforcement half of "no question
    #: text field at all". Pydantic ignores unknown keys by default, so a caller
    #: porting a ``KnowledgeEntry`` across would have its ``question_text`` and
    #: ``category`` silently dropped and the record stored as though the question
    #: had never been offered. A clarification is a quotation, and quietly
    #: discarding half of what a caller said about it is the failure this model
    #: exists to prevent — so it is refused loudly instead.
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    #: The semantic anchor the id is derived from, kept so a record can be
    #: re-derived and compared without trusting the stored id.
    repo: str
    pr_number: int
    #: The human text, verbatim. Never paraphrased and never completed: a
    #: clarification is a quotation, and a summarised quotation is no longer
    #: checkable against the comment it claims to come from.
    statement: str
    #: The forge account that posted the comment. The account, not the agent:
    #: a declared agent is recorded separately because nothing in the platform
    #: verifies it.
    author: str
    #: What the repository is to this account, as the forge reported it. Recorded
    #: on the model rather than in ``metadata`` because it is part of what makes
    #: the record checkable, not a detail about the person.
    author_association: str
    github_comment_id: int
    #: When the human wrote the statement, which is a different fact from
    #: ``captured_at`` (when this system saw it).
    declared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Never ``ASSERTED``: see the class docstring. Required rather than defaulted
    #: so a caller has to state how the record was captured.
    capture_source: CaptureSource
    #: Whether the *structure* of this record was established or inferred.
    #: Attached at merge time from sibling ticket 4797HFN6's axis, which defines
    #: ``RecordStructure`` in this module; D50SGP0Y deliberately did not define it
    #: locally. Absent from the stored frontmatter when anchored, exactly as on
    #: ``KnowledgeEntry``, so an anchored clarification is byte-identical to one
    #: written before the axis existed.
    structure: RecordStructure = RecordStructure.ANCHORED
    captured_at: datetime | None = None
    capture_delivery_id: str | None = None
    #: A machine principal's *self-declared* identity, read from the comment's
    #: agent marker, on exactly the terms an answer is: present when the comment
    #: carries one, and never inferred from the absence of one.
    authored_by_agent: str | None = None
    #: The model that agent named, if it named one. Also a self-declaration —
    #: GitHub proves who posted a comment, never which model drafted it.
    authored_by_model: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("author_association")
    @classmethod
    def _association_is_normalised_or_refused(cls, value: str) -> str:
        """Upper-case the association, and refuse a blank one.

        A record that cannot say who the speaker was to the repository is missing
        the part that makes a clarification weighable — "the owner said this" and
        "somebody with commit access said this" are different statements. The gate
        on *who may* state one is the collector's, unchanged; this only refuses to
        store a blank.
        """
        normalised = value.strip().upper()
        if not normalised:
            raise ValueError("a clarification must record the author's association to the repo")
        return normalised

    @model_validator(mode="after")
    def _a_clarification_is_a_quotation_of_a_real_comment(self) -> Self:
        """Refuse a record that claims to be a quotation and cannot prove it.

        The mirror image of ``RationaleEntry._a_rationale_is_never_evidence``: that
        one refuses a rationale that claims a capture it cannot have, this one
        refuses a clarification that denies having one. Both are the same rule —
        a record's trust axis must be the one its content can support.
        """
        if self.capture_source is CaptureSource.ASSERTED:
            raise ValueError(
                "a clarification quotes a real comment, so it is captured evidence; "
                f"capture_source={self.capture_source.value} describes a typed claim, "
                "which is what RationaleEntry is for"
            )
        if not self.statement.strip():
            raise ValueError(
                "a clarification with no statement is a record of nothing; an empty "
                "body is not a quotation of an empty comment"
            )
        missing = capture_anchor_gaps(
            capture_source=self.capture_source,
            repo=self.repo,
            pr_number=self.pr_number,
            captured_at=self.captured_at,
            delivery_id=self.capture_delivery_id,
            comment_id=self.github_comment_id,
        )
        if missing:
            raise ValueError(
                f"capture_source={self.capture_source.value} requires verifiable "
                "provenance; missing: " + ", ".join(missing)
            )
        return self

    @property
    def is_agent_authored(self) -> bool:
        """True when the comment declared a machine principal, on its own authority."""
        return bool(self.authored_by_agent)


class EvaluationTarget(StrEnum):
    """What a measurement is a measurement *of*.

    Not a question category, and deliberately not one. Every
    :class:`QuestionCategory` value is retrospective: it names something decided
    about a change, and all of them read as findings about the code. A harness
    reporting on its own component produces a finding about *a model*, and filing
    it under ``design_decision`` or ``edge_case`` would make the report look like
    a claim about the repository the harness happened to be pointed at. A model
    performing well on one thread is not a fact about that repository.

    So the category lives here instead, and it is a single value on purpose. The
    moment this enum grows a second member, the first question is what a document
    carrying the old one now means -- and a measurement whose subject is
    ambiguous is the one kind of number here that is worse than none, because it
    gets compared against a number about something else.
    """

    MODEL = "model"


class EvaluationEntry(BaseModel):
    """One measurement Kojutsu took of one of its own components.

    The third record kind, and the one most at risk of being filed as something it
    is not. An evaluation answers a question about *this system*: how often a
    classifier recovered a pairing a person had established, how many comments it
    dropped, how much the answer moved between runs. It is not a fact about the
    pull request the harness was pointed at, and the difference is the whole
    reason this is a separate model:

    - A :class:`KnowledgeEntry` is a question and answer about a change. A reader
      holding one has evidence about somebody's work, anchored in a comment they
      can re-fetch.
    - An :class:`EvaluationEntry` is a number produced by running a model. The
      comments it scored are re-fetchable, but the *pairings* it scored them for
      were inferred, and a reader who saw the record in the answer namespace would
      have no way to tell which half they were holding.

    **The scope is part of the record, not a caveat in a docstring.** One thread,
    one repository, one reviewer, one model, one configuration. A precision figure
    from a single thread is a data point, and a record that states the numbers
    without the scope that makes them mean anything is the over-claim this model
    exists to refuse. So ``scope`` is required and there is no default: a
    measurement that cannot say what it measured is not a degraded measurement, it
    is an unreadable one.

    Permanently ``ASSERTED``, like a :class:`RationaleEntry` and for a stronger
    reason. A rationale at least has a human being who chose to state it. This was
    produced by a program grading a model, and the grading is done by the same
    kind of component whose output it is judging -- nothing outside the process
    signed it, and the component that produced the numbers is the one whose
    accuracy is in question. A non-asserted ``capture_source`` is refused rather
    than corrected, because a caller reaching for one is asking the store to
    vouch for a self-assessment.

    There is deliberately no ``structure`` field. The report's own question and
    answer are written by the harness and are as established as any record's; the
    model-inferred material inside it is named by ``subject`` and
    ``structure_inferred_by_model`` in ``metadata``, which is a statement about
    what the report is *about* rather than about how the report is put together.
    Setting ``structure: inferred`` here would misdescribe the document to buy an
    axis it already has an answer for.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    #: The semantic anchor the id is derived from, kept so a record can be
    #: re-derived and compared without trusting the stored id.
    repo: str
    pr_number: int
    #: What was measured. A model, never a repository and never a person.
    target: EvaluationTarget = EvaluationTarget.MODEL
    #: The model the numbers are about, named exactly as the provider adapter was
    #: asked for it. This is a self-assertion by the harness in the same sense
    #: ``declared_by_model`` is for a rationale: recorded because "a model guessed"
    #: is not weighable and a reader deciding how much to trust a number is
    #: deciding about a specific one.
    subject: str
    #: The question asked of the subject, in the harness's own words. Held apart
    #: from :attr:`subject` so a second question about the same subject is a
    #: second record rather than an overwrite of the first.
    measurement: str
    #: The numbers, verbatim, as the harness produced them. Never rounded on the
    #: way in and never summarised: a reader who has to trust the prose above a
    #: rounded figure is being asked to re-derive the measurement to check it.
    result_text: str
    #: What this measurement can and cannot stand for, in the harness's own words.
    #: Required, for the reason in the class docstring, and bounded by nothing here
    #: on purpose: the point is that a reader can see the limits without having
    #: found the design note that states them.
    scope: str
    measured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Permanently ``ASSERTED``. Refused at construction otherwise.
    capture_source: CaptureSource = CaptureSource.ASSERTED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _an_evaluation_is_never_evidence(self) -> Self:
        """Refuse a measurement that claims to have been captured.

        The same rule as ``RationaleEntry._a_rationale_is_never_evidence``, and
        the same reason. Nothing signed this: the numbers were produced in-process
        by the same class of component whose output they grade. A harness able to
        label its own report ``webhook`` would be a harness able to assert that a
        measurement about a model is review evidence from a trusted thread, which
        is the manufactured-consensus failure with the label removed.
        """
        if self.capture_source is not CaptureSource.ASSERTED:
            raise ValueError(
                "a measurement of this system has no provider delivery behind it, so "
                f"it cannot be capture_source={self.capture_source.value}; an "
                "evaluation is always asserted"
            )
        if self.target is not EvaluationTarget.MODEL:
            raise ValueError(
                f"an evaluation must name what it measured; target={self.target.value} "
                "describes something this record cannot be evidence about"
            )
        if not self.scope.strip():
            raise ValueError(
                "an evaluation must state what its numbers can and cannot stand for; "
                "a figure whose scope is blank is a number nobody can weigh"
            )
        return self
