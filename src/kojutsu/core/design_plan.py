"""The shape of a reconciled design plan, and what makes one malformed.

A plan is the one artefact in the design phase that turns prose into commitments:
the goals it serves, the decisions it took with the alternatives it discarded, and
the ticket drafts that a later phase creates mechanically. It is therefore the only
place in this pipeline where a mistake stops being a sentence and starts being a
ticket. The ticket this implements is explicit about the stakes -- *"an unvalidated
plan that generates malformed tickets is worse than no plan"* -- and the shape of
that sentence is the shape of this module. A ticket generated from a bad plan is
not a broken ticket: it is a **well-formed** ticket that is wrong in a way the
person reading it cannot detect, because a ticket carries no trace of the plan that
produced it. Everything below is a refusal of a specific class of that.

**The bounds are not input tidiness.** Every bound here answers one question: what
does this cap stop from reaching the store? ``MAX_PLAN_TICKETS`` is a reviewability
limit, because the human gate approves the plan once and then every draft is created
without further approval -- so a plan too large for one reader to hold in their head
has converted a single review into no review at all. An unbounded label list is a
way to carry a paragraph of rationale into a field every reader scans as a tag,
where it will not be read. Neither is a limit on what a model may say; both are
limits on what a human is asked to sign off on.

**Field names are the ticket store's field names.** ``title``, ``description``,
``acceptance_criteria``, ``test_command``, ``reference_files``, ``labels``,
``priority`` and ``depends_on`` are spelled here exactly as the ticket store spells
them. A mechanical transformation that renames a field as it crosses the boundary is
a place where the plan and the tickets it produced can disagree with nothing to
compare them -- and the whole claim of the ticket-creation phase is that the
transformation is mechanical, which it cannot be if a name is quietly remapped on the
way through. Renaming, if it is ever wanted, is a rename in the store.

**The plan carries no provenance.** No topic, no proposer principals, no model name,
no timestamp. That is not an omission, it is what keeps the document a pure input:
ticket creation must be **idempotent**, and a re-run has to be handed the same plan
and reach the same tickets. Provenance belongs to the reconciliation record that
wraps this document, where "which model reconciled which proposals, when" is a fact
about an event rather than a claim embedded in the artefact that event produced --
and an artefact that asserted its own origin would have a different identity, and
therefore different derived tickets, depending on who happened to write it down.

**One error type, and it names everything.** A malformed plan raises
:class:`DesignPlanError` carrying every problem found, not the first one. Two
reasons, and the second is the one that matters. Fixing a plan one error per cycle
means one model call per cycle, and each cycle is a fresh opportunity for the model
to *lose* the problems it was not shown -- so a four-fault plan becomes four rounds,
and the plan that finally validates is a plan that was edited four times against
partial information. The first reason is the plainer one: a reader who is told one
problem at a time cannot tell a plan that is nearly right from one that is not, and
those deserve different amounts of their remaining attention.

**Cross-field validation is a separate function, not a ``@model_validator``.** The
reason is in :func:`validate_design_plan`, and it follows ``_validate_ask_plan`` in
``kojutsu.cli``, which is the same split for the same reasons.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from kojutsu.core.text_hygiene import CONTROL_RE

#: The most goals a plan may state.
#:
#: Goals are read once, by a human, at the approval gate. Past this length the list
#: is no longer something anyone reads as a list, which means the field that exists to
#: make the plan's purpose checkable has stopped being checkable -- and the plan goes
#: on to produce tickets whose acceptance criteria nobody traced back to a goal.
MAX_PLAN_GOALS = 50

#: The most decisions a plan may record.
#:
#: A decision with no rationale and no rejected alternative is an assertion wearing a
#: decision's clothes, so the cap is on the number of *reasoned* statements the gate
#: is asked to accept rather than on how much text a model can emit.
MAX_PLAN_DECISIONS = 50

#: The most ticket drafts a plan may contain.
#:
#: This is the reviewability bound, and it is the one number in this module that a
#: future phase must not raise casually. Approval happens **once, on the plan**, and
#: every draft is then created without anyone looking at it again -- so the gate's
#: strength is exactly the number of drafts a person can hold in their head while
#: approving. A plan above this size has not been made more thoroughly reviewed; it
#: has been made unreviewed, while still carrying the appearance of a gate.
MAX_PLAN_TICKETS = 100

#: The most rejected alternatives one decision may list.
#:
#: Bounded because this is the field most likely to be filled by restating the chosen
#: option a dozen ways, which lengthens the gate's reading without adding a decision.
MAX_ALTERNATIVES_PER_DECISION = 20

#: The most acceptance criteria one draft may declare.
#:
#: A criterion is one verifiable condition -- the ticket store's own guidance is a
#: checklist of conditions that define done. A list too long to check item by item
#: has stopped being a definition of done, and an unverified definition of done is
#: how a malformed ticket passes review.
MAX_ACCEPTANCE_CRITERIA = 20

#: The most reference files one draft may point at.
#:
#: References are the files an executing agent reads before starting. Past this, the
#: list stops being a briefing and becomes a directory listing of the repository,
#: which the agent can already obtain for itself.
MAX_REFERENCE_FILES = 50

#: The most labels one draft may carry.
#:
#: Labels are a filtering surface. Twenty is past the point where a tag list can be
#: scanned, so this bounds the case where labels are being used as prose.
MAX_LABELS = 20

#: Longest single label.
#:
#: Short enough that one label cannot carry a sentence, because a label long enough
#: to read is a label nobody filters by -- and the labels that do get filtered by are
#: the ones this phase needs to keep usable as a query surface over plans.
MAX_LABEL_LENGTH = 64

#: Longest ticket title.
#:
#: A title past this is a description that has been misfiled, which means the draft's
#: description -- the field a reader reads to find out whether the ticket is the right
#: one -- is empty.
MAX_TITLE_LENGTH = 200

#: Longest ticket description.
#:
#: Long enough for the context a ticket needs to be worked from, short enough that a
#: reader deciding whether to work it is still reading.
MAX_DESCRIPTION_LENGTH = 4_000

#: Longest one acceptance criterion.
MAX_ACCEPTANCE_CRITERION_LENGTH = 1_000

#: Longest one test command.
#:
#: A command, not a script. Anything longer is a script that the plan is asking an
#: agent to run without being able to see what it does, which is the one thing a
#: plan-declared verification command must not be.
MAX_TEST_COMMAND_LENGTH = 1_000

#: Longest one reference file path.
#:
#: Paths are written into tickets and read by people scanning them; the bound is
#: generous enough for a deep path and short enough to catch a pasted directory
#: listing standing in for one.
MAX_REFERENCE_FILE_LENGTH = 500

#: Longest one goal.
MAX_GOAL_LENGTH = 1_000

#: Longest one decision summary -- what was decided, as a title.
MAX_DECISION_SUMMARY_LENGTH = 500

#: Longest one rationale, and longest one rejection reason.
#:
#: The rationale is the artefact that survives the reconciliation: the proposals it
#: reconciles become records nobody will read again, while this sentence is what a
#: later reader has instead. So the bound is generous, and :class:`DesignDecision`
#: makes it required rather than optional -- an unbounded-hope version of this field
#: is an empty string.
MAX_RATIONALE_LENGTH = 2_000

#: Longest one rejected alternative's description.
MAX_ALTERNATIVE_LENGTH = 500

#: Longest one draft id, and one decision-independent key length.
#:
#: Ids are keys local to a single plan -- dependency edges and nothing else point at
#: them -- so they do not need to be globally unique and do not need to be short.
#: They are bounded anyway because an id is quoted in error messages and in every
#: provenance line that references it, and a 4 KB identifier makes all of those
#: unreadable.
MAX_DRAFT_ID_LENGTH = 64

#: Longest one priority.
#:
#: See :class:`TicketDraft` for why priority is a bounded string rather than an
#: enumeration. The bound is deliberately roomy: it is here to stop a priority field
#: carrying prose, not to encode a vocabulary this module does not own.
MAX_PRIORITY_LENGTH = 32

#: A single acceptance criterion.
AcceptanceCriterion = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_ACCEPTANCE_CRITERION_LENGTH)
]

#: A single label.
Label = Annotated[str, StringConstraints(min_length=1, max_length=MAX_LABEL_LENGTH)]

#: A single reference file path.
ReferenceFile = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_REFERENCE_FILE_LENGTH)
]

#: A single goal.
Goal = Annotated[str, StringConstraints(min_length=1, max_length=MAX_GOAL_LENGTH)]

#: A draft id, used both where a draft is declared and where an edge names one.
#:
#: One type for both, rather than a bound repeated on each field, because the bound is
#: only the same bound by coincidence otherwise. An edge is a reference to a draft, so
#: "could this ever be an id" is a question with one answer, and a plan that could
#: express an edge naming something too long to *be* an id would be expressing a
#: reference that the declaration side says cannot exist.
DraftId = Annotated[str, StringConstraints(min_length=1, max_length=MAX_DRAFT_ID_LENGTH)]


class DesignPlanError(ValueError):
    """The plan cannot be trusted, and every reason it cannot is named here at once.

    A ``ValueError`` because that is what the rest of this codebase raises for input
    it will not act on -- ``AnswerSelectionError``, ``ThreadClassificationError``,
    ``AllowlistError`` -- and a caller that already catches one of those to keep an
    untrusted value from becoming state should not need a second clause for this one.

    ``problems`` is a tuple of sentences rather than a single message, in the shape
    :attr:`kojutsu.core.backfill.BackfillReport.errors` already uses for "several
    independent things went wrong and the reader needs all of them". A rendered
    message is still produced, one bullet per problem, because the overwhelmingly
    common caller is a person reading a log line.

    The problems are collected, not raised one at a time. See the module docstring:
    reporting the first failure turns one four-fault plan into four rounds of
    editing against partial information.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        count = len(self.problems)
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"Design plan is malformed: {count} problem{'' if count == 1 else 's'}"
            + (f"\n{listed}" if listed else "")
        )


class RejectedAlternative(BaseModel):
    """One option the reconciler did not take, and the reason it did not take it.

    Two fields rather than one string, and that is the load-bearing part of this
    class. A rejected alternative recorded as a bare name -- ``"use redis"`` -- is
    worth much less than it appears: a later reader cannot tell whether it was
    rejected as too slow, as too operationally expensive, or as fine but more than
    this needed, and those three readings imply different decisions next time. It is
    also the field most likely to be dropped entirely, since the rationale for the
    chosen option feels like it already carries the argument. Requiring
    ``why_rejected`` separately is what makes the cheaper failure -- a plan that
    argues only for what it chose -- structurally unavailable.

    What this deliberately does **not** require is that the rejection be real. A
    decision with an empty :attr:`DesignDecision.alternatives_rejected` list is
    accepted, because there are decisions with no live alternative -- a name fixed by
    a wire format, a bound fixed by an existing schema -- and demanding one anyway
    would convert a schema into a fabrication generator. An invented strawman
    recorded as "the alternative we rejected" is a **false** record of what was
    considered, which is strictly worse than an honest empty list: the one field whose
    entire purpose is to be believed.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    #: What the option was, stated as a reader would recognise it.
    alternative: str = Field(min_length=1, max_length=MAX_ALTERNATIVE_LENGTH)
    #: Why it was not taken, in terms of what it costs rather than how it sounds.
    why_rejected: str = Field(min_length=1, max_length=MAX_RATIONALE_LENGTH)


class DesignDecision(BaseModel):
    """One choice the reconciler made, with the case for it and the case against it.

    A decision and its rationale are one record rather than two fields on a plan,
    because the failure this guards against is a rationale that has drifted away
    from the decision it explains: separately editable, they diverge, and a plan
    whose reasoning no longer matches its decisions is worse than a plan with no
    reasoning, because it looks argued.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    #: What was decided, as one title. The reader's index into the decision; the
    #: rationale below is read because of this line, so it has to be the honest
    #: summary and not a heading.
    summary: str = Field(min_length=1, max_length=MAX_DECISION_SUMMARY_LENGTH)
    #: Why this option, for someone who did not make it. Required, not optional:
    #: see the module docstring on why the reason is the artefact that survives.
    rationale: str = Field(min_length=1, max_length=MAX_RATIONALE_LENGTH)
    #: The options considered and not taken. May be empty, and
    #: :class:`RejectedAlternative` explains why that is not a defect.
    alternatives_rejected: list[RejectedAlternative] = Field(
        default_factory=list, max_length=MAX_ALTERNATIVES_PER_DECISION
    )


class TicketDraft(BaseModel):
    """One ticket the plan says should exist, in every field the store will need.

    A draft, not a ticket: it carries an ``id`` that is a key inside this plan and
    nothing more, which is what dependency edges point at. The store assigns the
    real identifier later, and creation is idempotent by plan id rather than by
    store id -- see :mod:`kojutsu.core.design_plan` on why the plan document has to
    be a pure input for that to work.

    ``priority`` is required where ``labels`` and ``test_command`` are not, and the
    asymmetry is deliberate. A label is genuinely optional: most tickets have none,
    and inventing one to fill the field adds noise to a filtering surface. A
    **priority** is not optional in practice -- it is a triage judgement the
    reconciler has already made in deciding what to put in the plan at all -- and a
    mechanical creation path that leaves it to the store's default produces tickets
    whose priority nobody chose, in a store that cannot tell those apart from tickets
    where someone did.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    #: Key for dependency edges within this plan. Uniqueness is checked across the
    #: plan, because an edge pointing at an id that names two drafts resolves to
    #: neither of them.
    id: DraftId
    #: Verb and noun, as the store's own guidance asks. Bounded by
    #: :data:`MAX_TITLE_LENGTH` so a description cannot be misfiled here and leave
    #: the description field empty.
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    #: What the ticket is for, for a reader deciding whether to work it.
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH)
    #: The conditions that define done. **Non-empty, structurally**: this is the one
    #: list in the plan that ``min_length=1`` makes impossible to omit, and it is the
    #: ticket's own explicit warning about malformed tickets. A draft with no
    #: acceptance criteria is a draft nobody can finish and nobody can tell is
    #: finished, so creation would produce a ticket whose only record of what "done"
    #: means is that nothing was written.
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        min_length=1, max_length=MAX_ACCEPTANCE_CRITERIA
    )
    #: The command that verifies this ticket, when it has one. Optional because a
    #: documentation-only change genuinely may not, and requiring a command would
    #: mean writing a command that verifies nothing.
    test_command: str | None = Field(default=None, max_length=MAX_TEST_COMMAND_LENGTH)
    #: Files to read before starting. Optional: an agent working in a small area
    #: needs none.
    reference_files: list[ReferenceFile] = Field(
        default_factory=list, max_length=MAX_REFERENCE_FILES
    )
    #: A filtering surface, not a place for prose. Bounded twice -- each label by
    #: :data:`MAX_LABEL_LENGTH`, the list by :data:`MAX_LABELS` -- because either
    #: bound alone is satisfied by a list of long labels.
    labels: list[Label] = Field(default_factory=list, max_length=MAX_LABELS)
    #: The triage judgement. A bounded string, **not** an enumeration: see below.
    priority: str = Field(min_length=1, max_length=MAX_PRIORITY_LENGTH)
    #: Draft ids in this plan that must exist first. Empty by default -- most
    #: tickets depend on nothing, and an ordering among independent work is
    #: scheduling noise that a creator would otherwise have to interpret.
    depends_on: list[DraftId] = Field(default_factory=list, max_length=MAX_PLAN_TICKETS)

    #: Why priority is not an enum here.
    #:
    #: No closed priority vocabulary exists in this repository -- not in
    #: ``pyproject.toml``, not in :mod:`kojutsu.models`, not in any enum it
    #: declares -- and the ticket store treats priority as a free string. Inventing
    #: one here would create a second authority for a vocabulary this module does not
    #: own, and the failure would be silent in the exact direction that matters: a
    #: plan would validate here and be refused at creation there, or worse, be
    #: accepted by a store that interprets ``P2`` differently and never says so. An
    #: enum here cannot detect that drift; it can only hide it. What a bounded string
    #: *can* promise is the part that is actually this module's business -- that the
    #: field is present, non-empty, and small enough to be a triage label rather than
    #: a paragraph -- and it will keep that promise under any vocabulary the store
    #: later adopts. If the store ever publishes a closed set, that set belongs in the
    #: creation phase, which is the code that talks to the store.


class DesignPlan(BaseModel):
    """A whole plan: what it is for, what it decided, and what it says should exist.

    ``schema_version`` is ``Literal[1]`` and the field is **not** free-form on
    purpose. A plan is written by one model and read by code that did not write it,
    possibly across a model upgrade, possibly across a process restart months later.
    Without a version, a future format that reuses a field name for a different thing
    is a silent reinterpretation: the old reader parses the new document, finds every
    field present, and reports no error while acting on the wrong meaning. Pinning
    the literal turns that into a refusal, and a refusal at this boundary is
    recoverable in the way a silent reinterpretation is not -- nothing has been
    created yet.

    :attr:`goals`, :attr:`decisions` and :attr:`tickets` are all non-empty. A plan
    with no goals states no purpose, so there is nothing for its decisions or tickets
    to serve; one with no decisions is a wish list; one with no tickets is a document
    about a document. All three are refused rather than tolerated, because a tolerant
    read of an empty section is indistinguishable from a section the writer forgot.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    #: The version of this format. Present on every plan so a reader can tell a
    #: version-2 document from a version-1 one without guessing from its fields.
    schema_version: Literal[1] = 1
    #: What this plan is for. Non-empty: see the class docstring.
    goals: list[Goal] = Field(min_length=1, max_length=MAX_PLAN_GOALS)
    #: What was decided to reach those goals, with the reasoning kept. Non-empty.
    decisions: list[DesignDecision] = Field(min_length=1, max_length=MAX_PLAN_DECISIONS)
    #: The tickets this plan says should exist. Non-empty.
    tickets: list[TicketDraft] = Field(min_length=1, max_length=MAX_PLAN_TICKETS)


def validate_design_plan(plan: DesignPlan) -> DesignPlan:
    """Check every property that no single field can express, and raise if any fails.

    A separate function rather than a ``@model_validator``, following
    ``_validate_ask_plan`` in :mod:`kojutsu.cli`, for three reasons that all point the
    same way.

    **The checks need the whole document at once.** Duplicate ids, dangling edges and
    cycles are properties of the *set* of drafts: none of them is a fact about any
    one field, and each needs every other field resolved before it can be answered.
    That is why ``_validate_ask_plan`` takes the plan rather than a field too.

    **Raising from inside a validator forces the wrong error type.** A
    ``@model_validator`` can only raise a :class:`pydantic.ValidationError`, whose
    ``errors()`` are ``loc``/``msg`` pairs in a nested shape the caller has to walk.
    A graph problem has no meaningful location: "drafts a and b form a cycle" is
    attributed to the model root, so the caller is handed a path that points at
    nothing and a message it has to re-parse to learn which drafts are involved.
    :class:`DesignPlanError` carries one flat sentence per problem, already naming
    the ids.

    **A construction site is not a trust boundary.** Validators run wherever a model
    is built, including in a test that means to build a deliberately malformed plan,
    and ``DesignPlan.model_construct`` skips them entirely. Naming the check as one
    function means the pipeline has one call to make at the one place where untrusted
    model output enters, and a reader can see that call. ``parse_design_plan`` is that
    place, and it always makes it.

    Returns the plan unchanged so it can be chained onto parsing. That is the only
    difference from ``_validate_ask_plan``, which returns ``None`` because it has one
    private caller; this one is a public seam for two later phases.
    """
    problems: list[str] = []

    seen: set[str] = set()
    duplicates: list[str] = []
    for draft in plan.tickets:
        if draft.id in seen and draft.id not in duplicates:
            duplicates.append(draft.id)
        seen.add(draft.id)
    for identifier in duplicates:
        problems.append(
            f"ticket draft id {identifier!r} is used by more than one draft, so a "
            f"dependency edge naming it resolves to neither"
        )

    _refused_text_problems("plan goal", plan.goals, problems)
    for index, decision in enumerate(plan.decisions):
        where = f"decision {index} ({decision.summary!r})"
        _refused_text_problems(f"{where} summary", [decision.summary], problems)
        _refused_text_problems(f"{where} rationale", [decision.rationale], problems)
        for position, alternative in enumerate(decision.alternatives_rejected):
            alternative_where = f"{where} rejected alternative {position}"
            _refused_text_problems(
                f"{alternative_where} description", [alternative.alternative], problems
            )
            _refused_text_problems(
                f"{alternative_where} reason", [alternative.why_rejected], problems
            )

    for draft in plan.tickets:
        where = f"ticket draft {draft.id!r}"
        _refused_text_problems(f"{where} id", [draft.id], problems)
        _refused_text_problems(f"{where} title", [draft.title], problems)
        _refused_text_problems(f"{where} description", [draft.description], problems)
        _refused_text_problems(f"{where} acceptance criteria", draft.acceptance_criteria, problems)
        if draft.test_command is not None:
            _refused_text_problems(f"{where} test command", [draft.test_command], problems)
        _refused_text_problems(f"{where} reference files", draft.reference_files, problems)
        _refused_text_problems(f"{where} labels", draft.labels, problems)
        _refused_text_problems(f"{where} priority", [draft.priority], problems)

    problems.extend(_graph_problems(plan.tickets)[0])

    if problems:
        raise DesignPlanError(problems)
    return plan


def ticket_drafts_in_dependency_order(plan: DesignPlan) -> list[TicketDraft]:
    """Return every draft, ordered so each one follows everything it depends on.

    The order phase 3 creates tickets in, and the only ordering the plan implies.
    Exposed now because it is pure schema work and because the order is a property
    of the document rather than of any consumer: two consumers that each derived
    their own order would disagree silently, and a disagreement about what "first"
    means is invisible in the store afterwards.

    **Ties break by position in the plan, not by iteration order of a set.** The
    output is deterministic across processes: ``str`` hashing is salted per
    interpreter, so an implementation that pulled ready drafts out of a ``set`` would
    produce a different order on every run for the same document. That matters
    because ticket creation is required to be idempotent, and because a plan that
    reorders itself between runs cannot be diffed against its own earlier version.

    Refuses duplicate ids, dangling edges and cycles rather than returning a
    partial order. A partial order is the worst of the three outcomes available: it
    looks like a successful sort, and every ticket it happens to place correctly
    makes the ones it placed wrongly harder to notice. Re-validating here rather than
    trusting the caller is the point -- :func:`validate_design_plan` runs the *same*
    graph check, so there is one implementation of what a well-formed dependency
    graph is and no path that skips it.
    """
    problems, ordered = _graph_problems(plan.tickets)
    if problems:
        raise DesignPlanError(problems)
    return ordered


def parse_design_plan(payload: str | bytes | dict[str, Any]) -> DesignPlan:
    """Parse and validate plan output in one call, or raise naming every problem.

    The single entry point for untrusted output, and the reason
    :class:`DesignPlanError` is the *only* error a malformed plan produces. Without
    it a caller has to handle two unrelated types -- a ``pydantic.ValidationError``
    from the parse and a :class:`DesignPlanError` from the cross-field check -- and
    the class of plan that gets that wrong is the dangerous one: the plan that is
    well-formed field by field but has a dependency cycle, which a caller who only
    caught ``ValidationError`` treats as fine.

    Structural problems are reported through the same ``problems`` tuple as graph
    ones. A plan that failed to parse has no draft ids to quote, so those problems
    carry their pydantic location path (``tickets.3.acceptance_criteria``) instead of
    a draft id; a plan that *did* parse gets every problem naming the draft it is
    about. Resolving an index to a draft by hand is the one step left to the reader,
    and it is only left for documents that never became objects.

    The two layers never report each other's problems, and that is not a shortcut:
    a document that fails to parse has no drafts, so a graph problem is not merely
    unmentioned at that point, it is not yet expressible. Within either layer
    everything is reported at once, which is the property that matters -- the repair
    is one pass over one list.
    """
    try:
        if isinstance(payload, (str, bytes)):
            plan = DesignPlan.model_validate_json(payload)
        else:
            plan = DesignPlan.model_validate(payload)
    except ValidationError as error:
        raise DesignPlanError(_structural_problems(error)) from error
    return validate_design_plan(plan)


def _graph_problems(
    drafts: Sequence[TicketDraft],
) -> tuple[list[str], list[TicketDraft]]:
    """Return every dependency-graph fault, and the order if there are none.

    One implementation for both callers, and the reason
    :func:`ticket_drafts_in_dependency_order` can promise to fail loudly: it does not
    call :func:`validate_design_plan` and hope the answer is still fresh, it asks the
    same function that produces the order whether the graph is sound.

    The cycle search runs over the subgraph of edges that **resolve**. This is the
    detail that lets one pass report everything: a dangling edge leaves its target with
    no possible start, so a naive topological sort reports it as a cycle, and a plan
    with a dangling edge *and* a genuine cycle two drafts over would be told only
    about the cycle it does not have. Unresolvable edges are reported as dangling and
    then excluded, so the residual set means what it says.

    Self-reference is caught **by this same traversal** -- a self-edge is a cycle of
    length one -- and is additionally reported under its own name, because "draft 'a'
    depends on itself" and "drafts 'a' and 'b' form a cycle" call for different fixes:
    the first is a stray edge to delete, the second is a real ordering decision with
    no obviously right answer. The self-edge is then left out of the traversal so one
    fault produces one problem, rather than the same fault reported twice.
    """
    problems: list[str] = []
    known = {draft.id for draft in drafts}

    blockers: dict[str, list[str]] = {}
    for draft in drafts:
        resolved: list[str] = []
        for dependency in draft.depends_on:
            if dependency == draft.id:
                problems.append(
                    f"ticket draft {draft.id!r} depends on itself, which is a cycle of "
                    f"length one and never becomes startable"
                )
            elif dependency not in known:
                problems.append(
                    f"ticket draft {draft.id!r} depends on {dependency!r}, which is not a "
                    f"draft in this plan"
                )
            else:
                resolved.append(dependency)
        blockers[draft.id] = resolved

    ordered, stuck = _topological_order(drafts, blockers)
    if stuck:
        problems.append(
            f"ticket drafts {', '.join(repr(identifier) for identifier in stuck)} form a "
            f"dependency cycle, so no order can put each of them after everything it "
            f"depends on"
        )
    return problems, ordered


def _topological_order(
    drafts: Sequence[TicketDraft],
    blockers: dict[str, list[str]],
) -> tuple[list[TicketDraft], list[str]]:
    """Order drafts so each follows its blockers; return what could not be placed.

    Kahn's algorithm, scanning the drafts in plan order each round rather than
    holding the ready set in a ``set``. The scan is the linear-in-``n`` step a queue
    would also give, and it buys determinism for free: the tie-break is the order the
    reconciler wrote the plan in, which is a property of the document, instead of a
    hash order that changes per process. See
    :func:`ticket_drafts_in_dependency_order` for why that determinism is load
    bearing rather than tidy.

    Returns the unplaceable ids rather than reporting them, so the caller can name
    them in its own sentence.
    """
    remaining = list(drafts)
    placed: set[str] = set()
    ordered: list[TicketDraft] = []
    while remaining:
        for position, draft in enumerate(remaining):
            if all(dependency in placed for dependency in blockers[draft.id]):
                remaining.pop(position)
                placed.add(draft.id)
                ordered.append(draft)
                break
        else:
            return ordered, [draft.id for draft in remaining]
    return ordered, []


def _refused_text_problems(where: str, values: Iterable[str], problems: list[str]) -> None:
    """Append one problem per string carrying a character this project refuses to store.

    Uses :data:`kojutsu.core.text_hygiene.CONTROL_RE` rather than the inline
    ``ord(character) < 32`` that ``_validate_ask_plan`` applies to question ids, and
    the difference is not cosmetic. That check is right for an identifier, where a
    newline is always wrong; it is wrong here, because a goal, a rationale or a
    description is prose and prose is legitimately more than one line, and a rule that
    refuses the second line of a decision refuses the formatting the decision was
    written in. ``CONTROL_RE`` is the project's named policy -- it spares TAB and LF,
    and it refuses DEL and the C1 controls, which the ``ord`` check lets through --
    and it is named in one place so a change to what is refused is a change to one
    set rather than to every call site that re-derived it.

    Named fields one at a time rather than walked generically over the model. A walk
    would be shorter and would cover a field added next month for free, but it works
    by serialising, so it silently stops covering anything excluded from the dump,
    and a string nothing checks is exactly the silent-garbage case this module
    exists to prevent. The list is verbose; the verbosity is the audit trail.
    """
    for value in values:
        found = CONTROL_RE.search(value)
        if found is not None:
            problems.append(
                f"{where} contains the refused control character U+{ord(found.group()):04X}"
            )


def _structural_problems(error: ValidationError) -> list[str]:
    """Flatten pydantic's per-field errors into one sentence each.

    Pydantic already collects every field error rather than stopping at the first,
    which is the property this module needs and the reason it does not re-implement
    bound checking. What it does not do is say them in a shape a reader can act on:
    ``errors()`` is a list of ``loc``/``msg`` pairs for a caller that knows how to walk
    it, and the caller here is a model that has to repair a document.
    """
    return [
        f"{'.'.join(str(part) for part in detail['loc']) or 'plan'}: {detail['msg']}"
        for detail in error.errors()
    ]
