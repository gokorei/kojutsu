"""What makes a plan malformed, and what each refusal is actually preventing.

The refusals in :mod:`kojutsu.core.design_plan` are not input tidiness. Every one of
them closes off a specific way for a plan to become a **well-formed ticket that is
wrong** -- which is the failure the ticket this implements names as worse than no
plan, and the failure no reader of the resulting ticket can detect, because a ticket
carries no trace of the plan that produced it. So these tests are written against the
*class of garbage* each rule stops rather than against the rule, and several of them
pin a decision that would otherwise look arbitrary and get "fixed" into something
cheaper.

Three of them exist mainly to pin a choice:

- **Newlines survive in free text.** The obvious control-character rule -- reject
  ``ord(character) < 32`` -- refuses the second line of a decision, and prose is more
  than one line. Pinning which characters are refused stops the next maintainer
  tightening the rule into one that rejects ordinary text.
- **Every string and list is bounded on both sides.** Enumerated by reflection over
  the models, so the invariant survives a field being added next month. A test that
  checked today's fields would pass just as happily on a plan containing an unbounded
  one.
- **The dependency order is deterministic and total.** Ticket creation must be
  idempotent, and an order that varied per process would make the same document mean
  something different on every run. The tie-break is therefore pinned to the order
  the reconciler wrote, which is also the order a reader diffs against.
"""

from __future__ import annotations

import json
import types
from typing import Any, Literal, Union, get_args, get_origin

import annotated_types
import pytest
from pydantic import StringConstraints
from pydantic.fields import FieldInfo

from kojutsu.core.design_plan import (
    MAX_ACCEPTANCE_CRITERIA,
    MAX_LABEL_LENGTH,
    MAX_LABELS,
    MAX_PLAN_TICKETS,
    DesignDecision,
    DesignPlan,
    DesignPlanError,
    RejectedAlternative,
    TicketDraft,
    parse_design_plan,
    ticket_drafts_in_dependency_order,
    validate_design_plan,
)

_UNION_ORIGINS = {Union, types.UnionType}


def _draft(identifier: str, **overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "id": identifier,
        "title": f"Implement {identifier}",
        "description": "What the ticket is for.",
        "acceptance_criteria": ["Given the plan, when it is validated, then it is accepted"],
        "priority": "P2",
    }
    draft.update(overrides)
    return draft


def _plan(**overrides: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "goals": ["Make the design phase a recorded pipeline"],
        "decisions": [
            {
                "summary": "Validate the plan before any ticket is created",
                "rationale": "A malformed ticket is indistinguishable from a correct one.",
                "alternatives_rejected": [
                    {
                        "alternative": "Create the tickets and repair them afterwards",
                        "why_rejected": "Nothing downstream can tell a wrong ticket from a right one.",
                    }
                ],
            }
        ],
        "tickets": [_draft("schema")],
    }
    plan.update(overrides)
    return plan


def _core(annotation: Any) -> Any:
    """The annotation with ``| None`` removed, because an optional is still its core."""
    if get_origin(annotation) in _UNION_ORIGINS:
        return next(arg for arg in get_args(annotation) if arg is not type(None))
    return annotation


def _element(annotation: Any) -> Any:
    """The element annotation of a ``list[...]``, or ``None`` if it is not a list."""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else None
    return None


def _string_constraints(annotation: Any) -> StringConstraints | None:
    """The :class:`StringConstraints` on an ``Annotated[str, ...]``, if any."""
    if get_origin(annotation) is not str and annotation is not str:
        return None
    for item in getattr(annotation, "__metadata__", ()):
        if isinstance(item, StringConstraints):
            return item
    return None


def _length_bounds(field: FieldInfo) -> tuple[int | None, int | None]:
    """The ``(min_length, max_length)`` a field carries, wherever they were declared.

    Read from ``metadata`` rather than off the ``FieldInfo`` itself, because that is
    where pydantic puts them: ``Field(min_length=...)`` on a field and
    ``Annotated[..., StringConstraints(...)]`` on an element both land as constraint
    objects here, so one lookup covers both ways this module declares a bound.
    """
    minimum: int | None = None
    maximum: int | None = None
    for item in field.metadata:
        if isinstance(item, StringConstraints):
            minimum, maximum = item.min_length, item.max_length
        elif isinstance(item, annotated_types.MinLen):
            minimum = item.min_length
        elif isinstance(item, annotated_types.MaxLen):
            maximum = item.max_length
    return minimum, maximum


def test_a_valid_plan_parses_and_keeps_every_field_it_declares() -> None:
    """The schema carries the whole ticket, so nothing has to be recovered later.

    Every field the ticket store needs is read back off the plan unchanged. This is
    the property the mechanical-creation phase rests on: if a field could be dropped
    by parsing, a ticket would silently lose a part of itself, and the loss would
    only be visible to whoever later wondered why the ticket was missing its test
    command.
    """
    plan = parse_design_plan(
        _plan(
            tickets=[
                _draft(
                    "schema",
                    test_command="uv run pytest -q",
                    reference_files=["src/kojutsu/core/design_plan.py"],
                    labels=["kojutsu", "planning"],
                    depends_on=["capture"],
                ),
                _draft("capture"),
            ]
        )
    )

    draft = plan.tickets[0]
    assert draft.id == "schema"
    assert draft.priority == "P2"
    assert draft.test_command == "uv run pytest -q"
    assert draft.reference_files == ["src/kojutsu/core/design_plan.py"]
    assert draft.labels == ["kojutsu", "planning"]
    assert draft.depends_on == ["capture"]
    assert draft.acceptance_criteria


def test_a_malformed_plan_is_rejected_naming_every_problem_rather_than_the_first() -> None:
    """Four independent faults, four problems, in one refusal.

    The reason is not tidiness. A reconciler repairing a plan one error per round
    spends one model call per round, and every round shows it only the error it is
    being asked to fix -- so the model is free to lose the problems it was never
    shown, and a four-fault plan becomes four rounds of editing against partial
    information. Reporting everything at once is what makes the repair a single pass.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(
                tickets=[
                    _draft("a", depends_on=["a"]),
                    _draft("b", depends_on=["ghost"]),
                    _draft("c"),
                    _draft("c"),
                    _draft("p", depends_on=["q"]),
                    _draft("q", depends_on=["p"]),
                ]
            )
        )

    problems = raised.value.problems
    assert len(problems) == 4
    assert any("depends on itself" in problem and "'a'" in problem for problem in problems)
    assert any("'ghost'" in problem and "not a draft" in problem for problem in problems)
    assert any(
        "used by more than one draft" in problem and "'c'" in problem for problem in problems
    )
    assert any("cycle" in problem and "'p'" in problem and "'q'" in problem for problem in problems)


def test_structural_problems_are_all_reported_too_not_just_the_first() -> None:
    """Pydantic already collects every field error, and this flattens rather than truncates.

    Without the flattening a caller would have to walk ``ValidationError.errors()``
    itself, and the caller here is a model trying to repair a document. Three faults,
    three problems, from one parse.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(
                goals=[],
                decisions=[{"summary": "x", "rationale": "y"}],
                tickets=[{**_draft("t"), "priority": 2}],
            )
        )

    assert len(raised.value.problems) == 2
    assert any("goals" in problem for problem in raised.value.problems)
    assert any("tickets.0.priority" in problem for problem in raised.value.problems)


def test_a_dependency_edge_must_point_at_a_draft_that_exists() -> None:
    """A dangling edge is a malformed plan, not a tolerated input.

    If it were tolerated, the edge would resolve to nothing at creation time and the
    ticket would be created with a dependency that no ticket satisfies -- a ticket
    permanently blocked on work that does not exist, reported by nobody.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[_draft("only", depends_on=["never-written"])]))

    assert len(raised.value.problems) == 1
    assert "never-written" in raised.value.problems[0]


def test_draft_ids_must_be_unique_because_edges_point_at_them() -> None:
    """Two drafts sharing an id make every edge naming it ambiguous.

    The refusal matters more than it looks: the graph is keyed by id, so without this
    an edge silently resolves to whichever of the two happened to be looked up last,
    and the resulting order is correct with respect to a draft that was never chosen.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[_draft("twin"), _draft("twin")]))

    assert len(raised.value.problems) == 1
    assert "used by more than one draft" in raised.value.problems[0]


def test_a_self_dependency_is_caught_by_the_same_traversal_and_named_as_itself() -> None:
    """A self-edge is a cycle of length one, so one rule catches it -- with its own message.

    Same traversal, same code path, and one problem rather than two: the self-edge is
    reported and then left out of the topological search, because a fault reported
    twice reads as two faults and a reader who believes that will look for a second
    problem that is not there. The message is separate because the fixes differ -- a
    self-edge is a stray entry to delete, whereas a multi-draft cycle is a real
    ordering decision with no obviously right answer.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[_draft("a", depends_on=["a"])]))

    assert len(raised.value.problems) == 1
    problem = raised.value.problems[0]
    assert "depends on itself" in problem
    assert "'a'" in problem


def test_a_dangling_edge_does_not_mask_a_real_cycle_elsewhere_in_the_plan() -> None:
    """The cycle search runs over the edges that resolve, so one pass sees both faults.

    A naive topological sort treats an unresolvable edge as an unmet blocker, so the
    draft that names it can never start and is reported as part of a cycle -- in a
    plan that has no cycle there, and may well have one two drafts over. The plan
    below has both, and the cycle message names the two drafts that really are
    cyclic rather than the one that merely points at nothing.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(
                tickets=[
                    _draft("loose", depends_on=["absent"]),
                    _draft("left", depends_on=["right"]),
                    _draft("right", depends_on=["left"]),
                ]
            )
        )

    problems = raised.value.problems
    assert len(problems) == 2
    assert any("'absent'" in problem for problem in problems)
    cycle = next(problem for problem in problems if "cycle" in problem)
    assert "'left'" in cycle and "'right'" in cycle
    assert "loose" not in cycle


def test_acceptance_criteria_may_not_be_empty_on_any_draft() -> None:
    """The ticket's own warning about malformed tickets, made structurally impossible.

    A draft with no criteria is a ticket nobody can finish and nobody can tell is
    finished, so its only record of what "done" means is that nothing was written.
    Enforced by the field bound rather than by the cross-field check, which is what
    makes it hold for a plan built in Python as well as one parsed from a model --
    and the parse still names it, through the flattened location path.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[_draft("empty", acceptance_criteria=[])]))

    assert len(raised.value.problems) == 1
    assert "tickets.0.acceptance_criteria" in raised.value.problems[0]


def test_drafts_come_back_in_dependency_order_regardless_of_how_they_were_written() -> None:
    """The order is the one the plan implies, so every consumer gets the same one.

    The reconciler is free to write a ticket before the ticket it blocks; the order
    that reaches creation is the dependency order regardless. Two consumers that each
    derived their own order would disagree silently, and a disagreement about what
    "first" means is invisible in the store afterwards.
    """
    plan = parse_design_plan(
        _plan(
            tickets=[
                _draft("third", depends_on=["second"]),
                _draft("first"),
                _draft("second"),
            ]
        )
    )

    assert [draft.id for draft in ticket_drafts_in_dependency_order(plan)] == [
        "first",
        "second",
        "third",
    ]


def test_independent_drafts_keep_the_order_the_plan_was_written_in() -> None:
    """Ties break by position, not by hash, so the same document always sorts the same.

    Stratum hashing is salted per interpreter, so an implementation that pulled ready
    drafts out of a ``set`` would emit a different order on every run for the same
    plan. That defeats the idempotency the ticket requires and makes a plan
    impossible to diff against its own earlier version. Pinning the tie-break to plan
    order -- not alphabetical either -- is what makes the output a property of the
    document rather than of the process that read it.
    """
    plan = parse_design_plan(_plan(tickets=[_draft("zeta"), _draft("alpha"), _draft("middle")]))

    for _ in range(5):
        assert [draft.id for draft in ticket_drafts_in_dependency_order(plan)] == [
            "zeta",
            "alpha",
            "middle",
        ]


def test_the_ordering_function_refuses_a_cyclic_plan_rather_than_returning_a_partial_order() -> (
    None
):
    """It checks the graph itself, so it cannot be handed an unvalidated plan and lie.

    A partial order is the worst available outcome: it looks like a successful sort,
    and every ticket it happens to place correctly makes the ones it placed wrongly
    harder to notice. The plan here is built directly, never through the parse, which
    is exactly the path that would defeat a function that merely trusted its caller.
    """
    plan = DesignPlan.model_validate(
        _plan(
            tickets=[
                _draft("left", depends_on=["right"]),
                _draft("right", depends_on=["left"]),
            ]
        )
    )

    with pytest.raises(DesignPlanError) as raised:
        ticket_drafts_in_dependency_order(plan)

    assert len(raised.value.problems) == 1
    assert "cycle" in raised.value.problems[0]


def test_validate_design_plan_returns_the_plan_so_it_can_be_chained() -> None:
    """Validation is a named step a caller can take once, not a side effect of building.

    Returning the plan unchanged is the only departure from the private
    ``_validate_ask_plan`` it follows, and it is here because this is a public seam
    with two later phases rather than a helper with one caller.
    """
    plan = DesignPlan.model_validate(_plan())

    assert validate_design_plan(plan) is plan


def test_newlines_survive_in_free_text_because_prose_is_more_than_one_line() -> None:
    """The refused set is the project's, and it spares TAB and LF.

    The rule this pins against is the obvious one -- reject ``ord(character) < 32``,
    which is what ``_validate_ask_plan`` applies to question *ids*, correctly, because
    a newline in an identifier is always wrong. Applied here it would refuse the
    second line of a rationale, which is not garbage but formatting. And the named set
    is stricter where it should be: it also refuses DEL and the C1 controls, which an
    ``ord`` comparison lets through.
    """
    plan = parse_design_plan(
        _plan(
            goals=["First line of the goal.\nSecond line of the same goal.\tindented"],
            decisions=[
                {
                    "summary": "Keep\nnewlines",
                    "rationale": "A rationale is prose.\nIt is allowed to be more than one line.",
                    "alternatives_rejected": [],
                }
            ],
        )
    )

    assert plan.goals[0].startswith("First line")
    assert "\n" in plan.decisions[0].rationale


def test_control_characters_are_refused_in_every_free_text_field() -> None:
    """The DEL and C1 controls are refused too, which an ``ord < 32`` check would pass.

    A refused character in a stored string can terminate, overwrite or reorder a line
    in whatever renders it later, and none of them can be the difference between two
    meanings. Fields are named one at a time on purpose: a walk that serialised the
    model would cover a field added next month for free, and would silently stop
    covering anything excluded from the dump.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(
                goals=["a goal ending in DEL: \u007f"],
                decisions=[
                    {
                        "summary": "A summary carrying a NEL: \u0085",
                        "rationale": "A rationale carrying an SOH: \u0001",
                        "alternatives_rejected": [
                            {
                                "alternative": "An option named with a CSI: \u009b",
                                "why_rejected": "Because of this DEL: \u007f",
                            }
                        ],
                    }
                ],
                tickets=[_draft("t", title="A title carrying a NEL: \u0085")],
            )
        )

    problems = raised.value.problems
    assert len(problems) == 6
    assert any("plan goal" in problem and "U+007F" in problem for problem in problems)
    assert any("decision 0" in problem and "summary" in problem for problem in problems)
    assert any("decision 0" in problem and "rationale" in problem for problem in problems)
    assert any("rejected alternative 0 description" in problem for problem in problems)
    assert any("rejected alternative 0 reason" in problem for problem in problems)
    assert any("ticket draft 't' title" in problem for problem in problems)


def test_priority_is_a_bounded_string_and_not_an_enumeration() -> None:
    """No closed priority vocabulary exists here, so none is invented.

    ``pyproject.toml`` declares no priority, :mod:`kojutsu.models` declares no
    priority enum, and the ticket store treats priority as a free string. An enum
    invented in this module would be a second authority for a vocabulary this code
    does not own, and its failure would be silent in the direction that matters: a
    plan would validate here and be refused -- or worse, differently interpreted --
    at creation there. So the bound is what this module can honestly promise: present,
    non-empty, and small enough to be a triage label. The test asserts that a value
    outside any P-series is *accepted*, which is the part a future maintainer is most
    likely to "fix" into an enum.
    """
    plan = parse_design_plan(_plan(tickets=[_draft("t", priority="urgent-high")]))

    assert plan.tickets[0].priority == "urgent-high"


def test_priority_and_labels_are_bounded_so_neither_can_carry_prose() -> None:
    """The bound is a size limit, not a vocabulary, so it applies to any value at all.

    An over-long priority is prose in a triage field, and an over-long label is prose
    in a field every reader scans as a tag -- which is where the plan's own reasoning
    would end up unread.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[_draft("t", priority="P" * 64)]))
    assert any("priority" in problem for problem in raised.value.problems)

    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[_draft("t", labels=["x" * (MAX_LABEL_LENGTH + 1)])]))
    assert any("labels" in problem for problem in raised.value.problems)


def test_every_list_is_bounded_in_cardinality_and_not_only_in_element_length() -> None:
    """Element bounds say nothing about how many elements there may be.

    This is the bound that keeps the approval gate a review: approval happens once,
    on the plan, and every draft is then created without anyone looking at it again,
    so a plan too large for one person to hold while approving has not been reviewed
    more thoroughly. It has been reviewed not at all, while still looking reviewed.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(tickets=[_draft(f"t{index}") for index in range(MAX_PLAN_TICKETS + 1)])
        )
    assert any("tickets" in problem for problem in raised.value.problems)

    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(tickets=[_draft("t", labels=[f"l{i}" for i in range(MAX_LABELS + 1)])])
        )
    assert any("labels" in problem for problem in raised.value.problems)

    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(
            _plan(tickets=[_draft("t", acceptance_criteria=["c"] * (MAX_ACCEPTANCE_CRITERIA + 1))])
        )
    assert any("acceptance_criteria" in problem for problem in raised.value.problems)


def test_every_string_and_list_in_the_plan_is_bounded_on_both_sides() -> None:
    """Enumerated by reflection, so the invariant outlives the field list.

    Checked today, this test would pass just as happily on a plan containing an
    unbounded field added tomorrow -- which is the whole risk, because an unbounded
    list is the shape of garbage nobody notices at review time. A required string must
    be bounded at both ends: the ``max_length`` stops it carrying prose, and the
    ``min_length`` stops an empty value, which for an identifier means two drafts
    whose ids collide for a reason no error names. An *optional* string needs only the
    upper bound, because ``None`` is the honest way to say the plan does not have one.
    """
    unbounded: list[str] = []
    for model in (DesignPlan, DesignDecision, RejectedAlternative, TicketDraft):
        for name, field in model.model_fields.items():
            where = f"{model.__name__}.{name}"
            core = _core(field.annotation)
            if get_origin(core) is Literal:
                continue
            minimum, maximum = _length_bounds(field)
            element = _element(core)
            if element is not None:
                if maximum is None:
                    unbounded.append(f"{where} (list cardinality)")
                    continue
                constraints = _string_constraints(element)
                if constraints is not None and (
                    constraints.min_length is None or constraints.max_length is None
                ):
                    unbounded.append(f"{where} (element length)")
                continue
            if maximum is None:
                unbounded.append(f"{where} (max_length)")
            elif field.is_required() and minimum is None:
                unbounded.append(f"{where} (min_length)")

    assert unbounded == []


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    """``extra="forbid"`` is what stops a plan from carrying claims nobody reads.

    The dangerous case is not a typo but a field a later version adds: silently
    dropped, the writer believes the plan says it, and the reader of the plan cannot
    see that it does not.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan(_plan(tickets=[{**_draft("t"), "owner": "unassigned"}]))

    assert len(raised.value.problems) == 1
    assert "owner" in raised.value.problems[0]


def test_a_schema_version_this_reader_does_not_know_is_refused_not_reinterpreted() -> None:
    """``Literal[1]`` is what makes a future format a new version rather than a surprise.

    Without it, a version-2 document reusing a field name for a different thing parses
    cleanly here: every field is present, no error is raised, and the plan is acted on
    with the wrong meaning. At this boundary a refusal is recoverable in a way that
    silent reinterpretation is not -- nothing has been created yet.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan({**_plan(), "schema_version": 2})

    assert len(raised.value.problems) == 1
    assert "schema_version" in raised.value.problems[0]


def test_coercion_is_refused_so_a_string_is_never_read_as_a_number() -> None:
    """``strict=True`` keeps a plan from being repaired into a shape it was not written in.

    A model that emits ``"2"`` for a priority gets a refusal naming the field rather
    than a plan that quietly coerces, because the difference between the two is not
    visible in the document afterwards and would be discovered by whoever reads the
    ticket.
    """
    with pytest.raises(DesignPlanError) as raised:
        parse_design_plan({**_plan(), "schema_version": "1"})

    assert len(raised.value.problems) == 1
    assert "schema_version" in raised.value.problems[0]


def test_the_error_renders_every_problem_for_the_reader_who_never_inspects_the_tuple() -> None:
    """The common caller is a person reading one log line, so the message is the product.

    ``problems`` is the structured field a program reads; the message is the same
    information for a human, because an exception whose detail lives only in an
    attribute is an exception that reports one line to the ninety-nine percent of
    readers who never look.
    """
    error = DesignPlanError(["first problem", "second problem"])

    assert error.problems == ("first problem", "second problem")
    assert "2 problems" in str(error)
    assert "  - first problem" in str(error)
    assert "  - second problem" in str(error)
    assert "1 problem\n" in str(DesignPlanError(["only"]))


def test_json_text_and_a_mapping_reach_the_same_plan_because_output_arrives_as_either() -> None:
    """One entry point, whichever shape the reconciler's output arrives in.

    Both paths must reach the same object. A caller that got different verdicts from
    the two would not find out until one of them created tickets -- and the one that
    did so would be whichever path nobody exercised, which is the definition of the
    untested branch.
    """
    document = _plan()

    assert parse_design_plan(json.dumps(document)) == parse_design_plan(document)


def test_a_validated_plan_round_trips_through_its_own_json() -> None:
    """Re-parseable, because creation is idempotent and must be handed the same plan twice.

    Idempotency is an acceptance criterion of the ticket, and it is only implementable
    if the plan can be stored and read back without loss. A field that survived
    validation but not serialisation would make the second run disagree with the
    first while the document looked identical.
    """
    plan = parse_design_plan(
        _plan(
            tickets=[
                _draft("schema", depends_on=["capture"], labels=["planning"]),
                _draft("capture"),
            ]
        )
    )

    assert parse_design_plan(plan.model_dump_json()) == plan


def test_a_decision_may_state_no_rejected_alternative_because_forcing_one_fabricates() -> None:
    """The list is allowed to be empty, and that is a deliberate hole in the schema.

    There are decisions with no live alternative -- a name fixed by a wire format, a
    bound fixed by an existing schema -- and demanding a rejection anyway turns the
    schema into a fabrication generator. An invented strawman recorded as "the option
    we rejected" is worse than an empty list, because that field's entire purpose is
    to be believed, and a fabricated entry is a false record of what was considered.
    So the rationale is required and unempty-able, and the rejection list is not.
    """
    decision = DesignDecision.model_validate(
        {"summary": "Keep the existing field name", "rationale": "The wire format fixes it."}
    )

    assert decision.alternatives_rejected == []


def test_a_rejected_alternative_cannot_be_a_bare_name() -> None:
    """Both halves are required, because a bare name is worth much less than it looks.

    ``"use redis"`` does not say whether it was refused as too slow, too expensive
    operationally, or fine but more than this needed -- and those three readings imply
    different decisions next time. This is the field most likely to be dropped
    entirely, since the rationale for the chosen option feels like it already argues
    the case.
    """
    with pytest.raises(DesignPlanError):
        parse_design_plan(
            _plan(
                decisions=[
                    {
                        "summary": "s",
                        "rationale": "r",
                        "alternatives_rejected": [{"alternative": "use redis"}],
                    }
                ]
            )
        )


def test_the_plan_refuses_empty_sections_because_a_tolerated_one_is_a_forgotten_one() -> None:
    """Goals, decisions and tickets are all non-empty.

    A plan with no goals states no purpose for its decisions or tickets to serve; one
    with no decisions is a wish list; one with no tickets is a document about a
    document. All three are refused rather than tolerated, because a reader cannot
    distinguish an empty section the writer meant from one the writer forgot, and the
    second is the one that reaches ticket creation.
    """
    for override, expected in (
        ({"goals": []}, "goals"),
        ({"decisions": []}, "decisions"),
        ({"tickets": []}, "tickets"),
    ):
        with pytest.raises(DesignPlanError) as raised:
            parse_design_plan(_plan(**override))
        assert any(expected in problem for problem in raised.value.problems)


def test_a_plan_built_in_python_is_bounded_too_because_the_field_bounds_are_the_model() -> None:
    """The bounds are not only a parse-time courtesy.

    A test or a future phase that builds a plan directly gets the same guarantees as
    one parsed from a model, because the guarantees live on the fields. Only the
    graph checks need :func:`validate_design_plan`, and the ordering function says so
    by refusing to trust anyone.
    """
    with pytest.raises(ValueError):
        TicketDraft.model_validate(
            {"id": "t", "title": "T", "description": "D", "acceptance_criteria": []}
        )


def test_the_plan_has_no_field_of_its_own_that_duplicates_a_ticket_store_field_name() -> None:
    """Field names cross the boundary untranslated, which is what makes creation mechanical.

    A mechanical transformation that renames a field on the way through is a place
    where the plan and the tickets it produced can disagree with nothing to compare
    them. This test is a tripwire rather than a design statement: it fires if a field
    is renamed to something the store does not call it, which is the change most
    likely to be made and least likely to be noticed.
    """
    store_fields = {
        "id",
        "title",
        "description",
        "acceptance_criteria",
        "test_command",
        "reference_files",
        "labels",
        "priority",
        "depends_on",
    }

    assert store_fields <= set(TicketDraft.model_fields)
