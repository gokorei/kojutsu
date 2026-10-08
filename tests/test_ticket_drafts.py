"""What the design phase's last step refuses, and what it leaves behind afterwards.

The grouping is the argument, in the same way :mod:`tests.test_design_plan` and
:mod:`tests.test_design_capture` group theirs: a suite that only exercised the happy path
would pass for an implementation that appended the ticket titles to a list.

- **Identity is derived, and pinned.** The ticket id is the mechanism that makes
  re-running safe, so it is asserted as a golden -- a derivation that moves silently
  re-identifies every ticket and then creates a second copy of every one of them, which is
  invisible in a diff.
- **Nothing exists before a person approves.** The gate is *driven*, so the test asserts
  the refusal and the recorded decision together. Asserting only that no ticket appeared
  would pass for an implementation that consulted no gate at all, which is precisely the
  failure requirement one is about.
- **The approval is a record, not a flag.** Two approvals of the same plan, a blank
  approver, and a re-approval with a later clock are the cases where a flag and a record
  behave differently.
- **Creation is mechanical.** Every field the plan declared is read back off the row, and
  the dependency edges are asserted to be rows rather than prose -- a description is read
  by a person, a dependency by code.
- **Idempotency is pinned against the real store**, for the reason
  :mod:`tests.test_design_capture` gives: the claim is a claim about a primary key, and a
  fake would agree with any implementation. One test drives the real
  :func:`~kojutsu.core.design_capture.reconcile_design_proposals` twice so that
  ``captured=False`` -- what an idempotent re-run looks like -- is exercised rather than
  asserted in a docstring.
- **A changed plan is refused before anything is written**, and one test then shows the
  second line of defence: a changed plan finds no approval, because the approval is
  anchored on the digest.
- **A partial set is resumable, not rolled back.** The failure is injected, and the
  assertion is that the tickets already written survive and the next run finishes rather
  than duplicating.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kojutsu.core import ticket_drafts
from kojutsu.core.design_capture import (
    ReconciliationOutcome,
    capture_design_proposal,
    design_plan_digest,
    reconcile_design_proposals,
)
from kojutsu.core.design_plan import DesignPlan, DesignPlanError, parse_design_plan
from kojutsu.core.gates import (
    DEFAULT_GATES,
    Gate,
    GateAction,
    GateRegistry,
    GateVerdict,
    LifecyclePoint,
)
from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome, KnowledgeDeliveryStatus
from kojutsu.core.question_registry import SqliteQuestionRegistry
from kojutsu.core.text_hygiene import SANITISATION_KEY
from kojutsu.core.ticket_drafts import (
    PLAN_APPROVAL_IDENTITY_DOMAIN,
    PLAN_GATE_DECISION_IDENTITY_DOMAIN,
    TICKET_IDENTITY_DOMAIN,
    DerivedTicket,
    PlanApproval,
    PlanApprovalLedger,
    PlanApprovalRequiredError,
    PlanChangedError,
    RecordedDecision,
    SqlitePlanApprovalLedger,
    SqliteTicketSink,
    TicketCreationOutcome,
    TicketSetIncompleteError,
    TicketSink,
    create_tickets_from_plan,
    derive_ticket,
    plan_approval_gates,
    record_plan_approval,
    stable_plan_approval_id,
    stable_plan_gate_decision_id,
    stable_ticket_id,
)
from kojutsu.identity import identity_preimage
from kojutsu.text_limits import MAX_AGENT_ID_CHARS

PLAN_SHA = "a" * 64
OTHER_PLAN_SHA = "b" * 64
RECONCILIATION_ID = "design-reconciliation-v1-" + "0" * 64
APPROVER = "dana"
REASON = "Gate every write so the transaction boundary is the reason a ticket exists."
APPROVED_AT = datetime(2026, 10, 3, 9, 30, tzinfo=UTC)

#: The three derivations, pinned as goldens. Same property as
#: ``tests/test_identity_derivations.py`` holds for the other namespaces; the pin lives in
#: the file that owns the derivation because this file is the one allowed to touch it.
TICKET_ID = "design-ticket-v1-c7067e62282f4960fb0f8dbbc08734735751d972ce7580d4d97b763550eee1fa"
APPROVAL_ID = "plan-approval-v1-0f22fd4cb8e4c94d3255b1dadd358687854858c826f72fb96624cd971d4efae0"
DECISION_ID = (
    "plan-gate-decision-v1-83317affe3640b30e7622eb55efdc4299f675d65e6e7ce5144d0c06959ed4554"
)

#: U+202E RIGHT-TO-LEFT OVERRIDE, which :mod:`kojutsu.core.design_plan` does *not* refuse:
#: its character check is :data:`~kojutsu.core.design_plan.CONTROL_RE`, and a bidi control
#: is invisible. The store is where this is caught, which is the whole reason the policy
#: runs at the storage boundary.
RLO = "‮"  # noqa: PLE2502 - fixture data for the bidi filter


class FakeSink:
    """Records what was stored. No network, no store, no clock worth asserting on."""

    def __init__(self, outcome: KnowledgeDeliveryOutcome | None = None) -> None:
        self.entries: list[Any] = []
        self.outcome = outcome

    def store(self, entry: Any) -> KnowledgeDeliveryOutcome | None:
        self.entries.append(entry)
        return self.outcome


def _ticket_draft(identifier: str, *, depends_on: tuple[str, ...] = (), **overrides: Any):
    draft: dict[str, Any] = {
        "id": identifier,
        "title": f"Implement {identifier}",
        "description": f"Make {identifier} the thing the plan says it should be.",
        "acceptance_criteria": [f"Given {identifier}, when it runs, then nothing else changes"],
        "test_command": "uv run pytest -q",
        "reference_files": [f"src/kojutsu/core/{identifier}.py"],
        "labels": ["design-phase"],
        "priority": "P2",
        "depends_on": list(depends_on),
    }
    draft.update(overrides)
    return draft


def _plan(*drafts: dict[str, Any]) -> DesignPlan:
    """Parse a real plan, so every test starts from a document phase one accepted."""
    return parse_design_plan(
        {
            "goals": ["Make the design phase a recorded pipeline"],
            "decisions": [
                {
                    "summary": "Create tickets mechanically from an approved plan",
                    "rationale": REASON,
                    "alternatives_rejected": [
                        {
                            "alternative": "Let a person write each ticket",
                            "why_rejected": "The plan would be advice rather than a commitment.",
                        }
                    ],
                }
            ],
            "tickets": list(drafts) if drafts else [_ticket_draft("alpha")],
        }
    )


def _outcome(plan: DesignPlan, *, captured: bool = True, entry_id: str = RECONCILIATION_ID):
    """Build the reconciliation outcome phase two hands back.

    Hand-built rather than produced by :func:`~kojutsu.core.design_capture.reconcile_design_proposals`
    for the tests that are about *this* phase: the digest is computed with the real
    function, so a test cannot accidentally agree with a wrong derivation. One test below
    drives the real reconciler, so the composed pipeline is not left untested.
    """
    return ReconciliationOutcome(
        entry_id=entry_id,
        plan=plan,
        plan_sha256=design_plan_digest(plan),
        proposal_ids=("design-proposal-v1-" + "3" * 64,),
        discarded=(),
        captured=captured,
        delivery=KnowledgeDeliveryOutcome(
            entry_id=entry_id,
            status=KnowledgeDeliveryStatus.DELIVERED,
        ),
    )


@pytest.fixture
def approvals(tmp_path: Path):
    with SqlitePlanApprovalLedger(tmp_path / "approvals.db") as opened:
        yield opened


@pytest.fixture
def sink(tmp_path: Path):
    with SqliteTicketSink(tmp_path / "tickets.db") as opened:
        yield opened


def _approve(approvals: SqlitePlanApprovalLedger, outcome: ReconciliationOutcome) -> None:
    record_plan_approval(
        reconciliation=outcome,
        approved_by=APPROVER,
        approvals=approvals,
        note="Read the reconciliation; the drafts match the decisions.",
        approved_at=APPROVED_AT,
    )


def _create(
    *,
    plan: DesignPlan,
    outcome: ReconciliationOutcome,
    sink: TicketSink,
    approvals: PlanApprovalLedger,
    gates: GateRegistry | None = None,
) -> TicketCreationOutcome:
    return create_tickets_from_plan(
        plan=plan,
        reconciliation=outcome,
        sink=sink,
        approvals=approvals,
        gates=gates,
    )


# --- identity is derived, and pinned ---------------------------------------------


def test_the_ticket_id_is_pinned() -> None:
    """The one thing that must not move, because moving it duplicates rather than breaks.

    Every stored ticket's id is a key and a lookup path. Re-deriving the same plan after a
    change produces ids that match no stored row, so the store creates a *second* ticket
    for every draft -- and the first set stays, complete and un-referenced, looking exactly
    like the work the plan asked for. The fix is a version bump and a migration, never an
    edit here.
    """
    assert stable_ticket_id(plan_sha256=PLAN_SHA, draft_id="alpha") == TICKET_ID


def test_the_domain_is_inside_the_digest_and_not_only_painted_on_the_front() -> None:
    """What makes the prefix above mean anything.

    A prefix is a label; the domain is an element of the preimage. Without this, a ticket
    and an approval sharing a preimage layout would derive one digest and be one namespace
    wearing two prefixes.
    """
    preimage = identity_preimage(TICKET_IDENTITY_DOMAIN, (PLAN_SHA, "alpha"))

    assert preimage != identity_preimage(PLAN_APPROVAL_IDENTITY_DOMAIN, (PLAN_SHA, "alpha"))
    assert preimage != identity_preimage(PLAN_GATE_DECISION_IDENTITY_DOMAIN, (PLAN_SHA, "alpha"))
    assert TICKET_IDENTITY_DOMAIN.encode() in preimage


def test_the_ticket_id_is_a_function_of_the_plan_and_the_draft_and_nothing_else() -> None:
    """The property the mechanism depends on, stated as arithmetic.

    No approver, no clock, no reconciliation id in the preimage, so a second run and a
    second approver both compute the value already stored. A re-run that included either
    would create a second copy of every ticket, which is the exact failure requirement four
    is about.
    """
    assert stable_ticket_id(plan_sha256=PLAN_SHA, draft_id="alpha") == stable_ticket_id(
        plan_sha256=PLAN_SHA, draft_id="alpha"
    )
    assert stable_ticket_id(plan_sha256=OTHER_PLAN_SHA, draft_id="alpha") != TICKET_ID
    assert stable_ticket_id(plan_sha256=PLAN_SHA, draft_id="beta") != TICKET_ID


def test_the_other_two_derivations_are_pinned_as_goldens() -> None:
    """Two more durable keys, and the same argument for pinning each of them."""
    assert stable_plan_approval_id(plan_sha256=PLAN_SHA, approved_by=APPROVER) == APPROVAL_ID
    assert (
        stable_plan_gate_decision_id(plan_sha256=PLAN_SHA, gate="plan-approval", approval_id=None)
        == DECISION_ID
    )


def test_an_id_cannot_be_derived_from_nothing() -> None:
    """A blank anchor derives an id that names every plan, or none.

    Both are worse than refusing: a ticket keyed on an empty digest would collide with every
    other such ticket, and an approval on an empty digest would approve every plan that
    failed to carry one.
    """
    with pytest.raises(ValueError, match="empty one"):
        stable_ticket_id(plan_sha256="  ", draft_id="alpha")
    with pytest.raises(ValueError, match="empty one"):
        stable_ticket_id(plan_sha256=PLAN_SHA, draft_id="  ")
    with pytest.raises(ValueError, match="approves nothing"):
        stable_plan_approval_id(plan_sha256="", approved_by=APPROVER)


# --- the derivation is mechanical -------------------------------------------------


def test_every_declared_field_reaches_the_ticket(sink: SqliteTicketSink) -> None:
    """Requirement three, asserted field by field.

    A transformation that drops the test command, or defaults the labels, or folds the
    acceptance criteria into prose produces tickets that are *well formed* and wrong in a
    way no reviewer can detect, because a ticket carries no trace of the plan that made it.
    """
    plan = _plan(
        _ticket_draft(
            "alpha",
            title="Record the approval",
            description="A later reader can see who approved and when.",
            acceptance_criteria=["Given a gate decision, then it is recorded"],
            test_command="uv run pytest -q tests/test_ticket_drafts.py",
            reference_files=["src/kojutsu/core/gates.py", "src/kojutsu/core/ticket_drafts.py"],
            labels=["planning", "provenance"],
            priority="P2",
        )
    )
    derived = derive_ticket(
        plan.tickets[0],
        plan_sha256=design_plan_digest(plan),
        reconciliation_entry_id=RECONCILIATION_ID,
    )
    assert sink.record_ticket(derived) is True

    stored = sink.get_ticket(derived.ticket_id)
    assert stored is not None
    assert stored.title == "Record the approval"
    assert stored.description == "A later reader can see who approved and when."
    assert stored.acceptance_criteria == ("Given a gate decision, then it is recorded",)
    assert stored.test_command == "uv run pytest -q tests/test_ticket_drafts.py"
    assert stored.reference_files == (
        "src/kojutsu/core/gates.py",
        "src/kojutsu/core/ticket_drafts.py",
    )
    assert stored.labels == ("planning", "provenance")
    assert stored.priority == "P2"
    assert stored.plan_sha256 == design_plan_digest(plan)
    assert stored.reconciliation_entry_id == RECONCILIATION_ID
    assert stored.draft_id == "alpha"


def test_a_dependency_is_an_edge_and_not_a_line_in_the_description(
    sink: SqliteTicketSink,
) -> None:
    """Requirement six, and the reason it is worded the way it is.

    A dependency written into a description is true only while the prose stays intact and
    can be read by nobody but a person. The edge is a row with two foreign keys, so
    ``list_dependencies`` can answer "what is startable" and the database refuses an edge
    to a ticket that does not exist.
    """
    plan = _plan(_ticket_draft("alpha"), _ticket_draft("beta", depends_on=("alpha",)))
    outcome = _outcome(plan)
    alpha_id = stable_ticket_id(plan_sha256=outcome.plan_sha256, draft_id="alpha")
    beta_id = stable_ticket_id(plan_sha256=outcome.plan_sha256, draft_id="beta")

    result = _create(plan=plan, outcome=outcome, sink=sink, approvals=_always_allow())

    assert result.created == (alpha_id, beta_id)
    assert sink.list_dependencies(beta_id) == (alpha_id,)
    assert sink.list_dependencies(alpha_id) == ()
    beta = sink.get_ticket(beta_id)
    assert beta is not None
    assert alpha_id not in beta.description, "an edge written into prose is not an edge"


def test_tickets_are_created_in_dependency_order(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """Requirement six's other half, and it is not cosmetic.

    The edge carries a foreign key, so a ticket written before its blocker would be refused
    by the database rather than silently recorded. Order is therefore a correctness
    property, not a preference -- which is why it comes from phase one's
    :func:`~kojutsu.core.design_plan.ticket_drafts_in_dependency_order` instead of from a
    sort written here.
    """
    plan = _plan(
        _ticket_draft("gamma", depends_on=("beta",)),
        _ticket_draft("beta", depends_on=("alpha",)),
        _ticket_draft("alpha"),
    )
    outcome = _outcome(plan)
    _approve(approvals, outcome)

    _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    stored = sink.list_tickets(plan_sha256=outcome.plan_sha256)
    assert [ticket.draft_id for ticket in stored] == ["alpha", "beta", "gamma"]
    assert stored[-1].dependencies == (stored[1].ticket_id,)


def test_a_cyclic_plan_is_refused_before_anything_is_written(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """Re-validated here rather than trusted, and the check is phase one's.

    The plan is built through the real parser and then edited behind its back with
    ``model_copy``, which is the only way to produce one -- and it is what a plan assembled
    by ``model_construct``, or by a caller holding a stale model, looks like. The graph
    check therefore has to run again at the boundary that spends the plan; there is one
    implementation of "a well-formed dependency graph" and this is the call site, not a
    second one.

    The refusal also comes **before** the gate is consulted: there is no point asking a
    person to approve a set of tickets that cannot be created.
    """
    valid = _plan(_ticket_draft("alpha"), _ticket_draft("beta", depends_on=("alpha",)))
    cyclic = valid.model_copy(
        update={
            "tickets": [
                valid.tickets[0].model_copy(update={"depends_on": ["beta"]}),
                valid.tickets[1],
            ]
        }
    )
    outcome = _outcome(cyclic)

    with pytest.raises(DesignPlanError, match="cycle"):
        _create(plan=cyclic, outcome=outcome, sink=sink, approvals=approvals)

    assert sink.list_tickets() == []
    assert approvals.list_decisions(plan_sha256=outcome.plan_sha256) == []


# --- nothing exists before a person approves --------------------------------------


def test_no_ticket_exists_before_the_plan_is_approved(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """Requirement one.

    No approval has been recorded, so the gate blocks and the sink is never called. The
    assertion is on the sink, not on the exception: an implementation that raised for its
    own reasons would pass a test that only checked the type.
    """
    plan = _plan()
    outcome = _outcome(plan)

    with pytest.raises(PlanApprovalRequiredError) as caught:
        _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    assert outcome.plan_sha256 in str(caught.value)
    assert sink.list_tickets() == []


def test_the_block_is_recorded_before_the_refusal(approvals: SqlitePlanApprovalLedger) -> None:
    """Requirement two, and the half that is easy to leave out.

    The decision is written *before* the block is acted on, so "this plan reached the gate
    and nobody approved it" is a row. A record written only on success would say nothing at
    all about the plans that are waiting, which is the majority of them for most of their
    lives.
    """
    plan = _plan()
    outcome = _outcome(plan)
    with (
        SqliteTicketSink(approvals.path.parent / "tickets.db") as sink,
        pytest.raises(PlanApprovalRequiredError),
    ):
        _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    decisions = approvals.list_decisions(plan_sha256=outcome.plan_sha256)
    assert len(decisions) == 1
    assert decisions[0].gate == "plan-approval"
    assert decisions[0].point is LifecyclePoint.PLAN_PROPOSED
    assert decisions[0].verdict == GateVerdict.BLOCKED.value
    assert decisions[0].approval_id is None
    assert "before it becomes tickets" in decisions[0].reason


def test_the_decision_records_what_the_gate_was_looking_at(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """A decision without its context cannot be audited, only remembered.

    The context is stored whole rather than as the columns this table happens to have, so
    the digest and the reconciliation the gate was shown are both readable afterwards.
    """
    plan = _plan()
    outcome = _outcome(plan)
    _approve(approvals, outcome)

    result = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    recorded = approvals.list_decisions(plan_sha256=outcome.plan_sha256)
    assert [decision.verdict for decision in recorded] == [GateVerdict.ALLOW.value]
    assert result.approval is not None
    assert recorded[0].approval_id == result.approval.approval_id
    assert recorded[0].context is not None
    assert recorded[0].context["plan_sha256"] == outcome.plan_sha256
    assert recorded[0].context["reconciliation_entry_id"] == RECONCILIATION_ID


def test_an_allowing_decision_is_recorded_too(approvals: SqlitePlanApprovalLedger) -> None:
    """A gate that only logs when it blocks is a gate nobody can trust for the rest.

    ``Gate.evaluate`` emits a decision whether it allows or blocks, and the ledger stores
    both. The ALLOW row is what makes "this plan was reviewed" answerable without inferring
    it from the existence of tickets.
    """
    plan = _plan()
    outcome = _outcome(plan)
    _approve(approvals, outcome)

    with SqliteTicketSink(approvals.path.parent / "tickets.db") as sink:
        _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    approvals_for_plan = approvals.list_approvals(plan_sha256=outcome.plan_sha256)
    assert len(approvals_for_plan) == 1
    assert approvals_for_plan[0].approved_by == APPROVER
    assert approvals_for_plan[0].approved_at == APPROVED_AT
    assert approvals_for_plan[0].note is not None


def test_this_module_demotes_no_gate(approvals: SqlitePlanApprovalLedger) -> None:
    """The escape hatch is the caller's registry, not a convenience buried in the spender.

    ``cli_worker.CAPTURE_POLICY`` demotes ``plan-approval`` to ``NOTIFY`` because a
    capture-only worker never proposes a plan and would otherwise be blocked forever. That
    reasoning does not transfer: this function's contract *is* that no tickets exist until a
    person says so, and a demotion here would make the repository's strongest default a
    policy nobody chose. So the gate keeps ``BLOCK``, and the other default gates are
    untouched.
    """
    gates = plan_approval_gates(approvals=approvals)
    by_point = {gate.point: gate for gate in gates}

    assert by_point[LifecyclePoint.PLAN_PROPOSED].action is GateAction.BLOCK
    assert by_point[LifecyclePoint.PLAN_PROPOSED].name == "plan-approval"
    assert by_point[LifecyclePoint.PR_MERGED].action is GateAction.BLOCK
    assert len(gates) == len(DEFAULT_GATES)


def test_a_caller_may_pass_a_relaxed_policy_and_the_outcome_says_so(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """The demotion is available, and its cost is visible in the returned outcome.

    This is ``cli_worker.CAPTURE_POLICY``'s arrangement, driven with a ledger that has never
    heard of an approval. Tickets are created and ``approval`` is ``None``, which is not a
    defect to be smoothed over: it says this set was created with no recorded person behind
    it, and a caller that did not mean that has a bug.
    """
    plan = _plan()
    outcome = _outcome(plan)

    result = _create(
        plan=plan, outcome=outcome, sink=sink, approvals=approvals, gates=_notify_policy()
    )

    assert result.approval is None
    assert result.decisions[0].verdict is GateVerdict.NOTIFIED
    assert len(result.created) == 1
    assert approvals.get_approval(plan_sha256=outcome.plan_sha256) is None


def test_a_gate_shown_no_plan_digest_blocks_rather_than_passes(approvals: Any) -> None:
    """A context missing a field must not turn the default off.

    The predicate has two ways to not-fire and only one of them means "approved". Reading
    "I could not tell which plan this is" as "nothing to object to" is a flag-shaped
    failure in the one module whose point is that the gate is not a flag.
    """
    predicate = next(
        gate.predicate
        for gate in plan_approval_gates(approvals=approvals)
        if gate.point is LifecyclePoint.PLAN_PROPOSED
    )

    assert predicate({}) is not None
    assert predicate({"plan_sha256": ""}) is not None
    assert predicate({"plan_sha256": PLAN_SHA}) is not None


# --- the approval is a record, not a flag -----------------------------------------


def test_the_approval_names_a_person_a_moment_and_a_reconciliation(
    approvals: SqlitePlanApprovalLedger,
) -> None:
    """Requirement two: provenance a later reader can act on.

    Three facts and no more -- who, when, and which reconciliation they were reading.
    Deliberately not the plan itself: the approval is about a digest, and storing the
    document beside it would invite a reader to believe the two are still the same bytes.
    """
    outcome = _outcome(_plan())

    recorded = record_plan_approval(
        reconciliation=outcome,
        approved_by=APPROVER,
        approvals=approvals,
        note="Checked the reconciliation against the repository.",
        approved_at=APPROVED_AT,
    )

    assert recorded.approved_by == APPROVER
    assert recorded.approved_at == APPROVED_AT
    assert recorded.plan_sha256 == outcome.plan_sha256
    assert recorded.reconciliation_entry_id == RECONCILIATION_ID
    assert recorded.note == "Checked the reconciliation against the repository."
    assert approvals.get_approval(plan_sha256=outcome.plan_sha256) == recorded


def test_re_approping_keeps_the_first_moment(approvals: SqlitePlanApprovalLedger) -> None:
    """First-write-wins, because the moment is the provenance being believed.

    A later re-approval with a later clock would move the answer to "when did a person
    approve this", and the answer would then depend on how many times an operator clicked.
    This is the same rule ``record_question`` follows for ``head_sha``.
    """
    outcome = _outcome(_plan())
    later = datetime(2026, 10, 4, tzinfo=UTC)

    first = record_plan_approval(
        reconciliation=outcome,
        approved_by=APPROVER,
        approvals=approvals,
        approved_at=APPROVED_AT,
    )
    second = record_plan_approval(
        reconciliation=outcome,
        approved_by=APPROVER,
        approvals=approvals,
        approved_at=later,
        note="A second look changed nothing.",
    )

    assert second == first
    assert approvals.list_approvals(plan_sha256=outcome.plan_sha256) == [first]


def test_a_second_person_approving_adds_a_record_rather_than_replacing_one(
    approvals: SqlitePlanApprovalLedger,
) -> None:
    """Two approvals are two events, and a later reader wants both.

    The approver is in the approval identity for this reason. Overwriting would leave a
    reader unable to say whether the first person disagreed, never saw it, or was recorded
    at all.
    """
    outcome = _outcome(_plan())

    first = record_plan_approval(
        reconciliation=outcome, approved_by=APPROVER, approvals=approvals, approved_at=APPROVED_AT
    )
    second = record_plan_approval(
        reconciliation=outcome,
        approved_by="sam",
        approvals=approvals,
        approved_at=datetime(2026, 10, 4, tzinfo=UTC),
    )

    assert second.approval_id != first.approval_id
    assert [
        row.approved_by for row in approvals.list_approvals(plan_sha256=outcome.plan_sha256)
    ] == [
        APPROVER,
        "sam",
    ]


@pytest.mark.parametrize(
    "approver",
    ["", "   ", "\u200b", "d" * (MAX_AGENT_ID_CHARS + 1)],
    ids=["empty", "whitespace", "zero-width-space", "too-long"],
)
def test_an_approver_that_cannot_be_believed_is_refused(
    approvals: SqlitePlanApprovalLedger, approver: str
) -> None:
    """The one field here whose entire job is to name a person.

    A blank principal is a record of an approval by nobody. The zero-width space is the
    subtler case and the reason the character is refused rather than removed: stripped, it
    would let two approvers who differ only by an invisible glyph share one approval
    identity, and one of the two records would be lost. The oversized one is the same bound
    the marker parser applies -- one definition of a principal's length, not two.
    """
    with pytest.raises(ValueError):
        record_plan_approval(
            reconciliation=_outcome(_plan()), approved_by=approver, approvals=approvals
        )

    assert approvals.list_approvals(plan_sha256=_outcome(_plan()).plan_sha256) == []


def test_an_approval_note_must_be_text_or_nothing(approvals: SqlitePlanApprovalLedger) -> None:
    """An unstated note is ``None``; anything else is a sentence someone wrote.

    ``""`` is a claim that something was written, which is why it is not the same as
    ``None`` -- and a non-string is a caller that has lost track of what it holds.
    """
    outcome = _outcome(_plan())

    assert (
        record_plan_approval(
            reconciliation=outcome, approved_by=APPROVER, approvals=approvals, note=None
        ).note
        is None
    )
    with pytest.raises(ValueError, match="must be a string"):
        record_plan_approval(
            reconciliation=outcome,
            approved_by="sam",
            approvals=approvals,
            note=17,  # type: ignore[arg-type]
        )


def test_a_migration_statement_that_is_not_re_runnable_is_refused() -> None:
    """The guard on the forward-only migration, tested directly because nothing reaches it.

    Every statement this module ships is ``CREATE ... IF NOT EXISTS``, so the guard is
    unreachable from the code as written -- and an unreachable guard is one a future
    migration will discover by tripping it. Refusing the statement at apply time is the only
    place the failure can be loud: a migration interrupted half-way is safe to resume
    precisely because every statement in it can be re-run, and that has to be checked rather
    than assumed.
    """
    from kojutsu.core import ticket_approvals

    with closing(sqlite3.connect(":memory:")) as raw:
        with pytest.raises(RuntimeError, match="not idempotent"):
            ticket_approvals._apply_statement(raw, "CREATE TABLE design_tickets (id TEXT)")
        with pytest.raises(RuntimeError, match="not a supported form"):
            ticket_approvals._apply_statement(raw, "DELETE FROM design_tickets")
        with pytest.raises(RuntimeError, match="not a column addition"):
            ticket_approvals._apply_statement(raw, "ALTER TABLE design_tickets RENAME TO other")


def test_an_oversized_note_is_refused_rather_than_truncated(
    approvals: SqlitePlanApprovalLedger,
) -> None:
    """A cut sentence argues a different thing without saying so."""
    outcome = _outcome(_plan())

    with pytest.raises(ValueError, match="refused rather than truncated"):
        record_plan_approval(
            reconciliation=outcome,
            approved_by=APPROVER,
            approvals=approvals,
            note="x" * (ticket_drafts.MAX_APPROVAL_NOTE_CHARS + 1),
        )

    assert approvals.get_approval(plan_sha256=outcome.plan_sha256) is None


# --- idempotency, pinned against the real store -----------------------------------


def test_rerunning_the_same_approval_creates_nothing(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """Requirement four, first half: the same approval, run twice."""
    plan = _plan(_ticket_draft("alpha"), _ticket_draft("beta", depends_on=("alpha",)))
    outcome = _outcome(plan)
    _approve(approvals, outcome)

    first = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)
    second = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    assert len(first.created) == 2
    assert second.created == ()
    assert second.existing == first.created
    assert len(sink.list_tickets()) == 2


def test_a_different_approval_of_the_same_plan_creates_no_duplicates(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """Requirement four, second half, and the case a request-keyed store gets wrong.

    A second approver is a genuinely different *request*, so an idempotency key computed
    from the request would not match and the whole set would be created again. The derived
    id excludes the approver, so the second run finds every row already there.
    """
    plan = _plan(_ticket_draft("alpha"))
    outcome = _outcome(plan)
    _approve(approvals, outcome)
    first = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    record_plan_approval(
        reconciliation=outcome,
        approved_by="sam",
        approvals=approvals,
        approved_at=datetime(2026, 10, 4, tzinfo=UTC),
    )
    second = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    assert second.created == ()
    assert second.existing == first.created
    assert len(sink.list_tickets()) == 1
    assert len(approvals.list_approvals(plan_sha256=outcome.plan_sha256)) == 2


def test_repeated_blocks_do_not_inflate_the_ledger(approvals: SqlitePlanApprovalLedger) -> None:
    """A decision record is a fact about a plan, not a count of attempts.

    Anchored on the plan, the gate and the approval in hand, re-evaluating produces the
    same fact. A reader counting blocks would otherwise be reading a retry loop as though
    it were a policy.
    """
    plan = _plan()
    outcome = _outcome(plan)
    with SqliteTicketSink(approvals.path.parent / "tickets.db") as sink:
        for _ in range(3):
            with pytest.raises(PlanApprovalRequiredError):
                _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    assert len(approvals.list_decisions(plan_sha256=outcome.plan_sha256)) == 1


def test_an_unrecorded_reconciliation_still_creates_the_same_tickets(
    tmp_path: Path, sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """``captured=False`` is what a correct re-run looks like, not a fault.

    ``reconcile_design_proposals`` loses its claim the second time and says the stored
    reconciliation stands. Refusing on that flag would refuse the second run of every
    correct pipeline and make idempotency impossible to exercise -- so creation proceeds and
    the derived ids make it a no-op. Driven through the real reconciler, so the flag is the
    genuine one.
    """
    plan_payload = {
        "goals": ["Make the design phase a recorded pipeline"],
        "decisions": [{"summary": "Create tickets mechanically", "rationale": REASON}],
        "tickets": [_ticket_draft("alpha")],
    }
    proposal_sink = FakeSink()
    with SqliteQuestionRegistry(tmp_path / "registry.db") as registry:
        proposal = capture_design_proposal(
            repo="org/repo",
            topic="design-phase",
            principal="proposer",
            model="model/zero",
            text="Record the proposal, then reconcile it from outside the panel.",
            registry=registry,
            sink=proposal_sink,
            proposed_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        assert proposal.captured is True
        captured = reconcile_design_proposals(
            repo="org/repo",
            topic="design-phase",
            reconciled_by="reconciler",
            model="model/one",
            proposals=[proposal.proposal],
            plan_payload=plan_payload,
            registry=registry,
            sink=proposal_sink,
            reconciled_at=datetime(2026, 10, 2, tzinfo=UTC),
        )
        repeated = reconcile_design_proposals(
            repo="org/repo",
            topic="design-phase",
            reconciled_by="reconciler",
            model="model/one",
            proposals=[proposal.proposal],
            plan_payload=plan_payload,
            registry=registry,
            sink=proposal_sink,
            reconciled_at=datetime(2026, 10, 3, tzinfo=UTC),
        )

    assert captured.captured is True
    assert repeated.captured is False
    _approve(approvals, repeated)

    result = _create(plan=repeated.plan, outcome=repeated, sink=sink, approvals=approvals)

    assert result.reconciliation_captured is False
    assert len(result.created) == 1
    assert (
        _create(plan=repeated.plan, outcome=repeated, sink=sink, approvals=approvals).created == ()
    )


# --- a changed plan is refused ----------------------------------------------------


def test_a_plan_that_changed_after_reconciliation_is_refused_before_anything_is_written(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """Requirement five, and the check a derived id cannot make for itself.

    The ids are computed from the digest, so a changed plan derives *different* ids and
    would otherwise create a second, entirely unapproved set that looks exactly like a
    first run of a new plan. Comparing the digest against the reconciliation that recorded
    it is what turns "the ids are derived" from a mechanism into a guarantee.

    Nothing is written: not a ticket, not an edge, and not a gate decision -- the refusal
    happens before the gate is consulted, because there is no plan to put in front of it.
    """
    plan = _plan(_ticket_draft("alpha", description="The original description."))
    outcome = _outcome(plan)
    _approve(approvals, outcome)
    changed = _plan(_ticket_draft("alpha", description="A different description."))

    with pytest.raises(PlanChangedError) as caught:
        _create(plan=changed, outcome=outcome, sink=sink, approvals=approvals)

    assert caught.value.presented_sha256 == design_plan_digest(changed)
    assert caught.value.reconciled_sha256 == outcome.plan_sha256
    assert sink.list_tickets() == []
    assert approvals.list_decisions(plan_sha256=outcome.plan_sha256) == []


def test_a_changed_plan_finds_no_approval_because_the_approval_is_the_digest(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """The second line of defence, and the one that survives a lying outcome.

    A caller who hands in a reconciliation whose digest matches a *changed* plan has
    defeated the comparison above -- by construction, the two are then the same string. What
    they cannot do is reach the approval, because the approval is anchored on the digest
    the person actually signed off. So the plan that was changed after approval is still
    refused, one step later and for a different reason.
    """
    approved_plan = _plan(_ticket_draft("alpha"))
    approved_outcome = _outcome(approved_plan)
    _approve(approvals, approved_outcome)

    changed = _plan(_ticket_draft("alpha", description="A different description."))
    forged = ReconciliationOutcome(
        entry_id=RECONCILIATION_ID,
        plan=changed,
        plan_sha256=design_plan_digest(changed),
        proposal_ids=(),
        discarded=(),
        captured=True,
        delivery=None,
    )

    with pytest.raises(PlanApprovalRequiredError):
        _create(plan=changed, outcome=forged, sink=sink, approvals=approvals)

    assert sink.list_tickets() == []


# --- a partial set is resumable ----------------------------------------------------


class FlakySink:
    """A real store that stops working after ``fail_after`` writes, and counts the calls."""

    def __init__(self, inner: SqliteTicketSink, *, fail_after: int) -> None:
        self.inner = inner
        self.fail_after = fail_after
        self.attempts = 0

    def record_ticket(self, ticket: DerivedTicket) -> bool:
        self.attempts += 1
        if self.attempts > self.fail_after:
            raise RuntimeError("the store is unavailable")
        return self.inner.record_ticket(ticket)

    def get_ticket(self, ticket_id: str):
        return self.inner.get_ticket(ticket_id)

    def list_tickets(self, *, plan_sha256: str | None = None, limit: int | None = None):
        if limit is None:
            return self.inner.list_tickets(plan_sha256=plan_sha256)
        return self.inner.list_tickets(plan_sha256=plan_sha256, limit=limit)

    def list_dependencies(self, ticket_id: str) -> tuple[str, ...]:
        return self.inner.list_dependencies(ticket_id)


def test_a_failed_run_keeps_what_it_created_and_the_next_run_finishes_the_set(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """The answer to "what happens to a partially-created set": it stays, and it completes.

    Each ticket is its own committed row, so the two that were written are durable and keep
    their ids. Because those ids are derived from the plan digest, the next run creates the
    missing three and creates nothing new -- so the failure is resumable rather than needing
    a rollback nobody implemented. The exception names both halves, because "some tickets
    exist" is the least useful sentence available to whoever reads the log.
    """
    plan = _plan(
        _ticket_draft("alpha"),
        _ticket_draft("beta", depends_on=("alpha",)),
        _ticket_draft("gamma", depends_on=("beta",)),
        _ticket_draft("delta", depends_on=("gamma",)),
        _ticket_draft("epsilon"),
    )
    outcome = _outcome(plan)
    _approve(approvals, outcome)
    flaky = FlakySink(sink, fail_after=2)

    with pytest.raises(TicketSetIncompleteError) as caught:
        _create(plan=plan, outcome=outcome, sink=flaky, approvals=approvals)

    assert len(caught.value.created) == 2
    assert len(caught.value.pending) == 3
    assert stable_ticket_id(plan_sha256=outcome.plan_sha256, draft_id="alpha") in str(caught.value)
    assert [ticket.draft_id for ticket in sink.list_tickets()] == ["alpha", "beta"]

    resumed = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    assert len(resumed.created) == 3
    assert resumed.existing == caught.value.created
    assert [ticket.draft_id for ticket in sink.list_tickets()] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
    ]


def test_edges_are_reasserted_for_a_ticket_that_already_exists(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """A crash between the ticket and its edges must not be observable.

    Tickets and edges are written in one transaction, but a store that was edited underneath
    us can still lose one. Re-asserting the edges on an existing ticket is what keeps a
    ticket that looks startable from being one.
    """
    plan = _plan(_ticket_draft("alpha"), _ticket_draft("beta", depends_on=("alpha",)))
    outcome = _outcome(plan)
    _approve(approvals, outcome)
    _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)
    beta_id = stable_ticket_id(plan_sha256=outcome.plan_sha256, draft_id="beta")

    with closing(sqlite3.connect(sink.path)) as raw:
        raw.execute(
            "DELETE FROM design_ticket_edges WHERE ticket_id = ?",
            (beta_id,),
        )
        raw.commit()
    assert sink.list_dependencies(beta_id) == ()

    _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    assert sink.list_dependencies(beta_id) == (
        stable_ticket_id(plan_sha256=outcome.plan_sha256, draft_id="alpha"),
    )


# --- the character policy runs where text becomes stored ---------------------------


def test_an_invisible_character_is_removed_and_recorded(
    sink: SqliteTicketSink, approvals: SqlitePlanApprovalLedger
) -> None:
    """The policy is applied at the storage boundary, and says what it took out.

    The plan accepted this description: its check is ``CONTROL_RE``, and a
    right-to-left override is not a control character, it is a direction change that every
    renderer applies faithfully. That is why the check belongs here and not in the
    derivation -- and why the removal is recorded under
    :data:`~kojutsu.core.text_hygiene.SANITISATION_KEY` rather than performed silently.
    """
    plan = _plan(_ticket_draft("alpha", description=f"Do the thing. {RLO}gnp.exe"))
    outcome = _outcome(plan)
    _approve(approvals, outcome)
    result = _create(plan=plan, outcome=outcome, sink=sink, approvals=approvals)

    stored = sink.get_ticket(result.created[0])
    assert stored is not None
    assert RLO not in stored.description
    assert stored.text_sanitisation is not None
    assert "U+202E" in stored.text_sanitisation
    # Under the same key name every other stored record uses, so one query finds them all.
    assert stored.text_sanitisation is not None
    with closing(sqlite3.connect(sink.path)) as raw:
        columns = {str(row[1]) for row in raw.execute("PRAGMA table_info(design_tickets)")}
    assert SANITISATION_KEY in columns


def test_a_ticket_that_sanitises_to_nothing_is_refused(sink: SqliteTicketSink) -> None:
    """The case the plan's own bounds cannot catch.

    An invisible character is a perfectly legal non-empty string, so a title made of one
    passes every bound the schema applied and sanitises to the empty string. Storing that
    would be a ticket whose title is nothing, in a store where every other title is
    visible -- so it is refused rather than written as an unreadable row.
    """
    plan = _plan(_ticket_draft("alpha", title=RLO))
    derived = derive_ticket(
        plan.tickets[0],
        plan_sha256=design_plan_digest(plan),
        reconciliation_entry_id=RECONCILIATION_ID,
    )

    with pytest.raises(ValueError, match="reads as nothing"):
        sink.record_ticket(derived)

    assert sink.list_tickets() == []


# --- the store refuses what it cannot vouch for ------------------------------------


def test_a_ticket_row_that_is_someone_elses_is_not_reported_as_a_rerun(
    sink: SqliteTicketSink,
) -> None:
    """Returning ``False`` for a *collision* would report a silent overwrite as a no-op.

    That is the failure a derived id cannot detect on its own: it protects re-running, not
    the case where two different things derive the same value. The anchor is read back and
    compared, so the second one is refused rather than merged.
    """
    plan = _plan(_ticket_draft("alpha"))
    ticket_id = stable_ticket_id(plan_sha256=PLAN_SHA, draft_id="alpha")
    foreign = DerivedTicket(
        ticket_id=ticket_id,
        plan_sha256=OTHER_PLAN_SHA,
        reconciliation_entry_id=RECONCILIATION_ID,
        draft_id="alpha",
        title="A different plan's ticket",
        description="Something else entirely.",
        acceptance_criteria=("Nothing to do with the above",),
        priority="P1",
    )
    assert sink.record_ticket(foreign) is True

    mine = derive_ticket(
        plan.tickets[0],
        plan_sha256=PLAN_SHA,
        reconciliation_entry_id=RECONCILIATION_ID,
    )

    with pytest.raises(RuntimeError, match="is already on file"):
        sink.record_ticket(mine)


def test_a_rewritten_migration_history_is_refused_on_the_next_open(tmp_path: Path) -> None:
    """A checksum only means something if the stored history is compared every time.

    Not only when migrating: a store migrated by this application and edited by another
    should say so at open, where an operator is present, rather than at the first write that
    hits a column which is no longer there.
    """
    path = tmp_path / "approvals.db"
    with SqlitePlanApprovalLedger(path):
        pass
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("UPDATE design_store_migrations SET checksum = 'tampered'")
        raw.commit()

    with pytest.raises(RuntimeError, match="does not match"):
        SqlitePlanApprovalLedger(path)


def test_two_stores_on_one_file_are_refused_loudly(tmp_path: Path) -> None:
    """One SQLite file, one owner of ``PRAGMA user_version``.

    Sharing a file would mean each store reading the other's version and then migrating
    against tables it does not know. It is caught by the ledger checksum rather than by the
    table check, and it is caught *because* each store owns a different version-1 statement
    tuple: with one shared tuple, version 1 would create all four tables and either store
    would open either file without complaint. The message names which store was being
    opened, so an operator fixing two paths onto one file is told which half is unhappy.
    """
    path = tmp_path / "shared.db"
    with SqlitePlanApprovalLedger(path):
        pass

    with pytest.raises(RuntimeError, match="Ticket store migration 1 was recorded"):
        SqliteTicketSink(path)


def test_a_missing_column_is_refused_rather_than_read_back_as_nothing(tmp_path: Path) -> None:
    """The read side fails at open, not with a ``KeyError`` on one row.

    The expected column list is a second copy of the DDL, and the drift it catches always
    points the safe way: a schema this application does not recognise is refused, not
    guessed at.
    """
    path = tmp_path / "tickets.db"
    with SqliteTicketSink(path):
        pass
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("ALTER TABLE design_tickets DROP COLUMN labels")
        raw.commit()

    with pytest.raises(RuntimeError, match="has no column 'labels'"):
        SqliteTicketSink(path)


def test_reads_are_bounded_rather_than_refused(sink: SqliteTicketSink) -> None:
    """A limit is a hint about how much to read, so the worst a bad one does is read less."""
    assert sink.list_tickets(limit=-5) == []
    with pytest.raises(ValueError, match="must be an integer"):
        sink.list_tickets(limit="10")  # type: ignore[arg-type]


# --- helpers that make the tests above readable -----------------------------------


class PermissiveLedger:
    """Approves every digest, for the tests that are about the transformation itself.

    A real :class:`SqlitePlanApprovalLedger` is used everywhere the gate is what is under
    test. This one auto-approves so that "does every declared field arrive" is not answered
    through a gate at all -- while still recording decisions, so a test using it exercises
    the same call sequence the real ledger does.
    """

    def __init__(self) -> None:
        self.decisions: list[RecordedDecision] = []

    def get_approval(self, *, plan_sha256: str) -> PlanApproval:
        return PlanApproval(
            approval_id=stable_plan_approval_id(plan_sha256=plan_sha256, approved_by="test"),
            plan_sha256=plan_sha256,
            reconciliation_entry_id=RECONCILIATION_ID,
            approved_by="test",
            approved_at=APPROVED_AT,
        )

    def record_approval(self, **kwargs: Any) -> PlanApproval:
        return self.get_approval(plan_sha256=str(kwargs["plan_sha256"]))

    def list_approvals(self, *, plan_sha256: str, limit: int = 100) -> list[PlanApproval]:
        return [self.get_approval(plan_sha256=plan_sha256)]

    def record_decision(
        self, *, decision: Any, plan_sha256: str, **kwargs: Any
    ) -> RecordedDecision:
        recorded = RecordedDecision(
            decision_id=stable_plan_gate_decision_id(
                plan_sha256=plan_sha256, gate=decision.gate, approval_id=kwargs.get("approval_id")
            ),
            plan_sha256=plan_sha256,
            gate=decision.gate,
            point=decision.point,
            verdict=str(decision.verdict),
            reason=decision.reason,
            decided_at=APPROVED_AT,
            approval_id=kwargs.get("approval_id"),
            context=dict(decision.context),
        )
        self.decisions.append(recorded)
        return recorded

    def list_decisions(self, *, plan_sha256: str, limit: int = 100) -> list[RecordedDecision]:
        return [recorded for recorded in self.decisions if recorded.plan_sha256 == plan_sha256]


def _always_allow() -> PermissiveLedger:
    return PermissiveLedger()


def _notify_policy() -> GateRegistry:
    """``cli_worker``'s demotion, built here rather than imported as a worker policy."""
    return GateRegistry(
        tuple(
            gate
            if gate.point is not LifecyclePoint.PLAN_PROPOSED
            else Gate(
                name=gate.name,
                point=gate.point,
                action=GateAction.NOTIFY,
                predicate=gate.predicate,
            )
            for gate in DEFAULT_GATES
        )
    )
