"""The last step of the design phase: an approved plan becomes tickets, mechanically.

Phases one and two made a plan a **document** (:mod:`kojutsu.core.design_plan`) and then
made the reconciliation around it a **record** (:mod:`kojutsu.core.design_capture`).
This module is the step that spends them: it asks the gate whether a person approved
the plan, records the answer, and writes one ticket per draft. Its whole argument is
the word *mechanically*. The ticket says a person reviews the reconciliation **once**
"rather than approving every ticket it produces", so every ticket in the set is created
by a person who never read it. That inverts the usual risk: a bug in this file does not
produce one bad ticket, it produces a hundred plausible ones, none of which carries any
trace of the plan that produced it. So the properties defended below are all about
refusing, and about what can be re-derived from a stored row.

**Ticket identity is derived, not transmitted, and there is no other option here.**
An idempotency key that a caller transmits protects against exactly one thing: the same
call arriving twice. It has to be computed by the caller from the request, stored by the
server, and compared by the server. This repository has no ticket API to receive one --
``GitHubClient`` has a single POST and it posts issue comments,
``kojutsu.integrations.jira.JiraClient`` is read-only, and
:mod:`kojutsu.worker.sources` names the ticket system as the
:func:`~kojutsu.worker.loop.WorkSource` that is not implemented yet. So this module
defines the seam (:class:`TicketSink`) and ships the local implementation, and its
idempotency comes from the *identity* rather than from a deduplicated write: a ticket's
id is a digest of the plan's digest and the draft id, so the second run of the same
approval computes the same ids, the primary key refuses them, and nothing is created.

What a derived id buys, precisely:

- Re-running the same approval creates nothing new, because the ids are the same.
- A *different* approval of the *same* plan also creates nothing new, which a
  "have I seen this request" key keyed on the caller or the call would not give: a
  second person approving the same plan is a genuinely different request, and it must
  still not produce a second copy of every ticket.
- A crash midway is resumable. Each ticket is its own committed row, so the run after a
  failure completes the set instead of starting it again, and the answer to "what
  happens to a partial set" is "it stays, and it is finished by re-running" rather than
  a rollback nobody implemented.

What a derived id does **not** buy, and this is the limit worth stating in the record
and not only here: it makes re-running safe, and it cannot make a *different* plan's
ticket collide safely. A digest collision across two plans would mean one plan's draft
silently became another's ticket, and no amount of hashing prevents a SHA-256 collision
-- what prevents it is that a different plan derives a different digest and therefore
different ids. That argument needs the plan digest to be **honest**, and nothing local
can make it so, because the digest is computed from a document a caller supplies. So the
digest is checked against the reconciliation that recorded it
(:class:`PlanChangedError`), the approval is anchored on that digest rather than on a
plan object, and the gate asks the ledger whether *that digest* was approved. A plan
edited after approval is a different plan, finds no approval, and is refused -- which is
the property the derived id alone does not provide.

**Two stores, not one, and the separation is the point.** A component that records the
human approval *and* creates the tickets that approval authorises can authorise itself,
in one transaction, in one object, and the record it leaves is indistinguishable from
one a person made. The ticket asks for "a record with provenance, not a flag in memory",
and the cheapest way to get provenance rather than a flag is for the thing that writes
the authorisation not to be the thing that spends it. So :class:`PlanApprovalLedger` and
:class:`TicketSink` are separate protocols over separate files, and the test suite wires
a ledger that can refuse against a sink that would happily create.

**The gate is driven, not invented.** ``plan-approval`` already exists at
:data:`~kojutsu.core.gates.LifecyclePoint.PLAN_PROPOSED` with
:data:`~kojutsu.core.gates.GateAction.BLOCK` and a predicate that always fires, and it is
one of the two default blocking gates. So this module's job is to *bind that predicate to
the evidence* -- to the recorded approval -- and to honour the verdict. The action stays
``BLOCK``: a plan with no approval is refused, and a caller who wants tickets without one
passes a registry whose policy says so, which is the same escape hatch
:data:`kojutsu.cli_worker.CAPTURE_POLICY` uses for the capture-only worker. Nothing here
demotes anything; see :func:`plan_approval_gates` for why that would be the wrong way to
be convenient.

**Every decision is recorded, including the ones that let work through.** The gate emits
a :class:`~kojutsu.core.gates.GateDecision` whether it blocks or allows
(:meth:`kojutsu.core.gates.Gate.evaluate`), and this module writes every one of them
*before* it acts on the block -- so "no tickets exist because the gate said so, and here
is the gate's sentence and the moment it said it" is a query, not a reconstruction. The
approval record is the other half: a row naming who approved which plan digest, when, and
against which reconciliation.

**Edges are edges.** A dependency is stored in its own table keyed on two derived ticket
ids, never as a line of prose inside a description. A description is read by a person
deciding whether to work a ticket; a dependency is read by code deciding what is
startable, and a string a model wrote is not a relation a scheduler can follow.

**The plan is not sanitised here.** :class:`DerivedTicket` is a pure, field-for-field
copy of the draft, and the character policy runs in the store, because the character
policy is a property of the *destination*: this is the boundary where untrusted prose
becomes stored prose, which is where :mod:`kojutsu.core.text_hygiene` says it belongs.
Doing it in the transformation instead would mean the derived ticket and the stored
ticket were different documents, and the mechanical claim is that they are the same one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kojutsu.core.design_capture import ReconciliationOutcome, design_plan_digest
from kojutsu.core.design_plan import DesignPlan, TicketDraft, ticket_drafts_in_dependency_order
from kojutsu.core.gates import (
    GateDecision,
    GateRegistry,
    LifecyclePoint,
)
from kojutsu.core.text_hygiene import describe_removals, sanitise
from kojutsu.core.ticket_approvals import (
    MAX_APPROVAL_LIST_LIMIT,
    MAX_APPROVAL_NOTE_CHARS,
    PLAN_APPROVAL_IDENTITY_DOMAIN,
    PLAN_APPROVAL_IDENTITY_VERSION,
    PLAN_GATE_DECISION_IDENTITY_DOMAIN,
    PLAN_GATE_DECISION_IDENTITY_VERSION,
    SCHEMA_VERSION,
    PlanApproval,
    PlanApprovalLedger,
    RecordedDecision,
    SqlitePlanApprovalLedger,
    _bounded_limit,
    _now,
    _open_store,
    _parse_timestamp,
    plan_approval_gates,
    record_plan_approval,
    stable_plan_approval_id,
    stable_plan_gate_decision_id,
)
from kojutsu.identity import identity_preimage

#: Version of the ticket identity derivation. Bump this and keep both derivations, never
#: edit :func:`stable_ticket_id` in place, for the reason
#: :data:`kojutsu.core.question_registry.RATIONALE_IDENTITY_VERSION` states: the id is a
#: primary key in a durable store, so moving the computation re-identifies every ticket
#: already created and orphans the documents keyed on it -- while a run after the change
#: creates a second copy of every one of them, because the new id matches no stored row.
TICKET_IDENTITY_VERSION = 1

#: Domain separating a ticket created from a design plan from every other namespace.
#: Part of the hashed preimage rather than a prefix painted on the digest, so a change to
#: the encoding produces a different value instead of one that looks comparable and is not
#: -- see :func:`kojutsu.identity.identity_preimage`.
TICKET_IDENTITY_DOMAIN = "kojutsu.design_ticket.v1"
#: Upper bound on one read page of tickets or approvals, so an operator page cannot ask a
#: local store for an unbounded answer. The same argument as
#: :data:`kojutsu.core.question_registry.MAX_QUESTION_LIST_LIMIT`, and the same honest
#: limit: a plan holding more drafts than this could have some missed by a read. A plan
#: past :data:`kojutsu.core.design_plan.MAX_PLAN_TICKETS` is refused at the gate anyway.
MAX_TICKET_LIST_LIMIT = 1_000
#: Longest one ticket field as stored, matching the plan's bound rather than inventing
#: another. The sink is not the authority on how long a title may be; the plan is, and it
#: already refused anything longer. This exists so a row cannot be written by another
#: path with an unbounded string in it.
MAX_STORED_TICKET_FIELD_CHARS = 8_000


class PlanChangedError(ValueError):
    """The plan in hand is not the plan the reconciliation recorded.

    A ``ValueError`` because that is what this codebase raises for input it will not act
    on, and because the fault really is in an argument: the caller handed a plan object to
    a function whose contract is "the plan that ``ReconciliationOutcome`` describes". The
    alternative -- proceeding, and creating tickets whose ids are derived from a digest
    the ledger never saw -- produces a set nothing in the store can be traced back to,
    under an approval that was given for a different document.

    Both digests are on the exception and both are in the message. A reader who has two
    of these failures and needs to tell them apart cannot get there from a message saying
    "digest mismatch": the whole question is *which* two.
    """

    def __init__(self, *, presented_sha256: str, reconciled_sha256: str) -> None:
        self.presented_sha256 = presented_sha256
        self.reconciled_sha256 = reconciled_sha256
        super().__init__(
            "Refusing to create tickets: the plan handed to ticket creation digests to "
            f"{presented_sha256}, and the reconciliation recorded {reconciled_sha256}. "
            "A plan that changed after it was reconciled is a different plan, and "
            "creating tickets for it under an approval given for the other one is exactly "
            "the silent garbage this phase refuses. Reconcile the changed plan at the next "
            "revision to have its own approval and its own tickets."
        )


class PlanApprovalRequiredError(RuntimeError):
    """The gate blocked, so no ticket was created and the plan stays unapproved.

    A ``RuntimeError`` rather than a ``ValueError``, and the distinction is deliberate:
    nothing about the arguments is wrong. The plan parses, its digest matches the
    reconciliation, and its drafts are a well-formed graph -- the *world* simply has no
    recorded human approval for that digest yet. Raising a value error would send a
    caller looking for a malformed plan, and there is none.

    The decisions are carried, because the gate's own sentence is the useful part: it
    names the plan digest that is unapproved, and the caller who goes and gets the
    approval needs to know which plan to approve.
    """

    def __init__(self, decisions: Sequence[GateDecision], *, plan_sha256: str) -> None:
        self.decisions: tuple[GateDecision, ...] = tuple(decisions)
        blocking = [decision for decision in self.decisions if decision.blocks]
        said = "\n".join(
            f"  - {decision.gate} at {decision.point}: {decision.reason}" for decision in blocking
        )
        super().__init__(
            f"Refusing to create tickets: the gate blocked plan {plan_sha256} "
            f"({len(blocking)} blocking decision(s)).\n{said}\n"
            "The decision was recorded before this refusal, so the ledger shows that this "
            "plan reached the gate and was not approved."
        )


class TicketSetIncompleteError(RuntimeError):
    """Creation stopped partway; the tickets already written stay, and a re-run finishes.

    Raised rather than rolled back because there is nothing to roll back *to*: each
    ticket is its own committed row, so a failure at the fortieth draft leaves the first
    thirty-nine durable. The alternative -- one transaction around the whole set -- would
    hold the store's write lock for the length of a hundred inserts and buy one property
    this design already has another way to get: the ids are derived, so re-running
    completes the set rather than duplicating it.

    ``created`` and ``pending`` are on the exception and in the message. "Some tickets
    exist" is the least useful sentence available; a log line that names which ones and
    which are missing is the difference between a resumable run and an unexplained
    half-built backlog.
    """

    def __init__(
        self, *, created: Sequence[str], pending: Sequence[str], cause: BaseException
    ) -> None:
        self.created: tuple[str, ...] = tuple(created)
        self.pending: tuple[str, ...] = tuple(pending)
        super().__init__(
            f"Ticket creation stopped after {len(self.created)} ticket(s): {cause!r}. "
            f"Created: {', '.join(self.created) or 'none'}. "
            f"Not created: {', '.join(self.pending)}. "
            "The created tickets are durable and keep their ids, which are derived from the "
            "plan digest, so re-running this approval creates nothing new and finishes the "
            "set."
        )


@dataclass(frozen=True)
class DerivedTicket:
    """One ticket to create: a draft's fields, unchanged, under a derived id.

    A pure, field-for-field copy of one :class:`~kojutsu.core.design_plan.TicketDraft`.
    Every field the plan declared is present, and nothing has been defaulted: the plan
    made ``priority`` required and ``labels`` optional on purpose, and filling either in
    here would be a second, silent source of triage judgements that no plan ever made.

    ``depends_on`` holds **derived ticket ids**, not draft ids, because that is what the
    store relates. Translating here rather than in the store keeps the one place that
    knows how a draft id becomes a ticket id (:func:`stable_ticket_id`) the one place
    that does it.

    ``text_sanitisation`` is absent rather than empty, and lives on the *row* rather than
    here, for the reason :data:`kojutsu.core.text_hygiene.SANITISATION_KEY` documents: a
    note belongs to the record as stored, so a derived ticket that has not been stored
    cannot carry one.
    """

    ticket_id: str
    plan_sha256: str
    reconciliation_entry_id: str
    draft_id: str
    title: str
    description: str
    acceptance_criteria: tuple[str, ...]
    priority: str
    test_command: str | None = None
    reference_files: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredTicket:
    """One ticket as the store holds it, including its dependency edges.

    A dataclass rather than the ``dict[str, Any]`` :meth:`SqliteQuestionRegistry.list_rationales`
    returns, and the difference is what mypy can check. A dict read makes every consumer
    re-parse the same row shape and turns a renamed column into a ``KeyError`` in
    production; an attribute read is a compile error. The list fields are tuples because
    they came back out of JSON and because nothing here mutates a stored ticket.
    """

    ticket_id: str
    plan_sha256: str
    reconciliation_entry_id: str
    draft_id: str
    title: str
    description: str
    acceptance_criteria: tuple[str, ...]
    priority: str
    created_at: datetime
    test_command: str | None = None
    reference_files: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    text_sanitisation: str | None = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class TicketDependency:
    """One edge, as the caller is told about it: this ticket waits for that one."""

    ticket_id: str
    depends_on: str


@dataclass(frozen=True)
class TicketCreationOutcome:
    """What one approval produced: which tickets were new, which already existed.

    ``created`` and ``existing`` are both reported because "nothing happened" has two
    causes that a caller must be able to tell apart -- a run that had already been done,
    and a run that completed the set a failed run left half-built. The second is a
    success and the first is a no-op, and the tickets left by the failed run are visible
    in ``existing``.

    ``approval`` is ``None`` only when the caller supplied a registry whose policy let the
    plan through without one -- the demoted-policy case :data:`kojutsu.cli_worker.CAPTURE_POLICY`
    arranges for a worker that proposes no plans. It is not ``None`` under the default
    policy, and an outcome with ``approval=None`` is therefore a thing to notice.

    ``reconciliation_captured`` carries :attr:`~kojutsu.core.design_capture.ReconciliationOutcome.captured`
    forward rather than being acted on. ``False`` means the previously stored
    reconciliation stands: the ordinary outcome of a re-run, not a fault, and refusing on
    it would make idempotency impossible to exercise.
    """

    plan_sha256: str
    reconciliation_entry_id: str
    reconciliation_captured: bool
    created: tuple[str, ...]
    existing: tuple[str, ...]
    dependencies: tuple[TicketDependency, ...]
    decisions: tuple[GateDecision, ...]
    approval: PlanApproval | None = None


@runtime_checkable
class TicketSink(Protocol):
    """Where a derived ticket goes, and how a re-run is recognised as a re-run.

    Following :class:`kojutsu.worker.loop.WorkSource`: a Protocol with one local
    implementation, so the loop that uses it can be exercised without a remote system.

    :meth:`record_ticket` returning ``True`` for a new row and ``False`` for an existing
    one is the whole idempotency contract. The caller computes the id and asks the store
    to honour it; the store does not derive anything, so a network adapter written later
    can choose its own server-side identity without this module's meaning changing.

    **Edges are written by the same call as the ticket.** A crash between "ticket exists"
    and "its dependencies are recorded" would otherwise leave a ticket that looks
    startable and is not, with nothing in the store to say its dependencies were never
    recorded. Recording them together means a partially-written ticket cannot be observed,
    and a re-run re-asserts the edges for a ticket that already exists.
    """

    def record_ticket(self, ticket: DerivedTicket) -> bool: ...

    def get_ticket(self, ticket_id: str) -> StoredTicket | None: ...

    def list_tickets(
        self, *, plan_sha256: str | None = None, limit: int = MAX_TICKET_LIST_LIMIT
    ) -> list[StoredTicket]: ...

    def list_dependencies(self, ticket_id: str) -> tuple[str, ...]: ...


def stable_ticket_id(*, plan_sha256: str, draft_id: str) -> str:
    """Return the durable identity of one ticket: this plan's digest plus this draft.

    **Derived rather than transmitted, and the derivation is the whole mechanism.** An
    idempotency key has to be computed by the caller, sent to whoever will deduplicate,
    and compared there -- and there is nobody to send it to. Deriving the *identity* of
    the record instead means the deduplication is a primary key, which is the one
    mechanism that needs no cooperation from the destination. So a second run of the same
    approval, and a run by a different approver of the same plan, both compute this value,
    the store refuses it, and nothing is created.

    The digest excludes the plan's text beyond ``plan_sha256`` -- which is itself the
    digest of the text, so the text is in there -- and excludes the reconciliation id, the
    approver, and the clock. Every one of those exclusions is deliberate:

    - *The approver* is excluded so that a second person approving the same plan yields
      the same ticket ids. That is requirement four's second half, and it is the case a
      caller-keyed key gets wrong: the second approval is a different request, so a
      request-keyed store would happily create a second copy of the whole set.
    - *The clock* is excluded so a re-run is the same write. Including it would make every
      attempt a new ticket and turn idempotency into a coin toss.
    - *The reconciliation id* is excluded so that revision 2 of a reconciliation producing
      the **same** plan does not create a second set. The plan digest already changes when
      the plan changes, which is the only case that should produce more tickets.

    What this cannot do is make a different plan's ticket collide *safely*, and that is
    stated rather than implied: a collision would be one plan's draft becoming another's
    ticket, and the defence is not a longer digest, it is that a different plan derives a
    different ``plan_sha256`` and so different ids. Which is only true if the digest is
    compared against the reconciliation that recorded it -- :func:`create_tickets_from_plan`
    does that, and a derived id is not a substitute for it.
    """
    if not plan_sha256.strip():
        raise ValueError(
            "a ticket id is derived from the plan's digest; an empty one names nothing"
        )
    if not draft_id.strip():
        raise ValueError("a ticket id is derived from a draft id; an empty one names nothing")
    return (
        f"design-ticket-v{TICKET_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(TICKET_IDENTITY_DOMAIN, (plan_sha256, draft_id))
        ).hexdigest()
    )


def derive_ticket(
    draft: TicketDraft, *, plan_sha256: str, reconciliation_entry_id: str
) -> DerivedTicket:
    """Return the ticket one draft declares, under its derived id.

    Field for field, in the order :mod:`kojutsu.core.design_plan` spells them, and that is
    the whole point of this function being separate from creation. The plan names its
    fields exactly as the ticket store spells them -- see that module's docstring on why a
    renaming transformation is a place where the plan and the tickets it produced can
    disagree with nothing to compare them -- so the transformation is a copy with an id
    attached. Exposed rather than kept private so "the transformation is mechanical" is a
    claim a caller can check without writing anything.

    ``depends_on`` is translated from draft ids to ticket ids here because this is the one
    place that knows how a draft id becomes a ticket id. Doing it in the store would mean
    two implementations of that derivation, which is how one of them ends up being the one
    that ships.
    """
    return DerivedTicket(
        ticket_id=stable_ticket_id(plan_sha256=plan_sha256, draft_id=draft.id),
        plan_sha256=plan_sha256,
        reconciliation_entry_id=reconciliation_entry_id,
        draft_id=draft.id,
        title=draft.title,
        description=draft.description,
        acceptance_criteria=tuple(draft.acceptance_criteria),
        priority=draft.priority,
        test_command=draft.test_command,
        reference_files=tuple(draft.reference_files),
        labels=tuple(draft.labels),
        depends_on=tuple(
            stable_ticket_id(plan_sha256=plan_sha256, draft_id=dependency)
            for dependency in draft.depends_on
        ),
    )


def create_tickets_from_plan(
    *,
    plan: DesignPlan,
    reconciliation: ReconciliationOutcome,
    sink: TicketSink,
    approvals: PlanApprovalLedger,
    gates: GateRegistry | None = None,
) -> TicketCreationOutcome:
    """Create one ticket per draft of an **approved** plan, or create nothing.

    The order of the four steps is the design, and each position is load-bearing.

    **1. Compare the digest before anything else.** A plan whose
    :func:`~kojutsu.core.design_capture.design_plan_digest` is not the reconciliation's
    :attr:`~kojutsu.core.design_capture.ReconciliationOutcome.plan_sha256` is refused
    here, before the gate is consulted and before a row exists. This is the check a
    derived ticket id cannot make for itself: the ids are computed from the digest, so a
    changed plan would derive *different* ids and cheerfully create a second, unapproved
    set, which looks exactly like a first run of a new plan. Anchoring the ids on the digest
    is only half of that; *checking* the digest against the record is what makes the other
    half true. See :class:`PlanChangedError`.

    **2. Order the drafts, re-validating as it goes.**
    :func:`~kojutsu.core.design_plan.ticket_drafts_in_dependency_order` re-runs the graph
    check rather than trusting the caller, so a plan built with ``model_construct`` -- which
    skips every validator -- is still refused for a cycle or a dangling edge here. That is
    why it is used rather than a sort written here: one implementation of "a well-formed
    dependency graph" and no path that skips it.

    **3. Evaluate the gate, record every decision, and only then act on the block.** The
    decisions are recorded *before* the refusal, so "this plan reached the gate and was not
    approved" is a row rather than an inference from the absence of tickets. Recording the
    block and honouring it are then the same mechanism rather than two: the block *is* the
    record, and the refusal is only what the caller does after reading it.

    **4. Write the tickets in dependency order, one committed row each.** A blocker
    therefore exists before the ticket that waits for it, which is what lets the edge
    between them be a foreign key rather than a hope.

    On partial failure: the tickets already written stay, and
    :class:`TicketSetIncompleteError` names both halves. Nothing is rolled back, because
    each ticket is its own transaction and the ids are derived -- re-running finishes the
    set rather than duplicating it, so an all-or-nothing transaction would buy nothing
    this design does not already have and would hold the write lock across the whole set.

    ``gates`` defaults to the default policy bound to ``approvals``, which blocks. Passing
    a registry is how a caller relaxes that; nothing here does it for them.
    """
    presented = design_plan_digest(plan)
    if presented != reconciliation.plan_sha256:
        raise PlanChangedError(
            presented_sha256=presented, reconciled_sha256=reconciliation.plan_sha256
        )

    ordered = ticket_drafts_in_dependency_order(plan)
    registry = (
        gates if gates is not None else GateRegistry(plan_approval_gates(approvals=approvals))
    )

    approval = approvals.get_approval(plan_sha256=presented)
    decisions = registry.evaluate(
        LifecyclePoint.PLAN_PROPOSED,
        {
            "plan_sha256": presented,
            "reconciliation_entry_id": reconciliation.entry_id,
            "reconciliation_captured": reconciliation.captured,
        },
    )
    # Recorded before the refusal, including for a decision that blocks: the block is the
    # evidence, and a record written only on success says nothing about the plans that
    # waited for an approval.
    for decision in decisions:
        approvals.record_decision(
            decision=decision,
            plan_sha256=presented,
            approval_id=None if approval is None else approval.approval_id,
        )
    if GateRegistry.is_blocked(decisions):
        raise PlanApprovalRequiredError(decisions, plan_sha256=presented)

    created: list[str] = []
    existing: list[str] = []
    dependencies: list[TicketDependency] = []
    tickets = [
        derive_ticket(draft, plan_sha256=presented, reconciliation_entry_id=reconciliation.entry_id)
        for draft in ordered
    ]
    for position, ticket in enumerate(tickets):
        try:
            recorded = sink.record_ticket(ticket)
        except Exception as exc:
            raise TicketSetIncompleteError(
                created=created,
                pending=[pending.ticket_id for pending in tickets[position:]],
                cause=exc,
            ) from exc
        if recorded:
            created.append(ticket.ticket_id)
        else:
            existing.append(ticket.ticket_id)
        dependencies.extend(
            TicketDependency(ticket_id=ticket.ticket_id, depends_on=blocker)
            for blocker in ticket.depends_on
        )

    return TicketCreationOutcome(
        plan_sha256=presented,
        reconciliation_entry_id=reconciliation.entry_id,
        reconciliation_captured=reconciliation.captured,
        created=tuple(created),
        existing=tuple(existing),
        dependencies=tuple(dependencies),
        decisions=tuple(decisions),
        approval=approval,
    )


#: Schema v1 of the ticket store: the tickets, and the edges between them.
#:
#: **The edges are their own table with two foreign keys.** Writing a dependency as a line
#: in a description would make it true only as long as the prose stayed intact, and nothing
#: could query "what is startable". A row that references a ticket which does not exist is
#: refused by the database, which is the property that lets creation order matter: a
#: blocker is committed before the ticket that waits for it.
#:
#: ``created_at`` is first-write-wins, for the same reason an approval's timestamp is: a
#: re-run must not move the moment a ticket came into being, because that moment is the
#: provenance a reader checks.
_MIGRATION_V1_TICKETS: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS design_tickets ("
    "ticket_id TEXT PRIMARY KEY, "
    "plan_sha256 TEXT NOT NULL, "
    "reconciliation_entry_id TEXT NOT NULL, "
    "draft_id TEXT NOT NULL, "
    "title TEXT NOT NULL, "
    "description TEXT NOT NULL, "
    "acceptance_criteria TEXT NOT NULL, "
    "priority TEXT NOT NULL, "
    "test_command TEXT, "
    "reference_files TEXT NOT NULL, "
    "labels TEXT NOT NULL, "
    "text_sanitisation TEXT, "
    "created_at TEXT NOT NULL)",
    "CREATE UNIQUE INDEX IF NOT EXISTS design_tickets_identity_idx "
    "ON design_tickets (plan_sha256, draft_id)",
    "CREATE INDEX IF NOT EXISTS design_tickets_plan_idx ON design_tickets (plan_sha256)",
    "CREATE TABLE IF NOT EXISTS design_ticket_edges ("
    "ticket_id TEXT NOT NULL REFERENCES design_tickets (ticket_id), "
    "depends_on_ticket_id TEXT NOT NULL REFERENCES design_tickets (ticket_id), "
    "PRIMARY KEY (ticket_id, depends_on_ticket_id))",
)

#: The ticket store's own version-1 statements: one dictionary per store, so
#: opening the wrong store on the right file is a checksum mismatch and a
#: refusal. See ``ticket_approvals`` for why the dictionaries are not shared.
_TICKET_MIGRATIONS: dict[int, tuple[str, ...]] = {1: _MIGRATION_V1_TICKETS}
_TICKET_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "design_tickets": (
        "ticket_id",
        "plan_sha256",
        "reconciliation_entry_id",
        "draft_id",
        "title",
        "description",
        "acceptance_criteria",
        "priority",
        "test_command",
        "reference_files",
        "labels",
        "text_sanitisation",
        "created_at",
    ),
    "design_ticket_edges": ("ticket_id", "depends_on_ticket_id"),
}


def _read_json_list(value: object) -> tuple[str, ...]:
    """Read a stored JSON array of strings back into a tuple.

    Falls back to ``()`` on anything unparseable rather than raising. That is not leniency
    about a corrupted store: the list columns here are the ones a plan *may* leave empty,
    so an unreadable one and an empty one have the same consequence -- the field a reader
    would consult for optional detail is absent -- whereas raising would turn a display
    problem into a failed creation run.
    """
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(item for item in parsed if isinstance(item, str))


def _dumps_list(values: Sequence[str]) -> str:
    """Serialise a list column. Compact separators, so a row stays readable in ``sqlite3``."""
    return json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)


class SqliteTicketSink:
    """A local, durable :class:`TicketSink`, with no network anywhere in it.

    The only ticket store this repository has, and its reason for being local rather than
    an adapter is in the module docstring: there is no ticket API to send an idempotency
    key to. What a network adapter will eventually need from here is already here -- the
    derived id, which is the one part of idempotency that does not require the destination
    to cooperate -- so adding one later is a matter of implementing :class:`TicketSink` over
    a client, not of re-deriving what a ticket is.

    **The character policy runs here, not in the derivation.** A ticket body is model
    prose that will be displayed, and this is the boundary where untrusted text becomes
    stored text, which is where :mod:`kojutsu.core.text_hygiene` says the policy belongs.
    What was removed is recorded under :data:`~kojutsu.core.text_hygiene.SANITISATION_KEY`
    on the row rather than performed silently, and a field that is *empty after* the policy
    runs is refused: an invisible-only title sanitises to nothing, and a ticket nobody can
    find is not a ticket.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._db, self._lock = _open_store(
            self.path,
            migrations=_TICKET_MIGRATIONS,
            expected_columns=_TICKET_TABLE_COLUMNS,
            what="Ticket store",
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> SqliteTicketSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record_ticket(self, ticket: DerivedTicket) -> bool:
        """Write the ticket and its edges, or recognise both as already stored.

        Returns ``True`` when this call created the row. The primary key is the
        idempotency: the second run of an approval computes the same
        :attr:`DerivedTicket.ticket_id`, hits the key, and returns ``False``.

        The collision check afterwards is the part that makes that safe to rely on. A row
        could be taken by *another* ticket -- the derivation moved, or a digest collision --
        and returning ``False`` for that would report a silent overwrite as an idempotent
        no-op. So the stored anchor is read back and compared, and a row describing a
        different plan or draft is refused, on :meth:`SqliteQuestionRegistry.claim_rationale`'s
        argument.

        Edges are written in the same transaction as the ticket, and for an existing ticket
        they are re-asserted. Without both halves of that, a store that lost the edges would
        leave a ticket that looks startable and is not -- and the edges are the only record
        that it is not, so nothing downstream could notice.
        """
        stored = _stored_text(ticket)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                try:
                    self._db.execute(
                        "INSERT INTO design_tickets (ticket_id, plan_sha256, "
                        "reconciliation_entry_id, draft_id, title, description, "
                        "acceptance_criteria, priority, test_command, reference_files, "
                        "labels, text_sanitisation, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            ticket.ticket_id,
                            ticket.plan_sha256,
                            ticket.reconciliation_entry_id,
                            ticket.draft_id,
                            stored.title,
                            stored.description,
                            _dumps_list(stored.acceptance_criteria),
                            stored.priority,
                            stored.test_command,
                            _dumps_list(stored.reference_files),
                            _dumps_list(stored.labels),
                            stored.note,
                            _now(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    created = False
                    self._check_same_ticket(ticket)
                else:
                    created = True
                for blocker in stored.depends_on:
                    self._insert_edge(ticket_id=ticket.ticket_id, depends_on_ticket_id=blocker)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return created

    def _check_same_ticket(self, ticket: DerivedTicket) -> None:
        """Refuse to treat a row belonging to another plan or draft as this ticket's."""
        row = self._db.execute(
            "SELECT plan_sha256, draft_id FROM design_tickets WHERE ticket_id = ?",
            (ticket.ticket_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"ticket {ticket.ticket_id!r} was refused by a constraint but is not readable "
                "back, so the store cannot say whose ticket it is"
            )
        if (str(row[0]), str(row[1])) != (ticket.plan_sha256, ticket.draft_id):
            raise RuntimeError(
                f"ticket id {ticket.ticket_id!r} is already on file as the ticket for plan "
                f"{row[0]!r} draft {row[1]!r}, and this call is for plan {ticket.plan_sha256!r} "
                f"draft {ticket.draft_id!r}. Refusing rather than reporting a different "
                "ticket's row as this one."
            )

    def _insert_edge(self, *, ticket_id: str, depends_on_ticket_id: str) -> None:
        """Write one edge if it is not already there, or refuse it if it cannot exist.

        ``INSERT OR IGNORE`` is right here and wrong one level up: the primary key makes
        "already there" a normal outcome of a re-run, and there is no other constraint on
        this table that a re-run could trip by accident. The foreign keys are the point --
        they refuse an edge to a ticket that does not exist, which is what makes creation
        order load-bearing rather than merely tidy.
        """
        self._db.execute(
            "INSERT OR IGNORE INTO design_ticket_edges (ticket_id, depends_on_ticket_id) "
            "VALUES (?, ?)",
            (ticket_id, depends_on_ticket_id),
        )

    def get_ticket(self, ticket_id: str) -> StoredTicket | None:
        """Return one ticket with its edges, or ``None`` if it was never created."""
        with self._lock:
            row = self._db.execute(
                f"SELECT {_TICKET_COLUMNS} FROM design_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if row is None:
                return None
            dependencies = self._edges((str(row[0]),))[str(row[0])]
        return _ticket_from_row(row, dependencies=dependencies)

    def list_tickets(
        self, *, plan_sha256: str | None = None, limit: int = MAX_TICKET_LIST_LIMIT
    ) -> list[StoredTicket]:
        """Tickets in creation order, optionally narrowed to one plan.

        Creation order rather than id order, because id order is a digest and a reader
        asking "what did this plan become" wants the order the plan's dependency graph
        implied. ``ORDER BY rowid`` is that order and survives the ``VACUUM`` a store may
        have been through, because ``rowid`` is preserved rather than reassigned.
        """
        with self._lock:
            if plan_sha256 is None:
                rows = self._db.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM design_tickets ORDER BY rowid ASC LIMIT ?",
                    (_bounded_limit(limit),),
                ).fetchall()
            else:
                rows = self._db.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM design_tickets "
                    "WHERE plan_sha256 = ? ORDER BY rowid ASC LIMIT ?",
                    (plan_sha256, _bounded_limit(limit)),
                ).fetchall()
            dependencies = self._edges(tuple(str(row[0]) for row in rows))
        return [_ticket_from_row(row, dependencies=dependencies[str(row[0])]) for row in rows]

    def list_dependencies(self, ticket_id: str) -> tuple[str, ...]:
        """The ticket ids one ticket waits for, in the order the plan declared them."""
        with self._lock:
            return self._edges((ticket_id,)).get(ticket_id, ())

    def _edges(self, ticket_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        """Read the edges for several tickets at once.

        One query rather than one per ticket, because a plan of a hundred drafts would
        otherwise be a hundred queries on the read side -- and the read side is what an
        operator runs to answer "what did this plan become". Ordering by the insertion
        sequence rather than by the id keeps the declared order of a draft's dependencies,
        which is the order the plan's author wrote and the order a reader recognises.
        """
        if not ticket_ids:
            return {}
        placeholders = ", ".join("?" for _ in ticket_ids)
        rows = self._db.execute(
            f"SELECT ticket_id, depends_on_ticket_id FROM design_ticket_edges "
            f"WHERE ticket_id IN ({placeholders}) ORDER BY rowid ASC",
            ticket_ids,
        ).fetchall()
        collected: dict[str, tuple[str, ...]] = dict.fromkeys(ticket_ids, ())
        for ticket_id, depends_on in rows:
            collected[str(ticket_id)] = collected[str(ticket_id)] + (str(depends_on),)
        return collected


@dataclass(frozen=True)
class _StoredText:
    """One ticket's prose as the character policy left it, plus the note describing it."""

    title: str
    description: str
    acceptance_criteria: tuple[str, ...]
    priority: str
    test_command: str | None
    reference_files: tuple[str, ...]
    labels: tuple[str, ...]
    depends_on: tuple[str, ...]
    note: str | None


def _stored_text(ticket: DerivedTicket) -> _StoredText:
    """Apply the character policy to one ticket's text and refuse what it empties.

    Every field the plan required to be non-empty is checked **after** sanitising, which is
    the only order that catches the case: a title made entirely of an invisible character
    passes every bound the plan applied, because it is one perfectly legal character, and
    sanitises to the empty string. Storing that would be a ticket whose title is nothing,
    in a store where every other ticket's title is visible.

    The note is measured against the text **as it arrived**, over every field at once, for
    :func:`kojutsu.core.text_hygiene.describe_removals`'s reason: a per-field note would
    tell a reader three separate attacks where there was one.
    """
    title = sanitise(ticket.title).text
    description = sanitise(ticket.description).text
    criteria = tuple(sanitise(line).text for line in ticket.acceptance_criteria)
    priority = sanitise(ticket.priority).text
    test_command = None if ticket.test_command is None else sanitise(ticket.test_command).text
    references = tuple(sanitise(path).text for path in ticket.reference_files)
    labels = tuple(sanitise(label).text for label in ticket.labels)
    depends_on = tuple(ticket.depends_on)

    checked: list[tuple[str, str]] = [
        ("title", title),
        ("description", description),
        ("priority", priority),
        *((f"acceptance criterion {index}", line) for index, line in enumerate(criteria)),
        *((f"reference file {index}", path) for index, path in enumerate(references)),
        *((f"label {index}", label) for index, label in enumerate(labels)),
    ]
    if test_command is not None:
        checked.append(("test command", test_command))
    for name, value in checked:
        if not value.strip():
            raise ValueError(
                f"ticket {ticket.ticket_id!r} has a {name} that is empty once the character "
                "policy has run, so storing it would produce a field that reads as nothing. "
                "The plan's bounds cannot catch this: one invisible character is a legal "
                "non-empty string."
            )
        if len(value) > MAX_STORED_TICKET_FIELD_CHARS:
            raise ValueError(
                f"ticket {ticket.ticket_id!r} has a {name} of {len(value)} characters, past "
                f"the store's {MAX_STORED_TICKET_FIELD_CHARS}; refused rather than truncated"
            )

    note = describe_removals(
        ticket.title,
        ticket.description,
        *ticket.acceptance_criteria,
        ticket.priority,
        ticket.test_command,
        *ticket.reference_files,
        *ticket.labels,
    )
    return _StoredText(
        title=title,
        description=description,
        acceptance_criteria=criteria,
        priority=priority,
        test_command=test_command,
        reference_files=references,
        labels=labels,
        depends_on=depends_on,
        note=note,
    )


#: The columns every ticket read asks for, in :func:`_ticket_from_row`'s order. Named for
#: :data:`_APPROVAL_COLUMNS`'s reason, and spelled three times in the sink's read path.
_TICKET_COLUMNS = (
    "ticket_id, plan_sha256, reconciliation_entry_id, draft_id, title, description, "
    "acceptance_criteria, priority, test_command, reference_files, labels, "
    "text_sanitisation, created_at"
)


def _ticket_from_row(row: tuple[Any, ...], *, dependencies: tuple[str, ...]) -> StoredTicket:
    """Rebuild a stored ticket, with its edges attached."""
    return StoredTicket(
        ticket_id=str(row[0]),
        plan_sha256=str(row[1]),
        reconciliation_entry_id=str(row[2]),
        draft_id=str(row[3]),
        title=str(row[4]),
        description=str(row[5]),
        acceptance_criteria=_read_json_list(row[6]),
        priority=str(row[7]),
        test_command=None if row[8] is None else str(row[8]),
        reference_files=_read_json_list(row[9]),
        labels=_read_json_list(row[10]),
        text_sanitisation=None if row[11] is None else str(row[11]),
        created_at=_parse_timestamp(row[12]),
        dependencies=dependencies,
    )


#: The public surface, named rather than implied.
#:
#: Later phases and the CLI import from here, and a name that is not on this list is an
#: implementation detail a maintainer is free to restructure -- while one that is on it
#: carries the reasoning in its docstring with it. The two Protocols are here because the
#: two stores are separate on purpose; see the module docstring for why that is not
#: over-modularisation.
__all__ = [
    "MAX_APPROVAL_LIST_LIMIT",
    "MAX_APPROVAL_NOTE_CHARS",
    "MAX_STORED_TICKET_FIELD_CHARS",
    "MAX_TICKET_LIST_LIMIT",
    "PLAN_APPROVAL_IDENTITY_DOMAIN",
    "PLAN_APPROVAL_IDENTITY_VERSION",
    "PLAN_GATE_DECISION_IDENTITY_DOMAIN",
    "PLAN_GATE_DECISION_IDENTITY_VERSION",
    "SCHEMA_VERSION",
    "TICKET_IDENTITY_DOMAIN",
    "TICKET_IDENTITY_VERSION",
    "DerivedTicket",
    "PlanApproval",
    "PlanApprovalLedger",
    "PlanApprovalRequiredError",
    "PlanChangedError",
    "RecordedDecision",
    "SqlitePlanApprovalLedger",
    "SqliteTicketSink",
    "StoredTicket",
    "TicketCreationOutcome",
    "TicketDependency",
    "TicketSetIncompleteError",
    "TicketSink",
    "create_tickets_from_plan",
    "derive_ticket",
    "plan_approval_gates",
    "record_plan_approval",
    "stable_plan_approval_id",
    "stable_plan_gate_decision_id",
    "stable_ticket_id",
]
