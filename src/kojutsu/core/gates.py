"""Gates: the points where a human is the control rather than the bottleneck.

A gate answers one question -- may the loop proceed past this point? -- and it does
so from a registry rather than from a branch inside the worker, so adding a gate is a
declaration, not a change to the loop.

The default set is deliberately small. Two points block: a plan before it becomes
tickets, and a merge before it lands. Everything else proceeds unattended. That is a
policy choice, not a technical limit, and the whole point of the registry is that a
tighter or looser policy is a list, not a rewrite.

Every decision is recorded, including the ones that let work through. A gate that
only logs when it blocks is a gate nobody can trust for the times it did not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LifecyclePoint(StrEnum):
    """The named points at which gates are evaluated.

    Names are strings rather than an opaque enum so that configuration, logs, and
    the recorded decision stream all speak the same word.
    """

    PLAN_PROPOSED = "plan.proposed"
    PR_OPENED = "pr.opened"
    PR_UPDATED = "pr.updated"
    QUESTIONS_GENERATED = "questions.generated"
    EVIDENCE_PROPOSED = "evidence.proposed"
    PR_MERGED = "pr.merged"
    TICKET_COMPLETED = "ticket.completed"


class GateAction(StrEnum):
    """What a gate does when its predicate matches.

    ``BLOCK`` stops the loop and records the decision. ``WARN`` and ``NOTIFY`` let
    work through but leave a trace, so a policy can be tightened by watching before
    it is enforced.
    """

    BLOCK = "block"
    WARN = "warn"
    NOTIFY = "notify"


class GateVerdict(StrEnum):
    ALLOW = "allow"
    BLOCKED = "blocked"
    WARNED = "warned"
    NOTIFIED = "notified"


@dataclass(frozen=True)
class GateDecision:
    """One evaluated gate, recorded whether or not it intervened."""

    gate: str
    point: LifecyclePoint
    verdict: GateVerdict
    reason: str
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.verdict is GateVerdict.BLOCKED


#: A gate's predicate returns ``None`` to pass, or a reason string to intervene.
GatePredicate = Callable[[dict[str, Any]], str | None]


#: Maps each gate action to its recorded verdict. Module-level so evaluate()
#: does not rebuild the same dict on every call.
_VERDICT_BY_ACTION: dict[GateAction, GateVerdict] = {
    GateAction.BLOCK: GateVerdict.BLOCKED,
    GateAction.WARN: GateVerdict.WARNED,
    GateAction.NOTIFY: GateVerdict.NOTIFIED,
}


@dataclass(frozen=True)
class Gate:
    """One named policy rule at one lifecycle point.

    The predicate returns ``None`` when the gate does not apply, or a reason string
    explaining why it does. Returning the reason rather than a bare bool is
    deliberate: a blocked cycle with no stated reason is one an operator cannot act
    on, and a warned cycle that does not say what it saw is noise.
    """

    name: str
    point: LifecyclePoint
    action: GateAction
    predicate: GatePredicate

    def evaluate(self, context: dict[str, Any]) -> GateDecision:
        reason = self.predicate(context)
        if reason is None:
            return GateDecision(
                gate=self.name,
                point=self.point,
                verdict=GateVerdict.ALLOW,
                reason="not applicable",
                context=dict(context),
            )
        verdict = _VERDICT_BY_ACTION[self.action]
        return GateDecision(
            gate=self.name,
            point=self.point,
            verdict=verdict,
            reason=reason,
            context=dict(context),
        )


def _always_human(reason: str) -> GatePredicate:
    """A gate that always stops at its point, for a human-only transition."""

    def predicate(_context: dict[str, Any]) -> str:
        return reason

    return predicate


def _never(_context: dict[str, Any]) -> str | None:
    return None


#: The default policy: a human approves the plan and the merge, nothing else.
DEFAULT_GATES: tuple[Gate, ...] = (
    Gate(
        name="plan-approval",
        point=LifecyclePoint.PLAN_PROPOSED,
        action=GateAction.BLOCK,
        predicate=_always_human(
            "a reconciled plan must be approved by a person before it becomes tickets"
        ),
    ),
    Gate(
        name="merge-review",
        point=LifecyclePoint.PR_MERGED,
        action=GateAction.BLOCK,
        predicate=_always_human("a merge is the point where a person stays in the loop"),
    ),
    Gate(
        name="autonomous-capture",
        point=LifecyclePoint.EVIDENCE_PROPOSED,
        action=GateAction.NOTIFY,
        # Present and permissive by default so that tightening this policy is a
        # one-line edit rather than a code change, and so the decisions are already
        # being recorded before anyone asks to see them.
        predicate=_never,
    ),
)


class GateRegistry:
    """The evaluated set of gates, keyed by lifecycle point."""

    def __init__(self, gates: tuple[Gate, ...] = DEFAULT_GATES) -> None:
        self._gates = tuple(gates)

    def gates_at(self, point: LifecyclePoint) -> tuple[Gate, ...]:
        return tuple(gate for gate in self._gates if gate.point is point)

    def evaluate(self, point: LifecyclePoint, context: dict[str, Any]) -> list[GateDecision]:
        """Evaluate every gate at a point. A blocking verdict stops the rest.

        Gates at the same point are not independent votes: the first block wins,
        because running the remaining gates after one has already refused would
        produce decisions about work that is not going to happen.
        """
        decisions: list[GateDecision] = []
        for gate in self.gates_at(point):
            decision = gate.evaluate(context)
            decisions.append(decision)
            if decision.blocks:
                break
        return decisions

    @staticmethod
    def is_blocked(decisions: list[GateDecision]) -> bool:
        return any(decision.blocks for decision in decisions)
