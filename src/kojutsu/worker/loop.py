"""The unattended loop: claim work, do it, record the reasoning, close the ticket.

The loop exists to remove a person from the *dispatch* of work, not from the
decisions that matter. It polls for ready work, claims it with a lease, runs a
resumable sequence of steps, and hands the outcome to the ledger and the ticket
system. Nothing is returned to a caller, because there is no caller: the outcome is
observable in those two systems and nowhere else. That is the property that lets the
orchestrator dissolve into a reader.

Two failure modes matter more than throughput, and both are handled explicitly.

**A claimed item is never stranded.** Every step runs under a guard that releases
the claim if it raises, and a claim ceiling stops a permanently failing item from
being retried forever. A loop that can strand work is worse than no loop, because it
quietly stops making progress while looking alive.

**A cycle never answers itself without saying so.** The implementer and answerer
models are configured separately, and when they are the same the records it
produces are labelled ``self_certified`` by capture. The worker does not pretend to
an independence it cannot have; it makes the weakness visible and queryable, which
is the honest arrangement given a single-model deployment must still be able to run.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from kojutsu.core.gates import (
    GateDecision,
    GateRegistry,
    GateVerdict,
    LifecyclePoint,
)

from .state import WorkerState


class Step(StrEnum):
    """The stages of one cycle, in the order the loop actually performs them.

    ``INTAKE`` and ``GATE`` happen before the claim, deliberately: a worker must not
    take a lease on work it is about to refuse, or a gate left closed by a human
    becomes a queue of items held by nobody. ``COMPLETE`` happens last.
    """

    INTAKE = "intake"
    GATE = "gate"
    CLAIM = "claim"
    IMPLEMENT = "implement"
    ASK = "ask"
    ANSWER = "answer"
    CAPTURE = "capture"
    COMPLETE = "complete"


#: The stages the loop hands to its ``CycleSteps``, in order. The other stages of
#: :class:`Step` are performed by the loop itself, so the report records the full
#: cycle while the dispatch path only reaches the parts it can drive.
DELEGATED_STEPS = (Step.IMPLEMENT, Step.ASK, Step.ANSWER, Step.CAPTURE)

#: A cycle that reviews work someone else already wrote. It is a named constant
#: rather than a flag because the difference is the whole point: ``IMPLEMENT``
#: exists to open a pull request, and a knowledge-capture cycle has nothing to
#: implement, so running it would either fail every cycle or require an
#: implementer that opens an empty branch. Opting in says which cycle is meant.
CAPTURE_ONLY_STEPS = (Step.ASK, Step.ANSWER, Step.CAPTURE)


class CycleOutcome(StrEnum):
    DONE = "done"
    IDLE = "idle"
    GATED = "gated"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


@dataclass(frozen=True)
class WorkItem:
    """One claimable unit of work, as the loop sees it.

    Deliberately structural rather than tied to a ticket backend, so a cycle can be
    exercised end to end against a list in a test without a live ticket system.
    """

    item_id: str
    repo: str
    branch: str
    pr_number: int | None = None
    subject: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkSource(Protocol):
    """Where ready work comes from, and how a claim is given back."""

    def ready_work(self, *, limit: int) -> list[WorkItem]: ...

    def close(self) -> None:
        """Hand back whatever this source is holding open.

        Declared on the Protocol rather than probed for, because a source that
        owns a connection pool has no other way to say so, and a caller that
        reaches for ``getattr(source, "close", None)`` instead is writing a check
        that cannot fail when the leak is real. Implementations with nothing to
        release make this a no-op -- the method is part of the contract, not a
        request for cleanup.
        """

    def claim(self, item: WorkItem) -> str | None: ...

    def release(self, item: WorkItem, claim_token: str, reason: str) -> bool: ...

    def complete(self, item: WorkItem, claim_token: str, deliverable: str, notes: str) -> bool: ...

    def record_branch(self, item: WorkItem, branch: str) -> bool:
        """Write the working branch onto the ticket.

        The deliverable URL alone is not enough. A ticket whose branch is only
        discoverable by following a pull request is a ticket nobody can check for
        a conflicting in-flight change, and it is the one piece of context an
        operator needs when two things are working the same item.
        """

    def release_by_token(self, item_id: str, claim_token: str, reason: str) -> bool:
        """Give back a claim identified only by its token.

        Needed on start-up, where the claim is recovered from durable state and
        the ``WorkItem`` it was taken for no longer exists in this process.
        """


class CycleSteps(Protocol):
    """The work itself, injected so the loop can be tested without real systems."""

    def implement(self, item: WorkItem) -> str: ...

    def ask(self, item: WorkItem) -> str: ...

    def answer(self, item: WorkItem) -> str: ...

    def capture(self, item: WorkItem) -> str: ...


@dataclass
class WorkerConfig:
    """How one worker behaves. The model pair is the part worth reading twice.

    ``implement_model`` and ``answer_model`` being equal is allowed -- a
    single-model deployment has no other option -- but it is the configuration that
    produces self-certified records, so it is stated here rather than implied by the
    provider default.
    """

    name: str = "kojutsu-worker"
    poll_interval_seconds: float = 30.0
    max_items_per_cycle: int = 1
    claim_ceiling: int = 5
    backoff_base_seconds: float = 30.0
    backoff_cap_seconds: float = 300.0
    implement_model: str = ""
    answer_model: str = ""
    implement_agent: str = "kojutsu-worker"
    answer_agent: str = "kojutsu-worker"
    state_path: Path | None = None
    #: Which delegated stages this cycle runs. The default is every one of them;
    #: :data:`CAPTURE_ONLY_STEPS` is the explicit opt-in for a worker that reviews
    #: work it did not write. It is configuration rather than a flag on the loop so
    #: that a run with a smaller set is visible in :meth:`Worker.status` instead of
    #: being indistinguishable from one where a step quietly did nothing.
    steps: tuple[Step, ...] = DELEGATED_STEPS

    @property
    def uses_one_model(self) -> bool:
        """True when the implementer and the answerer are the same model.

        Not an error. It is the arrangement whose records capture will label
        ``self_certified``, and a reader filtering on independence needs to be able
        to ask the question.
        """
        return bool(self.implement_model) and self.implement_model == self.answer_model


@dataclass
class CycleReport:
    """The whole outcome of one cycle, for the status view and for tests."""

    outcome: CycleOutcome
    item: WorkItem | None = None
    steps_run: list[Step] = field(default_factory=list)
    decisions: list[GateDecision] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    requeue_after_seconds: float | None = None

    @property
    def summary(self) -> str:
        subject = self.item.item_id if self.item else "-"
        detail = f" after {len(self.steps_run)} step(s)" if self.steps_run else ""
        if self.error:
            return f"{self.outcome.value} {subject}{detail}: {self.error}"
        return f"{self.outcome.value} {subject}{detail}"


#: Which lifecycle point each stage reaches. A gate declared at one of these is
#: actually consulted, so registering a gate is not a silent no-op.
#:
#: ``PR_MERGED`` is deliberately absent. A merge is not this cycle's business -- the
#: cycle stops at capture -- so a merge gate belongs to whatever merges the pull
#: request. Claiming to enforce it here would report a gate as enforced that nothing
#: ever checks.
GATE_POINT_FOR_STEP: dict[Step, LifecyclePoint] = {
    Step.GATE: LifecyclePoint.PLAN_PROPOSED,
    Step.ASK: LifecyclePoint.QUESTIONS_GENERATED,
    Step.ANSWER: LifecyclePoint.EVIDENCE_PROPOSED,
}


def _pending_steps(resume_from: str | None, order: Sequence[Step]) -> tuple[Step, ...]:
    """The delegated steps still to run, given where the last attempt stopped.

    Ordered by position in ``order``, never by comparing the step's name. "implement"
    sorts after "ask" alphabetically while running before it, and a resume that trusts
    the alphabet re-runs the very step it was meant to skip -- which for
    ``implement`` means opening a second pull request.

    The order is the configured one rather than the module constant, so a cycle that
    opted out of a stage does not try to resume *into* it.
    """
    if resume_from is None:
        return tuple(order)
    try:
        reached = Step(resume_from)
    except ValueError:
        return tuple(order)
    if reached not in order:
        return tuple(order)
    return tuple(order[order.index(reached) + 1 :])


#: The lifecycle points this cycle actually evaluates, derived from the stages it
#: runs so the two cannot drift apart.
CONSULTED_POINTS = tuple(dict.fromkeys(GATE_POINT_FOR_STEP.values()))


def backoff_for(attempts: int, *, base: float, cap: float) -> float:
    """Exponential backoff with jitter, matching the outbox's shape.

    Reusing the outbox's curve rather than inventing one keeps a single answer to
    "how long does this wait" across every queue in the system.
    """
    delay = min(cap, base * (2 ** min(max(attempts - 1, 0), 8)))
    return delay * random.uniform(0.5, 1.5)  # noqa: S311 - backoff jitter, not cryptographic


class Worker:
    """One polling loop over a work source, gated, leased, and release-on-failure."""

    def __init__(
        self,
        *,
        config: WorkerConfig,
        source: WorkSource,
        steps: CycleSteps,
        gates: GateRegistry | None = None,
        state: WorkerState | None = None,
        on_decision: Callable[[GateDecision], None] | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.steps = steps
        self.gates = gates or GateRegistry()
        self.state = state if state is not None else WorkerState(config.state_path)
        self._on_decision = on_decision

    @property
    def dead_letter(self) -> WorkerState:
        """The durable state, under the name the status view and tests use."""
        return self.state

    def close(self) -> None:
        """Hand back what this worker is holding open.

        The source, because that is who owns the forge connection: a client is
        built once for the whole worker and given to the source, so the source is
        the only object that can release it. Closing it *here* rather than at the
        construction site is the whole point -- the client outlives the call that
        built it by design, so a close next to the constructor would end the worker's
        ability to read the forge on its first poll.
        """
        self.source.close()
        self.state.close()

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def requeue(self, item_id: str) -> bool:
        """Return a dead-lettered item to the queue.

        The only way an exhausted item comes back. It is an explicit operator
        action precisely because the alternative -- a loop that quietly keeps
        retrying what it has already decided it cannot do -- is the failure mode
        this whole mechanism exists to prevent.
        """
        return self.state.requeue(item_id)

    def release_orphans(self) -> list[str]:
        """Release claims left behind by a process that died holding them.

        Returns the item ids released. Called on start-up: a lease recorded by a
        previous boot is not one this process can honour, and holding it silently
        would be a claim with no worker behind it.
        """
        orphans = self.state.orphans()
        released: list[str] = []
        for item_id, record in orphans.items():
            self.source.release_by_token(record.item_id, record.claim_token, "orphaned claim")
            self.state.release_claim(item_id)
            released.append(item_id)
        return released

    # -- gates ----------------------------------------------------------------

    def _gate(
        self, point: LifecyclePoint, context: dict[str, Any]
    ) -> tuple[bool, list[GateDecision]]:
        decisions = self.gates.evaluate(point, context)
        for decision in decisions:
            if self._on_decision is not None:
                self._on_decision(decision)
        return GateRegistry.is_blocked(decisions), decisions

    # -- one cycle ------------------------------------------------------------

    def run_once(self) -> CycleReport:
        items = self.source.ready_work(limit=self.config.max_items_per_cycle)
        if not items:
            return CycleReport(outcome=CycleOutcome.IDLE)

        for item in items:
            report = self._run_item(item)
            if report.outcome is not CycleOutcome.IDLE:
                return report
        return CycleReport(outcome=CycleOutcome.IDLE)

    def _run_item(self, item: WorkItem) -> CycleReport:
        if item.item_id in self.state.dead_ids():
            return CycleReport(
                outcome=CycleOutcome.IDLE, item=item, error="dead-lettered; requeue to retry"
            )
        context = {
            "item_id": item.item_id,
            "repo": item.repo,
            "branch": item.branch,
            "pr_number": item.pr_number,
        }
        decisions: list[GateDecision] = []
        run: list[Step] = [Step.INTAKE]
        produced: dict[Step, str] = {}

        # The plan gate is consulted before the claim on purpose: a worker must not
        # take a lease on work it is about to refuse, or a gate a human has left
        # closed becomes a queue of items held by nobody.
        for step in (Step.GATE, Step.CLAIM):
            run.append(step)
            point = GATE_POINT_FOR_STEP.get(step)
            if point is not None:
                stage_blocked, stage_decisions = self._gate(point, context)
                decisions.extend(stage_decisions)
                if stage_blocked:
                    blocking = next(d for d in stage_decisions if d.blocks)
                    # Recorded, not discarded. A gate holding every item looks
                    # exactly like an empty queue from the outside, and an operator
                    # must never have to guess which one they are looking at.
                    self.state.block(item.item_id, reason=blocking.reason, gate=blocking.gate)
                    return CycleReport(
                        outcome=CycleOutcome.GATED,
                        item=item,
                        decisions=decisions,
                        error=blocking.reason,
                    )

        # Resume rather than restart. A worker killed after opening a pull request
        # must not open a second one, so the step it reached is durable and the
        # cycle picks up after it.
        resume_from = self.state.position(item.item_id)
        produced.update(
            (Step(step), value)
            for step, value in self.state.produced(item.item_id).items()
            if step in set(Step)
        )
        pending = _pending_steps(resume_from, self.config.steps)

        attempts = self.state.record_attempt(item.item_id)
        if attempts > self.config.claim_ceiling:
            reason = f"exceeded the claim ceiling of {self.config.claim_ceiling}"
            self.state.dead_letter(item.item_id, error=reason, attempts=attempts)
            return CycleReport(
                outcome=CycleOutcome.DEAD_LETTERED,
                item=item,
                attempts=attempts,
                error=reason,
            )

        claim_token = self.source.claim(item)
        run.pop()  # CLAIM is recorded only once the claim was actually taken
        if claim_token is None:
            # A refused claim is a normal outcome, not an error: under a single
            # worker it usually means another process got there first or the item
            # is already held.
            return CycleReport(outcome=CycleOutcome.IDLE, item=item, attempts=attempts)
        run.append(Step.CLAIM)
        # Record the branch on the ticket once the claim is held and before any
        # work runs, so a crash during implement cannot leave a completed ticket
        # whose branch is only discoverable by following the pull request. After
        # the claim rather than before, because a branch should only be written for
        # work this worker actually holds.
        if item.branch:
            self.source.record_branch(item, item.branch)
        self.state.record_claim(
            item.item_id,
            claim_token,
            repo=item.repo,
            branch=item.branch,
            pr_number=item.pr_number,
        )

        try:
            for step in pending:
                run.append(step)
                # The remaining gates are consulted as the cycle reaches their
                # lifecycle point, so a gate registered there is not decoration.
                point = GATE_POINT_FOR_STEP.get(step)
                if point is not None:
                    stage_blocked, stage_decisions = self._gate(point, context)
                    decisions.extend(stage_decisions)
                    if stage_blocked:
                        blocking = next(d for d in stage_decisions if d.blocks)
                        raise _GateStoppedError(blocking.reason, blocking.gate)
                # Keep what each step actually produced. The implement step's output
                # is the pull request the forge really opened, and it is the only
                # honest deliverable available -- guessing one from the branch name
                # puts a link on the ticket that may not exist.
                produced[step] = self._dispatch(step, item)
                # Record position and output before moving on, in one write, so
                # a crash between two steps resumes at the step that had not
                # run yet -- with its predecessor's output already stored.
                self.state.record_step(item.item_id, step.value, produced[step])
            run.append(Step.COMPLETE)
        except _GateStoppedError as gated:
            self.source.release(item, claim_token, gated.reason)
            self.state.release_claim(item.item_id)
            self.state.block(item.item_id, reason=gated.reason, gate=gated.gate)
            return CycleReport(
                outcome=CycleOutcome.GATED,
                item=item,
                steps_run=run,
                decisions=decisions,
                error=gated.reason,
                attempts=attempts,
            )
        except Exception as exc:  # any failure must release the claim
            reason = f"{type(exc).__name__}: {exc}"
            released = self.source.release(item, claim_token, reason)
            self.state.release_claim(item.item_id)
            delay = backoff_for(
                attempts,
                base=self.config.backoff_base_seconds,
                cap=self.config.backoff_cap_seconds,
            )
            return CycleReport(
                outcome=CycleOutcome.FAILED,
                item=item,
                steps_run=run,
                decisions=decisions,
                error=reason if released else f"{reason} (and the claim could not be released)",
                attempts=attempts,
                requeue_after_seconds=delay,
            )

        # Whatever a step actually produced, in preference order, and never a URL
        # assembled from the branch name: a link that may not exist is worse than no
        # link. A capture-only cycle has no implement step, so its honest
        # deliverable is the capture it recorded rather than a branch nobody opened.
        deliverable = (
            produced.get(Step.IMPLEMENT)
            or produced.get(Step.CAPTURE)
            or produced.get(Step.ANSWER)
            or item.branch
        )
        if not deliverable.strip():
            # A ticket completed with an empty deliverable URL is a completion
            # that points nowhere: the next reader cannot check the work, and
            # the ticket system records "done" either way. Refused here, before
            # the source is touched, and reported as a failure so the attempt
            # accounting and backoff treat it like any other unsuccessful cycle
            # rather than like success.
            reason = "cycle produced no deliverable; refusing to complete the item"
            released = self.source.release(item, claim_token, reason)
            self.state.release_claim(item.item_id)
            delay = backoff_for(
                attempts,
                base=self.config.backoff_base_seconds,
                cap=self.config.backoff_cap_seconds,
            )
            return CycleReport(
                outcome=CycleOutcome.FAILED,
                item=item,
                steps_run=run,
                decisions=decisions,
                error=reason if released else f"{reason} (and the claim could not be released)",
                attempts=attempts,
                requeue_after_seconds=delay,
            )
        answered_by = produced.get(Step.ANSWER, "unset")
        notes = (
            f"cycle completed on {self.config.name}; "
            f"implementer={self.config.implement_model or 'unset'}, "
            f"answerer={self.config.answer_model or 'unset'}; "
            f"answer_source={answered_by}"
        )
        self.source.complete(item, claim_token, deliverable, notes)
        # Only now is the item finished: clear attempts, position, produced values,
        # the claim and any block, so a success genuinely resets the history.
        self.state.clear_attempts(item.item_id)
        self.state.clear_progress(item.item_id)
        self.state.release_claim(item.item_id)
        self.state.clear_block(item.item_id)
        return CycleReport(
            outcome=CycleOutcome.DONE,
            item=item,
            steps_run=run,
            decisions=decisions,
            attempts=attempts,
        )

    def _dispatch(self, step: Step, item: WorkItem) -> str:
        handlers = {
            Step.IMPLEMENT: self.steps.implement,
            Step.ASK: self.steps.ask,
            Step.ANSWER: self.steps.answer,
            Step.CAPTURE: self.steps.capture,
        }
        handler = handlers.get(step)
        if handler is None:
            # Raising beats returning quietly: a step with no handler used to look
            # identical to a step that ran and did nothing.
            raise NotImplementedError(f"no handler wired for step {step!r}")
        return handler(item)

    # -- the loop -------------------------------------------------------------

    def run_forever(
        self,
        *,
        cycles: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_report: Callable[[CycleReport], None] | None = None,
    ) -> None:
        """Poll until cancelled, or for a fixed number of cycles in tests.

        Unlike the relay loop this does not swallow every exception and continue
        silently: a cycle that failed is reported through ``on_report`` and the loop
        keeps going, because an unattended worker that dies quietly is
        indistinguishable from one that is idle. The ``cycles`` bound is what makes
        the loop testable without a clock.
        """
        completed = 0
        while cycles is None or completed < cycles:
            report = self.run_once()
            if on_report is not None and report.outcome is not CycleOutcome.IDLE:
                on_report(report)
            completed += 1
            if cycles is not None and completed >= cycles:
                return
            if report.requeue_after_seconds:
                sleep(report.requeue_after_seconds)
            else:
                sleep(self.config.poll_interval_seconds)

    def may_merge(self, context: dict[str, Any]) -> tuple[bool, list[GateDecision]]:
        """Whether a pull request may be merged, and why.

        The merge gate is part of the default policy but nothing inside a cycle
        reaches a merge, so it needs an entry point of its own. Without this the
        policy would advertise a human checkpoint that no code ever consults, which
        is worse than not having the checkpoint.
        """
        return self._gate(LifecyclePoint.PR_MERGED, context)

    def status(self) -> dict[str, Any]:
        """One view of the loop, so an operator need not read logs.

        Claimed work is not reported here because it is not durable state: a claim
        is a lease, and the authoritative record of a claim is the registry that
        holds it. What is reported is what this process knows, and the dead letters
        it holds, both of which survive a restart.
        """
        return {
            "worker": self.config.name,
            "implement_model": self.config.implement_model,
            "answer_model": self.config.answer_model,
            "single_model": self.config.uses_one_model,
            "claim_ceiling": self.config.claim_ceiling,
            # Stated so a reader can tell a capture-only worker from one that was
            # supposed to open a pull request and did not.
            "steps": [step.value for step in self.config.steps],
            "capture_only": Step.IMPLEMENT not in self.config.steps,
            # Items with an attempt on record that are not dead: in flight, or
            # waiting out a backoff. Dead items are reported separately.
            "in_flight_attempts": {
                item_id: count
                for item_id, count in sorted(self.state.attempt_counts().items())
                if item_id not in self.state.dead_ids()
            },
            "claimed": {
                item_id: {
                    "claim_token": record.claim_token,
                    "repo": record.repo,
                    "branch": record.branch,
                    "pr_number": record.pr_number,
                    "orphaned": record.boot != self.state.boot,
                }
                for item_id, record in sorted(self.state.claims().items())
            },
            "blocked": self.state.blocked(),
            "resuming": dict(sorted(self.state.positions().items())),
            "dead_letter_ids": sorted(self.state.dead_ids()),
            "dead_lettered": self.state.dead_entries(),
            "durable": self.state.durable,
            "state_path": str(self.state.path) if self.state.path else None,
            "gates": [
                {
                    "name": gate.name,
                    "point": gate.point.value,
                    "action": gate.action.value,
                    # A gate at a point this cycle never reaches would be listed but
                    # never enforced, so say which it is rather than implying all.
                    "consulted_here": gate.point in CONSULTED_POINTS,
                }
                for point in CONSULTED_POINTS
                for gate in self.gates.gates_at(point)
            ],
            "gates_consulted": [point.value for point in CONSULTED_POINTS],
        }


class _GateStoppedError(Exception):
    """Internal signal that a gate stopped a cycle partway."""

    def __init__(self, reason: str, gate: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.gate = gate


def cycle_reports(reports: Sequence[CycleReport]) -> Iterator[str]:
    """Render a run of cycles, for a CLI status view."""
    for report in reports:
        yield report.summary


__all__ = [
    "CycleOutcome",
    "CycleReport",
    "GateVerdict",
    "Step",
    "WorkItem",
    "Worker",
    "WorkerConfig",
    "backoff_for",
    "cycle_reports",
]
