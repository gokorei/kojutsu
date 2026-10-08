"""Tests for the unattended loop and the gate registry.

The two properties that make an unattended loop safe rather than merely convenient:

* a claimed item is never stranded, whatever a step does;
* a gate blocks from a registry, so tightening the policy is a declaration and not a
  change to the loop.

Everything runs against fakes. A loop that can only be exercised against a live
ticket system is a loop nobody will run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kojutsu.core.gates import (
    DEFAULT_GATES,
    Gate,
    GateAction,
    GateRegistry,
    GateVerdict,
    LifecyclePoint,
)
from kojutsu.worker import (
    CycleOutcome,
    Step,
    Worker,
    WorkerConfig,
    WorkItem,
    backoff_for,
)


class FakeSource:
    def __init__(self, items: list[WorkItem] | None = None, *, claimable: bool = True) -> None:
        self.items = items if items is not None else [WorkItem("T-1", "org/repo", "feat/x", 1)]
        self.claimable = claimable
        self.claims: dict[str, str | None] = {}
        self.branches: list[tuple[str, str]] = []
        self.releases: list[tuple[str, str, str]] = []
        self.completions: list[tuple[str, str, str]] = []

    def ready_work(self, *, limit: int) -> list[WorkItem]:
        return self.items[:limit]

    def claim(self, item: WorkItem) -> str | None:
        if not self.claimable:
            return None
        token = f"token-{item.item_id}"
        self.claims[item.item_id] = token
        return token

    def release(self, item: WorkItem, claim_token: str, reason: str) -> bool:
        self.releases.append((item.item_id, claim_token, reason))
        return True

    def record_branch(self, item: WorkItem, branch: str) -> bool:
        self.branches.append((item.item_id, branch))
        return True

    def release_by_token(self, item_id: str, claim_token: str, reason: str) -> bool:
        self.releases.append((item_id, claim_token, reason))
        return True

    def complete(self, item: WorkItem, claim_token: str, deliverable: str, notes: str) -> bool:
        self.completions.append((item.item_id, deliverable, notes))
        return True


class FakeSteps:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []

    def _maybe_fail(self, name: str) -> str:
        self.calls.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"boom in {name}")
        return name

    def implement(self, item: WorkItem) -> str:
        return self._maybe_fail("implement")

    def ask(self, item: WorkItem) -> str:
        return self._maybe_fail("ask")

    def answer(self, item: WorkItem) -> str:
        return self._maybe_fail("answer")

    def capture(self, item: WorkItem) -> str:
        return self._maybe_fail("capture")


class OpenGates(GateRegistry):
    """Everything permitted. The default policy blocks a plan, which would stop
    every cycle here before the interesting part runs."""

    def __init__(self) -> None:
        super().__init__(tuple(g for g in DEFAULT_GATES if g.action is not GateAction.BLOCK))


def _worker(
    *, source: FakeSource | None = None, steps: FakeSteps | None = None, config=None, gates=None
) -> Worker:
    return Worker(
        config=config or WorkerConfig(),
        source=source or FakeSource(),
        steps=steps or FakeSteps(),
        gates=gates or OpenGates(),
    )


class EmptySteps(FakeSteps):
    def _maybe_fail(self, name: str) -> str:
        self.calls.append(name)
        return ""


def test_a_cycle_with_no_deliverable_fails_instead_of_completing() -> None:
    """Every step produced nothing and the item names no branch: completing it
    would record a ticket 'done' with a deliverable URL that points nowhere."""
    item = WorkItem("T-1", "org/repo", "", None)
    source = FakeSource(items=[item])
    report = _worker(source=source, steps=EmptySteps()).run_once()

    assert report.outcome is CycleOutcome.FAILED
    assert "no deliverable" in (report.error or "")
    assert source.completions == []


# --- the happy path ---------------------------------------------------------


def test_a_cycle_runs_every_step_in_order() -> None:
    steps = FakeSteps()
    source = FakeSource()
    report = _worker(source=source, steps=steps).run_once()

    assert report.outcome is CycleOutcome.DONE
    assert steps.calls == ["implement", "ask", "answer", "capture"]
    # The branch is on the ticket before any work runs, so a crash mid-cycle still
    # leaves a ticket a human can find the work on.
    assert source.branches == [("T-1", "feat/x")]
    assert list(report.steps_run) == list(Step)
    assert source.completions[0][0] == "T-1"


def test_an_empty_queue_is_idle_not_a_failure() -> None:
    report = _worker(source=FakeSource(items=[])).run_once()
    assert report.outcome is CycleOutcome.IDLE


def test_a_refused_claim_is_idle_not_an_error() -> None:
    """Under a single worker a refused claim usually means someone else got there."""
    report = _worker(source=FakeSource(claimable=False)).run_once()
    assert report.outcome is CycleOutcome.IDLE
    assert report.error is None


# --- nothing gets stranded --------------------------------------------------


@pytest.mark.parametrize("failing", ["implement", "ask", "answer", "capture"])
def test_a_failure_at_any_step_releases_the_claim(failing: str) -> None:
    source = FakeSource()
    steps = FakeSteps(fail_on=failing)
    report = _worker(source=source, steps=steps).run_once()

    assert report.outcome is CycleOutcome.FAILED
    assert len(source.releases) == 1, "the claim must be handed back"
    assert failing in source.releases[0][2]
    assert source.completions == [], "a failed cycle must not complete its item"


def test_a_failure_schedules_a_backoff_rather_than_an_immediate_retry() -> None:
    report = _worker(steps=FakeSteps(fail_on="implement")).run_once()
    assert report.requeue_after_seconds is not None
    assert report.requeue_after_seconds > 0


def test_the_claim_ceiling_dead_letters_rather_than_retrying_forever() -> None:
    """A ceiling of N allows N attempts; the N+1st is a dead letter, not a retry."""
    source = FakeSource()
    worker = _worker(
        source=source, steps=FakeSteps(fail_on="implement"), config=WorkerConfig(claim_ceiling=2)
    )
    outcomes = [worker.run_once().outcome for _ in range(4)]

    assert outcomes[:2] == [CycleOutcome.FAILED, CycleOutcome.FAILED]
    assert outcomes[2] is CycleOutcome.DEAD_LETTERED
    # And it stays out of the queue: a dead-lettered item is not silently picked up
    # again, because a loop that keeps retrying what it has already given up on is
    # the failure this whole mechanism exists to prevent.
    assert worker.run_once().outcome is CycleOutcome.IDLE
    assert len(worker.dead_letter.dead_entries()) == 1
    assert "T-1" in worker.status()["dead_letter_ids"]


def test_only_an_explicit_requeue_brings_a_dead_item_back() -> None:
    worker = _worker(steps=FakeSteps(fail_on="implement"), config=WorkerConfig(claim_ceiling=1))
    worker.run_once()
    worker.run_once()
    assert "T-1" in worker.status()["dead_letter_ids"]

    assert worker.requeue("T-1") is True
    assert worker.requeue("T-1") is False
    assert worker.status()["dead_letter_ids"] == []
    assert worker.dead_letter.dead_entries() == [], "requeue also clears the recorded letter"
    # And the requeued item is genuinely attempted again.
    assert worker.run_once().outcome is CycleOutcome.FAILED


def test_a_dead_letter_can_be_requeued() -> None:
    worker = _worker(steps=FakeSteps(fail_on="implement"), config=WorkerConfig(claim_ceiling=1))
    worker.run_once()
    worker.run_once()

    assert worker.dead_letter.dead_entries()
    assert worker.dead_letter.requeue("T-1") is True
    assert worker.dead_letter.requeue("T-1") is False


def test_a_dead_letter_survives_a_restart(tmp_path: Path) -> None:
    """A dead letter held only in a process that then exits is not actionable."""
    path = tmp_path / "worker-state.json"
    config = WorkerConfig(claim_ceiling=1, state_path=path)
    first = _worker(steps=FakeSteps(fail_on="implement"), config=config)
    first.run_once()
    first.run_once()

    letters = path.with_name("dead-letters.jsonl")
    assert [json.loads(line)["item_id"] for line in letters.read_text().splitlines()] == ["T-1"]

    # A fresh process over the same state must not pick the item up again.
    steps = FakeSteps(fail_on="implement")
    second = _worker(steps=steps, config=config)
    assert "T-1" in second.status()["dead_letter_ids"]
    assert second.run_once().outcome is CycleOutcome.IDLE
    assert steps.calls == [], "a restarted worker must not re-attempt a dead item"

    # And a requeue survives that restart too, or it was only ever a memory.
    assert second.requeue("T-1") is True
    third = _worker(steps=FakeSteps(), config=config)
    assert third.status()["dead_letter_ids"] == []
    assert third.run_once().outcome is CycleOutcome.DONE


def test_a_success_clears_the_attempt_count() -> None:
    """A flaky item must not die from failures it has already recovered from."""
    worker = _worker(steps=FakeSteps(fail_on="implement"), config=WorkerConfig(claim_ceiling=2))
    worker.run_once()  # fail 1
    worker.steps = FakeSteps()
    worker.run_once()  # success, counter must reset
    worker.steps = FakeSteps(fail_on="implement")
    assert [worker.run_once().outcome for _ in range(4)] == [
        CycleOutcome.FAILED,  # fail 1
        CycleOutcome.FAILED,  # fail 2
        CycleOutcome.DEAD_LETTERED,  # fail 3, past the ceiling
        CycleOutcome.IDLE,  # and it stays out of the queue
    ]


# --- gates ------------------------------------------------------------------


def test_the_default_policy_blocks_a_plan() -> None:
    decisions = GateRegistry().evaluate(LifecyclePoint.PLAN_PROPOSED, {})
    assert GateRegistry.is_blocked(decisions)
    assert decisions[0].gate == "plan-approval"


def test_the_default_policy_blocks_a_merge() -> None:
    decisions = GateRegistry().evaluate(LifecyclePoint.PR_MERGED, {})
    assert GateRegistry.is_blocked(decisions)


def test_the_default_policy_does_not_block_ordinary_points() -> None:
    for point in (
        LifecyclePoint.PR_OPENED,
        LifecyclePoint.QUESTIONS_GENERATED,
        LifecyclePoint.EVIDENCE_PROPOSED,
    ):
        assert not GateRegistry.is_blocked(GateRegistry().evaluate(point, {}))


def test_a_gate_at_an_unreached_point_is_marked_as_such() -> None:
    """A gate on ``pr.merged`` belongs to whoever merges, not to this cycle."""
    status = _worker().status()

    assert [g for g in status["gates"] if g["point"] == "pr.merged"] == []
    assert "pr.merged" not in status["gates_consulted"]


def test_the_merge_gate_is_reachable_by_something() -> None:
    """A default blocking gate that no code ever consults is a checkpoint in name only."""
    blocked, decisions = Worker(
        config=WorkerConfig(), source=FakeSource(), steps=FakeSteps()
    ).may_merge({"pr_number": 7, "repo": "org/repo"})

    assert blocked is True, "the default policy stops an unattended merge"
    assert decisions[0].gate == "merge-review"
    assert decisions[0].point is LifecyclePoint.PR_MERGED


def test_a_permissive_policy_lets_the_merge_through() -> None:
    blocked, _decisions = _worker(gates=OpenGates()).may_merge({"pr_number": 7})

    assert blocked is False


def test_a_blocked_plan_stops_the_cycle_before_claiming() -> None:
    source = FakeSource()
    steps = FakeSteps()
    report = Worker(config=WorkerConfig(), source=source, steps=steps).run_once()

    assert report.outcome is CycleOutcome.GATED
    assert steps.calls == [], "a gated cycle must not do any work"
    assert source.claims == {}, "a gated cycle must not claim anything"


def test_adding_a_gate_is_a_declaration_not_a_code_change() -> None:
    def no_prose_reviews(context: dict[str, Any]) -> str | None:
        return None if context.get("has_prose_review") else "evidence lacks a prose review"

    strict = GateRegistry(
        (
            *DEFAULT_GATES,
            Gate(
                name="require-prose-review",
                point=LifecyclePoint.EVIDENCE_PROPOSED,
                action=GateAction.BLOCK,
                predicate=no_prose_reviews,
            ),
        )
    )
    blocked = strict.evaluate(LifecyclePoint.EVIDENCE_PROPOSED, {"has_prose_review": False})
    allowed = strict.evaluate(LifecyclePoint.EVIDENCE_PROPOSED, {"has_prose_review": True})

    assert GateRegistry.is_blocked(blocked)
    assert not GateRegistry.is_blocked(allowed)


def test_a_blocking_gate_stops_the_other_gates_at_the_same_point() -> None:
    """Evaluating gates for work that will not happen produces misleading records."""
    calls: list[str] = []

    def make(name: str) -> Gate:
        def predicate(_context: dict[str, Any]) -> str:
            calls.append(name)
            return "blocked"

        return Gate(name, LifecyclePoint.PR_MERGED, GateAction.BLOCK, predicate)

    GateRegistry((make("first"), make("second"))).evaluate(LifecyclePoint.PR_MERGED, {})

    assert calls == ["first"]


def test_a_gate_records_a_decision_even_when_it_does_not_intervene() -> None:
    """A gate that only logs when it blocks cannot be trusted for the times it did not."""
    decisions = OpenGates().evaluate(LifecyclePoint.EVIDENCE_PROPOSED, {"item_id": "T-1"})
    assert [d.verdict for d in decisions] == [GateVerdict.ALLOW]
    assert decisions[0].reason == "not applicable"


def test_a_warning_gate_records_what_it_saw() -> None:
    def noisy(context: dict[str, Any]) -> str | None:
        return "answers are self-certified" if context.get("self_certified") else None

    registry = GateRegistry(
        (Gate("flag-self-certified", LifecyclePoint.EVIDENCE_PROPOSED, GateAction.WARN, noisy),)
    )
    warned = registry.evaluate(LifecyclePoint.EVIDENCE_PROPOSED, {"self_certified": True})
    quiet = registry.evaluate(LifecyclePoint.EVIDENCE_PROPOSED, {"self_certified": False})

    assert warned[0].verdict is GateVerdict.WARNED
    assert "self-certified" in warned[0].reason
    assert quiet[0].verdict is GateVerdict.ALLOW


def test_a_blocked_gate_states_a_reason() -> None:
    decisions = GateRegistry().evaluate(LifecyclePoint.PLAN_PROPOSED, {})
    assert decisions[0].reason
    assert "approved" in decisions[0].reason


# --- the loop itself ---------------------------------------------------------


def test_run_forever_stops_at_the_cycle_bound_without_a_clock() -> None:
    slept: list[float] = []
    source = FakeSource()
    worker = _worker(source=source)
    reports: list[Any] = []

    worker.run_forever(cycles=3, sleep=slept.append, on_report=reports.append)

    assert len(slept) == 2, "it sleeps between cycles, not after the last"
    assert len(reports) == 3
    assert all(r.outcome is CycleOutcome.DONE for r in reports)


def test_a_failed_cycle_still_reaches_the_status_view() -> None:
    reports: list[Any] = []
    _worker(steps=FakeSteps(fail_on="ask")).run_forever(
        cycles=1, sleep=lambda _s: None, on_report=reports.append
    )
    assert reports and reports[0].outcome is CycleOutcome.FAILED


def test_status_says_plainly_when_one_model_is_doing_both_jobs() -> None:
    """A single-model deployment must be visible as such, not implied by defaults."""
    same = Worker(
        config=WorkerConfig(implement_model="m", answer_model="m"),
        source=FakeSource(),
        steps=FakeSteps(),
    ).status()
    split = Worker(
        config=WorkerConfig(implement_model="m", answer_model="n"),
        source=FakeSource(),
        steps=FakeSteps(),
    ).status()

    assert same["single_model"] is True
    assert split["single_model"] is False
    # Only the gates this cycle can actually reach are listed, and it says which
    # points those are, so a gate elsewhere is not implied to be enforced.
    assert {g["name"] for g in same["gates"]} == {"plan-approval", "autonomous-capture"}
    assert {g["point"] for g in same["gates"]} <= set(same["gates_consulted"])
    assert "pr.merged" not in same["gates_consulted"]
    assert all(g["consulted_here"] for g in same["gates"])


def test_status_reports_dead_letters_and_in_flight_attempts() -> None:
    worker = _worker(steps=FakeSteps(fail_on="implement"), config=WorkerConfig(claim_ceiling=1))
    worker.run_once()
    assert worker.status()["in_flight_attempts"] == {"T-1": 1}

    worker.run_once()
    status = worker.status()

    assert status["dead_lettered"], "a dead letter must remain visible in the status view"
    assert status["worker"] == "kojutsu-worker"


# --- backoff ----------------------------------------------------------------


def test_backoff_grows_and_is_capped() -> None:
    early = backoff_for(1, base=30.0, cap=300.0)
    later = backoff_for(5, base=30.0, cap=300.0)
    capped = backoff_for(20, base=30.0, cap=300.0)

    assert early != later, "jitter must vary the wait"
    assert max(early, later) > 0
    assert capped <= 300.0 * 1.5


def test_backoff_is_never_negative_or_zero() -> None:
    for attempts in (0, 1, 2, 30):
        assert backoff_for(attempts, base=30.0, cap=300.0) > 0
