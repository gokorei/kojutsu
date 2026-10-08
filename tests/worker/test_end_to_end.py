"""One cycle, end to end, with every part of the seam faked.

The unit tests in ``test_loop.py`` prove each piece in isolation. This file exists
because the pieces are only as good as the way they meet: a worker that claims
something, asks a question, hands it to a model, captures the answer and completes
the ticket is the entire product, and it should be observable as one story before
anyone believes it works in pieces.

Nothing here touches a real ticket system, forge, or model. The fakes are strict
about the calls they receive, so a change that quietly skips a step fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kojutsu.core.gates import (
    DEFAULT_GATES,
    Gate,
    GateAction,
    GateRegistry,
    GateVerdict,
    LifecyclePoint,
)
from kojutsu.worker import (
    DELEGATED_STEPS,
    CycleOutcome,
    CycleSteps,
    Step,
    Worker,
    WorkerConfig,
    WorkItem,
)


class RecordingTicketSystem:
    """A ticket system that only admits the calls the loop is allowed to make.

    ``completed`` mirrors the real thing: once an item is done it stops being ready
    work, which is what stops a second cycle from completing it twice.
    """

    def __init__(self, item: WorkItem) -> None:
        self.item = item
        self.log: list[tuple[str, ...]] = []
        self.branches: list[tuple[str, str]] = []
        self.claimable = True
        self.completed = False
        #: Whether :meth:`close` has been called. ``WorkSource`` declares it because a
        #: real source owns a forge connection pool, so a double that omitted it
        #: would let a worker that never releases anything pass unnoticed.
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def ready_work(self, *, limit: int) -> list[WorkItem]:
        return [] if self.completed else [self.item][:limit]

    def claim(self, item: WorkItem) -> str | None:
        if not self.claimable:
            return None
        self.log.append(("claim", item.item_id))
        return "lease-1"

    def release(self, item: WorkItem, claim_token: str, reason: str) -> bool:
        self.log.append(("release", item.item_id, reason))
        return True

    def record_branch(self, item: WorkItem, branch: str) -> bool:
        self.branches.append((item.item_id, branch))
        self.log.append(("record_branch", item.item_id, branch))
        return True

    def release_by_token(self, item_id: str, claim_token: str, reason: str) -> bool:
        self.log.append(("release", item_id, reason))
        return True

    def complete(self, item: WorkItem, claim_token: str, deliverable: str, notes: str) -> bool:
        self.log.append(("complete", item.item_id, deliverable, notes))
        self.completed = True
        return True


class RecordingForge:
    def __init__(self) -> None:
        self.branches: list[tuple[str, str, str]] = []
        self.pulls: list[tuple[str, str]] = []

    def create_branch(self, repo: str, branch: str, base: str) -> str:
        self.branches.append((repo, branch, base))
        return f"https://git/{repo}/tree/{branch}"

    def open_pull_request(self, repo: str, branch: str) -> int:
        self.pulls.append((repo, branch))
        return 4242


class RecordingModel:
    """Answers with something that looks like a real answer, including a claim it
    could not have verified. The loop's job is to carry that through with its
    provenance intact, not to make the model right."""

    def __init__(self, *, model_id: str = "opencode/model") -> None:
        self.model_id = model_id
        self.calls: list[dict[str, Any]] = []

    def run(self, prompt: str, *, question_id: str, sandbox: Any) -> str:
        self.calls.append({"prompt": prompt, "question_id": question_id})
        return "PERSEVERE: the audit row is written in the same transaction as the state."


class Steps(CycleSteps):
    """Drives the forge and the model so the test can see both were really used."""

    def __init__(self, forge: RecordingForge, model: RecordingModel, record_path: Path) -> None:
        self.forge = forge
        self.model = model
        self.record_path = record_path
        self.order: list[Step] = []

    def implement(self, item: WorkItem) -> str:
        self.order.append(Step.IMPLEMENT)
        branch_url = self.forge.create_branch(item.repo, item.branch, "main")
        self.forge.open_pull_request(item.repo, item.branch)
        return branch_url

    def ask(self, item: WorkItem) -> str:
        self.order.append(Step.ASK)
        return "q-1"

    def answer(self, item: WorkItem) -> str:
        self.order.append(Step.ANSWER)
        return self.model.run("review this", question_id="q-1", sandbox=None)

    def capture(self, item: WorkItem) -> str:
        self.order.append(Step.CAPTURE)
        self.record_path.write_text(
            json.dumps(
                {"question_id": "q-1", "answer": "persevere", "answered_by": self.model.model_id}
            )
        )
        return str(self.record_path)


def _build(
    tmp_path: Path,
) -> tuple[Worker, Steps, RecordingTicketSystem, RecordingModel, RecordingForge]:
    item = WorkItem("CHRON-1", "acme/widgets", "feat/chron-1-loop", 1)
    tickets = RecordingTicketSystem(item)
    forge = RecordingForge()
    model = RecordingModel()
    steps = Steps(forge, model, tmp_path / "record.json")
    config = WorkerConfig(
        claim_ceiling=3,
        implement_model="opencode/model",
        answer_model="opencode/model",
        state_path=tmp_path / "worker-state.json",
    )
    permissive = GateRegistry(
        (
            *(g for g in DEFAULT_GATES if g.action is not GateAction.BLOCK),
            Gate("auditable", LifecyclePoint.EVIDENCE_PROPOSED, GateAction.WARN, lambda _c: None),
        )
    )
    return (
        Worker(config=config, source=tickets, steps=steps, gates=permissive),
        steps,
        tickets,
        model,
        forge,
    )


def test_a_full_cycle_produces_a_completed_ticket_and_a_captured_record(tmp_path: Path) -> None:
    worker, steps, tickets, model, forge = _build(tmp_path)

    report = worker.run_once()

    assert report.outcome is CycleOutcome.DONE
    # Every stage ran, in the order the product promises: intake and gate before
    # the claim, completion last.
    assert steps.order == list(DELEGATED_STEPS)
    assert list(report.steps_run) == list(Step)
    # The forge was driven, not stubbed at the boundary.
    assert forge.branches == [("acme/widgets", "feat/chron-1-loop", "main")]
    assert forge.pulls == [("acme/widgets", "feat/chron-1-loop")]
    # The model was asked once, and its identity is what will be recorded.
    assert len(model.calls) == 1
    assert model.calls[0]["question_id"] == "q-1"
    # The ticket went claim -> record_branch -> complete. Never released.
    assert [entry[0] for entry in tickets.log] == ["claim", "record_branch", "complete"]
    assert tickets.branches == [("CHRON-1", "feat/chron-1-loop")]
    # And the record on disk names the model that answered.
    assert json.loads((tmp_path / "record.json").read_text())["answered_by"] == model.model_id


def test_the_deliverable_is_what_the_forge_actually_returned(tmp_path: Path) -> None:
    """The ticket must carry the real pull request, not one guessed from the branch.

    A synthesised ``repo#pr-N`` looks plausible and links nowhere, which is the kind
    of unearned confidence this whole project is trying not to produce.
    """
    worker, _steps, tickets, _model, _forge = _build(tmp_path)

    worker.run_once()

    deliverable = tickets.log[-1][2]
    assert deliverable == "https://git/acme/widgets/tree/feat/chron-1-loop"
    assert "#pr-" not in deliverable


def test_the_completion_note_says_which_model_answered(tmp_path: Path) -> None:
    worker, _steps, tickets, _model, _forge = _build(tmp_path)

    worker.run_once()

    notes = tickets.log[-1][3]
    assert "answerer=opencode/model" in notes
    assert "answer_source=PERSEVERE" in notes


def test_a_cycle_that_cannot_ask_leaves_the_ticket_claimable(tmp_path: Path) -> None:
    """The single most important property: no stranded lease.

    If an item is claimed and the cycle then fails, the item must be released with
    the reason. Otherwise an unattended worker eventually wedges the entire queue
    behind leases it will never release.
    """
    worker, steps, tickets, _model, _forge = _build(tmp_path)

    def boom(item: WorkItem) -> str:
        raise RuntimeError("the model timed out")

    steps.answer = boom  # type: ignore[method-assign]

    report = worker.run_once()

    assert report.outcome is CycleOutcome.FAILED
    assert "the model timed out" in (report.error or "")
    kinds = [entry[0] for entry in tickets.log]
    assert kinds == ["claim", "record_branch", "release"]
    assert "the model timed out" in tickets.log[-1][-1]
    assert "complete" not in kinds


def test_a_blocked_gate_never_reaches_the_ticket_system(tmp_path: Path) -> None:
    """A human gate must be able to stop the loop before it takes ownership of work."""
    worker, _steps, tickets, model, forge = _build(tmp_path)
    worker.gates = GateRegistry(DEFAULT_GATES)  # the real, blocking policy

    report = worker.run_once()

    assert report.outcome is CycleOutcome.GATED
    assert tickets.log == [], "a gated cycle must not claim, and must not record a branch"
    assert model.calls == []
    assert forge.branches == []
    assert "plan-approval" in {decision.gate for decision in report.decisions}


def test_a_gate_that_blocks_ends_the_cycle_even_beside_a_permissive_gate(tmp_path: Path) -> None:
    """Approval that leaves no trace is indistinguishable from a gate that was
    switched off. The report is where an operator sees why work happened."""

    def approved(_context: dict[str, Any]) -> str | None:
        return None

    worker, _steps, _tickets, _model, _forge = _build(tmp_path)
    worker.gates = GateRegistry(
        (
            Gate(
                name="plan-approval",
                point=LifecyclePoint.PLAN_PROPOSED,
                action=GateAction.BLOCK,
                predicate=lambda _c: "waiting on a human",
            ),
            Gate("auto", LifecyclePoint.PLAN_PROPOSED, GateAction.NOTIFY, approved),
        )
    )

    report = worker.run_once()

    assert report.outcome is CycleOutcome.GATED
    blocking = [d for d in report.decisions if d.verdict is GateVerdict.BLOCKED]
    assert [d.gate for d in blocking] == ["plan-approval"]
    assert blocking[0].reason == "waiting on a human"


def test_two_cycles_do_not_double_complete_a_ticket(tmp_path: Path) -> None:
    """Restart safety: the second pass must find nothing left to do, or the same
    deliverable gets recorded twice."""
    worker, _steps, tickets, _model, _forge = _build(tmp_path)

    assert worker.run_once().outcome is CycleOutcome.DONE
    assert worker.run_once().outcome is CycleOutcome.IDLE
    assert [entry[0] for entry in tickets.log].count("complete") == 1


def test_a_healthy_run_leaves_no_outstanding_bookkeeping(tmp_path: Path) -> None:
    """A finished item must leave nothing behind that would affect a later run.

    The state file itself is expected to exist -- it is what makes a restart
    resume instead of repeating -- so what matters is that a completed cycle
    cleared its own attempt count, position, produced values and claim.
    """
    worker, _steps, _tickets, _model, _forge = _build(tmp_path)

    worker.run_once()

    assert not (tmp_path / "dead-letters.jsonl").exists(), "nothing died, so nothing to read"
    state = json.loads((tmp_path / "worker-state.json").read_text())
    assert state["attempts"] == {}
    assert state["position"] == {}
    assert state["produced"] == {}
    assert state["claims"] == {}
    assert state["blocked"] == {}
    assert worker.status()["dead_lettered"] == []
    assert worker.status()["dead_letter_ids"] == []
    assert worker.status()["claimed"] == {}
    assert worker.status()["resuming"] == {}
