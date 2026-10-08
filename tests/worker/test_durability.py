"""What happens when the process dies part-way through.

Every other worker test injects an exception. An exception is handled: the claim is
released, the attempt counted, the item backed off. A crash is not an exception --
there is no handler to run, no release, no increment -- so it is the case the
bookkeeping has to survive on its own.

These are the tests that would have caught the two defects that made unattended
operation unsafe:

* the attempt counter lived in a dict, so every restart began again at zero and a
  worker that kept being killed never reached its claim ceiling. It re-ran
  ``implement`` and opened a second pull request for the same ticket, forever.
* the cycle position was not persisted, so a restart could not resume and re-ran
  every step from the top -- the same second-pull-request, by a different route.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from kojutsu.core.gates import (
    DEFAULT_GATES,
    Gate,
    GateAction,
    GateRegistry,
    LifecyclePoint,
)
from kojutsu.worker import CycleOutcome, Step, Worker, WorkerConfig, WorkerState, WorkItem
from kojutsu.worker.state import WorkerStateOwnershipError

from .test_end_to_end import RecordingForge, RecordingModel, RecordingTicketSystem, Steps
from .test_loop import FakeSteps, OpenGates

ITEM = WorkItem("CHRON-9", "acme/widgets", "feat/chron-9", 77)


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / "worker-state.json"


def _durable_worker(
    tmp_path: Path,
    *,
    steps: Any,
    gates: Any = None,
    claim_ceiling: int = 3,
    config: WorkerConfig | None = None,
) -> Worker:
    return Worker(
        config=config
        or WorkerConfig(claim_ceiling=claim_ceiling, state_path=_state_path(tmp_path)),
        source=RecordingTicketSystem(ITEM),
        steps=steps,
        gates=gates or OpenGates(),
    )


class CrashingSteps:
    """Dies the way a crash dies: no exception, no handler, no chance to clean up.

    Implemented by calling ``os._exit`` in a forked child in the durability tests
    that need a real process death. Here it raises ``BaseException`` in a subclass
    that the loop's ``except Exception`` deliberately does not catch, which
    reproduces the same bookkeeping gap without killing the test runner.
    """

    def __init__(self, *, after: Step, delegate: Any = None) -> None:
        self.after = after
        self.delegate = delegate
        self.completed: list[Step] = []

    def _run(self, step: Step, item: WorkItem) -> str:
        self.completed.append(step)
        if step is self.after:
            raise SystemExit(f"power cut during {step.value}")
        # Delegate, so a step that ran before the crash really did its work --
        # a crash test where nothing happened proves nothing about resume.
        handler = getattr(self.delegate, step.value) if self.delegate else None
        return handler(item) if handler else f"out-{step.value}"

    def implement(self, item: WorkItem) -> str:
        return self._run(Step.IMPLEMENT, item)

    def ask(self, item: WorkItem) -> str:
        return self._run(Step.ASK, item)

    def answer(self, item: WorkItem) -> str:
        return self._run(Step.ANSWER, item)

    def capture(self, item: WorkItem) -> str:
        return self._run(Step.CAPTURE, item)


# --- a crash must not repeat the work that already happened -------------------


def test_a_restart_does_not_open_a_second_pull_request(tmp_path: Path) -> None:
    """The whole reason position is persisted.

    The first worker opens a pull request and is then killed. A worker that
    restarts from the top would open a second one for the same ticket, and the
    attempt counter would never climb because each process starts it at zero.
    """
    forge = RecordingForge()
    steps = CrashingSteps(
        after=Step.ANSWER, delegate=Steps(forge, RecordingModel(), tmp_path / "r.json")
    )

    first = _durable_worker(tmp_path, steps=steps)
    with pytest.raises(SystemExit):
        first.run_once()

    # The pull request really was opened before the crash.
    assert forge.pulls == [("acme/widgets", "feat/chron-9")]

    # A fresh worker over the same state resumes after the last completed step.
    forge2 = RecordingForge()
    resumed = _durable_worker(tmp_path, steps=Steps(forge2, RecordingModel(), tmp_path / "r.json"))
    report = resumed.run_once()

    assert report.outcome is CycleOutcome.DONE
    assert forge2.pulls == [], "the restart must not open a second pull request"
    assert Step.IMPLEMENT not in report.steps_run
    assert Step.CAPTURE in report.steps_run


def test_a_restart_still_reaches_the_deliverable_it_already_produced(tmp_path: Path) -> None:
    """Resuming at capture means the deliverable came from the run that opened the PR.

    If the produced values were not persisted, a resumed cycle would have nothing to
    report and would fall back to the branch name -- putting a link on the ticket
    that was never opened.
    """
    forge = RecordingForge()
    crash = CrashingSteps(
        after=Step.ANSWER, delegate=Steps(forge, RecordingModel(), tmp_path / "r.json")
    )
    with pytest.raises(SystemExit):
        _durable_worker(tmp_path, steps=crash).run_once()

    tickets = RecordingTicketSystem(ITEM)
    resumed = Worker(
        config=WorkerConfig(claim_ceiling=3, state_path=_state_path(tmp_path)),
        source=tickets,
        steps=Steps(RecordingForge(), RecordingModel(), tmp_path / "r.json"),
        gates=OpenGates(),
    )
    resumed.run_once()

    assert tickets.log[-1][2] == "https://git/acme/widgets/tree/feat/chron-9"


def test_the_attempt_count_survives_a_restart(tmp_path: Path) -> None:
    """A ceiling that resets on every restart is not a ceiling."""
    outcomes = []
    for _ in range(5):
        worker = _durable_worker(tmp_path, steps=FakeSteps(fail_on="implement"), claim_ceiling=2)
        outcomes.append(worker.run_once().outcome)

    assert outcomes[:2] == [CycleOutcome.FAILED, CycleOutcome.FAILED]
    assert outcomes[2] is CycleOutcome.DEAD_LETTERED, "the ceiling is reached across restarts"
    assert outcomes[3:] == [CycleOutcome.IDLE, CycleOutcome.IDLE], "once dead, it stays dead"
    assert outcomes.count(CycleOutcome.DEAD_LETTERED) == 1, "and it does not re-trigger every cycle"


def test_a_crash_loop_is_eventually_bounded(tmp_path: Path) -> None:
    """The scenario the whole mechanism exists for, stated as one test.

    A worker that is killed on every attempt, over and over, must still reach its
    ceiling. Before the counter was durable this looped forever.
    """
    outcomes = []
    for _ in range(6):
        worker = _durable_worker(
            tmp_path, steps=CrashingSteps(after=Step.IMPLEMENT), claim_ceiling=2
        )
        try:
            outcomes.append(worker.run_once().outcome)
        except SystemExit:
            outcomes.append("crashed")

    assert outcomes.count("crashed") == 2
    assert outcomes[2] is CycleOutcome.DEAD_LETTERED, "the ceiling survives being crashed past"
    assert outcomes[3] is CycleOutcome.IDLE


def test_a_crash_leaves_an_orphaned_claim_a_restart_can_recover(tmp_path: Path) -> None:
    """A lease held by a dead process is a claim with nobody behind it."""
    with pytest.raises(SystemExit):
        _durable_worker(tmp_path, steps=CrashingSteps(after=Step.ANSWER)).run_once()

    fresh = _durable_worker(tmp_path, steps=FakeSteps())
    assert fresh.status()["claimed"], "a new process must see the claim its predecessor took"
    assert fresh.status()["claimed"]["CHRON-9"]["orphaned"] is True

    released = fresh.release_orphans()
    assert released == ["CHRON-9"]
    assert fresh.status()["claimed"] == {}


def test_a_claim_taken_by_this_process_is_not_an_orphan(tmp_path: Path) -> None:
    worker = _durable_worker(tmp_path, steps=CrashingSteps(after=Step.ANSWER))
    with pytest.raises(SystemExit):
        worker.run_once()

    assert worker.status()["claimed"]["CHRON-9"]["orphaned"] is False
    assert worker.release_orphans() == []


# --- a blocked queue must not look like an empty one ---------------------------


def test_a_blocked_item_is_visible_in_the_status_view(tmp_path: Path) -> None:
    """A gate holding everything used to leave no trace at all.

    ``status()`` returned an empty, healthy-looking system while the worker
    reported ``gated`` every cycle and threw the report away. The one state an
    operator must never have to guess about is the one that looks like nothing.
    """
    worker = _durable_worker(tmp_path, steps=FakeSteps(), gates=GateRegistry(DEFAULT_GATES))

    report = worker.run_once()

    assert report.outcome is CycleOutcome.GATED
    blocked = worker.status()["blocked"]
    assert "CHRON-9" in blocked
    assert "approved" in blocked["CHRON-9"]["reason"]
    assert blocked["CHRON-9"]["gate"] == "plan-approval"


def test_a_blocked_queue_is_distinguishable_from_an_empty_queue(tmp_path: Path) -> None:
    blocked = _durable_worker(tmp_path, steps=FakeSteps(), gates=GateRegistry(DEFAULT_GATES))
    blocked.run_once()

    empty = _durable_worker(tmp_path.parent / "empty", steps=FakeSteps(), gates=OpenGates())
    empty.run_once()

    assert blocked.status()["blocked"]
    assert empty.status()["blocked"] == {}
    assert blocked.status()["dead_letter_ids"] == empty.status()["dead_letter_ids"] == []


def test_a_block_is_cleared_once_the_item_succeeds(tmp_path: Path) -> None:
    """Otherwise a stale block would keep an item reported as stuck forever."""
    worker = _durable_worker(tmp_path, steps=FakeSteps(), gates=GateRegistry(DEFAULT_GATES))
    worker.run_once()
    assert worker.status()["blocked"]

    permissive = GateRegistry(tuple(g for g in DEFAULT_GATES if g.action is not GateAction.BLOCK))
    worker.gates = permissive
    assert worker.run_once().outcome is CycleOutcome.DONE
    assert worker.status()["blocked"] == {}


def test_a_gate_that_blocks_mid_cycle_records_which_gate_and_releases(tmp_path: Path) -> None:
    """A post-claim block must not strand the claim it just took."""

    def no_evidence(context: dict[str, Any]) -> str | None:
        return None if context.get("has_evidence") else "no evidence was captured"

    gates = GateRegistry(
        (
            Gate(
                name="evidence-gate",
                point=LifecyclePoint.EVIDENCE_PROPOSED,
                action=GateAction.BLOCK,
                predicate=no_evidence,
            ),
        )
    )
    tickets = RecordingTicketSystem(ITEM)
    worker = Worker(
        config=WorkerConfig(claim_ceiling=3, state_path=_state_path(tmp_path)),
        source=tickets,
        steps=Steps(RecordingForge(), RecordingModel(), tmp_path / "r.json"),
        gates=gates,
    )

    report = worker.run_once()

    assert report.outcome is CycleOutcome.GATED
    assert [entry[0] for entry in tickets.log] == ["claim", "record_branch", "release"]
    assert "no evidence" in tickets.log[-1][-1]
    assert worker.status()["blocked"]["CHRON-9"]["gate"] == "evidence-gate"
    assert worker.status()["claimed"] == {}, "a released claim must not still read as held"


# --- the state file itself ----------------------------------------------------


def test_state_survives_a_corrupt_file_by_starting_clean(tmp_path: Path) -> None:
    """Refusing to start would leave nothing alive to fix the file."""
    path = _state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")

    state = WorkerState(path)

    assert state.attempt_counts() == {}
    assert state.durable is True


def test_state_from_a_future_version_is_not_guessed_at(tmp_path: Path) -> None:
    path = _state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 99, "attempts": {"T-1": 4}}))

    assert WorkerState(path).attempt_counts() == {}


def test_the_status_view_says_whether_it_is_durable(tmp_path: Path) -> None:
    """An operator should not have to infer durability from the absence of an error."""
    assert _durable_worker(tmp_path, steps=FakeSteps()).status()["durable"] is True
    forgetful = Worker(
        config=WorkerConfig(),
        source=RecordingTicketSystem(ITEM),
        steps=FakeSteps(),
        gates=OpenGates(),
    )
    assert forgetful.status()["durable"] is False
    assert forgetful.status()["state_path"] is None


def test_a_state_write_is_atomic(tmp_path: Path) -> None:
    """No temporary file may be left beside the state."""
    worker = _durable_worker(tmp_path, steps=FakeSteps())
    worker.run_once()

    names = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert "worker-state.json" in names
    assert not [n for n in names if n.endswith(".tmp")], (
        "an atomic write leaves no temp file behind"
    )


def _open_state_in_child(path: str, result: multiprocessing.Queue[str]) -> None:
    try:
        with WorkerState(Path(path)):
            result.put("opened")
    except WorkerStateOwnershipError:
        result.put("owned")


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(), reason="requires process spawning"
)
def test_a_second_process_is_refused_while_state_is_owned(tmp_path: Path) -> None:
    """Two writers sharing one state file last-writer-win every field, so the
    second worker fails fast at startup instead of silently splitting brain."""
    path = _state_path(tmp_path)
    context = multiprocessing.get_context("spawn")
    result: multiprocessing.Queue[str] = context.Queue()
    with WorkerState(path):
        process = context.Process(target=_open_state_in_child, args=(str(path), result))
        process.start()
        try:
            assert result.get(timeout=10) == "owned"
        except Empty:
            pytest.fail("child process did not report its state ownership result")
        finally:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def test_same_process_reentry_shares_ownership(tmp_path: Path) -> None:
    """A loop and its status view holding the same path must not deadlock
    themselves; the second writer in another *process* is what is refused."""
    path = _state_path(tmp_path)
    first = WorkerState(path)
    try:
        second = WorkerState(path)
        try:
            second.record_attempt("T-1")
            assert first.attempts("T-1") == 0, "separate in-memory views, shared lock"
        finally:
            second.close()
    finally:
        first.close()
    reopened = WorkerState(path)
    try:
        assert reopened.attempts("T-1") == 1
    finally:
        reopened.close()


def test_record_step_writes_position_and_output_in_one_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between a position write and its output write resumes past a step
    whose result was never stored. One write makes them the same fact."""
    import kojutsu.worker.state as state_module

    writes: list[str] = []
    real_write = state_module._write_text_atomic

    def counting_write(path: Path, text: str) -> None:
        writes.append(path.name)
        real_write(path, text)

    monkeypatch.setattr(state_module, "_write_text_atomic", counting_write)
    with WorkerState(_state_path(tmp_path)) as state:
        state.record_step("T-1", "implement", "https://example.test/pr/1")

    assert writes == ["worker-state.json"], "position and output share one write"
    with WorkerState(_state_path(tmp_path)) as reopened:
        assert reopened.position("T-1") == "implement"
        assert reopened.produced("T-1") == {"implement": "https://example.test/pr/1"}


def test_non_ascii_state_round_trips_as_utf8(tmp_path: Path) -> None:
    """An explicit encoding, not the platform default: a state written on one
    locale must read back on another. JSON escapes non-ASCII as ``\\uXXXX``,
    which is ASCII and therefore valid UTF-8 everywhere -- what matters is the
    file decodes strictly and the value survives."""
    path = _state_path(tmp_path)
    with WorkerState(path) as state:
        state.record_attempt("PRÓJ-日本語-1")

    raw = path.read_bytes().decode("utf-8")  # raises on non-UTF-8 bytes
    assert "PRÓJ-日本語-1" in json.loads(raw)["attempts"]
    with WorkerState(path) as reopened:
        assert reopened.attempts("PRÓJ-日本語-1") == 1
