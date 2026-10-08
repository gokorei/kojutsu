"""The capture-only cycle and the pull-request work source.

Two things are pinned here that the existing worker tests could not reach, because
both are about a worker that runs *unattended against a real repository* rather than
a worker driven by a test.

The first is that opting out of ``IMPLEMENT`` is visible. A cycle that skipped
implementation because nobody configured it, and a cycle that skipped it because it
was told to, must not look alike. So the step set is configuration, it is reported,
and a resume cannot re-enter a stage the worker never agreed to run -- which is the
subtle one, because a resume keyed on the step name would walk straight back into
``IMPLEMENT`` on a capture-only worker.

The second is that a review is not repeated. A worker that re-reviews every open
pull request on every restart does not fail loudly; it just posts the same questions
again, forever, on somebody else's pull request. The queue record is therefore the
thing under test: what is done, what is in flight, and what a dead process left
behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kojutsu.config import Settings
from kojutsu.worker import (
    CAPTURE_ONLY_STEPS,
    DELEGATED_STEPS,
    CaptureOnlyCycleSteps,
    CycleOutcome,
    ImplementerRequiredError,
    Step,
    Worker,
    WorkerConfig,
    WorkItem,
)
from kojutsu.worker.sources import OpenPullRequestSource

from .test_loop import OpenGates

MODEL = "opencode/model"
ITEM = WorkItem("acme/repo#7", "acme/repo", "feat/7", 7, subject="Fix the fence")


class _RecordingSteps:
    """Records which stages the loop actually dispatched, in order."""

    def __init__(self) -> None:
        self.ran: list[Step] = []

    def implement(self, item: WorkItem) -> str:
        self.ran.append(Step.IMPLEMENT)
        return "https://github.com/acme/repo/pull/99"

    def ask(self, item: WorkItem) -> str:
        self.ran.append(Step.ASK)
        return str(item.pr_number)

    def answer(self, item: WorkItem) -> str:
        self.ran.append(Step.ANSWER)
        return "posted 1 answer(s)"

    def capture(self, item: WorkItem) -> str:
        self.ran.append(Step.CAPTURE)
        return "1 comment(s) posted for capture"


# --- the step set is configuration, and is reported ----------------------------


def test_the_default_cycle_still_implements() -> None:
    """The opt-in must not have become the default."""
    assert WorkerConfig().steps == DELEGATED_STEPS
    assert Step.IMPLEMENT in WorkerConfig().steps


def test_a_capture_only_worker_says_it_is_one() -> None:
    class Steps:
        def implement(self, item: WorkItem) -> str:
            raise AssertionError("implement must not be dispatched")

        def ask(self, item: WorkItem) -> str:
            return str(item.pr_number)

        def answer(self, item: WorkItem) -> str:
            return "posted 1 answer(s)"

        def capture(self, item: WorkItem) -> str:
            return "1 comment(s) posted for capture"

    class Source:
        def __init__(self) -> None:
            self.log: list[str] = []

        def ready_work(self, *, limit: int) -> list[WorkItem]:
            return [ITEM] if limit else []

        def claim(self, item: WorkItem) -> str | None:
            self.log.append("claim")
            return "t1"

        def release(self, item: WorkItem, claim_token: str, reason: str) -> bool:
            self.log.append("release")
            return True

        def complete(self, item: WorkItem, claim_token: str, deliverable: str, notes: str) -> bool:
            self.log.append("complete")
            return True

        def record_branch(self, item: WorkItem, branch: str) -> bool:
            self.log.append("record_branch")
            return True

        def release_by_token(self, item_id: str, claim_token: str, reason: str) -> bool:
            return True

    source = Source()
    worker = Worker(
        config=WorkerConfig(state_path=None, answer_model=MODEL, steps=CAPTURE_ONLY_STEPS),
        source=source,
        steps=Steps(),  # type: ignore[arg-type]
        gates=OpenGates(),
    )

    report = worker.run_once()
    status = worker.status()

    assert report.outcome is CycleOutcome.DONE
    assert status["capture_only"] is True
    assert status["steps"] == ["ask", "answer", "capture"]
    assert Step.IMPLEMENT not in [Step(step) for step in report.steps_run]


def test_a_capture_only_cycle_delivers_what_it_actually_produced() -> None:
    """No implement step means no pull request, so the deliverable is the capture.

    Falling back to the branch name would put a link on the record for a branch
    nobody opened, which is the specific thing the loop's own comment warns about.
    """

    class Steps:
        def implement(self, item: WorkItem) -> str:
            raise AssertionError("unreachable")

        def ask(self, item: WorkItem) -> str:
            return "7"

        def answer(self, item: WorkItem) -> str:
            return "posted 2 answer(s)"

        def capture(self, item: WorkItem) -> str:
            return "2 comment(s) posted for capture"

    seen: dict[str, str] = {}

    class Source:
        def ready_work(self, *, limit: int) -> list[WorkItem]:
            return [ITEM]

        def claim(self, item: WorkItem) -> str | None:
            return "t1"

        def release(self, item: WorkItem, claim_token: str, reason: str) -> bool:
            return True

        def complete(self, item: WorkItem, claim_token: str, deliverable: str, notes: str) -> bool:
            seen["deliverable"] = deliverable
            return True

        def record_branch(self, item: WorkItem, branch: str) -> bool:
            return True

        def release_by_token(self, item_id: str, claim_token: str, reason: str) -> bool:
            return True

    Worker(
        config=WorkerConfig(state_path=None, steps=CAPTURE_ONLY_STEPS),
        source=Source(),  # type: ignore[arg-type]
        steps=Steps(),  # type: ignore[arg-type]
        gates=OpenGates(),
    ).run_once()

    assert seen["deliverable"] == "2 comment(s) posted for capture"
    assert seen["deliverable"] != ITEM.branch


def test_capture_only_steps_refuse_to_implement_if_ever_called() -> None:
    steps = CaptureOnlyCycleSteps(apply=False, answer_model=MODEL, settings=Settings())

    with pytest.raises(ImplementerRequiredError) as caught:
        steps.implement(ITEM)

    assert "capture-only" in str(caught.value)


def test_a_capture_only_worker_never_dispatches_implement(tmp_path: Path) -> None:
    """Durable state naming ``IMPLEMENT`` must not pull a capture-only worker into it.

    The state can legitimately say ``implement`` when a worker is reconfigured from
    a full cycle to a capture-only one, or when the record was written by an earlier
    boot. The step set is what decides what runs, so the stale position is ignored
    and the cycle resumes within the set it was actually given.
    """
    source = OpenPullRequestSource(
        _StubGitHub([_StubPull(7)]),  # type: ignore[arg-type]
        repo="acme/repo",
        state_path=tmp_path / "q.json",
    )
    steps = _RecordingSteps()
    worker = Worker(
        config=WorkerConfig(state_path=tmp_path / "s.json", steps=CAPTURE_ONLY_STEPS),
        source=source,
        steps=steps,  # type: ignore[arg-type]
        gates=OpenGates(),
    )
    worker.state.record_position(ITEM.item_id, Step.IMPLEMENT.value)

    report = worker.run_once()

    assert report.outcome is CycleOutcome.DONE
    assert steps.ran == [Step.ASK, Step.ANSWER, Step.CAPTURE]
    assert Step.IMPLEMENT not in steps.ran


# --- the pull request source --------------------------------------------------


class _StubPull:
    def __init__(self, number: int, *, state: str = "open", merged: bool = False) -> None:
        self.number = number
        self.state = state
        self.merged_at = object() if merged else None
        self.title = f"Pull request {number}"
        self.head = {"ref": f"feat/{number}"}


class _StubGitHub:
    """Stands in for GitHubClient, returning whatever pulls the test configured."""

    def __init__(self, pulls: list[_StubPull]) -> None:
        self.pulls = pulls
        self.listed: list[int] = []

    def list_open_pull_requests(self, owner: str, repo: str, *, limit: int = 30) -> list[Any]:
        self.listed.append(limit)
        return self.pulls[:limit]


def _source(tmp_path: Path, pulls: list[_StubPull]) -> OpenPullRequestSource:
    return OpenPullRequestSource(
        _StubGitHub(pulls),  # type: ignore[arg-type]
        repo="acme/repo",
        state_path=tmp_path / "queue.json",
    )


def test_open_pull_requests_are_ready_work(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1), _StubPull(2)])

    items = source.ready_work(limit=5)

    assert [item.item_id for item in items] == ["acme/repo#1", "acme/repo#2"]
    assert items[0].repo == "acme/repo"
    assert items[0].pr_number == 1
    assert items[0].branch == "feat/1"


def test_closed_and_merged_pull_requests_are_not_work(tmp_path: Path) -> None:
    source = _source(
        tmp_path, [_StubPull(1, state="closed"), _StubPull(2, merged=True), _StubPull(3)]
    )

    assert [item.item_id for item in source.ready_work(limit=5)] == ["acme/repo#3"]


def test_a_completed_review_is_never_handed_out_again(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1), _StubPull(2)])

    first = source.ready_work(limit=1)[0]
    token = source.claim(first)
    assert token is not None
    assert source.complete(first, token, "capture", "notes") is True

    assert [item.item_id for item in source.ready_work(limit=5)] == ["acme/repo#2"]


def test_a_claim_is_exclusive(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1)])
    item = source.ready_work(limit=1)[0]

    first = source.claim(item)
    second = source.claim(item)

    assert first is not None
    assert second is None, "a second claim was handed out for a live lease"


def test_release_returns_the_review_to_the_queue(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1)])
    item = source.ready_work(limit=1)[0]
    token = source.claim(item)
    assert token is not None

    assert source.release(item, token, "gate closed") is True
    assert [i.item_id for i in source.ready_work(limit=5)] == ["acme/repo#1"]


def test_a_wrong_token_cannot_release_or_complete(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1)])
    item = source.ready_work(limit=1)[0]
    token = source.claim(item)
    assert token is not None

    assert source.release(item, "not-the-token", "x") is False
    assert source.complete(item, "not-the-token", "d", "n") is False
    assert source.reviewed() == {}
    assert "acme/repo#1" in source.in_flight()


def test_a_lease_from_a_dead_process_is_reclaimed(tmp_path: Path) -> None:
    """A stale lease must not park a review forever."""
    source = OpenPullRequestSource(
        _StubGitHub([_StubPull(1)]),  # type: ignore[arg-type]
        repo="acme/repo",
        state_path=tmp_path / "queue.json",
        lease_seconds=0.01,
    )
    item = source.ready_work(limit=1)[0]
    assert source.claim(item) is not None

    import time

    time.sleep(0.05)

    assert [i.item_id for i in source.ready_work(limit=5)] == ["acme/repo#1"]
    assert source.in_flight() == {}


def test_the_queue_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    first = OpenPullRequestSource(
        _StubGitHub([_StubPull(1), _StubPull(2)]),  # type: ignore[arg-type]
        repo="acme/repo",
        state_path=path,
    )
    item = first.ready_work(limit=1)[0]
    token = first.claim(item)
    assert token is not None
    first.complete(item, token, "capture", "notes")

    second = OpenPullRequestSource(
        _StubGitHub([_StubPull(1), _StubPull(2)]),  # type: ignore[arg-type]
        repo="acme/repo",
        state_path=path,
    )

    assert [i.item_id for i in second.ready_work(limit=5)] == ["acme/repo#2"]
    assert list(second.reviewed()) == ["acme/repo#1"]


def test_requeue_returns_a_completed_review_to_the_queue(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1)])
    item = source.ready_work(limit=1)[0]
    token = source.claim(item)
    assert token is not None
    source.complete(item, token, "capture", "notes")

    assert source.forget("acme/repo#1") is True
    assert [i.item_id for i in source.ready_work(limit=5)] == ["acme/repo#1"]
    assert source.forget("acme/repo#404") is False


def test_an_unreadable_queue_is_a_fresh_one_not_a_fatal_error(tmp_path: Path) -> None:
    """A corrupt file must not make a fresh checkout unrunnable."""
    path = tmp_path / "queue.json"
    path.write_text("{not json", encoding="utf-8")

    source = OpenPullRequestSource(
        _StubGitHub([_StubPull(1)]),  # type: ignore[arg-type]
        repo="acme/repo",
        state_path=path,
    )

    assert [i.item_id for i in source.ready_work(limit=5)] == ["acme/repo#1"]


def test_the_queue_file_is_not_world_readable(tmp_path: Path) -> None:
    """The queue is not a secret, but it is per-operator state under a home dir."""
    import stat as stat_module

    source = _source(tmp_path, [_StubPull(1)])
    source.claim(source.ready_work(limit=1)[0])

    mode = (tmp_path / "queue.json").stat().st_mode

    assert not mode & (stat_module.S_IRGRP | stat_module.S_IROTH)


def test_a_malformed_repo_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owner/name"):
        OpenPullRequestSource(
            _StubGitHub([]),  # type: ignore[arg-type]
            repo="not-a-repo",
            state_path=tmp_path / "queue.json",
        )


def test_the_queue_is_written_as_parseable_json(tmp_path: Path) -> None:
    source = _source(tmp_path, [_StubPull(1)])
    source.claim(source.ready_work(limit=1)[0])

    payload = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))

    assert payload["repo"] == "acme/repo"
    assert "acme/repo#1" in payload["claims"]


def test_completing_with_a_blank_deliverable_is_refused(tmp_path: Path) -> None:
    """A completion that points nowhere must not be recorded as done: the item
    stays claimed so it is retried rather than forgotten."""
    source = _source(tmp_path, [_StubPull(1)])
    item = source.ready_work(limit=1)[0]
    token = source.claim(item)
    assert token is not None

    assert source.complete(item, token, "", "notes") is False
    assert source.complete(item, token, "   ", "notes") is False
    assert source.complete(item, token, "https://github.com/acme/repo/pull/9", "notes") is True
