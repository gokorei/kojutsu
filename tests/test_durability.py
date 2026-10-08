"""Crash safety: what a real SIGKILL leaves behind.

kojutsu acknowledges a capture before it has necessarily reached stable
storage, and the whole retry and dead-letter design assumes a process death
leaves recoverable state rather than a half-applied row. That was an
unfalsifiable claim until now: there were no crash tests at all.

These tests and the direct pragma assertions in ``test_sqlite_durability.py``
overlap on purpose, and the overlap was measured rather than assumed. Reverting
the outbox to ``synchronous=NORMAL`` fails all fifteen tests here, because a child
committing hundreds of entries in a tight loop leaves the page cache under enough
pressure for the gap to appear. But whether a crash suite catches a missing fsync
depends on kernel scheduling, page-cache pressure and filesystem, so that is a
property of the machine rather than of the code. The pragma assertion catches it
deterministically and names the culprit; this suite covers what a pragma cannot
show -- a torn write, a stranded lease, a lost acknowledgement.

Three rules govern how these tests are written.

**The crash is real.** Every child dies from an actual ``SIGKILL`` delivered by
the kernel -- no ``KeyboardInterrupt``, no ``os._exit``, no monkeypatched
``commit``. The child's death is asserted via its exit status, so a child that
somehow exited cleanly fails the test instead of passing it vacuously.

**Only invariants are asserted, never which side of a commit a round landed on.**
The arbitrary-moment test sleeps for a different delay each round and then kills
the writer mid-loop. Asserting the race outcome would be a flaky test; asserting
that whatever survived is coherent, deliverable, and free of duplicates is not.

**All-or-none is explicitly *not* asserted, because it is false here.** The
reference project's crash suite gets all-or-none for free from a batch append in
one transaction. kojutsu's ``enqueue`` is one transaction per entry, so a kill
mid-loop legitimately leaves a *prefix* of the batch. A test demanding all-or-none
would be testing a property this store was never built to have, and would fail for
the right reason at the wrong time.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.question_registry import SqliteQuestionRegistry

#: Delays swept across the window the write path actually occupies. Measured on
#: this machine the child commits roughly 400 entries in 60-90ms, so the sweep
#: runs from near-instant to just under the full run. Widen it carelessly and the
#: later rounds start killing a child that has already finished, which asserts
#: the complete case and proves nothing about a torn write.
ROUNDS = 10
FIRST_DELAY = 0.002
DELAY_STEP = 0.005
TIMEOUT = 120

#: A timestamp far enough in the past that every lease and claim is expired.
EXPIRED = "2000-01-01T00:00:00+00:00"


def _die() -> None:
    # No unwinding, no flushing, no atexit, no finally. The process stops here,
    # which is the only crash that is honest about what the test measures.
    os.kill(os.getpid(), signal.SIGKILL)


# Child bodies. Module level so the spawn start method can pickle them.


def _open_then_die(path: str, delay: float) -> None:
    """Take ownership of the outbox and die without ever closing it."""
    time.sleep(delay)
    TansekiOutbox(path)
    _die()


def _enqueue_then_die(path: str, entry_ids: tuple[str, ...], delay: float) -> None:
    """Commit every entry, then die before the caller is ever told."""
    time.sleep(delay)
    outbox = TansekiOutbox(path)
    for entry_id in entry_ids:
        outbox.enqueue(entry_id, {"id": entry_id})
    _die()


def _enqueue_lease_then_die(path: str, entry_id: str, delay: float) -> None:
    """Enqueue, take a delivery lease, then die while still holding it."""
    time.sleep(delay)
    outbox = TansekiOutbox(path)
    outbox.enqueue(entry_id, {"id": entry_id})
    outbox.claim_due(limit=1, lease_seconds=300)
    _die()


def _hammer_forever(path: str, ceiling: int, ready_path: str) -> None:
    """Enqueue distinct entries one after another, signalling once writing starts.

    Three details are load-bearing.

    The ready file: with the ``spawn`` start method the child re-imports this
    module and the package before any work happens, so a parent that starts
    counting immediately measures *import time* and kills the child before it
    ever opens the database. The delay is only meaningful once the child is
    actually writing, so the child says so.

    Distinct entries: re-enqueueing the same ids would converge to a fixed
    steady state, and a kill would be indistinguishable from a kill between
    writes. Distinct ids mean a kill leaves however many commits got through, so
    the survivor set has real structure -- and therefore something to assert
    holes in.

    Keep going until killed: the caller's delay sweep is expressed in
    milliseconds, and how many commits that buys depends entirely on the host.
    A child that stopped after a fixed count would finish before the longer
    delays on any fast machine, so those rounds would assert against a child
    that was already gone -- a crash test measuring nothing. ``ceiling`` is only
    a backstop against a runaway, set far beyond the sweep window.
    """
    outbox = TansekiOutbox(path)
    Path(ready_path).write_text("ready")
    for index in range(ceiling):
        outbox.enqueue(f"e{index}", {"id": f"e{index}"})


def _claim_then_die(registry_path: str, delay: float) -> None:
    """Claim a question, then die before the answer is written to the outbox."""
    time.sleep(delay)
    registry = SqliteQuestionRegistry(registry_path)
    registry.record_question(
        question_id="q1",
        github_comment_id=100,
        repo="org/repo",
        pr_number=1,
        pr_url="https://github.com/org/repo/pull/1",
        question_text="Why?",
        question_category="design_decision",
    )
    registry.claim_question_for_answer("q1", "worker-1")
    _die()


def _spawn(target: Callable[..., None], *args: object) -> BaseProcess:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=args)
    process.start()
    return process


def _run_child(target: Callable[..., None], *args: object) -> None:
    """Run a child that ends itself, and assert it really died by SIGKILL.

    A child that exited cleanly would make the test measure nothing, so its exit
    status is checked rather than assumed.
    """
    process = _spawn(target, *args)
    process.join(timeout=TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("child did not die within the timeout; the test measured nothing")
    assert process.exitcode == -signal.SIGKILL, (
        f"child exited with {process.exitcode}, not by SIGKILL; "
        "a child that exits cleanly makes the test vacuous"
    )


def _kill_child_after(
    target: Callable[..., None], args: tuple[object, ...], delay: float, ready_path: Path
) -> None:
    """Wait for the child to start writing, then kill it at a chosen offset.

    This is the harness for the arbitrary-moment test: the crash time is the
    parent's to pick, so it can be swept across the window the write path
    actually occupies. ``Process.kill`` sends SIGKILL on POSIX, so the child gets
    no chance to unwind, flush, or run an atexit hook.

    The delay is measured from the child's ready signal rather than from
    ``start()``. Counting from ``start()`` would time the child's module import
    instead of its writes, and the kill would land before the database was ever
    opened -- a test that passes while measuring nothing.
    """
    process = _spawn(target, *args)
    try:
        deadline = time.monotonic() + TIMEOUT
        while not ready_path.exists():
            if not process.is_alive():
                pytest.fail("child exited before signalling readiness")
            if time.monotonic() > deadline:
                pytest.fail("child never reached its write loop")
            time.sleep(0.001)
        time.sleep(delay)
        assert process.is_alive(), "child finished before the kill; the delay is too long"
        process.kill()
    finally:
        process.join(timeout=TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == -signal.SIGKILL, (
        f"child exited with {process.exitcode}, not by SIGKILL; "
        "a child that exits cleanly makes the test vacuous"
    )


pytestmark = pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(), reason="requires process spawning"
)


def test_a_dead_writer_releases_the_owner_lock(tmp_path: Path) -> None:
    """A crashed writer must be a stall, not a lock-out.

    The kernel releases an ``flock`` when the holding process dies, so a crash
    can never leave the file permanently unopenable. Without this, one killed
    process would require manual intervention to clear.
    """
    path = tmp_path / "outbox.db"
    _run_child(_open_then_die, str(path), 0.0)

    with TansekiOutbox(path) as outbox:
        assert outbox.pending_count() == 0


def test_an_acknowledged_enqueue_survives_sigkill(tmp_path: Path) -> None:
    """A committed write is still there after the writer is killed outright.

    ``enqueue`` returns only after its transaction commits, so a caller that saw
    the return has been told the write succeeded. This is the assertion that
    makes that promise falsifiable, and it only holds because the outbox is
    opened with ``synchronous=FULL``.
    """
    path = tmp_path / "outbox.db"
    _run_child(_enqueue_then_die, str(path), ("e1", "e2", "e3"), 0.0)

    with TansekiOutbox(path) as outbox:
        assert {item.entry_id for item in outbox.pending()} == {"e1", "e2", "e3"}
        assert outbox.status_counts() == {"pending": 3, "retrying": 0, "dead_letter": 0}


def test_a_retry_after_a_lost_acknowledgement_is_a_duplicate(tmp_path: Path) -> None:
    """A producer that never learned its write succeeded must not duplicate it.

    The child committed and then died before reporting, which is precisely the
    ambiguity a timeout or a lost connection produces. The retry has to converge
    on one event, because the alternative is the same answer captured twice and
    posted to the pull request twice.
    """
    path = tmp_path / "outbox.db"
    _run_child(_enqueue_then_die, str(path), ("e1",), 0.0)

    with TansekiOutbox(path) as outbox:
        assert {item.entry_id for item in outbox.pending()} == {"e1"}

        outbox.enqueue("e1", {"id": "e1"})
        outbox.enqueue("e1", {"id": "e1"})

        assert outbox.pending_count() == 1
        assert outbox.status_counts() == {"pending": 1, "retrying": 0, "dead_letter": 0}


def test_a_killed_lease_holder_leaves_its_row_recoverable(tmp_path: Path) -> None:
    """A row leased by a dead process must be findable and deliverable.

    The lease is still nominally live when the process dies, so the row is
    correctly not deliverable yet. What matters is that nothing is lost: once
    the lease lapses the way it would have without the crash, the row is
    claimable and completes normally. A row that only existed inside a dead
    process's memory would be invisible to every operator command, which is the
    failure this guards against.
    """
    path = tmp_path / "outbox.db"
    _run_child(_enqueue_lease_then_die, str(path), "e1", 0.0)

    with TansekiOutbox(path) as outbox:
        assert outbox.status_counts()["pending"] == 1
        # The lease stands, so the row is not offered for delivery.
        assert outbox.ready() == []

        outbox._db.execute("UPDATE outbox SET lease_expires_at = ?", (EXPIRED,))
        outbox._db.commit()

        claimed = outbox.claim_due(limit=10, lease_seconds=60)
        assert [item.entry_id for item in claimed] == ["e1"]
        assert claimed[0].lease_token is not None
        assert outbox.mark_sent("e1", lease_token=claimed[0].lease_token)
        assert outbox.pending_count() == 0
        assert outbox.status_counts() == {"pending": 0, "retrying": 0, "dead_letter": 0}


def test_a_crash_between_the_registry_claim_and_the_write_is_reconcilable(
    tmp_path: Path,
) -> None:
    """The two local files must never disagree in a way nothing can act on.

    A question claimed and then abandoned, with no answer written, is the state a
    crash between the two writes leaves. It has to be reclaimable once the claim
    lapses, and the retried capture has to write exactly one outbox row -- the
    pair converging on a single answer rather than one per attempt.
    """
    registry_path = tmp_path / "registry.db"
    outbox_path = tmp_path / "outbox.db"
    _run_child(_claim_then_die, str(registry_path), 0.0)

    with SqliteQuestionRegistry(registry_path) as registry, TansekiOutbox(outbox_path) as outbox:
        claimed = registry.list_questions(status="claimed")
        assert len(claimed) == 1
        assert claimed[0]["question_id"] == "q1"
        assert claimed[0]["attempts"] == 1
        assert outbox.pending_count() == 0, "nothing was written before the crash"

        registry._db.execute("UPDATE questions SET lease_expires_at = ?", (EXPIRED,))
        registry._db.commit()

        assert registry.claim_question_for_answer("q1", "worker-2") is not None
        outbox.enqueue("answer-201", {"id": "answer-201"})
        outbox.enqueue("answer-201", {"id": "answer-201"})
        assert outbox.pending_count() == 1


@pytest.mark.parametrize("round_index", range(ROUNDS))
def test_a_kill_at_an_arbitrary_moment_leaves_a_consistent_outbox(
    tmp_path: Path, round_index: int
) -> None:
    """Whatever the kill interrupts, the survivors are coherent and deliverable.

    Each round kills the writer at a different offset from the moment it starts
    writing, while it is committing a long run of distinct entries. Only
    invariants are asserted:

    - the file is still a usable outbox, and a writer can finish it;
    - the survivors are a **dense prefix** -- ``e0..eN`` with nothing missing in
      the middle. A hole would mean a commit that was acknowledged to nobody and
      landed nowhere, which is the corruption this whole area is about;
    - nothing exists that was never enqueued;
    - every remaining row is in a state the operator can act on;
    - a fresh writer can drain all of it to completion.

    Deliberately *not* asserted: that the batch is all-or-none. Each ``enqueue``
    is its own transaction, so a kill mid-run legitimately leaves a prefix, and
    that is the correct behaviour rather than a gap.
    """
    path = tmp_path / "outbox.db"
    # The child writes until killed; this is the backstop against a runaway, set
    # far beyond anything the delay sweep can reach. `bound` is only the ceiling
    # the assertions read against, and is deliberately larger than the longest
    # delay could produce.
    ceiling = 200_000
    bound = 20_000
    delay = FIRST_DELAY + DELAY_STEP * round_index
    _kill_child_after(
        _hammer_forever, (str(path), ceiling, str(tmp_path / "ready")), delay, tmp_path / "ready"
    )

    with TansekiOutbox(path) as outbox:
        # `pending()` is bounded at 1000, and a child that writes until killed can
        # leave more survivors than that on a slow host. So the exact total is read
        # separately and the bounded read is only used for the prefix shape. The two
        # are compared, so a truncated read can never masquerade as a whole one.
        expected_total = outbox.pending_count()
        remaining = outbox.pending(limit=bound)
        ids = [item.entry_id for item in remaining]
        assert len(ids) == min(expected_total, 1_000), (
            "the bounded read did not return what the exact count says is there"
        )

        # Dense prefix, no holes: this is the assertion that fails if a commit
        # can be lost after it was acknowledged.
        for position, entry_id in enumerate(ids):
            assert entry_id == f"e{position}", (
                f"expected a dense prefix but found {entry_id} at position {position}; "
                f"survivors were {ids[:12]}..."
            )
        assert len(ids) == len(set(ids)), "a repeated enqueue produced a duplicate row"
        assert set(outbox.status_counts()) == {"pending", "retrying", "dead_letter"}

        # Expire anything the dead writer had leased, as time would have.
        outbox._db.execute(
            "UPDATE outbox SET lease_expires_at = ?, next_attempt_at = ?", (EXPIRED, EXPIRED)
        )
        outbox._db.commit()

        drained = 0
        while True:
            claimed = outbox.claim_due(limit=bound, lease_seconds=60)
            if not claimed:
                break
            for item in claimed:
                assert item.lease_token is not None
                assert outbox.mark_sent(item.entry_id, lease_token=item.lease_token)
                drained += 1

        # Against the exact count, not the bounded read: a truncated read would make
        # this pass while rows were left behind.
        assert drained == expected_total
        assert outbox.status_counts() == {"pending": 0, "retrying": 0, "dead_letter": 0}
        assert outbox.pending_count() == 0
