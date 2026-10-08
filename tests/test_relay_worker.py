"""Tests for the background outbox relay worker."""

from __future__ import annotations

import asyncio
import contextlib

from kojutsu import relay_worker
from kojutsu import runtime as runtime_module
from kojutsu.core.outbox import RelayResult
from kojutsu.relay_worker import relay_once


def test_relay_once_drains_and_reports_sent(monkeypatch) -> None:
    calls: list[int] = []

    class FakeRuntime:
        def relay(self, limit: int = 100):
            calls.append(limit)
            return RelayResult(sent=2, failed=1, dead_lettered=0)

    monkeypatch.setattr(runtime_module, "get_runtime", lambda: FakeRuntime())

    result = asyncio.run(relay_once())
    assert result.sent == 2
    assert result.failed == 1
    assert result.dead_lettered == 0
    assert calls == [100]


def test_shutdown_waits_for_tasks_created_during_drain() -> None:
    """A relay scheduled mid-drain must hold shutdown open, not slip through.

    A snapshot implementation returns as soon as the tasks it saw finish, so a
    relay starting during the wait would still hold SQLite when shutdown
    proceeds to ``reset_runtime``. The drain loops until the set is empty.
    """

    async def main() -> None:
        event_a = asyncio.Event()
        event_b = asyncio.Event()
        tracked: list[asyncio.Task[None]] = []
        try:
            task_a = asyncio.create_task(event_a.wait())
            tracked.append(task_a)
            relay_worker._active_relay_tasks.add(task_a)
            task_a.add_done_callback(relay_worker._active_relay_tasks.discard)
            waiter = asyncio.create_task(relay_worker.wait_for_relay_shutdown(timeout_seconds=5.0))
            await asyncio.sleep(0.05)
            assert not waiter.done()
            task_b = asyncio.create_task(event_b.wait())
            tracked.append(task_b)
            relay_worker._active_relay_tasks.add(task_b)
            task_b.add_done_callback(relay_worker._active_relay_tasks.discard)
            event_a.set()
            await asyncio.sleep(0.05)
            assert not waiter.done(), "drain must cover tasks created during the drain"
            assert not task_b.done()
            event_b.set()
            assert await waiter is True
        finally:
            for task in tracked:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tracked, return_exceptions=True)

    asyncio.run(main())
    assert not relay_worker._active_relay_tasks


def test_relay_loop_counts_unexpected_failures(monkeypatch) -> None:
    """The loop swallows relay exceptions to keep draining, so it counts them:
    without a counter a relay that fails every iteration looks like a quiet one."""

    async def main() -> None:
        async def boom() -> int:
            raise RuntimeError("relay exploded")

        monkeypatch.setattr(relay_worker, "relay_once", boom)
        before_total, _ = relay_worker.relay_error_counts()
        task = asyncio.create_task(relay_worker.relay_loop(0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        total, consecutive = relay_worker.relay_error_counts()
        assert total > before_total
        assert consecutive >= 1

    asyncio.run(main())
