"""Background relay that drains the Tanseki outbox.

The webhook server captures decisions locally first; this worker periodically
flushes queued writes to Tanseki, so an outage self-heals without a restart.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any

from kojutsu import runtime as runtime_module
from kojutsu.core.outbox import RelayResult

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 30.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_active_relay_tasks: set[asyncio.Task[Any]] = set()

#: How many relay iterations have failed unexpectedly, in total and in a row.
#: Logged with each failure and exposed via :func:`relay_error_counts` so a
#: health check can tell a quiet outbox (nothing to send) from a broken one
#: (sends keep raising). ``relay_loop`` swallows the exception to keep draining
#: -- a relay that dies on the first poison row never delivers the valid rows
#: behind it -- so without a counter that swallow would be silent.
_relay_error_total: int = 0
_relay_consecutive_errors: int = 0


def relay_error_counts() -> tuple[int, int]:
    """Return ``(total_errors, consecutive_errors)`` for the relay loop."""
    return (_relay_error_total, _relay_consecutive_errors)


def resolve_relay_interval(interval: float | None = None) -> float:
    raw: str | float
    if interval is None:
        raw = os.getenv("KOJUTSU_RELAY_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS))
    else:
        raw = interval
    if isinstance(raw, bool):
        raise ValueError("KOJUTSU_RELAY_INTERVAL_SECONDS must be a finite positive number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "KOJUTSU_RELAY_INTERVAL_SECONDS must be a finite positive number"
        ) from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("KOJUTSU_RELAY_INTERVAL_SECONDS must be a finite positive number")
    return value


async def relay_once() -> RelayResult:
    """Drain the outbox once; return what the drain did.

    The full result, not just the sent count: a relay that sends nothing
    because every row failed looks exactly like an idle one on a bare count,
    and the loop's maintenance decision and the operator's log line both need
    the failures visible.
    """
    runtime = runtime_module.get_runtime()
    task = asyncio.create_task(asyncio.to_thread(runtime.relay))
    _active_relay_tasks.add(task)
    task.add_done_callback(_active_relay_tasks.discard)
    # Shielded deliberately, not defensively. Cancelling the waiter must not
    # cancel this task: the thread it tracks keeps running either way (a
    # cancelled ``to_thread`` await does not stop the thread), but an
    # unshielded cancel would mark the *task* done while the thread still
    # holds SQLite -- dropping it from ``_active_relay_tasks`` and letting
    # shutdown ``reset_runtime`` (close the database) underneath live work.
    # Shielded, the task stays tracked until the thread actually finishes, so
    # :func:`wait_for_relay_shutdown` waits for the work rather than for the
    # cancellation, and shutdown defers the reset while anything is outstanding.
    result = await asyncio.shield(task)
    if result.sent or result.failed or result.dead_lettered:
        logger.info(
            "Outbox relay: sent=%s failed=%s dead_lettered=%s",
            result.sent,
            result.failed,
            result.dead_lettered,
        )
    return result


async def wait_for_relay_shutdown(
    timeout_seconds: float | None = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
) -> bool:
    """Wait until no relay task is outstanding, or the timeout expires.

    Loops rather than snapshotting: a relay started *during* the drain (the
    loop schedules the next ``relay_once`` as soon as the previous finishes)
    joins the set while this waits, and a snapshot taken before it was created
    would return "drained" while it still holds SQLite. Returns ``False`` on
    expiry so the caller defers ``reset_runtime`` instead of closing the
    database underneath live work.
    """
    loop = asyncio.get_running_loop()
    deadline: float | None = None
    if timeout_seconds is not None:
        deadline = loop.time() + max(0.0, timeout_seconds)
    while True:
        tasks = tuple(_active_relay_tasks)
        if not tasks:
            return True
        remaining: float | None = None
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
        await asyncio.wait(tasks, timeout=remaining)


async def relay_loop(interval: float | None = None) -> None:
    """Periodically drain the outbox until cancelled."""
    global _relay_error_total, _relay_consecutive_errors
    interval = resolve_relay_interval(interval)
    while True:
        await asyncio.sleep(interval)
        try:
            await relay_once()
            _relay_consecutive_errors = 0
            runtime = runtime_module.get_runtime()
            maintenance = getattr(runtime, "maintenance", None)
            if callable(maintenance):
                result = await asyncio.to_thread(maintenance)
                if result:
                    logger.info(
                        "Outbox maintenance: registry_deleted=%s dead_letters_deleted=%s "
                        "dead_letters_remaining=%s",
                        getattr(result, "registry_deliveries_deleted", 0),
                        getattr(result, "outbox_dead_letters_deleted", 0),
                        getattr(result, "outbox_dead_letters_remaining", 0),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _relay_error_total += 1
            _relay_consecutive_errors += 1
            logger.exception(
                "Outbox relay failed unexpectedly: %s (total=%d consecutive=%d)",
                type(exc).__name__,
                _relay_error_total,
                _relay_consecutive_errors,
            )
