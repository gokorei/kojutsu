"""Process-wide composition of capture dependencies.

The webhook server and CLI need the same collaborators: the Tanseki client, the
durable outbox, the capture sink, and the local question registry. This module
builds them once per process (so we hold a single SQLite connection each) and
exposes small helpers for relay and status.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from kojutsu.config import Settings, get_settings
from kojutsu.core.knowledge_sink import KnowledgeSink, TansekiKnowledgeSink
from kojutsu.core.outbox import RelayResult, TansekiOutbox, relay, retry_dead_letter
from kojutsu.core.question_projection import project_terminal_questions
from kojutsu.core.question_registry import QuestionRegistry, SqliteQuestionRegistry
from kojutsu.integrations.tanseki import TansekiClient

DEFAULT_RUNTIME_CLOSE_TIMEOUT = 5.0


@dataclass(frozen=True)
class QuestionProjectionResult:
    """What one projection sweep wrote, and what it cost the delivery path.

    Deliberately not a :class:`MaintenanceResult`. That type is a deletion report
    — ``total_deleted``, ``__int__``, ``__bool__`` — and reusing it for a
    projection would make a sweep that wrote three hundred documents report a
    truthy "total deleted" of zero, which is the sort of small lie that makes an
    operational surface untrustworthy.
    """

    questions_projected: int = 0
    outbox_pending_after: int = 0
    outbox_dead_letters_remaining: int = 0

    @property
    def backpressure(self) -> bool:
        """True when the sweep left work queued behind it.

        This is the number the ticket asked to be measured. A question is
        re-derivable from the registry; a captured answer is the only copy of a
        human's words. If the sweep is queueing work, it is spending the valuable
        thing to store the derivable one, and the sweep should be reconsidered
        rather than tuned.
        """
        return self.outbox_pending_after > 0

    def __int__(self) -> int:
        return self.questions_projected

    def __bool__(self) -> bool:
        """True when the sweep wrote something, or found dead letters.

        Mirrors :class:`MaintenanceResult` so an operator reading either surface
        gets the same question answered: did this do anything. Without it, every
        instance would be truthy and a caller that filters on the result would
        keep every sweep.
        """
        return self.questions_projected > 0 or self.outbox_dead_letters_remaining > 0


@dataclass(frozen=True)
class MaintenanceResult:
    registry_deliveries_deleted: int = 0
    outbox_dead_letters_deleted: int = 0
    outbox_dead_letters_remaining: int = 0

    @property
    def total_deleted(self) -> int:
        return self.registry_deliveries_deleted + self.outbox_dead_letters_deleted

    def __int__(self) -> int:
        return self.total_deleted

    def __bool__(self) -> bool:
        return self.total_deleted > 0 or self.outbox_dead_letters_remaining > 0


@dataclass
class Runtime:
    """The wired-up capture dependencies for one process."""

    settings: Settings
    client: TansekiClient
    outbox: TansekiOutbox
    sink: KnowledgeSink
    registry: QuestionRegistry
    _closed: bool = False
    _closing: bool = field(default=False, init=False, repr=False)
    _close_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _delivery_done: threading.Condition = field(init=False, repr=False)
    _active_deliveries: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._delivery_done = threading.Condition(self._close_lock)

    def _begin_delivery(self) -> None:
        with self._close_lock:
            if self._closed or self._closing:
                raise RuntimeError("Runtime is shutting down")
            self._active_deliveries += 1

    def _end_delivery(self) -> None:
        with self._close_lock:
            self._active_deliveries -= 1
            if self._active_deliveries == 0:
                self._delivery_done.notify_all()

    def relay(self, limit: int = 100) -> RelayResult:
        """Drain queued writes to Tanseki."""
        self._begin_delivery()
        try:
            return relay(self.outbox, self.client, limit=limit)
        finally:
            self._end_delivery()

    def retry(self, entry_id: str) -> RelayResult:
        """Requeue and immediately deliver one dead-lettered entry."""
        self._begin_delivery()
        try:
            return retry_dead_letter(self.outbox, self.client, entry_id)
        finally:
            self._end_delivery()

    def maintenance(self, *, cleanup_dead_letters: bool = False) -> MaintenanceResult:
        self._begin_delivery()
        try:
            registry_cleanup = getattr(self.registry, "cleanup_deliveries", None)
            registry_deleted = registry_cleanup() if callable(registry_cleanup) else 0
            outbox_deleted = 0
            if cleanup_dead_letters:
                outbox_cleanup = getattr(self.outbox, "cleanup_dead_letters", None)
                outbox_deleted = outbox_cleanup() if callable(outbox_cleanup) else 0
            status_counts = getattr(self.outbox, "status_counts", None)
            counts = status_counts() if callable(status_counts) else {}
            dead_letters_remaining = counts.get("dead_letter", 0) if isinstance(counts, dict) else 0
            return MaintenanceResult(
                registry_deliveries_deleted=(
                    registry_deleted if isinstance(registry_deleted, int) else 0
                ),
                outbox_dead_letters_deleted=(
                    outbox_deleted if isinstance(outbox_deleted, int) else 0
                ),
                outbox_dead_letters_remaining=(
                    dead_letters_remaining if isinstance(dead_letters_remaining, int) else 0
                ),
            )
        finally:
            self._end_delivery()

    def project_questions(self) -> QuestionProjectionResult:
        """Project terminal decision requests into the store, and report the cost.

        The ticket this implements said to measure the outbox queue depth before
        trusting a periodic sweep, because a question is re-derivable and an answer
        is not, and the two now share one delivery path. So the sweep reports the
        pending count *after* it ran, which is the number that decides whether it
        should keep running — a sweep that quietly delays captured answers has
        traded the valuable thing for the derivable one.

        Not called from ``maintenance``. This is a read of the registry plus a
        write to the store, and folding it in with the dead-letter housekeeping
        would make "we did some tidy-up" mean two quite different things.
        """
        self._begin_delivery()
        try:
            outcomes = project_terminal_questions(self.registry, self.sink)
            counts = self.outbox.status_counts()
            return QuestionProjectionResult(
                questions_projected=len(outcomes),
                outbox_pending_after=self.outbox.pending_count(),
                outbox_dead_letters_remaining=(
                    counts.get("dead_letter", 0) if isinstance(counts, dict) else 0
                ),
            )
        finally:
            self._end_delivery()

    def status(self) -> dict[str, Any]:
        """Operational status for health/status surfaces."""
        counts = self.outbox.status_counts()
        return {
            "tanseki_url": self.settings.tanseki_url,
            "tanseki_collection": self.settings.tanseki_collection,
            "tanseki_reachable": self.client.health(),
            "outbox_path": str(self.outbox.path),
            "outbox_pending": self.outbox.pending_count(),
            "outbox_captured_locally": counts["pending"],
            "outbox_retrying": counts["retrying"],
            "outbox_delivery_failed": self.outbox.delivery_failed_count(),
            "outbox_dead_letter": counts["dead_letter"],
            "registry_path": str(
                getattr(self.registry, "path", self.settings.kojutsu_registry_path)
            ),
        }

    def close(self, timeout: float = DEFAULT_RUNTIME_CLOSE_TIMEOUT) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            deadline = monotonic() + max(0.0, timeout)
            while self._active_deliveries:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._closing = False
                    raise TimeoutError("Timed out waiting for in-flight relay work")
                self._delivery_done.wait(timeout=remaining)
            first_error: Exception | None = None
            try:
                for resource in (self.outbox, self.registry, self.client):
                    close = getattr(resource, "close", None)
                    if not callable(close):
                        continue
                    try:
                        close()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
            finally:
                self._closed = True
                self._closing = False
            if first_error is not None:
                raise first_error

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def validate_sqlite_topology(settings: Settings) -> None:
    """Reject deployment configurations that would split local SQLite state."""
    if settings.kojutsu_sqlite_workers != 1 or settings.kojutsu_sqlite_replicas != 1:
        raise ValueError(
            "Kojutsu SQLite deployment supports exactly one process and one replica; "
            "use a shared transactional store before scaling out."
        )


def build_runtime(settings: Settings | None = None) -> Runtime:
    """Build a fresh runtime from configuration (raises if Tanseki is unconfigured)."""
    settings = settings or get_settings()
    validate_sqlite_topology(settings)
    if not settings.tanseki_enabled:
        raise ValueError("TANSEKI_URL is required: Tanseki is the knowledge store")
    client = TansekiClient.from_settings(settings)
    outbox: TansekiOutbox | None = None
    registry: QuestionRegistry | None = None
    try:
        outbox = TansekiOutbox(settings.tanseki_outbox_path)
        registry = SqliteQuestionRegistry(settings.kojutsu_registry_path)
        sink = TansekiKnowledgeSink(client, outbox)
        runtime = Runtime(
            settings=settings,
            client=client,
            outbox=outbox,
            sink=sink,
            registry=registry,
        )
        runtime.maintenance()
    except BaseException:
        for resource in (outbox, registry, client):
            if resource is None:
                continue
            with suppress(Exception):
                resource.close()
        raise
    else:
        return runtime


_runtime: Runtime | None = None
_runtime_lock = threading.RLock()


def get_runtime() -> Runtime:
    """Return the process-wide runtime, building it on first use."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = build_runtime()
        return _runtime


def reset_runtime() -> None:
    """Close and clear the cached runtime (used by tests and shutdown)."""
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
        if runtime is not None:
            try:
                runtime.close()
            except TimeoutError:
                _runtime = runtime
                raise
