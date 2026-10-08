"""What the loop must remember when the process goes away.

Everything here exists because of one failure: a worker that is killed mid-cycle
loses its in-memory bookkeeping and, on restart, starts the cycle again from the
top. That re-runs ``implement``, which opens a *second* pull request for the same
ticket, and it resets the attempt counter, so the claim ceiling never fires and the
crash loop is unbounded. The tests for the loop all inject exceptions, which are
handled; a crash is not an exception and was escaping entirely.

So the loop's own state -- attempts, position, what each step produced, who is
claimed, what is dead, what is blocked -- is written to one file, atomically, and
read back on start-up.

One file, written whole, rather than an append log: this is mutable bookkeeping,
not a record of events, and a log of bookkeeping needs a compaction story nobody
needs. The dead-letter entries are the one part a human reads directly, so they
are written out as a separate ``dead-letters.jsonl`` beside it. Both are derived
from the same in-memory truth on every write, so they cannot disagree.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

STATE_VERSION = 1


class WorkerStateOwnershipError(RuntimeError):
    """A second worker owns this state file.

    Two writers sharing one ``worker-state.json`` last-writer-win every field:
    attempts counted by one vanish when the other rewrites, and the claim
    ceiling never fires. Refusing loudly at startup is the alternative to an
    unbounded crash loop that looks like one worker making no progress.
    """


@dataclass
class _StateOwnerRecord:
    descriptor: int
    references: int


class _StateOwnerRegistry:
    """One writer per state path, across processes and within them.

    The lock lives on a ``<name>.lock`` sidecar rather than the state file
    itself, because the state file is replaced on every write and a lock does
    not survive ``os.replace`` -- the sidecar's inode is stable, so the lock
    outlives any number of state replacements. Non-blocking: a second worker
    fails fast with :class:`WorkerStateOwnershipError` instead of queueing
    behind the first and looking, from the outside, like one slow worker.
    In-process re-entry shares the record by reference count, so a loop and its
    status view holding the same path do not deadlock themselves.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _owners: ClassVar[dict[Path, _StateOwnerRecord]] = {}

    @classmethod
    def _lock_path(cls, path: Path) -> Path:
        return path.with_name(f"{path.name}.lock")

    @classmethod
    def acquire(cls, path: Path) -> None:
        key = path.resolve(strict=False)
        with cls._lock:
            current = cls._owners.get(key)
            if current is not None:
                current.references += 1
                return
            lock_path = cls._lock_path(path)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                os.close(descriptor)
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise WorkerStateOwnershipError(
                        f"Worker state is already owned by another process: {path}"
                    ) from None
                raise
            cls._owners[key] = _StateOwnerRecord(descriptor=descriptor, references=1)

    @classmethod
    def release(cls, path: Path) -> None:
        key = path.resolve(strict=False)
        with cls._lock:
            current = cls._owners.get(key)
            if current is None:
                return
            current.references -= 1
            if current.references > 0:
                return
            cls._owners.pop(key, None)
            try:
                fcntl.flock(current.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(current.descriptor)


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so a crash leaves the previous intact.

    UTF-8 throughout, the file fsynced before the rename and the directory
    fsynced after it: without the first a crash can leave a truncated file in
    place of the old one, and without the second the rename itself may not
    survive. Same shape as the queue writer in ``sources.py``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    descriptor = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    tmp.replace(path)
    dir_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_descriptor)
    finally:
        os.close(dir_descriptor)


@dataclass(frozen=True)
class ClaimRecord:
    """A lease this worker is holding, and which process took it."""

    item_id: str
    claim_token: str
    boot: str
    repo: str = ""
    branch: str = ""
    pr_number: int | None = None
    since: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "claim_token": self.claim_token,
            "boot": self.boot,
            "repo": self.repo,
            "branch": self.branch,
            "pr_number": self.pr_number,
            "since": self.since,
        }


@dataclass
class WorkerState:
    """Durable loop bookkeeping, or an in-memory one when no path is configured.

    Everything is best-effort when ``path`` is ``None``: the loop still works, it
    just forgets. A caller that wants crash safety has to give it a path, and
    :attr:`durable` says plainly whether it got one rather than leaving an
    operator to guess from the absence of an error.
    """

    path: Path | None = None
    boot: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _attempts: dict[str, int] = field(default_factory=dict)
    _position: dict[str, str] = field(default_factory=dict)
    _produced: dict[str, dict[str, str]] = field(default_factory=dict)
    _claims: dict[str, ClaimRecord] = field(default_factory=dict)
    _dead: dict[str, dict[str, Any]] = field(default_factory=dict)
    _blocked: dict[str, dict[str, Any]] = field(default_factory=dict)
    _write_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.path is None:
            return
        # Owned before anything is read: a loser never even sees the file, so
        # two writers cannot diverge from a shared starting point.
        _StateOwnerRegistry.acquire(self.path)
        if self.path.exists():
            self._load()

    def close(self) -> None:
        """Release ownership of the state path. Idempotent.

        Called by the worker's own ``close``; safe to skip for short-lived
        readers that never contend, since the registry also releases on
        collection -- but an explicit close is what lets a second worker start
        promptly rather than whenever the garbage collector gets around to it.
        """
        if self._closed or self.path is None:
            return
        self._closed = True
        _StateOwnerRegistry.release(self.path)

    def __enter__(self) -> WorkerState:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort release only
        with contextlib.suppress(Exception):
            self.close()

    # -- durability ---------------------------------------------------------

    @property
    def durable(self) -> bool:
        return self.path is not None

    @property
    def dead_letter_path(self) -> Path | None:
        """The human-readable dead-letter file, guaranteed distinct from the state.

        A caller can name the state file anything, including
        ``dead-letters.jsonl``. If the two collided, whichever wrote last would
        silently overwrite the other and the state would be unreadable on restart,
        so the collision is made impossible rather than documented.
        """
        if self.path is None:
            return None
        candidate = self.path.with_name("dead-letters.jsonl")
        if candidate == self.path:
            return self.path.with_name(f"{self.path.name}.dead-letters")
        return candidate

    def _load(self) -> None:
        if self.path is None:
            raise RuntimeError("WorkerState._load called without a path")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt state file must not stop the worker from starting. Losing
            # attempt counts is bad; refusing to run at all is worse, because
            # nothing is left to fix it. Dead letters are re-read from their own
            # file, which is append-shaped and survives truncation better.
            return
        if raw.get("version") != STATE_VERSION:
            return
        self._attempts = {str(k): int(v) for k, v in raw.get("attempts", {}).items()}
        self._position = {str(k): str(v) for k, v in raw.get("position", {}).items()}
        self._produced = {str(k): dict(v) for k, v in raw.get("produced", {}).items()}
        self._claims = {str(k): ClaimRecord(**v) for k, v in raw.get("claims", {}).items()}
        self._blocked = {str(k): dict(v) for k, v in raw.get("blocked", {}).items()}
        # Dead letters are re-read from the append-shaped file, so a half-written
        # state file cannot lose the record an operator needs.
        self._dead = self._read_dead_letters()

    def _write(self) -> None:
        if self.path is None:
            return
        with self._write_lock:
            payload = {
                "version": STATE_VERSION,
                "attempts": self._attempts,
                "position": self._position,
                "produced": self._produced,
                "claims": {k: v.as_dict() for k, v in self._claims.items()},
                "blocked": self._blocked,
            }
            # Atomic: a worker killed mid-write leaves the previous state intact
            # rather than a truncated file that reads as "no state at all".
            _write_text_atomic(self.path, json.dumps(payload, indent=2, sort_keys=True))
            self._write_dead_letters()

    def _read_dead_letters(self) -> dict[str, dict[str, Any]]:
        target = self.dead_letter_path
        if target is None or not target.exists():
            return {}
        entries: dict[str, dict[str, Any]] = {}
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("item_id"):
                entries[str(entry["item_id"])] = entry
        return entries

    def _write_dead_letters(self) -> None:
        target = self.dead_letter_path
        if target is None:
            return
        if not self._dead:
            # Leave no file behind on a healthy run. An empty dead-letters.jsonl
            # reads as "the operator cleared these" rather than "nothing died".
            target.unlink(missing_ok=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(
            json.dumps(self._dead[item_id], sort_keys=True) + "\n" for item_id in sorted(self._dead)
        )
        _write_text_atomic(target, lines)

    # -- attempts -----------------------------------------------------------

    def attempts(self, item_id: str) -> int:
        return self._attempts.get(item_id, 0)

    def record_attempt(self, item_id: str) -> int:
        """Count one attempt and return the new total.

        Durable, because the ceiling is only a bound if it survives a restart. An
        in-memory counter bounds failures inside a process and does nothing about
        a process that keeps being killed, which is the case that actually loops.
        """
        self._attempts[item_id] = self._attempts.get(item_id, 0) + 1
        self._write()
        return self._attempts[item_id]

    def attempt_counts(self) -> dict[str, int]:
        return dict(self._attempts)

    def clear_attempts(self, item_id: str) -> None:
        if self._attempts.pop(item_id, None) is not None:
            self._write()

    # -- position and produced values ---------------------------------------

    def position(self, item_id: str) -> str | None:
        """The last step this item completed, if the item is part-way through."""
        return self._position.get(item_id)

    def record_position(self, item_id: str, step: str) -> None:
        self._position[item_id] = step
        self._write()

    def record_step(self, item_id: str, step: str, value: str) -> None:
        """Record that ``step`` completed and what it produced, in one write.

        Position and output travel together because a crash between two writes
        leaves a state that resumes *past* a step whose output was never stored:
        the cycle skips work it cannot see the result of. The attempt counter
        stays separate deliberately -- it counts cycles started, recorded before
        any step runs, while position counts steps completed. One write is what
        makes "resumed after X" and "holding X's output" the same fact.
        """
        self._position[item_id] = step
        self._produced.setdefault(item_id, {})[step] = value
        self._write()

    def record_produced(self, item_id: str, step: str, value: str) -> None:
        self._produced.setdefault(item_id, {})[step] = value
        self._write()

    def produced(self, item_id: str) -> dict[str, str]:
        return dict(self._produced.get(item_id, {}))

    def positions(self) -> dict[str, str]:
        """Every item part-way through a cycle, and the step it reached."""
        return dict(self._position)

    def clear_progress(self, item_id: str) -> None:
        """Forget position and produced values; called when a cycle completes."""
        changed = self._position.pop(item_id, None) is not None
        if self._produced.pop(item_id, None) is not None:
            changed = True
        if changed:
            self._write()

    # -- claims -------------------------------------------------------------

    def record_claim(
        self,
        item_id: str,
        claim_token: str,
        *,
        repo: str = "",
        branch: str = "",
        pr_number: int | None = None,
        since: str = "",
    ) -> ClaimRecord:
        record = ClaimRecord(
            item_id=item_id,
            claim_token=claim_token,
            boot=self.boot,
            repo=repo,
            branch=branch,
            pr_number=pr_number,
            since=since,
        )
        self._claims[item_id] = record
        self._write()
        return record

    def claims(self) -> dict[str, ClaimRecord]:
        return dict(self._claims)

    def orphans(self) -> dict[str, ClaimRecord]:
        """Claims held by a process that is not this one.

        After a crash these are leases this worker still believes it holds. The
        ticket system will expire them on its own schedule, but an operator needs
        to see them now: they are the difference between "recovering" and
        "silently waiting for a timeout".
        """
        return {k: v for k, v in self._claims.items() if v.boot != self.boot}

    def release_claim(self, item_id: str) -> None:
        if self._claims.pop(item_id, None) is not None:
            self._write()

    # -- dead letters -------------------------------------------------------

    def dead_entries(self) -> list[dict[str, Any]]:
        return [dict(entry) for _, entry in sorted(self._dead.items())]

    def dead_ids(self) -> set[str]:
        return set(self._dead)

    def dead_letter(self, item_id: str, *, error: str, attempts: int) -> None:
        self._dead[item_id] = {
            "item_id": item_id,
            "error": error,
            "attempts": attempts,
        }
        self._attempts.pop(item_id, None)
        self._position.pop(item_id, None)
        self._produced.pop(item_id, None)
        self._blocked.pop(item_id, None)
        self._claims.pop(item_id, None)
        self._write()

    def requeue(self, item_id: str) -> bool:
        """Return a dead-lettered item to the queue, durably.

        The only way an exhausted item comes back. A requeue that lived only in
        memory would be undone by the next restart, putting the item straight back
        into the dead set.
        """
        if item_id not in self._dead:
            return False
        self._dead.pop(item_id, None)
        self._attempts.pop(item_id, None)
        self._write()
        return True

    # -- blocked ------------------------------------------------------------

    def block(self, item_id: str, *, reason: str, gate: str = "") -> None:
        """Record that a gate is holding this item.

        Not a failure and not dead: a human may approve it later, so the loop must
        keep asking. But it is recorded rather than discarded, because a worker that
        has been blocked on every item for a day looks exactly like a worker with
        nothing to do, and that is the one state an operator must never have to
        guess about.
        """
        self._blocked[item_id] = {
            "reason": reason,
            "gate": gate,
            "attempts": self.attempts(item_id),
        }
        self._write()

    def blocked(self) -> dict[str, dict[str, Any]]:
        return dict(self._blocked)

    def clear_block(self, item_id: str) -> None:
        if self._blocked.pop(item_id, None) is not None:
            self._write()
