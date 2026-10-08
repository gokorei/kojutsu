"""A local log of read events, for the read path and nothing else.

A read event is not knowledge. Storing one in the Tanseki corpus would make the
store claim to contain something it does not — the same flattening as a
rationale recorded as an empty ``uncategorized`` row would be — and putting one in the SQLite registry would take a write lock on the
single-writer capture path to do it. So the log is a local file: an
operator-visible, append-only JSON Lines file that the knowledge store's contents
do not depend on and are unchanged by.

**What is recorded is the minimum.** Tool, the caller's own query, the filters
they stated, how many entries the answer carried, whether a bound or a filter
left anything out, the outcome, and the time. Not document bodies, not snippets,
not the evidence fence. The renderer in ``mcp_server/server.py`` is the only code
that can see that content and this module never receives it, so there is no
path by which stored text could reach a line of this file.

**There is no caller.** The MCP server is stdio with a single trust domain, so
nothing here can know who asked. What a caller's arguments say about the *read*
is recorded, in a field named as a claim — ``caller_claims`` — because a
repository name supplied by a caller is a statement by that caller and not an
authenticated fact about who it is. A field that read as an identity would be
the ``answered_by_model`` problem: trustworthy exactly as far as the claim, which
is nothing. Nothing in this module records an identity, and adding one is a
change this schema is deliberately shaped to make obvious.

**Refusals are recorded, not dropped.** A denied read and a read that found
nothing are both an absence from the log if refusals are not written, and they
are the two things an operator most needs to tell apart.

**Retention is a bound, not a preference.** Both bounds are enforced on every
recorded event, and anything removed is reported on the server's stderr. A
behavioural record that quietly shrinks is a record nobody can reason about
afterwards.

**Nothing here is measured.** This module counts entries and drops them. It
never compares one principal's reads to another's, and a consumer that wants to
does not get the vocabulary from here.

This module imports nothing from ``kojutsu``. That is deliberate rather than
incidental: a stdlib-only module cannot reach the knowledge store even by
accident, so the read path's structural read-only property survives whatever is
added next to it.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Stamped into every line so a consumer can tell this schema from a later one
#: instead of guessing from which keys happen to be present.
READ_EVENT_SCHEMA = "kojutsu.read-event.v1"

#: The recorded query is bounded well below the tool's own input bound. A search
#: accepts up to ``MAX_SEARCH_TEXT_CHARS`` (10 000) characters, and a read log
#: retained for days is a behavioural record: enough of the query to recognise
#: what was asked, not enough to reconstruct the prompt that carried it.
MAX_RECORDED_QUERY_CHARS = 200
_QUERY_TRUNCATED_SUFFIX = " [query truncated]"

#: Codes whose refusal is a policy decision, and codes refused at the input
#: edge. They are different events and the log must not fold them together: one
#: means kojutsu declined, the other means the caller sent something kojutsu
#: would not look at.
_DENIED_CODES = frozenset({"repository_not_authorized", "allowlist_invalid"})
_REJECTED_CODES = frozenset({"invalid_input", "repository_required"})


class ReadOutcome(StrEnum):
    """How one read ended, as a closed set.

    Closed on purpose: this is what an operator triages on, and a field that can
    hold any string is a field no query can rely on. Every terminal path of every
    read tool produces exactly one of these, so a refusal and a read that found
    nothing are never the same absence.
    """

    #: The store was asked and entries came back.
    SERVED = "served"
    #: The store was asked and no usable entry remained.
    NO_RESULTS = "no_results"
    #: The store held matches and the response budget carried none of them. Its
    #: own outcome because reporting this as ``no_results`` says the store was
    #: empty, which is the defect ``docs/design-review/read-path.md`` is about.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: Refused by policy: the repository was not authorized.
    DENIED = "denied"
    #: Refused at the input edge: an argument was missing, malformed, or too large.
    REJECTED = "rejected"
    #: There is no store to ask, so nothing was retrieved and nothing was denied.
    UNCONFIGURED = "unconfigured"
    #: The store or our own rendering failed.
    FAILED = "failed"


class ExclusionReason(StrEnum):
    """Why a matching entry was not in the answer.

    Closed for the same reason as :class:`ReadOutcome`. Each value names a
    different thing that went wrong, and ``response_budget`` is the one that
    makes a bounded answer visible as bounded.
    """

    #: The 100 000-character response budget could not fit it.
    RESPONSE_BUDGET = "response_budget"
    #: The store returned an entry belonging to another repository.
    CROSS_REPOSITORY = "cross_repository"
    #: The entry vanished between the search and the fetch.
    VANISHED = "vanished"
    #: The entry was below the independence threshold the caller asked for.
    BELOW_MIN_INDEPENDENCE = "below_min_independence"


class ReadLogError(RuntimeError):
    """The read log is not a file this process may write to."""


@dataclass(frozen=True)
class ReadEvent:
    """One recorded read.

    Every key is always present, including the empty ones. A stable key set is
    what lets a consumer write one query, and it is what makes a test able to pin
    the schema exactly — so adding a field is a deliberate act rather than a
    side effect of whatever a caller happened to pass.
    """

    tool: str
    outcome: ReadOutcome
    query: str
    query_truncated: bool
    caller_claims: Mapping[str, str]
    result_count: int
    excluded_count: int
    excluded_reasons: tuple[ExclusionReason, ...]
    truncated: bool
    error_code: str | None
    recorded_at: str
    schema: str = READ_EVENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Return the event as the JSON object written to one line."""
        return {
            "schema": self.schema,
            "recorded_at": self.recorded_at,
            "tool": self.tool,
            "outcome": self.outcome.value,
            "query": self.query,
            "query_truncated": self.query_truncated,
            "caller_claims": dict(self.caller_claims),
            "result_count": self.result_count,
            "excluded_count": self.excluded_count,
            "excluded_reasons": [reason.value for reason in self.excluded_reasons],
            "truncated": self.truncated,
            "error_code": self.error_code,
        }

    def to_line(self) -> str:
        """Render one JSON Lines record.

        ``ensure_ascii`` is on, as it is in the evidence renderer, for the reason
        ``docs/design-review/identity-and-limits.md`` gives: a query is text the
        caller wrote, and escaping it means a bidirectional control inside it
        cannot make a log line *display* as something it is not. It also means a
        query holding a character no UTF-8 file can hold is recorded rather than
        dropped, since the escape is pure ASCII.

        Parsing the line returns the caller's characters exactly as they were
        sent, escaped or not.
        """
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=False)


@dataclass
class ReadAccounting:
    """What one search delivered and what it left out.

    The read path fills this while it builds its answer; the log derives the
    outcome from it afterwards. Exclusions are counted per reason rather than as
    one running total, so ``excluded_count`` cannot disagree with the reasons it
    claims to be made of.
    """

    delivered: int = 0
    excluded: dict[ExclusionReason, int] = field(default_factory=dict)

    def delivered_one(self) -> None:
        self.delivered += 1

    def excluded_one(self, reason: ExclusionReason) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + 1

    @property
    def excluded_count(self) -> int:
        return sum(self.excluded.values())

    @property
    def excluded_reasons(self) -> tuple[ExclusionReason, ...]:
        return tuple(self.excluded)

    @property
    def truncated(self) -> bool:
        """Whether the response budget left at least one match out."""
        return ExclusionReason.RESPONSE_BUDGET in self.excluded

    @property
    def outcome(self) -> ReadOutcome:
        """Classify the answer the caller received.

        ``BUDGET_EXHAUSTED`` is reserved for the case where the only reason
        anything was left out is the response budget. When something was also
        dropped for a reason the caller chose (a threshold) or the store got
        wrong (a cross-repository document), the answer is ``no_results`` *with*
        the reasons recorded, and the reasons are what distinguish it from a
        genuinely empty store.
        """
        if self.delivered:
            return ReadOutcome.SERVED
        if tuple(self.excluded) == (ExclusionReason.RESPONSE_BUDGET,):
            return ReadOutcome.BUDGET_EXHAUSTED
        return ReadOutcome.NO_RESULTS


def outcome_for(code: str, accounting: ReadAccounting | None = None) -> ReadOutcome:
    """Classify one completed call from the code the caller received.

    Every non-``ok`` code maps to an outcome, and an unrecognised one maps to
    :attr:`ReadOutcome.FAILED` rather than to nothing. A future error code that
    nobody classified must still leave a trace; the cost of that fallback is that
    a new code is recorded as a failure until someone says otherwise, which is
    the cheaper mistake to make.
    """
    if code == "ok":
        return (accounting or ReadAccounting()).outcome
    if code in _DENIED_CODES:
        return ReadOutcome.DENIED
    if code == "tanseki_not_configured":
        return ReadOutcome.UNCONFIGURED
    if code in _REJECTED_CODES:
        return ReadOutcome.REJECTED
    if code == "not_found":
        return ReadOutcome.NO_RESULTS
    return ReadOutcome.FAILED


def bound_query(value: str | None) -> tuple[str, bool]:
    """Bound the caller's own text and say whether it was shortened.

    The suffix is not decoration: an answer shortened by a bound has to be
    visible as shortened, or a consumer reading the first 200 characters believes
    it is reading the whole query.
    """
    if value is None:
        return "", False
    if len(value) <= MAX_RECORDED_QUERY_CHARS:
        return value, False
    return f"{value[:MAX_RECORDED_QUERY_CHARS]}{_QUERY_TRUNCATED_SUFFIX}", True


def new_event(
    *,
    tool: str,
    code: str,
    query: str | None,
    caller_claims: Mapping[str, object] | None = None,
    accounting: ReadAccounting | None = None,
    now: datetime | None = None,
) -> ReadEvent:
    """Build one event from a completed call, without writing it anywhere.

    Claims are rendered as text. A limit of ``10`` and a limit of ``"10"`` are the
    same claim about the read, and the outcome already says which of them
    kojutsu accepted.
    """
    measured = accounting or ReadAccounting()
    bounded, truncated = bound_query(query)
    moment = now or datetime.now(UTC)
    return ReadEvent(
        tool=tool,
        outcome=outcome_for(code, measured),
        query=bounded,
        query_truncated=truncated,
        caller_claims={key: str(value) for key, value in (caller_claims or {}).items()},
        result_count=measured.delivered,
        excluded_count=measured.excluded_count,
        excluded_reasons=measured.excluded_reasons,
        truncated=measured.truncated,
        error_code=None if code == "ok" else code,
        recorded_at=moment.astimezone(UTC).isoformat(),
    )


def record_read(
    *,
    path: str | Path,
    tool: str,
    code: str,
    query: str | None = None,
    caller_claims: Mapping[str, object] | None = None,
    accounting: ReadAccounting | None = None,
    max_entries: int,
    max_age_days: int,
    now: datetime | None = None,
) -> ReadEvent:
    """Append one read event, then hold the file to both retention bounds.

    Returns the event whether or not it reached the disk, so a caller can see
    what was recorded. **A failed write never fails the read.** Telemetry that
    breaks the path it measures has made the store less available to get an
    observation out of it, which is the opposite of the point, so every error from
    writing is reported to this module's logger and the read proceeds.

    A bound below one is refused before anything is written, and raises. That is
    a programming error rather than a runtime condition, and a bound quietly
    treated as one would be a bound nobody set.
    """
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    if max_age_days < 1:
        raise ValueError("max_age_days must be at least 1")
    event = new_event(
        tool=tool,
        code=code,
        query=query,
        caller_claims=caller_claims,
        accounting=accounting,
        now=now,
    )
    target = Path(path).expanduser()
    recorded_at = event.recorded_at
    try:
        _append(target, event.to_line())
        _enforce_retention(
            target,
            max_entries=max_entries,
            max_age_days=max_age_days,
            now=recorded_at,
        )
    except (OSError, ReadLogError) as exc:
        logger.warning(
            "Could not record a %s read event in %s: %s. The read itself is "
            "unaffected; this log now has a gap.",
            tool,
            target,
            type(exc).__name__,
        )
    return event


def _verify_target(path: Path) -> None:
    """Refuse to touch a log that is not a regular file this user owns.

    A read log holds the text of what somebody asked for, so a path pointing at
    someone else's file — or at something that is not a file at all — is a place
    those questions would leak to. Appending is also opened without following a
    symlink, which covers the same ground before the file exists.
    """
    try:
        target_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(target_stat.st_mode):
        raise ReadLogError(f"read log is not a regular file: {path}")
    if hasattr(os, "geteuid") and target_stat.st_uid != os.geteuid():
        raise ReadLogError(f"read log is not owned by this user: {path}")


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _verify_target(path)
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        # Not fsynced, and deliberately so. This is a behavioural record, not
        # an acknowledged write: the outbox is where a crash must not lose
        # something, and a read that pays a disk flush on every call is a read
        # that can be made to fail. A crash can lose the last few events and the
        # log says so rather than implying a durability it does not have.
        os.write(descriptor, line.encode("utf-8") + b"\n")
    finally:
        os.close(descriptor)


def _parse_recorded_at(line: str) -> datetime | None:
    try:
        parsed = json.loads(line)
        return datetime.fromisoformat(str(parsed["recorded_at"]))
    except (ValueError, KeyError, TypeError):
        return None


def _enforce_retention(
    path: Path,
    *,
    max_entries: int,
    max_age_days: int,
    now: str,
) -> int:
    """Drop everything past either bound and report it. Returns how many went.

    The count bound keeps the newest ``max_entries`` lines in file order, which
    is chronological order because the file is only ever appended to. The age
    bound then drops any of those older than the cutoff, so a line that cannot be
    dated is dropped: an entry whose age cannot be shown to be inside the bound
    cannot be defended as inside it, and the removal is reported either way.

    Enforced on every recorded event rather than on a timer, which is what makes
    the bound checkable rather than aspirational — and the cost is bounded by
    ``max_entries``, so it stays in the low milliseconds at the default.

    ``now`` is the timestamp of the event just written, not the current clock, so
    a line is never judged against a clock that has moved since it was written.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0
    if not lines:
        return 0
    cutoff = datetime.fromisoformat(now) - timedelta(days=max_age_days)
    retained: list[str] = []
    dropped = 0
    for index, line in enumerate(lines):
        expired = False
        if line.strip():
            recorded_at = _parse_recorded_at(line)
            expired = recorded_at is None or recorded_at < cutoff
        else:
            expired = True
        if expired or index < len(lines) - max_entries:
            dropped += 1
            continue
        retained.append(line)
    if not dropped:
        return 0
    _rewrite(path, "\n".join(retained) + ("\n" if retained else ""))
    # Never silent. A log that shrinks without a word leaves an operator
    # reasoning about a gap they cannot see.
    logger.info(
        "read log retention removed %d of %d events (newest %d kept, older than "
        "%d day(s) dropped) from %s",
        dropped,
        len(lines),
        max_entries,
        max_age_days,
        path,
    )
    return dropped


def _rewrite(path: Path, text: str) -> None:
    """Replace the log in one step, or leave it as it was.

    Truncate-then-write would let a reader see an empty or half-written
    behavioural record, and a log that reads as empty reads as "nobody asked
    anything". The replacement is atomic and keeps the file's owner-only mode.
    """
    _verify_target(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
