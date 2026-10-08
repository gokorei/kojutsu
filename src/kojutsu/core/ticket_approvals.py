"""Plan approvals: gates, records, and the approval ledger.

Split out of ``ticket_drafts`` so approvals read without the ticket
sink: who approved a plan, through which gate, and what was decided is
one story; turning the approved plan into tickets is another. This
module also hosts the shared sqlite store plumbing both classes build
on (open/migrate/validate) -- one definition, imported by the sink side.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kojutsu.core.design_capture import ReconciliationOutcome
from kojutsu.core.gates import (
    DEFAULT_GATES,
    Gate,
    GateDecision,
    GatePredicate,
    LifecyclePoint,
)
from kojutsu.core.sqlite_durability import configure_durable_connection
from kojutsu.core.text_hygiene import REFUSED_RE
from kojutsu.identity import identity_preimage
from kojutsu.text_limits import MAX_AGENT_ID_CHARS

#: Version and domain of the approval derivation. A separate domain from the ticket's
#: because an approval and a ticket are different records about the same plan, and a
#: shared domain would let one namespace answer "is this the thing?" for both.
PLAN_APPROVAL_IDENTITY_VERSION = 1
PLAN_APPROVAL_IDENTITY_DOMAIN = "kojutsu.plan_approval.v1"

#: Version and domain of the gate-decision derivation, third namespace for the same
#: reason: the decision to create is not the decision that a ticket exists.
PLAN_GATE_DECISION_IDENTITY_VERSION = 1
PLAN_GATE_DECISION_IDENTITY_DOMAIN = "kojutsu.plan_gate_decision.v1"

MAX_APPROVAL_LIST_LIMIT = 1_000


#: Longest one approval note.
#:
#: Refused rather than truncated, because the note is the sentence a reader months later
#: uses to understand *why* this plan was allowed to become tickets, and a cut sentence
#: argues a different thing without saying so -- the argument
#: :data:`kojutsu.core.design_capture.MAX_DISCARD_REASON_CHARS` makes for the same field
#: on the reconciliation. The bound is generous because a person wrote this one.
MAX_APPROVAL_NOTE_CHARS = 2_000


@dataclass(frozen=True)
class PlanApproval:
    """One recorded human approval of one plan, with the provenance of who and when.

    ``plan_sha256`` is the anchor and the plan document deliberately is not here: an
    approval is about a digest, so re-reading it does not require re-reading a plan, and
    a plan object stored alongside it would invite a reader to believe the two are still
    the same bytes when only the digest is being asserted.

    ``approved_at`` is first-write-wins -- see :func:`record_plan_approval` -- because the
    approval id excludes the timestamp, so re-approving rewrites nothing and a
    re-approval cannot quietly move the moment the pipeline was authorised.
    """

    approval_id: str
    plan_sha256: str
    reconciliation_entry_id: str
    approved_by: str
    approved_at: datetime
    note: str | None = None


@dataclass(frozen=True)
class RecordedDecision:
    """One gate decision, as the ledger holds it after the gate emitted it.

    Distinct from :class:`~kojutsu.core.gates.GateDecision` on purpose. The emitted one
    is an in-memory verdict; this one is a row, and the difference shows up in
    :attr:`decided_at` and in :attr:`context`, neither of which the gate produces --
    a reader six months later is asking *when* this was said and *what the gate was
    looking at*, and an object that cannot answer either is not a record.
    """

    decision_id: str
    plan_sha256: str
    gate: str
    point: LifecyclePoint
    verdict: str
    reason: str
    decided_at: datetime
    approval_id: str | None = None
    context: dict[str, Any] | None = None


@runtime_checkable
class PlanApprovalLedger(Protocol):
    """Where a human's approval of a plan is recorded, and where gate decisions are kept.

    Separate from :class:`TicketSink` because a store that could both authorise and spend
    an authorisation could authorise itself; see the module docstring. Two protocols over
    two files, and the gap is the property.

    :meth:`record_decision` exists here rather than in the sink for the same reason. The
    gate decision is the evidence that the *authorisation* was consulted, so it belongs
    beside the authorisation rather than beside the tickets it released -- and a caller
    auditing "was this plan ever looked at by a human" should have to read exactly one
    store.
    """

    def get_approval(self, *, plan_sha256: str) -> PlanApproval | None: ...

    def record_approval(
        self,
        *,
        approval_id: str,
        plan_sha256: str,
        reconciliation_entry_id: str,
        approved_by: str,
        approved_at: datetime,
        note: str | None = None,
    ) -> PlanApproval: ...

    def list_approvals(
        self, *, plan_sha256: str, limit: int = MAX_APPROVAL_LIST_LIMIT
    ) -> list[PlanApproval]: ...

    def record_decision(
        self,
        *,
        decision: GateDecision,
        plan_sha256: str,
        approval_id: str | None = None,
        decided_at: datetime | None = None,
    ) -> RecordedDecision: ...

    def list_decisions(
        self, *, plan_sha256: str, limit: int = MAX_APPROVAL_LIST_LIMIT
    ) -> list[RecordedDecision]: ...


def stable_plan_approval_id(*, plan_sha256: str, approved_by: str) -> str:
    """Return the durable identity of one person's approval of one plan.

    The plan digest and the approver, and **not** the clock -- the same argument
    :func:`stable_ticket_id` makes, applied to the record that authorises rather than the
    record that spends it. Without it, re-running an approval would write a second row
    with a later timestamp and the ledger would hold two moments for one decision, so the
    answer to "when did a person approve this" would depend on how many times the operator
    clicked.

    The approver *is* in it, so a second person approving the same plan adds a row rather
    than overwriting the first. That is the honest shape: the two approvals are two events,
    and which one released the gate is answered by :attr:`RecordedDecision.approval_id`
    rather than by the presence of any approval at all.
    """
    if not plan_sha256.strip():
        raise ValueError("an approval is anchored on a plan digest; an empty one approves nothing")
    if not approved_by.strip():
        raise ValueError("an approval must name the person who gave it")
    return (
        f"plan-approval-v{PLAN_APPROVAL_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(PLAN_APPROVAL_IDENTITY_DOMAIN, (plan_sha256, approved_by))
        ).hexdigest()
    )


def stable_plan_gate_decision_id(*, plan_sha256: str, gate: str, approval_id: str | None) -> str:
    """Return the durable identity of one gate decision about one plan.

    Anchored on the plan, the gate, and **which approval was in hand when it was taken**,
    and that last component is what makes the record an idempotent one rather than a log.

    The obvious design -- a row per evaluation -- turns a retry loop into a ledger that
    grows with every attempt, and a reader counting blocks would be reading an
    implementation detail as though it were a fact about the plan. Anchored this way, the
    claim is precise and true: *the gate* ``gate`` *was taken about* this plan digest
    *with this approval in hand*. Re-evaluating that produces the same fact, so it is
    stored once. Before any approval exists ``approval_id`` is ``None``, and
    :func:`kojutsu.identity.identity_preimage` keeps that distinct from the empty
    string for free -- a decision taken with no approval is not a decision taken with an
    approval nobody recorded.
    """
    if not plan_sha256.strip():
        raise ValueError(
            "a gate decision is anchored on a plan digest; an empty one is about nothing"
        )
    if not gate.strip():
        raise ValueError("a gate decision must name the gate that produced it")
    return (
        f"plan-gate-decision-v{PLAN_GATE_DECISION_IDENTITY_VERSION}-"
        + hashlib.sha256(
            identity_preimage(PLAN_GATE_DECISION_IDENTITY_DOMAIN, (plan_sha256, gate, approval_id))
        ).hexdigest()
    )


def plan_approval_gates(*, approvals: PlanApprovalLedger) -> tuple[Gate, ...]:
    """Return :data:`~kojutsu.core.gates.DEFAULT_GATES` with the plan gate bound to evidence.

    The default ``plan-approval`` gate has a predicate that always returns its reason,
    because until now there was no approval to consult -- it is a declaration that *some*
    human-only transition belongs here, waiting for something to satisfy it. This function
    is what it was waiting for: the same gate, at the same point, with the same
    :data:`~kojutsu.core.gates.GateAction.BLOCK`, whose predicate now asks the ledger
    whether a person approved **this plan digest**.

    **The action is left alone, and that is the answer to whether this demotes anything.**
    Nothing is demoted. :data:`kojutsu.cli_worker.CAPTURE_POLICY` demotes to
    ``NOTIFY`` because a capture-only worker never proposes a plan, so requiring an
    approval it can never obtain would be a gate that blocks forever rather than a gate
    that blocks; that trade is correct there and wrong here, because this function's
    contract *is* "no tickets until a person says so". A caller who wants the relaxed
    arrangement passes a registry that says so -- the escape hatch is a policy the caller
    supplies, which is the same seam that policy already uses, rather than a convenience
    buried in the code that spends the approval.

    Only the ``PLAN_PROPOSED`` gate is rebuilt. The merge gate and the autonomous-capture
    gate are left exactly as :data:`~kojutsu.core.gates.DEFAULT_GATES` declares them,
    because replacing a whole policy set because one member of it needed an argument would
    be this module legislating about points it has nothing to do with.
    """
    return tuple(
        gate
        if gate.point is not LifecyclePoint.PLAN_PROPOSED
        else Gate(
            name=gate.name,
            point=gate.point,
            action=gate.action,
            predicate=_approved_plan_predicate(approvals),
        )
        for gate in DEFAULT_GATES
    )


def record_plan_approval(
    *,
    reconciliation: ReconciliationOutcome,
    approved_by: str,
    approvals: PlanApprovalLedger,
    note: str | None = None,
    approved_at: datetime | None = None,
) -> PlanApproval:
    """Record that a person approved this plan, and return the record as stored.

    The approval is anchored on :attr:`~kojutsu.core.design_capture.ReconciliationOutcome.plan_sha256`,
    which is what makes it *this* approval rather than a general willingness to proceed.
    A plan edited afterwards is a different digest, finds no approval, and is blocked --
    so an approval cannot be carried over to a document its approver never read.

    **Re-approving is a no-op and the first timestamp stands.**
    :func:`stable_plan_approval_id` excludes the clock, so the second call hits the same
    primary key and the stored row is returned unchanged. That is the same rule
    :meth:`SqliteQuestionRegistry.record_question` follows for ``head_sha``: a re-record
    must not relabel the moment an event happened, because the moment is the part of the
    provenance that exists to be believed. It also means this function is safe to call
    from a retry loop, which is what "a retried approval must not produce duplicate
    tickets" ends up needing at this seam.

    The reconciliation's own ``captured`` flag is not consulted, and the reason is worth
    writing down: ``captured=False`` is what a *successful* idempotent re-run looks like
    (see :func:`~kojutsu.core.design_capture.reconcile_design_proposals`), so refusing on
    it would refuse the second run of every correct pipeline.

    Neither is :attr:`~kojutsu.core.design_capture.ReconciliationOutcome.plan` re-digested
    here. Phase two computes the digest from the plan it parsed, so the two agree by
    construction, and a mismatch is not something this function can resolve -- it would only
    be able to complain about the caller's outcome object. The comparison that *can* settle
    it is made where the plan is handed over instead, in
    :func:`create_tickets_from_plan`.

    ``approved_by`` is refused rather than normalised when blank or carrying a refused
    character, for :func:`kojutsu.core.design_capture._check_principal`'s reasons: this is
    the one field in the whole design phase whose entire job is to name a person, and a
    blank or invisible-glyph spelling of one is a name a reader will trust as the wrong
    thing. That check is a copy rather than an import because the original is private to
    another phase's file; if it changes there, change it here.
    """
    _check_principal(approved_by)
    _check_note(note)
    return approvals.record_approval(
        approval_id=stable_plan_approval_id(
            plan_sha256=reconciliation.plan_sha256, approved_by=approved_by
        ),
        plan_sha256=reconciliation.plan_sha256,
        reconciliation_entry_id=reconciliation.entry_id,
        approved_by=approved_by,
        approved_at=approved_at or datetime.now(UTC),
        note=note,
    )


def _approved_plan_predicate(approvals: PlanApprovalLedger) -> GatePredicate:
    """Return a predicate that blocks until this plan digest has a recorded approval.

    The predicate is what the gate hands a reason to, and the reason is read by a person
    deciding whether to go and approve something -- so it names the digest and says what
    is missing. ``None`` means "does not apply", which is the registry's way of saying the
    gate has nothing to say, and it is returned here **only** when an approval exists.

    A context without a plan digest is a **block**, not a pass. The tempting reading is
    that a gate which cannot tell what it is looking at has nothing to object to; the
    consequence of that reading is that forgetting a field in the context turns the
    repository's strongest default off, which is a flag-shaped failure in a system whose
    point is that the gate is not a flag.
    """

    def predicate(context: dict[str, Any]) -> str | None:
        digest = context.get("plan_sha256")
        if not isinstance(digest, str) or not digest.strip():
            return (
                "no plan digest was presented, so no recorded approval can be matched to "
                "this plan and the gate cannot confirm a person approved it"
            )
        if approvals.get_approval(plan_sha256=digest) is None:
            return (
                f"plan {digest} has no recorded human approval; a reconciled plan must be "
                "approved by a person before it becomes tickets"
            )
        return None

    return predicate


def _check_principal(principal: str) -> None:
    """Refuse a blank, unbounded or invisible-glyph approval principal.

    A copy of :func:`kojutsu.core.design_capture._check_principal` rather than an import,
    because that one is private to a file this phase does not own. Duplicated policy is a
    real cost, and the alternative -- reaching into another module's underscore -- is a
    worse one: the next change to the rule would be applied to proposals and not to
    approvals, which is exactly the drift the single-definition argument in
    :mod:`kojutsu.identity` exists to prevent. If the original changes, change this.

    The bound is :data:`kojutsu.integrations.github.MAX_AGENT_ID_CHARS` for the reason
    that one is: a principal is a machine-readable identifier reaching a stored provenance
    field, and two definitions of its length is two answers that will eventually disagree.
    Refused characters are refused and not removed, because removing an invisible glyph
    from a name would let two approvers who differ only by one derive the same approval id
    and lose one of the two records -- the argument
    :func:`kojutsu.core.design_capture._refuse_refused_characters` makes.
    """
    if not isinstance(principal, str) or not principal.strip():
        raise ValueError(
            "an approval must name the person who gave it; a blank one is a record of an "
            "approval by nobody, which is the one field here whose whole job is to name a "
            "person"
        )
    if len(principal) > MAX_AGENT_ID_CHARS:
        raise ValueError(
            f"an approver must be at most {MAX_AGENT_ID_CHARS} characters, got {len(principal)}"
        )
    _refuse_refused_characters(principal, "approver")


def _check_note(note: str | None) -> None:
    """Refuse an approval note past :data:`MAX_APPROVAL_NOTE_CHARS`, and say why.

    Refused rather than truncated because the note is the sentence a later reader uses to
    understand why the plan was allowed to become tickets, and a cut sentence argues a
    different thing without saying so. Nothing is sanitised here: :func:`_check_principal`
    is the only field that reaches an *identity*, so a note -- which does not -- is stored
    as written and the store's character policy applies to it like any other stored prose.
    A note is never *required*, so a caller who has nothing to add passes ``None`` rather
    than an empty string, which would be a claim that something was written.
    """
    if note is None:
        return
    if not isinstance(note, str):
        raise ValueError(f"an approval note must be a string or None, got {type(note).__name__}")
    if len(note) > MAX_APPROVAL_NOTE_CHARS:
        raise ValueError(
            f"an approval note must be at most {MAX_APPROVAL_NOTE_CHARS} characters, got "
            f"{len(note)}; it is refused rather than truncated, because it is the sentence a "
            "later reader uses to understand why this plan became tickets"
        )


def _refuse_refused_characters(value: str, what: str) -> None:
    """Raise naming the code point if ``value`` carries a character the policy refuses.

    Shared with :func:`_check_principal` because both fields it checks are hashed into an
    identity, and that is the only property that decides refused-versus-removed: a value
    that reaches a digest must never be altered, because two values differing only by an
    invisible glyph are two records a reader can see and one record the store cannot.
    """
    found = REFUSED_RE.search(value)
    if found is not None:
        raise ValueError(
            f"a {what} must not contain U+{ord(found.group()):04X}, which this project refuses "
            f"to store; it is refused rather than removed because an approver is part of an "
            f"approval's identity, and removing a character from it would let two {what}s "
            "differing only by an invisible glyph share one approval record"
        )


def _now() -> str:
    """The stored form of a timestamp, so every row's moment is written the same way."""
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: object) -> datetime:
    """Read a stored ISO-8601 timestamp back, naming the column when it cannot.

    A bare ``ValueError`` from :meth:`datetime.fromisoformat` says ``Invalid isoformat
    string`` and nothing about which row or which field, which is the least actionable
    sentence available for a corrupted local store.
    """
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise RuntimeError(f"a stored timestamp could not be read back: {value!r} ({exc})") from exc


#: Schema v1 of the approval ledger: the approvals themselves, and the gate decisions.
#:
#: **Two tables, and not one.** A row that could be either "a person approved this" or "the
#: gate was consulted about this" would make every read a disambiguation, and the failure
#: mode of guessing wrong is reading a block as an approval. Separate tables cannot
#: disagree about which kind of record a row is.
#:
#: The unique index on ``(plan_sha256, approved_by)`` is the semantic identity and the
#: primary key is the derived one; both are needed, for
#: :meth:`SqliteQuestionRegistry.claim_rationale`'s reason. If the derivation ever moves,
#: the index is what notices that two different ids now describe one approval.
_MIGRATION_V1_APPROVALS: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS design_plan_approvals ("
    "approval_id TEXT PRIMARY KEY, "
    "plan_sha256 TEXT NOT NULL, "
    "reconciliation_entry_id TEXT NOT NULL, "
    "approved_by TEXT NOT NULL, "
    "approved_at TEXT NOT NULL, "
    "note TEXT)",
    "CREATE UNIQUE INDEX IF NOT EXISTS design_plan_approvals_identity_idx "
    "ON design_plan_approvals (plan_sha256, approved_by)",
    "CREATE INDEX IF NOT EXISTS design_plan_approvals_plan_idx "
    "ON design_plan_approvals (plan_sha256)",
    "CREATE TABLE IF NOT EXISTS design_gate_decisions ("
    "decision_id TEXT PRIMARY KEY, "
    "plan_sha256 TEXT NOT NULL, "
    "gate TEXT NOT NULL, "
    "point TEXT NOT NULL, "
    "verdict TEXT NOT NULL, "
    "reason TEXT NOT NULL, "
    "decided_at TEXT NOT NULL, "
    "approval_id TEXT, "
    "context TEXT NOT NULL DEFAULT '{}')",
    "CREATE INDEX IF NOT EXISTS design_gate_decisions_plan_idx "
    "ON design_gate_decisions (plan_sha256)",
)


#: Migrations recorded in each store's ledger and therefore checksum-verified on every open.
#:
#: **Two dictionaries, not one, and that is the point.** Each store owns its own tables, so
#: each carries its own version-1 statements. If both shared one tuple then version 1 would
#: create all four tables, either store could open either file, and the separation the two
#: Protocols exist to provide would be a naming convention rather than a property of the
#: schema. With two, opening the wrong store on the right file is a checksum mismatch and a
#: refusal -- which is also the honest failure for a caller who misconfigures two paths onto
#: one file.
#:
#: **No data repairs, and that is the point of these being version 1.** Every repair in
#: :data:`kojutsu.core.question_registry._MIGRATION_V4` exists because rows written by an
#: older application need information this one computes differently. A store that has never
#: existed cannot have rows that disagree with anything, so the first migration is pure
#: structure -- which is also why it has to be *forward-only*: at v1 there is nothing to
#: go back to, and every later version is additive.
_APPROVAL_MIGRATIONS: dict[int, tuple[str, ...]] = {1: _MIGRATION_V1_APPROVALS}

#: The schema version this application implements, for both stores. Owned here,
#: beside the generic ``_migrate`` that enforces it, and imported by the ticket
#: side: a version is shared plumbing, not sink state.
SCHEMA_VERSION = 1


#: Name of the migration ledger, in both stores.
#:
#: It is *not* ``registry_migrations``: that name belongs to
#: :mod:`kojutsu.core.question_registry`, and two applications sharing one ledger table
#: would each refuse the other's versions as unreadable history -- which is the correct
#: outcome, but it should arrive as a checksum mismatch about two known migrations rather
#: than as one application silently applying its statements to the other's file.
_MIGRATION_LEDGER = "design_store_migrations"


#: The columns each store's tables are expected to have, checked on every open.
#:
#: A second copy of the DDL, and deliberately so: the migration tuple is what gets applied
#: and checksummed, and nothing re-reads it to check that an existing file still matches.
#: Without this a file could be missing a column -- opened by an older application, or
#: truncated -- and the failure would surface as a ``KeyError`` on a row read rather than
#: as a refusal at open. The check fails closed and the drift it catches is always in the
#: safe direction: a schema this application does not recognise is refused, not guessed at.
_APPROVAL_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "design_plan_approvals": (
        "approval_id",
        "plan_sha256",
        "reconciliation_entry_id",
        "approved_by",
        "approved_at",
        "note",
    ),
    "design_gate_decisions": (
        "decision_id",
        "plan_sha256",
        "gate",
        "point",
        "verdict",
        "reason",
        "decided_at",
        "approval_id",
        "context",
    ),
}


def _open_store(
    path: str | Path,
    *,
    migrations: dict[int, tuple[str, ...]],
    expected_columns: dict[str, tuple[str, ...]],
    what: str,
) -> tuple[sqlite3.Connection, threading.RLock]:
    """Open one store's SQLite file: private, durable, migrated, or not opened at all.

    The file handling is :meth:`SqliteQuestionRegistry.__init__`'s, inlined rather than
    shared, and the cost of that duplication is named rather than hidden: it keeps this
    store's schema independent of the registry's, which is worth a copy. Sharing one file
    would mean two owners of ``PRAGMA user_version`` in one database, and each would read
    the other's version and migrate against tables it does not know -- a corruption that
    only shows up after the damage. Two stores, two files, one pragma each.

    ``migrations`` and ``expected_columns`` are arguments rather than module constants
    because each store owns a different schema; see :data:`_APPROVAL_MIGRATIONS` for why
    that is not one shared tuple. ``what`` names the store in the failure messages, so an
    operator reading "does not match this application" learns which half of the design phase
    is unhappy rather than having to work it out from a table name.

    Everything the pragma pass can be given up on is refused instead
    (:mod:`kojutsu.core.sqlite_durability`), because the alternative is a store that
    acknowledges a ticket write before it is on disk -- and this store's entire guarantee
    is that a ticket which was reported as created exists after a crash.
    """
    resolved = Path(path).expanduser()
    parent_created = not resolved.parent.exists()
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent_created:
        resolved.parent.chmod(0o700)
    _secure_store_file(resolved)
    db = sqlite3.connect(str(resolved), check_same_thread=False, timeout=5.0)
    lock = threading.RLock()
    try:
        configure_durable_connection(db)
        _migrate(db, lock, migrations=migrations, expected_columns=expected_columns, what=what)
    except Exception:
        db.close()
        raise
    return db, lock


def _secure_store_file(path: Path) -> None:
    """Open the store's file to check it is a regular file this user owns, mode 0600.

    Identical in intent to :meth:`SqliteQuestionRegistry._secure_database_file`: the file
    holds the approvals, and an approval is a record of what a person decided, so a store
    that another user or a symlink can write is a store whose provenance is worth nothing.
    """
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise PermissionError("Design store path must be a regular file")
        if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
            raise PermissionError("Design store file must be owned by the current user")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _migrate(
    db: sqlite3.Connection,
    lock: threading.RLock,
    *,
    migrations: dict[int, tuple[str, ...]],
    expected_columns: dict[str, tuple[str, ...]],
    what: str,
) -> None:
    """Bring the file to :data:`SCHEMA_VERSION` in one transaction, or leave it alone.

    A file already at the current version is only *validated*: it is not re-migrated, so a
    second process opening an already-migrated store does no writes at open. Everything
    else runs under ``BEGIN IMMEDIATE`` so two processes racing to create the same store
    produce one migration and one refusal, not two half-applied ones.

    ``what`` is the store's name, carried into every refusal. A file that holds one store's
    history and is opened as the other fails the ledger comparison, and the message says
    which store was being opened rather than leaving that to a table name.
    """
    with lock:
        try:
            db.execute("BEGIN IMMEDIATE")
            version_row = db.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row else 0
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"{what} schema is newer than this application")
            if version == SCHEMA_VERSION:
                _validate_migration_ledger(db, migrations=migrations, what=what)
                _validate_tables(db, expected_columns=expected_columns, what=what)
                db.rollback()
                return
            for step, statements in sorted(migrations.items()):
                _record_migration(db, step, statements, what=what)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            _validate_migration_ledger(db, migrations=migrations, what=what)
            _validate_tables(db, expected_columns=expected_columns, what=what)
            db.commit()
        except Exception:
            db.rollback()
            raise


def _migration_checksum(statements: tuple[str, ...]) -> str:
    """Checksum a migration's statements, ignoring incidental whitespace.

    The same canonicalisation :func:`kojutsu.core.question_registry._migration_checksum`
    uses. It exists so that a *stored* migration cannot be quietly rewritten: a checksum is
    only evidence of intent if the intent is fixed at the moment it is written, which means
    an edit to a statement in a released version has to show up as a mismatch on every open
    rather than being applied to the next database silently.
    """
    canonical = "\n".join(" ".join(statement.split()) for statement in statements)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _apply_statement(db: sqlite3.Connection, statement: str) -> None:
    """Apply one migration statement, refusing any form that is not re-runnable.

    A forward-only migration that has been interrupted half-way must be safe to resume,
    and the only way to guarantee that is to reject a statement whose repeatability nobody
    has argued for. ``CREATE ... IF NOT EXISTS`` is re-runnable by construction; an
    ``ADD COLUMN`` is skipped when the column is already there, which is the guard
    :func:`kojutsu.core.question_registry._apply_statement` uses for the same reason.
    Anything else -- a data repair, a drop, a rename -- is refused rather than guessed at,
    because this store has no rows to repair and a future version that needs one has to say
    so in a statement whose repeatability can be checked.
    """
    normalised = " ".join(statement.split())
    head = normalised.upper()
    if head.startswith("ALTER TABLE"):
        target, separator, addition = normalised.partition(" ADD COLUMN ")
        if not separator:
            raise RuntimeError(f"Migration statement is not a column addition: {normalised}")
        if addition.split()[0] in _table_columns(db, target[len("ALTER TABLE ") :].strip()):
            return
    elif head.startswith("CREATE "):
        if "IF NOT EXISTS" not in head:
            raise RuntimeError(f"Migration statement is not idempotent: {normalised}")
    else:
        raise RuntimeError(f"Migration statement is not a supported form: {normalised}")
    db.execute(statement)


def _record_migration(
    db: sqlite3.Connection, version: int, statements: tuple[str, ...], *, what: str
) -> None:
    """Apply one forward-only step and record its checksum; refuse a rewritten one.

    A step already in the ledger is not re-applied but its checksum is still compared, for
    :func:`kojutsu.core.question_registry._record_migration`'s reason: a store whose
    recorded history disagrees with this application was rewritten behind our back, and
    continuing from it could apply half of a migration and call it done.

    The comparison is also what catches the wrong store on the right file. Two stores, two
    version-1 tuples, and one file holding the other store's version 1: the checksums differ,
    so the refusal says which history was found and which was expected rather than applying
    this store's statements to tables it does not own.
    """
    checksum = _migration_checksum(statements)
    db.execute(
        f"CREATE TABLE IF NOT EXISTS {_MIGRATION_LEDGER} ("
        "version INTEGER PRIMARY KEY, "
        "checksum TEXT NOT NULL, "
        "applied_at TEXT NOT NULL)"
    )
    row = db.execute(
        f"SELECT checksum FROM {_MIGRATION_LEDGER} WHERE version = ?", (version,)
    ).fetchone()
    if row is not None:
        if str(row[0]) != checksum:
            raise RuntimeError(
                f"{what} migration {version} was recorded with checksum {row[0]}, but "
                f"this application implements it as {checksum}. The applied history does not "
                "match."
            )
        return
    for statement in statements:
        _apply_statement(db, statement)
    db.execute(
        f"INSERT INTO {_MIGRATION_LEDGER} (version, checksum, applied_at) VALUES (?, ?, ?)",
        (version, checksum, _now()),
    )


def _validate_migration_ledger(
    db: sqlite3.Connection, *, migrations: dict[int, tuple[str, ...]], what: str
) -> None:
    """Fail closed when the recorded history is missing, rewritten, or from the future.

    Every open, not only the migrating one. A store that was migrated by this application
    and has since been edited by another should say so at open, where an operator is
    present, rather than at the first write that hits a column which is no longer there.
    """
    recorded = {
        int(row[0]): str(row[1])
        for row in db.execute(f"SELECT version, checksum FROM {_MIGRATION_LEDGER}")
    }
    for version, statements in migrations.items():
        expected = _migration_checksum(statements)
        if version not in recorded:
            raise RuntimeError(f"{what} migration {version} is not recorded in the ledger")
        if recorded[version] != expected:
            raise RuntimeError(
                f"{what} migration {version} was recorded with checksum "
                f"{recorded[version]}, but this application implements it as {expected}. "
                "The applied history does not match."
            )
    for version in recorded:
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"Design store migration {version} is newer than this application")


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    """Return ``table``'s column names, or an empty set when the table is not there.

    An absent table reads as "no columns" rather than raising, because the only caller that
    can legitimately be asked about a table that does not exist yet is the migration adding
    it, and that is exactly the case where "already present" must be false.
    """
    try:
        rows = db.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _validate_tables(
    db: sqlite3.Connection, *, expected_columns: dict[str, tuple[str, ...]], what: str
) -> None:
    """Refuse a file whose tables are not the ones this store expects.

    Naming every missing column, not the first one, for
    :class:`kojutsu.core.design_plan.DesignPlanError`'s reason at a smaller scale: a caller
    whose file is one version behind should learn the whole distance in one run.
    """
    missing: list[str] = []
    for table, columns in expected_columns.items():
        present = _table_columns(db, table)
        if not present:
            missing.append(f"table {table!r} is absent")
            continue
        missing.extend(
            f"table {table!r} has no column {column!r}"
            for column in columns
            if column not in present
        )
    if missing:
        raise RuntimeError(
            f"{what} schema does not match this application (version "
            f"{SCHEMA_VERSION}): " + "; ".join(missing)
        )


class SqlitePlanApprovalLedger:
    """A local, durable :class:`PlanApprovalLedger`.

    One table for the approvals and one for the gate decisions, in one transaction per
    write, and no lease columns -- deliberately. Every other capture table in
    :mod:`kojutsu.core.question_registry` carries a claim token and an expiry, because its
    write is followed by a *delivery* that can fail and must be retried. Nothing here
    delivers: a record is either written or the transaction did not commit. A lease would
    be a way for one process to block another from recording that a person approved a plan,
    which is the opposite of what this table is for.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._db, self._lock = _open_store(
            self.path,
            migrations=_APPROVAL_MIGRATIONS,
            expected_columns=_APPROVAL_TABLE_COLUMNS,
            what="Plan approval store",
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> SqlitePlanApprovalLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_approval(self, *, plan_sha256: str) -> PlanApproval | None:
        """Return the first approval recorded for this plan digest, or ``None``.

        "First" by the recorded order rather than the identity order, because two people
        approving the same plan is two events and the gate asks a yes/no question. Which
        one released the gate is answered by :attr:`RecordedDecision.approval_id`, and a
        reader asking who approved a plan is served by :meth:`list_approvals`.
        """
        with self._lock:
            row = self._db.execute(
                f"SELECT {_APPROVAL_COLUMNS} FROM design_plan_approvals WHERE plan_sha256 = ? "
                "ORDER BY approved_at ASC, approval_id ASC LIMIT 1",
                (plan_sha256,),
            ).fetchone()
        return None if row is None else _approval_from_row(row)

    def record_approval(
        self,
        *,
        approval_id: str,
        plan_sha256: str,
        reconciliation_entry_id: str,
        approved_by: str,
        approved_at: datetime,
        note: str | None = None,
    ) -> PlanApproval:
        """Write the approval if it is new, and return the stored row either way.

        First-write-wins, and the check afterwards is not decoration. The insertion can
        fail because the id is taken *or* because the semantic index over
        ``(plan_sha256, approved_by)`` is taken, and those two mean different things:
        the first is the idempotent re-approval this function exists for, and the second
        means the derivation moved and two ids now describe one approval. The second is
        refused, on :meth:`SqliteQuestionRegistry.claim_rationale`'s argument -- continuing
        would store a record about an approval that the ledger already holds under another
        identity, which is precisely the silent re-identification this codebase spends
        derivations preventing.
        """
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                inserted = _insert_approval(
                    self._db,
                    approval_id=approval_id,
                    plan_sha256=plan_sha256,
                    reconciliation_entry_id=reconciliation_entry_id,
                    approved_by=approved_by,
                    approved_at=approved_at,
                    note=note,
                )
                if inserted:
                    self._db.commit()
                    return PlanApproval(
                        approval_id=approval_id,
                        plan_sha256=plan_sha256,
                        reconciliation_entry_id=reconciliation_entry_id,
                        approved_by=approved_by,
                        approved_at=approved_at,
                        note=note,
                    )
                existing = _read_approval(self._db, approval_id=approval_id)
                self._db.rollback()
            except Exception:
                self._db.rollback()
                raise
        if existing is not None and existing.plan_sha256 == plan_sha256:
            if existing.approved_by == approved_by:
                return existing
            raise RuntimeError(
                f"approval id {approval_id!r} is on file for plan {existing.plan_sha256!r} "
                f"approved by {existing.approved_by!r}, and this call is for the same id "
                f"approved by {approved_by!r}. Two approvals cannot share one identity; "
                "refusing rather than letting the derivation pick a winner."
            )
        raise RuntimeError(
            f"approval {approval_id!r} for plan {plan_sha256!r} was refused by a constraint "
            "and no row describes it. Either the derivation moved, so this approval now "
            "matches one already on file under another id, or the row is unreadable; both "
            "are refused rather than stored as a second approval of the same decision."
        )

    def list_approvals(
        self, *, plan_sha256: str, limit: int = MAX_APPROVAL_LIST_LIMIT
    ) -> list[PlanApproval]:
        """Every approval recorded for one plan digest, oldest first."""
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_APPROVAL_COLUMNS} FROM design_plan_approvals WHERE plan_sha256 = ? "
                "ORDER BY approved_at ASC, approval_id ASC LIMIT ?",
                (plan_sha256, _bounded_limit(limit)),
            ).fetchall()
        return [_approval_from_row(row) for row in rows]

    def record_decision(
        self,
        *,
        decision: GateDecision,
        plan_sha256: str,
        approval_id: str | None = None,
        decided_at: datetime | None = None,
    ) -> RecordedDecision:
        """Store one emitted gate decision, or return the one already stored for it.

        Idempotent by :func:`stable_plan_gate_decision_id`, so a retry loop records the
        decision once. The emitted :class:`~kojutsu.core.gates.GateDecision` is stored
        whole -- including its ``context`` -- rather than the fields this store happens to
        have columns for, so a policy change that starts putting something new in the
        context is visible in the ledger without a migration.
        """
        decision_id = stable_plan_gate_decision_id(
            plan_sha256=plan_sha256, gate=decision.gate, approval_id=approval_id
        )
        moment = decided_at or datetime.now(UTC)
        context = json.dumps(decision.context, separators=(",", ":"), default=str)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                try:
                    self._db.execute(
                        "INSERT INTO design_gate_decisions (decision_id, plan_sha256, gate, "
                        "point, verdict, reason, decided_at, approval_id, context) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            decision_id,
                            plan_sha256,
                            decision.gate,
                            str(decision.point),
                            str(decision.verdict),
                            decision.reason,
                            moment.astimezone(UTC).isoformat(),
                            approval_id,
                            context,
                        ),
                    )
                except sqlite3.IntegrityError:
                    row = self._db.execute(
                        f"SELECT {_DECISION_COLUMNS} FROM design_gate_decisions "
                        "WHERE decision_id = ?",
                        (decision_id,),
                    ).fetchone()
                    if row is None:
                        self._db.rollback()
                        raise RuntimeError(
                            f"gate decision {decision_id!r} collides with a row this store "
                            "cannot read back"
                        ) from None
                    self._db.commit()
                    return _decision_from_row(row)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return RecordedDecision(
            decision_id=decision_id,
            plan_sha256=plan_sha256,
            gate=decision.gate,
            point=decision.point,
            verdict=str(decision.verdict),
            reason=decision.reason,
            decided_at=moment,
            approval_id=approval_id,
            context=dict(decision.context),
        )

    def list_decisions(
        self, *, plan_sha256: str, limit: int = MAX_APPROVAL_LIST_LIMIT
    ) -> list[RecordedDecision]:
        """Every gate decision recorded about one plan digest, oldest first.

        This is the query requirement two and requirement one are answered with: it returns
        the block that stopped the pipeline as well as the allow that released it, so
        "did a person approve this, and when" and "what did the gate say before that" are
        the same read.
        """
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_DECISION_COLUMNS} FROM design_gate_decisions WHERE plan_sha256 = ? "
                "ORDER BY decided_at ASC, decision_id ASC LIMIT ?",
                (plan_sha256, _bounded_limit(limit)),
            ).fetchall()
        return [_decision_from_row(row) for row in rows]


def _approval_from_row(row: tuple[Any, ...]) -> PlanApproval:
    """Rebuild an approval from a stored row."""
    return PlanApproval(
        approval_id=str(row[0]),
        plan_sha256=str(row[1]),
        reconciliation_entry_id=str(row[2]),
        approved_by=str(row[3]),
        approved_at=_parse_timestamp(row[4]),
        note=None if row[5] is None else str(row[5]),
    )


#: The columns every approval read asks for, in the order :func:`_approval_from_row` reads
#: them. Named because four call sites spell this list and a fifth one added with a
#: forgotten column would be a read that fails only on the rows it happens to touch.
_APPROVAL_COLUMNS = (
    "approval_id, plan_sha256, reconciliation_entry_id, approved_by, approved_at, note"
)


#: The columns every decision read asks for, in :func:`_decision_from_row`'s order. Named
#: for :data:`_APPROVAL_COLUMNS`'s reason.
_DECISION_COLUMNS = (
    "decision_id, plan_sha256, gate, point, verdict, reason, decided_at, approval_id, context"
)


def _read_approval(db: sqlite3.Connection, *, approval_id: str) -> PlanApproval | None:
    """Read one approval by its derived id, or ``None`` when no row carries it."""
    row = db.execute(
        f"SELECT {_APPROVAL_COLUMNS} FROM design_plan_approvals WHERE approval_id = ?",
        (approval_id,),
    ).fetchone()
    return None if row is None else _approval_from_row(row)


def _insert_approval(
    db: sqlite3.Connection,
    *,
    approval_id: str,
    plan_sha256: str,
    reconciliation_entry_id: str,
    approved_by: str,
    approved_at: datetime,
    note: str | None,
) -> bool:
    """Insert one approval row, returning ``False`` if a constraint already holds one.

    ``False`` covers both possible refusals -- the derived primary key and the semantic
    index over ``(plan_sha256, approved_by)`` -- and the caller tells them apart by reading
    the row back and refusing an unreadable result. That is deliberate rather than lazy:
    a fresh ``sqlite3.IntegrityError`` raised to the caller would name a constraint and
    nothing about whether the caller's approval was a legitimate re-approval, which is the
    question the caller actually has.
    """
    try:
        db.execute(
            "INSERT INTO design_plan_approvals (approval_id, plan_sha256, "
            "reconciliation_entry_id, approved_by, approved_at, note) VALUES (?, ?, ?, ?, ?, ?)",
            (
                approval_id,
                plan_sha256,
                reconciliation_entry_id,
                approved_by,
                approved_at.astimezone(UTC).isoformat(),
                note,
            ),
        )
    except sqlite3.IntegrityError:
        return False
    return True


def _decision_from_row(row: tuple[Any, ...]) -> RecordedDecision:
    """Rebuild a recorded gate decision from a stored row."""
    try:
        context = json.loads(str(row[8]))
    except (TypeError, ValueError):
        context = {}
    return RecordedDecision(
        decision_id=str(row[0]),
        plan_sha256=str(row[1]),
        gate=str(row[2]),
        point=LifecyclePoint(str(row[3])),
        verdict=str(row[4]),
        reason=str(row[5]),
        decided_at=_parse_timestamp(row[6]),
        approval_id=None if row[7] is None else str(row[7]),
        context=context if isinstance(context, dict) else None,
    )


def _bounded_limit(limit: int) -> int:
    """Clamp a read limit into ``[0, MAX_APPROVAL_LIST_LIMIT]``.

    Clamped rather than refused, for :meth:`SqliteQuestionRegistry._bounded_question_limit`'s
    reason: a limit is a hint about how much to read, so the worst a caller can do by asking
    for too much is read too little, and the worst it can do by asking for something
    unusable is an exception where a bounded answer would have done. The clamp is reported
    nowhere, which is why the bound is a constant rather than a per-call parameter.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"a limit must be an integer, got {limit!r}")
    return max(0, min(limit, MAX_APPROVAL_LIST_LIMIT))
