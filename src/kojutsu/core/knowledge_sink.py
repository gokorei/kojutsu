"""Where captured knowledge entries are written.

Captured knowledge always goes to the Tanseki knowledge store, via the durable
outbox so an outage never loses a decision. Local state consists of the durable
outbox and the question registry (see ``core.question_registry``).

Composition lives in :mod:`kojutsu.runtime`; this module provides the sink
interfaces and the Tanseki implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, cast, runtime_checkable

from kojutsu.core.outbox import TansekiOutbox
from kojutsu.core.tanseki_mapping import (
    to_census_upsert_payload,
    to_clarification_upsert_payload,
    to_question_upsert_payload,
    to_rationale_upsert_payload,
    to_upsert_payload,
)
from kojutsu.integrations.tanseki import TansekiError, TansekiPermanentError, TansekiWriter
from kojutsu.models import (
    CensusRecord,
    ClarificationEntry,
    KnowledgeEntry,
    QuestionRecord,
    RationaleEntry,
)

#: What the sink accepts. A stated decision rationale, a projected decision request,
#: a captured clarification and an observed non-event are all different *kinds* of
#: record — different trust, different lifetime, different document shape — and are
#: kept that way deliberately. What they are not is a different *delivery*: all of
#: them go through the one outbox, under the one entry id, so a second queue for any
#: of them would be a second thing that can lose a record. The relay that retries
#: and dead-letters a failure sees every one of them, which is the only way a failed
#: delivery is ever noticed.
StorableRecord: TypeAlias = (
    KnowledgeEntry | RationaleEntry | ClarificationEntry | QuestionRecord | CensusRecord
)


def to_payload(record: StorableRecord) -> dict[str, object]:
    """Render one stored record into the Tanseki upsert payload it needs.

    Dispatch on type rather than on a flag, so a new record kind has to be added
    here to be delivered at all. A rationale that fell through to the entry
    renderer would be written as a question and an answer it never had, a
    clarification would be written as a question that was never asked, and a
    census record that fell through to either would be written as knowledge
    nobody produced.
    """
    match record:
        case RationaleEntry():
            return to_rationale_upsert_payload(record)
        case ClarificationEntry():
            return to_clarification_upsert_payload(record)
        case QuestionRecord():
            return to_question_upsert_payload(record)
        case CensusRecord():
            return to_census_upsert_payload(record)
        case _:
            return to_upsert_payload(record)


class KnowledgeDeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    QUEUED = "queued"
    DEAD_LETTERED = "dead_lettered"


@dataclass(frozen=True)
class KnowledgeDeliveryOutcome:
    entry_id: str
    status: KnowledgeDeliveryStatus
    detail: str | None = None
    retry_after: float | None = None

    @property
    def delivered(self) -> bool:
        return self.status is KnowledgeDeliveryStatus.DELIVERED

    @property
    def queued(self) -> bool:
        return self.status is KnowledgeDeliveryStatus.QUEUED

    @property
    def dead_lettered(self) -> bool:
        return self.status is KnowledgeDeliveryStatus.DEAD_LETTERED

    @property
    def retryable(self) -> bool:
        return self.status is not KnowledgeDeliveryStatus.DELIVERED


def delivered(entry_id: str) -> KnowledgeDeliveryOutcome:
    return KnowledgeDeliveryOutcome(entry_id=entry_id, status=KnowledgeDeliveryStatus.DELIVERED)


@runtime_checkable
class _IdentifiedRecord(Protocol):
    """The one thing every stored record has, whatever kind it is.

    A union of pydantic models hides the attribute they share, so the outbox
    key is read through this instead. It is the record's durable identity, and the
    outbox is keyed on it — so the field is load-bearing, not a convenience.
    """

    entry_id: str


@runtime_checkable
class KnowledgeSink(Protocol):
    """A destination for captured knowledge.

    Accepts every record kind because all of them are captured knowledge and all of
    them must reach the store. They are distinguished by the document they become,
    never by a second delivery path.
    """

    def store(self, entry: StorableRecord) -> KnowledgeDeliveryOutcome | None: ...


class TansekiKnowledgeSink:
    """Write entries to Tanseki via the durable outbox."""

    def __init__(self, client: TansekiWriter, outbox: TansekiOutbox) -> None:
        self._client = client
        self._outbox = outbox

    def store(self, entry: StorableRecord) -> KnowledgeDeliveryOutcome:
        payload = to_payload(entry)
        identified = cast(_IdentifiedRecord, entry)
        entry_id = identified.entry_id
        if not self._outbox.enqueue(entry_id, payload):
            return KnowledgeDeliveryOutcome(
                entry_id=entry_id,
                status=KnowledgeDeliveryStatus.QUEUED,
                detail="An active delivery lease already owns this entry",
            )
        claim = self._outbox.claim_entry(entry_id)
        if claim is None or claim.lease_token is None:
            return KnowledgeDeliveryOutcome(
                entry_id=entry_id,
                status=KnowledgeDeliveryStatus.QUEUED,
                detail="Entry is durably queued for relay",
            )
        failure_detail: str | None = None
        failure_retry_after: float | None = None
        try:
            self._client.upsert_document(claim.payload)
        except TansekiPermanentError as exc:
            failure_detail = str(exc)
            failure_retry_after = getattr(exc, "retry_after", None)
            status = self._outbox.mark_failed(
                entry_id,
                exc,
                retry_after=getattr(exc, "retry_after", None),
                lease_token=claim.lease_token,
            )
            if status == "dead_letter":
                return KnowledgeDeliveryOutcome(
                    entry_id=entry_id,
                    status=KnowledgeDeliveryStatus.DEAD_LETTERED,
                    detail=failure_detail,
                    retry_after=failure_retry_after,
                )
        except TansekiError as exc:
            failure_detail = str(exc)
            failure_retry_after = getattr(exc, "retry_after", None)
            status = self._outbox.mark_failed(
                entry_id,
                exc,
                retry_after=getattr(exc, "retry_after", None),
                lease_token=claim.lease_token,
            )
            if status == "dead_letter":
                return KnowledgeDeliveryOutcome(
                    entry_id=entry_id,
                    status=KnowledgeDeliveryStatus.DEAD_LETTERED,
                    detail=failure_detail,
                    retry_after=failure_retry_after,
                )
        except BaseException:
            self._outbox.release_claim(entry_id, lease_token=claim.lease_token)
            raise
        else:
            if self._outbox.mark_sent(entry_id, lease_token=claim.lease_token):
                return delivered(entry_id)
            return KnowledgeDeliveryOutcome(
                entry_id=entry_id,
                status=KnowledgeDeliveryStatus.QUEUED,
                detail="Delivery result was superseded by a newer durable entry",
            )
        return KnowledgeDeliveryOutcome(
            entry_id=entry_id,
            status=KnowledgeDeliveryStatus.QUEUED,
            detail=failure_detail,
            retry_after=failure_retry_after,
        )
