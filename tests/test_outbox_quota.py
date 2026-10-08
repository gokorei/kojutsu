"""Tests for the outbox growth quota.

The quota exists to convert an unbounded spool into a bounded one, and the only
honest way to apply a bound is to refuse. These tests therefore assert two
things at once: that the bound holds, and that it holds by declining rather than
by deleting -- a quota that quietly frees space for a new row has not bounded
anything, it has only decided which knowledge to lose.

Stating a limitation as a test is the only way it survives the next refactor, so
``test_the_bounds_have_no_unlimited_mode`` and the duplicate-preservation tests
are here deliberately: each one is a property a later change could break
without any other test noticing.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from kojutsu.core.knowledge_sink import (
    KnowledgeDeliveryStatus,
    TansekiKnowledgeSink,
)
from kojutsu.core.outbox import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTRIES,
    OutboxQuotaError,
    OutboxQuotaUsage,
    RelayResult,
    TansekiOutbox,
    relay,
)
from kojutsu.integrations.tanseki import TansekiError, TansekiPermanentError
from kojutsu.models import KnowledgeEntry, QuestionCategory


class _OkWriter:
    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert_document(self, payload: dict) -> dict:
        self.upserts.append(payload)
        return {"revision": "r1"}


class _DownWriter:
    def upsert_document(self, payload: dict) -> dict:
        raise TansekiError("store unavailable")


def _row_bytes(entry_id: str, payload: dict) -> int:
    """Bytes one queued row costs, spelled out rather than imported.

    Deriving it here instead of calling the outbox's own helper is the point: a
    boundary asserted through the same expression that enforces it proves only
    that the code agrees with itself. This states what a row costs -- the UTF-8
    length of the entry id plus the stored JSON -- so the boundary the quota
    refuses at is visible in the test.
    """
    return len(entry_id.encode()) + len(json.dumps(payload).encode())


def _entry(entry_id: str) -> KnowledgeEntry:
    return KnowledgeEntry(
        entry_id=entry_id,
        question_text="Why keep the spool bounded?",
        answer_text="Because the alternative is the disk, or something else breaking.",
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        metadata={"repo": "org/repo", "pr_number": 1},
    )


def test_the_default_limits_are_the_documented_numbers() -> None:
    """Pin the defaults.

    Asserting "some limit was hit" is not a test of a limit. If either default
    moves, an operator's sizing decision moves with it and nothing fails, so the
    numbers are pinned here rather than derived from the constants under test.
    """
    assert DEFAULT_MAX_ENTRIES == 10_000
    assert DEFAULT_MAX_BYTES == 256 * 1024 * 1024


def test_an_outbox_constructed_without_limits_is_bounded(tmp_path: Path) -> None:
    """Omission is not unlimited.

    Every call site in the product constructs the outbox without saying anything
    about capacity, so if the default were "no bound" the premise of this ticket
    would still be true in production after all this code.
    """
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        usage = outbox.quota_usage()
    assert usage == OutboxQuotaUsage(
        entries=0,
        bytes=0,
        max_entries=DEFAULT_MAX_ENTRIES,
        max_bytes=DEFAULT_MAX_BYTES,
    )
    assert not usage.full


def test_the_bounds_have_no_unlimited_mode(tmp_path: Path) -> None:
    """A larger bound is expressible; "no bound" deliberately is not.

    An operator with a legitimately deep queue raises the number, which stays
    visible in configuration. Accepting ``None`` would accept the absence of
    that decision, which is the condition this quota exists to close.
    """
    for unlimited in (None, 0, -1):
        with pytest.raises(ValueError, match="max_entries must be a positive integer"):
            TansekiOutbox(tmp_path / "outbox.db", max_entries=unlimited)
        with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
            TansekiOutbox(tmp_path / "outbox.db", max_bytes=unlimited)


def test_limits_reject_a_bool_that_would_mean_one_row(tmp_path: Path) -> None:
    """``True`` is an ``int``, and a one-row quota must never be an accident."""
    with pytest.raises(ValueError, match="max_entries must be a positive integer"):
        TansekiOutbox(tmp_path / "outbox.db", max_entries=True)
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        TansekiOutbox(tmp_path / "outbox.db", max_bytes=True)


def test_the_row_quota_refuses_exactly_the_entry_past_the_bound(tmp_path: Path) -> None:
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=3) as outbox:
        for index in range(3):
            assert outbox.enqueue(f"e{index}", {"id": f"d{index}"}) is True

        with pytest.raises(OutboxQuotaError) as refusal:
            outbox.enqueue("overflow", {"id": "overflow"})

        assert refusal.value.entry_id == "overflow"
        assert refusal.value.limit_name == "max_entries"
        assert refusal.value.limit == 3
        assert refusal.value.observed == 3
        assert refusal.value.usage.entries == 3
        assert outbox.quota_usage().entries == 3
        assert outbox.quota_usage().full


def test_the_byte_quota_refuses_exactly_the_payload_past_the_bound(tmp_path: Path) -> None:
    stored = {"id": "d0", "content": "a" * 200}
    with TansekiOutbox(tmp_path / "outbox.db", max_bytes=_row_bytes("e0", stored)) as outbox:
        # Exactly at the bound is allowed; the bound is a ceiling, not a target.
        assert outbox.enqueue("e0", stored) is True
        assert outbox.quota_usage().bytes == _row_bytes("e0", stored)

        with pytest.raises(OutboxQuotaError) as refusal:
            outbox.enqueue("e1", {"id": "d1"})

        assert refusal.value.limit_name == "max_bytes"
        assert refusal.value.limit == _row_bytes("e0", stored)
        assert refusal.value.observed == _row_bytes("e0", stored) + _row_bytes("e1", {"id": "d1"})


def test_a_payload_too_large_for_the_bound_itself_is_refused(tmp_path: Path) -> None:
    """An empty outbox still refuses a payload that cannot fit.

    The alternative -- accepting it and exceeding the bound, or accepting it and
    truncating it -- is worse than a refusal the operator can answer by raising
    ``max_bytes``. Silently truncating would store a knowledge record that is not
    the one the human wrote.
    """
    oversized = {"id": "d0", "content": "x" * 500}
    with TansekiOutbox(tmp_path / "outbox.db", max_bytes=200) as outbox:
        with pytest.raises(OutboxQuotaError) as refusal:
            outbox.enqueue("e0", oversized)
        assert refusal.value.limit_name == "max_bytes"
        assert outbox.pending() == []

        with TansekiOutbox(tmp_path / "wide.db", max_bytes=_row_bytes("e0", oversized)) as wider:
            assert wider.enqueue("e0", oversized) is True


def test_a_refusal_deletes_nothing_and_evicts_nothing(tmp_path: Path) -> None:
    """The bound is applied by declining.

    Every row that was durable before the refusal is still there afterwards, with
    the same content. If this test needed a carve-out for dead letters it would
    be asserting an eviction policy, which is the thing being ruled out.
    """
    payloads = {"keep-1": {"id": "d1", "content": "first"}, "keep-2": {"id": "d2"}}
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=2) as outbox:
        for entry_id, payload in payloads.items():
            outbox.enqueue(entry_id, payload)

        with pytest.raises(OutboxQuotaError):
            outbox.enqueue("overflow", {"id": "d3"})

        assert {item.entry_id: item.payload for item in outbox.pending()} == payloads
        assert outbox.pending_count() == 2
        assert outbox.quota_usage().entries == 2


def test_a_refusal_names_the_entry_and_what_the_operator_should_do(tmp_path: Path) -> None:
    """ "Quota exceeded" with no next step is half a feature.

    The commands named here are the ones the CLI already exposes, so the detail
    is executable advice rather than an apology.
    """
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=1) as outbox:
        outbox.enqueue("answer-1", {"id": "d1"})
        with pytest.raises(OutboxQuotaError) as refusal:
            outbox.enqueue("answer-2", {"id": "d2"})

    detail = str(refusal.value)
    assert "answer-2" in detail
    assert "Nothing was deleted to make room" in detail
    assert "kojutsu relay" in detail
    assert "kojutsu outbox-dead-letters" in detail
    assert "kojutsu outbox-cleanup" in detail


def test_a_refusal_reaches_the_caller_as_its_own_outcome(tmp_path: Path) -> None:
    """A refusal is distinguishable from "already present".

    ``enqueue`` returns ``False`` for an active delivery lease, and the sink
    reports that as ``QUEUED``. The sink maps a refusal to an exception, because
    the only other answers it has would tell the producer its record is safe.
    """
    writer = _DownWriter()
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=1) as outbox:
        sink = TansekiKnowledgeSink(writer, outbox)

        first = sink.store(_entry("answer-1"))
        assert first.status is KnowledgeDeliveryStatus.QUEUED

        # Same entry again: already stored, so this is not a refusal at all.
        again = sink.store(_entry("answer-1"))
        assert again.status is KnowledgeDeliveryStatus.QUEUED

        with pytest.raises(OutboxQuotaError) as refusal:
            sink.store(_entry("answer-2"))

    assert refusal.value.entry_id == "answer-2"
    assert refusal.value.limit_name == "max_entries"


def test_a_refusal_is_not_swallowed_as_a_store_outage(tmp_path: Path) -> None:
    """A full spool must not be mistaken for a store that is merely down.

    Both are "the capture did not land", and the difference decides whether the
    producer retries: a queue problem is an operator's, and a webhook that treats
    it as a transient store error will retry forever without anything changing.
    """
    writer = _DownWriter()
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=1) as outbox:
        sink = TansekiKnowledgeSink(writer, outbox)
        sink.store(_entry("answer-1"))
        with pytest.raises(OutboxQuotaError):
            sink.store(_entry("answer-2"))
        # The refused record left nothing behind to be confused with it.
        assert [item.entry_id for item in outbox.pending()] == ["answer-1"]
        assert outbox.status_counts() == {"pending": 0, "retrying": 1, "dead_letter": 0}


def test_re_enqueueing_a_stored_entry_never_consumes_a_second_slot(tmp_path: Path) -> None:
    """Duplicate preservation, in rows.

    A retry storm while the store is down repeats one fact many times. Counting
    those repeats as new slots would let a single unanswered capture exhaust the
    quota, which is the opposite of what a quota is for.
    """
    payload = {"id": "d1", "content": "the same fact"}
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=1) as outbox:
        assert outbox.enqueue("answer-1", payload) is True
        for _ in range(5):
            assert outbox.enqueue("answer-1", payload) is True

        assert outbox.quota_usage().entries == 1
        with pytest.raises(OutboxQuotaError):
            outbox.enqueue("answer-2", {"id": "d2"})

        assert [item.entry_id for item in outbox.pending()] == ["answer-1"]


def test_re_enqueueing_identical_content_at_a_full_byte_quota_is_not_a_refusal(
    tmp_path: Path,
) -> None:
    """Duplicate preservation, in bytes.

    Identical content is charged zero, because it changes nothing on disk. If it
    were charged, a byte quota would refuse the re-confirmation of a record it
    already holds, and the producer could not tell that apart from "not stored".
    """
    payload = {"id": "d1", "content": "x" * 120}
    with TansekiOutbox(tmp_path / "outbox.db", max_bytes=_row_bytes("answer-1", payload)) as outbox:
        assert outbox.enqueue("answer-1", payload) is True
        for _ in range(4):
            assert outbox.enqueue("answer-1", payload) is True

        assert outbox.quota_usage().bytes == _row_bytes("answer-1", payload)
        with pytest.raises(OutboxQuotaError):
            outbox.enqueue("answer-2", payload)


def test_a_replacement_is_charged_only_for_what_it_grows(tmp_path: Path) -> None:
    """Subtracting what the row already costs is what makes duplicates free.

    Growing a stored entry is new content and must be paid for; keeping it the
    same size must not be. The refused payload leaves the stored row untouched,
    because the row that was already durable is not the one being sacrificed.
    """
    small = {"id": "d1", "content": "s"}
    large = {"id": "d1", "content": "L" * 400}
    with TansekiOutbox(tmp_path / "outbox.db", max_bytes=_row_bytes("e1", large) - 1) as outbox:
        assert outbox.enqueue("e1", small) is True

        with pytest.raises(OutboxQuotaError) as refusal:
            outbox.enqueue("e1", large)

        assert refusal.value.limit_name == "max_bytes"
        assert outbox.pending()[0].payload == small
        assert outbox.quota_usage().bytes == _row_bytes("e1", small)


def test_staging_behind_an_active_lease_is_charged_and_refused(tmp_path: Path) -> None:
    """The leased path is the one place a duplicate could still cost bytes.

    Staging a replacement while a delivery is in flight parks a second copy of
    the payload in the row, so it is charged. Refusing it leaves the in-flight
    delivery alone: the caller learns its newer payload was not stored and can
    retry, rather than the queue silently keeping a copy it cannot afford.
    """
    payload = {"id": "d1", "content": "x" * 200}
    with TansekiOutbox(tmp_path / "outbox.db", max_bytes=_row_bytes("e1", payload)) as outbox:
        assert outbox.enqueue("e1", payload) is True
        claim = outbox.claim_entry("e1")
        assert claim is not None and claim.lease_token is not None

        with pytest.raises(OutboxQuotaError) as refusal:
            outbox.enqueue("e1", {"id": "d1", "content": "y" * 400})
        assert refusal.value.limit_name == "max_bytes"

        still_leased = outbox.pending()[0]
        assert still_leased.payload == payload
        assert still_leased.lease_token == claim.lease_token


def test_re_confirming_an_in_flight_entry_costs_nothing(tmp_path: Path) -> None:
    """A re-confirmation of what is already being delivered stages nothing.

    The content in flight is the content the caller would stage, so there is no
    new knowledge and no new bytes. Charging for it would refuse a producer for
    repeating itself, which is precisely the retry that then posts a duplicate
    GitHub comment.
    """
    payload = {"id": "d1", "content": "x" * 200}
    with TansekiOutbox(tmp_path / "outbox.db", max_bytes=_row_bytes("e1", payload)) as outbox:
        assert outbox.enqueue("e1", payload) is True
        claim = outbox.claim_entry("e1")
        assert claim is not None

        assert outbox.enqueue("e1", payload) is False
        assert outbox.quota_usage().bytes == _row_bytes("e1", payload)


def test_a_successful_drain_frees_a_slot_for_the_next_capture(tmp_path: Path) -> None:
    """The quota is a live bound, not a one-way door.

    If a refusal were permanent the outbox would simply be full, which is a
    different and much blunter failure than a bound. Delivery removing the row is
    what makes the refusal temporary.
    """
    writer = _OkWriter()
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=1) as outbox:
        outbox.enqueue("first", {"id": "d1"})
        with pytest.raises(OutboxQuotaError):
            outbox.enqueue("second", {"id": "d2"})

        assert relay(outbox, writer) == RelayResult(sent=1, failed=0)
        assert outbox.quota_usage().entries == 0
        assert not outbox.quota_usage().full

        assert outbox.enqueue("second", {"id": "d2"}) is True

    assert [payload["id"] for payload in writer.upserts] == ["d1"]


def test_a_dead_letter_holds_its_slot_until_an_operator_removes_it(tmp_path: Path) -> None:
    """An undelivered record occupies the outbox whether or not it retries.

    A dead letter is knowledge the store still does not have, so counting it is
    what keeps a store answering permanent errors from being the way around the
    quota. It is also why the refusal tells the operator about
    ``outbox-cleanup``: that removal is an operator's decision, not something the
    quota does on its own.
    """
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=1, retention_days=0) as outbox:
        outbox.enqueue("poison", {"id": "d1"})
        assert (
            outbox.mark_failed("poison", TansekiPermanentError("invalid payload")) == "dead_letter"
        )

        with pytest.raises(OutboxQuotaError):
            outbox.enqueue("next", {"id": "d2"})
        assert outbox.quota_usage().entries == 1

        assert outbox.cleanup_dead_letters() == 1
        assert outbox.quota_usage().entries == 0
        assert outbox.enqueue("next", {"id": "d2"}) is True


def test_an_unreachable_store_does_not_grow_the_spool_past_the_bound(tmp_path: Path) -> None:
    """The outage this ticket is about, end to end.

    A store that never comes back, more distinct records than the bound allows,
    and the same records retried through both entry points. The bound holds, the
    bytes charged are exactly the bytes stored, and nothing was evicted to keep
    the count down.
    """
    writer = _DownWriter()
    payloads = {f"answer-{index}": {"id": f"d{index}"} for index in range(40)}
    refused: list[str] = []
    with TansekiOutbox(tmp_path / "outbox.db", max_entries=25) as outbox:
        sink = TansekiKnowledgeSink(writer, outbox)
        for _ in range(10):
            for entry_id, payload in payloads.items():
                try:
                    outbox.enqueue(entry_id, payload)
                except OutboxQuotaError as exc:
                    refused.append(exc.entry_id)
            relay(outbox, writer)
            for entry_id in payloads:
                try:
                    sink.store(_entry(entry_id))
                except OutboxQuotaError as exc:
                    refused.append(exc.entry_id)

        usage = outbox.quota_usage()
        held = outbox.pending()
        counts = outbox.status_counts()

    assert refused
    assert usage.entries == usage.max_entries == 25
    assert usage.bytes <= usage.max_bytes
    assert usage.full
    assert len(held) == 25
    assert {item.entry_id for item in held} <= set(payloads)
    # No lease overlaps an enqueue in this loop, so nothing is staged, and the
    # only stored text is the entry id, the payload and any recorded failure.
    assert usage.bytes == sum(
        _row_bytes(item.entry_id, item.payload) + len((item.last_error or "").encode())
        for item in held
    )
    assert counts["retrying"] + counts["pending"] == 25
    assert counts["dead_letter"] == 0


def test_the_quota_survives_reopening_as_a_bound_on_existing_content(
    tmp_path: Path,
) -> None:
    """A bound is a property of the file, not of one process's memory.

    Reopening with a smaller limit refuses immediately, because the rows already
    on disk are what the quota counts. A limit that only applied to writes made
    after it was set would be a limit on the process, not on the spool.
    """
    path = tmp_path / "outbox.db"
    with TansekiOutbox(path, max_entries=10) as outbox:
        for index in range(4):
            outbox.enqueue(f"e{index}", {"id": f"d{index}"})

    with TansekiOutbox(path, max_entries=4) as reopened:
        assert reopened.quota_usage().entries == 4
        assert reopened.quota_usage().full
        with pytest.raises(OutboxQuotaError) as refusal:
            reopened.enqueue("e4", {"id": "d4"})
        assert refusal.value.limit_name == "max_entries"
        assert reopened.pending_count() == 4


# --- The operator surface -----------------------------------------------------
#
# The quota's job is to refuse, and a refusal nobody can see is a failure the
# operator learns about from a dropped capture instead. `quota_usage()` exists to
# be read by a human-facing command; these tests exist because nothing called it,
# so the bound was enforced perfectly and reported nowhere.


def test_status_reports_the_quota_so_a_breach_needs_no_second_command(
    monkeypatch, tmp_path: Path
) -> None:
    """Acceptance: a breach is visible in `kojutsu status` on its own.

    The outbox is the thing that will refuse, so `status` is where an operator
    looks before a capture is attempted, not only after one fails.
    """
    from typer.testing import CliRunner

    from kojutsu.cli import app

    monkeypatch.setenv("TANSEKI_URL", "")
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "outbox.db"))
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("answer-1", {"id": "d1"})

    result = CliRunner().invoke(app, ["status"])

    assert result.exit_code == 0
    assert f"outbox_quota: 1/{DEFAULT_MAX_ENTRIES} entries" in result.output
    assert f"{DEFAULT_MAX_BYTES} bytes" in result.output


def test_a_full_quota_is_named_rather_than_left_to_two_numbers_too_add_up(capsys) -> None:
    """The breach itself is stated, not merely implied by entries == max_entries.

    Rendered directly rather than through `status`, because the bound comes from
    the constructor and no setting carries it: a CLI-opened outbox always has the
    default limits, so the only way to see a full one through the command is to
    write ten thousand rows. What is under test is the wording, not the wiring --
    the wiring is the two tests above.
    """
    from kojutsu.cli_shared import _echo_outbox_quota

    _echo_outbox_quota(OutboxQuotaUsage(entries=10, bytes=900, max_entries=10, max_bytes=1000))
    full = capsys.readouterr().out

    assert "outbox_quota: 10/10 entries" in full
    assert "FULL: the next entry will be refused" in full

    _echo_outbox_quota(OutboxQuotaUsage(entries=9, bytes=900, max_entries=10, max_bytes=1000))
    near = capsys.readouterr().out

    assert "outbox_quota: 9/10 entries" in near
    assert "FULL" not in near


def test_status_reports_the_quota_before_it_is_full_so_the_approach_is_visible(
    monkeypatch, tmp_path: Path
) -> None:
    """Not breached, still reported.

    A quota that only speaks once it is full gives no warning at all, which is
    the failure it exists to prevent one step earlier. The usage line prints
    unconditionally; only the FULL marker is conditional.
    """
    from typer.testing import CliRunner

    from kojutsu.cli import app

    monkeypatch.setenv("TANSEKI_URL", "")
    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "outbox.db"))

    result = CliRunner().invoke(app, ["status"])

    assert result.exit_code == 0
    assert "outbox_quota: 0/" in result.output
    assert "FULL" not in result.output


def test_outbox_shows_a_retrying_deliverys_age_and_attempts_without_dead_lettering_it(
    monkeypatch, tmp_path: Path
) -> None:
    """Acceptance: age and attempt count visible for a long-retrying delivery.

    Delivery retries are unbounded on purpose -- a captured answer is the only
    copy of something a human typed -- so visibility is the only lever an
    operator has. A spool entry retrying for days must be obvious in the ordinary
    `outbox` listing, not only in the dead-letter command it has not reached.
    """
    from typer.testing import CliRunner

    from kojutsu.cli import app

    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "outbox.db"))
    with TansekiOutbox(tmp_path / "outbox.db") as outbox:
        outbox.enqueue("answer-1", {"id": "d1"})
        assert relay(outbox, _DownWriter()) == RelayResult(sent=0, failed=1)
        assert outbox.status_counts() == {"pending": 0, "retrying": 1, "dead_letter": 0}

    result = CliRunner().invoke(app, ["outbox"])

    assert result.exit_code == 0
    line = next(row for row in result.output.splitlines() if "answer-1" in row)
    assert "state=delivery-retrying" in line
    assert "attempts=1" in line
    assert "age=" in line
    assert "FULL" not in result.output


def test_outbox_reports_the_quota_alongside_the_entries_it_bounds(
    monkeypatch, tmp_path: Path
) -> None:
    """`outbox` reports the bound too, since it is the command about the spool."""
    from typer.testing import CliRunner

    from kojutsu.cli import app

    monkeypatch.setenv("TANSEKI_OUTBOX_PATH", str(tmp_path / "outbox.db"))

    result = CliRunner().invoke(app, ["outbox"])

    assert result.exit_code == 0
    assert f"outbox_quota: 0/{DEFAULT_MAX_ENTRIES} entries" in result.output


# --- Age formatting -----------------------------------------------------------


def test_the_age_column_reads_as_unknown_rather_than_zero_for_an_unstamped_row() -> None:
    """`-` and not `0s`: a row with no timestamp is not a row that just arrived.

    Blanking the column would hide the difference between a legacy row and a
    fresh one, and `0s` would assert the second when the truth is the first.
    """
    from kojutsu.cli_shared import _age

    assert _age("") == "-"
    assert _age("not-a-timestamp") == "-"


def test_a_future_timestamp_reads_as_now_rather_than_a_negative_age() -> None:
    """Clock skew is not an age.

    Printing `-3h` invites a reader to treat a broken clock as a real duration,
    and to reason about a retry that has not happened yet.
    """
    from datetime import UTC, datetime, timedelta

    from kojutsu.cli_shared import _age

    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    assert _age((now + timedelta(hours=3)).isoformat(), now=now) == "0s"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(seconds=40), "40s"),
        (timedelta(minutes=12), "12m"),
        (timedelta(hours=5), "5h"),
        (timedelta(days=3), "3d"),
    ],
)
def test_the_age_column_picks_a_coarse_enough_unit_to_be_scannable(
    delta: timedelta, expected: str
) -> None:
    """Units coarse enough to compare down a column at a glance."""
    from datetime import UTC, datetime

    from kojutsu.cli_shared import _age

    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    assert _age((now - delta).isoformat(), now=now) == expected


def test_the_age_column_treats_a_naive_stamp_as_utc_rather_than_guessing() -> None:
    """A stamp without an offset is read as UTC, matching how the outbox writes them.

    `_now()` in the outbox always emits an aware timestamp, so this only covers a
    hand-edited or imported row. Treating it as local time would make its age
    wrong by the operator's offset, which is worse than a stated convention.
    """
    from datetime import UTC, datetime

    from kojutsu.cli_shared import _age

    now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    assert _age("2026-10-02T09:00:00", now=now) == "3h"
