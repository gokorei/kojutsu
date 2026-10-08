"""A check run, stored as a machine report about a commit.

The properties here are the ones that keep a tool's output from reading as a
person's judgement, which is the specific risk in putting a CI conclusion into a
corpus whose whole provenance apparatus is built around who said what.

- the conclusion is the forge's wording, never kojutsu's restatement of it
- a check is never rendered where a review would be, and never carries an
  independence level, because nobody was positioned to disagree about it
- a run that has not concluded is not stored, so the record cannot be rewritten
  several times per run
"""

from __future__ import annotations

from kojutsu.core.answer_collector import (
    process_check_run_outcome,
)
from kojutsu.core.knowledge_sink import KnowledgeDeliveryOutcome, KnowledgeDeliveryStatus
from kojutsu.core.tanseki_mapping import build_frontmatter
from kojutsu.models import CaptureSource, KnowledgeEntry, QuestionCategory

SHA = "b" * 40


class _Sink:
    def __init__(self) -> None:
        self.entries: list[KnowledgeEntry] = []
        self.claim: str | None = None

    def store(self, entry: object) -> KnowledgeDeliveryOutcome:
        assert isinstance(entry, KnowledgeEntry)
        self.entries.append(entry)
        return KnowledgeDeliveryOutcome(
            entry_id=entry.entry_id, status=KnowledgeDeliveryStatus.DELIVERED
        )


class _Registry:
    def __init__(self) -> None:
        self.claimed: list[str] = []
        self.completed: list[str] = []
        self.released: list[tuple[str, str]] = []

    def claim_review_capture(self, **kwargs: object) -> str:
        self.claimed.append(str(kwargs.get("review_event_id")))
        return "token-1"

    def complete_review_capture(self, event_id: str, claim_token: str) -> bool:
        self.completed.append(event_id)
        return True

    def release_review_capture(self, event_id: str, claim_token: str, reason: str) -> bool:
        self.released.append((event_id, reason))
        return True


def _store(**overrides: object) -> tuple[KnowledgeEntry | None, _Sink, _Registry]:
    kwargs: dict[str, object] = {
        "repo": "org/repo",
        "check_run_id": 4242,
        "check_name": "build",
        "check_status": "completed",
        "check_conclusion": "failure",
        "head_sha": SHA,
        "pr_number": 42,
        "delivery_id": "delivery-1",
    }
    kwargs.update(overrides)
    sink, registry = _Sink(), _Registry()
    process_check_run_outcome(registry=registry, sink=sink, **kwargs)  # type: ignore[arg-type]
    return (sink.entries[0] if sink.entries else None), sink, registry


def test_a_concluded_check_is_stored() -> None:
    entry, _, _ = _store()
    assert entry is not None
    assert entry.category is QuestionCategory.SYSTEM_EVENT
    assert entry.capture_source is CaptureSource.WEBHOOK


def test_the_conclusion_is_stored_verbatim_as_the_forge_reported_it() -> None:
    """Kojutsu does not translate a check's wording into its own.

    A record that said "failed" where the forge said "failure" would be kojutsu
    editorialising, and the difference between the two is exactly the difference
    between a fact about a tool and a judgement about a change.
    """
    entry, _, _ = _store(check_conclusion="action_required")
    assert entry is not None
    frontmatter = build_frontmatter(entry)
    assert frontmatter["check_conclusion"] == "action_required"
    assert "concluded action_required" in frontmatter["title"]
    assert frontmatter["check_name"] == "build"
    assert frontmatter["check_status"] == "completed"


def test_the_record_says_concluded_rather_than_failed() -> None:
    entry, _, _ = _store()
    assert entry is not None
    # The *title* says the check concluded; the body reports what it reported.
    # Restating "failure" as "failed" anywhere is kojutsu editorialising.
    assert "concluded failure" in entry.question_text
    assert "conclusion 'failure'" in entry.answer_text
    assert "not a judgement about the change" in entry.answer_text


def test_a_check_never_carries_an_independence_level() -> None:
    """Nobody was positioned to disagree about whether a check passed.

    A check report is machine output, so it cannot be independent of itself, and
    giving it a level would let an evidence filter return it as a second opinion.
    """
    entry, _, _ = _store()
    assert entry is not None
    frontmatter = build_frontmatter(entry)
    assert "independence" not in frontmatter
    assert "independence_reason" not in frontmatter


def test_a_check_never_lands_in_the_review_namespace() -> None:
    entry, _, _ = _store()
    assert entry is not None
    # The kind is check_run; the tag is the projection of it. Neither is review,
    # and a check must never be findable where a person's judgement would be.
    assert build_frontmatter(entry)["record_kind"] == "check_run"
    assert "check" in entry.tags
    assert "review" not in entry.tags
    assert not any(tag.startswith("review_state_") for tag in entry.tags)


def test_an_unconcluded_check_is_not_stored() -> None:
    """A run in progress says nothing yet, and would be rewritten repeatedly."""
    for conclusion in (None, "", "   "):
        entry, sink, _ = _store(check_conclusion=conclusion)
        assert entry is None
        assert sink.entries == []


def test_a_re_run_is_a_separate_record_from_the_run_it_replaced() -> None:
    """GitHub gives a re-run a new check run id, so they must not collapse."""
    _, _, first = _store(check_run_id=1)
    _, _, second = _store(check_run_id=2)
    assert first.claimed != second.claimed
    assert first.claimed == ["check-run:org/repo:1"]
    assert second.claimed == ["check-run:org/repo:2"]


def test_the_repository_is_case_folded_in_the_identity() -> None:
    _, _, upper = _store(repo="Org/Repo", check_run_id=7)
    _, _, lower = _store(repo="org/repo", check_run_id=7)
    assert upper.claimed == lower.claimed


def test_a_check_records_the_commit_it_ran_against() -> None:
    """What makes the report usable beside a review record on the same change."""
    entry, _, _ = _store()
    assert entry is not None
    assert build_frontmatter(entry)["head_sha"] == SHA


def test_a_check_with_no_pull_request_is_ordinary() -> None:
    """Checks run on branches, not only on pull requests."""
    entry, _, _ = _store(pr_number=None)
    assert entry is not None
    frontmatter = build_frontmatter(entry)
    assert "pr" not in frontmatter
    assert frontmatter["head_sha"] == SHA


def test_a_delivery_failure_releases_the_claim_rather_than_losing_it() -> None:
    """A queued check is re-derivable from the forge, but a stranded claim is not."""

    class _Failing(_Sink):
        def store(self, entry: object) -> KnowledgeDeliveryOutcome:
            return KnowledgeDeliveryOutcome(
                entry_id="x", status=KnowledgeDeliveryStatus.DEAD_LETTERED, detail="nope"
            )

    sink, registry = _Failing(), _Registry()
    process_check_run_outcome(
        repo="org/repo",
        check_run_id=9,
        check_name="build",
        check_status="completed",
        check_conclusion="failure",
        head_sha=SHA,
        pr_number=42,
        registry=registry,  # type: ignore[arg-type]
        sink=sink,  # type: ignore[arg-type]
        delivery_id="d",
    )
    assert registry.released
    assert registry.completed == []
