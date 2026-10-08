"""What a stored record says about the *change* it is about, rather than itself.

Four tickets meet in this file, because they all add keys at the same boundary
and a test per ticket would be four files describing one mechanism:

- ``3QRPK52A`` — the account that opened the pull request
- ``QEVSTMYW`` — whether a closed change was merged or abandoned
- ``9PTQE45Y`` — when the change opened
- ``RJNVJK1P`` — ``review_id`` and an explicit ``record_kind``

Each of these arrived on the webhook payload, was used to reach a decision, and
was then dropped before the record was written. The store could not describe the
change it existed to describe. These tests pin what it says now, and — as
importantly — pin what it says *absent*, because a key that defaults is a key a
reader cannot trust.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kojutsu.core.answer_collector import (
    PR_OUTCOME_CLOSED_UNMERGED,
    PR_OUTCOME_MERGED,
    PR_OUTCOMES,
    RECORD_KINDS,
    RecordKind,
    _record_tags,
    semantic_pr_event_id,
)
from kojutsu.core.tanseki_mapping import (
    CHANGE_AUTHOR_KEY,
    FILES_KEY,
    HEAD_SHA_KEY,
    MAX_FRONTMATTER_FILES,
    PR_MERGED_AT_KEY,
    PR_OPENED_AT_KEY,
    PR_OUTCOME_KEY,
    RECORD_KIND_KEY,
    REVIEW_ID_KEY,
    build_content,
    build_frontmatter,
)
from kojutsu.models import KnowledgeEntry, QuestionCategory

AUTHOR_ASSOCIATION = "OWNER"
REPO = "org/repo"
PR = 42


def _entry(**metadata: object) -> KnowledgeEntry:
    base: dict[str, object] = {"repo": REPO, "pr_number": PR}
    base.update(metadata)
    return KnowledgeEntry(
        entry_id="entry-1",
        question_text="Why this approach?",
        answer_text="Because.",
        category=QuestionCategory.DESIGN_DECISION,
        answered_at=datetime.now(UTC),
        metadata=base,
    )


# -- 3QRPK52A: the change author ---------------------------------------------


def test_the_change_author_is_stored_and_is_not_the_reviewer() -> None:
    """The author of the change is a third party, distinct from both other names."""
    frontmatter = build_frontmatter(
        _entry(**{CHANGE_AUTHOR_KEY: "change-owner", "comment_author": "reviewer"})
    )
    assert frontmatter[CHANGE_AUTHOR_KEY] == "change-owner"
    assert frontmatter["comment_author"] == "reviewer"
    assert frontmatter[CHANGE_AUTHOR_KEY] != frontmatter["comment_author"]


def test_a_missing_change_author_is_absent_rather_than_a_placeholder() -> None:
    """GitHub sometimes reports no user. The key is then simply not written.

    Not written as ``"unknown"`` and not written as an empty string, because a
    reader filtering on the key would otherwise match a record that has no author
    against a record whose author happens to be a string nobody has seen.
    """
    frontmatter = build_frontmatter(_entry())
    assert CHANGE_AUTHOR_KEY not in frontmatter


def test_the_change_author_is_not_recoverable_from_the_independence_label() -> None:
    """A level without its inputs is a claim, so the inputs are stored too.

    ``INDEPENDENT`` collapses "a different account" and other cases a reader
    cannot otherwise separate. The label alone is lossy, and a record that could
    only be reconstructed by re-fetching the pull request is not self-describing.
    """
    assert CHANGE_AUTHOR_KEY in build_frontmatter(_entry(**{CHANGE_AUTHOR_KEY: "someone"}))


# -- QEVSTMYW: how a closed change ended --------------------------------------


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [(PR_OUTCOME_MERGED, "merged"), (PR_OUTCOME_CLOSED_UNMERGED, "closed_unmerged")],
)
def test_a_close_records_which_side_of_the_line_it_fell_on(outcome: str, expected: str) -> None:
    """GitHub says ``state == "closed"`` for a merge and an abandonment alike."""
    frontmatter = build_frontmatter(_entry(**{PR_OUTCOME_KEY: outcome}))
    assert frontmatter[PR_OUTCOME_KEY] == expected
    assert outcome in PR_OUTCOMES


def test_an_outcome_that_was_never_reported_is_absent_not_false() -> None:
    """Three states, not two.

    A boolean would have to collapse "closed without merging" and "we were never
    told" into one value, and those are facts about different records. The
    distinction a consumer most needs is exactly the one a boolean destroys.
    """
    frontmatter = build_frontmatter(_entry())
    assert PR_OUTCOME_KEY not in frontmatter
    assert "false" not in frontmatter.values()


def test_the_merge_time_is_stored_separately_from_the_outcome() -> None:
    """An outcome is a claim; a time is a fact. Neither derives the other."""
    merged_at = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)
    frontmatter = build_frontmatter(
        _entry(**{PR_OUTCOME_KEY: PR_OUTCOME_MERGED, PR_MERGED_AT_KEY: merged_at.isoformat()})
    )
    assert frontmatter[PR_MERGED_AT_KEY] == merged_at.isoformat()
    assert frontmatter[PR_OUTCOME_KEY] == PR_OUTCOME_MERGED


# -- 9PTQE45Y: when the change opened -----------------------------------------


def test_the_open_time_is_stored_under_its_own_name() -> None:
    """Never ``updated_at``, which on these records is a different event.

    On a lifecycle record ``updated_at`` is when the transition happened; on a
    review record it is when the review was submitted. Overloading either would
    file a fact about the write under a name that means the change.
    """
    opened_at = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    frontmatter = build_frontmatter(
        _entry(
            answered_at=datetime(2026, 3, 5, 9, 0, tzinfo=UTC),
            **{PR_OPENED_AT_KEY: opened_at.isoformat()},
        )
    )
    assert frontmatter[PR_OPENED_AT_KEY] == opened_at.isoformat()
    assert frontmatter["updated_at"] != opened_at.isoformat()


def test_the_open_time_is_absent_when_the_payload_carried_none() -> None:
    frontmatter = build_frontmatter(_entry())
    assert PR_OPENED_AT_KEY not in frontmatter


# -- RJNVJK1P: what kind of record this is ------------------------------------


def test_the_record_kind_is_readable_without_inferring_it_from_tags() -> None:
    """An answer written by someone who stated no agent carries *no* tags.

    That is the case the tag inference cannot handle: "no tags" is genuinely
    ambiguous across kinds, and a reader left guessing cannot tell a record with
    no tags from a record of a kind nobody has seen.
    """
    entry = _entry(tags=[], **{RECORD_KIND_KEY: RecordKind.ANSWER.value})
    assert build_frontmatter(entry)[RECORD_KIND_KEY] == "answer"
    assert build_frontmatter(entry)["tags"] == []


def test_a_legacy_record_with_no_kind_reads_as_having_no_kind() -> None:
    """Absence is reported, not inferred.

    A record written before this key existed genuinely has no kind. Deriving one
    from its tags and writing it back would be recording an interpretation of old
    code as though it were data, and it would be indistinguishable from a kind
    somebody stated.
    """
    frontmatter = build_frontmatter(_entry(tags=["review", "inline_comment"]))
    assert RECORD_KIND_KEY not in frontmatter


def test_the_record_kind_vocabulary_is_closed() -> None:
    """A kind is what a consumer filters on, so an open string breaks quietly."""
    assert {kind.value for kind in RecordKind} == RECORD_KINDS
    with pytest.raises(ValueError):
        RecordKind("pr_lifecycle_event")


def test_tags_are_a_projection_of_the_kind_and_cannot_drift() -> None:
    """Tags are derived from the kind, not spelled at each call site.

    Two spellings of one tag set is a second, quieter answer to "what kind of
    record is this", and the refinements are checked against the kind so a review
    state cannot land on an answer.
    """
    assert _record_tags(RecordKind.REVIEW_VERDICT, review_state="approved") == [
        "review",
        "review_state_approved",
    ]
    assert _record_tags(RecordKind.INLINE_REVIEW_COMMENT) == ["review", "inline_comment"]
    assert _record_tags(RecordKind.PR_LIFECYCLE, action="closed") == [
        "pr_state_change",
        "action_closed",
    ]
    assert _record_tags(RecordKind.ANSWER) == []
    with pytest.raises(ValueError):
        _record_tags(RecordKind.ANSWER, review_state="approved")
    with pytest.raises(ValueError):
        _record_tags(RecordKind.ANSWER, action="closed")


def test_agent_authored_is_orthogonal_to_the_kind() -> None:
    """It answers "who wrote this", which no kind does, so it is free-form."""
    assert _record_tags(RecordKind.ANSWER, agent_authored=True) == ["agent_authored"]


def test_the_review_id_is_stored() -> None:
    """The one stable way to pair a record with the review it came from."""
    frontmatter = build_frontmatter(_entry(**{REVIEW_ID_KEY: 987654}))
    assert frontmatter[REVIEW_ID_KEY] == 987654


# -- the identity derivations must not move -----------------------------------


def test_the_pr_event_identity_is_pinned_to_the_domain_separated_derivation() -> None:
    """Golden values for the derivation as ticket ``94A8A96W`` left it.

    The preimage gained its domain label, so these two values are *not* the ones the
    implementation emitted before it -- they are the ones it emits now, pinned so the
    next change to this function is a red test rather than a corpus-wide rewrite.
    Every lifecycle record captured before the label was added carries the old value,
    which is a re-identification of that subset and is why the label is in the
    preimage: an old id and a new id are visibly different namespaces rather than two
    values a reader has no way to tell apart.

    Both are the *no-timestamp* case, which is the shape the identifier was first
    pinned in and the one every stored record exists in, because every other shape
    used to raise.
    """
    assert semantic_pr_event_id("org/repo", 42, "opened", "open", None) == (
        "pr-event:ee2251bcf9629faea07fcfda6361d7d00266b38fef2a25dec4200d35fe17c492"
    )
    assert semantic_pr_event_id("org/repo", 42, "closed", "closed", None) == (
        "pr-event:f9db859d2e0fa8359cfe62113efde2ba62d823ae559055c05c5d7ed0eaf81f02"
    )


def test_a_timestamped_transition_produces_an_id_instead_of_raising() -> None:
    """Regression: this raised ``TypeError`` for every real close.

    The identity preimage is JSON, and the timestamps it folds in arrive from the
    payload as ``datetime`` objects, which ``json.dumps`` cannot encode. Every
    close and reopen whose payload carried a timestamp therefore failed, and only
    the all-``None`` case had ever been exercised — so the function was tested
    only on inputs the forge does not send.

    A close is the event that says a change was adopted or abandoned, so this is
    the path that decides whether the outcome is recorded at all.
    """
    closed_at = datetime(2026, 3, 4, 10, 0, tzinfo=UTC)
    merged_at = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)

    for marker in (
        semantic_pr_event_id(REPO, PR, "closed", "closed", closed_at, pr_merged_at=merged_at),
        semantic_pr_event_id(REPO, PR, "closed", "closed", closed_at),
        semantic_pr_event_id(REPO, PR, "reopened", "open", None, pr_updated_at=closed_at),
    ):
        assert marker.startswith("pr-event:")


def test_a_redelivered_close_deduplicates_rather_than_producing_a_second_record() -> None:
    """The identity keys on when the transition happened, not on whether it merged.

    That is the right contract: the same close event delivered twice must land on
    one record, and GitHub reports ``merged_at`` on the same delivery either way.
    A later, distinct transition carries a distinct close time and therefore a
    distinct id, which is what keeps the two facts apart.
    """
    closed_at = datetime(2026, 3, 4, 10, 0, tzinfo=UTC)
    assert semantic_pr_event_id(REPO, PR, "closed", "closed", closed_at) == semantic_pr_event_id(
        REPO, PR, "closed", "closed", closed_at
    )
    later = datetime(2026, 3, 9, 10, 0, tzinfo=UTC)
    assert semantic_pr_event_id(REPO, PR, "closed", "closed", later) != semantic_pr_event_id(
        REPO, PR, "closed", "closed", closed_at
    )


def test_an_absent_timestamp_still_serialises_as_null() -> None:
    """The no-timestamp case is the one stored records actually exist behind.

    It must stay a JSON ``null`` in the preimage and not an empty string, an omitted
    element or a rendered ``"None"`` -- those are three different preimages, and a
    close without a timestamp would then derive a different id depending on which
    shape of absence the encoder happened to produce. Pinned rather than computed
    because the whole point is that the byte sequence does not move. See
    :func:`_transition_marker` for why an absent timestamp is not an edge case but
    the ordinary one.
    """
    assert semantic_pr_event_id("org/repo", 42, "closed", "closed", None) == (
        "pr-event:f9db859d2e0fa8359cfe62113efde2ba62d823ae559055c05c5d7ed0eaf81f02"
    )


# -- MWF7Z1EN: the code anchor -------------------------------------------------


def test_the_changed_files_are_stored_as_a_list_rather_than_a_string() -> None:
    """A list is the only form a consumer can filter on.

    The ``extra`` dict stringifies everything it holds, and a Python list repr
    written into YAML is a value nothing can query. This is why ``files`` is
    built in the base mapping instead of added to that dict.
    """
    frontmatter = build_frontmatter(_entry(**{FILES_KEY: ["src/auth.py", "src/session.py"]}))
    assert frontmatter[FILES_KEY] == ["src/auth.py", "src/session.py"]

    import yaml

    # The stored document is frontmatter *and* a Markdown body, so the block is
    # split out before parsing — the same way the rationale tests read it.
    content = build_content(_entry(**{FILES_KEY: ["src/auth.py"]}))
    block = content.split("---", 2)[1]
    assert yaml.safe_load(block)["files"] == ["src/auth.py"]


def test_the_file_list_is_bounded_and_says_so_when_it_is_shortened() -> None:
    """A silently shortened list reads as a complete description of a change."""
    many = [f"src/module_{index}.py" for index in range(MAX_FRONTMATTER_FILES + 12)]
    files = build_frontmatter(_entry(**{FILES_KEY: many}))[FILES_KEY]
    assert len(files) == MAX_FRONTMATTER_FILES + 1
    assert files[-1] == "12 more files not listed"


def test_a_file_list_that_could_not_be_read_is_absent_rather_than_empty() -> None:
    """``None`` and ``[]`` are different facts and must stay different.

    ``None`` means the paths were not available — a failed or rate-limited read.
    ``[]`` means the change touched no files, which is not a thing a real pull
    request does. Collapsing them makes a store outage look like a file-free
    change, and the record looks complete either way.
    """
    assert FILES_KEY not in build_frontmatter(_entry())
    assert FILES_KEY not in build_frontmatter(_entry(**{FILES_KEY: []}))
    assert FILES_KEY not in build_frontmatter(_entry(**{FILES_KEY: None}))


def test_the_head_commit_is_recorded_as_an_anchor() -> None:
    """Which commit the capture was taken against, which is all it says."""
    sha = "a" * 40
    assert build_frontmatter(_entry(**{HEAD_SHA_KEY: sha}))[HEAD_SHA_KEY] == sha
    assert HEAD_SHA_KEY not in build_frontmatter(_entry())
