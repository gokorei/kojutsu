"""Tests for the stated-reason record: its identity, its revisions, and its storage.

The last two sections are the ones that matter most, and they are the ones a later
change is most likely to break quietly.

The identity test is a **golden** test. It pins an exact digest so that a refactor
of the derivation fails rather than silently re-identifying every stored record.
That is not hypothetical: the entry id is a durable ``UNIQUE`` key in the registry
and the Tanseki document id is path-derived from it, so a moved derivation orphans
documents already in the store. See ``docs/design-review/identity-and-limits.md``.

The revision tests are about the same failure the answerer guards against in
``select_questions``: silently attaching a statement to the nearest available
record instead of the one that was named.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from kojutsu.core.question_registry import (
    RATIONALE_IDENTITY_VERSION,
    stable_rationale_entry_id,
)
from kojutsu.core.tanseki_mapping import (
    build_rationale_content,
    build_rationale_frontmatter,
    rationale_document_id,
    to_rationale_upsert_payload,
)
from kojutsu.core.text_hygiene import SANITISATION_KEY
from kojutsu.models import UNKNOWN_MODEL, CaptureSource, RationaleEntry, RationaleSource

MODEL = "opencode/model"
DECLARED_AT = datetime(2026, 1, 1, tzinfo=UTC)

#: The golden digest. Changing ANY input to the derivation, or the derivation
#: itself, must make this test fail rather than quietly re-identify stored records.
#:
#: Provenance: SHA-256 over the length-prefixed, domain-labelled preimage
#: ``("kojutsu.rationale.v1", "org/repo", "42", "feat/x", "opencode", "1")``.
#: Recompute deliberately, and bump ``RATIONALE_IDENTITY_VERSION`` with it.
GOLDEN_ENTRY_ID = "rationale-v1-74acd61bf986f349b2571ed3a5d1c246600ad7ac43a76b42e414cc9c49d45da8"


def make_rationale(**overrides: object) -> RationaleEntry:
    fields: dict[str, object] = {
        "entry_id": stable_rationale_entry_id(
            repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
        ),
        "repo": "org/repo",
        "pr_number": 42,
        "branch": "feat/x",
        "declared_by": "opencode",
        "declared_model": MODEL,
        "rationale_text": "Chose exponential backoff because the API rate-limits on 429.",
        "source": RationaleSource.DECLARED,
        "revision": 1,
        "declared_at": DECLARED_AT,
    }
    fields.update(overrides)
    return RationaleEntry(**fields)  # type: ignore[arg-type]


# --- identity ----------------------------------------------------------------


def test_the_identity_derivation_is_pinned_to_a_golden_value() -> None:
    """A refactor that moves the digest must fail here, not in the store.

    Every other property of this test is a convention. This one is the only thing
    standing between an innocuous-looking edit to the derivation and re-identifying
    every stored rationale, which orphans the Tanseki documents whose ids are derived
    from it.
    """
    assert (
        stable_rationale_entry_id(
            repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
        )
        == GOLDEN_ENTRY_ID
    )
    assert RATIONALE_IDENTITY_VERSION == 1


def test_the_same_anchor_derives_the_same_id() -> None:
    """Determinism is the whole point: a re-delivery must collapse to one record."""
    first = stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
    )
    second = stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
    )
    assert first == second


def test_a_delimiter_in_a_component_cannot_forge_another_record() -> None:
    """Length-prefixing, not delimiter-joining.

    ``("a|b", "c")`` and ``("a", "b|c")`` both serialise to ``a|b|c`` under a
    delimiter join, so a branch name containing the delimiter could collide with a
    different branch name and the dedupe check would pass by accident. Repo and
    branch names are attacker-influenced, so the components may contain anything.
    """
    ambiguous_left = stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="a|b", declared_by="c", revision=1
    )
    ambiguous_right = stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="a", declared_by="b|c", revision=1
    )
    assert ambiguous_left != ambiguous_right, (
        "two different preimages produced one id, so a dedupe check would pass by "
        "accident and a real record could be dropped as a duplicate"
    )


def test_the_identity_is_separated_from_every_other_namespace() -> None:
    """The domain label keeps rationale identity from colliding with anything else."""
    from kojutsu.core.question_registry import stable_answer_entry_id

    rationale_id = stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
    )
    answer_id = stable_answer_entry_id("org/repo", 42, 201)

    assert rationale_id != answer_id
    assert rationale_id.startswith("rationale-v1-")
    assert answer_id.startswith("answer-")


def test_the_identity_is_not_derived_from_the_rationale_text() -> None:
    """Reworded text at the same position is the same record, not a new one.

    A digest over the text would make every rephrasing an unrelated record and
    silently orphan the earlier one, which is exactly the loss the revision model
    exists to prevent.
    """
    reworded = make_rationale(
        rationale_text="Exponential backoff, because the API returns 429 under load."
    )
    assert reworded.entry_id == make_rationale().entry_id


@pytest.mark.parametrize("changed", ["repo", "pr_number", "branch", "declared_by", "revision"])
def test_every_part_of_the_anchor_changes_the_id(changed: str) -> None:
    """The anchor is a tuple, so every component has to participate."""
    base: dict[str, str | int | None] = {
        "repo": "org/repo",
        "pr_number": 42,
        "branch": "feat/x",
        "declared_by": "opencode",
        "revision": 1,
    }
    altered: dict[str, str | int | None] = {
        "repo": "org/other",
        "pr_number": 43,
        "branch": "feat/y",
        "declared_by": "other",
        "revision": 2,
    }
    before = stable_rationale_entry_id(**base)  # type: ignore[arg-type]
    after = stable_rationale_entry_id(**{**base, changed: altered[changed]})  # type: ignore[arg-type]

    assert before != after, (
        f"changing {changed} did not change the id, so that component is not part of "
        "the anchor and two different declarations would share one record"
    )


def test_the_repository_is_matched_case_insensitively() -> None:
    """GitHub owner/name casing is not stable, so it must not fork identity."""
    lower = stable_rationale_entry_id(
        repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
    )
    mixed = stable_rationale_entry_id(
        repo="Org/Repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=1
    )
    assert lower == mixed


def test_a_declaration_before_the_pull_request_exists_still_has_an_identity() -> None:
    """A declaration is made against a change before the PR it produces exists."""
    without_pr = stable_rationale_entry_id(
        repo="org/repo", pr_number=None, branch="feat/x", declared_by="opencode", revision=1
    )
    assert without_pr.startswith("rationale-v1-")
    assert without_pr != stable_rationale_entry_id(
        repo="org/repo", pr_number=0, branch="feat/x", declared_by="opencode", revision=1
    ), "an absent PR number and a real one must not derive the same id"


def test_a_revision_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        stable_rationale_entry_id(
            repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=0
        )


# --- revisions, not overwrites -----------------------------------------------


def test_a_second_declaration_is_a_revision_not_a_replacement() -> None:
    """The earlier record survives intact and the new one points back at it.

    Silent replacement would let a later, worse rationale destroy an earlier,
    better one with nothing left to show that anything had been lost.
    """
    first = make_rationale()
    second = make_rationale(
        entry_id=stable_rationale_entry_id(
            repo="org/repo", pr_number=42, branch="feat/x", declared_by="opencode", revision=2
        ),
        revision=2,
        revises=first.entry_id,
        rationale_text="Kept the backoff; dropped the jitter because it hid the latency.",
    )

    assert first.revision == 1
    assert first.revises is None
    assert first.rationale_text.startswith("Chose exponential backoff"), (
        "the first rationale must be readable after a revision is appended; a "
        "revision that rewrites its predecessor is an overwrite"
    )
    assert second.revision == 2
    assert second.revises == first.entry_id
    assert second.entry_id != first.entry_id


def test_a_revision_that_names_nothing_is_refused() -> None:
    """A dangling revision would read as an update to something unnamed."""
    with pytest.raises(ValidationError, match="supersedes"):
        make_rationale(revision=2, revises=None)


def test_a_first_revision_may_not_supersede_anything() -> None:
    """Otherwise revision 1 could claim to be an update, which it cannot be."""
    with pytest.raises(ValidationError, match="cannot supersede"):
        make_rationale(revision=1, revises="rationale-v1-something")


# --- a rationale is never evidence -------------------------------------------


@pytest.mark.parametrize("source", [CaptureSource.WEBHOOK, CaptureSource.COLLECT])
def test_a_rationale_cannot_claim_to_have_been_captured(source: CaptureSource) -> None:
    """Construction is refused rather than the field corrected.

    The platform verifies who posted a comment, never which model drafted it or
    what it decided. A rationale that could claim a signed delivery would let an
    agent's own account of its work be served to a later reader as verified
    review evidence -- an agent asserting its own reliability.
    """
    with pytest.raises(ValidationError, match="always asserted"):
        make_rationale(capture_source=source)


def test_a_rationale_is_asserted_by_default_and_says_so() -> None:
    rationale = make_rationale()
    assert rationale.capture_source is CaptureSource.ASSERTED
    assert build_rationale_frontmatter(rationale)["capture_source"] == "asserted"


def test_a_declaration_is_first_hand_and_a_reconstruction_is_not() -> None:
    """The source label is what tells a reader which they are holding."""
    assert make_rationale(source=RationaleSource.DECLARED).is_first_hand is True
    assert make_rationale(source=RationaleSource.RECONSTRUCTED).is_first_hand is False


# --- storage boundary: the four places a field silently disappears ------------


def test_the_rationale_survives_the_frontmatter_whitelist() -> None:
    """``build_frontmatter``'s ``extra`` dict is a whitelist, and a key absent
    from it is dropped at the storage boundary.

    Provenance was already lost here once, which is why the entry's own
    docstring says so. A rationale that never reaches the store is worse than one
    that is stored wrongly: the declaration was made and nothing recorded it.
    """
    frontmatter = build_rationale_frontmatter(make_rationale())

    assert frontmatter["rationale_source"] == "declared"
    assert frontmatter["rationale_revision"] == 1
    assert frontmatter["repo"] == "org/repo"
    assert frontmatter["declared_by"] == "opencode"
    assert frontmatter["declared_by_model"] == MODEL
    assert "rationale" in frontmatter["tags"]


def test_an_unstated_model_leaves_the_frontmatter_key_absent() -> None:
    """The stated case above is not enough, and leaving it unasserted cost a bug.

    ``declared_by_model`` used to fall back to ``UNKNOWN_MODEL``, which is a
    *queryable* frontmatter key — so a truthy placeholder survived the
    ``None``-skipping filter and a principal who stated no model came to look like
    one who declared a model called ``"unknown"``. Nothing failed, because a
    filter for a real model would not match it and a filter asking "which records
    state no model" had no way to ask.

    The rationale *body* still prints ``UNKNOWN_MODEL``, because that line is
    prose and cannot leave the sentence out. Frontmatter is not prose: it is the
    part a consumer filters on, so absence there has to mean absence.

    This test exists because the gap survived a test that asserted the stated
    case. Asserting the happy path is not the same as asserting the boundary.
    """
    frontmatter = build_rationale_frontmatter(make_rationale(declared_model=None))
    assert "declared_by_model" not in frontmatter

    # And the two remain distinguishable from a model genuinely named that way,
    # which is the distinction the placeholder destroyed.
    named = build_rationale_frontmatter(make_rationale(declared_model=UNKNOWN_MODEL))
    assert named["declared_by_model"] == UNKNOWN_MODEL


def test_the_body_still_names_an_unstated_model_in_prose() -> None:
    """Prose cannot leave a sentence out, so the placeholder survives there.

    The attribution line would read "Declared model:" with nothing after it, which
    is worse than saying the author did not state one. This is the one place the
    constant is right, and it is a different layer from the key above.
    """
    body = build_rationale_content(make_rationale(declared_model=None))
    assert f"Declared model: {UNKNOWN_MODEL}" in body


def test_the_rationale_reaches_the_document_body() -> None:
    """The body is what a reader actually reads; frontmatter alone is metadata."""
    content = build_rationale_content(make_rationale())
    body = content.split("---", 2)[2]

    assert "Chose exponential backoff" in body
    assert "## Reason" in body
    assert "## Attribution" in body
    assert "asserted by the author" in body, (
        "the model is a self-assertion, so the document must say so where a reader "
        "will meet it rather than leaving it to frontmatter"
    )


def test_the_document_id_keeps_a_rationale_out_of_the_answer_namespace() -> None:
    """A rationale must not land where a reader would take it for a conclusion."""
    assert rationale_document_id(make_rationale()) == (
        "org/repo/pr-42/rationale/" + make_rationale().entry_id
    )
    assert "/rationale/" in rationale_document_id(make_rationale())


def test_the_upsert_payload_carries_the_frontmatter_and_the_body() -> None:
    payload = to_rationale_upsert_payload(make_rationale())

    assert payload["path"].endswith(".md")
    assert payload["frontmatter"]["rationale_source"] == "declared"
    assert "Chose exponential backoff" in payload["content"]
    # The store serialises what it is given, so the block must be valid YAML.
    assert yaml.safe_load(payload["content"].split("---")[1])["repo"] == "org/repo"


def test_a_revision_records_what_it_supersedes_in_both_places() -> None:
    """Both the frontmatter and the body, because a reader may consult either."""
    first = make_rationale()
    second = make_rationale(revision=2, revises=first.entry_id)

    frontmatter = build_rationale_frontmatter(second)
    body = build_rationale_content(second)

    assert frontmatter["rationale_revises"] == first.entry_id
    assert first.entry_id in body


def test_an_unknown_source_survives_storage_without_being_guessed() -> None:
    """A record predating the axis must not be read back as one of the two."""
    frontmatter = build_rationale_frontmatter(make_rationale(source=RationaleSource.UNKNOWN))
    assert frontmatter["rationale_source"] == "unknown"
    assert "rationale_unknown" in frontmatter["tags"]


# --- the two keys `metadata` was never read for ------------------------------
#
# ``build_rationale_frontmatter`` copies named fields out of the model and ignores
# ``metadata`` entirely. For most keys that is harmless, because the named fields
# already say everything. For these two it was not: a stored rationale was the only
# sanitised record in the store that could not say what had been taken out of it, and
# the only one whose re-fetchable anchor stopped at the model. Both were limits the
# seam document had to state rather than properties, which is the thing that made them
# gaps -- a document describing a fidelity claim the code did not make.

SANITISATION_NOTE = (
    "removed 2 characters before storing: "
    "U+0007 unnamed Cc (x1), U+202E RIGHT-TO-LEFT OVERRIDE (x1)"
)


def test_the_sanitisation_note_reaches_the_rationale_document() -> None:
    """The note is the reader's only way to tell sanitised evidence from raw.

    ``core/rationale_collector`` measures and records it, and before this it went
    nowhere -- so a document with no ``text_sanitisation`` could mean "this prose was
    never touched" or "this mapper never looked", and the first is the only one a
    reader is entitled to assume. Both routes write the key, so a document reads the
    same whichever wrote it.
    """
    rationale = make_rationale(metadata={SANITISATION_KEY: SANITISATION_NOTE})

    frontmatter = build_rationale_frontmatter(rationale)
    assert frontmatter[SANITISATION_KEY] == SANITISATION_NOTE
    # Into the document block too, not only the typed frontmatter: the two are built
    # from one dict, and a key that reached only the payload would not be readable by
    # anything that fetches the Markdown.
    block = build_rationale_content(rationale).split("---")[1]
    assert yaml.safe_load(block)[SANITISATION_KEY] == SANITISATION_NOTE


def test_a_declaration_nothing_was_taken_from_says_nothing_rather_than_saying_empty() -> None:
    """Absence, not ``""``. The distinction is the same one every other kind makes.

    An empty string would claim sanitisation ran and found nothing, which is a
    different statement from a declaration that never needed it -- and this writer
    skips empty strings, so an empty value would vanish and take the difference with
    it. This is the case that actually occurs: most declarations are clean.
    """
    frontmatter = build_rationale_frontmatter(make_rationale(metadata={}))
    assert SANITISATION_KEY not in frontmatter


def test_the_anchor_comment_id_reaches_the_rationale_document() -> None:
    """A self-asserted record with nothing to check it against is the worst kind.

    Every other record kind carries either a delivery id or a re-fetchable read id, so
    a reader can go and see whether it is still there. A rationale had neither in its
    document while the comment id sat in the model -- so nothing in the stored record
    invited the comparison that is the entire reason to keep a self-asserted record at
    all. An int from the forge, so there is no third-party text in it to sanitise.
    """
    rationale = make_rationale(metadata={"github_comment_id": 55501})

    assert build_rationale_frontmatter(rationale)["github_comment_id"] == 55501


def test_a_declaration_that_was_never_posted_records_no_comment_id() -> None:
    """Absent, because there is no comment to name -- and no channel test is needed.

    A branch-only declaration was never published anywhere, so there is nothing to
    re-fetch. The two cases are kept apart by the data rather than by a condition on
    ``channel``: the id is in ``metadata`` only when a post actually happened, and this
    writer skips ``None``. An empty string would assert a comment numbered "" exists.
    """
    from kojutsu.models import RationaleChannel

    published = make_rationale(metadata={"github_comment_id": 55501})
    unpublished = make_rationale(
        pr_number=None,
        branch="feat/local-only",
        channel=RationaleChannel.CAPTURE_SERVER,
        metadata={},
    )

    assert build_rationale_frontmatter(unpublished).get("github_comment_id") is None
    assert "github_comment_id" not in build_rationale_frontmatter(unpublished)
    assert build_rationale_frontmatter(published)["github_comment_id"] == 55501


def test_the_document_no_longer_contradicts_what_the_collector_recorded() -> None:
    """End to end through the real writer, so the two cannot drift apart again.

    The collector is the only thing that decides what belongs in ``metadata`` for a
    rationale, and this is the one test that runs its actual output through the actual
    mapper. Asserting the mapper against a hand-built dict would pass even if the key
    name the collector writes and the key name the mapper reads had drifted apart --
    which is the failure mode a two-module change actually has, and the one this gap
    was.
    """
    from kojutsu.core.rationale_collector import process_rationale_comment_outcome
    from kojutsu.integrations.github import rationale_comment_body_as_agent

    class _Sink:
        def __init__(self) -> None:
            self.entries: list[object] = []

        def store(self, entry: object) -> None:  # test double
            self.entries.append(entry)

    class _Registry:
        def claim_rationale(self, **kwargs: object) -> str:
            return "token"

        def complete_rationale(self, entry_id: str, token: str) -> bool:
            return True

        def release_rationale(self, entry_id: str, token: str, error: str) -> bool:
            return True

    sink = _Sink()
    hostile = "Kept the retry loop.\u202e\u0007 No new knob, deliberately."
    outcome = process_rationale_comment_outcome(
        comment_body=rationale_comment_body_as_agent(hostile, "opencode", MODEL, 1, "feat/x"),
        comment_id=55501,
        comment_author="kojutsu-bot",
        comment_created_at=DECLARED_AT,
        author_association="MEMBER",
        repo="org/repo",
        pr_number=42,
        branch="repo-default",
        registry=_Registry(),  # type: ignore[arg-type]
        sink=sink,  # type: ignore[arg-type]
    )

    assert outcome is not None
    entry = sink.entries[0]
    frontmatter = build_rationale_frontmatter(entry)  # type: ignore[arg-type]

    # Both keys arrived, spelled as the collector spells them.
    assert frontmatter[SANITISATION_KEY] == entry.metadata[SANITISATION_KEY]  # type: ignore[attr-defined]
    assert frontmatter["github_comment_id"] == 55501
    assert frontmatter[SANITISATION_KEY].startswith("removed 2 characters before storing: ")
    # And the prose is clean, so the note is describing something real.
    assert "\u202e" not in entry.rationale_text  # type: ignore[attr-defined]
