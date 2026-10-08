"""Tests for mapping KnowledgeEntry to Tanseki documents (Tanseki contract)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import yaml

from kojutsu.core.tanseki_mapping import (
    _unrecorded,
    _write_extras,
    build_census_frontmatter,
    build_clarification_frontmatter,
    build_content,
    build_evaluation_frontmatter,
    build_frontmatter,
    build_question_frontmatter,
    build_rationale_frontmatter,
    document_id,
    document_path,
    serialize_frontmatter,
    to_upsert_payload,
)
from kojutsu.models import (
    STRUCTURE_INFERRED_BY,
    UNKNOWN_MODEL,
    CaptureSource,
    CensusRecord,
    ClarificationEntry,
    EvaluationEntry,
    KnowledgeEntry,
    QuestionCategory,
    QuestionRecord,
    RationaleEntry,
    RationaleSource,
    RecordStructure,
)


def make_entry(
    answer: str = "Because we need X.",
    **overrides: object,
) -> KnowledgeEntry:
    metadata: dict[str, object] = {
        "repo": "org/repo",
        "pr_number": 42,
        "pr_url": "https://github.com/org/repo/pull/42",
        "jira_ticket_key": "PROJ-7",
    }
    metadata.update(overrides.pop("metadata", None) or {})  # type: ignore[arg-type]
    return KnowledgeEntry(
        entry_id="entry-1",
        session_id="session-1",
        question_text="Why this approach?",
        answer_text=answer,
        category=QuestionCategory.DESIGN_DECISION,
        author="dev",
        tags=["auth", "design"],
        structure=overrides.pop("structure", RecordStructure.ANCHORED),  # type: ignore[arg-type]
        metadata=metadata,
    )


def test_document_id_is_path_derived_without_collection() -> None:
    entry = make_entry()
    assert document_id(entry) == "org/repo/pr-42/entry-1"
    assert document_path(entry) == "org/repo/pr-42/entry-1.md"
    assert document_id(entry) == document_id(make_entry())


def test_frontmatter_uses_tanseki_keys_and_tag_list() -> None:
    entry = make_entry()
    frontmatter = build_frontmatter(entry)
    assert frontmatter["repo"] == "org/repo"
    # The PR number is the key Tanseki's edge deriver reads, and it reaches the store
    # as the int the model held. It used to arrive as `"42"`, which is a number's
    # spelling: a document a person opens quotes it, and nothing numeric can be
    # compared against it without parsing it back first.
    assert frontmatter["pr"] == 42
    assert isinstance(frontmatter["pr"], int)
    assert frontmatter["jira"] == "PROJ-7"
    assert frontmatter["author"] == "dev"
    assert frontmatter["tags"] == ["auth", "design"]
    assert frontmatter["updated_at"] == entry.answered_at.isoformat()
    assert frontmatter["category"] == "design_decision"


def test_content_includes_frontmatter_block() -> None:
    content = build_content(make_entry())
    assert content.startswith("---\n")
    assert '"title":' in content
    assert '"tags":\n- "auth"' in content
    assert "## Answer" in content
    assert "Because we need X." in content


def test_serialize_frontmatter_known_keys_first_then_sorted_extra() -> None:
    block = serialize_frontmatter(
        {
            "title": "T",
            "author": "A",
            "tags": ["b", "a"],
            "updated_at": "2026-01-01T00:00:00Z",
            "repo": "r",
            "pr": "1",
        }
    )
    lines = block.strip().split("\n")
    assert lines[0] == '"title": "T"'
    assert lines[1] == '"author": "A"'
    assert lines[2] == '"tags":'
    assert lines[3] == '- "b"'
    assert lines[4] == '- "a"'
    assert lines[5] == '"updated_at": "2026-01-01T00:00:00Z"'
    assert lines[6] == '"pr": "1"'
    assert lines[7] == '"repo": "r"'
    assert yaml.safe_load(block) == {
        "title": "T",
        "author": "A",
        "tags": ["b", "a"],
        "updated_at": "2026-01-01T00:00:00Z",
        "pr": "1",
        "repo": "r",
    }


def test_upsert_payload_has_no_edges_and_collection_is_client_set() -> None:
    payload = to_upsert_payload(make_entry())
    assert "edges" not in payload
    assert "collection" not in payload  # injected by the client
    assert payload["id"] == "org/repo/pr-42/entry-1"
    assert isinstance(payload["frontmatter"], dict)


def test_adversarial_frontmatter_round_trips_without_type_corruption() -> None:
    values = {
        "title": "quotes: \"double\" and 'single'\nnew line\\slash",
        "author": "  spaced: value  ",
        "tags": ["#hash", "true", "null", "a, b", "unicodé 🚀"],
        "boolean_like": "true",
        "null_like": "null",
        "number_like": "0123",
        "colon": "value: with colon",
        "windows_path": "C:\\temp\\file",
    }

    parsed = yaml.safe_load(serialize_frontmatter(values))

    assert parsed == values


def test_question_and_answer_text_survive_content_mapping() -> None:
    entry = make_entry(answer='Line one\nLine "two" with \\ slash and #hash')
    content = build_content(entry)
    frontmatter = yaml.safe_load(content.split("---\n", 2)[1])

    assert frontmatter["title"] == entry.question_text
    assert entry.answer_text in content


def test_oversized_frontmatter_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="10000-character limit"):
        serialize_frontmatter({"title": "x" * 10_001})


def test_excess_tags_are_rejected() -> None:
    with pytest.raises(ValueError, match="100-tag limit"):
        serialize_frontmatter({"tags": [f"tag-{index}" for index in range(101)]})


def test_non_finite_floats_are_rejected_by_name() -> None:
    """JSON has no literal for nan/inf, so they are refused, not serialised."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite float"):
            serialize_frontmatter({"title": "T", "score": bad})


def test_finite_floats_are_accepted_unchanged() -> None:
    parsed = yaml.safe_load(
        serialize_frontmatter({"title": "T", "zero": 0.0, "negative": -3.5, "big": 1e308})
    )
    assert parsed["zero"] == 0.0
    assert parsed["negative"] == -3.5
    assert parsed["big"] == 1e308


# -- the one rule every builder ends with --------------------------------------
#
# ``_unrecorded`` and ``_write_extras`` are tested directly, and that is a change of
# habit rather than a shortcut. These were six copies of a loop ending in
# ``frontmatter[key] = str(value)``, so there was nowhere to state what "absent"
# meant except in six assertions that each restated it. There is now one
# definition, and it is worth pinning on its own.


def test_absent_means_a_gap_and_not_a_result() -> None:
    """The one decision, stated once.

    ``None`` is a collector that never learned the value and ``""`` is a slot left
    blank: both are the absence of a fact, so both are left out. An empty
    collection is neither -- nothing matched, nothing was there -- so it is
    recorded as empty rather than dropped.

    Collapsing the last case into ``None`` is what makes an outage look like an
    answer. ``files`` is the one deliberate exception and says so in the comment at
    its bypass: an empty file list is not a thing a real pull request has, so that
    key treats it as a gap. Everywhere else, an empty collection is recorded.
    """
    # Gaps. A placeholder cannot be written instead: it passes any filter and is
    # then indistinguishable, downstream, from something a person actually stated.
    assert _unrecorded(None) is True
    assert _unrecorded("") is True
    # Results. Recorded, and typed -- an empty list used to be written as the string
    # "[]", which is a claim about a shape the document does not have.
    assert _unrecorded([]) is False
    assert _unrecorded({}) is False
    # And the falsy values that are facts, which a truthiness test would swallow.
    assert _unrecorded(0) is False
    assert _unrecorded(0.0) is False
    assert _unrecorded(False) is False


def test_a_falsy_number_and_a_falsy_flag_are_both_recorded() -> None:
    """``attempts=0`` is the observation, and dropping it is dropping it.

    Every ``if value`` spelling of this guard throws away ``0``, ``0.0`` and
    ``False``, and a projected question nobody retried is exactly ``attempts=0``.
    """
    frontmatter: dict[str, object] = {}
    _write_extras(frontmatter, {"attempts": 0, "reviewer_is_machine": False, "ratio": 0.0})

    assert frontmatter == {"attempts": 0, "reviewer_is_machine": False, "ratio": 0.0}


def test_an_extra_of_each_type_reaches_the_document_as_itself() -> None:
    """The type is the payload: one value per shape, pinned with ``type``.

    ``== "42"`` would have passed under both behaviours, so the assertions are
    written against the type rather than the spelling. What a consumer can now do
    that it could not before is arithmetic and comparison: nothing numeric was
    expressible against ``"42"`` without parsing it back first.
    """
    frontmatter: dict[str, object] = {}
    _write_extras(
        frontmatter,
        {
            "pr": 42,
            "reviewer_is_machine": True,
            "matched_files": ["src/a.py", "src/b.py"],
            "counters": {"reviews": 2, "comments": 1},
            "ratio": 1.5,
            "text": "plain",
        },
    )

    assert type(frontmatter["pr"]) is int
    assert type(frontmatter["reviewer_is_machine"]) is bool
    assert type(frontmatter["matched_files"]) is list
    assert type(frontmatter["counters"]) is dict
    assert type(frontmatter["ratio"]) is float
    assert type(frontmatter["text"]) is str

    # And they survive the document, which is where the type was previously lost.
    parsed = yaml.safe_load(serialize_frontmatter(dict(frontmatter)))
    assert parsed == {
        "pr": 42,
        "reviewer_is_machine": True,
        "matched_files": ["src/a.py", "src/b.py"],
        "counters": {"reviews": 2, "comments": 1},
        "ratio": 1.5,
        "text": "plain",
    }


def test_a_bool_is_not_a_number() -> None:
    """``bool`` subclasses ``int``, so ``isinstance(True, int)`` is true.

    Any test that recognises a number without excluding ``bool`` first reads
    ``True`` as ``1``, and then a reviewer flag and a count are the same value. The
    rule this module applies has no numeric branch at all, so it cannot make that
    mistake; pinned here so that a future numeric bound knows it has to.
    """
    frontmatter: dict[str, object] = {}
    _write_extras(frontmatter, {"reviewer_is_machine": False, "attempts": 0})

    flag, count = frontmatter["reviewer_is_machine"], frontmatter["attempts"]
    assert isinstance(flag, bool)
    assert isinstance(count, int) and not isinstance(count, bool)
    assert flag is not None and count is not None
    # Python considers these equal, which is the whole hazard: ``False == 0`` and
    # ``False == 0.0``. The recorded type is the only thing that keeps a reviewer
    # flag and an attempt count apart, so it is asserted above rather than assumed.
    assert flag == count
    assert yaml.safe_load(serialize_frontmatter(dict(frontmatter))) == {
        "reviewer_is_machine": False,
        "attempts": 0,
    }


def test_an_empty_collection_is_recorded_as_empty_and_not_dropped() -> None:
    """``None`` means nobody looked; ``[]`` means nobody found anything.

    Only the second is a finding, so it is stored, and the two are still told apart
    in the stored document. The known keys are the deliberate exception and have
    their own test below: ``tags: []`` is noise on an untagged document.
    """
    frontmatter: dict[str, object] = {}
    _write_extras(frontmatter, {"matched": [], "gap": None})

    assert "gap" not in frontmatter
    assert frontmatter["matched"] == []
    assert '"matched": []' in serialize_frontmatter(dict(frontmatter))


def test_an_empty_collection_on_a_known_key_is_still_left_out() -> None:
    """``tags: []`` is a line of noise, and dropping it is not a change of presence.

    Unchanged from before: ``tags`` is the only known key that can hold an empty
    collection, and an untagged document should carry no ``tags`` line at all. This
    is pinned so the asymmetry with extras above is a decision on the record rather
    than an accident of the comprehension.
    """
    assert '"tags"' not in serialize_frontmatter({"title": "T", "tags": []})
    assert '"tags":\n- "a"' in serialize_frontmatter({"title": "T", "tags": ["a"]})


def test_every_builder_applies_the_same_rule() -> None:
    """Six copies of one rule is six places to be wrong.

    Each builder carries a number, and the point is that all six now end the same
    way: ``pr`` from a rationale, a clarification, an evaluation, a census record
    and a question, plus ``rationale_revision`` and a question's ``attempts``.
    """
    at = datetime(2026, 3, 4, 10, 0, tzinfo=UTC)

    rationale = RationaleEntry(
        entry_id="rationale-v1-a",
        repo="org/repo",
        pr_number=42,
        branch="feat/x",
        declared_by="opencode",
        rationale_text="Chose a lease token because a timestamp cannot tell held from expired.",
        source=RationaleSource.DECLARED,
        revision=1,
        declared_at=at,
        metadata={"github_comment_id": 55501},
    )
    clarification = ClarificationEntry(
        entry_id="clarification-v1-a",
        repo="org/repo",
        pr_number=42,
        statement="The lease is intentional.",
        author="davy",
        author_association="OWNER",
        github_comment_id=201,
        declared_at=at,
        captured_at=at,
        capture_source=CaptureSource.COLLECT,
    )
    evaluation = EvaluationEntry(
        entry_id="evaluation-v1-a",
        repo="org/repo",
        pr_number=42,
        subject="opencode/model",
        measurement="pairing_precision",
        result_text="0.81",
        scope="One held-out thread.",
        measured_at=at,
    )
    census = CensusRecord(
        entry_id="census-org-repo-pr-42-opened",
        repo="org/repo",
        pr_number=42,
        action="opened",
        observed_at=at,
        delivery_id="delivery-1",
    )
    question = QuestionRecord(
        question_id="q1",
        repo="org/repo",
        pr_number=42,
        question_text="Which lock?",
        status="answered",
        attempts=0,
    )

    built = {
        "rationale": build_rationale_frontmatter(rationale),
        "clarification": build_clarification_frontmatter(clarification),
        "evaluation": build_evaluation_frontmatter(evaluation),
        "census": build_census_frontmatter(census),
        "question": build_question_frontmatter(question),
        "entry": build_frontmatter(make_entry()),
    }
    for kind, frontmatter in built.items():
        assert type(frontmatter["pr"]) is int, kind
    assert type(built["rationale"]["rationale_revision"]) is int
    assert type(built["rationale"]["github_comment_id"]) is int
    # ``attempts=0`` is falsy and is still there: the observation that nobody retried.
    assert built["question"]["attempts"] == 0
    assert type(built["question"]["attempts"]) is int
    # And a flag on an entry is a flag, in every document that carries one.
    flagged = build_frontmatter(make_entry(metadata={"reviewer_is_machine": False}))
    assert type(flagged["reviewer_is_machine"]) is bool


def test_the_deliberately_textual_values_stay_textual() -> None:
    """The sentinels are what a reader tells one absence from another with.

    ``UNKNOWN_MODEL`` says "a model was not named" where an absent key would say
    "this writer never looked", and those are different statements. A type change
    that turned a sentinel into a bare value would collapse them, so they are pinned
    as ``str`` alongside the numbers.
    """
    entry = make_entry()
    frontmatter = build_frontmatter(entry)

    assert isinstance(frontmatter["capture_source"], str)
    assert isinstance(frontmatter["category"], str)
    assert isinstance(frontmatter["author"], str)
    # A timestamp is ISO-8601 text. A number here would be a different kind of
    # wrong rather than a better-typed one.
    assert frontmatter["updated_at"] == entry.answered_at.isoformat()

    clarified = build_clarification_frontmatter(
        ClarificationEntry(
            entry_id="clarification-v1-a",
            repo="org/repo",
            pr_number=42,
            statement="The lease is intentional.",
            author="davy",
            author_association="OWNER",
            github_comment_id=201,
            declared_at=datetime(2026, 3, 4, 10, 0, tzinfo=UTC),
            captured_at=datetime(2026, 3, 4, 10, 0, tzinfo=UTC),
            capture_source=CaptureSource.COLLECT,
        )
    )
    # An unnamed model is written as the sentinel rather than omitted, so a reader
    # can tell "named no model" from "not recorded here".
    assert clarified["clarified_by_model"] == UNKNOWN_MODEL
    assert isinstance(clarified["clarified_by_model"], str)


def test_an_anchored_record_still_says_nothing_about_its_structure() -> None:
    """Presence is not what this change touches.

    ``structure`` is written only when it is not ``anchored``, so an anchored
    document stays byte-identical to one stored before the axis existed. The
    golden-bytes test in ``tests/test_provenance.py`` holds the whole block; this
    pins the one key, because a numeric value changing elsewhere would not fail it
    on its own.
    """
    assert "structure" not in build_frontmatter(make_entry())
    inferred = build_frontmatter(
        make_entry(
            structure=RecordStructure.INFERRED,
            metadata={STRUCTURE_INFERRED_BY: "opencode/model"},
        )
    )
    assert inferred["structure"] == "inferred"
    assert isinstance(inferred["structure"], str)


def _design_entry(
    metadata: dict[str, object],
    *,
    entry_id: str = "design-v1-x",
    declared_by: str = "carol",
    text: str = "Reconcile toward one plan.",
) -> RationaleEntry:
    """One design record with exactly the metadata the capture phase writes."""
    return RationaleEntry(
        entry_id=entry_id,
        repo="org/repo",
        pr_number=None,
        branch="postmortem for the sync stall",
        declared_by=declared_by,
        declared_model="opencode/model",
        rationale_text=text,
        source=RationaleSource.DECLARED,
        revision=1,
        declared_at=datetime(2026, 3, 4, 10, 0, tzinfo=UTC),
        metadata=metadata,
    )


def _rendered_design(metadata: dict[str, object]) -> str:
    """The document a reader holds, not the builder's dict.

    Every assertion below reads this string, because the builder is one step
    from the document and ``serialize_frontmatter`` is where absence and typing
    are decided. A test on the dict would pass while the document dropped the
    key.
    """
    return serialize_frontmatter(build_rationale_frontmatter(_design_entry(metadata)))


def test_a_proposal_renders_its_role_and_no_reconciliation_keys() -> None:
    """A proposal is not a reconciliation, and the document must not imply it is.

    The capture phase writes only the role onto a proposal -- no digest, no
    proposal ids, no discards, because there is nothing to put in them. Those
    keys are absent rather than empty: ``metadata.get`` yields ``None`` and the
    shared absence rule drops them, which is the existing predicate and not a
    new one written for this case.
    """
    rendered = _rendered_design({"design_role": "proposal"})
    assert '"design_role": "proposal"' in rendered
    assert "design_plan_digest" not in rendered
    assert "design_proposal_ids" not in rendered
    assert "design_discarded" not in rendered


def test_a_reconciliation_renders_all_four_keys_with_values_intact() -> None:
    """The digest, the ids and the discards reach the document as what they are.

    The digest is a 64-hex string and must render as a plain scalar -- not quoted
    into a different shape, not folded across lines. The proposal ids stay a
    list of strings. The type is the payload: a digest that arrived stringified
    would read as recorded while comparing against nothing.
    """
    digest = "ab" * 32
    rendered = _rendered_design(
        {
            "design_role": "reconciliation",
            "design_plan_digest": digest,
            "design_proposal_ids": ["design-proposal-v1-a", "design-proposal-v1-b"],
            "design_discarded": [
                {
                    "entry_id": "design-proposal-v1-b",
                    "principal": "bob",
                    "reason": "Argued for a second store; one store holds the pause.",
                }
            ],
        }
    )
    assert f'"design_plan_digest": "{digest}"' in rendered
    assert '"design_role": "reconciliation"' in rendered
    assert "design-proposal-v1-a" in rendered
    assert "design-proposal-v1-b" in rendered
    assert "Argued for a second store" in rendered
    parsed = yaml.safe_load(rendered)
    assert parsed["design_plan_digest"] == digest
    assert isinstance(parsed["design_plan_digest"], str)
    assert parsed["design_proposal_ids"] == ["design-proposal-v1-a", "design-proposal-v1-b"]
    assert parsed["design_discarded"] == [
        {
            "entry_id": "design-proposal-v1-b",
            "principal": "bob",
            "reason": "Argued for a second store; one store holds the pause.",
        }
    ]


def test_an_empty_discard_list_is_a_result_and_renders_as_one() -> None:
    """``design_discarded: []`` means "considered and discarded none".

    This is the deliberate asymmetry in :func:`_unrecorded`: an empty collection
    is a result rather than a gap, because ``None`` is "nobody looked" and ``[]``
    is "nobody found anything". A reconciliation that kept every proposal makes
    a claim worth keeping -- collapsing it into absence would make an outage
    look like an answer, which is the failure the shared predicate exists to
    prevent. So the key renders, empty, and the test pins that rather than
    letting a future tidy-up "fix" it into silence.
    """
    rendered = _rendered_design(
        {
            "design_role": "reconciliation",
            "design_plan_digest": "cd" * 32,
            "design_proposal_ids": ["design-proposal-v1-a"],
            "design_discarded": [],
        }
    )
    assert '"design_discarded": []' in rendered


def test_a_design_reason_sanitised_upstream_stays_sanitised_and_said_so() -> None:
    """The silence-is-not-a-fidelity-claim case, for model-authored metadata.

    The capture phase sanitises a discard reason before storing it and records
    what was removed under ``text_sanitisation``. The projection carries both
    through untouched: the document shows the cleaned reason, and the note says
    what was taken out. A document with the note missing would read as the
    declaration's own wording while not being it -- the exact defect the note
    exists to prevent.
    """
    from kojutsu.core.text_hygiene import SANITISATION_KEY, describe_removals, sanitise

    raw = "Argued for a second store\x07; one store holds the pause."
    field = sanitise(raw)
    rendered = _rendered_design(
        {
            "design_role": "reconciliation",
            "design_plan_digest": "ef" * 32,
            "design_proposal_ids": ["design-proposal-v1-a"],
            "design_discarded": [
                {
                    "entry_id": "design-proposal-v1-a",
                    "principal": "bob",
                    "reason": field.text,
                }
            ],
            SANITISATION_KEY: describe_removals(raw),
        }
    )
    assert "\x07" not in rendered
    assert "Argued for a second store" in rendered
    assert "U+0007" in rendered
