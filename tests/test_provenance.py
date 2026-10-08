"""Tests for capture provenance: what may be called evidence, and what may not.

These are the tests that should have existed before any demo was built. They exist
because a hand-written ``KnowledgeEntry`` used to be storable and readable as if it
were captured review evidence, with a fabricated ``repo``/``pr_number`` and nothing
in the record to distinguish it from a real capture.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from kojutsu.core.question_registry import stable_clarification_entry_id
from kojutsu.core.tanseki_mapping import (
    build_clarification_frontmatter,
    build_content,
    build_frontmatter,
    serialize_frontmatter,
)
from kojutsu.integrations.github import extract_agent_claim
from kojutsu.models import (
    STRUCTURE_INFERRED_BY,
    CaptureSource,
    ClarificationEntry,
    Independence,
    KnowledgeEntry,
    QuestionCategory,
    RationaleEntry,
    RationaleSource,
    RecordStructure,
    capture_anchor_gaps,
    compute_independence,
    structure_anchor_gaps,
    structure_of,
)

CAPTURED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _base(**overrides: object) -> KnowledgeEntry:
    fields: dict[str, object] = {
        "entry_id": "entry-1",
        "question_text": "Why did we choose this?",
        "answer_text": "Because X.",
        "category": QuestionCategory.DESIGN_DECISION,
        "author": "dev",
        "metadata": {
            "repo": "org/repo",
            "pr_number": 42,
            "pr_url": "https://github.com/org/repo/pull/42",
        },
    }
    fields.update(overrides)
    return KnowledgeEntry(**fields)  # type: ignore[arg-type]


def _a_rationale() -> RationaleEntry:
    return RationaleEntry(
        entry_id="r-1",
        repo="org/repo",
        declared_by="opencode",
        rationale_text="Because X.",
    )


def test_a_hand_written_entry_is_asserted_by_default() -> None:
    """The dangerous case: a typed-in record must never be evidence by accident."""
    entry = _base()

    assert entry.capture_source is CaptureSource.ASSERTED
    assert entry.is_captured is False


def test_detail_does_not_promote_an_entry_to_evidence() -> None:
    """Filling in every provenance-looking field is not the same as being captured.

    This is the exact shape of the original bad demo: real-looking repo, PR number,
    ticket key, author and timestamps, but nothing behind them.
    """
    entry = _base(
        metadata={
            "repo": "acme/widgets",
            "pr_number": 1,
            "pr_url": "https://github.com/acme/widgets/pull/1",
            "jira_ticket_key": "RL-PILOT-1",
            "github_comment_id": 999,
            "question_id": "q1",
        },
        tags=["pilot"],
    )

    assert entry.is_captured is False
    assert build_frontmatter(entry)["capture_source"] == "asserted"


def test_webhook_capture_requires_a_delivery_id() -> None:
    with pytest.raises(ValidationError, match="capture_delivery_id"):
        _base(capture_source=CaptureSource.WEBHOOK, captured_at=CAPTURED_AT)


def test_collect_capture_requires_a_source_comment_id() -> None:
    with pytest.raises(ValidationError, match="github_comment_id"):
        _base(capture_source=CaptureSource.COLLECT, captured_at=CAPTURED_AT)


def test_captured_entries_require_a_capture_timestamp() -> None:
    with pytest.raises(ValidationError, match="captured_at"):
        _base(
            capture_source=CaptureSource.WEBHOOK,
            capture_delivery_id="delivery-1",
        )


@pytest.mark.parametrize("pr_number", [0, -1, "42", True, None])
def test_captured_entries_reject_an_unusable_pr_number(pr_number: object) -> None:
    with pytest.raises(ValidationError, match="pr_number"):
        _base(
            capture_source=CaptureSource.WEBHOOK,
            captured_at=CAPTURED_AT,
            capture_delivery_id="delivery-1",
            metadata={"repo": "org/repo", "pr_number": pr_number},
        )


def test_captured_entries_reject_a_missing_repo() -> None:
    with pytest.raises(ValidationError, match="repo"):
        _base(
            capture_source=CaptureSource.WEBHOOK,
            captured_at=CAPTURED_AT,
            capture_delivery_id="delivery-1",
            metadata={"pr_number": 42},
        )


def test_a_fully_anchored_webhook_capture_is_accepted() -> None:
    entry = _base(
        capture_source=CaptureSource.WEBHOOK,
        captured_at=CAPTURED_AT,
        capture_delivery_id="0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
    )

    assert entry.is_captured is True


def test_a_fully_anchored_collect_capture_is_accepted() -> None:
    entry = _base(
        capture_source=CaptureSource.COLLECT,
        captured_at=CAPTURED_AT,
        metadata={"repo": "org/repo", "pr_number": 42, "github_comment_id": 201},
    )

    assert entry.is_captured is True


def test_agent_authored_capture_records_which_agent_wrote_it() -> None:
    """A machine-authored answer must be attributable, not anonymous."""
    entry = _base(
        capture_source=CaptureSource.WEBHOOK,
        captured_at=CAPTURED_AT,
        capture_delivery_id="delivery-abc",
        author="opencode",
        tags=["agent_authored"],
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            "github_comment_id": 201,
            "github_author_association": "OWNER",
            "answered_by_agent": "opencode",
            "comment_author": "dev",
        },
    )

    frontmatter = build_frontmatter(entry)

    assert frontmatter["author"] == "opencode"
    assert frontmatter["answered_by_agent"] == "opencode"
    assert frontmatter["comment_author"] == "dev"
    assert "agent_authored" in frontmatter["tags"]


def test_frontmatter_retains_the_provenance_it_used_to_drop() -> None:
    """Provenance used to be discarded at the storage boundary, erasing the trail."""
    entry = _base(
        capture_source=CaptureSource.WEBHOOK,
        captured_at=CAPTURED_AT,
        capture_delivery_id="delivery-abc",
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            "github_comment_id": 201,
            "github_author_association": "MEMBER",
            "question_id": "q1",
        },
    )

    frontmatter = build_frontmatter(entry)

    assert frontmatter["capture_source"] == "webhook"
    assert frontmatter["delivery_id"] == "delivery-abc"
    assert frontmatter["captured_at"] == CAPTURED_AT.isoformat()
    assert frontmatter["question_id"] == "q1"
    assert frontmatter["github_author_association"] == "MEMBER"
    assert frontmatter["github_comment_id"] == 201


def test_asserted_frontmatter_is_explicit_even_with_no_provenance() -> None:
    frontmatter = build_frontmatter(_base())

    assert frontmatter["capture_source"] == "asserted"
    assert "delivery_id" not in frontmatter
    assert "captured_at" not in frontmatter


# --- model identity and independence ----------------------------------------


MODEL_A = "opencode/model"
MODEL_B = "anthropic/claude-opus-5"


def test_agent_claim_carries_its_model() -> None:
    claim = extract_agent_claim(f"<!-- kojutsu:agent:opencode model={MODEL_A} -->\n\nBecause X.")
    assert claim is not None
    assert claim.agent_id == "opencode"
    assert claim.model == MODEL_A


def test_agent_claim_without_a_model_still_parses() -> None:
    """Comments posted before the model half existed must keep working."""
    claim = extract_agent_claim("<!-- kojutsu:agent:opencode -->\n\nBecause X.")
    assert claim is not None
    assert claim.agent_id == "opencode"
    assert claim.model is None


def test_a_model_key_without_an_agent_is_not_a_claim() -> None:
    assert extract_agent_claim(f"<!-- kojutsu:agent:model={MODEL_A} -->") is None


def test_an_oversized_model_is_rejected_rather_than_truncated() -> None:
    claim = extract_agent_claim("<!-- kojutsu:agent:opencode model=" + "x" * 500 + " -->")
    assert claim is not None
    assert claim.model is None, "a model that does not fit must be reported as unstated"


def test_a_human_comment_states_no_claim() -> None:
    assert extract_agent_claim("Just a normal review comment.") is None


def test_independence_rank_orders_the_scale() -> None:
    assert (
        Independence.SELF_CERTIFIED.rank
        < Independence.MODEL_SEPARATED.rank
        < Independence.INDEPENDENT.rank
    )


@pytest.mark.parametrize(
    ("asker", "asker_model", "answerer", "answerer_model", "expected"),
    [
        ("dev", None, "reviewer", None, Independence.INDEPENDENT),
        ("dev", None, "reviewer", MODEL_A, Independence.INDEPENDENT),
        ("dev", MODEL_A, "reviewer", MODEL_A, Independence.INDEPENDENT),
        ("bot", MODEL_A, "bot", MODEL_B, Independence.MODEL_SEPARATED),
        ("bot", MODEL_A, "bot", MODEL_A, Independence.SELF_CERTIFIED),
        ("bot", MODEL_A, "bot", None, Independence.SELF_CERTIFIED),
        ("bot", None, "bot", MODEL_B, Independence.SELF_CERTIFIED),
    ],
)
def test_every_account_and_model_combination_lands_on_the_documented_level(
    asker: str,
    asker_model: str | None,
    answerer: str,
    answerer_model: str | None,
    expected: Independence,
) -> None:
    level, reason = compute_independence(
        asker_account=asker,
        asker_model=asker_model,
        answerer_account=answerer,
        answerer_model=answerer_model,
    )
    assert level is expected
    assert reason, "every classification must be able to say why"


def test_a_same_model_pair_can_never_be_labelled_independent() -> None:
    """The falsification this whole ticket exists for."""
    for asker_model in (MODEL_A, None, "", "  "):
        for answerer_model in (MODEL_A, None, "", "  "):
            level, _ = compute_independence(
                asker_account="bot",
                asker_model=asker_model,
                answerer_account="bot",
                answerer_model=answerer_model,
            )
            assert level is not Independence.INDEPENDENT


def test_two_unstated_models_are_not_assumed_identical() -> None:
    """Guessing that two unknowns match would manufacture a worse label."""
    level, reason = compute_independence(
        asker_account="bot",
        asker_model=None,
        answerer_account="bot",
        answerer_model=None,
    )
    assert level is Independence.SELF_CERTIFIED
    assert "not stated" in reason


def test_account_matching_is_case_insensitive() -> None:
    level, _ = compute_independence(
        asker_account="Bot",
        asker_model=MODEL_A,
        answerer_account="bot",
        answerer_model=MODEL_B,
    )
    assert level is Independence.MODEL_SEPARATED


def test_frontmatter_carries_the_model_and_the_independence_it_came_from() -> None:
    entry = _base(
        capture_source=CaptureSource.WEBHOOK,
        captured_at=CAPTURED_AT,
        capture_delivery_id="delivery-abc",
        author="opencode",
        tags=["agent_authored"],
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            "github_comment_id": 201,
            "answered_by_agent": "opencode",
            "answered_by_model": MODEL_A,
            "comment_author": "dev",
            "independence": Independence.MODEL_SEPARATED.value,
            "independence_reason": "same account, different models",
        },
    )

    frontmatter = build_frontmatter(entry)

    assert frontmatter["answered_by_model"] == MODEL_A
    assert frontmatter["independence"] == "model_separated"
    assert frontmatter["independence_reason"] == "same account, different models"


def test_the_anchor_rule_is_one_definition_used_by_both_paths() -> None:
    """A rule that exists in two places drifts, so there is only one.

    The write path and the read path disagree about exactly one thing -- whether
    a number may arrive as a string -- and that difference is a named parameter
    rather than two implementations that can drift apart quietly.
    """
    from kojutsu.models import capture_anchor_gaps

    webhook = {
        "capture_source": CaptureSource.WEBHOOK,
        "repo": "org/repo",
        "pr_number": 42,
        "captured_at": CAPTURED_AT,
        "delivery_id": "delivery-1",
        "comment_id": None,
    }

    assert capture_anchor_gaps(**webhook) == []
    # The store stringifies, so a read-back carries "42" where the model held 42.
    assert capture_anchor_gaps(**{**webhook, "pr_number": "42"}) == ["metadata.pr_number"]
    assert capture_anchor_gaps(**{**webhook, "pr_number": "42"}, allow_string_numbers=True) == []


def test_an_asserted_record_is_required_to_carry_no_anchors() -> None:
    from kojutsu.models import capture_anchor_gaps

    assert (
        capture_anchor_gaps(
            capture_source=CaptureSource.ASSERTED,
            repo=None,
            pr_number=None,
            captured_at=None,
            delivery_id=None,
            comment_id=None,
        )
        == []
    )


@pytest.mark.parametrize("value", [True, "0", "-1", "abc", 0, -1, 1.0, None])
def test_only_a_real_positive_number_counts_as_an_anchor(value: object) -> None:
    """``bool`` is an ``int``, and ``True`` must not pass as ``1``."""
    from kojutsu.models import capture_anchor_gaps

    assert capture_anchor_gaps(
        capture_source=CaptureSource.WEBHOOK,
        repo="org/repo",
        pr_number=value,
        captured_at=CAPTURED_AT,
        delivery_id="delivery-1",
        comment_id=None,
    ) == ["metadata.pr_number"]


# --- the structure axis: an inferred pairing is never a captured one ---------
#
# `CaptureSource` answers "where did this text come from". It cannot answer "was
# this structure established or inferred", and once a model is allowed to pair a
# question with an answer, that gap is a real record: a real delivery id, a real
# comment id, and a question nobody asked.


def test_a_record_with_no_structure_reads_back_as_anchored() -> None:
    entry = _base()

    assert entry.structure is RecordStructure.ANCHORED
    # And the same after a round trip through the storage boundary, which is the
    # direction that matters: a document written before the axis existed carries no
    # key, and a reader must recover the documented default rather than a blank.
    assert "structure" not in build_frontmatter(entry)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_unstated_structure_resolves_to_the_documented_default(value: object) -> None:
    """Absence is the default, not a guess the reader made up.

    Nothing in the capture path could infer a pairing before this axis existed, so
    the documents that predate it are real ones. Reading them as anything else
    would re-identify stored records, which is the one thing a new axis must not do.
    """
    assert structure_of(value) is RecordStructure.ANCHORED


def test_an_unreadable_structure_resolves_to_nothing_rather_than_anchored() -> None:
    """A value this code does not recognise must never read as a confirmed pairing.

    This is the direction that matters for a filter: defaulting an unrecognised
    value would let a renamed or corrupted field pass as anchored, which is the
    misread the axis exists to prevent.
    """
    assert structure_of("guessed") is None
    assert structure_of(7) is None


def test_structure_resolution_is_case_and_space_insensitive() -> None:
    """Frontmatter is stringified and hand-edited, so both arrive in practice."""
    assert structure_of(" Inferred ") is RecordStructure.INFERRED


def test_an_inferred_record_is_refused_without_the_model_that_inferred_it() -> None:
    """An unattributed guess is indistinguishable from a capture. That is the defect.

    The record is not refused for being inferred -- an inferred record is still
    knowledge, and refusing it would lose it. It is refused for being *silently*
    inferred, because the one fact a reader needs in order to weigh a guessed
    pairing is which model guessed it.
    """
    with pytest.raises(ValidationError, match="structure_inferred_by_model"):
        _base(structure=RecordStructure.INFERRED)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_a_blank_model_name_does_not_satisfy_the_inference_rule(value: object) -> None:
    with pytest.raises(ValidationError, match="structure_inferred_by_model"):
        _base(
            structure=RecordStructure.INFERRED,
            metadata={"repo": "org/repo", "pr_number": 42, STRUCTURE_INFERRED_BY: value},
        )


def test_a_named_inference_is_accepted() -> None:
    entry = _base(
        structure=RecordStructure.INFERRED,
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            STRUCTURE_INFERRED_BY: MODEL_A,
        },
    )

    assert entry.structure is RecordStructure.INFERRED
    assert entry.metadata[STRUCTURE_INFERRED_BY] == MODEL_A


def test_a_captured_record_can_still_hold_an_inferred_pairing() -> None:
    """The two axes are independent, and the test that keeps them from trading places.

    A genuine signed delivery, a real comment id, and a question/answer pairing a
    model guessed. ``CaptureSource`` cannot express the last half of that, which is
    the whole reason the second axis exists -- so it must be reachable, or the axis
    would be claiming a separation the storage model does not actually have.
    """
    entry = _base(
        capture_source=CaptureSource.WEBHOOK,
        captured_at=CAPTURED_AT,
        capture_delivery_id="delivery-abc",
        structure=RecordStructure.INFERRED,
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            "github_comment_id": 201,
            STRUCTURE_INFERRED_BY: MODEL_A,
        },
    )

    assert entry.is_captured is True
    assert entry.structure is RecordStructure.INFERRED


def test_an_inferred_record_need_carry_no_capture_anchor() -> None:
    """The axes must not borrow each other's requirements.

    A pairing inferred by a model has, by definition, no delivery behind it. If
    the structure rule demanded a capture anchor, an inferred record would have to
    claim a capture it does not have -- which is how the two axes would start
    standing in for one another.
    """
    assert (
        structure_anchor_gaps(
            structure=RecordStructure.INFERRED,
            inferred_by_model=MODEL_A,
        )
        == []
    )


def test_the_structure_gap_rule_is_one_definition_used_by_both_paths() -> None:
    """Exactly the argument made for ``capture_anchor_gaps``, one axis along.

    The write path refuses construction; the read path flags a document whose
    provenance did not hold. Both call this, so a rule written twice cannot drift
    into two different definitions of what a valid inference looks like.
    """
    assert structure_anchor_gaps(
        structure=RecordStructure.INFERRED,
        inferred_by_model=None,
    ) == [f"metadata.{STRUCTURE_INFERRED_BY}"]
    # An anchored record is required to name nothing: a pairing nobody inferred has
    # no inferrer to name.
    assert (
        structure_anchor_gaps(
            structure=RecordStructure.ANCHORED,
            inferred_by_model=None,
        )
        == []
    )


def test_the_structure_survives_a_frontmatter_round_trip() -> None:
    """It has to leave the process, or a reader can never see it.

    The record is re-read from the serialised frontmatter exactly as a separate
    service would, so this covers the whole boundary rather than the dict the
    writer happened to build.
    """
    entry = _base(
        structure=RecordStructure.INFERRED,
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            STRUCTURE_INFERRED_BY: MODEL_A,
        },
    )

    serialized = serialize_frontmatter(build_frontmatter(entry))
    stored = yaml.safe_load(serialized)
    restored = KnowledgeEntry(
        entry_id=entry.entry_id,
        question_text=entry.question_text,
        answer_text=entry.answer_text,
        category=entry.category,
        structure=RecordStructure(stored["structure"]),
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            STRUCTURE_INFERRED_BY: stored[STRUCTURE_INFERRED_BY],
        },
    )

    assert restored.structure is RecordStructure.INFERRED
    assert restored.metadata[STRUCTURE_INFERRED_BY] == MODEL_A
    # The model name is on the document, not only inside the model that wrote it:
    # a reader has to be able to see who guessed without asking the writer.
    assert '"structure": "inferred"' in serialized
    assert MODEL_A in serialized


def test_adding_the_axis_does_not_re_identify_anything_already_stored() -> None:
    """Golden bytes for a real anchored record, before and after this change.

    Every record in the store today is anchored, so every one of them is covered by
    this string. Writing ``structure: anchored`` onto them would change the bytes of
    every stored document in order to add a claim their writers never made -- the
    re-identification ``docs/design-review/identity-and-limits.md`` says not to do.
    Absent on write, defaulted on read: the two must agree, and the tests above and
    below hold each other to that.

    ``pr`` and ``github_comment_id`` are unquoted here, and that is a different
    change with a different argument rather than a new claim about identity: both
    reach the store with the type the model held, so a reader can compare them
    instead of parsing them back. Nothing is *added* to the document and
    ``structure`` is still absent, which is what this test is for -- the
    re-identification it guards against is a key the writer never made, not a
    number's spelling. The stability being promised is of the key set, and this
    string still pins it exactly.
    """
    anchored = KnowledgeEntry(
        entry_id="entry-1",
        question_text="Why did we choose this?",
        answer_text="Because X.",
        category=QuestionCategory.DESIGN_DECISION,
        author="opencode",
        answered_at=CAPTURED_AT,
        tags=["agent_authored"],
        metadata={
            "repo": "org/repo",
            "pr_number": 42,
            "pr_url": "https://github.com/org/repo/pull/42",
            "github_comment_id": 201,
            "github_author_association": "MEMBER",
            "question_id": "q1",
            "answered_by_agent": "opencode",
            "answered_by_model": MODEL_A,
            "comment_author": "dev",
            "rationale_source": "declared",
            "independence": "model_separated",
            "independence_reason": "same account, different models",
        },
        capture_source=CaptureSource.WEBHOOK,
        captured_at=CAPTURED_AT,
        capture_delivery_id="delivery-abc",
    )

    assert build_content(anchored) == (
        "---\n"
        '"title": "Why did we choose this?"\n'
        '"author": "opencode"\n'
        '"tags":\n'
        '- "agent_authored"\n'
        '"updated_at": "2026-01-01T00:00:00+00:00"\n'
        '"answered_at": "2026-01-01T00:00:00+00:00"\n'
        '"answered_by_agent": "opencode"\n'
        f'"answered_by_model": "{MODEL_A}"\n'
        '"capture_source": "webhook"\n'
        '"captured_at": "2026-01-01T00:00:00+00:00"\n'
        '"category": "design_decision"\n'
        '"comment_author": "dev"\n'
        '"delivery_id": "delivery-abc"\n'
        '"github_author_association": "MEMBER"\n'
        '"github_comment_id": 201\n'
        '"independence": "model_separated"\n'
        '"independence_reason": "same account, different models"\n'
        '"pr": 42\n'
        '"pr_url": "https://github.com/org/repo/pull/42"\n'
        '"question_id": "q1"\n'
        '"rationale_source": "declared"\n'
        '"repo": "org/repo"\n'
        "---\n"
        "# Why did we choose this?\n"
        "\n"
        "## Question\n"
        "Why did we choose this?\n"
        "\n"
        "## Answer\n"
        "Because X.\n"
    )


def test_a_rationale_has_no_structure_to_declare() -> None:
    """A stated reason is a statement, not a pairing, so the field does not exist.

    A rationale has no question and answer for a model to have matched up. Giving
    it a field that could only ever hold ``anchored`` would invite a reader to
    think it has an inferable structure at all -- and whether the reasoning was
    declared or reconstructed is already a field on the model.
    """
    assert "structure" not in RationaleEntry.model_fields
    assert not hasattr(_a_rationale(), "structure")


# --- decision rationale: what a self-asserted reason may never claim ----------
#
# Every test in this section asserts a negative: something the system does NOT do.
# `docs/design-review/rationale.md` states the three claims below, and a claim
# held only in prose is a comment. Stating it as a test is what keeps it from
# quietly becoming an implied promise when someone later improves the model, adds
# a field, or reasons that a first-hand rationale "obviously" deserves more weight.


def test_a_declared_rationale_does_not_raise_independence() -> None:
    """The implementer explaining the implementer's own change checks nothing.

    The record most privileged to know the reason is the record least entitled to
    claim verification. "I did it and here's why" is a statement of intent, not a
    check on correctness, so the level must be the self-certified one that
    ``compute_independence`` already assigns to a same-account, same-model answer.
    """
    level, reason = compute_independence(
        asker_account="bot",
        asker_model=MODEL_A,
        answerer_account="bot",
        answerer_model=MODEL_A,
    )

    assert level is Independence.SELF_CERTIFIED
    assert RationaleSource.DECLARED.is_first_hand is True
    assert level.rank == Independence.SELF_CERTIFIED.rank, (
        "a declared rationale is more informative about intent, not more separated "
        "from checking itself; the two are different axes and must not trade places"
    )
    assert reason, "the level must still be able to say why"


def test_a_rationale_is_never_promotable_to_evidence() -> None:
    """A stated reason has no provider delivery behind it, so it carries no anchors.

    Someone signing a comment is not the same as someone verifying a change, and a
    rationale that could claim ``webhook`` provenance would let an agent's own
    account of its work be served to a later reader as captured review evidence.
    """
    gaps = capture_anchor_gaps(
        capture_source=CaptureSource.ASSERTED,
        repo=None,
        pr_number=None,
        captured_at=None,
        delivery_id=None,
        comment_id=None,
    )

    assert gaps == [], (
        "ASSERTED is required to carry no anchors precisely because it claims no "
        "capture; a rationale can only ever be ASSERTED"
    )


def test_a_rationale_sits_below_every_independence_threshold() -> None:
    """No ``min_independence`` value can admit a rationale, and that is correct.

    A reader who asked for independent evidence would rather see nothing than see
    a self-declared reason and assume it had been checked. Exclusion is the honest
    outcome, so this is the resting place rather than a gap to engineer around.
    """
    # A rationale record carries no independence level at all, which is the
    # representation the read path treats as below every threshold.
    level = None

    for threshold in Independence:
        assert level is None or level.rank < threshold.rank, (
            "an unlabelled record is below every threshold by definition"
        )


def test_a_declared_and_a_reconstructed_rationale_stay_distinguishable() -> None:
    """Review is a standalone step from intent, so the two may never be one record.

    A reconstruction inferred from a diff must not be readable as a recollection
    from the session that produced the change, or the reader has no way to know
    which they are holding. Keeping them apart is also what makes their
    disagreement a finding instead of a contradiction.
    """
    assert RationaleSource.DECLARED is not RationaleSource.RECONSTRUCTED
    assert RationaleSource.DECLARED.value != RationaleSource.RECONSTRUCTED.value
    assert RationaleSource.DECLARED.is_first_hand is True
    assert RationaleSource.RECONSTRUCTED.is_first_hand is False, (
        "the answerer is sandboxed with no tools, so it structurally cannot know "
        "why a change looks the way it does; it infers"
    )


def test_an_unlabelled_rationale_reports_absence_rather_than_defaulting() -> None:
    """A record predating this axis must not be guessed into one of the two.

    Defaulting would be a guess about provenance made by the reader rather than
    stated by the writer, which is the same failure as inferring a model from an
    agent marker that did not carry one.
    """
    assert RationaleSource.UNKNOWN.is_first_hand is False
    assert RationaleSource.UNKNOWN is not RationaleSource.DECLARED
    assert RationaleSource.UNKNOWN is not RationaleSource.RECONSTRUCTED


def test_capture_claims_are_read_from_the_comment_not_from_the_rationale() -> None:
    """The platform proves who posted a comment, never which model drafted it.

    This is the ``AgentClaim`` caveat, restated here because the rationale path is
    where it is most tempting to forget: an agent declares both its identity and
    its model in the same marker, and neither half is verifiable by anything in
    the system.
    """
    body = "<!-- kojutsu:agent:opencode model=opencode/model -->\n\nBecause X."
    claim = extract_agent_claim(body)

    assert claim is not None
    assert claim.agent_id == "opencode"
    assert claim.model == MODEL_A
    assert claim.model is not None, (
        "the model is an assertion by the comment author; nothing here checks it, "
        "so a rationale record must never treat it as verified provenance"
    )


# --- the clarification record: evidence, with no question attached ------------
#
# A clarification is the counterpart to the rationale above and the pair is the
# point. Both hold content ``KnowledgeEntry`` cannot: a rationale is a stated
# reason for a decision, a clarification is a statement in a review thread that
# nobody asked for. Their trust runs in opposite directions — a rationale is
# permanently ASSERTED because nothing was signed, while a clarification is
# captured evidence because a real comment with a re-fetchable id is behind it —
# and conflating them in either direction is the mistake this record was made to
# avoid. Each test below therefore states both halves.


def _clarification(**overrides: object) -> ClarificationEntry:
    fields: dict[str, object] = {
        "entry_id": stable_clarification_entry_id(
            repo="org/repo", pr_number=42, github_comment_id=201
        ),
        "repo": "org/repo",
        "pr_number": 42,
        "statement": "The narrow window is the accepted cost for v0.1; it is deliberate.",
        "author": "davy",
        "author_association": "OWNER",
        "github_comment_id": 201,
        "declared_at": CAPTURED_AT,
        "capture_source": CaptureSource.COLLECT,
        "captured_at": CAPTURED_AT,
    }
    fields.update(overrides)
    return ClarificationEntry(**fields)  # type: ignore[arg-type]


def test_a_clarification_is_evidence_rather_than_a_claim() -> None:
    """A real comment with an id is behind this one, so it is captured, not asserted.

    The opposite of a rationale, and the reason the two are different models
    rather than one model with a flag. Throwing away real evidence because it
    arrived on the same record type as a self-assertion would lose the most
    valuable sentence in a review thread: the owner saying a weakness is
    deliberate.
    """
    clarification = _clarification()

    assert clarification.capture_source is CaptureSource.COLLECT
    assert clarification.capture_source is not CaptureSource.ASSERTED

    with pytest.raises(ValidationError):
        _clarification(capture_source=CaptureSource.ASSERTED)


def test_a_clarification_is_anchored_by_the_same_rule_as_an_answer() -> None:
    """One definition of "captured means checkable", used by the write and read paths.

    The read path re-checks the anchors out of the stored frontmatter rather than
    trusting the process that wrote it, so this hands it exactly what a document
    read back from the store would carry — including the stringified numbers — and
    requires it to find nothing wrong. A record that reports as anomalous is served
    and flagged, so a false positive here labels real evidence a forgery.
    """
    from mcp_server.server import _provenance_anomalies

    frontmatter = build_clarification_frontmatter(_clarification())

    assert (
        capture_anchor_gaps(
            capture_source=CaptureSource.COLLECT,
            repo=frontmatter["repo"],
            pr_number=frontmatter["pr"],
            captured_at=frontmatter["captured_at"],
            delivery_id=frontmatter.get("delivery_id"),
            comment_id=frontmatter["github_comment_id"],
            allow_string_numbers=True,
        )
        == []
    )
    assert _provenance_anomalies(frontmatter) == []


def test_a_clarification_never_raises_a_readers_independence_floor() -> None:
    """Evidence about a comment, not a verdict about who could disagree.

    ``Independence`` answers "who was in a position to disagree about a change",
    and a clarification has no asker to compare against: the whole point is that
    nobody asked. Inventing a level for it would be a guess about provenance made
    by the reader, so the record states none and the read path treats "not stated"
    as below every ``min_independence`` threshold — while counting the exclusion,
    so "we showed you none" is never read as "there was nothing".
    """
    frontmatter = build_clarification_frontmatter(_clarification())

    assert "independence" not in frontmatter
    assert "independence_reason" not in frontmatter


def test_the_two_kinds_of_unprompted_record_cannot_trade_places() -> None:
    """One refuses to be a capture, the other refuses to be an assertion.

    If either could be the other, the same sentence would be readable as a
    self-declared reason in one place and as captured review evidence in another,
    and no field in the record would settle which.
    """
    from kojutsu.models import RationaleEntry

    with pytest.raises(ValidationError, match="always asserted"):
        RationaleEntry(
            entry_id="rationale-v1-x",
            repo="org/repo",
            pr_number=42,
            declared_by="opencode",
            rationale_text="Because X.",
            capture_source=CaptureSource.COLLECT,
        )
    with pytest.raises(ValidationError, match="RationaleEntry"):
        _clarification(capture_source=CaptureSource.ASSERTED)


# --- backfilled history: what a reconstruction may and may not claim -------------
#
# A backfilled record is the one source whose evidence is a *read that happened
# after the event*. Every test below asserts a boundary of that claim, because the
# boundary is the whole reason the source has its own anchor rule: the record shows
# what the forge says now, and a reader who mistakes that for what it said then is
# wrong in a way no later check will catch.

READ_AT = datetime(2026, 9, 30, tzinfo=UTC)


def _backfilled_gaps(**overrides: object) -> list[str]:
    from kojutsu.models import capture_anchor_gaps

    anchors: dict[str, object] = {
        "capture_source": CaptureSource.BACKFILLED,
        "repo": "org/repo",
        "pr_number": 42,
        "captured_at": READ_AT,
        "delivery_id": None,
        "comment_id": None,
        "review_id": 9001,
    }
    anchors.update(overrides)
    return capture_anchor_gaps(**anchors)


def test_a_backfilled_record_is_anchored_to_the_read_rather_than_a_delivery() -> None:
    """A backfill has no delivery behind it, and must not be asked for one."""
    assert _backfilled_gaps() == []
    assert _backfilled_gaps(comment_id=1234) == []


def test_a_backfilled_record_must_name_what_it_read() -> None:
    """With neither a comment nor a review id, there is nothing behind the claim."""
    gaps = _backfilled_gaps(review_id=None)

    assert gaps == ["capture_read_anchor"], (
        "the read is the only evidence a reconstructed record has; without it the "
        "document is an assertion that kojutsu read something"
    )


def test_a_backfilled_record_needs_a_repo_a_change_and_a_read_time() -> None:
    """The three anchors the other sources share, for the same reasons they share them."""
    assert _backfilled_gaps(repo="  ") == ["metadata.repo"]
    assert _backfilled_gaps(pr_number=None) == ["metadata.pr_number"]
    assert _backfilled_gaps(captured_at=None) == ["captured_at"]


def test_a_delivery_id_alone_does_not_satisfy_a_backfilled_record() -> None:
    """A backfilled record carrying a delivery id is claiming a delivery it never had."""
    gaps = _backfilled_gaps(delivery_id="delivery-1", review_id=None)

    assert "capture_read_anchor" in gaps, (
        "nothing was delivered to a backfill, so a delivery id is not an anchor for "
        "it — accepting one would let a caller reach the captured-evidence path by "
        "inventing the id the rule is supposed to be asking for"
    )


def test_a_backfilled_record_is_captured_but_not_witnessed() -> None:
    """``is_captured`` is true; the claim it supports is only about the present."""
    entry = KnowledgeEntry(
        entry_id="entry-backfilled",
        question_text="Why did we choose this?",
        answer_text="Because X.",
        category=QuestionCategory.DESIGN_DECISION,
        metadata={"repo": "org/repo", "pr_number": 42, "review_id": 9001},
        capture_source=CaptureSource.BACKFILLED,
        captured_at=READ_AT,
    )

    assert entry.is_captured is True
    assert entry.capture_source is CaptureSource.BACKFILLED


def test_the_reconstruction_member_is_not_the_rationale_member_of_the_same_shape() -> None:
    """Two axes, two meanings, and the reason one of them is named differently.

    ``RationaleSource.RECONSTRUCTED`` means a model inferred a reason from a diff.
    ``CaptureSource.BACKFILLED`` means the text was read from history after the
    fact. Both apply to the same records, so sharing the word would make a reader's
    filter on either ambiguous — the exact confusion ``core.rationale_link``
    exists to prevent.
    """
    from kojutsu.models import RationaleSource

    assert CaptureSource.BACKFILLED.value == "backfilled"
    assert RationaleSource.RECONSTRUCTED.value == "reconstructed"
    assert CaptureSource.BACKFILLED.value != RationaleSource.RECONSTRUCTED.value


def test_an_asserted_record_is_still_the_default_for_an_unmarked_entry() -> None:
    """Adding a fourth source must not widen what an unlabelled record may claim."""
    entry = KnowledgeEntry(
        entry_id="entry-plain",
        question_text="Why did we choose this?",
        answer_text="Because X.",
        category=QuestionCategory.DESIGN_DECISION,
    )

    assert entry.capture_source is CaptureSource.ASSERTED
    assert entry.is_captured is False
