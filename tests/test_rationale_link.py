"""Tests for linking a stated rationale to an inferred one across a change.

The test that matters most in this file is
``test_agreement_from_the_same_principal_is_reported_as_a_restatement``. It is the
one a future change is most likely to break, and the one whose failure would be
most damaging: a restatement presented as corroboration is the exact
manufactured-consensus outcome this product exists to prevent, arriving through
the feature built to prevent it.
"""

from __future__ import annotations

import pytest

from kojutsu.core.answer_collector import process_comment_reply_outcome
from kojutsu.core.answerer import draft_answers
from kojutsu.core.rationale_link import (
    ComparisonOutcome,
    RationaleSummary,
    compare_rationales,
    render_comparison,
)
from kojutsu.core.tanseki_mapping import build_frontmatter
from kojutsu.integrations.github import (
    answer_comment_body_as_agent,
    extract_agent_claim,
)
from kojutsu.models import (
    Independence,
    KnowledgeEntry,
    RationaleSource,
)

MODEL_A = "opencode/model"
MODEL_B = "anthropic/claude-opus-5"


def _summary(
    *,
    source: RationaleSource,
    by: str = "opencode",
    model: str | None = MODEL_A,
    text: str = "Chose exponential backoff for idempotency.",
    revision: int = 1,
) -> RationaleSummary:
    return RationaleSummary(
        entry_id=f"rationale-v1-{source.value}",
        source=source,
        declared_by=by,
        model=model,
        revision=revision,
        text=text,
    )


# --- the review rationale is labelled inferred ---------------------------------


def test_an_answerer_answer_declares_itself_reconstructed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reviewer is sandboxed with no tools, so it structurally cannot know.

    Without this label a reader holds a reconstruction and has no way to tell it
    from the agent's own recollection of why it wrote the code.
    """

    def fake_complete_task(prompt: str, config: object, **kwargs: object) -> str:
        return "The change looks correct."

    monkeypatch.setattr("kojutsu.core.answerer.complete_task", fake_complete_task)

    plans = draft_answers(
        [{"question_id": "q1", "question_text": "Why?"}],
        diff="+x = 1",
        pr_title="T",
        model=MODEL_A,
        agent="opencode",
    )

    claim = extract_agent_claim(plans[0].body)
    assert claim is not None
    assert claim.source is RationaleSource.RECONSTRUCTED
    assert claim.source.is_first_hand is False


def test_a_human_answer_carries_no_source_claim() -> None:
    """Absence is reported as absence, not defaulted to either real value.

    Defaulting would be the reader guessing at provenance the writer never
    claimed -- the same rule an unstated model already follows.
    """
    claim = extract_agent_claim("<!-- kojutsu:agent:opencode -->\n\nBecause X.")
    assert claim is not None
    assert claim.source is RationaleSource.UNKNOWN


def test_an_unrecognised_source_is_reported_as_unknown_not_guessed() -> None:
    claim = extract_agent_claim("<!-- kojutsu:agent:opencode source=psychic -->\n\nX.")
    assert claim is not None
    assert claim.source is RationaleSource.UNKNOWN


def test_the_model_key_does_not_swallow_the_source_key() -> None:
    """An earlier parser stopped at the first match, so the third key was dropped.

    A claim written by the tool and quietly lost on the way back in is the same
    silent-drop failure the frontmatter whitelist had once, and it would have
    made every reconstructed label read as unknown.
    """
    body = answer_comment_body_as_agent(
        "q1", "Because X.", "opencode", MODEL_A, RationaleSource.DECLARED
    )

    claim = extract_agent_claim(body)

    assert claim is not None
    assert claim.model == MODEL_A
    assert claim.source is RationaleSource.DECLARED


def test_the_collector_records_the_source_on_the_captured_entry() -> None:
    """The label has to reach the stored record, or nothing downstream can use it."""

    class Sink:
        def __init__(self) -> None:
            self.entries: list[KnowledgeEntry] = []

        def store(self, entry: KnowledgeEntry):
            self.entries.append(entry)

    class Registry:
        def answer_comment_seen(self, _cid: int) -> bool:
            return False

        def is_question_answered(self, _pid: int) -> bool:
            return False

        def get_question_by_comment_id(self, _pid: int) -> dict[str, object]:
            return {
                "question_id": "q1",
                "question_text": "Why?",
                "question_author": "dev",
                "category": "design_decision",
                "session_id": None,
                "pr_url": "https://github.com/org/repo/pull/42",
                "jira_ticket_key": None,
            }

        def claim_answer(self, **kwargs: object) -> str:
            return "token"

        def complete_answer(self, _cid: int, _token: str) -> bool:
            return True

        def release_answer(self, *_args: object) -> bool:
            return True

    sink = Sink()
    body = answer_comment_body_as_agent(
        "q1", "Because X.", "opencode", MODEL_A, RationaleSource.RECONSTRUCTED
    )
    process_comment_reply_outcome(
        new_comment_id=201,
        new_comment_body=body,
        new_comment_author="dev",
        new_comment_created_at=None,
        new_comment_author_association="MEMBER",
        parent_comment_id=101,
        repo="org/repo",
        pr_number=42,
        registry=Registry(),  # type: ignore[arg-type]
        sink=sink,  # type: ignore[arg-type]
    )

    assert len(sink.entries) == 1
    assert sink.entries[0].metadata["rationale_source"] == "reconstructed"
    assert build_frontmatter(sink.entries[0])["rationale_source"] == "reconstructed"


# --- the comparison, and what it is not ---------------------------------------


def test_agreement_from_the_same_principal_is_reported_as_a_restatement() -> None:
    """The test this whole module exists to make impossible to get wrong.

    Two rationales from the same account on the same model agreeing is the
    *expected* outcome and teaches nothing. Reporting it as corroboration is the
    manufactured-consensus failure arriving through the feature meant to prevent
    it.
    """
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, text="Backoff, for idempotency."),
        _summary(source=RationaleSource.RECONSTRUCTED, text="Backoff, for idempotency."),
    )

    assert comparison.outcome is ComparisonOutcome.RESTATEMENT
    assert comparison.is_informative is False
    assert "NO information" in comparison.note
    assert comparison.independence is Independence.SELF_CERTIFIED


def test_disagreement_from_the_same_principal_is_still_a_restatement() -> None:
    """Reaching different words from the same mind is not a second opinion either."""
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, text="For idempotency."),
        _summary(source=RationaleSource.RECONSTRUCTED, text="For rate limiting."),
    )

    assert comparison.outcome is ComparisonOutcome.RESTATEMENT
    assert comparison.is_informative is False


def test_agreement_across_principals_is_a_genuine_second_opinion() -> None:
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", model=MODEL_A),
        _summary(source=RationaleSource.RECONSTRUCTED, by="reviewer", model=MODEL_B),
    )

    assert comparison.outcome is ComparisonOutcome.CONCURRENT
    assert comparison.independence is Independence.INDEPENDENT
    assert comparison.is_informative is True


def test_disagreement_across_principals_is_the_finding() -> None:
    """The case the comparison exists for: the intent is not in the change."""
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", model=MODEL_A, text="For idempotency."),
        _summary(
            source=RationaleSource.RECONSTRUCTED,
            by="reviewer",
            model=MODEL_B,
            text="For rate limiting.",
        ),
    )

    assert comparison.outcome is ComparisonOutcome.DIVERGENT
    assert comparison.independence is Independence.INDEPENDENT
    assert "not visible" in comparison.note


def test_different_models_on_one_account_is_a_second_mind_not_a_second_party() -> None:
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", model=MODEL_A),
        _summary(
            source=RationaleSource.RECONSTRUCTED,
            by="bot",
            model=MODEL_B,
            text="For rate limiting.",
        ),
    )

    assert comparison.independence is Independence.MODEL_SEPARATED
    assert comparison.outcome is ComparisonOutcome.DIVERGENT, (
        "a second model reaching a different reason is a second mind, and the "
        "disagreement is the finding -- it is only a restatement when it is the "
        "same mind as well as the same account"
    )


def test_an_unstated_model_does_not_manufacture_a_separation() -> None:
    """Two unknowns are not assumed identical, and the record says why."""
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", model=None),
        _summary(source=RationaleSource.RECONSTRUCTED, by="bot", model=None),
    )

    assert comparison.independence is Independence.SELF_CERTIFIED
    assert "not stated" in comparison.independence_reason


@pytest.mark.parametrize(
    ("declared", "reconstructed", "expected"),
    [
        (True, True, ComparisonOutcome.RESTATEMENT),
        (True, False, ComparisonOutcome.DECLARED_ONLY),
        (False, True, ComparisonOutcome.RECONSTRUCTED_ONLY),
        (False, False, ComparisonOutcome.NEITHER),
    ],
)
def test_a_missing_rationale_is_named_rather_than_treated_as_agreement(
    declared: bool, reconstructed: bool, expected: ComparisonOutcome
) -> None:
    """Absence of evidence is not evidence of agreement.

    The both-present case here is ``RESTATEMENT``, not ``DIVERGENT``: the two
    summaries are identical text from the same default principal, so a reader must
    be told it taught them nothing rather than shown two texts that happen to
    differ nowhere.
    """
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED) if declared else None,
        _summary(source=RationaleSource.RECONSTRUCTED) if reconstructed else None,
    )

    assert comparison.outcome is expected
    if expected is ComparisonOutcome.RESTATEMENT:
        assert "NO information" in comparison.note
    else:
        assert "not" in comparison.note.lower() or "No rationale" in comparison.note


def test_rationales_the_budget_did_not_reach_are_named() -> None:
    """A bounded comparison must never read as a complete one.

    A report that compared two of five and said "no divergences found" is worse
    than one that says it compared two and names the three it did not reach.
    """
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot"),
        _summary(source=RationaleSource.RECONSTRUCTED, by="reviewer"),
        not_reached=("rationale-v1-c", "rationale-v1-d", "rationale-v1-e"),
    )

    rendered = render_comparison(comparison)
    assert "NOT COMPARED" in rendered
    for entry_id in ("rationale-v1-c", "rationale-v1-d", "rationale-v1-e"):
        assert entry_id in rendered


# --- rendering: the caveat comes before the prose ------------------------------


def test_the_rendering_states_the_independence_before_either_text() -> None:
    """A restatement that reads as two agreeing opinions is the failure mode.

    Presenting the texts first is how that happens, so the level and the note lead.
    """
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", text="Same words."),
        _summary(source=RationaleSource.RECONSTRUCTED, by="bot", text="Same words."),
    )

    rendered = render_comparison(comparison)

    assert rendered.index("Independence") < rendered.index("Stated by")
    assert rendered.index("Independence") < rendered.index("Inferred by")
    assert "NO information" in rendered


def test_the_rendering_names_both_principals_and_their_models() -> None:
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", model=MODEL_A),
        _summary(source=RationaleSource.RECONSTRUCTED, by="reviewer", model=MODEL_B),
    )

    rendered = render_comparison(comparison)

    assert "Stated by bot" in rendered
    assert "Inferred by reviewer" in rendered
    assert MODEL_A in rendered
    assert MODEL_B in rendered


def test_a_missing_model_is_rendered_as_unstated_rather_than_blank() -> None:
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, by="bot", model=None),
        _summary(source=RationaleSource.RECONSTRUCTED, by="reviewer", model=None),
    )

    assert "model not stated" in render_comparison(comparison)


def test_the_two_records_are_never_merged_into_one() -> None:
    """A combined view would hide which was stated and which was inferred."""
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, text="For idempotency."),
        _summary(source=RationaleSource.RECONSTRUCTED, text="For rate limiting."),
    )

    assert comparison.declared is not None
    assert comparison.reconstructed is not None
    assert comparison.declared.entry_id != comparison.reconstructed.entry_id
    assert comparison.declared.source is RationaleSource.DECLARED
    assert comparison.reconstructed.source is RationaleSource.RECONSTRUCTED


def test_only_the_latest_revision_of_a_declaration_is_compared() -> None:
    """An early intent is usually the one a later revision contradicts."""
    comparison = compare_rationales(
        _summary(source=RationaleSource.DECLARED, revision=2, text="Dropped the jitter."),
        _summary(source=RationaleSource.RECONSTRUCTED, text="Jitter is there for latency."),
    )

    assert comparison.declared is not None
    assert comparison.declared.revision == 2
    assert "revision 2" in render_comparison(comparison)
