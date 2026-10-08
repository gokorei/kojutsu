"""Compare the two rationales for one change, and report honestly what that says.

A single implementation produces two rationales: a ``DECLARED`` one stated by the
agent that did the work, and a ``RECONSTRUCTED`` one inferred from the diff by a
reviewer. This module reads both and says what their relationship is.

**The comparison is not a corroboration and must not read as one.** Two rationales
from the same principal on the same model agreeing with each other is a
*restatement* — the expected outcome, carrying no information at all. Reporting it
as agreement would reintroduce exactly the confusion the provenance axis exists to
prevent, one layer up. :func:`compare_rationales` therefore names the principals
and their independence level on every comparison, and says plainly when agreement
is uninformative.

**The interesting case is divergence.** When a stated reason and an inferred one
disagree, the intent is not visible in the diff — which is the thing a future
reader most needs and can recover no other way. It is also the only place the
manufactured-consensus risk becomes measurable: ``docs/github-seam.md`` predicts
that an unattended loop built on agreeable answers "manufactures consensus", and
two rationales coinciding across principals on a change where they should differ
is the signal to look for.

**A bounded comparison must never read as a complete one.** A report that compared
two of five rationales and said "no divergences found" is worse than one that says
it compared two and names the three it did not reach, so
:data:`ComparisonOutcome.not_reached` carries the ones the budget excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kojutsu.models import Independence, RationaleSource, compute_independence


class ComparisonOutcome(StrEnum):
    """What a comparison of two rationales actually established.

    Closed on purpose, for the same reason ``AnomalyKind`` is: this is a field a
    reader triages on, and a field that can hold any string is one no query can
    rely on.
    """

    #: Both present and they disagree. The intent is not visible in the change.
    DIVERGENT = "divergent"
    #: Both present and consistent. Informative only across separate principals.
    CONCURRENT = "concurrent"
    #: Both present, same principal and model. Agreement carries no information.
    RESTATEMENT = "restatement"
    #: Only a declaration exists. Absence is reported, not treated as agreement.
    DECLARED_ONLY = "declared_only"
    #: Only a reconstruction exists.
    RECONSTRUCTED_ONLY = "reconstructed_only"
    #: Neither exists.
    NEITHER = "neither"


#: Phrases a comparison reports verbatim, so a caller cannot paraphrase its way
#: past the caveat. Each names the thing that is *not* being claimed.
OUTCOME_NOTES: dict[ComparisonOutcome, str] = {
    ComparisonOutcome.DIVERGENT: (
        "The stated reason and the inferred one disagree, so the intent is not visible "
        "in the change. This is the finding, not a defect in either record."
    ),
    ComparisonOutcome.CONCURRENT: (
        "Both rationales agree, across separate principals, so this is a genuine second "
        "opinion rather than a restatement."
    ),
    ComparisonOutcome.RESTATEMENT: (
        "Both rationales come from the same principal and model. Agreement between them "
        "carries NO information: this is a restatement, not corroboration."
    ),
    ComparisonOutcome.DECLARED_ONLY: (
        "Only a stated rationale exists for this change. No reviewer rationalised it, "
        "which is an absence of evidence and not agreement."
    ),
    ComparisonOutcome.RECONSTRUCTED_ONLY: (
        "Only an inferred rationale exists for this change. Nothing stated what the "
        "author intended, so this is a guess about intent and is labelled as one."
    ),
    ComparisonOutcome.NEITHER: ("No rationale of either kind exists for this change."),
}


@dataclass(frozen=True)
class RationaleSummary:
    """One stored rationale, as the comparison needs to see it."""

    entry_id: str
    source: RationaleSource
    declared_by: str
    model: str | None
    revision: int
    text: str


@dataclass(frozen=True)
class RationaleComparison:
    """The relationship between two rationales for one change.

    Carries both records rather than a verdict alone, because a reader holding
    only "divergent" has learned a conclusion and not the reasons. Holding both
    also keeps the two sources visible, which is the whole point: a merged view
    would let a reader be unable to tell what was stated from what was inferred.
    """

    outcome: ComparisonOutcome
    independence: Independence
    independence_reason: str
    note: str
    declared: RationaleSummary | None = None
    reconstructed: RationaleSummary | None = None
    #: Rationales the response budget did not reach, named rather than dropped.
    not_reached: tuple[str, ...] = ()

    @property
    def is_informative(self) -> bool:
        """False when the outcome is a restatement or a bare absence.

        A caller filtering on this is asking "did this teach me anything the
        records did not already say independently?", and a restatement is the
        answer no.
        """
        return self.outcome not in (ComparisonOutcome.RESTATEMENT, ComparisonOutcome.NEITHER)


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def compare_rationales(
    declared: RationaleSummary | None,
    reconstructed: RationaleSummary | None,
    *,
    not_reached: tuple[str, ...] = (),
) -> RationaleComparison:
    """Compare a stated rationale with an inferred one, or report what is missing.

    The independence level is computed with :func:`compute_independence` rather
    than a scale invented here. That is deliberate: a comparison that implied more
    separation than the provenance supports would be inventing a second, vaguer
    version of the axis the codebase already has, and two axes disagreeing about
    the same pair of records is worse than one axis being conservative.
    """
    if declared is None and reconstructed is None:
        return RationaleComparison(
            outcome=ComparisonOutcome.NEITHER,
            independence=Independence.SELF_CERTIFIED,
            independence_reason="no rationale of either kind exists",
            note=OUTCOME_NOTES[ComparisonOutcome.NEITHER],
            not_reached=not_reached,
        )
    if declared is None:
        return RationaleComparison(
            outcome=ComparisonOutcome.RECONSTRUCTED_ONLY,
            independence=Independence.SELF_CERTIFIED,
            independence_reason="only an inferred rationale exists",
            note=OUTCOME_NOTES[ComparisonOutcome.RECONSTRUCTED_ONLY],
            reconstructed=reconstructed,
            not_reached=not_reached,
        )
    if reconstructed is None:
        return RationaleComparison(
            outcome=ComparisonOutcome.DECLARED_ONLY,
            independence=Independence.SELF_CERTIFIED,
            independence_reason="only a stated rationale exists",
            note=OUTCOME_NOTES[ComparisonOutcome.DECLARED_ONLY],
            declared=declared,
            not_reached=not_reached,
        )

    independence, reason = compute_independence(
        asker_account=declared.declared_by,
        asker_model=declared.model,
        answerer_account=reconstructed.declared_by,
        answerer_model=reconstructed.model,
    )
    agrees = _normalise(declared.text) == _normalise(reconstructed.text)

    if independence is Independence.SELF_CERTIFIED:
        # Agreement is uninformative and disagreement is a restatement reaching the
        # same words; either way these are the same mind, so the outcome says so
        # rather than reporting a finding.
        outcome = ComparisonOutcome.RESTATEMENT
    elif agrees:
        outcome = ComparisonOutcome.CONCURRENT
    else:
        outcome = ComparisonOutcome.DIVERGENT

    return RationaleComparison(
        outcome=outcome,
        independence=independence,
        independence_reason=reason,
        note=OUTCOME_NOTES[outcome],
        declared=declared,
        reconstructed=reconstructed,
        not_reached=not_reached,
    )


def render_comparison(comparison: RationaleComparison) -> str:
    """Render one comparison for a reader, caveats first.

    The independence level and the note come before either text, so a reader meets
    "these are the same account" before meeting prose that reads as two agreeing
    opinions. Presenting the texts first is how a restatement ends up looking like
    a second opinion.
    """
    lines = [
        f"Outcome: {comparison.outcome.value}",
        f"Independence: {comparison.independence.value} ({comparison.independence_reason})",
        comparison.note,
    ]
    if comparison.declared is not None:
        lines.append(
            f"Stated by {comparison.declared.declared_by} "
            f"({comparison.declared.model or 'model not stated'}), "
            f"revision {comparison.declared.revision}: {comparison.declared.text}"
        )
    if comparison.reconstructed is not None:
        lines.append(
            f"Inferred by {comparison.reconstructed.declared_by} "
            f"({comparison.reconstructed.model or 'model not stated'}): "
            f"{comparison.reconstructed.text}"
        )
    if comparison.not_reached:
        lines.append(
            "NOT COMPARED, and therefore not reported as agreeing: "
            + ", ".join(comparison.not_reached)
        )
    return "\n".join(lines)
